"""Turning a column into group ordinals, by whichever route is cheapest.

This is the operation a group by is built on. Every row gets an integer naming
the group it belongs to, the distinct keys come back so the ordinal can be turned
back into a value, and everything downstream aggregates into a flat array indexed
by that integer. pandas calls this `factorize` and the name is worth keeping.

There are two routes and the interesting one is that the fast route is not the
hash table.

**Direct indexing.** A column of integers whose values span a small range does
not need to be hashed at all. Allocate one slot per possible value, index it,
done. That is a group by on a year, on a category code, on a small identifier, on
anything that came out of a previous factorize, and it is a large fraction of the
group bys anyone actually runs. Both passes over the column are sequential.

**The hash table.** Everything else. Floats, wide-ranging integers, anything
where a direct table would be larger than the column it is replacing.

The switch is `DIRECT_LIMIT` and it is a memory bound rather than a guess about
cardinality: the direct table costs four bytes per possible value, so it is worth
taking exactly when the range is small enough that the table behaves like cache.

Text has a third route, `factorize_strings`, and it is the hash table with the
key comparison put back, because a string does not fit in the 64 bits the table
stores. It hands back a representative row per group rather than the keys
themselves, which is the one place its shape differs from the numeric one and is
explained on the function.

Nulls get a group. A null is a value that rows can share, pandas with
`dropna=False` gives it a group, and dropping it later is a filter on one ordinal
while inventing it later would be another pass over the column. When a column has
any nulls the null group is ordinal zero and the non-null keys follow in
first-appearance order. Fixing it at zero rather than at the position of the
first null is what keeps the ordinals the table hands out and the positions in
`keys` a constant apart, which is the difference between one addition per row and
a second bookkeeping structure.
"""

from std.sys.info import simd_width_of, size_of

from firepanda.array.array import Array, from_list
from firepanda.array.strings import StringArray
from firepanda.buffer.buffer import Buffer
from firepanda.kernel.accum import highest, lowest
from firepanda.kernel.agg import BLOCK

from .function import DEFAULT_SEED, hash_chunk, hash_strings_chunk
from .table import HashTable

comptime DIRECT_LIMIT = 1 << 16
"""Largest integer range that skips the hash table.

65536 slots is 256 KB of uint32, which is the last size that behaves like cache
rather than like memory on the machines this targets. Above it the direct table
stops being a lookup and starts being the same random access the hash table was
going to do anyway, with worse density.
"""

comptime SAFE_RANGE_BITS = 40
"""Values outside plus or minus 2^40 do not take the direct route.

Not because they could not, but because computing `high - low` for them can
overflow, and a range check that overflows is worse than a range check that
declines. Anything with a span small enough to qualify is inside this window
anyway; the check is about the arithmetic, not about the data.
"""

comptime CHUNK_ROWS = 1 << 10
"""Rows hashed before any of them are probed.

A thousand rows is eight kilobytes, which stays in L1 alongside the table itself
on a low cardinality column. Four thousand measured slower on every shape here,
which is what you would expect from a buffer that no longer fits. Smaller chunks
start losing the last `PROBE_LOOKAHEAD` rows of each one to a prefetch that has
nowhere to run.
"""


struct Factorized[dt: DType](Movable):
    """A column rewritten as group ordinals, plus the keys they name."""

    var codes: Array[DType.uint32]
    """One group ordinal per row of the input, in `[0, len(keys))`."""

    var keys: Array[Self.dt]
    """The distinct keys. Index it by an ordinal from `codes` to get the value
    back. Entry zero is the null group when there is one."""

    var null_group: Int
    """The ordinal the nulls got, which is zero, or -1 if the column had none."""

    def __init__(
        out self,
        var codes: Array[DType.uint32],
        var keys: Array[Self.dt],
        null_group: Int,
    ):
        """Constructs a result.

        Args:
            codes: The per-row ordinals.
            keys: The distinct keys.
            null_group: The ordinal for nulls, or -1.
        """
        self.codes = codes^
        self.keys = keys^
        self.null_group = null_group

    def count(self) -> Int:
        """Returns the number of groups.

        Returns:
            The group count, including the null group if there is one.
        """
        return len(self.keys)

    def into_codes(deinit self) -> Array[DType.uint32]:
        """Gives up the ordinals without copying them, dropping the keys.

        A caller that groups on several columns wants the ordinals and has no use
        for the keys, and Mojo will not let it reach in and move `codes` out from
        under a struct that still owns `keys`. `deinit` says the rest is being
        torn down, which is what makes the move legal. Same arrangement as
        `Series.into_values`.

        Returns:
            The per-row ordinals.
        """
        return self.codes^


def factorize[
    dt: DType
](col: Array[dt], seed: UInt64 = DEFAULT_SEED) -> Factorized[dt]:
    """Assigns a group ordinal to every row.

    Args:
        col: The column to group.
        seed: The per-query hash seed. Ignored on the direct route, which does
            not hash.

    Parameters:
        dt: The column's dtype.

    Returns:
        The ordinals, the distinct keys, and which ordinal the nulls got.
    """
    comptime if dt.is_integral():
        var plan = _direct_plan[dt](col)
        if plan.span >= 0:
            return _factorize_direct[dt](col, plan.span, plan.base)
    return _factorize_hashed[dt](col, seed)


@fieldwise_init
struct DirectPlan[dt: DType](Copyable, Movable):
    """Whether the direct route applies to a column, and what it needs."""

    var span: Int
    """The number of slots a direct table needs, or -1 for the hash route."""

    var base: Scalar[Self.dt]
    """The value that indexes slot zero, which is the column's minimum."""


def _direct_plan[dt: DType](col: Array[dt]) -> DirectPlan[dt]:
    """Decides whether the direct route applies, in one pass that can stop early.

    This used to be a `min_of` and a `max_of` and then a test on the two answers,
    which is two full passes over the column before the decision. Both of them
    are wasted on any column that ends up hashed, and on wide-ranging data most
    of the first one is wasted too, because the span passes the point of no
    return in the first few thousand rows and nothing later can bring it back.

    So the two reductions are fused and the test is moved inside them. The bounds
    only ever widen, so once the span is too large for a direct table, or once a
    value is outside the window the subtraction is safe in, no later row can
    change the answer and the scan returns. On a ten million row column of
    scattered keys that is eight milliseconds down to under one.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype, which the caller has already established is
            integral.

    Returns:
        The plan, whose span is -1 when the column should be hashed instead.
    """
    var n = len(col)
    if n == 0:
        return DirectPlan[dt](-1, Scalar[dt](0))

    comptime width = simd_width_of[dt]()
    comptime safe = Scalar[dt](1 << SAFE_RANGE_BITS)
    var ceiling = DIRECT_LIMIT if DIRECT_LIMIT < n else n

    var ptr = col.unsafe_ptr()
    var low = highest[dt]()
    var high = lowest[dt]()
    var seen = False

    for w in range(col.data.validity.word_count()):
        var word = col.data.validity.unsafe_word(w)
        if word == 0:
            continue

        var start = w * BLOCK
        var last = start + BLOCK
        if last > n:
            last = n

        if word == UInt64.MAX and last == start + BLOCK:
            seen = True
            var mins = SIMD[dt, width](highest[dt]())
            var maxes = SIMD[dt, width](lowest[dt]())
            var i = start
            while i + width <= last:
                var chunk = ptr.unsafe_offset(i).unsafe_load[width=width]()
                mins = min(mins, chunk)
                maxes = max(maxes, chunk)
                i += width
            var block_low = mins.reduce_min()
            var block_high = maxes.reduce_max()
            while i < last:
                var value = ptr.unsafe_offset(i).unsafe_load()
                if value < block_low:
                    block_low = value
                if value > block_high:
                    block_high = value
                i += 1
            if block_low < low:
                low = block_low
            if block_high > high:
                high = block_high
        else:
            for i in range(start, last):
                if (word >> UInt64(i - start)) & 1 == 0:
                    continue
                seen = True
                var value = ptr.unsafe_offset(i).unsafe_load()
                if value < low:
                    low = value
                if value > high:
                    high = value

        if seen:
            comptime if size_of[dt]() > 4:
                if high > safe:
                    return DirectPlan[dt](-1, Scalar[dt](0))
                comptime if dt.is_signed():
                    if low < -safe:
                        return DirectPlan[dt](-1, Scalar[dt](0))
            if Int(high) - Int(low) + 1 > ceiling:
                return DirectPlan[dt](-1, Scalar[dt](0))

    if not seen:
        # Every row is null. One group, no keys to spread out, and the direct
        # route allocates a single slot it never reads.
        return DirectPlan[dt](1, Scalar[dt](0))

    return DirectPlan[dt](Int(high) - Int(low) + 1, low)


def _factorize_direct[
    dt: DType
](col: Array[dt], span: Int, base: Scalar[dt]) -> Factorized[dt]:
    """Factorizes by indexing a table with the value itself.

    Args:
        col: The column.
        span: The table width, from `_direct_plan`.
        base: The value that indexes slot zero, which is the column's minimum.

    Parameters:
        dt: The column's dtype.

    Returns:
        The factorization.
    """
    var n = len(col)
    var values = col.unsafe_ptr()

    # Zero means unseen, so the ordinal is stored plus one and the table needs no
    # initialization pass. `Buffer` already handed back zeroed memory.
    var seen = Buffer(span * 4)
    var slots = seen.bitcast[DType.uint32]()

    var has_null = col.null_count() > 0
    var codes = Array[DType.uint32](n)
    var out = codes.unsafe_ptr()
    var keys = List[Scalar[dt]]()
    var null_group = -1
    if has_null:
        null_group = 0
        keys.append(Scalar[dt](0))

    for i in range(n):
        if has_null and not col.is_valid(i):
            out.unsafe_offset(i).unsafe_write(UInt32(0))
            continue

        var value = values.unsafe_offset(i).unsafe_load()
        var at = Int(value) - Int(base)
        var stored = slots.unsafe_offset(at).unsafe_load()
        if stored == 0:
            var assigned = len(keys)
            keys.append(value)
            slots.unsafe_offset(at).unsafe_write(UInt32(assigned + 1))
            out.unsafe_offset(i).unsafe_write(UInt32(assigned))
        else:
            out.unsafe_offset(i).unsafe_write(stored - 1)

    return _finish[dt](codes^, keys^, null_group)


def _factorize_hashed[
    dt: DType
](col: Array[dt], seed: UInt64) -> Factorized[dt]:
    """Factorizes through the hash table.

    Args:
        col: The column.
        seed: The per-query hash seed.

    Parameters:
        dt: The column's dtype.

    Returns:
        The factorization.
    """
    var n = len(col)
    var values = col.unsafe_ptr()

    var has_null = col.null_count() > 0
    var offset = 1 if has_null else 0
    var codes = Array[DType.uint32](n)
    var keys = List[Scalar[dt]]()
    var null_group = -1
    if has_null:
        null_group = 0
        keys.append(Scalar[dt](0))

    # Hash a chunk, probe that chunk, move on. Hashing ahead of the probe is what
    # makes the prefetch possible, because a loop that hashed and probed one row
    # at a time would have nothing to run ahead of; the address it wants to fetch
    # is not known until the row it is already working on has been hashed. But
    # hashing the whole column ahead of the probe means writing eight megabytes
    # to memory and reading them back, which buys nothing. A chunk gets the
    # lookahead and keeps the buffer in cache.
    var hashes = Buffer(CHUNK_ROWS * 8)

    # The table writes the ordinals and hands back the row that first produced
    # each one. Reading the key values out of those rows afterwards is a pass
    # over the group count rather than over the row count, and it keeps the table
    # free of any idea of what a value is.
    var firsts = List[Int]()
    var table = HashTable(0, seed)

    var base = 0
    while base < n:
        var count = min(CHUNK_ROWS, n - base)
        hash_chunk(col, base, count, seed, hashes)
        table.build(
            hashes,
            col.data.validity,
            has_null,
            base,
            count,
            n,
            offset,
            codes,
            firsts,
        )
        base += count

    for at in range(len(firsts)):
        keys.append(values.unsafe_offset(firsts[at]).unsafe_load())

    return _finish[dt](codes^, keys^, null_group)


def _finish[
    dt: DType
](
    var codes: Array[DType.uint32], var keys: List[Scalar[dt]], null_group: Int
) -> Factorized[dt]:
    """Packages the two routes' common output.

    Args:
        codes: The per-row ordinals.
        keys: The distinct keys.
        null_group: The ordinal for nulls, or -1.

    Parameters:
        dt: The column's dtype.

    Returns:
        The factorization, with the null group's key marked null.
    """
    var out = from_list[dt](keys)
    if null_group >= 0:
        out.set_null(null_group)
    return Factorized[dt](codes^, out^, null_group)


struct FactorizedStrings(Movable):
    """A string column rewritten as group ordinals, plus where to find the keys.
    """

    var codes: Array[DType.uint32]
    """One group ordinal per row of the input, in `[0, groups)`."""

    var firsts: List[Int]
    """For each non-null group, the first row that had it, in ordinal order.

    This is where it differs from `Factorized`, which hands back the key values
    themselves. A string key is not a scalar, so returning the keys would mean
    gathering them into a second column, and both callers there are, the frame
    layer's group by and the join, already recover key values by taking rows.
    Building a column they were going to throw away is a pass over the group
    count that nobody asked for.
    """

    var null_group: Int
    """The ordinal the nulls got, which is zero, or -1 if the column had none."""

    def __init__(
        out self,
        var codes: Array[DType.uint32],
        var firsts: List[Int],
        null_group: Int,
    ):
        """Constructs a result.

        Args:
            codes: The per-row ordinals.
            firsts: A representative row per non-null group.
            null_group: The ordinal for nulls, or -1.
        """
        self.codes = codes^
        self.firsts = firsts^
        self.null_group = null_group

    def count(self) -> Int:
        """Returns the number of groups.

        Returns:
            The group count, including the null group if there is one.
        """
        return len(self.firsts) + (1 if self.null_group >= 0 else 0)

    def into_codes(deinit self) -> Array[DType.uint32]:
        """Gives up the ordinals without copying them, dropping the rest.

        Returns:
            The per-row ordinals.
        """
        return self.codes^


def factorize_strings(
    col: StringArray, seed: UInt64 = DEFAULT_SEED
) -> FactorizedStrings:
    """Assigns a group ordinal to every element of a string column.

    There is no direct route here. The direct route exists because an integer
    can index a table by being itself, and a string cannot, so everything goes
    through the hash table.

    Args:
        col: The column to group.
        seed: The per-query hash seed.

    Returns:
        The ordinals, a representative row per group, and which ordinal the
        nulls got.
    """
    var n = len(col)
    var has_null = col.null_count() > 0
    var offset = 1 if has_null else 0
    var codes = Array[DType.uint32](n)
    var null_group = -1
    if has_null:
        null_group = 0

    # Same chunking as the numeric path and for the same reason, except that the
    # hashes cost more to produce here, so there is more to hide behind the
    # prefetch and the chunk earns more than it does there.
    var hashes = Buffer(CHUNK_ROWS * 8)
    var firsts = List[Int]()
    var table = HashTable(0, seed)

    var base = 0
    while base < n:
        var count = min(CHUNK_ROWS, n - base)
        hash_strings_chunk(col, base, count, seed, hashes)
        table.build_strings(
            hashes, col, has_null, base, count, n, offset, codes, firsts
        )
        base += count

    return FactorizedStrings(codes^, firsts^, null_group)
