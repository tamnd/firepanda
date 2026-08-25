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

Nulls get a group. A null is a value that rows can share, pandas with
`dropna=False` gives it a group, and dropping it later is a filter on one ordinal
while inventing it later would be another pass over the column. When a column has
any nulls the null group is ordinal zero and the non-null keys follow in
first-appearance order. Fixing it at zero rather than at the position of the
first null is what keeps the ordinals the table hands out and the positions in
`keys` a constant apart, which is the difference between one addition per row and
a second bookkeeping structure.
"""

from std.sys.info import size_of

from firepanda.array.array import Array, from_list
from firepanda.buffer.buffer import Buffer
from firepanda.kernel.agg import AggResult, max_of, min_of

from .function import DEFAULT_SEED, hash_chunk
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
        # Both bounds are wanted, and a reduction over the column is the cheapest
        # pass this operation has. Working them out here rather than inside the
        # test keeps it to one scan per bound rather than three.
        var low = min_of(col)
        var high = max_of(col)
        var span = _direct_span[dt](len(col), low, high)
        if span >= 0:
            var base = low.value if low.valid else Scalar[dt](0)
            return _factorize_direct[dt](col, span, base)
    return _factorize_hashed[dt](col, seed)


def _direct_span[
    dt: DType
](n: Int, low: AggResult[dt], high: AggResult[dt]) -> Int:
    """Decides whether the direct route applies, and how wide its table is.

    Args:
        n: The column length.
        low: The smallest non-null value.
        high: The largest non-null value.

    Parameters:
        dt: The column's dtype, which the caller has already established is
            integral.

    Returns:
        The number of slots a direct table needs, or -1 if the column should go
        through the hash table instead.
    """
    if n == 0:
        return -1

    if not high.valid:
        # Every row is null. One group, no keys to spread out, and the direct
        # route allocates a single slot it never reads.
        return 1

    comptime if size_of[dt]() > 4:
        comptime limit = Scalar[dt](1 << SAFE_RANGE_BITS)
        if high.value > limit:
            return -1
        comptime if dt.is_signed():
            if low.value < -limit:
                return -1

    var span = Int(high.value) - Int(low.value) + 1
    if span > DIRECT_LIMIT or span > n:
        return -1
    return span


def _factorize_direct[
    dt: DType
](col: Array[dt], span: Int, base: Scalar[dt]) -> Factorized[dt]:
    """Factorizes by indexing a table with the value itself.

    Args:
        col: The column.
        span: The table width, from `_direct_span`.
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
