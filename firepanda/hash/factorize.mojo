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

The hash route runs on every core when there is enough of a column to be worth
splitting and its keys repeat often enough for the split to pay. Each worker
builds its own table over a contiguous slice, and a sequential merge afterwards
renumbers what they found into one set of ordinals. That merge is as long as the
groups the workers found between them, so a column where every row is its own key
gets no benefit and stays on one thread; `_projected_groups` is the check that
tells the two apart before either has run, and `_factorize_hashed_parallel` is
where the argument for the shape lives.

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
from firepanda.exec import parallel_for, worker_count
from firepanda.kernel.accum import highest, lowest
from firepanda.kernel.agg import BLOCK

from .function import DEFAULT_SEED, hash_chunk, hash_strings_chunk
from .table import HashTable, project_groups

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

comptime PARALLEL_ROWS = 1 << 17
"""Rows below which the hash route stays on one thread.

The parallel route costs a merge over every group every worker found and a
second pass over the column to renumber them, and neither of those exists on the
serial one. A hundred and thirty thousand rows is where the threads start paying
for that on the machines this targets. Below it the whole column is a few
milliseconds anyway.
"""

comptime PARALLEL_MIN_SLICE = 1 << 15
"""Smallest slice a worker is given.

Two workers on a column that fits in L2 spend more time in the merge than they
save in the build. Capping the worker count by this rather than dropping to one
thread means a column just over `PARALLEL_ROWS` gets the few workers it can keep
busy instead of all of them.
"""

comptime MERGE_HEADROOM = 2
"""How many times a worker's slice has to outnumber the groups projected in it.

The merge after a parallel build is the one part of it that is sequential, and
its length is the number of groups the workers found between them. On a column
whose groups are far fewer than a slice is long that is a rounding error. On a
column where every row is its own key it is the whole column over again, and the
parallel route is then slower than the serial one it replaced, which was
measured at ten million rows before this check existed.

Requiring a slice to be twice the projected group count puts the merge at a
third of the column at the boundary, and at a half if the projection is off by a
factor of two, which is more than it usually is. Both of those are still ahead of
a serial build that walks all of it.
"""

comptime PARALLEL_SAMPLE = 1 << 16
"""Most rows the group count is projected from.

The projection is `project_groups` reading a discovery rate, which is what the
table already sizes itself with, and sixty five thousand rows is where that
function is asked to believe what it sees. On a shorter column the sample is a
sixteenth of it instead, so that the cost of a decision stays proportional to the
work the decision is about.
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
](col: Array[dt], seed: UInt64 = DEFAULT_SEED) raises -> Factorized[dt]:
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
](col: Array[dt], seed: UInt64) raises -> Factorized[dt]:
    """Factorizes through the hash table, on one thread or on all of them.

    Args:
        col: The column.
        seed: The per-query hash seed.

    Parameters:
        dt: The column's dtype.

    Returns:
        The factorization.
    """
    var n = len(col)
    if n >= PARALLEL_ROWS:
        var workers = min(worker_count(), n // PARALLEL_MIN_SLICE)
        if workers > 1:
            var groups = _projected_groups[dt](col, seed, n)
            if groups * MERGE_HEADROOM <= n // workers:
                return _factorize_hashed_parallel[dt](col, seed, workers)
    return _factorize_hashed_serial[dt](col, seed)


def _projected_groups[
    dt: DType
](col: Array[dt], seed: UInt64, n: Int) -> Int:
    """Guesses how many groups a column has, by building the front of it.

    The parallel route is worth taking when the groups are few and worth
    refusing when nearly every row is one, and nothing knows which of those a
    column is until it has been built. So this builds a sample of it and reads
    the discovery rate the same way the table sizes itself, through
    `project_groups`.

    The sample is thrown away rather than handed to whichever route runs next.
    It is a sixteenth of the column at most, and keeping it would mean both
    routes had to be able to start from a table somebody else had already put
    rows into, which is a lot of coupling to buy back that sixteenth.

    Args:
        col: The column.
        seed: The per-query hash seed.
        n: The column's length.

    Parameters:
        dt: The column's dtype.

    Returns:
        A guess at the number of distinct keys in the whole column.
    """
    var sample = min(PARALLEL_SAMPLE, n // 16)
    var has_null = col.null_count() > 0
    var codes = Array[DType.uint32](sample)
    var firsts = List[Int]()
    var hashes = Buffer(CHUNK_ROWS * 8)
    var table = HashTable(0, seed)

    var half = 0
    var measured = False
    var base = 0
    while base < sample:
        var count = min(CHUNK_ROWS, sample - base)
        hash_chunk(col, base, count, seed, hashes)
        table.build(
            hashes,
            col.data.validity,
            has_null,
            base,
            base,
            count,
            sample,
            0,
            codes,
            firsts,
        )
        base += count
        if not measured and base * 2 >= sample:
            half = len(table)
            measured = True

    return project_groups(len(table), half, sample, n)


def _factorize_hashed_serial[
    dt: DType
](col: Array[dt], seed: UInt64) -> Factorized[dt]:
    """Factorizes through one hash table on one thread.

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


def _factorize_hashed_parallel[
    dt: DType
](col: Array[dt], seed: UInt64, workers: Int) raises -> Factorized[dt]:
    """Factorizes through one hash table per core, then merges what they found.

    The column is cut into one contiguous slice per worker and each worker runs
    the serial build over its own slice into its own table, numbering its groups
    from zero. Nothing is shared, so nothing is locked, and the only writes that
    land in the same array are the ordinals, which are indexed by row and so
    never touch the same word twice.

    That leaves every worker with a private numbering, and a merge to turn those
    into one. The merge walks the workers in order and, within a worker, its
    groups in the order it found them, inserting each into a single table that
    hands out the final ordinals. That order is exactly first-appearance order
    over the whole column: the slices are contiguous and in order, and a worker
    numbers its groups by when it first saw them, so the first time any key turns
    up anywhere is the first time the merge is offered it. The ordinals come out
    identical to the serial route's, which is what makes this a substitution
    rather than a second set of semantics.

    The merge is the part that does not parallelize, and its cost is the number
    of groups the workers found between them rather than the number of rows. On
    the columns a group by is usually run on, where the groups are thousands and
    the rows are millions, it disappears. On a column where every row is its own
    key it is the whole operation again, and this route breaks even there rather
    than winning. It probes with the hashes the workers already computed, which
    `HashTable.keys_by_ordinal` reads back out of their slots, so at least it
    does not hash anything twice.

    Args:
        col: The column.
        seed: The per-query hash seed.
        workers: How many slices to cut the column into. At least two.

    Parameters:
        dt: The column's dtype.

    Returns:
        The factorization.
    """
    var n = len(col)
    var has_null = col.null_count() > 0
    var offset = 1 if has_null else 0
    var codes = Array[DType.uint32](n)

    # Slices land on chunk boundaries so that every worker's inner loop is the
    # same shape as the serial one's, with a short chunk only at the very end.
    var chunks = (n + CHUNK_ROWS - 1) // CHUNK_ROWS
    var bounds = List[Int](capacity=workers + 1)
    for w in range(workers):
        bounds.append(chunks * w // workers * CHUNK_ROWS)
    bounds.append(n)

    var tables = List[HashTable](capacity=workers)
    var founds = List[List[Int]](capacity=workers)
    for _ in range(workers):
        tables.append(HashTable(0, seed))
        founds.append(List[Int]())

    def one(w: Int) raises {mut tables, mut founds, mut codes, imm}:
        var start = bounds[w]
        var stop = bounds[w + 1]
        var hashes = Buffer(CHUNK_ROWS * 8)
        var base = start
        while base < stop:
            var count = min(CHUNK_ROWS, stop - base)
            hash_chunk(col, base, count, seed, hashes)
            tables[w].build(
                hashes,
                col.data.validity,
                has_null,
                base,
                base - start,
                count,
                stop - start,
                0,
                codes,
                founds[w],
            )
            base += count

    parallel_for(one, workers)

    # Where each worker's block of local ordinals starts once they are laid end
    # to end. This is the number the remap adds to a local ordinal to find its
    # entry in the map.
    var starts = List[Int](capacity=workers + 1)
    var total = 0
    for w in range(workers):
        starts.append(total)
        total += len(tables[w])
    starts.append(total)

    var found = Buffer(total * 8)

    def dump(w: Int) raises {mut found, imm}:
        tables[w].keys_by_ordinal(found, starts[w])

    parallel_for(dump, workers)

    var keys = List[Scalar[dt]]()
    var null_group = -1
    if has_null:
        null_group = 0
        keys.append(Scalar[dt](0))

    var values = col.unsafe_ptr()
    var hashed = found.bitcast[DType.uint64]()
    var map = Buffer(overwritten=total * 4)
    var mapped = map.bitcast[DType.uint32]()

    # Sized for every local group rather than for the distinct ones, because the
    # distinct ones are not known until this loop has run. It over-reserves by at
    # most the worker count, and only on a column whose groups are few enough
    # that the table is small either way.
    var merged = HashTable(total, seed)
    for w in range(workers):
        var at = starts[w]
        for k in range(len(tables[w])):
            var ordinal = merged.insert(
                hashed.unsafe_offset(at + k).unsafe_load()
            )
            mapped.unsafe_offset(at + k).unsafe_write(UInt32(ordinal + offset))
            if ordinal + offset == len(keys):
                keys.append(values.unsafe_offset(founds[w][k]).unsafe_load())

    def remap(w: Int) raises {mut codes, imm}:
        var out = codes.unsafe_ptr()
        var at = starts[w]
        for i in range(bounds[w], bounds[w + 1]):
            if has_null and not col.is_valid(i):
                out.unsafe_offset(i).unsafe_write(UInt32(0))
                continue
            var local = Int(out.unsafe_offset(i).unsafe_load())
            out.unsafe_offset(i).unsafe_write(
                mapped.unsafe_offset(at + local).unsafe_load()
            )

    parallel_for(remap, workers)

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
            hashes, col, has_null, base, base, count, n, offset, codes, firsts
        )
        base += count

    return FactorizedStrings(codes^, firsts^, null_group)
