"""Giving both sides of a join one number per row that means the same thing.

Matching rows means comparing key tuples across two frames, and a code from
`factorize` on one side is a number in that column's own space and means nothing
on the other. So something has to put the two sides in one space, and this file
is the two ways of doing it.

## The way that always works

Concatenate each key column with its opposite number, hand the whole set to
`group_ordinals`, and slice the codes back apart. Two rows share a code exactly
when they share a key tuple, whichever side they came from, and the multi-key
packing, the densifying, the string route and the small integer fast path all
come along for free because they are already in there.

The bill is a copy of both key columns and then a factorize of the copy. That is
one pass to write the copy, one to read it, and a build of a table holding every
distinct key on both sides. It is correct for every dtype and every key count and
it is what runs when the other route declines.

## The way a join actually wants

A join does not need both sides in one table. It needs the smaller side's keys in
a table and one question asked of it per row of the larger side. Nothing is
copied, nothing is built over the large side, and the large side's pass is a
lookup rather than an insert, which means it needs no growth check, no sizing
schedule and no writes, so it runs on every core at once.

On a join of ten million rows against a table of ten thousand, the first way
builds a dictionary over ten million and ten thousand keys and copies eighty
megabytes to do it. The second builds a dictionary over ten thousand keys, which
fits in L2, and reads the large side once. That is the difference between the
join being a factorize with a join stapled to it and the join being a probe.

This route is taken for a single key column of a fixed width dtype. One key
because two would need the packing that `group_ordinals` does, and fixed width
because the table compares hashes and a string needs its bytes compared on a hash
match against the other side's column, which the table has no way to reach.

## Which side is built

The smaller one, counted in rows. Not the right one, which is what the argument
order would say, and not the one the user wrote second.

Either side works because the ordinals only have to agree, not to mean anything
in particular. Build on the left and the right gets codes in the left's space;
build on the right and the left gets codes in the right's. What the caller does
with them, which is bucket one side by code and walk the other, is the same
either way.

## A row that matches nothing

The probe side gets one ordinal past the last real one, and the caller sizes its
tables to include it. Then a row that matched nothing carries a group that
nothing was ever put into, its bucket is empty, and the emit reads it the same
way it reads every other row rather than testing for it. The build side never
carries that ordinal, so if it is the left, the right's misses land in a bucket
no left row ever looks at.

Null keys are the same shape. A row whose key is null matches nothing, including
another null, which is what SQL and Polars do. Nulls on the probe side get the
miss ordinal for free. Nulls on the build side keep whatever code the loop
happened to write, because every read of a build side code is already guarded by
the caller's null list, and giving them a code nobody reads is cheaper than
branching to give them a better one.
"""

from firepanda.array.any import AnyArray, ColumnRefs, borrow_columns
from firepanda.array.array import Array
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import ALL
from firepanda.exec import parallel_morsels
from firepanda.hash.factorize import CHUNK_ROWS, DIRECT_LIMIT, direct_plan
from firepanda.hash.function import DEFAULT_SEED, hash_chunk
from firepanda.hash.grouping import group_ordinals
from firepanda.hash.table import HashTable
from firepanda.kernel.concat import concat_two_any


comptime PARALLEL_PROBE_ROWS = 1 << 17
"""Below this many probe rows the probe stays on one thread.

Same constant as `pairs.mojo`'s `PARALLEL_LEFT_ROWS` and picked the same way. A
probe is a handful of nanoseconds a row and a fork and join is tens of
microseconds, so the split has to have that many rows to pay for itself before it
is offered.
"""

comptime PROBE_MORSEL_ROWS = 1 << 16
"""Probe rows a worker takes at a time.

What a probe row costs is whether its slot is in cache, and on a table larger
than the cache that is not the same for every row: a key that half the rows share
is a hit in L1 and a key nothing else has is a trip to memory. So the pieces are
morsels rather than one slice per worker.

Larger than `pairs.mojo`'s emit morsel because a probe morsel also holds a hash
buffer and a chunk loop, and cutting it finer means paying that setup more often
for an imbalance that is smaller here than it is in the emit.
"""


struct KeyAlignment(Movable):
    """One ordinal per row of both sides, in an order they agree on."""

    var codes: Array[DType.uint32]
    """The ordinal of every row, the left side's rows first and then the right's.

    Two rows hold the same ordinal exactly when they hold the same key tuple. A
    row that matched nothing on the other side holds an ordinal that no row of
    the other side holds.
    """

    var groups: Int
    """How many ordinals there are, counting the one for rows that match nothing.

    Size a table indexed by ordinal to this and every ordinal in `codes` indexes
    it.
    """

    var absent: List[Bool]
    """Whether each row's key tuple contains a null, or empty if none does.

    Indexed the same way `codes` is. Guarded by `has_nulls` at every read, so the
    empty list is never indexed.
    """

    var has_nulls: Bool
    """Whether any key column on either side has a null."""

    def __init__(
        out self,
        var codes: Array[DType.uint32],
        groups: Int,
        var absent: List[Bool],
        has_nulls: Bool,
    ):
        """Constructs an alignment.

        Args:
            codes: The per-row ordinals.
            groups: How many there are.
            absent: The null key flags, or an empty list.
            has_nulls: Whether `absent` was filled.
        """
        self.codes = codes^
        self.groups = groups
        self.absent = absent^
        self.has_nulls = has_nulls


def align_keys[
    l: ImmOrigin, r: ImmOrigin
](
    left_columns: ColumnRefs[l],
    left_keys: List[Int],
    left_rows: Int,
    right_columns: ColumnRefs[r],
    right_keys: List[Int],
    right_rows: Int,
) raises -> KeyAlignment:
    """Puts both sides' key tuples in one ordinal space.

    Args:
        left_columns: The left frame's columns.
        left_keys: Which of them are keys, in key order.
        left_rows: How many rows the left frame has.
        right_columns: The right frame's columns.
        right_keys: Which of them are keys, in the matching order.
        right_rows: How many rows the right frame has.

    Returns:
        The ordinals, how many there are, and which rows have a null key.

    Raises:
        Error: If two matched key columns have different dtypes, or if a dtype
            has no physical layout.
    """
    for k in range(len(left_keys)):
        ref a = left_columns[left_keys[k]][]
        ref b = right_columns[right_keys[k]][]
        if a.dtype() != b.dtype() or a.is_string() != b.is_string():
            raise Error(
                "join: key columns must have the same dtype; got "
                + String(a.dtype())
                + " and "
                + String(b.dtype())
            )

    var absent = _null_keys(
        left_columns,
        left_keys,
        left_rows,
        right_columns,
        right_keys,
        right_rows,
    )
    var has_nulls = len(absent) > 0

    if len(left_keys) == 1 and left_rows > 0 and right_rows > 0:
        ref left = left_columns[left_keys[0]][]
        ref right = right_columns[right_keys[0]][]
        # Before the dispatch, because uint8 is in ALL and a string column would
        # match it and align on the first byte of each view. `group_ordinals`
        # guards the same way for the same reason.
        if not left.is_string():
            comptime for candidate in ALL:
                if left.dtype() == candidate:
                    var codes = Array[DType.uint32](left_rows + right_rows)
                    # Views and not copies. A join key column is as large as the
                    # side it belongs to, and copying both of them before
                    # looking at either was the same waste a group by used to
                    # pay on every key.
                    ref left_view = left.as_typed_view[candidate]()
                    ref right_view = right.as_typed_view[candidate]()
                    var groups = _probe_route[candidate](
                        left_view,
                        right_view,
                        left_rows,
                        codes,
                    )
                    return KeyAlignment(codes^, groups, absent^, has_nulls)

    var merged = List[AnyArray](capacity=len(left_keys))
    for k in range(len(left_keys)):
        merged.append(
            concat_two_any(
                left_columns[left_keys[k]][],
                right_columns[right_keys[k]][],
            )
        )
    var at = List[Int](capacity=len(merged))
    for k in range(len(merged)):
        at.append(k)

    var grouping = group_ordinals(
        borrow_columns(merged), at, left_rows + right_rows
    )
    var groups = grouping.groups
    return KeyAlignment(grouping^.into_codes(), groups, absent^, has_nulls)


def _null_keys[
    l: ImmOrigin, r: ImmOrigin
](
    left_columns: ColumnRefs[l],
    left_keys: List[Int],
    left_rows: Int,
    right_columns: ColumnRefs[r],
    right_keys: List[Int],
    right_rows: Int,
) raises -> List[Bool]:
    """Flags every row whose key tuple contains a null, on both sides at once.

    The null counts are asked first, because a key column with no nulls is the
    ordinary case and the loop below is a pass over both frames with a branch per
    key per row that answers False every time. Asking costs a popcount per
    validity word, which is a sixty fourth of the pass it decides against.

    Args:
        left_columns: The left frame's columns.
        left_keys: Which of them are keys.
        left_rows: How many rows the left frame has.
        right_columns: The right frame's columns.
        right_keys: Which of them are keys.
        right_rows: How many rows the right frame has.

    Returns:
        One flag per row, the left side's first, or an empty list when no key
        column on either side has a null.

    Raises:
        Error: If reading a validity bitmap fails.
    """
    var any = False
    for k in range(len(left_keys)):
        if left_columns[left_keys[k]][].null_count() > 0:
            any = True
            break
        if right_columns[right_keys[k]][].null_count() > 0:
            any = True
            break
    if not any:
        return List[Bool]()

    var out = List[Bool](capacity=left_rows + right_rows)
    for i in range(left_rows):
        var missing = False
        for k in range(len(left_keys)):
            if not left_columns[left_keys[k]][].is_valid(i):
                missing = True
                break
        out.append(missing)
    for i in range(right_rows):
        var missing = False
        for k in range(len(right_keys)):
            if not right_columns[right_keys[k]][].is_valid(i):
                missing = True
                break
        out.append(missing)
    return out^


def _probe_route[
    dt: DType
](
    left: Array[dt],
    right: Array[dt],
    left_rows: Int,
    mut codes: Array[DType.uint32],
) raises -> Int:
    """Builds a dictionary on the smaller side and probes the larger with it.

    Args:
        left: The left key column.
        right: The right key column.
        left_rows: How many rows the left side has, which is where the right
            side's codes start.
        codes: Filled with one ordinal per row of both sides.

    Parameters:
        dt: The key dtype.

    Returns:
        How many ordinals there are, counting the one for a row that matched
        nothing.

    Raises:
        Error: If the probe raises, which it does not.
    """
    if len(right) <= len(left):
        return _build_and_probe[dt](right, left_rows, left, 0, codes)
    return _build_and_probe[dt](left, 0, right, left_rows, codes)


def _build_and_probe[
    dt: DType
](
    build: Array[dt],
    build_at: Int,
    probe: Array[dt],
    probe_at: Int,
    mut codes: Array[DType.uint32],
) raises -> Int:
    """Picks the table the build side wants and runs both passes through it.

    An integer key whose values sit in a narrow enough range gets a table indexed
    by the value itself, which hashes nothing on either side and turns the probe
    into one load. The width worth accepting here is the build side's own height
    rather than `DIRECT_LIMIT`, because this table is standing in for a hash
    table over the same keys and costs four bytes a slot against sixteen, so a
    span near the row count is a table that is smaller than the one it replaces
    even when it is half empty. `factorize_dense` draws the line in the same
    place and says why at length.

    Args:
        build: The smaller side's key column.
        build_at: Where its codes start.
        probe: The larger side's key column.
        probe_at: Where its codes start.
        codes: Filled with one ordinal per row of both sides.

    Parameters:
        dt: The key dtype.

    Returns:
        The ordinal count, counting the miss ordinal.

    Raises:
        Error: If the parallel probe raises.
    """
    comptime if dt.is_integral():
        var ceiling = len(build)
        if ceiling < DIRECT_LIMIT:
            ceiling = DIRECT_LIMIT
        var plan = direct_plan[dt](build, ceiling)
        if plan.span >= 0:
            return _direct_route[dt](
                build, build_at, probe, probe_at, plan.span, plan.base, codes
            )
    return _hashed_route[dt](build, build_at, probe, probe_at, codes)


def _direct_route[
    dt: DType
](
    build: Array[dt],
    build_at: Int,
    probe: Array[dt],
    probe_at: Int,
    span: Int,
    base: Scalar[dt],
    mut codes: Array[DType.uint32],
) raises -> Int:
    """Aligns through a table indexed by the key value.

    Args:
        build: The smaller side's key column.
        build_at: Where its codes start.
        probe: The larger side's key column.
        probe_at: Where its codes start.
        span: How many slots the table needs.
        base: The value that indexes slot zero.
        codes: Filled with one ordinal per row of both sides.

    Parameters:
        dt: The key dtype.

    Returns:
        The ordinal count, counting the miss ordinal.

    Raises:
        Error: If the parallel probe raises.
    """
    # Zero means unseen, so the ordinal is stored plus one and the table needs no
    # initialization pass. `Buffer` already handed back zeroed memory.
    var seats = Buffer(span * 4)
    var table = seats.bitcast[DType.uint32]()

    var build_rows = len(build)
    var build_nulls = build.null_count() > 0
    var values = build.unsafe_ptr()
    var out = codes.unsafe_ptr()
    var found = 0
    for i in range(build_rows):
        if build_nulls and not build.data.validity.get(i):
            # Whatever is here is never read: every read of a build side code is
            # behind the caller's null list, and this row is in it.
            continue
        var at = Int(values.unsafe_offset(i).unsafe_load()) - Int(base)
        var stored = table.unsafe_offset(at).unsafe_load()
        if stored == 0:
            table.unsafe_offset(at).unsafe_write(UInt32(found + 1))
            out.unsafe_offset(build_at + i).unsafe_write(UInt32(found))
            found += 1
        else:
            out.unsafe_offset(build_at + i).unsafe_write(stored - 1)

    var miss = UInt32(found)
    var probe_rows = len(probe)
    var probe_nulls = probe.null_count() > 0

    def look(start: Int, stop: Int) raises {mut codes, imm}:
        var into = codes.unsafe_ptr()
        var reads = probe.unsafe_ptr()
        var slots = seats.bitcast[DType.uint32]()
        for i in range(start, stop):
            var to = probe_at + i
            if probe_nulls and not probe.data.validity.get(i):
                into.unsafe_offset(to).unsafe_write(miss)
                continue
            var at = Int(reads.unsafe_offset(i).unsafe_load()) - Int(base)
            if at < 0 or at >= span:
                # A key outside the build side's range, which the table has no
                # slot for and which nothing on the other side can match.
                into.unsafe_offset(to).unsafe_write(miss)
                continue
            var stored = slots.unsafe_offset(at).unsafe_load()
            into.unsafe_offset(to).unsafe_write(
                miss if stored == 0 else stored - 1
            )

    if probe_rows >= PARALLEL_PROBE_ROWS:
        parallel_morsels(look, probe_rows, PROBE_MORSEL_ROWS)
    else:
        look(0, probe_rows)

    _ = seats^
    return found + 1


def _hashed_route[
    dt: DType
](
    build: Array[dt],
    build_at: Int,
    probe: Array[dt],
    probe_at: Int,
    mut codes: Array[DType.uint32],
) raises -> Int:
    """Aligns through the hash table.

    The build writes into a list of its own and the result is copied across,
    because `HashTable.build` indexes the output by the same row number it
    indexes the validity bitmap by and the two cannot be pulled apart. The copy
    is over the smaller side, and it is a sequential write of four bytes a row
    against a build that is a random probe per row, so it does not show. The
    probe has no such problem: it was written for this and takes the offset.

    Args:
        build: The smaller side's key column.
        build_at: Where its codes start.
        probe: The larger side's key column.
        probe_at: Where its codes start.
        codes: Filled with one ordinal per row of both sides.

    Parameters:
        dt: The key dtype.

    Returns:
        The ordinal count, counting the miss ordinal.

    Raises:
        Error: If the parallel probe raises.
    """
    var build_rows = len(build)
    var build_nulls = build.null_count() > 0
    var table = HashTable(build_rows, DEFAULT_SEED)
    var firsts = List[Int]()
    var mine = Array[DType.uint32](build_rows)
    var hashes = Buffer(CHUNK_ROWS * 8)

    var at = 0
    while at < build_rows:
        var count = min(CHUNK_ROWS, build_rows - at)
        hash_chunk(build, at, count, DEFAULT_SEED, hashes)
        table.build(
            hashes,
            build.data.validity,
            build_nulls,
            at,
            at,
            count,
            build_rows,
            0,
            mine,
            firsts,
        )
        at += count

    var out = codes.unsafe_ptr()
    var got = mine.unsafe_ptr()
    for i in range(build_rows):
        out.unsafe_offset(build_at + i).unsafe_write(
            got.unsafe_offset(i).unsafe_load()
        )

    var miss = UInt32(len(table))
    var probe_rows = len(probe)
    var probe_nulls = probe.null_count() > 0

    def look(start: Int, stop: Int) raises {mut codes, imm}:
        # One buffer per morsel rather than one per job, because the whole point
        # of hashing a chunk at a time is that the chunk stays in this core's
        # cache, and a buffer every core is writing to would not.
        var scratch = Buffer(CHUNK_ROWS * 8)
        var here = start
        while here < stop:
            var count = min(CHUNK_ROWS, stop - here)
            hash_chunk(probe, here, count, DEFAULT_SEED, scratch)
            table.probe(
                scratch,
                probe.data.validity,
                probe_nulls,
                here,
                count,
                probe_at,
                miss,
                codes,
            )
            here += count
        _ = scratch^

    if probe_rows >= PARALLEL_PROBE_ROWS:
        parallel_morsels(look, probe_rows, PROBE_MORSEL_ROWS)
    else:
        look(0, probe_rows)

    return Int(miss) + 1
