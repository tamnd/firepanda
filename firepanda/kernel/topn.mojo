"""The best few rows of every group, without sorting anything.

`group.mojo` reduces each group to one value. This picks each group's `n` best
rows and reports their positions, which is what pandas spells
`groupby(...).nlargest(n)` and what db-benchmark's eighth query asks for.

## Why this is not a sort

The obvious implementation sorts the frame by the value column and then walks
each group taking the first `n` rows it sees. That is what pandas does and it
costs a full sort of every row, a permutation of the column and a pass to undo
it, to answer a question about two rows in a hundred. This keeps a small table
instead: `n` slots per group, and one comparison per row against the worst thing
currently in that group's slots. A row that loses that comparison is never
touched again, and on real data almost every row loses it.

The comparison is against a value held in its own array rather than against the
slots themselves, so the common path reads one entry and stops. The slots are
only touched when a row is going to be kept, which for a group of a hundred rows
keeping two of them happens a handful of times.

## Ties, and why they are broken by row number

Two rows with the same value both deserve the last slot and only one can have it.
Keeping whichever the machine happened to see first would make the answer depend
on how the rows were split across cores, which is a benchmark that gives a
different result every run and a test nobody can write. So the order is over the
pair rather than the value: a row beats another if its value is better, or if the
values are equal and it came first. That is a total order over distinct rows, so
the answer is the same whether the column was scanned by one core or by thirty
two, and it is the answer pandas gives, which keeps the earlier row on a tie.

## Nulls and NaN

A null is not a value, so a null row is never a candidate and a group with fewer
than `n` present values gives back fewer than `n` rows. NaN is dropped for the
same reason and for a second one: NaN loses every comparison it is in, so a NaN
that reached a slot would never be displaced and would hold a real value out.
pandas drops NaN from `nlargest` as well.

## How it runs on every core

The same shape as `group.mojo`'s replicated route and for the same reason: two
rows in two slices can belong to the same group, so each worker gets its own
table and the tables are merged afterwards. The merge folds every other worker's
slots into the first worker's, a block of groups at a time, and a block belongs
to exactly one worker, so no two of them write the same slot. Feeding a slot back
through the same insert the scan uses is what makes the merge correct: the order
over pairs does not care whether a candidate arrived from a row or from another
worker's table.

A table is `n` slots per group, so the memory is proportional to
`n * groups * workers` and the ceiling is the one `group.mojo` has. Past it the
scan stays on one core. The fix for that case is partitioning the rows by group
ordinal, which is `firepanda/hash/partition.mojo`'s job and is not done here.
"""

from std.sys.info import size_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import NUMERIC
from firepanda.exec.parallel import parallel_for, worker_count
from firepanda.kernel.accum import highest, lowest

comptime TOP_PRIVATE_BYTES = 64 * 1024 * 1024
"""How much memory the private slot tables may take, in total.

Twice `group.mojo`'s budget, because an entry here is wider than an accumulator
and the same group count would otherwise buy half as many workers. A group costs
`n` values and `n` row numbers plus a count and its current worst value, so at
two slots of float64 that is thirty six bytes against an accumulator's eight.
Sixty four megabytes buys seventeen workers at a hundred thousand groups, which
is the shape db-benchmark's eighth query has.
"""

comptime TOP_PRIVATE_ROWS = 1 << 16
"""Rows below which the scan stays on one core.

The number `group.mojo` uses, picked the same way: below a scatter of about this
size, starting a task per worker and folding their tables afterwards costs more
than the scan being split.
"""


struct GroupTop(Movable):
    """Which rows of each group were kept, and how many of them there were."""

    var rows_at: List[Int]
    """The kept rows, groups in ordinal order, best first inside a group."""

    var counts: List[Int]
    """How many rows each group kept, one entry per group. Fewer than `n` where
    the group did not have that many present values."""

    def __init__(out self, var rows_at: List[Int], var counts: List[Int]):
        """Constructs a result.

        Args:
            rows_at: The kept rows.
            counts: How many were kept per group.
        """
        self.rows_at = rows_at^
        self.counts = counts^


struct _Table[dt: DType](Copyable, Movable):
    """One worker's slot table, as four pointers into buffers it does not own.

    Parameters:
        dt: The value dtype.
    """

    var slots: Pointer[Scalar[Self.dt], MutUntrackedOrigin]
    """`groups * n` values, group major."""

    var rows_at: Pointer[Scalar[DType.uint32], MutUntrackedOrigin]
    """`groups * n` row numbers, alongside `slots`."""

    var filled: Pointer[Scalar[DType.uint32], MutUntrackedOrigin]
    """How many slots each group has used, one entry per group."""

    var worst: Pointer[Scalar[Self.dt], MutUntrackedOrigin]
    """Each group's least good slot value, one entry per group. Holds the
    identity until the group's slots are full, which is what lets the scan's gate
    be a single comparison."""

    var n: Int
    """How many slots each group has."""

    def __init__(
        out self,
        slots: Pointer[Scalar[Self.dt], MutUntrackedOrigin],
        rows_at: Pointer[Scalar[DType.uint32], MutUntrackedOrigin],
        filled: Pointer[Scalar[DType.uint32], MutUntrackedOrigin],
        worst: Pointer[Scalar[Self.dt], MutUntrackedOrigin],
        n: Int,
    ):
        """Constructs a view of one worker's table.

        Args:
            slots: The values.
            rows_at: The row numbers.
            filled: The per group counts.
            worst: The per group thresholds.
            n: The slots per group.
        """
        self.slots = slots
        self.rows_at = rows_at
        self.filled = filled
        self.worst = worst
        self.n = n


def _beats[
    dt: DType, largest: Bool
](
    value: Scalar[dt],
    at: Scalar[DType.uint32],
    other: Scalar[dt],
    other_at: Scalar[DType.uint32],
) -> Bool:
    """Reports whether one candidate row outranks another.

    Args:
        value: The first row's value.
        at: The first row's position.
        other: The second row's value.
        other_at: The second row's position.

    Parameters:
        dt: The value dtype.
        largest: True to rank high values first, False to rank low ones first.

    Returns:
        True if the first row outranks the second.
    """
    comptime if largest:
        if value > other:
            return True
    else:
        if value < other:
            return True
    return value == other and at < other_at


def _weakest[
    dt: DType, //, largest: Bool
](table: _Table[dt], base: Int, used: Int) -> Int:
    """Returns which of a group's slots holds the candidate everything beats.

    Args:
        table: The table to look in.
        base: Where the group's slots start.
        used: How many of them hold a candidate.

    Parameters:
        dt: The value dtype.
        largest: True if high values rank first.

    Returns:
        The slot's offset from `base`.
    """
    var found = 0
    var value = table.slots.unsafe_offset(base).unsafe_load()
    var at = table.rows_at.unsafe_offset(base).unsafe_load()
    for k in range(1, used):
        var other = table.slots.unsafe_offset(base + k).unsafe_load()
        var other_at = table.rows_at.unsafe_offset(base + k).unsafe_load()
        if _beats[largest=largest](value, at, other, other_at):
            value = other
            at = other_at
            found = k
    return found


def _offer[
    dt: DType, //, largest: Bool
](table: _Table[dt], value: Scalar[dt], at: Scalar[DType.uint32], group: Int,):
    """Puts a candidate into a group's slots if it belongs there.

    Args:
        table: The table to put it in.
        value: The candidate's value.
        at: The candidate's row number.
        group: Which group's slots to offer it to.

    Parameters:
        dt: The value dtype.
        largest: True to keep the high values, False to keep the low ones.
    """
    var base = group * table.n
    var used = Int(table.filled.unsafe_offset(group).unsafe_load())

    if used < table.n:
        table.slots.unsafe_offset(base + used).unsafe_store(value)
        table.rows_at.unsafe_offset(base + used).unsafe_store(at)
        used += 1
        table.filled.unsafe_offset(group).unsafe_store(
            Scalar[DType.uint32](used)
        )
        # While the slots have room every candidate is kept, so the threshold
        # only has to become real once there is something for it to turn away.
        if used == table.n:
            var beaten = _weakest[largest=largest](table, base, used)
            table.worst.unsafe_offset(group).unsafe_store(
                table.slots.unsafe_offset(base + beaten).unsafe_load()
            )
        return

    var beaten = _weakest[largest=largest](table, base, used)
    if not _beats[largest=largest](
        value,
        at,
        table.slots.unsafe_offset(base + beaten).unsafe_load(),
        table.rows_at.unsafe_offset(base + beaten).unsafe_load(),
    ):
        return

    table.slots.unsafe_offset(base + beaten).unsafe_store(value)
    table.rows_at.unsafe_offset(base + beaten).unsafe_store(at)
    var now = _weakest[largest=largest](table, base, used)
    table.worst.unsafe_offset(group).unsafe_store(
        table.slots.unsafe_offset(base + now).unsafe_load()
    )


def _rank_slots[
    dt: DType, //, largest: Bool
](table: _Table[dt], base: Int, used: Int):
    """Puts one group's slots in order, best first.

    An insertion sort, because `used` is at most `n` and `n` is the handful of
    rows somebody asked for rather than a length that grows with the data.

    Args:
        table: The table holding them.
        base: Where the group's slots start.
        used: How many of them hold a candidate.

    Parameters:
        dt: The value dtype.
        largest: True if high values rank first.
    """
    for k in range(1, used):
        var value = table.slots.unsafe_offset(base + k).unsafe_load()
        var at = table.rows_at.unsafe_offset(base + k).unsafe_load()
        var j = k
        while j > 0:
            var prior = table.slots.unsafe_offset(base + j - 1).unsafe_load()
            var prior_at = table.rows_at.unsafe_offset(
                base + j - 1
            ).unsafe_load()
            if _beats[largest=largest](prior, prior_at, value, at):
                break
            table.slots.unsafe_offset(base + j).unsafe_store(prior)
            table.rows_at.unsafe_offset(base + j).unsafe_store(prior_at)
            j -= 1
        table.slots.unsafe_offset(base + j).unsafe_store(value)
        table.rows_at.unsafe_offset(base + j).unsafe_store(at)


def _scan_into[
    dt: DType, //, origin: ImmOrigin, largest: Bool
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    codes: Array[DType.uint32],
    start: Int,
    stop: Int,
    table: _Table[dt],
):
    """Offers every row of a range to its group's slots.

    Args:
        source: The values being ranked.
        validity: Which rows of the values are present.
        codes: One group ordinal per row.
        start: The first row of the range.
        stop: One past the last row of the range.
        table: The slots to offer them to.

    Parameters:
        dt: The value dtype.
        origin: The values pointer's origin.
        largest: True to keep the high values, False to keep the low ones.
    """
    var ordinals = codes.unsafe_ptr()
    for i in range(start, stop):
        if not validity.get(i):
            continue
        var value = source.unsafe_offset(i).unsafe_load()
        comptime if dt.is_floating_point():
            if value != value:
                continue

        var group = Int(ordinals.unsafe_offset(i).unsafe_load())
        # A row that cannot reach the group's last slot is the common case and
        # costs one load and one comparison. A tie goes the long way round,
        # because the row number decides it, and ties are rare.
        comptime if largest:
            if value < table.worst.unsafe_offset(group).unsafe_load():
                continue
        else:
            if value > table.worst.unsafe_offset(group).unsafe_load():
                continue

        _offer[largest=largest](table, value, Scalar[DType.uint32](i), group)


def _table_workers[dt: DType](rows: Int, groups: Int, n: Int) -> Int:
    """Decides how many private slot tables the scan should build.

    Args:
        rows: How many rows are being scanned.
        groups: How many groups they land in.
        n: How many slots each group has.

    Parameters:
        dt: The value dtype, which sets how wide a slot is.

    Returns:
        The worker count, or one to say the scan should stay serial.
    """
    if rows < TOP_PRIVATE_ROWS or groups <= 0 or n <= 0:
        return 1

    var workers = worker_count()
    if workers <= 1:
        return 1

    var each = groups * (n * (size_of[dt]() + 4) + 4 + size_of[dt]())
    if each <= 0:
        return 1
    var affordable = TOP_PRIVATE_BYTES // each
    if affordable < 2:
        return 1
    return workers if workers < affordable else affordable


def _group_top_core[
    dt: DType, //, origin: ImmOrigin, largest: Bool
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    codes: Array[DType.uint32],
    groups: Int,
    n: Int,
) raises -> GroupTop:
    """Picks each group's best `n` rows out of a column.

    Args:
        source: The values being ranked.
        validity: Which rows of the values are present.
        codes: One group ordinal per row.
        groups: How many distinct ordinals there are.
        n: How many rows to keep per group.

    Parameters:
        dt: The value dtype.
        origin: The values pointer's origin.
        largest: True to keep the high values, False to keep the low ones.

    Returns:
        The kept rows and the per group counts.

    Raises:
        Error: If one of the workers cannot be run.
    """
    var rows = len(codes)
    var workers = _table_workers[dt](rows, groups, n)
    var span = groups * n

    # None of these four is read before it is written, so none of them needs the
    # memset an ordinary `Buffer` does. A slot and its row number are read at
    # position `k` of a group only once that group's count has passed `k`, and
    # the count only passes `k` after `_offer` has written both. The count and
    # the threshold are set by the worker that owns them, at the top of its own
    # scan. On a hundred thousand groups the memset being skipped here is tens
    # of megabytes and was most of what the call cost.
    var slots = Buffer(overwritten=span * workers * size_of[dt]())
    var rows_at = Buffer(overwritten=span * workers * 4)
    var filled = Buffer(overwritten=groups * workers * 4)
    var worst = Buffer(overwritten=groups * workers * size_of[dt]())

    # Nothing is in the slots yet, so every row has to be allowed through to
    # them. The identity is what the gate compares against until a group fills
    # up, and no value is worse than it. Each worker prepares its own stretch
    # rather than one thread preparing all of them, which makes the pass
    # parallel and leaves the pages resident on the core about to read them.
    var blank = lowest[dt]() if largest else highest[dt]()

    # A static split rather than a morsel queue, for the reason `group.mojo`
    # gives: every row of a scan costs about the same, so there is nothing for a
    # queue to balance, and a contiguous piece is read forwards.
    var bounds = List[Int](capacity=workers + 1)
    for w in range(workers):
        bounds.append(rows * w // workers)
    bounds.append(rows)

    def scan(
        w: Int,
    ) raises {mut slots, mut rows_at, mut filled, mut worst, imm}:
        var seeds = worst.bitcast[dt]().unsafe_offset(w * groups)
        var empty = filled.bitcast[DType.uint32]().unsafe_offset(w * groups)
        for g in range(groups):
            seeds.unsafe_offset(g).unsafe_store(blank)
            empty.unsafe_offset(g).unsafe_store(UInt32(0))

        _scan_into[largest=largest](
            source,
            validity,
            codes,
            bounds[w],
            bounds[w + 1],
            _Table[dt](
                slots.bitcast[dt]()
                .unsafe_offset(w * span)
                .unsafe_origin_cast[MutUntrackedOrigin](),
                rows_at.bitcast[DType.uint32]()
                .unsafe_offset(w * span)
                .unsafe_origin_cast[MutUntrackedOrigin](),
                filled.bitcast[DType.uint32]()
                .unsafe_offset(w * groups)
                .unsafe_origin_cast[MutUntrackedOrigin](),
                worst.bitcast[dt]()
                .unsafe_offset(w * groups)
                .unsafe_origin_cast[MutUntrackedOrigin](),
                n,
            ),
        )

    parallel_for(scan, workers)

    # The fold and the ordering both run a block of groups at a time. Everything
    # is folded into the first worker's table, and a block of groups belongs to
    # exactly one folding worker, so no two of them write the same slot.
    var blocks = workers
    if blocks > groups:
        blocks = groups

    def fold(
        b: Int,
    ) raises {mut slots, mut rows_at, mut filled, mut worst, imm}:
        var values = slots.bitcast[dt]().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        var marks = rows_at.bitcast[DType.uint32]().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        var used = filled.bitcast[DType.uint32]().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        var least = worst.bitcast[dt]().unsafe_origin_cast[MutUntrackedOrigin]()
        var into = _Table[dt](values, marks, used, least, n)

        var first = groups * b // blocks
        var last = groups * (b + 1) // blocks
        for g in range(first, last):
            for w in range(1, workers):
                var base = w * span + g * n
                var have = Int(used.unsafe_offset(w * groups + g).unsafe_load())
                for k in range(have):
                    _offer[largest=largest](
                        into,
                        values.unsafe_offset(base + k).unsafe_load(),
                        marks.unsafe_offset(base + k).unsafe_load(),
                        g,
                    )
            _rank_slots[largest=largest](
                into, g * n, Int(used.unsafe_offset(g).unsafe_load())
            )

    parallel_for(fold, blocks)

    var counts = List[Int](capacity=groups)
    var kept = List[Int]()
    var picked = rows_at.bitcast[DType.uint32]()
    var used = filled.bitcast[DType.uint32]()
    for g in range(groups):
        var have = Int(used.unsafe_offset(g).unsafe_load())
        counts.append(have)
        for k in range(have):
            kept.append(Int(picked.unsafe_offset(g * n + k).unsafe_load()))
    return GroupTop(kept^, counts^)


def _check(values: Int, codes: Int, n: Int) raises:
    """Rejects the arguments no route can answer.

    Args:
        values: How many values there are.
        codes: How many group ordinals there are.
        n: How many rows to keep per group.

    Raises:
        Error: If `n` is not positive, if the two lengths differ, or if a row
            number would not fit in the thirty two bits a slot holds.
    """
    if n <= 0:
        raise Error("group top: n must be at least one")
    if values != codes:
        raise Error(
            "group top: the values and the group ordinals are different lengths"
        )
    if values > Int(Scalar[DType.uint32].MAX):
        raise Error("group top: more than four billion rows")


def group_top_rows[
    dt: DType
](
    col: Array[dt],
    codes: Array[DType.uint32],
    groups: Int,
    n: Int,
    largest: Bool,
) raises -> GroupTop:
    """Picks each group's best `n` rows.

    Args:
        col: The values being ranked.
        codes: One group ordinal per row, as `group_ordinals` produces.
        groups: How many distinct ordinals there are.
        n: How many rows to keep per group. At least one.
        largest: True to keep each group's highest values, False its lowest.

    Parameters:
        dt: The value dtype.

    Returns:
        The kept rows and the per group counts.

    Raises:
        Error: If `n` is not positive, if the values and the codes are different
            lengths, or if there are more rows than a row number can hold.
    """
    _check(len(col), len(codes), n)
    if largest:
        return _group_top_core[largest=True](
            col.unsafe_ptr(), col.data.validity, codes, groups, n
        )
    return _group_top_core[largest=False](
        col.unsafe_ptr(), col.data.validity, codes, groups, n
    )


def group_top_rows_any(
    col: AnyArray,
    codes: Array[DType.uint32],
    groups: Int,
    n: Int,
    largest: Bool,
) raises -> GroupTop:
    """Picks each group's best `n` rows out of a column typed at runtime.

    Args:
        col: The values being ranked.
        codes: One group ordinal per row.
        groups: How many distinct ordinals there are.
        n: How many rows to keep per group. At least one.
        largest: True to keep each group's highest values, False its lowest.

    Returns:
        The kept rows and the per group counts.

    Raises:
        Error: As `group_top_rows` does, and if the column is not numeric.
    """
    _check(len(col), len(codes), n)
    comptime for candidate in NUMERIC:
        if col.dtype() == candidate:
            if largest:
                return _group_top_core[largest=True](
                    col.unsafe_ptr[candidate](),
                    col.data.validity,
                    codes,
                    groups,
                    n,
                )
            return _group_top_core[largest=False](
                col.unsafe_ptr[candidate](),
                col.data.validity,
                codes,
                groups,
                n,
            )
    raise Error("group top: the ranked column must be a numeric one")
