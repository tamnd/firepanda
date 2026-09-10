"""The group by that does not group.

A group by builds a hash table because equal keys are scattered through the
column and it has no other way to find them. When they are not scattered, when
the column is in order and every group is one run of adjacent rows, the table is
answering a question the column has already answered. Walking it and closing a
group each time the value changes gives the same ordinals for one comparison a
row, no table, no hashing, no per worker partials to merge, and the ordinals come
out in key order rather than in first appearance order, which is the order a
group by is asked to report in anyway.

How much that is worth is not what the argument above suggests, and the
measurement is worth stating because it says when to expect anything. On a
million rows on a 13900K, against `group_ordinals` over the same sorted column:

- a thousand groups, 639 us hashed against 569 walked, 1.12x
- a hundred thousand groups, 639 us against 623 us, 1.03x
- a million groups, every row its own, 14.199 ms against 987 us, 14.4x

The pattern is not about cardinality as such, it is about run length. A sorted
column is the best case a hash table ever gets, because consecutive equal keys
probe the same slot and the table stays in cache, so when the runs are long there
is nothing left for the walk to take away and both routes sit against memory
bandwidth. The walk wins exactly when the runs are short, because that is when
the table has to insert on nearly every row and being in order stops helping it.
So this is a large win on a sorted column of nearly distinct values, which is
what `drop_duplicates` on a sorted key is, and close to nothing on a sorted
column of a few long runs. It is never a loss, which is what makes it safe to
take whenever the flag is set rather than only when something predicts a gain.

This is what Polars' sorted group by is and it is the one place in this library
where a flag on a column changes which algorithm runs. `ChunkedArray.order`
carries the flag, `sort_values` sets it on the key it just sorted, and
`prove_sorted` sets it for a caller willing to pay one scan to find out. Nothing
here guesses: an unset flag takes the ordinary route and is right, slower.

Two things are deliberately not handled and both are refusals rather than
mistakes. A column holding a null does not come in here at all, because
`Sortedness` says nothing about which end a sort put the nulls at and
`mark_sorted` already refuses to set the flag on such a column, so the run walk
would be comparing values under nulls. And only one key column is taken, because
the flag is per column and two columns each sorted on their own say nothing about
whether the pairs are in lexicographic order.

The walk is split the way the parallel filter is split, and for the same reason.
Which ordinal a row gets depends on how many groups closed before it, which a
worker handed the middle of the column does not know, so the boundaries are
counted first and a prefix sum over the per worker counts is the ordinal each
worker starts at. The comparison a worker makes at the first row of its own
stretch reads the row before it, which belongs to the worker before, and reads
are what make that safe.
"""

from std.sys.info import size_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import ALL
from firepanda.exec.parallel import parallel_for, worker_count

from .grouping import Grouping


comptime PARALLEL_RUN_ROWS = 1 << 16
"""Below this many rows the walk stays on one thread.

The same sixty five thousand the filter uses, and picked to match rather than
measured on its own. Both split a single pass over a column into a counting pass
and a writing pass, both give a worker a run of output nobody else touches, and
the pass here is cheaper per row than the filter's, so a threshold that is right
for the filter cannot be too low for this. If it ever earns its own number, the
number goes here with the sweep that placed it.
"""


def _run_bounds(rows: Int, workers: Int) -> List[Int]:
    """Cuts the column into one contiguous stretch per worker.

    A fixed split rather than the morsel queue, because the prefix sum needs the
    stretches to have an order and a worker to know which one it has. The pass is
    one comparison a row with no data dependent work in it, so there is nothing
    for the queue's load balancing to correct for.

    Args:
        rows: How many rows there are.
        workers: How many stretches to cut.

    Returns:
        `workers + 1` positions, the first zero and the last `rows`.
    """
    var out = List[Int](capacity=workers + 1)
    var each = rows // workers
    var extra = rows % workers
    var at = 0
    out.append(0)
    for w in range(workers):
        at += each + (1 if w < extra else 0)
        out.append(at)
    return out^


def _run_ordinals[
    dt: DType
](col: Array[dt], mut rows_at: List[Int]) raises -> Array[DType.uint32]:
    """Numbers the runs of equal values in a sorted column.

    Args:
        col: The key column, in order and with no nulls.
        rows_at: Filled with the first row of each group, in ordinal order.

    Parameters:
        dt: The key dtype.

    Returns:
        One ordinal per row, in `[0, len(rows_at))`.

    Raises:
        Error: Only what the parallel runtime raises.
    """
    var n = len(col)
    var codes = Array[DType.uint32](overwritten=n)
    if n == 0:
        return codes^

    var values = col.unsafe_ptr()
    var out = codes.unsafe_ptr()
    var workers = worker_count()
    if n < PARALLEL_RUN_ROWS or workers <= 1:
        var g = -1
        for i in range(n):
            if (
                i == 0
                or values.unsafe_offset(i)[] != values.unsafe_offset(i - 1)[]
            ):
                g += 1
                rows_at.append(i)
            out.unsafe_offset(i).unsafe_write(UInt32(g))
        return codes^

    var most = n // PARALLEL_RUN_ROWS
    if workers > most:
        workers = most
    var bounds = _run_bounds(n, workers)

    # How many groups open inside each worker's stretch. A group opens at row
    # zero and at every row whose value differs from the one before it, so the
    # count is the same test the writing pass makes and the two passes cannot
    # disagree about where a group starts.
    var opened = Buffer(workers * size_of[DType.int64]())

    def measure(w: Int) raises {mut opened, imm}:
        var runs = 0
        for i in range(bounds[w], bounds[w + 1]):
            if (
                i == 0
                or values.unsafe_offset(i)[] != values.unsafe_offset(i - 1)[]
            ):
                runs += 1
        opened.bitcast[DType.int64]().unsafe_offset(w).unsafe_store(Int64(runs))

    parallel_for(measure, workers)

    var before = List[Int](length=workers + 1, fill=0)
    var counted = opened.bitcast[DType.int64]()
    for w in range(workers):
        before[w + 1] = before[w] + Int(counted.unsafe_offset(w).unsafe_load())

    var groups = before[workers]
    var firsts = Buffer(overwritten=groups * size_of[DType.int64]())

    def number(w: Int) raises {mut firsts, imm}:
        var starts = firsts.bitcast[DType.int64]()
        # One below the first ordinal this stretch opens, so that a stretch
        # beginning in the middle of a group carries that group's ordinal until
        # it closes. The first row of the column always opens a group, so worker
        # zero never writes the ordinal below zero it starts with.
        var g = before[w] - 1
        for i in range(bounds[w], bounds[w + 1]):
            if (
                i == 0
                or values.unsafe_offset(i)[] != values.unsafe_offset(i - 1)[]
            ):
                g += 1
                starts.unsafe_offset(g).unsafe_store(Int64(i))
            out.unsafe_offset(i).unsafe_write(UInt32(g))

    parallel_for(number, workers)

    # The one serial pass left, and it is over the groups rather than the rows.
    # `Grouping` wants a `List[Int]` and the writing pass wanted somewhere it
    # could store to by index from several threads at once, which a list is not.
    var starts = firsts.bitcast[DType.int64]()
    rows_at = List[Int](capacity=groups)
    for g in range(groups):
        rows_at.append(Int(starts.unsafe_offset(g).unsafe_load()))
    return codes^


def sorted_ordinals(col: AnyArray) raises -> Optional[Grouping]:
    """Groups a column whose values are already in order, if it can.

    The caller has established the order, normally by reading
    `ChunkedArray.order`, and this checks only the things that would make the
    walk wrong rather than slow: a null anywhere, or a dtype whose values are not
    laid out where the walk would read them.

    Args:
        col: The key column, which the caller has established is in order.

    Returns:
        The grouping, or nothing when this column cannot take the route, in which
        case the caller falls back to `group_ordinals` and gets the same answer.

    Raises:
        Error: Only what the parallel runtime raises.
    """
    # A string column would match the uint8 arm of the dispatch below and group
    # on the first byte of each view, which is the same trap `factorize_any`
    # guards against and the same guard. Text keys are worth having here and are
    # not here yet.
    if col.is_string() or col.is_dictionary() or col.is_nested():
        return None
    # `mark_sorted` refuses to set the flag on a column with nulls, so this
    # should not fire. It is checked rather than assumed because what it costs
    # is a popcount over the validity words and what it buys is that a caller
    # that sets the flag some other way cannot get a wrong answer out of it.
    if col.null_count() > 0:
        return None

    comptime for candidate in ALL:
        if col.dtype() == candidate:
            var rows_at = List[Int]()
            var codes = _run_ordinals[candidate](
                col.as_typed_view[candidate](), rows_at
            )
            var groups = len(rows_at)
            return Grouping(codes^, groups, rows_at^)
    return None
