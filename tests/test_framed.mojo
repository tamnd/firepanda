"""Tests for the windows that sort their partitions, in `kernel/framed.mojo`.

Every expected column is DuckDB 1.5.5's answer to the same query, run over the
same seven rows, quoted beside each test. The rows are two partitions, one with
two rows that tie on the order key and a null among the values, the other with
a null order key, which DuckDB sorts last:

    i  g  o     v
    0  1  3     10
    1  1  1     20
    2  1  2     null
    3  1  2     40
    4  2  5     5
    5  2  null  6
    6  2  4     7
"""

from std.testing import TestSuite, assert_equal, assert_raises

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.logical import LogicalType
from firepanda.kernel.cast import cast_any
from firepanda.kernel.framed import (
    EDGE_CURRENT_ROW,
    EDGE_FOLLOWING,
    EDGE_PRECEDING,
    EDGE_UNBOUNDED_FOLLOWING,
    EDGE_UNBOUNDED_PRECEDING,
    LEAVE_TIES,
    SPAN_GROUPS,
    SPAN_RANGE,
    SPAN_ROWS,
    WINDOW_CUME_DIST,
    WINDOW_DENSE_RANK,
    WINDOW_FIRST_VALUE,
    WINDOW_FOLD,
    WINDOW_LAG,
    WINDOW_LAST_VALUE,
    WINDOW_LEAD,
    WINDOW_NTH_VALUE,
    WINDOW_NTILE,
    WINDOW_PERCENT_RANK,
    WINDOW_RANK,
    WINDOW_ROW_NUMBER,
    WindowFrame,
    framed_windows,
    window_function_named,
    window_type,
)
from firepanda.kernel.group import AggKind


comptime NULL = -999
"""How a missing value is written in the columns below."""


def ints(values: List[Int]) -> AnyArray:
    """An int64 column, `NULL` marking a missing row."""
    var out = Array[DType.int64](len(values))
    for i in range(len(values)):
        if values[i] == NULL:
            out.set_null(i)
        else:
            out[i] = Int64(values[i])
    return AnyArray(out^)


def rows() -> List[AnyArray]:
    """The seven rows: `g`, `o`, `v` and the default `-1` for `lead`."""
    var flat = List[AnyArray]()
    flat.append(ints([1, 1, 1, 1, 2, 2, 2]))
    flat.append(ints([3, 1, 2, 2, 5, NULL, 4]))
    flat.append(ints([10, 20, NULL, 40, 5, 6, 7]))
    flat.append(ints([-1, -1, -1, -1, -1, -1, -1]))
    return flat^


def show(col: AnyArray, scale: Float64 = 1) raises -> String:
    """The column as text, each value times `scale` and rounded."""
    var wide = cast_any(col, LogicalType.FLOAT64, strict=False)
    var typed = wide.as_typed[DType.float64]()
    var out = String()
    for i in range(len(wide)):
        if i > 0:
            out += ","
        if not wide.is_valid(i):
            out += "null"
        else:
            var v = typed[i] * scale
            out += String(Int(v + 0.5) if v >= 0 else Int(v - 0.5))
    return out^


def run(
    var frame: WindowFrame,
    source: Int = 2,
    second: Int = -1,
    kind: AggKind = AggKind.SUM,
    ordered: Bool = True,
) raises -> AnyArray:
    """One window partitioned by `g`, ordered by `o` unless told otherwise."""
    var order = List[Int]()
    if ordered:
        order.append(1)
        frame.descending = [False]
        frame.nulls_last = [True]
    var frames = List[WindowFrame]()
    frames.append(frame^)
    var got = framed_windows(
        rows(),
        [0],
        order,
        frames,
        [source],
        [second],
        [kind],
        [True],
        [LogicalType.INT64],
    )
    return got[0].copy()


def called(function: Int, amount: Int = 1) -> WindowFrame:
    var frame = WindowFrame()
    frame.function = function
    frame.amount = amount
    return frame^


def test_the_ranks() raises:
    """The ranks: row_number, rank, dense_rank, percent_rank, cume_dist and ntile(3), over
    `(partition by g order by o)`."""
    assert_equal(show(run(called(WINDOW_ROW_NUMBER))), "4,1,2,3,2,3,1")
    assert_equal(show(run(called(WINDOW_RANK))), "4,1,2,2,2,3,1")
    assert_equal(show(run(called(WINDOW_DENSE_RANK))), "3,1,2,2,2,3,1")
    assert_equal(
        show(run(called(WINDOW_PERCENT_RANK)), 1000),
        "1000,0,333,333,500,1000,0",
    )
    assert_equal(
        show(run(called(WINDOW_CUME_DIST)), 1000),
        "1000,250,750,750,667,1000,333",
    )
    assert_equal(show(run(called(WINDOW_NTILE, 3))), "3,1,1,2,2,3,1")


def test_the_default_frame_runs_to_the_last_peer() raises:
    """The folds sum, count, avg, min and max over `(partition by g order by o)`, whose
    frame ends at the row's last peer, so the two rows that tie agree."""
    var fold = WindowFrame()
    assert_equal(show(run(fold.copy())), "70,20,60,60,12,18,7")
    assert_equal(
        show(run(fold.copy(), kind=AggKind.COUNT)), "3,1,2,2,2,3,1"
    )
    assert_equal(
        show(run(fold.copy(), kind=AggKind.MEAN), 1000),
        "23333,20000,30000,30000,6000,6000,7000",
    )
    assert_equal(
        show(run(fold.copy(), kind=AggKind.MIN)), "10,20,20,20,5,5,7"
    )
    assert_equal(
        show(run(fold.copy(), kind=AggKind.MAX)), "40,20,40,40,7,7,7"
    )


def test_no_order_is_the_whole_partition() raises:
    var fold = WindowFrame()
    assert_equal(show(run(fold^, ordered=False)), "70,70,70,70,18,18,18")


def test_rows_between_one_preceding_and_one_following() raises:
    var frame = WindowFrame()
    frame.mode = SPAN_ROWS
    frame.start = EDGE_PRECEDING
    frame.start_by = 1
    frame.end = EDGE_FOLLOWING
    frame.end_by = 1
    assert_equal(show(run(frame.copy())), "50,20,60,50,18,11,12")
    # median over the same frame, which is asked of the frame's own rows.
    assert_equal(
        show(run(frame^, kind=AggKind.MEDIAN), 10),
        "250,200,300,250,60,55,60",
    )


def test_range_between_one_preceding_and_current_row() raises:
    """The null order key counts along nothing, so its frame is its peers."""
    var frame = WindowFrame()
    frame.mode = SPAN_RANGE
    frame.start = EDGE_PRECEDING
    frame.start_by = 1
    frame.end = EDGE_CURRENT_ROW
    assert_equal(show(run(frame^)), "50,20,60,60,12,6,7")


def test_groups_between_one_preceding_and_one_preceding() raises:
    var frame = WindowFrame()
    frame.mode = SPAN_GROUPS
    frame.start = EDGE_PRECEDING
    frame.start_by = 1
    frame.end = EDGE_PRECEDING
    frame.end_by = 1
    assert_equal(show(run(frame^)), "40,null,20,20,7,5,null")


def test_exclude_ties_keeps_the_row() raises:
    var frame = WindowFrame()
    frame.mode = SPAN_ROWS
    frame.start = EDGE_UNBOUNDED_PRECEDING
    frame.end = EDGE_UNBOUNDED_FOLLOWING
    frame.exclude = LEAVE_TIES
    assert_equal(show(run(frame^)), "70,70,30,70,18,18,18")


def test_lag_and_lead() raises:
    assert_equal(show(run(called(WINDOW_LAG))), "40,null,20,null,7,5,null")
    assert_equal(
        show(run(called(WINDOW_LEAD), second=3)), "-1,null,40,10,6,-1,5"
    )
    var skip = called(WINDOW_LAG)
    skip.ignore_nulls = True
    assert_equal(show(run(skip^)), "40,null,20,20,7,5,null")


def test_the_values_of_the_frame() raises:
    assert_equal(
        show(run(called(WINDOW_FIRST_VALUE))), "20,20,20,20,7,7,7"
    )
    assert_equal(show(run(called(WINDOW_LAST_VALUE))), "10,20,40,40,5,6,7")
    assert_equal(
        show(run(called(WINDOW_NTH_VALUE, 2))),
        "null,null,null,null,5,5,null",
    )


def test_range_with_an_offset_needs_a_number() raises:
    var frame = WindowFrame()
    frame.mode = SPAN_RANGE
    frame.start = EDGE_PRECEDING
    frame.start_by = 1
    with assert_raises(contains="orders by one key"):
        _ = run(frame^, ordered=False)


def test_the_names_and_the_types() raises:
    assert_equal(window_function_named("dense_rank"), WINDOW_DENSE_RANK)
    assert_equal(window_function_named("sum"), -1)
    assert_equal(
        window_type(WINDOW_CUME_DIST, LogicalType.INT64), LogicalType.FLOAT64
    )
    assert_equal(
        window_type(WINDOW_LAG, LogicalType.FLOAT64), LogicalType.FLOAT64
    )
    assert_equal(
        window_type(WINDOW_ROW_NUMBER, LogicalType.FLOAT64),
        LogicalType.INT64,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
