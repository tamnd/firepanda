"""Tests for the rolling and expanding windows.

Two things are being checked and they fail differently. The first is where the
window sits, which is `window`, `min_periods`, `center`, `closed` and `step`,
and every one of those has an answer that is wrong in a way that still looks
like a column of plausible numbers. So those are checked against a running
pandas 3.0.5, quoted in each test, on a column of nought to nine where every
window sums to something a reader can check by hand.

The second is the arithmetic over a window that holds an infinity, and that one
is checked against what is true rather than against pandas, because pandas
carries a single running total and cannot recover from an infinity passing
through it. The tests that assert the difference say so.
"""

from std.math import isinf, isnan
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.frame.series import Series
from firepanda.kernel.window import (
    Shape,
    WindowEdge,
    WindowOp,
    edge_named,
    op_named,
)


def column(values: List[Float64]) -> Series:
    """Builds a float64 series, where a NaN is how a missing row is written.

    Args:
        values: The values.

    Returns:
        The series, named `v`.
    """
    var out = Array[DType.float64](len(values))
    for i in range(len(values)):
        out[i] = values[i]
    return Series("v", AnyArray(out^))


def counting() -> Series:
    """Returns nought to nine, which is the column most of these run on.

    Returns:
        The series.
    """
    var values = List[Float64]()
    for i in range(10):
        values.append(Float64(i))
    return column(values)


def rows(series: Series) raises -> List[Float64]:
    """Reads a float64 series out as plain numbers.

    Args:
        series: The series.

    Returns:
        Its values, with a missing row read back as a NaN.

    Raises:
        Error: If the series does not hold float64.
    """
    var out = List[Float64]()
    ref values = series.values.as_typed_view[DType.float64]()
    for i in range(len(series.values)):
        out.append(values[i])
    return out^


def rolled(
    series: Series,
    kind: StringSlice,
    width: Int,
    min_periods: Optional[Int] = None,
    center: Bool = False,
    closed: StringSlice = "right",
    step: Optional[Int] = None,
) raises -> List[Float64]:
    """Runs a rolling reduction and hands the answer back as numbers.

    Args:
        series: The column.
        kind: The reduction.
        width: How many rows wide.
        min_periods: How many values a window needs.
        center: Whether the window sits around its row.
        closed: Which ends the window keeps.
        step: How many rows apart the answered rows are.

    Returns:
        The answer.

    Raises:
        Error: Whatever the kernel raises.
    """
    return rows(
        series.rolling(
            op_named(kind),
            width,
            min_periods,
            center,
            edge_named(closed),
            step,
        )
    )


def expanded(
    series: Series, kind: StringSlice, min_periods: Int = 1
) raises -> List[Float64]:
    """Runs an expanding reduction and hands the answer back as numbers.

    Args:
        series: The column.
        kind: The reduction.
        min_periods: How many values a window needs.

    Returns:
        The answer.

    Raises:
        Error: Whatever the kernel raises.
    """
    return rows(series.expanding(op_named(kind), min_periods))


def assert_rows(
    got: List[Float64], want: List[Float64], message: String = ""
) raises:
    """Compares two columns of numbers, reading a NaN as a missing row.

    Args:
        got: What was answered.
        want: What pandas answers, or what is true where the two differ.
        message: What the caller is asserting.

    Raises:
        Error: If the two are different heights or differ in any row.
    """
    assert_equal(len(got), len(want), "height: " + message)
    for i in range(len(want)):
        if isnan(want[i]):
            assert_true(
                isnan(got[i]),
                "row " + String(i) + " should be missing: " + message,
            )
        else:
            assert_equal(got[i], want[i], "row " + String(i) + ": " + message)


comptime GONE = Float64(0) / Float64(0)
"""A missing row, written the way a float column writes one."""


def test_a_rolling_sum_is_missing_until_the_window_is_full() raises:
    """Pandas gives `[nan, nan, nan, nan, 10, 15, 20, 25, 30, 35]` here.

    The four holes at the top are the answer people are surprised by once, and
    they are there because a rolling window that is not told how many values it
    needs needs all of them.
    """
    assert_rows(
        rolled(counting(), "sum", 5),
        [GONE, GONE, GONE, GONE, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
    )


def test_asking_for_one_period_fills_the_holes_with_partial_sums() raises:
    """Pandas gives `[0, 1, 3, 6, 10, 15, 20, 25, 30, 35]` for the same window.

    A completely different column from the one above, out of the same five wide
    window, which is why `min_periods` is worth a test of its own.
    """
    assert_rows(
        rolled(counting(), "sum", 5, min_periods=1),
        [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
    )


def test_a_centred_window_sits_around_its_row() raises:
    """Pandas gives `[nan, nan, 10, 15, 20, 25, 30, 35, nan, nan]`."""
    assert_rows(
        rolled(counting(), "sum", 5, center=True),
        [GONE, GONE, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, GONE, GONE],
    )


def test_a_centred_window_of_an_even_width_leans_left() raises:
    """Pandas gives `[nan, nan, 6, 10, 14, 18, 22, 26, 30, nan]` for a width of
    four, so the window at row two is rows nought to three and takes two rows
    from the left and one from the right. Nobody guesses which way that leans,
    which is the only reason this test exists separately from the one above.
    """
    assert_rows(
        rolled(counting(), "sum", 4, center=True),
        [GONE, GONE, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, GONE],
    )


def test_the_four_closed_rules_move_the_two_ends() raises:
    """Pandas, on the same five wide window over nought to nine.

    `left` gives `[nan] * 5 + [10, 15, 20, 25, 30]`, which is the default
    shifted down a row. `both` gives `[nan] * 4 + [10, 15, 21, 27, 33, 39]`,
    which holds six rows rather than five once there are six rows to hold.
    `neither` holds four and needs five, so it is missing everywhere.
    """
    assert_rows(
        rolled(counting(), "sum", 5, closed="left"),
        [GONE, GONE, GONE, GONE, GONE, 10.0, 15.0, 20.0, 25.0, 30.0],
        "left",
    )
    assert_rows(
        rolled(counting(), "sum", 5, closed="both"),
        [GONE, GONE, GONE, GONE, 10.0, 15.0, 21.0, 27.0, 33.0, 39.0],
        "both",
    )
    var nothing = rolled(counting(), "sum", 5, closed="neither")
    for i in range(len(nothing)):
        assert_true(isnan(nothing[i]), "neither is four rows and needs five")


def test_a_step_answers_fewer_rows_and_keeps_their_labels() raises:
    """Pandas gives `[nan, 6, 18, 30]` at labels `[0, 3, 6, 9]` for a four wide
    window stepped by three.

    The step is a property of the answer and not of the window, so every value
    here also appears in the unstepped answer at the same label.
    """
    var stepped = counting().rolling(
        WindowOp.SUM, 4, None, False, WindowEdge.RIGHT, Optional(3)
    )
    assert_rows(rows(stepped), [GONE, 6.0, 18.0, 30.0])
    assert_equal(len(stepped.index), 4)
    var labels = stepped.index.materialize()
    ref kept = labels.as_typed_view[DType.int64]()
    assert_equal(kept[0], 0)
    assert_equal(kept[1], 3)
    assert_equal(kept[2], 6)
    assert_equal(kept[3], 9)


def test_a_window_of_one_is_the_column_itself() raises:
    """Pandas gives nought to nine back, which is the only window where there is
    nothing to accumulate and so the only one that can be compared exactly."""
    assert_rows(
        rolled(counting(), "sum", 1),
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )


def test_a_window_wider_than_the_column_is_missing_and_not_an_error() raises:
    """Pandas answers a column of nulls rather than raising."""
    var wide = rolled(counting(), "sum", 100)
    assert_equal(len(wide), 10)
    for i in range(len(wide)):
        assert_true(isnan(wide[i]))


def test_counting_asks_how_many_rows_and_not_how_many_values() raises:
    """Pandas gives `[nan, nan, 2, 1, 2, 2]` for a three wide count over
    `[1, nan, 3, nan, 5, 6]`.

    Row three holds one value out of three rows and answers one rather than
    being missing, which is the whole difference between `count` and every
    other reduction here: what `min_periods` is tested against is the height of
    the window rather than what is in it.
    """
    var holed = column([1.0, GONE, 3.0, GONE, 5.0, 6.0])
    assert_rows(rolled(holed, "count", 3), [GONE, GONE, 2.0, 1.0, 2.0, 2.0])


def test_a_missing_row_is_stepped_over_and_still_counted_against() raises:
    """Pandas gives all nulls for `rolling(3).sum()` over the same column,
    because no three wide window there holds three values, and
    `[1, 1, 2, 3, 4, 5.5]` for `rolling(3, min_periods=1).mean()`.
    """
    var holed = column([1.0, GONE, 3.0, GONE, 5.0, 6.0])
    var summed = rolled(holed, "sum", 3)
    for i in range(len(summed)):
        assert_true(isnan(summed[i]), "no window holds three values")
    assert_rows(
        rolled(holed, "mean", 3, min_periods=1),
        [1.0, 1.0, 2.0, 3.0, 4.0, 5.5],
    )


def test_an_expanding_window_starts_at_the_top_and_needs_one_value() raises:
    """Pandas gives `[0, 1, 3, 6, 10, 15, 21, 28, 36, 45]`, with no hole at the
    top, because an expanding window defaults to needing one value where a
    rolling one defaults to needing all of them."""
    assert_rows(
        expanded(counting(), "sum"),
        [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0],
    )


def test_an_expanding_window_can_be_told_to_wait() raises:
    """Pandas gives `[nan] * 4 + [10, 15, 21, 28, 36, 45]` for five periods."""
    assert_rows(
        expanded(counting(), "sum", min_periods=5),
        [GONE, GONE, GONE, GONE, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0],
    )


def test_the_last_row_of_an_expanding_window_is_the_whole_column() raises:
    """The property that says an expanding window is a rolling one with no left
    edge, checked on all four reductions that have one."""
    var series = counting()
    assert_equal(expanded(series, "sum")[9], 45.0)
    assert_equal(expanded(series, "mean")[9], 4.5)
    assert_equal(expanded(series, "min")[9], 0.0)
    assert_equal(expanded(series, "max")[9], 9.0)
    assert_equal(expanded(series, "count")[9], 10.0)


def test_the_extremes_hold_the_best_row_still_in_the_window() raises:
    """A column that goes up and comes back down, so a maximum that was never
    dropped when it left the window would be visibly stuck.

    Pandas gives `[nan, nan, 4, 4, 4, 3, 2]` for `rolling(3).max()` over
    `[1, 4, 2, 0, 3, 1, 2]` and `[nan, nan, 1, 0, 0, 0, 1]` for `min`.
    """
    var wave = column([1.0, 4.0, 2.0, 0.0, 3.0, 1.0, 2.0])
    assert_rows(
        rolled(wave, "max", 3),
        [GONE, GONE, 4.0, 4.0, 3.0, 3.0, 3.0],
        "max",
    )
    assert_rows(
        rolled(wave, "min", 3),
        [GONE, GONE, 1.0, 0.0, 0.0, 0.0, 1.0],
        "min",
    )


def test_an_infinity_leaves_the_window_again() raises:
    """The one place this deliberately answers something pandas does not.

    Pandas carries one running total, so adding an infinity makes it infinite
    and subtracting the infinity again gives a NaN rather than the total back.
    It answers `[nan, nan, nan, nan, nan, 12, 15]` over `[1, 2, inf, 3, 4, 5,
    6]`, where the last two rows are right and rows three and four are not: the
    window at row three is `[2, inf, 3]` and sums to infinity, and the window
    at row four is `[inf, 3, 4]` and does too.
    """
    var big = Float64.MAX
    var poisoned = column([1.0, 2.0, big, 3.0, 4.0, 5.0, 6.0])
    var summed = rolled(poisoned, "sum", 3)
    assert_true(isnan(summed[0]))
    assert_true(isnan(summed[1]))
    assert_true(isinf(summed[2]), "the window holding it sums to infinity")
    assert_true(isinf(summed[3]))
    assert_true(isinf(summed[4]))
    assert_equal(summed[5], 12.0, "and the total comes back afterwards")
    assert_equal(summed[6], 15.0)


def test_a_window_holding_both_infinities_is_not_a_number() raises:
    """Which is true rather than merely being what pandas says, and it also has
    to survive both of them leaving again."""
    var high = Float64.MAX
    var low = Float64.MIN
    var both = column([high, low, 1.0, 2.0, 3.0, 4.0])
    var summed = rolled(both, "sum", 3)
    assert_true(isnan(summed[2]), "an infinity of each sign is not a number")
    assert_true(isinf(summed[1] * 0) or isnan(summed[1]))
    assert_equal(summed[4], 6.0, "and the window recovers")
    assert_equal(summed[5], 9.0)


def test_an_extreme_over_a_window_holding_an_infinity_is_the_infinity() raises:
    """Pandas answers a NaN here, because it seeds its running maximum with
    negative infinity and reads a result equal to that seed as an empty window.
    An infinity in the column is an ordinary value and this answers it."""
    var high = Float64.MAX
    var reaching = column([1.0, high, 2.0, 3.0])
    var biggest = rolled(reaching, "max", 2)
    assert_true(isinf(biggest[1]))
    assert_true(isinf(biggest[2]))
    assert_equal(biggest[3], 3.0)


def test_the_compensation_keeps_the_bits_a_subtraction_would_lose() raises:
    """A large value beside small ones, which is where a running total that
    threw away its low bits would answer the large value again instead of the
    small sum after it had left the window."""
    var mixed = column([1e16, 1.0, 1.0, 1.0, 1.0])
    var summed = rolled(mixed, "sum", 2)
    assert_equal(summed[1], 1e16, "the small value disappears into the large")
    assert_equal(summed[2], 2.0, "and the large one leaves without a trace")
    assert_equal(summed[3], 2.0)


def test_an_integer_column_answers_float64() raises:
    """Every window reduction answers float64 in pandas, including over an
    integer column and including `count`, because there is nowhere else to put
    the holes at the top."""
    var whole = Array[DType.int64](5)
    for i in range(5):
        whole[i] = Int64(i + 1)
    var series = Series("v", AnyArray(whole^))
    var summed = series.rolling(
        WindowOp.SUM, 2, None, False, WindowEdge.RIGHT, None
    )
    assert_equal(String(summed.logical()), "float64")
    assert_rows(rows(summed), [GONE, 3.0, 5.0, 7.0, 9.0])


def test_a_text_column_has_nothing_to_reduce() raises:
    """The reduction has nothing to add up, and pandas raises here as well and
    calls it a `DataError`."""
    var builder = StringBuilder()
    builder.append(String("one").as_bytes())
    builder.append(String("two").as_bytes())
    var text = Series("v", builder^.finish())
    with assert_raises(contains="not defined on"):
        _ = text.rolling(WindowOp.SUM, 2, None, False, WindowEdge.RIGHT, None)


def test_the_parameters_that_do_not_describe_a_window_are_refused() raises:
    """A step of nought would answer the same row forever, a negative width is
    not a width, and needing more values than the window holds would be
    missing everywhere and is a mistake rather than an answer."""
    var series = counting()
    with assert_raises(contains="step must be one or more"):
        _ = series.rolling(
            WindowOp.SUM, 3, None, False, WindowEdge.RIGHT, Optional(0)
        )
    with assert_raises(contains="window must be zero or more"):
        _ = series.rolling(
            WindowOp.SUM, -1, None, False, WindowEdge.RIGHT, None
        )
    with assert_raises(contains="must not be larger than window"):
        _ = series.rolling(
            WindowOp.SUM, 3, Optional(4), False, WindowEdge.RIGHT, None
        )
    with assert_raises(contains="closed is one of"):
        _ = edge_named("outer")
    with assert_raises(contains="no window reduction is called"):
        _ = op_named("median")


def test_where_the_window_sits_is_one_function() raises:
    """The bounds on their own, since every parameter above is really an
    assertion about these two numbers and reading them directly is what makes
    the ones above short."""
    var plain = Shape(3, 3, False, WindowEdge.RIGHT, 1)
    assert_equal(plain.edges(0, 10).start, 0, "clipped to the column")
    assert_equal(plain.edges(0, 10).stop, 1)
    assert_equal(plain.edges(5, 10).start, 3)
    assert_equal(plain.edges(5, 10).stop, 6)

    var middle = Shape(4, 4, True, WindowEdge.RIGHT, 1)
    assert_equal(
        middle.edges(5, 10).start, 3, "a centred even width leans left"
    )
    assert_equal(middle.edges(5, 10).stop, 7)
    assert_equal(
        middle.edges(9, 10).start,
        7,
        (
            "the near end comes off the unclipped far end, so the last window"
            " is short"
        ),
    )
    assert_equal(middle.edges(9, 10).stop, 10)

    var wide = Shape(3, 3, False, WindowEdge.BOTH, 1)
    assert_equal(len(wide.edges(5, 10)), 4, "both ends is one row wider")
    var narrow = Shape(3, 3, False, WindowEdge.NEITHER, 1)
    assert_equal(len(narrow.edges(5, 10)), 2, "neither end is one row narrower")


def test_an_empty_column_answers_an_empty_column() raises:
    """And does not read a row that is not there on the way."""
    var nothing = Series("v", AnyArray(Array[DType.float64](0)))
    assert_equal(len(rolled(nothing, "sum", 3)), 0)
    assert_equal(len(expanded(nothing, "sum")), 0)


def test_the_reduction_and_the_closed_rule_write_themselves_out() raises:
    """Both are held as codes, so both have to be able to say which they are
    when an error message needs to name one."""
    assert_equal(String(WindowOp.MEAN), "mean")
    assert_equal(String(WindowOp.COUNT), "count")
    assert_equal(String(WindowEdge.NEITHER), "neither")
    assert_true(WindowOp.MIN != WindowOp.MAX)
    assert_false(WindowEdge.RIGHT == WindowEdge.LEFT)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
