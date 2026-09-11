"""Tests for the window reductions that read a position in the sorted window.

`test_window.mojo` checks where the window sits and `test_spread.mojo` checks
what the folds make of it. This file checks the structure that makes a window
ordered and the three reductions that read out of it, which are the median, the
quantile and the rank. Every number asserted against pandas here was read off a
running pandas 3.0.5 and is quoted in the test that asserts it.

The comparisons are exact rather than near. These answers are a value out of the
column, a weighting of two of them or a count of ranks, so unlike a skewness
there is no rounding to allow for, and asserting to the bit is what catches a
selection that is one off.

The structure is tested directly as well as through the median, because most of
what can go wrong in a Fenwick tree gives a plausible wrong answer rather than a
crash. A tree whose descent starts from the wrong power of two selects the wrong
value only for some widths, and a window that is updated with the wrong run of
rows is right until the first row that leaves, so both are asserted on their own
before any median is taken.
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
from firepanda.bitmap.bitmap import Bitmap
from firepanda.frame.series import Series
from firepanda.kernel.ordered import (
    BETWEEN_HIGHER,
    BETWEEN_LINEAR,
    BETWEEN_LOWER,
    BETWEEN_MIDPOINT,
    BETWEEN_NEAREST,
    TIED_AVERAGE,
    TIED_MAX,
    TIED_MIN,
    Ordered,
    Ranks,
    along,
    between_named,
    halved,
    middle,
    picked,
    placed,
    ranked,
    tied_named,
)
from firepanda.kernel.window import (
    WindowEdge,
    WindowOp,
    WindowSettings,
    op_named,
)

comptime GONE = Float64(0) / Float64(0)
"""A missing row, written the way a float column writes one."""

comptime HUGE = 1.7976931348623157e308
"""The largest finite double, which is where the two ways of halving differ."""


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
    width: Int,
    min_periods: Optional[Int] = None,
    center: Bool = False,
    closed: WindowEdge = WindowEdge.RIGHT,
    step: Optional[Int] = None,
) raises -> List[Float64]:
    """Runs a rolling median and hands the answer back as numbers.

    Args:
        series: The column.
        width: How many rows wide.
        min_periods: How many values a window needs.
        center: Whether the window sits around its row.
        closed: Which of its two ends the window keeps.
        step: How many rows apart the answered rows are.

    Returns:
        The answer.

    Raises:
        Error: Whatever the kernel raises.
    """
    return rows(
        series.rolling(
            WindowOp.MEDIAN, width, min_periods, center, closed, step
        )
    )


def expanded(series: Series, min_periods: Int = 1) raises -> List[Float64]:
    """Runs an expanding median and hands the answer back as numbers.

    Args:
        series: The column.
        min_periods: How many values a window needs.

    Returns:
        The answer.

    Raises:
        Error: Whatever the kernel raises.
    """
    return rows(series.expanding(WindowOp.MEDIAN, min_periods))


def quantiled(
    series: Series,
    width: Int,
    fraction: Float64,
    between: Int = BETWEEN_LINEAR,
    min_periods: Optional[Int] = None,
) raises -> List[Float64]:
    """Runs a rolling quantile and hands the answer back as numbers.

    Args:
        series: The column.
        width: How many rows wide.
        fraction: How far through the sorted window to read.
        between: Which rule to use when the position lands between two values.
        min_periods: How many values a window needs.

    Returns:
        The answer.

    Raises:
        Error: Whatever the kernel raises.
    """
    var settings = WindowSettings()
    settings.fraction = fraction
    settings.between = between
    return rows(
        series.rolling(
            WindowOp.QUANTILE,
            width,
            min_periods,
            False,
            WindowEdge.RIGHT,
            None,
            settings,
        )
    )


def ranking(
    series: Series,
    width: Int,
    tied: Int = TIED_AVERAGE,
    ascending: Bool = True,
    pct: Bool = False,
    center: Bool = False,
    closed: WindowEdge = WindowEdge.RIGHT,
    step: Optional[Int] = None,
    min_periods: Optional[Int] = None,
) raises -> List[Float64]:
    """Runs a rolling rank and hands the answer back as numbers.

    Carries where the window sits as well as the three rules, because a rank
    places the value in the window's last row rather than the one in the row
    being answered, and the two are only different rows once the window has been
    moved off the row it answers.

    Args:
        series: The column.
        width: How many rows wide.
        tied: What to do with values that are equal.
        ascending: Whether to count from the smallest value.
        pct: Whether to divide by how many values the window holds.
        center: Whether the window sits around its row.
        closed: Which of its two ends the window keeps.
        step: How many rows apart the answered rows are.
        min_periods: How many values a window needs.

    Returns:
        The answer.

    Raises:
        Error: Whatever the kernel raises.
    """
    var settings = WindowSettings()
    settings.tied = tied
    settings.ascending = ascending
    settings.pct = pct
    return rows(
        series.rolling(
            WindowOp.RANK, width, min_periods, center, closed, step, settings
        )
    )


def assert_rows(
    got: List[Float64], want: List[Float64], message: String = ""
) raises:
    """Compares two columns of numbers, reading a NaN as a missing row.

    Args:
        got: What was answered.
        want: What is expected.
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


def uneven() -> Series:
    """Returns the column most of these run on.

    Returns:
        A column with two large values in it and a repeated one, chosen so that
        a window that selected the wrong position would answer a different
        number rather than the same one by luck.
    """
    return column([1.0, 2.0, 10.0, 3.0, 1.0, 20.0, 2.0])


def indexed(values: List[Float64]) raises -> Ranks:
    """Ranks a list of values, reading a NaN as a missing row.

    Args:
        values: The values.

    Returns:
        Their distinct values in order and each row's place among them.

    Raises:
        Error: Only what allocation raises.
    """
    var held = Array[DType.float64](len(values))
    var present = Bitmap(len(values))
    for i in range(len(values)):
        held[i] = values[i]
        present.set(i, not isnan(values[i]))
    var wide = AnyArray(held^)
    return ranked(wide.unsafe_ptr[DType.float64](), present, len(values))


def loaded(values: List[Float64]) raises -> Ordered:
    """Builds a tree over a list and puts every one of its values in.

    Args:
        values: The values.

    Returns:
        The tree, holding the whole list.

    Raises:
        Error: Only what allocation raises.
    """
    var order = indexed(values)
    var tree = Ordered(len(order.values))
    for i in range(len(values)):
        if not isnan(values[i]):
            tree.add(Int(order.of_row[i]))
    return tree^


def test_a_rolling_median_matches_what_pandas_answers() raises:
    """Pandas gives `[nan, nan, 2, 3, 3, 3, 2]` for a three wide window over
    `[1, 2, 10, 3, 1, 20, 2]` and `[nan, nan, nan, 2.5, 2.5, 6.5, 2.5]` for a
    four wide one, and the second of those is the one that exercises the
    averaging of the two middle values rather than a plain selection.
    """
    assert_rows(
        rolled(uneven(), 3),
        [GONE, GONE, 2.0, 3.0, 3.0, 3.0, 2.0],
        "three wide",
    )
    assert_rows(
        rolled(uneven(), 4),
        [GONE, GONE, GONE, 2.5, 2.5, 6.5, 2.5],
        "four wide",
    )


def test_an_expanding_median_walks_up_the_whole_column() raises:
    """Pandas gives `[1, 1.5, 2, 2.5, 2, 2.5, 2]`, which is the one shape where
    the window only ever grows, so every row of it is a selection out of a tree
    that has never had anything dropped from it.
    """
    assert_rows(
        expanded(uneven()),
        [1.0, 1.5, 2.0, 2.5, 2.0, 2.5, 2.0],
        "expanding",
    )


def test_a_window_of_one_is_the_column_itself() raises:
    """Pandas gives the column back, which is the smallest check that the
    selection reads the row it was asked for rather than the first row of the
    tree.
    """
    assert_rows(
        rolled(uneven(), 1),
        [1.0, 2.0, 10.0, 3.0, 1.0, 20.0, 2.0],
        "one wide",
    )


def test_a_centred_median_sits_around_its_row() raises:
    """Pandas gives `[nan, 2, 3, 3, 3, 2, nan]`, which is the three wide answer
    moved up by one and losing its last row, because a centred window at the
    bottom of the column gets shorter rather than sliding back up.
    """
    assert_rows(
        rolled(uneven(), 3, None, True),
        [GONE, 2.0, 3.0, 3.0, 3.0, 2.0, GONE],
        "centred",
    )


def test_the_closed_rules_move_the_ends_under_a_median_too() raises:
    """Pandas gives `[nan, nan, nan, 2, 3, 3, 3]` for `left`, which is the
    three wide answer moved down by one, `[nan, nan, 2, 2.5, 2.5, 6.5, 2.5]`
    for `both`, where the window holds four rows and so averages two of them,
    and nothing at all for `neither`, where it holds two and three were asked
    for.
    """
    assert_rows(
        rolled(uneven(), 3, None, False, WindowEdge.LEFT),
        [GONE, GONE, GONE, 2.0, 3.0, 3.0, 3.0],
        "left",
    )
    assert_rows(
        rolled(uneven(), 3, None, False, WindowEdge.BOTH),
        [GONE, GONE, 2.0, 2.5, 2.5, 6.5, 2.5],
        "both",
    )
    assert_rows(
        rolled(uneven(), 3, None, False, WindowEdge.NEITHER),
        [GONE, GONE, GONE, GONE, GONE, GONE, GONE],
        "neither",
    )


def test_a_step_wider_than_the_window_leaves_nothing_behind() raises:
    """Pandas gives `[nan, nan, 2.5, 2.5]` for a four wide window stepped two
    rows at a time, where consecutive windows still overlap, and `[1, 2]` for a
    two wide window stepped four, where they do not. The second is the case the
    loop is written for. A tree that was cleared between two windows that share
    nothing would cost the whole column's distinct values on that row, and a
    tree that assumed the two overlapped would keep carrying rows that had
    already left.
    """
    assert_rows(
        rolled(uneven(), 4, None, False, WindowEdge.RIGHT, Optional(2)),
        [GONE, GONE, 2.5, 2.5],
        "stepped",
    )
    assert_rows(
        rolled(uneven(), 2, Optional(1), False, WindowEdge.RIGHT, Optional(4)),
        [1.0, 2.0],
        "stepped clear of the last window",
    )


def test_a_median_steps_over_the_missing_rows() raises:
    """Pandas gives `[1, 1, 2, 4, 4, 6, 8, 9]` for a three wide window over
    `[1, gone, 3, 5, gone, 7, 9, 11]` asking for one value, so a window of
    three rows holding two of them answers the mean of those two and a window
    holding one answers it.
    """
    var holed = column([1.0, GONE, 3.0, 5.0, GONE, 7.0, 9.0, 11.0])
    assert_rows(
        rolled(holed, 3, Optional(1)),
        [1.0, 1.0, 2.0, 4.0, 4.0, 6.0, 8.0, 9.0],
        "one value is enough",
    )
    assert_rows(
        rolled(holed, 3),
        [GONE, GONE, GONE, GONE, GONE, GONE, GONE, 9.0],
        "three values are asked for and only the last window has them",
    )
    assert_rows(
        expanded(holed),
        [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0],
        "expanding over the holes",
    )


def test_a_median_over_a_repeated_value_is_that_value() raises:
    """Pandas gives `[nan, nan, 2, 2, 2, 2]` over six twos, which is worth
    asserting because the column has one distinct value and so the tree has one
    rank and one entry, which is the width where a descent that started from
    the wrong power of two would still be right.
    """
    assert_rows(
        rolled(column([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]), 3),
        [GONE, GONE, 2.0, 2.0, 2.0, 2.0],
        "one distinct value",
    )


def test_an_integer_column_answers_float64() raises:
    """Pandas gives `[nan, 4, 6, 5, 4]` for a two wide window over
    `[5, 3, 9, 1, 7]` held as int64, because every window reduction answers
    float64 whatever it was given.
    """
    var whole = Array[DType.int64](5)
    whole[0] = 5
    whole[1] = 3
    whole[2] = 9
    whole[3] = 1
    whole[4] = 7
    var series = Series("v", AnyArray(whole^))
    assert_rows(
        rolled(series, 2), [GONE, 4.0, 6.0, 5.0, 4.0], "int64 in, float64 out"
    )


def test_a_median_over_a_window_holding_an_infinity_is_a_value() raises:
    """Pandas answers `[1, 1, 1.5, 2.5, 3]` over `[1, inf, 2, 3, 4]` asking for
    one value, because it replaces every infinity in the column with a missing
    row before any window is formed. Here an infinity is a value and it sorts
    above every finite one, so the window holding it has two values and not
    one, and rows one and two are `inf` and `2` rather than `1` and `1.5`.
    """
    var reaching = column([1.0, Float64.MAX * 2.0, 2.0, 3.0, 4.0])
    var got = rolled(reaching, 3, Optional(1))
    assert_equal(got[0], 1.0, "one row, one value")
    assert_true(isinf(got[1]), "the middle of one and an infinity is neither")
    assert_equal(got[2], 2.0, "the middle of one, an infinity and two is two")
    assert_equal(got[3], 3.0, "the middle of an infinity, two and three")
    assert_equal(got[4], 3.0, "the middle of two, three and four")


def test_two_huge_values_have_a_median_that_pandas_overflows() raises:
    """A window holding the largest finite double twice has that double as its
    median, because the mean of a value and itself is that value. pandas sums
    the two and halves the sum, the sum is an infinity, and it answers `inf`.
    The rows either side of that one agree to the bit, which is the point of
    halving the sum rather than each value everywhere else.
    """
    var enormous = column([HUGE, HUGE, 1.0, 2.0])
    var got = rolled(enormous, 2)
    assert_true(isnan(got[0]), "one row and two were asked for")
    assert_equal(got[1], HUGE, "pandas answers inf here and this answers HUGE")
    assert_equal(got[2], 8.988465674311579e307, "pandas agrees to the bit")
    assert_equal(got[3], 1.5, "and here")
    assert_equal(
        middle(-HUGE, HUGE), 0.0, "the two signs cancel rather than overflowing"
    )
    assert_true(
        isinf(middle(Float64.MAX * 2.0, HUGE)),
        "an infinity in the window is still an infinity out of it",
    )


def test_an_empty_column_answers_an_empty_column() raises:
    """Nothing to rank, nothing to select, and a tree of no ranks at all, which
    is the width where the descent has no power of two to start from.
    """
    assert_rows(rolled(column([]), 3), [], "empty")
    assert_rows(rolled(column([]), 3, Optional(0)), [], "empty asking for none")


def test_a_text_column_has_nothing_to_reduce() raises:
    """A median needs values it can order as numbers, and the refusal names the
    reduction and the type rather than failing inside the tree.
    """
    var builder = StringBuilder()
    builder.append(String("one").as_bytes())
    builder.append(String("two").as_bytes())
    var text = Series("v", builder^.finish())
    with assert_raises(contains="median is not defined on"):
        _ = rolled(text, 2)


def test_the_median_knows_it_is_one_of_the_order_statistics() raises:
    """The three predicates on a reduction partition the thirteen, and three
    of them answer yes to the third, because they read a position in a sorted
    window rather than folding the window into a number.
    """
    assert_true(WindowOp.MEDIAN.orders(), "the median is an order statistic")
    assert_true(WindowOp.QUANTILE.orders(), "so is the quantile")
    assert_true(WindowOp.RANK.orders(), "and so is the rank")
    assert_false(WindowOp.MEDIAN.spreads(), "and is not a spread")
    assert_false(WindowOp.MEDIAN.shapes(), "and is not a shape")
    assert_false(WindowOp.QUANTILE.spreads(), "nor is the quantile")
    assert_false(WindowOp.RANK.shapes(), "nor is the rank")
    assert_false(WindowOp.KURT.orders(), "the kurtosis is still a shape")
    assert_false(WindowOp.SUM.orders(), "and the total is still a fold")
    assert_equal(String(WindowOp.MEDIAN), "median", "writes itself out")
    assert_equal(String(WindowOp.QUANTILE), "quantile", "and so do these two")
    assert_equal(String(WindowOp.RANK), "rank", "and so do these two")
    assert_true(op_named("median") == WindowOp.MEDIAN, "and is named")
    assert_true(op_named("quantile") == WindowOp.QUANTILE, "and is named")
    assert_true(op_named("rank") == WindowOp.RANK, "and is named")


def test_the_ranks_collapse_the_repeats_and_skip_the_missing_rows() raises:
    """Two rows holding the same value get the same rank and a missing row gets
    none, which is what lets a tie be counted in one load and what keeps a
    column of one repeated value costing one entry rather than one per row.
    """
    var values: List[Float64] = [5.0, 1.0, 5.0, GONE, 3.0, 1.0]
    var order = indexed(values)
    assert_equal(len(order.values), 3, "one, three and five")
    assert_equal(order.values[0], 1.0, "smallest first")
    assert_equal(order.values[2], 5.0, "largest last")
    assert_equal(Int(order.of_row[0]), 2, "five is the third rank")
    assert_equal(Int(order.of_row[1]), 0, "one is the first")
    assert_equal(Int(order.of_row[2]), 2, "and the second five is the same")
    assert_equal(Int(order.of_row[3]), -1, "a missing row has no rank")
    assert_equal(Int(order.of_row[5]), 0, "and the second one is the first")


def test_the_tree_counts_below_and_at_a_rank_and_selects_out_of_it() raises:
    """The three questions a window asks of the structure, asserted on a tree
    that is filled by hand rather than by a window, so that a wrong answer here
    cannot be blamed on the edges.
    """
    var values: List[Float64] = [2.0, 2.0, 2.0, 5.0, 2.0, 2.0, 5.0, 5.0]
    var tree = loaded(values)
    assert_equal(tree.below(0), 0, "nothing is below the smallest rank")
    assert_equal(tree.same(0), 5, "five twos")
    assert_equal(tree.below(1), 5, "and they are all below the fives")
    assert_equal(tree.same(1), 3, "three fives")
    assert_equal(tree.select(0), 0, "the smallest is a two")
    assert_equal(tree.select(4), 0, "and so is the fifth")
    assert_equal(tree.select(5), 1, "the sixth is a five")
    assert_equal(tree.select(7), 1, "and so is the largest")


def test_dropping_from_the_tree_undoes_adding_to_it_exactly() raises:
    """A count is exact however long the pass runs, which is the one thing
    these three have that the folds do not, so a tree that has had a value put
    in and taken out again is the tree it was before.
    """
    var values: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var tree = loaded(values)
    assert_equal(tree.select(1), 1, "two is the second smallest")
    tree.drop(0)
    assert_equal(tree.below(1), 0, "one has left the window")
    assert_equal(tree.select(0), 1, "so two is now the smallest")
    tree.add(0)
    assert_equal(tree.below(1), 1, "and it is back")
    assert_equal(tree.select(0), 0, "so one is the smallest again")
    for i in range(4):
        tree.drop(i)
    assert_equal(tree.below(3), 0, "an emptied tree counts nothing")
    assert_equal(tree.same(2), 0, "at any rank")


def test_the_five_rules_for_reading_between_two_values() raises:
    """Pandas over `[10, 20, 30, 40, 50]` gives 15, 10, 20, 15 and 10 for the
    five at a quantile of an eighth, and 25, 20, 30, 25 and 30 at three
    eighths. The second row is what pins `nearest` as rounding a half to the
    even side rather than up, since the position is two and a half there and
    the answer is the third value and not the fourth.
    """
    var values: List[Float64] = [10.0, 20.0, 30.0, 40.0, 50.0]
    var tree = loaded(values)
    var order = indexed(values)
    assert_equal(picked(tree, order, 5, 0.125, BETWEEN_LINEAR), 15.0, "linear")
    assert_equal(picked(tree, order, 5, 0.125, BETWEEN_LOWER), 10.0, "lower")
    assert_equal(picked(tree, order, 5, 0.125, BETWEEN_HIGHER), 20.0, "higher")
    assert_equal(
        picked(tree, order, 5, 0.125, BETWEEN_MIDPOINT), 15.0, "midpoint"
    )
    assert_equal(
        picked(tree, order, 5, 0.125, BETWEEN_NEAREST), 10.0, "nearest is down"
    )
    assert_equal(
        picked(tree, order, 5, 0.375, BETWEEN_NEAREST),
        30.0,
        "and here it is up",
    )
    assert_equal(picked(tree, order, 5, 0.375, BETWEEN_LINEAR), 25.0, "linear")
    assert_equal(
        picked(tree, order, 5, 0.25, BETWEEN_LOWER), 20.0, "on a value"
    )
    assert_equal(
        picked(tree, order, 5, 0.25, BETWEEN_HIGHER),
        20.0,
        "so all five agree there",
    )
    assert_equal(picked(tree, order, 5, 0.0, BETWEEN_HIGHER), 10.0, "the first")
    assert_equal(picked(tree, order, 5, 1.0, BETWEEN_LOWER), 50.0, "the last")
    assert_equal(halved(tree, order, 5), 30.0, "and the middle")


def test_the_three_tie_rules_and_the_two_directions() raises:
    """Pandas over `[2, 2, 2, 5]` ranked ascending gives 2 for a two under the
    three rules as 2, 1 and 3, and 4, 4 and 4 for the five. Descending it gives
    3, 2 and 4 for a two and 1, 1 and 1 for the five, and a percentage divides
    by how many values the window holds rather than by how wide it is.
    """
    var values: List[Float64] = [2.0, 2.0, 2.0, 5.0]
    var tree = loaded(values)
    assert_equal(placed(tree, 0, 4, TIED_AVERAGE, True, False), 2.0, "average")
    assert_equal(placed(tree, 0, 4, TIED_MIN, True, False), 1.0, "min")
    assert_equal(placed(tree, 0, 4, TIED_MAX, True, False), 3.0, "max")
    assert_equal(placed(tree, 1, 4, TIED_AVERAGE, True, False), 4.0, "the five")
    assert_equal(placed(tree, 1, 4, TIED_MIN, True, False), 4.0, "alone")
    assert_equal(
        placed(tree, 0, 4, TIED_AVERAGE, False, False), 3.0, "descending"
    )
    assert_equal(placed(tree, 0, 4, TIED_MIN, False, False), 2.0, "descending")
    assert_equal(placed(tree, 0, 4, TIED_MAX, False, False), 4.0, "descending")
    assert_equal(
        placed(tree, 1, 4, TIED_AVERAGE, False, False), 1.0, "the five is first"
    )
    assert_equal(
        placed(tree, 0, 4, TIED_AVERAGE, True, True), 0.5, "as a fraction"
    )
    assert_equal(
        placed(tree, 1, 4, TIED_AVERAGE, False, True), 0.25, "and descending"
    )


def test_the_two_words_that_name_a_rule_are_checked() raises:
    """Both of these are what a caller typed, so the refusal names the word it
    was given and lists the ones it would have taken.
    """
    assert_equal(between_named("linear"), BETWEEN_LINEAR, "linear")
    assert_equal(between_named("nearest"), BETWEEN_NEAREST, "nearest")
    assert_equal(tied_named("average"), TIED_AVERAGE, "average")
    assert_equal(tied_named("max"), TIED_MAX, "max")
    with assert_raises(contains="interpolation 'cubic' is not one of"):
        _ = between_named("cubic")
    with assert_raises(contains="rank method 'dense' is not one of"):
        _ = tied_named("dense")


def test_a_rolling_quantile_matches_what_pandas_answers() raises:
    """Pandas over `[1, 2, 10, 3, 1, 20, 2]` with `rolling(4).quantile(0.75)`
    gives `[nan, nan, nan, 4.75, 4.75, 12.5, 7.25]` on the linear rule, and
    `[3, 3, 10, 3]`, `[10, 10, 20, 20]`, `[6.5, 6.5, 15, 11.5]` and
    `[3, 3, 10, 3]` in the four answered rows for the other four. `nearest` and
    `lower` agree here and come apart in the test of the five rules below.
    """
    var series = column([1.0, 2.0, 10.0, 3.0, 1.0, 20.0, 2.0])
    assert_rows(
        quantiled(series, 4, 0.75),
        [GONE, GONE, GONE, 4.75, 4.75, 12.5, 7.25],
        "linear",
    )
    assert_rows(
        quantiled(series, 4, 0.75, BETWEEN_LOWER),
        [GONE, GONE, GONE, 3.0, 3.0, 10.0, 3.0],
        "lower",
    )
    assert_rows(
        quantiled(series, 4, 0.75, BETWEEN_HIGHER),
        [GONE, GONE, GONE, 10.0, 10.0, 20.0, 20.0],
        "higher",
    )
    assert_rows(
        quantiled(series, 4, 0.75, BETWEEN_MIDPOINT),
        [GONE, GONE, GONE, 6.5, 6.5, 15.0, 11.5],
        "midpoint",
    )
    assert_rows(
        quantiled(series, 4, 0.75, BETWEEN_NEAREST),
        [GONE, GONE, GONE, 3.0, 3.0, 10.0, 3.0],
        "nearest",
    )


def test_the_two_ends_of_a_quantile_are_the_two_extremes() raises:
    """Pandas gives `[1, 1, 1, 1]` for `quantile(0)` and `[10, 10, 20, 20]` for
    `quantile(1)` over the same column, which are the same four rows `min` and
    `max` answer. Worth asserting because a position of nought and a position of
    the last rank are where an off by one in the descent shows up.
    """
    var series = column([1.0, 2.0, 10.0, 3.0, 1.0, 20.0, 2.0])
    assert_rows(
        quantiled(series, 4, 0.0),
        [GONE, GONE, GONE, 1.0, 1.0, 1.0, 1.0],
        "the smallest",
    )
    assert_rows(
        quantiled(series, 4, 1.0),
        [GONE, GONE, GONE, 10.0, 10.0, 20.0, 20.0],
        "the largest",
    )


def test_a_quantile_of_a_half_is_the_median() raises:
    """Pandas gives the same column for both, and so does this, by two different
    routes: the median reads the middle of the count of ranks and the quantile
    computes a position and reads that. They come apart only where the gap
    between the two middle values overflows, which the test below owns.
    """
    var series = column([1.0, 2.0, 10.0, 3.0, 1.0, 20.0, 2.0])
    assert_rows(
        quantiled(series, 4, 0.5),
        [GONE, GONE, GONE, 2.5, 2.5, 6.5, 2.5],
        "pandas answers this for both",
    )
    assert_rows(quantiled(series, 4, 0.5), rolled(series, 4), "and so do we")
    var holed = column([1.0, GONE, 3.0, GONE, 5.0, 6.0])
    assert_rows(
        quantiled(holed, 3, 0.5, BETWEEN_LINEAR, 1),
        rolled(holed, 3, 1),
        "over gaps too",
    )


def test_an_expanding_quantile_walks_up_the_whole_column() raises:
    """Pandas gives `[1, 1.5, 2, 2.5, 2, 2.5, 2]` for the same column under
    `expanding().quantile(0.5)`, which is its expanding median, and the last row
    is the quantile of the whole column.
    """
    var series = column([1.0, 2.0, 10.0, 3.0, 1.0, 20.0, 2.0])
    var settings = WindowSettings()
    var got = rows(series.expanding(WindowOp.QUANTILE, 1, settings))
    assert_rows(got, [1.0, 1.5, 2.0, 2.5, 2.0, 2.5, 2.0], "the whole column")
    settings.fraction = 1.0
    var largest = rows(series.expanding(WindowOp.QUANTILE, 1, settings))
    assert_equal(largest[6], 20.0, "and the top of it")


def test_a_quantile_between_two_huge_values_does_not_overflow() raises:
    """Pandas writes the linear rule as the lower value plus the fraction of the
    gap, so over `[-HUGE, HUGE, HUGE, HUGE]` with `rolling(2).quantile(0.5)` it
    gives `[nan, inf, HUGE, HUGE]`: the gap between the two infinities of sign
    overflows and the fraction of an infinity is an infinity. Its median of the
    same window is nought, so pandas answers the two of them inconsistently, and
    the same pair the other way round has an exact quantile and an infinite
    median. Both are the same overflow from two sides and neither happens here.
    """
    var series = column([-HUGE, HUGE, HUGE, HUGE])
    var got = quantiled(series, 2, 0.5)
    assert_true(isnan(got[0]), "one value is not two")
    assert_equal(got[1], 0.0, "where pandas answers an infinity")
    assert_equal(got[2], HUGE, "and the rest agree with pandas")
    assert_equal(got[3], HUGE)
    assert_rows(rolled(series, 2), got, "and the median agrees with all of it")
    assert_equal(along(-HUGE, HUGE, 0.5), 0.0, "the weighting is used here")
    assert_equal(along(HUGE, HUGE, 0.5), HUGE, "and here")
    assert_equal(
        along(1.0, 3.0, 0.5), 2.0, "and the plain rule everywhere else"
    )
    assert_equal(along(1.0, 3.0, 0.25), 1.5, "to the bit")
    assert_equal(middle(-HUGE, HUGE), 0.0, "which is what middle does too")


def test_a_rolling_rank_places_the_value_in_the_window_last_row() raises:
    """Pandas over `[1, 2, 10, 3, 1, 20, 2]` with `rolling(3).rank()` gives
    `[nan, nan, 3, 2, 1, 3, 2]`, descending gives `[nan, nan, 1, 2, 3, 1, 2]`
    and a percentage gives `[nan, nan, 1, 2/3, 1/3, 1, 2/3]`. The value being
    placed is the one in the window's last row, which here is the row being
    answered, and the test below moves the window so the two come apart.
    """
    var series = column([1.0, 2.0, 10.0, 3.0, 1.0, 20.0, 2.0])
    assert_rows(
        ranking(series, 3),
        [GONE, GONE, 3.0, 2.0, 1.0, 3.0, 2.0],
        "ascending",
    )
    assert_rows(
        ranking(series, 3, TIED_AVERAGE, False),
        [GONE, GONE, 1.0, 2.0, 3.0, 1.0, 2.0],
        "descending",
    )
    assert_rows(
        ranking(series, 3, TIED_AVERAGE, True, True),
        [GONE, GONE, 1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 2.0 / 3.0],
        "as a fraction of the count",
    )


def test_a_rank_reads_the_last_row_and_not_the_answered_row() raises:
    """Pandas gives `[nan, 3, 2, 1, 3, 2, nan]` for `rolling(3, center=True)`,
    `[nan, nan, nan, 3, 2, 1, 3]` for `rolling(3, closed='left')` and
    `[nan, nan, 1, 2]` for `rolling(4, step=2)` over the same column. Each of
    the three moves the window off the row it answers, and each of the three
    answers the rank of the value at the window's near end rather than the rank
    of the value in the row the answer is written to.
    """
    var series = column([1.0, 2.0, 10.0, 3.0, 1.0, 20.0, 2.0])
    assert_rows(
        ranking(series, 3, TIED_AVERAGE, True, False, True),
        [GONE, 3.0, 2.0, 1.0, 3.0, 2.0, GONE],
        "centred",
    )
    assert_rows(
        ranking(series, 3, TIED_AVERAGE, True, False, False, WindowEdge.LEFT),
        [GONE, GONE, GONE, 3.0, 2.0, 1.0, 3.0],
        "closed on the left",
    )
    assert_rows(
        ranking(
            series, 4, TIED_AVERAGE, True, False, False, WindowEdge.RIGHT, 2
        ),
        [GONE, GONE, 1.0, 2.0],
        "stepped",
    )


def test_a_rank_of_a_row_holding_nothing_is_missing() raises:
    """Pandas gives `[1, nan, 2, 2, nan, 2, 2, 3]` for `rolling(3,
    min_periods=1).rank()` over `[1, None, 3, 5, None, 7, 9, 11]`. There is no
    value to place in the rows that hold nothing, so there is no rank there
    however many values the window holds, and `min_periods` has nothing to say
    about it.
    """
    var series = column([1.0, GONE, 3.0, 5.0, GONE, 7.0, 9.0, 11.0])
    assert_rows(
        ranking(
            series,
            3,
            TIED_AVERAGE,
            True,
            False,
            False,
            WindowEdge.RIGHT,
            None,
            1,
        ),
        [1.0, GONE, 2.0, 2.0, GONE, 2.0, 2.0, 3.0],
        "a missing row has nothing to place",
    )


def test_the_settings_default_to_what_pandas_defaults_to() raises:
    """Five of the thirteen reductions read something out of the settings and
    the other eight read nothing, which is only safe if every field has a value
    whichever reduction is running. Five of the six are pandas' own defaults,
    and the fraction is a half because pandas makes the fraction required and
    there has to be a number in the field whatever is running.
    """
    var settings = WindowSettings()
    assert_equal(settings.ddof, 1, "pandas' degrees of freedom")
    assert_equal(
        settings.fraction, 0.5, "the middle, which pandas has no name for"
    )
    assert_equal(settings.between, BETWEEN_LINEAR, "pandas' interpolation")
    assert_equal(settings.tied, TIED_AVERAGE, "pandas' rank method")
    assert_true(settings.ascending, "pandas counts up")
    assert_false(settings.pct, "and does not divide")


def main() raises:
    """Runs the suite.

    Raises:
        Error: If any test fails.
    """
    TestSuite.discover_tests[__functions_in_module()]().run()
