"""Tests for the five window reductions that measure a spread or a shape.

`test_window.mojo` checks where the window sits and this file checks what comes
out of it, because the two fail in different ways and are worth reading apart.
Every number here was read off a running pandas 3.0.5 and is quoted in the test
that asserts it.

The exceptions are the families where pandas is measurably wrong, which are a
window holding an infinity, a window whose sum of squared deviations has
overflowed, and a window whose variance is small enough that pandas refuses to
give it a shape at all. Those are checked against what is true and the tests say
so and say why. `firepanda/kernel/spread.mojo` carries the argument.

The spread tests compare exactly and the shape tests do not, and the reason is
in the columns rather than in the code. Every column the spreads are asserted
on was chosen so that the right answer is one a double holds exactly. A
skewness is a cube over a three halves power and there is no column of more
than two rows where that lands on a double, so those are compared to within
`NEARLY` and the two exact answers pandas states for a constant window are the
only ones asserted to the bit.
"""

from std.math import isinf, isnan, sqrt
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.frame.series import Series
from firepanda.kernel.spread import (
    MOMENT_KURT,
    MOMENT_SKEW,
    SPREAD_SEM,
    SPREAD_STD,
    SPREAD_VAR,
    Moments,
    Spread,
)
from firepanda.kernel.window import WindowEdge, WindowOp, op_named

comptime GONE = Float64(0) / Float64(0)
"""A missing row, written the way a float column writes one."""

comptime NEARLY = 1e-12
"""How far a shape is allowed to sit from the figure pandas answered.

pandas reconstructs a third and a fourth central moment from raw sums of powers
and this library carries the moments themselves, so the two disagree in the
last digit or two of every shape. The largest gap on any column in this file is
in the sixteenth digit, so this is four figures looser than anything measured,
which leaves it able to catch a real regression and unable to fail because a
compiler reordered two multiplications."""


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
    kind: StringSlice,
    width: Int,
    ddof: Int = 1,
    min_periods: Optional[Int] = None,
) raises -> List[Float64]:
    """Runs a rolling spread and hands the answer back as numbers.

    Args:
        series: The column.
        kind: One of `var`, `std` and `sem`.
        width: How many rows wide.
        ddof: Subtracted from the count to give the divisor.
        min_periods: How many values a window needs.

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
            False,
            WindowEdge.RIGHT,
            None,
            ddof,
        )
    )


def expanded(
    series: Series, kind: StringSlice, ddof: Int = 1, min_periods: Int = 1
) raises -> List[Float64]:
    """Runs an expanding spread and hands the answer back as numbers.

    Args:
        series: The column.
        kind: One of `var`, `std` and `sem`.
        ddof: Subtracted from the count to give the divisor.
        min_periods: How many values a window needs.

    Returns:
        The answer.

    Raises:
        Error: Whatever the kernel raises.
    """
    return rows(series.expanding(op_named(kind), min_periods, ddof))


def assert_rows(
    got: List[Float64], want: List[Float64], message: String = ""
) raises:
    """Compares two columns of numbers, reading a NaN as a missing row.

    The comparison is exact, which is deliberate. Every column asserted here was
    chosen so that the right answer is one a double holds exactly, and the point
    of the file is that the accumulation does not lose it.

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


def assert_near(
    got: List[Float64], want: List[Float64], message: String = ""
) raises:
    """Compares two columns of numbers to within `NEARLY`, reading a NaN as a
    missing row.

    Args:
        got: What was answered.
        want: What pandas answered.
        message: What the caller is asserting.

    Raises:
        Error: If the two are different heights, or one has a value where the
            other has a missing row, or any pair is further apart than `NEARLY`
            relative to the larger of the two.
    """
    assert_equal(len(got), len(want), "height: " + message)
    for i in range(len(want)):
        var at = " row " + String(i) + ": " + message
        if isnan(want[i]):
            assert_true(isnan(got[i]), "should be missing:" + at)
            continue
        assert_false(isnan(got[i]), "should not be missing:" + at)
        var scale = max(max(abs(got[i]), abs(want[i])), 1.0)
        assert_true(
            abs(got[i] - want[i]) <= NEARLY * scale,
            String(got[i]) + " is not near " + String(want[i]) + ":" + at,
        )


def counting() -> Series:
    """Returns one to five, which most of these run on.

    Returns:
        The series.
    """
    return column([1.0, 2.0, 3.0, 4.0, 5.0])


def test_a_rolling_variance_over_consecutive_rows_is_the_same_every_row() raises:
    """Pandas gives `[nan, nan, 1, 1, 1]` for a three wide window over one to
    five, which it should, because every window holds three rows a step apart
    and the variance does not care where they sit.
    """
    assert_rows(rolled(counting(), "var", 3), [GONE, GONE, 1.0, 1.0, 1.0])


def test_the_degrees_of_freedom_choose_the_divisor() raises:
    """Pandas gives two thirds for `ddof=0`, one for the default, two for
    `ddof=2` and a half for `ddof=-1`, which are the sum of squared deviations
    over three, two, one and four.

    A negative one is not a mistake on the caller's part, or at least pandas
    does not treat it as one, and it is the shortest proof that the count and
    the divisor are two different numbers.
    """
    var third = 0.6666666666666666
    assert_rows(
        rolled(counting(), "var", 3, ddof=0),
        [GONE, GONE, third, third, third],
        "ddof=0",
    )
    assert_rows(
        rolled(counting(), "var", 3, ddof=2), [GONE, GONE, 2.0, 2.0, 2.0], "2"
    )
    assert_rows(
        rolled(counting(), "var", 3, ddof=-1), [GONE, GONE, 0.5, 0.5, 0.5], "-1"
    )


def test_a_window_with_no_degrees_of_freedom_left_is_missing() raises:
    """Pandas gives a column of nulls for `rolling(3).var(ddof=3)` and for
    `rolling(1).var()`, because a variance divided by nought is not a large
    number, it is a question with no answer. The rule is that the count has to
    be more than the degrees of freedom and not merely as many.
    """
    var starved = rolled(counting(), "var", 3, ddof=3)
    for i in range(len(starved)):
        assert_true(isnan(starved[i]), "ddof=3 over three rows")
    var alone = rolled(counting(), "var", 1)
    for i in range(len(alone)):
        assert_true(isnan(alone[i]), "one row and one degree of freedom")


def test_a_window_of_one_row_has_no_spread_at_all() raises:
    """Pandas gives `[0, 0, 0, 0, 0]` for `rolling(1).var(ddof=0)`. This is the
    exact answer rather than a small one, and it is exact here because it is
    read off the count rather than out of the accumulation.
    """
    assert_rows(rolled(counting(), "var", 1, ddof=0), [0.0, 0.0, 0.0, 0.0, 0.0])


def test_the_deviation_is_the_root_of_the_variance() raises:
    """Pandas gives `[nan, nan, 1, 1, 1]` for the three wide deviation, and the
    expanding one is `[nan, 0.7071067811865476, 1, 1.2909944487358056,
    1.5811388300841898]`, which is the root of `[nan, 0.5, 1,
    1.6666666666666667, 2.5]` row for row.
    """
    assert_rows(rolled(counting(), "std", 3), [GONE, GONE, 1.0, 1.0, 1.0])
    var spread = expanded(counting(), "std")
    var square = expanded(counting(), "var")
    assert_true(isnan(spread[0]), "one row and one degree of freedom")
    for i in range(1, 5):
        assert_equal(spread[i], sqrt(square[i]), "row " + String(i))
    assert_equal(spread[1], 0.7071067811865476)
    assert_equal(spread[3], 1.2909944487358056)
    assert_equal(spread[4], 1.5811388300841898)


def test_the_standard_error_divides_the_deviation_by_the_root_count() raises:
    """Pandas gives `0.5773502691896258` for the three wide standard error and
    `0.47140452079103173` for the same with `ddof=0`, and those are one and two
    thirds' root over the root of three. The count under that root is the plain
    count and is not moved by the degrees of freedom, which is the one thing
    about `sem` worth a test, and it is what pandas does because pandas writes
    `sem` as `std(ddof) / count ** 0.5`.
    """
    var expected = 0.5773502691896258
    assert_rows(
        rolled(counting(), "sem", 3),
        [GONE, GONE, expected, expected, expected],
        "default",
    )
    var lower = 0.47140452079103173
    assert_rows(
        rolled(counting(), "sem", 3, ddof=0),
        [GONE, GONE, lower, lower, lower],
        "ddof=0",
    )
    assert_equal(expected, 1.0 / sqrt(3.0), "the deviation is one")
    assert_rows(
        expanded(counting(), "sem"),
        [GONE, 0.5, expected, 0.6454972243679028, 0.7071067811865476],
        "expanding",
    )


def test_an_expanding_variance_ends_at_the_whole_column() raises:
    """Pandas gives `[nan, 0.5, 1, 1.6666666666666667, 2.5]`, and the last row
    is the variance of one to five, which is ten over four. Every expanding
    reduction has to end at the plain reduction of the whole column and this one
    is the one where an accumulation could quietly drift on the way.
    """
    assert_rows(
        expanded(counting(), "var"),
        [GONE, 0.5, 1.0, 1.6666666666666667, 2.5],
    )


def test_a_window_of_one_repeated_value_is_exactly_nought() raises:
    """Pandas gives exact zeros for six rows of two, and so does this, and
    neither of them gets there by dividing the accumulated deviations, because
    those would land a rounding either side of nought and a rounding is very
    visible next to a nought.
    """
    var flat = column([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    assert_rows(
        rolled(flat, "var", 3), [GONE, GONE, 0.0, 0.0, 0.0, 0.0], "variance"
    )
    assert_rows(
        rolled(flat, "std", 3), [GONE, GONE, 0.0, 0.0, 0.0, 0.0], "deviation"
    )
    assert_rows(
        rolled(flat, "sem", 3), [GONE, GONE, 0.0, 0.0, 0.0, 0.0], "error"
    )


def test_a_run_that_ends_stops_answering_nought() raises:
    """Pandas gives `[nan, nan, 0.3333333333333333, 0, 0,
    0.33333333333333326]` for a three wide variance over one, four twos and a
    three, so the two windows holding nothing but twos are exactly nought and
    the ones either side of them are not.

    The last row is one ulp below a third, in pandas and here, and a third is
    what is true. One pass over two, two and three lands there whichever way the
    pass is arranged, so it is not drift that a rebuild would fix, and it is the
    whole reason the conformance suite compares these under a tolerance rather
    than exactly.
    """
    var mixed = column([1.0, 2.0, 2.0, 2.0, 2.0, 3.0])
    assert_rows(
        rolled(mixed, "var", 3),
        [
            GONE,
            GONE,
            0.3333333333333333,
            0.0,
            0.0,
            0.33333333333333326,
        ],
    )


def test_a_large_value_leaving_the_window_does_not_take_the_answer_with_it() raises:
    """Pandas gives `[nan, nan, 3333333233333334, 1, 1, 1, 1]` for a three wide
    variance over ten to the eight and then one to six.

    This is the test the error bound exists for. The window holding the large
    value has almost all of its squared deviation there, and subtracting that
    value back out leaves the small remainder as the difference of two large
    numbers, where carrying the state answers three quarters instead of one. The
    bound notices and the window is rebuilt from its own three rows.
    """
    var looming = column([1e8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert_rows(
        rolled(looming, "var", 3),
        [GONE, GONE, 3333333233333334.0, 1.0, 1.0, 1.0, 1.0],
    )


def test_two_close_large_values_still_have_the_spread_between_them() raises:
    """Pandas gives a half for a two wide variance over ten to the fifteen and
    the next number up from it, which differ by one. A sum of squares would
    square both of them first and answer nought, having thrown away every digit
    that told them apart.
    """
    var high = column([1e15, 1e15 + 1.0])
    assert_rows(rolled(high, "var", 2), [GONE, 0.5])


def test_the_missing_rows_are_stepped_over_and_still_counted_against() raises:
    """Pandas gives `[nan, nan, 2, 2, 2, 2]` for a three wide variance over one,
    a hole, three, five, a hole and seven, all with one period.

    Every window that answers holds exactly two values, which is why every
    answer is the same, and the first two rows hold one value each and a count
    of one leaves no degrees of freedom. A hole is not a nought and it is not a
    row either.
    """
    var holes = column([1.0, GONE, 3.0, 5.0, GONE, 7.0])
    assert_rows(
        rolled(holes, "var", 3, min_periods=1),
        [GONE, GONE, 2.0, 2.0, 2.0, 2.0],
    )


def test_a_window_holding_an_infinity_is_not_a_number() raises:
    """Pandas answers `[0, 0, 0, 0.25, 0.25]` here and every one of the first
    three is wrong.

    Its window layer replaces every infinity with a missing value before the
    kernel runs, so the window over row nought and row one is a window over one
    row to pandas, and it answers the variance of a single value. The mean of a
    set holding an infinity is an infinity, the deviations from it are an
    infinity minus an infinity, and there is no number there, so this answers a
    NaN and the registry in the conformance suite records the difference.
    """
    var reaching = column([1.0, Float64.MAX, 2.0, 3.0, 4.0])
    assert_rows(
        rolled(reaching, "var", 2, ddof=0, min_periods=1),
        [0.0, GONE, GONE, 0.25, 0.25],
    )


def test_a_window_that_overflowed_recovers_once_the_value_leaves() raises:
    """Pandas answers `[nan, 0, inf, inf, inf]` here and the last two are wrong.

    Two rows of ten to the two hundred have a variance of nought. A window
    holding one of them and a small number has a variance too large for a double
    and genuinely is an infinity. Every window after that holds small numbers
    only and has a small variance, which pandas cannot reach because its
    accumulated deviations went infinite and no later subtraction brings them
    back. The bound notices that its own state has stopped being a number and
    rebuilds.
    """
    var huge = column([1e200, 1e200, 1.0, 2.0, 3.0])
    var spread = rolled(huge, "var", 2)
    assert_true(isnan(spread[0]), "one row and one degree of freedom")
    assert_equal(spread[1], 0.0, "two of the same value")
    assert_true(isinf(spread[2]), "and this one really is an infinity")
    assert_equal(spread[3], 0.5, "but this one is not")
    assert_equal(spread[4], 0.5)


def test_an_integer_column_answers_float64() raises:
    """A variance is not a whole number even when its rows are, and every window
    reduction answers float64 in pandas anyway.
    """
    var whole = Array[DType.int64](5)
    for i in range(5):
        whole[i] = Int64(i + 1)
    var series = Series("v", AnyArray(whole^))
    var spread = series.rolling(
        WindowOp.VAR, 3, None, False, WindowEdge.RIGHT, None, 1
    )
    assert_equal(String(spread.logical()), "float64")
    assert_rows(rows(spread), [GONE, GONE, 1.0, 1.0, 1.0])


def test_the_three_spreads_know_they_are_spreads() raises:
    """The dispatch in the kernel reads one question off the reduction rather
    than listing the three of them at every branch, so the question is worth
    asking directly.
    """
    assert_true(WindowOp.VAR.spreads())
    assert_true(WindowOp.STD.spreads())
    assert_true(WindowOp.SEM.spreads())
    assert_false(WindowOp.SUM.spreads())
    assert_false(WindowOp.MEAN.spreads())
    assert_false(WindowOp.COUNT.spreads())
    assert_false(WindowOp.MIN.spreads())
    assert_false(WindowOp.MAX.spreads())
    assert_equal(String(WindowOp.VAR), "var")
    assert_equal(String(WindowOp.STD), "std")
    assert_equal(String(WindowOp.SEM), "sem")
    assert_true(op_named("var") == WindowOp.VAR)
    assert_true(op_named("std") == WindowOp.STD)
    assert_true(op_named("sem") == WindowOp.SEM)


def test_an_empty_column_answers_an_empty_column() raises:
    """And does not read a row that is not there on the way."""
    var nothing = Series("v", AnyArray(Array[DType.float64](0)))
    assert_equal(len(rolled(nothing, "var", 3)), 0)
    assert_equal(len(expanded(nothing, "std")), 0)


def test_the_state_says_when_it_stops_being_worth_carrying() raises:
    """The bound on its own, because everything above is an assertion about it
    and reading it directly is what makes the ones above short.

    Three small values have a bound far under their own sum of squared
    deviations. Take the large value back out of a window that held one and the
    bound is larger than what is left, which is the state saying that the digits
    it is holding are not digits of the answer.
    """
    var plain = Spread()
    plain.add(1.0)
    plain.add(2.0)
    plain.add(3.0)
    assert_true(plain.settled(), "three small values are fine")
    assert_equal(plain.answer(3, 1, SPREAD_VAR), 1.0)
    assert_equal(plain.answer(3, 0, SPREAD_VAR), 0.6666666666666666)
    assert_equal(plain.answer(3, 1, SPREAD_STD), 1.0)
    assert_equal(plain.answer(3, 1, SPREAD_SEM), 1.0 / sqrt(3.0))

    var loaded = Spread()
    loaded.add(1e8)
    loaded.add(1.0)
    loaded.add(2.0)
    loaded.add(3.0)
    assert_true(loaded.settled(), "the large value is still in it")
    loaded.drop(1e8)
    assert_false(loaded.settled(), "and now the remainder is not trustworthy")


def test_the_state_counts_the_infinities_rather_than_folding_them_in() raises:
    """One infinity folded into a sum of squared deviations turns it into a NaN
    and no later subtraction brings it back, so the state holds them to one side
    and puts them back when the answer is asked for.
    """
    var passing = Spread()
    passing.add(1.0)
    passing.add(Float64.MAX)
    passing.add(2.0)
    assert_true(isnan(passing.answer(3, 1, SPREAD_VAR)), "while it is in there")
    passing.drop(Float64.MAX)
    assert_true(passing.settled(), "the finite state was never touched")
    assert_equal(passing.answer(2, 1, SPREAD_VAR), 0.5, "and it recovers")


def test_a_rolling_skewness_matches_what_pandas_answers() raises:
    """Seven rows with two spikes in them, which is a column where every window
    has a different shape and none of them is symmetric.

    pandas gives `[nan, nan, 1.6523167403329897, 1.6300591617118865,
    1.3896361387064922, 1.660818054190804, 1.715023568826812]` for a three wide
    skewness over one, two, ten, three, one, twenty, two.
    """
    var series = column([1.0, 2.0, 10.0, 3.0, 1.0, 20.0, 2.0])
    assert_near(
        rolled(series, "skew", 3),
        [
            GONE,
            GONE,
            1.6523167403329897,
            1.6300591617118865,
            1.3896361387064922,
            1.660818054190804,
            1.715023568826812,
        ],
    )


def test_a_rolling_kurtosis_matches_what_pandas_answers() raises:
    """The same column four wide, where a kurtosis has enough rows to answer.

    pandas gives `[nan, nan, nan, 3.227999999999996, 3.2279999999999935,
    -0.2482750148440891, 3.837900874635567]`. The third and fourth entries are
    the same four values in a different order, which is a column worth having
    here because a shape does not depend on the order and a carried state
    could.
    """
    var series = column([1.0, 2.0, 10.0, 3.0, 1.0, 20.0, 2.0])
    assert_near(
        rolled(series, "kurt", 4),
        [
            GONE,
            GONE,
            GONE,
            3.227999999999996,
            3.2279999999999935,
            -0.2482750148440891,
            3.837900874635567,
        ],
    )


def test_an_expanding_shape_ends_at_the_whole_column() raises:
    """The last row of an expanding window is the reduction over everything,
    which is the one row of it a reader can check against a whole column
    skewness.

    pandas gives 1.826596914434746 for the skewness of the whole seven rows and
    2.8988442665289256 for the kurtosis, and the same two numbers come out of
    `Series.skew` and `Series.kurt` on the column with no window at all.
    """
    var series = column([1.0, 2.0, 10.0, 3.0, 1.0, 20.0, 2.0])
    var skewed = expanded(series, "skew")
    var peaked = expanded(series, "kurt")
    assert_near(
        [skewed[len(skewed) - 1]], [1.826596914434746], "the whole skewness"
    )
    assert_near(
        [peaked[len(peaked) - 1]], [2.8988442665289256], "the whole kurtosis"
    )
    assert_near(
        [skewed[2]], [1.6523167403329897], "and the third row is three rows"
    )


def test_a_shape_needs_more_rows_than_min_periods_asks_for() raises:
    """A skewness needs three values and a kurtosis four, whatever was asked
    for.

    Below that the denominator of the standardized moment is nought and there
    is nothing to answer, so both refuse however low `min_periods` goes. pandas
    holds to the same floor: a five wide skewness over one to six asking for
    one value gives `[nan, nan, 0, 0, 0, 0]` and the kurtosis gives `[nan, nan,
    nan, -1.200000000000001, -1.1999999999999993, -1.1999999999999993]`.
    """
    var series = column([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert_rows(
        rolled(series, "skew", 5, 1, 1),
        [GONE, GONE, 0.0, 0.0, 0.0, 0.0],
        "a symmetric window is exactly nought",
    )
    assert_near(
        rolled(series, "kurt", 5, 1, 1),
        [
            GONE,
            GONE,
            GONE,
            -1.200000000000001,
            -1.1999999999999993,
            -1.1999999999999993,
        ],
    )


def test_a_window_of_one_repeated_value_answers_what_pandas_states() raises:
    """The two figures pandas states for a constant window, asserted to the bit.

    A constant is symmetric so a skewness of nought is right. A kurtosis of
    minus three is not a value arithmetic supports, since the standardized
    fourth moment of a real distribution is never below one, and it is copied
    rather than derived because a caller comparing the two libraries on a
    column of one repeated value would read a NaN as a bug in this one. pandas
    answers both at every width and disagrees with itself on the second: the
    same six rows through `Series.kurt` with no window at all answer nought.
    """
    var series = column([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    for width in [4, 5, 6]:
        var skewed = rolled(series, "skew", width)
        var peaked = rolled(series, "kurt", width)
        assert_equal(
            skewed[len(skewed) - 1], 0.0, "skew at width " + String(width)
        )
        assert_equal(
            peaked[len(peaked) - 1], -3.0, "kurt at width " + String(width)
        )


def test_a_run_that_ends_stops_answering_the_stated_figures() raises:
    """The run counter has to be a run and not a tally, so a column that goes
    back to one value after a spike answers the stated figures again and not
    before.

    pandas gives `[nan, nan, 0, 1.7320508075688776, 1.7320508075688776,
    1.7320508075688776, 0, 0]` for a three wide skewness over two, two, two,
    five, two, two, two, two, and `[nan, nan, nan, 4.000000000000002,
    4.000000000000002, 4.000000000000002, 4.000000000000002, -3]` for the four
    wide kurtosis. The kurtosis of two, two, two and five is exactly four and
    this answers exactly four, which is the one row in the pair of columns
    where pandas is the one that is out.
    """
    var series = column([2.0, 2.0, 2.0, 5.0, 2.0, 2.0, 2.0, 2.0])
    assert_near(
        rolled(series, "skew", 3),
        [
            GONE,
            GONE,
            0.0,
            1.7320508075688776,
            1.7320508075688776,
            1.7320508075688776,
            0.0,
            0.0,
        ],
    )
    var peaked = rolled(series, "kurt", 4)
    assert_equal(peaked[3], 4.0, "the exact answer, which pandas misses")
    assert_equal(peaked[7], -3.0, "and the run has come back")


def test_a_shape_does_not_move_when_the_whole_column_does() raises:
    """A skewness and a kurtosis are unchanged by adding the same number to
    every row, and a column near ten to the eight is where a raw sum of fourth
    powers stops being able to say so, because the fourth power is near ten to
    the thirty two and the answer is about a spread near one.

    pandas answers exactly nought and minus 1.200000000000001 here, because it
    subtracts a rounded column mean before it starts whenever the column sits
    close to its own mean. This carries the moments about a moving mean instead
    and answers the same thing without needing the column to cooperate.
    """
    var near = List[Float64]()
    for k in range(6):
        near.append(1e8 + Float64(k + 1))
    var series = column(near)
    assert_rows(
        rolled(series, "skew", 3),
        [GONE, GONE, 0.0, 0.0, 0.0, 0.0],
        "a symmetric window near ten to the eight",
    )
    assert_near(
        rolled(series, "kurt", 4),
        [
            GONE,
            GONE,
            GONE,
            -1.200000000000001,
            -1.200000000000001,
            -1.200000000000001,
        ],
    )


def test_a_large_value_leaving_the_window_does_not_take_the_shape_with_it() raises:
    """The rebuild, which is the whole reason the state carries a bound.

    Ten to the eight followed by one to six. The windows that hold the large
    value have whatever shape they have, and every window after it holds small
    numbers only and has to come back to their shape rather than to the
    difference of two large numbers. pandas answers the same column the same
    way, so this is not a difference, it is the test that fails if the bound
    stops firing.
    """
    var series = column([1e8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert_near(
        rolled(series, "skew", 3),
        [
            GONE,
            GONE,
            1.7320508075688767,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )
    assert_near(
        rolled(series, "kurt", 4),
        [
            GONE,
            GONE,
            GONE,
            3.9999999999999964,
            -1.200000000000001,
            -1.200000000000001,
            -1.200000000000001,
        ],
    )


def test_a_shape_counts_values_and_steps_over_the_missing_rows() raises:
    """Gaps in the column, where the floor of three or four values is counted in
    values and not in rows.

    pandas gives a three wide skewness asking for three values as missing
    everywhere but the last row, which is the only window holding three values,
    and a four wide kurtosis asking for four as missing everywhere.
    """
    var series = column([1.0, GONE, 3.0, 5.0, GONE, 7.0, 9.0, 11.0])
    assert_rows(
        rolled(series, "skew", 3, 1, 3),
        [GONE, GONE, GONE, GONE, GONE, GONE, GONE, 0.0],
        "one window holds three values and it is symmetric",
    )
    assert_rows(
        rolled(series, "kurt", 4, 1, 4),
        [GONE, GONE, GONE, GONE, GONE, GONE, GONE, GONE],
        "and no window ever holds four",
    )


def test_a_shape_over_a_window_holding_an_infinity_is_not_a_number() raises:
    """The same argument as the spread, one power further along.

    The mean of a set holding an infinity is an infinity and every deviation
    from it is an infinity minus an infinity, so there is no shape there.
    pandas had already replaced the infinity with a missing value, so its
    window over the rows either side answers their skewness, and this refuses.
    The rows after the infinity has left come back, which is what the count to
    one side is for.
    """
    var series = column(
        [1.0, Float64.MAX * 2.0, 2.0, 3.0, 4.0, 5.0],
    )
    assert_rows(
        rolled(series, "skew", 3),
        [GONE, GONE, GONE, GONE, 0.0, 0.0],
        "and it recovers once the infinity is out of the window",
    )


def test_a_shape_below_the_variance_pandas_refuses_still_answers() raises:
    """pandas answers NaN for both shapes whenever the population variance of
    the window is at or below ten to the minus fourteen, and the threshold is
    absolute rather than relative.

    So whether pandas will tell you the shape of your readings depends on the
    units you wrote them in. Four readings alternating by a hundredth of a
    micron have a population variance of ten to the minus sixteen and a
    kurtosis of exactly minus six, and pandas refuses it while answering the
    same four readings scaled up by eleven. A one pass power sum cannot compute
    a shape down there, so the threshold is doing real work for pandas, and a
    carried central moment with a rebuild can.
    """
    var tiny = 1e-8
    var series = column([-tiny, tiny, -tiny, tiny])
    assert_near(rolled(series, "kurt", 4), [GONE, GONE, GONE, -6.0])
    assert_near(rolled(series, "skew", 4), [GONE, GONE, GONE, 0.0])


def test_a_spread_that_underflowed_has_no_shape_either() raises:
    """Below the point where the squared deviations themselves underflow to
    nought there is no ratio to take and both answer NaN, which pandas also
    does. This is the one place the two agree about refusing."""
    var gone = 1e-170
    var series = column([-gone, gone, -gone, gone])
    assert_rows(
        rolled(series, "kurt", 4),
        [GONE, GONE, GONE, GONE],
        (
            "the fourth power of a ten to the minus one hundred and seventy is"
            " nought"
        ),
    )


def test_the_two_shapes_know_they_are_shapes() raises:
    """The dispatch in the kernel reads one question off the reduction rather
    than listing the two of them at every branch, and the two questions have to
    be exclusive or the spread branch would swallow them."""
    assert_true(WindowOp.SKEW.shapes())
    assert_true(WindowOp.KURT.shapes())
    assert_false(WindowOp.VAR.shapes())
    assert_false(WindowOp.SEM.shapes())
    assert_false(WindowOp.SUM.shapes())
    assert_false(WindowOp.SKEW.spreads())
    assert_false(WindowOp.KURT.spreads())
    assert_equal(String(WindowOp.SKEW), "skew")
    assert_equal(String(WindowOp.KURT), "kurt")
    assert_true(op_named("skew") == WindowOp.SKEW)
    assert_true(op_named("kurt") == WindowOp.KURT)


def test_the_moment_state_says_when_it_stops_being_worth_carrying() raises:
    """The bounds on their own, which is what every rebuild above depends on.

    Four small values have bounds far under their own moments. Take the large
    value back out of a window that held one and the bounds are larger than
    what is left, which is the state saying that the digits it is holding are
    not digits of the answer.
    """
    var plain = Moments()
    plain.add(1.0)
    plain.add(2.0)
    plain.add(3.0)
    plain.add(4.0)
    assert_true(plain.settled(), "four small values are fine")
    assert_near([plain.answer(4, MOMENT_SKEW)], [0.0], "and symmetric")
    assert_near([plain.answer(4, MOMENT_KURT)], [-1.2], "and flat")
    assert_true(
        isnan(plain.answer(2, MOMENT_SKEW)), "two values have no skewness"
    )
    assert_true(isnan(plain.answer(3, MOMENT_KURT)), "three have no kurtosis")

    var loaded = Moments()
    loaded.add(1e8)
    loaded.add(1.0)
    loaded.add(2.0)
    loaded.add(3.0)
    loaded.add(4.0)
    assert_true(loaded.settled(), "the large value is still in it")
    loaded.drop(1e8)
    assert_false(loaded.settled(), "and now the remainder is not trustworthy")


def test_the_moments_count_the_infinities_rather_than_folding_them_in() raises:
    """The same arrangement the spread uses, and it has to be the same one,
    because the moments hold a spread rather than repeating it. An infinity
    folded into a sum of fourth powers turns it into a NaN and no later
    subtraction brings it back.
    """
    var passing = Moments()
    passing.add(1.0)
    passing.add(2.0)
    passing.add(Float64.MAX * 2.0)
    passing.add(3.0)
    passing.add(4.0)
    assert_true(isnan(passing.answer(5, MOMENT_SKEW)), "while it is in there")
    passing.drop(Float64.MAX * 2.0)
    assert_true(passing.settled(), "the finite state was never touched")
    assert_near([passing.answer(4, MOMENT_KURT)], [-1.2], "and it recovers")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
