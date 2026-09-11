"""Tests for the exponentially weighted window.

Every number quoted here was read out of a running pandas 3.0.5 before the
kernel was written, which matters more for this window than for the rolling one
because there is nothing to check by hand. A rolling sum over nought to nine can
be added up on paper. An exponentially weighted mean cannot, so a wrong
implementation of it answers a column of numbers that look exactly as plausible
as the right ones and the only way to tell is to have the right ones written
down first.

The comparisons are to within a unit or two in the last place rather than exact,
and the reason is worth writing down because it looks like sloppiness. pandas
runs the same recurrence this kernel does, and on this machine it still answers a
different last bit on some rows, because the C compiler that built the pandas
wheel contracted the `old_wt * weighted + new_wt * cur` of the fold into a single
fused multiply-add. That rounds once where two operations round twice. On nought
to nine with a span of five it moves rows five and eight and leaves the other
eight alone. It is a property of how a wheel was compiled and not of what pandas
computes, so it is not something to reproduce, and every tolerance the
conformance board uses is several orders of magnitude looser than it. The
tolerance here is `1e-15` relative, which is about four units in the last place
and tight enough that a genuinely wrong recurrence still fails.
"""

from std.math import exp, inf, isinf, isnan, log, nan
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.frame.series import Series
from firepanda.kernel.ewm import EwmOp, EwmSpec, alpha_of, ewm_named


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
    """Returns nought to nine.

    Returns:
        The series.
    """
    var values = List[Float64]()
    for i in range(10):
        values.append(Float64(i))
    return column(values)


def gapped() -> Series:
    """Returns the column the missing row rules are checked on.

    It is `1, null, 3, null, null, 7, 9`, which has a single gap and then a
    double one, because one gap cannot tell the two readings of `ignore_na`
    apart by much and two can.

    Returns:
        The series.
    """
    var missing = nan[DType.float64]()
    var values: List[Float64] = [
        1.0,
        missing,
        3.0,
        missing,
        missing,
        7.0,
        9.0,
    ]
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


def spec(alpha: Float64, adjust: Bool, ignore_na: Bool) -> EwmSpec:
    """Builds a spec with the two recurrence flags spelled out.

    Args:
        alpha: The smoothing factor.
        adjust: Whether every row weighs one.
        ignore_na: Whether a missing row is skipped.

    Returns:
        The spec.
    """
    return EwmSpec(alpha, 0, adjust, ignore_na, False)


def assert_near(got: Float64, want: Float64) raises:
    """Compares against a number pandas answered, allowing the last bit or two.

    See the note at the top of the file for why this is not an exact comparison
    and why it is still tight enough to catch a wrong recurrence.

    Args:
        got: What the kernel answered.
        want: What pandas answered.

    Raises:
        Error: If they differ by more than a relative `1e-15`.
    """
    assert_almost_equal(got, want, atol=0.0, rtol=1e-15)


def test_the_four_spellings_of_the_decay_are_one_number() raises:
    """A span of five, a centre of mass of two and a third are one window."""
    var none = Optional[Float64]()
    assert_equal(alpha_of(none, 5.0, none, none), 1.0 / 3.0)
    assert_equal(alpha_of(2.0, none, none, none), 1.0 / 3.0)
    assert_equal(alpha_of(none, none, none, 0.25), 0.25)
    assert_almost_equal(
        alpha_of(none, none, 3.0, none), 0.2062994740159002, atol=1e-15
    )
    assert_equal(alpha_of(none, none, 3.0, none), 1.0 - exp(-log(2.0) / 3.0))


def test_the_decay_has_to_arrive_exactly_once() raises:
    """Four ways of writing one number, so none and two are both wrong."""
    var none = Optional[Float64]()
    with assert_raises(contains="Must pass one of"):
        _ = alpha_of(none, none, none, none)
    with assert_raises(contains="mutually exclusive"):
        _ = alpha_of(2.0, 5.0, none, none)
    with assert_raises(contains="mutually exclusive"):
        _ = alpha_of(none, none, 3.0, 0.5)


def test_each_spelling_keeps_its_own_range() raises:
    """The bounds differ between the four and pandas' sentences say which."""
    var none = Optional[Float64]()
    with assert_raises(contains="span >= 1"):
        _ = alpha_of(none, 0.5, none, none)
    with assert_raises(contains="comass >= 0"):
        _ = alpha_of(-1.0, none, none, none)
    with assert_raises(contains="halflife > 0"):
        _ = alpha_of(none, none, 0.0, none)
    with assert_raises(contains="0 < alpha <= 1"):
        _ = alpha_of(none, none, none, 0.0)
    with assert_raises(contains="0 < alpha <= 1"):
        _ = alpha_of(none, none, none, 1.5)
    assert_equal(alpha_of(none, none, none, 1.0), 1.0)
    assert_equal(alpha_of(0.0, none, none, none), 1.0)


def test_the_adjusted_mean_matches_pandas_on_a_plain_column() raises:
    """`pd.Series(range(10)).ewm(span=5).mean()`, which is the same loop."""
    var out = rows(counting().ewm(EwmOp.MEAN, EwmSpec(1.0 / 3.0)))
    var want: List[Float64] = [
        0.0,
        0.6,
        1.263157894736842,
        1.9846153846153844,
        2.758293838862559,
        3.5774436090225565,
        4.435162700339971,
        5.324821570182395,
        6.2403630483542845,
        7.176475657044377,
    ]
    assert_equal(len(out), 10)
    for i in range(10):
        assert_near(out[i], want[i])


def test_a_half_life_of_three_rows_is_the_same_loop_again() raises:
    """`ewm(halflife=3).mean()`, to check the conversion and not the loop."""
    var none = Optional[Float64]()
    var out = rows(
        counting().ewm(EwmOp.MEAN, EwmSpec(alpha_of(none, none, 3.0, none)))
    )
    assert_near(out[1], 0.5575066659755579)
    assert_near(out[9], 6.25407609681849)


def test_the_unadjusted_mean_is_the_plain_recursion() raises:
    """With an alpha of a half the answer halves the gap to each row.

    `ewm(alpha=0.5, adjust=False).mean()` on nought to nine, where pandas
    answers 8.001953125 on the last row and the recursion is short enough to
    follow: each row is the midpoint of the row before it and the answer before
    it.
    """
    var out = rows(
        counting().ewm(EwmOp.MEAN, spec(0.5, adjust=False, ignore_na=False))
    )
    assert_equal(out[1], 0.5)
    assert_equal(out[2], 1.25)
    assert_equal(out[3], 2.125)
    assert_almost_equal(out[9], 8.001953125, atol=1e-12)


def test_the_two_flags_give_four_different_answers() raises:
    """All four of `adjust` by `ignore_na` on `1, null, 3, null, null, 7, 9`.

    These are the numbers the file is really about. They were read out of pandas
    3.0.5 one combination at a time and they are four distinct answers, so a
    kernel that quietly ignores either flag fails here rather than somewhere
    downstream.
    """
    var adjusted_counting = rows(
        gapped().ewm(EwmOp.MEAN, spec(0.3, adjust=True, ignore_na=False))
    )
    assert_near(adjusted_counting[2], 2.3422818791946307)
    assert_near(adjusted_counting[6], 7.162170410482522)

    var adjusted_skipping = rows(
        gapped().ewm(EwmOp.MEAN, spec(0.3, adjust=True, ignore_na=True))
    )
    assert_near(adjusted_skipping[2], 2.1764705882352944)
    assert_near(adjusted_skipping[6], 6.203316225819187)

    var plain_counting = rows(
        gapped().ewm(EwmOp.MEAN, spec(0.3, adjust=False, ignore_na=False))
    )
    assert_near(plain_counting[2], 1.759493670886076)
    assert_almost_equal(plain_counting[6], 5.643163966375967, atol=1e-14)

    var plain_skipping = rows(
        gapped().ewm(EwmOp.MEAN, spec(0.3, adjust=False, ignore_na=True))
    )
    assert_near(plain_skipping[2], 1.5999999999999999)
    assert_almost_equal(plain_skipping[6], 4.954, atol=1e-14)


def test_a_missing_row_answers_the_row_before_it() raises:
    """Nothing arrived, so nothing changed, which is not the same as nothing."""
    var out = rows(gapped().ewm(EwmOp.MEAN, EwmSpec(0.3)))
    assert_equal(out[1], out[0])
    assert_equal(out[3], out[2])
    assert_equal(out[4], out[2])


def test_a_leading_gap_is_not_a_value_of_zero() raises:
    """The first row with something in it starts the recurrence, wherever it is.
    """
    var missing = nan[DType.float64]()
    var values: List[Float64] = [missing, 2.0, 4.0]
    var out = rows(column(values).ewm(EwmOp.MEAN, EwmSpec(0.3)))
    assert_true(isnan(out[0]))
    assert_equal(out[1], 2.0)
    assert_near(out[2], 3.1764705882352944)


def test_a_column_of_nothing_answers_nothing() raises:
    """Two missing rows, and pandas answers two missing rows."""
    var missing = nan[DType.float64]()
    var values: List[Float64] = [missing, missing]
    var out = rows(column(values).ewm(EwmOp.MEAN, EwmSpec(0.3)))
    assert_true(isnan(out[0]))
    assert_true(isnan(out[1]))


def test_one_value_repeated_comes_back_exactly() raises:
    """The guard copied out of pandas, which is what this test is for.

    Folding a value into a mean that already equals it is arithmetic that should
    change nothing and does, in the last bits. Skipping the fold is the only
    reason a flat column stays flat.
    """
    var values: List[Float64] = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
    var out = rows(column(values).ewm(EwmOp.MEAN, EwmSpec(0.3)))
    for i in range(6):
        assert_equal(out[i], 2.5)


def test_min_periods_counts_values_and_not_rows() raises:
    """Three values on a column with three gaps, so five rows answer nothing."""
    var out = rows(
        gapped().ewm(EwmOp.MEAN, EwmSpec(0.3, 3, True, False, False))
    )
    for i in range(5):
        assert_true(isnan(out[i]))
    assert_near(out[5], 5.424679200831199)
    assert_near(out[6], 7.162170410482522)


def test_a_count_of_nought_and_a_count_of_one_are_the_same() raises:
    """A default of nought that then answers the first row anyway."""
    var lenient = rows(
        gapped().ewm(EwmOp.MEAN, EwmSpec(0.3, 0, True, False, False))
    )
    var explicit = rows(
        gapped().ewm(EwmOp.MEAN, EwmSpec(0.3, 1, True, False, False))
    )
    for i in range(7):
        assert_equal(lenient[i], explicit[i])
    assert_equal(lenient[0], 1.0)


def test_the_total_is_the_mean_without_the_division() raises:
    """`ewm(span=5).sum()` on nought to nine, and then on the gapped column."""
    var out = rows(counting().ewm(EwmOp.SUM, EwmSpec(1.0 / 3.0)))
    assert_equal(out[1], 1.0)
    assert_near(out[2], 2.666666666666667)
    assert_near(out[9], 21.156073769242496)

    var gaps = rows(gapped().ewm(EwmOp.SUM, EwmSpec(0.3)))
    assert_equal(gaps[0], 1.0)
    assert_equal(gaps[1], 0.7)
    assert_near(gaps[2], 3.4899999999999998)
    assert_near(gaps[6], 14.737949)


def test_the_total_refuses_the_combination_pandas_refuses() raises:
    """`sum` with `adjust=False` has no pandas answer, so it has none here.

    pandas raises `NotImplementedError` with this sentence rather than choosing
    one of the two things it could mean. Inventing an answer would be the one
    place in this file where firepanda is not compatible, so the sentence is
    copied.
    """
    with assert_raises(contains="sum is not implemented with adjust=False"):
        _ = counting().ewm(EwmOp.SUM, spec(0.3, adjust=False, ignore_na=False))


def test_the_variance_matches_pandas_on_a_column_with_no_gaps() raises:
    """`ewm(span=5).var()` on nought to nine, corrected and not."""
    var corrected = rows(counting().ewm(EwmOp.VAR, EwmSpec(1.0 / 3.0)))
    assert_true(isnan(corrected[0]))
    assert_equal(corrected[1], 0.5)
    assert_near(corrected[2], 0.9736842105263156)
    assert_near(corrected[9], 5.301907596366596)

    var biased = rows(
        counting().ewm(EwmOp.VAR, EwmSpec(1.0 / 3.0, 0, True, False, True))
    )
    assert_equal(biased[0], 0.0)
    assert_near(biased[1], 0.24)
    assert_near(biased[9], 4.204099772026982)


def test_the_first_row_has_no_corrected_variance_and_a_biased_one() raises:
    """One value has a second moment of nought and no unbiased estimate at all.

    The correction divides by the sum of the weights squared less the sum of the
    squared weights, which for one row is nought, so there is no number rather
    than a zero. Reporting a zero would say the column is flat, which is not
    something one value can say.
    """
    var corrected = rows(counting().ewm(EwmOp.VAR, EwmSpec(0.3)))
    assert_true(isnan(corrected[0]))
    var biased = rows(
        counting().ewm(EwmOp.VAR, EwmSpec(0.3, 0, True, False, True))
    )
    assert_equal(biased[0], 0.0)


def test_the_variance_carries_the_gaps_the_same_way_the_mean_does() raises:
    """`ewm(alpha=0.3).var()` on the gapped column, both flags both ways."""
    var adjusted = rows(gapped().ewm(EwmOp.VAR, EwmSpec(0.3)))
    assert_true(isnan(adjusted[0]))
    assert_true(isnan(adjusted[1]))
    assert_near(adjusted[2], 1.9999999999999998)
    assert_near(adjusted[5], 10.347054105073973)
    assert_near(adjusted[6], 9.256066816173421)

    var plain = rows(
        gapped().ewm(EwmOp.VAR, EwmSpec(0.3, 0, False, False, False))
    )
    assert_almost_equal(plain[5], 11.613593041897571, atol=1e-13)
    assert_almost_equal(plain[6], 13.659241139082107, atol=1e-13)


def test_the_deviation_is_the_root_of_the_variance() raises:
    """`ewm(alpha=0.3).std()`, which is the one reduction with no loop of its
    own.
    """
    var out = rows(gapped().ewm(EwmOp.STD, EwmSpec(0.3)))
    assert_true(isnan(out[0]))
    assert_near(out[2], 1.414213562373095)
    assert_near(out[5], 3.216683712315212)
    assert_near(out[6], 3.0423784800996443)


def test_the_answer_is_as_tall_as_the_column_and_keeps_its_name() raises:
    """There is no step and no shape here, so there is nothing to relabel."""
    var out = counting().ewm(EwmOp.MEAN, EwmSpec(0.3))
    assert_equal(len(out), 10)
    assert_equal(out.name, "v")
    assert_equal(len(out.index), 10)


def test_a_whole_number_column_is_widened_before_it_is_weighted() raises:
    """A weighted mean of whole numbers is not a whole number."""
    var out = Array[DType.int64](4)
    for i in range(4):
        out[i] = Int64(i + 1)
    var series = Series("v", AnyArray(out^))
    var answer = series.ewm(EwmOp.MEAN, EwmSpec(0.5))
    assert_equal(String(answer.values.type), "float64")
    assert_near(rows(answer)[1], 1.6666666666666667)


def test_a_text_column_has_no_weighted_mean() raises:
    """The same refusal the rolling window gives, for the same reason."""
    var builder = StringBuilder()
    builder.append(String("one").as_bytes())
    builder.append(String("two").as_bytes())
    var series = Series("v", builder^.finish())
    with assert_raises(contains="is not defined on"):
        _ = series.ewm(EwmOp.MEAN, EwmSpec(0.5))


def test_the_kernel_checks_the_factor_at_its_own_door() raises:
    """`alpha_of` is not the only way in, because the Mojo API is one too."""
    with assert_raises(contains="0 < alpha <= 1"):
        _ = counting().ewm(EwmOp.MEAN, EwmSpec(0.0))
    with assert_raises(contains="0 < alpha <= 1"):
        _ = counting().ewm(EwmOp.MEAN, EwmSpec(1.5))


def test_a_factor_of_one_forgets_everything_before_this_row() raises:
    """The edge of the range, where the window is one row wide."""
    var out = rows(counting().ewm(EwmOp.MEAN, EwmSpec(1.0)))
    for i in range(10):
        assert_equal(out[i], Float64(i))


def test_the_four_reductions_are_named_after_the_pandas_methods() raises:
    """Both directions, because the Python layer goes through the name."""
    assert_true(ewm_named("mean") == EwmOp.MEAN)
    assert_true(ewm_named("sum") == EwmOp.SUM)
    assert_true(ewm_named("var") == EwmOp.VAR)
    assert_true(ewm_named("std") == EwmOp.STD)
    assert_equal(String(EwmOp.MEAN), "mean")
    assert_equal(String(EwmOp.SUM), "sum")
    assert_equal(String(EwmOp.VAR), "var")
    assert_equal(String(EwmOp.STD), "std")
    assert_true(EwmOp.VAR.spreads())
    assert_false(EwmOp.MEAN.spreads())
    assert_true(EwmOp.MEAN != EwmOp.SUM)
    with assert_raises(contains="no exponentially weighted reduction"):
        _ = ewm_named("median")


def test_an_infinity_is_carried_to_the_bottom_of_the_column() raises:
    """This window never drops a row, so there is nothing to recover from.

    A rolling window loses an infinity and gets the column back once the row
    that carried it has left. Here every row stays in the window forever and its
    weight only shrinks, so the weighted average of an infinity and a finite
    number is that infinity however far down the column it is asked.
    """
    var values: List[Float64] = [1.0, inf[DType.float64](), 2.0, 3.0, 4.0]
    var got = rows(column(values).ewm(EwmOp.MEAN, spec(0.3, True, False)))
    assert_equal(got[0], 1.0)
    assert_true(isinf(got[1]) and got[1] > 0.0)
    assert_true(isinf(got[4]) and got[4] > 0.0)


def test_a_cancelled_infinity_starts_the_recurrence_again() raises:
    """The one place a NaN means two things, and it is pandas' sentinel.

    Folding a positive infinity against a negative one gives a NaN, and the test
    for whether the recurrence has started is whether the carried value equals
    itself. So the row after the cancellation starts again from its own value,
    which is what pandas' kernel would do if pandas let an infinity reach it.
    The conflation is only reachable this way, because a NaN in the data is read
    as a missing row before it ever reaches the fold.
    """
    var values: List[Float64] = [
        inf[DType.float64](),
        -inf[DType.float64](),
        5.0,
        7.0,
    ]
    var got = rows(column(values).ewm(EwmOp.MEAN, spec(0.3, True, False)))
    assert_true(isinf(got[0]) and got[0] > 0.0)
    assert_true(isnan(got[1]))
    assert_equal(got[2], 5.0)
    assert_near(got[3], (0.7 * 5.0 + 1.0 * 7.0) / 1.7)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
