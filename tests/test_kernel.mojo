"""Tests for the kernel layer.

Most of these check a kernel against its scalar twin on data small enough to
reason about by hand. The fuzz harness does the same thing on millions of random
columns; these exist so that when the fuzzer finds something, there is a place to
put a small case that reproduces it.

Three of them are not comparisons. `test_sum_of_all_null_is_zero` and
`test_min_of_all_null_is_invalid` pin down the two disagreeing conventions for an
empty reduction, and `test_setitem_into_a_null_breaks_the_invariant` documents the
one way a caller can make `sum_of` give a wrong answer.
"""

from std.math import inf, isinf, isnan, nan
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.array import Array, from_list
from firepanda.dtype.lists import NUMERIC
from firepanda.kernel import (
    absolute,
    add,
    cast_to,
    count_of,
    divide,
    equal,
    filter_rows,
    floor_divide,
    greater,
    invert,
    less,
    less_equal,
    max_of,
    mean_of,
    min_of,
    modulo,
    multiply,
    negate,
    not_equal,
    power,
    subtract,
    sum_of,
    take_rows,
)
from firepanda.kernel.accum import accumulator
from firepanda.kernel.arith import OP_ADD, arith_const
from firepanda.kernel.scalar import (
    absolute_scalar,
    add_scalar,
    cast_scalar,
    equal_scalar,
    filter_scalar,
    floor_divide_scalar,
    invert_scalar,
    max_scalar,
    mean_scalar,
    min_scalar,
    modulo_scalar,
    multiply_scalar,
    negate_scalar,
    power_scalar,
    subtract_scalar,
    sum_scalar,
    take_scalar,
)


def build[dt: DType](length: Int, null_every: Int) -> Array[dt]:
    """Returns a column whose values vary with position and whose nulls are regular.

    The values stay under 100 so that they are exact in float16 and fit in int8,
    which is what lets the same test body run over every numeric dtype.

    Args:
        length: The number of rows.
        null_every: Every position divisible by this is null. Pass zero for a
            column with no nulls.

    Parameters:
        dt: The dtype.

    Returns:
        The column.
    """
    var out = Array[dt](length)
    for i in range(length):
        out[i] = Scalar[dt]((i * 7) % 90 + 1)
    if null_every > 0:
        for i in range(0, length, null_every):
            out.set_null(i)
    return out^


def test_sum_matches_the_twin_for_every_numeric_dtype() raises:
    comptime for dt in NUMERIC:
        var col = build[dt](301, 5)
        var fast = sum_of(col)
        assert_true(fast.valid)
        assert_equal(fast.value, sum_scalar(col))


def test_min_and_max_match_the_twin_for_every_numeric_dtype() raises:
    comptime for dt in NUMERIC:
        var col = build[dt](301, 5)

        var low = min_of(col)
        var low_twin = min_scalar(col)
        assert_equal(low.valid, low_twin[1])
        assert_equal(low.value, low_twin[0])

        var high = max_of(col)
        var high_twin = max_scalar(col)
        assert_equal(high.valid, high_twin[1])
        assert_equal(high.value, high_twin[0])


def test_min_and_max_see_the_last_partial_word() raises:
    # A length of 130 leaves two values in the third validity word, which is the
    # case the word-at-a-time loop gets wrong if it forgets to clamp to length.
    var col = Array[DType.int64](130)
    for i in range(130):
        col[i] = 50
    col[129] = -7
    col[128] = 900

    assert_equal(min_of(col).value, -7)
    assert_equal(max_of(col).value, 900)


def test_min_ignores_the_zeros_under_the_nulls() raises:
    var col = from_list[DType.int64]([10, 20, 30, 40])
    col.set_null(1)
    assert_equal(col[1], 0)
    assert_equal(min_of(col).value, 10)
    assert_equal(max_of(col).value, 40)


def test_count_and_mean_match_the_twin() raises:
    var col = build[DType.float64](257, 3)
    assert_equal(count_of(col), 171)

    var fast = mean_of(col)
    var twin = mean_scalar(col)
    assert_equal(fast.valid, twin[1])
    assert_almost_equal(fast.value, twin[0])


def test_sum_of_all_null_is_zero() raises:
    # pandas says the sum of nothing is zero. It is a strange thing to say and it
    # is what everyone downstream expects, so firepanda says it too.
    var col = Array[DType.int64](64)
    for i in range(64):
        col.set_null(i)

    var total = sum_of(col)
    assert_true(total.valid)
    assert_equal(total.value, 0)
    assert_equal(count_of(col), 0)


def test_min_of_all_null_is_invalid() raises:
    # And the minimum of nothing is not zero, it is nothing. The two conventions
    # disagree because zero is the identity for one operation and not the other.
    var col = Array[DType.int64](64)
    for i in range(64):
        col.set_null(i)

    assert_false(min_of(col).valid)
    assert_false(max_of(col).valid)
    assert_false(mean_of(col).valid)


def test_every_reduction_steps_over_a_nan() raises:
    # pandas has no presence bitmap on a float column, so NaN is the missing it
    # has there and every reduction skips one. firepanda has both spellings and
    # skips both. See #170.
    var col = from_list[DType.float64]([1.0, 0.0, 3.0, 0.0, 5.0])
    col[1] = nan[DType.float64]()
    col.set_null(3)

    assert_equal(Float64(sum_of(col).value), 9.0)
    assert_equal(min_of(col).value, 1.0)
    assert_equal(max_of(col).value, 5.0)
    assert_equal(mean_of(col).value, 3.0)

    # The divisor lost the NaN and the null, so it is three and not five and not
    # four. `count_of` is the exception on purpose: it is the Arrow answer and
    # counts the four rows whose validity bit is set.
    assert_equal(count_of(col), 4)


def test_a_column_of_nothing_but_nan_has_no_minimum() raises:
    # The case the vector path cannot answer by looking at what it folded to,
    # because a block of nothing but NaN folds to the identity and so would a
    # block holding one real infinity. Two hundred rows so the whole word path
    # runs three times and the tail once.
    var col = Array[DType.float64](200)
    for i in range(200):
        col[i] = nan[DType.float64]()

    assert_false(min_of(col).valid)
    assert_false(max_of(col).valid)
    assert_false(mean_of(col).valid)
    assert_true(sum_of(col).valid)
    assert_equal(Float64(sum_of(col).value), 0.0)


def test_a_real_infinity_is_not_mistaken_for_an_empty_block() raises:
    # The other half of the case above. Every value here folds to the identity a
    # minimum starts from, and the answer is that value rather than nothing.
    var col = Array[DType.float64](128)
    for i in range(128):
        col[i] = nan[DType.float64]()
    col[70] = Float64.MAX_FINITE * 2

    var low = min_of(col)
    assert_true(low.valid)
    assert_true(isinf(low.value))


def test_a_nan_inside_a_full_validity_word_is_still_skipped() raises:
    # No nulls anywhere, so every word takes the vectorized path and the NaNs
    # have to be found by the values rather than by the bitmap. The four
    # positions are the first row, the two either side of a word boundary and one
    # in the middle of the last word.
    var col = Array[DType.float64](256)
    for i in range(256):
        col[i] = Float64(i + 1)
    for at in [0, 63, 64, 200]:
        col[at] = nan[DType.float64]()

    var expected = Float64(0)
    for i in range(256):
        if i != 0 and i != 63 and i != 64 and i != 200:
            expected += Float64(i + 1)

    assert_equal(Float64(sum_of(col).value), expected)
    assert_equal(min_of(col).value, 2.0)
    assert_equal(max_of(col).value, 256.0)
    assert_equal(mean_of(col).value, expected / 252.0)


def test_a_nan_past_the_morsel_split_is_still_skipped() raises:
    # Longer than one morsel, so the reduction runs on every core and the NaN
    # counts have to be carried out of the workers and added up rather than being
    # counted once. Every value is one, so the total is the number of rows that
    # were read and the assertion is about which rows those were.
    var rows = 300000
    var col = Array[DType.float64](rows)
    for i in range(rows):
        col[i] = 1.0
    for i in range(rows):
        if i % 3 == 0:
            col[i] = nan[DType.float64]()
        elif i % 7 == 0:
            col.set_null(i)

    var wanted = 0
    for i in range(rows):
        if i % 3 != 0 and i % 7 != 0:
            wanted += 1

    assert_equal(Float64(sum_of(col).value), Float64(wanted))
    assert_equal(mean_of(col).value, 1.0)
    assert_equal(min_of(col).value, 1.0)
    assert_equal(max_of(col).value, 1.0)


def test_an_int_column_never_reads_a_nan_bit_pattern() raises:
    # 9221120237041090560 is the bit pattern of a quiet NaN read as an integer.
    # In an int64 column it is a number like any other and the reductions have no
    # business noticing it.
    var col = from_list[DType.int64]([1, 9221120237041090560, 2])

    assert_equal(count_of(col), 3)
    assert_equal(max_of(col).value, 9221120237041090560)
    assert_equal(min_of(col).value, 1)


def test_reductions_over_an_empty_column() raises:
    var col = Array[DType.int64](0)
    assert_equal(count_of(col), 0)
    assert_true(sum_of(col).valid)
    assert_equal(sum_of(col).value, 0)
    assert_false(min_of(col).valid)


def test_sum_widens_so_it_does_not_overflow() raises:
    # Two hundred rows of 100 in int8 is 20000, which int8 cannot hold. The
    # accumulator is int64 and this is the whole reason it exists.
    var col = Array[DType.int8](200)
    for i in range(200):
        col[i] = 100

    var total = sum_of(col)
    assert_equal(total.value, 20000)
    assert_equal(accumulator(DType.int8), DType.int64)


def test_setitem_into_a_null_breaks_the_invariant() raises:
    # This is the sharp edge documented in firepanda/kernel/__init__.mojo. It is
    # asserted rather than fixed because fixing it means making `__setitem__`
    # touch the validity bitmap, which puts a second store in the inner loop of
    # every builder in the library.
    var col = from_list[DType.int64]([1, 2, 3, 4])
    col.set_null(2)
    assert_equal(sum_of(col).value, 7)

    col[2] = 99
    assert_false(col.is_valid(2))
    assert_equal(sum_of(col).value, 106)
    assert_equal(sum_scalar(col), 7)

    # set_valid is the way to do it, and it agrees with the twin again.
    col.set_valid(2, 99)
    assert_equal(sum_of(col).value, 106)
    assert_equal(sum_scalar(col), 106)


def test_arithmetic_matches_the_twin() raises:
    comptime for dt in NUMERIC:
        var a = build[dt](193, 5)
        var b = build[dt](193, 7)

        var sums = add(a, b)
        var sums_twin = add_scalar(a, b)
        for i in range(193):
            assert_equal(sums.is_valid(i), sums_twin.is_valid(i))
            assert_equal(sums[i], sums_twin[i])

        var products = multiply(a, b)
        var products_twin = multiply_scalar(a, b)
        for i in range(193):
            assert_equal(products.is_valid(i), products_twin.is_valid(i))
            assert_equal(products[i], products_twin[i])


def test_floor_division_and_the_remainder_match_the_twin() raises:
    """The divisors cycle through zero, which is the row where the two dtype
    families part company: an integer column gets a null there and a float
    column gets an infinity or a NaN and stays present. The twin decides which,
    and this is what says the vector loop agreed."""
    comptime for dt in NUMERIC:
        var a = build[dt](193, 5)
        var b = Array[dt](193)
        for i in range(193):
            # Zero at every ninetieth row, so three of them and none of them on
            # a register boundary.
            b[i] = Scalar[dt]((i * 7) % 90)
        b.set_null(11)

        var quotients = floor_divide(a, b)
        var quotients_twin = floor_divide_scalar(a, b)
        var remainders = modulo(a, b)
        var remainders_twin = modulo_scalar(a, b)
        for i in range(193):
            assert_equal(quotients.is_valid(i), quotients_twin.is_valid(i))
            assert_equal(remainders.is_valid(i), remainders_twin.is_valid(i))
            # A NaN is not equal to itself, so the rows a float remainder by
            # zero produces are checked for being a NaN on both sides instead.
            comptime if dt.is_floating_point():
                assert_equal(isnan(quotients[i]), isnan(quotients_twin[i]))
                assert_equal(isnan(remainders[i]), isnan(remainders_twin[i]))
                if isnan(quotients[i]):
                    continue
            assert_equal(quotients[i], quotients_twin[i])
            if not isnan(remainders[i]):
                assert_equal(remainders[i], remainders_twin[i])


def test_a_float_quotient_of_an_infinity_is_a_nan() raises:
    """`inf // 2` is a NaN in numpy and an infinity in the expression everybody
    reaches for first, which is `floor(x / y)`. There is no floor of an infinite
    quotient, so numpy takes the remainder first and derives the quotient from
    it, and the remainder of an infinity is a NaN. A zero divisor is the other
    way round and stays an infinity, because that is a division and not a
    floor."""
    var huge = inf[DType.float64]()
    var top = from_list[DType.float64](
        [huge, -huge, 2.0, 2.0, 2.0, -2.0, huge, 0.0]
    )
    var bottom = from_list[DType.float64](
        [2.0, 2.0, huge, 0.0, -0.0, 0.0, huge, 0.0]
    )

    var quotients = floor_divide(top, bottom)
    assert_true(isnan(quotients[0]))
    assert_true(isnan(quotients[1]))
    assert_equal(quotients[2], 0.0)
    assert_true(isinf(quotients[3]))
    assert_true(quotients[3] > 0.0)
    assert_true(isinf(quotients[4]))
    assert_true(quotients[4] < 0.0)
    assert_true(isinf(quotients[5]))
    assert_true(quotients[5] < 0.0)
    assert_true(isnan(quotients[6]))
    assert_true(isnan(quotients[7]))

    var remainders = modulo(top, bottom)
    assert_true(isnan(remainders[0]))
    assert_true(isnan(remainders[1]))
    assert_equal(remainders[2], 2.0)
    assert_true(isnan(remainders[3]))
    assert_true(isnan(remainders[4]))
    assert_true(isnan(remainders[5]))
    assert_true(isnan(remainders[6]))
    assert_true(isnan(remainders[7]))


def test_a_float_remainder_takes_the_sign_of_the_divisor() raises:
    """The Python rule rather than the C one, which is the same rule the integer
    dtypes already follow, and the one place the two differ is a remainder of
    zero: a float has a negative zero and numpy gives it the divisor's sign."""
    var top = from_list[DType.float64]([7.0, -7.0, 7.0, -7.0, 6.0, 6.0])
    var bottom = from_list[DType.float64]([3.0, 3.0, -3.0, -3.0, 3.0, -3.0])

    var remainders = modulo(top, bottom)
    assert_equal(remainders[0], 1.0)
    assert_equal(remainders[1], 2.0)
    assert_equal(remainders[2], -2.0)
    assert_equal(remainders[3], -1.0)
    assert_equal(remainders[4], 0.0)
    assert_equal(remainders[5], 0.0)
    # The sign of a zero does not show up in an equality, so it is read off the
    # reciprocal, which is an infinity pointing the way the zero did.
    assert_true(1.0 / remainders[4] > 0.0)
    assert_true(1.0 / remainders[5] < 0.0)

    var quotients = floor_divide(top, bottom)
    assert_equal(quotients[0], 2.0)
    assert_equal(quotients[1], -3.0)
    assert_equal(quotients[2], -3.0)
    assert_equal(quotients[3], 2.0)


def test_a_float_remainder_stays_exact_past_the_whole_integers() raises:
    """This is the failure that matters and it is not about infinities. A float
    can hold every integer up to 2 to the 53 and only some of them after that,
    so `x - floor(x / y) * y` has no digits left the moment the quotient passes
    that line. 2 to the 53 is about 9.0e15, which a nanosecond timestamp passed
    in the spring of 1970, so this is an ordinary range and not a corner."""
    var top = from_list[DType.float64](
        [9007199254740992.0, 18014398509481984.0, 1e17, 1.7976931348623157e308]
    )
    var bottom = from_list[DType.float64]([3.0, 3.0, 3.0, 3.0])

    var remainders = modulo(top, bottom)
    assert_equal(remainders[0], 2.0)
    assert_equal(remainders[1], 1.0)
    assert_equal(remainders[2], 1.0)
    assert_equal(remainders[3], 2.0)


def test_a_float32_remainder_stays_exact_past_its_own_line() raises:
    """The same rule at the other width. A float32 holds every integer up to 2
    to the 24, which is 16777216, so the line is thirty two million times
    earlier and the failure is that much easier to reach."""
    var top = from_list[DType.float32](
        [16777216.0, 33554432.0, 1e17, 3.4028235e38]
    )
    var bottom = from_list[DType.float32]([3.0, 3.0, 3.0, 3.0])

    var remainders = modulo(top, bottom)
    assert_equal(remainders[0], 1.0)
    assert_equal(remainders[1], 2.0)
    assert_equal(remainders[2], 1.0)
    assert_equal(remainders[3], 0.0)


def test_the_power_matches_the_twin() raises:
    """The exponents stay under four so that the answer is a number rather than
    an overflow on int8, which would be testing the wrap rather than the loop.
    """
    comptime for dt in NUMERIC:
        var a = build[dt](193, 5)
        var b = Array[dt](193)
        for i in range(193):
            b[i] = Scalar[dt](i % 4)
        b.set_null(11)

        var powers = power(a, b)
        var powers_twin = power_scalar(a, b)
        for i in range(193):
            assert_equal(powers.is_valid(i), powers_twin.is_valid(i))
            assert_equal(powers[i], powers_twin[i])


def test_a_negative_exponent_stops_the_integer_loops_and_not_the_float_ones() raises:
    var bases = from_list[DType.int64]([2, 3, 4])
    var down = from_list[DType.int64]([1, -1, 1])
    with assert_raises(contains="negative integer powers"):
        _ = power(bases, down)
    with assert_raises(contains="negative integer powers"):
        _ = power_scalar(bases, down)

    var floats = from_list[DType.float64]([2.0, 4.0])
    var negative = from_list[DType.float64]([-1.0, -0.5])
    var got = power(floats, negative)
    assert_equal(got[0], 0.5)
    assert_equal(got[1], 0.5)


def test_the_unary_operations_match_the_twin() raises:
    """Every sign in the column is flipped on the signed dtypes first, because
    `build` hands back positive values only and a negation that was silently a
    copy would pass on those. The inversion is skipped on the float dtypes,
    where both sides refuse it rather than answering.
    """
    comptime for dt in NUMERIC:
        var a = build[dt](193, 5)
        comptime if dt.is_signed():
            for i in range(0, 193, 3):
                if a.is_valid(i):
                    a.set_valid(i, -a[i])

        var negated = negate(a)
        var negated_twin = negate_scalar(a)
        var magnitude = absolute(a)
        var magnitude_twin = absolute_scalar(a)
        for i in range(193):
            assert_equal(negated.is_valid(i), negated_twin.is_valid(i))
            assert_equal(negated[i], negated_twin[i])
            assert_equal(magnitude.is_valid(i), magnitude_twin.is_valid(i))
            assert_equal(magnitude[i], magnitude_twin[i])

        comptime if not dt.is_floating_point():
            var inverted = invert(a)
            var inverted_twin = invert_scalar(a)
            for i in range(193):
                assert_equal(inverted.is_valid(i), inverted_twin.is_valid(i))
                assert_equal(inverted[i], inverted_twin[i])


def test_subtraction_matches_the_twin_on_signed_values() raises:
    var a = build[DType.int64](193, 5)
    var b = build[DType.int64](193, 7)

    var diffs = subtract(a, b)
    var diffs_twin = subtract_scalar(a, b)
    for i in range(193):
        assert_equal(diffs.is_valid(i), diffs_twin.is_valid(i))
        assert_equal(diffs[i], diffs_twin[i])


def test_arithmetic_nulls_where_either_side_is_null() raises:
    var a = from_list[DType.int64]([1, 2, 3, 4])
    var b = from_list[DType.int64]([10, 20, 30, 40])
    a.set_null(1)
    b.set_null(2)

    var sums = add(a, b)
    assert_true(sums.is_valid(0))
    assert_equal(sums[0], 11)
    assert_false(sums.is_valid(1))
    assert_false(sums.is_valid(2))
    assert_true(sums.is_valid(3))
    assert_equal(sums[3], 44)

    # The null positions hold zero, not the arithmetic the kernel computed on the
    # way past. That is what keeps `sum_of` allowed to skip the bitmap.
    assert_equal(sums[1], 0)
    assert_equal(sums[2], 0)
    assert_equal(sum_of(sums).value, 55)


def test_division_is_float_and_does_not_null_on_zero() raises:
    var a = from_list[DType.int64]([7, 1, -1, 0])
    var b = from_list[DType.int64]([2, 0, 0, 0])

    var quotients = divide(a, b)
    assert_almost_equal(quotients[0], 3.5)

    # Dividing by zero is IEEE, not an error and not a null. pandas does the same
    # and a query that wants nulls there can say so with a mask.
    assert_true(quotients.is_valid(1))
    assert_true(Bool(isinf(quotients[1])))
    assert_true(Bool(quotients[1].gt(0)))
    assert_true(Bool(isinf(quotients[2])))
    assert_true(Bool(quotients[2].lt(0)))
    assert_true(Bool(isnan(quotients[3])))


def test_comparison_matches_the_twin() raises:
    comptime for dt in NUMERIC:
        var a = build[dt](193, 5)
        var b = build[dt](193, 7)

        var eq = equal(a, b)
        var eq_twin = equal_scalar(a, b)
        for i in range(193):
            assert_equal(eq.is_valid(i), eq_twin.is_valid(i))
            assert_equal(eq[i], eq_twin[i])


def assert_bools(actual: Array[DType.bool], expected: List[Bool]) raises:
    """Asserts that a boolean column holds the expected values, all present.

    Args:
        actual: The column under test.
        expected: What it should hold.

    Raises:
        If the length, the validity or any value differs.
    """
    assert_equal(len(actual), len(expected))
    for i in range(len(expected)):
        assert_true(actual.is_valid(i))
        assert_equal(Bool(actual[i]), expected[i])


def test_comparison_covers_the_six_operators() raises:
    var a = from_list[DType.int64]([1, 5, 5, 9])
    var b = from_list[DType.int64]([2, 5, 4, 3])

    assert_bools(equal(a, b), [False, True, False, False])
    assert_bools(not_equal(a, b), [True, False, True, True])
    assert_bools(less(a, b), [True, False, False, False])
    assert_bools(less_equal(a, b), [True, True, False, False])
    assert_bools(greater(a, b), [False, False, True, True])


def test_comparison_against_a_null_is_null() raises:
    var a = from_list[DType.int64]([1, 2, 3])
    var b = from_list[DType.int64]([1, 2, 3])
    b.set_null(1)

    var eq = equal(a, b)
    assert_true(eq.is_valid(0))
    assert_true(eq[0])
    assert_false(eq.is_valid(1))
    # Null, not false. The value under it is false because that is what the
    # invariant requires, which is exactly why the validity bit is the answer.
    assert_false(eq[1])


def test_reductions_past_the_split_agree_with_one_thread() raises:
    # A reduction does not split the way an elementwise kernel splits, because
    # there is one answer rather than one per row, so the morsel boundaries are
    # worth their own test. The length is a prime past three morsels, which
    # leaves the last one short. The interesting rows are the nulls: a minimum
    # carries a flag per morsel saying whether that morsel saw a value at all,
    # and a morsel that is entirely null has to leave the identity out of the
    # answer rather than reporting it.
    comptime rows = 393_241
    var col = build[DType.int64](rows, 9)

    # One whole morsel in the middle blanked, so at least one worker comes back
    # with nothing and the combine has to skip it. The first morsel is left
    # alone so the answer is still valid.
    for i in range(131_072, 262_144):
        col.set_null(i)

    var expected_sum = Scalar[accumulator(DType.int64)](0)
    var expected_min = Int64(0)
    var expected_max = Int64(0)
    var seen = False
    for i in range(rows):
        expected_sum += Scalar[accumulator(DType.int64)](col[i])
        if not col.is_valid(i):
            continue
        if not seen:
            seen = True
            expected_min = col[i]
            expected_max = col[i]
            continue
        if col[i] < expected_min:
            expected_min = col[i]
        if col[i] > expected_max:
            expected_max = col[i]

    var total = sum_of(col)
    assert_true(total.valid)
    assert_equal(total.value, expected_sum)

    var low = min_of(col)
    assert_true(low.valid)
    assert_equal(low.value, expected_min)

    var high = max_of(col)
    assert_true(high.valid)
    assert_equal(high.value, expected_max)


def test_a_reduction_over_nothing_but_nulls_past_the_split_is_invalid() raises:
    # Every morsel comes back empty, so the combine never sees a value and has to
    # say so rather than returning the identity it started with. A sum is the one
    # that disagrees, because a sum over nothing is zero and valid.
    comptime rows = 262_144
    var col = build[DType.int64](rows, 1)

    assert_false(min_of(col).valid)
    assert_false(max_of(col).valid)
    assert_false(mean_of(col).valid)
    assert_true(sum_of(col).valid)
    assert_equal(sum_of(col).value, 0)


def test_arithmetic_and_comparison_past_the_split_are_right_everywhere() raises:
    # Same boundary question the cast test asks, and the answer has to hold for
    # these too because they split over morsels the same way. The extra thing
    # here is the nulls: the loop computes over them and `apply_validity` blanks
    # them afterwards, and that repair walks the whole column on one thread
    # after the workers are done, so a row the workers got wrong under a null
    # would be hidden. The check is against the scalar answer at the present
    # rows and against zero at the absent ones.
    comptime rows = 393_241
    var a = build[DType.int64](rows, 7)
    var b = build[DType.int64](rows, 11)

    var sums = add(a, b)
    var lt = less(a, b)
    var shifted = arith_const[DType.int64, OP_ADD](a, 3)

    var wrong = -1
    for i in range(rows):
        var present = a.is_valid(i) and b.is_valid(i)
        if sums.is_valid(i) != present or lt.is_valid(i) != present:
            wrong = i
            break
        if present:
            if sums[i] != a[i] + b[i] or Bool(lt[i]) != (a[i] < b[i]):
                wrong = i
                break
        elif sums[i] != 0 or Bool(lt[i]):
            wrong = i
            break
        if shifted.is_valid(i) != a.is_valid(i):
            wrong = i
            break
        if a.is_valid(i) and shifted[i] != a[i] + 3:
            wrong = i
            break
    assert_equal(wrong, -1, "a row past the morsel split is wrong")


def test_cast_matches_the_twin_and_keeps_the_nulls() raises:
    var col = build[DType.int64](193, 5)
    var narrowed = cast_to[DType.int64, DType.int16](col)
    var narrowed_twin = cast_scalar[DType.int64, DType.int16](col)
    for i in range(193):
        assert_equal(narrowed.is_valid(i), narrowed_twin.is_valid(i))
        assert_equal(narrowed[i], narrowed_twin[i])

    var widened = cast_to[DType.int64, DType.float64](col)
    for i in range(193):
        assert_equal(widened.is_valid(i), col.is_valid(i))
        assert_almost_equal(widened[i], Float64(col[i]))


def test_a_cast_past_the_split_converts_every_row() raises:
    # The cast runs inline in one morsel and on every core past that, and the
    # loop steps a SIMD register at a time, so a worker whose morsel does not
    # end on a register boundary would write into the next worker's rows. A
    # morsel is a hundred and twenty eight thousand rows and the widest register
    # here is sixty four int8 lanes, so the boundaries divide; this is the test
    # that says so rather than the comment. The length is a prime past three
    # morsels, which puts the last one short and off every boundary at once.
    comptime rows = 393_241
    var col = build[DType.int64](rows, 7)

    var narrowed = cast_to[DType.int64, DType.int8](col)
    var widened = cast_to[DType.int64, DType.float64](col)
    assert_equal(len(narrowed), rows)
    assert_equal(len(widened), rows)

    var wrong = -1
    for i in range(rows):
        if narrowed[i] != Int8(col[i]) or widened[i] != Float64(col[i]):
            wrong = i
            break
    assert_equal(wrong, -1, "a converted row past the morsel split is wrong")


def test_take_matches_the_twin() raises:
    var col = build[DType.int64](64, 5)
    var picks = List[Int]()
    for i in range(200):
        picks.append((i * 13) % 64)
    picks.append(-1)
    picks.append(0)

    var taken = take_rows(col, picks)
    var taken_twin = take_scalar(col, picks)
    assert_equal(len(taken), len(picks))
    for i in range(len(picks)):
        assert_equal(taken.is_valid(i), taken_twin.is_valid(i))
        assert_equal(taken[i], taken_twin[i])


def first_wrong(
    got: Array[DType.int64], source: Array[DType.int64], picks: List[Int]
) -> Int:
    """Returns the first gathered row that is not what the source says, or -1.

    The scalar twin would answer the same question and would build a whole
    column to do it, and an assertion per row would format a message for each of
    sixty five thousand of them. These two tests have to gather that many rows to
    reach the length where the split turns on, so the check is written to cost
    less than the kernel it is checking.
    """
    for i in range(len(picks)):
        var at = picks[i]
        var want = at >= 0 and source.is_valid(at)
        if got.is_valid(i) != want:
            return i
        if want and got[i] != source[at]:
            return i
    return -1


def test_a_take_past_the_split_gathers_the_right_rows() raises:
    # The gather runs on one thread below `PARALLEL_TAKE_ROWS` and on every core
    # above it, and the workers share the validity bitmap, one word per sixty
    # four output rows. So the morsel boundaries have to land on word
    # boundaries, and a length that is not a multiple of sixty four is what
    # catches the last morsel keeping a partial word it never stored. Both are
    # out of reach of the short columns the other take tests use. The length is
    # one past a multiple of sixty four so that the tail is a partial word.
    var col = build[DType.int64](4096, 7)
    var picks = List[Int](capacity=65_601)
    for i in range(65_601):
        picks.append((i * 4093) % 4096)
    picks[64] = -1
    picks[65] = -1
    picks[65_600] = -1

    var taken = take_rows(col, picks)
    assert_equal(len(taken), len(picks))
    assert_equal(first_wrong(taken, col, picks), -1, "a gathered row is wrong")


def test_a_take_past_the_split_from_a_column_with_no_nulls() raises:
    # The validity probe is skipped outright when the source has none, so that
    # arm needs walking at this length too.
    var col = build[DType.int64](4096, 0)
    var picks = List[Int](capacity=65_601)
    for i in range(65_601):
        picks.append(-1 if i % 1000 == 0 else (i * 4093) % 4096)

    var taken = take_rows(col, picks)
    assert_equal(first_wrong(taken, col, picks), -1, "a gathered row is wrong")


def test_take_turns_a_negative_index_into_a_null() raises:
    var col = from_list[DType.int64]([10, 20, 30])
    var taken = take_rows(col, [2, -1, 0])
    assert_equal(len(taken), 3)
    assert_equal(taken[0], 30)
    assert_false(taken.is_valid(1))
    assert_equal(taken[1], 0)
    assert_equal(taken[2], 10)


def test_filter_matches_the_twin() raises:
    var col = build[DType.int64](193, 5)
    var other = build[DType.int64](193, 7)
    var mask = greater(col, other)

    var kept = filter_rows(col, mask)
    var kept_twin = filter_scalar(col, mask)
    assert_equal(len(kept), len(kept_twin))
    for i in range(len(kept)):
        assert_equal(kept.is_valid(i), kept_twin.is_valid(i))
        assert_equal(kept[i], kept_twin[i])


def test_filter_drops_the_rows_the_mask_is_null_at() raises:
    var col = from_list[DType.int64]([10, 20, 30, 40])
    var mask = from_list[DType.bool]([True, True, False, True])
    mask.set_null(1)

    var kept = filter_rows(col, mask)
    assert_equal(len(kept), 2)
    assert_equal(kept[0], 10)
    assert_equal(kept[1], 40)


def test_filter_carries_the_nulls_of_the_values_across() raises:
    var col = from_list[DType.int64]([10, 20, 30, 40])
    col.set_null(2)
    var mask = from_list[DType.bool]([False, True, True, True])

    var kept = filter_rows(col, mask)
    assert_equal(len(kept), 3)
    assert_equal(kept[0], 20)
    assert_false(kept.is_valid(1))
    assert_equal(kept[1], 0)
    assert_equal(kept[2], 40)


def test_slice_copies_values_and_validity() raises:
    var col = build[DType.int64](200, 5)
    var piece = col.slice(64, 130)
    assert_equal(len(piece), 66)
    for i in range(66):
        assert_equal(piece.is_valid(i), col.is_valid(64 + i))
        assert_equal(piece[i], col[64 + i])


def test_slice_from_an_unaligned_start() raises:
    var col = build[DType.int64](200, 3)
    var piece = col.slice(7, 71)
    assert_equal(len(piece), 64)
    for i in range(64):
        assert_equal(piece.is_valid(i), col.is_valid(7 + i))
        assert_equal(piece[i], col[7 + i])


def test_bitmap_word_accessors_agree_with_the_bits() raises:
    var col = build[DType.int64](200, 5)
    assert_equal(col.data.validity.word_count(), 4)
    for w in range(4):
        var word = col.data.validity.unsafe_word(w)
        for bit in range(64):
            var at = w * 64 + bit
            if at >= 200:
                assert_equal((word >> UInt64(bit)) & 1, 0)
            else:
                var set = ((word >> UInt64(bit)) & 1) != 0
                assert_equal(set, col.is_valid(at))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
