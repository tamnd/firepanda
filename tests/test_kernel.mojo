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

from std.math import isinf, isnan
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)

from firepanda.array.array import Array, from_list
from firepanda.dtype.lists import NUMERIC
from firepanda.kernel import (
    add,
    cast_to,
    count_of,
    divide,
    equal,
    filter_rows,
    greater,
    less,
    less_equal,
    max_of,
    mean_of,
    min_of,
    multiply,
    not_equal,
    subtract,
    sum_of,
    take_rows,
)
from firepanda.kernel.accum import accumulator
from firepanda.kernel.arith import OP_ADD, arith_const
from firepanda.kernel.scalar import (
    add_scalar,
    cast_scalar,
    equal_scalar,
    filter_scalar,
    max_scalar,
    mean_scalar,
    min_scalar,
    multiply_scalar,
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
