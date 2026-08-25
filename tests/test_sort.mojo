"""Tests for the sort kernel.

Most of these check `argsort` against `argsort_scalar`, which is an insertion
sort comparing values with `<`. The two agree on the permutation itself, not just
on the sorted values, which is what makes them a stability check as well as an
ordering one.

The rest pin down the parts the twin cannot see. `sort_key` is the whole reason a
float can be radix sorted and it is three lines of bit manipulation, so it gets
tested directly on every boundary the IEEE layout has: the sign bit, negative
zero, the infinities and NaN. The twin compares with `<`, which orders none of
those against each other, so if they were not tested here they would not be
tested at all.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.kernel.sort import (
    argsort,
    argsort_into,
    argsort_multi,
    is_sorted,
    key_width,
    sort_key,
    sort_rows,
)
from firepanda.kernel.scalar import argsort_scalar
from firepanda.testing.rng import Rng


def check_against_twin[
    dt: DType
](col: Array[dt], descending: Bool, nulls_first: Bool, what: String) raises:
    """Asserts that `argsort` produces exactly what the twin produces.

    Comparing the permutations rather than the sorted values is deliberate. Two
    stable sorts of the same column agree on both; a stable sort and an unstable
    one agree only on the values.

    Args:
        col: The column.
        descending: Largest first.
        nulls_first: Nulls at the front.
        what: A label for the failure message.

    Raises:
        If the two disagree.
    """
    var order = argsort(col, descending, nulls_first)
    var twin = argsort_scalar(col, descending, nulls_first)
    assert_equal(len(order), len(twin), what + ": length")
    for i in range(len(twin)):
        assert_equal(
            Int(order[i]),
            twin[i],
            what + ": row " + String(i),
        )


def test_sort_key_orders_unsigned_values_by_widening() raises:
    assert_equal(sort_key(UInt8(0)), UInt64(0))
    assert_equal(sort_key(UInt8(255)), UInt64(255))
    assert_equal(sort_key(UInt64.MAX), UInt64.MAX)


def test_sort_key_moves_negatives_below_positives() raises:
    assert_true(sort_key(Int32(-1)) < sort_key(Int32(0)))
    assert_true(sort_key(Int32.MIN) < sort_key(Int32(-1)))
    assert_true(sort_key(Int32(0)) < sort_key(Int32.MAX))
    # The smallest signed value is the bottom of the key range, which is the
    # whole point of flipping the sign bit.
    assert_equal(sort_key(Int64.MIN), UInt64(0))


def test_sort_key_widens_narrow_signed_values_to_the_same_key() raises:
    # An int8 and an int64 holding the same number must produce the same key,
    # for the same reason `key_bits` does it in the hash function: the day a
    # sort spans two columns of different width, the keys have to line up.
    assert_equal(sort_key(Int8(-7)), sort_key(Int64(-7)))
    assert_equal(sort_key(Int16(120)), sort_key(Int64(120)))


def test_sort_key_orders_floats_across_the_sign_bit() raises:
    assert_true(sort_key(Float64(-1.0)) < sort_key(Float64(0.0)))
    assert_true(sort_key(Float64(0.0)) < sort_key(Float64(1.0)))
    assert_true(sort_key(Float64(-2.0)) < sort_key(Float64(-1.0)))
    assert_true(
        sort_key(Float64(-1.0) / Float64(0.0)) < sort_key(Float64(-1e300))
    )
    assert_true(
        sort_key(Float64(1e300)) < sort_key(Float64(1.0) / Float64(0.0))
    )


def test_sort_key_puts_negative_zero_below_positive_zero() raises:
    # `<` says these two are equal and the bits say they are not. numpy sorts
    # them this way and so do we; it is documented rather than accidental.
    assert_true(sort_key(Float64(-0.0)) < sort_key(Float64(0.0)))


def test_sort_key_puts_nan_above_every_finite_value() raises:
    var nan = Float64(0.0) / Float64(0.0)
    assert_true(sort_key(Float64(1e300)) < sort_key(nan))
    assert_true(sort_key(Float64(1.0) / Float64(0.0)) < sort_key(nan))


def test_sort_key_agrees_between_float32_and_float64() raises:
    assert_equal(sort_key(Float32(-1.5)), sort_key(Float64(-1.5)))
    assert_equal(sort_key(Float32(0.25)), sort_key(Float64(0.25)))


def test_key_width_is_the_dtype_width_for_unsigned_and_eight_otherwise() raises:
    assert_equal(key_width[DType.uint8](), 1)
    assert_equal(key_width[DType.uint32](), 4)
    assert_equal(key_width[DType.uint64](), 8)
    # Signed and float both land in the top bits of a widened key, so neither
    # gets the narrow discount.
    assert_equal(key_width[DType.int8](), 8)
    assert_equal(key_width[DType.float32](), 8)


def test_argsort_of_an_empty_column_is_empty() raises:
    var col = Array[DType.int32](0)
    assert_equal(len(argsort(col)), 0)


def test_argsort_of_one_row_is_that_row() raises:
    var col = from_list[DType.int32]([42])
    var order = argsort(col)
    assert_equal(len(order), 1)
    assert_equal(Int(order[0]), 0)


def test_argsort_orders_a_small_signed_column() raises:
    var col = from_list[DType.int32]([3, -1, 7, 0, -9])
    var order = argsort(col)
    assert_equal(Int(order[0]), 4)
    assert_equal(Int(order[1]), 1)
    assert_equal(Int(order[2]), 3)
    assert_equal(Int(order[3]), 0)
    assert_equal(Int(order[4]), 2)


def test_argsort_is_stable_on_ties() raises:
    # Every value is the same, so a stable sort is the identity and anything
    # else is not.
    var col = from_list[DType.uint8]([5, 5, 5, 5, 5, 5, 5])
    var order = argsort(col)
    for i in range(len(order)):
        assert_equal(Int(order[i]), i)


def test_argsort_is_stable_on_ties_when_descending() raises:
    # Reversing the array after an ascending sort would pass the previous test
    # and fail this one, which is why descending complements the key instead.
    var col = from_list[DType.uint8]([2, 1, 2, 1, 2, 1])
    var order = argsort(col, descending=True)
    assert_equal(Int(order[0]), 0)
    assert_equal(Int(order[1]), 2)
    assert_equal(Int(order[2]), 4)
    assert_equal(Int(order[3]), 1)
    assert_equal(Int(order[4]), 3)
    assert_equal(Int(order[5]), 5)


def test_argsort_puts_nulls_last_by_default() raises:
    var col = from_list[DType.int32]([5, 0, 3, 0, 1])
    col.set_null(1)
    col.set_null(3)
    var order = argsort(col)
    assert_equal(Int(order[0]), 4)
    assert_equal(Int(order[1]), 2)
    assert_equal(Int(order[2]), 0)
    # The nulls keep their input order among themselves.
    assert_equal(Int(order[3]), 1)
    assert_equal(Int(order[4]), 3)


def test_argsort_puts_nulls_first_when_asked() raises:
    var col = from_list[DType.int32]([5, 0, 3, 0, 1])
    col.set_null(1)
    col.set_null(3)
    var order = argsort(col, nulls_first=True)
    assert_equal(Int(order[0]), 1)
    assert_equal(Int(order[1]), 3)
    assert_equal(Int(order[2]), 4)
    assert_equal(Int(order[3]), 2)
    assert_equal(Int(order[4]), 0)


def test_argsort_of_an_all_null_column_is_the_identity() raises:
    var col = Array[DType.int64](6)
    for i in range(6):
        col.set_null(i)
    var order = argsort(col)
    for i in range(6):
        assert_equal(Int(order[i]), i)


def test_argsort_orders_floats_including_the_specials() raises:
    var nan = Float64(0.0) / Float64(0.0)
    var high = Float64(1.0) / Float64(0.0)
    var low = Float64(-1.0) / Float64(0.0)
    var col = Array[DType.float64](6)
    col.set_valid(0, Float64(1.0))
    col.set_valid(1, nan)
    col.set_valid(2, low)
    col.set_valid(3, Float64(0.0))
    col.set_valid(4, high)
    col.set_valid(5, Float64(-2.5))
    var order = argsort(col)
    assert_equal(Int(order[0]), 2)
    assert_equal(Int(order[1]), 5)
    assert_equal(Int(order[2]), 3)
    assert_equal(Int(order[3]), 0)
    assert_equal(Int(order[4]), 4)
    assert_equal(Int(order[5]), 1)


def test_argsort_orders_bools() raises:
    var col = from_list[DType.bool]([True, False, True, False])
    var order = argsort(col)
    assert_equal(Int(order[0]), 1)
    assert_equal(Int(order[1]), 3)
    assert_equal(Int(order[2]), 0)
    assert_equal(Int(order[3]), 2)


def test_argsort_spans_the_full_uint64_range() raises:
    # Eight radix passes, none of them skippable, which is the case that proves
    # the ping pong between the two buffers ends up reading the right one.
    var col = from_list[DType.uint64](
        [
            UInt64.MAX,
            0,
            UInt64(1) << 63,
            UInt64(1) << 32,
            UInt64(1) << 8,
            UInt64(1),
        ]
    )
    var order = argsort(col)
    assert_equal(Int(order[0]), 1)
    assert_equal(Int(order[1]), 5)
    assert_equal(Int(order[2]), 4)
    assert_equal(Int(order[3]), 3)
    assert_equal(Int(order[4]), 2)
    assert_equal(Int(order[5]), 0)


def test_argsort_agrees_with_the_twin_on_random_columns() raises:
    var rng = Rng(0x5EED)
    for shape in range(4):
        for trial in range(24):
            var n = 1 + (rng.next_below(200))
            var col = Array[DType.int32](n)
            for i in range(n):
                col.set_valid(i, Int32((rng.next_below(64)) - 32))
                if shape > 0 and rng.next_below(shape + 1) == 0:
                    col.set_null(i)
            check_against_twin(
                col,
                descending=trial % 2 == 1,
                nulls_first=trial % 4 >= 2,
                what="shape " + String(shape) + " trial " + String(trial),
            )


def test_argsort_agrees_with_the_twin_across_dtypes() raises:
    var rng = Rng(0xC0FFEE)
    for trial in range(16):
        var n = 1 + (rng.next_below(150))

        var small = Array[DType.uint8](n)
        var wide = Array[DType.int64](n)
        var real = Array[DType.float32](n)
        for i in range(n):
            var draw = rng.next_below(1000)
            small.set_valid(i, UInt8(Int(draw) & 0xFF))
            wide.set_valid(i, Int64(Int(draw) - 500))
            real.set_valid(i, Float32(Int(draw) - 500) / Float32(8.0))

        var label = "trial " + String(trial)
        check_against_twin(small, False, False, label + " uint8")
        check_against_twin(wide, True, False, label + " int64")
        check_against_twin(real, False, True, label + " float32")


def test_sort_rows_returns_the_values_in_order() raises:
    var col = from_list[DType.int16]([4, -2, 9, 0])
    var out = sort_rows(col)
    assert_equal(Int(out[0]), -2)
    assert_equal(Int(out[1]), 0)
    assert_equal(Int(out[2]), 4)
    assert_equal(Int(out[3]), 9)
    assert_equal(out.null_count(), 0)


def test_sort_rows_carries_validity_to_the_right_end() raises:
    var col = from_list[DType.int16]([4, 0, 9, 0])
    col.set_null(1)
    col.set_null(3)
    var out = sort_rows(col)
    assert_true(out.is_valid(0))
    assert_true(out.is_valid(1))
    assert_false(out.is_valid(2))
    assert_false(out.is_valid(3))
    assert_equal(Int(out[0]), 4)
    assert_equal(Int(out[1]), 9)


def test_sort_rows_crosses_a_validity_word_boundary() raises:
    # Ninety rows means two validity words and a partial second one, which is
    # the case a fixed length of sixty four would miss.
    var col = Array[DType.int32](90)
    for i in range(90):
        col.set_valid(i, Int32(90 - i))
        if i % 7 == 0:
            col.set_null(i)
    var out = sort_rows(col)
    assert_equal(out.null_count(), col.null_count())
    assert_true(is_sorted(out))


def test_is_sorted_reads_what_argsort_wrote() raises:
    var rng = Rng(0xA11CE)
    for _ in range(8):
        var n = 1 + (rng.next_below(120))
        var col = Array[DType.int64](n)
        for i in range(n):
            col.set_valid(i, Int64((rng.next_below(1000)) - 500))
            if i % 5 == 0:
                col.set_null(i)
        assert_true(is_sorted(sort_rows(col)))
        assert_true(is_sorted(sort_rows(col, descending=True), descending=True))
        assert_true(
            is_sorted(sort_rows(col, nulls_first=True), nulls_first=True)
        )


def test_is_sorted_rejects_a_column_that_is_not() raises:
    assert_false(is_sorted(from_list[DType.int32]([1, 3, 2])))
    assert_false(is_sorted(from_list[DType.int32]([3, 2, 1])))
    assert_true(is_sorted(from_list[DType.int32]([3, 2, 1]), descending=True))


def test_is_sorted_rejects_a_null_in_the_wrong_place() raises:
    var col = from_list[DType.int32]([0, 1, 2])
    col.set_null(0)
    assert_false(is_sorted(col))
    assert_true(is_sorted(col, nulls_first=True))


def test_argsort_into_refines_rather_than_replaces() raises:
    # The second sort only moves rows its own key distinguishes. Both zeros keep
    # the order the first sort gave them, which is what the multi-key sort is
    # built on.
    var minor = from_list[DType.int32]([2, 1, 4, 3])
    var major = from_list[DType.int32]([0, 0, 1, 1])
    var order = argsort(minor)
    argsort_into(major, order)
    assert_equal(Int(order[0]), 1)
    assert_equal(Int(order[1]), 0)
    assert_equal(Int(order[2]), 3)
    assert_equal(Int(order[3]), 2)


def test_argsort_multi_makes_the_first_key_dominant() raises:
    var cols = List[AnyArray]()
    cols.append(AnyArray(from_list[DType.int32]([1, 1, 0, 0])))
    cols.append(AnyArray(from_list[DType.int32]([9, 2, 5, 5])))
    var order = argsort_multi(cols, [False, False], [False, False])
    assert_equal(Int(order[0]), 2)
    assert_equal(Int(order[1]), 3)
    assert_equal(Int(order[2]), 1)
    assert_equal(Int(order[3]), 0)


def test_argsort_multi_takes_a_direction_per_key() raises:
    var cols = List[AnyArray]()
    cols.append(AnyArray(from_list[DType.int32]([1, 1, 0, 0])))
    cols.append(AnyArray(from_list[DType.int32]([9, 2, 5, 4])))
    var order = argsort_multi(cols, [False, True], [False, False])
    assert_equal(Int(order[0]), 2)
    assert_equal(Int(order[1]), 3)
    assert_equal(Int(order[2]), 0)
    assert_equal(Int(order[3]), 1)


def test_argsort_multi_mixes_dtypes_across_keys() raises:
    var cols = List[AnyArray]()
    cols.append(AnyArray(from_list[DType.uint8]([1, 1, 0])))
    cols.append(AnyArray(from_list[DType.float64]([2.5, -1.0, 7.0])))
    var order = argsort_multi(cols, [False, False], [False, False])
    assert_equal(Int(order[0]), 2)
    assert_equal(Int(order[1]), 1)
    assert_equal(Int(order[2]), 0)


def test_argsort_multi_places_nulls_per_key() raises:
    var major = from_list[DType.int32]([0, 0, 0])
    var minor = from_list[DType.int32]([7, 0, 3])
    minor.set_null(1)
    var cols = List[AnyArray]()
    cols.append(AnyArray(major^))
    cols.append(AnyArray(minor^))
    var order = argsort_multi(cols, [False, False], [False, True])
    assert_equal(Int(order[0]), 1)
    assert_equal(Int(order[1]), 2)
    assert_equal(Int(order[2]), 0)


def test_argsort_multi_rejects_a_mismatched_flag_list() raises:
    var cols = List[AnyArray]()
    cols.append(AnyArray(from_list[DType.int32]([1, 2])))
    var raised = False
    try:
        _ = argsort_multi(cols, [False, False], [False])
    except:
        raised = True
    assert_true(raised, "a wrong length flag list should raise")


def test_argsort_multi_rejects_columns_of_different_lengths() raises:
    var cols = List[AnyArray]()
    cols.append(AnyArray(from_list[DType.int32]([1, 2])))
    cols.append(AnyArray(from_list[DType.int32]([1, 2, 3])))
    var raised = False
    try:
        _ = argsort_multi(cols, [False, False], [False, False])
    except:
        raised = True
    assert_true(raised, "ragged key columns should raise")


def test_argsort_multi_rejects_an_empty_key_list() raises:
    var cols = List[AnyArray]()
    var raised = False
    try:
        _ = argsort_multi(cols, [], [])
    except:
        raised = True
    assert_true(raised, "no key columns should raise")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
