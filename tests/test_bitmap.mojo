"""Tests for the validity bitmap.

Most of these are about the tail. A bitmap of length 5 occupies one byte, and the
three bits past the end have to stay zero, because `count_ones` is a popcount over
whole bytes and would otherwise count bits that do not correspond to a row. Every
mutator is checked for that, including the ones where it is not obvious the tail
is touched at all, such as `invert` and `or_with`.

The exhaustive coverage of this type lives in tests/fuzz/main.mojo, which runs the
same operations against a `List[Bool]` reference for ten million cases.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.bitmap.bitmap import Bitmap, bytes_for


def test_bytes_for() raises:
    assert_equal(bytes_for(0), 0)
    assert_equal(bytes_for(1), 1)
    assert_equal(bytes_for(8), 1)
    assert_equal(bytes_for(9), 2)
    assert_equal(bytes_for(64), 8)
    assert_equal(bytes_for(65), 9)


def test_defaults_to_all_valid() raises:
    var bitmap = Bitmap(10)
    assert_equal(len(bitmap), 10)
    assert_equal(bitmap.count_ones(), 10)
    assert_equal(bitmap.null_count(), 0)
    assert_true(bitmap.all_valid())
    for i in range(10):
        assert_true(bitmap.get(i))


def test_can_start_all_null() raises:
    var bitmap = Bitmap(10, all_valid=False)
    assert_equal(bitmap.count_ones(), 0)
    assert_equal(bitmap.null_count(), 10)
    assert_false(bitmap.any_valid())


def test_set_and_get() raises:
    var bitmap = Bitmap(20, all_valid=False)
    bitmap.set(0, True)
    bitmap.set(7, True)
    bitmap.set(8, True)
    bitmap.set(19, True)
    assert_equal(bitmap.count_ones(), 4)
    for i in range(20):
        var expected = i == 0 or i == 7 or i == 8 or i == 19
        assert_equal(bitmap.get(i), expected)

    bitmap.set(7, False)
    assert_false(bitmap.get(7))
    assert_equal(bitmap.count_ones(), 3)


def test_bit_order_is_arrow_lsb_first() raises:
    # Arrow numbers bits within a byte from the least significant end. Getting
    # this backwards would be invisible in Mojo and would corrupt every buffer
    # that crosses the Arrow boundary.
    var bitmap = Bitmap(8, all_valid=False)
    bitmap.set(0, True)
    assert_equal(bitmap.unsafe_ptr().unsafe_load(), UInt8(0b0000_0001))
    bitmap.set(7, True)
    assert_equal(bitmap.unsafe_ptr().unsafe_load(), UInt8(0b1000_0001))


def test_tail_bits_stay_clear() raises:
    for length in range(1, 65):
        var bitmap = Bitmap(length)
        assert_equal(bitmap.count_ones(), length)

        bitmap.set_all()
        assert_equal(bitmap.count_ones(), length)

        bitmap.invert()
        assert_equal(bitmap.count_ones(), 0)

        bitmap.invert()
        assert_equal(bitmap.count_ones(), length)

        bitmap.set_range(0, length, True)
        assert_equal(bitmap.count_ones(), length)


def test_set_range() raises:
    var bitmap = Bitmap(100, all_valid=False)
    bitmap.set_range(3, 70, True)
    assert_equal(bitmap.count_ones(), 67)
    for i in range(100):
        assert_equal(bitmap.get(i), i >= 3 and i < 70)

    bitmap.set_range(10, 20, False)
    assert_equal(bitmap.count_ones(), 57)
    for i in range(10, 20):
        assert_false(bitmap.get(i))


def test_set_range_within_one_byte() raises:
    var bitmap = Bitmap(8, all_valid=False)
    bitmap.set_range(2, 5, True)
    assert_equal(bitmap.unsafe_ptr().unsafe_load(), UInt8(0b0001_1100))


def test_empty_range_is_a_no_op() raises:
    var bitmap = Bitmap(16)
    bitmap.set_range(5, 5, False)
    assert_equal(bitmap.count_ones(), 16)


def test_and_with() raises:
    var left = Bitmap(20, all_valid=False)
    var right = Bitmap(20, all_valid=False)
    left.set_range(0, 15, True)
    right.set_range(10, 20, True)
    left.and_with(right)
    for i in range(20):
        assert_equal(left.get(i), i >= 10 and i < 15)


def test_or_with() raises:
    var left = Bitmap(20, all_valid=False)
    var right = Bitmap(20, all_valid=False)
    left.set_range(0, 5, True)
    right.set_range(15, 20, True)
    left.or_with(right)
    assert_equal(left.count_ones(), 10)
    for i in range(20):
        assert_equal(left.get(i), i < 5 or i >= 15)


def test_slice_byte_aligned() raises:
    var bitmap = Bitmap(64, all_valid=False)
    for i in range(64):
        if i % 3 == 0:
            bitmap.set(i, True)
    var part = bitmap.slice(8, 40)
    assert_equal(len(part), 32)
    for i in range(32):
        assert_equal(part.get(i), bitmap.get(i + 8))


def test_slice_unaligned() raises:
    var bitmap = Bitmap(64, all_valid=False)
    for i in range(64):
        if i % 5 == 1:
            bitmap.set(i, True)
    var part = bitmap.slice(3, 61)
    assert_equal(len(part), 58)
    for i in range(58):
        assert_equal(part.get(i), bitmap.get(i + 3))


def test_slice_leaves_the_source_alone() raises:
    var bitmap = Bitmap(32)
    var part = bitmap.slice(4, 12)
    part.clear_all()
    assert_equal(bitmap.count_ones(), 32)


def test_copy_is_deep() raises:
    var original = Bitmap(32)
    var duplicate = Bitmap(copy=original)
    duplicate.clear_all()
    assert_equal(original.count_ones(), 32)
    assert_equal(duplicate.count_ones(), 0)


def test_to_list() raises:
    var bitmap = Bitmap(5, all_valid=False)
    bitmap.set(1, True)
    bitmap.set(4, True)
    var values = bitmap.to_list()
    assert_equal(len(values), 5)
    assert_false(values[0])
    assert_true(values[1])
    assert_false(values[2])
    assert_false(values[3])
    assert_true(values[4])


def test_zero_length() raises:
    var bitmap = Bitmap(0)
    assert_equal(len(bitmap), 0)
    assert_equal(bitmap.count_ones(), 0)
    assert_equal(bitmap.byte_length(), 0)
    assert_true(bitmap.all_valid())
    assert_false(bitmap.any_valid())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
