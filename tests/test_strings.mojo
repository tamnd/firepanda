"""Tests for the variable width string column.

The interesting boundary is twelve bytes, because that is where an element stops
living inside its own view and moves to the payload, and almost every bug this
layout can have is one element on the wrong side of it. So most of what follows
is written twice, once short and once long, and there is a test that walks every
length from zero to thirty.

The other thing worth testing hard is that a null reads as empty rather than as
whatever was in the buffer, since the builder writes no bytes for a null and
relies on the allocation arriving zeroed.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.array.strview import INLINE_CAPACITY
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.logical import LogicalType
from firepanda.testing.rng import Rng


def repeated(var unit: String, times: Int) -> String:
    """Builds a string by repeating a shorter one.

    Args:
        unit: The string to repeat.
        times: How many copies.

    Returns:
        The repeated string.
    """
    var out = String()
    for _ in range(times):
        out += unit
    return out^


def test_an_empty_column_has_no_elements() raises:
    var column = StringBuilder().finish()
    assert_equal(len(column), 0)
    assert_equal(column.null_count(), 0)


def test_a_short_element_lives_in_its_view() raises:
    var column = strings_from_list(["ab", "cde"])
    assert_true(column.view(0).is_inline())
    assert_equal(column[0], "ab")
    assert_equal(column[1], "cde")


def test_twelve_bytes_is_still_inline() raises:
    var column = strings_from_list([repeated("x", INLINE_CAPACITY)])
    assert_true(column.view(0).is_inline())
    assert_equal(column.byte_length(0), INLINE_CAPACITY)


def test_thirteen_bytes_goes_to_the_payload() raises:
    var text = repeated("x", INLINE_CAPACITY + 1)
    var column = strings_from_list([text])
    assert_false(column.view(0).is_inline())
    assert_equal(column[0], text)


def test_every_length_from_zero_to_thirty_round_trips() raises:
    var values = List[String]()
    for n in range(31):
        var text = String()
        for k in range(n):
            text += chr(97 + k % 26)
        values.append(text^)
    var column = strings_from_list(values)
    for n in range(31):
        assert_equal(column.byte_length(n), n, "length " + String(n))
        assert_equal(column[n], values[n], "length " + String(n))


def test_a_null_is_empty_and_not_valid() raises:
    var builder = StringBuilder()
    builder.append(String("present").as_bytes())
    builder.append_null()
    var column = builder^.finish()
    assert_true(column.is_valid(0))
    assert_false(column.is_valid(1))
    assert_equal(column.byte_length(1), 0)
    assert_equal(column[1], "")
    assert_equal(column.null_count(), 1)


def test_a_null_after_a_long_element_is_still_empty() raises:
    # The null's view has to be zero rather than whatever the previous element
    # left, which is the failure this catches.
    var builder = StringBuilder()
    builder.append(String("a string well past the inline limit").as_bytes())
    builder.append_null()
    var column = builder^.finish()
    assert_equal(column[1], "")


def test_the_column_reports_the_string_type() raises:
    var column = strings_from_list(["a"])
    assert_true(column.dtype() == LogicalType.STRING)


def test_equals_against_bytes() raises:
    var long_text = repeated("q", 40)
    var column = strings_from_list(["ab", long_text])
    assert_true(column.equals(0, String("ab").as_bytes()))
    assert_false(column.equals(0, String("abc").as_bytes()))
    assert_false(column.equals(0, String("ac").as_bytes()))
    assert_true(column.equals(1, long_text.as_bytes()))
    assert_false(column.equals(1, repeated("q", 39).as_bytes()))


def test_equals_is_false_for_a_null() raises:
    var builder = StringBuilder()
    builder.append_null()
    var column = builder^.finish()
    assert_false(column.equals(0, String("").as_bytes()))


def test_element_equals_within_a_column() raises:
    var long_text = repeated("z", 30)
    var column = strings_from_list(["ab", "ab", long_text, long_text, "ac"])
    assert_true(column.element_equals(0, 1))
    assert_true(column.element_equals(2, 3))
    assert_false(column.element_equals(0, 4))
    assert_false(column.element_equals(0, 2))


def test_element_equals_separates_long_strings_by_prefix() raises:
    # Same length, differing only in the first four bytes, which is the case the
    # prefix in the view is there to settle without reading the payload.
    var left = "abcd" + repeated("s", 30)
    var right = "abce" + repeated("s", 30)
    var column = strings_from_list([left, right])
    assert_false(column.element_equals(0, 1))


def test_element_equals_separates_long_strings_after_the_prefix() raises:
    var left = repeated("s", 30) + "a"
    var right = repeated("s", 30) + "b"
    var column = strings_from_list([left, right])
    assert_false(column.element_equals(0, 1))


def test_element_equals_is_false_when_either_is_null() raises:
    var builder = StringBuilder()
    builder.append(String("a").as_bytes())
    builder.append_null()
    var column = builder^.finish()
    assert_false(column.element_equals(0, 1))
    assert_false(column.element_equals(1, 1))


def test_slice_copies_a_run() raises:
    var column = strings_from_list(["a", "b", repeated("c", 20), "d"])
    var cut = column.slice(1, 3)
    assert_equal(len(cut), 2)
    assert_equal(cut[0], "b")
    assert_equal(cut[1], repeated("c", 20))


def test_slice_keeps_nulls() raises:
    var builder = StringBuilder()
    builder.append(String("a").as_bytes())
    builder.append_null()
    builder.append(String("c").as_bytes())
    var column = builder^.finish()
    var cut = column.slice(1, 3)
    assert_false(cut.is_valid(0))
    assert_true(cut.is_valid(1))
    assert_equal(cut.null_count(), 1)


def test_slice_rejects_bad_bounds() raises:
    var column = strings_from_list(["a", "b"])
    var raised = False
    try:
        _ = column.slice(1, 5)
    except:
        raised = True
    assert_true(raised, "end past the column")

    raised = False
    try:
        _ = column.slice(2, 1)
    except:
        raised = True
    assert_true(raised, "reversed bounds")


def test_take_gathers_in_order_and_repeats() raises:
    var column = strings_from_list(["a", repeated("b", 20), "c"])
    var picked = column.take([2, 0, 1, 1])
    assert_equal(len(picked), 4)
    assert_equal(picked[0], "c")
    assert_equal(picked[1], "a")
    assert_equal(picked[2], repeated("b", 20))
    assert_equal(picked[3], repeated("b", 20))


def test_take_rejects_an_index_outside_the_column() raises:
    var column = strings_from_list(["a"])
    var raised = False
    try:
        _ = column.take([1])
    except:
        raised = True
    assert_true(raised, "index past the column")


def test_filter_keeps_the_set_bits() raises:
    var column = strings_from_list(["a", "b", repeated("c", 20), "d"])
    var mask = Bitmap(4)
    mask.clear_all()
    mask.set(1, True)
    mask.set(2, True)
    var kept = column.filter(mask)
    assert_equal(len(kept), 2)
    assert_equal(kept[0], "b")
    assert_equal(kept[1], repeated("c", 20))


def test_filter_carries_nulls_through() raises:
    var builder = StringBuilder()
    builder.append(String("a").as_bytes())
    builder.append_null()
    var column = builder^.finish()
    var mask = Bitmap(2)
    mask.set_all()
    var kept = column.filter(mask)
    assert_equal(len(kept), 2)
    assert_false(kept.is_valid(1))


def test_to_list_round_trips() raises:
    var values: List[String] = ["a", repeated("b", 40), ""]
    var column = strings_from_list(values)
    var back = column.to_list()
    assert_equal(len(back), 3)
    for i in range(3):
        assert_equal(back[i], values[i])


def test_a_copy_is_independent_of_the_payload() raises:
    var column = strings_from_list([repeated("m", 50)])
    var duplicate = StringArray(copy=column)
    assert_equal(duplicate[0], repeated("m", 50))
    assert_equal(len(duplicate), 1)


def test_many_random_elements_round_trip() raises:
    # Lengths drawn across the inline boundary in both directions, with nulls
    # mixed in, so the payload offsets have to stay in step with the views.
    var rng = Rng(0x5711)
    var values = List[String]()
    var present = List[Bool]()
    var builder = StringBuilder()
    for _ in range(2000):
        if rng.next_below(8) == 0:
            builder.append_null()
            values.append(String(""))
            present.append(False)
            continue
        var length = Int(rng.next_below(40))
        var text = String()
        for _ in range(length):
            text += chr(97 + Int(rng.next_below(26)))
        builder.append(text.as_bytes())
        values.append(text)
        present.append(True)
    var column = builder^.finish()
    assert_equal(len(column), 2000)
    for i in range(2000):
        assert_equal(column.is_valid(i), present[i], "row " + String(i))
        assert_equal(column[i], values[i], "row " + String(i))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
