"""Tests for changing a category column's categories after it exists.

Three operations carry the eleven names pandas puts on `Series.cat`. A rename is
decided by position and leaves every code alone. Setting the categories is
decided by value, and a row whose value is not in the new list becomes null.
Dropping the unused ones is decided by the codes. The tests below assert the
codes as well as the values, because every one of these is arithmetic on the
codes and that is where a mistake would hide rather than show.

The fourth thing under test is reading codes at a width other than int32. Every
dictionary column firepanda builds itself is int32, and every one that arrives
from pandas over Arrow is int8, so the widening is not an edge case, it is the
common case for imported data.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.dtype.logical import LogicalType
from firepanda.kernel.dictionary import (
    decode_dictionary,
    dictionary_codes,
    drop_unused_categories,
    encode_dictionary,
    rename_categories,
    set_categories,
    set_ordered,
)


def text(var values: List[String]) -> StringArray:
    """Builds a text column with nothing missing.

    Args:
        values: The values.

    Returns:
        The column.
    """
    return strings_from_list(values)


def made(var values: List[String]) raises -> AnyArray:
    """Builds a category column out of a list of words.

    Args:
        values: The values.

    Returns:
        The encoded column.
    """
    return encode_dictionary(strings_from_list(values))


def words(var values: List[String]) -> StringArray:
    """Builds a list of category labels.

    Args:
        values: The labels.

    Returns:
        The labels as a text column.
    """
    return strings_from_list(values)


def read(col: AnyArray) raises -> List[String]:
    """Reads a category column back as the values it stands for.

    A null comes back as the word `null`, which no test below uses as a value,
    so one list can carry both without a second list saying where the holes are.

    Args:
        col: The column.

    Returns:
        One entry per row.
    """
    var values = decode_dictionary(col)
    var out = List[String]()
    for i in range(len(values)):
        if not values.is_valid(i):
            out.append(String("null"))
            continue
        out.append(values[i])
    return out^


def levels(col: AnyArray) raises -> List[String]:
    """Reads a category column's categories, in the order it holds them.

    Args:
        col: The column.

    Returns:
        The labels.

    Raises:
        Error: If the column is not a category column.
    """
    ref held = col.categories()
    var out = List[String]()
    for at in range(len(held)):
        out.append(held[at])
    return out^


def test_renaming_the_categories_leaves_every_code_where_it_was() raises:
    var col = made(["rivet", "bolt", "rivet", "anchor"])
    var before = dictionary_codes(col)
    var out = rename_categories(col, words(["A", "B", "C"]), False)
    var after = dictionary_codes(out)
    for i in range(len(before)):
        assert_equal(Int(before[i]), Int(after[i]))
    assert_equal(read(out)[0], "C")
    assert_equal(read(out)[3], "A")


def test_a_rename_takes_the_labels_in_the_order_the_column_holds_them() raises:
    var col = made(["rivet", "bolt", "anchor"])
    var out = rename_categories(col, words(["first", "second", "third"]), False)
    assert_equal(levels(out)[0], "first")
    assert_equal(levels(out)[1], "second")
    assert_equal(levels(out)[2], "third")


def test_a_rename_with_fewer_labels_nulls_the_rows_that_fall_off() raises:
    # What `set_categories(rename=True)` does with a short list, which is the
    # reason the count is not checked down here.
    var col = made(["rivet", "bolt", "anchor"])
    var out = rename_categories(col, words(["first", "second"]), False)
    assert_equal(len(levels(out)), 2)
    assert_equal(read(out)[0], "null")
    assert_equal(read(out)[1], "second")
    assert_equal(read(out)[2], "first")


def test_a_rename_with_more_labels_leaves_the_extra_ones_unused() raises:
    var col = made(["rivet", "bolt"])
    var out = rename_categories(col, words(["a", "b", "c"]), False)
    assert_equal(len(levels(out)), 3)
    assert_equal(read(out)[0], "b")
    assert_equal(read(out)[1], "a")
    assert_equal(len(levels(drop_unused_categories(out))), 2)


def test_a_rename_that_repeats_a_label_is_refused() raises:
    var col = made(["rivet", "bolt"])
    var message = String("")
    try:
        _ = rename_categories(col, words(["same", "same"]), False)
    except cause:
        message = String(cause)
    assert_true("distinct" in message)


def test_a_rename_can_say_the_order_means_something() raises:
    var col = made(["rivet", "bolt"])
    var out = rename_categories(col, words(["a", "b"]), True)
    assert_true(out.type.ordered)


def test_setting_the_categories_matches_the_old_ones_by_value() raises:
    var col = made(["rivet", "bolt", "anchor"])
    var out = set_categories(col, words(["anchor", "bolt", "rivet"]), False)
    assert_equal(read(out)[0], "rivet")
    assert_equal(read(out)[1], "bolt")
    assert_equal(read(out)[2], "anchor")


def test_reordering_the_categories_moves_the_codes_to_match() raises:
    var col = made(["rivet", "bolt", "anchor"])
    var out = set_categories(col, words(["rivet", "bolt", "anchor"]), False)
    var codes = dictionary_codes(out)
    assert_equal(Int(codes[0]), 0)
    assert_equal(Int(codes[1]), 1)
    assert_equal(Int(codes[2]), 2)


def test_a_category_left_out_of_the_new_list_nulls_its_rows() raises:
    var col = made(["rivet", "bolt", "rivet", "anchor"])
    var out = set_categories(col, words(["anchor", "rivet"]), False)
    assert_equal(read(out)[0], "rivet")
    assert_equal(read(out)[1], "null")
    assert_equal(read(out)[2], "rivet")
    assert_equal(read(out)[3], "anchor")
    assert_equal(len(levels(out)), 2)


def test_a_category_nothing_uses_can_be_added() raises:
    var col = made(["rivet", "bolt"])
    var out = set_categories(col, words(["bolt", "rivet", "washer"]), False)
    assert_equal(len(levels(out)), 3)
    assert_equal(levels(out)[2], "washer")
    assert_equal(read(out)[0], "rivet")
    assert_equal(read(out)[1], "bolt")


def test_setting_the_categories_can_say_the_order_means_something() raises:
    var col = made(["rivet", "bolt"])
    assert_false(col.type.ordered)
    var out = set_categories(col, words(["bolt", "rivet"]), True)
    assert_true(out.type.ordered)


def test_a_new_category_list_that_repeats_is_refused() raises:
    var col = made(["rivet", "bolt"])
    var message = String("")
    try:
        _ = set_categories(col, words(["bolt", "bolt"]), False)
    except cause:
        message = String(cause)
    assert_true("distinct" in message)


def test_setting_the_categories_keeps_the_nulls_that_were_there() raises:
    var out = StringBuilder(capacity=3)
    out.append("rivet".as_bytes())
    out.append_null()
    out.append("bolt".as_bytes())
    var col = encode_dictionary(out^.finish())
    var same = set_categories(col, words(["bolt", "rivet"]), False)
    assert_equal(read(same)[0], "rivet")
    assert_equal(read(same)[1], "null")
    assert_equal(read(same)[2], "bolt")


def test_dropping_the_unused_categories_keeps_the_order_of_the_rest() raises:
    # The order under test is deliberately not the sorted one, since a version
    # that rebuilt the column from its values rather than from its codes would
    # sort what was left and would pass against sorted categories.
    var col = made(["rivet", "anchor"])
    var wide = set_categories(col, words(["rivet", "bolt", "anchor"]), False)
    var out = drop_unused_categories(wide)
    assert_equal(len(levels(out)), 2)
    assert_equal(levels(out)[0], "rivet")
    assert_equal(levels(out)[1], "anchor")
    assert_equal(read(out)[0], "rivet")
    assert_equal(read(out)[1], "anchor")


def test_dropping_the_unused_categories_drops_only_the_unused_ones() raises:
    var col = made(["rivet", "bolt"])
    var wide = set_categories(col, words(["anchor", "bolt", "rivet"]), False)
    var out = drop_unused_categories(wide)
    assert_equal(len(levels(out)), 2)
    assert_equal(levels(out)[0], "bolt")
    assert_equal(levels(out)[1], "rivet")
    assert_equal(read(out)[0], "rivet")
    assert_equal(read(out)[1], "bolt")


def test_dropping_the_unused_categories_keeps_the_ordered_flag() raises:
    var col = made(["rivet", "bolt"])
    var wide = set_categories(col, words(["anchor", "bolt", "rivet"]), True)
    var out = drop_unused_categories(wide)
    assert_true(out.type.ordered)


def test_a_column_of_nothing_but_nulls_loses_every_category() raises:
    var out = StringBuilder(capacity=2)
    out.append_null()
    out.append_null()
    var col = encode_dictionary(out^.finish())
    var thinned = drop_unused_categories(col)
    assert_equal(len(levels(thinned)), 0)
    assert_equal(len(thinned), 2)


def test_the_ordered_flag_can_be_set_and_unset_without_touching_the_codes() raises:
    var col = made(["rivet", "bolt", "rivet"])
    var before = dictionary_codes(col)
    var on = set_ordered(col, True)
    assert_true(on.type.ordered)
    var off = set_ordered(on, False)
    assert_false(off.type.ordered)
    var after = dictionary_codes(off)
    for i in range(len(before)):
        assert_equal(Int(before[i]), Int(after[i]))
    assert_equal(len(levels(off)), 2)


def test_none_of_the_three_will_touch_a_column_that_is_not_a_category() raises:
    var plain = AnyArray(strings_from_list(["rivet", "bolt"]))
    var message = String("")
    try:
        _ = drop_unused_categories(plain)
    except cause:
        message = String(cause)
    assert_true("not a dictionary column" in message)


def test_codes_held_at_a_narrower_width_are_read_as_well() raises:
    # What a pandas categorical looks like once it has come over Arrow, which is
    # int8 codes whenever the cardinality lets pandas write them.
    var codes = Array[DType.int8](3)
    codes.set_valid(0, 1)
    codes.set_null(1)
    codes.set_valid(2, 0)
    var col = AnyArray.dictionary[DType.int8](
        codes^, strings_from_list(["bolt", "rivet"]), False
    )
    var read_back = dictionary_codes(col)
    assert_equal(Int(read_back[0]), 1)
    assert_false(read_back.is_valid(1))
    assert_equal(Int(read_back[2]), 0)
    assert_equal(read(col)[0], "rivet")
    assert_equal(read(col)[1], "null")
    assert_equal(read(col)[2], "bolt")


def test_a_narrow_column_comes_out_at_int32_once_it_is_rewritten() raises:
    var codes = Array[DType.int8](2)
    codes.set_valid(0, 0)
    codes.set_valid(1, 1)
    var col = AnyArray.dictionary[DType.int8](
        codes^, strings_from_list(["bolt", "rivet"]), False
    )
    var out = set_categories(col, words(["rivet", "bolt"]), False)
    assert_equal(out.type.physical, DType.int32)
    assert_equal(read(out)[0], "bolt")
    assert_equal(read(out)[1], "rivet")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
