"""Tests for encoding a text column as codes and reading it back.

The three things worth pinning down are the ones a caller can see. The
categories are sorted rather than in first appearance order, because that is
what `.cat.categories` prints and what a groupby over the column produces. A
null does not become a category, it becomes a null code beside categories that
do not mention it. And the round trip gives back the column that went in,
including the nulls and including the rows that share a category.

The rest is arithmetic on the codes, which is where a bug would hide, so the
codes themselves are asserted rather than only the values they stand for.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.dtype.logical import LogicalType, TypeKind, named_type
from firepanda.kernel.cast import cast_any
from firepanda.kernel.dictionary import decode_dictionary, encode_dictionary


def text(var values: List[String]) -> StringArray:
    """Builds a text column with nothing missing.

    Args:
        values: The values.

    Returns:
        The column.
    """
    return strings_from_list(values)


def text_with_nulls(
    var values: List[String], null_at: List[Int]
) -> StringArray:
    """Builds a text column with nulls at the given rows.

    Args:
        values: One entry per row, ignored where the row is null.
        null_at: The rows that are missing.

    Returns:
        The column.
    """
    var out = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        var missing = False
        for at in null_at:
            if at == i:
                missing = True
        if missing:
            out.append_null()
        else:
            out.append(values[i].as_bytes())
    return out^.finish()


def test_the_categories_are_the_distinct_values_sorted() raises:
    var col = encode_dictionary(text(["rivet", "bolt", "rivet", "anchor"]))
    assert_true(col.is_dictionary())
    ref levels = col.categories()
    assert_equal(len(levels), 3)
    assert_equal(levels[0], "anchor")
    assert_equal(levels[1], "bolt")
    assert_equal(levels[2], "rivet")


def test_the_codes_point_at_the_sorted_positions() raises:
    # First appearance order would be rivet 0, bolt 1, anchor 2, which is the
    # order factorize hands back and is not the order that comes out.
    var col = encode_dictionary(text(["rivet", "bolt", "rivet", "anchor"]))
    var codes = col.codes[DType.int32]()
    assert_equal(codes[0], 2)
    assert_equal(codes[1], 1)
    assert_equal(codes[2], 2)
    assert_equal(codes[3], 0)


def test_a_null_is_not_a_category() raises:
    var col = encode_dictionary(
        text_with_nulls(["rivet", "", "bolt", ""], [1, 3])
    )
    assert_equal(len(col.categories()), 2)
    assert_false(col.is_valid(1))
    assert_false(col.is_valid(3))
    assert_true(col.is_valid(0))


def test_the_codes_beside_a_null_are_still_right() raises:
    # The null takes ordinal zero from factorize and pushes every real group up
    # by one, so this is the arithmetic that would go wrong quietly.
    var col = encode_dictionary(
        text_with_nulls(["rivet", "", "bolt", "anchor"], [1])
    )
    var codes = col.codes[DType.int32]()
    ref levels = col.categories()
    assert_equal(levels[Int(codes[0])], "rivet")
    assert_equal(levels[Int(codes[2])], "bolt")
    assert_equal(levels[Int(codes[3])], "anchor")


def test_a_column_with_no_nulls_starts_its_ordinals_at_zero() raises:
    var col = encode_dictionary(text(["b", "a"]))
    var codes = col.codes[DType.int32]()
    assert_equal(codes[0], 1)
    assert_equal(codes[1], 0)


def test_the_ordered_flag_is_carried_through() raises:
    var col = encode_dictionary(text(["low", "high"]), ordered=True)
    assert_true(col.type.ordered)
    var plain = encode_dictionary(text(["low", "high"]))
    assert_false(plain.type.ordered)


def test_an_empty_column_encodes_to_no_categories() raises:
    var empty = List[String]()
    var col = encode_dictionary(text(empty^))
    assert_true(col.is_dictionary())
    assert_equal(len(col), 0)
    assert_equal(len(col.categories()), 0)


def test_a_column_that_is_all_null_has_no_categories_either() raises:
    var col = encode_dictionary(text_with_nulls(["", ""], [0, 1]))
    assert_equal(len(col), 2)
    assert_equal(len(col.categories()), 0)
    assert_false(col.is_valid(0))
    assert_false(col.is_valid(1))


def test_one_value_repeated_is_one_category() raises:
    var col = encode_dictionary(text(["same", "same", "same"]))
    assert_equal(len(col.categories()), 1)
    var codes = col.codes[DType.int32]()
    assert_equal(codes[0], 0)
    assert_equal(codes[2], 0)


def test_the_round_trip_gives_back_the_column() raises:
    var words: List[String] = ["rivet", "bolt", "rivet", "anchor", "bolt"]
    var back = decode_dictionary(encode_dictionary(text(words.copy())))
    assert_equal(len(back), len(words))
    for i in range(len(words)):
        assert_equal(back[i], words[i])


def test_the_round_trip_keeps_the_nulls_where_they_were() raises:
    var col = text_with_nulls(["rivet", "", "bolt", ""], [1, 3])
    var back = decode_dictionary(encode_dictionary(col))
    assert_equal(len(back), 4)
    assert_equal(back[0], "rivet")
    assert_false(back.is_valid(1))
    assert_equal(back[2], "bolt")
    assert_false(back.is_valid(3))


def test_an_empty_string_is_a_category_and_not_a_null() raises:
    # The difference between a value nobody wrote and a value somebody wrote
    # nothing into, which the CSV reader argues about at length and which this
    # kernel has no business collapsing.
    var col = encode_dictionary(text(["", "a", ""]))
    assert_equal(len(col.categories()), 2)
    assert_true(col.is_valid(0))
    var back = decode_dictionary(col)
    assert_true(back.is_valid(0))
    assert_equal(back[0], "")


def test_decoding_something_that_is_not_a_dictionary_is_refused() raises:
    var raised = False
    try:
        _ = decode_dictionary(AnyArray(text(["a", "b"])))
    except:
        raised = True
    assert_true(raised)


def test_the_cast_to_a_category_encodes_a_text_column() raises:
    var col = cast_any(
        AnyArray(text(["rivet", "bolt", "rivet"])), named_type("category")
    )
    assert_true(col.is_dictionary())
    assert_equal(len(col.categories()), 2)


def test_the_cast_off_a_category_gives_the_values_and_not_the_codes() raises:
    # The codes here are 1, 0, 1 and the values are 20, 10, 20, so a cast that
    # read the physical layout would answer something that looks like an answer.
    var col = encode_dictionary(text(["20", "10", "20"]))
    var back = cast_any(col, LogicalType(TypeKind.INT, DType.int64))
    assert_equal(back.dtype(), DType.int64)
    ref values = back.as_typed_view[DType.int64]()
    assert_equal(values[0], 20)
    assert_equal(values[1], 10)
    assert_equal(values[2], 20)


def test_a_category_cast_to_text_is_the_values_again() raises:
    var col = encode_dictionary(text(["rivet", "bolt"]))
    var back = cast_any(col, LogicalType.STRING)
    assert_true(back.is_string())
    assert_equal(back.strings()[0], "rivet")


def test_casting_a_category_to_a_category_keeps_the_categories_it_had() raises:
    # Including the one nobody uses, which a decode and re-encode would drop and
    # which pandas keeps.
    var col = encode_dictionary(text(["b", "a", "b"]))
    var kept = cast_any(col, named_type("category"))
    assert_equal(len(kept.categories()), 2)
    var narrowed = cast_any(
        AnyArray(decode_dictionary(col).slice(0, 1)), named_type("category")
    )
    assert_equal(len(narrowed.categories()), 1)


def test_a_number_column_cannot_be_encoded_yet_and_says_so() raises:
    var values = Array[DType.int64](2)
    values.set_valid(0, 1)
    values.set_valid(1, 2)
    var message = String("")
    try:
        _ = cast_any(AnyArray(values^), named_type("category"))
    except cause:
        message = String(cause)
    assert_true("not supported" in message)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
