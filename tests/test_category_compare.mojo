"""Tests for comparing a category column.

The whole operation is codes against a code. A comparison never promotes and
never decodes, except in the one shape where a categorical meets an ordinary
text column under equality and there is nothing else to compare by.

Six rules come out of pandas and each of them has a test here. Equality works
whether the categories are ordered or not. An ordering comparison needs them
ordered. A scalar that is not one of the categories is all false under equality
and an error under an ordering, which is the asymmetry worth pinning down, since
equality can answer without a position and an ordering cannot. Two categoricals
have to agree about their categories exactly, list and order both. And the
ordering is category order rather than value order, which is the last test in the
file and is the entire point of `ordered=True`.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.array.value import Value
from firepanda.kernel.binary import BinaryOp, binary_any, binary_value_any
from firepanda.kernel.dictionary import (
    encode_dictionary,
    set_categories,
    set_ordered,
)


def made(var values: List[String], ordered: Bool = False) raises -> AnyArray:
    """Builds a category column out of a list of words.

    Args:
        values: The values.
        ordered: Whether the categories have a meaningful order.

    Returns:
        The encoded column.
    """
    var col = encode_dictionary(strings_from_list(values))
    if not ordered:
        return col^
    return set_ordered(col, True)


def reordered(
    var values: List[String], var names: List[String]
) raises -> AnyArray:
    """Builds an ordered category column whose categories are not sorted.

    Args:
        values: The values.
        names: The categories, in the order they are to have.

    Returns:
        The encoded column, ordered.

    Raises:
        Error: If a value is not one of the names.
    """
    return set_categories(
        encode_dictionary(strings_from_list(values)),
        strings_from_list(names),
        True,
    )


def holed(var values: List[String]) raises -> StringArray:
    """Builds a text column where the word `null` means a missing value.

    Args:
        values: The values, with `null` for the holes.

    Returns:
        The column.
    """
    var out = StringBuilder(capacity=len(values))
    for value in values:
        if value == "null":
            out.append_null()
            continue
        out.append(value.as_bytes())
    return out^.finish()


def answers(column: AnyArray) raises -> List[String]:
    """Reads a bool column as words, so a null is visible next to a false.

    Args:
        column: The column.

    Returns:
        One of `true`, `false` or `null` per row.

    Raises:
        Error: If the column is not a bool column.
    """
    var values = column.as_typed[DType.bool]()
    var read: List[String] = []
    for i in range(len(values)):
        if not values.is_valid(i):
            read.append("null")
        elif values[i]:
            read.append("true")
        else:
            read.append("false")
    return read^


def against(col: AnyArray, word: String, op: BinaryOp) raises -> List[String]:
    """Compares a column with one word and reads the answer.

    Args:
        col: The column.
        word: The scalar.
        op: The comparison.

    Returns:
        One of `true`, `false` or `null` per row.

    Raises:
        Error: If the comparison is refused.
    """
    return answers(binary_value_any(col, Value(word), op))


def test_equality_works_on_an_unordered_column() raises:
    var col = made(["bolt", "anchor", "rivet"])
    var out = against(col, "bolt", BinaryOp.EQ)
    assert_equal(out[0], "true")
    assert_equal(out[1], "false")
    assert_equal(out[2], "false")


def test_inequality_works_on_an_unordered_column() raises:
    var col = made(["bolt", "anchor", "rivet"])
    var out = against(col, "bolt", BinaryOp.NE)
    assert_equal(out[0], "false")
    assert_equal(out[1], "true")
    assert_equal(out[2], "true")


def test_an_ordering_needs_the_categories_ordered() raises:
    var col = made(["bolt", "anchor", "rivet"])
    var refused = False
    try:
        _ = against(col, "bolt", BinaryOp.LT)
    except error:
        refused = True
        assert_equal(
            String(error),
            "Unordered Categoricals can only compare equality or not",
        )
    assert_true(refused)


def test_an_ordering_runs_once_the_categories_are_ordered() raises:
    var col = made(["bolt", "anchor", "rivet"], ordered=True)
    var out = against(col, "bolt", BinaryOp.LT)
    assert_equal(out[0], "false")
    assert_equal(out[1], "true")
    assert_equal(out[2], "false")


def test_less_or_equal_takes_the_row_that_is_the_scalar() raises:
    var col = made(["bolt", "anchor", "rivet"], ordered=True)
    var out = against(col, "bolt", BinaryOp.LE)
    assert_equal(out[0], "true")
    assert_equal(out[1], "true")
    assert_equal(out[2], "false")


def test_a_scalar_that_is_not_a_category_is_equal_to_nothing() raises:
    var col = made(["bolt", "anchor", "rivet"])
    var out = against(col, "washer", BinaryOp.EQ)
    assert_equal(out[0], "false")
    assert_equal(out[1], "false")
    assert_equal(out[2], "false")


def test_a_scalar_that_is_not_a_category_is_unequal_to_everything() raises:
    var col = made(["bolt", "anchor", "rivet"])
    var out = against(col, "washer", BinaryOp.NE)
    assert_equal(out[0], "true")
    assert_equal(out[1], "true")
    assert_equal(out[2], "true")


def test_an_ordering_against_a_scalar_that_is_not_a_category_is_refused() raises:
    var col = made(["bolt", "anchor", "rivet"], ordered=True)
    var refused = False
    try:
        _ = against(col, "washer", BinaryOp.GT)
    except error:
        refused = True
        assert_equal(
            String(error),
            "Invalid comparison between dtype=category and str",
        )
    assert_true(refused)


def test_the_unordered_refusal_wins_over_the_missing_category_one() raises:
    # Both are wrong with the same call and pandas reports the ordering first,
    # because a caller who has not asked for an order has a different problem
    # from one whose scalar is a typo.
    var col = made(["bolt", "anchor"])
    var refused = False
    try:
        _ = against(col, "washer", BinaryOp.LT)
    except error:
        refused = True
        assert_equal(
            String(error),
            "Unordered Categoricals can only compare equality or not",
        )
    assert_true(refused)


def test_a_missing_row_answers_missing_rather_than_false() raises:
    var col = encode_dictionary(holed(["bolt", "null", "anchor"]))
    var out = against(col, "bolt", BinaryOp.EQ)
    assert_equal(out[0], "true")
    assert_equal(out[1], "null")
    assert_equal(out[2], "false")


def test_a_missing_row_stays_missing_under_an_ordering() raises:
    var col = set_ordered(
        encode_dictionary(holed(["bolt", "null", "anchor"])),
        True,
    )
    var out = against(col, "bolt", BinaryOp.LE)
    assert_equal(out[0], "true")
    assert_equal(out[1], "null")
    assert_equal(out[2], "true")


def test_a_null_scalar_makes_every_row_missing() raises:
    var col = made(["bolt", "anchor"])
    var out = answers(
        binary_value_any(col, Value(null=col.type.copy()), BinaryOp.EQ)
    )
    assert_equal(out[0], "null")
    assert_equal(out[1], "null")


def test_two_columns_with_the_same_categories_compare() raises:
    var left = made(["bolt", "anchor", "rivet", "anchor"])
    var right = made(["bolt", "rivet", "anchor", "anchor"])
    var out = answers(binary_any(left, right, BinaryOp.EQ))
    assert_equal(out[0], "true")
    assert_equal(out[1], "false")
    assert_equal(out[2], "false")
    assert_equal(out[3], "true")


def test_two_columns_that_disagree_about_their_categories_are_refused() raises:
    var left = made(["bolt", "anchor"])
    var right = made(["bolt", "rivet"])
    var refused = False
    try:
        _ = binary_any(left, right, BinaryOp.EQ)
    except error:
        refused = True
        assert_equal(
            String(error),
            "Categoricals can only be compared if 'categories' are the same.",
        )
    assert_true(refused)


def test_an_ordering_on_two_columns_needs_both_ordered() raises:
    var left = made(["bolt", "anchor"], ordered=True)
    var right = made(["bolt", "anchor"])
    var refused = False
    try:
        _ = binary_any(left, right, BinaryOp.LT)
    except error:
        refused = True
        assert_equal(
            String(error),
            "Unordered Categoricals can only compare equality or not",
        )
    assert_true(refused)


def test_an_ordering_on_two_ordered_columns_runs() raises:
    var left = made(["bolt", "anchor", "rivet"], ordered=True)
    var right = made(["rivet", "anchor", "bolt"], ordered=True)
    var out = answers(binary_any(left, right, BinaryOp.LT))
    assert_equal(out[0], "true")
    assert_equal(out[1], "false")
    assert_equal(out[2], "false")


def test_a_categorical_against_plain_text_compares_by_value() raises:
    var left = made(["bolt", "anchor", "rivet"])
    var right = AnyArray(strings_from_list(["bolt", "rivet", "rivet"]))
    var out = answers(binary_any(left, right, BinaryOp.EQ))
    assert_equal(out[0], "true")
    assert_equal(out[1], "false")
    assert_equal(out[2], "true")


def test_an_ordering_against_plain_text_is_refused() raises:
    var left = made(["bolt", "anchor"], ordered=True)
    var right = AnyArray(strings_from_list(["bolt", "rivet"]))
    var refused = False
    try:
        _ = binary_any(left, right, BinaryOp.LT)
    except error:
        refused = True
        assert_true(
            String(error).startswith(
                "Cannot compare a Categorical for op __lt__ with type"
            )
        )
    assert_true(refused)


def test_the_ordering_follows_the_categories_and_not_the_words() raises:
    # The whole reason `ordered=True` exists. These categories are small, medium
    # and large in that order, so medium is less than large even though the word
    # sorts after it, and a library that decoded and compared text would answer
    # the other way.
    var col = reordered(
        ["medium", "large", "small"], ["small", "medium", "large"]
    )
    var out = against(col, "large", BinaryOp.LT)
    assert_equal(out[0], "true")
    assert_equal(out[1], "false")
    assert_equal(out[2], "true")


def test_an_imported_narrow_column_compares_the_same() raises:
    var codes = Array[DType.int8](3)
    codes.set_valid(0, 1)
    codes.set_null(1)
    codes.set_valid(2, 0)
    var col = AnyArray.dictionary[DType.int8](
        codes^, strings_from_list(["anchor", "bolt"]), False
    )
    var out = against(col, "bolt", BinaryOp.EQ)
    assert_equal(out[0], "true")
    assert_equal(out[1], "null")
    assert_equal(out[2], "false")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
