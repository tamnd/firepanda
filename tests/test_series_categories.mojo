"""Tests for the category members on the Mojo `Series`.

The kernels these forward to have their own tests in `test_dictionary.mojo` and
`test_categories.mojo`, so what is checked here is the surface: that the members
exist, that they hand back a `Series` rather than an `AnyArray`, that the ones
answering one row per input row keep the row labels and the ones answering one
row per category do not, and that a column which is not a categorical is refused
by every one of them.

The last part matters more than it looks. Before these members existed, the only
way to reach a category column's codes from Mojo was to import the kernel package
and hold an `AnyArray`, which is the layer the frame package exists to cover. A
test that a refusal comes back with a message naming the type is a test that the
frame layer is answering rather than passing the question through.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.strings import strings_from_list
from firepanda.frame.series import Series
from firepanda.kernel.dictionary import encode_dictionary


def made(var values: List[String], name: String = "part") raises -> Series:
    """Builds a named category column out of a list of words.

    Args:
        values: The values.
        name: The column's name.

    Returns:
        The series.
    """
    return Series(name, encode_dictionary(strings_from_list(values)))


def words(column: Series) raises -> List[String]:
    """Reads a text series back as a list.

    Args:
        column: The series.

    Returns:
        One string per row, and `null` for a missing one.

    Raises:
        Error: If the series is not text.
    """
    var read: List[String] = []
    for i in range(len(column)):
        if not column.is_valid(i):
            read.append("null")
            continue
        read.append(column.text(i))
    return read^


def codes_of(column: Series) raises -> List[String]:
    """Reads a code series back as a list, so a null is visible next to a zero.

    Args:
        column: The series.

    Returns:
        One number per row as text, and `null` for a missing one.

    Raises:
        Error: If the series is not an int32 column.
    """
    var values = column.as_typed[DType.int32]()
    var read: List[String] = []
    for i in range(len(values)):
        if not values.is_valid(i):
            read.append("null")
            continue
        read.append(String(values[i]))
    return read^


def test_a_category_column_says_it_is_one() raises:
    assert_true(made(["bolt", "anchor"]).cat_is_category())


def test_a_text_column_says_it_is_not() raises:
    var plain = Series("part", strings_from_list(["bolt", "anchor"]))
    assert_false(plain.cat_is_category())


def test_the_categories_come_back_sorted_and_unnamed() raises:
    var out = made(["rivet", "bolt", "anchor", "bolt"]).cat_categories()
    assert_equal(out.name, "")
    assert_equal(len(out), 3)
    var read = words(out)
    assert_equal(read[0], "anchor")
    assert_equal(read[1], "bolt")
    assert_equal(read[2], "rivet")


def test_the_categories_are_shorter_than_the_column() raises:
    var col = made(["bolt", "bolt", "bolt", "bolt"])
    assert_equal(len(col), 4)
    assert_equal(len(col.cat_categories()), 1)


def test_the_codes_are_positions_in_the_categories() raises:
    var out = made(["rivet", "bolt", "anchor"]).cat_codes()
    assert_equal(out.name, "")
    var read = codes_of(out)
    assert_equal(read[0], "2")
    assert_equal(read[1], "1")
    assert_equal(read[2], "0")


def test_the_codes_keep_the_row_labels() raises:
    var col = made(["rivet", "bolt", "anchor"])
    var out = col.cat_codes()
    assert_equal(len(out.index), len(col.index))


def test_a_new_column_is_unordered() raises:
    assert_false(made(["bolt", "anchor"]).cat_ordered())


def test_the_order_flag_carries() raises:
    var col = made(["bolt", "anchor"]).cat_set_ordered(True)
    assert_true(col.cat_ordered())
    assert_equal(col.name, "part")


def test_turning_the_order_off_again_works() raises:
    var col = made(["bolt", "anchor"]).cat_set_ordered(True)
    assert_false(col.cat_set_ordered(False).cat_ordered())


def test_a_rename_leaves_every_code_where_it_is() raises:
    var col = made(["rivet", "bolt", "anchor"])
    var out = col.cat_rename_categories(strings_from_list(["a", "b", "c"]))
    assert_equal(codes_of(out.cat_codes())[0], "2")
    var read = words(out.cat_categories())
    assert_equal(read[0], "a")
    assert_equal(read[2], "c")


def test_setting_the_categories_matches_by_value() raises:
    var col = made(["rivet", "bolt", "anchor"])
    var out = col.cat_set_categories(
        strings_from_list(["rivet", "bolt", "anchor"]), True
    )
    assert_true(out.cat_ordered())
    assert_equal(codes_of(out.cat_codes())[0], "0")
    assert_equal(codes_of(out.cat_codes())[2], "2")


def test_a_row_whose_category_is_dropped_becomes_missing() raises:
    var col = made(["rivet", "bolt", "anchor"])
    var out = col.cat_set_categories(strings_from_list(["bolt", "anchor"]))
    assert_equal(codes_of(out.cat_codes())[0], "null")
    assert_equal(codes_of(out.cat_codes())[1], "0")


def test_dropping_the_unused_categories_keeps_the_order() raises:
    var col = made(["rivet", "bolt", "anchor"]).cat_set_categories(
        strings_from_list(["anchor", "bolt", "rivet", "washer"])
    )
    assert_equal(len(col.cat_categories()), 4)
    var out = col.cat_drop_unused_categories()
    assert_equal(len(out.cat_categories()), 3)
    var read = words(out.cat_categories())
    assert_equal(read[0], "anchor")
    assert_equal(read[2], "rivet")


def test_every_member_refuses_a_column_that_is_not_a_category() raises:
    # One test rather than seven, because the refusal is the same refusal and
    # what is being checked is that none of them was written without it.
    var plain = Series("part", strings_from_list(["bolt", "anchor"]))
    var refused = 0
    try:
        _ = plain.cat_categories()
    except:
        refused += 1
    try:
        _ = plain.cat_codes()
    except:
        refused += 1
    try:
        _ = plain.cat_ordered()
    except:
        refused += 1
    try:
        _ = plain.cat_set_ordered(True)
    except:
        refused += 1
    try:
        _ = plain.cat_rename_categories(strings_from_list(["a"]))
    except:
        refused += 1
    try:
        _ = plain.cat_set_categories(strings_from_list(["a"]))
    except:
        refused += 1
    try:
        _ = plain.cat_drop_unused_categories()
    except:
        refused += 1
    assert_equal(refused, 7)


def test_the_refusal_names_the_type_it_was_given() raises:
    var plain = Series("part", strings_from_list(["bolt", "anchor"]))
    var said = String("")
    try:
        _ = plain.cat_ordered()
    except error:
        said = String(error)
    assert_true(said.startswith("category: "))
    assert_true("is not a category column" in said)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
