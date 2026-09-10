"""Tests that a category column survives the kernels that move rows around.

A filter, a take, a fill and a concatenation all build their output through the
codes' own dtype and then put the input's logical type back on the result. For
every other type that is the whole job. For a dictionary the categories live
beside the buffer rather than in it, so before this the result was a column whose
type said `category` and which had nothing behind it, and the failure showed up
somewhere unrelated: the Arrow writer refused it, and anything that read the
codes got the plain integers they are stored as.

So most of what is here is one assertion repeated, that the result is still a
dictionary and its categories are the ones that went in. The other half is the
kernels with two inputs, which cannot simply carry one side's list across,
because a code is a position and two columns whose categories differ give the
same code to different values. Those refuse, and the refusal is the test.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.kernel.concat import concat_any, concat_two_any
from firepanda.kernel.dictionary import decode_dictionary, encode_dictionary
from firepanda.kernel.nulls import (
    coalesce_any,
    fill_backward_any,
    fill_forward_any,
)
from firepanda.kernel.pick import pick_any
from firepanda.kernel.select import filter_any, take_any
from firepanda.kernel.shift import shift_any


def made(var values: List[String]) raises -> AnyArray:
    """Builds a category column, using `null` for a missing row.

    Args:
        values: The values, one per row.

    Returns:
        The column.

    Raises:
        Error: If it cannot be built.
    """
    var built = StringBuilder(capacity=len(values))
    for value in values:
        if value == "null":
            built.append_null()
            continue
        built.append(value.as_bytes())
    return encode_dictionary(built^.finish())


def labels(column: StringArray) raises -> List[String]:
    """Reads a text column back as a list.

    Args:
        column: The column.

    Returns:
        One string per row, and `null` for a missing one.

    Raises:
        Error: If a row cannot be read.
    """
    var read: List[String] = []
    for i in range(len(column)):
        if not column.is_valid(i):
            read.append("null")
            continue
        read.append(String(column[i]))
    return read^


def rows(column: AnyArray) raises -> List[String]:
    """Decodes a category column back to the values a caller would see.

    This is the assertion that matters. A column can claim to be a dictionary
    and hold codes that point into the wrong list, and only decoding it says
    which values are actually in it.

    Args:
        column: The column.

    Returns:
        One string per row.

    Raises:
        Error: If the column is not a dictionary.
    """
    return labels(decode_dictionary(column))


def mask_of(var keep: List[Bool]) -> Array[DType.bool]:
    """Turns a list of flags into a mask column.

    Args:
        keep: One flag per row.

    Returns:
        The mask.
    """
    var out = Array[DType.bool](len(keep))
    for i in range(len(keep)):
        out[i] = keep[i]
    return out^


def test_a_filtered_category_column_is_still_a_category() raises:
    var out = filter_any(
        made(["low", "high", "mid"]), mask_of([True, False, True])
    )
    assert_true(out.is_dictionary())
    assert_equal(len(out), 2)


def test_a_filter_keeps_the_values_and_not_the_codes() raises:
    var out = filter_any(
        made(["low", "high", "mid"]), mask_of([True, False, True])
    )
    var read = rows(out)
    assert_equal(read[0], "low")
    assert_equal(read[1], "mid")


def test_a_filter_leaves_an_emptied_category_in_the_list() raises:
    # pandas does the same, and `drop_unused_categories` is what undoes it.
    var out = filter_any(
        made(["low", "high", "mid"]), mask_of([True, False, False])
    )
    assert_equal(len(out.categories()), 3)
    assert_equal(len(out), 1)


def test_dropping_the_nulls_leaves_a_readable_column() raises:
    # The case that found all of this. `drop_nulls` is a filter, the result said
    # it was a category, and the Arrow writer refused it for having no
    # categories behind it.
    var col = made(["low", "null", "high", "low"])
    var out = filter_any(col, mask_of([True, False, True, True]))
    assert_true(out.is_dictionary())
    assert_equal(rows(out)[2], "low")


def test_a_taken_category_column_is_still_a_category() raises:
    var out = take_any(made(["low", "high", "mid"]), [2, 0])
    assert_true(out.is_dictionary())
    var read = rows(out)
    assert_equal(read[0], "mid")
    assert_equal(read[1], "low")


def test_a_take_that_repeats_a_row_repeats_its_value() raises:
    var out = take_any(made(["low", "high"]), [1, 1, 1])
    assert_equal(len(out), 3)
    assert_equal(rows(out)[2], "high")


def test_a_sliced_category_column_is_still_a_category() raises:
    # A slice is the third path, and neither of the other two reaches it. It is
    # what `head` and `tail` are, and it copies bytes without dispatching at all,
    # so it does not go anywhere near the code the other two share.
    var out = made(["low", "high", "mid", "low"]).slice(1, 3)
    assert_true(out.is_dictionary())
    var read = rows(out)
    assert_equal(len(read), 2)
    assert_equal(read[0], "high")
    assert_equal(read[1], "mid")


def test_shifting_keeps_the_categories() raises:
    # A shift is a slice stacked onto a run of gap, and the gap is built from the
    # type alone, so it is the one place where the side carrying the categories
    # is the second argument rather than the first.
    var out = shift_any(made(["low", "high", "mid"]), 1)
    assert_true(out.is_dictionary())
    var read = rows(out)
    assert_equal(read[0], "null")
    assert_equal(read[1], "low")
    assert_equal(read[2], "high")


def test_shifting_backwards_keeps_the_categories() raises:
    var out = shift_any(made(["low", "high", "mid"]), -1)
    assert_true(out.is_dictionary())
    var read = rows(out)
    assert_equal(read[0], "high")
    assert_equal(read[2], "null")


def test_shifting_past_the_end_keeps_the_categories() raises:
    # Every row is gap, so nothing of the original column is in the answer and
    # the categories are still the original's. A caller who shifts a categorical
    # off the end has an empty column of a type, not a column of no type.
    var out = shift_any(made(["low", "high", "mid"]), 9)
    assert_true(out.is_dictionary())
    assert_equal(len(out.categories()), 3)
    assert_equal(rows(out)[0], "null")


def test_filling_forward_keeps_the_categories() raises:
    var out = fill_forward_any(made(["low", "null", "high"]))
    assert_true(out.is_dictionary())
    var read = rows(out)
    assert_equal(read[0], "low")
    assert_equal(read[1], "low")
    assert_equal(read[2], "high")


def test_filling_backward_keeps_the_categories() raises:
    var out = fill_backward_any(made(["low", "null", "high"]))
    assert_true(out.is_dictionary())
    assert_equal(rows(out)[1], "high")


def test_two_category_columns_that_agree_can_be_stacked() raises:
    # Same categories, because both hold the same two labels, and the encoder
    # sorts. Different rows, so the codes differ and the values still line up.
    var out = concat_two_any(made(["low", "high"]), made(["high", "low"]))
    assert_true(out.is_dictionary())
    var read = rows(out)
    assert_equal(read[0], "low")
    assert_equal(read[2], "high")
    assert_equal(read[3], "low")


def test_stacking_a_list_of_agreeing_columns_works_too() raises:
    var parts = List[AnyArray]()
    parts.append(made(["low", "high"]))
    parts.append(made(["high", "low"]))
    parts.append(made(["low", "high"]))
    var out = concat_any(parts)
    assert_true(out.is_dictionary())
    assert_equal(len(out), 6)
    assert_equal(rows(out)[5], "high")


def test_stacking_two_columns_that_disagree_is_refused() raises:
    with assert_raises(contains="do not have the same categories"):
        _ = concat_two_any(made(["low", "high"]), made(["low", "mid"]))


def test_stacking_a_list_that_disagrees_is_refused() raises:
    var parts = List[AnyArray]()
    parts.append(made(["low", "high"]))
    parts.append(made(["low", "mid"]))
    with assert_raises(contains="do not have the same categories"):
        _ = concat_any(parts)


def test_the_refusal_says_what_to_do_about_it() raises:
    var said = String("")
    try:
        _ = concat_two_any(made(["low", "high"]), made(["low", "mid"]))
    except error:
        said = String(error)
    assert_true("set_categories" in said)


def test_coalescing_two_agreeing_columns_keeps_the_categories() raises:
    var out = coalesce_any(
        made(["low", "null", "high"]), made(["low", "high", "high"])
    )
    assert_true(out.is_dictionary())
    var read = rows(out)
    assert_equal(read[0], "low")
    assert_equal(read[1], "high")
    assert_equal(read[2], "high")


def test_coalescing_two_columns_that_disagree_is_refused() raises:
    with assert_raises(contains="do not have the same categories"):
        _ = coalesce_any(made(["low", "high"]), made(["low", "mid"]))


def test_picking_between_two_agreeing_columns_keeps_the_categories() raises:
    var out = pick_any(
        mask_of([True, False]), made(["low", "high"]), made(["high", "low"])
    )
    assert_true(out.is_dictionary())
    var read = rows(out)
    assert_equal(read[0], "low")
    assert_equal(read[1], "low")


def test_picking_between_two_columns_that_disagree_is_refused() raises:
    with assert_raises(contains="do not have the same categories"):
        _ = pick_any(
            mask_of([True, False]), made(["low", "high"]), made(["low", "mid"])
        )


def test_a_column_that_is_not_a_category_is_untouched() raises:
    # The carrying helper is called on every dtype, so this is the check that it
    # does nothing at all to the rest of the library.
    var col = Array[DType.int64](3)
    col[0] = 7
    col[1] = 8
    col[2] = 9
    var out = filter_any(AnyArray(col^), mask_of([True, False, True]))
    assert_true(not out.is_dictionary())
    assert_equal(len(out), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
