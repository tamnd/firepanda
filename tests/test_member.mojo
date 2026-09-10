"""Tests for asking whether each row's value is one of a set.

There are two routes through this kernel and a threshold between them, so every
test that can be run on both is run on both. `over` and `text_over` take a set,
run the kernel, run the scalar twin, and compare row by row; then they pad the
set out past the threshold with members that are not in the column and do it
again. Padding cannot change any answer, so a test that passes small and fails
padded has caught the table route on its own, which is what it is there for.

The text table has a third route under it, the fallback for two members that hash
alike, and that one cannot be reached from outside: it needs a hash collision on
sixty four bits. It is exercised by pointing the same comparison at a set with
duplicates in it, which takes the same branch in `insert` without needing the
collision, and beyond that it is read rather than tested.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.array import Array
from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.exec.morsel import MORSEL_ROWS
from firepanda.frame.series import Series
from firepanda.kernel.agg import sum_of
from firepanda.kernel.compare import not_equal
from firepanda.kernel.concat import concat_arrays, concat_strings
from firepanda.kernel.member import (
    LINEAR_MAX,
    TEXT_LINEAR_MAX,
    is_in,
    text_is_in,
)
from firepanda.kernel.scalar import is_in_scalar, text_is_in_scalar


def numbers(values: List[Int]) -> Array[DType.int64]:
    """Builds an int64 column from a list.

    Args:
        values: The values.

    Returns:
        The column.
    """
    var out = Array[DType.int64](len(values))
    for i in range(len(values)):
        out.set_valid(i, Int64(values[i]))
    return out^


def sample() -> Array[DType.int64]:
    """Builds the number column every test below looks things up in.

    Returns:
        The column, with nulls at rows three and six.
    """
    var out = numbers([1, 2, 3, 0, 5, 2, 0, 9, -4, 7])
    out.set_null(3)
    out.set_null(6)
    return out^


def agrees(
    got: Array[DType.bool], want: Array[DType.bool], label: String
) raises:
    """Asserts that a kernel answer matches the twin's, row by row.

    Args:
        got: The kernel's answer.
        want: The twin's answer.
        label: What to name in the failure.

    Raises:
        AssertionError: On the first row that differs.
    """
    assert_equal(len(got), len(want), label + ": lengths differ")
    for i in range(len(got)):
        assert_equal(got.is_valid(i), want.is_valid(i), label + ": validity")
        if got.is_valid(i):
            assert_equal(got[i], want[i], label + ": row " + String(i))


def over(col: Array[DType.int64], set: List[Int], label: String) raises:
    """Runs both routes of the number kernel against the twin.

    The set is used as given and then padded past the threshold with values
    nothing in the column can equal, which forces the table route without
    changing a single answer.

    Args:
        col: The column.
        set: The members.
        label: What to name in the failure.

    Raises:
        AssertionError: If either route disagrees with the twin.
    """
    var small = numbers(set)
    agrees(is_in(col, small), is_in_scalar(col, small), label + " small")

    var padded = set.copy()
    var far = 1000
    while len(padded) <= LINEAR_MAX:
        padded.append(far)
        far += 1
    var big = numbers(padded)
    agrees(is_in(col, big), is_in_scalar(col, big), label + " padded")


def text_over(col: StringArray, set: List[String], label: String) raises:
    """Runs both routes of the text kernel against the twin.

    Args:
        col: The column.
        set: The members.
        label: What to name in the failure.

    Raises:
        AssertionError: If either route disagrees with the twin.
    """
    var small = strings_from_list(set)
    agrees(
        text_is_in(col, small), text_is_in_scalar(col, small), label + " small"
    )

    var padded = set.copy()
    var far = 0
    while len(padded) <= TEXT_LINEAR_MAX:
        padded.append("absent " + String(far))
        far += 1
    var big = strings_from_list(padded)
    agrees(text_is_in(col, big), text_is_in_scalar(col, big), label + " padded")


def test_numbers_match_the_twin() raises:
    var col = sample()
    over(col, [2], "one member")
    over(col, [2, 7], "two members")
    over(col, [1, 2, 3, 5, 7, 9, -4], "every value present")
    over(col, [11, 12], "nothing present")
    over(col, [0], "zero, which is also what a null holds")


def test_a_duplicate_member_changes_nothing() raises:
    var col = sample()
    over(col, [2, 2, 2, 7], "duplicates")


def test_an_empty_set_is_all_false_and_not_all_null() raises:
    var col = sample()
    var mask = is_in(col, Array[DType.int64](0))
    assert_equal(len(mask), len(col))
    for i in range(len(col)):
        if col.is_valid(i):
            assert_true(mask.is_valid(i))
            assert_false(mask[i])
        else:
            assert_false(mask.is_valid(i))


def test_a_null_row_stays_null_on_both_routes() raises:
    var col = sample()
    # Row three and row six are null and both hold zero in the values buffer,
    # which is the invariant `kernel/__init__.mojo` describes. A kernel that
    # compared straight through the buffer without repairing the validity would
    # report them as members of a set holding zero, and it would look right on
    # any set that did not.
    var mask = is_in(col, numbers([0]))
    assert_false(mask.is_valid(3))
    assert_false(mask.is_valid(6))


def test_a_null_member_is_not_a_member() raises:
    var col = sample()
    var set = numbers([2, 0])
    set.set_null(1)
    var mask = is_in(col, set)
    # Row zero, which is present and holds zero, is not a member of a set whose
    # only zero is a null one.
    assert_true(mask.is_valid(0))
    assert_false(mask[0])
    assert_true(mask[1])


def test_text_matches_the_twin() raises:
    var col = strings_from_list(
        [
            "13",
            "31",
            "23",
            "",
            "29",
            "a much longer value than fits inside a view",
            "another long one that does not fit inside a view",
            "13",
        ]
    )
    text_over(col, ["13"], "one short member")
    text_over(col, ["13", "31", "23", "29"], "the q22 shape")
    text_over(col, [""], "the empty string")
    text_over(col, ["nope"], "nothing present")
    text_over(
        col,
        ["a much longer value than fits inside a view"],
        "one long member",
    )
    text_over(col, ["13", "1", "133"], "prefixes of a member")


def test_text_nulls_stay_null() raises:
    # A null element holds the empty view, so a set containing the empty string
    # is the one that catches a kernel comparing straight through the views
    # without checking validity first.
    var builder = StringBuilder(capacity=4)
    builder.append("13".as_bytes())
    builder.append_null()
    builder.append("".as_bytes())
    builder.append("31".as_bytes())
    var col = builder^.finish()

    var mask = text_is_in(col, strings_from_list(["13", ""]))
    assert_true(mask[0])
    assert_false(mask.is_valid(1))
    assert_true(mask.is_valid(2))
    assert_true(mask[2])
    assert_false(mask[3])


def test_a_column_either_side_of_the_morsel_split_matches_the_twin() raises:
    # Above a row count no short column reaches this runs on every core, so the
    # split between morsels has to be walked as well as the loop inside one. The
    # second morsel is where a kernel writing at `i - start` rather than at `i`,
    # or repairing the validity of the wrong range, comes apart. Nothing is read
    # row by row: the twin is asked for the whole column, the two answers are
    # compared by a kernel and reduced by another, and only a count comes back.
    var unit = numbers([4, 17, 5, 99])
    var col = unit.copy()
    while len(col) < MORSEL_ROWS:
        var pair = List[Array[DType.int64]]()
        pair.append(col.copy())
        pair.append(col.copy())
        col = concat_arrays(pair)
    assert_equal(len(col), MORSEL_ROWS)
    var tail = List[Array[DType.int64]]()
    tail.append(col^)
    tail.append(unit.copy())
    col = concat_arrays(tail)

    # The twin runs once and both routes are checked against the one answer.
    # Padding the set with values the column does not hold cannot change a row,
    # so the two routes are answering the same question and the expensive half
    # of the comparison only has to be paid for once.
    var small = numbers([4, 17])
    var want = is_in_scalar(col, small)
    assert_equal(len(want), MORSEL_ROWS + 4)
    # And that the twin was not vacuously right. Two of every four rows are
    # members and doubling keeps that true.
    assert_equal(Int(sum_of(want).value), len(col) // 2)

    var padded: List[Int] = [4, 17]
    while len(padded) < 20:
        padded.append(1000 + len(padded))
    var big = numbers(padded)

    for got in [is_in(col, small), is_in(col, big)]:
        assert_equal(len(got), MORSEL_ROWS + 4)
        assert_equal(
            Int(sum_of(not_equal(got, want)).value), 0, "rows disagreeing"
        )


def test_a_text_column_either_side_of_the_morsel_split_matches_the_twin() raises:
    def unit() -> StringArray:
        return strings_from_list(
            ["13", "a value far too long to sit inside a view", "31", "17"]
        )

    var col = unit()
    while len(col) < MORSEL_ROWS:
        var pair = List[StringArray]()
        pair.append(col.copy())
        pair.append(col.copy())
        col = concat_strings(pair)
    assert_equal(len(col), MORSEL_ROWS)
    var tail = List[StringArray]()
    tail.append(col^)
    tail.append(unit())
    col = concat_strings(tail)

    var small = strings_from_list(
        ["13", "a value far too long to sit inside a view"]
    )
    var want = text_is_in_scalar(col, small)
    assert_equal(len(want), MORSEL_ROWS + 4)
    assert_equal(Int(sum_of(want).value), len(col) // 2)

    var padded: List[String] = [
        "13",
        "a value far too long to sit inside a view",
    ]
    while len(padded) < 20:
        padded.append("absent " + String(len(padded)))
    var big = strings_from_list(padded)

    for got in [text_is_in(col, small), text_is_in(col, big)]:
        assert_equal(len(got), MORSEL_ROWS + 4)
        assert_equal(
            Int(sum_of(not_equal(got, want)).value), 0, "rows disagreeing"
        )


def test_a_series_lookup_is_a_mask_and_then_a_filter() raises:
    # `is_in` hands back a mask rather than a series, the way `is_null` and the
    # four `str_` methods do, because a mask is what `filter` takes.
    var s = Series(
        "phone",
        strings_from_list(["13-123", "22-999", "31-000", "13-777"]),
    )
    var codes = Series("code", strings_from_list(["13", "31"]))
    var kept = s.filter(s.str_slice(0, 2).is_in(codes))
    assert_equal(len(kept), 3)
    assert_equal(kept.text(0), "13-123")
    assert_equal(kept.text(1), "31-000")
    assert_equal(kept.text(2), "13-777")


def test_a_series_lookup_across_types_is_refused() raises:
    var s = Series("n", Array[DType.int64](3))
    var codes = Series("code", strings_from_list(["13"]))
    with assert_raises(contains="cannot look up"):
        _ = s.is_in(codes)
    with assert_raises(contains="cannot look up"):
        _ = codes.is_in(s)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
