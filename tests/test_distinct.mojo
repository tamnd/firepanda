"""Tests for removing the repeated rows of a frame.

The two things worth aiming at are the two that make this different from a group
by with no reductions: it gives back every column and not just the keys, and it
keeps a row whose key holds a null. Both have a test that fails if the method
ever becomes a thin wrapper over `group_by`.

The third is the order, which is the order the rows appear in and not the order
of the keys. Every factorize route hands its ordinals out in first appearance
order, so the representative rows are already ascending and the method takes them
as they stand; the walk that handles a grouping where they are not is `_first_rows`
and nothing in the library reaches it, so it has a test of its own here rather
than a frame that cannot produce the input it wants.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_raises

from firepanda.array.array import Array
from firepanda.array.strings import strings_from_list
from firepanda.frame.frame import DataFrame, _first_rows
from firepanda.frame.series import Series


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


def only(name: String) -> List[String]:
    """Wraps one column name as a subset.

    Args:
        name: The column name.

    Returns:
        A list holding it.
    """
    var out = List[String]()
    out.append(name)
    return out^


def test_every_column_comes_back_and_not_just_the_keys() raises:
    # A group by with no reductions gives the keys. This gives the whole row,
    # which is the difference that makes it `drop_duplicates` and not that.
    var df = DataFrame()
    df = df.with_column(Series("k", strings_from_list(["a", "b", "a"])))
    df = df.with_column(Series("v", numbers([10, 20, 30])))

    var out = df.drop_duplicates(only("k"))

    assert_equal(out.rows, 2)
    assert_equal(len(out.schema), 2, "the value column was dropped")
    var v = out.column("v").as_typed[DType.int64]()
    # The first appearance of each key, so ten and twenty and not thirty.
    assert_equal(v[0], 10)
    assert_equal(v[1], 20)


def test_a_repeat_is_decided_by_the_subset_alone() raises:
    # The value column differs on every row, so a frame wide comparison would
    # keep all four. The subset says only the key counts.
    var df = DataFrame()
    df = df.with_column(Series("k", strings_from_list(["a", "b", "b", "c"])))
    df = df.with_column(Series("v", numbers([0, 1, 2, 3])))

    var out = df.drop_duplicates(only("k"))
    assert_equal(out.rows, 3)
    var v = out.column("v").as_typed[DType.int64]()
    assert_equal(v[1], 1, "kept the second b rather than the first")


def test_a_null_key_is_a_group_like_any_other() raises:
    # `group_by` drops these by default, which is pandas' rule there. It is not
    # pandas' rule here: a null is a value when deciding whether a row repeats,
    # and two nulls repeat each other.
    var k = numbers([1, 2, 2, 1, 7])
    k.set_null(1)
    k.set_null(2)
    var df = DataFrame()
    df = df.with_column(Series("k", k^))

    var out = df.drop_duplicates(only("k"))
    # One, the null, and seven. The two nulls collapse into one row rather than
    # both surviving and rather than both being dropped.
    assert_equal(out.rows, 3)
    var got = out.column("k").as_typed[DType.int64]()
    assert_equal(got[0], 1)
    assert_false(got.is_valid(1), "the null row was dropped")
    assert_equal(got[2], 7)


def test_with_no_subset_every_column_counts() raises:
    var df = DataFrame()
    df = df.with_column(Series("a", numbers([1, 1, 1])))
    df = df.with_column(Series("b", numbers([2, 3, 2])))

    var out = df.drop_duplicates()
    assert_equal(out.rows, 2)
    var b = out.column("b").as_typed[DType.int64]()
    assert_equal(b[0], 2)
    assert_equal(b[1], 3)


def test_the_walk_finds_the_first_row_of_each_group() raises:
    # `_first_rows` is the fallback for a grouping whose representative rows are
    # not ascending. Every factorize route hands its ordinals out in first
    # appearance order, so nothing in the library reaches it, which is why it is
    # tested here directly rather than through a frame that cannot produce one.
    var codes = Array[DType.uint32](7)
    var written: List[Int] = [2, 0, 2, 1, 0, 1, 2]
    for i in range(len(written)):
        codes.set_valid(i, UInt32(written[i]))

    var got = _first_rows(codes, 3, len(written))
    assert_equal(len(got), 3)
    # Group two is introduced on row zero, group zero on row one, group one on
    # row three, and the answer is in row order and not in ordinal order.
    assert_equal(got[0], 0)
    assert_equal(got[1], 1)
    assert_equal(got[2], 3)

    with assert_raises(contains="out of"):
        _ = _first_rows(codes, 2, len(written))


def test_the_rows_come_back_in_the_order_they_appeared() raises:
    # Two narrow integer keys, which is the packed route. Whichever route runs,
    # the answer has to be the first appearance of each tuple in input order,
    # and sorted by key this would come out starting with (1,8) instead.
    var df = DataFrame()
    df = df.with_column(Series("a", numbers([3, 1, 3, 2, 1, 2])))
    df = df.with_column(Series("b", numbers([9, 8, 9, 7, 8, 6])))

    var out = df.drop_duplicates()
    assert_equal(out.rows, 4)
    var a = out.column("a").as_typed[DType.int64]()
    var b = out.column("b").as_typed[DType.int64]()
    # (3,9) (1,8) (2,7) (2,6), which is the input with the two repeats gone and
    # nothing reordered. Sorted by key it would start with (1,8).
    assert_equal(a[0], 3)
    assert_equal(b[0], 9)
    assert_equal(a[1], 1)
    assert_equal(b[1], 8)
    assert_equal(a[2], 2)
    assert_equal(b[2], 7)
    assert_equal(a[3], 2)
    assert_equal(b[3], 6)


def test_a_frame_with_nothing_repeated_comes_back_whole() raises:
    var df = DataFrame()
    df = df.with_column(Series("a", numbers([5, 6, 7])))
    var out = df.drop_duplicates()
    assert_equal(out.rows, 3)
    var a = out.column("a").as_typed[DType.int64]()
    assert_equal(a[0], 5)
    assert_equal(a[2], 7)


def test_an_empty_frame_stays_empty() raises:
    var df = DataFrame()
    df = df.with_column(Series("a", Array[DType.int64](0)))
    var out = df.drop_duplicates()
    assert_equal(out.rows, 0)


def test_a_missing_or_repeated_column_is_refused() raises:
    var df = DataFrame()
    df = df.with_column(Series("a", numbers([1, 2])))

    with assert_raises():
        _ = df.drop_duplicates(only("nope"))

    var twice = List[String]()
    twice.append("a")
    twice.append("a")
    with assert_raises(contains="was given twice"):
        _ = df.drop_duplicates(twice)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
