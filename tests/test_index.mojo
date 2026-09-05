"""Tests for row labels.

The index has two representations of the same thing and almost every bug it can
have is one of them disagreeing with the other, so most of what is here checks a
property against both. A range and the array it materializes to must answer
`__len__` the same, label row `i` the same, and survive a slice the same.

The other half is `is_default`, which is narrower than `is_range` and is the
question the conformance suite is actually asking. A frame that has been sliced
from the front still carries no information in its labels; one that has been
sliced from the back carries the offset, and it is a range either way. There is a
test for each side of that.
"""

from std.collections import Optional
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.frame.frame import DataFrame
from firepanda.frame.index import Index
from firepanda.frame.series import Series


def unnamed() -> Optional[String]:
    """A level name of `None`, spelled once because it is written a lot."""
    return Optional[String]()


def labels(values: List[Int64]) raises -> Index:
    """An index over the given labels, unnamed."""
    return Index(AnyArray(from_list[DType.int64](values)), unnamed())


def label_at(col: AnyArray, i: Int) raises -> Int:
    """The label in row `i`, as an integer."""
    return Int(col.as_typed[DType.int64]()[i])


def test_a_new_index_is_the_default_range() raises:
    var index = Index(10)
    assert_equal(len(index), 10, "ten rows")
    assert_true(index.is_range(), "a new index is a range")
    assert_true(index.is_default(), "and it starts at zero")
    assert_false(Bool(index.name), "and it is unnamed")


def test_a_range_materializes_to_the_labels_it_stands_for() raises:
    var built = Index(10).materialize()
    assert_equal(len(built), 10, "ten labels")
    for i in range(10):
        assert_equal(
            label_at(built, i), i, "label " + String(i) + " is its position"
        )
    assert_equal(built.null_count(), 0, "a range has no missing label")


def test_a_range_that_does_not_start_at_zero_is_not_default() raises:
    var index = Index(5, 5, unnamed())
    assert_true(index.is_range(), "still a range, since nothing was gathered")
    assert_false(index.is_default(), "but the offset is information")
    var built = index.materialize()
    for i in range(5):
        assert_equal(label_at(built, i), 5 + i, "label " + String(i))


def test_slicing_from_the_front_keeps_the_default() raises:
    var index = Index(10).slice(0, 4)
    assert_equal(len(index), 4, "four rows")
    assert_true(index.is_default(), "head keeps the labels it started with")


def test_slicing_from_the_back_keeps_a_range_and_loses_the_default() raises:
    var index = Index(10).slice(6, 10)
    assert_equal(len(index), 4, "four rows")
    assert_true(index.is_range(), "no array was built")
    assert_false(index.is_default(), "tail is labelled six to nine")
    var built = index.materialize()
    for i in range(4):
        assert_equal(label_at(built, i), 6 + i, "label " + String(i))


def test_slicing_a_materialized_index_slices_the_labels() raises:
    var index = labels([Int64(10), 20, 30, 40, 50]).slice(1, 4)
    assert_equal(len(index), 3, "three rows")
    assert_false(index.is_range(), "it was an array before and still is")
    var built = index.materialize()
    assert_equal(label_at(built, 0), 20, "first kept label")
    assert_equal(label_at(built, 2), 40, "last kept label")


def test_taking_materializes_and_permutes() raises:
    var index = Index(5).take([3, 0, 4])
    assert_equal(len(index), 3, "three rows")
    assert_false(index.is_range(), "a gather cannot leave a range behind")
    assert_false(index.is_default(), "and so cannot leave the default either")
    var built = index.materialize()
    assert_equal(label_at(built, 0), 3, "first gathered label")
    assert_equal(label_at(built, 1), 0, "second")
    assert_equal(label_at(built, 2), 4, "third")


def test_a_negative_position_gathers_a_null_label() raises:
    var built = Index(5).take([1, -1, 2]).materialize()
    assert_true(built.is_valid(0), "a real row keeps its label")
    assert_false(built.is_valid(1), "a row that was not there has none")
    assert_true(built.is_valid(2), "and the row after it is unaffected")


def test_filtering_keeps_the_labels_of_the_rows_it_keeps() raises:
    var mask = from_list[DType.bool]([False, True, False, True, True])
    var built = Index(5).filter(mask).materialize()
    assert_equal(len(built), 3, "three rows survive")
    assert_equal(label_at(built, 0), 1, "the labels are positions, not ranks")
    assert_equal(label_at(built, 1), 3, "second kept label")
    assert_equal(label_at(built, 2), 4, "third kept label")


def test_filtering_a_gathered_index_keeps_the_original_labels() raises:
    var mask = from_list[DType.bool]([True, False, True])
    var built = labels([Int64(7), 8, 9]).filter(mask).materialize()
    assert_equal(len(built), 2, "two rows survive")
    assert_equal(label_at(built, 0), 7, "labels come from the index")
    assert_equal(label_at(built, 1), 9, "and not from the mask positions")


def test_a_name_survives_every_operation_that_chooses_rows() raises:
    var named = Index(10).renamed(Optional[String]("key"))
    assert_equal(named.name.value(), "key", "renamed")
    assert_equal(named.slice(2, 5).name.value(), "key", "through a slice")
    assert_equal(named.take([1, 2]).name.value(), "key", "and a take")
    var mask = from_list[DType.bool](
        [True, False, True, False, True, False, True, False, True, False]
    )
    assert_equal(named.filter(mask).name.value(), "key", "and a filter")


def test_an_empty_index_is_still_a_default_range() raises:
    var index = Index(0)
    assert_equal(len(index), 0, "no rows")
    assert_true(index.is_default(), "and nothing to be non default about")
    assert_equal(len(index.materialize()), 0, "materializing gives nothing")


def gathered(index: Index) raises -> Index:
    """The same index with its labels built out, which is the slow twin.

    Everything `take` and `filter` do on a range they also have to do on an array,
    and the array route is the one that is obviously right because it is the
    kernel doing the work. This turns a range into an array without changing what
    it labels, so the two routes can be run against each other.
    """
    return Index(index.materialize(), Optional[String](copy=index.name))


def test_the_range_fast_path_agrees_with_gathering_the_labels() raises:
    var index = Index(3, 40, unnamed())
    var picks = List[Int]()
    for i in range(40):
        picks.append((i * 17) % 40)
    picks.append(-1)
    picks.append(0)
    var fast = index.take(picks).materialize()
    var slow = gathered(index).take(picks).materialize()
    assert_equal(len(fast), len(slow), "the two routes keep the same rows")
    for i in range(len(fast)):
        assert_equal(
            fast.is_valid(i), slow.is_valid(i), "missing row " + String(i)
        )
        if fast.is_valid(i):
            assert_equal(
                label_at(fast, i), label_at(slow, i), "label " + String(i)
            )


def test_the_filter_fast_path_agrees_with_filtering_the_labels() raises:
    var index = Index(7, 130, unnamed())
    var bits = List[Scalar[DType.bool]]()
    for i in range(130):
        bits.append(i % 3 != 0)
    var mask = from_list[DType.bool](bits)
    # A null in the mask drops the row, and the two routes have to agree about
    # that as well, so one is put either side of the word boundary at 64.
    mask.set_null(63)
    mask.set_null(64)
    var fast = index.filter(mask).materialize()
    var slow = gathered(index).filter(mask).materialize()
    assert_equal(len(fast), len(slow), "the two routes keep the same rows")
    for i in range(len(fast)):
        assert_equal(label_at(fast, i), label_at(slow, i), "label " + String(i))
    assert_equal(fast.null_count(), 0, "a filter of a range writes no null")


def counted(name: String, n: Int) raises -> Series:
    """An int64 series holding zero to n minus one, so a row is its own value.
    """
    var col = Array[DType.int64](n)
    for i in range(n):
        col.set_valid(i, Int64(i))
    return Series(name, col^)


def frame_of(n: Int) raises -> DataFrame:
    """A one column frame of n rows, each holding its own position."""
    var columns = List[Series]()
    columns.append(counted("a", n))
    return DataFrame.from_series(columns^)


def test_a_frame_starts_with_the_default_index() raises:
    assert_true(frame_of(10).index.is_default(), "nothing has happened to it")


def test_a_frame_head_keeps_the_default_and_tail_does_not() raises:
    var df = frame_of(10)
    assert_true(
        df.head(5).index.is_default(), "the first five are still 0 to 4"
    )
    var labels = Index(copy=df.tail(5).index)
    assert_false(labels.is_default(), "the last five are 5 to 9")
    var built = labels.materialize()
    for i in range(5):
        assert_equal(label_at(built, i), 5 + i, "label " + String(i))


def test_a_frame_sort_carries_the_labels_of_the_rows_it_moved() raises:
    var columns = List[Series]()
    columns.append(Series("k", from_list[DType.int64]([Int64(30), 10, 20])))
    var df = DataFrame.from_series(columns^)
    var built = df.sort_values(["k"], [False], [False]).index.materialize()
    assert_equal(label_at(built, 0), 1, "the smallest key was row 1")
    assert_equal(label_at(built, 1), 2, "then row 2")
    assert_equal(label_at(built, 2), 0, "then row 0")


def test_dropping_nulls_leaves_holes_in_the_labels() raises:
    var col = Array[DType.int64](5)
    for i in range(5):
        col.set_valid(i, Int64(i))
    col.set_null(1)
    col.set_null(3)
    var columns = List[Series]()
    columns.append(Series("a", col^))
    var built = DataFrame.from_series(columns^).drop_nulls().index.materialize()
    assert_equal(len(built), 3, "three rows survive")
    assert_equal(label_at(built, 0), 0, "row 0 kept its label")
    assert_equal(label_at(built, 1), 2, "and row 2 kept its own")
    assert_equal(label_at(built, 2), 4, "and so did row 4")


def test_a_series_carries_labels_the_same_way_a_frame_does() raises:
    var s = counted("a", 10)
    assert_true(s.index.is_default(), "a new series is a default range")
    var built = s.tail(3).index.materialize()
    for i in range(3):
        assert_equal(label_at(built, i), 7 + i, "label " + String(i))


def test_labels_survive_two_operations_in_a_row() raises:
    var built = frame_of(10).tail(6).head(3).index.materialize()
    assert_equal(len(built), 3, "three rows")
    for i in range(3):
        assert_equal(label_at(built, i), 4 + i, "label " + String(i))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
