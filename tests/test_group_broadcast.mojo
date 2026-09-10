"""Tests for reducing each group and writing the answer back onto its rows.

This is the window aggregate with no frame and no ordering, and the thing that
makes it different from a group by is the height of what comes back: one row per
input row rather than one per group, in the input's order, with the input's
labels. Every test here checks that shape as well as the values, because a
version that quietly returned the group count of rows would still get the
arithmetic right.

The other difference is that a null key is not dropped. A group by drops it
because its result is one row per key; here the result is one row per input row,
and a dropped null key would be an input row with no answer.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_raises

from firepanda.array.array import Array
from firepanda.array.strings import strings_from_list
from firepanda.array.value import Value
from firepanda.frame.frame import DataFrame
from firepanda.frame.groupby import AggKind, AggSpec
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
    """Wraps one column name as a key list.

    Args:
        name: The column name.

    Returns:
        A list holding it.
    """
    var out = List[String]()
    out.append(name)
    return out^


def one(var spec: AggSpec) -> List[AggSpec]:
    """Wraps one spec as a spec list.

    Args:
        spec: The spec.

    Returns:
        A list holding it.
    """
    var out = List[AggSpec]()
    out.append(spec^)
    return out^


def test_a_group_sum_lands_on_every_row_of_the_group() raises:
    var df = DataFrame()
    df = df.with_column(Series("k", strings_from_list(["a", "b", "a", "b"])))
    df = df.with_column(Series("v", numbers([1, 10, 2, 20])))

    var got = df.group_broadcast(only("k"), one(AggSpec("v", AggKind.SUM)))
    assert_equal(got.rows, 4, "the result is as tall as the input")
    assert_equal(len(got.schema), 1, "only the aggregate comes back")

    var sums = got.column("v_sum").as_typed[DType.int64]()
    assert_equal(sums[0], 3)
    assert_equal(sums[1], 30)
    assert_equal(sums[2], 3)
    assert_equal(sums[3], 30)


def test_the_rows_stay_in_the_input_order() raises:
    # The keys here are not in order and the group by that answers this would
    # sort them. Nothing is sorted on this path, and row two has to hold the c
    # group's answer whatever order the groups were discovered in.
    var df = DataFrame()
    df = df.with_column(Series("k", numbers([9, 3, 9, 1, 3])))
    df = df.with_column(Series("v", numbers([5, 7, 6, 100, 8])))

    var got = df.group_broadcast(only("k"), one(AggSpec("v", AggKind.MAX)))
    var maxes = got.column("v_max").as_typed[DType.int64]()
    assert_equal(maxes[0], 6)
    assert_equal(maxes[1], 8)
    assert_equal(maxes[2], 6)
    assert_equal(maxes[3], 100)
    assert_equal(maxes[4], 8)


def test_a_null_key_gets_its_own_answer_rather_than_being_dropped() raises:
    var k = numbers([1, 2, 1, 2])
    k.set_null(1)
    k.set_null(3)
    var df = DataFrame()
    df = df.with_column(Series("k", k^))
    df = df.with_column(Series("v", numbers([4, 5, 6, 7])))

    var got = df.group_broadcast(only("k"), one(AggSpec("v", AggKind.SUM)))
    assert_equal(got.rows, 4, "a null keyed row was dropped")
    var sums = got.column("v_sum").as_typed[DType.int64]()
    # The two ones are a group and the two nulls are a group.
    assert_equal(sums[0], 10)
    assert_equal(sums[1], 12)
    assert_equal(sums[2], 10)
    assert_equal(sums[3], 12)


def test_a_null_value_is_skipped_the_way_a_group_by_skips_it() raises:
    var v = numbers([1, 2, 3, 4])
    v.set_null(0)
    var df = DataFrame()
    df = df.with_column(Series("k", strings_from_list(["a", "a", "b", "b"])))
    df = df.with_column(Series("v", v^))

    var got = df.group_broadcast(only("k"), one(AggSpec("v", AggKind.SUM)))
    var sums = got.column("v_sum").as_typed[DType.int64]()
    assert_equal(
        sums[0], 2, "the null was counted as a zero rather than skipped"
    )
    assert_equal(sums[1], 2)
    assert_equal(sums[2], 7)


def test_a_mean_over_a_group_reaches_every_row_of_it() raises:
    # q17's shape, a group's average compared against each of its own rows.
    var df = DataFrame()
    df = df.with_column(Series("part", numbers([1, 1, 1, 2, 2])))
    df = df.with_column(Series("qty", numbers([10, 20, 60, 4, 8])))

    var avg = df.group_broadcast(
        only("part"), one(AggSpec("qty", AggKind.MEAN))
    )
    var means = avg.column("qty_mean").as_typed[DType.float64]()
    assert_equal(means[0], 30.0)
    assert_equal(means[2], 30.0)
    assert_equal(means[3], 6.0)

    # And the filter it exists for, which is a mask over the input's own rows
    # and never a join back against the keys.
    var small = df.filter(
        (df.column("qty") < avg.column("qty_mean")).as_typed[DType.bool]()
    )
    assert_equal(small.rows, 3)
    var kept = small.column("qty").as_typed[DType.int64]()
    assert_equal(kept[0], 10)
    assert_equal(kept[1], 20)
    assert_equal(kept[2], 4)


def test_the_group_used_as_a_filter_on_its_own_rows() raises:
    # q18's shape. The rows wanted back are the input's, not the group's, so a
    # group by would have to join its surviving keys back on.
    var df = DataFrame()
    df = df.with_column(Series("order", numbers([1, 1, 2, 2, 3])))
    df = df.with_column(Series("qty", numbers([200, 200, 5, 5, 400])))

    var totals = df.group_broadcast(
        only("order"), one(AggSpec("qty", AggKind.SUM))
    )
    var big = df.filter(
        (totals.column("qty_sum") > Value(Int64(300))).as_typed[DType.bool]()
    )
    assert_equal(big.rows, 3)
    var orders = big.column("order").as_typed[DType.int64]()
    assert_equal(orders[0], 1)
    assert_equal(orders[1], 1)
    assert_equal(orders[2], 3)


def test_two_keys_and_two_aggregates() raises:
    var df = DataFrame()
    df = df.with_column(Series("a", numbers([1, 1, 2, 1])))
    df = df.with_column(Series("b", numbers([7, 8, 7, 7])))
    df = df.with_column(Series("v", numbers([3, 4, 5, 6])))

    var keys = List[String]()
    keys.append("a")
    keys.append("b")
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))
    specs.append(AggSpec("v", AggKind.COUNT))

    var got = df.group_broadcast(keys, specs)
    assert_equal(got.rows, 4)
    assert_equal(len(got.schema), 2)
    var sums = got.column("v_sum").as_typed[DType.int64]()
    var counts = got.column("v_count").as_typed[DType.int64]()
    # Rows zero and three share (1,7), row one is (1,8), row two is (2,7).
    assert_equal(sums[0], 9)
    assert_equal(sums[3], 9)
    assert_equal(sums[1], 4)
    assert_equal(sums[2], 5)
    assert_equal(counts[0], 2)
    assert_equal(counts[1], 1)


def test_a_text_aggregate_broadcasts_too() raises:
    var df = DataFrame()
    df = df.with_column(Series("k", numbers([1, 2, 1, 2])))
    df = df.with_column(
        Series("name", strings_from_list(["pear", "fig", "apple", "quince"]))
    )

    var got = df.group_broadcast(only("k"), one(AggSpec("name", AggKind.MIN)))
    assert_equal(got.rows, 4)
    var mins = got.column("name_min").as_strings()
    assert_equal(mins[0], "apple")
    assert_equal(mins[1], "fig")
    assert_equal(mins[2], "apple")
    assert_equal(mins[3], "fig")


def test_every_row_its_own_group() raises:
    var df = DataFrame()
    df = df.with_column(Series("k", numbers([4, 5, 6])))
    df = df.with_column(Series("v", numbers([10, 20, 30])))

    var got = df.group_broadcast(only("k"), one(AggSpec("v", AggKind.SUM)))
    assert_equal(got.rows, 3)
    var sums = got.column("v_sum").as_typed[DType.int64]()
    assert_equal(sums[0], 10)
    assert_equal(sums[1], 20)
    assert_equal(sums[2], 30)


def test_an_empty_frame_broadcasts_nothing() raises:
    var df = DataFrame()
    df = df.with_column(Series("k", Array[DType.int64](0)))
    df = df.with_column(Series("v", Array[DType.int64](0)))

    var got = df.group_broadcast(only("k"), one(AggSpec("v", AggKind.SUM)))
    assert_equal(got.rows, 0)


def test_the_refusals() raises:
    var df = DataFrame()
    df = df.with_column(Series("k", numbers([1, 2])))
    df = df.with_column(Series("v", numbers([3, 4])))

    with assert_raises(contains="at least one aggregate"):
        _ = df.group_broadcast(only("k"), List[AggSpec]())

    var twice = List[String]()
    twice.append("k")
    twice.append("k")
    with assert_raises(contains="was given twice"):
        _ = df.group_broadcast(twice, one(AggSpec("v", AggKind.SUM)))

    var same = List[AggSpec]()
    same.append(AggSpec("v", AggKind.SUM))
    same.append(AggSpec("v", AggKind.SUM))
    with assert_raises(contains="both be called"):
        _ = df.group_broadcast(only("k"), same)

    with assert_raises():
        _ = df.group_broadcast(only("nope"), one(AggSpec("v", AggKind.SUM)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
