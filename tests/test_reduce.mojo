"""Tests for reducing a whole column, from `reduce_any` up to `DataFrame.agg`.

There is an oracle here that most test files do not get, and it is what most of
these assert against. Reducing a whole frame is the same question as grouping it
on a key that is the same for every row, and the grouped path has been correct
and tested for a long time. So the interesting property is not "the sum of these
four numbers is ten", it is "this answers exactly what the group by answers".
Every reduction is checked that way, over the same column, including the ones
that take the slow route, because the slow route is the one that would silently
stop being reached if the fast route grew a branch that swallowed a kind.

What the oracle cannot check is the shape of the result, since a group by over a
constant key produces a key column that `agg` has no reason to produce. So the
row count, the column names and the collision refusal are checked directly.

The null policy is checked directly too. It matches the grouped one by
construction, but a column that is entirely null is exactly where a fast route
that forgot to look at the validity would still return a plausible number, and
the oracle would not catch it if both routes were wrong in the same way. Here
`sum` and `count` give zero, `mean`, `min` and `max` give null, and `size` gives
the row count, which is pandas' answer.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.dtype.lists import ALL
from firepanda.frame.frame import DataFrame
from firepanda.frame.groupby import AggSpec
from firepanda.frame.series import Series
from firepanda.kernel.group import AggKind
from firepanda.kernel.reduce import reduce_any


def ints(values: List[Scalar[DType.int64]]) -> Array[DType.int64]:
    """Builds an int64 column."""
    return from_list(values)


def floats(values: List[Scalar[DType.float64]]) -> Array[DType.float64]:
    """Builds a float64 column."""
    return from_list(values)


def strings_of(values: List[String]) raises -> StringArray:
    """Builds a string column with no nulls."""
    var out = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        out.append(values[i].as_bytes())
    return out^.finish()


def sample() raises -> Array[DType.int64]:
    """Eight rows with two nulls, spread so an early exit would be caught."""
    var col = ints([7, 2, 9, 4, 1, 8, 3, 6])
    col.set_null(0)
    col.set_null(5)
    return col^


def sample_frame() raises -> DataFrame:
    """The sample column beside a float one and a constant key to group on."""
    var series = List[Series]()
    series.append(Series("k", ints([0, 0, 0, 0, 0, 0, 0, 0])))
    series.append(Series("v", sample()))
    series.append(
        Series("f", floats([1.5, 2.5, -3.0, 4.0, 0.25, 8.0, 3.0, 6.0]))
    )
    return DataFrame.from_series(series^)


def as_float(col: AnyArray, i: Int) raises -> Float64:
    """Reads one value of a runtime dtype column as a float.

    Widening to a float is what lets one assertion cover every reduction here.
    The values in these tests are small whole numbers or exact halves, so nothing
    is lost on the way through, and comparing two answers that were summed in a
    different order wants a tolerance anyway.

    Args:
        col: The column.
        i: The row.

    Returns:
        The value as a float.

    Raises:
        If the dtype is one this cannot read.
    """
    comptime for candidate in ALL:
        if col.dtype() == candidate:
            return col.as_typed[candidate]()[i].cast[DType.float64]()
    raise Error("reduce test: unsupported dtype")


def grouped(
    frame: DataFrame, kind: AggKind, column: String
) raises -> DataFrame:
    """Answers the same question the slow way, by grouping on the constant key.
    """
    var specs = List[AggSpec]()
    specs.append(AggSpec(column, kind, "answer"))
    return frame.group_by(["k"], specs^, True, False)


def agreed(kind: AggKind, column: String = "v") raises:
    """Asserts that `agg` and a group by on a constant key give the same row.

    Args:
        kind: The reduction to compare.
        column: Which column to reduce.

    Raises:
        If the two disagree, or if either refuses the reduction.
    """
    var frame = sample_frame()
    var slow = grouped(frame, kind, column)
    var specs = List[AggSpec]()
    specs.append(AggSpec(column, kind, "answer"))
    var fast = frame.agg(specs^)

    assert_equal(len(slow), 1)
    assert_equal(len(fast), 1)
    assert_true(fast[0].dtype() == slow[1].dtype())
    assert_equal(fast[0].is_valid(0), slow[1].is_valid(0))
    if not fast[0].is_valid(0):
        return
    assert_almost_equal(as_float(fast[0], 0), as_float(slow[1], 0))


def test_sum_agrees_with_the_group_by() raises:
    agreed(AggKind.SUM)


def test_mean_agrees_with_the_group_by() raises:
    agreed(AggKind.MEAN)


def test_a_mean_is_not_computed_from_a_wrapped_sum() raises:
    """The mean of four copies of 2 to the 62, whose int64 sum is zero.

    Four of them add up to exactly 2 to the 64, so an int64 accumulator lands on
    zero and a mean divided out of that is zero. The answer is 2 to the 62, which
    is what pandas gives, because pandas converts to float64 before dividing.
    This is not a rounding difference and it is not close: it was off by eighteen
    orders of magnitude on the pandas conformance corpus.

    The sum is checked on the same line because the fix must not touch it. numpy
    wraps an int64 sum too, so zero is the right answer there, and a mean that
    was fixed by widening the sum would have broken the sum to do it.
    """
    var big = Int64(1) << 62
    var values = ints([big, big, big, big])
    var column = AnyArray(values^)

    var summed = reduce_any(column, AggKind.SUM)
    assert_true(summed.dtype() == DType.int64, "a sum stays an int64")
    assert_equal(as_float(summed, 0), 0.0, "and wraps, as pandas does")

    var averaged = reduce_any(column, AggKind.MEAN)
    assert_true(averaged.dtype() == DType.float64)
    assert_equal(as_float(averaged, 0), 4.611686018427388e18)


def test_min_agrees_with_the_group_by() raises:
    agreed(AggKind.MIN)


def test_max_agrees_with_the_group_by() raises:
    agreed(AggKind.MAX)


def test_count_agrees_with_the_group_by() raises:
    agreed(AggKind.COUNT)


def test_size_agrees_with_the_group_by() raises:
    agreed(AggKind.SIZE)


def test_first_agrees_with_the_group_by() raises:
    agreed(AggKind.FIRST)


def test_last_agrees_with_the_group_by() raises:
    agreed(AggKind.LAST)


def test_var_agrees_with_the_group_by() raises:
    agreed(AggKind.VAR)


def test_std_agrees_with_the_group_by() raises:
    agreed(AggKind.STD)


def test_median_agrees_with_the_group_by() raises:
    agreed(AggKind.MEDIAN)


def test_quantile_agrees_with_the_group_by() raises:
    agreed(AggKind.quantile_at(0.9))


def test_nunique_agrees_with_the_group_by() raises:
    agreed(AggKind.NUNIQUE)


def test_a_float_column_agrees_too() raises:
    agreed(AggKind.SUM, "f")
    agreed(AggKind.MEAN, "f")
    agreed(AggKind.MIN, "f")
    agreed(AggKind.MAX, "f")


def test_a_sum_widens_the_way_the_group_by_widens() raises:
    var small = from_list[DType.int32]([1, 2, 3])
    var out = reduce_any(AnyArray(small^), AggKind.SUM)
    assert_equal(len(out), 1)
    assert_equal(out.as_typed[DType.int64]()[0], 6)


def test_a_column_of_all_nulls_follows_the_pandas_policy() raises:
    var col = ints([1, 2, 3])
    col.set_null(0)
    col.set_null(1)
    col.set_null(2)

    var summed = reduce_any(AnyArray(col.copy()), AggKind.SUM)
    assert_true(summed.is_valid(0))
    assert_equal(summed.as_typed[DType.int64]()[0], 0)

    var counted = reduce_any(AnyArray(col.copy()), AggKind.COUNT)
    assert_equal(counted.as_typed[DType.int64]()[0], 0)

    var sized = reduce_any(AnyArray(col.copy()), AggKind.SIZE)
    assert_equal(sized.as_typed[DType.int64]()[0], 3)

    assert_false(reduce_any(AnyArray(col.copy()), AggKind.MEAN).is_valid(0))
    assert_false(reduce_any(AnyArray(col.copy()), AggKind.MIN).is_valid(0))
    assert_false(reduce_any(AnyArray(col^), AggKind.MAX).is_valid(0))


def test_an_empty_column_sums_to_zero_and_has_no_minimum() raises:
    var col = Array[DType.int64](0)
    var summed = reduce_any(AnyArray(col.copy()), AggKind.SUM)
    assert_true(summed.is_valid(0))
    assert_equal(summed.as_typed[DType.int64]()[0], 0)
    assert_equal(
        reduce_any(AnyArray(col.copy()), AggKind.SIZE).as_typed[DType.int64]()[
            0
        ],
        0,
    )
    assert_false(reduce_any(AnyArray(col^), AggKind.MIN).is_valid(0))


def test_a_boolean_column_reduces_one_value_at_a_time() raises:
    # Two hundred rows so the whole block path runs, which is the one that has a
    # vector reduction in it everywhere except here.
    var flags = Array[DType.bool](200)
    for i in range(200):
        flags[i] = i % 7 == 3
    assert_false(
        reduce_any(AnyArray(flags.copy()), AggKind.MIN).as_typed[DType.bool]()[
            0
        ]
    )
    assert_true(
        reduce_any(AnyArray(flags.copy()), AggKind.MAX).as_typed[DType.bool]()[
            0
        ]
    )
    assert_equal(
        reduce_any(AnyArray(flags^), AggKind.SUM).as_typed[DType.uint64]()[0],
        29,
    )


def test_a_string_column_counts_without_reading_the_values() raises:
    var col = strings_of(["a", "bb", "ccc"])
    var counted = reduce_any(AnyArray(col^), AggKind.COUNT)
    assert_equal(counted.as_typed[DType.int64]()[0], 3)


def test_a_string_column_takes_the_grouped_route_for_a_minimum() raises:
    var col = strings_of(["pear", "apple", "fig"])
    var out = reduce_any(AnyArray(col^), AggKind.MIN)
    assert_equal(len(out), 1)
    ref got = out.strings()
    assert_equal(got[0], "apple")


def test_a_string_column_refuses_a_sum() raises:
    var col = strings_of(["a", "b"])
    with assert_raises():
        _ = reduce_any(AnyArray(col^), AggKind.SUM)


def test_agg_returns_one_row_named_by_the_specs() raises:
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))
    specs.append(AggSpec("v", AggKind.MEAN))
    specs.append(AggSpec("f", AggKind.MAX, "biggest"))
    var out = sample_frame().agg(specs^)
    assert_equal(len(out), 1)
    assert_equal(out.width(), 3)
    assert_equal(out.schema[0].name, "v_sum")
    assert_equal(out.schema[1].name, "v_mean")
    assert_equal(out.schema[2].name, "biggest")


def test_agg_refuses_two_outputs_with_the_same_name() raises:
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM, "total"))
    specs.append(AggSpec("f", AggKind.SUM, "total"))
    with assert_raises(contains="would both be called"):
        _ = sample_frame().agg(specs^)


def test_agg_refuses_an_empty_spec_list() raises:
    with assert_raises(contains="at least one"):
        _ = sample_frame().agg(List[AggSpec]())


def test_agg_refuses_a_column_that_is_not_there() raises:
    var specs = List[AggSpec]()
    specs.append(AggSpec("nope", AggKind.SUM))
    with assert_raises():
        _ = sample_frame().agg(specs^)


def test_agg_all_reduces_every_column_and_keeps_the_names() raises:
    var out = sample_frame().agg_all(AggKind.MAX)
    assert_equal(len(out), 1)
    assert_equal(out.width(), 3)
    assert_equal(out.schema[0].name, "k")
    assert_equal(out.schema[1].name, "v")
    assert_equal(out.schema[2].name, "f")
    assert_equal(out[1].as_typed[DType.int64]()[0], 9)
    assert_almost_equal(out[2].as_typed[DType.float64]()[0], 8.0)


def test_a_correlation_over_the_whole_frame_agrees_with_the_group_by() raises:
    var frame = sample_frame()
    var slow_specs = List[AggSpec]()
    slow_specs.append(AggSpec("v", "f", AggKind.CORR, "answer"))
    var slow = frame.group_by(["k"], slow_specs^, True, False)

    var specs = List[AggSpec]()
    specs.append(AggSpec("v", "f", AggKind.CORR, "answer"))
    var fast = frame.agg(specs^)

    assert_equal(len(fast), 1)
    assert_almost_equal(
        fast[0].as_typed[DType.float64]()[0],
        slow[1].as_typed[DType.float64]()[0],
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
