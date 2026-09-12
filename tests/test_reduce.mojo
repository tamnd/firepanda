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

Three reductions get none of that. The product and the two truth values have a
whole column implementation and no grouped one, so there is no oracle to compare
them against and every assertion about them is written out by hand against what
a running pandas answers. The last test in that group asserts the missing
grouped branch on purpose, so that adding it later is a test to delete rather
than a silent change of behaviour.
"""

from std.math import isnan
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
from firepanda.exec import MORSEL_ROWS
from firepanda.frame.frame import DataFrame
from firepanda.frame.groupby import AggSpec
from firepanda.frame.series import Series
from firepanda.kernel.group import AggKind
from firepanda.kernel.reduce import distinct_count_any, reduce_any


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


def test_sem_agrees_with_the_group_by() raises:
    agreed(AggKind.SEM)


def test_skew_agrees_with_the_group_by() raises:
    agreed(AggKind.SKEW)


def test_median_agrees_with_the_group_by() raises:
    agreed(AggKind.MEDIAN)


def test_quantile_agrees_with_the_group_by() raises:
    agreed(AggKind.quantile_at(0.9))


def test_nunique_agrees_with_the_group_by() raises:
    agreed(AggKind.NUNIQUE)


def test_a_distinct_count_leaves_the_nulls_out() raises:
    # pandas' rule for `nunique`, and the rule the grouped form here already
    # follows. Two of the eight rows are null and the other six are distinct.
    assert_equal(distinct_count_any(AnyArray(sample())), 6)


def test_an_empty_string_is_a_value_and_a_null_is_not() raises:
    # One line apart in the code and a whole answer apart in a dataset that
    # spells its missing text as an empty string, which the ClickBench hits table
    # does on most of its text columns.
    var col = strings_of(["a", "", "a", "b", ""])
    assert_equal(distinct_count_any(AnyArray(col^)), 3)

    var builder = StringBuilder(capacity=3)
    builder.append(String("a").as_bytes())
    builder.append_null()
    builder.append(String("b").as_bytes())
    assert_equal(distinct_count_any(AnyArray(builder^.finish())), 2)


def test_the_two_integer_routes_agree() raises:
    # The same values twice, once packed into a range a bit set can cover and
    # once spread far enough that no bit set would be worth having. The route is
    # chosen by the spread and the answer must not be.
    var dense = Array[DType.int64](4096)
    var sparse = Array[DType.int64](4096)
    for i in range(4096):
        dense[i] = Int64(i % 700)
        sparse[i] = Int64(i % 700) * 1_000_000_007
    assert_equal(distinct_count_any(AnyArray(dense^)), 700)
    assert_equal(distinct_count_any(AnyArray(sparse^)), 700)


def test_a_distinct_count_of_a_negative_range_counts_from_the_minimum() raises:
    # The bit set indexes from the column's minimum rather than from zero, so a
    # column that never holds a non-negative number still takes the cheap route.
    var col = ints([-9, -4, -9, -1, -4])
    assert_equal(distinct_count_any(AnyArray(col^)), 3)


def test_a_distinct_count_of_nothing_is_zero() raises:
    var empty = Array[DType.int64](0)
    assert_equal(distinct_count_any(AnyArray(empty^)), 0)

    var none = ints([5, 5, 5])
    none.set_null(0)
    none.set_null(1)
    none.set_null(2)
    assert_equal(distinct_count_any(AnyArray(none^)), 0)


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

    # The mean answers in float64 and there says missing with a NaN in a row that
    # stays valid, which is what a pandas float column does. The two extremes
    # keep the column's own dtype, int64 here, where a NaN is not a value that
    # exists and a null is the only spelling available. See #170.
    var averaged = reduce_any(AnyArray(col.copy()), AggKind.MEAN)
    assert_true(averaged.is_valid(0), "the mean should not be null")
    assert_true(
        isnan(averaged.as_typed[DType.float64]()[0]),
        "the mean of nothing should be NaN",
    )
    assert_false(reduce_any(AnyArray(col.copy()), AggKind.MIN).is_valid(0))
    assert_false(reduce_any(AnyArray(col^), AggKind.MAX).is_valid(0))


def test_an_all_null_float_column_reduces_to_nan_and_not_to_a_null() raises:
    # The same column one dtype over. The minimum and the maximum keep the
    # column's own dtype rather than widening, so which spelling of missing they
    # use is the column's decision: the int64 above has no NaN to write and this
    # one does. The conformance suite reported `null, expected nan` on exactly
    # this, for `basics/min` and `basics/max` on `float64_all_null`.
    var col = Array[DType.float64](3)
    col.set_null(0)
    col.set_null(1)
    col.set_null(2)

    for kind in [AggKind.MEAN, AggKind.MIN, AggKind.MAX]:
        var reduced = reduce_any(AnyArray(col.copy()), kind)
        assert_true(reduced.is_valid(0), String(kind, " should not be null"))
        assert_true(
            isnan(reduced.as_typed[DType.float64]()[0]),
            String(kind, " should be NaN"),
        )


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


def test_a_product_multiplies_only_the_values_that_are_there() raises:
    # The one assertion in this file that would still pass if the kernel were
    # wrong in the most likely way, so it is worth saying what the likely way is.
    # A sum is allowed to ignore the validity bitmap because a null holds a zero,
    # and a product that copied that loop would multiply those zeros in and
    # answer nothing at all. The sample column has two nulls in it and 1296 is
    # the product of the six values that are left.
    var out = reduce_any(AnyArray(sample()), AggKind.PROD)
    assert_true(out.is_valid(0))
    assert_equal(out.as_typed[DType.int64]()[0], 1296)


def test_a_product_of_no_values_is_one() raises:
    var empty = Array[DType.int64](0)
    var over_nothing = reduce_any(AnyArray(empty^), AggKind.PROD)
    assert_true(over_nothing.is_valid(0))
    assert_equal(over_nothing.as_typed[DType.int64]()[0], 1)

    var blank = ints([4, 5, 6])
    blank.set_null(0)
    blank.set_null(1)
    blank.set_null(2)
    # One and not a null, which is the identity rather than a missing answer and
    # is what pandas hands back. A minimum over the same column has no answer,
    # and the difference is that an empty product is defined and an empty
    # minimum is not.
    var over_nulls = reduce_any(AnyArray(blank^), AggKind.PROD)
    assert_true(over_nulls.is_valid(0))
    assert_equal(over_nulls.as_typed[DType.int64]()[0], 1)


def test_a_product_steps_over_a_nan_the_way_the_extremes_do() raises:
    var col = floats([1.5, 0.0, -2.0, 4.0])
    col[1] = Float64("nan")
    var out = reduce_any(AnyArray(col^), AggKind.PROD)
    assert_almost_equal(out.as_typed[DType.float64]()[0], -12.0)


def test_a_product_wraps_rather_than_refusing_to_answer() raises:
    # pandas wraps here because numpy wraps, and two values of two to the
    # fortieth multiply to exactly nothing in int64. Copying that is the choice,
    # since an answer that disagrees with pandas is worse than one that is
    # obviously the wrong size.
    var col = ints([1 << 40, 1 << 40])
    var out = reduce_any(AnyArray(col^), AggKind.PROD)
    assert_equal(out.as_typed[DType.int64]()[0], 0)


def test_a_product_over_a_long_column_agrees_with_a_short_one() raises:
    # Past one morsel the product runs on every core and the slots are folded
    # afterwards. Every value is one except four of them, so the answer is small
    # enough to write down and the parallel fold is still the thing being tested.
    var rows = 300_000
    var col = Array[DType.float64](rows)
    for i in range(rows):
        col[i] = 1.0
    col[7] = 2.0
    col[MORSEL_ROWS + 3] = 3.0
    col[2 * MORSEL_ROWS + 11] = 5.0
    col[rows - 1] = 7.0
    col.set_null(9)
    var out = reduce_any(AnyArray(col^), AggKind.PROD)
    assert_almost_equal(out.as_typed[DType.float64]()[0], 210.0)


def test_any_and_all_read_a_value_as_true_when_it_is_not_zero() raises:
    var mixed = ints([0, 1, 0])
    assert_true(
        reduce_any(AnyArray(mixed.copy()), AggKind.ANY).as_typed[DType.bool]()[
            0
        ]
    )
    assert_false(
        reduce_any(AnyArray(mixed^), AggKind.ALL).as_typed[DType.bool]()[0]
    )

    var zeros = ints([0, 0, 0])
    assert_false(
        reduce_any(AnyArray(zeros.copy()), AggKind.ANY).as_typed[DType.bool]()[
            0
        ]
    )
    assert_false(
        reduce_any(AnyArray(zeros^), AggKind.ALL).as_typed[DType.bool]()[0]
    )


def test_a_column_with_no_values_answers_the_identity_of_the_operator() raises:
    var empty = Array[DType.int64](0)
    assert_false(
        reduce_any(AnyArray(empty.copy()), AggKind.ANY).as_typed[DType.bool]()[
            0
        ]
    )
    # True, and not because anything in the column was true. An `all` over
    # nothing is the identity of and, the same way an `any` over nothing is the
    # identity of or, and pandas answers both that way.
    assert_true(
        reduce_any(AnyArray(empty^), AggKind.ALL).as_typed[DType.bool]()[0]
    )

    var blank = ints([1, 1])
    blank.set_null(0)
    blank.set_null(1)
    assert_false(
        reduce_any(AnyArray(blank.copy()), AggKind.ANY).as_typed[DType.bool]()[
            0
        ]
    )
    assert_true(
        reduce_any(AnyArray(blank^), AggKind.ALL).as_typed[DType.bool]()[0]
    )


def test_a_nan_is_missing_rather_than_true() raises:
    # The one that a loop asking only about zero would get wrong, because a NaN
    # is not equal to zero and would sail through as a value that is true.
    var col = Array[DType.float64](2)
    col[0] = Float64("nan")
    col[1] = Float64("nan")
    assert_false(
        reduce_any(AnyArray(col.copy()), AggKind.ANY).as_typed[DType.bool]()[0]
    )
    assert_true(
        reduce_any(AnyArray(col^), AggKind.ALL).as_typed[DType.bool]()[0]
    )


def test_a_boolean_column_answers_both_truth_values() raises:
    # Two hundred rows so the whole block path runs, which for booleans is the
    # scalar loop under it rather than the vector unit.
    var flags = Array[DType.bool](200)
    for i in range(200):
        flags[i] = i % 7 == 3
    assert_true(
        reduce_any(AnyArray(flags.copy()), AggKind.ANY).as_typed[DType.bool]()[
            0
        ]
    )
    assert_false(
        reduce_any(AnyArray(flags^), AggKind.ALL).as_typed[DType.bool]()[0]
    )


def test_the_truth_values_fold_across_morsels() raises:
    var rows = 300_000
    var col = Array[DType.int64](rows)
    for i in range(rows):
        col[i] = 1
    assert_true(
        reduce_any(AnyArray(col.copy()), AggKind.ALL).as_typed[DType.bool]()[0]
    )
    # The one zero is in the last morsel, which is where a fold that stopped at
    # the first slot rather than reading all of them would miss it.
    col[rows - 1] = 0
    assert_false(
        reduce_any(AnyArray(col.copy()), AggKind.ALL).as_typed[DType.bool]()[0]
    )
    assert_true(
        reduce_any(AnyArray(col^), AggKind.ANY).as_typed[DType.bool]()[0]
    )


def test_a_text_column_reads_an_empty_string_as_false() raises:
    var words = strings_of(["a", "b"])
    assert_true(
        reduce_any(AnyArray(words^), AggKind.ALL).as_typed[DType.bool]()[0]
    )

    var one_blank = strings_of(["", "b"])
    assert_true(
        reduce_any(AnyArray(one_blank.copy()), AggKind.ANY).as_typed[
            DType.bool
        ]()[0]
    )
    assert_false(
        reduce_any(AnyArray(one_blank^), AggKind.ALL).as_typed[DType.bool]()[0]
    )

    var blanks = strings_of(["", ""])
    assert_false(
        reduce_any(AnyArray(blanks.copy()), AggKind.ANY).as_typed[DType.bool]()[
            0
        ]
    )
    assert_false(
        reduce_any(AnyArray(blanks^), AggKind.ALL).as_typed[DType.bool]()[0]
    )


def test_a_text_column_of_nulls_answers_the_identity_too() raises:
    var out = StringBuilder(capacity=2)
    out.append_null()
    out.append_null()
    var col = out^.finish()
    assert_false(
        reduce_any(AnyArray(col.copy()), AggKind.ANY).as_typed[DType.bool]()[0]
    )
    assert_true(
        reduce_any(AnyArray(col^), AggKind.ALL).as_typed[DType.bool]()[0]
    )


def test_a_string_column_refuses_a_product() raises:
    var col = strings_of(["a", "b"])
    with assert_raises(contains="multiplying two"):
        _ = reduce_any(AnyArray(col^), AggKind.PROD)


def test_the_three_new_reductions_have_no_grouped_form_yet() raises:
    # Every other reduction in this file is checked against the group by on a
    # constant key, and these three cannot be, because the grouped chain has no
    # branch for them. That is the whole reason they are asserted by hand above,
    # and this says out loud that the missing branch is known rather than
    # forgotten, so that whoever adds it has a test to delete.
    var frame = sample_frame()
    for kind in [AggKind.PROD, AggKind.ANY, AggKind.ALL]:
        var specs = List[AggSpec]()
        specs.append(AggSpec("v", kind, "answer"))
        with assert_raises(contains="no grouped one yet"):
            _ = frame.group_by(["k"], specs^, True, False)


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
