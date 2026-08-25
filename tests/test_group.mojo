"""Tests for grouped reductions, from the scatter loop up to `DataFrame.group_by`.

Three layers get exercised here and each one can be wrong in its own way.

The kernel is checked against hand-computed answers rather than against the twin,
because the twin is what `tests/fuzz` uses and a test that repeats the fuzz
comparison on eight rows adds nothing. What these check instead is the null
policy, which is the part of a grouped reduction that has an actual decision in
it: a group whose values are all null gives zero from `sum` and `count`, null
from `mean`, `min`, `max`, `first` and `last`, and its true row count from `size`.
That is pandas' answer and it is deliberate rather than incidental, so it is
pinned down one reduction at a time.

The ordinal machinery is checked separately, because combining several keys is
where a group by silently produces wrong answers rather than crashing. Two keys
whose ordinals collide under a bad packing would merge two groups into one and
the result would still look like a plausible frame. The tests here group on
column pairs chosen so that a merge changes the row count.

The frame layer is checked for the things that are its own: that the key columns
come back holding the right values, that `dropna` and `sort` do what pandas does,
and that an output name collision is refused rather than silently taking the last
one.
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
from firepanda.frame.frame import DataFrame
from firepanda.frame.groupby import AggSpec
from firepanda.frame.series import Series
from firepanda.hash.grouping import group_ordinals
from firepanda.kernel.group import (
    AggKind,
    aggregate_group,
    aggregate_group_any,
    group_count,
    group_first,
    group_last,
    group_max,
    group_mean,
    group_min,
    group_size,
    group_sum,
)


def codes_of(values: List[Scalar[DType.uint32]]) -> Array[DType.uint32]:
    """Builds a code column."""
    return from_list(values)


def ints(values: List[Scalar[DType.int64]]) -> Array[DType.int64]:
    """Builds an int64 column."""
    return from_list(values)


def sample_codes() -> Array[DType.uint32]:
    """Six rows across three groups: 0 twice, 1 three times, 2 once."""
    return codes_of([0, 1, 0, 1, 2, 1])


def sample_values() raises -> Array[DType.int64]:
    """The values those six rows carry, with row 3 null."""
    var col = ints([10, 20, 30, 40, 50, 60])
    col.set_null(3)
    return col^


def all_null_group() raises -> Array[DType.int64]:
    """Four rows in two groups where group 1 is entirely null."""
    var col = ints([7, 0, 9, 0])
    col.set_null(1)
    col.set_null(3)
    return col^


def sample_frame() raises -> DataFrame:
    """Seven rows keyed by `k`, with a null key on row 4 and a null value on row 3.

    Three real keys and a null one, so `dropna` changes the row count rather than
    just the contents. The keys are neither sorted nor reversed, so a test that
    asserts on the output order is asserting something. Row 4 is the null key and
    it carries a value of 50, which is what makes dropping it observable in the
    sums as well as in the length.
    """
    var keys = ints([20, 10, 20, 10, 99, 10, 5])
    keys.set_null(4)
    var values = ints([10, 20, 30, 40, 50, 60, 70])
    values.set_null(3)
    var series = List[Series]()
    series.append(Series("k", keys^))
    series.append(Series("v", values^))
    return DataFrame.from_series(series^)


def test_size_counts_every_row_including_nulls() raises:
    var out = group_size(sample_codes(), 3)
    assert_equal(len(out), 3)
    assert_equal(out[0], 2)
    assert_equal(out[1], 3)
    assert_equal(out[2], 1)


def test_count_skips_nulls() raises:
    var out = group_count(sample_values(), sample_codes(), 3)
    assert_equal(out[0], 2)
    assert_equal(out[1], 2, "row 3 is null so group 1 counts two of three")
    assert_equal(out[2], 1)


def test_sum_treats_a_null_as_zero() raises:
    var out = group_sum(sample_values(), sample_codes(), 3)
    assert_equal(out[0], 40)
    assert_equal(out[1], 80, "20 + 60, with the null contributing nothing")
    assert_equal(out[2], 50)


def test_sum_of_an_all_null_group_is_zero_and_present() raises:
    var out = group_sum(all_null_group(), codes_of([0, 1, 0, 1]), 2)
    assert_equal(out[1], 0)
    assert_true(out.is_valid(1), "pandas gives 0 here, not NA")


def test_count_of_an_all_null_group_is_zero_and_present() raises:
    var out = group_count(all_null_group(), codes_of([0, 1, 0, 1]), 2)
    assert_equal(out[1], 0)
    assert_true(out.is_valid(1))


def test_mean_divides_by_the_present_count() raises:
    var out = group_mean(sample_values(), sample_codes(), 3)
    assert_almost_equal(out[0], 20.0)
    assert_almost_equal(out[1], 40.0, msg="80 over 2, not 80 over 3")
    assert_almost_equal(out[2], 50.0)


def test_mean_of_an_all_null_group_is_null() raises:
    var out = group_mean(all_null_group(), codes_of([0, 1, 0, 1]), 2)
    assert_true(out.is_valid(0))
    assert_false(out.is_valid(1))


def test_min_and_max_ignore_nulls() raises:
    var low = group_min(sample_values(), sample_codes(), 3)
    var high = group_max(sample_values(), sample_codes(), 3)
    assert_equal(low[0], 10)
    assert_equal(high[0], 30)
    assert_equal(low[1], 20)
    assert_equal(high[1], 60)
    assert_equal(low[2], 50)
    assert_equal(high[2], 50)


def test_min_and_max_of_an_all_null_group_are_null() raises:
    var codes = codes_of([0, 1, 0, 1])
    var low = group_min(all_null_group(), codes, 2)
    var high = group_max(all_null_group(), codes, 2)
    assert_false(low.is_valid(1))
    assert_false(high.is_valid(1))
    assert_equal(low[1], 0, "a null holds a zero, not the seed identity")
    assert_equal(high[1], 0)


def test_first_and_last_skip_nulls() raises:
    var low = group_first(sample_values(), sample_codes(), 3)
    var high = group_last(sample_values(), sample_codes(), 3)
    assert_equal(low[1], 20)
    assert_equal(high[1], 60, "row 3 is null so the last present value is 60")


def test_first_and_last_of_an_all_null_group_are_null() raises:
    var codes = codes_of([0, 1, 0, 1])
    var low = group_first(all_null_group(), codes, 2)
    var high = group_last(all_null_group(), codes, 2)
    assert_false(low.is_valid(1))
    assert_false(high.is_valid(1))


def test_an_empty_column_gives_no_groups() raises:
    var out = group_sum(ints([]), codes_of([]), 0)
    assert_equal(len(out), 0)


def test_a_group_with_no_rows_still_appears() raises:
    # Groups come from the caller's count, not from the codes, so asking for
    # four when only two were used has to produce four rows.
    var out = group_sum(ints([1, 2]), codes_of([0, 3]), 4)
    assert_equal(len(out), 4)
    assert_equal(out[1], 0)
    assert_equal(out[3], 2)


def test_the_erased_entry_point_matches_the_typed_one() raises:
    var typed = aggregate_group(sample_values(), AggKind.SUM, sample_codes(), 3)
    var erased = aggregate_group_any(
        AnyArray(sample_values()), AggKind.SUM, sample_codes(), 3
    )
    assert_equal(typed.dtype(), erased.dtype())
    for g in range(3):
        assert_equal(
            typed.as_typed[DType.int64]()[g], erased.as_typed[DType.int64]()[g]
        )


def test_a_code_out_of_range_is_refused() raises:
    with assert_raises():
        _ = aggregate_group_any(
            AnyArray(ints([1, 2])), AggKind.SUM, codes_of([0, 5]), 2
        )


def test_a_length_mismatch_is_refused() raises:
    with assert_raises():
        _ = aggregate_group_any(
            AnyArray(ints([1, 2, 3])), AggKind.SUM, codes_of([0, 1]), 2
        )


def test_the_reduction_names_itself() raises:
    assert_equal(String(AggKind.SUM), "sum")
    assert_equal(String(AggKind.MEAN), "mean")
    assert_equal(String(AggKind.SIZE), "size")


def test_ordinals_on_one_key_are_dense() raises:
    var frame = sample_frame()
    var at = List[Int]()
    at.append(0)
    var grouping = group_ordinals(frame.columns, at, frame.rows)
    assert_equal(grouping.groups, 4, "20, 10, 5 and the null")
    assert_equal(len(grouping.rows_at), 4)
    for i in range(frame.rows):
        assert_true(Int(grouping.codes[i]) < grouping.groups)


def test_ordinals_agree_on_equal_keys() raises:
    var frame = sample_frame()
    var at = List[Int]()
    at.append(0)
    var grouping = group_ordinals(frame.columns, at, frame.rows)
    assert_equal(grouping.codes[0], grouping.codes[2], "both are 20")
    assert_equal(grouping.codes[1], grouping.codes[3])
    assert_equal(grouping.codes[3], grouping.codes[5])
    assert_equal(grouping.codes[0], 0, "first seen is ordinal zero")
    assert_equal(grouping.codes[1], 1)


def test_two_keys_do_not_merge_groups() raises:
    # Both columns have two distinct values and all four combinations appear, so
    # a packing that lost information would give fewer than four groups.
    var left = ints([0, 0, 1, 1])
    var right = ints([0, 1, 0, 1])
    var series = List[Series]()
    series.append(Series("a", left^))
    series.append(Series("b", right^))
    var frame = DataFrame.from_series(series^)
    var at = List[Int]()
    at.append(0)
    at.append(1)
    var grouping = group_ordinals(frame.columns, at, frame.rows)
    assert_equal(grouping.groups, 4)


def test_two_keys_collapse_when_the_pairs_repeat() raises:
    var left = ints([5, 5, 5, 5])
    var right = ints([1, 2, 1, 2])
    var series = List[Series]()
    series.append(Series("a", left^))
    series.append(Series("b", right^))
    var frame = DataFrame.from_series(series^)
    var at = List[Int]()
    at.append(0)
    at.append(1)
    var grouping = group_ordinals(frame.columns, at, frame.rows)
    assert_equal(grouping.groups, 2)
    assert_equal(grouping.codes[0], grouping.codes[2])
    assert_equal(grouping.codes[1], grouping.codes[3])


def test_no_keys_is_refused() raises:
    var frame = sample_frame()
    with assert_raises():
        _ = group_ordinals(frame.columns, List[Int](), frame.rows)


def test_group_by_produces_one_row_per_group() raises:
    var frame = sample_frame()
    var by = List[String]()
    by.append("k")
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))
    var out = frame.group_by(by, specs)
    assert_equal(len(out), 3, "the null key group is dropped by default")
    assert_equal(frame.rows, 7, "the input is untouched")
    assert_equal(out.width(), 2)
    assert_equal(out.names()[0], "k")
    assert_equal(out.names()[1], "v_sum")


def test_group_by_sorts_by_the_key() raises:
    var frame = sample_frame()
    var by = List[String]()
    by.append("k")
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))
    var out = frame.group_by(by, specs)
    var keys = out.column("k").as_typed[DType.int64]()
    assert_equal(keys[0], 5)
    assert_equal(keys[1], 10)
    assert_equal(keys[2], 20)
    var sums = out.column("v_sum").as_typed[DType.int64]()
    assert_equal(sums[0], 70)
    assert_equal(sums[1], 80, "20 + 60, with row 3 null")
    assert_equal(sums[2], 40)


def test_group_by_keeps_null_keys_when_asked() raises:
    var frame = sample_frame()
    var by = List[String]()
    by.append("k")
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))
    var out = frame.group_by(by, specs, dropna=False)
    assert_equal(len(out), 4)
    assert_equal(out.column("v_sum").as_typed[DType.int64]()[3], 50)
    var keys = out.column("k")
    var nulls = 0
    for i in range(len(out)):
        if not keys.is_valid(i):
            nulls += 1
    assert_equal(nulls, 1)


def test_group_by_unsorted_keeps_first_seen_order() raises:
    var frame = sample_frame()
    var by = List[String]()
    by.append("k")
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))
    var out = frame.group_by(by, specs, dropna=True, sort=False)
    var keys = out.column("k").as_typed[DType.int64]()
    assert_equal(keys[0], 20, "20 is on row 0")
    assert_equal(keys[1], 10)
    assert_equal(
        keys[2], 5, "the null key group sat between them and was dropped"
    )


def test_group_by_with_no_specs_gives_the_distinct_keys() raises:
    var frame = sample_frame()
    var by = List[String]()
    by.append("k")
    var out = frame.group_by(by, List[AggSpec]())
    assert_equal(out.width(), 1)
    assert_equal(len(out), 3)


def test_several_reductions_of_one_column() raises:
    var frame = sample_frame()
    var by = List[String]()
    by.append("k")
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.MIN))
    specs.append(AggSpec("v", AggKind.MAX))
    specs.append(AggSpec("v", AggKind.COUNT))
    var out = frame.group_by(by, specs)
    assert_equal(out.width(), 4)
    assert_equal(out.names()[1], "v_min")
    assert_equal(out.names()[2], "v_max")
    assert_equal(out.column("v_min").as_typed[DType.int64]()[1], 20)
    assert_equal(out.column("v_max").as_typed[DType.int64]()[1], 60)
    assert_equal(out.column("v_count").as_typed[DType.int64]()[1], 2)


def test_an_explicit_output_name_wins() raises:
    var frame = sample_frame()
    var by = List[String]()
    by.append("k")
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM, "total"))
    var out = frame.group_by(by, specs)
    assert_equal(out.names()[1], "total")


def test_colliding_output_names_are_refused() raises:
    var frame = sample_frame()
    var by = List[String]()
    by.append("k")
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM, "x"))
    specs.append(AggSpec("v", AggKind.MEAN, "x"))
    with assert_raises():
        _ = frame.group_by(by, specs)


def test_an_output_colliding_with_a_key_is_refused() raises:
    var frame = sample_frame()
    var by = List[String]()
    by.append("k")
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM, "k"))
    with assert_raises():
        _ = frame.group_by(by, specs)


def test_a_repeated_key_is_refused() raises:
    var frame = sample_frame()
    var by = List[String]()
    by.append("k")
    by.append("k")
    with assert_raises():
        _ = frame.group_by(by, List[AggSpec]())


def test_an_unknown_column_is_refused() raises:
    var frame = sample_frame()
    var by = List[String]()
    by.append("nope")
    with assert_raises():
        _ = frame.group_by(by, List[AggSpec]())


def test_group_agg_reduces_every_other_column() raises:
    var frame = sample_frame()
    var by = List[String]()
    by.append("k")
    var out = frame.group_agg(by, AggKind.SUM)
    assert_equal(out.width(), 2)
    assert_equal(out.names()[1], "v", "one reduction means no suffix")
    assert_equal(out.column("v").as_typed[DType.int64]()[1], 80)
    assert_equal(len(out), 3)


def test_group_count_counts_rows_not_values() raises:
    var frame = sample_frame()
    var by = List[String]()
    by.append("k")
    var out = frame.group_count(by)
    assert_equal(out.names()[1], "size")
    var sizes = out.column("size").as_typed[DType.int64]()
    assert_equal(sizes[0], 1)
    assert_equal(sizes[1], 3, "three rows keyed 10, one of which has a null v")
    assert_equal(sizes[2], 2)
    assert_equal(len(out), 3)


def test_grouping_on_a_float_key() raises:
    var keys = from_list[DType.float64]([1.5, 2.5, 1.5])
    var values = ints([1, 2, 3])
    var series = List[Series]()
    series.append(Series("k", keys^))
    series.append(Series("v", values^))
    var frame = DataFrame.from_series(series^)
    var by = List[String]()
    by.append("k")
    var out = frame.group_agg(by, AggKind.SUM)
    assert_equal(len(out), 2)
    assert_equal(out.column("v").as_typed[DType.int64]()[0], 4)


def test_grouping_on_a_bool_key() raises:
    var keys = from_list[DType.bool]([True, False, True, True])
    var values = ints([1, 2, 3, 4])
    var series = List[Series]()
    series.append(Series("k", keys^))
    series.append(Series("v", values^))
    var frame = DataFrame.from_series(series^)
    var by = List[String]()
    by.append("k")
    var out = frame.group_agg(by, AggKind.SUM)
    assert_equal(len(out), 2)
    var sums = out.column("v").as_typed[DType.int64]()
    assert_equal(sums[0], 2, "False sorts first")
    assert_equal(sums[1], 8)


def test_grouping_on_two_keys_through_the_frame() raises:
    var left = ints([1, 1, 2, 2, 1])
    var right = ints([9, 8, 9, 8, 9])
    var values = ints([10, 20, 30, 40, 50])
    var series = List[Series]()
    series.append(Series("a", left^))
    series.append(Series("b", right^))
    series.append(Series("v", values^))
    var frame = DataFrame.from_series(series^)
    var by = List[String]()
    by.append("a")
    by.append("b")
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM, "total"))
    var out = frame.group_by(by, specs)
    assert_equal(len(out), 4)
    var a = out.column("a").as_typed[DType.int64]()
    var b = out.column("b").as_typed[DType.int64]()
    var total = out.column("total").as_typed[DType.int64]()
    assert_equal(a[0], 1)
    assert_equal(b[0], 8)
    assert_equal(total[0], 20)
    assert_equal(a[1], 1)
    assert_equal(b[1], 9)
    assert_equal(total[1], 60, "rows 0 and 4 both key (1, 9)")


def test_grouping_an_empty_frame() raises:
    var keys = ints([])
    var values = ints([])
    var series = List[Series]()
    series.append(Series("k", keys^))
    series.append(Series("v", values^))
    var frame = DataFrame.from_series(series^)
    var by = List[String]()
    by.append("k")
    var out = frame.group_agg(by, AggKind.SUM)
    assert_equal(len(out), 0)
    assert_equal(out.width(), 2)


def test_a_spec_renders_as_it_reads() raises:
    assert_equal(String(AggSpec("v", AggKind.SUM)), "sum(v) as v_sum")
    assert_equal(String(AggSpec("v", AggKind.MAX, "top")), "max(v) as top")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
