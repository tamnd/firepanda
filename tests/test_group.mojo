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
from firepanda.hash.factorize import factorize
from firepanda.hash.grouping import group_ordinals
from firepanda.testing.rng import Rng
from firepanda.kernel.group import (
    AggKind,
    aggregate_group,
    aggregate_group_any,
    group_count,
    group_first,
    group_last,
    group_max,
    group_mean,
    group_median,
    group_min,
    group_nunique,
    group_quantile,
    group_size,
    group_std,
    group_sum,
    group_var,
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


def test_no_shape_of_column_factorizes_to_a_sparse_ordinal() raises:
    """Every ordinal below the count has to belong to a row, on both routes.

    `group_ordinals` skips a pass over every row for a key that is not the first
    one, and what makes that safe is the group count being exact. It reads the
    count straight off `factorize`, so a route that handed out an ordinal nothing
    carries would not produce a crash, it would produce an aggregation row with
    no key and everything after it shifted by one.

    Neither route does. The hashed one makes an ordinal only when a row asks for
    it, and the direct one indexes its table by value but appends to `keys` on
    first sight, so the gaps in the table are not gaps in the ordinals. That is a
    property of two implementations rather than a promise either of them states,
    which is why it is swept rather than spot checked: the row counts and value
    bounds here put roughly fifty columns down each route, including columns
    that are entirely null and columns of one row.
    """
    var rng = Rng(UInt64(0x0DE5))
    for _ in range(120):
        var n = 1 + rng.next_below(200)
        var bound = 1 + rng.next_below(500)
        var col = Array[DType.int64](n)
        for i in range(n):
            col[i] = Int64(rng.next_below(bound)) - 100
        for _ in range(rng.next_below(4)):
            col.set_null(rng.next_below(n))

        var found = factorize(col)
        var carried = List[Bool]()
        for _ in range(found.count()):
            carried.append(False)
        for i in range(n):
            carried[Int(found.codes[i])] = True
        for g in range(found.count()):
            assert_true(carried[g], "ordinal with no row behind it")


def test_a_numeric_key_with_nulls_is_still_renumbered() raises:
    """A null takes ordinal zero, so its representative rows are one short.

    `sample_frame`'s key has a null on row 4. The factorize reports four groups
    and hands back three representative rows, and `knows_rows` refuses that, so
    the pass runs and every group ends up with a row that carries its ordinal.
    """
    var frame = sample_frame()
    var at = List[Int]()
    at.append(0)
    var grouping = group_ordinals(frame.columns, at, frame.rows)
    assert_equal(grouping.groups, 4)
    assert_equal(len(grouping.rows_at), 4)
    for g in range(grouping.groups):
        assert_equal(Int(grouping.codes[grouping.rows_at[g]]), g)


def test_a_numeric_key_without_nulls_keeps_the_factorize_ordinals() raises:
    """With no null there is nothing to move, so the pass is skipped.

    Both numeric routes hand out ordinals in first-appearance order and know the
    row that introduced each one, so what the pass would produce is what they
    already have. This asserts the result rather than the skipping, because the
    skipping is only worth anything if the two agree: the ordinals are
    first-appearance and `rows_at` names the first row of each group.

    The span here is 15 against 7 rows, which is over `_direct_plan`'s ceiling,
    so this is the hashed route. The direct one is covered by the same
    assertions on a narrow column in `test_a_direct_route_key_keeps_them_too`.
    """
    var keys = ints([20, 10, 20, 10, 25, 10, 15])
    var values = ints([10, 20, 30, 40, 50, 60, 70])
    var series = List[Series]()
    series.append(Series("k", keys^))
    series.append(Series("v", values^))
    var frame = DataFrame.from_series(series^)

    var at = List[Int]()
    at.append(0)
    var grouping = group_ordinals(frame.columns, at, frame.rows)
    assert_equal(grouping.groups, 4)
    var want: List[Int] = [0, 1, 0, 1, 2, 1, 3]
    for i in range(len(want)):
        assert_equal(Int(grouping.codes[i]), want[i])
    assert_equal(len(grouping.rows_at), 4)
    var rows: List[Int] = [0, 1, 4, 6]
    for g in range(len(rows)):
        assert_equal(grouping.rows_at[g], rows[g])


def test_a_direct_route_key_keeps_them_too() raises:
    """The same, on the route that indexes a table by the value itself.

    Values 0 to 3 over four rows put the span inside `_direct_plan`'s ceiling,
    so no hashing happens. That route appends to `keys` on first sight, which is
    where its representative row comes from, and it is a different loop from the
    hashed one's so it gets its own assertion.
    """
    var keys = ints([3, 1, 3, 0])
    var values = ints([10, 20, 30, 40])
    var series = List[Series]()
    series.append(Series("k", keys^))
    series.append(Series("v", values^))
    var frame = DataFrame.from_series(series^)

    var at = List[Int]()
    at.append(0)
    var grouping = group_ordinals(frame.columns, at, frame.rows)
    assert_equal(grouping.groups, 3)
    var want: List[Int] = [0, 1, 0, 2]
    for i in range(len(want)):
        assert_equal(Int(grouping.codes[i]), want[i])
    var rows: List[Int] = [0, 1, 3]
    for g in range(len(rows)):
        assert_equal(grouping.rows_at[g], rows[g])


def test_a_null_key_lands_where_its_first_null_is() raises:
    """The renumbering moves it there, and the factorize puts it at zero.

    Row 4 is the null and rows 0 to 3 carry two keys between them, so the null
    group is the third one seen. A group by that skipped the renumbering would
    report it first.
    """
    var frame = sample_frame()
    var at = List[Int]()
    at.append(0)
    var grouping = group_ordinals(frame.columns, at, frame.rows)
    assert_equal(Int(grouping.codes[4]), 2)
    assert_equal(grouping.rows_at[2], 4)


def test_the_packed_column_keeps_the_factorize_ordinals() raises:
    """The combine step skips the pass too, and has to agree with it.

    `packed` is written a row at a time from two code arrays, so it has no nulls
    and its factorize hands out first-appearance ordinals with a representative
    row for each. That is a pass per key after the first. The pairs here repeat
    out of order and one of them appears only at the end, so an ordinal assigned
    by anything other than first appearance shows up in `want`.
    """
    var left = ints([7, 8, 7, 8, 7, 9])
    var right = ints([1, 1, 2, 1, 1, 3])
    var series = List[Series]()
    series.append(Series("a", left^))
    series.append(Series("b", right^))
    var frame = DataFrame.from_series(series^)
    var at = List[Int]()
    at.append(0)
    at.append(1)
    var grouping = group_ordinals(frame.columns, at, frame.rows)
    assert_equal(grouping.groups, 4)
    var want: List[Int] = [0, 1, 2, 1, 0, 3]
    for i in range(len(want)):
        assert_equal(Int(grouping.codes[i]), want[i])
    var rows: List[Int] = [0, 1, 2, 5]
    for g in range(len(rows)):
        assert_equal(grouping.rows_at[g], rows[g])


def test_a_null_in_the_first_of_two_keys_still_lands_in_place() raises:
    """The packed column has no nulls even when the key it came from does.

    A null key becomes ordinal zero in the first factorize and the renumbering
    moves it to where its first null is. What is packed after that is a code, not
    a value, so the combine step sees a column with no nulls and the group the
    null belongs to still has to come out third here.
    """
    var left = ints([20, 10, 20, 99, 10])
    left.set_null(3)
    var right = ints([1, 1, 1, 1, 1])
    var series = List[Series]()
    series.append(Series("a", left^))
    series.append(Series("b", right^))
    var frame = DataFrame.from_series(series^)
    var at = List[Int]()
    at.append(0)
    at.append(1)
    var grouping = group_ordinals(frame.columns, at, frame.rows)
    assert_equal(grouping.groups, 3)
    var want: List[Int] = [0, 1, 0, 2, 1]
    for i in range(len(want)):
        assert_equal(Int(grouping.codes[i]), want[i])
    assert_equal(grouping.rows_at[2], 3)
    assert_false(frame.columns[0].is_valid(grouping.rows_at[2]))


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


def test_enough_keys_to_overflow_the_packed_space_still_groups() raises:
    """Seven keys of a thousand values each, which is where `_condense` runs.

    The running space is the product of the group counts and it is not condensed
    until it would leave an int64, so this is the only shape in the suite that
    reaches that. A thousand to the sixth is inside the range and a thousand to
    the seventh is not, so the seventh key is the one that triggers the pass, and
    what comes out afterwards has to be the same grouping a smaller key list
    would have produced.

    Every key here is the same column, so there are exactly a thousand tuples and
    each row is in the group its own value names, which makes the answer easy to
    state and impossible to get right by accident on a packing that overflowed.
    """
    var rows = 1000
    var series = List[Series]()
    for k in range(7):
        var col = Array[DType.int64](rows)
        for i in range(rows):
            col[i] = Int64(i)
        series.append(Series(String("k", k), col^))
    var frame = DataFrame.from_series(series^)
    var at = List[Int]()
    for k in range(7):
        at.append(k)

    var grouping = group_ordinals(frame.columns, at, frame.rows)
    assert_equal(grouping.groups, rows)
    var bad = -1
    for i in range(rows):
        if Int(grouping.codes[i]) != i or grouping.rows_at[i] != i:
            bad = i
            break
    assert_equal(bad, -1, String("wrong group at row ", bad))


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


def test_variance_uses_a_sample_divisor_by_default() raises:
    var out = group_var(sample_values(), sample_codes(), 3)
    assert_almost_equal(out[0], 200.0, atol=1e-9)
    assert_almost_equal(out[1], 800.0, atol=1e-9)


def test_variance_of_a_single_value_is_null_at_the_default_ddof() raises:
    var out = group_var(sample_values(), sample_codes(), 3)
    assert_false(
        out.is_valid(2),
        "one value and ddof of 1 divides by zero; pandas reports NaN here",
    )


def test_variance_of_a_single_value_is_zero_at_ddof_zero() raises:
    var out = group_var(sample_values(), sample_codes(), 3, ddof=0)
    assert_true(out.is_valid(2))
    assert_almost_equal(out[2], 0.0, atol=1e-9)
    assert_almost_equal(out[0], 100.0, atol=1e-9)


def test_variance_skips_nulls_rather_than_counting_them_as_zero() raises:
    # Group 1 holds 20, null and 60. Counting the null as a zero would give a
    # mean of 26.67 and a variance of 933.33 instead of 800.
    var out = group_var(sample_values(), sample_codes(), 3)
    assert_almost_equal(out[1], 800.0, atol=1e-9)


def test_variance_of_an_all_null_group_is_null() raises:
    var out = group_var(all_null_group(), codes_of([0, 1, 0, 1]), 2)
    assert_false(out.is_valid(1))


def test_standard_deviation_is_the_root_of_the_variance() raises:
    var out = group_std(sample_values(), sample_codes(), 3)
    assert_almost_equal(out[0], 14.142135623730951, atol=1e-9)
    assert_almost_equal(out[1], 28.284271247461902, atol=1e-9)
    assert_false(out.is_valid(2))


def test_variance_keeps_its_digits_on_large_values() raises:
    # Five timestamps a second apart. The one pass formula computes this as the
    # difference of two numbers near 1.4e19 and comes back with garbage, or with
    # a negative variance. The right answer is 2.5.
    var base = Int64(1_700_000_000)
    var col = ints([base, base + 1, base + 2, base + 3, base + 4])
    var out = group_var(col, codes_of([0, 0, 0, 0, 0]), 1)
    assert_almost_equal(out[0], 2.5, atol=1e-9)


def test_median_interpolates_between_two_middle_values() raises:
    var out = group_median(sample_values(), sample_codes(), 3)
    assert_almost_equal(out[0], 20.0, atol=1e-9)
    assert_almost_equal(out[1], 40.0, atol=1e-9)
    assert_almost_equal(out[2], 50.0, atol=1e-9)


def test_median_of_an_even_count_of_integers_can_be_a_half() raises:
    var out = group_median(ints([1, 2, 3, 4]), codes_of([0, 0, 0, 0]), 1)
    assert_almost_equal(out[0], 2.5, atol=1e-9)


def test_median_of_an_all_null_group_is_null() raises:
    var out = group_median(all_null_group(), codes_of([0, 1, 0, 1]), 2)
    assert_false(out.is_valid(1))
    assert_almost_equal(out[0], 8.0, atol=1e-9)


def test_median_does_not_need_the_input_sorted() raises:
    var out = group_median(
        ints([50, 10, 40, 20, 30]), codes_of([0, 0, 0, 0, 0]), 1
    )
    assert_almost_equal(out[0], 30.0, atol=1e-9)


def test_quantile_at_zero_and_one_are_the_extremes() raises:
    var low = group_quantile(sample_values(), sample_codes(), 3, 0.0)
    var high = group_quantile(sample_values(), sample_codes(), 3, 1.0)
    assert_almost_equal(low[1], 20.0, atol=1e-9)
    assert_almost_equal(high[1], 60.0, atol=1e-9)


def test_quantile_interpolates_linearly() raises:
    # Group 0 holds 10 and 30, so the quarter point is a quarter of the way from
    # one to the other rather than either of them.
    var out = group_quantile(sample_values(), sample_codes(), 3, 0.25)
    assert_almost_equal(out[0], 15.0, atol=1e-9)
    assert_almost_equal(out[1], 30.0, atol=1e-9)


def test_quantile_outside_zero_to_one_is_refused() raises:
    with assert_raises(contains="between 0 and 1"):
        _ = group_quantile(sample_values(), sample_codes(), 3, 1.5)


def test_distinct_count_ignores_repeats_and_nulls() raises:
    var col = ints([5, 5, 7, 0, 5])
    col.set_null(3)
    var out = group_nunique(col, codes_of([0, 0, 0, 0, 0]), 1)
    assert_equal(out[0], 2)


def test_distinct_count_of_an_all_null_group_is_zero_and_present() raises:
    var out = group_nunique(all_null_group(), codes_of([0, 1, 0, 1]), 2)
    assert_equal(out[1], 0)
    assert_true(out.is_valid(1), "pandas gives 0 here, not NA")


def test_distinct_count_is_per_group_rather_than_overall() raises:
    # Both groups hold the same two values. Counting distinct values across the
    # column would give 2 once; per group it is 2 twice.
    var out = group_nunique(ints([1, 2, 1, 2]), codes_of([0, 0, 1, 1]), 2)
    assert_equal(out[0], 2)
    assert_equal(out[1], 2)


def test_the_new_reductions_agree_with_their_erased_spelling() raises:
    var kinds = List[AggKind]()
    kinds.append(AggKind.VAR)
    kinds.append(AggKind.STD)
    kinds.append(AggKind.MEDIAN)
    kinds.append(AggKind.QUANTILE)
    kinds.append(AggKind.NUNIQUE)
    for k in range(len(kinds)):
        var erased = AnyArray(sample_values())
        var through = aggregate_group_any(erased, kinds[k], sample_codes(), 3)
        var direct = aggregate_group(
            sample_values(), kinds[k], sample_codes(), 3
        )
        assert_equal(len(through), len(direct))
        assert_equal(
            String(through.dtype()),
            String(direct.dtype()),
            String("dtype under ", kinds[k]),
        )


def test_a_bare_code_carries_the_reduction_default() raises:
    # Constructing from a code alone has to give ddof 1 and quantile 0.5 rather
    # than zero, because zero is a legal value for both.
    assert_almost_equal(AggKind(UInt8(8)).param, 1.0, atol=1e-12)
    assert_almost_equal(AggKind(UInt8(9)).param, 1.0, atol=1e-12)
    assert_almost_equal(AggKind(UInt8(10)).param, 0.5, atol=1e-12)
    assert_almost_equal(AggKind(UInt8(11)).param, 0.5, atol=1e-12)
    assert_almost_equal(AggKind(UInt8(0)).param, 0.0, atol=1e-12)


def test_two_kinds_are_equal_by_reduction_and_not_by_parameter() raises:
    assert_true(
        AggKind.quantile_at(0.9) == AggKind.QUANTILE,
        "the dispatch chain compares against the bare kind",
    )
    assert_true(AggKind.var_with(0) == AggKind.VAR)
    assert_true(AggKind.std_with(0) == AggKind.STD)
    assert_true(AggKind.VAR != AggKind.STD)


def test_the_new_reductions_name_themselves() raises:
    assert_equal(String(AggKind.VAR), "var")
    assert_equal(String(AggKind.STD), "std")
    assert_equal(String(AggKind.MEDIAN), "median")
    assert_equal(String(AggKind.QUANTILE), "quantile")
    assert_equal(String(AggKind.NUNIQUE), "nunique")


def test_a_frame_groups_by_the_new_reductions() raises:
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.MEDIAN))
    specs.append(AggSpec("v", AggKind.NUNIQUE))
    specs.append(AggSpec("v", AggKind.quantile_at(0.0), "smallest"))
    var out = sample_frame().group_by(["k"], specs)

    # Keys 5, 10 and 20 survive; the null key is dropped. Key 10 holds 20, null
    # and 60, so its median is 40 and it has two distinct values.
    assert_equal(len(out), 3)
    var median = out.column("v_median").as_typed[DType.float64]()
    var distinct = out.column("v_nunique").as_typed[DType.int64]()
    var smallest = out.column("smallest").as_typed[DType.float64]()
    assert_almost_equal(median[1], 40.0, atol=1e-9)
    assert_equal(distinct[1], 2)
    assert_almost_equal(smallest[1], 20.0, atol=1e-9)


def test_a_frame_wide_variance_keeps_its_degrees_of_freedom() raises:
    var out = sample_frame().group_agg(["k"], AggKind.std_with(0))
    var spread = out.column("v").as_typed[DType.float64]()
    # Key 10 holds 20 and 60. With ddof 0 that is a population standard
    # deviation of 20, where the default would give 28.28.
    assert_almost_equal(spread[1], 20.0, atol=1e-9)


def test_two_quantiles_of_one_column_need_explicit_names() raises:
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.quantile_at(0.25)))
    specs.append(AggSpec("v", AggKind.quantile_at(0.75)))
    with assert_raises(contains="would both be called"):
        _ = sample_frame().group_by(["k"], specs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
