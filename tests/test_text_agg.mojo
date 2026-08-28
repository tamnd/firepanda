"""Tests for aggregating a text column rather than grouping by one.

Six of the thirteen reductions mean something over bytes. Two of them, `SIZE`
and `COUNT`, never look at a value and are the same loops the number path runs.
Four report a value the column held, and those are where the tests are, because
that is where the kernel has decisions to make: which row to keep while scanning,
and what to produce for a group whose every row is null.

`MIN` and `MAX` are the two worth being careful about. They order bytes, so the
keys here are built to disagree only late: "am" against "amsterdam", where one is
a prefix of the other and the shorter one wins, and a pair that agrees past the
twelve bytes an inline element holds, where the comparison has to leave the view
and read the payload. A comparison on the four byte prefix alone passes the easy
cases and fails both of these.

`NUNIQUE` takes a route of its own, through the factorize, so it is checked
against groups that repeat values across group boundaries. Two rows in different
groups holding the same bytes share an ordinal, and a count that forgot to
restrict itself to one group would notice neither.

The seven that are not defined here have to say so. A sum of names is not a slow
operation, it is not an operation, and the test that matters is that the error
names the reduction rather than being a generic refusal.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.frame.frame import DataFrame
from firepanda.frame.groupby import AggSpec
from firepanda.frame.series import Series
from firepanda.kernel.group import AggKind, aggregate_group_any
from firepanda.kernel.scalar import group_text_scalar
from firepanda.testing.rng import Rng


def text(values: List[String]) -> StringArray:
    """Builds a string column from a list."""
    return strings_from_list(values)


def with_nulls(values: List[String], present: List[Bool]) -> StringArray:
    """Builds a string column with nulls where asked.

    Args:
        values: The elements. The entry under a false flag is ignored.
        present: One flag per element.

    Returns:
        The column.
    """
    var builder = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        if present[i]:
            builder.append(values[i].as_bytes())
        else:
            builder.append_null()
    return builder^.finish()


def group_codes(values: List[Int]) -> Array[DType.uint32]:
    """Builds a codes column from a list of ordinals."""
    var out = Array[DType.uint32](len(values))
    var at = out.unsafe_ptr()
    for i in range(len(values)):
        at.unsafe_offset(i).unsafe_store(UInt32(values[i]))
    return out^


def reduce(
    var col: StringArray, kind: AggKind, codes: List[Int], groups: Int
) raises -> AnyArray:
    """Runs one grouped reduction over a text column."""
    return aggregate_group_any(AnyArray(col^), kind, group_codes(codes), groups)


def value_of(col: AnyArray, i: Int) raises -> String:
    """Returns one element of a text column as a string."""
    return col.strings()[i]


def test_count_ignores_the_nulls() raises:
    var col = with_nulls(["a", "b", "c", "d"], [True, False, True, False])
    var out = reduce(col^, AggKind.COUNT, [0, 0, 1, 1], 2)
    var counts = out.as_typed[DType.int64]()
    assert_equal(counts[0], 1)
    assert_equal(counts[1], 1)


def test_size_counts_the_nulls() raises:
    var col = with_nulls(["a", "b", "c"], [True, False, False])
    var out = reduce(col^, AggKind.SIZE, [0, 0, 1], 2)
    var sizes = out.as_typed[DType.int64]()
    assert_equal(sizes[0], 2)
    assert_equal(sizes[1], 1)


def test_first_and_last_take_the_edges_in_row_order() raises:
    var col = text(["oslo", "lima", "cairo", "quito"])
    var first = reduce(StringArray(copy=col), AggKind.FIRST, [0, 1, 0, 1], 2)
    var last = reduce(col^, AggKind.LAST, [0, 1, 0, 1], 2)
    assert_equal(value_of(first, 0), "oslo")
    assert_equal(value_of(first, 1), "lima")
    assert_equal(value_of(last, 0), "cairo")
    assert_equal(value_of(last, 1), "quito")


def test_first_and_last_skip_the_nulls() raises:
    var col = with_nulls(["x", "oslo", "lima", "y"], [False, True, True, False])
    var first = reduce(StringArray(copy=col), AggKind.FIRST, [0, 0, 0, 0], 1)
    var last = reduce(col^, AggKind.LAST, [0, 0, 0, 0], 1)
    assert_equal(value_of(first, 0), "oslo")
    assert_equal(value_of(last, 0), "lima")


def test_min_and_max_order_bytes() raises:
    var col = text(["oslo", "cairo", "quito"])
    var low = reduce(StringArray(copy=col), AggKind.MIN, [0, 0, 0], 1)
    var high = reduce(col^, AggKind.MAX, [0, 0, 0], 1)
    assert_equal(value_of(low, 0), "cairo")
    assert_equal(value_of(high, 0), "quito")


def test_a_prefix_sorts_before_what_extends_it() raises:
    """The shorter element wins on a tie up to its length, and a comparison that
    stopped at the view's four bytes would call these two equal."""
    var col = text(["amsterdam", "am"])
    var low = reduce(StringArray(copy=col), AggKind.MIN, [0, 0], 1)
    var high = reduce(col^, AggKind.MAX, [0, 0], 1)
    assert_equal(value_of(low, 0), "am")
    assert_equal(value_of(high, 0), "amsterdam")


def test_min_reads_past_the_inline_limit() raises:
    """Twelve bytes is everything an inline element holds, so the difference here
    is only in the payload."""
    var col = text(["shared_prefix_right", "shared_prefix_left"])
    var low = reduce(col^, AggKind.MIN, [0, 0], 1)
    assert_equal(value_of(low, 0), "shared_prefix_left")


def test_a_group_with_nothing_in_it_is_null() raises:
    var col = with_nulls(["a", "b"], [True, False])
    var out = reduce(col^, AggKind.MIN, [0, 1], 2)
    assert_true(out.is_valid(0))
    assert_false(out.is_valid(1))
    assert_equal(value_of(out, 0), "a")


def test_a_group_of_only_nulls_is_null_for_all_four() raises:
    var kinds = List[AggKind]()
    kinds.append(AggKind.FIRST)
    kinds.append(AggKind.LAST)
    kinds.append(AggKind.MIN)
    kinds.append(AggKind.MAX)
    for k in range(len(kinds)):
        var col = with_nulls(["a", "b"], [False, False])
        var out = reduce(col^, kinds[k], [0, 0], 1)
        assert_false(out.is_valid(0))


def test_nunique_counts_distinct_values_in_a_group() raises:
    var col = text(["a", "b", "a", "c", "c"])
    var out = reduce(col^, AggKind.NUNIQUE, [0, 0, 0, 1, 1], 2)
    var counts = out.as_typed[DType.int64]()
    assert_equal(counts[0], 2)
    assert_equal(counts[1], 1)


def test_nunique_does_not_leak_across_groups() raises:
    """Both groups hold "a", which shares one ordinal, so a count that worked on
    ordinals without restricting itself to a group would see one value twice."""
    var col = text(["a", "b", "a", "c"])
    var out = reduce(col^, AggKind.NUNIQUE, [0, 0, 1, 1], 2)
    var counts = out.as_typed[DType.int64]()
    assert_equal(counts[0], 2)
    assert_equal(counts[1], 2)


def test_nunique_ignores_the_nulls() raises:
    var col = with_nulls(["a", "x", "a"], [True, False, True])
    var out = reduce(col^, AggKind.NUNIQUE, [0, 0, 0], 1)
    var counts = out.as_typed[DType.int64]()
    assert_equal(counts[0], 1)


def test_an_empty_column_produces_no_groups() raises:
    var out = reduce(text(List[String]()), AggKind.MIN, List[Int](), 0)
    assert_equal(len(out), 0)


def test_summing_text_names_the_reduction() raises:
    var col = text(["a", "b"])
    with assert_raises(contains="sum is not defined for a string column"):
        _ = reduce(col^, AggKind.SUM, [0, 0], 1)


def test_the_other_numeric_reductions_are_refused_too() raises:
    var kinds = List[AggKind]()
    kinds.append(AggKind.MEAN)
    kinds.append(AggKind.VAR)
    kinds.append(AggKind.STD)
    kinds.append(AggKind.MEDIAN)
    kinds.append(AggKind.quantile_at(0.9))
    for k in range(len(kinds)):
        var col = text(["a", "b"])
        with assert_raises(contains="not defined for a string column"):
            _ = reduce(col^, kinds[k], [0, 0], 1)


def test_the_kernel_agrees_with_the_twin() raises:
    """Two hundred rows over ten groups, a quarter of them null, and values drawn
    from a three letter alphabet so that ties and shared prefixes are common.

    The twin is O(groups times rows) and holds a `String` per comparison, so the
    row count is what keeps this test in the tenths of a second rather than the
    tens of seconds. The shapes that matter here are ties, prefixes and empty
    groups, and none of them need a long column to happen.
    """
    var rng = Rng(0x51DE2026)
    var builder = StringBuilder(capacity=200)
    var codes = List[Int](capacity=200)
    for _ in range(200):
        codes.append(Int(rng.next_below(10)))
        if rng.next_below(4) == 0:
            builder.append_null()
            continue
        var value = String("")
        for _ in range(Int(rng.next_range(1, 15))):
            value += chr(Int(rng.next_range(97, 100)))
        builder.append(value.as_bytes())
    var col = builder^.finish()

    var kinds = List[AggKind]()
    kinds.append(AggKind.FIRST)
    kinds.append(AggKind.LAST)
    kinds.append(AggKind.MIN)
    kinds.append(AggKind.MAX)
    for k in range(len(kinds)):
        var want = group_text_scalar(
            StringArray(copy=col), kinds[k], group_codes(codes), 10
        )
        var values = want[0].copy()
        var valid = want[1].copy()
        var got = reduce(StringArray(copy=col), kinds[k], codes, 10)
        for g in range(10):
            var note = String(kinds[k]) + " group " + String(g)
            assert_true(got.is_valid(g) == valid[g], note)
            if valid[g]:
                assert_equal(value_of(got, g), values[g], note)


def test_a_frame_aggregates_a_text_column() raises:
    var series = List[Series]()
    series.append(Series("region", text(["west", "east", "west", "east"])))
    series.append(Series("city", text(["oslo", "lima", "cairo", "quito"])))
    var frame = DataFrame.from_series(series^)

    var specs = List[AggSpec]()
    specs.append(AggSpec("city", AggKind.MIN))
    specs.append(AggSpec("city", AggKind.MAX))
    specs.append(AggSpec("city", AggKind.COUNT))
    specs.append(AggSpec("city", AggKind.NUNIQUE))
    var out = frame.group_by(["region"], specs)

    assert_equal(len(out), 2)
    assert_equal(out.column("region").text(0), "east")
    assert_equal(out.column("city_min").text(0), "lima")
    assert_equal(out.column("city_max").text(0), "quito")
    assert_equal(out.column("city_min").text(1), "cairo")
    assert_equal(out.column("city_max").text(1), "oslo")
    var counts = out.column("city_count").as_typed[DType.int64]()
    assert_equal(counts[0], 2)
    var distinct = out.column("city_nunique").as_typed[DType.int64]()
    assert_equal(distinct[0], 2)


def test_a_frame_aggregates_text_beside_a_number() raises:
    var series = List[Series]()
    series.append(Series("k", text(["a", "b", "a"])))
    series.append(Series("city", text(["oslo", "lima", "cairo"])))
    var values = Array[DType.int64](3)
    var at = values.unsafe_ptr()
    at.unsafe_offset(0).unsafe_store(1)
    at.unsafe_offset(1).unsafe_store(2)
    at.unsafe_offset(2).unsafe_store(3)
    series.append(Series("v", values^))
    var frame = DataFrame.from_series(series^)

    var specs = List[AggSpec]()
    specs.append(AggSpec("city", AggKind.FIRST))
    specs.append(AggSpec("v", AggKind.SUM))
    var out = frame.group_by(["k"], specs)

    assert_equal(out.column("city_first").text(0), "oslo")
    var sums = out.column("v_sum").as_typed[DType.int64]()
    assert_equal(sums[0], 4)
    assert_equal(sums[1], 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
