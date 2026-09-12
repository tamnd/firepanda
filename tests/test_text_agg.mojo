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
from firepanda.kernel.reduce import reduce_any
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
    var at = out.unsafe_mut_ptr()
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


def test_any_and_all_read_an_empty_string_as_false() raises:
    """The whole reason a truth over text is a question rather than a constant.

    A dataset that spells its missing text as an empty string gets a different
    answer here from one that spells it as a null, and both answers are right.
    That is `text_truth`'s rule and this is the same rule one group at a time.
    """
    var col = with_nulls(["oslo", "", "", "lima"], [True, True, True, True])
    var truthy = reduce(StringArray(copy=col), AggKind.ANY, [0, 0, 1, 1], 2)
    var every = reduce(col^, AggKind.ALL, [0, 0, 1, 1], 2)
    var found = truthy.as_typed[DType.bool]()
    var all_of = every.as_typed[DType.bool]()
    assert_equal(found[0], True, "group 0 holds a name")
    assert_equal(found[1], True, "group 1 holds a name too")
    assert_equal(all_of[0], False, "group 0 also holds an empty string")
    assert_equal(all_of[1], False, "and so does group 1")


def test_any_and_all_step_over_a_null_in_the_text() raises:
    var col = with_nulls(["x", "y"], [False, False])
    var truthy = reduce(StringArray(copy=col), AggKind.ANY, [0, 1], 2)
    var every = reduce(col^, AggKind.ALL, [0, 1], 2)
    assert_equal(truthy.as_typed[DType.bool]()[0], False, "an any over nothing")
    assert_equal(every.as_typed[DType.bool]()[0], True, "an all over nothing")


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
    kinds.append(AggKind.PROD)
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


def test_the_parallel_route_merges_across_workers() raises:
    """Seventy thousand rows, which is past the point where the row numbers are
    accumulated in a table per worker and merged afterwards.

    The two values that are not the filler are planted at opposite ends of the
    column, so the smallest of group three is found by the first worker and the
    largest by the last one and neither of them can be right unless the merge
    reads every table. The first and the last of that same group are the filler,
    at the first and last row the group has, which is the other half of the
    merge: an edge is not a comparison, it is whichever table saw the group
    earliest or latest.
    """
    var n = 70000
    var groups = 8
    var low = 8 * 10 + 3
    var high = 8 * 8000 + 3
    var builder = StringBuilder(capacity=n)
    var codes = List[Int](capacity=n)
    for i in range(n):
        codes.append(i % groups)
        if i == low:
            builder.append("aaa".as_bytes())
        elif i == high:
            builder.append("zzz".as_bytes())
        else:
            builder.append("mmm".as_bytes())
    var col = builder^.finish()

    var smallest = reduce(StringArray(copy=col), AggKind.MIN, codes, groups)
    var largest = reduce(StringArray(copy=col), AggKind.MAX, codes, groups)
    var earliest = reduce(StringArray(copy=col), AggKind.FIRST, codes, groups)
    var latest = reduce(StringArray(copy=col), AggKind.LAST, codes, groups)

    for g in range(groups):
        var note = String("group ") + String(g)
        assert_equal(value_of(smallest, g), "aaa" if g == 3 else "mmm", note)
        assert_equal(value_of(largest, g), "zzz" if g == 3 else "mmm", note)
        assert_equal(value_of(earliest, g), "mmm", note)
        assert_equal(value_of(latest, g), "mmm", note)


def test_the_parallel_route_skips_the_workers_that_saw_nothing() raises:
    """The same shape again with the nulls arranged so that most of the tables
    have nothing to say about most of the groups.

    Group one is null all the way down and has to come back null, which is the
    case a merge that treated an untouched slot as a value would get wrong. Group
    two holds exactly one value, in the middle of the column, so every table but
    one is empty for it. Group three is null until forty thousand rows in, which
    puts its first value several workers along and makes the answer to FIRST
    depend on skipping the tables in front of it rather than on taking the first
    one.
    """
    var n = 70000
    var groups = 4
    var solo = 34002
    var builder = StringBuilder(capacity=n)
    var codes = List[Int](capacity=n)
    for i in range(n):
        var g = i % groups
        codes.append(g)
        if g == 0:
            builder.append("filler".as_bytes())
        elif g == 1:
            builder.append_null()
        elif g == 2:
            if i == solo:
                builder.append("solo".as_bytes())
            else:
                builder.append_null()
        elif i < 40000:
            builder.append_null()
        elif i == 40003:
            builder.append("b".as_bytes())
        else:
            builder.append("c".as_bytes())
    var col = builder^.finish()

    var smallest = reduce(StringArray(copy=col), AggKind.MIN, codes, groups)
    var largest = reduce(StringArray(copy=col), AggKind.MAX, codes, groups)
    var earliest = reduce(StringArray(copy=col), AggKind.FIRST, codes, groups)
    var latest = reduce(StringArray(copy=col), AggKind.LAST, codes, groups)

    assert_false(smallest.is_valid(1))
    assert_false(largest.is_valid(1))
    assert_false(earliest.is_valid(1))
    assert_false(latest.is_valid(1))

    assert_equal(value_of(smallest, 2), "solo")
    assert_equal(value_of(largest, 2), "solo")
    assert_equal(value_of(earliest, 2), "solo")
    assert_equal(value_of(latest, 2), "solo")

    assert_equal(value_of(smallest, 3), "b")
    assert_equal(value_of(largest, 3), "c")
    assert_equal(value_of(earliest, 3), "b")
    assert_equal(value_of(latest, 3), "c")


def test_the_parallel_route_agrees_with_the_serial_one() raises:
    """The same column reduced twice, once past the bound and once under it.

    A slice of the first sixty thousand rows takes the serial route because it is
    short, and the whole seventy thousand take the parallel one, so reducing the
    slice by hand against the head of the parallel answer is not the same thing.
    What is compared instead is the parallel answer against a plain loop written
    here, which is the shape the twin has and the reason it exists: the loop is
    obviously right and the kernel is not.
    """
    var n = 70000
    var groups = 6
    var rng = Rng(0x9E3779B9)
    var builder = StringBuilder(capacity=n)
    var codes = List[Int](capacity=n)
    var values = List[String](capacity=n)
    var present = List[Bool](capacity=n)
    for _ in range(n):
        codes.append(Int(rng.next_below(groups)))
        if rng.next_below(8) == 0:
            builder.append_null()
            values.append(String(""))
            present.append(False)
            continue
        var value = String("")
        for _ in range(Int(rng.next_range(1, 9))):
            value += chr(Int(rng.next_range(97, 101)))
        builder.append(value.as_bytes())
        values.append(value)
        present.append(True)
    var col = builder^.finish()

    var kinds = List[AggKind]()
    kinds.append(AggKind.FIRST)
    kinds.append(AggKind.LAST)
    kinds.append(AggKind.MIN)
    kinds.append(AggKind.MAX)
    for k in range(len(kinds)):
        var kind = kinds[k]
        var want = List[String](length=groups, fill=String(""))
        var found = List[Bool](length=groups, fill=False)
        for i in range(n):
            if not present[i]:
                continue
            var g = codes[i]
            if not found[g]:
                found[g] = True
                want[g] = values[i].copy()
                continue
            if kind == AggKind.FIRST:
                continue
            if kind == AggKind.LAST:
                want[g] = values[i].copy()
                continue
            if kind == AggKind.MIN:
                if values[i] < want[g]:
                    want[g] = values[i].copy()
            elif values[i] > want[g]:
                want[g] = values[i].copy()

        var got = reduce(StringArray(copy=col), kind, codes, groups)
        for g in range(groups):
            var note = String(kind) + " group " + String(g)
            assert_true(got.is_valid(g) == found[g], note)
            if found[g]:
                assert_equal(value_of(got, g), want[g], note)


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
    var at = values.unsafe_mut_ptr()
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


def whole(var col: StringArray, kind: AggKind) raises -> AnyArray:
    """Reduces a whole text column, with no grouping in the way."""
    return reduce_any(AnyArray(col^), kind)


def test_the_whole_column_extreme_matches_the_grouped_one() raises:
    """The two routes have to agree, because the grouped one with a single group
    is what the whole column one replaced and there is no third opinion."""
    var kinds = List[AggKind]()
    kinds.append(AggKind.MIN)
    kinds.append(AggKind.MAX)
    kinds.append(AggKind.FIRST)
    kinds.append(AggKind.LAST)

    var rng = Rng(0x5EED)
    var values = List[String]()
    var present = List[Bool]()
    for _ in range(5000):
        # A long shared prefix, so most comparisons have to leave the view and
        # read the payload, which is what a column of URLs looks like.
        values.append(
            "https://example.invalid/path/" + String(rng.next_below(9973))
        )
        present.append(rng.next_below(8) != 0)

    for k in range(len(kinds)):
        var direct = whole(with_nulls(values, present), kinds[k])
        var codes = List[Int]()
        for _ in range(len(values)):
            codes.append(0)
        var grouped = reduce(with_nulls(values, present), kinds[k], codes, 1)
        assert_equal(value_of(direct, 0), value_of(grouped, 0))


def test_a_whole_column_of_nulls_has_no_extreme() raises:
    var kinds = List[AggKind]()
    kinds.append(AggKind.MIN)
    kinds.append(AggKind.MAX)
    kinds.append(AggKind.FIRST)
    kinds.append(AggKind.LAST)
    for k in range(len(kinds)):
        var col = with_nulls(["a", "b", "c"], [False, False, False])
        var out = whole(col^, kinds[k])
        assert_equal(len(out), 1)
        assert_false(out.is_valid(0))


def test_an_empty_string_is_the_smallest_value_and_a_null_is_not() raises:
    """Two rules one line apart. The empty string is a value and sorts first, so
    it is the minimum. A null is not a value, so it is not the minimum even
    though it is the row a comparison would reach for."""
    var col = with_nulls(["b", "", "a"], [True, True, True])
    assert_equal(value_of(whole(col^, AggKind.MIN), 0), "")

    var holey = with_nulls(["b", "x", "a"], [True, False, True])
    assert_equal(value_of(whole(holey^, AggKind.MIN), 0), "a")


def test_the_whole_column_extreme_crosses_a_morsel() raises:
    """Past one morsel the scan is a row number per morsel and a merge over the
    slots, and a merge that kept the wrong side would still be right inside every
    morsel. The answer is put where only the merge can find it."""
    # A morsel is 128k rows, so this is two of them, the second one short. Every
    # row but two holds the same bytes, which keeps the column cheap to build and
    # leaves the answer somewhere only the merge over the slots can find it.
    var n = 128 * 1024 + 1000
    var filler = String("middling_row")
    var low = String("aaa_the_smallest")
    var high = String("zzz_the_largest")
    var builder = StringBuilder(capacity=n)
    for i in range(n):
        if i == 3:
            builder.append(high.as_bytes())
        elif i == n - 100:
            builder.append(low.as_bytes())
        else:
            builder.append(filler.as_bytes())
    var col = builder^.finish()
    assert_equal(
        value_of(whole(StringArray(copy=col), AggKind.MIN), 0),
        "aaa_the_smallest",
    )
    assert_equal(value_of(whole(col^, AggKind.MAX), 0), "zzz_the_largest")


def test_the_edges_of_a_whole_column_skip_the_nulls() raises:
    var col = with_nulls(["a", "b", "c", "d"], [False, True, True, False])
    assert_equal(value_of(whole(StringArray(copy=col), AggKind.FIRST), 0), "b")
    assert_equal(value_of(whole(col^, AggKind.LAST), 0), "c")


def test_nulls_and_an_empty_string_order_the_way_both_rivals_say() raises:
    """The ordering rule read off pandas 3.0.5 and DuckDB 1.5.5 rather than
    reasoned about.

    The column is `b`, a null, an empty string and `a`, which is the shape this
    dataset does not have and the next one will. Both rivals answer the same
    thing and this kernel answers it too:

        >>> pd.Series(["b", None, "", "a"], dtype="str").min()
        ''
        >>> pd.Series(["b", None, "", "a"], dtype="str").max()
        'b'

        select min(x), max(x) from (values ('b'), (null), (''), ('a')) t(x)
        ('', 'b')

    So an empty string is a value and sorts before every other value, and a null
    is not a value and is passed over. The two are one line apart in the kernel
    and one row apart here.

    pandas has a second answer available that this library does not offer.
    `min(skipna=False)` is NaN as soon as any element is null, which is a
    different question rather than a different result, and nothing in the
    ClickBench queries or in SQL asks it.
    """
    var col = with_nulls(["b", "x", "", "a"], [True, False, True, True])
    assert_equal(value_of(whole(StringArray(copy=col), AggKind.MIN), 0), "")
    assert_equal(value_of(whole(col^, AggKind.MAX), 0), "b")


def test_a_group_of_nulls_is_null_and_an_empty_string_group_is_not() raises:
    """The same rule one level down, with the group that has nothing in it beside
    the group whose only element is the empty string.

    Three groups: one holding `b` and a null, one holding an empty string and a
    null, one holding two nulls. pandas answers the same four ways for all four
    reductions, and so does this:

        >>> df.groupby("g")["v"].min().to_dict()
        {1: 'b', 2: '', 3: nan}

    pandas spells the missing group as NaN there because the column came back
    with a missing element in it, and a null is what that means here.

    DuckDB agrees on `min` and `max` and is not an authority on the other two.
    `first` and `last` over a group with no `ORDER BY` are not defined to follow
    row order in SQL, and DuckDB's `last` answered NULL for the first group in
    the same query, which is allowed rather than wrong. pandas' `first` and
    `last` skip nulls and follow row order, which is the rule this kernel
    follows, and it is the rule a dataframe user is asking for.
    """
    var kinds = List[AggKind]()
    kinds.append(AggKind.MIN)
    kinds.append(AggKind.MAX)
    kinds.append(AggKind.FIRST)
    kinds.append(AggKind.LAST)

    for k in range(len(kinds)):
        var col = with_nulls(
            ["b", "x", "", "y", "z", "w"],
            [True, False, True, False, False, False],
        )
        var out = reduce(col^, kinds[k], [0, 0, 1, 1, 2, 2], 3)
        assert_true(out.is_valid(0))
        assert_true(out.is_valid(1))
        assert_false(out.is_valid(2))
        assert_equal(value_of(out, 0), "b")
        assert_equal(value_of(out, 1), "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
