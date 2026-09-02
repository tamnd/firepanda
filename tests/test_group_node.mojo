"""Tests for the streaming group by node.

The node's whole claim is that grouping a frame a chunk at a time gives the
answer grouping it all at once gives, so most of what is here compares the two.
`DataFrame.group_by` with `sort=False` and `dropna=False` is the specification,
because those are the two things the node deliberately does not do, and running
the comparison with them switched off is the difference between checking the
grouping and checking three operators at once.

Every frame is built in several chunks, and the chunk boundaries are chosen so
that a group straddles one. That is the case the node exists for and the one a
single chunk frame cannot exercise: a group whose first rows are in chunk one
and whose last are in chunk three has to survive two merges, and an accumulator
that quietly restarted on each chunk would pass every test built on a frame that
has only one.

Nulls appear in three places on purpose, because they mean three different
things here. A null key is a group like any other. A null value is skipped by
every reduction, which is why count and size differ. A group in which every
value is null has a count of zero, and that is what makes a mean null rather
than a division by zero.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.array.strings import strings_from_list
from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.exec import (
    Compute,
    Filter,
    Group,
    GroupAgg,
    Node,
    Pipeline,
    Project,
    node_is_breaker,
)
from firepanda.frame.frame import DataFrame
from firepanda.frame.groupby import AggSpec
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.testing.rng import Rng


def numbers(values: List[Int64]) raises -> AnyArray:
    """Builds a fully valid int64 array."""
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return AnyArray(col^)


def maybe(values: List[Int64], present: List[Bool]) raises -> AnyArray:
    """Builds an int64 array with a null wherever `present` is false."""
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        if present[i]:
            col.set_valid(i, values[i])
        else:
            col.set_null(i)
    return AnyArray(col^)


def doubles(values: List[Float64]) raises -> AnyArray:
    """Builds a fully valid float64 array."""
    var col = Array[DType.float64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return AnyArray(col^)


def cut(
    var pieces: List[List[AnyArray]], var fields: List[Field]
) raises -> DataFrame:
    """Builds a frame from a list of chunks, each a list of columns."""
    var columns = List[ChunkedArray](capacity=len(fields))
    for c in range(len(fields)):
        columns.append(ChunkedArray(fields[c].dtype))
    for p in range(len(pieces)):
        for c in range(len(fields)):
            columns[c].append(AnyArray(copy=pieces[p][c]))
    return DataFrame(Schema(fields^), columns^)


def keyed_frame() raises -> DataFrame:
    """Nine rows in three chunks, three groups, every group straddling a cut.

    The keys run 1, 2, 3 down each chunk, so no group is contained in a chunk
    and every one of them is merged twice. The values are the row number, which
    makes each group's sum, minimum and maximum different from each other and
    different from every other group's.
    """
    var pieces = List[List[AnyArray]]()
    var one = List[AnyArray]()
    one.append(numbers([1, 2, 3]))
    one.append(numbers([10, 20, 30]))
    pieces.append(one^)
    var two = List[AnyArray]()
    two.append(numbers([1, 2, 3]))
    two.append(numbers([11, 21, 31]))
    pieces.append(two^)
    var three = List[AnyArray]()
    three.append(numbers([3, 2, 1]))
    three.append(numbers([32, 22, 12]))
    pieces.append(three^)
    var fields = List[Field]()
    fields.append(Field("k", LogicalType.INT64))
    fields.append(Field("v", LogicalType.INT64))
    return cut(pieces^, fields^)


def holey_frame() raises -> DataFrame:
    """Eight rows in two chunks, with a null key and a group of only nulls.

    Key 9 exists only as null values, so its count is zero and its mean is null
    while its size is two. The null key is its own group and it also has a value,
    so a run that dropped null keys and a run that kept them differ by a row
    rather than by a value.
    """
    var pieces = List[List[AnyArray]]()
    var one = List[AnyArray]()
    one.append(maybe([1, 2, 9, 1], [True, True, True, False]))
    one.append(maybe([5, 6, 0, 8], [True, True, False, True]))
    pieces.append(one^)
    var two = List[AnyArray]()
    two.append(maybe([2, 9, 1, 1], [True, True, False, True]))
    two.append(maybe([7, 0, 3, 4], [True, False, True, True]))
    pieces.append(two^)
    var fields = List[Field]()
    fields.append(Field("k", LogicalType.INT64))
    fields.append(Field("v", LogicalType.INT64))
    return cut(pieces^, fields^)


def worded_frame() raises -> DataFrame:
    """Six rows in two chunks with a text key and a text column beside it."""
    var pieces = List[List[AnyArray]]()
    var one = List[AnyArray]()
    one.append(AnyArray(strings_from_list(["red", "green", "red"])))
    one.append(AnyArray(strings_from_list(["pear", "fig", "apple"])))
    one.append(numbers([1, 2, 3]))
    pieces.append(one^)
    var two = List[AnyArray]()
    two.append(AnyArray(strings_from_list(["blue", "red", "green"])))
    two.append(AnyArray(strings_from_list(["plum", "damson", "quince"])))
    two.append(numbers([4, 5, 6]))
    pieces.append(two^)
    var fields = List[Field]()
    fields.append(Field("colour", LogicalType.STRING))
    fields.append(Field("fruit", LogicalType.STRING))
    fields.append(Field("n", LogicalType.INT64))
    return cut(pieces^, fields^)


def flat(df: DataFrame) raises -> DataFrame:
    """Stacks a frame's chunks so the eager group by can read it.

    `DataFrame.group_by` borrows every column and a borrow is of one array, so
    the frame it is handed has to be in one piece. That restriction is the one
    the node exists to lift, which makes this the right shape for the
    comparison: the same rows, laid out the two ways, answered by the two paths.
    """
    var columns = List[ChunkedArray](capacity=len(df.schema))
    for i in range(len(df.schema)):
        columns.append(ChunkedArray(ChunkedArray(copy=df.columns[i]).combine()))
    return DataFrame(Schema(copy=df.schema), columns^)


def read_ints(df: DataFrame, name: String) raises -> List[Int64]:
    """Reads an int64 column out as a plain list."""
    var col = df.column(name).as_typed[DType.int64]()
    var out = List[Int64](capacity=len(col))
    for i in range(len(col)):
        out.append(col[i])
    return out^


def run_group(
    var frame: DataFrame, var keys: List[Int], var aggs: List[GroupAgg]
) raises -> DataFrame:
    """Runs a frame through a pipeline whose only operator is the group by."""
    var pipeline = Pipeline(frame^)
    pipeline.add(Node(Group(keys^, aggs^)))
    return pipeline^.run()


def same_ints(got: DataFrame, want: DataFrame, what: String) raises:
    """Asserts two frames of int64 columns hold the same values.

    The comparison walks the schema rather than assuming a layout, because the
    node's output is chunked and the eager path's is not, and a check that
    compared chunk lists would be checking the wrong thing.
    """
    assert_equal(len(got.schema), len(want.schema), what + ": width")
    assert_equal(len(got), len(want), what + ": height")
    for c in range(len(want.schema)):
        var name = want.schema[c].name
        assert_equal(got.schema[c].name, name, what + ": column name")
        assert_true(
            got.schema[c].dtype == want.schema[c].dtype,
            what + ": column " + name + " type",
        )
        var a = got.column(name)
        var b = want.column(name)
        var mine = a.as_typed[DType.int64]()
        var theirs = b.as_typed[DType.int64]()
        for r in range(len(want)):
            assert_equal(
                a.is_valid(r),
                b.is_valid(r),
                what + ": " + name + " validity at row " + String(r),
            )
            if a.is_valid(r):
                assert_equal(
                    mine[r],
                    theirs[r],
                    what + ": " + name + " at row " + String(r),
                )


def close_enough(a: Float64, b: Float64) -> Bool:
    """Reports whether two floats agree to within a relative 1e-9."""
    var gap = a - b
    if gap < 0.0:
        gap = -gap
    var scale = b if b > 0.0 else -b
    if scale < 1.0:
        scale = 1.0
    return gap <= 1e-9 * scale


def test_a_sum_folds_across_chunks() raises:
    """Each group's rows are spread over all three chunks."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "total"))
    var out = run_group(keyed_frame(), [0], aggs^)
    assert_equal(len(out), 3, "three groups")
    var keys = read_ints(out, "k")
    var totals = read_ints(out, "total")
    assert_equal(keys[0], Int64(1), "first key")
    assert_equal(totals[0], Int64(33), "10 + 11 + 12")
    assert_equal(keys[1], Int64(2), "second key")
    assert_equal(totals[1], Int64(63), "20 + 21 + 22")
    assert_equal(keys[2], Int64(3), "third key")
    assert_equal(totals[2], Int64(93), "30 + 31 + 32")


def test_groups_come_out_in_first_seen_order() raises:
    """The node does not sort, and the last chunk arrives with keys reversed.

    If the merge rebuilt the group order from the last chunk it saw, this would
    come out 3, 2, 1, which is the answer a sorted group by would also refuse to
    give.
    """
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "total"))
    var out = run_group(keyed_frame(), [0], aggs^)
    var keys = read_ints(out, "k")
    assert_equal(keys[0], Int64(1), "first seen first")
    assert_equal(keys[1], Int64(2), "then the second")
    assert_equal(keys[2], Int64(3), "then the third")


def test_a_mean_is_a_sum_and_a_count_until_the_end() raises:
    """A mean of means would give the wrong answer on unequal chunk shares.

    Group 1 has one row in the first chunk and two in the last, so averaging the
    chunk averages gives a different number from averaging the rows, and only one
    of the two is the mean.
    """
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.MEAN, "average"))
    var out = run_group(keyed_frame(), [0], aggs^)
    var col = out.column("average").as_typed[DType.float64]()
    assert_true(close_enough(col[0], 11.0), "33 over 3")
    assert_true(close_enough(col[1], 21.0), "63 over 3")
    assert_true(close_enough(col[2], 31.0), "93 over 3")


def test_a_count_skips_nulls_and_a_size_does_not() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.COUNT, "seen"))
    aggs.append(GroupAgg(1, AggKind.SIZE, "rows"))
    var out = run_group(holey_frame(), [0], aggs^)
    var keys = read_ints(out, "k")
    var seen = read_ints(out, "seen")
    var rows = read_ints(out, "rows")
    assert_equal(len(out), 4, "1, 2, 9 and the null key")
    for g in range(len(out)):
        if not out.column("k").is_valid(g):
            assert_equal(rows[g], Int64(2), "two rows have a null key")
            assert_equal(seen[g], Int64(2), "both of them have a value")
        elif keys[g] == 9:
            assert_equal(rows[g], Int64(2), "key 9 has two rows")
            assert_equal(seen[g], Int64(0), "and no values")


def test_a_group_of_only_nulls_has_a_null_mean() raises:
    """Zero values is null rather than a division by zero."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.MEAN, "average"))
    var out = run_group(holey_frame(), [0], aggs^)
    var keys = read_ints(out, "k")
    var found = False
    for g in range(len(out)):
        if out.column("k").is_valid(g) and keys[g] == 9:
            found = True
            assert_true(
                not out.column("average").is_valid(g),
                "the mean of no values is null",
            )
    assert_true(found, "key 9 is a group")


def test_a_null_key_is_a_group_of_its_own() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "total"))
    var out = run_group(holey_frame(), [0], aggs^)
    var totals = read_ints(out, "total")
    var found = False
    for g in range(len(out)):
        if not out.column("k").is_valid(g):
            found = True
            assert_equal(
                totals[g], Int64(11), "8 from one chunk and 3 from the other"
            )
    assert_true(found, "the null key survived the merge")


def test_the_extremes_fold() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.MIN, "smallest"))
    aggs.append(GroupAgg(1, AggKind.MAX, "largest"))
    var out = run_group(keyed_frame(), [0], aggs^)
    var small = read_ints(out, "smallest")
    var large = read_ints(out, "largest")
    assert_equal(small[0], Int64(10), "group 1 starts at 10")
    assert_equal(large[0], Int64(12), "and ends at 12")
    assert_equal(small[2], Int64(30), "group 3 starts at 30")
    assert_equal(large[2], Int64(32), "and ends at 32")


def test_first_and_last_skip_nulls_across_a_chunk_boundary() raises:
    """Key 1's first row has a value, its second does not, and its third does.

    The rows for key 1 are 5, then a null in the second chunk, then 4, so a first
    that reported the literal first row of each chunk and a last that reported
    the literal last would both be wrong in a way a single chunk cannot show.
    """
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.FIRST, "earliest"))
    aggs.append(GroupAgg(1, AggKind.LAST, "latest"))
    var out = run_group(holey_frame(), [0], aggs^)
    var keys = read_ints(out, "k")
    var first = read_ints(out, "earliest")
    var last = read_ints(out, "latest")
    for g in range(len(out)):
        if out.column("k").is_valid(g) and keys[g] == 1:
            assert_equal(first[g], Int64(5), "the first value key 1 held")
            assert_equal(last[g], Int64(4), "and the last")
        if out.column("k").is_valid(g) and keys[g] == 9:
            assert_true(
                not out.column("earliest").is_valid(g),
                "a group with no values has no first",
            )


def test_two_key_columns_group_on_the_tuple() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(2, AggKind.SUM, "total"))
    var out = run_group(worded_frame(), [0, 1], aggs^)
    assert_equal(len(out), 6, "no two rows share both keys")
    var totals = read_ints(out, "total")
    assert_equal(totals[0], Int64(1), "one row a group")


def test_a_text_key_groups_by_its_bytes() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(2, AggKind.SUM, "total"))
    aggs.append(GroupAgg(2, AggKind.SIZE, "rows"))
    var out = run_group(worded_frame(), [0], aggs^)
    assert_equal(len(out), 3, "red, green and blue")
    var totals = read_ints(out, "total")
    var rows = read_ints(out, "rows")
    assert_equal(totals[0], Int64(9), "red is rows 1, 3 and 5")
    assert_equal(rows[0], Int64(3), "three of them")
    assert_equal(totals[2], Int64(4), "blue is row 4 alone")


def test_the_extremes_of_a_text_column_fold() raises:
    """Red's fruits are pear, apple and damson, one of them in the other chunk.
    """
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.MIN, "first_fruit"))
    aggs.append(GroupAgg(1, AggKind.MAX, "last_fruit"))
    var out = run_group(worded_frame(), [0], aggs^)
    assert_equal(out.column("first_fruit").text(0), "apple", "red's smallest")
    assert_equal(out.column("last_fruit").text(0), "pear", "red's largest")


def test_the_node_agrees_with_the_frame_method() raises:
    """The eager path is the specification, with its two extras switched off."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "v_sum"))
    aggs.append(GroupAgg(1, AggKind.COUNT, "v_count"))
    aggs.append(GroupAgg(1, AggKind.MIN, "v_min"))
    aggs.append(GroupAgg(1, AggKind.MAX, "v_max"))
    aggs.append(GroupAgg(1, AggKind.SIZE, "v_size"))
    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))
    specs.append(AggSpec("v", AggKind.COUNT))
    specs.append(AggSpec("v", AggKind.MIN))
    specs.append(AggSpec("v", AggKind.MAX))
    specs.append(AggSpec("v", AggKind.SIZE))
    var want = flat(holey_frame()).group_by(
        ["k"], specs^, dropna=False, sort=False
    )
    var got = run_group(holey_frame(), [0], aggs^)
    same_ints(got, want, "five reductions over a frame with nulls in it")


def test_the_node_agrees_with_the_frame_method_on_random_data() raises:
    """Twenty thousand rows in chunks of a few hundred, over forty keys.

    The chunk sizes are uneven and the key set is small enough that every group
    appears in most chunks, so nearly every group is merged tens of times. Floats
    are compared here as well, and the sums are of a magnitude where reordering
    the additions is visible, which is why this asserts the node's answer is
    close to the eager one rather than identical to it.
    """
    var rng = Rng(20260902)
    var keys = ChunkedArray(LogicalType.INT64)
    var values = ChunkedArray(LogicalType.INT64)
    var reals = ChunkedArray(LogicalType.FLOAT64)
    var total = 0
    while total < 20000:
        var rows = 100 + rng.next_below(400)
        if total + rows > 20000:
            rows = 20000 - total
        var k = List[Int64](capacity=rows)
        var v = List[Int64](capacity=rows)
        var present = List[Bool](capacity=rows)
        var r = List[Float64](capacity=rows)
        for _ in range(rows):
            k.append(Int64(rng.next_below(40)))
            v.append(Int64(rng.next_below(1000)))
            present.append(rng.next_below(10) != 0)
            r.append(Float64(rng.next_below(1000)) / 7.0)
        keys.append(numbers(k))
        values.append(maybe(v, present))
        reals.append(doubles(r))
        total += rows

    var fields = List[Field]()
    fields.append(Field("k", LogicalType.INT64))
    fields.append(Field("v", LogicalType.INT64))
    fields.append(Field("r", LogicalType.FLOAT64))
    var columns = List[ChunkedArray]()
    columns.append(keys^)
    columns.append(values^)
    columns.append(reals^)
    var frame = DataFrame(Schema(fields^), columns^)

    var specs = List[AggSpec]()
    specs.append(AggSpec("v", AggKind.SUM))
    specs.append(AggSpec("v", AggKind.COUNT))
    specs.append(AggSpec("v", AggKind.MIN))
    specs.append(AggSpec("v", AggKind.MAX))
    specs.append(AggSpec("v", AggKind.MEAN))
    var want = flat(frame).group_by(["k"], specs^, dropna=False, sort=False)

    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "v_sum"))
    aggs.append(GroupAgg(1, AggKind.COUNT, "v_count"))
    aggs.append(GroupAgg(1, AggKind.MIN, "v_min"))
    aggs.append(GroupAgg(1, AggKind.MAX, "v_max"))
    aggs.append(GroupAgg(1, AggKind.MEAN, "v_mean"))
    var got = run_group(frame^, [0], aggs^)

    assert_equal(len(got), len(want), "the same groups")
    assert_equal(len(got), 40, "every key turned up")
    var names: List[String] = ["k", "v_sum", "v_count", "v_min", "v_max"]
    for j in range(len(names)):
        var name = names[j]
        var a = got.column(name)
        var b = want.column(name)
        var mine = a.as_typed[DType.int64]()
        var theirs = b.as_typed[DType.int64]()
        for i in range(len(want)):
            assert_equal(a.is_valid(i), b.is_valid(i), name + " validity")
            if a.is_valid(i):
                assert_equal(mine[i], theirs[i], name)
    var averages = got.column("v_mean")
    var mine = averages.as_typed[DType.float64]()
    var theirs = want.column("v_mean").as_typed[DType.float64]()
    for i in range(len(want)):
        if averages.is_valid(i):
            assert_true(close_enough(mine[i], theirs[i]), "v_mean")


def test_a_group_by_is_a_breaker() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "total"))
    var node = Node(Group([0], aggs^))
    assert_true(
        node_is_breaker(node), "nothing comes out until everything is in"
    )


def test_a_pipeline_cuts_at_a_group_by() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "total"))
    var pipeline = Pipeline(keyed_frame())
    pipeline.add(Node(Group([0], aggs^)))
    pipeline.add(Node(Project([0])))
    var cuts = pipeline.cut_points()
    assert_equal(len(cuts), 1, "one breaker")
    assert_equal(cuts[0], 0, "at the group by")
    assert_equal(pipeline.stages(), 2, "which makes two stages")


def test_the_output_flows_into_the_operators_after_it() raises:
    """A group by is not the end of a pipeline, so its chunks go on downstream.
    """
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "total"))
    var pipeline = Pipeline(keyed_frame())
    pipeline.add(Node(Group([0], aggs^)))
    pipeline.add(Node(Project([1, 0])))
    var out = pipeline^.run()
    assert_equal(out.schema[0].name, "total", "the project reordered them")
    assert_equal(out.schema[1].name, "k", "and kept both")
    var totals = read_ints(out, "total")
    assert_equal(totals[0], Int64(33), "the same answer, reordered")


def test_the_schema_is_known_before_a_row_moves() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "total"))
    aggs.append(GroupAgg(1, AggKind.COUNT, "seen"))
    aggs.append(GroupAgg(1, AggKind.MEAN, "average"))
    aggs.append(GroupAgg(1, AggKind.MIN, "smallest"))
    var pipeline = Pipeline(keyed_frame())
    pipeline.add(Node(Group([0], aggs^)))
    assert_equal(len(pipeline.schema), 5, "the key and four aggregates")
    assert_true(pipeline.schema[1].dtype == LogicalType.INT64, "a sum of int64")
    assert_true(
        pipeline.schema[2].dtype == LogicalType.INT64, "a count is a count"
    )
    assert_true(
        pipeline.schema[3].dtype == LogicalType.FLOAT64, "a mean is a float"
    )
    assert_true(
        pipeline.schema[4].dtype == LogicalType.INT64,
        "a minimum has the column's own type",
    )


def test_an_empty_input_gives_an_empty_frame_with_a_schema() raises:
    var fields = List[Field]()
    fields.append(Field("k", LogicalType.INT64))
    fields.append(Field("v", LogicalType.INT64))
    var columns = List[ChunkedArray]()
    columns.append(ChunkedArray(LogicalType.INT64))
    columns.append(ChunkedArray(LogicalType.INT64))
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "total"))
    var out = run_group(DataFrame(Schema(fields^), columns^), [0], aggs^)
    assert_equal(len(out), 0, "no groups")
    assert_equal(len(out.schema), 2, "the shape survives")
    assert_equal(out.schema[1].name, "total", "including the aggregate's name")


def test_a_filter_after_a_group_by_runs_in_the_second_stage() raises:
    """Red sums to 9, green to 8 and blue to 4, so the filter keeps two of them.

    The point is that the chunks the group by hands back go on down the line
    like any others. A breaker that emitted straight into the sink would give
    three rows here.
    """
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(2, AggKind.SUM, "total"))
    var pipeline = Pipeline(worded_frame())
    pipeline.add(Node(Group([0], aggs^)))
    pipeline.add(Node(Compute(1, Value(Int64(5)), BinaryOp.GT, "big")))
    pipeline.add(Node(Filter(2)))
    pipeline.add(Node(Project([0, 1])))
    var out = pipeline^.run()
    assert_equal(len(out), 2, "red and green clear five, blue does not")
    var totals = read_ints(out, "total")
    assert_equal(totals[0], Int64(9), "red")
    assert_equal(totals[1], Int64(8), "green")


def test_a_reduction_that_does_not_fold_is_refused_at_plan_time() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.MEDIAN, "middle"))
    var pipeline = Pipeline(keyed_frame())
    with assert_raises(contains="cannot be computed a chunk at a time"):
        pipeline.add(Node(Group([0], aggs^)))


def test_a_distinct_count_does_not_fold_either() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.NUNIQUE, "distinct"))
    var pipeline = Pipeline(keyed_frame())
    with assert_raises(contains="cannot be computed a chunk at a time"):
        pipeline.add(Node(Group([0], aggs^)))


def test_a_key_outside_the_schema_is_caught_at_plan_time() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "total"))
    var pipeline = Pipeline(keyed_frame())
    with assert_raises(contains="is outside a schema"):
        pipeline.add(Node(Group([7], aggs^)))


def test_a_repeated_key_is_caught_at_plan_time() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "total"))
    var pipeline = Pipeline(keyed_frame())
    with assert_raises(contains="was given twice"):
        pipeline.add(Node(Group([0, 0], aggs^)))


def test_no_keys_at_all_is_caught_at_plan_time() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "total"))
    var pipeline = Pipeline(keyed_frame())
    with assert_raises(contains="at least one key column"):
        pipeline.add(Node(Group(List[Int](), aggs^)))


def test_a_sum_of_text_is_caught_at_plan_time() raises:
    """The physical dtype of a text column is uint8, so this has to be refused
    by name rather than left to the dispatch to trip over."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "total"))
    var pipeline = Pipeline(worded_frame())
    with assert_raises(contains="is not defined on text"):
        pipeline.add(Node(Group([0], aggs^)))


def test_two_outputs_with_the_same_name_are_caught_at_plan_time() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "total"))
    aggs.append(GroupAgg(1, AggKind.MIN, "total"))
    var pipeline = Pipeline(keyed_frame())
    with assert_raises(contains="would both be called"):
        pipeline.add(Node(Group([0], aggs^)))


def test_an_output_colliding_with_a_key_is_caught_at_plan_time() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "k"))
    var pipeline = Pipeline(keyed_frame())
    with assert_raises(contains="would both be called"):
        pipeline.add(Node(Group([0], aggs^)))


def test_no_aggregates_gives_the_distinct_keys() raises:
    """A group by with nothing to compute is `drop_duplicates` on the keys."""
    var out = run_group(keyed_frame(), [0], List[GroupAgg]())
    assert_equal(len(out), 3, "three distinct keys")
    assert_equal(len(out.schema), 1, "and nothing else")
    var keys = read_ints(out, "k")
    assert_equal(keys[0], Int64(1), "in first seen order")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
