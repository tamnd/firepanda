"""Tests for the chunk, the nodes and the pipeline driver.

Two things are being checked and they are different. The first is that a
pipeline gives the same answer the frame methods give, which is what makes the
engine safe to move operators onto one at a time. The second is that it gives it
without doing work it did not have to do, which is the only reason to have an
engine at all: a limit stops the scan, a filter that keeps nothing stops a chunk
travelling, and the chunk boundaries of the input survive to the output rather
than being flattened somewhere in the middle.

Frames here are built in several chunks on purpose. A single chunk frame is the
easy case and it hides every off by one in the prefix sums, in the reverse pop
order the scan uses, and in the way a breaker hands its result back.
"""

from std.math import isnan
from std.testing import TestSuite, assert_equal, assert_false, assert_raises
from std.testing import assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.array.strings import strings_from_list
from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.exec import (
    Apply,
    Cast,
    Chunk,
    Collect,
    Compute,
    Constant,
    Expand,
    Filter,
    Group,
    GroupAgg,
    Join,
    Limit,
    Match,
    Materialize,
    Node,
    NodeStatus,
    Pipeline,
    Project,
    Reduce,
    Scan,
    Sort,
    Unique,
    Window,
    node_apply,
    node_computes_per_row,
    node_ends_early,
    node_is_breaker,
    node_is_row_local,
    node_status,
)
from firepanda.frame.frame import DataFrame
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.pattern import MatchKind, Pattern
from firepanda.kernel.unary import UnaryOp


def numbers(values: List[Int64]) raises -> AnyArray:
    """Builds a fully valid int64 array."""
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return AnyArray(col^)


def flags(values: List[Bool]) raises -> AnyArray:
    """Builds a fully valid boolean array."""
    var col = Array[DType.bool](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return AnyArray(col^)


def sample_frame() raises -> DataFrame:
    """Six rows in one chunk: a counting column and a mask over it.

    The mask keeps rows 0, 2, 3 and 5, so it is neither everything nor a prefix,
    and the values it keeps are 1, 3, 4 and 6.
    """
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(flags([True, False, True, True, False, True]))
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.BOOL))
    return DataFrame(Schema(fields^), columns^)


def cut_frame() raises -> DataFrame:
    """The same six rows, in chunks of two, three and one."""
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2]))
    n.append(numbers([3, 4, 5]))
    n.append(numbers([6]))
    var keep = ChunkedArray(LogicalType.BOOL)
    keep.append(flags([True, False]))
    keep.append(flags([True, True, False]))
    keep.append(flags([True]))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(keep^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.BOOL))
    return DataFrame(Schema(fields^), columns^)


def many_chunk_frame() raises -> DataFrame:
    """Two hundred rows in forty chunks of five, with a mask over them.

    Forty is chosen to be more than one batch on this machine and more than one
    on a smaller one, since a batch is one chunk per worker and the driver takes
    the parallel route from two chunks upwards. The mask drops every third row,
    so what survives is neither a prefix nor a stride the reader could guess,
    and the numbers are consecutive, so the order of the answer is checkable by
    looking at it rather than by comparing against a second run.
    """
    var n = ChunkedArray(LogicalType.INT64)
    var keep = ChunkedArray(LogicalType.BOOL)
    for c in range(40):
        var values = List[Int64]()
        var mask = List[Bool]()
        for r in range(5):
            var v = Int64(c * 5 + r + 1)
            values.append(v)
            mask.append(v % 3 != 0)
        n.append(numbers(values))
        keep.append(flags(mask))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(keep^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.BOOL))
    return DataFrame(Schema(fields^), columns^)


def word_frame() raises -> DataFrame:
    """The same six rows with two text columns beside the numbers.

    `status` and `wanted` agree on rows 0, 2 and 4, so a comparison between the
    two columns keeps three rows, and `status` holds "ok" on four of the six, so a
    comparison against that constant keeps a different set. Two answers that
    differ is the point: a comparison that ignored one of its operands would give
    the same set twice.
    """
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(
        AnyArray(strings_from_list(["ok", "fail", "ok", "ok", "fail", "ok"]))
    )
    columns.append(
        AnyArray(strings_from_list(["ok", "ok", "ok", "fail", "fail", "no"]))
    )
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("status", LogicalType.STRING))
    fields.append(Field("wanted", LogicalType.STRING))
    return DataFrame(Schema(fields^), columns^)


def read_back(df: DataFrame, name: String) raises -> List[Int64]:
    """Reads an int64 column out as a plain list."""
    var col = df.column(name).as_typed[DType.int64]()
    var out = List[Int64](capacity=len(col))
    for i in range(len(col)):
        out.append(col[i])
    return out^


def gappy_frame() raises -> DataFrame:
    """Six rows in two chunks, with a null in each of them.

    The nulls sit in different chunks on purpose, since a sort flattens the
    column before it orders it and a null in the second chunk is the one whose
    validity bit has to travel the furthest.
    """
    var n = ChunkedArray(LogicalType.INT64)
    var first = Array[DType.int64](3)
    first.set_valid(0, Int64(4))
    first.set_null(1)
    first.set_valid(2, Int64(1))
    n.append(AnyArray(first^))
    var second = Array[DType.int64](3)
    second.set_valid(0, Int64(6))
    second.set_valid(1, Int64(2))
    second.set_null(2)
    n.append(AnyArray(second^))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64, True))
    return DataFrame(Schema(fields^), columns^)


def present(df: DataFrame, name: String) raises -> List[Bool]:
    """Which rows of an int64 column have a value in them."""
    var col = df.column(name).as_typed[DType.int64]()
    var out = List[Bool](capacity=len(col))
    for i in range(len(col)):
        out.append(col.is_valid(i))
    return out^


def identity(var df: DataFrame) raises -> DataFrame:
    """A whole frame operation that does nothing, for the fallback tests."""
    return df^


def first_two(var df: DataFrame) raises -> DataFrame:
    """A whole frame operation that keeps the first two rows."""
    return df.head(2)


def only_n(var df: DataFrame) raises -> DataFrame:
    """A whole frame operation that drops a column, so the schema changes."""
    return df.select(["n"])


def test_a_chunk_refuses_columns_of_different_lengths() raises:
    var mixed = List[AnyArray]()
    mixed.append(numbers([1, 2, 3]))
    mixed.append(numbers([1, 2]))
    with assert_raises(contains="column 1 has 2 rows"):
        _ = Chunk(mixed^)


def test_a_scan_hands_out_the_chunks_the_frame_was_made_of() raises:
    var scan = Scan(cut_frame())
    assert_equal(scan.num_chunks(), 3, "chunks to hand out")

    var sizes = List[Int]()
    while True:
        var chunk = scan.next()
        if not chunk:
            break
        var got = chunk.take()
        assert_equal(got.width(), 2, "columns in a chunk")
        sizes.append(len(got))
    assert_equal(len(sizes), 3, "chunks handed out")
    assert_equal(sizes[0], 2, "first chunk")
    assert_equal(sizes[1], 3, "second chunk")
    assert_equal(sizes[2], 1, "third chunk")


def test_a_scan_refuses_columns_cut_in_different_places() raises:
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2]))
    n.append(numbers([3]))
    var m = ChunkedArray(LogicalType.INT64)
    m.append(numbers([1, 2, 3]))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(m^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("m", LogicalType.INT64))
    var df = DataFrame(Schema(fields^), columns^)

    with assert_raises(contains="column 1 has 1 chunks"):
        _ = Scan(df^)


def test_a_pipeline_of_nothing_gives_the_frame_back() raises:
    var out = Pipeline(cut_frame()).run()
    assert_equal(len(out), 6, "rows")
    assert_equal(out.width(), 2, "columns")
    var got = read_back(out, "n")
    for i in range(6):
        assert_equal(got[i], Int64(i + 1), "row " + String(i))


def test_a_filter_and_a_projection_agree_with_the_frame_methods() raises:
    var expected = sample_frame()
    var mask = expected.column("keep").as_typed[DType.bool]()
    var by_hand = expected.filter(mask).select(["n"])

    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Project([0])))
    var out = pipeline^.run()

    assert_equal(out.width(), 1, "the mask column was projected away")
    assert_equal(len(out), len(by_hand), "rows")
    var got = read_back(out, "n")
    var want = read_back(by_hand, "n")
    for i in range(len(want)):
        assert_equal(got[i], want[i], "row " + String(i))


def test_a_filter_told_what_to_keep_does_the_projection_itself() raises:
    var expected = sample_frame()
    var mask = expected.column("keep").as_typed[DType.bool]()
    var by_hand = expected.filter(mask).select(["n"])

    # The same query as the test above, as one operator rather than two. The
    # mask is never written out, which is the point: filtering by a column
    # produces a column that is all true and that nobody reads.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Filter(1, [0])))
    var out = pipeline^.run()

    assert_equal(out.width(), 1, "only the column that was asked for")
    assert_equal(out.schema[0].name, "n", "and it is the right one")
    assert_equal(len(out), len(by_hand), "rows")
    var got = read_back(out, "n")
    var want = read_back(by_hand, "n")
    for i in range(len(want)):
        assert_equal(got[i], want[i], "row " + String(i))


def test_a_narrowing_filter_may_reorder_and_repeat() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Filter(1, [1, 0, 0])))
    var out = pipeline^.run()

    assert_equal(out.width(), 3, "three columns out of two")
    assert_equal(out.schema[0].name, "keep", "the mask, kept on purpose")
    assert_equal(out.schema[1].name, "n", "second")
    assert_equal(out.schema[2].name, "n", "and again")
    var first = read_back(out, "n")
    assert_equal(len(first), 4, "rows kept")


def test_a_narrowing_filter_that_keeps_nothing_still_counts() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Filter(1, List[Int]())))
    var out = pipeline^.run()

    # A filter asked for no columns is a row count, and it still has to know
    # how many rows survived so that whatever follows it is told the truth.
    assert_equal(out.width(), 0, "no columns")
    assert_equal(len(out), 0, "and so no rows to report")


def test_a_narrowing_filter_refuses_a_column_it_does_not_have() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="outside a schema of 2 columns"):
        pipeline.add(Node(Filter(1, [5])))


def test_an_expansion_writes_each_row_as_many_times_as_it_is_told() raises:
    # The counting column is its own count, so the one is written once, the two
    # twice and so on, which is twenty one rows out of six.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Expand(0, [0])))
    var out = pipeline^.run()

    assert_equal(out.width(), 1, "only the column that was asked for")
    assert_equal(
        len(out), 21, "one plus two plus three plus four plus five plus six"
    )
    var got = read_back(out, "n")
    assert_equal(got[0], 1, "the one, once")
    assert_equal(got[1], 2, "the two")
    assert_equal(got[2], 2, "and again")
    assert_equal(got[3], 3, "the three")
    assert_equal(got[20], 6, "the last of the sixes")


def test_an_expansion_writes_nothing_for_a_count_of_none() raises:
    var counts = List[AnyArray]()
    counts.append(numbers([2, 0, -3, 1]))
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    var pipeline = Pipeline(DataFrame(Schema(fields^), counts^))
    pipeline.add(Node(Expand(0, [0])))
    var out = pipeline^.run()

    # Zero and a negative both write no copies, which is what a difference of
    # two counts asks for when the right side had more of the row.
    assert_equal(len(out), 3, "two twos and one one")
    var got = read_back(out, "n")
    assert_equal(got[0], 2, "first")
    assert_equal(got[1], 2, "second")
    assert_equal(got[2], 1, "third")


def test_an_expansion_refuses_a_column_it_does_not_have() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="expand: column 5 is outside a schema of 2"):
        pipeline.add(Node(Expand(5, [0])))


def test_an_expansion_refuses_to_write_a_column_it_does_not_have() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="expand: column 7 is outside a schema of 2"):
        pipeline.add(Node(Expand(0, [7])))


def shelf_frame() raises -> DataFrame:
    """Six rows in chunks of two, three and one: a shop and what it held.

    Shop 1 is written three times and shop 2 twice, and the repeats are split
    over the chunk boundaries, so a distinct on the shop cannot answer from one
    chunk. What each row held differs inside a group, so which row of a group
    survived can be read off the answer.
    """
    var shop = ChunkedArray(LogicalType.INT64)
    shop.append(numbers([1, 2]))
    shop.append(numbers([1, 3, 2]))
    shop.append(numbers([1]))
    var held = ChunkedArray(LogicalType.INT64)
    held.append(numbers([10, 20]))
    held.append(numbers([30, 40, 50]))
    held.append(numbers([60]))
    var columns = List[ChunkedArray]()
    columns.append(shop^)
    columns.append(held^)
    var fields = List[Field]()
    fields.append(Field("shop", LogicalType.INT64))
    fields.append(Field("held", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def test_a_unique_keeps_the_first_row_of_each_group() raises:
    var pipeline = Pipeline(shelf_frame())
    pipeline.add(Node(Unique([0])))
    var out = pipeline^.run()

    assert_equal(len(out), 3, "one row per shop")
    var shops = read_back(out, "shop")
    assert_equal(shops[0], 1, "the first shop")
    assert_equal(shops[1], 2, "the second")
    assert_equal(shops[2], 3, "the third")
    # 10, 20 and 40 are the first rows of the three shops, so the rows that
    # came out are whole rows and not a fold over each group.
    var held = read_back(out, "held")
    assert_equal(held[0], 10, "what the first shop held")
    assert_equal(held[1], 20, "the second")
    assert_equal(held[2], 40, "the third")


def test_a_unique_on_a_column_with_no_repeats_keeps_everything() raises:
    var pipeline = Pipeline(shelf_frame())
    pipeline.add(Node(Unique([1])))
    var out = pipeline^.run()

    assert_equal(len(out), 6, "every row")
    var held = read_back(out, "held")
    for i in range(6):
        assert_equal(held[i], Int64(10 * (i + 1)), "in the order they arrived")


def test_a_unique_is_a_breaker() raises:
    var node = Node(Unique([0]))
    assert_true(node_is_breaker(node), "it holds every row it is given")
    assert_false(node_is_row_local(node), "a group spans the chunks")


def test_a_unique_refuses_a_column_it_does_not_have() raises:
    var pipeline = Pipeline(shelf_frame())
    with assert_raises(contains="unique: column 4 is outside a schema of 2"):
        pipeline.add(Node(Unique([4])))


def test_a_unique_refuses_the_same_column_twice() raises:
    var pipeline = Pipeline(shelf_frame())
    with assert_raises(contains="unique: column 0 was given twice"):
        pipeline.add(Node(Unique([0, 0])))


def test_a_unique_refuses_to_decide_on_nothing() raises:
    with assert_raises(contains="unique: a distinct on no column at all"):
        _ = Unique(List[Int]())


def test_a_filter_keeps_the_chunk_boundaries() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Filter(1)))
    var out = pipeline^.run()

    # Chunks of two, three and one keep one, two and one row, so all three
    # survive. A chunk that emptied would not, because the driver drops a chunk
    # of no rows rather than passing it on.
    assert_equal(out.columns[0].num_chunks(), 3, "chunks out")
    assert_equal(len(out), 4, "rows kept")


def test_a_filter_that_empties_a_chunk_drops_it() raises:
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2]))
    n.append(numbers([3, 4]))
    var keep = ChunkedArray(LogicalType.BOOL)
    keep.append(flags([True, False]))
    keep.append(flags([False, False]))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(keep^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.BOOL))

    var pipeline = Pipeline(DataFrame(Schema(fields^), columns^))
    pipeline.add(Node(Filter(1)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "rows kept")
    assert_equal(out.columns[0].num_chunks(), 1, "the empty chunk is gone")


def test_a_projection_may_repeat_a_column() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Project([0, 0])))
    var out = pipeline^.run()
    assert_equal(out.width(), 2, "columns")
    assert_equal(out.schema[0].name, "n", "first name")
    assert_equal(out.schema[1].name, "n", "second name")
    assert_equal(len(out), 6, "rows")


def test_a_projection_outside_the_schema_is_refused_when_it_is_added() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="outside a schema of 2 columns"):
        pipeline.add(Node(Project([5])))


def test_a_limit_stops_the_scan() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Limit(3)))
    var out = pipeline^.run()
    assert_equal(len(out), 3, "rows")
    var got = read_back(out, "n")
    assert_equal(got[0], 1, "first")
    assert_equal(got[1], 2, "second")
    assert_equal(got[2], 3, "third")


def test_a_limit_says_it_is_finished_and_the_driver_believes_it() raises:
    # The limit falls in the middle of the second chunk, so the third is never
    # read. What is being checked is the node's own answer, since the driver
    # asking is what turns it into work not done.
    var node = Node(Limit(3))
    assert_true(
        node_status(node) == NodeStatus.NEED_MORE_INPUT, "before any rows"
    )

    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Limit(3)))
    var out = pipeline^.run()
    assert_equal(len(out), 3, "rows")
    assert_equal(out.columns[0].num_chunks(), 2, "chunks the limit cut across")


def test_a_limit_of_zero_reads_nothing() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Limit(0)))
    var out = pipeline^.run()
    assert_equal(len(out), 0, "rows")
    assert_equal(out.width(), 2, "the schema still describes the result")


def test_a_projection_may_rename_what_it_keeps() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Project([0], ["count"])))
    var out = pipeline^.run()
    assert_equal(out.width(), 1, "columns")
    assert_equal(out.schema[0].name, "count", "the name it was given")
    assert_equal(len(out), 6, "rows")


def test_a_projection_with_a_name_per_column_renames_them_all() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Project([1, 0], ["yes", "count"])))
    var out = pipeline^.run()
    assert_equal(out.schema[0].name, "yes", "first")
    assert_equal(out.schema[1].name, "count", "second")


def test_a_projection_with_the_wrong_number_of_names_is_refused() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="1 names for 2 columns"):
        pipeline.add(Node(Project([0, 1], ["count"])))


def test_a_limit_can_skip_rows_before_it_counts() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Limit(2, 3)))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(len(got), 2, "rows")
    assert_equal(got[0], 4, "first")
    assert_equal(got[1], 5, "second")


def test_a_skip_that_covers_a_whole_chunk_drops_it() raises:
    # The first chunk is two rows and the skip is two, so the first chunk goes
    # entirely and nothing is cut.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Limit(2, 2)))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(len(got), 2, "rows")
    assert_equal(got[0], 3, "first")
    assert_equal(got[1], 4, "second")


def test_a_limit_with_no_bound_only_skips() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Limit(-1, 4)))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(len(got), 2, "rows")
    assert_equal(got[0], 5, "first")
    assert_equal(got[1], 6, "second")


def test_a_limit_with_no_bound_never_says_it_is_finished() raises:
    # A limit that only skips has nothing to stop for, and a node that said
    # FINISHED would cut the pipeline off before the rows it was keeping.
    var node = Node(Limit(-1, 2))
    assert_true(
        node_status(node) == NodeStatus.NEED_MORE_INPUT, "before any rows"
    )


def test_a_skip_past_the_end_lets_nothing_through() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Limit(3, 10)))
    var out = pipeline^.run()
    assert_equal(len(out), 0, "rows")
    assert_equal(out.width(), 2, "the schema still describes the result")


def test_a_fallback_runs_a_whole_frame_operation() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Materialize(identity)))
    var out = pipeline^.run()
    assert_equal(len(out), 6, "rows")
    assert_equal(out.columns[0].num_chunks(), 3, "chunks preserved through it")
    var got = read_back(out, "n")
    for i in range(6):
        assert_equal(got[i], Int64(i + 1), "row " + String(i))


def test_a_fallback_is_a_breaker_and_cuts_the_pipeline() raises:
    assert_false(node_is_breaker(Node(Filter(1))), "a filter is not a breaker")
    assert_false(node_is_breaker(Node(Limit(2))), "a limit is not a breaker")
    assert_true(
        node_is_breaker(Node(Materialize(identity))), "a fallback is one"
    )

    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Materialize(identity)))
    pipeline.add(Node(Limit(2)))
    var cuts = pipeline.cut_points()
    assert_equal(len(cuts), 1, "one breaker")
    assert_equal(cuts[0], 1, "at the second operator")
    assert_equal(pipeline.stages(), 2, "stages")


def test_operators_after_a_breaker_see_what_it_produced() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Materialize(first_two)))
    pipeline.add(Node(Project([0])))
    var out = pipeline^.run()
    assert_equal(len(out), 2, "rows")
    assert_equal(out.width(), 1, "columns")
    var got = read_back(out, "n")
    assert_equal(got[0], 1, "first")
    assert_equal(got[1], 2, "second")


def test_a_fallback_that_changes_the_schema_has_to_say_so() raises:
    var narrow = List[Field]()
    narrow.append(Field("n", LogicalType.INT64))

    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Materialize(only_n, Schema(narrow^))))
    var out = pipeline^.run()
    assert_equal(out.width(), 1, "columns")
    assert_equal(len(out), 6, "rows")


def test_a_fallback_that_lies_about_its_schema_is_caught() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Materialize(only_n)))
    with assert_raises(contains="was declared to return"):
        _ = pipeline^.run()


def test_a_sink_that_saw_nothing_still_has_the_right_shape() raises:
    var sink = Collect()
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.BOOL))
    var out = sink^.into_frame(Schema(fields^))
    assert_equal(len(out), 0, "rows")
    assert_equal(out.width(), 2, "columns")
    assert_true(out.schema[0].dtype == LogicalType.INT64, "first dtype")


def test_an_applied_column_goes_on_the_end_and_leaves_its_input() raises:
    # The whole difference between this and a `Cast`, which converts where it
    # lies. `SELECT a, -a` wants both, so turning column zero over would change
    # what every expression already bound against that position means.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Apply(0, UnaryOp.NEG, "down")))
    var out = pipeline^.run()

    assert_equal(out.width(), 3, "the answer was appended")
    assert_equal(out.schema[2].name, "down", "the name it was given")
    assert_true(out.schema[2].dtype == LogicalType.INT64, "the same type")
    var got = read_back(out, "down")
    var kept = read_back(out, "n")
    for i in range(6):
        assert_equal(got[i], Int64(-(i + 1)), "the negation at " + String(i))
        assert_equal(kept[i], Int64(i + 1), "the input at " + String(i))


def test_a_plus_applied_to_a_column_copies_it() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Apply(0, UnaryOp.POS, "same")))
    var out = pipeline^.run()
    var got = read_back(out, "same")
    for i in range(6):
        assert_equal(got[i], Int64(i + 1), "row " + String(i))


def test_an_apply_over_a_missing_column_is_caught_at_plan_time() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="is outside a schema of 2 columns"):
        pipeline.add(Node(Apply(7, UnaryOp.NEG, "nope")))


def test_an_apply_with_no_answer_on_that_type_is_caught_at_plan_time() raises:
    # Column 1 of `word_frame` is text and there is no negation of a string, so
    # the pipeline refuses to be built rather than raising on the first chunk.
    var pipeline = Pipeline(word_frame())
    with assert_raises():
        pipeline.add(Node(Apply(1, UnaryOp.NEG, "nope")))


def test_a_computed_column_goes_on_the_end_with_the_name_it_was_given() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Compute(0, 0, BinaryOp.ADD, "twice")))
    var out = pipeline^.run()

    assert_equal(out.width(), 3, "the computed column was appended")
    assert_equal(out.schema[2].name, "twice", "the name it was given")
    assert_true(out.schema[2].dtype == LogicalType.INT64, "the result type")
    var got = read_back(out, "twice")
    for i in range(6):
        assert_equal(got[i], Int64(2 * (i + 1)), "row " + String(i))


def test_a_comparison_makes_the_mask_a_filter_then_reads() raises:
    """This is the pair the engine exists to run: a node writes a bool column
    and the next one filters on its position. Neither knows about the other."""
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Compute(0, 0, BinaryOp.ADD, "twice")))
    pipeline.add(Node(Compute(2, 0, BinaryOp.GT, "big")))
    pipeline.add(Node(Filter(3)))
    pipeline.add(Node(Project([0])))
    var out = pipeline^.run()

    # Twice a positive number is greater than the number, so every row survives
    # except none of them, which is the uninteresting half. The interesting half
    # is that the mask was found at position 3 because the plan counted.
    assert_equal(out.width(), 1, "the intermediates were projected away")
    assert_equal(len(out), 6, "rows kept")


def _matched(
    kind: MatchKind, first: String, second: String
) raises -> List[Int64]:
    """Runs one search over `wanted` and reads back the numbers it kept.

    Args:
        kind: Which search to run.
        first: The run of bytes to look for.
        second: The run that has to follow it, empty for the other three.

    Returns:
        The `n` of every row the pattern matched.
    """
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Match(2, Pattern(kind, first, second), "hit")))
    pipeline.add(Node(Filter(3)))
    pipeline.add(Node(Project([0])))
    return read_back(pipeline^.run(), "n")


def test_a_match_appends_a_column_saying_which_rows_it_found() raises:
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Match(2, Pattern(MatchKind.STARTS_WITH, "o", ""), "hit")))
    var out = pipeline^.run()
    assert_equal(out.width(), 4, "the answer was appended")
    assert_equal(out.schema[3].name, "hit", "the name it was given")
    assert_true(out.schema[3].dtype == LogicalType.BOOL, "a yes or no")


def test_each_of_the_four_searches_keeps_a_different_set() raises:
    # `wanted` is ok, ok, ok, fail, fail, no. Four answers that differ is the
    # point: a node that read the pattern and then ran one kernel whatever the
    # pattern said would give the same rows four times.
    var starts = _matched(MatchKind.STARTS_WITH, "o", "")
    assert_equal(len(starts), 3, "starts with o")
    var contains = _matched(MatchKind.CONTAINS, "o", "")
    assert_equal(len(contains), 4, "contains an o")
    var ends = _matched(MatchKind.ENDS_WITH, "o", "")
    assert_equal(len(ends), 1, "ends with o")
    assert_equal(ends[0], Int64(6), "the row that ends with an o")
    var order = _matched(MatchKind.IN_ORDER, "f", "l")
    assert_equal(len(order), 2, "an f and then an l")
    assert_equal(order[0], Int64(4), "the first of them")


def test_a_match_over_a_missing_column_is_caught_at_plan_time() raises:
    var pipeline = Pipeline(word_frame())
    with assert_raises(contains="is outside a schema of 3 columns"):
        pipeline.add(
            Node(Match(9, Pattern(MatchKind.CONTAINS, "o", ""), "nope"))
        )


def test_a_match_over_a_column_that_is_not_text_is_caught() raises:
    var pipeline = Pipeline(word_frame())
    with assert_raises(contains="a LIKE reads text and column 0 holds"):
        pipeline.add(
            Node(Match(0, Pattern(MatchKind.CONTAINS, "o", ""), "nope"))
        )


def test_a_match_on_a_pattern_with_no_wildcard_is_refused() raises:
    # There is a kernel for it and it is the one an equality already uses, so
    # the node says which node to build rather than growing a fifth branch.
    with assert_raises(contains="is an equality against a constant"):
        _ = Match(2, Pattern(MatchKind.EQUALS, "ok", ""), "nope")


def test_a_computed_column_keeps_the_chunk_boundaries() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Compute(0, 0, BinaryOp.MUL, "square")))
    var out = pipeline^.run()
    assert_equal(out.columns[2].num_chunks(), 3, "chunks out")


def test_a_compute_over_a_missing_column_is_caught_when_the_plan_is_built() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="is outside a schema of 2 columns"):
        pipeline.add(Node(Compute(0, 7, BinaryOp.ADD, "nope")))


def test_a_compute_with_no_answer_on_those_types_is_caught_at_plan_time() raises:
    """Column 1 is bool, and subtracting two bools has no answer, so the
    pipeline refuses to be built rather than raising on the first chunk."""
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="use the bitwise_xor"):
        pipeline.add(Node(Compute(1, 1, BinaryOp.SUB, "nope")))


def test_a_cast_changes_a_column_in_place_and_says_so_in_the_schema() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Cast(0, LogicalType.FLOAT64)))
    var out = pipeline^.run()

    assert_equal(out.width(), 2, "the width did not change")
    assert_equal(out.schema[0].name, "n", "the name did not change")
    assert_true(out.schema[0].dtype == LogicalType.FLOAT64, "the new type")
    var col = out.column("n").as_typed[DType.float64]()
    for i in range(6):
        assert_equal(col[i], Float64(i + 1), "row " + String(i))


def test_a_cast_that_was_given_a_name_appends_instead() raises:
    # What SELECT n, CAST(n AS DOUBLE) needs: the column it read is still
    # wanted at the type it had, so the converted one lands beside it.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Cast(0, LogicalType.FLOAT64, "wide")))
    var out = pipeline^.run()

    assert_equal(out.width(), 3, "one column more than went in")
    assert_true(out.schema[0].dtype == LogicalType.INT64, "n as it was")
    assert_equal(out.schema[2].name, "wide", "the name it was given")
    assert_true(out.schema[2].dtype == LogicalType.FLOAT64, "the new type")
    var col = out.column("wide").as_typed[DType.float64]()
    for i in range(6):
        assert_equal(col[i], Float64(i + 1), "row " + String(i))


def test_a_cast_over_a_missing_column_is_caught_when_the_plan_is_built() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="is outside a schema of 2 columns"):
        pipeline.add(Node(Cast(4, LogicalType.FLOAT64)))


def test_a_constant_node_appends_one_value_in_every_row() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Constant(Value(Int64(9)), LogicalType.INT64, "nine")))
    var out = pipeline^.run()

    assert_equal(out.width(), 3, "the constant column was appended")
    assert_equal(out.schema[2].name, "nine", "the name it was given")
    assert_true(
        out.schema[2].dtype == LogicalType.INT64, "the type it was given"
    )
    var col = out.column("nine").as_typed[DType.int64]()
    assert_equal(len(col), 6, "as many rows as the frame")
    for i in range(6):
        assert_equal(col[i], Int64(9), "row " + String(i))


def test_a_constant_node_takes_the_type_it_was_told_rather_than_the_value_s() raises:
    """A constant has no width of its own, so the plan decides. Nine as a
    float64 column is a float64 column and not an integer of the same width."""
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Constant(Value(Int64(9)), LogicalType.FLOAT64, "nine")))
    var out = pipeline^.run()

    assert_true(out.schema[2].dtype == LogicalType.FLOAT64, "the new type")
    var col = out.column("nine").as_typed[DType.float64]()
    for i in range(6):
        assert_equal(col[i], Float64(9), "row " + String(i))


def test_a_constant_node_can_fill_a_text_column() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(
        Node(Constant(Value(String("hi")), LogicalType.STRING, "greeting"))
    )
    var out = pipeline^.run()

    var col = out.column("greeting").as_strings()
    assert_equal(len(col), 6, "as many rows as the frame")
    for i in range(6):
        assert_equal(col[i], "hi", "row " + String(i))


def test_a_constant_node_that_is_null_leaves_every_row_missing() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(
        Node(
            Constant(
                Value(null=LogicalType.INT64), LogicalType.INT64, "nothing"
            )
        )
    )
    var out = pipeline^.run()

    assert_true(out.schema[2].nullable, "the field says it can be missing")
    var col = out.column("nothing").as_typed[DType.int64]()
    for i in range(6):
        assert_false(col.is_valid(i), "row " + String(i))


def test_a_constant_node_after_a_filter_fills_what_survived() raises:
    """The filter cuts six rows to four, and the constant is as long as what
    reached it rather than as long as what the pipeline started with."""
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Constant(Value(Int64(1)), LogicalType.INT64, "one")))
    var out = pipeline^.run()

    assert_equal(len(out), 4, "four rows survived")
    var col = out.column("one").as_typed[DType.int64]()
    for i in range(4):
        assert_equal(col[i], Int64(1), "row " + String(i))


def test_none_of_the_elementwise_nodes_break_a_pipeline() raises:
    """A breaker is where a pipeline is cut, and an operation whose output row
    depends only on its own input row is never one."""
    assert_false(
        node_is_breaker(Node(Compute(0, 0, BinaryOp.ADD, "x"))), "compute"
    )
    assert_false(node_is_breaker(Node(Cast(0, LogicalType.FLOAT64))), "cast")
    assert_true(
        node_status(Node(Cast(0, LogicalType.FLOAT64)))
        == NodeStatus.NEED_MORE_INPUT,
        "a cast always wants more input",
    )


def test_a_constant_is_an_operand_like_a_column_is() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Compute(0, Value(Int64(10)), BinaryOp.MUL, "tens")))
    var out = pipeline^.run()
    assert_equal(out.width(), 3, "the computed column was appended")
    var values = read_back(out, "tens")
    assert_equal(values[0], Int64(10), "row 0")
    assert_equal(values[5], Int64(60), "row 5")


def test_the_predicate_the_engine_exists_for_runs_end_to_end() raises:
    """`n > 3` is the shape of nearly every query anybody writes, and until the
    constant existed it could not be spelled at all."""
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Compute(0, Value(Int64(3)), BinaryOp.GT, "big")))
    pipeline.add(Node(Filter(2)))
    pipeline.add(Node(Project([0])))
    var out = pipeline^.run()
    assert_equal(len(out), 3, "four, five and six survived")
    var values = read_back(out, "n")
    assert_equal(values[0], Int64(4), "the first survivor")


def test_a_constant_on_the_left_of_a_plan_comparison_is_turned_round() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Compute(0, Value(Int64(3)), BinaryOp.GT, "small", True)))
    pipeline.add(Node(Filter(2)))
    pipeline.add(Node(Project([0])))
    var out = pipeline^.run()
    assert_equal(len(out), 2, "3 > n holds for one and two")


def test_a_constant_computes_its_result_type_at_plan_time() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Compute(0, Value(Float64(2.0)), BinaryOp.MUL, "doubled")))
    assert_true(
        pipeline.schema[2].dtype == LogicalType.FLOAT64,
        "an int64 column times a float64 constant is float64",
    )


def test_a_constant_with_no_answer_on_that_type_is_caught_at_plan_time() raises:
    """Dividing a bool column by a bool constant is one of the three pandas
    refuses, and the plan is where the refusal happens.

    The same operation on two bool columns answers a float64, which is pandas'
    own asymmetry rather than one invented here, so this test needs the constant
    form specifically. The one above uses the subtraction, which has no answer
    in either shape.
    """
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="operator 'truediv' not implemented"):
        pipeline.add(Node(Compute(1, Value(True), BinaryOp.DIV, "nope")))


def test_a_text_predicate_runs_end_to_end() raises:
    """A filter on a label is the other half of what a query looks like, and it
    is the last thing the elementwise operator line was waiting on."""
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Compute(1, Value(String("ok")), BinaryOp.EQ, "hit")))
    pipeline.add(Node(Filter(3)))
    pipeline.add(Node(Project([0])))
    var out = pipeline^.run()
    assert_equal(len(out), 4, "four rows say ok")
    var values = read_back(out, "n")
    assert_equal(values[0], Int64(1), "the first survivor")
    assert_equal(values[3], Int64(6), "the last survivor")


def test_two_text_columns_compare_against_each_other() raises:
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Compute(1, 2, BinaryOp.EQ, "agrees")))
    pipeline.add(Node(Filter(3)))
    pipeline.add(Node(Project([0])))
    var out = pipeline^.run()
    assert_equal(len(out), 3, "three rows agree")
    var values = read_back(out, "n")
    assert_equal(values[0], Int64(1), "the first survivor")
    assert_equal(values[2], Int64(5), "the last survivor")


def test_a_text_comparison_is_a_bool_column_at_plan_time() raises:
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Compute(1, 2, BinaryOp.LT, "before")))
    assert_true(
        pipeline.schema[3].dtype == LogicalType.BOOL,
        "a comparison on text answers bool like any other",
    )


def test_arithmetic_on_text_is_caught_at_plan_time() raises:
    var pipeline = Pipeline(word_frame())
    with assert_raises(contains="is not defined on"):
        pipeline.add(Node(Compute(1, 2, BinaryOp.ADD, "nope")))


def counter() raises -> Group:
    """A group by on column 0 that counts the rows in each group."""
    var keys = List[Int]()
    keys.append(0)
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.COUNT, "rows"))
    return Group(keys^, aggs^)


def test_the_elementwise_nodes_are_the_row_local_ones() raises:
    assert_true(node_is_row_local(Node(Filter(0))), "filter")
    assert_true(node_is_row_local(Node(Project([0]))), "project")
    assert_true(
        node_is_row_local(Node(Compute(0, 0, BinaryOp.ADD, "x"))), "compute"
    )
    assert_true(node_is_row_local(Node(Cast(0, LogicalType.INT32))), "cast")
    assert_false(node_is_row_local(Node(Limit(3))), "limit counts rows")
    assert_false(node_is_row_local(Node(counter())), "a group by holds a table")


def test_only_a_limit_can_end_before_its_input_does() raises:
    assert_true(node_ends_early(Node(Limit(3))), "limit")
    assert_false(node_ends_early(Node(Filter(0))), "filter")
    assert_false(
        node_ends_early(Node(counter())),
        "a group by finishes after the source does, not before",
    )


def test_a_stateful_node_refuses_to_be_applied_without_being_mutated() raises:
    """`node_apply` is what several workers call on one shared node, so a node
    that would be racing itself has to be turned away rather than run."""
    var node = Node(Limit(3))
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3]))
    with assert_raises(contains="carries state between chunks"):
        _ = node_apply(node, Chunk(columns^))


def test_the_parallel_prefix_keeps_the_rows_in_the_order_the_source_had_them() raises:
    """Forty chunks through a filter and a projection, which is the shape the
    driver runs on every core. The rows come back in source order or the batch
    boundary lost one."""
    var pipeline = Pipeline(many_chunk_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Project([0])))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(len(got), 134, "two hundred rows less the multiples of three")
    var want = 0
    for i in range(len(got)):
        want += 1
        if want % 3 == 0:
            want += 1
        assert_equal(got[i], Int64(want), "row " + String(i))


def test_the_parallel_prefix_agrees_with_the_frame_methods() raises:
    var pipeline = Pipeline(many_chunk_frame())
    pipeline.add(Node(Compute(0, Value(Int64(10)), BinaryOp.MUL, "ten")))
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Project([2])))
    var out = pipeline^.run()
    var got = read_back(out, "ten")

    var whole = many_chunk_frame()
    var mask = whole.column("keep").as_typed[DType.bool]()
    var direct = whole.filter(mask)
    assert_equal(len(got), len(direct), "the same number of rows")
    var expected = read_back(direct, "n")
    for i in range(len(got)):
        assert_equal(got[i], expected[i] * 10, "row " + String(i))


def test_a_breaker_after_the_parallel_prefix_sees_every_row() raises:
    """The prefix runs on every core and the breaker after it does not, so what
    is being checked is that the hand off between the two loses nothing."""
    var pipeline = Pipeline(many_chunk_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Project([0])))
    pipeline.add(Node(counter()))
    var out = pipeline^.run()
    assert_equal(len(out), 134, "one group per surviving row")


def test_a_limit_over_many_chunks_still_reads_one_chunk() raises:
    """The parallel route reads a batch ahead, which is the wrong thing to do
    when a limit is going to throw most of the batch away, so a pipeline holding
    one is fed a chunk at a time. The chunk count of the answer is what says so:
    three rows out of chunks of five is one chunk, cut."""
    var pipeline = Pipeline(many_chunk_frame())
    pipeline.add(Node(Limit(3)))
    var out = pipeline^.run()
    assert_equal(len(out), 3, "rows")
    assert_equal(out.columns[0].num_chunks(), 1, "chunks read")


def test_only_the_operators_that_do_arithmetic_are_worth_a_task() raises:
    """A task costs about as much to create as a projection costs to run, so the
    driver asks what the prefix computes before it hands it out."""
    assert_true(node_computes_per_row(Node(Filter(0))), "a predicate per row")
    assert_true(
        node_computes_per_row(Node(Compute(0, 0, BinaryOp.ADD, "x"))),
        "an expression per row",
    )
    assert_false(
        node_computes_per_row(Node(Project([0]))),
        "a projection rebuilds a chunk out of columns it already has",
    )
    assert_false(
        node_computes_per_row(Node(Cast(0, LogicalType.INT32))),
        "a cast waits on the allocator, not on arithmetic",
    )
    assert_false(node_computes_per_row(Node(Limit(3))), "limit")
    assert_false(node_computes_per_row(Node(counter())), "group by")


def test_a_projection_on_its_own_still_returns_every_row_in_order() raises:
    """Nothing in this pipeline is worth a task, so it runs on the calling
    thread. What it returns has to be the same either way."""
    var pipeline = Pipeline(many_chunk_frame())
    pipeline.add(Node(Project([0])))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(len(got), 200, "every row of every chunk")
    for i in range(len(got)):
        assert_equal(got[i], Int64(i + 1), "row " + String(i))


def totals() raises -> Reduce:
    """A reduction over column 0 that asks for every kind that folds."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.SUM, "total"))
    aggs.append(GroupAgg(0, AggKind.MIN, "low"))
    aggs.append(GroupAgg(0, AggKind.MAX, "high"))
    aggs.append(GroupAgg(0, AggKind.COUNT, "seen"))
    return Reduce(aggs^)


def one_int(df: DataFrame, name: String) raises -> Int64:
    """Reads the single row of an int64 output column."""
    return df.column(name).as_typed[DType.int64]()[0]


def test_a_reduction_over_chunks_gives_the_whole_frame_answer() raises:
    """Six rows arriving in chunks of two, three and one. Every one of these is
    the answer `agg` gives over the same frame in one piece."""
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(totals()))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    assert_equal(one_int(out, "total"), 21, "1 through 6")
    assert_equal(one_int(out, "low"), 1, "the smallest, from the first chunk")
    assert_equal(one_int(out, "high"), 6, "the largest, from the last chunk")
    assert_equal(one_int(out, "seen"), 6, "every row, counted once")


def test_a_mean_over_uneven_chunks_is_not_a_mean_of_means() raises:
    """The chunks are two, three and one row long, so the means of the chunks
    are 1.5, 4 and 6 and averaging those gives 3.833. The answer is 3.5, which
    is what keeping a sum and a count apart until the last moment buys."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.MEAN, "average"))
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    var got = out.column("average").as_typed[DType.float64]()[0]
    assert_equal(got, Float64(3.5), "the mean of one through six")


def big_frame() raises -> DataFrame:
    """Six values near 1.9e18, in chunks of two, three and one.

    Their total is about 1.14e19 and int64 stops at 9.22e18, so a sum over this
    column wraps, and it is supposed to: pandas wraps there and firepanda
    follows it. What must not wrap is the sum a mean is divided out of, which is
    why the values are this size and why the chunks are uneven, so that the
    wrap has to survive the merge as well as the chunk.
    """
    var base = Int64(1_900_000_000_000_000_000)
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([base, base + 2]))
    n.append(numbers([base + 4, base + 6, base + 8]))
    n.append(numbers([base + 10]))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def test_a_mean_of_large_ints_is_not_divided_out_of_a_wrapped_sum() raises:
    """The mean here is 1.9e18 plus five and the wrapped sum gives -1.17e18.

    Not a near miss in the last bits: the wrong answer is negative and there is
    no negative value in the column. The operator keeps a running sum and a
    running count so that the state folds, and that sum is a numerator rather
    than a sum anybody asked for, so it accumulates in float64. See #673.
    """
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.MEAN, "average"))
    var pipeline = Pipeline(big_frame())
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    var got = out.column("average").as_typed[DType.float64]()[0]
    var want = Float64(1_900_000_000_000_000_005)
    assert_true(got > 0.0, "no value in the column is negative")
    var gap = got - want
    if gap < 0.0:
        gap = -gap
    assert_true(gap <= 1e-9 * want, "the mean of the six values")


def test_a_sum_of_large_ints_still_wraps() raises:
    """The other half of the same rule, and the reason the fix is a flag on one
    slot rather than a wider accumulator everywhere. A sum over int64 wraps in
    pandas, so it wraps here, and the mean above is not allowed to change that.
    """
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.SUM, "total"))
    var pipeline = Pipeline(big_frame())
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    assert_true(one_int(out, "total") < 0, "six times 1.9e18 does not fit")


def kept_nothing() raises -> DataFrame:
    """Three rows behind a mask that keeps none of them."""
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2, 3]))
    var keep = ChunkedArray(LogicalType.BOOL)
    keep.append(flags([False, False, False]))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(keep^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.BOOL))
    return DataFrame(Schema(fields^), columns^)


def test_a_reduction_over_an_input_that_kept_nothing_is_one_row() raises:
    """A fold with no key is one group whether or not anything was read, so it
    hands out one row rather than none. What is in it is what `agg` answers over
    an empty column: a zero for the count and for the sum, and a null for the
    two extremes, which found nothing to be the extreme of."""
    var pipeline = Pipeline(kept_nothing())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(totals()))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    assert_equal(one_int(out, "seen"), 0, "nothing to count")
    assert_equal(one_int(out, "total"), 0, "the sum of nothing")
    var low = out.column("low").as_typed[DType.int64]()
    assert_false(low.is_valid(0), "the smallest of nothing")
    var high = out.column("high").as_typed[DType.int64]()
    assert_false(high.is_valid(0), "the largest of nothing")


def test_a_mean_over_an_input_that_kept_nothing_is_null() raises:
    """The mean is the one fold whose state is two slots, so the empty answer
    goes through the same division the full one does: a sum of zero over a count
    of zero, which `_mean_of` calls null rather than dividing. That is the same
    answer a group with nothing but nulls in it gets."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.MEAN, "average"))
    var pipeline = Pipeline(kept_nothing())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    var got = out.column("average").as_typed[DType.float64]()
    assert_false(got.is_valid(0), "the mean of nothing")


def test_a_held_column_over_an_input_that_kept_nothing_answers_anyway() raises:
    """A reduction whose state is the values held no chunks at all here, so what
    it flattens is a column of no rows and the kernel answers over that. Nothing
    about the empty case is written down in the operator, which is the point of
    the test: the answer is the kernel's and the two folds above get theirs the
    same way.

    A median of nothing is a NaN that is still valid rather than a null, which
    is `_quantile_core`'s own rule and pandas' as well, where a float column
    carries a missing value as a NaN. It disagrees with the null a minimum of
    nothing gets and that disagreement is #170's, not this node's."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.MEDIAN, "middle"))
    var pipeline = Pipeline(kept_nothing())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    var got = out.column("middle").as_typed[DType.float64]()
    assert_true(got.is_valid(0), "valid, the way pandas would have it")
    assert_true(isnan(got[0]), "and not a number")


def test_a_reduction_reduces_what_reached_it() raises:
    """Two hundred rows through a filter that drops every third one. The
    reduction is at the end of a pipeline whose front runs on every core, so
    what this checks is that nothing was folded twice and nothing was lost."""
    var pipeline = Pipeline(many_chunk_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(totals()))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")

    var whole = many_chunk_frame()
    var mask = whole.column("keep").as_typed[DType.bool]()
    var direct = whole.filter(mask)
    var want = Int64(0)
    var values = read_back(direct, "n")
    for i in range(len(values)):
        want += values[i]
    assert_equal(one_int(out, "seen"), Int64(len(values)), "rows kept")
    assert_equal(one_int(out, "total"), want, "the sum of the rows kept")
    assert_equal(one_int(out, "low"), 1, "the first row survives the mask")
    assert_equal(one_int(out, "high"), 200, "so does the last")


def test_a_reduction_holds_everything_until_the_input_is_done() raises:
    """It is a breaker, it is not row local, and it is not worth a task, so the
    driver treats it the way it treats a group by."""
    assert_true(node_is_breaker(Node(totals())), "nothing comes out early")
    assert_false(node_is_row_local(Node(totals())), "it holds a running answer")
    assert_false(node_ends_early(Node(totals())), "it needs the last row")
    assert_false(node_computes_per_row(Node(totals())), "not a prefix operator")
    assert_equal(
        node_status(Node(totals())),
        NodeStatus.NEED_MORE_INPUT,
        "before anything has arrived",
    )


def repeat_frame() raises -> DataFrame:
    """Six rows in chunks of two, three and one, with repeats across the joins.

    The 2 is in the first chunk and the second, and the 1 is in the first and
    the second as well, so a distinct count that added the chunks up would say
    eight where the answer is four. That is the case a held column exists for
    and a per chunk partial cannot see.
    """
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2]))
    n.append(numbers([2, 3, 1]))
    n.append(numbers([4]))
    var keep = ChunkedArray(LogicalType.BOOL)
    keep.append(flags([True, True]))
    keep.append(flags([True, True, True]))
    keep.append(flags([True]))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(keep^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.BOOL))
    return DataFrame(Schema(fields^), columns^)


def test_a_median_over_chunks_is_the_median_of_the_whole_input() raises:
    """A median of medians is not a median, so the column is held and the
    kernel runs once over all of it. Six rows, so it is the middle pair."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.MEDIAN, "middle"))
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    var got = out.column("middle").as_typed[DType.float64]()[0]
    assert_equal(got, Float64(3.5), "between the third and the fourth")


def test_a_distinct_count_counts_a_value_in_two_chunks_once() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.NUNIQUE, "distinct"))
    aggs.append(GroupAgg(0, AggKind.COUNT, "seen"))
    var pipeline = Pipeline(repeat_frame())
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    assert_equal(one_int(out, "seen"), 6, "six rows arrived")
    assert_equal(one_int(out, "distinct"), 4, "1, 2, 3 and 4")


def test_a_fold_and_a_hold_in_one_reduction_both_answer() raises:
    """The sum is folded a chunk at a time and the median holds the column, and
    the two run beside each other over the same input."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.SUM, "total"))
    aggs.append(GroupAgg(0, AggKind.MEDIAN, "middle"))
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    assert_equal(one_int(out, "total"), 21, "1 through 6")
    var got = out.column("middle").as_typed[DType.float64]()[0]
    assert_equal(got, Float64(3.5), "the median of the same six")


def test_a_held_column_survives_the_parallel_route() raises:
    """Two hundred rows in forty chunks through a filter, which is the shape
    that runs the front of the pipeline on every core and hands the reduction
    one partial per chunk. A held column rides in those partials, so this is
    the check that none of them was dropped or counted twice."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.NUNIQUE, "distinct"))
    var pipeline = Pipeline(many_chunk_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")

    var whole = many_chunk_frame()
    var mask = whole.column("keep").as_typed[DType.bool]()
    var kept = len(read_back(whole.filter(mask), "n"))
    # The numbers are consecutive and the mask keeps whole rows, so every row
    # that survives holds a value no other row holds.
    assert_equal(one_int(out, "distinct"), Int64(kept), "all of them differ")


def test_a_reduction_carrying_an_operation_runs_it_a_chunk_at_a_time() raises:
    """The node level of what the lowering builds for a whole frame reduction
    over a column and a constant. The frame arrives in chunks of two, three and
    one, and the operation has to run on each of them, so an answer that is
    right here is an answer the chunking did not change."""
    var aggs = List[GroupAgg]()
    aggs.append(
        GroupAgg(0, AggKind.SUM, "total", BinaryOp.ADD, Value(Int64(10)))
    )
    aggs.append(GroupAgg(0, AggKind.MAX, "high", BinaryOp.MUL, Value(Int64(2))))
    aggs.append(
        GroupAgg(0, AggKind.MIN, "low", BinaryOp.SUB, Value(Int64(10)), True)
    )
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    assert_equal(one_int(out, "total"), 81, "one through six and six tens")
    assert_equal(one_int(out, "high"), 12, "twice the largest")
    assert_equal(one_int(out, "low"), 4, "ten less the largest")


def test_a_group_by_refuses_a_reduction_carrying_an_operation() raises:
    """Only a reduction folds one in, because only a reduction reads the whole
    column. A group by scatters its rows and the operation would have to go with
    them, which is the `Compute` the lowering puts in front of it."""
    var aggs = List[GroupAgg]()
    aggs.append(
        GroupAgg(0, AggKind.SUM, "total", BinaryOp.ADD, Value(Int64(1)))
    )
    var keys = List[Int]()
    keys.append(1)
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="cannot carry an operation"):
        pipeline.add(Node(Group(keys^, aggs^)))


def test_a_reduction_that_reads_two_columns_is_refused() raises:
    """A correlation wants a pair and a `GroupAgg` names one column, so there
    is nothing for the second one to be."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.CORR, "together"))
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="reads two columns"):
        pipeline.add(Node(Reduce(aggs^)))


def test_a_reduction_with_nothing_to_reduce_is_an_error() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="at least one aggregate"):
        pipeline.add(Node(Reduce(List[GroupAgg]())))


def test_two_reductions_cannot_be_given_the_same_name() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.SUM, "answer"))
    aggs.append(GroupAgg(0, AggKind.MIN, "answer"))
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="would both be called answer"):
        pipeline.add(Node(Reduce(aggs^)))


def test_a_reduction_of_a_column_that_is_not_there_is_an_error() raises:
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(7, AggKind.SUM, "total"))
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="outside a schema"):
        pipeline.add(Node(Reduce(aggs^)))


def test_a_sum_of_a_column_of_names_is_an_error() raises:
    """A minimum of a column of names means something and a sum does not."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(1, AggKind.SUM, "total"))
    var pipeline = Pipeline(word_frame())
    with assert_raises(contains="not defined on text"):
        pipeline.add(Node(Reduce(aggs^)))


def lookup_frame() raises -> DataFrame:
    """Four rows to join against, keyed on `n`.

    The keys are 2, 4, 6 and 8, so of the six rows in `cut_frame` three match
    and three do not, and the ones that do are not a prefix. Key 8 matches
    nothing on the other side, which is what an outer join would have to notice
    and this node does not.
    """
    var columns = List[AnyArray]()
    columns.append(numbers([2, 4, 6, 8]))
    columns.append(numbers([20, 40, 60, 80]))
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("tag", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def key_words() raises -> AnyArray:
    """The six left keys, as text, for the joins on a text key.

    Three of them are past twelve bytes and three are not, which is the split
    that matters here and not anywhere else in this file. A view of twelve bytes
    or fewer holds its bytes inside itself and a longer one holds an offset into
    a payload, so a comparison against a long key has to reach into the column
    the table was built from and a comparison against a short one never does.
    Only the long ones would notice a probe reading its own payload instead.

    The keys are laid out so that the answer is the same set the integer joins in
    this file give, which is rows 2, 4 and 6. Two of those match on a long key
    and one on a short one, and of the three that match nothing, one is long.
    """
    return AnyArray(
        strings_from_list(
            [
                "k1",
                "a-key-well-past-twelve-2",
                "k3",
                "a-key-well-past-twelve-4",
                "a-key-well-past-twelve-5",
                "k6",
            ]
        )
    )


def key_fields() raises -> List[Field]:
    """The schema the text keyed frames share."""
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("key", LogicalType.STRING))
    return fields^


def word_key_frame() raises -> DataFrame:
    """The six rows in one chunk, keyed on text."""
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(key_words())
    return DataFrame(Schema(key_fields()), columns^)


def cut_word_key_frame() raises -> DataFrame:
    """The same six rows in chunks of two, three and one.

    The cuts fall where they fall in `cut_frame`, so the chunk that holds the
    long key that matches nothing is not the chunk that holds the short key that
    does.
    """
    var whole = key_words()
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2]))
    n.append(numbers([3, 4, 5]))
    n.append(numbers([6]))
    var key = ChunkedArray(LogicalType.STRING)
    key.append(AnyArray(whole.strings().slice(0, 2)))
    key.append(AnyArray(whole.strings().slice(2, 5)))
    key.append(AnyArray(whole.strings().slice(5, 6)))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(key^)
    return DataFrame(Schema(key_fields()), columns^)


def word_lookup_frame() raises -> DataFrame:
    """Four rows to join against, keyed on text.

    `lookup_frame` with the keys written out. The fourth is a long key that
    matches nothing on the other side, which is the row an outer join would have
    to notice and this node does not.
    """
    var columns = List[AnyArray]()
    columns.append(
        AnyArray(
            strings_from_list(
                [
                    "a-key-well-past-twelve-2",
                    "a-key-well-past-twelve-4",
                    "k6",
                    "a-key-well-past-twelve-8",
                ]
            )
        )
    )
    columns.append(numbers([20, 40, 60, 80]))
    var fields = List[Field]()
    fields.append(Field("key", LogicalType.STRING))
    fields.append(Field("tag", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def byte_lookup_frame() raises -> DataFrame:
    """A lookup frame whose `key` is a column of bytes rather than of text.

    A string column's physical dtype is uint8, so this is the frame that agrees
    with a text keyed one on everything a dtype comparison can see and means
    something completely different.
    """
    var col = Array[DType.uint8](4)
    for i in range(4):
        col.set_valid(i, UInt8(i + 1))
    var columns = List[AnyArray]()
    columns.append(AnyArray(col^))
    columns.append(numbers([20, 40, 60, 80]))
    var fields = List[Field]()
    fields.append(Field("key", LogicalType.UINT8))
    fields.append(Field("tag", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def joined_rows(var pipeline: Pipeline) raises -> List[Int64]:
    """Runs a pipeline and reads its `n` column back, sorted.

    A join emits a chunk per chunk and the driver may hand the chunks to
    different cores, so the order rows come back in is the order the chunks
    finished in. Every test here compares sets, and sorting is how a list is
    compared as a set.
    """
    var out = pipeline^.run()
    var values = read_back(out, "n")
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[j] < values[i]:
                var swap = values[i]
                values[i] = values[j]
                values[j] = swap
    return values^


def test_a_join_in_a_pipeline_gives_what_the_frame_join_gives() raises:
    var whole = sample_frame().join_on(
        lookup_frame(), ["n"], ["n"], JoinKind.INNER
    )
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Join(lookup_frame(), "n", "n", JoinKind.INNER)))
    assert_equal(
        String(pipeline.schema), String(whole.schema), "the same output schema"
    )
    var values = joined_rows(pipeline^)
    assert_equal(len(values), len(whole), "the same number of rows")
    assert_equal(values[0], Int64(2), "the first key that matched")
    assert_equal(values[1], Int64(4), "the second")
    assert_equal(values[2], Int64(6), "the third")


def test_a_join_over_chunks_agrees_with_a_join_over_one_chunk() raises:
    """The whole claim of a streaming join: what came out of the six chunks is
    what came out of the one."""
    var one = Pipeline(sample_frame())
    one.add(Node(Join(lookup_frame(), "n", "n", JoinKind.INNER)))
    var many = Pipeline(cut_frame())
    many.add(Node(Join(lookup_frame(), "n", "n", JoinKind.INNER)))
    var a = joined_rows(one^)
    var b = joined_rows(many^)
    assert_equal(len(a), len(b), "the same height")
    for i in range(len(a)):
        assert_equal(a[i], b[i], "row " + String(i))


def test_a_join_over_forty_chunks_matches_the_whole_frame_join() raises:
    """Forty chunks is more than one batch, so this is the join run on every
    core at once rather than one chunk after another. The chunks are five rows
    each and the four keys land in the first two of them, so most workers are
    handed a chunk that pairs with nothing."""
    var pipeline = Pipeline(many_chunk_frame())
    pipeline.add(Node(Join(lookup_frame(), "n", "n", JoinKind.INNER)))
    var values = joined_rows(pipeline^)
    assert_equal(len(values), 4, "the four keys the lookup has")
    assert_equal(values[0], Int64(2), "the smallest key that matched")
    assert_equal(values[3], Int64(8), "the largest")


def test_a_left_join_keeps_a_row_that_matched_nothing() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Join(lookup_frame(), "n", "n", JoinKind.LEFT)))
    var values = joined_rows(pipeline^)
    assert_equal(len(values), 6, "every left row survived")
    assert_equal(values[0], Int64(1), "the row that matched nothing is here")


def test_a_semi_join_keeps_the_left_columns_and_nothing_else() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Join(lookup_frame(), "n", "n", JoinKind.SEMI)))
    assert_equal(len(pipeline.schema), 2, "no column came from the right")
    var values = joined_rows(pipeline^)
    assert_equal(len(values), 3, "the three keys that matched")
    assert_equal(values[0], Int64(2), "the first of them")


def test_an_anti_join_keeps_the_rows_that_matched_nothing() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Join(lookup_frame(), "n", "n", JoinKind.ANTI)))
    var values = joined_rows(pipeline^)
    assert_equal(len(values), 3, "the three keys that missed")
    assert_equal(values[0], Int64(1), "the first of them")
    assert_equal(values[2], Int64(5), "the last of them")


def test_a_join_builds_only_the_columns_it_was_asked_for() raises:
    """Gathering is most of what a join costs, so a column that is going to be
    dropped is not a small waste at the end, it is most of the work."""
    var pipeline = Pipeline(cut_frame())
    pipeline.add(
        Node(
            Join(
                lookup_frame(),
                "n",
                "n",
                JoinKind.INNER,
                "_right",
                ["tag", "n"],
            )
        )
    )
    assert_equal(len(pipeline.schema), 2, "two of the four")
    assert_equal(pipeline.schema[0].name, "tag", "in the order asked for")
    assert_equal(pipeline.schema[1].name, "n", "and not the natural one")
    var values = joined_rows(pipeline^)
    assert_equal(len(values), 3, "the three keys that matched")


def test_a_join_feeding_a_reduction_never_writes_its_output() raises:
    """The query the node exists for. What the join emits is folded away by the
    reduction while it is still in cache, so the intermediate that a whole frame
    join would have written is never written at all."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(2, AggKind.SUM, "total"))
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Join(lookup_frame(), "n", "n", JoinKind.INNER)))
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row came out")
    var total = out.column("total").as_typed[DType.int64]()
    assert_equal(total[0], Int64(120), "twenty and forty and sixty")


def test_a_join_is_row_local_and_does_not_break_the_pipeline() raises:
    """Its table is finished before the first chunk arrives and only read
    afterwards, so every core can be handed the same node."""
    var node = Node(Join(lookup_frame(), "n", "n", JoinKind.INNER))
    assert_true(node_is_row_local(node), "row local")
    assert_false(node_is_breaker(node), "not a breaker")
    assert_true(node_computes_per_row(node), "worth spreading over the cores")
    assert_false(node_ends_early(node), "it reads its whole input")
    assert_true(
        node_status(node) == NodeStatus.NEED_MORE_INPUT, "always wants more"
    )


def test_a_join_that_matched_nothing_in_a_chunk_drops_the_chunk() raises:
    """Every key in this lookup misses, so every chunk emits nothing and the
    sink still knows the shape of what it did not see."""
    var far = List[Int64]()
    far.append(Int64(100))
    far.append(Int64(200))
    var columns = List[AnyArray]()
    columns.append(numbers(far))
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    var pipeline = Pipeline(cut_frame())
    pipeline.add(
        Node(
            Join(
                DataFrame(Schema(fields^), columns^),
                "n",
                "n",
                JoinKind.INNER,
            )
        )
    )
    var out = pipeline^.run()
    assert_equal(len(out), 0, "nothing paired")
    assert_equal(out.width(), 2, "and the schema survived anyway")


def test_a_null_key_matches_nothing_on_either_side() raises:
    var keys = Array[DType.int64](3)
    keys.set_valid(0, Int64(2))
    keys.set_null(1)
    keys.set_valid(2, Int64(4))
    var columns = List[AnyArray]()
    columns.append(AnyArray(keys^))
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64, True))
    var pipeline = Pipeline(cut_frame())
    pipeline.add(
        Node(
            Join(
                DataFrame(Schema(fields^), columns^),
                "n",
                "n",
                JoinKind.INNER,
            )
        )
    )
    var values = joined_rows(pipeline^)
    assert_equal(len(values), 2, "two matched and the null matched nothing")
    assert_equal(values[0], Int64(2), "the first")
    assert_equal(values[1], Int64(4), "the second")


def test_a_join_told_its_columns_by_position_renames_nothing() raises:
    """Both frames call their key `n`, so left to itself the node drops the
    right one, which moves every column a caller had numbered. Asked for the two
    schemas end to end it writes them out as they are."""
    var pipeline = Pipeline(cut_frame())
    pipeline.add(
        Node(
            Join(
                lookup_frame(),
                "n",
                "n",
                JoinKind.INNER,
                "_right",
                List[String](),
                [0, 1, 2, 3],
            )
        )
    )
    assert_equal(len(pipeline.schema), 4, "both keys survived")
    assert_equal(pipeline.schema[0].name, "n", "the probe side's")
    assert_equal(pipeline.schema[1].name, "keep")
    assert_equal(pipeline.schema[2].name, "n", "the build side's, unrenamed")
    assert_equal(pipeline.schema[3].name, "tag")
    var out = pipeline^.run()
    assert_equal(len(out), 3, "the three keys that matched")


def test_a_join_told_where_its_key_is_does_not_go_by_name() raises:
    """A name finds the first column that has it, and the caller here means the
    second. Both are called `n` and they hold different numbers, so a join that
    looked the name up would pair the wrong rows rather than fail."""
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(numbers([2, 4, 6, 8, 10, 12]))
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("n", LogicalType.INT64))
    var pipeline = Pipeline(DataFrame(Schema(fields^), columns^))
    pipeline.add(
        Node(
            Join(
                lookup_frame(),
                "n",
                "n",
                JoinKind.INNER,
                "_right",
                List[String](),
                [0],
                1,
                0,
            )
        )
    )
    # All four keys match against the second column and only three match
    # against the first, so the count alone says which one it read.
    var values = joined_rows(pipeline^)
    assert_equal(len(values), 4, "the second column matched every key")
    assert_equal(values[0], Int64(1), "and these are the rows it was")
    assert_equal(values[1], Int64(2))
    assert_equal(values[2], Int64(3))
    assert_equal(values[3], Int64(4))


def test_a_join_asked_for_a_column_neither_side_has_says_so() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="the two sides have 4 between them"):
        pipeline.add(
            Node(
                Join(
                    lookup_frame(),
                    "n",
                    "n",
                    JoinKind.INNER,
                    "_right",
                    List[String](),
                    [0, 9],
                )
            )
        )


def test_an_outer_join_in_a_pipeline_is_refused() raises:
    """It has to emit right rows nothing matched, which is not known until the
    last chunk, so it is a breaker wearing this node's clothes."""
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="not known until the last chunk"):
        pipeline.add(Node(Join(lookup_frame(), "n", "n", JoinKind.OUTER)))


def test_a_right_join_in_a_pipeline_is_refused() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="not known until the last chunk"):
        pipeline.add(Node(Join(lookup_frame(), "n", "n", JoinKind.RIGHT)))


def test_a_join_on_a_text_key_gives_what_the_frame_join_gives() raises:
    """The refusal this replaces said a text key needs both sides concatenated.

    It does not. It needs a table that compares the bytes when two hashes agree,
    and once it has one it is built from one side and read from the other like
    any other key.
    """
    var whole = word_key_frame().join_on(
        word_lookup_frame(), ["key"], ["key"], JoinKind.INNER
    )
    var pipeline = Pipeline(cut_word_key_frame())
    pipeline.add(Node(Join(word_lookup_frame(), "key", "key", JoinKind.INNER)))
    assert_equal(
        String(pipeline.schema), String(whole.schema), "the same output schema"
    )
    var values = joined_rows(pipeline^)
    assert_equal(len(values), len(whole), "the same number of rows")
    assert_equal(values[0], Int64(2), "matched on a long key")
    assert_equal(values[1], Int64(4), "matched on a long key")
    assert_equal(values[2], Int64(6), "matched on a key held inline")


def test_a_text_join_over_chunks_agrees_with_one_over_one_chunk() raises:
    """The same claim the integer joins make, on the key that could not make it.

    Worth its own test rather than being folded into the one above, because the
    thing that would break it is per chunk: the probe compares a probe row's
    bytes against a view that points into a column it is not reading, and a chunk
    that read its own payload would answer differently from the whole frame
    without answering nonsense.
    """
    var one = Pipeline(word_key_frame())
    one.add(Node(Join(word_lookup_frame(), "key", "key", JoinKind.INNER)))
    var many = Pipeline(cut_word_key_frame())
    many.add(Node(Join(word_lookup_frame(), "key", "key", JoinKind.INNER)))
    assert_equal(
        String(joined_rows(one^)), String(joined_rows(many^)), "the same rows"
    )


def test_a_left_join_on_a_text_key_keeps_a_row_that_matched_nothing() raises:
    var pipeline = Pipeline(cut_word_key_frame())
    pipeline.add(Node(Join(word_lookup_frame(), "key", "key", JoinKind.LEFT)))
    var values = joined_rows(pipeline^)
    assert_equal(len(values), 6, "every left row, matched or not")


def test_a_join_on_keys_of_different_dtypes_is_refused() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="the same dtype on each side"):
        pipeline.add(Node(Join(lookup_frame(), "keep", "n")))


def test_a_join_of_a_text_key_against_a_byte_key_is_refused() raises:
    """The one pair of keys whose physical dtypes agree and whose meanings do
    not. A string column is laid out as bytes, so without the string test beside
    the dtype test this would build a table over the first byte of each view."""
    var pipeline = Pipeline(cut_word_key_frame())
    with assert_raises(contains="the same dtype on each side"):
        pipeline.add(Node(Join(byte_lookup_frame(), "key", "key")))


def test_a_join_asked_for_a_column_the_result_has_not_got_is_refused() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="no column 'nope' to keep"):
        pipeline.add(
            Node(
                Join(
                    lookup_frame(),
                    "n",
                    "n",
                    JoinKind.INNER,
                    "_right",
                    ["nope"],
                )
            )
        )


def test_a_right_column_that_collides_is_suffixed() raises:
    """`keep` is on both sides here, and the key names differ, so the shared
    name is the one that has to move out of the way."""
    var columns = List[AnyArray]()
    columns.append(numbers([2, 4]))
    columns.append(numbers([20, 40]))
    var fields = List[Field]()
    fields.append(Field("id", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.INT64))
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Join(DataFrame(Schema(fields^), columns^), "n", "id")))
    assert_equal(len(pipeline.schema), 4, "both keys and both payloads")
    assert_equal(pipeline.schema[2].name, "id", "the right key kept its name")
    assert_equal(pipeline.schema[3].name, "keep_right", "and this one moved")


def test_a_mean_folded_on_every_core_is_not_a_mean_of_means() raises:
    """A mean is a sum and a count kept apart, so it is two columns of the
    partial row a worker hands back, and this is the test that says both of them
    survive the trip."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.MEAN, "average"))
    var pipeline = Pipeline(many_chunk_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()

    var whole = many_chunk_frame()
    var mask = whole.column("keep").as_typed[DType.bool]()
    var values = read_back(whole.filter(mask), "n")
    var total = Int64(0)
    for i in range(len(values)):
        total += values[i]
    var want = Float64(total) / Float64(len(values))
    var got = out.column("average").as_typed[DType.float64]()[0]
    assert_equal(got, want, "the mean of what got through the filter")


def test_folding_on_every_core_gives_what_folding_on_one_gives() raises:
    """The same query over the same rows in one chunk and in forty. One chunk
    is fewer than the driver spreads out, so the first runs the fold on this
    thread and the second runs it on every core."""
    var one = Pipeline(sample_frame())
    one.add(Node(Filter(1)))
    one.add(Node(totals()))
    var here = one^.run()

    var many = Pipeline(many_chunk_frame())
    many.add(Node(Filter(1)))
    many.add(Node(totals()))
    var there = many^.run()

    assert_equal(one_int(here, "low"), one_int(there, "low"), "the smallest")
    assert_equal(one_int(here, "seen"), 4, "the four rows the mask keeps")
    assert_equal(one_int(there, "seen"), 134, "the rows the other mask keeps")


def test_a_reduction_behind_a_join_folds_on_the_core_that_probed() raises:
    """The query the whole line of work exists for. The join is the parallel
    prefix and the reduction is immediately behind it, so each chunk is probed,
    gathered and folded away on one core without going to memory in between."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(2, AggKind.SUM, "tag_total"))
    aggs.append(GroupAgg(2, AggKind.COUNT, "paired"))
    var pipeline = Pipeline(many_chunk_frame())
    pipeline.add(Node(Join(lookup_frame(), "n", "n", JoinKind.INNER)))
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    assert_equal(one_int(out, "paired"), 4, "the four keys that matched")
    assert_equal(
        one_int(out, "tag_total"), 200, "twenty and forty and sixty and eighty"
    )


def test_a_partial_row_of_the_wrong_width_is_refused() raises:
    """A mean makes two columns of partial answers out of one aggregation, so a
    row that has as many columns as the caller asked for aggregations is the
    wrong row."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.MEAN, "average"))
    var node = Reduce(aggs^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    _ = node.bind(Schema(fields^))
    var columns = List[AnyArray]()
    columns.append(numbers([21]))
    with assert_raises(contains="this reduction produces"):
        node.absorb(Chunk(columns^))


def ints_of(col: AnyArray, rows: Int) raises -> List[Int64]:
    """Reads a whole int64 column out, for comparing against a literal list."""
    ref view = col.as_typed_view[DType.int64]()
    var out = List[Int64](capacity=rows)
    for i in range(rows):
        out.append(view[i])
    return out^


def selected_chunk() raises -> Chunk:
    """Six rows of values with a selection keeping the second and the fourth.

    Column 0 is the six values and is not dense, so its rows are at positions 1
    and 3. Column 1 is two values and is dense, which is what a column computed
    after the selection was made looks like. The two together are the only shape
    that matters, since a chunk whose columns are all one or all the other is
    handled by the same loop.
    """
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(numbers([70, 80]))
    var picks = List[Int]()
    picks.append(1)
    picks.append(3)
    var dense = List[Bool]()
    dense.append(False)
    dense.append(True)
    return Chunk(columns^, picks^, dense^)


def test_a_chunk_built_from_columns_carries_no_selection() raises:
    """Every chunk in the engine before selections existed, and most of them
    after. The row count comes from the columns and nothing is read through
    anything."""
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3]))
    var chunk = Chunk(columns^)
    assert_false(chunk.selected(), "no selection")
    assert_equal(len(chunk), 3, "three rows")


def test_a_selected_chunk_has_a_row_per_position() raises:
    """The row count is the length of the selection and not the length of any
    column, which is the whole point: the columns are longer."""
    var chunk = selected_chunk()
    assert_true(chunk.selected(), "a selection")
    assert_equal(len(chunk), 2, "two rows, though a column holds six")
    assert_equal(chunk.width(), 2, "two columns")


def test_flattening_gathers_what_is_not_dense_and_leaves_what_is() raises:
    """The first column is read through the selection and the second is already
    at the chunk's rows, so flattening has to gather one and copy neither."""
    var chunk = selected_chunk()
    chunk.flatten()
    assert_false(chunk.selected(), "the selection is gone")
    assert_equal(len(chunk), 2, "still two rows")
    var first = ints_of(chunk.columns[0], 2)
    assert_equal(first[0], 2, "position 1 of 1 through 6")
    assert_equal(first[1], 4, "position 3 of 1 through 6")
    var second = ints_of(chunk.columns[1], 2)
    assert_equal(second[0], 70, "the dense column, untouched")
    assert_equal(second[1], 80, "and its second row")


def test_flattening_twice_is_the_same_as_flattening_once() raises:
    """Called on every chunk entering every operator that has not been taught
    about selections, so the second call has to be free rather than wrong."""
    var chunk = selected_chunk()
    chunk.flatten()
    chunk.flatten()
    assert_false(chunk.selected(), "still gone")
    var first = ints_of(chunk.columns[0], 2)
    assert_equal(first[0], 2, "not gathered through the positions again")
    assert_equal(first[1], 4, "nor this one")


def test_flattening_a_chunk_that_has_no_selection_changes_nothing() raises:
    """The common case, and the reason the call is cheap enough to make
    unconditionally."""
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3]))
    var chunk = Chunk(columns^)
    chunk.flatten()
    assert_false(chunk.selected(), "nothing appeared")
    assert_equal(len(chunk), 3, "three rows")
    var values = ints_of(chunk.columns[0], 3)
    assert_equal(values[0], 1, "the first row")
    assert_equal(values[2], 3, "the last row")


def test_one_column_can_be_asked_for_without_flattening_the_rest() raises:
    """What an operator reading two columns of a wide chunk wants. The chunk
    keeps its selection, so the columns nobody asked about are not gathered."""
    var chunk = selected_chunk()
    var got = ints_of(chunk.column(0), 2)
    assert_equal(got[0], 2, "gathered through the selection")
    assert_equal(got[1], 4, "and the second row")
    assert_true(chunk.selected(), "the chunk is unchanged")


def test_asking_for_a_dense_column_does_not_gather_it() raises:
    """A column computed since the selection was made is already at the chunk's
    rows, so reading it through the positions would be reading the wrong
    rows."""
    var chunk = selected_chunk()
    var got = ints_of(chunk.column(1), 2)
    assert_equal(got[0], 70, "the first row as it stands")
    assert_equal(got[1], 80, "and the second")


def test_asking_for_a_column_the_chunk_does_not_have_is_refused() raises:
    """The same message shape the filter and the projection give, because the
    mistake is the same one."""
    var chunk = selected_chunk()
    with assert_raises(contains="outside a chunk of 2 columns"):
        _ = chunk.column(2)


def test_giving_up_the_columns_flattens_first() raises:
    """Everything that consumes a chunk's arrays wants one array per column at
    the chunk's rows, so this is where a selection stops rather than being
    something every caller has to remember."""
    var chunk = selected_chunk()
    var columns = chunk^.into_columns()
    assert_equal(len(columns), 2, "two columns")
    assert_equal(len(columns[0]), 2, "gathered down to the rows")
    var first = ints_of(columns[0], 2)
    assert_equal(first[0], 2, "position 1")
    assert_equal(first[1], 4, "position 3")


def test_a_node_that_does_not_read_a_selection_is_given_a_flat_chunk() raises:
    """The safety property the whole step rests on. Nothing has been taught to
    read a selection yet, so a node handed a selected chunk sees it flattened
    and gives the answer it would have given anyway."""
    var keep = List[Int]()
    keep.append(1)
    keep.append(0)
    var node = Node(Project(keep^))
    var out = node_apply(node, selected_chunk())
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_false(got.selected(), "and it is not selected")
    assert_equal(len(got), 2, "two rows")
    var swapped = ints_of(got.columns[0], 2)
    assert_equal(swapped[0], 70, "what was column 1")
    var under = ints_of(got.columns[1], 2)
    assert_equal(under[0], 2, "and what was column 0, gathered")
    assert_equal(under[1], 4, "and its second row")


def test_a_sort_orders_rows_that_arrived_in_different_chunks() raises:
    # The whole of what a sort is for in a pipeline. Six rows arrive in three
    # chunks and the answer interleaves all three, which no per chunk operator
    # could produce.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Sort([0], [True], [False])))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(len(got), 6, "rows")
    for i in range(6):
        assert_equal(got[i], Int64(6 - i), "row " + String(i))


def test_a_sort_is_a_breaker_and_cuts_the_pipeline() raises:
    assert_true(node_is_breaker(Node(Sort([0], [False], [False]))), "it is one")

    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Sort([0], [False], [False])))
    pipeline.add(Node(Limit(2)))
    var cuts = pipeline.cut_points()
    assert_equal(len(cuts), 1, "one breaker")
    assert_equal(cuts[0], 1, "at the second operator")
    assert_equal(pipeline.stages(), 2, "stages")


def test_a_sort_hands_back_the_chunks_it_was_given() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Sort([0], [False], [False])))
    var out = pipeline^.run()
    assert_equal(out.columns[0].num_chunks(), 3, "two, three and one again")


def test_a_sort_leaves_the_schema_alone() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Sort([0], [True], [False])))
    var out = pipeline^.run()
    assert_equal(out.width(), 2, "columns")
    assert_equal(out.schema[0].name, "n", "the first")
    assert_equal(out.schema[1].name, "keep", "the second")


def test_a_sort_after_a_filter_orders_what_survived() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Sort([0], [True], [False])))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(len(got), 4, "the rows the mask kept")
    assert_equal(got[0], 6, "largest first")
    assert_equal(got[3], 1, "smallest last")


def test_a_sort_before_a_limit_is_the_top_of_the_frame() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Sort([0], [True], [False])))
    pipeline.add(Node(Limit(2)))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(len(got), 2, "rows")
    assert_equal(got[0], 6, "first")
    assert_equal(got[1], 5, "second")


def test_a_sort_puts_the_nulls_where_it_was_told_to() raises:
    var pipeline = Pipeline(gappy_frame())
    pipeline.add(Node(Sort([0], [False], [True])))
    var out = pipeline^.run()
    var there = present(out, "n")
    assert_equal(len(there), 6, "rows")
    assert_false(there[0], "the first is missing")
    assert_false(there[1], "and so is the second")
    for i in range(2, 6):
        assert_true(there[i], "row " + String(i) + " has a value")
    var got = read_back(out, "n")
    assert_equal(got[2], 1, "then the values, upwards")
    assert_equal(got[5], 6, "to the largest")


def test_a_sort_can_put_the_nulls_at_the_other_end() raises:
    var pipeline = Pipeline(gappy_frame())
    pipeline.add(Node(Sort([0], [False], [False])))
    var out = pipeline^.run()
    var there = present(out, "n")
    assert_true(there[0], "a value first")
    assert_false(there[4], "and the two missing ones at the end")
    assert_false(there[5], "both of them")


def test_a_sort_on_two_keys_breaks_the_first_key_ties() raises:
    # The mask is the dominant key and the number refines it, so the rows the
    # mask kept come last and each run is ordered by the number inside it.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Sort([1, 0], [False, True], [False, False])))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(got[0], 5, "the largest of the two it dropped")
    assert_equal(got[1], 2, "then the other")
    assert_equal(got[2], 6, "then the largest it kept")
    assert_equal(got[5], 1, "down to the smallest")


def test_a_sort_with_no_key_is_refused() raises:
    with assert_raises(contains="does not order anything"):
        _ = Sort(List[Int](), List[Bool](), List[Bool]())


def test_a_sort_whose_flags_do_not_match_its_keys_is_refused() raises:
    with assert_raises(contains="null placements"):
        _ = Sort([0, 1], [True], [False, False])


def test_a_sort_on_a_column_the_chunk_does_not_have_is_refused() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="outside a schema of 2 columns"):
        pipeline.add(Node(Sort([7], [False], [False])))


def test_a_sort_over_nothing_gives_nothing_back() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Limit(0)))
    pipeline.add(Node(Sort([0], [False], [False])))
    var out = pipeline^.run()
    assert_equal(len(out), 0, "rows")
    assert_equal(out.width(), 2, "and the schema still describes the result")


def test_a_window_over_the_whole_frame_writes_one_value_on_every_row() raises:
    # No partition keys is one partition, and the six rows arrived in three
    # chunks, so the answer on the first row is a sum over rows that were not
    # in the chunk it came in.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Window(List[Int](), [0], [AggKind.SUM], ["total"])))
    var out = pipeline^.run()
    assert_equal(out.width(), 3, "the input's columns and then the window")
    assert_equal(out.schema[2].name, "total", "under the name it was given")
    var got = read_back(out, "total")
    assert_equal(len(got), 6, "every row comes back")
    for i in range(6):
        assert_equal(got[i], 21, "row " + String(i))


def test_a_window_partitions_on_a_column_of_the_chunk() raises:
    # Two windows over one partitioning, which is one grouping pass. The mask
    # splits the six rows into four and two, and each row reads its own side.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(
        Node(
            Window(
                [1],
                [0, 0],
                [AggKind.SUM, AggKind.COUNT],
                ["total", "how_many"],
            )
        )
    )
    var out = pipeline^.run()
    assert_equal(out.width(), 4, "two windows on two columns")
    var totals = read_back(out, "total")
    assert_equal(totals[0], 14, "the rows the mask kept sum to this")
    assert_equal(totals[1], 7, "and the two it dropped to this")
    assert_equal(totals[4], 7, "which the last of them reads too")
    assert_equal(totals[5], 14, "and the last kept row reads the other")
    var counts = read_back(out, "how_many")
    assert_equal(counts[0], 4, "four rows on that side")
    assert_equal(counts[1], 2, "and two on this one")


def test_a_window_leaves_the_rows_where_they_were() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Window([1], [0], [AggKind.SUM], ["total"])))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    for i in range(6):
        assert_equal(got[i], Int64(i + 1), "row " + String(i))


def test_a_window_hands_back_the_chunks_it_was_given() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Window(List[Int](), [0], [AggKind.SUM], ["total"])))
    var out = pipeline^.run()
    assert_equal(out.columns[0].num_chunks(), 3, "two, three and one again")
    assert_equal(out.columns[2].num_chunks(), 3, "the window is cut the same")


def test_a_window_is_a_breaker_and_cuts_the_pipeline() raises:
    assert_true(
        node_is_breaker(
            Node(Window(List[Int](), [0], [AggKind.SUM], ["total"]))
        ),
        "it is one",
    )

    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Window(List[Int](), [0], [AggKind.SUM], ["total"])))
    pipeline.add(Node(Limit(2)))
    var cuts = pipeline.cut_points()
    assert_equal(len(cuts), 1, "one breaker")
    assert_equal(cuts[0], 1, "at the second operator")
    assert_equal(pipeline.stages(), 2, "stages")


def test_a_window_after_a_filter_reduces_what_survived() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Window(List[Int](), [0], [AggKind.SUM], ["total"])))
    var out = pipeline^.run()
    var got = read_back(out, "total")
    assert_equal(len(got), 4, "the rows the mask kept")
    for i in range(4):
        assert_equal(got[i], 14, "row " + String(i))


def test_a_window_over_nothing_gives_nothing_back() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Limit(0)))
    pipeline.add(Node(Window(List[Int](), [0], [AggKind.SUM], ["total"])))
    var out = pipeline^.run()
    assert_equal(len(out), 0, "rows")
    assert_equal(out.width(), 3, "and the schema still describes the result")


def test_a_window_with_no_window_in_it_is_refused() raises:
    with assert_raises(contains="is the operator below it"):
        _ = Window(List[Int](), List[Int](), List[AggKind](), List[String]())


def test_a_window_whose_lists_do_not_line_up_is_refused() raises:
    with assert_raises(contains="2 columns, 1 reductions"):
        _ = Window(List[Int](), [0, 0], [AggKind.SUM], ["total", "how_many"])


def test_a_window_on_a_column_the_chunk_does_not_have_is_refused() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="window: column 7 is outside"):
        pipeline.add(Node(Window(List[Int](), [7], [AggKind.SUM], ["total"])))


def test_a_window_partitioned_on_a_column_that_is_not_there_is_refused() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="partition column 7 is outside"):
        pipeline.add(Node(Window([7], [0], [AggKind.SUM], ["total"])))


def test_a_window_given_the_same_partition_column_twice_is_refused() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="partition column 1 was given twice"):
        pipeline.add(Node(Window([1, 1], [0], [AggKind.SUM], ["total"])))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
