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

from std.testing import TestSuite, assert_equal, assert_false, assert_raises
from std.testing import assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.exec import (
    Chunk,
    Collect,
    Filter,
    Limit,
    Materialize,
    Node,
    NodeStatus,
    Pipeline,
    Project,
    Scan,
    node_is_breaker,
    node_status,
)
from firepanda.frame.frame import DataFrame


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


def read_back(df: DataFrame, name: String) raises -> List[Int64]:
    """Reads an int64 column out as a plain list."""
    var col = df.column(name).as_typed[DType.int64]()
    var out = List[Int64](capacity=len(col))
    for i in range(len(col)):
        out.append(col[i])
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
