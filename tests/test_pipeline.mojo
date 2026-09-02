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
from firepanda.array.strings import strings_from_list
from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.exec import (
    Cast,
    Chunk,
    Collect,
    Compute,
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
from firepanda.kernel.binary import BinaryOp


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
    """Column 1 is bool, and adding two bools has no answer, so the pipeline
    refuses to be built rather than raising on the first chunk."""
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="is not defined on"):
        pipeline.add(Node(Compute(1, 1, BinaryOp.ADD, "nope")))


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


def test_a_cast_over_a_missing_column_is_caught_when_the_plan_is_built() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="is outside a schema of 2 columns"):
        pipeline.add(Node(Cast(4, LogicalType.FLOAT64)))


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
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="is not defined on"):
        pipeline.add(Node(Compute(1, Value(True), BinaryOp.ADD, "nope")))


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
