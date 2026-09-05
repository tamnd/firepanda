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
    Group,
    GroupAgg,
    Join,
    Limit,
    Materialize,
    Node,
    NodeStatus,
    Pipeline,
    Project,
    Reduce,
    Scan,
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


def test_a_reduction_that_does_not_fold_is_refused_at_plan_time() raises:
    """A median of medians is not a median, and there is no state short of the
    values that would make it one, so this is an error before a row moves."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.MEDIAN, "middle"))
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="cannot be computed a chunk at a time"):
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


def test_a_join_on_a_text_key_is_refused() raises:
    """A text key needs the ordinal space that comes from concatenating both
    sides, and having both sides is what a stream does not have."""
    var pipeline = Pipeline(word_frame())
    with assert_raises(contains="fixed width key"):
        pipeline.add(Node(Join(word_frame(), "status", "status")))


def test_a_join_on_keys_of_different_dtypes_is_refused() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="fixed width key"):
        pipeline.add(Node(Join(lookup_frame(), "keep", "n")))


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
