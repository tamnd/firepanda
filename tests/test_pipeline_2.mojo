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

Part 2 of 3. The fixtures are in tests/support/pipeline.mojo.
"""


from std.math import isnan, nan
from std.testing import TestSuite, assert_equal, assert_false, assert_raises
from std.testing import assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.array.strings import strings_from_list
from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.dtype.temporal import TimeUnit
from firepanda.exec import (
    Apply,
    Case,
    Cast,
    Chunk,
    Collect,
    Compute,
    Constant,
    Cut,
    Expand,
    Fill,
    Filter,
    Group,
    GroupAgg,
    Join,
    Length,
    Limit,
    Locate,
    Match,
    Materialize,
    Node,
    NodeStatus,
    Part,
    Pipeline,
    Presence,
    Project,
    Reduce,
    Scan,
    Sort,
    Trim,
    Truncate,
    Unique,
    Window,
    node_apply,
    node_computes_per_row,
    node_ends_early,
    node_is_breaker,
    node_is_row_local,
    node_process,
    node_status,
)
from firepanda.exec.morsel import MORSEL_ROWS
from firepanda.frame.frame import DataFrame
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.pattern import MatchKind, Pattern
from firepanda.kernel.temporal import (
    TRUNC_DAY,
    TRUNC_MONTH,
    TRUNC_WEEK,
    TRUNC_YEAR,
    TemporalField,
)
from firepanda.kernel.unary import UnaryOp

from tests.support.pipeline import (
    _matched,
    _presence,
    barren_frame,
    big_frame,
    byte_lookup_frame,
    counter,
    cut_frame,
    cut_lookup_frame,
    cut_pair_frame,
    cut_word_key_frame,
    dated_frame,
    first_two,
    flags,
    gappy_frame,
    hollow_frame,
    identity,
    ints_of,
    joined_rows,
    kept_nothing,
    key_fields,
    key_words,
    lookup_frame,
    many_chunk_frame,
    masked_chunk,
    nans,
    numbers,
    one_int,
    only_n,
    pair_fields,
    pair_frame,
    pair_lookup_frame,
    pair_words,
    present,
    read_back,
    repeat_frame,
    sample_frame,
    selected_chunk,
    selected_masked_chunk,
    shelf_frame,
    six_rows,
    spaced_frame,
    tall_frame,
    totals,
    truths_of,
    two_masked_chunk,
    two_under_a_selection,
    word_frame,
    word_key_frame,
    word_lookup_frame,
)


def test_a_null_test_over_a_column_with_none_answers_for_every_row() raises:
    # Nothing to find, and the answer is still a column of six falses rather
    # than an empty one or a refusal.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Presence(0, True, "gone")))
    var out = pipeline^.run()
    var col = out.column("gone").as_typed[DType.bool]()
    assert_equal(len(col), 6, "one answer per row")
    for i in range(6):
        assert_true(not col[i], "nothing is missing at " + String(i))


def test_a_null_test_over_a_missing_column_is_caught_at_plan_time() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="is outside a schema of 2 columns"):
        pipeline.add(Node(Presence(4, True, "nope")))


def test_a_fill_takes_the_gaps_from_the_other_column() raises:
    # `gappy_frame` is 4, null, 1, 6, 2, null, and the constant beside it is a
    # column of nines, so the answer is the two nulls turned into nines and
    # nothing else moved.
    var pipeline = Pipeline(gappy_frame())
    pipeline.add(Node(Constant(Value(Int64(9)), LogicalType.INT64, "nine")))
    pipeline.add(Node(Fill(0, 1, "filled")))
    var out = pipeline^.run()
    assert_equal(out.width(), 3, "the answer was appended")
    var got = read_back(out, "filled")
    var want = [Int64(4), 9, 1, 6, 2, 9]
    assert_equal(len(got), 6, "one answer per row")
    for i in range(6):
        assert_equal(got[i], want[i], "filled at " + String(i))
    assert_true(
        not out.schema[2].nullable,
        "a fallback with nothing missing leaves nothing missing",
    )


def test_a_fill_leaves_the_column_it_read_where_it_was() raises:
    var pipeline = Pipeline(gappy_frame())
    pipeline.add(Node(Constant(Value(Int64(9)), LogicalType.INT64, "nine")))
    pipeline.add(Node(Fill(0, 1, "filled")))
    var out = pipeline^.run()
    var there = present(out, "n")
    assert_false(there[1], "the column it read still has its first gap")
    assert_false(there[5], "and its second")


def test_a_fill_from_a_column_that_has_gaps_of_its_own_keeps_them() raises:
    # Both sides are missing the same two rows, so there is nothing to fill
    # from and the answer is as gappy as what went in, which is the case a node
    # that assumed the fallback was whole would get wrong.
    var pipeline = Pipeline(gappy_frame())
    pipeline.add(Node(Fill(0, 0, "filled")))
    var out = pipeline^.run()
    var there = present(out, "filled")
    assert_false(there[1], "the first gap is still a gap")
    assert_false(there[5], "and so is the second")
    assert_true(there[0], "and the rows that had values still do")
    assert_true(out.schema[1].nullable, "the schema says so too")


def test_a_fill_over_a_missing_column_is_caught_at_plan_time() raises:
    var pipeline = Pipeline(gappy_frame())
    with assert_raises(contains="is outside a schema of 1 columns"):
        pipeline.add(Node(Fill(0, 3, "nope")))


def test_a_fill_between_two_types_is_caught_at_plan_time() raises:
    var pipeline = Pipeline(cut_frame())
    with assert_raises(contains="filled from one of its own type"):
        pipeline.add(Node(Fill(0, 1, "nope")))


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


def test_a_limit_above_a_breaker_leaves_the_prefix_on_every_core() raises:
    """A limit takes a pipeline off the cores because it can stop the source in
    the middle of a batch. Put a sort under it and it cannot: the sort holds
    every row and emits nothing until the source has run out, so the limit
    counts its first row after the last chunk has been read.

    The assertion is against the same pipeline without the limit rather than
    against a number, because how many operators run in parallel depends on how
    many workers the machine has and whether the limit is there does not."""
    var bounded = Pipeline(many_chunk_frame())
    bounded.add(Node(Filter(1)))
    bounded.add(Node(Sort([0], [False], [False])))
    bounded.add(Node(Limit(3)))
    var plain = Pipeline(many_chunk_frame())
    plain.add(Node(Filter(1)))
    plain.add(Node(Sort([0], [False], [False])))
    assert_equal(
        bounded._parallel_lead(),
        plain._parallel_lead(),
        "the limit above the sort decides nothing about the filter below it",
    )


def test_a_limit_with_nothing_holding_the_rows_under_it_still_stops() raises:
    """The other half of the rule. Nothing here holds a row, so the limit is
    reached while the source is still being read and reading a batch ahead would
    be reading rows it was about to make unnecessary."""
    var pipeline = Pipeline(many_chunk_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Limit(3)))
    assert_equal(pipeline._parallel_lead(), 0, "on the calling thread")


def test_a_filter_over_a_tall_chunk_is_run_on_morsels() raises:
    """A filter is row local and computes per row, so there is a prefix to hand
    out and the tall chunk under it is worth cutting. The chunks that come out
    are the morsels that went in, which is how the cut is visible from outside:
    without it the whole column arrives at the filter in one piece and leaves in
    one piece."""
    var pipeline = Pipeline(tall_frame())
    pipeline.add(Node(Filter(1)))
    var out = pipeline^.run()
    assert_equal(out.columns[0].num_chunks(), 2, "the two morsels")
    assert_equal(len(out), 2 * (MORSEL_ROWS + 3) // 3, "two rows in three")


def test_a_tall_chunk_under_a_breaker_is_left_in_one_piece() raises:
    """The other half, and the reason the cut waits for the line to be built. A
    sort holds every row it is given, so nothing above it runs while the source
    is being read and there is no prefix to spread the morsels over. Cutting
    would hand the sort its rows in eight pieces on one thread instead of once,
    and a kernel given a morsel cannot spread the work itself. See #918."""
    var pipeline = Pipeline(tall_frame())
    pipeline.add(Node(Sort([0], [True], [False])))
    assert_equal(pipeline._prefix_lead(), 0, "nothing to hand out")
    var out = pipeline^.run()
    assert_equal(len(out), MORSEL_ROWS + 3, "the rows are all still there")
    # A sort cuts its answer where its input was cut, so one chunk out is the
    # source saying it was never cut.
    assert_equal(out.columns[0].num_chunks(), 1, "one piece in, one out")
    var got = read_back(out, "n")
    assert_equal(got[0], Int64(MORSEL_ROWS + 3), "sorted downwards")


def test_a_project_over_a_tall_chunk_does_not_ask_for_morsels() raises:
    """A project is row local but it is not worth a task, so `_prefix_lead` is
    zero for the same reason it is zero in `_parallel_lead`, and a line that is
    not going to spread the work has no use for pieces to spread."""
    var pipeline = Pipeline(tall_frame())
    pipeline.add(Node(Project([0])))
    assert_equal(pipeline._prefix_lead(), 0, "not worth a task")
    var out = pipeline^.run()
    assert_equal(out.columns[0].num_chunks(), 1, "one chunk in, one out")


def test_a_filter_under_a_sort_under_a_limit_answers_the_same() raises:
    """The shape the rule is for, run for its rows rather than for its route.
    Every third row is dropped by the mask, the rest are ordered downwards, and
    three come back, so the answer is the three largest that survived."""
    var pipeline = Pipeline(many_chunk_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Sort([0], [True], [False], bound=3)))
    pipeline.add(Node(Limit(3)))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(len(got), 3, "rows")
    assert_equal(got[0], 200, "the largest that survived the mask")
    assert_equal(got[1], 199, "then the next")
    assert_equal(got[2], 197, "and 198 is a multiple of three")


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


def test_a_reduction_over_no_rows_answers_what_the_kernel_answers() raises:
    """The pandas answers, which is what a reduction carrying no mark gives. A
    sum of nothing is zero, a count of nothing is zero, and a minimum and a
    maximum of nothing are null."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.SUM, "total"))
    aggs.append(GroupAgg(0, AggKind.COUNT, "seen"))
    aggs.append(GroupAgg(0, AggKind.MAX, "high"))
    var pipeline = Pipeline(barren_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(
        len(out), 1, "one row, because a fold with no key is one group"
    )
    assert_equal(one_int(out, "total"), 0, "pandas sums nothing to zero")
    assert_equal(one_int(out, "seen"), 0, "and counts nothing as zero")
    var high = out.column("high").as_typed[DType.int64]()
    assert_true(not high.is_valid(0), "the maximum of nothing")


def test_a_marked_reduction_over_no_rows_answers_null() raises:
    """SQL's answer for the same input. The mark is on the sum and the maximum
    and not on the count, which is what the SQL front end does, and the only
    column it changes is the sum."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.SUM, "total", empty_is_null=True))
    aggs.append(GroupAgg(0, AggKind.COUNT, "seen"))
    aggs.append(GroupAgg(0, AggKind.MAX, "high", empty_is_null=True))
    var pipeline = Pipeline(barren_frame())
    pipeline.add(Node(Filter(1)))
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    var total = out.column("total").as_typed[DType.int64]()
    assert_true(not total.is_valid(0), "SQL sums nothing to null")
    assert_equal(one_int(out, "seen"), 0, "a count is zero on both sides")
    var high = out.column("high").as_typed[DType.int64]()
    assert_true(not high.is_valid(0), "the maximum of nothing, either way")


def test_a_marked_reduction_over_rows_is_the_ordinary_sum() raises:
    """The mark decides one row of one case and nothing else, so an input with
    rows in it answers what it always answered."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.SUM, "total", empty_is_null=True))
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    assert_equal(one_int(out, "total"), 21, "1 through 6")


def test_a_reduction_over_nothing_but_nulls_answers_zero() raises:
    """pandas' answer. Rows arrived and none of them held a value, and adding no
    numbers gives zero, which is what the accessor on a frame promises and what
    `firepanda/frame/groupby.mojo` writes down."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(2, AggKind.SUM, "total"))
    aggs.append(GroupAgg(2, AggKind.COUNT, "seen"))
    var pipeline = Pipeline(hollow_frame())
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(one_int(out, "total"), 0, "pandas adds nothing to zero")
    assert_equal(one_int(out, "seen"), 0, "and there was nothing to add")


def test_a_marked_reduction_over_nothing_but_nulls_answers_null() raises:
    """SQL's answer for the same four rows, which is #836. The mark is the only
    difference between this and the test above it, and the count is not marked
    on either side because a count of nothing is zero in both front ends."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(2, AggKind.SUM, "total", empty_is_null=True))
    aggs.append(GroupAgg(2, AggKind.COUNT, "seen"))
    var pipeline = Pipeline(hollow_frame())
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(present(out, "total"), [False], "SQL adds nothing to null")
    assert_equal(one_int(out, "seen"), 0, "a count is zero on both sides")


def test_a_marked_reduction_over_a_column_of_nans_answers_null_too() raises:
    """A NaN is not a value here either, so a column of them is a column that
    holds nothing even though the schema says it cannot hold a null. That is
    why `_sum_counts` asks about a float column separately. See #170."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(3, AggKind.SUM, "total", empty_is_null=True))
    var pipeline = Pipeline(hollow_frame())
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    var total = out.column("total").as_typed[DType.float64]()
    assert_true(not total.is_valid(0), "nothing was added")


def test_a_marked_reduction_that_added_something_is_the_ordinary_sum() raises:
    """The one value in the frame is still the answer, so the mark costs a
    count and changes nothing about the number it is beside."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.SUM, "total", empty_is_null=True))
    var pipeline = Pipeline(hollow_frame())
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(one_int(out, "total"), 5, "the five that was there")


def test_a_group_of_nothing_but_nulls_sums_to_zero() raises:
    """pandas' answer again, one group at a time."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.SUM, "total"))
    var keys = List[Int]()
    keys.append(1)
    var pipeline = Pipeline(hollow_frame())
    pipeline.add(Node(Group(keys^, aggs^)))
    var out = pipeline^.run()
    assert_equal(
        read_back(out, "g"), [1, 2], "the two groups, first seen first"
    )
    assert_equal(read_back(out, "total"), [5, 0], "the five and a zero")


def test_a_marked_group_of_nothing_but_nulls_sums_to_null() raises:
    """SQL's answer, and the half of #836 a group by owns. The group is there,
    because a row made it, and what it holds is three nulls, so the sum has
    nothing to report rather than a zero to report."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.SUM, "total", empty_is_null=True))
    var keys = List[Int]()
    keys.append(1)
    var pipeline = Pipeline(hollow_frame())
    pipeline.add(Node(Group(keys^, aggs^)))
    var out = pipeline^.run()
    assert_equal(read_back(out, "g"), [1, 2], "the two groups")
    assert_equal(present(out, "total"), [True, False], "the five and a null")
    assert_equal(read_back(out, "total")[0], 5, "the five is still a five")


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


def test_a_marked_sum_carrying_an_operation_shares_the_column_it_counts() raises:
    """The sum and the count beside it are two slots over one column under one
    operation, and `partial` builds that column once and hands it to both. The
    answers are what says the two halves read the same thing. `n` holds a five
    and three nulls, so the fold added one value and has a sum to report, and
    `bare` holds four nulls, so it added none and has not. See #922."""
    var aggs = List[GroupAgg]()
    aggs.append(
        GroupAgg(
            0,
            AggKind.SUM,
            "shifted",
            BinaryOp.ADD,
            Value(Int64(10)),
            empty_is_null=True,
        )
    )
    aggs.append(
        GroupAgg(
            2,
            AggKind.SUM,
            "nothing",
            BinaryOp.ADD,
            Value(Int64(10)),
            empty_is_null=True,
        )
    )
    var pipeline = Pipeline(hollow_frame())
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    assert_equal(present(out, "shifted"), [True], "one value was there")
    assert_equal(one_int(out, "shifted"), 15, "the five and its ten")
    assert_equal(present(out, "nothing"), [False], "no value was there")


def test_a_mean_carrying_an_operation_divides_by_what_it_counted() raises:
    """The other pair of slots over one column, and the same check on it. One
    through six with ten added to each is a mean of 13.5, and a count that had
    been given a different column to read would divide by the wrong number."""
    var aggs = List[GroupAgg]()
    aggs.append(
        GroupAgg(0, AggKind.MEAN, "middle", BinaryOp.ADD, Value(Int64(10)))
    )
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Reduce(aggs^)))
    var out = pipeline^.run()
    assert_equal(len(out), 1, "one row")
    var got = out.column("middle").as_typed[DType.float64]()[0]
    assert_equal(got, Float64(13.5), "one through six and six tens")


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


def test_a_build_side_in_three_chunks_joins_the_way_one_chunk_does() raises:
    """Issue #583. `only` is the borrow a column of exactly one chunk has, and a
    build side that arrived in pieces raised out of the build before a row was
    probed. It is stacked now, so which side of the join a chunked frame is on
    stops deciding whether the query runs at all."""
    var one = Pipeline(cut_frame())
    one.add(Node(Join(lookup_frame(), "n", "n", JoinKind.INNER)))
    var many = Pipeline(cut_frame())
    many.add(Node(Join(cut_lookup_frame(), "n", "n", JoinKind.INNER)))
    var a = joined_rows(one^)
    var b = joined_rows(many^)
    assert_equal(len(b), 3, "the three keys that matched")
    assert_equal(len(a), len(b), "the same height either way")
    for i in range(len(a)):
        assert_equal(a[i], b[i], "row " + String(i))


def test_a_chunked_build_side_brings_its_own_columns_across() raises:
    # The keys alone would pass the test above with the payload dropped, since
    # the answer is read off the probe side's key column. This reads the build
    # side's other column, whose rows are split across the same three chunks, so
    # a stack that lost a piece or reordered them shows up here.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Join(cut_lookup_frame(), "n", "n", JoinKind.INNER)))
    var out = pipeline^.run()
    var tags = read_back(out, "tag")
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            if tags[j] < tags[i]:
                var swap = tags[i]
                tags[i] = tags[j]
                tags[j] = swap
    assert_equal(len(tags), 3, "one tag per matched row")
    assert_equal(tags[0], Int64(20), "the tag from the first chunk")
    assert_equal(tags[1], Int64(40), "one from the second")
    assert_equal(tags[2], Int64(60), "and the other from the second")


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


def test_a_join_on_two_keys_gives_what_the_frame_join_gives() raises:
    var whole = pair_frame().join_on(
        pair_lookup_frame(), ["a", "b"], ["a2", "b2"], JoinKind.INNER
    )
    var pipeline = Pipeline(cut_pair_frame())
    pipeline.add(
        Node(
            Join(
                pair_lookup_frame(),
                "a",
                "a2",
                JoinKind.INNER,
                left_keys=[1, 2],
                right_keys=[0, 1],
            )
        )
    )
    var values = joined_rows(pipeline^)
    assert_equal(len(values), len(whole), "the same number of rows")
    assert_equal(len(values), 2, "and the two rows that pair on both keys")
    assert_equal(values[0], Int64(2), "the integer and the short string")
    assert_equal(values[1], Int64(4), "the integer and the long string")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
