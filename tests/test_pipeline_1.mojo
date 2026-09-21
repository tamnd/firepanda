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

Part 1 of 3. The fixtures are in tests/support/pipeline.mojo.
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


def test_a_scan_leaves_a_tall_chunk_alone_until_it_is_asked() raises:
    # A reader hands back one chunk however many rows it read, and whether that
    # chunk is worth cutting into morsels depends on what is above it, which the
    # scan does not know when it is built. So it is built whole and cut later.
    var scan = Scan(tall_frame())
    assert_equal(scan.num_chunks(), 1, "before the cut")
    scan.cut()
    assert_equal(scan.num_chunks(), 2, "after it")

    var sizes = List[Int]()
    var first = Int64(0)
    while True:
        var chunk = scan.next()
        if not chunk:
            break
        var got = chunk.take()
        sizes.append(len(got))
        if len(sizes) == 1:
            var col = got.column(0)
            first = col.as_typed[DType.int64]()[0]
    assert_equal(sizes[0], MORSEL_ROWS, "a whole morsel")
    assert_equal(sizes[1], 3, "and the rows past it")
    assert_equal(first, Int64(1), "the pieces come out in order")


def test_cutting_a_scan_twice_changes_nothing_the_second_time() raises:
    # `run` asks once, but the piece that is already a morsel long has to be
    # left where it is rather than counted again, which is the off by one a
    # second call would show.
    var scan = Scan(tall_frame())
    scan.cut()
    scan.cut()
    assert_equal(scan.num_chunks(), 2, "still two")


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


def test_a_cut_takes_the_characters_the_two_numbers_name() raises:
    # `word_frame` holds ok, fail, ok, ok, fail, ok in `status`, so a cut of two
    # from the front keeps the whole of the short rows and half of the long
    # ones, and the two answers differ, which they would not if the cut had
    # ignored one of its numbers.
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Cut(1, 1, 2, "front")))
    var out = pipeline^.run()
    assert_equal(out.width(), 4, "the answer was appended")
    assert_true(out.schema[3].dtype == LogicalType.STRING, "text out")
    var got = out.column("front").as_strings()
    assert_equal(len(got), 6, "one answer per row")
    assert_equal(got[0], "ok", "a row shorter than the window")
    assert_equal(got[1], "fa", "and one longer than it")
    assert_equal(got[5], "ok", "and the last row")


def test_a_cut_with_no_length_runs_to_the_end_of_each_row() raises:
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Cut(1, 2, None, "back")))
    var out = pipeline^.run()
    var got = out.column("back").as_strings()
    assert_equal(got[0], "k", "what was left of a two character row")
    assert_equal(got[1], "ail", "and of a four character one")


def test_a_cut_keeps_the_column_it_read_where_it_was() raises:
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Cut(1, 1, 1, "first")))
    var out = pipeline^.run()
    var whole = out.column("status").as_strings()
    assert_equal(whole[1], "fail", "the column it read is as it was")
    assert_equal(out.column("first").as_strings()[1], "f", "and the cut is new")


def test_a_cut_over_a_missing_column_is_caught_at_plan_time() raises:
    var pipeline = Pipeline(word_frame())
    with assert_raises(contains="is outside a schema of 3 columns"):
        pipeline.add(Node(Cut(9, 1, 2, "nope")))


def test_a_cut_over_a_column_that_is_not_text_is_caught_at_plan_time() raises:
    var pipeline = Pipeline(word_frame())
    with assert_raises(contains="a substring reads text"):
        pipeline.add(Node(Cut(0, 1, 2, "nope")))


def test_a_length_counts_the_characters_of_every_row() raises:
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Length(1, False, "wide")))
    var out = pipeline^.run()
    assert_equal(out.width(), 4, "the answer was appended")
    assert_true(out.schema[3].dtype == LogicalType.INT64, "a number out")
    var got = read_back(out, "wide")
    assert_equal(len(got), 6, "one answer per row")
    assert_equal(got[0], 2, "ok")
    assert_equal(got[1], 4, "fail")
    assert_equal(got[5], 2, "and the last row")


def test_a_length_keeps_the_column_it_read_where_it_was() raises:
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Length(1, False, "wide")))
    var out = pipeline^.run()
    assert_equal(out.column("status").as_strings()[1], "fail", "as it was")


def test_a_length_counts_the_bytes_when_it_is_asked_to() raises:
    # The other half of the flag. `word_frame` is all ASCII, so this pair says
    # the flag reaches the kernel and `test_sql_run` says the two kernels
    # answer different things on a row that is not.
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Length(1, True, "wide")))
    var out = pipeline^.run()
    var got = read_back(out, "wide")
    assert_equal(got[0], 2, "two bytes and two characters")
    assert_equal(got[1], 4, "and four of each here")


def test_a_length_over_a_missing_column_is_caught_at_plan_time() raises:
    var pipeline = Pipeline(word_frame())
    with assert_raises(contains="is outside a schema of 3 columns"):
        pipeline.add(Node(Length(9, False, "nope")))


def test_a_length_over_a_column_that_holds_no_text_is_refused() raises:
    var pipeline = Pipeline(word_frame())
    with assert_raises(contains="a length reads text"):
        pipeline.add(Node(Length(0, False, "nope")))


def test_a_case_change_rewrites_every_row() raises:
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Case(1, True, "loud")))
    var out = pipeline^.run()
    assert_equal(out.width(), 4, "the answer was appended")
    assert_true(out.schema[3].dtype == LogicalType.STRING, "text out")
    var got = out.column("loud").as_strings()
    assert_equal(got[0], "OK", "the first row was raised")
    assert_equal(got[1], "FAIL", "and so was the second")
    assert_equal(got[5], "OK", "and the last row")


def test_a_case_change_the_other_way_lowers() raises:
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Case(1, True, "loud")))
    pipeline.add(Node(Case(3, False, "quiet")))
    var out = pipeline^.run()
    var got = out.column("quiet").as_strings()
    assert_equal(got[0], "ok", "it came back down to what it was")
    assert_equal(got[1], "fail", "and so did this one")


def test_a_case_change_keeps_the_column_it_read_where_it_was() raises:
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Case(1, True, "loud")))
    var out = pipeline^.run()
    assert_equal(out.column("status").as_strings()[1], "fail", "as it was")


def test_a_case_change_over_a_missing_column_is_caught_at_plan_time() raises:
    var pipeline = Pipeline(word_frame())
    with assert_raises(contains="is outside a schema of 3 columns"):
        pipeline.add(Node(Case(9, True, "nope")))


def test_a_case_change_over_a_column_that_holds_no_text_is_refused() raises:
    var pipeline = Pipeline(word_frame())
    with assert_raises(contains="a case change reads text"):
        pipeline.add(Node(Case(0, True, "nope")))


def test_a_trim_takes_the_spaces_off_both_ends() raises:
    var pipeline = Pipeline(spaced_frame())
    pipeline.add(Node(Trim(1, "", False, True, True, "cut")))
    var out = pipeline^.run()
    assert_equal(out.width(), 3, "the answer was appended")
    assert_true(out.schema[2].dtype == LogicalType.STRING, "text out")
    var got = out.column("cut").as_strings()
    assert_equal(got[0], "hi", "both ends came off")
    assert_equal(got[2], "xxaxx", "a row with nothing on its ends is as it was")
    assert_equal(got[3], "", "an empty row stays empty")
    assert_equal(got[4], "", "and a row that is nothing but spaces becomes one")
    assert_equal(got[5], "end", "and the last row")


def test_a_trim_leaves_a_tab_where_sql_leaves_it() raises:
    # The one row that says which whitespace table this is asking. A tab is
    # whitespace to Python and is not one of the Zs characters, so DuckDB hands
    # this row back exactly as it arrived.
    var pipeline = Pipeline(spaced_frame())
    pipeline.add(Node(Trim(1, "", False, True, True, "cut")))
    var out = pipeline^.run()
    assert_equal(out.column("cut").as_strings()[1], "\tgo\t", "tabs stay on")


def test_a_trim_works_on_the_near_end_alone() raises:
    var pipeline = Pipeline(spaced_frame())
    pipeline.add(Node(Trim(1, "", False, True, False, "cut")))
    var out = pipeline^.run()
    var got = out.column("cut").as_strings()
    assert_equal(got[0], "hi  ", "the far end was left alone")
    assert_equal(got[5], "end  ", "and so was this one")


def test_a_trim_works_on_the_far_end_alone() raises:
    var pipeline = Pipeline(spaced_frame())
    pipeline.add(Node(Trim(1, "", False, False, True, "cut")))
    var out = pipeline^.run()
    var got = out.column("cut").as_strings()
    assert_equal(got[0], "  hi", "the near end was left alone")
    assert_equal(got[5], "end", "and the far end came off")


def test_a_trim_of_a_set_takes_any_of_those_characters_off() raises:
    var pipeline = Pipeline(spaced_frame())
    pipeline.add(Node(Trim(1, "x", True, True, True, "cut")))
    var out = pipeline^.run()
    var got = out.column("cut").as_strings()
    assert_equal(got[2], "a", "the characters in the set came off")
    assert_equal(got[0], "  hi  ", "and a set does not mean whitespace as well")


def test_a_trim_keeps_the_column_it_read_where_it_was() raises:
    var pipeline = Pipeline(spaced_frame())
    pipeline.add(Node(Trim(1, "", False, True, True, "cut")))
    var out = pipeline^.run()
    assert_equal(out.column("padded").as_strings()[0], "  hi  ", "as it was")


def test_a_trim_over_a_missing_column_is_caught_at_plan_time() raises:
    var pipeline = Pipeline(spaced_frame())
    with assert_raises(contains="is outside a schema of 2 columns"):
        pipeline.add(Node(Trim(9, "", False, True, True, "nope")))


def test_a_trim_over_a_column_that_holds_no_text_is_refused() raises:
    var pipeline = Pipeline(spaced_frame())
    with assert_raises(contains="a trim reads text"):
        pipeline.add(Node(Trim(0, "", False, True, True, "nope")))


def test_a_locate_counts_from_one_and_says_zero_for_a_miss() raises:
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Locate(1, "a", "where")))
    var out = pipeline^.run()
    assert_equal(out.width(), 4, "the answer was appended")
    assert_true(out.schema[3].dtype == LogicalType.INT64, "a number out")
    var got = read_back(out, "where")
    assert_equal(len(got), 6, "one answer per row")
    assert_equal(got[0], 0, "ok does not hold an a")
    assert_equal(got[1], 2, "and fail holds one in second place")


def test_a_locate_of_a_run_that_is_the_whole_element_answers_one() raises:
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Locate(1, "ok", "where")))
    var out = pipeline^.run()
    var got = read_back(out, "where")
    assert_equal(got[0], 1, "the run starts at the first character")
    assert_equal(got[1], 0, "and is not in this one at all")


def test_a_locate_keeps_the_column_it_read_where_it_was() raises:
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Locate(1, "a", "where")))
    var out = pipeline^.run()
    assert_equal(out.column("status").as_strings()[1], "fail", "as it was")


def test_a_locate_over_a_missing_column_is_caught_at_plan_time() raises:
    var pipeline = Pipeline(word_frame())
    with assert_raises(contains="is outside a schema of 3 columns"):
        pipeline.add(Node(Locate(9, "a", "nope")))


def test_a_locate_over_a_column_that_holds_no_text_is_refused() raises:
    var pipeline = Pipeline(word_frame())
    with assert_raises(contains="a search reads text"):
        pipeline.add(Node(Locate(0, "a", "nope")))


def test_a_part_appends_the_field_it_was_asked_for() raises:
    var pipeline = Pipeline(dated_frame())
    pipeline.add(Node(Part(0, TemporalField.MONTH, "m")))
    var out = pipeline^.run()
    assert_equal(out.width(), 3, "the answer was appended")
    assert_true(
        out.schema[2].dtype == LogicalType.INT64,
        "a whole number of the width DuckDB answers",
    )
    var got = read_back(out, "m")
    assert_equal(len(got), 4, "one answer per row")
    assert_equal(got[0], 6, "the last day of June")
    assert_equal(got[1], 7, "the first of July")
    assert_equal(got[3], 8, "and the first of August")


def test_a_part_of_a_missing_day_is_missing() raises:
    var pipeline = Pipeline(dated_frame())
    pipeline.add(Node(Part(0, TemporalField.YEAR, "y")))
    var out = pipeline^.run()
    var there = present(out, "y")
    assert_true(there[0], "a day that is there")
    assert_true(not there[2], "and one that is not")


def test_a_part_keeps_the_column_it_read_where_it_was() raises:
    var pipeline = Pipeline(dated_frame())
    pipeline.add(Node(Part(0, TemporalField.DAY, "dd")))
    var out = pipeline^.run()
    assert_true(out.schema[0].dtype == LogicalType.DATE32, "the days are days")
    assert_equal(read_back(out, "dd")[0], 30, "and the field is new")


def test_a_part_over_a_missing_column_is_caught_at_plan_time() raises:
    var pipeline = Pipeline(dated_frame())
    with assert_raises(contains="is outside a schema of 2 columns"):
        pipeline.add(Node(Part(9, TemporalField.YEAR, "nope")))


def test_a_part_over_a_column_that_is_not_temporal_is_caught() raises:
    var pipeline = Pipeline(dated_frame())
    with assert_raises(contains="a date or a timestamp"):
        pipeline.add(Node(Part(1, TemporalField.YEAR, "nope")))


def test_a_part_asked_for_a_yes_or_no_field_is_refused() raises:
    # The seven predicates are names on `dt` and SQL has no spelling for any of
    # them, so a plan that asks for one was not built from a query.
    var pipeline = Pipeline(dated_frame())
    with assert_raises(contains="answers yes or no rather than a number"):
        pipeline.add(Node(Part(0, TemporalField.IS_LEAP_YEAR, "nope")))


def test_a_truncate_appends_the_period_start_of_every_row() raises:
    var pipeline = Pipeline(dated_frame())
    pipeline.add(Node(Truncate(0, TRUNC_MONTH, "m")))
    var out = pipeline^.run()
    assert_equal(out.width(), 3, "the answer was appended")
    var got = read_back(out, "m")
    assert_equal(len(got), 4, "one answer per row")
    assert_equal(got[0], 1370044800000000, "the 30th of June is the 1st")
    assert_equal(got[1], 1372636800000000, "the 1st of July is itself")
    assert_equal(got[3], 1375315200000000, "and so is the 1st of August")


def test_a_truncate_answers_a_timestamp_even_off_a_date() raises:
    # DuckDB's rule, and the one thing about `DATE_TRUNC` that surprises
    # people: truncating a date to the year gives a timestamp at midnight.
    var pipeline = Pipeline(dated_frame())
    pipeline.add(Node(Truncate(0, TRUNC_YEAR, "y")))
    var out = pipeline^.run()
    assert_true(
        out.schema[2].dtype == LogicalType.timestamp(TimeUnit.MICRO),
        "microseconds and not days",
    )
    assert_equal(read_back(out, "y")[0], 1356998400000000, "the 1st of 2013")


def test_a_truncate_of_a_missing_day_is_missing() raises:
    var pipeline = Pipeline(dated_frame())
    pipeline.add(Node(Truncate(0, TRUNC_WEEK, "w")))
    var out = pipeline^.run()
    var there = present(out, "w")
    assert_true(there[0], "a day that is there")
    assert_true(not there[2], "and one that is not")


def test_a_truncate_keeps_the_column_it_read_where_it_was() raises:
    var pipeline = Pipeline(dated_frame())
    pipeline.add(Node(Truncate(0, TRUNC_DAY, "start")))
    var out = pipeline^.run()
    assert_true(out.schema[0].dtype == LogicalType.DATE32, "the days are days")
    assert_equal(
        read_back(out, "start")[0], 1372550400000000, "and the answer is new"
    )


def test_a_truncate_over_a_missing_column_is_caught_at_plan_time() raises:
    var pipeline = Pipeline(dated_frame())
    with assert_raises(contains="is outside a schema of 2 columns"):
        pipeline.add(Node(Truncate(9, TRUNC_YEAR, "nope")))


def test_a_truncate_over_a_column_that_is_not_temporal_is_caught() raises:
    var pipeline = Pipeline(dated_frame())
    with assert_raises(contains="a date or a timestamp is what there is"):
        pipeline.add(Node(Truncate(1, TRUNC_YEAR, "nope")))


def test_a_truncate_to_a_period_that_is_not_a_unit_is_refused() raises:
    var pipeline = Pipeline(dated_frame())
    with assert_raises(contains="is not one of the periods"):
        pipeline.add(Node(Truncate(0, 99, "nope")))


def test_a_null_test_appends_a_column_of_yes_and_no() raises:
    var pipeline = Pipeline(gappy_frame())
    pipeline.add(Node(Presence(0, True, "gone")))
    var out = pipeline^.run()
    assert_equal(out.width(), 2, "the answer was appended")
    assert_equal(out.schema[1].name, "gone", "the name it was given")
    assert_true(out.schema[1].dtype == LogicalType.BOOL, "a yes or no")
    assert_true(
        not out.schema[1].nullable,
        "the answer is never a null, whatever the column under it holds",
    )


def test_the_two_null_tests_are_the_opposite_of_each_other() raises:
    # `gappy_frame` is 4, null, 1, 6, 2, null over two chunks, so the second
    # null is in the chunk the first one is not, and a node that read the
    # validity of the first chunk twice would answer the fourth row wrongly.
    var missing = _presence(True)
    var there = _presence(False)
    assert_equal(len(missing), 6, "one answer per row")
    for i in range(6):
        assert_equal(missing[i], i == 1 or i == 5, "missing at " + String(i))
        assert_equal(there[i], not missing[i], "the other way at " + String(i))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
