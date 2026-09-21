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

Part 3 of 3. The fixtures are in tests/support/pipeline.mojo.
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


def test_a_left_join_on_two_keys_keeps_the_rows_that_matched_nothing() raises:
    """What TPC-H q20 asks for. A left join on a pair of keys used to be refused
    outright, because pairing on the first key and testing the second above the
    join cannot tell a row that matched nothing from a row it is about to drop,
    and there was nothing else to try."""
    var pipeline = Pipeline(cut_pair_frame())
    pipeline.add(
        Node(
            Join(
                pair_lookup_frame(),
                "a",
                "a2",
                JoinKind.LEFT,
                left_keys=[1, 2],
                right_keys=[0, 1],
            )
        )
    )
    var values = joined_rows(pipeline^)
    assert_equal(len(values), 6, "every left row, matched or not")


def test_a_semi_join_on_two_keys_keeps_the_rows_that_matched() raises:
    var pipeline = Pipeline(cut_pair_frame())
    pipeline.add(
        Node(
            Join(
                pair_lookup_frame(),
                "a",
                "a2",
                JoinKind.SEMI,
                left_keys=[1, 2],
                right_keys=[0, 1],
            )
        )
    )
    var values = joined_rows(pipeline^)
    assert_equal(len(values), 2, "the two rows that pair on both keys")
    assert_equal(values[0], Int64(2), "the first of them")
    assert_equal(values[1], Int64(4), "the second")


def test_a_two_key_join_over_chunks_agrees_with_one_over_one_chunk() raises:
    var one = Pipeline(pair_frame())
    one.add(
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
    var many = Pipeline(cut_pair_frame())
    many.add(
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
    var a = joined_rows(one^)
    var b = joined_rows(many^)
    assert_equal(len(a), len(b), "the same height")
    for i in range(len(a)):
        assert_equal(a[i], b[i], "row " + String(i))


def test_a_null_in_one_of_two_keys_pairs_with_nothing() raises:
    """A null key matches nothing in SQL whatever the rest of the tuple says, so
    the packed element is null when any key is, and the node's own null handling
    drops it from there."""
    var probe = pair_frame()
    var key = probe.columns[1].only().as_typed[DType.int64]()
    key.set_null(1)
    var columns = List[AnyArray]()
    columns.append(AnyArray(copy=probe.columns[0].only()))
    columns.append(AnyArray(key^))
    columns.append(AnyArray(copy=probe.columns[2].only()))
    var holed = DataFrame(Schema(pair_fields()), columns^)
    var pipeline = Pipeline(holed^)
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
    assert_equal(len(values), 1, "the row whose integer went null is gone")
    assert_equal(values[0], Int64(4), "and the other one is still there")


def test_a_join_given_a_different_number_of_keys_on_each_side_is_refused() raises:
    var pipeline = Pipeline(cut_pair_frame())
    with assert_raises(contains="on the probe side against"):
        pipeline.add(
            Node(
                Join(
                    pair_lookup_frame(),
                    "a",
                    "a2",
                    JoinKind.INNER,
                    left_keys=[1, 2],
                    right_keys=[0],
                )
            )
        )


def test_a_two_key_join_whose_second_pair_disagrees_on_dtype_is_refused() raises:
    var pipeline = Pipeline(cut_pair_frame())
    with assert_raises(contains="key 1 is text and int64"):
        pipeline.add(
            Node(
                Join(
                    pair_lookup_frame(),
                    "a",
                    "a2",
                    JoinKind.INNER,
                    left_keys=[1, 2],
                    right_keys=[0, 2],
                )
            )
        )


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
    """The safety property the whole step rests on. A constant column has not
    been taught to read a selection, so a node that appends one sees its input
    flattened and gives the answer it would have given anyway."""
    var node = Node(Constant(Value(Int64(9)), LogicalType.INT64, "nine"))
    var out = node_process(node, selected_chunk())
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_false(got.selected(), "and it is not selected")
    assert_equal(len(got), 2, "two rows")
    var first = ints_of(got.columns[0], 2)
    assert_equal(first[0], 2, "position 1 of the six, gathered")
    assert_equal(first[1], 4, "and position 3")
    var second = ints_of(got.columns[1], 2)
    assert_equal(second[0], 70, "the dense column as it stood")


def test_a_filter_over_a_chunk_with_no_selection_writes_one() raises:
    """The point of the whole exercise. Two rows of six survive and not one
    value is moved: the column that comes back is the column that went in, and
    the two rows are the two positions the mask was true on."""
    var out = node_apply(
        Node(Filter(1)), masked_chunk([True, False, False, True, False, False])
    )
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_true(got.selected(), "and it carries a selection")
    assert_equal(len(got), 2, "two rows")
    assert_equal(len(got.columns[0]), 6, "over a column that still holds six")
    var values = ints_of(got.column(0), 2)
    assert_equal(values[0], 1, "the first row the mask kept")
    assert_equal(values[1], 4, "the second")
    var mask = truths_of(got.column(1), 2)
    assert_true(mask[0], "the mask read through its own selection is all true")
    assert_true(mask[1], "including the last row of it")


def test_a_filter_that_keeps_nearly_every_row_copies_instead() raises:
    """Five rows of six, well over `SELECTION_KEEP_LIMIT`, so the gathers a
    selection would cost downstream come to more than the copy does. The answer
    is the same answer, and it arrives flat."""
    var out = node_apply(
        Node(Filter(1)), masked_chunk([True, True, False, True, True, True])
    )
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_false(got.selected(), "copied rather than selected")
    assert_equal(len(got), 5, "five rows")
    var values = ints_of(got.columns[0], 5)
    assert_equal(values[0], 1, "the first row")
    assert_equal(values[2], 4, "the row after the one that was dropped")
    assert_equal(values[4], 6, "and the last")


def test_a_filter_over_a_selected_chunk_composes_the_two() raises:
    """The case #532 could not do. The values are not touched a second time,
    the two selections are composed into one, and only the mask, which is the
    column written since the first filter, is gathered."""
    var out = node_apply(Node(Filter(1)), selected_masked_chunk())
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_true(got.selected(), "still under a selection")
    assert_equal(len(got), 2, "two of the three rows")
    assert_equal(len(got.columns[0]), 6, "the values were left where they were")
    var values = ints_of(got.column(0), 2)
    assert_equal(values[0], 2, "position 1 of the original six")
    assert_equal(values[1], 6, "and position 5, which the mask kept")
    assert_equal(len(got.columns[1]), 2, "the mask was gathered down to them")
    var mask = truths_of(got.column(1), 2)
    assert_true(mask[0], "and what it kept is what it says")
    assert_true(mask[1], "on both rows")


def test_a_filter_narrowing_to_gathered_columns_comes_back_flat() raises:
    """A filter that asks only for columns computed above it gathers every one
    of them, so there is nothing left for a selection to point at. Handing one
    on would make every operator downstream flatten a chunk already flat."""
    var keep = List[Int]()
    keep.append(1)
    var out = node_apply(Node(Filter(1, keep^)), selected_masked_chunk())
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_false(got.selected(), "no selection to carry")
    assert_equal(len(got), 2, "two rows")
    assert_equal(got.width(), 1, "one column, the one that was asked for")
    var mask = truths_of(got.columns[0], 2)
    assert_true(mask[0], "gathered, not read through anything")
    assert_true(mask[1], "and its second row")


def test_a_mask_read_through_a_selection_picks_the_right_rows() raises:
    """The rare shape: the mask itself arrived under the selection, because
    nothing computed it since the last filter. It is gathered before it is read,
    which costs a byte a row, and the positions it gives are positions into the
    chunk's rows like any other mask."""
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(flags([True, False, True, False, True, False]))
    var picks = List[UInt32]()
    picks.append(1)
    picks.append(2)
    picks.append(4)
    var dense = List[Bool]()
    dense.append(False)
    dense.append(False)
    var out = node_apply(Node(Filter(1)), Chunk(columns^, picks^, dense^))
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_true(got.selected(), "under a composed selection")
    assert_equal(len(got), 2, "the rows at positions 2 and 4 survived")
    var values = ints_of(got.column(0), 2)
    assert_equal(values[0], 3, "position 2 of the original six")
    assert_equal(values[1], 5, "and position 4")


def test_a_filter_that_keeps_nothing_under_a_selection_drops_it() raises:
    """A chunk of no rows is work for everything downstream and no information,
    so it is dropped here rather than passed on, whichever route the filter
    would have taken."""
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(flags([False, False, False]))
    var picks = List[UInt32]()
    picks.append(1)
    picks.append(3)
    picks.append(5)
    var dense = List[Bool]()
    dense.append(False)
    dense.append(True)
    var out = node_apply(Node(Filter(1)), Chunk(columns^, picks^, dense^))
    assert_false(out.__bool__(), "nothing came back")


def test_a_repeated_column_under_a_selection_comes_back_twice() raises:
    """A narrowing filter is a projection, and a projection may repeat a
    position. The last use of an array can take it and an earlier one has to
    copy, which is the same rule `Project` follows."""
    var keep = List[Int]()
    keep.append(0)
    keep.append(0)
    var out = node_apply(Node(Filter(1, keep^)), selected_masked_chunk())
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_true(got.selected(), "both columns are read through the selection")
    assert_equal(got.width(), 2, "two columns out of one")
    var left = ints_of(got.column(0), 2)
    var right = ints_of(got.column(1), 2)
    assert_equal(left[0], 2, "the first copy")
    assert_equal(left[1], 6, "and its second row")
    assert_equal(right[0], 2, "the second copy, which is the same rows")
    assert_equal(right[1], 6, "and the same second row")


def test_two_filters_agree_with_the_same_pair_flattened_between() raises:
    """The equivalence the step rests on. Two filters over eight rows, once
    with the selection carried from the first into the second and once with the
    chunk flattened in the middle, and the answers have to agree row for row."""
    var first: List[Bool] = [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
    ]
    var second: List[Bool] = [
        True,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
    ]

    var composed = node_apply(Node(Filter(1)), two_masked_chunk(first, second))
    assert_true(composed.__bool__(), "the first filter kept three rows")
    var carried = node_apply(Node(Filter(2)), composed.take())
    assert_true(carried.__bool__(), "and the second kept two of them")
    var under = carried.take()
    var through = ints_of(under.column(0), len(under))

    var again = node_apply(Node(Filter(1)), two_masked_chunk(first, second))
    assert_true(again.__bool__(), "the same three rows")
    var flat = again.take()
    flat.flatten()
    var copied = node_apply(Node(Filter(2)), flat^)
    assert_true(copied.__bool__(), "and the same two")
    var plain = copied.take()
    var moved = ints_of(plain.column(0), len(plain))

    assert_equal(len(through), 2, "two rows either way")
    assert_equal(len(moved), 2, "two the other way too")
    for i in range(2):
        assert_equal(through[i], moved[i], "the same row in the same place")
    assert_equal(through[0], 1, "the first row both masks kept")
    assert_equal(through[1], 7, "and the second")


def test_a_projection_passes_a_selection_through() raises:
    """Reordering columns moves no rows, so the positions come out the way they
    went in and the dense flags follow the columns they belong to. Nothing is
    gathered here at all, which is the point: the gather is left for whoever
    actually reads a column, and it may never happen for a column that is
    dropped further up."""
    var keep = List[Int]()
    keep.append(1)
    keep.append(0)
    var out = node_apply(Node(Project(keep^)), selected_masked_chunk())
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_true(got.selected(), "still under the selection it arrived with")
    assert_equal(len(got), 3, "three rows")
    assert_equal(got.width(), 2, "two columns, swapped")
    assert_equal(len(got.columns[1]), 6, "the values were not touched")
    var mask = truths_of(got.column(0), 3)
    assert_true(mask[0], "what was the dense mask, read as it stands")
    assert_false(mask[1], "its second row")
    var values = ints_of(got.column(1), 3)
    assert_equal(values[0], 2, "and the values, read through the positions")
    assert_equal(values[1], 4, "the second row")
    assert_equal(values[2], 6, "and the third")


def test_a_projection_of_dense_columns_only_drops_the_selection() raises:
    """Nothing left for the positions to point at, so carrying them on would
    make every operator above this flatten a chunk that is already flat. The
    same rule a narrowing filter follows."""
    var keep = List[Int]()
    keep.append(1)
    var out = node_apply(Node(Project(keep^)), selected_masked_chunk())
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_false(got.selected(), "no selection to carry")
    assert_equal(len(got), 3, "three rows, which is what the selection said")
    assert_equal(got.width(), 1, "one column")
    var mask = truths_of(got.columns[0], 3)
    assert_true(mask[0], "the dense column, unmoved")
    assert_false(mask[1], "and its second row")


def test_a_projection_repeating_a_column_under_a_selection() raises:
    """The last use of an array takes it and an earlier one copies, and neither
    of them is gathered, so the two copies are two views of the same six values
    through the same three positions."""
    var keep = List[Int]()
    keep.append(0)
    keep.append(0)
    var out = node_apply(Node(Project(keep^)), selected_masked_chunk())
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_true(got.selected(), "both columns are read through the selection")
    assert_equal(got.width(), 2, "two columns out of one")
    var left = ints_of(got.column(0), 3)
    var right = ints_of(got.column(1), 3)
    assert_equal(left[0], 2, "the first copy")
    assert_equal(left[2], 6, "and its last row")
    assert_equal(right[0], 2, "the second copy, the same rows")
    assert_equal(right[2], 6, "and the same last row")


def test_a_projection_agrees_with_the_same_one_flattened_first() raises:
    """The equivalence this step rests on, for the operator it was added to. A
    filter then a projection, once with the selection carried across and once
    flattened in between, and the answers have to agree row for row."""
    var mask: List[Bool] = [True, False, False, True, False, False, True, False]
    var second = List[Bool](length=8, fill=False)

    var keep = List[Int]()
    keep.append(2)
    keep.append(0)

    var filtered = node_apply(Node(Filter(1)), two_masked_chunk(mask, second))
    assert_true(filtered.__bool__(), "the filter kept three rows")
    var carried = node_apply(
        Node(Project(List[Int](copy=keep))), filtered.take()
    )
    assert_true(carried.__bool__(), "and the projection passed them on")
    var under = carried.take()
    var through = ints_of(under.column(1), len(under))

    var again = node_apply(Node(Filter(1)), two_masked_chunk(mask, second))
    assert_true(again.__bool__(), "the same three rows")
    var flat = again.take()
    flat.flatten()
    var copied = node_apply(Node(Project(keep^)), flat^)
    assert_true(copied.__bool__(), "and the same projection over them")
    var plain = copied.take()
    var moved = ints_of(plain.column(1), len(plain))

    assert_equal(len(through), 3, "three rows either way")
    assert_equal(len(moved), 3, "three the other way too")
    for i in range(3):
        assert_equal(through[i], moved[i], "the same row in the same place")
    assert_equal(through[0], 1, "the first row the mask kept")
    assert_equal(through[2], 7, "and the last")


def test_a_computed_column_gathers_only_the_operands_it_names() raises:
    """An expression names one or two columns and a flatten gathers all of
    them, which on a wide chunk is most of the work the selection was written to
    avoid. The column that was not named is left exactly as the scan handed it
    over."""
    var out = node_apply(
        Node(Compute(0, Value(Int64(2)), BinaryOp.MUL, "double")),
        two_under_a_selection(),
    )
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_true(got.selected(), "still under the selection")
    assert_equal(got.width(), 3, "the computed column is on the end")
    assert_equal(len(got.columns[0]), 3, "the operand was gathered into place")
    assert_equal(len(got.columns[1]), 6, "and the other column was not touched")
    var made = ints_of(got.columns[2], 3)
    assert_equal(made[0], 4, "twice the value at position 1")
    assert_equal(made[1], 8, "at position 3")
    assert_equal(made[2], 12, "and at position 5")


def test_computing_twice_over_one_column_gathers_it_once() raises:
    """The gathered column goes back into the chunk, so a line of expressions
    over the same column costs one gather rather than one each. The second
    compute finds it already at the chunk's rows."""
    var first = node_apply(
        Node(Compute(0, Value(Int64(2)), BinaryOp.MUL, "double")),
        two_under_a_selection(),
    )
    assert_true(first.__bool__(), "the first came back")
    var once = first.take()
    assert_equal(len(once.columns[0]), 3, "gathered by the first compute")
    var second = node_apply(
        Node(Compute(0, Value(Int64(1)), BinaryOp.ADD, "up")), once^
    )
    assert_true(second.__bool__(), "the second came back")
    var got = second.take()
    assert_true(got.selected(), "the other column still needs the positions")
    assert_equal(got.width(), 4, "both computed columns are there")
    var up = ints_of(got.columns[3], 3)
    assert_equal(up[0], 3, "one more than the value at position 1")
    assert_equal(up[2], 7, "and than the one at position 5")


def test_computing_over_every_column_drops_the_selection() raises:
    """Once every column has been gathered there is nothing left for the
    positions to point at, so the chunk comes out flat and nothing above it
    flattens a chunk that is already flat."""
    var out = node_apply(
        Node(Compute(0, 1, BinaryOp.ADD, "both")), two_under_a_selection()
    )
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_false(got.selected(), "no selection left")
    assert_equal(len(got), 3, "three rows")
    var sum = ints_of(got.columns[2], 3)
    assert_equal(sum[0], 22, "2 and 20")
    assert_equal(sum[1], 44, "4 and 40")
    assert_equal(sum[2], 66, "6 and 60")


def test_a_cast_under_a_selection_converts_only_its_own_column() raises:
    """A cast reads a row and writes a row, so it could convert the array where
    it lies, but that converts six values to answer for three. The column is
    gathered first and the rest of the chunk is left alone."""
    var out = node_apply(
        Node(Cast(0, LogicalType.INT32)), two_under_a_selection()
    )
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_true(got.selected(), "the other column still reads through it")
    assert_equal(len(got.columns[0]), 3, "converted down to the rows")
    assert_equal(len(got.columns[1]), 6, "and the other one was not touched")
    ref view = got.columns[0].as_typed_view[DType.int32]()
    assert_equal(Int(view[0]), 2, "the value at position 1")
    assert_equal(Int(view[2]), 6, "and the one at position 5")


def test_a_compute_agrees_with_the_same_one_over_a_flattened_chunk() raises:
    """The equivalence the step rests on, for the two operators it was added
    to. The same expression over a selected chunk and over the same chunk
    flattened first has to give the same rows."""
    var under = node_apply(
        Node(Compute(0, 1, BinaryOp.MUL, "product")), two_under_a_selection()
    )
    assert_true(under.__bool__(), "a chunk came back")
    var carried = under.take()
    var through = ints_of(carried.column(2), len(carried))

    var flat = two_under_a_selection()
    flat.flatten()
    var over = node_apply(Node(Compute(0, 1, BinaryOp.MUL, "product")), flat^)
    assert_true(over.__bool__(), "and one the other way")
    var plain = over.take()
    var moved = ints_of(plain.column(2), len(plain))

    assert_equal(len(through), 3, "three rows either way")
    assert_equal(len(moved), 3, "three the other way too")
    for i in range(3):
        assert_equal(through[i], moved[i], "the same row in the same place")
    assert_equal(through[0], 40, "2 times 20")
    assert_equal(through[2], 360, "and 6 times 60")


def test_materializing_a_column_leaves_it_where_it_can_be_found() raises:
    """The chunk method the two operators share. One column is gathered and put
    back, the rest of the chunk is untouched, and a second call on the same
    column does nothing at all."""
    var chunk = two_under_a_selection()
    chunk.materialize(0)
    assert_true(chunk.selected(), "one column still reads through it")
    assert_equal(len(chunk.columns[0]), 3, "gathered down to the rows")
    assert_equal(len(chunk.columns[1]), 6, "and the other one is as it was")
    chunk.materialize(0)
    var values = ints_of(chunk.columns[0], 3)
    assert_equal(values[0], 2, "not gathered through the positions again")
    assert_equal(values[2], 6, "nor this one")
    chunk.materialize(1)
    assert_false(chunk.selected(), "and with both gathered the selection goes")
    assert_equal(len(chunk), 3, "three rows, which is what it said")


def test_a_limit_cuts_the_selection_rather_than_the_columns() raises:
    """`LIMIT 2` after a filter used to gather every row the filter kept in
    order to throw all but two of them away. The positions are cut instead and
    the columns are left where they are, so what gets gathered later is two
    rows."""
    var node = Node(Limit(2))
    var out = node_process(node, two_under_a_selection())
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_true(got.selected(), "still under a selection")
    assert_equal(len(got), 2, "two rows of the three")
    assert_equal(len(got.columns[0]), 6, "over columns that still hold six")
    assert_equal(len(got.columns[1]), 6, "both of them")
    var values = ints_of(got.column(0), 2)
    assert_equal(values[0], 2, "the row at position 1")
    assert_equal(values[1], 4, "and the row at position 3")


def test_a_limit_that_skips_under_a_selection_skips_rows() raises:
    """The offset counts rows of the chunk and not positions in the arrays
    underneath, which is the thing that would be wrong if the two were
    confused."""
    var node = Node(Limit(1, 1))
    var out = node_process(node, two_under_a_selection())
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_true(got.selected(), "still under a selection")
    assert_equal(len(got), 1, "one row")
    var values = ints_of(got.column(0), 1)
    assert_equal(values[0], 4, "the second row, not the second value")


def test_a_limit_slices_a_dense_column_with_the_positions() raises:
    """A column already at the chunk's rows has to be cut where the selection
    is cut, since its element i is row i and the rows are what a limit takes."""
    var node = Node(Limit(2))
    var out = node_process(node, selected_masked_chunk())
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_true(got.selected(), "still under a selection")
    assert_equal(len(got), 2, "two rows")
    assert_equal(len(got.columns[0]), 6, "the values were left where they were")
    assert_equal(len(got.columns[1]), 2, "and the dense mask was sliced")
    var values = ints_of(got.column(0), 2)
    assert_equal(values[0], 2, "the row at position 1")
    assert_equal(values[1], 4, "and the row at position 3")
    var mask = truths_of(got.column(1), 2)
    assert_true(mask[0], "the first row of the mask")
    assert_false(mask[1], "and the second")


def test_a_limit_under_a_selection_agrees_with_one_over_a_flat_chunk() raises:
    """The equivalence again, for the operator this one was added to."""
    var node = Node(Limit(2, 1))
    var under = node_process(node, two_under_a_selection())
    assert_true(under.__bool__(), "a chunk came back")
    var carried = under.take()
    var through = ints_of(carried.column(0), len(carried))

    var flat = two_under_a_selection()
    flat.flatten()
    var again = Node(Limit(2, 1))
    var over = node_process(again, flat^)
    assert_true(over.__bool__(), "and one the other way")
    var plain = over.take()
    var moved = ints_of(plain.column(0), len(plain))

    assert_equal(len(through), 2, "two rows either way")
    assert_equal(len(moved), 2, "two the other way too")
    for i in range(2):
        assert_equal(through[i], moved[i], "the same row in the same place")
    assert_equal(through[0], 4, "the row after the one that was skipped")


def test_a_filter_can_do_its_own_comparison() raises:
    """The mask a filter reads was written by the compute underneath it and
    read by nobody else, so the filter may do the comparison itself and write no
    column at all. The rows it keeps are the rows the pair kept."""
    var out = node_apply(
        Node(Filter(0, Value(Int64(3)), BinaryOp.GT)), six_rows()
    )
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_equal(got.width(), 2, "no mask column was added")
    assert_equal(len(got), 3, "three rows above three")
    var kept = ints_of(got.column(0), 3)
    assert_equal(kept[0], 4, "the first row over the constant")
    assert_equal(kept[2], 6, "and the last")


def test_a_comparing_filter_takes_the_constant_on_either_side() raises:
    """`3 > x` is `x < 3` and the filter mirrors the comparison rather than
    having a second loop for it."""
    var out = node_apply(
        Node(Filter(0, Value(Int64(3)), BinaryOp.GT, value_on_left=True)),
        six_rows(),
    )
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_equal(len(got), 2, "the two rows under the constant")
    var kept = ints_of(got.column(0), 2)
    assert_equal(kept[0], 1, "the first")
    assert_equal(kept[1], 2, "and the second")


def test_a_comparing_filter_reads_its_operand_where_it_lies() raises:
    """The half that pays. A compute has to produce a column at the chunk's
    rows and so gathers its operand first, and a comparison that only wants to
    know which rows it keeps reads them through the positions and gathers
    nothing."""
    var out = node_apply(
        Node(Filter(1, Value(Int64(30)), BinaryOp.GT)), two_under_a_selection()
    )
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_true(got.selected(), "still under a selection")
    assert_equal(len(got.columns[0]), 6, "the first column was not gathered")
    assert_equal(len(got.columns[1]), 6, "nor was the one it compared")
    assert_equal(len(got), 2, "two of the three rows are over thirty")
    var kept = ints_of(got.column(1), 2)
    assert_equal(kept[0], 40, "the value at position 3")
    assert_equal(kept[1], 60, "and the one at position 5")


def test_a_comparing_filter_agrees_with_a_compute_and_a_filter() raises:
    """The equivalence this rests on. The same predicate written the long way,
    as a comparison into a column and a filter over that column, has to keep the
    same rows in the same order."""
    var fused = node_apply(
        Node(Filter(1, Value(Int64(30)), BinaryOp.GT)), two_under_a_selection()
    )
    assert_true(fused.__bool__(), "a chunk came back")
    var short = fused.take()
    var one = ints_of(short.column(0), len(short))

    var made = node_apply(
        Node(Compute(1, Value(Int64(30)), BinaryOp.GT, "hit")),
        two_under_a_selection(),
    )
    assert_true(made.__bool__(), "the mask was written")
    var long = node_apply(Node(Filter(2, [0, 1])), made.take())
    assert_true(long.__bool__(), "and the filter read it")
    var other = long.take()
    var two = ints_of(other.column(0), len(other))

    assert_equal(len(one), len(two), "the same number of rows")
    for i in range(len(one)):
        assert_equal(one[i], two[i], "the same row in the same place")
    assert_equal(one[0], 4, "the row at position 3")


def test_a_comparing_filter_drops_the_rows_with_nothing_in_them() raises:
    """A comparison against a null is null and a filter drops a row its mask is
    null on, so a row with nothing in it is not a row this keeps. The two routes
    have to agree about that or a predicate would mean one thing fused and
    another one not."""
    var column = Array[DType.int64](4)
    column.set_valid(0, Int64(9))
    column.set_null(1)
    column.set_valid(2, Int64(1))
    column.set_valid(3, Int64(9))
    var columns = List[AnyArray]()
    columns.append(AnyArray(column^))
    var out = node_apply(
        Node(Filter(0, Value(Int64(5)), BinaryOp.GT)), Chunk(columns^)
    )
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_equal(len(got), 2, "the missing row is not over five either")
    var kept = ints_of(got.column(0), 2)
    assert_equal(kept[0], 9, "the first row")
    assert_equal(kept[1], 9, "and the last")


def test_a_comparing_filter_that_keeps_nothing_emits_nothing() raises:
    """A chunk of no rows is work for everything downstream and no
    information, which is as true of a comparison that is never true as it is of
    a mask that is never set."""
    var out = node_apply(
        Node(Filter(0, Value(Int64(600)), BinaryOp.GT)), six_rows()
    )
    assert_false(out.__bool__(), "nothing came back")


def test_a_comparing_filter_asked_for_no_columns_counts() raises:
    """A filter that writes no columns is a row count, and a comparison has
    counted the rows it kept by the time it has found them."""
    var out = node_apply(
        Node(Filter(0, Value(Int64(2)), BinaryOp.GT, List[Int]())), six_rows()
    )
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_equal(got.width(), 0, "no columns")
    assert_equal(len(got), 4, "four rows over two")


def test_a_comparing_filter_writes_only_the_columns_it_is_asked_for() raises:
    """It is a filter and a projection in one pass whichever way it got its
    rows, and the column it compared is as droppable as a spent mask."""
    var out = node_apply(
        Node(Filter(1, Value(Int64(20)), BinaryOp.GT, [0])), six_rows()
    )
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_equal(got.width(), 1, "the compared column was not asked for")
    assert_equal(len(got), 4, "four rows over twenty")
    var kept = ints_of(got.column(0), 4)
    assert_equal(kept[0], 3, "the row beside the first value over twenty")


def test_a_comparing_filter_falls_back_for_a_pair_it_has_no_loop_for() raises:
    """Text, category and temporal columns compare perfectly well the ordinary
    way, and none of them is the shape the fused loops were written for. The
    filter builds the mask for those and reads the rows off it, so the answer is
    the same and only the cost differs."""
    var columns = List[AnyArray]()
    columns.append(
        AnyArray(strings_from_list(["ok", "fail", "ok", "ok", "fail", "ok"]))
    )
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    var out = node_apply(
        Node(Filter(0, Value(String("ok")), BinaryOp.EQ, [1])),
        Chunk(columns^),
    )
    assert_true(out.__bool__(), "a chunk came back")
    var got = out.take()
    assert_equal(len(got), 4, "four rows say ok")
    var kept = ints_of(got.column(0), 4)
    assert_equal(kept[0], 1, "the first of them")
    assert_equal(kept[3], 6, "and the last")


def test_a_comparing_filter_runs_in_a_pipeline() raises:
    """The same node with a scan under it and a frame on top, which is where it
    has to work rather than in a call by hand."""
    var pipeline = Pipeline(sample_frame())
    pipeline.add(Node(Filter(0, Value(Int64(2)), BinaryOp.GT)))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(len(got), 4, "four rows over two")
    assert_equal(got[0], 3, "the first of them")
    assert_equal(got[3], 6, "and the last")


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


def test_a_bounded_sort_gives_what_the_limit_above_would_have_taken() raises:
    # The whole claim. A sort told it only needs two rows and a limit of two
    # over an unbounded sort are the same two rows in the same order, and the
    # bounded one never builds a permutation of the six.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Sort([0], [True], [False], bound=2)))
    pipeline.add(Node(Limit(2)))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(len(got), 2, "rows")
    assert_equal(got[0], 6, "first")
    assert_equal(got[1], 5, "second")


def test_a_bound_covers_the_offset_as_well_as_the_length() raises:
    # The rows a limit skips still have to be found, so the bound the plan
    # writes is the offset plus the length. A bound of two under a limit that
    # skips one would answer with a row the sort never looked for.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Sort([0], [True], [False], bound=3)))
    pipeline.add(Node(Limit(2, 1)))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(len(got), 2, "rows")
    assert_equal(got[0], 5, "the second largest, since one was skipped")
    assert_equal(got[1], 4, "and the third")


def test_a_bound_wider_than_the_input_sorts_the_whole_thing() raises:
    # A bound that bounds nothing has to answer what an unbounded sort answers,
    # every row and not the first n of them, because the operator above it may
    # not be a limit at all.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Sort([0], [True], [False], bound=99)))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(len(got), 6, "every row")
    for i in range(6):
        assert_equal(got[i], Int64(6 - i), "row " + String(i))


def test_a_bounded_sort_over_two_keys_breaks_the_ties_the_same_way() raises:
    # The bounded route is a different kernel, so the tie rule is worth asking
    # again rather than assuming. The mask is the dominant key and the number
    # refines it, which is the unbounded test above with a bound on it.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Sort([1, 0], [False, True], [False, False], bound=3)))
    var out = pipeline^.run()
    var got = read_back(out, "n")
    assert_equal(len(got), 3, "the three the bound asked for")
    assert_equal(got[0], 5, "the largest of the two it dropped")
    assert_equal(got[1], 2, "then the other")
    assert_equal(got[2], 6, "then the largest it kept")


def test_a_bounded_sort_puts_the_nulls_where_it_was_told_to() raises:
    # Two rows of the six are missing and the nulls are asked for first, so a
    # bound of two answers with the two rows that have nothing in them.
    var pipeline = Pipeline(gappy_frame())
    pipeline.add(Node(Sort([0], [False], [True], bound=2)))
    var out = pipeline^.run()
    var there = present(out, "n")
    assert_equal(len(there), 2, "rows")
    assert_false(there[0], "the first is missing")
    assert_false(there[1], "and so is the second")


def test_a_bound_of_nothing_answers_nothing() raises:
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Sort([0], [True], [False], bound=0)))
    var out = pipeline^.run()
    assert_equal(len(out), 0, "rows")
    assert_equal(out.width(), 2, "and the schema still describes the result")


def test_a_bounded_sort_cuts_its_answer_at_the_chunks_it_was_given() raises:
    # A sort hands back the chunk sizes it was given, and a bounded one hands
    # back as many of them as the bound reached. The frame arrives as two, three
    # and one, so a bound of four is the first chunk and most of the second.
    var pipeline = Pipeline(cut_frame())
    pipeline.add(Node(Sort([0], [True], [False], bound=4)))
    var out = pipeline^.run()
    assert_equal(len(out), 4, "rows")
    assert_equal(out.columns[0].num_chunks(), 2, "two of the three")


def test_a_sort_bounded_at_a_number_that_is_not_one_is_refused() raises:
    with assert_raises(contains="is not a number of rows to keep"):
        _ = Sort([0], [True], [False], bound=-2)


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


def test_a_window_over_a_partition_of_nothing_but_nulls_sums_to_zero() raises:
    # The pandas answer, which is what an unmarked window asks for. The first
    # partition holds a five and the second holds three nulls, and a sum that
    # added nothing is holding the zero it started at.
    var pipeline = Pipeline(hollow_frame())
    pipeline.add(Node(Window([1], [0], [AggKind.SUM], ["total"])))
    var out = pipeline^.run()
    assert_equal(read_back(out, "total"), [5, 0, 0, 0], "one per row")


def test_a_marked_window_over_a_partition_of_nothing_but_nulls_is_null() raises:
    # The same windows with the mark the SQL front end sets. A partition is
    # never empty, so the only thing the mark can decide here is whether the
    # rows it did hold had anything in them, and these three did not. See #877.
    var pipeline = Pipeline(hollow_frame())
    pipeline.add(Node(Window([1], [0], [AggKind.SUM], ["total"], [True])))
    var out = pipeline^.run()
    assert_equal(
        present(out, "total"), [True, False, False, False], "one per row"
    )
    assert_equal(read_back(out, "total")[0], 5, "the partition that had one")


def test_a_marked_window_over_a_partition_of_nans_is_null_too() raises:
    # The column the schema says holds no null, holding four NaNs. A NaN is not
    # a value to a count either, so the sum has nothing and answers null.
    var pipeline = Pipeline(hollow_frame())
    pipeline.add(
        Node(Window(List[Int](), [3], [AggKind.SUM], ["total"], [True]))
    )
    var out = pipeline^.run()
    var total = out.column("total").as_typed[DType.float64]()
    for i in range(4):
        assert_true(not total.is_valid(i), "row " + String(i))


def test_a_marked_window_that_added_something_is_the_ordinary_sum() raises:
    # The mark costs a count and changes nothing where a value turned up, which
    # is the case that has to keep working.
    var pipeline = Pipeline(hollow_frame())
    pipeline.add(
        Node(Window(List[Int](), [0], [AggKind.SUM], ["total"], [True]))
    )
    var out = pipeline^.run()
    assert_equal(read_back(out, "total"), [5, 5, 5, 5], "the one value there")


def test_a_marked_count_window_still_answers_zero() raises:
    # A count answers a number over nothing on both sides, so the SQL front end
    # leaves it unmarked and the mark would not reach it anyway.
    var pipeline = Pipeline(hollow_frame())
    pipeline.add(Node(Window([1], [0], [AggKind.COUNT], ["how_many"], [True])))
    var out = pipeline^.run()
    assert_equal(read_back(out, "how_many"), [1, 0, 0, 0], "one per row")


def test_a_window_told_the_wrong_number_of_marks_is_refused() raises:
    with assert_raises(contains="said whether folding nothing answers null"):
        _ = Window(
            List[Int](), [0, 0], [AggKind.SUM, AggKind.SUM], ["a", "b"], [True]
        )


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
