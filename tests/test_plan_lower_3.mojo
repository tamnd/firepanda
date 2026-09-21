"""Tests for turning a logical plan into a pipeline.

Every test here builds a plan, binds it, lowers it and runs it, and checks the
rows that come out. Checking the rows rather than the shape of the pipeline is
deliberate: the number of operators a plan lowers to is an implementation
detail that a later pass is allowed to change, and the rows are not. The two
tests that do look at the operator count are the ones where the count is the
point, which is the conjunction becoming a line of filters.

The other half is the refusals. Lowering is allowed to say no, and what it says
no to is the list of things nobody has written an operator for yet, so a test
per refusal is what stops one of them being quietly half lowered into a wrong
answer later.

Part 3 of 3. The fixtures are in tests/support/plan_lower.mojo.
"""


from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.unary import UnaryOp
from firepanda.plan.bind import bind
from firepanda.plan.limits import limits
from firepanda.plan.lower import lower
from firepanda.plan.merge import merge
from firepanda.plan.node import NO_LIMIT, SET_EXCEPT, SET_INTERSECT, Plan
from firepanda.plan.simplify import simplify

from tests.support.plan_lower import (
    copies,
    counted,
    crate_frames,
    crate_join,
    crate_schemas,
    crated,
    crates,
    decimals,
    echoes,
    emptied,
    gappy,
    gauge_frame,
    gauge_schemas,
    gauges,
    holey,
    joined,
    numbers,
    one_frame,
    present,
    read_back,
    run,
    run_frames,
    run_pair,
    run_two,
    sales,
    same,
    schemas,
    series,
    shift_frame,
    shift_schemas,
    shifts,
    tiers,
    truths,
    two_frames,
    two_schemas,
    valid,
)


def test_a_cross_join_onto_one_row_is_a_column_per_right_column() raises:
    # One right row moves nothing, so every left row comes back in its own
    # order with the right row's values beside it.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var one = plan.filter(
        right,
        plan.exprs.binary(
            BinaryOp.EQ,
            plan.exprs.column("band"),
            plan.exprs.literal(Value(Int64(3))),
        ),
    )
    var root = plan.join(left, one, List[Int](), List[Int](), JoinKind.CROSS)
    var out = run_two(plan, root)
    same(read_back(out, "qty"), [5, 20, 3, 40, 12, 8, 25, 1, 30, 15], "qty")
    same(read_back(out, "band"), [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], "band")
    same(
        read_back(out, "rate"),
        [300, 300, 300, 300, 300, 300, 300, 300, 300, 300],
        "rate",
    )


def test_a_cross_join_onto_no_rows_is_refused_rather_than_empty() raises:
    # An empty right side makes the whole answer empty, which is a row count
    # this operator cannot produce, since it adds a column and keeps the rows.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var none = plan.filter(
        right,
        plan.exprs.binary(
            BinaryOp.GT,
            plan.exprs.column("band"),
            plan.exprs.literal(Value(Int64(1000))),
        ),
    )
    var root = plan.join(left, none, List[Int](), List[Int](), JoinKind.CROSS)
    _ = bind(plan, root, two_schemas())
    with assert_raises(contains="right side of 0 rows"):
        _ = lower(plan, root, two_frames())


def test_a_join_whose_right_side_is_a_filter_builds_it_first() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var keep = plan.exprs.binary(
        BinaryOp.GT,
        plan.exprs.column("band"),
        plan.exprs.literal(Value(Int64(5))),
    )
    var root = plan.join(
        left,
        plan.filter(right, keep),
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        JoinKind.INNER,
    )
    var out = run_two(plan, root)
    same(read_back(out, "qty"), [20, 40], "qty")
    same(read_back(out, "rate"), [200, 400], "rate")


def test_a_join_can_build_from_a_projection() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var narrowed = plan.project(
        right,
        [plan.exprs.column("rate"), plan.exprs.column("band")],
        [String("rate"), String("band")],
    )
    var root = plan.join(
        left,
        narrowed,
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        JoinKind.INNER,
    )
    var out = run_two(plan, root)
    same(read_back(out, "qty"), [20, 3, 40], "qty")
    same(read_back(out, "rate"), [200, 300, 400], "rate")


def test_a_build_side_takes_the_relation_it_read() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("sales", List[String](), 0)
    var keep = plan.exprs.binary(
        BinaryOp.GT,
        plan.exprs.column("qty"),
        plan.exprs.literal(Value(Int64(5))),
    )
    var root = plan.join(
        left,
        plan.filter(right, keep),
        [plan.exprs.column("qty")],
        [plan.exprs.column("qty")],
        JoinKind.SEMI,
    )
    _ = bind(plan, root, two_schemas())
    with assert_raises(contains="both read relation 0"):
        _ = lower(plan, root, two_frames())


def test_a_build_side_that_cannot_be_lowered_says_what_it_was() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var computed = plan.exprs.binary(
        BinaryOp.ADD, plan.exprs.column("band"), plan.exprs.column("rate")
    )
    var root = plan.join(
        left,
        plan.distinct(right, [computed]),
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        JoinKind.INNER,
    )
    _ = bind(plan, root, two_schemas())
    with assert_raises(contains="decides on a binary expression"):
        _ = lower(plan, root, two_frames())


def test_a_join_on_two_key_pairs_that_agree_on_nothing_is_empty() raises:
    # Three sales rows have a quantity one of the bands matches, and none of
    # the three has the rate that band charges. So the pairing makes three
    # pairs and the second key drops all three, which is the case where every
    # chunk the filter sees comes back empty.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("qty"), plan.exprs.column("price")],
        [plan.exprs.column("band"), plan.exprs.column("rate")],
        JoinKind.INNER,
    )
    var out = run_two(plan, root)

    assert_equal(len(out), 0, "no row agrees on both")
    assert_equal(out.width(), 4, "and the schema is still both sides")


def test_a_computed_key_is_refused_by_name() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var key = plan.exprs.binary(
        BinaryOp.ADD,
        plan.exprs.column("qty"),
        plan.exprs.literal(Value(Int64(0))),
    )
    var root = plan.join(
        left, right, [key], [plan.exprs.column("band")], JoinKind.INNER
    )
    _ = bind(plan, root, two_schemas())
    with assert_raises(contains="joins on a column on each side"):
        _ = lower(plan, root, two_frames())


def test_a_name_both_sides_have_comes_back_twice() raises:
    # A join binds to the two schemas end to end, names and all, and the
    # operator is told that by position rather than left to work it out from
    # names it has two of. This used to be refused.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("echoes", List[String](), 1)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("qty")],
        [plan.exprs.column("qty")],
        JoinKind.INNER,
    )
    var schemas = List[Schema]()
    schemas.append(Schema(copy=sales().schema))
    schemas.append(Schema(copy=echoes().schema))
    var bound = bind(plan, root, schemas)
    var frames = List[DataFrame]()
    frames.append(sales())
    frames.append(echoes())
    var out = lower(plan, root, frames^).run()
    assert_equal(len(out.schema), 4, "qty and price from each side")
    assert_equal(out.schema[0].name, "qty")
    assert_equal(out.schema[1].name, "price")
    assert_equal(out.schema[2].name, "qty", "the right one, unrenamed")
    assert_equal(out.schema[3].name, "price", "and this one too")
    assert_equal(len(bound), 4, "which is what the plan bound to")
    assert_equal(len(out), 3, "three quantities are in both")


def test_two_scans_of_one_relation_say_so() raises:
    # A relation is a frame, so a table joined to itself is two relations and
    # two frames. A plan that says one of each is a plan that was built wrong,
    # and saying that here beats a missing column further down.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("sales", List[String](), 0)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("qty")],
        [plan.exprs.column("qty")],
        JoinKind.INNER,
    )
    _ = bind(plan, root, schemas())
    with assert_raises(contains="both read relation 0"):
        _ = lower(plan, root, one_frame())


def test_a_literal_table_is_the_rows_it_names() raises:
    var plan = Plan()
    var root = plan.values(
        [
            plan.exprs.literal(Value(Int64(1))),
            plan.exprs.literal(Value(Int64(2))),
            plan.exprs.literal(Value(Int64(3))),
            plan.exprs.literal(Value(Int64(4))),
        ],
        [String("a"), String("b")],
    )
    _ = bind(plan, root, List[Schema]())
    var pipe = lower(plan, root, List[DataFrame]())
    var out = pipe^.run()
    same(read_back(out, "a"), [1, 3], "the first column")
    same(read_back(out, "b"), [2, 4], "the second")


def test_a_literal_table_can_be_filtered_and_projected() raises:
    var plan = Plan()
    var table = plan.values(
        [
            plan.exprs.literal(Value(Int64(1))),
            plan.exprs.literal(Value(Int64(9))),
            plan.exprs.literal(Value(Int64(2))),
            plan.exprs.literal(Value(Int64(8))),
            plan.exprs.literal(Value(Int64(3))),
            plan.exprs.literal(Value(Int64(7))),
        ],
        [String("a"), String("b")],
    )
    var keep = plan.exprs.binary(
        BinaryOp.GT,
        plan.exprs.column("a"),
        plan.exprs.literal(Value(Int64(1))),
    )
    var total = plan.exprs.binary(
        BinaryOp.ADD, plan.exprs.column("a"), plan.exprs.column("b")
    )
    var root = plan.project(
        plan.filter(table, keep), [total], [String("total")]
    )
    _ = bind(plan, root, List[Schema]())
    var pipe = lower(plan, root, List[DataFrame]())
    var out = pipe^.run()
    same(read_back(out, "total"), [10, 10], "the sums of the rows kept")


def test_a_literal_table_holds_a_missing_value() raises:
    var plan = Plan()
    var root = plan.values(
        [
            plan.exprs.literal(Value(Int64(1))),
            plan.exprs.literal(Value(null=LogicalType.INT64)),
        ],
        [String("a")],
    )
    _ = bind(plan, root, List[Schema]())
    var pipe = lower(plan, root, List[DataFrame]())
    var out = pipe^.run()
    valid(present(out, "a"), [True, False], "the second row is missing")


def test_a_union_all_of_two_scans_is_one_on_top_of_the_other() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.union([top, bottom], all=True)
    var out = run_two(plan, root)

    assert_equal(out.width(), 1, "one column")
    same(
        read_back(out, "qty"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15, 3, 20, 40, 99],
        "the first input then the second",
    )


def test_a_union_takes_the_first_input_s_names() raises:
    """The two sides call their column different things, and a union lines its
    inputs up by position, so the answer is called what the first one calls
    it."""
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.union([top, bottom], all=True)
    var out = run_two(plan, root)

    assert_equal(out.schema[0].name, "qty", "the first input's name")


def test_a_union_without_all_keeps_one_of_each_row() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.union([top, bottom], all=False)
    var out = run_two(plan, root)

    # 3, 20 and 40 are in both inputs, and 99 is only in the second.
    same(
        read_back(out, "qty"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15, 99],
        "one of each, in the order each was first seen",
    )


def test_a_union_of_three_inputs_is_one_node() raises:
    """Stacking is associative, so a chain of unions is one node with a list of
    inputs rather than a nest of two input nodes."""
    var plan = Plan()
    var a = plan.scan("tiers", ["band"], 1)
    var b = plan.scan("sales", ["qty"], 0)
    var c = plan.values(
        [
            plan.exprs.literal(Value(Int64(7))),
            plan.exprs.literal(Value(Int64(8))),
        ],
        [String("n")],
    )
    var root = plan.union([a, b, c], all=True)
    var out = run_two(plan, root)

    same(
        read_back(out, "band"),
        [3, 20, 40, 99, 5, 20, 3, 40, 12, 8, 25, 1, 30, 15, 7, 8],
        "all three in order",
    )


def test_each_input_of_a_union_may_be_a_query_of_its_own() raises:
    var plan = Plan()
    var scan = plan.scan("sales", ["qty"], 0)
    var qty = plan.exprs.column("qty")
    var big = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(25)))
    )
    var top = plan.filter(scan, big)
    var other = plan.scan("tiers", ["band"], 1)
    var band = plan.exprs.column("band")
    var small = plan.exprs.binary(
        BinaryOp.LT, band, plan.exprs.literal(Value(Int64(10)))
    )
    var bottom = plan.filter(other, small)
    var root = plan.union([top, bottom], all=True)
    var out = run_two(plan, root)

    same(read_back(out, "qty"), [40, 30, 3], "what each side kept")


def test_a_union_is_a_source_the_rest_of_the_line_reads() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var stacked = plan.union([top, bottom], all=True)
    var qty = plan.exprs.column("qty")
    var big = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(25)))
    )
    var root = plan.filter(stacked, big)
    var out = run_two(plan, root)

    same(read_back(out, "qty"), [40, 30, 40, 99], "the filter over the stack")


def test_a_union_whose_inputs_are_different_widths_is_refused_by_name() raises:
    """Binding refuses this before lowering sees it, which is the earlier and
    the better place, so the check in lowering is the one that catches a plan
    that was never bound rather than this."""
    var plan = Plan()
    var top = plan.scan("sales", List[String](), 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.union([top, bottom], all=True)

    with assert_raises(contains="a union is between a 2 column input"):
        _ = bind(plan, root, two_schemas())


def test_a_difference_keeps_the_rows_the_other_side_lacks() raises:
    # The bands are 3, 20, 40 and 99, so the three quantities that are also
    # bands go and the rest stay in the order the left side had them.
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_EXCEPT, all=False)
    var out = run_two(plan, root)

    assert_equal(out.width(), 1, "the tag the stack carried is gone")
    same(read_back(out, "qty"), [5, 12, 8, 25, 1, 30, 15], "what is left")


def test_a_difference_keeps_one_copy_of_a_row_it_keeps() raises:
    # A difference is over sets, so a quantity the left side wrote twice comes
    # back once. The tens of the quantities repeat and none of them is a band.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var ones = plan.exprs.binary(
        BinaryOp.MOD,
        plan.exprs.column("qty"),
        plan.exprs.literal(Value(Int64(10))),
    )
    var top = plan.project(scan, [ones], [String("ones")])
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_EXCEPT, all=False)
    var out = run_two(plan, root)

    same(read_back(out, "ones"), [5, 0, 2, 8, 1], "the threes went with 3")


def test_an_intersection_keeps_the_rows_both_sides_have() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_INTERSECT, all=False)
    var out = run_two(plan, root)

    assert_equal(out.width(), 1, "the tag the stack carried is gone")
    same(read_back(out, "qty"), [20, 3, 40], "99 is a band and not a quantity")


def test_a_set_operation_decides_on_every_column_of_the_row() raises:
    # Two columns rather than one, so a row that matches on the first and not
    # on the second is a row the other side does not have.
    var plan = Plan()
    var top = plan.scan("sales", ["qty", "price"], 0)
    var bottom = plan.values(
        [
            plan.exprs.literal(Value(Int64(20))),
            plan.exprs.literal(Value(Int64(2))),
            plan.exprs.literal(Value(Int64(40))),
            plan.exprs.literal(Value(Int64(7))),
        ],
        [String("qty"), String("price")],
    )
    var root = plan.setop([top, bottom], SET_INTERSECT, all=False)
    var out = run_two(plan, root)

    same(read_back(out, "qty"), [20], "40 goes with a price of 1 and not 7")
    same(read_back(out, "price"), [2], "and the price came along")


def test_a_difference_treats_two_nulls_as_the_same_row() raises:
    # Which is the rule a set operation has and a join does not, and the reason
    # this is a group by over a stack rather than an anti join.
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_EXCEPT, all=False)
    var frames = List[DataFrame]()
    frames.append(holey("qty", [5, 0, 20], [1]))
    frames.append(holey("band", [0, 20], [0]))
    var out = run_frames(plan, root, frames^)

    assert_equal(len(out), 1, "the null went with the other side's null")
    same(read_back(out, "qty"), [5], "and 5 is all that is left")


def test_an_intersection_keeps_a_null_both_sides_have() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_INTERSECT, all=False)
    var frames = List[DataFrame]()
    frames.append(holey("qty", [5, 0, 20], [1]))
    frames.append(holey("band", [0, 20], [0]))
    var out = run_frames(plan, root, frames^)

    valid(present(out, "qty"), [False, True], "the null is one of the two")
    assert_equal(read_back(out, "qty")[1], 20, "and 20 is the other")


def test_a_difference_written_all_subtracts_a_copy_at_a_time() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_EXCEPT, all=True)
    var out = run_two(plan, root)

    # No quantity is written twice, so ALL and the set answer agree here and
    # the test below is the one that tells them apart.
    same(read_back(out, "qty"), [5, 12, 8, 25, 1, 30, 15], "qty")


def test_a_difference_written_all_keeps_the_copies_over() raises:
    var plan = Plan()
    var top = plan.scan("copies", ["band"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_EXCEPT, all=True)
    var out = run_pair(plan, root, copies(), tiers())

    # Two threes on the left and one on the right leaves one three, and the
    # set answer leaves none.
    same(read_back(out, "band"), [3, 77], "band")


def test_an_intersection_written_all_keeps_the_thinner_count() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_INTERSECT, all=True)
    var out = run_two(plan, root)

    same(read_back(out, "qty"), [20, 3, 40], "qty")


def test_an_intersection_written_all_over_itself_is_itself() raises:
    var plan = Plan()
    var top = plan.scan("copies", ["band"], 0)
    var bottom = plan.scan("copies", ["band"], 1)
    var root = plan.setop([top, bottom], SET_INTERSECT, all=True)
    var out = run_pair(plan, root, copies(), copies())

    # Every count is the same on both sides, so the smaller of the two is the
    # count itself and the answer is the arm back again.
    same(read_back(out, "band"), [3, 3, 20, 77], "band")


def test_a_range_of_one_argument_starts_at_zero() raises:
    var plan = Plan()
    var root = counted(plan, "range", [5])
    var out = series(plan, root)
    same(read_back(out, "i"), [0, 1, 2, 3, 4], "five rows and no table")


def test_a_generate_series_stops_on_its_bound() raises:
    # The whole difference between the two functions is this one row.
    var plan = Plan()
    var root = counted(plan, "generate_series", [5])
    var out = series(plan, root)
    same(read_back(out, "i"), [0, 1, 2, 3, 4, 5], "six rows")


def test_a_range_of_two_arguments_is_a_start_and_a_stop() raises:
    var plan = Plan()
    var root = counted(plan, "range", [10, 14])
    var out = series(plan, root)
    same(read_back(out, "i"), [10, 11, 12, 13], "from the first to the second")


def test_a_range_of_three_arguments_counts_by_the_third() raises:
    var plan = Plan()
    var root = counted(plan, "range", [0, 10, 3])
    var out = series(plan, root)
    same(read_back(out, "i"), [0, 3, 6, 9], "and stops before ten")


def test_a_negative_step_counts_down() raises:
    var plan = Plan()
    var root = counted(plan, "range", [5, 0, -2])
    var out = series(plan, root)
    same(read_back(out, "i"), [5, 3, 1], "down to the bound and not past it")


def test_a_series_that_never_reaches_its_end_is_no_rows() raises:
    var plan = Plan()
    var root = counted(plan, "range", [5, 0])
    var out = series(plan, root)
    assert_equal(len(out), 0, "counting up from five to zero is nothing")


def test_a_step_of_zero_is_refused_by_name() raises:
    var plan = Plan()
    var root = counted(plan, "range", [0, 10, 0])
    _ = bind(plan, root, List[Schema]())

    with assert_raises(contains="a series that never moves"):
        _ = lower(plan, root, List[DataFrame]())


def test_a_series_with_a_null_end_is_no_rows() raises:
    var plan = Plan()
    var missing = plan.exprs.literal(Value(null=LogicalType.NULL))
    var root = plan.table_function("range", [missing], ["i"])
    var out = series(plan, root)
    assert_equal(len(out), 0, "nobody said where it ends")


def test_a_series_that_is_too_long_to_build_is_refused_by_name() raises:
    # It is built before the query starts, so a series of a trillion rows is an
    # allocation rather than a query that takes a while.
    var plan = Plan()
    var root = counted(plan, "range", [1_000_000_000_000])
    _ = bind(plan, root, List[Schema]())

    with assert_raises(contains="longer than 100000000"):
        _ = lower(plan, root, List[DataFrame]())


def test_a_series_takes_the_name_the_call_gave_it() raises:
    var plan = Plan()
    var three = plan.exprs.literal(Value(Int64(3)))
    var root = plan.table_function("range", [three], ["day"])
    var out = series(plan, root)
    same(read_back(out, "day"), [0, 1, 2], "under the name the call chose")


def test_a_series_is_a_source_the_rest_of_the_line_reads() raises:
    var plan = Plan()
    var rows = counted(plan, "range", [10])
    var i = plan.exprs.column("i")
    var four = plan.exprs.literal(Value(Int64(4)))
    var big = plan.exprs.binary(BinaryOp.GT, i, four)
    var kept = plan.filter(rows, big)
    var out = series(plan, kept)
    same(read_back(out, "i"), [5, 6, 7, 8, 9], "the filter ran over the series")


def test_an_argument_that_is_computed_is_folded_before_it_gets_here() raises:
    # Lowering reads the arguments off the tree, so `range(2 + 3)` only works
    # because simplify has already folded it. One that is not folded is refused
    # by name rather than computed, the same way a VALUES refuses one.
    var plan = Plan()
    var two = plan.exprs.literal(Value(Int64(2)))
    var three = plan.exprs.literal(Value(Int64(3)))
    var sum = plan.exprs.binary(BinaryOp.ADD, two, three)
    var root = plan.table_function("range", [sum], ["i"])
    _ = bind(plan, root, List[Schema]())

    with assert_raises(contains="argument 1 of range is a binary expression"):
        _ = lower(plan, root, List[DataFrame]())


def test_a_window_over_the_whole_frame_broadcasts_one_value() raises:
    # No partition keys is one partition, so every row gets the same answer and
    # the columns the frame had come through beside it.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var total = plan.exprs.window(AggKind.SUM, qty, List[Int](), List[Int]())
    var root = plan.window(scan, [total], ["total"])
    var got = run(plan, root)

    assert_equal(len(got.schema), 3, "the window adds a column")
    assert_equal(got.schema[2].name, "total", "and it is called what it asked")
    same(
        read_back(got, "qty"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15],
        "the rows come back as they were",
    )
    same(
        read_back(got, "total"),
        [159, 159, 159, 159, 159, 159, 159, 159, 159, 159],
        "every row gets the sum of all of them",
    )


def test_a_window_partitions_and_each_row_reads_its_own() raises:
    # Two windows over one partitioning, which is one grouping pass and one
    # operator. The rows stay where they were and each one gets the answer for
    # the team it is on.
    var plan = Plan()
    var scan = plan.scan("shifts", List[String](), 0)
    var team = plan.exprs.column("team")
    var hours = plan.exprs.column("hours")
    var worked = plan.exprs.window(AggKind.SUM, hours, [team], List[Int]())
    var shifts = plan.exprs.window(AggKind.COUNT, hours, [team], List[Int]())
    var root = plan.window(scan, [worked, shifts], ["worked", "shifts"])
    _ = bind(plan, root, shift_schemas())
    var pipe = lower(plan, root, shift_frame())
    var got = pipe^.run()

    assert_equal(len(got.schema), 4, "two windows on two columns")
    same(
        read_back(got, "team"),
        [1, 2, 1, 2, 3, 1, 3, 2, 1],
        "the rows come back in the order they arrived",
    )
    same(
        read_back(got, "worked"),
        [13, 20, 13, 20, 12, 13, 12, 20, 13],
        "each row reads its own team's hours",
    )
    same(
        read_back(got, "shifts"),
        [4, 3, 4, 3, 2, 4, 2, 3, 4],
        "and its own team's count",
    )


def test_a_window_over_a_computed_expression_drops_the_intermediate() raises:
    # The product is appended before the breaker and the window lands after it,
    # so what the node hands up is the frame's columns and then the window, with
    # the column in between cut out.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var line = plan.exprs.binary(BinaryOp.MUL, qty, price)
    var total = plan.exprs.window(AggKind.SUM, line, List[Int](), List[Int]())
    var root = plan.window(scan, [total], ["revenue"])
    var got = run(plan, root)

    assert_equal(len(got.schema), 3, "the product is not one of the columns")
    assert_equal(
        got.schema[0].name, "qty", "the frame's columns keep their place"
    )
    assert_equal(got.schema[1].name, "price", "both of them")
    same(
        read_back(got, "revenue"),
        [668, 668, 668, 668, 668, 668, 668, 668, 668, 668],
        "every row gets the revenue of all of them",
    )


def test_a_column_above_a_window_reads_what_the_window_added() raises:
    # The node above is bound against the input's columns and then the window's,
    # so a projection that asks for the last one has to find it there.
    var plan = Plan()
    var scan = plan.scan("shifts", List[String](), 0)
    var team = plan.exprs.column("team")
    var hours = plan.exprs.column("hours")
    var worked = plan.exprs.window(AggKind.SUM, hours, [team], List[Int]())
    var window = plan.window(scan, [worked], ["worked"])
    var back = plan.exprs.column("worked")
    var root = plan.project(window, [back], ["worked"])
    _ = bind(plan, root, shift_schemas())
    var pipe = lower(plan, root, shift_frame())
    var got = pipe^.run()

    assert_equal(len(got.schema), 1, "the projection keeps the one column")
    same(
        read_back(got, "worked"),
        [13, 20, 13, 20, 12, 13, 12, 20, 13],
        "and it is the column the window added",
    )


def test_two_windows_that_partition_differently_are_refused() raises:
    # One operator makes one set of ordinals. Splitting the node here would put
    # the second window in a column the node above was not bound against, so
    # the split belongs to the plan and this says so.
    var plan = Plan()
    var scan = plan.scan("shifts", List[String](), 0)
    var team = plan.exprs.column("team")
    var hours = plan.exprs.column("hours")
    var by_team = plan.exprs.window(AggKind.SUM, hours, [team], List[Int]())
    var overall = plan.exprs.window(
        AggKind.SUM, hours, List[Int](), List[Int]()
    )
    var root = plan.window(scan, [by_team, overall], ["worked", "total"])
    _ = bind(plan, root, shift_schemas())

    with assert_raises(contains="partitions differently from the first one"):
        _ = lower(plan, root, shift_frame())


def test_an_ordered_window_is_refused_by_name() raises:
    # An ordering inside the window is a running fold, which is a different loop
    # rather than an argument to this one.
    var plan = Plan()
    var scan = plan.scan("shifts", List[String](), 0)
    var team = plan.exprs.column("team")
    var hours = plan.exprs.column("hours")
    var running = plan.exprs.window(AggKind.SUM, hours, [team], [hours])
    var root = plan.window(scan, [running], ["running"])
    _ = bind(plan, root, shift_schemas())

    with assert_raises(contains="is ordered, and an ordered window"):
        _ = lower(plan, root, shift_frame())


def test_a_conditional_picks_between_two_columns() raises:
    # The plain shape. A condition over one column, a column on each side, and
    # the answer is one column with rows from both.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var ten = plan.exprs.literal(Value(Int64(10)))
    var over = plan.exprs.binary(BinaryOp.GT, qty, ten)
    var picked = plan.exprs.conditional(over, qty, price)
    var root = plan.project(scan, [picked], ["taken"])
    var out = run(plan, root)

    assert_equal(len(out.schema), 1, "the projection keeps the one column")
    same(
        read_back(out, "taken"),
        [10, 20, 7, 40, 12, 9, 25, 100, 30, 15],
        "the quantity over ten and the price otherwise",
    )


def test_a_null_condition_takes_the_else_side() raises:
    # The rule worth pinning. A row the question could not be asked about is a
    # row the question did not hold for, which is what SQL says and is not what
    # an operation would say, since every one of those answers a null instead.
    var plan = Plan()
    var scan = plan.scan("gauges", List[String](), 0)
    var a = plan.exprs.column("a")
    var b = plan.exprs.column("b")
    var pair = plan.exprs.binary(BinaryOp.EQ, a, b)
    var seven = plan.exprs.literal(Value(Int64(7)))
    var nine = plan.exprs.literal(Value(Int64(9)))
    var picked = plan.exprs.conditional(pair, seven, nine)
    var root = plan.project(scan, [picked], ["taken"])
    _ = bind(plan, root, gauge_schemas())
    var pipe = lower(plan, root, gauge_frame())
    var out = pipe^.run()

    # Four of the five pairs have a null on one side or the other, so only the
    # last row asks a question that has an answer.
    same(
        read_back(out, "taken"),
        [9, 9, 9, 9, 7],
        "the else side wherever the condition was null",
    )


def test_a_chain_of_whens_is_a_chain_of_nodes() raises:
    # A conditional's else side is another conditional, which is how the parser
    # already builds a chain, so nothing here counts branches.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var high = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(20)))
    )
    var some = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(5)))
    )
    var inner = plan.exprs.conditional(
        some,
        plan.exprs.literal(Value(Int64(2))),
        plan.exprs.literal(Value(Int64(1))),
    )
    var outer = plan.exprs.conditional(
        high, plan.exprs.literal(Value(Int64(3))), inner
    )
    var root = plan.project(scan, [outer], ["band"])
    var out = run(plan, root)

    same(
        read_back(out, "band"),
        [1, 2, 1, 3, 2, 2, 3, 1, 3, 2],
        "three bands over the quantity",
    )


def test_a_conditional_over_two_types_casts_the_side_that_moves() raises:
    # Binding promotes the two sides and the node needs them to agree, so the
    # integer column is converted into a column of its own. In its own column
    # rather than where it lies, because the scan's column is still the scan's.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var ten = plan.exprs.literal(Value(Int64(10)))
    var over = plan.exprs.binary(BinaryOp.GT, qty, ten)
    var half = plan.exprs.literal(Value(Float64(0.5)))
    var picked = plan.exprs.conditional(over, qty, half)
    var root = plan.project(scan, [picked], ["taken"])
    var out = run(plan, root)

    assert_equal(
        out.schema[0].dtype,
        LogicalType.FLOAT64,
        "the two sides promote to the wider one",
    )
    var got = decimals(out, "taken")
    var want = [0.5, 20.0, 0.5, 40.0, 12.0, 0.5, 25.0, 0.5, 30.0, 15.0]
    assert_equal(len(got), len(want), "how many rows")
    for i in range(len(want)):
        assert_equal(got[i], want[i], "taken at " + String(i))


def test_a_conditional_may_be_a_predicate() raises:
    # The answer is a boolean column like any other, so a filter can read it.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var over = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(10)))
    )
    var dear = plan.exprs.binary(
        BinaryOp.GT, price, plan.exprs.literal(Value(Int64(5)))
    )
    var asked = plan.exprs.conditional(
        over, dear, plan.exprs.literal(Value(False))
    )
    var root = plan.filter(scan, asked)
    var out = run(plan, root)

    # Six rows are over ten and one of those six costs more than five.
    same(read_back(out, "qty"), [15], "the one row both halves kept")


def test_each_side_of_a_conditional_may_be_an_expression() raises:
    # Both sides lower the way anything else does, so an expression on each is
    # two lines of appends and then the choice between the two columns.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var over = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(10)))
    )
    var less = plan.exprs.binary(BinaryOp.SUB, qty, price)
    var more = plan.exprs.binary(BinaryOp.ADD, qty, price)
    var picked = plan.exprs.conditional(over, less, more)
    var root = plan.project(scan, [picked], ["net"])
    var out = run(plan, root)

    same(
        read_back(out, "net"),
        [15, 18, 10, 39, 7, 17, 22, 101, 26, 9],
        "the difference over ten and the sum otherwise",
    )


def test_a_join_on_two_keys_pairs_on_both_of_them() raises:
    var plan = Plan()
    var root = crate_join(plan, JoinKind.INNER)
    var out = run_frames(plan, root, crate_frames())

    assert_equal(out.width(), 5, "both schemas end to end")
    same(read_back(out, "qty"), [5, 3, 40], "the rows both keys agree on")
    same(read_back(out, "kept"), [100, 300, 200], "paired with the right row")


def test_a_join_on_two_keys_drops_what_one_key_agrees_on() raises:
    # The same join on the shop alone, which pairs the row the test above
    # drops. Here so that the test above is known to be about the second key
    # rather than about a build side that happened to hold nothing else.
    var plan = Plan()
    var left = plan.scan("crates", List[String](), 0)
    var right = plan.scan("crated", List[String](), 1)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("shop")],
        [plan.exprs.column("place")],
        JoinKind.INNER,
    )
    var out = run_frames(plan, root, crate_frames())

    same(read_back(out, "qty"), [5, 5, 20, 3, 3, 40], "one key pairs six rows")


def test_a_semi_join_on_two_keys_keeps_the_rows_both_keys_agree_on() raises:
    # A semi join keeps none of the right side's columns, so the second key
    # cannot be asked after the pairing the way the inner join above used to
    # ask it. It pairs on the whole key or it answers the wrong rows.
    var plan = Plan()
    var root = crate_join(plan, JoinKind.SEMI)
    var out = run_frames(plan, root, crate_frames())

    assert_equal(out.width(), 2, "the left schema and nothing of the right")
    same(read_back(out, "qty"), [5, 3, 40], "the rows both keys agree on")
    same(read_back(out, "shop"), [1, 1, 2], "each one's shop")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
