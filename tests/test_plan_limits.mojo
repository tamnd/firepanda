"""Tests for slice pushdown and top n.

Three rules and one trick, and the tests are grouped that way. What a limit does
when it meets another limit, which is arithmetic and is where the off by one
lives. What it does when it meets a sort, which is a bound and not a move. What
it does when it meets a projection, which is the swap. And then the nodes it has
nothing to say about, which is most of them.

The composition cases are worth reading as a table. A slice of a slice is one
slice, and the four ways a length can be absent or present on either side are
four different answers.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.plan.limits import limits
from firepanda.plan.node import NO_LIMIT, Plan
from firepanda.plan.print import explain


def _lineitem() -> Schema:
    """Returns a schema shaped like a cut down TPC-H lineitem table.

    Returns:
        Five columns, the key not nullable.
    """
    var out = Schema()
    out.append(Field("l_orderkey", LogicalType.INT64, False))
    out.append(Field("l_partkey", LogicalType.INT64, False))
    out.append(Field("l_quantity", LogicalType.FLOAT64, True))
    out.append(Field("l_extendedprice", LogicalType.FLOAT64, True))
    out.append(Field("l_discount", LogicalType.FLOAT64, True))
    return out^


def _under(plan: Plan, at: Int) -> String:
    """Returns the kind of the node one level below another, as a word."""
    return String(plan.nodes[plan.nodes[at].inputs[0]].kind)


def _below(plan: Plan, at: Int) -> Int:
    """Returns the index of the node one level below another."""
    return plan.nodes[at].inputs[0]


def _times(mut plan: Plan, name: String, by: Int) raises -> Int:
    """Returns one column multiplied by one number."""
    return plan.exprs.binary(
        BinaryOp.MUL, plan.exprs.column(name), plan.exprs.literal(Value(by))
    )


def _sorted(mut plan: Plan, input: Int) raises -> Int:
    """Returns the input sorted by price, biggest first."""
    return plan.sort(
        input, [plan.exprs.column("l_extendedprice")], [True], [True]
    )


def test_a_limit_above_a_sort_bounds_the_sort() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var ordered = _sorted(plan, scan)
    var root = plan.limit(ordered, 0, 10)
    _ = limits(plan, root, [_lineitem()])
    assert_equal(plan.nodes[ordered].length, 10, "the sort only owes ten rows")
    assert_true("top 10" in explain(plan, root), "and it says so when printed")


def test_the_limit_stays_above_the_sort_it_bounded() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var ordered = _sorted(plan, scan)
    var root = plan.limit(ordered, 0, 10)
    _ = limits(plan, root, [_lineitem()])
    # The bound is an optimisation and not the answer. An operator that has not
    # learned to read it is slower than one that has and is not wrong.
    assert_equal(_under(plan, root), "SORT", "the limit is still doing the cut")


def test_an_offset_counts_toward_what_the_sort_owes() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var ordered = _sorted(plan, scan)
    var root = plan.limit(ordered, 20, 10)
    _ = limits(plan, root, [_lineitem()])
    assert_equal(plan.nodes[ordered].length, 30, "rows twenty one to thirty")


def test_a_limit_with_no_length_bounds_nothing() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var ordered = _sorted(plan, scan)
    var root = plan.limit(ordered, 20, NO_LIMIT)
    _ = limits(plan, root, [_lineitem()])
    # Skipping twenty and keeping the rest still wants every row in order.
    assert_equal(plan.nodes[ordered].length, NO_LIMIT, "no bound to give")


def test_the_tighter_of_two_bounds_wins() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var ordered = _sorted(plan, scan)
    var inner = plan.limit(ordered, 0, 5)
    var root = plan.limit(inner, 0, 100)
    _ = limits(plan, root, [_lineitem()])
    assert_equal(
        plan.nodes[ordered].length, 5, "the inner slice is the shorter"
    )


def test_two_limits_become_one() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var inner = plan.limit(scan, 0, 100)
    var root = plan.limit(inner, 0, 10)
    _ = limits(plan, root, [_lineitem()])
    assert_equal(_under(plan, root), "SCAN", "one limit left")
    assert_equal(plan.nodes[root].length, 10, "keeping the shorter of the two")


def test_the_offsets_of_two_limits_add_up() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var inner = plan.limit(scan, 5, 100)
    var root = plan.limit(inner, 3, 10)
    _ = limits(plan, root, [_lineitem()])
    assert_equal(plan.nodes[root].offset, 8, "five skipped then three more")
    assert_equal(plan.nodes[root].length, 10, "and ten of what is left")


def test_an_outer_limit_can_run_off_the_end_of_an_inner_one() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var inner = plan.limit(scan, 0, 10)
    var root = plan.limit(inner, 4, 100)
    _ = limits(plan, root, [_lineitem()])
    # Ten rows, four skipped, so six are reachable however many were asked for.
    assert_equal(
        plan.nodes[root].length, 6, "what is left and not what is asked"
    )


def test_an_offset_past_the_end_of_an_inner_limit_keeps_nothing() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var inner = plan.limit(scan, 0, 10)
    var root = plan.limit(inner, 50, 5)
    _ = limits(plan, root, [_lineitem()])
    assert_equal(plan.nodes[root].length, 0, "there was nothing there to keep")


def test_an_inner_limit_with_no_length_leaves_the_outer_one_in_charge() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var inner = plan.limit(scan, 7, NO_LIMIT)
    var root = plan.limit(inner, 0, 10)
    _ = limits(plan, root, [_lineitem()])
    assert_equal(plan.nodes[root].offset, 7, "the skip survives")
    assert_equal(plan.nodes[root].length, 10, "and the length is the outer one")


def test_two_limits_with_no_length_stay_without_one() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var inner = plan.limit(scan, 2, NO_LIMIT)
    var root = plan.limit(inner, 3, NO_LIMIT)
    _ = limits(plan, root, [_lineitem()])
    assert_equal(plan.nodes[root].offset, 5, "the skips add up")
    assert_equal(plan.nodes[root].length, NO_LIMIT, "and nothing bounds it")


def test_three_limits_become_one() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var one = plan.limit(scan, 0, 1000)
    var two = plan.limit(one, 0, 100)
    var root = plan.limit(two, 0, 10)
    _ = limits(plan, root, [_lineitem()])
    assert_equal(_under(plan, root), "SCAN", "all three folded")
    assert_equal(plan.nodes[root].length, 10, "down to the tightest")


def test_a_limit_swaps_past_a_projection() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var out = plan.project(scan, [_times(plan, "l_quantity", 2)], ["doubled"])
    var root = plan.limit(out, 0, 10)
    _ = limits(plan, root, [_lineitem()])
    # The multiply now runs ten times rather than six million.
    assert_equal(String(plan.nodes[root].kind), "PROJECT", "the order flipped")
    assert_equal(_under(plan, root), "LIMIT", "with the limit underneath")
    assert_equal(_under(plan, _below(plan, root)), "SCAN", "above the scan")


def test_the_swap_keeps_the_root_where_the_caller_left_it() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var out = plan.project(scan, [_times(plan, "l_quantity", 2)], ["doubled"])
    var root = plan.limit(out, 0, 10)
    var schema = limits(plan, root, [_lineitem()])
    # Two unary nodes trade contents rather than places, so the index a caller
    # is holding goes on meaning the top of the plan.
    assert_equal(len(schema), 1, "one column out, as the projection said")
    assert_equal(schema[0].name, "doubled", "under the name it was given")


def test_a_limit_swaps_past_two_projections() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var one = plan.project(scan, [_times(plan, "l_quantity", 2)], ["doubled"])
    var two = plan.project(one, [_times(plan, "doubled", 3)], ["more"])
    var root = plan.limit(two, 0, 10)
    _ = limits(plan, root, [_lineitem()])
    var printed = explain(plan, root)
    assert_true(
        printed.find("LIMIT") > printed.find("PROJECT"),
        "the limit went below both of them",
    )


def test_a_limit_does_not_swap_past_a_window() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var ranked = plan.exprs.window(
        AggKind.SUM,
        plan.exprs.column("l_extendedprice"),
        [plan.exprs.column("l_orderkey")],
        List[Int](),
    )
    var out = plan.project(scan, [ranked], ["running"])
    var root = plan.limit(out, 0, 10)
    _ = limits(plan, root, [_lineitem()])
    # A window reads its whole partition, so ten rows in is a different answer
    # rather than the same answer sooner.
    assert_equal(String(plan.nodes[root].kind), "LIMIT", "the limit stayed put")


def test_a_limit_does_not_move_past_a_filter() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var kept = plan.filter(
        scan,
        plan.exprs.binary(
            BinaryOp.LT,
            plan.exprs.column("l_orderkey"),
            plan.exprs.literal(Value(10)),
        ),
    )
    var root = plan.limit(kept, 0, 10)
    var before = explain(plan, root)
    _ = limits(plan, root, [_lineitem()])
    assert_equal(explain(plan, root), before, "it would count dropped rows")


def test_a_limit_does_not_move_past_a_distinct() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var once = plan.distinct(scan, List[Int]())
    var root = plan.limit(once, 0, 10)
    var before = explain(plan, root)
    _ = limits(plan, root, [_lineitem()])
    assert_equal(explain(plan, root), before, "ten rows in is not ten rows out")


def test_a_limit_does_not_move_past_an_aggregate() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var totals = plan.aggregate(
        scan,
        [plan.exprs.column("l_orderkey")],
        [
            plan.exprs.aggregate(
                AggKind.SUM, plan.exprs.column("l_extendedprice")
            )
        ],
        ["l_orderkey", "revenue"],
    )
    var root = plan.limit(totals, 0, 10)
    var before = explain(plan, root)
    _ = limits(plan, root, [_lineitem()])
    assert_equal(explain(plan, root), before, "the groups are not the rows")


def test_a_projection_two_nodes_read_is_not_swapped_with() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var shared = plan.project(
        scan,
        [plan.exprs.column("l_orderkey"), _times(plan, "l_quantity", 2)],
        ["l_orderkey", "doubled"],
    )
    var cut = plan.limit(shared, 0, 10)
    var root = plan.join(
        cut,
        shared,
        [plan.exprs.column("l_orderkey")],
        [plan.exprs.column("l_orderkey")],
        JoinKind.INNER,
    )
    var before = explain(plan, root)
    _ = limits(plan, root, [_lineitem()])
    # Swapping would put the limit under a node the join reads on its own, and
    # the join did not ask for ten rows.
    assert_equal(explain(plan, root), before, "nothing moved")


def test_a_plan_with_no_limit_comes_back_unchanged() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = _sorted(plan, scan)
    var before = explain(plan, root)
    _ = limits(plan, root, [_lineitem()])
    assert_equal(explain(plan, root), before, "nothing to do and nothing done")


def test_a_sort_starts_out_without_a_bound() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = _sorted(plan, scan)
    # Zero would mean a sort that owes nobody any rows, so the absence has to be
    # written as the absence.
    assert_equal(plan.nodes[root].length, NO_LIMIT, "no bound until one is put")
    assert_true("top" not in explain(plan, root), "and none printed")


def test_a_top_n_over_a_projection_gets_both_rules() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var ordered = _sorted(plan, scan)
    var out = plan.project(
        ordered, [_times(plan, "l_quantity", 2)], ["doubled"]
    )
    var root = plan.limit(out, 0, 10)
    _ = limits(plan, root, [_lineitem()])
    # The limit swaps past the projection and then meets the sort, which is the
    # shape four of the twenty two TPC-H queries actually have.
    assert_equal(String(plan.nodes[root].kind), "PROJECT", "projection on top")
    assert_equal(plan.nodes[ordered].length, 10, "and the sort got its bound")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
