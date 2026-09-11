"""Tests for common subplan elimination.

These count nodes rather than reading the printed plan. The printed plan is a
tree, so a node reached twice prints twice and prints the same before and after,
which makes the text the one thing that cannot tell whether the pass ran.

The groups are: what gets shared, what does not, and what a caller sees.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.plan.node import Plan
from firepanda.plan.print import explain
from firepanda.plan.subplan import subplan


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


def _nodes(plan: Plan, root: Int) -> Int:
    """Counts the plan nodes reachable from a root.

    Over what is reachable rather than over the arena, since the arena keeps
    every node that was ever built and the ones this pass leaves behind are the
    ones it just stopped anybody reading.
    """
    var seen = List[Int]()
    var stack = List[Int]()
    stack.append(root)
    while len(stack) > 0:
        var one = stack.pop()
        var had = False
        for i in range(len(seen)):
            if seen[i] == one:
                had = True
                break
        if had:
            continue
        seen.append(one)
        var inputs = plan.nodes[one].inputs.copy()
        for i in range(len(inputs)):
            stack.append(inputs[i])
    return len(seen)


def _cheap(mut plan: Plan, over: Int) raises -> Int:
    """Returns the input filtered down to the rows below twenty four."""
    return plan.filter(
        over,
        plan.exprs.binary(
            BinaryOp.LT,
            plan.exprs.column("l_quantity"),
            plan.exprs.literal(Value(Int64(24))),
        ),
    )


def test_two_arms_of_one_shape_become_one_node() raises:
    var plan = Plan()
    var left = _cheap(plan, plan.scan("lineitem", List[String](), 0))
    var right = _cheap(plan, plan.scan("lineitem", List[String](), 0))
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("l_orderkey")],
        [plan.exprs.column("l_orderkey")],
        JoinKind.INNER,
    )
    assert_equal(_nodes(plan, root), 5, "two scans, two filters and the join")
    _ = subplan(plan, root, [_lineitem(), _lineitem()])
    # The filters are the same shape over scans that are the same shape, so the
    # join reads one arm twice.
    assert_equal(_nodes(plan, root), 3, "one scan, one filter and the join")


def test_the_shared_part_of_two_different_branches_is_shared() raises:
    var plan = Plan()
    var left = _cheap(plan, plan.scan("lineitem", List[String](), 0))
    var right = _cheap(plan, plan.scan("lineitem", List[String](), 0))
    var one = plan.project(left, [plan.exprs.column("l_orderkey")], ["a"])
    var two = plan.project(right, [plan.exprs.column("l_partkey")], ["b"])
    var root = plan.join(
        one,
        two,
        [plan.exprs.column("a")],
        [plan.exprs.column("b")],
        JoinKind.INNER,
    )
    _ = subplan(plan, root, [_lineitem(), _lineitem()])
    # Two lines of Python that filter the same way and then select different
    # columns. The projections differ and everything under them is one.
    assert_equal(
        _nodes(plan, root), 5, "one scan, one filter, two selects, join"
    )


def test_a_shape_repeated_three_times_becomes_one() raises:
    var plan = Plan()
    var first = plan.scan("lineitem", List[String](), 0)
    var second = plan.scan("lineitem", List[String](), 0)
    var third = plan.scan("lineitem", List[String](), 0)
    var root = plan.union([first, second, third], True)
    _ = subplan(plan, root, [_lineitem(), _lineitem(), _lineitem()])
    assert_equal(_nodes(plan, root), 2, "the union over one scan, three times")


def test_two_filters_with_different_predicates_stay_two() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var left = _cheap(plan, scan)
    var right = plan.filter(
        scan,
        plan.exprs.binary(
            BinaryOp.GT,
            plan.exprs.column("l_quantity"),
            plan.exprs.literal(Value(Int64(24))),
        ),
    )
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("l_orderkey")],
        [plan.exprs.column("l_orderkey")],
        JoinKind.INNER,
    )
    _ = subplan(plan, root, [_lineitem(), _lineitem()])
    assert_equal(
        _nodes(plan, root), 4, "the scan was already one, the rest is two"
    )


def test_two_scans_of_different_relations_stay_two() raises:
    var plan = Plan()
    var left = plan.scan("lineitem", List[String](), 0)
    var right = plan.scan("orders", List[String](), 1)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("l_orderkey")],
        [plan.exprs.column("l_orderkey")],
        JoinKind.INNER,
    )
    _ = subplan(plan, root, [_lineitem(), _lineitem()])
    assert_equal(_nodes(plan, root), 3, "two tables are two tables")


def test_two_projections_with_different_names_stay_two() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var left = plan.project(scan, [plan.exprs.column("l_orderkey")], ["a"])
    var right = plan.project(scan, [plan.exprs.column("l_orderkey")], ["b"])
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("a")],
        [plan.exprs.column("b")],
        JoinKind.INNER,
    )
    _ = subplan(plan, root, [_lineitem(), _lineitem()])
    # The same column under two names is two columns to everybody above, so the
    # name is part of the shape.
    assert_equal(_nodes(plan, root), 4, "one scan and two ways of reading it")


def test_two_sorts_in_different_directions_stay_two() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var up = plan.sort(scan, [plan.exprs.column("l_quantity")], [False], [True])
    var down = plan.sort(
        scan, [plan.exprs.column("l_quantity")], [True], [True]
    )
    var root = plan.union([up, down], True)
    _ = subplan(plan, root, [_lineitem(), _lineitem()])
    assert_equal(_nodes(plan, root), 4, "the directions are part of the shape")


def test_two_limits_of_different_lengths_stay_two() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.union(
        [plan.limit(scan, 0, 10), plan.limit(scan, 0, 20)], True
    )
    _ = subplan(plan, root, [_lineitem(), _lineitem()])
    assert_equal(_nodes(plan, root), 4, "ten rows and twenty rows")


def test_two_folds_of_different_kinds_stay_two() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var summed = plan.aggregate(
        scan,
        List[Int](),
        [plan.exprs.aggregate(AggKind.SUM, plan.exprs.column("l_quantity"))],
        ["answer"],
    )
    var biggest = plan.aggregate(
        scan,
        List[Int](),
        [plan.exprs.aggregate(AggKind.MAX, plan.exprs.column("l_quantity"))],
        ["answer"],
    )
    var root = plan.union([summed, biggest], True)
    _ = subplan(plan, root, [_lineitem(), _lineitem()])
    assert_equal(_nodes(plan, root), 4, "a total is not a maximum")


def test_a_subtree_nothing_reaches_is_not_a_candidate() raises:
    var plan = Plan()
    var _orphan = _cheap(plan, plan.scan("lineitem", List[String](), 0))
    var root = _cheap(plan, plan.scan("lineitem", List[String](), 0))
    var before = _nodes(plan, root)
    _ = subplan(plan, root, [_lineitem(), _lineitem()])
    # The arena keeps every node that was ever built, including the ones earlier
    # passes stopped reading, and unifying with one would bring it back.
    assert_equal(
        _nodes(plan, root), before, "the twin nobody reads stayed dead"
    )


def test_the_schema_survives() raises:
    var plan = Plan()
    var left = _cheap(plan, plan.scan("lineitem", List[String](), 0))
    var right = _cheap(plan, plan.scan("lineitem", List[String](), 0))
    var root = plan.union([left, right], True)
    var out = subplan(plan, root, [_lineitem(), _lineitem()])
    var printed = explain(plan, root)
    # A tree printer walks a shared node once per way in, so the output says
    # nothing about the sharing and everything about it still being correct.
    assert_equal(
        _nodes(plan, root), 3, "one scan and one filter under the union"
    )
    assert_equal(len(out), 5, "the five columns the union stacks")
    assert_true("l_quantity" in printed, "reading what it read")


def test_a_plan_with_nothing_repeated_is_unchanged() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = _cheap(plan, scan)
    var before = explain(plan, root)
    _ = subplan(plan, root, [_lineitem()])
    assert_equal(explain(plan, root), before, "nothing to say")


def test_running_it_again_changes_nothing() raises:
    var plan = Plan()
    var left = _cheap(plan, plan.scan("lineitem", List[String](), 0))
    var right = _cheap(plan, plan.scan("lineitem", List[String](), 0))
    var root = plan.union([left, right], True)
    _ = subplan(plan, root, [_lineitem(), _lineitem()])
    var held = _nodes(plan, root)
    var printed = explain(plan, root)
    _ = subplan(plan, root, [_lineitem(), _lineitem()])
    assert_equal(_nodes(plan, root), held, "settled after the first run")
    assert_equal(explain(plan, root), printed, "and reading the same way")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
