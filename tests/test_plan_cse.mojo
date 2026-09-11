"""Tests for common subexpression elimination.

What the pass does is make two indices into one, so most of these count
expressions rather than reading the printed plan. Counting is done over the tree
the node's expressions reach, because the arena keeps every node that was ever
built and the ones this pass leaves behind are the ones it just got rid of.

The groups are: what gets unified, what does not, and the plans the pass has
nothing to do with.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.plan.cse import cse
from firepanda.plan.node import Plan
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


def _nodes(plan: Plan, at: Int) raises -> Int:
    """Counts the distinct expression nodes one plan node's expressions reach.

    Over the tree rather than over the arena, since the arena keeps everything
    that was ever built and the point of the pass is what the node still reads.
    """
    var seen = List[Int]()
    var stack = plan.nodes[at].exprs.copy()
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
        var kids = plan.exprs.nodes[one].children.copy()
        for i in range(len(kids)):
            stack.append(kids[i])
    return len(seen)


def _rate(mut plan: Plan) raises -> Int:
    """Returns `1 - l_discount`, freshly built every time it is called."""
    return plan.exprs.binary(
        BinaryOp.SUB,
        plan.exprs.literal(Value(Int64(1))),
        plan.exprs.column("l_discount"),
    )


def _priced(mut plan: Plan) raises -> Int:
    """Returns `l_extendedprice * (1 - l_discount)`, freshly built."""
    return plan.exprs.binary(
        BinaryOp.MUL, plan.exprs.column("l_extendedprice"), _rate(plan)
    )


def test_two_expressions_of_one_shape_become_one() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.project(scan, [_priced(plan), _priced(plan)], ["a", "b"])
    # Five nodes each: the price, the one, the discount, and the multiply and
    # the subtraction over them.
    assert_equal(_nodes(plan, root), 10, "two trees with nothing in common")
    _ = cse(plan, root, [_lineitem()])
    assert_equal(_nodes(plan, root), 5, "one tree, read twice")


def test_the_shared_part_of_two_different_expressions_is_unified() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var one = _priced(plan)
    var two = plan.exprs.binary(
        BinaryOp.ADD, _priced(plan), plan.exprs.literal(Value(Int64(7)))
    )
    var root = plan.project(scan, [one, two], ["disc_price", "charge"])
    _ = cse(plan, root, [_lineitem()])
    # The q1 shape. The product is built twice by a caller who wrote it twice,
    # and one of the two trees is left holding the other's product.
    assert_equal(_nodes(plan, root), 7, "the product and the add above it")
    var printed = explain(plan, root)
    assert_true("l_extendedprice" in printed, "still reads what it read")


def test_a_shape_repeated_three_times_becomes_one() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.project(
        scan, [_rate(plan), _rate(plan), _rate(plan)], ["a", "b", "c"]
    )
    _ = cse(plan, root, [_lineitem()])
    assert_equal(_nodes(plan, root), 3, "one subtraction over one pair")


def test_a_repeated_column_is_one_column() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.project(
        scan,
        [plan.exprs.column("l_quantity"), plan.exprs.column("l_quantity")],
        ["a", "b"],
    )
    _ = cse(plan, root, [_lineitem()])
    assert_equal(_nodes(plan, root), 1, "one column read twice")


def test_two_folds_over_one_shape_read_one_expression() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.aggregate(
        scan,
        List[Int](),
        [
            plan.exprs.aggregate(AggKind.SUM, _priced(plan)),
            plan.exprs.aggregate(AggKind.MAX, _priced(plan)),
        ],
        ["revenue", "biggest"],
    )
    _ = cse(plan, root, [_lineitem()])
    # The two folds differ, so they stay two, and what is under them is one.
    assert_equal(_nodes(plan, root), 7, "two folds over one product")


def test_a_filter_and_a_projection_are_not_unified_with_each_other() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var kept = plan.filter(
        scan,
        plan.exprs.binary(
            BinaryOp.GT, _priced(plan), plan.exprs.literal(Value(Int64(0)))
        ),
    )
    var root = plan.project(kept, [_priced(plan)], ["disc_price"])
    _ = cse(plan, root, [_lineitem()])
    # One node at a time, because a column binds to a position and the same
    # name under two nodes can bind to two. Both still hold their own product.
    assert_equal(_nodes(plan, root), 5, "the projection kept its own")
    assert_equal(_nodes(plan, kept), 7, "and the filter kept its own")


def test_two_literals_of_different_types_are_not_the_same_literal() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.project(
        scan,
        [
            plan.exprs.literal(Value(Int64(1))),
            plan.exprs.literal(Value(Float64(1))),
        ],
        ["a", "b"],
    )
    _ = cse(plan, root, [_lineitem()])
    # They print the same and they are not the same value, which is why the
    # type goes into the key.
    assert_equal(_nodes(plan, root), 2, "a one and a one point oh")


def test_two_different_operations_are_not_unified() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var price = plan.exprs.column("l_extendedprice")
    var discount = plan.exprs.column("l_discount")
    var root = plan.project(
        scan,
        [
            plan.exprs.binary(BinaryOp.ADD, price, discount),
            plan.exprs.binary(BinaryOp.SUB, price, discount),
        ],
        ["a", "b"],
    )
    _ = cse(plan, root, [_lineitem()])
    assert_equal(_nodes(plan, root), 4, "two operations over one pair")


def test_operands_the_other_way_round_are_not_unified() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var price = plan.exprs.column("l_extendedprice")
    var discount = plan.exprs.column("l_discount")
    var root = plan.project(
        scan,
        [
            plan.exprs.binary(BinaryOp.SUB, price, discount),
            plan.exprs.binary(BinaryOp.SUB, discount, price),
        ],
        ["a", "b"],
    )
    _ = cse(plan, root, [_lineitem()])
    # Recognising these as one would need a rule about which operations
    # commute, and a subtraction is not one of them anyway.
    assert_equal(_nodes(plan, root), 4, "order is part of the shape")


def test_a_join_is_left_alone() raises:
    var plan = Plan()
    var left = plan.scan("lineitem", List[String](), 0)
    var right = plan.scan("lineitem", List[String](), 0)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("l_orderkey")],
        [plan.exprs.column("l_orderkey")],
        JoinKind.INNER,
    )
    var before = _nodes(plan, root)
    _ = cse(plan, root, [_lineitem(), _lineitem()])
    # A join's two key lists read two different inputs, so one shape is not one
    # value and unifying them would join a column against itself.
    assert_equal(_nodes(plan, root), before, "both keys stayed")


def test_a_plan_with_nothing_repeated_is_unchanged() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.project(scan, [_priced(plan)], ["disc_price"])
    var before = explain(plan, root)
    _ = cse(plan, root, [_lineitem()])
    assert_equal(explain(plan, root), before, "nothing to say")


def test_the_schema_survives() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.project(scan, [_priced(plan), _priced(plan)], ["a", "b"])
    var out = cse(plan, root, [_lineitem()])
    # Two outputs that are now one expression are still two outputs, and the
    # names are what every reader above is going by.
    assert_equal(len(out), 2, "two columns out, as before")
    assert_equal(out[0].name, "a", "the first name")
    assert_equal(out[1].name, "b", "and the second")


def test_running_it_again_changes_nothing() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.project(scan, [_priced(plan), _priced(plan)], ["a", "b"])
    _ = cse(plan, root, [_lineitem()])
    var once = explain(plan, root)
    var held = _nodes(plan, root)
    _ = cse(plan, root, [_lineitem()])
    assert_equal(explain(plan, root), once, "settled after the first run")
    assert_equal(_nodes(plan, root), held, "and built nothing new")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
