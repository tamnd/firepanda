"""Tests for projection merging.

Two things are worth asserting about this pass and they need different tools.
How many projections came out is a shape, so those tests count the nodes the
root reaches. What the surviving projection computes is an expression, so those
tests read the printed plan and look for the column the substitution should have
reached through to, and for the intermediate name it should have left behind.

The groups are: what merges, what the substitution does to the expressions, the
three refusals, and the plans the pass has nothing to do with.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.plan.merge import merge
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


def _projects(plan: Plan, root: Int) -> Int:
    """Counts the projections the root reaches.

    Over the tree rather than over the arena, because a merged out node stays
    in the arena on purpose and counting it would be counting the thing the
    pass just got rid of.
    """
    var out = 0
    var stack = List[Int]()
    stack.append(root)
    while len(stack) > 0:
        var at = stack.pop()
        if String(plan.nodes[at].kind) == "PROJECT":
            out += 1
        for i in range(len(plan.nodes[at].inputs)):
            stack.append(plan.nodes[at].inputs[i])
    return out


def _under(plan: Plan, at: Int) -> String:
    """Returns the kind of the node one level below another, as a word."""
    return String(plan.nodes[plan.nodes[at].inputs[0]].kind)


def _times(mut plan: Plan, name: String, by: Int) raises -> Int:
    """Returns one column multiplied by one number."""
    return plan.exprs.binary(
        BinaryOp.MUL, plan.exprs.column(name), plan.exprs.literal(Value(by))
    )


def _plus(mut plan: Plan, name: String, by: Int) raises -> Int:
    """Returns one column with one number added to it."""
    return plan.exprs.binary(
        BinaryOp.ADD, plan.exprs.column(name), plan.exprs.literal(Value(by))
    )


def test_two_projections_become_one() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var lower = plan.project(scan, [_times(plan, "l_quantity", 2)], ["doubled"])
    var root = plan.project(lower, [_plus(plan, "doubled", 1)], ["bumped"])
    _ = merge(plan, root, [_lineitem()])
    assert_equal(_projects(plan, root), 1, "the two folded into one")
    assert_equal(_under(plan, root), "SCAN", "and it reads the scan directly")


def test_the_substitution_reaches_through_to_the_base_column() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var lower = plan.project(scan, [_times(plan, "l_quantity", 2)], ["doubled"])
    var root = plan.project(lower, [_plus(plan, "doubled", 1)], ["bumped"])
    _ = merge(plan, root, [_lineitem()])
    var printed = explain(plan, root)
    assert_true("l_quantity" in printed, "the multiply came up with the add")
    assert_true(
        "doubled" not in printed, "and the name between them is gone with it"
    )


def test_the_output_names_survive_the_merge() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var lower = plan.project(scan, [_times(plan, "l_quantity", 2)], ["doubled"])
    var root = plan.project(lower, [_plus(plan, "doubled", 1)], ["bumped"])
    var out = merge(plan, root, [_lineitem()])
    # The schema above the node is what makes the merge invisible to everything
    # that reads it, and the names are most of the schema.
    assert_equal(len(out), 1, "one column out, as before")
    assert_equal(out[0].name, "bumped", "under the name the caller asked for")


def test_three_projections_fold_in_one_sweep() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var one = plan.project(scan, [_times(plan, "l_quantity", 2)], ["a"])
    var two = plan.project(one, [_plus(plan, "a", 1)], ["b"])
    var root = plan.project(two, [_times(plan, "b", 3)], ["c"])
    _ = merge(plan, root, [_lineitem()])
    assert_equal(_projects(plan, root), 1, "all three folded")
    assert_equal(_under(plan, root), "SCAN", "down onto the scan")


def test_a_rename_is_followed() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var lower = plan.project(
        scan, [plan.exprs.column("l_quantity")], ["quantity"]
    )
    var root = plan.project(lower, [_plus(plan, "quantity", 1)], ["bumped"])
    _ = merge(plan, root, [_lineitem()])
    var printed = explain(plan, root)
    assert_equal(_projects(plan, root), 1, "a rename is an expression too")
    assert_true("l_quantity" in printed, "and the name it renamed came back")


def test_an_output_nothing_above_reads_is_dropped() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var lower = plan.project(
        scan,
        [_times(plan, "l_quantity", 2), _times(plan, "l_discount", 7)],
        ["doubled", "unwanted"],
    )
    var root = plan.project(lower, [_plus(plan, "doubled", 1)], ["bumped"])
    _ = merge(plan, root, [_lineitem()])
    var printed = explain(plan, root)
    # Projection pushdown would have got this one too. Merging gets it for free
    # because an output that is not substituted anywhere simply is not written.
    assert_true("l_discount" not in printed, "the unread multiply went away")


def test_a_column_read_twice_still_merges() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var lower = plan.project(
        scan, [plan.exprs.column("l_quantity")], ["quantity"]
    )
    var root = plan.project(
        lower,
        [
            plan.exprs.binary(
                BinaryOp.ADD,
                plan.exprs.column("quantity"),
                plan.exprs.column("quantity"),
            )
        ],
        ["twice"],
    )
    _ = merge(plan, root, [_lineitem()])
    # Reading a column twice costs nothing, so the cost rule has nothing to say
    # and the merge goes ahead.
    assert_equal(_projects(plan, root), 1, "a column is free to read again")


def test_an_expression_read_twice_is_not_merged() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var lower = plan.project(scan, [_times(plan, "l_quantity", 2)], ["doubled"])
    var root = plan.project(
        lower,
        [
            plan.exprs.binary(
                BinaryOp.ADD,
                plan.exprs.column("doubled"),
                plan.exprs.column("doubled"),
            )
        ],
        ["twice"],
    )
    _ = merge(plan, root, [_lineitem()])
    # Merging here would compute the multiply twice, which is the opposite of
    # what a pass about not walking the data twice is for.
    assert_equal(_projects(plan, root), 2, "both projections stayed")


def test_a_literal_read_twice_still_merges() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var lower = plan.project(scan, [plan.exprs.literal(Value(7))], ["seven"])
    var root = plan.project(
        lower,
        [
            plan.exprs.binary(
                BinaryOp.ADD,
                plan.exprs.column("seven"),
                plan.exprs.column("seven"),
            )
        ],
        ["fourteen"],
    )
    _ = merge(plan, root, [_lineitem()])
    assert_equal(_projects(plan, root), 1, "a constant costs nothing twice")


def test_a_projection_with_two_readers_is_left_alone() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var shared = plan.project(
        scan,
        [plan.exprs.column("l_orderkey"), _times(plan, "l_quantity", 2)],
        ["l_orderkey", "doubled"],
    )
    var left = plan.project(
        shared,
        [plan.exprs.column("l_orderkey"), _plus(plan, "doubled", 1)],
        ["l_orderkey", "a"],
    )
    var right = plan.project(
        shared,
        [plan.exprs.column("l_orderkey"), _plus(plan, "doubled", 2)],
        ["l_orderkey", "b"],
    )
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("l_orderkey")],
        [plan.exprs.column("l_orderkey")],
        JoinKind.INNER,
    )
    var before = explain(plan, root)
    _ = merge(plan, root, [_lineitem()])
    # Substituting would have written the multiply into both arms, turning one
    # evaluation into two.
    assert_equal(explain(plan, root), before, "nothing moved")


def test_a_projection_with_two_outputs_of_one_name_is_left_alone() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var lower = plan.project(
        scan,
        [plan.exprs.column("l_orderkey"), plan.exprs.column("l_partkey")],
        ["x", "x"],
    )
    var root = plan.project(lower, [_plus(plan, "x", 1)], ["bumped"])
    _ = merge(plan, root, [_lineitem()])
    # Which of the two was meant is not this pass's decision to make.
    assert_equal(_projects(plan, root), 2, "both projections stayed")


def test_a_projection_over_something_else_is_left_alone() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var first = plan.limit(scan, 0, 10)
    var root = plan.project(first, [_times(plan, "l_quantity", 2)], ["doubled"])
    var before = explain(plan, root)
    _ = merge(plan, root, [_lineitem()])
    assert_equal(explain(plan, root), before, "there is nothing to fold into")


def test_a_plan_with_no_projection_comes_back_unchanged() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.aggregate(
        scan,
        [plan.exprs.column("l_orderkey")],
        [
            plan.exprs.aggregate(
                AggKind.SUM, plan.exprs.column("l_extendedprice")
            )
        ],
        ["l_orderkey", "revenue"],
    )
    var before = explain(plan, root)
    _ = merge(plan, root, [_lineitem()])
    assert_equal(explain(plan, root), before, "nothing to do and nothing done")


def test_projections_separated_by_a_filter_are_not_merged() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var lower = plan.project(
        scan,
        [plan.exprs.column("l_orderkey"), _times(plan, "l_quantity", 2)],
        ["l_orderkey", "doubled"],
    )
    var kept = plan.filter(
        lower,
        plan.exprs.binary(
            BinaryOp.LT,
            plan.exprs.column("l_orderkey"),
            plan.exprs.literal(Value(10)),
        ),
    )
    var root = plan.project(kept, [_plus(plan, "doubled", 1)], ["bumped"])
    _ = merge(plan, root, [_lineitem()])
    # Only adjacent projections. Getting the filter out of the way is predicate
    # pushdown's job and this pass runs after it for exactly that reason.
    assert_equal(_projects(plan, root), 2, "both projections stayed")


def test_a_merge_below_a_filter_still_happens() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var one = plan.project(
        scan,
        [plan.exprs.column("l_orderkey"), _times(plan, "l_quantity", 2)],
        ["l_orderkey", "doubled"],
    )
    var two = plan.project(
        one,
        [plan.exprs.column("l_orderkey"), _plus(plan, "doubled", 1)],
        ["l_orderkey", "bumped"],
    )
    var root = plan.filter(
        two,
        plan.exprs.binary(
            BinaryOp.LT,
            plan.exprs.column("l_orderkey"),
            plan.exprs.literal(Value(10)),
        ),
    )
    _ = merge(plan, root, [_lineitem()])
    assert_equal(_projects(plan, root), 1, "the pass looks at every node")


def test_an_upper_projection_of_only_literals_drops_the_one_below() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var lower = plan.project(scan, [_times(plan, "l_quantity", 2)], ["doubled"])
    var root = plan.project(lower, [plan.exprs.literal(Value(1))], ["one"])
    _ = merge(plan, root, [_lineitem()])
    var printed = explain(plan, root)
    assert_equal(_projects(plan, root), 1, "one projection left")
    assert_true("l_quantity" not in printed, "and the multiply with it")


def test_grafting_an_expression_with_nothing_to_replace_returns_it() raises:
    var plan = Plan()
    var at = _times(plan, "l_quantity", 2)
    var same = plan.exprs.graft(at, ["other"], [plan.exprs.literal(Value(1))])
    # Only the nodes on the path to a replacement are copied, so an expression
    # that has no replacement in it comes back as the index that went in.
    assert_equal(same, at, "no copy was made")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
