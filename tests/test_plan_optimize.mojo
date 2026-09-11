"""Tests for the pass pipeline.

The individual passes have their own files and are tested there. What is
left for here is the three things only the pipeline can get wrong: that the
passes run in an order where each leaves the next something it can use, that
running it again changes nothing, and that the root it hands back is the root a
caller should go on using.

The plans are shaped like real TPC-H queries rather than like the smallest thing
that would exercise a rule, because the point of a pipeline test is what several
passes do to one plan between them.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.plan.bind import bind
from firepanda.plan.node import Plan
from firepanda.plan.optimize import optimize
from firepanda.plan.print import explain


def _lineitem() -> Schema:
    """Returns a schema shaped like a cut down TPC-H lineitem table.

    Returns:
        Six columns, the keys not nullable.
    """
    var out = Schema()
    out.append(Field("l_orderkey", LogicalType.INT64, False))
    out.append(Field("l_partkey", LogicalType.INT64, False))
    out.append(Field("l_quantity", LogicalType.FLOAT64, True))
    out.append(Field("l_extendedprice", LogicalType.FLOAT64, True))
    out.append(Field("l_discount", LogicalType.FLOAT64, True))
    out.append(Field("l_shipmode", LogicalType.STRING, True))
    return out^


def _small(mut plan: Plan, name: String, than: Int) raises -> Int:
    """Returns a predicate that one column is below one number."""
    return plan.exprs.binary(
        BinaryOp.LT, plan.exprs.column(name), plan.exprs.literal(Value(than))
    )


def test_a_filter_over_a_sort_comes_out_under_it() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var ordered = plan.sort(
        scan, [plan.exprs.column("l_extendedprice")], [True], [True]
    )
    var root = plan.filter(ordered, _small(plan, "l_partkey", 10))
    var at = optimize(plan, root, [_lineitem()])
    var printed = explain(plan, at)
    assert_true(
        printed.find("SORT") < printed.find("FILTER"),
        "predicate pushdown ran and the sort is now above the filter",
    )


def test_the_pipeline_narrows_the_scan_and_pushes_the_filter() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var kept = plan.filter(scan, _small(plan, "l_quantity", 24))
    var root = plan.aggregate(
        kept,
        List[Int](),
        [
            plan.exprs.aggregate(
                AggKind.SUM, plan.exprs.column("l_extendedprice")
            )
        ],
        ["revenue"],
    )
    var at = optimize(plan, root, [_lineitem()])
    var printed = explain(plan, at)
    # The q6 shape. Two columns are read out of six and nothing else is touched.
    assert_true("l_quantity" in printed, "the predicate's column is read")
    assert_true("l_extendedprice" in printed, "and the one summed")
    assert_true("l_shipmode" not in printed, "and none of the other four")
    assert_true("l_orderkey" not in printed, "including the key")


def test_stacked_projections_come_out_as_one() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var one = plan.project(
        scan,
        [
            plan.exprs.binary(
                BinaryOp.SUB,
                plan.exprs.literal(Value(1)),
                plan.exprs.column("l_discount"),
            )
        ],
        ["rate"],
    )
    var root = plan.project(
        one,
        [
            plan.exprs.binary(
                BinaryOp.MUL,
                plan.exprs.column("rate"),
                plan.exprs.literal(Value(100)),
            )
        ],
        ["percent"],
    )
    var at = optimize(plan, root, [_lineitem()])
    var printed = explain(plan, at)
    assert_true("rate" not in printed, "the name between them is gone")
    assert_true("l_discount" in printed, "and the column underneath came up")


def test_a_top_n_over_a_projection_gets_the_last_two_passes() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var ordered = plan.sort(
        scan, [plan.exprs.column("l_extendedprice")], [True], [True]
    )
    var out = plan.project(
        ordered,
        [
            plan.exprs.column("l_orderkey"),
            plan.exprs.column("l_extendedprice"),
        ],
        ["l_orderkey", "l_extendedprice"],
    )
    var root = plan.limit(out, 0, 10)
    var at = optimize(plan, root, [_lineitem()])
    var printed = explain(plan, at)
    # The limit swapped past the projection and then bounded the sort, which is
    # the shape four of the twenty two queries have.
    assert_true("top 10" in printed, "the sort only owes ten rows")


def test_running_it_again_changes_nothing() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var kept = plan.filter(scan, _small(plan, "l_quantity", 24))
    var ordered = plan.sort(
        kept, [plan.exprs.column("l_extendedprice")], [True], [True]
    )
    var out = plan.project(
        ordered, [plan.exprs.column("l_extendedprice")], ["price"]
    )
    var root = plan.limit(out, 0, 10)
    var once = optimize(plan, root, [_lineitem()])
    var printed = explain(plan, once)
    var twice = optimize(plan, once, [_lineitem()])
    # A pipeline that is not idempotent is one where the order is wrong, so
    # this is the test that would notice a pass added in the wrong place.
    assert_equal(explain(plan, twice), printed, "settled after the first run")


def test_the_root_it_hands_back_is_the_one_to_use() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var ordered = plan.sort(
        scan, [plan.exprs.column("l_extendedprice")], [True], [True]
    )
    var root = plan.filter(ordered, _small(plan, "l_partkey", 10))
    var at = optimize(plan, root, [_lineitem()])
    var schema = bind(plan, at, [_lineitem()])
    # Predicate pushdown rebuilds the node list, so the index that went in
    # means nothing on the way out and the schema is the proof.
    assert_equal(len(schema), 6, "the whole row, as the filter hands it on")
    assert_equal(schema[0].name, "l_orderkey", "in the order the table has")


def test_a_plan_with_nothing_to_do_comes_back_the_same() raises:
    var plan = Plan()
    var root = plan.scan("lineitem", ["l_orderkey"], 0)
    var before = explain(plan, root)
    var at = optimize(plan, root, [_lineitem()])
    assert_equal(explain(plan, at), before, "one scan and no opinions")


def test_a_constant_predicate_is_folded_before_anything_moves() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var cutoff = plan.exprs.binary(
        BinaryOp.SUB,
        plan.exprs.literal(Value(Int64(100))),
        plan.exprs.literal(Value(Int64(10))),
    )
    var kept = plan.filter(
        scan,
        plan.exprs.binary(BinaryOp.LT, plan.exprs.column("l_quantity"), cutoff),
    )
    var root = plan.project(
        kept, [plan.exprs.column("l_extendedprice")], ["price"]
    )
    var at = optimize(plan, root, [_lineitem()])
    var printed = explain(plan, at)
    # Simplification runs first so that nothing downstream ever sees the
    # subtraction, and nothing evaluates it once a row.
    assert_true("90" in printed, "the arithmetic happened once, here")
    assert_true("100" not in printed, "and the pieces of it are gone")


def test_a_predicate_that_folds_to_false_empties_the_plan() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var never = plan.exprs.binary(
        BinaryOp.LT,
        plan.exprs.literal(Value(Int64(100))),
        plan.exprs.literal(Value(Int64(10))),
    )
    var kept = plan.filter(scan, never)
    var ordered = plan.sort(
        kept, [plan.exprs.column("l_extendedprice")], [True], [True]
    )
    var root = plan.limit(ordered, 0, 10)
    var at = optimize(plan, root, [_lineitem()])
    var printed = explain(plan, at)
    # Simplification folds the comparison, pruning turns the filter into the
    # emptiness and then takes the sort and the limit off the top of it, and
    # neither of the two nodes that went is a node the passes below had to see.
    assert_true("LIMIT 0" in printed, "no rows come out of it")
    assert_true("SORT" not in printed, "and nothing is sorted to find that out")


def _reached(plan: Plan, at: Int) raises -> Int:
    """Counts the distinct expression nodes one plan node's expressions reach.
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


def test_a_repeated_expression_is_unified_by_the_pipeline() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.project(
        scan,
        [
            plan.exprs.binary(
                BinaryOp.MUL,
                plan.exprs.column("l_extendedprice"),
                plan.exprs.column("l_discount"),
            ),
            plan.exprs.binary(
                BinaryOp.MUL,
                plan.exprs.column("l_extendedprice"),
                plan.exprs.column("l_discount"),
            ),
        ],
        ["a", "b"],
    )
    var at = optimize(plan, root, [_lineitem()])
    # Six nodes went in and three came out. The printed plan is the same either
    # way, which is why this one counts rather than reading the text.
    assert_equal(_reached(plan, at), 3, "one product between the two outputs")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
