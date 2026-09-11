"""Tests for predicate pushdown.

Almost all of these assert a whole printed plan, which is the opposite of what
the projection pushdown tests do and is the right choice for this pass. What
moved and how far is a shape rather than a list, and a printed plan is the one
form of it a reader can check without holding the node arena in their head.

The groups are: what a filter passes through and what stops it, the two nodes
that have an opinion about names rather than about rows, the join, the union,
splitting a filter at its `and` nodes, the places the pass declines to move
anything, and a q19 shaped plan where the point of the pass is visible.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.plan.node import NO_LIMIT, Plan
from firepanda.plan.print import explain
from firepanda.plan.push import push


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
    out.append(Field("l_shipmode", LogicalType.STRING, True))
    return out^


def _part() -> Schema:
    """Returns a schema shaped like a cut down TPC-H part table.

    Returns:
        Four columns, the key not nullable.
    """
    var out = Schema()
    out.append(Field("p_partkey", LogicalType.INT64, False))
    out.append(Field("p_brand", LogicalType.STRING, True))
    out.append(Field("p_size", LogicalType.INT32, True))
    out.append(Field("p_container", LogicalType.STRING, True))
    return out^


def _under(plan: Plan, at: Int) -> String:
    """Returns the kind of the node one level below another, as a word."""
    return String(plan.nodes[plan.nodes[at].inputs[0]].kind)


def _small(mut plan: Plan, name: String, than: Int) raises -> Int:
    """Returns a predicate that one integer column is below one number."""
    return plan.exprs.binary(
        BinaryOp.LT, plan.exprs.column(name), plan.exprs.literal(Value(than))
    )


def _filters(plan: Plan, root: Int) -> Int:
    """Counts the filter nodes the root reaches."""
    var out = 0
    for at in range(root + 1):
        if String(plan.nodes[at].kind) == "FILTER":
            out += 1
    return out


def test_a_filter_moves_below_a_sort() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var sorted = plan.sort(
        scan, [plan.exprs.column("l_quantity")], [True], [True]
    )
    var root = plan.filter(sorted, _small(plan, "l_partkey", 10))
    var at = push(plan, root, [_lineitem()])
    assert_equal(
        _under(plan, at), "FILTER", "the sort now reads a filtered scan"
    )
    assert_true("SORT" in explain(plan, at), "and the sort is still there")


def test_a_filter_stops_above_a_limit() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var first = plan.limit(scan, 0, 10)
    var root = plan.filter(first, _small(plan, "l_partkey", 10))
    var at = push(plan, root, [_lineitem()])
    # Which rows a limit keeps depends on which rows arrive, so a filter that
    # went below it would be answering a different question.
    assert_equal(_under(plan, at), "LIMIT", "the filter stayed on top")


def test_a_filter_moves_below_a_distinct_on_the_whole_row() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var once = plan.distinct(scan, List[Int]())
    var root = plan.filter(once, _small(plan, "l_partkey", 10))
    var at = push(plan, root, [_lineitem()])
    assert_equal(_under(plan, at), "FILTER", "nothing to read but the key")


def test_a_filter_on_a_distinct_key_moves_below_it() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var once = plan.distinct(scan, [plan.exprs.column("l_partkey")])
    var root = plan.filter(once, _small(plan, "l_partkey", 10))
    var at = push(plan, root, [_lineitem()])
    assert_equal(_under(plan, at), "FILTER", "a key means the same both sides")


def test_a_filter_on_a_column_a_distinct_does_not_key_on_stays_put() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var once = plan.distinct(scan, [plan.exprs.column("l_partkey")])
    var root = plan.filter(once, _small(plan, "l_orderkey", 10))
    var at = push(plan, root, [_lineitem()])
    # A keyed distinct keeps one row per key and does not say which, so
    # filtering first could keep a row that filtering afterwards would drop.
    assert_equal(_under(plan, at), "DISTINCT", "the filter stayed on top")


def test_a_filter_moves_below_a_pass_through_projection() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var out = plan.project(
        scan,
        [plan.exprs.column("l_partkey"), plan.exprs.column("l_quantity")],
        ["l_partkey", "l_quantity"],
    )
    var root = plan.filter(out, _small(plan, "l_partkey", 10))
    var at = push(plan, root, [_lineitem()])
    assert_equal(_under(plan, at), "FILTER", "the projection reads a filter")


def test_a_filter_on_a_renamed_column_stays_above_the_projection() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var out = plan.project(scan, [plan.exprs.column("l_partkey")], ["part"])
    var root = plan.filter(out, _small(plan, "part", 10))
    var at = push(plan, root, [_lineitem()])
    # `part` is a name that only exists above the projection.
    assert_equal(_under(plan, at), "PROJECT", "the filter stayed on top")


def test_a_filter_on_a_computed_column_stays_above_the_projection() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var doubled = plan.exprs.binary(
        BinaryOp.MUL,
        plan.exprs.column("l_partkey"),
        plan.exprs.literal(Value(Int64(2))),
    )
    var out = plan.project(scan, [doubled], ["l_partkey"])
    var root = plan.filter(out, _small(plan, "l_partkey", 10))
    var at = push(plan, root, [_lineitem()])
    assert_equal(_under(plan, at), "PROJECT", "a product is not a column")


def test_a_filter_on_a_group_key_moves_below_the_aggregate() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var agg = plan.aggregate(
        scan,
        [plan.exprs.column("l_partkey")],
        [plan.exprs.aggregate(AggKind.SUM, plan.exprs.column("l_quantity"))],
        ["l_partkey", "total"],
    )
    var root = plan.filter(agg, _small(plan, "l_partkey", 10))
    var at = push(plan, root, [_lineitem()])
    assert_equal(_under(plan, at), "FILTER", "a where, not a having")


def test_a_filter_on_an_aggregate_output_stays_above_it() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var agg = plan.aggregate(
        scan,
        [plan.exprs.column("l_partkey")],
        [plan.exprs.aggregate(AggKind.SUM, plan.exprs.column("l_quantity"))],
        ["l_partkey", "total"],
    )
    var total = plan.exprs.binary(
        BinaryOp.LT,
        plan.exprs.column("total"),
        plan.exprs.literal(Value(Float64(5.0))),
    )
    var root = plan.filter(agg, total)
    var at = push(plan, root, [_lineitem()])
    # This one is a having and there is no row it could be asked of below.
    assert_equal(_under(plan, at), "AGGREGATE", "the filter stayed on top")


def test_a_filter_goes_into_the_side_of_a_join_that_provides_it() raises:
    var plan = Plan()
    var left = plan.scan("part", List[String](), 0)
    var right = plan.scan("lineitem", List[String](), 1)
    var joined = plan.join(
        left,
        right,
        [plan.exprs.column("p_partkey")],
        [plan.exprs.column("l_partkey")],
        JoinKind.INNER,
    )
    var root = plan.filter(joined, _small(plan, "p_size", 15))
    var at = push(plan, root, [_part(), _lineitem()])
    var printed = explain(plan, at)
    assert_true(
        printed.find("FILTER") > printed.find("JOIN"),
        "the filter is below the join now",
    )
    assert_equal(_filters(plan, at), 1, "and there is only the one of it")


def test_each_side_of_a_join_gets_its_own_filter() raises:
    var plan = Plan()
    var left = plan.scan("part", List[String](), 0)
    var right = plan.scan("lineitem", List[String](), 1)
    var joined = plan.join(
        left,
        right,
        [plan.exprs.column("p_partkey")],
        [plan.exprs.column("l_partkey")],
        JoinKind.INNER,
    )
    var both = List[Int]()
    both.append(_small(plan, "p_size", 15))
    both.append(
        plan.exprs.binary(
            BinaryOp.LT,
            plan.exprs.column("l_quantity"),
            plan.exprs.literal(Value(Float64(30.0))),
        )
    )
    var root = plan.filter(
        joined, plan.exprs.call(String("and"), both^, rowwise=True)
    )
    var at = push(plan, root, [_part(), _lineitem()])
    assert_equal(_filters(plan, at), 2, "one condition went to each side")
    assert_true(
        String(plan.nodes[at].kind) == "JOIN", "and nothing stayed above"
    )


def test_a_filter_reading_both_sides_of_a_join_stays_above_it() raises:
    var plan = Plan()
    var left = plan.scan("part", List[String](), 0)
    var right = plan.scan("lineitem", List[String](), 1)
    var joined = plan.join(
        left,
        right,
        [plan.exprs.column("p_partkey")],
        [plan.exprs.column("l_partkey")],
        JoinKind.INNER,
    )
    var across = plan.exprs.binary(
        BinaryOp.LT,
        plan.exprs.column("p_size"),
        plan.exprs.column("l_quantity"),
    )
    var root = plan.filter(joined, across)
    var at = push(plan, root, [_part(), _lineitem()])
    # It is a join condition written as a filter, not a filter on one side.
    assert_equal(_under(plan, at), "JOIN", "the filter stayed on top")


def test_a_filter_below_an_outer_join_is_left_where_it_is() raises:
    var plan = Plan()
    var left = plan.scan("part", List[String](), 0)
    var right = plan.scan("lineitem", List[String](), 1)
    var joined = plan.join(
        left,
        right,
        [plan.exprs.column("p_partkey")],
        [plan.exprs.column("l_partkey")],
        JoinKind.LEFT,
    )
    var root = plan.filter(joined, _small(plan, "p_size", 15))
    var at = push(plan, root, [_part(), _lineitem()])
    # Only inner joins for now, and the pass would rather do nothing than
    # guess at what an invented null row does to a predicate.
    assert_equal(_under(plan, at), "JOIN", "the filter stayed on top")


def test_a_filter_goes_into_both_arms_of_a_union() raises:
    var plan = Plan()
    var one = plan.scan("lineitem", List[String](), 0)
    var two = plan.scan("lineitem", List[String](), 0)
    var both = plan.union([one, two], all=True)
    var root = plan.filter(both, _small(plan, "l_partkey", 10))
    var at = push(plan, root, [_lineitem()])
    assert_equal(_filters(plan, at), 2, "one filter per arm")
    assert_equal(String(plan.nodes[at].kind), "UNION", "and none above")


def test_a_filter_splits_at_its_ands_and_the_halves_go_separate_ways() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var out = plan.project(scan, [plan.exprs.column("l_partkey")], ["part"])
    var pieces = List[Int]()
    pieces.append(_small(plan, "part", 10))
    pieces.append(_small(plan, "part", 20))
    var root = plan.filter(
        out, plan.exprs.call(String("and"), pieces^, rowwise=True)
    )
    var at = push(plan, root, [_lineitem()])
    # Both halves read the renamed column, so both stop above the projection,
    # and the point of the test is that they come back as one filter and not
    # as two stacked on each other.
    assert_equal(_filters(plan, at), 1, "put back together as one")


def test_an_or_is_not_split() raises:
    var plan = Plan()
    var left = plan.scan("part", List[String](), 0)
    var right = plan.scan("lineitem", List[String](), 1)
    var joined = plan.join(
        left,
        right,
        [plan.exprs.column("p_partkey")],
        [plan.exprs.column("l_partkey")],
        JoinKind.INNER,
    )
    var either = List[Int]()
    either.append(_small(plan, "p_size", 15))
    either.append(
        plan.exprs.binary(
            BinaryOp.LT,
            plan.exprs.column("l_quantity"),
            plan.exprs.literal(Value(Float64(30.0))),
        )
    )
    var root = plan.filter(
        joined, plan.exprs.call(String("or"), either^, rowwise=True)
    )
    var at = push(plan, root, [_part(), _lineitem()])
    # Neither half has to hold for the row to survive, so neither can be run
    # on its own side.
    assert_equal(_under(plan, at), "JOIN", "the whole disjunction stayed")


def test_a_filter_already_on_a_scan_is_left_alone() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.filter(scan, _small(plan, "l_partkey", 10))
    var at = push(plan, root, [_lineitem()])
    assert_equal(_under(plan, at), "SCAN", "nowhere further to go")
    assert_equal(_filters(plan, at), 1, "and it was not duplicated")


def test_a_plan_with_no_filter_in_it_comes_back_the_same_shape() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.limit(scan, 0, NO_LIMIT)
    var before = explain(plan, root)
    var at = push(plan, root, [_lineitem()])
    assert_equal(explain(plan, at), before, "nothing to move, nothing moved")


def test_two_stacked_filters_come_out_as_one() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var one = plan.filter(scan, _small(plan, "l_partkey", 10))
    var root = plan.filter(one, _small(plan, "l_orderkey", 99))
    var at = push(plan, root, [_lineitem()])
    assert_equal(_filters(plan, at), 1, "two predicates, one node")
    assert_equal(_under(plan, at), "SCAN", "sitting on the scan")


def test_a_shared_subtree_is_left_untouched() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var one = plan.project(
        scan, [plan.exprs.column("l_partkey")], ["l_partkey"]
    )
    var two = plan.project(
        scan, [plan.exprs.column("l_partkey")], ["l_partkey"]
    )
    var both = plan.union([one, two], all=True)
    var root = plan.filter(both, _small(plan, "l_partkey", 10))
    var before = explain(plan, root)
    var at = push(plan, root, [_lineitem()])
    # Rebuilding a tree bottom up would write the shared scan out twice, which
    # is the opposite of what subplan elimination is for, so the pass refuses.
    assert_equal(at, root, "the root did not move")
    assert_equal(explain(plan, at), before, "and neither did anything else")


def test_a_q19_shaped_plan_filters_both_tables_before_the_join() raises:
    var plan = Plan()
    var part = plan.scan("part", List[String](), 0)
    var line = plan.scan("lineitem", List[String](), 1)
    var joined = plan.join(
        part,
        line,
        [plan.exprs.column("p_partkey")],
        [plan.exprs.column("l_partkey")],
        JoinKind.INNER,
    )
    # The four cheap conditions the spec says q19's three disjuncts share
    # between them, written flat because flattening a disjunction into the
    # conditions it implies is a later pass than this one.
    var small = _small(plan, "p_size", 15)
    var light = plan.exprs.binary(
        BinaryOp.LT,
        plan.exprs.column("l_quantity"),
        plan.exprs.literal(Value(Float64(30.0))),
    )
    var pair = List[Int]()
    pair.append(small)
    pair.append(light)
    var kept = plan.filter(
        joined, plan.exprs.call(String("and"), pair^, rowwise=True)
    )
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
    var at = push(plan, root, [_part(), _lineitem()])
    var printed = explain(plan, at)
    assert_equal(_filters(plan, at), 2, "one filter on each table")
    assert_true(
        printed.find("JOIN") < printed.find("FILTER"),
        "and both of them below the join",
    )


def test_a_filter_over_a_values_stays_where_it_was() raises:
    # The other node with no input, so a predicate that gets down to it has
    # nowhere further to go and is written back above.
    var plan = Plan()
    var one = plan.exprs.literal(Value(Int64(1)))
    var two = plan.exprs.literal(Value(Int64(2)))
    var table = plan.values([one, two], ["n"])
    var n = plan.exprs.column("n")
    var big = plan.exprs.binary(BinaryOp.GT, n, one)
    var kept = plan.filter(table, big)
    var at = push(plan, kept, List[Schema]())
    assert_equal(
        explain(plan, at),
        "FILTER n > 1\n  VALUES [n] (1), (2)\n",
        "one filter, still above the rows",
    )


def test_a_filter_over_a_table_function_stays_where_it_was() raises:
    # The third node with no input, and the same answer as the other two.
    var plan = Plan()
    var five = plan.exprs.literal(Value(Int64(5)))
    var rows = plan.table_function("range", [five], ["i"])
    var i = plan.exprs.column("i")
    var two = plan.exprs.literal(Value(Int64(2)))
    var big = plan.exprs.binary(BinaryOp.GT, i, two)
    var kept = plan.filter(rows, big)
    var at = push(plan, kept, List[Schema]())
    assert_equal(
        explain(plan, at),
        "FILTER i > 2\n  range(5) [i]\n",
        "one filter, still above the call",
    )


def test_a_filter_over_a_window_stays_above_it() raises:
    # Not conservatism. A window reads the whole partition a row is in, so a
    # row thrown away below it changes the answer for every row beside it, and
    # the predicate here reads a column the window does not even mention.
    var plan = Plan()
    var scan = plan.scan("lineitem", ["l_orderkey", "l_quantity"], 0)
    var qty = plan.exprs.column("l_quantity")
    var key = plan.exprs.column("l_orderkey")
    var running = plan.exprs.window(AggKind.SUM, qty, [key], List[Int]())
    var over = plan.window(scan, [running], ["line_total"])
    var thirty = plan.exprs.literal(Value(Float64(30.0)))
    var few = plan.exprs.binary(BinaryOp.LT, qty, thirty)
    var kept = plan.filter(over, few)
    var at = push(plan, kept, [_lineitem()])
    assert_equal(
        explain(plan, at),
        (
            "FILTER l_quantity < 30.0\n"
            "  WINDOW [sum(l_quantity) over (partition l_orderkey) as"
            " line_total]\n"
            "    SCAN lineitem [l_orderkey, l_quantity]\n"
        ),
        "the filter is where it was written",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
