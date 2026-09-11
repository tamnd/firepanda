"""Tests for the logical plan arena, its checks, and what it prints.

Most of these are a plan in and a printed plan out, compared as text. That is
the shape `docs/specs/planner/01-what-a-plan-is.md` asks for and it is the shape
worth having, because the alternative is a test that reaches into the node's
fields and asserts the thing the builder just wrote, which passes whatever the
node means.

The checks are the other half. A plan node holds expressions by index and an
index on its own says nothing about whether the expression makes sense in the
position it was put in, so every builder checks what it can without a schema:
that the inputs exist, that the lists that have to be the same length are, and
that the expressions in the positions read once per row can be. The last one is
the only interesting check and it has a test for each of the four positions.

The TPC-H shaped plan at the end is there to say that nine node kinds are enough
to write a real query in, which is the claim the spec makes and is the only
thing that would show it was wrong.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.unary import UnaryOp
from firepanda.plan.expr import Expressions
from firepanda.plan.node import (
    NO_LIMIT,
    SET_EXCEPT,
    SET_INTERSECT,
    NodeKind,
    Plan,
)
from firepanda.plan.print import explain, render_expr


def test_a_scan_prints_its_source_and_its_columns() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", ["l_shipdate", "l_discount"], 0)
    assert_equal(
        explain(plan, scan),
        "SCAN lineitem [l_shipdate, l_discount]\n",
        "the source and the columns read",
    )


def test_a_filter_prints_above_what_it_filters() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", ["l_quantity"], 0)
    var q = plan.exprs.column("l_quantity")
    var limit = plan.exprs.literal(Value(Float64(24.0)))
    var small = plan.exprs.binary(BinaryOp.LT, q, limit)
    var kept = plan.filter(scan, small)
    assert_equal(
        explain(plan, kept),
        "FILTER l_quantity < 24.0\n  SCAN lineitem [l_quantity]\n",
        "the filter over its input",
    )


def test_a_compound_operand_gets_brackets() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var b = tree.column("b")
    var two = tree.literal(Value(Int64(2)))
    var scaled = tree.binary(BinaryOp.MUL, b, two)
    var total = tree.binary(BinaryOp.ADD, a, scaled)
    # Noisier than precedence needs on this one and right on everything else,
    # because a printed plan that leans on the reader knowing the precedence
    # table is a printed plan that gets misread.
    assert_equal(render_expr(tree, total), "a + (b * 2)", "brackets go on")


def test_a_projection_drops_a_name_that_says_nothing() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", ["l_extendedprice", "l_discount"], 0)
    var price = plan.exprs.column("l_extendedprice")
    var discount = plan.exprs.column("l_discount")
    var revenue = plan.exprs.binary(BinaryOp.MUL, price, discount)
    var out = plan.project(
        scan, [price, revenue], ["l_extendedprice", "revenue"]
    )
    assert_equal(
        explain(plan, out),
        (
            "PROJECT [l_extendedprice, l_extendedprice * l_discount as"
            " revenue]\n  SCAN lineitem [l_extendedprice, l_discount]\n"
        ),
        "only the one that renames something prints its name",
    )


def test_an_aggregate_prints_its_keys_and_its_folds() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", ["l_returnflag", "l_quantity"], 0)
    var flag = plan.exprs.column("l_returnflag")
    var quantity = plan.exprs.column("l_quantity")
    var total = plan.exprs.aggregate(AggKind.SUM, quantity)
    var counted = plan.exprs.aggregate(AggKind.COUNT, quantity)
    var grouped = plan.aggregate(
        scan, [flag], [total, counted], ["l_returnflag", "sum_qty", "n"]
    )
    assert_equal(
        explain(plan, grouped),
        (
            "AGGREGATE [l_returnflag] -> [sum(l_quantity), count(l_quantity)]\n"
            "  SCAN lineitem [l_returnflag, l_quantity]\n"
        ),
        "keys on the left of the arrow and folds on the right",
    )


def test_a_reduction_is_an_aggregate_with_no_keys() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", ["l_quantity"], 0)
    var quantity = plan.exprs.column("l_quantity")
    var total = plan.exprs.aggregate(AggKind.SUM, quantity)
    var reduced = plan.aggregate(scan, List[Int](), [total], ["revenue"])
    # Whether to hash every row to find its group or to know there is one group
    # is a physical choice, so an empty key list is the same logical node.
    assert_true(
        plan.nodes[reduced].kind == NodeKind.AGGREGATE, "still an aggregate"
    )
    assert_equal(
        explain(plan, reduced),
        "AGGREGATE [] -> [sum(l_quantity)]\n  SCAN lineitem [l_quantity]\n",
        "with nothing on the left of the arrow",
    )


def test_a_join_prints_both_inputs_under_it() raises:
    var plan = Plan()
    var orders = plan.scan("orders", ["o_custkey"], 0)
    var customer = plan.scan("customer", ["c_custkey"], 1)
    var o = plan.exprs.column("o_custkey")
    var c = plan.exprs.column("c_custkey")
    var joined = plan.join(orders, customer, [o], [c], JoinKind.INNER)
    assert_equal(
        explain(plan, joined),
        (
            "JOIN inner [o_custkey = c_custkey]\n"
            "  SCAN orders [o_custkey]\n"
            "  SCAN customer [c_custkey]\n"
        ),
        "the left input first and the indent says which is which",
    )


def test_a_sort_prints_a_direction_for_each_key() raises:
    var plan = Plan()
    var scan = plan.scan("orders", ["o_orderdate", "o_totalprice"], 0)
    var date = plan.exprs.column("o_orderdate")
    var price = plan.exprs.column("o_totalprice")
    var sorted = plan.sort(scan, [price, date], [True, False], [False, True])
    assert_equal(
        explain(plan, sorted),
        (
            "SORT [o_totalprice desc, o_orderdate asc nulls last]\n"
            "  SCAN orders [o_orderdate, o_totalprice]\n"
        ),
        "a direction each and the null placement when it is not the default",
    )


def test_a_limit_prints_its_offset_only_when_it_has_one() raises:
    var plan = Plan()
    var scan = plan.scan("orders", ["o_orderkey"], 0)
    var ten = plan.limit(scan, 0, 10)
    var skipped = plan.limit(scan, 5, 10)
    var rest = plan.limit(scan, 5, NO_LIMIT)
    assert_true(explain(plan, ten).startswith("LIMIT 10\n"), "no offset")
    assert_true(
        explain(plan, skipped).startswith("LIMIT 10 offset 5\n"), "an offset"
    )
    assert_true(
        explain(plan, rest).startswith("LIMIT all offset 5\n"), "no length"
    )


def test_a_distinct_with_no_keys_is_the_whole_row() raises:
    var plan = Plan()
    var scan = plan.scan("nation", ["n_name"], 0)
    var whole = plan.distinct(scan, List[Int]())
    var name = plan.exprs.column("n_name")
    var by_name = plan.distinct(scan, [name])
    assert_true(explain(plan, whole).startswith("DISTINCT [*]\n"), "all of it")
    assert_true(
        explain(plan, by_name).startswith("DISTINCT [n_name]\n"), "one key"
    )


def test_a_union_says_whether_duplicates_survive() raises:
    var plan = Plan()
    var a = plan.scan("a", ["k"], 0)
    var b = plan.scan("b", ["k"], 1)
    var kept = plan.union([a, b], all=True)
    var dropped = plan.union([a, b], all=False)
    assert_equal(
        explain(plan, kept),
        "UNION all\n  SCAN a [k]\n  SCAN b [k]\n",
        "concat is a union that keeps them",
    )
    assert_true(explain(plan, dropped).startswith("UNION\n"), "and this drops")


def test_the_other_two_set_operations_are_the_same_node() raises:
    var plan = Plan()
    var a = plan.scan("a", ["k"], 0)
    var b = plan.scan("b", ["k"], 1)
    var without = plan.setop([a, b], SET_EXCEPT, all=False)
    var both = plan.setop([a, b], SET_INTERSECT, all=True)
    assert_equal(
        explain(plan, without),
        "EXCEPT\n  SCAN a [k]\n  SCAN b [k]\n",
        "the rows of the left that the right does not have",
    )
    assert_equal(
        explain(plan, both),
        "INTERSECT all\n  SCAN a [k]\n  SCAN b [k]\n",
        "and the rows both of them have",
    )


def test_a_difference_and_an_intersection_are_between_two_inputs() raises:
    # A union stacks any number and these two do not, because being on both
    # sides is a question about a pair and says nothing about a third.
    var plan = Plan()
    var a = plan.scan("a", ["k"], 0)
    var b = plan.scan("b", ["k"], 1)
    var c = plan.scan("c", ["k"], 2)
    with assert_raises(contains="a difference is between two inputs"):
        _ = plan.setop([a, b, c], SET_EXCEPT, all=False)
    with assert_raises(contains="an intersection is between two inputs"):
        _ = plan.setop([a], SET_INTERSECT, all=False)
    assert_equal(
        explain(plan, plan.union([a, b, c], all=True)),
        "UNION all\n  SCAN a [k]\n  SCAN b [k]\n  SCAN c [k]\n",
        "and a union takes all three",
    )


def test_a_set_operation_is_one_of_three() raises:
    var plan = Plan()
    var a = plan.scan("a", ["k"], 0)
    var b = plan.scan("b", ["k"], 1)
    with assert_raises(contains="set operation 7 is not one of three"):
        _ = plan.setop([a, b], 7, all=False)


def test_a_values_holds_the_rows_it_was_given() raises:
    var plan = Plan()
    var rows = List[Int]()
    for i in range(1, 5):
        rows.append(plan.exprs.literal(Value(Int64(i))))
    var table = plan.values(rows^, ["a", "b"])
    assert_equal(
        explain(plan, table),
        "VALUES [a, b] (1, 2), (3, 4)\n",
        "two columns and two rows",
    )


def test_a_values_has_to_divide_into_whole_rows() raises:
    var plan = Plan()
    var one = plan.exprs.literal(Value(Int64(1)))
    with assert_raises(contains="2 columns wide cannot be made of 3 values"):
        _ = plan.values([one, one, one], ["a", "b"])
    with assert_raises(contains="a table of no columns"):
        _ = plan.values([one], List[String]())
    with assert_raises(contains="still has to say it has none"):
        _ = plan.values(List[Int](), ["a"])


def test_a_values_cannot_read_anything() raises:
    # There is nothing under it to read, so a column reference in one is a
    # query the caller has not written rather than a name that will resolve
    # later.
    var plan = Plan()
    var one = plan.exprs.literal(Value(Int64(1)))
    var k = plan.exprs.column("k")
    with assert_raises(contains="column 1 of row 0 reads something"):
        _ = plan.values([one, k], ["a", "b"])
    var total = plan.exprs.aggregate(AggKind.SUM, one)
    with assert_raises(contains="column 0 of row 1 reads something"):
        _ = plan.values([one, one, total, one], ["a", "b"])


def test_a_table_function_prints_its_name_and_its_arguments() raises:
    var plan = Plan()
    var start = plan.exprs.literal(Value(Int64(1)))
    var stop = plan.exprs.literal(Value(Int64(10)))
    var rows = plan.table_function("range", [start, stop], ["i"])
    assert_equal(
        explain(plan, rows),
        "range(1, 10) [i]\n",
        "what it is called, what it was given, and what comes out",
    )


def test_a_table_function_with_no_arguments_still_prints() raises:
    # There is no such function yet, and the node does not know which functions
    # there are, so the empty argument list has to print as an empty argument
    # list rather than as something the reader has to guess at.
    var plan = Plan()
    var rows = plan.table_function("now", List[Int](), ["t"])
    assert_equal(explain(plan, rows), "now() [t]\n", "the call with nothing in")


def test_a_table_function_needs_a_name_and_a_column() raises:
    var plan = Plan()
    var one = plan.exprs.literal(Value(Int64(1)))
    with assert_raises(contains="a table function with no name"):
        _ = plan.table_function("", [one], ["i"])
    with assert_raises(contains="a table of no columns"):
        _ = plan.table_function("range", [one], List[String]())


def test_a_table_function_cannot_read_anything() raises:
    # Same as a VALUES. It is called where a table goes, so there is nothing
    # under it for a column reference to resolve against.
    var plan = Plan()
    var one = plan.exprs.literal(Value(Int64(1)))
    var k = plan.exprs.column("k")
    with assert_raises(contains="argument 2 of range reads something"):
        _ = plan.table_function("range", [one, k], ["i"])


def test_a_window_prints_the_columns_it_adds() raises:
    var plan = Plan()
    var scan = plan.scan("orders", ["o_custkey", "o_totalprice"], 0)
    var price = plan.exprs.column("o_totalprice")
    var who = plan.exprs.column("o_custkey")
    var running = plan.exprs.window(AggKind.SUM, price, [who], List[Int]())
    var at = plan.window(scan, [running], ["spent"])
    assert_equal(
        explain(plan, at),
        (
            "WINDOW [sum(o_totalprice) over (partition o_custkey) as spent]\n"
            "  SCAN orders [o_custkey, o_totalprice]\n"
        ),
        "what it adds, over what it adds it, and what is underneath",
    )


def test_a_window_needs_a_name_for_each_column_it_adds() raises:
    var plan = Plan()
    var scan = plan.scan("orders", ["o_totalprice"], 0)
    var price = plan.exprs.column("o_totalprice")
    var total = plan.exprs.window(AggKind.SUM, price, List[Int](), List[Int]())
    var most = plan.exprs.window(AggKind.MAX, price, List[Int](), List[Int]())
    with assert_raises(contains="computes 2 columns and has 1 names"):
        _ = plan.window(scan, [total, most], ["total"])


def test_a_window_node_with_no_window_in_it_is_refused() raises:
    var plan = Plan()
    var scan = plan.scan("orders", ["o_totalprice"], 0)
    with assert_raises(contains="computes no window"):
        _ = plan.window(scan, List[Int](), List[String]())


def test_only_a_window_goes_in_a_window_node() raises:
    # The rule that keeps every other pass free to treat a project as
    # elementwise. An aggregate here wanted a GROUP BY and a column here wanted
    # a projection, and both of those have a node of their own already.
    var plan = Plan()
    var scan = plan.scan("orders", ["o_totalprice"], 0)
    var price = plan.exprs.column("o_totalprice")
    var folded = plan.exprs.aggregate(AggKind.SUM, price)
    with assert_raises(
        contains="column 1 of a window node is of kind aggregate"
    ):
        _ = plan.window(scan, [folded], ["total"])
    with assert_raises(contains="column 1 of a window node is of kind column"):
        _ = plan.window(scan, [price], ["price"])


def test_an_input_outside_the_plan_is_refused() raises:
    var plan = Plan()
    _ = plan.scan("lineitem", ["l_quantity"], 0)
    var q = plan.exprs.column("l_quantity")
    with assert_raises(contains="is not in a plan of 1"):
        _ = plan.filter(4, q)


def test_a_projection_needs_a_name_for_each_output() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", ["a", "b"], 0)
    var a = plan.exprs.column("a")
    var b = plan.exprs.column("b")
    with assert_raises(contains="2 outputs and 1 names"):
        _ = plan.project(scan, [a, b], ["a"])


def test_an_aggregate_needs_a_name_for_every_column_it_makes() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", ["k", "v"], 0)
    var k = plan.exprs.column("k")
    var v = plan.exprs.column("v")
    var total = plan.exprs.aggregate(AggKind.SUM, v)
    with assert_raises(contains="produces 2 columns and has 1 names"):
        _ = plan.aggregate(scan, [k], [total], ["k"])


def test_a_join_needs_the_same_number_of_keys_on_both_sides() raises:
    var plan = Plan()
    var left = plan.scan("a", ["k", "j"], 0)
    var right = plan.scan("b", ["k"], 1)
    var k = plan.exprs.column("k")
    var j = plan.exprs.column("j")
    with assert_raises(contains="2 keys on the left and 1 on the right"):
        _ = plan.join(left, right, [k, j], [k], JoinKind.INNER)


def test_a_sort_needs_a_direction_for_each_key() raises:
    var plan = Plan()
    var scan = plan.scan("a", ["k", "j"], 0)
    var k = plan.exprs.column("k")
    var j = plan.exprs.column("j")
    with assert_raises(contains="2 keys has 1 directions"):
        _ = plan.sort(scan, [k, j], [True], [False, False])


def test_a_limit_cannot_be_negative() raises:
    var plan = Plan()
    var scan = plan.scan("a", ["k"], 0)
    with assert_raises(contains="cannot skip -1 rows"):
        _ = plan.limit(scan, -1, 10)
    with assert_raises(contains="cannot keep -2 rows"):
        _ = plan.limit(scan, 0, -2)


def test_a_union_needs_something_to_stack() raises:
    var plan = Plan()
    with assert_raises(contains="needs something to stack"):
        _ = plan.union(List[Int](), all=True)


def test_a_fold_cannot_be_a_filter_predicate() raises:
    var plan = Plan()
    var scan = plan.scan("a", ["v"], 0)
    var v = plan.exprs.column("v")
    var total = plan.exprs.aggregate(AggKind.SUM, v)
    var big = plan.exprs.binary(BinaryOp.GT, v, total)
    # This is a query the caller has not written yet rather than a filter with
    # an aggregate in it, and saying so here names the position.
    with assert_raises(contains="a filter predicate has to be read a row"):
        _ = plan.filter(scan, big)


def test_a_fold_cannot_be_a_group_key() raises:
    var plan = Plan()
    var scan = plan.scan("a", ["v"], 0)
    var v = plan.exprs.column("v")
    var total = plan.exprs.aggregate(AggKind.SUM, v)
    with assert_raises(contains="a group key has to be read a row"):
        _ = plan.aggregate(scan, [total], List[Int](), ["k"])


def test_a_fold_cannot_be_a_join_key() raises:
    var plan = Plan()
    var left = plan.scan("a", ["v"], 0)
    var right = plan.scan("b", ["v"], 1)
    var v = plan.exprs.column("v")
    var total = plan.exprs.aggregate(AggKind.SUM, v)
    with assert_raises(contains="a join key has to be read a row"):
        _ = plan.join(left, right, [v], [total], JoinKind.INNER)


def test_a_fold_cannot_be_a_sort_key() raises:
    var plan = Plan()
    var scan = plan.scan("a", ["v"], 0)
    var v = plan.exprs.column("v")
    var total = plan.exprs.aggregate(AggKind.SUM, v)
    with assert_raises(contains="a sort key has to be read a row"):
        _ = plan.sort(scan, [total], [True], [False])


def test_an_input_is_always_built_before_the_node_that_reads_it() raises:
    var plan = Plan()
    var a = plan.scan("a", ["k"], 0)
    var b = plan.scan("b", ["k"], 1)
    var k = plan.exprs.column("k")
    var joined = plan.join(a, b, [k], [k], JoinKind.INNER)
    var limited = plan.limit(joined, 0, 10)
    # A pass that walks the arena backwards visits every node after its inputs
    # only if this holds.
    for at in range(len(plan)):
        ref node = plan.nodes[at]
        for i in range(len(node.inputs)):
            assert_true(node.inputs[i] < at, "an input sits below its reader")
    assert_true(limited > joined, "and the plan was built bottom up")


def test_nine_kinds_are_enough_for_a_real_query() raises:
    # TPC-H q3, which is the smallest query that needs a join, an aggregate, a
    # sort and a limit at once. If nine kinds were not enough this is where it
    # would show.
    var plan = Plan()
    var customer = plan.scan("customer", ["c_custkey", "c_mktsegment"], 0)
    var orders = plan.scan("orders", ["o_orderkey", "o_custkey"], 1)

    var segment = plan.exprs.column("c_mktsegment")
    var building = plan.exprs.literal(Value(String("BUILDING")))
    var wanted = plan.exprs.binary(BinaryOp.EQ, segment, building)
    var kept = plan.filter(customer, wanted)

    var c_key = plan.exprs.column("c_custkey")
    var o_custkey = plan.exprs.column("o_custkey")
    var joined = plan.join(kept, orders, [c_key], [o_custkey], JoinKind.INNER)

    var o_key = plan.exprs.column("o_orderkey")
    var price = plan.exprs.column("o_totalprice")
    var revenue = plan.exprs.aggregate(AggKind.SUM, price)
    var grouped = plan.aggregate(
        joined, [o_key], [revenue], ["o_orderkey", "revenue"]
    )

    var out = plan.exprs.column("revenue")
    var ranked = plan.sort(grouped, [out], [True], [False])
    var top = plan.limit(ranked, 0, 10)

    assert_equal(
        explain(plan, top),
        (
            "LIMIT 10\n"
            "  SORT [revenue desc]\n"
            "    AGGREGATE [o_orderkey] -> [sum(o_totalprice)]\n"
            "      JOIN inner [c_custkey = o_custkey]\n"
            "        FILTER c_mktsegment == BUILDING\n"
            "          SCAN customer [c_custkey, c_mktsegment]\n"
            "        SCAN orders [o_orderkey, o_custkey]\n"
        ),
        "the whole query, indented by depth",
    )


def test_a_column_that_says_which_input_it_is_from_prints_it() raises:
    var tree = Expressions()
    var theirs = tree.column_of(1, "key")
    assert_equal(render_expr(tree, theirs), "#1.key", "the input and the name")


def test_a_column_that_says_nothing_about_an_input_prints_the_name() raises:
    var tree = Expressions()
    var plain = tree.column("key")
    assert_equal(render_expr(tree, plain), "key", "the name on its own")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
