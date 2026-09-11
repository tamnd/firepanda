"""Tests for the expression simplification pass.

Every one of these is an expression in and a printed expression out, compared as
text, which is how an optimizer is tested everywhere and is easier to read than
an assertion about which index ended up holding what. A test that says the
answer is `l_quantity < 24.0` says what a person would have checked by eye. A
test that says node seven is a binary whose second child is node five says
nothing at all when it fails.

There are four groups. The folds, which are the reason the pass exists and are
asserted as the value the kernel would really have produced. The rewrites that
change shape without changing meaning, which are the turn round and the
flattening. The identities, which are small and are what a fold leaves behind.
And the refusals, where the pass hands an expression back rather than answering,
because an expression the kernel would raise on is the user's error to see from
execution and not this pass's to report.

The last test is a whole predicate of the shape TPC-H q6 has, because that is
the one where the cost of not folding is six million subtractions of two
constants.
"""

from std.testing import TestSuite, assert_equal

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.unary import UnaryOp
from firepanda.plan.expr import ExprKind, Expressions
from firepanda.plan.node import Plan
from firepanda.plan.print import explain, render_expr
from firepanda.plan.simplify import simplify, simplify_expr


def _done(mut tree: Expressions, root: Int) raises -> String:
    """Simplifies an expression and gives back what it prints as."""
    return render_expr(tree, simplify_expr(tree, root))


def test_two_constants_added_become_one() raises:
    var tree = Expressions()
    var a = tree.literal(Value(Int64(40)))
    var b = tree.literal(Value(Int64(2)))
    assert_equal(
        _done(tree, tree.binary(BinaryOp.ADD, a, b)),
        "42",
        "the whole expression is the answer",
    )


def test_a_fold_keeps_the_type_the_kernel_would_have_given() raises:
    # An int64 over a float64 promotes to float64 in the kernel, so the folded
    # constant has to be a float64 as well. Folding this to an integer would be
    # a different column type arriving at whatever reads it.
    var tree = Expressions()
    var a = tree.literal(Value(Int64(7)))
    var b = tree.literal(Value(Float64(2.0)))
    var at = simplify_expr(tree, tree.binary(BinaryOp.DIV, a, b))
    assert_equal(render_expr(tree, at), "3.5", "the value")
    assert_equal(
        tree.nodes[at].value.type,
        LogicalType.FLOAT64,
        "and the promoted type, not the integer one",
    )


def test_a_fold_happens_under_another_fold() raises:
    var tree = Expressions()
    var a = tree.literal(Value(Int64(2)))
    var b = tree.literal(Value(Int64(3)))
    var c = tree.literal(Value(Int64(4)))
    var inner = tree.binary(BinaryOp.MUL, a, b)
    assert_equal(
        _done(tree, tree.binary(BinaryOp.ADD, inner, c)),
        "10",
        "the whole tree collapses in one pass, bottom up",
    )


def test_a_negation_of_a_constant_folds() raises:
    var tree = Expressions()
    var a = tree.literal(Value(Int64(5)))
    assert_equal(_done(tree, tree.unary(UnaryOp.NEG, a)), "-5", "the answer")


def test_a_comparison_between_two_constants_folds_to_a_bool() raises:
    var tree = Expressions()
    var a = tree.literal(Value(Int64(1)))
    var b = tree.literal(Value(Int64(1)))
    var at = simplify_expr(tree, tree.binary(BinaryOp.EQ, a, b))
    assert_equal(
        tree.nodes[at].value.type, LogicalType.BOOL, "a comparison answers bool"
    )
    assert_equal(_truthy(tree, at), True, "and this one is true")


def _truthy(tree: Expressions, at: Int) -> Bool:
    return tree.nodes[at].value.bits != 0


def test_two_strings_compare_at_plan_time() raises:
    # The fold builds a one row text column and runs the same comparison loop a
    # column would have run, which is the only reason text works here at all.
    var tree = Expressions()
    var a = tree.literal(Value(String("BUILDING")))
    var b = tree.literal(Value(String("MACHINERY")))
    var at = simplify_expr(tree, tree.binary(BinaryOp.LT, a, b))
    assert_equal(tree.nodes[at].kind, ExprKind.LITERAL, "it folded")
    assert_equal(_truthy(tree, at), True, "B sorts before M")


def test_a_column_is_left_alone() raises:
    var tree = Expressions()
    var x = tree.column("l_quantity")
    var one = tree.literal(Value(Int64(1)))
    assert_equal(
        _done(tree, tree.binary(BinaryOp.ADD, x, one)),
        "l_quantity + 1",
        "nothing here reads no rows, so nothing folds",
    )


def test_a_constant_on_the_left_of_a_comparison_moves_right() raises:
    var tree = Expressions()
    var cut = tree.literal(Value(Float64(24.0)))
    var q = tree.column("l_quantity")
    assert_equal(
        _done(tree, tree.binary(BinaryOp.GT, cut, q)),
        "l_quantity < 24.0",
        "and the operator turns round with it",
    )


def test_a_constant_on_the_left_of_a_subtraction_stays_there() raises:
    var tree = Expressions()
    var ten = tree.literal(Value(Int64(10)))
    var x = tree.column("x")
    assert_equal(
        _done(tree, tree.binary(BinaryOp.SUB, ten, x)),
        "10 - x",
        "because 10 - x is not x - 10",
    )


def test_two_constants_do_not_send_the_pass_round_in_circles() raises:
    # Both sides are literals, so the fold fires first and there is nothing left
    # to turn round. If the turn round ran first the pass would swap the two
    # forever, which is what the round bound exists to survive.
    var tree = Expressions()
    var a = tree.literal(Value(Int64(3)))
    var b = tree.literal(Value(Int64(4)))
    var at = simplify_expr(tree, tree.binary(BinaryOp.LT, a, b))
    assert_equal(tree.nodes[at].kind, ExprKind.LITERAL, "it folded")
    assert_equal(_truthy(tree, at), True, "3 is less than 4")


def test_a_nested_conjunction_flattens() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var b = tree.column("b")
    var c = tree.column("c")
    var inner = tree.call("and", [b, c], rowwise=True)
    assert_equal(
        _done(tree, tree.call("and", [a, inner], rowwise=True)),
        "and(a, b, c)",
        "one call with three arguments",
    )


def test_a_disjunction_flattens_the_same_way() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var b = tree.column("b")
    var c = tree.column("c")
    var inner = tree.call("or", [a, b], rowwise=True)
    assert_equal(
        _done(tree, tree.call("or", [inner, c], rowwise=True)),
        "or(a, b, c)",
        "and it flattens from the left as well as the right",
    )


def test_an_or_inside_an_and_stays_where_it_is() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var b = tree.column("b")
    var c = tree.column("c")
    var inner = tree.call("or", [b, c], rowwise=True)
    assert_equal(
        _done(tree, tree.call("and", [a, inner], rowwise=True)),
        "and(a, or(b, c))",
        "flattening two different connectives would change the meaning",
    )


def test_a_true_drops_out_of_a_conjunction() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var yes = tree.literal(Value(True))
    assert_equal(
        _done(tree, tree.call("and", [a, yes], rowwise=True)),
        "a",
        "one argument left, so the call goes as well",
    )


def test_a_false_collapses_a_conjunction() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var no = tree.literal(Value(False))
    var at = simplify_expr(tree, tree.call("and", [a, no], rowwise=True))
    assert_equal(tree.nodes[at].kind, ExprKind.LITERAL, "it is a constant now")
    assert_equal(_truthy(tree, at), False, "and the constant is false")


def test_a_true_collapses_a_disjunction() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var yes = tree.literal(Value(True))
    var at = simplify_expr(tree, tree.call("or", [a, yes], rowwise=True))
    assert_equal(tree.nodes[at].kind, ExprKind.LITERAL, "it is a constant")
    assert_equal(_truthy(tree, at), True, "and the constant is true")


def test_a_conjunction_of_nothing_but_truths_is_true() raises:
    var tree = Expressions()
    var one = tree.literal(Value(True))
    var two = tree.literal(Value(True))
    var at = simplify_expr(tree, tree.call("and", [one, two], rowwise=True))
    assert_equal(tree.nodes[at].kind, ExprKind.LITERAL, "nothing is left")
    assert_equal(_truthy(tree, at), True, "and what is left is the identity")


def test_a_null_is_not_a_truth_and_stays_in_the_conjunction() raises:
    # A null under `and` is not false. `a and null` is null when `a` is true,
    # so dropping the null would change the answer on those rows.
    var tree = Expressions()
    var a = tree.column("a")
    var none = tree.literal(Value(null=LogicalType.BOOL))
    assert_equal(
        _done(tree, tree.call("and", [a, none], rowwise=True)),
        "and(a, null)",
        "the null is still an argument",
    )


def test_two_negations_cancel() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var once = tree.call("not", [a], rowwise=True)
    assert_equal(
        _done(tree, tree.call("not", [once], rowwise=True)),
        "a",
        "and what is left is the column itself",
    )


def test_three_negations_leave_one() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var once = tree.call("not", [a], rowwise=True)
    var twice = tree.call("not", [once], rowwise=True)
    assert_equal(
        _done(tree, tree.call("not", [twice], rowwise=True)),
        "not(a)",
        "the pass runs to a fixed point rather than once",
    )


def test_a_negated_constant_folds() raises:
    var tree = Expressions()
    var yes = tree.literal(Value(True))
    var at = simplify_expr(tree, tree.call("not", [yes], rowwise=True))
    assert_equal(tree.nodes[at].kind, ExprKind.LITERAL, "it folded")
    assert_equal(_truthy(tree, at), False, "not true is false")


def test_an_unknown_call_is_left_alone() raises:
    var tree = Expressions()
    var a = tree.column("a")
    assert_equal(
        _done(tree, tree.call("upper", [a], rowwise=True)),
        "upper(a)",
        "the pass knows three functions and does not guess about the rest",
    )


def test_a_division_by_a_constant_zero_folds_to_what_the_kernel_says() raises:
    # Division promotes to float64 in this library, so one over zero is an
    # infinity rather than an error, on a column and therefore here. The pass
    # does not get an opinion about that: it asks and writes down the answer.
    # The test is here because the tempting alternative, refusing to fold a
    # division by zero, would have produced a plan that computes a different
    # thing from the one it replaced.
    var tree = Expressions()
    var a = tree.literal(Value(Int64(1)))
    var b = tree.literal(Value(Int64(0)))
    assert_equal(
        _done(tree, tree.binary(BinaryOp.DIV, a, b)),
        "inf",
        "the same infinity the kernel would have put in every row",
    )


def test_an_expression_the_kernel_refuses_is_handed_back() raises:
    # Text and an integer have no common type, so the kernel raises. The pass
    # returns the expression it was given rather than failing the plan, because
    # a type error is binding's to report and binding has already run.
    var tree = Expressions()
    var s = tree.literal(Value(String("x")))
    var n = tree.literal(Value(Int64(3)))
    var at = tree.binary(BinaryOp.ADD, s, n)
    var after = simplify_expr(tree, at)
    assert_equal(after, at, "the same node came back")
    assert_equal(render_expr(tree, after), "x + 3", "unchanged")


def test_an_aggregate_is_not_folded_even_over_a_constant() raises:
    # `sum(1)` is the row count, so folding it to 1 would be wrong. This is the
    # same answer `input_independent` gives and the pass does not second guess
    # it.
    var tree = Expressions()
    var one = tree.literal(Value(Int64(1)))
    assert_equal(
        _done(tree, tree.aggregate(AggKind.SUM, one)),
        "sum(1)",
        "left alone",
    )


def test_the_q6_predicate_folds_its_date_arithmetic() raises:
    # The shape that pays for this pass. Two of the four comparisons are against
    # arithmetic over constants, which without folding is evaluated once per row
    # over six million rows and answers the same thing every time.
    var plan = Plan()
    var scan = plan.scan(
        "lineitem", ["l_shipdate", "l_discount", "l_quantity"], 0
    )
    var shipdate = plan.exprs.column("l_shipdate")
    var discount = plan.exprs.column("l_discount")
    var quantity = plan.exprs.column("l_quantity")

    var base = plan.exprs.literal(Value(Float64(0.05)))
    var slack = plan.exprs.literal(Value(Float64(0.01)))
    var low = plan.exprs.binary(BinaryOp.SUB, base, slack)
    var high = plan.exprs.binary(BinaryOp.ADD, base, slack)

    var after = plan.exprs.binary(BinaryOp.GE, discount, low)
    var before = plan.exprs.binary(BinaryOp.LE, discount, high)
    var cap = plan.exprs.literal(Value(Float64(24.0)))
    var small = plan.exprs.binary(BinaryOp.LT, quantity, cap)
    var window = plan.exprs.literal(Value(Int64(1994)))
    var year = plan.exprs.binary(BinaryOp.GE, shipdate, window)

    var inner = plan.exprs.call("and", [after, before], rowwise=True)
    var outer = plan.exprs.call("and", [inner, small], rowwise=True)
    var whole = plan.exprs.call("and", [outer, year], rowwise=True)
    var kept = plan.filter(scan, whole)

    simplify(plan, kept)
    # The upper bound prints as 0.060000000000000005 and that is not a defect
    # in the fold, it is what 0.05 + 0.01 is in float64 and therefore what the
    # kernel would have computed on all six million rows. A fold that tidied it
    # to 0.06 would be answering a different query from the one written.
    assert_equal(
        explain(plan, kept),
        (
            "FILTER and(l_discount >= 0.04, l_discount <="
            " 0.060000000000000005, l_quantity < 24.0, l_shipdate >= 1994)\n"
            "  SCAN lineitem [l_shipdate, l_discount, l_quantity]\n"
        ),
        "both bounds folded and the three conjunctions became one",
    )


def test_a_plan_with_nothing_to_do_comes_back_identical() raises:
    var plan = Plan()
    var scan = plan.scan("orders", ["o_orderkey", "o_totalprice"], 0)
    var price = plan.exprs.column("o_totalprice")
    var cap = plan.exprs.literal(Value(Float64(1000.0)))
    var kept = plan.filter(scan, plan.exprs.binary(BinaryOp.GT, price, cap))
    var before = explain(plan, kept)
    simplify(plan, kept)
    assert_equal(explain(plan, kept), before, "byte for byte the same plan")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
