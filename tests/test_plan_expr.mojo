"""Tests for the expression arena and the three analyses over it.

The arena itself has little to check beyond that an index means what it meant,
so most of what is here is the analyses, and the shape of most tests is the
same: build a small tree, ask one question, check the answer. The trees are
small on purpose, because the analyses are recursive walks and a walk that is
wrong is wrong on two nodes as readily as on twenty.

The cases worth naming are the ones where the obvious answer is the wrong one.
A sum of a constant is not input independent, because it reads the height of the
input even though it reads none of its values. A cumulative sum is not
elementwise even though every one of its arguments is. And the table set of an
unbound column is not the empty set, because the empty set is what an input
independent expression has and it is the answer that tells pushdown a predicate
can go anywhere.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_raises
from std.testing import assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.unary import UnaryOp
from firepanda.plan.expr import UNBOUND, ExprKind, Expressions


def bound(mut tree: Expressions, var name: String, at: Int, table: Int) -> Int:
    """Builds a column reference that binding has already been over.

    Binding is not written yet, so the tests that need a bound column write the
    position and the table on themselves. When `bind.mojo` lands this becomes a
    call to it and the tests below do not change.

    Args:
        tree: The arena.
        name: The column name.
        at: The position.
        table: The table.

    Returns:
        The index of the new node.
    """
    var col = tree.column(name^)
    tree.nodes[col].at = at
    tree.nodes[col].table = table
    return col


def test_an_index_keeps_meaning_what_it_meant() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var one = tree.literal(Value(Int64(1)))
    var sum = tree.binary(BinaryOp.ADD, a, one)
    assert_equal(len(tree), 3, "three nodes went in")
    assert_true(tree.nodes[a].kind == ExprKind.COLUMN, "a is still the column")
    assert_equal(tree.nodes[a].name, "a", "and still by that name")
    assert_true(tree.nodes[sum].kind == ExprKind.BINARY, "the sum is a binary")
    assert_equal(tree.nodes[sum].children[0], a, "with a on the left")
    assert_equal(tree.nodes[sum].children[1], one, "and the constant right")


def test_a_child_is_always_built_before_its_parent() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var b = tree.column("b")
    var product = tree.binary(BinaryOp.MUL, a, b)
    var negated = tree.unary(UnaryOp.NEG, product)
    # Several passes walk the arena backwards instead of traversing, and that
    # only visits a node after its children if this holds.
    for at in range(len(tree)):
        ref node = tree.nodes[at]
        for i in range(len(node.children)):
            assert_true(node.children[i] < at, "a child sits below its parent")
    assert_true(negated > product, "and the tree was built bottom up")


def test_an_operand_outside_the_arena_is_refused() raises:
    var tree = Expressions()
    var a = tree.column("a")
    with assert_raises(contains="is not in an arena of 1"):
        _ = tree.unary(UnaryOp.NEG, 7)
    with assert_raises(contains="is not in an arena of 1"):
        _ = tree.binary(BinaryOp.ADD, a, -1)


def test_a_constant_knows_its_type_before_binding_runs() raises:
    var tree = Expressions()
    var one = tree.literal(Value(Int64(1)))
    var a = tree.column("a")
    assert_true(
        tree.nodes[one].type == LogicalType.INT64, "a constant carries its type"
    )
    assert_true(
        tree.nodes[a].type == LogicalType.NULL, "a column waits for binding"
    )
    assert_equal(tree.nodes[a].at, UNBOUND, "with no position on it yet")
    assert_equal(tree.nodes[a].table, UNBOUND, "and no table")


def test_a_cast_carries_the_type_it_was_asked_for() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var widened = tree.cast(LogicalType.FLOAT64, a)
    assert_true(
        tree.nodes[widened].type == LogicalType.FLOAT64, "the target is on it"
    )


def test_a_window_says_where_its_partition_keys_end() raises:
    var tree = Expressions()
    var v = tree.column("v")
    var k = tree.column("k")
    var t = tree.column("t")
    var ranked = tree.window(AggKind.SUM, v, [k], [t])
    ref node = tree.nodes[ranked]
    assert_equal(len(node.children), 3, "the value and one key of each sort")
    assert_equal(node.children[0], v, "the value comes first")
    assert_equal(node.parts, 1, "one partition key")
    assert_equal(node.children[1], k, "which is the child after the value")
    assert_equal(node.children[2], t, "and the order key is the rest")


def test_arithmetic_over_columns_is_elementwise() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var b = tree.column("b")
    var two = tree.literal(Value(Int64(2)))
    var scaled = tree.binary(BinaryOp.MUL, b, two)
    var total = tree.binary(BinaryOp.ADD, a, scaled)
    assert_true(tree.elementwise(total), "a + b * 2 is a row at a time")


def test_an_aggregate_is_not_elementwise() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var total = tree.aggregate(AggKind.SUM, a)
    assert_false(tree.elementwise(total), "a sum folds many rows into one")


def test_an_aggregate_makes_everything_above_it_not_elementwise() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var b = tree.column("b")
    var total = tree.aggregate(AggKind.SUM, a)
    var share = tree.binary(BinaryOp.DIV, b, total)
    assert_false(tree.elementwise(share), "the division inherits the fold")


def test_a_call_is_elementwise_when_the_function_is() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var rounded = tree.call("round", [a], rowwise=True)
    var running = tree.call("cum_sum", [a], rowwise=False)
    assert_true(tree.elementwise(rounded), "round reads its own row")
    assert_false(
        tree.elementwise(running), "a cumulative sum reads the ones before it"
    )


def test_a_window_is_not_elementwise() raises:
    var tree = Expressions()
    var v = tree.column("v")
    var k = tree.column("k")
    var ranked = tree.window(AggKind.SUM, v, [k], List[Int]())
    assert_false(tree.elementwise(ranked), "a window reads its partition")


def test_a_conditional_is_elementwise_when_all_three_arms_are() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var zero = tree.literal(Value(Int64(0)))
    var positive = tree.binary(BinaryOp.GT, a, zero)
    var plain = tree.conditional(positive, a, zero)
    assert_true(tree.elementwise(plain), "three elementwise arms")

    var total = tree.aggregate(AggKind.SUM, a)
    var folded = tree.conditional(positive, a, total)
    assert_false(tree.elementwise(folded), "one arm that is not")


def test_arithmetic_over_constants_is_input_independent() raises:
    var tree = Expressions()
    var day = tree.literal(Value(Int32(10561)))
    var ninety = tree.literal(Value(Int32(90)))
    # This is q1's date predicate, and an engine without the analysis evaluates
    # it once a row rather than once a query.
    var cutoff = tree.binary(BinaryOp.SUB, day, ninety)
    assert_true(tree.input_independent(cutoff), "no row is read")


def test_a_column_is_not_input_independent() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var one = tree.literal(Value(Int64(1)))
    var shifted = tree.binary(BinaryOp.ADD, a, one)
    assert_false(tree.input_independent(a), "a column reads a row")
    assert_false(
        tree.input_independent(shifted), "and so does anything over it"
    )


def test_a_sum_of_a_constant_is_not_input_independent() raises:
    var tree = Expressions()
    var one = tree.literal(Value(Int64(1)))
    var counted = tree.aggregate(AggKind.SUM, one)
    # sum(1) is the row count, so it reads the input even though it reads none
    # of the values in it. Folding it to 1 at plan time would be wrong.
    assert_false(tree.input_independent(counted), "sum of one is the height")


def test_a_running_call_over_constants_is_not_input_independent() raises:
    var tree = Expressions()
    var one = tree.literal(Value(Int64(1)))
    var running = tree.call("cum_sum", [one], rowwise=False)
    assert_false(tree.input_independent(running), "the height decides how many")


def test_a_cast_of_a_constant_is_still_input_independent() raises:
    var tree = Expressions()
    var one = tree.literal(Value(Int64(1)))
    var widened = tree.cast(LogicalType.FLOAT64, one)
    assert_true(tree.input_independent(widened), "casting reads no row")


def test_the_table_set_of_a_bound_column_is_its_own_bit() raises:
    var tree = Expressions()
    var a = bound(tree, "a", 0, 0)
    var b = bound(tree, "b", 3, 2)
    assert_equal(tree.tables(a), UInt64(1), "table zero is bit zero")
    assert_equal(tree.tables(b), UInt64(4), "table two is bit two")


def test_the_table_set_of_a_tree_is_the_union_of_its_columns() raises:
    var tree = Expressions()
    var a = bound(tree, "a", 0, 0)
    var b = bound(tree, "b", 1, 1)
    var c = bound(tree, "c", 2, 1)
    var left = tree.binary(BinaryOp.EQ, a, b)
    var right = tree.binary(BinaryOp.GT, c, a)
    var both = tree.call("and", [left, right], rowwise=True)
    assert_equal(tree.tables(both), UInt64(3), "tables zero and one")


def test_an_input_independent_expression_reads_no_table() raises:
    var tree = Expressions()
    var one = tree.literal(Value(Int64(1)))
    var two = tree.literal(Value(Int64(2)))
    var three = tree.binary(BinaryOp.ADD, one, two)
    assert_equal(tree.tables(three), UInt64(0), "a constant reads nothing")


def test_the_table_set_of_an_unbound_column_is_refused() raises:
    var tree = Expressions()
    var a = tree.column("a")
    var one = tree.literal(Value(Int64(1)))
    var shifted = tree.binary(BinaryOp.ADD, a, one)
    # The empty set would be a wrong answer rather than an unknown one, and a
    # pushdown pass reading it would move this predicate past the scan that is
    # the only thing able to provide `a`.
    with assert_raises(contains="has no table until binding has run"):
        _ = tree.tables(shifted)


def test_a_table_too_far_out_for_the_mask_is_refused() raises:
    var tree = Expressions()
    var a = bound(tree, "a", 0, 64)
    with assert_raises(contains="does not fit in the mask"):
        _ = tree.tables(a)


def test_the_kinds_print_as_the_words_explain_uses() raises:
    assert_equal(String(ExprKind.COLUMN), "column", "a column reads as one")
    assert_equal(String(ExprKind.LITERAL), "literal", "and a constant as one")
    assert_equal(String(ExprKind.BINARY), "binary", "and a binary as one")
    assert_equal(String(ExprKind.WINDOW), "window", "and a window as one")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
