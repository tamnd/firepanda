"""Aggregate and window classification, and the group key rule over it.

Every rule here was measured against DuckDB 1.5. The ones worth naming are the
asymmetries, because they are the ones that look like bugs until you see the
query that needs them: an aggregate may sit inside a window and a window may
not sit inside an aggregate, an aggregate exempts its arguments from the group
key rule and a window does not, and the same group key complaint is worded two
different ways depending on which clause asked.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.sql.ast import Ast, EXPR_FUNCTION, Expr, LITERAL_NUMBER
from firepanda.sql.classify import (
    CLAUSE_GROUP,
    CLAUSE_HAVING,
    CLAUSE_ORDER,
    CLAUSE_SELECT,
    CLAUSE_WHERE,
    aggregate_names,
    check_clause,
    check_nesting,
    covered,
    inspect,
    is_aggregate,
    needs_over,
    not_grouped,
    not_grouped_in_having,
    same,
    window_only_names,
)


def _parts(*names: String) -> List[String]:
    """A name run for a column reference.

    Args:
        names: The parts, outermost first.

    Returns:
        The list.
    """
    var out = List[String]()
    for name in names:
        out.append(name)
    return out^


def _column(mut ast: Ast, name: String) -> UInt32:
    """A bare column reference.

    Args:
        ast: The arenas.
        name: The column.

    Returns:
        The node.
    """
    return ast.column(_parts(name))


def _sum(mut ast: Ast, argument: UInt32) -> UInt32:
    """`sum(x)`.

    Args:
        ast: The arenas.
        argument: What to sum.

    Returns:
        The node.
    """
    var args = List[UInt32]()
    args.append(argument)
    return ast.call("sum", args)


def _over(
    mut ast: Ast, name: String, arguments: List[UInt32], window: UInt32
) -> UInt32:
    """A call with an `OVER` on it.

    The arena has no builder that attaches a window, because the transformer
    is the only thing that builds one and it writes the node out by hand.

    Args:
        ast: The arenas.
        name: The function name.
        arguments: The arguments, in order.
        window: The `EXPR_WINDOW`.

    Returns:
        The node.
    """
    var names = List[UInt32]()
    names.append(ast.intern(name))
    return ast.add(
        Expr(
            kind=EXPR_FUNCTION,
            token=0,
            b=window,
            children=ast.run(arguments),
            payload=ast.run(names),
        )
    )


def test_a_plain_expression_holds_neither() raises:
    var ast = Ast()
    var node = ast.binary("+", _column(ast, "a"), _column(ast, "b"))
    var uses = inspect(ast, node)
    assert_true(uses.plain())
    assert_false(uses.aggregate)
    assert_false(uses.window)


def test_an_aggregate_is_found_under_an_operator() raises:
    var ast = Ast()
    var node = ast.binary(
        "+", _sum(ast, _column(ast, "a")), ast.literal(LITERAL_NUMBER, "1")
    )
    var uses = inspect(ast, node)
    assert_true(uses.aggregate)
    assert_false(uses.window)


def test_a_call_with_over_on_it_is_a_window_and_not_an_aggregate() raises:
    var ast = Ast()
    var args = List[UInt32]()
    args.append(_column(ast, "a"))
    var node = _over(ast, "sum", args, ast.window())
    var uses = inspect(ast, node)
    assert_true(uses.window)
    assert_false(uses.aggregate)


def test_an_aggregate_inside_a_subquery_belongs_to_the_subquery() raises:
    # The walk stops at the subquery, so `select a from t where exists
    # (select sum(b) ...)` does not turn the outer query into an aggregate
    # one. Following the index would read the statement arena as if it were
    # the expression arena, which is the other reason to stop.
    var ast = Ast()
    var inner = _sum(ast, _column(ast, "b"))
    var statement = ast.select(ast.query())
    var node = ast.exists(statement)
    assert_true(inspect(ast, node).plain())
    assert_true(inspect(ast, inner).aggregate)


def test_a_window_may_hold_an_aggregate() raises:
    var ast = Ast()
    var args = List[UInt32]()
    args.append(_sum(ast, _column(ast, "a")))
    var node = _over(ast, "sum", args, ast.window())
    check_nesting(ast, node)


def test_an_aggregate_may_not_hold_an_aggregate() raises:
    var ast = Ast()
    var node = _sum(ast, _sum(ast, _column(ast, "a")))
    with assert_raises(
        contains="Binder Error: aggregate function calls cannot be nested"
    ):
        check_nesting(ast, node)


def test_an_aggregate_may_not_hold_a_window() raises:
    var ast = Ast()
    var args = List[UInt32]()
    args.append(_column(ast, "a"))
    var inner = _over(ast, "row_number", List[UInt32](), ast.window())
    var node = _sum(ast, inner)
    with assert_raises(
        contains=(
            "Binder Error: aggregate function calls cannot contain window"
            " function calls"
        )
    ):
        check_nesting(ast, node)


def test_a_window_may_not_hold_a_window() raises:
    var ast = Ast()
    var inner = _over(ast, "row_number", List[UInt32](), ast.window())
    var partition = List[UInt32]()
    partition.append(inner)
    var node = _over(
        ast, "row_number", List[UInt32](), ast.window(partition=partition)
    )
    with assert_raises(
        contains=(
            "Parser Error: window functions are not allowed in window"
            " definitions"
        )
    ):
        check_nesting(ast, node)


def test_where_refuses_an_aggregate() raises:
    var ast = Ast()
    var node = _sum(ast, _column(ast, "a"))
    with assert_raises(
        contains="Binder Error: WHERE clause cannot contain aggregates!"
    ):
        check_clause(ast, node, CLAUSE_WHERE)


def test_where_refuses_a_window() raises:
    var ast = Ast()
    var node = _over(ast, "row_number", List[UInt32](), ast.window())
    with assert_raises(
        contains="Binder Error: WHERE clause cannot contain window functions!"
    ):
        check_clause(ast, node, CLAUSE_WHERE)


def test_group_by_refuses_both() raises:
    var ast = Ast()
    var aggregate = _sum(ast, _column(ast, "a"))
    with assert_raises(
        contains="Binder Error: GROUP BY clause cannot contain aggregates!"
    ):
        check_clause(ast, aggregate, CLAUSE_GROUP)
    var window = _over(ast, "row_number", List[UInt32](), ast.window())
    with assert_raises(
        contains=(
            "Binder Error: GROUP BY clause cannot contain window functions!"
        )
    ):
        check_clause(ast, window, CLAUSE_GROUP)


def test_having_takes_an_aggregate_and_refuses_a_window() raises:
    var ast = Ast()
    check_clause(ast, _sum(ast, _column(ast, "a")), CLAUSE_HAVING)
    var window = _over(ast, "row_number", List[UInt32](), ast.window())
    with assert_raises(
        contains="Binder Error: HAVING clause cannot contain window functions!"
    ):
        check_clause(ast, window, CLAUSE_HAVING)


def test_order_by_takes_both() raises:
    var ast = Ast()
    check_clause(ast, _sum(ast, _column(ast, "a")), CLAUSE_ORDER)
    var window = _over(ast, "row_number", List[UInt32](), ast.window())
    check_clause(ast, window, CLAUSE_ORDER)
    check_clause(ast, window, CLAUSE_SELECT)


def test_a_query_that_does_not_aggregate_may_say_anything() raises:
    var ast = Ast()
    var node = _column(ast, "a")
    assert_equal(covered(ast, node, List[UInt32](), False), 0)


def test_a_reference_with_no_group_key_is_reported() raises:
    var ast = Ast()
    var node = _column(ast, "a")
    assert_equal(covered(ast, node, List[UInt32](), True), node)


def test_a_group_key_covers_the_reference() raises:
    var ast = Ast()
    var node = _column(ast, "a")
    var keys = List[UInt32]()
    keys.append(_column(ast, "a"))
    assert_equal(covered(ast, node, keys, True), 0)


def test_an_expression_built_out_of_keys_is_covered() raises:
    # `select a + 1 from t group by a` binds, so the check is over the
    # references rather than over the whole select list entry.
    var ast = Ast()
    var node = ast.binary(
        "+", _column(ast, "a"), ast.literal(LITERAL_NUMBER, "1")
    )
    var keys = List[UInt32]()
    keys.append(_column(ast, "a"))
    assert_equal(covered(ast, node, keys, True), 0)


def test_a_key_matches_a_whole_expression_too() raises:
    var ast = Ast()
    var node = ast.binary(
        "+", _column(ast, "a"), ast.literal(LITERAL_NUMBER, "1")
    )
    var keys = List[UInt32]()
    keys.append(
        ast.binary("+", _column(ast, "a"), ast.literal(LITERAL_NUMBER, "1"))
    )
    assert_equal(covered(ast, node, keys, True), 0)


def test_an_aggregate_covers_what_is_in_it() raises:
    var ast = Ast()
    var node = _sum(ast, _column(ast, "a"))
    assert_equal(covered(ast, node, List[UInt32](), True), 0)


def test_a_window_does_not_cover_what_is_in_it() raises:
    # `max(a) over (partition by sum(b)) from t group by g` is refused over
    # the `a`, which is the difference between the two that matters.
    var ast = Ast()
    var args = List[UInt32]()
    var reference = _column(ast, "a")
    args.append(reference)
    var node = _over(ast, "max", args, ast.window())
    assert_equal(covered(ast, node, List[UInt32](), True), reference)


def test_an_aggregate_inside_a_window_covers_its_own_argument() raises:
    var ast = Ast()
    var args = List[UInt32]()
    args.append(_sum(ast, _column(ast, "a")))
    var node = _over(ast, "sum", args, ast.window())
    assert_equal(covered(ast, node, List[UInt32](), True), 0)


def test_two_spellings_of_one_name_are_one_expression() raises:
    var ast = Ast()
    assert_true(same(ast, ast.column(_parts("A")), ast.column(_parts("a"))))
    assert_false(same(ast, _column(ast, "a"), _column(ast, "b")))


def test_two_spellings_of_one_value_are_two_expressions() raises:
    var ast = Ast()
    assert_false(
        same(
            ast,
            ast.literal(LITERAL_NUMBER, "1"),
            ast.literal(LITERAL_NUMBER, "1.0"),
        )
    )


def test_a_different_operator_is_a_different_expression() raises:
    var ast = Ast()
    var left = ast.binary("+", _column(ast, "a"), _column(ast, "b"))
    var right = ast.binary("-", _column(ast, "a"), _column(ast, "b"))
    assert_false(same(ast, left, right))


def test_the_operands_are_compared_in_order() raises:
    var ast = Ast()
    var left = ast.binary("-", _column(ast, "a"), _column(ast, "b"))
    var right = ast.binary("-", _column(ast, "b"), _column(ast, "a"))
    assert_false(same(ast, left, right))


def test_a_cast_target_is_part_of_the_expression() raises:
    var ast = Ast()
    var left = ast.cast(_column(ast, "a"), "INTEGER")
    var right = ast.cast(_column(ast, "a"), "BIGINT")
    assert_false(same(ast, left, right))
    assert_true(same(ast, left, ast.cast(_column(ast, "a"), "INTEGER")))


def test_the_group_key_complaint_has_two_wordings() raises:
    assert_equal(
        not_grouped("a"),
        (
            'Binder Error: column "a" must appear in the GROUP BY clause or'
            " must be part of an aggregate function.\nEither add it to the"
            ' GROUP BY list, or use "ANY_VALUE(a)" if the exact value of "a"'
            " is not important."
        ),
    )
    assert_equal(
        not_grouped_in_having("a"),
        (
            "Binder Error: column a must appear in the GROUP BY clause or be"
            " used in an aggregate function"
        ),
    )


def test_the_aggregate_names_are_the_ones_duckdb_has() raises:
    assert_equal(len(aggregate_names()), 88)
    assert_equal(len(window_only_names()), 13)
    assert_true(is_aggregate("SUM"))
    assert_true(is_aggregate("sum"))
    assert_false(is_aggregate("abs"))
    assert_false(is_aggregate("nope"))


def test_a_name_is_not_matched_inside_another_name() raises:
    # `sum` sits inside `kahan_sum` in the table, and both are aggregates, so
    # the test that the delimiters work needs a name that is not one.
    assert_true(is_aggregate("kahan_sum"))
    assert_false(is_aggregate("kahan"))
    assert_false(is_aggregate("um"))
    assert_false(is_aggregate("ma"))


def test_thirteen_of_them_will_not_be_called_without_over() raises:
    assert_true(needs_over("row_number"))
    assert_true(needs_over("LAG"))
    assert_true(needs_over("first_value"))
    assert_false(needs_over("sum"))
    assert_false(needs_over("first"))
    for name in window_only_names():
        assert_true(is_aggregate(name))


def test_a_quantified_comparison_is_walked_on_the_left_only() raises:
    # The right side is a statement index, so a walk that followed it would be
    # reading the wrong arena, and an aggregate in there belongs to that query.
    var ast = Ast()
    var node = ast.quantified(
        _sum(ast, _column(ast, "a")), ">", True, ast.query()
    )
    var uses = inspect(ast, node)
    assert_true(uses.aggregate)
    assert_false(uses.window)


def test_two_quantified_comparisons_are_two_expressions() raises:
    var ast = Ast()
    var left = ast.quantified(_column(ast, "a"), ">", True, ast.query())
    var right = ast.quantified(_column(ast, "a"), ">", True, ast.query())
    assert_true(same(ast, left, left))
    assert_false(same(ast, left, right))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
