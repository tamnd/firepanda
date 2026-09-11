"""The four shapes a subquery is written in, and what each one requires.

Two halves. The first parses real queries and checks which shape came out of
the transformer, because the shape is what the rest of the binder switches on
and getting it from the node rather than from the text is the whole point. The
second builds a chain of levels by hand and asks which subqueries are
correlated, which is the question no amount of parsing answers.

Every message here was read off DuckDB 1.5, including the two that read wrong:
a subquery of one column reports `returns 1 columns`, and an `IN` whose types
do not match is reported as an `IN/ANY/ALL` problem whichever of the three was
written.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.dtype.logical import LogicalType
from firepanda.sql import Grammar, Transform
from firepanda.sql.ast import Ast, EXPR_QUANTIFIED, NO_NODE
from firepanda.sql.bind import SOURCE_FRAME, Scopes
from firepanda.sql.catalog import NOT_FOUND
from firepanda.sql.printer import print_stmt
from firepanda.sql.subquery import (
    SHAPE_EXISTS,
    SHAPE_IN,
    SHAPE_NONE,
    SHAPE_QUANTIFIED,
    SHAPE_SCALAR,
    about,
    cannot_compare,
    check_columns,
    correlated,
    outer_references,
    too_many_rows,
    wrong_column_count,
)


def _table(
    mut scopes: Scopes, level: Int, name: StringSlice, columns: List[String]
) -> Int:
    """Adds a binding with those columns, all of them integers.

    Args:
        scopes: The chain.
        level: The level to add at.
        name: What the query calls the binding.
        columns: The column names.

    Returns:
        The binding's position.
    """
    var at = scopes.add(level, name, SOURCE_FRAME, 0)
    for column in columns:
        scopes.levels[level].bindings[at].add(column, LogicalType.INT32)
    return at


def _printed(
    sql: StringSlice, grammar: Grammar, mut rules: Transform
) raises -> String:
    """Parses one statement and prints it back.

    Args:
        sql: The whole statement.
        grammar: A loaded grammar.
        rules: The jump table built from it.

    Returns:
        The printed text.

    Raises:
        Error: If it does not parse or holds something with no case.
    """
    var ast = Ast()
    return print_stmt(ast, rules.parse_statement(sql, grammar, ast), grammar)


def _filter(
    sql: StringSlice, grammar: Grammar, mut rules: Transform
) raises -> Ast:
    """Parses a statement and gives back the arenas it built.

    Args:
        sql: The whole statement.
        grammar: A loaded grammar.
        rules: The jump table built from it.

    Returns:
        The arenas.

    Raises:
        Error: If it does not parse or holds something with no case.
    """
    var ast = Ast()
    _ = rules.parse_statement(sql, grammar, ast)
    return ast^


def _shape_of(ast: Ast, kind_at: UInt32) raises -> UInt8:
    """The shape of one expression node.

    Args:
        ast: The arenas.
        kind_at: The node.

    Returns:
        The `SHAPE_` constant.

    Raises:
        Error: If the index is not in the arena.
    """
    return about(ast, kind_at).shape


def _found(ast: Ast, kind: UInt8) -> UInt32:
    """The first node of a kind in the expression arena.

    Args:
        ast: The arenas.
        kind: The `EXPR_` constant.

    Returns:
        The node index, or `NO_NODE` if there is none.
    """
    for at in range(1, len(ast.exprs)):
        if ast.exprs[at].kind == kind:
            return UInt32(at)
    return NO_NODE


def test_a_scalar_subquery_is_tagged_scalar() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = _filter("SELECT (SELECT x FROM u) FROM t", g, rules)
    var node = _found(ast, 15)
    assert_equal(_shape_of(ast, node), SHAPE_SCALAR)
    var tagged = about(ast, node)
    assert_true(tagged.wants_one_column())
    assert_true(tagged.at_most_one_row())
    assert_equal(tagged.operand, NO_NODE)


def test_an_exists_is_tagged_exists_and_never_counts_columns() raises:
    # EXISTS throws the select list away, so two columns under it are fine and
    # so is a division by zero that never runs.
    var g = Grammar()
    var rules = Transform(g)
    var ast = _filter(
        "SELECT a FROM t WHERE EXISTS (SELECT x, y FROM u)", g, rules
    )
    var tagged = about(ast, _found(ast, 16))
    assert_equal(tagged.shape, SHAPE_EXISTS)
    assert_false(tagged.wants_one_column())
    assert_false(tagged.at_most_one_row())
    assert_false(tagged.negated)


def test_a_not_exists_in_a_where_clause_is_a_not_over_an_exists() raises:
    # The NOT binds at the level above, so the flag on the node stays off and
    # the negation is a node of its own. Both spellings mean one thing and the
    # shape is still EXISTS either way.
    var g = Grammar()
    var rules = Transform(g)
    var ast = _filter(
        "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM u)", g, rules
    )
    var tagged = about(ast, _found(ast, 16))
    assert_equal(tagged.shape, SHAPE_EXISTS)
    assert_false(tagged.negated)
    assert_equal(
        _printed(
            "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM u)", g, rules
        ),
        "SELECT a FROM t WHERE (NOT (EXISTS (SELECT 1 FROM u)))",
    )


def test_the_exists_node_carries_a_not_of_its_own_when_it_has_one() raises:
    var ast = Ast()
    var node = ast.exists(ast.query(), negated=True)
    var tagged = about(ast, node)
    assert_equal(tagged.shape, SHAPE_EXISTS)
    assert_true(tagged.negated)


def test_an_in_subquery_is_tagged_in_and_keeps_its_operand() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = _filter("SELECT a FROM t WHERE a IN (SELECT x FROM u)", g, rules)
    var tagged = about(ast, _found(ast, 17))
    assert_equal(tagged.shape, SHAPE_IN)
    assert_true(tagged.wants_one_column())
    assert_false(tagged.at_most_one_row())
    assert_true(tagged.operand != NO_NODE)
    assert_false(tagged.negated)


def test_a_not_in_subquery_keeps_the_not() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = _filter(
        "SELECT a FROM t WHERE a NOT IN (SELECT x FROM u)", g, rules
    )
    assert_true(about(ast, _found(ast, 17)).negated)


def test_a_quantified_comparison_keeps_the_operator_and_the_quantifier() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = _filter(
        "SELECT a FROM t WHERE a >= ALL (SELECT x FROM u)", g, rules
    )
    var tagged = about(ast, _found(ast, EXPR_QUANTIFIED))
    assert_equal(tagged.shape, SHAPE_QUANTIFIED)
    assert_equal(tagged.operator, ">=")
    assert_true(tagged.every)
    assert_true(tagged.wants_one_column())


def test_any_is_the_other_quantifier() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = _filter(
        "SELECT a FROM t WHERE a < ANY (SELECT x FROM u)", g, rules
    )
    var tagged = about(ast, _found(ast, EXPR_QUANTIFIED))
    assert_equal(tagged.operator, "<")
    assert_false(tagged.every)


def test_an_expression_that_holds_no_subquery_is_tagged_nothing() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = _filter("SELECT a + 1 FROM t", g, rules)
    assert_equal(about(ast, UInt32(1)).shape, SHAPE_NONE)


def test_a_quantified_comparison_reads_back_as_itself() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t WHERE a > ALL (SELECT x FROM u)", g, rules),
        "SELECT a FROM t WHERE (a > ALL (SELECT x FROM u))",
    )
    assert_equal(
        _printed("SELECT a FROM t WHERE a = ANY (SELECT x FROM u)", g, rules),
        "SELECT a FROM t WHERE (a = ANY (SELECT x FROM u))",
    )


def test_only_six_comparisons_may_carry_a_quantifier() raises:
    # DuckDB names six and takes eight, since != and == are other spellings of
    # two of them, and it says so while parsing rather than while binding.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(
        contains=(
            "ANY and ALL operators require one of =,<>,>,<,>=,<= comparisons!"
        )
    ):
        _ = _printed(
            "SELECT a FROM t WHERE a || ALL (SELECT x FROM u)", g, rules
        )
    assert_equal(
        _printed("SELECT a FROM t WHERE a != ANY (SELECT x FROM u)", g, rules),
        "SELECT a FROM t WHERE (a != ANY (SELECT x FROM u))",
    )


def test_a_quantifier_over_a_value_is_refused_by_name() raises:
    # DuckDB unnests a list on the right, which is IN written the long way.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="ANY or ALL over a value"):
        _ = _printed("SELECT a FROM t WHERE a = ANY ([1, 2])", g, rules)


def test_the_wrong_number_of_columns_is_counted_in_the_message() raises:
    with assert_raises(contains="Subquery returns 2 columns - expected 1"):
        check_columns(2)
    check_columns(1)


def test_one_column_is_reported_with_the_plural_too() raises:
    # DuckDB's wording, kept because a query that binds here and fails there is
    # worse than one that reads oddly in both.
    assert_equal(
        wrong_column_count(1, 2),
        "Binder Error: Subquery returns 1 columns - expected 2",
    )


def test_the_second_row_message_names_the_setting() raises:
    assert_true(
        "scalar_subquery_error_on_multiple_rows=false" in too_many_rows()
    )
    assert_true(too_many_rows().startswith("Invalid Input Error: "))


def test_one_message_covers_in_and_any_and_all() raises:
    assert_equal(
        cannot_compare("INTEGER", "VARCHAR"),
        (
            "Binder Error: Cannot compare values of type INTEGER and VARCHAR"
            " in IN/ANY/ALL clause - an explicit cast is required"
        ),
    )


def test_a_subquery_that_names_nothing_outside_is_not_correlated() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a")])
    var inner = scopes.open(top)
    _ = _table(scopes, inner, "u", [String("x")])
    _ = scopes.resolve(inner, "x")
    assert_false(correlated(scopes, inner))
    assert_equal(len(outer_references(scopes, inner)), 0)


def test_a_bare_name_from_the_outer_query_correlates_it() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a")])
    var inner = scopes.open(top)
    _ = _table(scopes, inner, "u", [String("x")])
    _ = scopes.resolve(inner, "a")
    assert_true(correlated(scopes, inner))
    assert_equal(len(outer_references(scopes, inner)), 1)
    assert_equal(outer_references(scopes, inner)[0].depth, 1)


def test_a_qualified_name_from_the_outer_query_correlates_it_too() raises:
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a")])
    var inner = scopes.open(top)
    _ = _table(scopes, inner, "u", [String("x")])
    _ = scopes.resolve_qualified(inner, "t", "a")
    assert_true(correlated(scopes, inner))


def test_a_reference_two_levels_down_correlates_everything_it_crosses() raises:
    # The innermost level names the top, so both subqueries around it are
    # correlated, and a decorrelation pass has to know about both.
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a")])
    var middle = scopes.open(top)
    _ = _table(scopes, middle, "u", [String("x")])
    var inner = scopes.open(middle)
    _ = _table(scopes, inner, "v", [String("y")])
    _ = scopes.resolve(inner, "a")
    assert_true(correlated(scopes, inner))
    assert_true(correlated(scopes, middle))
    assert_false(correlated(scopes, top))


def test_a_reference_that_stops_inside_the_subquery_does_not_correlate_it() raises:
    # The innermost level names the middle one, so the middle subquery is not
    # correlated even though a level under it reached outward.
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a")])
    var middle = scopes.open(top)
    _ = _table(scopes, middle, "u", [String("x")])
    var inner = scopes.open(middle)
    _ = _table(scopes, inner, "v", [String("y")])
    _ = scopes.resolve(inner, "x")
    assert_true(correlated(scopes, inner))
    assert_false(correlated(scopes, middle))


def test_a_sibling_subquery_does_not_correlate_the_other_one() raises:
    # Two subqueries under one query are not under each other, so one reaching
    # outward says nothing about the other.
    var scopes = Scopes()
    var top = scopes.open(NOT_FOUND)
    _ = _table(scopes, top, "t", [String("a")])
    var left = scopes.open(top)
    _ = _table(scopes, left, "u", [String("x")])
    var right = scopes.open(top)
    _ = _table(scopes, right, "v", [String("y")])
    _ = scopes.resolve(left, "a")
    assert_true(correlated(scopes, left))
    assert_false(correlated(scopes, right))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
