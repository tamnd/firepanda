"""`PIVOT`, in both of the spellings DuckDB takes for it.

The shape of these tests is the shape of test_sql_select.mojo: parse, transform,
print, and name the text that should come back. A pivot has four parts and any
of them can be left out, so most of what is here is one test per combination,
and reading the text back is what catches a part that was quietly dropped.

Three things come back written differently from how they went in and each has a
test of its own saying so. `PIVOT_WIDER` prints as `PIVOT`, because DuckDB takes
the two as one word and the AST records the pivot rather than the spelling.
`GROUP BY (g, h)` prints without the parentheses, for the same reason. And the
whole of `FROM t PIVOT (...)` prints as `FROM (PIVOT t ON ...)`, which is the
one worth reading the note in `STMT_PIVOT` about, because it is the direction
that can carry every query and the other one is not.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.sql import Grammar, Transform
from firepanda.sql.ast import (
    Ast,
    NO_NODE,
    REF_SUBQUERY,
    STMT_PIVOT,
    STMT_PIVOT_ON,
    STMT_SELECT,
)
from firepanda.sql.printer import print_stmt


def _printed(
    sql: StringSlice, grammar: Grammar, rules: Transform
) raises -> String:
    """Parses one statement, transforms it, prints it and prints it again.

    Args:
        sql: The whole statement.
        grammar: A loaded grammar.
        rules: The jump table built from it.

    Returns:
        The printed text.

    Raises:
        Error: If it does not parse, or holds something with no case, or a
            second pass changes it.
    """
    var ast = Ast()
    var once = print_stmt(
        ast, rules.parse_statement(sql, grammar, ast), grammar
    )
    var again = Ast()
    var twice = print_stmt(
        again, rules.parse_statement(once, grammar, again), grammar
    )
    if once != twice:
        raise Error(
            String(
                "a second pass changed the statement:\n  first  ",
                once,
                "\n  second ",
                twice,
            )
        )
    return once^


def test_a_pivot_with_nothing_but_a_table() raises:
    # Every one of the three lists is optional, so this is the smallest pivot
    # the grammar takes, and it asks DuckDB to work the rest out.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("PIVOT t", g, rules), "PIVOT t")


def test_the_three_lists_a_pivot_is_made_of() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("PIVOT t ON a", g, rules), "PIVOT t ON a")
    assert_equal(
        _printed("PIVOT t USING sum(x)", g, rules), "PIVOT t USING sum(x)"
    )
    assert_equal(_printed("PIVOT t GROUP BY g", g, rules), "PIVOT t GROUP BY g")


def test_all_three_at_once_keep_their_order() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            "PIVOT t ON a, b USING sum(x) AS s, count(*) GROUP BY g, h",
            g,
            rules,
        ),
        "PIVOT t ON a, b USING sum(x) AS s, count(*) GROUP BY g, h",
    )


def test_the_four_things_a_pivot_column_can_say_about_its_values() raises:
    # `ON a` on its own leaves the values to be read off the column, and the
    # other three name them, so the node has one field per way of naming them
    # and at most one of the three is ever set.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("PIVOT t ON a", g, rules), "PIVOT t ON a")
    assert_equal(
        _printed("PIVOT t ON a IN (1, 2)", g, rules), "PIVOT t ON a IN (1, 2)"
    )
    assert_equal(
        _printed("PIVOT t ON a IN an_enum", g, rules),
        "PIVOT t ON a IN an_enum",
    )
    assert_equal(
        _printed("PIVOT t ON a IN (SELECT x FROM u)", g, rules),
        "PIVOT t ON a IN (SELECT x FROM u)",
    )


def test_a_pivot_column_is_a_whole_expression_and_not_just_a_name() raises:
    # A column with no `IN` after it is any expression at all. One with an
    # `IN` is narrower, because the grammar reads the header at the level that
    # stops before an operator, which is what keeps the `IN` from being read as
    # part of the header instead of as the start of the value list.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("PIVOT t ON a || b", g, rules), "PIVOT t ON (a || b)")
    assert_equal(
        _printed("PIVOT t ON lower(a) IN ('x')", g, rules),
        "PIVOT t ON lower(a) IN ('x')",
    )
    assert_equal(
        _printed("PIVOT t ON t.a IN ('x')", g, rules),
        "PIVOT t ON t.a IN ('x')",
    )


def test_a_pivot_reads_a_subquery_and_a_join_under_it() raises:
    # The table under a pivot is a whole `TableRef`, so everything a `FROM`
    # takes goes there too.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("PIVOT (SELECT a, x FROM u) AS t ON a", g, rules),
        "PIVOT (SELECT a, x FROM u) AS t ON a",
    )
    assert_equal(
        _printed("PIVOT t JOIN u ON t.k = u.k ON a", g, rules),
        "PIVOT t JOIN u ON (t.k = u.k) ON a",
    )


def test_pivot_wider_is_the_same_word_and_prints_as_pivot() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("PIVOT_WIDER t ON a USING sum(x)", g, rules),
        "PIVOT t ON a USING sum(x)",
    )


def test_a_group_by_in_parentheses_prints_without_them() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("PIVOT t ON a GROUP BY (g, h)", g, rules),
        "PIVOT t ON a GROUP BY g, h",
    )


def test_the_from_spelling_becomes_the_statement_spelling() raises:
    # This is the normalization the `STMT_PIVOT` docstring argues for. The
    # standard spelling wants an `IN` on every column and the statement one
    # does not, so only one of the two can carry every pivot, and that is the
    # one both of them turn into.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            "SELECT * FROM t PIVOT (sum(x) FOR a IN (1, 2) GROUP BY g) AS p",
            g,
            rules,
        ),
        "SELECT * FROM (PIVOT t ON a IN (1, 2) USING sum(x) GROUP BY g) AS p",
    )


def test_the_from_spelling_keeps_its_alias_and_its_columns() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            "SELECT * FROM t PIVOT (sum(x) FOR a IN (1, 2)) p(u, v)", g, rules
        ),
        "SELECT * FROM (PIVOT t ON a IN (1, 2) USING sum(x)) AS p (u, v)",
    )


def test_the_from_spelling_writes_more_than_one_column_with_no_comma() raises:
    # One `FOR` covers all of them and they run on with nothing in between,
    # which is the other place the two spellings differ. The statement one
    # separates its columns with commas, so that is what comes back out.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            "SELECT * FROM t PIVOT (sum(x) FOR a IN (1, 2) b IN (3)) AS p",
            g,
            rules,
        ),
        "SELECT * FROM (PIVOT t ON a IN (1, 2), b IN (3) USING sum(x)) AS p",
    )


def test_both_spellings_build_the_same_nodes() raises:
    # Printing the same text is not quite the same claim as building the same
    # tree, and the tree is what the binder will read, so this walks it.
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var one = rules.parse_statement(
        "SELECT * FROM t PIVOT (sum(x) FOR a IN (1, 2)) AS p", g, ast
    )
    var two = rules.parse_statement(
        "SELECT * FROM (PIVOT t ON a IN (1, 2) USING sum(x)) AS p", g, ast
    )
    assert_equal(print_stmt(ast, one, g), print_stmt(ast, two, g))


def test_a_pivot_inside_a_from_is_a_subquery_and_says_so() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        "SELECT * FROM t PIVOT (sum(x) FOR a IN (1, 2)) AS p", g, ast
    )
    var query = ast.stmts[Int(ast.stmts[Int(node)].a)]
    var tables = ast.items(ast.slot(query.children, 1))
    assert_equal(len(tables), 1)
    assert_equal(ast.refs[Int(tables[0])].kind, REF_SUBQUERY)
    var inside = ast.refs[Int(tables[0])].a
    assert_equal(ast.stmts[Int(inside)].kind, STMT_SELECT)
    assert_equal(ast.stmts[Int(ast.stmts[Int(inside)].a)].kind, STMT_PIVOT)


def test_a_pivot_column_holds_one_kind_of_value_at_a_time() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement("PIVOT t ON a IN an_enum", g, ast)
    var pivot = ast.stmts[Int(ast.stmts[Int(node)].a)]
    assert_equal(pivot.kind, STMT_PIVOT)
    assert_equal(ast.length(pivot.b), 1)
    var column = ast.stmts[Int(ast.at(pivot.b, 0))]
    assert_equal(column.kind, STMT_PIVOT_ON)
    assert_equal(ast.text(column.payload), "an_enum")
    assert_equal(column.b, NO_NODE)
    assert_equal(ast.length(column.children), 0)


def test_unpivot_still_says_it_is_not_read_yet() raises:
    # The two arrived one at a time, and the message names which one is which
    # so a user does not read the refusal as covering both.
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    with assert_raises(contains="UNPIVOT"):
        _ = rules.parse_statement("UNPIVOT t ON a", g, ast)
    with assert_raises(contains="UNPIVOT"):
        _ = rules.parse_statement(
            "SELECT * FROM t UNPIVOT (v FOR n IN (a, b))", g, ast
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
