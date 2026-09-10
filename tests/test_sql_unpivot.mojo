"""`UNPIVOT`, in both of the spellings DuckDB takes for it.

This is the other half of test_sql_pivot.mojo and reads the same way: parse,
transform, print, and name the text that should come back. The two statements
are shaped alike enough that the tests are too, and the interesting part is
where they differ.

Where a pivot has four optional parts, an unpivot has one, the `INTO NAME n
VALUE v` clause that names the two columns it writes. That one clause is why
the statement spelling is the one both of them turn into: the `FROM t UNPIVOT
(...)` spelling has to write it and cannot leave it out, so it cannot carry
every unpivot and the statement one can.

Two things the standard spelling can say are refused rather than recorded, and
each has a test. `INCLUDE NULLS` has no spelling in the statement form at all.
More than one `FOR` group needs a second name column, and the node holds one.
"""

from std.testing import TestSuite, assert_equal, assert_raises

from firepanda.sql import Grammar, Transform
from firepanda.sql.ast import Ast, REF_SUBQUERY, STMT_SELECT, STMT_UNPIVOT
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


def test_an_unpivot_with_nothing_but_the_columns_to_fold() raises:
    # The `INTO` names the two columns it writes and leaving it out asks DuckDB
    # to name them, so this is the smallest unpivot there is.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("UNPIVOT t ON a, b", g, rules), "UNPIVOT t ON a, b")


def test_the_into_clause_names_both_of_the_columns_it_writes() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("UNPIVOT t ON a, b INTO NAME n VALUE v", g, rules),
        "UNPIVOT t ON a, b INTO NAME n VALUE v",
    )


def test_value_and_values_are_the_same_word_and_the_count_picks_one() raises:
    # DuckDB takes either one whichever way the query is written, so the node
    # keeps the columns and not the spelling, and the count decides on the way
    # back out.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("UNPIVOT t ON a INTO NAME n VALUES v", g, rules),
        "UNPIVOT t ON a INTO NAME n VALUE v",
    )
    assert_equal(
        _printed("UNPIVOT t ON a INTO NAME n VALUES v1, v2", g, rules),
        "UNPIVOT t ON a INTO NAME n VALUES v1, v2",
    )


def test_a_value_list_in_parentheses_prints_without_them() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("UNPIVOT t ON a INTO NAME n VALUE (v)", g, rules),
        "UNPIVOT t ON a INTO NAME n VALUE v",
    )


def test_pivot_longer_is_the_same_word_and_prints_as_unpivot() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("PIVOT_LONGER t ON a, b", g, rules), "UNPIVOT t ON a, b"
    )


def test_the_columns_it_folds_are_whole_expressions() raises:
    # The `ON` list is a target list, the same one a `SELECT` has, so a column
    # can be qualified or aliased or be anything else that goes there.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("UNPIVOT t ON t.a, t.b", g, rules), "UNPIVOT t ON t.a, t.b"
    )
    assert_equal(
        _printed("UNPIVOT t ON a AS x, b", g, rules), "UNPIVOT t ON a AS x, b"
    )


def test_an_unpivot_reads_a_subquery_under_it() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("UNPIVOT (SELECT a, b FROM u) AS t ON a, b", g, rules),
        "UNPIVOT (SELECT a, b FROM u) AS t ON a, b",
    )


def test_the_from_spelling_becomes_the_statement_spelling() raises:
    # This is the normalization the `STMT_UNPIVOT` docstring argues for, and it
    # runs the other way from what the shorter text suggests, because the
    # spelling that looks smaller is the one that cannot leave the `INTO` out.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT * FROM t UNPIVOT (v FOR n IN (a, b))", g, rules),
        "SELECT * FROM (UNPIVOT t ON a, b INTO NAME n VALUE v)",
    )


def test_the_from_spelling_keeps_its_alias_and_its_columns() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT * FROM t UNPIVOT (v FOR n IN (a, b)) AS u", g, rules),
        "SELECT * FROM (UNPIVOT t ON a, b INTO NAME n VALUE v) AS u",
    )
    assert_equal(
        _printed(
            "SELECT * FROM t UNPIVOT (v FOR n IN (a, b)) u(x, y)", g, rules
        ),
        "SELECT * FROM (UNPIVOT t ON a, b INTO NAME n VALUE v) AS u (x, y)",
    )


def test_the_rest_of_the_query_still_reads_around_it() raises:
    # An unpivot in a `FROM` turns into a subquery in the same position, so
    # everything that came after it has to still be there.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            "SELECT * FROM t UNPIVOT (v FOR n IN (a, b)) WHERE v > 1", g, rules
        ),
        "SELECT * FROM (UNPIVOT t ON a, b INTO NAME n VALUE v) WHERE (v > 1)",
    )
    assert_equal(
        _printed(
            "SELECT * FROM t UNPIVOT (v FOR n IN (a, b)) JOIN u ON u.k = t.k",
            g,
            rules,
        ),
        (
            "SELECT * FROM (UNPIVOT t ON a, b INTO NAME n VALUE v) JOIN u ON"
            " (u.k = t.k)"
        ),
    )


def test_both_spellings_build_the_same_nodes() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var one = rules.parse_statement(
        "SELECT * FROM t UNPIVOT (v FOR n IN (a, b)) AS u", g, ast
    )
    var two = rules.parse_statement(
        "SELECT * FROM (UNPIVOT t ON a, b INTO NAME n VALUE v) AS u", g, ast
    )
    assert_equal(print_stmt(ast, one, g), print_stmt(ast, two, g))


def test_an_unpivot_inside_a_from_is_a_subquery_and_says_so() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        "SELECT * FROM t UNPIVOT (v FOR n IN (a, b)) AS u", g, ast
    )
    var query = ast.stmts[Int(ast.stmts[Int(node)].a)]
    var tables = ast.items(ast.slot(query.children, 1))
    assert_equal(len(tables), 1)
    assert_equal(ast.refs[Int(tables[0])].kind, REF_SUBQUERY)
    var inside = ast.refs[Int(tables[0])].a
    assert_equal(ast.stmts[Int(inside)].kind, STMT_SELECT)
    assert_equal(ast.stmts[Int(ast.stmts[Int(inside)].a)].kind, STMT_UNPIVOT)


def test_the_two_columns_the_into_names_are_read_off_the_node() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        "UNPIVOT t ON a, b INTO NAME n VALUE v", g, ast
    )
    var unpivot = ast.stmts[Int(ast.stmts[Int(node)].a)]
    assert_equal(unpivot.kind, STMT_UNPIVOT)
    assert_equal(ast.length(unpivot.children), 2)
    assert_equal(ast.text(unpivot.payload), "n")
    assert_equal(ast.length(unpivot.b), 1)
    assert_equal(ast.text(ast.at(unpivot.b, 0)), "v")


def test_no_into_leaves_both_of_those_empty_together() raises:
    # They arrive as one clause, so one of them being empty and both of them
    # being empty are the same state and either one answers the question.
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement("UNPIVOT t ON a, b", g, ast)
    var unpivot = ast.stmts[Int(ast.stmts[Int(node)].a)]
    assert_equal(unpivot.payload, 0)
    assert_equal(ast.length(unpivot.b), 0)


def test_exclude_nulls_is_the_default_and_include_nulls_refuses() raises:
    # Writing the default changes nothing, so it is read and dropped the way
    # `EXCLUDE NO OTHERS` is on a window frame. The other one does change
    # something and the statement spelling has nowhere to write it.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            "SELECT * FROM t UNPIVOT EXCLUDE NULLS (v FOR n IN (a, b))",
            g,
            rules,
        ),
        "SELECT * FROM (UNPIVOT t ON a, b INTO NAME n VALUE v)",
    )
    with assert_raises(contains="INCLUDE NULLS on an UNPIVOT"):
        _ = _printed(
            "SELECT * FROM t UNPIVOT INCLUDE NULLS (v FOR n IN (a, b))",
            g,
            rules,
        )


def test_a_second_for_group_needs_a_second_node_and_refuses() raises:
    # One node holds one name column and one set of value columns. A second
    # group and a name column written as a list both ask for more than that.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="more than one FOR group"):
        _ = _printed(
            "SELECT * FROM t UNPIVOT (v FOR n IN (a, b) m IN (c, d))", g, rules
        )
    with assert_raises(contains="more than one FOR group"):
        _ = _printed(
            "SELECT * FROM t UNPIVOT (v FOR (n1, n2) IN (a, b))", g, rules
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
