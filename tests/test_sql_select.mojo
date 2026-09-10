"""The transformer, from a parse tree to the statement AST.

This is the other half of test_sql_transform.mojo. That file takes an
expression and this one takes a whole `SELECT`, and both work the same way: the
query is parsed, transformed and printed, and the test names the text it
expects. Reading the text back is what makes these tests worth writing, because
a clause that is silently dropped shows up as text that is missing rather than
as a field somewhere in an arena that nobody looks at.

Most of the queries in here come back exactly as they went in. The ones that do
not are the printer's normalizations, and each of those has a test of its own
saying which spelling wins and why, because a normalization nobody wrote down
is indistinguishable from a bug. `FROM t SELECT a` is the clearest one: DuckDB
takes the clauses in either order and means the same query by both, so the
printer picks one.

Every test here also checks that printing twice gives the same text. That is
the property the corpus round trip leans on, and the parentheses are why it
needs checking: the printer puts them back around operands, and a transformer
that read its own output as one paren deeper each time would still pass a single
pass test. See docs/specs/sql/05-ast-and-binder.md.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.sql import Grammar, Transform
from firepanda.sql.ast import (
    Ast,
    CLAUSE_FROM,
    CLAUSE_GROUP,
    GROUP_TUPLE,
    REF_JOIN,
)
from firepanda.sql.printer import print_stmt


def _printed(
    sql: StringSlice, grammar: Grammar, rules: Transform
) raises -> String:
    """Parses one statement, transforms it, prints it and prints it again.

    The second pass is here rather than in a test of its own because every
    query in this file wants it, and a helper that only sometimes checks the
    property is a helper somebody will forget to use.

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
    var twice: String
    try:
        twice = print_stmt(
            again, rules.parse_statement(once, grammar, again), grammar
        )
    except e:
        raise Error(
            String(
                (
                    "the printer wrote something the transformer cannot read"
                    " back: "
                ),
                once,
                "\n",
                e,
            )
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


def test_the_smallest_statement_there_is() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("SELECT 1", g, rules), "SELECT 1")


def test_a_select_list_keeps_its_order_and_its_aliases() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT b, a AS x, 1 FROM t", g, rules),
        "SELECT b, a AS x, 1 FROM t",
    )


def test_a_star_survives_the_statement_side() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("SELECT * FROM t", g, rules), "SELECT * FROM t")
    assert_equal(_printed("SELECT t.* FROM t", g, rules), "SELECT t.* FROM t")


def test_distinct_and_distinct_on() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT DISTINCT a FROM t", g, rules),
        "SELECT DISTINCT a FROM t",
    )
    assert_equal(
        _printed("SELECT DISTINCT ON (a, b) c FROM t", g, rules),
        "SELECT DISTINCT ON (a, b) c FROM t",
    )


def test_all_is_the_default_and_is_still_written_back() raises:
    # `ALL` means what leaving it out means, so nothing downstream reads it.
    # It is kept because the printer's job is to give back the query that was
    # written, and a query that says `ALL` said `ALL`.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT ALL a FROM t", g, rules), "SELECT ALL a FROM t"
    )


def test_from_first_is_the_same_query_written_the_other_way_round() raises:
    # DuckDB takes the two clauses in either order. They mean the same query,
    # the AST has one shape for it, and the printer writes the `SELECT` first.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("FROM t SELECT a", g, rules), "SELECT a FROM t")


def test_from_on_its_own_stays_a_from_on_its_own() raises:
    # A query with no `SELECT` list has no projection to write, so there is
    # nothing to move in front and the printer leaves it where it is. It is not
    # rewritten to `SELECT *` because a missing list and a star are different
    # things once a binder starts resolving names.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("FROM t", g, rules), "FROM t")


def test_a_table_keeps_its_qualification_and_its_aliases() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM main.t AS x (p, q)", g, rules),
        "SELECT a FROM main.t AS x (p, q)",
    )


def test_a_comma_in_a_from_is_a_list_and_not_a_join() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("SELECT a FROM t, u", g, rules), "SELECT a FROM t, u")


def test_the_clauses_that_hold_one_expression() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t WHERE a > 1", g, rules),
        "SELECT a FROM t WHERE (a > 1)",
    )
    assert_equal(
        _printed("SELECT a FROM t GROUP BY a HAVING count(*) > 1", g, rules),
        "SELECT a FROM t GROUP BY a HAVING (count(*) > 1)",
    )
    assert_equal(
        _printed("SELECT a FROM t QUALIFY row_number() > 1", g, rules),
        "SELECT a FROM t QUALIFY (row_number() > 1)",
    )


def test_the_group_by_spellings() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t GROUP BY a, b", g, rules),
        "SELECT a FROM t GROUP BY a, b",
    )
    assert_equal(
        _printed("SELECT a FROM t GROUP BY ALL", g, rules),
        "SELECT a FROM t GROUP BY ALL",
    )
    assert_equal(
        _printed("SELECT a FROM t GROUP BY CUBE (a, b)", g, rules),
        "SELECT a FROM t GROUP BY CUBE (a, b)",
    )
    assert_equal(
        _printed("SELECT a FROM t GROUP BY ROLLUP (a, b)", g, rules),
        "SELECT a FROM t GROUP BY ROLLUP (a, b)",
    )


def test_a_grouping_set_of_several_columns_is_not_an_expression() raises:
    # The grammar has no rule for a tuple in a `GROUPING SETS`, so `(a, b)`
    # arrives as the expression that a row is written as. Reading it as one
    # would group by a row rather than by two columns, so the entry is a tag of
    # its own and the transformer looks for it.
    var g = Grammar()
    var rules = Transform(g)
    var sql = "SELECT a FROM t GROUP BY GROUPING SETS ((a, b), c, ())"
    assert_equal(_printed(sql, g, rules), sql)

    var ast = Ast()
    var node = rules.parse_statement(sql, g, ast)
    var clauses = ast.stmts[Int(ast.stmts[Int(node)].a)].children
    var sets = ast.stmts[Int(ast.at(ast.at(clauses, CLAUSE_GROUP), 0))]
    assert_equal(ast.length(sets.children), 3)
    var first = ast.stmts[Int(ast.at(sets.children, 0))]
    assert_equal(Int(first.b), Int(GROUP_TUPLE))
    assert_equal(ast.length(first.children), 2)


def test_a_row_is_still_refused_where_it_is_really_a_row() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="does not support"):
        _ = _printed("SELECT (a, b) FROM t", g, rules)


def test_order_by_keeps_its_direction_and_its_null_placement() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t ORDER BY a DESC NULLS LAST, b ASC", g, rules),
        "SELECT a FROM t ORDER BY a DESC NULLS LAST, b ASC",
    )
    assert_equal(
        _printed("SELECT a FROM t ORDER BY a NULLS FIRST", g, rules),
        "SELECT a FROM t ORDER BY a NULLS FIRST",
    )


def test_order_by_all() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t ORDER BY ALL DESC", g, rules),
        "SELECT a FROM t ORDER BY ALL DESC",
    )


def test_limit_and_offset_in_either_order() raises:
    # SQL takes them either way round and means the same thing, so the AST has
    # one slot for each and the printer writes the limit first.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t LIMIT 10 OFFSET 5", g, rules),
        "SELECT a FROM t LIMIT 10 OFFSET 5",
    )
    assert_equal(
        _printed("SELECT a FROM t OFFSET 5 LIMIT 10", g, rules),
        "SELECT a FROM t LIMIT 10 OFFSET 5",
    )


def test_the_other_limit_spellings() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t LIMIT ALL", g, rules),
        "SELECT a FROM t LIMIT ALL",
    )
    assert_equal(
        _printed("SELECT a FROM t LIMIT 10%", g, rules),
        "SELECT a FROM t LIMIT 10%",
    )


def test_fetch_first_is_a_limit_written_the_long_way() raises:
    # `FETCH FIRST n ROWS ONLY` is the standard spelling of `LIMIT n` and means
    # exactly that, so it lands on the same slot and prints back as the short
    # one. The AST holds what a query means, and a formatter is a different
    # program.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t FETCH FIRST 10 ROWS ONLY", g, rules),
        "SELECT a FROM t LIMIT 10",
    )
    assert_equal(
        _printed("SELECT a FROM t OFFSET 5 FETCH NEXT 10 ROWS ONLY", g, rules),
        "SELECT a FROM t LIMIT 10 OFFSET 5",
    )


def test_the_joins_that_take_a_condition() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t JOIN u ON t.k = u.k", g, rules),
        "SELECT a FROM t JOIN u ON (t.k = u.k)",
    )
    assert_equal(
        _printed("SELECT a FROM t LEFT OUTER JOIN u ON t.k = u.k", g, rules),
        "SELECT a FROM t LEFT OUTER JOIN u ON (t.k = u.k)",
    )
    assert_equal(
        _printed("SELECT a FROM t JOIN u USING (k, j)", g, rules),
        "SELECT a FROM t JOIN u USING (k, j)",
    )


def test_the_joins_that_take_no_condition() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t CROSS JOIN u", g, rules),
        "SELECT a FROM t CROSS JOIN u",
    )
    assert_equal(
        _printed("SELECT a FROM t NATURAL LEFT JOIN u", g, rules),
        "SELECT a FROM t NATURAL LEFT JOIN u",
    )
    assert_equal(
        _printed("SELECT a FROM t POSITIONAL JOIN u", g, rules),
        "SELECT a FROM t POSITIONAL JOIN u",
    )


def test_a_join_is_kept_as_the_words_it_was_written_with() raises:
    # There are no join flags anywhere. `LEFT OUTER JOIN` is four words in the
    # query and one string in the node, which is why `NATURAL LEFT JOIN` and
    # `ASOF JOIN` needed no new cases when they were tried.
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        "SELECT a FROM t LEFT OUTER JOIN u ON t.k = u.k", g, ast
    )
    var clauses = ast.stmts[Int(ast.stmts[Int(node)].a)].children
    var join = ast.refs[Int(ast.at(ast.at(clauses, CLAUSE_FROM), 0))]
    assert_equal(Int(join.kind), Int(REF_JOIN))
    assert_equal(String(ast.text(join.payload)), "LEFT OUTER JOIN")


def test_joins_chain_to_the_left() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t JOIN u USING (k) JOIN v USING (k)", g, rules),
        "SELECT a FROM t JOIN u USING (k) JOIN v USING (k)",
    )


def test_parentheses_in_a_from_are_kept_because_they_change_the_shape() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM (t JOIN u USING (k)) AS w", g, rules),
        "SELECT a FROM (t JOIN u USING (k)) AS w",
    )


def test_a_subquery_in_a_from() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM (SELECT 1) AS s", g, rules),
        "SELECT a FROM (SELECT 1) AS s",
    )
    assert_equal(
        _printed("SELECT a FROM LATERAL (SELECT 1) AS s (p)", g, rules),
        "SELECT a FROM LATERAL (SELECT 1) AS s (p)",
    )


def test_a_table_function_in_a_from() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM read_csv('x.csv') AS f", g, rules),
        "SELECT a FROM read_csv('x.csv') AS f",
    )


def test_values_as_a_table() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM (VALUES (1), (2)) AS v (p)", g, rules),
        "SELECT a FROM (VALUES (1), (2)) AS v (p)",
    )


def test_the_set_operations() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT 1 UNION SELECT 2", g, rules), "SELECT 1 UNION SELECT 2"
    )
    assert_equal(
        _printed("SELECT 1 EXCEPT SELECT 2", g, rules),
        "SELECT 1 EXCEPT SELECT 2",
    )
    assert_equal(
        _printed("SELECT 1 INTERSECT ALL SELECT 2", g, rules),
        "SELECT 1 INTERSECT ALL SELECT 2",
    )


def test_a_set_operation_is_kept_as_the_words_it_was_written_with() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT 1 UNION ALL BY NAME SELECT 2", g, rules),
        "SELECT 1 UNION ALL BY NAME SELECT 2",
    )


def test_set_operations_chain_to_the_left() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT 1 UNION SELECT 2 EXCEPT SELECT 3", g, rules),
        "SELECT 1 UNION SELECT 2 EXCEPT SELECT 3",
    )


def test_parentheses_around_an_operand_are_kept() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("(SELECT 1) UNION (SELECT 2)", g, rules),
        "(SELECT 1) UNION (SELECT 2)",
    )


def test_a_with_and_its_column_aliases() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("WITH c AS (SELECT 1) SELECT * FROM c", g, rules),
        "WITH c AS (SELECT 1) SELECT * FROM c",
    )
    assert_equal(
        _printed("WITH c (p) AS (SELECT 1) SELECT * FROM c", g, rules),
        "WITH c (p) AS (SELECT 1) SELECT * FROM c",
    )


def test_a_recursive_with() raises:
    var g = Grammar()
    var rules = Transform(g)
    var sql = (
        "WITH RECURSIVE c (p) AS (SELECT 1 UNION ALL SELECT (p + 1) FROM c)"
        " SELECT * FROM c"
    )
    assert_equal(_printed(sql, g, rules), sql)


def test_both_materialized_spellings() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("WITH c AS MATERIALIZED (SELECT 1) SELECT * FROM c", g, rules),
        "WITH c AS MATERIALIZED (SELECT 1) SELECT * FROM c",
    )
    assert_equal(
        _printed(
            "WITH c AS NOT MATERIALIZED (SELECT 1) SELECT * FROM c", g, rules
        ),
        "WITH c AS NOT MATERIALIZED (SELECT 1) SELECT * FROM c",
    )


def test_two_entries_in_one_with() raises:
    var g = Grammar()
    var rules = Transform(g)
    var sql = "WITH c AS (SELECT 1), d AS (SELECT 2) SELECT * FROM c, d"
    assert_equal(_printed(sql, g, rules), sql)


def test_values_as_a_statement() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("VALUES (1, 2), (3, 4)", g, rules), "VALUES (1, 2), (3, 4)"
    )


def test_table_as_a_statement() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("TABLE main.t", g, rules), "TABLE main.t")


def test_the_three_expressions_that_hold_a_statement() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT (SELECT 1) AS a", g, rules), "SELECT (SELECT 1) AS a"
    )
    assert_equal(
        _printed("SELECT a FROM t WHERE EXISTS (SELECT 1)", g, rules),
        "SELECT a FROM t WHERE (EXISTS (SELECT 1))",
    )
    assert_equal(
        _printed("SELECT a FROM t WHERE a IN (SELECT 1)", g, rules),
        "SELECT a FROM t WHERE (a IN (SELECT 1))",
    )


def test_the_negated_forms_of_those() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t WHERE NOT EXISTS (SELECT 1)", g, rules),
        "SELECT a FROM t WHERE (NOT (EXISTS (SELECT 1)))",
    )
    assert_equal(
        _printed("SELECT a FROM t WHERE a NOT IN (SELECT 1)", g, rules),
        "SELECT a FROM t WHERE (a NOT IN (SELECT 1))",
    )


def test_a_query_with_every_clause_at_once() raises:
    var g = Grammar()
    var rules = Transform(g)
    var sql = (
        "WITH c AS (SELECT 1) SELECT DISTINCT a, count(*) AS n FROM c JOIN u"
        " ON (c.k = u.k) WHERE (a > 1) GROUP BY a HAVING (count(*) > 1)"
        " QUALIFY (n > 0) ORDER BY a DESC NULLS LAST LIMIT 10 OFFSET 5"
    )
    assert_equal(_printed(sql, g, rules), sql)


def test_a_subquery_nested_three_deep() raises:
    var g = Grammar()
    var rules = Transform(g)
    var sql = "SELECT a FROM (SELECT b FROM (SELECT c FROM t) AS s1) AS s2"
    assert_equal(_printed(sql, g, rules), sql)


def test_a_long_union_chain_is_built_once() raises:
    # The same argument as the long expression chain in test_sql_transform.mojo.
    # The walk holds one result per parse node, so a chain of 200 operands is
    # 200 statements and not 200 factorial visits.
    var g = Grammar()
    var rules = Transform(g)
    var sql = String("SELECT 0")
    for i in range(1, 200):
        sql += String(" UNION ALL SELECT ", i)
    assert_equal(_printed(sql, g, rules), sql)


def test_a_pivot_after_a_table_is_read_and_an_unpivot_is_not() raises:
    # What a pivot turns into is tested in `test_sql_pivot.mojo`. The one thing
    # this file cares about is that the modifier position after a table reaches
    # it, and that the modifier that is still refused says which one it is.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t PIVOT (sum(a) FOR b IN (1))", g, rules),
        "SELECT a FROM (PIVOT t ON b IN (1) USING sum(a))",
    )
    with assert_raises(contains="UNPIVOT on a table"):
        _ = _printed("SELECT a FROM t UNPIVOT (v FOR n IN (a, b))", g, rules)


def test_a_sample_refuses_and_names_itself() raises:
    var g = Grammar()
    var rules = Transform(g)
    # A sample written straight after the table belongs to the table, because
    # that is the first rule with a slot for one. It takes a clause in between
    # to reach the one on the statement.
    with assert_raises(contains="a sample on a table"):
        _ = _printed("SELECT a FROM t TABLESAMPLE 10%", g, rules)
    with assert_raises(contains="a sample on a SELECT"):
        _ = _printed("SELECT a FROM t WHERE x USING SAMPLE 10%", g, rules)


def test_a_window_clause_names_a_window_for_the_query_to_use() raises:
    # What can go inside the parentheses is tested in `test_sql_window.mojo`.
    # The one thing this file cares about is that the clause comes out in the
    # place `SimpleSelect` wants it, so the printed text parses again.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t WINDOW w AS ()", g, rules),
        "SELECT a FROM t WINDOW w AS ()",
    )


def test_the_colon_alias_on_a_select_item_is_the_same_as_as() raises:
    # DuckDB takes `x: a` and `a AS x` as two spellings of one thing, so the
    # AST has one node for both and the printer writes the `AS` one.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT x: a FROM t", g, rules), "SELECT a AS x FROM t"
    )


def test_the_colon_alias_on_a_table_refuses() raises:
    # The same spelling in front of a table is a different rule, and the AST
    # keeps a table's alias after the name rather than before it.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="the name: table spelling"):
        _ = _printed("SELECT a FROM x: t", g, rules)


def test_a_refusal_says_where_it_was() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="TABLESAMPLE"):
        _ = _printed("SELECT a FROM t TABLESAMPLE 10%", g, rules)


def test_a_grouping_set_of_one_column_is_the_column() raises:
    # `GROUPING SETS ((a, ))` is a set of one column, and so is
    # `GROUPING SETS (a)`. The trailing comma makes the grammar hand back a row
    # where the plain spelling hands back an expression, and if the row stayed
    # it would print as `(a)`, which reads back as the plain spelling. The two
    # texts would differ with nothing between them having changed meaning.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t GROUP BY GROUPING SETS ((a, ))", g, rules),
        "SELECT a FROM t GROUP BY GROUPING SETS (a)",
    )


def test_a_trailing_semicolon_is_part_of_the_statement() raises:
    # A query copied out of a file or a shell has one on the end, and a reader
    # told that is a syntax error will not believe it.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("SELECT 1;", g, rules), "SELECT 1")
    assert_equal(_printed("SELECT 1 ;;;", g, rules), "SELECT 1")


def test_nothing_at_all_is_a_syntax_error() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="syntax error"):
        _ = _printed("", g, rules)
    with assert_raises(contains="syntax error"):
        _ = _printed(";", g, rules)


def test_a_statement_firepanda_will_run_later_refuses_by_name() raises:
    # Tier two. The word `yet` is the whole point of the entry being separate:
    # it tells a reader to wait rather than to rewrite the query.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="the CREATE statement yet"):
        _ = _printed("CREATE TABLE t (a INT)", g, rules)
    with assert_raises(contains="the INSERT statement yet"):
        _ = _printed("INSERT INTO t VALUES (1)", g, rules)


def test_a_statement_firepanda_will_not_run_refuses_by_name() raises:
    # Tier three, and the message has no `yet` in it.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="the ATTACH statement."):
        _ = _printed("ATTACH 'x.db'", g, rules)
    with assert_raises(contains="the UPDATE statement."):
        _ = _printed("UPDATE t SET a = 1", g, rules)


def test_a_statement_refusal_says_where_it_was_and_where_to_read() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="LINE 1: DROP TABLE t"):
        _ = _printed("DROP TABLE t", g, rules)
    with assert_raises(contains="issues/"):
        _ = _printed("DROP TABLE t", g, rules)


def test_a_statement_that_is_not_sql_at_all_is_still_a_syntax_error() raises:
    # The tier tables must not turn a typo into a refusal. `SELCT` is not a
    # statement keyword, so nothing matches and the matcher says so.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="syntax error"):
        _ = _printed("SELCT 1", g, rules)


def test_an_expression_form_refuses_by_name_rather_than_by_rule_number() raises:
    # A rule number is a fact about firepanda's build of the grammar and it
    # means nothing to the person who wrote the query. Every one of these used
    # to print one.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="a row value"):
        _ = _printed("SELECT (1, 2)", g, rules)
    with assert_raises(contains="a row value"):
        _ = _printed("SELECT ROW(1, 2)", g, rules)
    with assert_raises(contains="an INTERVAL literal"):
        _ = _printed("SELECT INTERVAL '1 day'", g, rules)
    with assert_raises(contains="a typed literal"):
        _ = _printed("SELECT DATE '2020-01-01'", g, rules)
    with assert_raises(contains="a lambda"):
        _ = _printed("SELECT list_apply(l, lambda x: x + 1)", g, rules)
    with assert_raises(contains="a list comprehension"):
        _ = _printed("SELECT [x FOR x IN l]", g, rules)
    with assert_raises(contains="an argument passed by name"):
        _ = _printed("SELECT f(a := 1)", g, rules)
    with assert_raises(contains="COLUMNS"):
        _ = _printed("SELECT COLUMNS('a')", g, rules)
    with assert_raises(contains="a MAP literal"):
        _ = _printed("SELECT MAP {'a': 1}", g, rules)
    with assert_raises(contains="GROUPING"):
        _ = _printed("SELECT GROUPING(a) FROM t GROUP BY a", g, rules)
    with assert_raises(contains="a column written as #1"):
        _ = _printed("SELECT #1 FROM t", g, rules)


def test_a_call_spelled_with_keywords_refuses_under_its_own_name() raises:
    # One table entry for the lot of them, and the message still says which one
    # it was, which is what the `{}` slot is for.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="EXTRACT yet"):
        _ = _printed("SELECT EXTRACT(YEAR FROM x)", g, rules)
    with assert_raises(contains="SUBSTRING yet"):
        _ = _printed("SELECT SUBSTRING(a FROM 1 FOR 2)", g, rules)
    with assert_raises(contains="TRIM yet"):
        _ = _printed("SELECT TRIM(BOTH ' ' FROM a)", g, rules)
    with assert_raises(contains="POSITION yet"):
        _ = _printed("SELECT POSITION(a IN b)", g, rules)
    with assert_raises(contains="OVERLAY yet"):
        _ = _printed("SELECT OVERLAY(a PLACING b FROM 1)", g, rules)


def test_a_statement_inside_a_with_refuses_under_its_own_name() raises:
    # `WITH x AS (INSERT ...)` used to blame `CTEDMLBody`, which is a rule the
    # user did not write and cannot look up. The statement inside says its own
    # name and the caret lands on it.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="the INSERT statement yet"):
        _ = _printed(
            "WITH x AS (INSERT INTO t VALUES (1)) SELECT * FROM x", g, rules
        )
    with assert_raises(contains="the DELETE statement."):
        _ = _printed("WITH x AS (DELETE FROM t) SELECT * FROM x", g, rules)


def test_a_query_that_is_not_a_select_refuses_by_name() raises:
    # These produce rows, so they hang off the select rule rather than off the
    # statement rule, and the tier loop never reached them. `PIVOT` was here
    # too and is read now.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="the DESCRIBE statement yet"):
        _ = _printed("DESCRIBE t", g, rules)
    with assert_raises(contains="the UNPIVOT statement yet"):
        _ = _printed("UNPIVOT t ON a", g, rules)


def test_a_parenthesised_expression_is_still_just_the_expression() raises:
    # `ParenthesisExpression` is the rule that refuses a row value, and it is
    # also the rule around `(1 + 2)`. The one that is a plain expression has to
    # keep working.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT (1 + 2) * 3", g, rules), "SELECT ((1 + 2) * 3)"
    )
    assert_equal(
        _printed(
            "SELECT a FROM t GROUP BY GROUPING SETS (a, (b, c))", g, rules
        ),
        "SELECT a FROM t GROUP BY GROUPING SETS (a, (b, c))",
    )


def test_the_table_has_an_entry_for_the_statement_rule() raises:
    var g = Grammar()
    var rules = Transform(g)
    assert_true(rules.statement_rule >= 0)
    assert_true(rules.parens_rule >= 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
