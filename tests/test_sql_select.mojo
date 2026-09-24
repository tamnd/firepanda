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


def test_a_row_and_a_grouping_tuple_are_told_apart_by_where_they_are() raises:
    # The two are spelled the same way and mean different things. The row used
    # to refuse here, which is what kept them apart, and now both build, so
    # each one has to come back where it was written and not as the other.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT (a, b) FROM t", g, rules), "SELECT (a, b) FROM t"
    )
    assert_equal(
        _printed("SELECT a FROM t GROUP BY GROUPING SETS ((a, b))", g, rules),
        "SELECT a FROM t GROUP BY GROUPING SETS ((a, b))",
    )


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


def test_ordinality_reaches_the_printer_and_keeps_its_place() raises:
    # The two words go after the parentheses and before the alias, which is
    # where the query writes them and the only place the grammar takes them.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT * FROM range(3) WITH ORDINALITY", g, rules),
        "SELECT * FROM range(3) WITH ORDINALITY",
    )
    assert_equal(
        _printed(
            "SELECT * FROM range(3) WITH ORDINALITY AS t (a, b)", g, rules
        ),
        "SELECT * FROM range(3) WITH ORDINALITY AS t (a, b)",
    )
    # LATERAL and the two words sit in the same field and are not the same
    # word, so a call that was written both ways prints both back.
    assert_equal(
        _printed(
            "SELECT * FROM t, LATERAL range(t.a) WITH ORDINALITY", g, rules
        ),
        "SELECT * FROM t, LATERAL range(t.a) WITH ORDINALITY",
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


def test_using_key_reaches_the_printer() raises:
    # It was turned down at the parse until the transformer learned to build
    # one. The list takes whatever a SELECT list takes, so an entry may be a
    # call and may carry an alias, and it sits between the column aliases and
    # the AS.
    var g = Grammar()
    var rules = Transform(g)
    var sql = "WITH c USING KEY (k) AS (SELECT 1 AS k) SELECT * FROM c"
    assert_equal(_printed(sql, g, rules), sql)
    sql = (
        "WITH c (p, q) USING KEY (k, min(v)) AS NOT MATERIALIZED (SELECT 1, 2)"
        " SELECT * FROM c"
    )
    assert_equal(_printed(sql, g, rules), sql)
    sql = "WITH c USING KEY (k AS j) AS MATERIALIZED (SELECT 1) SELECT 1"
    assert_equal(_printed(sql, g, rules), sql)


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


def test_a_pivot_or_an_unpivot_after_a_table_is_read() raises:
    # What the two turn into is tested in `test_sql_pivot.mojo` and
    # `test_sql_unpivot.mojo`. The one thing this file cares about is that the
    # modifier position after a table reaches both of them, which is the same
    # position a join goes in.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t PIVOT (sum(a) FOR b IN (1))", g, rules),
        "SELECT a FROM (PIVOT t ON b IN (1) USING sum(a))",
    )
    assert_equal(
        _printed("SELECT a FROM t UNPIVOT (v FOR n IN (a, b))", g, rules),
        "SELECT a FROM (UNPIVOT t ON a, b INTO NAME n VALUE v)",
    )


def test_a_sample_reaches_the_printer_in_both_places() raises:
    var g = Grammar()
    var rules = Transform(g)
    # A sample written straight after the table belongs to the table, because
    # that is the first rule with a slot for one. It takes a clause in between
    # to reach the one on the statement.
    assert_equal(
        _printed("SELECT a FROM t TABLESAMPLE 10%", g, rules),
        "SELECT a FROM t TABLESAMPLE 10%",
    )
    assert_equal(
        _printed("SELECT a FROM t WHERE x USING SAMPLE 10%", g, rules),
        "SELECT a FROM t WHERE x USING SAMPLE 10%",
    )
    # A parenthesised reference takes one too, and it goes after the closing
    # parenthesis rather than inside it.
    assert_equal(
        _printed(
            "SELECT * FROM (t JOIN u ON t.a = u.a) USING SAMPLE 5", g, rules
        ),
        "SELECT * FROM (t JOIN u ON (t.a = u.a)) USING SAMPLE 5",
    )


def test_a_sample_is_printed_the_way_it_was_written() raises:
    # The keyword, the unit and whether the method stands in front of the
    # parentheses or inside them are all kept rather than normalized, since the
    # grammar takes every one of them in both positions and none of them
    # changes what the sample means.
    var g = Grammar()
    var rules = Transform(g)
    var written: List[String] = [
        "SELECT * FROM t USING SAMPLE 10",
        "SELECT * FROM t USING SAMPLE 10%",
        "SELECT * FROM t USING SAMPLE 10 PERCENT",
        "SELECT * FROM t USING SAMPLE 10 ROWS",
        "SELECT * FROM t USING SAMPLE 10 ROWS (system, 377)",
        "SELECT * FROM t USING SAMPLE 10% (bernoulli)",
        "SELECT * FROM t USING SAMPLE ?",
        "SELECT * FROM t TABLESAMPLE 1.5%",
        "SELECT * FROM t TABLESAMPLE reservoir(10)",
        "SELECT * FROM t TABLESAMPLE reservoir(10%) REPEATABLE (377)",
        "SELECT * FROM t TABLESAMPLE (10 ROWS) REPEATABLE (377)",
    ]
    for item in written:
        assert_equal(_printed(item, g, rules), item)


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


def test_the_colon_alias_on_a_table_is_the_same_alias() raises:
    # The same spelling in front of a table is a different rule, and DuckDB
    # takes this one or the `AS` one and not both. They mean the same
    # reference, the AST has one shape for it, and the printer writes the `AS`.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM x: t", g, rules), "SELECT a FROM t AS x"
    )
    assert_equal(
        _printed("SELECT a FROM x: main.t", g, rules),
        "SELECT a FROM main.t AS x",
    )
    # Every reference that takes an alias takes this one, which is four more
    # shapes and not just the table.
    assert_equal(
        _printed("SELECT a FROM x: range(3)", g, rules),
        "SELECT a FROM range(3) AS x",
    )
    assert_equal(
        _printed("SELECT a FROM x: (SELECT 1)", g, rules),
        "SELECT a FROM (SELECT 1) AS x",
    )
    assert_equal(
        _printed("SELECT a FROM x: (VALUES (1))", g, rules),
        "SELECT a FROM (VALUES (1)) AS x",
    )
    assert_equal(
        _printed("SELECT a FROM x: (t JOIN u ON t.a = u.a)", g, rules),
        "SELECT a FROM (t JOIN u ON (t.a = u.a)) AS x",
    )


def test_a_refusal_says_where_it_was() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="SELECT MAP {'a': 1} FROM t"):
        _ = _printed("SELECT MAP {'a': 1} FROM t", g, rules)


def test_the_at_on_a_table_reads_and_prints() raises:
    # `AT` asks for a table as of a version or a moment. There are two units
    # and what follows the arrow is a whole expression, so all of it goes in
    # the AST and comes back out the way it was written.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t AT (VERSION => 42)", g, rules),
        "SELECT a FROM t AT (VERSION => 42)",
    )
    # A typed literal is the cast it means everywhere else and is the cast it
    # means here too, which is the point of reading the value as an expression
    # rather than giving the clause a little language of its own.
    assert_equal(
        _printed(
            "SELECT a FROM t AT (TIMESTAMP => TIMESTAMP '2020-01-01')",
            g,
            rules,
        ),
        "SELECT a FROM t AT (TIMESTAMP => CAST('2020-01-01' AS TIMESTAMP))",
    )
    # The corpus writes a call, an arithmetic and a subquery in this position,
    # which is why the value is read as an expression and not as a literal.
    assert_equal(
        _printed("SELECT a FROM t AT (TIMESTAMP => now())", g, rules),
        "SELECT a FROM t AT (TIMESTAMP => now())",
    )
    assert_equal(
        _printed(
            "SELECT a FROM t AT (TIMESTAMP => (SELECT min(ts) FROM u))",
            g,
            rules,
        ),
        "SELECT a FROM t AT (TIMESTAMP => (SELECT min(ts) FROM u))",
    )


def test_the_at_is_written_after_the_alias_and_before_the_sample() raises:
    # The grammar takes the alias, then the `AT`, then the sample, and the
    # three live in three places on the reference, so the order they go back
    # out in is the printer's to get right rather than something it inherits.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a FROM t s(c1, c2) AT (VERSION => 1)", g, rules),
        "SELECT a FROM t AS s (c1, c2) AT (VERSION => 1)",
    )
    assert_equal(
        _printed(
            "SELECT a FROM t AS s AT (VERSION => 1) TABLESAMPLE 10%", g, rules
        ),
        "SELECT a FROM t AS s AT (VERSION => 1) TABLESAMPLE 10%",
    )


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
    with assert_raises(contains="a MAP literal"):
        _ = _printed("SELECT MAP {'a': 1}", g, rules)


def test_grouping_prints_back_in_capitals() raises:
    # GROUPING_ID is the same call under another name, and DuckDB names the
    # column it answers GROUPING either way.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed(
            "SELECT grouping(a), grouping_id(a, b) FROM t GROUP BY CUBE (a, b)",
            g,
            rules,
        ),
        "SELECT GROUPING(a), GROUPING(a, b) FROM t GROUP BY CUBE (a, b)",
    )
    with assert_raises(contains='syntax error at or near ")"'):
        _ = _printed("SELECT GROUPING() FROM t GROUP BY a", g, rules)


def test_a_positional_column_prints_back_as_it_was_written() raises:
    # DuckDB takes only a whole number with no sign that fits in 32 bits after
    # the `#`, and says so as a syntax error at the part it would not read.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(_printed("SELECT #1 FROM t", g, rules), "SELECT #1 FROM t")
    assert_equal(
        _printed("SELECT # 2 + #01 FROM t WHERE #1 > 0", g, rules),
        "SELECT (#2 + #1) FROM t WHERE (#1 > 0)",
    )
    with assert_raises(contains="needs to be >= 1"):
        _ = _printed("SELECT #0 FROM t", g, rules)
    with assert_raises(contains='syntax error at or near "1.5"'):
        _ = _printed("SELECT #1.5 FROM t", g, rules)
    with assert_raises(contains='syntax error at or near "2147483648"'):
        _ = _printed("SELECT #2147483648 FROM t", g, rules)


def test_the_keyword_calls_that_no_longer_refuse() raises:
    # SUBSTRING and EXTRACT were in the list above until the kernels behind them
    # were wired up. Each comes out of the transform as the call the planner
    # reads, which is why what is printed back is not what was written.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT SUBSTRING(a FROM 1 FOR 2) FROM t", g, rules),
        "SELECT substring(a, 1, 2) FROM t",
    )
    assert_equal(
        _printed("SELECT EXTRACT(YEAR FROM a) FROM t", g, rules),
        "SELECT date_part('year', a) FROM t",
    )
    assert_equal(
        _printed("SELECT TRIM(BOTH ' ' FROM a) FROM t", g, rules),
        "SELECT trim(a, ' ') FROM t",
    )
    assert_equal(
        _printed("SELECT TRIM(LEADING FROM a) FROM t", g, rules),
        "SELECT ltrim(a) FROM t",
    )
    assert_equal(
        _printed("SELECT POSITION(a IN b) FROM t", g, rules),
        "SELECT instr(b, a) FROM t",
    )


def test_the_last_three_keyword_calls_read_as_the_calls_they_are() raises:
    # These three were the whole of the special-call entry. OVERLAY is a call
    # with two spellings the way SUBSTRING is, and DuckDB has no function of
    # that name either, so the query stops on the name in both engines. TRY and
    # UNPACK are one argument each and are turned down at lowering.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT OVERLAY(a PLACING b FROM 1 FOR 2) FROM t", g, rules),
        "SELECT overlay(a, b, 1, 2) FROM t",
    )
    assert_equal(
        _printed("SELECT OVERLAY(a PLACING b FROM 1) FROM t", g, rules),
        "SELECT overlay(a, b, 1) FROM t",
    )
    # The comma spelling is the same call and prints the same way.
    assert_equal(
        _printed("SELECT overlay(a, b, 1, 2) FROM t", g, rules),
        "SELECT overlay(a, b, 1, 2) FROM t",
    )
    assert_equal(
        _printed("SELECT TRY(a) FROM t", g, rules), "SELECT try(a) FROM t"
    )
    assert_equal(
        _printed("SELECT UNPACK(a) FROM t", g, rules),
        "SELECT unpack(a) FROM t",
    )


def test_an_interval_reaches_the_printer_and_is_refused_further_on() raises:
    # It was in the list above until the transformer learned to build one. The
    # type is what is missing rather than the syntax, so the statement prints
    # and the refusal waits for the stage that would have to name a type.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT INTERVAL '1 day' FROM t", g, rules),
        "SELECT INTERVAL '1 day' FROM t",
    )
    assert_equal(
        _printed("SELECT a + INTERVAL 3 MONTHS FROM t", g, rules),
        "SELECT (a + INTERVAL 3 MONTH) FROM t",
    )


def test_a_row_value_reaches_the_printer_in_both_spellings() raises:
    # `(a, b)` and `ROW(a, b)` are one thing written two ways, and the word is
    # kept so that what comes back is what was written.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT (a, b) FROM t", g, rules), "SELECT (a, b) FROM t"
    )
    assert_equal(
        _printed("SELECT ROW(a, b) FROM t", g, rules),
        "SELECT ROW(a, b) FROM t",
    )
    assert_equal(
        _printed("SELECT ROW() FROM t", g, rules), "SELECT ROW() FROM t"
    )


def test_an_argument_passed_by_name_reaches_the_printer() raises:
    # The same rule serves a call in a select list and a table function in a
    # `FROM`, so both paths are checked here. The second is the one the corpus
    # is full of, since `read_csv` takes most of its settings by name.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT f(a := 1) FROM t", g, rules),
        "SELECT f(a := 1) FROM t",
    )
    assert_equal(
        _printed("SELECT * FROM read_csv('x.csv', header := TRUE)", g, rules),
        "SELECT * FROM read_csv('x.csv', header := TRUE)",
    )


def test_a_subscript_reaches_the_printer_too() raises:
    # Same as the interval above it. The syntax is read and the stage that
    # would have to know what the operand holds is the one that refuses.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT a[1:2] FROM t", g, rules),
        "SELECT a[1:2] FROM t",
    )


def test_columns_reaches_the_printer_in_a_select_list() raises:
    # It was in the list of transformer refusals above until the transformer
    # learned to build one. What is missing is the bindings rather than the
    # syntax, so the statement prints and the refusal waits for the stage that
    # knows which columns there are to match against.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT COLUMNS('a') FROM t", g, rules),
        "SELECT COLUMNS('a') FROM t",
    )
    assert_equal(
        _printed("SELECT min(COLUMNS(*)) FROM t", g, rules),
        "SELECT min(COLUMNS(*)) FROM t",
    )
    assert_equal(
        _printed("SELECT *COLUMNS(['a', 'b']) FROM t", g, rules),
        "SELECT *COLUMNS(['a', 'b']) FROM t",
    )


def test_a_typed_literal_is_the_cast_it_means() raises:
    # `DATE '2020-01-01'` is a type name in front of a string, and the type is
    # what decides how the string is read, which is the whole of what a cast
    # does. DuckDB rewrites it the same way, so this prints back longer than it
    # was written.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT DATE '2020-01-01' FROM t", g, rules),
        "SELECT CAST('2020-01-01' AS DATE) FROM t",
    )
    assert_equal(
        _printed("SELECT INTEGER '42' FROM t", g, rules),
        "SELECT CAST('42' AS INTEGER) FROM t",
    )


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
    # statement rule, and the tier loop never reached them. `PIVOT` and
    # `UNPIVOT` were here too and are read now, so `DESCRIBE` is what is left.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="the DESCRIBE statement yet"):
        _ = _printed("DESCRIBE t", g, rules)


def test_a_parenthesised_expression_is_still_just_the_expression() raises:
    # `ParenthesisExpression` is the rule that builds a row value, and a single
    # expression in parentheses is a different rule that has to keep going
    # through untouched.
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


def test_the_table_is_total_over_everything_it_can_reach() raises:
    # Building the table is the check. `Transform.__init__` walks the grammar
    # from `Statement` and refuses to hand back a table with a reachable rule
    # that has no case, so every test in every file is already running it and
    # this line is what says so out loud.
    var g = Grammar()
    var rules = Transform(g)

    # Which leaves one thing worth asserting from out here: that the check is
    # not passing because everything got a case. Rules with none are still
    # there in their hundreds, all of them under a statement that refuses
    # before anything below it is read, and that is the shape to keep.
    var without = 0
    for i in range(len(g.names)):
        if rules.actions[i] == 0:
            without += 1
    assert_true(without > 0, "every rule in the grammar has a case, somehow")
    assert_true(
        without < len(g.names) // 2,
        String(without, " of ", len(g.names), " rules have no case"),
    )


def test_a_dotted_name_says_which_position_turned_it_down() raises:
    # Twenty five places in the grammar take a plain name and they all used to
    # refuse the same six words. In a statement with several names in it that
    # is true and no help at all, since the reader is left to work out which of
    # the names was the one. The position is the part only the caller knows.
    # One position is left that can reach it, which is the name a REPLACE gives
    # its value, because that is the one of the three DuckDB's own parser will
    # not take a dot in either.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="where a column REPLACE names goes"):
        _ = _printed("SELECT * REPLACE (1 AS a.b) FROM t", g, rules)
    with assert_raises(contains="Write the last part on its own"):
        _ = _printed("SELECT * REPLACE (1 AS a.b) FROM t", g, rules)


def test_only_the_star_modifiers_can_reach_the_dotted_name_refusal() raises:
    # Three of the twenty five positions take a node the grammar will put a dot
    # in. The rest stop at the parser, which is a better error than the refusal
    # would have been and is why naming the position was worth measuring rather
    # than assuming. Each of these is a name in one of the other positions and
    # each of them stops one step earlier.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="syntax error"):
        _ = _printed("SELECT * RENAME (a AS b.c) FROM t", g, rules)
    with assert_raises(contains="syntax error"):
        _ = _printed("SELECT 1 AS a.b FROM t", g, rules)
    with assert_raises(contains="syntax error"):
        _ = _printed("SELECT * FROM t JOIN u USING (a.b)", g, rules)
    with assert_raises(contains="syntax error"):
        _ = _printed("WITH a.b AS (SELECT 1) SELECT 1", g, rules)


def test_a_star_modifier_keeps_the_binding_it_named() raises:
    # EXCLUDE and the left of a RENAME are the two DuckDB takes a qualifier in,
    # and the printer has to put it back where the query had it, because a
    # modifier that came out bare would name the column of whichever binding
    # has it and that is a different query.
    var g = Grammar()
    var rules = Transform(g)
    assert_equal(
        _printed("SELECT * EXCLUDE (t.a) FROM t, u", g, rules),
        "SELECT * EXCLUDE (t.a) FROM t, u",
    )
    assert_equal(
        _printed("SELECT * RENAME (u.c AS d) FROM t, u", g, rules),
        "SELECT * RENAME (u.c AS d) FROM t, u",
    )
    assert_equal(
        _printed('SELECT * EXCLUDE ("t.x"."c.y") FROM t', g, rules),
        'SELECT * EXCLUDE ("t.x"."c.y") FROM t',
    )
    # A bare one still prints bare, which is the empty qualifier coming back
    # out as nothing rather than as a leading dot.
    assert_equal(
        _printed("SELECT * EXCLUDE (a) RENAME (b AS c) FROM t", g, rules),
        "SELECT * EXCLUDE (a) RENAME (b AS c) FROM t",
    )


def test_a_star_modifier_of_three_parts_says_it_needs_the_catalog() raises:
    # Two parts are a binding and a column. Three start at a schema or at a
    # struct and nothing short of the catalog tells those apart, which is the
    # same wall the star's own qualifier stops at.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="a name of three parts or more"):
        _ = _printed("SELECT * EXCLUDE (s.t.a) FROM t", g, rules)
    with assert_raises(contains="where the column RENAME renames goes"):
        _ = _printed("SELECT * RENAME (s.t.a AS b) FROM t", g, rules)
    with assert_raises(contains="telling those apart needs the catalog"):
        _ = _printed("SELECT * EXCLUDE (a.b.c.d) FROM t", g, rules)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
