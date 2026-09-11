"""A `WITH` clause: what each name stands for, and where each name can be said.

Every rule and every message here was read off DuckDB 1.5. The ones worth
naming are the ones that surprise: a `WITH` that names a later entry gets a
missing table rather than a forward reference, a column alias list of the wrong
length is silently half applied either way, and one circular reference message
covers three different mistakes, only one of which is the one it describes.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.sql import Grammar, Transform
from firepanda.sql.ast import (
    Ast,
    MATERIALIZE_DEFAULT,
    MATERIALIZE_NO,
    MATERIALIZE_YES,
)
from firepanda.sql.cte import (
    NOT_A_CTE,
    aliased,
    anchored,
    circular_reference,
    duplicate_name,
    no_limit,
    no_ordering,
    read_ctes,
    reference_count,
)


def _clause(sql: StringSlice, grammar: Grammar, mut rules: Transform) raises:
    """Parses a statement and reads its `WITH`, throwing the clause away.

    Args:
        sql: The whole statement.
        grammar: A loaded grammar.
        rules: The jump table built from it.

    Raises:
        Error: If it does not parse, or if the `WITH` breaks a rule.
    """
    var ast = Ast()
    var node = rules.parse_statement(sql, grammar, ast)
    _ = read_ctes(ast, node)


def _names(*parts: String) -> List[String]:
    """A list of names.

    Args:
        parts: The names, in order.

    Returns:
        The list.
    """
    var out = List[String]()
    for part in parts:
        out.append(part)
    return out^


def test_the_entries_come_back_in_order() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        "WITH a AS (SELECT 1), b AS (SELECT 2) SELECT * FROM b", g, ast
    )
    var clause = read_ctes(ast, node)
    assert_equal(len(clause), 2)
    assert_equal(clause.entries[0].name, "a")
    assert_equal(clause.entries[1].name, "b")
    assert_false(clause.recursive)


def test_a_statement_with_no_with_has_an_empty_clause() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement("SELECT 1", g, ast)
    var clause = read_ctes(ast, node)
    assert_equal(len(clause), 0)
    assert_equal(clause.find("a"), NOT_A_CTE)


def test_a_name_is_looked_up_without_regard_to_case() raises:
    # A bare name is folded on the way into the arena, so the entry keeps the
    # folded spelling. A quoted one keeps the case it was written with and is
    # still found without it, because DuckDB matches a quoted name the same
    # way it matches a bare one.
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        'WITH Totals AS (SELECT 1), "Counts" AS (SELECT 2) SELECT 3', g, ast
    )
    var clause = read_ctes(ast, node)
    assert_equal(clause.entries[0].name, "totals")
    assert_equal(clause.entries[1].name, "Counts")
    assert_equal(clause.find("TOTALS"), 0)
    assert_equal(clause.find("counts"), 1)
    assert_equal(clause.find("total"), NOT_A_CTE)


def test_an_entry_sees_the_ones_before_it_and_not_itself() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        "WITH a AS (SELECT 1), b AS (SELECT * FROM a) SELECT * FROM b", g, ast
    )
    var clause = read_ctes(ast, node)
    assert_equal(clause.visible("a", 1), 0)
    assert_equal(clause.visible("b", 1), NOT_A_CTE)
    assert_equal(clause.visible("a", 0), NOT_A_CTE)


def test_one_name_twice_is_a_parser_error() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains='Duplicate CTE name "x"'):
        _clause("WITH x AS (SELECT 1), x AS (SELECT 2) SELECT 1", g, rules)


def test_the_duplicate_is_caught_without_regard_to_case() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="Duplicate CTE name"):
        _clause("WITH x AS (SELECT 1), X AS (SELECT 2) SELECT 1", g, rules)


def test_a_use_is_counted_once_for_the_body_and_once_for_a_later_entry() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        "WITH a AS (SELECT 1), b AS (SELECT * FROM a) SELECT * FROM a, b, a",
        g,
        ast,
    )
    var clause = read_ctes(ast, node)
    assert_equal(clause.entries[0].uses, 3)
    assert_equal(clause.entries[1].uses, 1)


def test_a_name_used_nowhere_is_counted_zero() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement("WITH a AS (SELECT 1) SELECT 2", g, ast)
    var clause = read_ctes(ast, node)
    assert_equal(clause.entries[0].uses, 0)


def test_a_use_inside_a_subquery_is_still_a_use() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        "WITH a AS (SELECT 1) SELECT (SELECT count(*) FROM a)", g, ast
    )
    var clause = read_ctes(ast, node)
    assert_equal(clause.entries[0].uses, 1)


def test_an_inner_with_that_binds_the_name_again_hides_it() raises:
    # The inner binding is a different table that happens to be spelled the
    # same, so naming it is not naming the outer one.
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        (
            "WITH a AS (SELECT 1) SELECT * FROM (WITH a AS (SELECT 2) SELECT *"
            " FROM a)"
        ),
        g,
        ast,
    )
    var clause = read_ctes(ast, node)
    assert_equal(clause.entries[0].uses, 0)


def test_a_forward_reference_is_not_a_use() raises:
    # DuckDB reports this as a missing table rather than a forward reference,
    # because entries bind in order and the name is simply not bound yet.
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        "WITH a AS (SELECT * FROM b), b AS (SELECT 1) SELECT * FROM a", g, ast
    )
    var clause = read_ctes(ast, node)
    assert_equal(clause.entries[1].uses, 0)
    assert_equal(clause.visible("b", 0), NOT_A_CTE)


def test_a_qualified_name_is_not_a_cte() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        "WITH a AS (SELECT 1) SELECT * FROM s.a", g, ast
    )
    var clause = read_ctes(ast, node)
    assert_equal(clause.entries[0].uses, 0)


def test_the_materialize_hint_is_recorded_and_not_acted_on() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        (
            "WITH a AS MATERIALIZED (SELECT 1), b AS NOT MATERIALIZED (SELECT"
            " 2), c AS (SELECT 3) SELECT 4"
        ),
        g,
        ast,
    )
    var clause = read_ctes(ast, node)
    assert_equal(clause.entries[0].materialize, MATERIALIZE_YES)
    assert_equal(clause.entries[1].materialize, MATERIALIZE_NO)
    assert_equal(clause.entries[2].materialize, MATERIALIZE_DEFAULT)


def test_the_column_alias_list_is_read_in_order() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        "WITH a(p, q) AS (SELECT 1, 2) SELECT * FROM a", g, ast
    )
    var clause = read_ctes(ast, node)
    assert_equal(len(clause.entries[0].columns), 2)
    assert_equal(clause.entries[0].columns[0], "p")
    assert_equal(clause.entries[0].columns[1], "q")


def test_a_short_alias_list_renames_a_prefix_and_says_nothing() raises:
    # `WITH x(p) AS (SELECT 1 AS a, 2 AS b)` gives back p and b.
    var renamed = aliased(_names("a", "b"), _names("p"))
    assert_equal(len(renamed), 2)
    assert_equal(renamed[0], "p")
    assert_equal(renamed[1], "b")


def test_a_long_alias_list_drops_the_extra_names_and_says_nothing() raises:
    # `WITH x(p, q, r) AS (SELECT 1 AS a, 2 AS b)` gives back p and q.
    var renamed = aliased(_names("a", "b"), _names("p", "q", "r"))
    assert_equal(len(renamed), 2)
    assert_equal(renamed[0], "p")
    assert_equal(renamed[1], "q")


def test_no_alias_list_leaves_every_name_alone() raises:
    var renamed = aliased(_names("a", "b"), List[String]())
    assert_equal(len(renamed), 2)
    assert_equal(renamed[0], "a")
    assert_equal(renamed[1], "b")


def test_a_recursive_entry_is_marked_and_the_others_are_not() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        (
            "WITH RECURSIVE a AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM a"
            " WHERE n < 5), b AS (SELECT 2) SELECT * FROM a"
        ),
        g,
        ast,
    )
    var clause = read_ctes(ast, node)
    assert_true(clause.recursive)
    assert_true(clause.entries[0].recursive)
    assert_false(clause.entries[1].recursive)


def test_union_without_all_is_an_anchor_too() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        (
            "WITH RECURSIVE a AS (SELECT 1 AS n UNION SELECT n + 1 FROM a"
            " WHERE n < 5) SELECT * FROM a"
        ),
        g,
        ast,
    )
    var clause = read_ctes(ast, node)
    assert_true(clause.entries[0].recursive)


def test_two_self_references_in_one_from_are_allowed() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        (
            "WITH RECURSIVE a AS (SELECT 1 AS n UNION ALL SELECT x.n + 1 FROM"
            " a AS x, a AS y WHERE x.n < 5) SELECT * FROM a"
        ),
        g,
        ast,
    )
    var clause = read_ctes(ast, node)
    assert_true(clause.entries[0].recursive)


def test_the_keyword_missing_is_a_circular_reference() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains='Circular reference to CTE "a"'):
        _clause(
            (
                "WITH a AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM a WHERE"
                " n < 5) SELECT * FROM a"
            ),
            g,
            rules,
        )


def test_the_keyword_alone_is_not_enough_without_a_union() raises:
    # Nothing to start the fixed point from, and DuckDB answers with the
    # message about the keyword even though the keyword is right there.
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="Circular reference to CTE"):
        _clause("WITH RECURSIVE a AS (SELECT * FROM a) SELECT 1", g, rules)


def test_except_and_intersect_are_not_anchors() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="Circular reference to CTE"):
        _clause(
            (
                "WITH RECURSIVE a AS (SELECT 1 AS n EXCEPT SELECT n FROM a)"
                " SELECT 1"
            ),
            g,
            rules,
        )
    with assert_raises(contains="Circular reference to CTE"):
        _clause(
            (
                "WITH RECURSIVE a AS (SELECT 1 AS n INTERSECT SELECT n FROM a)"
                " SELECT 1"
            ),
            g,
            rules,
        )


def test_the_anchor_is_the_left_side_and_may_not_recurse() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="Circular reference to CTE"):
        _clause(
            "WITH RECURSIVE a AS (SELECT n FROM a UNION ALL SELECT 1) SELECT 1",
            g,
            rules,
        )


def test_a_recursive_entry_may_not_order_itself() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(contains="ORDER BY in a recursive query is not allowed"):
        _clause(
            (
                "WITH RECURSIVE a AS (SELECT 1 AS n UNION ALL SELECT n + 1"
                " FROM a WHERE n < 5 ORDER BY n) SELECT * FROM a"
            ),
            g,
            rules,
        )


def test_a_recursive_entry_may_not_limit_itself() raises:
    var g = Grammar()
    var rules = Transform(g)
    with assert_raises(
        contains="LIMIT or OFFSET in a recursive query is not allowed"
    ):
        _clause(
            (
                "WITH RECURSIVE a AS (SELECT 1 AS n UNION ALL SELECT n + 1"
                " FROM a WHERE n < 5 LIMIT 10) SELECT * FROM a"
            ),
            g,
            rules,
        )


def test_the_rule_reaches_the_entry_and_nothing_under_it() raises:
    # A subquery inside the recursive term may order all it likes, because the
    # clause that is refused is the entry's own trailing one.
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        (
            "WITH RECURSIVE a AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM a"
            " WHERE n < (SELECT max(n) FROM a ORDER BY n)) SELECT * FROM a"
        ),
        g,
        ast,
    )
    var clause = read_ctes(ast, node)
    assert_true(clause.entries[0].recursive)


def test_a_plain_entry_may_order_and_limit_all_it_likes() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        "WITH a AS (SELECT n FROM t ORDER BY n LIMIT 3) SELECT * FROM a", g, ast
    )
    var clause = read_ctes(ast, node)
    assert_false(clause.entries[0].recursive)


def test_a_materialized_recursive_entry_is_both() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        (
            "WITH RECURSIVE a AS MATERIALIZED (SELECT 1 AS n UNION ALL SELECT"
            " n + 1 FROM a WHERE n < 5) SELECT * FROM a"
        ),
        g,
        ast,
    )
    var clause = read_ctes(ast, node)
    assert_true(clause.entries[0].recursive)
    assert_equal(clause.entries[0].materialize, MATERIALIZE_YES)


def test_a_reference_is_counted_in_every_clause_that_can_hold_one() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        (
            "WITH a AS (SELECT 1) SELECT (SELECT 1 FROM a) FROM t WHERE x IN"
            " (SELECT 1 FROM a) GROUP BY x HAVING count(*) > (SELECT 1 FROM a)"
            " ORDER BY (SELECT 1 FROM a) LIMIT (SELECT 1 FROM a)"
        ),
        g,
        ast,
    )
    var clause = read_ctes(ast, node)
    assert_equal(clause.entries[0].uses, 5)


def test_a_values_statement_holds_no_tables() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement(
        "WITH a AS (VALUES (1), (2)) SELECT * FROM a", g, ast
    )
    var clause = read_ctes(ast, node)
    assert_equal(clause.entries[0].uses, 1)
    assert_equal(reference_count(ast, clause.entries[0].statement, "a"), 0)


def test_the_short_table_spelling_names_a_table() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement("WITH a AS (SELECT 1) TABLE a", g, ast)
    var clause = read_ctes(ast, node)
    assert_equal(clause.entries[0].uses, 1)


def test_a_statement_with_no_union_has_no_anchor() raises:
    var g = Grammar()
    var rules = Transform(g)
    var ast = Ast()
    var node = rules.parse_statement("SELECT 1 AS n", g, ast)
    assert_false(anchored(ast, node, "a"))


def test_the_messages_are_the_ones_duckdb_writes() raises:
    assert_equal(duplicate_name("x"), 'Parser Error: Duplicate CTE name "x"')
    assert_equal(
        circular_reference("x"),
        (
            'Binder Error: Circular reference to CTE "x", use WITH RECURSIVE'
            " to use recursive CTEs."
        ),
    )
    assert_equal(
        no_ordering(),
        "Parser Error: ORDER BY in a recursive query is not allowed",
    )
    assert_equal(
        no_limit(),
        "Parser Error: LIMIT or OFFSET in a recursive query is not allowed",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
