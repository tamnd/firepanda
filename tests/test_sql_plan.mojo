"""Lowering a SELECT to a logical plan.

The tests read the plan back as the text `explain` prints, for the reason the
transform tests read SQL back as text: a clause that is silently dropped shows
up as a missing line rather than as a field in an arena nobody looks at. The
indentation is the tree, so the order the nodes come out in is the order the
query runs in, and that order is most of what this stage decides.

The last two are the ones worth having. One asserts that the plan a query
produces is the plan the equivalent dataframe calls produce, which is the rule
in docs/specs/sql/08-plan-and-optimizer.md section 6 written as something that
can fail. The other asserts that every shape this does not lower yet says so by
name, because a lowering that quietly returned a plan for a join would be a
wrong answer and a refusal is not.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame import DataFrame
from firepanda.kernel.binary import BinaryOp
from firepanda.plan.bind import bind
from firepanda.plan.node import Plan
from firepanda.plan.print import explain
from firepanda.sql import Grammar, Transform
from firepanda.sql.ast import Ast
from firepanda.sql.catalog import Catalog
from firepanda.sql.plan import lower


def _schema() -> Schema:
    """Four columns, one of each type the tests need.

    Returns:
        The schema.
    """
    var out = Schema()
    out.append(Field("a", LogicalType.INT64, False))
    out.append(Field("b", LogicalType.INT64, False))
    out.append(Field("g", LogicalType.STRING, True))
    out.append(Field("f", LogicalType.FLOAT64, True))
    return out^


def _catalog() raises -> Catalog:
    """A session holding one frame called `t`.

    Returns:
        The catalog.

    Raises:
        Error: If the name cannot be registered.
    """
    var frame = DataFrame()
    frame.schema = _schema()
    var catalog = Catalog()
    catalog.register("t", frame^)
    return catalog^


def _plan(sql: StringSlice) raises -> String:
    """Parses a query, lowers it, binds it and prints the plan.

    Binding is not optional here even though nothing reads the types back. It is
    what turns every name into a position, so a lowering that emitted a column
    no scan provides fails in this helper rather than in a later stage that has
    already forgotten which clause the name was written in.

    Args:
        sql: The whole statement.

    Returns:
        The plan as `explain` writes it.

    Raises:
        Error: If it does not parse, does not lower or does not bind.
    """
    var grammar = Grammar()
    var rules = Transform(grammar)
    var ast = Ast()
    var node = rules.parse_statement(sql, grammar, ast)
    var out = lower(ast, node, _catalog())
    _ = bind(out.plan, out.root, out.sources)
    return explain(out.plan, out.root)


def test_the_smallest_query_that_reads_a_table() raises:
    assert_equal(_plan("SELECT a FROM t"), "PROJECT [a]\n  SCAN t []\n")


def test_a_star_is_every_column_the_scan_has() raises:
    # The empty bracket on the scan is a scan that was not told which columns to
    # read, which is every one of them. Column pruning is what fills it in, and
    # it runs over the plan rather than in the lowering.
    assert_equal(
        _plan("SELECT * FROM t"), "PROJECT [a, b, g, f]\n  SCAN t []\n"
    )


def test_a_where_sits_under_the_projection_and_not_over_it() raises:
    # The order is the whole point. A WHERE that landed above the projection
    # could name an alias the select list invented, which DuckDB refuses, and
    # nothing else in the front end would catch it.
    assert_equal(
        _plan("SELECT a FROM t WHERE b > 1"),
        "PROJECT [a]\n  FILTER b > 1\n    SCAN t []\n",
    )


def test_the_clauses_come_out_in_the_order_they_run_in() raises:
    assert_equal(
        _plan("SELECT DISTINCT a FROM t WHERE b > 1 ORDER BY a LIMIT 10"),
        (
            "LIMIT 10\n"
            "  SORT [a asc nulls last]\n"
            "    DISTINCT [*]\n"
            "      PROJECT [a]\n"
            "        FILTER b > 1\n"
            "          SCAN t []\n"
        ),
    )


def test_a_group_by_puts_the_keys_and_the_folds_in_one_node() raises:
    assert_equal(
        _plan("SELECT g, sum(a) FROM t GROUP BY g"),
        (
            "PROJECT [g, __agg_0 as __expr_1]\n"
            "  AGGREGATE [g] -> [sum(a)]\n"
            "    SCAN t []\n"
        ),
    )


def test_an_aggregate_with_no_group_by_still_aggregates() raises:
    # No GROUP BY and one fold is one group, and the node that computes it is
    # the same node, which is why whether a query aggregates has to be decided
    # from the select list rather than from the presence of a clause.
    assert_equal(
        _plan("SELECT sum(a) FROM t"),
        (
            "PROJECT [__agg_0 as __expr_0]\n"
            "  AGGREGATE [] -> [sum(a)]\n"
            "    SCAN t []\n"
        ),
    )


def test_a_having_filters_the_column_the_aggregate_produced() raises:
    # The fold is computed once, in the AGGREGATE node, and the HAVING reads the
    # column it landed in. A filter holding a sum is refused by the plan itself,
    # which is the check that keeps this honest.
    assert_equal(
        _plan("SELECT g FROM t GROUP BY g HAVING sum(a) > 10"),
        (
            "PROJECT [g]\n"
            "  FILTER __agg_0 > 10\n"
            "    AGGREGATE [g] -> [sum(a)]\n"
            "      SCAN t []\n"
        ),
    )


def test_count_star_folds_over_a_constant() raises:
    # There is no column in `count(*)` to fold over, and counting rows is
    # counting a value every row has, so the lowering supplies one.
    assert_true("count(1)" in _plan("SELECT count(*) FROM t GROUP BY g"))


def test_a_descending_sort_keeps_its_nulls_where_duckdb_puts_them() raises:
    assert_true(
        _plan("SELECT a FROM t ORDER BY a DESC").startswith("SORT [a desc]\n")
    )


def test_an_offset_with_no_limit_keeps_the_rest() raises:
    assert_true(
        _plan("SELECT a FROM t OFFSET 5").startswith("LIMIT all offset 5\n")
    )


def test_the_sql_path_and_the_frame_path_build_the_same_plan() raises:
    # The rule from the plan spec, as a test that can fail. If these two ever
    # disagree then one of the two front ends has quietly become a second
    # engine, which is the thing one shared plan exists to stop.
    var by_hand = Plan()
    var at = by_hand.scan("t", List[String](), 0)
    var predicate = by_hand.exprs.binary(
        BinaryOp.GT,
        by_hand.exprs.column("b"),
        by_hand.exprs.literal(Value(Int64(1))),
    )
    at = by_hand.filter(at, predicate)
    at = by_hand.project(at, [by_hand.exprs.column("a")], ["a"])
    _ = bind(by_hand, at, [_schema()])

    assert_equal(_plan("SELECT a FROM t WHERE b > 1"), explain(by_hand, at))


def test_a_values_is_the_rows_that_were_written() raises:
    assert_equal(
        _plan("VALUES (1, 'a'), (2, 'b')"),
        "VALUES [col0, col1] (1, a), (2, b)\n",
    )


def test_a_select_with_no_from_projects_over_one_row() raises:
    # The row is there so that the projection has something to be one row of,
    # and its value is never read.
    assert_equal(
        _plan("SELECT 1 AS a, 2 AS b"),
        "PROJECT [1 as a, 2 as b]\n  VALUES [__row] (0)\n",
    )


def test_a_star_with_no_from_has_nothing_to_stand_for() raises:
    with assert_raises(contains="nothing for it to stand for"):
        _ = _plan("SELECT *")


def test_every_row_of_a_values_is_the_same_width() raises:
    with assert_raises(contains="row 2 of a VALUES has 1 values"):
        _ = _plan("VALUES (1, 2), (3)")


def test_a_values_column_is_the_type_that_holds_every_row() raises:
    var grammar = Grammar()
    var rules = Transform(grammar)
    var ast = Ast()
    var node = rules.parse_statement("VALUES (1), (NULL)", grammar, ast)
    var out = lower(ast, node, _catalog())
    var schema = bind(out.plan, out.root, out.sources)
    assert_equal(len(schema), 1, "one column")
    assert_equal(schema[0].name, "col0", "named the way DuckDB names it")
    assert_true(schema[0].nullable, "one NULL in the column is enough")


def test_a_union_stacks_two_blocks_and_drops_duplicates_by_default() raises:
    assert_equal(
        _plan("SELECT a FROM t UNION SELECT b FROM t"),
        "UNION\n  PROJECT [a]\n    SCAN t []\n  PROJECT [b]\n    SCAN t []\n",
    )
    assert_equal(
        _plan("SELECT a FROM t UNION ALL SELECT b FROM t"),
        (
            "UNION all\n  PROJECT [a]\n    SCAN t []\n  PROJECT [b]\n   "
            " SCAN t []\n"
        ),
    )


def test_the_other_two_set_operations_lower_to_the_same_node() raises:
    assert_equal(
        _plan("SELECT a FROM t EXCEPT SELECT b FROM t"),
        "EXCEPT\n  PROJECT [a]\n    SCAN t []\n  PROJECT [b]\n    SCAN t []\n",
    )
    assert_equal(
        _plan("SELECT a FROM t INTERSECT ALL SELECT b FROM t"),
        (
            "INTERSECT all\n  PROJECT [a]\n    SCAN t []\n  PROJECT [b]\n   "
            " SCAN t []\n"
        ),
    )


def test_a_chain_of_set_operations_nests_to_the_left() raises:
    # Not an argument about style. `a EXCEPT b EXCEPT c` and
    # `a EXCEPT (b EXCEPT c)` hold different rows, so which way the chain leans
    # is part of the answer.
    assert_equal(
        _plan("SELECT a FROM t EXCEPT SELECT b FROM t EXCEPT SELECT a FROM t"),
        (
            "EXCEPT\n  EXCEPT\n    PROJECT [a]\n      SCAN t []\n   "
            " PROJECT [b]\n      SCAN t []\n  PROJECT [a]\n    SCAN t []\n"
        ),
    )


def test_an_order_by_after_a_union_sorts_the_union() raises:
    # Written after the right arm and applying to both of them, which is why the
    # modifiers are put on outside the block rather than at the end of one.
    assert_equal(
        _plan(
            "SELECT a FROM t UNION ALL SELECT b FROM t ORDER BY a DESC LIMIT 3"
        ),
        (
            "LIMIT 3\n  SORT [a desc]\n    UNION all\n      PROJECT [a]\n     "
            "   SCAN t []\n      PROJECT [b]\n        SCAN t []\n"
        ),
    )


def test_the_two_arms_of_a_set_operation_bind_against_their_own_tables() raises:
    # Two scans and two schemas, so the second scan has to carry relation one.
    # Carrying zero used to be right when a plan held one table and is a wrong
    # answer that binds anyway now that it can hold two.
    var grammar = Grammar()
    var rules = Transform(grammar)
    var ast = Ast()
    var node = rules.parse_statement(
        "SELECT a FROM t UNION SELECT b FROM t", grammar, ast
    )
    var out = lower(ast, node, _catalog())
    assert_equal(len(out.sources), 2, "one schema for each arm")
    var schema = bind(out.plan, out.root, out.sources)
    assert_equal(len(schema), 1, "one column out")
    assert_equal(schema[0].name, "a", "named by the left arm")


def test_a_set_operation_written_by_name_is_refused_by_name() raises:
    with assert_raises(contains="BY NAME"):
        _ = _plan("SELECT a FROM t UNION BY NAME SELECT b FROM t")


def test_two_arms_of_different_widths_do_not_stack() raises:
    with assert_raises(contains="2 column input and a 1 column one"):
        _ = _plan("SELECT a, b FROM t UNION SELECT b FROM t")


def test_a_name_the_catalog_does_not_have_suggests_one_it_does() raises:
    with assert_raises(contains="t"):
        _ = _plan("SELECT a FROM tt")


def test_a_join_is_refused_by_name_rather_than_lowered_wrong() raises:
    with assert_raises(contains="one table in a FROM"):
        _ = _plan("SELECT a FROM t, t")
    with assert_raises(contains="a named table in a FROM"):
        _ = _plan("SELECT a FROM t JOIN t u ON t.a = u.a")


def test_a_decimal_literal_is_refused_rather_than_made_a_double() raises:
    # The one refusal in here that is not about effort. A double in place of an
    # exact decimal answers a different question and says nothing about it.
    with assert_raises(contains="3.3000000000000003"):
        _ = _plan("SELECT a FROM t WHERE f > 1.5")


def test_the_shapes_with_no_node_yet_each_say_which_one() raises:
    with assert_raises(contains="WITH"):
        _ = _plan("WITH x AS (SELECT 1 AS a) SELECT a FROM x")
    with assert_raises(contains="window function"):
        _ = _plan("SELECT row_number() OVER () FROM t")
    with assert_raises(contains="QUALIFY"):
        _ = _plan("SELECT a FROM t QUALIFY row_number() OVER () = 1")
    with assert_raises(contains="GROUPING SETS"):
        _ = _plan("SELECT g FROM t GROUP BY CUBE (g)")
    with assert_raises(contains="BETWEEN"):
        _ = _plan("SELECT a FROM t WHERE b BETWEEN 1 AND 2")
    with assert_raises(contains="CAST"):
        _ = _plan("SELECT CAST(a AS BIGINT) FROM t")


def test_a_column_can_say_which_table_it_is_from() raises:
    # The qualifier resolves here, at lowering, because the FROM that gives it
    # its meaning has already been walked. Binding then has a relation to look
    # the name up in rather than a word it would have to resolve itself.
    assert_equal(_plan("SELECT t.a FROM t"), "PROJECT [a]\n  SCAN t []\n")


def test_an_alias_is_what_the_table_is_called_after_it() raises:
    assert_equal(_plan("SELECT l.a FROM t AS l"), "PROJECT [a]\n  SCAN t []\n")


def test_an_alias_hides_the_table_name_rather_than_adding_to_it() raises:
    # SQL's rule, and the reason it is a rule is that a query which could write
    # either would have two names for one thing and no way to tell them apart
    # once a second table arrived.
    with assert_raises(contains="nothing in this query is called 't'"):
        _ = _plan("SELECT t.a FROM t AS l")


def test_the_message_says_what_the_from_did_bring() raises:
    with assert_raises(contains="the FROM brought 'l'"):
        _ = _plan("SELECT x.a FROM t AS l")


def test_a_qualified_name_in_a_where_reads_the_same_way() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE t.b > 1"),
        "PROJECT [a]\n  FILTER b > 1\n    SCAN t []\n",
    )


def test_a_qualified_name_a_table_does_not_have_is_still_missing() raises:
    with assert_raises(contains="there is no column named 'nope'"):
        _ = _plan("SELECT t.nope FROM t")


def test_a_qualified_name_with_no_from_has_nothing_to_qualify() raises:
    with assert_raises(contains="the FROM brought nothing"):
        _ = _plan("SELECT t.a")


def test_a_three_part_column_is_refused_by_name() raises:
    with assert_raises(contains="a table and a name so far"):
        _ = _plan("SELECT s.t.a FROM t")


def test_a_qualified_name_can_be_ordered_by() raises:
    # The ORDER BY is lowered outside the block, so it only sees the table if
    # the block hands its scope back, and this is the test that says it does.
    assert_equal(
        _plan("SELECT a, b FROM t ORDER BY t.b"),
        "SORT [b asc nulls last]\n  PROJECT [a, b]\n    SCAN t []\n",
    )


def test_column_aliases_on_a_table_are_refused_by_name() raises:
    with assert_raises(contains="column aliases on a table reference"):
        _ = _plan("SELECT x FROM t AS l (x, y, z, w)")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
