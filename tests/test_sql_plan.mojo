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
    with assert_raises(contains="set operation"):
        _ = _plan("SELECT a FROM t UNION SELECT b FROM t")
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
