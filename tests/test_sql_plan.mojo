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
name, because a lowering that quietly returned the wrong plan for a join would
be a wrong answer and a refusal is not.
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


def _other() -> Schema:
    """Three columns, one of which `t` also has.

    The shared name is what a join test needs: a query over both tables that
    writes it unqualified is ambiguous and one that qualifies it is not, and
    neither case exists with one table in the catalog.

    Returns:
        The schema.
    """
    var out = Schema()
    out.append(Field("b", LogicalType.INT64, False))
    out.append(Field("k", LogicalType.INT64, False))
    out.append(Field("z", LogicalType.STRING, True))
    return out^


def _catalog() raises -> Catalog:
    """A session holding a frame called `t` and one called `u`.

    Returns:
        The catalog.

    Raises:
        Error: If a name cannot be registered.
    """
    var frame = DataFrame()
    frame.schema = _schema()
    var catalog = Catalog()
    catalog.register("t", frame^)
    var second = DataFrame()
    second.schema = _other()
    catalog.register("u", second^)
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


def _outer(word: StringSlice) -> String:
    """One query with the join word written into it.

    Args:
        word: The words in front of `JOIN`.

    Returns:
        The query.
    """
    return String("SELECT a FROM t ", word, " JOIN u ON t.a = u.k")


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


def test_a_decimal_literal_is_refused_rather_than_made_a_double() raises:
    # The one refusal in here that is not about effort. A double in place of an
    # exact decimal answers a different question and says nothing about it.
    with assert_raises(contains="3.3000000000000003"):
        _ = _plan("SELECT a FROM t WHERE f > 1.5")


def test_the_shapes_with_no_node_yet_each_say_which_one() raises:
    with assert_raises(contains="GROUPING SETS"):
        _ = _plan("SELECT g FROM t GROUP BY CUBE (g)")
    with assert_raises(contains="subquery in an expression"):
        _ = _plan("SELECT a FROM t WHERE b IN (SELECT b FROM u)")
    with assert_raises(contains="subquery in an expression"):
        _ = _plan("SELECT a FROM t WHERE EXISTS (SELECT b FROM u)")
    with assert_raises(contains="TRY_CAST"):
        _ = _plan("SELECT TRY_CAST(a AS BIGINT) FROM t")


def test_a_cast_reads_its_type_name_against_the_dialect_type_set() raises:
    assert_equal(
        _plan("SELECT CAST(a AS BIGINT) AS wide FROM t"),
        "PROJECT [a::int64 as wide]\n  SCAN t []\n",
    )


def test_a_spelling_is_not_a_type_here_either() raises:
    # int8 is BIGINT and not TINYINT, so this is the one that would silently
    # narrow the column if the spelling table were read the other way round.
    assert_equal(
        _plan("SELECT CAST(a AS int8) AS wide FROM t"),
        "PROJECT [a::int64 as wide]\n  SCAN t []\n",
    )
    assert_equal(
        _plan("SELECT CAST(a AS int1) AS narrow FROM t"),
        "PROJECT [a::int8 as narrow]\n  SCAN t []\n",
    )


def test_a_cast_to_a_type_the_engine_has_no_column_for_is_refused() raises:
    with assert_raises(contains="integers stop at 64 bits"):
        _ = _plan("SELECT CAST(a AS HUGEINT) FROM t")
    with assert_raises(contains="no exact decimal"):
        _ = _plan("SELECT CAST(a AS DECIMAL(9,2)) FROM t")
    with assert_raises(contains="firepanda does not cast to DATE yet"):
        _ = _plan("SELECT CAST(a AS DATE) FROM t")


def test_a_cast_to_a_type_nobody_spells_that_way_is_refused() raises:
    with assert_raises(contains="does not exist"):
        _ = _plan("SELECT CAST(a AS BIGGINT) FROM t")


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


def test_a_join_is_one_node_over_two_scans() raises:
    assert_equal(
        _plan("SELECT t.a, u.z FROM t JOIN u ON t.a = u.k"),
        "PROJECT [a, z]\n  JOIN inner [a = k]\n    SCAN t []\n    SCAN u []\n",
    )


def test_a_comma_in_a_from_is_a_join_with_no_condition() raises:
    # The comma form and the JOIN form are the same query, and this is where
    # that stops being a claim: both reach a cross join with the equality over
    # it, and predicate pushdown is what turns either into a hash join later.
    assert_equal(
        _plan("SELECT a FROM t, u WHERE t.a = u.k"),
        (
            "PROJECT [a]\n"
            "  FILTER a == k\n"
            "    JOIN cross []\n"
            "      SCAN t []\n"
            "      SCAN u []\n"
        ),
    )


def test_a_cross_join_is_the_same_node_the_comma_built() raises:
    assert_equal(
        _plan("SELECT a FROM t CROSS JOIN u"),
        "PROJECT [a]\n  JOIN cross []\n    SCAN t []\n    SCAN u []\n",
    )


def test_each_outer_join_keeps_the_side_its_word_names() raises:
    assert_true("JOIN left [a = k]" in _plan(_outer("LEFT")))
    assert_true("JOIN right [a = k]" in _plan(_outer("RIGHT")))
    assert_true("JOIN outer [a = k]" in _plan(_outer("FULL OUTER")))
    # OUTER says nothing LEFT did not, so the two spellings are one join.
    assert_equal(_plan(_outer("LEFT")), _plan(_outer("LEFT OUTER")))


def test_the_rest_of_an_inner_condition_is_tested_above_the_join() raises:
    # The equality is a key pair and the comparison is not, and every pairing
    # the join produces is a pairing the condition asked about, so testing what
    # is left over above the join answers the same query.
    assert_equal(
        _plan("SELECT a FROM t JOIN u ON t.a = u.k AND t.b > u.k"),
        (
            "PROJECT [a]\n"
            "  FILTER b > k\n"
            "    JOIN inner [a = k]\n"
            "      SCAN t []\n"
            "      SCAN u []\n"
        ),
    )


def test_the_rest_of_an_outer_condition_is_refused_rather_than_moved() raises:
    # The same rewrite on an outer join is wrong. A row with no match is padded
    # and kept, and a filter above the join would then test the padding and
    # throw the row away, which is a different query and not a slower one.
    with assert_raises(contains="decides which rows are padded"):
        _ = _plan("SELECT a FROM t LEFT JOIN u ON t.a = u.k AND t.b > 1")


def test_two_equalities_are_two_key_pairs() raises:
    assert_true(
        "JOIN inner [b = b, a = k]"
        in _plan("SELECT a FROM t JOIN u ON t.b = u.b AND t.a = u.k")
    )


def test_a_pair_written_right_first_still_comes_out_left_first() raises:
    # The node holds a left key and a right key, so which side of the equals
    # sign each was written on is not what decides which list it goes in.
    assert_equal(
        _plan("SELECT a FROM t JOIN u ON u.k = t.a"),
        _plan("SELECT a FROM t JOIN u ON t.a = u.k"),
    )


def test_a_key_may_be_computed_rather_than_a_bare_column() raises:
    assert_true(
        "JOIN inner [a + 1 = k]"
        in _plan("SELECT a FROM t JOIN u ON t.a + 1 = u.k")
    )


def test_an_equality_with_only_one_side_in_it_is_not_a_key() raises:
    # Nothing about `t.a = 1` pairs a left row with a right row, so there is no
    # key to carry and what is left is every pairing with the test over it.
    assert_equal(
        _plan("SELECT a FROM t JOIN u ON t.a = 1"),
        (
            "PROJECT [a]\n"
            "  FILTER a == 1\n"
            "    JOIN cross []\n"
            "      SCAN t []\n"
            "      SCAN u []\n"
        ),
    )


def test_a_name_both_sides_have_is_refused_in_a_condition() raises:
    with assert_raises(contains="both sides of this join"):
        _ = _plan("SELECT a FROM t JOIN u ON b = k")


def test_a_name_neither_side_has_says_that_rather_than_guessing() raises:
    with assert_raises(contains="on either side of this join"):
        _ = _plan("SELECT a FROM t JOIN u ON t.a = q")


def test_a_star_over_a_join_is_the_two_sides_end_to_end() raises:
    # Both `b` columns are in it, which is what DuckDB answers, and each of them
    # says which input it is so that the projection can read one of each.
    assert_equal(
        _plan("SELECT * FROM t JOIN u ON t.a = u.k"),
        (
            "PROJECT [a, b, g, f, b, k, z]\n"
            "  JOIN inner [a = k]\n"
            "    SCAN t []\n"
            "    SCAN u []\n"
        ),
    )


def test_a_table_joined_to_itself_is_told_apart_by_its_aliases() raises:
    assert_equal(
        _plan("SELECT x.a, y.b FROM t x JOIN t y ON x.a = y.b"),
        "PROJECT [a, b]\n  JOIN inner [a = b]\n    SCAN t []\n    SCAN t []\n",
    )


def test_three_tables_nest_to_the_left() raises:
    assert_equal(
        _plan("SELECT t.a FROM t JOIN u ON t.a = u.k JOIN t s ON s.a = u.k"),
        (
            "PROJECT [a]\n"
            "  JOIN inner [k = a]\n"
            "    JOIN inner [a = k]\n"
            "      SCAN t []\n"
            "      SCAN u []\n"
            "    SCAN t []\n"
        ),
    )


def test_a_join_in_parentheses_is_the_join_inside_them() raises:
    assert_equal(
        _plan("SELECT a FROM (t JOIN u ON t.a = u.k)"),
        _plan("SELECT a FROM t JOIN u ON t.a = u.k"),
    )


def test_a_table_function_is_a_source_that_names_no_table() raises:
    assert_equal(
        _plan("SELECT * FROM range(5)"),
        "PROJECT [range]\n  range(5) [range]\n",
        "the column is called after the function, the way DuckDB calls it",
    )


def test_a_table_function_carries_the_arguments_it_was_given() raises:
    assert_equal(
        _plan("SELECT * FROM generate_series(1, 10, 2)"),
        (
            "PROJECT [generate_series]\n"
            "  generate_series(1, 10, 2) [generate_series]\n"
        ),
        "a start, a stop and a step",
    )


def test_a_lateral_table_function_is_refused_by_name() raises:
    with assert_raises(contains="LATERAL table function"):
        _ = _plan("SELECT a FROM t, LATERAL range(t.a)")


def test_a_subquery_in_a_from_is_the_plan_it_lowers_to() raises:
    # Nothing wraps it. A derived table is a statement whose output becomes a
    # source, so the node under the outer projection is the inner projection
    # itself and not a node that stands for one.
    assert_equal(
        _plan("SELECT x FROM (SELECT a AS x FROM t) v"),
        "PROJECT [x]\n  PROJECT [a as x]\n    SCAN t []\n",
    )


def test_a_derived_table_may_be_written_without_a_name() raises:
    # The name is only what a column may be qualified by, so leaving it out
    # costs the qualifier and nothing else.
    assert_equal(
        _plan("SELECT a FROM (SELECT a FROM t)"),
        "PROJECT [a]\n  PROJECT [a]\n    SCAN t []\n",
    )


def test_a_column_of_a_derived_table_may_be_qualified_by_its_name() raises:
    # `v` is not a relation, so the qualifier does not survive into the plan.
    # What it does is say which columns are meant, and the plan then holds the
    # same bare name the unqualified spelling holds.
    assert_equal(
        _plan("SELECT v.x FROM (SELECT a AS x FROM t) v"),
        _plan("SELECT x FROM (SELECT a AS x FROM t) v"),
    )


def test_a_name_qualified_by_a_derived_table_has_to_be_one_it_produces() raises:
    # `b` is a column of `t` and the subquery did not hand it out, so the outer
    # query cannot read it. Without the check the name would lower to a bare
    # `b`, and binding would answer it from the scan below and return a column
    # the query has no way to name.
    with assert_raises(contains="'v' produces no column called 'b'"):
        _ = _plan("SELECT v.b FROM (SELECT a AS x FROM t) v")


def test_a_star_over_a_derived_table_is_what_the_subquery_produces() raises:
    assert_equal(
        _plan("SELECT * FROM (SELECT b, a AS x FROM t) v"),
        "PROJECT [b, x]\n  PROJECT [b, a as x]\n    SCAN t []\n",
    )


def test_a_derived_table_keeps_its_own_order_by_and_limit() raises:
    assert_equal(
        _plan("SELECT x FROM (SELECT a AS x FROM t ORDER BY x LIMIT 3) v"),
        (
            "PROJECT [x]\n"
            "  LIMIT 3\n"
            "    SORT [x asc nulls last]\n"
            "      PROJECT [a as x]\n"
            "        SCAN t []\n"
        ),
    )


def test_a_derived_table_may_aggregate_and_the_query_reads_it() raises:
    assert_equal(
        _plan(
            "SELECT g, total FROM"
            " (SELECT g, sum(a) AS total FROM t GROUP BY g) v"
        ),
        (
            "PROJECT [g, total]\n"
            "  PROJECT [g, __agg_0 as total]\n"
            "    AGGREGATE [g] -> [sum(a)]\n"
            "      SCAN t []\n"
        ),
    )


def test_a_derived_table_joins_on_the_columns_it_produces() raises:
    # The key pair is found by name, because a column of a derived table has no
    # relation to be found by, and the names are what the source carries
    # alongside its node for exactly this.
    assert_equal(
        _plan("SELECT a FROM t JOIN (SELECT k FROM u) v ON t.a = v.k"),
        (
            "PROJECT [a]\n"
            "  JOIN inner [a = k]\n"
            "    SCAN t []\n"
            "    PROJECT [k]\n"
            "      SCAN u []\n"
        ),
    )


def test_a_name_two_sources_both_have_is_refused_not_answered() raises:
    # `b` is a column of `t` and the subquery hands out a `b` of its own, and
    # the qualifier cannot save it because a derived column is unpinned. So the
    # answer is binding's ambiguity, which is the refusal this is here to pin:
    # the alternative is picking one of the two and being right half the time.
    with assert_raises(contains="more than one column"):
        _ = _plan("SELECT v.b FROM t, (SELECT b FROM u) v")


def test_the_scans_inside_a_derived_table_number_with_the_rest() raises:
    # Two tables, one of them inside the subquery, and the qualified names on
    # the outside have to reach the right one. A relation number that counted
    # from zero inside the subquery would make `t.a` and the inner scan's first
    # column the same relation.
    assert_equal(
        _plan("SELECT t.a, v.z FROM t, (SELECT z FROM u) v"),
        (
            "PROJECT [a, z]\n"
            "  JOIN cross []\n"
            "    SCAN t []\n"
            "    PROJECT [z]\n"
            "      SCAN u []\n"
        ),
    )


def test_a_lateral_subquery_is_refused_by_name() raises:
    with assert_raises(contains="LATERAL subquery"):
        _ = _plan("SELECT a FROM t, LATERAL (SELECT b FROM u WHERE b = t.a) v")


def test_the_column_aliases_on_a_derived_table_are_refused_by_name() raises:
    with assert_raises(contains="column aliases on a subquery"):
        _ = _plan("SELECT n FROM (SELECT a FROM t) v(n)")


def test_a_cte_is_the_plan_its_statement_lowers_to() raises:
    # A CTE reference is a derived table under a name that was written further
    # up the query, so the plan is the one the subquery spelling gives.
    assert_equal(
        _plan("WITH v AS (SELECT a AS x FROM t) SELECT x FROM v"),
        _plan("SELECT x FROM (SELECT a AS x FROM t) v"),
    )


def test_a_cte_may_be_given_a_name_where_it_is_read() raises:
    assert_equal(
        _plan("WITH v AS (SELECT a AS x FROM t) SELECT w.x FROM v AS w"),
        "PROJECT [x]\n  PROJECT [a as x]\n    SCAN t []\n",
    )


def test_a_cte_read_twice_is_lowered_twice() raises:
    # The alternative is one node with two parents, which would make the plan a
    # graph, and every pass over it walks a tree.
    assert_equal(
        _plan(
            "WITH v AS (SELECT k FROM u) SELECT k FROM v UNION ALL SELECT k"
            " FROM v"
        ),
        (
            "UNION all\n"
            "  PROJECT [k]\n"
            "    PROJECT [k]\n"
            "      SCAN u []\n"
            "  PROJECT [k]\n"
            "    PROJECT [k]\n"
            "      SCAN u []\n"
        ),
    )


def test_a_cte_may_read_the_one_bound_before_it() raises:
    assert_equal(
        _plan(
            "WITH x AS (SELECT a FROM t), y AS (SELECT a FROM x) SELECT a"
            " FROM y"
        ),
        "PROJECT [a]\n  PROJECT [a]\n    PROJECT [a]\n      SCAN t []\n",
    )


def test_a_cte_reading_a_later_one_is_a_missing_table() raises:
    # Not a forward reference. The names bind in the order they were written
    # and an entry sees only the ones before it, which is what makes this the
    # catalog's answer rather than a plan that reads a name from further down.
    with assert_raises(contains="Table with name later does not exist"):
        _ = _plan(
            "WITH first AS (SELECT a FROM later),"
            " later AS (SELECT a FROM t) SELECT a FROM first"
        )


def test_a_cte_hides_a_table_of_the_same_name() raises:
    # The CTE names are looked in before the catalog, so `u` here is the one
    # the WITH bound. The star is what says so: the registered `u` has three
    # columns and none of them is `g`.
    assert_equal(
        _plan("WITH u AS (SELECT g FROM t) SELECT * FROM u"),
        "PROJECT [g]\n  PROJECT [g]\n    SCAN t []\n",
    )


def test_a_cte_that_names_a_table_it_hides_is_circular() raises:
    # The inner `t` is the CTE rather than the frame, because the name is bound
    # by the time its own body is read, and a name that stands for itself with
    # no anchor under it is the recursion the word RECURSIVE asks for. DuckDB
    # refuses it in the same words.
    with assert_raises(contains="Circular reference to CTE"):
        _ = _plan("WITH t AS (SELECT b FROM t) SELECT b FROM t")


def test_the_column_aliases_on_a_cte_rename_by_prefix() raises:
    # A name past the end of the column list is dropped and a column past the
    # end of the name list keeps the name it had, which is DuckDB's rule and
    # not the count match a derived table's alias list asks for.
    assert_equal(
        _plan("WITH v(p) AS (SELECT a, b FROM t) SELECT p, b FROM v"),
        (
            "PROJECT [p, b]\n"
            "  PROJECT [a as p, b]\n"
            "    PROJECT [a, b]\n"
            "      SCAN t []\n"
        ),
    )


def test_a_name_a_cte_does_not_produce_is_not_reachable_through_it() raises:
    with assert_raises(contains="'v' produces no column called 'b'"):
        _ = _plan("WITH v AS (SELECT a AS x FROM t) SELECT v.b FROM v")


def test_a_with_inside_a_subquery_shadows_the_one_outside_it() raises:
    # Both are called `v` and the inner one is bound later, and a lookup runs
    # from the end, so the subquery reads its own.
    assert_equal(
        _plan(
            "WITH v AS (SELECT a FROM t)"
            " SELECT z FROM (WITH v AS (SELECT z FROM u) SELECT z FROM v) w"
        ),
        "PROJECT [z]\n  PROJECT [z]\n    PROJECT [z]\n      SCAN u []\n",
    )


def test_a_recursive_cte_is_refused_by_name() raises:
    with assert_raises(contains="recursive CTE"):
        _ = _plan(
            "WITH RECURSIVE n(i) AS"
            " (SELECT 1 AS i UNION ALL SELECT i + 1 FROM n WHERE i < 5)"
            " SELECT i FROM n"
        )


def test_a_condition_may_reach_only_the_two_tables_it_joins() raises:
    # The comma binds looser than the JOIN word, so `u` and `t s` are the join
    # and `t` is beside it, and a condition naming `t` there is reaching out of
    # the join it was written on.
    with assert_raises(contains="a table this join does not read"):
        _ = _plan("SELECT a FROM t, u JOIN t s ON t.a = s.a")


def test_the_joins_with_no_node_yet_each_say_which_one() raises:
    with assert_raises(contains="USING join"):
        _ = _plan("SELECT a FROM t JOIN u USING (b)")
    with assert_raises(contains="NATURAL join"):
        _ = _plan("SELECT a FROM t NATURAL JOIN u")
    with assert_raises(contains="POSITIONAL join"):
        _ = _plan("SELECT a FROM t POSITIONAL JOIN u")
    with assert_raises(contains="ASOF join"):
        _ = _plan("SELECT a FROM t ASOF JOIN u ON t.a = u.k")
    with assert_raises(contains="SEMI or ANTI join"):
        _ = _plan("SELECT a FROM t SEMI JOIN u ON t.a = u.k")
    with assert_raises(contains="alias on a table function"):
        _ = _plan("SELECT a FROM range(10) r")
    with assert_raises(contains="parenthesised table reference"):
        _ = _plan("SELECT a FROM (t JOIN u ON t.a = u.k) v")


def test_a_between_is_the_two_comparisons_it_stands_for() raises:
    # The operand is lowered once and both comparisons read it, so the arena
    # holds one subtree with two parents. The printer walks the tree and so
    # writes it twice, which is the only place the sharing is not visible.
    assert_equal(
        _plan("SELECT a FROM t WHERE b BETWEEN 1 AND 2"),
        "PROJECT [a]\n  FILTER and(b >= 1, b <= 2)\n    SCAN t []\n",
    )


def test_a_not_between_negates_the_test_rather_than_turning_it_around() raises:
    # `b < 1 OR b > 2` is the same answer only when nothing is null. With a null
    # bound the positive test is false and its negation is true, where the two
    # comparisons turned around answer null.
    assert_equal(
        _plan("SELECT a FROM t WHERE b NOT BETWEEN 1 AND 2"),
        "PROJECT [a]\n  FILTER not(and(b >= 1, b <= 2))\n    SCAN t []\n",
    )


def test_a_between_may_be_written_anywhere_an_expression_may() raises:
    assert_equal(
        _plan("SELECT b BETWEEN 1 AND 2 AS ok FROM t"),
        "PROJECT [and(b >= 1, b <= 2) as ok]\n  SCAN t []\n",
    )


def test_a_between_over_a_fold_is_a_having_like_any_other() raises:
    # The bounds are lowered against the same scope the operand is, so a fold
    # inside one is found by the walk and computed by the aggregate.
    assert_equal(
        _plan("SELECT g FROM t GROUP BY g HAVING sum(a) BETWEEN 1 AND 2"),
        (
            "PROJECT [g]\n"
            "  FILTER and(__agg_0 >= 1, __agg_0 <= 2)\n"
            "    AGGREGATE [g] -> [sum(a)]\n"
            "      SCAN t []\n"
        ),
    )


def test_an_in_is_one_equality_per_candidate() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE b IN (1, 2, 3)"),
        "PROJECT [a]\n  FILTER or(or(b == 1, b == 2), b == 3)\n    SCAN t []\n",
    )


def test_an_in_of_one_is_one_comparison() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE g IN ('x')"),
        "PROJECT [a]\n  FILTER g == x\n    SCAN t []\n",
    )


def test_a_not_in_negates_the_chain_rather_than_inverting_it() raises:
    # The one that matters. `b <> 1 AND b <> 2` answers true for a row that a
    # null in the list should have made null, which is the classic wrong answer
    # for NOT IN and is silent.
    assert_equal(
        _plan("SELECT a FROM t WHERE b NOT IN (1, 2)"),
        "PROJECT [a]\n  FILTER not(or(b == 1, b == 2))\n    SCAN t []\n",
    )


def test_a_window_is_a_node_of_its_own_under_the_projection() raises:
    # The node comes out wider than it went in, so the projection above it is
    # what narrows the answer back to the two columns that were asked for.
    assert_equal(
        _plan("SELECT a, sum(b) OVER () FROM t"),
        (
            "PROJECT [a, __win_0 as __expr_1]\n"
            "  WINDOW [sum(b) over () as __win_0]\n"
            "    SCAN t []\n"
        ),
    )


def test_a_window_partitions_by_what_the_over_was_given() raises:
    assert_equal(
        _plan("SELECT a, sum(b) OVER (PARTITION BY g) FROM t"),
        (
            "PROJECT [a, __win_0 as __expr_1]\n"
            "  WINDOW [sum(b) over (partition g) as __win_0]\n"
            "    SCAN t []\n"
        ),
    )


def test_two_windows_over_the_same_keys_are_one_node() raises:
    # One pass over the rows answers both, and the partitioning is what decides
    # that, so the windows are grouped by the keys they were written with.
    assert_equal(
        _plan(
            "SELECT sum(b) OVER (PARTITION BY g), count(*) OVER (PARTITION BY"
            " g) FROM t"
        ),
        (
            "PROJECT [__win_0 as __expr_0, __win_1 as __expr_1]\n"
            "  WINDOW [sum(b) over (partition g) as __win_0, count(1) over"
            " (partition g) as __win_1]\n"
            "    SCAN t []\n"
        ),
    )


def test_two_windows_over_different_keys_are_a_node_each() raises:
    # Stacked, and the order they stack in does not matter, because a window
    # adds columns rather than replacing them and the projection above reads
    # each one by name.
    assert_equal(
        _plan("SELECT sum(b) OVER (PARTITION BY g), sum(b) OVER () FROM t"),
        (
            "PROJECT [__win_0 as __expr_0, __win_1 as __expr_1]\n"
            "  WINDOW [sum(b) over () as __win_1]\n"
            "    WINDOW [sum(b) over (partition g) as __win_0]\n"
            "      SCAN t []\n"
        ),
    )


def test_a_window_sits_above_the_aggregate_whose_answer_it_reads() raises:
    # `sum(sum(b))` is a fold of a fold, and the inner one is the GROUP BY's, so
    # the window has to be the node above it rather than beside it.
    assert_equal(
        _plan("SELECT g, sum(b), sum(sum(b)) OVER () FROM t GROUP BY g"),
        (
            "PROJECT [g, __agg_0 as __expr_1, __win_0 as __expr_2]\n"
            "  WINDOW [sum(__agg_1) over () as __win_0]\n"
            "    AGGREGATE [g] -> [sum(b), sum(b)]\n"
            "      SCAN t []\n"
        ),
    )


def test_a_window_is_not_what_makes_a_query_aggregate() raises:
    # A call to `sum` with an OVER on it is not a fold over the whole query, so
    # the plain column beside it is not an error and no aggregate is built. The
    # count is the one that would have gone wrong quietly: without this the
    # query would have come back with one row.
    assert_equal(
        _plan("SELECT a, count(*) OVER () FROM t"),
        (
            "PROJECT [a, __win_0 as __expr_1]\n"
            "  WINDOW [count(1) over () as __win_0]\n"
            "    SCAN t []\n"
        ),
    )


def test_a_qualify_is_a_filter_above_the_window_it_reads() raises:
    # Which is the whole reason QUALIFY exists. A WHERE runs under the window
    # and so cannot see what the window computed.
    assert_equal(
        _plan("SELECT a FROM t QUALIFY sum(b) OVER (PARTITION BY g) > 1"),
        (
            "PROJECT [a]\n"
            "  FILTER __win_0 > 1\n"
            "    WINDOW [sum(b) over (partition g) as __win_0]\n"
            "      SCAN t []\n"
        ),
    )


def test_a_where_still_runs_under_the_window() raises:
    assert_equal(
        _plan("SELECT sum(b) OVER () FROM t WHERE a > 1"),
        (
            "PROJECT [__win_0 as __expr_0]\n"
            "  WINDOW [sum(b) over () as __win_0]\n"
            "    FILTER a > 1\n"
            "      SCAN t []\n"
        ),
    )


def test_a_qualify_with_no_window_in_it_is_a_where_written_wrong() raises:
    with assert_raises(contains="no window function in it"):
        _ = _plan("SELECT a FROM t QUALIFY b > 1")


def test_the_windows_with_no_operator_yet_each_say_which_one() raises:
    with assert_raises(contains="OVER an ORDER BY"):
        _ = _plan("SELECT sum(b) OVER (ORDER BY a) FROM t")
    with assert_raises(contains="OVER a frame"):
        _ = _plan(
            "SELECT sum(b) OVER (ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)"
            " FROM t"
        )
    with assert_raises(contains="WINDOW clause"):
        _ = _plan("SELECT sum(b) OVER w FROM t WINDOW w AS (PARTITION BY g)")
    with assert_raises(contains="rather than a fold"):
        _ = _plan("SELECT row_number() OVER () FROM t")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
