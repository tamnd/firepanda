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
from firepanda.dtype.temporal import TimeUnit
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


def _dated() -> Schema:
    """A day and a clock reading, for the queries that read a calendar field.

    A third table rather than two more columns on `t`, because ten tests in here
    name every column of `t` in the text they expect and a query about a date
    has nothing to do with any of them.

    Returns:
        The schema.
    """
    var out = Schema()
    out.append(Field("d", LogicalType.DATE32, True))
    out.append(Field("ts", LogicalType.timestamp(TimeUnit.MICRO), True))
    out.append(Field("n", LogicalType.INT64, False))
    return out^


def _catalog() raises -> Catalog:
    """A session holding a frame called `t`, one called `u` and one called `w`.

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
    var third = DataFrame()
    third.schema = _dated()
    catalog.register("w", third^)
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


def test_a_qualified_star_keeps_one_side_of_a_join() raises:
    # Which relation a column came from is the number the star already carried,
    # so keeping one side is a filter over that number rather than anything new.
    assert_equal(
        _plan("SELECT t.* FROM t JOIN u ON t.a = u.k"),
        (
            "PROJECT [a, b, g, f]\n"
            "  JOIN inner [a = k]\n"
            "    SCAN t []\n"
            "    SCAN u []\n"
        ),
    )
    assert_equal(
        _plan("SELECT u.* FROM t JOIN u ON t.a = u.k"),
        (
            "PROJECT [b, k, z]\n"
            "  JOIN inner [a = k]\n"
            "    SCAN t []\n"
            "    SCAN u []\n"
        ),
    )


def test_a_qualified_star_may_name_an_alias() raises:
    assert_equal(
        _plan("SELECT l.* FROM t AS l"), "PROJECT [a, b, g, f]\n  SCAN t []\n"
    )


def test_a_qualified_star_naming_nothing_in_the_from_is_refused() raises:
    with assert_raises(contains="nothing in this query is called 'nosuch'"):
        _ = _plan("SELECT nosuch.* FROM t")


def test_a_qualified_star_over_a_subquery_is_refused() raises:
    # A derived table is not a relation, so its columns come through with no
    # number on them and there is nothing to keep them apart by.
    with assert_raises(contains="is not a relation"):
        _ = _plan("SELECT v.* FROM (SELECT a FROM t) v")


def test_an_exclude_drops_the_columns_it_names() raises:
    assert_equal(
        _plan("SELECT * EXCLUDE (b, f) FROM t"),
        "PROJECT [a, g]\n  SCAN t []\n",
    )


def test_an_exclude_naming_a_column_twice_over_a_join_drops_both() raises:
    # DuckDB's rule, and the reason a bare modifier name is not a column
    # reference: a name two sources both have is not ambiguous here, it is two
    # columns and the modifier applies to each of them.
    assert_equal(
        _plan("SELECT * EXCLUDE (b) FROM t JOIN u ON t.a = u.k"),
        (
            "PROJECT [a, g, f, k, z]\n"
            "  JOIN inner [a = k]\n"
            "    SCAN t []\n"
            "    SCAN u []\n"
        ),
    )


def test_a_replace_stands_an_expression_in_for_a_column() raises:
    # The column keeps its place and its name and the expression is what the
    # projection computes there.
    assert_equal(
        _plan("SELECT * REPLACE (a + 1 AS a) FROM t"),
        "PROJECT [a + 1 as a, b, g, f]\n  SCAN t []\n",
    )


def test_a_rename_changes_what_the_output_calls_a_column() raises:
    assert_equal(
        _plan("SELECT * RENAME (a AS z) FROM t"),
        "PROJECT [a as z, b, g, f]\n  SCAN t []\n",
    )


def test_the_three_modifiers_apply_in_the_order_they_are_written() raises:
    assert_equal(
        _plan(
            "SELECT * EXCLUDE (g) REPLACE (b * 2 AS b) RENAME (a AS z) FROM t"
        ),
        "PROJECT [a as z, b * 2 as b, f]\n  SCAN t []\n",
    )


def test_a_modifier_naming_no_column_is_refused_the_way_duckdb_does() raises:
    with assert_raises(contains='Column "zz" in EXCLUDE list not found'):
        _ = _plan("SELECT * EXCLUDE (zz) FROM t")
    with assert_raises(contains='Column "zz" in REPLACE list not found'):
        _ = _plan("SELECT * REPLACE (1 AS zz) FROM t")
    # A RENAME naming nothing is not an error, which is DuckDB's again and is
    # the one modifier that lets a typo through.
    assert_equal(
        _plan("SELECT * RENAME (zz AS q) FROM t"),
        "PROJECT [a, b, g, f]\n  SCAN t []\n",
    )


def test_two_modifiers_naming_the_same_column_are_refused() raises:
    with assert_raises(contains="cannot occur in both EXCLUDE and REPLACE"):
        _ = _plan("SELECT * EXCLUDE (a) REPLACE (1 AS a) FROM t")
    with assert_raises(contains='Duplicate entry "a" in EXCLUDE list'):
        _ = _plan("SELECT * EXCLUDE (a, a) FROM t")


def test_a_star_that_excludes_everything_leaves_no_select_list() raises:
    with assert_raises(contains="SELECT list is empty after resolving"):
        _ = _plan("SELECT * EXCLUDE (a, b, g, f) FROM t")


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


def test_a_group_by_all_groups_by_the_items_that_do_not_fold() raises:
    # The same plan the query with the key written out gives, because that is
    # what GROUP BY ALL means rather than being a node of its own.
    assert_equal(
        _plan("SELECT g, sum(a) FROM t GROUP BY ALL"),
        _plan("SELECT g, sum(a) FROM t GROUP BY g"),
    )


def test_a_group_by_all_over_a_star_groups_by_every_column() raises:
    assert_equal(
        _plan("SELECT * FROM t GROUP BY ALL"),
        _plan("SELECT * FROM t GROUP BY a, b, g, f"),
    )


def test_a_group_by_all_with_no_key_left_is_one_group() raises:
    # Every item folds, so there is nothing to group by and the answer is the
    # one row a bare aggregate gives.
    assert_equal(
        _plan("SELECT sum(a) FROM t GROUP BY ALL"),
        _plan("SELECT sum(a) FROM t"),
    )


def test_a_group_by_all_with_no_fold_is_a_distinct() raises:
    assert_equal(
        _plan("SELECT g FROM t GROUP BY ALL"),
        _plan("SELECT g FROM t GROUP BY g"),
    )


def test_a_group_by_may_name_an_alias_the_select_list_wrote() raises:
    # The expression is in the aggregate once, as the key, and the projection
    # reads the column back by the name the query gave it. Computing it again
    # up there is not a slower way of getting the same answer: `a` is not a
    # column the aggregate hands out, so there would be nothing to compute it
    # over.
    assert_equal(
        _plan("SELECT a + 1 AS k, count(*) FROM t GROUP BY k"),
        (
            "PROJECT [k, __agg_0 as __expr_1]\n"
            "  AGGREGATE [a + 1] -> [count(1)]\n"
            "    SCAN t []\n"
        ),
    )


def test_an_alias_of_a_plain_column_as_a_key_is_the_column() raises:
    # The same plan as the query that wrote the column itself, with the
    # renaming where it always was, in the projection. The key keeps the
    # column's own name because a physical group by hands the key field through
    # as it found it, so a key named anything else is a refusal two stages
    # down rather than a rename.
    assert_equal(
        _plan("SELECT g AS k, count(*) FROM t GROUP BY k"),
        (
            "PROJECT [g as k, __agg_0 as __expr_1]\n"
            "  AGGREGATE [g] -> [count(1)]\n"
            "    SCAN t []\n"
        ),
    )


def test_a_table_column_wins_over_an_alias_of_the_same_name() raises:
    # `g` is a column of `t` and also the name this select list gives to `b`,
    # and the table is asked first, so the keys are `g` and `b` and not `b`
    # twice. Nothing that bound before this rule existed binds to anything else
    # because of it, which is what writing the qualified name beside it says:
    # `t.g` can only be the column and the two plans are the same plan.
    var out = _plan("SELECT b AS g, count(*) FROM t GROUP BY g, b")
    assert_equal(out, _plan("SELECT b AS g, count(*) FROM t GROUP BY t.g, b"))
    assert_true("AGGREGATE [g, b]" in out, "the column and not the alias")


def test_the_same_alias_named_twice_is_one_key() raises:
    assert_equal(
        _plan("SELECT a + 1 AS k, count(*) FROM t GROUP BY k, k"),
        _plan("SELECT a + 1 AS k, count(*) FROM t GROUP BY k"),
    )


def test_a_group_by_that_names_the_alias_of_a_fold_says_so() raises:
    with assert_raises(contains="computed over the groups"):
        _ = _plan("SELECT g, count(*) AS c FROM t GROUP BY c")


def test_two_items_with_one_alias_do_not_say_which_is_the_key() raises:
    with assert_raises(contains="does not say which"):
        _ = _plan("SELECT a AS k, b AS k, count(*) FROM t GROUP BY k")


def test_a_group_by_name_that_is_neither_still_says_it_is_neither() raises:
    # The refusal a name that is nothing was always given, unchanged, because
    # the select list is searched second and finding nothing there leaves the
    # name to be lowered as it was written.
    with assert_raises(contains="no column named 'nope'"):
        _ = _plan("SELECT g, count(*) FROM t GROUP BY nope")


def test_a_group_by_may_write_the_expression_the_select_list_writes() raises:
    # The same aggregate the query naming the key by its alias builds, and the
    # projection reads the key's column back rather than computing `a + 1` a
    # second time over an `a` the aggregate no longer hands out.
    assert_equal(
        _plan("SELECT a + 1, count(*) FROM t GROUP BY a + 1"),
        (
            "PROJECT [__expr_0, __agg_0 as __expr_1]\n"
            "  AGGREGATE [a + 1] -> [count(1)]\n"
            "    SCAN t []\n"
        ),
    )


def test_the_key_written_out_twice_is_read_back_twice() raises:
    # One key and one aggregate, with both items reading the one column, which
    # is the same thing the aggregate list does with a fold written twice.
    assert_equal(
        _plan("SELECT a + 1, a + 1, count(*) FROM t GROUP BY a + 1"),
        (
            "PROJECT [__expr_0, __expr_0 as __expr_1, __agg_0 as __expr_2]\n"
            "  AGGREGATE [a + 1] -> [count(1)]\n"
            "    SCAN t []\n"
        ),
    )


def test_a_key_written_inside_a_larger_item_is_read_where_it_sits() raises:
    # The item is not the key, it holds the key, so what the projection computes
    # is the rest of the item over the column the key came out in.
    assert_equal(
        _plan("SELECT (a + 1) * 2 AS m, count(*) FROM t GROUP BY a + 1"),
        (
            "PROJECT [__expr_0 * 2 as m, __agg_0 as __expr_1]\n"
            "  AGGREGATE [a + 1] -> [count(1)]\n"
            "    SCAN t []\n"
        ),
    )


def test_a_having_that_writes_the_key_out_reads_the_key() raises:
    assert_equal(
        _plan("SELECT a + 1, count(*) FROM t GROUP BY a + 1 HAVING a + 1 > 2"),
        (
            "PROJECT [__expr_0, __agg_0 as __expr_1]\n"
            "  FILTER __expr_0 > 2\n"
            "    AGGREGATE [a + 1] -> [count(1)]\n"
            "      SCAN t []\n"
        ),
    )


def test_an_order_by_that_writes_the_key_out_sorts_on_the_item() raises:
    # The sort reads the projection's own output column rather than the key's,
    # because it sits above the projection, which is also why nothing has to be
    # widened underneath it.
    assert_equal(
        _plan(
            "SELECT a + 1 AS m, count(*) FROM t GROUP BY a + 1 ORDER BY a + 1"
        ),
        (
            "SORT [m asc nulls last]\n"
            "  PROJECT [__expr_0 as m, __agg_0 as __expr_1]\n"
            "    AGGREGATE [a + 1] -> [count(1)]\n"
            "      SCAN t []\n"
        ),
    )


def test_a_group_by_of_a_date_trunc_is_the_shape_q42_writes() raises:
    # ClickBench q42, with the same expression in the select list, the GROUP BY
    # and the ORDER BY, which is three copies of one column.
    assert_equal(
        _plan(
            "SELECT date_trunc('minute', ts) AS m, count(*) AS c FROM w GROUP"
            " BY date_trunc('minute', ts) ORDER BY date_trunc('minute', ts)"
        ),
        (
            "SORT [m asc nulls last]\n"
            "  PROJECT [__expr_0 as m, __agg_0 as c]\n"
            "    AGGREGATE [date_trunc(minute, ts)] -> [count(1)]\n"
            "      SCAN w []\n"
        ),
    )


def test_a_group_by_of_an_extract_is_the_shape_q18_writes() raises:
    assert_equal(
        _plan(
            "SELECT extract(minute FROM ts) AS m, count(*) FROM w GROUP BY"
            " extract(minute FROM ts)"
        ),
        (
            "PROJECT [__expr_0 as m, __agg_0 as __expr_1]\n"
            "  AGGREGATE [date_part(minute, ts)] -> [count(1)]\n"
            "    SCAN w []\n"
        ),
    )


def test_an_item_that_is_not_the_key_is_refused_as_it_always_was() raises:
    # The rule is about an expression the query wrote twice and nothing else,
    # so a column that is neither a key nor folded is still a column the
    # aggregate does not hand out.
    with assert_raises(contains="no column named 'b'"):
        _ = _plan("SELECT b, count(*) FROM t GROUP BY a + 1")


def test_an_order_by_of_a_position_sorts_on_that_column() raises:
    # The number counts the columns the query returns rather than being the
    # number itself, which is what DuckDB reads it as.
    assert_equal(
        _plan("SELECT a, b FROM t ORDER BY 2"),
        "SORT [b asc nulls last]\n  PROJECT [a, b]\n    SCAN t []\n",
    )


def test_an_order_by_of_a_position_counts_a_star_after_it_expands() raises:
    # The names come off the plan rather than out of the select list, which is
    # where a star that has already become four columns is four columns to
    # count through.
    assert_equal(
        _plan("SELECT * FROM t ORDER BY 3"),
        "SORT [g asc nulls last]\n  PROJECT [a, b, g, f]\n    SCAN t []\n",
    )


def test_an_order_by_of_a_position_takes_the_direction_written_on_it() raises:
    assert_equal(
        _plan("SELECT a, b FROM t ORDER BY 1 DESC"),
        "SORT [a desc]\n  PROJECT [a, b]\n    SCAN t []\n",
    )


def test_an_order_by_of_a_position_past_the_end_says_how_many() raises:
    with assert_raises(contains="positions this query has are 1 to 2"):
        _ = _plan("SELECT a, b FROM t ORDER BY 3")


def test_an_order_by_of_position_zero_is_past_the_end_too() raises:
    # The count starts at one, so a zero is not the first column and is not a
    # constant either. DuckDB refuses it the same way.
    with assert_raises(contains="an ORDER BY of position 0"):
        _ = _plan("SELECT a FROM t ORDER BY 0")


def test_an_order_by_of_arithmetic_is_a_constant_and_not_a_position() raises:
    # Only a number written on its own is a position. `1 + 1` is the number two
    # in DuckDB as well, so it sorts every row on the same value and leaves
    # them where they were.
    var out = _plan("SELECT a, b FROM t ORDER BY 1 + 1")
    assert_true(out.startswith("SORT ["), "the sort node is still built")
    assert_true(
        not out.startswith("SORT [b"),
        "on a constant and not on the second column",
    )


def test_a_group_by_of_a_position_is_the_item_it_counts_to() raises:
    # The same plan the query writing the expression out in both places gets,
    # and the same one the alias route gets. Three spellings, one aggregate.
    assert_equal(
        _plan("SELECT a + 1, count(*) FROM t GROUP BY 1"),
        (
            "PROJECT [__expr_0, __agg_0 as __expr_1]\n"
            "  AGGREGATE [a + 1] -> [count(1)]\n"
            "    SCAN t []\n"
        ),
    )


def test_a_group_by_of_a_position_keeps_the_alias_the_item_wrote() raises:
    assert_equal(
        _plan("SELECT a + 1 AS k, count(*) FROM t GROUP BY 1"),
        _plan("SELECT a + 1 AS k, count(*) FROM t GROUP BY k"),
    )


def test_a_group_by_of_a_position_on_a_plain_column_is_the_column() raises:
    assert_equal(
        _plan("SELECT g, count(*) FROM t GROUP BY 1"),
        _plan("SELECT g, count(*) FROM t GROUP BY g"),
    )


def test_a_group_by_of_a_position_past_the_end_says_how_many() raises:
    with assert_raises(contains="the select list has are 1 to 2"):
        _ = _plan("SELECT g, count(*) FROM t GROUP BY 4")


def test_a_group_by_of_a_position_that_lands_on_a_fold_says_so() raises:
    # The same refusal the alias of a fold gets, since a position is another
    # way of pointing at the same item.
    with assert_raises(contains="computed over the groups"):
        _ = _plan("SELECT g, count(*) FROM t GROUP BY 2")


def test_a_group_by_of_a_position_that_lands_on_a_star_says_so() raises:
    with assert_raises(contains="wrote a star there"):
        _ = _plan("SELECT *, count(*) FROM t GROUP BY 1")


def test_an_order_by_may_name_a_column_the_query_does_not_return() raises:
    # The projection under the sort is one column wider than the query asked
    # for and the one above it takes the query's own columns back.
    assert_equal(
        _plan("SELECT a FROM t ORDER BY b"),
        (
            "PROJECT [a]\n"
            "  SORT [b asc nulls last]\n"
            "    PROJECT [a, b]\n"
            "      SCAN t []\n"
        ),
    )


def test_an_order_by_may_name_a_column_the_query_renamed() raises:
    assert_equal(
        _plan("SELECT a AS z FROM t ORDER BY a"),
        (
            "PROJECT [z]\n"
            "  SORT [a asc nulls last]\n"
            "    PROJECT [a as z, a]\n"
            "      SCAN t []\n"
        ),
    )


def test_an_order_by_on_an_output_name_beats_the_column_under_it() raises:
    # `a` is what the second output is called, so that is what the sort reads
    # and the column the table has by that name is not in it at all.
    assert_equal(
        _plan("SELECT b AS z, a AS b FROM t ORDER BY b"),
        "SORT [b asc nulls last]\n  PROJECT [b as z, a as b]\n    SCAN t []\n",
    )


def test_an_order_by_adds_a_column_once_however_often_it_is_read() raises:
    assert_equal(
        _plan("SELECT a FROM t ORDER BY b, b DESC"),
        (
            "PROJECT [a]\n"
            "  SORT [b asc nulls last, b desc]\n"
            "    PROJECT [a, b]\n"
            "      SCAN t []\n"
        ),
    )


def test_an_order_by_may_sort_on_a_fold_the_query_returns() raises:
    # The fold is one slot in the aggregate whether it is written once or twice,
    # so the sort reads the column the select list already asked for. This shape
    # is most of ClickBench: group, count, and put the biggest count first.
    assert_equal(
        _plan("SELECT g, sum(a) FROM t GROUP BY g ORDER BY sum(a) DESC"),
        (
            "PROJECT [g, __expr_1]\n"
            "  SORT [__agg_0 desc]\n"
            "    PROJECT [g, __agg_0 as __expr_1, __agg_0]\n"
            "      AGGREGATE [g] -> [sum(a)]\n"
            "        SCAN t []\n"
        ),
    )


def test_an_order_by_may_sort_on_a_fold_the_query_does_not_return() raises:
    # A slot of its own, because nothing else asked for it. That is why the
    # ORDER BY is read before the aggregate is built rather than after: above
    # the node a column it does not compute cannot be added to it.
    assert_equal(
        _plan("SELECT g FROM t GROUP BY g ORDER BY sum(a) DESC"),
        (
            "PROJECT [g]\n"
            "  SORT [__agg_0 desc]\n"
            "    PROJECT [g, __agg_0]\n"
            "      AGGREGATE [g] -> [sum(a)]\n"
            "        SCAN t []\n"
        ),
    )


def test_an_order_by_may_sort_on_an_expression_over_a_fold() raises:
    assert_equal(
        _plan("SELECT g FROM t GROUP BY g ORDER BY sum(a) * 2 DESC"),
        (
            "PROJECT [g]\n"
            "  SORT [__agg_0 * 2 desc]\n"
            "    PROJECT [g, __agg_0]\n"
            "      AGGREGATE [g] -> [sum(a)]\n"
            "        SCAN t []\n"
        ),
    )


def test_an_order_by_on_a_folds_alias_reads_the_output_column() raises:
    # No fold is written in the ORDER BY here, only the name the select list
    # gave one, so this goes the way every other name does and sorts on the
    # column the projection produced. No widening and no second projection.
    assert_equal(
        _plan("SELECT g, count(*) AS c FROM t GROUP BY g ORDER BY c DESC"),
        (
            "SORT [c desc]\n"
            "  PROJECT [g, __agg_0 as c]\n"
            "    AGGREGATE [g] -> [count(1)]\n"
            "      SCAN t []\n"
        ),
    )


def test_two_folds_of_one_shape_across_the_order_by_are_one_slot() raises:
    assert_equal(
        _plan(
            "SELECT g, sum(a) AS x FROM t GROUP BY g HAVING sum(a) > 1 ORDER BY"
            " sum(a)"
        ),
        (
            "PROJECT [g, x]\n"
            "  SORT [__agg_0 asc nulls last]\n"
            "    PROJECT [g, __agg_0 as x, __agg_0]\n"
            "      FILTER __agg_0 > 1\n"
            "        AGGREGATE [g] -> [sum(a)]\n"
            "          SCAN t []\n"
        ),
    )


def test_a_fold_in_an_order_by_over_a_query_that_does_not_fold_is_refused() raises:
    # The same refusal a fold written anywhere else in such a query gets. DuckDB
    # refuses it too, for the column in the select list rather than for the
    # fold, and either way the query does not run.
    with assert_raises(contains="is an aggregate and this query has no GROUP"):
        _ = _plan("SELECT a FROM t ORDER BY count(*) DESC")


def test_a_distinct_on_sits_over_the_projection_it_was_written_in() raises:
    assert_equal(
        _plan("SELECT DISTINCT ON (a) a, b FROM t"),
        "DISTINCT [a]\n  PROJECT [a, b]\n    SCAN t []\n",
    )


def test_a_distinct_on_runs_over_the_order_that_chose_the_row() raises:
    # The sort is underneath, because an ORDER BY written with a DISTINCT ON
    # picks which row of each group survives as well as ordering the answer.
    assert_equal(
        _plan("SELECT DISTINCT ON (a) a, b FROM t ORDER BY b DESC LIMIT 3"),
        (
            "LIMIT 3\n"
            "  DISTINCT [a]\n"
            "    SORT [b desc]\n"
            "      PROJECT [a, b]\n"
            "        SCAN t []\n"
        ),
    )


def test_a_distinct_on_may_decide_on_a_column_it_does_not_return() raises:
    # The same widening an ORDER BY gets: the column is added below, read, and
    # taken back off above, so it never leaves the query.
    assert_equal(
        _plan("SELECT DISTINCT ON (a) b FROM t"),
        "PROJECT [b]\n  DISTINCT [a]\n    PROJECT [b, a]\n      SCAN t []\n",
    )


def test_a_distinct_on_in_an_arm_stays_inside_the_arm() raises:
    # Nothing can be written between an arm and the set operation over it, so
    # there is no order to run underneath and the arm applies its own.
    assert_equal(
        _plan(
            "SELECT DISTINCT ON (a) a, b FROM t UNION ALL SELECT a, b FROM t"
        ),
        (
            "UNION all\n"
            "  DISTINCT [a]\n"
            "    PROJECT [a, b]\n"
            "      SCAN t []\n"
            "  PROJECT [a, b]\n"
            "    SCAN t []\n"
        ),
    )


def test_an_order_by_over_a_distinct_may_only_name_what_it_returns() raises:
    with assert_raises(contains="there is no column named 'b'"):
        _ = _plan("SELECT DISTINCT a FROM t ORDER BY b")


def test_an_order_by_all_sorts_on_every_output_column() raises:
    assert_equal(
        _plan("SELECT a, b FROM t ORDER BY ALL"),
        _plan("SELECT a, b FROM t ORDER BY a, b"),
    )


def test_an_order_by_all_takes_the_direction_written_once() raises:
    assert_equal(
        _plan("SELECT a, b FROM t ORDER BY ALL DESC"),
        _plan("SELECT a, b FROM t ORDER BY a DESC, b DESC"),
    )


def test_an_order_by_all_reads_the_names_the_query_produced() raises:
    # The rename is what it sorts on, not the column under it, because the
    # names come off the node below the sort and that node is the projection.
    assert_equal(
        _plan("SELECT a AS z FROM t ORDER BY ALL"),
        "SORT [z asc nulls last]\n  PROJECT [a as z]\n    SCAN t []\n",
    )


def test_an_order_by_all_over_a_union_sorts_the_stack() raises:
    assert_equal(
        _plan("SELECT a FROM t UNION ALL SELECT b FROM u ORDER BY ALL"),
        _plan("SELECT a FROM t UNION ALL SELECT b FROM u ORDER BY a"),
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


def test_a_limit_with_no_order_by_does_not_sort() raises:
    # Worth pinning rather than assuming. A limit is the whole reason a sort
    # would be cheap to add here by accident, and sorting a hundred million
    # rows to hand back ten is the most expensive way there is to answer a
    # query that needs one pass and a counter.
    assert_equal(
        _plan("SELECT a FROM t LIMIT 10"),
        "LIMIT 10\n  PROJECT [a]\n    SCAN t []\n",
    )


def test_a_having_repeating_a_fold_reads_the_one_the_select_list_asked_for() raises:
    # Two calls that compute the same thing are one slot in the aggregate. This
    # shape is ClickBench q27 and q28, and without the sharing the node counts
    # every group twice and answers the same number both times.
    assert_equal(
        _plan("SELECT g, count(*) AS c FROM t GROUP BY g HAVING count(*) > 1"),
        (
            "PROJECT [g, __agg_0 as c]\n"
            "  FILTER __agg_0 > 1\n"
            "    AGGREGATE [g] -> [count(1)]\n"
            "      SCAN t []\n"
        ),
    )


def test_two_folds_of_one_shape_in_a_select_list_are_one_slot() raises:
    assert_equal(
        _plan("SELECT g, sum(a) AS x, sum(a) AS y FROM t GROUP BY g"),
        (
            "PROJECT [g, __agg_0 as x, __agg_0 as y]\n"
            "  AGGREGATE [g] -> [sum(a)]\n"
            "    SCAN t []\n"
        ),
    )


def test_two_folds_that_differ_are_two_slots() raises:
    # The guard on the sharing. `sum(a)` and `sum(b)` read the same kind over
    # different columns, and a shape that only looked at the kind would fold
    # them into one and answer the first one twice.
    assert_equal(
        _plan("SELECT g, sum(a), sum(b) FROM t GROUP BY g"),
        (
            "PROJECT [g, __agg_0 as __expr_1, __agg_1 as __expr_2]\n"
            "  AGGREGATE [g] -> [sum(a), sum(b)]\n"
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


def test_a_union_by_name_lines_the_arms_up_by_column_name() raises:
    # The left arm's order is the output's order, so the left needs nothing and
    # the right gets a projection that puts its two columns the other way round.
    assert_equal(
        _plan("SELECT a, b FROM t UNION ALL BY NAME SELECT b, a FROM t"),
        (
            "UNION all\n"
            "  PROJECT [a, b]\n"
            "    SCAN t []\n"
            "  PROJECT [a, b]\n"
            "    PROJECT [b, a]\n"
            "      SCAN t []\n"
        ),
    )


def test_a_union_by_name_fills_a_column_an_arm_lacks_with_null() raises:
    # Neither arm has the other's column, so both get a projection and the one
    # that is missing writes a null. The null is untyped and promotes to the
    # other arm's type, which is how the union settles on one.
    assert_equal(
        _plan("SELECT a FROM t UNION ALL BY NAME SELECT k FROM u"),
        (
            "UNION all\n"
            "  PROJECT [a, null as k]\n"
            "    PROJECT [a]\n"
            "      SCAN t []\n"
            "  PROJECT [null as a, k]\n"
            "    PROJECT [k]\n"
            "      SCAN u []\n"
        ),
    )


def test_a_union_by_name_over_arms_that_agree_is_the_positional_plan() raises:
    # An arm that already produces the output list in order gets no projection,
    # which is what makes the two spellings comparable rather than merely equal
    # in their answers.
    assert_equal(
        _plan("SELECT a, b FROM t UNION BY NAME SELECT a, b FROM t"),
        _plan("SELECT a, b FROM t UNION SELECT a, b FROM t"),
    )


def test_a_union_by_name_matches_a_name_whatever_its_case() raises:
    # The names line up folded and the left arm's spelling is what comes out,
    # which is the same rule every other name in this front end gets.
    assert_equal(
        _plan("SELECT a FROM t UNION ALL BY NAME SELECT b AS A FROM t"),
        (
            "UNION all\n"
            "  PROJECT [a]\n"
            "    SCAN t []\n"
            "  PROJECT [b as a]\n"
            "    SCAN t []\n"
        ),
    )


def test_a_union_by_name_refuses_an_arm_that_names_a_column_twice() raises:
    # Position is what tells two columns of the same name apart, and lining up
    # by name throws position away, so there is no answer to give. DuckDB's
    # wording, since it refuses the same thing.
    with assert_raises(contains="occurs multiple times"):
        _ = _plan(
            "SELECT a AS x, b AS x FROM t UNION BY NAME SELECT b AS x FROM t"
        )


def test_by_name_on_a_difference_is_refused() raises:
    # DuckDB's grammar only hangs the words off a UNION. This grammar hangs
    # them off EXCEPT as well, so the refusal is here instead of in the parser,
    # with the reason in it. INTERSECT has its own rule with no room for them
    # and stops one step earlier, in the parser, which is why only one of the
    # two is written here.
    with assert_raises(contains="does not take BY NAME"):
        _ = _plan("SELECT a FROM t EXCEPT BY NAME SELECT b FROM t")
    with assert_raises(contains="syntax error"):
        _ = _plan("SELECT a FROM t INTERSECT BY NAME SELECT b FROM t")


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
    with assert_raises(contains="one row by construction"):
        _ = _plan("SELECT a FROM t WHERE (SELECT b FROM u) > 1")
    with assert_raises(contains="written somewhere else"):
        _ = _plan(
            "SELECT g FROM t GROUP BY g HAVING sum(a) > ANY (SELECT k FROM u)"
        )
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
    with assert_raises(contains="left join on equalities"):
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


def test_the_column_aliases_on_a_derived_table_rename_its_output() raises:
    # The alias list is a projection over the statement's root, so the name the
    # subquery handed out is gone and the one the list wrote is what the outer
    # query reads.
    assert_equal(
        _plan("SELECT n FROM (SELECT a FROM t) v(n)"),
        "PROJECT [n]\n  PROJECT [a as n]\n    PROJECT [a]\n      SCAN t []\n",
    )


def test_a_column_a_derived_table_renamed_away_is_not_reachable() raises:
    # The list renamed `a` to `n`, and a name the source no longer produces is
    # a name the query cannot read, the same as any other one it never had.
    with assert_raises(contains="there is no column named 'a'"):
        _ = _plan("SELECT a FROM (SELECT a FROM t) v(n)")


def test_the_column_aliases_on_a_derived_table_are_a_prefix() raises:
    # A list shorter than the statement renames what it reaches and leaves the
    # rest alone, which is the rule a CTE's list gets as well.
    assert_equal(
        _plan("SELECT p, b FROM (SELECT a, b FROM t) v(p)"),
        (
            "PROJECT [p, b]\n"
            "  PROJECT [a as p, b]\n"
            "    PROJECT [a, b]\n"
            "      SCAN t []\n"
        ),
    )


def test_more_column_aliases_than_a_derived_table_produces_is_refused() raises:
    # This is the one place the derived table and the CTE disagree. A CTE drops
    # the names it has no column for and a derived table refuses the whole
    # query, and DuckDB's wording is what says so.
    with assert_raises(contains="has 2 columns available but 3 columns"):
        _ = _plan("SELECT p FROM (SELECT a, b FROM t) v(p, q, r)")


def test_a_column_alias_on_a_derived_table_may_be_qualified() raises:
    # The alias list runs before the name goes into reach, so `v.n` is the name
    # the list wrote and not the one the statement produced.
    assert_equal(
        _plan("SELECT v.n FROM (SELECT a FROM t) v(n)"),
        _plan("SELECT n FROM (SELECT a FROM t) v(n)"),
    )


def test_a_star_over_a_derived_table_reads_its_column_aliases() raises:
    assert_equal(
        _plan("SELECT * FROM (SELECT a, b FROM t) v(p, q)"),
        (
            "PROJECT [p, q]\n"
            "  PROJECT [a as p, b as q]\n"
            "    PROJECT [a, b]\n"
            "      SCAN t []\n"
        ),
    )


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


def test_a_using_join_is_the_join_the_on_spelling_builds() raises:
    # The node is the same node. Everything a USING does that an ON does not is
    # about what the query may write afterwards.
    assert_equal(
        _plan("SELECT a FROM t JOIN u USING (b)"),
        _plan("SELECT a FROM t JOIN u ON t.b = u.b"),
    )


def test_a_star_over_a_using_join_writes_the_pair_once() raises:
    # The ON spelling of the same join answers seven columns with `b` twice.
    assert_equal(
        _plan("SELECT * FROM t JOIN u USING (b)"),
        (
            "PROJECT [a, b, g, f, k, z]\n"
            "  JOIN inner [b = b]\n"
            "    SCAN t []\n"
            "    SCAN u []\n"
        ),
    )


def test_the_name_a_using_join_merged_may_be_written_bare() raises:
    # The node below still produces both columns called `b`, so without the
    # merge this is the ambiguity binding refuses. The join is what decides,
    # and the decision is written into the expression here.
    assert_equal(
        _plan("SELECT b FROM t JOIN u USING (b)"),
        "PROJECT [b]\n  JOIN inner [b = b]\n    SCAN t []\n    SCAN u []\n",
    )


def test_either_side_may_still_be_written_in_front_of_a_merged_name() raises:
    # DuckDB's rule and Postgres's. Merging the pair does not take the two
    # columns out of reach, it only gives the bare name a meaning.
    assert_equal(
        _plan("SELECT t.b FROM t JOIN u USING (b)"),
        _plan("SELECT u.b FROM t JOIN u USING (b)"),
    )


def test_a_natural_join_pairs_every_name_the_two_sides_share() raises:
    assert_equal(
        _plan("SELECT * FROM t NATURAL JOIN u"),
        _plan("SELECT * FROM t JOIN u USING (b)"),
    )


def test_a_natural_join_with_nothing_in_common_is_a_cross_join() raises:
    assert_equal(
        _plan("SELECT * FROM u NATURAL JOIN (SELECT a FROM t) v"),
        (
            "PROJECT [b, k, z, a]\n"
            "  JOIN cross []\n"
            "    SCAN u []\n"
            "    PROJECT [a]\n"
            "      SCAN t []\n"
        ),
    )


def test_a_using_join_that_names_a_column_one_side_lacks_says_so() raises:
    with assert_raises(contains="has no column called that"):
        _ = _plan("SELECT a FROM t JOIN u USING (a)")


def test_a_full_join_with_a_using_is_refused_by_name() raises:
    # The merged column of a full join is the first of the pair that is not
    # null, and that is a coalesce over the join rather than a column of it.
    with assert_raises(contains="FULL USING join"):
        _ = _plan("SELECT b FROM t FULL JOIN u USING (b)")


def test_a_using_join_over_a_subquery_is_refused_by_name() raises:
    with assert_raises(contains="USING join over a subquery"):
        _ = _plan("SELECT b FROM t JOIN (SELECT b FROM u) v USING (b)")


def test_a_semi_join_keeps_the_left_rows_that_matched() raises:
    assert_equal(
        _plan("SELECT a FROM t SEMI JOIN u ON t.b = u.b"),
        "PROJECT [a]\n  JOIN semi [b = b]\n    SCAN t []\n    SCAN u []\n",
    )


def test_an_anti_join_is_the_same_node_and_the_other_answer() raises:
    assert_equal(
        _plan("SELECT a FROM t ANTI JOIN u ON t.b = u.b"),
        "PROJECT [a]\n  JOIN anti [b = b]\n    SCAN t []\n    SCAN u []\n",
    )


def test_a_star_over_a_semi_join_is_the_left_side_alone() raises:
    # The same query with an inner join answers seven columns. A semi join asks
    # a question about the right side and keeps none of the answer.
    assert_equal(
        _plan("SELECT * FROM t SEMI JOIN u ON t.b = u.b"),
        (
            "PROJECT [a, b, g, f]\n"
            "  JOIN semi [b = b]\n"
            "    SCAN t []\n"
            "    SCAN u []\n"
        ),
    )


def test_the_right_side_of_a_semi_join_is_out_of_reach_above_it() raises:
    # DuckDB answers `Referenced table "u" not found` for this, because the
    # right side of a semi join is not one of the tables the query is selecting
    # from. The condition is the exception and still reads it.
    with assert_raises(contains="nothing in this query is called 'u'"):
        _ = _plan("SELECT u.k FROM t SEMI JOIN u ON t.b = u.b")


def test_a_semi_join_may_name_its_key_with_using() raises:
    assert_equal(
        _plan("SELECT a FROM t SEMI JOIN u USING (b)"),
        _plan("SELECT a FROM t SEMI JOIN u ON t.b = u.b"),
    )


def test_a_semi_join_needs_an_equality_between_its_two_sides() raises:
    # DuckDB takes any predicate here. This join node carries key pairs, and the
    # rest of a condition is ordinarily a filter above the join, which here
    # would read columns the join did not keep.
    with assert_raises(contains="semi join on equalities"):
        _ = _plan("SELECT a FROM t SEMI JOIN u ON t.b > u.b")
    with assert_raises(contains="anti join on equalities"):
        _ = _plan("SELECT a FROM t ANTI JOIN u ON t.b = u.b AND t.a > u.k")


def test_an_in_over_a_subquery_is_a_semi_join() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE b IN (SELECT k FROM u)"),
        (
            "PROJECT [a]\n"
            "  JOIN semi [b = k]\n"
            "    SCAN t []\n"
            "    PROJECT [k]\n"
            "      SCAN u []\n"
        ),
    )


def test_the_rest_of_a_where_stays_a_filter_under_the_semi_join() raises:
    # A semi join hands out the left side unchanged, so the filter and the join
    # commute and the filter goes first, which is the side with fewer rows to
    # probe with.
    assert_equal(
        _plan("SELECT a FROM t WHERE a > 1 AND b IN (SELECT k FROM u)"),
        (
            "PROJECT [a]\n"
            "  JOIN semi [b = k]\n"
            "    FILTER a > 1\n"
            "      SCAN t []\n"
            "    PROJECT [k]\n"
            "      SCAN u []\n"
        ),
    )


def test_two_ins_in_one_where_are_two_semi_joins() raises:
    assert_true(
        _plan(
            "SELECT a FROM t WHERE b IN (SELECT k FROM u)"
            " AND a IN (SELECT b FROM u)"
        ).startswith(
            "PROJECT [a]\n  JOIN semi [a = b]\n    JOIN semi [b = k]\n"
        )
    )


def test_a_subquery_naming_a_column_the_outer_query_has_is_not_ambiguous() raises:
    # Both tables have a `b`. The right key binds against the build side alone,
    # so the name resolves there and nowhere else.
    assert_equal(
        _plan("SELECT a FROM t WHERE b IN (SELECT b FROM u)"),
        (
            "PROJECT [a]\n"
            "  JOIN semi [b = b]\n"
            "    SCAN t []\n"
            "    PROJECT [b]\n"
            "      SCAN u []\n"
        ),
    )


def test_a_not_in_over_a_subquery_is_a_mark_join_and_a_not() raises:
    # An anti join is the classic wrong answer here. One null in the subquery
    # makes NOT IN null for every row rather than true, and an anti join keeps
    # those rows rather than dropping them. The mark join marks such a row null,
    # the NOT over it is null in turn, and a filter does not keep a null.
    assert_equal(
        _plan("SELECT a FROM t WHERE b NOT IN (SELECT k FROM u)"),
        (
            "PROJECT [a]\n"
            "  FILTER not(__mark_0)\n"
            "    JOIN mark [b = k] -> __mark_0\n"
            "      SCAN t []\n"
            "      PROJECT [k]\n"
            "        SCAN u []\n"
        ),
    )


def test_an_in_written_in_the_select_list_is_a_mark_join() raises:
    # Written there it is a value rather than a filter, so the answer has to
    # arrive on every row and not only on the rows that matched.
    assert_equal(
        _plan("SELECT a, b IN (SELECT k FROM u) AS hit FROM t"),
        (
            "PROJECT [a, __mark_0 as hit]\n"
            "  JOIN mark [b = k] -> __mark_0\n"
            "    SCAN t []\n"
            "    PROJECT [k]\n"
            "      SCAN u []\n"
        ),
    )


def test_an_in_under_an_or_is_a_mark_join_rather_than_a_semi_join() raises:
    # A semi join answers which rows to keep, and under an OR that is not the
    # question being asked: a row the IN did not match can still be kept.
    assert_equal(
        _plan("SELECT a FROM t WHERE b IN (SELECT k FROM u) OR a > 5"),
        (
            "PROJECT [a]\n"
            "  FILTER or(__mark_0, a > 5)\n"
            "    JOIN mark [b = k] -> __mark_0\n"
            "      SCAN t []\n"
            "      PROJECT [k]\n"
            "        SCAN u []\n"
        ),
    )


def test_the_in_a_where_is_the_and_of_is_still_the_semi_join() raises:
    # The mark join answers the same question and a semi join is the cheaper
    # way to ask it, so the one place a semi join is right keeps it.
    assert_equal(
        _plan("SELECT a FROM t WHERE b IN (SELECT k FROM u) AND a > 5"),
        (
            "PROJECT [a]\n"
            "  JOIN semi [b = k]\n"
            "    FILTER a > 5\n"
            "      SCAN t []\n"
            "    PROJECT [k]\n"
            "      SCAN u []\n"
        ),
    )


def test_two_ins_written_as_values_each_get_a_column() raises:
    # Named by how many were taken out before, so two of them in one query are
    # two joins and two columns rather than one name meaning both.
    assert_equal(
        _plan(
            "SELECT a FROM t WHERE b IN (SELECT k FROM u) OR b NOT IN"
            " (SELECT b FROM u)"
        ),
        (
            "PROJECT [a]\n"
            "  FILTER or(__mark_0, not(__mark_1))\n"
            "    JOIN mark [b = b] -> __mark_1\n"
            "      JOIN mark [b = k] -> __mark_0\n"
            "        SCAN t []\n"
            "        PROJECT [k]\n"
            "          SCAN u []\n"
            "      PROJECT [b]\n"
            "        SCAN u []\n"
        ),
    )


def test_an_in_written_over_an_aggregate_says_why_it_is_refused() raises:
    # The mark join goes above the FROM, which is under the aggregate, and an
    # aggregate hands up its keys and its folds rather than everything it read.
    with assert_raises(contains="written somewhere else"):
        _ = _plan(
            "SELECT g FROM t GROUP BY g HAVING sum(a) IN (SELECT k FROM u)"
        )


def test_an_in_whose_subquery_hands_out_two_columns_is_refused() raises:
    with assert_raises(contains="hands out 2 columns"):
        _ = _plan("SELECT a FROM t WHERE b IN (SELECT k, z FROM u)")


def test_a_correlated_in_is_refused_by_the_scope_it_lowers_against() raises:
    # The subquery lowers against a scope of its own, which is the ordinary rule
    # and is also what keeps this rewrite honest: a subquery that reads an outer
    # column runs once per outer row and a join's build side runs once.
    with assert_raises(contains="nothing in this query is called 't'"):
        _ = _plan(
            "SELECT a FROM t WHERE b IN (SELECT k FROM u WHERE u.b = t.b)"
        )


def test_an_equals_any_is_the_semi_join_an_in_is() raises:
    # `x = ANY (S)` is true when some row of S equals x, which is the whole of
    # `x IN (S)`, so it is read as the one it is rather than lowered twice.
    assert_equal(
        _plan("SELECT a FROM t WHERE b = ANY (SELECT k FROM u)"),
        _plan("SELECT a FROM t WHERE b IN (SELECT k FROM u)"),
    )


def test_some_does_not_parse_because_the_grammar_omits_it() raises:
    # DuckDB itself takes `= SOME` and answers it the way it answers `= ANY`,
    # and the PEG grammar it publishes has `SubqueryAny <- 'ANY'` with no SOME
    # beside it. The grammar here is vendored verbatim, so this is a syntax
    # error until upstream adds the word rather than something to patch in.
    with assert_raises(contains="syntax error"):
        _ = _plan("SELECT a FROM t WHERE b = SOME (SELECT k FROM u)")


def test_a_not_equals_all_is_the_mark_join_a_not_in_is() raises:
    # And so it gets the null aware answer for free. A subquery holding a null
    # marks a row that matched nothing null rather than false, the NOT over it
    # is null, and a filter does not keep a null, which is what SQL says.
    assert_equal(
        _plan("SELECT a FROM t WHERE b <> ALL (SELECT k FROM u)"),
        (
            "PROJECT [a]\n"
            "  FILTER not(__mark_0)\n"
            "    JOIN mark [b = k] -> __mark_0\n"
            "      SCAN t []\n"
            "      PROJECT [k]\n"
            "        SCAN u []\n"
        ),
    )


def test_the_other_spellings_of_the_two_comparisons_go_the_same_way() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE b != ALL (SELECT k FROM u)"),
        _plan("SELECT a FROM t WHERE b NOT IN (SELECT k FROM u)"),
    )


def test_an_equals_any_in_a_select_list_is_a_mark_join() raises:
    assert_equal(
        _plan("SELECT a, b = ANY (SELECT k FROM u) AS hit FROM t"),
        (
            "PROJECT [a, __mark_0 as hit]\n"
            "  JOIN mark [b = k] -> __mark_0\n"
            "    SCAN t []\n"
            "    PROJECT [k]\n"
            "      SCAN u []\n"
        ),
    )


def test_an_equals_any_under_an_or_is_a_mark_join() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE b = ANY (SELECT k FROM u) OR a > 5"),
        (
            "PROJECT [a]\n"
            "  FILTER or(__mark_0, a > 5)\n"
            "    JOIN mark [b = k] -> __mark_0\n"
            "      SCAN t []\n"
            "      PROJECT [k]\n"
            "        SCAN u []\n"
        ),
    )


def test_a_greater_than_any_reads_the_smallest_row_of_the_subquery() raises:
    # Some row is under `a` when the smallest one is, so the whole of what the
    # subquery has to say is a fold over it.
    assert_equal(
        _plan("SELECT a FROM t WHERE a > ANY (SELECT b FROM u)"),
        (
            "PROJECT [a]\n"
            "  FILTER and(or(a > __low_0, __pad_0), __fill_0)\n"
            "    JOIN cross []\n"
            "      SCAN t []\n"
            "      PROJECT [__low as __low_0, __high as __high_0,"
            " and(__rows != __seen, null) as __pad_0, __rows > 0 as"
            " __fill_0]\n"
            "        AGGREGATE [] -> [min(b), max(b), count(1), count(b)]\n"
            "          PROJECT [b]\n"
            "            SCAN u []\n"
        ),
    )


def test_a_greater_than_all_reads_the_largest_row_instead() raises:
    # And pads with a true rather than a false, since `ALL` over nothing is
    # true and `ANY` over nothing is false.
    assert_equal(
        _plan("SELECT a FROM t WHERE b >= ALL (SELECT k FROM u)"),
        (
            "PROJECT [a]\n"
            "  FILTER or(and(b >= __high_0, __pad_0), __fill_0)\n"
            "    JOIN cross []\n"
            "      SCAN t []\n"
            "      PROJECT [__low as __low_0, __high as __high_0,"
            " or(__rows == __seen, null) as __pad_0, __rows == 0 as"
            " __fill_0]\n"
            "        AGGREGATE [] -> [min(k), max(k), count(1), count(k)]\n"
            "          PROJECT [k]\n"
            "            SCAN u []\n"
        ),
    )


def test_a_less_than_any_and_a_less_than_all_read_the_other_end() raises:
    assert_true(
        _plan("SELECT a FROM t WHERE b < ANY (SELECT k FROM u)").startswith(
            "PROJECT [a]\n  FILTER and(or(b < __high_0, __pad_0), __fill_0)\n"
        )
    )
    assert_true(
        _plan("SELECT a FROM t WHERE b <= ALL (SELECT k FROM u)").startswith(
            "PROJECT [a]\n  FILTER or(and(b <= __low_0, __pad_0), __fill_0)\n"
        )
    )


def test_an_equals_all_asks_about_both_ends_at_once() raises:
    # Every row equals `b` when both ends do, which is the one comparison that
    # neither end answers on its own.
    assert_true(
        _plan("SELECT a FROM t WHERE b = ALL (SELECT k FROM u)").startswith(
            "PROJECT [a]\n  FILTER or(and(and(b == __low_0, b =="
            " __high_0), __pad_0), __fill_0)\n"
        )
    )


def test_a_not_equals_any_asks_about_both_ends_the_other_way() raises:
    assert_true(
        _plan("SELECT a FROM t WHERE b <> ANY (SELECT k FROM u)").startswith(
            "PROJECT [a]\n  FILTER and(or(or(b != __low_0, b !="
            " __high_0), __pad_0), __fill_0)\n"
        )
    )


def test_a_quantified_comparison_in_a_select_list_is_the_same_row() raises:
    assert_equal(
        _plan("SELECT a, b < ALL (SELECT k FROM u) AS small FROM t"),
        (
            "PROJECT [a, or(and(b < __low_0, __pad_0), __fill_0) as"
            " small]\n"
            "  JOIN cross []\n"
            "    SCAN t []\n"
            "    PROJECT [__low as __low_0, __high as __high_0, or(__rows =="
            " __seen, null) as __pad_0, __rows == 0 as __fill_0]\n"
            "      AGGREGATE [] -> [min(k), max(k), count(1), count(k)]\n"
            "        PROJECT [k]\n"
            "          SCAN u []\n"
        ),
    )


def test_two_quantified_comparisons_each_get_a_row() raises:
    assert_true(
        _plan(
            "SELECT a FROM t WHERE b > ANY (SELECT k FROM u)"
            " AND b < ALL (SELECT b FROM u)"
        ).startswith(
            "PROJECT [a]\n  FILTER and(and(or(b > __low_0, __pad_0),"
            " __fill_0), or(and(b < __low_1, __pad_1), __fill_1))\n"
        )
    )


def test_a_correlated_quantified_comparison_says_why_it_is_refused() raises:
    with assert_raises(contains="dependent join"):
        _ = _plan(
            "SELECT a FROM t WHERE b > ALL (SELECT k FROM u WHERE u.b = t.b)"
        )


def test_a_quantified_subquery_of_two_columns_is_refused() raises:
    with assert_raises(contains="hands out 2 columns"):
        _ = _plan("SELECT a FROM t WHERE b > ANY (SELECT k, z FROM u)")


def test_a_quantified_comparison_over_an_aggregate_is_refused() raises:
    # The row goes above the FROM, which is under the aggregate, and an
    # aggregate hands up its keys and its folds rather than everything it read.
    with assert_raises(contains="written somewhere else"):
        _ = _plan(
            "SELECT g FROM t GROUP BY g HAVING max(a) < ALL (SELECT k FROM u)"
        )


def test_an_exists_written_under_an_or_is_counted_under_a_cross_join() raises:
    # An `IN` written there is the mark join, and this is the same boolean per
    # row without the pair of keys that join is given. Counting the subquery's
    # rows answers it instead, and the comparison goes under the cross join so
    # that it is done once rather than once per outer row.
    assert_equal(
        _plan("SELECT a FROM t WHERE a > 1 OR EXISTS (SELECT k FROM u)"),
        (
            "PROJECT [a]\n"
            "  FILTER or(a > 1, __has_0)\n"
            "    JOIN cross []\n"
            "      SCAN t []\n"
            "      PROJECT [__rows > 0 as __has_0]\n"
            "        AGGREGATE [] -> [count(1)]\n"
            "          PROJECT [k]\n"
            "            SCAN u []\n"
        ),
    )


def test_an_exists_in_a_select_list_is_the_same_count() raises:
    assert_equal(
        _plan("SELECT a, EXISTS (SELECT k FROM u) AS any_u FROM t"),
        (
            "PROJECT [a, __has_0 as any_u]\n"
            "  JOIN cross []\n"
            "    SCAN t []\n"
            "    PROJECT [__rows > 0 as __has_0]\n"
            "      AGGREGATE [] -> [count(1)]\n"
            "        PROJECT [k]\n"
            "          SCAN u []\n"
        ),
    )


def test_a_not_exists_over_a_table_is_that_negated() raises:
    # The `NOT` stays where it was written and reads the column the counting
    # answered, the way a `NOT IN` reads the mark join's column.
    assert_equal(
        _plan("SELECT a FROM t WHERE NOT EXISTS (SELECT k FROM u)"),
        (
            "PROJECT [a]\n"
            "  FILTER not(__has_0)\n"
            "    JOIN cross []\n"
            "      SCAN t []\n"
            "      PROJECT [__rows > 0 as __has_0]\n"
            "        AGGREGATE [] -> [count(1)]\n"
            "          PROJECT [k]\n"
            "            SCAN u []\n"
        ),
    )


def test_an_exists_written_over_an_aggregate_says_why_it_is_refused() raises:
    # The cross join goes above the FROM and below everything else, so the
    # column it wrote is under the aggregate rather than over it.
    with assert_raises(contains="written somewhere else"):
        _ = _plan(
            "SELECT count(a) AS n FROM t GROUP BY g HAVING"
            " EXISTS (SELECT k FROM u)"
        )


def test_a_subquery_that_answers_one_value_is_a_cross_join() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE a > (SELECT max(b) FROM u)"),
        (
            "PROJECT [a]\n"
            "  FILTER a > __sub_0\n"
            "    JOIN cross []\n"
            "      SCAN t []\n"
            "      PROJECT [__expr_0 as __sub_0]\n"
            "        PROJECT [__agg_0 as __expr_0]\n"
            "          AGGREGATE [] -> [max(b)]\n"
            "            SCAN u []\n"
        ),
    )


def test_a_subquery_in_a_select_list_is_the_same_join() raises:
    assert_equal(
        _plan("SELECT a, (SELECT max(b) FROM u) AS top FROM t"),
        (
            "PROJECT [a, __sub_0 as top]\n"
            "  JOIN cross []\n"
            "    SCAN t []\n"
            "    PROJECT [__expr_0 as __sub_0]\n"
            "      PROJECT [__agg_0 as __expr_0]\n"
            "        AGGREGATE [] -> [max(b)]\n"
            "          SCAN u []\n"
        ),
    )


def test_a_subquery_over_no_table_is_one_row_too() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE a > (SELECT 1)"),
        (
            "PROJECT [a]\n"
            "  FILTER a > __sub_0\n"
            "    JOIN cross []\n"
            "      SCAN t []\n"
            "      PROJECT [__expr_0 as __sub_0]\n"
            "        PROJECT [1 as __expr_0]\n"
            "          VALUES [__row] (0)\n"
        ),
    )


def test_two_subqueries_in_one_query_are_two_cross_joins() raises:
    var text = _plan(
        "SELECT a FROM t WHERE a > (SELECT max(b) FROM u)"
        " AND b < (SELECT min(k) FROM u)"
    )
    assert_true("__sub_0" in text)
    assert_true("__sub_1" in text)


def test_the_answer_is_renamed_so_it_cannot_clash() raises:
    # Both tables have a column called b, and the subquery's answer is called
    # that too until it is renamed on the way out.
    assert_true(
        "__sub_0" in _plan("SELECT a FROM t WHERE b > (SELECT max(b) FROM u)")
    )


def test_a_subquery_that_is_not_one_row_by_construction_is_refused() raises:
    with assert_raises(contains="one row by construction"):
        _ = _plan("SELECT a FROM t WHERE a > (SELECT b FROM u)")
    with assert_raises(contains="one row by construction"):
        _ = _plan("SELECT a FROM t WHERE a > (SELECT max(b) FROM u GROUP BY z)")


def test_a_limit_on_a_subquery_is_not_read_as_a_row_count() raises:
    # Over a table with nothing in it a LIMIT 1 answers no rows, where SQL says
    # the subquery is null, so this is not the same guarantee a fold gives.
    with assert_raises(contains="answers no rows"):
        _ = _plan("SELECT a FROM t WHERE a > (SELECT b FROM u LIMIT 1)")


def test_a_subquery_that_hands_out_two_columns_is_refused() raises:
    with assert_raises(contains="hands out 2 columns"):
        _ = _plan("SELECT a FROM t WHERE a > (SELECT max(b), min(k) FROM u)")


def test_a_correlated_one_is_a_group_under_a_left_join() raises:
    # The subquery is asked once rather than once per outer row. The equality
    # that made it correlated is the group key and the join key, and the fold
    # comes out of the aggregate under a name the outer query cannot collide
    # with.
    assert_equal(
        _plan(
            "SELECT a FROM t WHERE a > (SELECT max(k) FROM u WHERE u.b = t.b)"
        ),
        (
            "PROJECT [a]\n"
            "  FILTER a > __sub_0\n"
            "    JOIN left [b = __by_0]\n"
            "      SCAN t []\n"
            "      PROJECT [b as __by_0, __agg_0 as __sub_0]\n"
            "        AGGREGATE [b] -> [max(k)]\n"
            "          SCAN u []\n"
        ),
    )


def test_a_correlated_one_in_a_select_list_is_the_same_join() raises:
    assert_equal(
        _plan("SELECT a, (SELECT max(k) FROM u WHERE u.b = t.b) AS m FROM t"),
        (
            "PROJECT [a, __sub_0 as m]\n"
            "  JOIN left [b = __by_0]\n"
            "    SCAN t []\n"
            "    PROJECT [b as __by_0, __agg_0 as __sub_0]\n"
            "      AGGREGATE [b] -> [max(k)]\n"
            "        SCAN u []\n"
        ),
    )


def test_a_correlated_one_puts_its_own_condition_under_the_fold() raises:
    # `k > 2` reads the subquery's table and nothing else, so it runs once
    # over that table rather than once per pairing.
    assert_equal(
        _plan(
            "SELECT a FROM t WHERE a > (SELECT avg(k) + 1 FROM u WHERE u.b ="
            " t.b AND k > 2)"
        ),
        (
            "PROJECT [a]\n"
            "  FILTER a > __sub_0\n"
            "    JOIN left [b = __by_0]\n"
            "      SCAN t []\n"
            "      PROJECT [b as __by_0, __agg_0 + 1 as __sub_0]\n"
            "        AGGREGATE [b] -> [mean(k)]\n"
            "          FILTER k > 2\n"
            "            SCAN u []\n"
        ),
    )


def test_a_correlated_one_that_counts_is_refused_by_name() raises:
    # The count bug. A left join answers null for an outer row whose group has
    # no rows in it, and a count over nothing is zero rather than null.
    with assert_raises(contains="a count of nothing is zero"):
        _ = _plan(
            "SELECT a FROM t WHERE a > (SELECT count(k) FROM u WHERE u.b = t.b)"
        )


def test_a_correlated_one_read_another_way_is_refused_by_name() raises:
    with assert_raises(contains="which is the dependent join"):
        _ = _plan(
            "SELECT a FROM t WHERE a > (SELECT max(k) FROM u WHERE u.k > t.b)"
        )


def test_a_correlated_one_that_does_not_fold_is_refused_by_name() raises:
    with assert_raises(contains="a fold answers one value per group"):
        _ = _plan("SELECT a FROM t WHERE a > (SELECT k FROM u WHERE u.b = t.b)")


def test_a_correlated_one_with_its_own_group_by_is_refused_by_name() raises:
    with assert_raises(contains="the group this builds is the correlation"):
        _ = _plan(
            "SELECT a FROM t WHERE a > (SELECT max(k) FROM u WHERE u.b = t.b"
            " GROUP BY z)"
        )


def test_an_outer_name_nothing_is_called_is_still_refused() raises:
    with assert_raises(contains="nothing in this query is called 'v'"):
        _ = _plan(
            "SELECT a FROM t WHERE a > (SELECT max(k) FROM u WHERE u.b = v.b)"
        )


def test_a_subquery_above_an_aggregate_says_why_it_cannot_be_read() raises:
    with assert_raises(contains="hands up its keys and its folds"):
        _ = _plan(
            "SELECT g FROM t GROUP BY g HAVING sum(a) > (SELECT max(b) FROM u)"
        )
    with assert_raises(contains="hands up its keys and its folds"):
        _ = _plan("SELECT sum(a) + (SELECT max(b) FROM u) FROM t")


def test_a_correlated_exists_is_a_semi_join() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.b = t.b)"),
        "PROJECT [a]\n  JOIN semi [b = b]\n    SCAN t []\n    SCAN u []\n",
    )


def test_a_correlated_not_exists_is_the_anti_join() raises:
    # The one place where the negation is the other join rather than a refusal.
    # A left row with a null key matches nothing, so an anti join keeps it, and
    # `NOT EXISTS` over a null is true, which is the same answer. `NOT IN` is
    # the one that differs and it is why that one is refused.
    assert_equal(
        _plan(
            "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.b = t.b)"
        ),
        "PROJECT [a]\n  JOIN anti [b = b]\n    SCAN t []\n    SCAN u []\n",
    )


def test_the_uncorrelated_half_of_an_exists_is_a_filter_under_the_join() raises:
    # Under the join rather than over it, because it reads the subquery's own
    # table and there it runs once rather than once per pairing.
    assert_equal(
        _plan(
            "SELECT a FROM t WHERE EXISTS"
            " (SELECT 1 FROM u WHERE u.b = t.b AND u.k > 3)"
        ),
        (
            "PROJECT [a]\n"
            "  JOIN semi [b = b]\n"
            "    SCAN t []\n"
            "    FILTER k > 3\n"
            "      SCAN u []\n"
        ),
    )


def test_two_correlated_equalities_are_two_key_pairs() raises:
    assert_true(
        "JOIN semi [a = k, b = b]"
        in _plan(
            "SELECT g FROM t WHERE EXISTS"
            " (SELECT 1 FROM u WHERE u.k = t.a AND t.b = u.b)"
        )
    )


def test_the_subquery_of_an_exists_is_out_of_reach_above_it() raises:
    with assert_raises(contains="nothing in this query is called 'u'"):
        _ = _plan(
            "SELECT u.k FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.b = t.b)"
        )


def test_an_exists_that_reads_no_outer_column_is_counted() raises:
    # It asks whether the table has any row at all, which every outer row gets
    # the same answer to, so it is a value rather than a semi join. Which of the
    # two it is has to be settled before the FROM under it is lowered, and a
    # subquery whose WHERE holds no equality has no key pair to give whatever
    # its names turn out to mean.
    assert_equal(
        _plan("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k > 3)"),
        (
            "PROJECT [a]\n"
            "  FILTER __has_0\n"
            "    JOIN cross []\n"
            "      SCAN t []\n"
            "      PROJECT [__rows > 0 as __has_0]\n"
            "        AGGREGATE [] -> [count(1)]\n"
            "          PROJECT [1 as __expr_0]\n"
            "            FILTER k > 3\n"
            "              SCAN u []\n"
        ),
    )


def test_a_correlation_that_is_not_an_equality_is_refused() raises:
    # No equality in the subquery's WHERE means no key pair whatever the names
    # mean, so this goes to the value form and the scope there is what refuses
    # it. The message says which shape that is.
    with assert_raises(contains="dependent join"):
        _ = _plan(
            "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.b > t.b)"
        )


def test_an_exists_over_an_aggregate_runs_when_it_reads_no_outer_column() raises:
    # An aggregate with no GROUP BY answers one row over no rows, so an EXISTS
    # over one is true even where the subquery found nothing. The semi join has
    # no way to say that and the counting needs no way, since it counts the one
    # row the fold hands out.
    assert_equal(
        _plan("SELECT a FROM t WHERE EXISTS (SELECT sum(k) FROM u)"),
        (
            "PROJECT [a]\n"
            "  FILTER __has_0\n"
            "    JOIN cross []\n"
            "      SCAN t []\n"
            "      PROJECT [__rows > 0 as __has_0]\n"
            "        AGGREGATE [] -> [count(1)]\n"
            "          PROJECT [__agg_0 as __expr_0]\n"
            "            AGGREGATE [] -> [sum(k)]\n"
            "              SCAN u []\n"
        ),
    )


def test_a_correlated_exists_over_an_aggregate_is_refused() raises:
    with assert_raises(contains="dependent join"):
        _ = _plan(
            "SELECT a FROM t WHERE EXISTS"
            " (SELECT sum(k) FROM u WHERE u.b = t.b)"
        )


def test_a_correlated_exists_with_a_limit_on_it_is_refused() raises:
    with assert_raises(contains="dependent join"):
        _ = _plan(
            "SELECT a FROM t WHERE EXISTS"
            " (SELECT 1 FROM u WHERE u.b = t.b LIMIT 1)"
        )


def test_an_exists_with_a_limit_on_it_counts_what_the_limit_left() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE EXISTS (SELECT k FROM u LIMIT 0)"),
        (
            "PROJECT [a]\n"
            "  FILTER __has_0\n"
            "    JOIN cross []\n"
            "      SCAN t []\n"
            "      PROJECT [__rows > 0 as __has_0]\n"
            "        AGGREGATE [] -> [count(1)]\n"
            "          LIMIT 0\n"
            "            PROJECT [k]\n"
            "              SCAN u []\n"
        ),
    )


def test_an_exists_beside_a_plain_condition_keeps_both() raises:
    assert_equal(
        _plan(
            "SELECT a FROM t WHERE a > 1 AND EXISTS"
            " (SELECT 1 FROM u WHERE u.b = t.b)"
        ),
        (
            "PROJECT [a]\n"
            "  JOIN semi [b = b]\n"
            "    FILTER a > 1\n"
            "      SCAN t []\n"
            "    SCAN u []\n"
        ),
    )


def test_a_condition_may_reach_only_the_two_tables_it_joins() raises:
    # The comma binds looser than the JOIN word, so `u` and `t s` are the join
    # and `t` is beside it, and a condition naming `t` there is reaching out of
    # the join it was written on.
    with assert_raises(contains="a table this join does not read"):
        _ = _plan("SELECT a FROM t, u JOIN t s ON t.a = s.a")


def test_the_joins_with_no_node_yet_each_say_which_one() raises:
    with assert_raises(contains="POSITIONAL join"):
        _ = _plan("SELECT a FROM t POSITIONAL JOIN u")
    with assert_raises(contains="ASOF join"):
        _ = _plan("SELECT a FROM t ASOF JOIN u ON t.a = u.k")
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


def test_a_minus_in_front_of_a_column_is_a_unary() raises:
    assert_equal(
        _plan("SELECT -a FROM t"), "PROJECT [-a as __expr_0]\n  SCAN t []\n"
    )


def test_a_minus_in_front_of_an_expression_wraps_it() raises:
    assert_equal(
        _plan("SELECT -(a + 1) FROM t"),
        "PROJECT [-(a + 1) as __expr_0]\n  SCAN t []\n",
    )


def test_a_minus_in_front_of_a_number_is_a_unary_until_it_is_folded() raises:
    # This helper binds and stops, so what it shows is the plan before any pass
    # has run, and at that point the sign is still an operation over a literal.
    # The simplify pass answers it and the operator never sees one, which is why
    # it has no constant form the way `Compute` does.
    assert_equal(
        _plan("SELECT a FROM t WHERE b > -5"),
        "PROJECT [a]\n  FILTER b > (-5)\n    SCAN t []\n",
    )


def test_a_plus_in_front_of_a_column_is_a_unary_too() raises:
    assert_equal(
        _plan("SELECT +a FROM t"), "PROJECT [+a as __expr_0]\n  SCAN t []\n"
    )


def test_a_like_is_a_call_rather_than_an_operation() raises:
    # The right side is a pattern rather than an operand, and what runs it reads
    # that pattern once while the plan is being lowered, so it is a named call
    # the way `and` and `or` are and not a thirteenth binary operator.
    assert_equal(
        _plan("SELECT a FROM t WHERE g LIKE 'a%'"),
        "PROJECT [a]\n  FILTER like(g, a%)\n    SCAN t []\n",
    )


def test_a_not_like_is_the_call_with_a_not_over_it() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE g NOT LIKE '%a%'"),
        "PROJECT [a]\n  FILTER not(like(g, %a%))\n    SCAN t []\n",
    )


def test_the_operator_spelling_of_a_like_is_the_same_call() raises:
    # `~~` is what postgres calls it and duckdb takes both, so the two have to
    # reach the same plan or a query means one thing and its rewrite another.
    assert_equal(
        _plan("SELECT a FROM t WHERE g ~~ 'a%'"),
        _plan("SELECT a FROM t WHERE g LIKE 'a%'"),
    )
    assert_equal(
        _plan("SELECT a FROM t WHERE g !~~ 'a%'"),
        _plan("SELECT a FROM t WHERE g NOT LIKE 'a%'"),
    )


def test_a_like_over_a_number_is_refused() raises:
    with assert_raises(contains="'like' reads text and argument 0 is"):
        _ = _plan("SELECT a FROM t WHERE b LIKE 'a%'")


def test_an_ilike_says_it_is_the_case_fold_that_is_missing() raises:
    with assert_raises(contains="matches a LIKE pattern byte for byte"):
        _ = _plan("SELECT a FROM t WHERE g ILIKE 'a%'")


def test_a_similar_to_says_there_is_no_regular_expression_engine() raises:
    with assert_raises(contains="no regular expression engine"):
        _ = _plan("SELECT a FROM t WHERE g SIMILAR TO 'a.*'")


def test_an_is_null_is_a_call_and_not_a_comparison() raises:
    # The shape is the whole point. An equality against a null would print the
    # same way a query writes it and would answer null for every row.
    assert_equal(
        _plan("SELECT a FROM t WHERE g IS NULL"),
        "PROJECT [a]\n  FILTER is_null(g)\n    SCAN t []\n",
    )


def test_an_is_not_null_is_the_other_call() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE g IS NOT NULL"),
        "PROJECT [a]\n  FILTER is_not_null(g)\n    SCAN t []\n",
    )


def test_the_one_word_spellings_build_the_same_two_calls() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE g ISNULL"),
        _plan("SELECT a FROM t WHERE g IS NULL"),
    )
    assert_equal(
        _plan("SELECT a FROM t WHERE g NOTNULL"),
        _plan("SELECT a FROM t WHERE g IS NOT NULL"),
    )


def test_an_is_true_asks_whether_the_value_is_there_and_holds() raises:
    # Two questions rather than one, because a null has to come out false here
    # and a comparison against it does not. The and is Kleene's, so a null on
    # one side of a false is a false, which is what makes the pair total.
    assert_equal(
        _plan("SELECT a FROM t WHERE (b > 1) IS TRUE"),
        "PROJECT [a]\n  FILTER and(is_not_null(b > 1), b > 1)\n    SCAN t []\n",
    )


def test_an_is_false_puts_a_not_over_the_value_half() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE (b > 1) IS FALSE"),
        (
            "PROJECT [a]\n  FILTER and(is_not_null(b > 1), not(b > 1))\n"
            "    SCAN t []\n"
        ),
    )


def test_an_is_not_true_asks_the_pair_the_other_way_round() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE (b > 1) IS NOT TRUE"),
        "PROJECT [a]\n  FILTER or(is_null(b > 1), not(b > 1))\n    SCAN t []\n",
    )


def test_an_is_not_false_keeps_the_value_half_as_it_was_written() raises:
    assert_equal(
        _plan("SELECT a FROM t WHERE (b > 1) IS NOT FALSE"),
        "PROJECT [a]\n  FILTER or(is_null(b > 1), b > 1)\n    SCAN t []\n",
    )


def test_a_substring_is_a_call_with_its_two_numbers_on_it() raises:
    assert_equal(
        _plan("SELECT substring(g, 2, 3) FROM t"),
        "PROJECT [substring(g, 2, 3) as __expr_0]\n  SCAN t []\n",
    )


def test_the_three_spellings_of_a_substring_build_the_same_plan() raises:
    var want = _plan("SELECT substring(g, 2, 3) FROM t")
    assert_equal(_plan("SELECT substr(g, 2, 3) FROM t"), want)
    assert_equal(_plan("SELECT SUBSTRING(g FROM 2 FOR 3) FROM t"), want)


def test_a_substring_written_with_only_a_for_starts_at_one() raises:
    assert_equal(
        _plan("SELECT SUBSTRING(g FOR 3) FROM t"),
        _plan("SELECT substring(g, 1, 3) FROM t"),
    )


def test_a_substring_with_no_length_keeps_the_one_number() raises:
    assert_equal(
        _plan("SELECT substring(g, 2) FROM t"),
        "PROJECT [substring(g, 2) as __expr_0]\n  SCAN t []\n",
    )


def test_a_substring_of_a_number_is_refused_while_it_binds() raises:
    with assert_raises(contains="'substring' reads text"):
        _ = _plan("SELECT substring(a, 1, 2) FROM t")


def test_a_character_count_is_the_one_call_whatever_it_was_written_as() raises:
    assert_equal(
        _plan("SELECT strlen(g) FROM t"),
        "PROJECT [length(g) as __expr_0]\n  SCAN t []\n",
    )


def test_the_three_names_for_a_character_count_build_the_same_plan() raises:
    var want = _plan("SELECT strlen(g) FROM t")
    assert_equal(_plan("SELECT length(g) FROM t"), want)
    assert_equal(_plan("SELECT len(g) FROM t"), want)
    assert_equal(_plan("SELECT STRLEN(g) FROM t"), want)


def test_a_character_count_of_a_number_is_refused_while_it_binds() raises:
    with assert_raises(contains="'length' counts the characters of text"):
        _ = _plan("SELECT strlen(a) FROM t")


def test_a_character_count_of_two_things_is_refused_while_it_binds() raises:
    with assert_raises(contains="'length' takes 1 argument and was given 2"):
        _ = _plan("SELECT strlen(g, g) FROM t")


def test_a_trim_is_the_call_it_was_written_as() raises:
    assert_equal(
        _plan("SELECT trim(g) FROM t"),
        "PROJECT [trim(g) as __expr_0]\n  SCAN t []\n",
    )


def test_the_one_sided_trims_are_calls_of_their_own() raises:
    assert_equal(
        _plan("SELECT ltrim(g) FROM t"),
        "PROJECT [ltrim(g) as __expr_0]\n  SCAN t []\n",
    )
    assert_equal(
        _plan("SELECT rtrim(g) FROM t"),
        "PROJECT [rtrim(g) as __expr_0]\n  SCAN t []\n",
    )


def test_a_trim_carries_the_characters_it_was_asked_to_take_off() raises:
    assert_equal(
        _plan("SELECT trim(g, 'ab') FROM t"),
        "PROJECT [trim(g, ab) as __expr_0]\n  SCAN t []\n",
    )


def test_the_keyword_spelling_of_a_trim_is_the_same_plan() raises:
    var want = _plan("SELECT trim(g, 'x') FROM t")
    assert_equal(_plan("SELECT TRIM(BOTH 'x' FROM g) FROM t"), want)
    assert_equal(_plan("SELECT TRIM('x' FROM g) FROM t"), want)


def test_a_direction_on_a_keyword_trim_picks_the_one_sided_call() raises:
    assert_equal(
        _plan("SELECT TRIM(LEADING FROM g) FROM t"),
        _plan("SELECT ltrim(g) FROM t"),
    )
    assert_equal(
        _plan("SELECT TRIM(TRAILING 'x' FROM g) FROM t"),
        _plan("SELECT rtrim(g, 'x') FROM t"),
    )


def test_a_keyword_trim_with_nothing_before_the_from_is_a_plain_one() raises:
    assert_equal(
        _plan("SELECT TRIM(BOTH FROM g) FROM t"), _plan("SELECT trim(g) FROM t")
    )


def test_a_trim_of_a_number_is_refused_while_it_binds() raises:
    with assert_raises(contains="'trim' reads text and argument 0 is"):
        _ = _plan("SELECT trim(a) FROM t")


def test_a_trim_of_a_set_that_is_not_text_is_refused_while_it_binds() raises:
    with assert_raises(contains="'trim' reads text and argument 1 is"):
        _ = _plan("SELECT trim(g, a) FROM t")


def test_a_trim_of_three_things_is_refused_where_it_is_read() raises:
    # `TRIM` has a grammar rule of its own, so a count it does not take is
    # caught while the call is built rather than while it binds.
    with assert_raises(contains="this one was written with 3"):
        _ = _plan("SELECT trim(g, 'a', 'b') FROM t")


def test_an_ltrim_of_three_things_is_refused_while_it_binds() raises:
    # `LTRIM` is an ordinary call and reaches the binder, which is the other
    # side of the same check.
    with assert_raises(contains="'ltrim' takes one or two arguments"):
        _ = _plan("SELECT ltrim(g, 'a', 'b') FROM t")


def test_a_keyword_trim_that_says_the_set_twice_is_refused() raises:
    with assert_raises(contains="cannot say it again after it"):
        _ = _plan("SELECT TRIM(BOTH 'x' FROM g, 'y') FROM t")


def test_a_search_is_the_one_call_whatever_it_was_written_as() raises:
    assert_equal(
        _plan("SELECT strpos(g, 'a') FROM t"),
        "PROJECT [instr(g, a) as __expr_0]\n  SCAN t []\n",
    )


def test_the_names_for_a_search_build_the_same_plan() raises:
    var want = _plan("SELECT strpos(g, 'a') FROM t")
    assert_equal(_plan("SELECT instr(g, 'a') FROM t"), want)
    assert_equal(_plan("SELECT STRPOS(g, 'a') FROM t"), want)


def test_the_keyword_spelling_of_a_search_reads_the_other_way() raises:
    # `POSITION(x IN y)` looks for x in y, and the call takes the haystack
    # first, so the two are the same search written in opposite orders.
    assert_equal(
        _plan("SELECT POSITION('a' IN g) FROM t"),
        _plan("SELECT strpos(g, 'a') FROM t"),
    )


def test_a_search_of_a_number_is_refused_while_it_binds() raises:
    with assert_raises(contains="'instr' searches text and argument 0 is"):
        _ = _plan("SELECT strpos(a, 'x') FROM t")


def test_a_search_for_a_number_is_refused_while_it_binds() raises:
    with assert_raises(contains="'instr' searches text and argument 1 is"):
        _ = _plan("SELECT strpos(g, a) FROM t")


def test_a_search_of_one_thing_is_refused_while_it_binds() raises:
    with assert_raises(contains="'instr' takes 2 arguments and was given 1"):
        _ = _plan("SELECT strpos(g) FROM t")


def test_an_extract_is_the_date_part_call_duckdb_says_it_is() raises:
    assert_equal(
        _plan("SELECT EXTRACT(YEAR FROM d) FROM w"),
        "PROJECT [date_part(year, d) as __expr_0]\n  SCAN w []\n",
    )


def test_the_three_spellings_of_a_field_build_the_same_plan() raises:
    var want = _plan("SELECT EXTRACT(YEAR FROM d) FROM w")
    assert_equal(_plan("SELECT date_part('year', d) FROM w"), want)
    assert_equal(_plan("SELECT datepart('year', d) FROM w"), want)


def test_the_field_is_folded_down_the_way_a_name_is() raises:
    var want = _plan("SELECT EXTRACT(YEAR FROM d) FROM w")
    assert_equal(_plan("SELECT extract(Year FROM d) FROM w"), want)
    assert_equal(_plan("SELECT extract('YEAR' FROM d) FROM w"), want)

    # The function spelling does not go through the rule that folds the keyword
    # one, so the fold has to happen again where the call is lowered. Without
    # it the field reaches the operator as it was typed and is looked up in a
    # table that holds it in lower case only.
    assert_equal(_plan("SELECT date_part('YEAR', d) FROM w"), want)


def test_a_day_of_week_is_written_as_the_iso_day_read_modulo_seven() raises:
    # The one field the two systems number differently. DuckDB starts the week
    # at Sunday and everything else in here starts it at Monday, so the query
    # asks for the ISO day, which both agree on, and moves it.
    assert_equal(
        _plan("SELECT EXTRACT(DOW FROM d) FROM w"),
        "PROJECT [(date_part(isodow, d)) % 7 as __expr_0]\n  SCAN w []\n",
    )
    assert_equal(
        _plan("SELECT EXTRACT(DAYOFWEEK FROM d) FROM w"),
        _plan("SELECT EXTRACT(DOW FROM d) FROM w"),
    )


def test_a_field_read_off_a_timestamp_is_the_same_call() raises:
    assert_equal(
        _plan("SELECT EXTRACT(HOUR FROM ts) FROM w"),
        "PROJECT [date_part(hour, ts) as __expr_0]\n  SCAN w []\n",
    )


def test_an_extract_of_a_number_is_refused_while_it_binds() raises:
    with assert_raises(contains="reads a date or a timestamp"):
        _ = _plan("SELECT EXTRACT(YEAR FROM n) FROM w")


def test_a_field_nobody_has_a_kernel_for_is_refused_by_name() raises:
    with assert_raises(contains="no field SQL calls epoch"):
        _ = _plan("SELECT EXTRACT(EPOCH FROM ts) FROM w")


def test_a_field_that_is_not_a_name_at_all_is_refused() raises:
    with assert_raises(contains="no field SQL calls nosuch"):
        _ = _plan("SELECT EXTRACT(NOSUCH FROM d) FROM w")


def test_a_field_worked_out_per_row_is_refused() raises:
    with assert_raises(contains="has to be written out"):
        _ = _plan("SELECT date_part(g, d) FROM w, t")


def test_a_date_trunc_is_a_call_of_its_own() raises:
    assert_equal(
        _plan("SELECT date_trunc('month', d) FROM w"),
        "PROJECT [date_trunc(month, d) as __expr_0]\n  SCAN w []\n",
    )


def test_the_two_spellings_of_a_truncation_build_the_same_plan() raises:
    var want = _plan("SELECT date_trunc('month', d) FROM w")
    assert_equal(_plan("SELECT datetrunc('month', d) FROM w"), want)


def test_the_unit_is_folded_down_the_way_a_field_is() raises:
    var want = _plan("SELECT date_trunc('month', d) FROM w")
    assert_equal(_plan("SELECT date_trunc('MONTH', d) FROM w"), want)
    assert_equal(_plan("SELECT DATE_TRUNC('Month', d) FROM w"), want)


def test_a_truncation_of_a_number_is_refused_while_it_binds() raises:
    with assert_raises(contains="truncates a date or a timestamp"):
        _ = _plan("SELECT date_trunc('month', n) FROM w")


def test_a_period_nobody_has_a_unit_for_is_refused_by_name() raises:
    with assert_raises(contains="nothing to truncate to called fortnight"):
        _ = _plan("SELECT date_trunc('fortnight', d) FROM w")


def test_a_field_name_is_not_a_period_even_where_duckdb_takes_one() raises:
    with assert_raises(contains="nothing to truncate to called dayofweek"):
        _ = _plan("SELECT date_trunc('dayofweek', d) FROM w")


def test_a_period_worked_out_per_row_is_refused() raises:
    with assert_raises(contains="has to be written out"):
        _ = _plan("SELECT date_trunc(g, d) FROM w, t")


def test_a_function_the_catalog_has_is_a_kernel_that_is_missing() raises:
    with assert_raises(contains="no kernel for the function upper yet"):
        _ = _plan("SELECT upper(g) FROM t")


def test_an_aggregate_the_catalog_has_is_a_fold_that_is_missing() raises:
    with assert_raises(contains="no fold for the aggregate bit_and yet"):
        _ = _plan("SELECT bit_and(a) FROM t")


def test_a_name_the_catalog_does_not_have_is_not_called_missing() raises:
    # `levenshtein` is a real DuckDB function that the tier 1 table does not
    # carry, so the sentence says what firepanda has and nothing about DuckDB.
    with assert_raises(contains="there is no function named levenshtein here"):
        _ = _plan("SELECT levenshtein(g, g) FROM t")


def test_a_misspelled_name_gets_the_one_it_is_a_typo_for() raises:
    with assert_raises(contains='Did you mean "length"?'):
        _ = _plan("SELECT lenght(g) FROM t")


def test_a_name_that_resembles_nothing_is_refused_without_a_guess() raises:
    var caught = String()
    try:
        _ = _plan("SELECT zzzzzzqq(g) FROM t")
    except e:
        caught = String(e)
    assert_true(
        "there is no function named zzzzzzqq here" in caught,
        "it says the name back",
    )
    assert_true("Did you mean" not in caught, "and guesses nothing")


def test_a_fold_the_catalog_does_not_carry_is_still_folded() raises:
    # `mean` is not in the tier 1 table and DuckDB runs it, so the catalog
    # check has to let it past rather than read absence as a missing name.
    assert_equal(
        _plan("SELECT mean(a) FROM t"),
        _plan("SELECT avg(a) FROM t"),
    )


def test_a_function_name_is_read_without_regard_to_case() raises:
    # The name comes back in lower case whatever it was written in, because the
    # arena holds it folded and the catalog is looked up by the folded name.
    with assert_raises(contains="no kernel for the function upper yet"):
        _ = _plan("SELECT UPPER(g) FROM t")


def test_a_coalesce_is_a_call_of_its_own() raises:
    assert_equal(
        _plan("SELECT coalesce(a, b) FROM t"),
        "PROJECT [coalesce(a, b) as __expr_0]\n  SCAN t []\n",
    )


def test_an_ifnull_is_the_two_argument_coalesce_under_another_name() raises:
    assert_equal(
        _plan("SELECT ifnull(a, b) FROM t"),
        _plan("SELECT coalesce(a, b) FROM t"),
    )


def test_an_ifnull_that_is_not_a_pair_says_what_it_fills() raises:
    with assert_raises(contains="fills one column from one other"):
        _ = _plan("SELECT ifnull(a, b, 1) FROM t")


def test_a_nullif_is_the_conditional_the_standard_defines_it_as() raises:
    # Not an operator of its own. `NULLIF(a, b)` is defined as the CASE, and
    # writing it as the CASE is what makes a null on either side come out right
    # without anything here arranging for it.
    assert_equal(
        _plan("SELECT nullif(a, b) FROM t"),
        "PROJECT [if a == b then null else a as __expr_0]\n  SCAN t []\n",
    )


def test_a_coalesce_over_a_number_and_some_text_is_refused() raises:
    with assert_raises(contains="have to agree on a type"):
        _ = _plan("SELECT coalesce(a, g) FROM t")


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
    # the window has to be the node above it rather than beside it. The inner
    # one is also the `sum(b)` the query already asked for, so both read the one
    # slot the aggregate computes.
    assert_equal(
        _plan("SELECT g, sum(b), sum(sum(b)) OVER () FROM t GROUP BY g"),
        (
            "PROJECT [g, __agg_0 as __expr_1, __win_0 as __expr_2]\n"
            "  WINDOW [sum(__agg_0) over () as __win_0]\n"
            "    AGGREGATE [g] -> [sum(b)]\n"
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
