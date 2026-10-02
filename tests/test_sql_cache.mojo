"""Tests for the prepared statement cache in `firepanda.sql.cache`.

What a cached plan answers has to be what a fresh one would, so every case
runs the statement through the cache and checks the rows, and then checks the
hit and miss counts to see which way the answer came.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.dtype.logical import LogicalType
from firepanda.frame.frame import DataFrame
from firepanda.kernel.cast import cast_any
from firepanda.sql.cache import PlanCache, is_query
from firepanda.sql.catalog import Catalog
from firepanda.sql.ddl import execute
from firepanda.sql.parameters import Arguments
from firepanda.sql.run import Dialect


def shown(frame: DataFrame, column: String) raises -> String:
    """One column written out, comma separated, with `null` for a gap."""
    var values = frame.column(column).into_values()
    var wide = cast_any(values, LogicalType.INT64, strict=False)
    var typed = wide.as_typed[DType.int64]()
    var text = String()
    for i in range(len(wide)):
        if i > 0:
            text += ","
        if not wide.is_valid(i):
            text += "null"
        else:
            text += String(typed[i])
    return text^


def session(
    mut cache: PlanCache,
    dialect: Dialect,
    sql: String,
    mut catalog: Catalog,
) raises -> DataFrame:
    """Runs a statement the way a session does, keyed on the generation."""
    return cache.answer(dialect, sql, catalog, String(catalog.generation()))


def test_a_query_run_twice_is_planned_once() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    var cache = PlanCache()
    _ = session(
        cache, dialect, "CREATE TABLE t AS SELECT * FROM range(4) r(i)", catalog
    )
    var sql = "SELECT sum(i) AS s FROM t WHERE i > 0"
    assert_equal(shown(session(cache, dialect, sql, catalog), "s"), "6")
    assert_equal(shown(session(cache, dialect, sql, catalog), "s"), "6")
    assert_equal(shown(session(cache, dialect, sql, catalog), "s"), "6")
    assert_equal(cache.misses, 1)
    assert_equal(cache.hits, 2)
    assert_equal(len(cache), 1)


def test_a_statement_that_changes_the_catalog_is_not_kept() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    var cache = PlanCache()
    _ = session(cache, dialect, "CREATE TABLE t(i BIGINT)", catalog)
    _ = session(cache, dialect, "INSERT INTO t VALUES (1)", catalog)
    _ = session(cache, dialect, "SET default_order = 'desc'", catalog)
    assert_equal(cache.misses, 0)
    assert_equal(cache.hits, 0)
    assert_equal(len(cache), 0)


def test_a_change_to_the_catalog_plans_again_and_reads_the_new_rows() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    var cache = PlanCache()
    _ = session(cache, dialect, "CREATE TABLE t(i BIGINT)", catalog)
    _ = session(cache, dialect, "INSERT INTO t VALUES (1), (2)", catalog)
    var sql = "SELECT i FROM t ORDER BY i"
    assert_equal(shown(session(cache, dialect, sql, catalog), "i"), "1,2")
    _ = session(cache, dialect, "INSERT INTO t VALUES (3)", catalog)
    assert_equal(shown(session(cache, dialect, sql, catalog), "i"), "1,2,3")
    assert_equal(cache.misses, 2)
    # A table dropped and made again with other columns binds afresh.
    _ = session(cache, dialect, "DROP TABLE t", catalog)
    _ = session(
        cache, dialect, "CREATE TABLE t AS SELECT 'x' AS j, 5 AS i", catalog
    )
    assert_equal(shown(session(cache, dialect, sql, catalog), "i"), "5")
    assert_equal(cache.misses, 3)
    assert_equal(cache.hits, 0)


def test_a_setting_plans_again() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    var cache = PlanCache()
    var sql = "SELECT * FROM range(3) r(i) ORDER BY i"
    assert_equal(shown(session(cache, dialect, sql, catalog), "i"), "0,1,2")
    _ = session(cache, dialect, "SET default_order = 'desc'", catalog)
    assert_equal(shown(session(cache, dialect, sql, catalog), "i"), "2,1,0")
    _ = session(cache, dialect, "RESET default_order", catalog)
    assert_equal(shown(session(cache, dialect, sql, catalog), "i"), "0,1,2")
    assert_equal(cache.hits, 0)


def build(dialect: Dialect, sql: String) raises -> Catalog:
    """A catalog holding what one statement made."""
    var catalog = Catalog()
    _ = execute(dialect, sql, catalog)
    return catalog^


def test_a_catalog_of_the_same_shape_reuses_the_plan() raises:
    """What `firepanda.sql` does: a fresh catalog per call, keyed on shape."""
    var dialect = Dialect()
    var cache = PlanCache()
    var sql = "SELECT max(i) AS m FROM t"
    var first = build(dialect, "CREATE TABLE t AS SELECT * FROM range(3) r(i)")
    assert_equal(
        shown(cache.answer(dialect, sql, first, first.shape()), "m"), "2"
    )
    var second = build(
        dialect, "CREATE TABLE t AS SELECT * FROM range(10) r(i)"
    )
    assert_equal(
        shown(cache.answer(dialect, sql, second, second.shape()), "m"), "9"
    )
    assert_equal(cache.hits, 1)
    var other = build(
        dialect, "CREATE TABLE t AS SELECT CAST(i AS INTEGER) AS i FROM range(4) r(i)"
    )
    assert_true(other.shape() != first.shape())
    assert_equal(
        shown(cache.answer(dialect, sql, other, other.shape()), "m"), "3"
    )
    assert_equal(cache.misses, 2)


def test_the_values_an_execute_passed_are_in_the_key() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    var cache = PlanCache()
    var sql = "SELECT ? + 1 AS x"
    var names = List[String]()
    names.append("1")
    var texts = List[String]()
    texts.append("41")
    _ = catalog.bind(Arguments(names.copy(), texts^))
    assert_equal(shown(session(cache, dialect, sql, catalog), "x"), "42")
    texts = List[String]()
    texts.append("1")
    _ = catalog.bind(Arguments(names^, texts^))
    assert_equal(shown(session(cache, dialect, sql, catalog), "x"), "2")
    assert_equal(cache.misses, 2)
    assert_equal(cache.hits, 0)


def test_a_query_that_fails_says_what_execute_says() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    var cache = PlanCache()
    var direct = String()
    try:
        _ = execute(dialect, "SELECT nope FROM missing", catalog)
    except e:
        direct = String(e)
    var cached = String()
    try:
        _ = session(cache, dialect, "SELECT nope FROM missing", catalog)
    except e:
        cached = String(e)
    assert_true(direct.byte_length() > 0)
    assert_equal(cached, direct)
    assert_equal(len(cache), 0)


def test_a_transaction_is_left_to_execute() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    var cache = PlanCache()
    _ = session(cache, dialect, "BEGIN", catalog)
    _ = session(cache, dialect, "SELECT 1 AS one", catalog)
    _ = session(cache, dialect, "COMMIT", catalog)
    assert_equal(len(cache), 0)
    _ = session(cache, dialect, "SELECT 1 AS one", catalog)
    assert_equal(len(cache), 1)


def test_which_statements_are_queries() raises:
    assert_true(is_query("SELECT 1"))
    assert_true(is_query("  select 1"))
    assert_true(is_query("-- a note\n/* and another */ WITH x AS (SELECT 1) FROM x"))
    assert_true(is_query("(SELECT 1) UNION (SELECT 2)"))
    assert_true(is_query("VALUES (1)"))
    assert_true(is_query("FROM t"))
    assert_true(is_query("pivot t ON a USING sum(b)"))
    assert_false(is_query("CREATE TABLE t AS SELECT 1"))
    assert_false(is_query("INSERT INTO t SELECT 1"))
    assert_false(is_query("SET threads = 1"))
    assert_false(is_query("EXPLAIN SELECT 1"))
    assert_false(is_query("EXECUTE q(1)"))
    assert_false(is_query("SELECTED"))
    assert_false(is_query(""))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
