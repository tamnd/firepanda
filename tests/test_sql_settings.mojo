"""Tests for `SET`, `RESET` and `PRAGMA`, run through
`firepanda.sql.ddl.execute`.

Every expected order and message is DuckDB 1.5.5's answer to the same
statements in the same order.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.dtype.logical import LogicalType
from firepanda.frame.frame import DataFrame
from firepanda.kernel.cast import cast_any
from firepanda.sql.catalog import Catalog
from firepanda.sql.ddl import execute
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


def table_t(dialect: Dialect, mut catalog: Catalog) raises:
    _ = execute(dialect, "CREATE TABLE t(x INTEGER)", catalog)
    _ = execute(dialect, "INSERT INTO t VALUES (1), (NULL), (3)", catalog)


def test_a_setting_that_changes_no_answer_is_taken() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    var statements: List[String] = [
        "SET threads = 4",
        "SET threads TO 1",
        "PRAGMA threads=2",
        "SET GLOBAL memory_limit = '1GB'",
        "RESET threads",
        "PRAGMA enable_verification",
        "PRAGMA verify_parallelism",
        "SET preserve_insertion_order = false",
    ]
    for statement in statements:
        var answer = execute(dialect, statement, catalog)
        assert_equal(answer.width(), 0, statement)
    var rows = execute(dialect, "SELECT x FROM t ORDER BY x", catalog)
    assert_equal(shown(rows, "x"), "1,3,null")


def test_nulls_first_puts_a_null_first_where_nothing_was_said() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    _ = execute(dialect, "SET default_null_order = 'nulls_first'", catalog)
    var rows = execute(dialect, "SELECT x FROM t ORDER BY x", catalog)
    assert_equal(shown(rows, "x"), "null,1,3")
    rows = execute(dialect, "SELECT x FROM t ORDER BY x NULLS LAST", catalog)
    assert_equal(shown(rows, "x"), "1,3,null")


def test_the_postgres_rule_turns_with_the_direction() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    _ = execute(dialect, "SET default_null_order = postgres", catalog)
    var rows = execute(dialect, "SELECT x FROM t ORDER BY x DESC", catalog)
    assert_equal(shown(rows, "x"), "null,3,1")
    rows = execute(dialect, "SELECT x FROM t ORDER BY x", catalog)
    assert_equal(shown(rows, "x"), "1,3,null")


def test_a_reset_puts_the_nulls_back_last() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    _ = execute(dialect, "PRAGMA default_null_order='NULLS FIRST'", catalog)
    _ = execute(dialect, "RESET default_null_order", catalog)
    var rows = execute(dialect, "SELECT x FROM t ORDER BY x", catalog)
    assert_equal(shown(rows, "x"), "1,3,null")


def test_a_descending_default_leaves_a_written_direction_alone() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    _ = execute(dialect, "SET default_order TO 'DESC'", catalog)
    var rows = execute(dialect, "SELECT x FROM t ORDER BY x", catalog)
    assert_equal(shown(rows, "x"), "3,1,null")
    rows = execute(dialect, "SELECT x FROM t ORDER BY x ASC", catalog)
    assert_equal(shown(rows, "x"), "1,3,null")


def test_a_window_and_a_view_follow_the_setting() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    _ = execute(dialect, "CREATE VIEW v AS SELECT x FROM t ORDER BY x", catalog)
    _ = execute(dialect, "SET default_null_order = 'nulls_first'", catalog)
    var rows = execute(dialect, "SELECT * FROM v", catalog)
    assert_equal(shown(rows, "x"), "null,1,3")
    rows = execute(
        dialect,
        "SELECT x, row_number() OVER (ORDER BY x) AS r FROM t ORDER BY r",
        catalog,
    )
    assert_equal(shown(rows, "x"), "null,1,3")
    assert_equal(shown(rows, "r"), "1,2,3")


def test_a_value_a_setting_does_not_take_says_so() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    try:
        _ = execute(dialect, "SET default_order = 'up'", catalog)
        assert_true(False, "the value was taken")
    except e:
        assert_equal(
            String(e),
            (
                "Invalid Input Error: Unrecognized parameter for option"
                ' DEFAULT_ORDER "up". Expected ASC or DESC.'
            ),
        )
    try:
        _ = execute(dialect, "SET default_null_order = 1", catalog)
        assert_true(False, "the value was taken")
    except e:
        assert_equal(
            String(e),
            (
                "Parser Error: Unrecognized parameter for option NULL_ORDER"
                ' "1", expected either NULLS FIRST, NULLS LAST, SQLite, MySQL'
                " or Postgres"
            ),
        )


def test_a_setting_that_would_change_an_answer_is_refused() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    with assert_raises(contains="does not support the setting TimeZone"):
        _ = execute(dialect, "SET TimeZone = 'UTC'", catalog)
    with assert_raises(contains="PRAGMA"):
        _ = execute(dialect, "PRAGMA table_info('t')", catalog)


def refused(
    dialect: Dialect, sql: String, mut catalog: Catalog
) raises -> String:
    """The message a statement raises, or a note that it did not raise."""
    try:
        _ = execute(dialect, sql, catalog)
    except e:
        return String(e)
    return String("no error from ", sql)


comptime NO_WAY_BACK = (
    "Invalid Input Error: Cannot enable external access while database is"
    " running"
)


def test_external_access_turns_off_and_never_back_on() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    assert_equal(
        refused(dialect, "SET enable_external_access = true", catalog),
        NO_WAY_BACK,
    )
    assert_equal(
        refused(dialect, "RESET enable_external_access", catalog), NO_WAY_BACK
    )
    assert_true(catalog.settings.external)
    _ = execute(dialect, "SET enable_external_access = false", catalog)
    assert_true(not catalog.settings.external)
    _ = execute(dialect, "SET enable_external_access = 'no'", catalog)
    _ = execute(dialect, "SET enable_external_access = 0", catalog)
    var ons: List[String] = ["1", "2", "'yes'", "'T'", "true"]
    for on in ons:
        assert_equal(
            refused(
                dialect, String("SET enable_external_access = ", on), catalog
            ),
            NO_WAY_BACK,
        )
    assert_equal(
        refused(dialect, "RESET enable_external_access", catalog), NO_WAY_BACK
    )
    assert_true(not catalog.settings.external)
    assert_equal(shown(execute(dialect, "SELECT 3 AS z", catalog), "z"), "3")


def test_external_access_takes_only_a_boolean() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    assert_equal(
        refused(dialect, "SET enable_external_access = 'on'", catalog),
        (
            "Invalid Input Error: Failed to cast value: Could not convert"
            " string 'on' to BOOL"
        ),
    )
    assert_equal(
        refused(dialect, "SET enable_external_access = ' false'", catalog),
        (
            "Invalid Input Error: Failed to cast value: Could not convert"
            " string ' false' to BOOL"
        ),
    )
    assert_true(catalog.settings.external)


def orders(dialect: Dialect, mut catalog: Catalog) raises:
    _ = execute(dialect, "CREATE TABLE o(k INTEGER, c INTEGER)", catalog)
    _ = execute(
        dialect,
        "INSERT INTO o VALUES (1, 10), (2, 20), (3, 10), (4, 30), (5, 20)",
        catalog,
    )
    _ = execute(dialect, "CREATE TABLE l(k INTEGER, q INTEGER)", catalog)
    _ = execute(
        dialect,
        (
            "INSERT INTO l VALUES (1, 5), (1, 7), (2, 1), (3, 9), (4, 2), (5,"
            " 4), (5, 6)"
        ),
        catalog,
    )


comptime JOINED = (
    "SELECT o.c, sum(l.q) AS s FROM o, l WHERE o.k = l.k AND l.q > 1"
    " GROUP BY o.c ORDER BY s DESC LIMIT 2"
)
"""A join written as a product, a filter to push, a group, a sort and a
limit: something for every pass to do."""


def test_every_pass_turned_off_on_its_own_leaves_the_answer() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    orders(dialect, catalog)
    var rows = execute(dialect, JOINED, catalog)
    assert_equal(shown(rows, "c"), "10,20")
    assert_equal(shown(rows, "s"), "21,10")
    var names: List[String] = [
        "expression_rewriter",
        "empty_result_pullup",
        "unused_columns",
        "join_order",
        "filter_pushdown",
        "common_subexpressions",
        "projection_merge",
        "limit_pushdown",
        "common_subplan",
    ]
    for name in names:
        _ = execute(
            dialect,
            String("SET disabled_optimizers = '", name, "'"),
            catalog,
        )
        rows = execute(dialect, JOINED, catalog)
        assert_equal(shown(rows, "c"), "10,20", name)
        assert_equal(shown(rows, "s"), "21,10", name)


def test_the_optimizer_turned_off_leaves_the_answer() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    orders(dialect, catalog)
    _ = execute(dialect, "PRAGMA disable_optimizer", catalog)
    var rows = execute(dialect, JOINED, catalog)
    assert_equal(shown(rows, "c"), "10,20")
    assert_equal(shown(rows, "s"), "21,10")
    _ = execute(dialect, "PRAGMA enable_optimizer", catalog)
    _ = execute(
        dialect,
        "SET disabled_optimizers = 'filter_pushdown,join_order'",
        catalog,
    )
    rows = execute(dialect, JOINED, catalog)
    assert_equal(shown(rows, "s"), "21,10")
    _ = execute(dialect, "RESET disabled_optimizers", catalog)
    rows = execute(dialect, JOINED, catalog)
    assert_equal(shown(rows, "s"), "21,10")


def test_an_optimizer_name_nobody_has_is_refused() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    with assert_raises(contains='Optimizer type "bogus" not recognized'):
        _ = execute(dialect, "SET disabled_optimizers = 'bogus'", catalog)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
