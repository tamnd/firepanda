"""Tests for `CREATE TABLE`, `CREATE VIEW`, `INSERT` and `DROP`, run through
`firepanda.sql.ddl.execute`.

Every expected value and message is DuckDB 1.5.5's answer to the same
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
    var text = String()
    if values.is_string():
        for i in range(len(values)):
            if i > 0:
                text += ","
            if not values.is_valid(i):
                text += "null"
            else:
                text += values.text_at(i)
        return text^
    var wide = cast_any(values, LogicalType.INT64, strict=False)
    var typed = wide.as_typed[DType.int64]()
    for i in range(len(wide)):
        if i > 0:
            text += ","
        if not wide.is_valid(i):
            text += "null"
        else:
            text += String(typed[i])
    return text^


def table_t(dialect: Dialect, mut catalog: Catalog) raises:
    _ = execute(dialect, "CREATE TABLE t(a INTEGER, b VARCHAR)", catalog)
    _ = execute(dialect, "INSERT INTO t VALUES (1, 'x'), (2, 'y')", catalog)


def test_a_created_table_takes_rows_and_answers_them() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(dialect, "CREATE TABLE t(a INTEGER, b VARCHAR)", catalog)
    var added = execute(
        dialect, "INSERT INTO t VALUES (1, 'x'), (2, 'y')", catalog
    )
    assert_equal(shown(added, "Count"), "2")
    var rows = execute(dialect, "SELECT a, b FROM t ORDER BY a", catalog)
    assert_equal(shown(rows, "a"), "1,2")
    assert_equal(shown(rows, "b"), "x,y")


def test_a_new_table_has_its_columns_and_no_rows() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(dialect, "CREATE TABLE t(a INTEGER, b VARCHAR)", catalog)
    var rows = execute(dialect, "SELECT count(*) AS n FROM t", catalog)
    assert_equal(shown(rows, "n"), "0")
    var typed = catalog.frame_at(catalog.find("t")).copy()
    assert_true(typed.schema[0].dtype == LogicalType.INT32)
    assert_true(typed.schema[1].dtype == LogicalType.STRING)


def test_inserts_add_up_in_the_order_they_came() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    _ = execute(
        dialect, "INSERT INTO t BY NAME SELECT 'w' AS b, 5 AS a", catalog
    )
    _ = execute(dialect, "INSERT INTO t VALUES ('7', 3)", catalog)
    _ = execute(dialect, "INSERT INTO t (b) VALUES ('z')", catalog)
    _ = execute(dialect, "INSERT INTO t DEFAULT VALUES", catalog)
    var rows = execute(dialect, "SELECT a, b FROM t", catalog)
    assert_equal(shown(rows, "a"), "1,2,5,7,null,null")
    assert_equal(shown(rows, "b"), "x,y,w,3,z,null")


def test_an_insert_from_a_query_reads_another_table() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    _ = execute(dialect, "CREATE TABLE s(n BIGINT)", catalog)
    var added = execute(
        dialect, "INSERT INTO s SELECT a * 10 FROM t WHERE a > 1", catalog
    )
    assert_equal(shown(added, "Count"), "1")
    assert_equal(shown(execute(dialect, "SELECT n FROM s", catalog), "n"), "20")


def test_a_table_made_from_a_query_holds_its_rows() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    _ = execute(dialect, "CREATE TABLE u AS SELECT a * 2 AS d FROM t", catalog)
    _ = execute(
        dialect, "CREATE TABLE u2 (x) AS SELECT 1 AS p, 2 AS q", catalog
    )
    assert_equal(
        shown(execute(dialect, "SELECT d FROM u ORDER BY d", catalog), "d"),
        "2,4",
    )
    var both = execute(dialect, "SELECT * FROM u2", catalog)
    assert_equal(shown(both, "x"), "1")
    assert_equal(shown(both, "q"), "2")


def test_a_view_sees_rows_inserted_after_it() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    _ = execute(
        dialect, "CREATE VIEW v AS SELECT a FROM t WHERE a > 1", catalog
    )
    _ = execute(dialect, "INSERT INTO t VALUES (3, 'z')", catalog)
    assert_equal(
        shown(execute(dialect, "SELECT a FROM v ORDER BY a", catalog), "a"),
        "2,3",
    )


def test_a_view_that_reaches_itself_is_refused_when_it_is_read() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    _ = execute(dialect, "CREATE VIEW v AS SELECT a FROM t", catalog)
    _ = execute(dialect, "CREATE VIEW w AS SELECT a FROM v", catalog)
    _ = execute(dialect, "CREATE OR REPLACE VIEW v AS SELECT a FROM w", catalog)
    with assert_raises(
        contains=(
            "Binder Error: infinite recursion detected: attempting to"
            ' recursively bind view "v"'
        )
    ):
        _ = execute(dialect, "SELECT * FROM v", catalog)


def test_a_view_that_names_nothing_is_refused_when_it_is_made() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    with assert_raises():
        _ = execute(dialect, "CREATE VIEW w AS SELECT zz FROM t", catalog)
    assert_true(catalog.find("w") < 0)


def test_a_name_made_twice_is_refused_unless_it_says_how() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    with assert_raises(
        contains='Catalog Error: Table with name "t" already exists!'
    ):
        _ = execute(dialect, "CREATE TABLE t(a INTEGER)", catalog)
    _ = execute(dialect, "CREATE TABLE IF NOT EXISTS t(c INTEGER)", catalog)
    assert_equal(
        shown(execute(dialect, "SELECT count(*) AS n FROM t", catalog), "n"),
        "2",
    )
    _ = execute(dialect, "CREATE OR REPLACE TABLE t(c INTEGER)", catalog)
    assert_equal(
        shown(execute(dialect, "SELECT count(*) AS n FROM t", catalog), "n"),
        "0",
    )
    _ = execute(dialect, "CREATE VIEW v AS SELECT 1 AS one", catalog)
    with assert_raises(
        contains='Catalog Error: View with name "v" already exists!'
    ):
        _ = execute(dialect, "CREATE VIEW v AS SELECT 2 AS one", catalog)
    with assert_raises(
        contains=(
            "Catalog Error: Existing object t is of type Table, trying to"
            " replace with type View"
        )
    ):
        _ = execute(dialect, "CREATE OR REPLACE VIEW t AS SELECT 1", catalog)


def test_an_insert_that_does_not_fit_is_refused_as_duckdb_refuses_it() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    with assert_raises(
        contains=(
            "Binder Error: table t has 2 columns but 1 values were supplied"
        )
    ):
        _ = execute(dialect, "INSERT INTO t VALUES (1)", catalog)
    with assert_raises(
        contains=(
            "Binder Error: Column name/value mismatch for insert on t: expected"
            " 1 columns but 2 values were supplied"
        )
    ):
        _ = execute(dialect, "INSERT INTO t (a) VALUES (1, 2)", catalog)
    with assert_raises(
        contains='Binder Error: Table "t" does not have a column with name "c"'
    ):
        _ = execute(dialect, "INSERT INTO t (c) VALUES (1)", catalog)
    with assert_raises(contains="Table with name nope does not exist!"):
        _ = execute(dialect, "INSERT INTO nope VALUES (1)", catalog)
    _ = execute(dialect, "CREATE VIEW v AS SELECT 1 AS one", catalog)
    with assert_raises(contains="Catalog Error: v is not an table"):
        _ = execute(dialect, "INSERT INTO v VALUES (1)", catalog)


def test_a_not_null_column_refuses_a_null() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(
        dialect, "CREATE TABLE n(a INTEGER NOT NULL, b INTEGER)", catalog
    )
    with assert_raises(
        contains="Constraint Error: NOT NULL constraint failed: n.a"
    ):
        _ = execute(dialect, "INSERT INTO n VALUES (NULL, 1)", catalog)
    with assert_raises(
        contains="Constraint Error: NOT NULL constraint failed: n.a"
    ):
        _ = execute(dialect, "INSERT INTO n (b) VALUES (1)", catalog)
    _ = execute(dialect, "INSERT INTO n VALUES (1, NULL)", catalog)
    assert_equal(shown(execute(dialect, "SELECT a FROM n", catalog), "a"), "1")


def test_a_drop_takes_the_name_away() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    table_t(dialect, catalog)
    _ = execute(dialect, "CREATE VIEW v AS SELECT a FROM t", catalog)
    with assert_raises(
        contains=(
            "Catalog Error: Existing object t is of type Table, trying to drop"
            " type View"
        )
    ):
        _ = execute(dialect, "DROP VIEW t", catalog)
    _ = execute(dialect, "DROP VIEW v", catalog)
    _ = execute(dialect, "DROP TABLE t", catalog)
    assert_true(catalog.find("t") < 0)
    assert_true(catalog.find("v") < 0)
    with assert_raises(
        contains="Catalog Error: Table with name t does not exist!"
    ):
        _ = execute(dialect, "DROP TABLE t", catalog)
    with assert_raises(
        contains="Catalog Error: View with name v does not exist!"
    ):
        _ = execute(dialect, "DROP VIEW v", catalog)
    _ = execute(dialect, "DROP TABLE IF EXISTS t", catalog)


def test_what_a_frame_cannot_hold_is_refused_by_name() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    with assert_raises(contains="CHECK"):
        _ = execute(dialect, "CREATE TABLE p(a INTEGER CHECK (a > 0))", catalog)
    with assert_raises(contains="a CHECK or a FOREIGN KEY on a table"):
        _ = execute(
            dialect, "CREATE TABLE p(a INTEGER, CHECK (a > 0))", catalog
        )
    with assert_raises(contains="DEFAULT"):
        _ = execute(dialect, "CREATE TABLE p(a INTEGER DEFAULT 1)", catalog)
    assert_true(catalog.find("p") < 0)
    _ = execute(dialect, "CREATE TABLE p(a INTEGER)", catalog)
    with assert_raises(contains="ON CONFLICT"):
        _ = execute(
            dialect, "INSERT INTO p VALUES (1) ON CONFLICT DO NOTHING", catalog
        )


def test_a_query_goes_through_as_a_query() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    assert_equal(shown(execute(dialect, "SELECT 1 AS x", catalog), "x"), "1")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
