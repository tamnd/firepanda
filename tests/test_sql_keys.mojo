"""Tests for `PRIMARY KEY` and `UNIQUE`, run through
`firepanda.sql.ddl.execute`.

Every expected answer and message is DuckDB 1.5.5's to the same statements in
the same order.
"""

from std.testing import TestSuite, assert_equal, assert_true

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


def refused(
    dialect: Dialect, sql: String, mut catalog: Catalog
) raises -> String:
    """The message a statement raises, or a note that it did not raise."""
    try:
        _ = execute(dialect, sql, catalog)
    except e:
        return String(e)
    return String("no error from ", sql)


def test_a_primary_key_refuses_a_row_that_repeats_one() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(
        dialect, "CREATE TABLE t(i INTEGER PRIMARY KEY, j VARCHAR)", catalog
    )
    _ = execute(dialect, "INSERT INTO t VALUES (1, 'a'), (2, 'b')", catalog)
    assert_equal(
        refused(dialect, "INSERT INTO t VALUES (1, 'c')", catalog),
        'Constraint Error: Duplicate key "i: 1" violates primary key constraint.',
    )
    assert_equal(
        refused(dialect, "INSERT INTO t VALUES (3, 'c'), (3, 'd')", catalog),
        (
            "Constraint Error: PRIMARY KEY or UNIQUE constraint violation:"
            ' duplicate key "3"'
        ),
    )
    assert_equal(
        refused(dialect, "INSERT INTO t VALUES (NULL, 'x')", catalog),
        "Constraint Error: NOT NULL constraint failed: t.i",
    )
    var answer = execute(dialect, "SELECT i FROM t ORDER BY i", catalog)
    assert_equal(shown(answer, "i"), "1,2")
    _ = execute(dialect, "INSERT INTO t VALUES (3, 'c')", catalog)
    answer = execute(dialect, "SELECT count(*) AS n FROM t", catalog)
    assert_equal(shown(answer, "n"), "3")


def test_a_key_of_two_columns_is_one_key() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(
        dialect,
        "CREATE TABLE u(a INTEGER, b VARCHAR, PRIMARY KEY (a, b))",
        catalog,
    )
    _ = execute(dialect, "INSERT INTO u VALUES (1, 'x'), (1, 'y')", catalog)
    assert_equal(
        refused(dialect, "INSERT INTO u VALUES (1, 'x')", catalog),
        (
            'Constraint Error: Duplicate key "a: 1, b: x" violates primary key'
            " constraint."
        ),
    )
    assert_equal(
        refused(dialect, "INSERT INTO u VALUES (2, 'q'), (2, 'q')", catalog),
        (
            "Constraint Error: PRIMARY KEY or UNIQUE constraint violation:"
            ' duplicate key "2, q"'
        ),
    )
    # A repeat of a row already there is found before a repeat among the new.
    assert_equal(
        refused(
            dialect,
            "INSERT INTO u VALUES (5, 'e'), (5, 'e'), (1, 'y')",
            catalog,
        ),
        (
            'Constraint Error: Duplicate key "a: 1, b: y" violates primary key'
            " constraint."
        ),
    )


def test_a_unique_lets_any_number_of_nulls_in() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(dialect, "CREATE TABLE w(a INTEGER UNIQUE, b INTEGER)", catalog)
    _ = execute(
        dialect, "INSERT INTO w VALUES (NULL, 1), (NULL, 2), (5, 3)", catalog
    )
    assert_equal(
        refused(dialect, "INSERT INTO w VALUES (5, 9)", catalog),
        'Constraint Error: Duplicate key "a: 5" violates unique constraint.',
    )
    var answer = execute(dialect, "SELECT count(*) AS n FROM w", catalog)
    assert_equal(shown(answer, "n"), "3")


def test_the_keys_are_checked_in_the_order_they_were_written() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(
        dialect,
        "CREATE TABLE w(a INTEGER UNIQUE, b INTEGER PRIMARY KEY)",
        catalog,
    )
    _ = execute(dialect, "INSERT INTO w VALUES (1, 1)", catalog)
    assert_equal(
        refused(dialect, "INSERT INTO w VALUES (1, 1)", catalog),
        'Constraint Error: Duplicate key "a: 1" violates unique constraint.',
    )
    assert_equal(
        refused(
            dialect,
            "INSERT INTO w SELECT 7, 8 UNION ALL SELECT 7, 9",
            catalog,
        ),
        (
            "Constraint Error: PRIMARY KEY or UNIQUE constraint violation:"
            ' duplicate key "7"'
        ),
    )


def test_a_named_key_and_a_table_key_make_their_columns_not_null() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(
        dialect, "CREATE TABLE t5(a INT, CONSTRAINT k PRIMARY KEY (a))", catalog
    )
    assert_equal(
        refused(dialect, "INSERT INTO t5 VALUES (1), (1)", catalog),
        (
            "Constraint Error: PRIMARY KEY or UNIQUE constraint violation:"
            ' duplicate key "1"'
        ),
    )
    _ = execute(
        dialect, "CREATE TABLE t7(a INT, b INT, PRIMARY KEY (b))", catalog
    )
    assert_equal(
        refused(dialect, "INSERT INTO t7 VALUES (1, NULL)", catalog),
        "Constraint Error: NOT NULL constraint failed: t7.b",
    )


def test_text_keys_compare_as_written() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(dialect, "CREATE TABLE t8(s VARCHAR PRIMARY KEY)", catalog)
    _ = execute(dialect, "INSERT INTO t8 VALUES ('Ab')", catalog)
    assert_equal(
        refused(dialect, "INSERT INTO t8 VALUES ('ab'), ('Ab')", catalog),
        'Constraint Error: Duplicate key "s: Ab" violates primary key constraint.',
    )
    _ = execute(dialect, "INSERT INTO t8 VALUES ('ab')", catalog)


def test_a_key_written_wrong_is_refused() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    assert_equal(
        refused(dialect, "CREATE TABLE t(a INT, PRIMARY KEY (b))", catalog),
        'Catalog Error: table "t" does not have a column named "b"',
    )
    assert_equal(
        refused(dialect, "CREATE TABLE t(a INT, PRIMARY KEY (a, a))", catalog),
        'Parser Error: column "a" appears twice in primary key constraint',
    )
    assert_equal(
        refused(
            dialect, "CREATE TABLE y(a INTEGER PRIMARY KEY, PRIMARY KEY (a))", catalog
        ),
        'Parser Error: table "y" has more than one primary key',
    )
    assert_true(catalog.find("t") < 0)
    assert_true(catalog.find("y") < 0)


def test_a_rollback_keeps_the_key() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(dialect, "CREATE TABLE t(i INTEGER PRIMARY KEY)", catalog)
    _ = execute(dialect, "BEGIN", catalog)
    _ = execute(dialect, "INSERT INTO t VALUES (1)", catalog)
    _ = execute(dialect, "ROLLBACK", catalog)
    _ = execute(dialect, "INSERT INTO t VALUES (1)", catalog)
    assert_equal(
        refused(dialect, "INSERT INTO t VALUES (1)", catalog),
        'Constraint Error: Duplicate key "i: 1" violates primary key constraint.',
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
