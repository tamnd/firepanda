"""Tests for `BEGIN`, `COMMIT` and `ROLLBACK`, run through
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


def total(dialect: Dialect, mut catalog: Catalog) raises -> String:
    """The sum of `t.x`."""
    return shown(execute(dialect, "SELECT sum(x) AS s FROM t", catalog), "s")


def started(dialect: Dialect) raises -> Catalog:
    """A catalog holding `t` with the rows 1 and 2."""
    var catalog = Catalog()
    _ = execute(dialect, "CREATE TABLE t(x INTEGER)", catalog)
    _ = execute(dialect, "INSERT INTO t VALUES (1), (2)", catalog)
    return catalog^


def test_a_rollback_undoes_every_change_the_transaction_made() raises:
    var dialect = Dialect()
    var catalog = started(dialect)
    var answer = execute(dialect, "BEGIN", catalog)
    assert_equal(answer.width(), 0)
    _ = execute(dialect, "INSERT INTO t VALUES (3)", catalog)
    assert_equal(total(dialect, catalog), "6")
    _ = execute(dialect, "DROP TABLE t", catalog)
    _ = execute(dialect, "CREATE VIEW v AS SELECT 7 AS y", catalog)
    _ = execute(dialect, "ROLLBACK", catalog)
    assert_equal(total(dialect, catalog), "3")
    assert_true(
        refused(dialect, "SELECT * FROM v", catalog).startswith(
            "Catalog Error: Table with name v does not exist!"
        )
    )


def test_a_commit_keeps_what_the_transaction_did() raises:
    var dialect = Dialect()
    var catalog = started(dialect)
    _ = execute(dialect, "START TRANSACTION", catalog)
    _ = execute(dialect, "INSERT INTO t VALUES (10)", catalog)
    _ = execute(dialect, "END", catalog)
    assert_equal(total(dialect, catalog), "13")
    _ = execute(dialect, "BEGIN TRANSACTION READ WRITE", catalog)
    _ = execute(dialect, "INSERT INTO t VALUES (1000)", catalog)
    _ = execute(dialect, "ABORT", catalog)
    assert_equal(total(dialect, catalog), "13")


def test_a_statement_that_fails_to_bind_leaves_the_transaction_going() raises:
    var dialect = Dialect()
    var catalog = started(dialect)
    _ = execute(dialect, "BEGIN WORK", catalog)
    # The refusal is the plan binding's, in its own words, and it comes before
    # a row is read, so the transaction goes on as DuckDB's does.
    var message = refused(dialect, "SELECT nope FROM t", catalog)
    assert_true("nope" in message, message)
    _ = execute(dialect, "INSERT INTO t VALUES (100)", catalog)
    _ = execute(dialect, "COMMIT WORK", catalog)
    assert_equal(total(dialect, catalog), "103")


def test_a_statement_that_fails_as_it_runs_aborts_the_transaction() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(dialect, "BEGIN", catalog)
    _ = execute(dialect, "CREATE TABLE b(x INTEGER NOT NULL)", catalog)
    assert_equal(
        refused(dialect, "INSERT INTO b VALUES (NULL)", catalog),
        "Constraint Error: NOT NULL constraint failed: b.x",
    )
    assert_equal(
        refused(dialect, "SELECT 3 AS three", catalog),
        (
            "TransactionContext Error: Current transaction is aborted (please"
            " ROLLBACK)"
        ),
    )
    # A commit of an aborted transaction rolls it back without a word.
    _ = execute(dialect, "COMMIT", catalog)
    assert_true(
        refused(dialect, "SELECT * FROM b", catalog).startswith(
            "Catalog Error: Table with name b does not exist!"
        )
    )
    assert_equal(shown(execute(dialect, "SELECT 3 AS x", catalog), "x"), "3")


def test_a_begin_inside_a_transaction_is_refused_and_aborts_it() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(dialect, "BEGIN", catalog)
    _ = execute(dialect, "CREATE TABLE a(x INTEGER)", catalog)
    assert_equal(
        refused(dialect, "BEGIN", catalog),
        (
            "TransactionContext Error: cannot start a transaction within a"
            " transaction"
        ),
    )
    assert_true(
        refused(dialect, "SELECT 1 AS one", catalog).find(
            "Current transaction is aborted"
        )
        >= 0
    )
    _ = execute(dialect, "COMMIT", catalog)
    assert_true(
        refused(dialect, "SELECT * FROM a", catalog).startswith(
            "Catalog Error: Table with name a does not exist!"
        )
    )


def test_an_end_with_no_transaction_is_refused() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    assert_equal(
        refused(dialect, "COMMIT", catalog),
        "TransactionContext Error: cannot commit - no transaction is active",
    )
    assert_equal(
        refused(dialect, "ROLLBACK", catalog),
        "TransactionContext Error: cannot rollback - no transaction is active",
    )
    _ = execute(dialect, "BEGIN", catalog)
    _ = execute(dialect, "COMMIT", catalog)
    assert_equal(
        refused(dialect, "COMMIT", catalog),
        "TransactionContext Error: cannot commit - no transaction is active",
    )


def test_a_read_only_transaction_refuses_a_write() raises:
    var dialect = Dialect()
    var catalog = started(dialect)
    _ = execute(dialect, "BEGIN READ ONLY", catalog)
    assert_equal(
        shown(execute(dialect, "SELECT count(*) AS n FROM t", catalog), "n"),
        "2",
    )
    assert_equal(
        refused(dialect, "INSERT INTO t VALUES (1)", catalog),
        (
            'TransactionContext Error: Cannot write to database "memory" -'
            " transaction is launched in read-only mode"
        ),
    )
    assert_true(
        refused(dialect, "SELECT 1 AS one", catalog).find(
            "Current transaction is aborted"
        )
        >= 0
    )
    _ = execute(dialect, "COMMIT", catalog)
    assert_equal(total(dialect, catalog), "3")
    _ = execute(dialect, "INSERT INTO t VALUES (4)", catalog)
    assert_equal(total(dialect, catalog), "7")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
