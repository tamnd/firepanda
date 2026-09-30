"""Tests for `PREPARE`, `EXECUTE` and `DEALLOCATE`, run through
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


def test_each_value_goes_where_its_parameter_was() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    var answer = execute(
        dialect, "PREPARE q AS SELECT ? + 1 AS x, ? AS y", catalog
    )
    assert_equal(answer.width(), 0)
    answer = execute(dialect, "EXECUTE q(41, 7)", catalog)
    assert_equal(shown(answer, "x"), "42")
    assert_equal(shown(answer, "y"), "7")
    answer = execute(dialect, "EXECUTE q(1 + 1, -5)", catalog)
    assert_equal(shown(answer, "x"), "3")
    assert_equal(shown(answer, "y"), "-5")


def test_a_numbered_parameter_is_read_as_often_as_it_is_written() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(dialect, "PREPARE q AS SELECT $1 AS x, $1 AS y", catalog)
    var answer = execute(dialect, "EXECUTE q(7)", catalog)
    assert_equal(shown(answer, "x"), "7")
    assert_equal(shown(answer, "y"), "7")
    _ = execute(dialect, "PREPARE q AS SELECT $2 - $1 AS d", catalog)
    answer = execute(dialect, "EXECUTE q(1, 10)", catalog)
    assert_equal(shown(answer, "d"), "9")


def test_a_named_parameter_takes_the_value_passed_by_its_name() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(dialect, "PREPARE q AS SELECT $a * 2 AS x", catalog)
    var answer = execute(dialect, "EXECUTE q(a := 21)", catalog)
    assert_equal(shown(answer, "x"), "42")
    answer = execute(dialect, "EXECUTE q(A := 4)", catalog)
    assert_equal(shown(answer, "x"), "8")


def test_a_prepared_query_reads_the_tables_as_they_are_when_run() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(dialect, "CREATE TABLE t(x INTEGER)", catalog)
    _ = execute(
        dialect, "PREPARE ins AS INSERT INTO t VALUES ($1), ($1 + 10)", catalog
    )
    _ = execute(dialect, "EXECUTE ins(1)", catalog)
    _ = execute(dialect, "EXECUTE ins(2)", catalog)
    _ = execute(
        dialect,
        "PREPARE q AS SELECT x FROM t WHERE x > $1 ORDER BY x LIMIT $2",
        catalog,
    )
    var answer = execute(dialect, "EXECUTE q(1, 2)", catalog)
    assert_equal(shown(answer, "x"), "2,11")
    _ = execute(dialect, "INSERT INTO t VALUES (5)", catalog)
    answer = execute(dialect, "EXECUTE q(1, 5)", catalog)
    assert_equal(shown(answer, "x"), "2,5,11,12")


def test_a_prepare_again_replaces_and_a_deallocate_forgets() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(dialect, "PREPARE q AS SELECT 1 AS x", catalog)
    _ = execute(dialect, "PREPARE q AS SELECT 2 AS x", catalog)
    assert_equal(shown(execute(dialect, "EXECUTE q", catalog), "x"), "2")
    _ = execute(dialect, "DEALLOCATE q", catalog)
    assert_equal(
        refused(dialect, "EXECUTE q", catalog),
        'Binder Error: Prepared statement "q" does not exist',
    )
    _ = execute(dialect, "DEALLOCATE PREPARE nope", catalog)


def test_a_value_missing_or_left_over_says_which() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(dialect, "PREPARE q AS SELECT ? AS x, ? AS y", catalog)
    assert_equal(
        refused(dialect, "EXECUTE q(1)", catalog),
        (
            "Invalid Input Error: Values were not provided for the following"
            " prepared statement parameters: 2"
        ),
    )
    _ = execute(dialect, "PREPARE q AS SELECT $2 AS x", catalog)
    assert_equal(
        refused(dialect, "EXECUTE q(1, 2, 3)", catalog),
        (
            "Invalid Input Error: Parameter argument/count mismatch,"
            " identifiers of the excess parameters: 1, 3"
        ),
    )
    _ = execute(dialect, "PREPARE q AS SELECT $a AS x", catalog)
    assert_equal(
        refused(dialect, "EXECUTE q(b := 2)", catalog),
        (
            "Invalid Input Error: Values were not provided for the following"
            " prepared statement parameters: a"
        ),
    )
    assert_equal(
        refused(dialect, "EXECUTE q(a := 1, b := 2)", catalog),
        (
            "Invalid Input Error: Parameter argument/count mismatch,"
            " identifiers of the excess parameters: b"
        ),
    )


def test_a_question_mark_counts_on_from_the_numbers_before_it() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    _ = execute(dialect, "PREPARE q AS SELECT $1 + 1 AS x, ? AS y", catalog)
    var answer = execute(dialect, "EXECUTE q(1, 2)", catalog)
    assert_equal(shown(answer, "x"), "2")
    assert_equal(shown(answer, "y"), "2")
    _ = execute(dialect, "PREPARE q AS SELECT ? AS x, $a AS y", catalog)
    assert_equal(
        refused(dialect, "EXECUTE q(1, a := 2)", catalog),
        (
            "Not implemented Error: Mixing named and positional parameters is"
            " not supported yet"
        ),
    )


def test_a_parameter_outside_an_execute_is_refused() raises:
    var dialect = Dialect()
    var catalog = Catalog()
    var message = refused(dialect, "SELECT $1", catalog)
    assert_true(
        message.find("Prepared statement parameters cannot be used directly")
        >= 0,
        message,
    )
    # A failed EXECUTE leaves nothing bound behind it.
    _ = execute(dialect, "PREPARE q AS SELECT ? AS x", catalog)
    _ = refused(dialect, "EXECUTE q(1, 2)", catalog)
    message = refused(dialect, "SELECT ?", catalog)
    assert_true(message.find("cannot be used directly") >= 0, message)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
