"""Tests for `unnest` over a list written out, run end to end through
`sql/run.mojo`.

Every expected column is DuckDB 1.5.5's answer to the same query over the same
table `t`, one column `x` holding 1 and 3.
"""

from std.testing import TestSuite, assert_equal, assert_raises

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame
from firepanda.kernel.cast import cast_any
from firepanda.sql.catalog import Catalog
from firepanda.sql.run import run


def numbers(values: List[Int64]) raises -> AnyArray:
    var out = Array[DType.int64](len(values))
    for i in range(len(values)):
        out[i] = values[i]
    return AnyArray(out^)


def table() raises -> DataFrame:
    var columns = List[AnyArray]()
    columns.append(numbers([1, 3]))
    var fields = List[Field]()
    fields.append(Field("x", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def session() raises -> Catalog:
    var catalog = Catalog()
    catalog.register("t", table())
    return catalog^


def shown(sql: StringSlice, column: String = "x") raises -> String:
    """Runs a query and writes one of its columns out, comma separated."""
    var out = run(sql, session())
    var values = out.column(column).into_values()
    var text = String()
    if values.is_string():
        for i in range(len(values)):
            if i > 0:
                text += ","
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


def test_a_list_comes_out_a_row_per_element() raises:
    assert_equal(shown("SELECT unnest([1, 2, 3]) AS x"), "1,2,3")


def test_each_row_is_written_once_per_element() raises:
    var sql = "SELECT x, unnest([x, x * 2]) AS y FROM t"
    assert_equal(shown(sql, "x"), "1,1,3,3")
    assert_equal(shown(sql, "y"), "1,2,3,6")


def test_a_shorter_list_is_padded_with_nulls() raises:
    var sql = "SELECT unnest([1, 2, 3]) AS x, unnest([10, 20]) AS y"
    assert_equal(shown(sql, "x"), "1,2,3")
    assert_equal(shown(sql, "y"), "10,20,null")


def test_an_empty_list_writes_no_rows() raises:
    assert_equal(
        shown("SELECT count(*) AS x FROM (SELECT unnest([]) AS u)"), "0"
    )


def test_a_null_is_a_list_of_nothing() raises:
    assert_equal(
        shown("SELECT count(*) AS x FROM (SELECT unnest(NULL) AS u)"), "0"
    )


def test_a_null_element_is_a_null_row() raises:
    assert_equal(shown("SELECT unnest(['a', NULL]) AS x"), "a,")


def test_the_where_runs_before_the_unnest() raises:
    assert_equal(
        shown("SELECT unnest([x, 10]) AS x FROM t WHERE x > 1"), "3,10"
    )


def test_the_unnest_can_be_ordered_by() raises:
    assert_equal(
        shown("SELECT x, unnest([1, 2]) AS u FROM t ORDER BY u, x"), "1,3,1,3"
    )


def test_an_unnest_is_a_value_like_any_other() raises:
    assert_equal(shown("SELECT unnest([1, 2]) + 1 AS x"), "2,3")


def test_a_query_over_the_unnest_folds_it() raises:
    assert_equal(
        shown("SELECT sum(u) AS x FROM (SELECT unnest([x, 1]) AS u FROM t)"),
        "6",
    )


def test_the_column_is_named_after_the_list_value_call() raises:
    var out = run("SELECT unnest([1, 2, 3]), unnest([1, 2]) + 1", session())
    assert_equal(out.names()[0], "unnest(main.list_value(1, 2, 3))")
    assert_equal(out.names()[1], "(unnest(main.list_value(1, 2)) + 1)")


def test_unnest_in_the_from_names_its_column_unnest() raises:
    assert_equal(shown("SELECT * FROM unnest([4, 5])", "unnest"), "4,5")


def test_an_unnest_outside_the_select_list_is_turned_down() raises:
    with assert_raises(contains="select list"):
        _ = run("SELECT x FROM t WHERE unnest([1]) = 1", session())


def test_a_column_is_not_a_list_yet() raises:
    with assert_raises(contains="written out"):
        _ = run("SELECT unnest(x) FROM t", session())


def test_an_unnest_inside_an_unnest_is_turned_down() raises:
    with assert_raises(contains="select list"):
        _ = run("SELECT unnest([unnest([1])])", session())


def test_an_unnest_in_an_aggregating_query_is_turned_down() raises:
    with assert_raises(contains="aggregates"):
        _ = run("SELECT count(*), unnest([1, 2]) FROM t", session())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
