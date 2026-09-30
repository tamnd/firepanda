"""Tests for `PIVOT` run end to end through `sql/run.mojo`.

Every expected column is DuckDB 1.5.5's answer to the same query over the same
table `s`, whose blank is a null:

    city  yr    prod  amt
    NL    2020  a      10
    NL    2020  b       5
    NL    2021  a       7
    US    2020  a       1
    US    2021  b
    US    2021  c       3
"""

from std.testing import TestSuite, assert_equal

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame
from firepanda.kernel.cast import cast_any
from firepanda.sql.catalog import Catalog
from firepanda.sql.run import run


def numbers(values: List[Int64], nulls: List[Int] = []) raises -> AnyArray:
    var out = Array[DType.int64](len(values))
    for i in range(len(values)):
        out[i] = values[i]
    for i in range(len(nulls)):
        out.set_null(nulls[i])
    return AnyArray(out^)


def names(values: List[String]) raises -> AnyArray:
    var text = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        text.append(values[i].as_bytes())
    return AnyArray(text^.finish())


def session() raises -> Catalog:
    var columns = List[AnyArray]()
    columns.append(names(["NL", "NL", "NL", "US", "US", "US"]))
    columns.append(numbers([2020, 2020, 2021, 2020, 2021, 2021]))
    columns.append(names(["a", "b", "a", "a", "b", "c"]))
    columns.append(numbers([10, 5, 7, 1, 0, 3], [4]))
    var fields = List[Field]()
    fields.append(Field("city", LogicalType.STRING))
    fields.append(Field("yr", LogicalType.INT64))
    fields.append(Field("prod", LogicalType.STRING))
    fields.append(Field("amt", LogicalType.INT64, True))
    var catalog = Catalog()
    catalog.register("s", DataFrame(Schema(fields^), columns^))
    return catalog^


def shown(sql: StringSlice, column: String) raises -> String:
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


def columns(sql: StringSlice) raises -> String:
    """Runs a query and writes its column names out, comma separated."""
    var out = run(sql, session())
    var text = String()
    for i in range(len(out.schema)):
        if i > 0:
            text += ","
        text += out.schema[i].name
    return text^


def test_each_value_is_a_column_of_sums_per_group() raises:
    var sql = (
        "SELECT * FROM (PIVOT s ON yr IN (2020, 2021) USING sum(amt)"
        " GROUP BY city) ORDER BY city"
    )
    assert_equal(columns(sql), "city,2020,2021")
    assert_equal(shown(sql, "city"), "NL,US")
    assert_equal(shown(sql, "2020"), "15,1")
    assert_equal(shown(sql, "2021"), "7,3")


def test_without_a_group_by_every_other_column_groups() raises:
    var sql = (
        "SELECT * FROM (PIVOT s ON yr IN (2020, 2021) USING sum(amt))"
        " ORDER BY ALL"
    )
    assert_equal(columns(sql), "city,prod,2020,2021")
    assert_equal(shown(sql, "city"), "NL,NL,US,US,US")
    assert_equal(shown(sql, "prod"), "a,b,a,b,c")
    assert_equal(shown(sql, "2020"), "10,5,1,null,null")
    assert_equal(shown(sql, "2021"), "7,null,null,null,3")


def test_without_a_using_each_cell_counts_its_rows() raises:
    var sql = (
        "SELECT * FROM (PIVOT s ON yr IN (2020, 2021) GROUP BY city)"
        " ORDER BY city"
    )
    assert_equal(shown(sql, "2020"), "2,1")
    assert_equal(shown(sql, "2021"), "1,2")


def test_an_aliased_aggregate_is_named_after_the_value_and_the_alias() raises:
    var sql = (
        "SELECT * FROM (PIVOT s ON yr IN (2020, 2021) USING sum(amt) AS"
        " total, max(amt) AS top GROUP BY city) ORDER BY city"
    )
    assert_equal(columns(sql), "city,2020_total,2020_top,2021_total,2021_top")
    assert_equal(shown(sql, "2020_total"), "15,1")
    assert_equal(shown(sql, "2020_top"), "10,1")
    assert_equal(shown(sql, "2021_total"), "7,3")
    assert_equal(shown(sql, "2021_top"), "7,3")


def test_two_aggregates_without_aliases_are_named_as_printed() raises:
    var sql = (
        "SELECT * FROM (PIVOT s ON yr IN (2020, 2021) USING sum(amt),"
        " max(amt) GROUP BY city) ORDER BY city"
    )
    assert_equal(
        columns(sql),
        "city,2020_sum(amt),2020_max(amt),2021_sum(amt),2021_max(amt)",
    )
    assert_equal(shown(sql, "2020_max(amt)"), "10,1")


def test_two_pivot_columns_make_a_cell_for_each_pair() raises:
    var sql = (
        "SELECT * FROM (PIVOT s ON yr IN (2020, 2021), prod IN ('a', 'b')"
        " USING sum(amt) GROUP BY city) ORDER BY city"
    )
    assert_equal(columns(sql), "city,2020_a,2020_b,2021_a,2021_b")
    assert_equal(shown(sql, "2020_a"), "10,1")
    assert_equal(shown(sql, "2020_b"), "5,null")
    assert_equal(shown(sql, "2021_a"), "7,null")
    assert_equal(shown(sql, "2021_b"), "null,null")


def test_an_aliased_value_names_its_column() raises:
    var sql = (
        "SELECT * FROM (PIVOT s ON yr IN (2020 AS y0, 2021) USING sum(amt)"
        " GROUP BY city) ORDER BY city"
    )
    assert_equal(columns(sql), "city,y0,2021")
    assert_equal(shown(sql, "y0"), "15,1")


def test_a_value_no_row_holds_is_a_column_of_nulls() raises:
    var sql = (
        "SELECT * FROM (PIVOT s ON yr IN (2020, 2022) USING sum(amt)"
        " GROUP BY city) ORDER BY city"
    )
    assert_equal(shown(sql, "2022"), "null,null")


def test_the_from_spelling_is_the_same_pivot() raises:
    var sql = (
        "SELECT * FROM s PIVOT (sum(amt) FOR yr IN (2020, 2021) GROUP BY"
        " city) ORDER BY city"
    )
    assert_equal(columns(sql), "city,2020,2021")
    assert_equal(shown(sql, "2020"), "15,1")
    assert_equal(shown(sql, "2021"), "7,3")


def test_a_pivot_column_without_a_list_pivots_on_its_values() raises:
    var sql = (
        "SELECT * FROM (PIVOT s ON yr USING sum(amt) GROUP BY city) ORDER BY"
        " city"
    )
    assert_equal(columns(sql), "city,2020,2021")
    assert_equal(shown(sql, "2020"), "15,1")
    assert_equal(shown(sql, "2021"), "7,3")


def test_the_values_read_from_data_are_in_order_and_leave_nulls_out() raises:
    var sql = (
        "SELECT * FROM (PIVOT s ON prod USING sum(amt) GROUP BY city) ORDER BY"
        " city"
    )
    assert_equal(columns(sql), "city,a,b,c")
    assert_equal(shown(sql, "a"), "17,1")
    assert_equal(shown(sql, "b"), "5,null")
    assert_equal(shown(sql, "c"), "null,3")


def test_two_columns_read_from_data_pivot_on_every_pair() raises:
    var sql = (
        "SELECT * FROM (PIVOT s ON yr, prod USING count(*) GROUP BY city)"
        " ORDER BY city"
    )
    assert_equal(
        columns(sql), "city,2020_a,2020_b,2020_c,2021_a,2021_b,2021_c"
    )
    assert_equal(shown(sql, "2020_a"), "1,1")
    assert_equal(shown(sql, "2021_c"), "0,1")


def test_a_column_read_from_no_rows_leaves_only_the_groups() raises:
    var sql = "SELECT * FROM (PIVOT (SELECT * FROM s WHERE yr > 3000) ON yr)"
    assert_equal(columns(sql), "city,prod,amt")
    assert_equal(shown(sql, "city"), "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
