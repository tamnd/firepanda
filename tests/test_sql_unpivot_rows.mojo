"""Tests for `UNPIVOT` run end to end through `sql/run.mojo`.

Every expected column is DuckDB 1.5.5's answer to the same query over the same
table `m`, whose blanks are nulls:

    id  jan  feb  mar
    x     1    2
    y     3         6
"""

from std.testing import TestSuite, assert_equal, assert_raises

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
    columns.append(names(["x", "y"]))
    columns.append(numbers([1, 3]))
    columns.append(numbers([2, 0], [1]))
    columns.append(numbers([0, 6], [0]))
    var fields = List[Field]()
    fields.append(Field("id", LogicalType.STRING))
    fields.append(Field("jan", LogicalType.INT64))
    fields.append(Field("feb", LogicalType.INT64, True))
    fields.append(Field("mar", LogicalType.INT64, True))
    var catalog = Catalog()
    catalog.register("m", DataFrame(Schema(fields^), columns^))
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


def test_each_row_hands_out_its_folded_columns_in_order() raises:
    var sql = "UNPIVOT m ON jan, feb, mar"
    assert_equal(columns(sql), "id,name,value")
    assert_equal(shown(sql, "id"), "x,x,y,y")
    assert_equal(shown(sql, "name"), "jan,feb,jan,mar")
    assert_equal(shown(sql, "value"), "1,2,3,6")


def test_the_into_clause_names_the_two_columns() raises:
    var sql = "UNPIVOT m ON jan, feb INTO NAME month VALUE amount"
    assert_equal(columns(sql), "id,mar,month,amount")
    assert_equal(shown(sql, "mar"), "null,null,6")
    assert_equal(shown(sql, "month"), "jan,feb,jan")
    assert_equal(shown(sql, "amount"), "1,2,3")


def test_the_from_spelling_is_the_same_fold() raises:
    var sql = (
        "SELECT id, n, v FROM m UNPIVOT (v FOR n IN (jan, feb, mar))"
        " ORDER BY ALL"
    )
    assert_equal(shown(sql, "id"), "x,x,y,y")
    assert_equal(shown(sql, "n"), "feb,jan,jan,mar")
    assert_equal(shown(sql, "v"), "2,1,3,6")


def test_an_expression_is_named_by_its_alias() raises:
    var sql = "UNPIVOT m ON (jan + 1) AS j2, feb"
    assert_equal(shown(sql, "name"), "j2,feb,j2")
    assert_equal(shown(sql, "value"), "2,2,4")


def test_an_expression_with_no_alias_is_named_by_its_column() raises:
    var sql = "UNPIVOT m ON jan + 1, feb"
    assert_equal(shown(sql, "name"), "jan,feb,jan")
    assert_equal(shown(sql, "value"), "2,2,4")


def test_a_query_reads_it_as_a_subquery() raises:
    var sql = (
        "SELECT name, sum(value) AS s FROM (UNPIVOT m ON jan, feb, mar)"
        " GROUP BY name ORDER BY name"
    )
    assert_equal(shown(sql, "name"), "feb,jan,mar")
    assert_equal(shown(sql, "s"), "2,4,6")


def test_an_expression_over_two_columns_is_turned_down() raises:
    with assert_raises(contains="exactly one column"):
        _ = run("UNPIVOT m ON feb + jan, mar", session())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
