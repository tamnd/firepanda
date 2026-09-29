"""Tests for an `ASOF` join written with `USING`, run end to end through
`sql/run.mojo`.

Every expected column is DuckDB 1.5.5's answer to the same query over the same
two tables, trades `t` and prices `p`:

    t          p
    k  ts      k  ts  px
    1  10      1  5   a
    1  20      1  12  b
    2  15      2  20  c
               3  1   d
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


from firepanda.array.strings import StringBuilder


def texts(values: List[String]) raises -> AnyArray:
    var text = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        text.append(values[i].as_bytes())
    return AnyArray(text^.finish())


def trades() raises -> DataFrame:
    var columns = List[AnyArray]()
    columns.append(numbers([1, 1, 2]))
    columns.append(numbers([10, 20, 15]))
    var fields = List[Field]()
    fields.append(Field("k", LogicalType.INT64))
    fields.append(Field("ts", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def prices() raises -> DataFrame:
    var columns = List[AnyArray]()
    columns.append(numbers([1, 1, 2, 3]))
    columns.append(numbers([5, 12, 20, 1]))
    columns.append(texts(["a", "b", "c", "d"]))
    var fields = List[Field]()
    fields.append(Field("k", LogicalType.INT64))
    fields.append(Field("ts", LogicalType.INT64))
    fields.append(Field("px", LogicalType.STRING))
    return DataFrame(Schema(fields^), columns^)


def session() raises -> Catalog:
    var catalog = Catalog()
    catalog.register("t", trades())
    catalog.register("p", prices())
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


def test_the_last_column_named_is_the_inequality() raises:
    var sql = "SELECT * FROM t ASOF JOIN p USING (k, ts) ORDER BY ALL"
    assert_equal(shown(sql, "k"), "1,1")
    assert_equal(shown(sql, "ts"), "10,20")
    assert_equal(shown(sql, "px"), "a,b")


def test_a_star_writes_each_named_column_once() raises:
    var out = run("SELECT * FROM t ASOF JOIN p USING (k, ts)", session())
    assert_equal(out.width(), 3)


def test_each_side_can_still_be_read_by_name() raises:
    var sql = (
        "SELECT t.ts AS x, p.ts AS y FROM t ASOF JOIN p USING (k, ts)"
        " ORDER BY ALL"
    )
    assert_equal(shown(sql, "x"), "10,20")
    assert_equal(shown(sql, "y"), "5,12")


def test_a_left_one_keeps_the_trade_with_no_price() raises:
    var sql = "SELECT * FROM t ASOF LEFT JOIN p USING (k, ts) ORDER BY ALL"
    assert_equal(shown(sql, "k"), "1,1,2")
    assert_equal(shown(sql, "ts"), "10,20,15")
    assert_equal(shown(sql, "px"), "a,b,")


def test_one_column_is_the_inequality_alone() raises:
    var sql = "SELECT t.k AS x, px FROM t ASOF JOIN p USING (ts) ORDER BY ALL"
    assert_equal(shown(sql, "x"), "1,1,2")
    assert_equal(shown(sql, "px"), "a,c,b")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
