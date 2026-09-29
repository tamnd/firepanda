"""Tests for `ASOF RIGHT JOIN` and `ASOF FULL JOIN` run end to end through
`sql/run.mojo`.

Every expected column is DuckDB 1.5.5's answer to the same query over the same
two tables, trades `t` and prices `p`:

    t: k  ts        p: k  ts  px
       1  10           1   5  100
       1  25           1  20  200
       2  15           1  30  300
       3   5           2  20  400
                       4   1  500
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


def frame(var names: List[String], var columns: List[AnyArray]) raises -> DataFrame:
    var fields = List[Field]()
    for i in range(len(names)):
        fields.append(Field(names[i], LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def session() raises -> Catalog:
    var catalog = Catalog()
    var trades = List[AnyArray]()
    trades.append(numbers([1, 1, 2, 3]))
    trades.append(numbers([10, 25, 15, 5]))
    catalog.register("t", frame(["k", "ts"], trades^))
    var prices = List[AnyArray]()
    prices.append(numbers([1, 1, 1, 2, 4]))
    prices.append(numbers([5, 20, 30, 20, 1]))
    prices.append(numbers([100, 200, 300, 400, 500]))
    catalog.register("p", frame(["k", "ts", "px"], prices^))
    return catalog^


def shown(sql: StringSlice, column: String) raises -> String:
    """Runs a query and writes one of its columns out, comma separated."""
    var out = run(sql, session())
    var values = out.column(column).into_values()
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


comptime _COLUMNS = "SELECT t.k, t.ts, p.k AS pk, p.ts AS pts, px FROM t "


def test_a_right_one_hands_out_the_prices_no_trade_took() raises:
    var sql = (
        _COLUMNS
        + "ASOF RIGHT JOIN p ON t.k = p.k AND t.ts >= p.ts ORDER BY ALL"
    )
    assert_equal(shown(sql, "k"), "1,1,null,null,null")
    assert_equal(shown(sql, "ts"), "10,25,null,null,null")
    assert_equal(shown(sql, "pk"), "1,1,1,2,4")
    assert_equal(shown(sql, "px"), "100,200,300,400,500")


def test_a_full_one_keeps_both_sides() raises:
    var sql = (
        _COLUMNS + "ASOF FULL JOIN p ON t.k = p.k AND t.ts >= p.ts ORDER BY ALL"
    )
    assert_equal(shown(sql, "k"), "1,1,2,3,null,null,null")
    assert_equal(shown(sql, "ts"), "10,25,15,5,null,null,null")
    assert_equal(shown(sql, "pts"), "5,20,null,null,30,20,1")
    assert_equal(shown(sql, "px"), "100,200,null,null,300,400,500")


def test_a_full_outer_one_counts_every_row_once() raises:
    var sql = (
        "SELECT count(*) AS n, count(t.k) AS nk FROM t ASOF FULL OUTER JOIN p"
        " ON t.k = p.k AND t.ts > p.ts"
    )
    assert_equal(shown(sql, "n"), "7")
    assert_equal(shown(sql, "nk"), "4")


def test_a_filter_above_sees_the_padded_left_side() raises:
    var sql = (
        "SELECT px FROM t ASOF RIGHT JOIN p ON t.k = p.k AND t.ts >= p.ts"
        " WHERE t.k IS NULL ORDER BY px"
    )
    assert_equal(shown(sql, "px"), "300,400,500")


def test_a_forward_one_hands_out_what_nothing_looked_ahead_to() raises:
    var sql = (
        "SELECT px FROM t ASOF RIGHT JOIN p ON t.k = p.k AND t.ts <= p.ts"
        " ORDER BY px"
    )
    assert_equal(shown(sql, "px"), "100,200,300,400,500")


def test_a_right_one_with_using_merges_to_the_right_side() raises:
    # As in any right join, the merged columns are the right side's.
    var sql = "SELECT * FROM t ASOF RIGHT JOIN p USING (k, ts) ORDER BY ALL"
    assert_equal(shown(sql, "k"), "1,1,1,2,4")
    assert_equal(shown(sql, "ts"), "5,20,30,20,1")
    assert_equal(shown(sql, "px"), "100,200,300,400,500")


def test_a_full_one_with_using_is_turned_down() raises:
    with assert_raises(contains="FULL USING"):
        _ = run("SELECT * FROM t ASOF FULL JOIN p USING (k, ts)", session())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
