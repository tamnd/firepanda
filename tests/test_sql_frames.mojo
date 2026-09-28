"""Windows with an ORDER BY, a frame, or a function of their own, end to end.

Every expected column is what DuckDB 1.5.5 answers for the same query over the
same rows, read back in the order the rows were stored in. Two frames: `sales`,
ten rows in three chunks with the two shops alternating, and `gappy`, six rows
with a repeat and two nulls in the one column.

Ties are part of the point. Where two rows tie on the ORDER BY, DuckDB numbers
them in the order they were stored, and so does firepanda, because the sort
under the window is stable.
"""

from std.testing import TestSuite, assert_equal, assert_raises

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame
from firepanda.kernel.cast import cast_any
from firepanda.sql.catalog import Catalog
from firepanda.sql.run import run


def numbers(values: List[Int64]) raises -> AnyArray:
    """Builds a fully valid int64 array."""
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return AnyArray(col^)


def sales() raises -> DataFrame:
    """Ten rows in three chunks: a quantity, a price and which shop sold it."""
    var qty = ChunkedArray(LogicalType.INT64)
    qty.append(numbers([5, 20, 3]))
    qty.append(numbers([40, 12, 8, 25]))
    qty.append(numbers([1, 30, 15]))
    var price = ChunkedArray(LogicalType.INT64)
    price.append(numbers([10, 2, 7]))
    price.append(numbers([1, 5, 9, 3]))
    price.append(numbers([100, 4, 6]))
    var shop = ChunkedArray(LogicalType.INT64)
    shop.append(numbers([1, 2, 1]))
    shop.append(numbers([2, 1, 2, 1]))
    shop.append(numbers([2, 1, 2]))
    var columns = List[ChunkedArray]()
    columns.append(qty^)
    columns.append(price^)
    columns.append(shop^)
    var fields = List[Field]()
    fields.append(Field("qty", LogicalType.INT64))
    fields.append(Field("price", LogicalType.INT64))
    fields.append(Field("shop", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def gappy() raises -> DataFrame:
    """Six rows in two chunks, `i` counting them and `mark` with two nulls."""
    var i = ChunkedArray(LogicalType.INT64)
    i.append(numbers([0, 1, 2]))
    i.append(numbers([3, 4, 5]))
    var first = Array[DType.int64](3)
    first.set_valid(0, 4)
    first.set_valid(1, 4)
    first.set_null(2)
    var second = Array[DType.int64](3)
    second.set_valid(0, 9)
    second.set_null(1)
    second.set_valid(2, 1)
    var mark = ChunkedArray(LogicalType.INT64)
    mark.append(AnyArray(first^))
    mark.append(AnyArray(second^))
    var columns = List[ChunkedArray]()
    columns.append(i^)
    columns.append(mark^)
    var fields = List[Field]()
    fields.append(Field("i", LogicalType.INT64))
    fields.append(Field("mark", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def session() raises -> Catalog:
    var catalog = Catalog()
    catalog.register("sales", sales())
    catalog.register("gappy", gappy())
    return catalog^


def shown(sql: StringSlice, scale: Float64 = 1) raises -> String:
    """Runs a query and writes its column `x` out, each value times `scale`."""
    var out = run(sql, session())
    var wide = cast_any(
        out.column("x").into_values(), LogicalType.FLOAT64, strict=False
    )
    var typed = wide.as_typed[DType.float64]()
    var text = String()
    for i in range(len(wide)):
        if i > 0:
            text += ","
        if not wide.is_valid(i):
            text += "null"
        else:
            var v = typed[i] * scale
            text += String(Int(v + 0.5) if v >= 0 else Int(v - 0.5))
    return text^


def test_a_running_sum_ends_at_the_row() raises:
    assert_equal(
        shown("SELECT sum(qty) OVER (ORDER BY qty) AS x FROM sales"),
        "9,64,4,159,29,17,89,1,119,44",
    )
    assert_equal(
        shown(
            "SELECT sum(qty) OVER (PARTITION BY shop ORDER BY price) AS x"
            " FROM sales"
        ),
        "75,60,70,40,67,83,25,84,55,75",
    )


def test_the_ranks() raises:
    assert_equal(
        shown(
            "SELECT row_number() OVER (PARTITION BY shop ORDER BY qty DESC)"
            " AS x FROM sales"
        ),
        "4,2,5,1,3,4,2,5,1,3",
    )
    assert_equal(
        shown("SELECT rank() OVER (ORDER BY shop) AS x FROM sales"),
        "1,6,1,6,1,6,1,6,1,6",
    )
    assert_equal(
        shown("SELECT dense_rank() OVER (ORDER BY shop DESC) AS x FROM sales"),
        "2,1,2,1,2,1,2,1,2,1",
    )
    assert_equal(
        shown("SELECT ntile(4) OVER (ORDER BY qty) AS x FROM sales"),
        "1,3,1,4,2,2,3,1,4,2",
    )


def test_lag_and_lead() raises:
    assert_equal(
        shown(
            "SELECT lag(qty) OVER (PARTITION BY shop ORDER BY qty) AS x FROM"
            " sales"
        ),
        "3,15,null,20,5,1,12,null,25,8",
    )
    assert_equal(
        shown("SELECT lead(qty, 2, 0) OVER (ORDER BY qty) AS x FROM sales"),
        "12,30,8,0,20,15,40,5,0,25",
    )
    assert_equal(
        shown("SELECT lag(mark IGNORE NULLS) OVER (ORDER BY i) AS x FROM gappy"),
        "null,4,4,4,9,9",
    )


def test_rows_frames() raises:
    assert_equal(
        shown(
            "SELECT sum(qty) OVER (ORDER BY qty ROWS BETWEEN 1 PRECEDING AND 1"
            " FOLLOWING) AS x FROM sales"
        ),
        "16,60,9,70,35,25,75,4,95,47",
    )
    assert_equal(
        shown(
            "SELECT max(price) OVER (ORDER BY qty ROWS BETWEEN 2 PRECEDING AND"
            " CURRENT ROW) AS x FROM sales"
        ),
        "100,6,100,4,10,10,6,100,4,9",
    )
    assert_equal(
        shown(
            "SELECT avg(qty) OVER (PARTITION BY shop ORDER BY qty ROWS 1"
            " PRECEDING) AS x FROM sales",
            10,
        ),
        "40,175,30,300,85,45,185,10,275,115",
    )
    assert_equal(
        shown(
            "SELECT sum(mark) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND"
            " CURRENT ROW) AS x FROM gappy"
        ),
        "4,8,4,9,9,1",
    )


def test_range_and_groups_frames() raises:
    assert_equal(
        shown(
            "SELECT count(*) OVER (ORDER BY qty RANGE BETWEEN 10 PRECEDING AND"
            " 10 FOLLOWING) AS x FROM sales"
        ),
        "6,5,5,2,6,6,4,4,4,6",
    )
    assert_equal(
        shown(
            "SELECT sum(qty) OVER (ORDER BY shop GROUPS BETWEEN CURRENT ROW AND"
            " 1 FOLLOWING EXCLUDE CURRENT ROW) AS x FROM sales"
        ),
        "154,64,156,44,147,76,134,83,129,69",
    )


def test_the_values_of_a_frame() raises:
    assert_equal(
        shown(
            "SELECT first_value(qty) OVER (PARTITION BY shop ORDER BY price)"
            " AS x FROM sales"
        ),
        "25,40,25,40,25,40,25,40,25,40",
    )
    assert_equal(
        shown(
            "SELECT last_value(qty) OVER (PARTITION BY shop ORDER BY price ROWS"
            " BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS x FROM"
            " sales"
        ),
        "5,1,5,1,5,1,5,1,5,1",
    )
    assert_equal(
        shown("SELECT nth_value(qty, 3) OVER (ORDER BY qty) AS x FROM sales"),
        "5,5,null,5,5,5,5,null,5,5",
    )


def test_nulls_sort_last_either_way_unless_told() raises:
    assert_equal(
        shown("SELECT row_number() OVER (ORDER BY mark DESC) AS x FROM gappy"),
        "2,3,5,1,6,4",
    )
    assert_equal(
        shown(
            "SELECT row_number() OVER (ORDER BY mark NULLS FIRST) AS x FROM"
            " gappy"
        ),
        "4,5,1,6,2,3",
    )


def test_a_qualify_over_a_rank_keeps_the_top_of_each_partition() raises:
    var out = run(
        "SELECT qty AS x FROM sales QUALIFY row_number() OVER (PARTITION BY"
        " shop ORDER BY qty DESC) <= 2",
        session(),
    )
    var col = out.column("x").as_typed[DType.int64]()
    assert_equal(len(col), 4)
    assert_equal(col[0], 20)
    assert_equal(col[1], 40)
    assert_equal(col[2], 25)
    assert_equal(col[3], 30)


def test_two_windows_that_sort_differently_are_two_nodes() raises:
    assert_equal(
        shown(
            "SELECT sum(qty) OVER (ORDER BY qty) - sum(qty) OVER (PARTITION BY"
            " shop) AS x FROM sales"
        ),
        "-66,-20,-71,75,-46,-67,14,-83,44,-40",
    )


def test_the_shapes_that_are_turned_down() raises:
    with assert_raises(contains="negative"):
        _ = run(
            "SELECT sum(qty) OVER (ORDER BY qty ROWS -1 PRECEDING) FROM sales",
            session(),
        )
    with assert_raises(contains="one ORDER BY key"):
        _ = run(
            "SELECT sum(qty) OVER (ORDER BY qty, price RANGE 1 PRECEDING) FROM"
            " sales",
            session(),
        )
    with assert_raises(contains="takes no IGNORE NULLS"):
        _ = run(
            "SELECT sum(qty IGNORE NULLS) OVER (ORDER BY qty) FROM sales",
            session(),
        )
    with assert_raises(contains="takes no arguments"):
        _ = run("SELECT rank(qty) OVER (ORDER BY qty) FROM sales", session())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
