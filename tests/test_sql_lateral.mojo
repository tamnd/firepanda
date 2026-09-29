"""Tests for `LEFT JOIN LATERAL`, run end to end through `sql/run.mojo`.

Every expected column is DuckDB 1.5.5's answer to the same query over the same
two tables. Shop 3 has no sales and shop 2 has one small one:

    shops              sales
    shop  name         shop  qty
    1     a            1     10
    2     b            1     20
    3     c            2     5
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


def names(values: List[String]) raises -> AnyArray:
    var text = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        text.append(values[i].as_bytes())
    return AnyArray(text^.finish())


def shops() raises -> DataFrame:
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3]))
    columns.append(names(["a", "b", "c"]))
    var fields = List[Field]()
    fields.append(Field("shop", LogicalType.INT64))
    fields.append(Field("name", LogicalType.STRING))
    return DataFrame(Schema(fields^), columns^)


def sales() raises -> DataFrame:
    var columns = List[AnyArray]()
    columns.append(numbers([1, 1, 2]))
    columns.append(numbers([10, 20, 5]))
    var fields = List[Field]()
    fields.append(Field("shop", LogicalType.INT64))
    fields.append(Field("qty", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def session() raises -> Catalog:
    var catalog = Catalog()
    catalog.register("shops", shops())
    catalog.register("sales", sales())
    return catalog^


def shown(sql: StringSlice, column: String = "qty") raises -> String:
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


def test_a_left_row_with_nothing_to_its_right_is_padded() raises:
    var sql = (
        "SELECT name, qty FROM shops LEFT JOIN LATERAL (SELECT qty FROM sales"
        " WHERE sales.shop = shops.shop) s ON true ORDER BY name, qty"
    )
    assert_equal(shown(sql, "name"), "a,a,b,c")
    assert_equal(shown(sql), "10,20,5,null")


def test_the_padding_is_on_the_answer_and_not_on_what_it_read() raises:
    # A constant in the select list is null for a padded row, since the
    # subquery answered nothing for it.
    var sql = (
        "SELECT name, one, dbl FROM shops LEFT JOIN LATERAL (SELECT 1 AS one,"
        " qty * 2 AS dbl FROM sales WHERE sales.shop = shops.shop AND qty > 7)"
        " s ON true ORDER BY name, dbl"
    )
    assert_equal(shown(sql, "name"), "a,a,b,c")
    assert_equal(shown(sql, "one"), "1,1,null,null")
    assert_equal(shown(sql, "dbl"), "20,40,null,null")


def test_a_coalesce_in_the_subquery_does_not_fill_the_padding() raises:
    var sql = (
        "SELECT name, c FROM shops LEFT JOIN LATERAL (SELECT coalesce(qty, 0)"
        " AS c FROM sales WHERE sales.shop = shops.shop) s ON true"
        " ORDER BY name, c"
    )
    assert_equal(shown(sql, "c"), "10,20,5,null")


def test_a_filter_above_sees_the_padding() raises:
    var sql = (
        "SELECT name, n FROM shops LEFT JOIN LATERAL (SELECT qty AS n FROM"
        " sales WHERE sales.shop = shops.shop) s ON true WHERE n IS NULL"
    )
    assert_equal(shown(sql, "name"), "c")


def test_a_condition_on_the_join_is_turned_down() raises:
    with assert_raises(contains="ON true"):
        _ = run(
            (
                "SELECT name FROM shops LEFT JOIN LATERAL (SELECT qty FROM"
                " sales WHERE sales.shop = shops.shop) s ON qty > 5"
            ),
            session(),
        )


def test_a_select_list_that_reads_the_left_side_is_turned_down() raises:
    with assert_raises(contains="reads the left side"):
        _ = run(
            (
                "SELECT name FROM shops LEFT JOIN LATERAL (SELECT qty +"
                " shops.shop AS q FROM sales WHERE sales.shop = shops.shop) s"
                " ON true"
            ),
            session(),
        )


def test_a_comparison_that_is_not_an_equality_is_turned_down() raises:
    with assert_raises(contains="equalities alone"):
        _ = run(
            (
                "SELECT name FROM shops LEFT JOIN LATERAL (SELECT qty FROM"
                " sales WHERE sales.shop = shops.shop AND qty > shops.shop) s"
                " ON true"
            ),
            session(),
        )


def test_a_fold_hands_every_left_row_one_row() raises:
    var sql = (
        "SELECT shop, total FROM shops, LATERAL (SELECT sum(qty) AS total FROM"
        " sales WHERE sales.shop = shops.shop) ORDER BY shop"
    )
    assert_equal(shown(sql, "shop"), "1,2,3")
    assert_equal(shown(sql, "total"), "30,5,null")


def test_a_count_over_nothing_is_zero() raises:
    assert_equal(
        shown(
            (
                "SELECT shop, n FROM shops, LATERAL (SELECT count(*) AS n FROM"
                " sales WHERE sales.shop = shops.shop) ORDER BY shop"
            ),
            "n",
        ),
        "2,1,0",
    )


def test_a_left_join_lateral_that_folds_is_the_same_rows() raises:
    var sql = (
        "SELECT shop, n, m FROM shops LEFT JOIN LATERAL (SELECT count(qty) AS"
        " n, max(qty) AS m FROM sales WHERE sales.shop = shops.shop) ON true"
        " ORDER BY shop"
    )
    assert_equal(shown(sql, "n"), "2,1,0")
    assert_equal(shown(sql, "m"), "20,5,null")


def test_a_fold_may_be_added_to_a_left_column() raises:
    assert_equal(
        shown(
            (
                "SELECT shop, x FROM shops, LATERAL (SELECT count(*) +"
                " shops.shop AS x FROM sales WHERE sales.shop = shops.shop)"
                " ORDER BY shop"
            ),
            "x",
        ),
        "3,3,3",
    )


def test_a_filter_on_the_subquery_folds_fewer_rows() raises:
    assert_equal(
        shown(
            (
                "SELECT shop, t FROM shops, LATERAL (SELECT sum(qty) AS t FROM"
                " sales WHERE qty > 5 AND sales.shop = shops.shop) ORDER BY"
                " shop"
            ),
            "t",
        ),
        "30,null,null",
    )


def test_an_uncorrelated_fold_is_one_row_for_all() raises:
    assert_equal(
        shown(
            (
                "SELECT shop, t FROM shops, LATERAL (SELECT sum(qty) AS t FROM"
                " sales) ORDER BY shop"
            ),
            "t",
        ),
        "35,35,35",
    )


def test_a_fold_correlated_by_an_inequality_is_turned_down() raises:
    with assert_raises(contains="equalities alone"):
        _ = run(
            (
                "SELECT shop, t FROM shops, LATERAL (SELECT sum(qty) AS t FROM"
                " sales WHERE sales.shop < shops.shop)"
            ),
            session(),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
