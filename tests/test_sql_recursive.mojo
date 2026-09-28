"""Tests for recursive CTEs run end to end through `sql/run.mojo`.

Every expected column is DuckDB 1.5.5's answer to the same query over the same
`edges` table, five edges with a cycle through 1, 2 and 3 and an island of 5
and 6:

    src  dst
    1    2
    2    3
    3    1
    3    4
    5    6
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


def edges() raises -> DataFrame:
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 3, 5]))
    columns.append(numbers([2, 3, 1, 4, 6]))
    var fields = List[Field]()
    fields.append(Field("src", LogicalType.INT64))
    fields.append(Field("dst", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def session() raises -> Catalog:
    var catalog = Catalog()
    catalog.register("edges", edges())
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


def test_a_counter_runs_until_the_step_is_empty() raises:
    assert_equal(
        shown(
            "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n"
            " WHERE x < 5) SELECT x FROM n"
        ),
        "1,2,3,4,5",
    )


def test_the_anchor_is_everything_left_of_the_last_union() raises:
    # Both anchor rows go first and each round follows, which is also the order
    # DuckDB hands the rows back in.
    assert_equal(
        shown(
            "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT 10 UNION ALL"
            " SELECT x + 1 FROM n WHERE x < 3) SELECT x FROM n"
        ),
        "1,10,2,3",
    )


def test_a_union_stops_on_a_cycle() raises:
    assert_equal(
        shown(
            "WITH RECURSIVE r(x) AS (SELECT 1 UNION SELECT (x % 3) + 1 FROM r)"
            " SELECT x FROM r ORDER BY x"
        ),
        "1,2,3",
    )


def test_what_a_node_reaches_through_a_join() raises:
    assert_equal(
        shown(
            "WITH RECURSIVE reach(x) AS (SELECT 1 UNION SELECT dst FROM edges"
            " JOIN reach ON src = x) SELECT x FROM reach ORDER BY x"
        ),
        "1,2,3,4",
    )


def test_two_columns_carried_round_by_round() raises:
    assert_equal(
        shown(
            "WITH RECURSIVE fib(a, b) AS (SELECT 0, 1 UNION ALL SELECT b, a + b"
            " FROM fib WHERE b < 50) SELECT a AS x FROM fib"
        ),
        "0,1,1,2,3,5,8,13,21,34",
    )


def test_the_step_is_cast_to_the_anchor_type() raises:
    # `x + 0.75` rounds back to an integer every round, so 1 becomes 2 and
    # not 1.75, and the recursion stops at 3.
    assert_equal(
        shown(
            "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 0.75 FROM r"
            " WHERE x < 3) SELECT x FROM r"
        ),
        "1,2,3",
    )


def test_the_step_may_join_the_working_rows_to_themselves() raises:
    assert_equal(
        shown(
            "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT a.x + b.x FROM"
            " r a, r b WHERE a.x < 8) SELECT x FROM r"
        ),
        "1,2,4,8",
    )


def test_the_cte_may_be_read_twice() raises:
    assert_equal(
        shown(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n"
            " WHERE i < 5) SELECT count(*) AS x FROM n"
            " JOIN (SELECT i + 1 AS j FROM n) m ON n.i = m.j"
        ),
        "4",
    )


def test_a_filter_above_stays_above() raises:
    # Pushed into the anchor it would keep the recursion from starting.
    assert_equal(
        shown(
            "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n"
            " WHERE x < 5) SELECT x FROM n WHERE x > 3"
        ),
        "4,5",
    )


def test_text_is_carried_from_round_to_round() raises:
    assert_equal(
        shown(
            "WITH RECURSIVE t AS (SELECT 1 AS depth, 'a' AS path UNION ALL"
            " SELECT depth + 1, CASE WHEN path = 'a' THEN 'ab' ELSE 'abb' END"
            " FROM t WHERE depth < 3) SELECT path FROM t",
            "path",
        ),
        "a,ab,abb",
    )


def test_a_later_entry_reads_the_recursion() raises:
    assert_equal(
        shown(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n"
            " WHERE i < 3), m AS (SELECT i * 10 AS x FROM n) SELECT x FROM m"
        ),
        "10,20,30",
    )


def test_a_union_by_name_is_turned_down() raises:
    with assert_raises(contains="by name"):
        _ = run(
            (
                "WITH RECURSIVE n(x) AS (SELECT 1 AS x UNION ALL BY NAME"
                " SELECT x + 1 AS x FROM n WHERE x < 5) SELECT x FROM n"
            ),
            session(),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
