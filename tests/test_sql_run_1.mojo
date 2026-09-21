"""A query text in, rows out.

Every other test in the SQL front end checks one stage against the stage's own
idea of an answer. The transform tests check SQL printed back, the lowering
tests check the text `explain` produces, and the pipeline tests check frames
built by hand. All of them can pass while the stages do not fit together, which
is what this file is for: it names no stage and asserts nothing about a plan, it
writes a query and checks the rows.

The queries are deliberately small and the numbers are deliberately not in
order, so that a filter that keeps a middle range is not a slice and a sort that
does nothing is visible. The frames have three chunks for the same reason they
do in the pipeline tests, which is that a chunk boundary is where an off by one
in a position lives.

The refusals matter as much as the answers. A shape nothing runs yet has to say
so by name, because a query that quietly returned the wrong rows would be the
kind of defect that only a differential run against DuckDB finds, and a refusal
is something a caller can act on.

Part 1 of 3. The fixtures are in tests/support/sql_run.mojo.
"""


from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.array.strings import StringBuilder
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.dtype.temporal import TimeUnit
from firepanda.frame.frame import DataFrame
from firepanda.sql.catalog import Catalog
from firepanda.sql.run import run

from tests.support.sql_run import (
    _marks,
    answer,
    cuts,
    days,
    dupes,
    gapped,
    gappy,
    gaps,
    glyphs,
    hits,
    lengths,
    moments,
    numbers,
    padded,
    read_back,
    sales,
    same,
    session,
    shifts,
    shops,
    stock,
    tiers,
    truths,
    visits,
    words,
)


def test_the_smallest_query_reads_a_column() raises:
    same(
        answer("SELECT qty FROM sales", "qty"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15],
        "qty",
    )


def test_a_star_reads_every_column() raises:
    var out = run("SELECT * FROM sales", session())
    assert_equal(len(out.schema), 3)
    assert_equal(len(out), 10)
    same(read_back(out, "price"), [10, 2, 7, 1, 5, 9, 3, 100, 4, 6], "price")


def test_a_where_keeps_the_rows_it_says() raises:
    same(
        answer("SELECT qty FROM sales WHERE qty > 20", "qty"),
        [40, 25, 30],
        "qty",
    )


def test_two_conditions_both_have_to_hold() raises:
    same(
        answer(
            "SELECT qty FROM sales WHERE qty > 4 AND qty < 20",
            "qty",
        ),
        [5, 12, 8, 15],
        "qty",
    )


def test_conditions_run_in_a_different_order_than_they_were_written() raises:
    # The planner runs the equality first and the range over what it kept, and
    # the point of this test is that the rows do not know that. Written the
    # other way round it is the same four rows in the same order, because the
    # order of an `and` decides what each condition reads and nothing else.
    same(
        answer(
            "SELECT qty FROM sales WHERE qty > 4 AND shop = 1",
            "qty",
        ),
        [5, 12, 25, 30],
        "qty",
    )
    same(
        answer(
            "SELECT qty FROM sales WHERE shop = 1 AND qty > 4",
            "qty",
        ),
        [5, 12, 25, 30],
        "the same rows the other way round",
    )


def test_an_expression_in_the_select_list_is_computed() raises:
    same(
        answer("SELECT qty * price AS total FROM sales", "total"),
        [50, 40, 21, 40, 60, 72, 75, 100, 120, 90],
        "total",
    )


def test_an_integer_division_rounds_the_way_the_dialect_rounds() raises:
    """A query asked for SQL, so `//` truncates towards zero and `%` takes the
    sign of the dividend. Both lists came off DuckDB 1.5.1.

    The frame surface answers the other rounding to the same expression and that
    is not a bug on either side. What was a bug is that this went through the
    pandas kernels until issue #770, so `-7 // 3` came back `-3` here where the
    engine being copied says `-2`, and the rows that are positive agreed all
    along, which is why nothing noticed."""
    same(
        answer("SELECT (qty - 12) // 3 AS q FROM sales", "q"),
        [-2, 2, -3, 9, 0, -1, 4, -3, 6, 1],
        "truncated",
    )
    same(
        answer("SELECT (qty - 12) % 3 AS r FROM sales", "r"),
        [-1, 2, 0, 1, 0, -1, 1, -2, 0, 0],
        "remainder",
    )
    same(
        answer("SELECT (qty - 12) // (-3) AS q FROM sales", "q"),
        [2, -2, 3, -9, 0, 1, -4, 3, -6, -1],
        "a negative divisor",
    )


def test_a_zero_divisor_in_a_query_is_a_null_and_not_an_error() raises:
    """DuckDB answers `NULL` rather than raising, and the divisor here is a
    column, which is the path that has to find the zero a row at a time rather
    than once above the loop. Row four is the one where `qty` is twelve."""
    var out = run("SELECT 12 // (qty - 12) AS q FROM sales", session())
    var col = out.column("q").as_typed[DType.int64]()
    var want: List[Int64] = [-1, 1, -1, 0, 0, -3, 0, -1, 0, 4]
    assert_equal(len(col), 10, "how many rows")
    for i in range(10):
        if i == 4:
            assert_true(not col.is_valid(i), "the zero divisor is a null")
            continue
        assert_true(col.is_valid(i), "row " + String(i))
        assert_equal(col[i], want[i], "row " + String(i))


def test_two_columns_divide_the_same_way_a_column_and_a_constant_do() raises:
    """The two column loop is a different loop from the constant one, and a
    dialect that was right in one of them and pandas' in the other is exactly
    the failure the differential found. Both columns hold negatives here and
    `price - 5` holds a zero."""
    var out = run(
        (
            "SELECT (qty - 12) // (price - 5) AS q, (qty - 12) % (price - 5) AS"
            " r FROM sales"
        ),
        session(),
    )
    var quotients = out.column("q").as_typed[DType.int64]()
    var remainders = out.column("r").as_typed[DType.int64]()
    var want_q: List[Int64] = [-1, -2, -4, -7, 0, -1, -6, 0, -18, 3]
    var want_r: List[Int64] = [-2, 2, -1, 0, 0, 0, 1, -11, 0, 0]
    for i in range(10):
        if i == 4:
            assert_true(not quotients.is_valid(i), "the zero divisor")
            assert_true(not remainders.is_valid(i), "and its remainder")
            continue
        assert_equal(quotients[i], want_q[i], "quotient at " + String(i))
        assert_equal(remainders[i], want_r[i], "remainder at " + String(i))


def test_an_alias_is_the_name_the_answer_comes_back_under() raises:
    var out = run("SELECT qty AS howmany FROM sales", session())
    assert_equal(out.schema[0].name, "howmany")


def test_an_aggregate_over_the_whole_table_is_one_row() raises:
    same(answer("SELECT SUM(qty) AS total FROM sales", "total"), [159], "total")


def test_the_shape_of_q29_sums_one_column_under_many_constants() raises:
    # ClickBench q29 with three constants instead of ninety and a smaller
    # column. The lowering folds each operation into its own reduction rather
    # than putting a compute in front of it, so the answers are what matters
    # here: a wrong side or a dropped constant would still give three numbers.
    var out = run(
        (
            "SELECT SUM(qty) AS a, SUM(qty + 1) AS b, SUM(qty + 2) AS c,"
            " SUM(qty * 2) AS d, SUM(100 - qty) AS e FROM sales"
        ),
        session(),
    )
    assert_equal(len(out), 1, "one row")
    same(read_back(out, "a"), [159], "the column itself")
    same(read_back(out, "b"), [169], "and ten ones")
    same(read_back(out, "c"), [179], "and ten twos")
    same(read_back(out, "d"), [318], "twice each")
    same(read_back(out, "e"), [841], "a thousand less the total")


def test_q29_under_a_group_by_answers_the_same_thing() raises:
    # The same expressions with a key on them, which takes the other route
    # through the lowering and has to come out agreeing with it. Every shop's
    # rows are counted, so the constant lands once per row on both sides.
    var out = run(
        (
            "SELECT shop, SUM(qty) AS a, SUM(qty + 1) AS b, SUM(100 - qty)"
            " AS e FROM sales GROUP BY shop ORDER BY shop"
        ),
        session(),
    )
    same(read_back(out, "shop"), [1, 2], "shop")
    same(read_back(out, "a"), [75, 84], "the column itself")
    same(read_back(out, "b"), [80, 89], "and one per row")
    same(read_back(out, "e"), [425, 416], "a hundred per row less the total")


def test_a_group_by_folds_the_rows_into_groups() raises:
    # Ordered by the key, because which group comes out first is the hash
    # table's business and not the query's.
    var out = run(
        "SELECT shop, SUM(qty) AS total FROM sales GROUP BY shop ORDER BY shop",
        session(),
    )
    same(read_back(out, "shop"), [1, 2], "shop")
    same(read_back(out, "total"), [75, 84], "total")


def test_a_filter_keeps_only_the_rows_it_names_out_of_a_fold() raises:
    # Five of the ten rows have a price over five, and their quantities come to
    # thirty two. The whole table is still read, so a filter that was quietly
    # dropped would answer a hundred and fifty nine and be obvious.
    var out = run(
        (
            "SELECT sum(qty) FILTER (WHERE price > 5) AS s,"
            " count(*) FILTER (WHERE price > 5) AS c FROM sales"
        ),
        session(),
    )
    same(read_back(out, "s"), [32], "the quantities of the dearer rows")
    same(read_back(out, "c"), [5], "how many of them there are")


def test_a_filter_under_a_group_by_applies_inside_each_group() raises:
    # Each group tests its own rows, so the two answers are not the whole
    # table's answer split in half and a filter lifted into a `WHERE` would
    # change the other column as well.
    var out = run(
        (
            "SELECT shop, sum(qty) FILTER (WHERE price > 5) AS s,"
            " count(*) FILTER (WHERE qty > 20) AS c, count(*) AS n FROM sales"
            " GROUP BY shop ORDER BY shop"
        ),
        session(),
    )
    same(read_back(out, "shop"), [1, 2], "shop")
    same(read_back(out, "s"), [8, 24], "the dearer rows of each shop")
    same(read_back(out, "c"), [2, 1], "the larger rows of each shop")
    same(read_back(out, "n"), [5, 5], "every row of each shop")


def test_a_filter_rides_on_a_distinct_count_and_on_an_extreme() raises:
    var out = run(
        (
            "SELECT count(DISTINCT shop) FILTER (WHERE price > 5) AS d,"
            " max(qty) FILTER (WHERE price > 5) AS m FROM sales"
        ),
        session(),
    )
    same(read_back(out, "d"), [2], "both shops sold a dearer row")
    same(read_back(out, "m"), [15], "the largest of the dearer rows")


def test_a_filter_that_keeps_no_row_answers_null() raises:
    # A filter does not take rows away, it turns the ones it does not want into
    # nulls, so this is a fold that saw rows and found every value in it null
    # rather than a fold over no rows at all. Those were two answers until
    # #836 and they are one now.
    same(
        gapped(
            run(
                "SELECT sum(qty) FILTER (WHERE price > 500) AS s FROM sales",
                session(),
            ),
            "s",
        ),
        [-1],
        "s",
    )


def test_a_filter_on_a_fold_that_reads_a_null_as_a_value_is_refused() raises:
    # These two take the value in the first or the last row rather than folding
    # over the values, and a null is one of the values to them. A row the filter
    # turned into a null is not a row it took away as far as they are concerned,
    # so the rewrite would answer a different question and there is no other
    # one. See #888.
    with assert_raises(contains="FILTER on first"):
        _ = run(
            "SELECT first(qty) FILTER (WHERE price > 5) AS f FROM sales",
            session(),
        )
    with assert_raises(contains="FILTER on last"):
        _ = run(
            "SELECT last(qty) FILTER (WHERE price > 5) AS f FROM sales",
            session(),
        )


def test_a_filter_on_any_value_is_the_case_rewritten() raises:
    # `any_value` was refused beside the two above until the three stopped
    # being one thing. It passes over a null, so a row turned into a null and a
    # row taken away are the same row to it, and the `CASE` the rewrite writes
    # says what the filter said. The five rows priced over five carry 5, 3, 8, 1
    # and 15, and the first of them is the answer DuckDB gives too.
    same(
        gapped(
            run(
                (
                    "SELECT any_value(qty) FILTER (WHERE price > 5) AS f"
                    " FROM sales"
                ),
                session(),
            ),
            "f",
        ),
        [5],
        "f",
    )


def test_a_filter_on_something_that_is_not_a_fold_is_refused() raises:
    # DuckDB says the same thing about it, since a filter describes which rows
    # a fold reads and a scalar call reads one row by definition.
    with assert_raises(contains="FILTER on upper"):
        _ = run(
            "SELECT upper(word) FILTER (WHERE n > 1) AS u FROM words",
            session(),
        )


def test_a_group_by_folds_by_a_name_the_select_list_gave() raises:
    # The same two groups as the query above, which wrote the column itself.
    var out = run(
        "SELECT shop AS k, SUM(qty) AS total FROM sales GROUP BY k ORDER BY k",
        session(),
    )
    same(read_back(out, "k"), [1, 2], "the alias")
    same(read_back(out, "total"), [75, 84], "total")


def test_a_group_by_folds_by_an_expression_a_name_stands_for() raises:
    # Ten rows and nine sums, because two of them come to the same number, so
    # a grouping that quietly grouped by something else would be visible in
    # the row count before it was visible in the sums.
    var out = run(
        (
            "SELECT qty + price AS k, COUNT(*) AS c FROM sales GROUP BY k"
            " ORDER BY k"
        ),
        session(),
    )
    same(read_back(out, "k"), [10, 15, 17, 21, 22, 28, 34, 41, 101], "the sums")
    same(read_back(out, "c"), [1, 1, 2, 1, 1, 1, 1, 1, 1], "how many rows")


def test_several_keys_and_one_of_them_is_a_name() raises:
    # The shape q39 of ClickBench is written in: a case expression and a plain
    # column both given names in the select list and both named again in the
    # GROUP BY, with a fold beside them.
    var out = run(
        (
            "SELECT shop AS s, CASE WHEN qty > 10 THEN 1 ELSE 0 END AS big,"
            " COUNT(*) AS n FROM sales GROUP BY s, big ORDER BY s, big"
        ),
        session(),
    )
    same(read_back(out, "s"), [1, 1, 2, 2], "shop")
    same(read_back(out, "big"), [0, 1, 0, 1], "over ten or not")
    same(read_back(out, "n"), [2, 3, 2, 3], "how many rows")


def test_a_column_of_the_table_wins_over_a_name_of_the_same_spelling() raises:
    # `qty` is a column of sales and the name this select list gives to `shop`,
    # and the ten quantities are all different, so grouping by the column gives
    # ten groups where grouping by the shop would give two. The table is asked
    # first, so ten is the answer. The shop is written in the GROUP BY as well
    # because a query that reads it has to group by it either way, and that is
    # what leaves the row count as the only thing the two readings differ on.
    var out = run(
        "SELECT shop AS qty, COUNT(*) AS c FROM sales GROUP BY qty, shop",
        session(),
    )
    assert_equal(len(out), 10, "one group per quantity")


def test_a_group_by_that_names_a_fold_says_why_not() raises:
    with assert_raises(contains="computed over the groups"):
        _ = run("SELECT shop, COUNT(*) AS c FROM sales GROUP BY c", session())


def test_a_group_by_all_folds_the_same_groups() raises:
    var out = run(
        "SELECT shop, SUM(qty) AS total FROM sales GROUP BY ALL ORDER BY shop",
        session(),
    )
    same(read_back(out, "shop"), [1, 2], "shop")
    same(read_back(out, "total"), [75, 84], "total")


def test_a_group_by_all_with_nothing_to_fold_keeps_one_of_each() raises:
    same(
        answer("SELECT shop FROM sales GROUP BY ALL ORDER BY shop", "shop"),
        [1, 2],
        "the two shops, once each",
    )


def test_a_group_by_all_over_a_star_groups_by_the_whole_row() raises:
    # Every column of the row is a key, so nothing folds and every row of a
    # table with no repeat in it comes back.
    var out = run("SELECT * FROM shops GROUP BY ALL ORDER BY shop", session())
    same(read_back(out, "shop"), [1, 2, 3], "shop")
    same(read_back(out, "floor"), [11, 22, 33], "floor")


def test_an_order_by_sorts_on_a_column_the_answer_does_not_hold() raises:
    var out = run("SELECT price FROM sales ORDER BY qty", session())
    assert_equal(out.width(), 1, "the column the sort read is not in it")
    same(
        read_back(out, "price"),
        [100, 7, 10, 9, 5, 6, 2, 3, 4, 1],
        "the prices in quantity order",
    )


def test_an_order_by_sorts_on_the_name_the_query_renamed_away() raises:
    same(
        answer(
            "SELECT qty AS howmany FROM sales ORDER BY qty DESC LIMIT 3",
            "howmany",
        ),
        [40, 30, 25],
        "the three largest",
    )


def test_an_order_by_all_sorts_on_every_column_it_produced() raises:
    var out = run("SELECT qty, shop FROM sales ORDER BY ALL", session())
    same(read_back(out, "qty"), [1, 3, 5, 8, 12, 15, 20, 25, 30, 40], "qty")
    same(read_back(out, "shop"), [2, 1, 1, 2, 1, 2, 2, 1, 1, 2], "shop")


def test_an_order_by_all_descending_reverses_all_of_them() raises:
    same(
        answer("SELECT band FROM tiers ORDER BY ALL DESC", "band"),
        [99, 40, 20, 3],
        "band",
    )


def test_a_having_drops_a_whole_group() raises:
    var out = run(
        (
            "SELECT shop, SUM(qty) AS total FROM sales GROUP BY shop"
            " HAVING SUM(qty) > 80"
        ),
        session(),
    )
    same(read_back(out, "shop"), [2], "shop")
    same(read_back(out, "total"), [84], "total")


def test_an_order_by_orders_the_answer() raises:
    same(
        answer("SELECT qty FROM sales ORDER BY qty", "qty"),
        [1, 3, 5, 8, 12, 15, 20, 25, 30, 40],
        "qty",
    )


def test_an_order_by_descending_reverses_it() raises:
    same(
        answer("SELECT qty FROM sales ORDER BY qty DESC", "qty"),
        [40, 30, 25, 20, 15, 12, 8, 5, 3, 1],
        "qty",
    )


def test_an_order_by_of_a_position_orders_the_answer() raises:
    # The same rows the query naming the column gets. Before this the number
    # lowered as a constant and every row sorted the same, so the answer came
    # back in the order it was read in and nothing said so.
    same(
        answer("SELECT qty FROM sales ORDER BY 1", "qty"),
        [1, 3, 5, 8, 12, 15, 20, 25, 30, 40],
        "qty",
    )


def test_an_order_by_of_a_position_counts_the_columns_returned() raises:
    var out = run("SELECT qty, price FROM sales ORDER BY 2", session())
    same(read_back(out, "price"), [1, 2, 3, 4, 5, 6, 7, 9, 10, 100], "price")
    same(read_back(out, "qty"), [40, 20, 25, 30, 12, 15, 3, 8, 5, 1], "qty")


def test_an_order_by_of_a_position_past_the_end_is_refused() raises:
    with assert_raises(contains="positions this query has are 1 to 1"):
        _ = run("SELECT qty FROM sales ORDER BY 2", session())


def test_a_group_by_of_a_position_folds_on_the_item_it_counts_to() raises:
    # ClickBench and the TPC-H reference queries both write their keys this
    # way, and so does most SQL a tool generates.
    var out = run(
        "SELECT qty % 2 AS r, count(*) AS n FROM sales GROUP BY 1 ORDER BY 1",
        session(),
    )
    same(read_back(out, "r"), [0, 1], "the two remainders")
    same(read_back(out, "n"), [5, 5], "and how many rows are in each")


def test_a_group_by_of_a_position_still_folds_the_other_columns() raises:
    # qty is 5, 20, 3, 40, 12, 8, 25, 1, 30, 15 and price is 10, 2, 7, 1, 5, 9,
    # 3, 100, 4, 6, so the even rows carry 2, 1, 5, 9, 4 and the odd ones carry
    # 10, 7, 3, 100, 6.
    var out = run(
        "SELECT qty % 2 AS r, sum(price) AS s FROM sales GROUP BY 1 ORDER BY 1",
        session(),
    )
    same(read_back(out, "r"), [0, 1], "the two remainders")
    same(read_back(out, "s"), [21, 126], "and the sum over each")


def test_a_group_by_of_a_position_past_the_end_is_refused() raises:
    with assert_raises(contains="the select list has are 1 to 2"):
        _ = run("SELECT qty, count(*) FROM sales GROUP BY 3", session())


def test_a_limit_cuts_the_answer() raises:
    same(answer("SELECT qty FROM sales LIMIT 3", "qty"), [5, 20, 3], "qty")


def test_a_limit_with_an_offset_starts_further_in() raises:
    same(
        answer("SELECT qty FROM sales LIMIT 3 OFFSET 4", "qty"),
        [12, 8, 25],
        "qty",
    )


def test_an_order_by_with_a_limit_is_the_top_of_the_table() raises:
    same(
        answer("SELECT qty FROM sales ORDER BY qty DESC LIMIT 3", "qty"),
        [40, 30, 25],
        "qty",
    )


def test_an_order_by_sorts_on_a_fold_the_query_returns() raises:
    # The shop totals are 37, 40 and 7, so the biggest first is shop 2, shop 1
    # and shop 3. DuckDB answers the same three rows in the same order.
    var out = run(
        (
            "SELECT shop, sum(qty) AS total FROM stock GROUP BY shop"
            " ORDER BY sum(qty) DESC"
        ),
        session(),
    )
    same(read_back(out, "shop"), [2, 1, 3], "shop")
    same(read_back(out, "total"), [40, 37, 7], "total")


def test_an_order_by_sorts_on_a_fold_the_query_does_not_return() raises:
    # One column out and the sum nowhere in it. The aggregate computes it all
    # the same, because the ORDER BY is read before that node is built.
    var out = run(
        "SELECT shop FROM stock GROUP BY shop ORDER BY sum(qty) DESC",
        session(),
    )
    assert_equal(out.width(), 1)
    same(read_back(out, "shop"), [2, 1, 3], "shop")


def test_an_order_by_sorts_on_the_name_a_fold_was_given() raises:
    var out = run(
        (
            "SELECT shop, sum(qty) AS total FROM stock GROUP BY shop"
            " ORDER BY total"
        ),
        session(),
    )
    same(read_back(out, "shop"), [3, 1, 2], "shop")
    same(read_back(out, "total"), [7, 37, 40], "total")


def test_an_order_by_sorts_on_an_expression_over_two_folds() raises:
    # Four rows for shop 1 and one each for the others, so the counts decide it
    # and the sums only break a tie there is not one of.
    same(
        answer(
            (
                "SELECT shop FROM stock GROUP BY shop"
                " ORDER BY count(*) * 100 + sum(qty) DESC"
            ),
            "shop",
        ),
        [1, 2, 3],
        "shop",
    )


def test_a_limit_over_a_fold_in_the_order_by_is_the_biggest_group() raises:
    same(
        answer(
            (
                "SELECT shop FROM stock GROUP BY shop ORDER BY sum(qty) DESC"
                " LIMIT 1"
            ),
            "shop",
        ),
        [2],
        "shop",
    )


def test_a_join_pairs_the_rows_that_match() raises:
    var out = run(
        (
            "SELECT sales.qty, tiers.rate FROM sales JOIN tiers"
            " ON sales.qty = tiers.band ORDER BY sales.qty"
        ),
        session(),
    )
    same(read_back(out, "qty"), [3, 20, 40], "qty")
    same(read_back(out, "rate"), [300, 200, 400], "rate")


def test_a_comma_in_the_from_pairs_the_same_rows_a_join_does() raises:
    # Two tables and the condition in the `WHERE`, which lowers to a cross join
    # under a filter and is turned back into the pairing by the optimizer. The
    # answer is the one the test above gets, which is the point of it.
    var out = run(
        "SELECT qty, rate FROM sales, tiers WHERE qty = band ORDER BY qty",
        session(),
    )
    same(read_back(out, "qty"), [3, 20, 40], "qty")
    same(read_back(out, "rate"), [300, 200, 400], "rate")


def test_a_comma_join_with_more_than_the_condition_in_the_where() raises:
    # The equality becomes the join's and what is left of the `WHERE` is still
    # a filter, pushed onto the side that can answer it.
    var out = run(
        (
            "SELECT qty, rate FROM sales, tiers WHERE qty = band AND rate > 250"
            " ORDER BY qty"
        ),
        session(),
    )
    same(read_back(out, "qty"), [3, 40], "qty")
    same(read_back(out, "rate"), [300, 400], "rate")


def test_a_condition_that_reads_both_sides_runs_above_the_join() raises:
    # A condition on one side is pushed below the join by the optimizer and a
    # condition on both cannot be, so this one is the residual filter that stays
    # where it was written.
    var out = run(
        (
            "SELECT qty, rate FROM sales JOIN tiers ON qty = band"
            " WHERE qty + rate > 250 ORDER BY qty"
        ),
        session(),
    )
    same(read_back(out, "qty"), [3, 40], "qty")
    same(read_back(out, "rate"), [300, 400], "rate")


def test_a_where_the_optimizer_pushes_onto_the_build_side_runs() raises:
    # The predicate is written about the left table, and the equality carries it
    # to the right one as well, so both sides end up with a filter under them.
    # The right side is then not a scan, which is the shape a join has in almost
    # every real query, and its build is a pipeline of its own.
    var out = run(
        (
            "SELECT qty, rate FROM sales JOIN tiers ON qty = band"
            " WHERE qty > 10 ORDER BY qty"
        ),
        session(),
    )
    same(read_back(out, "qty"), [20, 40], "qty")
    same(read_back(out, "rate"), [200, 400], "rate")


def test_a_where_about_the_right_table_alone_still_runs() raises:
    # Nothing about this one is on the probe side, so the whole predicate ends
    # up under the build and the left is read as it stands.
    var out = run(
        (
            "SELECT qty, rate FROM sales JOIN tiers ON qty = band"
            " WHERE rate > 250 ORDER BY qty"
        ),
        session(),
    )
    same(read_back(out, "qty"), [3, 40], "qty")
    same(read_back(out, "rate"), [300, 400], "rate")


def test_a_table_of_three_chunks_joins_from_either_side() raises:
    # Issue #583. `sales` is ten rows in three chunks and `tiers` is four rows
    # in one, and writing `sales` on the right put a chunked frame on the build
    # side, which raised out of the operator before a row was probed. Which side
    # of the word JOIN a table is written on is not supposed to decide whether
    # the query runs, and both spellings answer the same rows now.
    var one_way = run(
        "SELECT qty, rate FROM sales JOIN tiers ON qty = band ORDER BY qty",
        session(),
    )
    var other_way = run(
        "SELECT qty, rate FROM tiers JOIN sales ON band = qty ORDER BY qty",
        session(),
    )
    same(read_back(one_way, "qty"), [3, 20, 40], "qty")
    same(read_back(one_way, "rate"), [300, 200, 400], "rate")
    same(
        read_back(other_way, "qty"), read_back(one_way, "qty"), "qty either way"
    )
    same(
        read_back(other_way, "rate"),
        read_back(one_way, "rate"),
        "rate either way",
    )


def test_the_whole_shape_of_a_query_runs_at_once() raises:
    # A WHERE, a GROUP BY, a HAVING, an ORDER BY and a LIMIT in one statement,
    # which is the clause order the lowering builds bottom up.
    var out = run(
        (
            "SELECT shop, SUM(qty) AS total FROM sales WHERE qty < 30"
            " GROUP BY shop HAVING SUM(qty) > 20 ORDER BY total DESC LIMIT 1"
        ),
        session(),
    )
    same(read_back(out, "shop"), [1], "shop")
    same(read_back(out, "total"), [45], "total")


def test_a_table_nobody_registered_is_refused() raises:
    with assert_raises(contains="nosuch"):
        _ = run("SELECT qty FROM nosuch", session())


def test_a_column_no_table_has_is_refused() raises:
    with assert_raises(contains="nosuch"):
        _ = run("SELECT nosuch FROM sales", session())


def test_a_distinct_keeps_one_row_of_each() raises:
    # Two shops over ten rows, and the second one is seen on the second row, so
    # first seen order is the order they were written in the frame.
    var out = run("SELECT DISTINCT shop FROM sales", session())
    same(read_back(out, "shop"), [1, 2], "the shops")


def test_a_distinct_over_two_columns_reads_the_pair() raises:
    # Every quantity is its own row, so the pair of a shop and a quantity is as
    # tall as the frame and the distinct keeps all ten.
    var out = run(
        "SELECT DISTINCT shop, qty FROM sales ORDER BY qty", session()
    )
    same(
        read_back(out, "qty"),
        [1, 3, 5, 8, 12, 15, 20, 25, 30, 40],
        "every quantity",
    )


def test_a_distinct_runs_after_the_where() raises:
    var out = run(
        "SELECT DISTINCT shop FROM sales WHERE qty > 25 ORDER BY shop",
        session(),
    )
    same(read_back(out, "shop"), [1, 2], "both shops sold a large order")


def test_a_distinct_on_keeps_the_first_row_of_each_key() raises:
    # Ten rows and two shops, so this keeps two whole rows, and the quantity it
    # carries out says which row of each shop survived.
    var out = run("SELECT DISTINCT ON (shop) shop, qty FROM sales", session())
    same(read_back(out, "shop"), [1, 2], "the shops")
    same(read_back(out, "qty"), [5, 20], "the first row of each")


def test_a_distinct_on_lets_the_order_by_choose_the_row() raises:
    # The sort runs underneath, so the row that survives each shop is the one
    # with the largest quantity rather than the first one written.
    var out = run(
        "SELECT DISTINCT ON (shop) shop, qty FROM sales ORDER BY qty DESC",
        session(),
    )
    same(read_back(out, "shop"), [2, 1], "the busiest shop first")
    same(read_back(out, "qty"), [40, 30], "the largest of each")


def test_a_distinct_on_may_decide_on_a_column_it_does_not_return() raises:
    var out = run("SELECT DISTINCT ON (shop) qty FROM sales", session())
    assert_equal(len(out.schema), 1, "the shop was read and not returned")
    same(read_back(out, "qty"), [5, 20], "the first row of each shop")


def test_a_distinct_on_treats_a_null_key_as_a_value() raises:
    # Two nulls in the marks, and they are the same key as each other, which is
    # what SQL says for DISTINCT and is the opposite of a group by's rule.
    var out = run("SELECT DISTINCT ON (mark) mark FROM gappy", session())
    same(gapped(out, "mark"), [4, -1, 9, 1], "one row per mark, nulls counted")


def test_a_distinct_on_inside_an_arm_stays_inside_it() raises:
    var out = run(
        (
            "SELECT DISTINCT ON (shop) shop, qty FROM sales"
            " UNION ALL SELECT DISTINCT ON (shop) shop, qty FROM sales"
        ),
        session(),
    )
    same(read_back(out, "shop"), [1, 2, 1, 2], "each arm kept two rows")
    same(read_back(out, "qty"), [5, 20, 5, 20], "and the same two")


def test_a_computed_distinct_on_key_is_refused() raises:
    # A computed key is a column the row does not have, so the rows kept would
    # not be the rows the plan said.
    with assert_raises(contains="decides on a binary expression"):
        _ = run("SELECT DISTINCT ON (qty % 2) qty FROM sales", session())


def test_a_values_is_the_table_it_writes_out() raises:
    # The rows are in the query rather than in the catalog, so this is the one
    # statement that reads nothing at all.
    var out = run("VALUES (1, 2), (3, 4)", session())
    same(read_back(out, "col0"), [1, 3], "the first column")
    same(read_back(out, "col1"), [2, 4], "the second")


def test_a_values_folds_what_it_can_before_it_is_a_table() raises:
    # A row of this table is written as an expression and the constant folding
    # pass turns it into a value. Lowering refuses one that is still an
    # expression, because there is no chunk under a VALUES to compute it over.
    same(
        read_back(run("VALUES (1 + 1), (10 * 2)", session()), "col0"),
        [2, 20],
        "the folded rows",
    )


def test_a_values_written_where_a_table_goes_is_a_derived_table() raises:
    # A VALUES is a table and a parenthesised one in a FROM is a subquery whose
    # body is that table, so it goes through the derived table and comes back
    # under the names a VALUES invents.
    var out = run("SELECT * FROM (VALUES (1, 2), (3, 4)) AS t", session())
    same(read_back(out, "col0"), [1, 3], "the first column")
    same(read_back(out, "col1"), [2, 4], "the second")


def test_a_query_with_no_from_answers_a_constant() raises:
    # A statement with no FROM lowers to a literal table of one row, and the
    # projection above it needs a column for the constant to land in. The column
    # is called `1`, after the text it was written as, which is what DuckDB
    # calls it too.
    same(answer("SELECT 1", "1"), [1], "the constant")


def test_a_constant_query_folds_before_it_is_a_column() raises:
    same(answer("SELECT 1 + 1 AS two", "two"), [2], "the folded constant")


def test_a_constant_is_named_by_the_query_or_by_itself() raises:
    var out = run("SELECT 7 AS lucky", session())
    assert_equal(out.schema[0].name, "lucky", "the name the query gave it")
    same(read_back(out, "lucky"), [7], "the constant")


def test_a_constant_beside_a_column_is_as_long_as_the_column() raises:
    var out = run("SELECT qty, 1 AS one FROM sales", session())
    same(read_back(out, "qty"), [5, 20, 3, 40, 12, 8, 25, 1, 30, 15], "qty")
    same(read_back(out, "one"), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "the constant")


def test_a_constant_after_a_where_is_as_long_as_what_survived() raises:
    same(
        answer("SELECT 1 AS one FROM sales WHERE qty > 12", "one"),
        [1, 1, 1, 1, 1],
        "one per surviving row",
    )


def test_a_union_all_stacks_the_two_sides() raises:
    same(
        answer("SELECT qty FROM sales UNION ALL SELECT band FROM tiers", "qty"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15, 3, 20, 40, 99],
        "the first side then the second",
    )


def test_a_union_drops_the_rows_both_sides_have() raises:
    same(
        answer("SELECT qty FROM sales UNION SELECT band FROM tiers", "qty"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15, 99],
        "3, 20 and 40 are in both",
    )


def test_a_union_takes_the_name_the_first_side_uses() raises:
    var out = run(
        "SELECT qty FROM sales UNION ALL SELECT band FROM tiers", session()
    )
    assert_equal(out.schema[0].name, "qty", "the first side's name")


def test_each_side_of_a_union_may_have_its_own_where() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty > 25 UNION ALL SELECT band"
                " FROM tiers WHERE band < 10"
            ),
            "qty",
        ),
        [40, 30, 3],
        "what each side kept",
    )


def test_a_union_of_two_values_reads_nothing_at_all() raises:
    same(
        answer("VALUES (1), (2) UNION ALL VALUES (3)", "col0"),
        [1, 2, 3],
        "three rows from a query that names no table",
    )


def test_a_union_by_name_stacks_the_columns_that_share_a_name() raises:
    # The two arms write their columns in opposite orders and the answer is the
    # left arm's order, so the pairing is by name and not by position.
    var out = run(
        (
            "SELECT shop, qty FROM sales WHERE qty > 25 UNION ALL BY NAME"
            " SELECT qty, shop FROM sales WHERE qty > 25 ORDER BY qty"
        ),
        session(),
    )
    same(read_back(out, "qty"), [30, 30, 40, 40], "qty")
    same(read_back(out, "shop"), [1, 1, 2, 2], "shop")


def test_a_union_by_name_writes_a_null_where_an_arm_has_no_column() raises:
    # Neither side has the other's column, so the answer is two columns wide
    # and each row holds a null in the one its arm did not produce.
    var out = run(
        (
            "SELECT qty FROM sales WHERE qty > 25 UNION ALL BY NAME"
            " SELECT band FROM tiers WHERE band > 90"
        ),
        session(),
    )
    same(gapped(out, "qty"), [40, 30, -1], "qty")
    same(gapped(out, "band"), [-1, -1, 99], "band")


def test_an_order_by_over_a_union_sorts_the_stack() raises:
    same(
        answer(
            (
                "SELECT band FROM tiers UNION ALL SELECT qty FROM sales WHERE"
                " qty > 25 ORDER BY band"
            ),
            "band",
        ),
        [3, 20, 30, 40, 40, 99],
        "the sort runs over both sides",
    )


def test_an_except_keeps_the_rows_the_second_side_lacks() raises:
    same(
        answer("SELECT qty FROM sales EXCEPT SELECT band FROM tiers", "qty"),
        [5, 12, 8, 25, 1, 30, 15],
        "3, 20 and 40 are bands as well",
    )


def test_an_intersect_keeps_the_rows_both_sides_have() raises:
    same(
        answer("SELECT qty FROM sales INTERSECT SELECT band FROM tiers", "qty"),
        [20, 3, 40],
        "99 is a band nobody sold",
    )


def test_an_except_keeps_one_copy_of_a_row_it_keeps() raises:
    # A set operation is over sets, so the two fours the left side has come
    # back as one four whatever the right side holds.
    same(
        answer("SELECT mark FROM gappy EXCEPT SELECT band FROM gaps", "mark"),
        [4, 9, 1],
        "one of each, and the null went with the other side's null",
    )


def test_an_intersect_counts_two_nulls_as_the_same_row() raises:
    # Which is the rule a set operation has and a comparison does not. Written
    # as a join this would be empty, because a null is equal to nothing at all.
    same(
        gapped(
            run(
                "SELECT mark FROM gappy INTERSECT SELECT band FROM gaps",
                session(),
            ),
            "mark",
        ),
        [-1],
        "the null is the only thing both sides wrote",
    )


def test_an_except_all_subtracts_a_copy_for_a_copy() raises:
    # Two threes on the left and one on the right leaves one three, where the
    # set answer leaves none, and that is the whole of the difference between
    # the two spellings.
    same(
        answer(
            "SELECT band FROM dupes EXCEPT ALL SELECT band FROM tiers", "band"
        ),
        [3, 77],
        "one three survives the one the right side had",
    )


def test_an_except_all_over_rows_that_all_differ_is_the_set_answer() raises:
    same(
        answer(
            "SELECT qty FROM sales EXCEPT ALL SELECT band FROM tiers", "qty"
        ),
        [5, 12, 8, 25, 1, 30, 15],
        "no quantity is written twice, so the two spellings agree",
    )


def test_an_except_all_keeps_every_copy_the_right_side_lacks() raises:
    same(
        answer(
            (
                "SELECT band FROM dupes EXCEPT ALL SELECT band FROM tiers"
                " WHERE band > 10"
            ),
            "band",
        ),
        [3, 3, 77],
        "the right side has no three at all, so both of them come back",
    )


def test_an_intersect_all_keeps_as_many_as_the_thinner_side_has() raises:
    same(
        answer(
            "SELECT band FROM dupes INTERSECT ALL SELECT band FROM tiers",
            "band",
        ),
        [3, 20],
        "one three on the right is the smaller of the two counts",
    )


def test_an_intersect_all_over_one_arm_twice_is_that_arm() raises:
    same(
        answer(
            "SELECT band FROM dupes INTERSECT ALL SELECT band FROM dupes",
            "band",
        ),
        [3, 3, 20, 77],
        "the two counts are the same one, so the smaller is the count itself",
    )


def test_a_range_is_a_table_that_reads_no_table() raises:
    var out = run("SELECT * FROM range(5)", session())
    assert_equal(len(out.schema), 1, "one column")
    assert_equal(
        out.schema[0].name, "range", "called after the function, as DuckDB does"
    )
    same(read_back(out, "range"), [0, 1, 2, 3, 4], "five rows out of nothing")


def test_a_generate_series_stops_on_its_bound() raises:
    same(
        read_back(
            run("SELECT * FROM generate_series(3)", session()),
            "generate_series",
        ),
        [0, 1, 2, 3],
        "the one row that tells the two functions apart",
    )


def test_a_range_takes_a_start_and_a_stop_and_a_step() raises:
    same(
        read_back(run("SELECT * FROM range(10, 20, 4)", session()), "range"),
        [10, 14, 18],
        "and stops before the bound",
    )


def test_an_argument_is_folded_before_the_series_is_built() raises:
    same(
        read_back(run("SELECT * FROM range(2 + 3)", session()), "range"),
        [0, 1, 2, 3, 4],
        "simplify folded it on the way down",
    )


def test_a_where_runs_over_a_series() raises:
    same(
        read_back(
            run("SELECT * FROM range(10) WHERE range > 6", session()), "range"
        ),
        [7, 8, 9],
        "the rest of the query does not care where the rows came from",
    )


def test_a_series_can_be_counted() raises:
    same(
        read_back(run("SELECT count(*) AS n FROM range(7)", session()), "n"),
        [7],
        "an aggregate over a source that read nothing",
    )


def test_a_table_function_nobody_wrote_is_refused_by_name() raises:
    with assert_raises(contains="there is no table function called read_csv"):
        _ = run("SELECT * FROM read_csv(1)", session())


def test_an_alias_on_a_table_function_is_refused_by_name() raises:
    with assert_raises(contains="does not lower an alias on a table function"):
        _ = run("SELECT * FROM range(5) AS r(i)", session())


def test_a_between_keeps_the_rows_inside_both_bounds() raises:
    # Closed at both ends, so the five and the twenty are in it.
    same(
        answer("SELECT qty FROM sales WHERE qty BETWEEN 5 AND 20", "qty"),
        [5, 20, 12, 8, 15],
        "qty",
    )


def test_a_between_may_be_computed_on_both_sides() raises:
    # The operand is an expression and so is a bound, which is the case that
    # would have gone wrong if the rewrite had lowered the operand twice.
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty * price BETWEEN qty AND price"
                " * 10"
            ),
            "qty",
        ),
        [5, 3, 8, 1],
        "qty",
    )


def test_an_in_of_one_candidate_runs_as_the_comparison_it_is() raises:
    same(answer("SELECT qty FROM sales WHERE qty IN (25)", "qty"), [25], "qty")


def test_a_plain_or_in_a_where_keeps_both_sides() raises:
    # The shape that had no operator until the connectives were written, and
    # the one everything else here is built out of.
    same(
        answer("SELECT qty FROM sales WHERE qty = 3 OR qty = 25", "qty"),
        [3, 25],
        "qty",
    )


def test_an_in_of_several_candidates_runs_as_the_set_it_is() raises:
    same(
        answer("SELECT qty FROM sales WHERE qty IN (3, 25)", "qty"),
        [3, 25],
        "qty",
    )


def test_an_in_over_text_runs_as_a_set_of_text() raises:
    # The other half of the set lookup, and the half with the lower threshold
    # inside the kernel: a set of three strings is already worth a hash table
    # where a set of three numbers is not. The row with no word in it answers
    # null and is dropped, which is the chain of equalities' answer too.
    same(
        answer(
            "SELECT n FROM words WHERE word IN ('apple', 'grape', 'pear')", "n"
        ),
        [1, 4],
        "n",
    )


def test_a_not_in_keeps_everything_the_in_dropped() raises:
    same(
        answer("SELECT qty FROM sales WHERE qty NOT IN (3, 25)", "qty"),
        [5, 20, 40, 12, 8, 1, 30, 15],
        "qty",
    )


def test_a_not_between_keeps_the_rows_outside_both_bounds() raises:
    # The complement of the BETWEEN test above, down to the closed ends: the
    # five and the twenty are in the range and so are not in this answer.
    same(
        answer("SELECT qty FROM sales WHERE qty NOT BETWEEN 5 AND 20", "qty"),
        [3, 40, 25, 1, 30],
        "qty",
    )


def test_a_minus_in_front_of_a_column_turns_it_over() raises:
    same(
        read_back(run("SELECT -qty AS down FROM sales", session()), "down"),
        [-5, -20, -3, -40, -12, -8, -25, -1, -30, -15],
        "down",
    )


def test_a_minus_in_front_of_an_expression_turns_the_answer_over() raises:
    same(
        read_back(
            run("SELECT -(qty * price) AS down FROM sales", session()), "down"
        ),
        [-50, -40, -21, -40, -60, -72, -75, -100, -120, -90],
        "down",
    )


def test_a_minus_in_a_where_keeps_the_rows_it_says() raises:
    # The same rows `qty > 10` keeps, written the way a reader would not, which
    # is the point: the operator has to run over the column before the
    # comparison rather than the comparison folding the sign away.
    same(
        answer("SELECT qty FROM sales WHERE -qty < -10", "qty"),
        [20, 40, 12, 25, 30, 15],
        "qty",
    )


def test_a_minus_over_a_null_is_a_null() raises:
    # Ten is added before the sign is applied so that no row's real answer is
    # minus one, which is what `gapped` writes a null as. `-mark` on its own
    # would put a genuine minus one next to the two nulls and the assertion
    # would pass whichever of the three the operator got wrong.
    same(
        gapped(
            run("SELECT -(mark + 10) AS down FROM gappy", session()), "down"
        ),
        [-14, -14, -1, -19, -1, -11],
        "down",
    )


def test_a_character_count_counts_characters_rather_than_bytes() raises:
    # Rows two and three are both five characters long and are six and fifteen
    # bytes, so a count that measured the payload would answer them differently.
    same(
        answer("SELECT LENGTH(word) AS c FROM glyphs WHERE n < 5", "c"),
        [3, 5, 5, 0],
        "c",
    )


def test_a_byte_count_counts_bytes_rather_than_characters() raises:
    # The other question, and the reason the two are separate calls. DuckDB
    # answers 3, 6, 15 and 0 for these four rows and so does this.
    same(
        answer("SELECT STRLEN(word) AS c FROM glyphs WHERE n < 5", "c"),
        [3, 6, 15, 0],
        "c",
    )


def test_the_two_names_for_a_character_count_answer_the_same() raises:
    var one = answer("SELECT LENGTH(word) AS c FROM glyphs WHERE n = 3", "c")
    var two = answer("SELECT LEN(word) AS c FROM glyphs WHERE n = 3", "c")
    same(one, [5], "length")
    same(two, [5], "len")


def test_a_row_with_nothing_known_about_it_has_no_length() raises:
    # DuckDB answers null rather than zero, and the two are different things to
    # anything that folds the column afterwards. Both kernels, because the
    # repair is per kernel and one of them could have it without the other.
    for sql in [
        String("SELECT LENGTH(word) AS c FROM glyphs WHERE n = 5"),
        String("SELECT STRLEN(word) AS c FROM glyphs WHERE n = 5"),
    ]:
        var out = run(sql, session())
        var col = out.column("c").as_typed[DType.int64]()
        assert_equal(len(col), 1, "one row")
        assert_true(not col.is_valid(0), "and nothing in it")


def test_a_byte_count_folds_the_way_q27_folds_one() raises:
    # The shape ClickBench q27 is: a length worked out per row and folded per
    # group, with the group key read back beside it. q27 averages `strlen` over
    # `URL`, so this is the byte counting one.
    var out = run(
        "SELECT g, SUM(STRLEN(word)) AS s FROM glyphs GROUP BY g ORDER BY g",
        session(),
    )
    same(read_back(out, "g"), [1, 2], "g")
    same(read_back(out, "s"), [9, 15], "s")


def test_a_length_of_a_number_says_so() raises:
    with assert_raises(contains="'strlen' measures text"):
        _ = run("SELECT STRLEN(n) FROM glyphs", session())
    with assert_raises(contains="'length' measures text"):
        _ = run("SELECT LENGTH(n) FROM glyphs", session())


def test_a_like_with_a_percent_at_the_end_is_a_prefix() raises:
    same(answer("SELECT n FROM words WHERE word LIKE 'a%'", "n"), [1, 2], "n")


def test_a_like_with_a_percent_at_the_start_is_a_suffix() raises:
    same(
        answer("SELECT n FROM words WHERE word LIKE '%e'", "n"), [1, 4, 7], "n"
    )


def test_a_like_with_a_percent_at_both_ends_is_a_substring() raises:
    same(answer("SELECT n FROM words WHERE word LIKE '%an%'", "n"), [3], "n")


def test_a_like_with_two_runs_reads_them_in_the_order_written() raises:
    # The pair is the one search where the pattern's order is the whole
    # question, so both ways round are asserted. `pineapple` holds an n and an
    # e, and only one of the two orders is in it.
    same(answer("SELECT n FROM words WHERE word LIKE '%n%e%'", "n"), [7], "n")
    same(answer("SELECT n FROM words WHERE word LIKE '%e%n%'", "n"), [], "n")


def test_a_like_with_no_wildcard_is_an_equality() raises:
    same(answer("SELECT n FROM words WHERE word LIKE 'grape'", "n"), [4], "n")


def test_a_like_against_one_percent_keeps_every_row_that_is_not_null() raises:
    # The row with nothing in it is kept and the row with nothing known about it
    # is not, which is the difference a search written the easy way loses.
    same(
        answer("SELECT n FROM words WHERE word LIKE '%'", "n"),
        [1, 2, 3, 4, 5, 7],
        "n",
    )
    same(
        answer("SELECT n FROM words WHERE word LIKE '%%'", "n"),
        [1, 2, 3, 4, 5, 7],
        "n",
    )


def test_a_like_against_nothing_keeps_the_row_holding_nothing() raises:
    same(answer("SELECT n FROM words WHERE word LIKE ''", "n"), [5], "n")


def test_a_not_like_drops_the_matches_and_the_null_with_them() raises:
    # A null is not a match and its negation is not one either, both being null,
    # so the sixth row is missing from this answer and from the one above it.
    same(
        answer("SELECT n FROM words WHERE word NOT LIKE 'a%'", "n"),
        [3, 4, 5, 7],
        "n",
    )


def test_a_like_in_a_select_list_is_a_column_of_answers() raises:
    same(
        truths(
            run("SELECT word LIKE '%e' AS ends FROM words", session()), "ends"
        ),
        [1, 0, 0, 1, 0, -1, 1],
        "ends",
    )


def test_a_like_with_an_underscore_in_it_keeps_the_rows_it_names() raises:
    # `a_p%` is apple and not apricot, the underscore standing for exactly one
    # character and not for the two apricot would need.
    same(answer("SELECT n FROM words WHERE word LIKE 'a_p%'", "n"), [1], "n")
    # Five characters and nothing else, which is apple and grape. The empty row
    # has none and the null has no answer at all.
    same(
        answer("SELECT n FROM words WHERE word LIKE '_____'", "n"), [1, 4], "n"
    )


def test_a_like_with_a_run_in_the_middle_keeps_the_rows_it_names() raises:
    # `a%e` is a prefix and a suffix at once, which is more than either kernel
    # can say on its own and which the matcher behind them answers. Reading it
    # as the prefix alone would have kept apricot as well.
    same(answer("SELECT n FROM words WHERE word LIKE 'a%e'", "n"), [1], "n")
    same(answer("SELECT n FROM words WHERE word LIKE '%an%n%'", "n"), [3], "n")
    same(answer("SELECT n FROM words WHERE word LIKE 'p_ne%le'", "n"), [7], "n")


def test_a_like_with_wildcards_in_a_select_list_answers_every_row() raises:
    # Including the null, which is null and not false, the same as it is for
    # the four searches.
    same(
        truths(
            run("SELECT word LIKE '%a_e' AS hit FROM words", session()), "hit"
        ),
        [0, 0, 0, 1, 0, -1, 0],
        "hit",
    )


def test_a_like_against_a_column_is_refused() raises:
    with assert_raises(contains="pattern of a LIKE has to be written out"):
        _ = run("SELECT n FROM words WHERE word LIKE word", session())


def test_an_escape_makes_a_wildcard_into_a_byte_to_look_for() raises:
    # The three things an escape is put in front of, and then a byte that was
    # never a wildcard, where DuckDB drops the escape rather than refusing.
    same(_marks("a!%b", "!"), [1, 0, 0, 0, 0, 0, 0, 0], "hit")
    same(_marks("a!_b", "!"), [0, 0, 1, 0, 0, 0, 0, 0], "hit")
    same(_marks("a!!b", "!"), [0, 0, 0, 1, 0, 0, 0, 0], "hit")
    same(_marks("a!cb", "!"), [0, 0, 0, 0, 1, 0, 0, 0], "hit")


def test_an_escaped_pattern_keeps_the_wildcards_it_did_not_escape() raises:
    same(_marks("%!%%", "!"), [1, 0, 0, 0, 0, 1, 1, 1], "hit")
    same(_marks("_!%_", "!"), [1, 0, 0, 0, 0, 0, 0, 0], "hit")


def test_an_escape_that_is_itself_a_wildcard_leaves_no_wildcard() raises:
    # `ESCAPE '%'` is legal and means every `%` in the pattern is an escape, so
    # `a%%b` is the three bytes `a%b` and not a prefix and a suffix.
    same(_marks("a%%b", "%"), [1, 0, 0, 0, 0, 0, 0, 0], "hit")
    same(_marks("a__b", "_"), [0, 0, 1, 0, 0, 0, 0, 0], "hit")


def test_an_empty_escape_reads_the_pattern_as_a_plain_one() raises:
    # Which is DuckDB's rule too, rather than a refusal. The pattern is then
    # the prefix and the suffix it looks like.
    same(_marks("a%b", ""), [1, 1, 1, 1, 1, 0, 1, 0], "hit")
    same(_marks("a!%b", ""), [0, 0, 0, 1, 0, 0, 0, 0], "hit")


def test_a_negated_like_takes_an_escape_the_same_way() raises:
    same(
        truths(
            run(
                "SELECT word NOT LIKE 'a!%' ESCAPE '!' AS hit FROM words",
                session(),
            ),
            "hit",
        ),
        [1, 1, 1, 1, 1, -1, 1],
        "hit",
    )


def test_a_pattern_that_ends_with_its_escape_is_refused() raises:
    # There is no byte after it to make literal, so the pattern is written
    # wrong. DuckDB raises for the same pattern but only once a row walks far
    # enough into it, so `'ab' LIKE 'ab!' ESCAPE '!'` is false there. Refusing
    # it while the plan is built is the divergence, and it is from an error
    # that depends on the data rather than from an answer.
    with assert_raises(contains="ends with its escape character"):
        _ = _marks("ab!", "!")
    # And an escaped escape at the end is not one, the last byte having been
    # spoken for by the one in front of it.
    same(_marks("a!!%", "!"), [0, 0, 0, 1, 0, 0, 0, 0], "hit")


def test_an_escape_that_is_not_one_character_is_refused() raises:
    with assert_raises(contains="ESCAPE of a LIKE is one character"):
        _ = _marks("a%b", "!!")
    with assert_raises(contains="ESCAPE of a LIKE has to be written out"):
        _ = run(
            "SELECT n FROM words WHERE word LIKE 'a%' ESCAPE word", session()
        )


def test_an_escape_on_an_operator_that_has_no_escaping_form_is_refused() raises:
    # DuckDB says the same about this one, a custom escape on a SIMILAR TO
    # being unimplemented there rather than meaningless.
    with assert_raises(contains="ESCAPE on an operator"):
        _ = run(
            "SELECT n FROM words WHERE word SIMILAR TO 'a' ESCAPE '!'",
            session(),
        )
    # An ILIKE has the form and is refused for the reason an ILIKE always is.
    with assert_raises(contains="without regard to case"):
        _ = run(
            "SELECT n FROM words WHERE word ILIKE 'a!%' ESCAPE '!'", session()
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
