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

Part 3 of 3. The fixtures are in tests/support/sql_run.mojo.
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


def test_a_subquery_in_a_select_list_runs() raises:
    same(
        answer(
            (
                "SELECT (SELECT max(rate) FROM tiers) AS top FROM sales"
                " WHERE qty = 1"
            ),
            "top",
        ),
        [900],
        "top",
    )


def test_a_subquery_in_a_having_runs() raises:
    # Half of everything sold is 79.5, shop one sold 75 and shop two sold 84.
    # The subquery is one value for the whole query, and the only thing that
    # makes it different from one written in a WHERE is that the clause reading
    # it is above the aggregate and the column has to be joined on up there.
    same(
        answer(
            (
                "SELECT shop FROM sales GROUP BY shop HAVING sum(qty) >"
                " (SELECT sum(qty) / 2 FROM sales)"
            ),
            "shop",
        ),
        [2],
        "shop",
    )


def test_a_subquery_in_the_select_list_of_a_fold_runs() raises:
    # The same join in the same place, read by the projection rather than by
    # the HAVING. The smallest quantity is one, so each shop's total loses one.
    var got = run(
        (
            "SELECT shop, sum(qty) - (SELECT min(qty) FROM sales) AS total FROM"
            " sales GROUP BY shop ORDER BY shop"
        ),
        session(),
    )
    same(read_back(got, "shop"), [1, 2], "shop")
    same(read_back(got, "total"), [74, 83], "total")


def test_a_subquery_in_a_having_that_reads_the_group_is_refused() raises:
    # Correlated rather than not, and the correlation is what there is no
    # answer for here: the fold under the FROM answers one value per outer row
    # and the outer rows this one would need are the rows the aggregate folded
    # away.
    with assert_raises(
        contains="written above the aggregate, where the columns it correlates"
    ):
        _ = run(
            (
                "SELECT shop FROM sales GROUP BY shop HAVING sum(qty) >"
                " (SELECT sum(rate) FROM tiers WHERE band = shop)"
            ),
            session(),
        )


def test_a_correlated_subquery_folds_per_outer_row() raises:
    # Shop one sold 75 and shop two sold 84, so both floors are under their
    # own shop's total. Shop three sold nothing, its total is a null, and a
    # comparison against a null keeps no row.
    same(
        answer(
            (
                "SELECT shop FROM shops WHERE floor < (SELECT sum(qty) FROM"
                " sales WHERE sales.shop = shops.shop)"
            ),
            "shop",
        ),
        [1, 2],
        "shop",
    )


def test_a_correlated_subquery_in_a_select_list_runs() raises:
    var got = run(
        (
            "SELECT shop, (SELECT sum(qty) FROM sales WHERE sales.shop ="
            " shops.shop) AS total FROM shops"
        ),
        session(),
    )
    same(read_back(got, "shop"), [1, 2, 3], "shop")
    same(gapped(got, "total"), [75, 84, -1], "total")


def test_a_correlated_subquery_keeps_its_own_condition_under_the_fold() raises:
    # `qty > 10` reads the subquery's table alone, so it runs once under the
    # aggregate rather than once per outer row, and the answer is the same.
    var got = run(
        (
            "SELECT shop, (SELECT sum(qty) FROM sales WHERE sales.shop ="
            " shops.shop AND qty > 10) AS big FROM shops"
        ),
        session(),
    )
    same(gapped(got, "big"), [67, 75, -1], "big")


def test_a_correlated_subquery_may_compute_over_its_fold() raises:
    var got = run(
        (
            "SELECT shop, (SELECT max(qty) + 1 FROM sales WHERE sales.shop ="
            " shops.shop) AS top FROM shops"
        ),
        session(),
    )
    same(gapped(got, "top"), [31, 41, -1], "top")


def test_a_correlated_count_answers_zero_for_an_empty_group() raises:
    # The count bug. Shop three sells nothing, so the left join pads its row,
    # and the null the padding writes is read as the zero a count of nothing
    # answers rather than being handed on as a null.
    var got = run(
        (
            "SELECT shop, (SELECT count(qty) FROM sales WHERE sales.shop ="
            " shops.shop) AS n FROM shops"
        ),
        session(),
    )
    same(read_back(got, "shop"), [1, 2, 3], "shop")
    same(read_back(got, "n"), [5, 5, 0], "n")


def test_a_correlated_count_of_zero_is_a_row_a_filter_keeps() raises:
    # The wrong answer the reading exists to stop. A null is a row no filter
    # keeps, so the shop that sold nothing used to be the one row this query
    # could not find.
    same(
        answer(
            (
                "SELECT shop FROM shops WHERE (SELECT count(qty) FROM sales"
                " WHERE sales.shop = shops.shop) = 0"
            ),
            "shop",
        ),
        [3],
        "shop",
    )


def test_a_correlated_count_star_counts_rows_and_not_values() raises:
    var got = run(
        (
            "SELECT shop, (SELECT count(*) FROM sales WHERE sales.shop ="
            " shops.shop AND qty > 10) AS n FROM shops"
        ),
        session(),
    )
    same(read_back(got, "n"), [3, 3, 0], "n")


def test_a_correlated_distinct_count_answers_zero_too() raises:
    var got = run(
        (
            "SELECT shop, (SELECT count(DISTINCT qty) FROM sales WHERE"
            " sales.shop = shops.shop) AS n FROM shops"
        ),
        session(),
    )
    same(read_back(got, "n"), [5, 5, 0], "n")


def test_a_correlated_count_inside_a_larger_value_is_refused_by_name() raises:
    with assert_raises(contains="counts inside a larger expression"):
        _ = run(
            (
                "SELECT shop, (SELECT count(qty) + 1 FROM sales WHERE"
                " sales.shop = shops.shop) AS n FROM shops"
            ),
            session(),
        )


def test_a_correlated_subquery_read_another_way_is_refused_by_name() raises:
    with assert_raises(contains="which is the dependent join"):
        _ = run(
            (
                "SELECT shop FROM shops WHERE floor < (SELECT sum(qty) FROM"
                " sales WHERE sales.shop > shops.shop)"
            ),
            session(),
        )


def test_a_correlation_written_bare_is_still_a_correlation() raises:
    # `band` is a column of `tiers` and not one of `sales`, so the bare name
    # inside the subquery is the outer query's and the subquery is asking for
    # a different average per band. Nothing in the query says so twice.
    same(
        answer(
            (
                "SELECT band FROM tiers WHERE rate > (SELECT avg(price) FROM"
                " sales WHERE qty = band)"
            ),
            "band",
        ),
        [3, 20, 40],
        "band",
    )


def test_a_correlation_written_bare_reads_the_same_as_a_qualified_one() raises:
    var bare = run(
        (
            "SELECT band, (SELECT sum(price) FROM sales WHERE qty = band) AS"
            " took FROM tiers"
        ),
        session(),
    )
    var qualified = run(
        (
            "SELECT band, (SELECT sum(price) FROM sales WHERE qty ="
            " tiers.band) AS took FROM tiers"
        ),
        session(),
    )
    same(gapped(bare, "took"), [7, 2, 1, -1], "took")
    same(gapped(qualified, "took"), gapped(bare, "took"), "took")


def test_a_bare_name_the_subquery_has_itself_is_the_subquerys_own() raises:
    # Both frames have a `shop`, so the bare one is the inner one, which makes
    # this uncorrelated and gives every outer row the same total.
    var got = run(
        (
            "SELECT shop, (SELECT sum(qty) FROM sales WHERE shop = 1) AS one"
            " FROM shops"
        ),
        session(),
    )
    same(read_back(got, "shop"), [1, 2, 3], "shop")
    same(read_back(got, "one"), [75, 75, 75], "one")


def test_a_bare_name_neither_query_has_is_refused_as_a_missing_column() raises:
    with assert_raises(contains="there is no column named 'nope'"):
        _ = run(
            (
                "SELECT band FROM tiers WHERE rate > (SELECT avg(price) FROM"
                " sales WHERE qty = nope)"
            ),
            session(),
        )


def test_a_bare_correlation_over_a_derived_table_is_not_claimed_yet() raises:
    # The columns a subquery in a FROM hands out are decided by lowering it,
    # and this question is asked before anything is lowered, so a bare name is
    # left alone rather than guessed at. Which is a refusal and not an answer.
    with assert_raises(contains="there is no column named 'band'"):
        _ = run(
            (
                "SELECT band FROM tiers WHERE rate > (SELECT avg(price) FROM"
                " (SELECT qty, price FROM sales) AS s WHERE qty = band)"
            ),
            session(),
        )


def test_a_table_named_on_both_sides_of_a_correlation_shadows() raises:
    # TPC-H q17's shape. The subquery's FROM is lowered into the caller's
    # scope, because a condition reading both sides of the correlation can only
    # be written where both sides are in reach, so there are two `sales` in one
    # scope here. SQL says the inner one shadows the outer, which makes `qty`
    # and `price` under the subquery its own copy's.
    #
    # Read the other way the subquery would fold over every row the outer
    # `sales` handed it, which is 246 for every band rather than 7, 2 and 1 for
    # bands 3, 20 and 40, and no band would be kept at all.
    same(
        answer(
            (
                "SELECT band FROM sales, tiers WHERE qty = band AND rate >"
                " (SELECT sum(price) * 100 FROM sales WHERE qty = band)"
            ),
            "band",
        ),
        [40],
        "band",
    )


def test_the_shadowed_table_is_back_under_its_own_name_after() raises:
    # The line between the two queries goes back where it was once the
    # subquery is lowered, so `sales` written in the rest of the WHERE is the
    # outer one again and means what it meant before the subquery was written.
    same(
        answer(
            (
                "SELECT band FROM sales, tiers WHERE rate > (SELECT sum(price)"
                " * 100 FROM sales WHERE qty = band) AND sales.qty = band"
            ),
            "band",
        ),
        [40],
        "band",
    )


def test_the_same_table_twice_in_one_from_is_still_refused() raises:
    # Shadowing is across the line and not within it. Two relations of the same
    # name in one FROM is a query that cannot say which it means, and letting
    # the second shadow the first there would answer half of it silently.
    with assert_raises(
        contains="'sales' is the name of more than one table in this FROM"
    ):
        _ = run("SELECT qty FROM sales, sales", session())


def test_a_qualifier_inside_a_subquery_means_the_subquerys_own_table() raises:
    # `sales.qty` is written where two relations are called `sales`, and the
    # one it means is the subquery's. Written out it reads the same as the bare
    # spelling above, which is the point: the qualifier is not what decides it.
    same(
        answer(
            (
                "SELECT band FROM sales, tiers WHERE qty = band AND rate >"
                " (SELECT sum(price) * 100 FROM sales WHERE sales.qty = band)"
            ),
            "band",
        ),
        [40],
        "band",
    )


def test_a_name_two_of_the_subquerys_tables_have_is_ambiguous() raises:
    # `shop` is on both tables the subquery reads and on neither table the
    # query around it reads, so there is no innermost relation to pin it to and
    # the ambiguity is the subquery's own. Shadowing decides between two levels
    # and has nothing to say within one, which is what this holds it to.
    with assert_raises(
        contains="is the name of more than one column on one side"
    ):
        _ = run(
            (
                "SELECT band FROM tiers WHERE rate > (SELECT avg(price) FROM"
                " sales, shops WHERE shop = 1 AND qty = band)"
            ),
            session(),
        )


def test_a_subquery_over_no_table_runs() raises:
    same(
        answer(
            "SELECT qty FROM sales WHERE qty > (SELECT 24) ORDER BY qty", "qty"
        ),
        [25, 30, 40],
        "qty",
    )


def test_a_fold_over_no_rows_hands_out_one_null_row() raises:
    # A fold with no GROUP BY is one group whether or not anything was read, so
    # the answer is one row and the maximum of nothing is a null in it.
    var got = run(
        "SELECT max(band) AS top FROM tiers WHERE band > 1000", session()
    )
    assert_equal(len(got), 1)
    var col = got.column("top").as_typed[DType.int64]()
    assert_true(not col.is_valid(0), "the maximum of nothing")


def test_a_count_over_no_rows_hands_out_a_zero() raises:
    # The one fold that finds an answer in nothing, because counting what is
    # there is a question an empty column can answer.
    same(
        answer("SELECT count(*) AS n FROM tiers WHERE band > 1000", "n"),
        [0],
        "n",
    )


def test_a_sum_over_no_rows_is_a_null_and_not_a_zero() raises:
    # Where SQL and pandas part. A sum of nothing is zero in pandas, because
    # zero is what adding no numbers gives, and it is null in SQL, because a
    # total of nothing is not a total. The plan says which was asked for.
    same(
        gapped(
            run(
                "SELECT sum(band) AS total FROM tiers WHERE band > 1000",
                session(),
            ),
            "total",
        ),
        [-1],
        "total",
    )


def test_a_sum_over_no_rows_carries_its_null_upward() raises:
    # TPC-H q17's shape, which is a sum divided by a constant over a query that
    # keeps no rows at some scales. A zero there would come out as a zero and a
    # null comes out as a null, so the whole answer turns on the row above.
    same(
        gapped(
            run(
                "SELECT sum(band) * 2 AS doubled FROM tiers WHERE band > 1000",
                session(),
            ),
            "doubled",
        ),
        [-1],
        "doubled",
    )


def test_a_sum_over_rows_is_still_the_sum() raises:
    # The other half of it. Nothing about an input with rows in it changed.
    same(
        answer("SELECT sum(band) AS total FROM tiers", "total"),
        [162],
        "total",
    )


def test_a_sum_over_an_empty_group_does_not_arise() raises:
    # A group exists because a row made it, so a GROUP BY over an input that
    # keeps nothing has no groups rather than a group of nothing, and the sum
    # that is null above has no row to be null in.
    var got = run(
        (
            "SELECT band, sum(rate) AS total FROM tiers WHERE band > 1000 GROUP"
            " BY band"
        ),
        session(),
    )
    assert_equal(len(got), 0)


def test_a_sum_over_nothing_but_nulls_is_a_null_as_well() raises:
    # The other half of the same disagreement, and the half that is not free.
    # Rows arrived and every value in them was missing, so the sum added
    # nothing and the zero it is holding is not an answer. What tells that
    # apart from values that really summed to zero is a count of what was
    # added, which is a second state slot beside the sum. Issue #836.
    same(
        gapped(
            run(
                "SELECT sum(mark) AS total FROM gappy WHERE mark IS NULL",
                session(),
            ),
            "total",
        ),
        [-1],
        "total",
    )


def test_a_count_over_nothing_but_nulls_is_still_a_zero() raises:
    # A count is the fold the two front ends agree about, so it is a zero
    # beside the null above rather than a null of its own.
    same(
        answer(
            "SELECT count(mark) AS seen FROM gappy WHERE mark IS NULL", "seen"
        ),
        [0],
        "seen",
    )


def test_a_group_of_nothing_but_nulls_sums_to_null_and_the_rest_do_not() raises:
    # Two groups, one of which holds nothing but nulls. That group is there
    # because rows made it, and what the sum has to say about it is nothing,
    # while the group beside it is the number it always was.
    same(
        gapped(
            run(
                (
                    "SELECT mark IS NULL AS gap, sum(mark) AS total FROM gappy"
                    " GROUP BY gap ORDER BY gap"
                ),
                session(),
            ),
            "total",
        ),
        [18, -1],
        "total",
    )


def test_a_sum_of_an_expression_over_nothing_but_nulls_is_null() raises:
    # The count has to be a count of what the sum read rather than of the
    # column it came from, since a fold can carry an operation and reduce what
    # that produces. Doubling a null is a null, so there is still nothing here.
    same(
        gapped(
            run(
                "SELECT sum(mark * 2) AS total FROM gappy WHERE mark IS NULL",
                session(),
            ),
            "total",
        ),
        [-1],
        "total",
    )


def test_a_window_over_a_partition_of_nothing_but_nulls_is_null() raises:
    # The same question asked of the operator that writes one value on every
    # row. A partition is never empty, since it is a partition because a row is
    # in it, so this is the only thing the mark decides on a window. The rows
    # where the mark is null read a partition that held nothing. Issue #877.
    same(
        gapped(
            run(
                (
                    "SELECT sum(mark) OVER (PARTITION BY mark IS NULL) AS total"
                    " FROM gappy ORDER BY mark IS NULL"
                ),
                session(),
            ),
            "total",
        ),
        [18, 18, 18, 18, -1, -1],
        "total",
    )


def test_a_window_over_a_whole_frame_of_nothing_but_nulls_is_null() raises:
    # No partition keys is one partition, and every row of it is missing.
    same(
        gapped(
            run(
                (
                    "SELECT sum(mark) OVER () AS whole FROM gappy WHERE mark IS"
                    " NULL"
                ),
                session(),
            ),
            "whole",
        ),
        [-1, -1],
        "whole",
    )


def test_a_count_window_over_nothing_but_nulls_is_still_a_zero() raises:
    # The fold both front ends agree about, so the mark is never set on it and
    # a partition of nothing but nulls counts zero rather than answering null.
    same(
        answer(
            (
                "SELECT count(mark) OVER (PARTITION BY mark IS NULL) AS c FROM"
                " gappy ORDER BY mark IS NULL"
            ),
            "c",
        ),
        [4, 4, 4, 4, 0, 0],
        "c",
    )


def test_first_and_last_report_the_row_and_not_the_value() raises:
    # `gappy` is 4, 4, null, 9, null, 1 in that order, and the group is the
    # three rows in the middle of it: a null, a nine and a null. DuckDB answers
    # null for both, because both name a row and read whatever is in it, and
    # this is the case that separated them from `any_value`. See #888.
    same(
        gapped(
            run(
                (
                    "SELECT first(mark) AS f FROM gappy"
                    " GROUP BY mark IS NULL OR mark = 9"
                    " ORDER BY mark IS NULL OR mark = 9"
                ),
                session(),
            ),
            "f",
        ),
        [4, -1],
        "f",
    )
    same(
        gapped(
            run(
                (
                    "SELECT last(mark) AS l FROM gappy"
                    " GROUP BY mark IS NULL OR mark = 9"
                    " ORDER BY mark IS NULL OR mark = 9"
                ),
                session(),
            ),
            "l",
        ),
        [1, -1],
        "l",
    )


def test_any_value_is_the_one_that_passes_over_a_null() raises:
    # The same two groups and the same three rows in the second of them, where
    # the nine is the only value there is. DuckDB answers nine here, which is
    # why `any_value` kept the fold `first` used to share with it.
    same(
        answer(
            (
                "SELECT any_value(mark) AS a FROM gappy"
                " GROUP BY mark IS NULL OR mark = 9"
                " ORDER BY mark IS NULL OR mark = 9"
            ),
            "a",
        ),
        [4, 9],
        "a",
    )


def test_first_over_a_whole_column_of_nothing_but_nulls_is_null() raises:
    # No group by, so the rows are the whole of one fold, and the two rows that
    # reach it are both null. DuckDB answers null for all three here, since even
    # the one that passes over a null has nothing left to report.
    same(
        gapped(
            run(
                "SELECT first(mark) AS f FROM gappy WHERE mark IS NULL",
                session(),
            ),
            "f",
        ),
        [-1],
        "f",
    )
    same(
        gapped(
            run(
                "SELECT last(mark) AS l FROM gappy WHERE mark IS NULL",
                session(),
            ),
            "l",
        ),
        [-1],
        "l",
    )
    same(
        gapped(
            run(
                "SELECT any_value(mark) AS a FROM gappy WHERE mark IS NULL",
                session(),
            ),
            "a",
        ),
        [-1],
        "a",
    )


def test_first_over_no_rows_at_all_is_null() raises:
    # A fold over nothing, which is the other half of `EMPTY_IS_NULL` and is
    # not what #888 changed. Kept beside the three above because a row kind
    # that answered its slot's zero here would look like a first row holding
    # one.
    same(
        gapped(
            run(
                "SELECT first(mark) AS f FROM gappy WHERE mark > 1000",
                session(),
            ),
            "f",
        ),
        [-1],
        "f",
    )
    same(
        gapped(
            run(
                "SELECT last(mark) AS l FROM gappy WHERE mark > 1000", session()
            ),
            "l",
        ),
        [-1],
        "l",
    )


def test_a_subquery_over_no_rows_keeps_no_rows() raises:
    # The fold above is one null row, the cross join puts that null on every
    # row, and a comparison against a null keeps nothing.
    var got = run(
        (
            "SELECT qty FROM sales WHERE qty > (SELECT max(band) FROM"
            " tiers WHERE band > 1000)"
        ),
        session(),
    )
    assert_equal(len(got), 0)


def test_an_exists_under_an_or_keeps_what_either_side_keeps() raises:
    # The subquery has no row in it, so the `EXISTS` is false on every row and
    # what is left is the other side of the `OR`.
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty > 25 OR EXISTS"
                " (SELECT band FROM tiers WHERE band > 1000) ORDER BY qty"
            ),
            "qty",
        ),
        [30, 40],
        "qty",
    )


def test_an_exists_beside_a_condition_with_nothing_to_pair_is_counted() raises:
    # Written where the `WHERE` is the `AND` of it and other things, which is
    # where a correlated one is a semi join. This one has no equality to pair
    # on, so it is a value there too and it is true, because the table has rows.
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE EXISTS (SELECT band FROM tiers)"
                " AND qty > 25 ORDER BY qty"
            ),
            "qty",
        ),
        [30, 40],
        "qty",
    )


def test_a_not_exists_over_an_empty_subquery_keeps_them_all() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE NOT EXISTS"
                " (SELECT band FROM tiers WHERE band > 1000) ORDER BY qty"
            ),
            "qty",
        ),
        [1, 3, 5, 8, 12, 15, 20, 25, 30, 40],
        "qty",
    )


def test_an_exists_written_in_the_select_list_answers_on_every_row() raises:
    same(
        truths(
            run(
                (
                    "SELECT qty, EXISTS (SELECT band FROM tiers WHERE"
                    " band > 30) AS any_big FROM sales"
                ),
                session(),
            ),
            "any_big",
        ),
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "any_big",
    )


def test_an_exists_over_a_fold_that_read_nothing_is_true() raises:
    # The fold has no GROUP BY and so hands out one row whatever it read, and
    # SQL says an EXISTS over one row is true even when the row is a null. This
    # is the shape the semi join refuses by name and the counting gets right.
    assert_equal(
        len(
            run(
                (
                    "SELECT qty FROM sales WHERE EXISTS"
                    " (SELECT max(band) FROM tiers WHERE band > 1000)"
                ),
                session(),
            )
        ),
        10,
    )


def test_an_exists_over_a_limit_of_none_is_false() raises:
    assert_equal(
        len(
            run(
                (
                    "SELECT qty FROM sales WHERE EXISTS"
                    " (SELECT band FROM tiers LIMIT 0)"
                ),
                session(),
            )
        ),
        0,
    )


def test_a_cross_join_onto_one_row_runs() raises:
    # One right row adds a column and moves nothing, so it is a constant per
    # right column rather than the whole frame join the general case needs.
    same(
        answer(
            (
                "SELECT qty FROM sales CROSS JOIN"
                " (SELECT max(band) AS top FROM tiers)"
                " WHERE qty > top - 60 ORDER BY qty"
            ),
            "qty",
        ),
        [40],
        "qty",
    )


def test_the_one_row_of_a_cross_join_is_readable() raises:
    same(
        answer(
            (
                "SELECT top FROM sales CROSS JOIN"
                " (SELECT min(band) AS top FROM tiers) WHERE qty = 1"
            ),
            "top",
        ),
        [3],
        "top",
    )


def test_a_cross_join_onto_more_than_one_row_says_why_it_is_refused() raises:
    with assert_raises(contains="right side of 4 rows"):
        _ = run("SELECT qty FROM sales CROSS JOIN tiers", session())


def test_a_right_join_has_no_operator_yet_either() raises:
    with assert_raises(contains="breaker rather than an operator"):
        _ = run(
            "SELECT shop FROM sales RIGHT JOIN shops USING (shop)", session()
        )


def test_count_distinct_over_a_partition_counts_the_values() raises:
    # A window holds its whole partition, so the fold it runs there is the
    # whole frame one and a distinct count is no harder than a sum.
    same(
        answer(
            (
                "SELECT count(DISTINCT shop) OVER (PARTITION BY shop) AS n"
                " FROM sales"
            ),
            "n",
        ),
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "one shop a partition",
    )


def test_count_distinct_over_a_partition_does_not_count_a_null() raises:
    same(
        answer(
            "SELECT count(DISTINCT mark) OVER () AS n FROM gappy",
            "n",
        ),
        [3, 3, 3, 3, 3, 3],
        "three values and two nulls",
    )


def test_count_distinct_counts_the_values_rather_than_the_rows() raises:
    # It used to answer count(band), which is 4.
    same(answer("SELECT count(band) AS n FROM dupes", "n"), [4], "rows")
    same(
        answer("SELECT count(DISTINCT band) AS n FROM dupes", "n"),
        [3],
        "values",
    )


def test_count_distinct_does_not_count_a_null() raises:
    same(
        answer("SELECT count(DISTINCT mark) AS n FROM gappy", "n"),
        [3],
        "three values",
    )
    same(answer("SELECT count(mark) AS n FROM gappy", "n"), [4], "not null")


def test_an_average_of_large_ids_is_not_a_wrapped_sum_divided() raises:
    """The answer is 4e18 plus five. Through the wrapped sum it was 9.3e17.

    This is #673, which was found on ClickBench q3, `SELECT AVG(UserID) FROM
    hits`, where the answer came back negative over a column with no negative
    value in it. The plan splits a mean into a running sum and a running count
    so the state folds across chunks, and that sum has to be taken in float64
    because it is a numerator and not a sum anybody asked for.
    """
    var out = run("SELECT avg(user_id) AS m FROM visits", session())
    assert_equal(len(out), 1, "one row")
    var got = out.column("m").as_typed[DType.float64]()[0]
    var want = Float64(4_000_000_000_000_000_005)
    assert_true(abs(got - want) <= 1e-9 * want, "the mean of the six ids")


def test_a_grouped_average_of_large_ids_is_not_either() raises:
    """Each site's three ids add up to 1.2e19, which wraps to minus 6.4e18, so
    the two group means were negative as well."""
    var out = run(
        (
            "SELECT site, avg(user_id) AS m FROM visits GROUP BY site ORDER BY"
            " site"
        ),
        session(),
    )
    assert_equal(len(out), 2, "two sites")
    var got = out.column("m").as_typed[DType.float64]()
    var base = Float64(4_000_000_000_000_000_000)
    assert_true(abs(got[0] - (base + 4.0)) <= 1e-9 * base, "site one")
    assert_true(abs(got[1] - (base + 6.0)) <= 1e-9 * base, "site two")


def test_a_sum_of_large_ids_still_wraps() raises:
    """pandas wraps an int64 sum and firepanda follows it, so the fix for the
    two above is a flag on the one slot a mean owns rather than a wider
    accumulator everywhere."""
    var out = run("SELECT sum(user_id) AS t FROM visits", session())
    assert_equal(len(out), 1, "one row")
    assert_equal(
        out.column("t").as_typed[DType.int64]()[0],
        Int64(5_553_255_926_290_448_414),
        "six times 4e18 plus 30, wrapped back round",
    )


def test_an_average_of_instants_is_an_instant() raises:
    """`avg` over a column of times answered a count of seconds from SQL and a
    point in time from `DataFrame.group_by`, on the same column. See #552.

    The frame API is the specification here, because that is the half with a
    pandas to be measured against, and the test that runs the same reduction
    both ways lives beside the node in `test_group_node.mojo`. What this checks
    is that a query reaches it.
    """
    var out = run(
        "SELECT crew, avg(start) AS m FROM shifts GROUP BY crew ORDER BY crew",
        session(),
    )
    assert_equal(len(out), 2, "two crews")
    assert_true(
        out.schema[1].dtype == LogicalType.timestamp(TimeUnit.SECOND),
        "the mean of a set of points in time is a point in time",
    )
    same(read_back(out, "m"), [200, 20], "each crew's average start")


def test_an_average_of_instants_over_the_whole_table_is_one_too() raises:
    """The other operator. A query with no group by reduces the whole column
    through `Reduce`, which keeps its own state and had its own answer."""
    var out = run("SELECT avg(start) AS m FROM shifts", session())
    assert_equal(len(out), 1, "one row")
    assert_true(
        out.schema[0].dtype == LogicalType.timestamp(TimeUnit.SECOND),
        "and it is still an instant",
    )
    same(read_back(out, "m"), [110], "the six starts average to 110.33")


def test_an_average_of_spans_is_a_span() raises:
    var out = run("SELECT avg(span) AS m FROM shifts", session())
    assert_true(
        out.schema[0].dtype == LogicalType.duration(TimeUnit.SECOND),
        "the mean of a set of lengths is a length",
    )
    same(read_back(out, "m"), [110], "truncated towards zero, as pandas does")


def test_a_total_of_spans_is_a_span_and_a_total_of_instants_is_refused() raises:
    """A sum is the reduction the two halves of the table differ on. Adding two
    lengths of time gives a length of time, and adding two points in time gives
    nothing, so one of these answers and the other says why not."""
    var out = run("SELECT sum(span) AS t FROM shifts", session())
    assert_true(
        out.schema[0].dtype == LogicalType.duration(TimeUnit.SECOND),
        "a total of lengths keeps its units",
    )
    same(read_back(out, "t"), [662], "all six added up")
    with assert_raises(contains="a sum over datetime64[s] has no answer"):
        _ = run("SELECT sum(start) AS t FROM shifts", session())


def test_a_median_comes_back_through_sql() raises:
    # Not a distinct count, and here for the same reason: it does not fold
    # either, it went through the same hole, and nothing covered it.
    var out = run("SELECT median(qty) AS mid FROM sales", session())
    assert_equal(len(out), 1, "one row")
    assert_equal(
        out.column("mid").as_typed[DType.float64]()[0],
        Float64(13.5),
        "between 12 and 15",
    )


def test_a_spread_comes_back_through_sql() raises:
    # qty is 1, 3, 5, 8, 12, 15, 20, 25, 30, 40, whose mean is 15.9 and whose
    # squared distances from it add up to 1464.9 over nine degrees of freedom.
    var out = run(
        "SELECT var_samp(qty) AS v, stddev(qty) AS s FROM sales", session()
    )
    assert_equal(len(out), 1, "one row")
    var v = out.column("v").as_typed[DType.float64]()[0]
    var s = out.column("s").as_typed[DType.float64]()[0]
    assert_true(abs(v - Float64(1464.9) / 9.0) < 1e-9, "the variance")
    assert_true(abs(s * s - v) < 1e-9, "the deviation squared is the variance")


def test_a_grouped_count_distinct_counts_each_group_on_its_own() raises:
    # Every qty is different, so this checks the wiring rather than the
    # counting, which the group node's own tests do over repeated values. Shop
    # 1 sold five of the ten and shop 2 sold the other five.
    var out = run(
        (
            "SELECT shop, count(DISTINCT qty) AS n FROM sales GROUP BY shop"
            " ORDER BY shop"
        ),
        session(),
    )
    assert_equal(len(out), 2, "two shops")
    same(read_back(out, "n"), [5, 5], "five each")


def test_a_grouped_median_comes_back_through_sql() raises:
    # Shop 1 sold 5, 3, 12, 25 and 30, whose middle is 12. Shop 2 sold 20, 40,
    # 8, 1 and 15, whose middle is 15.
    var out = run(
        (
            "SELECT shop, median(qty) AS mid FROM sales GROUP BY shop"
            " ORDER BY shop"
        ),
        session(),
    )
    assert_equal(len(out), 2, "two shops")
    var mid = out.column("mid").as_typed[DType.float64]()
    assert_equal(mid[0], Float64(12.0), "shop 1")
    assert_equal(mid[1], Float64(15.0), "shop 2")


def test_a_grouped_spread_comes_back_through_sql() raises:
    # Shop 1 sold 5, 3, 12, 25 and 30, whose mean is 15 and whose squared
    # distances from it add up to 578 over four degrees of freedom.
    var out = run(
        (
            "SELECT shop, var_samp(qty) AS v, stddev(qty) AS s FROM sales"
            " GROUP BY shop ORDER BY shop"
        ),
        session(),
    )
    assert_equal(len(out), 2, "two shops")
    var v = out.column("v").as_typed[DType.float64]()
    var s = out.column("s").as_typed[DType.float64]()
    assert_true(abs(v[0] - Float64(578.0) / 4.0) < 1e-9, "shop 1's variance")
    assert_true(abs(s[0] * s[0] - v[0]) < 1e-9, "and its deviation squared")


def test_distinct_inside_another_aggregate_is_refused_rather_than_ignored() raises:
    with assert_raises(contains="DISTINCT inside count and not inside sum"):
        _ = run("SELECT sum(DISTINCT qty) AS n FROM sales", session())


def test_count_distinct_star_has_no_column_to_count() raises:
    with assert_raises(contains="no column to count the distinct values of"):
        _ = run("SELECT count(DISTINCT *) AS n FROM sales", session())


def test_a_case_picks_between_two_columns() raises:
    same(
        answer(
            (
                "SELECT CASE WHEN qty > 10 THEN qty ELSE price END AS taken"
                " FROM sales"
            ),
            "taken",
        ),
        [10, 20, 7, 40, 12, 9, 25, 100, 30, 15],
        "the quantity over ten and the price otherwise",
    )


def test_a_case_over_a_null_takes_the_else_side() raises:
    # Not a null answer. A row the question could not be asked about is a row
    # the question did not hold for, which is what the standard says.
    same(
        answer(
            "SELECT CASE WHEN mark > 3 THEN 1 ELSE 0 END AS big FROM gappy",
            "big",
        ),
        [1, 1, 0, 1, 0, 0],
        "the two nulls take the else side",
    )


def test_a_case_with_several_whens_takes_the_first_that_holds() raises:
    same(
        answer(
            (
                "SELECT CASE WHEN qty > 25 THEN 3 WHEN qty > 10 THEN 2 ELSE 1"
                " END AS band FROM sales"
            ),
            "band",
        ),
        [1, 2, 1, 3, 2, 1, 2, 1, 3, 2],
        "three bands over the quantity",
    )


def test_a_case_with_no_else_answers_a_null() raises:
    same(
        gapped(
            run(
                "SELECT CASE WHEN qty > 10 THEN qty END AS big FROM sales",
                session(),
            ),
            "big",
        ),
        [-1, 20, -1, 40, 12, -1, 25, -1, 30, 15],
        "no else is an else of null",
    )


def test_a_case_inside_a_sum_is_the_shape_tpch_asks_for() raises:
    # q8, q12 and q14 are all this: a condition over one column, the value on
    # the true side and a zero on the false side, summed.
    same(
        answer(
            (
                "SELECT sum(CASE WHEN shop = 1 THEN qty ELSE 0 END) AS mine"
                " FROM sales"
            ),
            "mine",
        ),
        [75],
        "the five rows of shop one",
    )


def test_a_case_may_be_the_whole_of_a_where() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales"
                " WHERE CASE WHEN shop = 1 THEN qty > 10 ELSE false END"
            ),
            "qty",
        ),
        [12, 25, 30],
        "shop one and over ten",
    )


def test_a_simple_case_compares_the_subject_against_each_arm() raises:
    # `CASE x WHEN v` is the searched form with the comparison written out, so
    # the subject is lowered once and every arm shares it.
    same(
        answer(
            (
                "SELECT CASE shop WHEN 1 THEN 100 WHEN 2 THEN 200 END AS tag"
                " FROM sales"
            ),
            "tag",
        ),
        [100, 200, 100, 200, 100, 200, 100, 200, 100, 200],
        "one arm per shop",
    )


def test_a_simple_case_over_a_null_subject_takes_the_else() raises:
    # A comparison against a null is null, and a null condition takes the arm
    # below it, so a null subject falls all the way through to the ELSE.
    same(
        answer(
            (
                "SELECT CASE mark WHEN 4 THEN 1 WHEN 9 THEN 2 ELSE 0 END AS tag"
                " FROM gappy"
            ),
            "tag",
        ),
        [1, 1, 0, 2, 0, 0],
        "the two null marks take the else",
    )


def test_a_simple_case_whose_arm_is_null_matches_nothing() raises:
    # Including the null rows, which is the point. `x = NULL` is null and never
    # true, so WHEN NULL is an arm nothing reaches, the same as in DuckDB.
    same(
        answer(
            "SELECT CASE mark WHEN NULL THEN 1 ELSE 0 END AS tag FROM gappy",
            "tag",
        ),
        [0, 0, 0, 0, 0, 0],
        "no row matches a null arm",
    )


def test_a_join_on_two_keys_pairs_on_both() raises:
    # The stock frame holds a shop and a quantity the sales frame has in
    # different rows, and a join on the shop alone would pair it. It does not
    # appear here, and neither does the row whose quantity is null nor the shop
    # that sells nothing. Checked against DuckDB.
    var got = run(
        (
            "SELECT sales.qty AS qty, held FROM sales JOIN stock"
            " ON sales.shop = stock.shop AND sales.qty = stock.qty"
            " ORDER BY qty"
        ),
        session(),
    )
    same(read_back(got, "qty"), [5, 12, 40], "three rows agree on both keys")
    same(read_back(got, "held"), [100, 300, 200], "and carry their stock")


def test_a_join_on_two_keys_may_write_them_in_either_order() raises:
    # The order the pairs are written in is the order they are packed in, and
    # both sides pack the same way whichever order that is, so the writing
    # decides the packing and must not decide the answer.
    var got = run(
        (
            "SELECT sales.qty AS qty, held FROM sales JOIN stock"
            " ON sales.qty = stock.qty AND sales.shop = stock.shop"
            " ORDER BY qty"
        ),
        session(),
    )
    same(read_back(got, "qty"), [5, 12, 40], "the same three rows")
    same(read_back(got, "held"), [100, 300, 200], "and the same stock")


def test_a_join_on_two_keys_may_be_aggregated_over() raises:
    # The operator hands a packed key column to its own table and must not hand
    # it on, so the thing above it has to see an ordinary chunk. A reduction is
    # the cheapest way to ask that.
    same(
        answer(
            (
                "SELECT sum(held) AS total FROM sales JOIN stock"
                " ON sales.shop = stock.shop AND sales.qty = stock.qty"
            ),
            "total",
        ),
        [600],
        "the three matched rows and nothing else",
    )


def test_a_semi_join_on_two_keys_keeps_the_rows_that_matched() raises:
    # An inner join keeps both sides' columns, so the rest of the key could be
    # asked after the pairing. A semi join keeps none of them and cannot, so
    # this one is only right if the operator pairs on the whole key at once.
    var got = run(
        (
            "SELECT qty FROM sales WHERE EXISTS (SELECT 1 FROM stock"
            " WHERE stock.shop = sales.shop AND stock.qty = sales.qty)"
        ),
        session(),
    )
    same(
        read_back(got, "qty"), [5, 40, 12], "the three rows both keys agree on"
    )


def test_a_qualified_star_runs_one_side_of_a_join() raises:
    var got = run(
        "SELECT shops.* FROM sales JOIN shops ON sales.shop = shops.shop",
        session(),
    )
    assert_equal(got.width(), 2, "the right side and nothing of the left")
    same(
        read_back(got, "floor"),
        [11, 22, 11, 22, 11, 22, 11, 22, 11, 22],
        "the floor of each sale's shop",
    )


def test_an_exclude_runs_without_the_columns_it_names() raises:
    var got = run("SELECT * EXCLUDE (price, shop) FROM sales", session())
    assert_equal(got.width(), 1, "one column left of three")
    same(
        read_back(got, "qty"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15],
        "the quantities",
    )


def test_a_replace_computes_the_column_where_it_stood() raises:
    var got = run("SELECT * REPLACE (qty * 2 AS qty) FROM sales", session())
    assert_equal(got.width(), 3, "still three columns")
    same(
        read_back(got, "qty"),
        [10, 40, 6, 80, 24, 16, 50, 2, 60, 30],
        "twice each quantity, under the name it had",
    )
    same(
        read_back(got, "price"),
        [10, 2, 7, 1, 5, 9, 3, 100, 4, 6],
        "and the rest untouched",
    )


def test_a_rename_changes_the_name_and_nothing_else() raises:
    var got = run("SELECT * RENAME (qty AS many) FROM sales", session())
    assert_equal(got.width(), 3, "still three columns")
    same(
        read_back(got, "many"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15],
        "the quantities under the new name",
    )


def test_a_column_is_found_however_the_query_spells_it() raises:
    same(
        answer("SELECT AdvEngineID FROM hits", "AdvEngineID"),
        [0, 2, 2, 3],
        "written the way the schema writes it",
    )
    same(
        answer("SELECT advengineid FROM hits", "AdvEngineID"),
        [0, 2, 2, 3],
        "written flat",
    )
    same(
        answer("SELECT ADVENGINEID FROM hits", "AdvEngineID"),
        [0, 2, 2, 3],
        "written shouting",
    )


def test_a_quoted_name_folds_too_because_duckdb_folds_it() raises:
    same(
        answer('SELECT "advengineid" FROM hits', "AdvEngineID"),
        [0, 2, 2, 3],
        "quoting changes what a name is and not how it compares",
    )


def test_the_answer_keeps_the_schema_spelling() raises:
    var out = run("SELECT advengineid FROM hits", session())
    assert_equal(
        String(out.schema[0].name),
        "AdvEngineID",
        "the column comes back called what it is called",
    )


def test_a_folded_name_works_everywhere_a_name_works() raises:
    var out = run(
        (
            "SELECT regionid, SUM(advengineid) AS s FROM hits"
            " WHERE advengineid <> 0 GROUP BY regionid ORDER BY s DESC"
        ),
        session(),
    )
    same(read_back(out, "s"), [5, 2], "the filter, the grouping and the sum")
    same(
        read_back(out, "RegionID"),
        [9, 7],
        "the key, spelled as the schema does",
    )


def test_a_qualified_name_folds_as_well() raises:
    same(
        answer("SELECT h.advengineid FROM hits AS h", "AdvEngineID"),
        [0, 2, 2, 3],
        "written in front of an alias",
    )


def test_a_name_nothing_has_still_says_so() raises:
    with assert_raises(contains="there is no column named"):
        _ = run("SELECT advengineidx FROM hits", session())


def test_a_date_column_takes_a_string_literal_as_a_bound() raises:
    # #680, and seven of the 43 ClickBench statements are this shape. The
    # literal is text and the column holds days, and in SQL that pair is a date
    # bound rather than a type error.
    same(
        answer(
            (
                "SELECT advengineid FROM hits WHERE eventdate >= '2013-07-01'"
                " AND eventdate <= '2013-07-31'"
            ),
            "AdvEngineID",
        ),
        [2, 2],
        "the two rows inside July",
    )


def test_a_date_bound_reads_the_same_written_either_way_round() raises:
    same(
        answer(
            (
                "SELECT advengineid FROM hits WHERE '2013-07-31' >= eventdate"
                " AND '2013-07-01' <= eventdate"
            ),
            "AdvEngineID",
        ),
        [2, 2],
        "the literal on the left says the same thing",
    )


def test_a_date_column_equals_a_string_literal() raises:
    same(
        answer(
            "SELECT advengineid FROM hits WHERE eventdate = '2013-07-15'",
            "AdvEngineID",
        ),
        [2],
        "one day",
    )


def test_a_between_on_a_date_column_reads_both_of_its_bounds() raises:
    same(
        answer(
            (
                "SELECT advengineid FROM hits WHERE eventdate BETWEEN"
                " '2013-07-01' AND '2013-07-31'"
            ),
            "AdvEngineID",
        ),
        [2, 2],
        "a BETWEEN is the two comparisons",
    )


def test_an_in_list_of_date_literals_reads_every_one_of_them() raises:
    same(
        answer(
            (
                "SELECT advengineid FROM hits WHERE eventdate IN ('2013-06-30',"
                " '2013-08-01')"
            ),
            "AdvEngineID",
        ),
        [0, 3],
        "a list is a comparison each",
    )


def test_text_that_is_not_a_date_is_refused_and_quoted() raises:
    with assert_raises(contains="'the first of July'"):
        _ = run(
            (
                "SELECT advengineid FROM hits WHERE eventdate >= 'the first of"
                " July'"
            ),
            session(),
        )


def test_a_clock_reading_against_a_date_column_is_refused() raises:
    # The literal is a real instant and the column has nowhere to put the time
    # of day, so this is refused rather than truncated. A bound of `>=
    # '2013-07-01 12:00:00'` read as midnight would quietly keep more rows than
    # it was asked for.
    with assert_raises(contains="carries a time of day"):
        _ = run(
            (
                "SELECT advengineid FROM hits WHERE eventdate >= '2013-07-01"
                " 12:00:00'"
            ),
            session(),
        )


def test_a_date_literal_written_with_its_type_is_the_same_bound() raises:
    # The spelling with the type in front of it, which is the one TPC-H writes
    # and the one that says what it means without the column next to it.
    same(
        answer(
            (
                "SELECT advengineid FROM hits WHERE eventdate >= DATE"
                " '2013-07-01' AND eventdate <= DATE '2013-07-31'"
            ),
            "AdvEngineID",
        ),
        [2, 2],
        "the two rows inside July",
    )


def test_a_date_literal_and_a_bare_string_are_the_same_bound() raises:
    # Both spellings end up as one typed constant, so the plan cannot tell them
    # apart by the time it runs and neither can the answer.
    same(
        answer(
            "SELECT advengineid FROM hits WHERE eventdate = DATE '2013-07-15'",
            "AdvEngineID",
        ),
        answer(
            "SELECT advengineid FROM hits WHERE eventdate = '2013-07-15'",
            "AdvEngineID",
        ),
        "one day, written twice",
    )


def test_a_date_literal_reads_before_the_column_is_looked_at() raises:
    # The string is read where the query is planned rather than once per row,
    # which is the whole reason the literal becomes a constant. Nothing here
    # sees that directly, so what is checked is the consequence: a literal that
    # is not a date is refused by a query that never reaches a row.
    with assert_raises(contains="'the first of July'"):
        _ = run(
            (
                "SELECT count(*) AS c FROM hits WHERE eventdate >= DATE 'the"
                " first of July'"
            ),
            session(),
        )


def test_a_date_literal_carrying_a_clock_reading_is_refused() raises:
    # Same refusal the bare string gets, and for the same reason: a date holds
    # whole days and the time of day would have nowhere to go.
    with assert_raises(contains="carries a time of day"):
        _ = run(
            (
                "SELECT advengineid FROM hits WHERE eventdate >= DATE"
                " '2013-07-01 12:00:00'"
            ),
            session(),
        )


def test_a_timestamp_literal_against_a_date_column_is_refused() raises:
    # A timestamp literal is a count of microseconds and the column is a count
    # of days, and the promotion has no common type for the two. DuckDB widens
    # the date to a timestamp here and firepanda does not, which is a
    # divergence and a refusal rather than a wrong answer.
    with assert_raises(contains="differ in kind"):
        _ = run(
            (
                "SELECT advengineid FROM hits WHERE eventdate = TIMESTAMP"
                " '2013-07-15 00:00:00'"
            ),
            session(),
        )


def test_a_number_literal_with_an_exponent_is_a_double() raises:
    # DuckDB reads the type off how the number was written and not off what it
    # is worth, so `1e3` is a DOUBLE even though one thousand is exact. The
    # plan holds a double, so there is nothing lost and nothing to refuse.
    var out = run("SELECT 1e3 AS a, 1.5e3 AS b, 1.1e-2 AS c", session())

    assert_true(out.schema[0].dtype == LogicalType.FLOAT64, "a double")
    assert_equal(out.column("a").as_typed[DType.float64]()[0], 1000.0, "1e3")
    assert_equal(out.column("b").as_typed[DType.float64]()[0], 1500.0, "1.5e3")
    assert_equal(out.column("c").as_typed[DType.float64]()[0], 0.011, "1.1e-2")


def test_a_decimal_literal_past_the_widest_decimal_is_a_double() raises:
    # The count is of digits as written, so the trailing zeros are what pushes
    # this one over 38 and makes DuckDB read a DOUBLE. Written as `1.5` it is a
    # DECIMAL(2,1) and is refused by the test below.
    var out = run(
        "SELECT 1.5000000000000000000000000000000000000000 AS a", session()
    )

    assert_true(out.schema[0].dtype == LogicalType.FLOAT64, "a double")
    assert_equal(out.column("a").as_typed[DType.float64]()[0], 1.5, "1.5")


def test_a_decimal_literal_on_its_own_is_refused() raises:
    # The refusal that stands, and it is about the answer's own type rather
    # than about the literal. Nothing above either of these turns the decimal
    # into anything else, so the column handed back would have to be a decimal
    # and there is no decimal column to hand back.
    with assert_raises(contains="does not lower the decimal literal"):
        _ = run("SELECT 1.1 AS a", session())
    with assert_raises(contains="does not lower the decimal literal"):
        _ = run(
            "SELECT 1234567890123456789012345678901234567.8 AS a", session()
        )
    with assert_raises(contains="expression of decimal literals"):
        _ = run("SELECT 1.1 + 2.2 AS a", session())
    with assert_raises(contains="expression of decimal literals"):
        _ = run("SELECT -1.5 * 4 AS a", session())


def test_a_decimal_literal_against_a_column_is_read_as_a_double() raises:
    # Against a column there is somewhere for it to go, because the column is
    # not a decimal either and DuckDB casts the literal to a double at exactly
    # this point. So this answers rather than refusing, and answers what DuckDB
    # answers.
    var out = run("SELECT qty * 0.5 AS half FROM sales", session())

    assert_true(out.schema[0].dtype == LogicalType.FLOAT64, "a double")
    var col = out.column("half").as_typed[DType.float64]()
    assert_equal(col[0], 2.5, "five halves")
    assert_equal(col[7], 0.5, "one half")


def test_the_decimal_arithmetic_happens_before_the_double_does() raises:
    # The whole reason the fold is exact. Both sides here are folded in the
    # scaled integers a decimal really is and converted once, so both land on
    # the double nearest five hundredths and the two agree. Lowering each
    # literal to a double first and subtracting would put the left side at
    # 0.049999999999999996, which is a different number, and TPC-H q6 writes
    # its lower bound exactly this way.
    var out = run("SELECT 0.06 - 0.01 = 0.05 AS same", session())

    assert_equal(truths(out, "same"), [Int64(1)], "five hundredths twice")

    # The one everybody knows. DuckDB answers true because it adds decimals,
    # and so does this for the same reason.
    var known = run("SELECT 0.1 + 0.2 = 0.3 AS same", session())

    assert_equal(truths(known, "same"), [Int64(1)], "three tenths")


def test_an_expression_of_whole_numbers_is_not_touched_by_any_of_this() raises:
    # No point written anywhere, so there is no decimal and nothing folds. The
    # answer stays an integer, which it would not if the fold read every
    # literal expression rather than the ones with a scale.
    var out = run("SELECT qty + 2 * 3 AS grown FROM sales", session())

    assert_true(out.schema[0].dtype == LogicalType.INT64, "still an integer")
    assert_equal(read_back(out, "grown")[0], 11, "five and six")


def test_an_integer_literal_past_a_bigint_is_refused_rather_than_wrapped() raises:
    # DuckDB reads a HUGEINT here and the plan has no 128 bit integer. It used
    # to wrap and answer -9223372036854775808, which is the one kind of failure
    # this front end is not allowed to have.
    assert_equal(
        run("SELECT 9223372036854775807 AS a", session())
        .column("a")
        .as_typed[DType.int64]()[0],
        9223372036854775807,
        "the largest one that fits",
    )
    with assert_raises(contains="does not lower the integer literal"):
        _ = run("SELECT 9223372036854775808 AS a", session())


def test_an_extract_reads_the_field_off_every_row() raises:
    same(
        answer("SELECT EXTRACT(YEAR FROM eventdate) AS y FROM hits", "y"),
        [2013, 2013, 2013, 2013],
        "the year of each of the four days",
    )
    same(
        answer("SELECT EXTRACT(MONTH FROM eventdate) AS m FROM hits", "m"),
        [6, 7, 7, 8],
        "and the month",
    )
    same(
        answer("SELECT EXTRACT(DAY FROM eventdate) AS d FROM hits", "d"),
        [30, 1, 15, 1],
        "and the day of the month",
    )


def test_the_three_spellings_answer_the_same_column() raises:
    var want: List[Int64] = [2013, 2013, 2013, 2013]
    same(
        answer("SELECT EXTRACT(YEAR FROM eventdate) AS y FROM hits", "y"),
        want,
        "the keyword spelling",
    )
    same(
        answer("SELECT date_part('year', eventdate) AS y FROM hits", "y"),
        want,
        "the function DuckDB names it",
    )
    same(
        answer("SELECT datepart('year', eventdate) AS y FROM hits", "y"),
        want,
        "and its other spelling",
    )


def test_the_fields_that_count_across_a_year() raises:
    same(
        answer("SELECT EXTRACT(QUARTER FROM eventdate) AS q FROM hits", "q"),
        [2, 3, 3, 3],
        "the quarter the day falls in",
    )
    same(
        answer("SELECT EXTRACT(DOY FROM eventdate) AS n FROM hits", "n"),
        [181, 182, 196, 213],
        "the day of the year",
    )
    same(
        answer("SELECT EXTRACT(WEEK FROM eventdate) AS w FROM hits", "w"),
        [26, 27, 29, 31],
        "and the week, which DuckDB counts the ISO way",
    )


def test_a_day_of_week_is_numbered_the_way_duckdb_numbers_it() raises:
    # Sunday is zero here and Monday is zero everywhere else in firepanda, so
    # this is the one field that is rewritten rather than read straight off. The
    # first of the four days is a Sunday, which is what makes the difference
    # visible at all.
    same(
        answer("SELECT EXTRACT(DOW FROM eventdate) AS n FROM hits", "n"),
        [0, 1, 1, 4],
        "Sunday is a zero",
    )
    same(
        answer("SELECT EXTRACT(ISODOW FROM eventdate) AS n FROM hits", "n"),
        [7, 1, 1, 4],
        "and the ISO numbering makes it a seven",
    )
    same(
        answer("SELECT EXTRACT(WEEKDAY FROM eventdate) AS n FROM hits", "n"),
        [0, 1, 1, 4],
        "and weekday is a second spelling of the first question",
    )


def test_the_periods_longer_than_a_year() raises:
    # All four days are in 2013, so the three answers are the same on every row
    # and what the test is about is which number each period gets. A decade
    # counts from zero and the other two count from one, which is why 2013 is
    # in decade 201 and in century 21 rather than in century 20.
    same(
        answer("SELECT EXTRACT(DECADE FROM eventdate) AS n FROM hits", "n"),
        [201, 201, 201, 201],
        "the decade",
    )
    same(
        answer("SELECT EXTRACT(CENTURY FROM eventdate) AS n FROM hits", "n"),
        [21, 21, 21, 21],
        "the century",
    )
    same(
        answer("SELECT EXTRACT(MILLENNIUM FROM eventdate) AS n FROM hits", "n"),
        [3, 3, 3, 3],
        "the millennium",
    )


def test_the_century_and_the_millennium_count_from_one() raises:
    # The year a period ends on is the one worth checking, because it is the
    # one an arithmetic that counted from zero would get wrong. 1900 is the
    # last year of the nineteenth century and 2000 is the last of the twentieth.
    same(
        answer(
            (
                "SELECT EXTRACT(CENTURY FROM col0) AS n FROM (VALUES (DATE"
                " '1900-12-31'), (DATE '1901-01-01'), (DATE '2000-12-31'),"
                " (DATE '2001-01-01')) AS t"
            ),
            "n",
        ),
        [19, 20, 20, 21],
        "the century each year is in",
    )
    same(
        answer(
            (
                "SELECT EXTRACT(MILLENNIUM FROM col0) AS n FROM (VALUES (DATE"
                " '2000-12-31'), (DATE '2001-01-01')) AS t"
            ),
            "n",
        ),
        [2, 3],
        "and the millennium",
    )


def test_a_year_week_is_the_iso_year_and_the_iso_week_together() raises:
    # The ISO year rather than the year, which is the whole point of the field.
    # The last days of a December can fall in the first ISO week of the year
    # after, and a number built from the year would put them in week one of the
    # year that is ending.
    same(
        answer("SELECT EXTRACT(YEARWEEK FROM eventdate) AS n FROM hits", "n"),
        [201326, 201327, 201329, 201331],
        "the four days",
    )
    same(
        answer(
            (
                "SELECT EXTRACT(YEARWEEK FROM col0) AS n FROM (VALUES (DATE"
                " '2024-12-30'), (DATE '2021-01-03')) AS t"
            ),
            "n",
        ),
        [202501, 202053],
        "a day either side of a year boundary",
    )


def test_a_day_of_month_is_the_day() raises:
    same(
        answer("SELECT EXTRACT(DAYOFMONTH FROM eventdate) AS n FROM hits", "n"),
        [30, 1, 15, 1],
        "the same number DAY answers",
    )


def test_an_era_is_one_for_every_date_and_null_for_no_date() raises:
    # Every date firepanda can hold is after the year zero, so the number is a
    # one wherever there is a date at all. The second half is why it is not
    # written as the literal one: a row with no date has no era either.
    same(
        answer("SELECT EXTRACT(ERA FROM eventdate) AS n FROM hits", "n"),
        [1, 1, 1, 1],
        "one for each of the four days",
    )
    same(
        gapped(
            run(
                (
                    "SELECT EXTRACT(ERA FROM CASE WHEN advengineid = 0 THEN"
                    " NULL ELSE eventdate END) AS n FROM hits"
                ),
                session(),
            ),
            "n",
        ),
        [-1, 1, 1, 1],
        "and null where the date was null",
    )


def test_a_field_duckdb_has_and_firepanda_does_not_says_what_is_missing() raises:
    # Four specifiers DuckDB answers and this does not, and the point of each
    # message is that the gap is under the name rather than in it.
    with assert_raises(contains="no cast that does that yet"):
        _ = run("SELECT EXTRACT(EPOCH FROM eventdate) FROM hits", session())
    with assert_raises(contains="the whole of the seconds and the fraction"):
        _ = run(
            "SELECT EXTRACT(MICROSECOND FROM eventdate) FROM hits", session()
        )
    with assert_raises(contains="the whole of the seconds and the fraction"):
        _ = run(
            "SELECT EXTRACT(MILLISECOND FROM eventdate) FROM hits", session()
        )
    with assert_raises(contains="no time zone aware timestamp"):
        _ = run("SELECT EXTRACT(TIMEZONE FROM eventdate) FROM hits", session())


def test_a_field_read_in_a_where_keeps_the_rows_it_names() raises:
    same(
        answer(
            (
                "SELECT advengineid FROM hits WHERE EXTRACT(MONTH FROM"
                " eventdate) = 7"
            ),
            "AdvEngineID",
        ),
        [2, 2],
        "the two days inside July",
    )


def test_a_field_read_in_a_group_by_folds_on_what_it_answers() raises:
    # The field is read in the derived table and grouped on by name, which is
    # one of the three ways of writing this query and was for a while the only
    # one that lowered.
    var out = run(
        (
            "SELECT m, count(*) AS n FROM (SELECT EXTRACT(MONTH FROM eventdate)"
            " AS m FROM hits) GROUP BY m ORDER BY 1"
        ),
        session(),
    )
    same(read_back(out, "m"), [6, 7, 8], "one row per month")
    same(read_back(out, "n"), [1, 2, 1], "and the count in each")


def test_the_field_read_written_out_in_both_clauses_folds_the_same() raises:
    # The second of the three ways, and the one ClickBench q18 is written as.
    # The expression is computed once, as the key, and the select list reads the
    # column the aggregate put it in.
    var out = run(
        (
            "SELECT EXTRACT(MONTH FROM eventdate) AS m, count(*) AS n FROM hits"
            " GROUP BY EXTRACT(MONTH FROM eventdate) ORDER BY 1"
        ),
        session(),
    )
    same(read_back(out, "m"), [6, 7, 8], "one row per month")
    same(read_back(out, "n"), [1, 2, 1], "and the count in each")


def test_a_having_may_write_out_what_the_group_by_wrote() raises:
    # One clause further up and the same rule: the filter over the aggregate
    # reads the key's column rather than a date the aggregate no longer has.
    var out = run(
        (
            "SELECT EXTRACT(MONTH FROM eventdate) AS m, count(*) AS n FROM hits"
            " GROUP BY EXTRACT(MONTH FROM eventdate) HAVING EXTRACT(MONTH FROM"
            " eventdate) = 7"
        ),
        session(),
    )
    same(read_back(out, "m"), [7], "the one month the HAVING kept")
    same(read_back(out, "n"), [2], "and its count")


def test_an_extract_answers_a_whole_number_the_width_duckdb_answers() raises:
    # The kernel under this answers the narrower types pandas answers with and
    # DuckDB answers a BIGINT, so the widening happens in the operator. A query
    # that groups on a year and joins that against a count needs the two to be
    # the same width.
    var out = run(
        "SELECT EXTRACT(YEAR FROM eventdate) AS y FROM hits", session()
    )
    assert_true(
        out.schema[0].dtype == LogicalType.INT64, "the width DuckDB answers"
    )


def test_a_field_nobody_has_a_kernel_for_says_so_by_name() raises:
    with assert_raises(contains="no field SQL calls fortnight"):
        _ = run("SELECT EXTRACT(FORTNIGHT FROM eventdate) FROM hits", session())


def test_an_extract_off_a_column_that_is_not_a_date_is_refused() raises:
    with assert_raises(contains="reads a date or a timestamp"):
        _ = run("SELECT EXTRACT(YEAR FROM qty) FROM sales", session())


def test_a_truncation_moves_every_row_back_to_a_period_start() raises:
    # The four days are the 30th of June, the 1st and the 15th of July and the
    # 1st of August 2013, and the numbers are microseconds since the epoch,
    # which is what DuckDB answers a `DATE_TRUNC` with.
    same(
        answer("SELECT date_trunc('month', eventdate) AS m FROM hits", "m"),
        [
            1370044800000000,
            1372636800000000,
            1372636800000000,
            1375315200000000,
        ],
        "the first of each day's month",
    )
    same(
        answer("SELECT date_trunc('year', eventdate) AS y FROM hits", "y"),
        [
            1356998400000000,
            1356998400000000,
            1356998400000000,
            1356998400000000,
        ],
        "and one year for all four",
    )


def test_the_two_spellings_of_a_truncation_answer_the_same_column() raises:
    var want: List[Int64] = [
        1364774400000000,
        1372636800000000,
        1372636800000000,
        1372636800000000,
    ]
    same(
        answer("SELECT date_trunc('quarter', eventdate) AS q FROM hits", "q"),
        want,
        "the name DuckDB gives it",
    )
    same(
        answer("SELECT datetrunc('quarter', eventdate) AS q FROM hits", "q"),
        want,
        "and its other spelling",
    )


def test_a_truncation_to_a_week_lands_on_the_monday_before() raises:
    # The 30th of June 2013 was a Sunday, so it goes back six days, and the
    # 1st of July was the Monday after it and stays where it is.
    same(
        answer("SELECT date_trunc('week', eventdate) AS w FROM hits", "w"),
        [
            1372032000000000,
            1372636800000000,
            1373846400000000,
            1375056000000000,
        ],
        "the Monday of each day's week",
    )


def test_a_truncation_answers_a_timestamp_even_off_a_date() raises:
    var out = run(
        "SELECT date_trunc('year', eventdate) AS y FROM hits", session()
    )
    assert_true(
        out.schema[0].dtype == LogicalType.timestamp(TimeUnit.MICRO),
        "the type DuckDB answers",
    )


def test_a_truncation_read_in_a_group_by_folds_on_what_it_answers() raises:
    # Grouped through a derived table, which is the way the same test on a field
    # read writes it.
    var out = run(
        (
            "SELECT m, count(*) AS n FROM (SELECT date_trunc('month',"
            " eventdate) AS m FROM hits) GROUP BY m ORDER BY 1"
        ),
        session(),
    )
    same(
        read_back(out, "m"),
        [1370044800000000, 1372636800000000, 1375315200000000],
        "one row per month",
    )
    same(read_back(out, "n"), [1, 2, 1], "and the count in each")


def test_the_truncation_written_out_in_all_three_clauses_folds_once() raises:
    # ClickBench q42's shape, which writes the same truncation in the select
    # list, the GROUP BY and the ORDER BY. One column is computed and read three
    # times, and the answer is the derived table's above.
    var out = run(
        (
            "SELECT date_trunc('month', eventdate) AS m, count(*) AS n FROM"
            " hits GROUP BY date_trunc('month', eventdate) ORDER BY"
            " date_trunc('month', eventdate)"
        ),
        session(),
    )
    same(
        read_back(out, "m"),
        [1370044800000000, 1372636800000000, 1375315200000000],
        "one row per month",
    )
    same(read_back(out, "n"), [1, 2, 1], "and the count in each")


def test_a_period_nobody_has_a_unit_for_says_so_by_name() raises:
    with assert_raises(contains="nothing to truncate to called fortnight"):
        _ = run(
            "SELECT date_trunc('fortnight', eventdate) FROM hits", session()
        )


def test_a_truncation_off_a_column_that_is_not_a_date_is_refused() raises:
    with assert_raises(contains="truncates a date or a timestamp"):
        _ = run("SELECT date_trunc('month', qty) FROM sales", session())


def test_an_answer_of_no_rows_can_still_be_read() raises:
    var out = run("SELECT qty, price FROM sales WHERE qty = 999", session())
    assert_equal(len(out), 0, "no rows")
    assert_equal(out.width(), 2, "and both columns")
    assert_equal(len(out[0]), 0, "the first reads as empty")
    assert_equal(len(out[1]), 0, "and so does the second")
    same(read_back(out, "qty"), List[Int64](), "nothing under the name either")


def test_an_answer_of_no_text_rows_can_still_be_read() raises:
    var out = run("SELECT 'sold' AS tag FROM sales WHERE qty = 999", session())
    assert_equal(len(out), 0, "no rows")
    assert_true(out[0].is_string(), "and the column is still a text one")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
