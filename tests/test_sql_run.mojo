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
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame
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


def tiers() raises -> DataFrame:
    """Four bands and the rate each one charges.

    The names are disjoint from the sales frame's, so a join over the two can
    write either side's columns without qualifying them and a test that wants to
    qualify one still can.
    """
    var band = ChunkedArray(LogicalType.INT64)
    band.append(numbers([3, 20, 40, 99]))
    var rate = ChunkedArray(LogicalType.INT64)
    rate.append(numbers([300, 200, 400, 900]))
    var columns = List[ChunkedArray]()
    columns.append(band^)
    columns.append(rate^)
    var fields = List[Field]()
    fields.append(Field("band", LogicalType.INT64))
    fields.append(Field("rate", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def shops() raises -> DataFrame:
    """Three shops and the floor each one is on.

    This one shares a column name with the sales frame on purpose, which the
    other two do not. A `USING` join names its keys by a shared name and a
    `NATURAL` join finds them that way, so neither has anything to say about
    two frames with nothing in common. The third shop sells nothing, which is
    the row an outer join has to keep.
    """
    var shop = ChunkedArray(LogicalType.INT64)
    shop.append(numbers([1, 2, 3]))
    var floor = ChunkedArray(LogicalType.INT64)
    floor.append(numbers([11, 22, 33]))
    var columns = List[ChunkedArray]()
    columns.append(shop^)
    columns.append(floor^)
    var fields = List[Field]()
    fields.append(Field("shop", LogicalType.INT64))
    fields.append(Field("floor", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def dupes() raises -> DataFrame:
    """One column of bands with a repeat in it.

    A semi join answers a left row once however many right rows matched it, and
    a right side where every key is unique cannot tell that from a join that
    kept them all.
    """
    var band = ChunkedArray(LogicalType.INT64)
    band.append(numbers([3, 3, 20, 77]))
    var columns = List[ChunkedArray]()
    columns.append(band^)
    var fields = List[Field]()
    fields.append(Field("band", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def session() raises -> Catalog:
    """A catalog holding the four frames under the names the queries write."""
    var catalog = Catalog()
    catalog.register("sales", sales())
    catalog.register("tiers", tiers())
    catalog.register("shops", shops())
    catalog.register("dupes", dupes())
    return catalog^


def read_back(df: DataFrame, name: String) raises -> List[Int64]:
    """Reads an int64 column out as a plain list."""
    var col = df.column(name).as_typed[DType.int64]()
    var out = List[Int64](capacity=len(col))
    for i in range(len(col)):
        out.append(col[i])
    return out^


def truths(df: DataFrame, name: String) raises -> List[Int64]:
    """Reads a bool column out as ones and zeroes, and a null as a minus one."""
    var col = df.column(name).as_typed[DType.bool]()
    var out = List[Int64](capacity=len(col))
    for i in range(len(col)):
        if not col.is_valid(i):
            out.append(-1)
        else:
            out.append(Int64(1) if col[i] else Int64(0))
    return out^


def answer(sql: StringSlice, name: String) raises -> List[Int64]:
    """Runs a query against the session and reads one column of the answer."""
    return read_back(run(sql, session()), name)


def same(got: List[Int64], want: List[Int64], what: String) raises:
    """Checks a column read back against the numbers it should hold."""
    assert_equal(len(got), len(want), what + ": how many rows")
    for i in range(len(want)):
        assert_equal(got[i], want[i], what + " at " + String(i))


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


def test_an_expression_in_the_select_list_is_computed() raises:
    same(
        answer("SELECT qty * price AS total FROM sales", "total"),
        [50, 40, 21, 40, 60, 72, 75, 100, 120, 90],
        "total",
    )


def test_an_alias_is_the_name_the_answer_comes_back_under() raises:
    var out = run("SELECT qty AS howmany FROM sales", session())
    assert_equal(out.schema[0].name, "howmany")


def test_an_aggregate_over_the_whole_table_is_one_row() raises:
    same(answer("SELECT SUM(qty) AS total FROM sales", "total"), [159], "total")


def test_a_group_by_folds_the_rows_into_groups() raises:
    # Ordered by the key, because which group comes out first is the hash
    # table's business and not the query's.
    var out = run(
        "SELECT shop, SUM(qty) AS total FROM sales GROUP BY shop ORDER BY shop",
        session(),
    )
    same(read_back(out, "shop"), [1, 2], "shop")
    same(read_back(out, "total"), [75, 84], "total")


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


def test_a_distinct_on_part_of_the_row_is_refused_by_name() raises:
    # `DISTINCT ON` keeps whole rows chosen by some of their columns, and a
    # group by carries its keys in front of what it reduced, so the answer would
    # not be the columns in the order the query asked for them. The front end
    # already has it in the unsupported table, so the refusal comes from there
    # rather than from lowering, which is the earlier and better of the two.
    with assert_raises(contains="does not lower DISTINCT ON yet"):
        _ = run("SELECT DISTINCT ON (shop) shop, qty FROM sales", session())


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
    # projection above it needs a column for the constant to land in.
    # DuckDB calls this column `1`, after the text it was written as. firepanda
    # names an expression the query did not name after its position instead,
    # which is the gap `_name_of` describes and is not this change.
    same(answer("SELECT 1", "__expr_0"), [1], "the constant")


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


def test_an_except_is_refused_by_name() raises:
    with assert_raises(contains="a difference is not a stack of its inputs"):
        _ = run(
            "SELECT qty FROM sales EXCEPT SELECT band FROM tiers", session()
        )


def test_an_intersect_is_refused_by_name() raises:
    with assert_raises(contains="an intersection is not a stack of its inputs"):
        _ = run(
            "SELECT qty FROM sales INTERSECT SELECT band FROM tiers", session()
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


def test_an_in_of_several_candidates_runs_as_the_chain_it_is() raises:
    same(
        answer("SELECT qty FROM sales WHERE qty IN (3, 25)", "qty"),
        [3, 25],
        "qty",
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


def test_a_chain_of_ors_folds_left_to_right_and_keeps_every_arm() raises:
    same(
        answer(
            "SELECT qty FROM sales WHERE qty = 3 OR qty = 25 OR qty > 29",
            "qty",
        ),
        [3, 40, 25, 30],
        "qty",
    )


def test_an_and_nested_under_an_or_is_not_split_into_filters() raises:
    # The conjunction here cannot become a line of filters, because it only has
    # to hold on the rows the disjunction did not already keep, so this is the
    # one shape where an AND reaches the connective operator.
    same(
        answer(
            "SELECT qty FROM sales WHERE qty = 1 OR (qty > 10 AND shop = 1)",
            "qty",
        ),
        [12, 25, 1, 30],
        "qty",
    )


def test_a_boolean_expression_in_a_select_list_is_a_column() raises:
    same(
        truths(
            run("SELECT qty > 10 AND shop = 1 AS big FROM sales", session()),
            "big",
        ),
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        "big",
    )


def test_a_not_in_a_select_list_turns_the_column_over() raises:
    same(
        truths(
            run("SELECT NOT (qty > 10) AS small FROM sales", session()),
            "small",
        ),
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        "small",
    )


def test_a_cast_of_a_column_leaves_the_column_it_read_alone() raises:
    # The converted column is a column of its own, so qty is still the int64
    # every other expression in the query was bound against.
    var out = run(
        "SELECT qty, CAST(qty AS DOUBLE) AS wide FROM sales", session()
    )

    assert_equal(len(out.schema), 2, "two columns")
    assert_true(out.schema[0].dtype == LogicalType.INT64, "qty as it was")
    assert_true(out.schema[1].dtype == LogicalType.FLOAT64, "and the cast")
    same(read_back(out, "qty"), [5, 20, 3, 40, 12, 8, 25, 1, 30, 15], "qty")

    var wide = out.column("wide").as_typed[DType.float64]()
    assert_equal(wide[0], 5.0, "the first row converted")
    assert_equal(wide[6], 25.0, "and one from the middle chunk")


def test_a_cast_narrows_a_column_to_the_type_the_query_named() raises:
    var out = run("SELECT CAST(qty AS SMALLINT) AS small FROM sales", session())

    assert_true(out.schema[0].dtype == LogicalType.INT16, "int16")
    var col = out.column("small").as_typed[DType.int16]()
    assert_equal(len(col), 10, "every row")
    assert_equal(col[3], 40, "the value came across")


def test_a_cast_to_varchar_writes_the_numbers_out() raises:
    var out = run(
        "SELECT CAST(qty AS VARCHAR) AS written FROM sales", session()
    )

    assert_true(out.schema[0].dtype == LogicalType.STRING, "text")
    var col = out.column("written").as_strings()
    assert_equal(col[0], "5", "the first")
    assert_equal(col[3], "40", "and a two digit one")


def test_a_cast_of_an_expression_converts_what_the_expression_made() raises:
    var out = run(
        "SELECT CAST(qty * price AS DOUBLE) AS total FROM sales", session()
    )

    assert_true(out.schema[0].dtype == LogicalType.FLOAT64, "float64")
    var col = out.column("total").as_typed[DType.float64]()
    assert_equal(col[0], 50.0, "the first product")
    assert_equal(col[9], 90.0, "and the last")


def test_a_cast_in_a_where_runs_before_the_rows_are_kept() raises:
    same(
        answer(
            "SELECT qty FROM sales WHERE CAST(qty AS DOUBLE) > 20",
            "qty",
        ),
        [40, 25, 30],
        "qty",
    )


def test_a_cast_to_a_type_the_engine_has_no_column_for_says_so() raises:
    with assert_raises(contains="integers stop at 64 bits"):
        _ = run("SELECT CAST(qty AS HUGEINT) FROM sales", session())
    with assert_raises(contains="no exact decimal"):
        _ = run("SELECT CAST(qty AS DECIMAL(9,2)) FROM sales", session())
    with assert_raises(contains="TRY_CAST"):
        _ = run("SELECT TRY_CAST(qty AS BIGINT) FROM sales", session())


def test_a_window_over_the_whole_table_is_on_every_row() raises:
    # The same 159 the aggregate test asks for, except that here it arrives
    # beside the ten rows rather than instead of them.
    same(
        answer("SELECT qty, SUM(qty) OVER () AS total FROM sales", "total"),
        [159, 159, 159, 159, 159, 159, 159, 159, 159, 159],
        "total",
    )


def test_a_window_partitions_and_each_row_reads_its_own() raises:
    # The shops alternate, so the two totals alternate with them, and the rows
    # stay in the order they were read in rather than being gathered by shop.
    var out = run(
        "SELECT shop, SUM(qty) OVER (PARTITION BY shop) AS total FROM sales",
        session(),
    )
    same(read_back(out, "shop"), [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], "shop")
    same(
        read_back(out, "total"),
        [75, 84, 75, 84, 75, 84, 75, 84, 75, 84],
        "total",
    )


def test_a_window_counts_the_rows_it_partitions_over() raises:
    same(
        answer("SELECT COUNT(*) OVER (PARTITION BY shop) AS n FROM sales", "n"),
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        "n",
    )


def test_a_qualify_keeps_the_rows_the_window_says_to() raises:
    # Shop 2 totals 84 and shop 1 totals 75, so the bound keeps one shop, and
    # it keeps every row of it rather than one row standing for the group.
    same(
        answer(
            (
                "SELECT qty FROM sales QUALIFY SUM(qty) OVER (PARTITION BY"
                " shop) > 80"
            ),
            "qty",
        ),
        [20, 40, 8, 1, 15],
        "qty",
    )


def test_a_where_under_a_window_changes_what_the_window_reduces() raises:
    # The filter runs first, so the total is over what survived it and not over
    # the table, which is the difference between a WHERE and a QUALIFY.
    same(
        answer(
            "SELECT qty, SUM(qty) OVER () AS total FROM sales WHERE qty > 20",
            "total",
        ),
        [95, 95, 95],
        "total",
    )


def test_a_running_window_is_refused_by_name() raises:
    with assert_raises(contains="OVER an ORDER BY"):
        _ = run("SELECT SUM(qty) OVER (ORDER BY qty) FROM sales", session())


def test_a_query_may_read_a_subquery_where_a_table_goes() raises:
    same(
        answer(
            "SELECT total FROM (SELECT qty * price AS total FROM sales) v",
            "total",
        ),
        [50, 40, 21, 40, 60, 72, 75, 100, 120, 90],
        "total",
    )


def test_the_outer_query_filters_what_the_subquery_handed_out() raises:
    # The filter is written over the alias the subquery invented, which is a
    # name the query inside it produced and the table underneath does not have.
    same(
        answer(
            (
                "SELECT total FROM (SELECT qty * price AS total FROM sales) v"
                " WHERE total > 80"
            ),
            "total",
        ),
        [100, 120, 90],
        "total",
    )


def test_a_column_of_a_derived_table_may_be_written_with_its_name() raises:
    same(
        answer(
            (
                "SELECT v.total FROM (SELECT qty * price AS total FROM sales) v"
                " WHERE v.total > 100"
            ),
            "total",
        ),
        [120],
        "total",
    )


def test_an_aggregate_inside_a_subquery_folds_before_the_outer_query() raises:
    # The group by runs inside and the outer query filters the answers it
    # produced, which is the shape a HAVING has and the shape a query uses when
    # it wants to filter on something a HAVING cannot say.
    var out = run(
        (
            "SELECT shop, total FROM (SELECT shop, SUM(qty) AS total FROM sales"
            " GROUP BY shop) v WHERE total > 80 ORDER BY shop"
        ),
        session(),
    )
    same(read_back(out, "shop"), [2], "shop")
    same(read_back(out, "total"), [84], "total")


def test_a_subquery_may_be_joined_to_a_table() raises:
    # The subquery is on the left because a join whose right input arrives in
    # more than one chunk raises out of the operator, which is #583 and is the
    # same with two plain tables.
    var out = run(
        (
            "SELECT band, total FROM (SELECT qty, qty * price AS total FROM"
            " sales) v JOIN tiers ON v.qty = tiers.band ORDER BY band"
        ),
        session(),
    )
    same(read_back(out, "band"), [3, 20, 40], "band")
    same(read_back(out, "total"), [21, 40, 40], "total")


def test_a_subquery_inside_a_subquery_runs_too() raises:
    same(
        answer(
            (
                "SELECT total FROM (SELECT total FROM (SELECT qty * price AS"
                " total FROM sales) inner_v WHERE total > 90) v"
            ),
            "total",
        ),
        [100, 120],
        "total",
    )


def test_a_with_hands_its_rows_to_the_query_that_names_it() raises:
    same(
        answer(
            (
                "WITH big AS (SELECT qty FROM sales WHERE qty > 20)"
                " SELECT qty FROM big"
            ),
            "qty",
        ),
        [40, 25, 30],
        "qty",
    )


def test_a_cte_read_twice_answers_the_same_both_times() raises:
    same(
        answer(
            (
                "WITH big AS (SELECT qty FROM sales WHERE qty > 20)"
                " SELECT qty FROM big UNION ALL SELECT qty FROM big"
            ),
            "qty",
        ),
        [40, 25, 30, 40, 25, 30],
        "qty",
    )


def test_a_cte_may_read_the_one_bound_before_it() raises:
    same(
        answer(
            (
                "WITH bigger AS (SELECT qty FROM sales WHERE qty > 10),"
                " fewer AS (SELECT qty FROM bigger WHERE qty < 30)"
                " SELECT qty FROM fewer"
            ),
            "qty",
        ),
        [20, 12, 25, 15],
        "qty",
    )


def test_the_alias_list_on_a_cte_renames_what_it_hands_out() raises:
    same(
        answer(
            (
                "WITH v(n) AS (SELECT qty, price FROM sales WHERE qty > 25)"
                " SELECT n FROM v"
            ),
            "n",
        ),
        [40, 30],
        "n",
    )


def test_a_cte_may_fold_and_the_query_reads_the_answer() raises:
    same(
        answer(
            (
                "WITH per AS (SELECT shop, sum(qty) AS total FROM sales"
                " GROUP BY shop) SELECT total FROM per ORDER BY total"
            ),
            "total",
        ),
        [75, 84],
        "total",
    )


def test_a_cte_may_be_joined_to_a_table() raises:
    # The CTE is on the left for the reason the subquery above it is, which is
    # #583.
    var out = run(
        (
            "WITH v AS (SELECT qty, qty * price AS total FROM sales)"
            " SELECT band, total FROM v JOIN tiers ON v.qty = tiers.band"
            " ORDER BY band"
        ),
        session(),
    )
    same(read_back(out, "band"), [3, 20, 40], "band")
    same(read_back(out, "total"), [21, 40, 40], "total")


def test_a_recursive_cte_is_refused_by_name() raises:
    with assert_raises(contains="recursive CTE"):
        _ = run(
            (
                "WITH RECURSIVE n(i) AS (SELECT 1 AS i UNION ALL"
                " SELECT i + 1 FROM n WHERE i < 5) SELECT i FROM n"
            ),
            session(),
        )


def test_two_tables_that_share_a_name_join_on_that_name() raises:
    # The star writes the shared name twice here, once from each side, which is
    # what DuckDB writes for the same query and is the whole reason the operator
    # is told its output by position.
    var out = run(
        (
            "SELECT * FROM sales JOIN shops ON sales.shop = shops.shop"
            " ORDER BY qty"
        ),
        session(),
    )
    assert_equal(len(out.schema), 5, "five columns")
    assert_equal(out.schema[2].name, "shop", "the left one")
    assert_equal(out.schema[3].name, "shop", "and the right one under its name")
    same(read_back(out, "qty"), [1, 3, 5, 8, 12, 15, 20, 25, 30, 40], "qty")
    same(read_back(out, "floor"), [22, 11, 11, 22, 11, 22, 22, 11, 11, 22], "f")


def test_a_using_join_runs_and_writes_the_pair_once() raises:
    var out = run(
        "SELECT qty, floor FROM sales JOIN shops USING (shop) ORDER BY qty",
        session(),
    )
    same(read_back(out, "qty"), [1, 3, 5, 8, 12, 15, 20, 25, 30, 40], "qty")
    same(read_back(out, "floor"), [22, 11, 11, 22, 11, 22, 22, 11, 11, 22], "f")

    var stars = run("SELECT * FROM sales NATURAL JOIN shops", session())
    assert_equal(len(stars.schema), 4, "the shared name written once")
    assert_equal(stars.schema[2].name, "shop")
    assert_equal(stars.schema[3].name, "floor")
    same(
        read_back(stars, "floor"), [11, 22, 11, 22, 11, 22, 11, 22, 11, 22], "f"
    )


def test_a_semi_join_keeps_the_left_rows_that_matched() raises:
    var out = run(
        (
            "SELECT qty, price FROM sales SEMI JOIN tiers ON qty = band"
            " ORDER BY qty"
        ),
        session(),
    )
    assert_equal(len(out.schema), 2, "the right side hands out no column")
    same(read_back(out, "qty"), [3, 20, 40], "qty")
    same(read_back(out, "price"), [7, 2, 1], "price")


def test_an_anti_join_keeps_the_left_rows_that_did_not() raises:
    # The other half of the same ten rows, which is the property worth having:
    # a semi and an anti join over one condition partition the left side.
    var out = run(
        "SELECT qty FROM sales ANTI JOIN tiers ON qty = band ORDER BY qty",
        session(),
    )
    same(read_back(out, "qty"), [1, 5, 8, 12, 15, 25, 30], "qty")


def test_a_semi_join_writes_a_left_row_once_however_many_matched() raises:
    # Band 3 is in the dupes frame twice. An inner join would answer with two
    # rows here and a semi join answers whether there was a match at all, so the
    # count is what tells the two apart.
    var out = run(
        "SELECT qty FROM sales SEMI JOIN dupes ON qty = band ORDER BY qty",
        session(),
    )
    same(read_back(out, "qty"), [3, 20], "qty")


def test_a_star_over_a_semi_join_writes_the_left_side_alone() raises:
    var out = run("SELECT * FROM sales SEMI JOIN shops USING (shop)", session())
    assert_equal(len(out.schema), 3, "the sales columns and nothing else")
    assert_equal(out.schema[0].name, "qty")
    assert_equal(out.schema[1].name, "price")
    assert_equal(out.schema[2].name, "shop")


def test_the_right_side_of_a_semi_join_cannot_be_read_above_it() raises:
    # DuckDB refuses the same query, and for the same reason: the join answered
    # a question about that table rather than joining it, so above the node
    # there is no such table to name.
    with assert_raises(contains="shops"):
        _ = run(
            "SELECT shops.floor FROM sales SEMI JOIN shops USING (shop)",
            session(),
        )


def test_an_in_over_a_subquery_runs_as_a_semi_join() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty IN (SELECT band FROM tiers)"
                " ORDER BY qty"
            ),
            "qty",
        ),
        [3, 20, 40],
        "qty",
    )


def test_an_in_answers_a_row_once_however_many_matched_it() raises:
    # Band 3 is in the dupes frame twice, and `IN` asks whether a value is in a
    # set rather than how many times.
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty IN (SELECT band FROM dupes)"
                " ORDER BY qty"
            ),
            "qty",
        ),
        [3, 20],
        "qty",
    )


def test_the_rest_of_the_where_still_holds_beside_an_in() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE price > 4"
                " AND qty IN (SELECT band FROM tiers) ORDER BY qty"
            ),
            "qty",
        ),
        [3],
        "qty",
    )


def test_a_not_in_over_a_subquery_says_why_it_is_refused() raises:
    with assert_raises(contains="null aware anti join"):
        _ = run(
            "SELECT qty FROM sales WHERE qty NOT IN (SELECT band FROM tiers)",
            session(),
        )


def test_a_correlated_exists_runs_as_a_semi_join() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE EXISTS"
                " (SELECT 1 FROM tiers WHERE tiers.band = sales.qty)"
                " ORDER BY qty"
            ),
            "qty",
        ),
        [3, 20, 40],
        "qty",
    )


def test_a_correlated_not_exists_runs_as_the_anti_join() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE NOT EXISTS"
                " (SELECT 1 FROM tiers WHERE tiers.band = sales.qty)"
                " ORDER BY qty"
            ),
            "qty",
        ),
        [1, 5, 8, 12, 15, 25, 30],
        "qty",
    )


def test_the_uncorrelated_half_of_an_exists_still_holds() raises:
    # Band 20 is a tier and its rate is under the bar, so the row the first of
    # these tests kept for it goes.
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE EXISTS"
                " (SELECT 1 FROM tiers WHERE tiers.band = sales.qty"
                " AND tiers.rate > 250) ORDER BY qty"
            ),
            "qty",
        ),
        [3, 40],
        "qty",
    )


def test_an_exists_answers_a_row_once_however_many_matched_it() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE EXISTS"
                " (SELECT 1 FROM dupes WHERE dupes.band = sales.qty)"
                " ORDER BY qty"
            ),
            "qty",
        ),
        [3, 20],
        "qty",
    )


def test_an_exists_correlated_on_a_column_that_repeats() raises:
    # The key is the shop rather than the quantity, so several outer rows share
    # a match and each is still written once.
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE EXISTS"
                " (SELECT 1 FROM shops WHERE shops.shop = sales.shop"
                " AND shops.floor = 22) ORDER BY qty"
            ),
            "qty",
        ),
        [1, 8, 15, 20, 40],
        "qty",
    )


def test_the_rest_of_the_where_still_holds_beside_an_exists() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE price > 4 AND EXISTS"
                " (SELECT 1 FROM tiers WHERE tiers.band = sales.qty)"
                " ORDER BY qty"
            ),
            "qty",
        ),
        [3],
        "qty",
    )


def test_an_uncorrelated_exists_says_why_it_is_refused() raises:
    with assert_raises(contains="mark join"):
        _ = run(
            "SELECT qty FROM sales WHERE EXISTS (SELECT 1 FROM tiers)",
            session(),
        )


def test_a_right_join_has_no_operator_yet_either() raises:
    with assert_raises(contains="breaker rather than an operator"):
        _ = run(
            "SELECT shop FROM sales RIGHT JOIN shops USING (shop)", session()
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
