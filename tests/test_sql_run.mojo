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


def stock() raises -> DataFrame:
    """Six rows keyed by a shop and a quantity together.

    Here to be joined against the sales frame on both of its key columns at
    once, so it is built so that neither key on its own gives the right answer.
    Three rows are a shop and a quantity the sales frame has in the same row.
    One is a shop and a quantity it has in different rows, which a join on the
    shop alone would pair and a join on both must not. One is a shop that sells
    nothing. The last has no quantity at all, which pairs with nothing for the
    ordinary reason a null key does.
    """
    var shop = ChunkedArray(LogicalType.INT64)
    shop.append(numbers([1, 2, 1, 1, 3, 1]))
    var counted = Array[DType.int64](6)
    counted.set_valid(0, 5)
    counted.set_valid(1, 40)
    counted.set_valid(2, 12)
    counted.set_valid(3, 20)
    counted.set_valid(4, 7)
    counted.set_null(5)
    var qty = ChunkedArray(LogicalType.INT64)
    qty.append(AnyArray(counted^))
    var held = ChunkedArray(LogicalType.INT64)
    held.append(numbers([100, 200, 300, 400, 500, 600]))
    var columns = List[ChunkedArray]()
    columns.append(shop^)
    columns.append(qty^)
    columns.append(held^)
    var fields = List[Field]()
    fields.append(Field("shop", LogicalType.INT64))
    fields.append(Field("qty", LogicalType.INT64))
    fields.append(Field("held", LogicalType.INT64))
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


def gappy() raises -> DataFrame:
    """One column with a repeat and two nulls in it.

    A distinct count does not count a null, so a column that has some is the
    only way to tell that rule from the one that counts them as a value of
    their own. The repeat is why this is not `gaps` below: with no repeat a
    distinct count and a count of the non null values are the same number and
    a test over it proves nothing. Six rows, three distinct values, two nulls.
    """
    var mark = Array[DType.int64](6)
    mark.set_valid(0, 4)
    mark.set_valid(1, 4)
    mark.set_null(2)
    mark.set_valid(3, 9)
    mark.set_null(4)
    mark.set_valid(5, 1)
    var column = ChunkedArray(LogicalType.INT64)
    column.append(AnyArray(mark^))
    var columns = List[ChunkedArray]()
    columns.append(column^)
    var fields = List[Field]()
    fields.append(Field("mark", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def gaps() raises -> DataFrame:
    """Three bands with a null among them.

    A `NOT IN` over a subquery holding a null keeps no rows at all, because a
    row that matched nothing cannot be told apart from a row that matched the
    null, and every other frame here would answer that question the easy way.
    """
    var col = Array[DType.int64](3)
    col.set_valid(0, 3)
    col.set_valid(1, 20)
    col.set_null(2)
    var band = ChunkedArray(LogicalType.INT64)
    band.append(AnyArray(col^))
    var columns = List[ChunkedArray]()
    columns.append(band^)
    var fields = List[Field]()
    fields.append(Field("band", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def session() raises -> Catalog:
    """A catalog holding the seven frames under the names the queries write."""
    var catalog = Catalog()
    catalog.register("sales", sales())
    catalog.register("tiers", tiers())
    catalog.register("shops", shops())
    catalog.register("stock", stock())
    catalog.register("dupes", dupes())
    catalog.register("gappy", gappy())
    catalog.register("gaps", gaps())
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


def gapped(df: DataFrame, name: String) raises -> List[Int64]:
    """Reads an int64 column out, with a null as a minus one.

    Minus one rather than an option because nothing in these fixtures holds
    one, so a minus one in the answer is a null and reads as one.
    """
    var col = df.column(name).as_typed[DType.int64]()
    var out = List[Int64](capacity=len(col))
    for i in range(len(col)):
        if not col.is_valid(i):
            out.append(-1)
        else:
            out.append(col[i])
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


def test_the_column_aliases_on_a_derived_table_run() raises:
    # The list renames what the subquery produced, so the outer query reads
    # `worth` and the name the subquery gave the column is gone.
    same(
        answer(
            (
                "SELECT worth FROM (SELECT qty * price AS total FROM sales)"
                " v(worth) WHERE worth > 90"
            ),
            "worth",
        ),
        [100, 120],
        "worth",
    )


def test_a_short_alias_list_on_a_derived_table_leaves_the_rest_alone() raises:
    # Two columns and one name, so the first is renamed and the second keeps
    # what it had, which is the prefix rule.
    var out = run(
        (
            "SELECT much, price FROM (SELECT qty, price FROM sales) v(much)"
            " WHERE much > 25"
        ),
        session(),
    )
    same(read_back(out, "much"), [40, 30], "much")
    same(read_back(out, "price"), [1, 4], "price")


def test_more_aliases_than_a_derived_table_produces_is_refused() raises:
    with assert_raises(contains="has 1 columns available but 2 columns"):
        _ = run("SELECT a FROM (SELECT qty FROM sales) v(a, b)", session())


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


def test_a_not_in_over_a_subquery_keeps_the_rows_that_matched_nothing() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty NOT IN"
                " (SELECT band FROM tiers) ORDER BY qty"
            ),
            "qty",
        ),
        [1, 5, 8, 12, 15, 25, 30],
        "qty",
    )


def test_a_not_in_over_a_subquery_holding_a_null_keeps_nothing() raises:
    # The answer everyone gets wrong and DuckDB gets right. A row that matched
    # nothing might have matched the null, so it is null rather than true, the
    # NOT over it stays null, and a WHERE keeps a row on true.
    assert_equal(
        len(
            run(
                (
                    "SELECT qty FROM sales WHERE qty NOT IN"
                    " (SELECT band FROM gaps)"
                ),
                session(),
            )
        ),
        0,
    )


def test_an_in_written_in_the_select_list_answers_on_every_row() raises:
    same(
        truths(
            run(
                "SELECT qty IN (SELECT band FROM tiers) AS hit FROM sales",
                session(),
            ),
            "hit",
        ),
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "hit",
    )


def test_an_in_over_a_subquery_holding_a_null_is_null_where_it_missed() raises:
    same(
        truths(
            run(
                "SELECT qty IN (SELECT band FROM gaps) AS hit FROM sales",
                session(),
            ),
            "hit",
        ),
        [-1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
        "hit",
    )


def test_a_not_in_written_in_the_select_list_is_that_negated() raises:
    same(
        truths(
            run(
                "SELECT qty NOT IN (SELECT band FROM gaps) AS gone FROM sales",
                session(),
            ),
            "gone",
        ),
        [-1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
        "gone",
    )


def test_an_in_under_an_or_keeps_what_either_side_keeps() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE price > 4 OR qty IN"
                " (SELECT band FROM tiers) ORDER BY qty"
            ),
            "qty",
        ),
        [1, 3, 5, 8, 12, 15, 20, 40],
        "qty",
    )


def test_an_equals_any_keeps_what_an_in_keeps() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty = ANY"
                " (SELECT band FROM tiers) ORDER BY qty"
            ),
            "qty",
        ),
        [3, 20, 40],
        "qty",
    )


def test_a_not_equals_all_over_a_null_keeps_nothing() raises:
    # The same null aware answer a NOT IN gets, because it is the same lowering
    # and there is nothing written for this case anywhere.
    assert_equal(
        len(
            run(
                (
                    "SELECT qty FROM sales WHERE qty <> ALL"
                    " (SELECT band FROM gaps)"
                ),
                session(),
            )
        ),
        0,
    )


def test_an_equals_any_answers_on_every_row() raises:
    same(
        truths(
            run(
                "SELECT qty = ANY (SELECT band FROM tiers) AS hit FROM sales",
                session(),
            ),
            "hit",
        ),
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "hit",
    )


def test_a_greater_than_any_keeps_what_beat_the_smallest_row() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty > ANY"
                " (SELECT band FROM tiers) ORDER BY qty"
            ),
            "qty",
        ),
        [5, 8, 12, 15, 20, 25, 30, 40],
        "qty",
    )


def test_a_greater_than_any_answers_on_every_row() raises:
    same(
        truths(
            run(
                "SELECT qty > ANY (SELECT band FROM tiers) AS over FROM sales",
                session(),
            ),
            "over",
        ),
        [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
        "over",
    )


def test_a_greater_or_equal_all_is_false_where_nothing_reaches_the_top() raises:
    # The largest band is 99 and no sale reaches it, so every row is false and
    # none of them is null, because the subquery holds no null.
    same(
        truths(
            run(
                "SELECT qty >= ALL (SELECT band FROM tiers) AS top FROM sales",
                session(),
            ),
            "top",
        ),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "top",
    )


def test_a_less_than_all_is_true_only_under_the_smallest_row() raises:
    same(
        truths(
            run(
                "SELECT qty < ALL (SELECT band FROM tiers) AS under FROM sales",
                session(),
            ),
            "under",
        ),
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "under",
    )


def test_a_not_equals_any_is_true_wherever_the_two_ends_differ() raises:
    same(
        truths(
            run(
                (
                    "SELECT qty <> ANY (SELECT band FROM tiers) AS other"
                    " FROM sales"
                ),
                session(),
            ),
            "other",
        ),
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "other",
    )


def test_an_equals_all_is_false_over_a_subquery_of_several_rows() raises:
    same(
        truths(
            run(
                "SELECT qty = ALL (SELECT band FROM tiers) AS only FROM sales",
                session(),
            ),
            "only",
        ),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "only",
    )


def test_a_null_in_the_subquery_turns_a_false_into_a_null() raises:
    # The answer this shape is for. A row that beat every band there was to
    # beat did not beat the null, so it is neither true nor false, while a row
    # that lost to a band it could see is false whatever the null was.
    same(
        truths(
            run(
                "SELECT qty > ALL (SELECT band FROM gaps) AS over FROM sales",
                session(),
            ),
            "over",
        ),
        [0, 0, 0, -1, 0, 0, -1, 0, -1, 0],
        "over",
    )


def test_a_null_in_the_subquery_leaves_a_true_alone() raises:
    same(
        truths(
            run(
                "SELECT qty > ANY (SELECT band FROM gaps) AS over FROM sales",
                session(),
            ),
            "over",
        ),
        [1, 1, -1, 1, 1, 1, 1, -1, 1, 1],
        "over",
    )


def test_a_quantified_comparison_over_no_rows_is_the_quantifier() raises:
    # `ANY` over nothing is false and `ALL` over nothing is true, whatever is
    # on the other side of the comparison, and the fold hands out the one row
    # that says the subquery was empty.
    same(
        truths(
            run(
                (
                    "SELECT qty > ANY (SELECT band FROM tiers WHERE band >"
                    " 1000) AS over FROM sales"
                ),
                session(),
            ),
            "over",
        ),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "over",
    )
    same(
        truths(
            run(
                (
                    "SELECT qty > ALL (SELECT band FROM tiers WHERE band >"
                    " 1000) AS over FROM sales"
                ),
                session(),
            ),
            "over",
        ),
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "over",
    )


def test_a_where_keeps_a_row_on_true_and_not_on_a_null() raises:
    assert_equal(
        len(
            run(
                "SELECT qty FROM sales WHERE qty > ALL (SELECT band FROM gaps)",
                session(),
            )
        ),
        0,
    )


def test_an_in_over_a_subquery_that_kept_no_rows_keeps_no_rows() raises:
    # The build side is a column no rows reached, which has no chunks at all
    # rather than one empty chunk, and the join used to raise on that. #611.
    assert_equal(
        len(
            run(
                (
                    "SELECT qty FROM sales WHERE qty IN"
                    " (SELECT band FROM tiers WHERE band > 1000)"
                ),
                session(),
            )
        ),
        0,
    )


def test_an_in_over_a_subquery_that_kept_no_rows_is_false_as_a_value() raises:
    # False rather than null, since there is nothing to match and no null in
    # what was not matched against.
    same(
        truths(
            run(
                (
                    "SELECT qty IN (SELECT band FROM tiers WHERE band > 1000)"
                    " AS hit FROM sales"
                ),
                session(),
            ),
            "hit",
        ),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "hit",
    )


def test_a_not_in_over_a_subquery_that_kept_no_rows_keeps_them_all() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty NOT IN"
                " (SELECT band FROM tiers WHERE band > 1000) ORDER BY qty"
            ),
            "qty",
        ),
        [1, 3, 5, 8, 12, 15, 20, 25, 30, 40],
        "qty",
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


def test_an_uncorrelated_exists_keeps_every_row_or_none() raises:
    # It asks whether the table has any row at all, which every outer row gets
    # the same answer to, so it is counted under a cross join rather than joined
    # on. The table has rows, so every row is kept.
    assert_equal(
        len(
            run(
                "SELECT qty FROM sales WHERE EXISTS (SELECT 1 FROM tiers)",
                session(),
            )
        ),
        10,
    )


def test_a_subquery_that_answers_one_value_runs() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty > (SELECT min(band) FROM"
                " tiers) ORDER BY qty"
            ),
            "qty",
        ),
        [5, 8, 12, 15, 20, 25, 30, 40],
        "qty",
    )


def test_two_subqueries_in_one_where_both_run() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty > (SELECT min(band) FROM"
                " tiers) AND price < (SELECT min(band) FROM tiers) ORDER BY qty"
            ),
            "qty",
        ),
        [20, 40],
        "qty",
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
    # The first pair is the one the table is built from and the rest are asked
    # afterwards, so which pair is written first decides the plan and must not
    # decide the answer.
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
    # A join on two keys is an operator and a filter rather than one operator,
    # so the thing above it has to see an ordinary chunk. A reduction is the
    # cheapest way to ask that.
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


def test_a_semi_join_on_two_keys_is_refused_for_now() raises:
    # An inner join keeps both sides' columns, so the rest of the key can be
    # asked after the pairing. A semi join keeps none of them and cannot.
    with assert_raises(contains="2 key pairs would need the ordinal space"):
        _ = run(
            (
                "SELECT qty FROM sales WHERE EXISTS (SELECT 1 FROM stock"
                " WHERE stock.shop = sales.shop AND stock.qty = sales.qty)"
            ),
            session(),
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
