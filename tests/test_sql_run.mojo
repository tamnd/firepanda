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
from firepanda.array.strings import StringBuilder
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.dtype.temporal import TimeUnit
from firepanda.frame.frame import DataFrame
from firepanda.sql.catalog import Catalog
from firepanda.sql.run import run


def numbers(values: List[Int64]) raises -> AnyArray:
    """Builds a fully valid int64 array."""
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return AnyArray(col^)


def days(values: List[Int32]) raises -> ChunkedArray:
    """Builds a one chunk date32 column out of counts of days since 1970."""
    var col = Array[DType.int32](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    var out = ChunkedArray(LogicalType.DATE32)
    out.append(AnyArray(col^.into_data(), LogicalType.DATE32))
    return out^


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


def words() raises -> DataFrame:
    """Seven pieces of text and a number saying which row each one is.

    The only frame here with a string column in it, which is what a `LIKE` needs
    and what nothing else in this file was written for. The rows are picked so
    that each of the four searches keeps a different set: two share a prefix,
    two share a suffix, one holds a run in the middle, one holds two runs in
    order, one is empty and one is null.

    The empty string and the null are the two that catch a search written the
    easy way. An empty element matches `%` and matches nothing else, and a null
    matches nothing at all and is not false either.
    """
    var text = StringBuilder(capacity=7)
    text.append(String("apple").as_bytes())
    text.append(String("apricot").as_bytes())
    text.append(String("banana").as_bytes())
    text.append(String("grape").as_bytes())
    text.append(String("").as_bytes())
    text.append_null()
    text.append(String("pineapple").as_bytes())
    var word = ChunkedArray(LogicalType.STRING)
    word.append(AnyArray(text^.finish()))
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2, 3, 4, 5, 6, 7]))
    var columns = List[ChunkedArray]()
    columns.append(word^)
    columns.append(n^)
    var fields = List[Field]()
    fields.append(Field("word", LogicalType.STRING, True))
    fields.append(Field("n", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def hits() raises -> DataFrame:
    """Four rows under ClickBench's spelling, which is not the query's.

    A parquet file writes its column names however it likes and `AdvEngineID`
    is how that suite writes one. The tokenizer folds a bare name down, so
    nothing a query writes bare arrives spelled this way and the resolver is
    what has to bridge it.

    `EventDate` is a date32 column and the four days it holds are a day either
    side of July 2013 and two inside it, so a range written the way seven of
    the ClickBench statements write one keeps the middle two and neither bound
    is the whole column.
    """
    var engine = ChunkedArray(LogicalType.INT64)
    engine.append(numbers([0, 2, 2, 3]))
    var region = ChunkedArray(LogicalType.INT64)
    region.append(numbers([7, 7, 9, 9]))
    var columns = List[ChunkedArray]()
    columns.append(engine^)
    columns.append(region^)
    columns.append(days([15886, 15887, 15901, 15918]))
    var fields = List[Field]()
    fields.append(Field("AdvEngineID", LogicalType.INT64))
    fields.append(Field("RegionID", LogicalType.INT64))
    fields.append(Field("EventDate", LogicalType.DATE32))
    return DataFrame(Schema(fields^), columns^)


def visits() raises -> DataFrame:
    """Six user ids near 4e18, three to a site.

    Real user ids are this size and there are a lot of them, which is the whole
    reason #673 existed: three of these add up to 1.2e19 and int64 stops at
    9.22e18. A sum over the column is allowed to wrap, and does. A mean is not.
    """
    var base = Int64(4_000_000_000_000_000_000)
    var user = ChunkedArray(LogicalType.INT64)
    user.append(numbers([base, base + 2, base + 4]))
    user.append(numbers([base + 6, base + 8, base + 10]))
    var site = ChunkedArray(LogicalType.INT64)
    site.append(numbers([1, 2, 1]))
    site.append(numbers([2, 1, 2]))
    var columns = List[ChunkedArray]()
    columns.append(user^)
    columns.append(site^)
    var fields = List[Field]()
    fields.append(Field("user_id", LogicalType.INT64))
    fields.append(Field("site", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def session() raises -> Catalog:
    """A catalog holding the nine frames under the names the queries write."""
    var catalog = Catalog()
    catalog.register("words", words())
    catalog.register("sales", sales())
    catalog.register("tiers", tiers())
    catalog.register("shops", shops())
    catalog.register("stock", stock())
    catalog.register("dupes", dupes())
    catalog.register("gappy", gappy())
    catalog.register("gaps", gaps())
    catalog.register("hits", hits())
    catalog.register("visits", visits())
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


def test_a_like_with_an_underscore_in_it_says_what_it_cannot_do() raises:
    with assert_raises(contains="stands for any one character"):
        _ = run("SELECT n FROM words WHERE word LIKE 'a_p%'", session())


def test_a_like_with_a_run_in_the_middle_is_refused() raises:
    # `a%e` is a prefix and a suffix at once and neither kernel answers it, and
    # answering it as one of the two would keep rows the query did not ask for.
    with assert_raises(contains="is none of those"):
        _ = run("SELECT n FROM words WHERE word LIKE 'a%e'", session())


def test_a_like_against_a_column_is_refused() raises:
    with assert_raises(contains="pattern of a LIKE has to be written out"):
        _ = run("SELECT n FROM words WHERE word LIKE word", session())


def test_an_is_null_keeps_the_rows_with_nothing_in_them() raises:
    same(
        gapped(
            run("SELECT mark FROM gappy WHERE mark IS NULL", session()), "mark"
        ),
        [-1, -1],
        "mark",
    )


def test_an_is_not_null_keeps_the_others() raises:
    same(
        answer("SELECT mark FROM gappy WHERE mark IS NOT NULL", "mark"),
        [4, 4, 9, 1],
        "mark",
    )


def test_an_is_null_in_a_select_list_answers_yes_or_no_for_every_row() raises:
    # Never a null itself, whatever the column under it holds, which is what
    # tells this apart from `mark = NULL` and is the reason SQL has the words.
    same(
        truths(
            run("SELECT mark IS NULL AS gone FROM gappy", session()), "gone"
        ),
        [0, 0, 1, 0, 1, 0],
        "gone",
    )


def test_the_one_word_spellings_mean_the_same_two_tests() raises:
    # `ISNULL` and `NOTNULL` are postfix words rather than functions, and the
    # parser folds each of them into the node the two word form builds.
    same(
        answer("SELECT mark FROM gappy WHERE mark NOTNULL", "mark"),
        [4, 4, 9, 1],
        "mark",
    )
    same(
        gapped(
            run("SELECT mark FROM gappy WHERE mark ISNULL", session()), "mark"
        ),
        [-1, -1],
        "mark",
    )


def test_an_is_null_reads_a_column_of_text_too() raises:
    same(answer("SELECT n FROM words WHERE word IS NULL", "n"), [6], "n")


def test_an_is_true_keeps_the_rows_the_comparison_held_for() raises:
    # `mark > 3` is true, true, null, true, null, false down the six rows, so
    # each of the four tests below keeps a different set and no two of them
    # would agree if the null were being read as a false.
    same(
        answer("SELECT mark FROM gappy WHERE (mark > 3) IS TRUE", "mark"),
        [4, 4, 9],
        "mark",
    )


def test_an_is_false_keeps_the_row_it_did_not_hold_for() raises:
    same(
        answer("SELECT mark FROM gappy WHERE (mark > 3) IS FALSE", "mark"),
        [1],
        "mark",
    )


def test_an_is_not_true_keeps_the_nulls_with_the_false() raises:
    same(
        gapped(
            run(
                "SELECT mark FROM gappy WHERE (mark > 3) IS NOT TRUE", session()
            ),
            "mark",
        ),
        [-1, -1, 1],
        "mark",
    )


def test_an_is_not_false_keeps_the_nulls_with_the_true() raises:
    same(
        gapped(
            run(
                "SELECT mark FROM gappy WHERE (mark > 3) IS NOT FALSE",
                session(),
            ),
            "mark",
        ),
        [4, 4, -1, 9, -1],
        "mark",
    )


def test_a_yes_written_out_keeps_every_row_and_a_no_keeps_none() raises:
    # The word is written in capitals by the parser whatever the query spelled
    # it with, and reading it as lower case made every one of them a no.
    same(
        gapped(run("SELECT mark FROM gappy WHERE true", session()), "mark"),
        [4, 4, -1, 9, -1, 1],
        "mark",
    )
    assert_equal(
        len(run("SELECT mark FROM gappy WHERE FALSE", session())),
        0,
        "a no keeps nothing",
    )


def cuts(sql: StringSlice) raises -> List[String]:
    """Runs a query that answers one text column called `piece` and reads it."""
    var col = run(sql, session()).column("piece").as_strings()
    var out = List[String](capacity=len(col))
    for i in range(len(col)):
        out.append("null" if not col.is_valid(i) else String(col[i]))
    return out^


def test_a_substring_takes_the_characters_the_query_named() raises:
    # `words` is apple, apricot, banana, grape, the empty string, a null and
    # pineapple, so three from the front keeps a different set of letters for
    # each of them and leaves the last two alone.
    var got = cuts("SELECT substring(word, 1, 3) AS piece FROM words")
    assert_equal(len(got), 7, "one answer per row")
    assert_equal(got[0], "app", "the first")
    assert_equal(got[2], "ban", "and one from the middle")
    assert_equal(got[4], "", "the empty string has nothing to take")
    assert_equal(got[5], "null", "and a null stays a null")


def test_a_substring_with_no_length_runs_to_the_end() raises:
    var got = cuts("SELECT substring(word, 4) AS piece FROM words")
    assert_equal(got[0], "le", "what was left of a five letter word")
    assert_equal(got[6], "eapple", "and of a nine letter one")


def test_a_substr_is_the_same_function_under_duckdbs_other_name() raises:
    var got = cuts("SELECT substr(word, 2, 2) AS piece FROM words")
    assert_equal(got[0], "pp", "the first")
    assert_equal(got[6], "in", "and the last")


def test_the_keyword_spelling_reads_the_same_two_numbers() raises:
    var got = cuts("SELECT SUBSTRING(word FROM 2 FOR 2) AS piece FROM words")
    assert_equal(got[0], "pp", "the first")
    assert_equal(got[6], "in", "and the last")


def test_the_keyword_spelling_without_a_from_starts_at_the_first_letter() raises:
    var got = cuts("SELECT SUBSTRING(word FOR 3) AS piece FROM words")
    assert_equal(got[0], "app", "the first")
    assert_equal(got[6], "pin", "and the last")


def test_a_substring_in_a_where_reads_the_cut_column() raises:
    same(
        answer("SELECT n FROM words WHERE substring(word, 1, 2) = 'ap'", "n"),
        [1, 2],
        "n",
    )


def test_a_substring_whose_start_is_a_column_is_refused() raises:
    with assert_raises(contains="have to be written out"):
        _ = run("SELECT substring(word, n, 2) FROM words", session())


def test_a_substring_of_a_number_is_refused() raises:
    with assert_raises(contains="'substring' reads text"):
        _ = run("SELECT substring(n, 1, 2) FROM words", session())


def test_a_coalesce_fills_the_gaps_from_the_second_argument() raises:
    same(
        read_back(
            run("SELECT coalesce(mark, 99) AS m FROM gappy", session()), "m"
        ),
        [4, 4, 99, 9, 99, 1],
        "m",
    )


def test_a_coalesce_reads_its_arguments_in_the_order_written() raises:
    # The middle one is a null and fills nothing, so the third is what the gaps
    # come from, and a version that stopped at the first fallback would answer
    # a column that still had two gaps in it.
    same(
        read_back(
            run("SELECT coalesce(mark, NULL, 7) AS m FROM gappy", session()),
            "m",
        ),
        [4, 4, 7, 9, 7, 1],
        "m",
    )


def test_a_coalesce_of_one_argument_is_that_argument() raises:
    same(
        gapped(run("SELECT coalesce(mark) AS m FROM gappy", session()), "m"),
        [4, 4, -1, 9, -1, 1],
        "m",
    )


def test_an_ifnull_is_a_coalesce_of_two() raises:
    same(
        read_back(
            run("SELECT ifnull(mark, 0) AS m FROM gappy", session()), "m"
        ),
        [4, 4, 0, 9, 0, 1],
        "m",
    )


def test_a_nullif_takes_the_value_out_where_the_two_agree() raises:
    # The two nulls stay nulls. `mark = 4` is null on those rows rather than
    # false, the conditional takes its else side, and the else side is `mark`.
    same(
        gapped(run("SELECT nullif(mark, 4) AS m FROM gappy", session()), "m"),
        [-1, -1, -1, 9, -1, 1],
        "m",
    )


def test_a_coalesce_in_a_where_reads_the_filled_column() raises:
    same(
        answer("SELECT mark FROM gappy WHERE coalesce(mark, 0) > 3", "mark"),
        [4, 4, 9],
        "mark",
    )


def test_a_coalesce_moves_both_sides_to_the_type_they_agree_on() raises:
    # The column holds whole numbers and the fallback does not, so the answer is
    # the wider of the two and the column is what moves. The fallback is cast
    # rather than written as a fraction because the decimal literal is still
    # refused, which is a gap of its own and not this one.
    var out = run(
        "SELECT coalesce(mark, CAST(0 AS DOUBLE)) AS m FROM gappy", session()
    )

    assert_true(
        out.schema[0].dtype == LogicalType.FLOAT64, "the wider of the two"
    )
    var col = out.column("m").as_typed[DType.float64]()
    assert_equal(len(col), 6, "one answer per row")
    assert_equal(col[0], 4.0, "the value that was already there")
    assert_equal(col[2], 0.0, "and the gap taken from the fallback")


def test_a_coalesce_fills_a_column_of_text() raises:
    same(
        answer("SELECT n FROM words WHERE coalesce(word, 'zz') = 'zz'", "n"),
        [6],
        "n",
    )


def test_a_coalesce_whose_arguments_do_not_agree_is_refused() raises:
    with assert_raises(contains="have to agree on a type"):
        _ = run("SELECT coalesce(mark, 'a') FROM gappy", session())


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


def test_a_correlated_subquery_that_counts_is_refused_by_name() raises:
    with assert_raises(contains="a count of nothing is zero"):
        _ = run(
            (
                "SELECT shop, (SELECT count(qty) FROM sales WHERE sales.shop ="
                " shops.shop) AS n FROM shops"
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
    # The field is read in the derived table and grouped on by name, rather than
    # written out twice in the one statement, because a GROUP BY of an
    # expression does not lower yet whatever the expression is. The select list
    # is lowered above the aggregate over the columns the aggregate produces,
    # and nothing there compares an item against the group keys, so only a key
    # that is a plain column survives, since that one the aggregate carries
    # through under the name it already had. That is a gap in grouping and not
    # one in EXTRACT.
    var out = run(
        (
            "SELECT m, count(*) AS n FROM (SELECT EXTRACT(MONTH FROM eventdate)"
            " AS m FROM hits) GROUP BY m ORDER BY 1"
        ),
        session(),
    )
    same(read_back(out, "m"), [6, 7, 8], "one row per month")
    same(read_back(out, "n"), [1, 2, 1], "and the count in each")


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
    with assert_raises(contains="no field SQL calls epoch"):
        _ = run("SELECT EXTRACT(EPOCH FROM eventdate) FROM hits", session())


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
    # Grouped through a derived table, for the reason the same test on a field
    # read is: a GROUP BY of an expression does not lower yet.
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
