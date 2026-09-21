"""The frames and helpers the sql run tests share.

These were the top of one file before it was cut into 3, which was done
because a test file is a program and every one of them compiles the slice
of the library its imports reach. That slice is the whole stack here, so
the file was the longest thing in its CI shard and the shard could not
finish faster than it did.
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

    The frame a `LIKE` needs, and nothing else in this file was written for it.
    Every row is ASCII, which is what `glyphs` is for. The rows are picked so
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


def glyphs() raises -> DataFrame:
    """Five pieces of text whose character count is not their byte count.

    Rows two and three are five characters each and six and fifteen bytes, which
    is the pair that separates a character count from a byte count. The other
    three are the cases a count written the easy way gets wrong: a row of plain
    ASCII, a row with nothing in it, and a row with nothing known about it.
    """
    var text = StringBuilder(capacity=5)
    text.append(String("abc").as_bytes())
    text.append(String("héllo").as_bytes())
    text.append(String("日本語です").as_bytes())
    text.append(String("").as_bytes())
    text.append_null()
    var word = ChunkedArray(LogicalType.STRING)
    word.append(AnyArray(text^.finish()))
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2, 3, 4, 5]))
    var g = ChunkedArray(LogicalType.INT64)
    g.append(numbers([1, 1, 2, 2, 1]))
    var columns = List[ChunkedArray]()
    columns.append(word^)
    columns.append(n^)
    columns.append(g^)
    var fields = List[Field]()
    fields.append(Field("word", LogicalType.STRING, True))
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("g", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def padded() raises -> DataFrame:
    """Six rows of text with something on the ends of most of them.

    Row two carries tabs rather than spaces, which is the row that says a `TRIM`
    asks the Zs characters and not the ones `str.strip` removes. Row four is
    empty, row five is nothing but spaces, and row six has nothing in it at all,
    which are the three rows an off by one in the scan over the ends gets wrong.
    """
    var text = StringBuilder(capacity=6)
    text.append(String("  hi  ").as_bytes())
    text.append(String("\tgo\t").as_bytes())
    text.append(String("xxaxx").as_bytes())
    text.append(String("").as_bytes())
    text.append(String("   ").as_bytes())
    text.append_null()
    var word = ChunkedArray(LogicalType.STRING)
    word.append(AnyArray(text^.finish()))
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2, 3, 4, 5, 6]))
    var columns = List[ChunkedArray]()
    columns.append(word^)
    columns.append(n^)
    var fields = List[Field]()
    fields.append(Field("word", LogicalType.STRING, True))
    fields.append(Field("n", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def moments(values: List[Int64]) raises -> AnyArray:
    """Builds a column of instants counted in seconds since the epoch."""
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return AnyArray(col^.into_data(), LogicalType.timestamp(TimeUnit.SECOND))


def lengths(values: List[Int64]) raises -> AnyArray:
    """Builds a column of lengths of time counted in seconds."""
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return AnyArray(col^.into_data(), LogicalType.duration(TimeUnit.SECOND))


def shifts() raises -> DataFrame:
    """Six shifts in two chunks, two crews, each shift a start and a length.

    A point in time and a length of one are the two halves of the temporal
    table and they answer different reductions, so both are here. The starts
    average to a fraction on purpose, since the answer truncates towards zero
    and a whole number would not show that.
    """
    var crew = ChunkedArray(LogicalType.INT64)
    crew.append(numbers([1, 2, 1]))
    crew.append(numbers([2, 1, 2]))
    var start = ChunkedArray(LogicalType.timestamp(TimeUnit.SECOND))
    start.append(moments([100, 10, 200]))
    start.append(moments([20, 301, 31]))
    var span = ChunkedArray(LogicalType.duration(TimeUnit.SECOND))
    span.append(lengths([100, 10, 200]))
    span.append(lengths([20, 301, 31]))
    var columns = List[ChunkedArray]()
    columns.append(crew^)
    columns.append(start^)
    columns.append(span^)
    var fields = List[Field]()
    fields.append(Field("crew", LogicalType.INT64))
    fields.append(Field("start", LogicalType.timestamp(TimeUnit.SECOND)))
    fields.append(Field("span", LogicalType.duration(TimeUnit.SECOND)))
    return DataFrame(Schema(fields^), columns^)


def session() raises -> Catalog:
    """A catalog holding the thirteen frames the queries write by name."""
    var catalog = Catalog()
    catalog.register("words", words())
    catalog.register("glyphs", glyphs())
    catalog.register("sales", sales())
    catalog.register("tiers", tiers())
    catalog.register("shops", shops())
    catalog.register("stock", stock())
    catalog.register("dupes", dupes())
    catalog.register("gappy", gappy())
    catalog.register("gaps", gaps())
    catalog.register("hits", hits())
    catalog.register("visits", visits())
    catalog.register("padded", padded())
    catalog.register("shifts", shifts())
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


def _marks(pattern: String, escape: String) raises -> List[Int64]:
    """Runs one escaped pattern over eight rows written around the wildcards.

    Nothing in `words` holds a `%` or a `_`, which is the whole subject of an
    escape, so these rows are written here rather than added to a frame five
    other tests read. Every answer below is DuckDB 1.5.1's over the same eight.

    Args:
        pattern: The pattern, escapes and wildcards and all.
        escape: What the `ESCAPE` names.

    Returns:
        One per row, a one for a match and a nought for none.

    Raises:
        Error: If the query is refused.
    """
    return truths(
        run(
            String(
                "SELECT col0 LIKE '",
                pattern,
                "' ESCAPE '",
                escape,
                (
                    "' AS hit FROM (VALUES ('a%b'), ('axb'), ('a_b'), ('a!b'),"
                    " ('acb'), ('ab%'), ('a%%b'), ('%')) AS t"
                ),
            ),
            session(),
        ),
        "hit",
    )


def cuts(sql: StringSlice) raises -> List[String]:
    """Runs a query that answers one text column called `piece` and reads it."""
    var col = run(sql, session()).column("piece").as_strings()
    var out = List[String](capacity=len(col))
    for i in range(len(col)):
        out.append("null" if not col.is_valid(i) else String(col[i]))
    return out^
