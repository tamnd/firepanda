"""The frames and helpers the pipeline tests share.

These were the top of one file before it was cut into 3, which was done
because a test file is a program and every one of them compiles the slice
of the library its imports reach. That slice is the whole stack here, so
the file was the longest thing in its CI shard and the shard could not
finish faster than it did.
"""


from std.math import isnan, nan
from std.testing import TestSuite, assert_equal, assert_false, assert_raises
from std.testing import assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.array.strings import strings_from_list
from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.dtype.temporal import TimeUnit
from firepanda.exec import (
    Apply,
    Case,
    Cast,
    Chunk,
    Collect,
    Compute,
    Constant,
    Cut,
    Expand,
    Fill,
    Filter,
    Group,
    GroupAgg,
    Join,
    Length,
    Limit,
    Locate,
    Match,
    Materialize,
    Node,
    NodeStatus,
    Part,
    Pipeline,
    Presence,
    Project,
    Reduce,
    Scan,
    Sort,
    Trim,
    Truncate,
    Unique,
    Window,
    node_apply,
    node_computes_per_row,
    node_ends_early,
    node_is_breaker,
    node_is_row_local,
    node_process,
    node_status,
)
from firepanda.exec.morsel import MORSEL_ROWS
from firepanda.frame.frame import DataFrame
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.pattern import MatchKind, Pattern
from firepanda.kernel.temporal import (
    TRUNC_DAY,
    TRUNC_MONTH,
    TRUNC_WEEK,
    TRUNC_YEAR,
    TemporalField,
)
from firepanda.kernel.unary import UnaryOp


def numbers(values: List[Int64]) raises -> AnyArray:
    """Builds a fully valid int64 array."""
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return AnyArray(col^)


def flags(values: List[Bool]) raises -> AnyArray:
    """Builds a fully valid boolean array."""
    var col = Array[DType.bool](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return AnyArray(col^)


def sample_frame() raises -> DataFrame:
    """Six rows in one chunk: a counting column and a mask over it.

    The mask keeps rows 0, 2, 3 and 5, so it is neither everything nor a prefix,
    and the values it keeps are 1, 3, 4 and 6.
    """
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(flags([True, False, True, True, False, True]))
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.BOOL))
    return DataFrame(Schema(fields^), columns^)


def cut_frame() raises -> DataFrame:
    """The same six rows, in chunks of two, three and one."""
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2]))
    n.append(numbers([3, 4, 5]))
    n.append(numbers([6]))
    var keep = ChunkedArray(LogicalType.BOOL)
    keep.append(flags([True, False]))
    keep.append(flags([True, True, False]))
    keep.append(flags([True]))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(keep^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.BOOL))
    return DataFrame(Schema(fields^), columns^)


def many_chunk_frame() raises -> DataFrame:
    """Two hundred rows in forty chunks of five, with a mask over them.

    Forty is chosen to be more than one batch on this machine and more than one
    on a smaller one, since a batch is one chunk per worker and the driver takes
    the parallel route from two chunks upwards. The mask drops every third row,
    so what survives is neither a prefix nor a stride the reader could guess,
    and the numbers are consecutive, so the order of the answer is checkable by
    looking at it rather than by comparing against a second run.
    """
    var n = ChunkedArray(LogicalType.INT64)
    var keep = ChunkedArray(LogicalType.BOOL)
    for c in range(40):
        var values = List[Int64]()
        var mask = List[Bool]()
        for r in range(5):
            var v = Int64(c * 5 + r + 1)
            values.append(v)
            mask.append(v % 3 != 0)
        n.append(numbers(values))
        keep.append(flags(mask))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(keep^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.BOOL))
    return DataFrame(Schema(fields^), columns^)


def tall_frame() raises -> DataFrame:
    """One chunk of a morsel and three rows, with a mask over it.

    This is the shape every reader hands back, one chunk however many rows were
    read, and it is the only shape the morsel cut has anything to say about. The
    three rows past the morsel are there so the second piece is a short one,
    which is the piece an off by one in the cut gets wrong.
    """
    var rows = MORSEL_ROWS + 3
    var values = List[Int64](capacity=rows)
    var mask = List[Bool](capacity=rows)
    for i in range(rows):
        values.append(Int64(i + 1))
        mask.append(i % 3 != 0)
    var columns = List[AnyArray]()
    columns.append(numbers(values))
    columns.append(flags(mask))
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.BOOL))
    return DataFrame(Schema(fields^), columns^)


def word_frame() raises -> DataFrame:
    """The same six rows with two text columns beside the numbers.

    `status` and `wanted` agree on rows 0, 2 and 4, so a comparison between the
    two columns keeps three rows, and `status` holds "ok" on four of the six, so a
    comparison against that constant keeps a different set. Two answers that
    differ is the point: a comparison that ignored one of its operands would give
    the same set twice.
    """
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(
        AnyArray(strings_from_list(["ok", "fail", "ok", "ok", "fail", "ok"]))
    )
    columns.append(
        AnyArray(strings_from_list(["ok", "ok", "ok", "fail", "fail", "no"]))
    )
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("status", LogicalType.STRING))
    fields.append(Field("wanted", LogicalType.STRING))
    return DataFrame(Schema(fields^), columns^)


def spaced_frame() raises -> DataFrame:
    """Six rows of text with something on the ends of most of them.

    Row one has tabs on it rather than spaces, which is the row that says which
    whitespace table a trim is asking. Row three is empty and row four is
    nothing but spaces, which are the two rows that come out empty and are the
    ones an off by one in the scan over the ends would get wrong.
    """
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(
        AnyArray(
            strings_from_list(["  hi  ", "\tgo\t", "xxaxx", "", "   ", "end  "])
        )
    )
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("padded", LogicalType.STRING))
    return DataFrame(Schema(fields^), columns^)


def read_back(df: DataFrame, name: String) raises -> List[Int64]:
    """Reads an int64 column out as a plain list."""
    var col = df.column(name).as_typed[DType.int64]()
    var out = List[Int64](capacity=len(col))
    for i in range(len(col)):
        out.append(col[i])
    return out^


def gappy_frame() raises -> DataFrame:
    """Six rows in two chunks, with a null in each of them.

    The nulls sit in different chunks on purpose, since a sort flattens the
    column before it orders it and a null in the second chunk is the one whose
    validity bit has to travel the furthest.
    """
    var n = ChunkedArray(LogicalType.INT64)
    var first = Array[DType.int64](3)
    first.set_valid(0, Int64(4))
    first.set_null(1)
    first.set_valid(2, Int64(1))
    n.append(AnyArray(first^))
    var second = Array[DType.int64](3)
    second.set_valid(0, Int64(6))
    second.set_valid(1, Int64(2))
    second.set_null(2)
    n.append(AnyArray(second^))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64, True))
    return DataFrame(Schema(fields^), columns^)


def dated_frame() raises -> DataFrame:
    """Four days, one of them missing, and a number beside them.

    The days are the 30th of June 2013, the 1st and the 15th of July and the 1st
    of August, counted from the epoch, which is a month boundary either side of
    a month and one day inside it. The missing row is there because a field read
    off nothing has to come back as nothing.
    """
    var d = ChunkedArray(LogicalType.DATE32)
    var days = Array[DType.int32](4)
    days.set_valid(0, Int32(15886))
    days.set_valid(1, Int32(15887))
    days.set_null(2)
    days.set_valid(3, Int32(15918))
    d.append(AnyArray(days^.into_data(), LogicalType.DATE32))
    var n = ChunkedArray(LogicalType.INT64)
    var counts = Array[DType.int64](4)
    for i in range(4):
        counts.set_valid(i, Int64(i))
    n.append(AnyArray(counts^))
    var columns = List[ChunkedArray]()
    columns.append(d^)
    columns.append(n^)
    var fields = List[Field]()
    fields.append(Field("d", LogicalType.DATE32, True))
    fields.append(Field("n", LogicalType.INT64, False))
    return DataFrame(Schema(fields^), columns^)


def present(df: DataFrame, name: String) raises -> List[Bool]:
    """Which rows of an int64 column have a value in them."""
    var col = df.column(name).as_typed[DType.int64]()
    var out = List[Bool](capacity=len(col))
    for i in range(len(col)):
        out.append(col.is_valid(i))
    return out^


def identity(var df: DataFrame) raises -> DataFrame:
    """A whole frame operation that does nothing, for the fallback tests."""
    return df^


def first_two(var df: DataFrame) raises -> DataFrame:
    """A whole frame operation that keeps the first two rows."""
    return df.head(2)


def only_n(var df: DataFrame) raises -> DataFrame:
    """A whole frame operation that drops a column, so the schema changes."""
    return df.select(["n"])


def shelf_frame() raises -> DataFrame:
    """Six rows in chunks of two, three and one: a shop and what it held.

    Shop 1 is written three times and shop 2 twice, and the repeats are split
    over the chunk boundaries, so a distinct on the shop cannot answer from one
    chunk. What each row held differs inside a group, so which row of a group
    survived can be read off the answer.
    """
    var shop = ChunkedArray(LogicalType.INT64)
    shop.append(numbers([1, 2]))
    shop.append(numbers([1, 3, 2]))
    shop.append(numbers([1]))
    var held = ChunkedArray(LogicalType.INT64)
    held.append(numbers([10, 20]))
    held.append(numbers([30, 40, 50]))
    held.append(numbers([60]))
    var columns = List[ChunkedArray]()
    columns.append(shop^)
    columns.append(held^)
    var fields = List[Field]()
    fields.append(Field("shop", LogicalType.INT64))
    fields.append(Field("held", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def _matched(
    kind: MatchKind, first: String, second: String
) raises -> List[Int64]:
    """Runs one search over `wanted` and reads back the numbers it kept.

    Args:
        kind: Which search to run.
        first: The run of bytes to look for.
        second: The run that has to follow it, empty for the other three.

    Returns:
        The `n` of every row the pattern matched.
    """
    var pipeline = Pipeline(word_frame())
    pipeline.add(Node(Match(2, Pattern(kind, first, second), "hit")))
    pipeline.add(Node(Filter(3)))
    pipeline.add(Node(Project([0])))
    return read_back(pipeline^.run(), "n")


def _presence(missing: Bool) raises -> List[Bool]:
    """Runs one null test over `gappy_frame` and reads the answer back.

    Args:
        missing: True to ask which rows are null.

    Returns:
        One flag per row, in order.
    """
    var pipeline = Pipeline(gappy_frame())
    pipeline.add(Node(Presence(0, missing, "answer")))
    var out = pipeline^.run()
    var col = out.column("answer").as_typed[DType.bool]()
    var flags = List[Bool](capacity=len(col))
    for i in range(len(col)):
        flags.append(col[i])
    return flags^


def counter() raises -> Group:
    """A group by on column 0 that counts the rows in each group."""
    var keys = List[Int]()
    keys.append(0)
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.COUNT, "rows"))
    return Group(keys^, aggs^)


def totals() raises -> Reduce:
    """A reduction over column 0 that asks for every kind that folds."""
    var aggs = List[GroupAgg]()
    aggs.append(GroupAgg(0, AggKind.SUM, "total"))
    aggs.append(GroupAgg(0, AggKind.MIN, "low"))
    aggs.append(GroupAgg(0, AggKind.MAX, "high"))
    aggs.append(GroupAgg(0, AggKind.COUNT, "seen"))
    return Reduce(aggs^)


def one_int(df: DataFrame, name: String) raises -> Int64:
    """Reads the single row of an int64 output column."""
    return df.column(name).as_typed[DType.int64]()[0]


def big_frame() raises -> DataFrame:
    """Six values near 1.9e18, in chunks of two, three and one.

    Their total is about 1.14e19 and int64 stops at 9.22e18, so a sum over this
    column wraps, and it is supposed to: pandas wraps there and firepanda
    follows it. What must not wrap is the sum a mean is divided out of, which is
    why the values are this size and why the chunks are uneven, so that the
    wrap has to survive the merge as well as the chunk.
    """
    var base = Int64(1_900_000_000_000_000_000)
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([base, base + 2]))
    n.append(numbers([base + 4, base + 6, base + 8]))
    n.append(numbers([base + 10]))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def kept_nothing() raises -> DataFrame:
    """Three rows behind a mask that keeps none of them."""
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2, 3]))
    var keep = ChunkedArray(LogicalType.BOOL)
    keep.append(flags([False, False, False]))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(keep^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.BOOL))
    return DataFrame(Schema(fields^), columns^)


def repeat_frame() raises -> DataFrame:
    """Six rows in chunks of two, three and one, with repeats across the joins.

    The 2 is in the first chunk and the second, and the 1 is in the first and
    the second as well, so a distinct count that added the chunks up would say
    eight where the answer is four. That is the case a held column exists for
    and a per chunk partial cannot see.
    """
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2]))
    n.append(numbers([2, 3, 1]))
    n.append(numbers([4]))
    var keep = ChunkedArray(LogicalType.BOOL)
    keep.append(flags([True, True]))
    keep.append(flags([True, True, True]))
    keep.append(flags([True]))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(keep^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.BOOL))
    return DataFrame(Schema(fields^), columns^)


def barren_frame() raises -> DataFrame:
    """The same six rows with a mask that keeps none of them.

    A filter over this hands the reduction below it no chunk with rows in it,
    which is the one input the two front ends answer differently about.
    """
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2]))
    n.append(numbers([3, 4, 5]))
    n.append(numbers([6]))
    var keep = ChunkedArray(LogicalType.BOOL)
    keep.append(flags([False, False]))
    keep.append(flags([False, False, False]))
    keep.append(flags([False]))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(keep^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("keep", LogicalType.BOOL))
    return DataFrame(Schema(fields^), columns^)


def nans(count: Int) raises -> AnyArray:
    """Builds a float64 array of NaNs, every one of them marked present."""
    var col = Array[DType.float64](count)
    for i in range(count):
        col.set_valid(i, nan[DType.float64]())
    return AnyArray(col^)


def hollow_frame() raises -> DataFrame:
    """Four rows in two chunks, one group of which holds no value at all.

    Group 1 holds a five and group 2 holds three nulls, so a sum over group 2 is
    a sum that saw rows and found nothing in them. That is the case the two
    front ends disagree about and it is not the case a filter that keeps no rows
    makes, because this group is there. `bare` is the same thing without a key,
    and `drift` is it spelled with NaN on a column the schema says cannot hold a
    null, which is the other way a value can fail to be there. See #170.

    The nulls are split across the two chunks on purpose. Each chunk is folded
    on its own and the partials are merged, so a group that held a value in one
    chunk and nothing in another has to come out the same as one that held it
    all at once.
    """
    var n = ChunkedArray(LogicalType.INT64)
    var first = Array[DType.int64](2)
    first.set_valid(0, Int64(5))
    first.set_null(1)
    n.append(AnyArray(first^))
    var second = Array[DType.int64](2)
    second.set_null(0)
    second.set_null(1)
    n.append(AnyArray(second^))

    var g = ChunkedArray(LogicalType.INT64)
    g.append(numbers([1, 2]))
    g.append(numbers([2, 2]))

    var bare = ChunkedArray(LogicalType.INT64)
    var here = Array[DType.int64](2)
    here.set_null(0)
    here.set_null(1)
    bare.append(AnyArray(here^))
    var there = Array[DType.int64](2)
    there.set_null(0)
    there.set_null(1)
    bare.append(AnyArray(there^))

    var drift = ChunkedArray(LogicalType.FLOAT64)
    drift.append(nans(2))
    drift.append(nans(2))

    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(g^)
    columns.append(bare^)
    columns.append(drift^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64, True))
    fields.append(Field("g", LogicalType.INT64, False))
    fields.append(Field("bare", LogicalType.INT64, True))
    fields.append(Field("drift", LogicalType.FLOAT64, False))
    return DataFrame(Schema(fields^), columns^)


def lookup_frame() raises -> DataFrame:
    """Four rows to join against, keyed on `n`.

    The keys are 2, 4, 6 and 8, so of the six rows in `cut_frame` three match
    and three do not, and the ones that do are not a prefix. Key 8 matches
    nothing on the other side, which is what an outer join would have to notice
    and this node does not.
    """
    var columns = List[AnyArray]()
    columns.append(numbers([2, 4, 6, 8]))
    columns.append(numbers([20, 40, 60, 80]))
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("tag", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def cut_lookup_frame() raises -> DataFrame:
    """The same four rows to join against, in chunks of one, two and one.

    A build side arrives in pieces as soon as it is a table read from several
    row groups or a derived table, and everything this node does indexes it by a
    single row number, so the pieces have to be stacked before the table is
    built. The key is split across two chunks and so is the payload, which is
    what a stack that dropped a piece or put them back in the wrong order would
    show up as.
    """
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([2]))
    n.append(numbers([4, 6]))
    n.append(numbers([8]))
    var tag = ChunkedArray(LogicalType.INT64)
    tag.append(numbers([20]))
    tag.append(numbers([40, 60]))
    tag.append(numbers([80]))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(tag^)
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("tag", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def key_words() raises -> AnyArray:
    """The six left keys, as text, for the joins on a text key.

    Three of them are past twelve bytes and three are not, which is the split
    that matters here and not anywhere else in this file. A view of twelve bytes
    or fewer holds its bytes inside itself and a longer one holds an offset into
    a payload, so a comparison against a long key has to reach into the column
    the table was built from and a comparison against a short one never does.
    Only the long ones would notice a probe reading its own payload instead.

    The keys are laid out so that the answer is the same set the integer joins in
    this file give, which is rows 2, 4 and 6. Two of those match on a long key
    and one on a short one, and of the three that match nothing, one is long.
    """
    return AnyArray(
        strings_from_list(
            [
                "k1",
                "a-key-well-past-twelve-2",
                "k3",
                "a-key-well-past-twelve-4",
                "a-key-well-past-twelve-5",
                "k6",
            ]
        )
    )


def key_fields() raises -> List[Field]:
    """The schema the text keyed frames share."""
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("key", LogicalType.STRING))
    return fields^


def word_key_frame() raises -> DataFrame:
    """The six rows in one chunk, keyed on text."""
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(key_words())
    return DataFrame(Schema(key_fields()), columns^)


def cut_word_key_frame() raises -> DataFrame:
    """The same six rows in chunks of two, three and one.

    The cuts fall where they fall in `cut_frame`, so the chunk that holds the
    long key that matches nothing is not the chunk that holds the short key that
    does.
    """
    var whole = key_words()
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2]))
    n.append(numbers([3, 4, 5]))
    n.append(numbers([6]))
    var key = ChunkedArray(LogicalType.STRING)
    key.append(AnyArray(whole.strings().slice(0, 2)))
    key.append(AnyArray(whole.strings().slice(2, 5)))
    key.append(AnyArray(whole.strings().slice(5, 6)))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(key^)
    return DataFrame(Schema(key_fields()), columns^)


def word_lookup_frame() raises -> DataFrame:
    """Four rows to join against, keyed on text.

    `lookup_frame` with the keys written out. The fourth is a long key that
    matches nothing on the other side, which is the row an outer join would have
    to notice and this node does not.
    """
    var columns = List[AnyArray]()
    columns.append(
        AnyArray(
            strings_from_list(
                [
                    "a-key-well-past-twelve-2",
                    "a-key-well-past-twelve-4",
                    "k6",
                    "a-key-well-past-twelve-8",
                ]
            )
        )
    )
    columns.append(numbers([20, 40, 60, 80]))
    var fields = List[Field]()
    fields.append(Field("key", LogicalType.STRING))
    fields.append(Field("tag", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def byte_lookup_frame() raises -> DataFrame:
    """A lookup frame whose `key` is a column of bytes rather than of text.

    A string column's physical dtype is uint8, so this is the frame that agrees
    with a text keyed one on everything a dtype comparison can see and means
    something completely different.
    """
    var col = Array[DType.uint8](4)
    for i in range(4):
        col.set_valid(i, UInt8(i + 1))
    var columns = List[AnyArray]()
    columns.append(AnyArray(col^))
    columns.append(numbers([20, 40, 60, 80]))
    var fields = List[Field]()
    fields.append(Field("key", LogicalType.UINT8))
    fields.append(Field("tag", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def joined_rows(var pipeline: Pipeline) raises -> List[Int64]:
    """Runs a pipeline and reads its `n` column back, sorted.

    A join emits a chunk per chunk and the driver may hand the chunks to
    different cores, so the order rows come back in is the order the chunks
    finished in. Every test here compares sets, and sorting is how a list is
    compared as a set.
    """
    var out = pipeline^.run()
    var values = read_back(out, "n")
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[j] < values[i]:
                var swap = values[i]
                values[i] = values[j]
                values[j] = swap
    return values^


def pair_fields() raises -> List[Field]:
    """The schema the two key probe frames share."""
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    fields.append(Field("a", LogicalType.INT64))
    fields.append(Field("b", LogicalType.STRING))
    return fields^


def pair_words() raises -> AnyArray:
    """The six second keys, as text.

    Two of them are past twelve bytes, for the reason `key_words` gives: a long
    view holds an offset into a payload rather than its own bytes, so it is the
    one a probe that read the wrong payload gets wrong. A packed key puts the
    length in front of the bytes and then the whole tuple is one element, which
    is well past twelve as soon as there is an eight byte key beside it, so the
    long and the short case both go through the payload here. The short ones are
    still worth having because they are what a single key join would have kept
    inside its view.
    """
    return AnyArray(
        strings_from_list(
            [
                "x",
                "x",
                "y",
                "a-key-well-past-twelve",
                "x",
                "y",
            ]
        )
    )


def pair_frame() raises -> DataFrame:
    """Six rows keyed on an integer and a string together, in one chunk."""
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(numbers([1, 2, 2, 3, 4, 5]))
    columns.append(pair_words())
    return DataFrame(Schema(pair_fields()), columns^)


def cut_pair_frame() raises -> DataFrame:
    """The same six rows in chunks of two, three and one.

    The two rows that pair land in different chunks, so a packing that is right
    for the first chunk and wrong for the rest shows up.
    """
    var whole = pair_words()
    var n = ChunkedArray(LogicalType.INT64)
    n.append(numbers([1, 2]))
    n.append(numbers([3, 4, 5]))
    n.append(numbers([6]))
    var a = ChunkedArray(LogicalType.INT64)
    a.append(numbers([1, 2]))
    a.append(numbers([2, 3, 4]))
    a.append(numbers([5]))
    var b = ChunkedArray(LogicalType.STRING)
    b.append(AnyArray(whole.strings().slice(0, 2)))
    b.append(AnyArray(whole.strings().slice(2, 5)))
    b.append(AnyArray(whole.strings().slice(5, 6)))
    var columns = List[ChunkedArray]()
    columns.append(n^)
    columns.append(a^)
    columns.append(b^)
    return DataFrame(Schema(pair_fields()), columns^)


def pair_lookup_frame() raises -> DataFrame:
    """Four rows to join against, keyed on the same pair.

    Laid out so that neither key on its own decides anything. Two build rows
    share the integer 2 and differ on the string, one shares the string "x" with
    a build row it does not share the integer with, and the last shares neither
    with anything that arrives. So a join that paired on the first key alone and
    a join that paired on the second key alone both answer something other than
    the two rows that really pair, which are probe rows 2 and 4.
    """
    var columns = List[AnyArray]()
    columns.append(numbers([2, 2, 3, 9]))
    columns.append(
        AnyArray(strings_from_list(["x", "z", "a-key-well-past-twelve", "x"]))
    )
    columns.append(numbers([20, 40, 60, 80]))
    var fields = List[Field]()
    fields.append(Field("a2", LogicalType.INT64))
    fields.append(Field("b2", LogicalType.STRING))
    fields.append(Field("tag", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def ints_of(col: AnyArray, rows: Int) raises -> List[Int64]:
    """Reads a whole int64 column out, for comparing against a literal list."""
    ref view = col.as_typed_view[DType.int64]()
    var out = List[Int64](capacity=rows)
    for i in range(rows):
        out.append(view[i])
    return out^


def selected_chunk() raises -> Chunk:
    """Six rows of values with a selection keeping the second and the fourth.

    Column 0 is the six values and is not dense, so its rows are at positions 1
    and 3. Column 1 is two values and is dense, which is what a column computed
    after the selection was made looks like. The two together are the only shape
    that matters, since a chunk whose columns are all one or all the other is
    handled by the same loop.
    """
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(numbers([70, 80]))
    var picks = List[UInt32]()
    picks.append(1)
    picks.append(3)
    var dense = List[Bool]()
    dense.append(False)
    dense.append(True)
    return Chunk(columns^, picks^, dense^)


def truths_of(col: AnyArray, rows: Int) raises -> List[Bool]:
    """Reads a whole boolean column out, the way `ints_of` reads an int64 one.
    """
    ref view = col.as_typed_view[DType.bool]()
    var out = List[Bool](capacity=rows)
    for i in range(rows):
        out.append(Bool(view[i]))
    return out^


def masked_chunk(keep: List[Bool]) raises -> Chunk:
    """A counting column and a mask over it, with no selection.

    Column 0 counts from one, so a value says which row of the input it came
    from and an answer can be read without keeping a second list around. Column
    1 is the mask, which is where every filter in the suite gets its predicate.
    """
    var values = List[Int64](capacity=len(keep))
    for i in range(len(keep)):
        values.append(Int64(i + 1))
    var columns = List[AnyArray]()
    columns.append(numbers(values))
    columns.append(flags(keep))
    return Chunk(columns^)


def selected_masked_chunk() raises -> Chunk:
    """Six values under a selection of three, with a dense mask over them.

    What a second filter in a pipeline is handed. Column 0 is the values and is
    read through the selection, so its rows are at positions 1, 3 and 5. Column
    1 is the mask a comparison wrote after the first filter ran, so it holds one
    value per row of the chunk rather than one per value underneath, and it
    keeps the first row and the third.
    """
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(flags([True, False, True]))
    var picks = List[UInt32]()
    picks.append(1)
    picks.append(3)
    picks.append(5)
    var dense = List[Bool]()
    dense.append(False)
    dense.append(True)
    return Chunk(columns^, picks^, dense^)


def two_masked_chunk(first: List[Bool], second: List[Bool]) raises -> Chunk:
    """A counting column and two masks over it, with no selection.

    A chain of two filters needs two predicates in the chunk before the first
    one runs, since the second mask is not computed anywhere in a test that
    calls the operators by hand.
    """
    var values = List[Int64](capacity=len(first))
    for i in range(len(first)):
        values.append(Int64(i + 1))
    var columns = List[AnyArray]()
    columns.append(numbers(values))
    columns.append(flags(first))
    columns.append(flags(second))
    return Chunk(columns^)


def two_under_a_selection() raises -> Chunk:
    """Two columns of six, neither dense, under a selection of three.

    What a chunk looks like between a filter and the first thing that reads a
    column. Both columns are still the arrays the scan handed over and the rows
    of the chunk are at positions 1, 3 and 5 of them.
    """
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(numbers([10, 20, 30, 40, 50, 60]))
    var picks = List[UInt32]()
    picks.append(1)
    picks.append(3)
    picks.append(5)
    var dense = List[Bool]()
    dense.append(False)
    dense.append(False)
    return Chunk(columns^, picks^, dense^)


def six_rows() raises -> Chunk:
    """Two columns of six rows, flat, with nothing missing."""
    var columns = List[AnyArray]()
    columns.append(numbers([1, 2, 3, 4, 5, 6]))
    columns.append(numbers([10, 20, 30, 40, 50, 60]))
    return Chunk(columns^)
