"""The frames and helpers the plan lower tests share.

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
from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.unary import UnaryOp
from firepanda.plan.bind import bind
from firepanda.plan.limits import limits
from firepanda.plan.lower import lower
from firepanda.plan.merge import merge
from firepanda.plan.node import NO_LIMIT, SET_EXCEPT, SET_INTERSECT, Plan
from firepanda.plan.simplify import simplify


def numbers(values: List[Int64]) raises -> AnyArray:
    """Builds a fully valid int64 array."""
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return AnyArray(col^)


def sales() raises -> DataFrame:
    """Ten rows in three chunks, with a quantity and a price.

    Three chunks rather than one because a chunk boundary is where an off by
    one in a position lives, and the quantities are not sorted so a filter that
    keeps a middle range is not a slice.
    """
    var qty = ChunkedArray(LogicalType.INT64)
    qty.append(numbers([5, 20, 3]))
    qty.append(numbers([40, 12, 8, 25]))
    qty.append(numbers([1, 30, 15]))
    var price = ChunkedArray(LogicalType.INT64)
    price.append(numbers([10, 2, 7]))
    price.append(numbers([1, 5, 9, 3]))
    price.append(numbers([100, 4, 6]))
    var columns = List[ChunkedArray]()
    columns.append(qty^)
    columns.append(price^)
    var fields = List[Field]()
    fields.append(Field("qty", LogicalType.INT64))
    fields.append(Field("price", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def one_frame() raises -> List[DataFrame]:
    """The sales frame as the single relation a plan can scan."""
    var frames = List[DataFrame]()
    frames.append(sales())
    return frames^


def schemas() raises -> List[Schema]:
    """The schema of that one relation, for binding."""
    var out = List[Schema]()
    out.append(Schema(copy=sales().schema))
    return out^


def shifts() raises -> DataFrame:
    """Nine rows in two chunks over three teams, for a window to partition.

    The teams are interleaved and they straddle the chunk boundary, so a window
    that reduced a chunk at a time or that let the rows move would give an
    answer that looks nearly right.
    """
    var team = ChunkedArray(LogicalType.INT64)
    team.append(numbers([1, 2, 1, 2]))
    team.append(numbers([3, 1, 3, 2, 1]))
    var hours = ChunkedArray(LogicalType.INT64)
    hours.append(numbers([4, 7, 2, 5]))
    hours.append(numbers([9, 1, 3, 8, 6]))
    var columns = List[ChunkedArray]()
    columns.append(team^)
    columns.append(hours^)
    var fields = List[Field]()
    fields.append(Field("team", LogicalType.INT64))
    fields.append(Field("hours", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def shift_frame() raises -> List[DataFrame]:
    """The shifts frame as the single relation a plan can scan."""
    var frames = List[DataFrame]()
    frames.append(shifts())
    return frames^


def shift_schemas() raises -> List[Schema]:
    """The schema of that one relation, for binding."""
    var out = List[Schema]()
    out.append(Schema(copy=shifts().schema))
    return out^


def gauges() raises -> DataFrame:
    """Five rows over two columns that hold nulls, for the three valued rule.

    The pairs cover every combination that makes a connective differ from what
    two valued logic would say: a null against a false, a null against a true,
    a false against a null, a null against a null, and a pair with no null in it
    at all to hold the ordinary case down.
    """
    var a = ChunkedArray(LogicalType.INT64)
    a.append(gappy([0, 0, 0, 0, 1], [0, 1, 3]))
    var b = ChunkedArray(LogicalType.INT64)
    b.append(gappy([0, 1, 0, 0, 1], [2, 3]))
    var columns = List[ChunkedArray]()
    columns.append(a^)
    columns.append(b^)
    var fields = List[Field]()
    fields.append(Field("a", LogicalType.INT64))
    fields.append(Field("b", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def gappy(values: List[Int64], nulls: List[Int]) raises -> AnyArray:
    """Builds an int64 array with nulls at the positions given."""
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    for i in range(len(nulls)):
        col.set_null(nulls[i])
    return AnyArray(col^)


def gauge_frame() raises -> List[DataFrame]:
    """The gauges frame as the single relation a plan can scan."""
    var frames = List[DataFrame]()
    frames.append(gauges())
    return frames^


def gauge_schemas() raises -> List[Schema]:
    """The schema of that one relation, for binding."""
    var out = List[Schema]()
    out.append(Schema(copy=gauges().schema))
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


def read_back(df: DataFrame, name: String) raises -> List[Int64]:
    """Reads an int64 column out as a plain list."""
    var col = df.column(name).as_typed[DType.int64]()
    var out = List[Int64](capacity=len(col))
    for i in range(len(col)):
        out.append(col[i])
    return out^


def decimals(df: DataFrame, name: String) raises -> List[Float64]:
    """Reads a float64 column out as a plain list."""
    var col = df.column(name).as_typed[DType.float64]()
    var out = List[Float64](capacity=len(col))
    for i in range(len(col)):
        out.append(col[i])
    return out^


def same(got: List[Int64], want: List[Int64], what: String) raises:
    """Checks a column read back against the numbers it should hold."""
    assert_equal(len(got), len(want), what + ": how many rows")
    for i in range(len(want)):
        assert_equal(got[i], want[i], what + " at " + String(i))


def valid(got: List[Bool], want: List[Bool], what: String) raises:
    """Checks which rows of a column have a value against which should."""
    assert_equal(len(got), len(want), what + ": how many rows")
    for i in range(len(want)):
        assert_equal(got[i], want[i], what + " at " + String(i))


def run(mut plan: Plan, root: Int) raises -> DataFrame:
    """Binds, lowers and runs a plan over the sales frame."""
    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())
    return pipe^.run()


def tiers() raises -> DataFrame:
    """Four bands and the rate each one charges.

    The names are disjoint from the sales frame's on purpose, so that a test
    joining the two can write either side's columns without qualifying them.
    Two frames that share a name are a case of their own and have a frame of
    their own below.
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


def echoes() raises -> DataFrame:
    """Four rows whose column names are the sales frame's, both of them.

    A join over this and the sales frame produces a schema with `qty` twice and
    `price` twice, which is what a plan says a join produces and used to be more
    than the operator could emit. One chunk, because the build side of a join is
    hashed as one column and a build side of three chunks is #583.
    """
    var qty = ChunkedArray(LogicalType.INT64)
    qty.append(numbers([3, 20, 40, 99]))
    var price = ChunkedArray(LogicalType.INT64)
    price.append(numbers([300, 200, 400, 900]))
    var columns = List[ChunkedArray]()
    columns.append(qty^)
    columns.append(price^)
    var fields = List[Field]()
    fields.append(Field("qty", LogicalType.INT64))
    fields.append(Field("price", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def copies() raises -> DataFrame:
    """Four bands with one of them written twice.

    For the set operations written `ALL`, which answer the same rows as the
    set answer on a frame whose rows all differ. A frame with a row on it twice
    is the only thing that tells the two apart.
    """
    var band = ChunkedArray(LogicalType.INT64)
    band.append(numbers([3, 3, 20, 77]))
    var columns = List[ChunkedArray]()
    columns.append(band^)
    var fields = List[Field]()
    fields.append(Field("band", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def crates() raises -> DataFrame:
    """Four rows keyed by a shop and a quantity together, the probe side."""
    var shop = ChunkedArray(LogicalType.INT64)
    shop.append(numbers([1, 2, 1, 2]))
    var qty = ChunkedArray(LogicalType.INT64)
    qty.append(numbers([5, 20, 3, 40]))
    var columns = List[ChunkedArray]()
    columns.append(shop^)
    columns.append(qty^)
    var fields = List[Field]()
    fields.append(Field("shop", LogicalType.INT64))
    fields.append(Field("qty", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def crated() raises -> DataFrame:
    """Three rows keyed the same way, the build side.

    The names are the crates frame's names spelt differently, so that a test
    can say which side a key came from without qualifying anything. Its second
    row is a shop the crates frame has and a quantity the crates frame has in a
    different row, so a join on the shop alone pairs it and a join on both
    keys must not.
    """
    var place = ChunkedArray(LogicalType.INT64)
    place.append(numbers([1, 2, 1]))
    var many = ChunkedArray(LogicalType.INT64)
    many.append(numbers([5, 40, 3]))
    var kept = ChunkedArray(LogicalType.INT64)
    kept.append(numbers([100, 200, 300]))
    var columns = List[ChunkedArray]()
    columns.append(place^)
    columns.append(many^)
    columns.append(kept^)
    var fields = List[Field]()
    fields.append(Field("place", LogicalType.INT64))
    fields.append(Field("many", LogicalType.INT64))
    fields.append(Field("kept", LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def crate_frames() raises -> List[DataFrame]:
    """The crates frame as relation zero and the crated one as relation one."""
    var frames = List[DataFrame]()
    frames.append(crates())
    frames.append(crated())
    return frames^


def crate_schemas() raises -> List[Schema]:
    """The schema of each of those two, for binding."""
    var out = List[Schema]()
    out.append(Schema(copy=crates().schema))
    out.append(Schema(copy=crated().schema))
    return out^


def two_frames() raises -> List[DataFrame]:
    """The sales frame as relation zero and the tiers frame as relation one."""
    var frames = List[DataFrame]()
    frames.append(sales())
    frames.append(tiers())
    return frames^


def two_schemas() raises -> List[Schema]:
    """The schema of each of those two, for binding."""
    var out = List[Schema]()
    out.append(Schema(copy=sales().schema))
    out.append(Schema(copy=tiers().schema))
    return out^


def run_two(mut plan: Plan, root: Int) raises -> DataFrame:
    """Binds, lowers and runs a plan over both frames."""
    _ = bind(plan, root, two_schemas())
    var pipe = lower(plan, root, two_frames())
    return pipe^.run()


def run_pair(
    mut plan: Plan, root: Int, var left: DataFrame, var right: DataFrame
) raises -> DataFrame:
    """Binds, lowers and runs a plan over two frames the caller built.

    `run_frames` below takes the frames and needs the schemas separately, and
    a set operation's two arms are two relations, so this is the same call with
    the schemas read off the frames rather than handed in twice.
    """
    var schemas = List[Schema]()
    schemas.append(Schema(copy=left.schema))
    schemas.append(Schema(copy=right.schema))
    _ = bind(plan, root, schemas)
    var frames = List[DataFrame]()
    frames.append(left^)
    frames.append(right^)
    var pipe = lower(plan, root, frames^)
    return pipe^.run()


def run_frames(
    mut plan: Plan, root: Int, var frames: List[DataFrame]
) raises -> DataFrame:
    """Binds, lowers and runs a plan over frames the caller built.

    For the tests that need a fixture with a null in it, which is a frame of
    their own rather than something the shared ones should carry.
    """
    var fields = List[Schema]()
    for i in range(len(frames)):
        fields.append(Schema(copy=frames[i].schema))
    _ = bind(plan, root, fields)
    var pipe = lower(plan, root, frames^)
    return pipe^.run()


def holey(
    name: String, values: List[Int64], nulls: List[Int]
) raises -> DataFrame:
    """One int64 column of that name with nulls at the positions given."""
    var col = ChunkedArray(LogicalType.INT64)
    col.append(gappy(values, nulls))
    var columns = List[ChunkedArray]()
    columns.append(col^)
    var fields = List[Field]()
    fields.append(Field(name, LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def present(df: DataFrame, name: String) raises -> List[Bool]:
    """Which rows of an int64 column have a value in them."""
    var col = df.column(name).as_typed[DType.int64]()
    var out = List[Bool](capacity=len(col))
    for i in range(len(col)):
        out.append(col.is_valid(i))
    return out^


def joined(
    mut plan: Plan, kind: JoinKind, left_names: List[String] = List[String]()
) raises -> Int:
    """A join of sales to tiers on the quantity and the band.

    Args:
        plan: Where the nodes go.
        kind: Which rows to keep.
        left_names: The columns the left scan reads, or empty for all of them.

    Returns:
        The join node.
    """
    var left = plan.scan("sales", left_names.copy(), 0)
    var right = plan.scan("tiers", List[String](), 1)
    return plan.join(
        left,
        right,
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        kind,
    )


def emptied(
    mut plan: Plan, kind: JoinKind, mark: String = String()
) raises -> Int:
    """A join of sales onto a tiers that a filter has left nothing in."""
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var none = plan.exprs.binary(
        BinaryOp.GT,
        plan.exprs.column("band"),
        plan.exprs.literal(Value(Int64(1000))),
    )
    return plan.join(
        left,
        plan.filter(right, none),
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        kind,
        mark,
    )


def series(mut plan: Plan, root: Int) raises -> DataFrame:
    """Binds, lowers and runs a plan that reads no relation at all."""
    _ = bind(plan, root, List[Schema]())
    var pipe = lower(plan, root, List[DataFrame]())
    return pipe^.run()


def counted(mut plan: Plan, name: String, args: List[Int64]) raises -> Int:
    """A call to one of the two series functions over whole number arguments."""
    var written = List[Int](capacity=len(args))
    for i in range(len(args)):
        written.append(plan.exprs.literal(Value(args[i])))
    return plan.table_function(name, written^, ["i"])


def crate_join(mut plan: Plan, kind: JoinKind) raises -> Int:
    """A join of the crates frame to the crated one on both of their keys."""
    var left = plan.scan("crates", List[String](), 0)
    var right = plan.scan("crated", List[String](), 1)
    return plan.join(
        left,
        right,
        [plan.exprs.column("shop"), plan.exprs.column("qty")],
        [plan.exprs.column("place"), plan.exprs.column("many")],
        kind,
    )
