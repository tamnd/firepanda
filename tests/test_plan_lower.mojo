"""Tests for turning a logical plan into a pipeline.

Every test here builds a plan, binds it, lowers it and runs it, and checks the
rows that come out. Checking the rows rather than the shape of the pipeline is
deliberate: the number of operators a plan lowers to is an implementation
detail that a later pass is allowed to change, and the rows are not. The two
tests that do look at the operator count are the ones where the count is the
point, which is the conjunction becoming a line of filters.

The other half is the refusals. Lowering is allowed to say no, and what it says
no to is the list of things nobody has written an operator for yet, so a test
per refusal is what stops one of them being quietly half lowered into a wrong
answer later.
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


def test_a_bare_scan_gives_the_frame_back() raises:
    var plan = Plan()
    var root = plan.scan("sales", List[String](), 0)
    var out = run(plan, root)

    assert_equal(out.width(), 2, "both columns")
    assert_equal(len(out), 10, "every row")
    var got = read_back(out, "qty")
    assert_equal(got[0], 5, "first")
    assert_equal(got[9], 15, "last")


def test_a_scan_that_names_one_column_reads_one_column() raises:
    var plan = Plan()
    var root = plan.scan("sales", ["price"], 0)
    var out = run(plan, root)

    assert_equal(out.width(), 1, "one column")
    assert_equal(out.schema[0].name, "price", "and it is the named one")
    assert_equal(len(out), 10, "every row")


def test_a_scan_reads_its_columns_in_the_order_it_names_them() raises:
    var plan = Plan()
    var root = plan.scan("sales", ["price", "qty"], 0)
    var out = run(plan, root)

    assert_equal(out.schema[0].name, "price", "first")
    assert_equal(out.schema[1].name, "qty", "second")


def test_a_filter_against_a_constant_keeps_the_rows_it_should() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var ten = plan.exprs.literal(Value(Int64(10)))
    var root = plan.filter(scan, plan.exprs.binary(BinaryOp.GT, qty, ten))
    var out = run(plan, root)

    # 20, 40, 12, 25, 30 and 15 are over ten, in that order.
    var got = read_back(out, "qty")
    assert_equal(len(got), 6, "rows kept")
    assert_equal(got[0], 20, "first")
    assert_equal(got[5], 15, "last")


def test_the_mask_a_filter_computed_does_not_reach_the_output() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var ten = plan.exprs.literal(Value(Int64(10)))
    var root = plan.filter(scan, plan.exprs.binary(BinaryOp.GT, qty, ten))
    var out = run(plan, root)

    # The filter reads its mask by position, so lowering appended one, and the
    # node's output schema is what binding said it was, which has no mask in it.
    assert_equal(out.width(), 2, "the two input columns and nothing else")
    assert_equal(out.schema[0].name, "qty", "first")
    assert_equal(out.schema[1].name, "price", "second")


def test_a_filter_between_two_columns_reads_both() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var root = plan.filter(scan, plan.exprs.binary(BinaryOp.GT, qty, price))
    var out = run(plan, root)

    # qty over price on 20/2, 3/... no: 5/10 no, 20/2 yes, 3/7 no, 40/1 yes,
    # 12/5 yes, 8/9 no, 25/3 yes, 1/100 no, 30/4 yes, 15/6 yes.
    var got = read_back(out, "qty")
    assert_equal(len(got), 6, "rows kept")
    assert_equal(got[0], 20, "first")
    assert_equal(got[5], 15, "last")


def test_a_filter_over_an_expression_computes_the_expression_first() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var total = plan.exprs.binary(BinaryOp.MUL, qty, price)
    var hundred = plan.exprs.literal(Value(Int64(100)))
    var root = plan.filter(scan, plan.exprs.binary(BinaryOp.GE, total, hundred))
    var out = run(plan, root)

    # The products are 50, 40, 21, 40, 60, 72, 75, 100, 120 and 90, so two
    # reach a hundred and the one that gets closest without doing so is 90.
    var got = read_back(out, "qty")
    assert_equal(len(got), 2, "rows kept")
    assert_equal(got[0], 1, "the row whose product is exactly a hundred")
    assert_equal(got[1], 30, "the row whose product is a hundred and twenty")


def test_a_conjunction_becomes_one_filter_per_part() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var low = plan.exprs.binary(
        BinaryOp.GE, qty, plan.exprs.literal(Value(Int64(10)))
    )
    var high = plan.exprs.binary(
        BinaryOp.LE, qty, plan.exprs.literal(Value(Int64(25)))
    )
    var both = plan.exprs.call(String("and"), [low, high], rowwise=True)
    var root = plan.filter(scan, both)

    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())

    # A compare and a filter for each half, and no projection afterwards,
    # because each filter drops the mask it just spent as it writes. Computing
    # an and mask would have been four operators too, but both comparisons
    # would have run on all ten rows.
    assert_equal(len(pipe.operators), 4, "operators")

    var out = pipe^.run()
    var got = read_back(out, "qty")
    assert_equal(len(got), 4, "rows kept")
    assert_equal(got[0], 20, "first")
    assert_equal(got[3], 15, "last")


def test_a_nested_conjunction_flattens_into_the_same_line() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var a = plan.exprs.binary(
        BinaryOp.GE, qty, plan.exprs.literal(Value(Int64(10)))
    )
    var b = plan.exprs.binary(
        BinaryOp.LE, qty, plan.exprs.literal(Value(Int64(25)))
    )
    var c = plan.exprs.binary(
        BinaryOp.GT, price, plan.exprs.literal(Value(Int64(3)))
    )
    var inner = plan.exprs.call(String("and"), [b, c], rowwise=True)
    var outer = plan.exprs.call(String("and"), [a, inner], rowwise=True)
    var root = plan.filter(scan, outer)

    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())
    assert_equal(len(pipe.operators), 6, "three compares and three filters")

    var out = pipe^.run()
    # Of 20, 12, 25 and 15, the prices are 2, 5, 3 and 6, so two survive.
    var got = read_back(out, "qty")
    assert_equal(len(got), 2, "rows kept")
    assert_equal(got[0], 12, "first")
    assert_equal(got[1], 15, "last")


def test_a_conjunction_the_simplify_pass_flattened_lowers_the_same() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var a = plan.exprs.binary(
        BinaryOp.GE, qty, plan.exprs.literal(Value(Int64(10)))
    )
    var yes = plan.exprs.literal(Value(True))
    var both = plan.exprs.call(String("and"), [a, yes], rowwise=True)
    var root = plan.filter(scan, both)

    _ = bind(plan, root, schemas())
    simplify(plan, root)
    var pipe = lower(plan, root, one_frame())

    # The literal true dropped out in the pass, so what is left is one
    # comparison and one filter, which puts the schema back itself.
    assert_equal(len(pipe.operators), 2, "operators")
    var out = pipe^.run()
    assert_equal(len(out), 6, "rows kept")


def test_a_disjunction_computes_a_mask_and_filters_on_it() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var three = plan.exprs.binary(
        BinaryOp.EQ, qty, plan.exprs.literal(Value(Int64(3)))
    )
    var lots = plan.exprs.binary(
        BinaryOp.EQ, qty, plan.exprs.literal(Value(Int64(25)))
    )
    var either = plan.exprs.call(String("or"), [three, lots], rowwise=True)
    var root = plan.filter(scan, either)
    var out = run(plan, root)

    # The other half of the conjunction test above. This one cannot become a
    # line of filters, because each arm keeps rows the other one drops, so both
    # comparisons run on all ten rows and something has to join them.
    same(read_back(out, "qty"), [3, 25], "the two rows named")


def test_a_negation_in_a_filter_keeps_what_the_predicate_dropped() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var lots = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(10)))
    )
    var root = plan.filter(
        scan, plan.exprs.call(String("not"), [lots], rowwise=True)
    )
    var out = run(plan, root)

    same(read_back(out, "qty"), [5, 3, 8, 1], "the rows at ten or under")


def test_a_chain_of_three_ors_is_two_operators_folded_left() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var a = plan.exprs.binary(
        BinaryOp.EQ, qty, plan.exprs.literal(Value(Int64(3)))
    )
    var b = plan.exprs.binary(
        BinaryOp.EQ, qty, plan.exprs.literal(Value(Int64(25)))
    )
    var c = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(29)))
    )
    var any = plan.exprs.call(String("or"), [a, b, c], rowwise=True)
    var root = plan.filter(scan, any)

    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())

    # A call with three arguments is not a node that takes three columns. Three
    # comparisons, two connectives over the pairs, and the filter, which drops
    # all five intermediates as it writes the way it drops one.
    assert_equal(len(pipe.operators), 6, "operators")

    var out = pipe^.run()
    same(read_back(out, "qty"), [3, 40, 25, 30], "the rows any arm names")


def test_a_conjunction_below_a_disjunction_reaches_the_operator() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var one = plan.exprs.binary(
        BinaryOp.EQ, qty, plan.exprs.literal(Value(Int64(1)))
    )
    var lots = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(10)))
    )
    var cheap = plan.exprs.binary(
        BinaryOp.LT, price, plan.exprs.literal(Value(Int64(5)))
    )
    var both = plan.exprs.call(String("and"), [lots, cheap], rowwise=True)
    var either = plan.exprs.call(String("or"), [one, both], rowwise=True)
    var root = plan.filter(scan, either)
    var out = run(plan, root)

    # The and is not at the top of the filter, so it is not a line of filters
    # and it is computed as a column like the or above it. Of the six rows over
    # ten, the prices are 2, 1, 5, 3, 4 and 6, so four are under five, and the
    # single row the other arm names is on top of those.
    same(
        read_back(out, "qty"), [20, 40, 25, 1, 30], "the rows either arm names"
    )


def test_a_disjunction_over_nulls_follows_the_three_valued_rule() raises:
    var plan = Plan()
    var scan = plan.scan("gauges", List[String](), 0)
    var zero = plan.exprs.literal(Value(Int64(0)))
    var a = plan.exprs.binary(BinaryOp.GT, plan.exprs.column("a"), zero)
    var b = plan.exprs.binary(BinaryOp.GT, plan.exprs.column("b"), zero)
    var either = plan.exprs.call(String("or"), [a, b], rowwise=True)
    var both = plan.exprs.call(String("and"), [a, b], rowwise=True)
    var root = plan.project(scan, [either, both], ["either", "both"])

    _ = bind(plan, root, gauge_schemas())
    var pipe = lower(plan, root, gauge_frame())
    var out = pipe^.run()

    # A true settles the or and a false settles the and, and the rows where
    # neither operand settles anything stay null.
    same(truths(out, "either"), [-1, 1, -1, -1, 1], "either")
    same(truths(out, "both"), [0, -1, 0, -1, 1], "both")


def test_a_filter_drops_the_rows_a_connective_could_not_decide() raises:
    var plan = Plan()
    var scan = plan.scan("gauges", List[String](), 0)
    var zero = plan.exprs.literal(Value(Int64(0)))
    var a = plan.exprs.binary(BinaryOp.GT, plan.exprs.column("a"), zero)
    var b = plan.exprs.binary(BinaryOp.GT, plan.exprs.column("b"), zero)
    var either = plan.exprs.call(String("or"), [a, b], rowwise=True)
    var root = plan.filter(scan, either)

    _ = bind(plan, root, gauge_schemas())
    var pipe = lower(plan, root, gauge_frame())
    var out = pipe^.run()

    # A predicate has to be true to keep a row, and the three null rows are not
    # true, so two rows come out of five.
    assert_equal(len(out), 2, "rows kept")
    same(read_back(out, "b"), [1, 1], "the two rows that were true")


def test_a_connective_over_a_column_that_is_not_boolean_never_lowers() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var either = plan.exprs.call(
        String("or"),
        [plan.exprs.column("qty"), plan.exprs.column("price")],
        rowwise=True,
    )
    var root = plan.project(scan, [either], ["either"])

    # Binding refuses it, so lowering never sees it. The operator checks the
    # same thing again in `bind` and so does the kernel, because a plan that
    # was built by hand rather than by the binder can still reach either one.
    with assert_raises(contains="reads yes or no"):
        _ = bind(plan, root, schemas())


def test_a_negation_of_more_than_one_thing_never_lowers() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var a = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(10)))
    )
    var b = plan.exprs.binary(
        BinaryOp.LT, qty, plan.exprs.literal(Value(Int64(30)))
    )
    var root = plan.filter(
        scan, plan.exprs.call(String("not"), [a, b], rowwise=True)
    )

    with assert_raises(contains="takes 1 argument and was given 2"):
        _ = bind(plan, root, schemas())


def test_a_projection_selects_by_position() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var price = plan.exprs.column("price")
    var root = plan.project(scan, [price], ["price"])
    var out = run(plan, root)

    assert_equal(out.width(), 1, "one column")
    assert_equal(out.schema[0].name, "price", "the one asked for")
    var got = read_back(out, "price")
    assert_equal(got[0], 10, "first")


def test_a_projection_computes_the_column_it_was_asked_for() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var total = plan.exprs.binary(BinaryOp.MUL, qty, price)
    var root = plan.project(scan, [qty, total], ["qty", "total"])
    var out = run(plan, root)

    assert_equal(out.width(), 2, "the key and the computed column")
    assert_equal(out.schema[1].name, "total", "named by the projection")
    var got = read_back(out, "total")
    assert_equal(got[0], 50, "five times ten")
    assert_equal(got[7], 100, "one times a hundred")


def test_a_projection_can_name_the_same_column_twice() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var root = plan.project(scan, [qty, qty], ["qty", "qty"])
    var out = run(plan, root)

    assert_equal(out.width(), 2, "twice over")
    assert_equal(len(out), 10, "every row")


def test_a_filter_under_a_projection_runs_first() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var ten = plan.exprs.literal(Value(Int64(10)))
    var kept = plan.filter(scan, plan.exprs.binary(BinaryOp.GT, qty, ten))
    var root = plan.project(kept, [qty], ["qty"])
    var out = run(plan, root)

    assert_equal(out.width(), 1, "one column")
    var got = read_back(out, "qty")
    assert_equal(len(got), 6, "rows kept")
    assert_equal(got[0], 20, "first")


def test_a_limit_takes_the_first_rows() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var root = plan.limit(scan, 0, 4)
    var out = run(plan, root)

    var got = read_back(out, "qty")
    assert_equal(len(got), 4, "rows")
    assert_equal(got[3], 40, "the fourth row of the frame")


def test_a_limit_that_keeps_everything_adds_no_operator() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var root = plan.limit(scan, 0, NO_LIMIT)

    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())
    assert_equal(len(pipe.operators), 0, "nothing to do")


def test_a_limit_over_a_filter_counts_what_survived() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var ten = plan.exprs.literal(Value(Int64(10)))
    var kept = plan.filter(scan, plan.exprs.binary(BinaryOp.GT, qty, ten))
    var root = plan.limit(kept, 0, 2)
    var out = run(plan, root)

    var got = read_back(out, "qty")
    assert_equal(len(got), 2, "rows")
    assert_equal(got[0], 20, "first over ten")
    assert_equal(got[1], 40, "second over ten")


def test_a_constant_on_the_left_keeps_its_side() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var hundred = plan.exprs.literal(Value(Int64(100)))
    # 100 - qty, which is not qty - 100, and the point of the flag.
    var left = plan.exprs.binary(BinaryOp.SUB, hundred, qty)
    var root = plan.project(scan, [left], ["rest"])
    var out = run(plan, root)

    var got = read_back(out, "rest")
    assert_equal(got[0], 95, "a hundred less five")
    assert_equal(got[3], 60, "a hundred less forty")


def test_an_unbound_plan_says_so_rather_than_reading_position_zero() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var ten = plan.exprs.literal(Value(Int64(10)))
    var root = plan.filter(scan, plan.exprs.binary(BinaryOp.GT, qty, ten))

    with assert_raises(contains="was not bound"):
        _ = lower(plan, root, one_frame())


def test_a_scan_with_no_frame_for_its_relation_says_so() raises:
    var plan = Plan()
    var root = plan.scan("sales", List[String](), 1)
    var two = List[Schema]()
    two.append(Schema(copy=sales().schema))
    two.append(Schema(copy=sales().schema))
    _ = bind(plan, root, two^)

    with assert_raises(contains="it was given 1 frames"):
        _ = lower(plan, root, one_frame())


def test_a_whole_frame_reduction_answers_one_row() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var total = plan.exprs.aggregate(AggKind.SUM, qty)
    var root = plan.aggregate(scan, List[Int](), [total], ["total"])
    var out = run(plan, root)

    assert_equal(out.width(), 1, "one column")
    assert_equal(len(out), 1, "one row")
    var got = read_back(out, "total")
    assert_equal(got[0], 159, "five and twenty and the other eight")


def test_a_reduction_over_an_expression_folds_the_expression() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var line = plan.exprs.binary(BinaryOp.MUL, qty, price)
    var revenue = plan.exprs.aggregate(AggKind.SUM, line)
    var root = plan.aggregate(scan, List[Int](), [revenue], ["revenue"])
    var out = run(plan, root)

    # 50, 40, 21, 40, 60, 72, 75, 100, 120 and 90.
    var got = read_back(out, "revenue")
    assert_equal(got[0], 668, "the sum of the products")


def test_a_whole_frame_reduction_over_a_constant_folds_it_into_the_fold() raises:
    # The count is the point here, against the file's usual rule. A `Compute`
    # in front of the reduction would hold a whole column of its own for as
    # long as the chunk lives, and a query with ninety of these would hold
    # ninety, so whether one is added is the thing the change was made about.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var wider = plan.exprs.binary(
        BinaryOp.ADD, qty, plan.exprs.literal(Value(Int64(1)))
    )
    var total = plan.exprs.aggregate(AggKind.SUM, wider)
    var root = plan.aggregate(scan, List[Int](), [total], ["total"])
    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())

    assert_equal(len(pipe.operators), 1, "the reduction, and no add in front")

    var out = pipe^.run()
    var got = read_back(out, "total")
    assert_equal(got[0], 169, "the sum of ten quantities and ten ones")


def test_a_folded_constant_may_sit_on_either_side() raises:
    # Subtraction is the one that tells the two apart, and getting the side
    # wrong here would give the negation of the right answer, which is exactly
    # the kind of wrong that reads as right.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var left = plan.exprs.binary(
        BinaryOp.SUB, plan.exprs.literal(Value(Int64(100))), qty
    )
    var right = plan.exprs.binary(
        BinaryOp.SUB, qty, plan.exprs.literal(Value(Int64(100)))
    )
    var root = plan.aggregate(
        scan,
        List[Int](),
        [
            plan.exprs.aggregate(AggKind.SUM, left),
            plan.exprs.aggregate(AggKind.SUM, right),
        ],
        ["down", "up"],
    )
    var out = run(plan, root)

    assert_equal(read_back(out, "down")[0], 841, "a thousand less the sum")
    assert_equal(read_back(out, "up")[0], -841, "the sum less a thousand")


def test_a_folded_operation_answers_what_the_long_way_answers() raises:
    # The same query lowered both ways. The fused one reads the column and the
    # operand off the aggregate, the other one reads the column a `Compute`
    # wrote, and if the two ever disagree one of the two paths has a bug in it.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var over = plan.exprs.binary(
        BinaryOp.MUL, qty, plan.exprs.literal(Value(Int64(3)))
    )
    var root = plan.aggregate(
        scan,
        List[Int](),
        [
            plan.exprs.aggregate(AggKind.SUM, over),
            plan.exprs.aggregate(AggKind.MIN, over),
            plan.exprs.aggregate(AggKind.MAX, over),
            plan.exprs.aggregate(AggKind.MEAN, over),
            plan.exprs.aggregate(AggKind.COUNT, over),
        ],
        ["total", "least", "most", "middle", "rows"],
    )
    var out = run(plan, root)

    assert_equal(read_back(out, "total")[0], 477, "three times the sum")
    assert_equal(read_back(out, "least")[0], 3, "three times the smallest")
    assert_equal(read_back(out, "most")[0], 120, "three times the largest")
    assert_equal(decimals(out, "middle")[0], 47.7, "three times the mean")
    assert_equal(read_back(out, "rows")[0], 10, "every row is still counted")


def test_a_folded_operation_reaches_a_reduction_that_holds_its_column() raises:
    # A distinct count does not fold, so its column is kept whole and the
    # operation has to run on the way in rather than on the way through. The
    # two halves of `Reduce` take different routes to it and both are wrong in
    # their own way if only one of them was wired up.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var tens = plan.exprs.binary(
        BinaryOp.FLOORDIV, qty, plan.exprs.literal(Value(Int64(10)))
    )
    var root = plan.aggregate(
        scan, List[Int](), [plan.exprs.aggregate(AggKind.NUNIQUE, tens)], ["n"]
    )
    var out = run(plan, root)

    # The tens are 0, 2, 0, 4, 1, 0, 2, 0, 3 and 1, so five of them are distinct.
    assert_equal(read_back(out, "n")[0], 5, "distinct tens")


def test_a_folded_operation_promotes_the_type_the_way_a_compute_does() raises:
    # The declared dtype comes from the same two calls `Compute.bind` makes, so
    # an integer column against a decimal constant sums as a decimal here for
    # the same reason it would have there.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var half = plan.exprs.binary(
        BinaryOp.MUL, qty, plan.exprs.literal(Value(Float64(0.5)))
    )
    var root = plan.aggregate(
        scan, List[Int](), [plan.exprs.aggregate(AggKind.SUM, half)], ["half"]
    )
    var out = run(plan, root)

    assert_equal(
        out.schema[0].dtype, LogicalType.FLOAT64, "the constant widened it"
    )
    assert_equal(decimals(out, "half")[0], 79.5, "half of the total")


def test_a_group_by_over_a_constant_still_computes_the_column() raises:
    # A group by cannot fold the operation in, because its rows scatter into
    # groups and the operation would have to scatter with them. It gets the
    # `Compute` it always got, and the answer is the same either way.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var wider = plan.exprs.binary(
        BinaryOp.ADD, qty, plan.exprs.literal(Value(Int64(1)))
    )
    var root = plan.aggregate(
        scan,
        [price],
        [plan.exprs.aggregate(AggKind.SUM, wider)],
        ["price", "total"],
    )
    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())

    assert_equal(len(pipe.operators), 2, "the add and the group by")

    var out = pipe^.run()
    assert_equal(len(out), 10, "one row per price")
    var totals = read_back(out, "total")
    assert_equal(totals[0], 6, "the first quantity and one")


def test_a_group_by_keeps_the_key_and_names_the_fold() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var price = plan.exprs.column("price")
    var qty = plan.exprs.column("qty")
    var total = plan.exprs.aggregate(AggKind.SUM, qty)
    var root = plan.aggregate(scan, [price], [total], ["price", "total"])
    var out = run(plan, root)

    # Every price is distinct, so the grouping is the rows in the order they
    # were first seen, which is what makes the check readable.
    assert_equal(out.width(), 2, "the key and the fold")
    assert_equal(out.schema[0].name, "price", "the key keeps its name")
    assert_equal(out.schema[1].name, "total", "the fold takes the plan's")
    assert_equal(len(out), 10, "one row per price")


def test_a_group_by_on_a_computed_key_groups_by_what_it_computed() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var bucket = plan.exprs.binary(
        BinaryOp.FLOORDIV, qty, plan.exprs.literal(Value(Int64(10)))
    )
    var counted = plan.exprs.aggregate(AggKind.COUNT, qty)
    var root = plan.aggregate(scan, [bucket], [counted], ["tens", "rows"])
    var out = run(plan, root)

    # 5, 20, 3, 40, 12, 8, 25, 1, 30 and 15 fall in tens 0, 2, 0, 4, 1, 0, 2, 0,
    # 3 and 1, so the buckets first seen are 0, 2, 4, 1 and 3.
    assert_equal(len(out), 5, "buckets")
    var tens = read_back(out, "tens")
    assert_equal(tens[0], 0, "the first bucket seen")
    assert_equal(tens[1], 2, "the second")
    var rows = read_back(out, "rows")
    assert_equal(rows[0], 4, "four rows under ten")


def test_the_shape_of_q6_lowers_and_runs() raises:
    # Scan, a conjunction of three range predicates, a product, and one sum,
    # which is the whole of TPC-H q6 with the column names changed.
    var plan = Plan()
    var scan = plan.scan("sales", ["qty", "price"], 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var low = plan.exprs.binary(
        BinaryOp.GE, price, plan.exprs.literal(Value(Int64(2)))
    )
    var high = plan.exprs.binary(
        BinaryOp.LE, price, plan.exprs.literal(Value(Int64(9)))
    )
    var small = plan.exprs.binary(
        BinaryOp.LT, qty, plan.exprs.literal(Value(Int64(25)))
    )
    var all_of = plan.exprs.call(
        String("and"), [low, high, small], rowwise=True
    )
    var kept = plan.filter(scan, all_of)
    var line = plan.exprs.binary(BinaryOp.MUL, qty, price)
    var revenue = plan.exprs.aggregate(AggKind.SUM, line)
    var root = plan.aggregate(kept, List[Int](), [revenue], ["revenue"])

    _ = bind(plan, root, schemas())
    simplify(plan, root)
    var pipe = lower(plan, root, one_frame())
    var out = pipe^.run()

    # Prices between two and nine with a quantity under twenty five keep
    # 20 at 2, 3 at 7, 12 at 5, 8 at 9, and 15 at 6, for 40, 21, 60, 72 and 90.
    assert_equal(len(out), 1, "one row")
    var got = read_back(out, "revenue")
    assert_equal(got[0], 283, "the revenue")


def test_an_aggregation_whose_output_is_not_a_fold_is_refused() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var root = plan.aggregate(scan, [price], [qty], ["price", "qty"])
    _ = bind(plan, root, schemas())

    with assert_raises(contains="rather than a fold"):
        _ = lower(plan, root, one_frame())


def test_a_group_by_that_renames_its_key_is_refused() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var price = plan.exprs.column("price")
    var qty = plan.exprs.column("qty")
    var total = plan.exprs.aggregate(AggKind.SUM, qty)
    var root = plan.aggregate(scan, [price], [total], ["each", "total"])
    _ = bind(plan, root, schemas())

    with assert_raises(contains="carries the key field through"):
        _ = lower(plan, root, one_frame())


def test_a_reduction_that_cannot_fold_a_chunk_at_a_time_still_lowers() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var middle = plan.exprs.aggregate(AggKind.MEDIAN, qty)
    var root = plan.aggregate(scan, List[Int](), [middle], ["middle"])
    _ = bind(plan, root, schemas())

    # It used to be refused here. `Reduce` holds the column for a reduction
    # whose state is the values themselves, so lowering has nothing to say
    # about it and what folds stays a property of the reduction.
    var pipe = lower(plan, root, one_frame())
    var out = pipe^.run()
    assert_equal(len(out), 1, "one row")


def test_a_sort_puts_the_rows_in_order() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var root = plan.sort(scan, [qty], [False], [True])
    var out = run(plan, root)

    same(
        read_back(out, "qty"),
        [1, 3, 5, 8, 12, 15, 20, 25, 30, 40],
        "ascending",
    )


def test_a_sort_carries_the_other_columns_with_the_row() raises:
    # The point of sorting a frame rather than a column. A price that stayed
    # where it was would be a wrong answer that reads like a right one.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var root = plan.sort(scan, [qty], [False], [True])
    var out = run(plan, root)

    same(
        read_back(out, "price"),
        [100, 7, 10, 9, 5, 6, 2, 3, 4, 1],
        "the price of each row",
    )


def test_a_sort_descending_reverses_it() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var root = plan.sort(scan, [qty], [True], [True])
    var out = run(plan, root)

    same(
        read_back(out, "qty"),
        [40, 30, 25, 20, 15, 12, 8, 5, 3, 1],
        "descending",
    )


def test_a_sort_keeps_the_chunks_it_was_given() raises:
    # Ten rows arrive as three, four and three, and a breaker that handed the
    # ten back as one chunk would make every operator above it pay for the
    # sort's memory rather than its own.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var root = plan.sort(scan, [qty], [False], [True])
    _ = bind(plan, root, schemas())
    var out = lower(plan, root, one_frame())^.run()

    assert_equal(out.columns[0].num_chunks(), 3, "three chunks back")


def test_a_second_key_orders_what_the_first_one_tied() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var ten = plan.exprs.literal(Value(Int64(10)))
    var band = plan.exprs.binary(BinaryOp.GT, plan.exprs.column("qty"), ten)
    var over = plan.project(
        scan,
        [band, plan.exprs.column("qty")],
        ["over", "qty"],
    )
    var root = plan.sort(
        over,
        [plan.exprs.column("over"), plan.exprs.column("qty")],
        [False, True],
        [True, True],
    )
    var out = run(plan, root)

    # Everything at or under ten first, and inside each of those two runs the
    # quantity downwards, which only the second key decides.
    same(
        read_back(out, "qty"),
        [8, 5, 3, 1, 40, 30, 25, 20, 15, 12],
        "the second key inside the first",
    )


def test_a_key_may_be_computed_rather_than_a_column() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var total = plan.exprs.binary(
        BinaryOp.MUL, plan.exprs.column("qty"), plan.exprs.column("price")
    )
    var root = plan.sort(scan, [total], [True], [True])
    var out = run(plan, root)

    # The computed key is read and never emitted, so the two input columns are
    # what comes out and the products are in order behind them.
    assert_equal(out.width(), 2, "the input columns and nothing else")
    same(
        read_back(out, "qty"),
        [30, 1, 15, 25, 8, 12, 5, 20, 40, 3],
        "by quantity times price",
    )


def test_a_sort_under_a_limit_is_the_top_of_it() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var sorted = plan.sort(scan, [qty], [True], [True])
    var root = plan.limit(sorted, 0, 3)
    var out = run(plan, root)

    same(read_back(out, "qty"), [40, 30, 25], "the three largest")


def test_a_sort_over_an_aggregate_orders_the_groups() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var ten = plan.exprs.literal(Value(Int64(10)))
    var band = plan.exprs.binary(BinaryOp.GT, plan.exprs.column("qty"), ten)
    var over = plan.project(
        scan, [band, plan.exprs.column("qty")], ["over", "qty"]
    )
    var total = plan.exprs.aggregate(AggKind.SUM, plan.exprs.column("qty"))
    var grouped = plan.aggregate(
        over, [plan.exprs.column("over")], [total], ["over", "total"]
    )
    var root = plan.sort(grouped, [plan.exprs.column("total")], [False], [True])
    var out = run(plan, root)

    # Seventeen at or under ten, a hundred and forty two over it.
    same(read_back(out, "total"), [17, 142], "the group totals in order")


def test_a_distinct_over_the_whole_row_keeps_one_of_each() raises:
    # The last digit of each quantity, which repeats where the quantity does
    # not, and repeats across a chunk boundary as well as inside one.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var ones = plan.exprs.binary(
        BinaryOp.MOD,
        plan.exprs.column("qty"),
        plan.exprs.literal(Value(Int64(10))),
    )
    var narrowed = plan.project(scan, [ones], [String("ones")])
    var out = run(plan, plan.distinct(narrowed, List[Int]()))
    same(read_back(out, "ones"), [5, 0, 3, 2, 8, 1], "one of each, first seen")


def test_a_distinct_decides_on_every_column_it_keeps() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var ones = plan.exprs.binary(
        BinaryOp.MOD,
        plan.exprs.column("qty"),
        plan.exprs.literal(Value(Int64(10))),
    )
    var evens = plan.exprs.binary(
        BinaryOp.MOD,
        plan.exprs.column("price"),
        plan.exprs.literal(Value(Int64(2))),
    )
    var narrowed = plan.project(
        scan, [ones, evens], [String("ones"), String("evens")]
    )
    var out = run(plan, plan.distinct(narrowed, List[Int]()))
    same(read_back(out, "ones"), [5, 0, 3, 0, 2, 8, 5, 1], "the first column")
    same(read_back(out, "evens"), [0, 0, 1, 1, 1, 1, 1, 0], "the second")


def test_a_distinct_may_name_the_columns_it_decides_on() raises:
    # The same thing written the other way. An empty key list means the whole
    # row and so does naming every column of it in order.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var ones = plan.exprs.binary(
        BinaryOp.MOD,
        plan.exprs.column("qty"),
        plan.exprs.literal(Value(Int64(10))),
    )
    var narrowed = plan.project(scan, [ones], [String("ones")])
    var root = plan.distinct(narrowed, [plan.exprs.column("ones")])
    var out = run(plan, root)
    same(read_back(out, "ones"), [5, 0, 3, 2, 8, 1], "one of each, first seen")


def test_a_distinct_on_part_of_the_row_keeps_the_first_of_each() raises:
    # The last digit of each quantity repeats where the quantity does not, so
    # the rows that survive are chosen by a column and carry another one out
    # with them, which is the whole difference from the distinct above.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var ones = plan.exprs.binary(
        BinaryOp.MOD,
        plan.exprs.column("qty"),
        plan.exprs.literal(Value(Int64(10))),
    )
    var narrowed = plan.project(
        scan, [ones, plan.exprs.column("qty")], [String("ones"), String("qty")]
    )
    var root = plan.distinct(narrowed, [plan.exprs.column("ones")])
    var out = run(plan, root)
    same(read_back(out, "ones"), [5, 0, 3, 2, 8, 1], "one of each, first seen")
    same(read_back(out, "qty"), [5, 20, 3, 12, 8, 1], "the row it came from")


def test_a_distinct_that_reorders_the_row_keeps_the_first_of_each() raises:
    # Naming every column but in the other order is still the whole row, and
    # the answer has to come back in the order the plan numbered the columns
    # rather than in the order the keys were written.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var ones = plan.exprs.binary(
        BinaryOp.MOD,
        plan.exprs.column("qty"),
        plan.exprs.literal(Value(Int64(10))),
    )
    var evens = plan.exprs.binary(
        BinaryOp.MOD,
        plan.exprs.column("price"),
        plan.exprs.literal(Value(Int64(2))),
    )
    var narrowed = plan.project(
        scan, [ones, evens], [String("ones"), String("evens")]
    )
    var root = plan.distinct(
        narrowed, [plan.exprs.column("evens"), plan.exprs.column("ones")]
    )
    var out = run(plan, root)
    same(read_back(out, "ones"), [5, 0, 3, 0, 2, 8, 5, 1], "the first column")
    same(read_back(out, "evens"), [0, 0, 1, 1, 1, 1, 1, 0], "the second")


def test_a_computed_distinct_key_is_refused_by_name() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var pair = plan.exprs.binary(
        BinaryOp.ADD,
        plan.exprs.column("qty"),
        plan.exprs.column("price"),
    )
    var root = plan.distinct(scan, [pair, plan.exprs.column("qty")])
    _ = bind(plan, root, schemas())

    with assert_raises(contains="decides on a binary expression"):
        _ = lower(plan, root, one_frame())


def test_a_unary_expression_lands_in_a_column_of_its_own() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var negated = plan.exprs.unary(UnaryOp.NEG, qty)
    var root = plan.project(scan, [negated], ["negated"])
    _ = bind(plan, root, schemas())
    var out = lower(plan, root, one_frame())^.run()

    assert_equal(out.width(), 1, "the projection kept one column")
    same(
        read_back(out, "negated"),
        [-5, -20, -3, -40, -12, -8, -25, -1, -30, -15],
        "negated",
    )


def test_a_unary_reads_the_column_the_one_under_it_wrote() raises:
    # The operand is an expression rather than an input column, so the operator
    # has to read the position the node under it appended and not a column of
    # the scan. That is what a plan built by hand catches and a query does not,
    # a query having no way to write a position down.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var doubled = plan.exprs.binary(
        BinaryOp.MUL,
        plan.exprs.column("qty"),
        plan.exprs.literal(Value(Int64(2))),
    )
    var root = plan.project(
        scan, [plan.exprs.unary(UnaryOp.NEG, doubled)], ["down"]
    )
    _ = bind(plan, root, schemas())
    var out = lower(plan, root, one_frame())^.run()

    same(
        read_back(out, "down"),
        [-10, -40, -6, -80, -24, -16, -50, -2, -60, -30],
        "down",
    )


def test_a_projection_of_a_bare_constant_is_a_column_of_it() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var one = plan.exprs.literal(Value(Int64(1)))
    var root = plan.project(scan, [one], ["one"])
    var out = run(plan, root)

    assert_equal(len(out.schema), 1, "only the constant comes out")
    same(read_back(out, "one"), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "every row")


def test_a_constant_column_keeps_the_name_the_query_gave_it() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var tag = plan.exprs.literal(Value(Int64(7)))
    var root = plan.project(scan, [qty, tag], ["qty", "tag"])
    var out = run(plan, root)

    same(
        read_back(out, "qty"), [5, 20, 3, 40, 12, 8, 25, 1, 30, 15], "the input"
    )
    same(read_back(out, "tag"), [7, 7, 7, 7, 7, 7, 7, 7, 7, 7], "the constant")


def test_a_constant_can_be_an_operand_of_what_comes_after_it() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var one = plan.exprs.literal(Value(Int64(1)))
    var two = plan.exprs.literal(Value(Int64(2)))
    var sum = plan.exprs.binary(BinaryOp.ADD, one, two)
    var root = plan.project(scan, [sum], ["three"])
    _ = bind(plan, root, schemas())
    # Simplify has not run here, so both operands are still constants and the
    # binary branch refuses them rather than building a column for each.
    with assert_raises(contains="both operands of an operation are constants"):
        _ = lower(plan, root, one_frame())


def test_a_constant_is_an_operand_in_one_output_and_a_column_in_another() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var one = plan.exprs.literal(Value(Int64(1)))
    var up = plan.exprs.binary(BinaryOp.ADD, qty, one)
    var root = plan.project(scan, [up, one], ["up", "one"])
    var out = run(plan, root)

    same(read_back(out, "up"), [6, 21, 4, 41, 13, 9, 26, 2, 31, 16], "the sum")
    same(read_back(out, "one"), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "the constant")


def test_a_constant_that_is_null_fills_the_column_with_nothing() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var nothing = plan.exprs.literal(Value(null=LogicalType.INT64))
    var root = plan.project(scan, [nothing], ["nothing"])
    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())
    var out = pipe^.run()

    valid(
        present(out, "nothing"),
        [False, False, False, False, False, False, False, False, False, False],
        "no row has a value",
    )


def test_a_cast_of_an_input_column_lands_in_a_column_of_its_own() raises:
    # Converting position zero where it lies would change what that position
    # means for every expression already bound against it, so the converted
    # column is appended and the input keeps the type it arrived with.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var wider = plan.exprs.cast(LogicalType.FLOAT64, qty)
    var root = plan.project(scan, [qty, wider], ["qty", "wide"])
    var out = run(plan, root)

    assert_equal(out.width(), 2, "both columns")
    assert_true(out.schema[0].dtype == LogicalType.INT64, "the input as it was")
    assert_true(out.schema[1].dtype == LogicalType.FLOAT64, "and the cast")
    same(
        read_back(out, "qty"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15],
        "the input column is untouched",
    )


def test_a_cast_of_an_input_column_converts_the_values() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var narrow = plan.exprs.cast(LogicalType.INT32, qty)
    var root = plan.project(scan, [narrow], ["small"])
    var out = run(plan, root)

    assert_equal(out.width(), 1, "one column")
    var col = out.column("small").as_typed[DType.int32]()
    assert_equal(len(col), 10, "every row")
    assert_equal(col[0], 5, "the first")
    assert_equal(col[6], 25, "and one from the middle chunk")


def test_a_cast_of_a_computed_column_is_allowed() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var total = plan.exprs.binary(BinaryOp.MUL, qty, price)
    var wider = plan.exprs.cast(LogicalType.FLOAT64, total)
    var root = plan.project(scan, [wider], ["total"])
    var out = run(plan, root)

    assert_equal(out.width(), 1, "one column")
    assert_true(
        out.schema[0].dtype == LogicalType.FLOAT64, "and it was converted"
    )


def test_a_limit_that_skips_rows_starts_further_in() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var root = plan.limit(scan, 2, 3)
    var out = run(plan, root)

    same(read_back(out, "qty"), [3, 40, 12], "skipped two and kept three")


def test_a_skip_that_lands_inside_a_chunk_cuts_it() raises:
    # The first chunk is three rows and the skip is four, so the second chunk
    # is the one that is cut, and it is cut one row in.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var root = plan.limit(scan, 4, 2)
    var out = run(plan, root)

    same(read_back(out, "qty"), [12, 8], "the middle of a chunk")


def test_a_skip_with_no_limit_keeps_the_rest() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var root = plan.limit(scan, 7, NO_LIMIT)
    var out = run(plan, root)

    same(read_back(out, "qty"), [1, 30, 15], "everything after the skip")


def test_a_skip_past_the_end_gives_nothing_back() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var root = plan.limit(scan, 20, 5)
    var out = run(plan, root)

    assert_equal(len(out), 0, "no rows")


def test_a_projection_may_rename_the_column_it_keeps() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var root = plan.project(scan, [qty], ["howmany"])
    var out = run(plan, root)

    assert_equal(out.width(), 1, "one column")
    assert_equal(out.schema[0].name, "howmany", "under the name asked for")
    same(
        read_back(out, "howmany"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15],
        "and the rows are the ones it renamed",
    )


def test_a_scan_naming_a_column_the_frame_lacks_is_refused() raises:
    var plan = Plan()
    var root = plan.scan("sales", ["qty"], 0)
    _ = bind(plan, root, schemas())

    # Binding passed because the schema it was given has the column. The frame
    # handed to lowering is the one that decides, and here it is a frame of
    # nothing but prices.
    var narrow = List[DataFrame]()
    narrow.append(sales().select(["price"]))

    with assert_raises(contains="does not have"):
        _ = lower(plan, root, narrow^)


def test_a_shared_subexpression_is_computed_once() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var total = plan.exprs.binary(BinaryOp.MUL, qty, price)
    var one = plan.exprs.binary(
        BinaryOp.ADD, total, plan.exprs.literal(Value(Int64(1)))
    )
    var two = plan.exprs.binary(
        BinaryOp.SUB, total, plan.exprs.literal(Value(Int64(2)))
    )
    var root = plan.project(scan, [one, two], ["up", "down"])

    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())

    # The product, the two the outputs add to it, and the projection. Without
    # the memo the product would be there twice.
    assert_equal(len(pipe.operators), 4, "operators")

    var out = pipe^.run()
    var up = read_back(out, "up")
    var down = read_back(out, "down")
    assert_equal(up[0], 51, "five times ten and one")
    assert_equal(down[0], 48, "and the same product less two")


def test_two_outputs_that_are_the_same_expression_keep_their_names() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var total = plan.exprs.binary(BinaryOp.MUL, qty, price)
    var root = plan.project(scan, [total, total], ["total", "again"])
    var out = run(plan, root)

    # The top of an output is never shared, because the column it lands in
    # carries the name and one column cannot answer to two.
    assert_equal(out.width(), 2, "two columns")
    assert_equal(out.schema[0].name, "total", "the first name")
    assert_equal(out.schema[1].name, "again", "and the second")
    assert_equal(read_back(out, "again")[0], 50, "with the same value in it")


def test_two_folds_over_one_expression_read_one_column() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var total = plan.exprs.binary(BinaryOp.MUL, qty, price)
    var root = plan.aggregate(
        scan,
        List[Int](),
        [
            plan.exprs.aggregate(AggKind.SUM, total),
            plan.exprs.aggregate(AggKind.MAX, total),
        ],
        ["revenue", "biggest"],
    )

    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())

    # One product and one reduction. A fold names its own output, so both of
    # them can read the column the other made.
    assert_equal(len(pipe.operators), 2, "operators")

    var out = pipe^.run()
    assert_equal(read_back(out, "revenue")[0], 668, "the whole frame summed")
    assert_equal(read_back(out, "biggest")[0], 120, "and the largest of them")


def test_a_cast_of_a_shared_column_does_not_convert_it_twice() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var total = plan.exprs.binary(BinaryOp.MUL, qty, price)
    var wider = plan.exprs.cast(LogicalType.FLOAT64, total)
    var root = plan.project(scan, [wider, total], ["wide", "narrow"])
    var out = run(plan, root)

    # The cast converted the product where it lay, so the second output cannot
    # have the column it was handed. It gets its own, still an integer.
    assert_true(out.schema[0].dtype == LogicalType.FLOAT64, "converted")
    assert_true(out.schema[1].dtype == LogicalType.INT64, "and not converted")
    assert_equal(read_back(out, "narrow")[0], 50, "with the product in it")


def test_a_merged_projection_computes_the_shared_expression_once() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var lower_node = plan.project(
        scan, [plan.exprs.binary(BinaryOp.MUL, qty, price)], ["total"]
    )
    var total = plan.exprs.column("total")
    var root = plan.project(
        lower_node,
        [plan.exprs.binary(BinaryOp.ADD, total, total)],
        ["doubled"],
    )
    _ = merge(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())

    # This is the reason the merging pass is allowed to substitute an
    # expression an upper projection reads twice. The graft put one index in
    # two places and the memo is what keeps that from being two multiplies.
    assert_equal(len(pipe.operators), 3, "a multiply, an add and a projection")

    var out = pipe^.run()
    assert_equal(read_back(out, "doubled")[0], 100, "twice five times ten")


def test_an_inner_join_pairs_the_rows_that_match() raises:
    # The three quantities a band has, in the order the probe side arrived in,
    # which is what a join in a pipeline keeps and a whole frame join does not
    # have to.
    var plan = Plan()
    var root = joined(plan, JoinKind.INNER)
    var out = run_two(plan, root)
    same(read_back(out, "qty"), [20, 3, 40], "the matched quantities")
    same(read_back(out, "rate"), [200, 300, 400], "the rate of each")


def test_a_join_produces_the_two_schemas_end_to_end() raises:
    # The positions the plan numbered. A projection above the join reads by
    # position, so a right column landing anywhere else is a wrong answer with
    # nothing to catch it.
    var plan = Plan()
    var root = joined(plan, JoinKind.INNER)
    var out = run_two(plan, root)
    assert_equal(len(out.schema), 4)
    assert_equal(out.schema[0].name, "qty")
    assert_equal(out.schema[1].name, "price")
    assert_equal(out.schema[2].name, "band")
    assert_equal(out.schema[3].name, "rate")


def test_a_projection_over_a_join_reads_either_side() raises:
    var plan = Plan()
    var at = joined(plan, JoinKind.INNER)
    var root = plan.project(
        at,
        [plan.exprs.column("rate"), plan.exprs.column("price")],
        ["rate", "price"],
    )
    var out = run_two(plan, root)
    same(read_back(out, "rate"), [200, 300, 400], "the rate of each")
    same(read_back(out, "price"), [2, 7, 1], "the price of each")


def test_a_left_join_keeps_the_rows_that_matched_nothing() raises:
    var plan = Plan()
    var root = joined(plan, JoinKind.LEFT)
    var out = run_two(plan, root)
    assert_equal(out.rows, 10)
    same(
        read_back(out, "qty"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15],
        "every row of the left",
    )
    # The rate is missing wherever the quantity was not a band, which is the
    # padding a left join is for.
    valid(
        present(out, "rate"),
        [
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        "the padded rows",
    )


def test_a_semi_join_keeps_the_left_row_and_none_of_the_right() raises:
    var plan = Plan()
    var root = joined(plan, JoinKind.SEMI)
    var out = run_two(plan, root)
    assert_equal(len(out.schema), 2)
    same(read_back(out, "qty"), [20, 3, 40], "the matched quantities")


def test_an_anti_join_keeps_the_rows_no_band_matched() raises:
    var plan = Plan()
    var root = joined(plan, JoinKind.ANTI)
    var out = run_two(plan, root)
    same(
        read_back(out, "qty"),
        [5, 12, 8, 25, 1, 30, 15],
        "the quantities no band matched",
    )


def test_a_mark_join_answers_a_boolean_on_every_left_row() raises:
    # The semi join above keeps three rows. This keeps all ten and says which
    # three they were, which is the difference between a filter and a value.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        JoinKind.MARK,
        "found",
    )
    var out = run_two(plan, root)
    assert_equal(out.rows, 10)
    assert_equal(len(out.schema), 3, "the left side and the mark")
    assert_equal(out.schema[2].name, "found", "what the mark is called")
    same(
        read_back(out, "qty"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15],
        "every row of the left",
    )
    same(
        truths(out, "found"),
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "which quantities were a band",
    )


def test_a_mark_join_keeps_no_column_of_the_right_side() raises:
    # The rate is what an inner join would have brought across, and the mark
    # join asks about the right side rather than reading it.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        JoinKind.MARK,
        "found",
    )
    var out = run_two(plan, root)
    assert_equal(out.schema.has("rate"), False, "no rate came across")


def test_a_mark_that_missed_a_side_holding_a_null_is_null() raises:
    # SQL's `IN` is three valued. A quantity that matched no band is only known
    # to have matched nothing when there was nothing it might have matched, and
    # a null band is a band nobody wrote down.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        JoinKind.MARK,
        "found",
    )
    var frames = List[DataFrame]()
    frames.append(holey("qty", [5, 20, 3, 40], List[Int]()))
    frames.append(holey("band", [3, 20, 0, 99], [2]))
    var out = run_frames(plan, root, frames^)
    same(
        truths(out, "found"),
        [-1, 1, 1, -1],
        "a miss against a null band is unknown",
    )


def test_a_mark_whose_own_key_is_null_is_null_even_on_a_clean_side() raises:
    # The other half of the same rule, on the other side of the comparison. A
    # quantity nobody wrote down might have been any band there is.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        JoinKind.MARK,
        "found",
    )
    var frames = List[DataFrame]()
    frames.append(holey("qty", [5, 20, 0, 40], [2]))
    frames.append(holey("band", [3, 20, 40, 99], List[Int]()))
    var out = run_frames(plan, root, frames^)
    same(
        truths(out, "found"),
        [0, 1, -1, 1],
        "the null quantity is unknown and the rest are not",
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


def test_a_semi_join_whose_build_side_filtered_to_nothing_keeps_no_rows() raises:
    # A column that no rows reached has no chunks at all rather than one empty
    # chunk, which is the column's own rule, and building the key table used to
    # read the one chunk and raise on this. #611.
    var plan = Plan()
    var root = emptied(plan, JoinKind.SEMI)
    assert_equal(len(run_two(plan, root)), 0)


def test_an_anti_join_over_nothing_keeps_every_left_row() raises:
    # The other half of the same question. Nothing matched, so nothing is
    # dropped, and a left row with a null key is kept too.
    var plan = Plan()
    var root = emptied(plan, JoinKind.ANTI)
    assert_equal(len(run_two(plan, root)), 10)


def test_a_left_join_over_nothing_keeps_every_left_row_with_nulls() raises:
    # The one kind that reads the build side's other columns rather than only
    # its key table, so this is the one that gathers from the empty array.
    var plan = Plan()
    var root = emptied(plan, JoinKind.LEFT)
    var got = run_two(plan, root)
    assert_equal(len(got), 10)
    assert_equal(len(got.schema), 4, "both sides end to end")
    var held = present(got, "rate")
    for i in range(len(held)):
        assert_true(not held[i], String("row ", i))


def test_a_mark_join_over_a_build_side_with_nothing_in_it_is_all_false() raises:
    # False rather than null on every row, because there is nothing to match
    # and no null in the build side to be unsure about.
    var plan = Plan()
    var root = emptied(plan, JoinKind.MARK, "found")
    var got = run_two(plan, root)
    assert_equal(len(got), 10)
    same(
        truths(got, "found"),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "nothing to match",
    )


def test_a_mark_join_has_to_be_told_what_to_call_its_column() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    with assert_raises(contains="the name is not optional"):
        _ = plan.join(
            left,
            right,
            [plan.exprs.column("qty")],
            [plan.exprs.column("band")],
            JoinKind.MARK,
        )


def test_only_a_mark_join_is_given_a_mark_name() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    with assert_raises(contains="the mark column is the mark join's"):
        _ = plan.join(
            left,
            right,
            [plan.exprs.column("qty")],
            [plan.exprs.column("band")],
            JoinKind.SEMI,
            "found",
        )


def test_a_mark_join_cannot_take_a_name_the_left_side_already_uses() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        JoinKind.MARK,
        "price",
    )
    with assert_raises(contains="already has a column of that name"):
        _ = bind(plan, root, two_schemas())


def test_a_filter_under_a_join_runs_before_the_probe() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var keep = plan.exprs.binary(
        BinaryOp.GT,
        plan.exprs.column("qty"),
        plan.exprs.literal(Value(Int64(5))),
    )
    var root = plan.join(
        plan.filter(left, keep),
        right,
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        JoinKind.INNER,
    )
    var out = run_two(plan, root)
    same(read_back(out, "qty"), [20, 40], "over five and a band")


def test_a_narrowed_scan_narrows_the_build_side_too() raises:
    # The build side is a frame rather than a stream, so the scan's column list
    # is applied by selecting columns of it and not by an operator over it.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", ["band"], 1)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        JoinKind.INNER,
    )
    var out = run_two(plan, root)
    assert_equal(len(out.schema), 3)
    same(read_back(out, "qty"), [20, 3, 40], "the matched quantities")


def test_a_right_join_is_refused_by_name() raises:
    var plan = Plan()
    var root = joined(plan, JoinKind.RIGHT)
    _ = bind(plan, root, two_schemas())
    with assert_raises(contains="not known until the last chunk"):
        _ = lower(plan, root, two_frames())


def test_an_outer_join_is_refused_by_name() raises:
    var plan = Plan()
    var root = joined(plan, JoinKind.OUTER)
    _ = bind(plan, root, two_schemas())
    with assert_raises(contains="not known until the last chunk"):
        _ = lower(plan, root, two_frames())


def test_a_cross_join_onto_more_than_one_row_is_refused() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var root = plan.join(left, right, List[Int](), List[Int](), JoinKind.CROSS)
    _ = bind(plan, root, two_schemas())
    with assert_raises(contains="right side of 4 rows"):
        _ = lower(plan, root, two_frames())


def test_a_cross_join_onto_one_row_is_a_column_per_right_column() raises:
    # One right row moves nothing, so every left row comes back in its own
    # order with the right row's values beside it.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var one = plan.filter(
        right,
        plan.exprs.binary(
            BinaryOp.EQ,
            plan.exprs.column("band"),
            plan.exprs.literal(Value(Int64(3))),
        ),
    )
    var root = plan.join(left, one, List[Int](), List[Int](), JoinKind.CROSS)
    var out = run_two(plan, root)
    same(read_back(out, "qty"), [5, 20, 3, 40, 12, 8, 25, 1, 30, 15], "qty")
    same(read_back(out, "band"), [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], "band")
    same(
        read_back(out, "rate"),
        [300, 300, 300, 300, 300, 300, 300, 300, 300, 300],
        "rate",
    )


def test_a_cross_join_onto_no_rows_is_refused_rather_than_empty() raises:
    # An empty right side makes the whole answer empty, which is a row count
    # this operator cannot produce, since it adds a column and keeps the rows.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var none = plan.filter(
        right,
        plan.exprs.binary(
            BinaryOp.GT,
            plan.exprs.column("band"),
            plan.exprs.literal(Value(Int64(1000))),
        ),
    )
    var root = plan.join(left, none, List[Int](), List[Int](), JoinKind.CROSS)
    _ = bind(plan, root, two_schemas())
    with assert_raises(contains="right side of 0 rows"):
        _ = lower(plan, root, two_frames())


def test_a_join_whose_right_side_is_a_filter_builds_it_first() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var keep = plan.exprs.binary(
        BinaryOp.GT,
        plan.exprs.column("band"),
        plan.exprs.literal(Value(Int64(5))),
    )
    var root = plan.join(
        left,
        plan.filter(right, keep),
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        JoinKind.INNER,
    )
    var out = run_two(plan, root)
    same(read_back(out, "qty"), [20, 40], "qty")
    same(read_back(out, "rate"), [200, 400], "rate")


def test_a_join_can_build_from_a_projection() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var narrowed = plan.project(
        right,
        [plan.exprs.column("rate"), plan.exprs.column("band")],
        [String("rate"), String("band")],
    )
    var root = plan.join(
        left,
        narrowed,
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        JoinKind.INNER,
    )
    var out = run_two(plan, root)
    same(read_back(out, "qty"), [20, 3, 40], "qty")
    same(read_back(out, "rate"), [200, 300, 400], "rate")


def test_a_build_side_takes_the_relation_it_read() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("sales", List[String](), 0)
    var keep = plan.exprs.binary(
        BinaryOp.GT,
        plan.exprs.column("qty"),
        plan.exprs.literal(Value(Int64(5))),
    )
    var root = plan.join(
        left,
        plan.filter(right, keep),
        [plan.exprs.column("qty")],
        [plan.exprs.column("qty")],
        JoinKind.SEMI,
    )
    _ = bind(plan, root, two_schemas())
    with assert_raises(contains="both read relation 0"):
        _ = lower(plan, root, two_frames())


def test_a_build_side_that_cannot_be_lowered_says_what_it_was() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var computed = plan.exprs.binary(
        BinaryOp.ADD, plan.exprs.column("band"), plan.exprs.column("rate")
    )
    var root = plan.join(
        left,
        plan.distinct(right, [computed]),
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        JoinKind.INNER,
    )
    _ = bind(plan, root, two_schemas())
    with assert_raises(contains="decides on a binary expression"):
        _ = lower(plan, root, two_frames())


def test_a_join_on_two_key_pairs_that_agree_on_nothing_is_empty() raises:
    # Three sales rows have a quantity one of the bands matches, and none of
    # the three has the rate that band charges. So the pairing makes three
    # pairs and the second key drops all three, which is the case where every
    # chunk the filter sees comes back empty.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("qty"), plan.exprs.column("price")],
        [plan.exprs.column("band"), plan.exprs.column("rate")],
        JoinKind.INNER,
    )
    var out = run_two(plan, root)

    assert_equal(len(out), 0, "no row agrees on both")
    assert_equal(out.width(), 4, "and the schema is still both sides")


def test_a_computed_key_is_refused_by_name() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var key = plan.exprs.binary(
        BinaryOp.ADD,
        plan.exprs.column("qty"),
        plan.exprs.literal(Value(Int64(0))),
    )
    var root = plan.join(
        left, right, [key], [plan.exprs.column("band")], JoinKind.INNER
    )
    _ = bind(plan, root, two_schemas())
    with assert_raises(contains="joins on a column on each side"):
        _ = lower(plan, root, two_frames())


def test_a_name_both_sides_have_comes_back_twice() raises:
    # A join binds to the two schemas end to end, names and all, and the
    # operator is told that by position rather than left to work it out from
    # names it has two of. This used to be refused.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("echoes", List[String](), 1)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("qty")],
        [plan.exprs.column("qty")],
        JoinKind.INNER,
    )
    var schemas = List[Schema]()
    schemas.append(Schema(copy=sales().schema))
    schemas.append(Schema(copy=echoes().schema))
    var bound = bind(plan, root, schemas)
    var frames = List[DataFrame]()
    frames.append(sales())
    frames.append(echoes())
    var out = lower(plan, root, frames^).run()
    assert_equal(len(out.schema), 4, "qty and price from each side")
    assert_equal(out.schema[0].name, "qty")
    assert_equal(out.schema[1].name, "price")
    assert_equal(out.schema[2].name, "qty", "the right one, unrenamed")
    assert_equal(out.schema[3].name, "price", "and this one too")
    assert_equal(len(bound), 4, "which is what the plan bound to")
    assert_equal(len(out), 3, "three quantities are in both")


def test_two_scans_of_one_relation_say_so() raises:
    # A relation is a frame, so a table joined to itself is two relations and
    # two frames. A plan that says one of each is a plan that was built wrong,
    # and saying that here beats a missing column further down.
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("sales", List[String](), 0)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("qty")],
        [plan.exprs.column("qty")],
        JoinKind.INNER,
    )
    _ = bind(plan, root, schemas())
    with assert_raises(contains="both read relation 0"):
        _ = lower(plan, root, one_frame())


def test_a_literal_table_is_the_rows_it_names() raises:
    var plan = Plan()
    var root = plan.values(
        [
            plan.exprs.literal(Value(Int64(1))),
            plan.exprs.literal(Value(Int64(2))),
            plan.exprs.literal(Value(Int64(3))),
            plan.exprs.literal(Value(Int64(4))),
        ],
        [String("a"), String("b")],
    )
    _ = bind(plan, root, List[Schema]())
    var pipe = lower(plan, root, List[DataFrame]())
    var out = pipe^.run()
    same(read_back(out, "a"), [1, 3], "the first column")
    same(read_back(out, "b"), [2, 4], "the second")


def test_a_literal_table_can_be_filtered_and_projected() raises:
    var plan = Plan()
    var table = plan.values(
        [
            plan.exprs.literal(Value(Int64(1))),
            plan.exprs.literal(Value(Int64(9))),
            plan.exprs.literal(Value(Int64(2))),
            plan.exprs.literal(Value(Int64(8))),
            plan.exprs.literal(Value(Int64(3))),
            plan.exprs.literal(Value(Int64(7))),
        ],
        [String("a"), String("b")],
    )
    var keep = plan.exprs.binary(
        BinaryOp.GT,
        plan.exprs.column("a"),
        plan.exprs.literal(Value(Int64(1))),
    )
    var total = plan.exprs.binary(
        BinaryOp.ADD, plan.exprs.column("a"), plan.exprs.column("b")
    )
    var root = plan.project(
        plan.filter(table, keep), [total], [String("total")]
    )
    _ = bind(plan, root, List[Schema]())
    var pipe = lower(plan, root, List[DataFrame]())
    var out = pipe^.run()
    same(read_back(out, "total"), [10, 10], "the sums of the rows kept")


def test_a_literal_table_holds_a_missing_value() raises:
    var plan = Plan()
    var root = plan.values(
        [
            plan.exprs.literal(Value(Int64(1))),
            plan.exprs.literal(Value(null=LogicalType.INT64)),
        ],
        [String("a")],
    )
    _ = bind(plan, root, List[Schema]())
    var pipe = lower(plan, root, List[DataFrame]())
    var out = pipe^.run()
    valid(present(out, "a"), [True, False], "the second row is missing")


def test_a_union_all_of_two_scans_is_one_on_top_of_the_other() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.union([top, bottom], all=True)
    var out = run_two(plan, root)

    assert_equal(out.width(), 1, "one column")
    same(
        read_back(out, "qty"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15, 3, 20, 40, 99],
        "the first input then the second",
    )


def test_a_union_takes_the_first_input_s_names() raises:
    """The two sides call their column different things, and a union lines its
    inputs up by position, so the answer is called what the first one calls
    it."""
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.union([top, bottom], all=True)
    var out = run_two(plan, root)

    assert_equal(out.schema[0].name, "qty", "the first input's name")


def test_a_union_without_all_keeps_one_of_each_row() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.union([top, bottom], all=False)
    var out = run_two(plan, root)

    # 3, 20 and 40 are in both inputs, and 99 is only in the second.
    same(
        read_back(out, "qty"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15, 99],
        "one of each, in the order each was first seen",
    )


def test_a_union_of_three_inputs_is_one_node() raises:
    """Stacking is associative, so a chain of unions is one node with a list of
    inputs rather than a nest of two input nodes."""
    var plan = Plan()
    var a = plan.scan("tiers", ["band"], 1)
    var b = plan.scan("sales", ["qty"], 0)
    var c = plan.values(
        [
            plan.exprs.literal(Value(Int64(7))),
            plan.exprs.literal(Value(Int64(8))),
        ],
        [String("n")],
    )
    var root = plan.union([a, b, c], all=True)
    var out = run_two(plan, root)

    same(
        read_back(out, "band"),
        [3, 20, 40, 99, 5, 20, 3, 40, 12, 8, 25, 1, 30, 15, 7, 8],
        "all three in order",
    )


def test_each_input_of_a_union_may_be_a_query_of_its_own() raises:
    var plan = Plan()
    var scan = plan.scan("sales", ["qty"], 0)
    var qty = plan.exprs.column("qty")
    var big = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(25)))
    )
    var top = plan.filter(scan, big)
    var other = plan.scan("tiers", ["band"], 1)
    var band = plan.exprs.column("band")
    var small = plan.exprs.binary(
        BinaryOp.LT, band, plan.exprs.literal(Value(Int64(10)))
    )
    var bottom = plan.filter(other, small)
    var root = plan.union([top, bottom], all=True)
    var out = run_two(plan, root)

    same(read_back(out, "qty"), [40, 30, 3], "what each side kept")


def test_a_union_is_a_source_the_rest_of_the_line_reads() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var stacked = plan.union([top, bottom], all=True)
    var qty = plan.exprs.column("qty")
    var big = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(25)))
    )
    var root = plan.filter(stacked, big)
    var out = run_two(plan, root)

    same(read_back(out, "qty"), [40, 30, 40, 99], "the filter over the stack")


def test_a_union_whose_inputs_are_different_widths_is_refused_by_name() raises:
    """Binding refuses this before lowering sees it, which is the earlier and
    the better place, so the check in lowering is the one that catches a plan
    that was never bound rather than this."""
    var plan = Plan()
    var top = plan.scan("sales", List[String](), 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.union([top, bottom], all=True)

    with assert_raises(contains="a union is between a 2 column input"):
        _ = bind(plan, root, two_schemas())


def test_a_difference_keeps_the_rows_the_other_side_lacks() raises:
    # The bands are 3, 20, 40 and 99, so the three quantities that are also
    # bands go and the rest stay in the order the left side had them.
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_EXCEPT, all=False)
    var out = run_two(plan, root)

    assert_equal(out.width(), 1, "the tag the stack carried is gone")
    same(read_back(out, "qty"), [5, 12, 8, 25, 1, 30, 15], "what is left")


def test_a_difference_keeps_one_copy_of_a_row_it_keeps() raises:
    # A difference is over sets, so a quantity the left side wrote twice comes
    # back once. The tens of the quantities repeat and none of them is a band.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var ones = plan.exprs.binary(
        BinaryOp.MOD,
        plan.exprs.column("qty"),
        plan.exprs.literal(Value(Int64(10))),
    )
    var top = plan.project(scan, [ones], [String("ones")])
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_EXCEPT, all=False)
    var out = run_two(plan, root)

    same(read_back(out, "ones"), [5, 0, 2, 8, 1], "the threes went with 3")


def test_an_intersection_keeps_the_rows_both_sides_have() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_INTERSECT, all=False)
    var out = run_two(plan, root)

    assert_equal(out.width(), 1, "the tag the stack carried is gone")
    same(read_back(out, "qty"), [20, 3, 40], "99 is a band and not a quantity")


def test_a_set_operation_decides_on_every_column_of_the_row() raises:
    # Two columns rather than one, so a row that matches on the first and not
    # on the second is a row the other side does not have.
    var plan = Plan()
    var top = plan.scan("sales", ["qty", "price"], 0)
    var bottom = plan.values(
        [
            plan.exprs.literal(Value(Int64(20))),
            plan.exprs.literal(Value(Int64(2))),
            plan.exprs.literal(Value(Int64(40))),
            plan.exprs.literal(Value(Int64(7))),
        ],
        [String("qty"), String("price")],
    )
    var root = plan.setop([top, bottom], SET_INTERSECT, all=False)
    var out = run_two(plan, root)

    same(read_back(out, "qty"), [20], "40 goes with a price of 1 and not 7")
    same(read_back(out, "price"), [2], "and the price came along")


def test_a_difference_treats_two_nulls_as_the_same_row() raises:
    # Which is the rule a set operation has and a join does not, and the reason
    # this is a group by over a stack rather than an anti join.
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_EXCEPT, all=False)
    var frames = List[DataFrame]()
    frames.append(holey("qty", [5, 0, 20], [1]))
    frames.append(holey("band", [0, 20], [0]))
    var out = run_frames(plan, root, frames^)

    assert_equal(len(out), 1, "the null went with the other side's null")
    same(read_back(out, "qty"), [5], "and 5 is all that is left")


def test_an_intersection_keeps_a_null_both_sides_have() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_INTERSECT, all=False)
    var frames = List[DataFrame]()
    frames.append(holey("qty", [5, 0, 20], [1]))
    frames.append(holey("band", [0, 20], [0]))
    var out = run_frames(plan, root, frames^)

    valid(present(out, "qty"), [False, True], "the null is one of the two")
    assert_equal(read_back(out, "qty")[1], 20, "and 20 is the other")


def test_a_difference_written_all_subtracts_a_copy_at_a_time() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_EXCEPT, all=True)
    var out = run_two(plan, root)

    # No quantity is written twice, so ALL and the set answer agree here and
    # the test below is the one that tells them apart.
    same(read_back(out, "qty"), [5, 12, 8, 25, 1, 30, 15], "qty")


def test_a_difference_written_all_keeps_the_copies_over() raises:
    var plan = Plan()
    var top = plan.scan("copies", ["band"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_EXCEPT, all=True)
    var out = run_pair(plan, root, copies(), tiers())

    # Two threes on the left and one on the right leaves one three, and the
    # set answer leaves none.
    same(read_back(out, "band"), [3, 77], "band")


def test_an_intersection_written_all_keeps_the_thinner_count() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_INTERSECT, all=True)
    var out = run_two(plan, root)

    same(read_back(out, "qty"), [20, 3, 40], "qty")


def test_an_intersection_written_all_over_itself_is_itself() raises:
    var plan = Plan()
    var top = plan.scan("copies", ["band"], 0)
    var bottom = plan.scan("copies", ["band"], 1)
    var root = plan.setop([top, bottom], SET_INTERSECT, all=True)
    var out = run_pair(plan, root, copies(), copies())

    # Every count is the same on both sides, so the smaller of the two is the
    # count itself and the answer is the arm back again.
    same(read_back(out, "band"), [3, 3, 20, 77], "band")


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


def test_a_range_of_one_argument_starts_at_zero() raises:
    var plan = Plan()
    var root = counted(plan, "range", [5])
    var out = series(plan, root)
    same(read_back(out, "i"), [0, 1, 2, 3, 4], "five rows and no table")


def test_a_generate_series_stops_on_its_bound() raises:
    # The whole difference between the two functions is this one row.
    var plan = Plan()
    var root = counted(plan, "generate_series", [5])
    var out = series(plan, root)
    same(read_back(out, "i"), [0, 1, 2, 3, 4, 5], "six rows")


def test_a_range_of_two_arguments_is_a_start_and_a_stop() raises:
    var plan = Plan()
    var root = counted(plan, "range", [10, 14])
    var out = series(plan, root)
    same(read_back(out, "i"), [10, 11, 12, 13], "from the first to the second")


def test_a_range_of_three_arguments_counts_by_the_third() raises:
    var plan = Plan()
    var root = counted(plan, "range", [0, 10, 3])
    var out = series(plan, root)
    same(read_back(out, "i"), [0, 3, 6, 9], "and stops before ten")


def test_a_negative_step_counts_down() raises:
    var plan = Plan()
    var root = counted(plan, "range", [5, 0, -2])
    var out = series(plan, root)
    same(read_back(out, "i"), [5, 3, 1], "down to the bound and not past it")


def test_a_series_that_never_reaches_its_end_is_no_rows() raises:
    var plan = Plan()
    var root = counted(plan, "range", [5, 0])
    var out = series(plan, root)
    assert_equal(len(out), 0, "counting up from five to zero is nothing")


def test_a_step_of_zero_is_refused_by_name() raises:
    var plan = Plan()
    var root = counted(plan, "range", [0, 10, 0])
    _ = bind(plan, root, List[Schema]())

    with assert_raises(contains="a series that never moves"):
        _ = lower(plan, root, List[DataFrame]())


def test_a_series_with_a_null_end_is_no_rows() raises:
    var plan = Plan()
    var missing = plan.exprs.literal(Value(null=LogicalType.NULL))
    var root = plan.table_function("range", [missing], ["i"])
    var out = series(plan, root)
    assert_equal(len(out), 0, "nobody said where it ends")


def test_a_series_that_is_too_long_to_build_is_refused_by_name() raises:
    # It is built before the query starts, so a series of a trillion rows is an
    # allocation rather than a query that takes a while.
    var plan = Plan()
    var root = counted(plan, "range", [1_000_000_000_000])
    _ = bind(plan, root, List[Schema]())

    with assert_raises(contains="longer than 100000000"):
        _ = lower(plan, root, List[DataFrame]())


def test_a_series_takes_the_name_the_call_gave_it() raises:
    var plan = Plan()
    var three = plan.exprs.literal(Value(Int64(3)))
    var root = plan.table_function("range", [three], ["day"])
    var out = series(plan, root)
    same(read_back(out, "day"), [0, 1, 2], "under the name the call chose")


def test_a_series_is_a_source_the_rest_of_the_line_reads() raises:
    var plan = Plan()
    var rows = counted(plan, "range", [10])
    var i = plan.exprs.column("i")
    var four = plan.exprs.literal(Value(Int64(4)))
    var big = plan.exprs.binary(BinaryOp.GT, i, four)
    var kept = plan.filter(rows, big)
    var out = series(plan, kept)
    same(read_back(out, "i"), [5, 6, 7, 8, 9], "the filter ran over the series")


def test_an_argument_that_is_computed_is_folded_before_it_gets_here() raises:
    # Lowering reads the arguments off the tree, so `range(2 + 3)` only works
    # because simplify has already folded it. One that is not folded is refused
    # by name rather than computed, the same way a VALUES refuses one.
    var plan = Plan()
    var two = plan.exprs.literal(Value(Int64(2)))
    var three = plan.exprs.literal(Value(Int64(3)))
    var sum = plan.exprs.binary(BinaryOp.ADD, two, three)
    var root = plan.table_function("range", [sum], ["i"])
    _ = bind(plan, root, List[Schema]())

    with assert_raises(contains="argument 1 of range is a binary expression"):
        _ = lower(plan, root, List[DataFrame]())


def test_a_window_over_the_whole_frame_broadcasts_one_value() raises:
    # No partition keys is one partition, so every row gets the same answer and
    # the columns the frame had come through beside it.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var total = plan.exprs.window(AggKind.SUM, qty, List[Int](), List[Int]())
    var root = plan.window(scan, [total], ["total"])
    var got = run(plan, root)

    assert_equal(len(got.schema), 3, "the window adds a column")
    assert_equal(got.schema[2].name, "total", "and it is called what it asked")
    same(
        read_back(got, "qty"),
        [5, 20, 3, 40, 12, 8, 25, 1, 30, 15],
        "the rows come back as they were",
    )
    same(
        read_back(got, "total"),
        [159, 159, 159, 159, 159, 159, 159, 159, 159, 159],
        "every row gets the sum of all of them",
    )


def test_a_window_partitions_and_each_row_reads_its_own() raises:
    # Two windows over one partitioning, which is one grouping pass and one
    # operator. The rows stay where they were and each one gets the answer for
    # the team it is on.
    var plan = Plan()
    var scan = plan.scan("shifts", List[String](), 0)
    var team = plan.exprs.column("team")
    var hours = plan.exprs.column("hours")
    var worked = plan.exprs.window(AggKind.SUM, hours, [team], List[Int]())
    var shifts = plan.exprs.window(AggKind.COUNT, hours, [team], List[Int]())
    var root = plan.window(scan, [worked, shifts], ["worked", "shifts"])
    _ = bind(plan, root, shift_schemas())
    var pipe = lower(plan, root, shift_frame())
    var got = pipe^.run()

    assert_equal(len(got.schema), 4, "two windows on two columns")
    same(
        read_back(got, "team"),
        [1, 2, 1, 2, 3, 1, 3, 2, 1],
        "the rows come back in the order they arrived",
    )
    same(
        read_back(got, "worked"),
        [13, 20, 13, 20, 12, 13, 12, 20, 13],
        "each row reads its own team's hours",
    )
    same(
        read_back(got, "shifts"),
        [4, 3, 4, 3, 2, 4, 2, 3, 4],
        "and its own team's count",
    )


def test_a_window_over_a_computed_expression_drops_the_intermediate() raises:
    # The product is appended before the breaker and the window lands after it,
    # so what the node hands up is the frame's columns and then the window, with
    # the column in between cut out.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var line = plan.exprs.binary(BinaryOp.MUL, qty, price)
    var total = plan.exprs.window(AggKind.SUM, line, List[Int](), List[Int]())
    var root = plan.window(scan, [total], ["revenue"])
    var got = run(plan, root)

    assert_equal(len(got.schema), 3, "the product is not one of the columns")
    assert_equal(
        got.schema[0].name, "qty", "the frame's columns keep their place"
    )
    assert_equal(got.schema[1].name, "price", "both of them")
    same(
        read_back(got, "revenue"),
        [668, 668, 668, 668, 668, 668, 668, 668, 668, 668],
        "every row gets the revenue of all of them",
    )


def test_a_column_above_a_window_reads_what_the_window_added() raises:
    # The node above is bound against the input's columns and then the window's,
    # so a projection that asks for the last one has to find it there.
    var plan = Plan()
    var scan = plan.scan("shifts", List[String](), 0)
    var team = plan.exprs.column("team")
    var hours = plan.exprs.column("hours")
    var worked = plan.exprs.window(AggKind.SUM, hours, [team], List[Int]())
    var window = plan.window(scan, [worked], ["worked"])
    var back = plan.exprs.column("worked")
    var root = plan.project(window, [back], ["worked"])
    _ = bind(plan, root, shift_schemas())
    var pipe = lower(plan, root, shift_frame())
    var got = pipe^.run()

    assert_equal(len(got.schema), 1, "the projection keeps the one column")
    same(
        read_back(got, "worked"),
        [13, 20, 13, 20, 12, 13, 12, 20, 13],
        "and it is the column the window added",
    )


def test_two_windows_that_partition_differently_are_refused() raises:
    # One operator makes one set of ordinals. Splitting the node here would put
    # the second window in a column the node above was not bound against, so
    # the split belongs to the plan and this says so.
    var plan = Plan()
    var scan = plan.scan("shifts", List[String](), 0)
    var team = plan.exprs.column("team")
    var hours = plan.exprs.column("hours")
    var by_team = plan.exprs.window(AggKind.SUM, hours, [team], List[Int]())
    var overall = plan.exprs.window(
        AggKind.SUM, hours, List[Int](), List[Int]()
    )
    var root = plan.window(scan, [by_team, overall], ["worked", "total"])
    _ = bind(plan, root, shift_schemas())

    with assert_raises(contains="partitions differently from the first one"):
        _ = lower(plan, root, shift_frame())


def test_an_ordered_window_is_refused_by_name() raises:
    # An ordering inside the window is a running fold, which is a different loop
    # rather than an argument to this one.
    var plan = Plan()
    var scan = plan.scan("shifts", List[String](), 0)
    var team = plan.exprs.column("team")
    var hours = plan.exprs.column("hours")
    var running = plan.exprs.window(AggKind.SUM, hours, [team], [hours])
    var root = plan.window(scan, [running], ["running"])
    _ = bind(plan, root, shift_schemas())

    with assert_raises(contains="is ordered, and an ordered window"):
        _ = lower(plan, root, shift_frame())


def test_a_conditional_picks_between_two_columns() raises:
    # The plain shape. A condition over one column, a column on each side, and
    # the answer is one column with rows from both.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var ten = plan.exprs.literal(Value(Int64(10)))
    var over = plan.exprs.binary(BinaryOp.GT, qty, ten)
    var picked = plan.exprs.conditional(over, qty, price)
    var root = plan.project(scan, [picked], ["taken"])
    var out = run(plan, root)

    assert_equal(len(out.schema), 1, "the projection keeps the one column")
    same(
        read_back(out, "taken"),
        [10, 20, 7, 40, 12, 9, 25, 100, 30, 15],
        "the quantity over ten and the price otherwise",
    )


def test_a_null_condition_takes_the_else_side() raises:
    # The rule worth pinning. A row the question could not be asked about is a
    # row the question did not hold for, which is what SQL says and is not what
    # an operation would say, since every one of those answers a null instead.
    var plan = Plan()
    var scan = plan.scan("gauges", List[String](), 0)
    var a = plan.exprs.column("a")
    var b = plan.exprs.column("b")
    var pair = plan.exprs.binary(BinaryOp.EQ, a, b)
    var seven = plan.exprs.literal(Value(Int64(7)))
    var nine = plan.exprs.literal(Value(Int64(9)))
    var picked = plan.exprs.conditional(pair, seven, nine)
    var root = plan.project(scan, [picked], ["taken"])
    _ = bind(plan, root, gauge_schemas())
    var pipe = lower(plan, root, gauge_frame())
    var out = pipe^.run()

    # Four of the five pairs have a null on one side or the other, so only the
    # last row asks a question that has an answer.
    same(
        read_back(out, "taken"),
        [9, 9, 9, 9, 7],
        "the else side wherever the condition was null",
    )


def test_a_chain_of_whens_is_a_chain_of_nodes() raises:
    # A conditional's else side is another conditional, which is how the parser
    # already builds a chain, so nothing here counts branches.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var high = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(20)))
    )
    var some = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(5)))
    )
    var inner = plan.exprs.conditional(
        some,
        plan.exprs.literal(Value(Int64(2))),
        plan.exprs.literal(Value(Int64(1))),
    )
    var outer = plan.exprs.conditional(
        high, plan.exprs.literal(Value(Int64(3))), inner
    )
    var root = plan.project(scan, [outer], ["band"])
    var out = run(plan, root)

    same(
        read_back(out, "band"),
        [1, 2, 1, 3, 2, 2, 3, 1, 3, 2],
        "three bands over the quantity",
    )


def test_a_conditional_over_two_types_casts_the_side_that_moves() raises:
    # Binding promotes the two sides and the node needs them to agree, so the
    # integer column is converted into a column of its own. In its own column
    # rather than where it lies, because the scan's column is still the scan's.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var ten = plan.exprs.literal(Value(Int64(10)))
    var over = plan.exprs.binary(BinaryOp.GT, qty, ten)
    var half = plan.exprs.literal(Value(Float64(0.5)))
    var picked = plan.exprs.conditional(over, qty, half)
    var root = plan.project(scan, [picked], ["taken"])
    var out = run(plan, root)

    assert_equal(
        out.schema[0].dtype,
        LogicalType.FLOAT64,
        "the two sides promote to the wider one",
    )
    var got = decimals(out, "taken")
    var want = [0.5, 20.0, 0.5, 40.0, 12.0, 0.5, 25.0, 0.5, 30.0, 15.0]
    assert_equal(len(got), len(want), "how many rows")
    for i in range(len(want)):
        assert_equal(got[i], want[i], "taken at " + String(i))


def test_a_conditional_may_be_a_predicate() raises:
    # The answer is a boolean column like any other, so a filter can read it.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var over = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(10)))
    )
    var dear = plan.exprs.binary(
        BinaryOp.GT, price, plan.exprs.literal(Value(Int64(5)))
    )
    var asked = plan.exprs.conditional(
        over, dear, plan.exprs.literal(Value(False))
    )
    var root = plan.filter(scan, asked)
    var out = run(plan, root)

    # Six rows are over ten and one of those six costs more than five.
    same(read_back(out, "qty"), [15], "the one row both halves kept")


def test_each_side_of_a_conditional_may_be_an_expression() raises:
    # Both sides lower the way anything else does, so an expression on each is
    # two lines of appends and then the choice between the two columns.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var over = plan.exprs.binary(
        BinaryOp.GT, qty, plan.exprs.literal(Value(Int64(10)))
    )
    var less = plan.exprs.binary(BinaryOp.SUB, qty, price)
    var more = plan.exprs.binary(BinaryOp.ADD, qty, price)
    var picked = plan.exprs.conditional(over, less, more)
    var root = plan.project(scan, [picked], ["net"])
    var out = run(plan, root)

    same(
        read_back(out, "net"),
        [15, 18, 10, 39, 7, 17, 22, 101, 26, 9],
        "the difference over ten and the sum otherwise",
    )


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


def test_a_join_on_two_keys_asks_the_second_after_pairing() raises:
    var plan = Plan()
    var root = crate_join(plan, JoinKind.INNER)
    var out = run_frames(plan, root, crate_frames())

    assert_equal(out.width(), 5, "both schemas end to end")
    same(read_back(out, "qty"), [5, 3, 40], "the rows both keys agree on")
    same(read_back(out, "kept"), [100, 300, 200], "paired with the right row")


def test_a_join_on_two_keys_drops_what_one_key_agrees_on() raises:
    # The same join on the shop alone, which pairs the row the test above
    # drops. Here so that the test above is known to be about the second key
    # rather than about a build side that happened to hold nothing else.
    var plan = Plan()
    var left = plan.scan("crates", List[String](), 0)
    var right = plan.scan("crated", List[String](), 1)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("shop")],
        [plan.exprs.column("place")],
        JoinKind.INNER,
    )
    var out = run_frames(plan, root, crate_frames())

    same(read_back(out, "qty"), [5, 5, 20, 3, 3, 40], "one key pairs six rows")


def test_a_semi_join_on_two_keys_is_refused() raises:
    var plan = Plan()
    var root = crate_join(plan, JoinKind.SEMI)
    _ = bind(plan, root, crate_schemas())

    with assert_raises(contains="2 key pairs would need the ordinal space"):
        _ = lower(plan, root, crate_frames())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
