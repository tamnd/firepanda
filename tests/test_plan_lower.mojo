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


def read_back(df: DataFrame, name: String) raises -> List[Int64]:
    """Reads an int64 column out as a plain list."""
    var col = df.column(name).as_typed[DType.int64]()
    var out = List[Int64](capacity=len(col))
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

    The names are disjoint from the sales frame's on purpose. The probe operator
    renames a right column whose name the left already has, which moves the
    columns the plan's schema numbered, so a join over two frames that share a
    name is refused and has a test of its own.
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


def test_a_reduction_that_cannot_fold_a_chunk_at_a_time_is_refused() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var middle = plan.exprs.aggregate(AggKind.MEDIAN, qty)
    var root = plan.aggregate(scan, List[Int](), [middle], ["middle"])
    _ = bind(plan, root, schemas())

    # The refusal comes from the physical node rather than from lowering, which
    # is the right place for it: what folds is a property of the reduction.
    with assert_raises(contains="cannot be computed a chunk at a time"):
        _ = lower(plan, root, one_frame())


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


def test_a_distinct_on_part_of_the_row_is_refused_by_name() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var root = plan.distinct(scan, [plan.exprs.column("qty")])
    _ = bind(plan, root, schemas())

    with assert_raises(contains="decides on 1 of the row's 2 columns"):
        _ = lower(plan, root, one_frame())


def test_a_distinct_that_reorders_the_row_is_refused_by_name() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var root = plan.distinct(
        scan, [plan.exprs.column("price"), plan.exprs.column("qty")]
    )
    _ = bind(plan, root, schemas())

    with assert_raises(contains="in another order"):
        _ = lower(plan, root, one_frame())


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


def test_a_unary_expression_is_refused() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var negated = plan.exprs.unary(UnaryOp.NEG, qty)
    var root = plan.project(scan, [negated], ["negated"])
    _ = bind(plan, root, schemas())

    with assert_raises(contains="computes a unary expression"):
        _ = lower(plan, root, one_frame())


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


def test_a_cast_of_an_input_column_is_refused() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var wider = plan.exprs.cast(LogicalType.FLOAT64, qty)
    var root = plan.project(scan, [wider], ["qty"])
    _ = bind(plan, root, schemas())

    with assert_raises(contains="where it lies"):
        _ = lower(plan, root, one_frame())


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


def test_a_cross_join_is_refused_by_name() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("tiers", List[String](), 1)
    var root = plan.join(left, right, List[Int](), List[Int](), JoinKind.CROSS)
    _ = bind(plan, root, two_schemas())
    with assert_raises(contains="no key to build a table from"):
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
    var root = plan.join(
        left,
        plan.distinct(right, [plan.exprs.column("band")]),
        [plan.exprs.column("qty")],
        [plan.exprs.column("band")],
        JoinKind.INNER,
    )
    _ = bind(plan, root, two_schemas())
    with assert_raises(contains="decides on 1 of the row's 2 columns"):
        _ = lower(plan, root, two_frames())


def test_a_join_on_two_key_pairs_is_refused_by_name() raises:
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
    _ = bind(plan, root, two_schemas())
    with assert_raises(contains="joins on one column"):
        _ = lower(plan, root, two_frames())


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


def test_a_name_both_sides_have_is_refused_by_name() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("sales", List[String](), 1)
    var root = plan.join(
        left,
        right,
        [plan.exprs.column("qty")],
        [plan.exprs.column("qty")],
        JoinKind.INNER,
    )
    var schemas = List[Schema]()
    schemas.append(Schema(copy=sales().schema))
    schemas.append(Schema(copy=sales().schema))
    _ = bind(plan, root, schemas)
    var frames = List[DataFrame]()
    frames.append(sales())
    frames.append(sales())
    with assert_raises(contains="both sides of this join have a column"):
        _ = lower(plan, root, frames^)


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


def test_a_difference_is_refused_by_name() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_EXCEPT, all=True)
    _ = bind(plan, root, two_schemas())

    with assert_raises(contains="a difference is not a stack of its inputs"):
        _ = lower(plan, root, two_frames())


def test_an_intersection_is_refused_by_name() raises:
    var plan = Plan()
    var top = plan.scan("sales", ["qty"], 0)
    var bottom = plan.scan("tiers", ["band"], 1)
    var root = plan.setop([top, bottom], SET_INTERSECT, all=True)
    _ = bind(plan, root, two_schemas())

    with assert_raises(contains="an intersection is not a stack of its inputs"):
        _ = lower(plan, root, two_frames())


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
