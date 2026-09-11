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
from firepanda.plan.node import NO_LIMIT, Plan
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


def run(mut plan: Plan, root: Int) raises -> DataFrame:
    """Binds, lowers and runs a plan over the sales frame."""
    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())
    return pipe^.run()


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

    # A compare and a filter for each half, then one projection to put the
    # schema back. Computing an and mask would have been four operators too,
    # but both comparisons would have run on all ten rows.
    assert_equal(len(pipe.operators), 5, "operators")

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
    assert_equal(len(pipe.operators), 7, "three compares, three filters, a cut")

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
    # comparison, one filter and the cut back to the input schema.
    assert_equal(len(pipe.operators), 3, "operators")
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


def test_a_sort_is_refused_by_name() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var root = plan.sort(scan, [qty], [False], [True])
    _ = bind(plan, root, schemas())

    with assert_raises(contains="no operator for a SORT node"):
        _ = lower(plan, root, one_frame())


def test_a_distinct_is_refused_by_name() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var root = plan.distinct(scan, [qty])
    _ = bind(plan, root, schemas())

    with assert_raises(contains="no operator for a DISTINCT node"):
        _ = lower(plan, root, one_frame())


def test_a_join_is_refused_by_name() raises:
    var plan = Plan()
    var left = plan.scan("sales", List[String](), 0)
    var right = plan.scan("sales", List[String](), 0)
    var a = plan.exprs.column("qty")
    var b = plan.exprs.column("qty")
    var root = plan.join(left, right, [a], [b], JoinKind.INNER)
    _ = bind(plan, root, schemas())

    with assert_raises(contains="no operator for a JOIN node"):
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


def test_a_projection_of_a_bare_constant_is_refused() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var one = plan.exprs.literal(Value(Int64(1)))
    var root = plan.project(scan, [one], ["one"])
    _ = bind(plan, root, schemas())

    with assert_raises(contains="makes a column out of a constant"):
        _ = lower(plan, root, one_frame())


def test_a_projection_that_renames_a_column_is_refused() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var root = plan.project(scan, [qty], ["amount"])
    _ = bind(plan, root, schemas())

    with assert_raises(contains="selects by position"):
        _ = lower(plan, root, one_frame())


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


def test_a_limit_that_skips_rows_is_refused() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var root = plan.limit(scan, 2, 3)
    _ = bind(plan, root, schemas())

    with assert_raises(contains="counts from the first row"):
        _ = lower(plan, root, one_frame())


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
