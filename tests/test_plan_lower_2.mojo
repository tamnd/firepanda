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

Part 2 of 3. The fixtures are in tests/support/plan_lower.mojo.
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

from tests.support.plan_lower import (
    copies,
    counted,
    crate_frames,
    crate_join,
    crate_schemas,
    crated,
    crates,
    decimals,
    echoes,
    emptied,
    gappy,
    gauge_frame,
    gauge_schemas,
    gauges,
    holey,
    joined,
    numbers,
    one_frame,
    present,
    read_back,
    run,
    run_frames,
    run_pair,
    run_two,
    sales,
    same,
    schemas,
    series,
    shift_frame,
    shift_schemas,
    shifts,
    tiers,
    truths,
    two_frames,
    two_schemas,
    valid,
)


def test_a_sort_under_a_limit_is_the_top_of_it() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var sorted = plan.sort(scan, [qty], [True], [True])
    var root = plan.limit(sorted, 0, 3)
    var out = run(plan, root)

    same(read_back(out, "qty"), [40, 30, 25], "the three largest")


def test_a_bounded_sort_answers_what_the_unbounded_one_would_have() raises:
    # The limit pass writes the bound and the lowering hands it over, so this is
    # the test above with the pass in the middle. The limit is still there and
    # still does the cutting, which is why the two answers have to agree.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var sorted = plan.sort(scan, [plan.exprs.column("qty")], [True], [True])
    var root = plan.limit(sorted, 0, 3)
    _ = limits(plan, root, schemas())
    assert_equal(plan.nodes[sorted].length, 3, "the sort owes three rows")
    var out = run(plan, root)

    same(read_back(out, "qty"), [40, 30, 25], "the three largest")


def test_a_bounded_sort_carries_the_other_columns_with_the_row() raises:
    # The bounded route gathers the columns that are not keys at the rows it
    # chose, the same as the unbounded one, and getting that wrong is a wrong
    # answer that reads like a right one.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var sorted = plan.sort(scan, [plan.exprs.column("qty")], [True], [True])
    var root = plan.limit(sorted, 0, 3)
    _ = limits(plan, root, schemas())
    var out = run(plan, root)

    same(read_back(out, "price"), [1, 4, 3], "the price of each row")


def test_a_bound_covers_the_rows_an_offset_skips() raises:
    # The pass bounds the sort at the offset plus the length, because the rows
    # the limit throws away still have to be found to know what is behind them.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var sorted = plan.sort(scan, [plan.exprs.column("qty")], [True], [True])
    var root = plan.limit(sorted, 2, 3)
    _ = limits(plan, root, schemas())
    assert_equal(plan.nodes[sorted].length, 5, "three rows after two")
    var out = run(plan, root)

    same(read_back(out, "qty"), [25, 20, 15], "the third, fourth and fifth")


def test_a_bounded_sort_on_two_keys_breaks_the_ties_the_same_way() raises:
    # A second key on the bounded route, which is a different kernel from the
    # passes the unbounded one runs, so the tie rule is worth asking again.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var ten = plan.exprs.literal(Value(Int64(10)))
    var band = plan.exprs.binary(BinaryOp.GT, plan.exprs.column("qty"), ten)
    var over = plan.project(
        scan, [band, plan.exprs.column("qty")], ["over", "qty"]
    )
    var sorted = plan.sort(
        over,
        [plan.exprs.column("over"), plan.exprs.column("qty")],
        [False, True],
        [True, True],
    )
    var root = plan.limit(sorted, 0, 5)
    _ = limits(plan, root, schemas())
    var out = run(plan, root)

    same(read_back(out, "qty"), [8, 5, 3, 1, 40], "the second key inside")


def test_a_bound_wider_than_the_table_is_the_whole_sort() raises:
    # A bound that bounds nothing has to give every row back, since the limit
    # above it is not cutting anything either.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var sorted = plan.sort(scan, [plan.exprs.column("qty")], [False], [True])
    var root = plan.limit(sorted, 0, 50)
    _ = limits(plan, root, schemas())
    var out = run(plan, root)

    same(
        read_back(out, "qty"),
        [1, 3, 5, 8, 12, 15, 20, 25, 30, 40],
        "ascending, all ten",
    )


def test_a_sort_the_pass_never_saw_still_sorts_everything() raises:
    # The bound is an opportunity, so a plan that skipped the limit pass has to
    # lower to a sort of the whole table and answer the same way.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var sorted = plan.sort(scan, [plan.exprs.column("qty")], [True], [True])
    var root = plan.limit(sorted, 0, 3)
    assert_equal(plan.nodes[sorted].length, NO_LIMIT, "no bound was written")
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
