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

Part 1 of 3. The fixtures are in tests/support/plan_lower.mojo.
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


def test_a_comparison_against_a_constant_lowers_to_one_filter() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var ten = plan.exprs.literal(Value(Int64(10)))
    var root = plan.filter(scan, plan.exprs.binary(BinaryOp.GT, qty, ten))

    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())

    # One operator, where this was a compare and a filter. The mask the compare
    # wrote was read by the filter above it and by nobody else, so the filter
    # carries the comparison instead and the column is never written.
    assert_equal(len(pipe.operators), 1, "operators")

    var out = pipe^.run()
    var got = read_back(out, "qty")
    assert_equal(len(got), 6, "rows kept")
    assert_equal(got[0], 20, "first")
    assert_equal(got[5], 15, "last")


def test_a_constant_on_the_left_lowers_to_the_mirrored_comparison() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var ten = plan.exprs.literal(Value(Int64(10)))
    var root = plan.filter(scan, plan.exprs.binary(BinaryOp.LT, ten, qty))

    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())
    assert_equal(len(pipe.operators), 1, "operators")

    # `10 < qty` is `qty > 10`, so it is the same six rows as the test above and
    # not the four the unmirrored reading would keep.
    var out = pipe^.run()
    var got = read_back(out, "qty")
    assert_equal(len(got), 6, "rows kept")
    assert_equal(got[0], 20, "first")
    assert_equal(got[5], 15, "last")


def test_a_comparison_over_an_expression_still_lowers_the_expression() raises:
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var product = plan.exprs.binary(BinaryOp.MUL, qty, price)
    var hundred = plan.exprs.literal(Value(Int64(100)))
    var root = plan.filter(
        scan, plan.exprs.binary(BinaryOp.GE, product, hundred)
    )

    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())

    # The multiply is still an operator, because it is an expression and not a
    # comparison, and the filter reads the column it wrote and compares that
    # column itself. Two operators where there were three, and the product is
    # dropped by the filter as it writes rather than by a projection.
    assert_equal(len(pipe.operators), 2, "operators")

    var out = pipe^.run()
    # The products are 50, 40, 21, 40, 60, 72, 75, 100, 120 and 90, so two
    # reach a hundred.
    var got = read_back(out, "qty")
    assert_equal(len(got), 2, "rows kept")
    assert_equal(got[0], 1, "the row whose product is exactly a hundred")
    assert_equal(got[1], 30, "the row whose product is a hundred and twenty")


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

    # One filter for each half and nothing else. Each half is a comparison
    # against a constant, so the filter does the comparison itself and no mask
    # column is written, and there is no projection afterwards because there is
    # nothing left over to drop. Computing an and mask would have been two
    # operators as well, but both comparisons would have run on all ten rows.
    assert_equal(len(pipe.operators), 2, "operators")

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
    assert_equal(len(pipe.operators), 3, "one filter per part, comparing")

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

    # The literal true dropped out in the pass, so what is left is one filter
    # doing its own comparison, which puts the schema back itself.
    assert_equal(len(pipe.operators), 1, "operators")
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


def test_a_chain_of_three_ors_is_one_operator_over_three_columns() raises:
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

    # Three comparisons, one connective reading all three of them, and the
    # filter, which drops all four intermediates as it writes the way it drops
    # one. Folded into pairs this was six, and the extra node existed only to
    # write a column for the next one to read straight back.
    assert_equal(len(pipe.operators), 5, "operators")

    var out = pipe^.run()
    same(read_back(out, "qty"), [3, 40, 25, 30], "the rows any arm names")


def test_a_disjunction_of_equalities_is_one_set_lookup() raises:
    # `qty IN (3, 25, 40)` as SQL writes it out, which is an equality per
    # member joined by `or`. Read back it is a set, and a set is one pass
    # rather than a comparison per member and a disjunction over all of them.
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
        BinaryOp.EQ, qty, plan.exprs.literal(Value(Int64(40)))
    )
    var any = plan.exprs.call(String("or"), [a, b, c], rowwise=True)
    var root = plan.filter(scan, any)

    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())

    # The lookup and the filter. As a chain it is four, three of which write a
    # boolean column that only the fourth ever reads.
    assert_equal(len(pipe.operators), 2, "operators")

    var out = pipe^.run()
    same(read_back(out, "qty"), [3, 40, 25], "the rows the set names")


def test_a_set_of_two_is_a_lookup_as_well() raises:
    # Two is the smallest disjunction anybody can write and it is still three
    # nodes against one, so there is no width below which the chain wins and no
    # threshold here to find.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var a = plan.exprs.binary(
        BinaryOp.EQ, qty, plan.exprs.literal(Value(Int64(20)))
    )
    var b = plan.exprs.binary(
        BinaryOp.EQ, qty, plan.exprs.literal(Value(Int64(1)))
    )
    var root = plan.filter(
        scan, plan.exprs.call(String("or"), [a, b], rowwise=True)
    )

    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())

    assert_equal(len(pipe.operators), 2, "operators")
    var out = pipe^.run()
    same(read_back(out, "qty"), [20, 1], "the rows the pair names")


def test_a_set_with_a_null_in_it_stays_a_chain_of_equalities() raises:
    # `x = NULL` is null and not false, so a disjunction holding one answers
    # null for every row the other arms miss. A set lookup answers false there,
    # because a null is not a member of anything, and the two are not the same
    # predicate. Both come out to the same rows under a filter, and that is the
    # coincidence this guards against relying on.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var a = plan.exprs.binary(
        BinaryOp.EQ, qty, plan.exprs.literal(Value(Int64(3)))
    )
    var b = plan.exprs.binary(
        BinaryOp.EQ,
        qty,
        plan.exprs.literal(Value(null=LogicalType.INT64)),
    )
    var root = plan.filter(
        scan, plan.exprs.call(String("or"), [a, b], rowwise=True)
    )

    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())

    assert_equal(len(pipe.operators), 4, "operators")
    var out = pipe^.run()
    same(read_back(out, "qty"), [3], "the one row the present arm names")


def test_a_disjunction_over_two_columns_stays_a_chain() raises:
    # A set is a set of values one column is looked up in. Two columns is two
    # lookups and a disjunction over them, which is what it already was.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var price = plan.exprs.column("price")
    var a = plan.exprs.binary(
        BinaryOp.EQ, qty, plan.exprs.literal(Value(Int64(3)))
    )
    var b = plan.exprs.binary(
        BinaryOp.EQ, price, plan.exprs.literal(Value(Int64(4)))
    )
    var root = plan.filter(
        scan, plan.exprs.call(String("or"), [a, b], rowwise=True)
    )

    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())

    assert_equal(len(pipe.operators), 4, "operators")
    var out = pipe^.run()
    same(read_back(out, "qty"), [3, 30], "the rows either column names")


def test_a_constant_the_column_cannot_hold_stays_a_chain() raises:
    # `qty = 3.5` is false for every row of an integer column. Held in a set of
    # integers it would become `qty = 3`, which is true for a row, so the set is
    # built only when every constant comes back the number it went in as.
    var plan = Plan()
    var scan = plan.scan("sales", List[String](), 0)
    var qty = plan.exprs.column("qty")
    var a = plan.exprs.binary(
        BinaryOp.EQ, qty, plan.exprs.literal(Value(Float64(3.5)))
    )
    var b = plan.exprs.binary(
        BinaryOp.EQ, qty, plan.exprs.literal(Value(Float64(5.5)))
    )
    var root = plan.filter(
        scan, plan.exprs.call(String("or"), [a, b], rowwise=True)
    )

    _ = bind(plan, root, schemas())
    var pipe = lower(plan, root, one_frame())

    assert_equal(len(pipe.operators), 4, "operators")
    var out = pipe^.run()
    assert_equal(out.rows, 0, "no row equals either")


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


def test_three_operands_follow_the_same_rule_as_two() raises:
    var plan = Plan()
    var scan = plan.scan("gauges", List[String](), 0)
    var zero = plan.exprs.literal(Value(Int64(0)))
    var five = plan.exprs.literal(Value(Int64(5)))
    var a = plan.exprs.binary(BinaryOp.GT, plan.exprs.column("a"), zero)
    var b = plan.exprs.binary(BinaryOp.GT, plan.exprs.column("b"), zero)
    var big = plan.exprs.binary(BinaryOp.GT, plan.exprs.column("a"), five)
    var either = plan.exprs.call(String("or"), [a, b, big], rowwise=True)
    var all_of = plan.exprs.call(String("and"), [a, b, big], rowwise=True)
    var root = plan.project(scan, [either, all_of], ["either", "all"])

    _ = bind(plan, root, gauge_schemas())
    var pipe = lower(plan, root, gauge_frame())
    var out = pipe^.run()

    # One node reads all three, and the rule it applies is the one the pairwise
    # chain applied: a single true settles the or wherever the nulls fall, a
    # single false settles the and, and a row every operand is null on stays
    # null. The third operand is false where it is present, so it decides the
    # and on the last row that the other two agreed was true.
    same(truths(out, "either"), [-1, 1, -1, -1, 1], "either")
    same(truths(out, "all"), [0, -1, 0, -1, 0], "all")


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
