"""Tests for empty and constant pruning.

These read the printed plan, because what the pass does is take nodes out and a
printed plan is the shape of what is left. The counting helper is here for the
two tests where the question is how many of one kind survived rather than which.

The groups are: the two constant predicates, what collapses over something
empty, what does not, and the root, which is the one node with nobody above it
to be told that it went.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.plan.empty import empty
from firepanda.plan.node import Plan
from firepanda.plan.print import explain


def _lineitem() -> Schema:
    """Returns a schema shaped like a cut down TPC-H lineitem table.

    Returns:
        Four columns, the key not nullable.
    """
    var out = Schema()
    out.append(Field("l_orderkey", LogicalType.INT64, False))
    out.append(Field("l_quantity", LogicalType.FLOAT64, True))
    out.append(Field("l_extendedprice", LogicalType.FLOAT64, True))
    out.append(Field("l_discount", LogicalType.FLOAT64, True))
    return out^


def _lines(plan: Plan, root: Int, word: String) raises -> Int:
    """Counts how many lines of the printed plan start with a word."""
    var out = 0
    for line in explain(plan, root).split("\n"):
        if line.strip().startswith(word):
            out += 1
    return out


def _sorted(mut plan: Plan, over: Int) raises -> Int:
    """Returns the input ordered by one column."""
    return plan.sort(over, [plan.exprs.column("l_orderkey")], [True], [True])


def test_a_filter_that_keeps_nothing_becomes_a_limit_of_zero() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.filter(scan, plan.exprs.literal(Value(False)))
    var at = empty(plan, root, [_lineitem()])
    var printed = explain(plan, at)
    assert_true("LIMIT 0" in printed, "nothing comes out of it")
    assert_true("FILTER" not in printed, "and it is not a filter any more")
    assert_true("SCAN" in printed, "over the input it had")


def test_a_filter_that_keeps_everything_goes() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var kept = plan.filter(scan, plan.exprs.literal(Value(True)))
    var root = plan.project(
        kept, [plan.exprs.column("l_quantity")], ["l_quantity"]
    )
    var at = empty(plan, root, [_lineitem()])
    var printed = explain(plan, at)
    assert_true("FILTER" not in printed, "a filter that filters nothing")
    assert_true("PROJECT" in printed, "and the projection reads the scan")
    assert_equal(_lines(plan, at, "SCAN"), 1, "which is still there")


def test_the_schema_is_what_it_was() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.filter(scan, plan.exprs.literal(Value(False)))
    var at = empty(plan, root, [_lineitem()])
    var printed = explain(plan, at)
    # A limit of zero over the same input is an empty relation with the right
    # schema, which is the whole reason it is the shape this pass makes.
    assert_true("LIMIT 0" in printed, "empty")
    assert_equal(_lines(plan, at, "SCAN"), 1, "and the columns still come from")


def test_a_sort_over_nothing_goes() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var none = plan.filter(scan, plan.exprs.literal(Value(False)))
    var root = _sorted(plan, none)
    var at = empty(plan, root, [_lineitem()])
    var printed = explain(plan, at)
    assert_true("SORT" not in printed, "nothing to put in order")
    assert_true("LIMIT 0" in printed, "and what is left is the emptiness")


def test_a_distinct_over_nothing_goes() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var none = plan.filter(scan, plan.exprs.literal(Value(False)))
    var root = plan.distinct(none, List[Int]())
    var at = empty(plan, root, [_lineitem()])
    assert_true("DISTINCT" not in explain(plan, at), "no rows to be distinct")


def test_a_filter_over_nothing_goes() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var none = plan.filter(scan, plan.exprs.literal(Value(False)))
    var root = plan.filter(
        none,
        plan.exprs.binary(
            BinaryOp.GT,
            plan.exprs.column("l_quantity"),
            plan.exprs.literal(Value(Int64(5))),
        ),
    )
    var at = empty(plan, root, [_lineitem()])
    assert_true("FILTER" not in explain(plan, at), "nothing left to throw away")


def test_a_limit_over_nothing_goes() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var none = plan.filter(scan, plan.exprs.literal(Value(False)))
    var root = plan.limit(none, 0, 10)
    var at = empty(plan, root, [_lineitem()])
    assert_equal(_lines(plan, at, "LIMIT"), 1, "the first ten of nothing")


def test_a_line_of_them_collapses_in_one_call() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var none = plan.filter(scan, plan.exprs.literal(Value(False)))
    var ordered = _sorted(plan, none)
    var once = plan.distinct(ordered, List[Int]())
    var root = plan.limit(once, 0, 10)
    var at = empty(plan, root, [_lineitem()])
    var printed = explain(plan, at)
    assert_equal(_lines(plan, at, "LIMIT"), 1, "one node left above the scan")
    assert_true("SORT" not in printed, "the sort went")
    assert_true("DISTINCT" not in printed, "and so did the distinct")


def test_a_projection_over_nothing_is_left_alone() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var none = plan.filter(scan, plan.exprs.literal(Value(False)))
    var root = plan.project(
        none, [plan.exprs.column("l_quantity")], ["l_quantity"]
    )
    var at = empty(plan, root, [_lineitem()])
    # A projection changes the schema, so it cannot be taken out from above an
    # empty input without taking the schema with it. It costs nothing anyway,
    # since it is a projection over no rows.
    assert_true("PROJECT" in explain(plan, at), "the projection stayed")


def test_an_aggregate_over_nothing_is_left_alone() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var none = plan.filter(scan, plan.exprs.literal(Value(False)))
    var root = plan.aggregate(
        none,
        List[Int](),
        [
            plan.exprs.aggregate(
                AggKind.SUM, plan.exprs.column("l_extendedprice")
            )
        ],
        ["revenue"],
    )
    var at = empty(plan, root, [_lineitem()])
    # A whole frame reduction over no rows answers one row rather than none, so
    # this is not a case of the same answer written shorter.
    assert_true("AGGREGATE" in explain(plan, at), "the reduction stayed")


def test_a_predicate_that_is_not_constant_is_left_alone() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.filter(
        scan,
        plan.exprs.binary(
            BinaryOp.GT,
            plan.exprs.column("l_quantity"),
            plan.exprs.literal(Value(Int64(5))),
        ),
    )
    var before = explain(plan, root)
    var at = empty(plan, root, [_lineitem()])
    assert_equal(explain(plan, at), before, "nothing to say about it")
    assert_equal(at, root, "and the root did not move")


def test_a_predicate_that_is_null_is_left_alone() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.filter(
        scan, plan.exprs.literal(Value(null=LogicalType.BOOL))
    )
    var at = empty(plan, root, [_lineitem()])
    # A null keeps no rows, so this could be the limit of zero, but saying so
    # here would be a rule about what a null does in a predicate and the place
    # for that one is the kernel that already has it.
    assert_true("FILTER" in explain(plan, at), "the filter stayed")
    assert_equal(at, root, "and the root did not move")


def test_a_root_that_keeps_everything_hands_back_its_input() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var root = plan.filter(scan, plan.exprs.literal(Value(True)))
    var at = empty(plan, root, [_lineitem()])
    # Nobody is above the root to be told it went, which is why the pass hands
    # back a node rather than a schema.
    assert_equal(at, scan, "the scan is the answer now")
    assert_true("FILTER" not in explain(plan, at), "and the filter is gone")


def test_a_root_over_nothing_hands_back_the_emptiness() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var none = plan.filter(scan, plan.exprs.literal(Value(False)))
    var root = _sorted(plan, none)
    var at = empty(plan, root, [_lineitem()])
    assert_equal(at, none, "the limit of zero is the answer")


def test_running_it_again_changes_nothing() raises:
    var plan = Plan()
    var scan = plan.scan("lineitem", List[String](), 0)
    var none = plan.filter(scan, plan.exprs.literal(Value(False)))
    var ordered = _sorted(plan, none)
    var root = plan.limit(ordered, 0, 10)
    var once = empty(plan, root, [_lineitem()])
    var printed = explain(plan, once)
    var twice = empty(plan, once, [_lineitem()])
    assert_equal(explain(plan, twice), printed, "settled after the first run")
    assert_equal(twice, once, "and the root stayed where it was")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
