"""Tests for transitive predicates across equality joins.

The rule lives inside predicate pushdown, so these run `push` and read the plan
it writes. That is the right level to test it at anyway: what the rule is for is
the predicate arriving at the other side's scan, and only pushdown can put it
there.

The groups are: the copy happens, the copy is right, the copy does not happen,
and the copy happens once.
"""

from std.testing import TestSuite, assert_equal, assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.plan.node import Plan
from firepanda.plan.print import explain
from firepanda.plan.push import push


def _lineitem() -> Schema:
    """Returns a schema shaped like a cut down TPC-H lineitem table.

    Returns:
        Three columns, the key not nullable.
    """
    var out = Schema()
    out.append(Field("l_orderkey", LogicalType.INT64, False))
    out.append(Field("l_quantity", LogicalType.FLOAT64, True))
    out.append(Field("l_extendedprice", LogicalType.FLOAT64, True))
    return out^


def _orders() -> Schema:
    """Returns a schema shaped like a cut down TPC-H orders table.

    Returns:
        Three columns, the key not nullable.
    """
    var out = Schema()
    out.append(Field("o_orderkey", LogicalType.INT64, False))
    out.append(Field("o_custkey", LogicalType.INT64, False))
    out.append(Field("o_totalprice", LogicalType.FLOAT64, True))
    return out^


def _both() -> List[Schema]:
    """Returns the two schemas, in the order the scans name them."""
    var out = List[Schema]()
    out.append(_lineitem())
    out.append(_orders())
    return out^


def _joined(mut plan: Plan) raises -> Int:
    """Returns the two tables joined on their order key."""
    return plan.join(
        plan.scan("lineitem", List[String](), 0),
        plan.scan("orders", List[String](), 1),
        [plan.exprs.column("l_orderkey")],
        [plan.exprs.column("o_orderkey")],
        JoinKind.INNER,
    )


def _small(mut plan: Plan, name: String, than: Int) raises -> Int:
    """Returns a predicate that one column is below one number."""
    return plan.exprs.binary(
        BinaryOp.LT,
        plan.exprs.column(name),
        plan.exprs.literal(Value(Int64(than))),
    )


def _filters(plan: Plan, root: Int) raises -> Int:
    """Counts the filter nodes in the printed plan."""
    var out = 0
    for line in explain(plan, root).split("\n"):
        if line.strip().startswith("FILTER"):
            out += 1
    return out


def test_a_filter_on_one_key_reaches_the_other() raises:
    var plan = Plan()
    var root = plan.filter(_joined(plan), _small(plan, "l_orderkey", 100))
    var at = push(plan, root, _both())
    var printed = explain(plan, at)
    # The caller only said it about the line items. It is true of the orders
    # too, for every order that is going to survive the join.
    assert_true("o_orderkey < 100" in printed, "the orders are filtered")
    assert_true("l_orderkey < 100" in printed, "and so are the line items")
    assert_equal(_filters(plan, at), 2, "one above each scan")


def test_it_works_the_other_way_round() raises:
    var plan = Plan()
    var root = plan.filter(_joined(plan), _small(plan, "o_orderkey", 100))
    var at = push(plan, root, _both())
    assert_true(
        "l_orderkey < 100" in explain(plan, at), "the line items are filtered"
    )


def test_the_copy_lands_under_the_join() raises:
    var plan = Plan()
    var root = plan.filter(_joined(plan), _small(plan, "l_orderkey", 100))
    var at = push(plan, root, _both())
    var printed = explain(plan, at)
    # A predicate that arrives above the join and is not pushed below it saves
    # nothing, so where it ends up is the whole point.
    assert_true(
        printed.find("JOIN") < printed.find("o_orderkey < 100"),
        "the copy is inside the join's right arm",
    )


def test_a_predicate_already_at_the_scan_still_crosses() raises:
    var plan = Plan()
    var lines = plan.filter(
        plan.scan("lineitem", List[String](), 0),
        _small(plan, "l_orderkey", 100),
    )
    var root = plan.join(
        lines,
        plan.scan("orders", List[String](), 1),
        [plan.exprs.column("l_orderkey")],
        [plan.exprs.column("o_orderkey")],
        JoinKind.INNER,
    )
    var at = push(plan, root, _both())
    # By the second sweep of the pipeline everything has been pushed already, so
    # a rule that only read what was still above the join would fire once and
    # then never again.
    assert_true("o_orderkey < 100" in explain(plan, at), "found in the arm")


def test_a_whole_conjunction_crosses_a_piece_at_a_time() raises:
    var plan = Plan()
    var whole = plan.exprs.call(
        String("and"),
        [_small(plan, "l_orderkey", 100), _small(plan, "l_quantity", 24)],
        rowwise=True,
    )
    var root = plan.filter(_joined(plan), whole)
    var at = push(plan, root, _both())
    var printed = explain(plan, at)
    assert_true("o_orderkey < 100" in printed, "the half that is about the key")
    assert_true("l_quantity < 24" in printed, "the half that is not stayed put")
    assert_true("o_quantity" not in printed, "and did not cross")


def test_a_predicate_on_a_column_that_is_not_a_key_does_not_cross() raises:
    var plan = Plan()
    var root = plan.filter(_joined(plan), _small(plan, "l_quantity", 24))
    var at = push(plan, root, _both())
    assert_equal(_filters(plan, at), 1, "nothing to say about the orders")


def test_a_predicate_reading_two_columns_does_not_cross() raises:
    var plan = Plan()
    var root = plan.filter(
        _joined(plan),
        plan.exprs.binary(
            BinaryOp.LT,
            plan.exprs.column("l_orderkey"),
            plan.exprs.column("l_quantity"),
        ),
    )
    var at = push(plan, root, _both())
    # The equality says what the other side's key is and says nothing about the
    # other column, so half of this has no translation.
    assert_equal(_filters(plan, at), 1, "the line items only")


def test_an_outer_join_does_not_cross() raises:
    var plan = Plan()
    var joined = plan.join(
        plan.scan("lineitem", List[String](), 0),
        plan.scan("orders", List[String](), 1),
        [plan.exprs.column("l_orderkey")],
        [plan.exprs.column("o_orderkey")],
        JoinKind.LEFT,
    )
    var root = plan.filter(joined, _small(plan, "l_orderkey", 100))
    var at = push(plan, root, _both())
    # An outer join invents rows where one side had none, so filtering the side
    # that was going to be invented changes which rows get invented.
    assert_true("o_orderkey < 100" not in explain(plan, at), "left alone")


def test_a_key_that_is_not_a_column_does_not_cross() raises:
    var plan = Plan()
    var joined = plan.join(
        plan.scan("lineitem", List[String](), 0),
        plan.scan("orders", List[String](), 1),
        [
            plan.exprs.binary(
                BinaryOp.ADD,
                plan.exprs.column("l_orderkey"),
                plan.exprs.literal(Value(Int64(1))),
            )
        ],
        [plan.exprs.column("o_orderkey")],
        JoinKind.INNER,
    )
    var root = plan.filter(joined, _small(plan, "l_orderkey", 100))
    var at = push(plan, root, _both())
    # The equality is about the sum and not about the column, so a predicate on
    # the column says nothing about the other side.
    assert_equal(_filters(plan, at), 1, "the line items only")


def test_a_predicate_the_other_side_already_has_is_not_copied() raises:
    var plan = Plan()
    var whole = plan.exprs.call(
        String("and"),
        [_small(plan, "l_orderkey", 100), _small(plan, "o_orderkey", 100)],
        rowwise=True,
    )
    var root = plan.filter(_joined(plan), whole)
    var at = push(plan, root, _both())
    var printed = explain(plan, at)
    assert_equal(_filters(plan, at), 2, "one above each scan and no more")
    assert_equal(
        printed.count("o_orderkey < 100"), 1, "said once about the orders"
    )


def test_running_it_again_copies_nothing_further() raises:
    var plan = Plan()
    var root = plan.filter(_joined(plan), _small(plan, "l_orderkey", 100))
    var once = push(plan, root, _both())
    var printed = explain(plan, once)
    var twice = push(plan, once, _both())
    # Without the check against what the other side already asks, this would
    # copy left to right on one run and right to left on the next, forever.
    assert_equal(explain(plan, twice), printed, "settled after the first run")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
