"""Tests for the ordering of a comma FROM that was written with a product in it.

Every one of these builds the chain a comma `FROM` lowers to, which is a left
deep run of cross joins under one filter, and asks what the pass did with it.
The printed plan is what most of them read, for the reason the pushdown tests
give: which relation ended up next to which is a shape and a printed plan is the
one form of a shape a reader can check without holding the arena in their head.

The groups are: a chain the pass reorders, the chains it leaves exactly as they
were and why each one of those is not a bug, the self join where a qualifier
decides which relation a column belongs to, and the two together with predicate
pushdown, which is the only pair of them that says what the pass is for.
"""

from std.testing import TestSuite, assert_equal, assert_raises
from std.testing import assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.plan.node import Plan
from firepanda.plan.order import order
from firepanda.plan.print import explain
from firepanda.plan.push import push


def _part() -> Schema:
    """Returns a schema shaped like a cut down TPC-H part table.

    Returns:
        Three columns, the key not nullable.
    """
    var out = Schema()
    out.append(Field("p_partkey", LogicalType.INT64, False))
    out.append(Field("p_name", LogicalType.STRING, True))
    out.append(Field("p_size", LogicalType.INT32, True))
    return out^


def _supplier() -> Schema:
    """Returns a schema shaped like a cut down TPC-H supplier table.

    Returns:
        Three columns, the key not nullable.
    """
    var out = Schema()
    out.append(Field("s_suppkey", LogicalType.INT64, False))
    out.append(Field("s_name", LogicalType.STRING, True))
    out.append(Field("s_nationkey", LogicalType.INT64, True))
    return out^


def _lineitem() -> Schema:
    """Returns a schema shaped like a cut down TPC-H lineitem table.

    Returns:
        Four columns, the two keys not nullable.
    """
    var out = Schema()
    out.append(Field("l_partkey", LogicalType.INT64, False))
    out.append(Field("l_suppkey", LogicalType.INT64, False))
    out.append(Field("l_quantity", LogicalType.FLOAT64, True))
    out.append(Field("l_shipmode", LogicalType.STRING, True))
    return out^


def _nation() -> Schema:
    """Returns a schema shaped like the TPC-H nation table.

    Returns:
        Two columns, the key not nullable.
    """
    var out = Schema()
    out.append(Field("n_nationkey", LogicalType.INT64, False))
    out.append(Field("n_name", LogicalType.STRING, True))
    return out^


def _three() -> List[Schema]:
    """Returns the schemas of part, supplier and lineitem in that order.

    Returns:
        Three schemas, indexed by the relation id the scans carry.
    """
    return [_part(), _supplier(), _lineitem()]


def _same(mut plan: Plan, one: String, other: String) raises -> Int:
    """Returns a predicate that two columns are equal."""
    return plan.exprs.binary(
        BinaryOp.EQ, plan.exprs.column(one), plan.exprs.column(other)
    )


def _all(mut plan: Plan, one: Int, other: Int) raises -> Int:
    """Returns the conjunction of two predicates."""
    var parts = List[Int]()
    parts.append(one)
    parts.append(other)
    return plan.exprs.call(String("and"), parts^, rowwise=True)


def _cross(mut plan: Plan, left: Int, right: Int) raises -> Int:
    """Returns a join with no condition, which is what a comma lowers to."""
    return plan.join(left, right, List[Int](), List[Int](), JoinKind.CROSS)


def _crosses(plan: Plan, root: Int) -> Int:
    """Counts the cross joins the root reaches."""
    var out = 0
    for at in range(root + 1):
        if String(plan.nodes[at].kind) != "JOIN":
            continue
        if plan.nodes[at].op == Int(JoinKind.CROSS.code):
            out += 1
    return out


def _written(mut plan: Plan) raises -> Int:
    """Returns the plan TPC-H q9 writes in miniature.

    `FROM part, supplier, lineitem WHERE p_partkey = l_partkey AND s_suppkey =
    l_suppkey`, which pairs every part with every supplier before either of them
    meets the table both of them have an equality with.

    Args:
        plan: The plan, added to.

    Returns:
        The filter at the top of it.
    """
    var part = plan.scan("part", List[String](), 0)
    var supplier = plan.scan("supplier", List[String](), 1)
    var lineitem = plan.scan("lineitem", List[String](), 2)
    var chain = _cross(plan, _cross(plan, part, supplier), lineitem)
    return plan.filter(
        chain,
        _all(
            plan,
            _same(plan, "p_partkey", "l_partkey"),
            _same(plan, "s_suppkey", "l_suppkey"),
        ),
    )


def test_a_relation_with_nothing_to_join_on_is_moved_out_of_the_middle() raises:
    var plan = Plan()
    var root = _written(plan)
    var at = order(plan, root, _three())
    var printed = explain(plan, at)
    # Read from the bottom, the lowest join now has the two relations that have
    # an equality between them, and the supplier joins what they produced.
    assert_true(
        "SCAN part" in printed and "SCAN lineitem" in printed,
        "both relations are still read",
    )
    assert_true(
        printed.find("SCAN lineitem") < printed.find("SCAN supplier"),
        "and the lineitem is now the one written next to the part",
    )


def test_an_order_with_no_product_in_it_already_is_left_where_it_was() raises:
    var plan = Plan()
    var part = plan.scan("part", List[String](), 0)
    var lineitem = plan.scan("lineitem", List[String](), 2)
    var supplier = plan.scan("supplier", List[String](), 1)
    var chain = _cross(plan, _cross(plan, part, lineitem), supplier)
    var root = plan.filter(
        chain,
        _all(
            plan,
            _same(plan, "p_partkey", "l_partkey"),
            _same(plan, "s_suppkey", "l_suppkey"),
        ),
    )
    var before = explain(plan, root)
    var at = order(plan, root, _three())
    assert_equal(at, root, "nothing was rewritten, so the root is the old one")
    assert_equal(explain(plan, at), before, "and the plan is the old plan")


def test_a_from_that_really_does_ask_for_a_product_is_left_as_written() raises:
    var plan = Plan()
    var part = plan.scan("part", List[String](), 0)
    var supplier = plan.scan("supplier", List[String](), 1)
    var lineitem = plan.scan("lineitem", List[String](), 2)
    var chain = _cross(plan, _cross(plan, part, supplier), lineitem)
    var root = plan.filter(chain, _same(plan, "p_partkey", "l_partkey"))
    var before = explain(plan, root)
    var at = order(plan, root, _three())
    # The supplier has an equality with nothing, so every order of these three
    # has a product in it and the one the query wrote is as good as any. The
    # refusal the query gets is the one it already had.
    assert_equal(explain(plan, at), before, "the plan is the old plan")


def test_two_relations_are_one_join_and_are_never_swapped() raises:
    var plan = Plan()
    var part = plan.scan("part", List[String](), 0)
    var lineitem = plan.scan("lineitem", List[String](), 2)
    var root = plan.filter(
        _cross(plan, part, lineitem), _same(plan, "p_partkey", "l_partkey")
    )
    var before = explain(plan, root)
    var at = order(plan, root, _three())
    assert_equal(explain(plan, at), before, "the plan is the old plan")


def test_a_join_the_query_wrote_a_condition_on_is_not_part_of_a_chain() raises:
    var plan = Plan()
    var part = plan.scan("part", List[String](), 0)
    var supplier = plan.scan("supplier", List[String](), 1)
    var lineitem = plan.scan("lineitem", List[String](), 2)
    # `FROM part JOIN supplier ON ... , lineitem`, so the lower join is not one
    # this may take apart and the chain above it is one relation long.
    var keyed = plan.join(
        part,
        supplier,
        [plan.exprs.column("p_size")],
        [plan.exprs.column("s_nationkey")],
        JoinKind.INNER,
    )
    var chain = _cross(plan, keyed, lineitem)
    var root = plan.filter(
        chain,
        _all(
            plan,
            _same(plan, "p_partkey", "l_partkey"),
            _same(plan, "s_suppkey", "l_suppkey"),
        ),
    )
    var before = explain(plan, root)
    var at = order(plan, root, _three())
    assert_equal(explain(plan, at), before, "the plan is the old plan")


def test_an_equality_that_is_not_two_plain_columns_ties_nothing() raises:
    var plan = Plan()
    var root = _written(plan)
    # The same two equalities, except that one of them compares a column with a
    # number, which is a filter on one relation rather than an edge between two.
    var plan_two = Plan()
    var part = plan_two.scan("part", List[String](), 0)
    var supplier = plan_two.scan("supplier", List[String](), 1)
    var lineitem = plan_two.scan("lineitem", List[String](), 2)
    var chain = _cross(plan_two, _cross(plan_two, part, supplier), lineitem)
    var number = plan_two.exprs.binary(
        BinaryOp.EQ,
        plan_two.exprs.column("s_suppkey"),
        plan_two.exprs.literal(Value(7)),
    )
    var second = plan_two.filter(
        chain,
        _all(plan_two, _same(plan_two, "p_partkey", "l_partkey"), number),
    )
    var before = explain(plan_two, second)
    var at = order(plan_two, second, _three())
    assert_equal(explain(plan_two, at), before, "the plan is the old plan")
    # And the first one, which differs only in that second equality, does move.
    var moved = order(plan, root, _three())
    assert_true(
        explain(plan, moved) != before, "where two plain columns tie two sides"
    )


def _self_join(mut plan: Plan, qualified: Bool) raises -> Int:
    """Returns a chain that reads one table twice, with a product in the middle.

    `FROM nation n1, nation n2, supplier WHERE s_nationkey = n1.n_nationkey AND
    s_suppkey = n2.n_nationkey`, which pairs every nation with every nation
    before either of them meets the relation they both have an equality with.
    Both equalities read a column called `n_nationkey` and there are two of
    those, so which relation each of them ties is a question the name cannot
    answer and only the qualifier can.

    Args:
        plan: The plan, added to.
        qualified: Whether the two references say which nation they meant.

    Returns:
        The filter at the top of it.
    """
    var first = plan.scan("nation", List[String](), 0)
    var second = plan.scan("nation", List[String](), 1)
    var supplier = plan.scan("supplier", List[String](), 2)
    var chain = _cross(plan, _cross(plan, first, second), supplier)
    var here = plan.exprs.column("n_nationkey")
    var there = plan.exprs.column("n_nationkey")
    if qualified:
        plan.exprs.nodes[here].table = 0
        plan.exprs.nodes[there].table = 1
    return plan.filter(
        chain,
        _all(
            plan,
            plan.exprs.binary(
                BinaryOp.EQ, plan.exprs.column("s_nationkey"), here
            ),
            plan.exprs.binary(
                BinaryOp.EQ, plan.exprs.column("s_suppkey"), there
            ),
        ),
    )


def test_a_qualified_column_of_a_self_join_names_the_relation_it_reads() raises:
    var plan = Plan()
    var root = _self_join(plan, qualified=True)
    var at = order(plan, root, [_nation(), _nation(), _supplier()])
    at = push(plan, at, [_nation(), _nation(), _supplier()])
    assert_equal(_crosses(plan, at), 0, "no product is left in the plan")


def test_an_unqualified_column_of_a_self_join_is_refused_before_this() raises:
    var plan = Plan()
    var root = _self_join(plan, qualified=False)
    # Which is why the reordering never has to guess. A name two relations both
    # hand out is refused where the plan is bound, one pass before this one, so
    # every column reference that reaches here either says which relation it
    # meant or is the only one of its name.
    with assert_raises(contains="is the name of more than one column here"):
        _ = order(plan, root, [_nation(), _nation(), _supplier()])


def test_the_order_the_pass_chose_is_one_the_pushdown_can_key() raises:
    var plan = Plan()
    var root = _written(plan)
    var at = order(plan, root, _three())
    at = push(plan, at, _three())
    # The point of the whole pass. Written as it was, the lower join is a
    # product of every part against every supplier and the operator refuses to
    # run it. Reordered, both joins have keys on them.
    assert_equal(_crosses(plan, at), 0, "no product is left in the plan")
    assert_true("JOIN inner" in explain(plan, at), "and the joins are keyed")


def test_the_pass_finds_nothing_to_do_on_a_plan_it_has_been_over() raises:
    var plan = Plan()
    var root = _written(plan)
    var once = order(plan, root, _three())
    var before = explain(plan, once)
    var twice = order(plan, once, _three())
    assert_equal(twice, once, "the second run rewrote nothing")
    assert_equal(explain(plan, twice), before, "so the plan is unchanged")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
