"""Join ordering, the half of it that is not about cost: no needless product.

Section 10 of `docs/specs/planner/02-the-pass-pipeline.md` and the whole of
`docs/specs/planner/03-join-ordering.md`. That document puts ordering third in
line behind the two pushdowns, says greedy is enough for the case we actually
hit, and says not to build a cardinality estimator in order to build an orderer.
This pass is the part of the job that needs no estimator at all.

## What it is for

`FROM a, b, c WHERE a.x = c.x AND b.y = c.y` is a join written the older way. A
comma is a join with no condition, the clause nests left, so it lowers to `a`
crossed with `b` crossed with `c` under a filter, and predicate pushdown then
puts each equality on the join that can carry it. Neither equality can go on the
lower join, because `a` and `b` have nothing between them, so that one stays a
product of every row of `a` against every row of `b`.

A product of two real tables is not slow, it is refused: the operator pairs a
whole frame against a right side of a single row and says so when the right side
has more rows than that. So the query does not run. Written as `FROM a, c, b` it
runs, and the two spellings are the same query. That is the gap this closes.

TPC-H q9 is the case. Its `FROM` is `part, supplier, lineitem, partsupp, orders,
nation` and every one of its six equalities is against `lineitem`, which is
third, so the first join in the chain pairs every part with every supplier and
nothing else ever runs. Read in the order `part, lineitem, supplier, partsupp,
orders, nation` each table has an equality with something already joined. q8 is
the same shape over eight tables.

## The rule

Keep the first relation where the query put it and then repeatedly take the
first one left that has an equality with something already taken. Stop and
change nothing at all if that ever fails, because a relation with no equality to
anything already taken is a product the query really did ask for, and leaving it
exactly as written is what keeps this pass from turning one refusal into a
different one.

An equality counts when both sides are a plain column and each of them is handed
out by exactly one relation in the chain. That is deliberately the same test
predicate pushdown uses to decide whether an equality can become a join key,
since a reordering that satisfies a weaker test would produce an order whose
joins still could not be keyed.

There is no cost in any of this. Among the orders with no product in them it
takes the first one it finds, which is the one closest to what the query wrote.
Choosing between them is the estimator question and document 03 says when to
answer it.

## Where it runs and why the list is rebuilt

Before predicate pushdown, because pushdown is what puts the equalities onto the
joins and it can only do that once they are next to the relations they read. On
a later sweep the joins it reordered are keyed inner joins rather than empty
cross joins, so there is no chain left to find and the pass does nothing, which
is what makes running it inside the loop safe.

Rewiring is done in place and then the whole node list is written out again in a
new order. The arena holds a plan as a list in which a node reads only nodes
before it, and reordering a chain breaks that: the fifth relation of a `FROM`
sits after the first join in the list and a reorder can want it as that join's
input. Nothing about the plan is different afterwards except the numbering, and
the rebuild also drops the nodes the rewiring left with nobody reading them.
"""

from firepanda.dtype.schema import Schema
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.plan.bind import Bound, bind, bind_all
from firepanda.plan.expr import ExprKind, Expressions
from firepanda.plan.node import NodeKind, Plan, PlanNode


def order(mut plan: Plan, root: Int, sources: List[Schema]) raises -> Int:
    """Reorders every comma `FROM` that was written with a product in it.

    Args:
        plan: The plan, rewritten in place.
        root: The node whose output is the answer.
        sources: The schema of each relation, indexed by the id a scan carries.

    Returns:
        The new index of the root, which is the old one when nothing moved and
        is not when something did.

    Raises:
        If the plan does not bind, before or after the rewrite.
    """
    var bound = bind_all(plan, root, sources)
    var moved = False
    _walk(plan, root, bound, moved)
    if not moved:
        return root
    var at = _compact(plan, root)
    _ = bind(plan, at, sources)
    return at


def _walk(mut plan: Plan, at: Int, bound: List[Bound], mut moved: Bool) raises:
    """Looks for a chain under every filter in the plan.

    The bindings are read after an inner chain may already have been reordered,
    and that is sound because the only question asked of them is whether a
    relation hands out a column of a given name and origin. Reordering a chain
    changes the order of the columns its top produces and changes neither which
    columns those are nor where each of them came from.

    Args:
        plan: The plan, rewritten in place.
        at: The node to look at.
        bound: What every node produces.
        moved: Set to true if anything was reordered.

    Raises:
        If an expression is not in the arena.
    """
    if plan.nodes[at].kind == NodeKind.FILTER:
        _reorder(
            plan,
            plan.nodes[at].exprs[0],
            plan.nodes[at].inputs[0],
            bound,
            moved,
        )
    var inputs = plan.nodes[at].inputs.copy()
    for i in range(len(inputs)):
        _walk(plan, inputs[i], bound, moved)


def _reorder(
    mut plan: Plan,
    restriction: Int,
    under: Int,
    bound: List[Bound],
    mut moved: Bool,
) raises:
    """Puts the relations of one chain into an order with no product in it.

    Args:
        plan: The plan, whose join nodes are rewired in place.
        restriction: The filter's predicate, which is where the equalities
            are.
        under: The node the filter reads.
        bound: What every node produces.
        moved: Set to true if anything was reordered.

    Raises:
        If an expression is not in the arena.
    """
    var joins = List[Int]()
    var leaves = List[Int]()
    _chain(plan, under, joins, leaves)
    # Two relations are one join and swapping its sides cannot key it, so the
    # shortest chain worth reading has three.
    if len(leaves) < 3:
        return

    var pieces = List[Int]()
    plan.exprs.conjuncts(restriction, pieces)
    var ones = List[Int]()
    var others = List[Int]()
    for i in range(len(pieces)):
        var piece = pieces[i]
        if plan.exprs.nodes[piece].kind != ExprKind.BINARY:
            continue
        if plan.exprs.nodes[piece].op != Int(BinaryOp.EQ.code):
            continue
        var a = plan.exprs.nodes[piece].children[0]
        var b = plan.exprs.nodes[piece].children[1]
        if plan.exprs.nodes[a].kind != ExprKind.COLUMN:
            continue
        if plan.exprs.nodes[b].kind != ExprKind.COLUMN:
            continue
        var one = _whose(plan.exprs, a, bound, leaves)
        var other = _whose(plan.exprs, b, bound, leaves)
        if one < 0 or other < 0 or one == other:
            continue
        ones.append(one)
        others.append(other)

    var taken = List[Bool](length=len(leaves), fill=False)
    var picked = List[Int](capacity=len(leaves))
    picked.append(0)
    taken[0] = True
    for _ in range(1, len(leaves)):
        var next = _nearest(ones, others, taken)
        if next < 0:
            # Something left has no equality with anything taken, so the query
            # asked for a product and no order takes it out. Left as written,
            # which is the refusal the query already had rather than a new one.
            return
        taken[next] = True
        picked.append(next)

    var same = True
    for i in range(len(picked)):
        if picked[i] != i:
            same = False
            break
    if same:
        return

    # Bottom up, so that the join being given a left input is given the one
    # already rewired. `joins` came back top down, which is the order a chain is
    # read in and the reverse of the order it is written in.
    for i in range(len(joins)):
        var join = joins[len(joins) - 1 - i]
        var left = leaves[picked[0]] if i == 0 else joins[len(joins) - i]
        plan.nodes[join].inputs = [left, leaves[picked[i + 1]]]
    moved = True


def _chain(plan: Plan, at: Int, mut joins: List[Int], mut leaves: List[Int]):
    """Reads a left deep run of cross joins written with no condition.

    Args:
        plan: The plan.
        at: The node the chain is read down from.
        joins: The join nodes, appended to from the top down.
        leaves: What the chain joins, appended to left to right.
    """
    if (
        plan.nodes[at].kind == NodeKind.JOIN
        and plan.nodes[at].op == Int(JoinKind.CROSS.code)
        and len(plan.nodes[at].exprs) == 0
    ):
        joins.append(at)
        _chain(plan, plan.nodes[at].inputs[0], joins, leaves)
        leaves.append(plan.nodes[at].inputs[1])
        return
    leaves.append(at)


def _whose(
    exprs: Expressions, at: Int, bound: List[Bound], leaves: List[Int]
) -> Int:
    """Which relation of a chain hands out the column a reference reads.

    Args:
        exprs: The arena.
        at: The column reference.
        bound: What every node produces.
        leaves: What the chain joins.

    Returns:
        The position in `leaves`, or a negative number when no relation has the
        column or more than one could.
    """
    ref node = exprs.nodes[at]
    var found = -1
    for i in range(len(leaves)):
        if not bound[leaves[i]].provides(node.name, node.table):
            continue
        if found >= 0:
            return -1
        found = i
    return found


def _nearest(ones: List[Int], others: List[Int], taken: List[Bool]) -> Int:
    """The first relation not taken that an equality ties to one that is.

    Args:
        ones: One end of each equality, as a position in the chain.
        others: The other end of each.
        taken: Which positions are already in the order.

    Returns:
        The position to take next, or a negative number when nothing left is
        tied to anything taken.
    """
    for i in range(len(taken)):
        if taken[i]:
            continue
        for e in range(len(ones)):
            if ones[e] == i and taken[others[e]]:
                return i
            if others[e] == i and taken[ones[e]]:
                return i
    return -1


def _compact(mut plan: Plan, root: Int) raises -> Int:
    """Writes the node list out again with every node after the ones it reads.

    Args:
        plan: The plan, whose node list is replaced.
        root: The node whose output is the answer.

    Returns:
        The new index of the root.

    Raises:
        Never, and the signature carries it because every caller is a `def`.
    """
    var into = List[PlanNode]()
    var moved = List[Int](length=len(plan.nodes), fill=-1)
    var at = _copy(plan, root, moved, into)
    plan.nodes = into^
    return at


def _copy(
    plan: Plan, old: Int, mut moved: List[Int], mut into: List[PlanNode]
) raises -> Int:
    """Copies one node and everything under it, deepest first.

    A node that more than one other node reads is copied once and the second
    reader is given the index the first one got, so a plan that is a graph stays
    the same graph rather than becoming a tree the size of its own expansion.

    Args:
        plan: The plan being read.
        old: The node.
        moved: Where each old node ended up, or a negative number for one that
            has not been copied yet.
        into: The new node list, appended to.

    Returns:
        Its new index.

    Raises:
        Never, and the signature carries it because every caller is a `def`.
    """
    if moved[old] >= 0:
        return moved[old]
    var inputs = plan.nodes[old].inputs.copy()
    var mapped = List[Int](capacity=len(inputs))
    for i in range(len(inputs)):
        mapped.append(_copy(plan, inputs[i], moved, into))
    var at = len(into)
    into.append(
        PlanNode(
            plan.nodes[old].kind,
            mapped^,
            plan.nodes[old].exprs.copy(),
            plan.nodes[old].parts,
            plan.nodes[old].names.copy(),
            plan.nodes[old].flags.copy(),
            plan.nodes[old].op,
            plan.nodes[old].offset,
            plan.nodes[old].length,
            plan.nodes[old].table,
            plan.nodes[old].source.copy(),
        )
    )
    moved[old] = at
    return at
