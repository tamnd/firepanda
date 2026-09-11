"""Predicate pushdown: a row is thrown away before it is paid for.

The fourth pass in `docs/specs/planner/02-the-pass-pipeline.md` and the one the
spec says to build with `04-predicate-transfer.md` in mind. A filter moves
toward the scan until it cannot go further, so rows are eliminated before they
are joined, aggregated or projected rather than after. The spec's own numbers,
taken by hand on the TPC-H driver: q19 went from 83 to 70 milliseconds by
running its four cheap conditions on `part` and on `lineitem` before the join
instead of running the disjunction over every shipped line, and q21 went from
285 to 169 by filtering before its joins rather than after them.

## Conjuncts, not predicates

A filter holds one expression and that expression is usually an `and` of several
conditions that want to end up in different places. `l_quantity < 30 and
p_size <= 15` over a join has one half belonging on each side, so the first
thing the pass does is split every predicate at its `and` nodes and treat the
pieces separately. Whatever is left over at a node is put back together as one
filter above it, which is where an `and` with two halves that could not be
separated ends up unchanged.

Splitting only at `and`. An `or` cannot be split, because neither side of it has
to hold for the row to survive.

## What stops a predicate

A predicate can move below a node when the rows that come out of that node still
answer it and when the answer means the same thing. Those are two different
questions and the second one is the one that is easy to get wrong.

A sort passes everything through, because sorting changes the order of the rows
and not which rows there are. A limit passes nothing through, because the rows
it keeps depend on which rows arrive. A projection and an aggregate pass through
a predicate that reads only columns they hand out under the name they arrived
with: a computed output is not a column of the input, and a renamed one is a
different name below than it is above. An aggregate's group keys qualify and its
aggregate outputs never do, which is the difference between `where` and
`having`.

A distinct with no keys passes everything through, since deduplicating whole
rows and then dropping some is the same set as dropping some and then
deduplicating. A distinct with keys passes through only a predicate that reads
its keys, and the reason is worth writing down: a keyed distinct keeps one row
per key and does not promise which one, so a predicate on a non key column
filtered afterwards can empty a group that filtering beforehand would have kept
a different row of.

A join passes a predicate into the side that provides every column it reads, and
only when the other side provides none of them, so the pass never has to decide
which of two columns of one name was meant. Only inner joins, for now. An outer
join invents null rows and a semi join is already a filter, so what is safe
there is a longer argument than what is safe here, and the pass would rather do
nothing than do it on a guess.

## The join also gains predicates here

Reaching an inner join, the pass calls `transit.derive`, which reads the join's
equality conditions and adds to the list being carried the copy across the
equality of every predicate that has one. That happens before the routing below,
so a copy is placed the way a predicate the caller wrote above the join would
have been placed, and nothing in the routing knows or cares that it is new.

It lives there rather than in a pass of its own for the reason the next section
gives. A new filter above a join arm needs an index between a node and its
parent and there is no such index, so the pass that already rebuilds the node
list is the only one that can place one.

## Why it rebuilds rather than moves

Node indices are handed out in creation order, so an input always sits below the
node that reads it, and every pass in this package relies on that. Moving a
filter down the tree makes new parents for old children, which is exactly the
thing that invariant forbids doing in place. So the pass walks the plan and
writes a new node list bottom up, which restores the invariant by construction
and drops anything the root does not reach along the way.

That is why `push` returns the new root rather than the schema. The indices are
not the ones the caller handed in and the old ones mean nothing afterwards.

The expressions are not rebuilt, only the nodes, so an index into the expression
arena stays valid throughout. Positions do not, because the schemas move, and
the pass hands the plan back to `bind` for the same reason projection pushdown
does: names survive the move and binding resolves by name.

## The one refusal

A node read by more than one other node is a shared subtree, and a pass that
rebuilds a tree bottom up would write it out once per reader. That turns a plan
that says do this once into a plan that says do it twice, which is the opposite
of what common subplan elimination is for. So the pass looks for sharing first
and, if it finds any, hands the plan back untouched. Every plan the frame API
builds is a tree and this is a guard rather than a limitation, but it is a real
refusal and it is quiet, so it is written down here.
"""

from firepanda.dtype.schema import Schema
from firepanda.join.pairs import JoinKind
from firepanda.plan.bind import Bound, bind, bind_all
from firepanda.plan.expr import UNBOUND, ExprKind, Expressions
from firepanda.plan.node import NodeKind, Plan, PlanNode
from firepanda.plan.transit import derive


def push(mut plan: Plan, root: Int, sources: List[Schema]) raises -> Int:
    """Moves every filter as far toward the scans as it can go.

    Args:
        plan: The plan, rewritten in place.
        root: The node whose output is the answer.
        sources: The schema of each relation, indexed by the id a scan carries.

    Returns:
        The new index of the root, which is not the one handed in.

    Raises:
        If the plan does not bind, before or after the rewrite.
    """
    var bound = bind_all(plan, root, sources)
    if _shared(plan, root):
        return root

    var nodes = List[PlanNode]()
    var nothing = List[Int]()
    var at = _rebuild(plan, root, bound, nothing^, nodes)
    plan.nodes = nodes^
    _ = bind(plan, at, sources)
    return at


def _rebuild(
    mut plan: Plan,
    old: Int,
    bound: List[Bound],
    var carried: List[Int],
    mut into: List[PlanNode],
) raises -> Int:
    """Writes one node and everything under it into a new list, bottom up.

    Args:
        plan: The plan being read and whose expression arena is added to.
        old: The node to write, as an index into the old list.
        bound: What every old node produces.
        carried: The predicates from above that are still looking for a home.
        into: The new node list, appended to.

    Returns:
        The new index of the node, or of the filter written above it.

    Raises:
        If a node holds an index that is not in the plan.
    """
    var kind = plan.nodes[old].kind
    var inputs = plan.nodes[old].inputs.copy()

    if kind == NodeKind.FILTER:
        # The node itself disappears here and is rebuilt wherever its pieces
        # come to rest, which for a filter that cannot move at all is directly
        # above the same input it was above before.
        plan.exprs.conjuncts(plan.nodes[old].exprs[0], carried)
        return _rebuild(plan, inputs[0], bound, carried^, into)

    if kind == NodeKind.SCAN:
        var at = _emit(plan, old, List[Int](), into)
        return _apply(plan, at, carried^, into)

    if kind == NodeKind.JOIN:
        return _join(plan, old, bound, carried^, into)

    if kind == NodeKind.UNION:
        return _union(plan, old, bound, carried^, into)

    var down = List[Int]()
    var here = List[Int]()
    _split(plan, old, bound, carried, down, here)

    var input = _rebuild(plan, inputs[0], bound, down^, into)
    var at = _emit(plan, old, [input], into)
    return _apply(plan, at, here^, into)


def _split(
    mut plan: Plan,
    old: Int,
    bound: List[Bound],
    carried: List[Int],
    mut down: List[Int],
    mut here: List[Int],
) raises:
    """Sorts the carried predicates into the ones that can go below one node.

    The five kinds with a single input that are not a filter: the project, the
    aggregate, the sort, the limit and the distinct. Each of them answers the
    question with a set of column names that mean the same thing on both sides
    of it, and the two that do not need a set say so by passing everything or
    nothing.

    Args:
        plan: The plan.
        old: The node.
        bound: What every old node produces.
        carried: The predicates arriving from above.
        down: The ones that can go below, appended to.
        here: The ones that cannot, appended to.

    Raises:
        If an expression is not in the arena.
    """
    var kind = plan.nodes[old].kind

    # A sort changes the order of the rows and not which rows there are, and a
    # distinct over the whole row has no column that is not part of its key, so
    # both of them pass everything through without having to look at a name.
    var whole = kind == NodeKind.DISTINCT and len(plan.nodes[old].exprs) == 0
    if kind == NodeKind.SORT or whole:
        for i in range(len(carried)):
            down.append(carried[i])
        return

    if kind == NodeKind.LIMIT:
        # Which rows a limit keeps is exactly what it depends on.
        for i in range(len(carried)):
            here.append(carried[i])
        return

    var through = _through(plan, old)
    for i in range(len(carried)):
        var names = plan.exprs.names(carried[i])
        var ok = True
        for j in range(len(names)):
            if not _holds(through, names[j]):
                ok = False
                break
        if ok:
            down.append(carried[i])
        else:
            here.append(carried[i])


def _through(mut plan: Plan, old: Int) raises -> List[String]:
    """Which names mean the same column above a node and below it.

    Args:
        plan: The plan.
        old: The node.

    Returns:
        The names, each once.

    Raises:
        If an expression is not in the arena.
    """
    var out = List[String]()
    var kind = plan.nodes[old].kind

    if kind == NodeKind.DISTINCT:
        # Only the keys that are plain columns. A key that is computed does not
        # hand its inputs through, because two rows that differ in one of them
        # can still land in the same group and only one of the two survives.
        var keys = plan.nodes[old].exprs.copy()
        for i in range(len(keys)):
            if plan.exprs.nodes[keys[i]].kind == ExprKind.COLUMN:
                out.append(plan.exprs.nodes[keys[i]].name)
        return out^

    if kind != NodeKind.PROJECT and kind != NodeKind.AGGREGATE:
        return out^

    # An aggregate hands through its group keys and nothing else, which is the
    # whole of the difference between a `where` and a `having`. A project has no
    # aggregates in it, so all of its outputs are candidates.
    var upto = len(plan.nodes[old].exprs)
    if kind == NodeKind.AGGREGATE:
        upto = plan.nodes[old].parts
    for i in range(upto):
        var at = plan.nodes[old].exprs[i]
        if plan.exprs.nodes[at].kind != ExprKind.COLUMN:
            continue
        if plan.exprs.nodes[at].name != plan.nodes[old].names[i]:
            # Renamed, so the name above is not the name below and moving the
            # predicate unchanged would have it read something else or nothing.
            continue
        if _twice(plan.nodes[old].names, plan.nodes[old].names[i]):
            # Two outputs of one name, and deciding which was meant is not this
            # pass's decision to make. The same refusal `prune` makes.
            continue
        out.append(plan.exprs.nodes[at].name)
    return out^


def _holds(through: List[String], name: String) -> Bool:
    """Whether one name passes through a node.

    Args:
        through: The names that do.
        name: The name asked about.

    Returns:
        True if a predicate reading that name can move below the node.
    """
    for i in range(len(through)):
        if through[i] == name:
            return True
    return False


def _join(
    mut plan: Plan,
    old: Int,
    bound: List[Bound],
    var carried: List[Int],
    mut into: List[PlanNode],
) raises -> Int:
    """Writes a join, sending each predicate to the side that can answer it.

    Args:
        plan: The plan.
        old: The join.
        bound: What every old node produces.
        carried: The predicates arriving from above.
        into: The new node list, appended to.

    Returns:
        The new index of the join, or of the filter written above it.

    Raises:
        If a node or an expression holds an index that is not in the plan.
    """
    var left = plan.nodes[old].inputs[0]
    var right = plan.nodes[old].inputs[1]
    var inner = plan.nodes[old].op == Int(JoinKind.INNER.code)
    if inner:
        # Transitive predicates. A filter on one side of an equality reaches
        # the other, and it arrives here as though it had been written above
        # the join, so the routing below places it without knowing it is new.
        derive(plan, old, bound, carried)

    var to_left = List[Int]()
    var to_right = List[Int]()
    var here = List[Int]()

    for i in range(len(carried)):
        if not inner:
            here.append(carried[i])
            continue
        var names = plan.exprs.names(carried[i])
        var all_left = True
        var all_right = True
        var any_left = False
        var any_right = False
        for j in range(len(names)):
            if bound[left].schema.has(names[j]):
                any_left = True
            else:
                all_left = False
            if bound[right].schema.has(names[j]):
                any_right = True
            else:
                all_right = False
        if all_left and not any_right:
            to_left.append(carried[i])
        elif all_right and not any_left:
            to_right.append(carried[i])
        else:
            # Either it reads both sides, which makes it a join condition
            # rather than a filter on one of them, or it reads a name both
            # sides have, and that is the ambiguity this pass refuses.
            here.append(carried[i])

    var new_left = _rebuild(plan, left, bound, to_left^, into)
    var new_right = _rebuild(plan, right, bound, to_right^, into)
    var at = _emit(plan, old, [new_left, new_right], into)
    return _apply(plan, at, here^, into)


def _union(
    mut plan: Plan,
    old: Int,
    bound: List[Bound],
    var carried: List[Int],
    mut into: List[PlanNode],
) raises -> Int:
    """Writes a union, sending each predicate into every arm or into none.

    All or nothing, because a union's arms line up by position and a predicate
    that one arm can answer by name and another cannot is a predicate that would
    mean two different things depending on which arm the row came from.

    The same rewrite is right for a difference and an intersection, which are the
    same node with a different code on it. A row of the right arm that the
    predicate throws away could only ever have cancelled a row of the left arm
    that the predicate throws away too, so pushing into both sides is the answer
    the filter above would have given.

    Args:
        plan: The plan.
        old: The union.
        bound: What every old node produces.
        carried: The predicates arriving from above.
        into: The new node list, appended to.

    Returns:
        The new index of the union, or of the filter written above it.

    Raises:
        If a node or an expression holds an index that is not in the plan.
    """
    var arms = plan.nodes[old].inputs.copy()

    var down = List[Int]()
    var here = List[Int]()
    for i in range(len(carried)):
        var names = plan.exprs.names(carried[i])
        var ok = True
        for a in range(len(arms)):
            for j in range(len(names)):
                if not bound[arms[a]].schema.has(names[j]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            down.append(carried[i])
        else:
            here.append(carried[i])

    var new_arms = List[Int]()
    for a in range(len(arms)):
        new_arms.append(_rebuild(plan, arms[a], bound, down.copy(), into))
    var at = _emit(plan, old, new_arms^, into)
    return _apply(plan, at, here^, into)


def _emit(
    mut plan: Plan, old: Int, var inputs: List[Int], mut into: List[PlanNode]
) raises -> Int:
    """Copies one node into the new list with new input indices.

    Args:
        plan: The plan being read.
        old: The node.
        inputs: What it reads, as indices into the new list.
        into: The new node list, appended to.

    Returns:
        Its new index.

    Raises:
        Never, and the signature carries it because every caller is a `def`.
    """
    var at = len(into)
    into.append(
        PlanNode(
            plan.nodes[old].kind,
            inputs^,
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
    return at


def _apply(
    mut plan: Plan, input: Int, var preds: List[Int], mut into: List[PlanNode]
) raises -> Int:
    """Puts the predicates that stopped here back together as one filter.

    Args:
        plan: The plan, whose expression arena gets the `and` chain.
        input: What the filter reads, as an index into the new list.
        preds: The predicates, in the order they were met.
        into: The new node list, appended to.

    Returns:
        The filter's new index, or the input when there is nothing to filter.

    Raises:
        If an expression is not in the arena.
    """
    if len(preds) == 0:
        return input
    var whole = preds[0]
    for i in range(1, len(preds)):
        var pair = List[Int]()
        pair.append(whole)
        pair.append(preds[i])
        whole = plan.exprs.call(String("and"), pair^, rowwise=True)
    var at = len(into)
    into.append(
        PlanNode(
            NodeKind.FILTER,
            [input],
            [whole],
            0,
            List[String](),
            List[Bool](),
            0,
            0,
            0,
            UNBOUND,
            String(),
        )
    )
    return at


def _shared(plan: Plan, root: Int) raises -> Bool:
    """Whether any node the root reaches is read by more than one other.

    Args:
        plan: The plan.
        root: The node whose output is the answer.

    Returns:
        True if the reachable part of the plan is not a tree.

    Raises:
        If the root is not in the plan.
    """
    plan.check(root)
    var seen = List[Bool](length=root + 1, fill=False)
    seen[root] = True
    var readers = List[Int](length=root + 1, fill=0)
    for at in range(root, -1, -1):
        if not seen[at]:
            continue
        for i in range(len(plan.nodes[at].inputs)):
            var input = plan.nodes[at].inputs[i]
            seen[input] = True
            readers[input] += 1
            if readers[input] > 1:
                return True
    return False


def _twice(names: List[String], name: String) -> Bool:
    """Whether a name appears more than once in a list.

    Args:
        names: The list.
        name: The name.

    Returns:
        True if two or more entries match.
    """
    var count = 0
    for i in range(len(names)):
        if names[i] == name:
            count += 1
    return count > 1
