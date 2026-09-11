"""Empty and constant pruning: a node that cannot change the answer goes.

Section 8 of `docs/specs/planner/02-the-pass-pipeline.md`. A provably false
filter collapses its subtree to something empty with the right schema, and a
provably true one is not a filter at all. Rare in a benchmark written by hand
and common in a generated query, which is most queries once a `WHERE` clause is
built out of parameters that were not all supplied.

## The three rules

A filter whose predicate folded to false becomes a limit of zero rows over the
same input. It does not become a node kind of its own, because a limit of zero
already produces no rows and already has the schema of its input, which is
exactly what an empty relation is. The alternative is a tenth node kind that
carries a schema, and there is nothing this rule needs it for.

A filter whose predicate folded to true is spliced out. Everything that read the
filter reads what the filter read.

A filter, sort, distinct or limit over something empty is spliced out for the
same reason. All four hand the input's schema on unchanged, so removing one
from above an empty input leaves an empty input with the same schema, which is
the same answer.

## Why a splice is safe in place

Predicate pushdown has to rebuild the node list because moving a filter down
makes new parents for old children, and the arena hands out indices in creation
order so that an input is always below the node reading it. Removing a node goes
the other way. A reader points at what the removed node pointed at, and that is
below the removed node which is below the reader, so the order still holds.

The spliced out node stays in the arena. Nothing reaches it and it costs one
struct, which is the trade every pass here makes rather than paying a
compaction.

## What it does not do

It does not drop an arm of a join whose other side is empty, and it does not
collapse an aggregate or a projection over an empty input. Those change the
schema or the row count in ways a limit of zero over the same input cannot
express, so they are what would need the node kind this pass avoided. An inner
join against an empty side is the one worth having and it is worth having with
that kind rather than without it.

It also does not decide that a scan is empty. Nothing here reads a file.
"""

from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Schema
from firepanda.plan.bind import bind
from firepanda.plan.expr import ExprKind
from firepanda.plan.node import NodeKind, Plan


def empty(mut plan: Plan, root: Int, sources: List[Schema]) raises -> Int:
    """Removes the nodes a constant predicate made pointless.

    Args:
        plan: The plan, rewritten in place.
        root: The node whose output is the answer.
        sources: A schema per relation a scan can name.

    Returns:
        The new root, which is not the old one when the root itself was a filter
        that was always true or sat over something empty.

    Raises:
        If the plan does not bind, or an expression is not in the arena.
    """
    _ = bind(plan, root, sources)

    var changed = False
    for at in range(len(plan.nodes)):
        if _falsehood(plan, at):
            # A limit of zero over the same input. The node keeps its index and
            # its input, so nothing that pointed at it has to be told.
            plan.nodes[at].kind = NodeKind.LIMIT
            plan.nodes[at].exprs = List[Int]()
            plan.nodes[at].offset = 0
            plan.nodes[at].length = 0
            changed = True

    var at = root
    for _ in range(len(plan.nodes)):
        var moved = False
        while _pointless(plan, at):
            at = plan.nodes[at].inputs[0]
            moved = True
        for node in range(len(plan.nodes)):
            for i in range(len(plan.nodes[node].inputs)):
                var kid = plan.nodes[node].inputs[i]
                if _pointless(plan, kid):
                    plan.nodes[node].inputs[i] = plan.nodes[kid].inputs[0]
                    moved = True
        if not moved:
            break
        changed = True

    if changed:
        _ = bind(plan, at, sources)
    return at


def _constant(plan: Plan, at: Int, want: Bool) raises -> Bool:
    """Whether a node is a filter on a literal boolean of one value.

    Args:
        plan: The plan.
        at: The node.
        want: Which of the two to look for.

    Returns:
        True if the node is that filter.

    Raises:
        If an expression is not in the arena.
    """
    if plan.nodes[at].kind != NodeKind.FILTER:
        return False
    var over = plan.nodes[at].exprs[0]
    plan.exprs.check(over)
    ref node = plan.exprs.nodes[over]
    if node.kind != ExprKind.LITERAL:
        return False
    # A null predicate keeps no rows, so it could be folded, but saying so is a
    # rule about what a null means in a predicate rather than a rule about a
    # constant and the kernel that filters already holds that one.
    if node.value.is_null() or node.type != LogicalType.BOOL:
        return False
    return node.value.as_scalar[DType.bool]() == want


def _falsehood(plan: Plan, at: Int) raises -> Bool:
    """Whether a node is a filter that keeps nothing.

    Args:
        plan: The plan.
        at: The node.

    Returns:
        True for a filter on a literal false.

    Raises:
        If an expression is not in the arena.
    """
    return _constant(plan, at, False)


def _blank(plan: Plan, at: Int) -> Bool:
    """Whether a node provably produces no rows.

    Args:
        plan: The plan.
        at: The node.

    Returns:
        True for a limit of zero, which is the only shape this pass makes and
        the only one it recognises.
    """
    return plan.nodes[at].kind == NodeKind.LIMIT and plan.nodes[at].length == 0


def _pointless(plan: Plan, at: Int) raises -> Bool:
    """Whether a node can be taken out without changing the answer.

    Args:
        plan: The plan.
        at: The node.

    Returns:
        True for a filter that keeps everything, and for a filter, sort,
        distinct or limit over something that produces nothing.

    Raises:
        If an expression is not in the arena.
    """
    if _constant(plan, at, True):
        return True
    if _blank(plan, at):
        # The thing everything else collapses into. Taking it out would take
        # out the emptiness with it.
        return False
    var kind = plan.nodes[at].kind
    var passes = (
        kind == NodeKind.FILTER
        or kind == NodeKind.SORT
        or kind == NodeKind.DISTINCT
        or kind == NodeKind.LIMIT
    )
    if not passes:
        return False
    return _blank(plan, plan.nodes[at].inputs[0])
