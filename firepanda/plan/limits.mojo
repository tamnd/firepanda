"""Slice pushdown and top n: not sorting six million rows to print ten.

Section 7 of `docs/specs/planner/02-the-pass-pipeline.md`. Four of the twenty
two TPC-H queries end in an order by with a limit on it, and the difference
between answering one of those by sorting the whole thing and answering it by
keeping the best ten rows as they go past is the whole query.

## The three rules

A limit above a sort puts a bound on the sort. The sort is now allowed to get
only the first n rows right, which is what a heap of size n does in one pass
instead of what a full sort does in several. The limit stays where it is and
goes on doing the cutting, so an operator that has not learned to read it is
slower than one that has and is not wrong. That matters more than it sounds,
because it means this rule can land before anything honours it.

A limit above a limit becomes one limit. Two slices of a row sequence compose
into one slice, and working out which one is the sort of arithmetic that is
easier to get right once here than at every place that builds a pair.

A limit above a projection swaps with it, so the projection evaluates n rows
rather than all of them. Only when every expression in the projection is
elementwise. A window function reads its whole partition, and cutting the rows
down before it runs gives a different answer rather than the same answer sooner.

## Why it can swap instead of rebuilding

Pushing a limit down makes a new parent for an old child, which is what the
arena's creation order forbids and what makes predicate pushdown rebuild the
whole node list. The way around it here is that both nodes have exactly one
input, so nothing has to move. The two nodes trade contents and keep their
indices and their inputs. The upper index goes on being the upper index, it just
holds the projection now, and the lower index holds the limit. Same plan, same
order, no rebuild, and the root a caller is holding is still the root.

That trick works for any two adjacent unary nodes and nothing about it is
specific to limits, which is worth knowing for the next pass that wants it.

## What it does not do

A union is not pushed into, because capping each arm of a union all means a new
limit node above each arm, and a new node comes out above the union rather than
below it. That one wants the rebuild machinery and belongs with a pass that
already has it.

A filter is never swapped with, for the obvious reason that a limit below a
filter counts rows the filter was going to throw away.

Nothing here needs a node that more than one other node reads, and every rule
declines when it finds one, since both the swap and the combine change what the
node below produces and a second reader did not ask for that.
"""

from firepanda.dtype.schema import Schema
from firepanda.plan.bind import bind
from firepanda.plan.node import NO_LIMIT, NodeKind, Plan


def limits(mut plan: Plan, root: Int, sources: List[Schema]) raises -> Schema:
    """Combines, pushes down and turns into bounds every limit in the plan.

    Args:
        plan: The plan, rewritten in place.
        root: The node whose output is the answer.
        sources: The schema of each relation, indexed by the id a scan carries.

    Returns:
        The schema of the root, which is what it was before, because a limit
        decides how many rows come out and not what a row looks like.

    Raises:
        Whatever binding raises, since this binds the plan to check it before
        rewriting anything.
    """
    var out = bind(plan, root, sources)

    var live = List[Bool](length=len(plan.nodes), fill=False)
    var readers = List[Int](length=len(plan.nodes), fill=0)
    live[root] = True
    for at in range(len(plan.nodes) - 1, -1, -1):
        if not live[at]:
            continue
        var inputs = plan.nodes[at].inputs.copy()
        for i in range(len(inputs)):
            live[inputs[i]] = True
            readers[inputs[i]] += 1

    # Downwards, so that a limit which swaps its way past a projection is picked
    # up again at its new index and carries on going.
    for at in range(len(plan.nodes) - 1, -1, -1):
        if not live[at]:
            continue
        while _step(plan, at, readers, live):
            pass

    return out^


def _step(
    mut plan: Plan, at: Int, mut readers: List[Int], mut live: List[Bool]
) raises -> Bool:
    """Applies one rule at one node.

    Args:
        plan: The plan, rewritten in place.
        at: The node.
        readers: How many nodes read each node, kept up to date.
        live: Which nodes the root reaches, kept up to date.

    Returns:
        True when something moved and the node is worth looking at again. A
        bound put on a sort returns False, because putting the same bound on
        the same sort a second time would go round for ever.

    Raises:
        If an expression is not in the arena.
    """
    if plan.nodes[at].kind != NodeKind.LIMIT:
        return False
    var below = plan.nodes[at].inputs[0]
    if readers[below] != 1:
        return False
    var kind = plan.nodes[below].kind

    if kind == NodeKind.LIMIT:
        var deeper = plan.nodes[below].inputs[0]
        var offset = plan.nodes[below].offset + plan.nodes[at].offset
        var length = _compose(
            plan.nodes[below].length,
            plan.nodes[at].offset,
            plan.nodes[at].length,
        )
        plan.nodes[at].offset = offset
        plan.nodes[at].length = length
        plan.nodes[at].inputs[0] = deeper
        readers[below] = 0
        live[below] = False
        return True

    if kind == NodeKind.SORT:
        if plan.nodes[at].length == NO_LIMIT:
            # A limit with no length skips rows and keeps the rest, so the sort
            # still has to get all of them right.
            return False
        var want = plan.nodes[at].offset + plan.nodes[at].length
        var had = plan.nodes[below].length
        plan.nodes[below].length = want if had == NO_LIMIT else min(had, want)
        return False

    if kind == NodeKind.PROJECT:
        var exprs = plan.nodes[below].exprs.copy()
        for i in range(len(exprs)):
            if not plan.exprs.elementwise(exprs[i]):
                return False
        _swap(plan, at, below)
        return True

    return False


def _compose(inner: Int, offset: Int, outer: Int) -> Int:
    """Returns the length of one slice taken out of another.

    The inner slice hands out its rows from `offset` onwards and the outer one
    wants `outer` of them, so what comes out is the smaller of what was asked
    for and what is left, and `NO_LIMIT` on either side means that side is not
    the one doing the limiting.

    Args:
        inner: The length of the slice below, or `NO_LIMIT`.
        offset: How many rows the slice above skips.
        outer: The length of the slice above, or `NO_LIMIT`.

    Returns:
        The length of the two together, or `NO_LIMIT` when neither bounds it.
    """
    if inner == NO_LIMIT:
        return outer
    var left = inner - offset
    if left < 0:
        left = 0
    if outer == NO_LIMIT:
        return left
    return min(outer, left)


def _swap(mut plan: Plan, upper: Int, lower: Int) raises:
    """Exchanges the contents of two adjacent nodes with one input each.

    Everything except the inputs, which are what hold the two in place. The
    node at the upper index keeps reading the node at the lower index and the
    node at the lower index keeps reading whatever it read, so after the trade
    the plan says the other order while the arena says what it said before.

    Args:
        plan: The plan, rewritten in place.
        upper: The node above.
        lower: The node below, which is the upper node's only input.

    Raises:
        Never, and takes `raises` because a plan rewrite is written that way
        throughout.
    """
    var was = plan.nodes[upper].copy()
    var now = plan.nodes[lower].copy()

    plan.nodes[upper].kind = now.kind
    plan.nodes[upper].exprs = now.exprs.copy()
    plan.nodes[upper].parts = now.parts
    plan.nodes[upper].names = now.names.copy()
    plan.nodes[upper].flags = now.flags.copy()
    plan.nodes[upper].op = now.op
    plan.nodes[upper].offset = now.offset
    plan.nodes[upper].length = now.length
    plan.nodes[upper].table = now.table
    plan.nodes[upper].source = now.source.copy()

    plan.nodes[lower].kind = was.kind
    plan.nodes[lower].exprs = was.exprs.copy()
    plan.nodes[lower].parts = was.parts
    plan.nodes[lower].names = was.names.copy()
    plan.nodes[lower].flags = was.flags.copy()
    plan.nodes[lower].op = was.op
    plan.nodes[lower].offset = was.offset
    plan.nodes[lower].length = was.length
    plan.nodes[lower].table = was.table
    plan.nodes[lower].source = was.source.copy()
