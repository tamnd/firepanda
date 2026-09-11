"""Projection merging: two projections in a row are one projection.

Section 6 of `docs/specs/planner/02-the-pass-pipeline.md`, and the pass that
pays for the two before it. Projection pushdown and predicate pushdown both
leave projections behind, because narrowing a node to the columns above it is
done by putting a projection there, and a frame API that answers `df.assign(a =
...).assign(b = ...)` builds one node per call whatever the planner does. So a
plan arrives here with a line of projections in it, each one walking every row
of the chunk below it to hand most of the columns straight back.

Every one of those is a pass over the data. The q1 shape in the spec has two
additions stacked, and folding them into one node takes it from 93 ms to 37 ms,
which is not the arithmetic getting faster. It is the second walk not happening.

## What it does

An upper projection reads the lower one's outputs by name. Merging the two means
putting the lower one's expression where the name was, so `b = a + 1` over
`a = x * 2` becomes `b = x * 2 + 1` reading `x` directly, and the node that
computed `a` goes away. `Expressions.graft` does the substitution and this file
decides when it is allowed.

## Why it can rewrite in place

Unlike predicate pushdown, which has to rebuild the node list because moving a
filter down makes new parents for old children. Merging goes the other way. The
upper node keeps its index, takes over the lower node's input, and since the
upper index was already above the lower one and the lower one was already above
its own input, the arena's creation order still holds. Nothing moves, one node
stops being reachable, and the indices a caller holds still mean what they did.

The merged out node is left in the arena rather than compacted away. Nothing
reaches it from the root, so it costs one struct and no work, and paying a
rebuild to reclaim it would give up the in place property that makes this pass
cheap.

## The three refusals

The lower node has to have exactly one reader. Two readers and the substitution
would write its expressions into both of them, which turns one evaluation into
two and is the opposite of what the pass is for. Common subplan elimination is
what creates that shape deliberately, and this pass leaving it alone is how the
two stay out of each other's way.

The lower node's output names have to be distinct. A name that appears twice
resolves to the first, so substituting by name would be guessing which was
meant. The same refusal `prune` makes, for the same reason.

An output the upper node reads more than once may be substituted as long as no
mention of it is a whole output on its own. `b = a + a` over `a = expensive(x)`
merges, because the grafted expression is one index in two places and lowering
computes an index it meets twice once. `b = a, c = a + 1` over the same lower
node does not, because `b` is the top of an output, the column an output lands
in carries that output's name, and a column cannot answer to two names. So that
one mention would be computed on its own and the pass would have arranged for
the expensive thing to run twice.

That last refusal comes off when a physical projection can rename, which is the
only reason a top cannot be shared. It is not waiting on common subexpression
elimination any more.
"""

from firepanda.dtype.schema import Schema
from firepanda.plan.bind import bind
from firepanda.plan.expr import ExprKind, Expressions
from firepanda.plan.node import NodeKind, Plan


def merge(mut plan: Plan, root: Int, sources: List[Schema]) raises -> Schema:
    """Folds every line of adjacent projections down to one node each.

    Args:
        plan: The plan, rewritten in place.
        root: The node whose output is the answer.
        sources: The schema of each relation, indexed by the id a scan carries.

    Returns:
        The schema of the root, which is what it was before, because merging
        projections changes how the columns are computed and not what they are.

    Raises:
        Whatever binding raises, since this binds the plan before it rewrites
        anything and again afterwards.
    """
    _ = bind(plan, root, sources)

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

    # Downwards, so that a line of three projections folds in one sweep: the top
    # two become one, and the one they became is still the node being looked at,
    # so the third is the next thing under it.
    for at in range(len(plan.nodes) - 1, -1, -1):
        if not live[at]:
            continue
        while _mergeable(plan, at, readers):
            var below = plan.nodes[at].inputs[0]
            _fold(plan, at)
            readers[below] = 0
            live[below] = False

    return bind(plan, root, sources)


def _mergeable(plan: Plan, at: Int, readers: List[Int]) raises -> Bool:
    """Whether the node below this one can be folded into it.

    Args:
        plan: The plan.
        at: The upper node.
        readers: How many nodes read each node.

    Returns:
        True when both are projections and none of the three refusals applies.
        The third one only bites when a repeated output is also a whole output
        of the upper node, since lowering shares everything else.

    Raises:
        If an expression is not in the arena.
    """
    if plan.nodes[at].kind != NodeKind.PROJECT:
        return False
    var below = plan.nodes[at].inputs[0]
    if plan.nodes[below].kind != NodeKind.PROJECT:
        return False
    if readers[below] != 1:
        return False

    var names = plan.nodes[below].names.copy()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if names[i] == names[j]:
                return False

    var used = List[Int](length=len(names), fill=0)
    var above = plan.nodes[at].exprs.copy()
    for i in range(len(above)):
        if not _tally(plan.exprs, above[i], names, used):
            # A column the upper node reads that the lower one does not produce.
            # A bound plan should not have one, and a pass that rewrote the
            # expression rather than declining would be papering over the bug.
            return False

    # A mention that is a whole output of the upper node is the one lowering
    # cannot share, because the column it lands in carries that output's name.
    var tops = List[Int](length=len(names), fill=0)
    for i in range(len(above)):
        if plan.exprs.nodes[above[i]].kind != ExprKind.COLUMN:
            continue
        for j in range(len(names)):
            if names[j] == plan.exprs.nodes[above[i]].name:
                tops[j] += 1
                break

    var onto = plan.nodes[below].exprs.copy()
    for i in range(len(used)):
        if used[i] < 2 or tops[i] == 0:
            continue
        var kind = plan.exprs.nodes[onto[i]].kind
        if kind != ExprKind.COLUMN and kind != ExprKind.LITERAL:
            return False
    return True


def _tally(
    exprs: Expressions, root: Int, names: List[String], mut used: List[Int]
) raises -> Bool:
    """Counts how often one expression reads each of a list of names.

    Every mention counts, not every distinct name, because the question the
    caller is asking is whether substituting would compute anything twice and
    two mentions of one name in one expression is exactly that.

    Args:
        exprs: The arena.
        root: The expression.
        names: The names to count.
        used: The counts, added to.

    Returns:
        False as soon as a column is read that is not in the list.

    Raises:
        If an expression is not in the arena.
    """
    exprs.check(root)
    ref node = exprs.nodes[root]
    if node.kind == ExprKind.COLUMN:
        for i in range(len(names)):
            if names[i] == node.name:
                used[i] += 1
                return True
        return False
    for i in range(len(node.children)):
        if not _tally(exprs, node.children[i], names, used):
            return False
    return True


def _fold(mut plan: Plan, at: Int) raises:
    """Substitutes the node below into this one and takes over its input.

    Args:
        plan: The plan, rewritten in place.
        at: The upper node, which is the one that survives.

    Raises:
        If an expression is not in the arena.
    """
    var below = plan.nodes[at].inputs[0]
    var names = plan.nodes[below].names.copy()
    var onto = plan.nodes[below].exprs.copy()
    var deeper = plan.nodes[below].inputs[0]

    var above = plan.nodes[at].exprs.copy()
    var grown = List[Int]()
    for i in range(len(above)):
        grown.append(plan.exprs.graft(above[i], names, onto))

    # The names do not change, which is the whole reason the schema above this
    # node is the schema it was before the merge.
    plan.nodes[at].exprs = grown^
    plan.nodes[at].inputs[0] = deeper
