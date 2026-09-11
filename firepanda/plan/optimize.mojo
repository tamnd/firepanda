"""The pipeline: every pass, in the order the spec fixes, run until it settles.

`docs/specs/planner/02-the-pass-pipeline.md` under "What order, and why fixed".
Up to here every pass has been a thing a caller could run on its own and nothing
ran any of them. This is the one entry point, and it is the first code in the
planner that a query could go through end to end.

## The order

Simplification, then projection pushdown, then predicate pushdown, then
projection merging, then slice pushdown and top n. The spec gives the reason for
each adjacency and none of them is arbitrary.

Simplification runs first so that every later pass sees the simplest form of
every expression. A predicate that folds to a constant is cheaper to move and a
conjunction that has been flattened is one that predicate pushdown can split.

Projection pushdown runs before predicate pushdown so that a predicate arriving
at a scan finds a column list that already exists rather than one that is about
to be written.

Projection merging runs after both, because both of them leave projections
behind. Narrowing a node to the columns above it is done by putting a projection
there, and a filter that moves past a projection leaves that projection where it
was.

Slice pushdown runs last of the five, since a limit is happiest once the nodes
it might swap past have stopped being rearranged underneath it.

The passes the spec lists that are not written yet slot into this function and
nowhere else, which is the point of having it.

## Why it runs more than once

Because a pass can make work for a pass that has already run. Predicate pushdown
moving a filter below a projection can leave two projections adjacent, and
projection merging has been and gone. The spec says to run the whole pipeline
again if the second run changes anything, up to a small bound, and notes that
DuckDB repeats expression rewriting for the same reason and that it costs
microseconds on a plan of tens of nodes.

Whether anything changed is decided by printing the plan and comparing the text.
That is a string compare on a few hundred bytes against a pipeline that has just
walked every node several times, and it has the property that matters, which is
that it is the same notion of changed that a reader has. A structural comparison
would need a definition of equal that every future pass would have to keep
honest, and there is no version of that which is cheaper than being obviously
right.

## What it returns

The root, because predicate pushdown rebuilds the node list and the index the
caller went in with means nothing afterwards. The plan comes back bound, since
every pass in the list binds before it finishes, so a caller that wants the
output schema can ask for it without paying for another bind.
"""

from firepanda.dtype.schema import Schema
from firepanda.plan.limits import limits
from firepanda.plan.merge import merge
from firepanda.plan.node import Plan
from firepanda.plan.print import explain
from firepanda.plan.prune import prune
from firepanda.plan.push import push
from firepanda.plan.simplify import simplify

comptime SWEEPS = 4
"""How many times the whole pipeline may run before it stops looking.

The spec asks for twice and a small bound. Four is the small bound, and nothing
written so far gets past two, because the passes that can make work for each
other are the two pushdowns and they are both idempotent once the plan has
settled. The bound exists so that a pass added later which oscillates costs a
few microseconds rather than the query.
"""


def optimize(mut plan: Plan, root: Int, sources: List[Schema]) raises -> Int:
    """Runs every pass in the fixed order until the plan stops changing.

    Args:
        plan: The plan, rewritten in place.
        root: The node whose output is the answer.
        sources: The schema of each relation, indexed by the id a scan carries.

    Returns:
        The new root, which is not the old one when predicate pushdown has
        rebuilt the node list.

    Raises:
        Whatever any of the passes raises, which for a plan that binds is
        nothing.
    """
    var at = root
    var before = explain(plan, at)
    for _ in range(SWEEPS):
        at = _sweep(plan, at, sources)
        var after = explain(plan, at)
        if after == before:
            break
        before = after^
    return at


def _sweep(mut plan: Plan, root: Int, sources: List[Schema]) raises -> Int:
    """Runs each pass once, in order.

    Args:
        plan: The plan, rewritten in place.
        root: The node whose output is the answer.
        sources: The schema of each relation, indexed by the id a scan carries.

    Returns:
        The new root.

    Raises:
        Whatever any of the passes raises.
    """
    simplify(plan, root)
    _ = prune(plan, root, sources)
    var at = push(plan, root, sources)
    _ = merge(plan, at, sources)
    _ = limits(plan, at, sources)
    return at
