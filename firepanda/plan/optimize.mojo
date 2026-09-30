"""The pipeline: every pass, in the order the spec fixes, run until it settles.

`docs/specs/planner/02-the-pass-pipeline.md` under "What order, and why fixed".
Up to here every pass has been a thing a caller could run on its own and nothing
ran any of them. This is the one entry point, and it is the first code in the
planner that a query could go through end to end.

## The order

Simplification, then empty and constant pruning, then projection pushdown, then
join ordering, then predicate pushdown, then common subexpression elimination,
then projection merging, then slice pushdown and top n. The spec gives the
reason for each adjacency and none of them is arbitrary.

Simplification runs first so that every later pass sees the simplest form of
every expression. A predicate that folds to a constant is cheaper to move and a
conjunction that has been flattened is one that predicate pushdown can split.

Empty and constant pruning runs second because simplification is what turns a
predicate into the constant it needs, and because every node it takes out is a
node none of the five passes below it has to look at. It is the only pass here
that makes the plan smaller rather than different.

Projection pushdown runs before predicate pushdown so that a predicate arriving
at a scan finds a column list that already exists rather than one that is about
to be written.

Join ordering runs between them, which is not where the spec's list has it, and
the reason is which half of it is written. The spec puts ordering tenth because
the half that chooses between two orders wants the cardinalities the pushed
filters imply. The half that is written chooses nothing: it only reorders a
comma `FROM` whose relations were written in an order that leaves a product in
the middle, and it has to run before predicate pushdown because pushdown is what
turns an equality into a join key and it can only do that once the two relations
the equality reads are next to each other. The cost half, when it is written,
goes where the spec puts it.

Common subexpression elimination runs after the two pushdowns, because the spec
asks for elimination after both so that subtrees the pushdowns made identical
are recognised. It runs before merging rather than after, and merging then has
the chance to bring two expressions together that elimination has already been
past, which is one of the things the second sweep is for.

Projection merging runs after all three, because the pushdowns leave projections
behind. Narrowing a node to the columns above it is done by putting a projection
there, and a filter that moves past a projection leaves that projection where it
was.

Slice pushdown runs last of the seven, since a limit is happiest once the nodes
it might swap past have stopped being rearranged underneath it.

The passes the spec lists that are not written yet slot into this function and
nowhere else, which is the point of having it.

## The one pass outside the loop

Common subplan elimination runs once, after the loop has settled, and no pass in
the loop ever sees what it did. It is the only pass that leaves the plan a graph
rather than a tree and the rest are written for a tree. Two of them would be
wrong on a node with two parents: projection pushdown narrows a node to the
columns the node above it asked for, and a node with two parents has two answers
to that, and slice pushdown swaps the contents of two adjacent nodes, which
rewrites the lower one under whoever else was reading it.

That is also where the saving is. Nothing is saved by the plan being smaller
while it is being rewritten. The saving is one scan and one filter at run time
instead of two, and that is cashed when the plan is lowered.

## Why it runs more than once

Because a pass can make work for a pass that has already run. Predicate pushdown
moving a filter below a projection can leave two projections adjacent, and
projection merging has been and gone. The spec says to run the whole pipeline
again if the second run changes anything, up to a small bound, and notes that
DuckDB repeats expression rewriting for the same reason and that it costs
microseconds on a plan of tens of nodes.

One pass is invisible to that comparison. Elimination makes two indices into
one and the printed plan does not say which index an expression is, so a sweep
where it was the only pass that did anything reads as a sweep where nothing
happened. That is the right answer rather than a gap: there is nothing left for
another sweep to find, because the work it does is idempotent by construction.

Whether anything changed is decided by printing the plan and comparing the text.
That is a string compare on a few hundred bytes against a pipeline that has just
walked every node several times, and it has the property that matters, which is
that it is the same notion of changed that a reader has. A structural comparison
would need a definition of equal that every future pass would have to keep
honest, and there is no version of that which is cheaper than being obviously
right.

## Turning a pass off

Each pass has a name, and a caller can hand in a list of names to leave out.
The names are DuckDB's where DuckDB has a pass that does the same job, so that
`SET disabled_optimizers = 'filter_pushdown'` means here what it means there:
`expression_rewriter`, `empty_result_pullup`, `unused_columns`, `join_order`,
`filter_pushdown`, `common_subexpressions`, `limit_pushdown` and
`common_subplan`. Projection merging has no DuckDB counterpart, because DuckDB
never builds two projections in a row to merge, so it goes by
`projection_merge`, which DuckDB does not know.

A pass turned off is skipped and nothing else changes. The order of the rest is
the order above, and every one of them is a rewrite to a plan that answers the
same thing, so the answer does not move. That is the property the optimizer off
equivalence test leans on.

## What it returns

The root, because predicate pushdown rebuilds the node list and the index the
caller went in with means nothing afterwards. The plan comes back bound, since
every pass in the list binds before it finishes, so a caller that wants the
output schema can ask for it without paying for another bind.
"""

from firepanda.dtype.schema import Schema
from firepanda.plan.cse import cse
from firepanda.plan.empty import empty
from firepanda.plan.limits import limits
from firepanda.plan.merge import merge
from firepanda.plan.node import Plan
from firepanda.plan.order import order
from firepanda.plan.print import explain
from firepanda.plan.prune import prune
from firepanda.plan.push import push
from firepanda.plan.simplify import simplify
from firepanda.plan.subplan import subplan

comptime SWEEPS = 4
"""How many times the whole pipeline may run before it stops looking.

The spec asks for twice and a small bound. Four is the small bound, and nothing
written so far gets past two, because the passes that can make work for each
other are the two pushdowns and they are both idempotent once the plan has
settled. The bound exists so that a pass added later which oscillates costs a
few microseconds rather than the query.
"""


def passes() -> List[StaticString]:
    """The name of every pass, in the order they run.

    Returns:
        The names a caller can turn one off by.
    """
    return [
        "expression_rewriter",
        "empty_result_pullup",
        "unused_columns",
        "join_order",
        "filter_pushdown",
        "common_subexpressions",
        "projection_merge",
        "limit_pushdown",
        "common_subplan",
    ]


def optimize(
    mut plan: Plan,
    root: Int,
    sources: List[Schema],
    disabled: List[String] = List[String](),
) raises -> Int:
    """Runs every pass in the fixed order until the plan stops changing.

    Args:
        plan: The plan, rewritten in place.
        root: The node whose output is the answer.
        sources: The schema of each relation, indexed by the id a scan carries.
        disabled: The names of the passes to leave out, from `passes`. A name
            that is not one of them turns nothing off.

    Returns:
        The new root, which is not the old one when predicate pushdown has
        rebuilt the node list or pruning has taken the root itself out.

    Raises:
        Whatever any of the passes raises, which for a plan that binds is
        nothing.
    """
    var at = root
    var before = explain(plan, at)
    for _ in range(SWEEPS):
        at = _sweep(plan, at, sources, disabled)
        var after = explain(plan, at)
        if after == before:
            break
        before = after^
    # Once and at the end, because it is the only pass that leaves the plan a
    # graph and every pass in the sweep is written for a tree.
    if _on(disabled, "common_subplan"):
        _ = subplan(plan, at, sources)
    return at


def _on(disabled: List[String], name: StringSlice) -> Bool:
    """Whether the pass called `name` is to run."""
    for entry in disabled:
        if entry == name:
            return False
    return True


def _sweep(
    mut plan: Plan, root: Int, sources: List[Schema], disabled: List[String]
) raises -> Int:
    """Runs each pass once, in order, leaving out the ones turned off.

    Args:
        plan: The plan, rewritten in place.
        root: The node whose output is the answer.
        sources: The schema of each relation, indexed by the id a scan carries.
        disabled: The names of the passes to leave out.

    Returns:
        The new root.

    Raises:
        Whatever any of the passes raises.
    """
    var at = root
    if _on(disabled, "expression_rewriter"):
        simplify(plan, at)
    if _on(disabled, "empty_result_pullup"):
        at = empty(plan, at, sources)
    if _on(disabled, "unused_columns"):
        _ = prune(plan, at, sources)
    if _on(disabled, "join_order"):
        at = order(plan, at, sources)
    if _on(disabled, "filter_pushdown"):
        at = push(plan, at, sources)
    if _on(disabled, "common_subexpressions"):
        _ = cse(plan, at, sources)
    if _on(disabled, "projection_merge"):
        _ = merge(plan, at, sources)
    if _on(disabled, "limit_pushdown"):
        _ = limits(plan, at, sources)
    return at
