"""Transitive predicates: a filter on one side of a join reaches the other.

Section 4 of `docs/specs/planner/02-the-pass-pipeline.md`. If a query says
`a.k = b.k` and also says something about `a.k`, then the same thing is true of
`b.k` for every row that survives the join, so it can be asked of `b` before the
join rather than discovered during it.

It is the logical half of what `docs/specs/planner/04-predicate-transfer.md`
wants. That document's subject is moving information around the whole join graph
at run time with Bloom filters, and this is the part of the same idea that needs
no run time at all: where the information is a predicate the query already wrote
down, the planner can copy it across an equality and be done.

## What makes it sound

An inner join on `a.k = b.k` only produces a row when the two keys are equal and
present, so any predicate that is a function of `a.k` alone has the same answer
on `b.k`. A row of `b` the copy throws away is a row that could only have paired
with rows of `a` the original predicate had already thrown away, and a null is
not an exception: a predicate that answers null drops its row, and it answers
null on both sides or neither.

Inner joins only. An outer join invents rows where one side had none, and
filtering the side that was going to be invented changes which rows get
invented. A semi join is a filter already.

## Where it lives

Inside predicate pushdown, rather than in a pass of its own, and the reason is
the arena. Node indices are handed out in creation order so that an input sits
below the node reading it, which means a filter cannot be inserted above an
existing node in place: there is no index between a node and its parent.
Pushdown already rebuilds the node list for exactly that reason, and it already
has the machinery for deciding which side of a join a predicate belongs on. So
this adds predicates to the list pushdown is carrying when it reaches a join,
and pushdown routes and places them the way it routes everything else.

That also gets the ordering right for free. A copied predicate arrives at the
join the same way one written above the join would, so it is pushed down to the
scan on its new side in the same pass that created it.

## Where the predicates come from

Two places: the ones pushdown is carrying down from above, and the ones already
sitting in a filter inside either arm. The second matters because after the
first sweep everything has already been pushed to the scans, so by the second
sweep there is nothing left above the join to carry.

The walk into an arm stops at a projection or an aggregate. Below one of those
the names are the input's names rather than the arm's, and a filter on a name
that a projection happens to reuse for a different column would be copied across
as though it were about the key. Stopping is the conservative answer and it
costs little, because pushdown puts filters directly above scans and merging
folds projections together.

## What stops it copying the same thing forever

A predicate is copied only if the side receiving it does not already have one
that reads the same. Sameness is decided by printing both, which is the notion
of equal the pipeline already uses to decide whether a sweep changed anything,
and which ignores the binding, so a predicate written by the caller and one this
produced compare equal when they say the same thing.

Without that the pass would copy left to right on one sweep and right to left on
the next, forever, and the sweep bound would be the only thing stopping it.
"""

from firepanda.dtype.schema import Schema
from firepanda.plan.bind import Bound
from firepanda.plan.expr import ExprKind, Expressions
from firepanda.plan.node import NodeKind, Plan
from firepanda.plan.print import render_expr


def derive(
    mut plan: Plan, old: Int, bound: List[Bound], mut carried: List[Int]
) raises:
    """Adds the copy across the join of every predicate that has one.

    Args:
        plan: The plan, whose expression arena gets the copies.
        old: The join, as an index into the old node list.
        bound: What every old node produces.
        carried: The predicates pushdown is carrying, added to.

    Raises:
        If a node or an expression holds an index that is not in the plan.
    """
    var left = plan.nodes[old].inputs[0]
    var right = plan.nodes[old].inputs[1]
    var parts = plan.nodes[old].parts
    var keys = plan.nodes[old].exprs.copy()
    if parts == 0 or parts * 2 != len(keys):
        return

    # Everything either side could be asked, which is what pushdown is carrying
    # plus what is already sitting in the arm.
    var on_left = List[Int]()
    var on_right = List[Int]()
    _inside(plan, left, on_left)
    _inside(plan, right, on_right)
    for i in range(len(carried)):
        var names = plan.exprs.names(carried[i])
        if _only(names, bound[left].schema, bound[right].schema):
            on_left.append(carried[i])
        elif _only(names, bound[right].schema, bound[left].schema):
            on_right.append(carried[i])

    var said_left = List[String]()
    for i in range(len(on_left)):
        said_left.append(render_expr(plan.exprs, on_left[i]))
    var said_right = List[String]()
    for i in range(len(on_right)):
        said_right.append(render_expr(plan.exprs, on_right[i]))

    for i in range(parts):
        var one = keys[i]
        var other = keys[parts + i]
        if plan.exprs.nodes[one].kind != ExprKind.COLUMN:
            continue
        if plan.exprs.nodes[other].kind != ExprKind.COLUMN:
            continue
        var here = plan.exprs.nodes[one].name.copy()
        var there = plan.exprs.nodes[other].name.copy()
        _across(plan, on_left, here, there, said_right, carried)
        _across(plan, on_right, there, here, said_left, carried)


def _across(
    mut plan: Plan,
    held: List[Int],
    whom: String,
    into: String,
    mut said: List[String],
    mut carried: List[Int],
) raises:
    """Copies every predicate about one key column onto the other.

    Args:
        plan: The plan, whose expression arena gets the copies.
        held: The predicates the side already asks.
        whom: The key column they would have to be about.
        into: The key column on the other side.
        said: What the other side already asks, in print, added to.
        carried: The predicates pushdown is carrying, added to.

    Raises:
        If an expression is not in the arena.
    """
    for i in range(len(held)):
        var names = plan.exprs.names(held[i])
        if len(names) != 1 or names[0] != whom:
            continue
        var made = _mirror(plan.exprs, held[i], whom, into)
        if made == held[i]:
            continue
        var printed = render_expr(plan.exprs, made)
        var had = False
        for j in range(len(said)):
            if said[j] == printed:
                had = True
                break
        if had:
            continue
        said.append(printed^)
        carried.append(made)


def _mirror(
    mut exprs: Expressions, root: Int, whom: String, into: String
) raises -> Int:
    """Returns the same expression with one column read under another name.

    Nothing is rewritten in place, for the reason `graft` gives: an index can be
    read by more than one node. Only the path down to a renamed column is
    copied, so a constant under the predicate is shared with the original, which
    is right because a constant does not care which side it is read on.

    Args:
        exprs: The arena, added to.
        root: The predicate.
        whom: The name to stop reading.
        into: The name to read instead.

    Returns:
        The new expression, or `root` when it does not read that column.

    Raises:
        If an expression is not in the arena.
    """
    exprs.check(root)
    if exprs.nodes[root].kind == ExprKind.COLUMN:
        if exprs.nodes[root].name == whom:
            return exprs.column(into.copy())
        return root

    var kids = exprs.nodes[root].children.copy()
    var grown = List[Int](capacity=len(kids))
    var same = True
    for i in range(len(kids)):
        var one = _mirror(exprs, kids[i], whom, into)
        if one != kids[i]:
            same = False
        grown.append(one)
    if same:
        return root
    return exprs.rebuild(root, grown^)


def _only(names: List[String], mine: Schema, theirs: Schema) -> Bool:
    """Whether every name is one schema's and none of them is the other's.

    The same question pushdown asks before sending a predicate into one side of
    a join, and asked here for the same reason: a name both sides have is a name
    this cannot tell apart.

    Args:
        names: The columns the predicate reads.
        mine: The schema it would belong to.
        theirs: The schema on the other side.

    Returns:
        True if it is unambiguously one side's.
    """
    if len(names) == 0:
        return False
    for i in range(len(names)):
        if not mine.has(names[i]) or theirs.has(names[i]):
            return False
    return True


def _inside(plan: Plan, at: Int, mut out: List[Int]) raises:
    """Collects the conjuncts of every filter an arm asks before the join.

    Stops at a projection or an aggregate, since below one of those the names
    are the input's rather than the arm's.

    Args:
        plan: The plan.
        at: The node to walk from.
        out: The conjuncts, appended to.

    Raises:
        If an expression is not in the arena.
    """
    var kind = plan.nodes[at].kind
    if kind == NodeKind.PROJECT or kind == NodeKind.AGGREGATE:
        return
    if kind == NodeKind.FILTER:
        plan.exprs.conjuncts(plan.nodes[at].exprs[0], out)
    var inputs = plan.nodes[at].inputs.copy()
    for i in range(len(inputs)):
        _inside(plan, inputs[i], out)
