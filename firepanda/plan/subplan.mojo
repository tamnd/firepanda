"""Common subplan elimination: one shape of subtree is one node.

The other half of section 5 of `docs/specs/planner/02-the-pass-pipeline.md`, one
level up from the expressions. If the same subtree appears twice it is computed
once. Polars turns this on by default in `collect()`.

For a dataframe library this matters more than it does for SQL, because a person
writing Python naturally repeats themselves. `df.filter(cond).select(a)` and
`df.filter(cond).select(b)` on two adjacent lines is a common subplan and nobody
would write the SQL that way.

## What it does

Two plan nodes that are the same shape over the same inputs become one node.
Everything that read the second reads the first, and the second is left in the
arena with nothing pointing at it.

The shape is a key built out of every field of the node: its kind, its join
kind, its offset and length, its table and source, its names and its sort
directions, the shapes of its expressions, and the nodes its inputs settled on.
Two nodes with equal keys produce equal output, because a key holds every field
that goes into the answer.

An expression's shape is the key `cse.key_for` writes, with each operand's own
key in the place where elimination within a node puts an index. It has to be the
key rather than the index here, because the two nodes being compared have never
been unified against each other and so share no indices at all, even where they
compute the same thing.

## Why the root never moves

Unlike the passes that take a node out, this one hands back a schema rather than
a node, because the root cannot be the one that goes. A node it unified the root
with would have to be a node the root reaches, a node the root reaches is
strictly shallower than the root, and two nodes with equal keys have equal
structure and so equal depth. So there is no such node, and the caller's root is
still the caller's root.

It only considers the nodes the root reaches, for the same reason nothing else
here looks at the rest. The arena keeps every node that was ever built,
including the ones earlier passes stopped reading, and unifying a live node with
a dead one would bring the dead one back.

## Why it runs after the sweeps rather than inside one

It is the one pass that leaves the plan a graph rather than a tree, and every
other pass is written for a tree.

Two of them would be wrong on a node with two parents. Projection pushdown
narrows a node to the columns the node above it asked for, and a node with two
parents has two answers to that. Slice pushdown swaps the contents of two
adjacent nodes, which rewrites the lower node under whoever else was reading it.
The rest are merely wasteful rather than wrong, since they would walk a shared
node once per path to it.

So the sharing happens once, after the loop has settled, and no pass ever sees
it. That also matches where the saving is. Nothing is saved by the plan being
smaller; the saving is one scan and one filter at run time instead of two, and
that is cashed when the plan is lowered rather than while it is being rewritten.

## What it does not do yet

The executor cannot spend it. Lowering walks a line of operators and a shared
node is a fork, so a plan that this pass turned into a graph lowers as though
the sharing were not there, if it lowers at all. The pass is still right and
still worth having now: it is what the spec asks for, it is what tells the
lowering work what shape of plan to expect, and the day lowering grows a node
that can hand one chunk stream to two readers, the plans arriving at it already
say where to put one.

## Why it is invisible to `explain`

The printed plan is a tree, so a node reached twice prints twice and prints the
same before and after. That is the same property common subexpression
elimination has, and it means a sweep in which this was the only pass to act
reads as a sweep in which nothing happened. It is also why this pass is tested
by counting nodes rather than by reading the printed plan.
"""

from firepanda.dtype.schema import Schema
from firepanda.plan.bind import bind
from firepanda.plan.cse import key_for
from firepanda.plan.expr import Expressions
from firepanda.plan.node import Plan


def subplan(mut plan: Plan, root: Int, sources: List[Schema]) raises -> Schema:
    """Makes each shape of subtree one node.

    Args:
        plan: The plan, rewritten in place.
        root: The node whose output is the answer.
        sources: A schema per relation a scan can name.

    Returns:
        The schema of the root, which the pass does not change.

    Raises:
        If the plan does not bind, or an expression is not in the arena.
    """
    var out = bind(plan, root, sources)

    var live = List[Bool](length=len(plan.nodes), fill=False)
    live[root] = True
    for at in range(len(plan.nodes) - 1, -1, -1):
        if not live[at]:
            continue
        var inputs = plan.nodes[at].inputs.copy()
        for i in range(len(inputs)):
            live[inputs[i]] = True

    # The arena hands out indices in creation order, so an input is always below
    # the node reading it and one pass upward settles every node after its
    # inputs have settled.
    var shapes = List[String](length=len(plan.exprs.nodes), fill=String())
    var known = List[Bool](length=len(plan.exprs.nodes), fill=False)
    var keys = List[String]()
    var canon = List[Int]()
    var stands = List[Int](length=len(plan.nodes), fill=0)
    var changed = False

    for at in range(len(plan.nodes)):
        stands[at] = at
        if not live[at]:
            continue

        var inputs = plan.nodes[at].inputs.copy()
        for i in range(len(inputs)):
            var was = inputs[i]
            inputs[i] = stands[was]
            if inputs[i] != was:
                changed = True
        plan.nodes[at].inputs = inputs.copy()

        var key = _key(plan, at, inputs, shapes, known)
        var had = -1
        for i in range(len(keys)):
            if keys[i] == key:
                had = canon[i]
                break
        if had >= 0:
            stands[at] = had
            changed = True
            continue
        keys.append(key^)
        canon.append(at)

    if not changed:
        return out^
    return bind(plan, root, sources)


def _key(
    plan: Plan,
    at: Int,
    inputs: List[Int],
    mut shapes: List[String],
    mut known: List[Bool],
) raises -> String:
    """Writes down everything about a plan node that decides what it produces.

    Args:
        plan: The plan.
        at: The node.
        inputs: The nodes its inputs settled on.
        shapes: The key each expression has, once anybody has asked for it.
        known: Which of those have been asked for.

    Returns:
        The key.

    Raises:
        If an expression is not in the arena.
    """
    ref node = plan.nodes[at]
    var written = String(
        node.kind.code,
        "|",
        node.parts,
        "|",
        node.op,
        "|",
        node.offset,
        "|",
        node.length,
        "|",
        node.table,
        "|",
        node.source,
    )
    for i in range(len(node.names)):
        written += String("|n", node.names[i])
    for i in range(len(node.flags)):
        written += String("|f", node.flags[i])
    for i in range(len(node.exprs)):
        written += String(
            "|e", _shape(plan.exprs, node.exprs[i], shapes, known)
        )
    for i in range(len(inputs)):
        written += String("|i", inputs[i])
    return written^


def _shape(
    exprs: Expressions,
    root: Int,
    mut shapes: List[String],
    mut known: List[Bool],
) raises -> String:
    """Returns the key of an expression tree, computing it if nobody has.

    Kept per index rather than recomputed, because two nodes that share a
    subtree would otherwise walk it once each and the whole point of asking is
    that they might.

    Args:
        exprs: The arena.
        root: The expression.
        shapes: The key each expression has, once anybody has asked for it.
        known: Which of those have been asked for.

    Returns:
        The key.

    Raises:
        If the expression is not in the arena.
    """
    exprs.check(root)
    if known[root]:
        return shapes[root].copy()

    var kids = exprs.nodes[root].children.copy()
    var tokens = List[String](capacity=len(kids))
    for i in range(len(kids)):
        tokens.append(_shape(exprs, kids[i], shapes, known))

    var key = key_for(exprs, root, tokens)
    shapes[root] = key.copy()
    known[root] = True
    return key^
