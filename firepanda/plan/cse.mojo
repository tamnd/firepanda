"""Common subexpression elimination: one shape is one expression.

Section 5 of `docs/specs/planner/02-the-pass-pipeline.md`, the half of it about
expressions rather than about subplans. If the same subtree appears twice it is
computed once.

For a dataframe library this matters more than it does for SQL, because the
arena hands out an index per call rather than per shape and a caller writing
Python naturally repeats themselves. `df.with_columns(disc_price = price * (1 -
discount), charge = price * (1 - discount) * (1 + tax))` is TPC-H q1 written the
way a person writes it, and it builds the product twice, as two indices that
have never heard of each other.

## What it does

Within one plan node, two expressions that are the same shape become one index.
Lowering is what turns that into a saving: it keeps a memo from arena index to
column position and computes an index it meets twice once, so making two shapes
into one index is exactly the same thing as making two passes over the data into
one.

The shape is decided by a key built out of everything the node is, which is its
kind, its operation, its name, its constant and its type, followed by the keys
its operands got. Two nodes with equal keys compute equal values, because a key
holds every field that goes into the answer.

## Why one node at a time

An expression is unified against the other expressions of the plan node holding
it and never against another node's. Binding writes a position and a table onto
a column, and the same column name under two plan nodes can bind to two
different positions, so two subtrees that look alike across nodes are not alike.
Inside one node every expression binds against the same input schema, which is
what makes equal shapes mean equal values.

It is also where the saving is. Lowering's memo is per node for the same
reason, so an index shared across two nodes would be computed once per node
anyway and unifying it would buy nothing.

## Why it adds nodes rather than editing them

An index can be read by more than one plan node, so rewriting one in place
would change an expression under a node that never asked. `Expressions.rebuild`
copies a node over new operands and the walk only copies the path it changed,
which is the same rule `graft` follows.

The nodes it leaves behind stay in the arena. Nothing reaches them and they cost
one struct each, which is the trade every pass here makes rather than paying a
compaction to reclaim them.
"""

from firepanda.dtype.schema import Schema
from firepanda.plan.bind import bind
from firepanda.plan.expr import ExprKind, Expressions
from firepanda.plan.node import NodeKind, Plan


def cse(mut plan: Plan, root: Int, sources: List[Schema]) raises -> Schema:
    """Makes each shape of expression one expression, within each node.

    Args:
        plan: The plan, rewritten.
        root: The node whose answer is the query's.
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

    var changed = False
    for at in range(len(plan.nodes)):
        if not live[at] or len(plan.nodes[at].exprs) == 0:
            continue
        # A join's expressions do not all bind against one schema. Its left
        # keys read the left input and its right keys read the right, so two of
        # them being the same shape does not make them the same value. They are
        # bare columns anyway and there is nothing here to save.
        if plan.nodes[at].kind == NodeKind.JOIN:
            continue
        var keys = List[String]()
        var canon = List[Int]()
        var held = plan.nodes[at].exprs.copy()
        var made = List[Int](capacity=len(held))
        var same = True
        for i in range(len(held)):
            var one = _canon(plan.exprs, held[i], keys, canon)
            if one != held[i]:
                same = False
            made.append(one)
        if not same:
            plan.nodes[at].exprs = made^
            changed = True

    if not changed:
        return out^
    return bind(plan, root, sources)


def _canon(
    mut exprs: Expressions,
    root: Int,
    mut keys: List[String],
    mut canon: List[Int],
) raises -> Int:
    """Returns the one index that stands for an expression's shape.

    Args:
        exprs: The arena, added to.
        root: The expression.
        keys: The shapes seen so far in this node, added to.
        canon: The index each of those shapes stands as, added to.

    Returns:
        The first index that had this shape, which is `root` itself the first
        time the shape is seen and nothing under it moved.

    Raises:
        If an expression is not in the arena.
    """
    exprs.check(root)
    var kids = exprs.nodes[root].children.copy()
    var grown = List[Int](capacity=len(kids))
    var same = True
    for i in range(len(kids)):
        var one = _canon(exprs, kids[i], keys, canon)
        if one != kids[i]:
            same = False
        grown.append(one)

    var tokens = List[String](capacity=len(grown))
    for i in range(len(grown)):
        tokens.append(String(grown[i]))

    var key = key_for(exprs, root, tokens)
    for i in range(len(keys)):
        if keys[i] == key:
            return canon[i]

    var at = root if same else exprs.rebuild(root, grown^)
    keys.append(key^)
    canon.append(at)
    return at


def key_for(exprs: Expressions, root: Int, kids: List[String]) raises -> String:
    """Writes down everything about a node that decides what it computes.

    Not a hash. A key that collides is a wrong answer rather than a slow one,
    and a node has a handful of expressions in it, so comparing the strings is
    cheaper than being sure a hash cannot collide.

    The operands arrive as tokens rather than as indices because the two callers
    want two different tokens out of the same fields. This pass unifies within
    one node, where equal shapes below already share an index, so an index is a
    shape and passing one keeps the key a constant amount of work per node
    rather than growing with the depth of the tree. Subplan elimination compares
    expressions under two different nodes, which have never been unified against
    each other and so have no index in common, and it passes the operand's own
    key.

    Args:
        exprs: The arena.
        root: The node.
        kids: One token per operand, standing for what the operand computes.

    Returns:
        The key.

    Raises:
        If the node is not in the arena.
    """
    ref node = exprs.nodes[root]
    var written = String(
        node.kind.code,
        "|",
        node.type,
        "|",
        node.name,
        "|",
        node.at,
        "|",
        node.table,
        "|",
        node.op,
        "|",
        node.rowwise,
        "|",
        node.parts,
    )
    # A literal's value is what it computes, and two constants of one type that
    # print the same are the same constant.
    if node.kind == ExprKind.LITERAL:
        written += String("|", node.value)
    for i in range(len(kids)):
        written += String("|", kids[i])
    return written^
