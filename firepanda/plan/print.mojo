"""What a plan looks like when someone asks what it is going to do.

Two forms, and the reason to have both is the reason to keep the DSL and the IR
apart in the first place. One prints what the caller wrote and one prints what
will run, and every hour spent debugging a plan is spent looking at the two side
by side. There is only one plan representation so far, so `explain` prints it and
the second form arrives with the first pass that changes anything.

The shape is a tree, one node a line, indented by depth, children below their
parent. A `JOIN` prints both of its inputs and a `UNION` prints all of them, so
the indent is what says which subtree a line belongs to.

An expression prints in the notation it was written in rather than as a tree,
because a filter over five predicates is one line that way and eleven lines the
other, and the point of an explain output is to be read. Parentheses go around
every compound operand rather than only where precedence needs them. That is
noisier on `a + b * 2` and it is right on everything else, since a printed plan
that leans on the reader knowing the precedence table is a printed plan that gets
misread.

A bound column prints as its name and not as its position, for the same reason.
The position is what execution uses and the name is what the caller wrote.
"""

from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.unary import UnaryOp
from firepanda.plan.expr import ExprKind, Expressions
from firepanda.plan.node import NO_LIMIT, NodeKind, Plan


def _compound(tree: Expressions, at: Int) -> Bool:
    """Whether an expression needs brackets around it as an operand.

    Args:
        tree: The arena.
        at: The expression.

    Returns:
        True if it is anything but a leaf.
    """
    ref node = tree.nodes[at]
    return not (node.kind == ExprKind.COLUMN or node.kind == ExprKind.LITERAL)


def render_expr(tree: Expressions, root: Int) raises -> String:
    """Writes an expression the way it was written.

    Args:
        tree: The arena.
        root: The expression.

    Returns:
        The text.

    Raises:
        If the expression is not in the arena.
    """
    tree.check(root)
    ref node = tree.nodes[root]

    if node.kind == ExprKind.COLUMN:
        return node.name

    if node.kind == ExprKind.LITERAL:
        return String(node.value)

    if node.kind == ExprKind.UNARY:
        var over = render_expr(tree, node.children[0])
        if _compound(tree, node.children[0]):
            over = String("(", over, ")")
        return String(UnaryOp(node.op), over)

    if node.kind == ExprKind.BINARY:
        var left = render_expr(tree, node.children[0])
        if _compound(tree, node.children[0]):
            left = String("(", left, ")")
        var right = render_expr(tree, node.children[1])
        if _compound(tree, node.children[1]):
            right = String("(", right, ")")
        return String(left, " ", BinaryOp(UInt8(node.op)), " ", right)

    if node.kind == ExprKind.CAST:
        return String(render_expr(tree, node.children[0]), "::", node.type)

    if node.kind == ExprKind.AGGREGATE:
        return String(
            AggKind(UInt8(node.op)),
            "(",
            render_expr(tree, node.children[0]),
            ")",
        )

    if node.kind == ExprKind.CONDITIONAL:
        return String(
            "if ",
            render_expr(tree, node.children[0]),
            " then ",
            render_expr(tree, node.children[1]),
            " else ",
            render_expr(tree, node.children[2]),
        )

    if node.kind == ExprKind.WINDOW:
        var written = String(
            AggKind(UInt8(node.op)),
            "(",
            render_expr(tree, node.children[0]),
            ") over (",
        )
        for i in range(node.parts):
            written += "partition " if i == 0 else ", "
            written += render_expr(tree, node.children[1 + i])
        for i in range(1 + node.parts, len(node.children)):
            written += "order " if i == 1 + node.parts else ", "
            written += render_expr(tree, node.children[i])
        return written + ")"

    var written = String(node.name, "(")
    for i in range(len(node.children)):
        if i != 0:
            written += ", "
        written += render_expr(tree, node.children[i])
    return written + ")"


def _list(
    tree: Expressions, exprs: List[Int], first: Int, last: Int
) raises -> String:
    """Writes a run of expressions as a comma separated list.

    Args:
        tree: The arena.
        exprs: The expressions.
        first: Where the run starts.
        last: Where it ends, exclusive.

    Returns:
        The text.

    Raises:
        If an expression is not in the arena.
    """
    var written = String()
    for i in range(first, last):
        if i != first:
            written += ", "
        written += render_expr(tree, exprs[i])
    return written


def _line(plan: Plan, at: Int) raises -> String:
    """Writes one node, without its children and without its indent.

    Args:
        plan: The plan.
        at: The node.

    Returns:
        The text.

    Raises:
        If an expression on the node is not in the arena.
    """
    ref node = plan.nodes[at]

    if node.kind == NodeKind.SCAN:
        var written = String("SCAN ", node.source, " [")
        for i in range(len(node.names)):
            if i != 0:
                written += ", "
            written += node.names[i]
        return written + "]"

    if node.kind == NodeKind.FILTER:
        return String("FILTER ", render_expr(plan.exprs, node.exprs[0]))

    if node.kind == NodeKind.PROJECT:
        var written = String("PROJECT [")
        for i in range(len(node.exprs)):
            if i != 0:
                written += ", "
            var out = render_expr(plan.exprs, node.exprs[i])
            written += out
            # A name that repeats what the expression already says is noise, and
            # a projection of plain columns is most of the projections there
            # are, so the alias is only printed when it renames something.
            if node.names[i] != out:
                written += String(" as ", node.names[i])
        return written + "]"

    if node.kind == NodeKind.AGGREGATE:
        return String(
            "AGGREGATE [",
            _list(plan.exprs, node.exprs, 0, node.parts),
            "] -> [",
            _list(plan.exprs, node.exprs, node.parts, len(node.exprs)),
            "]",
        )

    if node.kind == NodeKind.JOIN:
        var pairs = String()
        for i in range(node.parts):
            if i != 0:
                pairs += ", "
            pairs += render_expr(plan.exprs, node.exprs[i])
            pairs += " = "
            pairs += render_expr(plan.exprs, node.exprs[node.parts + i])
        return String("JOIN ", JoinKind(UInt8(node.op)), " [", pairs, "]")

    if node.kind == NodeKind.SORT:
        var written = String("SORT [")
        for i in range(len(node.exprs)):
            if i != 0:
                written += ", "
            written += render_expr(plan.exprs, node.exprs[i])
            written += " desc" if node.flags[i] else " asc"
            if node.flags[len(node.exprs) + i]:
                written += " nulls last"
        return written + "]"

    if node.kind == NodeKind.LIMIT:
        var written = String("LIMIT ")
        written += "all" if node.length == NO_LIMIT else String(node.length)
        if node.offset != 0:
            written += String(" offset ", node.offset)
        return written

    if node.kind == NodeKind.DISTINCT:
        if len(node.exprs) == 0:
            return String("DISTINCT [*]")
        return String(
            "DISTINCT [",
            _list(plan.exprs, node.exprs, 0, len(node.exprs)),
            "]",
        )

    return String("UNION all" if node.flags[0] else "UNION")


def explain(plan: Plan, root: Int) raises -> String:
    """Writes a plan as an indented tree, one node a line.

    Args:
        plan: The plan.
        root: Where to start, which is the node whose answer is the query's.

    Returns:
        The text, with a newline after every line including the last.

    Raises:
        If the node is not in the plan, or an expression on it is not in the
        arena.
    """
    return _explain_at(plan, root, 0)


def _explain_at(plan: Plan, root: Int, depth: Int) raises -> String:
    """Writes a subtree at a depth.

    Args:
        plan: The plan.
        root: The node.
        depth: How far to indent it.

    Returns:
        The text.

    Raises:
        As `explain` does.
    """
    if root < 0 or root >= len(plan.nodes):
        raise Error(
            String("plan node ", root, " is not in a plan of ", len(plan.nodes))
        )
    var written = String()
    for _ in range(depth):
        written += "  "
    written += _line(plan, root)
    written += "\n"
    ref node = plan.nodes[root]
    for i in range(len(node.inputs)):
        written += _explain_at(plan, node.inputs[i], depth + 1)
    return written
