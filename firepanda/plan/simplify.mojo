"""The first pass: make every expression smaller before any row is read.

This is the cheapest pass in the pipeline and the one whose absence costs the
most per row, because an expression that does not depend on any row is
otherwise evaluated once per row anyway. `date '1998-12-01' - interval '90
days'` in a TPC-H predicate is six million subtractions of two constants, and
every one of them answers the same thing.

The pass does four things and runs them to a fixed point.

### Folding

An expression that reads no column is replaced by its answer. What decides
"reads no column" is `Expressions.input_independent`, which already exists and
is already tested, so this pass does not get its own opinion about it.

The answer itself is not computed here either. Folding builds a one row column
holding the left operand, hands it to the same `binary_value_any` or
`unary_any` the executor would have called, and reads row zero back out. That
costs an allocation per fold, at plan time, once, and it buys the only property
that matters: the folded constant is bit for bit the value the kernel would
have produced on every row. A second implementation of the arithmetic that
agreed with the kernel almost always would be worse than no folding at all,
because the disagreement would show up as a wrong answer in exactly the queries
this pass was added to speed up.

A fold that raises is not a fold. A constant divided by a constant zero, or a
comparison between two types with no common type, hands the expression back
untouched so that the error surfaces from execution where the user can see
which row and which value produced it, rather than from a pass they did not ask
to run. Binding is where a type error is supposed to be caught, and binding has
already run by the time this does.

### Turning a comparison round

`5 > x` becomes `x < 5`. Nothing is faster for having done it, and that is not
the point: every later pass that asks "is this a comparison between a column
and a constant" has to ask it once rather than twice, and pushdown into a
Parquet reader, a runtime filter and a range check are all that question. Only
comparisons turn round, because `5 - x` is not `x - 5`.

### Flattening the connectives

`a AND (b AND c)` becomes `and(a, b, c)`. `AND` and `OR` are calls with a child
list rather than binaries, so flattening is the natural shape rather than a
trick, and it is what makes the next pass able to look at a conjunction and
decide which of its parts can be pushed where. A conjunction that stays a tree
has to be walked to find its parts, and every pass that walks it has to know
that `AND` is associative, which is knowledge better spent once here.

### The identities

`and(x)` is `x`, `and(x, true)` is `x`, `and(x, false)` is false, and the same
three with `or` the other way up. `not(not(x))` is `x`. These look too small to
write down and they are not: they are what a folded subexpression leaves behind.
A predicate like `l_shipdate >= '1994-01-01' and 1 = 1` is not something anybody
writes, but it is exactly what a folded `1 = 1` produces, and without the
identities the pass would have made the plan larger rather than smaller.

### What is deliberately not here

Collapsing two comparisons on the same column into a range, and collapsing a
comparison against a value outside the column's type range into a constant.
Both are listed in docs/specs/planner/02-the-pass-pipeline.md and both want a
range representation that the predicate pushdown pass wants too, so they land
with it rather than being written twice.

Nothing here looks at a plan node other than to find the expressions hanging off
it. Subplan and subexpression elimination are separate passes and they work on
the tree this one leaves behind, which is smaller and more uniform than the one
it was given, which is the whole reason this pass runs first.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.array.value import Value
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import LogicalType
from firepanda.frame.align import value_at
from firepanda.kernel.binary import BinaryOp, binary_value_any
from firepanda.kernel.unary import UnaryOp, unary_any
from firepanda.plan.expr import ExprKind, Expressions
from firepanda.plan.node import Plan

comptime ROUNDS = 8
"""How many times the pass may rewrite an expression before it gives up.

A fixed point is reached in one or two rounds on everything measured, because
every rewrite here makes the tree strictly smaller except turning a comparison
round, which cannot repeat. The bound is here so that a rule added later which
does repeat is a slow plan rather than a hung process, and so that the loop has
an answer to "what if two rules undo each other" that does not depend on nobody
ever writing that pair.
"""


def _one_row(value: Value) raises -> AnyArray:
    """Builds a column of exactly one row holding a constant.

    This is what lets the fold ask the kernel rather than answer for itself. It
    is not `align._constant`, which refuses text because a fill value has no
    arithmetic to take part in; a folded comparison between two strings is a
    real thing to ask and the answer comes from the same comparison loop that
    runs over a column.

    Args:
        value: The constant.

    Returns:
        A one row column of the value's own type, holding a null if the value
        is one.

    Raises:
        Error: If the value's type has no physical layout.
    """
    if value.type.is_variable_width():
        var builder = StringBuilder(capacity=1)
        if value.is_null():
            builder.append_null()
        else:
            builder.append(value.as_string().as_bytes())
        return AnyArray(builder^.finish())

    comptime for target in ALL:
        if value.type.physical == target:
            var out = Array[target](1)
            if not value.is_null():
                out.set_valid(0, value.as_scalar[target]())
            return AnyArray(out^).retyped(value.type)
    raise Error(
        String(
            "simplify: there is no way to hold a constant of type ",
            value.type,
            " in a column, so it cannot be folded",
        )
    )


def _fold(mut exprs: Expressions, root: Int) raises -> Int:
    """Evaluates a binary or unary over constants and returns a literal.

    Args:
        exprs: The arena.
        root: The expression, whose children are already literals.

    Returns:
        The index of a new literal holding the answer, or `root` unchanged if
        the kernel refused it.

    Raises:
        Error: Only what the arena raises.
    """
    var kind = exprs.nodes[root].kind
    var op = exprs.nodes[root].op
    var kids = exprs.nodes[root].children.copy()

    # A fold that raises hands the expression back rather than failing the
    # plan. See the module docstring: a division by a constant zero is the
    # user's error to see from execution, not this pass's to report.
    try:
        if kind == ExprKind.UNARY:
            var only = _one_row(exprs.nodes[kids[0]].value)
            var answered = unary_any(only, UnaryOp(op))
            return exprs.literal(value_at(answered, 0))

        var left = _one_row(exprs.nodes[kids[0]].value)
        var right = Value(copy=exprs.nodes[kids[1]].value)
        var answer = binary_value_any(left, right, BinaryOp(UInt8(op)))
        return exprs.literal(value_at(answer, 0))
    except:
        return root


def _all_literal(exprs: Expressions, of: List[Int]) -> Bool:
    """Whether every one of a node's children is already a literal.

    Args:
        exprs: The arena.
        of: The children.

    Returns:
        True if all of them are literals. An empty list is True, which no
        caller relies on.
    """
    for i in range(len(of)):
        if exprs.nodes[of[i]].kind != ExprKind.LITERAL:
            return False
    return True


def _truth(exprs: Expressions, at: Int) -> Int:
    """Reads a literal as a known truth value.

    Args:
        exprs: The arena.
        at: The expression.

    Returns:
        1 for a literal true, 0 for a literal false, and -1 for anything else,
        including a null, which is neither and cannot be dropped from a
        conjunction.
    """
    ref node = exprs.nodes[at]
    if node.kind != ExprKind.LITERAL:
        return -1
    if node.value.type != LogicalType.BOOL or node.value.is_null():
        return -1
    return 1 if node.value.bits != 0 else 0


def _connective(exprs: Expressions, at: Int, name: String) -> Bool:
    """Whether an expression is a call to one named connective.

    Args:
        exprs: The arena.
        at: The expression.
        name: `and`, `or` or `not`.

    Returns:
        True if it is that call.
    """
    ref node = exprs.nodes[at]
    return node.kind == ExprKind.CALL and node.name == name


def _rewrite(mut exprs: Expressions, root: Int) raises -> Int:
    """Applies every rule once, bottom up, and returns the new root.

    Bottom up because every rule reads its children and each of them wants to
    see the rewritten child rather than the original. A literal that appears
    only after its own subtree folded is the common case, not the exception.

    Args:
        exprs: The arena.
        root: The expression.

    Returns:
        The index of the rewritten expression, which is `root` when no rule
        fired.

    Raises:
        Error: If the expression is not in the arena.
    """
    exprs.check(root)
    var kind = exprs.nodes[root].kind
    if kind == ExprKind.COLUMN or kind == ExprKind.LITERAL:
        return root

    var kids = exprs.nodes[root].children.copy()
    var moved = False
    for i in range(len(kids)):
        var after = _rewrite(exprs, kids[i])
        if after != kids[i]:
            kids[i] = after
            moved = True
    if moved:
        exprs.nodes[root].children = kids.copy()

    if kind == ExprKind.UNARY or kind == ExprKind.BINARY:
        if _all_literal(exprs, kids):
            return _fold(exprs, root)

    if kind == ExprKind.BINARY:
        var op = BinaryOp(UInt8(exprs.nodes[root].op))
        # The constant goes on the right so that later passes ask one question
        # rather than two. Only a comparison can be turned round.
        if (
            op.is_comparison()
            and exprs.nodes[kids[0]].kind == ExprKind.LITERAL
            and exprs.nodes[kids[1]].kind != ExprKind.LITERAL
        ):
            return exprs.binary(op.mirrored(), kids[1], kids[0])
        return root

    if kind != ExprKind.CALL:
        return root

    var name = exprs.nodes[root].name
    if name == "not":
        # Two negations cancel. A folded `not` over a literal has already been
        # handled by the call fold below.
        if _connective(exprs, kids[0], "not"):
            return exprs.nodes[kids[0]].children[0]
        var known = _truth(exprs, kids[0])
        if known >= 0:
            return exprs.literal(Value(known == 0))
        return root

    if name != "and" and name != "or":
        return root

    var conjunction = name == "and"
    # The value that makes the whole connective collapse: a false under `and`,
    # a true under `or`. The other one is the identity and simply drops out.
    var absorbing = 0 if conjunction else 1
    var flat = List[Int]()
    for i in range(len(kids)):
        var at = kids[i]
        if _connective(exprs, at, name):
            ref inner = exprs.nodes[at]
            for j in range(len(inner.children)):
                flat.append(inner.children[j])
            continue
        var known = _truth(exprs, at)
        if known == absorbing:
            return exprs.literal(Value(absorbing == 1))
        if known >= 0:
            continue
        flat.append(at)

    if len(flat) == 0:
        # Everything was the identity, so the connective is that identity.
        return exprs.literal(Value(absorbing == 0))
    if len(flat) == 1:
        return flat[0]
    if len(flat) == len(kids):
        var same = True
        for i in range(len(flat)):
            if flat[i] != kids[i]:
                same = False
                break
        if same:
            return root
    return exprs.call(String(name), flat^, rowwise=True)


def simplify_expr(mut exprs: Expressions, root: Int) raises -> Int:
    """Rewrites one expression until nothing more changes.

    Args:
        exprs: The arena.
        root: The expression.

    Returns:
        The index of the simplified expression, which is `root` if no rule ever
        fired. The original nodes stay in the arena and are simply no longer
        reachable, which is what an arena is for.

    Raises:
        Error: If the expression is not in the arena.
    """
    var at = root
    for _ in range(ROUNDS):
        var after = _rewrite(exprs, at)
        if after == at:
            return at
        at = after
    return at


def simplify(mut plan: Plan, root: Int) raises:
    """Rewrites every expression hanging off every node of a plan.

    Nodes are visited in index order rather than from the root, because an
    expression belongs to the node that holds it and cannot be reached from
    anywhere else, so reachability does not change which expressions want
    rewriting and a flat loop is the honest shape.

    Args:
        plan: The plan, rewritten in place.
        root: The root node, checked so that a caller passing a stale index
            finds out here rather than two passes later.

    Raises:
        Error: If the root is not in the plan.
    """
    plan.check(root)
    for at in range(len(plan.nodes)):
        var held = plan.nodes[at].exprs.copy()
        var moved = False
        for i in range(len(held)):
            var after = simplify_expr(plan.exprs, held[i])
            if after != held[i]:
                held[i] = after
                moved = True
        if moved:
            plan.nodes[at].exprs = held^
