"""Turning a logical plan into a pipeline of physical operators.

This is the join between the two halves of the engine. Everything above it
says what the query wants, in a tree of nodes holding a tree of expressions.
Everything below it says how one chunk is computed, in a line of operators that
each take a chunk and give one back. Nothing in `firepanda/plan/` was reachable
from a running query before this file, which is why the passes could be written
and tested without moving a row, and why they stop being worth anything until
this exists.

### An expression tree becomes a line of appends

`Compute` says it in its own docstring: an expression is a tree and a tree is a
line of these. `(a + b) < 5` lowers to a `Compute` that appends `a + b` at the
end of the chunk and a second one that compares that new column against five,
and the intermediate is dropped by the `Project` at the end rather than by
anything in between. So lowering an expression is a post order walk that returns
a column position, and the position of a subexpression is wherever its `Compute`
put it.

Appending is what makes this safe. A `Compute` never touches a position that
already exists, so every position a bound expression carries still means what
binding said it meant, no matter how many intermediates have been added since.
That is the one property this file leans on everywhere, and it is why `Cast`,
which converts in place, is only allowed here on a column this lowering made.
Casting an input column would change what a position means underneath an
expression that was bound before the cast was added.

### An expression reached twice is computed once

Two of a node's expressions can be the same expression. The arena hands out an
index per call rather than per shape, so that only happens when something shared
one on purpose, which is what `graft` does when a projection merges into the one
below it and what a caller does when it holds on to a handle. Either way the
walk would meet the same index twice and append the same `Compute` twice, which
over six million rows is a second pass for a column that is already sitting in
the chunk.

So the walk remembers where it put things. The memo is per node, because the
positions in it are positions in a chunk that only exists between one node's
`base` and its trim, and it is consulted for every subexpression but not for the
top of an output. An output has a name and the column it lands in carries that
name, so two outputs that share a top would be one column answering to two
names. Sharing what is underneath them costs nothing and is where the work is.

A cast is the one thing that can make the memo wrong, because it converts in
place rather than appending, so the position it was handed now holds something
else. Lowering one forgets that position. The expression that was there gets
computed again if it is wanted again, which is the slow answer rather than the
wrong one.

### A conjunction becomes a line of filters

The physical layer has thirteen binary operations and none of them is `AND`, so
there is no column that holds the answer to `a AND b`. That sounds like a gap
and the thing that fills it is better than the gap: a filter whose predicate is
a conjunction lowers to one filter per conjunct, in order. Three predicates over
six million rows become a filter that reads six million, a filter that reads
what survived, and a filter that reads what survived that. Computing an and
mask would have evaluated all three on all six million rows first.

It also falls out of the representation rather than being arranged. `AND` is a
call with a child list, the simplify pass already flattened the nested ones, so
the conjuncts are just the children, and a predicate that is not a conjunction
is the one conjunct case of the same loop.

### A conditional is three columns and one node

`CASE WHEN c THEN a ELSE b END` lowers the way everything else does, three
appends and then a `Choose` that reads the three positions. A chain of `WHEN`s
arrives already nested, each one's else side being the next, so nothing here
counts branches and a four branch expression is three nodes.

Two things about it are not the general pattern. The first is the null: a null
condition takes the else side rather than making the answer null, which is what
the standard says and what separates this from an operation, where a null
operand is a null answer. The second is the types. Binding promoted the two
sides and the node needs them to agree, so a side that is not already the
promoted type gets a cast into a column of its own. Into its own column and not
in place, because the side may be an input column that something else still
reads at its own type, which is the rule this whole file is built on.

### A join stays on the line, because one of its sides is a table

A pipeline is a line and a join has two inputs, which sounds like the end of the
line and is not. The probe operator holds its build side whole and hashes it once
before the first chunk arrives, so only one of the two sides is a stream. That
side is the left one, the right one is a frame, and the join is an operator on
the left side's line the same way a filter is.

Being a frame by the time the operator is made is the condition. A scan already
is one, and its column list is applied by selecting columns of the frame rather
than by a projection, because there is no chunk to project. Anything else is a
line of the same plan, so it is lowered to a pipeline of its own and run, and
what comes back is the frame. That is why lowering a join does work rather than
only describing it: the operator holds a frame, it holds one because a pipeline
is what builds pipelines and an operator cannot hold the thing that holds it,
and the build has to finish before the left side's first chunk is read either
way.

Almost every join query needs that. A `WHERE` over the build side is what the
optimizer pushes down there, so the side stops being a scan as soon as the query
has a condition on it, which is most of the time.

The frame a scan reads is taken out of the list it was given and an empty one is
left in its place, because the numbers are relation ids rather than positions in
a list, and a list that closes up renumbers every relation above the one that
went. With one scan per plan that was invisible.

### A sort is a breaker whose schema is its input's

Every other node this lowers either leaves the columns alone or replaces them.
A sort does the first and holds every row while doing it, which makes it the one
place where a trim happens above a breaker rather than below one. `ORDER BY
a + b` appends the sum, the sort carries it along with everything else, and the
projection that drops it sits after the sort, because a position below the sort
is the same position above it.

The bound the plan may have written on the node is not read here. A sort that
only has to get the first n rows right is a different operator with a heap in it,
and the limit that wrote the bound down is still sitting above the sort and still
doing the cutting, so ignoring it is slow rather than wrong.

### A window is a breaker that comes back wider than it went in

A window holds every row the way a sort does, and for the same reason: the value
on the first row of a partition is a reduction over rows that have not arrived.
What it does that nothing else here does is hand back more columns than it was
given, because `SELECT x, sum(x) OVER ()` wants both and the node above it was
bound expecting both.

That makes the trim above it a selection rather than a prefix. The partition
keys and the expression each window reduces are computed by appends like
everything else, so the operator's own columns land after those intermediates,
and what comes out is the input's columns and then the windows with the
intermediates cut from between them.

One partitioning per operator, because that is one grouping pass and one
ordinal column. A node whose windows partition two different ways is refused
rather than split here, since splitting it changes which column each window
lands in and the node above it has already been bound against the order the
plan wrote down.

Only the partition is a frame. `OVER (ORDER BY ...)` is a running fold, which is
a different loop rather than an argument to this one, so a window with an
ordering is refused by name.

### A literal table is a frame this file builds

A scan says which frame to read and a VALUES says what the rows are, so the one
is found and the other is made. Making it is the only place lowering allocates
an array, and it costs nothing at run time because the rows were known before
the query started. `SELECT 1 + 1` is that: a query with no `FROM` is a literal
table of one row that exists only so the projection above it has something to
be evaluated over.

Every value has to be a literal by then. Constant folding has already run, so
`VALUES (1 + 1)` is one, and anything left computed is refused by name rather
than evaluated, because there is no chunk under a VALUES to evaluate it over.

### A constant that is an output gets a column of its own

A constant is nearly always an operand, and the operation it feeds reads it off
the expression tree and hands it to the kernel, so nothing is ever built for it.
What is left over is a constant that is an output on its own, which is the `1`
in `SELECT 1`, and that has to land somewhere a projection can read it back by
position like everything else. `Constant` is the node that puts it there, and it
writes one value down a buffer as wide as the chunk.

That is the second half of `SELECT 1 + 1`. The literal table under it gives the
projection a row to be evaluated over, and this gives the projection a column to
evaluate into.

### A union is its inputs stacked, and the stack is the source

Every input of a union is a line of the same plan, so each one is lowered and
run the way a join's build side is, and the frames that come back are stacked by
position into the frame the pipeline reads. By position rather than by name,
because that is what a union means: the answer takes the first input's names and
the rest line up under them whatever they call themselves.

Nothing is read to stack them. A frame is a list of chunks per column, and a
union of two frames is those lists one after the other.

A union without `ALL` is the stack with a distinct above it, which is the group
by with nothing to reduce from the section before.

A difference and an intersection are the same node with another code on it, and
they are the same stack with one more column on it saying which arm each row came
from. A group by over the query's own columns then puts every copy of a row in
one group whichever arm it came from, and the smallest and the largest tag in a
group say which arms had it, so an intersection keeps the groups whose two tags
differ and a difference keeps the groups whose largest tag is still the left's.
The tags and the mask are dropped by the filter that reads the mask.

Grouping is why this is the right shape rather than a join. SQL compares two rows
of a set operation with a null equal to a null, and a join key is never equal to
a null, so a difference written as an anti join drops every row with a null in it
and does it without saying so. A group by puts the nulls of a column in one
group, which is the rule SQL asked for. `ALL` is still refused on both, because
each needs a group to come back out as a number of rows rather than as one.

### A table function is a source with no table under it

A `range` or a `generate_series` is a source the same way a literal table is,
and the frame it stands for is built here out of the arguments the query wrote
down. That makes `SELECT * FROM range(5)` a query that reads no file and touches
no catalog, which is the shortest end to end query there is and the one worth
having in a test.

The whole series is a frame before the query starts, so a series longer than
`LONGEST_SERIES` is refused by name rather than allocated. DuckDB hands out one
chunk at a time and never holds all of it, and a source that produced its rows
as they were asked for is a driver change rather than an operator.

### A distinct is a group by that reduces nothing

A group by holds one row per group and the rows of a group differ only in what
was not the key. When the key is every column there is nothing they can differ
in, so `SELECT DISTINCT` is the group by with every position as a key and no
folds at all, and there is no operator to write. It streams for the same reason
the group by does, holding one row per distinct row rather than the input.

Part of the row is `DISTINCT ON` and goes to `Unique` instead. It keeps whole
rows chosen by some of their columns, so the rest have to come back untouched
and in place, and a group by cannot do that: it carries its keys in front of
what it reduced, which moves the columns the plan's schema numbered, and its
first skips nulls. `Unique` holds its input rather than one row per group, so
the streaming route is worth keeping for the case that can use it.

### What it refuses, and why refusing is the design

A unary expression is not lowered here, and neither is an ordered window or a
cast over an input column.
Every one of those raises an error that names what it was. It does not fall
back to `Materialize`, and the reason is that `Materialize` holds a function
pointer that captures nothing, so there is no way to hand it the expression
that could not be lowered. A fallback that cannot carry the thing it is falling
back from is not a fallback.

Raising by name is what makes this file safe to grow. A caller that gets an
error keeps whatever route it had, so lowering can be tried first and cost
nothing when it does not fit, and each operator added here is one self contained
change with a query that starts working attached to it. That is the same
argument `Materialize` makes for the physical layer, one level up.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.array.strings import StringBuilder
from firepanda.array.value import Value
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.exec.node import (
    Apply,
    Cast,
    Choose,
    Compute,
    Connective,
    Constant,
    Cut,
    Expand,
    Fill,
    Filter,
    Group,
    GroupAgg,
    Join,
    Limit,
    Match,
    Node,
    Part,
    Presence,
    Project,
    Reduce,
    Sort,
    Unique,
    Window,
)
from firepanda.exec.pipeline import Pipeline
from firepanda.frame.align import value_at
from firepanda.frame.frame import DataFrame
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.logic import LogicOp, is_logic_name, logic_op
from firepanda.kernel.pattern import MatchKind, read_pattern
from firepanda.kernel.temporal import sql_field_named
from firepanda.kernel.unary import UnaryOp
from firepanda.plan.expr import UNBOUND, ExprKind, Expressions
from firepanda.plan.node import (
    NO_LIMIT,
    SET_EXCEPT,
    SET_INTERSECT,
    SET_UNION,
    NodeKind,
    Plan,
)

comptime LONGEST_SERIES = 100_000_000
"""How many rows a table function is allowed to produce here.

A series is built before the query starts, the way a literal table is, so
`range(1000000000000)` is an allocation of eight terabytes rather than a query
that takes a while. DuckDB produces one chunk at a time and never holds the
whole thing, and until this does too the length is refused by name at a hundred
million rows, which is eight hundred megabytes and already more than a test
should ask for.
"""

comptime _SIDE = "__side"
"""What the column saying which arm a row came from is called.

A difference and an intersection stack their two inputs with this appended, and
the group by above the stack reads it back. Nothing resolves it by name, since
it is appended at the end and read at the position that puts it, so a query with
a column called the same thing is not a collision. The name is for the reader of
an explain or a crash, and the two underscores are the convention the rest of
this file already uses for a column it invented.
"""


def _spine(plan: Plan, root: Int) raises -> List[Int]:
    """Walks from the root down to the scan, following first inputs.

    A pipeline is a line, so the only plans that lower are the ones that are a
    line. A join has two inputs and is still on the line, because its left input
    is the side that stays on the pipeline and its right is the side that gets
    built into a table before the first chunk arrives. Following the first input
    is what says which of the two that is.

    Args:
        plan: The plan.
        root: The node whose output is the answer.

    Returns:
        The nodes from the scan up to the root, in the order a chunk meets them.

    Raises:
        Error: If a node on the way down has no input and is not a scan.
    """
    plan.check(root)
    var upwards = List[Int]()
    var at = root
    while True:
        upwards.append(at)
        if (
            plan.nodes[at].kind == NodeKind.SCAN
            or plan.nodes[at].kind == NodeKind.VALUES
            or plan.nodes[at].kind == NodeKind.TABLE_FUNCTION
            or plan.nodes[at].kind == NodeKind.UNION
        ):
            break
        if len(plan.nodes[at].inputs) == 0:
            raise Error(
                String(
                    "lower: a ",
                    plan.nodes[at].kind,
                    (
                        " has no input and is not a scan, so there is nothing"
                        " for a pipeline to read"
                    ),
                )
            )
        at = plan.nodes[at].inputs[0]
    var order = List[Int](capacity=len(upwards))
    while len(upwards) > 0:
        order.append(upwards.pop())
    return order^


struct Memo(Movable):
    """Where each expression a node has lowered so far ended up.

    A list rather than a dictionary, because a node has a handful of expressions
    and a scan of a handful is cheaper than hashing one, and because forgetting
    a position means looking entries up by what is in them rather than by their
    key, which a dictionary is the wrong shape for.
    """

    var of: List[Int]
    """The expression indices, in the order they were lowered."""

    var at: List[Int]
    """Where each one's value is, in the same order."""

    def __init__(out self):
        """Constructs an empty memo."""
        self.of = List[Int]()
        self.at = List[Int]()

    def place(self, of: Int) -> Int:
        """Reports where an expression already is.

        Args:
            of: The expression.

        Returns:
            Its position, or -1 if this node has not lowered it.
        """
        for i in range(len(self.of)):
            if self.of[i] == of:
                return self.at[i]
        return -1

    def remember(mut self, of: Int, at: Int):
        """Records where an expression was put.

        Args:
            of: The expression.
            at: Its position.
        """
        self.of.append(of)
        self.at.append(at)

    def forget(mut self, at: Int):
        """Drops whatever was recorded at a position, because it has changed.

        Args:
            at: The position.
        """
        for i in range(len(self.of) - 1, -1, -1):
            if self.at[i] == at:
                _ = self.of.pop(i)
                _ = self.at.pop(i)


def _conjuncts(exprs: Expressions, root: Int) -> List[Int]:
    """Splits a predicate into the parts that have to hold at once.

    Args:
        exprs: The arena.
        root: The predicate.

    Returns:
        The children of a top level `and`, flattened, or a list holding the
        predicate itself when it is not one.
    """
    var parts = List[Int]()
    var pending = List[Int]()
    pending.append(root)
    while len(pending) > 0:
        var at = pending.pop()
        ref node = exprs.nodes[at]
        if node.kind == ExprKind.CALL and node.name == "and":
            for i in range(len(node.children) - 1, -1, -1):
                pending.append(node.children[i])
            continue
        parts.append(at)
    return parts^


def _lower_expr(
    exprs: Expressions,
    root: Int,
    mut pipe: Pipeline,
    base: Int,
    name: String,
    mut memo: Memo,
    reuse: Bool = False,
) raises -> Int:
    """Appends whatever computes an expression and returns its position.

    Args:
        exprs: The arena.
        root: The expression, already bound.
        pipe: The pipeline, added to.
        base: The width of the chunk before this node started lowering.
            Positions below it are input columns and positions from it up are
            intermediates this lowering made.
        name: What to call the column, used only if the top of the expression
            is what makes one.
        memo: What this node has already computed and where it put it. Read and
            added to.
        reuse: Whether a column this node already computed may be handed back
            instead of computing it again. False from the caller, because the
            top of an output owns the name of the column it lands in, and true
            everywhere below that.

    Returns:
        The position of the column holding the expression's value.

    Raises:
        Error: If the expression has a kind no operator computes, or reads a
            column binding never gave a position to.
    """
    exprs.check(root)
    if reuse:
        var had = memo.place(root)
        if had >= 0:
            return had
    var kind = exprs.nodes[root].kind

    if kind == ExprKind.COLUMN:
        var at = exprs.nodes[root].at
        if at == UNBOUND:
            raise Error(
                String(
                    "lower: the column '",
                    exprs.nodes[root].name,
                    (
                        "' has no position, so the plan was not bound before it"
                        " was lowered"
                    ),
                )
            )
        return at

    if kind == ExprKind.LITERAL:
        # A constant is usually an operand of a `Compute` and the binary branch
        # below reads it off the tree without ever coming through here. What
        # reaches here is a constant that is an output on its own, which is
        # `SELECT 1`, and that needs a column of its own to land in.
        var type = exprs.nodes[root].type
        pipe.add(
            Node(Constant(Value(copy=exprs.nodes[root].value), type, name))
        )
        memo.remember(root, len(pipe.schema) - 1)
        return len(pipe.schema) - 1

    if kind == ExprKind.CAST:
        var over = exprs.nodes[root].children[0]
        var at = _lower_expr(exprs, over, pipe, base, name, memo, reuse=True)
        if at < base:
            # Converting an input column where it lies would change what that
            # position means for every expression already bound against it, so
            # a cast of one lands in a column of its own, the way every other
            # expression does. A cast of a column this expression just built
            # converts in place, because that column is what the cast is for.
            pipe.add(Node(Cast(at, exprs.nodes[root].type, name)))
            memo.remember(root, len(pipe.schema) - 1)
            return len(pipe.schema) - 1
        pipe.add(Node(Cast(at, exprs.nodes[root].type)))
        # The position holds the converted column now, so whatever the memo
        # says is there is no longer there, and the cast itself is not recorded
        # because a second cast of it would convert it twice.
        memo.forget(at)
        return at

    if kind == ExprKind.CALL and is_logic_name(exprs.nodes[root].name):
        return _lower_connective(exprs, root, pipe, base, name, memo)

    if kind == ExprKind.CALL and exprs.nodes[root].name == "like":
        return _lower_like(exprs, root, pipe, base, name, memo)

    if kind == ExprKind.CALL and exprs.nodes[root].name == "substring":
        return _lower_cut(exprs, root, pipe, base, name, memo)
    if kind == ExprKind.CALL and exprs.nodes[root].name == "date_part":
        return _lower_part(exprs, root, pipe, base, name, memo)

    if kind == ExprKind.CALL and (
        exprs.nodes[root].name == "is_null"
        or exprs.nodes[root].name == "is_not_null"
    ):
        return _lower_presence(exprs, root, pipe, base, name, memo)

    if kind == ExprKind.CALL and exprs.nodes[root].name == "coalesce":
        return _lower_coalesce(exprs, root, pipe, base, name, memo)

    if kind == ExprKind.CONDITIONAL:
        return _lower_conditional(exprs, root, pipe, base, name, memo)

    if kind == ExprKind.UNARY:
        # A unary over a constant has already been folded by the simplify pass,
        # so what reaches here reads a column and there is no constant form to
        # write, which is the one way this differs from the binary case below.
        var over = _lower_expr(
            exprs,
            exprs.nodes[root].children[0],
            pipe,
            base,
            name,
            memo,
            reuse=True,
        )
        pipe.add(Node(Apply(over, UnaryOp(exprs.nodes[root].op), name)))
        memo.remember(root, len(pipe.schema) - 1)
        return len(pipe.schema) - 1

    if kind != ExprKind.BINARY:
        raise Error(
            String(
                "lower: there is no operator that computes a ",
                kind,
                " expression yet",
            )
        )

    var op = BinaryOp(UInt8(exprs.nodes[root].op))
    var left = exprs.nodes[root].children[0]
    var right = exprs.nodes[root].children[1]
    var left_is_value = exprs.nodes[left].kind == ExprKind.LITERAL
    var right_is_value = exprs.nodes[right].kind == ExprKind.LITERAL

    if left_is_value and right_is_value:
        # Simplify folds these, so one here means the fold was refused, and the
        # kernel that refused it is the one that would run.
        raise Error(
            "lower: both operands of an operation are constants, which is an"
            " expression the simplify pass could not fold and no operator can"
            " compute"
        )

    if right_is_value:
        var at = _lower_expr(exprs, left, pipe, base, name, memo, reuse=True)
        pipe.add(
            Node(Compute(at, Value(copy=exprs.nodes[right].value), op, name))
        )
        memo.remember(root, len(pipe.schema) - 1)
        return len(pipe.schema) - 1

    if left_is_value:
        var at = _lower_expr(exprs, right, pipe, base, name, memo, reuse=True)
        pipe.add(
            Node(
                Compute(
                    at,
                    Value(copy=exprs.nodes[left].value),
                    op,
                    name,
                    value_on_left=True,
                )
            )
        )
        memo.remember(root, len(pipe.schema) - 1)
        return len(pipe.schema) - 1

    var at_left = _lower_expr(exprs, left, pipe, base, name, memo, reuse=True)
    var at_right = _lower_expr(exprs, right, pipe, base, name, memo, reuse=True)
    pipe.add(Node(Compute(at_left, at_right, op, name)))
    memo.remember(root, len(pipe.schema) - 1)
    return len(pipe.schema) - 1


def _lower_connective(
    exprs: Expressions,
    root: Int,
    mut pipe: Pipeline,
    base: Int,
    name: String,
    mut memo: Memo,
) raises -> Int:
    """Appends whatever computes an and, an or or a not.

    The three are calls rather than binary operations, because their rule for a
    null is not the one the operations share, and a conjunction or a disjunction
    is written with as many arguments as the query had rather than as a tree of
    pairs. So this is where the tree comes back: the arguments are folded left to
    right, one `Connective` per pair, which is what an `a AND b AND c` in a
    select list ends up as.

    Left to right is not arbitrary even though the two connectives are
    associative. It is the order the query was written in, which is the order a
    reader of the plan expects to see, and once there is a cost model to reorder
    on, reordering something is better than having to recover what was written.

    Args:
        exprs: The arena.
        root: The call, already bound.
        pipe: The pipeline, added to.
        base: The width of the chunk before this node started lowering.
        name: What to call the column the whole call lands in.
        memo: What this node has already computed and where it put it.

    Returns:
        The position of the column holding the call's value.

    Raises:
        Error: If the call has the wrong number of arguments for its
            connective, or an argument has a kind no operator computes.
    """
    var connective = logic_op(exprs.nodes[root].name)
    var args = exprs.nodes[root].children.copy()

    if connective == LogicOp.NOT:
        if len(args) != 1:
            raise Error(
                String(
                    "lower: not reads one argument and was given ",
                    len(args),
                )
            )
        var at = _lower_expr(exprs, args[0], pipe, base, name, memo, reuse=True)
        pipe.add(Node(Connective(at, name)))
        memo.remember(root, len(pipe.schema) - 1)
        return len(pipe.schema) - 1

    if len(args) < 2:
        raise Error(
            String(
                "lower: ",
                connective,
                " reads two arguments or more and was given ",
                len(args),
            )
        )

    var at = _lower_expr(exprs, args[0], pipe, base, name, memo, reuse=True)
    for i in range(1, len(args)):
        var other = _lower_expr(
            exprs, args[i], pipe, base, name, memo, reuse=True
        )
        pipe.add(Node(Connective(at, other, connective, name)))
        at = len(pipe.schema) - 1
    memo.remember(root, at)
    return at


def _lower_like(
    exprs: Expressions,
    root: Int,
    mut pipe: Pipeline,
    base: Int,
    name: String,
    mut memo: Memo,
) raises -> Int:
    """Appends whatever answers a `LIKE`.

    The pattern is read here rather than per row. A `LIKE` whose right side is a
    constant, which is every one anybody writes, is one of five searches once
    the wildcards have been counted, and which of the five it is does not change
    from row to row. So the work of deciding is done once, at plan time, and
    what goes in the pipeline is a node that knows what it is looking for.

    Four of the five are a `Match`, which is the substring kernels. The fifth is
    a pattern with no wildcard in it at all, which is an equality against a
    constant, and that is a `Compute` with the same kernel `x = 'abc'` already
    runs. Writing `LIKE 'abc'` is unusual but it is legal, and it costs nothing
    to send it somewhere that already exists.

    Args:
        exprs: The arena.
        root: The call, already bound.
        pipe: The pipeline, added to.
        base: The width of the chunk before this node started lowering.
        name: What to call the column the answer lands in.
        memo: What this node has already computed and where it put it.

    Returns:
        The position of the column holding the answer.

    Raises:
        Error: If the call has the wrong number of arguments, if the pattern is
            not a constant, if it is null, or if it has a shape none of the five
            searches covers.
    """
    var args = exprs.nodes[root].children.copy()
    if len(args) != 2:
        raise Error(
            String("lower: like reads two arguments and was given ", len(args))
        )

    if exprs.nodes[args[1]].kind != ExprKind.LITERAL:
        raise Error(
            "lower: the pattern of a LIKE has to be written out, and this one"
            " is an expression, which would mean reading a new pattern for"
            " every row and there is no kernel that does that"
        )

    if exprs.nodes[args[1]].value.is_null():
        # True for no row and false for no row either, so it is a column of
        # nulls, and there is no node that makes one of those out of nothing.
        raise Error(
            "lower: a LIKE against a null pattern is null for every row, and"
            " there is no operator that answers a column of nulls yet"
        )

    var pattern = read_pattern(exprs.nodes[args[1]].value.as_string())
    var at = _lower_expr(exprs, args[0], pipe, base, name, memo, reuse=True)

    if pattern.kind == MatchKind.EQUALS:
        # Nothing was taken out of the pattern on the way here, no wildcard
        # having been in it, so the literal itself is what to compare against.
        pipe.add(
            Node(
                Compute(
                    at,
                    Value(copy=exprs.nodes[args[1]].value),
                    BinaryOp.EQ,
                    name,
                )
            )
        )
    else:
        pipe.add(Node(Match(at, pattern, name)))

    memo.remember(root, len(pipe.schema) - 1)
    return len(pipe.schema) - 1


def _lower_cut(
    exprs: Expressions,
    root: Int,
    mut pipe: Pipeline,
    base: Int,
    name: String,
    mut memo: Memo,
) raises -> Int:
    """Appends whatever answers a `SUBSTRING`.

    The two positions are read here rather than per row, for the reason the
    pattern of a `LIKE` is: they do not change from row to row, and resolving
    them once means the node carries two numbers instead of two expressions.
    That is also the limit of what this covers. A substring whose start is
    itself a column is a different kernel, one that works out a new window for
    every row, and it is refused here rather than lowered into something that
    would answer the first row's window for all of them.

    A null position is refused for the same reason a null pattern is. The answer
    would be a column of nulls and there is no operator that makes one of those
    out of nothing yet.

    Args:
        exprs: The arena.
        root: The call, already bound.
        pipe: The pipeline, added to.
        base: The width of the chunk before this node started lowering.
        name: What to call the column the answer lands in.
        memo: What this node has already computed and where it put it.

    Returns:
        The position of the column holding the answer.

    Raises:
        Error: If the call has the wrong number of arguments, or a position is
            not a number written in the query.
    """
    var args = exprs.nodes[root].children.copy()
    if len(args) != 2 and len(args) != 3:
        raise Error(
            String(
                (
                    "lower: substring reads a column and one or two positions,"
                    " so two or three arguments, and was given "
                ),
                len(args),
            )
        )

    var numbers = List[Int]()
    for i in range(1, len(args)):
        if exprs.nodes[args[i]].kind != ExprKind.LITERAL:
            raise Error(
                String(
                    (
                        "lower: the positions of a SUBSTRING have to be written"
                        " out, and argument "
                    ),
                    i,
                    (
                        " is an expression, which would mean a new window for"
                        " every row and there is no kernel that does that"
                    ),
                )
            )
        if exprs.nodes[args[i]].value.is_null():
            raise Error(
                "lower: a SUBSTRING with a null position is null for every row,"
                " and there is no operator that answers a column of nulls yet"
            )
        numbers.append(Int(exprs.nodes[args[i]].value.as_scalar[DType.int64]()))

    var length = Optional[Int]()
    if len(numbers) == 2:
        length = numbers[1]

    var at = _lower_expr(exprs, args[0], pipe, base, name, memo, reuse=True)
    pipe.add(Node(Cut(at, numbers[0], length, name)))
    memo.remember(root, len(pipe.schema) - 1)
    return len(pipe.schema) - 1


def _lower_part(
    exprs: Expressions,
    root: Int,
    mut pipe: Pipeline,
    base: Int,
    name: String,
    mut memo: Memo,
) raises -> Int:
    """Appends whatever answers an `EXTRACT`.

    The specifier is read here rather than per row, for the reason the positions
    of a `SUBSTRING` are: `EXTRACT(year FROM d)` names the field in the text of
    the query and a column of field names is not a thing anybody writes, so the
    node carries a field code rather than an expression.

    A specifier DuckDB has and firepanda has no field for is refused by name.
    There are eight of those and the reason is different for each, which is why
    the message comes from the table rather than from here.

    Args:
        exprs: The arena.
        root: The call, already bound.
        pipe: The pipeline, added to.
        base: The width of the chunk before this node started lowering.
        name: What to call the column the answer lands in.
        memo: What this node has already computed and where it put it.

    Returns:
        The position of the column holding the answer.

    Raises:
        Error: If the call has the wrong number of arguments, or the specifier
            is not a name written in the query, or nothing is called that.
    """
    var args = exprs.nodes[root].children.copy()
    if len(args) != 2:
        raise Error(
            String(
                (
                    "lower: date_part reads a field name and a column, so two"
                    " arguments, and was given "
                ),
                len(args),
            )
        )
    if exprs.nodes[args[0]].kind != ExprKind.LITERAL:
        raise Error(
            "lower: the field an EXTRACT reads has to be written out, and this"
            " one is an expression, which would mean a different field for"
            " every row and there is no kernel that does that"
        )
    if exprs.nodes[args[0]].value.is_null():
        raise Error(
            "lower: an EXTRACT of a null field is null for every row, and there"
            " is no operator that answers a column of nulls yet"
        )

    var field = sql_field_named(exprs.nodes[args[0]].value.as_string())
    var at = _lower_expr(exprs, args[1], pipe, base, name, memo, reuse=True)
    pipe.add(Node(Part(at, field, name)))
    memo.remember(root, len(pipe.schema) - 1)
    return len(pipe.schema) - 1


def _lower_presence(
    exprs: Expressions,
    root: Int,
    mut pipe: Pipeline,
    base: Int,
    name: String,
    mut memo: Memo,
) raises -> Int:
    """Appends whatever answers an `IS NULL` or an `IS NOT NULL`.

    One line either way. The operand is lowered wherever it lands and a
    `Presence` reads the column it landed in, and which of the two tests it is
    is a flag on the node rather than two nodes, because the two kernels behind
    it differ by a flipped byte.

    Args:
        exprs: The arena.
        root: The call, already bound.
        pipe: The pipeline, added to.
        base: The width of the chunk before this node started lowering.
        name: What to call the column the answer lands in.
        memo: What this node has already computed and where it put it.

    Returns:
        The position of the column holding the answer.

    Raises:
        Error: If the call has the wrong number of arguments.
    """
    var args = exprs.nodes[root].children.copy()
    if len(args) != 1:
        raise Error(
            String(
                "lower: ",
                exprs.nodes[root].name,
                " reads one argument and was given ",
                len(args),
            )
        )

    var at = _lower_expr(exprs, args[0], pipe, base, name, memo, reuse=True)
    pipe.add(Node(Presence(at, exprs.nodes[root].name == "is_null", name)))
    memo.remember(root, len(pipe.schema) - 1)
    return len(pipe.schema) - 1


def _lower_coalesce(
    exprs: Expressions,
    root: Int,
    mut pipe: Pipeline,
    base: Int,
    name: String,
    mut memo: Memo,
) raises -> Int:
    """Appends whatever answers a `COALESCE`.

    A line of `Fill` nodes, each one reading what the one before it wrote, so
    three arguments are two nodes and one intermediate. A node that took the
    whole list would be one node and no intermediate, and it would not be
    better: the kernel fills from one column at a time either way, and the
    intermediates are dropped by the projection at the end the way every other
    expression's are.

    Every argument is converted to the type binding worked out for the call
    before it goes in, which is what `_lower_side` does for the two sides of a
    conditional and for the same reason. `COALESCE(a, 0)` over a float column is
    the ordinary case: the literal is the side that moves.

    A `COALESCE` of one argument is that argument. It is legal to write and
    DuckDB allows it, and there is nothing to fill from, so no node is added.

    Args:
        exprs: The arena.
        root: The call, already bound.
        pipe: The pipeline, added to.
        base: The width of the chunk before this node started lowering.
        name: What to call the column the answer lands in.
        memo: What this node has already computed and where it put it.

    Returns:
        The position of the column holding the answer.

    Raises:
        Error: If the call was given no arguments, or one of them has a kind no
            operator computes.
    """
    var args = exprs.nodes[root].children.copy()
    if len(args) == 0:
        raise Error(
            "lower: a coalesce answers the first of its arguments that is not"
            " null, and was given none to choose from"
        )

    var want = exprs.nodes[root].type
    var at = _lower_side(exprs, args[0], pipe, base, name, memo, want)
    for i in range(1, len(args)):
        var next = _lower_side(exprs, args[i], pipe, base, name, memo, want)
        pipe.add(Node(Fill(at, next, name)))
        at = len(pipe.schema) - 1
    memo.remember(root, at)
    return at


def _lower_conditional(
    exprs: Expressions,
    root: Int,
    mut pipe: Pipeline,
    base: Int,
    name: String,
    mut memo: Memo,
) raises -> Int:
    """Appends whatever computes a `CASE WHEN c THEN a ELSE b END`.

    The three children are lowered into columns and one `Choose` reads them.
    A chain of `WHEN`s arrives here already nested, each one's else side being
    the next, so a four branch expression is three of these and the recursion
    does the nesting without anything here counting branches.

    The two sides are made to agree on a type before the node is built. Binding
    worked out what the whole expression produces, by promoting the two, and a
    side that is not already that type gets a cast of its own rather than being
    converted where it lies, because it may be an input column that something
    else still reads at its own type. `CASE WHEN a > 0 THEN a ELSE 0.5 END` over
    an integer column is the ordinary case of this, and the integer side is the
    one that moves.

    Args:
        exprs: The arena.
        root: The conditional, already bound.
        pipe: The pipeline, added to.
        base: The width of the chunk before this node started lowering.
        name: What to call the column the whole expression lands in.
        memo: What this node has already computed and where it put it.

    Returns:
        The position of the column holding the answer.

    Raises:
        Error: If the conditional does not have three children, or one of them
            has a kind no operator computes.
    """
    var kids = exprs.nodes[root].children.copy()
    if len(kids) != 3:
        raise Error(
            String(
                (
                    "lower: a conditional reads a condition and two sides and"
                    " was given "
                ),
                len(kids),
                " children",
            )
        )

    var want = exprs.nodes[root].type
    var on = _lower_expr(exprs, kids[0], pipe, base, name, memo, reuse=True)
    var left = _lower_side(exprs, kids[1], pipe, base, name, memo, want)
    var right = _lower_side(exprs, kids[2], pipe, base, name, memo, want)
    pipe.add(Node(Choose(on, left, right, name)))
    memo.remember(root, len(pipe.schema) - 1)
    return len(pipe.schema) - 1


def _lower_side(
    exprs: Expressions,
    root: Int,
    mut pipe: Pipeline,
    base: Int,
    name: String,
    mut memo: Memo,
    want: LogicalType,
) raises -> Int:
    """Lowers one side of a conditional and converts it if it is not the type.

    Args:
        exprs: The arena.
        root: The side, already bound.
        pipe: The pipeline, added to.
        base: The width of the chunk before this node started lowering.
        name: What to call any column this appends.
        memo: What this node has already computed and where it put it.
        want: The type both sides have to agree on.

    Returns:
        The position of the column holding the side at that type.

    Raises:
        Error: If the side has a kind no operator computes.
    """
    if (
        exprs.nodes[root].kind == ExprKind.LITERAL
        and exprs.nodes[root].value.is_null()
    ):
        # A null literal has no type of its own, so it is written down at the
        # type the other side decided rather than being built as a column of
        # nulls and then converted into one. `CASE WHEN c THEN x END` is where
        # this comes from, since a CASE with no ELSE has a null there.
        pipe.add(Node(Constant(Value(null=want), want, name)))
        return len(pipe.schema) - 1
    var at = _lower_expr(exprs, root, pipe, base, name, memo, reuse=True)
    if pipe.schema[at].dtype == want:
        return at
    pipe.add(Node(Cast(at, want, name)))
    return len(pipe.schema) - 1


def _trim(mut pipe: Pipeline, base: Int) raises:
    """Drops every intermediate column a node's lowering appended.

    A node's output schema is what binding said it was, so anything lowering
    added to compute it has to be gone before the next node up is lowered,
    or every position above this one is off by the number of intermediates.

    Args:
        pipe: The pipeline, added to.
        base: The width to go back to.

    Raises:
        Error: Only what the projection raises.
    """
    if len(pipe.schema) == base:
        return
    var keep = List[Int](capacity=base)
    for i in range(base):
        keep.append(i)
    pipe.add(Node(Project(keep^)))


def _reaches(exprs: Expressions, root: Int, of: Int) -> Bool:
    """Whether an expression contains another one anywhere below it.

    Args:
        exprs: The arena.
        root: The expression to look in.
        of: The expression to look for.

    Returns:
        True if it is root or is under it.
    """
    if root == of:
        return True
    ref node = exprs.nodes[root]
    for i in range(len(node.children)):
        if _reaches(exprs, node.children[i], of):
            return True
    return False


def _lower_filter(plan: Plan, at: Int, mut pipe: Pipeline) raises:
    """Lowers a filter into one physical filter per conjunct.

    One filter per conjunct rather than one for the whole predicate, because
    each one narrows what the next one has to look at, and a predicate of four
    conditions where the first is selective is three conditions evaluated on
    almost nothing.

    Each filter is told which columns to write, which is what keeps that from
    being a bad trade. Filtering a column means writing a new one, and a mask
    that has just been used is all true and is read by nobody, so a filter that
    wrote every column would carry every spent mask through every later
    condition. What survives a conjunct is the node's input columns plus
    whatever a later conjunct still reads, and everything else is dropped in the
    same pass that does the filtering rather than by a projection afterwards.

    Args:
        plan: The plan.
        at: The filter node.
        pipe: The pipeline, added to.

    Raises:
        Error: If a conjunct has a kind no operator computes.
    """
    var base = len(pipe.schema)
    var memo = Memo()
    var parts = _conjuncts(plan.exprs, plan.nodes[at].exprs[0])
    for i in range(len(parts)):
        var mask = _lower_expr(plan.exprs, parts[i], pipe, base, "mask", memo)
        if len(pipe.schema) == base:
            # The predicate was a column of the input, so there is nothing
            # this conjunct left behind and nothing to drop.
            pipe.add(Node(Filter(mask)))
            continue
        var keep = List[Int](capacity=base)
        for j in range(base):
            keep.append(j)
        var moved = Memo()
        for j in range(len(memo.at)):
            if memo.at[j] < base:
                moved.remember(memo.of[j], memo.at[j])
                continue
            var live = False
            for k in range(i + 1, len(parts)):
                if _reaches(plan.exprs, parts[k], memo.of[j]):
                    live = True
                    break
            if not live:
                continue
            var now = -1
            for k in range(base, len(keep)):
                if keep[k] == memo.at[j]:
                    now = k
                    break
            if now < 0:
                now = len(keep)
                keep.append(memo.at[j])
            moved.remember(memo.of[j], now)
        pipe.add(Node(Filter(mask, keep^)))
        memo = moved^
    _trim(pipe, base)


def _lower_project(plan: Plan, at: Int, mut pipe: Pipeline) raises:
    """Lowers a projection into the computes it needs and one projection.

    The projection carries the output names as well as the positions. A name is
    free to carry, because a chunk is arrays and the names live on the
    pipeline's schema, and it is the only place they can come from: an
    expression that was computed was named by whatever computed it, an
    aggregate names its answer after itself, and an input column has the name
    the scan gave it. What the query calls each of them is written here.

    Args:
        plan: The plan.
        at: The projection node.
        pipe: The pipeline, added to.

    Raises:
        Error: If an output has a kind no operator computes.
    """
    var base = len(pipe.schema)
    var memo = Memo()
    var outputs = plan.nodes[at].exprs.copy()
    var names = plan.nodes[at].names.copy()
    var keep = List[Int](capacity=len(outputs))
    for i in range(len(outputs)):
        keep.append(
            _lower_expr(plan.exprs, outputs[i], pipe, base, names[i], memo)
        )
    pipe.add(Node(Project(keep^, names^)))


def _lower_aggregate(plan: Plan, at: Int, mut pipe: Pipeline) raises:
    """Lowers an aggregation into the computes it needs and one breaker.

    No keys is a whole frame reduction and goes to `Reduce`, and any keys go to
    `Group`. That is the one place lowering makes a physical choice, and it is
    the choice the plan node's own docstring says is physical: knowing there is
    one group rather than hashing every row to find out is not a difference in
    what the query means.

    No trim is needed afterwards. Both breakers produce the key columns and the
    aggregate columns and nothing else, so whatever intermediates the keys and
    the aggregated expressions needed are gone by the time a row comes out.

    A whole frame reduction over a column and a constant folds the operation
    into the reduction rather than putting a `Compute` in front of it. A query
    that sums the same column ninety times under ninety different constants is
    ninety `Compute` nodes otherwise, and every one of them holds a full column
    for as long as the chunk is alive, so the chunk costs ninety times what the
    column it reads costs. Folded, `Reduce` builds one of them at a time and
    drops it before it builds the next, and the chunk costs one column and
    change. Only a reduction folds this. A group by cannot, because its rows
    scatter and the operation would have to move with them, which is what the
    `Compute` in front of it already does.

    It is worth saying what this does not do, because the shortcut is sitting
    right there. The sum of a column plus a constant is the sum of the column
    plus the constant times the row count, so a clever enough lowering answers
    all ninety from one sum and never touches the data again. Do not add that.
    The query exists to measure whether an engine fuses an expression into a
    reduction, and an engine that answers it with algebra has measured nothing
    and has published a number it did not earn. The operation runs on every row.

    Args:
        plan: The plan.
        at: The aggregation node.
        pipe: The pipeline, added to.

    Raises:
        Error: If an aggregate is not a fold over something, if a key is
            renamed, which the breakers cannot do because they carry the input
            field through, or if an expression has a kind no operator computes.
    """
    var base = len(pipe.schema)
    var memo = Memo()
    var held = plan.nodes[at].exprs.copy()
    var names = plan.nodes[at].names.copy()
    var count = plan.nodes[at].parts

    var keys = List[Int](capacity=count)
    for i in range(count):
        var made = _lower_expr(plan.exprs, held[i], pipe, base, names[i], memo)
        if pipe.schema[made].name != names[i]:
            raise Error(
                String(
                    "lower: the aggregation calls the group key '",
                    pipe.schema[made].name,
                    "' by the name '",
                    names[i],
                    (
                        "', and a physical group by carries the key field"
                        " through as it found it"
                    ),
                )
            )
        keys.append(made)

    var aggs = List[GroupAgg](capacity=len(held) - count)
    for i in range(count, len(held)):
        if plan.exprs.nodes[held[i]].kind != ExprKind.AGGREGATE:
            raise Error(
                String(
                    "lower: the aggregation's output '",
                    names[i],
                    "' is a ",
                    plan.exprs.nodes[held[i]].kind,
                    (
                        " expression rather than a fold, and there is no"
                        " operator that computes one beside the groups"
                    ),
                )
            )
        var over = plan.exprs.nodes[held[i]].children[0]
        var kind = AggKind(UInt8(plan.exprs.nodes[held[i]].op))

        if count == 0 and plan.exprs.nodes[over].kind == ExprKind.BINARY:
            var op = BinaryOp(UInt8(plan.exprs.nodes[over].op))
            var left = plan.exprs.nodes[over].children[0]
            var right = plan.exprs.nodes[over].children[1]
            var left_is_value = plan.exprs.nodes[left].kind == ExprKind.LITERAL
            var right_is_value = (
                plan.exprs.nodes[right].kind == ExprKind.LITERAL
            )
            # Both constant is an expression simplify refused to fold, and
            # neither constant is a second column, and neither of those is the
            # shape this folds. They go the long way and get the error or the
            # `Compute` pair they would have got before.
            if left_is_value != right_is_value:
                var side = right if left_is_value else left
                var value = left if left_is_value else right
                var at = _lower_expr(
                    plan.exprs, side, pipe, base, names[i], memo, reuse=True
                )
                aggs.append(
                    GroupAgg(
                        at,
                        kind,
                        names[i],
                        op,
                        Value(copy=plan.exprs.nodes[value].value),
                        value_on_left=left_is_value,
                    )
                )
                continue

        # The fold names its own output, so unlike a projection this one does
        # not own the name of the column it reads, and two folds over the same
        # expression can read the same column.
        var made = _lower_expr(
            plan.exprs, over, pipe, base, names[i], memo, reuse=True
        )
        aggs.append(GroupAgg(made, kind, names[i]))

    if count == 0:
        pipe.add(Node(Reduce(aggs^)))
        return
    pipe.add(Node(Group(keys^, aggs^)))


def _decide(plan: Plan, at: Int, mut pipe: Pipeline) raises:
    """Turns the tagged stack under a difference or an intersection into its rows.

    `_stacked` has put both arms one after the other with a column saying which
    arm each row came from, zero for the left and one for the right. A group by
    over the query's own columns then puts every copy of a row in one group
    whichever arm it came from, and the smallest and the largest tag in a group
    say which arms had it. Both arms is a smallest of zero and a largest of one,
    the left alone is zero and zero, and the right alone is one and one.

    So an intersection keeps the groups whose two tags differ, which is one
    comparison and needs both folds. A difference keeps the groups whose largest
    tag is still the left's, which needs only the largest, and it is a
    comparison against a constant, so the constant becomes a column first. The
    mask and the tags are dropped by the filter, which writes only the positions
    asked for, so what comes out of here is the row the query wrote and nothing
    else.

    Grouping is the reason this shape is the right one rather than a join. SQL
    compares two rows of a set operation with a null equal to a null, and a join
    key is never equal to a null, so a difference written as an anti join drops
    every row with a null in it and does it silently. A group by puts the nulls
    of a column in one group, which is the rule SQL asked for.

    `ALL` is the same group by counting instead of comparing. `EXCEPT ALL`
    subtracts one copy of a row per copy on the right and `INTERSECT ALL` keeps
    as many copies as the thinner arm has, so both want the number of copies
    each arm put in the group rather than which arms were in it. Adding the tag
    up is the right arm's count, since the tag is one there and zero on the
    left, and counting the rows is both arms together, so the left's count is
    the one subtracted from the other. A difference then asks for the left's
    count less the right's, an intersection asks for the smaller of the two, and
    `Expand` turns whichever number that is into that many rows.

    A difference of counts goes negative whenever the right arm has more copies,
    and that is left alone rather than clamped here, because `Expand` writes
    nothing for a count of zero or less. It is the one place in this file where
    an operator's rule about its own edge saves a node.

    Args:
        plan: The plan, bound.
        at: The union node, which is a difference or an intersection.
        pipe: The pipeline over the tagged stack, added to.

    Raises:
        Error: If the node is not a difference or an intersection.
    """
    var op = plan.nodes[at].op

    # One short of the width, because the last column is the tag `_stacked` put
    # there and the query's own row is everything in front of it.
    var width = len(pipe.schema) - 1
    var keys = List[Int](capacity=width)
    for i in range(width):
        keys.append(i)

    var keep = List[Int](capacity=width)
    for i in range(width):
        keep.append(i)

    if plan.nodes[at].flags[0]:
        var counts = List[GroupAgg]()
        counts.append(GroupAgg(width, AggKind.SUM, "__right"))
        counts.append(GroupAgg(width, AggKind.COUNT, "__rows"))
        pipe.add(Node(Group(keys^, counts^)))
        # Both arms together less the right arm's, which is the left arm's.
        pipe.add(Node(Compute(width + 1, width, BinaryOp.SUB, "__left_rows")))
        var copies: Int
        if op == SET_INTERSECT:
            pipe.add(Node(Compute(width + 2, width, BinaryOp.LT, "__thinner")))
            pipe.add(Node(Choose(width + 3, width + 2, width, "__copies")))
            copies = width + 4
        else:
            pipe.add(Node(Compute(width + 2, width, BinaryOp.SUB, "__copies")))
            copies = width + 3
        pipe.add(Node(Expand(copies, keep^)))
        return

    # The group by puts its keys in front and its folds after them, so the tags
    # are the positions past the query's own row.
    var aggs = List[GroupAgg]()
    var mask: Int
    if op == SET_INTERSECT:
        aggs.append(GroupAgg(width, AggKind.MIN, "__first"))
        aggs.append(GroupAgg(width, AggKind.MAX, "__last"))
        pipe.add(Node(Group(keys^, aggs^)))
        pipe.add(Node(Compute(width, width + 1, BinaryOp.NE, "__both")))
        mask = width + 2
    else:
        # Only the largest, because a group the right arm was in has a one in
        # it and a group it was not in does not, whatever the left arm did. The
        # comparison is against a constant and an operation takes two columns,
        # so the constant is a column of its own first.
        aggs.append(GroupAgg(width, AggKind.MAX, "__last"))
        pipe.add(Node(Group(keys^, aggs^)))
        pipe.add(Node(Constant(Value(Int8(0)), LogicalType.INT8, "__left")))
        pipe.add(Node(Compute(width, width + 1, BinaryOp.EQ, "__only")))
        mask = width + 2

    pipe.add(Node(Filter(mask, keep^)))


def _lower_distinct(plan: Plan, at: Int, mut pipe: Pipeline) raises:
    """Lowers a distinct into a group by, or into `Unique` when it has keys.

    A group by holds one row per group, and which row that is does not matter
    when the key is every column, because every row of a group is the same row.
    So `SELECT DISTINCT` is `Group` with every position as a key and nothing to
    reduce, and there is no operator to write. The rows come out in the order
    the first of each was seen, which is what DuckDB gives back for a distinct
    with no ordering asked for.

    A key list shorter than the row, or one in another order, is `DISTINCT ON`,
    and that goes to `Unique`. It keeps whole rows chosen by part of them, so
    the columns that are not keys have to come back untouched and in place, and
    a group by cannot do that: it carries its keys in front, which moves the
    columns the plan's schema numbered, and its first skips nulls. Both
    distincts are breakers, but the group by holds one row per group and
    `Unique` holds every row, so the cheaper one is worth keeping for the case
    that can use it.

    Args:
        plan: The plan.
        at: The distinct node.
        pipe: The pipeline, added to.

    Raises:
        Error: If a key is a computed expression rather than a column.
    """
    var held = plan.nodes[at].exprs.copy()
    var width = len(pipe.schema)
    var memo = Memo()
    var keys = List[Int](capacity=width)
    if len(held) == 0:
        for i in range(width):
            keys.append(i)
        pipe.add(Node(Group(keys^, List[GroupAgg]())))
        return
    for i in range(len(held)):
        if plan.exprs.nodes[held[i]].kind != ExprKind.COLUMN:
            raise Error(
                String(
                    "lower: this distinct decides on a ",
                    plan.exprs.nodes[held[i]].kind,
                    (
                        " expression, and a computed key is a column the row"
                        " does not have, so the rows kept would not be the rows"
                        " the plan said"
                    ),
                )
            )
        keys.append(_lower_expr(plan.exprs, held[i], pipe, width, "", memo))
    var whole = len(keys) == width
    if whole:
        for i in range(width):
            if keys[i] != i:
                whole = False
                break
    if whole:
        pipe.add(Node(Group(keys^, List[GroupAgg]())))
        return
    pipe.add(Node(Unique(keys^)))


def _lower_sort(plan: Plan, at: Int, mut pipe: Pipeline) raises:
    """Lowers a sort into the computes its keys need and one breaker.

    A key is an expression rather than a column, so `ORDER BY a + b` appends the
    sum and sorts on that. The appended columns are trimmed afterwards, which a
    projection above the breaker does, because the sort hands its input schema
    back unchanged and a position below it is the same position above it.

    Two keys that are the same expression share one column. Unlike a projection
    a sort key owns no name, it is read and not emitted, so there is nothing to
    stop the second from being the first.

    The bound the plan may have put on the node is not read. A sort that only
    has to get the first n rows right is a different operator, and the limit
    that wrote the bound down is still above the sort and still doing the
    cutting, so ignoring it is slow rather than wrong.

    Args:
        plan: The plan.
        at: The sort node.
        pipe: The pipeline, added to.

    Raises:
        Error: If a key has a kind no operator computes.
    """
    var base = len(pipe.schema)
    var memo = Memo()
    var held = plan.nodes[at].exprs.copy()
    var keys = List[Int](capacity=len(held))
    for i in range(len(held)):
        keys.append(
            _lower_expr(
                plan.exprs, held[i], pipe, base, "key", memo, reuse=True
            )
        )

    # The plan writes the directions down first and the null placements after
    # them, in one list, and it says where the nulls go rather than where they
    # come first, which is the other way round from the kernel.
    var descending = List[Bool](capacity=len(held))
    var nulls_first = List[Bool](capacity=len(held))
    for i in range(len(held)):
        descending.append(plan.nodes[at].flags[i])
        nulls_first.append(not plan.nodes[at].flags[len(held) + i])

    pipe.add(Node(Sort(keys^, descending^, nulls_first^)))
    _trim(pipe, base)


def _lower_window(plan: Plan, at: Int, mut pipe: Pipeline) raises:
    """Lowers a window into the computes it needs and one breaker.

    Every window in the node has to partition the same way, because one operator
    does one grouping pass. The first window says what the partitioning is and
    the rest have to agree with it, by expression rather than by shape, which
    they do when they were written the same way, since the arena hands the same
    index back to a binder that resolved the same name twice.

    The trim afterwards is a selection rather than a prefix. The keys and the
    aggregated expressions were appended before the breaker and the windows land
    after them, so what is kept is the input's columns and then the last few,
    with the intermediates cut from between.

    Args:
        plan: The plan.
        at: The window node.
        pipe: The pipeline, added to.

    Raises:
        Error: If a window is ordered, if two windows partition differently, or
            if an expression has a kind no operator computes.
    """
    var base = len(pipe.schema)
    var memo = Memo()
    var held = plan.nodes[at].exprs.copy()
    var names = plan.nodes[at].names.copy()

    var keys = List[Int]()
    var sources = List[Int](capacity=len(held))
    var kinds = List[AggKind](capacity=len(held))
    for i in range(len(held)):
        ref node = plan.exprs.nodes[held[i]]
        if node.kind != ExprKind.WINDOW:
            raise Error(
                String(
                    "lower: the window node's output '",
                    names[i],
                    "' is a ",
                    node.kind,
                    (
                        " expression rather than a window, and there is no"
                        " operator that computes one beside the partitions"
                    ),
                )
            )
        if len(node.children) - 1 > node.parts:
            raise Error(
                String(
                    "lower: the window '",
                    names[i],
                    (
                        "' is ordered, and an ordered window is a running fold"
                        " over the partition rather than one value broadcast"
                        " across it, so there is no operator for it yet"
                    ),
                )
            )
        # The first window decides the partitioning and the rest have to be the
        # same partitioning, since one operator makes one set of ordinals.
        if i == 0:
            for k in range(node.parts):
                keys.append(
                    _lower_expr(
                        plan.exprs,
                        node.children[1 + k],
                        pipe,
                        base,
                        "key",
                        memo,
                        reuse=True,
                    )
                )
        else:
            var same = node.parts == len(keys)
            if same:
                for k in range(node.parts):
                    if (
                        _lower_expr(
                            plan.exprs,
                            node.children[1 + k],
                            pipe,
                            base,
                            "key",
                            memo,
                            reuse=True,
                        )
                        != keys[k]
                    ):
                        same = False
            if not same:
                raise Error(
                    String(
                        "lower: the window '",
                        names[i],
                        (
                            "' partitions differently from the first one in the"
                            " same node, and one window operator has one"
                            " partitioning, so this wants a node each"
                        ),
                    )
                )
        # The operator names its own output, so the column it reads is an
        # intermediate with a name of no consequence, and two windows over the
        # same expression can read the same one. Naming it after the output
        # would put two columns of that name in the chunk at once, since this
        # breaker's output holds what it was given as well as what it computed.
        sources.append(
            _lower_expr(
                plan.exprs,
                node.children[0],
                pipe,
                base,
                "over",
                memo,
                reuse=True,
            )
        )
        kinds.append(AggKind(UInt8(node.op)))

    var made = len(pipe.schema) - base
    pipe.add(Node(Window(keys^, sources^, kinds^, names^)))
    if made == 0:
        return
    var keep = List[Int](capacity=base + len(held))
    for i in range(base):
        keep.append(i)
    for i in range(len(held)):
        keep.append(base + made + i)
    pipe.add(Node(Project(keep^)))


def _lower_limit(plan: Plan, at: Int, mut pipe: Pipeline) raises:
    """Lowers a limit, and the offset it may start at.

    A limit that keeps every row and starts at the first one is nothing at all,
    and lowers to no operator rather than to one that passes everything
    through. That is not only a saving: a limit is the one operator that can
    stop the pipeline early, and the driver feeds a pipeline that has one in it
    a chunk at a time, so an operator that never stops anything would cost the
    query its read ahead for nothing.

    Args:
        plan: The plan.
        at: The limit node.
        pipe: The pipeline, added to.
    """
    if plan.nodes[at].length == NO_LIMIT and plan.nodes[at].offset == 0:
        return
    pipe.add(Node(Limit(plan.nodes[at].length, plan.nodes[at].offset)))


def _take(
    mut frames: List[DataFrame],
    mut taken: List[Bool],
    plan: Plan,
    at: Int,
) raises -> DataFrame:
    """Takes the frame a scan reads, leaving the relation numbering alone.

    An empty frame goes back where the taken one was rather than the list
    closing up, because the numbers are relation ids and a list that closes up
    renumbers every relation above the one that went. With one scan that was
    invisible. With a join there are two, and the second would read the wrong
    frame.

    Which ones have gone is tracked beside the list rather than read off it,
    because the thing left behind is an empty frame and an empty frame is a
    thing a caller may pass. Two scans of one relation is a plan that says two
    tables and was given one, and saying so here is better than a missing column
    further down.

    The scan's own column list is applied here rather than by a projection
    afterwards, because a build side is a frame and not a stream, so narrowing
    it is selecting columns rather than adding an operator.

    Args:
        frames: One frame per relation, taken from.
        taken: Which relations have already gone, written through.
        plan: The plan.
        at: The scan node.

    Returns:
        The frame, narrowed to the columns the scan reads.

    Raises:
        Error: If the scan names a relation there is no frame for, one that has
            already been read, or a column the frame does not have.
    """
    var table = plan.nodes[at].table
    if table < 0 or table >= len(frames):
        raise Error(
            String(
                "lower: the scan of ",
                plan.nodes[at].source,
                " reads relation ",
                table,
                " and it was given ",
                len(frames),
                " frames",
            )
        )
    if taken[table]:
        raise Error(
            String(
                "lower: two scans of this plan both read relation ",
                table,
                (
                    ", and a relation is one frame, so a table joined to itself"
                    " is two relations and two frames rather than one of each"
                ),
            )
        )
    taken[table] = True
    var out = frames.pop(table)
    frames.insert(table, DataFrame())
    var names = plan.nodes[at].names.copy()
    if len(names) == 0:
        return out^
    try:
        return out.select(names)
    except e:
        raise Error(
            String(
                "lower: the scan of ",
                plan.nodes[at].source,
                " reads a column the frame it was given does not have: ",
                e,
            )
        )


def _column(values: List[Value], type: LogicalType) raises -> AnyArray:
    """Builds a column out of the values a VALUES node wrote down.

    This is the one place in lowering that makes an array rather than arranging
    for one to be made, and it is why a literal table costs nothing at run time:
    the rows are known before the query starts, so they are a frame before the
    query starts.

    Args:
        values: One value per row, in row order.
        type: The type binding gave the column.

    Returns:
        The column.

    Raises:
        Error: If the type is one there is no way to write a value of down,
            which is the nested ones and a dictionary.
    """
    if type == LogicalType.STRING:
        var text = StringBuilder(len(values))
        for i in range(len(values)):
            if values[i].is_null():
                text.append_null()
            else:
                text.append(values[i].as_string().as_bytes())
        return AnyArray(text^.finish())

    if type.is_nested() or type.is_dictionary() or type.is_variable_width():
        raise Error(
            String(
                "lower: a literal table of ",
                type,
                (
                    " has no column to write it into, so this VALUES cannot be"
                    " a frame"
                ),
            )
        )

    comptime for candidate in ALL:
        if type.physical == candidate:
            var col = Array[candidate](len(values))
            for i in range(len(values)):
                if values[i].is_null():
                    col.set_null(i)
                else:
                    col.set_valid(i, values[i].as_scalar[candidate]())
            var out = AnyArray(col^)
            # A timestamp is an int64 count of units, so the array is the right
            # array and only its label is wrong. Writing the bound type over it
            # is what keeps `VALUES (DATE '2020-01-01')` a date.
            out.type = type
            return out^
    raise Error(
        String(
            "lower: a literal table of ",
            type,
            " has no array behind it, so this VALUES cannot be a frame",
        )
    )


def _literals(plan: Plan, at: Int) raises -> DataFrame:
    """Turns a VALUES node into the frame a pipeline reads.

    A VALUES is the one source that names no table. Its rows were written in
    the query, so the frame is built here rather than found, and the pipeline
    starts from it the same way it starts from a scan's frame.

    Every value has to be a literal by the time this runs. A `VALUES (1 + 1)`
    is one after constant folding, and one that is not folded is refused by
    name rather than computed, because there is no chunk to compute it over and
    an operator that made a column out of nothing is a bigger change than this.

    Args:
        plan: The plan, bound.
        at: The values node.

    Returns:
        The frame the node stands for.

    Raises:
        Error: If a value is not a literal, if a column's rows do not agree on
            a type, or if the type is one no column holds.
    """
    var width = plan.nodes[at].parts
    var names = plan.nodes[at].names.copy()
    var held = plan.nodes[at].exprs.copy()
    var rows = len(held) // width

    var columns = List[ChunkedArray](capacity=width)
    var fields = List[Field](capacity=width)
    for c in range(width):
        var values = List[Value](capacity=rows)
        var type = plan.exprs.nodes[held[c]].type
        for r in range(rows):
            var e = held[r * width + c]
            if plan.exprs.nodes[e].kind != ExprKind.LITERAL:
                raise Error(
                    String(
                        "lower: row ",
                        r + 1,
                        " of this literal table computes '",
                        names[c],
                        "' with a ",
                        plan.exprs.nodes[e].kind,
                        (
                            " expression, and there is no chunk under a VALUES"
                            " to compute it over"
                        ),
                    )
                )
            if plan.exprs.nodes[e].type != type:
                raise Error(
                    String(
                        "lower: the column '",
                        names[c],
                        "' of this literal table is a ",
                        type,
                        " on its first row and a ",
                        plan.exprs.nodes[e].type,
                        " on row ",
                        r + 1,
                        ", and a column holds one type",
                    )
                )
            values.append(plan.exprs.nodes[e].value.copy())
        var chunk = ChunkedArray(type)
        chunk.append(_column(values, type))
        columns.append(chunk^)
        fields.append(Field(names[c], type))
    return DataFrame(Schema(fields^), columns^)


def _series(plan: Plan, at: Int) raises -> DataFrame:
    """Turns a `range` or a `generate_series` into the frame a pipeline reads.

    The two functions differ by one row. `range` stops before the bound it was
    given and `generate_series` stops on it, which is why `range(3)` is three
    rows and `generate_series(3)` is four. Everything else about them is the
    same, including what one argument means: the bound is the last argument, so
    a single argument is the bound and the series starts at zero and counts by
    one.

    A null argument answers no rows rather than raising, which is what DuckDB
    does and is the only sensible reading of a series whose end nobody knows.
    A step of zero is refused instead, because that one is not a series with
    nothing in it, it is a series that never ends.

    Args:
        plan: The plan, bound.
        at: The table function node.

    Returns:
        The frame the node stands for.

    Raises:
        Error: If an argument is not a literal, if the step is zero, or if the
            series is longer than `LONGEST_SERIES`.
    """
    var name = plan.nodes[at].source.copy()
    var inclusive = name == "generate_series"

    var bounds = List[Int64](capacity=3)
    var unknown = False
    for i in range(len(plan.nodes[at].exprs)):
        ref arg = plan.exprs.nodes[plan.nodes[at].exprs[i]]
        if arg.kind != ExprKind.LITERAL:
            raise Error(
                String(
                    "lower: argument ",
                    i + 1,
                    " of ",
                    name,
                    " is a ",
                    arg.kind,
                    (
                        " expression, and there is no chunk under a table"
                        " function to compute it over"
                    ),
                )
            )
        if arg.value.is_null():
            unknown = True
        else:
            bounds.append(arg.value.as_scalar[DType.int64]())

    var values = List[Int64]()
    if not unknown:
        var start = Int64(0)
        var step = Int64(1)
        var stop = bounds[0]
        if len(bounds) > 1:
            start = bounds[0]
            stop = bounds[1]
        if len(bounds) > 2:
            step = bounds[2]
        if step == 0:
            raise Error(
                String(
                    "lower: ",
                    name,
                    (
                        " was given a step of zero, and a series that never"
                        " moves never reaches its end"
                    ),
                )
            )
        # How many rows there are is arithmetic rather than something to find
        # out by counting them, and working it out first is what makes a series
        # that is too long to build refused in no time rather than refused once
        # it has been half built. The span is worked out unsigned so that a
        # start below zero and a stop above it is a width and not an overflow.
        var rows = UInt64(0)
        if step > 0 and stop > start:
            var span = UInt64(stop) - UInt64(start)
            var by = UInt64(step)
            rows = span // by
            if inclusive or span % by != 0:
                rows += 1
        elif step < 0 and stop < start:
            var span = UInt64(start) - UInt64(stop)
            var by = UInt64(0) - UInt64(step)
            rows = span // by
            if inclusive or span % by != 0:
                rows += 1
        elif stop == start and inclusive:
            rows = 1
        if rows > LONGEST_SERIES:
            raise Error(
                String(
                    "lower: this ",
                    name,
                    " is ",
                    rows,
                    " rows, which is longer than ",
                    LONGEST_SERIES,
                    ", and the whole of it is built before the query starts",
                )
            )
        values = List[Int64](capacity=Int(rows))
        var cur = start
        for _ in range(Int(rows)):
            values.append(cur)
            cur += step

    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    var chunk = ChunkedArray(LogicalType.INT64)
    chunk.append(AnyArray(col^))
    var columns = List[ChunkedArray](capacity=1)
    columns.append(chunk^)
    var fields = List[Field](capacity=1)
    fields.append(Field(plan.nodes[at].names[0], LogicalType.INT64))
    return DataFrame(Schema(fields^), columns^)


def _stacked(
    plan: Plan,
    at: Int,
    mut frames: List[DataFrame],
    mut taken: List[Bool],
) raises -> DataFrame:
    """Turns a UNION node into the frame a pipeline reads.

    Every input is a line of the same plan, so each one is lowered and run the
    way a join's build side is, and what comes back is stacked by position. By
    position rather than by name, because that is what SQL says a union means:
    the answer takes the first input's names and the rest line up under them
    whatever they call themselves.

    Stacking is moving chunks rather than copying rows. A frame is already a
    list of chunks per column, and a union of two frames is those lists one
    after the other, so nothing is read and nothing is allocated beyond the
    lists themselves.

    Running the inputs here rather than streaming them is the same trade the
    join's build side makes, and it is worse here in one way: a union streams
    perfectly well in principle, since a row of the second input needs nothing
    from the first. What stops it is that a pipeline reads one source, and
    giving it two is a change to the driver rather than to this file. Until
    then, everything is in memory at once and it says so here.

    A difference and an intersection stack too, and the stack carries one more
    column saying which side each row came from. What decides the answer is the
    group by over the row that `_lower_from` puts above this, which is why the
    tag is appended here and read there. `_SIDE` is what it is called, and the
    name is one no column of a query has, since a query cannot write it.

    The tag is a byte for the distinct answer, which only ever asks whether a
    side was in a group, and a whole number of sixty four bits for `ALL`, which
    adds the tag up to count how many rows a side put in one.

    Args:
        plan: The plan, bound.
        at: The union node.
        frames: One frame per relation, taken from.
        taken: Which relations have already gone, written through.

    Returns:
        The frame the node stands for, with the side tag on it for a difference
        and an intersection.

    Raises:
        Error: If two inputs disagree on how many columns they have or on a
            column's type, or if an input is a line this file cannot lower.
    """
    var tagged = plan.nodes[at].op != SET_UNION
    var counted = tagged and plan.nodes[at].flags[0]
    var inputs = plan.nodes[at].inputs.copy()
    var fields = List[Field]()
    var columns = List[ChunkedArray]()
    for i in range(len(inputs)):
        var side = _lower_from(plan, inputs[i], frames, taken)
        if tagged:
            if counted:
                # A whole number wide enough to be added up, because `ALL` sums
                # the tag to count the rows one side put in a group and a sum of
                # a byte would run out on a group of more than a hundred and
                # twenty seven rows.
                side.add(
                    Node(Constant(Value(Int64(i)), LogicalType.INT64, _SIDE))
                )
            else:
                side.add(
                    Node(Constant(Value(Int8(i)), LogicalType.INT8, _SIDE))
                )
        var out = side^.run()
        if i == 0:
            for c in range(len(out.schema)):
                fields.append(out.schema.fields[c].copy())
            columns = out^.into_columns()
            continue
        if len(out.schema) != len(fields):
            raise Error(
                String(
                    "lower: input 1 of this union has ",
                    len(fields),
                    " columns and input ",
                    i + 1,
                    " has ",
                    len(out.schema),
                    ", and a stack lines its inputs up by position",
                )
            )
        for c in range(len(fields)):
            if out.schema.fields[c].dtype != fields[c].dtype:
                raise Error(
                    String(
                        "lower: column ",
                        c + 1,
                        " of this union is a ",
                        fields[c].dtype,
                        " on input 1 and a ",
                        out.schema.fields[c].dtype,
                        " on input ",
                        i + 1,
                        ", and a column holds one type",
                    )
                )
            if out.schema.fields[c].nullable and not fields[c].nullable:
                fields[c] = Field(fields[c].name, fields[c].dtype, True)
        var more = out^.into_columns()
        for c in range(len(fields) - 1, -1, -1):
            # Popped onto a second list and popped back off it, which is the
            # only way to hand the chunks over in their own order without
            # copying one. Moving out of the middle of a list is what is not
            # available, and a copy here is the whole cost of the union.
            var chunks = more.pop()^.into_chunks()
            var backwards = List[AnyArray](capacity=len(chunks))
            while len(chunks) > 0:
                backwards.append(chunks.pop())
            while len(backwards) > 0:
                columns[c].append(backwards.pop())
    return DataFrame(Schema(fields^), columns^)


def _lower_cross(
    plan: Plan,
    right: Int,
    mut frames: List[DataFrame],
    mut taken: List[Bool],
    mut pipe: Pipeline,
) raises:
    """Lowers a cross join whose right side is one row into constant columns.

    A cross join in general is every left row against every right row, which is
    a whole frame operation rather than anything a chunk at a time operator can
    do, and it stays refused. One row is the exception, and it is the case that
    matters, because that is what an uncorrelated subquery that answers one
    value becomes. Pairing every left row with one right row adds a column to
    each row and moves nothing, so it is a constant per right column and the
    left side streams past untouched.

    The right side is run here for the same reason a probe's build side is: it
    has to be a frame before the first left chunk is read, and a pipeline is
    what builds pipelines so an operator cannot hold the thing that holds it.

    Args:
        plan: The plan.
        right: The join's right input.
        frames: One frame per relation, taken from.
        taken: Which relations have already gone, written through.
        pipe: The pipeline, added to.

    Raises:
        Error: If the right side has any row count but one, or whatever the
            right side itself refuses.
    """
    var side: DataFrame
    if plan.nodes[right].kind == NodeKind.SCAN:
        side = _take(frames, taken, plan, right)
    else:
        var built = _lower_from(plan, right, frames, taken)
        side = built^.run()
    if len(side) != 1:
        raise Error(
            String(
                (
                    "lower: a cross join pairs every left row with every right"
                    " row, and this one has a right side of "
                ),
                len(side),
                (
                    " rows, which is the whole frame join rather than a column"
                    " added as each chunk goes past. One right row is the case"
                    " that lowers"
                ),
            )
        )
    var fields = Schema(copy=side.schema)
    var columns = side^.into_columns()
    for c in range(len(columns)):
        var one = value_at(columns[c].chunks[0], 0)
        pipe.add(
            Node(
                Constant(
                    one^,
                    fields.fields[c].dtype,
                    String(fields.fields[c].name),
                )
            )
        )


def _lower_join(
    plan: Plan,
    at: Int,
    mut frames: List[DataFrame],
    mut taken: List[Bool],
    mut pipe: Pipeline,
) raises:
    """Lowers a join into one probe operator, building its right side first.

    The physical join holds its build side whole and hashes it once, so the
    right input has to be a frame by the time the operator is made. A scan
    already is one. Anything else is a line of the same plan, so it is lowered
    to its own pipeline and run here, and what comes back is the frame.

    Running it here rather than when the outer pipeline starts is what the
    operator's shape asks for. It holds a frame, and it holds one because a
    pipeline is what builds pipelines and an operator cannot hold the thing
    that holds it. The work is the same work in either place and it is work
    that has to finish before the first chunk of the left side is read.

    A join on more than one key pair builds the table from the first pair and
    asks the rest afterwards, as a comparison and a filter per pair over the
    paired chunk. That is a real plan rather than a stopgap, it is what an
    engine does with any join condition its table cannot answer, and the cost of
    it is the pairs it makes and drops. It is inner joins only, for the reason
    written where it happens.

    Args:
        plan: The plan.
        at: The join node.
        frames: One frame per relation, taken from.
        taken: Which relations have already gone, written through.
        pipe: The pipeline, added to.

    Raises:
        Error: If the join is one the probe operator does not do, if a key is
            anything but a plain column, if a join on more than one key pair is
            anything but inner, or whatever the build side itself refuses.
    """
    var kind = JoinKind(UInt8(plan.nodes[at].op))
    if kind == JoinKind.RIGHT or kind == JoinKind.OUTER:
        raise Error(
            String(
                "lower: a ",
                kind,
                (
                    " join has to emit right rows that nothing matched, which"
                    " is not known until the last chunk has gone past, so it is"
                    " a breaker rather than an operator"
                ),
            )
        )
    var right = plan.nodes[at].inputs[1]
    if kind == JoinKind.CROSS:
        _lower_cross(plan, right, frames, taken, pipe)
        return

    var parts = plan.nodes[at].parts
    if parts < 1:
        raise Error(
            String(
                "lower: a ",
                kind,
                " join pairs rows a key agrees on and this one has no key",
            )
        )
    if parts > 1 and kind != JoinKind.INNER:
        raise Error(
            String(
                "lower: this operator builds its table from one column, so a ",
                kind,
                " join on ",
                parts,
                (
                    " key pairs would need the ordinal space that concatenating"
                    " both key columns builds, and concatenating both sides is"
                    " having them both. An inner join is the one kind that does"
                    " not need it, because it keeps both sides' columns and so"
                    " the rest of the key can be asked after the pairing"
                ),
            )
        )
    for i in range(parts):
        if (
            plan.exprs.nodes[plan.nodes[at].exprs[i]].kind != ExprKind.COLUMN
            or plan.exprs.nodes[plan.nodes[at].exprs[parts + i]].kind
            != ExprKind.COLUMN
        ):
            raise Error(
                "lower: this operator joins on a column on each side, and a"
                " computed key would have to be computed on the build side too,"
                " which is a projection over a frame rather than over a chunk"
            )
    var left_key = plan.nodes[at].exprs[0]
    var right_key = plan.nodes[at].exprs[parts]
    var left_on = plan.exprs.nodes[left_key].name.copy()
    var right_on = plan.exprs.nodes[right_key].name.copy()

    var build: DataFrame
    if plan.nodes[right].kind == NodeKind.SCAN:
        build = _take(frames, taken, plan, right)
    else:
        var side = _lower_from(plan, right, frames, taken)
        build = side^.run()
    # Left to itself the operator renames a right column whose name the left
    # already has and drops the right key outright when the two keys are called
    # the same, and either one moves a column the plan's schema numbered. So it
    # is told the numbers instead. The output is the two schemas end to end,
    # which is what the join binds to, and the keys are the positions binding
    # gave them rather than the first column with the name.
    var width = len(pipe.schema)
    var wanted = List[Int](capacity=width + len(build.schema))
    for i in range(width):
        wanted.append(i)
    if kind.keeps_right_columns():
        for i in range(len(build.schema)):
            wanted.append(width + i)
    var mark = String()
    if kind == JoinKind.MARK:
        mark = plan.nodes[at].names[0].copy()
    pipe.add(
        Node(
            Join(
                build^,
                left_on^,
                right_on^,
                kind,
                "_right",
                List[String](),
                wanted^,
                plan.exprs.nodes[left_key].at,
                plan.exprs.nodes[right_key].at,
                mark^,
            )
        )
    )
    if parts == 1:
        return
    # The first key pair built the table and the rest are asked afterwards, one
    # comparison and one filter each, over the paired chunk. Asking afterwards
    # is not the same plan as pairing on the whole key, it pairs on less and
    # throws the extra pairs away, but it is the same answer: a row survives
    # only if every pair agreed, and a pair a key is null on answers null and is
    # dropped, which is what a key that is null does anyway. Which of the pairs
    # builds the table is what it costs, and until something here counts rows
    # per value the first one is as good a guess as any other.
    #
    # An inner join is the only kind this works for. A left join has to emit the
    # rows that matched nothing and a filter after the pairing cannot tell those
    # from the rows it is dropping, and semi, anti and mark keep no right column
    # to compare against in the first place.
    var joined = len(pipe.schema)
    for i in range(1, parts):
        var here = plan.exprs.nodes[plan.nodes[at].exprs[i]].at
        var there = width + plan.exprs.nodes[plan.nodes[at].exprs[parts + i]].at
        pipe.add(Node(Compute(here, there, BinaryOp.EQ, "mask")))
        var keep = List[Int](capacity=joined)
        for c in range(joined):
            keep.append(c)
        pipe.add(Node(Filter(joined, keep^)))


def lower(
    plan: Plan, root: Int, var frames: List[DataFrame]
) raises -> Pipeline:
    """Turns a bound plan into a pipeline that produces its answer.

    The plan has to be bound, because every column reference is lowered to the
    position binding gave it and there is nothing here that resolves a name.

    Args:
        plan: The plan, read only. Lowering decides nothing a pass should have
            decided, so it has no reason to write to it.
        root: The node whose output is the answer.
        frames: One frame per relation, indexed by the id a scan carries, the
            same way `bind` is given one schema per relation. A join reads two
            of them. Consumed.

    Returns:
        A pipeline whose `run` produces what the plan says.

    Raises:
        Error: If the plan is not a line from a scan to the root, if a node or
            an expression has a kind no operator computes yet, or if a scan
            names a relation there is no frame for. The error names what could
            not be lowered so that a caller can decide what to do instead.
    """
    var taken = List[Bool](length=len(frames), fill=False)
    return _lower_from(plan, root, frames, taken)


def _lower_from(
    plan: Plan,
    root: Int,
    mut frames: List[DataFrame],
    mut taken: List[Bool],
) raises -> Pipeline:
    """Builds the pipeline for one line of the plan.

    `lower` is this with the bookkeeping set up. It is a separate function
    because a join calls it for its build side, and a build side is a line of
    the same plan reading out of the same frames, so it has to see which
    relations have already gone and leave its own behind it.

    Args:
        plan: The plan.
        root: The node whose output this line produces.
        frames: One frame per relation, taken from.
        taken: Which relations have already gone, written through.

    Returns:
        A pipeline whose `run` produces what that line says.

    Raises:
        Error: Whatever the line refuses. See `lower`.
    """
    var order = _spine(plan, root)
    var first = order[0]
    var source: DataFrame
    if plan.nodes[first].kind == NodeKind.VALUES:
        source = _literals(plan, first)
    elif plan.nodes[first].kind == NodeKind.TABLE_FUNCTION:
        source = _series(plan, first)
    elif plan.nodes[first].kind == NodeKind.UNION:
        source = _stacked(plan, first, frames, taken)
    else:
        source = _take(frames, taken, plan, first)
    var pipe = Pipeline(source^)

    if plan.nodes[first].kind == NodeKind.UNION:
        if plan.nodes[first].op == SET_UNION:
            if not plan.nodes[first].flags[0]:
                # A `UNION` without `ALL` is the stack with the duplicates
                # dropped, and dropping them is the distinct above, which is the
                # same group by with nothing to reduce that `SELECT DISTINCT`
                # lowers to.
                var keys = List[Int](capacity=len(pipe.schema))
                for i in range(len(pipe.schema)):
                    keys.append(i)
                pipe.add(Node(Group(keys^, List[GroupAgg]())))
        else:
            _decide(plan, first, pipe)

    for i in range(1, len(order)):
        var at = order[i]
        var kind = plan.nodes[at].kind
        if kind == NodeKind.JOIN:
            _lower_join(plan, at, frames, taken, pipe)
        elif kind == NodeKind.AGGREGATE:
            _lower_aggregate(plan, at, pipe)
        elif kind == NodeKind.FILTER:
            _lower_filter(plan, at, pipe)
        elif kind == NodeKind.SORT:
            _lower_sort(plan, at, pipe)
        elif kind == NodeKind.WINDOW:
            _lower_window(plan, at, pipe)
        elif kind == NodeKind.DISTINCT:
            _lower_distinct(plan, at, pipe)
        elif kind == NodeKind.PROJECT:
            _lower_project(plan, at, pipe)
        elif kind == NodeKind.LIMIT:
            _lower_limit(plan, at, pipe)
        else:
            raise Error(
                String(
                    "lower: there is no operator for a ",
                    kind,
                    " node yet, so this plan has to run the way it did before",
                )
            )
    return pipe^
