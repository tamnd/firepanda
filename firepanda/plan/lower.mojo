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

### A distinct is a group by that reduces nothing

A group by holds one row per group and the rows of a group differ only in what
was not the key. When the key is every column there is nothing they can differ
in, so `SELECT DISTINCT` is the group by with every position as a key and no
folds at all, and there is no operator to write. It streams for the same reason
the group by does, holding one row per distinct row rather than the input.

Part of the row is `DISTINCT ON` and is refused. It keeps whole rows chosen by
some of their columns, so the rest would come back through a first over each,
and a group by carries its keys in front of what it reduced, which moves the
columns the plan's schema numbered.

### What it refuses, and why refusing is the design

A distinct on part of the row is not lowered here, and neither is a union, a
unary expression, a conditional, a window, or a cast over an input column.
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
    Cast,
    Compute,
    Filter,
    Group,
    GroupAgg,
    Join,
    Limit,
    Node,
    Project,
    Reduce,
    Sort,
)
from firepanda.exec.pipeline import Pipeline
from firepanda.frame.frame import DataFrame
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.plan.expr import UNBOUND, ExprKind, Expressions
from firepanda.plan.node import NO_LIMIT, NodeKind, Plan


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
        # A constant reaches an operator as an operand of a `Compute` and never
        # on its own, because no node makes a column out of thin air.
        raise Error(
            String(
                "lower: the constant ",
                exprs.nodes[root].value,
                (
                    " is not an operand of anything, and there is no operator"
                    " that makes a column out of a constant"
                ),
            )
        )

    if kind == ExprKind.CAST:
        var over = exprs.nodes[root].children[0]
        var at = _lower_expr(exprs, over, pipe, base, name, memo, reuse=True)
        if at < base:
            raise Error(
                String(
                    "lower: a cast of the input column at position ",
                    at,
                    (
                        " would convert it where it lies and change what that"
                        " position means for every expression already bound"
                        " against it"
                    ),
                )
            )
        pipe.add(Node(Cast(at, exprs.nodes[root].type)))
        # The position holds the converted column now, so whatever the memo
        # says is there is no longer there, and the cast itself is not recorded
        # because a second cast of it would convert it twice.
        memo.forget(at)
        return at

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
        # The fold names its own output, so unlike a projection this one does
        # not own the name of the column it reads, and two folds over the same
        # expression can read the same column.
        var made = _lower_expr(
            plan.exprs, over, pipe, base, names[i], memo, reuse=True
        )
        var kind = AggKind(UInt8(plan.exprs.nodes[held[i]].op))
        aggs.append(GroupAgg(made, kind, names[i]))

    if count == 0:
        pipe.add(Node(Reduce(aggs^)))
        return
    pipe.add(Node(Group(keys^, aggs^)))


def _lower_distinct(plan: Plan, at: Int, mut pipe: Pipeline) raises:
    """Lowers a distinct over the whole row into a group by with no folds.

    A group by holds one row per group, and which row that is does not matter
    when the key is every column, because every row of a group is the same row.
    So `SELECT DISTINCT` is `Group` with every position as a key and nothing to
    reduce, and there is no operator to write. The rows come out in the order
    the first of each was seen, which is what DuckDB gives back for a distinct
    with no ordering asked for.

    A key list shorter than the row is `DISTINCT ON`, and it is refused. It
    keeps whole rows chosen by part of them, so the columns that are not keys
    would have to come back through a first over each, and a group by puts its
    keys in front, which moves the columns the plan's schema numbered.

    Args:
        plan: The plan.
        at: The distinct node.
        pipe: The pipeline, added to.

    Raises:
        Error: If the distinct has a key list that is not the whole row.
    """
    var held = plan.nodes[at].exprs.copy()
    var width = len(pipe.schema)
    var memo = Memo()
    var keys = List[Int](capacity=width)
    if len(held) == 0:
        for i in range(width):
            keys.append(i)
    else:
        for i in range(len(held)):
            if plan.exprs.nodes[held[i]].kind != ExprKind.COLUMN:
                raise Error(
                    String(
                        "lower: this distinct decides on a ",
                        plan.exprs.nodes[held[i]].kind,
                        (
                            " expression, and a computed key is a column the"
                            " row does not have, so the rows kept would not be"
                            " the rows the plan said"
                        ),
                    )
                )
            keys.append(_lower_expr(plan.exprs, held[i], pipe, width, "", memo))
        if len(keys) != width:
            raise Error(
                String(
                    "lower: this distinct decides on ",
                    len(keys),
                    " of the row's ",
                    width,
                    (
                        " columns, and keeping a whole row chosen by part of it"
                        " means a first over every other column, which a group"
                        " by puts behind its keys rather than where the plan"
                        " numbered them"
                    ),
                )
            )
        for i in range(width):
            if keys[i] != i:
                raise Error(
                    String(
                        "lower: this distinct decides on the row's columns in"
                        " another order, and a group by carries its keys in the"
                        " order it was given them, so the answer would not be"
                        " the columns the plan numbered"
                    )
                )
    pipe.add(Node(Group(keys^, List[GroupAgg]())))


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

    Args:
        plan: The plan.
        at: The join node.
        frames: One frame per relation, taken from.
        taken: Which relations have already gone, written through.
        pipe: The pipeline, added to.

    Raises:
        Error: If the join is one the probe operator does not do, if it has
            anything other than one key pair of plain columns, if the two sides
            share a column name, or whatever the build side itself refuses.
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
    if kind == JoinKind.CROSS:
        raise Error(
            "lower: a cross join has no key to build a table from, and pairing"
            " every row with every row is the whole frame join rather than a"
            " probe"
        )

    var right = plan.nodes[at].inputs[1]
    var parts = plan.nodes[at].parts
    if parts != 1:
        raise Error(
            String(
                "lower: this operator joins on one column and this join has ",
                parts,
                (
                    " key pairs, which needs the ordinal space that"
                    " concatenating both key columns builds, and concatenating"
                    " both sides is having them both"
                ),
            )
        )
    var left_key = plan.nodes[at].exprs[0]
    var right_key = plan.nodes[at].exprs[1]
    if (
        plan.exprs.nodes[left_key].kind != ExprKind.COLUMN
        or plan.exprs.nodes[right_key].kind != ExprKind.COLUMN
    ):
        raise Error(
            "lower: this operator joins on a column on each side, and a"
            " computed key would have to be computed on the build side too,"
            " which is a projection over a frame rather than over a chunk"
        )
    var left_on = plan.exprs.nodes[left_key].name.copy()
    var right_on = plan.exprs.nodes[right_key].name.copy()

    var build: DataFrame
    if plan.nodes[right].kind == NodeKind.SCAN:
        build = _take(frames, taken, plan, right)
    else:
        var side = _lower_from(plan, right, frames, taken)
        build = side^.run()
    if kind.keeps_right_columns():
        # The operator renames a right column whose name the left already has,
        # and drops the right key outright when the two keys are called the
        # same. Either one moves the columns the plan's schema numbered, and a
        # position that means something else is a wrong answer rather than a
        # missing feature, so it is refused while the operator carries no output
        # names of its own.
        for i in range(len(build.schema)):
            for j in range(len(pipe.schema)):
                if build.schema[i].name == pipe.schema[j].name:
                    raise Error(
                        String(
                            (
                                "lower: both sides of this join have a column"
                                " called '"
                            ),
                            build.schema[i].name,
                            (
                                "', and the probe operator renames the right"
                                " one, so the result would not be the two"
                                " schemas end to end the way the plan numbered"
                                " them"
                            ),
                        )
                    )
    pipe.add(Node(Join(build^, left_on^, right_on^, kind)))


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
    else:
        source = _take(frames, taken, plan, first)
    var pipe = Pipeline(source^)

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
