"""Binding: a name becomes a position, and every expression gets a type.

A plan comes out of the builders in `node.mojo` holding names. `col("l_discount")`
is a string, and finding out which column that is means asking a schema. Binding
is the pass that asks, once, and writes the answer onto the nodes: a position, a
relation, and the logical type the expression produces.

Two things fall out of that and both of them are the point.

The first is that execution stops looking names up. `DataFrame.column(name)`
copies and flattens the column it finds, which is ninety six megabytes on a text
column of six million rows, and every hand written query in the TPC-H driver has
a helper whose only job is to turn a name into an index so that the borrowing
accessor can be used instead. A bound plan holds the index already, so the
question stops existing rather than getting a faster answer.

The second is that a type error becomes a plan error. Adding a string to a date
is refused here, with the operation named, before a single row has moved.
`binary_type`, `unary_type` and `agg_type` are the same functions the kernels
use to decide what they produce, so the type on the node is the type the kernel
will really answer and not a second opinion about it.

## What binding computes for a node

A `Bound`: the schema the node produces, and one relation id per output column
saying where that column came from. The schema is what the node above binds
against. The origins are what a column reference copies into `Expr.table`, which
is what the table set analysis reads and what predicate pushdown is written on.

A computed column has no single relation. `l_extendedprice * (1 - l_discount)`
reads one, so it keeps it, but `1 as n` reads none and `o_totalprice + c_acctbal`
reads two, and neither of those is a bit index. Both get `UNBOUND`, and a
reference to one of them is bound for position and type and left without a
table. `Expressions.tables` then refuses it, which is the honest answer: pushdown
cannot decide where that predicate can go by looking at the reference alone, it
has to substitute the projection's expression first and ask again about what
comes out.

## The walk

Node indices are handed out in creation order, so an input always sits at a
lower index than the node that reads it. Binding walks forwards from zero and
every input it needs is already done, with no recursion and no memo. Nodes that
are not under the root are skipped, since a plan can hold a subtree that a
rewrite has already detached and binding it would report errors about a query
nobody asked for.

Expressions are the other way around. The arena is shared by every node and each
node sees a different schema, so an expression is bound from its root downwards,
against the schema of the node holding it. A subtree shared by two roots under
the same node gets bound twice with the same answer, which is why binding writes
rather than accumulates.
"""

from firepanda.dtype.logical import LogicalType, promote
from firepanda.dtype.schema import Field, Schema
from firepanda.join.pairs import JoinKind
from firepanda.kernel.binary import BinaryOp, binary_type
from firepanda.kernel.group import AggKind, agg_type
from firepanda.kernel.unary import UnaryOp, unary_type
from firepanda.plan.expr import UNBOUND, ExprKind, Expressions
from firepanda.plan.node import NodeKind, Plan


comptime SUGGESTIONS = 3
"""How many names a `Did you mean` line will offer at most. Three is the number
the SQL matcher settled on and there is no reason for the two to differ."""


struct Bound(Copyable, Movable):
    """What binding worked out about one plan node.

    Two lists of the same length in effect, held as a schema and a list beside
    it rather than as one list of triples, because the schema is what gets
    handed back to the caller and what the node above binds against, and it
    should not have a planner only field welded into it.
    """

    var schema: Schema
    """What the node produces, in position order."""

    var origin: List[Int]
    """One relation id per column, or `UNBOUND` for a column that no single
    relation produced. See the module docstring for why both of the cases that
    give `UNBOUND` deserve it."""

    def __init__(out self, var schema: Schema, var origin: List[Int]):
        """Builds a bound node.

        Args:
            schema: What the node produces.
            origin: Where each column came from.
        """
        self.schema = schema^
        self.origin = origin^

    def __init__(out self, var schema: Schema, table: Int):
        """Builds a bound node whose every column came from one relation.

        Args:
            schema: What the node produces.
            table: The relation all of it came from.
        """
        self.origin = List[Int](length=len(schema), fill=table)
        self.schema = schema^


def _edits(a: String, b: String) -> Int:
    """Returns the number of single character edits between two names.

    The plain Levenshtein distance, one row at a time so that the working set is
    one name wide rather than two names square. It runs on the error path only,
    against the columns of one schema, so it is written for clarity.

    Args:
        a: One name.
        b: The other.

    Returns:
        How many insertions, deletions and substitutions turn one into the
        other.
    """
    var left = a.as_bytes()
    var right = b.as_bytes()
    var row = List[Int](length=len(right) + 1, fill=0)
    for j in range(len(right) + 1):
        row[j] = j
    for i in range(1, len(left) + 1):
        var corner = row[0]
        row[0] = i
        for j in range(1, len(right) + 1):
            var was = row[j]
            var cost = 0 if left[i - 1] == right[j - 1] else 1
            var best = corner + cost
            if row[j] + 1 < best:
                best = row[j] + 1
            if row[j - 1] + 1 < best:
                best = row[j - 1] + 1
            row[j] = best
            corner = was
    return row[len(right)]


def _near(schema: Schema, name: String) -> String:
    """Builds a `Did you mean` line for a name that did not resolve.

    Offers only names within a distance that scales with the length of what was
    written, since one edit out of four characters is a different word and one
    edit out of fourteen is a typo. A schema with no near miss in it gets no
    line at all, which is the common case and is better than a suggestion that
    is obviously not what was meant.

    Ties are kept in schema order rather than sorted, so the same query always
    produces the same message.

    Args:
        schema: The columns that were visible.
        name: What was written.

    Returns:
        A line to put under the message, or an empty string.
    """
    var limit = 1 + name.byte_length() // 4
    var best = limit + 1
    for i in range(len(schema)):
        var d = _edits(name, schema[i].name)
        if d < best:
            best = d
    if best > limit:
        return String()
    var near = List[String]()
    for i in range(len(schema)):
        if _edits(name, schema[i].name) == best:
            var seen = False
            for j in range(len(near)):
                if near[j] == schema[i].name:
                    seen = True
            if not seen and len(near) < SUGGESTIONS:
                near.append(schema[i].name)
    if len(near) == 0:
        return String()
    var out = String("Did you mean ")
    for i in range(len(near)):
        if i > 0:
            out += " or " if i == len(near) - 1 else ", "
        out += String("'", near[i], "'")
    out += "?"
    return out^


def _resolve(schema: Schema, name: String) raises -> Int:
    """Returns the position of a column, or refuses with a suggestion.

    Args:
        schema: The columns that are visible.
        name: What was written.

    Returns:
        The position of the first column with that name.

    Raises:
        If nothing has that name.
    """
    for i in range(len(schema)):
        if schema[i].name == name:
            return i
    var message = String("there is no column named '", name, "' here")
    var hint = _near(schema, name)
    if hint:
        message += ". " + hint
    raise Error(message)


def _bool(t: LogicalType) -> Bool:
    """Whether a type can be read as a yes or a no.

    Null counts, because a predicate that is null everywhere keeps no rows and
    that is an answer rather than an error, and it is what an untyped literal
    null binds to.

    Args:
        t: The type.

    Returns:
        True if it is boolean or null.
    """
    return t == LogicalType.BOOL or t == LogicalType.NULL


def _call_type(name: String, args: List[LogicalType]) raises -> LogicalType:
    """Returns what a named function answers, for the three that exist.

    There is no function registry yet, and the plan needs the connectives now
    because `a AND b` is a call rather than a binary operation, so this is a
    table of three entries instead. When the registry arrives this function
    becomes a lookup in it and the table goes away.

    Args:
        name: The function name.
        args: What each argument binds to.

    Returns:
        The type the call answers.

    Raises:
        If the name is not one of the three, or an argument is not a boolean.
    """
    if name != "and" and name != "or" and name != "not":
        raise Error(
            String(
                "there is no function named '",
                name,
                "' yet, so a plan cannot say what it answers",
            )
        )
    var wanted = 1 if name == "not" else 2
    if len(args) != wanted:
        raise Error(
            String(
                "'",
                name,
                "' takes ",
                wanted,
                " arguments and was given ",
                len(args),
            )
        )
    for i in range(len(args)):
        if not _bool(args[i]):
            raise Error(
                String(
                    "'",
                    name,
                    "' reads yes or no and argument ",
                    i,
                    " is ",
                    args[i],
                )
            )
    return LogicalType.BOOL


def bind_expr(
    mut exprs: Expressions, root: Int, schema: Schema, origin: List[Int]
) raises:
    """Binds one expression tree against the columns a node can see.

    Writes a position, a relation and a type onto every node of the tree.
    Children first, since every type here is a function of the types below it.

    Args:
        exprs: The arena, written through.
        root: The expression.
        schema: The columns that are visible.
        origin: Where each of those columns came from.

    Raises:
        If a name does not resolve, or an operation has no answer for the types
        it was handed.
    """
    exprs.check(root)
    var kind = exprs.nodes[root].kind

    if kind == ExprKind.COLUMN:
        var at = _resolve(schema, exprs.nodes[root].name)
        exprs.nodes[root].at = at
        exprs.nodes[root].table = origin[at]
        exprs.nodes[root].type = schema[at].dtype
        return

    var kids = exprs.nodes[root].children.copy()
    for i in range(len(kids)):
        bind_expr(exprs, kids[i], schema, origin)
    var below = List[LogicalType]()
    for i in range(len(kids)):
        below.append(exprs.nodes[kids[i]].type)

    if kind == ExprKind.LITERAL or kind == ExprKind.CAST:
        # Both knew what they produce before binding started. A literal carries
        # its value's type and a cast carries its target, and neither of them is
        # a question about the input.
        return

    if kind == ExprKind.UNARY:
        exprs.nodes[root].type = unary_type(
            UnaryOp(exprs.nodes[root].op), below[0]
        )
    elif kind == ExprKind.BINARY:
        exprs.nodes[root].type = binary_type(
            BinaryOp(UInt8(exprs.nodes[root].op)), below[0], below[1]
        )
    elif kind == ExprKind.CALL:
        exprs.nodes[root].type = _call_type(exprs.nodes[root].name, below)
    elif kind == ExprKind.AGGREGATE or kind == ExprKind.WINDOW:
        exprs.nodes[root].type = agg_type(
            AggKind(UInt8(exprs.nodes[root].op)), below[0]
        )
    else:
        if not _bool(below[0]):
            raise Error(
                String(
                    "a conditional asks a yes or no question and this one asks",
                    " ",
                    below[0],
                )
            )
        exprs.nodes[root].type = promote(below[1], below[2])


def _origin_of(exprs: Expressions, root: Int) -> Int:
    """Returns the one relation an expression reads, if there is exactly one.

    Not `Expressions.tables`, on purpose. That one raises on an unbound column
    because an empty mask is a real answer there and handing it back for a
    column nobody has resolved yet would let pushdown move a predicate past the
    node that feeds it. Here every column below has just been bound, and a
    column with no relation is a reference to something a projection computed,
    which is a case rather than a mistake.

    Args:
        exprs: The arena.
        root: The expression, already bound.

    Returns:
        The relation, or `UNBOUND` when the expression reads none or more than
        one.
    """
    ref node = exprs.nodes[root]
    if node.kind == ExprKind.COLUMN:
        return node.table
    var seen = UNBOUND
    for i in range(len(node.children)):
        var here = _origin_of(exprs, node.children[i])
        if here == UNBOUND:
            continue
        if seen != UNBOUND and seen != here:
            return UNBOUND
        seen = here
    return seen


def _nullable(exprs: Expressions, root: Int, schema: Schema) -> Bool:
    """Whether an expression can produce a missing value.

    Only the two cases worth being sure about are answered precisely. A
    reference to a column that cannot be null cannot be null, and a literal that
    is not null is not null. Everything else says yes, which is the safe
    direction: claiming a column has no nulls when it does would let a later
    pass drop a validity check that was doing something.

    Args:
        exprs: The arena.
        root: The expression, already bound.
        schema: The columns it was bound against.

    Returns:
        True unless the expression is one of the two that provably cannot be
        null.
    """
    ref node = exprs.nodes[root]
    if node.kind == ExprKind.COLUMN and node.at != UNBOUND:
        return schema[node.at].nullable
    if node.kind == ExprKind.LITERAL:
        return node.value.is_null()
    return True


def _scan(
    node_names: List[String], source: String, table: Int, src: Schema
) raises -> Bound:
    """Works out what a scan produces.

    An empty column list means the whole table, which is what a scan built
    before projection pushdown has run looks like.

    Args:
        node_names: The columns the scan reads, or none for all of them.
        source: The table name, for the message.
        table: Which relation the scan is.
        src: The schema of the table being read.

    Returns:
        The scan's output.

    Raises:
        If a named column is not in the table.
    """
    if len(node_names) == 0:
        return Bound(Schema(copy=src), table)
    var out = Schema()
    for i in range(len(node_names)):
        try:
            out.append(src[_resolve(src, node_names[i])].copy())
        except e:
            raise Error(String("scan of ", source, ": ", e))
    return Bound(out^, table)


def _widen(a: Bound, b: Bound, first: Bool, second: Bool) raises -> Bound:
    """Stacks two bound nodes side by side for a join.

    Args:
        a: The left input.
        b: The right input.
        first: Whether the left columns can go missing in the output.
        second: Whether the right columns can.

    Returns:
        The two schemas concatenated, with nullability widened where the join
        kind says a side can go missing.

    Raises:
        Never, and says so because a caller in a raising chain reads better than
        a caller that has to know this one is not.
    """
    var out = Schema()
    var origin = List[Int]()
    for i in range(len(a.schema)):
        var f = a.schema[i].copy()
        f.nullable = f.nullable or first
        out.append(f^)
        origin.append(a.origin[i])
    for i in range(len(b.schema)):
        var f = b.schema[i].copy()
        f.nullable = f.nullable or second
        out.append(f^)
        origin.append(b.origin[i])
    return Bound(out^, origin^)


def bind(mut plan: Plan, root: Int, sources: List[Schema]) raises -> Schema:
    """Binds a whole plan and returns what it produces.

    Every column reference under `root` comes out with a position, a relation
    and a type on it, and so does every expression above them. A plan that binds
    is a plan whose names all resolve and whose operations all have an answer
    for the types they were handed, which is most of what can be wrong with a
    query before it runs.

    Args:
        plan: The plan, written through.
        root: The node whose output is the answer.
        sources: The schema of each relation, indexed by the id a scan carries.

    Returns:
        The schema of the root.

    Raises:
        If the root is not in the plan, a scan names a relation with no schema,
        a column does not resolve, an operation has no answer for its operands,
        or two sides of a join or a union do not line up.
    """
    var done = bind_all(plan, root, sources)
    return Schema(copy=done[root].schema)


def bind_all(
    mut plan: Plan, root: Int, sources: List[Schema]
) raises -> List[Bound]:
    """Binds a whole plan and returns what every node of it produces.

    The same work `bind` does and the whole of its answer rather than the last
    line. A pass that rewrites a node has to know the width of what the node
    below it produces, which is what the schema of that node is, and throwing
    every one of them away except the root's would mean each pass binding the
    plan again to get them back.

    A node the root does not reach comes back with an empty schema, which is
    also what a pass wants: a subtree some earlier rewrite detached is not an
    error and is not something to rewrite either.

    Args:
        plan: The plan, written through.
        root: The node whose output is the answer.
        sources: The schema of each relation, indexed by the id a scan carries.

    Returns:
        One `Bound` per node from zero to `root`, in node order.

    Raises:
        If the root is not in the plan, a scan names a relation with no schema,
        a column does not resolve, an operation has no answer for its operands,
        or two sides of a join or a union do not line up.
    """
    plan.check(root)

    # Backwards from the root, so that a subtree a rewrite has detached is left
    # alone rather than reported on. Inputs sit below the nodes that read them,
    # so one reverse pass reaches everything.
    var wanted = List[Bool](length=root + 1, fill=False)
    wanted[root] = True
    for at in range(root, -1, -1):
        if not wanted[at]:
            continue
        for i in range(len(plan.nodes[at].inputs)):
            wanted[plan.nodes[at].inputs[i]] = True

    var done = List[Bound]()
    for _ in range(root + 1):
        done.append(Bound(Schema(), List[Int]()))

    for at in range(root + 1):
        if not wanted[at]:
            continue
        done[at] = _bind_node(plan, at, sources, done)
    return done^


def _bind_node(
    mut plan: Plan, at: Int, sources: List[Schema], done: List[Bound]
) raises -> Bound:
    """Binds one node, given its inputs already bound.

    Args:
        plan: The plan, written through.
        at: The node.
        sources: The schema of each relation.
        done: What every node below this one produces.

    Returns:
        What this node produces.

    Raises:
        If anything about this node does not line up.
    """
    var kind = plan.nodes[at].kind

    if kind == NodeKind.SCAN:
        var table = plan.nodes[at].table
        if table < 0 or table >= len(sources):
            raise Error(
                String(
                    "scan of ",
                    plan.nodes[at].source,
                    " is relation ",
                    table,
                    " and ",
                    len(sources),
                    " schemas were given",
                )
            )
        return _scan(
            plan.nodes[at].names,
            plan.nodes[at].source,
            table,
            sources[table],
        )

    if kind == NodeKind.UNION:
        return _bind_union(plan, at, done)

    if kind == NodeKind.JOIN:
        return _bind_join(plan, at, done)

    ref input = done[plan.nodes[at].inputs[0]]
    var exprs = plan.nodes[at].exprs.copy()
    for i in range(len(exprs)):
        bind_expr(plan.exprs, exprs[i], input.schema, input.origin)

    if kind == NodeKind.FILTER:
        var t = plan.exprs.nodes[exprs[0]].type
        if not _bool(t):
            raise Error(
                String(
                    "a filter keeps the rows a yes or no question says to keep",
                    " and this one asks ",
                    t,
                )
            )
        return Bound(Schema(copy=input.schema), input.origin.copy())

    if (
        kind == NodeKind.SORT
        or kind == NodeKind.LIMIT
        or kind == NodeKind.DISTINCT
    ):
        # None of the three changes a column, only which rows survive and in
        # what order, so the schema below is the schema above unchanged.
        return Bound(Schema(copy=input.schema), input.origin.copy())

    var out = Schema()
    var origin = List[Int]()
    for i in range(len(exprs)):
        out.append(
            Field(
                plan.nodes[at].names[i],
                plan.exprs.nodes[exprs[i]].type,
                _nullable(plan.exprs, exprs[i], input.schema),
            )
        )
        origin.append(_origin_of(plan.exprs, exprs[i]))
    return Bound(out^, origin^)


def _bind_join(mut plan: Plan, at: Int, done: List[Bound]) raises -> Bound:
    """Binds a join and works out what it produces.

    The two key lists bind against different schemas, which is the whole reason
    a join is not just another node with a list of expressions on it. The left
    keys see the left input and the right keys see the right one, and the output
    is the two schemas end to end so that the node above can reach either.

    Args:
        plan: The plan, written through.
        at: The node.
        done: What every node below produces.

    Returns:
        What the join produces.

    Raises:
        If a key does not resolve, or a key pair has no type both sides can be
        compared in.
    """
    ref left = done[plan.nodes[at].inputs[0]]
    ref right = done[plan.nodes[at].inputs[1]]
    var exprs = plan.nodes[at].exprs.copy()
    var keys = plan.nodes[at].parts
    for i in range(keys):
        bind_expr(plan.exprs, exprs[i], left.schema, left.origin)
    for i in range(keys, len(exprs)):
        bind_expr(plan.exprs, exprs[i], right.schema, right.origin)
    for i in range(keys):
        var a = plan.exprs.nodes[exprs[i]].type
        var b = plan.exprs.nodes[exprs[keys + i]].type
        try:
            _ = promote(a, b)
        except:
            raise Error(
                String(
                    "join key ",
                    i,
                    " compares ",
                    a,
                    " against ",
                    b,
                    ", and there is no type that holds both",
                )
            )

    var kind = JoinKind(UInt8(plan.nodes[at].op))
    if kind == JoinKind.SEMI or kind == JoinKind.ANTI:
        # Both ask a question about the right side and keep none of it, so the
        # output is the left input unchanged.
        return Bound(Schema(copy=left.schema), left.origin.copy())
    return _widen(
        left,
        right,
        kind == JoinKind.RIGHT or kind == JoinKind.OUTER,
        kind == JoinKind.LEFT or kind == JoinKind.OUTER,
    )


def _bind_union(mut plan: Plan, at: Int, done: List[Bound]) raises -> Bound:
    """Binds a union and works out what it produces.

    Takes the names from the first input, because that is what pandas and SQL
    both do and because the alternative is refusing a query over two tables that
    spell the same column differently, which nobody wants. The types are
    promoted pairwise, so stacking an int32 column on an int64 one gives int64
    rather than an error.

    Args:
        plan: The plan.
        at: The node.
        done: What every node below produces.

    Returns:
        What the union produces.

    Raises:
        If the inputs are different widths, or a column pair has no type that
        holds both.
    """
    ref inputs = plan.nodes[at].inputs
    var out = Schema(copy=done[inputs[0]].schema)
    var origin = done[inputs[0]].origin.copy()
    for i in range(1, len(inputs)):
        ref other = done[inputs[i]]
        if len(other.schema) != len(out):
            raise Error(
                String(
                    "a union stacks a ",
                    len(out),
                    " column input on a ",
                    len(other.schema),
                    " column one",
                )
            )
        for j in range(len(out)):
            try:
                out.fields[j].dtype = promote(
                    out[j].dtype, other.schema[j].dtype
                )
            except:
                raise Error(
                    String(
                        "a union stacks ",
                        other.schema[j].dtype,
                        " on ",
                        out[j].dtype,
                        " in column ",
                        j,
                        ", and there is no type that holds both",
                    )
                )
            out.fields[j].nullable = out[j].nullable or other.schema[j].nullable
            if origin[j] != other.origin[j]:
                origin[j] = UNBOUND
    return Bound(out^, origin^)
