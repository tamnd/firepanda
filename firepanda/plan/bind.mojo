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
will really answer and not a second opinion about it. Comparing a date column
against a string literal is the one place binding reads a value rather than a
type alone, and `_instant_literal` has the whole of why.

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
from firepanda.kernel.binary import BinaryOp, binary_type, resolve_constant
from firepanda.kernel.group import AggKind, agg_type
from firepanda.kernel.unary import UnaryOp, unary_type
from firepanda.plan.expr import UNBOUND, ExprKind, Expressions
from firepanda.plan.node import (
    SET_EXCEPT,
    SET_INTERSECT,
    SET_UNION,
    NodeKind,
    Plan,
)


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


def _resolve(
    schema: Schema, origin: List[Int], name: String, pin: Int
) raises -> Int:
    """Returns the position of a column, or refuses and says why.

    A name that is in the schema twice is refused rather than answered with the
    first of them. A join puts its two inputs end to end, so two tables that
    both have a `key` produce a schema with two columns called `key`, and
    answering the left one is a wrong answer that nothing downstream can notice.
    The caller says which input it meant, with `column_of`, and the ambiguity
    stops existing.

    Args:
        schema: The columns that are visible.
        origin: Which input each of those columns came from.
        name: What was written.
        pin: The input the name is to be looked for in, or `UNBOUND` to look in
            all of them.

    Returns:
        The position of the column.

    Raises:
        If nothing has that name, or more than one thing does.
    """
    var found = -1
    var again = -1
    for i in range(len(schema)):
        if schema[i].name == name and (pin == UNBOUND or origin[i] == pin):
            if found == -1:
                found = i
            else:
                again = i
                break
    if again != -1:
        var message = String(
            "'",
            name,
            "' is the name of more than one column here, at ",
            found,
            " and at ",
            again,
        )
        if origin[found] != origin[again]:
            # Two inputs each have one, which is what a join of two tables that
            # share a key looks like, and the caller has a way to say which.
            message += ", so say which input it is from"
        raise Error(message)
    if found != -1:
        return found

    if pin != UNBOUND:
        # The name may well be in the schema, on a column from another input,
        # and saying it is not here at all would send the reader looking for a
        # typo that is not there.
        for i in range(len(schema)):
            if schema[i].name == name:
                raise Error(
                    String(
                        "there is no column named '",
                        name,
                        "' in input ",
                        pin,
                        ", though another input has one",
                    )
                )
        raise Error(
            String(
                "there is no column named '", name, "' in input ", pin, " here"
            )
        )
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
    """Returns what a named function answers, for the seven that exist.

    There is no function registry yet, and the plan needs the connectives now
    because `a AND b` is a call rather than a binary operation, so this is a
    table of seven entries instead. When the registry arrives this function
    becomes a lookup in it and the table goes away.

    `like` is here with them because a pattern match is not a binary operation
    either. Its right side is a pattern rather than an operand, and the node
    that runs it reads that pattern once at plan time.

    `is_null` and `is_not_null` are here for the same kind of reason. `x = NULL`
    is null for every row, so the test that SQL spells with words cannot be the
    equality it looks like, and holding it as a call is what keeps the two
    apart all the way down to the operator.

    `coalesce` is the one whose answer is not a boolean. It reads any number of
    arguments and answers whichever type they all promote to, and the promotion
    is worked out here rather than in lowering, because lowering has to cast
    each argument to it and needs to be told what it is.

    Args:
        name: The function name.
        args: What each argument binds to.

    Returns:
        The type the call answers.

    Raises:
        If the name is not one of the seven, or an argument has the wrong type.
    """
    if name == "coalesce":
        if len(args) == 0:
            raise Error(
                "'coalesce' answers the first of its arguments that is not"
                " null, and was given none to choose from"
            )
        var want = args[0]
        for i in range(1, len(args)):
            try:
                want = promote(want, args[i])
            except e:
                raise Error(
                    String(
                        (
                            "'coalesce' answers one column, so its arguments"
                            " have to agree on a type, and argument "
                        ),
                        i,
                        " is a ",
                        args[i],
                        " among ",
                        want,
                        ": ",
                        e,
                    )
                )
        return want
    if name == "is_null" or name == "is_not_null":
        # No check on the type. Every column can hold a null, and a column that
        # cannot is still a fair thing to ask about: the answer is then the same
        # for every row, which is a true answer and not an error.
        if len(args) != 1:
            raise Error(
                String(
                    "'", name, "' takes 1 argument and was given ", len(args)
                )
            )
        return LogicalType.BOOL
    if name == "like":
        # The pattern is a constant and lowering refuses it as anything else,
        # but that is lowering's rule rather than a typing one, so what is
        # checked here is only that both sides are text.
        if len(args) != 2:
            raise Error(
                String("'like' takes 2 arguments and was given ", len(args))
            )
        for i in range(2):
            if args[i] != LogicalType.STRING and args[i] != LogicalType.NULL:
                raise Error(
                    String(
                        "'like' reads text and argument ",
                        i,
                        " is ",
                        args[i],
                    )
                )
        return LogicalType.BOOL
    if name != "and" and name != "or" and name != "not":
        raise Error(
            String(
                "there is no function named '",
                name,
                "' yet, so a plan cannot say what it answers",
            )
        )
    # `and` and `or` take two or more, because the simplify pass flattens
    # `a AND (b AND c)` into one call with three arguments and a plan that has
    # been through a pass has to bind again afterwards.
    if name == "not":
        if len(args) != 1:
            raise Error(
                String("'not' takes 1 argument and was given ", len(args))
            )
    elif len(args) < 2:
        raise Error(
            String(
                "'",
                name,
                "' takes two or more arguments and was given ",
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


def _instant_literal(
    exprs: Expressions,
    kids: List[Int],
    mut below: List[LogicalType],
    op: BinaryOp,
) raises:
    """Reads a text literal in a comparison against instants as an instant.

    `WHERE EventDate >= '2013-07-01'` is how every SQL dialect writes a date
    bound and it is what seven of the 43 ClickBench statements are made of. The
    literal is text and the column holds days, the two have no common type, and
    without this the comparison is refused before a row moves.

    It is a rule about a literal and not a rule about the text type. A literal
    is something the person writing the query typed, and typing a date between
    quotes is how the language spells one, so reading it as a date is reading
    what they wrote. A column of text is not that: nobody said those rows are
    dates, and the comparison stays refused so that whoever wrote it says which
    they meant with a cast.

    The type is worked out here and the value is not touched. `resolve_constant`
    is what reads the text, and it is already what `ComputeNode.schema` and
    `binary_value_any` call on the constant when the comparison runs, so the
    type this declares is the type the loop will really answer and the literal
    gets read in one place rather than in two. The arena is left alone on
    purpose: an index can be read by more than one plan node, which is the
    reason `Expressions.rebuild` adds a node rather than editing one, and a
    literal rewritten in place here would change a comparison under a node that
    never asked.

    Args:
        exprs: The arena, read for the operands' kinds and values.
        kids: The two operands.
        below: The operands' types, with the literal's replaced by what it will
            be read as.
        op: The operation, since only a comparison reads a literal this way.

    Raises:
        Error: If the text is not a date or a timestamp the column can be
            compared against, which quotes the text and comes from
            `parse_instant`.
    """
    if not op.is_comparison():
        # Arithmetic on an instant and text has no answer whatever the text
        # says, and reading it first would report a badly written date to
        # somebody whose real problem is that they added two things that do not
        # add. The promotion's refusal is the one that fits.
        return
    if below[0].is_temporal() == below[1].is_temporal():
        return
    var side = 1 if below[0].is_temporal() else 0
    if exprs.nodes[kids[side]].kind != ExprKind.LITERAL:
        return
    if not below[side].is_variable_width():
        return
    if not exprs.nodes[kids[side]].value.present:
        # A null literal is a comparison that answers null for every row and
        # the kernels have that already. There is no text to read.
        return
    below[side] = resolve_constant(
        below[1 - side], exprs.nodes[kids[side]].value, op
    ).type


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
        # A table already on the node is a caller saying which input the name is
        # in, so it narrows the lookup rather than being overwritten by it. The
        # write afterwards is the same number again in that case, and the one
        # binding worked out when there was nothing there.
        var pin = exprs.nodes[root].table
        var at = _resolve(schema, origin, exprs.nodes[root].name, pin)
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
        var op = BinaryOp(UInt8(exprs.nodes[root].op))
        _instant_literal(exprs, kids, below, op)
        exprs.nodes[root].type = binary_type(op, below[0], below[1])
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
    var one = List[Int](length=len(src), fill=table)
    var out = Schema()
    for i in range(len(node_names)):
        try:
            out.append(src[_resolve(src, one, node_names[i], UNBOUND)].copy())
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

    if kind == NodeKind.VALUES:
        return _bind_values(plan, at)

    if kind == NodeKind.TABLE_FUNCTION:
        return _bind_table_function(plan, at)

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

    if kind == NodeKind.WINDOW:
        # The one node that adds to what is below rather than replacing it, so
        # everything the input produces comes through at the position it had and
        # the windows are appended in the order they were written.
        var wider = Schema(copy=input.schema)
        var whose = input.origin.copy()
        for i in range(len(exprs)):
            wider.append(
                Field(
                    plan.nodes[at].names[i],
                    plan.exprs.nodes[exprs[i]].type,
                    _nullable(plan.exprs, exprs[i], input.schema),
                )
            )
            whose.append(_origin_of(plan.exprs, exprs[i]))
        return Bound(wider^, whose^)

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
    if kind == JoinKind.MARK:
        # The same question kept as an answer instead of acted on, so the output
        # is the left input with one boolean on the end of it. It is nullable
        # whatever the keys are, since a row that matched nothing against a side
        # holding a null does not know whether it matched.
        var name = plan.nodes[at].names[0].copy()
        if left.schema.has(name):
            raise Error(
                String(
                    "a mark join was told to call its column '",
                    name,
                    "', and the left side already has a column of that name",
                )
            )
        var fields = List[Field]()
        for i in range(len(left.schema)):
            fields.append(left.schema[i].copy())
        fields.append(Field(name^, LogicalType.BOOL))
        var origin = left.origin.copy()
        origin.append(UNBOUND)
        return Bound(Schema(fields^), origin^)
    return _widen(
        left,
        right,
        kind == JoinKind.RIGHT or kind == JoinKind.OUTER,
        kind == JoinKind.LEFT or kind == JoinKind.OUTER,
    )


def _bind_table_function(mut plan: Plan, at: Int) raises -> Bound:
    """Binds a call to a function that produces rows.

    Two functions are known and they are the same function with two ends on it.
    `range` stops before the value it was given and `generate_series` stops on
    it, which is DuckDB's rule and is the whole difference between them. Both
    produce one column of int64, and both take a stop, a start and a stop, or a
    start, a stop and a step.

    Anything else is refused by name, with the names that do work in the
    message. A plan that carried a function nobody had written would bind
    happily and fail when a row was asked for, which moves the error from where
    the query was written to where it was run.

    The arguments are bound against nothing, the way a `VALUES` binds its rows,
    because a table function is called where a table goes and has nothing under
    it to read. What binding does here is type them, since `range(2 + 3)` knows
    its type only after arithmetic.

    Args:
        plan: The plan, written through.
        at: The node.

    Returns:
        What the call produces.

    Raises:
        If the function is not one of the two, if it has the wrong number of
        arguments, or if an argument is not a whole number and is not `NULL`.
    """
    var name = plan.nodes[at].source
    if name != "range" and name != "generate_series":
        raise Error(
            String(
                "there is no table function called ",
                name,
                ", and the ones there are are range and generate_series",
            )
        )
    var args = plan.nodes[at].exprs.copy()
    if len(args) == 0 or len(args) > 3:
        raise Error(
            String(
                name,
                (
                    " takes a stop, a start and a stop, or a start, a stop and"
                    " a step, and this call has "
                ),
                len(args),
                " arguments",
            )
        )
    for i in range(len(args)):
        bind_expr(plan.exprs, args[i], Schema(), List[Int]())
        var t = plan.exprs.nodes[args[i]].type
        # A `NULL` argument passes, because a series whose end nobody knows is
        # a series of no rows rather than a query that was written wrong, which
        # is DuckDB's reading of it too.
        if not t.is_integer() and t != LogicalType.NULL:
            raise Error(
                String(
                    "argument ",
                    i + 1,
                    " of ",
                    name,
                    " counts rows and is a ",
                    t,
                    ", and counting is done in whole numbers",
                )
            )
    if len(plan.nodes[at].names) != 1:
        raise Error(
            String(
                name,
                " produces one column and this call names ",
                len(plan.nodes[at].names),
            )
        )
    var out = Schema()
    out.append(Field(plan.nodes[at].names[0], LogicalType.INT64, False))
    var origin = List[Int]()
    origin.append(UNBOUND)
    return Bound(out^, origin^)


def _bind_values(mut plan: Plan, at: Int) raises -> Bound:
    """Binds a literal table and works out what it produces.

    Nothing to bind against, which is the whole point of the node, so this is
    typing rather than resolving. A column's type is the one that holds every
    row's value in that position, worked out the same way a union works out the
    type that holds both of its arms, and for the same reason: a column of an
    int32 and an int64 is an int64 column and not an error.

    A column can be missing if any row's value in it can be, so a single `NULL`
    in a column of a hundred literals makes that column nullable and the other
    ninety nine do not make it otherwise.

    No column belongs to a relation, since there is no relation. A qualified name
    over a `VALUES` has nothing to qualify, and leaving the origin unbound is
    what makes that a name that does not resolve rather than one that resolves to
    whatever relation zero happens to be.

    The expressions are still bound, against nothing, because a literal knows
    its type and `1 + 1` does not. Binding is what works out the type of the
    second, and against an empty schema it is arithmetic and nothing else.

    Args:
        plan: The plan, written through.
        at: The node.

    Returns:
        What the literal table produces.

    Raises:
        If a column holds two values with no type that holds both.
    """
    var width = plan.nodes[at].parts
    var rows = plan.nodes[at].exprs.copy()
    for i in range(len(rows)):
        bind_expr(plan.exprs, rows[i], Schema(), List[Int]())
    var out = Schema()
    var origin = List[Int]()
    for j in range(width):
        var dtype = plan.exprs.nodes[rows[j]].type
        var nullable = _nullable(plan.exprs, rows[j], Schema())
        for i in range(width + j, len(rows), width):
            var next = plan.exprs.nodes[rows[i]].type
            try:
                dtype = promote(dtype, next)
            except:
                raise Error(
                    String(
                        "column ",
                        j,
                        " of a VALUES puts ",
                        next,
                        " against ",
                        dtype,
                        ", and there is no type that holds both",
                    )
                )
            nullable = nullable or _nullable(plan.exprs, rows[i], Schema())
        out.append(Field(plan.nodes[at].names[j], dtype, nullable))
        origin.append(UNBOUND)
    return Bound(out^, origin^)


def _bind_union(mut plan: Plan, at: Int, done: List[Bound]) raises -> Bound:
    """Binds a set operation and works out what it produces.

    Takes the names from the first input, because that is what pandas and SQL
    both do and because the alternative is refusing a query over two tables that
    spell the same column differently, which nobody wants. The types are
    promoted pairwise, so stacking an int32 column on an int64 one gives int64
    rather than an error, and that is true of all three operations: comparing an
    int32 against an int64 to decide whether a row is in both needs a type that
    holds both as much as stacking them does.

    Where the three differ is what can be missing. Every row a union produces
    came from one of its inputs, so a column is nullable if any input's is. Every
    row a difference produces came from the left one, so only the left's
    nullability counts. And every row an intersection produces was in both, so a
    column that cannot be missing on either side cannot be missing in the answer,
    which is the only one of the three where a nullable column comes out not
    nullable.

    Which relation a column came from follows the same reasoning. A union's
    column is two columns stacked, so a qualified name above it would be a
    guess and the origin is dropped. A difference's rows are the left's rows, and
    an intersection's rows are in both and therefore are the left's rows too, so
    both of them keep the left input's origin and a qualified name still
    resolves.

    Args:
        plan: The plan.
        at: The node.
        done: What every node below produces.

    Returns:
        What the set operation produces.

    Raises:
        If the inputs are different widths, or a column pair has no type that
        holds both.
    """
    var op = plan.nodes[at].op
    ref inputs = plan.nodes[at].inputs
    var out = Schema(copy=done[inputs[0]].schema)
    var origin = done[inputs[0]].origin.copy()
    var word = "a union"
    if op == SET_EXCEPT:
        word = "a difference"
    elif op == SET_INTERSECT:
        word = "an intersection"
    for i in range(1, len(inputs)):
        ref other = done[inputs[i]]
        if len(other.schema) != len(out):
            raise Error(
                String(
                    word,
                    " is between a ",
                    len(out),
                    " column input and a ",
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
                        word,
                        " puts ",
                        other.schema[j].dtype,
                        " against ",
                        out[j].dtype,
                        " in column ",
                        j,
                        ", and there is no type that holds both",
                    )
                )
            if op == SET_UNION:
                out.fields[j].nullable = (
                    out[j].nullable or other.schema[j].nullable
                )
            elif op == SET_INTERSECT:
                out.fields[j].nullable = (
                    out[j].nullable and other.schema[j].nullable
                )
            # A difference keeps the left side's, which it already has.
            if op == SET_UNION and origin[j] != other.origin[j]:
                origin[j] = UNBOUND
    return Bound(out^, origin^)
