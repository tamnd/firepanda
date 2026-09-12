"""Turning a bound `SELECT` into a logical plan.

This is where the SQL front end stops being its own thing and starts being a
caller of the engine. Everything before it is DuckDB's dialect: the grammar, the
AST, the type set, the overload table. Everything after it is shared with the
dataframe API, and the rule from docs/specs/sql/08-plan-and-optimizer.md section
6 is that no plan node may have only a SQL constructor. A shape SQL can express
and `firepanda/plan/` cannot is a gap in the plan to close, not a private node
to add on the side.

### The order the clauses come out in

A `SELECT` is written in one order and evaluated in another, and the plan is
built in the evaluation order, bottom up:

    SCAN      the table in the FROM
    FILTER    the WHERE
    AGGREGATE the GROUP BY and the aggregates in the select list
    FILTER    the HAVING
    PROJECT   the select list
    DISTINCT  the DISTINCT
    SORT      the ORDER BY
    LIMIT     the LIMIT and the OFFSET

Nothing about that is a choice this file makes. It is why `WHERE` cannot see an
alias from the select list and `ORDER BY` can, and why `HAVING` may name an
aggregate and `WHERE` may not. Writing the lowering in this order is what makes
those rules fall out rather than be enforced.

### A query with no table still reads one

`SELECT 1` has no `FROM` and a plan has no node that produces a row out of
nothing, until the literal table node. A query with no `FROM` lowers to a
projection over one row of one constant, and the projection drops the constant
again, so the only thing that row does is exist. `VALUES (1), (2)` is the same
node with the rows the query wrote in it.

That is what makes `SELECT 1` a plan rather than a special case, and it is why
the node went in rather than a flag somewhere saying this query has no input.

### A set operation is two of those with a node over them

`UNION`, `EXCEPT` and `INTERSECT` each take a whole query on both sides, so the
block above is what `_block` lowers and `_combine` is what stacks two of them.
The `ORDER BY` and the `LIMIT` written at the end belong to the set operation
rather than to its right arm, which is why they are put on in `lower` after the
arms are joined and not at the bottom of a block.

The chain nests left, so `a EXCEPT b EXCEPT c` lowers as `(a EXCEPT b) EXCEPT c`
and not as the other reading, which is a different answer and not a different
spelling of the same one. Each arm brings its own table into `sources`, so a scan
carries the offset of its own schema rather than always zero.

### A join condition becomes key pairs, and what is left becomes a filter

`plan.join` holds a left key and a right key per pair rather than a predicate,
because that is what a hash join runs. So an `ON` is split on `AND`, and a part
that is an equality with one side reading the left input and the other reading
the right is a key pair. Anything else is a residual.

A residual over an inner join is a filter above it. Every pairing the join
produces is a pairing the condition asked about, so testing the rest on top
answers the same query, and an inner join with no equality at all is a cross
join with the whole condition over it. A residual over an outer join is refused
instead, because an outer join's condition decides which rows are padded as well
as which rows match, and a filter above one would test the padding and drop the
row.

Which side a part reads is answered by the relation a qualified column carries,
and by the name for one that is not qualified. A comma in the `FROM` is a join
with no condition, which is why `FROM a, b WHERE a.x = b.y` and the same query
written with a `JOIN` reach the same plan once the filter is pushed down.

### A HAVING is a filter over a column, not over an aggregate

`plan.filter` refuses a predicate that is not elementwise, which is correct and
is the reason `HAVING sum(x) > 10` cannot lower to a filter holding a `sum`. The
aggregate has to be computed by the `AGGREGATE` node and named there, and the
filter then reads the name. So a `HAVING` over an aggregate adds an output to
the aggregate under a generated name, filters on that name, and the projection
above it drops the column again. The generated names start with two underscores
because a query cannot write one, and they are visible in `EXPLAIN`, which is
the right trade: a reader who sees `__having_0` learns something true about how
the query runs.

### A function may be written where a table goes

`FROM range(5)` is a source that reads no file and names no table, so nothing
about it touches the catalog and nothing is added to `sources`. Its arguments
are lowered as expressions against nothing, the way a `VALUES` lowers its rows,
because there is nothing under a table function to read.

The column comes out called after the function, which is what DuckDB calls it.
An alias is refused for now: the column belongs to no relation, so a name
written in front of it would have nothing to resolve against, and a scope entry
pointing at a relation that does not exist would resolve to the wrong one rather
than to none.

### A subquery may be written where a table goes

`FROM (SELECT ...) v` is a whole statement whose output becomes a source, so it
lowers to that statement's own root and nothing wraps it. What the query around
it gets is the names that root produces, which are read back off the plan rather
than threaded out of every function that lowers a body.

Its alias is not a relation. A scan is one because it has a schema in `sources`
and a number that names it, and a derived table has neither, since its columns
are computed by the nodes under it. So the alias goes into the scope as a name
that says which columns are meant, and `v.x` is checked against the columns the
subquery produces and then lowered as a bare `x`. Two sources in one `FROM` that
both produce a column of that name come back from binding as an ambiguity, which
is a refusal rather than a wrong answer.

### A `WITH` is a name bound to a statement, and it lowers where it is named

Every reference to a CTE lowers the statement again, in the place the name was
written, which is the derived table above with the statement found by name
instead of written out. That is DuckDB's default and it is the right one here
for a plainer reason: a plan holding one node twice would be a graph rather than
a tree, and every pass over it walks a tree. `MATERIALIZED` asks for the other
thing, and it is read and not acted on, which costs a query that writes it the
work of computing the statement more than once and never costs it a different
answer. Deciding it is the optimizer's, where the use count `read_ctes` already
worked out can be weighed against what the statement costs.

The names bind in the order they were written and a reference lowers against the
ones bound before it, which is what makes `WITH y AS (SELECT n FROM x), x AS
(...)` a missing table rather than a forward reference. A `WITH` inside a
subquery binds after the one outside it and a lookup runs from the end, so the
inner one shadows the outer, and the whole list is looked in before the catalog,
so a CTE hides a registered frame of the same name. The column alias list is a
projection over the statement's root, applied as the prefix rule that
`cte.mojo` and DuckDB both have.

A recursive CTE is refused by name. The fixed point it asks for is a node that
runs its own input until no new rows come out, and the plan has no such node.

### A star stands for the columns the FROM produces, minus what was hung off it

A bare `*` is one output per column in FROM order, each carrying the relation it
came from so that binding can tell two sources with the same column name apart.
`t.*` is the same list with the columns of every other relation dropped, which
works because that number is exactly what says where a column came from. The
three modifiers are applied in the order the grammar forces them to be written,
exclude then replace then rename, and `firepanda/sql/star.mojo` holds the rules
for what a modifier may name and the wording of every refusal, so the binder and
this stage do not disagree about what the query said.

`v.*` where `v` is a subquery in the `FROM` is refused. A derived table is not a
relation, so its columns arrive with no number on them and there is nothing here
to tell them from the columns of the source beside them. A modifier is never
qualified, because a dotted name in one is refused while the AST is built.

### A `USING` or a `NATURAL` join names its keys by column name

Both are one equality per named column, and `NATURAL` is `USING` over every
name the two sides share. The node is the join the `ON` spelling builds. What
differs is afterwards: each pair is handed out as one column rather than two,
so a star over the join writes the name once, and an unqualified reference to
it lowers pinned to one side instead of being refused for naming two columns.
Either side's name may still be written in front of it. Which side the merged
column comes from is which side the join keeps every row of, so a `RIGHT` join
hands out the right's and every other kind hands out the left's. A `FULL` one
is refused, because there the answer is the first of the pair that is not null
and that is a `coalesce` over the join rather than a column of it.

The node underneath still produces both columns of every pair. Dropping one
would mean a projection between the join and the query above it, and a
projection is where a column stops carrying the table it came from, which is
what `t.b` needs to still work. So the pair's other column is left in the node
and taken out of the names the query may reach instead.

### A `SEMI` or an `ANTI` join asks a question and keeps no answer

Both keep left rows and no right columns, the first the ones that matched and
the second the ones that did not. So the right side goes out of reach the moment
it has been lowered: its tables are taken back out of the scope, and what the
rest of the query may write is the left side alone, which is what DuckDB does
and is why `SELECT u.k FROM t SEMI JOIN u USING (b)` is a missing table there.

Both need at least one equality between the two sides and nothing else, which is
tighter than DuckDB, where the condition may be any predicate. The join node
carries key pairs rather than a predicate, and the rest of a condition is
ordinarily tested in a filter above the join, which here would be a filter over
columns the join did not keep.

### An `IN` over a subquery is that join rather than a predicate

`WHERE x IN (SELECT k FROM u)` is a semi join between whatever the `FROM` built
and the subquery's own plan, on `x = k`. So the `WHERE` is split on `AND` first
and the parts that are an `IN` over a subquery are taken out of it and put above
the rest, which stays one filter underneath. A query with no such part is split
and put back exactly as it was written, so its plan is the shape it always was.

The rewrite is the whole of `IN` and not a near miss. A left row that matched
several right rows comes back once, which is what a semi join does anyway, and a
null on either side matches nothing, which is what `IN` answers.

An `IN` written anywhere else is a value rather than a filter, and a value has to
arrive on every row rather than only on the rows that matched. That is the mark
join, which hands the left side out unchanged with one boolean column beside it,
and the `IN` in the expression becomes a read of that column. It goes above the
`FROM` and below everything else, the same place and for the same reason as the
cross join a subquery written as a value gets.

`NOT IN` goes there too, wherever it is written, because the anti join it looks
like is the classic wrong answer: one null anywhere in the subquery makes
`NOT IN` null for every row rather than true, and an anti join keeps those rows
rather than dropping them. The mark join marks such a row null rather than false,
the `NOT` over the column is null in turn, and a filter does not keep a null. So
the null aware anti join is a mark join and a `NOT`, with nothing written for it.

`x = ANY (SELECT k FROM u)` is that same `IN` and `x <> ALL (...)` is that same
`NOT IN`, so both are read as the one they are and go to the same two joins
rather than being lowered a second time. The nulls line up exactly as well:
`<> ALL` over a subquery holding a null is null
on every row that matched nothing, which is what the mark join and the `NOT`
over it already answer. The other four are the section after next.

The subquery lowers against a scope of its own, which is the ordinary rule, and
here it is also what makes the rewrite safe: a subquery that reads no outer
column runs once, and running it once is what a join does with its build side. A
correlated one refuses, since the name it reaches for is not in reach inside it.

### A correlated `EXISTS` is the same join, decorrelated

`WHERE EXISTS (SELECT 1 FROM u WHERE u.b = t.b AND u.k > 3)` is a semi join on
`t.b = u.b` with `u.k > 3` as a filter under it, and `NOT EXISTS` is the same
join asking the other way. That is decorrelation, done for the one shape where
it is a rewrite rather than a pass, and it is the shape worth having first: an
`EXISTS` read literally is a query run once per outer row.

So this one lowers its `FROM` into the scope the outer query is already using,
which is what puts both sides of the join in reach at once, and then splits its
`WHERE` the way a join condition is split. A part with one side out and one side
in is a key pair. A part that reads the subquery's own tables and nothing else
is a filter under the join, where it runs once rather than once per outer row.
A part that reads the outer query any other way is refused, which is the
dependent join a decorrelation pass removes rather than one this rewrite can.

An `EXISTS` that reads no outer column at all does not come here. It asks
whether a table has any row, which is the same answer for every outer row, and
that is the value form below. Which of the two an `EXISTS` is has to be decided
before its `FROM` is lowered, so it is decided off what is written: a subquery
with no equality at the top level of its `WHERE` has no key pair to give
whatever its names turn out to mean, and so it is a value.

The subquery's select list is not lowered, because `EXISTS` asks whether there
is a row rather than what is in it. DuckDB does bind it and so refuses a name in
there that no table has, and here that name goes unread. Its rows also have to
be the rows of one block over one `FROM`, so an aggregate, a `LIMIT`, a set
operation and a `WITH` inside one are none of them this join. Each of those goes
to the value form instead, which lowers the subquery as a whole statement and so
has no trouble with any of them.

### An uncorrelated `EXISTS` is a count under a cross join

Written anywhere but the `AND` of a `WHERE`, and written there without a key
pair to give, an `EXISTS` is a value: the same boolean on every row, because
whether the subquery has a row in it does not depend on which outer row asks.
So it takes the shape below, the cross join onto one row, and the row is
`count(*) > 0`. That fold has no `GROUP BY` and so is one row whatever the
subquery read, including nothing, which is what makes the shape work at all. The
comparison sits under the cross join so that it is done once rather than once
per outer row.

There is no mark join in this and there is no null in it either. An `IN` has to
be three valued because it compares values and a null compares to nothing;
`EXISTS` counts rows without looking in them, so it is true or false and a
`NOT EXISTS` is the plain opposite of it.

### The other four quantified comparisons are a range under a cross join

`x > ANY (SELECT k FROM u)` asks whether any `k` is under `x`, which is whether
the smallest one is, and `x > ALL (...)` asks whether every `k` is, which is
whether the largest one is. `<` and `<=` ask the same two questions the other
way round, and `= ALL` and `<> ANY` ask about both ends at once, since every row
equals `x` when both ends do and some row differs from `x` when either end does.
So none of the four reads the subquery more than a fold does.

The fold has four columns in it and no `GROUP BY`, so it is one row whatever the
subquery read: the smallest value, the largest, how many rows there were and how
many of those were not null. That row is cross joined on above the `FROM`, the
way the count an `EXISTS` goes through is, and the comparison itself is done
where it was written, because that is the side `x` is on.

Two of the four columns are the three valued rule and neither is spare. A null
in the subquery makes the two counts differ, and it is what turns a false into a
null: `x > ALL (S)` is not true just because `x` beat every row that was there
to beat, since the null might have been larger, and it is not false either. So
the answer is the comparison joined to a padding by the quantifier's own
operator, `OR` for `ANY` and `AND` for `ALL`, where the padding is a null when
the counts differ and is that operator's identity when they do not. Joined on by
the other operator is the filling, which is the quantifier's answer over an
empty subquery and is the identity of that operator otherwise, so it decides a
subquery with no rows in it and decides nothing anywhere else. A null `x` needs
nothing written for it, because comparing it against either end is null already.

Nothing in any of that is a `CASE`, and that is not a stylistic choice. `AND`
and `OR` over a null are the three valued operators the whole rule is made of,
so writing it with them says what it means, and a conditional wrapped around a
comparison would be saying the same thing in a way a reader has to unpick.
There is an operator for a conditional now, so this would run either way, and
it was written before there was one.

### A subquery that answers one value is a cross join onto one row

An uncorrelated subquery written where a value goes answers the same value for
every row of the query around it, so it is taken out of the expression, lowered
into a plan of its own, and cross joined on above the `FROM`. The expression is
then an ordinary expression over an ordinary column, the same way an aggregate
in a select list is. The lowering turns a cross join onto one row into a
constant per right column, so the subquery runs once rather than once per row.

Only a subquery that is one row by construction is taken, which is one that
folds with no `GROUP BY` or one with no `FROM`. In SQL both answer exactly one
row whatever is in the tables, including nothing, where a fold answers a null. A
`LIMIT 1` looks like the same guarantee and is not, since over an empty table it
answers no rows where SQL says the subquery is null, so it is refused rather
than read as one.

The cross join goes above the `FROM` and below everything else, so a subquery
written in a `WHERE` is always reachable, and one written in a select list is
reachable when the query does not aggregate. Above an aggregate it is not, since
an aggregate hands up its keys and its folds rather than everything it read, and
that is a refusal with the reason in it rather than a binding error. A
correlated one is refused too, by the scope it lowers against, which is the same
refusal a correlated `IN` gets and the same dependent join behind it.

### A `CASE` is a chain of conditionals, and the simple form is the same chain

`CASE WHEN c1 THEN r1 WHEN c2 THEN r2 ELSE e END` lowers right to left, each
arm's else side being the arm below it and the last one's being the `ELSE`. So
a chain is nested conditionals rather than a list, and the first arm that holds
is the one that answers without anything counting arms. An `ELSE` that is not
written is an `ELSE` of null, which is what the standard says, so no shape of
`CASE` is refused for want of one.

`CASE x WHEN v1 THEN r1 ... END` is the simple form and it is the searched one
with the comparison written out, so `x` is lowered once and every arm compares
against the same handle. That gives the null rule for free in both directions:
a null `x` is null against every arm and falls through to the `ELSE`, and an
arm that names a null is never matched by anything, including by a null `x`.
Checked against DuckDB both ways.

### What is not lowered yet

Named tables, the table functions above, the derived tables and the CTEs above,
and the joins over them. So are the column aliases on a derived table and a
`LATERAL` one.
`POSITIONAL` and `ASOF` are refused by name: the first pairs its two sides by
row number and so reads no column at all, and `ASOF` matches on the nearest
value rather than an equal one. A `USING` or `NATURAL` join over a subquery is
refused as well, since the merged name is on both sides and a column a subquery
computed carries no table to tell the two apart. A set
operation written `BY NAME` is refused too, since lining two arms up by column
name is a projection on each arm rather than a different node, and that needs
each arm's output names threaded back out of the block. Each is a refusal by
name rather than a silence, so `pixi run sql-support` lists them and the
conformance harness can tell a missing feature from a crash.

A decimal literal is refused too, and that one is not about effort. The engine's
`LogicalType` has no decimal and no 128 bit integer, which is the whole reason
`firepanda/sql/types.mojo` exists as a second type set. Lowering `1.1` to a
double would make `1.1 + 2.2` come back `3.3000000000000003` where DuckDB
answers exactly `3.3`, and a wrong answer with no error attached is the one
failure this front end is not allowed to have. A refusal is visible and a double
is not, so the refusal stands until the plan can carry an exact decimal.
"""

from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.unary import UnaryOp
from firepanda.array.value import Value
from firepanda.join.pairs import JoinKind
from firepanda.plan.cse import key_for
from firepanda.plan.expr import UNBOUND, ExprKind, Expressions
from firepanda.plan.node import (
    NO_LIMIT,
    SET_EXCEPT,
    SET_INTERSECT,
    SET_UNION,
    NodeKind,
    Plan,
)

from .ast import (
    Ast,
    CLAUSE_FROM,
    CLAUSE_GROUP,
    CLAUSE_HAVING,
    CLAUSE_PROJECTION,
    CLAUSE_QUALIFY,
    CLAUSE_WHERE,
    CLAUSE_WINDOW,
    EXPR_BETWEEN,
    EXPR_BINARY,
    EXPR_CASE,
    EXPR_CAST,
    EXPR_COLUMN,
    EXPR_EXISTS,
    EXPR_FUNCTION,
    EXPR_IN,
    EXPR_IN_SUBQUERY,
    EXPR_LITERAL,
    EXPR_QUANTIFIED,
    EXPR_STAR,
    EXPR_SUBQUERY,
    EXPR_UNARY,
    CALL_DISTINCT,
    CALL_STAR,
    GROUP_EXPRESSION,
    LIMIT_PERCENT,
    LITERAL_BOOLEAN,
    LITERAL_NULL,
    LITERAL_NUMBER,
    LITERAL_STRING,
    NO_NODE,
    NULLS_LAST,
    REF_FUNCTION,
    REF_JOIN,
    REF_JOIN_USING,
    REF_PARENS,
    REF_SUBQUERY,
    REF_TABLE,
    SELECT_DISTINCT,
    SORT_DESCENDING,
    STMT_QUERY,
    STMT_SELECT,
    STMT_SET_OPERATION,
    STMT_VALUES,
)
from .catalog import Catalog, KIND_FRAME, fold
from .cte import NOT_A_CTE, aliased, read_ctes
from .star import (
    NOT_REPLACED,
    Renaming,
    Replacement,
    Target,
    check,
    empty_select_list,
    not_in_from,
)
from .types import engine_type, parse_type


struct Lowered(Movable):
    """A plan, its root, and the schemas it was built against.

    The three travel together because none of them means anything without the
    others. A root is an index into these nodes, and `firepanda.plan.bind` wants
    the sources in the order the scans' table indices name them, which is the
    order this file resolved them in.
    """

    var plan: Plan
    """The nodes and the expressions they index into."""

    var root: Int
    """The node the query produces, which is the last one built."""

    var sources: List[Schema]
    """One schema per scan, in table index order."""

    def __init__(
        out self, var plan: Plan, root: Int, var sources: List[Schema]
    ):
        """Holds the three together.

        Args:
            plan: The nodes and expressions.
            root: The node the query produces.
            sources: One schema per scan.
        """
        self.plan = plan^
        self.root = root
        self.sources = sources^


def _one_name(ast: Ast, run: UInt32, what: StringSlice) raises -> String:
    """Reads a run of name parts that has to be a single part.

    A qualified name is a real thing to write and this stage resolves nothing
    against a schema, so the parts that would need resolving are refused here by
    name rather than silently joined into one string.

    Args:
        ast: The arenas.
        run: The run of interned parts.
        what: What the name names, for the message.

    Returns:
        The one part.

    Raises:
        If the run is empty or has more than one part.
    """
    var count = ast.length(run)
    if count == 0:
        raise Error(String("a ", what, " with no name"))
    if count != 1:
        var joined = String()
        for i in range(count):
            if i != 0:
                joined += "."
            joined += ast.text(ast.at(run, i))
        raise Error(
            String(
                "firepanda lowers ",
                what,
                " written as one part, and ",
                joined,
                " is ",
                count,
            )
        )
    return ast.text(ast.at(run, 0))


def _number(text: String) raises -> Value:
    """Turns the text of a number literal into a constant.

    An integer becomes an `Int64` and anything with a point or an exponent in it
    is refused, for the reason in the module docstring: the plan has no exact
    decimal and a double in its place is a wrong answer nobody is told about.

    Args:
        text: The literal as it was written, already decoded.

    Returns:
        The constant.

    Raises:
        If the number is not an integer.
    """
    for i in range(text.byte_length()):
        var c = text[byte=i]
        if c == "." or c == "e" or c == "E":
            raise Error(
                String(
                    "firepanda does not lower the decimal literal ",
                    text,
                    (
                        " yet, because a plan cannot hold an exact decimal and"
                        " a double in its place would answer 1.1 + 2.2 with"
                        " 3.3000000000000003"
                    ),
                )
            )
    return Value(Int64(atol(text)))


def _binary_op(text: String) raises -> BinaryOp:
    """The kernel operator an infix operator's text names.

    The AST keeps an operator as the words it was written with, so this is the
    one place the text becomes a code. `AND` and `OR` are not here, because the
    plan holds them as calls rather than as binary operations.

    Args:
        text: The operator as SQL spells it.

    Returns:
        The kernel operator.

    Raises:
        If firepanda has no kernel for it.
    """
    if text == "+":
        return BinaryOp.ADD
    if text == "-":
        return BinaryOp.SUB
    if text == "*":
        return BinaryOp.MUL
    if text == "/":
        return BinaryOp.DIV
    if text == "//":
        return BinaryOp.FLOORDIV
    if text == "%":
        return BinaryOp.MOD
    if text == "**" or text == "^":
        return BinaryOp.POW
    if text == "=" or text == "==":
        return BinaryOp.EQ
    if text == "<>" or text == "!=":
        return BinaryOp.NE
    if text == "<":
        return BinaryOp.LT
    if text == "<=":
        return BinaryOp.LE
    if text == ">":
        return BinaryOp.GT
    if text == ">=":
        return BinaryOp.GE
    raise Error(
        String("firepanda has no kernel for the operator ", text, " yet")
    )


def _agg_kind(name: String, distinct: Bool = False) raises -> AggKind:
    """The fold a function name is, or nothing if the name is not an aggregate.

    Only the names the engine has a fold for are here. An aggregate the registry
    knows about and the engine cannot compute is refused by the caller, which is
    the same refusal a scalar function with no kernel gets and for the same
    reason.

    `DISTINCT` inside the call changes which fold it is rather than decorating
    the one the name picks. `count(DISTINCT x)` is `NUNIQUE`, which is a
    different kernel and not a count of anything. Every other aggregate is
    refused with `DISTINCT` on it rather than folded without it, because
    ignoring the word answers a question nobody asked and answers it without
    saying so.

    Args:
        name: The function name, already folded to lower case.
        distinct: Whether the call was written `f(DISTINCT x)`.

    Returns:
        The fold.

    Raises:
        If the name is not one the engine folds, or if it is one the engine
        folds and `DISTINCT` on it has no meaning here.
    """
    if distinct:
        if name == "count":
            return AggKind.NUNIQUE
        raise Error(
            String(
                "firepanda folds DISTINCT inside count and not inside ",
                name,
                ", and it refuses rather than answering ",
                name,
                " of the values with the duplicates still in",
            )
        )
    if name == "sum":
        return AggKind.SUM
    if name == "avg" or name == "mean":
        return AggKind.MEAN
    if name == "min":
        return AggKind.MIN
    if name == "max":
        return AggKind.MAX
    if name == "count":
        return AggKind.COUNT
    if name == "first" or name == "any_value":
        return AggKind.FIRST
    if name == "last":
        return AggKind.LAST
    if name == "stddev" or name == "stddev_samp":
        return AggKind.STD
    if name == "var_samp" or name == "variance":
        return AggKind.VAR
    if name == "median":
        return AggKind.MEDIAN
    raise Error(String(name, " is not an aggregate firepanda folds"))


def _is_aggregate(name: String) -> Bool:
    """Whether a name is one of the folds above.

    Separate from `_agg_kind` because the walk has to ask before it decides how
    to descend, and asking by catching would make a refusal into control flow.

    Args:
        name: The function name, already folded to lower case.

    Returns:
        True if the engine has a fold for it.
    """
    try:
        _ = _agg_kind(name)
        return True
    except:
        return False


struct _Bindings(Copyable, Movable):
    """The CTE names in reach, in the order they were bound.

    Order is the whole visibility rule and it does two jobs at once. Entries of
    one `WITH` bind left to right and each sees the ones before it, which is why
    `WITH y AS (SELECT n FROM x), x AS (...)` is a missing table rather than a
    forward reference. And a `WITH` inside a subquery binds after the one
    outside it, so looking a name up from the end is what makes the inner one
    shadow the outer.

    Lowering a reference at position `at` lowers its statement against the first
    `at` entries, which is both rules in one line.
    """

    var keys: List[String]
    """The folded names, which is what a lookup compares."""

    var stmts: List[UInt32]
    """The statement each name stands for."""

    var columns: List[List[String]]
    """The column alias list each was written with, empty when none was."""

    def __init__(out self):
        """Nothing bound, which is what a query with no WITH in it has."""
        self.keys = List[String]()
        self.stmts = List[UInt32]()
        self.columns = List[List[String]]()

    def bind(
        mut self, var key: String, stmt: UInt32, var columns: List[String]
    ):
        """Binds one name.

        Args:
            key: The folded name.
            stmt: The statement it stands for.
            columns: Its column alias list.
        """
        self.keys.append(key^)
        self.stmts.append(stmt)
        self.columns.append(columns^)

    def find(self, name: StringSlice) -> Int:
        """Which entry a name is, latest first.

        Args:
            name: The name as the query wrote it.

        Returns:
            The entry's position, or `NOT_A_CTE`.
        """
        var key = fold(name)
        for at in range(len(self.keys) - 1, -1, -1):
            if self.keys[at] == key:
                return at
        return NOT_A_CTE

    def upto(self, before: Int) raises -> Self:
        """The entries bound before a position, which is what one of them sees.

        Args:
            before: The position asking.

        Returns:
            A copy holding the first `before` entries.
        """
        var out = Self()
        for at in range(before):
            out.bind(
                String(self.keys[at]), self.stmts[at], self.columns[at].copy()
            )
        return out^


comptime NOT_IN_REACH = -1
"""What `_Scope.find` answers for a name the `FROM` did not put in reach."""

comptime DERIVED = -2
"""What `_Scope` holds for a name that is a subquery rather than a scan.

A relation is a schema in `sources` and a number that names it, and a derived
table has neither: its columns are computed by the nodes under it. So the alias
of one is a name that says which columns are meant rather than a number a column
can carry, and this is the value that says so."""


struct _Scope(Movable):
    """What the `FROM` put in reach, and which relation each name is.

    A name in here is what the query is allowed to write in front of a column.
    A table brings its own name, and an alias replaces it rather than adding to
    it, which is SQL's rule and is why `SELECT t.a FROM t AS l` is an error and
    not a second way of writing the same thing.

    The relation is the number the scan was built with, which is the number
    `Expr.table` holds and the offset of that table's schema in `sources`. So a
    qualified name resolves here, at lowering, and the plan carries the relation
    rather than the word. A plan holding the word would be a plan holding a name
    that something later has to resolve against a scope the plan does not have.
    """

    var names: List[String]
    """The names, in the order the FROM introduced them."""

    var tables: List[Int]
    """Which relation each of those is, or `DERIVED` for a subquery."""

    var columns: List[List[String]]
    """What a derived one produces, so that a name written in front of it can be
    checked against the columns it actually has. Empty for a scan, which is
    answered by its relation instead."""

    var merged: List[String]
    """The column names a `USING` or a `NATURAL` join merged."""

    var pinned: List[Int]
    """Which relation each merged name means. A merged name is on both sides of
    the join, so a search for it finds two columns and would be refused, and
    the join is what decides which of the two the query gets."""

    def __init__(out self):
        """Starts with nothing in reach, which is what a query with no FROM has.
        """
        self.names = List[String]()
        self.tables = List[Int]()
        self.columns = List[List[String]]()
        self.merged = List[String]()
        self.pinned = List[Int]()

    def add(mut self, var name: String, table: Int) raises:
        """Puts one name in reach.

        Args:
            name: What the query may write in front of a column.
            table: Which relation it is.

        Raises:
            If something is already called that.
        """
        for i in range(len(self.names)):
            if self.names[i] == name:
                raise Error(
                    String(
                        "'",
                        name,
                        (
                            "' is the name of more than one table in this FROM,"
                            " so a column written in front of it would not say"
                            " which. Give one of them an alias"
                        ),
                    )
                )
        self.names.append(name^)
        self.tables.append(table)
        self.columns.append(List[String]())

    def derive(mut self, var name: String, var columns: List[String]) raises:
        """Puts the alias of a subquery in reach, with what it produces.

        Args:
            name: The alias.
            columns: The columns the subquery hands out, left to right.

        Raises:
            If something is already called that.
        """
        self.add(name^, DERIVED)
        self.columns[len(self.columns) - 1] = columns^

    def produces(self, name: StringSlice, column: StringSlice) -> Bool:
        """Whether a derived table hands out a column of a given name.

        Args:
            name: The alias.
            column: The column written after it.

        Returns:
            True when that alias is a derived table that produces that column.
        """
        for i in range(len(self.names)):
            if self.names[i] == name:
                for j in range(len(self.columns[i])):
                    if self.columns[i][j] == column:
                        return True
                return False
        return False

    def hide(mut self, names: Int, merged: Int):
        """Takes back everything put in reach since a point.

        A `SEMI` or an `ANTI` join keeps no column of its right side, so the
        right side is lowered and then goes out of reach, and the caller says
        where reach ended before it. Truncating rather than removing by name is
        what keeps an alias that shadowed an outer one from taking the outer one
        with it.

        Args:
            names: How many names were in reach before.
            merged: How many merged column names there were before.
        """
        while len(self.names) > names:
            _ = self.names.pop()
            _ = self.tables.pop()
            _ = self.columns.pop()
        while len(self.merged) > merged:
            _ = self.merged.pop()
            _ = self.pinned.pop()

    def merge(mut self, var name: String, table: Int):
        """Records that a join merged a column name into one column.

        A second join merging the same name replaces the first, because the
        one that decides is the join a reference is written under and that is
        the innermost one that merged it.

        Args:
            name: The column name.
            table: Which relation the merged column comes from.
        """
        for i in range(len(self.merged)):
            if self.merged[i] == name:
                self.pinned[i] = table
                return
        self.merged.append(name^)
        self.pinned.append(table)

    def merged_at(self, name: StringSlice) -> Int:
        """Which relation a merged name means.

        Args:
            name: The column name, written with nothing in front of it.

        Returns:
            The relation, or `NOT_IN_REACH` when no join merged that name.
        """
        for i in range(len(self.merged)):
            if self.merged[i] == name:
                return self.pinned[i]
        return NOT_IN_REACH

    def find(self, name: StringSlice) -> Int:
        """Which relation a name is, or minus one when nothing is called that.

        Args:
            name: What was written in front of the column.

        Returns:
            The relation, `DERIVED` for a subquery, or `NOT_IN_REACH`.
        """
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.tables[i]
        return NOT_IN_REACH

    def written(self) -> String:
        """The names in reach, for a message that has to list them.

        Returns:
            The names, comma separated, or a phrase saying there are none.
        """
        if len(self.names) == 0:
            return String("nothing")
        var out = String()
        for i in range(len(self.names)):
            if i != 0:
                out += ", "
            out += String("'", self.names[i], "'")
        return out^


struct _From(Movable):
    """What a `FROM` clause built, and what that node produces.

    The schema travels alongside the node because nothing can get it back out of
    the plan at this stage. A plan here is unbound, so the only thing that knows
    what a node produces is the code that built it, and both the star expansion
    and the next join up need to know.

    The nullability in it is the inputs' and not the join's. An outer join
    widens it and binding is what works that out, and nothing here reads it, so
    copying binding's rule into this file would be a second place to keep the
    same thing right.
    """

    var at: Int
    """The node the clause produced."""

    var schema: Schema
    """Its columns, left to right, which for a join is the two sides end to end
    the same way binding puts them."""

    var origin: List[Int]
    """Which relation each of those columns came from, which is the number a
    qualified column carries and the offset of that table in `sources`."""

    def __init__(out self, at: Int, var schema: Schema, var origin: List[Int]):
        """Holds the three together.

        Args:
            at: The node.
            schema: Its columns.
            origin: Which relation each column came from.
        """
        self.at = at
        self.schema = schema^
        self.origin = origin^

    def has(self, name: StringSlice) -> Int:
        """How many of its columns are called something.

        Args:
            name: The column name.

        Returns:
            The count, which is what tells a name one side has from a name it
            has twice and from one it does not have at all.
        """
        var out = 0
        for i in range(len(self.schema)):
            if self.schema[i].name == name:
                out += 1
        return out

    def reads(self, table: Int) -> Bool:
        """Whether one of the relations it read is a given one.

        Args:
            table: The relation.

        Returns:
            True when a column of it came from there.
        """
        for i in range(len(self.origin)):
            if self.origin[i] == table:
                return True
        return False


struct _Walk(Movable):
    """The aggregates and the windows one statement's lowering has found so far.

    The plan is not in here. A walk that owned the plan would have to hand it
    back at the end, and moving one field out of a live struct is not a thing
    Mojo allows, so the plan is passed alongside instead and this holds only
    what is genuinely walk state.
    """

    var aggs: List[Int]
    """The aggregate expressions found in the select list and the `HAVING`, in
    the order they were found, which is the order they come out of the
    `AGGREGATE` node in after the group keys."""

    var agg_names: List[String]
    """What each of those is called in the aggregate's output."""

    var windows: List[Int]
    """The window expressions found in the select list and the `QUALIFY`, in the
    order they were found."""

    var window_names: List[String]
    """What each of those is called in the window node's output."""

    var window_keys: List[String]
    """What each one partitions by, written out, so that two windows over the
    same keys land in the same node and two over different keys do not."""

    var scalars: List[UInt32]
    """The uncorrelated subqueries a cross join has already been built for, as
    they are written in the SQL arena."""

    var scalar_names: List[String]
    """What the one column each of those produced is called."""

    var marks: List[UInt32]
    """The `IN` over a subquery a mark join has already been built for, as they
    are written in the SQL arena."""

    var mark_names: List[String]
    """What the boolean column each of those produced is called."""

    var asked: List[UInt32]
    """The `EXISTS` written as a value a cross join has already been built for,
    as they are written in the SQL arena."""

    var asked_names: List[String]
    """What the boolean column each of those produced is called."""

    var compared: List[UInt32]
    """The quantified comparisons a cross join has already been built for, as
    they are written in the SQL arena. What each one's four columns are called
    is read off its position here rather than kept beside it."""

    def __init__(out self):
        """Starts an empty walk."""
        self.aggs = List[Int]()
        self.agg_names = List[String]()
        self.windows = List[Int]()
        self.window_names = List[String]()
        self.window_keys = List[String]()
        self.scalars = List[UInt32]()
        self.scalar_names = List[String]()
        self.marks = List[UInt32]()
        self.mark_names = List[String]()
        self.asked = List[UInt32]()
        self.asked_names = List[String]()
        self.compared = List[UInt32]()

    def _scalar(self, at: UInt32) -> Int:
        """Where a subquery's answer landed, if a cross join was built for it.

        Args:
            at: The subquery, in the SQL arena.

        Returns:
            Its position among the ones taken out, or -1 if this is not one of
            them and so is a subquery the lowering still refuses.
        """
        for i in range(len(self.scalars)):
            if self.scalars[i] == at:
                return i
        return -1

    def _mark(self, at: UInt32) -> Int:
        """Where an `IN`'s answer landed, if a mark join was built for it.

        Args:
            at: The `IN` over a subquery, in the SQL arena.

        Returns:
            Its position among the ones taken out, or -1 if this is not one of
            them and so is an `IN` the lowering still refuses.
        """
        for i in range(len(self.marks)):
            if self.marks[i] == at:
                return i
        return -1

    def _asked(self, at: UInt32) -> Int:
        """Where an `EXISTS`'s answer landed, if a cross join was built for it.

        Args:
            at: The `EXISTS`, in the SQL arena.

        Returns:
            Its position among the ones taken out, or -1 if this is not one of
            them and so is an `EXISTS` the lowering still refuses.
        """
        for i in range(len(self.asked)):
            if self.asked[i] == at:
                return i
        return -1

    def _compared(self, at: UInt32) -> Int:
        """Where a quantified comparison's row landed, if one was built for it.

        Args:
            at: The comparison, in the SQL arena.

        Returns:
            Its position among the ones taken out, which is also the number in
            the names of its four columns, or -1 if this is not one of them and
            so is a comparison the lowering still refuses.
        """
        for i in range(len(self.compared)):
            if self.compared[i] == at:
                return i
        return -1

    def _record(mut self, at: Int, var name: String) -> Int:
        """Adds an aggregate to the list the `AGGREGATE` node will compute.

        Args:
            at: The lowered aggregate expression.
            name: What to call its output column.

        Returns:
            Its position among the aggregates.
        """
        var place = len(self.aggs)
        self.aggs.append(at)
        self.agg_names.append(name^)
        return place

    def _record_window(
        mut self, at: Int, var name: String, var key: String
    ) -> Int:
        """Adds a window to the list a `WINDOW` node will compute.

        Args:
            at: The lowered window expression.
            name: What to call its output column.
            key: What it partitions by, written out.

        Returns:
            Its position among the windows.
        """
        var place = len(self.windows)
        self.windows.append(at)
        self.window_names.append(name^)
        self.window_keys.append(key^)
        return place


def _lower_expr(
    ast: Ast,
    at: UInt32,
    mut plan: Plan,
    mut walk: _Walk,
    scope: _Scope,
    grouped: Bool,
) raises -> Int:
    """Lowers one SQL expression into the plan's expression arena.

    An aggregate call is not lowered in place. It is lowered, handed to
    `_record`, and replaced by a reference to the column the `AGGREGATE` node
    will put it in, which is what lets the projection above the aggregate be an
    ordinary elementwise expression over ordinary columns.

    Args:
        ast: The arenas the SQL expression lives in.
        at: The expression.
        plan: Where the lowered expressions go.
        walk: The aggregates found so far.
        scope: What the FROM put in reach, for a qualified name.
        grouped: Whether the query aggregates, which decides whether an
            aggregate call is allowed here at all.

    Returns:
        The index in the plan's expression arena.

    Raises:
        If the expression is a shape this does not lower yet.
    """
    if at == NO_NODE:
        raise Error("an expression that is not there")
    var node = ast.exprs[Int(at)]

    if node.kind == EXPR_LITERAL:
        var tag = node.b
        if tag == LITERAL_NULL:
            return plan.exprs.literal(Value(null=LogicalType.NULL))
        var text = ast.text(node.payload)
        if tag == LITERAL_BOOLEAN:
            return plan.exprs.literal(Value(text == "true"))
        if tag == LITERAL_NUMBER:
            return plan.exprs.literal(_number(text))
        if tag == LITERAL_STRING:
            return plan.exprs.literal(Value(text))
        raise Error("a literal whose kind this does not lower yet")

    if node.kind == EXPR_COLUMN:
        var parts = ast.length(node.children)
        if parts == 0:
            raise Error("a column with no name")
        if parts == 1:
            var bare = String(ast.text(ast.at(node.children, 0)))
            # A name a USING or a NATURAL join merged is on both sides of that
            # join, so a search would find two columns and refuse. The join
            # already decided which of the two it hands out, and this is where
            # that decision is written into the expression.
            var whose = scope.merged_at(bare)
            if whose != NOT_IN_REACH:
                return plan.exprs.column_of(whose, bare^)
            return plan.exprs.column(bare^)
        if parts == 2:
            var qualifier = ast.text(ast.at(node.children, 0))
            var found = scope.find(qualifier)
            if found == NOT_IN_REACH:
                raise Error(
                    String(
                        "nothing in this query is called '",
                        qualifier,
                        "', and the FROM brought ",
                        scope.written(),
                    )
                )
            var column = String(ast.text(ast.at(node.children, 1)))
            if found == DERIVED:
                if not scope.produces(qualifier, column):
                    raise Error(
                        String(
                            "'",
                            qualifier,
                            "' produces no column called '",
                            column,
                            "'",
                        )
                    )
                # There is no relation number for the alias of a subquery to
                # lower to, so the name goes on unqualified. Checking it
                # against what the subquery hands out is the work the qualifier
                # does, and a name that two sources in the same FROM both have
                # then comes back from binding as an ambiguity rather than as
                # the wrong column.
                return plan.exprs.column(column^)
            return plan.exprs.column_of(found, column^)
        raise Error(
            "firepanda reads a column as a name or as a table and a name so"
            " far, and a third part is either a schema or a struct field and"
            " telling those two apart needs the catalog"
        )

    if node.kind == EXPR_UNARY:
        var op = ast.text(node.payload)
        var over = _lower_expr(ast, node.a, plan, walk, scope, grouped)
        if op == "-":
            return plan.exprs.unary(UnaryOp.NEG, over)
        if op == "+":
            return plan.exprs.unary(UnaryOp.POS, over)
        if op == "NOT":
            return plan.exprs.call("not", [over], True)
        raise Error(
            String("firepanda has no kernel for the prefix operator ", op)
        )

    if node.kind == EXPR_BINARY:
        var op = ast.text(node.payload)
        var left = _lower_expr(ast, node.a, plan, walk, scope, grouped)
        var right = _lower_expr(ast, node.b, plan, walk, scope, grouped)
        if op == "AND":
            return plan.exprs.call("and", [left, right], True)
        if op == "OR":
            return plan.exprs.call("or", [left, right], True)
        return plan.exprs.binary(_binary_op(op), left, right)

    if node.kind == EXPR_CASE:
        var arms = ast.items(node.children)
        if len(arms) == 0 or len(arms) % 2 != 0:
            raise Error(
                String(
                    "a CASE is a run of WHEN and THEN pairs and this one has ",
                    len(arms),
                    " halves",
                )
            )
        # An ELSE that is not written is an ELSE of null, which is what the
        # standard says and is why nothing here refuses the shape.
        var built: Int
        if node.b == NO_NODE:
            built = plan.exprs.literal(Value(null=LogicalType.NULL))
        else:
            built = _lower_expr(ast, node.b, plan, walk, scope, grouped)
        # Right to left, because each arm's else side is the arm below it and
        # the last one's is the ELSE. A chain is nested conditionals rather
        # than a list, so the first WHEN that holds is the one that answers
        # without anything counting arms.
        # `CASE x WHEN v` is the simple form, and it is the searched one with
        # the comparison written out: each arm asks whether `x` equals what the
        # arm names. `x` is lowered once and the arms share the handle, so it
        # is one column however many arms there are, and an arm that asks about
        # a null answers null and so takes the arm below it, which is what the
        # standard says the simple form does.
        var simple = node.a != NO_NODE
        var subject = 0
        if simple:
            subject = _lower_expr(ast, node.a, plan, walk, scope, grouped)
        var i = len(arms) - 2
        while i >= 0:
            var when = _lower_expr(ast, arms[i], plan, walk, scope, grouped)
            if simple:
                when = plan.exprs.binary(BinaryOp.EQ, subject, when)
            var then = _lower_expr(ast, arms[i + 1], plan, walk, scope, grouped)
            built = plan.exprs.conditional(when, then, built)
            i -= 2
        return built

    if node.kind == EXPR_FUNCTION:
        var name = fold(_one_name(ast, node.payload, "a function"))
        if node.b != NO_NODE:
            return _lower_over(ast, at, name, plan, walk, scope, grouped)
        var args = ast.items(node.children)
        if _is_aggregate(name):
            if not grouped:
                raise Error(
                    String(
                        name,
                        (
                            " is an aggregate and this query has no GROUP BY"
                            " and no other aggregate in it"
                        ),
                    )
                )
            var distinct = (node.a & CALL_DISTINCT) != 0
            var over: Int
            if name == "count" and (node.a & CALL_STAR) != 0:
                if distinct:
                    raise Error(
                        "count(DISTINCT *) has no column to count the distinct"
                        " values of, and counting distinct whole rows is"
                        " SELECT DISTINCT with a count around it"
                    )
                over = plan.exprs.literal(Value(Int64(1)))
            elif len(args) == 1:
                over = _lower_expr(ast, args[0], plan, walk, scope, grouped)
            else:
                raise Error(
                    String(
                        "firepanda folds ",
                        name,
                        " over one argument, and this call has ",
                        len(args),
                    )
                )
            var built = plan.exprs.aggregate(_agg_kind(name, distinct), over)
            var place = walk._record(built, String("__agg_", len(walk.aggs)))
            return plan.exprs.column(String(walk.agg_names[place]))
        var lowered = List[Int]()
        for i in range(len(args)):
            lowered.append(
                _lower_expr(ast, args[i], plan, walk, scope, grouped)
            )
        return plan.exprs.call(name, lowered^, True)

    if node.kind == EXPR_BETWEEN:
        return _lower_between(ast, at, plan, walk, scope, grouped)

    if node.kind == EXPR_IN:
        return _lower_in(ast, at, plan, walk, scope, grouped)

    if node.kind == EXPR_CAST:
        if node.b == 1:
            raise Error(
                "firepanda does not lower a TRY_CAST yet, because the plan's"
                " cast has no way to say that a value it cannot convert is a"
                " null rather than an error"
            )
        var written = ast.text(node.payload)
        var over = _lower_expr(ast, node.a, plan, walk, scope, grouped)
        return plan.exprs.cast(engine_type(parse_type(written)), over)
    if node.kind == EXPR_STAR:
        raise Error("a star outside a select list")
    if node.kind == EXPR_SUBQUERY:
        var place = walk._scalar(at)
        if place >= 0:
            return plan.exprs.column(String(walk.scalar_names[place]))
        raise Error(
            "firepanda lowers an uncorrelated subquery that answers one value"
            " where it is written in a WHERE, or in the select list of a query"
            " that does not aggregate, and this one is written somewhere else."
            " The answer is a column cross joined on above the FROM, which is"
            " under the aggregate, and an aggregate hands up its keys and its"
            " folds rather than everything it read"
        )
    if node.kind == EXPR_IN_SUBQUERY or node.kind == EXPR_QUANTIFIED:
        var shape = _in_shape(ast, at)
        if shape < 0:
            var row = walk._compared(at)
            if row >= 0:
                return _quantified_read(
                    ast, at, plan, walk, scope, grouped, row
                )
            raise Error(
                "firepanda lowers a quantified comparison where it is written"
                " in a WHERE, or in the select list of a query that does not"
                " aggregate, and this one is written somewhere else. The answer"
                " is read off a row cross joined on above the FROM, which is"
                " under the aggregate, and an aggregate hands up its keys and"
                " its folds rather than everything it read"
            )
        var place = walk._mark(at)
        if place >= 0:
            # The mark join answered the `IN` itself, so what is left here is
            # reading the column it wrote. A `NOT IN` is the same column with a
            # `NOT` over it, and that is the whole of the three valued rule: a
            # row that matched nothing on a side holding a null is marked null
            # rather than false, a `NOT` over a null is a null, and a filter
            # does not keep a null.
            var over = plan.exprs.column(String(walk.mark_names[place]))
            if shape == 1:
                return plan.exprs.call("not", [over], True)
            return over
        raise Error(
            "firepanda lowers an IN over a subquery where it is written in a"
            " WHERE, or in the select list of a query that does not aggregate,"
            " and this one is written somewhere else. The answer is a column a"
            " mark join wrote above the FROM, which is under the aggregate, and"
            " an aggregate hands up its keys and its folds rather than"
            " everything it read"
        )
    if node.kind == EXPR_EXISTS:
        var place = walk._asked(at)
        if place >= 0:
            # The counting under the cross join answered it, so what is left
            # here is reading the column. There is no null to worry about the
            # way there is with an `IN`, because a count of rows is a number
            # whatever is in them and `EXISTS` is true or false and never null.
            return plan.exprs.column(String(walk.asked_names[place]))
        raise Error(
            "firepanda lowers an EXISTS written as a value where it is written"
            " in a WHERE, or in the select list of a query that does not"
            " aggregate, and this one is written somewhere else. The answer is"
            " a column cross joined on above the FROM, which is under the"
            " aggregate, and an aggregate hands up its keys and its folds"
            " rather than everything it read"
        )
    raise Error("an expression shape firepanda does not lower yet")


def _lower_between(
    ast: Ast,
    at: UInt32,
    mut plan: Plan,
    mut walk: _Walk,
    scope: _Scope,
    grouped: Bool,
) raises -> Int:
    """Lowers `x BETWEEN lo AND hi` into the two comparisons it stands for.

    The plan has no range test of its own and does not want one. A predicate
    shape is something every pass over a filter has to know about, and a range
    is already sayable, so adding it would buy nothing and cost pushdown, the
    optimizer and the printer a case each.

    The operand is lowered once and both comparisons point at what came back, so
    the arena holds one subtree with two parents rather than two copies of it.
    That is the same thing common subexpression elimination arrives at, and it
    is the reason the AST keeps BETWEEN whole: rewriting it in the parser would
    have written the operand out twice.

    `NOT BETWEEN` is the whole test negated rather than the two comparisons
    turned around. With `hi` null and `x` below `lo`, the positive test is false
    whatever the null bound would have said, so the negation is true, while
    `x < lo OR x > hi` answers null. Negating leaves the three valued logic in
    the one place that already implements it.

    Args:
        ast: The arenas.
        at: The `EXPR_BETWEEN`.
        plan: Where the lowered expressions go.
        walk: What the aggregates and windows found so far are recorded in.
        scope: What the FROM put in reach.
        grouped: Whether the block aggregates.

    Returns:
        The predicate.

    Raises:
        If the bounds are not a pair, or a part of it does not lower.
    """
    var node = ast.exprs[Int(at)]
    var bounds = ast.items(node.children)
    if len(bounds) != 2:
        raise Error(
            String(
                (
                    "a BETWEEN wants a low bound and a high one, and this one"
                    " was given "
                ),
                len(bounds),
            )
        )
    var over = _lower_expr(ast, node.a, plan, walk, scope, grouped)
    var low = _lower_expr(ast, bounds[0], plan, walk, scope, grouped)
    var high = _lower_expr(ast, bounds[1], plan, walk, scope, grouped)
    var within = plan.exprs.call(
        "and",
        [
            plan.exprs.binary(BinaryOp.GE, over, low),
            plan.exprs.binary(BinaryOp.LE, over, high),
        ],
        True,
    )
    if node.payload == 1:
        return plan.exprs.call("not", [within], True)
    return within


def _lower_in(
    ast: Ast,
    at: UInt32,
    mut plan: Plan,
    mut walk: _Walk,
    scope: _Scope,
    grouped: Bool,
) raises -> Int:
    """Lowers `x IN (a, b, c)` into the equalities it stands for.

    One equality per candidate, joined by `OR`, with the operand shared the way
    a BETWEEN shares it. A list of three is three comparisons, which is what
    DuckDB does with a short list as well. A long list wants a hash set instead,
    and that is a physical choice rather than a different meaning, so it belongs
    in the operator and not here.

    `NOT IN` is the negation of the whole test, which is what makes it the
    classic wrong answer when it is written any other way. `x NOT IN (1, NULL)`
    with `x` two is null rather than true, because the positive test cannot rule
    the null out, and the chain of `OR` answers exactly that before the negation
    turns it into a null as well. Written as a chain of `<>` joined by `AND` it
    would answer true, which is the defect this shape avoids by not existing.

    Args:
        ast: The arenas.
        at: The `EXPR_IN`.
        plan: Where the lowered expressions go.
        walk: What the aggregates and windows found so far are recorded in.
        scope: What the FROM put in reach.
        grouped: Whether the block aggregates.

    Returns:
        The predicate.

    Raises:
        If the list is empty, or a part of it does not lower.
    """
    var node = ast.exprs[Int(at)]
    var candidates = ast.items(node.children)
    if len(candidates) == 0:
        raise Error("an IN with nothing in the list to be in")
    var over = _lower_expr(ast, node.a, plan, walk, scope, grouped)
    var built = plan.exprs.binary(
        BinaryOp.EQ,
        over,
        _lower_expr(ast, candidates[0], plan, walk, scope, grouped),
    )
    for i in range(1, len(candidates)):
        var more = plan.exprs.binary(
            BinaryOp.EQ,
            over,
            _lower_expr(ast, candidates[i], plan, walk, scope, grouped),
        )
        built = plan.exprs.call("or", [built, more], True)
    if node.payload == 1:
        return plan.exprs.call("not", [built], True)
    return built


def _lower_over(
    ast: Ast,
    at: UInt32,
    name: String,
    mut plan: Plan,
    mut walk: _Walk,
    scope: _Scope,
    grouped: Bool,
) raises -> Int:
    """Lowers a call that has an `OVER` on it into a window the plan computes.

    The same arrangement the aggregates have. The window is put on the walk and
    what comes back in its place is a reference to the column the `WINDOW` node
    will produce, so the expression the window sits inside stays whatever shape
    it was and the node that computes it is built afterwards, once the walk has
    found all of them.

    What it partitions by is written out and kept beside it, because two windows
    over the same keys are one grouping pass and belong in one node, and two
    over different keys are two nodes. Written out rather than compared by arena
    index, since `PARTITION BY k` lowered twice is two indices for one key.

    Args:
        ast: The arenas.
        at: The `EXPR_FUNCTION` that has the `OVER` on it.
        name: Its function name, already folded.
        plan: Where the lowered expressions go.
        walk: Where the window is recorded.
        scope: What the FROM put in reach.
        grouped: Whether the block aggregates.

    Returns:
        A reference to the column the window node will produce.

    Raises:
        If the window is a shape this does not lower yet, or if the function is
        not one the engine folds.
    """
    var node = ast.exprs[Int(at)]
    var window = ast.exprs[Int(node.b)]
    if window.payload != 0:
        raise Error(
            String(
                (
                    "firepanda lowers a window written out in full so far, and"
                    " this one is written as OVER "
                ),
                ast.text(window.payload),
            )
        )
    if ast.length(window.a) != 0:
        raise Error(
            String(
                name,
                (
                    " is computed OVER an ORDER BY, which is a running fold"
                    " over the partition rather than one value across it, and"
                    " firepanda has no operator for that yet"
                ),
            )
        )
    if window.b != NO_NODE:
        raise Error(
            String(
                name,
                (
                    " is computed OVER a frame, and the only frame firepanda"
                    " has is the whole partition"
                ),
            )
        )
    if not _is_aggregate(name):
        raise Error(
            String(
                "firepanda computes a fold OVER a partition, and ",
                name,
                (
                    " is a window function of its own rather than a fold, so"
                    " there is nothing for it to reduce"
                ),
            )
        )

    var args = ast.items(node.children)
    var distinct = (node.a & CALL_DISTINCT) != 0
    var over: Int
    if name == "count" and (node.a & CALL_STAR) != 0:
        if distinct:
            raise Error(
                "count(DISTINCT *) has no column to count the distinct values"
                " of, and counting distinct whole rows is SELECT DISTINCT with"
                " a count around it"
            )
        over = plan.exprs.literal(Value(Int64(1)))
    elif len(args) == 1:
        over = _lower_expr(ast, args[0], plan, walk, scope, grouped)
    else:
        raise Error(
            String(
                "firepanda folds ",
                name,
                " over one argument, and this call has ",
                len(args),
            )
        )

    var partition = List[Int]()
    var key = String()
    for entry in ast.items(window.children):
        partition.append(_lower_expr(ast, entry, plan, walk, scope, grouped))
        key += String(_shape(plan.exprs, partition[len(partition) - 1]), ";")

    var built = plan.exprs.window(
        _agg_kind(name, distinct), over, partition^, List[Int]()
    )
    var place = walk._record_window(
        built, String("__win_", len(walk.windows)), key^
    )
    return plan.exprs.column(String(walk.window_names[place]))


def _shape(exprs: Expressions, root: Int) raises -> String:
    """Writes an expression tree out as the thing it computes.

    Two trees that compute the same thing write the same string, which is what
    decides whether two windows partition the same way. `cse.key_for` is the
    one node of it and this is the recursion, the same way subplan elimination
    does it.

    Args:
        exprs: The arena.
        root: The expression.

    Returns:
        The key.

    Raises:
        If the expression is not in the arena.
    """
    exprs.check(root)
    var kids = List[String](capacity=len(exprs.nodes[root].children))
    for i in range(len(exprs.nodes[root].children)):
        kids.append(_shape(exprs, exprs.nodes[root].children[i]))
    return key_for(exprs, root, kids^)


def _has_aggregate(ast: Ast, at: UInt32) -> Bool:
    """Whether an expression holds an aggregate call anywhere in it.

    The question the select list asks before anything is lowered, because
    whether the query aggregates decides what node goes under the projection and
    that has to be known before the projection is built.

    The walk does not descend into a subquery, since an aggregate written inside
    one belongs to that query. It does not need to check here, because a
    subquery is refused by the lowering anyway, and the check is written down so
    that the day it is not refused this does not quietly become wrong.

    A call with `OVER` on it is not a fold for this purpose, whatever its name
    is. `sum(x) OVER ()` reads every row and answers every row, so a query that
    has one and nothing else does not aggregate, and reading it as one would put
    an `AGGREGATE` node under the projection and lose every row but one. What is
    written inside the window still counts, since `sum(sum(a)) OVER ()` does
    aggregate and so does a partition key that folds.

    Args:
        ast: The arenas.
        at: The expression.

    Returns:
        True if a fold is written anywhere in it.
    """
    if at == NO_NODE:
        return False
    var node = ast.exprs[Int(at)]
    if node.kind == EXPR_FUNCTION:
        if node.b == NO_NODE:
            try:
                if _is_aggregate(
                    fold(_one_name(ast, node.payload, "a function"))
                ):
                    return True
            except:
                pass
        else:
            var window = ast.exprs[Int(node.b)]
            for key in ast.items(window.children):
                if _has_aggregate(ast, key):
                    return True
        for arg in ast.items(node.children):
            if _has_aggregate(ast, arg):
                return True
        return False
    if node.kind == EXPR_BINARY:
        return _has_aggregate(ast, node.a) or _has_aggregate(ast, node.b)
    if node.kind == EXPR_UNARY or node.kind == EXPR_CAST:
        return _has_aggregate(ast, node.a)
    if node.kind == EXPR_CASE:
        if _has_aggregate(ast, node.a) or _has_aggregate(ast, node.b):
            return True
        for arm in ast.items(node.children):
            if _has_aggregate(ast, arm):
                return True
        return False
    if node.kind == EXPR_BETWEEN or node.kind == EXPR_IN:
        for part in ast.items(node.children):
            if _has_aggregate(ast, part):
                return True
        return _has_aggregate(ast, node.a)
    return False


def _taken(ast: Ast, at: UInt32, kind: UInt8, mut found: List[UInt32]) raises:
    """Collects every subquery of one shape inside one expression.

    The same walk `_has_aggregate` does and for the same reason. A subquery in
    an expression is not lowered in place, it is lowered into a plan of its own
    and joined on below, and that has to happen before the expression around it
    is lowered at all.

    It does not descend into a subquery it has found, because whatever is
    written inside one belongs to that query and is lowered when that query is.

    Args:
        ast: The arenas.
        at: The expression.
        kind: The expression kind to collect, which is `EXPR_SUBQUERY` for the
            ones a cross join onto one value answers, `EXPR_IN_SUBQUERY` for the
            ones a mark join does, and `EXPR_EXISTS` for the ones a count under
            a cross join does. Asking for the mark join's kind also collects the
            two quantified comparisons that are a membership test, since those
            are an `IN` and go to the same join.
        found: The list to add to, in the order the subqueries are written.

    Raises:
        If the text of a quantified comparison is not there to read.
    """
    if at == NO_NODE:
        return
    var node = ast.exprs[Int(at)]
    if node.kind == kind:
        # A quantified comparison that is a membership test is an `IN`, so it is
        # collected when the mark join's kind is asked for and not when the
        # quantified kind is, and the two pre-passes do not both take it.
        if kind != EXPR_QUANTIFIED or _in_shape(ast, at) < 0:
            found.append(at)
        return
    if (
        kind == EXPR_IN_SUBQUERY
        and node.kind == EXPR_QUANTIFIED
        and _in_shape(ast, at) >= 0
    ):
        found.append(at)
        return
    if (
        node.kind == EXPR_SUBQUERY
        or node.kind == EXPR_IN_SUBQUERY
        or node.kind == EXPR_EXISTS
        or node.kind == EXPR_QUANTIFIED
    ):
        # A subquery of the other shape is still a subquery, and what is written
        # inside one is lowered when that query is rather than out here.
        return
    if node.kind == EXPR_FUNCTION:
        if node.b != NO_NODE:
            var window = ast.exprs[Int(node.b)]
            for key in ast.items(window.children):
                _taken(ast, key, kind, found)
        for arg in ast.items(node.children):
            _taken(ast, arg, kind, found)
        return
    if node.kind == EXPR_BINARY:
        _taken(ast, node.a, kind, found)
        _taken(ast, node.b, kind, found)
        return
    if node.kind == EXPR_UNARY or node.kind == EXPR_CAST:
        _taken(ast, node.a, kind, found)
        return
    if node.kind == EXPR_CASE:
        _taken(ast, node.a, kind, found)
        _taken(ast, node.b, kind, found)
        for arm in ast.items(node.children):
            _taken(ast, arm, kind, found)
        return
    if node.kind == EXPR_BETWEEN or node.kind == EXPR_IN:
        for part in ast.items(node.children):
            _taken(ast, part, kind, found)
        _taken(ast, node.a, kind, found)


def _scalars(ast: Ast, at: UInt32, mut found: List[UInt32]) raises:
    """Collects every subquery written as a value inside one expression.

    An `EXISTS`, an `IN` and a quantified comparison are not collected. Each of
    those answers a boolean per row rather than one value, which is a mark join
    rather than a cross join onto one row.

    Args:
        ast: The arenas.
        at: The expression.
        found: The list to add to, in the order the subqueries are written.

    Raises:
        If the text of a quantified comparison is not there to read.
    """
    _taken(ast, at, EXPR_SUBQUERY, found)


def _marks(ast: Ast, at: UInt32, mut found: List[UInt32]) raises:
    """Collects every `IN` over a subquery inside one expression.

    A `= ANY` and a `<> ALL` are collected with them, because each is a
    membership test written the other way round and goes to the same join. The
    other four quantified comparisons are not, and neither is an `EXISTS`, even
    though all of those answer a boolean per row the same way. A mark join is
    given a pair of keys and none of those is written with one.

    Args:
        ast: The arenas.
        at: The expression.
        found: The list to add to, in the order the subqueries are written.

    Raises:
        If the text of a quantified comparison is not there to read.
    """
    _taken(ast, at, EXPR_IN_SUBQUERY, found)


def _askings(ast: Ast, at: UInt32, mut found: List[UInt32]) raises:
    """Collects every `EXISTS` written as a value inside one expression.

    A quantified comparison is not collected. It answers the same boolean per
    row, and what it is asking is whether a comparison holds against some or
    every row rather than whether there was a row at all, so counting the rows
    is not the answer to it.

    Args:
        ast: The arenas.
        at: The expression.
        found: The list to add to, in the order the subqueries are written.

    Raises:
        If the text of a quantified comparison is not there to read.
    """
    _taken(ast, at, EXPR_EXISTS, found)


def _comparisons(ast: Ast, at: UInt32, mut found: List[UInt32]) raises:
    """Collects every quantified comparison that is not a membership test.

    `= ANY` and `<> ALL` are not collected. Each is an `IN` and goes to the mark
    join, which has a key pair to join on and so needs none of the counting the
    other four go through.

    Args:
        ast: The arenas.
        at: The expression.
        found: The list to add to, in the order the comparisons are written.

    Raises:
        If the text of a quantified comparison is not there to read.
    """
    _taken(ast, at, EXPR_QUANTIFIED, found)


comptime _NEITHER = -2
"""An expression that reads no column of either side of a join."""

comptime _BOTH = -1
"""An expression that reads a column of both sides of a join."""

comptime _LEFT = 0
"""An expression that reads the left side of a join and only that."""

comptime _RIGHT = 1
"""An expression that reads the right side of a join and only that."""


def _join_kind(text: StringSlice) raises -> JoinKind:
    """Reads the words in front of `JOIN` as one of the kinds the plan has.

    The text is what the query wrote, so it is matched a word at a time rather
    than as a whole. `LEFT JOIN` and `LEFT OUTER JOIN` are the same join and
    `OUTER` says nothing that `LEFT` did not.

    Args:
        text: The join as SQL spells it, such as `LEFT OUTER JOIN`.

    Returns:
        Which rows the join keeps.

    Raises:
        If it is a join this does not lower yet.
    """
    var said = String(text).upper()
    if said.find("ASOF") != -1:
        raise Error(
            "firepanda does not lower an ASOF join yet, which matches a row"
            " against the nearest value rather than an equal one and so is a"
            " different operator and not a different condition"
        )
    if said.find("POSITIONAL") != -1:
        raise Error(
            "firepanda does not lower a POSITIONAL join yet, which pairs the"
            " two sides by row number and so reads no column at all"
        )
    # Neither takes a LEFT or a RIGHT in front of it, in DuckDB or here, so
    # both are read before the words that do and there is nothing to order
    # these two against each other.
    if said.find("SEMI") != -1:
        return JoinKind.SEMI
    if said.find("ANTI") != -1:
        return JoinKind.ANTI
    if said.find("CROSS") != -1:
        return JoinKind.CROSS
    if said.find("FULL") != -1:
        return JoinKind.OUTER
    if said.find("LEFT") != -1:
        return JoinKind.LEFT
    if said.find("RIGHT") != -1:
        return JoinKind.RIGHT
    return JoinKind.INNER


def _conjuncts(ast: Ast, at: UInt32, mut out: List[UInt32]) raises:
    """Splits a condition into the parts it is the `AND` of.

    Done on the AST rather than on a lowered expression, because the point of
    the split is to lower the two sides of an equality apart from each other and
    by the time there is a lowered `and` call to take to pieces they have
    already been lowered together.

    Args:
        ast: The arenas.
        at: The condition.
        out: Where the parts go, in the order they were written.
    """
    var node = ast.exprs[Int(at)]
    if node.kind == EXPR_BINARY and ast.text(node.payload) == "AND":
        _conjuncts(ast, node.a, out)
        _conjuncts(ast, node.b, out)
        return
    out.append(at)


def _side(plan: Plan, at: Int, left: _From, right: _From) raises -> Int:
    """Which side of a join an expression reads, when it is only one of them.

    This is what turns a condition into key pairs. `plan.join` holds a left key
    and a right key per pair rather than a predicate, so an equality is only a
    key pair if one of its sides reads the left input and the other reads the
    right, and working that out means asking each column where it came from.

    A column that says its relation is answered by that, and one that does not
    is answered by the name, which is the same order of preference the rest of
    this file uses. The names are asked of the schemas the two sides produce
    rather than of the scope, because the scope says what a table is called and
    this question is about what a node produces.

    Args:
        plan: The plan holding the expression.
        at: The expression.
        left: The left input.
        right: The right input.

    Returns:
        `_LEFT`, `_RIGHT`, `_NEITHER` for an expression that reads no column at
        all, or `_BOTH` for one that reads columns of each.

    Raises:
        If a column names a table this join does not read, or a name that both
        sides have with nothing saying which.
    """
    ref node = plan.exprs.nodes[at]
    if node.kind == ExprKind.COLUMN:
        if node.table != UNBOUND:
            if left.reads(node.table):
                return _LEFT
            if right.reads(node.table):
                return _RIGHT
            raise Error(
                String(
                    "'",
                    node.name,
                    (
                        "' is written in front of a table this join does not"
                        " read, and a join condition reaches the two tables it"
                        " joins and nothing else"
                    ),
                )
            )
        var on_left = left.has(node.name)
        var on_right = right.has(node.name)
        if on_left == 0 and on_right == 0:
            raise Error(
                String(
                    "there is no column named '",
                    node.name,
                    "' on either side of this join",
                )
            )
        if on_left != 0 and on_right != 0:
            raise Error(
                String(
                    "'",
                    node.name,
                    (
                        "' is the name of a column on both sides of this join,"
                        " so say which table it is from"
                    ),
                )
            )
        if on_left > 1 or on_right > 1:
            raise Error(
                String(
                    "'",
                    node.name,
                    (
                        "' is the name of more than one column on one side of"
                        " this join"
                    ),
                )
            )
        if on_left != 0:
            return _LEFT
        return _RIGHT

    var seen = _NEITHER
    for i in range(len(node.children)):
        var here = _side(plan, node.children[i], left, right)
        if here == _NEITHER:
            continue
        if here == _BOTH:
            return _BOTH
        if seen != _NEITHER and seen != here:
            return _BOTH
        seen = here
    return seen


def _pair(
    mut plan: Plan,
    left: _From,
    right: _From,
    var left_keys: List[Int],
    var right_keys: List[Int],
    kind: JoinKind,
) raises -> _From:
    """Puts one join over two inputs and says what the result produces.

    Args:
        plan: Where the node goes.
        left: The left input.
        right: The right input.
        left_keys: The keys on the left.
        right_keys: The keys on the right, one per left key.
        kind: Which rows the join keeps.

    Returns:
        The join, with the two schemas end to end, which is the order binding
        puts them in and therefore the order the positions above will mean. A
        `SEMI` or an `ANTI` join keeps none of the right side, so there it is
        the left schema alone and binding agrees.

    Raises:
        Whatever `plan.join` raises.
    """
    var at = plan.join(left.at, right.at, left_keys^, right_keys^, kind)
    var schema = Schema(copy=left.schema)
    var origin = left.origin.copy()
    if kind.keeps_right_columns():
        for i in range(len(right.schema)):
            schema.append(right.schema[i].copy())
            origin.append(right.origin[i])
    return _From(at, schema^, origin^)


def _only(side: _From, name: StringSlice, word: StringSlice) raises -> Int:
    """Where a side's one column of a given name is.

    Args:
        side: The input.
        name: The column name the join named.
        word: `USING` or `NATURAL`, for the message.

    Returns:
        Its position in that side's columns.

    Raises:
        If the side has no column of that name, or more than one.
    """
    var found = -1
    for i in range(len(side.schema)):
        if side.schema[i].name == name:
            if found != -1:
                raise Error(
                    String(
                        "a ",
                        word,
                        " join names '",
                        name,
                        (
                            "', and one side of it has two columns called that,"
                            " so there is no one column for the pair to be"
                        ),
                    )
                )
            found = i
    if found == -1:
        raise Error(
            String(
                "a ",
                word,
                " join names '",
                name,
                "', and one side of it has no column called that",
            )
        )
    return found


def _merged(
    mut plan: Plan,
    left: _From,
    right: _From,
    names: List[String],
    kind: JoinKind,
    mut scope: _Scope,
    word: StringSlice,
) raises -> _From:
    """Lowers a join that names its keys by the columns the two sides share.

    `USING` and `NATURAL` are one thing written two ways. Both join on an
    equality per named column and both hand out one column of each pair rather
    than two, where a condition written with `ON` hands out both. So the node
    is the same join the `ON` spelling builds and the difference is all in what
    the query may write afterwards.

    The node underneath still produces both columns of every pair, because a
    join is a join and nothing here drops a column. What changes is the two
    things that decide what a name means. The pair's column on the right comes
    out of the schema this hands back, so a star over the join writes it once.
    And the name goes into the scope as merged, so an unqualified reference to
    it lowers pinned to one side instead of being refused for naming two
    columns. Either side may still be written in front of it, which is DuckDB's
    rule and Postgres's.

    Which side the merged column comes from is which side the join keeps every
    row of. A `RIGHT` join pads the left, so the left's copy is null on a row
    the left did not match and the right's is the answer. Every other kind
    works the other way round. A `FULL` join pads both and the answer there is
    the first of the two that is not null, which is a `coalesce` and not a
    column, so that one is refused.

    Args:
        plan: Where the node goes.
        left: The left input.
        right: The right input.
        names: The columns the join pairs, in the order they were written.
        kind: Which rows the join keeps.
        scope: What the FROM has put in reach, told about the merged names.
        word: `USING` or `NATURAL`, for the messages.

    Returns:
        The join, with one column of each pair rather than both.

    Raises:
        If a named column is not on both sides exactly once, if it carries no
        relation to pin it to, or if the join is a FULL one.
    """
    if kind == JoinKind.OUTER:
        raise Error(
            String(
                "firepanda does not lower a FULL ",
                word,
                (
                    " join yet. A full join pads both sides, so the merged"
                    " column is the first of the pair that is not null rather"
                    " than one side's, and that is a coalesce over the join"
                    " rather than a column of it"
                ),
            )
        )

    var left_keys = List[Int]()
    var right_keys = List[Int]()
    for i in range(len(names)):
        ref name = names[i]
        var here = _only(left, name, word)
        var there = _only(right, name, word)
        if left.origin[here] == UNBOUND or right.origin[there] == UNBOUND:
            raise Error(
                String(
                    "firepanda does not lower a ",
                    word,
                    " join over a subquery yet. '",
                    name,
                    (
                        "' is a column of both sides, and telling the two apart"
                        " takes the table each came from, which a column a"
                        " subquery computed does not carry"
                    ),
                )
            )
        left_keys.append(plan.exprs.column_of(left.origin[here], String(name)))
        right_keys.append(
            plan.exprs.column_of(right.origin[there], String(name))
        )
        var whose = left.origin[here]
        if kind == JoinKind.RIGHT:
            whose = right.origin[there]
        scope.merge(String(name), whose)

    if len(names) == 0 and not kind.keeps_right_columns():
        raise Error(
            String(
                "a ",
                kind,
                " ",
                word,
                (
                    " join whose two sides share no column name, which asks"
                    " whether the right side has any row at all rather than"
                    " whether it has a matching one, and this join node carries"
                    " key pairs rather than a question"
                ),
            )
        )

    # No shared column at all is every pairing, which is what a NATURAL join of
    # two tables with nothing in common means and what DuckDB answers.
    var built = JoinKind.CROSS if len(names) == 0 else kind
    var at = plan.join(left.at, right.at, left_keys^, right_keys^, built)

    var schema = Schema(copy=left.schema)
    var origin = left.origin.copy()
    for i in range(len(names)):
        # The pair keeps the left's position, which is where DuckDB leaves it,
        # and the relation is whichever side the rows are certainly from.
        origin[_only(left, names[i], word)] = scope.merged_at(names[i])
    if not kind.keeps_right_columns():
        # A SEMI or an ANTI join hands out no column of the right side, so
        # there is no pair to write once and nothing left to append.
        return _From(at, schema^, origin^)
    for i in range(len(right.schema)):
        var paired = False
        for j in range(len(names)):
            if names[j] == right.schema[i].name:
                paired = True
                break
        if paired:
            continue
        schema.append(right.schema[i].copy())
        origin.append(right.origin[i])
    return _From(at, schema^, origin^)


def _shared(left: _From, right: _From) raises -> List[String]:
    """The column names both sides of a join have, which is what NATURAL pairs.

    Args:
        left: The left input.
        right: The right input.

    Returns:
        The names, in the order the left side has them.
    """
    var out = List[String]()
    for i in range(len(left.schema)):
        if right.has(left.schema[i].name) != 0:
            out.append(String(left.schema[i].name))
    return out^


def _table(
    ast: Ast,
    at: UInt32,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    mut scope: _Scope,
    ctes: _Bindings,
) raises -> _From:
    """Lowers one named table in a `FROM` to a scan.

    Args:
        ast: The arenas.
        at: The `REF_TABLE`.
        catalog: What the name is resolved against.
        plan: Where the node goes.
        sources: One schema per scan, appended to.
        scope: What the FROM has put in reach, added to.
        ctes: The CTE names in reach, which a table name is looked up in
            before the catalog.

    Returns:
        The scan and what it produces.

    Raises:
        If the name is not a table this can read, or the reference carries
        column aliases.
    """
    var source = ast.refs[Int(at)]
    var name = _one_name(ast, source.children, "a table")
    var named = ast.length(source.payload)
    if named > 1:
        raise Error(
            "firepanda does not lower the column aliases on a table reference"
            " yet, which rename what the table produces rather than what it is"
            " called"
        )
    var called = name.copy()
    if named == 1:
        called = String(ast.text(ast.at(source.payload, 0)))

    # The CTE names are looked in first, which is what makes a CTE hide a
    # registered frame of the same name rather than collide with one.
    var bound = ctes.find(name)
    if bound != NOT_A_CTE:
        return _cte(ast, bound, called^, catalog, plan, sources, scope, ctes)

    var found = catalog.find(name)
    if found < 0:
        raise Error(catalog.missing(name))
    if catalog.kind_at(found) != KIND_FRAME:
        raise Error(
            String(
                name,
                (
                    " is a view, and firepanda does not lower a view yet"
                    " because the text it holds has to be parsed and lowered in"
                    " the place the name was written"
                ),
            )
        )

    # The scan carries the offset of its own schema in `sources`, so a query
    # over two tables hands binding two schemas and each scan reaches its own.
    var schema = Schema(copy=catalog.frame_at(found).schema)
    var table = len(sources)
    sources.append(Schema(copy=schema))
    scope.add(called^, table)
    var origin = List[Int](length=len(schema), fill=table)
    return _From(plan.scan(name, List[String](), table), schema^, origin^)


def _function(ast: Ast, at: UInt32, mut plan: Plan) raises -> _From:
    """Lowers a function written where a table goes.

    `FROM range(5)` is a source that reads no file and names no table, so
    nothing here touches the catalog and nothing is added to `sources`. The
    arguments are lowered as expressions against nothing, the way a `VALUES`
    lowers its rows, because there is nothing under a table function to read.

    The column is called after the function, which is what DuckDB calls it, so
    `SELECT * FROM range(5)` comes back with a column called `range`. A name
    firepanda invents is a name a query can write, so inventing the same one
    DuckDB does is the difference between a query that ports and one that
    almost does.

    The type is written here as well as in binding, which is one thing known in
    two places and is the price of the schema travelling alongside the node.
    See `_From` for why it has to.

    Args:
        ast: The arenas.
        at: The `REF_FUNCTION`.
        plan: Where the node goes.

    Returns:
        The node and what it produces.

    Raises:
        If the call is `LATERAL`, if it carries an alias, or if an argument is
        an expression this does not lower.
    """
    var source = ast.refs[Int(at)]
    if source.b == 1:
        raise Error(
            "firepanda does not lower a LATERAL table function yet, which is"
            " called once for every row to the left of it rather than once for"
            " the query"
        )
    var name = _one_name(ast, source.a, "a table function")
    if ast.length(source.payload) != 0:
        raise Error(
            String(
                (
                    "firepanda does not lower an alias on a table function yet."
                    " The column "
                ),
                name,
                (
                    " produces belongs to no relation, so a name written in"
                    " front of it would have nothing to resolve against"
                ),
            )
        )

    # Nothing under it, so nothing can aggregate and nothing can be qualified.
    # Both are here because lowering an expression takes them.
    var walk = _Walk()
    var nothing = _Scope()
    var written = ast.items(source.children)
    var args = List[Int](capacity=len(written))
    for i in range(len(written)):
        args.append(_lower_expr(ast, written[i], plan, walk, nothing, False))

    var schema = Schema()
    schema.append(Field(name, LogicalType.INT64, False))
    var origin = List[Int](length=1, fill=UNBOUND)
    var names = List[String](capacity=1)
    names.append(name.copy())
    return _From(plan.table_function(name^, args^, names^), schema^, origin^)


def _subquery(
    ast: Ast,
    at: UInt32,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    mut scope: _Scope,
    ctes: _Bindings,
) raises -> _From:
    """Lowers a subquery written where a table goes.

    A derived table is a whole statement whose output becomes a source, so the
    node this returns is that statement's own root and nothing is wrapped around
    it. What the outer query gets out of it is the names that root produces,
    which is what `SELECT v.x FROM (SELECT a AS x FROM t) v` resolves against.

    The scans inside it go into the same `sources` as the ones outside, because
    a relation number is a number in the whole plan rather than in one block.
    The names do not work that way: a name the outer query put in reach is not
    in reach inside the subquery, so the statement lowers against a scope of its
    own and that scope ends here.

    The alias is not a relation. A scan is one because it has a schema in
    `sources` and a number that names it, and a derived table has neither. So
    the alias goes into the scope as `DERIVED` and a column written in front of
    it is checked against the columns the subquery produces and then lowered
    unqualified. Two sources in the same `FROM` that both produce a column of
    the same name come back from binding as an ambiguity rather than as the
    wrong column, which is a refusal rather than a wrong answer.

    The column aliases in `v(x, y)` rename the columns and not the thing they
    came out of, so they lower to a projection over the statement's root. A
    short list is a prefix, which is what `aliased` does and what a CTE gets,
    but a list longer than the statement produced is refused here while a CTE
    accepts it. That difference is DuckDB's and not an accident of this code.

    Args:
        ast: The arenas.
        at: The `REF_SUBQUERY`.
        catalog: What the table names inside it are resolved against.
        plan: Where the nodes go.
        sources: One schema per scan, appended to.
        scope: What the FROM has put in reach, added to.
        ctes: The CTE names in reach, which the statement inside it can name.

    Returns:
        The statement's root and what it produces.

    Raises:
        If it is `LATERAL`, if it names more columns than the statement
        produces, or if the statement inside it is one this does not lower.
    """
    var source = ast.refs[Int(at)]
    if source.b == 1:
        raise Error(
            "firepanda does not lower a LATERAL subquery yet, which reads the"
            " columns of the sources written to the left of it and so runs once"
            " per row of them rather than once for the query"
        )
    var named = ast.length(source.payload)

    var inner = _Scope()
    var root = _statement(ast, source.a, catalog, plan, sources, inner, ctes)

    # A subquery with no alias on it still produces columns and they are still
    # in reach, so the only thing the missing name costs is the ability to
    # qualify one. DuckDB invents a name here and reading a column through the
    # name it invented is not a thing a query that ports would do.
    var called = String()
    if named >= 1:
        called = String(ast.text(ast.at(source.payload, 0)))

    # The column aliases rename what the statement produced, so they are a
    # projection over its root and not a note kept beside it. A list shorter
    # than the statement is the prefix rule a CTE gets, but a list longer than
    # it is an error here and is not one there, so the check is this one's and
    # the wording is DuckDB's.
    if named > 1:
        var produced = _produces(plan, root)
        if named - 1 > len(produced):
            raise Error(
                String(
                    'Binder Error: table "',
                    called,
                    '" has ',
                    len(produced),
                    " columns available but ",
                    named - 1,
                    " columns specified",
                )
            )
        var columns = List[String](capacity=named - 1)
        for i in range(1, named):
            columns.append(String(ast.text(ast.at(source.payload, i))))
        var outputs = List[Int](capacity=len(produced))
        for i in range(len(produced)):
            outputs.append(plan.exprs.column(String(produced[i])))
        root = plan.project(root, outputs^, aliased(produced, columns))
    return _derived(plan, root, called^, scope)


def _cte(
    ast: Ast,
    at: Int,
    var called: String,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    mut scope: _Scope,
    ctes: _Bindings,
) raises -> _From:
    """Lowers one reference to a CTE, as the plan the name stands for.

    Every reference lowers the statement again. That is DuckDB's default and it
    is the right one here for a plainer reason: a plan holding one node twice
    would be a graph rather than a tree, and every pass in the optimizer walks
    it as a tree. `MATERIALIZED` asks for the other thing and is read and not
    acted on, which costs a query that writes it the work of computing the
    statement twice and never costs it a different answer. The place that
    decision belongs is the optimizer, where the count of uses that `read_ctes`
    already worked out can be weighed against what the statement costs.

    Args:
        ast: The arenas.
        at: Which binding it is.
        called: The name the reference is known by, its alias when it has one.
        catalog: What the table names inside it are resolved against.
        plan: Where the nodes go.
        sources: One schema per scan, appended to.
        scope: What the FROM has put in reach, added to.
        ctes: The CTE names in reach.

    Returns:
        The statement's root and what it produces.

    Raises:
        If the statement the name stands for is one this does not lower.
    """
    # An entry sees the names bound before it and not itself or the ones after
    # it, which is the rule that makes a forward reference a missing table.
    var inner = _Scope()
    var root = _statement(
        ast, ctes.stmts[at], catalog, plan, sources, inner, ctes.upto(at)
    )

    # The alias list renames columns rather than the thing they came out of, so
    # it is a projection and not a note kept on the side. It is a prefix rather
    # than a list, which is `aliased`'s rule and DuckDB's.
    if len(ctes.columns[at]) != 0:
        var produced = _produces(plan, root)
        var outputs = List[Int](capacity=len(produced))
        for i in range(len(produced)):
            outputs.append(plan.exprs.column(String(produced[i])))
        root = plan.project(root, outputs^, aliased(produced, ctes.columns[at]))
    return _derived(plan, root, called^, scope)


def _derived(
    plan: Plan, root: Int, var called: String, mut scope: _Scope
) raises -> _From:
    """Makes a source out of a statement that was lowered where a table goes.

    The types in the schema are the placeholder one. Nothing between here and
    binding reads a type off a source at this stage, since what a source has to
    carry is the names and the relation each column came from, and binding is
    what works the types out from the nodes underneath.

    Args:
        plan: The plan the statement was lowered into.
        root: Its root.
        called: What the query may write in front of its columns, empty when it
            was not given a name.
        scope: What the FROM has put in reach, added to.

    Returns:
        The root and what it produces.

    Raises:
        If the root is a node whose output names are not in the plan, or if the
        name is already taken.
    """
    var names = _produces(plan, root)
    var schema = Schema()
    for i in range(len(names)):
        schema.append(Field(String(names[i]), LogicalType.NULL, True))
    if called.byte_length() != 0:
        scope.derive(called^, names.copy())
    return _From(root, schema^, List[Int](length=len(names), fill=UNBOUND))


def _joined(
    ast: Ast,
    at: UInt32,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    mut scope: _Scope,
    ctes: _Bindings,
) raises -> _From:
    """Lowers a `JOIN` in a `FROM`.

    The condition is split on `AND` and each part is asked which sides it reads.
    A part that is an equality with one side on the left and the other on the
    right is a key pair, which is what the plan's join carries. Anything else is
    a residual, and a residual over an inner join is a filter above it, since
    every pairing the join produces is then tested and that is what the
    condition asked for.

    An outer join is not the same. Its condition decides which rows are padded
    rather than only which rows match, so moving a part of it above the join
    would test the padded rows too and answer a different query. A residual on
    one is refused rather than moved.

    A `USING` or a `NATURAL` join has no condition to split. It names its keys
    by column name instead, and `_merged` is that half.

    Args:
        ast: The arenas.
        at: The `REF_JOIN` or `REF_JOIN_USING`.
        catalog: What the table names are resolved against.
        plan: Where the nodes go.
        sources: One schema per scan, appended to.
        scope: What the FROM has put in reach, added to.
        ctes: The CTE names in reach, for each side.

    Returns:
        The join, and any filter above it, and what it produces.

    Raises:
        If the join or its condition is a shape this does not lower yet.
    """
    var node = ast.refs[Int(at)]
    var kind = _join_kind(ast.text(node.payload))
    var left = _source(ast, node.a, catalog, plan, sources, scope, ctes)
    var reach = len(scope.names)
    var pairs = len(scope.merged)
    var right = _source(ast, node.b, catalog, plan, sources, scope, ctes)

    # A SEMI or an ANTI join keeps left rows and no right column, so the right
    # side goes back out of reach once it has been lowered. Writing its name in
    # front of a column is a missing table afterwards, which is what DuckDB
    # answers for the same query. The join's own condition is the exception and
    # still reads it, so a join that has one hides it further down instead.
    var conditional = (
        node.kind != REF_JOIN_USING and ast.length(node.children) != 0
    )
    if not kind.keeps_right_columns() and not conditional:
        scope.hide(reach, pairs)

    if node.kind == REF_JOIN_USING:
        var named = List[String]()
        for i in range(ast.length(node.children)):
            named.append(String(ast.text(ast.at(node.children, i))))
        return _merged(plan, left, right, named, kind, scope, "USING")
    if String(ast.text(node.payload)).upper().find("NATURAL") != -1:
        return _merged(
            plan, left, right, _shared(left, right), kind, scope, "NATURAL"
        )

    var written = ast.items(node.children)
    var conjuncts = List[UInt32]()
    if len(written) != 0:
        _conjuncts(ast, written[0], conjuncts)
    elif kind != JoinKind.CROSS and kind != JoinKind.INNER:
        raise Error(
            String(
                "a ",
                kind,
                (
                    " join with no condition, which decides nothing about which"
                    " rows match and so has no reading SQL gives it"
                ),
            )
        )

    var left_keys = List[Int]()
    var right_keys = List[Int]()
    var rest = List[Int]()

    # A join condition is not part of any block, so the aggregates it finds
    # belong to nothing. It cannot hold one, which `_lower_expr` refuses on its
    # own because nothing here is grouped, so this walk stays empty.
    var walk = _Walk()
    for i in range(len(conjuncts)):
        var one = ast.exprs[Int(conjuncts[i])]
        if one.kind == EXPR_BINARY and ast.text(one.payload) == "=":
            var a = _lower_expr(ast, one.a, plan, walk, scope, False)
            var b = _lower_expr(ast, one.b, plan, walk, scope, False)
            var first = _side(plan, a, left, right)
            var second = _side(plan, b, left, right)
            if first == _LEFT and second == _RIGHT:
                left_keys.append(a)
                right_keys.append(b)
                continue
            if first == _RIGHT and second == _LEFT:
                left_keys.append(b)
                right_keys.append(a)
                continue
            rest.append(plan.exprs.binary(BinaryOp.EQ, a, b))
            continue
        rest.append(_lower_expr(ast, conjuncts[i], plan, walk, scope, False))

    if not kind.keeps_right_columns():
        scope.hide(reach, pairs)

    if len(rest) != 0 and kind != JoinKind.INNER:
        raise Error(
            String(
                "firepanda lowers a ",
                kind,
                (
                    " join on equalities between its two sides so far, and the"
                    " rest of this condition decides which rows are kept rather"
                    " than which rows match, so it cannot be tested above the"
                    " join instead"
                ),
            )
        )
    if (
        len(left_keys) == 0
        and kind != JoinKind.INNER
        and kind != JoinKind.CROSS
    ):
        raise Error(
            String(
                "a ",
                kind,
                (
                    " join with no equality between its two sides, and"
                    " firepanda's join node carries key pairs rather than a"
                    " predicate"
                ),
            )
        )

    var built = kind
    if len(left_keys) == 0:
        # An inner join with no equality between its sides is every pairing
        # with the condition tested over it, which is a cross join and a filter
        # and is what the rest of the list below builds.
        built = JoinKind.CROSS
    var out = _pair(plan, left, right, left_keys^, right_keys^, built)
    for i in range(len(rest)):
        out.at = plan.filter(out.at, rest[i])
    return out^


def _source(
    ast: Ast,
    at: UInt32,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    mut scope: _Scope,
    ctes: _Bindings,
) raises -> _From:
    """Lowers one table reference in a `FROM`.

    Args:
        ast: The arenas.
        at: The reference.
        catalog: What the table names are resolved against.
        plan: Where the nodes go.
        sources: One schema per scan, appended to.
        scope: What the FROM has put in reach, added to.
        ctes: The CTE names in reach.

    Returns:
        The node and what it produces.

    Raises:
        If the reference is a shape this does not lower yet.
    """
    var source = ast.refs[Int(at)]
    if source.kind == REF_TABLE:
        return _table(ast, at, catalog, plan, sources, scope, ctes)
    if source.kind == REF_JOIN:
        return _joined(ast, at, catalog, plan, sources, scope, ctes)
    if source.kind == REF_PARENS:
        if ast.length(source.payload) != 0:
            raise Error(
                "firepanda does not lower an alias on a parenthesised table"
                " reference yet, which gives a whole join one name and so takes"
                " the names written inside it back out of reach"
            )
        return _source(ast, source.a, catalog, plan, sources, scope, ctes)
    if source.kind == REF_JOIN_USING:
        return _joined(ast, at, catalog, plan, sources, scope, ctes)
    if source.kind == REF_SUBQUERY:
        return _subquery(ast, at, catalog, plan, sources, scope, ctes)
    if source.kind == REF_FUNCTION:
        return _function(ast, at, plan)
    raise Error(
        String("firepanda does not lower table reference kind ", source.kind)
    )


def _from(
    ast: Ast,
    clause: UInt32,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    mut scope: _Scope,
    ctes: _Bindings,
) raises -> _From:
    """Lowers a whole `FROM` clause.

    A comma between two references is a join with no condition. That is what
    makes `FROM a, b WHERE a.x = b.y` and `FROM a JOIN b ON a.x = b.y` the same
    query: the first lowers to a cross join under a filter and the second to a
    cross join with the filter already in the condition, and predicate pushdown
    turns both into the same join. The comma nests left, the same as a chain of
    JOIN words does.

    Args:
        ast: The arenas.
        clause: The `FROM` clause slot.
        catalog: What the table names are resolved against.
        plan: Where the nodes go.
        sources: One schema per scan, appended to.
        scope: What the FROM puts in reach, filled in.
        ctes: The CTE names in reach.

    Returns:
        The node the clause produces and what it produces.

    Raises:
        If the clause is empty or holds a reference this does not lower.
    """
    var refs = ast.items(clause)
    if len(refs) == 0:
        raise Error("a FROM with nothing in it")
    var out = _source(ast, refs[0], catalog, plan, sources, scope, ctes)
    for i in range(1, len(refs)):
        var more = _source(ast, refs[i], catalog, plan, sources, scope, ctes)
        out = _pair(plan, out, more, List[Int](), List[Int](), JoinKind.CROSS)
    return out^


def lower(ast: Ast, statement: UInt32, catalog: Catalog) raises -> Lowered:
    """Lowers a `SELECT` into a logical plan.

    The plan comes back unbound, meaning every column reference in it is a name
    rather than a position. Binding it is `firepanda.plan.bind` over the sources
    this returns, which is the same call the dataframe front end makes, and
    running one binder over both is what stops the two front ends drifting into
    two answers for the same question.

    Args:
        ast: The arenas the statement lives in.
        statement: The `STMT_SELECT` node.
        catalog: What the table names are resolved against.

    Returns:
        The plan, its root and the schemas the scans read.

    Raises:
        If the statement is a shape this does not lower yet.
    """
    var plan = Plan()
    var sources = List[Schema]()
    var scope = _Scope()
    var at = _statement(
        ast, statement, catalog, plan, sources, scope, _Bindings()
    )
    return Lowered(plan^, at, sources^)


def _statement(
    ast: Ast,
    statement: UInt32,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    mut scope: _Scope,
    ctes: _Bindings,
) raises -> Int:
    """Lowers one `STMT_SELECT`: its body, and the `ORDER BY` and `LIMIT` on it.

    Here rather than inside `lower` because a subquery in a `FROM` is a whole
    statement too, with a body and modifiers of its own, and reading that node
    in two places would be two readings of it to keep the same.

    Args:
        ast: The arenas.
        statement: The `STMT_SELECT`.
        catalog: What the table names are resolved against.
        plan: Where the nodes go.
        sources: One schema per scan, appended to in scan order.
        scope: Filled in with what the statement's own FROM put in reach.
        ctes: The CTE names the query around it bound, which its own WITH is
            added to.

    Returns:
        The node the statement produces.

    Raises:
        If the statement is a shape this does not lower yet.
    """
    if statement == NO_NODE:
        raise Error("a statement that is not there")
    var top = ast.stmts[Int(statement)]
    if top.kind != STMT_SELECT:
        raise Error("firepanda lowers a SELECT, and this statement is not one")

    # Read whether or not there is a WITH, because reading one checks it: a
    # name bound twice, a self reference without the keyword, and an ORDER BY
    # on a recursive entry are all refused in there and none of them is a thing
    # this file should be checking a second time.
    var visible = ctes.copy()
    var clause = read_ctes(ast, statement)
    for i in range(len(clause)):
        ref entry = clause.entries[i]
        if entry.recursive:
            raise Error(
                "firepanda does not lower a recursive CTE yet, because the"
                " fixed point it asks for is a node that runs its own input"
                " until no new rows come out, and the plan has no such node"
            )
        visible.bind(String(entry.key), entry.statement, entry.columns.copy())

    var at = _combine(ast, top.a, catalog, plan, sources, scope, visible)

    # The ORDER BY and the LIMIT written after a set operation apply to the
    # whole of it rather than to its last arm, which is why they are put on
    # here and not inside the block.
    if top.b != NO_NODE:
        var walk = _Walk()
        # The scope is the block's when there was one block, so `ORDER BY t.b`
        # names the same table the rest of the query named. A set operation
        # leaves it empty, because the tables were inside the arms and an ORDER
        # BY written after one sorts what the whole of it produced.
        at = _modifiers(ast, top.b, plan, walk, scope, at)
    return at


def _produces(plan: Plan, at: Int) raises -> List[String]:
    """What a node hands out, by name, read back off the plan.

    A derived table needs the names its statement produced, and the thing that
    has them is the plan. Reading them back here rather than threading a name
    list out through every function that lowers a body keeps the one caller that
    wants them from changing the shape of the several that do not.

    Args:
        plan: The nodes.
        at: The node.

    Returns:
        The names it produces, left to right.

    Raises:
        If it is a node whose output names are not written down in the plan.
    """
    ref node = plan.nodes[at]
    var kind = node.kind
    if (
        kind == NodeKind.PROJECT
        or kind == NodeKind.AGGREGATE
        or kind == NodeKind.VALUES
        or kind == NodeKind.TABLE_FUNCTION
    ):
        return node.names.copy()
    if kind == NodeKind.WINDOW:
        # A window adds its columns to the ones below it rather than replacing
        # them, so it is the one node whose output names are its input's and
        # then its own.
        var out = _produces(plan, node.inputs[0])
        for i in range(len(node.names)):
            out.append(String(node.names[i]))
        return out^
    if len(node.inputs) != 0:
        # A union takes the names of its first arm, and the rest of the ones in
        # the middle hand out the names they were given.
        return _produces(plan, node.inputs[0])
    raise Error(
        "a subquery whose output names are not in the plan, which is a scan"
        " with nothing above it saying what it produces"
    )


def _words(text: StringSlice) -> List[String]:
    """Splits a run of text on its spaces, folding each word down.

    Args:
        text: The text.

    Returns:
        The words, in order, with the empty ones between two spaces dropped.
    """
    var words = List[String]()
    var word = List[UInt8]()
    for byte in text.as_bytes():
        if (
            byte == UInt8(32)
            or byte == UInt8(9)
            or byte == UInt8(10)
            or byte == UInt8(13)
        ):
            if len(word) != 0:
                words.append(fold(StringSlice(unsafe_from_utf8=Span(word))))
                word = List[UInt8]()
        else:
            word.append(byte)
    if len(word) != 0:
        words.append(fold(StringSlice(unsafe_from_utf8=Span(word))))
    return words^


def _set_operation(text: StringSlice) raises -> Tuple[Int, Bool]:
    """Reads the words a set operation was written with.

    The AST keeps the operator as the source span it came from, so
    `UNION ALL BY NAME` arrives as one string and this is where it stops being
    one. Splitting on the spaces rather than matching whole strings is what makes
    `union  all` written with two spaces the same operator as `UNION ALL`, and
    folding each word is what makes the lower case spelling the same one too.

    Args:
        text: The operator as it was written.

    Returns:
        The `SET_` code, and whether duplicates survive.

    Raises:
        If the operator is not one of the three, or carries `BY NAME`.
    """
    var words = _words(text)
    if len(words) == 0:
        raise Error("a set operation with no operator on it")

    var op: Int
    if words[0] == "union":
        op = SET_UNION
    elif words[0] == "except":
        op = SET_EXCEPT
    elif words[0] == "intersect":
        op = SET_INTERSECT
    else:
        raise Error(String(words[0], " is not a set operation firepanda knows"))

    # Absent means DISTINCT, which is the one place in SQL where leaving a word
    # out asks for the slower answer.
    var all = False
    for i in range(1, len(words)):
        if words[i] == "all":
            all = True
        elif words[i] == "distinct":
            all = False
        elif words[i] == "by":
            raise Error(
                "firepanda does not lower a set operation written BY NAME yet,"
                " which lines the two sides up by column name rather than by"
                " position and is a projection on each side rather than a"
                " different node"
            )
        else:
            raise Error(
                String(words[i], " is not a word a set operation takes")
            )
    return (op, all)


def _combine(
    ast: Ast,
    body: UInt32,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    mut scope: _Scope,
    ctes: _Bindings,
) raises -> Int:
    """Lowers one query body: a block, a `VALUES`, or a set operation over two.

    A set operation nests rather than flattening. `a EXCEPT b EXCEPT c` and
    `a EXCEPT (b EXCEPT c)` are different answers, so the left leaning shape the
    parser built is the shape the plan gets, and the node is built after both
    sides so that the arena stays in the order binding walks it in.

    Args:
        ast: The arenas.
        body: The `STMT_QUERY`, `STMT_VALUES` or `STMT_SET_OPERATION`.
        catalog: What the table names are resolved against.
        plan: Where the nodes go.
        sources: One schema per scan, appended to in scan order.
        scope: Filled in with what a single block's FROM put in reach, and left
            empty for a set operation and for a VALUES, since neither of those
            leaves a table in reach of an ORDER BY written after it.
        ctes: The CTE names in reach.

    Returns:
        The node the body produces.

    Raises:
        If the body is a shape this does not lower yet.
    """
    if body == NO_NODE:
        raise Error("a query body that is not there")
    var node = ast.stmts[Int(body)]
    if node.kind == STMT_SET_OPERATION:
        var read = _set_operation(ast.text(node.payload))
        # Each arm brings its own scope and neither survives the set operation,
        # so the arms get one apiece and the caller's stays empty.
        var arm = _Scope()
        var left = _combine(ast, node.a, catalog, plan, sources, arm, ctes)
        arm = _Scope()
        var right = _combine(ast, node.b, catalog, plan, sources, arm, ctes)
        return plan.setop([left, right], read[0], read[1])
    if node.kind == STMT_VALUES:
        return _values(ast, body, plan)
    return _block(ast, body, catalog, plan, sources, scope, ctes)


def _values(ast: Ast, body: UInt32, mut plan: Plan) raises -> Int:
    """Lowers a `VALUES` into the literal table node.

    The first row decides how wide the table is and every other row has to
    agree, which is what SQL says and is also the only rule that gives a column
    a position to be in.

    The columns are called `col0`, `col1` and so on, because a `VALUES` names
    nothing and that is what DuckDB calls them. A name it invents is a name a
    query can write, so inventing the same ones DuckDB does is the difference
    between a query that ports and one that almost does.

    Args:
        ast: The arenas.
        body: The `STMT_VALUES`.
        plan: Where the nodes go.

    Returns:
        The node the rows produce.

    Raises:
        If there are no rows, if two rows are different widths, or if a value is
        an expression this does not lower.
    """
    var rows = ast.items(ast.stmts[Int(body)].children)
    if len(rows) == 0:
        raise Error("a VALUES with no rows in it")

    # Nothing under a VALUES, so nothing can aggregate and the walk collects
    # nothing, and no table is in reach so nothing can be qualified either.
    # Both are here because lowering an expression takes them.
    var walk = _Walk()
    var scope = _Scope()
    var width = len(ast.items(rows[0]))
    var lowered = List[Int]()
    for i in range(len(rows)):
        var row = ast.items(rows[i])
        if len(row) != width:
            raise Error(
                String(
                    "row ",
                    i + 1,
                    " of a VALUES has ",
                    len(row),
                    " values and the first row has ",
                    width,
                )
            )
        for j in range(len(row)):
            lowered.append(_lower_expr(ast, row[j], plan, walk, scope, False))

    var names = List[String]()
    for j in range(width):
        names.append(String("col", j))
    return plan.values(lowered^, names^)


def _in_shape(ast: Ast, at: UInt32) raises -> Int:
    """Whether a node is a membership test, and whether it is a negated one.

    Two of the six quantified comparisons are a membership test written the
    other way round. `x = ANY (SELECT k FROM u)` is true when some `k` equals
    `x`, which is the whole of `x IN (SELECT k FROM u)`, and `x <> ALL (...)`
    is true when no `k` does, which is the whole of `x NOT IN (...)` down to
    the nulls: both are null rather than true when nothing matched and some `k`
    was null. `==` and `!=` are other spellings of the two comparisons and come
    here too. `SOME` is another spelling of `ANY` and would come here as well,
    except that the vendored grammar has no word for it and so nothing written
    with it reaches this far.

    So they are not lowered a second time. They are read as the `IN` they are
    and go to the same semi join and the same mark join, which means the null
    aware anti join a `<> ALL` needs is already written and already tested.

    The other four ask whether a comparison holds against some row or every row
    rather than whether a value is in a set. `= ALL` and `<> ANY` are not
    membership either, since `x = ALL (S)` asks that every row equal `x` and
    one matching row does not answer it.

    Args:
        ast: The arenas.
        at: The expression.

    Returns:
        0 for an `IN` over a subquery, 1 for a `NOT IN`, and -1 for anything
        else, including a quantified comparison that is not one of the two.

    Raises:
        If the text of the comparison is not there to read.
    """
    var node = ast.exprs[Int(at)]
    if node.kind == EXPR_IN_SUBQUERY:
        return 1 if node.payload == 1 else 0
    if node.kind != EXPR_QUANTIFIED:
        return -1
    var written = ast.text(node.payload)
    var equality = written == "=" or written == "=="
    if node.children == 0 and equality:
        return 0
    if node.children == 1 and (written == "<>" or written == "!="):
        return 1
    return -1


def _in_join(
    ast: Ast,
    at: UInt32,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    mut scope: _Scope,
    mut walk: _Walk,
    ctes: _Bindings,
    var left: _From,
) raises -> _From:
    """Turns `x IN (SELECT ...)` written in a `WHERE` into a semi join.

    The subquery is a query of its own and it lowers as one, against a scope of
    its own, so a name the outer query put in reach is not in reach inside it.
    That is the ordinary rule for a subquery and it is also what makes this
    rewrite safe: a subquery that reads no outer column runs once, and running
    it once is what a join does with its build side.

    The join is a semi join on one equality, the left side being whatever was
    written in front of the `IN` and the right being the one column the subquery
    hands out. That is the whole of `IN` and not an approximation of it. A left
    row that matched several right rows comes back once, which is what a semi
    join does anyway, and a null on either side matches nothing, which is what
    `IN` answers: `NULL IN (1, 2)` is null and a row a filter does not keep, and
    `3 IN (1, NULL)` is null and the same.

    Args:
        ast: The arenas.
        at: The `EXPR_IN_SUBQUERY`.
        catalog: What the table names inside it are resolved against.
        plan: Where the nodes go.
        sources: One schema per scan, appended to.
        scope: What the outer `FROM` put in reach, which the left side reads.
        walk: What the aggregates found so far are recorded in.
        ctes: The CTE names in reach, which the subquery may name.
        left: What the join probes with.

    Returns:
        The join, which produces what the left side produced and nothing else.

    Raises:
        If the subquery hands out other than one column, or if the subquery is
        one this does not lower.
    """
    var node = ast.exprs[Int(at)]
    if _in_shape(ast, at) == 1:
        # `_asks` does not hand a NOT IN here, because an anti join is the
        # classic wrong answer for one: a single null anywhere in the subquery
        # makes NOT IN null for every row rather than true, and an anti join
        # keeps those rows instead of dropping them. It goes to the mark join
        # with every other NOT IN. This is the guard that says so.
        raise Error(
            "a NOT IN reached the semi join, and a NOT IN is a mark join"
        )
    var key = _lower_expr(ast, node.a, plan, walk, scope, False)

    var inner = _Scope()
    var root = _statement(ast, node.b, catalog, plan, sources, inner, ctes)
    var names = _produces(plan, root)
    if len(names) != 1:
        raise Error(
            String(
                "an IN whose subquery hands out ",
                len(names),
                " columns, and what a value is looked for in is one column",
            )
        )

    # The right key is a bare name and it binds against the build side alone,
    # which is why a subquery that hands out a column the outer query also has
    # is not an ambiguity here.
    var found = plan.exprs.column(String(names[0]))
    var schema = Schema()
    schema.append(Field(String(names[0]), LogicalType.NULL, True))
    var right = _From(root, schema^, List[Int](length=1, fill=UNBOUND))

    var keys = List[Int]()
    keys.append(key)
    var others = List[Int]()
    others.append(found)
    return _pair(plan, left, right, keys^, others^, JoinKind.SEMI)


def _asks(ast: Ast, at: UInt32) raises -> UInt32:
    """The `IN` or the `EXISTS` one part of a `WHERE` is, if it is one.

    A `NOT EXISTS` parses as a `NOT` written over a plain `EXISTS` rather than
    as an `EXISTS` carrying its own negation, so the `NOT` is looked through
    here and what comes back is the `EXISTS` under it. A `NOT` over anything
    else is not looked through, so a `NOT (x IN (SELECT ...))` goes the way an
    `IN` written as a value goes, which is the mark join.

    A `NOT IN` is not claimed here either. It carries its negation on itself
    rather than in a `NOT` above it, and an anti join is the classic wrong
    answer for it, so it goes to the mark join too and the `NOT` over the
    column the mark join wrote is what makes it right.

    A `= ANY` is claimed, because it is an `IN` written the other way round, and
    a `<> ALL` is not, because it is a `NOT IN` and goes where those go. The
    other four quantified comparisons are claimed by neither and are refused
    where an expression is lowered.

    An `EXISTS` is claimed only when `_may_pair` says the semi join could have a
    key pair to join on. One that could not is uncorrelated, since correlation
    written anywhere but an equality in the subquery's `WHERE` is refused rather
    than lowered, and an uncorrelated `EXISTS` is the same answer for every row.
    That is the value form and it is counted under a cross join instead.

    Args:
        ast: The arenas.
        at: One part of the `WHERE`.

    Returns:
        The `EXPR_IN_SUBQUERY` or the `EXPR_EXISTS`, or `NO_NODE` if this part
        is neither and so is an ordinary condition or a value.

    Raises:
        If the text of an operator is not there to read.
    """
    var node = ast.exprs[Int(at)]
    if node.kind == EXPR_IN_SUBQUERY or node.kind == EXPR_QUANTIFIED:
        return at if _in_shape(ast, at) == 0 else NO_NODE
    if node.kind == EXPR_EXISTS:
        return at if _may_pair(ast, node.a) else NO_NODE
    if node.kind == EXPR_UNARY and ast.text(node.payload) == "NOT":
        var inner = ast.exprs[Int(node.a)]
        if inner.kind == EXPR_EXISTS:
            return node.a if _may_pair(ast, inner.a) else NO_NODE
    return NO_NODE


def _may_pair(ast: Ast, at: UInt32) raises -> Bool:
    """Whether an `EXISTS`'s subquery could give the semi join a key pair.

    Read before anything is lowered, and read off what is written rather than
    off what the names turn out to mean, because the choice it decides has to be
    made before the `FROM` under it exists.

    A key pair comes from an equality in the subquery's `WHERE` with one side in
    and one side out. So a subquery with no `WHERE`, or with one holding no
    equality at the top level of its `AND`, has no pair to give whatever its
    names mean, and it is therefore uncorrelated: correlation written any other
    way is refused rather than lowered. Those go to the value form, which counts
    the subquery's rows under a cross join.

    The shapes the semi join refuses outright answer False here as well, which
    sends them to the value form too. Each of them is a subquery the value form
    lowers correctly: an aggregate with no `GROUP BY` is one row and so an
    `EXISTS` over it is true, a `LIMIT` changes how many rows there are and the
    counting sees the change, and a set operation or a `VALUES` is a statement
    like any other there.

    What is left ambiguous is an equality that reads neither side of the join,
    like `WHERE u.k = 3`. That is claimed and then refused by the semi join for
    reading no outer column, which is what it did before this and is a query
    nobody writes.

    Args:
        ast: The arenas.
        at: The subquery's statement.

    Returns:
        True if the semi join should be given it, False if it is a value.
    """
    var top = ast.stmts[Int(at)]
    if top.kind != STMT_SELECT or top.b != NO_NODE:
        return False
    if len(read_ctes(ast, at)) != 0:
        return False
    var body = ast.stmts[Int(top.a)]
    if body.kind != STMT_QUERY:
        return False

    var clauses = body.children
    if ast.slot(clauses, CLAUSE_FROM) == NO_NODE:
        return False
    if (
        ast.length(ast.slot(clauses, CLAUSE_GROUP)) != 0
        or ast.slot(clauses, CLAUSE_HAVING) != NO_NODE
    ):
        return False
    for item in ast.items(ast.slot(clauses, CLAUSE_PROJECTION)):
        if _has_aggregate(ast, ast.stmts[Int(item)].a):
            return False

    var restriction = ast.slot(clauses, CLAUSE_WHERE)
    if restriction == NO_NODE:
        return False
    var parts = List[UInt32]()
    _conjuncts(ast, restriction, parts)
    for i in range(len(parts)):
        var one = ast.exprs[Int(parts[i])]
        if one.kind == EXPR_BINARY and ast.text(one.payload) == "=":
            return True
    return False


def _scalar_join(
    ast: Ast,
    at: UInt32,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    ctes: _Bindings,
    mut walk: _Walk,
    left: Int,
) raises -> Int:
    """Puts an uncorrelated subquery that answers one value under the query.

    The subquery is a plan of its own and its answer is one value, the same
    value for every row of the query around it, so it is a cross join onto one
    row. That is a column added to each row and nothing moved, and the lowering
    turns it into a constant per right column, so the cost is the subquery run
    once rather than once per row.

    Only a subquery that is one row by construction is taken, which means one
    that aggregates with no `GROUP BY`, or one with no `FROM` at all. Both of
    those answer exactly one row whatever is in the tables, including no rows,
    where a fold answers a null and that is the right answer. A `LIMIT 1` looks
    like the same guarantee and is not, because over an empty table it answers
    no rows, where SQL says the subquery is null and the cross join here would
    raise instead.

    A correlated one is not taken either, and it is refused by where it lowers
    rather than by a check: the subquery gets a scope of its own, so an outer
    name written inside it is a name nothing in that query has. That is the
    dependent join, and it is the same refusal a correlated `IN` gets.

    Args:
        ast: The arenas.
        at: The `EXPR_SUBQUERY`.
        catalog: What the table names inside it are resolved against.
        plan: Where the nodes go.
        sources: One schema per scan, appended to.
        ctes: The CTE names in reach.
        walk: Told what the answer's column is called, so that the expression
            around it can read it back.
        left: What the cross join is put over.

    Returns:
        The cross join, which produces what the left side produced and the one
        column the subquery answered.

    Raises:
        If the subquery is a shape whose row count is not one by construction,
        or if it hands out more than one column.
    """
    var node = ast.exprs[Int(at)]
    var top = ast.stmts[Int(node.a)]
    if top.kind != STMT_SELECT:
        raise Error("a subquery over a statement that is not a SELECT")
    if top.b != NO_NODE:
        raise Error(
            "firepanda lowers a subquery that answers one value when its block"
            " is one row by construction, and an ORDER BY or a LIMIT on one"
            " does not make it one: a LIMIT 1 over a table with nothing in it"
            " answers no rows, where SQL says the subquery is null"
        )
    var body = ast.stmts[Int(top.a)]
    if body.kind != STMT_QUERY:
        raise Error(
            "firepanda lowers a subquery that answers one value over one SELECT"
            " block so far, and a VALUES or a set operation inside one is a"
            " different node"
        )

    var clauses = body.children
    var items = ast.items(ast.slot(clauses, CLAUSE_PROJECTION))
    var from_clause = ast.slot(clauses, CLAUSE_FROM)
    var folds = False
    for one in items:
        if _has_aggregate(ast, ast.stmts[Int(one)].a):
            folds = True
            break
    if from_clause != NO_NODE:
        if ast.length(ast.slot(clauses, CLAUSE_GROUP)) != 0 or not folds:
            raise Error(
                "firepanda lowers a subquery that answers one value when its"
                " block is one row by construction, which is a fold with no"
                " GROUP BY under it or a SELECT with no FROM. This one hands"
                " out a row per row of its table, and the check that there is"
                " exactly one of them is a node nobody has written"
            )

    var inner = _Scope()
    var root = _statement(ast, node.a, catalog, plan, sources, inner, ctes)
    var names = _produces(plan, root)
    if len(names) != 1:
        raise Error(
            String(
                "a subquery written as a value that hands out ",
                len(names),
                " columns, and a value is one column",
            )
        )

    # Renamed on the way out, because the name the subquery gave its column is
    # whatever was written in there and the query around it may already have a
    # column called that. Nothing reads the new name but the expression this
    # was taken out of, which is told it here.
    var called = String("__sub_", len(walk.scalars))
    var only = List[Int]()
    only.append(plan.exprs.column(String(names[0])))
    var renamed = List[String]()
    renamed.append(String(called))
    var one_row = plan.project(root, only^, renamed^)
    walk.scalars.append(at)
    walk.scalar_names.append(called^)
    return plan.join(left, one_row, List[Int](), List[Int](), JoinKind.CROSS)


def _mark_join(
    ast: Ast,
    at: UInt32,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    ctes: _Bindings,
    mut walk: _Walk,
    scope: _Scope,
    left: Int,
) raises -> Int:
    """Puts an `IN` over a subquery under the query as a mark join.

    An `IN` written where a `WHERE` is the `AND` of it and other things is a
    semi join, because there the question is which rows to keep. Written
    anywhere else it is a value, and a value has to arrive on every row rather
    than on the rows that matched. That is the mark join: the left side comes
    out unchanged and one boolean column comes out beside it.

    A `NOT IN` is the same join. The negation stays in the expression, where it
    is a `NOT` over the column this wrote, and the three valued rule falls out
    of that rather than being written anywhere: a row that matched nothing over
    a build side holding a null is marked null, a `NOT` over a null is null, and
    a filter does not keep a null. So `x NOT IN (SELECT y FROM t)` keeps no rows
    at all when any `y` is null, which is what SQL says and what an anti join
    gets wrong.

    The subquery is a query of its own and it lowers as one, against a scope of
    its own, so a name the outer query put in reach is not in reach inside it. A
    correlated one is refused by that rather than by a check, the same way a
    correlated subquery written as a value is.

    Args:
        ast: The arenas.
        at: The `EXPR_IN_SUBQUERY`.
        catalog: What the table names inside it are resolved against.
        plan: Where the nodes go.
        sources: One schema per scan, appended to.
        ctes: The CTE names in reach.
        walk: Told what the answer's column is called, so that the expression
            around it can read it back.
        scope: What the outer `FROM` put in reach, which the left key reads.
        left: What the mark join is put over.

    Returns:
        The mark join, which produces what the left side produced and the one
        boolean column.

    Raises:
        If the subquery hands out other than one column, or if it is one this
        does not lower.
    """
    var node = ast.exprs[Int(at)]
    var key = _lower_expr(ast, node.a, plan, walk, scope, False)

    var inner = _Scope()
    var root = _statement(ast, node.b, catalog, plan, sources, inner, ctes)
    var names = _produces(plan, root)
    if len(names) != 1:
        raise Error(
            String(
                "an IN whose subquery hands out ",
                len(names),
                " columns, and what a value is looked for in is one column",
            )
        )

    # Named rather than positioned, because the query around it may already have
    # a column called whatever the subquery called its own, and nothing reads
    # this name but the expression the `IN` was taken out of.
    var called = String("__mark_", len(walk.marks))
    var keys = List[Int]()
    keys.append(key)
    var others = List[Int]()
    others.append(plan.exprs.column(String(names[0])))
    walk.marks.append(at)
    walk.mark_names.append(String(called))
    return plan.join(left, root, keys^, others^, JoinKind.MARK, called^)


def _exists_value(
    ast: Ast,
    at: UInt32,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    ctes: _Bindings,
    mut walk: _Walk,
    left: Int,
) raises -> Int:
    """Puts an `EXISTS` written as a value under the query, as a count.

    An `EXISTS` written where a `WHERE` is the `AND` of it and other things is a
    semi join, because there the question is which rows to keep. Written
    anywhere else it is a value, and an uncorrelated one is the same value on
    every row, because whether the subquery has a row in it does not depend on
    which outer row is asking.

    So it is the same shape a subquery answering one value gets: the subquery is
    a plan of its own, one row is worked out from it, and that row is cross
    joined on above the `FROM`. The row here is `count(*) > 0`, which is a fold
    with no `GROUP BY` and so is one row whatever the subquery read, including
    nothing. The comparison sits under the cross join rather than over it so
    that it is done once rather than once per outer row.

    There is no mark join in this and there is no null either. An `IN` has to be
    three valued because it compares values and a null compares to nothing, and
    `EXISTS` counts rows without looking in them, so it is true or false and a
    `NOT EXISTS` is the plain opposite. That is why this is the cross join
    rather than the mark join, even though both answer a boolean per row.

    A correlated one is not taken, and it is refused by where it lowers rather
    than by a check: the subquery gets a scope of its own, so an outer name
    written inside it is a name nothing in that query has. Those are the ones
    the semi join above is for, and the ones that are correlated and not written
    in a `WHERE` are the dependent join.

    Args:
        ast: The arenas.
        at: The `EXPR_EXISTS`.
        catalog: What the table names inside it are resolved against.
        plan: Where the nodes go.
        sources: One schema per scan, appended to.
        ctes: The CTE names in reach.
        walk: Told what the answer's column is called, so that the expression
            around it can read it back.
        left: What the cross join is put over.

    Returns:
        The cross join, which produces what the left side produced and the one
        boolean column the counting answered.

    Raises:
        If the subquery does not lower.
    """
    var node = ast.exprs[Int(at)]
    var inner = _Scope()
    var root: Int
    try:
        root = _statement(ast, node.a, catalog, plan, sources, inner, ctes)
    except failed:
        # The scope is the refusal a correlated one gets, and a name nothing in
        # the subquery has is what that looks like from in here. Saying so
        # beside what went wrong costs nothing and is the difference between a
        # message about a missing table and a message about the shape of the
        # query.
        raise Error(
            String(
                "an EXISTS written as a value did not lower: ",
                failed,
                (
                    ". It is lowered against a scope of its own, so a name it"
                    " takes from the query around it is a name nothing in it"
                    " has, and a correlated EXISTS written anywhere but the AND"
                    " of a WHERE, or written there over a shape the semi join"
                    " does not take, is the dependent join"
                ),
            )
        )

    # The select list is not read, as the semi join does not read it either. What
    # is counted is a one, which is what `count(*)` lowers to everywhere else,
    # and counting a constant is counting rows.
    var one = plan.exprs.literal(Value(Int64(1)))
    var counted = List[Int]()
    counted.append(plan.exprs.aggregate(AggKind.COUNT, one))
    var under = List[String]()
    under.append(String("__rows"))
    var folded = plan.aggregate(root, List[Int](), counted^, under^)

    # Named rather than positioned, for the reason the subquery answering one
    # value is: the query around it may already have a column called anything,
    # and nothing reads this name but the expression the `EXISTS` was taken out
    # of.
    var called = String("__has_", len(walk.asked))
    # A negation written on the node rather than in a `NOT` above it, which is
    # the other spelling the parser can hand over, is the other comparison. A
    # `NOT` above it is left where it is and read as an ordinary `NOT`.
    var against = BinaryOp.EQ if node.b == 1 else BinaryOp.GT
    var found = List[Int]()
    found.append(
        plan.exprs.binary(
            against,
            plan.exprs.column("__rows"),
            plan.exprs.literal(Value(Int64(0))),
        )
    )
    var renamed = List[String]()
    renamed.append(String(called))
    var one_row = plan.project(folded, found^, renamed^)
    walk.asked.append(at)
    walk.asked_names.append(called^)
    return plan.join(left, one_row, List[Int](), List[Int](), JoinKind.CROSS)


def _quantified_value(
    ast: Ast,
    at: UInt32,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    ctes: _Bindings,
    mut walk: _Walk,
    left: Int,
) raises -> Int:
    """Puts the row a quantified comparison is answered against under the query.

    `x > ANY (SELECT k FROM u)` asks whether any `k` is under `x`, which is
    whether the smallest one is, and `x > ALL (...)` asks whether every `k` is,
    which is whether the largest one is. So the whole of the subquery that the
    comparison needs is four numbers, and they are a fold with no `GROUP BY`
    over it: the smallest `k`, the largest one, how many rows there were and how
    many of those were not null. That row is cross joined on above the `FROM`
    the way an `EXISTS` written as a value is, and the comparison itself is done
    where it was written, because that is the side `x` is on.

    Four is what the three valued rule costs, and none of the four is spare. The
    two counts are not the same number when the subquery holds a null, and a
    null in there is what turns a false into a null: `x > ALL (S)` is not true
    just because `x` beat every `k` that was there to beat, since the null might
    have been larger. It is also not false, so the answer is neither. That is
    `__pad`, a null when the two counts differ and the operator's identity when
    they agree, joined to the comparison by the quantifier's own operator.
    `__fill` is the other end, the subquery with no rows at all, where `ANY` is
    false and `ALL` is true whatever `x` is, and it is joined on by the other
    operator so that it decides the answer there and nothing anywhere else.

    A correlated one is not taken, and it is refused by where it lowers rather
    than by a check, which is what happens to an `EXISTS` written as a value
    too. The subquery gets a scope of its own, so an outer name written inside
    it is a name nothing in that query has.

    Args:
        ast: The arenas.
        at: The `EXPR_QUANTIFIED`.
        catalog: What the table names inside it are resolved against.
        plan: Where the nodes go.
        sources: One schema per scan, appended to.
        ctes: The CTE names in reach.
        walk: Told that the row is there, so that the expression around it can
            read the four columns back.
        left: What the cross join is put over.

    Returns:
        The cross join, which produces what the left side produced and the four
        columns the fold answered.

    Raises:
        If the subquery does not lower, or hands out other than one column.
    """
    var node = ast.exprs[Int(at)]
    var inner = _Scope()
    var root: Int
    try:
        root = _statement(ast, node.b, catalog, plan, sources, inner, ctes)
    except failed:
        raise Error(
            String(
                "the subquery of a quantified comparison did not lower: ",
                failed,
                (
                    ". It is lowered against a scope of its own, so a name it"
                    " takes from the query around it is a name nothing in it"
                    " has, and a correlated ANY or ALL is the dependent join"
                ),
            )
        )

    var names = _produces(plan, root)
    if len(names) != 1:
        raise Error(
            String(
                "an ANY or an ALL whose subquery hands out ",
                len(names),
                " columns, and what a value is compared against is one column",
            )
        )

    var over = plan.exprs.column(String(names[0]))
    var folded = List[Int]()
    folded.append(plan.exprs.aggregate(AggKind.MIN, over))
    folded.append(plan.exprs.aggregate(AggKind.MAX, over))
    folded.append(
        plan.exprs.aggregate(AggKind.COUNT, plan.exprs.literal(Value(Int64(1))))
    )
    folded.append(plan.exprs.aggregate(AggKind.COUNT, over))
    var under = List[String]()
    under.append(String("__low"))
    under.append(String("__high"))
    under.append(String("__rows"))
    under.append(String("__seen"))
    var row = plan.aggregate(root, List[Int](), folded^, under^)

    # Numbered by how many were taken out before, the way the mark join's column
    # is, so two comparisons in one query read two rows rather than one name
    # meaning both.
    var place = len(walk.compared)
    var every = node.children == 1
    var other = String("or") if every else String("and")

    # Both of these are worked out here rather than at the comparison, because
    # each is the same value for every outer row and doing it under the cross
    # join does it once.
    #
    # The padding says what a null in the subquery did. For `ALL` it is a true
    # when the two counts agree and a null when they do not, so joining it on
    # with `AND` leaves a false alone and turns a true into a null, which is
    # exactly what a null the comparison never saw is worth. For `ANY` it is a
    # false and a null the same way, joined on with `OR`.
    var pad = plan.exprs.call(
        other,
        [
            plan.exprs.binary(
                BinaryOp.EQ if every else BinaryOp.NE,
                plan.exprs.column("__rows"),
                plan.exprs.column("__seen"),
            ),
            plan.exprs.literal(Value(null=LogicalType.BOOL)),
        ],
        True,
    )
    # The filling is the empty subquery, where `ALL` is true and `ANY` is false
    # whatever is on the other side. It is joined on with the operator the
    # padding was not, so over an empty subquery it decides the answer and over
    # any other it is the operator's identity and decides nothing.
    var fill = plan.exprs.binary(
        BinaryOp.EQ if every else BinaryOp.GT,
        plan.exprs.column("__rows"),
        plan.exprs.literal(Value(Int64(0))),
    )
    var values = List[Int]()
    values.append(plan.exprs.column("__low"))
    values.append(plan.exprs.column("__high"))
    values.append(pad)
    values.append(fill)
    var renamed = List[String]()
    renamed.append(String("__low_", place))
    renamed.append(String("__high_", place))
    renamed.append(String("__pad_", place))
    renamed.append(String("__fill_", place))
    var one_row = plan.project(row, values^, renamed^)
    walk.compared.append(at)
    return plan.join(left, one_row, List[Int](), List[Int](), JoinKind.CROSS)


def _quantified_read(
    ast: Ast,
    at: UInt32,
    mut plan: Plan,
    mut walk: _Walk,
    scope: _Scope,
    grouped: Bool,
    place: Int,
) raises -> Int:
    """Reads a quantified comparison off the row that was cross joined on.

    The comparison is done here rather than under the join because `x` is on
    this side of it. Which of the two ends of the subquery `x` is compared
    against is the whole of the rewrite: `> ANY` and `>= ANY` ask about the
    smallest row and `> ALL` and `>= ALL` about the largest, and `<` and `<=`
    ask the other way round under each quantifier.

    `= ALL` and `<> ANY` are not one comparison, because neither end of the
    range answers them on its own: every row equals `x` when both ends do, and
    some row differs from `x` when either end does. So each is the two ends
    joined, which is still two comparisons rather than a scan of the subquery.

    What is built over that is the three valued rule, and it is two operators
    wide. The comparison is joined to the padding by the quantifier's own
    operator, `OR` for `ANY` and `AND` for `ALL`, which is what the quantifier
    means read across the rows: a true out of `ANY` survives a null and so does
    a false out of `ALL`, and everything the null was going to decide becomes a
    null. The filling is joined on by the other operator, where it is that
    operator's identity unless the subquery was empty and so decides the answer
    there and nowhere else. A null `x` needs nothing written for it, since
    comparing it against either end is null already.

    Args:
        ast: The arenas.
        at: The `EXPR_QUANTIFIED`.
        plan: Where the nodes go.
        walk: The walk the operand is lowered against.
        scope: What the `FROM` put in reach, which the operand reads.
        grouped: Whether the query aggregates, for the operand.
        place: Which of the rows taken out is this one's.

    Returns:
        The expression, which is a boolean and may be null.

    Raises:
        If the comparison is not one this answers, or the operand does not
        lower.
    """
    var node = ast.exprs[Int(at)]
    var every = node.children == 1
    var written = ast.text(node.payload)
    var value = _lower_expr(ast, node.a, plan, walk, scope, grouped)
    var low = plan.exprs.column(String("__low_", place))
    var high = plan.exprs.column(String("__high_", place))

    var core: Int
    if written == "=" or written == "==":
        core = plan.exprs.call(
            "and",
            [
                plan.exprs.binary(BinaryOp.EQ, value, low),
                plan.exprs.binary(BinaryOp.EQ, value, high),
            ],
            True,
        )
    elif written == "<>" or written == "!=":
        core = plan.exprs.call(
            "or",
            [
                plan.exprs.binary(BinaryOp.NE, value, low),
                plan.exprs.binary(BinaryOp.NE, value, high),
            ],
            True,
        )
    else:
        var under = written == "<" or written == "<="
        core = plan.exprs.binary(
            _binary_op(written), value, low if under == every else high
        )

    var joined = plan.exprs.call(
        "and" if every else "or",
        [core, plan.exprs.column(String("__pad_", place))],
        True,
    )
    return plan.exprs.call(
        "or" if every else "and",
        [joined, plan.exprs.column(String("__fill_", place))],
        True,
    )


def _exists_join(
    ast: Ast,
    at: UInt32,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    mut scope: _Scope,
    ctes: _Bindings,
    var left: _From,
    negated: Bool,
) raises -> _From:
    """Turns a correlated `EXISTS` written in a `WHERE` into a semi join.

    This is decorrelation, done for the one shape it is a rewrite rather than a
    pass. An `EXISTS` whose subquery reads the query around it is a question
    asked once per outer row, and on anything the size of a benchmark that is
    the difference between a query that finishes and one that does not. When the
    reading it does is equalities against the outer tables, the answer for every
    outer row at once is a semi join on those equalities, and `NOT EXISTS` is
    the same join asking the other way.

    So the subquery's `FROM` is lowered into the scope the outer query is
    already using, which puts both sides of the eventual join in reach at once,
    and its `WHERE` is then split the way a join condition is split. A part with
    one side out and one side in is a key pair. A part that reads the subquery's
    own tables and nothing else is a filter under the join, where it runs once
    rather than once per outer row. A part that reads the outer query any other
    way is refused, since it is correlation the key pairs cannot carry.

    Afterwards the subquery's tables go back out of reach, for the reason every
    semi join's right side does: the join kept no column of them.

    The subquery's select list is not lowered at all, because `EXISTS` asks
    whether there is a row rather than what is in it. DuckDB does bind it and so
    refuses a name in there that no table has, and here that name goes unread.

    Args:
        ast: The arenas.
        at: The `EXPR_EXISTS`.
        catalog: What the table names inside it are resolved against.
        plan: Where the nodes go.
        sources: One schema per scan, appended to.
        scope: What the outer `FROM` put in reach, added to and put back.
        ctes: The CTE names in reach.
        left: What the join probes with.
        negated: Whether a `NOT` is written over it, which is how a `NOT EXISTS`
            reaches here.

    Returns:
        The join, which produces what the left side produced and nothing else.

    Raises:
        If the subquery is a shape whose rows are not the rows of one block over
        one `FROM`, if it reads no outer column, or if it reads one any way but
        an equality.
    """
    var node = ast.exprs[Int(at)]
    # Either spelling of the negation arrives, since the node carries one and a
    # `NOT` written in front of it is the other, and two of them cancel.
    var anti = (node.b == 1) != negated
    var kind = JoinKind.ANTI if anti else JoinKind.SEMI
    var word = "NOT EXISTS" if anti else "EXISTS"

    var top = ast.stmts[Int(node.a)]
    if top.kind != STMT_SELECT:
        raise Error("an EXISTS over a statement that is not a SELECT")
    if top.b != NO_NODE:
        raise Error(
            String(
                "firepanda lowers a ",
                word,
                (
                    " whose subquery is one SELECT block so far, and a LIMIT on"
                    " one changes how many rows it has and so whether it has"
                    " any"
                ),
            )
        )
    if len(read_ctes(ast, node.a)) != 0:
        raise Error(
            "firepanda does not lower a WITH written inside an EXISTS yet,"
            " which binds a name for that subquery alone"
        )

    var body = ast.stmts[Int(top.a)]
    if body.kind != STMT_QUERY:
        raise Error(
            "firepanda lowers an EXISTS over one SELECT block so far, and a"
            " VALUES or a set operation inside one is a different node"
        )
    var clauses = body.children
    if (
        ast.length(ast.slot(clauses, CLAUSE_GROUP)) != 0
        or ast.slot(clauses, CLAUSE_HAVING) != NO_NODE
    ):
        raise Error(
            "firepanda lowers an EXISTS over a SELECT that does not aggregate,"
            " because an aggregate with no group by answers one row over no"
            " rows and so an EXISTS over one is true where the subquery is"
            " empty"
        )
    for item in ast.items(ast.slot(clauses, CLAUSE_PROJECTION)):
        if _has_aggregate(ast, ast.stmts[Int(item)].a):
            raise Error(
                "firepanda lowers an EXISTS over a SELECT that does not"
                " aggregate, and this one folds in its select list"
            )
    if ast.length(ast.slot(clauses, CLAUSE_WINDOW)) != 0:
        raise Error("firepanda does not lower a WINDOW clause yet")

    var from_clause = ast.slot(clauses, CLAUSE_FROM)
    if from_clause == NO_NODE:
        raise Error(
            "an EXISTS over a SELECT with no FROM, which has one row and so is"
            " a constant rather than a question about a table"
        )

    # Lowered into the caller's scope rather than a scope of its own, which is
    # the whole difference between this and the uncorrelated case: a condition
    # that reads both sides can only be written where both sides are in reach.
    var reach = len(scope.names)
    var merged = len(scope.merged)
    var right = _from(ast, from_clause, catalog, plan, sources, scope, ctes)

    var conjuncts = List[UInt32]()
    var restriction = ast.slot(clauses, CLAUSE_WHERE)
    if restriction != NO_NODE:
        _conjuncts(ast, restriction, conjuncts)

    var left_keys = List[Int]()
    var right_keys = List[Int]()
    var inside = List[Int]()
    var walk = _Walk()
    for i in range(len(conjuncts)):
        var one = ast.exprs[Int(conjuncts[i])]
        if one.kind == EXPR_BINARY and ast.text(one.payload) == "=":
            var a = _lower_expr(ast, one.a, plan, walk, scope, False)
            var b = _lower_expr(ast, one.b, plan, walk, scope, False)
            var first = _side(plan, a, left, right)
            var second = _side(plan, b, left, right)
            if first == _LEFT and second == _RIGHT:
                left_keys.append(a)
                right_keys.append(b)
                continue
            if first == _RIGHT and second == _LEFT:
                left_keys.append(b)
                right_keys.append(a)
                continue
            if (
                first != _LEFT
                and first != _BOTH
                and second != _LEFT
                and (second != _BOTH)
            ):
                inside.append(plan.exprs.binary(BinaryOp.EQ, a, b))
                continue
        else:
            var whole = _lower_expr(ast, conjuncts[i], plan, walk, scope, False)
            var reads = _side(plan, whole, left, right)
            if reads != _LEFT and reads != _BOTH:
                inside.append(whole)
                continue
        raise Error(
            String(
                "firepanda decorrelates a ",
                word,
                (
                    " whose subquery reads the query around it through"
                    " equalities and nothing else, and this part of its"
                    " condition reads it another way, which is the dependent"
                    " join that a decorrelation pass removes rather than one"
                    " this rewrite can"
                ),
            )
        )

    # Back out of reach, for the reason a written out semi join's right side
    # goes out of reach: the join hands out no column of it.
    scope.hide(reach, merged)

    if len(left_keys) == 0:
        raise Error(
            String(
                "a ",
                word,
                (
                    " that reads no column of the query around it, which asks"
                    " whether its own table has any row at all rather than"
                    " whether it has a matching one, and that is a mark join"
                    " rather than a semi join"
                ),
            )
        )

    # Under the join rather than over it. A condition that reads the subquery
    # alone is the same answer wherever it is tested, and tested here it runs
    # once over that table instead of once per pairing.
    for i in range(len(inside)):
        right.at = plan.filter(right.at, inside[i])
    return _pair(plan, left, right, left_keys^, right_keys^, kind)


def _block(
    ast: Ast,
    body: UInt32,
    catalog: Catalog,
    mut plan: Plan,
    mut sources: List[Schema],
    mut scope: _Scope,
    ctes: _Bindings,
) raises -> Int:
    """Lowers one `SELECT ... FROM ... WHERE ...` block.

    The clause order is the one in the module docstring and the reason for it is
    there too. Every block brings its own `_Walk`, because an aggregate belongs
    to the block it was written in and two sides of a union are two queries.

    Args:
        ast: The arenas.
        body: The `STMT_QUERY`.
        catalog: What the table names are resolved against.
        plan: Where the nodes go.
        sources: One schema per scan, appended to in scan order.
        scope: Filled in with what the block's FROM put in reach, so that an
            ORDER BY written outside the block can qualify a name the same way
            the block's own clauses can.
        ctes: The CTE names in reach.

    Returns:
        The node the block produces.

    Raises:
        If the block is a shape this does not lower yet.
    """
    var query = ast.stmts[Int(body)]
    if query.kind != STMT_QUERY:
        raise Error(
            "firepanda lowers a SELECT block, a VALUES and a set operation over"
            " two so far, and a TABLE is a different node"
        )
    if ast.length(query.b) != 0:
        raise Error(
            "firepanda does not lower DISTINCT ON yet, which is a distinct on a"
            " key list under an order"
        )

    var clauses = query.children
    if ast.length(ast.slot(clauses, CLAUSE_WINDOW)) != 0:
        raise Error(
            "firepanda does not lower a WINDOW clause yet, and the same window"
            " written out after the OVER of each call does lower"
        )

    var from_clause = ast.slot(clauses, CLAUSE_FROM)

    var group_clause = ast.slot(clauses, CLAUSE_GROUP)
    var having = ast.slot(clauses, CLAUSE_HAVING)
    var items = ast.items(ast.slot(clauses, CLAUSE_PROJECTION))

    # Whether the query aggregates is decided before anything is lowered,
    # because it decides what node the projection sits on and a projection
    # cannot be built against a node that does not exist yet.
    var grouped = ast.length(group_clause) != 0 or having != NO_NODE
    if not grouped:
        for at in items:
            if _has_aggregate(ast, ast.stmts[Int(at)].a):
                grouped = True
                break

    var walk = _Walk()
    var at: Int
    var schema = Schema()
    var origin = List[Int]()
    if from_clause == NO_NODE:
        # A query with no FROM still has to project over something, and one row
        # of one constant is the smallest something there is. The projection
        # above drops the column again, so the value in it is never read and
        # only its row count matters.
        at = plan.values([plan.exprs.literal(Value(Int64(0)))], ["__row"])
    else:
        var source = _from(
            ast, from_clause, catalog, plan, sources, scope, ctes
        )
        at = source.at
        schema = Schema(copy=source.schema)
        origin = source.origin.copy()

    # Not called `where`, which the formatter reads as the start of a
    # parameter constraint and then cannot parse the rest of the file.
    var restriction = ast.slot(clauses, CLAUSE_WHERE)

    # An uncorrelated subquery written as a value is taken out and cross joined
    # on here, above the FROM and below everything else, so that every clause
    # under the aggregate can read the column it answered. The select list is
    # only searched when the query does not aggregate, because there the
    # projection sits straight on this and the column reaches it. Above an
    # aggregate it does not, and the refusal in `_lower_expr` says so.
    var found = List[UInt32]()
    _scalars(ast, restriction, found)
    if not grouped:
        for one in items:
            _scalars(ast, ast.stmts[Int(one)].a, found)
    for i in range(len(found)):
        at = _scalar_join(ast, found[i], catalog, plan, sources, ctes, walk, at)

    # An `IN` over a subquery goes the same way, as a mark join rather than a
    # cross join. The ones the `WHERE` is the `AND` of are left alone, because
    # each of those is a semi join below and a semi join is the cheaper answer
    # to the same question. What is left is every `IN` written as a value, which
    # includes every `NOT IN` wherever it is written.
    var asking = List[UInt32]()
    if restriction != NO_NODE:
        var parts = List[UInt32]()
        _conjuncts(ast, restriction, parts)
        for i in range(len(parts)):
            if _asks(ast, parts[i]) == NO_NODE:
                _marks(ast, parts[i], asking)
    if not grouped:
        for one in items:
            _marks(ast, ast.stmts[Int(one)].a, asking)
    for i in range(len(asking)):
        at = _mark_join(
            ast, asking[i], catalog, plan, sources, ctes, walk, scope, at
        )

    # An `EXISTS` written as a value goes the same way again, and back to the
    # cross join, because whether the subquery has a row in it is one answer for
    # the whole query rather than one per outer row. The ones the `WHERE` is the
    # `AND` of are left alone here too, for the same reason: a semi join is the
    # cheaper answer and a correlated one only has that answer.
    var wanting = List[UInt32]()
    if restriction != NO_NODE:
        var parts = List[UInt32]()
        _conjuncts(ast, restriction, parts)
        for i in range(len(parts)):
            if _asks(ast, parts[i]) == NO_NODE:
                _askings(ast, parts[i], wanting)
    if not grouped:
        for one in items:
            _askings(ast, ast.stmts[Int(one)].a, wanting)
    for i in range(len(wanting)):
        at = _exists_value(
            ast, wanting[i], catalog, plan, sources, ctes, walk, at
        )

    # A quantified comparison other than the two that are an `IN` goes to the
    # cross join as well, and for the same reason: the smallest and the largest
    # row of the subquery are the same two values whichever outer row is asking.
    # The comparison against them is not put here, because `x` is on the other
    # side of the join, so only the row is built and the comparison is made
    # where it was written.
    var ranged = List[UInt32]()
    if restriction != NO_NODE:
        var parts = List[UInt32]()
        _conjuncts(ast, restriction, parts)
        for i in range(len(parts)):
            _comparisons(ast, parts[i], ranged)
    if not grouped:
        for one in items:
            _comparisons(ast, ast.stmts[Int(one)].a, ranged)
    for i in range(len(ranged)):
        at = _quantified_value(
            ast, ranged[i], catalog, plan, sources, ctes, walk, at
        )

    if restriction != NO_NODE:
        # An `IN` or a correlated `EXISTS` over a subquery is a join rather than
        # a predicate, so the WHERE is split on `AND` and the parts that are one
        # are taken out and put above whatever is left. Splitting only when
        # there is one keeps every other query's plan the shape it already was,
        # which is one filter holding the condition as it was written.
        var conjuncts = List[UInt32]()
        _conjuncts(ast, restriction, conjuncts)
        var asked = List[UInt32]()
        var flipped = List[Bool]()
        for i in range(len(conjuncts)):
            var part = _asks(ast, conjuncts[i])
            if part != NO_NODE:
                asked.append(part)
                flipped.append(part != conjuncts[i])
        if len(asked) == 0:
            at = plan.filter(
                at, _lower_expr(ast, restriction, plan, walk, scope, False)
            )
        else:
            var tested = -1
            for i in range(len(conjuncts)):
                if _asks(ast, conjuncts[i]) != NO_NODE:
                    continue
                var one = _lower_expr(
                    ast, conjuncts[i], plan, walk, scope, False
                )
                if tested < 0:
                    tested = one
                else:
                    tested = plan.exprs.call("and", [tested, one], True)
            # The rest of the condition goes under the joins rather than over
            # them, because a semi join hands out the left side unchanged and
            # so the two commute, and testing first is the side with fewer rows
            # to probe with.
            if tested >= 0:
                at = plan.filter(at, tested)
            var source = _From(at, Schema(copy=schema), origin.copy())
            for i in range(len(asked)):
                if ast.exprs[Int(asked[i])].kind == EXPR_EXISTS:
                    source = _exists_join(
                        ast,
                        asked[i],
                        catalog,
                        plan,
                        sources,
                        scope,
                        ctes,
                        source^,
                        flipped[i],
                    )
                    continue
                source = _in_join(
                    ast,
                    asked[i],
                    catalog,
                    plan,
                    sources,
                    scope,
                    walk,
                    ctes,
                    source^,
                )
            at = source.at

    var keys = List[Int]()
    var key_names = List[String]()
    if grouped:
        for entry in ast.items(group_clause):
            var group = ast.stmts[Int(entry)]
            if group.b != GROUP_EXPRESSION:
                raise Error(
                    "firepanda lowers a GROUP BY of plain expressions so far,"
                    " and GROUPING SETS, CUBE, ROLLUP and GROUP BY ALL are"
                    " masks on one aggregate the plan cannot carry yet"
                )
            keys.append(_lower_expr(ast, group.a, plan, walk, scope, False))
            key_names.append(_name_of(ast, group.a, len(key_names)))

    # The select list is lowered before the aggregate is built, because
    # lowering it is what finds the aggregates the node has to compute.
    var outputs = List[Int]()
    var names = List[String]()
    for i in range(len(items)):
        var item = ast.stmts[Int(items[i])]
        if ast.exprs[Int(item.a)].kind == EXPR_STAR:
            if from_clause == NO_NODE:
                raise Error(
                    "a star in a SELECT with no FROM, and there is nothing for"
                    " it to stand for"
                )
            _expand(
                ast,
                item.a,
                schema,
                origin,
                plan,
                walk,
                scope,
                grouped,
                outputs,
                names,
            )
            continue
        outputs.append(_lower_expr(ast, item.a, plan, walk, scope, grouped))
        if item.payload != NO_NODE:
            names.append(ast.text(item.payload))
        else:
            names.append(_name_of(ast, item.a, i))

    # A star whose EXCLUDE names every column it stood for leaves nothing, and
    # a select list of nothing is not a query. DuckDB's wording, since this is
    # the same thing it refuses.
    if len(outputs) == 0:
        raise Error(empty_select_list())

    var predicate = -1
    if having != NO_NODE:
        predicate = _lower_expr(ast, having, plan, walk, scope, True)

    # Lowered here and filtered on further down, because a QUALIFY is where the
    # windows it reads are written and the node that computes them has to be
    # built before anything reads it.
    var qualify = -1
    var qualifying = ast.slot(clauses, CLAUSE_QUALIFY)
    if qualifying != NO_NODE:
        var before = len(walk.windows)
        qualify = _lower_expr(ast, qualifying, plan, walk, scope, grouped)
        if len(walk.windows) == before and len(walk.windows) == 0:
            raise Error(
                "a QUALIFY over a query with no window function in it, and the"
                " same condition written in WHERE does lower"
            )

    if grouped:
        var aggs = walk.aggs.copy()
        var agg_names = walk.agg_names.copy()
        var both = key_names.copy()
        for i in range(len(agg_names)):
            both.append(String(agg_names[i]))
        at = plan.aggregate(at, keys^, aggs^, both^)

    if predicate >= 0:
        at = plan.filter(at, predicate)

    at = _windows(plan, at, walk)

    if qualify >= 0:
        at = plan.filter(at, qualify)

    at = plan.project(at, outputs^, names^)

    if (query.a & SELECT_DISTINCT) != 0:
        at = plan.distinct(at, List[Int]())

    return at


def _windows(mut plan: Plan, at: Int, walk: _Walk) raises -> Int:
    """Puts one `WINDOW` node over the block for each partitioning it uses.

    One node per partitioning rather than one node for all of them, because a
    node is one grouping pass and two windows over different keys cannot share
    one. The nodes stack in the order the partitionings were first written, and
    which one a window landed in does not matter above them, since a window node
    adds its columns to what is below it and the projection reads all of them
    back by name.

    Args:
        plan: Where the nodes go.
        at: The node the windows are computed over.
        walk: The windows the block found.

    Returns:
        The topmost node, which is `at` when the block has no window in it.

    Raises:
        Only what the builder raises.
    """
    var done = List[Bool](length=len(walk.windows), fill=False)
    var out = at
    for i in range(len(walk.windows)):
        if done[i]:
            continue
        var exprs = List[Int]()
        var names = List[String]()
        for j in range(i, len(walk.windows)):
            if not done[j] and walk.window_keys[j] == walk.window_keys[i]:
                done[j] = True
                exprs.append(walk.windows[j])
                names.append(String(walk.window_names[j]))
        out = plan.window(out, exprs^, names^)
    return out


def _name_of(ast: Ast, at: UInt32, place: Int) raises -> String:
    """What an output column is called when the query did not say.

    A bare column keeps its own name, which is what makes `SELECT a FROM t` come
    back with a column called `a`, and a qualified one keeps the last part of
    it, so `SELECT t.a FROM t` comes back with a column called `a` too and not
    one called `t.a`. Anything else gets a name from its position,
    because DuckDB's own default names an expression after the text it was
    written as and reproducing that needs the printer over the original tokens,
    which is a thing to do once rather than here.

    Args:
        ast: The arenas.
        at: The expression.
        place: Where it sits in the list, counting from zero.

    Returns:
        The name.

    Raises:
        If the expression is not there.
    """
    if at == NO_NODE:
        raise Error("an expression that is not there")
    var node = ast.exprs[Int(at)]
    var parts = ast.length(node.children)
    if node.kind == EXPR_COLUMN and parts != 0:
        return String(ast.text(ast.at(node.children, parts - 1)))
    return String("__expr_", place)


def _expand(
    ast: Ast,
    at: UInt32,
    schema: Schema,
    origin: List[Int],
    mut plan: Plan,
    mut walk: _Walk,
    scope: _Scope,
    grouped: Bool,
    mut outputs: List[Int],
    mut names: List[String],
) raises:
    """Expands a `*` into one output per column the FROM produces.

    A qualified star keeps the columns of one relation and drops the rest, and
    the three modifiers are applied in the order the grammar forces them to be
    written: exclude, then replace, then rename. The rules for what a modifier
    may name and what happens when two of them name the same column are in
    `firepanda/sql/star.mojo`, which is also where the wording of every refusal
    here comes from, so the two stages that expand a star do not disagree about
    what the query said.

    What this does not take is a qualified star in front of a subquery's name.
    A derived table is not a relation and its columns come through with no
    number on them, so there is nothing here to keep them apart from the
    columns of the source beside them.

    Args:
        ast: The arenas.
        at: The `EXPR_STAR`.
        schema: What the FROM produces.
        origin: Which relation each of those columns came from.
        plan: Where the lowered expressions go.
        walk: The aggregates found so far, for a `REPLACE` that holds one.
        scope: What the FROM put in reach, for the qualifier.
        grouped: Whether the statement has a `GROUP BY`.
        outputs: Where the expressions go.
        names: Where their names go.

    Raises:
        If the qualifier names nothing in the FROM or names a subquery, if a
        modifier names a column the star does not stand for, or if two
        modifiers name the same column.
    """
    var node = ast.exprs[Int(at)]

    # Every modifier name is one identifier. A dotted one is refused while the
    # AST is built, since the node has nowhere to put the two halves, so
    # nothing here has a qualifier to match and every `Target` is written bare.
    var excluded = List[Target]()
    for i in range(ast.length(node.a)):
        excluded.append(Target("", ast.text(ast.at(node.a, i))))
    var replaced = List[Replacement]()
    var i = 0
    while i + 1 < ast.length(node.b):
        replaced.append(
            Replacement(
                Target("", ast.text(ast.at(node.b, i))), ast.at(node.b, i + 1)
            )
        )
        i += 2
    var renamed = List[Renaming]()
    i = 0
    while i + 1 < ast.length(node.payload):
        renamed.append(
            Renaming(
                Target("", ast.text(ast.at(node.payload, i))),
                ast.text(ast.at(node.payload, i + 1)),
            )
        )
        i += 2
    check(excluded, replaced, renamed)

    var qualified = False
    var only = 0
    var parts = ast.length(node.children)
    if parts != 0:
        if parts != 1:
            raise Error(
                "firepanda reads a star as a name and a star so far, and a"
                " third part is a schema, which needs the catalog to tell from"
                " a table"
            )
        var qualifier = String(ast.text(ast.at(node.children, 0)))
        var found = scope.find(qualifier)
        if found == NOT_IN_REACH:
            raise Error(
                String(
                    "nothing in this query is called '",
                    qualifier,
                    "', and the FROM brought ",
                    scope.written(),
                )
            )
        if found == DERIVED:
            raise Error(
                String(
                    "firepanda does not lower '",
                    qualifier,
                    (
                        ".*' yet, because a subquery in a FROM is not a"
                        " relation and its columns arrive with no number"
                        " saying which source they came from"
                    ),
                )
            )
        qualified = True
        only = found

    var used_exclude = List[Bool](length=len(excluded), fill=False)
    var used_replace = List[Bool](length=len(replaced), fill=False)
    for at_column in range(len(schema)):
        if qualified and origin[at_column] != only:
            continue
        var column = String(schema[at_column].name)

        var dropped = False
        for entry in range(len(excluded)):
            if excluded[entry].matches("", column):
                used_exclude[entry] = True
                dropped = True
        if dropped:
            continue

        # A REPLACE that matches twice replaces the first column and drops the
        # second one, which is DuckDB losing a column without saying so and is
        # reproduced rather than fixed, for the reason star.mojo gives.
        var stood = NOT_REPLACED
        var taken = False
        for entry in range(len(replaced)):
            if not replaced[entry].target.matches("", column):
                continue
            if used_replace[entry]:
                taken = True
                break
            used_replace[entry] = True
            stood = replaced[entry].node
        if taken:
            continue

        var called = String(column)
        for entry in renamed:
            if entry.target.matches("", column):
                called = String(entry.name)

        if stood != NOT_REPLACED:
            outputs.append(_lower_expr(ast, stood, plan, walk, scope, grouped))
        elif origin[at_column] == UNBOUND:
            # Every column that came from a table says which one. A join of two
            # tables that share a column name puts both of them in the star and
            # an unqualified reference to either would be refused, and a USING
            # join hands out one of a pair while the node below it still
            # produces both, so in neither case is the name on its own enough.
            # Which position it is remains binding's to work out, and that is
            # the one thing this stage cannot know. A column a projection
            # computed has no relation to name and goes on as it is.
            outputs.append(plan.exprs.column(column^))
        else:
            outputs.append(plan.exprs.column_of(origin[at_column], column^))
        names.append(called^)

    for entry in range(len(excluded)):
        if not used_exclude[entry]:
            raise Error(not_in_from("EXCLUDE", excluded[entry]))
    for entry in range(len(replaced)):
        if not used_replace[entry]:
            raise Error(not_in_from("REPLACE", replaced[entry].target))


def _modifiers(
    ast: Ast,
    at: UInt32,
    mut plan: Plan,
    mut walk: _Walk,
    scope: _Scope,
    input: Int,
) raises -> Int:
    """Puts the `ORDER BY`, `LIMIT` and `OFFSET` on top of a plan.

    Args:
        ast: The arenas.
        at: The `STMT_MODIFIERS`.
        plan: Where the lowered expressions go.
        walk: The aggregates found so far.
        scope: What the FROM put in reach, for a qualified name.
        input: What they apply to.

    Returns:
        The topmost node built.

    Raises:
        If a modifier is a shape this does not lower yet.
    """
    var node = ast.stmts[Int(at)]
    var out = input

    var orders = ast.items(node.children)
    if len(orders) != 0:
        var keys = List[Int]()
        var descending = List[Bool]()
        var nulls_last = List[Bool]()
        for i in range(len(orders)):
            var entry = ast.stmts[Int(orders[i])]
            if entry.a == NO_NODE:
                raise Error(
                    "firepanda does not lower ORDER BY ALL yet, which sorts on"
                    " every output column in order"
                )
            keys.append(_lower_expr(ast, entry.a, plan, walk, scope, False))
            descending.append(entry.b == SORT_DESCENDING)
            # DuckDB puts the missing values last when the sort goes up and
            # first when it goes down, so the default follows the direction
            # rather than being one answer for both.
            if entry.payload == NULLS_LAST:
                nulls_last.append(True)
            elif entry.payload == NO_NODE or entry.payload == 0:
                nulls_last.append(entry.b != SORT_DESCENDING)
            else:
                nulls_last.append(False)
        out = plan.sort(out, keys^, descending^, nulls_last^)

    if (node.payload & LIMIT_PERCENT) != 0:
        raise Error(
            "firepanda does not lower LIMIT n PERCENT yet, which needs the row"
            " count before it knows how many rows it keeps"
        )

    var offset = 0
    if node.b != NO_NODE:
        offset = _constant_count(ast, node.b, "an OFFSET")
    var length = NO_LIMIT
    if node.a != NO_NODE:
        length = _constant_count(ast, node.a, "a LIMIT")
    if offset != 0 or length != NO_LIMIT:
        out = plan.limit(out, offset, length)
    return out


def _constant_count(ast: Ast, at: UInt32, what: StringSlice) raises -> Int:
    """Reads a `LIMIT` or an `OFFSET` that has to be a constant.

    The plan holds both as integers rather than as expressions, so an expression
    here is a refusal rather than a lowering. A query writing one is rare and a
    query writing a number is not.

    Args:
        ast: The arenas.
        at: The expression.
        what: Which clause, for the message.

    Returns:
        The number.

    Raises:
        If the expression is not a non negative integer literal.
    """
    var node = ast.exprs[Int(at)]
    if node.kind != EXPR_LITERAL or node.b != LITERAL_NUMBER:
        raise Error(
            String(
                "firepanda lowers ",
                what,
                (
                    " written as a number, and the plan holds it as one rather"
                    " than as an expression"
                ),
            )
        )
    var text = ast.text(node.payload)
    var value = _number(text)
    var count = Int(value.bits)
    if count < 0:
        raise Error(String(what, " of ", text, " is not a count"))
    return count
