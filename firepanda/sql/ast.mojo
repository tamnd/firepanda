"""The expression AST: arenas, node layout and the text they carry.

The parse tree from `matcher.mojo` has one node per grammar rule, which for an
ordinary comparison is a dozen nodes of pure syntax. Binding against that would
tie every later stage to grammar rule names, so a grammar bump would break the
binder rather than a small translation layer. This is the shape that layer
produces instead. See docs/specs/sql/05-ast-and-binder.md section 2.

Three things decide the layout.

Nodes are fixed size and live in an arena, referenced by index rather than by
pointer. The alternative, a variant with an inline `List` of children, allocates
per node and fights ownership for no benefit. A node that needs a variable
number of children stores a run in a side list instead, and a run is a count
followed by that many entries, so the node itself stays one size.

Index 0 is the null node in every arena, the same convention `matcher.mojo`
uses, so a missing operand is 0 and no caller needs a separate flag to say a
field is absent. Building a node returns an index that is never 0.

The AST owns its text. A token points into the query string and keeps its
quotes, its escapes and its original case, because that is what makes a token
twelve bytes. Nothing downstream wants any of that, so text is decoded once on
the way in and interned here, and an `Ast` is therefore independent of the
query it came from.

Every node carries a token index. Not for debugging: every binder and runtime
error renders a caret, and a node with no position produces the error message
nobody can act on.

There are three arenas because there are three kinds of thing: an expression, a
table reference and a statement. They refer to each other in both directions. A
`FROM` holds a subquery, which is a statement, and a `WHERE` holds an `EXISTS`,
which is an expression holding a statement. An index is only meaningful with
the arena it came from, so every field says which arena it points into.

A node whose kind needs more parts than the four fields hold keeps them in a
fixed length run instead, and the entries are named by constant. The query node
is the only one that does this, because seven clauses do not fit in four fields
and splitting a `SELECT` across two nodes to make them fit would be worse.

What is not here yet is `UNPIVOT`, and the statements that are not a `SELECT`.
They arrive with the rest of S2.
"""

comptime NO_NODE: UInt32 = 0
"""The null node, and the empty run.

Index 0 of every arena is a node nobody builds, so an absent operand is 0 and a
node with no variadic children has a `children` of 0.
"""


comptime EXPR_LITERAL: UInt8 = 1
"""A constant.

`payload` is the interned text of the value and `b` is one of the `LITERAL_`
tags saying how to print it. The text is the decoded value, so a string literal
holds what it means rather than what was typed, and the printer puts the quotes
back.
"""

comptime EXPR_COLUMN: UInt8 = 2
"""A column reference, qualified or not.

`children` is a run of interned name parts, outermost first, so `a` is one part
and `s.t.c` is three. Nothing here says whether a part named a table or a
struct field, because deciding that needs the catalog and this stage has none.
"""

comptime EXPR_STAR: UInt8 = 3
"""`*`, or `t.*`, with the three modifiers DuckDB allows on it.

`children` is a run of name parts qualifying the star, empty for a bare one.
`a` is a run of interned names for `EXCLUDE`. `b` is a run of alternating name
and expression for `REPLACE`. `payload` is a run of alternating name and name
for `RENAME`. All four are runs and all four may be empty, which is the whole
of `SELECT * EXCLUDE (a) REPLACE (x + 1 AS b) RENAME (c AS d)`.
"""

comptime EXPR_FUNCTION: UInt8 = 4
"""A call, `f(x)`.

`payload` is a run of interned name parts, so a qualified `main.f` keeps its
qualification. `children` is a run of argument expressions. `a` is a bit set of
the `CALL_` flags. `b` is the `EXPR_WINDOW` the `OVER` names, and is 0 for a
call with no `OVER` on it, which is most of them.

An operator is not one of these even where the grammar spells it as one. The
printer has to know that `+` goes between its operands and `f` goes before
them, and that is the difference the kind records.
"""

comptime EXPR_UNARY: UInt8 = 5
"""A prefix operator, `-x` or `NOT x`.

`payload` is the interned operator text and `a` is the operand. The text is the
operator as SQL spells it, so the printer needs no table to print it back, and
a word operator like `NOT` prints with a space after it because the text says
it is a word.
"""

comptime EXPR_BINARY: UInt8 = 6
"""An infix operator, `a + b` or `a AND b`.

`payload` is the interned operator text, `a` is the left operand and `b` is the
right one. Precedence is not stored, because by the time a node exists the tree
already has the shape precedence gave it, and the printer parenthesizes every
operand rather than recomputing what it could leave out.
"""

comptime EXPR_CAST: UInt8 = 7
"""`CAST(x AS t)`, and `TRY_CAST` when `b` is 1.

`a` is the operand and `payload` is the interned type text. The type is text
and not a parsed type, because resolving a type name needs the type registry
and this stage resolves nothing.
"""

comptime EXPR_CASE: UInt8 = 8
"""`CASE`, both the searched form and the simple one.

`a` is the operand a simple `CASE x WHEN` compares against, or 0 for the
searched form. `children` is a run of alternating condition and result, so it
always has an even length. `b` is the `ELSE` result, or 0.
"""

comptime EXPR_BETWEEN: UInt8 = 9
"""`x BETWEEN lo AND hi`.

`a` is the operand, `children` is a run of exactly two, the bounds in order,
and `payload` is 1 for `NOT BETWEEN`. It is its own kind rather than two
comparisons joined by `AND` because rewriting it here would evaluate the
operand twice and print back something the user did not write.
"""

comptime EXPR_IN: UInt8 = 10
"""`x IN (a, b, c)`.

`a` is the operand, `children` is a run of candidates and `payload` is 1 for
`NOT IN`. The form where the right side is a subquery is a different kind and
arrives with the statement arena.
"""

comptime EXPR_LIST: UInt8 = 11
"""A list constructor, `[1, 2, 3]`.

`children` is a run of elements, which may be empty.
"""

comptime EXPR_STRUCT: UInt8 = 12
"""A struct constructor, `{'a': 1}`.

`children` is a run of alternating interned field name and value expression, so
it always has an even length. The names are stored as text rather than as
literal nodes because a struct field name is not an expression, and letting it
be one would mean the printer had to decide when a literal was really a name.
"""

comptime EXPR_COLLATE: UInt8 = 13
"""`x COLLATE c`.

`a` is the operand and `payload` is the interned collation name.
"""

comptime EXPR_PARAMETER: UInt8 = 14
"""A prepared statement parameter, `?`, `$1` or `$name`.

`payload` is the interned text after the sigil, empty for a bare `?`, and `b`
is the interned sigil itself, so the printer puts back the form that was
written. The two forms are not interchangeable in DuckDB: a statement may use
positional parameters or named ones, not both.
"""


comptime EXPR_SUBQUERY: UInt8 = 15
"""A scalar subquery, `(SELECT ...)` where a value is wanted.

`a` is a statement index. Nothing here checks that the statement gives back one
column and at most one row, because that needs to know what the columns are and
this stage resolves nothing.
"""

comptime EXPR_EXISTS: UInt8 = 16
"""`EXISTS (SELECT ...)`, and `NOT EXISTS` when `b` is 1.

`a` is a statement index.
"""

comptime EXPR_IN_SUBQUERY: UInt8 = 17
"""`x IN (SELECT ...)`, and `NOT IN` when `payload` is 1.

`a` is the operand and `b` is a statement index. It is a different kind from
`EXPR_IN` because the right side lives in a different arena, and one kind
holding an index that means one of two things is how a wrong arena read gets
written.
"""


comptime EXPR_WINDOW: UInt8 = 18
"""What follows `OVER`, and what `WINDOW w AS (...)` defines.

`children` is a run of the `PARTITION BY` expressions, `a` is a run of
`STMT_ORDER` nodes for the window's own `ORDER BY`, `b` is the `EXPR_FRAME`,
and `payload` is the interned name of the window this one starts from, which
is 0 for the usual case of a window written out in full.

It is an expression rather than a statement because the only thing that holds
one is a call, and a call is an expression. `OVER w` and `OVER (w)` both come
out as a node with nothing but the name on it, because the two spellings mean
the same thing and DuckDB's grammar gives them separate rules only so it can
tell the parentheses apart.
"""

comptime EXPR_FRAME: UInt8 = 19
"""The `ROWS`, `RANGE` or `GROUPS` clause inside a window.

`a` is the expression on the start bound and `b` is the one on the end bound,
either of which is 0 when that bound is written in keywords rather than as a
count. `payload` is the tags, packed by `frame_tags`.

A bound is a tag and an expression rather than a node of its own, because a
bound has nothing else in it and a node per bound would be two more arena
entries for every window that names a frame.
"""


comptime FRAME_ROWS: UInt32 = 1
"""`ROWS`, which counts rows."""

comptime FRAME_RANGE: UInt32 = 2
"""`RANGE`, which counts by the value of the `ORDER BY` expression."""

comptime FRAME_GROUPS: UInt32 = 3
"""`GROUPS`, which counts peer groups."""


comptime BOUND_NONE: UInt32 = 0
"""No such bound, which is what the end bound of a frame with no `BETWEEN` is.
"""

comptime BOUND_PRECEDING: UInt32 = 1
"""`n PRECEDING`, where `n` is the bound's expression."""

comptime BOUND_FOLLOWING: UInt32 = 2
"""`n FOLLOWING`, where `n` is the bound's expression."""

comptime BOUND_UNBOUNDED_PRECEDING: UInt32 = 3
"""`UNBOUNDED PRECEDING`, which carries no expression."""

comptime BOUND_UNBOUNDED_FOLLOWING: UInt32 = 4
"""`UNBOUNDED FOLLOWING`, which carries no expression."""

comptime BOUND_CURRENT_ROW: UInt32 = 5
"""`CURRENT ROW`, which carries no expression."""


comptime EXCLUDE_NONE: UInt32 = 0
"""No `EXCLUDE`, which is the same as `EXCLUDE NO OTHERS` and is the default."""

comptime EXCLUDE_CURRENT_ROW: UInt32 = 1
"""`EXCLUDE CURRENT ROW`."""

comptime EXCLUDE_GROUP: UInt32 = 2
"""`EXCLUDE GROUP`."""

comptime EXCLUDE_TIES: UInt32 = 3
"""`EXCLUDE TIES`."""

comptime EXCLUDE_NO_OTHERS: UInt32 = 4
"""`EXCLUDE NO OTHERS`, which is the default said out loud.

It is a tag of its own rather than `EXCLUDE_NONE`, because the printer writes
back what was written and dropping the words would be a change to the text for
no reason.
"""


comptime _FRAME_FIELD: UInt32 = 15
"""The mask one packed frame tag fits in, four bits."""

comptime _FRAME_MODE_SHIFT: UInt32 = 0
comptime _FRAME_START_SHIFT: UInt32 = 4
comptime _FRAME_END_SHIFT: UInt32 = 8
comptime _FRAME_EXCLUDE_SHIFT: UInt32 = 12


def frame_tags(
    mode: UInt32, start: UInt32, end: UInt32, exclude: UInt32
) -> UInt32:
    """Packs the four tags of a frame into one field.

    Four tags and four usable fields on an `Expr`, two of which the bound
    expressions need, so the tags share one. None of them has more than six
    values, so four bits each is room to spare and the packing needs no
    thought when a value gets added.

    Args:
        mode: One of the `FRAME_` constants.
        start: One of the `BOUND_` constants.
        end: One of the `BOUND_` constants, `BOUND_NONE` for no `BETWEEN`.
        exclude: One of the `EXCLUDE_` constants.

    Returns:
        The packed value, for an `EXPR_FRAME` payload.
    """
    return (
        (mode << _FRAME_MODE_SHIFT)
        | (start << _FRAME_START_SHIFT)
        | (end << _FRAME_END_SHIFT)
        | (exclude << _FRAME_EXCLUDE_SHIFT)
    )


def frame_mode(tags: UInt32) -> UInt32:
    """Reads the framing out of a packed frame payload.

    Args:
        tags: An `EXPR_FRAME` payload.

    Returns:
        One of the `FRAME_` constants.
    """
    return (tags >> _FRAME_MODE_SHIFT) & _FRAME_FIELD


def frame_start(tags: UInt32) -> UInt32:
    """Reads the start bound out of a packed frame payload.

    Args:
        tags: An `EXPR_FRAME` payload.

    Returns:
        One of the `BOUND_` constants.
    """
    return (tags >> _FRAME_START_SHIFT) & _FRAME_FIELD


def frame_end(tags: UInt32) -> UInt32:
    """Reads the end bound out of a packed frame payload.

    Args:
        tags: An `EXPR_FRAME` payload.

    Returns:
        One of the `BOUND_` constants, `BOUND_NONE` for no `BETWEEN`.
    """
    return (tags >> _FRAME_END_SHIFT) & _FRAME_FIELD


def frame_exclude(tags: UInt32) -> UInt32:
    """Reads the exclusion out of a packed frame payload.

    Args:
        tags: An `EXPR_FRAME` payload.

    Returns:
        One of the `EXCLUDE_` constants.
    """
    return (tags >> _FRAME_EXCLUDE_SHIFT) & _FRAME_FIELD


comptime LITERAL_NULL: UInt32 = 0
"""`NULL`, whose interned text is empty because the kind is the whole value."""

comptime LITERAL_BOOLEAN: UInt32 = 1
"""`TRUE` or `FALSE`, interned as the word itself in upper case."""

comptime LITERAL_NUMBER: UInt32 = 2
"""A number, interned as the digits with the underscores already taken out.

Whether it is an integer, a decimal or a double is a typing decision and this
stage makes none, so the text is kept and the binder reads it.
"""

comptime LITERAL_STRING: UInt32 = 3
"""A string, interned as the decoded value with no quotes and no escapes."""


comptime CALL_DISTINCT: UInt32 = 1
"""`f(DISTINCT x)`."""

comptime CALL_STAR: UInt32 = 2
"""`count(*)`, which has no arguments rather than one star argument."""


struct Expr(ImplicitlyCopyable, Movable):
    """One expression node, twenty one bytes, holding no pointers.

    What `a`, `b`, `children` and `payload` mean depends on `kind`, and every
    kind says so in its own docstring. Four fields is what the widest kind
    needs, which is `EXPR_STAR` with its three modifier runs and its
    qualification.
    """

    var kind: UInt8
    """One of the `EXPR_` constants."""

    var a: UInt32
    """An operand index, a run or a flag set, by kind."""

    var b: UInt32
    """A second operand index, a run or a tag, by kind."""

    var children: UInt32
    """A run in the side list, or 0 for none."""

    var token: UInt32
    """The token this node starts at, for the caret in an error."""

    var payload: UInt32
    """A string index or a run, by kind."""

    def __init__(
        out self,
        kind: UInt8,
        token: UInt32,
        a: UInt32 = NO_NODE,
        b: UInt32 = NO_NODE,
        children: UInt32 = NO_NODE,
        payload: UInt32 = NO_NODE,
    ):
        """Builds a node.

        Args:
            kind: One of the `EXPR_` constants.
            token: The token the node starts at.
            a: By kind.
            b: By kind.
            children: A run, or 0.
            payload: By kind.
        """
        self.kind = kind
        self.a = a
        self.b = b
        self.children = children
        self.token = token
        self.payload = payload


comptime REF_TABLE: UInt8 = 1
"""A named table, `t` or `s.t`.

`children` is a run of interned name parts, outermost first. `payload` is the
alias run. Whether the name is a table, a view or a file that reads as one is
the catalog's business and not this stage's.
"""

comptime REF_SUBQUERY: UInt8 = 2
"""A subquery in a `FROM`, `(SELECT ...) t`.

`a` is a statement index, `payload` is the alias run and `b` is 1 for
`LATERAL`.
"""

comptime REF_FUNCTION: UInt8 = 3
"""A table function, `range(10) t`.

`a` is a run of interned name parts, `children` is a run of argument
expressions, `payload` is the alias run and `b` is 1 for `LATERAL`.
"""

comptime REF_JOIN: UInt8 = 4
"""A join with an `ON`, or with no condition at all.

`a` is the left reference, `b` is the right one and `payload` is the interned
join text as SQL spells it, so `LEFT OUTER JOIN` and `POSITIONAL JOIN` need no
table to print back. `children` is a run of one holding the `ON` expression,
and is empty for a `CROSS`, `NATURAL` or `POSITIONAL` join, which take no
condition.
"""

comptime REF_JOIN_USING: UInt8 = 5
"""A join with a `USING`.

`a`, `b` and `payload` are what `REF_JOIN` has. `children` is a run of interned
column names. It is its own kind because `children` holds names here and an
expression there, and one kind holding either would be a wrong read waiting to
happen.
"""


comptime REF_PARENS: UInt8 = 6
"""A table reference the query wrote in parentheses.

`a` is the reference inside and `payload` is the alias run. It is kept rather
than dropped because the parentheses are the only thing that can put a join on
the right of another join, and a printer that guessed where to put them back
would be a second implementation of the same rule.
"""


comptime STMT_SELECT: UInt8 = 1
"""A whole `SELECT` statement, which is what a subquery holds.

`a` is the query node, `b` is the modifiers node or 0, `children` is a run of
`STMT_CTE` nodes for the `WITH`, and `payload` is 1 for `WITH RECURSIVE`. The
grammar puts the `WITH` and the `ORDER BY` outside the set operation chain, so
they sit here and not on the query node, and a parenthesized inner select is
one of these in its own right.
"""

comptime STMT_QUERY: UInt8 = 2
"""One `SELECT ... FROM ... WHERE ...` block.

`a` is a bit set of the `SELECT_` flags. `b` is a run of `DISTINCT ON`
expressions. `children` is the clause run, which always has `CLAUSE_SLOTS`
entries, read with the `CLAUSE_` constants.
"""

comptime STMT_SET_OPERATION: UInt8 = 3
"""`UNION`, `INTERSECT` or `EXCEPT` between two query nodes.

`a` is the left side, `b` is the right one and `payload` is the interned
operator text as SQL spells it, so `UNION ALL BY NAME` is one string and the
printer needs no table. Storing the text rather than a tag with flags is the
same choice `EXPR_BINARY` makes for the same reason.
"""

comptime STMT_VALUES: UInt8 = 4
"""`VALUES (1, 2), (3, 4)`.

`children` is a run of rows, and each entry is itself a run handle holding that
row's expressions. A run of runs rather than one flat run, because the rows do
not have to be the same length and a flat run could not say where one ended.
"""

comptime STMT_TABLE: UInt8 = 5
"""`TABLE t`, which is `SELECT * FROM t` written short.

`children` is a run of interned name parts. It stays its own statement rather
than being rewritten into a query, because rewriting here would print back
something the user did not write.
"""

comptime STMT_MODIFIERS: UInt8 = 6
"""The `ORDER BY`, `LIMIT` and `OFFSET` that trail a statement.

`children` is a run of `STMT_ORDER` nodes, `a` is the limit expression or 0,
`b` is the offset expression or 0, and `payload` is a bit set of the `LIMIT_`
flags. `FETCH FIRST n ROWS ONLY` and `LIMIT n` land on the same node, since
they mean the same thing and the printer is not a formatter.
"""

comptime STMT_CTE: UInt8 = 7
"""One entry of a `WITH`.

`payload` is the interned name, `a` is the statement, `children` is a run of
interned column aliases and `b` is one of the `MATERIALIZE_` tags.
"""

comptime STMT_ITEM: UInt8 = 8
"""One entry of a `SELECT` list.

`a` is the expression and `payload` is the interned alias, or 0 when it was
written without one.
"""

comptime STMT_ORDER: UInt8 = 9
"""One entry of an `ORDER BY`.

`a` is the expression, or 0 for `ORDER BY ALL`. `b` is one of the `SORT_` tags
and `payload` is one of the `NULLS_` tags. A default direction is kept as a
default rather than filled in with `ASC`, because which way the default goes is
a setting and this stage reads no settings.
"""

comptime STMT_GROUP: UInt8 = 10
"""One entry of a `GROUP BY`.

`b` is one of the `GROUP_` tags. For `GROUP_EXPRESSION` the expression is `a`,
and for `GROUP_SETS`, `GROUP_CUBE`, `GROUP_ROLLUP` and `GROUP_TUPLE` the nested
entries are a run of further `STMT_GROUP` nodes in `children`. `GROUP_ALL` and
`GROUP_EMPTY` carry nothing.
"""

comptime STMT_WINDOW: UInt8 = 11
"""One entry of a `WINDOW` clause, `w AS (PARTITION BY x)`.

`a` is the `EXPR_WINDOW` and `payload` is the interned name it is given.

It is not a `STMT_ITEM` with the name in the alias slot, even though the two
have the same shape, because a reader who finds a `STMT_ITEM` in a clause run
has every reason to think it is part of a `SELECT` list.
"""

comptime STMT_PIVOT: UInt8 = 12
"""A `PIVOT`, in the spelling that is a statement of its own.

`a` is the table reference being pivoted, `b` is a run of `STMT_PIVOT_ON`
nodes, `children` is a run of `STMT_ITEM` nodes for the `USING` aggregates, and
`payload` is a run of interned `GROUP BY` names. Every one of the three lists
can be empty, because the grammar makes all three optional.

DuckDB has two spellings and only one of them is here. `PIVOT t ON a USING
sum(x)` is this node. `FROM t PIVOT (sum(x) FOR a IN (1, 2))` is the standard
one, and it becomes this node inside a subquery reference, so `FROM (PIVOT t ON
a IN (1, 2) USING sum(x))` is what comes back out.

Normalizing that way round and not the other is forced rather than chosen. The
standard spelling requires an `IN` on every pivot column, this one does not, so
`PIVOT t ON a USING sum(x)` cannot be written in the standard form at all. A
printer that picked a spelling per query would be deciding which features each
one can carry, which is a rule nobody could read off the node.

`PIVOT_WIDER` is the same word as `PIVOT` and is not recorded, so a query that
wrote it gets `PIVOT` back.
"""

comptime STMT_PIVOT_ON: UInt8 = 13
"""One pivot column, which is one entry of an `ON` list.

`a` is the header expression, which every entry has. The rest say what the
column's values are, and at most one of them is set:

- `children` is a run of `STMT_ITEM` nodes for `a IN (1, 2)`
- `payload` is an interned name for `a IN an_enum`
- `b` is a statement index for `a IN (SELECT ...)`

All three being empty is the bare `ON a`, which asks DuckDB to find the values
by reading the column. `IN ()` is not a thing the grammar can write, so an
empty run and no run mean the same and there is nothing to tell apart.
"""


comptime CLAUSE_SLOTS: Int = 7
"""How many entries a query node's clause run always has."""

comptime CLAUSE_PROJECTION: Int = 0
"""The `SELECT` list, a run of `STMT_ITEM` nodes, empty for a bare `FROM`."""

comptime CLAUSE_FROM: Int = 1
"""The `FROM`, a run of table references, empty when there is no `FROM`."""

comptime CLAUSE_WHERE: Int = 2
"""The `WHERE` expression, or 0."""

comptime CLAUSE_GROUP: Int = 3
"""The `GROUP BY`, a run of `STMT_GROUP` nodes, empty when there is none."""

comptime CLAUSE_HAVING: Int = 4
"""The `HAVING` expression, or 0."""

comptime CLAUSE_QUALIFY: Int = 5
"""The `QUALIFY` expression, or 0."""

comptime CLAUSE_WINDOW: Int = 6
"""The `WINDOW`, a run of `STMT_WINDOW` nodes, empty when there is none."""


comptime SELECT_DISTINCT: UInt32 = 1
"""`SELECT DISTINCT`."""

comptime SELECT_ALL: UInt32 = 2
"""`SELECT ALL`, which is the default said out loud."""


comptime LIMIT_ALL: UInt32 = 1
"""`LIMIT ALL`, which is no limit said out loud."""

comptime LIMIT_PERCENT: UInt32 = 2
"""`LIMIT n%`, where the limit is a share of the rows and not a count."""


comptime SORT_DEFAULT: UInt32 = 0
"""No direction was written."""

comptime SORT_ASCENDING: UInt32 = 1
"""`ASC`."""

comptime SORT_DESCENDING: UInt32 = 2
"""`DESC`."""


comptime NULLS_DEFAULT: UInt32 = 0
"""No null placement was written."""

comptime NULLS_FIRST: UInt32 = 1
"""`NULLS FIRST`."""

comptime NULLS_LAST: UInt32 = 2
"""`NULLS LAST`."""


comptime GROUP_EXPRESSION: UInt32 = 0
"""An ordinary grouping expression."""

comptime GROUP_ALL: UInt32 = 1
"""`GROUP BY ALL`."""

comptime GROUP_EMPTY: UInt32 = 2
"""`()`, the empty grouping, which is only legal inside `GROUPING SETS`."""

comptime GROUP_SETS: UInt32 = 3
"""`GROUPING SETS (...)`."""

comptime GROUP_CUBE: UInt32 = 4
"""`CUBE (...)`."""

comptime GROUP_ROLLUP: UInt32 = 5
"""`ROLLUP (...)`."""

comptime GROUP_TUPLE: UInt32 = 6
"""`(a, b)`, several columns grouped as one set inside `GROUPING SETS`."""


comptime MATERIALIZE_DEFAULT: UInt32 = 0
"""Neither word was written, so the engine decides."""

comptime MATERIALIZE_YES: UInt32 = 1
"""`AS MATERIALIZED`."""

comptime MATERIALIZE_NO: UInt32 = 2
"""`AS NOT MATERIALIZED`."""


struct Ref(ImplicitlyCopyable, Movable):
    """One table reference node, the same shape as an expression node.

    What `a`, `b`, `children` and `payload` mean depends on `kind`, and every
    kind says so in its own docstring.
    """

    var kind: UInt8
    """One of the `REF_` constants."""

    var a: UInt32
    """An index or a run, by kind."""

    var b: UInt32
    """A second index or a flag, by kind."""

    var children: UInt32
    """A run in the side list, or 0 for none."""

    var token: UInt32
    """The token this node starts at, for the caret in an error."""

    var payload: UInt32
    """A string index or a run, by kind."""

    def __init__(
        out self,
        kind: UInt8,
        token: UInt32,
        a: UInt32 = NO_NODE,
        b: UInt32 = NO_NODE,
        children: UInt32 = NO_NODE,
        payload: UInt32 = NO_NODE,
    ):
        """Builds a node.

        Args:
            kind: One of the `REF_` constants.
            token: The token the node starts at.
            a: By kind.
            b: By kind.
            children: A run, or 0.
            payload: By kind.
        """
        self.kind = kind
        self.a = a
        self.b = b
        self.children = children
        self.token = token
        self.payload = payload


struct Stmt(ImplicitlyCopyable, Movable):
    """One statement node, the same shape as an expression node.

    What `a`, `b`, `children` and `payload` mean depends on `kind`, and every
    kind says so in its own docstring. The clause pieces of a `SELECT` are
    nodes in here too, so a `SELECT` list entry and a `GROUP BY` entry have a
    token position and can be pointed at by an error.
    """

    var kind: UInt8
    """One of the `STMT_` constants."""

    var a: UInt32
    """An index, a run or a flag set, by kind."""

    var b: UInt32
    """A second index, a run or a tag, by kind."""

    var children: UInt32
    """A run in the side list, or 0 for none."""

    var token: UInt32
    """The token this node starts at, for the caret in an error."""

    var payload: UInt32
    """A string index, a run or a tag, by kind."""

    def __init__(
        out self,
        kind: UInt8,
        token: UInt32,
        a: UInt32 = NO_NODE,
        b: UInt32 = NO_NODE,
        children: UInt32 = NO_NODE,
        payload: UInt32 = NO_NODE,
    ):
        """Builds a node.

        Args:
            kind: One of the `STMT_` constants.
            token: The token the node starts at.
            a: By kind.
            b: By kind.
            children: A run, or 0.
            payload: By kind.
        """
        self.kind = kind
        self.a = a
        self.b = b
        self.children = children
        self.token = token
        self.payload = payload


struct Ast(Movable):
    """The arenas, the side list and the string pool.

    One of these owns a whole statement's worth of nodes and the text they
    refer to, so it outlives the query string it was built from.
    """

    var exprs: List[Expr]
    """The expression arena. Index 0 is the null node."""

    var refs: List[Ref]
    """The table reference arena. Index 0 is the null node."""

    var stmts: List[Stmt]
    """The statement arena. Index 0 is the null node."""

    var runs: List[UInt32]
    """The side list holding every variadic child run.

    A run is a count followed by that many entries, which is why a run index of
    0 can mean empty: index 0 holds a count of zero and nothing follows it.
    """

    var strings: List[String]
    """The pool. Index 0 is the empty string."""

    var interned: Dict[String, UInt32]
    """Text to pool index, so the same name is stored once.

    Names repeat heavily in a real query, since every column reference in a
    `SELECT` list and every mention of it in `GROUP BY` and `ORDER BY` is the
    same handful of words.
    """

    def __init__(out self):
        """Builds an empty AST, with the null node and the empty run in place.
        """
        self.exprs = List[Expr]()
        self.exprs.append(Expr(kind=0, token=0))
        self.refs = List[Ref]()
        self.refs.append(Ref(kind=0, token=0))
        self.stmts = List[Stmt]()
        self.stmts.append(Stmt(kind=0, token=0))
        self.runs = List[UInt32]()
        self.runs.append(0)
        self.strings = List[String]()
        self.strings.append(String())
        self.interned = Dict[String, UInt32]()

    def intern(mut self, text: StringSlice) -> UInt32:
        """Puts text in the pool and returns its index.

        Args:
            text: The decoded text, with no quotes and no escapes left in it.

        Returns:
            The pool index, which is 0 for the empty string.
        """
        if text.byte_length() == 0:
            return 0
        var key = String(text)
        var found = self.interned.get(key)
        if found:
            return found.value()
        var at = UInt32(len(self.strings))
        self.strings.append(key)
        self.interned[key^] = at
        return at

    def text(ref self, index: UInt32) -> ref[self.strings[Int(index)]] String:
        """Returns the text a pool index refers to.

        A reference rather than a slice, because a short `String` may keep its
        bytes inside itself, and a slice into one would dangle the moment the
        pool grew and moved it. A reference is rebound by the same move and
        stays correct.

        Args:
            index: A pool index, where 0 is the empty string.

        Returns:
            The text, borrowed from the pool.
        """
        return self.strings[Int(index)]

    def run(mut self, items: List[UInt32]) -> UInt32:
        """Stores a run of child indices and returns a handle to it.

        Args:
            items: The children, in order.

        Returns:
            The run handle, which is 0 for an empty run.
        """
        if len(items) == 0:
            return NO_NODE
        var at = UInt32(len(self.runs))
        self.runs.append(UInt32(len(items)))
        for item in items:
            self.runs.append(item)
        return at

    def length(self, run: UInt32) -> Int:
        """Returns how many entries a run has.

        Args:
            run: A run handle, or 0.

        Returns:
            The count, which is 0 for an empty run.
        """
        if run == NO_NODE:
            return 0
        return Int(self.runs[Int(run)])

    def at(self, run: UInt32, index: Int) -> UInt32:
        """Returns one entry of a run.

        Args:
            run: A run handle, which must not be 0.
            index: Which entry, from 0.

        Returns:
            The entry.
        """
        return self.runs[Int(run) + 1 + index]

    def items(self, run: UInt32) -> List[UInt32]:
        """Collects a whole run.

        Args:
            run: A run handle, or 0.

        Returns:
            The entries in order, empty for an empty run.
        """
        var out = List[UInt32]()
        for i in range(self.length(run)):
            out.append(self.at(run, i))
        return out^

    def slot(self, run: UInt32, index: Int) -> UInt32:
        """Returns one entry of a fixed length run, or 0 if there is no run.

        Args:
            run: A run handle, or 0.
            index: Which entry, from 0.

        Returns:
            The entry, or 0.
        """
        if run == NO_NODE:
            return NO_NODE
        return self.at(run, index)

    def names(mut self, parts: List[String]) -> UInt32:
        """Interns a list of names and stores them as a run.

        Args:
            parts: The names, in order.

        Returns:
            The run handle, which is 0 for an empty list.
        """
        var interned = List[UInt32]()
        for part in parts:
            interned.append(self.intern(part))
        return self.run(interned)

    def alias(
        mut self, name: StringSlice, columns: List[String] = List[String]()
    ) -> UInt32:
        """Builds the alias run a table reference carries.

        The run is the alias name followed by the column aliases, so an empty
        run means the reference was written without an alias. Column aliases
        without a table alias are not a thing SQL can write, which is what lets
        one run hold both.

        Args:
            name: The alias, empty for none.
            columns: The column aliases, in order.

        Returns:
            The run handle, which is 0 when there is no alias.
        """
        if name.byte_length() == 0:
            return NO_NODE
        var parts = List[UInt32]()
        parts.append(self.intern(name))
        for column in columns:
            parts.append(self.intern(column))
        return self.run(parts)

    def add(mut self, var node: Expr) -> UInt32:
        """Puts a node in the expression arena.

        Args:
            node: The node.

        Returns:
            Its index, which is never 0.
        """
        var at = UInt32(len(self.exprs))
        self.exprs.append(node^)
        return at

    def literal(
        mut self, tag: UInt32, value: StringSlice, token: UInt32 = 0
    ) -> UInt32:
        """Builds a constant.

        Args:
            tag: One of the `LITERAL_` constants.
            value: The decoded value, empty for `NULL`.
            token: The token it starts at.

        Returns:
            The node index.
        """
        var text = self.intern(value)
        return self.add(
            Expr(kind=EXPR_LITERAL, token=token, b=tag, payload=text)
        )

    def column(mut self, parts: List[String], token: UInt32 = 0) -> UInt32:
        """Builds a column reference.

        Args:
            parts: The name parts, outermost first, at least one.
            token: The token it starts at.

        Returns:
            The node index.
        """
        var run = self.names(parts)
        return self.add(Expr(kind=EXPR_COLUMN, token=token, children=run))

    def binary(
        mut self,
        operator: StringSlice,
        left: UInt32,
        right: UInt32,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds an infix operator node.

        Args:
            operator: The operator as SQL spells it.
            left: The left operand.
            right: The right operand.
            token: The token the operator is at.

        Returns:
            The node index.
        """
        var text = self.intern(operator)
        return self.add(
            Expr(kind=EXPR_BINARY, token=token, a=left, b=right, payload=text)
        )

    def unary(
        mut self, operator: StringSlice, operand: UInt32, token: UInt32 = 0
    ) -> UInt32:
        """Builds a prefix operator node.

        Args:
            operator: The operator as SQL spells it.
            operand: What it applies to.
            token: The token the operator is at.

        Returns:
            The node index.
        """
        var text = self.intern(operator)
        return self.add(
            Expr(kind=EXPR_UNARY, token=token, a=operand, payload=text)
        )

    def call(
        mut self,
        name: StringSlice,
        arguments: List[UInt32],
        flags: UInt32 = 0,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds a function call with an unqualified name.

        Args:
            name: The function name.
            arguments: The argument expressions, in order.
            flags: A bit set of the `CALL_` constants.
            token: The token the name is at.

        Returns:
            The node index.
        """
        var parts = List[UInt32]()
        parts.append(self.intern(name))
        var named = self.run(parts)
        var args = self.run(arguments)
        return self.add(
            Expr(
                kind=EXPR_FUNCTION,
                token=token,
                a=flags,
                children=args,
                payload=named,
            )
        )

    def cast(
        mut self,
        operand: UInt32,
        type_name: StringSlice,
        tries: Bool = False,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds a cast.

        Args:
            operand: What is being cast.
            type_name: The target type, as written.
            tries: Whether this is `TRY_CAST`.
            token: The token it starts at.

        Returns:
            The node index.
        """
        var text = self.intern(type_name)
        return self.add(
            Expr(
                kind=EXPR_CAST,
                token=token,
                a=operand,
                b=UInt32(1) if tries else UInt32(0),
                payload=text,
            )
        )

    def case(
        mut self,
        arms: List[UInt32],
        otherwise: UInt32 = NO_NODE,
        operand: UInt32 = NO_NODE,
        token: UInt32 = 0,
    ) raises -> UInt32:
        """Builds a `CASE`.

        Args:
            arms: Alternating condition and result, so an even length.
            otherwise: The `ELSE` result, or 0 for none.
            operand: What a simple `CASE x WHEN` compares against, or 0.
            token: The token it starts at.

        Returns:
            The node index.

        Raises:
            Error: If `arms` is empty or has an odd length.
        """
        if len(arms) == 0 or len(arms) % 2 != 0:
            raise Error(
                String(
                    (
                        "a CASE wants an even number of condition and result"
                        " entries, and at least two, but got "
                    ),
                    len(arms),
                )
            )
        var run = self.run(arms)
        return self.add(
            Expr(
                kind=EXPR_CASE,
                token=token,
                a=operand,
                b=otherwise,
                children=run,
            )
        )

    def between(
        mut self,
        operand: UInt32,
        low: UInt32,
        high: UInt32,
        negated: Bool = False,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds `x BETWEEN lo AND hi`.

        Args:
            operand: What is being tested.
            low: The lower bound.
            high: The upper bound.
            negated: Whether this is `NOT BETWEEN`.
            token: The token it starts at.

        Returns:
            The node index.
        """
        var bounds = List[UInt32]()
        bounds.append(low)
        bounds.append(high)
        var run = self.run(bounds)
        return self.add(
            Expr(
                kind=EXPR_BETWEEN,
                token=token,
                a=operand,
                children=run,
                payload=UInt32(1) if negated else UInt32(0),
            )
        )

    def in_list(
        mut self,
        operand: UInt32,
        candidates: List[UInt32],
        negated: Bool = False,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds `x IN (a, b, c)`.

        Named with a trailing underscore because `in` is a keyword.

        Args:
            operand: What is being tested.
            candidates: The values on the right, in order.
            negated: Whether this is `NOT IN`.
            token: The token it starts at.

        Returns:
            The node index.
        """
        var run = self.run(candidates)
        return self.add(
            Expr(
                kind=EXPR_IN,
                token=token,
                a=operand,
                children=run,
                payload=UInt32(1) if negated else UInt32(0),
            )
        )

    def list_of(mut self, elements: List[UInt32], token: UInt32 = 0) -> UInt32:
        """Builds a list constructor, `[1, 2, 3]`.

        Args:
            elements: The elements, in order, possibly none.
            token: The token it starts at.

        Returns:
            The node index.
        """
        var run = self.run(elements)
        return self.add(Expr(kind=EXPR_LIST, token=token, children=run))

    def struct_of(
        mut self,
        names: List[String],
        values: List[UInt32],
        token: UInt32 = 0,
    ) raises -> UInt32:
        """Builds a struct constructor, `{'a': 1}`.

        Args:
            names: The field names, in order.
            values: The field values, in the same order.
            token: The token it starts at.

        Returns:
            The node index.

        Raises:
            Error: If there are not as many values as names.
        """
        if len(names) != len(values):
            raise Error(
                String(
                    "a struct wants one value per field name, but got ",
                    len(names),
                    " names and ",
                    len(values),
                    " values",
                )
            )
        var flat = List[UInt32]()
        for i in range(len(names)):
            flat.append(self.intern(names[i]))
            flat.append(values[i])
        var run = self.run(flat)
        return self.add(Expr(kind=EXPR_STRUCT, token=token, children=run))

    def collate(
        mut self, operand: UInt32, collation: StringSlice, token: UInt32 = 0
    ) -> UInt32:
        """Builds `x COLLATE c`.

        Args:
            operand: What is being collated.
            collation: The collation name.
            token: The token it starts at.

        Returns:
            The node index.
        """
        var text = self.intern(collation)
        return self.add(
            Expr(kind=EXPR_COLLATE, token=token, a=operand, payload=text)
        )

    def parameter(
        mut self, sigil: StringSlice, name: StringSlice = "", token: UInt32 = 0
    ) -> UInt32:
        """Builds a prepared statement parameter.

        Args:
            sigil: Either `?` or `$`.
            name: The number or name after it, empty for a bare `?`.
            token: The token it starts at.

        Returns:
            The node index.
        """
        return self.add(
            Expr(
                kind=EXPR_PARAMETER,
                token=token,
                b=self.intern(sigil),
                payload=self.intern(name),
            )
        )

    def star(
        mut self,
        qualifier: List[String] = List[String](),
        exclude: List[String] = List[String](),
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds `*`, or `t.*`, with an optional `EXCLUDE`.

        `REPLACE` and `RENAME` are the other two modifiers the node has room
        for. They take expressions and name pairs rather than plain names, so
        they are built directly rather than through this.

        Args:
            qualifier: The name parts before the star, empty for a bare one.
            exclude: The names to leave out.
            token: The token the star is at.

        Returns:
            The node index.
        """
        var parts = List[UInt32]()
        for part in qualifier:
            parts.append(self.intern(part))
        var excluded = List[UInt32]()
        for name in exclude:
            excluded.append(self.intern(name))
        return self.add(
            Expr(
                kind=EXPR_STAR,
                token=token,
                a=self.run(excluded),
                children=self.run(parts),
            )
        )

    def subquery(mut self, statement: UInt32, token: UInt32 = 0) -> UInt32:
        """Builds a scalar subquery.

        Args:
            statement: The statement, in the statement arena.
            token: The token the opening parenthesis is at.

        Returns:
            The expression node index.
        """
        return self.add(Expr(kind=EXPR_SUBQUERY, token=token, a=statement))

    def exists(
        mut self, statement: UInt32, negated: Bool = False, token: UInt32 = 0
    ) -> UInt32:
        """Builds `EXISTS (SELECT ...)`.

        Args:
            statement: The statement, in the statement arena.
            negated: Whether this is `NOT EXISTS`.
            token: The token it starts at.

        Returns:
            The expression node index.
        """
        return self.add(
            Expr(
                kind=EXPR_EXISTS,
                token=token,
                a=statement,
                b=UInt32(1) if negated else UInt32(0),
            )
        )

    def in_subquery(
        mut self,
        operand: UInt32,
        statement: UInt32,
        negated: Bool = False,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds `x IN (SELECT ...)`.

        Args:
            operand: What is being tested.
            statement: The statement on the right, in the statement arena.
            negated: Whether this is `NOT IN`.
            token: The token it starts at.

        Returns:
            The expression node index.
        """
        return self.add(
            Expr(
                kind=EXPR_IN_SUBQUERY,
                token=token,
                a=operand,
                b=statement,
                payload=UInt32(1) if negated else UInt32(0),
            )
        )

    def window(
        mut self,
        partition: List[UInt32] = List[UInt32](),
        ordering: List[UInt32] = List[UInt32](),
        frame: UInt32 = NO_NODE,
        base: StringSlice = "",
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds the window a call is computed `OVER`.

        Args:
            partition: The `PARTITION BY` expressions, in order.
            ordering: The window's `ORDER BY`, a list of `STMT_ORDER` nodes.
            frame: The `EXPR_FRAME`, or 0 for a window with no frame.
            base: The window this one starts from, empty for none.
            token: The token it starts at.

        Returns:
            The expression node index.
        """
        return self.add(
            Expr(
                kind=EXPR_WINDOW,
                token=token,
                a=self.run(ordering),
                b=frame,
                children=self.run(partition),
                payload=self.intern(base),
            )
        )

    def frame(
        mut self,
        mode: UInt32,
        start: UInt32,
        end: UInt32 = BOUND_NONE,
        exclude: UInt32 = EXCLUDE_NONE,
        start_at: UInt32 = NO_NODE,
        end_at: UInt32 = NO_NODE,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds the `ROWS`, `RANGE` or `GROUPS` clause of a window.

        Args:
            mode: One of the `FRAME_` constants.
            start: The start bound, one of the `BOUND_` constants.
            end: The end bound, `BOUND_NONE` when there is no `BETWEEN`.
            exclude: One of the `EXCLUDE_` constants.
            start_at: The start bound's expression, or 0.
            end_at: The end bound's expression, or 0.
            token: The token it starts at.

        Returns:
            The expression node index.
        """
        return self.add(
            Expr(
                kind=EXPR_FRAME,
                token=token,
                a=start_at,
                b=end_at,
                payload=frame_tags(mode, start, end, exclude),
            )
        )

    def add_ref(mut self, var node: Ref) -> UInt32:
        """Puts a node in the table reference arena.

        Args:
            node: The node.

        Returns:
            Its index, which is never 0.
        """
        var at = UInt32(len(self.refs))
        self.refs.append(node^)
        return at

    def table(
        mut self,
        parts: List[String],
        name: StringSlice = "",
        columns: List[String] = List[String](),
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds a named table reference.

        Args:
            parts: The name parts, outermost first, at least one.
            name: The alias, empty for none.
            columns: The column aliases, in order.
            token: The token it starts at.

        Returns:
            The reference node index.
        """
        return self.add_ref(
            Ref(
                kind=REF_TABLE,
                token=token,
                children=self.names(parts),
                payload=self.alias(name, columns),
            )
        )

    def subquery_ref(
        mut self,
        statement: UInt32,
        name: StringSlice = "",
        columns: List[String] = List[String](),
        lateral: Bool = False,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds a subquery in a `FROM`.

        Args:
            statement: The statement, in the statement arena.
            name: The alias, empty for none.
            columns: The column aliases, in order.
            lateral: Whether it was written `LATERAL`.
            token: The token it starts at.

        Returns:
            The reference node index.
        """
        return self.add_ref(
            Ref(
                kind=REF_SUBQUERY,
                token=token,
                a=statement,
                b=UInt32(1) if lateral else UInt32(0),
                payload=self.alias(name, columns),
            )
        )

    def function_ref(
        mut self,
        parts: List[String],
        arguments: List[UInt32] = List[UInt32](),
        name: StringSlice = "",
        columns: List[String] = List[String](),
        lateral: Bool = False,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds a table function reference.

        Args:
            parts: The function name parts, outermost first.
            arguments: The argument expressions, in order.
            name: The alias, empty for none.
            columns: The column aliases, in order.
            lateral: Whether it was written `LATERAL`.
            token: The token it starts at.

        Returns:
            The reference node index.
        """
        return self.add_ref(
            Ref(
                kind=REF_FUNCTION,
                token=token,
                a=self.names(parts),
                b=UInt32(1) if lateral else UInt32(0),
                children=self.run(arguments),
                payload=self.alias(name, columns),
            )
        )

    def parens_ref(
        mut self,
        inner: UInt32,
        name: StringSlice = "",
        columns: List[String] = List[String](),
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds a parenthesized table reference.

        Args:
            inner: The reference inside the parentheses.
            name: The alias, empty for none.
            columns: The column aliases, in order.
            token: The token the opening parenthesis is at.

        Returns:
            The reference node index.
        """
        return self.add_ref(
            Ref(
                kind=REF_PARENS,
                token=token,
                a=inner,
                payload=self.alias(name, columns),
            )
        )

    def join(
        mut self,
        operator: StringSlice,
        left: UInt32,
        right: UInt32,
        on: UInt32 = NO_NODE,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds a join with an `ON`, or with no condition.

        Args:
            operator: The join as SQL spells it, such as `LEFT OUTER JOIN`.
            left: The left reference.
            right: The right reference.
            on: The condition, or 0 for a join that takes none.
            token: The token the join word is at.

        Returns:
            The reference node index.
        """
        var condition = List[UInt32]()
        if on != NO_NODE:
            condition.append(on)
        return self.add_ref(
            Ref(
                kind=REF_JOIN,
                token=token,
                a=left,
                b=right,
                children=self.run(condition),
                payload=self.intern(operator),
            )
        )

    def join_using(
        mut self,
        operator: StringSlice,
        left: UInt32,
        right: UInt32,
        columns: List[String],
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds a join with a `USING`.

        Args:
            operator: The join as SQL spells it, such as `JOIN`.
            left: The left reference.
            right: The right reference.
            columns: The column names shared by both sides.
            token: The token the join word is at.

        Returns:
            The reference node index.
        """
        return self.add_ref(
            Ref(
                kind=REF_JOIN_USING,
                token=token,
                a=left,
                b=right,
                children=self.names(columns),
                payload=self.intern(operator),
            )
        )

    def add_stmt(mut self, var node: Stmt) -> UInt32:
        """Puts a node in the statement arena.

        Args:
            node: The node.

        Returns:
            Its index, which is never 0.
        """
        var at = UInt32(len(self.stmts))
        self.stmts.append(node^)
        return at

    def select(
        mut self,
        query: UInt32,
        modifiers: UInt32 = NO_NODE,
        ctes: List[UInt32] = List[UInt32](),
        recursive: Bool = False,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds a whole `SELECT` statement.

        Args:
            query: The query node, which is a query, a set operation, a
                `VALUES` or a `TABLE`.
            modifiers: The `STMT_MODIFIERS` node, or 0 for none.
            ctes: The `WITH` entries, in order.
            recursive: Whether the `WITH` was written `RECURSIVE`.
            token: The token it starts at.

        Returns:
            The statement node index.
        """
        return self.add_stmt(
            Stmt(
                kind=STMT_SELECT,
                token=token,
                a=query,
                b=modifiers,
                children=self.run(ctes),
                payload=UInt32(1) if recursive else UInt32(0),
            )
        )

    def query(
        mut self,
        projection: List[UInt32] = List[UInt32](),
        tables: List[UInt32] = List[UInt32](),
        filter: UInt32 = NO_NODE,
        grouping: List[UInt32] = List[UInt32](),
        having: UInt32 = NO_NODE,
        qualify: UInt32 = NO_NODE,
        flags: UInt32 = 0,
        distinct_on: List[UInt32] = List[UInt32](),
        windows: List[UInt32] = List[UInt32](),
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds one `SELECT ... FROM ... WHERE ...` block.

        Args:
            projection: The `SELECT` list, a list of `STMT_ITEM` nodes.
            tables: The `FROM`, a list of table reference indices.
            filter: The `WHERE` expression, or 0.
            grouping: The `GROUP BY`, a list of `STMT_GROUP` nodes.
            having: The `HAVING` expression, or 0.
            qualify: The `QUALIFY` expression, or 0.
            flags: A bit set of the `SELECT_` constants.
            distinct_on: The `DISTINCT ON` expressions, in order.
            windows: The `WINDOW` clause, a list of `STMT_WINDOW` nodes.
            token: The token it starts at.

        Returns:
            The statement node index.
        """
        var clauses = List[UInt32](length=CLAUSE_SLOTS, fill=NO_NODE)
        clauses[CLAUSE_PROJECTION] = self.run(projection)
        clauses[CLAUSE_FROM] = self.run(tables)
        clauses[CLAUSE_WHERE] = filter
        clauses[CLAUSE_GROUP] = self.run(grouping)
        clauses[CLAUSE_HAVING] = having
        clauses[CLAUSE_QUALIFY] = qualify
        clauses[CLAUSE_WINDOW] = self.run(windows)
        return self.add_stmt(
            Stmt(
                kind=STMT_QUERY,
                token=token,
                a=flags,
                b=self.run(distinct_on),
                children=self.run(clauses),
            )
        )

    def set_operation(
        mut self,
        operator: StringSlice,
        left: UInt32,
        right: UInt32,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds a `UNION`, `INTERSECT` or `EXCEPT`.

        Args:
            operator: The operator as SQL spells it, such as `UNION ALL`.
            left: The left query node.
            right: The right query node.
            token: The token the operator is at.

        Returns:
            The statement node index.
        """
        return self.add_stmt(
            Stmt(
                kind=STMT_SET_OPERATION,
                token=token,
                a=left,
                b=right,
                payload=self.intern(operator),
            )
        )

    def values(mut self, rows: List[List[UInt32]], token: UInt32 = 0) -> UInt32:
        """Builds a `VALUES`.

        Args:
            rows: The rows, each a list of expressions.
            token: The token it starts at.

        Returns:
            The statement node index.
        """
        var handles = List[UInt32]()
        for row in rows:
            handles.append(self.run(row))
        return self.add_stmt(
            Stmt(kind=STMT_VALUES, token=token, children=self.run(handles))
        )

    def table_statement(
        mut self, parts: List[String], token: UInt32 = 0
    ) -> UInt32:
        """Builds `TABLE t`.

        Args:
            parts: The name parts, outermost first, at least one.
            token: The token it starts at.

        Returns:
            The statement node index.
        """
        return self.add_stmt(
            Stmt(kind=STMT_TABLE, token=token, children=self.names(parts))
        )

    def pivot(
        mut self,
        source: UInt32,
        on: List[UInt32] = List[UInt32](),
        using: List[UInt32] = List[UInt32](),
        groups: List[String] = List[String](),
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds a `PIVOT` statement.

        Args:
            source: The table reference being pivoted.
            on: The `STMT_PIVOT_ON` nodes, in order.
            using: The `STMT_ITEM` nodes of the `USING` list, in order.
            groups: The `GROUP BY` names, in order.
            token: The token it starts at.

        Returns:
            The statement node index.
        """
        return self.add_stmt(
            Stmt(
                kind=STMT_PIVOT,
                token=token,
                a=source,
                b=self.run(on),
                children=self.run(using),
                payload=self.names(groups),
            )
        )

    def pivot_on(
        mut self,
        header: UInt32,
        values: List[UInt32] = List[UInt32](),
        name: StringSlice = "",
        statement: UInt32 = NO_NODE,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds one pivot column.

        At most one of `values`, `name` and `statement` says anything. All
        three empty is the bare `ON a`.

        Args:
            header: The header expression.
            values: The `STMT_ITEM` nodes after `IN`, in order.
            name: The enum name after `IN`, empty for none.
            statement: The subquery after `IN`, or 0.
            token: The token the header starts at.

        Returns:
            The statement node index.
        """
        return self.add_stmt(
            Stmt(
                kind=STMT_PIVOT_ON,
                token=token,
                a=header,
                b=statement,
                children=self.run(values),
                payload=self.intern(name),
            )
        )

    def modifiers(
        mut self,
        order: List[UInt32] = List[UInt32](),
        limit: UInt32 = NO_NODE,
        offset: UInt32 = NO_NODE,
        flags: UInt32 = 0,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds the `ORDER BY`, `LIMIT` and `OFFSET` that trail a statement.

        Args:
            order: The `ORDER BY` entries, a list of `STMT_ORDER` nodes.
            limit: The limit expression, or 0.
            offset: The offset expression, or 0.
            flags: A bit set of the `LIMIT_` constants.
            token: The token it starts at.

        Returns:
            The statement node index.
        """
        return self.add_stmt(
            Stmt(
                kind=STMT_MODIFIERS,
                token=token,
                a=limit,
                b=offset,
                children=self.run(order),
                payload=flags,
            )
        )

    def cte(
        mut self,
        name: StringSlice,
        statement: UInt32,
        columns: List[String] = List[String](),
        materialize: UInt32 = MATERIALIZE_DEFAULT,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds one entry of a `WITH`.

        Args:
            name: The name the entry is bound to.
            statement: The statement it stands for.
            columns: The column aliases, in order.
            materialize: One of the `MATERIALIZE_` constants.
            token: The token the name is at.

        Returns:
            The statement node index.
        """
        return self.add_stmt(
            Stmt(
                kind=STMT_CTE,
                token=token,
                a=statement,
                b=materialize,
                children=self.names(columns),
                payload=self.intern(name),
            )
        )

    def item(
        mut self,
        expression: UInt32,
        name: StringSlice = "",
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds one entry of a `SELECT` list.

        Args:
            expression: What is being selected.
            name: The alias, empty for none.
            token: The token it starts at.

        Returns:
            The statement node index.
        """
        return self.add_stmt(
            Stmt(
                kind=STMT_ITEM,
                token=token,
                a=expression,
                payload=self.intern(name),
            )
        )

    def order(
        mut self,
        expression: UInt32 = NO_NODE,
        direction: UInt32 = SORT_DEFAULT,
        nulls: UInt32 = NULLS_DEFAULT,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds one entry of an `ORDER BY`.

        Args:
            expression: What to sort by, or 0 for `ORDER BY ALL`.
            direction: One of the `SORT_` constants.
            nulls: One of the `NULLS_` constants.
            token: The token it starts at.

        Returns:
            The statement node index.
        """
        return self.add_stmt(
            Stmt(
                kind=STMT_ORDER,
                token=token,
                a=expression,
                b=direction,
                payload=nulls,
            )
        )

    def group(
        mut self,
        tag: UInt32 = GROUP_EXPRESSION,
        expression: UInt32 = NO_NODE,
        entries: List[UInt32] = List[UInt32](),
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds one entry of a `GROUP BY`.

        Args:
            tag: One of the `GROUP_` constants.
            expression: The expression, for `GROUP_EXPRESSION`.
            entries: The nested entries, for the set forms.
            token: The token it starts at.

        Returns:
            The statement node index.
        """
        return self.add_stmt(
            Stmt(
                kind=STMT_GROUP,
                token=token,
                a=expression,
                b=tag,
                children=self.run(entries),
            )
        )

    def window_definition(
        mut self,
        name: StringSlice,
        specification: UInt32,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds one entry of a `WINDOW` clause.

        Args:
            name: The name the window is given.
            specification: The `EXPR_WINDOW` it stands for.
            token: The token it starts at.

        Returns:
            The statement node index.
        """
        return self.add_stmt(
            Stmt(
                kind=STMT_WINDOW,
                token=token,
                a=specification,
                payload=self.intern(name),
            )
        )
