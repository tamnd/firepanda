"""The four shapes a subquery can be written in, and what each one requires.

A `SELECT` inside an expression is one of four things and they are four
different questions, not one question with a flag on it. A scalar subquery
asks for a value, `EXISTS` asks whether there is a row, `IN` asks whether a
value is among the rows, and a quantified comparison asks whether a comparison
holds against all of them or against any of them. Each has its own arity rule
and its own answer when the subquery gives back nothing. See
docs/specs/sql/05-ast-and-binder.md section 11.

Three of the four want exactly one column and say so while binding, with a
message that counts what it got and what it wanted: `Subquery returns 2
columns - expected 1`. `EXISTS` is the one that never looks, because it throws
the select list away, so `EXISTS (SELECT x, y FROM u)` is a fine query and
`EXISTS (SELECT 1/0 FROM u)` never divides anything.

A scalar subquery also has a rule the binder cannot check, which is that it
returns at most one row. DuckDB checks it while running and the message names
the setting that turns the check off, which used to be the default and used to
give back a row picked at random. That default changing is the kind of thing
that silently rewrote somebody's answers, so the message is reproduced whole.

Correlation is recorded rather than discovered. `bind.mojo` writes down every
outer reference at the level that reached for it, so a subquery is correlated
when some level at or under it reached past the subquery's own level. Asking
after the fact instead means walking the tree looking for outer references,
which is a walk that can miss one, and a missed correlation is a decorrelation
that drops rows.

What is not here is the row value form, `(a, b) IN (SELECT x, y FROM u)`. The
transformer refuses a row value by name, so the arity rule takes the number it
wants as an argument and gets 1 from everything that can reach it today. The
count is written that way so that turning row values on is a change in one
place rather than a rewrite here.
"""

from .ast import (
    Ast,
    EXPR_EXISTS,
    EXPR_IN_SUBQUERY,
    EXPR_QUANTIFIED,
    EXPR_SUBQUERY,
    NO_NODE,
)
from .bind import Reference, Scopes
from .catalog import NOT_FOUND


comptime SHAPE_NONE: UInt8 = 0
"""What an expression that holds no subquery tags as."""

comptime SHAPE_SCALAR: UInt8 = 1
"""`(SELECT ...)` where a value is wanted."""

comptime SHAPE_EXISTS: UInt8 = 2
"""`EXISTS (SELECT ...)`, and `NOT EXISTS`."""

comptime SHAPE_IN: UInt8 = 3
"""`x IN (SELECT ...)`, and `NOT IN`."""

comptime SHAPE_QUANTIFIED: UInt8 = 4
"""`x > ALL (SELECT ...)` and `x > ANY (SELECT ...)`."""


struct Subquery(Copyable, Movable):
    """One subquery, tagged, with everything the binder needs off the node."""

    var shape: UInt8
    """One of the `SHAPE_` constants."""

    var statement: UInt32
    """The statement, in the statement arena."""

    var operand: UInt32
    """What is being tested, or `NO_NODE` for a scalar subquery and `EXISTS`."""

    var negated: Bool
    """Whether the query wrote `NOT EXISTS` or `NOT IN`."""

    var every: Bool
    """True for `ALL`, false for `ANY`, meaningless for the other shapes."""

    var operator: String
    """The comparison a quantified one carries, empty for the other shapes."""

    def __init__(
        out self,
        shape: UInt8,
        statement: UInt32,
        operand: UInt32 = NO_NODE,
        negated: Bool = False,
        every: Bool = False,
        operator: StringSlice = "",
    ):
        """One tagged subquery.

        Args:
            shape: One of the `SHAPE_` constants.
            statement: The statement index.
            operand: What is being tested, or `NO_NODE`.
            negated: Whether `NOT` was written.
            every: True for `ALL`.
            operator: The comparison, for a quantified one.
        """
        self.shape = shape
        self.statement = statement
        self.operand = operand
        self.negated = negated
        self.every = every
        self.operator = String(operator)

    def wants_one_column(self) -> Bool:
        """Whether the shape cares how many columns come back.

        Returns:
            True for every shape but `EXISTS`, which throws the select list
            away and never counts it.
        """
        return self.shape != SHAPE_EXISTS and self.shape != SHAPE_NONE

    def at_most_one_row(self) -> Bool:
        """Whether the shape refuses a second row at run time.

        Returns:
            True for a scalar subquery, which is the only shape that stands
            where a single value goes.
        """
        return self.shape == SHAPE_SCALAR


def about(ast: Ast, node: UInt32) raises -> Subquery:
    """Tags one expression as whichever of the four shapes it is.

    Args:
        ast: The arenas.
        node: The expression.

    Returns:
        The tagged subquery, or one with `SHAPE_NONE` when the node is not a
        subquery at all.

    Raises:
        Error: If the node index is not in the arena.
    """
    if node == 0 or Int(node) >= len(ast.exprs):
        raise Error("an expression index that is not in the arena")
    ref item = ast.exprs[Int(node)]
    if item.kind == EXPR_SUBQUERY:
        return Subquery(SHAPE_SCALAR, item.a)
    if item.kind == EXPR_EXISTS:
        return Subquery(SHAPE_EXISTS, item.a, negated=item.b == 1)
    if item.kind == EXPR_IN_SUBQUERY:
        return Subquery(
            SHAPE_IN, item.b, operand=item.a, negated=item.payload == 1
        )
    if item.kind == EXPR_QUANTIFIED:
        return Subquery(
            SHAPE_QUANTIFIED,
            item.b,
            operand=item.a,
            every=item.children == 1,
            operator=ast.text(item.payload),
        )
    return Subquery(SHAPE_NONE, NO_NODE)


def check_columns(returned: Int, expected: Int = 1) raises:
    """Refuses a subquery that gives back the wrong number of columns.

    Args:
        returned: How many the select list produced.
        expected: How many the shape wants, which is 1 everywhere until row
            values are taken.

    Raises:
        Error: If the two disagree.
    """
    if returned != expected:
        raise Error(wrong_column_count(returned, expected))


def correlated(scopes: Scopes, level: Int) -> Bool:
    """Whether a subquery's level, or any level under it, reached outward.

    Args:
        scopes: The chain of levels.
        level: The level the subquery opened.

    Returns:
        True if something inside it names a column from further out than
        itself.
    """
    return len(outer_references(scopes, level)) != 0


def outer_references(scopes: Scopes, level: Int) -> List[Reference]:
    """Every column a subquery reaches for from outside itself.

    A reference is recorded at the level that wrote it, so this looks at that
    level and at every level under it, and keeps the ones whose target sits
    further out than the subquery. A reference from two levels down to one
    level down is inside the subquery and is not a correlation of it.

    Args:
        scopes: The chain of levels.
        level: The level the subquery opened.

    Returns:
        The references, in the order they were recorded.
    """
    var out = List[Reference]()
    for at in range(len(scopes)):
        if at != level and not _under(scopes, at, level):
            continue
        for entry in scopes.levels[at].correlations:
            var target = at
            for _ in range(entry.depth):
                if target == NOT_FOUND:
                    break
                target = scopes.levels[target].parent
            if target != NOT_FOUND and _under(scopes, level, target):
                out.append(entry)
    return out^


def wrong_column_count(returned: Int, expected: Int) -> String:
    """DuckDB's error for a subquery of the wrong width.

    The count is spelled with the same word either way, so a subquery of one
    column reads as `returns 1 columns`. That is DuckDB's and it is the sort
    of thing somebody will notice in a log and file against us.

    Args:
        returned: How many came back.
        expected: How many were wanted.

    Returns:
        The message.
    """
    return String(
        "Binder Error: Subquery returns ",
        returned,
        " columns - expected ",
        expected,
    )


def too_many_rows() -> String:
    """DuckDB's error for a scalar subquery that gave back a second row.

    Returns:
        The message, with the setting it names on the second line.
    """
    return String(
        "Invalid Input Error: More than one row returned by a subquery used"
        " as an expression - scalar subqueries can only return a single"
        ' row.\nUse "SET scalar_subquery_error_on_multiple_rows=false" to'
        " revert to previous behavior of returning a random row."
    )


def cannot_compare(left: StringSlice, right: StringSlice) -> String:
    """DuckDB's error for an `IN` or a quantified comparison over two types.

    One message covers all three spellings, so a mistyped `IN` is reported as
    an `IN/ANY/ALL` problem whichever of them was written.

    Args:
        left: The operand's type name.
        right: The subquery column's type name.

    Returns:
        The message.
    """
    return String(
        "Binder Error: Cannot compare values of type ",
        left,
        " and ",
        right,
        " in IN/ANY/ALL clause - an explicit cast is required",
    )


def _under(scopes: Scopes, level: Int, ancestor: Int) -> Bool:
    """Whether one level is inside another, not counting itself.

    Args:
        scopes: The chain of levels.
        level: The inner level.
        ancestor: The one it might sit under.

    Returns:
        True if walking parents from `level` reaches `ancestor`.
    """
    if level == NOT_FOUND or ancestor == NOT_FOUND:
        return False
    var at = scopes.levels[level].parent
    var steps = 0
    while at != NOT_FOUND:
        if at == ancestor:
            return True
        steps += 1
        if steps > len(scopes):
            return False
        at = scopes.levels[at].parent
    return False
