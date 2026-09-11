"""A `WITH` clause: what each name stands for, and where each name can be said.

A CTE is a name bound to a statement for the length of one statement. The
binding rules are short and every one of them is a rule somebody gets wrong,
so they are here rather than spread through whatever binds a `FROM`. See
docs/specs/sql/05-ast-and-binder.md section 12.

Entries bind in order and each is visible to the ones after it and to the body,
so a `WITH y AS (SELECT n FROM x), x AS (...)` is not a forward reference but a
missing table, and DuckDB reports it as one. A name that is already bound is a
parser error rather than a shadowing, and it is caught while parsing, so a
duplicate is refused even in a statement that never runs. A CTE hides a
registered frame of the same name, and an inner `WITH` hides an outer one, both
of which are shadowing rather than an error.

The column alias list is positional and it is a prefix. `WITH x(p) AS (SELECT 1
AS a, 2 AS b)` gives back `p` and `b`, and `WITH x(p, q, r)` over those same two
columns gives back `p` and `q` with the third name dropped on the floor. Neither
is an error, which means a rename list that is the wrong length is silently
half applied. That is DuckDB's and it is reproduced, because a query that binds
there and fails here is a query somebody has to rewrite for no gain.

Recursion is a property of the statement rather than of the keyword. A CTE that
names itself is recursive, `WITH RECURSIVE` is what permits it, and the error
when the keyword is missing says so. The keyword on its own is not enough: the
statement also has to be a `UNION` whose left side does not name the CTE, since
that side is the anchor the fixed point starts from. A self reference under an
`EXCEPT` or an `INTERSECT`, or with no set operation at all, gets the same
circular reference error as a missing keyword, which is DuckDB answering a
different question than the one the query asked.

A recursive entry may not carry its own `ORDER BY`, `LIMIT` or `OFFSET`, and
those are two separate messages. The rule is about the entry's own trailing
clause and nothing deeper, so a subquery inside the recursive term may order
all it likes.

What is not here is the decision to inline. DuckDB inlines a CTE unless it is
named more than once, and `MATERIALIZED` and `NOT MATERIALIZED` override that.
The hint and the count are both recorded here, and the planner decides, because
the decision is a cost question and this file knows no costs.
"""

from .ast import (
    Ast,
    CLAUSE_FROM,
    CLAUSE_GROUP,
    CLAUSE_HAVING,
    CLAUSE_PROJECTION,
    CLAUSE_QUALIFY,
    CLAUSE_WHERE,
    CLAUSE_WINDOW,
    EXPR_WINDOW,
    MATERIALIZE_DEFAULT,
    NO_NODE,
    REF_FUNCTION,
    REF_JOIN,
    REF_JOIN_USING,
    REF_PARENS,
    REF_SUBQUERY,
    REF_TABLE,
    STMT_CTE,
    STMT_GROUP,
    STMT_ITEM,
    STMT_MODIFIERS,
    STMT_ORDER,
    STMT_PIVOT,
    STMT_PIVOT_ON,
    STMT_QUERY,
    STMT_SELECT,
    STMT_SET_OPERATION,
    STMT_TABLE,
    STMT_UNPIVOT,
    STMT_VALUES,
    STMT_WINDOW,
)
from .catalog import NOT_FOUND, fold
from .classify import children as expression_children
from .subquery import SHAPE_NONE, about


comptime NOT_A_CTE: Int = NOT_FOUND
"""What a lookup gives back for a name no entry binds."""


struct Cte(Copyable, Movable):
    """One name a `WITH` binds, and what is known about it."""

    var name: String
    """The name as the query wrote it."""

    var key: String
    """The folded name, which is what a lookup compares."""

    var node: UInt32
    """The `STMT_CTE` node it was read from."""

    var statement: UInt32
    """The statement it stands for."""

    var columns: List[String]
    """The column alias list, in order, empty when none was written."""

    var materialize: UInt32
    """One of the `MATERIALIZE_` tags, which the planner reads."""

    var recursive: Bool
    """Whether the statement names the entry itself."""

    var uses: Int
    """How many times a later entry or the body names it."""

    def __init__(
        out self,
        name: StringSlice,
        node: UInt32,
        statement: UInt32,
        var columns: List[String],
        materialize: UInt32 = MATERIALIZE_DEFAULT,
    ):
        """One entry, before anything has counted its uses.

        Args:
            name: The name as written.
            node: The `STMT_CTE` node.
            statement: The statement it stands for.
            columns: The column alias list.
            materialize: One of the `MATERIALIZE_` tags.
        """
        self.name = String(name)
        self.key = fold(name)
        self.node = node
        self.statement = statement
        self.columns = columns^
        self.materialize = materialize
        self.recursive = False
        self.uses = 0


struct Ctes(Movable, Sized):
    """The entries of one `WITH`, in the order they were written."""

    var entries: List[Cte]
    """The entries. Position is the whole visibility rule."""

    var recursive: Bool
    """Whether the query wrote `WITH RECURSIVE`."""

    def __init__(out self, recursive: Bool = False):
        """An empty clause.

        Args:
            recursive: Whether the `RECURSIVE` keyword was written.
        """
        self.entries = List[Cte]()
        self.recursive = recursive

    def __len__(self) -> Int:
        """How many names it binds.

        Returns:
            The count.
        """
        return len(self.entries)

    def find(self, name: StringSlice) -> Int:
        """Looks a name up among every entry.

        Args:
            name: The name as written.

        Returns:
            The entry's position, or `NOT_A_CTE`.
        """
        return self.visible(name, len(self.entries))

    def visible(self, name: StringSlice, before: Int) -> Int:
        """Looks a name up among the entries written before a position.

        Args:
            name: The name as written.
            before: The position asking, so entry `before` cannot see itself.

        Returns:
            The entry's position, or `NOT_A_CTE`.
        """
        var key = fold(name)
        var last = before if before < len(self.entries) else len(self.entries)
        for at in range(last):
            if self.entries[at].key == key:
                return at
        return NOT_A_CTE

    def add(mut self, var entry: Cte) raises:
        """Binds one name, refusing one that is already bound.

        Args:
            entry: The entry.

        Raises:
            Error: If a name is bound twice.
        """
        if self.find(entry.name) != NOT_A_CTE:
            raise Error(duplicate_name(entry.name))
        self.entries.append(entry^)


def read_ctes(ast: Ast, statement: UInt32) raises -> Ctes:
    """Reads a statement's `WITH` into a clause, checking what it can.

    Args:
        ast: The arenas.
        statement: A `STMT_SELECT`.

    Returns:
        The clause, with each entry's recursion and use count filled in. A
        statement with no `WITH` gives back an empty clause.

    Raises:
        Error: If a name is bound twice, if an entry names itself without the
            keyword or without an anchor, or if a recursive entry carries its
            own `ORDER BY`, `LIMIT` or `OFFSET`.
    """
    if statement == 0 or Int(statement) >= len(ast.stmts):
        raise Error("a statement index that is not in the arena")
    ref head = ast.stmts[Int(statement)]
    if head.kind != STMT_SELECT:
        return Ctes()

    var out = Ctes(recursive=head.payload == 1)
    for node in ast.items(head.children):
        ref item = ast.stmts[Int(node)]
        if item.kind != STMT_CTE:
            raise Error("a WITH holding something that is not a CTE")
        var names = List[String]()
        for part in ast.items(item.children):
            names.append(String(ast.text(part)))
        out.add(
            Cte(
                ast.text(item.payload),
                node,
                item.a,
                names^,
                materialize=item.b,
            )
        )

    for at in range(len(out.entries)):
        ref entry = out.entries[at]
        var key = String(entry.key)
        entry.recursive = reference_count(ast, entry.statement, key) != 0
        if not entry.recursive:
            continue
        if not out.recursive or not anchored(ast, entry.statement, key):
            raise Error(circular_reference(entry.name))
        check_modifiers(ast, entry.statement)

    for at in range(len(out.entries)):
        var key = String(out.entries[at].key)
        # The body and its trailing clauses both, since an ORDER BY and a
        # LIMIT can each hold a subquery and the modifiers hang off the
        # statement rather than off the query node.
        var uses = reference_count(ast, head.a, key) + reference_count(
            ast, head.b, key
        )
        for later in range(at + 1, len(out.entries)):
            uses += reference_count(ast, out.entries[later].statement, key)
        out.entries[at].uses = uses
    return out^


def aliased(columns: List[String], names: List[String]) -> List[String]:
    """Applies a `WITH x(a, b)` alias list, which is a prefix and not a list.

    A name past the end of the query's own columns is dropped and a column
    past the end of the name list keeps the name it had. Neither is an error,
    which means a list of the wrong length is silently half applied, and that
    is DuckDB's.

    Args:
        columns: The names the statement produces, in order.
        names: The alias list, in order, possibly empty.

    Returns:
        The names the CTE offers.
    """
    var out = List[String]()
    for at in range(len(columns)):
        if at < len(names):
            out.append(names[at])
        else:
            out.append(columns[at])
    return out^


def reference_count(
    ast: Ast, statement: UInt32, key: StringSlice
) raises -> Int:
    """Counts how many times a statement names a table by a folded name.

    The walk crosses into subqueries, because a CTE named inside one is named,
    and stops at a statement whose own `WITH` binds the name again, because an
    inner binding is a different table with the same spelling.

    Args:
        ast: The arenas.
        statement: Where to start.
        key: The folded name.

    Returns:
        How many `REF_TABLE` nodes say it.

    Raises:
        Error: If the walk runs past the end of an arena, which would mean a
            cycle.
    """
    var found = 0
    var statements = List[UInt32]()
    var refs = List[UInt32]()
    var exprs = List[UInt32]()
    statements.append(statement)
    var steps = 0
    # A node is reached once, and an empty slot is pushed and thrown away, so
    # the bound is the arenas with room for the slots that held nothing.
    var budget = 8 * (len(ast.stmts) + len(ast.refs) + len(ast.exprs)) + 16

    while len(statements) != 0 or len(refs) != 0 or len(exprs) != 0:
        steps += 1
        if steps > budget:
            raise Error("a statement that refers back to itself")

        if len(exprs) != 0:
            var at = exprs.pop()
            if at == 0:
                continue
            var inner = about(ast, at)
            if inner.shape != SHAPE_NONE:
                statements.append(inner.statement)
                if inner.operand != NO_NODE:
                    exprs.append(inner.operand)
                continue
            if ast.exprs[Int(at)].kind == EXPR_WINDOW:
                # A window's own ORDER BY is a run of STMT_ORDER nodes in the
                # statement arena, so the expression walk does not see it.
                for order in ast.items(ast.exprs[Int(at)].a):
                    statements.append(order)
            for child in expression_children(ast, at):
                exprs.append(child)
            continue

        if len(refs) != 0:
            var at = refs.pop()
            if at == 0:
                continue
            ref item = ast.refs[Int(at)]
            if item.kind == REF_TABLE:
                var parts = ast.length(item.children)
                if (
                    parts == 1
                    and fold(ast.text(ast.at(item.children, 0))) == key
                ):
                    found += 1
                continue
            if item.kind == REF_SUBQUERY:
                statements.append(item.a)
                continue
            if item.kind == REF_FUNCTION:
                for argument in ast.items(item.children):
                    exprs.append(argument)
                continue
            if item.kind == REF_PARENS:
                refs.append(item.a)
                continue
            if item.kind == REF_JOIN or item.kind == REF_JOIN_USING:
                refs.append(item.a)
                refs.append(item.b)
                if item.kind == REF_JOIN:
                    for condition in ast.items(item.children):
                        exprs.append(condition)
                continue
            raise Error("a table reference kind with no walk rule")

        var at = statements.pop()
        if at == 0:
            continue
        ref item = ast.stmts[Int(at)]
        if item.kind == STMT_SELECT:
            var shadowed = False
            for node in ast.items(item.children):
                ref bound = ast.stmts[Int(node)]
                if (
                    bound.kind == STMT_CTE
                    and fold(ast.text(bound.payload)) == key
                ):
                    shadowed = True
                statements.append(bound.a)
            if shadowed:
                continue
            statements.append(item.a)
            statements.append(item.b)
            continue
        if item.kind == STMT_SET_OPERATION:
            statements.append(item.a)
            statements.append(item.b)
            continue
        if item.kind == STMT_QUERY:
            for entry in ast.items(item.b):
                exprs.append(entry)
            for entry in ast.items(ast.slot(item.children, CLAUSE_PROJECTION)):
                statements.append(entry)
            for entry in ast.items(ast.slot(item.children, CLAUSE_FROM)):
                refs.append(entry)
            for entry in ast.items(ast.slot(item.children, CLAUSE_GROUP)):
                statements.append(entry)
            for entry in ast.items(ast.slot(item.children, CLAUSE_WINDOW)):
                statements.append(entry)
            exprs.append(ast.slot(item.children, CLAUSE_WHERE))
            exprs.append(ast.slot(item.children, CLAUSE_HAVING))
            exprs.append(ast.slot(item.children, CLAUSE_QUALIFY))
            continue
        if item.kind == STMT_TABLE:
            var parts = ast.length(item.children)
            if parts == 1 and fold(ast.text(ast.at(item.children, 0))) == key:
                found += 1
            continue
        if item.kind == STMT_CTE:
            statements.append(item.a)
            continue
        if item.kind == STMT_VALUES:
            # A run of runs, since the rows do not have to be the same length.
            for row in ast.items(item.children):
                for entry in ast.items(row):
                    exprs.append(entry)
            continue
        if item.kind == STMT_MODIFIERS:
            for order in ast.items(item.children):
                statements.append(order)
            exprs.append(item.a)
            exprs.append(item.b)
            continue
        if item.kind == STMT_ITEM or item.kind == STMT_ORDER:
            exprs.append(item.a)
            continue
        if item.kind == STMT_GROUP:
            exprs.append(item.a)
            for nested in ast.items(item.children):
                statements.append(nested)
            continue
        if item.kind == STMT_WINDOW:
            exprs.append(item.a)
            continue
        if item.kind == STMT_PIVOT:
            refs.append(item.a)
            for on in ast.items(item.b):
                statements.append(on)
            for using in ast.items(item.children):
                statements.append(using)
            continue
        if item.kind == STMT_PIVOT_ON:
            exprs.append(item.a)
            statements.append(item.b)
            for entry in ast.items(item.children):
                statements.append(entry)
            continue
        if item.kind == STMT_UNPIVOT:
            refs.append(item.a)
            for entry in ast.items(item.children):
                statements.append(entry)
            continue
        raise Error("a statement kind with no walk rule")
    return found


def anchored(ast: Ast, statement: UInt32, key: StringSlice) raises -> Bool:
    """Whether a recursive entry has a first branch that does not recurse.

    DuckDB wants a `UNION`, with the anchor on the left and the recursive term
    on the right. `EXCEPT` and `INTERSECT` do not count, and neither does a
    statement with no set operation in it at all.

    Args:
        ast: The arenas.
        statement: The entry's statement.
        key: The entry's folded name.

    Returns:
        Whether there is an anchor to start the fixed point from.

    Raises:
        Error: If the walk runs past the end of an arena.
    """
    var at = statement
    var steps = 0
    while at != 0 and Int(at) < len(ast.stmts):
        steps += 1
        if steps > len(ast.stmts) + 1:
            raise Error("a statement that refers back to itself")
        ref item = ast.stmts[Int(at)]
        if item.kind == STMT_SELECT:
            at = item.a
            continue
        if item.kind != STMT_SET_OPERATION:
            return False
        if not ast.text(item.payload).startswith("UNION"):
            return False
        return reference_count(ast, item.a, key) == 0
    return False


def check_modifiers(ast: Ast, statement: UInt32) raises:
    """Refuses the trailing clauses a recursive entry may not carry.

    Two messages rather than one, because DuckDB has two. The rule reaches the
    entry's own trailing clause and nothing under it, so a subquery in the
    recursive term may order and limit as it likes.

    Args:
        ast: The arenas.
        statement: The entry's statement.

    Raises:
        Error: If it carries an `ORDER BY`, a `LIMIT` or an `OFFSET`.
    """
    if statement == 0 or Int(statement) >= len(ast.stmts):
        return
    ref head = ast.stmts[Int(statement)]
    if head.kind != STMT_SELECT or head.b == 0:
        return
    ref modifiers = ast.stmts[Int(head.b)]
    if ast.length(modifiers.children) != 0:
        raise Error(no_ordering())
    if modifiers.a != 0 or modifiers.b != 0:
        raise Error(no_limit())


def duplicate_name(name: StringSlice) -> String:
    """DuckDB's error for a `WITH` that binds one name twice.

    Args:
        name: The name, as the second entry wrote it.

    Returns:
        The message.
    """
    return String('Parser Error: Duplicate CTE name "', name, '"')


def circular_reference(name: StringSlice) -> String:
    """DuckDB's error for a CTE that names itself and may not.

    One message covers three different mistakes: the missing keyword, a self
    reference with no `UNION` around it, and one under an `EXCEPT` or an
    `INTERSECT`. Only the first of the three is what the message describes.

    Args:
        name: The entry's name.

    Returns:
        The message.
    """
    return String(
        'Binder Error: Circular reference to CTE "',
        name,
        '", use WITH RECURSIVE to use recursive CTEs.',
    )


def no_ordering() -> String:
    """DuckDB's error for an `ORDER BY` on a recursive entry.

    Returns:
        The message.
    """
    return String("Parser Error: ORDER BY in a recursive query is not allowed")


def no_limit() -> String:
    """DuckDB's error for a `LIMIT` or an `OFFSET` on a recursive entry.

    Returns:
        The message.
    """
    return String(
        "Parser Error: LIMIT or OFFSET in a recursive query is not allowed"
    )
