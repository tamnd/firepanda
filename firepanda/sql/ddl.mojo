"""The statements that change what a session's names hold.

`CREATE TABLE`, `CREATE TABLE ... AS`, `CREATE VIEW`, `INSERT INTO` and `DROP
TABLE` or `DROP VIEW`. Each one is a frame operation with SQL spelling, which
is document 05's tier two: a table is a frame under a name in the catalog, a
view is a query under a name, an insert is a concatenation, and a drop takes
the name away. Nothing here outlives the session, because the catalog does not.

They are read straight off the parse rather than through the transform, since
none of them has an expression of its own worth an AST node. The one part that
does, the query a `CREATE TABLE ... AS`, a `CREATE VIEW` or an `INSERT` carries,
goes through the transform and the lowering as any query does.

What a statement says and this does not do is refused by name, never parsed and
dropped: a `PRIMARY KEY` nothing would check, a `DEFAULT` nothing would fill, an
`ON CONFLICT` with no key to conflict on. A `NOT NULL` is kept, on the field,
and an insert checks it. Anything that is not one of these statements goes on to
`Dialect.run`, which runs a query and refuses the rest with the sentence it
always had.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.dtype.logical import LogicalType, TypeKind
from firepanda.dtype.schema import Field, Schema
from firepanda.frame import DataFrame
from firepanda.kernel.binary import all_null
from firepanda.kernel.cast import cast_any
from firepanda.kernel.concat import concat_any

from .ast import NO_NODE, Ast
from .catalog import KIND_FRAME, KIND_VIEW, NOT_FOUND, Catalog, View, fold
from .matcher import Parse, parse_rule
from .run import Dialect
from .token import token_text
from .types import engine_type, parse_type


def execute(
    dialect: Dialect, sql: StringSlice, mut catalog: Catalog
) raises -> DataFrame:
    """Runs one statement, which may change what the catalog holds.

    Args:
        dialect: The grammar, the transform and the function catalog.
        sql: The statement, with or without a trailing semicolon.
        catalog: The session's names, written to by a `CREATE`, an `INSERT` or
            a `DROP`.

    Returns:
        What the statement answers: a query's rows, one row with the count of
        rows an `INSERT` added in a column called `Count`, as DuckDB answers,
        and a frame with no columns for the rest.

    Raises:
        If the statement does not parse, is a shape this refuses, or names
        something the catalog does or does not hold against what it says.
    """
    var tree = parse_rule(sql, dialect.grammar, dialect.rules.statement_rule)
    var reading = _Read(tree^, sql, dialect)
    # The search stops at a statement below where it starts, so it starts at
    # the one statement this is.
    var top = reading.find(reading.tree.root, "Statement")
    if top == NO_NODE:
        top = reading.tree.root
    var create = reading.find(top, "CreateStatement")
    if create != NO_NODE:
        var variation = reading.find(create, "CreateTableStmt")
        if variation != NO_NODE:
            return _create_table(dialect, reading, create, variation, catalog)
        variation = reading.find(create, "CreateViewStmt")
        if variation != NO_NODE:
            return _create_view(dialect, reading, create, variation, catalog)
        return dialect.run(sql, catalog)
    var insert = reading.find(top, "InsertStatement")
    if insert != NO_NODE:
        return _insert(dialect, reading, insert, catalog)
    var drop = reading.find(top, "DropStatement")
    if drop != NO_NODE:
        var table = reading.find(drop, "DropTable")
        if table != NO_NODE:
            return _drop(reading, drop, table, catalog)
    return dialect.run(sql, catalog)


struct _Read(Movable):
    """A reading, the text under it, and the name of every rule it can hold."""

    var tree: Parse
    var sql: String
    var names: List[String]

    def __init__(out self, var tree: Parse, sql: StringSlice, dialect: Dialect):
        self.tree = tree^
        self.sql = String(sql)
        self.names = dialect.grammar.names.copy()

    def name(self, node: UInt32) -> String:
        """The rule a node matched."""
        return self.names[Int(self.tree.nodes[Int(node)].rule)].copy()

    def find(self, node: UInt32, rule: StringSlice) -> UInt32:
        """The first node under `node`, itself included, that matched `rule`.

        Depth first and left to right, and not into a query, so a rule a query
        can also hold is only ever found in the statement around it.

        Returns:
            The node, or `NO_NODE`.
        """
        var stack = List[UInt32]()
        stack.append(node)
        while len(stack) > 0:
            var at = stack.pop()
            var called = self.name(at)
            if called == rule:
                return at
            if at != node and (
                called == "Statement" or called == "SelectStatementInternal"
            ):
                continue
            var below = self.tree.children(at)
            for i in range(len(below) - 1, -1, -1):
                stack.append(below[i])
        return NO_NODE

    def all(self, node: UInt32, rule: StringSlice) -> List[UInt32]:
        """Every node under `node` that matched `rule`, in order, without
        looking inside a match or inside a query."""
        var out = List[UInt32]()
        var stack = List[UInt32]()
        stack.append(node)
        while len(stack) > 0:
            var at = stack.pop()
            var called = self.name(at)
            if called == rule:
                out.append(at)
                continue
            if at != node and (
                called == "Statement" or called == "SelectStatementInternal"
            ):
                continue
            var below = self.tree.children(at)
            for i in range(len(below) - 1, -1, -1):
                stack.append(below[i])
        return out^

    def has(self, node: UInt32, rule: StringSlice) -> Bool:
        return self.find(node, rule) != NO_NODE

    def text(self, node: UInt32) -> String:
        """The statement text a node covers, as written."""
        var first = Int(self.tree.nodes[Int(node)].token_start)
        var end = Int(self.tree.nodes[Int(node)].token_end)
        if end <= first:
            return String()
        var start = Int(self.tree.tokens[first].start)
        var last = self.tree.tokens[end - 1]
        var stop = Int(last.start) + Int(last.length)
        return String(
            StringSlice(unsafe_from_utf8=self.sql.as_bytes()[start:stop])
        )

    def run(
        self, dialect: Dialect, node: UInt32, catalog: Catalog
    ) raises -> DataFrame:
        """Transforms the query at `node`, and lowers and runs it."""
        var ast = Ast()
        var statement = dialect.rules.walk(self.tree, self.sql, node, ast)
        return dialect.run_ast(ast, statement, catalog)

    def check(self, dialect: Dialect, node: UInt32, catalog: Catalog) raises:
        """Transforms and lowers the query at `node` without running it, so a
        name it cannot resolve is an error now rather than at every use."""
        var ast = Ast()
        var statement = dialect.rules.walk(self.tree, self.sql, node, ast)
        dialect.check_ast(ast, statement, catalog)


def _identifier(text: StringSlice) -> String:
    """A name as a statement wrote it, with its quotes off and any schema in
    front of it dropped, since a session has the one schema."""
    var parts = List[String]()
    var current = List[Byte]()
    var quote = Byte(0)
    var bytes = text.as_bytes()
    var i = 0
    while i < len(bytes):
        var c = bytes[i]
        if quote != 0:
            if c == quote:
                if i + 1 < len(bytes) and bytes[i + 1] == quote:
                    current.append(c)
                    i += 2
                    continue
                quote = 0
            else:
                current.append(c)
        elif c == Byte(ord('"')) or c == Byte(ord("'")):
            quote = c
        elif c == Byte(ord(".")):
            parts.append(String(StringSlice(unsafe_from_utf8=Span(current))))
            current.clear()
        elif c != Byte(ord(" ")):
            current.append(c)
        i += 1
    parts.append(String(StringSlice(unsafe_from_utf8=Span(current))))
    return parts[len(parts) - 1].copy()


def _already(kind: StringSlice, name: StringSlice) -> Error:
    return Error(
        String(
            "Catalog Error: ", kind, ' with name "', name, '" already exists!'
        )
    )


def _nothing() -> DataFrame:
    return DataFrame()


def _refuse(what: StringSlice, why: StringSlice) raises:
    raise Error(String("firepanda does not support ", what, ", ", why))


def _create_table(
    dialect: Dialect,
    reading: _Read,
    create: UInt32,
    table: UInt32,
    mut catalog: Catalog,
) raises -> DataFrame:
    """`CREATE TABLE t (a INTEGER, ...)` and `CREATE TABLE t AS query`."""
    var name = _identifier(reading.text(reading.find(table, "QualifiedName")))
    var replace = reading.has(create, "OrReplace")
    var if_missing = reading.has(table, "IfNotExists")
    if reading.has(table, "CommitAction"):
        _refuse(
            "ON COMMIT",
            "because a session has no transactions for it to act at the end of",
        )
    if reading.has(table, "PartitionSortedOptions"):
        _refuse(
            "PARTITIONED BY or SORTED BY on a table",
            "because a table is a frame held in memory as it came",
        )
    if reading.has(table, "WithList"):
        _refuse("WITH options on a table", "because it has no storage to tune")

    var at = catalog.find(name)
    if at != NOT_FOUND:
        if if_missing:
            return _nothing()
        if not replace:
            raise _already("Table", name)
        if catalog.kind_at(at) != KIND_FRAME:
            raise Error(
                String(
                    "Catalog Error: Existing object ",
                    name,
                    " is of type View, trying to replace with type Table",
                )
            )

    var built = reading.find(table, "CreateTableAs")
    if built != NO_NODE:
        if reading.has(built, "WithNoData"):
            _refuse(
                "WITH NO DATA",
                "because it would keep the columns of a query it never ran",
            )
        var rows = reading.run(
            dialect, reading.find(built, "Statement"), catalog
        )
        var listed = reading.find(built, "IdentifierList")
        if listed != NO_NODE:
            var names = reading.all(listed, "Identifier")
            if len(names) > rows.width():
                raise Error(
                    String(
                        "Binder Error: Target table has ",
                        rows.width(),
                        " columns but ",
                        len(names),
                        " column names were specified",
                    )
                )
            var fields = List[Field]()
            var columns = List[AnyArray]()
            for i in range(rows.width()):
                var called = rows.schema[i].name.copy()
                if i < len(names):
                    called = _identifier(reading.text(names[i]))
                fields.append(Field(called, rows.schema[i].dtype))
                columns.append(_whole(rows, i))
            rows = DataFrame(Schema(fields^), columns^)
        catalog.register(name, rows^)
        return _nothing()

    var fields = List[Field]()
    var columns = List[AnyArray]()
    var list = reading.find(table, "CreateColumnList")
    if reading.has(list, "CreateTableConstraint"):
        _refuse(
            "a table constraint",
            "because a table is a frame and nothing would check it",
        )
    var seen = List[String]()
    for column in reading.all(list, "ColumnDefinition"):
        var called = _identifier(
            reading.text(reading.find(column, "DottedIdentifier"))
        )
        var key = fold(called)
        for i in range(len(seen)):
            if seen[i] == key:
                raise Error(
                    String(
                        'Catalog Error: Column with name "',
                        called,
                        '" already exists!',
                    )
                )
        seen.append(key^)
        var written = reading.find(column, "Type")
        if written == NO_NODE:
            _refuse(
                "a column with no type",
                "because a frame's column has to have one",
            )
        if reading.has(column, "GeneratedColumn"):
            _refuse("a generated column", "because nothing would compute it")
        if reading.has(column, "ConstraintNameClause"):
            _refuse("a named constraint", "because nothing would check it")
        var nullable = True
        for constraint in reading.all(column, "ColumnConstraint"):
            if reading.has(constraint, "NotNullColumnConstraint"):
                nullable = False
            elif not reading.has(constraint, "NullConstraint"):
                _refuse(
                    String(reading.text(constraint).upper(), " on a column"),
                    (
                        "because a table is a frame and nothing would check or"
                        " fill it. NOT NULL is the constraint it keeps"
                    ),
                )
        var type = engine_type(parse_type(reading.text(written)))
        var field = Field(called, type)
        field.nullable = nullable
        fields.append(field^)
        columns.append(_nulls(type, 0))
    if len(fields) == 0:
        raise Error("Parser Error: Table must have at least one column!")
    catalog.register(name, DataFrame(Schema(fields^), columns^))
    return _nothing()


def _create_view(
    dialect: Dialect,
    reading: _Read,
    create: UInt32,
    view: UInt32,
    mut catalog: Catalog,
) raises -> DataFrame:
    """`CREATE VIEW v [(a, b)] AS query`."""
    var name = _identifier(reading.text(reading.find(view, "QualifiedName")))
    if reading.has(view, "CreateRecursive"):
        _refuse(
            "CREATE RECURSIVE VIEW",
            "and the same query written WITH RECURSIVE inside a view runs",
        )
    if reading.has(view, "WithList"):
        _refuse("WITH options on a view", "because it has no storage to tune")
    var at = catalog.find(name)
    if at != NOT_FOUND:
        if reading.has(view, "IfNotExists"):
            return _nothing()
        if not reading.has(create, "OrReplace"):
            raise _already("View", name)
        if catalog.kind_at(at) != KIND_VIEW:
            raise Error(
                String(
                    "Catalog Error: Existing object ",
                    name,
                    " is of type Table, trying to replace with type View",
                )
            )
    var query = reading.find(view, "SelectStatementInternal")
    reading.check(dialect, query, catalog)
    var columns = List[String]()
    var listed = reading.find(view, "InsertColumnList")
    if listed != NO_NODE:
        for entry in reading.all(listed, "ColId"):
            columns.append(_identifier(reading.text(entry)))
    catalog.define(name, View(reading.text(query), columns^))
    return _nothing()


def _insert(
    dialect: Dialect, reading: _Read, insert: UInt32, mut catalog: Catalog
) raises -> DataFrame:
    """`INSERT INTO t [(a, b)] [BY NAME] query` and `... DEFAULT VALUES`."""
    if reading.has(insert, "WithClause"):
        _refuse(
            "a WITH in front of an INSERT",
            "and the same WITH written inside the query it inserts runs",
        )
    if reading.has(insert, "OrAction"):
        _refuse(
            "INSERT OR REPLACE or INSERT OR IGNORE",
            "because a table here has no key for a row to conflict on",
        )
    if reading.has(insert, "OnConflictClause"):
        _refuse(
            "ON CONFLICT",
            "because a table here has no key for a row to conflict on",
        )
    if reading.has(insert, "ReturningClause"):
        _refuse("RETURNING", "yet")

    var target = reading.find(insert, "InsertTarget")
    var name = _identifier(reading.text(reading.find(target, "BaseTableName")))
    var at = catalog.find(name)
    if at == NOT_FOUND:
        raise Error(catalog.missing(name))
    if catalog.kind_at(at) != KIND_FRAME:
        raise Error(String("Catalog Error: ", name, " is not an table"))
    var table = catalog.frame_at(at).copy()
    var width = table.width()

    var rows = DataFrame()
    var count = 1
    var values = reading.find(insert, "SelectInsertValues")
    if values != NO_NODE:
        rows = reading.run(
            dialect, reading.find(values, "SelectStatementInternal"), catalog
        )
        count = len(rows)

    # Where each column of the rows lands in the table, which is the table's
    # own order unless a column list or BY NAME says otherwise.
    var places = List[Int]()
    var listed = reading.find(insert, "InsertColumnList")
    var by_name = reading.has(insert, "InsertByNameOrder")
    if listed != NO_NODE and by_name:
        raise Error(
            "Binder Error: Can't combine INSERT BY NAME with a column list"
        )
    if listed != NO_NODE or by_name:
        var named = List[String]()
        if listed != NO_NODE:
            for entry in reading.all(listed, "ColId"):
                named.append(_identifier(reading.text(entry)))
        else:
            for i in range(rows.width()):
                named.append(rows.schema[i].name.copy())
        for i in range(len(named)):
            var found = -1
            for j in range(width):
                if fold(table.schema[j].name) == fold(named[i]):
                    found = j
            if found < 0:
                raise Error(
                    String(
                        'Binder Error: Table "',
                        name,
                        '" does not have a column with name "',
                        named[i],
                        '"',
                    )
                )
            for k in range(len(places)):
                if places[k] == found:
                    raise Error(
                        String(
                            'Binder Error: Duplicate column name "',
                            named[i],
                            '" in INSERT',
                        )
                    )
            places.append(found)
        if values != NO_NODE and len(places) != rows.width():
            raise Error(
                String(
                    "Binder Error: Column name/value mismatch for insert on ",
                    name,
                    ": expected ",
                    len(places),
                    " columns but ",
                    rows.width(),
                    " values were supplied",
                )
            )
    elif values != NO_NODE:
        if rows.width() != width:
            raise Error(
                String(
                    "Binder Error: table ",
                    name,
                    " has ",
                    width,
                    " columns but ",
                    rows.width(),
                    " values were supplied",
                )
            )
        for j in range(width):
            places.append(j)

    var columns = List[AnyArray](capacity=width)
    for j in range(width):
        var type = table.schema[j].dtype
        var added = _nulls(type, count)
        for i in range(len(places)):
            if places[i] == j and values != NO_NODE:
                var given = _whole(rows, i)
                if given.null_count() != len(given):
                    added = cast_any(given, type)
        if not table.schema[j].nullable and added.null_count() != 0:
            raise Error(
                String(
                    "Constraint Error: NOT NULL constraint failed: ",
                    name,
                    ".",
                    table.schema[j].name,
                )
            )
        var parts = table.columns[j].chunks.copy()
        parts.append(added^)
        columns.append(concat_any(parts))
    catalog.register(name, DataFrame(table.schema.copy(), columns^))

    var counted = Array[DType.int64](1)
    counted[0] = Int64(count)
    var fields = List[Field]()
    fields.append(Field("Count", LogicalType.INT64))
    var out = List[AnyArray]()
    out.append(AnyArray(counted^))
    return DataFrame(Schema(fields^), out^)


def _drop(
    reading: _Read, drop: UInt32, table: UInt32, mut catalog: Catalog
) raises -> DataFrame:
    """`DROP TABLE [IF EXISTS] t, ...` and the same for `VIEW`."""
    if (
        reading.has(drop, "DropBehavior")
        and reading.text(reading.find(drop, "DropBehavior")).upper()
        == "CASCADE"
    ):
        _refuse(
            "DROP ... CASCADE",
            "because a view here is its text and nothing records what it reads",
        )
    var view = (
        reading.text(reading.find(table, "TableOrView")).upper() == "VIEW"
    )
    var if_there = reading.has(table, "IfExists")
    for entry in reading.all(table, "BaseTableName"):
        var name = _identifier(reading.text(entry))
        var at = catalog.find(name)
        if at == NOT_FOUND:
            if if_there:
                continue
            var message = catalog.missing(name)
            if view:
                message = message.replace(
                    "Catalog Error: Table with", "Catalog Error: View with"
                )
            raise Error(message)
        var is_view = catalog.kind_at(at) == KIND_VIEW
        if is_view != view:
            raise Error(
                String(
                    "Catalog Error: Existing object ",
                    name,
                    " is of type ",
                    "View" if is_view else "Table",
                    ", trying to drop type ",
                    "View" if view else "Table",
                )
            )
        _ = catalog.drop(name)
    return _nothing()


def _whole(frame: DataFrame, i: Int) raises -> AnyArray:
    """A frame's column as one array, however many chunks it came in."""
    var parts = frame.columns[i].chunks.copy()
    if len(parts) == 0:
        return _nulls(frame.schema[i].dtype, 0)
    if len(parts) == 1:
        return parts[0].copy()
    return concat_any(parts)


def _nulls(type: LogicalType, rows: Int) raises -> AnyArray:
    """A column of `type` with every row missing, strings included, which
    `all_null` would lay out as bytes."""
    if type.kind == TypeKind.STRING:
        var builder = StringBuilder(rows)
        for _ in range(rows):
            builder.append_null()
        return AnyArray(builder^.finish())
    return all_null(type, rows)
