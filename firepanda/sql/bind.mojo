"""Where a name is looked up, and what it turns into.

A query level has a set of things a column can come from: the tables, subqueries
and joins in its `FROM`. This file holds that set, the chain of levels a
subquery sits inside, and the rules that turn a name written in a query into a
position. See docs/specs/sql/05-ast-and-binder.md section 8.

The output of resolution is a pair of indices and never a name. That is the
property the optimizer is later built on: a pass that reorders or duplicates
nodes cannot change what a column means, because there is no scope left for a
name to be reinterpreted in.

Four rules here are DuckDB's rather than ours, and each one is a wrong answer
rather than an error if it is not followed.

**A bare name that two bindings both have is ambiguous**, and the error names
both ways of writing it. The exception is a column merged by `USING` or
`NATURAL`, which is one column with two homes rather than two columns, so the
bare name is unambiguous and the qualified name still works from either side.

**A qualified name is tried longest first.** `a.b` is a column `b` of a table
`a` if `a` is a binding, and a field `b` of a struct column `a` if it is not.
The order those are attempted in is observable, so it is written down here
rather than left to whichever branch happened to come first.

**Resolution walks outward and records what it crossed.** A name that resolves
at an outer level makes the subquery correlated, and the exact outer column is
recorded at the point it is found rather than rediscovered later by a pass
walking for outer references. A pass that has to go looking is a pass that can
miss one.

**Names fold.** `SELECT X` and `select x` name the same column, and so does
`SELECT "X"`, because DuckDB is case insensitive throughout and quoting changes
what a name is rather than how it is compared. The transformer already folds a
bare identifier, so folding here is what makes the quoted one behave.
"""

from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Schema

from .catalog import NOT_FOUND, edit_distance, fold


comptime SOURCE_FRAME: UInt8 = 0
"""A frame out of the catalog."""


comptime SOURCE_SUBQUERY: UInt8 = 1
"""A parenthesized query in the `FROM`."""


comptime SOURCE_CTE: UInt8 = 2
"""A `WITH` binding."""


comptime SOURCE_JOIN: UInt8 = 3
"""A join whose `USING` or `NATURAL` columns were merged into one binding."""


comptime SOURCE_VALUES: UInt8 = 4
"""A `VALUES` list."""


comptime SOURCE_FUNCTION: UInt8 = 5
"""A table function, `read_csv` and the rest."""


comptime NO_BINDING: Int = -1
"""What a resolution that found nothing puts in `binding`."""


comptime AMBIGUOUS: Int = -2
"""What a resolution that found more than one puts in `binding`."""


comptime MAX_CANDIDATES: Int = 3
"""How many near misses an error offers.

DuckDB's number. Three is enough to cover a typo in a wide table and short
enough that the message stays one line.
"""


struct Column(Copyable, Movable):
    """One column a binding offers."""

    var name: String
    """The name as the query will have to write it, unfolded."""

    var key: String
    """The folded name, which is what resolution compares."""

    var dtype: LogicalType
    """The column's type."""

    var hidden: Bool
    """Whether a bare name skips this column.

    True for the right hand copy of a `USING` or `NATURAL` column. The merged
    column is one column with two qualified spellings, so `b.x` still finds it,
    and a bare `x` finds the left hand copy once rather than twice.
    """

    def __init__(out self, name: StringSlice, dtype: LogicalType):
        """A visible column.

        Args:
            name: The name as written.
            dtype: The type.
        """
        self.name = String(name)
        self.key = fold(name)
        self.dtype = dtype
        self.hidden = False


struct Binding(Movable):
    """One thing a column can come from at a query level."""

    var name: String
    """What the query calls it: the alias if there is one, else the table."""

    var key: String
    """The folded name, which is what a qualified reference compares."""

    var columns: List[Column]
    """The columns it offers, in the order `*` expands them."""

    var source: UInt8
    """Which of the six kinds of thing it is."""

    var slot: Int
    """Where the thing itself is, meaning depends on the source.

    A catalog entry index for a frame, a plan child index for a subquery, a CTE
    index for a CTE. The binder that built the binding knows; nothing here
    looks at it.
    """

    def __init__(out self, name: StringSlice, source: UInt8, slot: Int):
        """An empty binding, to be filled with columns.

        Args:
            name: What the query calls it.
            source: One of the `SOURCE_` constants.
            slot: Where the underlying thing is.
        """
        self.name = String(name)
        self.key = fold(name)
        self.columns = List[Column]()
        self.source = source
        self.slot = slot

    def add(mut self, name: StringSlice, dtype: LogicalType):
        """Appends a visible column.

        Args:
            name: The column name as written.
            dtype: Its type.
        """
        self.columns.append(Column(name, dtype))

    def add_schema(mut self, schema: Schema):
        """Appends every column of a frame's schema, in order.

        Args:
            schema: The schema.
        """
        for field in schema.fields:
            self.columns.append(Column(field.name, field.dtype))

    def find(self, key: StringSlice, *, visible_only: Bool) -> Int:
        """Looks a folded column name up in this binding.

        Args:
            key: The folded name.
            visible_only: Whether to skip the merged away side of a `USING`.

        Returns:
            The column's position, or `NOT_FOUND`.
        """
        for at in range(len(self.columns)):
            if visible_only and self.columns[at].hidden:
                continue
            if self.columns[at].key == key:
                return at
        return NOT_FOUND


@fieldwise_init
struct Reference(Copyable, ImplicitlyCopyable, Movable):
    """What a resolved name turned into."""

    var binding: Int
    """The binding's position at its level, `NO_BINDING` or `AMBIGUOUS`."""

    var column: Int
    """The column's position within that binding."""

    var depth: Int
    """How many levels out it was found. Zero is the level that asked."""

    def found(self) -> Bool:
        """Whether this is a resolution rather than a failure.

        Returns:
            True if a single binding claimed the name.
        """
        return self.binding >= 0

    def correlated(self) -> Bool:
        """Whether resolving it crossed a query level.

        Returns:
            True if the name came from an outer query, which is what makes the
            subquery it was written in correlated.
        """
        return self.binding >= 0 and self.depth > 0


struct Scope(Movable):
    """One query level: its bindings, its parent, and what it reached out for.

    A level is a `SELECT`'s `FROM`, and a subquery inside it is another level
    whose parent is this one. The parent is an index rather than a pointer,
    because the levels live in one list on the binder and an index survives that
    list growing.
    """

    var bindings: List[Binding]
    """The `FROM` entries, in the order they were written."""

    var parent: Int
    """The enclosing level, or `NOT_FOUND` at the top."""

    var correlations: List[Reference]
    """Every outer column this level or a level under it reached for.

    Recorded here rather than discovered later, because a decorrelation pass
    that has to go looking for outer references is a pass that can miss one.
    Each entry's depth is counted from the level that reached, so an entry with
    depth 2 means two levels above the one that recorded it.
    """

    def __init__(out self, parent: Int):
        """An empty level.

        Args:
            parent: The enclosing level's index, or `NOT_FOUND`.
        """
        self.bindings = List[Binding]()
        self.parent = parent
        self.correlations = List[Reference]()

    def find(self, key: StringSlice) -> Int:
        """Looks a folded binding name up at this level.

        Args:
            key: The folded name.

        Returns:
            The binding's position, or `NOT_FOUND`.
        """
        for at in range(len(self.bindings)):
            if self.bindings[at].key == key:
                return at
        return NOT_FOUND


struct Scopes(Movable, Sized):
    """The chain of query levels, and the resolution that walks it."""

    var levels: List[Scope]
    """Every level opened so far. Indices into this are stable."""

    def __init__(out self):
        """Constructs a chain with no levels in it."""
        self.levels = List[Scope]()

    def __len__(self) -> Int:
        """How many levels have been opened.

        Returns:
            The count, including closed ones, since indices stay valid.
        """
        return len(self.levels)

    def open(mut self, parent: Int) -> Int:
        """Opens a level under another one.

        Args:
            parent: The enclosing level, or `NOT_FOUND` for the top.

        Returns:
            The new level's index.
        """
        self.levels.append(Scope(parent))
        return len(self.levels) - 1

    def add(
        mut self, level: Int, name: StringSlice, source: UInt8, slot: Int
    ) -> Int:
        """Adds a binding to a level.

        Args:
            level: The level.
            name: What the query calls the binding.
            source: One of the `SOURCE_` constants.
            slot: Where the underlying thing is.

        Returns:
            The binding's position at that level.
        """
        self.levels[level].bindings.append(Binding(name, source, slot))
        return len(self.levels[level].bindings) - 1

    def merge(mut self, level: Int, right: Int, name: StringSlice):
        """Marks a column merged by `USING` or `NATURAL`.

        The right hand copy stops answering a bare name and keeps answering its
        qualified one, which is what makes `SELECT x FROM a JOIN b USING (x)`
        one column rather than an ambiguity, while `b.x` still means something.

        Args:
            level: The level both bindings are at.
            right: The binding whose copy is hidden.
            name: The merged column's name.
        """
        var key = fold(name)
        var at = (
            self.levels[level].bindings[right].find(key, visible_only=False)
        )
        if at != NOT_FOUND:
            self.levels[level].bindings[right].columns[at].hidden = True

    def resolve(mut self, level: Int, name: StringSlice) -> Reference:
        """Resolves a bare column name, walking outward.

        Args:
            level: The level the name was written at.
            name: The name as written, quoted or bare.

        Returns:
            The reference, which may be `NO_BINDING` or `AMBIGUOUS`.
        """
        var key = fold(name)
        var at = level
        var depth = 0
        while at != NOT_FOUND:
            var found = self._bare(at, key)
            if found.binding != NO_BINDING:
                var answer = Reference(found.binding, found.column, depth)
                if answer.correlated():
                    self.levels[level].correlations.append(answer)
                return answer
            at = self.levels[at].parent
            depth += 1
        return Reference(NO_BINDING, NOT_FOUND, 0)

    def resolve_qualified(
        mut self, level: Int, table: StringSlice, column: StringSlice
    ) -> Reference:
        """Resolves `table.column`, walking outward.

        A qualified name cannot be ambiguous at one level, because two bindings
        cannot share a name, so the only question is which level answers.

        Args:
            level: The level the name was written at.
            table: The binding's name as written.
            column: The column's name as written.

        Returns:
            The reference, or `NO_BINDING` if no level has that table with that
            column in it.
        """
        var table_key = fold(table)
        var column_key = fold(column)
        var at = level
        var depth = 0
        while at != NOT_FOUND:
            var binding = self.levels[at].find(table_key)
            if binding != NOT_FOUND:
                var found = (
                    self.levels[at]
                    .bindings[binding]
                    .find(column_key, visible_only=False)
                )
                if found != NOT_FOUND:
                    var answer = Reference(binding, found, depth)
                    if answer.correlated():
                        self.levels[level].correlations.append(answer)
                    return answer
            at = self.levels[at].parent
            depth += 1
        return Reference(NO_BINDING, NOT_FOUND, 0)

    def knows_table(self, level: Int, table: StringSlice) -> Bool:
        """Whether any level in the chain has a binding by that name.

        This is the question that decides what `a.b` means. If `a` is a binding
        the reference is a column, and if it is not the reference is a field of
        a struct column called `a`, so the longest interpretation is tried by
        asking this first.

        Args:
            level: The level the name was written at.
            table: The name before the dot.

        Returns:
            True if some level binds it.
        """
        var key = fold(table)
        var at = level
        while at != NOT_FOUND:
            if self.levels[at].find(key) != NOT_FOUND:
                return True
            at = self.levels[at].parent
        return False

    def visible_columns(self, level: Int) -> List[Reference]:
        """Every column a `*` at this level expands to, in order.

        Args:
            level: The level.

        Returns:
            One reference per visible column, bindings in `FROM` order and
            columns in schema order, which is the order the corpus observes.
        """
        var out = List[Reference]()
        for binding in range(len(self.levels[level].bindings)):
            ref columns = self.levels[level].bindings[binding].columns
            for at in range(len(columns)):
                if not columns[at].hidden:
                    out.append(Reference(binding, at, 0))
        return out^

    def ambiguity(self, level: Int, name: StringSlice) -> String:
        """DuckDB's error for a bare name two bindings both have.

        Args:
            level: The level the name was written at.
            name: The name as written.

        Returns:
            The message, naming both ways of writing it.
        """
        var key = fold(name)
        var first = String()
        var second = String()
        for at in range(len(self.levels[level].bindings)):
            if self.levels[level].bindings[at].find(key, visible_only=True) == (
                NOT_FOUND
            ):
                continue
            var qualified = String(
                self.levels[level].bindings[at].name, ".", name
            )
            if first.byte_length() == 0:
                first = qualified
            elif second.byte_length() == 0:
                second = qualified
        return String(
            'Binder Error: Ambiguous reference to column name "',
            name,
            '" (use: "',
            first,
            '" or "',
            second,
            '")',
        )

    def no_such_column(self, level: Int, name: StringSlice) -> String:
        """DuckDB's error for a bare name nothing has.

        Args:
            level: The level the name was written at.
            name: The name as written.

        Returns:
            The message, with up to three near misses after it.
        """
        var message = String(
            'Binder Error: Referenced column "',
            name,
            '" not found in FROM clause!',
        )
        var near = self._near_columns(level, name)
        if len(near) == 0:
            return message
        return String(message, "\nCandidate bindings: ", _quoted_list(near))

    def no_such_table(self, level: Int, table: StringSlice) -> String:
        """DuckDB's error for a qualifier nothing binds.

        Args:
            level: The level the name was written at.
            table: The name before the dot.

        Returns:
            The message, listing the tables that are in scope.
        """
        var names = List[String]()
        for at in range(len(self.levels[level].bindings)):
            names.append(self.levels[level].bindings[at].name)
        var message = String(
            'Binder Error: Referenced table "', table, '" not found!'
        )
        if len(names) == 0:
            return message
        return String(message, "\nCandidate tables: ", _quoted_list(names))

    def no_such_column_in(
        self, level: Int, table: StringSlice, column: StringSlice
    ) -> String:
        """DuckDB's error for a table that has no such column.

        Args:
            level: The level the name was written at.
            table: The binding's name.
            column: The column that is not in it.

        Returns:
            The message, listing that binding's columns.
        """
        var message = String(
            'Binder Error: Table "',
            table,
            '" does not have a column named "',
            column,
            '"',
        )
        var at = self.levels[level].find(fold(table))
        if at == NOT_FOUND:
            return message
        var names = List[String]()
        for entry in self.levels[level].bindings[at].columns:
            names.append(entry.name)
        if len(names) == 0:
            return message
        return String(message, "\n\nCandidate bindings: ", _quoted_list(names))

    def _bare(self, level: Int, key: StringSlice) -> Reference:
        """Looks a folded bare name up at one level and no further.

        Args:
            level: The level.
            key: The folded name.

        Returns:
            The reference, `NO_BINDING` if nothing has it, `AMBIGUOUS` if more
            than one binding does.
        """
        var binding = NO_BINDING
        var column = NOT_FOUND
        for at in range(len(self.levels[level].bindings)):
            var found = (
                self.levels[level].bindings[at].find(key, visible_only=True)
            )
            if found == NOT_FOUND:
                continue
            if binding != NO_BINDING:
                return Reference(AMBIGUOUS, NOT_FOUND, 0)
            binding = at
            column = found
        return Reference(binding, column, 0)

    def _near_columns(self, level: Int, name: StringSlice) -> List[String]:
        """The columns at a level whose names are closest to one that missed.

        Args:
            level: The level.
            name: The name that missed.

        Returns:
            Up to `MAX_CANDIDATES` names, closest first, ties in `FROM` order.
        """
        var probe = fold(name)
        var names = List[String]()
        var distances = List[Int]()
        for binding in range(len(self.levels[level].bindings)):
            ref columns = self.levels[level].bindings[binding].columns
            for column in columns:
                if column.hidden:
                    continue
                if _holds(names, column.name):
                    continue
                var distance = edit_distance(probe, column.key, 64)
                var at = len(names)
                while at > 0 and distances[at - 1] > distance:
                    at -= 1
                names.insert(at, column.name)
                distances.insert(at, distance)
        while len(names) > MAX_CANDIDATES:
            _ = names.pop()
        return names^


def _holds(names: List[String], name: StringSlice) -> Bool:
    """Whether a candidate list already offers a name.

    Two bindings that both have a column called `x` are one suggestion and not
    two, because the suggestion is what to type and typing it twice does not
    help.

    Args:
        names: The candidates so far.
        name: The name being considered.

    Returns:
        True if it is already there.
    """
    for at in range(len(names)):
        if names[at] == name:
            return True
    return False


def _quoted_list(names: List[String]) -> String:
    """Renders names the way DuckDB renders a candidate list.

    Args:
        names: The names, already in the order they should appear.

    Returns:
        Each one double quoted, comma separated.
    """
    var out = String()
    for at in range(len(names)):
        if at > 0:
            out += ", "
        out += String('"', names[at], '"')
    return out^
