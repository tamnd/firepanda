"""What a `*` stands for, and the three modifiers that can be hung off it.

`bind.mojo` already knows the columns a query level offers and the order they
come back in. This is what a query is allowed to write in front of that: a
bare `*`, a qualified `t.*`, and `EXCLUDE`, `REPLACE` and `RENAME` over either
of them. See docs/specs/sql/05-ast-and-binder.md section 9.

The modifiers apply in the order the grammar forces them to be written, which
is exclude, then replace, then rename. Writing them the other way round is a
syntax error rather than a different meaning, so there is one order to
implement and no precedence question.

A bare modifier name applies to every column that has it. `SELECT * EXCLUDE
(a) FROM t, u` where both tables have an `a` removes both, and the matching
`RENAME` renames both and gives the query two columns with the same name,
which DuckDB allows. A qualified one applies to a single binding, so `EXCLUDE
(t.a)` leaves `u.a` alone. `REPLACE` cannot be qualified at all, because the
name after its `AS` is an identifier and a dot there is a syntax error.

Two things here are DuckDB's and are wrong. A `REPLACE` naming a column that
two bindings both have replaces the first one and silently drops the second,
so `SELECT * REPLACE (99 AS a) FROM t, u` gives three columns where `SELECT *`
gives four and nothing says a column went missing. And a duplicate entry in a
`RENAME` list is reported as a duplicate in the `EXCLUDE` list, naming a list
the query did not write. Both are reproduced, because a query that binds here
and fails there is worse than a query that is wrong the same way in both.

A qualified star does not walk outward the way a qualified column does.
`SELECT (SELECT o.* FROM u) FROM t o` is an error while `SELECT (SELECT o.a
FROM u) FROM t o` is fine, so the qualifier is looked up at one level only.

What is not here is `COLUMNS()`, which is refused by name in
`unsupported.mojo`, and struct unpacking, where `s.*` over a struct column
expands to its fields. That one needs the nested types the type set does not
carry yet.
"""

from .bind import Reference, Scopes
from .catalog import NOT_FOUND, fold


struct Target(Copyable, ImplicitlyCopyable, Movable):
    """A column a modifier names, qualified or not."""

    var table: String
    """The binding's name, empty when the modifier was written bare."""

    var column: String
    """The column's name as written."""

    def __init__(out self, table: StringSlice, column: StringSlice):
        """A name a modifier wrote.

        Args:
            table: The qualifier, or empty for a bare name.
            column: The column's name.
        """
        self.table = String(table)
        self.column = String(column)

    def written(self) -> String:
        """How the query wrote it, for an error message.

        Returns:
            `a` for a bare name and `t.a` for a qualified one.
        """
        if self.table.byte_length() == 0:
            return self.column
        return String(self.table, ".", self.column)

    def matches(self, table: StringSlice, column: StringSlice) -> Bool:
        """Whether this modifier names that column of that binding.

        Args:
            table: The binding's name.
            column: The column's name.

        Returns:
            True if it does, folding both sides.
        """
        if fold(self.column) != fold(column):
            return False
        if self.table.byte_length() == 0:
            return True
        return fold(self.table) == fold(table)


@fieldwise_init
struct Replacement(Copyable, ImplicitlyCopyable, Movable):
    """One `REPLACE (expression AS name)` entry."""

    var target: Target
    """The column it stands in for, never qualified."""

    var node: UInt32
    """The expression, which nothing here looks at."""


struct Renaming(Copyable, ImplicitlyCopyable, Movable):
    """One `RENAME (name AS name)` entry."""

    var target: Target
    """The column being renamed, qualified or not."""

    var name: String
    """What it is called afterwards."""

    def __init__(out self, target: Target, name: StringSlice):
        """One renaming.

        Args:
            target: The column being renamed.
            name: What it is called afterwards.
        """
        self.target = target
        self.name = String(name)


@fieldwise_init
struct Selected(Copyable, Movable):
    """One column a star turned into."""

    var reference: Reference
    """Where the column comes from."""

    var name: String
    """What the output calls it, after any `RENAME`."""

    var node: UInt32
    """A `REPLACE` expression standing in for the column, or `NOT_REPLACED`."""

    def replaced(self) -> Bool:
        """Whether a `REPLACE` stood in for this column.

        Returns:
            True if it did.
        """
        return self.node != NOT_REPLACED


comptime NOT_REPLACED: UInt32 = 0xFFFFFFFF
"""What `Selected.node` holds when the column is itself."""


def expand(
    scopes: Scopes,
    level: Int,
    qualifier: StringSlice,
    excluded: List[Target],
    replaced: List[Replacement],
    renamed: List[Renaming],
) raises -> List[Selected]:
    """What a star at one query level stands for.

    Args:
        scopes: The chain of levels.
        level: The level the star was written at.
        qualifier: The name before the dot, empty for a bare star.
        excluded: The `EXCLUDE` list.
        replaced: The `REPLACE` list.
        renamed: The `RENAME` list.

    Returns:
        One entry per column, in `FROM` order and then schema order. The list
        can come back empty, which is an error at the select list rather than
        here, since a star is not the only thing a select list can hold.

    Raises:
        If the qualifier names no binding, or a modifier names no column, or
        the same column is named twice.
    """
    check(excluded, replaced, renamed)
    var only = NOT_FOUND
    if qualifier.byte_length() != 0:
        # Level local on purpose. A qualified star cannot reach an outer
        # table, which is measured rather than assumed.
        only = scopes.levels[level].find(fold(qualifier))
        if only == NOT_FOUND:
            raise Error(scopes.no_such_table(level, qualifier))

    var out = List[Selected]()
    var used_exclude = List[Bool](length=len(excluded), fill=False)
    var used_replace = List[Bool](length=len(replaced), fill=False)
    for reference in scopes.visible_columns(level):
        if only != NOT_FOUND and reference.binding != only:
            continue
        ref binding = scopes.levels[level].bindings[reference.binding]
        var table = binding.name
        var column = binding.columns[reference.column].name

        var dropped = False
        for at in range(len(excluded)):
            if excluded[at].matches(table, column):
                used_exclude[at] = True
                dropped = True
        if dropped:
            continue

        var node = NOT_REPLACED
        var taken = False
        for at in range(len(replaced)):
            if not replaced[at].target.matches(table, column):
                continue
            # A REPLACE that matches twice replaces the first column and drops
            # the second one. DuckDB's, and it loses a column without saying
            # so.
            if used_replace[at]:
                taken = True
                break
            used_replace[at] = True
            node = replaced[at].node
        if taken:
            continue

        var name = String(column)
        for entry in renamed:
            if entry.target.matches(table, column):
                name = entry.name
        out.append(Selected(reference, name^, node))

    for at in range(len(excluded)):
        if not used_exclude[at]:
            raise Error(not_in_from("EXCLUDE", excluded[at]))
    for at in range(len(replaced)):
        if not used_replace[at]:
            raise Error(not_in_from("REPLACE", replaced[at].target))
    # A RENAME naming nothing is not an error, which is the one modifier that
    # lets a typo through. DuckDB's again.
    return out^


def check(
    excluded: List[Target],
    replaced: List[Replacement],
    renamed: List[Renaming],
) raises:
    """Refuses a modifier list that names the same column twice.

    DuckDB catches these while parsing rather than while binding, so they are
    reported with its parser's wording even though they are found here. It
    takes the three lists in the order they are written and each entry in the
    order it appears, and the first entry that collides with something
    already seen is the one reported, so `EXCLUDE (a) REPLACE (1 AS a, 2 AS
    a)` is a collision between two lists rather than a duplicate inside one.

    Args:
        excluded: The `EXCLUDE` list.
        replaced: The `REPLACE` list.
        renamed: The `RENAME` list.

    Raises:
        If one list holds a name twice, or two lists hold the same name.
    """
    var seen_exclude = List[Target]()
    for entry in excluded:
        _no_duplicate(seen_exclude, entry, "EXCLUDE")
        seen_exclude.append(entry)

    var seen_replace = List[Target]()
    for entry in replaced:
        _no_duplicate(seen_replace, entry.target, "REPLACE")
        _no_overlap(seen_exclude, entry.target, "EXCLUDE", "REPLACE")
        seen_replace.append(entry.target)

    var seen_rename = List[Target]()
    for entry in renamed:
        _no_overlap(seen_exclude, entry.target, "EXCLUDE", "RENAME")
        _no_overlap(seen_replace, entry.target, "REPLACE", "RENAME")
        # Reported against the EXCLUDE list rather than the RENAME one, which
        # is DuckDB naming a list the query may not have written at all.
        _no_duplicate(seen_rename, entry.target, "EXCLUDE")
        seen_rename.append(entry.target)


def not_in_from(list: StringSlice, target: Target) -> String:
    """DuckDB's error for a modifier naming a column no binding offers.

    Args:
        list: `EXCLUDE` or `REPLACE`.
        target: The name that matched nothing.

    Returns:
        The message.
    """
    return String(
        'Binder Error: Column "',
        target.written(),
        '" in ',
        list,
        " list not found in FROM clause",
    )


def empty_select_list() -> String:
    """DuckDB's error for a select list with nothing left in it.

    Returns:
        The message.
    """
    return String(
        "Binder Error: SELECT list is empty after resolving * expressions!"
    )


def _no_duplicate(seen: List[Target], target: Target, list: StringSlice) raises:
    """Refuses a name the same list already holds.

    Args:
        seen: The entries before this one.
        target: The entry.
        list: What to call the list in the message.

    Raises:
        If the name is already there.
    """
    for before in seen:
        if not _same(before, target):
            continue
        raise Error(
            String(
                'Parser Error: Duplicate entry "',
                target.written(),
                '" in ',
                list,
                " list",
            )
        )


def _no_overlap(
    seen: List[Target], target: Target, first: StringSlice, second: StringSlice
) raises:
    """Refuses a name an earlier modifier already holds.

    Args:
        seen: The earlier modifier's entries.
        target: The entry.
        first: What to call the earlier modifier.
        second: What to call this one.

    Raises:
        If the name is in both.
    """
    for before in seen:
        if not _same(before, target):
            continue
        raise Error(
            String(
                'Parser Error: Column "',
                target.written(),
                '" cannot occur in both ',
                first,
                " and ",
                second,
                " list",
            )
        )


def _same(a: Target, b: Target) -> Bool:
    """Whether two modifier names could name the same column.

    A bare name counts against a qualified one, so `EXCLUDE (t.a, a)` is a
    duplicate while `EXCLUDE (t.a, u.a)` is two different columns and is
    fine.

    Args:
        a: One name.
        b: The other.

    Returns:
        True if the columns fold the same and the qualifiers do not rule it
        out.
    """
    if fold(a.column) != fold(b.column):
        return False
    if a.table.byte_length() == 0 or b.table.byte_length() == 0:
        return True
    return fold(a.table) == fold(b.table)
