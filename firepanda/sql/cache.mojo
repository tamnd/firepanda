"""The prepared statement cache: a query seen before goes straight to its plan.

Parsing, lowering, binding and optimizing a query is the whole of the front
end's cost, and none of it depends on the rows. Parsing is a function of the
text and the grammar, and binding is a function of the catalog's names and
their columns, so a query run twice against the same names runs the same plan
twice. This keeps the plan the first run made and hands it to the second, which
then does nothing but find its frames and run.

### The key

The exact text, then what the catalog says about it, then the values an
`EXECUTE` passed if one is running. What the catalog says is one of two things
and the caller picks which.

A session keeps one catalog for its whole life and that catalog counts its own
changes, so the generation is enough: it moves on every register, drop, insert,
`SET` and `RESET`, and a plan made at one generation is not looked at in the
next. That throws away every plan whenever anything changes rather than
working out which plans the change reached, because a shell registers rarely
and queries constantly, and a precise rule would be a correctness risk bought
for no measurable gain.

`firepanda.sql` builds a fresh catalog for every call, so its generation says
nothing: two calls registering the same number of frames count the same. Its
key carries the catalog's shape instead, every name with its columns and their
types, which is exactly what binding read, and so a loop calling it with the
same frames, or with different frames of the same columns, plans once.

A value passed to `firepanda.sql` is not in its text and not in the key. It is a
row in a frame of its own, read by a scalar subquery whose text is fixed, so the
plan is the same whatever the value is, and a loop over a parameterized query is
the case this pays most for.

### What it does not keep

Only a statement whose first word says it is a query is looked up or kept. A
`CREATE`, an `INSERT` or a `SET` changes the catalog, and the generation it
moves is what the next lookup reads, so caching one would be caching a change.
Nothing is looked up inside a transaction either, since a statement there can
abort it and the rules for that live in `execute`.

A cache that only grows is a leak in a long session running generated text, so
past `CAPACITY` plans it is emptied and starts again, which costs one plan per
query that comes back.
"""

from std.collections import Dict
from std.memory import ArcPointer

from firepanda.frame import DataFrame

from .catalog import Catalog
from .ddl import execute
from .run import Dialect, Prepared


comptime CAPACITY = 512
"""How many plans a cache holds before it is emptied."""


def _first_word(sql: StringSlice) -> String:
    """The statement's first word, upper cased, past space, comments and
    opening parentheses.

    Args:
        sql: The statement.

    Returns:
        The word, or `(` if the text opens a parenthesis, or empty.
    """
    var bytes = sql.as_bytes()
    var at = 0
    var end = len(bytes)
    while at < end:
        var byte = Int(bytes[at])
        if byte == ord(" ") or byte == ord("\t") or byte == ord(
            "\n"
        ) or byte == ord("\r"):
            at += 1
        elif byte == ord("-") and at + 1 < end and Int(bytes[at + 1]) == ord("-"):
            while at < end and Int(bytes[at]) != ord("\n"):
                at += 1
        elif byte == ord("/") and at + 1 < end and Int(bytes[at + 1]) == ord("*"):
            at += 2
            while at + 1 < end and not (
                Int(bytes[at]) == ord("*") and Int(bytes[at + 1]) == ord("/")
            ):
                at += 1
            at += 2
        else:
            break
    if at >= end:
        return String()
    if Int(bytes[at]) == ord("("):
        return String("(")
    var start = at
    while at < end and (
        (Int(bytes[at]) >= ord("a") and Int(bytes[at]) <= ord("z"))
        or (Int(bytes[at]) >= ord("A") and Int(bytes[at]) <= ord("Z"))
    ):
        at += 1
    return String(sql[byte=start:at]).upper()


def is_query(sql: StringSlice) -> Bool:
    """Whether a statement's first word makes it a query and nothing else.

    Args:
        sql: The statement.

    Returns:
        True for `SELECT`, `WITH`, `VALUES`, `FROM`, `TABLE`, `PIVOT`,
        `UNPIVOT` and a parenthesis.
    """
    var word = _first_word(sql)
    return (
        word == "SELECT"
        or word == "WITH"
        or word == "VALUES"
        or word == "FROM"
        or word == "TABLE"
        or word == "PIVOT"
        or word == "UNPIVOT"
        or word == "("
    )


struct PlanCache(Movable, Sized):
    """Plans by the text and catalog they were made for."""

    var _plans: Dict[String, ArcPointer[Prepared]]
    """Every plan kept, under its key."""

    var hits: Int
    """How many lookups found a plan."""

    var misses: Int
    """How many lookups of a query found none and made one."""

    def __init__(out self):
        """An empty cache."""
        self._plans = Dict[String, ArcPointer[Prepared]]()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> Int:
        """How many plans are kept.

        Returns:
            The count.
        """
        return len(self._plans)

    def clear(mut self):
        """Forgets every plan."""
        self._plans = Dict[String, ArcPointer[Prepared]]()

    @staticmethod
    def key(sql: StringSlice, scope: StringSlice, catalog: Catalog) -> String:
        """The text a plan is kept under.

        Args:
            sql: The statement.
            scope: What the catalog says about it: its generation for a
                session, its shape for a catalog built for one call.
            catalog: The catalog, for the values an `EXECUTE` passed.

        Returns:
            The key.
        """
        var out = String(sql.byte_length(), ":", sql, "\x1c", scope)
        if catalog.arguments.bound:
            for i in range(len(catalog.arguments.names)):
                out.write(
                    "\x1c",
                    catalog.arguments.names[i],
                    "=",
                    catalog.arguments.texts[i].byte_length(),
                    ":",
                    catalog.arguments.texts[i],
                )
        return out^

    def answer(
        mut self,
        dialect: Dialect,
        sql: StringSlice,
        mut catalog: Catalog,
        scope: StringSlice,
    ) raises -> DataFrame:
        """Runs one statement, from a kept plan if it is a query seen before.

        Args:
            dialect: The grammar, the transform and the function catalog.
            sql: The statement.
            catalog: The session's names.
            scope: As `key` takes it.

        Returns:
            What the statement answers, as `execute` answers it.

        Raises:
            As `execute` does.
        """
        if catalog.in_transaction() or not is_query(sql):
            return execute(dialect, sql, catalog)
        var key = Self.key(sql, scope, catalog)
        var found = self._plans.get(key)
        if found:
            self.hits += 1
            return found.value()[].run(catalog)
        var made: Prepared
        try:
            made = dialect.plan(sql, catalog)
        except:
            # What failed is the query, and `execute` says so in its own words.
            return execute(dialect, sql, catalog)
        self.misses += 1
        if len(self._plans) >= CAPACITY:
            self.clear()
        var kept = ArcPointer(made^)
        self._plans[key] = kept
        return kept[].run(catalog)
