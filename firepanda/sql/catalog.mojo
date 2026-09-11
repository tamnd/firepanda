"""The names a query is allowed to say.

firepanda has no storage, so there is no database to open and nothing here
survives the process. A catalog is a session scoped namespace: a name, and
either a frame somebody registered or a view somebody defined. No schemas, no
`ATTACH`, no persistence. See docs/specs/sql/05-ast-and-binder.md section 7.

Three things about it are worth stating, because each one is a decision rather
than an accident.

**It is one namespace and not two.** A frame called `t` and a view called `t`
cannot both exist, because a query that says `FROM t` has to mean one of them
and picking by kind would be a rule nobody could guess. Registering over a name
that is already taken replaces whatever was there, which is what
`CREATE OR REPLACE` means and what a REPL user expects when they register the
same name twice.

**Lookup folds and storage does not.** DuckDB is case insensitive throughout and
case preserving in what it prints back, and quoting an identifier does not
change that: `CREATE TABLE "MyTable"(i int)` is reachable as `mytable`, as
`"MYTABLE"` and as `MyTable`. So the catalog compares on a folded key and keeps
the spelling the caller used, which is the spelling that comes back in
`Did you mean`. This is the one place where a quoted name behaves like a bare
one, and getting it the other way round would refuse queries DuckDB accepts.

**It scans rather than hashes.** The catalog a REPL actually has holds a handful
of names, and a linear scan over a short list of short strings beats a hash of
the probe followed by a probe of the table. Section 14 of the specification says
so and this is that. If somebody registers ten thousand frames the scan is the
wrong shape, and the fix at that point is an index beside the list rather than a
different design.

A frame is held by move and handed back by reference, so registering costs one
copy at the caller's discretion and reading costs none. A catalog full of frames
is the working set of the session and copying it per query would be the most
expensive thing in the front end.
"""

from firepanda.frame import DataFrame


comptime NOT_FOUND: Int = -1
"""What `find` returns for a name nobody registered."""


comptime KIND_FRAME: UInt8 = 0
"""A registered frame."""


comptime KIND_VIEW: UInt8 = 1
"""A view, which is a query text under a name."""


struct View(Movable):
    """A named query, unexpanded.

    The text rather than a plan, because there is no plan layer to hold yet. A
    reference to a view parses and binds its text where the reference is, which
    is what inlining means and is DuckDB's default anyway. When the plan layer
    lands this becomes the bound plan and the reference stops reparsing, which
    is a change to this field and to nothing that reads it.
    """

    var sql: String
    """The query the view stands for, as the definer wrote it."""

    var columns: List[String]
    """The names the definer gave the view's columns, if any.

    `CREATE VIEW v (a, b) AS SELECT ...` renames the query's output, and an
    empty list means the query's own names are the view's names.
    """

    def __init__(out self, sql: StringSlice, var columns: List[String]):
        """Defines a view over a query.

        Args:
            sql: The query text.
            columns: The column names the definer gave, or an empty list.
        """
        self.sql = String(sql)
        self.columns = columns^


struct Catalog(Movable, Sized):
    """A session scoped namespace of frames and views."""

    var _keys: List[String]
    """The folded name of each entry, which is what a lookup compares."""

    var _names: List[String]
    """The spelling the caller registered, which is what an error prints."""

    var _kinds: List[UInt8]
    """`KIND_FRAME` or `KIND_VIEW`, one per entry."""

    var _slots: List[Int]
    """Where the entry's payload is, in `_frames` or in `_views`."""

    var _frames: List[DataFrame]
    """The registered frames, in registration order."""

    var _views: List[View]
    """The defined views, in definition order."""

    var _generation: UInt64
    """How many times the set of names has changed.

    A prepared statement caches a bound plan and a bound plan names entries in
    this catalog, so the cache key has to carry something that moves when the
    catalog does. A counter is enough: it is not trying to say what changed,
    only that something did, and a plan bound against generation 4 is not sound
    against generation 5 whatever the difference was.
    """

    def __init__(out self):
        """Constructs an empty catalog."""
        self._keys = List[String]()
        self._names = List[String]()
        self._kinds = List[UInt8]()
        self._slots = List[Int]()
        self._frames = List[DataFrame]()
        self._views = List[View]()
        self._generation = 0

    def __len__(self) -> Int:
        """How many names are registered.

        Returns:
            The entry count, frames and views together.
        """
        return len(self._keys)

    def generation(self) -> UInt64:
        """How many times the set of names has changed.

        Returns:
            The counter, which starts at zero and moves on every register,
            define and drop that changed anything.
        """
        return self._generation

    def find(self, name: StringSlice) -> Int:
        """Looks a name up, folding it first.

        Args:
            name: The name as the query spelled it, quoted or bare.

        Returns:
            The entry's index, or `NOT_FOUND`.
        """
        var key = fold(name)
        for at in range(len(self._keys)):
            if self._keys[at] == key:
                return at
        return NOT_FOUND

    def contains(self, name: StringSlice) -> Bool:
        """Whether a name resolves.

        Args:
            name: The name as the query spelled it.

        Returns:
            True if something is registered under it.
        """
        return self.find(name) != NOT_FOUND

    def kind_at(self, at: Int) -> UInt8:
        """What kind of thing an entry is.

        Args:
            at: The index `find` returned.

        Returns:
            `KIND_FRAME` or `KIND_VIEW`.
        """
        return self._kinds[at]

    def name_at(ref self, at: Int) -> ref[self._names[at]] String:
        """The spelling an entry was registered under.

        Args:
            at: The index `find` returned.

        Returns:
            The name as the caller wrote it, not folded.
        """
        return self._names[at]

    def frame_at(
        ref self, at: Int
    ) -> ref[self._frames[self._slots[at]]] DataFrame:
        """The frame an entry holds.

        Borrowed rather than returned by value, because a frame is the session's
        working set and a query that copied it to read its schema would be the
        most expensive thing in the front end.

        Args:
            at: The index `find` returned, whose kind is `KIND_FRAME`.

        Returns:
            The frame, borrowed from the catalog.
        """
        return self._frames[self._slots[at]]

    def view_at(ref self, at: Int) -> ref[self._views[self._slots[at]]] View:
        """The view an entry holds.

        Args:
            at: The index `find` returned, whose kind is `KIND_VIEW`.

        Returns:
            The view, borrowed from the catalog.
        """
        return self._views[self._slots[at]]

    def register(mut self, name: StringSlice, var frame: DataFrame) raises:
        """Puts a frame under a name, replacing whatever was there.

        Replacing rather than refusing, because registering the same name twice
        is what a notebook does every time a cell is rerun, and an error there
        would be an error about the notebook rather than about the data.

        Args:
            name: The name queries will say. Folded for lookup, kept as written
                for messages.
            frame: The frame, taken by move.

        Raises:
            Error: If the name is empty, which no query can say.
        """
        self._require_a_name(name)
        var at = self.find(name)
        if at != NOT_FOUND and self._kinds[at] == KIND_FRAME:
            self._frames[self._slots[at]] = frame^
            self._names[at] = String(name)
            self._generation += 1
            return
        if at != NOT_FOUND:
            _ = self.drop(name)
        self._keys.append(fold(name))
        self._names.append(String(name))
        self._kinds.append(KIND_FRAME)
        self._slots.append(len(self._frames))
        self._frames.append(frame^)
        self._generation += 1

    def define(mut self, name: StringSlice, var view: View) raises:
        """Puts a view under a name, replacing whatever was there.

        Args:
            name: The name queries will say.
            view: The query the name stands for.

        Raises:
            Error: If the name is empty.
        """
        self._require_a_name(name)
        var at = self.find(name)
        if at != NOT_FOUND and self._kinds[at] == KIND_VIEW:
            self._views[self._slots[at]] = view^
            self._names[at] = String(name)
            self._generation += 1
            return
        if at != NOT_FOUND:
            _ = self.drop(name)
        self._keys.append(fold(name))
        self._names.append(String(name))
        self._kinds.append(KIND_VIEW)
        self._slots.append(len(self._views))
        self._views.append(view^)
        self._generation += 1

    def drop(mut self, name: StringSlice) -> Bool:
        """Takes a name out of the namespace.

        Args:
            name: The name as the caller spells it.

        Returns:
            True if it was there, so a caller can tell a drop from a no-op
            without asking twice.
        """
        var at = self.find(name)
        if at == NOT_FOUND:
            return False
        self._forget(at)
        _ = self._keys.pop(at)
        _ = self._names.pop(at)
        _ = self._kinds.pop(at)
        _ = self._slots.pop(at)
        self._generation += 1
        return True

    def names(self) -> List[String]:
        """Every registered name, in registration order.

        Returns:
            The spellings, not the folded keys, because this is what a caller
            prints.
        """
        var out = List[String](capacity=len(self._names))
        for name in self._names:
            out.append(name)
        return out^

    def resolve(self, name: StringSlice) raises -> Int:
        """Looks a name up and raises DuckDB's error if it is not there.

        Args:
            name: The name the query said.

        Returns:
            The entry's index.

        Raises:
            Error: A catalog error naming the table, with a suggestion when one
                of the registered names is close enough to be worth offering.
        """
        var at = self.find(name)
        if at != NOT_FOUND:
            return at
        raise Error(self.missing(name))

    def missing(self, name: StringSlice) -> String:
        """The error text for a name that is not registered.

        Byte for byte DuckDB's, because the conformance corpus matches error
        text by substring and a message that says the same thing in different
        words fails those tests for no reason.

        Args:
            name: The name the query said.

        Returns:
            The message, with a `Did you mean` line when something is close.
        """
        var message = String(
            "Catalog Error: Table with name ", name, " does not exist!"
        )
        var near = self.nearest(name)
        if near == NOT_FOUND:
            return message
        return String(message, '\nDid you mean "', self._names[near], '"?')

    def nearest(self, name: StringSlice) -> Int:
        """The registered name closest to one that is not registered.

        DuckDB suggests out of its whole catalog, which for it includes the
        Postgres compatibility tables, so it can answer a name that resembles
        nothing the user registered with a system table they have never heard
        of. We have no such tables, so a bad guess here would be worse than
        none, and the threshold is the point of the function rather than a
        detail of it: an edit distance below half the shorter name is a typo,
        and above it is a different word.

        Args:
            name: The name the query said.

        Returns:
            The index of the closest entry, or `NOT_FOUND` if none is close.
        """
        var probe = fold(name)
        var best = NOT_FOUND
        var best_distance = 0
        for at in range(len(self._keys)):
            var candidate = self._keys[at]
            var shorter = min(probe.byte_length(), candidate.byte_length())
            var limit = shorter // 2 + 1
            var distance = edit_distance(probe, candidate, limit)
            if distance >= limit:
                continue
            if best == NOT_FOUND or distance < best_distance:
                best = at
                best_distance = distance
        return best

    def _forget(mut self, at: Int):
        """Drops the payload an entry points at and closes the gap.

        The payload lists are packed rather than tombstoned, so a slot after the
        removed one moves down by one and every entry pointing past it has to be
        told. That is a scan over the entries per drop, which is the right trade
        at this size: drops are rare and lookups are not, and a tombstone would
        make every lookup walk holes forever.

        Args:
            at: The entry whose payload goes.
        """
        var kind = self._kinds[at]
        var slot = self._slots[at]
        if kind == KIND_FRAME:
            _ = self._frames.pop(slot)
        else:
            _ = self._views.pop(slot)
        for other in range(len(self._slots)):
            if (
                other != at
                and self._kinds[other] == kind
                and self._slots[other] > slot
            ):
                self._slots[other] -= 1

    def _require_a_name(self, name: StringSlice) raises:
        """Refuses the empty name.

        Args:
            name: The name the caller passed.

        Raises:
            Error: If it is empty.
        """
        if name.byte_length() == 0:
            raise Error(
                "Catalog Error: a registered name has to have something in it"
            )


def fold(name: StringSlice) -> String:
    """Folds a name down for comparison.

    ASCII only and down rather than up, which is the tokenizer's rule for a bare
    identifier and, at the catalog, the rule for a quoted one too. Anything
    outside ASCII is left alone, so a name in another script compares byte for
    byte, which is what DuckDB does and is the only answer that does not need a
    case folding table for every script in Unicode.

    Args:
        name: The name as it was written.

    Returns:
        The folded key.
    """
    var bytes = List[UInt8](capacity=name.byte_length() + 1)
    for byte in name.as_bytes():
        if byte >= UInt8(65) and byte <= UInt8(90):
            bytes.append(byte | UInt8(32))
        else:
            bytes.append(byte)
    return String(StringSlice(unsafe_from_utf8=Span(bytes)))


def edit_distance(a: StringSlice, b: StringSlice, limit: Int) -> Int:
    """Levenshtein distance between two names, given up on past a limit.

    Bounded because the caller only cares whether the distance is small, and the
    unbounded answer for two long unrelated names costs the product of their
    lengths to find out something a length difference already said. The limit is
    returned rather than the true distance when the true distance is at least
    the limit, so a caller compares against the limit and never against a
    distance it did not ask for.

    Args:
        a: One name, already folded.
        b: The other name, already folded.
        limit: The distance the caller stops caring at.

    Returns:
        The distance, or `limit` if it is at least that.
    """
    var left = a.as_bytes()
    var right = b.as_bytes()
    var rows = len(left)
    var columns = len(right)
    if rows - columns >= limit or columns - rows >= limit:
        return limit
    var previous = List[Int](capacity=columns + 1)
    for at in range(columns + 1):
        previous.append(at)
    var current = List[Int](capacity=columns + 1)
    for _ in range(columns + 1):
        current.append(0)
    for row in range(1, rows + 1):
        current[0] = row
        var best = current[0]
        for column in range(1, columns + 1):
            var substitution = previous[column - 1]
            if left[row - 1] != right[column - 1]:
                substitution += 1
            var deletion = previous[column] + 1
            var insertion = current[column - 1] + 1
            var cell = min(substitution, min(deletion, insertion))
            current[column] = cell
            if cell < best:
                best = cell
        if best >= limit:
            return limit
        for column in range(columns + 1):
            previous[column] = current[column]
    var distance = previous[columns]
    if distance >= limit:
        return limit
    return distance
