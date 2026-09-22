"""A key to ordinal map that outlives the chunk it was given.

`factorize` answers one question about one column: which of these rows share a
key. A streaming operator asks a different one. It has a chunk in hand and a
running table beside it, and what it needs is which slot of that table each row
belongs in, where the slot a key was given on the first chunk is the slot it
keeps for the whole query. That is the same map `factorize` builds and throws
away, kept instead.

Keeping it is what removes the merge. An operator that factorizes each chunk on
its own gets ordinals that mean something only inside that chunk, so it has to
work out how the chunk's groups line up with the running table's, and that is a
pass over the running table per chunk. Keeping the map makes both sides agree on
the numbering before either is read, and then a chunk costs a pass over the
chunk.

## The two routes and why both are here

`factorize` picks between a direct table and a hash table by scanning the column,
and the reason it bothers is that the two are far apart. A direct table is an
array indexed by the key minus the smallest key, so a lookup is a subtraction and
a load. A hash table is a multiply, a mask, a load and a comparison, and it takes
a cache miss on a table that does not fit in cache. On a thousand groups the
first is around two and a half nanoseconds a row and the second is around six and
a half, so a map that only knew how to hash would be two and a half times slower
than the per chunk factorize it replaced on every query with few groups, which is
most of them.

So both routes are here. The first chunk decides, the same way `factorize`
decides, and the decision holds for the query.

## Why the window does not move

A direct table needs a range of key values fixed in advance, and the first chunk
is what fixes it. A later chunk can hold a key outside that range, and when it
does this gives up the direct table and rebuilds a hash table from the keys it
has already collected, which is one pass over the groups and happens at most once
in a query. The rest of the chunk that found the key goes through the new table,
so the handover costs nothing beyond the rebuild and no row is looked at twice.

Widening the window instead would be possible and is deliberately not done. The
shape that would want it is data arriving in key order, where each chunk holds a
range just past the last one, and on that shape the window would widen on every
chunk until it hit the ceiling and then give up anyway. Rebuilding once is the
same answer for less machinery.

The alternative to catching the key in the loop is scanning each chunk for its
range before looking anything up, which is what the first chunk does because it
has to decide something. It was measured on every chunk and it is not worth it:
the scan is a pass over the key column and the test it saves is a compare against
a constant on a value that is already in a register, and that trade came out at
around a nanosecond a row against a lookup that costs two.

## What the ceiling is

`DIRECT_LIMIT` is the widest span `factorize` will build a table over, and it is
set against one pass over one column. A map that lasts is paid off by every chunk
of the query rather than by one, so it can afford a wider one, and `LASTING_SPAN`
is that number. At its widest the table is a megabyte, which is less than the
hash table for the same group count, so the wider ceiling costs no memory
anywhere it is taken.

## Text

A text key gets a map of its own, `LastingText`, and the reason it cannot use
the one above is what a stored key is. The fixed width routes store the hash and
nothing else, because the hash is a bijection on the key bits and two different
keys cannot land on one, so a slot's hash matching is the answer. Two names
longer than eight bytes can land on one hash, so a match there is a candidate
and the bytes have to be compared, and comparing them means the bytes are still
somewhere to be read.

The per chunk factorize keeps a view per group into the chunk it grouped, which
is enough for a map that dies with its chunk and useless for one that does not:
a long view is an offset into a payload buffer, and the chunk that owned the
buffer is gone by the time the next chunk asks. So this copies the bytes of a
key the first time it sees one, into a builder it keeps, and compares against
that. The copy is one per group rather than one per row, and the per chunk route
was going to copy the same bytes anyway when it gathered its representative
rows, so the copy is not new work. The builder is also the key column the
operator wants at the end, so the keys are stored once and not twice.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import ALL
from firepanda.kernel.concat import concat_any
from firepanda.kernel.select import take_any

from .factorize import CHUNK_ROWS, DIRECT_LIMIT, direct_plan
from .function import DEFAULT_SEED, hash_chunk, hash_strings_chunk
from .table import HashTable


comptime LASTING_SPAN = 1 << 18
"""The widest key range a lasting direct table will cover.

Four bytes a slot, so a megabyte at the top of it, against a hash table for the
same group count that is four megabytes once the load factor and the two words a
slot are counted. The direct table is the smaller of the two everywhere it is
taken, which is why this sits four doublings above the per column `DIRECT_LIMIT`
rather than at it.
"""


comptime LASTING_TEXT_SLOTS = 1 << 10
"""How many slots a text map starts with, before it doubles."""


struct LastingText(Movable):
    """The text keys a streaming operator has seen, and the ordinal each got.

    An open addressed table of hashes beside a builder holding one copy of each
    distinct key, in ordinal order. A slot whose hash matches sends the row to
    the builder to have its bytes compared, and a mismatch keeps probing the way
    an occupied slot with a different hash does, so two keys that collide get
    two ordinals rather than one group between them.

    The table doubles from a small start rather than being sized from a guess at
    the group count. `HashTable` guesses because it is built once per column and
    a rehash it could have avoided is a real share of that; this one is built
    once per query, the rehashes on the way to a size total about one more pass
    over the groups than arriving there directly would have cost, and the guess
    would have to be made from the first chunk, which is the part of the input
    that says least about the whole of it.
    """

    var slots: Buffer
    """Two words a slot: the key's hash, and its ordinal plus one so that a
    fresh buffer of zeros reads as all empty."""

    var mask: UInt64
    """One less than the slot count, which is a power of two."""

    var capacity: Int
    """The slot count."""

    var store: StringBuilder
    """One copy of each distinct key, in ordinal order. The keys the operator
    asks for at the end and the bytes the comparison reads are the same bytes."""

    var groups: Int
    """Ordinals handed out so far."""

    def __init__(out self):
        """Constructs an empty map with no table allocated yet."""
        self.slots = Buffer(0)
        self.mask = 0
        self.capacity = 0
        self.store = StringBuilder()
        self.groups = 0

    def __len__(self) -> Int:
        """Returns the number of ordinals handed out.

        Returns:
            The group count.
        """
        return self.groups

    def ordinals(
        mut self, col: StringArray, rows: Int, mut codes: Array[DType.uint32]
    ):
        """Gives every row of one chunk the ordinal its key holds in the map.

        Args:
            col: The chunk's key column. Must have no nulls, for the reason
                `LastingKeys.ordinals` gives.
            rows: The chunk's height.
            codes: Filled with one ordinal per row of the chunk.
        """
        if self.capacity == 0:
            self._resize(LASTING_TEXT_SLOTS)
        var hashes = Buffer(CHUNK_ROWS * 8)
        var at = 0
        while at < rows:
            var count = min(CHUNK_ROWS, rows - at)
            hash_strings_chunk(col, at, count, DEFAULT_SEED, hashes)
            self._insert(col, at, count, hashes, codes)
            at += count

    def take_keys(mut self) -> StringArray:
        """Gives up the key store as a column.

        Returns:
            One row per group, in ordinal order.
        """
        var held = self.store^
        self.store = StringBuilder()
        self.groups = 0
        self.slots = Buffer(0)
        self.mask = 0
        self.capacity = 0
        return held^.finish()

    def _insert(
        mut self,
        col: StringArray,
        base: Int,
        count: Int,
        hashes: Buffer,
        mut codes: Array[DType.uint32],
    ):
        """Looks one hashed run of the chunk up, inserting what it has not seen.
        """
        var hash = hashes.bitcast[DType.uint64]()
        var out = codes.unsafe_mut_ptr()
        var slots = self.slots.mut_bitcast[DType.uint64]()
        var mask = self.mask

        for j in range(count):
            var i = base + j
            if (self.groups + 1) * 2 > self.capacity:
                self._resize(self.capacity * 2)
                slots = self.slots.mut_bitcast[DType.uint64]()
                mask = self.mask

            var wanted = hash.unsafe_offset(j).unsafe_load()
            var at = wanted & mask
            while True:
                var slot = Int(at) * 2
                var ordinal = slots.unsafe_offset(slot + 1).unsafe_load()
                if ordinal == 0:
                    slots.unsafe_offset(slot).unsafe_store(wanted)
                    slots.unsafe_offset(slot + 1).unsafe_store(
                        UInt64(self.groups + 1)
                    )
                    out.unsafe_offset(i).unsafe_write(UInt32(self.groups))
                    self.store.append(col.unsafe_bytes(i))
                    self.groups += 1
                    break
                if slots.unsafe_offset(slot).unsafe_load() == wanted:
                    if self.store.element_equals_foreign(
                        Int(ordinal) - 1, col.view(i), col
                    ):
                        out.unsafe_offset(i).unsafe_write(
                            UInt32(Int(ordinal) - 1)
                        )
                        break
                at = (at + 1) & mask

    def _resize(mut self, capacity: Int):
        """Moves every live slot into a fresh table of a given size.

        Reinsertion needs no hashing and no comparison. The stored hash is what
        the slot is found by, and two keys already in the table are already
        known to be different keys, so a collision in the new table is settled
        by probing on alone.

        Args:
            capacity: The new slot count. A power of two.
        """
        var bigger = Buffer(capacity * 2 * 8)
        var into = bigger.mut_bitcast[DType.uint64]()
        var mask = UInt64(capacity - 1)
        if self.capacity > 0:
            var from_ = self.slots.bitcast[DType.uint64]()
            for s in range(self.capacity):
                var ordinal = from_.unsafe_offset(s * 2 + 1).unsafe_load()
                if ordinal == 0:
                    continue
                var wanted = from_.unsafe_offset(s * 2).unsafe_load()
                var at = wanted & mask
                while into.unsafe_offset(Int(at) * 2 + 1).unsafe_load() != 0:
                    at = (at + 1) & mask
                into.unsafe_offset(Int(at) * 2).unsafe_store(wanted)
                into.unsafe_offset(Int(at) * 2 + 1).unsafe_store(ordinal)
        self.slots = bigger^
        self.mask = mask
        self.capacity = capacity


struct LastingKeys(Movable):
    """The keys a streaming operator has seen, and the ordinal each was given.

    Holds one of two maps and the keys themselves. Which map it holds is decided
    by the first chunk and can change once, from the direct table to the hash
    table, if a later chunk holds a key the direct table's window does not cover.
    """

    var table: HashTable
    """The hash route's map. Empty while the direct route is live."""

    var slots: Array[DType.uint32]
    """The direct route's table, one slot per value in the window, holding the
    ordinal plus one so that a fresh array of zeros reads as all unseen."""

    var span: Int
    """How many values the direct window covers, and zero when the direct route
    is not the one in use."""

    var base: Int
    """The key value that indexes slot zero of `slots`."""

    var groups: Int
    """Ordinals handed out so far."""

    var seen: Int
    """Rows given so far. The hash table's sizing schedule counts in these, so it
    has to carry across chunks the way the table does."""

    var opened: Bool
    """Whether the first chunk has arrived and picked a route."""

    var keys: List[AnyArray]
    """The key values, one row per group, in ordinal order, in as many pieces as
    there were chunks that introduced a group. Kept in pieces because a chunk
    knows its own new keys and joining them up is work for the end."""

    var text: LastingText
    """The text route's map. Untouched unless the key column is text, and it
    holds the keys itself when it is, so `keys` stays empty."""

    var textual: Bool
    """Whether the key column is text, and so whether `text` is the map."""

    def __init__(out self):
        """Constructs an empty map that has not yet picked a route."""
        self.table = HashTable()
        self.slots = Array[DType.uint32](1)
        self.span = 0
        self.base = 0
        self.groups = 0
        self.seen = 0
        self.opened = False
        self.keys = List[AnyArray]()
        self.text = LastingText()
        self.textual = False

    def __len__(self) -> Int:
        """Returns the number of ordinals handed out.

        Returns:
            The group count.
        """
        return self.groups

    def ordinals(
        mut self, key: AnyArray, rows: Int, mut codes: Array[DType.uint32]
    ) raises:
        """Gives every row of one chunk the ordinal its key holds in the map.

        Args:
            key: The chunk's key column. Must have no nulls, because an ordinal
                for the null group is the caller's decision about where in the
                output it belongs rather than this map's.
            rows: The chunk's height.
            codes: Filled with one ordinal per row of the chunk.

        Raises:
            If the key dtype has no physical layout.
        """
        if key.is_string():
            # Text has one route and the first chunk decides nothing, so there
            # is no plan to make and no window to fix. The map holds the keys
            # itself, which is why nothing is appended to the store here.
            self.opened = True
            self.textual = True
            self.text.ordinals(key.strings(), rows, codes)
            self.groups = self.text.groups
            return

        var firsts = List[Int]()
        comptime for candidate in ALL:
            if key.dtype() == candidate:
                self._chunk[candidate](key, rows, codes, firsts)
                # The rows in `firsts` are the ones that introduced a group, in
                # ordinal order, so gathering the key by them appends exactly
                # the new groups' keys to the end of the store and nothing else
                # has to be worked out.
                if len(firsts) > 0:
                    self.keys.append(take_any(key, firsts))
                return
        raise Error(
            "lasting: key dtype "
            + String(key.dtype())
            + " has no physical layout"
        )

    def take_keys(mut self) raises -> AnyArray:
        """Stacks the key pieces into one column and gives up the store.

        Returns:
            One row per group, in ordinal order.

        Raises:
            If the pieces cannot be stacked.
        """
        if self.textual:
            return AnyArray(self.text.take_keys())
        var out = concat_any(self.keys)
        self.keys = List[AnyArray]()
        return out^

    def _chunk[
        dt: DType
    ](
        mut self,
        key: AnyArray,
        rows: Int,
        mut codes: Array[DType.uint32],
        mut firsts: List[Int],
    ) raises:
        """Routes one chunk of a known dtype, picking the route if it is first.
        """
        ref col = key.as_typed_view[dt]()

        comptime if dt.is_integral():
            if not self.opened:
                self.opened = True
                var ceiling = rows
                if ceiling < DIRECT_LIMIT:
                    ceiling = DIRECT_LIMIT
                if ceiling > LASTING_SPAN:
                    ceiling = LASTING_SPAN
                var plan = direct_plan[dt](col, ceiling)
                if plan.span > 0:
                    self.span = plan.span
                    self.base = Int(plan.base)
                    self.slots = Array[DType.uint32](plan.span)
            if self.span > 0:
                var stopped = self._direct[dt](col, rows, codes, firsts)
                if stopped == rows:
                    return
                # A key landed outside the window. What the direct table knows
                # is in the key store once this chunk's share of it is added, so
                # the hash table is built from that and the ordinals come out
                # unchanged, and the rest of the chunk goes through the table.
                if len(firsts) > 0:
                    self.keys.append(take_any(key, firsts))
                    firsts = List[Int]()
                self._rehash()
                self._hashed[dt](col, stopped, rows, codes, firsts)
                return
        self.opened = True
        self._hashed[dt](col, 0, rows, codes, firsts)

    def _direct[
        dt: DType
    ](
        mut self,
        col: Array[dt],
        rows: Int,
        mut codes: Array[DType.uint32],
        mut firsts: List[Int],
    ) -> Int:
        """Looks a chunk up in the direct table until a key leaves the window.

        Returns the row it stopped on, which is the height when it did not stop.
        """
        var base = self.base
        var span = self.span
        var into = self.slots.unsafe_mut_ptr()
        var src = col.unsafe_ptr()
        var out = codes.unsafe_mut_ptr()
        var n = self.groups
        var i = 0
        while i < rows:
            var at = Int(src.unsafe_offset(i).unsafe_load()) - base
            if at < 0 or at >= span:
                break
            var found = into.unsafe_offset(at).unsafe_load()
            if found == 0:
                into.unsafe_offset(at).unsafe_store(UInt32(n + 1))
                out.unsafe_offset(i).unsafe_store(UInt32(n))
                firsts.append(i)
                n += 1
            else:
                out.unsafe_offset(i).unsafe_store(found - 1)
            i += 1
        self.groups = n
        self.seen += i
        return i

    def _hashed[
        dt: DType
    ](
        mut self,
        col: Array[dt],
        start: Int,
        rows: Int,
        mut codes: Array[DType.uint32],
        mut firsts: List[Int],
    ):
        """Inserts a chunk into the hash table.

        This is the same call the join's build makes, for the same reason: hash a
        thousand rows into a scratch buffer while they are in cache, insert them,
        move on.

        The chunk's height goes in as the total the table should expect, which is
        a lie on the second chunk and every one after it, and it is the right
        lie. Both of the sizing checkpoints fall inside a chunk of the size the
        engine uses, so the table is sized once from the first chunk and doubles
        from there, and a stream has no honest total to give it anyway.
        """
        var hashes = Buffer(CHUNK_ROWS * 8)
        var at = start
        while at < rows:
            var count = min(CHUNK_ROWS, rows - at)
            hash_chunk(col, at, count, DEFAULT_SEED, hashes)
            self.table.build(
                hashes,
                col.data.validity,
                False,
                at,
                self.seen + at - start,
                count,
                rows - start,
                0,
                codes,
                firsts,
            )
            at += count
        self.seen += rows - start
        self.groups = len(self.table)

    def _rehash(mut self) raises:
        """Rebuilds the hash table from the keys the direct table collected.

        The keys are distinct and they are in ordinal order, so inserting them in
        that order into an empty table gives each one back the ordinal it already
        had, and nothing downstream has to be told the route changed.
        """
        self.span = 0
        self.slots = Array[DType.uint32](1)
        self.table = HashTable()
        self.seen = 0
        var held = self.groups
        self.groups = 0
        if held == 0:
            return

        var store = List[AnyArray]()
        store.append(concat_any(self.keys))
        var codes = Array[DType.uint32](held)
        var firsts = List[Int]()
        comptime for candidate in ALL:
            if store[0].dtype() == candidate:
                self._hashed[candidate](
                    store[0].as_typed_view[candidate](),
                    0,
                    held,
                    codes,
                    firsts,
                )
        self.keys = store^
