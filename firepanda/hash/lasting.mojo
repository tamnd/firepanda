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

## A tuple of keys

Two or more key columns have no single hash that is exact, and until
`LastingTuple` that sent every such query back to stacking the running table
with each chunk's table and grouping the two, which costs the height of the
running table on every chunk. ClickBench's q18 and q39 group on three and five
keys with hundreds of thousands of groups, and that term was most of what they
cost.

`LastingTuple` makes the tuple exact by writing it out. Every row's tuple is
written as bytes, a presence byte per key and then the value's own bytes, with
a length in front of text, in two passes on the cores: one to size each row and
one to write it where a prefix sum put it. Two tuples are equal exactly when
their bytes are, so the rows go into a `LastingText`, whose hash match is only
a candidate the bytes settle, and each row comes back with its group's ordinal.

Grouping the chunk on its own first and writing only its distinct tuples was
tried, and was slower: the chunk's grouping pass costs about what the writing
saves. The route only pays when there is text among the keys and more than one
chunk, and `Group` in `exec/node` is where that is decided.

A null is a presence byte of zero and nothing else, so a tuple with a null in
it is a tuple like any other here and does not send the query off this route,
which the single key maps cannot say.
"""

from std.memory import unsafe_memcpy
from std.sys.info import simd_width_of
from std.sys.intrinsics import PrefetchOptions, prefetch

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.array.strview import (
    INLINE_CAPACITY,
    VIEW_SIZE,
    StringView,
    make_inline_at,
    make_long_at,
)
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import ALL, dtype_size
from firepanda.dtype.logical import LogicalType
from firepanda.exec.morsel import parallel_morsels
from firepanda.exec.parallel import parallel_for
from firepanda.kernel.concat import concat_any
from firepanda.kernel.select import take_any

from .factorize import CHUNK_ROWS, DIRECT_LIMIT, direct_plan
from .function import DEFAULT_SEED, hash_bytes, key_bits, mix
from .table import PROBE_LOOKAHEAD


comptime LASTING_SPAN = 1 << 18
"""The widest key range a lasting direct table will cover.

Four bytes a slot, so a megabyte at the top of it, against a hash table for the
same group count that is four megabytes once the load factor and the two words a
slot are counted. The direct table is the smaller of the two everywhere it is
taken, which is why this sits four doublings above the per column `DIRECT_LIMIT`
rather than at it.
"""


comptime LASTING_TEXT_PARTS = 32
"""How many tables a text map is split into, by the top bits of the hash.

A part is the unit a worker takes when the map is written, so there have to be
more of them than there are cores for the last one finishing not to be the whole
tail, and few enough that bucketing a chunk's rows by part is a small table of
counts rather than a pass of its own.
"""


comptime LASTING_TEXT_PART_SHIFT = 59
"""How far to shift a hash to get its part. Five bits, for thirty two parts.

The top bits, because the slot inside a part is taken from the bottom ones, and
a part that took its rows from the same bits it indexes by would use one slot in
thirty two."""


comptime LASTING_TEXT_SLOTS = 1 << 6
"""How many slots a part starts with, before it grows."""


comptime LASTING_TEXT_PENDING = UInt64(1) << 63
"""Marks a slot opened by the chunk being taken, whose ordinal is not out yet.

The rest of the word is the row that opened it, so a later row of the same chunk
with the same hash compares its bytes against that row's."""


comptime LASTING_TEXT_MORSEL = 1 << 12
"""Rows a worker takes at a time on the passes that go by row.

Smaller than `MORSEL_ROWS`, and that is the whole point of naming it. A chunk is
one morsel, so a pass that asked for the engine's morsel size would be handed its
own chunk back as a single piece and would run on the thread that called it. Four
thousand rows puts thirty two pieces in front of ten workers on a chunk, which is
enough that the last one finishing is not the whole tail.
"""


comptime LASTING_TEXT_GRAIN = 1 << 13
"""Rows below which a pass that goes by part does not take another worker.

The walk and the settling go by part, and thirty two parts on a chunk that a
filter left a few thousand rows of cost more in handing out than they save. A
task takes a run of parts and there is one task per this many rows, so a small
chunk runs on the thread that called it.
"""


def _part_tasks(rows: Int) -> Int:
    """How many tasks a pass by part splits a chunk of `rows` into.

    Args:
        rows: The rows of the chunk.

    Returns:
        One per `LASTING_TEXT_GRAIN` rows, at least one and at most one a part.
    """
    return max(1, min(LASTING_TEXT_PARTS, rows // LASTING_TEXT_GRAIN))


struct _Parts(Movable):
    """A table of hashes cut into parts, which both lasting maps write through.

    Two words a slot, the parts one after another: the key's hash, and its
    ordinal plus one so that a fresh buffer of zeros reads as all empty. While a
    chunk is being taken the second word can instead be `LASTING_TEXT_PENDING`
    and a row. `LastingText`'s docstring says why it is cut into parts and how a
    chunk goes through it.
    """

    var slots: Buffer
    """The slots."""

    var capacity: Int
    """The slot count of each part, a power of two."""

    var held: List[Int]
    """How many keys each part holds."""

    def __init__(out self):
        """Constructs an empty table with nothing allocated yet."""
        self.slots = Buffer(0)
        self.capacity = 0
        self.held = List[Int]()

    def make_room(mut self, starts: List[Int]) raises:
        """Grows the table to room for every row of a chunk being new.

        Args:
            starts: Where each part's rows begin in the chunk's bucketed rows,
                with the chunk's height after the last.
        """
        if self.capacity == 0:
            self.slots = Buffer(LASTING_TEXT_PARTS * LASTING_TEXT_SLOTS * 2 * 8)
            self.capacity = LASTING_TEXT_SLOTS
            self.held = List[Int](length=LASTING_TEXT_PARTS, fill=0)
        var want = 0
        for p in range(LASTING_TEXT_PARTS):
            want = max(want, (self.held[p] + starts[p + 1] - starts[p]) * 2)
        if want > self.capacity:
            var grown = self.capacity
            while grown < want:
                grown *= 2
            self.slots = _rehash(self.slots, self.capacity, grown)
            self.capacity = grown

    def add(mut self, added: List[Int]):
        """Counts the slots a chunk opened into each part.

        Args:
            added: How many slots each part opened.
        """
        for p in range(LASTING_TEXT_PARTS):
            self.held[p] += added[p]


struct LastingText(Movable):
    """The text keys a streaming operator has seen, and the ordinal each got.

    An open addressed table of hashes cut into thirty two parts by the top five
    bits of the hash, beside the distinct keys themselves, in ordinal order. A
    slot whose hash matches sends the row off to have its bytes compared, and a
    mismatch keeps probing the way an occupied slot with a different hash does,
    so two keys that collide get two ordinals rather than one group between
    them. A probe that runs off the end of its part wraps to the start of the
    same part.

    The parts are so that the map can be written on the cores. A key only ever
    lives in the part its hash picks, so two workers on two parts never touch
    the same slot. What the parts cannot give on their own is the order the
    ordinals come out in, which is the order the keys were first seen in,
    because that is a question about the whole chunk and a part only sees its
    own rows. So a chunk is taken in steps:

    1. Every row is hashed, on the cores.
    2. The rows are bucketed by part, in row order inside each part.
    3. Each part walks its own rows on a core. A row whose key the map already
       held gets its ordinal. A row whose key is new opens a slot and is marked
       the first of its key, and a later row of the same chunk that finds that
       slot is marked a repeat of it.
    4. The first rows are counted in row order, on the cores by morsel with a
       prefix sum over the morsels, which is what hands each new key the next
       ordinal in the order the chunk introduced it.
    5. The new keys are gathered out of the chunk as one piece of the key
       store, and each part writes the ordinals into the slots it opened and
       gives every repeat its first row's ordinal.

    Before this there was one table and one thread writing it, and that thread
    was most of the map. On ClickBench's URL column it copied a quarter of a
    million keys into the store one at a time and compared every repeat of a new
    key against it. #997 tried parts once and kept the ordinals and the copy on
    one thread, and that came out about even with the table it replaced. Here
    neither is.

    The table grows before a chunk's rows are walked, to room for every one of
    them being new in the fullest part, so no slot moves while the walk holds
    its index. Every part is the same size, which keeps the table one buffer
    and the growing one pass, at the cost of the parts the hash filled less
    than the fullest being a little emptier than they need to be.
    """

    var parts: _Parts
    """The table of hashes."""

    var store: _TextStore
    """The distinct keys, in ordinal order."""

    def __init__(out self):
        """Constructs an empty map with no table allocated yet."""
        self.parts = _Parts()
        self.store = _TextStore()

    def __len__(self) -> Int:
        """Returns the number of ordinals handed out.

        Returns:
            The group count.
        """
        return self.store.groups

    def ordinals(
        mut self, col: StringArray, rows: Int, mut codes: Array[DType.uint32]
    ) raises:
        """Gives every row of one chunk the ordinal its key holds in the map.

        The steps are the ones the struct's docstring lists. The ordinals a
        chunk's new keys get are consecutive and in the order the chunk first
        carried them, which `LastingTuple` relies on to find its first rows.

        Args:
            col: The chunk's key column. Must have no nulls, for the reason
                `LastingKeys.ordinals` gives.
            rows: The chunk's height.
            codes: Filled with one ordinal per row of the chunk.
        """
        if rows <= 0:
            return
        var hashes = Buffer(rows * 8)
        _hash_rows(col, rows, hashes)

        var order = Array[DType.uint32](overwritten=rows)
        var starts = _bucket(rows, hashes, order)
        self.parts.make_room(starts)

        # What the walk found out about each row: zero for a key the map held,
        # one for the first row of a new key and two for a repeat of one.
        # `lead` is the slot a first row opened and the first row of a repeat.
        var kind = Array[DType.uint8](overwritten=rows)
        var lead = Array[DType.uint32](overwritten=rows)
        var added = _walk(
            self.parts.slots,
            self.parts.capacity,
            self.store,
            col,
            hashes,
            order,
            starts,
            codes,
            kind,
            lead,
        )
        self.parts.add(added)

        var firsts = _rank(rows, kind, codes, self.store.groups)
        if len(firsts) > 0:
            self.store.add(
                take_any(AnyArray(col.copy()), firsts).strings().copy()
            )
        _place(self.parts.slots, order, starts, codes, kind, lead)

    def take_keys(mut self) raises -> StringArray:
        """Gives up the key store as a column.

        Returns:
            One row per group, in ordinal order.

        Raises:
            If the pieces cannot be stacked.
        """
        var out = self.store.stacked()
        self.parts = _Parts()
        self.store = _TextStore()
        return out^


struct _TextStore(Movable):
    """The distinct keys of a `LastingText`, in ordinal order.

    In pieces, one per chunk that brought new keys, because a chunk knows its
    own new keys and joining them up is work for the end.
    """

    var pieces: List[StringArray]
    """The keys, a piece at a time."""

    var bases: List[Int]
    """The first ordinal in each piece."""

    var groups: Int
    """Ordinals handed out so far."""

    def __init__(out self):
        """Constructs an empty store."""
        self.pieces = List[StringArray]()
        self.bases = List[Int]()
        self.groups = 0

    def add(mut self, var piece: StringArray):
        """Appends a chunk's new keys, which take the next ordinals.

        Args:
            piece: The keys, in ordinal order.
        """
        self.bases.append(self.groups)
        self.groups += len(piece)
        self.pieces.append(piece^)

    def holds(self, ordinal: Int, col: StringArray, i: Int) -> Bool:
        """Compares a stored key against a row of a chunk.

        Args:
            ordinal: The stored key's ordinal.
            col: The chunk's key column.
            i: The row.

        Returns:
            True if the two are byte-identical.
        """
        # The last piece whose first ordinal is not past this one. There is a
        # piece per chunk that brought new keys, which is a handful, and the
        # probe that got here has already paid a cache miss.
        var low = 0
        var high = len(self.bases) - 1
        while low < high:
            var middle = (low + high + 1) // 2
            if self.bases[middle] <= ordinal:
                low = middle
            else:
                high = middle - 1
        return self.pieces[low].element_equals_foreign(
            ordinal - self.bases[low], col.view(i), col
        )

    def stacked(self) raises -> StringArray:
        """Returns every key as one column.

        Returns:
            One row per group, in ordinal order.

        Raises:
            If the pieces cannot be stacked.
        """
        if len(self.pieces) == 0:
            return StringBuilder().finish()
        if len(self.pieces) == 1:
            return self.pieces[0].copy()
        var each = List[AnyArray](capacity=len(self.pieces))
        for k in range(len(self.pieces)):
            each.append(AnyArray(self.pieces[k].copy()))
        return concat_any(each).strings().copy()


def _walk(
    mut slots: Buffer,
    capacity: Int,
    store: _TextStore,
    col: StringArray,
    hashes: Buffer,
    order: Array[DType.uint32],
    starts: List[Int],
    mut codes: Array[DType.uint32],
    mut kind: Array[DType.uint8],
    mut lead: Array[DType.uint32],
) raises -> List[Int]:
    """Looks each part's rows up in that part, opening slots for new keys.

    One part per task and its rows in row order, so the first row of a new key
    is the one that opens its slot and every later row finds it. Nothing outside
    the part is written except the three per row arrays, and those only at the
    part's own rows.

    Args:
        slots: The table.
        capacity: The slot count of each part.
        store: The keys the map held before this chunk.
        col: The chunk's key column.
        hashes: One hash per row, indexed by row.
        order: The chunk's rows bucketed by part.
        starts: Where each part's rows begin in `order`.
        codes: The ordinal of every row whose key the map held.
        kind: What the walk found for each row.
        lead: The slot a first row opened, or the first row of a repeat.

    Returns:
        How many slots each part opened.
    """
    var added = List[Int](length=LASTING_TEXT_PARTS, fill=0)

    var tasks = _part_tasks(starts[LASTING_TEXT_PARTS])

    def walk(
        t: Int,
    ) raises {mut slots, mut codes, mut kind, mut lead, mut added, imm}:
        for p in range(
            t * LASTING_TEXT_PARTS // tasks,
            (t + 1) * LASTING_TEXT_PARTS // tasks,
        ):
            var table = slots.mut_bitcast[DType.uint64]()
            var hash = hashes.bitcast[DType.uint64]()
            var rows_of = order.unsafe_ptr()
            var found = codes.unsafe_mut_ptr()
            var kinds = kind.unsafe_mut_ptr()
            var leads = lead.unsafe_mut_ptr()
            var mask = UInt64(capacity - 1)
            var base = p * capacity
            var stop = starts[p + 1]
            var opened = 0
            for k in range(starts[p], stop):
                # The table is past every cache on the column this exists
                # for, so the slot the row after next wants is read for
                # while this row is still waiting on its own.
                if k + PROBE_LOOKAHEAD < stop:
                    var next_row = Int(
                        rows_of.unsafe_offset(k + PROBE_LOOKAHEAD).unsafe_load()
                    )
                    var ahead = (
                        hash.unsafe_offset(next_row).unsafe_load() & mask
                    )
                    prefetch[PrefetchOptions().for_read().high_locality()](
                        table.unsafe_offset((base + Int(ahead)) * 2)
                    )
                var i = Int(rows_of.unsafe_offset(k).unsafe_load())
                var wanted = hash.unsafe_offset(i).unsafe_load()
                var at = wanted & mask
                while True:
                    var slot = (base + Int(at)) * 2
                    var word = table.unsafe_offset(slot + 1).unsafe_load()
                    if word == 0:
                        table.unsafe_offset(slot).unsafe_store(wanted)
                        table.unsafe_offset(slot + 1).unsafe_store(
                            LASTING_TEXT_PENDING | UInt64(i)
                        )
                        kinds.unsafe_offset(i).unsafe_write(UInt8(1))
                        leads.unsafe_offset(i).unsafe_write(UInt32(slot))
                        opened += 1
                        break
                    if table.unsafe_offset(slot).unsafe_load() == wanted:
                        if word & LASTING_TEXT_PENDING != 0:
                            var first = Int(word & ~LASTING_TEXT_PENDING)
                            if col.element_equals(i, first):
                                kinds.unsafe_offset(i).unsafe_write(UInt8(2))
                                leads.unsafe_offset(i).unsafe_write(
                                    UInt32(first)
                                )
                                break
                        elif store.holds(Int(word) - 1, col, i):
                            kinds.unsafe_offset(i).unsafe_write(UInt8(0))
                            found.unsafe_offset(i).unsafe_write(
                                UInt32(Int(word) - 1)
                            )
                            break
                    at = (at + 1) & mask
            added[p] = opened

    parallel_for(walk, tasks)
    return added^


def _walk_exact(
    mut slots: Buffer,
    capacity: Int,
    hashes: Buffer,
    order: Array[DType.uint32],
    starts: List[Int],
    mut codes: Array[DType.uint32],
    mut kind: Array[DType.uint8],
    mut lead: Array[DType.uint32],
) raises -> List[Int]:
    """`_walk` for a key whose hash is the key, so a matching hash is a match.

    A fixed width key is hashed by a bijection on its bits, so two keys with one
    hash are one key and there is nothing to compare. That takes the column and
    the key store out of the walk, and it is a function of its own rather than a
    parameter on `_walk` because those are what `_walk` is built around.

    Args:
        slots: The table.
        capacity: The slot count of each part.
        hashes: One hash per row, indexed by row.
        order: The chunk's rows bucketed by part.
        starts: Where each part's rows begin in `order`.
        codes: The ordinal of every row whose key the map held.
        kind: What the walk found for each row.
        lead: The slot a first row opened, or the first row of a repeat.

    Returns:
        How many slots each part opened.
    """
    var added = List[Int](length=LASTING_TEXT_PARTS, fill=0)

    var tasks = _part_tasks(starts[LASTING_TEXT_PARTS])

    def walk(
        t: Int,
    ) raises {mut slots, mut codes, mut kind, mut lead, mut added, imm}:
        for p in range(
            t * LASTING_TEXT_PARTS // tasks,
            (t + 1) * LASTING_TEXT_PARTS // tasks,
        ):
            var table = slots.mut_bitcast[DType.uint64]()
            var hash = hashes.bitcast[DType.uint64]()
            var rows_of = order.unsafe_ptr()
            var found = codes.unsafe_mut_ptr()
            var kinds = kind.unsafe_mut_ptr()
            var leads = lead.unsafe_mut_ptr()
            var mask = UInt64(capacity - 1)
            var base = p * capacity
            var stop = starts[p + 1]
            var opened = 0
            for k in range(starts[p], stop):
                if k + PROBE_LOOKAHEAD < stop:
                    var next_row = Int(
                        rows_of.unsafe_offset(k + PROBE_LOOKAHEAD).unsafe_load()
                    )
                    var ahead = (
                        hash.unsafe_offset(next_row).unsafe_load() & mask
                    )
                    prefetch[PrefetchOptions().for_read().high_locality()](
                        table.unsafe_offset((base + Int(ahead)) * 2)
                    )
                var i = Int(rows_of.unsafe_offset(k).unsafe_load())
                var wanted = hash.unsafe_offset(i).unsafe_load()
                var at = wanted & mask
                while True:
                    var slot = (base + Int(at)) * 2
                    var word = table.unsafe_offset(slot + 1).unsafe_load()
                    if word == 0:
                        table.unsafe_offset(slot).unsafe_store(wanted)
                        table.unsafe_offset(slot + 1).unsafe_store(
                            LASTING_TEXT_PENDING | UInt64(i)
                        )
                        kinds.unsafe_offset(i).unsafe_write(UInt8(1))
                        leads.unsafe_offset(i).unsafe_write(UInt32(slot))
                        opened += 1
                        break
                    if table.unsafe_offset(slot).unsafe_load() == wanted:
                        if word & LASTING_TEXT_PENDING != 0:
                            kinds.unsafe_offset(i).unsafe_write(UInt8(2))
                            leads.unsafe_offset(i).unsafe_write(
                                UInt32(Int(word & ~LASTING_TEXT_PENDING))
                            )
                        else:
                            kinds.unsafe_offset(i).unsafe_write(UInt8(0))
                            found.unsafe_offset(i).unsafe_write(
                                UInt32(Int(word) - 1)
                            )
                        break
                    at = (at + 1) & mask
            added[p] = opened

    parallel_for(walk, tasks)
    return added^


def _place(
    mut slots: Buffer,
    order: Array[DType.uint32],
    starts: List[Int],
    mut codes: Array[DType.uint32],
    kind: Array[DType.uint8],
    lead: Array[DType.uint32],
) raises:
    """Settles the slots a chunk opened and the rows that repeat a new key.

    After `_rank`, which is what gave every first row its ordinal. A slot a
    first row opened gets that ordinal, and a repeat gets its first row's, which
    is in the same part because the two have the same hash.

    Args:
        slots: The table.
        order: The chunk's rows bucketed by part.
        starts: Where each part's rows begin in `order`.
        codes: The ordinal of every row. Read for first rows and written for
            repeats.
        kind: What the walk found for each row.
        lead: The slot a first row opened, or the first row of a repeat.
    """

    var tasks = _part_tasks(starts[LASTING_TEXT_PARTS])

    def place(t: Int) raises {mut slots, mut codes, imm}:
        for p in range(
            t * LASTING_TEXT_PARTS // tasks,
            (t + 1) * LASTING_TEXT_PARTS // tasks,
        ):
            var table = slots.mut_bitcast[DType.uint64]()
            var rows_of = order.unsafe_ptr()
            var out = codes.unsafe_mut_ptr()
            var kinds = kind.unsafe_ptr()
            var leads = lead.unsafe_ptr()
            for k in range(starts[p], starts[p + 1]):
                var i = Int(rows_of.unsafe_offset(k).unsafe_load())
                var what = kinds.unsafe_offset(i).unsafe_load()
                if what == 1:
                    var slot = Int(leads.unsafe_offset(i).unsafe_load())
                    table.unsafe_offset(slot + 1).unsafe_store(
                        UInt64(out.unsafe_offset(i).unsafe_load()) + 1
                    )
                elif what == 2:
                    var first = Int(leads.unsafe_offset(i).unsafe_load())
                    out.unsafe_offset(i).unsafe_write(
                        out.unsafe_offset(first).unsafe_load()
                    )

    parallel_for(place, tasks)


def _hash_rows(col: StringArray, rows: Int, mut hashes: Buffer) raises:
    """Hashes every row of a text chunk, on the cores.

    Args:
        col: The chunk's key column.
        rows: The chunk's height.
        hashes: Filled with one hash per row, indexed by row.
    """
    var out = hashes.mut_bitcast[DType.uint64]()

    def body(begin: Int, stop: Int) raises {imm}:
        for i in range(begin, stop):
            out.unsafe_offset(i).unsafe_store(
                hash_bytes(col.unsafe_bytes(i), DEFAULT_SEED)
            )

    parallel_morsels(body, rows, LASTING_TEXT_MORSEL)


def _hash_keys[
    dt: DType
](col: Array[dt], start: Int, rows: Int, mut hashes: Buffer) raises:
    """Hashes a run of a fixed width column like `hash_chunk`, on the cores.

    A whole register at a time, the tail too, for the reason `hash_chunk`
    gives. A morsel is a multiple of the register width, so only the last one
    has a tail, and it runs into the padding of both buffers rather than into
    another morsel's rows.

    Args:
        col: The key column.
        start: The first row to hash.
        rows: How many rows to hash.
        hashes: Filled with one hash per row, indexed from `start`.

    Parameters:
        dt: The key dtype.
    """
    comptime width = simd_width_of[DType.uint64]()

    def body(begin: Int, stop: Int) raises {mut hashes, imm}:
        var src = col.unsafe_ptr().unsafe_offset(start)
        var out = hashes.mut_bitcast[DType.uint64]()
        var i = begin
        while i < stop:
            var k = key_bits(src.unsafe_offset(i).unsafe_load[width=width]())
            out.unsafe_offset(i).unsafe_store(mix(k, DEFAULT_SEED))
            i += width

    parallel_morsels(body, rows, LASTING_TEXT_MORSEL)


def _bucket(
    rows: Int, hashes: Buffer, mut order: Array[DType.uint32]
) raises -> List[Int]:
    """Sorts a chunk's rows by part, keeping row order inside each part.

    A counting sort by morsel: each morsel counts its rows per part, a prefix
    sum over part and then morsel says where each morsel writes each part's
    rows, and each morsel writes its own. A part's rows come out in morsel
    order and in row order inside a morsel, which is row order.

    Args:
        rows: The chunk's height.
        hashes: One hash per row, indexed by row.
        order: Filled with the rows, bucketed by part.

    Returns:
        Where each part's rows begin in `order`, with `rows` after the last.
    """
    var morsels = (rows + LASTING_TEXT_MORSEL - 1) // LASTING_TEXT_MORSEL
    var counts = List[Int](length=morsels * LASTING_TEXT_PARTS, fill=0)

    def count(m: Int) raises {mut counts, imm}:
        var hash = hashes.bitcast[DType.uint64]()
        var at = m * LASTING_TEXT_PARTS
        for i in range(
            m * LASTING_TEXT_MORSEL, min(rows, (m + 1) * LASTING_TEXT_MORSEL)
        ):
            var p = Int(
                hash.unsafe_offset(i).unsafe_load() >> LASTING_TEXT_PART_SHIFT
            )
            counts[at + p] += 1

    parallel_for(count, morsels)

    var starts = List[Int](capacity=LASTING_TEXT_PARTS + 1)
    var total = 0
    for p in range(LASTING_TEXT_PARTS):
        starts.append(total)
        for m in range(morsels):
            var n = counts[m * LASTING_TEXT_PARTS + p]
            counts[m * LASTING_TEXT_PARTS + p] = total
            total += n
    starts.append(total)

    def scatter(m: Int) raises {mut counts, mut order, imm}:
        var hash = hashes.bitcast[DType.uint64]()
        var into = order.unsafe_mut_ptr()
        var at = m * LASTING_TEXT_PARTS
        for i in range(
            m * LASTING_TEXT_MORSEL, min(rows, (m + 1) * LASTING_TEXT_MORSEL)
        ):
            var p = Int(
                hash.unsafe_offset(i).unsafe_load() >> LASTING_TEXT_PART_SHIFT
            )
            into.unsafe_offset(counts[at + p]).unsafe_write(UInt32(i))
            counts[at + p] += 1

    parallel_for(scatter, morsels)
    return starts^


def _rank(
    rows: Int,
    kind: Array[DType.uint8],
    mut codes: Array[DType.uint32],
    base: Int,
) raises -> List[Int]:
    """Hands every first row the next ordinal, in row order, on the cores.

    A count of first rows per morsel, a prefix sum over the morsels, and then
    each morsel numbers its own from where the sum says it starts.

    Args:
        rows: The chunk's height.
        kind: What the walk found for each row. One marks a first row.
        codes: Written at every first row.
        base: The ordinal the chunk's first new key gets.

    Returns:
        The first rows, in row order, which is ordinal order.
    """
    var morsels = (rows + LASTING_TEXT_MORSEL - 1) // LASTING_TEXT_MORSEL
    var counts = List[Int](length=morsels, fill=0)

    def count(m: Int) raises {mut counts, imm}:
        var kinds = kind.unsafe_ptr()
        var n = 0
        for i in range(
            m * LASTING_TEXT_MORSEL, min(rows, (m + 1) * LASTING_TEXT_MORSEL)
        ):
            if kinds.unsafe_offset(i).unsafe_load() == 1:
                n += 1
        counts[m] = n

    parallel_for(count, morsels)

    var total = 0
    for m in range(morsels):
        var n = counts[m]
        counts[m] = total
        total += n

    var firsts = List[Int](unsafe_uninit_length=total)

    def number(m: Int) raises {mut firsts, mut codes, imm}:
        var kinds = kind.unsafe_ptr()
        var out = codes.unsafe_mut_ptr()
        var into = firsts.unsafe_ptr()
        var at = counts[m]
        for i in range(
            m * LASTING_TEXT_MORSEL, min(rows, (m + 1) * LASTING_TEXT_MORSEL)
        ):
            if kinds.unsafe_offset(i).unsafe_load() == 1:
                into.unsafe_offset(at).unsafe_write(i)
                out.unsafe_offset(i).unsafe_write(UInt32(base + at))
                at += 1

    parallel_for(number, morsels)
    return firsts^


def _rehash(table: Buffer, capacity: Int, grown: Int) raises -> Buffer:
    """Moves every live slot into a fresh table with bigger parts.

    Each part moves into its own part of the new table, on the cores. That needs
    no hashing and no comparison: the stored hash is what the slot is found by,
    and two keys already in the table are already known to be different keys,
    so a collision in the new table is settled by probing on alone. Only
    settled slots are ever here, because the table grows before a chunk's rows
    are walked and not during.

    Args:
        table: The slots.
        capacity: The slot count of each part now.
        grown: The slot count of each part after. A power of two.

    Returns:
        The new slots.
    """
    var bigger = Buffer(LASTING_TEXT_PARTS * grown * 2 * 8)

    def move(p: Int) raises {mut bigger, imm}:
        var into = bigger.mut_bitcast[DType.uint64]()
        var from_ = table.bitcast[DType.uint64]()
        var mask = UInt64(grown - 1)
        var base = p * grown
        for s in range(p * capacity, (p + 1) * capacity):
            var ordinal = from_.unsafe_offset(s * 2 + 1).unsafe_load()
            if ordinal == 0:
                continue
            var wanted = from_.unsafe_offset(s * 2).unsafe_load()
            var at = wanted & mask
            while (
                into.unsafe_offset((base + Int(at)) * 2 + 1).unsafe_load() != 0
            ):
                at = (at + 1) & mask
            into.unsafe_offset((base + Int(at)) * 2).unsafe_store(wanted)
            into.unsafe_offset((base + Int(at)) * 2 + 1).unsafe_store(ordinal)

    parallel_for(move, LASTING_TEXT_PARTS)
    return bigger^


struct LastingKeys(Movable):
    """The keys a streaming operator has seen, and the ordinal each was given.

    Holds one of two maps and the keys themselves. Which map it holds is decided
    by the first chunk and can change once, from the direct table to the hash
    table, if a later chunk holds a key the direct table's window does not cover.
    """

    var parts: _Parts
    """The hash route's map, the same table `LastingText` writes and written
    the same way. Empty while the direct route is live."""

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
        self.parts = _Parts()
        self.slots = Array[DType.uint32](1)
        self.span = 0
        self.base = 0
        self.groups = 0
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
        if not key.is_flat():
            self.ordinals(key.decoded(), rows, codes)
            return
        if key.is_string():
            # Text has one route and the first chunk decides nothing, so there
            # is no plan to make and no window to fix. The map holds the keys
            # itself, which is why nothing is appended to the store here.
            self.opened = True
            self.textual = True
            self.text.ordinals(key.strings(), rows, codes)
            self.groups = self.text.__len__()
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
    ) raises:
        """Takes the rows of a chunk from `start` on through the hash table.

        The steps `LastingText` lists, with the comparison gone because the
        hash is the key. This was one table written by one thread, the join's
        build called a thousand rows at a time, and on ClickBench's UserID that
        thread was most of what q15 cost.

        The rows before `start` are the ones the direct table took before a key
        left its window, which happens at most once in a query, so the rows
        after it go through arrays of their own and are copied back.
        """
        var count = rows - start
        if count <= 0:
            return
        if start == 0:
            self._take[dt](col, 0, count, codes, firsts)
            return
        var ours = Array[DType.uint32](overwritten=count)
        self._take[dt](col, start, count, ours, firsts)
        unsafe_memcpy(
            dest=codes.unsafe_mut_ptr().unsafe_offset(start),
            src=ours.unsafe_ptr(),
            count=count,
        )

    def _take[
        dt: DType
    ](
        mut self,
        col: Array[dt],
        start: Int,
        count: Int,
        mut ours: Array[DType.uint32],
        mut firsts: List[Int],
    ) raises:
        """Does what `_hashed` says for `count` rows from `start`, writing
        their ordinals to the front of `ours`."""
        var hashes = Buffer(count * 8)
        _hash_keys[dt](col, start, count, hashes)

        var order = Array[DType.uint32](overwritten=count)
        var starts = _bucket(count, hashes, order)
        self.parts.make_room(starts)

        var kind = Array[DType.uint8](overwritten=count)
        var lead = Array[DType.uint32](overwritten=count)
        var added = _walk_exact(
            self.parts.slots,
            self.parts.capacity,
            hashes,
            order,
            starts,
            ours,
            kind,
            lead,
        )
        self.parts.add(added)

        var fresh = _rank(count, kind, ours, self.groups)
        _place(self.parts.slots, order, starts, ours, kind, lead)
        self.groups += len(fresh)
        for f in fresh:
            firsts.append(start + f)

    def _rehash(mut self) raises:
        """Rebuilds the hash table from the keys the direct table collected.

        The keys are distinct and they are in ordinal order, so inserting them in
        that order into an empty table gives each one back the ordinal it already
        had, and nothing downstream has to be told the route changed.
        """
        self.span = 0
        self.slots = Array[DType.uint32](1)
        self.parts = _Parts()
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


def tuple_holds(type: LogicalType) -> Bool:
    """Reports whether `LastingTuple` can write a key of this type as bytes.

    Text and binary, the integers and the three temporal types. A float is left
    out because two of its bit patterns can be one value, zero and minus zero,
    and a byte comparison would make them two groups where the chunk's own
    grouping made one. A boolean is left out because it is stored as bits, and a
    dictionary or a nested column because what it stores is not its value.

    Args:
        type: The key column's type.

    Returns:
        True if a key of this type can be part of a lasting tuple.
    """
    if type.is_variable_width():
        return True
    if type.physical == DType.bool:
        return False
    return type.is_integer() or type.is_temporal()


struct LastingTuple(Movable):
    """The key tuples a streaming operator has seen, and the ordinal each got.

    The module docstring has the argument. Each chunk is grouped on its own,
    its distinct tuples are written out as bytes, and the bytes are looked up
    in a `LastingText`, which hands an ordinal to a tuple the first time it
    sees one and the same ordinal every time after.
    """

    var text: LastingText
    """The map from a tuple's bytes to its ordinal."""

    var keys: List[List[AnyArray]]
    """Per key column, its values one row per group in ordinal order, in as
    many pieces as there were chunks that introduced a group."""

    var groups: Int
    """Ordinals handed out so far."""

    def __init__(out self):
        """Constructs an empty map."""
        self.text = LastingText()
        self.keys = List[List[AnyArray]]()
        self.groups = 0

    def __len__(self) -> Int:
        """Returns the number of ordinals handed out.

        Returns:
            The group count.
        """
        return self.groups

    def ordinals(
        mut self,
        columns: List[AnyArray],
        at: List[Int],
        rows: Int,
        mut codes: Array[DType.uint32],
    ) raises:
        """Gives every row of one chunk the ordinal its tuple holds in the map.

        Args:
            columns: The chunk's columns, borrowed.
            at: Which of them are the keys, in the order the output has them.
                Every one must be a type `tuple_holds` accepts.
            rows: The chunk's height.
            codes: Filled with one ordinal per row of the chunk.

        Raises:
            If the chunk cannot be grouped or a key cannot be gathered.
        """
        if rows <= 0:
            return
        if len(self.keys) == 0:
            for _ in range(len(at)):
                self.keys.append(List[AnyArray]())

        # Two passes on the cores, one to size every row's bytes and one to
        # write them, with the running total between them the only serial part.
        # What a row asks of each key is read out here once rather than asked
        # of the column on every row, which was a third of what writing cost.
        var width = List[Int](capacity=len(at))
        var gaps = List[Bool](capacity=len(at))
        var texts = List[StringArray](capacity=len(at))
        for k in range(len(at)):
            ref col = columns[at[k]]
            gaps.append(col.null_count() > 0)
            if not col.is_flat():
                width.append(0)
                var flat = col.decoded()
                texts.append(StringArray(copy=flat.strings()))
            elif col.is_string():
                width.append(0)
                texts.append(StringArray(copy=col.strings()))
            else:
                width.append(dtype_size(col.dtype()))
                texts.append(StringBuilder().finish())
        # A key that is fixed width and never missing costs every row the same,
        # so it is counted once here rather than per row.
        var keys = len(at)
        var fixed = keys
        for k in range(keys):
            if width[k] > 0 and not gaps[k]:
                fixed += width[k]
        var sizes = Array[DType.int64](overwritten=rows)
        var sized = sizes.unsafe_mut_ptr()

        # Both passes go a key at a time down the morsel rather than a row at a
        # time across the keys. A row at a time asked the lists which key was
        # which on every row and paid a bounds check for each question, which
        # was a third of what writing cost on q18.
        def size(begin: Int, stop: Int) raises {imm}:
            for i in range(begin, stop):
                sized[i] = Int64(fixed)
            for k in range(keys):
                ref col = columns[at[k]]
                var gap = gaps[k]
                if width[k] == 0:
                    ref text = texts[k]
                    for i in range(begin, stop):
                        if gap and not col.is_valid(i):
                            continue
                        sized[i] += Int64(4 + text.byte_length(i))
                elif gap:
                    var w = Int64(width[k])
                    for i in range(begin, stop):
                        if col.is_valid(i):
                            sized[i] += w

        parallel_morsels(size, rows, LASTING_TEXT_MORSEL)

        # Where each row's bytes start. The write pass moves each entry along as
        # it writes a key, so once it is done an entry is where its row ends.
        var starts = Array[DType.int64](overwritten=rows)
        var begun = starts.unsafe_mut_ptr()
        var total = 0
        for i in range(rows):
            begun[i] = Int64(total)
            total += Int(sized[i])

        var payload = Buffer(overwritten=max(total, 1))
        var out = payload.unsafe_mut_ptr()
        var views = Buffer(rows * VIEW_SIZE)
        var target = views.unsafe_mut_ptr().unsafe_bitcast[StringView]()

        def write(begin: Int, stop: Int) raises {imm}:
            for k in range(keys):
                ref col = columns[at[k]]
                var gap = gaps[k]
                var w = width[k]
                if w == 0:
                    ref text = texts[k]
                    for i in range(begin, stop):
                        var end = Int(begun[i])
                        if gap and not col.is_valid(i):
                            out[end] = 0
                            begun[i] = Int64(end + 1)
                            continue
                        out[end] = 1
                        var held = text.unsafe_bytes(i)
                        var n = len(held)
                        out.unsafe_offset(end + 1).unsafe_bitcast[
                            UInt32
                        ]().unsafe_store[alignment=1](UInt32(n))
                        unsafe_memcpy(
                            dest=out + end + 5, src=held.unsafe_ptr(), count=n
                        )
                        begun[i] = Int64(end + 5 + n)
                    continue
                var src = col.unsafe_ptr[DType.uint8]()
                for i in range(begin, stop):
                    var end = Int(begun[i])
                    if gap and not col.is_valid(i):
                        out[end] = 0
                        begun[i] = Int64(end + 1)
                        continue
                    out[end] = 1
                    var at_value = src + i * w
                    var to = out.unsafe_offset(end + 1)
                    if w == 8:
                        to.unsafe_bitcast[UInt64]().unsafe_store[alignment=1](
                            at_value.unsafe_bitcast[UInt64]().unsafe_load[
                                alignment=1
                            ]()
                        )
                    elif w == 4:
                        to.unsafe_bitcast[UInt32]().unsafe_store[alignment=1](
                            at_value.unsafe_bitcast[UInt32]().unsafe_load[
                                alignment=1
                            ]()
                        )
                    else:
                        unsafe_memcpy(dest=to, src=at_value, count=w)
                    begun[i] = Int64(end + 1 + w)
            for i in range(begin, stop):
                var n = Int(sized[i])
                var first = Int(begun[i]) - n
                if n <= INLINE_CAPACITY:
                    target[i] = make_inline_at(out + first, n)
                else:
                    target[i] = make_long_at(out + first, n, 0, first)

        parallel_morsels(write, rows, LASTING_TEXT_MORSEL)

        var tuples = StringArray(views^, payload^, Bitmap(rows), rows)
        self.text.ordinals(tuples, rows, codes)

        # The map hands ordinals out in row order, so a group's first row is the
        # row whose ordinal is the next one due.
        var firsts = List[Int]()
        var seen = codes.unsafe_ptr()
        for i in range(rows):
            if Int(seen[i]) == self.groups:
                firsts.append(i)
                self.groups += 1
        if len(firsts) > 0:
            for k in range(len(at)):
                self.keys[k].append(take_any(columns[at[k]], firsts))

    def take_keys(mut self) raises -> List[AnyArray]:
        """Stacks each key column's pieces and gives up the store.

        Returns:
            One column per key, one row per group, in ordinal order.

        Raises:
            If the pieces cannot be stacked.
        """
        var out = List[AnyArray](capacity=len(self.keys))
        for k in range(len(self.keys)):
            out.append(concat_any(self.keys[k]))
        self.keys = List[List[AnyArray]]()
        self.text = LastingText()
        self.groups = 0
        return out^
