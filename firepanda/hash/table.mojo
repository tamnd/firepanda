"""Open addressing with linear probing, over 64-bit hashes.

The table is untyped. It maps a hash from `firepanda/hash/function.mojo` to a
dense group ordinal and knows nothing about the dtype it came from, which is what
lets a join between an int32 column and an int64 column work without
instantiating a table per pair.

It stores the hash rather than the key, and that is not the usual shortcut of
comparing hashes and hoping. `mix` is a bijection on 64 bits, so two keys collide
in the stored value exactly when they are the same key, and the comparison is
exact. What it buys is that the caller hands over one number per row instead of
two, the probe does one load per step instead of two, and a growth reinserts
without hashing anything again.

That argument covers every fixed width dtype and does not cover text, because
sixteen bytes of name do not fit in eight bytes of hash and no function can make
them. `build_strings` is the same probe with the comparison put back: on a hash
match it compares the row against the row that first produced that ordinal, and
keeps probing if they differ. The two builds are otherwise the same loop, kept
apart rather than merged behind a flag so that the fixed width one stays a loop
with nothing in it.

Layout is one flat buffer of 16-byte slots, the hash in the first eight bytes and
the ordinal in the second eight. Two parallel buffers would be the obvious
alternative and would touch two cache lines on every probe instead of one, which
on a table that does not fit in cache is the whole cost of the operation.

The ordinal is stored plus one so that zero means empty. `Buffer` hands back
zeroed memory, so an empty table needs no initialization pass at all, and a fresh
one after a growth needs none either. That is worth more than it sounds like on a
table that doubles four or five times during a build.

Load factor is one half. Linear probing degrades sharply above that and the
memory is not the scarce thing here.

Sizing is measured rather than asked for. A caller who knew the group count would
not need the table, and the two obvious guesses are both bad: starting small
rehashes a million-group column five or six times, and starting at the row count
hands a ten-thousand-group column a thirty two megabyte table where every probe
is a cache miss and a TLB miss. Both of those were measured here, and the second
one was the worse of the two. So the build watches its own discovery rate at two
checkpoints and sizes from what it sees; `project_groups` is where that
extrapolation lives.
"""

from std.sys.intrinsics import PrefetchOptions, prefetch

from firepanda.array.array import Array
from firepanda.array.strings import StringArray
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer

from .function import DEFAULT_SEED

comptime MIN_CAPACITY = 16
"""Slots in the smallest table. Small enough not to matter, large enough that a
handful of groups never triggers a growth."""

comptime SLOT_WORDS = 2
"""64-bit words per slot: the hash, then the ordinal plus one."""

comptime SIZING_EARLY = 1 << 12
"""Rows after which the build first sizes the table for the whole column.

Early enough that the doublings it skips are the ones that have not happened yet,
late enough that the group count it is reading means something.
"""

comptime SIZING_LATE = 1 << 16
"""Rows after which the build sizes the table again, and this time believes it.

Four thousand rows cannot tell a column with ten thousand groups apart from a
column where every row is its own group; both look entirely distinct that early.
Sixty five thousand can. So the early estimate is capped and the late one is not.
"""

comptime EARLY_JUMP = 16
"""Most the early estimate is allowed to multiply the group count seen so far.

Without it a column of four thousand distinct values in its first four thousand
rows would size itself for the whole column, which is right if the values keep
being distinct and a hundredfold over-allocation if they start repeating.
"""

comptime PROBE_LOOKAHEAD = 8
"""Rows the batch probe runs ahead of itself when issuing prefetches.

Far enough that the line has arrived by the time the probe wants it, near enough
that it has not been evicted again. Eight is where it stopped improving on the
reference machine; there is nothing fundamental about the number.
"""


def next_power_of_two(n: Int) -> Int:
    """Rounds up to a power of two.

    Args:
        n: The value to round.

    Returns:
        The smallest power of two that is at least `n`, and at least 1.
    """
    var p = 1
    while p < n:
        p *= 2
    return p


def project_groups(seen: Int, half: Int, rows: Int, n: Int) -> Int:
    """Guesses a column's total group count from the part of it already built.

    Two numbers go in: how many groups the first `rows` rows produced, and how
    many the first half of them produced. The difference is the discovery rate,
    and the shape of a group by is that the rate either collapses almost
    immediately, because the column is a category and every value has already
    been seen, or it does not collapse at all, because the column is an
    identifier and every value is new. Those two cases are most real columns and
    this tells them apart.

    Between them sits the case this is worst at, a column of genuinely random
    values with a cardinality somewhere in the middle. The rate is still falling
    when the sample ends, extrapolating it flat overshoots, and the table ends up
    several times larger than it needed to be. That costs memory and it does not
    cost correctness, which is the right way round; the opposite error makes the
    build rehash itself repeatedly and that is the expensive one.

    Args:
        seen: Groups found in the first `rows` rows.
        half: Groups found in the first `rows / 2` of them.
        rows: How many rows have been built so far.
        n: The column length.

    Returns:
        A guess at the final group count, never above `n`.
    """
    var found = seen - half
    if found * 4 < half:
        # Discovery has all but stopped. A quarter of headroom covers the tail
        # without sizing for a column that is not going to arrive.
        return min(n, seen + seen // 4)
    return min(n, seen + found * (n - rows) // (rows // 2))


struct HashTable(Movable, Sized):
    """A key-bits to group-ordinal map."""

    var _slots: Buffer
    var _capacity: Int
    var _mask: UInt64
    var _count: Int
    var _seed: UInt64
    var _stage: Int
    var _half: Int
    var _mark: Int

    def __init__(out self, expected: Int = 0, seed: UInt64 = DEFAULT_SEED):
        """Constructs a table.

        Args:
            expected: A guess at the number of distinct keys. The table is sized
                to hold that many at the load factor without growing. A wrong
                guess costs a rehash, not correctness.
            seed: The per-query seed. See `function.mojo` for why it is per query.
        """
        var capacity = next_power_of_two(expected * 2)
        if capacity < MIN_CAPACITY:
            capacity = MIN_CAPACITY
        self._slots = Buffer(capacity * SLOT_WORDS * 8)
        self._capacity = capacity
        self._mask = UInt64(capacity - 1)
        self._count = 0
        self._seed = seed

        # The sizing schedule lives here rather than in `build` because `build`
        # runs once per chunk and the schedule runs once per column.
        self._stage = 0
        self._half = 0
        self._mark = SIZING_EARLY // 2

    def __len__(self) -> Int:
        """Returns the number of distinct keys inserted.

        Returns:
            The group count.
        """
        return self._count

    def capacity(self) -> Int:
        """Returns the number of slots.

        Returns:
            A power of two.
        """
        return self._capacity

    def seed(self) -> UInt64:
        """Returns the seed this table hashes with.

        Returns:
            The seed.
        """
        return self._seed

    def find(self, hash: UInt64) -> Int:
        """Looks a key up without inserting it.

        Args:
            hash: `hash_of(value, seed())`. Passed in rather than computed here
                because the caller hashes a chunk of rows at a time.

        Returns:
            The group ordinal, or -1 if the key is not in the table.
        """
        var slots = self._slots.bitcast[DType.uint64]()
        var i = hash & self._mask
        while True:
            var at = Int(i) * SLOT_WORDS
            var ordinal = slots.unsafe_offset(at + 1).unsafe_load()
            if ordinal == 0:
                return -1
            if slots.unsafe_offset(at).unsafe_load() == hash:
                return Int(ordinal) - 1
            i = (i + 1) & self._mask

    def insert(mut self, hash: UInt64) -> Int:
        """Looks a key up and inserts it if it is not there.

        Args:
            hash: `hash_of(value, seed())`.

        Returns:
            The group ordinal, which is the one it already had or the next one up.
        """
        # The growth check comes first so the probe below can assume there is an
        # empty slot to find. A full table with linear probing does not fail, it
        # spins.
        if (self._count + 1) * 2 > self._capacity:
            self._grow()

        var slots = self._slots.bitcast[DType.uint64]()
        var i = hash & self._mask
        while True:
            var at = Int(i) * SLOT_WORDS
            var ordinal = slots.unsafe_offset(at + 1).unsafe_load()
            if ordinal == 0:
                var assigned = self._count
                slots.unsafe_offset(at).unsafe_write(hash)
                slots.unsafe_offset(at + 1).unsafe_write(UInt64(assigned + 1))
                self._count = assigned + 1
                return assigned
            if slots.unsafe_offset(at).unsafe_load() == hash:
                return Int(ordinal) - 1
            i = (i + 1) & self._mask

    def insert_string(
        mut self,
        hash: UInt64,
        row: Int,
        col: StringArray,
        mut firsts: List[Int],
    ) -> Int:
        """Looks a string key up and inserts it if it is not there.

        `insert` with the comparison `build_strings` does. It exists for the
        merge at the end of a parallel string build, which is handed one key per
        group per worker and has to fold them into a single table. A hash match
        is a candidate rather than an answer here, and what settles it is the row
        that first produced the ordinal, which is why `firsts` is passed in and
        written to rather than being something the caller keeps to itself.

        Args:
            hash: The key. A worker's table already holds it, so the merge does
                not hash the row again.
            row: The row this key came from, which becomes the group's
                representative row if the key turns out to be new.
            col: The column, for the comparison. Indexed by absolute row.
            firsts: The representative row per ordinal, read when a hash matches
                and appended to when a key is new.

        Returns:
            The group ordinal, which is the one it already had or the next one up.
        """
        if (self._count + 1) * 2 > self._capacity:
            self._grow()

        var slots = self._slots.bitcast[DType.uint64]()
        var i = hash & self._mask
        while True:
            var at = Int(i) * SLOT_WORDS
            var ordinal = slots.unsafe_offset(at + 1).unsafe_load()
            if ordinal == 0:
                var assigned = self._count
                slots.unsafe_offset(at).unsafe_write(hash)
                slots.unsafe_offset(at + 1).unsafe_write(UInt64(assigned + 1))
                self._count = assigned + 1
                firsts.append(row)
                return assigned
            if slots.unsafe_offset(at).unsafe_load() == hash:
                if col.element_equals(row, firsts[Int(ordinal) - 1]):
                    return Int(ordinal) - 1
            i = (i + 1) & self._mask

    def build(
        mut self,
        hashes: Buffer,
        validity: Bitmap,
        has_null: Bool,
        base: Int,
        rank: Int,
        count: Int,
        rows: Int,
        offset: Int,
        mut codes: Array[DType.uint32],
        mut firsts: List[Int],
    ):
        """Inserts a chunk of a column's keys in one call.

        This is `insert` in a loop and it exists because `insert` in a loop is
        not the same speed. Every field this touches lives in a local for the
        duration, so the probe reads a register instead of reloading the table
        through a pointer on every row, and the compiler can keep the whole loop
        in flight. Measured on the reference machine that was 5.1 ns per row down
        to 2.5 ns on a low cardinality column, which is the difference between
        losing to `Dict` and beating it. It is the reason this is a method rather
        than something `factorize` writes for itself; the alternative was handing
        the slot pointer out and hoping nobody kept it.

        It takes a chunk rather than a column so that the caller can hash a few
        thousand rows into a pair of small buffers and probe them while they are
        still in cache, instead of hashing the column into two buffers the size
        of the column and reading them back from memory. The sizing schedule
        survives across the calls because it lives on the table, so a column
        built in three hundred chunks sizes itself exactly as one built in a
        single call would. Call the chunks in row order and pass the same `rows`
        every time, which is what building a column in pieces means.

        The row a chunk starts at and the number of rows this table has already
        seen are two arguments rather than one because a parallel build gives a
        worker a slice out of the middle of a column. Its rows keep their
        absolute numbers, because that is what the validity bitmap and the output
        are indexed by, while its table is sized against the slice it was given
        rather than against a column it will only ever see a fraction of.

        The prefetch is issued from here for the same reason. It only pays on a
        table too large for cache, and on a small one it is not free, but a
        version of this loop with the prefetch removed measured no faster on
        either, so it stays.

        Nulls are handled here rather than by the caller because splitting the
        column around them would break the run this is trying to keep together.
        The branch predicts perfectly on a column with no nulls, which is most of
        them.

        Args:
            hashes: Hashes for this chunk, indexed from zero, from `hash_chunk`.
            validity: The column's validity bitmap, indexed by absolute row. Read
                only when `has_null`.
            has_null: Whether the column has any nulls at all.
            base: The absolute row index this chunk starts at.
            rank: How many rows this table has already been given. Equal to
                `base` for a column built front to back on one thread.
            count: How many rows are in this chunk.
            rows: How many rows this table will be given in total, which the
                sizing schedule needs and which does not change between chunks.
            offset: Added to every ordinal written to `codes`. This is how the
                caller reserves low ordinals for groups the table knows nothing
                about, which right now means the null group.
            codes: Where the per-row ordinals go, indexed by absolute row. Must
                hold `rows` rows.
            firsts: Appended with the absolute row index of every key that was
                new, in ordinal order, so the caller can read the key values back
                out of its own column without the table knowing what a value is.
        """
        var hash = hashes.bitcast[DType.uint64]()
        var out = codes.unsafe_ptr()
        var slots = self._slots.bitcast[DType.uint64]()
        var mask = self._mask
        var capacity = self._capacity
        var found = self._count

        # The schedule is walked with one comparison per row rather than four.
        # `stage` says which of the two checkpoints is next and whether the row
        # coming up is its halfway mark or its end.
        var stage = self._stage
        var half = self._half
        var mark = self._mark
        if stage == 0 and rows <= SIZING_EARLY * 2:
            mark = -1

        for j in range(count):
            var i = base + j
            var seen = rank + j

            if seen == mark:
                if stage == 0:
                    half = found
                    mark = SIZING_EARLY
                    stage = 1
                elif stage == 1:
                    self._count = found
                    self._reserve(
                        min(
                            project_groups(found, half, seen, rows),
                            found * EARLY_JUMP,
                        )
                    )
                    slots = self._slots.bitcast[DType.uint64]()
                    mask = self._mask
                    capacity = self._capacity
                    mark = SIZING_LATE // 2
                    stage = 2
                    if rows <= SIZING_LATE:
                        mark = -1
                elif stage == 2:
                    half = found
                    mark = SIZING_LATE
                    stage = 3
                else:
                    self._count = found
                    self._reserve(project_groups(found, half, seen, rows))
                    slots = self._slots.bitcast[DType.uint64]()
                    mask = self._mask
                    capacity = self._capacity
                    mark = -1

            if has_null and not validity.get(i):
                out.unsafe_offset(i).unsafe_write(UInt32(0))
                continue

            if (found + 1) * 2 > capacity:
                self._count = found
                self._grow()
                slots = self._slots.bitcast[DType.uint64]()
                mask = self._mask
                capacity = self._capacity

            if j + PROBE_LOOKAHEAD < count:
                var ahead = (
                    hash.unsafe_offset(j + PROBE_LOOKAHEAD).unsafe_load() & mask
                )
                prefetch[PrefetchOptions().for_read().high_locality()](
                    slots.unsafe_offset(Int(ahead) * SLOT_WORDS)
                )

            var wanted = hash.unsafe_offset(j).unsafe_load()
            var at = wanted & mask
            while True:
                var slot = Int(at) * SLOT_WORDS
                var ordinal = slots.unsafe_offset(slot + 1).unsafe_load()
                if ordinal == 0:
                    slots.unsafe_offset(slot).unsafe_write(wanted)
                    slots.unsafe_offset(slot + 1).unsafe_write(
                        UInt64(found + 1)
                    )
                    out.unsafe_offset(i).unsafe_write(UInt32(found + offset))
                    firsts.append(i)
                    found += 1
                    break
                if slots.unsafe_offset(slot).unsafe_load() == wanted:
                    out.unsafe_offset(i).unsafe_write(
                        UInt32(Int(ordinal) - 1 + offset)
                    )
                    break
                at = (at + 1) & mask

        self._count = found
        self._stage = stage
        self._half = half
        self._mark = mark

    def probe(
        self,
        hashes: Buffer,
        validity: Bitmap,
        has_null: Bool,
        base: Int,
        count: Int,
        out_at: Int,
        miss: UInt32,
        mut codes: Array[DType.uint32],
    ):
        """Looks a chunk of a column's keys up without inserting any of them.

        `build`'s read only twin, and the reason it exists is the join. A join
        does not want both sides in one table. It wants the smaller side's keys
        in a table and then one question asked of that table per row of the
        larger side, and the answer to a question a table has never seen is "not
        here" rather than a new group.

        Being read only takes out more than the insert. There is no growth
        check, no sizing schedule and no `firsts` to append to, so the loop is a
        load, a compare and a branch, and every worker can run it against the
        same table at once because none of them writes to it.

        Chunked for the reason `build` is chunked: the caller hashes a few
        thousand rows into a small buffer and probes them while they are still in
        cache. The prefetch is the same one and pays for the same reason, which
        is a table larger than the cache and an address that is known eight rows
        early.

        Args:
            hashes: Hashes for this chunk, indexed from zero, from `hash_chunk`.
                Hashed with this table's seed, or nothing matches.
            validity: The probe column's validity bitmap, indexed by absolute
                row. Read only when `has_null`.
            has_null: Whether the probe column has any nulls at all.
            base: The absolute row index this chunk starts at.
            count: How many rows are in this chunk.
            out_at: Added to the row index to get where in `codes` it goes. A
                join writes both sides into one list, so the side that is not
                first starts partway along it, and `build` cannot do the same
                because its row index is also its index into the validity bitmap.
            miss: The ordinal written for a row whose key is not in the table,
                and for a null row. A caller reserves one past the last real
                ordinal for it, so that a miss reads as a group nothing was ever
                put into rather than as a value needing a branch.
            codes: Where the per-row ordinals go, indexed by absolute row.
        """
        var hash = hashes.bitcast[DType.uint64]()
        var out = codes.unsafe_ptr()
        var slots = self._slots.bitcast[DType.uint64]()
        var mask = self._mask

        for j in range(count):
            var i = base + j
            var to = out_at + i
            if has_null and not validity.get(i):
                out.unsafe_offset(to).unsafe_write(miss)
                continue

            if j + PROBE_LOOKAHEAD < count:
                var ahead = (
                    hash.unsafe_offset(j + PROBE_LOOKAHEAD).unsafe_load() & mask
                )
                prefetch[PrefetchOptions().for_read().high_locality()](
                    slots.unsafe_offset(Int(ahead) * SLOT_WORDS)
                )

            var wanted = hash.unsafe_offset(j).unsafe_load()
            var at = wanted & mask
            while True:
                var slot = Int(at) * SLOT_WORDS
                var ordinal = slots.unsafe_offset(slot + 1).unsafe_load()
                if ordinal == 0:
                    out.unsafe_offset(to).unsafe_write(miss)
                    break
                if slots.unsafe_offset(slot).unsafe_load() == wanted:
                    out.unsafe_offset(to).unsafe_write(UInt32(Int(ordinal) - 1))
                    break
                at = (at + 1) & mask

    def build_strings(
        mut self,
        hashes: Buffer,
        col: StringArray,
        has_null: Bool,
        base: Int,
        rank: Int,
        count: Int,
        rows: Int,
        offset: Int,
        mut codes: Array[DType.uint32],
        mut firsts: List[Int],
    ):
        """Inserts a chunk of a string column's keys in one call.

        `build` with the key comparison put back. A hash match is a candidate
        rather than an answer, so the row is compared against the row that first
        produced that ordinal, and a mismatch keeps probing exactly as an
        occupied slot with a different hash does.

        The comparison is nearly free in the usual case. `element_equals` settles
        two elements on their length and their first four bytes without reading
        either payload, and the row it compares against was written when the
        group was created and so tends to already be in cache on a low
        cardinality column. What it costs is a load of the other row's view,
        which on a column where every row is its own group is a cache miss per
        row and is the price of being right.

        It is worth saying how rarely the comparison changes the answer. Two
        different strings landing on the same 64 bits happens about once in every
        two hundred thousand columns of ten million distinct keys, so a version
        of this that skipped the check would pass every test anybody wrote and
        would merge two groups in production some months later. That is the
        reason it is here rather than a measurement.

        The sizing schedule is shared with `build`, so a column built by one and
        then the other would size itself correctly, though nothing does that.

        Args:
            hashes: Hashes for this chunk, indexed from zero, from
                `hash_strings_chunk`.
            col: The column, needed for the comparison. Indexed by absolute row.
            has_null: Whether the column has any nulls at all.
            base: The absolute row index this chunk starts at.
            rank: How many rows this table has already been given.
            count: How many rows are in this chunk.
            rows: How many rows this table will be given in total.
            offset: Added to every ordinal written to `codes`.
            codes: Where the per-row ordinals go, indexed by absolute row.
            firsts: Appended with the absolute row index of every key that was
                new, in ordinal order. This build reads it as well as writing it,
                because it is where the key to compare against lives.
        """
        var hash = hashes.bitcast[DType.uint64]()
        var out = codes.unsafe_ptr()
        var slots = self._slots.bitcast[DType.uint64]()
        var mask = self._mask
        var capacity = self._capacity
        var found = self._count

        var stage = self._stage
        var half = self._half
        var mark = self._mark
        if stage == 0 and rows <= SIZING_EARLY * 2:
            mark = -1

        for j in range(count):
            var i = base + j
            var seen = rank + j

            if seen == mark:
                if stage == 0:
                    half = found
                    mark = SIZING_EARLY
                    stage = 1
                elif stage == 1:
                    self._count = found
                    self._reserve(
                        min(
                            project_groups(found, half, seen, rows),
                            found * EARLY_JUMP,
                        )
                    )
                    slots = self._slots.bitcast[DType.uint64]()
                    mask = self._mask
                    capacity = self._capacity
                    mark = SIZING_LATE // 2
                    stage = 2
                    if rows <= SIZING_LATE:
                        mark = -1
                elif stage == 2:
                    half = found
                    mark = SIZING_LATE
                    stage = 3
                else:
                    self._count = found
                    self._reserve(project_groups(found, half, seen, rows))
                    slots = self._slots.bitcast[DType.uint64]()
                    mask = self._mask
                    capacity = self._capacity
                    mark = -1

            if has_null and not col.is_valid(i):
                out.unsafe_offset(i).unsafe_write(UInt32(0))
                continue

            if (found + 1) * 2 > capacity:
                self._count = found
                self._grow()
                slots = self._slots.bitcast[DType.uint64]()
                mask = self._mask
                capacity = self._capacity

            if j + PROBE_LOOKAHEAD < count:
                var ahead = (
                    hash.unsafe_offset(j + PROBE_LOOKAHEAD).unsafe_load() & mask
                )
                prefetch[PrefetchOptions().for_read().high_locality()](
                    slots.unsafe_offset(Int(ahead) * SLOT_WORDS)
                )

            var wanted = hash.unsafe_offset(j).unsafe_load()
            var at = wanted & mask
            while True:
                var slot = Int(at) * SLOT_WORDS
                var ordinal = slots.unsafe_offset(slot + 1).unsafe_load()
                if ordinal == 0:
                    slots.unsafe_offset(slot).unsafe_write(wanted)
                    slots.unsafe_offset(slot + 1).unsafe_write(
                        UInt64(found + 1)
                    )
                    out.unsafe_offset(i).unsafe_write(UInt32(found + offset))
                    firsts.append(i)
                    found += 1
                    break
                if slots.unsafe_offset(slot).unsafe_load() == wanted:
                    if col.element_equals(i, firsts[Int(ordinal) - 1]):
                        out.unsafe_offset(i).unsafe_write(
                            UInt32(Int(ordinal) - 1 + offset)
                        )
                        break
                at = (at + 1) & mask

        self._count = found
        self._stage = stage
        self._half = half
        self._mark = mark

    def keys_by_ordinal(self, mut out: Buffer, at: Int):
        """Writes every key this table holds out, indexed by its ordinal.

        A parallel build ends with one table per worker and a merge that has to
        put their keys into a single table. The keys are already here, as the
        hashes that are what this table calls a key, so the merge can probe
        without hashing a single row again. All it needs is them in ordinal
        order, which the slots are not in, and this is the pass that fixes that.
        It costs a scan of the slots, which is a scan of twice the group count,
        and it replaces hashing the whole group count over again.

        Args:
            out: Where the hashes go. Must hold `at + len(self)` of them.
            at: The index the first ordinal writes to, so that several tables can
                fill one buffer back to back.
        """
        var slots = self._slots.bitcast[DType.uint64]()
        var dest = out.bitcast[DType.uint64]()
        for slot in range(self._capacity):
            var word = slot * SLOT_WORDS
            var ordinal = slots.unsafe_offset(word + 1).unsafe_load()
            if ordinal == 0:
                continue
            dest.unsafe_offset(at + Int(ordinal) - 1).unsafe_write(
                slots.unsafe_offset(word).unsafe_load()
            )

    def _reserve(mut self, groups: Int):
        """Grows the table to hold a group count without further growth.

        Never shrinks. A sizing estimate that came in below what the table is
        already holding is an estimate that arrived too late to be useful, and
        rehashing downwards to act on it would cost more than the sparseness it
        was trying to fix.

        Args:
            groups: The group count to make room for.
        """
        var capacity = next_power_of_two(groups * 2)
        if capacity <= self._capacity:
            return
        self._rehash_into(capacity)

    def _grow(mut self):
        """Doubles the table."""
        self._rehash_into(self._capacity * 2)

    def _rehash_into(mut self, capacity: Int):
        """Moves every live key into a fresh table of a given size.

        Reinsertion needs no hashing. The stored value is the hash, so the new
        slot is a mask away, which is the second thing storing hashes instead of
        keys buys and the reason a growth costs about what a memcpy costs.

        Args:
            capacity: The new slot count. Must be a power of two, and must be
                large enough to hold what the table already holds.
        """
        var grown = Buffer(capacity * SLOT_WORDS * 8)
        var mask = UInt64(capacity - 1)

        var old = self._slots.bitcast[DType.uint64]()
        var new = grown.bitcast[DType.uint64]()

        for slot in range(self._capacity):
            var at = slot * SLOT_WORDS
            var ordinal = old.unsafe_offset(at + 1).unsafe_load()
            if ordinal == 0:
                continue
            var hash = old.unsafe_offset(at).unsafe_load()
            var i = hash & mask
            while True:
                var to = Int(i) * SLOT_WORDS
                if new.unsafe_offset(to + 1).unsafe_load() == 0:
                    new.unsafe_offset(to).unsafe_write(hash)
                    new.unsafe_offset(to + 1).unsafe_write(ordinal)
                    break
                i = (i + 1) & mask

        self._slots = grown^
        self._capacity = capacity
        self._mask = mask
