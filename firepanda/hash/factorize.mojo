"""Turning a column into group ordinals, by whichever route is cheapest.

This is the operation a group by is built on. Every row gets an integer naming
the group it belongs to, a row per group comes back so the ordinal can be turned
back into a key, and everything downstream aggregates into a flat array indexed
by that integer. pandas calls this `factorize` and the name is worth keeping.

There are two routes and the interesting one is that the fast route is not the
hash table.

**Direct indexing.** A column of integers whose values span a small range does
not need to be hashed at all. Allocate one slot per possible value, index it,
done. That is a group by on a year, on a category code, on a small identifier, on
anything that came out of a previous factorize, and it is a large fraction of the
group bys anyone actually runs. Both passes over the column are sequential.

**The hash table.** Everything else. Floats, wide-ranging integers, anything
where a direct table would be larger than the column it is replacing.

The switch is a memory bound rather than a guess about cardinality. A direct
table costs four bytes per possible value and a hash slot costs sixteen, and the
hash table needs its own set of slots for every worker, so a direct table stays
the smaller of the two well past the point where it stops fitting in any cache.
The rule is a slot for every four rows, which is one byte a row against the four
that the ordinals cost, and `DIRECT_LIMIT` is the floor under it for columns too
short for a share of their height to be much of a bound.

`factorize_dense` is the same route for a caller who built the values and can say
what range they are in. It accepts a wider table than the scan will, because a
constructed range comes with an idea of how much of itself is occupied, and it
skips the scan for the same reason.

Text has a third route, `factorize_strings`, and it is the hash table with the
key comparison put back, because a string does not fit in the 64 bits the table
stores.

Neither route hands back the key values. All three record the row that introduced
each group, which they are standing on anyway, and `Factorized.keys` turns that
into a column of keys for whoever wants one. Nobody in the library does: a group
by gathers its key columns by those rows, with the take kernel it already has and
for every key at once, and a join reads the ordinals and nothing else. Building
the keys eagerly was a random read per group and two copies of the result, which
on a join of ten million distinct keys is eighty megabytes written twice and then
dropped.

The hash route splits a long column across cores. Each worker builds its own
table over a contiguous slice, and a merge afterwards renumbers what they found
into one set of ordinals. That merge runs on every core too: equal keys hash
equally, so bucketing the workers' groups on the top bits of their hash gives
each bucket a key range no other bucket can hold, and the buckets have nothing to
agree about. `_projected_groups` guesses the key count before either route has
run, `_parallel_workers` turns that into a worker count, and
`_factorize_hashed_parallel` is where the argument for the shape lives. Text
takes the same route through `_factorize_strings_parallel`, which differs only in
that its merge compares two rows where the numeric one trusts a hash.

That shape stops paying when the groups outnumber what a worker's table can hold
in cache, because a contiguous slice of such a column contains very nearly every
key the whole column has and so every worker builds very nearly the whole answer.
`_crowded` is the question, and when it says yes the rows are cut by hash instead
of by position. Partitions of a hash cannot share a key, so a partition's table
holds only its own share of the groups and there is no merge at the end at all.
What that costs is first-appearance order, which the slice route got out of the
shape of its own work, and `_rank_by_first_row` buys it back by numbering the
groups by the row that introduced them. `_factorize_hashed_partitioned` is the
numeric one and `_factorize_strings_partitioned` is the text one, and the only
difference between them is that a string key has to be compared, so its build is
told which row each of its positions came from.

Nulls get a group. A null is a value that rows can share, pandas with
`dropna=False` gives it a group, and dropping it later is a filter on one ordinal
while inventing it later would be another pass over the column. When a column has
any nulls the null group is ordinal zero and the non-null keys follow in
first-appearance order. Fixing it at zero rather than at the position of the
first null is what keeps the ordinals the table hands out and the ordinals that
come back a constant apart, which is the difference between one addition per row
and a second bookkeeping structure.
"""

from std.math import exp
from std.sys.info import simd_width_of, size_of

from firepanda.array.array import Array
from firepanda.array.strings import StringArray
from firepanda.array.strview import StringView
from firepanda.buffer.buffer import Buffer
from firepanda.exec import parallel_for, worker_count
from firepanda.kernel.accum import highest, lowest
from firepanda.kernel.agg import BLOCK
from firepanda.kernel.select import take_rows

from .function import DEFAULT_SEED, hash_chunk, hash_strings_chunk
from .table import HashTable

comptime DIRECT_LIMIT = 1 << 16
"""Smallest integer range that always skips the hash table.

65536 slots is 256 KB of uint32, which is the last size that behaves like cache
rather than like memory on the machines this targets, so a table this wide is
worth taking whatever the column costs to hash.

This used to be a ceiling as well as a floor, on the argument that a wider table
stops being a lookup and starts being the same random access the hash table was
going to do anyway. The second half of that is true and the conclusion did not
follow: the random access is the same and everything around it is cheaper, so a
much wider direct table still beats the hash. `DIRECT_SHARE` is the bound
`factorize` uses on a long column now, and this is what a column too short for
that bound to mean anything gets instead.
"""

comptime DIRECT_SHARE = 4
"""Rows per slot of the widest direct table `factorize` will accept.

A slot is four bytes, so a quarter of a slot per row is one byte per row, which
is a quarter of what the ordinals coming out of the factorize cost. A table that
small cannot be a surprise in a query's memory, which is the point: the
measurement in `tools/probes/direct_ceiling.mojo` says the direct route is still
ahead a good way past this, so the number is chosen to be uncontroversial rather
than to be the last place the route wins.
"""

comptime SAFE_RANGE_BITS = 40
"""Values outside plus or minus 2^40 do not take the direct route.

Not because they could not, but because computing `high - low` for them can
overflow, and a range check that overflows is worse than a range check that
declines. Anything with a span small enough to qualify is inside this window
anyway; the check is about the arithmetic, not about the data.
"""

comptime CHUNK_ROWS = 1 << 10
"""Rows hashed before any of them are probed.

A thousand rows is eight kilobytes, which stays in L1 alongside the table itself
on a low cardinality column. Four thousand measured slower on every shape here,
which is what you would expect from a buffer that no longer fits. Smaller chunks
start losing the last `PROBE_LOOKAHEAD` rows of each one to a prefetch that has
nowhere to run.
"""

comptime PARALLEL_ROWS = 1 << 22
"""Rows below which a factorize of a fixed width column stays on one thread.

The parallel route costs a merge over every group every worker found and a
second pass over the column to renumber them, and neither of those exists on the
serial one. The serial route for a number is close to one pass over memory, so
there is a lot of column to get through before threads make that back.

It used to be `1 << 17`, which happened to be exactly `MORSEL_ROWS`. A chunk is
one morsel and the gate is `>=`, so every factorize the streaming engine did
landed one row over the line and took the parallel route on four slices, which
is the fewest workers it will ever run on. Measured on an idle i9-13900K,
`group/ordinals_one_key` on a thousand groups: 131071 rows is 0.588 ns a row and
131072 rows was 2.957, so one row past a chunk cost 5.0x.

Closing that cliff turned out not to be the same question as putting the line in
the right place. A build that never takes the parallel route, measured against
one that always does, puts the crossover four to five doublings further out than
where the constant was. On a thousand groups, serial against parallel in ns a
row: 262144 is 0.598 against 2.277, a million is 0.625 against 1.611, three
million is 1.432 against 1.685, and four million is 0.947 against 0.816. So
serial wins by 3.8x at a quarter million and the two cross a little under four
million. A hundred thousand groups, which goes through the hash table rather
than the direct one, crosses in the same place: 9.112 against 11.043 at a
quarter million and 1.237 against 1.140 at four million.

That is why this is `1 << 22` rather than something near a chunk. Everything
below it was being split into slices that could not pay for the merge, and the
band from a quarter million to three million rows is a size real queries land on
constantly.

The number is from one machine and the shape of the answer is what carries over
rather than the value. What decides it is the ratio between what a row costs on
one thread and what the merge costs, so a slower core or a shorter key moves it
in, and the honest version of this constant is a cost model over the row count
and the group count rather than a row count alone. `PARALLEL_STRING_ROWS` is the
same rule applied to a key that costs thirty times as much per row, and it comes
out thirty times lower, which is the evidence that the ratio is the thing.
"""

comptime PARALLEL_STRING_ROWS = 1 << 18
"""Rows below which a factorize of a text column stays on one thread.

Text gets its own line because it is nowhere near the numeric one. A string key
is hashed over its bytes and settled with a comparison over its bytes, which
measures around 19 ns a row against about 0.6 for an int64, so the fixed cost of
a split is paid back roughly thirty times sooner and the crossover sits roughly
thirty times lower. Using one constant for both meant one of the two was always
in the wrong place.

Measured the same way on the same machine, parallel against serial in ns a row
on `text/group_distinct`: at 131072 rows they are level at 17.4 against 17.3, at
262144 rows parallel wins at 18.893 against 23.746, and at a million rows it
wins at 12.698 against 24.083. `text/group_medium` runs 1.46x and then 1.79x
over the same pair. So text wants the split almost as early as it can get it,
and the only reason this is not lower still is the one thing the numeric
constant also has to respect, which is that a chunk sized factorize must not be
split. Level is not worth a fork and a join.
"""

comptime PARALLEL_MIN_SLICE = 1 << 15
"""Smallest slice a worker is given.

Two workers on a column that fits in L2 spend more time in the merge than they
save in the build. Capping the worker count by this rather than dropping to one
thread means a column just over `PARALLEL_ROWS` gets the few workers it can keep
busy instead of all of them.

This is a floor on the slice and the two `PARALLEL_*_ROWS` constants are floors
on the column, and they are not the same rule. A column can clear its threshold
and still be cut into fewer pieces than there are cores, which is what this is
for. What it must not do is clear the threshold and be cut into pieces smaller
than this, and since both thresholds are many multiples of it, neither can.

`PARALLEL_ROWS` is 128 of these and `PARALLEL_STRING_ROWS` is 8. That gap is the
per row cost of the key rather than anything about slicing, so it belongs to
those two constants and not to this one, which is why this one is shared.
"""

comptime TABLE_BYTES_PER_GROUP = 32
"""Bytes of hash table a group takes up while it is being built.

A slot is sixteen bytes, a hash and an ordinal, and the table doubles when it
passes half full, so a group has two slots standing behind it by the time the
build is over.
"""

comptime SHARED_CACHE_BYTES = 32 * 1024 * 1024
"""Last level cache the workers' tables have to share.

Thirty two megabytes is the reference machine's L3 rounded down. There is no way
to ask the machine for this at runtime from here, and it is a floor rather than a
fit: reading it low costs a few workers on a column whose curve is flat across
that range anyway, while reading it high puts every worker's probes in memory,
which is the thing this number exists to avoid.
"""

comptime CROWDED_SHARE = 2
"""Fraction of the cores taken when the tables cannot all fit in cache.

Past the point where the tables stop fitting, every extra worker is another
table competing for the same cache and the probes go to memory. More workers
still help there, because more of those misses are in flight at once, but not by
as much, and past half the cores the memory system is the limit and they start
getting in each other's way. Half was measured against a hundred thousand and a
million groups, in both integers and text, and it is within about a tenth of the
best worker count on all four.
"""

comptime PARALLEL_SAMPLE = 1 << 16
"""Most rows the group count is estimated from.

Two counts come out of it, one at the halfway mark and one at the end, and
`_estimate_groups` reads the ratio between them. Sixty five thousand rows is
enough for that ratio to be steady on every shape measured here and small enough
that building it is under a percent of the column it is deciding about. On a
shorter column the sample is a sixteenth of it instead, so the cost of the
decision stays proportional to the work the decision is about.
"""


struct Factorized[dt: DType](Movable):
    """A column rewritten as group ordinals, plus a row per group they name."""

    var codes: Array[DType.uint32]
    """One group ordinal per row of the input, in `[0, count())`.

    Every ordinal in that range belongs to at least one row, on both routes. The
    hashed one only makes an ordinal because a row asked for one, and the direct
    one indexes its table by value but hands out an ordinal only on first sight,
    so the gaps in the table are not gaps in the ordinals. A group by relies on that
    to skip a renumbering pass,
    `test_no_shape_of_column_factorizes_to_a_sparse_ordinal`
    holds it, and a route added later that does not hold it has to say so.
    """

    var firsts: List[Int]
    """For each non-null group, the first row that had it, in ordinal order.

    All three routes have this row already, because each of them recognizes a new
    group by the row that introduced it, so recording it costs one append per
    group rather than a pass. The frame layer's group by wants exactly this and
    used to rebuild it with a pass over every row.

    The null group is not in here, and the offset that leaves is why. A column
    with nulls puts them at ordinal zero regardless of which row the first one is
    on, so the rows in here name groups one and up and are not in first
    appearance order over the whole column. `KeyCodes.knows_rows` catches that by
    length and sends the caller back to the renumbering pass.
    """

    var null_group: Int
    """The ordinal the nulls got, which is zero, or -1 if the column had none."""

    def __init__(
        out self,
        var codes: Array[DType.uint32],
        var firsts: List[Int],
        null_group: Int,
    ):
        """Constructs a result.

        Args:
            codes: The per-row ordinals.
            firsts: A representative row per non-null group.
            null_group: The ordinal for nulls, or -1.
        """
        self.codes = codes^
        self.firsts = firsts^
        self.null_group = null_group

    def count(self) -> Int:
        """Returns the number of groups.

        Returns:
            The group count, including the null group if there is one.
        """
        return len(self.firsts) + (1 if self.null_group >= 0 else 0)

    def keys(self, col: Array[Self.dt]) raises -> Array[Self.dt]:
        """Builds the distinct keys, in ordinal order.

        Index the result by an ordinal from `codes` to get the value back. Entry
        zero is the null group when there is one, and it is null.

        This is a gather by `firsts` and it is not done unless it is asked for.
        Every group's row is a row this factorize already stood on, so recording
        it cost an append per group, while reading the value out of it costs a
        random load per group and a column to put it in. The library's own
        callers want the rows rather than the values: a group by gathers all of
        its key columns by them at once, and a join wants neither.

        Args:
            col: The column this factorization came from.

        Returns:
            A column of `count()` rows, the key of each group in ordinal order.

        Raises:
            If the gather fails, which for indices this produced it will not.
        """
        var picks = List[Int](capacity=self.count())
        if self.null_group >= 0:
            # A negative index gathers a null, which is what the null group's key
            # is, so the offset that `firsts` leaves at the front fills itself in.
            picks.append(-1)
        for at in range(len(self.firsts)):
            picks.append(self.firsts[at])
        return take_rows(col, picks)

    def into_codes(deinit self) -> Array[DType.uint32]:
        """Gives up the ordinals without copying them, dropping the rest.

        A caller that groups on several columns wants the ordinals and has no use
        for the rows, and Mojo will not let it reach in and move `codes` out from
        under a struct that still owns `firsts`. `deinit` says the rest is being
        torn down, which is what makes the move legal. Same arrangement as
        `Series.into_values`.

        Returns:
            The per-row ordinals.
        """
        return self.codes^

    def into_parts(
        deinit self, mut codes: Array[DType.uint32], mut firsts: List[Int]
    ):
        """Gives up the ordinals and the representative rows together.

        Two out parameters rather than a returned pair, for the reason
        `FactorizedStrings.into_parts` gives.

        Args:
            codes: Overwritten with the per-row ordinals.
            firsts: Overwritten with a representative row per non-null group.
        """
        codes = self.codes^
        firsts = self.firsts^


def factorize[
    dt: DType
](col: Array[dt], seed: UInt64 = DEFAULT_SEED) raises -> Factorized[dt]:
    """Assigns a group ordinal to every row.

    On a long column the direct table is offered a slot for every four rows,
    which is far wider than `DIRECT_LIMIT`. The reason is what the table is being
    compared against rather than what fits in cache. A direct slot is four bytes
    and a hash slot is sixteen, and the hash table needs its own set of them for
    every worker, so a direct table of this width is smaller than the thing it
    replaces even when most of it is empty.

    Measured on ten million rows of int64 on the i9-13900K, direct against
    hashed, with the span fully occupied: a hundred thousand slots is 12 ms
    against 36, a million is 22 against 99, and two and a half million, which is
    the ceiling, is 36 against 129.

    The other shape is a wide span with hardly anything in it, which is the table
    mostly full of holes `DIRECT_LIMIT` was written to decline, and there the two
    routes are close. A span of a million holding a thousand values is 11 ms
    direct against 7 hashed, a span of five million holding a hundred is 12
    against 13, and a span of a million holding a hundred thousand is 17 against
    33. What the sparse cases cost the direct route is the scan and the ordinals,
    not the table: a table with few values in it only ever touches a few slots
    and they stay in cache however large it is. So the penalty is a fixed few
    milliseconds that does not grow with the span, while the gain on the dense
    cases does.

    That argument would carry a ceiling of a slot a row. It is set at a quarter
    of that anyway, because a table that is one byte a row cannot be a surprise
    in anybody's memory budget, and because everything above the ceiling is
    something the hash route is already handling correctly.

    A column of fewer than four times `DIRECT_LIMIT` rows keeps the old bound,
    which was the smaller of that limit and its own height.

    What a wider bound costs is the scan. `direct_plan` stops as soon as no later
    row could bring the span back under the ceiling, and raising the ceiling
    moves that point later, so a column whose span lands just above the bound is
    now scanned further before being hashed anyway. That scan is sequential and
    the route it is deciding between costs tens of milliseconds, so it is a
    small price for the case above, where the same scan finds a span the table
    can take.

    Args:
        col: The column to group.
        seed: The per-query hash seed. Ignored on the direct route, which does
            not hash.

    Parameters:
        dt: The column's dtype.

    Returns:
        The ordinals, a representative row per group, and which ordinal the
        nulls got.
    """
    comptime if dt.is_integral():
        var ceiling = len(col) // DIRECT_SHARE
        if ceiling < DIRECT_LIMIT:
            ceiling = min(DIRECT_LIMIT, len(col))
        var plan = direct_plan[dt](col, ceiling)
        if plan.span >= 0:
            return _factorize_direct[dt](col, plan.span, plan.base)
    return _factorize_hashed[dt](col, seed)


def factorize_dense[
    dt: DType
](col: Array[dt], span: Int, seed: UInt64 = DEFAULT_SEED) raises -> Factorized[
    dt
]:
    """Factorizes a column whose values the caller knows are in `[0, span)`.

    `factorize` has to learn the range by scanning, and what a scan finds is a
    bound on data the library did not construct. A span of ten million says
    nothing about whether ten values occupy it or ten million do, and a direct
    table over a span that is mostly empty is the same random access the hash
    was going to be with worse density and a larger allocation. That is what
    `DIRECT_LIMIT` declines, and against an unknown column it is right to.

    A caller that built the values knows more than the scan can. The packed key
    of a group by on several columns is the case this exists for: its span is
    the product of the key group counts, its occupancy is the number of key
    tuples actually present, and the shapes where the span is large are exactly
    the shapes where a large fraction of it is occupied. So the bound here is
    the table against the column instead of the table against the cache, which
    keeps the direct route from ever costing more memory than four bytes a row,
    and the scan is skipped because its answer was already known.

    Args:
        col: The column, with every value in `[0, span)`.
        span: The width of that range.
        seed: The per-query hash seed, used only if the table is declined.

    Parameters:
        dt: The column's dtype.

    Returns:
        The ordinals, a representative row per group, and which ordinal the
        nulls got.
    """
    comptime if dt.is_integral():
        if span >= 0 and (span <= DIRECT_LIMIT or span <= len(col)):
            return _factorize_direct[dt](col, span, Scalar[dt](0))
    return _factorize_hashed[dt](col, seed)


@fieldwise_init
struct DirectPlan[dt: DType](Copyable, Movable):
    """Whether the direct route applies to a column, and what it needs."""

    var span: Int
    """The number of slots a direct table needs, or -1 for the hash route."""

    var base: Scalar[Self.dt]
    """The value that indexes slot zero, which is the column's minimum."""


def direct_plan[dt: DType](col: Array[dt], ceiling: Int) -> DirectPlan[dt]:
    """Decides whether the direct route applies, in one pass that can stop early.

    This used to be a `min_of` and a `max_of` and then a test on the two answers,
    which is two full passes over the column before the decision. Both of them
    are wasted on any column that ends up hashed, and on wide-ranging data most
    of the first one is wasted too, because the span passes the point of no
    return in the first few thousand rows and nothing later can bring it back.

    So the two reductions are fused and the test is moved inside them. The bounds
    only ever widen, so once the span is too large for a direct table, or once a
    value is outside the window the subtraction is safe in, no later row can
    change the answer and the scan returns. On a ten million row column of
    scattered keys that is eight milliseconds down to under one.

    The widest span worth a table is the caller's to say, because it depends on
    what the table is for and on how many rows are going to be pushed through it.
    Every caller today asks for some share of a slot per row of the input it is
    building over, which is a table smaller than the hash table it replaces, but
    they arrive at that number from different row counts and they do not agree on
    the share. A join's build side takes a slot a row, because every probe row
    that misses the table is a hash saved. A factorize takes a quarter of that,
    because it has one pass to pay the table off against rather than two. So the
    number comes in rather than being decided here.

    Args:
        col: The column.
        ceiling: The widest span to accept. A wider one returns the hash plan.

    Parameters:
        dt: The column's dtype, which the caller has already established is
            integral.

    Returns:
        The plan, whose span is -1 when the column should be hashed instead.
    """
    var n = len(col)
    if n == 0:
        return DirectPlan[dt](-1, Scalar[dt](0))

    comptime width = simd_width_of[dt]()
    comptime safe = Scalar[dt](1 << SAFE_RANGE_BITS)

    var ptr = col.unsafe_ptr()
    var low = highest[dt]()
    var high = lowest[dt]()
    var seen = False

    for w in range(col.data.validity.word_count()):
        var word = col.data.validity.unsafe_word(w)
        if word == 0:
            continue

        var start = w * BLOCK
        var last = start + BLOCK
        if last > n:
            last = n

        if word == UInt64.MAX and last == start + BLOCK:
            seen = True
            var mins = SIMD[dt, width](highest[dt]())
            var maxes = SIMD[dt, width](lowest[dt]())
            var i = start
            while i + width <= last:
                var chunk = ptr.unsafe_offset(i).unsafe_load[width=width]()
                mins = min(mins, chunk)
                maxes = max(maxes, chunk)
                i += width
            var block_low = mins.reduce_min()
            var block_high = maxes.reduce_max()
            while i < last:
                var value = ptr.unsafe_offset(i).unsafe_load()
                if value < block_low:
                    block_low = value
                if value > block_high:
                    block_high = value
                i += 1
            if block_low < low:
                low = block_low
            if block_high > high:
                high = block_high
        else:
            for i in range(start, last):
                if (word >> UInt64(i - start)) & 1 == 0:
                    continue
                seen = True
                var value = ptr.unsafe_offset(i).unsafe_load()
                if value < low:
                    low = value
                if value > high:
                    high = value

        if seen:
            comptime if size_of[dt]() > 4:
                if high > safe:
                    return DirectPlan[dt](-1, Scalar[dt](0))
                comptime if dt.is_signed():
                    if low < -safe:
                        return DirectPlan[dt](-1, Scalar[dt](0))
            if Int(high) - Int(low) + 1 > ceiling:
                return DirectPlan[dt](-1, Scalar[dt](0))

    if not seen:
        # Every row is null. One group, no keys to spread out, and the direct
        # route allocates a single slot it never reads.
        return DirectPlan[dt](1, Scalar[dt](0))

    return DirectPlan[dt](Int(high) - Int(low) + 1, low)


comptime DIRECT_MERGE_BYTES = 256 * 1024
"""How much table the merge is allowed to walk for one value.

The merge reads one slot out of every worker's table at each value in the range,
and those slots are a table apart, so it touches one cache line per worker per
value and the footprint of that walk is the range times the worker count. Past
the point where it leaves a core's private cache every value in it is a miss,
and the measurement says that is what turns the split from a win into a loss: a
range of a hundred takes every core and runs in two thirds of the serial time, a
range of ten thousand wants six workers, and a range of sixty five thousand is
better off on one thread. A quarter of a megabyte is the number that picks all
three of those correctly.
"""

comptime DIRECT_MIN_WORKERS = 4
"""The fewest workers worth splitting a direct factorize across."""

comptime DIRECT_CLAIM_BLOCK = 1 << 12
"""Values one task of the direct merge settles.

The merge walks the value range rather than the rows, reading one slot per
worker at each value, so a block of four thousand values is the worker count
times sixteen kilobytes of reads. Small enough that the whole block's worth of
slots stays in cache while it is being walked, and large enough that a range at
`DIRECT_LIMIT` is sixteen tasks rather than sixteen thousand.
"""


def _factorize_direct[
    dt: DType
](col: Array[dt], span: Int, base: Scalar[dt]) raises -> Factorized[dt]:
    """Factorizes by indexing a table with the value itself.

    Args:
        col: The column.
        span: The table width, from `direct_plan`.
        base: The value that indexes slot zero, which is the column's minimum.

    Parameters:
        dt: The column's dtype.

    Returns:
        The factorization.
    """
    var n = len(col)
    if n >= PARALLEL_ROWS and span > 0:
        var affordable = DIRECT_MERGE_BYTES // (span * 4)
        var workers = min(worker_count(), n // PARALLEL_MIN_SLICE)
        if affordable < workers:
            workers = affordable
        # Four and not two. Two workers were slower than one thread on every
        # range measured, by about half again, because the split pays a second
        # read of the column and two cores are not enough to make that back.
        if workers >= DIRECT_MIN_WORKERS:
            return _factorize_direct_parallel[dt](col, span, base, workers)
    return _factorize_direct_serial[dt](col, span, base)


def _factorize_direct_serial[
    dt: DType
](col: Array[dt], span: Int, base: Scalar[dt]) -> Factorized[dt]:
    """Factorizes by indexing a table with the value itself, on one thread.

    Args:
        col: The column.
        span: The table width, from `direct_plan`.
        base: The value that indexes slot zero, which is the column's minimum.

    Parameters:
        dt: The column's dtype.

    Returns:
        The factorization.
    """
    var n = len(col)
    var values = col.unsafe_ptr()

    # Zero means unseen, so the ordinal is stored plus one and the table needs no
    # initialization pass. `Buffer` already handed back zeroed memory.
    var seen = Buffer(span * 4)
    var slots = seen.bitcast[DType.uint32]()

    var has_null = col.null_count() > 0
    var offset = 1 if has_null else 0
    # Every row below is written, the null ones with a zero, so the array does
    # not need the pass that zeroes it first.
    var codes = Array[DType.uint32](overwritten=n)
    var out = codes.unsafe_ptr()
    var firsts = List[Int]()
    var null_group = -1
    if has_null:
        null_group = 0

    for i in range(n):
        if has_null and not col.is_valid(i):
            out.unsafe_offset(i).unsafe_write(UInt32(0))
            continue

        var value = values.unsafe_offset(i).unsafe_load()
        var at = Int(value) - Int(base)
        var stored = slots.unsafe_offset(at).unsafe_load()
        if stored == 0:
            var assigned = len(firsts) + offset
            # The row that introduced the group, which this loop is standing on
            # anyway. The other two routes have the same row for the same reason,
            # and it is the only thing any of them records about the key.
            firsts.append(i)
            slots.unsafe_offset(at).unsafe_write(UInt32(assigned + 1))
            out.unsafe_offset(i).unsafe_write(UInt32(assigned))
        else:
            out.unsafe_offset(i).unsafe_write(stored - 1)

    return Factorized[dt](codes^, firsts^, null_group)


def _factorize_direct_parallel[
    dt: DType
](
    col: Array[dt], span: Int, base: Scalar[dt], workers: Int
) raises -> Factorized[dt]:
    """Factorizes by value with one direct table per core, then merges them.

    Same trade the hashed route makes and for the same reason: two rows in two
    slices can hold the same value, so the workers cannot share a table, and
    what they get instead is one each and a merge afterwards. What differs is
    that the merge here has no probing in it. A direct table is indexed by the
    value, so the workers' tables are already aligned with each other, and
    finding every worker that saw a value is reading the same slot out of each
    of them. The first worker that has it is the one that saw it first, because
    the slices are contiguous and in order.

    That walk is over the value range rather than over the rows or the groups,
    which is what makes it worth doing in parallel by value: no two tasks touch
    the same value, so no two of them can discover the same group. The final
    numbering is `_rank_entries`, exactly as the hashed merge uses it, over the
    same entry order, so the ordinals come out in first-appearance order and
    identical to the serial route's.

    The shape here is two passes over the column and not three. The obvious
    arrangement is that the first pass writes each row a local ordinal and a
    third pass rewrites those into the merged ones, which is what this did, and
    the measurement said two workers were slower than one thread. The direct
    loop is not short of arithmetic, it is short of memory bandwidth, so the
    extra read and write of a forty megabyte ordinal column cost more than a
    second core bought. So the first pass only discovers groups, the merge
    projects itself back down onto a single table indexed by value, and the
    second pass reads that table and writes each row once.

    Args:
        col: The column.
        span: The table width, from `direct_plan`.
        base: The value that indexes slot zero, which is the column's minimum.
        workers: How many slices to cut the column into. At least two.

    Parameters:
        dt: The column's dtype.

    Returns:
        The factorization.
    """
    var n = len(col)
    var has_null = col.null_count() > 0
    var offset = 1 if has_null else 0

    var bounds = List[Int](capacity=workers + 1)
    for w in range(workers):
        bounds.append(n * w // workers)
    bounds.append(n)

    # One table per worker, laid end to end so that the merge reads the same
    # value out of all of them by striding rather than by chasing a list of
    # allocations. Zero means unseen and the local ordinal is stored plus one,
    # which is what lets a zeroed buffer be a ready table.
    var tables = Buffer(workers * span * 4)
    var founds = List[List[Int]](capacity=workers)
    for _ in range(workers):
        founds.append(List[Int]())

    def discover(w: Int) raises {mut tables, mut founds, imm}:
        var slots = tables.bitcast[DType.uint32]().unsafe_offset(w * span)
        var values = col.unsafe_ptr()
        # The found rows are collected into a local list and handed over once at
        # the end rather than appended straight into the shared one. A list is
        # three words, so four of the workers' slots share a cache line, and
        # appending into them from four cores turns every new group into a line
        # the other three have to fetch again.
        var seen = List[Int]()
        for i in range(bounds[w], bounds[w + 1]):
            if has_null and not col.is_valid(i):
                continue
            var at = Int(values.unsafe_offset(i).unsafe_load()) - Int(base)
            if slots.unsafe_offset(at).unsafe_load() == 0:
                seen.append(i)
                slots.unsafe_offset(at).unsafe_write(UInt32(len(seen)))
        founds[w] = seen^

    parallel_for(discover, workers)

    var starts = List[Int](capacity=workers + 1)
    var total = 0
    for w in range(workers):
        starts.append(total)
        total += len(founds[w])
    starts.append(total)

    var reps = _flatten_reps(founds, starts, workers)
    var owners = Buffer(overwritten=total * 4)
    var marks = Buffer(total)
    # Which entry won each value, so that the merged ordinal can be put back on
    # the value it belongs to once the ranking has assigned one.
    var held = Buffer(overwritten=span * 4)
    var chunks = (span + DIRECT_CLAIM_BLOCK - 1) // DIRECT_CLAIM_BLOCK

    def claim(c: Int) raises {mut owners, mut marks, mut held, imm}:
        var slots = tables.bitcast[DType.uint32]()
        var own = owners.bitcast[DType.uint32]()
        var hold = held.bitcast[DType.uint32]()
        var mark = marks.unsafe_ptr()
        var stop = min((c + 1) * DIRECT_CLAIM_BLOCK, span)
        for v in range(c * DIRECT_CLAIM_BLOCK, stop):
            var winner = -1
            for w in range(workers):
                var stored = slots.unsafe_offset(w * span + v).unsafe_load()
                if stored == 0:
                    continue
                var entry = starts[w] + Int(stored) - 1
                if winner < 0:
                    winner = entry
                    mark.unsafe_offset(entry).unsafe_store(UInt8(1))
                own.unsafe_offset(entry).unsafe_write(UInt32(winner))
            hold.unsafe_offset(v).unsafe_write(UInt32(winner))

    parallel_for(claim, chunks)

    var map = Buffer(0)
    var firsts = List[Int]()
    _rank_entries(marks, owners, reps, total, offset).into_parts(map, firsts)
    var mapped = map.bitcast[DType.uint32]()

    # The merged ordinals collapsed back onto the value range. A value nothing
    # held keeps a zero, which no row can read because no row has that value.
    var settled = Buffer(overwritten=span * 4)

    def settle(c: Int) raises {mut settled, imm}:
        var hold = held.bitcast[DType.uint32]()
        var final = settled.bitcast[DType.uint32]()
        var stop = min((c + 1) * DIRECT_CLAIM_BLOCK, span)
        for v in range(c * DIRECT_CLAIM_BLOCK, stop):
            var winner = hold.unsafe_offset(v).unsafe_load()
            if winner == UInt32.MAX:
                final.unsafe_offset(v).unsafe_write(UInt32(0))
            else:
                final.unsafe_offset(v).unsafe_write(
                    mapped.unsafe_offset(Int(winner)).unsafe_load()
                )

    parallel_for(settle, chunks)

    var codes = Array[DType.uint32](overwritten=n)

    def assign(w: Int) raises {mut codes, imm}:
        var final = settled.bitcast[DType.uint32]()
        var values = col.unsafe_ptr()
        var out = codes.unsafe_ptr()
        for i in range(bounds[w], bounds[w + 1]):
            if has_null and not col.is_valid(i):
                out.unsafe_offset(i).unsafe_write(UInt32(0))
                continue
            var at = Int(values.unsafe_offset(i).unsafe_load()) - Int(base)
            out.unsafe_offset(i).unsafe_write(
                final.unsafe_offset(at).unsafe_load()
            )

    parallel_for(assign, workers)

    var null_group = -1
    if has_null:
        null_group = 0
    return Factorized[dt](codes^, firsts^, null_group)


def _factorize_hashed[
    dt: DType
](col: Array[dt], seed: UInt64) raises -> Factorized[dt]:
    """Factorizes through the hash table, on one thread or on several.

    Args:
        col: The column.
        seed: The per-query hash seed.

    Parameters:
        dt: The column's dtype.

    Returns:
        The factorization.
    """
    var n = len(col)
    if n >= PARALLEL_ROWS and worker_count() > 1:
        var groups = _projected_groups[dt](col, seed, n)
        var most = min(worker_count(), n // PARALLEL_MIN_SLICE)
        # A column whose tables would not fit in the shared cache is one where
        # the slice route is paying for the same group in every worker, and that
        # is the whole reason the partitioned route exists. Rows have to number
        # inside a uint32 there, because the partitions carry row indices rather
        # than the column itself.
        if (
            most >= 2
            and groups >= 1
            and _crowded(groups, most)
            and n <= Int(UInt32.MAX)
        ):
            return _factorize_hashed_partitioned[dt](col, seed, most)
        var workers = _parallel_workers(groups, n)
        if workers > 1:
            return _factorize_hashed_parallel[dt](col, seed, workers)
    return _factorize_hashed_serial[dt](col, seed)


comptime MERGE_BUCKETS_PER_WORKER = 4
"""Hash buckets the merge cuts the workers' groups into, per worker.

More buckets than workers, for two reasons. A bucket that draws more than its
share of the groups is then a quarter of a worker's turn rather than the whole
tail of the phase, and a bucket's table is a quarter of the size, which is what
keeps it in L1 while it is being built. Eight measured no better than four on
any of the shapes here and pays the per bucket setup twice as often.
"""

comptime MERGE_BLOCK = 1 << 16
"""Entries per block in the pass that numbers the merged groups.

That pass is a prefix sum, which is one serial walk over the block totals and
two parallel walks over the entries themselves. Sixty five thousand entries is
sixty five kilobytes of marks and a quarter of a megabyte of ranks, so a block
stays in L2 across both of its walks, and a merge with only a few thousand
groups in it still has more blocks than the machine has cores.
"""


struct _Buckets(Movable):
    """Group entry indices grouped by the top bits of their hash."""

    var order: Buffer
    """Entry indices as uint32, grouped by bucket and in entry order within one.
    """

    var offsets: List[Int]
    """Where each bucket starts in `order`, with a final entry holding the total.
    """

    def __init__(out self, var order: Buffer, var offsets: List[Int]):
        """Constructs a bucketing.

        Args:
            order: The grouped entry indices.
            offsets: The bucket boundaries.
        """
        self.order = order^
        self.offsets = offsets^


struct _Merged(Movable):
    """What a merge hands back: a renumbering and a row per group."""

    var map: Buffer
    """Final ordinal as uint32 for every entry, indexed by entry."""

    var firsts: List[Int]
    """The first row of each merged group, in final ordinal order."""

    def __init__(out self, var map: Buffer, var firsts: List[Int]):
        """Constructs a merge result.

        Args:
            map: The per entry renumbering.
            firsts: The representative rows.
        """
        self.map = map^
        self.firsts = firsts^

    def into_parts(deinit self, mut map: Buffer, mut firsts: List[Int]):
        """Gives up both halves together, for the reason `Factorized` gives.

        Args:
            map: Overwritten with the per entry renumbering.
            firsts: Overwritten with the representative rows.
        """
        map = self.map^
        firsts = self.firsts^


def _merge_bits(workers: Int) -> Int:
    """Chooses how many high hash bits the merge partitions the groups on.

    Args:
        workers: How many workers built tables.

    Returns:
        A bit count of at least one.
    """
    var bits = 1
    while (1 << bits) < workers * MERGE_BUCKETS_PER_WORKER and bits < 16:
        bits += 1
    return bits


def _flatten_reps(
    founds: List[List[Int]], starts: List[Int], workers: Int
) raises -> Buffer:
    """Lays the workers' representative rows end to end, indexed by entry.

    The merge reads them out of order, once per entry rather than once per
    worker, and a list of lists costs two loads and a bounds check to do that.
    Flattening them first is one pass over a number of rows the column dwarfs.

    Args:
        founds: Each worker's representative row per local ordinal.
        starts: Where each worker's entries begin, with a final total.
        workers: How many workers there were.

    Returns:
        One int64 row index per entry.
    """
    var reps = Buffer(overwritten=starts[workers] * 8)

    def one(w: Int) raises {mut reps, imm}:
        var out = reps.bitcast[DType.int64]()
        var at = starts[w]
        for k in range(len(founds[w])):
            out.unsafe_offset(at + k).unsafe_write(Int64(founds[w][k]))

    parallel_for(one, workers)
    return reps^


def _bucket_entries(
    found: Buffer, starts: List[Int], workers: Int, bits: Int
) raises -> _Buckets:
    """Groups every worker's groups by the top bits of their hash.

    This is `radix_partition` over group entries rather than over rows, written
    here because it counts in parallel and that one does not. Each worker owns
    the run of entries its own table produced, counts its buckets into its own
    row of the count table, and places its entries once the prefix sum has told
    it where its run of each bucket starts.

    The prefix sum runs bucket major and worker minor, which is what keeps the
    sort stable. Worker zero's entries for a bucket come before worker one's,
    and within a worker the entries are already in the order it found them, so
    walking a bucket afterwards visits its entries in exactly the order the
    serial merge would have.

    Partitions come from the high bits and the buckets inside each merge table
    come from the low ones, which have to be different bits or every key in a
    bucket would probe the same region of that bucket's table.

    Args:
        found: One hash per entry, from `HashTable.keys_by_ordinal`.
        starts: Where each worker's entries begin, with a final total.
        workers: How many workers there were.
        bits: How many high bits to bucket on. `2 ** bits` buckets.

    Returns:
        The entry indices grouped by bucket, and the bucket boundaries.
    """
    var buckets = 1 << bits
    var shift = UInt64(64 - bits)
    var total = starts[workers]
    var counts = List[Int](length=workers * buckets, fill=0)

    def tally(w: Int) raises {mut counts, imm}:
        var hashed = found.bitcast[DType.uint64]()
        var at = w * buckets
        for e in range(starts[w], starts[w + 1]):
            var b = Int(hashed.unsafe_offset(e).unsafe_load() >> shift)
            counts[at + b] += 1

    parallel_for(tally, workers)

    # Rewritten in place into the write cursor each worker uses for each bucket,
    # with the bucket boundaries copied out first.
    var offsets = List[Int](capacity=buckets + 1)
    var running = 0
    for b in range(buckets):
        offsets.append(running)
        for w in range(workers):
            var seen = counts[w * buckets + b]
            counts[w * buckets + b] = running
            running += seen
    offsets.append(running)

    var order = Buffer(overwritten=total * 4)

    def place(w: Int) raises {mut order, imm}:
        var hashed = found.bitcast[DType.uint64]()
        var slot = order.bitcast[DType.uint32]()
        var at = List[Int](capacity=buckets)
        for b in range(buckets):
            at.append(counts[w * buckets + b])
        for e in range(starts[w], starts[w + 1]):
            var b = Int(hashed.unsafe_offset(e).unsafe_load() >> shift)
            slot.unsafe_offset(at[b]).unsafe_write(UInt32(e))
            at[b] += 1

    parallel_for(place, workers)
    return _Buckets(order^, offsets^)


def _rank_entries(
    marks: Buffer, owners: Buffer, reps: Buffer, total: Int, offset: Int
) raises -> _Merged:
    """Numbers the merged groups in entry order and renumbers every entry.

    A bucket knows which of its entries were the first to offer their key and
    which entry each of the rest belongs to, but it cannot know what number to
    give any of them, because the numbering is over all the buckets at once.
    That is a prefix sum over the marks in entry order, and entry order is
    worker order and then within a worker the order it found its groups, which
    is first-appearance order over the whole column. So the count of marks
    before an entry is the ordinal that entry's group gets, and it comes out
    identical to what one thread inserting in one pass would have assigned.

    Args:
        marks: One byte per entry, one where the entry is the first to offer its
            key and zero otherwise.
        owners: For every entry, the entry that owns its group.
        reps: One row index per entry.
        total: How many entries there are.
        offset: Added to every final ordinal, reserving the low ones for the
            null group.

    Returns:
        The renumbering and the representative row of each merged group.
    """
    var blocks = (total + MERGE_BLOCK - 1) // MERGE_BLOCK
    var base = List[Int](length=blocks + 1, fill=0)

    def tally(b: Int) raises {mut base, imm}:
        var mark = marks.unsafe_ptr()
        var stop = min((b + 1) * MERGE_BLOCK, total)
        var seen = 0
        for e in range(b * MERGE_BLOCK, stop):
            seen += Int(mark.unsafe_offset(e).unsafe_load())
        base[b] = seen

    parallel_for(tally, blocks)

    var groups = 0
    for b in range(blocks):
        var seen = base[b]
        base[b] = groups
        groups += seen
    base[blocks] = groups

    var firsts = List[Int](length=groups, fill=0)
    var ranks = Buffer(overwritten=total * 4)

    def number(b: Int) raises {mut ranks, mut firsts, imm}:
        var mark = marks.unsafe_ptr()
        var rank = ranks.bitcast[DType.uint32]()
        var rep = reps.bitcast[DType.int64]()
        var stop = min((b + 1) * MERGE_BLOCK, total)
        var at = base[b]
        for e in range(b * MERGE_BLOCK, stop):
            if mark.unsafe_offset(e).unsafe_load() == 0:
                continue
            rank.unsafe_offset(e).unsafe_write(UInt32(at))
            firsts[at] = Int(rep.unsafe_offset(e).unsafe_load())
            at += 1

    parallel_for(number, blocks)

    var map = Buffer(overwritten=total * 4)

    def project(b: Int) raises {mut map, imm}:
        var mapped = map.bitcast[DType.uint32]()
        var rank = ranks.bitcast[DType.uint32]()
        var own = owners.bitcast[DType.uint32]()
        var stop = min((b + 1) * MERGE_BLOCK, total)
        for e in range(b * MERGE_BLOCK, stop):
            var winner = Int(own.unsafe_offset(e).unsafe_load())
            mapped.unsafe_offset(e).unsafe_write(
                rank.unsafe_offset(winner).unsafe_load() + UInt32(offset)
            )

    parallel_for(project, blocks)
    return _Merged(map^, firsts^)


comptime RANK_BLOCK = 1 << 16
"""Rows per block in the pass that numbers a partitioned build's groups.

That pass gives every block a scratch table with a slot per row in it, so the
block is a quarter of a megabyte of scratch and stays in L2 while it is walked.
Sixty five thousand rows also means a ten million row column has a hundred and
fifty three blocks, which is enough for every core to have several and few enough
that the empty ones cost nothing worth counting.
"""


def _rank_by_first_row(
    reps: Buffer, total: Int, rows: Int, offset: Int
) raises -> _Merged:
    """Numbers groups by the row that introduced them, on every core.

    The slice route gets first-appearance order for free: its workers hold
    contiguous slices in order, so laying their groups end to end is already the
    order the whole column first saw them in, and `_rank_entries` only has to
    count. A partitioned build has no such luck. Its partitions are cut by hash,
    so a partition's groups are in first-appearance order among themselves and
    say nothing about any other partition's, and the numbering has to be
    recovered from the representative rows.

    That is a sort of the groups by a key that is distinct, is an index into the
    column, and is therefore already about as bounded as a sort key gets. So it
    is done as a bucket sort in two levels. The groups are radix partitioned by
    which block of `RANK_BLOCK` rows their representative row falls in, which is
    one counting pass and one placing pass over the groups. Then every block
    sorts its own by writing each group into a scratch table indexed by the row
    itself and reading the table back in order, which needs no comparisons at
    all because two groups cannot share a representative row.

    Blocks that no group landed in are skipped before they allocate, which is
    what keeps a column with a handful of groups from walking a scratch table per
    block of a column it barely touched.

    Args:
        reps: One int64 row index per group, indexed by entry.
        total: How many groups there are.
        rows: The column's length, which bounds the representative rows.
        offset: Added to every final ordinal, reserving the low ones for the
            null group.

    Returns:
        The renumbering and the representative row of each group, both in the
        order the column first saw them.
    """
    if total == 0:
        return _Merged(Buffer(0), List[Int]())

    var blocks = (rows + RANK_BLOCK - 1) // RANK_BLOCK
    var pieces = (total + MERGE_BLOCK - 1) // MERGE_BLOCK
    var counts = List[Int](length=pieces * blocks, fill=0)

    def tally(k: Int) raises {mut counts, imm}:
        var rep = reps.bitcast[DType.int64]()
        var at = k * blocks
        var stop = min((k + 1) * MERGE_BLOCK, total)
        for e in range(k * MERGE_BLOCK, stop):
            var b = Int(rep.unsafe_offset(e).unsafe_load()) // RANK_BLOCK
            counts[at + b] += 1

    parallel_for(tally, pieces)

    # Block major and piece minor, so that a block's ordinals are contiguous and
    # start where the blocks before it ended. Rewritten in place into the write
    # cursor each piece uses for each block.
    var base = List[Int](length=blocks + 1, fill=0)
    var running = 0
    for b in range(blocks):
        base[b] = running
        for k in range(pieces):
            var seen = counts[k * blocks + b]
            counts[k * blocks + b] = running
            running += seen
    base[blocks] = running

    var order = Buffer(overwritten=total * 4)

    def place(k: Int) raises {mut order, imm}:
        var rep = reps.bitcast[DType.int64]()
        var slot = order.bitcast[DType.uint32]()
        var at = List[Int](capacity=blocks)
        for b in range(blocks):
            at.append(counts[k * blocks + b])
        var stop = min((k + 1) * MERGE_BLOCK, total)
        for e in range(k * MERGE_BLOCK, stop):
            var b = Int(rep.unsafe_offset(e).unsafe_load()) // RANK_BLOCK
            slot.unsafe_offset(at[b]).unsafe_write(UInt32(e))
            at[b] += 1

    parallel_for(place, pieces)

    var firsts = List[Int](length=total, fill=0)
    var map = Buffer(overwritten=total * 4)

    def number(b: Int) raises {mut map, mut firsts, imm}:
        var lo = base[b]
        var hi = base[b + 1]
        if lo == hi:
            return
        var rep = reps.bitcast[DType.int64]()
        var slot = order.bitcast[DType.uint32]()
        var mapped = map.bitcast[DType.uint32]()
        # Zero means no group had this row, and `Buffer` hands back zeroed
        # memory, so the entry is stored plus one and the scratch needs no pass
        # of its own to be ready.
        var seat = Buffer(RANK_BLOCK * 4)
        var seats = seat.bitcast[DType.uint32]()
        var floor = b * RANK_BLOCK
        for j in range(lo, hi):
            var e = Int(slot.unsafe_offset(j).unsafe_load())
            var r = Int(rep.unsafe_offset(e).unsafe_load()) - floor
            seats.unsafe_offset(r).unsafe_write(UInt32(e + 1))
        var at = lo
        for r in range(RANK_BLOCK):
            var held = seats.unsafe_offset(r).unsafe_load()
            if held == 0:
                continue
            mapped.unsafe_offset(Int(held) - 1).unsafe_write(
                UInt32(at + offset)
            )
            firsts[at] = floor + r
            at += 1

    parallel_for(number, blocks)
    return _Merged(map^, firsts^)


def _merge_hashed(
    found: Buffer,
    starts: List[Int],
    founds: List[List[Int]],
    workers: Int,
    offset: Int,
    seed: UInt64,
) raises -> _Merged:
    """Folds the workers' numeric tables into one numbering, on every core.

    Every entry with the same key has the same hash and so lands in the same
    bucket, which is what makes the buckets independent: no two of them can
    discover the same group, so no two of them have anything to agree about.
    Each one dedupes its own entries into a table of its own and records, for
    every entry, which entry owns the group it belongs to. `_rank_entries` turns
    those into the final numbers.

    Args:
        found: One hash per entry.
        starts: Where each worker's entries begin, with a final total.
        founds: Each worker's representative row per local ordinal.
        workers: How many workers built tables.
        offset: Added to every final ordinal.
        seed: The per-query hash seed.

    Returns:
        The renumbering and the representative row of each merged group.
    """
    var total = starts[workers]
    var bits = _merge_bits(workers)
    var buckets = 1 << bits
    var split = _bucket_entries(found, starts, workers, bits)
    var reps = _flatten_reps(founds, starts, workers)
    var owners = Buffer(overwritten=total * 4)
    var marks = Buffer(total)

    def one(b: Int) raises {mut owners, mut marks, imm}:
        var hashed = found.bitcast[DType.uint64]()
        var slot = split.order.bitcast[DType.uint32]()
        var own = owners.bitcast[DType.uint32]()
        var mark = marks.unsafe_ptr()
        var start = split.offsets[b]
        var stop = split.offsets[b + 1]
        var table = HashTable(stop - start, seed)
        var winners = List[Int]()
        for at in range(start, stop):
            var e = Int(slot.unsafe_offset(at).unsafe_load())
            var ordinal = table.insert(hashed.unsafe_offset(e).unsafe_load())
            if ordinal == len(winners):
                winners.append(e)
                mark.unsafe_offset(e).unsafe_store(UInt8(1))
            own.unsafe_offset(e).unsafe_write(UInt32(winners[ordinal]))

    parallel_for(one, buckets)
    return _rank_entries(marks, owners, reps, total, offset)


def _merge_strings(
    found: Buffer,
    starts: List[Int],
    founds: List[List[Int]],
    workers: Int,
    offset: Int,
    seed: UInt64,
    col: StringArray,
) raises -> _Merged:
    """`_merge_hashed` with the key comparison put back.

    A hash match is a candidate rather than an answer for text, so a bucket
    carries the representative row of each of its own groups and compares the
    two rows behind a match. Nothing else differs: equal keys hash equally, so
    they still land in the same bucket, and the buckets are still independent.

    Args:
        found: One hash per entry.
        starts: Where each worker's entries begin, with a final total.
        founds: Each worker's representative row per local ordinal.
        workers: How many workers built tables.
        offset: Added to every final ordinal.
        seed: The per-query hash seed.
        col: The column, for the comparison.

    Returns:
        The renumbering and the representative row of each merged group.
    """
    var total = starts[workers]
    var bits = _merge_bits(workers)
    var buckets = 1 << bits
    var split = _bucket_entries(found, starts, workers, bits)
    var reps = _flatten_reps(founds, starts, workers)
    var owners = Buffer(overwritten=total * 4)
    var marks = Buffer(total)

    def one(b: Int) raises {mut owners, mut marks, imm}:
        var hashed = found.bitcast[DType.uint64]()
        var rep = reps.bitcast[DType.int64]()
        var slot = split.order.bitcast[DType.uint32]()
        var own = owners.bitcast[DType.uint32]()
        var mark = marks.unsafe_ptr()
        var start = split.offsets[b]
        var stop = split.offsets[b + 1]
        var table = HashTable(stop - start, seed)
        var local = List[Int]()
        var winners = List[Int]()
        for at in range(start, stop):
            var e = Int(slot.unsafe_offset(at).unsafe_load())
            var ordinal = table.insert_string(
                hashed.unsafe_offset(e).unsafe_load(),
                Int(rep.unsafe_offset(e).unsafe_load()),
                col,
                local,
            )
            if ordinal == len(winners):
                winners.append(e)
                mark.unsafe_offset(e).unsafe_store(UInt8(1))
            own.unsafe_offset(e).unsafe_write(UInt32(winners[ordinal]))

    parallel_for(one, buckets)
    return _rank_entries(marks, owners, reps, total, offset)


def _parallel_workers(groups: Int, n: Int) -> Int:
    """Chooses how many workers to split a column across, or one to stay serial.

    Every worker builds a table that ends up holding roughly all of the column's
    groups, because a slice drawn from anywhere in a column sees the same keys
    the whole column has. So the tables together are the group count times the
    worker count, and where that lands decides the answer. While they all fit in
    the shared cache, every probe is a cache hit and the build is the only cost
    that matters, so take every core. Once they stop fitting, every probe is a
    trip to memory and the extra workers are buying overlap rather than
    throughput, so take half of them.

    That is the whole rule. It used to be a cost model with a merge term in it,
    because the merge was serial and grew with the worker count, and the model
    spent most of its effort deciding when the merge had eaten the split. The
    merge runs on every core now, so what is left is a question about cache.

    Measured on ten million rows on the reference machine, against every worker
    count from two to thirty two, on integer and text columns of a hundred, a
    hundred thousand and a million groups. The rule picks thirty two on the low
    cardinality columns, which is their best, and sixteen on the other four,
    where the best times are at six to twenty four and sixteen is within about a
    tenth of all of them. Every one of the six is at least twice the serial
    route. So is a column where every row is its own key, by a smaller margin,
    at a hundred and twenty nine milliseconds against a hundred and seventy
    four, which is why nothing here returns one on a column this long.

    Args:
        groups: The estimated distinct key count of the whole column.
        n: The column's length.

    Returns:
        A worker count of two or more, or one to stay on the serial route.
    """
    var most = min(worker_count(), n // PARALLEL_MIN_SLICE)
    if most < 2 or groups < 1:
        return 1

    if not _crowded(groups, most):
        return most
    return max(2, most // CROWDED_SHARE)


def _crowded(groups: Int, workers: Int) -> Bool:
    """Reports whether a table per worker would leave the shared cache.

    Every worker on the slice route builds a table that ends up holding roughly
    all of the column's groups, so what the machine is asked to hold is the group
    count times the worker count times the bytes a group takes up. While that
    fits, every probe is a cache hit. Once it does not, every probe is a trip to
    memory, and the workers are also each paying to discover groups the others
    have already discovered and the merge is paying to fold the copies back
    together.

    Two routes read this and they read it differently. `_parallel_workers` uses
    it to decide how many workers the slice route gets, and `_factorize_hashed`
    uses it to decide whether to take the slice route at all. It is one statement
    of the rule so the two cannot drift apart.

    Args:
        groups: The estimated distinct key count of the whole column.
        workers: How many tables would be built.

    Returns:
        True when the tables together are larger than the shared cache.
    """
    return groups * TABLE_BYTES_PER_GROUP * workers > SHARED_CACHE_BYTES


def _estimate_groups(seen: Int, half: Int, rows: Int, n: Int) -> Int:
    """Estimates a column's distinct key count from two points on its curve.

    `project_groups` answers a different question from the same two numbers and
    it is deliberately wrong in a direction this cannot use. It sizes a table,
    where guessing high costs memory and guessing low costs a rehash, so it
    extrapolates the discovery rate flat and overshoots on purpose.
    `_parallel_workers` needs an estimate rather than a bound. A column of ten
    million rows with a hundred thousand groups is one the split wins on by a
    lot, and the flat extrapolation calls it six million and sends it to one
    thread.

    So this fits the curve instead of the tangent. Drawing `m` rows out of a
    column with `g` evenly spread keys turns up `g * (1 - e^(-m/g))` of them,
    which is the coupon collector shape, and the ratio between the count at `m`
    and the count at `m / 2` pins down `m / g` on its own without the scale of
    either count entering into it. That ratio runs from two, when every row is
    still a new key, down to one, when the keys ran out long ago, and it is
    monotone in between, so forty steps of bisection settle it to more digits
    than anything downstream reads.

    Keys that are not evenly spread make it read low, because a skewed column
    hides its rare keys from a sample. Reading low means taking the parallel
    route with more workers than the column deserved, and the cost of that is
    bounded: the worst it can do is take every core on a column whose tables
    would rather have had half of them.

    Args:
        seen: Groups found in the first `rows` rows.
        half: Groups found in the first `rows / 2` of them.
        rows: How many rows were sampled.
        n: The column length.

    Returns:
        An estimate of the distinct key count of the whole column, never above
        `n` and never below what the sample already found.
    """
    if half == 0 or seen <= half:
        # Discovery stopped inside the sample, so what the sample holds is what
        # the column holds, give or take a tail no sample can see.
        return min(n, seen + seen // 4)

    var ratio = Float64(seen) / Float64(half)
    if ratio >= 1.999:
        # Still finding a new key in nearly every row, which the model can only
        # answer with a number larger than the column.
        return n

    # `f` falls from two to one as `r` grows, so a bisection that walks towards
    # the larger `r` whenever `f` is still above the ratio converges on it.
    var low = 1.0e-6
    var high = 64.0
    for _ in range(40):
        var mid = (low + high) * 0.5
        var f = (1.0 - exp(-mid)) / (1.0 - exp(-mid * 0.5))
        if f > ratio:
            low = mid
        else:
            high = mid

    var estimate = Float64(rows) / ((low + high) * 0.5)
    if estimate >= Float64(n):
        return n
    return max(seen, Int(estimate))


def _projected_groups[dt: DType](col: Array[dt], seed: UInt64, n: Int) -> Int:
    """Guesses how many groups a column has, by building the front of it.

    The parallel route is worth taking when the groups are few and worth
    refusing when nearly every row is one, and nothing knows which of those a
    column is until it has been built. So this builds a sample of it, counting
    the groups at the halfway mark as well as at the end, and hands both to
    `_estimate_groups`.

    The sample is thrown away rather than handed to whichever route runs next.
    It is a sixteenth of the column at most, and keeping it would mean both
    routes had to be able to start from a table somebody else had already put
    rows into, which is a lot of coupling to buy back that sixteenth.

    Args:
        col: The column.
        seed: The per-query hash seed.
        n: The column's length.

    Parameters:
        dt: The column's dtype.

    Returns:
        A guess at the number of distinct keys in the whole column.
    """
    var sample = min(PARALLEL_SAMPLE, n // 16)
    var has_null = col.null_count() > 0
    var codes = Array[DType.uint32](sample)
    var firsts = List[Int]()
    var hashes = Buffer(CHUNK_ROWS * 8)
    var table = HashTable(0, seed)

    var half = 0
    var measured = False
    var base = 0
    while base < sample:
        var count = min(CHUNK_ROWS, sample - base)
        hash_chunk(col, base, count, seed, hashes)
        table.build(
            hashes,
            col.data.validity,
            has_null,
            base,
            base,
            count,
            sample,
            0,
            codes,
            firsts,
        )
        base += count
        if not measured and base * 2 >= sample:
            half = len(table)
            measured = True

    return _estimate_groups(len(table), half, sample, n)


def _factorize_hashed_serial[
    dt: DType
](col: Array[dt], seed: UInt64) -> Factorized[dt]:
    """Factorizes through one hash table on one thread.

    Args:
        col: The column.
        seed: The per-query hash seed.

    Parameters:
        dt: The column's dtype.

    Returns:
        The factorization.
    """
    var n = len(col)

    var has_null = col.null_count() > 0
    var offset = 1 if has_null else 0
    # Every row below is written, the null ones with a zero, so the array does
    # not need the pass that zeroes it first.
    var codes = Array[DType.uint32](overwritten=n)
    var null_group = -1
    if has_null:
        null_group = 0

    # Hash a chunk, probe that chunk, move on. Hashing ahead of the probe is what
    # makes the prefetch possible, because a loop that hashed and probed one row
    # at a time would have nothing to run ahead of; the address it wants to fetch
    # is not known until the row it is already working on has been hashed. But
    # hashing the whole column ahead of the probe means writing eight megabytes
    # to memory and reading them back, which buys nothing. A chunk gets the
    # lookahead and keeps the buffer in cache.
    var hashes = Buffer(CHUNK_ROWS * 8)

    # The table writes the ordinals and hands back the row that first produced
    # each one, which keeps it free of any idea of what a value is. Reading the
    # key values out of those rows is `Factorized.keys` and is not done here,
    # because no caller in the library wants them.
    var firsts = List[Int]()
    var table = HashTable(0, seed)

    var base = 0
    while base < n:
        var count = min(CHUNK_ROWS, n - base)
        hash_chunk(col, base, count, seed, hashes)
        table.build(
            hashes,
            col.data.validity,
            has_null,
            base,
            base,
            count,
            n,
            offset,
            codes,
            firsts,
        )
        base += count

    return Factorized[dt](codes^, firsts^, null_group)


def _factorize_hashed_parallel[
    dt: DType
](col: Array[dt], seed: UInt64, workers: Int) raises -> Factorized[dt]:
    """Factorizes through one hash table per core, then merges what they found.

    The column is cut into one contiguous slice per worker and each worker runs
    the serial build over its own slice into its own table, numbering its groups
    from zero. Nothing is shared, so nothing is locked, and the only writes that
    land in the same array are the ordinals, which are indexed by row and so
    never touch the same word twice.

    That leaves every worker with a private numbering, and a merge to turn those
    into one. Laid end to end, the workers' groups are already in
    first-appearance order over the whole column: the slices are contiguous and
    in order, and a worker numbers its groups by when it first saw them, so the
    first time any key turns up anywhere is the first time it appears in that
    sequence. The merge's job is to find the repeats and to number what is left
    without disturbing that order, and `_merge_hashed` does it on every core by
    bucketing on the hash. The ordinals come out identical to the serial route's,
    which is what makes this a substitution rather than a second set of
    semantics.

    It probes with the hashes the workers already computed, which
    `HashTable.keys_by_ordinal` reads back out of their slots, so nothing is
    hashed twice. What is left that no thread helps with is a prefix sum over
    the block totals of the ranking pass, which is one number per sixty five
    thousand groups.

    Args:
        col: The column.
        seed: The per-query hash seed.
        workers: How many slices to cut the column into. At least two.

    Parameters:
        dt: The column's dtype.

    Returns:
        The factorization.
    """
    var n = len(col)
    var has_null = col.null_count() > 0
    var offset = 1 if has_null else 0
    # The slices below cover the column between them and the build writes every
    # row of the one it is given, the null ones with a zero, so the array does
    # not need the pass that zeroes it first. That pass is one thread writing
    # forty megabytes on ten million rows, in front of a section that is on
    # every core, which is the worst place in the routine to put it.
    var codes = Array[DType.uint32](overwritten=n)

    # Slices land on chunk boundaries so that every worker's inner loop is the
    # same shape as the serial one's, with a short chunk only at the very end.
    var chunks = (n + CHUNK_ROWS - 1) // CHUNK_ROWS
    var bounds = List[Int](capacity=workers + 1)
    for w in range(workers):
        bounds.append(chunks * w // workers * CHUNK_ROWS)
    bounds.append(n)

    var tables = List[HashTable](capacity=workers)
    var founds = List[List[Int]](capacity=workers)
    for _ in range(workers):
        tables.append(HashTable(0, seed))
        founds.append(List[Int]())

    def one(w: Int) raises {mut tables, mut founds, mut codes, imm}:
        var start = bounds[w]
        var stop = bounds[w + 1]
        var hashes = Buffer(CHUNK_ROWS * 8)
        var base = start
        while base < stop:
            var count = min(CHUNK_ROWS, stop - base)
            hash_chunk(col, base, count, seed, hashes)
            tables[w].build(
                hashes,
                col.data.validity,
                has_null,
                base,
                base - start,
                count,
                stop - start,
                0,
                codes,
                founds[w],
            )
            base += count

    parallel_for(one, workers)

    # Where each worker's block of local ordinals starts once they are laid end
    # to end. This is the number the remap adds to a local ordinal to find its
    # entry in the map.
    var starts = List[Int](capacity=workers + 1)
    var total = 0
    for w in range(workers):
        starts.append(total)
        total += len(tables[w])
    starts.append(total)

    var found = Buffer(total * 8)

    def dump(w: Int) raises {mut found, imm}:
        tables[w].keys_by_ordinal(found, starts[w])

    parallel_for(dump, workers)

    var null_group = -1
    if has_null:
        null_group = 0

    var map = Buffer(0)
    var firsts = List[Int]()
    _merge_hashed(found, starts, founds, workers, offset, seed).into_parts(
        map, firsts
    )
    var mapped = map.bitcast[DType.uint32]()

    def remap(w: Int) raises {mut codes, imm}:
        var out = codes.unsafe_ptr()
        var at = starts[w]
        for i in range(bounds[w], bounds[w + 1]):
            if has_null and not col.is_valid(i):
                out.unsafe_offset(i).unsafe_write(UInt32(0))
                continue
            var local = Int(out.unsafe_offset(i).unsafe_load())
            out.unsafe_offset(i).unsafe_write(
                mapped.unsafe_offset(at + local).unsafe_load()
            )

    parallel_for(remap, workers)

    return Factorized[dt](codes^, firsts^, null_group)


def _factorize_hashed_partitioned[
    dt: DType
](col: Array[dt], seed: UInt64, workers: Int) raises -> Factorized[dt]:
    """Factorizes by cutting the rows into partitions no two workers share.

    `_factorize_hashed_parallel` gives every worker a contiguous slice, which
    means every worker sees very nearly every key the column has, so every
    worker's table grows to hold very nearly every group. On a column with a
    million groups and eight workers that is eight million entries built and then
    folded back down to one million, and it is why the slice route stops scaling:
    measured on ten million int64 rows, sixteen cores took a million group column
    from a hundred and thirty eight milliseconds to eighty two, which is not what
    sixteen cores are for.

    So this cuts by the hash instead of by the row. Every row goes to the
    partition its top hash bits name, no two partitions can hold the same key,
    and a partition's table therefore holds its own share of the groups and
    nothing else. Nothing has to be deduplicated afterwards because nothing was
    duplicated, which deletes `_merge_hashed` from the route entirely.

    What it costs is that the rows have to be moved. Three passes over the column
    where the slice route has one: hash every row and count where it would go,
    then write the row's number and its hash into the partition, then build. The
    partition arrays are twelve bytes a row against the eighty a ten million row
    int64 column already is, and the writes go to one cursor per partition rather
    than anywhere, so the move is bandwidth rather than misses.

    What it also costs is the numbering. First-appearance order is not free here
    the way it is on the slice route, and `_rank_by_first_row` is what buys it
    back. The ordinals come out identical to the serial route's, which is what
    makes this a substitution rather than a second set of semantics.

    Args:
        col: The column.
        seed: The per-query hash seed.
        workers: How many slices to hash and scatter in parallel. At least two.

    Parameters:
        dt: The column's dtype.

    Returns:
        The factorization.
    """
    var n = len(col)
    var has_null = col.null_count() > 0
    var offset = 1 if has_null else 0

    # The partition count and the merge's bucket count answer the same question,
    # so they are the same number: enough partitions that one drawing more than
    # its share is a fraction of a core's turn rather than the tail of the phase,
    # and small enough that the scatter has one write cursor per partition and
    # they all stay live.
    var bits = _merge_bits(workers)
    var parts = 1 << bits
    var shift = UInt64(64 - bits)

    # Slices land on chunk boundaries so that the hashing is the same shape as
    # everywhere else, with a short chunk only at the very end.
    var chunks = (n + CHUNK_ROWS - 1) // CHUNK_ROWS
    var bounds = List[Int](capacity=workers + 1)
    for w in range(workers):
        bounds.append(chunks * w // workers * CHUNK_ROWS)
    bounds.append(n)

    var hashes = List[Buffer](capacity=workers)
    for w in range(workers):
        hashes.append(Buffer(overwritten=(bounds[w + 1] - bounds[w]) * 8))
    var counts = List[Int](length=workers * parts, fill=0)

    def scan(w: Int) raises {mut hashes, mut counts, imm}:
        var start = bounds[w]
        var stop = bounds[w + 1]
        hash_chunk(col, start, stop - start, seed, hashes[w])
        var hash = hashes[w].bitcast[DType.uint64]()
        var at = w * parts
        for k in range(stop - start):
            if has_null and not col.is_valid(start + k):
                continue
            counts[at + Int(hash.unsafe_offset(k).unsafe_load() >> shift)] += 1

    parallel_for(scan, workers)

    # Partition major and worker minor, which is what puts each partition's rows
    # in increasing row order: worker zero's rows for a partition come before
    # worker one's, and within a worker they are already in order. That order is
    # what makes a partition's groups come out in the order it first saw them.
    var offsets = List[Int](capacity=parts + 1)
    var running = 0
    for p in range(parts):
        offsets.append(running)
        for w in range(workers):
            var seen = counts[w * parts + p]
            counts[w * parts + p] = running
            running += seen
    offsets.append(running)
    var valid = running

    # The hash travels with the row rather than being looked up again from the
    # row's number. Reading it back would be a miss per row into a buffer the
    # size of the column; carrying it is eight more bytes written in order.
    var rows_at = Buffer(overwritten=valid * 4)
    var keys = Buffer(overwritten=valid * 8)

    def place(w: Int) raises {mut rows_at, mut keys, imm}:
        var start = bounds[w]
        var stop = bounds[w + 1]
        var hash = hashes[w].bitcast[DType.uint64]()
        var row = rows_at.bitcast[DType.uint32]()
        var key = keys.bitcast[DType.uint64]()
        var at = List[Int](capacity=parts)
        for p in range(parts):
            at.append(counts[w * parts + p])
        for k in range(stop - start):
            if has_null and not col.is_valid(start + k):
                continue
            var found = hash.unsafe_offset(k).unsafe_load()
            var p = Int(found >> shift)
            row.unsafe_offset(at[p]).unsafe_write(UInt32(start + k))
            key.unsafe_offset(at[p]).unsafe_write(found)
            at[p] += 1

    parallel_for(place, workers)
    # The hashes are in the partitions now, and they are eight bytes a row, so
    # the copy they were scattered out of is worth dropping before the build
    # starts allocating tables.
    _ = hashes^

    # One ordinal per placed row, local to its partition. Indexed by where the
    # row sits in the partition arrays rather than by the row itself, which is
    # what lets `HashTable.build` be called here unchanged: it is being handed a
    # run of hashes and a place to put a number per hash, and it does not care
    # that the run is a partition rather than a slice.
    var local = Array[DType.uint32](overwritten=valid)
    var founds = List[List[Int]](capacity=parts)
    for _ in range(parts):
        founds.append(List[Int]())

    def dig(p: Int) raises {mut founds, mut local, imm}:
        var start = offsets[p]
        var stop = offsets[p + 1]
        var table = HashTable(0, seed)
        var seen = List[Int]()
        var base = start
        while base < stop:
            var count = min(CHUNK_ROWS, stop - base)
            table.build(
                keys,
                col.data.validity,
                False,
                base,
                base - start,
                count,
                stop - start,
                0,
                local,
                seen,
                base,
            )
            base += count
        founds[p] = seen^

    parallel_for(dig, parts)

    var starts = List[Int](capacity=parts + 1)
    var total = 0
    for p in range(parts):
        starts.append(total)
        total += len(founds[p])
    starts.append(total)

    # `build` reports the position in the partition arrays that introduced each
    # group, because that is what it was given as a row. The row it stands for is
    # one load away and the numbering wants the row.
    var reps = Buffer(overwritten=total * 8)

    def gather(p: Int) raises {mut reps, imm}:
        var out = reps.bitcast[DType.int64]()
        var row = rows_at.bitcast[DType.uint32]()
        var at = starts[p]
        for l in range(len(founds[p])):
            out.unsafe_offset(at + l).unsafe_write(
                Int64(row.unsafe_offset(founds[p][l]).unsafe_load())
            )

    parallel_for(gather, parts)

    var map = Buffer(0)
    var firsts = List[Int]()
    _rank_by_first_row(reps, total, n, offset).into_parts(map, firsts)
    var mapped = map.bitcast[DType.uint32]()

    var codes = Array[DType.uint32](overwritten=n)

    def assign(p: Int) raises {mut codes, imm}:
        var out = codes.unsafe_ptr()
        var row = rows_at.bitcast[DType.uint32]()
        var code = local.unsafe_ptr()
        var at = starts[p]
        for e in range(offsets[p], offsets[p + 1]):
            var to = Int(row.unsafe_offset(e).unsafe_load())
            var entry = at + Int(code.unsafe_offset(e).unsafe_load())
            out.unsafe_offset(to).unsafe_write(
                mapped.unsafe_offset(entry).unsafe_load()
            )

    parallel_for(assign, parts)

    # The nulls never entered a partition, and `codes` was allocated without a
    # pass to zero it, so they are the rows nothing has written yet.
    if has_null:

        def blank(w: Int) raises {mut codes, imm}:
            var out = codes.unsafe_ptr()
            for i in range(bounds[w], bounds[w + 1]):
                if not col.is_valid(i):
                    out.unsafe_offset(i).unsafe_write(UInt32(0))

        parallel_for(blank, workers)

    var null_group = -1
    if has_null:
        null_group = 0
    return Factorized[dt](codes^, firsts^, null_group)


struct FactorizedStrings(Movable):
    """A string column rewritten as group ordinals, plus where to find the keys.
    """

    var codes: Array[DType.uint32]
    """One group ordinal per row of the input, in `[0, groups)`."""

    var firsts: List[Int]
    """For each non-null group, the first row that had it, in ordinal order.

    `Factorized` records the same thing, and this is the shape both of them are.
    A string key is not a scalar, so there is no column of keys this one could
    hand back even on request, which is the only way the two still differ.
    """

    var null_group: Int
    """The ordinal the nulls got, which is zero, or -1 if the column had none."""

    def __init__(
        out self,
        var codes: Array[DType.uint32],
        var firsts: List[Int],
        null_group: Int,
    ):
        """Constructs a result.

        Args:
            codes: The per-row ordinals.
            firsts: A representative row per non-null group.
            null_group: The ordinal for nulls, or -1.
        """
        self.codes = codes^
        self.firsts = firsts^
        self.null_group = null_group

    def count(self) -> Int:
        """Returns the number of groups.

        Returns:
            The group count, including the null group if there is one.
        """
        return len(self.firsts) + (1 if self.null_group >= 0 else 0)

    def into_codes(deinit self) -> Array[DType.uint32]:
        """Gives up the ordinals without copying them, dropping the rest.

        Returns:
            The per-row ordinals.
        """
        return self.codes^

    def into_parts(
        deinit self, mut codes: Array[DType.uint32], mut firsts: List[Int]
    ):
        """Gives up the ordinals and the representative rows together.

        A caller that keeps both cannot move them out one at a time, because the
        first move leaves a struct that still owns the second, and it cannot take
        them as a tuple either, because unpacking one copies. `deinit` says the
        rest is being torn down, which is what makes both moves legal, and the
        two arguments are what makes them reachable.

        Args:
            codes: Overwritten with the per-row ordinals.
            firsts: Overwritten with a representative row per non-null group.
        """
        codes = self.codes^
        firsts = self.firsts^


def factorize_strings(
    col: StringArray, seed: UInt64 = DEFAULT_SEED
) raises -> FactorizedStrings:
    """Assigns a group ordinal to every element of a string column.

    There is no direct route here. The direct route exists because an integer
    can index a table by being itself, and a string cannot, so everything goes
    through the hash table, on one thread or on several.

    Args:
        col: The column to group.
        seed: The per-query hash seed.

    Returns:
        The ordinals, a representative row per group, and which ordinal the
        nulls got.
    """
    var n = len(col)
    if n >= PARALLEL_STRING_ROWS and worker_count() > 1:
        var groups = _projected_groups_strings(col, seed, n)
        var most = min(worker_count(), n // PARALLEL_MIN_SLICE)
        # The same three questions the numeric route asks, in the same order and
        # for the same reasons. A string key makes the partitioned route look
        # better rather than worse, because what the slice route wastes is a
        # table entry per worker per group and a string group costs a comparison
        # to establish as well as a slot to hold.
        if (
            most >= 2
            and groups >= 1
            and _crowded(groups, most)
            and n <= Int(UInt32.MAX)
        ):
            return _factorize_strings_partitioned(col, seed, most)
        var workers = _parallel_workers(groups, n)
        if workers > 1:
            return _factorize_strings_parallel(col, seed, workers)
    return _factorize_strings_serial(col, seed)


def _projected_groups_strings(col: StringArray, seed: UInt64, n: Int) -> Int:
    """Guesses how many groups a string column has, by building the front of it.

    `_projected_groups` for text, and it exists for the same reason and answers
    the same question. It is worth a little more here than it is there, because
    a string key costs more to hash and to compare than an integer one does, so
    the sample is a larger share of the work it is deciding about and the wrong
    decision is a larger share of it too.

    Args:
        col: The column.
        seed: The per-query hash seed.
        n: The column's length.

    Returns:
        A guess at the number of distinct keys in the whole column.
    """
    var sample = min(PARALLEL_SAMPLE, n // 16)
    var has_null = col.null_count() > 0
    var codes = Array[DType.uint32](sample)
    var firsts = List[Int]()
    var keyviews = List[StringView]()
    var hashes = Buffer(CHUNK_ROWS * 8)
    var table = HashTable(0, seed)

    var half = 0
    var measured = False
    var base = 0
    while base < sample:
        var count = min(CHUNK_ROWS, sample - base)
        hash_strings_chunk(col, base, count, seed, hashes)
        table.build_strings(
            hashes,
            col,
            has_null,
            base,
            base,
            count,
            sample,
            0,
            codes,
            firsts,
            keyviews,
        )
        base += count
        if not measured and base * 2 >= sample:
            half = len(table)
            measured = True

    return _estimate_groups(len(table), half, sample, n)


def _factorize_strings_parallel(
    col: StringArray, seed: UInt64, workers: Int
) raises -> FactorizedStrings:
    """Factorizes a string column with one hash table per core.

    `_factorize_hashed_parallel` with the key comparison put back, and the
    argument for the shape is the one made there. What differs is the merge.
    The numeric merge probes on the hash alone, because for it the hash is the
    key; this one probes on the hash and then compares the two rows behind it,
    which is `_merge_strings` rather than `_merge_hashed`.

    The workers' representative rows feed it. A worker records the absolute row
    of every key that was new to its own table, and the merge keeps those in the
    same workers-in-order, ordinals-in-order sequence that gives the numeric
    route its first-appearance ordering, so the row that ends up representing a
    group is the earliest row in the column that has that key, exactly as one
    thread would have chosen.

    Args:
        col: The column.
        seed: The per-query hash seed.
        workers: How many slices to cut the column into.

    Returns:
        The ordinals, a representative row per group, and which ordinal the
        nulls got.
    """
    var n = len(col)
    var has_null = col.null_count() > 0
    var offset = 1 if has_null else 0
    # The slices below cover the column between them and the build writes every
    # row of the one it is given, the null ones with a zero, so the array does
    # not need the pass that zeroes it first. That pass is one thread writing
    # forty megabytes on ten million rows, in front of a section that is on
    # every core, which is the worst place in the routine to put it.
    var codes = Array[DType.uint32](overwritten=n)
    var null_group = -1
    if has_null:
        null_group = 0

    var chunks = (n + CHUNK_ROWS - 1) // CHUNK_ROWS
    var bounds = List[Int](capacity=workers + 1)
    for w in range(workers):
        bounds.append(chunks * w // workers * CHUNK_ROWS)
    bounds.append(n)

    var tables = List[HashTable](capacity=workers)
    var founds = List[List[Int]](capacity=workers)
    var keyviews = List[List[StringView]](capacity=workers)
    for _ in range(workers):
        tables.append(HashTable(0, seed))
        founds.append(List[Int]())
        keyviews.append(List[StringView]())

    def one(
        w: Int,
    ) raises {mut tables, mut founds, mut keyviews, mut codes, imm}:
        var start = bounds[w]
        var stop = bounds[w + 1]
        var hashes = Buffer(CHUNK_ROWS * 8)
        var base = start
        while base < stop:
            var count = min(CHUNK_ROWS, stop - base)
            hash_strings_chunk(col, base, count, seed, hashes)
            tables[w].build_strings(
                hashes,
                col,
                has_null,
                base,
                base - start,
                count,
                stop - start,
                0,
                codes,
                founds[w],
                keyviews[w],
            )
            base += count

    parallel_for(one, workers)

    var starts = List[Int](capacity=workers + 1)
    var total = 0
    for w in range(workers):
        starts.append(total)
        total += len(tables[w])
    starts.append(total)

    var found = Buffer(total * 8)

    def dump(w: Int) raises {mut found, imm}:
        tables[w].keys_by_ordinal(found, starts[w])

    parallel_for(dump, workers)

    var map = Buffer(0)
    var firsts = List[Int]()
    _merge_strings(
        found, starts, founds, workers, offset, seed, col
    ).into_parts(map, firsts)
    var mapped = map.bitcast[DType.uint32]()

    def remap(w: Int) raises {mut codes, imm}:
        var out = codes.unsafe_ptr()
        var at = starts[w]
        for i in range(bounds[w], bounds[w + 1]):
            if has_null and not col.is_valid(i):
                out.unsafe_offset(i).unsafe_write(UInt32(0))
                continue
            var local = Int(out.unsafe_offset(i).unsafe_load())
            out.unsafe_offset(i).unsafe_write(
                mapped.unsafe_offset(at + local).unsafe_load()
            )

    parallel_for(remap, workers)

    return FactorizedStrings(codes^, firsts^, null_group)


def _factorize_strings_partitioned(
    col: StringArray, seed: UInt64, workers: Int
) raises -> FactorizedStrings:
    """Factorizes a string column by cutting the rows into disjoint partitions.

    `_factorize_hashed_partitioned` for text, and the argument for it is the one
    made there, only stronger. What the slice route wastes is a table entry per
    worker per group, and a string group costs more than a numeric one to
    establish, because a hash match has to be settled by comparing two rows of
    the column. Doing that work once per worker instead of once is the thing this
    route stops doing.

    The comparison is also why the build needs the row rather than the position.
    A partition holds rows from all over the column, so the table is told where
    each of its positions came from and looks the column up through that. That is
    the only difference between this and the numeric partitioned route, and it is
    two lines of `build_strings`.

    The keys are compared as bytes when they collide, so the merge this route
    does not do is `_merge_strings` rather than `_merge_hashed`, and the saving is
    the larger of the two.

    Args:
        col: The column.
        seed: The per-query hash seed.
        workers: How many slices to hash and scatter in parallel. At least two.

    Returns:
        The ordinals, a representative row per group, and which ordinal the
        nulls got.
    """
    var n = len(col)
    var has_null = col.null_count() > 0
    var offset = 1 if has_null else 0

    var bits = _merge_bits(workers)
    var parts = 1 << bits
    var shift = UInt64(64 - bits)

    var chunks = (n + CHUNK_ROWS - 1) // CHUNK_ROWS
    var bounds = List[Int](capacity=workers + 1)
    for w in range(workers):
        bounds.append(chunks * w // workers * CHUNK_ROWS)
    bounds.append(n)

    var hashes = List[Buffer](capacity=workers)
    for w in range(workers):
        hashes.append(Buffer(overwritten=(bounds[w + 1] - bounds[w]) * 8))
    var counts = List[Int](length=workers * parts, fill=0)

    def scan(w: Int) raises {mut hashes, mut counts, imm}:
        var start = bounds[w]
        var stop = bounds[w + 1]
        hash_strings_chunk(col, start, stop - start, seed, hashes[w])
        var hash = hashes[w].bitcast[DType.uint64]()
        var at = w * parts
        for k in range(stop - start):
            if has_null and not col.is_valid(start + k):
                continue
            counts[at + Int(hash.unsafe_offset(k).unsafe_load() >> shift)] += 1

    parallel_for(scan, workers)

    # Partition major and worker minor, so each partition's rows come out in
    # increasing row order and its groups come out in the order it first saw
    # them.
    var offsets = List[Int](capacity=parts + 1)
    var running = 0
    for p in range(parts):
        offsets.append(running)
        for w in range(workers):
            var seen = counts[w * parts + p]
            counts[w * parts + p] = running
            running += seen
    offsets.append(running)
    var valid = running

    var rows_at = Buffer(overwritten=valid * 4)
    var keys = Buffer(overwritten=valid * 8)

    def place(w: Int) raises {mut rows_at, mut keys, imm}:
        var start = bounds[w]
        var stop = bounds[w + 1]
        var hash = hashes[w].bitcast[DType.uint64]()
        var row = rows_at.bitcast[DType.uint32]()
        var key = keys.bitcast[DType.uint64]()
        var at = List[Int](capacity=parts)
        for p in range(parts):
            at.append(counts[w * parts + p])
        for k in range(stop - start):
            if has_null and not col.is_valid(start + k):
                continue
            var found = hash.unsafe_offset(k).unsafe_load()
            var p = Int(found >> shift)
            row.unsafe_offset(at[p]).unsafe_write(UInt32(start + k))
            key.unsafe_offset(at[p]).unsafe_write(found)
            at[p] += 1

    parallel_for(place, workers)
    # A string hash is the expensive one to produce, which is the whole reason it
    # travelled with the row rather than being recomputed, and now that it has
    # arrived the copy it came from is worth dropping.
    _ = hashes^

    var local = Array[DType.uint32](overwritten=valid)
    var founds = List[List[Int]](capacity=parts)
    for _ in range(parts):
        founds.append(List[Int]())

    def dig(p: Int) raises {mut founds, mut local, imm}:
        var start = offsets[p]
        var stop = offsets[p + 1]
        var table = HashTable(0, seed)
        var seen = List[Int]()
        var keyviews = List[StringView]()
        var base = start
        while base < stop:
            var count = min(CHUNK_ROWS, stop - base)
            table.build_strings[True](
                keys,
                col,
                False,
                base,
                base - start,
                count,
                stop - start,
                0,
                local,
                seen,
                keyviews,
                base,
                rows_at,
            )
            base += count
        founds[p] = seen^

    parallel_for(dig, parts)

    var starts = List[Int](capacity=parts + 1)
    var total = 0
    for p in range(parts):
        starts.append(total)
        total += len(founds[p])
    starts.append(total)

    # The representative rows are already rows here, because the comparison made
    # the build read them, so there is no lookup to do before the numbering. The
    # numeric route has to translate its positions first.
    var reps = Buffer(overwritten=total * 8)

    def gather(p: Int) raises {mut reps, imm}:
        var out = reps.bitcast[DType.int64]()
        var at = starts[p]
        for l in range(len(founds[p])):
            out.unsafe_offset(at + l).unsafe_write(Int64(founds[p][l]))

    parallel_for(gather, parts)

    var map = Buffer(0)
    var firsts = List[Int]()
    _rank_by_first_row(reps, total, n, offset).into_parts(map, firsts)
    var mapped = map.bitcast[DType.uint32]()

    var codes = Array[DType.uint32](overwritten=n)

    def assign(p: Int) raises {mut codes, imm}:
        var out = codes.unsafe_ptr()
        var row = rows_at.bitcast[DType.uint32]()
        var code = local.unsafe_ptr()
        var at = starts[p]
        for e in range(offsets[p], offsets[p + 1]):
            var to = Int(row.unsafe_offset(e).unsafe_load())
            var entry = at + Int(code.unsafe_offset(e).unsafe_load())
            out.unsafe_offset(to).unsafe_write(
                mapped.unsafe_offset(entry).unsafe_load()
            )

    parallel_for(assign, parts)

    # The nulls never entered a partition, so they are the rows nothing has
    # written yet.
    if has_null:

        def blank(w: Int) raises {mut codes, imm}:
            var out = codes.unsafe_ptr()
            for i in range(bounds[w], bounds[w + 1]):
                if not col.is_valid(i):
                    out.unsafe_offset(i).unsafe_write(UInt32(0))

        parallel_for(blank, workers)

    var null_group = -1
    if has_null:
        null_group = 0
    return FactorizedStrings(codes^, firsts^, null_group)


def _factorize_strings_serial(
    col: StringArray, seed: UInt64
) -> FactorizedStrings:
    """Factorizes a string column through one hash table on one thread.

    Args:
        col: The column.
        seed: The per-query hash seed.

    Returns:
        The ordinals, a representative row per group, and which ordinal the
        nulls got.
    """
    var n = len(col)
    var has_null = col.null_count() > 0
    var offset = 1 if has_null else 0
    # Every row below is written, the null ones with a zero, so the array does
    # not need the pass that zeroes it first.
    var codes = Array[DType.uint32](overwritten=n)
    var null_group = -1
    if has_null:
        null_group = 0

    # Same chunking as the numeric path and for the same reason, except that the
    # hashes cost more to produce here, so there is more to hide behind the
    # prefetch and the chunk earns more than it does there.
    var hashes = Buffer(CHUNK_ROWS * 8)
    var firsts = List[Int]()
    var keyviews = List[StringView]()
    var table = HashTable(0, seed)

    var base = 0
    while base < n:
        var count = min(CHUNK_ROWS, n - base)
        hash_strings_chunk(col, base, count, seed, hashes)
        table.build_strings(
            hashes,
            col,
            has_null,
            base,
            base,
            count,
            n,
            offset,
            codes,
            firsts,
            keyviews,
        )
        base += count

    return FactorizedStrings(codes^, firsts^, null_group)
