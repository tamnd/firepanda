"""Turning one or more key columns into a single dense group ordinal per row.

`factorize` does this for one column of one known dtype. A group by needs it for
several columns whose dtypes are runtime values and which have to be combined
into one ordinal that identifies the tuple rather than any single key. That is
what this file is, and it lives in `firepanda/hash` rather than in the frame
layer because it is the same operation `factorize` is and it belongs next to it.

## How several keys become one

The obvious approach is to hash the tuple. This does not, because `factorize`
already exists and composing it is both cheaper and easier to be sure of.

Group on the first key and you have an ordinal per row in `[0, g0)`. Group on the
second and you have another in `[0, g1)`. The pair `(c0, c1)` identifies the tuple
exactly, and `c0 * g1 + c1` packs the pair into one integer without collisions.
Multiply by `g2` and add the third key's ordinal, and so on down the key list. The
running value is in `[0, g0 * g1 * g2 * ...)`, it identifies the tuple exactly at
every step, and one factorize of it at the end turns it into the dense
first-appearance ordering the result wants.

Only at the end. Refactorizing after every combine also works, and is what this
did, and the reason to stop is that it is a full hashed pass over every row per
key on exactly the group bys that are already the slowest. What it buys is a
smaller running space, which matters only when the space would leave an int64.
The product of the group counts reaches nineteen digits on twenty six columns of
a hundred values each, or six of a thousand, so a real table does not get there;
`_condense` is what runs if one does, and the bound is checked either way,
because "cannot happen" is not a thing to rely on in the one function every group
by goes through.

Carrying the space forward is also what lets the last factorize be
`factorize_dense` rather than `factorize`. The packed value's range is not
something anyone has to scan for, it is `g0 * g1 * g2 * ...` and it was computed
on the way down, and a range that a group by constructed out of dense ordinals is
itself densely occupied. So the group bys where the space is large enough to
matter are the ones where a table over it is mostly full, and they index it
instead of hashing.

There is still a real cost here and it should not be hidden: `n` keys means `n`
factorize passes plus `n - 1` packing passes plus one more factorize. A single
fused hash of the tuple would be one pass. The reason to start here is that the
composition reuses the direct route, so grouping on three small integer columns
hashes once instead of three times, and a fused tuple hash would hash all of
them. Which of those wins is a measurement, and the benchmark that settles it is
`group/ordinals_two_keys` against `group/ordinals_one_key`.

## Skipping the per key factorize

That measurement was taken. On ten million rows, one key costs 8.4 milliseconds
and two cost 23.8, so a second key of eight values costs nearly twice what the
first key of a thousand did. Nothing about the second key is expensive. What is
expensive is that the composition insists on giving every key its own dense
ordinals first, and each of those is a scan for the range, a pass to assign, and
a forty megabyte column of ordinals for the packing pass to read back.

None of that is needed when the key is an integer in a narrow range, and the
paragraphs above say why: only a key's group count and its ordinals as values are
read here, never their order and never their density. So the value itself will
do. Subtract the column's minimum and the key is already an ordinal in
`[0, high - low + 1)`, which is a range that is at worst as wide as the one
`factorize` would have produced and is usually the same one. That is a plan, not
a pass, and `direct_plan` already computes it in the scan `factorize` was going
to do anyway.

`_tuple_plan` asks that of every key. If all of them answer, and the product of
their ranges is a space a table can be laid over, then one pass over the raw key
columns writes the packed value directly and the per key factorizes never
happen. `n` scans plus one pass plus one factorize, rather than `n` scans plus
`n` passes plus `n - 1` packing passes plus one factorize.

The route is declined for a key that is not an integer, for a key whose values
are spread too far apart for the subtraction to be a dense ordinal, for a mix of
dtypes, and for a key with a null in it. The first three are the same questions
`factorize` asks. The null is a narrower rule than it has to be: a null holds no
value, so it has no place in a range, and reserving it a slot the way
`_factorize_direct` does is a thing this could learn. Until a group by on a
nullable key pair shows up in a measurement it is a case the general route
already handles correctly.

## Packing the codes in one pass

That route is declined by every group by in db-benchmark, because three of the
six key columns there are text and every multi key query in the suite uses one
of them. A text key has to be factorized, there is no arithmetic that turns a
string into an ordinal, so for those the `n` factorize passes are not something
to remove. The `n - 1` packing passes are.

They exist because the packing was written as a fold: take the running value,
multiply it by the next key's group count, add the next key's ordinal, and go
round again. Each turn of that reads the running column, reads a key's ordinals
and writes the running column back, and the running column is int64 because the
product it accumulates has no bound. Six keys is five turns and a gigabyte of
traffic on ten million rows.

Nothing forces it to be a fold. The number the fold arrives at is
`c0 * g1 * g2 * ... + c1 * g2 * ... + ... + cn`, which is positional notation,
and every one of those multipliers is known before the first row is read: the
last key's is one and each earlier key's is the product of the group counts to
its right. So the whole thing is one weighted sum per row over ordinals that are
all sitting in memory already, and one pass computes it.

The width falls out too. A fold has to be int64 because it cannot know where the
product stops, but a pass that has all the counts in hand can check first, and
where the product fits a uint32 the packed column is four bytes a row rather
than eight and the vector is twice as wide. Two low cardinality keys is the
common shape and it always fits.

What it costs is that all `n` ordinal columns have to be alive at once, where the
fold used to hold two and its running column however many keys there were. Four
bytes a row a key against the fold's flat sixteen is a saving on two keys, a wash
on three and a cost above that: six keys packed into an int64 is thirty two bytes
a row. Three hundred and twenty megabytes on ten million rows, to take nine
hundred and sixty out of the traffic.

The fold pays that too, because the factorizes all happen before either route is
chosen and the choice needs every group count to have been made. It could be
arranged not to, and it is not worth arranging: what sends a group by to the fold
is a product that leaves an int64, which needs six keys of a thousand values or
twenty six of a hundred and does not happen to a real table. The fold is kept
because renumbering partway through is what `_condense` does and a single pass
has no partway, and the check is on the product rather than on the key count, so
nothing has to predict which tables are real.

## The representative row

An aggregation produces one row per group and those rows need their key values
back. Rather than teach every dtype how to reconstruct a key from the hash table,
this records the first row index at which each group was seen, and the frame layer
gathers the key columns by those indices with the take kernel it already has.
That works for every dtype including ones with no key representation at all, and
it gets null keys right for free, because the row it points at holds the null.

For that to be possible every group has to have a row, which is not something
`factorize` promises, so `_densify` renumbers the ordinals into the order they
were first seen and discards the ones nothing carries. Its docstring has the case
that forced it.

`_densify` is a pass over every row, and most of what it was written for is no
longer true. No route produces an ordinal that no row carries, so it is not the
density that needs fixing. What it still does is put the null group where its
first null is rather than at ordinal zero, and build the representative row list.

A key with no nulls needs neither, on any of the four routes. Its ordinals are
already in first-appearance order, because every route recognizes a group by the
row that introduced it, and that row is the representative one. So the pass would
rewrite every code to the value it already held.

No key of a group by on several needs it either, first or not, whatever its
dtype and whatever its nulls, because only its group count and its ordinals as
values are read there. Where those ordinals put their null group changes which
integer a tuple packs to and changes nothing else, and the factorize at the
bottom is what fixes the ordering for the result.

Nor does the packed column itself. It is written a row at a time, so it has no
nulls at all, and its factorize's ordinals and representative rows are already
what the pass would have made of them.

What is left paying is a group by on one key that has nulls, of any dtype,
because all four routes put the null group at ordinal zero wherever its first
null is and there is no later factorize to fix it. `_factorize_any` reports what
its route knows in a `KeyCodes` and `group_ordinals` decides from that.
"""

from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray, ColumnRefs
from firepanda.array.array import Array
from firepanda.dtype.lists import ALL
from firepanda.exec.morsel import parallel_morsels

from firepanda.kernel.agg import max_of

from .factorize import (
    DIRECT_LIMIT,
    direct_plan,
    factorize,
    factorize_dense,
    factorize_strings,
)


@fieldwise_init
struct TuplePlan[dt: DType](Movable):
    """Where each key of a fused tuple pack sits inside the packed integer."""

    var bases: List[Scalar[Self.dt]]
    """Per key, the value that packs to zero, which is that key's minimum."""

    var steps: List[Int]
    """Per key, what its distance from that minimum is multiplied by. The last
    key's step is one and each earlier key's is the product of the ranges to its
    right, which is the same positional numbering the packing loop builds one
    key at a time."""

    var space: Int
    """The width of the packed range, or -1 when the fused route does not
    apply."""


struct Grouping(Movable):
    """One group ordinal per row, plus what is needed to describe the groups."""

    var codes: Array[DType.uint32]
    """The ordinal each row belongs to, in `[0, groups)`."""

    var groups: Int
    """How many distinct key tuples were seen."""

    var rows_at: List[Int]
    """For each group, the first row index that had it. Gather the key columns by
    this to get the group's key values back."""

    def __init__(
        out self,
        var codes: Array[DType.uint32],
        groups: Int,
        var rows_at: List[Int],
    ):
        """Constructs a grouping.

        Args:
            codes: The per-row ordinals.
            groups: The number of distinct ordinals.
            rows_at: A representative row per group.
        """
        self.codes = codes^
        self.groups = groups
        self.rows_at = rows_at^

    def into_codes(deinit self) -> Array[DType.uint32]:
        """Gives up the ordinals without copying them, dropping the rest.

        A join wants the ordinals and has nothing to do with the group's
        representative rows, and Mojo will not let it move `codes` out from under
        a struct that still owns `rows_at`. `deinit` says the rest is being torn
        down, which is what makes the move legal. Same arrangement as
        `Factorized.into_codes`.

        Returns:
            The per-row ordinals.
        """
        return self.codes^


def group_ordinals[
    o: ImmOrigin
](columns: ColumnRefs[o], at: List[Int], rows: Int) raises -> Grouping:
    """Assigns a dense ordinal to every distinct tuple of key values.

    Args:
        columns: The frame's columns, borrowed.
        at: Which of them are keys, in the order they should be combined.
        rows: The frame's height.

    Returns:
        The ordinals, the group count and a representative row per group.

    Raises:
        If no keys were given, if a key dtype has no physical layout, or if the
        combined ordinal space would not fit in an int64.
    """
    if len(at) == 0:
        raise Error("group by: at least one key column is required")

    # Every key an integer of the same dtype in a range narrow enough to index,
    # and the tuple packs straight out of the raw columns with no per key
    # factorize at all. The dtypes have to agree because the packing loop reads
    # all of the keys in one pass and a loop instantiated on each key's own dtype
    # would be twelve copies of it per key. They nearly always do agree, because
    # the keys of a group by are columns of one table.
    if len(at) > 1:
        var uniform = not columns[at[0]][].is_string()
        for k in range(1, len(at)):
            if (
                columns[at[k]][].is_string()
                or columns[at[k]][].dtype() != columns[at[0]][].dtype()
            ):
                uniform = False
        if uniform:
            var kind = columns[at[0]][].dtype()
            comptime for dt in ALL:
                comptime if dt.is_integral():
                    if kind == dt:
                        var plan = _tuple_plan[dt](columns, at, rows)
                        if plan.space > 0:
                            var packed = _tuple_pack[dt](
                                columns, at, rows, plan
                            )
                            return _packed_grouping(packed, plan.space)

    var first = _factorize_any(columns[at[0]][])
    var groups = first.groups
    var known = first.knows_rows()
    var codes = Array[DType.uint32](0)
    var rows_at = List[Int]()
    first^.into_parts(codes, rows_at)

    if len(at) == 1:
        # One key is the whole answer, so its ordinals are the result's and the
        # only thing that can be wrong with them is where the null group sits.
        if not known:
            rows_at = List[Int]()
            groups = _densify(codes, rows_at)
        return Grouping(codes^, groups, rows_at^)

    # Two or more. Nothing about any key's ordinals is read from here on except
    # their value, so a null group in the wrong place does not matter and
    # `_densify` is called only on a route that cannot say how many groups it
    # made. The packed column carries the tuple and the one factorize at the
    # bottom is what makes the result dense and ordered.
    #
    # Every key is factorized here rather than inside the packing loop, because
    # the fused pass below needs all of the ordinals at once and the fold needs
    # to know they exist either way. The counts come with them, and the counts
    # are what decides which of the two runs.
    var stacked = List[Array[DType.uint32]]()
    var counts = List[Int]()
    stacked.append(codes^)
    counts.append(groups)
    for k in range(1, len(at)):
        var key = _factorize_any(columns[at[k]][])
        var next_groups = key.groups
        var next = Array[DType.uint32](0)
        var spare = List[Int]()
        key^.into_parts(next, spare)
        if next_groups < 0:
            # Only the count matters here, not the order and not the rows, for
            # the same reason the first key's order does not matter. A key that
            # knows how many groups it has can skip the pass even when it cannot
            # say which row each one is on.
            spare = List[Int]()
            next_groups = _densify(next, spare)
        stacked.append(next^)
        counts.append(next_groups)

    # The multipliers of the positional notation the fold would have arrived at,
    # rightmost first, alongside the product they end at. The test is a division
    # rather than a multiply because the product is the thing that would
    # overflow, and there is nothing to test it with afterwards.
    var steps = List[Int]()
    for _ in range(len(counts)):
        steps.append(0)
    var fused = True
    var product = 1
    for k in range(len(counts) - 1, -1, -1):
        steps[k] = product
        if counts[k] < 1 or product > Int(Int64.MAX) // counts[k]:
            fused = False
            break
        product *= counts[k]

    # Four bytes a row when the product fits and eight when it does not, which
    # is the same choice the fold does not get to make. Anything that overflows
    # an int64 keeps the fold, because renumbering partway through is what
    # `_condense` is for and a single pass has no partway.
    if fused:
        if product <= Int(UInt32.MAX):
            return _packed_grouping(
                _code_pack[DType.uint32](stacked, steps, rows), product
            )
        return _packed_grouping(
            _code_pack[DType.int64](stacked, steps, rows), product
        )

    comptime lanes = simd_width_of[DType.int64]()

    # The packing passes below are elementwise over every row, and on ten
    # million rows each of them moves more memory than the factorize that
    # produced their input. They run on every core, in morsels, for the same
    # reason every other whole column loop in the engine does.
    var running = Array[DType.int64](overwritten=rows)

    def widen(begin: Int, stop: Int) raises {mut running, imm}:
        var into = running.unsafe_ptr()
        var from_ = stacked[0].unsafe_ptr()
        var i = begin
        while i + lanes <= stop:
            into.unsafe_offset(i).unsafe_store(
                from_.unsafe_offset(i)
                .unsafe_load[width=lanes]()
                .cast[DType.int64]()
            )
            i += lanes
        while i < stop:
            into.unsafe_offset(i).unsafe_store(
                Int64(from_.unsafe_offset(i).unsafe_load())
            )
            i += 1

    # The widening is not a pass of its own. The first key's ordinals are read
    # by the first packing step and by nothing else, and that step is already
    # reading a column and writing `running`, so it can widen as it goes: the
    # multiply takes the first key's ordinal straight from the uint32 and the
    # int64 is only ever written. Doing it separately reads the ordinals, writes
    # eighty megabytes, then reads those eighty megabytes back, which on ten
    # million rows is a hundred and sixty megabytes moved to arrange numbers the
    # next loop was about to load anyway. `widen` stays because the overflow
    # route below needs `running` populated before it can renumber it, and that
    # route cannot be reached without more rows than a uint32 ordinal can name.
    var space = counts[0]
    var packed = False

    for k in range(1, len(at)):
        var next_groups = counts[k]

        if next_groups > 0 and space > Int(Int64.MAX) // next_groups:
            if not packed:
                parallel_morsels(widen, rows)
                packed = True
            space = _condense(running, space)
            if space > Int(Int64.MAX) // next_groups:
                raise Error(
                    "group by: the combined key space of "
                    + String(space)
                    + " by "
                    + String(next_groups)
                    + " does not fit in an int64"
                )

        def combine(begin: Int, stop: Int) raises {mut running, imm}:
            var pack = running.unsafe_ptr()
            var right = stacked[k].unsafe_ptr()
            var i = begin
            while i + lanes <= stop:
                pack.unsafe_offset(i).unsafe_store(
                    pack.unsafe_offset(i).unsafe_load[width=lanes]()
                    * Int64(next_groups)
                    + right.unsafe_offset(i)
                    .unsafe_load[width=lanes]()
                    .cast[DType.int64]()
                )
                i += lanes
            while i < stop:
                pack.unsafe_offset(i).unsafe_store(
                    pack.unsafe_offset(i).unsafe_load() * Int64(next_groups)
                    + Int64(right.unsafe_offset(i).unsafe_load())
                )
                i += 1

        def start(begin: Int, stop: Int) raises {mut running, imm}:
            var pack = running.unsafe_ptr()
            var left = stacked[0].unsafe_ptr()
            var right = stacked[k].unsafe_ptr()
            var i = begin
            while i + lanes <= stop:
                pack.unsafe_offset(i).unsafe_store(
                    left.unsafe_offset(i)
                    .unsafe_load[width=lanes]()
                    .cast[DType.int64]()
                    * Int64(next_groups)
                    + right.unsafe_offset(i)
                    .unsafe_load[width=lanes]()
                    .cast[DType.int64]()
                )
                i += lanes
            while i < stop:
                pack.unsafe_offset(i).unsafe_store(
                    Int64(left.unsafe_offset(i).unsafe_load())
                    * Int64(next_groups)
                    + Int64(right.unsafe_offset(i).unsafe_load())
                )
                i += 1

        if packed:
            parallel_morsels(combine, rows)
        else:
            parallel_morsels(start, rows)
            packed = True
        space *= next_groups

    # Every row of `running` was written by the loop above, so it has no nulls
    # and this factorize's own ordinals and representative rows are what
    # `_densify` would have made of them. The check is on the result rather than
    # on that argument, because a route that stopped reporting either one should
    # cost a pass here and not an answer.
    #
    # `space` is passed along because it is exactly the range the packing put the
    # values in, and knowing it is what lets a group by whose key space is dense
    # index a table instead of hashing.
    return _packed_grouping(running, space)


def _packed_grouping[
    pt: DType
](packed: Array[pt], space: Int) raises -> Grouping:
    """Turns a packed tuple key into the grouping it describes.

    Args:
        packed: One packed key per row, every value in `[0, space)`.
        space: The width of that range.

    Parameters:
        pt: The packed column's dtype.

    Returns:
        The dense ordinals, the group count and a representative row per group.

    Raises:
        If the factorize raises.
    """
    var combined = factorize_dense(packed, space)
    var combined_groups = combined.count()
    var combined_firsts = List[Int]()
    var out = Array[DType.uint32](0)
    combined^.into_parts(out, combined_firsts)
    if len(combined_firsts) == combined_groups:
        return Grouping(out^, combined_groups, combined_firsts^)
    var out_rows = List[Int]()
    var out_groups = _densify(out, out_rows)
    return Grouping(out^, out_groups, out_rows^)


def _code_pack[
    pt: DType, o: ImmOrigin
](
    ref[o] stacked: List[Array[DType.uint32]], steps: List[Int], rows: Int
) raises -> Array[pt]:
    """Packs every key tuple into one integer in a single pass over the ordinals.

    The weighted sum this computes is the number the fold arrives at, written out
    rather than accumulated, so the two routes produce the same packed value for
    the same row and the tests can hold one against the other.

    Args:
        stacked: One column of ordinals per key, all of them `rows` long.
        steps: What each key's ordinal is multiplied by, rightmost step one.
        rows: The frame's height.

    Parameters:
        pt: What to pack into, which the caller picks from the product.
        o: The origin the ordinal columns are borrowed from.

    Returns:
        One packed key per row, every value below the product of the counts.
    """
    comptime lanes = simd_width_of[pt]()

    # The pointers are collected once rather than indexed per morsel, and they
    # are cast to the origin the list was borrowed from, which is where they all
    # really live.
    var keys = List[Pointer[UInt32, o]]()
    for k in range(len(stacked)):
        keys.append(stacked[k].unsafe_ptr().unsafe_origin_cast[o]())

    # Every row is written by the loop below and the packed value is a number in
    # a range, never absent, so the zero fill would be a pass for nothing.
    var packed = Array[pt](overwritten=rows)

    def body(begin: Int, stop: Int) raises {mut packed, imm}:
        var into = packed.unsafe_ptr()
        var i = begin
        while i + lanes <= stop:
            # The accumulator is the packed column's own dtype, and the caller
            # chose that from the product every partial sum stays below, so
            # nothing here can overflow.
            var acc = SIMD[pt, lanes](0)
            for k in range(len(keys)):
                acc += keys[k].unsafe_offset(i).unsafe_load[width=lanes]().cast[
                    pt
                ]() * Scalar[pt](steps[k])
            into.unsafe_offset(i).unsafe_store(acc)
            i += lanes
        while i < stop:
            var acc = Scalar[pt](0)
            for k in range(len(keys)):
                acc += keys[k].unsafe_offset(i).unsafe_load().cast[
                    pt
                ]() * Scalar[pt](steps[k])
            into.unsafe_offset(i).unsafe_store(acc)
            i += 1

    parallel_morsels(body, rows)
    return packed^


def _tuple_plan[
    dt: DType, o: ImmOrigin
](columns: ColumnRefs[o], at: List[Int], rows: Int) raises -> TuplePlan[dt]:
    """Works out whether the keys can be packed without factorizing them first.

    A key qualifies when it has no nulls and `direct_plan` gives it a range, and
    the tuple qualifies when the product of those ranges is a space the factorize
    at the bottom will lay a table over rather than hash. That bound is the wider
    of `DIRECT_LIMIT` and the row count, which is `factorize_dense`'s own rule,
    capped at what a uint32 can hold so the packed column is four bytes a row
    rather than eight.

    The scans are the cost of asking, and they are wasted on a key list that
    turns out not to qualify, so the asking is arranged to be cheap when the
    answer is no. What each key is allowed to spend is not the fixed ceiling
    `factorize` uses but what the tuple has left, `bound` divided by the product
    so far, and `direct_plan` returns the moment a key passes the ceiling it was
    given. So a second key wide enough to put the tuple out of reach is declined
    within a few thousand rows rather than after a pass over all of them, and
    the same holds for every key after it. Only the first key can cost a full
    scan for nothing, whatever the key count.

    Args:
        columns: The frame's columns, borrowed.
        at: Which of them are keys.
        rows: The frame's height.

    Parameters:
        dt: The dtype all of the keys share.
        o: The origin the columns are borrowed from.

    Returns:
        The plan, whose space is -1 when the fused route does not apply.

    Raises:
        If a column cannot be viewed at `dt`.
    """
    var bound = rows if rows > DIRECT_LIMIT else DIRECT_LIMIT
    if bound > Int(UInt32.MAX):
        bound = Int(UInt32.MAX)

    var bases = List[Scalar[dt]]()
    var spans = List[Int]()
    var space = 1
    for k in range(len(at)):
        ref col = columns[at[k]][].as_typed_view[dt]()
        if col.null_count() > 0:
            return TuplePlan[dt](List[Scalar[dt]](), List[Int](), -1)
        # The division rather than a multiply, because the product is what would
        # overflow and there is nothing to test it with afterwards. It is also
        # the ceiling the scan is given, so the scan stops itself.
        var plan = direct_plan[dt](col, bound // space)
        if plan.span < 1 or plan.span > bound // space:
            return TuplePlan[dt](List[Scalar[dt]](), List[Int](), -1)
        bases.append(plan.base)
        spans.append(plan.span)
        space *= plan.span

    var steps = List[Int]()
    for _ in range(len(at)):
        steps.append(0)
    var step = 1
    for k in range(len(at) - 1, -1, -1):
        steps[k] = step
        step *= spans[k]
    return TuplePlan[dt](bases^, steps^, space)


def _tuple_pack[
    dt: DType, o: ImmOrigin
](
    columns: ColumnRefs[o], at: List[Int], rows: Int, plan: TuplePlan[dt]
) raises -> Array[DType.uint32]:
    """Packs every key tuple into one integer in a single pass over the columns.

    Args:
        columns: The frame's columns, borrowed.
        at: Which of them are keys.
        rows: The frame's height.
        plan: What to subtract from each key and what to multiply it by.

    Parameters:
        dt: The dtype all of the keys share.
        o: The origin the columns are borrowed from.

    Returns:
        One packed key per row, every value in `[0, plan.space)`.

    Raises:
        If a column cannot be viewed at `dt`.
    """
    comptime lanes = simd_width_of[DType.int64]()

    # The pointers are collected once rather than viewed per morsel, and they are
    # cast to the origin the columns were borrowed from, which is the one they
    # all really live in.
    var keys = List[Pointer[Scalar[dt], o]]()
    for k in range(len(at)):
        ref col = columns[at[k]][].as_typed_view[dt]()
        keys.append(col.unsafe_ptr().unsafe_origin_cast[o]())

    # Every row is written by the loop below and the packed value is a number in
    # a range, never absent, so the zero fill would be a pass for nothing.
    var packed = Array[DType.uint32](overwritten=rows)

    def body(begin: Int, stop: Int) raises {mut packed, imm}:
        var into = packed.unsafe_ptr()
        var i = begin
        while i + lanes <= stop:
            # int64 rather than the key's own dtype, because the distance from
            # the minimum can leave a narrow dtype's range and the product with
            # the step certainly can. `direct_plan` will not hand back a range
            # whose ends are past forty bits, so the accumulator cannot overflow.
            var acc = SIMD[DType.int64, lanes](0)
            for k in range(len(keys)):
                acc += (
                    keys[k]
                    .unsafe_offset(i)
                    .unsafe_load[width=lanes]()
                    .cast[DType.int64]()
                    - plan.bases[k].cast[DType.int64]()
                ) * Int64(plan.steps[k])
            into.unsafe_offset(i).unsafe_store(acc.cast[DType.uint32]())
            i += lanes
        while i < stop:
            var acc = Int64(0)
            for k in range(len(keys)):
                acc += (
                    keys[k].unsafe_offset(i).unsafe_load().cast[DType.int64]()
                    - plan.bases[k].cast[DType.int64]()
                ) * Int64(plan.steps[k])
            into.unsafe_offset(i).unsafe_store(acc.cast[DType.uint32]())
            i += 1

    parallel_morsels(body, rows)
    return packed^


def _condense(mut running: Array[DType.int64], space: Int) raises -> Int:
    """Renumbers the running key into the tuples it actually holds.

    Called only when the next key would push the packed value past an int64,
    which needs a real column to reach: six keys of a thousand values each get
    nowhere near it, because the running space is the product of the group
    counts and int64 holds nineteen digits of that.

    Args:
        running: The packed key, rewritten in place.
        space: The range the packing has put its values in.

    Returns:
        The number of distinct tuples, which is the new bound on its values.

    Raises:
        If the factorize does.
    """
    var found = factorize_dense(running, space)
    var groups = found.count()
    var codes = Array[DType.uint32](0)
    var spare = List[Int]()
    found^.into_parts(codes, spare)

    comptime lanes = simd_width_of[DType.int64]()
    var rows = len(running)

    def widen(begin: Int, stop: Int) raises {mut running, imm}:
        var into = running.unsafe_ptr()
        var from_ = codes.unsafe_ptr()
        var i = begin
        while i + lanes <= stop:
            into.unsafe_offset(i).unsafe_store(
                from_.unsafe_offset(i)
                .unsafe_load[width=lanes]()
                .cast[DType.int64]()
            )
            i += lanes
        while i < stop:
            into.unsafe_offset(i).unsafe_store(
                Int64(from_.unsafe_offset(i).unsafe_load())
            )
            i += 1

    parallel_morsels(widen, rows)
    return groups


struct KeyCodes(Movable):
    """One key column's ordinals, and what its factorize already knows of them.
    """

    var codes: Array[DType.uint32]
    """The ordinal each row was given."""

    var groups: Int
    """How many ordinals are in use. Never -1 today, because both numeric routes
    and the string one hand out ordinals only to rows, but the field is read
    rather than assumed so a route that cannot say has somewhere to say it."""

    var firsts: List[Int]
    """The first row of each group in ordinal order, or empty if the route did
    not produce one. Empty is not the same as unknown for a column with no rows,
    but the two want the same handling, so nothing distinguishes them."""

    def __init__(
        out self,
        var codes: Array[DType.uint32],
        groups: Int,
        var firsts: List[Int],
    ):
        """Constructs a result.

        Args:
            codes: The per-row ordinals.
            groups: The group count, or -1 if the ordinals may be sparse.
            firsts: A representative row per group, or empty.
        """
        self.codes = codes^
        self.groups = groups
        self.firsts = firsts^

    def into_parts(
        deinit self, mut codes: Array[DType.uint32], mut firsts: List[Int]
    ):
        """Gives up both lists, since a caller that wants one usually wants both.

        Two arguments rather than a return value because a struct cannot have two
        of its fields moved out one at a time, and unpacking a returned tuple
        copies.

        Args:
            codes: Overwritten with the per-row ordinals.
            firsts: Overwritten with the representative rows, which may be empty.
        """
        codes = self.codes^
        firsts = self.firsts^

    def knows_rows(self) -> Bool:
        """Whether these ordinals can be used as they are, with no renumbering.

        Both halves have to hold. A sparse count means an ordinal with no row
        behind it and an aggregation row nobody asked for. A missing or partial
        representative list means there is no key value to report for some group,
        which is the case for any column with nulls: every route returns a row
        for each non-null group and puts the null group at ordinal zero, so the
        list is one short and the ordinals are no longer in first-seen order
        either.

        Returns:
            Whether `groups` and `firsts` between them describe every group.
        """
        return self.groups >= 0 and len(self.firsts) == self.groups


def _factorize_any(col: AnyArray) raises -> KeyCodes:
    """Factorizes a column whose dtype is a runtime value.

    This used to make a typed copy of the whole column, because `factorize` takes
    an `Array[dt]` and a borrowed `AnyArray` had no way to produce one without
    copying. On a forty megabyte key column that was about a third of the
    ordinals, and a group by on six keys paid it six times. It reads through
    `as_typed_view` now, which borrows.

    Args:
        col: The key column.

    Returns:
        One ordinal per row, with the group count and the representative rows
        when the route that produced them knows them.

    Raises:
        If the dtype has no physical layout.
    """
    # Before the dispatch, because uint8 is in ALL and a string column would
    # match it and group on the first byte of each view, which puts every name
    # starting with the same letter in one group.
    if col.is_string():
        var text = factorize_strings(col.strings())
        var groups = text.count()
        # With no nulls the ordinals are the merge's, which are handed out in
        # first-appearance order, and `firsts` is the row list the merge built to
        # compare keys with. That is exactly what `_densify` would have produced.
        # With nulls the null group takes ordinal zero regardless of where its
        # first null is, and `firsts` covers the other groups only, so the pass
        # is still needed.
        if text.null_group >= 0:
            return KeyCodes(text^.into_codes(), groups, List[Int]())
        var text_codes = Array[DType.uint32](0)
        var text_firsts = List[Int]()
        text^.into_parts(text_codes, text_firsts)
        return KeyCodes(text_codes^, groups, text_firsts^)

    comptime for candidate in ALL:
        if col.dtype() == candidate:
            # Same shape as the string branch above, and for the same reason. All
            # three numeric routes recognize a group by the row that introduced
            # it, so `firsts` is what one thread would have picked and is already
            # in ordinal order. Nulls are the exception again: they take ordinal
            # zero wherever their first one is, and `firsts` does not cover them.
            ref view = col.as_typed_view[candidate]()
            var found = factorize(view)
            var groups = found.count()
            if found.null_group >= 0:
                return KeyCodes(found^.into_codes(), groups, List[Int]())
            var found_codes = Array[DType.uint32](0)
            var found_firsts = List[Int]()
            found^.into_parts(found_codes, found_firsts)
            return KeyCodes(found_codes^, groups, found_firsts^)
    raise Error("group by: unsupported key dtype")


def _densify(
    mut codes: Array[DType.uint32], mut rows_at: List[Int]
) raises -> Int:
    """Renumbers ordinals into first-seen order and collects a row for each.

    This was written because `factorize` did not promise that every ordinal it
    can produce belongs to a row, and it dropped the ones that did not. It does
    promise that now, on every route, so dropping is no longer the reason to call
    it. Two reasons are left and both are real.

    It puts the null group where its first null is. Both factorizes fix the null
    group at ordinal zero, because that keeps the ordinals the table hands out
    and the positions in the key list a constant apart. First appearance order is
    what `sort=False` means at the frame layer, which is what pandas and Polars
    both do, and a null group is a group like any other there.

    And it fills the representative row table, which is how the frame layer gets
    a group's key values back. Every route returns one of those now, covering its
    non-null groups, so a first key with no nulls skips the pass entirely and a
    first key with nulls is the only thing left calling it for either reason.

    Args:
        codes: The ordinals, rewritten in place.
        rows_at: Filled with the first row index of each ordinal, in order.

    Returns:
        The number of ordinals that at least one row carries.

    Raises:
        Error: Only what the morsel runtime raises, through `max_of`.
    """
    var at = codes.unsafe_ptr()
    var n = len(codes)
    var top = -1
    if n > 0:
        var found = max_of(codes)
        if found.valid:
            top = Int(found.value)

    var remap = List[Int](capacity=top + 1)
    for _ in range(top + 1):
        remap.append(-1)

    var groups = 0
    for i in range(n):
        var raw = Int(at.unsafe_offset(i).unsafe_load())
        if remap[raw] < 0:
            remap[raw] = groups
            rows_at.append(i)
            groups += 1
        at.unsafe_offset(i).unsafe_store(UInt32(remap[raw]))
    return groups
