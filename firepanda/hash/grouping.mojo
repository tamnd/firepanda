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

from .factorize import factorize, factorize_dense, factorize_strings


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

    # Two or more. Nothing about the first key's ordinals is read from here on
    # except their value, so a null group in the wrong place does not matter and
    # `_densify` is not called on it. The packed column carries the tuple and the
    # one factorize at the bottom is what makes the result dense and ordered.
    comptime lanes = simd_width_of[DType.int64]()

    # The packing passes below are elementwise over every row, and on ten
    # million rows each of them moves more memory than the factorize that
    # produced their input. They run on every core, in morsels, for the same
    # reason every other whole column loop in the engine does.
    var running = Array[DType.int64](overwritten=rows)

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
    # Nothing below reads `codes`, so the pass above is its last use and the
    # forty megabytes go back here rather than at the end of the function. That
    # matters because the factorize at the bottom allocates a table sized by the
    # tuple count. Each later key's codes are released by the iteration that
    # made them for the same reason.
    var space = groups

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

        if next_groups > 0 and space > Int(Int64.MAX) // next_groups:
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
            var right = next.unsafe_ptr()
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

        parallel_morsels(combine, rows)
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
    var combined = factorize_dense(running, space)
    var combined_groups = combined.count()
    var combined_firsts = List[Int]()
    var out = Array[DType.uint32](0)
    combined^.into_parts(out, combined_firsts)
    if len(combined_firsts) == combined_groups:
        return Grouping(out^, combined_groups, combined_firsts^)
    var out_rows = List[Int]()
    var out_groups = _densify(out, out_rows)
    return Grouping(out^, out_groups, out_rows^)


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


def _densify(mut codes: Array[DType.uint32], mut rows_at: List[Int]) -> Int:
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
