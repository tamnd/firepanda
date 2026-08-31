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
That integer is then factorized itself, which makes it dense again, and the
process repeats for the third key and the fourth.

Refactorizing after every combine is what keeps this from overflowing. Without
it, the running product is `g0 * g1 * g2 * ...`, which is the size of the full
cross product and reaches int64 on four columns of a thousand values each. With
it, the running ordinal count is the number of key tuples actually observed,
which can never exceed the row count, so the product being packed is bounded by
`rows * g_next` and stays inside int64 for any table that fits on a machine. The
bound is checked anyway, because "cannot happen" is not a thing to rely on in the
one function every group by goes through.

There is a real cost to this and it should not be hidden: `n` keys means `n`
factorize passes plus `n - 1` more over the combined column. A single fused hash
of the tuple would be one pass. The reason to start here is that the composition
reuses the direct route, so grouping on three small integer columns never hashes
anything at all, and a fused tuple hash would hash all three. Which of those wins
is a measurement, and the benchmark that settles it is `group/ordinals_two_keys`
against `group/ordinals_one_key`.

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

A key that is not the first one needs neither either, whatever its dtype and
whatever its nulls, because only its group count is read here. The packed column
is factorized again and that is what fixes the ordinals for the result.

Nor does the packed column itself. It is written a row at a time from two code
arrays, so it has no nulls at all, and its factorize's ordinals and representative
rows are already what the pass would have made of them. That is one pass per key
after the first, which is five of them on a six key group by.

What is left paying is a first key with nulls, of any dtype, because all four
routes put the null group at ordinal zero wherever its first null is.
`_factorize_any` reports what its route knows in a `KeyCodes` and `group_ordinals`
decides from that.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.lists import ALL

from firepanda.kernel.agg import max_of

from .factorize import factorize, factorize_strings


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


def group_ordinals(
    columns: List[AnyArray], at: List[Int], rows: Int
) raises -> Grouping:
    """Assigns a dense ordinal to every distinct tuple of key values.

    Args:
        columns: The frame's columns.
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

    var first = _factorize_any(columns[at[0]])
    var groups = first.groups
    var known = first.knows_rows()
    var codes = Array[DType.uint32](0)
    var rows_at = List[Int]()
    first^.into_parts(codes, rows_at)
    if not known:
        rows_at = List[Int]()
        groups = _densify(codes, rows_at)

    for k in range(1, len(at)):
        var key = _factorize_any(columns[at[k]])
        var next_groups = key.groups
        var next = Array[DType.uint32](0)
        var spare = List[Int]()
        key^.into_parts(next, spare)
        if next_groups < 0:
            # Only the count matters here, not the order and not the rows. The
            # packed column is factorized again below and that is what fixes the
            # ordinals for the result, so a key that knows how many groups it has
            # can skip the pass even when it cannot say which row each one is on.
            spare = List[Int]()
            next_groups = _densify(next, spare)
        if next_groups > 0 and groups > Int(Int64.MAX) // next_groups:
            raise Error(
                "group by: the combined key space of "
                + String(groups)
                + " by "
                + String(next_groups)
                + " does not fit in an int64"
            )

        # Pack the pair, then make it dense again. The packing is exact and the
        # refactorize is what stops the product from compounding across keys.
        var packed = Array[DType.int64](rows)
        var left = codes.unsafe_ptr()
        var right = next.unsafe_ptr()
        var into = packed.unsafe_ptr()
        for i in range(rows):
            into.unsafe_offset(i).unsafe_store(
                Int64(left.unsafe_offset(i).unsafe_load()) * Int64(next_groups)
                + Int64(right.unsafe_offset(i).unsafe_load())
            )
        # Every row of `packed` was just written, so it has no nulls and the
        # factorize's own ordinals and representative rows are what the pass
        # would have produced. The check is on the result rather than on that
        # argument, because a route that stopped reporting either one should
        # cost a pass here and not an answer.
        var combined = factorize(packed)
        var combined_groups = combined.count()
        var combined_firsts = List[Int]()
        codes = Array[DType.uint32](0)
        combined^.into_parts(codes, combined_firsts)
        if len(combined_firsts) == combined_groups:
            groups = combined_groups
            rows_at = combined_firsts^
        else:
            rows_at = List[Int]()
            groups = _densify(codes, rows_at)

    return Grouping(codes^, groups, rows_at^)


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

    The typed copy this takes is the one avoidable cost in the whole group by and
    it is here because `factorize` reduces the column with `min_of` and `max_of`
    to decide whether it can skip hashing, and those take an `Array` rather than a
    pointer. Rewriting that path to pointer form is worth doing and is not worth
    doing in the change that introduces group by.

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
            var found = factorize(col.as_typed[candidate]())
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
