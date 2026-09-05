"""Row labels, and the arithmetic range that most frames have instead of them.

pandas identifies a row by a label rather than by a position, and every operation
that chooses rows carries the labels of the rows it chose. `df.tail(5)` of a ten row
frame comes back labelled 5 through 9, `sort_values` comes back permuted, and
`dropna` comes back with holes in it. firepanda had none of that: a frame was a
schema, some columns and a height, and the answer to "which row is this" was its
position and nothing else. This is the smallest index that fixes it.

The whole design is in one sentence. An index is either an arithmetic range, which is
a start and a length and no memory at all, or an array of labels, and every frame
starts with the first kind and only pays for the second when something takes it apart.
A frame read from a file, built from columns or sliced from the front is a range. A
frame that has been gathered, filtered or sorted is not, because the labels it kept
are no longer consecutive.

That split is not an optimization bolted on afterwards, it is the reason this can
exist at all. Every frame in the library grows an index field, including the ones in
the inner loop of a benchmark, so what it costs when nobody asked for it is the
first question and not the last one. A range is two integers and it copies in two
moves, and `slice` on a range gives another range without touching a byte, so
`head`, `tail` and every slice really are free.

`take` and `filter` are not free and cannot be. Labels that survive a gather have to
be written down, so the honest description is that a frame carrying an index has one
more column in it than it looks like it has. It is not a fixed overhead, it is one
column, so what it costs on your data depends on how many columns you already had,
and a twenty column frame pays a twentieth of what a one column frame pays.

What that comes to was measured rather than reasoned about, and the reasoning would
have got it wrong in both directions. On a three column frame of a million rows,
against a `frame/slice_half` control that this change does not touch and that moved
4.6 per cent between the two binaries anyway, `frame/take` moved 2.8 per cent and
`frame/filter` moved 16.8 per cent. So a gather is at the noise floor and a filter is
not, which is the opposite of the guess that one more column costs one more column's
worth on both. The two kernel rows say why. `kernel/take_range` runs in 0.794 ms
against `kernel/take_scattered` at 2.459, because a gather is a random read a row and
a label needs no read at all, so a fourth column of labels is worth about a third of a
fourth column of data and it goes on every core beside the other three. A compaction
reads sequentially and was being prefetched anyway, so dropping the read buys nothing
and `kernel/filter_range` lands level with `kernel/filter`, 1.708 ms against 1.768,
which makes a fourth set of labels cost very nearly a fourth column. Every number here
is a minimum rather than a mean, over twenty interleaved runs for the frame pair and
eight for the kernel pair, taken on a machine that was busy throughout. That is why
the control is quoted beside them and why a minimum is the right statistic: contention
can only add time, so the smallest run is the closest thing to a quiet one. Read 5 per
cent as the floor and the filter number as the only one clearly above it.

Two earlier versions were much worse than one column and both are worth keeping,
because both were the tidy thing to write. The first materialized the whole range into
an array and gathered that, which costs an allocation and a pass proportional to the
height going in rather than the height coming out, and took `frame/take` from 5.96 ms
to 15.16. The second fixed that by writing `start + at` straight into the output, and
was still 46 per cent on `frame/take` and 129 per cent on `frame/filter`, because
writing the loop here in the obvious way meant a bitmap read-modify-write per null, an
unhoisted bounds-checked read per row, and, worst of the three, a serial pass bolted
onto a gather that the kernel had gone to some trouble to put on every core. The loops
live in `take_range` and `filter_range` in `kernel/select.mojo` now, next to the
general forms they shadow and written in the same style, which is where they should
have been put first. The lesson is not about ranges. A frame level loop that runs
beside a kernel inherits the kernel's standards, and this one did not meet them.

Deferring instead was considered and not done. An index could hold the mask a filter
was given rather than the labels it implies, which is a byte a row against eight and
would make a filter nearly free until something read the labels. It needs a third
representation and composition rules for every operation that can follow a filter,
which is all of them, and the whole prize is one column. If the profile ever says that
column matters, that is the design to reach for.

The distinction pandas draws between `RangeIndex(0, n)` and everything else matters
here too, because a frame whose labels are zero to n minus one carries no information
and compares equal to a frame with no index at all. `is_default` is that question,
and it is deliberately narrower than "is a range": `tail` produces the range 5 to 9,
which is a range and is not default, and the difference is the whole of what those
cases are testing.

The lookups are the second thing written here and they are what makes the index an
object rather than a field. `loc` is `get_loc`, `reindex` is `get_indexer`, a merge on
an index is `get_indexer_non_unique`, and `align` is `equals` and `get_indexer` in that
order. Writing any of those four before the lookups exist means writing a lookup inside
each of them and then deleting three.

All of them are built on one idea. Put the index's labels and the labels being looked
up into one array, factorize that array once, and two labels are the same label exactly
when they came out with the same ordinal. That is one pass over each side rather than a
hash table built for the occasion and a probe per row, it reuses the factorize that the
group by spent a great deal of effort on, and it gets the null rule right for free: the
factorize puts every null in one group, so a null in the index is found by a null in the
target, which is what pandas does and is the one place in the library where a missing
value equals a missing value.

A range does not do any of that. The position of a label in a range starting at `start`
is `label - start`, so a lookup against the index that most frames have is arithmetic
and touches no memory, which is the same argument the representation was chosen for.

What this still does not do is align. There is no `union`, no `reindex` and no `loc`,
and the four set operations are the other half of
https://github.com/tamnd/firepanda/issues/154. What is here is what those are made of.
"""

from std.collections import Optional

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.lists import ALL
from firepanda.hash.function import key_bits
from firepanda.hash.grouping import KeyCodes, factorize_any
from firepanda.kernel.concat import concat_two_any
from firepanda.kernel.select import (
    filter_any,
    filter_range,
    take_any,
    take_range,
)
from firepanda.kernel.sort import is_sorted_any


comptime NOT_FOUND = Int64(-1)
"""What a lookup writes where a label is not in the index.

pandas uses the same value, and it is a sentinel rather than a null because a
caller passes an indexer straight to a gather and a gather already reads a
negative position as a row that is not there.
"""


struct Matches(Movable):
    """Every position a set of labels was found at, and the ones found nowhere.

    This is what `get_indexer_non_unique` returns and it is a pair rather than
    one array because the pair cannot be recovered from either half. Once a
    label may match several rows the answer is no longer as long as the target
    was, so a caller cannot tell which of its labels produced which run of
    positions, and the labels that produced nothing are the ones it most often
    wants.
    """

    var positions: Array[DType.int64]
    """Every matching position, in target order, with a `NOT_FOUND` standing in
    for a label that matched nothing so that no target label is silent."""

    var missing: Array[DType.int64]
    """Which rows of the target matched nothing, as positions in the target."""

    def __init__(
        out self,
        var positions: Array[DType.int64],
        var missing: Array[DType.int64],
    ):
        """Constructs a result.

        Args:
            positions: Every matching position, in target order.
            missing: The target rows that matched nothing.
        """
        self.positions = positions^
        self.missing = missing^


def _same_label(a: AnyArray, i: Int, b: AnyArray, j: Int) raises -> Bool:
    """Reports whether two rows hold the same label.

    Deliberately not the join's `_same_key`, which answers False when either
    side is null because a null key joins to nothing. A label is not a key. Two
    missing labels are the same label to pandas, `Index([nan]).equals(...)` says
    so, and an index equality that answered False for a pair of nulls would call
    an index unequal to itself.

    Args:
        a: The first column.
        i: The row in it.
        b: The second column.
        j: The row in it.

    Returns:
        Whether the two labels are the same, with null equal to null.

    Raises:
        Error: If the dtypes differ or have no physical layout.
    """
    if a.dtype() != b.dtype() or a.is_string() != b.is_string():
        raise Error(
            String(
                "index: labels must have the same dtype; got ",
                a.dtype(),
                " and ",
                b.dtype(),
            )
        )
    var a_here = a.is_valid(i)
    if a_here != b.is_valid(j):
        return False
    if not a_here:
        return True
    if a.is_string():
        ref mine = a.strings()
        ref theirs = b.strings()
        return mine.equals(i, theirs.unsafe_bytes(j))
    comptime for candidate in ALL:
        if a.dtype() == candidate:
            return key_bits(
                a.unsafe_ptr[candidate]().unsafe_offset(i).unsafe_load()
            ) == key_bits(
                b.unsafe_ptr[candidate]().unsafe_offset(j).unsafe_load()
            )
    raise Error(String("index: unsupported label dtype ", a.dtype()))


def _space_of(codes: KeyCodes) -> Int:
    """How many ordinals a factorize may have handed out.

    `KeyCodes.groups` is the count when the route that produced it knows one, and
    it is documented as being able to say it does not. A table sized by the row
    count is always big enough and is what a route that cannot say leaves as the
    only safe answer, so this asks for the count and falls back rather than
    trusting a field that has a way to be absent.

    Args:
        codes: What the factorize returned.

    Returns:
        A size that every ordinal in `codes.codes` is a valid position in.
    """
    if codes.groups >= 0:
        return codes.groups
    return len(codes.codes)


def _filled(length: Int, value: Int64) -> Array[DType.int64]:
    """Builds an int64 array with the same number in every row.

    Args:
        length: How many rows.
        value: What to put in each.

    Returns:
        The array, with every row present.
    """
    var out = Array[DType.int64](overwritten=length)
    for i in range(length):
        out[i] = value
    return out^


struct Index(Copyable, Movable, Sized):
    """The labels of a frame's rows, as a range or as an array."""

    var name: Optional[String]
    """What the level is called. `None` is unnamed, which is what a frame that
    was never grouped or reindexed has, and is different from a level named with
    the empty string."""

    var labels: Optional[AnyArray]
    """The labels, when they have been materialized. `None` means this is the
    arithmetic range `start` to `start + length - 1` and no array exists."""

    var start: Int
    """The first label, when this is a range. Meaningless otherwise."""

    var length: Int
    """The number of rows, either way. Kept as a field rather than read off
    `labels` so that a range does not have to answer `__len__` differently from
    an array."""

    def __init__(out self, length: Int):
        """Constructs the default index for a frame of a given height.

        Args:
            length: The number of rows.
        """
        self.name = None
        self.labels = None
        self.start = 0
        self.length = length

    def __init__(out self, start: Int, length: Int, var name: Optional[String]):
        """Constructs a range index that does not have to start at zero.

        Args:
            start: The first label.
            length: The number of rows.
            name: The level name, or `None` for unnamed.
        """
        self.name = name^
        self.labels = None
        self.start = start
        self.length = length

    def __init__(out self, var labels: AnyArray, var name: Optional[String]):
        """Constructs an index over labels that have already been built.

        Args:
            labels: The labels. Consumed, with no copy of the buffers.
            name: The level name, or `None` for unnamed.
        """
        self.length = len(labels)
        self.name = name^
        self.labels = Optional[AnyArray](labels^)
        self.start = 0

    def __init__(out self, *, copy: Self):
        """Deep-copies an index.

        Args:
            copy: The index to copy.
        """
        self.name = Optional[String](copy=copy.name)
        self.labels = Optional[AnyArray](copy=copy.labels)
        self.start = copy.start
        self.length = copy.length

    def __len__(self) -> Int:
        """The number of rows this index labels.

        Returns:
            The length.
        """
        return self.length

    def is_range(self) -> Bool:
        """Whether the labels are still an arithmetic range rather than an array.

        Returns:
            True when no labels have been materialized.
        """
        return not self.labels

    def is_default(self) -> Bool:
        """Whether the labels are zero to n minus one and so carry no information.

        This is narrower than `is_range` on purpose. The index `tail` leaves behind
        is a range and is not this, and telling those apart is most of the point of
        having labels at all.

        Returns:
            True when this is the range starting at zero.
        """
        return self.is_range() and self.start == 0

    def materialize(self) raises -> AnyArray:
        """The labels as an array, building the range out if it is still a range.

        Every label is present, including the ones a range produces, so nothing
        here can return a null. A gathered index can hold one and that comes from
        `take`, which is the only thing in this file that writes one.

        Returns:
            The labels, as int64 when they came from a range.

        Raises:
            If the array cannot be built.
        """
        if self.labels:
            return AnyArray(copy=self.labels.value())
        var out = Array[DType.int64](overwritten=self.length)
        var base = Int64(self.start)
        for i in range(self.length):
            out.store[1](i, base + Int64(i))
        return AnyArray(out^)

    def renamed(self, var name: Optional[String]) raises -> Self:
        """Returns this index under a different level name.

        Args:
            name: The new name, or `None` for unnamed.

        Returns:
            A copy carrying the new name.
        """
        var out = Self(copy=self)
        out.name = name^
        return out^

    def slice(self, start: Int, end: Int) raises -> Self:
        """Returns the labels of a half-open range of rows.

        A range stays a range, which is what makes `head` and `tail` cost nothing
        at all, and is the reason the range case exists.

        Args:
            start: The first row, inclusive.
            end: The last row, exclusive.

        Returns:
            The index of those rows.

        Raises:
            If the labels cannot be sliced.
        """
        if self.is_range():
            return Self(
                self.start + start,
                end - start,
                Optional[String](copy=self.name),
            )
        return Self(
            self.labels.value().slice(start, end),
            Optional[String](copy=self.name),
        )

    def take(self, indices: List[Int]) raises -> Self:
        """Returns the labels of rows gathered by position.

        Args:
            indices: The rows to gather. A negative index produces a null label,
                the way it produces a null row everywhere else.

        Returns:
            The index of those rows.

        Raises:
            If the labels cannot be gathered.
        """
        if not self.is_range():
            return Self(
                take_any(self.labels.value(), indices),
                Optional[String](copy=self.name),
            )

        return Self(
            AnyArray(take_range(self.start, indices)),
            Optional[String](copy=self.name),
        )

    def filter(self, mask: Array[DType.bool]) raises -> Self:
        """Returns the labels of the rows a mask keeps.

        Args:
            mask: The mask. Must be as long as this index.

        Returns:
            The index of the kept rows.

        Raises:
            If the labels cannot be filtered.
        """
        if not self.is_range():
            return Self(
                filter_any(self.labels.value(), mask),
                Optional[String](copy=self.name),
            )

        return Self(
            AnyArray(filter_range(self.start, mask)),
            Optional[String](copy=self.name),
        )

    def is_unique(self) raises -> Bool:
        """Whether no label appears twice.

        A range is unique without being looked at, which is the answer for every
        frame that has not been gathered or concatenated and is the reason this
        is cheap in the case that matters.

        Nothing is remembered. An index is copied into every frame that derives
        from it and a cached answer would have to be invalidated by `take` and
        `filter`, which is a correctness problem in exchange for a factorize.
        When a caller asks this in a loop it should hold the answer itself.

        Returns:
            True when the labels are all different, counting two nulls as the
            same label.

        Raises:
            Error: If the label dtype cannot be factorized.
        """
        if self.is_range():
            return True
        if self.length <= 1:
            return True
        return factorize_any(self.labels.value()).groups == self.length

    def has_duplicates(self) raises -> Bool:
        """Whether some label appears twice. The other spelling pandas has.

        Returns:
            The opposite of `is_unique`.

        Raises:
            Error: If the label dtype cannot be factorized.
        """
        return not self.is_unique()

    def is_monotonic_increasing(self) raises -> Bool:
        """Whether the labels never decrease.

        A range never decreases, since the step is one. Labels are scanned, and
        an index holding a null is not monotonic, which is what pandas says and
        what `Series` already says about a column.

        Returns:
            True if every label is at least the one before it.

        Raises:
            Error: If the label dtype is not one firepanda can order.
        """
        if self.is_range():
            return True
        ref values = self.labels.value()
        if values.null_count() > 0:
            return False
        return is_sorted_any(values, descending=False)

    def is_monotonic_decreasing(self) raises -> Bool:
        """Whether the labels never increase.

        A range of two or more never satisfies this, because its step is one and
        it has no descending form. A range of one or none satisfies both, which
        is what pandas answers for an empty index and is worth stating because
        the obvious implementation of a range fast path gets it wrong.

        Returns:
            True if every label is at most the one before it.

        Raises:
            Error: If the label dtype is not one firepanda can order.
        """
        if self.is_range():
            return self.length <= 1
        ref values = self.labels.value()
        if values.null_count() > 0:
            return False
        return is_sorted_any(values, descending=True)

    def _range_indexer(self, target: AnyArray) raises -> Array[DType.int64]:
        """Looks labels up in a range, by subtracting rather than by searching.

        Args:
            target: The labels to look for. Must be int64, which is what a range
                materializes to.

        Returns:
            One position per target row, `NOT_FOUND` where the label is outside
            the range.

        Raises:
            Error: If the target is not int64.
        """
        ref labels = target.as_typed_view[DType.int64]()
        var out = Array[DType.int64](overwritten=len(target))
        var base = Int64(self.start)
        var stop = base + Int64(self.length)
        for i in range(len(target)):
            if not target.is_valid(i):
                out[i] = NOT_FOUND
                continue
            var label = labels[i]
            if label < base or label >= stop:
                out[i] = NOT_FOUND
            else:
                out[i] = label - base
        return out^

    def _ordinals(self, target: AnyArray) raises -> KeyCodes:
        """Puts this index's labels and a target's into one ordinal space.

        Two rows hold the same ordinal exactly when they hold the same label,
        with every null in one group, which is what makes a null find a null.
        The index's rows come first, so row `i` of the result is this index's row
        `i` and row `len(self) + j` is the target's row `j`.

        Args:
            target: The labels to look for.

        Returns:
            One ordinal per row of the concatenation, this index's rows first,
            with the ordinal count beside them.

        Raises:
            Error: If the dtypes differ or cannot be factorized.
        """
        var mine = self.materialize()
        if mine.dtype() != target.dtype() or (
            mine.is_string() != target.is_string()
        ):
            raise Error(
                String(
                    "index: looking a ",
                    target.dtype(),
                    " label up in a ",
                    mine.dtype(),
                    (
                        " index is not written yet, because it needs a common"
                        " type rather than a comparison"
                    ),
                )
            )
        return factorize_any(concat_two_any(mine, target))

    def get_indexer(self, target: AnyArray) raises -> Array[DType.int64]:
        """Where each of a set of labels sits in this index.

        This is the primitive `reindex` and `align` are made of, and the one
        worth being fast. A range answers by subtraction. Labels answer with one
        factorize over both sides and two linear passes, rather than with a hash
        table built for the occasion and a probe per row.

        Args:
            target: The labels to look for, in the order the answer should come
                back in.

        Returns:
            One position per target row, `NOT_FOUND` where there is no such
            label.

        Raises:
            Error: If this index has duplicates, which pandas refuses here for
                the reason that there is no single position to report. Use
                `get_indexer_non_unique`. Also if the dtypes differ.
        """
        if not self.is_unique():
            raise Error(
                "index: get_indexer needs a unique index, because a duplicated"
                " label has more than one position; use"
                " get_indexer_non_unique"
            )
        if self.is_range() and target.dtype() == DType.int64:
            return self._range_indexer(target)

        var rows = self.length
        var found = self._ordinals(target)
        ref codes = found.codes
        var out = _filled(len(target), NOT_FOUND)
        # One position per ordinal is enough because the index is unique, so the
        # first pass never writes the same slot twice.
        var at = _filled(_space_of(found), NOT_FOUND)
        for i in range(rows):
            at[Int(codes[i])] = Int64(i)
        for j in range(len(target)):
            out[j] = at[Int(codes[rows + j])]
        return out^

    def get_indexer_non_unique(self, target: AnyArray) raises -> Matches:
        """Every position each of a set of labels sits at, and the ones absent.

        The form to use when the index may hold a label twice, which is what a
        merge on an index and a `loc` on a repeated key both need. A label that
        matches three rows contributes three positions, in index order, and a
        label that matches nothing contributes one `NOT_FOUND` so that the answer
        never goes quiet about a label it was given.

        Args:
            target: The labels to look for.

        Returns:
            The positions and the target rows that found none.

        Raises:
            Error: If the dtypes differ or cannot be factorized.
        """
        var rows = self.length
        var grouped = self._ordinals(target)
        ref codes = grouped.codes
        var space = _space_of(grouped)

        # A counting sort over the ordinals, so that the positions holding one
        # label end up next to each other and in index order. Three passes over
        # numbers already in cache, against a list per ordinal and an allocation
        # for each of them.
        var starts = _filled(space + 1, Int64(0))
        for i in range(rows):
            var slot = Int(codes[i]) + 1
            starts[slot] = starts[slot] + 1
        for c in range(space):
            starts[c + 1] = starts[c + 1] + starts[c]
        var by_ordinal = Array[DType.int64](overwritten=rows)
        var cursor = Array[DType.int64](overwritten=space)
        for c in range(space):
            cursor[c] = starts[c]
        for i in range(rows):
            var ordinal = Int(codes[i])
            by_ordinal[Int(cursor[ordinal])] = Int64(i)
            cursor[ordinal] = cursor[ordinal] + 1

        var total = 0
        var absent = 0
        for j in range(len(target)):
            var ordinal = Int(codes[rows + j])
            var hits = Int(starts[ordinal + 1] - starts[ordinal])
            if hits == 0:
                absent += 1
                total += 1
            else:
                total += hits

        var positions = Array[DType.int64](overwritten=total)
        var missing = Array[DType.int64](overwritten=absent)
        var out_at = 0
        var miss_at = 0
        for j in range(len(target)):
            var ordinal = Int(codes[rows + j])
            var first = Int(starts[ordinal])
            var last = Int(starts[ordinal + 1])
            if first == last:
                positions[out_at] = NOT_FOUND
                out_at += 1
                missing[miss_at] = Int64(j)
                miss_at += 1
                continue
            for k in range(first, last):
                positions[out_at] = by_ordinal[k]
                out_at += 1
        return Matches(positions^, missing^)

    def get_loc(self, label: AnyArray) raises -> List[Int]:
        """Where one label sits, raising rather than answering with a sentinel.

        pandas returns an integer for a unique index and a slice or a mask for a
        repeated one, which is three return types on one name and is a decision
        for the layer that has a Python object to hand back. This returns every
        position and lets that layer choose.

        A label that is not there raises, and that is the whole difference
        between this and `get_indexer`. It is what makes `df.loc["nope"]` a
        `KeyError` rather than a row of nulls.

        Args:
            label: The label, as a column of exactly one row.

        Returns:
            Every position holding it, in order, never empty.

        Raises:
            Error: If the label is not in the index, if more than one label was
                passed, or if the dtypes differ.
        """
        if len(label) != 1:
            raise Error(
                String(
                    "index: get_loc takes one label, got ",
                    len(label),
                    "; use get_indexer for several",
                )
            )
        var found = self.get_indexer_non_unique(label)
        var out = List[Int]()
        for i in range(len(found.positions)):
            if found.positions[i] != NOT_FOUND:
                out.append(Int(found.positions[i]))
        if not out:
            raise Error("index: label is not in the index")
        return out^

    def equals(self, other: Self) raises -> Bool:
        """Whether two indexes hold the same labels in the same order.

        The name is not part of it and neither is the representation, so a range
        equals the array it materializes to and an unnamed index equals a named
        one holding the same labels. That is what pandas means by `equals` and it
        is what `align` asks, because two frames whose rows are labelled the same
        need no alignment whatever their indexes are called.

        Both sides being ranges is answered without materializing either, which
        is the case that runs on every frame in the library.

        Args:
            other: The index to compare against.

        Returns:
            Whether the labels match.

        Raises:
            Error: If the label dtypes cannot be compared.
        """
        if self.length != other.length:
            return False
        if self.is_range() and other.is_range():
            return self.start == other.start
        var mine = self.materialize()
        var theirs = other.materialize()
        if mine.dtype() != theirs.dtype() or (
            mine.is_string() != theirs.is_string()
        ):
            # pandas compares an int64 index equal to a float64 one holding the
            # same numbers, because it promotes before it compares. Doing that
            # here means a cast and a decision about which direction is safe, so
            # for now two dtypes are two indexes and the divergence is written
            # down rather than guessed at.
            return False
        for i in range(self.length):
            if not _same_label(mine, i, theirs, i):
                return False
        return True

    def identical(self, other: Self) raises -> Bool:
        """Whether two indexes are the same index rather than merely equal.

        The name has to match as well. `equals` and this one disagree on exactly
        the pairs that `align` should accept and `assert_frame_equal` should
        reject, which is why both exist and why having only one of them gets one
        of those two wrong.

        Args:
            other: The index to compare against.

        Returns:
            Whether the labels and the name both match.

        Raises:
            Error: If the label dtypes cannot be compared.
        """
        if Bool(self.name) != Bool(other.name):
            return False
        if self.name and self.name.value() != other.name.value():
            return False
        return self.equals(other)
