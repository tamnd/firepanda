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

The four set operations came next and they are the same machine with a different rule
about which rows survive. Concatenate, factorize, count each ordinal on each side, and
then a union keeps each label the larger of the two counts times, an intersection keeps
the ones with a count on both sides, a difference keeps the ones with none on the right
and a symmetric difference keeps the ones with none on the other. Every expected answer
in their tests was read off a running pandas 3.0.5 rather than off its documentation,
which is wrong about `intersection` and duplicates, and the three parts that look
arbitrary are arbitrary in pandas too: `union` sorts by default while `intersection` and
`difference` keep the left side's order, `union` is the only one that keeps duplicates,
and a union with an empty index or with itself returns unsorted because it takes an
early return before the sort.

The editing operations came third and none of them needed a new idea. `append` is a
concatenation, `delete` and `insert` are a gather over a list of positions, `drop` is a
lookup and then the same gather, and `putmask` is a gather over the two sides
concatenated. Writing them as gathers rather than as loops over the labels is not
tidiness: the gather already knows about nulls, about strings and about every dtype, so
a null inserted into a string index works without a line being written about it, and
there is one place to fix if it does not.

The slice locators are the part of that group with something in it. A `loc` slice
includes both of its ends, and what makes that true is that the left bound of a label is
the first row at least it and the right bound is the first row past it, so a label
sitting in four rows gives a pair covering all four. On a monotonic index that pair is
two binary searches, on a descending one the same searches with both comparisons turned
over, and on an index that is neither there is nothing to search: pandas looks the label
up instead, refuses when it is absent or repeated, and says in the message that sorting
the index is the fix. The monotonic check in front of the search is a scan and is not
remembered, for the same reason `is_unique` is not remembered, so what the search buys
against the fallback is the factorize rather than the pass.

What this still does not do is align. There is no `reindex` and no `loc`, and the
remaining names on https://github.com/tamnd/firepanda/issues/154 are the level
accessors and `asof_locs`. What is here is what those are made of.
"""

from std.collections import Optional

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.lists import ALL, ORDERED
from firepanda.frame.display import DisplayOptions, render_value
from firepanda.hash.function import key_bits
from firepanda.hash.grouping import KeyCodes, factorize_any
from firepanda.kernel.concat import concat_two_any
from firepanda.kernel.select import (
    filter_any,
    filter_range,
    take_any,
    take_range,
)
from firepanda.kernel.sort import argsort_any, is_sorted_any, sort_key


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

    def take_positions(deinit self) -> Array[DType.int64]:
        """Gives up the positions and throws the rest away.

        For a caller that asked a question the misses do not answer, which is
        `get_indexer_for` and nothing else so far. It consumes the pair because a
        field cannot be moved out of a struct that still has to be destroyed, and
        copying an array the size of the answer to avoid saying so would be a
        strange trade.

        Returns:
            The positions.
        """
        return self.positions^

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


struct Joined(Movable):
    """Two label sets laid end to end and factorized together.

    Every lookup and every set operation on an index is a pass over this. Two
    rows carry the same ordinal exactly when they carry the same label, so a
    question about whether a label appears on both sides becomes a question about
    a number, and the labels themselves are only read again at the end when the
    surviving rows are gathered out of them.

    Keeping the concatenated column rather than throwing it away is what lets the
    set operations answer with labels instead of with positions. A lookup does not
    need it and pays nothing for it, since the concatenation had to be built to be
    factorized in the first place.
    """

    var labels: AnyArray
    """This index's labels, then the other side's."""

    var found: KeyCodes
    """What the factorize returned, held whole rather than taken apart, because a
    field cannot be moved out of it without leaving the rest undestroyable."""

    var space: Int
    """How many ordinals there may be, so a table indexed by one can be sized."""

    var rows: Int
    """Where the other side starts, which is this index's length."""

    def __init__(
        out self,
        var labels: AnyArray,
        var found: KeyCodes,
        space: Int,
        rows: Int,
    ):
        """Holds a factorized concatenation.

        Args:
            labels: The two label sets, this index's first.
            found: The ordinals.
            space: An upper bound on the ordinals.
            rows: Where the second set starts.
        """
        self.labels = labels^
        self.found = found^
        self.space = space
        self.rows = rows

    def ordinal(self, i: Int) -> Int:
        """The ordinal of row `i`, as something a table can be indexed by.

        Args:
            i: A row of the concatenation.

        Returns:
            Its ordinal.
        """
        return Int(self.found.codes[i])

    def total(self) -> Int:
        """How many rows there are in both sides together.

        Returns:
            The length of the concatenation.
        """
        return len(self.found.codes)


def _side_counts(
    joined: Joined, first: Int, last: Int
) raises -> Array[DType.int64]:
    """How many rows in `[first, last)` carry each ordinal.

    Counting the two sides separately is what makes the duplicate rules fall out.
    A label is in the intersection when both counts are above zero, in the
    difference when the second is zero, and appears in the union the larger of
    the two counts times, which is what pandas does and is not what a set would
    do.

    Args:
        joined: The factorized concatenation.
        first: The first row of the side.
        last: One past the last row of the side.

    Returns:
        One count per ordinal.
    """
    var out = _filled(joined.space, Int64(0))
    for i in range(first, last):
        var at = joined.ordinal(i)
        out[at] = out[at] + 1
    return out^


def _shared_name(a: Optional[String], b: Optional[String]) -> Optional[String]:
    """The name a result of two indexes carries.

    pandas keeps the name when both sides agree and drops it when they do not,
    on the reasoning that a label set drawn from two differently named levels is
    not either of them. Two unnamed indexes agree, and the answer is unnamed.

    Args:
        a: One name.
        b: The other.

    Returns:
        The common name, or `None`.
    """
    if not a or not b:
        return Optional[String]()
    if a.value() != b.value():
        return Optional[String]()
    return Optional[String](a.value())


def _gathered(
    labels: AnyArray,
    picks: List[Int],
    var name: Optional[String],
    sort: Bool,
) raises -> Index:
    """Builds the index a set operation decided on, sorted or in picked order.

    Args:
        labels: The concatenation the positions point into.
        picks: The rows to keep, in the order to keep them.
        name: What to call the result.
        sort: Whether to order by label rather than by the order picked.

    Returns:
        The index.

    Raises:
        Error: If the labels cannot be gathered or sorted.
    """
    var out = take_any(labels, picks)
    if not sort:
        return Index(out^, name^)
    # `argsort_any` puts nulls last, which is where pandas puts them in a sorted
    # set operation, so there is nothing to say about them here.
    var order = argsort_any(out)
    var by = List[Int](capacity=len(order))
    for i in range(len(order)):
        by.append(Int(order[i]))
    return Index(take_any(out, by), name^)


def _label_order(a: AnyArray, i: Int, b: AnyArray, j: Int) raises -> Int:
    """Compares two labels the way a sort would.

    The ordering companion of `_same_label`, and it agrees with it on which pairs
    are equal. A null is greater than every present label, which puts nulls at
    the end and is where `argsort_any` puts them, so a bound looked up with a
    null lands past the last row rather than in the middle of the index.

    Args:
        a: The first column.
        i: The row in it.
        b: The second column.
        j: The row in it.

    Returns:
        A negative number when the first label is smaller, zero when they are
        the same label, and a positive number when it is larger.

    Raises:
        Error: If the dtypes differ or cannot be ordered.
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
    var b_here = b.is_valid(j)
    if not a_here or not b_here:
        if a_here:
            return -1
        if b_here:
            return 1
        return 0
    if a.is_string():
        ref mine = a.strings()
        ref theirs = b.strings()
        var left = mine.unsafe_bytes(i)
        var right = theirs.unsafe_bytes(j)
        var shared = len(left)
        if len(right) < shared:
            shared = len(right)
        for k in range(shared):
            if left[k] != right[k]:
                return -1 if left[k] < right[k] else 1
        if len(left) == len(right):
            return 0
        return -1 if len(left) < len(right) else 1
    comptime for candidate in ORDERED:
        if a.dtype() == candidate:
            # `sort_key` is the same total order the sort uses, which is what
            # keeps a bound consistent with the sortedness that let us search
            # for it, and it is why a float compares here without a special
            # case for the sign bit.
            var left = sort_key(
                a.unsafe_ptr[candidate]().unsafe_offset(i).unsafe_load()
            )
            var right = sort_key(
                b.unsafe_ptr[candidate]().unsafe_offset(j).unsafe_load()
            )
            if left < right:
                return -1
            if left > right:
                return 1
            return 0
    raise Error(String("index: unorderable label dtype ", a.dtype()))


def _shown_label(col: AnyArray, i: Int) -> String:
    """Renders one label for an error message.

    The frame printer already knows how to spell every dtype and how to spell a
    null, and an error that names the label the caller passed is worth a great
    deal more than one that says a label is missing.

    Args:
        col: The column.
        i: The row.

    Returns:
        The label as text.
    """
    return render_value(col, i, DisplayOptions())


def _searched(
    values: AnyArray, label: AnyArray, right: Bool, ascending: Bool
) raises -> Int:
    """Binary searches a monotonic column for where a label belongs.

    One function covers all four cases because they differ only in which way the
    comparison has to fall for the answer to be further right. On an ascending
    column the left bound is the first row at least the label and the right bound
    is the first row past it, and on a descending column both tests turn over.

    Args:
        values: The labels, ascending or descending with no nulls in the middle.
        label: The label to place, as a column of one row.
        right: Look for the end of a run of equal labels rather than its start.
        ascending: Whether the column increases.

    Returns:
        A position between zero and the length of the column.

    Raises:
        Error: If the dtypes differ or cannot be ordered.
    """
    var lo = 0
    var hi = len(values)
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        var order = _label_order(values, mid, label, 0)
        var before = (order < 0) if ascending else (order > 0)
        if right and order == 0:
            before = True
        if before:
            lo = mid + 1
        else:
            hi = mid
    return lo


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

    def _join_with(self, target: AnyArray) raises -> Joined:
        """Puts this index's labels and a target's into one ordinal space.

        Two rows hold the same ordinal exactly when they hold the same label,
        with every null in one group, which is what makes a null find a null.
        The index's rows come first, so row `i` of the result is this index's row
        `i` and row `len(self) + j` is the target's row `j`.

        Args:
            target: The labels to look for.

        Returns:
            The concatenated labels, one ordinal per row, and where the target's
            rows start.

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
        var labels = concat_two_any(mine, target)
        var found = factorize_any(labels)
        var space = _space_of(found)
        return Joined(labels^, found^, space, self.length)

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
        var found = self._join_with(target)
        ref codes = found.found.codes
        var out = _filled(len(target), NOT_FOUND)
        # One position per ordinal is enough because the index is unique, so the
        # first pass never writes the same slot twice.
        var at = _filled(found.space, NOT_FOUND)
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
        var grouped = self._join_with(target)
        ref codes = grouped.found.codes
        var space = grouped.space

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

    def get_indexer_for(self, target: AnyArray) raises -> Array[DType.int64]:
        """Where a set of labels sits, whether or not the index is unique.

        `get_indexer` refuses a repeated label and `get_indexer_non_unique`
        answers with a variable number of rows, so code that does not know which
        kind of index it holds cannot call either. This picks, which is what
        pandas does under the same name and is what most of pandas calls
        internally.

        The answer has one row per target label only when the index is unique. A
        caller that needs that guarantee should be calling `get_indexer` and
        letting it refuse.

        Args:
            target: The labels to look for.

        Returns:
            The positions, `NOT_FOUND` for a label found nowhere.

        Raises:
            Error: If the dtypes differ or cannot be factorized.
        """
        if self.is_unique():
            return self.get_indexer(target)
        var found = self.get_indexer_non_unique(target)
        return found^.take_positions()

    def unique(self) raises -> Self:
        """The labels, each kept once, in the order they first appear.

        First appearance rather than sorted, because that is what pandas does and
        because a caller that wanted them sorted would rather say so than have it
        done twice.

        Returns:
            An index of the distinct labels, with this one's name.

        Raises:
            Error: If the labels cannot be factorized.
        """
        var name = self.name
        if self.is_range():
            return Self(copy=self)
        var labels = self.materialize()
        var found = factorize_any(labels)
        var picks = List[Int]()
        var taken = _filled(_space_of(found), Int64(0))
        for i in range(self.length):
            var at = Int(found.codes[i])
            if taken[at] != 0:
                continue
            taken[at] = 1
            picks.append(i)
        return Self(take_any(labels, picks), name^)

    def union(self, other: Self, sort: Bool = True) raises -> Self:
        """Every label in either index.

        The duplicate rule is the one that surprises people. A label appears the
        larger of the two counts times rather than once and rather than the sum,
        so `[1, 1, 2, 3]` union `[1, 2, 2, 4]` is `[1, 1, 2, 2, 3, 4]`. That is
        what pandas does and the reasoning is that a union of two label sets
        should be able to label at least as many rows as either of them could.

        Three short circuits are observable rather than internal. An index unioned
        with itself, with an empty index, or an empty index unioned with one, all
        return in the original order without sorting, so `Index([3, 1, 2])` union
        an empty index is `[3, 1, 2]` and not `[1, 2, 3]`. pandas returns the same
        thing for the same reason, which is that it takes an early return before
        the sort, and an implementation that always sorted would quietly disagree
        with it on the two most common calls there are.

        Args:
            other: The other index.
            sort: Order the result by label. False keeps this index's order
                followed by the labels only the other side has, in the order it
                has them.

        Returns:
            The union, named what both sides agree on.

        Raises:
            Error: If the label dtypes differ, which pandas answers by promoting
                both sides and firepanda does not do yet.
        """
        var name = _shared_name(self.name, other.name)
        if other.length == 0 or self.equals(other):
            return self.renamed(name^)
        if self.length == 0:
            return other.renamed(name^)

        var joined = self._join_with(other.materialize())
        var mine = _side_counts(joined, 0, joined.rows)
        var theirs = _side_counts(joined, joined.rows, joined.total())
        var picks = List[Int]()
        var taken = _filled(joined.space, Int64(0))
        for i in range(joined.total()):
            var at = joined.ordinal(i)
            if taken[at] != 0:
                continue
            taken[at] = 1
            var copies = mine[at]
            if theirs[at] > copies:
                copies = theirs[at]
            # The same row gathered several times, which is how one label becomes
            # the several copies of it the rule asks for.
            for _ in range(Int(copies)):
                picks.append(i)
        return _gathered(joined.labels, picks, name^, sort)

    def intersection(self, other: Self, sort: Bool = False) raises -> Self:
        """Every label in both indexes, each once.

        The result is unique even when both inputs are not, which the pandas
        docstring still describes as keeping the smaller of the two counts and
        which pandas has not done for some versions now. Checked against a running
        3.0.5 rather than read off the page.

        The default order is this index's rather than sorted, which is the
        opposite of `union`. That asymmetry is pandas' and it is not arbitrary: an
        intersection is a filter of the left side and a union is not a filter of
        anything, so one has an order to inherit and the other has to invent one.

        Args:
            other: The other index.
            sort: Order the result by label rather than by this index's order.

        Returns:
            The intersection, named what both sides agree on.

        Raises:
            Error: If the label dtypes differ.
        """
        var name = _shared_name(self.name, other.name)
        if self.equals(other):
            if self.is_unique():
                return self.renamed(name^)
            return self.unique().renamed(name^)

        var joined = self._join_with(other.materialize())
        var theirs = _side_counts(joined, joined.rows, joined.total())
        var picks = List[Int]()
        var taken = _filled(joined.space, Int64(0))
        for i in range(joined.rows):
            var at = joined.ordinal(i)
            if taken[at] != 0 or theirs[at] == 0:
                continue
            taken[at] = 1
            picks.append(i)
        return _gathered(joined.labels, picks, name^, sort)

    def difference(self, other: Self, sort: Bool = True) raises -> Self:
        """Every label in this index and not in the other, each once.

        Unique output, like `intersection` and unlike `union`, and sorted by
        default, like `union` and unlike `intersection`. Those two defaults are
        not a pattern and there is no rule to derive them from; they are what
        pandas does, and a difference that came back in a different order would
        break code that indexes into the result.

        Args:
            other: The labels to remove.
            sort: Order the result by label rather than by this index's order.

        Returns:
            The difference, named what both sides agree on.

        Raises:
            Error: If the label dtypes differ.
        """
        var name = _shared_name(self.name, other.name)
        var joined = self._join_with(other.materialize())
        var theirs = _side_counts(joined, joined.rows, joined.total())
        var picks = List[Int]()
        var taken = _filled(joined.space, Int64(0))
        for i in range(joined.rows):
            var at = joined.ordinal(i)
            if taken[at] != 0 or theirs[at] != 0:
                continue
            taken[at] = 1
            picks.append(i)
        return _gathered(joined.labels, picks, name^, sort)

    def symmetric_difference(
        self,
        other: Self,
        var result_name: Optional[String] = Optional[String](),
        sort: Bool = True,
    ) raises -> Self:
        """Every label in exactly one of the two indexes, each once.

        Unsorted, it is this index's leftovers followed by the other's, which is
        the order the two passes below produce and the order pandas produces for
        the same reason.

        `result_name` is the one place in the four where the caller can override
        the name rule, and it exists in pandas because a symmetric difference is
        the operation where neither side has a better claim to the name than the
        other.

        Args:
            other: The other index.
            result_name: What to call the result, overriding the usual rule.
            sort: Order the result by label rather than by side.

        Returns:
            The symmetric difference.

        Raises:
            Error: If the label dtypes differ.
        """
        var name = _shared_name(self.name, other.name)
        if result_name:
            name = result_name^

        var joined = self._join_with(other.materialize())
        var mine = _side_counts(joined, 0, joined.rows)
        var theirs = _side_counts(joined, joined.rows, joined.total())
        var picks = List[Int]()
        var taken = _filled(joined.space, Int64(0))
        for i in range(joined.rows):
            var at = joined.ordinal(i)
            if taken[at] != 0 or theirs[at] != 0:
                continue
            taken[at] = 1
            picks.append(i)
        for i in range(joined.rows, joined.total()):
            var at = joined.ordinal(i)
            if taken[at] != 0 or mine[at] != 0:
                continue
            taken[at] = 1
            picks.append(i)
        return _gathered(joined.labels, picks, name^, sort)

    def append(self, other: Self) raises -> Self:
        """This index's labels followed by another's.

        Not a union. Nothing is deduplicated and nothing is sorted, so appending
        an index to itself gives every label twice, which is what `concat` wants
        and is the whole difference between the two operations.

        Two adjacent ranges stay a range, so appending the index of one slice of
        a frame to the index of the next costs two integers and no memory. That
        is the same argument the representation exists for and it is worth having
        here because a concatenation of frames read in pieces is exactly this
        case.

        Args:
            other: The labels to put after this index's.

        Returns:
            The concatenation, named what both sides agree on.

        Raises:
            Error: If the label dtypes differ.
        """
        var name = _shared_name(self.name, other.name)
        if other.length == 0:
            return self.renamed(name^)
        if self.length == 0:
            return other.renamed(name^)
        if (
            self.is_range()
            and other.is_range()
            and other.start == self.start + self.length
        ):
            return Self(self.start, self.length + other.length, name^)
        return Self(
            concat_two_any(self.materialize(), other.materialize()), name^
        )

    def append(self, others: List[Self]) raises -> Self:
        """This index's labels followed by several others', in order.

        The name survives only when every side agrees on it, which follows from
        applying the two index rule down the list and is what pandas does.

        Args:
            others: The indexes to put after this one, in order.

        Returns:
            The concatenation.

        Raises:
            Error: If the label dtypes differ.
        """
        var out = Self(copy=self)
        for i in range(len(others)):
            out = out.append(others[i])
        return out^

    def delete(self, positions: List[Int]) raises -> Self:
        """The index without the rows at a set of positions.

        By position rather than by label, which is what separates this from
        `drop`. A position appearing twice removes one row, since a row can only
        be removed once.

        Args:
            positions: The rows to remove. A negative position counts from the
                end, so -1 is the last row.

        Returns:
            The remaining labels in their original order.

        Raises:
            Error: If a position is outside the index.
        """
        var marks = _filled(self.length, Int64(0))
        for i in range(len(positions)):
            var at = positions[i]
            if at < 0:
                at += self.length
            if at < 0 or at >= self.length:
                raise Error(
                    String(
                        "index: delete position ",
                        positions[i],
                        " is out of bounds for an index of length ",
                        self.length,
                    )
                )
            marks[at] = 1
        var picks = List[Int]()
        for i in range(self.length):
            if marks[i] == 0:
                picks.append(i)
        return self.take(picks)

    def delete(self, position: Int) raises -> Self:
        """The index without the row at one position.

        Args:
            position: The row to remove, counting from the end when negative.

        Returns:
            The remaining labels.

        Raises:
            Error: If the position is outside the index.
        """
        var one: List[Int] = [position]
        return self.delete(one)

    def insert(self, position: Int, label: AnyArray) raises -> Self:
        """The index with one label put in at a position.

        Args:
            position: Where the new label lands, from zero to the length of the
                index inclusive, counting from the end when negative.
            label: The label, as a column of exactly one row.

        Returns:
            An index one row longer.

        Raises:
            Error: If the position is outside the index, if more than one label
                was passed, or if the dtypes differ.
        """
        if len(label) != 1:
            raise Error(
                String("index: insert takes one label, got ", len(label))
            )
        var at = position
        if at < 0:
            at += self.length
        if at < 0 or at > self.length:
            raise Error(
                String(
                    "index: insert position ",
                    position,
                    " is out of bounds for an index of length ",
                    self.length,
                )
            )
        # The new label is row `self.length` of the concatenation, so putting it
        # in is a gather that names that row where it belongs. One pass and no
        # per row branch, and the null and string cases come along with the
        # gather rather than being written twice.
        var both = concat_two_any(self.materialize(), label)
        var picks = List[Int](capacity=self.length + 1)
        for i in range(at):
            picks.append(i)
        picks.append(self.length)
        for i in range(at, self.length):
            picks.append(i)
        return Self(take_any(both, picks), Optional[String](copy=self.name))

    def drop(self, labels: AnyArray, errors: String = "raise") raises -> Self:
        """The index without every row carrying one of a set of labels.

        By label rather than by position, and every occurrence goes, so dropping
        `1` from `[1, 1, 2]` leaves `[2]` rather than `[1, 2]`.

        Args:
            labels: The labels to remove.
            errors: `"raise"` to refuse when a label is not in the index, naming
                the ones that are not, or `"ignore"` to remove what is there and
                say nothing about the rest.

        Returns:
            The remaining labels in their original order.

        Raises:
            Error: If a label is missing and `errors` is `"raise"`, if `errors`
                is neither word, or if the dtypes differ.
        """
        if errors != "raise" and errors != "ignore":
            raise Error(
                String(
                    "index: drop errors must be 'raise' or 'ignore'; got ",
                    errors,
                )
            )
        var found = self.get_indexer_non_unique(labels)
        if errors == "raise" and len(found.missing) > 0:
            var absent = String()
            for i in range(len(found.missing)):
                if i > 0:
                    absent += ", "
                absent += _shown_label(labels, Int(found.missing[i]))
            raise Error(String("index: [", absent, "] not found in axis"))
        var marks = _filled(self.length, Int64(0))
        ref positions = found.positions
        for i in range(len(positions)):
            var at = positions[i]
            if at != NOT_FOUND:
                marks[Int(at)] = 1
        var picks = List[Int]()
        for i in range(self.length):
            if marks[i] == 0:
                picks.append(i)
        return self.take(picks)

    def putmask(self, mask: Array[DType.bool], value: AnyArray) raises -> Self:
        """The index with the rows a mask picks out replaced.

        The replacement is either one label, which goes everywhere the mask is
        true, or a column as long as the index, in which case row `i` takes row
        `i`. A null in the mask is a false, since a row we cannot say to replace
        is a row we leave alone.

        Args:
            mask: Which rows to replace. Must be as long as this index.
            value: The replacement, either one row or as many rows as this index.

        Returns:
            An index of the same length.

        Raises:
            Error: If the mask is the wrong length, if the replacement is neither
                one row nor the whole length, or if the dtypes differ.
        """
        if len(mask) != self.length:
            raise Error(
                String(
                    (
                        "index: putmask mask and index must be the same"
                        " length; got"
                    ),
                    " ",
                    len(mask),
                    " and ",
                    self.length,
                )
            )
        if len(value) != 1 and len(value) != self.length:
            raise Error(
                String(
                    "index: putmask replacement must be one label or ",
                    self.length,
                    "; got ",
                    len(value),
                )
            )
        var any_set = False
        for i in range(self.length):
            if mask.is_valid(i) and Bool(mask[i]):
                any_set = True
                break
        if not any_set:
            return Self(copy=self)
        var wide = len(value) != 1
        var both = concat_two_any(self.materialize(), value)
        var picks = List[Int](capacity=self.length)
        for i in range(self.length):
            if mask.is_valid(i) and Bool(mask[i]):
                picks.append(self.length + i if wide else self.length)
            else:
                picks.append(i)
        return Self(take_any(both, picks), Optional[String](copy=self.name))

    def get_slice_bound(self, label: AnyArray, side: String) raises -> Int:
        """Where a label sits when the index is read as an ordered thing.

        A `loc` slice is inclusive of both ends, and this is what makes that
        true: the left bound of a label is the first row that is at least it and
        the right bound is the first row past it, so a label that appears four
        times gives a pair that covers all four.

        The search is binary and the monotonic check in front of it is a scan, so
        the bound costs a pass over the labels and not a lookup. Nothing is
        cached, for the same reason `is_unique` caches nothing: an index is copied
        into every frame derived from it and a remembered answer would have to be
        invalidated by `take` and `filter`. What the search buys against the
        fallback is the factorize, not the scan.

        An index that is neither ascending nor descending has no bound to search
        for, so pandas looks the label up instead and refuses when it is not
        there or is there twice, and says in the message that sorting the index
        is the fix. This does the same.

        Args:
            label: The label, as a column of exactly one row.
            side: `"left"` for the first row that is at least the label, or
                `"right"` for the first row past it.

        Returns:
            A position between zero and the length of the index.

        Raises:
            Error: If `side` is neither word, if more than one label was passed,
                if the dtypes differ, or if the index is not monotonic and the
                label is missing or repeated.
        """
        if side != "left" and side != "right":
            raise Error(
                String("index: side must be 'left' or 'right'; got ", side)
            )
        if len(label) != 1:
            raise Error(
                String(
                    "index: a slice bound is one label, got ",
                    len(label),
                )
            )
        var right = side == "right"

        if self.is_range():
            if (
                not label.is_string()
                and label.dtype() == DType.int64
                and label.is_valid(0)
            ):
                # A range is sorted, has no duplicates and holds every integer
                # between its ends, so the bound is arithmetic and reads nothing.
                ref only = label.as_typed_view[DType.int64]()
                var at = Int(only[0]) - self.start
                if right:
                    at += 1
                if at < 0:
                    return 0
                if at > self.length:
                    return self.length
                return at
            # Anything else is either a null or the wrong dtype, and both are
            # answered by the general path rather than by a second rule here.
            var built = self.materialize()
            return _searched(built, label, right, ascending=True)

        ref values = self.labels.value()
        if self.is_monotonic_increasing():
            return _searched(values, label, right, ascending=True)
        if self.is_monotonic_decreasing():
            return _searched(values, label, right, ascending=False)

        var hits = 0
        var first = 0
        for i in range(self.length):
            if _same_label(values, i, label, 0):
                if hits == 0:
                    first = i
                hits += 1
        if hits == 0:
            raise Error(
                String(
                    "index: cannot get the ",
                    side,
                    " slice bound of a non-monotonic index for the missing",
                    " label ",
                    _shown_label(label, 0),
                    "; either sort the index or use a label it has",
                )
            )
        if hits > 1:
            raise Error(
                String(
                    "index: cannot get the ",
                    side,
                    " slice bound of a non-monotonic index for the repeated",
                    " label ",
                    _shown_label(label, 0),
                )
            )
        return first + 1 if right else first

    def slice_locs(
        self,
        start: Optional[AnyArray] = Optional[AnyArray](),
        end: Optional[AnyArray] = Optional[AnyArray](),
    ) raises -> Tuple[Int, Int]:
        """The half-open row range a pair of labels describes.

        The left bound of the first label and the right bound of the second,
        which is why a `loc` slice includes both ends while the pair it returns
        is half open like every other range in the library. A missing bound means
        the end of the index on that side.

        Both labels are looked up the same way whether the index ascends or
        descends, so on a descending index the caller names the larger label
        first, which is what reading the index in order means.

        Args:
            start: The first label, or nothing for the start of the index.
            end: The last label, or nothing for the end of the index.

        Returns:
            The first row and one past the last.

        Raises:
            Error: If a bound cannot be placed.
        """
        var first = 0
        if start:
            first = self.get_slice_bound(start.value(), "left")
        var last = self.length
        if end:
            last = self.get_slice_bound(end.value(), "right")
        return (first, last)

    def slice_indexer(
        self,
        start: Optional[AnyArray] = Optional[AnyArray](),
        end: Optional[AnyArray] = Optional[AnyArray](),
        step: Int = 1,
    ) raises -> Tuple[Int, Int, Int]:
        """The same range as `slice_locs`, with the step carried through.

        The step is not used to find the bounds and pandas does not use it
        either, so this is `slice_locs` and a third number. It exists because the
        caller of a `loc` slice has a step to pass on and would otherwise have to
        rebuild the triple itself.

        Args:
            start: The first label, or nothing for the start of the index.
            end: The last label, or nothing for the end of the index.
            step: The stride, returned unchanged.

        Returns:
            The first row, one past the last, and the step.

        Raises:
            Error: If a bound cannot be placed.
        """
        var found = self.slice_locs(start, end)
        return (found[0], found[1], step)
