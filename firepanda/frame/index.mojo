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

What this does not do is align. Nothing in the library looks a row up by its label,
`loc` does not exist, and two frames added together do not match their indexes first.
That is the `Index` API in https://github.com/tamnd/firepanda/issues/154, and it needs
somewhere to live before it can be written. This is that somewhere.
"""

from std.collections import Optional

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.kernel.select import (
    filter_any,
    filter_range,
    take_any,
    take_range,
)


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
