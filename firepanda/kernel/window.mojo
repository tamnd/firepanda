"""A window is a pair of row numbers, and everything else here is arithmetic.

pandas has three window types and fifty three callables on them, and the thing
they have in common is smaller than the surface suggests. `rolling`, `expanding`
and every one of `window`, `min_periods`, `center`, `closed` and `step` decide
one thing only, which is the first row and the last row of the window that row
`i` reduces. Once that pair is known the reduction does not care which of them
asked for it. So the pair is computed in one place, `Shape.edges` is that place,
and an expanding window is a rolling one whose width is the height of the
column.

## What the five parameters actually do

The window that ends at row `i` runs from `i + 1 - window` up to `i + 1`, not
including the last. `center` moves the far end forward by `(window - 1) // 2`
before the near end is derived from it, which is why a centred window of an even
width is not symmetric and why it takes one more row from the left than from the
right. `closed` moves one or both ends by one: `left` moves both back, `both`
moves only the near end back and so holds one row more than the width asks for,
and `neither` moves only the far end back and so holds one row less. `step` does
not touch the window at all and only decides which rows are asked, so it is a
property of the output and not of the reduction.

Clipping happens after all of that and only against the column, which matters:
the near end is derived from the unclipped far end, so a centred window at the
bottom of the column loses rows rather than sliding back up the column to keep
its width.

## Why the reductions are incremental and what that costs

Summing each window separately is the width of the window times the height of
the column. Carrying a total and adding the row that arrives and subtracting the
row that leaves is the height of the column, and that is the whole reason
anybody uses these. The edges are non decreasing in `i` for every combination of
the five parameters above, including `step`, which is what makes the carrying
legal.

pandas carries the same total and pays a price for it that is visible in the
answers. Its total is one number, so an infinity that enters the window makes it
infinite, and subtracting that infinity when the row leaves gives a NaN rather
than giving the total back. Every window after that reads NaN until the window
empties. That is not a rounding difference, it is a wrong answer on real data,
and this file does not copy it. The infinities are counted rather than summed,
the total carries only the finite rows, and the answer is assembled from the two
at the end. A window holding one positive infinity sums to positive infinity, a
window holding both signs sums to a NaN, and both of those survive the infinity
leaving again.

That leaves the case where the finite rows themselves overflow. It is rare and
it is real, and the guard is to rebuild that one window from its rows, which
costs the width of the window on the rows where it fires and nothing anywhere
else.

## The extremes are not summable and are not recomputed either

A maximum cannot be undone: knowing the maximum of a window and the value that
just left does not give the maximum of what remains. The standing answer is a
deque of row numbers whose values decrease from front to back, where a row
arriving evicts every row behind it that it beats, since a row that is both
older and smaller can never be the answer again. Every row is pushed once and
popped once, so the whole pass is the height of the column no matter how wide
the window is.

pandas does the same thing and then loses the two infinities on the way out,
because it starts its running extreme at negative infinity and reads a result
equal to that sentinel as an empty window. So `rolling(3).max()` over a column
holding a positive infinity answers a NaN there. This file answers the infinity.

## What a missing row is

Whatever `present_bitmap_any` says, which on a float column means a NaN is
missing and not a value. That is the library rule and it is also pandas', so
none of the reductions here has to think about NaN at all: a missing row is not
added to the total, not pushed into the deque and not counted toward
`min_periods`.

`count` is the one that reads differently, and the difference is not a decision
made here. pandas computes it as a rolling sum over the presence indicator,
which is a column with nothing missing in it, so what `min_periods` is tested
against is how many rows the window covers rather than how many of them hold a
value. A window of three rows holding one value counts one, and a window of two
rows holding two values counts nothing when three were asked for.

## The answer is always float64

Every one of these answers float64 in pandas, including `count` and including
the extremes over an integer column, so the column is cast once on the way in
and there is one loop rather than one per dtype. A missing row in the answer is
a NaN with no validity bit behind it, which is what every float column in this
package holds and what `nan_over_nulls` exists to say.

## What is not here

The spread reductions, which are `std`, `var`, `sem`, `skew` and `kurt`. They
carry more state and they square the error, so a carried variance is a different
argument from a carried sum and it belongs in its own file next to its own
tests. The order statistics, `median`, `quantile` and `rank`, which need the
window sorted rather than folded and are a different data structure again. The
exponentially weighted window, which has no edges at all and so has nothing to
do with this file. And the windows given as a frequency rather than a count,
which need a calendar before they need any of this.
"""

from std.math import isinf, isnan, nan

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.logical import LogicalType

from .cast import cast_any
from .nulls import present_bitmap_any

comptime OP_SUM = 0
"""Operation code for the total over the window."""

comptime OP_MEAN = 1
"""Operation code for the mean over the window."""

comptime OP_COUNT = 2
"""Operation code for how many rows of the window hold a value."""

comptime OP_MIN = 3
"""Operation code for the smallest value in the window."""

comptime OP_MAX = 4
"""Operation code for the largest value in the window."""

comptime EDGE_RIGHT = 0
"""Closed code for a window that drops the first row of its span."""

comptime EDGE_LEFT = 1
"""Closed code for a window that drops the last row of its span."""

comptime EDGE_BOTH = 2
"""Closed code for a window that keeps both ends of its span."""

comptime EDGE_NEITHER = 3
"""Closed code for a window that drops both ends of its span."""


@fieldwise_init
struct WindowOp(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which reduction a window is being run through.

    Held as a code for the same reason `CumulativeOp` is, which is that the
    erased entry point takes it as an ordinary argument while the loops below
    take it as a decision made once before any row is read.
    """

    var code: Int
    """The operation, as one of the five values below."""

    comptime SUM = Self(OP_SUM)
    """The total over the window."""

    comptime MEAN = Self(OP_MEAN)
    """The mean over the window."""

    comptime COUNT = Self(OP_COUNT)
    """How many rows of the window hold a value."""

    comptime MIN = Self(OP_MIN)
    """The smallest value in the window."""

    comptime MAX = Self(OP_MAX)
    """The largest value in the window."""

    def __eq__(self, other: Self) -> Bool:
        """Compares two operations.

        Args:
            other: The operation to compare against.

        Returns:
            True if they are the same operation.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two operations for difference.

        Args:
            other: The operation to compare against.

        Returns:
            True if they are different operations.
        """
        return self.code != other.code

    def write_to(self, mut writer: Some[Writer]):
        """Writes the operation as the pandas method it answers.

        Args:
            writer: Where to write.
        """
        if self == Self.SUM:
            writer.write("sum")
        elif self == Self.MEAN:
            writer.write("mean")
        elif self == Self.COUNT:
            writer.write("count")
        elif self == Self.MIN:
            writer.write("min")
        else:
            writer.write("max")


def op_named(name: StringSlice) raises -> WindowOp:
    """Turns a pandas method name into a reduction.

    Args:
        name: The method name.

    Returns:
        The reduction it names.

    Raises:
        Error: If it is not one of the five this file answers.
    """
    if name == "sum":
        return WindowOp.SUM
    if name == "mean":
        return WindowOp.MEAN
    if name == "count":
        return WindowOp.COUNT
    if name == "min":
        return WindowOp.MIN
    if name == "max":
        return WindowOp.MAX
    raise Error("window: no window reduction is called " + String(name))


@fieldwise_init
struct WindowEdge(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which of its two ends a window keeps.

    pandas spells these as the four words below and means the four half open
    and closed intervals they name. Only the ends move; the width the caller
    asked for is the distance between them before either one is moved, which is
    why `both` holds one row more than the width and `neither` one row less.
    """

    var code: Int
    """The rule, as one of the four values below."""

    comptime RIGHT = Self(EDGE_RIGHT)
    """The default, which drops the first row of the span and keeps the last."""

    comptime LEFT = Self(EDGE_LEFT)
    """Keeps the first row of the span and drops the last."""

    comptime BOTH = Self(EDGE_BOTH)
    """Keeps both, so the window is one row wider than the width."""

    comptime NEITHER = Self(EDGE_NEITHER)
    """Drops both, so the window is one row narrower than the width."""

    def __eq__(self, other: Self) -> Bool:
        """Compares two rules.

        Args:
            other: The rule to compare against.

        Returns:
            True if they are the same rule.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two rules for difference.

        Args:
            other: The rule to compare against.

        Returns:
            True if they are different rules.
        """
        return self.code != other.code

    def write_to(self, mut writer: Some[Writer]):
        """Writes the rule as pandas spells it.

        Args:
            writer: Where to write.
        """
        if self == Self.RIGHT:
            writer.write("right")
        elif self == Self.LEFT:
            writer.write("left")
        elif self == Self.BOTH:
            writer.write("both")
        else:
            writer.write("neither")


def edge_named(name: StringSlice) raises -> WindowEdge:
    """Turns one of pandas' four words into a rule.

    Args:
        name: The word.

    Returns:
        The rule it names.

    Raises:
        Error: If it is not one of the four.
    """
    if name == "right":
        return WindowEdge.RIGHT
    if name == "left":
        return WindowEdge.LEFT
    if name == "both":
        return WindowEdge.BOTH
    if name == "neither":
        return WindowEdge.NEITHER
    raise Error(
        "window: closed is one of right, left, both or neither, and not "
        + String(name)
    )


@fieldwise_init
struct Edges(ImplicitlyCopyable, Movable, Sized):
    """The first row of a window and the row after its last."""

    var start: Int
    """The first row the window covers."""

    var stop: Int
    """One past the last row the window covers."""

    def __len__(self) -> Int:
        """Returns how many rows the window covers.

        Returns:
            The number of rows, which is zero for an empty window.
        """
        return self.stop - self.start


@fieldwise_init
struct Shape(ImplicitlyCopyable, Movable):
    """The five parameters that decide where a window sits.

    An expanding window is one of these with the width set to the height of the
    column, which puts its near end at row zero for every row and is exactly
    what an expanding window is. There is no second struct and no second loop.
    """

    var window: Int
    """How many rows wide, before the ends are moved."""

    var min_periods: Int
    """How many values a window needs before it answers anything."""

    var center: Bool
    """Whether the window sits around its row rather than behind it."""

    var closed: WindowEdge
    """Which of its two ends the window keeps."""

    var step: Int
    """How many rows apart the answered rows are."""

    def edges(self, row: Int, rows: Int) -> Edges:
        """Returns the window that ends at a row.

        The near end is derived from the far end before either is clipped, so a
        centred window that runs off the bottom of the column gets shorter
        rather than sliding back up to keep its width. That is pandas and it is
        the part of `center` nobody guesses right.

        Args:
            row: The row the window ends at.
            rows: How tall the column is.

        Returns:
            The rows the window covers, clipped to the column.
        """
        var stop = row + 1
        if self.center:
            stop += (self.window - 1) // 2
        var start = stop - self.window
        if self.closed == WindowEdge.LEFT or self.closed == WindowEdge.BOTH:
            start -= 1
        if self.closed == WindowEdge.LEFT or self.closed == WindowEdge.NEITHER:
            stop -= 1
        start = min(max(start, 0), rows)
        stop = min(max(stop, 0), rows)
        return Edges(start, max(stop, start))

    def answered(self, rows: Int) -> Int:
        """Returns how many rows the answer has.

        Args:
            rows: How tall the column is.

        Returns:
            One row per step, counting from the top of the column.
        """
        if rows <= 0:
            return 0
        return (rows + self.step - 1) // self.step


def rolling_shape(
    window: Int,
    min_periods: Optional[Int],
    center: Bool,
    closed: WindowEdge,
    step: Optional[Int],
) raises -> Shape:
    """Checks a caller's five parameters and fills in the two defaults.

    A rolling window that is not told how many values it needs needs all of
    them, which is why the leading rows of a rolling sum are missing and why
    `min_periods=1` is such a different answer over the same column.

    Args:
        window: How many rows wide.
        min_periods: How many values a window needs, or nothing for the width.
        center: Whether the window sits around its row.
        closed: Which ends the window keeps.
        step: How many rows apart the answered rows are, or nothing for one.

    Returns:
        The shape.

    Raises:
        Error: If the width is negative, the step is not positive, or more
            values are required than the window can hold.
    """
    if window < 0:
        raise Error("window: window must be zero or more")
    var taken = step.value() if step else 1
    if taken < 1:
        raise Error("window: step must be one or more")
    var needed = min_periods.value() if min_periods else window
    if needed < 0:
        raise Error("window: min_periods must be zero or more")
    if needed > window:
        raise Error("window: min_periods must not be larger than window")
    return Shape(window, needed, center, closed, taken)


def expanding_shape(min_periods: Int, rows: Int) raises -> Shape:
    """Builds the shape of an expanding window over a column of a given height.

    The width is the height of the column, which puts the near end at row zero
    everywhere, and `min_periods` defaults to one rather than to the width, so
    the first row of an expanding sum is the first value and not a hole.

    Args:
        min_periods: How many values a window needs.
        rows: How tall the column is.

    Returns:
        The shape.

    Raises:
        Error: If fewer than zero values are required.
    """
    if min_periods < 0:
        raise Error("window: min_periods must be zero or more")
    return Shape(
        max(rows, min_periods), min_periods, False, WindowEdge.RIGHT, 1
    )


def window_agg(col: AnyArray, op: WindowOp, shape: Shape) raises -> AnyArray:
    """Runs a reduction over every window of a column.

    Args:
        col: The column.
        op: Which reduction to run.
        shape: Where the windows sit.

    Returns:
        A float64 column with one row per step, holding a NaN wherever the
        window had fewer values than it needed.

    Raises:
        Error: If the column is not something a window can reduce, or the cast
            to float64 fails.
    """
    if (
        col.is_string()
        or col.type.is_temporal()
        or not (col.type.is_numeric() or col.type == LogicalType.BOOL)
    ):
        raise Error(
            "window: " + String(op) + " is not defined on " + String(col.type)
        )
    var rows = len(col)
    var present = present_bitmap_any(col)
    var seen = _running_count(present, rows)
    if op == WindowOp.COUNT:
        return AnyArray(_count(seen, rows, shape))
    var wide = AnyArray(copy=col) if col.dtype() == DType.float64 else cast_any(
        col, DType.float64
    )
    var src = wide.unsafe_ptr[DType.float64]()
    if op == WindowOp.MIN or op == WindowOp.MAX:
        return AnyArray(
            _extreme(src, present, seen, rows, shape, op == WindowOp.MAX)
        )
    return AnyArray(
        _total(src, present, seen, rows, shape, op == WindowOp.MEAN)
    )


def _running_count(present: Bitmap, rows: Int) -> List[Int]:
    """Counts the rows holding a value up to each row, so a window can subtract.

    One pass and one integer per row buys every window its count of values in
    two loads, which is what lets `step` jump the window forward without the
    count having to be walked. The reductions that carry state cannot be jumped
    that way and do not try.

    Args:
        present: Which rows hold a value.
        rows: How tall the column is.

    Returns:
        A list one longer than the column, where entry `i` is how many of the
        first `i` rows hold a value.
    """
    var seen = List[Int](length=rows + 1, fill=0)
    var running = 0
    for i in range(rows):
        if present.get(i):
            running += 1
        seen[i + 1] = running
    return seen^


def _count(
    seen: List[Int], rows: Int, shape: Shape
) raises -> Array[DType.float64]:
    """Answers how many rows of each window hold a value.

    The one reduction whose `min_periods` is tested against the height of the
    window rather than against how many values are in it. See the note at the
    top of the file: pandas gets that by counting a column that has nothing
    missing in it, and this is the same rule written out directly.

    Args:
        seen: The running count of rows holding a value.
        rows: How tall the column is.
        shape: Where the windows sit.

    Returns:
        A float64 column, holding a NaN wherever the window covered fewer rows
        than the number of periods asked for.

    Raises:
        Error: Only what allocation raises.
    """
    var answer = Array[DType.float64](shape.answered(rows))
    var target = answer.unsafe_ptr()
    for k in range(len(answer)):
        var span = shape.edges(k * shape.step, rows)
        if len(span) < shape.min_periods:
            target.unsafe_offset(k).unsafe_store(nan[DType.float64]())
        else:
            target.unsafe_offset(k).unsafe_store(
                Float64(seen[span.stop] - seen[span.start])
            )
    return answer^


struct _Total(ImplicitlyCopyable, Movable):
    """A running total that survives an infinity passing through it.

    The finite rows are summed with a compensation term, which is the ordinary
    trick for keeping the low bits that a floating point addition drops, and it
    matters more here than in a plain sum because a row is subtracted again
    later and the error of the subtraction does not cancel the error of the
    addition. The compensation is kept beside the total and added back at the
    end rather than being folded into the next row on the way in. That is the
    difference between this working and not working: folding it in rounds the
    row before it is added, so a small row arriving next to a large total loses
    its low bits twice and they are gone when the large row leaves the window.

    The two infinities are counted rather than added, which is the whole point:
    a count goes back down when the row leaves and an infinity in a sum does
    not.
    """

    var total: Float64
    """The sum of the finite rows in the window, before the low bits are added
    back."""

    var lost: Float64
    """The low bits the additions have dropped so far."""

    var highs: Int
    """How many positive infinities the window holds."""

    var lows: Int
    """How many negative infinities the window holds."""

    def __init__(out self):
        """Starts a total with nothing in it."""
        self.total = 0.0
        self.lost = 0.0
        self.highs = 0
        self.lows = 0

    def add(mut self, value: Float64):
        """Folds one row into the total.

        Args:
            value: The row's value, which is never a NaN because a NaN is not a
                row that holds a value.
        """
        if isinf(value):
            if value > 0:
                self.highs += 1
            else:
                self.lows += 1
            return
        var moved = self.total + value
        # Whichever of the two is larger is the one whose low bits survived the
        # addition, so the other one is where the dropped bits have to be read
        # back from.
        if abs(self.total) >= abs(value):
            self.lost += (self.total - moved) + value
        else:
            self.lost += (value - moved) + self.total
        if isnan(self.lost):
            self.lost = 0.0
        self.total = moved

    def drop(mut self, value: Float64):
        """Takes one row back out of the total.

        Args:
            value: The row's value.
        """
        if isinf(value):
            if value > 0:
                self.highs -= 1
            else:
                self.lows -= 1
            return
        self.add(-value)

    def answer(self) -> Float64:
        """Returns what the window sums to.

        Returns:
            A NaN when the window holds infinities of both signs, the infinity
            when it holds one sign, and the compensated total otherwise.
        """
        if self.highs > 0 and self.lows > 0:
            return nan[DType.float64]()
        if self.highs > 0:
            # The maximum of a float dtype is positive infinity rather than the
            # largest finite value, which is the same fact `accum.mojo` leans on
            # for the identity of a maximum.
            return Float64.MAX
        if self.lows > 0:
            return Float64.MIN
        return self.total + self.lost

    def settled(self) -> Bool:
        """Says whether the carried total is still a number worth carrying.

        Returns:
            False once the finite rows have overflowed, which is when the
            caller has to rebuild the total from the window's own rows.
        """
        return not (
            isinf(self.total)
            or isnan(self.total)
            or isinf(self.lost)
            or isnan(self.lost)
        )


def _total[
    origin: ImmOrigin
](
    src: Pointer[Scalar[DType.float64], origin],
    present: Bitmap,
    seen: List[Int],
    rows: Int,
    shape: Shape,
    mean: Bool,
) raises -> Array[DType.float64]:
    """Sums or averages every window, carrying the total from one to the next.

    Args:
        src: The values, already float64.
        present: Which rows hold a value.
        seen: The running count of rows holding a value.
        rows: How tall the column is.
        shape: Where the windows sit.
        mean: Whether to divide by the number of values.

    Parameters:
        origin: The origin of the values.

    Returns:
        A float64 column with one row per step.

    Raises:
        Error: Only what allocation raises.
    """
    var answer = Array[DType.float64](shape.answered(rows))
    var target = answer.unsafe_ptr()
    var carried = _Total()
    var last = Edges(0, 0)
    for k in range(len(answer)):
        var span = shape.edges(k * shape.step, rows)
        if k == 0 or span.start >= last.stop:
            carried = _Total()
            _fold(src, present, span.start, span.stop, carried)
        else:
            for j in range(last.start, span.start):
                if present.get(j):
                    carried.drop(src.unsafe_offset(j).unsafe_load())
            for j in range(last.stop, span.stop):
                if present.get(j):
                    carried.add(src.unsafe_offset(j).unsafe_load())
        if not carried.settled():
            # The finite rows have overflowed, and a total that has reached
            # infinity cannot be brought back by subtracting the row that put it
            # there. Rebuilding from the window costs the width of the window
            # and only happens on the rows where the overflow is live.
            carried = _Total()
            _fold(src, present, span.start, span.stop, carried)
        last = span
        var found = seen[span.stop] - seen[span.start]
        var value = nan[DType.float64]()
        if found >= shape.min_periods:
            if found > 0:
                value = carried.answer()
                if mean:
                    value = value / Float64(found)
            elif not mean:
                # An empty window that was asked for nothing sums to zero, which
                # is the identity and is what pandas answers. An empty mean has
                # no such value and stays missing.
                value = 0.0
        target.unsafe_offset(k).unsafe_store(value)
    return answer^


def _fold[
    origin: ImmOrigin
](
    src: Pointer[Scalar[DType.float64], origin],
    present: Bitmap,
    start: Int,
    stop: Int,
    mut into: _Total,
):
    """Adds a run of rows to a total that was just started.

    Args:
        src: The values.
        present: Which rows hold a value.
        start: The first row.
        stop: One past the last row.
        into: The total to add them to.

    Parameters:
        origin: The origin of the values.
    """
    for j in range(start, stop):
        if present.get(j):
            into.add(src.unsafe_offset(j).unsafe_load())


def _extreme[
    origin: ImmOrigin
](
    src: Pointer[Scalar[DType.float64], origin],
    present: Bitmap,
    seen: List[Int],
    rows: Int,
    shape: Shape,
    largest: Bool,
) raises -> Array[DType.float64]:
    """Answers the smallest or largest value of every window.

    The deque holds row numbers and never values, so a comparison always reads
    the column and the deque stays one integer per row. A row arriving evicts
    every row behind it that it beats, which is what keeps the front of the
    deque the answer and what makes the whole pass linear: a row is pushed once
    and popped once no matter how many windows it belongs to.

    Args:
        src: The values, already float64.
        present: Which rows hold a value.
        seen: The running count of rows holding a value.
        rows: How tall the column is.
        shape: Where the windows sit.
        largest: Whether to answer the maximum rather than the minimum.

    Parameters:
        origin: The origin of the values.

    Returns:
        A float64 column with one row per step.

    Raises:
        Error: Only what allocation raises.
    """
    var answer = Array[DType.float64](shape.answered(rows))
    var target = answer.unsafe_ptr()
    var deck = List[Int](length=rows, fill=0)
    var head = 0
    var tail = 0
    var reached = 0
    for k in range(len(answer)):
        var span = shape.edges(k * shape.step, rows)
        if span.start > reached:
            head = 0
            tail = 0
            reached = span.start
        for j in range(max(reached, span.start), span.stop):
            if present.get(j):
                var arriving = src.unsafe_offset(j).unsafe_load()
                while tail > head:
                    var behind = src.unsafe_offset(deck[tail - 1]).unsafe_load()
                    var evicted = (
                        behind <= arriving if largest else behind >= arriving
                    )
                    if not evicted:
                        break
                    tail -= 1
                deck[tail] = j
                tail += 1
        reached = max(reached, span.stop)
        while tail > head and deck[head] < span.start:
            head += 1
        var found = seen[span.stop] - seen[span.start]
        if found < shape.min_periods or found == 0:
            target.unsafe_offset(k).unsafe_store(nan[DType.float64]())
        else:
            target.unsafe_offset(k).unsafe_store(
                src.unsafe_offset(deck[head]).unsafe_load()
            )
    return answer^
