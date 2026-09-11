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

## The moment reductions are here and their arithmetic is not

`var`, `std`, `sem`, `skew` and `kurt` walk the same edges as everything else
and are folded through the same loop shape, so they are dispatched from here.
What they carry is not a number, it is a state, and the argument about when
that state stops being worth carrying is long enough and separate enough to
live in `spread.mojo` next to its own tests. This file asks it whether it has
settled and rebuilds the window when it says no, which is the same conversation
the running total already has, and knows nothing else about how a variance is
computed.

The first three are also where the window parameters run out of room at the
Python boundary. `ddof` belongs to the reduction rather than to the window, and
adding it makes seven arguments after the object, which document 13 section 4
measured as the ceiling for a bound method. The eighth window parameter,
whichever it turns out to be, goes through a keyword route. `skew` and `kurt`
take no `ddof` in pandas and so cost nothing on that door.

## The order statistics are here on the same terms

`median` is the first reduction that is not a fold. What it carries is not a
number and not a state that can be corrected, it is a count of how many values
the window holds at each rank, and selecting the middle of that is a descent of
a tree rather than an arithmetic step. The structure and the three answers read
out of it live in `ordered.mojo`, and this file does for them what it does for
the spreads, which is to walk the edges and hand over the rows that changed.

The one respect in which they are easier is that there is nothing to rebuild.
A count is exact however long the pass runs, so the loop below has no bound to
check and no window to redo, and the rows to add and drop are written as the
difference between two windows rather than as two runs that assume an overlap.

## What is not here

`quantile` and `rank`, which are the other two order statistics and are waiting
on room at the Python boundary rather than on anything in the kernel. The
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
from .ordered import (
    BETWEEN_LINEAR,
    TIED_AVERAGE,
    Ordered,
    Ranks,
    halved,
    picked,
    placed,
    ranked,
)
from .spread import (
    MOMENT_KURT,
    MOMENT_SKEW,
    SPREAD_SEM,
    SPREAD_STD,
    SPREAD_VAR,
    Moments,
    Spread,
)

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

comptime OP_VAR = 5
"""Operation code for the variance of the window."""

comptime OP_STD = 6
"""Operation code for the standard deviation of the window."""

comptime OP_SEM = 7
"""Operation code for the standard error of the window's mean."""

comptime OP_SKEW = 8
"""Operation code for the skewness of the window."""

comptime OP_KURT = 9
"""Operation code for the excess kurtosis of the window."""

comptime OP_MEDIAN = 10
"""Operation code for the middle value of the window."""

comptime OP_QUANTILE = 11
"""Operation code for the value at a fraction of the way through the window."""

comptime OP_RANK = 12
"""Operation code for the position of the window's last row among its values."""

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
    """The operation, as one of the eleven values below."""

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

    comptime VAR = Self(OP_VAR)
    """The variance of the window."""

    comptime STD = Self(OP_STD)
    """The standard deviation of the window."""

    comptime SEM = Self(OP_SEM)
    """The standard error of the window's mean."""

    comptime SKEW = Self(OP_SKEW)
    """The skewness of the window."""

    comptime KURT = Self(OP_KURT)
    """The excess kurtosis of the window."""

    comptime MEDIAN = Self(OP_MEDIAN)
    """The middle value of the window."""

    comptime QUANTILE = Self(OP_QUANTILE)
    """The value at a fraction of the way through the window."""

    comptime RANK = Self(OP_RANK)
    """The position of the window's last row among the window's values."""

    def spreads(self) -> Bool:
        """Says whether this reduction measures a spread rather than a level.

        Returns:
            True for the variance, the deviation and the standard error, which
            are the three that carry a state rather than a number and are the
            three that read `ddof`.
        """
        return self.code >= OP_VAR and self.code <= OP_SEM

    def shapes(self) -> Bool:
        """Says whether this reduction measures the shape of the window.

        Returns:
            True for the skewness and the kurtosis, which carry a third and a
            fourth moment beside the spread and which take no `ddof`, because
            pandas gives neither of them one.
        """
        return self.code == OP_SKEW or self.code == OP_KURT

    def orders(self) -> Bool:
        """Says whether this reduction asks about the order of the values.

        Returns:
            True for the median, the quantile and the rank, which read a
            position in the sorted window rather than folding the window into a
            number and so run through `ordered.mojo`. Written as three
            comparisons rather than as one against `OP_MEDIAN`, for the reason
            `shapes` is: the codes past the last one of a group are the codes
            that have not been added yet.
        """
        return (
            self.code == OP_MEDIAN
            or self.code == OP_QUANTILE
            or self.code == OP_RANK
        )

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
        elif self == Self.MAX:
            writer.write("max")
        elif self == Self.VAR:
            writer.write("var")
        elif self == Self.STD:
            writer.write("std")
        elif self == Self.SEM:
            writer.write("sem")
        elif self == Self.SKEW:
            writer.write("skew")
        elif self == Self.KURT:
            writer.write("kurt")
        elif self == Self.MEDIAN:
            writer.write("median")
        elif self == Self.QUANTILE:
            writer.write("quantile")
        else:
            writer.write("rank")


def op_named(name: StringSlice) raises -> WindowOp:
    """Turns a pandas method name into a reduction.

    Args:
        name: The method name.

    Returns:
        The reduction it names.

    Raises:
        Error: If it is not one of the thirteen this file answers.
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
    if name == "var":
        return WindowOp.VAR
    if name == "std":
        return WindowOp.STD
    if name == "sem":
        return WindowOp.SEM
    if name == "skew":
        return WindowOp.SKEW
    if name == "kurt":
        return WindowOp.KURT
    if name == "median":
        return WindowOp.MEDIAN
    if name == "quantile":
        return WindowOp.QUANTILE
    if name == "rank":
        return WindowOp.RANK
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
struct WindowSettings(ImplicitlyCopyable, Movable):
    """The parameters that belong to a reduction rather than to a window.

    Five of the thirteen reductions read something the window itself has no
    opinion about. The three spreads read a degrees of freedom, the quantile
    reads a fraction and a rule for landing between two values, and the rank
    reads a tie rule, a direction and whether to divide by the count. The other
    eight read none of it.

    They are one value rather than six arguments because of the Python door. A
    bound method gets seven real arguments after the object, which document 13
    section 4 measured, and the window itself already needs six of them, so the
    seventh is the whole budget for every reduction's own parameters put
    together. Passing them as separate arguments worked for exactly as long as
    there was one of them.

    So the seventh slot carries a tuple, `firepanda/py/window.mojo` reads it
    apart against the reduction's name, and from that point inwards these are
    ordinary typed fields. Every one of them has a default that is the default
    pandas documents, which is what lets the eight reductions that read none of
    them pass nothing at all.
    """

    var ddof: Int
    """Subtracted from the count of values to give the divisor of a variance."""

    var fraction: Float64
    """How far through the sorted window a quantile reads, nought to one."""

    var between: Int
    """Which of the five rules a quantile uses between two values."""

    var tied: Int
    """Which of the three rules a rank uses for values that are equal."""

    var ascending: Bool
    """Whether a rank counts from the smallest value rather than the largest."""

    var pct: Bool
    """Whether a rank is divided by how many values the window holds."""

    def __init__(out self):
        """Builds the settings every reduction gets when it asks for nothing.

        The fraction is a half so that a quantile with no fraction given reads
        the middle, which is not a default pandas has, because pandas makes the
        fraction required. It is here so that the value is a number rather than
        whatever was in the field, and nothing reaches `picked` without a
        fraction having crossed the door.
        """
        self = Self(1, 0.5, BETWEEN_LINEAR, TIED_AVERAGE, True, False)


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


def window_agg(
    col: AnyArray,
    op: WindowOp,
    shape: Shape,
    settings: WindowSettings = WindowSettings(),
) raises -> AnyArray:
    """Runs a reduction over every window of a column.

    Args:
        col: The column.
        op: Which reduction to run.
        shape: Where the windows sit.
        settings: The parameters the reduction reads and the window does not.
            Five of the thirteen read something out of it and the other eight
            read nothing, which is why it has a default and they pass none.

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
    if op.spreads():
        var form = SPREAD_VAR
        if op == WindowOp.STD:
            form = SPREAD_STD
        elif op == WindowOp.SEM:
            form = SPREAD_SEM
        return AnyArray(
            _spread(src, present, seen, rows, shape, settings.ddof, form)
        )
    if op.shapes():
        var form = MOMENT_SKEW if op == WindowOp.SKEW else MOMENT_KURT
        return AnyArray(_moments(src, present, seen, rows, shape, form))
    if op.orders():
        return AnyArray(_ordered(src, present, seen, rows, shape, op, settings))
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
    var target = answer.unsafe_mut_ptr()
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
    var target = answer.unsafe_mut_ptr()
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


def _spread[
    origin: ImmOrigin
](
    src: Pointer[Scalar[DType.float64], origin],
    present: Bitmap,
    seen: List[Int],
    rows: Int,
    shape: Shape,
    ddof: Int,
    form: Int,
) raises -> Array[DType.float64]:
    """Measures the spread of every window, carrying the state between them.

    The same loop as `_total` with a different thing carried, and the difference
    worth noticing is where the rebuild sits. The total rebuilds when it has
    overflowed, which is rare. The spread rebuilds when its error bound has
    grown past the eighth digit of its answer, which happens whenever a value
    much larger than the rest leaves the window, and that is not rare. Both cost
    the width of the window on the rows where they fire and nothing anywhere
    else.

    Args:
        src: The values, already float64.
        present: Which rows hold a value.
        seen: The running count of rows holding a value.
        rows: How tall the column is.
        shape: Where the windows sit.
        ddof: Subtracted from the count to give the divisor.
        form: Which of the three spreads to answer.

    Parameters:
        origin: The origin of the values.

    Returns:
        A float64 column with one row per step.

    Raises:
        Error: Only what allocation raises.
    """
    var answer = Array[DType.float64](shape.answered(rows))
    var target = answer.unsafe_mut_ptr()
    var carried = Spread()
    var last = Edges(0, 0)
    for k in range(len(answer)):
        var span = shape.edges(k * shape.step, rows)
        if k == 0 or span.start >= last.stop:
            carried = Spread()
            _gather(src, present, span.start, span.stop, carried)
        else:
            for j in range(last.start, span.start):
                if present.get(j):
                    carried.drop(src.unsafe_offset(j).unsafe_load())
            for j in range(last.stop, span.stop):
                if present.get(j):
                    carried.add(src.unsafe_offset(j).unsafe_load())
            if not carried.settled():
                carried = Spread()
                _gather(src, present, span.start, span.stop, carried)
        last = span
        var found = seen[span.stop] - seen[span.start]
        var value = nan[DType.float64]()
        if found >= shape.min_periods:
            value = carried.answer(found, ddof, form)
        target.unsafe_offset(k).unsafe_store(value)
    return answer^


def _gather[
    origin: ImmOrigin
](
    src: Pointer[Scalar[DType.float64], origin],
    present: Bitmap,
    start: Int,
    stop: Int,
    mut into: Spread,
):
    """Adds a run of rows to a spread that was just started.

    Args:
        src: The values.
        present: Which rows hold a value.
        start: The first row.
        stop: One past the last row.
        into: The spread to add them to.

    Parameters:
        origin: The origin of the values.
    """
    for j in range(start, stop):
        if present.get(j):
            into.add(src.unsafe_offset(j).unsafe_load())


def _moments[
    origin: ImmOrigin
](
    src: Pointer[Scalar[DType.float64], origin],
    present: Bitmap,
    seen: List[Int],
    rows: Int,
    shape: Shape,
    form: Int,
) raises -> Array[DType.float64]:
    """Measures the shape of every window, carrying the state between them.

    The same loop as `_spread` again, over a state that carries two more
    numbers.
    There is no `ddof` here because neither pandas method takes one, and the
    rebuild fires on whichever of the two moments has lost the most, so a fourth
    moment that has given up takes the third down with it rather than the third
    being trusted on its own.

    Args:
        src: The values, already float64.
        present: Which rows hold a value.
        seen: The running count of rows holding a value.
        rows: How tall the column is.
        shape: Where the windows sit.
        form: Which of the two shapes to answer.

    Parameters:
        origin: The origin of the values.

    Returns:
        A float64 column with one row per step.

    Raises:
        Error: Only what allocation raises.
    """
    var answer = Array[DType.float64](shape.answered(rows))
    var target = answer.unsafe_mut_ptr()
    var carried = Moments()
    var last = Edges(0, 0)
    for k in range(len(answer)):
        var span = shape.edges(k * shape.step, rows)
        if k == 0 or span.start >= last.stop:
            carried = Moments()
            _gather(src, present, span.start, span.stop, carried)
        else:
            for j in range(last.start, span.start):
                if present.get(j):
                    carried.drop(src.unsafe_offset(j).unsafe_load())
            for j in range(last.stop, span.stop):
                if present.get(j):
                    carried.add(src.unsafe_offset(j).unsafe_load())
            if not carried.settled():
                carried = Moments()
                _gather(src, present, span.start, span.stop, carried)
        last = span
        var found = seen[span.stop] - seen[span.start]
        var value = nan[DType.float64]()
        if found >= shape.min_periods:
            value = carried.answer(found, form)
        target.unsafe_offset(k).unsafe_store(value)
    return answer^


def _gather[
    origin: ImmOrigin
](
    src: Pointer[Scalar[DType.float64], origin],
    present: Bitmap,
    start: Int,
    stop: Int,
    mut into: Moments,
):
    """Adds a run of rows to a moment state that was just started.

    Args:
        src: The values.
        present: Which rows hold a value.
        start: The first row.
        stop: One past the last row.
        into: The moments to add them to.

    Parameters:
        origin: The origin of the values.
    """
    for j in range(start, stop):
        if present.get(j):
            into.add(src.unsafe_offset(j).unsafe_load())


def _ordered[
    origin: ImmOrigin
](
    src: Pointer[Scalar[DType.float64], origin],
    present: Bitmap,
    seen: List[Int],
    rows: Int,
    shape: Shape,
    op: WindowOp,
    settings: WindowSettings,
) raises -> Array[DType.float64]:
    """Reads a position out of every window, carrying the counts between them.

    The same walk as `_spread` with two differences. What is carried is a count
    of the window's values by rank rather than a number, so there is no bound to
    check and no window that has to be rebuilt. And the rows that changed are
    written as the difference between the last window and this one rather than
    as two runs either side of it, which is the same thing whenever the two
    overlap and is still right when a step has moved the window clear of where
    it was. Doing it that way costs one comparison per row and saves clearing
    the tree, which would be the number of distinct values in the whole column
    every time a stepped window jumped.

    The three reductions differ only in what they ask the tree at the end, which
    is why they are one loop. The median reads the middle, the quantile reads a
    fraction of the way through, and the rank asks where one particular value
    sits. That value is the one in the window's last row and not the one in the
    row being answered, which is pandas' rule and is a real difference under
    `center`, under a step and under the two closed rules that drop the
    answered row: `rolling(3, closed='left').rank()` ranks the row before the
    window's near end. If that row holds nothing the answer is missing however
    many values the window holds, because there is no value to place.

    Args:
        src: The values, already float64.
        present: Which rows hold a value.
        seen: The running count of rows holding a value.
        rows: How tall the column is.
        shape: Where the windows sit.
        op: Which of the three order statistics to answer.
        settings: The fraction, the rule between two values, and the three a
            rank reads.

    Parameters:
        origin: The origin of the values.

    Returns:
        A float64 column with one row per step.

    Raises:
        Error: Only what allocation raises.
    """
    var answer = Array[DType.float64](shape.answered(rows))
    var target = answer.unsafe_mut_ptr()
    var ranks = ranked(src, present, rows)
    var carried = Ordered(len(ranks.values))
    var last = Edges(0, 0)
    for k in range(len(answer)):
        var span = shape.edges(k * shape.step, rows)
        for j in range(last.start, min(last.stop, span.start)):
            if present.get(j):
                carried.drop(Int(ranks.of_row[j]))
        for j in range(max(span.start, last.stop), span.stop):
            if present.get(j):
                carried.add(Int(ranks.of_row[j]))
        last = span
        var found = seen[span.stop] - seen[span.start]
        var value = nan[DType.float64]()
        if found >= shape.min_periods and found > 0:
            if op == WindowOp.MEDIAN:
                value = halved(carried, ranks, found)
            elif op == WindowOp.QUANTILE:
                value = picked(
                    carried,
                    ranks,
                    found,
                    settings.fraction,
                    settings.between,
                )
            elif present.get(span.stop - 1):
                value = placed(
                    carried,
                    Int(ranks.of_row[span.stop - 1]),
                    found,
                    settings.tied,
                    settings.ascending,
                    settings.pct,
                )
        target.unsafe_offset(k).unsafe_store(value)
    return answer^


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
    var target = answer.unsafe_mut_ptr()
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
