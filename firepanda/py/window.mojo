"""The one door behind `Rolling`, `Expanding` and the fifty three names on them.

pandas gives a caller two objects, `s.rolling(5)` and `s.expanding()`, and then
puts the same list of reductions on each. The two objects differ in one thing,
which is where the near end of the window sits, and the reductions do not differ
at all. So there is one crossing here rather than two, and the width is what
says which of the two was asked for.

### Why an absent width means an expanding window

The same reason an absent bound in `text.mojo` means the far end. An expanding
window is one with no left edge, and a width of nothing is how you write that.
The alternative is a flag beside the width saying which of them to believe,
which is a parameter that exists to say that another parameter does not, and
those are the ones that get out of step.

The kernel then turns the absence into the height of the column, and from that
point on there is one window type and one loop. `firepanda/kernel/window.mojo`
argues that at length and this file is only where the argument is spelled in
Python's vocabulary.

### Why min_periods is also allowed to be absent

Because its default is not one number. A rolling window that is not told needs
all of its rows, and an expanding window that is not told needs one. Those are
pandas' two defaults and they are genuinely different questions, so the value
that means neither of them has to survive the crossing rather than being filled
in on the Python side by a layer that would then own the rule.

### What is not here

`skew` and `kurt`, which carry the third and fourth moments. `median`, `quantile`
and `rank`, which need the window sorted. `apply`, `corr` and `cov`. The
exponentially weighted window, which has no edges and so shares nothing with any
of this. And a window given as a frequency, which needs a calendar first. None of
them resolves rather than resolving and refusing, for the reason document 07
gives.

### Why ddof is the last parameter this door can take

`std`, `var` and `sem` read a degrees of freedom, which is a property of the
reduction rather than of the window, and it arrives here anyway because there is
one door and not fifty three. With it the bound method on the Python side has
seven arguments after the object, which document 13 section 4 measured as the
most a bound method can have. Whatever the next window parameter turns out to be,
it comes through keyword arguments, which do not count against that.
"""

from firepanda.dtype.logical import LogicalType
from firepanda.frame.frame import DataFrame
from firepanda.frame.index import Index
from firepanda.frame.series import Series
from firepanda.kernel.window import edge_named, op_named
from firepanda.py.errors import DTYPE, VALUE, tagged


def _reducible(column: Series) raises:
    """Refuses a column no window reduction has an answer for.

    Checked here rather than left to the kernel so that the refusal reaches
    Python as the type error pandas raises. pandas calls this one `DataError`
    and puts it under `TypeError`, which is where a caller's except clause is
    going to be looking.

    The column's name is in the message, which pandas' own is not. On a column
    the caller knows which one they asked about. On a frame they do not, and
    `Cannot aggregate non-numeric type: str` over a frame of forty columns is a
    sentence that sends the reader back to look for the column themselves.

    Args:
        column: The column.

    Raises:
        Error: Tagged `dtype` if the column is not a number or a bool.
    """
    var t = column.values.type
    if t.is_numeric() or t == LogicalType.BOOL:
        return
    raise tagged(
        DTYPE,
        String(
            "window: there is nothing to aggregate in column '",
            column.name,
            "', which holds ",
            t,
        ),
    )


def window(
    column: Series,
    kind: String,
    width: Optional[Int],
    min_periods: Optional[Int],
    center: Bool,
    closed: String,
    step: Optional[Int],
    ddof: Int,
) raises -> Series:
    """Runs one reduction over every window of a column.

    Args:
        column: The column to read.
        kind: The reduction, as pandas spells the method.
        width: How many rows wide the window is, or nothing for an expanding
            window.
        min_periods: How many values a window needs, or nothing for the default
            of whichever window type this is.
        center: Whether the window sits around its row rather than behind it.
        closed: Which of the two ends the window keeps, as one of pandas' four
            words.
        step: How many rows apart the answered rows are, or nothing for every
            row.
        ddof: Subtracted from the count of values to give the divisor of a
            variance, read by `std`, `var` and `sem` and ignored by the rest.

    Returns:
        A float64 column, as tall as the one it read unless a step made it
        shorter.

    Raises:
        Error: Tagged `dtype` if the column holds something no window can
            reduce, and tagged `value` if the reduction has no name, the closed
            rule has no name, or the parameters do not describe a window.
    """
    _reducible(column)
    try:
        var op = op_named(kind)
        if width:
            return column.rolling(
                op,
                width.value(),
                min_periods,
                center,
                edge_named(closed),
                step,
                ddof,
            )
        if center or step:
            # pandas has no place to put either of these on an expanding
            # window, so a caller who reached this asked for a window that does
            # not exist rather than for one this library has not written.
            raise Error(
                "window: an expanding window takes neither center nor step"
            )
        return column.expanding(
            op, min_periods.value() if min_periods else 1, ddof
        )
    except e:
        raise tagged(VALUE, String(e))


def window_frame(
    frame: DataFrame,
    kind: String,
    width: Optional[Int],
    min_periods: Optional[Int],
    center: Bool,
    closed: String,
    step: Optional[Int],
    ddof: Int,
) raises -> DataFrame:
    """Runs one reduction over every window of every column.

    A frame window is the columns windowed one at a time and put back together,
    which is what pandas does and is the same arrangement `transform` uses for
    the eleven per column transformations. There is nothing a frame window can
    do that a column one cannot, because a window is a pair of row numbers and
    every column of a frame has the same rows.

    The one thing that is not per column is the refusal. Every column is checked
    before any of them is read, so a frame with a text column in the middle of
    it raises instead of computing half an answer and then raising. pandas does
    the same and it matters more here than it looks: the reductions are cheap
    and the wasted work is not the point, the point is that a caller who gets an
    error should not have to wonder what was already spent.

    Args:
        frame: The frame to read.
        kind: The reduction, as pandas spells the method.
        width: How many rows wide the window is, or nothing for an expanding
            window.
        min_periods: How many values a window needs, or nothing for the default
            of whichever window type this is.
        center: Whether the window sits around its row rather than behind it.
        closed: Which of the two ends the window keeps, as one of pandas' four
            words.
        step: How many rows apart the answered rows are, or nothing for every
            row.
        ddof: Subtracted from the count of values to give the divisor of a
            variance, read by `std`, `var` and `sem` and ignored by the rest.

    Returns:
        A frame of the same column names in the same order, every one of them
        float64, as tall as the frame it read unless a step made it shorter.

    Raises:
        Error: Tagged `dtype` if any column holds something no window can
            reduce, and tagged `value` if the reduction has no name, the closed
            rule has no name, or the parameters do not describe a window.
    """
    for i in range(frame.width()):
        _reducible(frame.column(frame.schema[i].name))
    if frame.width() == 0:
        # `from_series` with nothing in it makes a frame of no rows, which is a
        # different answer from a frame of no columns. Handing the frame back is
        # the honest one: there was nothing to reduce and nothing was reduced.
        return DataFrame(copy=frame)
    var parts = List[Series](capacity=frame.width())
    for i in range(frame.width()):
        parts.append(
            window(
                frame.column(frame.schema[i].name),
                kind,
                width,
                min_periods,
                center,
                closed,
                step,
                ddof,
            )
        )
    # Taken off the first answer rather than off the frame, because a step makes
    # the answer shorter and gives it the labels of the rows it sampled, and
    # `from_series` takes the values out of each column and leaves the labels
    # behind.
    var labels = Index(copy=parts[0].index)
    var out = DataFrame.from_series(parts^)
    out.index = labels^
    return out^
