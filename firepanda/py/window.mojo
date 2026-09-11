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

`apply`, `corr` and `cov`. The exponentially weighted window, which has no edges
and so shares nothing with any of this. And a window given as a frequency, which
needs a calendar first. None of them resolves rather than resolving and
refusing, for the reason document 07 gives.

### Why the reduction's own parameters arrive as one value

Five of the thirteen reductions read something the window has no opinion about.
`std`, `var` and `sem` read a degrees of freedom, `quantile` reads a fraction
and a rule for landing between two values, and `rank` reads a tie rule, a
direction and whether to divide by the count. All of it arrives here anyway,
because there is one door and not fifty three.

It cannot arrive as six arguments. The window itself needs six of the seven a
bound method gets after the object, which document 13 section 4 measured, so the
seventh slot is the entire budget for every reduction's own parameters put
together. That worked while there was one of them and stopped working the day
there were two.

So the seventh slot is a tuple and `window_settings` below reads it apart
against the reduction's name. A reduction that reads nothing sends an empty one,
the five that read something send theirs in the order pandas declares them, and
the door stays at seven however many reductions land on it.
"""

from std.python import PythonObject

from firepanda.dtype.logical import LogicalType
from firepanda.frame.frame import DataFrame
from firepanda.frame.index import Index
from firepanda.frame.series import Series
from firepanda.kernel.ordered import between_named, tied_named
from firepanda.kernel.window import WindowSettings, edge_named, op_named
from firepanda.py.args import flag, number, whole, words
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


def window_settings(
    kind: String, settings: PythonObject
) raises -> WindowSettings:
    """Reads a reduction's own parameters out of the tuple they arrived in.

    The tuple is positional and its length is decided by the reduction, so the
    length is checked before anything is read out of it. A caller cannot reach
    this with the wrong length, because the Python layer builds the tuple from
    the method's own declared parameters, which means a mismatch here is the two
    halves of the library disagreeing about a reduction and is worth saying so.

    The words and the fraction are checked again here, having already been
    checked on the Python side where pandas' own sentences for them live. That
    is the arrangement `closed` already has, and the reason is that this
    function is also the door the Mojo API comes through.

    Args:
        kind: The reduction, as pandas spells the method.
        settings: The tuple, which is empty for the eight that read nothing.

    Returns:
        The settings, with a default in every field nothing was read into.

    Raises:
        Error: Tagged `value` if the tuple is the wrong length for the
            reduction, if the fraction is outside nought to one, or if a word is
            not one the reduction accepts. Tagged `dtype` if a value is of the
            wrong type.
    """
    var wanted = 0
    if kind == "var" or kind == "std" or kind == "sem":
        wanted = 1
    elif kind == "quantile":
        wanted = 2
    elif kind == "rank":
        wanted = 3
    var given = Int(len(settings))
    if given != wanted:
        raise tagged(
            VALUE,
            String(
                "window: ",
                kind,
                " reads ",
                wanted,
                " of its own parameters and ",
                given,
                " arrived",
            ),
        )
    var out = WindowSettings()
    if wanted == 0:
        return out
    if kind == "quantile":
        out.fraction = number(settings[0], "q")
        if not (out.fraction >= 0.0 and out.fraction <= 1.0):
            # Written as a pair of comparisons rather than as a range so that
            # a NaN fails it. pandas accepts a NaN here, because its own check
            # asks whether the fraction is below nought or above one and a NaN
            # is neither, and then answers a column of NaN. A caller who wrote
            # that meant a position in the window and did not get one.
            raise tagged(
                VALUE,
                String(
                    "window: q is a fraction from 0 to 1 and not ",
                    out.fraction,
                ),
            )
        try:
            out.between = between_named(words(settings[1], "interpolation"))
        except e:
            raise tagged(VALUE, String(e))
        return out
    if kind == "rank":
        try:
            out.tied = tied_named(words(settings[0], "method"))
        except e:
            raise tagged(VALUE, String(e))
        out.ascending = flag(settings[1], "ascending")
        out.pct = flag(settings[2], "pct")
        return out
    out.ddof = whole(settings[0], "ddof")
    return out


def window(
    column: Series,
    kind: String,
    width: Optional[Int],
    min_periods: Optional[Int],
    center: Bool,
    closed: String,
    step: Optional[Int],
    settings: WindowSettings,
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
        settings: The parameters the reduction reads and the window does not,
            already read apart from the tuple they crossed in.

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
                settings,
            )
        if center or step:
            # pandas has no place to put either of these on an expanding
            # window, so a caller who reached this asked for a window that does
            # not exist rather than for one this library has not written.
            raise Error(
                "window: an expanding window takes neither center nor step"
            )
        return column.expanding(
            op, min_periods.value() if min_periods else 1, settings
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
    settings: WindowSettings,
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
        settings: The parameters the reduction reads and the window does not,
            already read apart from the tuple they crossed in.

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
                settings,
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
