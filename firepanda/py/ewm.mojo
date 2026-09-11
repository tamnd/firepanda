"""The door behind `ewm` and the four reductions on the object it answers.

The window kernel and this one are separate for the reason
`firepanda/kernel/ewm.mojo` gives at length, which is that an exponentially
weighted window is a weight and every other window is a pair of row numbers. The
crossing is separate for a smaller and more practical reason on top of that:
nothing an EWM needs to be told fits in the slots a rolling window has already
spent. There is no width, no centring, no closed rule and no step, and there are
four things in their place.

### Why the decay collapses to one number before it crosses

pandas takes the decay as exactly one of `com`, `span`, `halflife` and `alpha`,
and the four are four spellings of one number. A caller reaches this through
`ewm(span=5)` or `ewm(com=2)` and both mean a smoothing factor of a third, so
sending all four across and sorting them out on the far side would mean carrying
three absent values and a rule about which of them to believe, which is the
arrangement the width and the `center` flag were deliberately not given in
`window.mojo`.

So the collapse happens on the Python side, where pandas' own sentences for the
refusals live, and one number crosses. The factor is checked again at the
kernel's door, which is the same arrangement `closed` has and for the same
reason: the Mojo API is also a caller and it does not come through Python.

### Where the parameters cross

An EWM is the reduction kind, the smoothing factor, `min_periods`, `adjust` and
`ignore_na`, which is five of the seven slots a bound method gets after the
object. Document 13 section 4 measured that ceiling and the window crossing
spends six of the seven, so there was never a question of putting the two
reductions through one door.

The sixth slot is the same tuple the window crossing uses for a reduction's own
parameters, which here is `bias` for `var` and `std` and nothing at all for
`mean` and `sum`. That leaves a slot spare, which is worth keeping rather than
spending, because `corr` and `cov` are coming and they need somewhere to put the
other column.

### What is not here

`corr` and `cov`, `agg` and `aggregate`, `online`, a decay given as a real
duration through `times`, and `method="table"`. None of them resolves rather
than resolving and refusing, for the reason document 07 gives.
"""

from std.python import PythonObject

from firepanda.dtype.logical import LogicalType
from firepanda.frame.frame import DataFrame
from firepanda.frame.index import Index
from firepanda.frame.series import Series
from firepanda.kernel.ewm import EwmSpec, ewm_named
from firepanda.py.args import flag
from firepanda.py.errors import DTYPE, VALUE, tagged


def _reducible(column: Series) raises:
    """Refuses a column no exponentially weighted reduction has an answer for.

    The same check and the same sentence the rolling window gives, for the same
    reason, which is that pandas raises a type error here and a caller's except
    clause is looking for one.

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
            "ewm: there is nothing to aggregate in column '",
            column.name,
            "', which holds ",
            t,
        ),
    )


def ewm_spec(
    kind: String,
    alpha: Float64,
    min_periods: Int,
    adjust: Bool,
    ignore_na: Bool,
    settings: PythonObject,
) raises -> EwmSpec:
    """Gathers everything an exponentially weighted window was told.

    The tuple is positional and its length is decided by the reduction, so the
    length is checked before anything is read out of it, which is the
    arrangement `window_settings` has and is there for the same reason: a
    mismatch means the two halves of the library disagree about a reduction
    rather than that a caller made a mistake.

    Args:
        kind: The reduction, as pandas spells the method.
        alpha: The smoothing factor, already collapsed from whichever of the
            four spellings arrived.
        min_periods: How many values a row needs before it is answered.
        adjust: Whether every row weighs one rather than the smoothing factor.
        ignore_na: Whether a missing row is skipped rather than taking up a slot
            in the decay.
        settings: The tuple, which holds `bias` for the two spreads and nothing
            for the mean and the total.

    Returns:
        The spec.

    Raises:
        Error: Tagged `value` if the tuple is the wrong length for the
            reduction, or if the smoothing factor or the count is out of range.
            Tagged `dtype` if a value is of the wrong type.
    """
    var wanted = 1 if kind == "var" or kind == "std" else 0
    var given = Int(len(settings))
    if given != wanted:
        raise tagged(
            VALUE,
            String(
                "ewm: ",
                kind,
                " reads ",
                wanted,
                " of its own parameters and ",
                given,
                " arrived",
            ),
        )
    if not (alpha > 0.0 and alpha <= 1.0):
        raise tagged(VALUE, "alpha must satisfy: 0 < alpha <= 1")
    if min_periods < 0:
        raise tagged(VALUE, String("ewm: min_periods cannot be ", min_periods))
    var out = EwmSpec(alpha, min_periods, adjust, ignore_na, False)
    if wanted == 1:
        out.bias = flag(settings[0], "bias")
    return out


def ewm(
    column: Series,
    kind: String,
    alpha: Float64,
    min_periods: Int,
    adjust: Bool,
    ignore_na: Bool,
    settings: PythonObject,
) raises -> Series:
    """Runs one exponentially weighted reduction down a column.

    Args:
        column: The column to read.
        kind: The reduction, as pandas spells the method.
        alpha: The smoothing factor.
        min_periods: How many values a row needs before it is answered.
        adjust: Whether every row weighs one.
        ignore_na: Whether a missing row is skipped.
        settings: The parameters the reduction reads and the decay does not.

    Returns:
        A float64 column as tall as the one it read.

    Raises:
        Error: Tagged `dtype` if the column holds something no reduction can
            read, and tagged `value` if the reduction has no name, a parameter
            is out of range, or the combination has no pandas answer.
    """
    _reducible(column)
    var spec = ewm_spec(kind, alpha, min_periods, adjust, ignore_na, settings)
    try:
        return column.ewm(ewm_named(kind), spec)
    except e:
        raise tagged(VALUE, String(e))


def ewm_frame(
    frame: DataFrame,
    kind: String,
    alpha: Float64,
    min_periods: Int,
    adjust: Bool,
    ignore_na: Bool,
    settings: PythonObject,
) raises -> DataFrame:
    """Runs one exponentially weighted reduction down every column.

    The columns are done one at a time and put back together, which is what
    pandas does with `method="single"` and is the only thing this library
    answers. The other reading, `method="table"`, decays across the columns of a
    row rather than down them, and it is a different computation and not a
    faster arrangement of this one.

    Every column is checked before any of them is read, so a frame with a text
    column in the middle of it raises rather than computing half an answer.

    Args:
        frame: The frame to read.
        kind: The reduction, as pandas spells the method.
        alpha: The smoothing factor.
        min_periods: How many values a row needs before it is answered.
        adjust: Whether every row weighs one.
        ignore_na: Whether a missing row is skipped.
        settings: The parameters the reduction reads and the decay does not.

    Returns:
        A frame of the same column names in the same order, every one of them
        float64 and as tall as the frame it read.

    Raises:
        Error: Tagged `dtype` if any column holds something no reduction can
            read, and tagged `value` if the reduction has no name, a parameter
            is out of range, or the combination has no pandas answer.
    """
    for i in range(frame.width()):
        _reducible(frame.column(frame.schema[i].name))
    if frame.width() == 0:
        return DataFrame(copy=frame)
    var parts = List[Series](capacity=frame.width())
    for i in range(frame.width()):
        parts.append(
            ewm(
                frame.column(frame.schema[i].name),
                kind,
                alpha,
                min_periods,
                adjust,
                ignore_na,
                settings,
            )
        )
    var labels = Index(copy=parts[0].index)
    var out = DataFrame.from_series(parts^)
    out.index = labels^
    return out^
