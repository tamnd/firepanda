"""An exponentially weighted window is a weight, and it has no edges at all.

Every other window in this library is a pair of row numbers, and document 31
argues that at length. This one is not. Every row of the column is inside every
window and what changes from row to row is how much each earlier row counts,
which falls off geometrically with distance. So none of `Shape`, `Edges`,
`rolling_shape` or `expanding_shape` means anything here, and neither does the
loop they feed, which is why this is a second kernel beside `window.mojo`
rather than four more reductions inside it.

## The four spellings of the decay are one number

pandas takes the decay as exactly one of `com`, `span`, `halflife` and `alpha`
and turns whichever arrived into a smoothing factor between nought and one. The
conversions are `alpha = 1 / (1 + com)`, `alpha = 2 / (span + 1)` and
`alpha = 1 - exp(-ln 2 / halflife)`, and measured against pandas 3.0.5 all
three routes agree with the closed form to the last bit. A span of five and a
centre of mass of two are the same window, which is a fact a caller relies on
without checking, so the conversion happens once in `alpha_of` and everything
below this line sees one number.

## Why the adjusted form is not a quadratic sum

pandas documents `adjust=True` as a finite weighted sum, where row `t` is the
sum of `(1 - alpha)^i` times the value `i` rows back over the sum of those
weights, and `adjust=False` as the plain recursion, where row `t` is
`(1 - alpha)` times the previous answer plus `alpha` times this row. Written
that way they look like two algorithms and the first one looks like it costs the
height of the column on every row.

They are one algorithm. Carry the answer so far and the weight standing behind
it. Each row multiplies that weight by `(1 - alpha)`, folds the new row in as a
weighted average of the carried answer against the new value, and then adds the
new row's weight to the carried weight. The only difference between the two
modes is the new row's weight and what becomes of the carried weight after the
fold: with `adjust=True` the new weight is one and the carried weight keeps
growing, and with `adjust=False` the new weight is `alpha` and the carried
weight is reset to one every time, which makes the weighted average collapse
into exactly the plain recursion. So the cost is the height of the column in
both modes and there is one loop.

## Where the sum differs from the mean, and what pandas does about it

`sum` is the same recurrence with the division left out, so the answer is
`(1 - alpha)` times the previous answer plus this row rather than a weighted
average of the two. That is only meaningful when every row's weight is one,
because with a new weight of `alpha` the total is a weighted average scaled by a
factor nobody asked for, and pandas agrees: `ExponentialMovingWindow.sum`
raises `NotImplementedError` when `adjust` is off, with the message `sum is not
implemented with adjust=False`. There is a real pandas signature there behind
which there is no pandas answer, and this kernel refuses the same combination
rather than inventing one.

## A missing row is a decision and not a gap

`ignore_na` decides whether a missing row takes up a slot in the decay. With the
default of `False` it does: the carried weight is multiplied by `(1 - alpha)`
for the missing row even though there is nothing to fold in, so a value two rows
after a null counts as two rows away. With `True` the missing row is skipped
entirely and that value counts as one row away. Those are two different answers
and not a rounding of each other. On the column `1, null, 3, null, null, 7, 9`
with an alpha of three tenths the last row is 7.16217041 one way and 6.20331623
the other, and both were measured against pandas 3.0.5 before either was
written.

`min_periods` counts the values seen so far and not the rows, and a row before
the count is met is missing however much weight has piled up behind it.

## The one guard that is copied rather than derived

The fold has a test in pandas reading `if weighted != cur`, commented as
avoiding numerical errors on a constant series. Skipping the fold when the new
value already equals the carried answer is not an optimisation, it is what makes
a column of one repeated value come back as exactly that value rather than
drifting in the last bits, and leaving it out is visible on ordinary data. It is
kept here for that reason and not because pandas has it.

One more thing about the fold is worth knowing before anyone compares the last
bit of an answer against pandas and finds it off by one. The pandas wheel this
was measured against was compiled with the multiply and the add of the fold
contracted into a single fused instruction, which rounds once where two
operations round twice. On nought to nine with a span of five that moves two of
the ten rows by one unit in the last place. It is a fact about a compiler and not
about pandas, so this file does not chase it, and the gap is four orders of
magnitude inside the tightest tolerance the conformance board applies.

## An infinity never leaves this window

pandas replaces every infinity in a column with a missing value before any
window kernel sees it, which document 31 section 3 records and which applies
here too, since `ExponentialMovingWindow` is a `BaseWindow`. Here an infinity is
a value. The difference from the rolling side is that a rolling window
eventually drops the row that carried it and recovers, and this one never drops
a row, so an infinity is carried to the bottom of the column.

One of each sign is the case worth naming. The fold reaches `inf` plus `-inf`,
the carried value becomes a NaN, and the test at the top of the loop for whether
the recurrence has started is `weighted == weighted`, which is pandas' own
sentinel and is copied. So a cancelled infinity reads as nothing having arrived
yet and the next value starts the recurrence again. The conflation is only
reachable through an infinity, because a NaN in the data is turned into a
missing row before it reaches the fold, and keeping the sentinel is what makes
the two engines agree again from the row after the cancellation.

## The variance carries two weight totals and a correction

`var` and `std` take a `bias` flag that none of the ten window spreads has, and
default it to off. The state is the mean's state with a weighted second moment
beside it and two weight totals, the sum of the weights and the sum of their
squares. The biased answer is the second moment itself. The unbiased one scales
it by the sum squared over the sum squared minus the sum of squares, which is
the weighted analogue of dividing by `n - 1` instead of `n`, and answers a
missing value rather than a zero when that denominator is not positive, which is
every row where only one value has been seen.

There is no error bound and no rebuild here, which is the one place this file is
simpler than `window.mojo` rather than merely different. A rolling variance has
to un-see a row, and document 31 section 6 is entirely about why that cannot be
done safely. An exponentially weighted window never un-sees anything: a row's
weight shrinks towards nothing and is never subtracted, so the recurrence only
ever moves forwards and there is nothing for a bound to guard against.

## What is not here

`corr` and `cov`, which need a second column and are the same piece of work as
`Rolling.corr` and `Rolling.cov`. `agg` and `aggregate`, which belong with the
rolling ones. `online`, which is a streaming object pandas implements only in
numba. A decay given as a real duration through `times`, which needs a calendar
first. And `method="table"`, which decays across the columns of a frame rather
than down them.
"""

from std.math import exp, isnan, log, nan, sqrt

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.logical import LogicalType

from .cast import cast_any
from .nulls import present_bitmap_any

comptime EWM_MEAN = 0
"""Operation code for the exponentially weighted mean."""

comptime EWM_SUM = 1
"""Operation code for the exponentially weighted total."""

comptime EWM_VAR = 2
"""Operation code for the exponentially weighted variance."""

comptime EWM_STD = 3
"""Operation code for the root of the exponentially weighted variance."""


@fieldwise_init
struct EwmOp(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which of the four reductions an exponentially weighted window is running.

    Held as a code for the same reason `WindowOp` is, which is that the erased
    entry point takes it as an ordinary argument while the loop takes it as a
    decision made once before any row is read.
    """

    var code: Int
    """The operation, as one of the four values below."""

    comptime MEAN = Self(EWM_MEAN)
    """The weighted mean of everything seen so far."""

    comptime SUM = Self(EWM_SUM)
    """The weighted total of everything seen so far."""

    comptime VAR = Self(EWM_VAR)
    """The weighted variance of everything seen so far."""

    comptime STD = Self(EWM_STD)
    """The root of the weighted variance."""

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

    def spreads(self) -> Bool:
        """Whether this one carries a second moment as well as a mean.

        Returns:
            True for the variance and its root.
        """
        return self.code == EWM_VAR or self.code == EWM_STD

    def write_to(self, mut writer: Some[Writer]):
        """Writes the operation as the pandas method it answers.

        Args:
            writer: Where to write.
        """
        if self == Self.MEAN:
            writer.write("mean")
        elif self == Self.SUM:
            writer.write("sum")
        elif self == Self.VAR:
            writer.write("var")
        else:
            writer.write("std")


def ewm_named(name: StringSlice) raises -> EwmOp:
    """Turns a pandas method name into one of the four reductions.

    Args:
        name: The method name.

    Returns:
        The reduction it names.

    Raises:
        Error: If it is not one of the four this file answers.
    """
    if name == "mean":
        return EwmOp.MEAN
    if name == "sum":
        return EwmOp.SUM
    if name == "var":
        return EwmOp.VAR
    if name == "std":
        return EwmOp.STD
    raise Error(
        "ewm: no exponentially weighted reduction is called " + String(name)
    )


def alpha_of(
    com: Optional[Float64],
    span: Optional[Float64],
    halflife: Optional[Float64],
    alpha: Optional[Float64],
) raises -> Float64:
    """Turns whichever spelling of the decay arrived into a smoothing factor.

    Exactly one of the four has to be given, which is pandas' rule and is the
    right one: they are four ways of writing one number and a caller who gives
    two of them has said something contradictory rather than something
    redundant.

    Args:
        com: The centre of mass, or nothing.
        span: The span, or nothing.
        halflife: The half life in rows, or nothing.
        alpha: The smoothing factor itself, or nothing.

    Returns:
        The smoothing factor, above nought and at most one.

    Raises:
        Error: If none of the four arrived, if more than one did, or if the one
            that did is outside the range pandas allows for it.
    """
    var given = 0
    if com:
        given += 1
    if span:
        given += 1
    if halflife:
        given += 1
    if alpha:
        given += 1
    if given == 0:
        raise Error("Must pass one of comass, span, halflife, or alpha")
    if given > 1:
        raise Error("comass, span, halflife, and alpha are mutually exclusive")
    if com:
        var value = com.value()
        if not (value >= 0.0):
            raise Error("comass must satisfy: comass >= 0")
        return 1.0 / (1.0 + value)
    if span:
        var value = span.value()
        if not (value >= 1.0):
            raise Error("span must satisfy: span >= 1")
        return 2.0 / (value + 1.0)
    if halflife:
        var value = halflife.value()
        if not (value > 0.0):
            raise Error("halflife must satisfy: halflife > 0")
        return 1.0 - exp(-log(2.0) / value)
    var value = alpha.value()
    if not (value > 0.0 and value <= 1.0):
        raise Error("alpha must satisfy: 0 < alpha <= 1")
    return value


@fieldwise_init
struct EwmSpec(ImplicitlyCopyable, Movable):
    """Everything an exponentially weighted window needs to know.

    The decay has already collapsed to one number by the time a spec exists,
    because `alpha_of` is the only way to build one honestly and it answers a
    single factor. What is left is the two flags that change the recurrence, the
    count of values a row needs before it is answered, and the one flag the
    reduction itself reads.
    """

    var alpha: Float64
    """The smoothing factor, above nought and at most one."""

    var min_periods: Int
    """How many values a row needs behind it before it is answered."""

    var adjust: Bool
    """Whether every row's weight is one rather than the smoothing factor."""

    var ignore_na: Bool
    """Whether a missing row is skipped rather than taking a slot in the decay.
    """

    var bias: Bool
    """Whether the variance is the second moment itself, uncorrected."""

    def __init__(out self, alpha: Float64):
        """Builds a spec with pandas' defaults for everything but the decay.

        Args:
            alpha: The smoothing factor.
        """
        self = Self(alpha, 0, True, False, False)


def ewm_agg(col: AnyArray, op: EwmOp, spec: EwmSpec) raises -> AnyArray:
    """Runs one exponentially weighted reduction down a column.

    Args:
        col: The column.
        op: Which reduction to run.
        spec: The decay, the two flags that change the recurrence, the count a
            row needs and the bias flag.

    Returns:
        A float64 column of the same height, holding a NaN wherever fewer
        values had been seen than the row needed.

    Raises:
        Error: If the column is not something a window can reduce, if the
            smoothing factor is out of range, if a total was asked for with
            `adjust` off, or if the cast to float64 fails.
    """
    if (
        col.is_string()
        or col.type.is_temporal()
        or not (col.type.is_numeric() or col.type == LogicalType.BOOL)
    ):
        raise Error(
            "ewm: " + String(op) + " is not defined on " + String(col.type)
        )
    if not (spec.alpha > 0.0 and spec.alpha <= 1.0):
        raise Error("alpha must satisfy: 0 < alpha <= 1")
    if op == EwmOp.SUM and not spec.adjust:
        raise Error("sum is not implemented with adjust=False")
    var rows = len(col)
    var present = present_bitmap_any(col)
    var wide = AnyArray(copy=col) if col.dtype() == DType.float64 else cast_any(
        col, DType.float64
    )
    var src = wide.unsafe_ptr[DType.float64]()
    if op.spreads():
        return AnyArray(_spread(src, present, rows, spec, op == EwmOp.STD))
    return AnyArray(_weighted(src, present, rows, spec, op == EwmOp.SUM))


def _weighted[
    origin: ImmOrigin
](
    src: Pointer[Scalar[DType.float64], origin],
    present: Bitmap,
    rows: Int,
    spec: EwmSpec,
    total: Bool,
) raises -> Array[DType.float64]:
    """Carries the weighted mean or the weighted total down the column.

    The two share one loop because they share the whole recurrence except the
    division, and separating them would mean writing the decay, the missing row
    rule and the count of values twice for one line of difference.

    Args:
        src: The values, already float64.
        present: Which rows hold a value.
        rows: How tall the column is.
        spec: The decay, the flags and the count a row needs.
        total: Whether to leave the division out and answer the total.

    Parameters:
        origin: The origin of the values.

    Returns:
        A float64 column of the same height.

    Raises:
        Error: Only what allocation raises.
    """
    var answer = Array[DType.float64](rows)
    var target = answer.unsafe_mut_ptr()
    var falls = 1.0 - spec.alpha
    var fresh = 1.0 if spec.adjust else spec.alpha
    var carried = nan[DType.float64]()
    var behind = 1.0
    var found = 0
    var wanted = spec.min_periods if spec.min_periods > 1 else 1
    for i in range(rows):
        var here = present.get(i)
        var value = src.unsafe_offset(i).unsafe_load() if here else 0.0
        if here and isnan(value):
            # A NaN sitting in the values rather than a gap in the bitmap is
            # still a row with nothing in it, which is the reading pandas takes
            # because a NaN is the only way it can spell one.
            here = False
        if here:
            found += 1
        if not isnan(carried):
            if here or not spec.ignore_na:
                behind *= falls
                if total:
                    carried *= falls
            if here:
                if total:
                    carried += value
                elif carried != value:
                    carried = (behind * carried + fresh * value) / (
                        behind + fresh
                    )
                behind = behind + fresh if spec.adjust else 1.0
        elif here:
            carried = value
            behind = 1.0
        target.unsafe_offset(i).unsafe_store(
            carried if found >= wanted else nan[DType.float64]()
        )
    return answer^


def _spread[
    origin: ImmOrigin
](
    src: Pointer[Scalar[DType.float64], origin],
    present: Bitmap,
    rows: Int,
    spec: EwmSpec,
    rooted: Bool,
) raises -> Array[DType.float64]:
    """Carries the weighted variance down the column beside its mean.

    Four quantities move together and none of them can be dropped. The mean is
    needed because a second moment is measured around it and it moves under the
    moment as the moment is being built, which is why the fold carries the shift
    of the mean into the moment rather than recentring afterwards. The sum of
    the weights and the sum of their squares are needed because the correction
    for the bias is a ratio of those two and not a count of rows.

    Args:
        src: The values, already float64.
        present: Which rows hold a value.
        rows: How tall the column is.
        spec: The decay, the flags, the count a row needs and the bias flag.
        rooted: Whether to answer the root of the variance.

    Parameters:
        origin: The origin of the values.

    Returns:
        A float64 column of the same height.

    Raises:
        Error: Only what allocation raises.
    """
    var answer = Array[DType.float64](rows)
    var target = answer.unsafe_mut_ptr()
    var falls = 1.0 - spec.alpha
    var fresh = 1.0 if spec.adjust else spec.alpha
    var mean = nan[DType.float64]()
    var moment = nan[DType.float64]()
    var weights = 1.0
    var squares = 1.0
    var behind = 1.0
    var found = 0
    var wanted = spec.min_periods if spec.min_periods > 1 else 1
    for i in range(rows):
        var here = present.get(i)
        var value = src.unsafe_offset(i).unsafe_load() if here else 0.0
        if here and isnan(value):
            here = False
        if here:
            found += 1
        if not isnan(mean):
            if here or not spec.ignore_na:
                weights *= falls
                squares *= falls * falls
                behind *= falls
            if here:
                var was = mean
                if mean != value:
                    mean = (behind * was + fresh * value) / (behind + fresh)
                moment = (
                    behind * (moment + (was - mean) * (was - mean))
                    + fresh * (value - mean) * (value - mean)
                ) / (behind + fresh)
                weights += fresh
                squares += fresh * fresh
                behind += fresh
                if not spec.adjust:
                    weights /= behind
                    squares /= behind * behind
                    behind = 1.0
        elif here:
            mean = value
            moment = 0.0
            weights = 1.0
            squares = 1.0
            behind = 1.0
        var value_out = nan[DType.float64]()
        if found >= wanted and not isnan(moment):
            if spec.bias:
                value_out = moment
            else:
                var whole = weights * weights
                var divisor = whole - squares
                if divisor > 0.0:
                    value_out = (whole / divisor) * moment
        if rooted and not isnan(value_out):
            value_out = sqrt(value_out)
        target.unsafe_offset(i).unsafe_store(value_out)
    return answer^
