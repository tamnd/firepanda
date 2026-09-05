"""Reducing a whole column to one value, with no grouping in the way.

Tier: unstable, documented. docs/specs/11-package-layout.md.

`group.mojo` can already answer this question. Hand it a code per row that is
always zero and a group count of one and it reduces the whole column, which is
exactly what a frame wanting a total wants. It is also the slowest possible way
to ask. Ten million rows means allocating and zeroing forty megabytes of codes,
then walking them beside the values and scattering into a table with one entry
in it. The scatter is a dependent store per row and none of it vectorizes.

`agg.mojo` has had the fast answer the whole time. A sum with nothing to group by
is a vectorized add over the values buffer and a minimum is a walk of the
validity a word at a time. On ten million float64 rows on gamingpc the group
route takes 85 ms and this one takes 5.

So this file is a dispatch and not an implementation. The reductions that
`agg.mojo` has get the fast route. The rest, meaning the variance, the order
statistics and the distinct count, go to `aggregate_group_any` with a single
group, because they have no whole column spelling yet and a correct slow answer
is better than a missing one. The dividing line is written down in
`_takes_fast_route` rather than being spread through the branches, so that adding
a whole column variance later is a change in two places and not a hunt.

Every result is a column of one row, not a scalar. That is what the frame layer
wants back, because `DataFrame.agg` builds a one row frame out of these and a
scalar would have to be widened into a column anyway. It also means the dtype
rules are the same rules the group by uses, which they have to be: a sum over an
int32 column widens to int64 whether or not anybody grouped it.
"""

from std.math import nan

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL

from .accum import accumulator
from .agg import extreme_over, mean_over, sum_over
from .group import AggKind, aggregate_group_any
from .nulls import missing_count_any


def _takes_fast_route(kind: AggKind) -> Bool:
    """Reports whether this reduction has a whole column implementation here.

    Args:
        kind: Which reduction.

    Returns:
        True for the five that `agg.mojo` covers plus the two that are counts.
    """
    return (
        kind == AggKind.SUM
        or kind == AggKind.MEAN
        or kind == AggKind.MIN
        or kind == AggKind.MAX
        or kind == AggKind.COUNT
        or kind == AggKind.SIZE
    )


def reduce_any(col: AnyArray, kind: AggKind) raises -> AnyArray:
    """Reduces a column to a single row.

    Args:
        col: The column.
        kind: Which reduction.

    Returns:
        A column of exactly one row, in the dtype the grouped reduction would
        have produced for the same kind.

    Raises:
        If the dtype has no physical layout, or if the reduction is one no path
        here implements for this column.
    """
    if kind == AggKind.SIZE:
        var sized = Array[DType.int64](1)
        sized[0] = Int64(len(col))
        return AnyArray(sized^)

    if kind == AggKind.COUNT:
        # The one reduction that works on a column of strings on the same line as
        # a column of numbers, since counting what is there needs no order and no
        # arithmetic. It used to need no values either and was a subtraction of
        # two numbers the column already knew. On a float column it is a scan
        # now, because a NaN is missing and the validity bitmap does not know
        # that. See #170.
        var counted = Array[DType.int64](1)
        counted[0] = Int64(len(col) - missing_count_any(col))
        return AnyArray(counted^)

    # As in `aggregate_group_any`: uint8 is in ALL, so a string column would
    # match it and a sum over a column of names would return a number taken from
    # the first byte of every view rather than an error. The grouped path already
    # knows what to do with strings, so send them there.
    if not col.is_string() and _takes_fast_route(kind):
        comptime for candidate in ALL:
            if col.dtype() == candidate:
                return _reduce_core(
                    col.unsafe_ptr[candidate](),
                    col.data.validity,
                    len(col) - col.null_count(),
                    len(col),
                    kind,
                )

    var codes = Array[DType.uint32](len(col))
    return aggregate_group_any(col, kind, codes^, 1, trusted=True)


def _reduce_core[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    present: Int,
    n: Int,
    kind: AggKind,
) raises -> AnyArray:
    """Runs one whole column reduction. One instantiation per dtype.

    Args:
        source: The values.
        validity: Which of them are present.
        present: How many of them are present.
        n: How many there are.
        kind: Which reduction.

    Parameters:
        dt: The value dtype.
        origin: Where the values live.

    Returns:
        A column of one row.

    Raises:
        If the kind is not one this function was told it handles, which would
        mean `_takes_fast_route` and this disagree.
    """
    if kind == AggKind.SUM:
        comptime acc = accumulator(dt)
        var summed = Array[acc](1)
        # A sum is always valid, including over an empty column and over a column
        # that is entirely null, where it is zero. `agg.mojo` explains why that
        # is pandas' answer and why disagreeing with it would be worse.
        summed[0] = sum_over(source, n).value
        return AnyArray(summed^)

    if kind == AggKind.MEAN:
        var averaged = Array[DType.float64](1)
        var mean = mean_over(source, present, n)
        # A mean always answers in float64, so this one is always the NaN branch
        # of `_place`. The two extremes below keep the column's own dtype and get
        # whichever branch that dtype has.
        _place(averaged, Float64(mean.value), mean.valid)
        return AnyArray(averaged^)

    if kind == AggKind.MIN:
        var smallest = Array[dt](1)
        var low = extreme_over[want_min=True](source, validity, n)
        _place(smallest, low.value, low.valid)
        return AnyArray(smallest^)

    if kind == AggKind.MAX:
        var largest = Array[dt](1)
        var high = extreme_over[want_min=False](source, validity, n)
        _place(largest, high.value, high.valid)
        return AnyArray(largest^)

    raise Error("reduce: unsupported aggregation")


def _place[dt: DType](mut out: Array[dt], value: Scalar[dt], valid: Bool):
    """Writes the one row of a result, or says it found nothing.

    How it says so depends on the dtype, because pandas has two spellings and
    only one of them is available in each. A float column takes a NaN and stays
    valid, since that is the only missing a pandas float column has. Every other
    dtype takes a zero behind a cleared validity bit, since there is no NaN to
    write. See #170.

    A missing row of the second kind still holds a zero rather than whatever the
    reduction was carrying when it gave up, because every null in the package
    holds a zero and a minimum that left negative infinity behind the validity
    bit would be the one exception.

    Args:
        out: The one row column.
        value: The reduced value.
        valid: Whether the reduction found anything.
    """
    if valid:
        out[0] = value
        return
    comptime if dt.is_floating_point():
        out[0] = nan[dt]()
        return
    out[0] = Scalar[dt](0)
    out.data.validity.set(0, False)
