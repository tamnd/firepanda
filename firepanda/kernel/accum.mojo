"""Accumulator widths and range endpoints for the aggregation kernels.

Summing a column of int8 in int8 is arithmetically useless: a hundred and thirty
rows of ones overflows. numpy widens to the platform integer for exactly this
reason and pandas inherits the behaviour, so firepanda does the same and
`accumulator` is where that decision lives.

`accumulator` is a plain function rather than a `comptime` table because Mojo
folds a `def` whose argument is compile-time known and lets the result be used as
a parameter, which makes `Scalar[accumulator(dt)]` a legal return type. That is
worth knowing; it removes the need for a parallel set of aliases per dtype.
"""


def accumulator(dt: DType) -> DType:
    """Returns the dtype a sum over `dt` should accumulate in.

    Signed integers widen to int64, unsigned to uint64, floats to float64.
    Booleans count as unsigned, so `sum_of` over a bool column gives the number
    of true values, which is what pandas does.

    Args:
        dt: The column's dtype.

    Returns:
        The accumulator dtype.
    """
    if dt.is_floating_point():
        return DType.float64
    if dt.is_signed():
        return DType.int64
    return DType.uint64


def lowest[dt: DType]() -> Scalar[dt]:
    """Returns the identity for a max reduction over a dtype.

    For floats this is negative infinity rather than the most negative finite
    value, which is what `Scalar.MIN` gives and what a max over a column
    containing infinities needs.

    Parameters:
        dt: The dtype.

    Returns:
        The smallest value the dtype can hold.
    """
    return Scalar[dt].MIN


def highest[dt: DType]() -> Scalar[dt]:
    """Returns the identity for a min reduction over a dtype.

    Parameters:
        dt: The dtype.

    Returns:
        The largest value the dtype can hold.
    """
    return Scalar[dt].MAX
