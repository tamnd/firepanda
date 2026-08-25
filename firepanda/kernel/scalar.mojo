"""The slow, obviously correct version of every kernel.

Nothing in this file is called in production. Every function here is a loop over
one element at a time with an explicit validity check, written the way you would
write it if you had never heard of SIMD and did not know that a null holds a zero.
It exists so that the fast kernels have something to be checked against that is
too simple to be wrong.

The rule when a twin and its kernel disagree is that the kernel is wrong. If it
ever turns out that the twin was the wrong one, that is a bug in this file and it
gets fixed here first, before the kernel is touched. Otherwise the two drift into
agreeing with each other about something neither of them should be doing.

Read the twins to find out what a kernel is supposed to do. They are the
specification in the only form that runs.
"""

from firepanda.array.array import Array

from .accum import accumulator


def sum_scalar[dt: DType](col: Array[dt]) -> Scalar[accumulator(dt)]:
    """Adds up the non-null values, one at a time, checking validity for each.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        The total.
    """
    comptime acc = accumulator(dt)
    var total = Scalar[acc](0)
    for i in range(len(col)):
        if col.is_valid(i):
            total += Scalar[acc](col[i])
    return total


def count_scalar[dt: DType](col: Array[dt]) -> Int:
    """Counts the non-null values, one at a time.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        The count.
    """
    var total = 0
    for i in range(len(col)):
        if col.is_valid(i):
            total += 1
    return total


def min_scalar[dt: DType](col: Array[dt]) -> Tuple[Scalar[dt], Bool]:
    """Finds the smallest non-null value, one at a time.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        The minimum and whether there was one.
    """
    var best = Scalar[dt](0)
    var seen = False
    for i in range(len(col)):
        if not col.is_valid(i):
            continue
        var value = col[i]
        if not seen or value < best:
            best = value
            seen = True
    return (best, seen)


def max_scalar[dt: DType](col: Array[dt]) -> Tuple[Scalar[dt], Bool]:
    """Finds the largest non-null value, one at a time.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        The maximum and whether there was one.
    """
    var best = Scalar[dt](0)
    var seen = False
    for i in range(len(col)):
        if not col.is_valid(i):
            continue
        var value = col[i]
        if not seen or value > best:
            best = value
            seen = True
    return (best, seen)


def mean_scalar[dt: DType](col: Array[dt]) -> Tuple[Float64, Bool]:
    """Averages the non-null values, one at a time.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        The mean and whether there was one.
    """
    var total = Float64(0)
    var present = 0
    for i in range(len(col)):
        if col.is_valid(i):
            total += Float64(col[i])
            present += 1
    if present == 0:
        return (Float64(0), False)
    return (total / Float64(present), True)


def add_scalar[dt: DType](a: Array[dt], b: Array[dt]) -> Array[dt]:
    """Adds two columns, one element at a time.

    The three arithmetic twins below are the same eight lines with one operator
    changed. Factoring them through a function parameter would be tidier and
    would also mean the thing the fast kernels are checked against has a piece of
    generic machinery in it. They stay written out.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of sums, null wherever either input is null.
    """
    var out = Array[dt](len(a))
    for i in range(len(a)):
        if a.is_valid(i) and b.is_valid(i):
            out.set_valid(i, a[i] + b[i])
        else:
            out.set_null(i)
    return out^


def subtract_scalar[dt: DType](a: Array[dt], b: Array[dt]) -> Array[dt]:
    """Subtracts two columns, one element at a time.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of differences, null wherever either input is null.
    """
    var out = Array[dt](len(a))
    for i in range(len(a)):
        if a.is_valid(i) and b.is_valid(i):
            out.set_valid(i, a[i] - b[i])
        else:
            out.set_null(i)
    return out^


def multiply_scalar[dt: DType](a: Array[dt], b: Array[dt]) -> Array[dt]:
    """Multiplies two columns, one element at a time.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of products, null wherever either input is null.
    """
    var out = Array[dt](len(a))
    for i in range(len(a)):
        if a.is_valid(i) and b.is_valid(i):
            out.set_valid(i, a[i] * b[i])
        else:
            out.set_null(i)
    return out^


def divide_scalar[
    dt: DType
](a: Array[dt], b: Array[dt]) -> Array[DType.float64]:
    """Divides two columns, one element at a time.

    Division always produces float64, whatever went in, which is what pandas does
    with the true division operator. Division by zero follows IEEE and gives an
    infinity or a NaN rather than a null, again matching pandas.

    Args:
        a: The numerator column.
        b: The denominator column. Must be the same length as `a`.

    Parameters:
        dt: The input dtype.

    Returns:
        A float64 column, null wherever either input is null.
    """
    var out = Array[DType.float64](len(a))
    for i in range(len(a)):
        if a.is_valid(i) and b.is_valid(i):
            out.set_valid(i, Float64(a[i]) / Float64(b[i]))
        else:
            out.set_null(i)
    return out^


def equal_scalar[dt: DType](a: Array[dt], b: Array[dt]) -> Array[DType.bool]:
    """Compares two columns for equality, one element at a time.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A bool column, null wherever either input is null.
    """
    var out = Array[DType.bool](len(a))
    for i in range(len(a)):
        if a.is_valid(i) and b.is_valid(i):
            out.set_valid(i, a[i] == b[i])
        else:
            out.set_null(i)
    return out^


def less_scalar[dt: DType](a: Array[dt], b: Array[dt]) -> Array[DType.bool]:
    """Compares two columns with less-than, one element at a time.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A bool column, null wherever either input is null.
    """
    var out = Array[DType.bool](len(a))
    for i in range(len(a)):
        if a.is_valid(i) and b.is_valid(i):
            out.set_valid(i, a[i] < b[i])
        else:
            out.set_null(i)
    return out^


def cast_scalar[src: DType, dst: DType](col: Array[src]) -> Array[dst]:
    """Converts a column to another dtype, one element at a time.

    Args:
        col: The column.

    Parameters:
        src: The input dtype.
        dst: The output dtype.

    Returns:
        A column of the target dtype, null where the input was null.
    """
    var out = Array[dst](len(col))
    for i in range(len(col)):
        if col.is_valid(i):
            out.set_valid(i, col[i].cast[dst]())
        else:
            out.set_null(i)
    return out^


def take_scalar[dt: DType](col: Array[dt], indices: List[Int]) -> Array[dt]:
    """Gathers rows by position, one at a time.

    A negative index means null, which is how a left join says that the row on
    the right did not exist.

    Args:
        col: The column to gather from.
        indices: The positions to gather. Each must be less than `len(col)`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of length `len(indices)`.
    """
    var out = Array[dt](len(indices))
    for i in range(len(indices)):
        var at = indices[i]
        if at < 0 or not col.is_valid(at):
            out.set_null(i)
        else:
            out.set_valid(i, col[at])
    return out^


def filter_scalar[
    dt: DType
](col: Array[dt], mask: Array[DType.bool]) -> Array[dt]:
    """Keeps the rows where the mask is true, one at a time.

    A null in the mask drops the row. That is what pandas does with a nullable
    boolean mask and it is the only choice that keeps `filter(m)` and
    `filter(not m)` from both containing the same row.

    Args:
        col: The column to filter.
        mask: The mask. Must be the same length as `col`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of the kept rows.
    """
    var kept = 0
    for i in range(len(mask)):
        if mask.is_valid(i) and Bool(mask[i]):
            kept += 1

    var out = Array[dt](kept)
    var at = 0
    for i in range(len(mask)):
        if not (mask.is_valid(i) and Bool(mask[i])):
            continue
        if col.is_valid(i):
            out.set_valid(at, col[i])
        else:
            out.set_null(at)
        at += 1
    return out^
