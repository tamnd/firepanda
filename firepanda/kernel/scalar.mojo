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

from std.math import sqrt

from firepanda.array.array import Array

from .accum import accumulator
from .group import AggKind


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


def argsort_scalar[
    dt: DType
](col: Array[dt], descending: Bool = False, nulls_first: Bool = False) -> List[
    Int
]:
    """Sorts row indices by insertion, comparing values with `<`.

    An insertion sort is stable by construction as long as it stops shifting on
    the first element that does not strictly belong after the one being placed,
    which is what the loop below does, so this twin checks the real kernel's
    stability and not only its ordering.

    It deliberately does not go through `sort_key`. Comparing the values means
    the transform is checked rather than assumed, which is the whole point of
    having a twin. The two places that costs something are NaN, which no
    comparison orders, and negative zero, which compares equal to positive zero
    while the kernel sorts it below. Both are pinned down by name in
    `tests/test_sort.mojo` instead, and the fuzz generator keeps them out of the
    float columns it builds.

    Args:
        col: The column.
        descending: Largest first.
        nulls_first: Put the nulls at the front rather than the back.

    Parameters:
        dt: The dtype.

    Returns:
        A permutation of `[0, len(col))`.
    """
    var live = List[Int]()
    var nulls = List[Int]()
    for i in range(len(col)):
        if col.is_valid(i):
            live.append(i)
        else:
            nulls.append(i)

    for i in range(1, len(live)):
        var at = live[i]
        var value = col[at]
        var j = i - 1
        while j >= 0:
            var other = col[live[j]]
            var shift = other < value if descending else value < other
            if not shift:
                break
            live[j + 1] = live[j]
            j -= 1
        live[j + 1] = at

    var out = List[Int]()
    if nulls_first:
        for i in range(len(nulls)):
            out.append(nulls[i])
    for i in range(len(live)):
        out.append(live[i])
    if not nulls_first:
        for i in range(len(nulls)):
            out.append(nulls[i])
    return out^


def group_scalar[
    dt: DType
](
    col: Array[dt],
    kind: AggKind,
    codes: Array[DType.uint32],
    groups: Int,
) raises -> Tuple[List[Float64], List[Bool]]:
    """Aggregates each group by collecting its rows into a list and reducing it.

    This is the twin's whole idea made literal. The kernel never materializes a
    group; it scatters into an accumulator and the groups only exist as indices.
    This one builds the actual list of values belonging to each group and then
    reduces that list with a loop anyone can read, which is slow enough to be
    useless and simple enough to be right.

    Everything comes back as `Float64` regardless of the input dtype, with a
    parallel list saying which entries are present. That loses precision above
    2^53 and the fuzz generator keeps the columns it builds inside that range,
    which is the same trade the twin makes everywhere else: it is checking that
    the grouping and the reduction are right, not that int64 arithmetic is.

    Args:
        col: The column being aggregated.
        kind: Which reduction.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Parameters:
        dt: The column's dtype.

    Returns:
        One value per group and one validity flag per group.

    Raises:
        If the reduction is not one of the eight.
    """
    var values = List[Float64](capacity=groups)
    var valid = List[Bool](capacity=groups)

    for g in range(groups):
        # Collect this group's rows the naive way: walk every row and keep the
        # ones that name this group. That is O(groups * rows) and it is meant to
        # be, because the alternative is a bucketing pass and a bucketing pass is
        # the thing being checked.
        var present = List[Float64]()
        var rows = 0
        for i in range(len(codes)):
            if Int(codes[i]) != g:
                continue
            rows += 1
            if kind.counts_rows():
                continue
            if col.is_valid(i):
                present.append(Float64(col[i]))

        if kind == AggKind.SIZE:
            values.append(Float64(rows))
            valid.append(True)
        elif kind == AggKind.COUNT:
            values.append(Float64(len(present)))
            valid.append(True)
        elif kind == AggKind.SUM:
            var total = Float64(0)
            for k in range(len(present)):
                total += present[k]
            values.append(total)
            valid.append(True)
        elif kind == AggKind.MEAN:
            if len(present) == 0:
                values.append(Float64(0))
                valid.append(False)
            else:
                var total = Float64(0)
                for k in range(len(present)):
                    total += present[k]
                values.append(total / Float64(len(present)))
                valid.append(True)
        elif kind == AggKind.MIN or kind == AggKind.MAX:
            if len(present) == 0:
                values.append(Float64(0))
                valid.append(False)
            else:
                var best = present[0]
                for k in range(1, len(present)):
                    if kind == AggKind.MIN:
                        if present[k] < best:
                            best = present[k]
                    elif present[k] > best:
                        best = present[k]
                values.append(best)
                valid.append(True)
        elif kind == AggKind.FIRST or kind == AggKind.LAST:
            if len(present) == 0:
                values.append(Float64(0))
                valid.append(False)
            else:
                var at = 0 if kind == AggKind.FIRST else len(present) - 1
                values.append(present[at])
                valid.append(True)
        elif kind == AggKind.VAR or kind == AggKind.STD:
            var divisor = len(present) - Int(kind.param)
            if divisor <= 0:
                values.append(Float64(0))
                valid.append(False)
            else:
                var total = Float64(0)
                for k in range(len(present)):
                    total += present[k]
                var centre = total / Float64(len(present))
                var squares = Float64(0)
                for k in range(len(present)):
                    squares += (present[k] - centre) * (present[k] - centre)
                var spread = squares / Float64(divisor)
                if kind == AggKind.STD:
                    spread = sqrt(spread)
                values.append(spread)
                valid.append(True)
        elif kind == AggKind.MEDIAN or kind == AggKind.QUANTILE:
            if len(present) == 0:
                values.append(Float64(0))
                valid.append(False)
            else:
                # An insertion sort, because the point of the twin is that the
                # reader can see it is a sort rather than take one on trust.
                var ordered = List[Float64]()
                for k in range(len(present)):
                    var at = len(ordered)
                    ordered.append(present[k])
                    while at > 0 and ordered[at - 1] > ordered[at]:
                        var swap = ordered[at - 1]
                        ordered[at - 1] = ordered[at]
                        ordered[at] = swap
                        at -= 1
                var position = kind.param * Float64(len(ordered) - 1)
                var lower = Int(position)
                var upper = lower + 1 if lower + 1 < len(ordered) else lower
                values.append(
                    ordered[lower]
                    + (ordered[upper] - ordered[lower])
                    * (position - Float64(lower))
                )
                valid.append(True)
        elif kind == AggKind.NUNIQUE:
            var seen = List[Float64]()
            for k in range(len(present)):
                var known = False
                for s in range(len(seen)):
                    if seen[s] == present[k]:
                        known = True
                        break
                if not known:
                    seen.append(present[k])
            values.append(Float64(len(seen)))
            valid.append(True)
        else:
            raise Error("group by: unsupported aggregation")

    return (values^, valid^)
