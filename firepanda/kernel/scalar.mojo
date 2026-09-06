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

from std.ffi import external_call
from std.math import isnan, nan, sqrt

from firepanda.array.array import Array
from firepanda.array.strings import StringArray

from .accum import accumulator
from .arith import OP_ADD, OP_MUL, OP_SUB
from .compare import CMP_EQ, CMP_GE, CMP_GT, CMP_LE, CMP_LT, CMP_NE
from .group import AggKind


def _is_there[dt: DType](col: Array[dt], i: Int) -> Bool:
    """Says whether row `i` holds a value the reductions should read.

    Its validity bit has to be set, and on a float dtype it must not be a NaN,
    because pandas steps over a NaN exactly as it steps over a value that was
    never there. The fast kernels do this with a compare and a select per vector
    and this does it one row at a time, which is the point of the twin. See #170.

    Args:
        col: The column.
        i: The row.

    Parameters:
        dt: The column's dtype.

    Returns:
        True if the row holds a value.
    """
    if not col.is_valid(i):
        return False
    comptime if dt.is_floating_point():
        if isnan(col[i]):
            return False
    return True


def sum_scalar[dt: DType](col: Array[dt]) -> Scalar[accumulator(dt)]:
    """Adds up the values that are there, one at a time.

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
        if _is_there(col, i):
            total += Scalar[acc](col[i])
    return total


def count_scalar[dt: DType](col: Array[dt]) -> Int:
    """Counts the set validity bits, one at a time.

    This one does not use `_is_there`, and the difference is deliberate. It is
    the twin of `count_of`, which is the Arrow count and says what is in the
    buffers. The pandas count, which a NaN is missing from, is
    `nulls.missing_count_any` and has its own twin in the tests.

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
    """Finds the smallest value that is there, one at a time.

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
        if not _is_there(col, i):
            continue
        var value = col[i]
        if not seen or value < best:
            best = value
            seen = True
    return (best, seen)


def max_scalar[dt: DType](col: Array[dt]) -> Tuple[Scalar[dt], Bool]:
    """Finds the largest value that is there, one at a time.

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
        if not _is_there(col, i):
            continue
        var value = col[i]
        if not seen or value > best:
            best = value
            seen = True
    return (best, seen)


def mean_scalar[dt: DType](col: Array[dt]) -> Tuple[Float64, Bool]:
    """Averages the values that are there, one at a time.

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
        if _is_there(col, i):
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


def floor_divide_scalar[dt: DType](a: Array[dt], b: Array[dt]) -> Array[dt]:
    """Floor divides two columns, one element at a time.

    The result keeps the operand type, which is what pandas does with `//` and
    is not what it does with `/`. On an integer dtype a zero divisor has no
    answer in the type, so the row is null; on a float dtype it is an infinity
    or a NaN and the row is present. That rule is decided here and the kernel
    follows it, which is the whole arrangement in this file.

    Args:
        a: The numerator column.
        b: The denominator column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of quotients, null wherever either input is null and, on an
        integer dtype, wherever the divisor is zero.
    """
    var out = Array[dt](len(a))
    for i in range(len(a)):
        if not a.is_valid(i) or not b.is_valid(i):
            out.set_null(i)
            continue
        comptime if dt.is_integral():
            if b[i] == 0:
                out.set_null(i)
                continue
        out.set_valid(i, a[i] // b[i])
    return out^


def modulo_scalar[dt: DType](a: Array[dt], b: Array[dt]) -> Array[dt]:
    """Takes the remainder of two columns, one element at a time.

    Same rule about a zero divisor as the floor division above, for the same
    reason: the two come from one rounding and cannot disagree about which rows
    have an answer.

    Args:
        a: The numerator column.
        b: The denominator column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of remainders, null wherever either input is null and, on an
        integer dtype, wherever the divisor is zero.
    """
    var out = Array[dt](len(a))
    for i in range(len(a)):
        if not a.is_valid(i) or not b.is_valid(i):
            out.set_null(i)
            continue
        comptime if dt.is_integral():
            if b[i] == 0:
                out.set_null(i)
                continue
        out.set_valid(i, a[i] % b[i])
    return out^


def power_scalar[dt: DType](a: Array[dt], b: Array[dt]) raises -> Array[dt]:
    """Raises one column to another, one element at a time.

    Two rules live here rather than in the kernel. A negative exponent on a
    signed integer dtype raises, because there is no answer in the integers and
    numpy raises rather than answering the zero the instruction gives. And a
    float power comes from the C library's `pow` rather than from the language's
    operator, because the operator is a fast approximation and the library is
    what numpy calls; see `_powers` in `arith.mojo` for the size of the
    difference.

    Args:
        a: The base column.
        b: The exponent column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of powers, null wherever either input is null.

    Raises:
        Error: If the dtype is a signed integer and any exponent is negative.
    """
    var out = Array[dt](len(a))
    for i in range(len(a)):
        if not a.is_valid(i) or not b.is_valid(i):
            out.set_null(i)
            continue
        comptime if dt.is_integral():
            comptime if dt.is_signed():
                if b[i] < 0:
                    raise Error(
                        "power: integers to negative integer powers are not"
                        " allowed"
                    )
            out.set_valid(i, a[i] ** b[i])
        elif dt == DType.float64:
            out.set_valid(
                i,
                external_call["pow", Float64](
                    a[i].cast[DType.float64](), b[i].cast[DType.float64]()
                ).cast[dt](),
            )
        else:
            out.set_valid(
                i,
                external_call["powf", Float32](
                    a[i].cast[DType.float32](), b[i].cast[DType.float32]()
                ).cast[dt](),
            )
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


def arith_const_scalar[
    dt: DType, op: Int
](a: Array[dt], b: Scalar[dt], flip: Bool = False) -> Array[dt]:
    """Applies an arithmetic operation against a constant, one element at a time.

    The three twins above are written out rather than shared, and this one is
    not, because the kernel it checks takes the operation as a code too. A twin
    that took a name would have a different shape from the thing it is checking,
    and the fuzzer would have to know which of three functions matches which
    code, which is exactly the sort of bookkeeping that hides a mismatch.

    Args:
        a: The column.
        b: The constant.
        flip: True if the constant is the left operand.

    Parameters:
        dt: The dtype.
        op: One of `OP_ADD`, `OP_SUB` or `OP_MUL`.

    Returns:
        A column of results, null where the column is null.
    """
    var out = Array[dt](len(a))
    for i in range(len(a)):
        if not a.is_valid(i):
            out.set_null(i)
        elif op == OP_ADD:
            out.set_valid(i, a[i] + b)
        elif op == OP_MUL:
            out.set_valid(i, a[i] * b)
        elif flip:
            out.set_valid(i, b - a[i])
        else:
            out.set_valid(i, a[i] - b)
    return out^


def divide_const_scalar[
    dt: DType
](a: Array[dt], b: Scalar[dt], flip: Bool = False) -> Array[DType.float64]:
    """Divides against a constant, one element at a time.

    Args:
        a: The column.
        b: The constant.
        flip: True if the constant is the numerator.

    Parameters:
        dt: The input dtype.

    Returns:
        A float64 column, null where the column is null.
    """
    var out = Array[DType.float64](len(a))
    for i in range(len(a)):
        if not a.is_valid(i):
            out.set_null(i)
        elif flip:
            out.set_valid(i, Float64(b) / Float64(a[i]))
        else:
            out.set_valid(i, Float64(a[i]) / Float64(b))
    return out^


def floor_divide_const_scalar[
    dt: DType
](a: Array[dt], b: Scalar[dt], flip: Bool = False) -> Array[dt]:
    """Floor divides against a constant, one element at a time.

    Which operand is the divisor depends on the flag, and on an integer dtype
    that is the whole difference between the two cases. Unflipped, the divisor
    is the same for every row and the column is either all null or has no nulls
    the input did not already have. Flipped, the divisor comes out of the column
    and every row has to be looked at. The kernel takes those two apart and this
    does not, which is the point.

    Args:
        a: The column.
        b: The constant.
        flip: True if the constant is the numerator.

    Parameters:
        dt: The dtype.

    Returns:
        A column of quotients, null where the column is null and, on an integer
        dtype, wherever the divisor is zero.
    """
    var out = Array[dt](len(a))
    for i in range(len(a)):
        if not a.is_valid(i):
            out.set_null(i)
            continue
        var numerator = b if flip else a[i]
        var divisor = a[i] if flip else b
        comptime if dt.is_integral():
            if divisor == 0:
                out.set_null(i)
                continue
        out.set_valid(i, numerator // divisor)
    return out^


def modulo_const_scalar[
    dt: DType
](a: Array[dt], b: Scalar[dt], flip: Bool = False) -> Array[dt]:
    """Takes the remainder against a constant, one element at a time.

    Same rule about a zero divisor as the floor division above, and for the same
    reason: the two come from one rounding and cannot disagree about which rows
    have an answer.

    Args:
        a: The column.
        b: The constant.
        flip: True if the constant is the numerator.

    Parameters:
        dt: The dtype.

    Returns:
        A column of remainders, null where the column is null and, on an integer
        dtype, wherever the divisor is zero.
    """
    var out = Array[dt](len(a))
    for i in range(len(a)):
        if not a.is_valid(i):
            out.set_null(i)
            continue
        var numerator = b if flip else a[i]
        var divisor = a[i] if flip else b
        comptime if dt.is_integral():
            if divisor == 0:
                out.set_null(i)
                continue
        out.set_valid(i, numerator % divisor)
    return out^


def power_const_scalar[
    dt: DType
](a: Array[dt], b: Scalar[dt], flip: Bool = False) raises -> Array[dt]:
    """Raises against a constant, one element at a time.

    The refusal of a negative exponent on a signed integer dtype is hoisted out
    of the loop when the constant is the exponent, and left in the loop when the
    column is. That is not an optimisation carried over from the kernel. A
    constant exponent is a property of the operation and not of a row, so it is
    wrong on a column with nothing in it just as much as on a full one, and the
    call has to fail either way.

    Args:
        a: The column.
        b: The constant.
        flip: True if the constant is the base.

    Parameters:
        dt: The dtype.

    Returns:
        A column of powers, null where the column is null.

    Raises:
        Error: If the dtype is a signed integer and an exponent is negative.
    """
    comptime if dt.is_integral() and dt.is_signed():
        if not flip and b < 0:
            raise Error(
                "power: integers to negative integer powers are not allowed"
            )
    var out = Array[dt](len(a))
    for i in range(len(a)):
        if not a.is_valid(i):
            out.set_null(i)
            continue
        var base = b if flip else a[i]
        var exponent = a[i] if flip else b
        comptime if dt.is_integral():
            comptime if dt.is_signed():
                if exponent < 0:
                    raise Error(
                        "power: integers to negative integer powers are not"
                        " allowed"
                    )
            out.set_valid(i, base**exponent)
        elif dt == DType.float64:
            out.set_valid(
                i,
                external_call["pow", Float64](
                    base.cast[DType.float64](),
                    exponent.cast[DType.float64](),
                ).cast[dt](),
            )
        else:
            out.set_valid(
                i,
                external_call["powf", Float32](
                    base.cast[DType.float32](),
                    exponent.cast[DType.float32](),
                ).cast[dt](),
            )
    return out^


def compare_const_scalar[
    dt: DType, op: Int
](a: Array[dt], b: Scalar[dt]) -> Array[DType.bool]:
    """Compares against a constant, one element at a time.

    Args:
        a: The column.
        b: The constant.

    Parameters:
        dt: The dtype.
        op: One of the `CMP_` codes.

    Returns:
        A bool column, null where the column is null.
    """
    var out = Array[DType.bool](len(a))
    for i in range(len(a)):
        if not a.is_valid(i):
            out.set_null(i)
        elif op == CMP_EQ:
            out.set_valid(i, a[i] == b)
        elif op == CMP_NE:
            out.set_valid(i, a[i] != b)
        elif op == CMP_LT:
            out.set_valid(i, a[i] < b)
        elif op == CMP_LE:
            out.set_valid(i, a[i] <= b)
        elif op == CMP_GT:
            out.set_valid(i, a[i] > b)
        else:
            out.set_valid(i, a[i] >= b)
    return out^


def compare_text_scalar[
    op: Int
](a: StringArray, b: StringArray) -> Array[DType.bool]:
    """Compares two text columns, one element at a time, on copies of the bytes.

    Each element is pulled out as a `String` and the two are compared with the
    language's own comparison. That is what makes this a check rather than a
    second copy of the kernel: the kernel compares packed views, and settles most
    pairs without reading either string's bytes, so a twin that walked the views
    the same way would agree with it about anything that arrangement got wrong.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        op: One of the `CMP_` codes.

    Returns:
        A bool column, null wherever either input is null.
    """
    var out = Array[DType.bool](len(a))
    for i in range(len(a)):
        if not a.is_valid(i) or not b.is_valid(i):
            out.set_null(i)
            continue
        var left = a[i]
        var right = b[i]
        if op == CMP_EQ:
            out.set_valid(i, left == right)
        elif op == CMP_NE:
            out.set_valid(i, left != right)
        elif op == CMP_LT:
            out.set_valid(i, left < right)
        elif op == CMP_LE:
            out.set_valid(i, left <= right)
        elif op == CMP_GT:
            out.set_valid(i, left > right)
        else:
            out.set_valid(i, left >= right)
    return out^


def compare_text_const_scalar[
    op: Int
](a: StringArray, b: String) -> Array[DType.bool]:
    """Compares a text column against one string, one element at a time.

    Args:
        a: The column.
        b: The constant.

    Parameters:
        op: One of the `CMP_` codes.

    Returns:
        A bool column, null where the column is null.
    """
    var out = Array[DType.bool](len(a))
    for i in range(len(a)):
        if not a.is_valid(i):
            out.set_null(i)
            continue
        var left = a[i]
        if op == CMP_EQ:
            out.set_valid(i, left == b)
        elif op == CMP_NE:
            out.set_valid(i, left != b)
        elif op == CMP_LT:
            out.set_valid(i, left < b)
        elif op == CMP_LE:
            out.set_valid(i, left <= b)
        elif op == CMP_GT:
            out.set_valid(i, left > b)
        else:
            out.set_valid(i, left >= b)
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
        If the reduction is not one of the single column kinds.
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
            # `_is_there` and not `is_valid`, so a NaN is left out of the group
            # the way a null is. The kernel asks the same question inside its
            # own loops with `_there`; this asks it a row at a time, which is
            # what makes the two agreeing worth something. `SIZE` is above this
            # line because it counts rows and not values. See #170.
            if _is_there(col, i):
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
                # A group with nothing in it says so with a NaN in a row that
                # stays valid, because that is the only missing a pandas float
                # column has. A mean always answers in float64 so it always
                # takes that branch. `min`, `max`, `first` and `last` below keep
                # the column's own dtype, so they ask the dtype. See #170.
                values.append(nan[DType.float64]())
                valid.append(True)
            else:
                var total = Float64(0)
                for k in range(len(present)):
                    total += present[k]
                values.append(total / Float64(len(present)))
                valid.append(True)
        elif kind == AggKind.MIN or kind == AggKind.MAX:
            if len(present) == 0:
                # These two keep the column's own dtype, so the spelling of
                # missing is the column's decision and not the reduction's. A
                # float column has a NaN to write and takes it. Anything else
                # has no NaN and a null is all there is.
                comptime if dt.is_floating_point():
                    values.append(nan[DType.float64]())
                    valid.append(True)
                else:
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
                # Same rule as the two extremes above and for the same reason,
                # since these also answer in the column's own dtype.
                comptime if dt.is_floating_point():
                    values.append(nan[DType.float64]())
                    valid.append(True)
                else:
                    values.append(Float64(0))
                    valid.append(False)
            else:
                var at = 0 if kind == AggKind.FIRST else len(present) - 1
                values.append(present[at])
                valid.append(True)
        elif kind == AggKind.VAR or kind == AggKind.STD or kind == AggKind.SEM:
            var divisor = len(present) - Int(kind.param)
            if divisor <= 0:
                values.append(nan[DType.float64]())
                valid.append(True)
            else:
                var total = Float64(0)
                for k in range(len(present)):
                    total += present[k]
                var centre = total / Float64(len(present))
                var squares = Float64(0)
                for k in range(len(present)):
                    squares += (present[k] - centre) * (present[k] - centre)
                var spread = squares / Float64(divisor)
                if kind != AggKind.VAR:
                    spread = sqrt(spread)
                # The count under this root is the plain one and not the
                # corrected divisor. The degrees of freedom belong to the
                # variance, and applying the correction twice is a different
                # statistic from the one pandas reports.
                if kind == AggKind.SEM:
                    spread = spread / sqrt(Float64(len(present)))
                values.append(spread)
                valid.append(True)
        elif kind == AggKind.SKEW:
            if len(present) < 3:
                values.append(nan[DType.float64]())
                valid.append(True)
            else:
                var total = Float64(0)
                for k in range(len(present)):
                    total += present[k]
                var size = Float64(len(present))
                var centre = total / size
                var second = Float64(0)
                var third = Float64(0)
                for k in range(len(present)):
                    var delta = present[k] - centre
                    second += delta * delta
                    third += delta * delta * delta
                second = second / size
                third = third / size
                if second == 0.0:
                    # A group whose values are all the same is symmetric, not
                    # undefined, and pandas reports zero for it.
                    values.append(Float64(0))
                else:
                    var adjust = sqrt(size * (size - 1.0)) / (size - 2.0)
                    values.append(adjust * third / (second * sqrt(second)))
                valid.append(True)
        elif kind == AggKind.MEDIAN or kind == AggKind.QUANTILE:
            if len(present) == 0:
                values.append(nan[DType.float64]())
                valid.append(True)
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


def concat_scalar[dt: DType](parts: List[Array[dt]]) -> Array[dt]:
    """Stacks columns end to end, one element at a time.

    Args:
        parts: The columns, in output order.

    Parameters:
        dt: The dtype.

    Returns:
        A column as tall as the parts put together.
    """
    var total = 0
    for p in range(len(parts)):
        total += len(parts[p])

    var out = Array[dt](total)
    var at = 0
    for p in range(len(parts)):
        for i in range(len(parts[p])):
            if parts[p].is_valid(i):
                out.set_valid(at, parts[p][i])
            else:
                out.set_null(at)
            at += 1
    return out^


def coalesce_scalar[dt: DType](a: Array[dt], b: Array[dt]) -> Array[dt]:
    """Picks the first present value of the two, one row at a time.

    A fallback of one row is used for every row, which is how filling with a
    scalar is spelled.

    Args:
        a: The preferred column.
        b: The fallback, either as tall as `a` or one row.

    Parameters:
        dt: The dtype.

    Returns:
        A column as tall as `a`.
    """
    var out = Array[dt](len(a))
    for i in range(len(a)):
        if a.is_valid(i):
            out.set_valid(i, a[i])
            continue
        var at = 0 if len(b) == 1 else i
        if at < len(b) and b.is_valid(at):
            out.set_valid(i, b[at])
        else:
            out.set_null(i)
    return out^


def fill_scalar[
    dt: DType, //, forward: Bool
](col: Array[dt], limit: Int) -> Array[dt]:
    """Carries the nearest present value over the missing rows, one at a time.

    `_is_there` and not `is_valid`, so a NaN is filled over and is never carried,
    which is the rule the kernel gets to by folding a vector `isnan` into the
    block copy and testing the value inline in the row loop. This walks out from
    each missing row until it finds one that is there, which cannot get a word
    boundary wrong because it has no words, and cannot get the fold wrong because
    it has no blocks. See #170.

    Args:
        col: The column.
        limit: The longest run of nulls to fill, or zero for no limit.

    Parameters:
        dt: The dtype.
        forward: True to carry from earlier rows, false to carry from later ones.

    Returns:
        A column of the same height, with a row that could not be filled spelled
        the way the dtype spells missing, as in the kernel.
    """
    var out = Array[dt](len(col))
    for i in range(len(col)):
        if _is_there(col, i):
            out.set_valid(i, col[i])
            continue

        var run = 0
        var found = False
        var step = i - 1 if forward else i + 1
        while step >= 0 and step < len(col):
            run += 1
            if _is_there(col, step):
                if limit <= 0 or run <= limit:
                    out.set_valid(i, col[step])
                    found = True
                break
            step = step - 1 if forward else step + 1
        if not found:
            comptime if dt.is_floating_point():
                out.set_valid(i, nan[dt]())
            else:
                out.set_null(i)
    return out^


def is_null_scalar[dt: DType](col: Array[dt]) -> Array[DType.bool]:
    """Reports which rows are missing, one row at a time.

    A NaN is missing on a float column, which is what the kernel says and what
    pandas says. The kernel gets there by asking a word of sixty four rows
    whether it holds any NaN before it looks at a single bit; this asks each row
    on its own and cannot get a boundary wrong, which is the whole point of it
    being here. See #170.

    Args:
        col: The column.

    Parameters:
        dt: The dtype.

    Returns:
        A bool column with no nulls of its own.
    """
    var out = Array[DType.bool](len(col))
    for i in range(len(col)):
        var missing = not col.is_valid(i)
        comptime if dt.is_floating_point():
            if isnan(col[i]):
                missing = True
        out[i] = missing
    return out^


def group_text_scalar(
    col: StringArray,
    kind: AggKind,
    codes: Array[DType.uint32],
    groups: Int,
) raises -> Tuple[List[String], List[Bool]]:
    """Aggregates a text column by collecting each group's values and reducing.

    The twin for the four reductions that report a value the column held. The
    three that count are not here, because a count over text is the same loop as
    a count over numbers and `group_scalar` already covers it, and a twin that
    exists only to be a second copy of another twin is a second place to be
    wrong.

    Values come back as `String` rather than as rows, which is the difference
    that makes this a check rather than a restatement. The kernel decides which
    row to keep and gathers at the end; this one holds the bytes and compares
    them with the language's own comparison, so an error in the byte comparison
    the kernel uses cannot hide here.

    Args:
        col: The text column being aggregated.
        kind: Which reduction.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.

    Returns:
        One value per group and one validity flag per group.

    Raises:
        If the reduction is not one of the four.
    """
    if not (
        kind == AggKind.FIRST
        or kind == AggKind.LAST
        or kind == AggKind.MIN
        or kind == AggKind.MAX
    ):
        raise Error(
            "group by twin: " + String(kind) + " does not report a text value"
        )

    var values = List[String](capacity=groups)
    var valid = List[Bool](capacity=groups)

    for g in range(groups):
        # Every row, for every group, the same as the number twin. It is O(groups
        # times rows) on purpose: the bucketing pass is the thing being checked,
        # so the twin cannot use one.
        var held = String("")
        var seen = False
        for i in range(len(codes)):
            if Int(codes[i]) != g or not col.is_valid(i):
                continue
            var value = col[i]
            if not seen:
                held = value
                seen = True
            elif kind == AggKind.LAST:
                held = value
            elif kind == AggKind.MIN and value < held:
                held = value
            elif kind == AggKind.MAX and value > held:
                held = value
        values.append(held)
        valid.append(seen)

    return (values^, valid^)


def group_top_scalar[
    dt: DType
](
    col: Array[dt],
    codes: Array[DType.uint32],
    groups: Int,
    n: Int,
    largest: Bool,
) raises -> List[Int]:
    """Picks each group's best rows by scanning the column once per slot.

    The definition of a top-n made literal. For each group, and then for each of
    its `n` slots, walk every row and keep the best one that belongs to the group
    and has not already been taken. That is `groups * n * rows` work and it is
    exactly what the phrase means, which is the whole point of a twin.

    Args:
        col: The column being ranked.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.
        n: How many rows to keep per group.
        largest: True to keep the high values, False to keep the low ones.

    Parameters:
        dt: The column's dtype.

    Returns:
        `groups * n` row numbers, group major and best first, with `-1` in a slot
        the group had no value for.

    Raises:
        If the column and the ordinals are different lengths.
    """
    if len(col) != len(codes):
        raise Error("group top twin: the column and the ordinals disagree")

    var picked = List[Int](length=groups * n, fill=-1)
    for g in range(groups):
        for k in range(n):
            var best = -1
            for i in range(len(codes)):
                if Int(codes[i]) != g or not col.is_valid(i):
                    continue
                var value = col[i]
                if value != value:
                    continue

                var taken = False
                for j in range(k):
                    if picked[g * n + j] == i:
                        taken = True
                        break
                if taken:
                    continue

                if best < 0:
                    best = i
                    continue
                var against = col[best]
                if largest:
                    if value > against:
                        best = i
                elif value < against:
                    best = i
            picked[g * n + k] = best
            if best < 0:
                break
    return picked^
