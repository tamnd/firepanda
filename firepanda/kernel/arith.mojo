"""Elementwise arithmetic on two columns.

All four operations are the same loop with one instruction changed, so they share
a body and pick the instruction with a `comptime if` on an operation code. A
function-valued parameter would read better; an integer that the compiler folds
away before the loop exists is what actually compiles here today.

Every operation computes across the whole values buffer and repairs the null
positions afterwards. See `mask.mojo` for why that is cheaper than checking.

Division is the exception to the shape. It always produces float64, whatever went
in, because that is what pandas does with `/` and because integer division that
silently truncates is a bug generator. Division by zero gives an infinity or a
NaN, not a null, which is also pandas.
"""

from std.sys.info import simd_width_of

from firepanda.array.array import Array

from .mask import apply_validity, combined_validity

comptime OP_ADD = 0
"""Operation code for addition."""

comptime OP_SUB = 1
"""Operation code for subtraction."""

comptime OP_MUL = 2
"""Operation code for multiplication."""


def add[dt: DType](a: Array[dt], b: Array[dt]) -> Array[dt]:
    """Adds two columns elementwise.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of sums, null wherever either input is null.
    """
    return _arith[dt, OP_ADD](a, b)


def subtract[dt: DType](a: Array[dt], b: Array[dt]) -> Array[dt]:
    """Subtracts one column from another elementwise.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of differences, null wherever either input is null.
    """
    return _arith[dt, OP_SUB](a, b)


def multiply[dt: DType](a: Array[dt], b: Array[dt]) -> Array[dt]:
    """Multiplies two columns elementwise.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of products, null wherever either input is null.
    """
    return _arith[dt, OP_MUL](a, b)


def _arith[dt: DType, op: Int](a: Array[dt], b: Array[dt]) -> Array[dt]:
    """Applies an arithmetic operation elementwise.

    The tail past the last full register is handled by loading a full register
    anyway. The values buffer is padded to a 64-byte multiple and the padding is
    zero, so the read is in bounds and the lanes past the length are written back
    into padding nobody reads.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.
        op: One of `OP_ADD`, `OP_SUB` or `OP_MUL`.

    Returns:
        A column of results, null wherever either input is null.
    """
    comptime width = simd_width_of[dt]()

    var n = len(a)
    var out = Array[dt](n)
    var lhs = a.unsafe_ptr()
    var rhs = b.unsafe_ptr()
    var dst = out.unsafe_ptr()

    var i = 0
    while i < n:
        var x = lhs.unsafe_offset(i).unsafe_load[width=width]()
        var y = rhs.unsafe_offset(i).unsafe_load[width=width]()

        comptime if op == OP_ADD:
            dst.unsafe_offset(i).unsafe_store(x + y)
        elif op == OP_SUB:
            dst.unsafe_offset(i).unsafe_store(x - y)
        else:
            dst.unsafe_offset(i).unsafe_store(x * y)
        i += width

    apply_validity(out, combined_validity(a.data.validity, b.data.validity))
    return out^


def divide[dt: DType](a: Array[dt], b: Array[dt]) -> Array[DType.float64]:
    """Divides two columns elementwise, in float64.

    Args:
        a: The numerator column.
        b: The denominator column. Must be the same length as `a`.

    Parameters:
        dt: The input dtype.

    Returns:
        A float64 column, null wherever either input is null.
    """
    comptime width = simd_width_of[DType.float64]()

    var n = len(a)
    var out = Array[DType.float64](n)
    var lhs = a.unsafe_ptr()
    var rhs = b.unsafe_ptr()
    var dst = out.unsafe_ptr()

    var i = 0
    while i < n:
        var x = (
            lhs.unsafe_offset(i)
            .unsafe_load[width=width]()
            .cast[DType.float64]()
        )
        var y = (
            rhs.unsafe_offset(i)
            .unsafe_load[width=width]()
            .cast[DType.float64]()
        )
        dst.unsafe_offset(i).unsafe_store(x / y)
        i += width

    apply_validity(out, combined_validity(a.data.validity, b.data.validity))
    return out^
