"""Elementwise comparison of two columns, producing a boolean column.

Six operations, one loop, an operation code picked at compile time. Same shape as
`arith.mojo` and for the same reason.

The output is a `bool` column, one byte per value, not a bitmap. A bitmap would be
eight times smaller and would then have to be unpacked by every kernel that
consumes a mask, which is all of them. Arrow stores boolean arrays as bits and
pays that cost; firepanda stores bytes and pays the memory instead. The filter
kernel reads a byte per row with no shifting as a result.

Comparison against a null is null, not false. That is three-valued logic and it
is what both pandas and SQL do, and it is why `filter_rows` has to decide
separately what a null in a mask means; see `select.mojo`.

There is a second form that takes a constant on the right, and it needs no flag
for a constant on the left, because `5 < x` is `x > 5` and the caller mirrors the
operation rather than the operands. That mirroring is exact, NaN included: both
readings of the pair are false when either side is not a number, so nothing is
smuggled in by rewriting one as the other.
"""

from std.sys.info import simd_width_of

from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.exec import parallel_morsels

from .mask import apply_validity, combined_validity

comptime CMP_EQ = 0
"""Operation code for equality."""

comptime CMP_NE = 1
"""Operation code for inequality."""

comptime CMP_LT = 2
"""Operation code for less-than."""

comptime CMP_LE = 3
"""Operation code for less-than-or-equal."""

comptime CMP_GT = 4
"""Operation code for greater-than."""

comptime CMP_GE = 5
"""Operation code for greater-than-or-equal."""


def equal[dt: DType](a: Array[dt], b: Array[dt]) raises -> Array[DType.bool]:
    """Compares two columns for equality.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A bool column, null wherever either input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _compare[dt, CMP_EQ](a, b)


def not_equal[
    dt: DType
](a: Array[dt], b: Array[dt]) raises -> Array[DType.bool]:
    """Compares two columns for inequality.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A bool column, null wherever either input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _compare[dt, CMP_NE](a, b)


def less[dt: DType](a: Array[dt], b: Array[dt]) raises -> Array[DType.bool]:
    """Reports elementwise whether the left column is smaller.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A bool column, null wherever either input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _compare[dt, CMP_LT](a, b)


def less_equal[
    dt: DType
](a: Array[dt], b: Array[dt]) raises -> Array[DType.bool]:
    """Reports elementwise whether the left column is smaller or equal.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A bool column, null wherever either input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _compare[dt, CMP_LE](a, b)


def greater[dt: DType](a: Array[dt], b: Array[dt]) raises -> Array[DType.bool]:
    """Reports elementwise whether the left column is larger.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A bool column, null wherever either input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _compare[dt, CMP_GT](a, b)


def greater_equal[
    dt: DType
](a: Array[dt], b: Array[dt]) raises -> Array[DType.bool]:
    """Reports elementwise whether the left column is larger or equal.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A bool column, null wherever either input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _compare[dt, CMP_GE](a, b)


def _compare[
    dt: DType, op: Int
](a: Array[dt], b: Array[dt]) raises -> Array[DType.bool]:
    """Applies a comparison elementwise.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.
        op: One of the `CMP_` codes.

    Returns:
        A bool column, null wherever either input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[dt]()

    var n = len(a)
    # Every row is written below, the null ones included, and `apply_validity`
    # blanks those afterwards. The zeroing constructor would be a wasted pass.
    var out = Array[DType.bool](overwritten=n)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var lhs = a.unsafe_ptr()
        var rhs = b.unsafe_ptr()
        var dst = out.unsafe_ptr()
        var i = start
        while i < stop:
            var x = lhs.unsafe_offset(i).unsafe_load[width=width]()
            var y = rhs.unsafe_offset(i).unsafe_load[width=width]()

            # `x < y` on a register does not mean what it looks like it means.
            # The operators on `SIMD` are constrained to width one, because a
            # whole-vector `<` would have to answer with a single Bool and there
            # is no honest answer. The lanewise forms are the named methods.
            comptime if op == CMP_EQ:
                dst.unsafe_offset(i).unsafe_store(x.eq(y))
            elif op == CMP_NE:
                dst.unsafe_offset(i).unsafe_store(x.ne(y))
            elif op == CMP_LT:
                dst.unsafe_offset(i).unsafe_store(x.lt(y))
            elif op == CMP_LE:
                dst.unsafe_offset(i).unsafe_store(x.le(y))
            elif op == CMP_GT:
                dst.unsafe_offset(i).unsafe_store(x.gt(y))
            else:
                dst.unsafe_offset(i).unsafe_store(x.ge(y))
            i += width

    parallel_morsels(compute, n)

    apply_validity(out, combined_validity(a.data.validity, b.data.validity))
    return out^


def compare_const[
    dt: DType, op: Int
](a: Array[dt], b: Scalar[dt]) raises -> Array[DType.bool]:
    """Compares a column against one constant.

    The constant is splatted once, before the loop, so each row costs one load
    rather than two. There is no flipped form, because a constant on the left is
    the mirrored operation on the right and the caller does that swap.

    Args:
        a: The column.
        b: The constant.

    Parameters:
        dt: The dtype. The constant is already at it; promotion happened above.
        op: One of the `CMP_` codes.

    Returns:
        A bool column, null wherever the column is null. A null constant makes
        the whole answer null and never reaches here.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[dt]()

    var n = len(a)
    # Every row is written below, so the zeroing allocation is a wasted pass.
    var out = Array[DType.bool](overwritten=n)
    var y = SIMD[dt, width](b)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var src = a.unsafe_ptr()
        var dst = out.unsafe_ptr()
        var i = start
        while i < stop:
            var x = src.unsafe_offset(i).unsafe_load[width=width]()

            comptime if op == CMP_EQ:
                dst.unsafe_offset(i).unsafe_store(x.eq(y))
            elif op == CMP_NE:
                dst.unsafe_offset(i).unsafe_store(x.ne(y))
            elif op == CMP_LT:
                dst.unsafe_offset(i).unsafe_store(x.lt(y))
            elif op == CMP_LE:
                dst.unsafe_offset(i).unsafe_store(x.le(y))
            elif op == CMP_GT:
                dst.unsafe_offset(i).unsafe_store(x.gt(y))
            else:
                dst.unsafe_offset(i).unsafe_store(x.ge(y))
            i += width

    parallel_morsels(compute, n)

    apply_validity(out, Bitmap(copy=a.data.validity))
    return out^
