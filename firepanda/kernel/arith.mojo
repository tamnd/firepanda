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

Each operation has a second form that takes a constant on one side instead of a
second column. That is not a convenience wrapper over building a column of a
repeated value: the constant is splatted into a register once, outside the loop,
so the loop reads one operand instead of two and touches half the memory. On a
column that does not fit in cache that is the whole difference. The constant
forms live here rather than in a file of their own because a reader comparing
`x + y` against `x + 5` should see both loops at once.
"""

from std.sys.info import simd_width_of

from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.exec import parallel_morsels

from .mask import apply_validity, combined_validity

comptime OP_ADD = 0
"""Operation code for addition."""

comptime OP_SUB = 1
"""Operation code for subtraction."""

comptime OP_MUL = 2
"""Operation code for multiplication."""


def add[dt: DType](a: Array[dt], b: Array[dt]) raises -> Array[dt]:
    """Adds two columns elementwise.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of sums, null wherever either input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _arith[dt, OP_ADD](a, b)


def subtract[dt: DType](a: Array[dt], b: Array[dt]) raises -> Array[dt]:
    """Subtracts one column from another elementwise.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of differences, null wherever either input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _arith[dt, OP_SUB](a, b)


def multiply[dt: DType](a: Array[dt], b: Array[dt]) raises -> Array[dt]:
    """Multiplies two columns elementwise.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of products, null wherever either input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _arith[dt, OP_MUL](a, b)


def _arith[dt: DType, op: Int](a: Array[dt], b: Array[dt]) raises -> Array[dt]:
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

    Raises:
        Error: Only what the morsel runtime raises. The arithmetic itself
            cannot fail; a division by zero is an infinity and lives in
            `divide` anyway.
    """
    comptime width = simd_width_of[dt]()

    var n = len(a)
    # The loop below writes a result for every row, the null ones included,
    # which is the whole point of computing first and repairing afterwards. So
    # the allocation does not need the pass that zeroes it.
    var out = Array[dt](overwritten=n)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var lhs = a.unsafe_ptr()
        var rhs = b.unsafe_ptr()
        var dst = out.unsafe_ptr()
        var i = start
        while i < stop:
            var x = lhs.unsafe_offset(i).unsafe_load[width=width]()
            var y = rhs.unsafe_offset(i).unsafe_load[width=width]()

            comptime if op == OP_ADD:
                dst.unsafe_offset(i).unsafe_store(x + y)
            elif op == OP_SUB:
                dst.unsafe_offset(i).unsafe_store(x - y)
            else:
                dst.unsafe_offset(i).unsafe_store(x * y)
            i += width

    parallel_morsels(compute, n)

    apply_validity(out, combined_validity(a.data.validity, b.data.validity))
    return out^


def divide[
    dt: DType
](a: Array[dt], b: Array[dt]) raises -> Array[DType.float64]:
    """Divides two columns elementwise, in float64.

    Args:
        a: The numerator column.
        b: The denominator column. Must be the same length as `a`.

    Parameters:
        dt: The input dtype.

    Returns:
        A float64 column, null wherever either input is null.

    Raises:
        Error: Only what the morsel runtime raises. Dividing by zero here is an
            infinity or a NaN rather than an error.
    """
    comptime width = simd_width_of[DType.float64]()

    var n = len(a)
    # Every row is written below, nulls included, so the zeroing constructor
    # would be a pass thrown away.
    var out = Array[DType.float64](overwritten=n)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var lhs = a.unsafe_ptr()
        var rhs = b.unsafe_ptr()
        var dst = out.unsafe_ptr()
        var i = start
        while i < stop:
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

    parallel_morsels(compute, n)

    apply_validity(out, combined_validity(a.data.validity, b.data.validity))
    return out^


def arith_const[
    dt: DType, op: Int
](a: Array[dt], b: Scalar[dt], flip: Bool = False) raises -> Array[dt]:
    """Applies an arithmetic operation between a column and one constant.

    The four operations do not need four entry points here the way the column
    forms do, because the caller that reaches this already holds the operation as
    a code rather than as a name. It arrived from a plan.

    `flip` puts the constant on the left, which only changes anything for
    subtraction. The branch on it is outside the loop rather than a parameter,
    so there is one instantiation of this per dtype rather than two, and the
    loop the processor runs is still straight line. Compile time is a real
    budget here: this is instantiated once per dtype in the erased dispatch, and
    doubling that doubles the cost of a file nothing has even called yet.

    Args:
        a: The column.
        b: The constant.
        flip: True if the constant is the left operand.

    Parameters:
        dt: The dtype. The constant is already at it; promotion happened above.
        op: One of `OP_ADD`, `OP_SUB` or `OP_MUL`.

    Returns:
        A column of results, null wherever the column is null. A constant is
        never null here; a null constant makes the whole answer null and is
        handled before any loop runs.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[dt]()

    var n = len(a)
    # Every row is written below, so the zeroing allocation is a wasted pass.
    var out = Array[dt](overwritten=n)
    var y = SIMD[dt, width](b)

    # The branch on `flip` stays outside the inner loop, one test per morsel
    # rather than one per register, which is what it was when there was a
    # single loop over the whole column.
    def compute(start: Int, stop: Int) {mut out, imm}:
        var src = a.unsafe_ptr()
        var dst = out.unsafe_ptr()
        comptime if op == OP_ADD:
            var i = start
            while i < stop:
                var x = src.unsafe_offset(i).unsafe_load[width=width]()
                dst.unsafe_offset(i).unsafe_store(x + y)
                i += width
        elif op == OP_MUL:
            var i = start
            while i < stop:
                var x = src.unsafe_offset(i).unsafe_load[width=width]()
                dst.unsafe_offset(i).unsafe_store(x * y)
                i += width
        else:
            if flip:
                var i = start
                while i < stop:
                    var x = src.unsafe_offset(i).unsafe_load[width=width]()
                    dst.unsafe_offset(i).unsafe_store(y - x)
                    i += width
            else:
                var i = start
                while i < stop:
                    var x = src.unsafe_offset(i).unsafe_load[width=width]()
                    dst.unsafe_offset(i).unsafe_store(x - y)
                    i += width

    parallel_morsels(compute, n)

    apply_validity(out, Bitmap(copy=a.data.validity))
    return out^


def divide_const[
    dt: DType
](a: Array[dt], b: Scalar[dt], flip: Bool = False) raises -> Array[
    DType.float64
]:
    """Divides a column by a constant, or a constant by a column, in float64.

    Args:
        a: The column.
        b: The constant.
        flip: True if the constant is the numerator.

    Parameters:
        dt: The input dtype.

    Returns:
        A float64 column, null wherever the column is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[DType.float64]()

    var n = len(a)
    # Every row is written below, so the zeroing allocation is a wasted pass.
    var out = Array[DType.float64](overwritten=n)
    var y = SIMD[DType.float64, width](b.cast[DType.float64]())

    def compute(start: Int, stop: Int) {mut out, imm}:
        var src = a.unsafe_ptr()
        var dst = out.unsafe_ptr()
        if flip:
            var i = start
            while i < stop:
                var x = (
                    src.unsafe_offset(i)
                    .unsafe_load[width=width]()
                    .cast[DType.float64]()
                )
                dst.unsafe_offset(i).unsafe_store(y / x)
                i += width
        else:
            var i = start
            while i < stop:
                var x = (
                    src.unsafe_offset(i)
                    .unsafe_load[width=width]()
                    .cast[DType.float64]()
                )
                dst.unsafe_offset(i).unsafe_store(x / y)
                i += width

    parallel_morsels(compute, n)

    apply_validity(out, Bitmap(copy=a.data.validity))
    return out^
