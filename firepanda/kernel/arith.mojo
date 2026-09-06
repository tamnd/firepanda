"""Elementwise arithmetic on two columns.

Most of these operations are the same loop with one instruction changed, so they
share a body and pick the instruction with a `comptime if` on an operation code.
A function-valued parameter would read better; an integer that the compiler folds
away before the loop exists is what actually compiles here today.

Every operation computes across the whole values buffer and repairs the null
positions afterwards. See `mask.mojo` for why that is cheaper than checking.

Division is the exception to the shape. It always produces float64, whatever went
in, because that is what pandas does with `/` and because integer division that
silently truncates is a bug generator. Division by zero gives an infinity or a
NaN, not a null, which is also pandas.

Floor division and remainder keep the operand type, which is what pandas does
with `//` and `%` right up until a divisor is zero. There pandas has four
answers depending on which array is underneath a series: a numpy int64 column
widens to float64 to make room for an infinity, a masked Int64 column answers
zero and says nothing, an Arrow backed column raises, and a float column answers
the infinity the hardware produced. firepanda answers a null and keeps the
operand type, which is the only one of those that is both total and a function
of the types alone, and it is a registered divergence rather than a quiet
difference. Floats still answer the infinity, because a float column can hold
one.

Nulling those rows is the only place in this file where the result depends on a
value, so it is the only loop that touches the validity it was handed. The test
is a horizontal reduce over the register, which is a few instructions against
the twenty or more an integer division costs, and the per row walk that clears
the bits only runs on a register that actually holds a zero.

Raising a column to a power is the same loop again with two exceptions of its
own. A negative exponent on an integer column has no answer in the integers, and
numpy raises rather than truncating to zero, so the loop tests the exponent
register and raises the same way. And on a float column the loop is not a vector
instruction at all, because the vector one is a fast approximation that gets the
square root of two wrong in the thirteenth digit; see `_powers`.

Each operation has a second form that takes a constant on one side instead of a
second column. That is not a convenience wrapper over building a column of a
repeated value: the constant is splatted into a register once, outside the loop,
so the loop reads one operand instead of two and touches half the memory. On a
column that does not fit in cache that is the whole difference. The constant
forms live here rather than in a file of their own because a reader comparing
`x + y` against `x + 5` should see both loops at once.

A constant is also where a whole loop can sometimes be skipped. A constant zero
divisor makes every row null and a constant negative exponent raises, and both
are known before a row is read, so both are answered above the loop rather than
found out inside it once per register.
"""

from std.ffi import external_call
from std.sys.info import simd_width_of

from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.exec import parallel_morsels

from .mask import combined_validity, repair_range

comptime OP_ADD = 0
"""Operation code for addition."""

comptime OP_SUB = 1
"""Operation code for subtraction."""

comptime OP_MUL = 2
"""Operation code for multiplication."""

comptime OP_FLOORDIV = 3
"""Operation code for floor division."""

comptime OP_MOD = 4
"""Operation code for the remainder."""

comptime OP_POW = 5
"""Operation code for raising to a power."""


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


def floor_divide[dt: DType](a: Array[dt], b: Array[dt]) raises -> Array[dt]:
    """Divides two columns elementwise and rounds towards minus infinity.

    The rounding is the Python one and not the C one, so a negative numerator
    rounds away from zero rather than towards it and `-7 // 3` is `-3`. That is
    what numpy answers and what the hardware instruction underneath already
    does here.

    Args:
        a: The numerator column.
        b: The denominator column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of quotients of the same dtype, null wherever either input is
        null and, on an integer dtype, null wherever the divisor is zero. A
        float column divided by zero holds an infinity rather than a null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime if dt.is_integral():
        return _int_divide[dt, OP_FLOORDIV](a, b)
    else:
        return _arith[dt, OP_FLOORDIV](a, b)


def modulo[dt: DType](a: Array[dt], b: Array[dt]) raises -> Array[dt]:
    """Takes the remainder of one column divided by another elementwise.

    The remainder takes the sign of the divisor, which is the Python rule and
    not the C one, so `-7 % 3` is `2` rather than `-1`. It is the remainder that
    goes with the floor division above, and the two are consistent for the same
    reason they are in Python: the quotient and the remainder come from the same
    rounding.

    Args:
        a: The numerator column.
        b: The denominator column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of remainders of the same dtype, null wherever either input is
        null and, on an integer dtype, null wherever the divisor is zero. A
        float column has a NaN there rather than a null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime if dt.is_integral():
        return _int_divide[dt, OP_MOD](a, b)
    else:
        return _arith[dt, OP_MOD](a, b)


def power[dt: DType](a: Array[dt], b: Array[dt]) raises -> Array[dt]:
    """Raises one column to the power of another elementwise.

    Args:
        a: The base column.
        b: The exponent column. Must be the same length as `a`.

    Parameters:
        dt: The dtype.

    Returns:
        A column of powers of the same dtype, null wherever either input is
        null.

    Raises:
        Error: If the dtype is a signed integer and any exponent is negative,
            which has no answer in the integers and is the error numpy raises
            rather than a zero. Otherwise only what the morsel runtime raises.
    """
    return _arith[dt, OP_POW](a, b)


def _powers[
    dt: DType, width: Int
](x: SIMD[dt, width], y: SIMD[dt, width]) -> SIMD[dt, width]:
    """Raises a register of floats to a register of powers, one lane at a time.

    This is the only loop body in the file that is not a vector instruction, and
    it is worth saying why. Mojo's `**` on a register of float64 is a fast
    approximation: it answers `1.4142135623734946` for the square root of two,
    where the correctly rounded answer is `1.4142135623730951`, so it is wrong
    by about six hundred of the last bits. That is invisible in a plot and very
    visible in a test that compares a column against pandas, and a difference
    like that registered as a divergence would be a bug wearing a decision's
    clothing.

    So the answer comes from the same `pow` in the C library that NumPy calls,
    which makes the result identical rather than close. The cost is a call per
    element rather than an instruction per register, which is the cost NumPy
    pays for the same reason.

    Half precision goes through the single precision call, which is what NumPy
    does with a half as well.

    Args:
        x: The bases.
        y: The exponents.

    Parameters:
        dt: A floating point dtype.
        width: How many lanes.

    Returns:
        A register of powers.
    """
    var out = SIMD[dt, width]()
    comptime if dt == DType.float64:
        for lane in range(width):
            out[lane] = external_call["pow", Float64](
                x[lane].cast[DType.float64](), y[lane].cast[DType.float64]()
            ).cast[dt]()
    else:
        for lane in range(width):
            out[lane] = external_call["powf", Float32](
                x[lane].cast[DType.float32](), y[lane].cast[DType.float32]()
            ).cast[dt]()
    return out


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
        op: One of the six operation codes. Floor division and the remainder
            reach here on a float dtype only; the integer forms need the
            validity and go through `_int_divide`.

    Returns:
        A column of results, null wherever either input is null.

    Raises:
        Error: If the operation is a power on a signed integer dtype and any
            exponent is negative. Otherwise only what the morsel runtime
            raises.
    """
    comptime width = simd_width_of[dt]()

    var n = len(a)
    # The loop below writes a result for every row, the null ones included,
    # which is the whole point of computing first and repairing afterwards. So
    # the allocation does not need the pass that zeroes it.
    var out = Array[dt](overwritten=n)
    var validity = combined_validity(a.data.validity, b.data.validity)

    def compute(start: Int, stop: Int) raises {mut out, imm}:
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
            elif op == OP_MUL:
                dst.unsafe_offset(i).unsafe_store(x * y)
            elif op == OP_FLOORDIV:
                dst.unsafe_offset(i).unsafe_store(x // y)
            elif op == OP_MOD:
                dst.unsafe_offset(i).unsafe_store(x % y)
            else:
                comptime if dt.is_integral():
                    comptime if dt.is_signed():
                        # The padding past the last row is zero and a null holds
                        # a zero, so neither can make this fire.
                        if y.lt(SIMD[dt, width](0)).reduce_or():
                            raise Error(
                                "power: integers to negative integer powers are"
                                " not allowed"
                            )
                    dst.unsafe_offset(i).unsafe_store(x**y)
                else:
                    dst.unsafe_offset(i).unsafe_store(_powers(x, y))
            i += width

        # These rows are in this core's cache right now, so the repair is
        # nearly free here and is a second walk over the column anywhere else.
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def _int_divide[
    dt: DType, op: Int
](a: Array[dt], b: Array[dt]) raises -> Array[dt]:
    """Floor divides or takes a remainder on integers, nulling the zero rows.

    This is the only loop in the file whose answer depends on a value rather
    than only on the types, and it is why the shape differs from `_arith`. The
    validity is not a copy of the inputs' any more: a row whose divisor is zero
    has no integer answer and comes out null, so the bitmap is written inside
    the loop and not just read at the end of it.

    Writing it there is safe because a morsel boundary is a multiple of 64
    rows, so two workers never reach for the same byte of the bitmap. It is
    also where the repair wants it: `repair_range` runs at the end of the same
    morsel and zeroes the values under the bits this just cleared, so the
    garbage the division left in those rows never leaves the cache line it was
    written on.

    The hardware is not the reason for any of this. Mojo's integer division
    already answers zero for a zero divisor rather than trapping, and already
    answers the wrapped value for the most negative integer divided by minus
    one, so there is no undefined behaviour to steer around and the loop stays
    branch free apart from the reduce.

    Args:
        a: The numerator column.
        b: The denominator column. Must be the same length as `a`.

    Parameters:
        dt: An integer dtype.
        op: `OP_FLOORDIV` or `OP_MOD`.

    Returns:
        A column of results, null wherever either input is null or the divisor
        is zero.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[dt]()

    var n = len(a)
    # Every row is written below, the ones about to be nulled included, so the
    # zeroing constructor would be a pass thrown away.
    var out = Array[dt](overwritten=n)
    var validity = combined_validity(a.data.validity, b.data.validity)

    def compute(start: Int, stop: Int) {mut out, mut validity, imm}:
        var lhs = a.unsafe_ptr()
        var rhs = b.unsafe_ptr()
        var dst = out.unsafe_ptr()
        var zeros = SIMD[dt, width](0)
        var i = start
        while i < stop:
            var x = lhs.unsafe_offset(i).unsafe_load[width=width]()
            var y = rhs.unsafe_offset(i).unsafe_load[width=width]()

            comptime if op == OP_FLOORDIV:
                dst.unsafe_offset(i).unsafe_store(x // y)
            else:
                dst.unsafe_offset(i).unsafe_store(x % y)

            # A zero divisor is rare, so the whole register is tested at once
            # and the walk that clears the bits only runs on a register that
            # holds one. The last register of the column reads into padding,
            # which is zero and would fire this, so the walk stops at `stop`.
            if y.eq(zeros).reduce_or():
                var last = min(i + width, stop)
                for k in range(i, last):
                    if (
                        rhs.unsafe_offset(k)
                        .unsafe_load[width=1]()
                        .eq(Scalar[dt](0))
                    ):
                        validity.set(k, False)
            i += width

        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
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
    var validity = combined_validity(a.data.validity, b.data.validity)

    def compute(start: Int, stop: Int) raises {mut out, imm}:
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

        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def arith_const[
    dt: DType, op: Int
](a: Array[dt], b: Scalar[dt], flip: Bool = False) raises -> Array[dt]:
    """Applies an arithmetic operation between a column and one constant.

    Addition and multiplication do not need entry points of their own here the
    way the column forms do, because the caller that reaches this already holds
    the operation as a code rather than as a name. It arrived from a plan. Floor
    division, the remainder and the power do have wrappers, because those three
    have to decide between this loop and a guarded one first.

    `flip` puts the constant on the left, which changes nothing for addition and
    multiplication and everything for the other four. The branch on it is
    outside the loop rather than a parameter, so there is one instantiation of
    this per dtype rather than two, and the loop the processor runs is still
    straight line. Compile time is a real budget here: this is instantiated once
    per dtype in the erased dispatch, and doubling that doubles the cost of a
    file nothing has even called yet.

    Args:
        a: The column.
        b: The constant.
        flip: True if the constant is the left operand.

    Parameters:
        dt: The dtype. The constant is already at it; promotion happened above.
        op: One of the six operation codes. Floor division and the remainder
            reach here on a float dtype only.

    Returns:
        A column of results, null wherever the column is null. A constant is
        never null here; a null constant makes the whole answer null and is
        handled before any loop runs.

    Raises:
        Error: If the operation is a power on a signed integer dtype and the
            exponent, whichever side it is on, is negative. Otherwise only what
            the morsel runtime raises.
    """
    comptime width = simd_width_of[dt]()

    comptime if op == OP_POW and dt.is_integral() and dt.is_signed():
        # The constant is the exponent unless it is flipped, and a constant
        # exponent is one test rather than one per register.
        if not flip and Bool(b.lt(Scalar[dt](0))):
            raise Error(
                "power: integers to negative integer powers are not allowed"
            )

    var n = len(a)
    # Every row is written below, so the zeroing allocation is a wasted pass.
    var out = Array[dt](overwritten=n)
    var validity = Bitmap(copy=a.data.validity)
    var y = SIMD[dt, width](b)

    # The branch on `flip` stays outside the inner loop, one test per morsel
    # rather than one per register, which is what it was when there was a
    # single loop over the whole column.
    def compute(start: Int, stop: Int) raises {mut out, imm}:
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
        elif op == OP_SUB:
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
        elif op == OP_FLOORDIV:
            if flip:
                var i = start
                while i < stop:
                    var x = src.unsafe_offset(i).unsafe_load[width=width]()
                    dst.unsafe_offset(i).unsafe_store(y // x)
                    i += width
            else:
                var i = start
                while i < stop:
                    var x = src.unsafe_offset(i).unsafe_load[width=width]()
                    dst.unsafe_offset(i).unsafe_store(x // y)
                    i += width
        elif op == OP_MOD:
            if flip:
                var i = start
                while i < stop:
                    var x = src.unsafe_offset(i).unsafe_load[width=width]()
                    dst.unsafe_offset(i).unsafe_store(y % x)
                    i += width
            else:
                var i = start
                while i < stop:
                    var x = src.unsafe_offset(i).unsafe_load[width=width]()
                    dst.unsafe_offset(i).unsafe_store(x % y)
                    i += width
        else:
            if flip:
                var i = start
                while i < stop:
                    var x = src.unsafe_offset(i).unsafe_load[width=width]()
                    comptime if dt.is_integral():
                        comptime if dt.is_signed():
                            # Flipped, the column is the exponent, so this is
                            # the one shape of power whose check cannot be
                            # hoisted above the loop.
                            if x.lt(SIMD[dt, width](0)).reduce_or():
                                raise Error(
                                    "power: integers to negative integer powers"
                                    " are not allowed"
                                )
                        dst.unsafe_offset(i).unsafe_store(y**x)
                    else:
                        dst.unsafe_offset(i).unsafe_store(_powers(y, x))
                    i += width
            else:
                var i = start
                while i < stop:
                    var x = src.unsafe_offset(i).unsafe_load[width=width]()
                    comptime if dt.is_integral():
                        dst.unsafe_offset(i).unsafe_store(x**y)
                    else:
                        dst.unsafe_offset(i).unsafe_store(_powers(x, y))
                    i += width

        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def floor_divide_const[
    dt: DType
](a: Array[dt], b: Scalar[dt], flip: Bool = False) raises -> Array[dt]:
    """Floor divides a column by a constant, or a constant by a column.

    Args:
        a: The column.
        b: The constant.
        flip: True if the constant is the numerator.

    Parameters:
        dt: The dtype.

    Returns:
        A column of quotients of the same dtype, null wherever the column is
        null and, on an integer dtype, wherever the divisor is zero.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime if dt.is_integral():
        return _int_divide_const[dt, OP_FLOORDIV](a, b, flip)
    else:
        return arith_const[dt, OP_FLOORDIV](a, b, flip)


def modulo_const[
    dt: DType
](a: Array[dt], b: Scalar[dt], flip: Bool = False) raises -> Array[dt]:
    """Takes the remainder of a column and a constant, either way round.

    Args:
        a: The column.
        b: The constant.
        flip: True if the constant is the numerator.

    Parameters:
        dt: The dtype.

    Returns:
        A column of remainders of the same dtype, null wherever the column is
        null and, on an integer dtype, wherever the divisor is zero.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime if dt.is_integral():
        return _int_divide_const[dt, OP_MOD](a, b, flip)
    else:
        return arith_const[dt, OP_MOD](a, b, flip)


def power_const[
    dt: DType
](a: Array[dt], b: Scalar[dt], flip: Bool = False) raises -> Array[dt]:
    """Raises a column to a constant power, or a constant to a column of them.

    Args:
        a: The column.
        b: The constant.
        flip: True if the constant is the base.

    Parameters:
        dt: The dtype.

    Returns:
        A column of powers of the same dtype, null wherever the column is null.

    Raises:
        Error: If the dtype is a signed integer and the exponent is negative,
            whichever side it is on. Otherwise only what the morsel runtime
            raises.
    """
    return arith_const[dt, OP_POW](a, b, flip)


def _int_divide_const[
    dt: DType, op: Int
](a: Array[dt], b: Scalar[dt], flip: Bool = False) raises -> Array[dt]:
    """Floor divides or takes a remainder on integers against one constant.

    Which side the constant is on decides how much work there is. With the
    column on top the divisor is the constant, so whether any row is null is
    one test above the loop and the loop itself has nothing to check. Flipped,
    the divisor is the column and the check is the same per register reduce the
    two column form does.

    Args:
        a: The column.
        b: The constant.
        flip: True if the constant is the numerator.

    Parameters:
        dt: An integer dtype.
        op: `OP_FLOORDIV` or `OP_MOD`.

    Returns:
        A column of results, null wherever the column is null or the divisor is
        zero.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[dt]()

    var n = len(a)
    var zero = Scalar[dt](0)

    # A constant zero divisor makes every row null whatever is in the column,
    # so there is nothing for a loop to compute. The allocation is the zeroing
    # one here because nothing is going to write over it.
    if not flip and Bool(b.eq(zero)):
        var blank = Array[dt](n)
        blank.data.validity = Bitmap(n, all_valid=False)
        return blank^

    # Every row is written below, so the zeroing allocation is a wasted pass.
    var out = Array[dt](overwritten=n)
    var validity = Bitmap(copy=a.data.validity)
    var y = SIMD[dt, width](b)
    var zeros = SIMD[dt, width](0)

    def compute(start: Int, stop: Int) {mut out, mut validity, imm}:
        var src = a.unsafe_ptr()
        var dst = out.unsafe_ptr()
        if flip:
            var i = start
            while i < stop:
                var x = src.unsafe_offset(i).unsafe_load[width=width]()
                comptime if op == OP_FLOORDIV:
                    dst.unsafe_offset(i).unsafe_store(y // x)
                else:
                    dst.unsafe_offset(i).unsafe_store(y % x)

                # Same reduce and same walk as the two column form, and the
                # walk stops at `stop` for the same reason: the padding past
                # the last row is zero and would otherwise look like a divisor.
                if x.eq(zeros).reduce_or():
                    var last = min(i + width, stop)
                    for k in range(i, last):
                        if src.unsafe_offset(k).unsafe_load[width=1]().eq(zero):
                            validity.set(k, False)
                i += width
        else:
            var i = start
            while i < stop:
                var x = src.unsafe_offset(i).unsafe_load[width=width]()
                comptime if op == OP_FLOORDIV:
                    dst.unsafe_offset(i).unsafe_store(x // y)
                else:
                    dst.unsafe_offset(i).unsafe_store(x % y)
                i += width

        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
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
    var validity = Bitmap(copy=a.data.validity)
    var y = SIMD[DType.float64, width](b.cast[DType.float64]())

    def compute(start: Int, stop: Int) raises {mut out, imm}:
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

        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^
