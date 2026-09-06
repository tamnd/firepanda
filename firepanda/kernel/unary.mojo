"""Elementwise operations that read one column and write one.

There are four of them at the Python level, `-x`, `+x`, `abs(x)` and `~x`, and
three loops down here, because unary plus is the column unchanged and is
answered by a copy rather than by a pass over the values.

The same shape as `arith.mojo`: compute across the whole values buffer including
the null rows and repair them afterwards, because a branch per row costs more
than the arithmetic it skips. The validity a unary operation produces is the
validity it was handed, so unlike floor division there is nothing here that can
change which rows are present.

What pandas does with these is not what numpy does with them, and the difference
is all about bool. Every rule below was measured against a running pandas 3.0
rather than read out of its documentation.

`-x` on a bool column is the logical not, so `-Series([True, False])` is
`[False, True]`. numpy refuses the same expression outright and says the boolean
negative is not supported. pandas catches that and sends it to the inversion, so
firepanda does too.

`abs(x)` on a bool column is the column unchanged, which follows from the values
already being zero and one and is worth stating because the loop has to be told
not to try a signed absolute on a byte holding a boolean.

`~x` is two different operations picked by the type. On a bool column it is the
logical not and on an integer column it is the bitwise not, so `~Series([1, 2])`
is `[-2, -3]` and not `[False, False]`. On a float column it raises, because
there is no bitwise not of a float and numpy says so first.

Two of the three wrap rather than raising, which is what the hardware does and
what pandas passes on. Negating the most negative int8 answers the most negative
int8 again, taking the absolute value of it does the same, and negating an
unsigned one answers the wrapped complement, so `-Series([1], dtype='uint8')` is
`[255]`. All three were measured. A checked version of any of them would be a
divergence rather than a fix.
"""

from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import LogicalType
from firepanda.exec import parallel_morsels

from .mask import repair_range

comptime OP_NEG = 0
"""Operation code for the negation."""

comptime OP_ABS = 1
"""Operation code for the absolute value."""

comptime OP_INVERT = 2
"""Operation code for the bitwise or logical not."""


@fieldwise_init
struct UnaryOp(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which operation a column is being put through.

    Held as a code rather than as a function so that the erased entry point can
    take it as an ordinary argument and the typed loops can take it as a
    parameter the compiler folds away.
    """

    var code: Int
    """The operation, as one of the four values below."""

    comptime NEG = Self(0)
    """Unary minus. The logical not on a bool column."""

    comptime POS = Self(1)
    """Unary plus. The column unchanged, on every type that has it at all."""

    comptime ABS = Self(2)
    """The absolute value. The column unchanged on an unsigned or bool column."""

    comptime INVERT = Self(3)
    """The bitwise not, and the logical not on a bool column."""

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

    def write_to(self, mut writer: Some[Writer]):
        """Writes the operation as the symbol it is written with.

        Args:
            writer: Where to write.
        """
        if self == Self.NEG:
            writer.write("-")
        elif self == Self.POS:
            writer.write("+")
        elif self == Self.ABS:
            writer.write("abs")
        else:
            writer.write("~")


def unary_type(op: UnaryOp, t: LogicalType) raises -> LogicalType:
    """Returns the type an operation answers, without touching a value.

    All four keep the operand type, so this is a check rather than a
    computation. It exists anyway, because a plan needs to be told that `~` on a
    float column has no answer before it starts reading rows, and because the
    binary side has the same function and the two are read together.

    Args:
        op: The operation.
        t: The operand type.

    Returns:
        The operand type.

    Raises:
        If the type is not a number or a bool, or the operation is an inversion
        of a floating point column.
    """
    if not t.is_numeric() and t != LogicalType.BOOL:
        raise Error("unary: " + String(op) + " is not defined on " + String(t))
    if op == UnaryOp.INVERT and t.physical.is_floating_point():
        raise Error("unary: ~ is not defined on " + String(t))
    return t


def negate[dt: DType](a: Array[dt]) raises -> Array[dt]:
    """Negates a column elementwise.

    On a bool column this is the logical not, which is pandas rather than numpy
    and is explained at the top of the file.

    Args:
        a: The column.

    Parameters:
        dt: The dtype.

    Returns:
        A column of negated values, null wherever the input is null.
    """
    comptime if dt == DType.bool:
        return _unary[dt, OP_INVERT](a)
    else:
        return _unary[dt, OP_NEG](a)


def absolute[dt: DType](a: Array[dt]) raises -> Array[dt]:
    """Takes the absolute value of a column elementwise.

    On an unsigned or a bool column every value is already its own absolute
    value, so the column is copied rather than walked.

    Args:
        a: The column.

    Parameters:
        dt: The dtype.

    Returns:
        A column of absolute values, null wherever the input is null.
    """
    comptime if dt == DType.bool or not dt.is_signed():
        return Array[dt](copy=a)
    else:
        return _unary[dt, OP_ABS](a)


def invert[dt: DType](a: Array[dt]) raises -> Array[dt]:
    """Inverts a column elementwise.

    The bitwise not on an integer column and the logical not on a bool one,
    which is one operator in pandas and two instructions here.

    Args:
        a: The column.

    Parameters:
        dt: The dtype. Must not be a floating point one.

    Returns:
        A column of inverted values, null wherever the input is null.

    Raises:
        Error: If the dtype is a floating point one, which has no bitwise not.
    """
    comptime if dt.is_floating_point():
        raise Error("unary: ~ is not defined on a floating point column")
    else:
        return _unary[dt, OP_INVERT](a)


def _unary[dt: DType, op: Int](a: Array[dt]) raises -> Array[dt]:
    """Applies a unary operation elementwise.

    The tail past the last full register is handled by loading a full register
    anyway. The values buffer is padded to a 64-byte multiple and the padding is
    zero, so the read is in bounds and the lanes past the length are written back
    into padding nobody reads.

    Args:
        a: The column.

    Parameters:
        dt: The dtype.
        op: One of the three operation codes.

    Returns:
        A column of results, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[dt]()

    var n = len(a)
    # A result is written for every row, the null ones included, so the
    # allocation does not need the pass that zeroes it.
    var out = Array[dt](overwritten=n)
    var validity = Bitmap(copy=a.data.validity)

    def compute(start: Int, stop: Int) raises {mut out, imm}:
        var src = a.unsafe_ptr()
        var dst = out.unsafe_ptr()
        var i = start
        while i < stop:
            var x = src.unsafe_offset(i).unsafe_load[width=width]()

            comptime if op == OP_NEG:
                dst.unsafe_offset(i).unsafe_store(-x)
            elif op == OP_ABS:
                dst.unsafe_offset(i).unsafe_store(abs(x))
            else:
                # One instruction for both meanings of `~`. A bool lane is one
                # bit wide, so the bitwise not of it is the logical not, and
                # writing the bool case as a compare against False instead
                # would not even type check: the compare answers a bool
                # register and the store wants a register of `dt`, which the
                # checker will not read as the same thing inside a branch it
                # has not folded yet.
                dst.unsafe_offset(i).unsafe_store(~x)
            i += width

        # These rows are in this core's cache right now, so the repair is
        # nearly free here and is a second walk over the column anywhere else.
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)
    out.data.validity = validity^
    return out^


def unary_any(a: AnyArray, op: UnaryOp) raises -> AnyArray:
    """Applies a unary operation to a column whose dtype is a value.

    Args:
        a: The column.
        op: The operation.

    Returns:
        A column of the same type, null wherever the input is null.

    Raises:
        If the operation is not defined on the column's type, or the type has no
        physical layout.
    """
    # Checked first and for its own sake, so that a column of a type the
    # operation has no answer for is turned away in the same words whether or
    # not it happens to have a physical layout.
    _ = unary_type(op, a.type)

    if op == UnaryOp.POS:
        return AnyArray(copy=a)

    comptime for target in ALL:
        if a.type.physical == target:
            ref x = a.as_typed_view[target]()
            if op == UnaryOp.NEG:
                return AnyArray(negate(x))
            if op == UnaryOp.ABS:
                return AnyArray(absolute(x))
            return AnyArray(invert(x))
    raise Error("unary: unsupported dtype")
