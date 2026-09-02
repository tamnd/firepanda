"""Elementwise arithmetic and comparison on two columns whose dtypes are values.

`arith.mojo` and `compare.mojo` hold the loops, one per dtype, and both take the
dtype as a parameter. A frame does not have one: at the frame boundary a column
is an `AnyArray` and its dtype is a field. This is the boundary crossing, and it
is the only file that has to know that the ten operations are two families.

Three things happen before a loop runs. The two operand types are promoted to a
common type, by the same `promote` a concat and a coalesce use, so int32 with
float32 is float64 here for the reason it is float64 there. Both sides are then
converted to that type, which is a copy for the side that moves and nothing for
the side already there. Only then is the dtype resolved to a parameter, once,
and the typed kernel called.

The promotion is what makes the operation's answer depend on the types rather
than on the values. int64 with uint64 is float64 and is lossy above 2^53, which
is what NumPy answers and is wrong in the same way NumPy is wrong. Being wrong
in a way people already expect is worth more here than being right in a way that
makes an expression's type depend on what is in the column.

Division is not part of the arithmetic family. It always answers float64,
whatever went in, which is what `/` does in pandas, so it promotes for the sake
of checking that the operands are numbers and then ignores the answer.

Two things are deliberately absent. There is no comparison on text, because the
loops underneath are over fixed width registers and a string comparison is a
byte loop against a second column's bytes, which is a kernel rather than an arm
of a dispatch. And there is no constant on either side, so `x > 5` cannot be
written yet, only `x > y`. Both want the same missing piece, which is a value
type that carries its own dtype, and that piece is also what the series level
reductions are waiting on. It is one change and it should be one change rather
than three partial ones.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import LogicalType, promote

from .arith import add, divide, multiply, subtract
from .cast import cast_any
from .compare import (
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
    not_equal,
)


struct BinaryOp(Equatable, ImplicitlyCopyable, Movable, Writable):
    """One of the ten elementwise operations over a pair of columns."""

    var code: UInt8
    """The operation, as a small integer."""

    comptime ADD = Self(0)
    """Addition."""

    comptime SUB = Self(1)
    """Subtraction."""

    comptime MUL = Self(2)
    """Multiplication."""

    comptime DIV = Self(3)
    """Division, which always answers float64."""

    comptime EQ = Self(4)
    """Equality."""

    comptime NE = Self(5)
    """Inequality."""

    comptime LT = Self(6)
    """Less than."""

    comptime LE = Self(7)
    """Less than or equal."""

    comptime GT = Self(8)
    """Greater than."""

    comptime GE = Self(9)
    """Greater than or equal."""

    def __init__(out self, code: UInt8):
        """Constructs an operation from its code.

        Args:
            code: The operation.
        """
        self.code = code

    def __eq__(self, other: Self) -> Bool:
        """Compares two operations.

        Args:
            other: The operation to compare against.

        Returns:
            True if they are the same operation.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two operations.

        Args:
            other: The operation to compare against.

        Returns:
            True if they are different operations.
        """
        return self.code != other.code

    def is_comparison(self) -> Bool:
        """Reports whether the operation answers a bool column.

        Returns:
            True for the six comparisons, false for the four arithmetic ones.
        """
        return self.code >= Self.EQ.code

    def write_to(self, mut writer: Some[Writer]):
        """Writes the operation as the symbol it is spelled with.

        Args:
            writer: The destination.
        """
        if self == Self.ADD:
            writer.write("+")
        elif self == Self.SUB:
            writer.write("-")
        elif self == Self.MUL:
            writer.write("*")
        elif self == Self.DIV:
            writer.write("/")
        elif self == Self.EQ:
            writer.write("==")
        elif self == Self.NE:
            writer.write("!=")
        elif self == Self.LT:
            writer.write("<")
        elif self == Self.LE:
            writer.write("<=")
        elif self == Self.GT:
            writer.write(">")
        else:
            writer.write(">=")


def binary_type(
    op: BinaryOp, a: LogicalType, b: LogicalType
) raises -> LogicalType:
    """Returns the type an operation answers, without touching a value.

    A plan needs this before any row moves, so it is a function of the two
    operand types and the operation and nothing else. That is the same property
    the promotion has, and it is why an expression's type can be known at plan
    time even though the dtypes are values.

    Args:
        op: The operation.
        a: The left operand type.
        b: The right operand type.

    Returns:
        The result type: bool for a comparison, float64 for a division, and the
        promoted operand type for the other three.

    Raises:
        If the two types have no common type, or the operation is arithmetic and
        the common type is not a number.
    """
    var common = promote(a, b)
    if op.is_comparison():
        if common.is_variable_width():
            raise Error(
                "binary: "
                + String(op)
                + " on text is not implemented, only on numbers"
            )
        return LogicalType.BOOL
    if not common.is_numeric():
        raise Error(
            "binary: " + String(op) + " is not defined on " + String(common)
        )
    if op == BinaryOp.DIV:
        return LogicalType.FLOAT64
    return common


def binary_any(a: AnyArray, b: AnyArray, op: BinaryOp) raises -> AnyArray:
    """Applies an operation elementwise to two columns of the same length.

    Args:
        a: The left column.
        b: The right column.
        op: The operation.

    Returns:
        A column of `binary_type(op, a.type, b.type)`, null wherever either
        input is null.

    Raises:
        If the columns are different lengths, if the types have no common type,
        or if the operation is not defined on that common type.
    """
    if len(a) != len(b):
        raise Error(
            "binary: the left column has "
            + String(len(a))
            + " rows and the right has "
            + String(len(b))
        )
    # Checked first and for its own sake. Everything below assumes the pair has
    # a common type and that the operation is defined on it.
    _ = binary_type(op, a.type, b.type)

    # The common type is the promotion in all four shapes, including division:
    # the divide loop reads its operands at their own width and answers float64
    # whatever they were, so converting them to float64 first would be two
    # copies of the column to reach the same answer.
    var common = promote(a.type, b.type)
    var left = AnyArray(copy=a) if a.type == common else cast_any(
        a, common.physical
    )
    var right = AnyArray(copy=b) if b.type == common else cast_any(
        b, common.physical
    )
    return _binary_erased(left, right, op, common.physical)


def _binary_erased(
    a: AnyArray, b: AnyArray, op: BinaryOp, dt: DType
) raises -> AnyArray:
    """Resolves the dtype to a parameter and calls the typed kernel.

    Both columns are already of `dt` by the time this runs, so there is one
    dispatch rather than the two a naive erasure would do.

    Args:
        a: The left column, of dtype `dt`.
        b: The right column, of dtype `dt`.
        op: The operation.
        dt: The dtype both columns share.

    Returns:
        The result column.

    Raises:
        If the dtype has no physical layout, or the operation is arithmetic on
        a dtype the arithmetic loops do not cover.
    """
    comptime for target in ALL:
        if dt == target:
            ref x = a.as_typed_view[target]()
            ref y = b.as_typed_view[target]()
            if op == BinaryOp.EQ:
                return AnyArray(equal(x, y))
            if op == BinaryOp.NE:
                return AnyArray(not_equal(x, y))
            if op == BinaryOp.LT:
                return AnyArray(less(x, y))
            if op == BinaryOp.LE:
                return AnyArray(less_equal(x, y))
            if op == BinaryOp.GT:
                return AnyArray(greater(x, y))
            if op == BinaryOp.GE:
                return AnyArray(greater_equal(x, y))
            comptime if target == DType.bool:
                raise Error(
                    "binary: " + String(op) + " is not defined on bool columns"
                )
            else:
                if op == BinaryOp.ADD:
                    return AnyArray(add(x, y))
                if op == BinaryOp.SUB:
                    return AnyArray(subtract(x, y))
                if op == BinaryOp.MUL:
                    return AnyArray(multiply(x, y))
                return AnyArray(divide(x, y))
    raise Error("binary: unsupported dtype")
