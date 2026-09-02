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

A constant on either side goes through `binary_value_any`, which is the same
three steps with a `Value` where the second column would be. It promotes against
the constant's type rather than against its value, so `x + 1` on an int32 column
stays int32 and does not widen because the literal happened to arrive as an
int64. The column is the only thing that gets converted; the constant is read at
the common type when the dtype is resolved, which costs nothing because it is one
element.

A null constant is answered before any loop runs. Every row of the result is
null, whatever the operation and whatever is in the column, so the loop would be
a pass over the column to write zeros it already holds.

One thing is still deliberately absent, and that is comparison on text. The loops
underneath are over fixed width registers and a string comparison is a byte loop,
which is a kernel rather than an arm of a dispatch.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.value import Value
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import LogicalType, promote

from .arith import (
    OP_ADD,
    OP_MUL,
    OP_SUB,
    add,
    arith_const,
    divide,
    divide_const,
    multiply,
    subtract,
)
from .cast import cast_any
from .compare import (
    CMP_EQ,
    CMP_GE,
    CMP_GT,
    CMP_LE,
    CMP_LT,
    CMP_NE,
    compare_const,
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

    def mirrored(self) -> Self:
        """Returns the operation that means the same thing with the sides swapped.

        `5 < x` and `x > 5` are the same question, so a constant on the left of a
        comparison is handled by turning it round rather than by a second set of
        loops. The swap is exact for floats too: both readings are false when
        either side is a NaN.

        Equality and inequality are their own mirrors, and so is every
        arithmetic operation as far as this is concerned, because subtraction and
        division take a flag instead. Only the four ordered comparisons change.

        Returns:
            The mirrored operation.
        """
        if self == Self.LT:
            return Self.GT
        if self == Self.LE:
            return Self.GE
        if self == Self.GT:
            return Self.LT
        if self == Self.GE:
            return Self.LE
        return self

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


def binary_value_any(
    a: AnyArray, b: Value, op: BinaryOp, value_on_left: Bool = False
) raises -> AnyArray:
    """Applies an operation elementwise between a column and one constant.

    Args:
        a: The column.
        b: The constant.
        op: The operation.
        value_on_left: True for `5 - x` rather than `x - 5`.

    Returns:
        A column of `binary_type(op, a.type, b.type)` with as many rows as `a`,
        null wherever `a` is null and null everywhere if the constant is null.

    Raises:
        If the two types have no common type, or the operation is not defined on
        that common type.
    """
    var left = a.type if not value_on_left else b.type
    var right = b.type if not value_on_left else a.type
    var answer = binary_type(op, left, right)

    if b.is_null():
        return _all_null(answer, len(a))

    var common = promote(a.type, b.type)
    var column = AnyArray(copy=a) if a.type == common else cast_any(
        a, common.physical
    )
    # The comparison with the constant on the left is turned round rather than
    # given loops of its own. Subtraction and division cannot be turned round, so
    # they carry the flag into the loop, where the branch on it sits outside.
    var applied = op.mirrored() if value_on_left and op.is_comparison() else op
    var flip = value_on_left and not op.is_comparison()
    return _binary_const_erased(column, b, applied, common.physical, flip)


def _all_null(type: LogicalType, rows: Int) raises -> AnyArray:
    """Builds a column of a given type with every row missing.

    The values buffer starts zeroed and a null holds a zero, so this only has to
    install a bitmap. Running the repair pass over it would be a write of the
    zeros that are already there.

    Args:
        type: The column's type.
        rows: How many rows.

    Returns:
        A column of `rows` nulls.

    Raises:
        If the type has no physical layout.
    """
    comptime for target in ALL:
        if type.physical == target:
            var out = Array[target](rows)
            out.data.validity = Bitmap(rows, all_valid=False)
            return AnyArray(out^)
    raise Error("binary: unsupported dtype")


def _binary_const_erased(
    a: AnyArray, b: Value, op: BinaryOp, dt: DType, flip: Bool
) raises -> AnyArray:
    """Resolves the dtype to a parameter and calls the typed constant kernel.

    The column is already of `dt`. The constant is not converted before this
    runs, because reading it at `dt` is one instruction and doing it here means
    the conversion happens exactly where the dtype is known.

    Args:
        a: The column, of dtype `dt`.
        b: The constant, present and convertible to `dt`.
        op: The operation, already mirrored if the constant was on the left.
        dt: The column's dtype.
        flip: True if the constant is the left operand of a subtraction or a
            division.

    Returns:
        The result column.

    Raises:
        If the dtype has no physical layout, or the operation is arithmetic on a
        dtype the arithmetic loops do not cover.
    """
    comptime for target in ALL:
        if dt == target:
            ref x = a.as_typed_view[target]()
            var y = b.as_scalar[target]()
            if op == BinaryOp.EQ:
                return AnyArray(compare_const[target, CMP_EQ](x, y))
            if op == BinaryOp.NE:
                return AnyArray(compare_const[target, CMP_NE](x, y))
            if op == BinaryOp.LT:
                return AnyArray(compare_const[target, CMP_LT](x, y))
            if op == BinaryOp.LE:
                return AnyArray(compare_const[target, CMP_LE](x, y))
            if op == BinaryOp.GT:
                return AnyArray(compare_const[target, CMP_GT](x, y))
            if op == BinaryOp.GE:
                return AnyArray(compare_const[target, CMP_GE](x, y))
            comptime if target == DType.bool:
                raise Error(
                    "binary: " + String(op) + " is not defined on bool columns"
                )
            else:
                if op == BinaryOp.ADD:
                    return AnyArray(arith_const[target, OP_ADD](x, y))
                if op == BinaryOp.SUB:
                    return AnyArray(arith_const[target, OP_SUB](x, y, flip))
                if op == BinaryOp.MUL:
                    return AnyArray(arith_const[target, OP_MUL](x, y))
                return AnyArray(divide_const[target](x, y, flip))
    raise Error("binary: unsupported dtype")
