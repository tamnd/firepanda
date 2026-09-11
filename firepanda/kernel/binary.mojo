"""Elementwise arithmetic and comparison on two columns whose dtypes are values.

`arith.mojo` and `compare.mojo` hold the loops, one per dtype, and both take the
dtype as a parameter. A frame does not have one: at the frame boundary a column
is an `AnyArray` and its dtype is a field. This is the boundary crossing, and it
is the only file that has to know that the thirteen operations are two families.

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

Division is the one operation whose answer is not the type it was given, and
only when it was given integers. Two integers have no integer quotient so the
answer is float64, and a float divided by anything it promotes against keeps the
promoted width, so `float32 / float32` is float32. That is what `/` does in
pandas. It means division uses the promotion rather than ignoring it, and the
one thing it adds is the widening of the integer case.

Floor division and the remainder look like division and are not. `//` and `%`
keep the operand type in pandas, so they promote like addition does and an
integer column stays an integer column. What they do about a zero divisor is
`arith.mojo`'s subject and is the one place in the file where firepanda answers
something pandas does not.

A constant on either side goes through `binary_value_any`, which is the same
three steps with a `Value` where the second column would be, plus one step in
front of them. A constant that came from Python has no width of its own, because
a Python `2` does not have one, so it takes the column's: `x + 1` on an int32
column stays int32 rather than widening because the literal arrived as an int64.
`resolve_constant` is that step and the rule it applies is numpy's weak scalar
rule, which pandas inherits whole. After it, the constant has a width and the
three steps are the ones a second column would go through. The column is the only
thing that gets converted; the constant is read at the common type when the dtype
is resolved, which costs nothing because it is one element.

A null constant is answered before any loop runs. Every row of the result is
null, whatever the operation and whatever is in the column, so the loop would be
a pass over the column to write zeros it already holds.

Comparison on text does not go through the dtype dispatch at all. A text column's
physical dtype is uint8 and the loop over it would read a view byte as a value,
so the variable width case is answered before the dispatch is reached, by the
kernels in `text.mojo`. Arithmetic on text is still an error, and it is the same
error it was: there is no common type between a string and a number, and adding
two strings is a concatenation, which is a function rather than an operator here.

Comparison on a category column is answered before the dispatch too, and for a
stronger reason than text. `promote` refuses every mixture involving a dictionary
type, because what two categoricals combine to depends on their categories and
the categories are held by the column rather than by the type. That refusal is
right and a comparison does not need it: comparing codes to a code answers the
question without promoting anything, and it is cheaper than the text comparison
the decoded column would do. The six rules pandas has for it are in
`_dictionary_erased` and `_dictionary_const_erased`.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.array.value import Value
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import LogicalType, TypeKind, promote
from firepanda.dtype.temporal import TimeUnit, TimeZone, finer_unit
from firepanda.kernel.dictionary import (
    decode_dictionary,
    dictionary_codes,
    same_categories,
)
from firepanda.kernel.parse_time import parse_instant

from .arith import (
    OP_ADD,
    OP_MUL,
    OP_SUB,
    add,
    arith_const,
    divide,
    divide_const,
    divide_float,
    divide_float_const,
    floor_divide,
    floor_divide_const,
    logical_and,
    logical_and_const,
    logical_or,
    logical_or_const,
    modulo,
    modulo_const,
    multiply,
    power,
    power_const,
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
from .temporal import temporal_as_unit
from .text import compare_text, compare_text_const


struct BinaryOp(Equatable, ImplicitlyCopyable, Movable, Writable):
    """One of the thirteen elementwise operations over a pair of columns.

    The codes are not arbitrary. Every arithmetic operation sorts below every
    comparison, because that is what lets `is_comparison` be one integer
    comparison instead of a list, and a new operation goes in on the side it
    belongs to rather than on the end.
    """

    var code: UInt8
    """The operation, as a small integer."""

    comptime ADD = Self(0)
    """Addition."""

    comptime SUB = Self(1)
    """Subtraction."""

    comptime MUL = Self(2)
    """Multiplication."""

    comptime DIV = Self(3)
    """Division, which answers float64 on integers and the promoted width on
    floats."""

    comptime FLOORDIV = Self(4)
    """Floor division, which keeps the operand type."""

    comptime MOD = Self(5)
    """The remainder that goes with the floor division."""

    comptime POW = Self(6)
    """Raising to a power."""

    comptime EQ = Self(7)
    """Equality."""

    comptime NE = Self(8)
    """Inequality."""

    comptime LT = Self(9)
    """Less than."""

    comptime LE = Self(10)
    """Less than or equal."""

    comptime GT = Self(11)
    """Greater than."""

    comptime GE = Self(12)
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
            True for the six comparisons, false for the seven arithmetic ones.
        """
        return self.code >= Self.EQ.code

    def mirrored(self) -> Self:
        """Returns the operation that means the same thing with the sides swapped.

        `5 < x` and `x > 5` are the same question, so a constant on the left of a
        comparison is handled by turning it round rather than by a second set of
        loops. The swap is exact for floats too: both readings are false when
        either side is a NaN.

        Equality and inequality are their own mirrors, and so is every
        arithmetic operation as far as this is concerned, because the five that
        are not symmetric take a flag into the loop instead. Only the four
        ordered comparisons change.

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
        elif self == Self.FLOORDIV:
            writer.write("//")
        elif self == Self.MOD:
            writer.write("%")
        elif self == Self.POW:
            writer.write("**")
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

    Two bools are the one pair that does not simply promote, and
    `bool_arithmetic_type` has the whole of that.

    Args:
        op: The operation.
        a: The left operand type.
        b: The right operand type.

    Returns:
        The result type: bool for a comparison, the promoted operand type for
        six of the seven arithmetic operations, and for division the promoted
        type when that is already a float and float64 when it is not. Floor
        division and the remainder are in the first group and not with division,
        because `//` and `%` keep the operand type in pandas and `/` does not.

        The condition on division is what pandas does and it is easy to get
        wrong in the direction of always answering float64. Dividing two
        integers produces a float because there is no integer answer, and that
        is the whole of the rule. Once one side is already a float there is
        nothing to widen, so `float32 / float32` is float32 and `float32 / int8`
        is float32 as well, since `promote` has already decided that an int8 is
        representable in a float32. Answering float64 to those doubles the
        memory of a column for a precision nobody asked for, and it stays
        doubled for every operation after it.

    Raises:
        If the two types have no common type, or the operation is arithmetic and
        the common type is not a number.
    """
    if a.is_temporal() or b.is_temporal():
        return temporal_binary_type(op, a, b)

    var common = promote(a, b)
    if op.is_comparison():
        return LogicalType.BOOL
    if common.kind == TypeKind.BOOL:
        return bool_arithmetic_type(op)
    if not common.is_numeric():
        raise Error(
            "binary: " + String(op) + " is not defined on " + String(common)
        )
    if op == BinaryOp.DIV:
        return common if common.is_float() else LogicalType.FLOAT64
    return common


def temporal_binary_type(
    op: BinaryOp, a: LogicalType, b: LogicalType
) raises -> LogicalType:
    """Returns the type an operation answers when a temporal column is in it.

    Four pairs have an answer and the rest do not. Two instants subtract to an
    elapsed time. An instant and an elapsed time add to an instant, either way
    round, and an instant minus an elapsed time is an instant too. Two elapsed
    times add and subtract to an elapsed time. And any two of the same kind
    compare, which is the one case where the answer is a bool and the units stop
    mattering.

    Every one of those is stated at the finer of the two resolutions, which is
    `promote`'s rule and is applied here by asking it. A second column minus a
    nanosecond column is a nanosecond duration, and it is a nanosecond duration
    whichever side the second column is on, because the rule is about what
    reconciling them loses rather than about which operand came first.

    What is refused is refused because pandas refuses it or because firepanda
    has not written it. Two instants do not add, and pandas says so in as many
    words: there is no instant halfway between two instants that is their sum. A
    duration times or divided by a number is a duration and is a real pandas
    answer that firepanda has no loop for yet, and a duration divided by a
    duration is a float and is the same. Those are absences and say so.

    Args:
        op: The operation.
        a: The left operand type.
        b: The right operand type.

    Returns:
        Bool for a comparison, a duration for a difference of instants, a
        timestamp for an instant shifted by an elapsed time, and a duration for
        a sum or difference of elapsed times.

    Raises:
        Error: If the pair has no answer, or has one that firepanda has not
            written yet.
    """
    var instants = a.kind == TypeKind.TIMESTAMP and b.kind == TypeKind.TIMESTAMP
    var spans = a.kind == TypeKind.DURATION and b.kind == TypeKind.DURATION

    if op.is_comparison():
        # `promote` refuses a mixture of kinds and a mixture of zones, which is
        # what decides this, so the comparison rule is one line and the message
        # for a pair that cannot compare is the promotion's own.
        _ = promote(a, b)
        return LogicalType.BOOL

    if instants:
        if op == BinaryOp.SUB:
            return LogicalType.duration(promote(a, b).unit)
        raise Error(
            "binary: "
            + String(op)
            + " is not defined on "
            + String(a)
            + " and "
            + String(b)
            + ", because two points in time have a difference and nothing"
            " else, and pandas will not add them either"
        )

    if spans:
        if op == BinaryOp.ADD or op == BinaryOp.SUB:
            return promote(a, b)
        raise Error(
            "binary: "
            + String(op)
            + " is not defined on "
            + String(a)
            + " and "
            + String(b)
            + ", because two elapsed times add and subtract and pandas gives"
            " their quotient as a plain number, which firepanda has not"
            " written yet"
        )

    var shift_left = a.kind == TypeKind.TIMESTAMP and b.kind == (
        TypeKind.DURATION
    )
    var shift_right = a.kind == TypeKind.DURATION and b.kind == (
        TypeKind.TIMESTAMP
    )
    if shift_left or shift_right:
        var stamp = a if shift_left else b
        var span = b if shift_left else a
        if op == BinaryOp.ADD or (shift_left and op == BinaryOp.SUB):
            return LogicalType.timestamp(
                finer_unit(stamp.unit, span.unit), TimeZone(copy=stamp.zone)
            )
        if op == BinaryOp.SUB:
            raise Error(
                "binary: an elapsed time minus a point in time has no answer,"
                " because the point in time is the thing being subtracted from"
                " and there is nothing on the left to subtract it from"
            )
        raise Error(
            "binary: "
            + String(op)
            + " is not defined on "
            + String(a)
            + " and "
            + String(b)
            + ", because a point in time and an elapsed time add and subtract"
            " and do nothing else"
        )

    # One temporal and something that is not temporal at all. `promote` has the
    # three messages for this, separated by which mixture it is, so it says it.
    _ = promote(a, b)
    raise Error(
        "binary: "
        + String(op)
        + " is not defined on "
        + String(a)
        + " and "
        + String(b)
    )


def unsupported_on_bool(op: BinaryOp) -> Bool:
    """Reports which arithmetic pandas declines to define on two bools.

    pandas raises `NotImplementedError` for three of the seven and a `TypeError`
    for a fourth, and the difference is not an accident of where the failure
    happens. `truediv`, `floordiv` and `pow` are turned away by a check on the
    dtype before anything is dispatched, and the message says the operator is
    not implemented for bool dtypes. Subtraction gets past that check and as far
    as numpy, which refuses it with a sentence naming `^` instead.

    The binding needs this to tag the error, since the two classes are different
    on the Python side and the core does not tag anything itself. It is a
    property of the operation and nothing else, so both halves can ask it.

    Args:
        op: The operation.

    Returns:
        True for the three pandas answers `NotImplementedError` to.
    """
    return op == BinaryOp.DIV or op == BinaryOp.FLOORDIV or op == BinaryOp.POW


def bool_arithmetic_type(op: BinaryOp) raises -> LogicalType:
    """The type an arithmetic operation answers on two bools.

    None of this is a design anybody would choose from scratch and all of it is
    measured against pandas 3.0.3. `+` is the logical or and `*` is the logical
    and, which is the boolean semiring and is defensible. `%` widens to int8 and
    is zero everywhere, which is numpy showing through. `-` has no answer at
    all, and neither do `/`, `//` and `**`, which is the part that surprises
    people: numpy answers all three on two bool arrays, giving a float64 and two
    int8s, and pandas puts a dtype check in front of them and refuses.

    So the operand that decides is the dtype and not the shape. `s / t` and
    `s / True` both raise, on a bool column, and they raise the same sentence.

    firepanda copies pandas rather than numpy here, because the pandas API is
    what it claims, and both refusals keep pandas' wording because the wording
    is the part a user acts on: the subtraction message names the operator that
    does work.

    Args:
        op: The operation, which is not a comparison. Every comparison is
            defined on bools and answers bool, and that is decided before this
            is reached.

    Returns:
        Bool for `+` and `*`, and int8 for `%`, which are the three of the seven
        that have an answer.

    Raises:
        For `-`, `/`, `//` and `**`. The message is pandas' own, and
        `unsupported_on_bool` says which of the two classes it should become.
    """
    if op == BinaryOp.ADD or op == BinaryOp.MUL:
        return LogicalType.BOOL
    if op == BinaryOp.SUB:
        raise Error(
            "numpy boolean subtract, the `-` operator, is not supported, use"
            " the bitwise_xor, the `^` operator, or the logical_xor function"
            " instead."
        )
    if unsupported_on_bool(op):
        raise Error(
            String(
                "operator '",
                _pandas_name(op),
                "' not implemented for bool dtypes",
            )
        )
    return LogicalType.INT8


def _pandas_name(op: BinaryOp) -> String:
    """The name pandas uses for an operation inside an error message.

    `binary_type` writes an operation as the symbol it is spelled with, which is
    what a message about two dtypes wants. This message is pandas' and pandas
    names the dunder instead, so the three that need it are spelled out here
    rather than the symbol being bent to serve both.

    Args:
        op: One of the three operations pandas declines on bools.

    Returns:
        The pandas name.
    """
    if op == BinaryOp.DIV:
        return "truediv"
    if op == BinaryOp.FLOORDIV:
        return "floordiv"
    return "pow"


def weak_operand_type(column: LogicalType, scalar: LogicalType) -> LogicalType:
    """The width a Python scalar takes when it meets a column.

    A Python `2` has no width. numpy 2 calls it a weak scalar and pandas 3
    inherits the rule whole, so this is the rule and not an interpretation of
    one. A weak scalar has a kind, the kinds are ordered bool then integer then
    float, and:

    - if the scalar's kind is no higher than the column's, the operation runs in
      the column's own dtype and the answer is the column's dtype;
    - if the scalar's kind is higher, the answer is the default width of the
      scalar's kind, which is int64 for an integer and float64 for a float.

    There is no case where a scalar makes a column wider than the default of its
    own kind, which is the whole point of it. An int8 column that becomes int64
    the first time somebody adds one to it is eight times the memory for a range
    of values that never needed it.

    Every measured pandas answer falls out of those two sentences: `int8 + 2` is
    int8, `int8 + True` is int8, `int8 + 2.5` is float64, `float32 + 2` is
    float32, `bool + 2` is int64 and `bool + True` is bool.

    A bool column is not numeric here, so it takes the first branch and the
    scalar keeps its own width. That is the right answer for all three of its
    cases and not a coincidence: a bool column has no width to give a number,
    which is exactly what the first branch says about a column that is not a
    number. `bool + 2` is int64 and `bool + 2.5` is float64 because that is what
    the scalar arrived as, and `bool + True` is bool for the same reason.

    Args:
        column: The dtype of the column the scalar is being applied to.
        scalar: The dtype the scalar was read as, which is the widest of its
            kind because that is all a Python object can say.

    Returns:
        The width to run the operation in.
    """
    if not column.is_numeric():
        return scalar
    if scalar.is_float():
        return column if column.is_float() else LogicalType.FLOAT64
    return column


def _fits[target: DType](value: Int64) -> Bool:
    """Reports whether an integer survives being narrowed to a dtype.

    The check is a round trip rather than a pair of bounds, because the bounds
    of a dtype are exactly what a round trip measures and writing them out again
    is a second copy of the same fact. The one case a round trip cannot see is a
    negative number against an unsigned dtype, which wraps to a large positive
    one and wraps back to the same negative one, so that is tested first.

    Parameters:
        target: The dtype to narrow to.

    Args:
        value: The number, read as an int64.

    Returns:
        True if narrowing loses nothing.
    """
    comptime if target == DType.bool:
        return value == 0 or value == 1
    comptime if not target.is_signed():
        if value < 0:
            return False
    return value.cast[target]().cast[DType.int64]() == value


def resolve_constant(
    column: LogicalType, value: Value, op: BinaryOp
) raises -> Value:
    """Gives a Python scalar the width it takes against this column.

    Called in two places and they have to agree: `binary_value_any` runs the
    operation and `ComputeNode.schema` declares what the operation will produce,
    so a rule applied in one and not the other is a plan whose declared dtype is
    not the dtype of the data it describes.

    A comparison is not an arithmetic operation here and the difference is
    measured rather than assumed. pandas answers `int8 < 128` with a column of
    True rather than raising, because that is a true statement about every int8
    there is, and it answers `int8 == 128` with a column of False. So a
    comparison against a value the column cannot hold is not an error and is not
    a narrowing: it is answered at the value's own magnitude, which is what the
    promotion already does. The narrowing still happens when it is exact, since
    `s > 5` on an int8 column should not widen the column to compare against a
    five.

    Args:
        column: The dtype of the column.
        value: The constant, which is returned unchanged unless it arrived from
            Python without a width.
        op: The operation, because a comparison resolves differently.

    Returns:
        The constant, at the width the operation should run in.

    Raises:
        If the operation is arithmetic and the scalar does not fit the column's
        dtype. The message is pandas' own, `Python integer {value} out of bounds
        for {dtype}`, since that is what somebody who hits this will search for.
        The binding turns it into an `OverflowError`.
    """
    # A text constant against a column of instants is the date literal, and it
    # is read here rather than anywhere further in, so that everything below this
    # line is comparing two numbers. Both callers get it: the operation runs on
    # the parsed constant and the plan declares the type the parsed constant
    # produces, which is the whole reason this function is shared.
    if (
        value.present
        and value.type.is_variable_width()
        and column.is_temporal()
    ):
        return parse_instant(value.text.value().as_bytes(), column)

    if not value.weak or value.is_null() or not column.is_numeric():
        return Value(copy=value)
    var want = weak_operand_type(column, value.type)
    if want == value.type:
        return Value(copy=value)

    if value.type.is_float():
        # A float scalar only ever narrows into a float column, and a float
        # column has no value it cannot hold: what does not fit becomes an
        # infinity, which is what pandas stores too.
        return _narrowed(value, want)

    var number = value.as_scalar[DType.int64]()
    if not _fits_type(want, number):
        if op.is_comparison():
            # Answered at full magnitude rather than refused. `int8 < 128` is
            # true of every int8 there is and pandas says so.
            return Value(copy=value)
        raise Error(
            String(
                "Python integer ",
                number,
                " out of bounds for ",
                want,
            )
        )
    return _narrowed(value, want)


def _fits_type(target: LogicalType, value: Int64) raises -> Bool:
    """Resolves a runtime dtype to a parameter and asks `_fits`.

    Args:
        target: The dtype to narrow to.
        value: The number.

    Returns:
        True if narrowing loses nothing.

    Raises:
        If the dtype has no physical layout here.
    """
    if target.is_float():
        return True
    comptime for t in ALL:
        if target.physical == t:
            return _fits[t](value)
    raise Error("binary: unsupported dtype " + String(target))


def _narrowed(value: Value, target: LogicalType) raises -> Value:
    """Rebuilds a constant at another width.

    The result is not weak. It has a width now, and asking the same question of
    it twice has to give the same answer the second time.

    Args:
        value: The constant.
        target: The width to rebuild it at.

    Returns:
        The constant at `target`.

    Raises:
        If the dtype has no physical layout here.
    """
    comptime for t in ALL:
        if target.physical == t:
            return Value(value.as_scalar[t]())
    raise Error("binary: unsupported dtype " + String(target))


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
    # Ahead of `binary_type`, because that reaches `promote` and `promote`
    # refuses every mixture involving a dictionary type. Arithmetic falls
    # through on purpose and collects that refusal, which is the right answer
    # for it.
    if (a.is_dictionary() or b.is_dictionary()) and op.is_comparison():
        return _dictionary_erased(a, b, op)

    # Checked first and for its own sake. Everything below assumes the pair has
    # a common type and that the operation is defined on it.
    var answer = binary_type(op, a.type, b.type)

    if a.type.is_temporal() or b.type.is_temporal():
        return _temporal_erased(a, b, op, answer)

    # The common type is the promotion in all four shapes, including division.
    # A float division answers at the common type, so there is nothing else to
    # convert to. An integer division answers float64, and its loop reads the
    # operands at their own width and widens each register as it goes, so
    # casting both columns to float64 first would be two copies of the column
    # to reach the same answer.
    var common = promote(a.type, b.type)
    # Bool is the one case where the common type is not the type the loop runs
    # in. The remainder answers int8 on two bools, so both columns widen to int8
    # first and the ordinary integer loop does the work, zero divisor rule and
    # all, rather than there being a bool remainder loop that answers zero. The
    # two that stay bool are `+` and `*`, whose answer type is the common type,
    # so this leaves them alone, and the other four never get here.
    if common.kind == TypeKind.BOOL and not op.is_comparison():
        common = answer
    if common.is_variable_width():
        return _compare_text_erased(a, b, op)

    var left = AnyArray(copy=a) if a.type == common else cast_any(
        a, common.physical
    )
    var right = AnyArray(copy=b) if b.type == common else cast_any(
        b, common.physical
    )
    return _binary_erased(left, right, op, common.physical)


def _compare_text_erased(
    a: AnyArray, b: AnyArray, op: BinaryOp
) raises -> AnyArray:
    """Sends a comparison on two text columns to the byte loops.

    No conversion happens first. Two variable width columns only have a common
    type when they are the same kind already, so there is nothing to promote, and
    the operation is known to be a comparison because arithmetic on text was
    turned away by `binary_type`.

    Args:
        a: The left column.
        b: The right column, of the same kind.
        op: The operation.

    Returns:
        A bool column.

    Raises:
        If either column has a variable width type but does not carry the
        elements, which is what an all-null column of no type looks like.
    """
    ref x = a.strings()
    ref y = b.strings()
    if op == BinaryOp.EQ:
        return AnyArray(compare_text[CMP_EQ](x, y))
    if op == BinaryOp.NE:
        return AnyArray(compare_text[CMP_NE](x, y))
    if op == BinaryOp.LT:
        return AnyArray(compare_text[CMP_LT](x, y))
    if op == BinaryOp.LE:
        return AnyArray(compare_text[CMP_LE](x, y))
    if op == BinaryOp.GT:
        return AnyArray(compare_text[CMP_GT](x, y))
    return AnyArray(compare_text[CMP_GE](x, y))


comptime _UNORDERED = "Unordered Categoricals can only compare equality or not"
"""What pandas says when an ordering comparison meets a categorical that has no
meaning to its order. Reproduced word for word, because a program that catches
the `TypeError` and matches on its text is a program that exists."""


def _scalar_kind(type: LogicalType) -> String:
    """The Python type name pandas puts in an invalid comparison message.

    pandas names the type of the object the caller passed rather than the dtype
    it was read at, so a `5` is an `int` and not an int64. Only the four kinds a
    Python literal can arrive as need a name here.

    Args:
        type: The dtype the scalar was read as.

    Returns:
        The name.
    """
    if type.is_variable_width():
        return "str"
    if type.kind == TypeKind.BOOL:
        return "bool"
    if type.is_float():
        return "float"
    return "int"


def _dunder(op: BinaryOp) -> String:
    """The dunder pandas names a comparison by inside an error message.

    Args:
        op: The comparison.

    Returns:
        The dunder, with the underscores on it.
    """
    if op == BinaryOp.EQ:
        return "__eq__"
    if op == BinaryOp.NE:
        return "__ne__"
    if op == BinaryOp.LT:
        return "__lt__"
    if op == BinaryOp.LE:
        return "__le__"
    if op == BinaryOp.GT:
        return "__gt__"
    return "__ge__"


def _dictionary_erased(
    a: AnyArray, b: AnyArray, op: BinaryOp
) raises -> AnyArray:
    """Compares a category column with another column.

    Three shapes and three answers. Two categoricals that hold the same
    categories in the same order compare as their codes do, which is an integer
    comparison over columns that already exist and is cheaper than comparing the
    text they decode to. Two that disagree about their categories are refused,
    because a code means nothing outside the list it indexes and there is no
    reading of `<` that spans two different lists. A categorical against an
    ordinary column compares by value under equality, which means decoding, and
    is refused under an ordering, which is pandas' rule and not one invented
    here.

    An ordering comparison needs both sides ordered. That is what `ordered=True`
    is for and it is the one thing a caller sets it to get.

    The decoding arm only has somewhere to go when the other column holds text,
    since a categorical holds text and a number column has no reading of `==`
    against it that is not already false. That is refused rather than answered
    all false, because a caller comparing a category column against a number has
    made a mistake rather than asked a question with a boring answer.

    Args:
        a: The left column.
        b: The right column, of the same length.
        op: The comparison.

    Returns:
        A bool column, null wherever either input is null.

    Raises:
        Error: If the two sides disagree about their categories, if an ordering
            comparison meets an unordered categorical, if an ordering comparison
            meets a column that is not a categorical at all, or if the other
            column is neither a categorical nor text.
    """
    var equality = op == BinaryOp.EQ or op == BinaryOp.NE
    if a.is_dictionary() and b.is_dictionary():
        if not same_categories(a.categories(), b.categories()):
            raise Error(
                "Categoricals can only be compared if 'categories' are the"
                " same."
            )
        if not equality and not (a.type.ordered and b.type.ordered):
            raise Error(_UNORDERED)
        return _binary_erased(
            AnyArray(dictionary_codes(a)),
            AnyArray(dictionary_codes(b)),
            op,
            DType.int32,
        )

    if not equality:
        raise Error(
            String(
                "Cannot compare a Categorical for op ",
                _dunder(op),
                " with type ",
                (b if a.is_dictionary() else a).type,
                (
                    ". If you want to compare values, decode the categorical"
                    " first with astype(str)."
                ),
            )
        )

    var other = (b.type if a.is_dictionary() else a.type).copy()
    if not other.is_variable_width():
        raise Error(
            String(
                "no common type for category and ",
                other,
                (
                    ", because a categorical holds text and comparing it to a"
                    " number is a comparison between two kinds of thing"
                ),
            )
        )

    # `_compare_text_erased` rather than `binary_any`, which is what
    # `binary_any` would reach anyway with two text columns and a comparison.
    # Going back through the front door would make this function and that one
    # mutually recursive, and the compiler pays for that: the same call written
    # as recursion took the binary tests from a two second build to one that had
    # not finished in ten minutes.
    var left = AnyArray(
        decode_dictionary(a)
    ) if a.is_dictionary() else AnyArray(copy=a)
    var right = AnyArray(
        decode_dictionary(b)
    ) if b.is_dictionary() else AnyArray(copy=b)
    return _compare_text_erased(left, right, op)


def _dictionary_const_erased(
    a: AnyArray, b: Value, op: BinaryOp, value_on_left: Bool
) raises -> AnyArray:
    """Compares a category column with one constant.

    The whole operation is a lookup and an integer comparison. The constant is
    found in the categories once, which gives a code, and then the column's
    codes are compared against that one number. Nothing is decoded and nothing
    is promoted.

    The asymmetry between equality and ordering is where the constant is not one
    of the categories. Equality can still answer, because a value that is not a
    category is not equal to any row, and pandas answers all false. An ordering
    cannot, because there is no position to compare against, and pandas raises.
    The all false answer is written as a comparison against minus one rather
    than as a loop of its own, since no code is ever negative and the ordinary
    constant loop already carries the nulls across.

    Args:
        a: The category column.
        b: The constant.
        op: The comparison.
        value_on_left: True for `"b" < s` rather than `s < "b"`.

    Returns:
        A bool column, null wherever the column is null and null everywhere if
        the constant is null.

    Raises:
        Error: If an ordering comparison meets an unordered categorical, or one
            whose categories do not include the constant.
    """
    if b.is_null():
        return all_null(LogicalType.BOOL, len(a))

    var applied = op.mirrored() if value_on_left else op
    var equality = applied == BinaryOp.EQ or applied == BinaryOp.NE

    var at = -1
    if b.type.is_variable_width():
        ref categories = a.categories()
        var text = b.as_string()
        for k in range(len(categories)):
            if categories.is_valid(k) and categories[k] == text:
                at = k
                break

    if at < 0:
        if not equality:
            if not a.type.ordered:
                raise Error(_UNORDERED)
            raise Error(
                String(
                    "Invalid comparison between dtype=category and ",
                    _scalar_kind(b.type),
                )
            )
        return _binary_const_erased(
            AnyArray(dictionary_codes(a)),
            Value(Int32(-1)),
            applied,
            DType.int32,
            False,
        )

    if not equality and not a.type.ordered:
        raise Error(_UNORDERED)
    return _binary_const_erased(
        AnyArray(dictionary_codes(a)),
        Value(Int32(at)),
        applied,
        DType.int32,
        False,
    )


def _temporal_erased(
    a: AnyArray, b: AnyArray, op: BinaryOp, answer: LogicalType
) raises -> AnyArray:
    """Runs an operation on a pair where at least one side is temporal.

    This is the temporal twin of the three steps at the top of `binary_any` and
    it differs in exactly one of them. Reconciling two temporal columns is a
    rescale and not a cast: a second column read in nanoseconds is every value
    times a thousand million, where a cast from int32 to int64 is the same
    number in a wider box. So the conversion goes through `temporal_as_unit`,
    which multiplies and refuses a column that will not fit afterwards, and the
    loop underneath is the ordinary int64 one.

    A pair of dates is the exception and needs no rescale at all, because a day
    is a day. They only compare, since `binary_type` refuses everything else on
    them, so they go straight to the int32 loop.

    Args:
        a: The left column.
        b: The right column.
        op: The operation, already known to have an answer on this pair.
        answer: The type that answer carries, from `binary_type`.

    Returns:
        The result, tagged with `answer` rather than with the int64 the loop
        worked in.

    Raises:
        Error: If reconciling the two resolutions puts a value outside an int64,
            or the loop cannot run.
    """
    if a.type.kind == TypeKind.DATE:
        return _binary_erased(a, b, op, DType.int32)

    var working = finer_unit(a.type.unit, b.type.unit)
    var left = temporal_as_unit(a, working)
    var right = temporal_as_unit(b, working)
    var out = _binary_erased(left, right, op, DType.int64)
    if op.is_comparison():
        return out^
    return AnyArray(out^.into_typed[DType.int64]().into_data(), answer)


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
                # Two of the seven arrive here still bool. The remainder widened
                # to int8 before the call and is running the integer loop
                # somewhere else in this same `comptime for`, and the other four
                # were turned away by `binary_type` before any column moved.
                if op == BinaryOp.ADD:
                    return AnyArray(logical_or(x, y))
                if op == BinaryOp.MUL:
                    return AnyArray(logical_and(x, y))
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
                if op == BinaryOp.FLOORDIV:
                    return AnyArray(floor_divide(x, y))
                if op == BinaryOp.MOD:
                    return AnyArray(modulo(x, y))
                if op == BinaryOp.POW:
                    return AnyArray(power(x, y))
                # Division is the one operation whose answer is not the type it
                # was given, and only when it was given integers.
                comptime if target.is_floating_point():
                    return AnyArray(divide_float(x, y))
                else:
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
        A column with as many rows as `a`, null wherever `a` is null and null
        everywhere if the constant is null. The dtype is `binary_type` of the
        column and the constant, after the constant has been given a width by
        `resolve_constant`.

    Raises:
        If the two types have no common type, if the operation is not defined on
        that common type, or if the constant is an integer the column's dtype
        cannot hold and the operation is arithmetic.
    """
    # The same arm `binary_any` has, and ahead of `resolve_constant` for the
    # same reason: a category column has no width to give a weak scalar and no
    # common type with anything, and a comparison needs neither.
    if a.is_dictionary() and op.is_comparison():
        return _dictionary_const_erased(a, b, op, value_on_left)

    # A Python scalar arrives without a width and takes the column's, so this
    # runs before anything reads the constant's type.
    var scalar = resolve_constant(a.type, b, op)
    var left = a.type if not value_on_left else scalar.type
    var right = scalar.type if not value_on_left else a.type
    var answer = binary_type(op, left, right)

    if scalar.is_null():
        return all_null(answer, len(a))

    if a.type.is_temporal() or scalar.type.is_temporal():
        return _temporal_const_erased(a, scalar, op, answer, value_on_left)

    var common = promote(a.type, scalar.type)
    # The same widening `binary_any` does, and the same reason. A Python `True`
    # against a bool column is the only way to reach it here, and writing it the
    # same way in both places is what keeps them from drifting.
    if common.kind == TypeKind.BOOL and not op.is_comparison():
        common = answer
    # The comparison with the constant on the left is turned round rather than
    # given loops of its own. Subtraction and division cannot be turned round, so
    # they carry the flag into the loop, where the branch on it sits outside.
    var applied = op.mirrored() if value_on_left and op.is_comparison() else op
    if common.is_variable_width():
        return _compare_text_const_erased(a, scalar, applied)

    var column = AnyArray(copy=a) if a.type == common else cast_any(
        a, common.physical
    )
    var flip = value_on_left and not op.is_comparison()
    return _binary_const_erased(column, scalar, applied, common.physical, flip)


def all_null(type: LogicalType, rows: Int) raises -> AnyArray:
    """Builds a column of a given type with every row missing.

    The values buffer starts zeroed and a null holds a zero, so this only has to
    install a bitmap. Running the repair pass over it would be a write of the
    zeros that are already there.

    The type comes back out as well as in. A block of missing timestamps is a
    block of missing timestamps and not a block of missing integers that happen
    to be the same width, and the difference shows up the moment somebody tries
    to stack the block onto a real column.

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
            return AnyArray(out^.into_data(), type)
    raise Error("binary: unsupported dtype")


def filled_block(type: LogicalType, rows: Int, fill: Value) raises -> AnyArray:
    """Builds a column of a given type with every row holding the same value.

    `all_null` above is this with nothing to put in the rows, and it stays a
    function of its own rather than becoming a branch here because it writes no
    values at all: the buffer arrives zeroed and a null holds a zero, so it only
    has to install a bitmap.

    Two callers want a block of one value repeated and they are further apart
    than they look. A shift fills the gap it opens at one end of a column, and a
    reindex fills the rows whose label the frame does not have. In both cases
    the value arrives without a width of its own and takes the column's, which
    is why the type is a parameter rather than being read off the value.

    Args:
        type: The type the block should have. A filled timestamp is a timestamp
            and not an integer of the same width, and the difference shows up
            the moment somebody stacks the block onto a real column.
        rows: How many rows.
        fill: What to put in them. A null value leaves them missing.

    Returns:
        A column of `rows` rows, all of them the same.

    Raises:
        Error: If the type has no physical layout, or the value cannot be read
            as that type.
    """
    if type.is_variable_width():
        var builder = StringBuilder(capacity=rows)
        if fill.is_null():
            for _ in range(rows):
                builder.append_null()
        else:
            var text = fill.as_string()
            for _ in range(rows):
                builder.append(text.as_bytes())
        return AnyArray(builder^.finish())

    if fill.is_null():
        return all_null(type, rows)

    comptime for candidate in ALL:
        if type.physical == candidate:
            var out = Array[candidate](rows)
            var one = fill.as_scalar[candidate]()
            for i in range(rows):
                out[i] = one
            return AnyArray(out^.into_data(), type)
    raise Error("binary: unsupported dtype " + String(type))


def _compare_text_const_erased(
    a: AnyArray, b: Value, op: BinaryOp
) raises -> AnyArray:
    """Sends a comparison against a text constant to the byte loops.

    The constant's bytes are borrowed from the string it is holding, so the
    string has to outlive the call and is kept in a local rather than read out of
    the value inside the argument list.

    Args:
        a: The column.
        b: The constant, present and holding text.
        op: The operation, already mirrored if the constant was on the left.

    Returns:
        A bool column.

    Raises:
        If the column does not carry elements, or the constant is not text.
    """
    ref x = a.strings()
    var text = b.as_string()
    var probe = text.as_bytes()
    if op == BinaryOp.EQ:
        return AnyArray(compare_text_const[CMP_EQ](x, probe))
    if op == BinaryOp.NE:
        return AnyArray(compare_text_const[CMP_NE](x, probe))
    if op == BinaryOp.LT:
        return AnyArray(compare_text_const[CMP_LT](x, probe))
    if op == BinaryOp.LE:
        return AnyArray(compare_text_const[CMP_LE](x, probe))
    if op == BinaryOp.GT:
        return AnyArray(compare_text_const[CMP_GT](x, probe))
    return AnyArray(compare_text_const[CMP_GE](x, probe))


def _temporal_const_erased(
    a: AnyArray,
    b: Value,
    op: BinaryOp,
    answer: LogicalType,
    value_on_left: Bool,
) raises -> AnyArray:
    """Runs an operation between a temporal column and one temporal constant.

    The constant side of `_temporal_erased`, and the same one difference from
    the ordinary path: reconciling two resolutions is a multiply rather than a
    cast. The column goes through `temporal_as_unit` and the constant is one
    multiplication, which is where a constant earns its keep, since the column
    is the only thing that has to be walked.

    The constant is the side that is more likely to move, because a
    `Timedelta(hours=1)` is microseconds whatever column it meets, so an hour
    added to a column of seconds drags the whole column up to microseconds. That
    is pandas' answer and it is a surprising one, and it is surprising in pandas
    rather than here.

    Args:
        a: The column.
        b: The constant, present and temporal, or a temporal column met by a
            constant of some other kind, which `binary_type` has already refused.
        op: The operation.
        answer: The type the answer carries, from `binary_type`.
        value_on_left: True for `Timedelta(...) + s` rather than `s + ...`.

    Returns:
        The result, tagged with `answer`.

    Raises:
        Error: If reconciling the two resolutions puts a value outside an int64,
            or the loop cannot run.
    """
    var applied = op.mirrored() if value_on_left and op.is_comparison() else op
    var flip = value_on_left and not op.is_comparison()
    if a.type.kind == TypeKind.DATE:
        return _binary_const_erased(a, b, applied, DType.int32, flip)

    var working = finer_unit(a.type.unit, b.type.unit)
    var column = temporal_as_unit(a, working)
    var count = b.as_scalar[DType.int64]() * (
        working.per_second() // b.type.unit.per_second()
    )
    var scalar = Value(count)
    var out = _binary_const_erased(column, scalar, applied, DType.int64, flip)
    if op.is_comparison():
        return out^
    return AnyArray(out^.into_typed[DType.int64]().into_data(), answer)


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
                # The same two as the two column path, and no branch on
                # `flip`, because both of them commute.
                if op == BinaryOp.ADD:
                    return AnyArray(logical_or_const(x, y))
                if op == BinaryOp.MUL:
                    return AnyArray(logical_and_const(x, y))
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
                if op == BinaryOp.FLOORDIV:
                    return AnyArray(floor_divide_const[target](x, y, flip))
                if op == BinaryOp.MOD:
                    return AnyArray(modulo_const[target](x, y, flip))
                if op == BinaryOp.POW:
                    return AnyArray(power_const[target](x, y, flip))
                comptime if target.is_floating_point():
                    return AnyArray(divide_float_const[target](x, y, flip))
                else:
                    return AnyArray(divide_const[target](x, y, flip))
    raise Error("binary: unsupported dtype")
