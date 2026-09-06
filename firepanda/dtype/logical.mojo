"""The logical type lattice.

Mojo's `DType` describes how many bits a value occupies and how to interpret
them. It does not describe what the value means. A pandas column of timestamps
and a pandas column of nanosecond counts are both `int64` physically and are not
the same type to a user. `LogicalType` carries the meaning; `DType` carries the
layout. Everything below the array layer works on `DType` alone.

See docs/specs/03-dtype-dispatch.md.
"""

from .lists import ALL, FLOAT, INTEGER, SIGNED, UNSIGNED, contains, dtype_size
from .temporal import TimeUnit, TimeZone


@fieldwise_init
struct TypeKind(Equatable, ImplicitlyCopyable, Movable, Writable):
    """What a column of values means, independent of its physical layout."""

    var code: UInt8

    comptime NULL = Self(0)
    """A column whose every element is null. Has no physical buffer."""

    comptime BOOL = Self(1)
    """Booleans, stored one bit per value."""

    comptime INT = Self(2)
    """Signed or unsigned integers."""

    comptime FLOAT_KIND = Self(3)
    """IEEE 754 binary floating point."""

    comptime STRING = Self(4)
    """UTF-8 text. Physically uint8 payload bytes plus offsets."""

    comptime BINARY = Self(5)
    """Opaque bytes. Same physical layout as STRING, no encoding promise."""

    comptime TIMESTAMP = Self(6)
    """An instant, counted from the Unix epoch in the type's own unit. Stored
    int64, which is the one place in this file where the physical layout says
    nothing at all about the meaning: a count of microseconds since 1970 and a
    count of apples are the same eight bytes."""

    comptime DATE = Self(7)
    """A calendar day, counted from the Unix epoch. Stored int32."""

    def __eq__(self, other: Self) -> Bool:
        """Compares two kinds.

        Args:
            other: The kind to compare against.

        Returns:
            True if the kinds are the same.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two kinds for inequality.

        Args:
            other: The kind to compare against.

        Returns:
            True if the kinds differ.
        """
        return self.code != other.code

    def write_to(self, mut writer: Some[Writer]):
        """Writes the kind name.

        Args:
            writer: The destination.
        """
        if self == Self.NULL:
            writer.write("null")
        elif self == Self.BOOL:
            writer.write("bool")
        elif self == Self.INT:
            writer.write("int")
        elif self == Self.FLOAT_KIND:
            writer.write("float")
        elif self == Self.STRING:
            writer.write("string")
        elif self == Self.BINARY:
            writer.write("binary")
        elif self == Self.TIMESTAMP:
            writer.write("timestamp")
        else:
            writer.write("date")


@fieldwise_init
struct LogicalType(Equatable, ImplicitlyCopyable, Movable, Writable):
    """A physical dtype plus the meaning attached to it."""

    var kind: TypeKind
    """What the values mean."""

    var physical: DType
    """How the values are laid out in memory."""

    var unit: TimeUnit
    """How many of the values make one second. Read only for a temporal type,
    where it is part of the type in the sense that a microsecond column and a
    nanosecond column are not the same type and never compare equal."""

    var zone: TimeZone
    """The wall clock a timestamp is read against, naive for everything else."""

    def __init__(out self, kind: TypeKind, physical: DType):
        """Constructs a type with no temporal meaning attached.

        This is the two argument form every type but a timestamp and a date is
        built with, and it exists so that the fifteen constants below and the
        several hundred call sites that predate the temporal types read the way
        they always did.

        Args:
            kind: What the values mean.
            physical: How the values are laid out.
        """
        self.kind = kind
        self.physical = physical
        self.unit = TimeUnit.SECOND
        self.zone = TimeZone()

    comptime NULL = Self(TypeKind.NULL, DType.bool)
    comptime BOOL = Self(TypeKind.BOOL, DType.bool)
    comptime INT8 = Self(TypeKind.INT, DType.int8)
    comptime INT16 = Self(TypeKind.INT, DType.int16)
    comptime INT32 = Self(TypeKind.INT, DType.int32)
    comptime INT64 = Self(TypeKind.INT, DType.int64)
    comptime UINT8 = Self(TypeKind.INT, DType.uint8)
    comptime UINT16 = Self(TypeKind.INT, DType.uint16)
    comptime UINT32 = Self(TypeKind.INT, DType.uint32)
    comptime UINT64 = Self(TypeKind.INT, DType.uint64)
    comptime FLOAT16 = Self(TypeKind.FLOAT_KIND, DType.float16)
    comptime FLOAT32 = Self(TypeKind.FLOAT_KIND, DType.float32)
    comptime FLOAT64 = Self(TypeKind.FLOAT_KIND, DType.float64)
    comptime STRING = Self(TypeKind.STRING, DType.uint8)
    comptime BINARY = Self(TypeKind.BINARY, DType.uint8)
    comptime DATE32 = Self(TypeKind.DATE, DType.int32)

    @staticmethod
    def timestamp(unit: TimeUnit, zone: TimeZone = TimeZone()) -> Self:
        """Constructs a timestamp type.

        Args:
            unit: How many of the column's integers make one second.
            zone: The wall clock the instants are read against. Left out, the
                column is naive, which is not the same promise as UTC.

        Returns:
            The type.
        """
        return Self(TypeKind.TIMESTAMP, DType.int64, unit, zone)

    def __eq__(self, other: Self) -> Bool:
        """Compares two logical types.

        The unit and the zone are part of the comparison because they are part
        of the type. A microsecond column and a nanosecond column hold different
        instants for the same integer, and a New York column and a naive one are
        different questions with the same answer only by accident, so neither
        pair is allowed to compare equal and slip through a kernel that checks
        whether two operands match.

        Args:
            other: The type to compare against.

        Returns:
            True if the kind, the physical layout, the unit and the zone match.
        """
        return (
            self.kind == other.kind
            and self.physical == other.physical
            and self.unit == other.unit
            and self.zone == other.zone
        )

    def __ne__(self, other: Self) -> Bool:
        """Compares two logical types for inequality.

        Args:
            other: The type to compare against.

        Returns:
            True if the types differ.
        """
        return not (self == other)

    def is_numeric(self) -> Bool:
        """Reports whether arithmetic is defined on this type.

        Returns:
            True for integers and floats, False otherwise.
        """
        return self.kind == TypeKind.INT or self.kind == TypeKind.FLOAT_KIND

    def is_integer(self) -> Bool:
        """Reports whether this is an integer type.

        Returns:
            True for signed and unsigned integers.
        """
        return self.kind == TypeKind.INT

    def is_float(self) -> Bool:
        """Reports whether this is a floating point type.

        Returns:
            True for float16, float32 and float64.
        """
        return self.kind == TypeKind.FLOAT_KIND

    def is_signed(self) -> Bool:
        """Reports whether this is a signed integer type.

        Returns:
            True for int8 through int64.
        """
        return self.kind == TypeKind.INT and contains[SIGNED](self.physical)

    def is_temporal(self) -> Bool:
        """Reports whether this type means a point in time.

        Returns:
            True for timestamps and dates.
        """
        return self.kind == TypeKind.TIMESTAMP or self.kind == TypeKind.DATE

    def is_variable_width(self) -> Bool:
        """Reports whether values are stored out of line behind an offsets array.

        Returns:
            True for string and binary.
        """
        return self.kind == TypeKind.STRING or self.kind == TypeKind.BINARY

    def bit_width(self) -> Int:
        """Returns the number of bits one physical element occupies.

        Returns:
            1 for bool, otherwise the width of the physical dtype.
        """
        if self.physical == DType.bool:
            return 1
        return dtype_size(self.physical) * 8

    def write_to(self, mut writer: Some[Writer]):
        """Writes the type in the form users see it.

        The two temporal spellings are pandas' rather than Arrow's, because this
        string is what `dtype` hands back and a user comparing it against
        `datetime64[ns]` is comparing against pandas. The date is the exception
        and is spelled the way Arrow spells it, since pandas on the numpy
        backend has no date dtype at all and calls the column `object`, which is
        a thing firepanda would be lying to say.

        Args:
            writer: The destination.
        """
        if self.kind == TypeKind.NULL:
            writer.write("null")
        elif self.kind == TypeKind.STRING:
            writer.write("string")
        elif self.kind == TypeKind.BINARY:
            writer.write("binary")
        elif self.kind == TypeKind.TIMESTAMP:
            writer.write("datetime64[", self.unit)
            if not self.zone.is_naive():
                writer.write(", ", self.zone)
            writer.write("]")
        elif self.kind == TypeKind.DATE:
            writer.write("date32[day]")
        else:
            writer.write(self.physical)


def logical_for(dt: DType) -> LogicalType:
    """Returns the default logical type for a physical dtype.

    This is the inverse used when an array is built from raw values with no
    stated meaning. A `uint8` array defaults to an integer column, not a string
    column, because string columns are never constructed this way.

    Args:
        dt: The physical dtype.

    Returns:
        The logical type a bare array of `dt` carries.
    """
    if dt == DType.bool:
        return LogicalType(TypeKind.BOOL, dt)
    if contains[FLOAT](dt):
        return LogicalType(TypeKind.FLOAT_KIND, dt)
    return LogicalType(TypeKind.INT, dt)


def promote(a: LogicalType, b: LogicalType) raises -> LogicalType:
    """Returns the type both operands are converted to before a binary operation.

    The rules follow NumPy and pandas rather than SQL:

    - null promotes to the other side.
    - float wins over integer, and the result is at least as wide as the integer
      operand can be represented in. int64 with float32 gives float64.
    - signed with unsigned of the same or greater width gives the next signed
      width up, and uint64 with a signed type gives float64, which is what NumPy
      does and is lossy for values above 2^53.
    - bool with anything numeric gives the numeric type.
    - string only promotes with string.

    Args:
        a: The left operand type.
        b: The right operand type.

    Returns:
        The common type.

    Raises:
        If no common type exists, for example string with int64.
    """
    if a == b:
        return a
    if a.kind == TypeKind.NULL:
        return b
    if b.kind == TypeKind.NULL:
        return a

    if a.is_temporal() or b.is_temporal():
        # Two identical temporal types were returned above, so anything arriving
        # here is either a timestamp against a number, which has no answer, or
        # two timestamps that disagree about the unit or the zone, which has one
        # and firepanda cannot spell it yet: the difference of two instants is a
        # duration and there is no duration type. Both refuse, and the message
        # says which of the two it was, because a user who wrote `a - b` on two
        # microsecond columns in different zones has a different problem from
        # one who wrote `a - 1`.
        if a.is_temporal() and b.is_temporal():
            raise Error(
                "no common type for "
                + String(a)
                + " and "
                + String(b)
                + ", because the two differ in unit or in time zone and"
                " reconciling them is a conversion rather than a promotion"
            )
        raise Error(
            "no common type for "
            + String(a)
            + " and "
            + String(b)
            + ", because a point in time and a number are not the same kind of"
            " thing and pandas will not add them either"
        )

    if a.is_variable_width() or b.is_variable_width():
        if a.kind == b.kind:
            return a
        raise Error("no common type for " + String(a) + " and " + String(b))

    if a.kind == TypeKind.BOOL and b.is_numeric():
        return b
    if b.kind == TypeKind.BOOL and a.is_numeric():
        return a

    if a.is_float() or b.is_float():
        return _promote_with_float(a, b)

    return _promote_integers(a, b)


def _promote_with_float(a: LogicalType, b: LogicalType) -> LogicalType:
    """Promotes a pair where at least one side is floating point.

    Args:
        a: The left operand type.
        b: The right operand type.

    Returns:
        The common floating point type.
    """
    if a.is_float() and b.is_float():
        return a if a.bit_width() >= b.bit_width() else b

    var flt = a if a.is_float() else b
    var intg = b if a.is_float() else a

    # An integer wider than the float's mantissa forces the result up to
    # float64, matching NumPy. int32 has 31 significant bits and float32 has 24,
    # so int32 with float32 gives float64.
    if intg.bit_width() > 16 and flt.physical == DType.float16:
        return (
            LogicalType.FLOAT32 if intg.bit_width()
            <= 24 else LogicalType.FLOAT64
        )
    if intg.bit_width() > 24 and flt.physical == DType.float32:
        return LogicalType.FLOAT64
    if intg.bit_width() > 8 and flt.physical == DType.float16:
        return LogicalType.FLOAT32
    return flt


def _promote_integers(a: LogicalType, b: LogicalType) -> LogicalType:
    """Promotes a pair of integer types.

    Args:
        a: The left operand type.
        b: The right operand type.

    Returns:
        The common type, which may be float64 when uint64 meets a signed type.
    """
    var a_signed = contains[SIGNED](a.physical)
    var b_signed = contains[SIGNED](b.physical)

    if a_signed == b_signed:
        return a if a.bit_width() >= b.bit_width() else b

    var signed = a if a_signed else b
    var unsigned = b if a_signed else a

    if unsigned.bit_width() >= 64:
        # No signed integer holds the top of uint64. NumPy answers float64 here
        # and so do we, lossily and on purpose, so that the result of an
        # expression never depends on the values in the column.
        return LogicalType.FLOAT64
    if signed.bit_width() > unsigned.bit_width():
        return signed
    return _next_signed_width(unsigned.bit_width())


def _next_signed_width(bits: Int) -> LogicalType:
    """Returns the smallest signed type that holds every value of `bits` unsigned bits.

    Args:
        bits: The width of the unsigned operand.

    Returns:
        The signed type one step wider.
    """
    if bits <= 8:
        return LogicalType.INT16
    if bits <= 16:
        return LogicalType.INT32
    return LogicalType.INT64
