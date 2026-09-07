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

    comptime DURATION = Self(8)
    """An elapsed amount of time in the type's own unit, stored int64. Not an
    instant and not counted from anything: it is what you get by subtracting two
    timestamps, and adding two of them is a sensible thing to do where adding
    two instants is not."""

    comptime DICTIONARY = Self(9)
    """Values held once in a separate array and referred to by position. The
    physical dtype is the index type and says nothing about what the values are,
    which makes this the one kind where reading the buffer as its dtype gives a
    number that is not the value the user put there."""

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
        elif self == Self.DATE:
            writer.write("date")
        elif self == Self.DURATION:
            writer.write("duration")
        else:
            writer.write("dictionary")


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

    var ordered: Bool
    """Whether the values of a dictionary have a meaning to their order, so that
    one category is less than another. False for every other kind, where the
    question does not arise. It lives on the type and not on the array because
    pandas puts it in the dtype, where it decides whether `<` on two categorical
    columns is an error or an answer."""

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
        self.ordered = False

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
        return Self(TypeKind.TIMESTAMP, DType.int64, unit, zone, False)

    @staticmethod
    def duration(unit: TimeUnit) -> Self:
        """Constructs a duration type.

        There is no zone argument and there is no zone. An elapsed amount of
        time is the same amount of time wherever it is read, which is the whole
        difference between this type and a timestamp.

        Args:
            unit: How many of the column's integers make one second.

        Returns:
            The type.
        """
        return Self(TypeKind.DURATION, DType.int64, unit, TimeZone(), False)

    @staticmethod
    def dictionary(index: DType, ordered: Bool = False) -> Self:
        """Constructs a dictionary type.

        The categories are not here and cannot be. A `LogicalType` is copied
        around by value everywhere in the library and owns no memory, and the
        categories are a string array. They live on the column instead, which
        means two dictionary columns over different categories carry types that
        compare equal, and anything that needs to know whether two categorical
        columns can be compared has to ask the arrays and not the types. Pandas
        does not have this split, because its dtype is heap allocated and holds
        the categories itself.

        Args:
            index: The dtype of the codes, which is what the column stores per
                row. Arrow allows any signed integer width here.
            ordered: Whether the categories have a meaningful order.

        Returns:
            The type.
        """
        return Self(
            TypeKind.DICTIONARY, index, TimeUnit.SECOND, TimeZone(), ordered
        )

    def __eq__(self, other: Self) -> Bool:
        """Compares two logical types.

        The unit and the zone are part of the comparison because they are part
        of the type. A microsecond column and a nanosecond column hold different
        instants for the same integer, and a New York column and a naive one are
        different questions with the same answer only by accident, so neither
        pair is allowed to compare equal and slip through a kernel that checks
        whether two operands match. The categories of a dictionary are the one
        part of a type that is not compared here, because they are not stored
        here, so two categorical columns over different categories do compare
        equal at this level and the array layer has to do the real check.

        Args:
            other: The type to compare against.

        Returns:
            True if the kind, the physical layout, the unit, the zone and the
            ordered flag all match.
        """
        return (
            self.kind == other.kind
            and self.physical == other.physical
            and self.unit == other.unit
            and self.zone == other.zone
            and self.ordered == other.ordered
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
        """Reports whether this type is about time rather than about a number.

        Two of the three are a point in time and the third is a length of one,
        which is a real difference and not one this predicate makes. What it is
        for is keeping all three out of the paths that would otherwise treat
        them as the integers they are stored as.

        Returns:
            True for timestamps, dates and durations.
        """
        return (
            self.kind == TypeKind.TIMESTAMP
            or self.kind == TypeKind.DATE
            or self.kind == TypeKind.DURATION
        )

    def is_dictionary(self) -> Bool:
        """Reports whether the stored values are positions into a category list.

        This is the predicate that keeps a categorical column out of the kernels
        that would otherwise take its codes for its values and cheerfully sum
        them.

        Returns:
            True for a dictionary type.
        """
        return self.kind == TypeKind.DICTIONARY

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

        The timestamp and the duration are spelled pandas' way rather than
        Arrow's, because this string is what `dtype` hands back and a user
        comparing it against `datetime64[ns]` or `timedelta64[ns]` is comparing
        against pandas. The date is the exception and is spelled the way Arrow
        spells it, since pandas on the numpy backend has no date dtype at all
        and calls the column `object`, which is a thing firepanda would be lying
        to say. The dictionary is spelled `category` and says nothing about its
        index width or its categories, which is also what pandas prints, and it
        is the reason two categorical columns that hold different things print
        the same dtype in both libraries.

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
        elif self.kind == TypeKind.DURATION:
            writer.write("timedelta64[", self.unit, "]")
        elif self.kind == TypeKind.DICTIONARY:
            writer.write("category")
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
    - a dictionary promotes with nothing, itself included.

    Args:
        a: The left operand type.
        b: The right operand type.

    Returns:
        The common type.

    Raises:
        If no common type exists, for example string with int64.
    """
    if a.is_dictionary() or b.is_dictionary():
        # Ahead of the equality check below, and deliberately, because two
        # dictionary types comparing equal does not mean the two columns hold
        # the same categories. What two categoricals combine to in pandas is
        # decided by their categories: matching ones keep the category type and
        # differing ones fall back to object, and the categories are not in the
        # type here. Answering that with half the information would be worse
        # than not answering, so this refuses every mixture, itself included,
        # and says which half is missing. A dictionary against anything else has
        # an answer in pandas too, which is to work against the decoded values,
        # and firepanda has no decode to do it with yet.
        raise Error(
            "no common type for "
            + String(a)
            + " and "
            + String(b)
            + ", because what two categoricals promote to depends on their"
            " categories and the categories are held by the column rather than"
            " by the type"
        )

    if a == b:
        return a
    if a.kind == TypeKind.NULL:
        return b
    if b.kind == TypeKind.NULL:
        return a

    if a.is_temporal() or b.is_temporal():
        # Two identical temporal types were returned above, so everything
        # arriving here is a mixture, and there are three of them. Two temporals
        # that disagree about the kind, the unit or the zone have an answer and
        # it is a conversion rather than a promotion, since reconciling a second
        # column with a nanosecond one means multiplying every value. A duration
        # against a number has an answer too and pandas gives it, because
        # scaling an elapsed time by a factor is a sensible thing to do, and
        # firepanda has no arithmetic on these columns to give it with yet. An
        # instant against a number has no answer at all. The three messages are
        # separate because a user who wrote `a - b` on two microsecond columns
        # in different zones has a different problem from one who wrote `a - 1`.
        if a.is_temporal() and b.is_temporal():
            raise Error(
                "no common type for "
                + String(a)
                + " and "
                + String(b)
                + ", because the two differ in kind, in unit or in time"
                " zone and"
                " reconciling them is a conversion rather than a promotion"
            )
        if a.kind == TypeKind.DURATION or b.kind == TypeKind.DURATION:
            raise Error(
                "no common type for "
                + String(a)
                + " and "
                + String(b)
                + ", because pandas scales an elapsed time by a number rather"
                " than promoting the two to something they have in common, and"
                " firepanda has no arithmetic on a duration column yet"
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
