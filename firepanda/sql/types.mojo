"""The type a SQL expression has, and the names a query can call it.

This is DuckDB's type set and not firepanda's. The two do not line up and this
file is the reason there is a layer between them: DuckDB has `DECIMAL(p, s)`
with a derived width and `HUGEINT` at 128 bits, and firepanda's `LogicalType`
has neither. A front end that reused the engine's types would have to answer
`1.1 + 2.2` with a binary float, which is `3.3000000000000003` where DuckDB says
exactly `3.3`, and that is a wrong answer rather than a missing feature. See
docs/specs/sql/06-types-and-semantics.md.

What a type is here is an identifier and, for a decimal, a width and a scale.
That is all a scalar type needs, and every type in tier one is scalar. `LIST`,
`ARRAY`, `STRUCT` and `MAP` have identifiers so a query mentioning one is not a parse
failure, but a nested type also needs its element types, which needs an arena
the way the AST has one, and that arrives with the nested work rather than being
guessed at now. `type_name` gives the bare word for those instead of inventing an
element type to go with it.

Two measured things drive most of the code below.

**`typeof()` is observable and the corpus checks it.** A `sum` that gives the
right number under the wrong type name is a test failure, so the name a type
prints is part of the type and is written here rather than derived from
whatever the engine happens to store the column as.

**A spelling is not a type.** DuckDB accepts 82 spellings for 39 types, and the
mapping is not the one a C programmer would guess: `int8` is `BIGINT` because it
is eight bytes, `int1` is `TINYINT`, `float8` is `DOUBLE` and `float4` is
`FLOAT`. Getting that backwards silently narrows a column. The table below was
read out of `duckdb_types()` on 1.5 rather than out of the documentation, and
`int64` is in there as another way to write `BIGINT`.

`VARCHAR(3)` carries no semantics at all. The length parses and is discarded,
`typeof('a'::VARCHAR(3))` is `VARCHAR` and `'abcd'::VARCHAR(3)` is `'abcd'`. A
helpful truncation would be a wrong answer, so there is nowhere to put the
length and that is deliberate.
"""

from firepanda.dtype.logical import LogicalType

from .catalog import edit_distance, fold


comptime TYPE_INVALID: UInt8 = 0
"""Not a type. What a failed lookup gives back."""


comptime TYPE_NULL: UInt8 = 1
"""The type of the bare `NULL` literal, which casts to anything."""


comptime TYPE_BOOLEAN: UInt8 = 2
"""`BOOLEAN`."""


comptime TYPE_TINYINT: UInt8 = 3
"""`TINYINT`, eight bits and signed."""


comptime TYPE_SMALLINT: UInt8 = 4
"""`SMALLINT`, sixteen bits and signed."""


comptime TYPE_INTEGER: UInt8 = 5
"""`INTEGER`, thirty two bits and signed."""


comptime TYPE_BIGINT: UInt8 = 6
"""`BIGINT`, sixty four bits and signed."""


comptime TYPE_HUGEINT: UInt8 = 7
"""`HUGEINT`, one hundred and twenty eight bits and signed.

Not optional. `sum()` over any integer column is this type, so a `sum` that
returns `BIGINT` is a compatibility failure even when every value fits.
"""


comptime TYPE_UTINYINT: UInt8 = 8
"""`UTINYINT`."""


comptime TYPE_USMALLINT: UInt8 = 9
"""`USMALLINT`."""


comptime TYPE_UINTEGER: UInt8 = 10
"""`UINTEGER`."""


comptime TYPE_UBIGINT: UInt8 = 11
"""`UBIGINT`."""


comptime TYPE_UHUGEINT: UInt8 = 12
"""`UHUGEINT`."""


comptime TYPE_FLOAT: UInt8 = 13
"""`FLOAT`, thirty two bits."""


comptime TYPE_DOUBLE: UInt8 = 14
"""`DOUBLE`, sixty four bits."""


comptime TYPE_DECIMAL: UInt8 = 15
"""`DECIMAL(p, s)`, fixed point, the width and scale carried on the type.

The one type here whose identifier is not the whole story. An unsuffixed
decimal literal is this and not a double, which is why `1.1 + 2.2` is exactly
`3.3`, and the width and scale of a result are derived from the operands rather
than fixed.
"""


comptime TYPE_VARCHAR: UInt8 = 16
"""`VARCHAR`, UTF-8, no length semantics."""


comptime TYPE_BLOB: UInt8 = 17
"""`BLOB`."""


comptime TYPE_DATE: UInt8 = 18
"""`DATE`, a day count."""


comptime TYPE_TIME: UInt8 = 19
"""`TIME`, microseconds since midnight, no zone."""


comptime TYPE_TIME_TZ: UInt8 = 20
"""`TIME WITH TIME ZONE`."""


comptime TYPE_TIMESTAMP: UInt8 = 21
"""`TIMESTAMP`, microseconds, no zone.

Microsecond and truncating rather than rounding, where pandas defaults to
nanoseconds, which is the one conversion in the front door with a documented
loss.
"""


comptime TYPE_TIMESTAMP_TZ: UInt8 = 22
"""`TIMESTAMP WITH TIME ZONE`, which is what `now()` gives back."""


comptime TYPE_TIMESTAMP_S: UInt8 = 23
"""`TIMESTAMP_S`, second precision."""


comptime TYPE_TIMESTAMP_MS: UInt8 = 24
"""`TIMESTAMP_MS`, millisecond precision."""


comptime TYPE_TIMESTAMP_NS: UInt8 = 25
"""`TIMESTAMP_NS`, nanosecond precision."""


comptime TYPE_TIME_NS: UInt8 = 26
"""`TIME_NS`."""


comptime TYPE_INTERVAL: UInt8 = 27
"""`INTERVAL`, a triple of months, days and microseconds.

Not a duration. Months and days are not fixed length, which is why an interval
cannot be normalized and why interval arithmetic does not commute across a
daylight saving boundary.
"""


comptime TYPE_LIST: UInt8 = 28
"""`LIST`. The element type is not carried yet."""


comptime TYPE_ARRAY: UInt8 = 29
"""`ARRAY`, a list with a fixed length, which is a different type from `LIST`.

`typeof([1, 2, 3]::INT[3])` is `INTEGER[3]` and not `INTEGER[]`, so the two
cannot share an identifier.
"""


comptime TYPE_STRUCT: UInt8 = 30
"""`STRUCT`. The field types are not carried yet."""


comptime TYPE_MAP: UInt8 = 31
"""`MAP`. The key and value types are not carried yet."""


comptime TYPE_UNION: UInt8 = 32
"""`UNION`, out of scope, named so a query saying it is not a parse failure."""


comptime TYPE_ENUM: UInt8 = 33
"""`ENUM`, out of scope."""


comptime TYPE_UUID: UInt8 = 34
"""`UUID`, out of scope."""


comptime TYPE_BIT: UInt8 = 35
"""`BIT`, out of scope."""


comptime TYPE_BIGNUM: UInt8 = 36
"""`BIGNUM`, out of scope."""


comptime TYPE_VARIANT: UInt8 = 37
"""`VARIANT`, out of scope."""


comptime TYPE_GEOMETRY: UInt8 = 38
"""`GEOMETRY`, out of scope."""


comptime TYPE_TYPE: UInt8 = 39
"""`TYPE`, which is the type of a type, and is out of scope."""


comptime TYPE_COUNT: UInt8 = 40
"""One past the last identifier, so a table can be sized from it."""


comptime DECIMAL_MAX_WIDTH: UInt8 = 38
"""The widest `DECIMAL` there is, which is what fits in 128 bits."""


comptime DECIMAL_DEFAULT_WIDTH: UInt8 = 18
"""What a bare `DECIMAL` means. Measured: `typeof(1::DECIMAL)` is
`DECIMAL(18,3)`."""


comptime DECIMAL_DEFAULT_SCALE: UInt8 = 3
"""What a bare `DECIMAL` scales to."""


@fieldwise_init
struct SqlType(Copyable, Equatable, ImplicitlyCopyable, Movable, Writable):
    """One SQL type: an identifier, and a width and scale for a decimal."""

    var id: UInt8
    """One of the `TYPE_` constants."""

    var width: UInt8
    """A decimal's total digits. Zero for everything else."""

    var scale: UInt8
    """A decimal's digits after the point. Zero for everything else."""

    def __init__(out self, id: UInt8):
        """A type with nothing else to carry.

        Args:
            id: One of the `TYPE_` constants, not `TYPE_DECIMAL`.
        """
        self.id = id
        self.width = 0
        self.scale = 0

    def __eq__(self, other: Self) -> Bool:
        """Whether two types are the same type.

        A decimal is the same as another decimal only at the same width and
        scale, since `DECIMAL(4,2)` and `DECIMAL(5,2)` hold different values and
        the corpus can see the difference through `typeof()`.

        Args:
            other: The other type.

        Returns:
            True if they are equal.
        """
        return (
            self.id == other.id
            and self.width == other.width
            and self.scale == other.scale
        )

    def __ne__(self, other: Self) -> Bool:
        """Whether two types differ.

        Args:
            other: The other type.

        Returns:
            True if they are not equal.
        """
        return not self == other

    def write_to[W: Writer](self, mut writer: W):
        """Writes the type the way `typeof()` prints it.

        Parameters:
            W: The writer's type.

        Args:
            writer: Where it goes.
        """
        writer.write(self.name())

    def name(self) -> String:
        """What `typeof()` calls this type.

        Returns:
            The canonical spelling, with the width and scale on a decimal.
        """
        if self.id == TYPE_DECIMAL:
            return String("DECIMAL(", self.width, ",", self.scale, ")")
        return String(type_name(self.id))

    def is_integer(self) -> Bool:
        """Whether this is one of the ten integer types.

        Returns:
            True for the signed and unsigned integers.
        """
        return self.id >= TYPE_TINYINT and self.id <= TYPE_UHUGEINT

    def is_signed(self) -> Bool:
        """Whether this is one of the five signed integer types.

        Returns:
            True for `TINYINT` through `HUGEINT`.
        """
        return self.id >= TYPE_TINYINT and self.id <= TYPE_HUGEINT

    def is_float(self) -> Bool:
        """Whether this is `FLOAT` or `DOUBLE`.

        Returns:
            True for the two binary floating point types.
        """
        return self.id == TYPE_FLOAT or self.id == TYPE_DOUBLE

    def is_decimal(self) -> Bool:
        """Whether this is a fixed point decimal.

        Returns:
            True for `DECIMAL`.
        """
        return self.id == TYPE_DECIMAL

    def is_numeric(self) -> Bool:
        """Whether arithmetic applies to this type.

        Returns:
            True for the integers, the floats and a decimal.
        """
        return self.is_integer() or self.is_float() or self.is_decimal()

    def is_temporal(self) -> Bool:
        """Whether this is a date, a time or a timestamp.

        `INTERVAL` is not one of these. It is what the difference of two of
        them is, which is a different thing from a point on a line.

        Returns:
            True for the nine date, time and timestamp types.
        """
        return self.id >= TYPE_DATE and self.id <= TYPE_TIME_NS

    def bits(self) -> Int:
        """How wide an integer type is.

        Returns:
            The width in bits, or zero for a type that is not an integer.
        """
        if self.id == TYPE_TINYINT or self.id == TYPE_UTINYINT:
            return 8
        if self.id == TYPE_SMALLINT or self.id == TYPE_USMALLINT:
            return 16
        if self.id == TYPE_INTEGER or self.id == TYPE_UINTEGER:
            return 32
        if self.id == TYPE_BIGINT or self.id == TYPE_UBIGINT:
            return 64
        if self.id == TYPE_HUGEINT or self.id == TYPE_UHUGEINT:
            return 128
        return 0


comptime NULL = SqlType(TYPE_NULL, 0, 0)
"""The bare `NULL` literal's type."""


comptime BOOLEAN = SqlType(TYPE_BOOLEAN, 0, 0)
"""`BOOLEAN`."""


comptime TINYINT = SqlType(TYPE_TINYINT, 0, 0)
"""`TINYINT`."""


comptime SMALLINT = SqlType(TYPE_SMALLINT, 0, 0)
"""`SMALLINT`."""


comptime INTEGER = SqlType(TYPE_INTEGER, 0, 0)
"""`INTEGER`."""


comptime BIGINT = SqlType(TYPE_BIGINT, 0, 0)
"""`BIGINT`."""


comptime HUGEINT = SqlType(TYPE_HUGEINT, 0, 0)
"""`HUGEINT`."""


comptime FLOAT = SqlType(TYPE_FLOAT, 0, 0)
"""`FLOAT`."""


comptime DOUBLE = SqlType(TYPE_DOUBLE, 0, 0)
"""`DOUBLE`."""


comptime VARCHAR = SqlType(TYPE_VARCHAR, 0, 0)
"""`VARCHAR`."""


comptime BLOB = SqlType(TYPE_BLOB, 0, 0)
"""`BLOB`."""


comptime DATE = SqlType(TYPE_DATE, 0, 0)
"""`DATE`."""


comptime TIME = SqlType(TYPE_TIME, 0, 0)
"""`TIME`."""


comptime TIMESTAMP = SqlType(TYPE_TIMESTAMP, 0, 0)
"""`TIMESTAMP`."""


comptime TIMESTAMP_TZ = SqlType(TYPE_TIMESTAMP_TZ, 0, 0)
"""`TIMESTAMP WITH TIME ZONE`."""


comptime INTERVAL = SqlType(TYPE_INTERVAL, 0, 0)
"""`INTERVAL`."""


comptime INVALID = SqlType(TYPE_INVALID, 0, 0)
"""What a spelling nobody recognizes gives back."""


def decimal(width: UInt8, scale: UInt8) raises -> SqlType:
    """Builds a `DECIMAL(width, scale)`, checking it the way DuckDB does.

    Args:
        width: The total digits, one to thirty eight.
        scale: The digits after the point, no more than the width.

    Returns:
        The type.

    Raises:
        Error: DuckDB's own message, on a width outside one to thirty eight or
            a scale wider than the width.
    """
    if width == 0 or width > DECIMAL_MAX_WIDTH:
        raise Error("Binder Error: DECIMAL type width must be between 1 and 38")
    if scale > width:
        raise Error(
            "Binder Error: DECIMAL type scale cannot be greater than width"
        )
    return SqlType(TYPE_DECIMAL, width, scale)


def type_name(id: UInt8) -> StaticString:
    """The canonical spelling of an identifier.

    A decimal's name has its width and scale in it, so this gives the bare word
    and `SqlType.name` is what a caller wants.

    Args:
        id: One of the `TYPE_` constants.

    Returns:
        What `typeof()` prints, or `INVALID` for an identifier that is not one.
    """
    if id >= TYPE_COUNT:
        return "INVALID"
    return type_names()[Int(id)]


def type_names() -> List[StaticString]:
    """The name each identifier prints under.

    `typeof(NULL)` really does print `"NULL"` with the quotes in it, which is a
    DuckDB oddity rather than a typo here.

    Returns:
        One canonical name per identifier, in identifier order.
    """
    return [
        "INVALID",
        '"NULL"',
        "BOOLEAN",
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "UHUGEINT",
        "FLOAT",
        "DOUBLE",
        "DECIMAL",
        "VARCHAR",
        "BLOB",
        "DATE",
        "TIME",
        "TIME WITH TIME ZONE",
        "TIMESTAMP",
        "TIMESTAMP WITH TIME ZONE",
        "TIMESTAMP_S",
        "TIMESTAMP_MS",
        "TIMESTAMP_NS",
        "TIME_NS",
        "INTERVAL",
        "LIST",
        "ARRAY",
        "STRUCT",
        "MAP",
        "UNION",
        "ENUM",
        "UUID",
        "BIT",
        "BIGNUM",
        "VARIANT",
        "GEOMETRY",
        "TYPE",
    ]


@fieldwise_init
struct Spelling(Copyable, ImplicitlyCopyable, Movable):
    """One way a query is allowed to write a type name."""

    var text: StaticString
    """The spelling, folded."""

    var id: UInt8
    """What it means."""


def spellings() -> List[Spelling]:
    """Every spelling DuckDB accepts, folded, read out of `duckdb_types()` on 1.5.

    Seventy nine of them for thirty nine types. Grouped by what they mean rather
    than sorted, so the aliases of a type read together and a missing one is
    visible.

    What each spelling means, in the same order.

    Two entries here are the reason the whole table is a measurement rather than a
    guess. `int8` is `BIGINT`, because DuckDB counts bytes where a Mojo or Rust
    programmer reads bits, and `int1` is `TINYINT` for the same reason. Reading
    either one the other way round narrows a column without saying anything.

    Returns:
        Every spelling and what it means.
    """
    return [
        Spelling("bigint", TYPE_BIGINT),
        Spelling("int64", TYPE_BIGINT),
        Spelling("int8", TYPE_BIGINT),
        Spelling("long", TYPE_BIGINT),
        Spelling("oid", TYPE_BIGINT),
        Spelling("bignum", TYPE_BIGNUM),
        Spelling("varint", TYPE_BIGNUM),
        Spelling("bit", TYPE_BIT),
        Spelling("bitstring", TYPE_BIT),
        Spelling("blob", TYPE_BLOB),
        Spelling("binary", TYPE_BLOB),
        Spelling("bytea", TYPE_BLOB),
        Spelling("varbinary", TYPE_BLOB),
        Spelling("boolean", TYPE_BOOLEAN),
        Spelling("bool", TYPE_BOOLEAN),
        Spelling("logical", TYPE_BOOLEAN),
        Spelling("date", TYPE_DATE),
        Spelling("decimal", TYPE_DECIMAL),
        Spelling("dec", TYPE_DECIMAL),
        Spelling("numeric", TYPE_DECIMAL),
        Spelling("double", TYPE_DOUBLE),
        Spelling("float8", TYPE_DOUBLE),
        Spelling("enum", TYPE_ENUM),
        Spelling("float", TYPE_FLOAT),
        Spelling("float4", TYPE_FLOAT),
        Spelling("real", TYPE_FLOAT),
        Spelling("geometry", TYPE_GEOMETRY),
        Spelling("hugeint", TYPE_HUGEINT),
        Spelling("int128", TYPE_HUGEINT),
        Spelling("integer", TYPE_INTEGER),
        Spelling("int", TYPE_INTEGER),
        Spelling("int32", TYPE_INTEGER),
        Spelling("int4", TYPE_INTEGER),
        Spelling("integral", TYPE_INTEGER),
        Spelling("signed", TYPE_INTEGER),
        Spelling("interval", TYPE_INTERVAL),
        Spelling("list", TYPE_LIST),
        Spelling("array", TYPE_ARRAY),
        Spelling("map", TYPE_MAP),
        Spelling("null", TYPE_NULL),
        Spelling("smallint", TYPE_SMALLINT),
        Spelling("int16", TYPE_SMALLINT),
        Spelling("int2", TYPE_SMALLINT),
        Spelling("short", TYPE_SMALLINT),
        Spelling("struct", TYPE_STRUCT),
        Spelling("row", TYPE_STRUCT),
        Spelling("time", TYPE_TIME),
        Spelling("time with time zone", TYPE_TIME_TZ),
        Spelling("timetz", TYPE_TIME_TZ),
        Spelling("timestamp", TYPE_TIMESTAMP),
        Spelling("datetime", TYPE_TIMESTAMP),
        Spelling("timestamp_us", TYPE_TIMESTAMP),
        Spelling("timestamp with time zone", TYPE_TIMESTAMP_TZ),
        Spelling("timestamptz", TYPE_TIMESTAMP_TZ),
        Spelling("timestamp_ms", TYPE_TIMESTAMP_MS),
        Spelling("timestamp_ns", TYPE_TIMESTAMP_NS),
        Spelling("timestamp_s", TYPE_TIMESTAMP_S),
        Spelling("time_ns", TYPE_TIME_NS),
        Spelling("tinyint", TYPE_TINYINT),
        Spelling("int1", TYPE_TINYINT),
        Spelling("type", TYPE_TYPE),
        Spelling("ubigint", TYPE_UBIGINT),
        Spelling("uint64", TYPE_UBIGINT),
        Spelling("uhugeint", TYPE_UHUGEINT),
        Spelling("uint128", TYPE_UHUGEINT),
        Spelling("uinteger", TYPE_UINTEGER),
        Spelling("uint32", TYPE_UINTEGER),
        Spelling("usmallint", TYPE_USMALLINT),
        Spelling("uint16", TYPE_USMALLINT),
        Spelling("utinyint", TYPE_UTINYINT),
        Spelling("uint8", TYPE_UTINYINT),
        Spelling("union", TYPE_UNION),
        Spelling("uuid", TYPE_UUID),
        Spelling("guid", TYPE_UUID),
        Spelling("varchar", TYPE_VARCHAR),
        Spelling("bpchar", TYPE_VARCHAR),
        Spelling("char", TYPE_VARCHAR),
        Spelling("json", TYPE_VARCHAR),
        Spelling("nvarchar", TYPE_VARCHAR),
        Spelling("string", TYPE_VARCHAR),
        Spelling("text", TYPE_VARCHAR),
        Spelling("variant", TYPE_VARIANT),
    ]


def type_for(spelling: StringSlice) -> UInt8:
    """Looks up what a written type name means.

    Args:
        spelling: The name as the query wrote it, in any case.

    Returns:
        The identifier, or `TYPE_INVALID` if nothing spells that.
    """
    var key = fold(spelling)
    for entry in spellings():
        if key == entry.text:
            return entry.id
    return TYPE_INVALID


def nearest_type(spelling: StringSlice) -> StaticString:
    """The spelling closest to one nobody recognizes, if anything is close.

    DuckDB always names something here, because it searches every type it
    knows and takes the best of them however bad the best is: `1::nosuchtype`
    answers `Did you mean "struct"?`, which is not a guess anybody can use.
    The rule taken instead is the catalog's, that a name within half its own
    length of a real one is a typo and anything further away is not, so a
    suggestion that appears is worth reading.

    Args:
        spelling: The name the query wrote, in any case.

    Returns:
        The closest spelling, or an empty string if none of them are close.
    """
    var key = fold(spelling)
    var best = StaticString("")
    var closest = key.byte_length() + 1
    for entry in spellings():
        var shorter = min(key.byte_length(), entry.text.byte_length())
        var limit = shorter // 2 + 1
        var distance = edit_distance(key, entry.text, limit)
        if distance < limit and distance < closest:
            closest = distance
            best = entry.text
    return best


def _no_such_type(spelling: StringSlice) -> String:
    """DuckDB's message for a type name that is not one.

    Args:
        spelling: The name the query wrote.

    Returns:
        The message, with a suggestion after it when there is one to make.
    """
    var message = String(
        "Catalog Error: Type with name ", spelling, " does not exist!"
    )
    var near = nearest_type(spelling)
    if near.byte_length() > 0:
        message += String('\nDid you mean "', near, '"?')
    return message


def parse_type(text: StringSlice) raises -> SqlType:
    """Reads a whole type as a query writes it, arguments and all.

    `DECIMAL(9, 2)` keeps both numbers. `DECIMAL(9)` takes a zero scale, and a
    bare `DECIMAL` is `DECIMAL(18,3)`. `VARCHAR(3)` parses and throws the
    length away, because the length carries no semantics in DuckDB and a
    helpful truncation would be a wrong answer.

    Args:
        text: The type as written.

    Returns:
        The type.

    Raises:
        Error: If nothing spells that, or if a decimal's width or scale is out
            of range.
    """
    var trimmed = text.strip()
    var open = trimmed.find("(")
    if open == NOT_A_PAREN:
        var id = type_for(trimmed)
        if id == TYPE_INVALID:
            raise Error(_no_such_type(trimmed))
        if id == TYPE_DECIMAL:
            return SqlType(
                TYPE_DECIMAL, DECIMAL_DEFAULT_WIDTH, DECIMAL_DEFAULT_SCALE
            )
        return SqlType(id)
    # Stripped, because the type a query wrote reaches here with its tokens
    # joined by single spaces, so `DECIMAL(9,2)` arrives as `DECIMAL ( 9 , 2 )`
    # and the name has a space after it.
    var head = trimmed[byte=0:open].strip()
    var id = type_for(head)
    if id == TYPE_INVALID:
        raise Error(_no_such_type(head))
    if id != TYPE_DECIMAL:
        # A length on anything else parses and is dropped, which is what
        # VARCHAR(3) does and is the whole of what it does.
        return SqlType(id)
    var close = trimmed.rfind(")")
    if close == NOT_A_PAREN:
        raise Error("Parser Error: a type argument list has no closing bracket")
    var arguments = trimmed[byte = open + 1 : close]
    var comma = arguments.find(",")
    if comma == NOT_A_PAREN:
        return decimal(_digits(arguments), 0)
    return decimal(
        _digits(arguments[byte=0:comma]),
        _digits(arguments[byte = comma + 1 : arguments.byte_length()]),
    )


comptime NOT_A_PAREN: Int = -1
"""What `find` gives back when there is no bracket."""


def _digits(text: StringSlice) raises -> UInt8:
    """Reads a small whole number out of a type argument.

    Args:
        text: The argument, with whatever spaces the query put around it.

    Returns:
        The number.

    Raises:
        Error: If it is not digits, or if it is too big to be a width.
    """
    var trimmed = text.strip()
    if trimmed.byte_length() == 0:
        raise Error("Parser Error: a type argument is empty")
    var value = 0
    for byte in trimmed.as_bytes():
        if byte < UInt8(48) or byte > UInt8(57):
            raise Error(
                String(
                    "Parser Error: ",
                    trimmed,
                    " is not a number a type can be given",
                )
            )
        value = value * 10 + Int(byte - UInt8(48))
        if value > 255:
            raise Error(
                "Binder Error: DECIMAL type width must be between 1 and 38"
            )
    return UInt8(value)


def engine_type(type: SqlType) raises -> LogicalType:
    """The engine type a SQL type becomes, where the two sets agree.

    Twelve of the thirty nine cross over with nothing lost: `BOOLEAN`, the eight
    fixed width integers, `FLOAT`, `DOUBLE` and `VARCHAR`. Those are the types a
    `CAST` runs today. The rest are refused, and the refusals split into two
    kinds that are worth keeping apart because they end at different times.

    `HUGEINT`, `UHUGEINT` and `DECIMAL` have no engine type at all. `LogicalType`
    has no 128 bit integer and no exact decimal, so there is nothing to name on
    the other side. Reaching for the nearest thing would be worse than the
    refusal: a decimal read as a double answers `3.3000000000000003` where
    DuckDB answers `3.3`, which is a wrong answer rather than a missing feature.

    `DATE`, the timestamps and the times do have engine types, and what is
    missing is the conversion rather than the type. `cast_any` converts a column
    to the physical layout its target sits on, so a cast to `DATE` would come
    back an int32 holding the source numbers rather than the days they stand
    for, and the column would answer to a date's name while holding something
    else. These wait on a cast that converts values.

    Args:
        type: The SQL type, as `parse_type` read it.

    Returns:
        The engine type it becomes.

    Raises:
        Error: If the engine has no type for it, or has one that no cast reaches
            yet.
    """
    if type.id == TYPE_BOOLEAN:
        return LogicalType.BOOL
    if type.id == TYPE_TINYINT:
        return LogicalType.INT8
    if type.id == TYPE_SMALLINT:
        return LogicalType.INT16
    if type.id == TYPE_INTEGER:
        return LogicalType.INT32
    if type.id == TYPE_BIGINT:
        return LogicalType.INT64
    if type.id == TYPE_UTINYINT:
        return LogicalType.UINT8
    if type.id == TYPE_USMALLINT:
        return LogicalType.UINT16
    if type.id == TYPE_UINTEGER:
        return LogicalType.UINT32
    if type.id == TYPE_UBIGINT:
        return LogicalType.UINT64
    if type.id == TYPE_FLOAT:
        return LogicalType.FLOAT32
    if type.id == TYPE_DOUBLE:
        return LogicalType.FLOAT64
    if type.id == TYPE_VARCHAR:
        return LogicalType.STRING

    if type.id == TYPE_HUGEINT or type.id == TYPE_UHUGEINT:
        raise Error(
            String(
                "firepanda has no engine type for ",
                type.name(),
                ", because the engine's integers stop at 64 bits",
            )
        )
    if type.id == TYPE_DECIMAL:
        raise Error(
            String(
                "firepanda has no engine type for ",
                type.name(),
                (
                    ", because the engine has no exact decimal and a double"
                    " would answer 3.3000000000000003 where DuckDB answers 3.3"
                ),
            )
        )
    if type.is_temporal():
        raise Error(
            String(
                "firepanda does not cast to ",
                type.name(),
                (
                    " yet, because the cast converts to the layout a temporal"
                    " type sits on and would hand back the integer underneath"
                    " rather than the "
                ),
                type.name(),
            )
        )
    if type.id == TYPE_BLOB:
        raise Error(
            "firepanda does not cast to BLOB yet, because the cast writes the"
            " bytes out as text and the column would come back a VARCHAR"
        )
    raise Error(
        String(
            "firepanda does not cast to ",
            type.name(),
            " yet, because there is no engine type that holds one",
        )
    )
