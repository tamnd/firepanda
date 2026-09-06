"""What a timestamp column carries besides its integers.

An Arrow timestamp is an int64 and nothing else. What that integer means depends
on two facts stored beside it: how many of them make one second, and which wall
clock the instant is read against. pandas keeps both in the dtype, which is why
`datetime64[us]` and `datetime64[ns]` are two dtypes rather than one dtype at two
settings, and why a library that assumes nanoseconds is wrong about every file
written by something that did not.

Both types here are values with no allocation in them. `TimeZone` holds its name
in thirty two bytes of SIMD rather than in a `String` because `LogicalType`
embeds one, a `LogicalType` is copied everywhere a column type is mentioned, and
a type that allocates when it is copied would put a malloc on paths that are a
register move today. Thirty two is not a guess. The longest name in the IANA
database is `America/Argentina/ComodRivadavia`, which is thirty two characters,
so a name that does not fit is one Arrow should not have produced, and it is
refused at the boundary rather than truncated into a different zone.

The thirty two bytes are two SIMD halves of sixteen and not one vector of thirty
two, and that is a workaround for a toolchain bug rather than a design. A
`SIMD[DType.uint8, 32]` is aligned to thirty two bytes, that alignment travels up
through `LogicalType` into `AnyArray`, and an `AnyArray` passed by value into a
function then arrives with the wrong bytes in it. What that looked like was
`Index.__init__` reading a length of zero from a column of five, an `Optional`
that held nothing reading as though it held something, and two segmentation
faults in the compiler, none of which pointed anywhere near a time zone. Two
halves at sixteen byte alignment hold the same thirty two bytes and miscompile
into nothing. Anything here that reads a byte by its position across both halves
is paying for that and would be one line if the bug were fixed.

See docs/specs/03-dtype-dispatch.md.
"""

comptime ZONE_CAPACITY = 32
"""The longest zone name that fits, which is the longest one there is."""

comptime HALF_CAPACITY = 16
"""How much of it each half holds."""


@fieldwise_init
struct TimeUnit(Equatable, ImplicitlyCopyable, Movable, Writable):
    """How many of a temporal column's integers make one second."""

    var code: UInt8
    """The Arrow `TimeUnit` enumerator, which this matches on purpose so that
    the reader and the writer can pass it through without a lookup."""

    comptime SECOND = Self(0)
    comptime MILLI = Self(1)
    comptime MICRO = Self(2)
    comptime NANO = Self(3)

    def __eq__(self, other: Self) -> Bool:
        """Compares two units.

        Args:
            other: The unit to compare against.

        Returns:
            True if the units are the same.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two units for inequality.

        Args:
            other: The unit to compare against.

        Returns:
            True if the units differ.
        """
        return self.code != other.code

    def per_second(self) -> Int64:
        """Returns how many of this unit there are in a second.

        Returns:
            1, 1000, 1000000 or 1000000000.
        """
        if self == Self.SECOND:
            return 1
        if self == Self.MILLI:
            return 1_000
        if self == Self.MICRO:
            return 1_000_000
        return 1_000_000_000

    def code_letter(self) -> StaticString:
        """Returns the letter Arrow's format strings use for this unit.

        Not the same spelling as `write_to`, and the difference is the one that
        matters: a millisecond is `ms` in a pandas dtype and `m` in a format
        string, so `tsm:` and `datetime64[ms]` are the same type spelled two
        ways and neither spelling can be used for the other.

        Returns:
            One of `s`, `m`, `u` and `n`.
        """
        if self == Self.SECOND:
            return "s"
        if self == Self.MILLI:
            return "m"
        if self == Self.MICRO:
            return "u"
        return "n"

    def write_to(self, mut writer: Some[Writer]):
        """Writes the unit the way pandas spells it inside a dtype.

        Args:
            writer: The destination.
        """
        if self == Self.SECOND:
            writer.write("s")
        elif self == Self.MILLI:
            writer.write("ms")
        elif self == Self.MICRO:
            writer.write("us")
        else:
            writer.write("ns")


def unit_for_code(code: Int) raises -> TimeUnit:
    """Turns an Arrow `TimeUnit` enumerator into a unit.

    Args:
        code: The enumerator, as it appears in the schema message.

    Returns:
        The unit.

    Raises:
        Error: If the enumerator is not one of the four Arrow defines.
    """
    if code < 0 or code > 3:
        raise Error(
            String(
                "arrow: time unit ",
                code,
                (
                    " is not one of the four Arrow has, which are second,"
                    " millisecond, microsecond and nanosecond"
                ),
            )
        )
    return TimeUnit(UInt8(code))


@fieldwise_init
struct TimeZone(Equatable, ImplicitlyCopyable, Movable, Writable):
    """The wall clock a column of instants is read against, or none at all."""

    var lo: SIMD[DType.uint8, HALF_CAPACITY]
    """The first sixteen bytes of the name, left aligned, the rest of it zero."""

    var hi: SIMD[DType.uint8, HALF_CAPACITY]
    """The last sixteen, on the same terms. Most names do not reach it."""

    var size: UInt8
    """How many of those bytes are the name. Zero means the column is naive."""

    def __init__(out self):
        """Constructs the naive zone, which is the absence of one."""
        self.lo = SIMD[DType.uint8, HALF_CAPACITY](0)
        self.hi = SIMD[DType.uint8, HALF_CAPACITY](0)
        self.size = 0

    def __init__(out self, name: StringSlice) raises:
        """Constructs a zone from its name.

        Args:
            name: The IANA name, or the fixed offset spelling Arrow also allows.

        Raises:
            Error: If the name is longer than any real zone name is.
        """
        var raw = name.as_bytes()
        if len(raw) > ZONE_CAPACITY:
            raise Error(
                String(
                    "arrow: time zone '",
                    name,
                    "' is ",
                    len(raw),
                    " bytes, and no zone name is longer than ",
                    ZONE_CAPACITY,
                )
            )
        self.lo = SIMD[DType.uint8, HALF_CAPACITY](0)
        self.hi = SIMD[DType.uint8, HALF_CAPACITY](0)
        for i in range(len(raw)):
            if i < HALF_CAPACITY:
                self.lo[i] = raw[i]
            else:
                self.hi[i - HALF_CAPACITY] = raw[i]
        self.size = UInt8(len(raw))

    def byte_at(self, i: Int) -> UInt8:
        """Returns one byte of the name by its position in the whole name.

        Args:
            i: The position, which the caller is trusted to keep under `size`.

        Returns:
            The byte.
        """
        if i < HALF_CAPACITY:
            return self.lo[i]
        return self.hi[i - HALF_CAPACITY]

    def is_naive(self) -> Bool:
        """Reports whether the column carries no zone.

        Returns:
            True when there is no zone, which is a different thing from UTC.
        """
        return self.size == 0

    def __eq__(self, other: Self) -> Bool:
        """Compares two zones.

        The comparison is on the name and not on the offset the name resolves
        to, because two zones that agree today part company on a transition and
        a column is not allowed to change type in July.

        Args:
            other: The zone to compare against.

        Returns:
            True if both are naive or both carry the same name.
        """
        return (
            self.size == other.size
            and Bool(self.lo.eq(other.lo).reduce_and())
            and Bool(self.hi.eq(other.hi).reduce_and())
        )

    def __ne__(self, other: Self) -> Bool:
        """Compares two zones for inequality.

        Args:
            other: The zone to compare against.

        Returns:
            True if the zones differ.
        """
        return not (self == other)

    def write_to(self, mut writer: Some[Writer]):
        """Writes the zone name, and nothing at all when there is none.

        Args:
            writer: The destination.
        """
        for i in range(Int(self.size)):
            writer.write(chr(Int(self.byte_at(i))))
