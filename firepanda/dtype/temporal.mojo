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


def finer_unit(a: TimeUnit, b: TimeUnit) -> TimeUnit:
    """Returns whichever of two resolutions is the finer one.

    This is the rule pandas reconciles two temporal columns with, and it is the
    same rule for every pair and every operation: a second column against a
    nanosecond one is read in nanoseconds. It goes one way only because it is
    the only direction that loses nothing. Meeting in the middle, or taking the
    left operand's unit, would throw away digits the user still has.

    Args:
        a: One resolution.
        b: The other.

    Returns:
        The one with more of itself in a second.
    """
    return a if a.per_second() >= b.per_second() else b


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

    def _digit_at(self, i: Int) -> Int:
        """Reads one byte of the name as a decimal digit.

        Args:
            i: The position, which the caller keeps under `size`.

        Returns:
            The digit, or minus one if that byte is not one.
        """
        var byte = self.byte_at(i)
        if byte < UInt8(ord("0")) or byte > UInt8(ord("9")):
            return -1
        return Int(byte) - ord("0")

    def _two_digits(self, i: Int) -> Int:
        """Reads two bytes of the name as a two digit number.

        Args:
            i: The position of the first of them.

        Returns:
            The number, or minus one if either byte is not a digit or the name
            ends before the second one.
        """
        if i + 1 >= Int(self.size):
            return -1
        var high = self._digit_at(i)
        var low = self._digit_at(i + 1)
        if high < 0 or low < 0:
            return -1
        return high * 10 + low

    def _starts_with_utc(self) -> Bool:
        """Reports whether the name opens with the three letters of UTC.

        Returns:
            True for `UTC` in any case, which is how pandas prints a fixed
            offset it made itself, as in `UTC+05:30`.
        """
        if self.size < 3:
            return False
        return (
            (self.byte_at(0) | 32) == UInt8(ord("u"))
            and (self.byte_at(1) | 32) == UInt8(ord("t"))
            and (self.byte_at(2) | 32) == UInt8(ord("c"))
        )

    def fixed_offset(self) -> Optional[Int64]:
        """Returns how far ahead of UTC a zone that names its own offset is.

        A zone name is either a rule or a number. `America/New_York` is a rule,
        and reading a clock against it needs the IANA database, because the
        answer changes twice a year, changed on different days before 2007, and
        will change again when somebody legislates. `UTC` and `+05:30` are
        numbers. They state the whole answer in themselves, every instant in
        such a column is read against the same offset forever, and no database
        can tell you anything about them you cannot get from the name.

        That is the line this library draws while it has no database, and the
        line is a property of the name rather than a list of zones somebody has
        to keep up to date. The spellings accepted are `UTC`, the Arrow form
        `+HH:MM` and `-HH:MM` with the colon and the minutes both optional, and
        the `UTC+HH:MM` form pandas prints when it made the zone itself.

        `Etc/GMT+5` is deliberately not one of them. It is an IANA name, its
        sign is inverted against every other spelling in this list, and reading
        it as a number would be five hours wrong in the direction nobody checks.

        Returns:
            The offset in seconds, or nothing for a naive column or a name that
            is a rule.
        """
        var size = Int(self.size)
        if size == 0:
            return None

        var at = 0
        if self._starts_with_utc():
            if size == 3:
                return Int64(0)
            at = 3

        if at >= size:
            return None
        var sign = self.byte_at(at)
        if sign != UInt8(ord("+")) and sign != UInt8(ord("-")):
            return None
        at += 1

        var hours = self._two_digits(at)
        if hours < 0:
            return None
        at += 2

        var minutes = 0
        if at < size:
            if self.byte_at(at) == UInt8(ord(":")):
                at += 1
            minutes = self._two_digits(at)
            if minutes < 0:
                return None
            at += 2

        if at != size or hours > 23 or minutes > 59:
            return None

        var total = Int64(hours) * 3600 + Int64(minutes) * 60
        return -total if sign == UInt8(ord("-")) else total

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
