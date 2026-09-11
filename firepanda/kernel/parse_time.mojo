"""Reading a column of text as a column of instants.

This is the inbound half of `temporal.mojo`. That file turns an instant into
text through a format string, and this one turns text back into an instant
through the same format string, taken apart by the same `parse_format` into the
same list of steps. Sharing the step list is not a tidiness argument. It is what
makes a format that renders and a format that reads agree by construction rather
than by two people keeping two tables in step, and it is why the fuzz twin for
this file is a round trip: render a random instant, read it back, and the two
answers have to be the same number.

There is no SIMD version of any of this and there is nothing here to compare a
vector answer against. Reading text is a walk with a cursor whose next step
depends on the last byte, and a version of that with a vector unit in it would
be a different program rather than a faster one. So the package rule that every
kernel has a scalar twin is met the other way round: this file is the scalar
one, and the thing it is checked against is the renderer next door.

The one place where pandas and this part company is guessing. pandas looks at
the first value, guesses a format for the whole column, and its guesser knows
about `01/02/2026` and decides on its own which of the two numbers is the month.
This guesses ISO 8601 and nothing else, and refuses anything it does not
recognise with a message asking for a format. A wrong guess is a column of wrong
instants that nothing reports and that the caller finds out about from a chart.
A refusal is one sentence and one argument.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.kernel.temporal import STEP_BYTE, STEP_SLICE, Step, parse_format


comptime SECONDS_PER_DAY = Int64(86_400)
"""How many seconds are in a day on the clock this library keeps, which has no
leap seconds in it because neither Arrow nor pandas has any either."""

comptime MAX_FRACTION_DIGITS = 9
"""How many digits may follow a decimal point. Nine is a nanosecond, which is
the finest resolution Arrow has, so a tenth digit is a value this library has
nowhere to put rather than a precision it rounds away."""

comptime MOST_SECONDS = Int64.MAX
"""The largest count an Int64 holds, which is what every range check here is
against. Divided by the units in a second it gives the range of each column
type: about 292 billion years for seconds, 292 thousand for microseconds and
just 1677 to 2262 for nanoseconds, which is the one that matters."""


def days_from_civil(year: Int64, month: Int64, day: Int64) -> Int64:
    """Turns a calendar date into a count of days since 1970-01-01.

    Hinnant's algorithm, the exact inverse of `civil_from_days` next door, and
    written here rather than there because that one converts a register full of
    day numbers at a time and this one runs once per row of text with three
    separate integers already in hand.

    It is written with floor division throughout, where the published version
    corrects afterwards for a truncating one, because Mojo's `//` rounds down.
    That is the same reason `civil_from_days` needs no correction either, and it
    is the only difference between the two spellings.

    Args:
        year: The calendar year, negative before the year zero.
        month: The month from 1 to 12, already checked by the caller.
        day: The day of the month from 1, already checked by the caller.

    Returns:
        Days since the epoch, negative before it.
    """
    # The year is moved to start in March, so that the leap day is the last day
    # of it and every month in front of it has a length following the pattern
    # the one division below recovers. January and February belong to the year
    # before under that arrangement, which is what this subtraction does.
    var shifted = year - Int64(1) if month <= 2 else year
    var era = shifted // 400
    var year_of_era = shifted - era * 400

    var march_month = month + 9 if month <= 2 else month - 3
    var day_of_march_year = (153 * march_month + 2) // 5 + day - 1
    var day_of_era = (
        year_of_era * 365
        + year_of_era // 4
        - year_of_era // 100
        + day_of_march_year
    )
    return era * 146_097 + day_of_era - 719_468


def days_in_month(year: Int64, month: Int64) -> Int64:
    """Returns how many days a month has, which needs the year for February.

    Args:
        year: The calendar year.
        month: The month from 1 to 12.

    Returns:
        28, 29, 30 or 31.
    """
    if month == 2:
        var leap = (year % 4) == 0 and ((year % 100) != 0 or (year % 400) == 0)
        return Int64(29) if leap else Int64(28)
    if month == 4 or month == 6 or month == 9 or month == 11:
        return Int64(30)
    return Int64(31)


struct Digits(Copyable, ImplicitlyCopyable, Movable):
    """What one run of digits was and where it left the cursor.

    A named thing rather than a pair because every reader in this file needs
    three answers about a run of digits and only one of them is the number: how
    many digits there were decides the resolution a fraction asks for, and
    whether there were any at all is how a row is found not to match its
    format.
    """

    var value: Int64
    """The number the digits spell."""

    var at: Int
    """The position after the last digit taken."""

    var count: Int
    """How many digits were taken, which is zero when there were none."""

    def __init__(out self, value: Int64, at: Int, count: Int):
        """Holds the three answers.

        Args:
            value: The number.
            at: Where the cursor ended.
            count: How many digits there were.
        """
        self.value = value
        self.at = at
        self.count = count


struct Fields(Copyable, ImplicitlyCopyable, Movable):
    """What one row of text yielded, before it becomes a single number.

    Every field starts at the value it has when nothing said otherwise, which
    is midnight on the first of January 1900 for the date and zero for the
    clock. That is Python's own `strptime` default, so a caller whose format
    leaves the year out gets the answer the standard library would have given
    them rather than a different one.
    """

    var year: Int64
    """The calendar year."""

    var month: Int64
    """The month from 1 to 12."""

    var day: Int64
    """The day of the month from 1."""

    var hour: Int64
    """The hour from 0 to 23."""

    var minute: Int64
    """The minute from 0 to 59."""

    var second: Int64
    """The second from 0 to 59, with no leap second above it."""

    var nanosecond: Int64
    """Whatever followed the decimal point, scaled up to nanoseconds."""

    var fraction_digits: Int
    """How many digits followed the decimal point, which is what decides
    whether the column ends up counting microseconds or nanoseconds."""

    var offset: Int64
    """Seconds ahead of UTC, from a `%z` or a trailing Z."""

    var has_offset: Bool
    """Whether the row carried a zone at all, which is a different thing from
    carrying one that is zero: `+00:00` and a naive reading are the same
    instant and not the same column type."""

    def __init__(out self):
        """Starts every field at what it means when the format does not say."""
        self.year = 1900
        self.month = 1
        self.day = 1
        self.hour = 0
        self.minute = 0
        self.second = 0
        self.nanosecond = 0
        self.fraction_digits = 0
        self.offset = 0
        self.has_offset = False

    def seconds(self) raises -> Int64:
        """Folds the fields into one count of whole seconds since the epoch.

        Seconds rather than nanoseconds, which is the correction this method
        exists in this shape for. Nanoseconds is the finest unit there is and
        it is also the narrowest range: an Int64 of them reaches 1677 to 2262
        and nothing further, so a column holding the year 2300 overflowed on
        the way in and came back as 1715 with nothing reporting it. Seconds
        reach past the year 292 billion, the fraction is carried separately in
        `nanosecond`, and the two are put together at the column's own unit
        once every row has been read and that unit is known.

        Returns:
            Seconds since 1970-01-01T00:00:00, with any offset removed, not
            counting the fraction.

        Raises:
            Error: If the date the fields name does not exist, meaning a month
                outside 1 to 12, a day outside that month's own length, or a
                reading no clock has.
        """
        if self.month < 1 or self.month > 12:
            raise Error(
                "temporal: month must be in 1..12, not " + String(self.month)
            )
        var length = days_in_month(self.year, self.month)
        if self.day < 1 or self.day > length:
            raise Error(
                String(
                    "temporal: day ",
                    self.day,
                    " is outside 1..",
                    length,
                    " for month ",
                    self.month,
                    " of year ",
                    self.year,
                )
            )
        if self.hour > 23 or self.minute > 59 or self.second > 59:
            raise Error(
                String(
                    "temporal: ",
                    self.hour,
                    ":",
                    self.minute,
                    ":",
                    self.second,
                    " is not a reading a clock has",
                )
            )

        var days = days_from_civil(self.year, self.month, self.day)
        return (
            days * SECONDS_PER_DAY
            + self.hour * 3600
            + self.minute * 60
            + self.second
            - self.offset
        )

    def scaled(self, unit: TimeUnit) raises -> Int64:
        """Folds the fields into one count of the unit the column will hold.

        The seconds and the fraction are put together here rather than at some
        finer unit and divided down, so the only range that has to hold is the
        one the answer is actually stored at. That is what lets a column of
        microseconds hold the year 2300, which a column of nanoseconds cannot.

        Args:
            unit: What the column counts in.

        Returns:
            The count since the epoch, with any offset removed.

        Raises:
            Error: For a date that does not exist, and for an instant that does
                not fit an Int64 count of this unit, which is the year 2262 for
                nanoseconds and no year anybody will write for the rest.
        """
        var per = unit.per_second()
        var whole = self.seconds()
        if whole > MOST_SECONDS // per or whole < -(MOST_SECONDS // per):
            raise Error(
                String(
                    "temporal: ",
                    self.year,
                    "-",
                    self.month,
                    "-",
                    self.day,
                    " is outside the range a column of ",
                    unit,
                    " can hold",
                )
            )
        return whole * per + self.nanosecond // (1_000_000_000 // per)


def is_digit(byte: UInt8) -> Bool:
    """Says whether a byte is an ASCII digit.

    Args:
        byte: The byte.

    Returns:
        True for `0` through `9`.
    """
    return byte >= UInt8(ord("0")) and byte <= UInt8(ord("9"))


def read_digits(bytes: Span[UInt8, _], at: Int, most: Int) -> Digits:
    """Reads a run of digits, stopping at the first byte that is not one.

    Args:
        bytes: The row.
        at: Where to start.
        most: The largest number of digits to take, which is what stops a four
            digit year from eating the month when the two are not separated.

    Returns:
        The number, the position after it, and how many digits there were.
    """
    var cursor = at
    var value = Int64(0)
    var limit = at + most
    while cursor < len(bytes) and cursor < limit and is_digit(bytes[cursor]):
        value = value * 10 + Int64(Int(bytes[cursor]) - ord("0"))
        cursor += 1
    return Digits(value, cursor, cursor - at)


def _skip_blanks(bytes: Span[UInt8, _], at: Int) -> Int:
    """Steps over the spaces a space padded directive may have written.

    Args:
        bytes: The row.
        at: Where to start.

    Returns:
        The first position that is not a space.
    """
    var cursor = at
    while cursor < len(bytes) and bytes[cursor] == UInt8(ord(" ")):
        cursor += 1
    return cursor


def _readable(letter: UInt8) -> Bool:
    """Says whether a directive is one this can read as well as write.

    The renderer knows more directives than this does, and the difference is
    not an oversight. `%V`, `%U` and `%W` are week numbers, `%G` and `%g` are
    the year a week belongs to, `%j` is a day of the year, `%C` is a century
    and `%a`, `%A`, `%b` and `%B` are names. Every one of them is computed from
    a date and none of them names one on its own, so a format built out of them
    describes text that cannot be read back to the instant it came from.
    Refusing them by name is the honest answer.

    Args:
        letter: The directive letter.

    Returns:
        True for the directives carrying a field a date is made of.
    """
    return (
        letter == UInt8(ord("Y"))
        or letter == UInt8(ord("y"))
        or letter == UInt8(ord("m"))
        or letter == UInt8(ord("d"))
        or letter == UInt8(ord("e"))
        or letter == UInt8(ord("H"))
        or letter == UInt8(ord("k"))
        or letter == UInt8(ord("M"))
        or letter == UInt8(ord("S"))
        or letter == UInt8(ord("f"))
        or letter == UInt8(ord("z"))
        or letter == UInt8(ord("Z"))
    )


def _read_offset(
    bytes: Span[UInt8, _], at: Int, mut into: Fields
) raises -> Int:
    """Reads a zone, which is either the letter Z or a signed offset.

    Both spellings of an offset are taken, with the colon and without it,
    because both are ISO 8601 and pandas reads both. A `Z` and a `+00:00` are
    the same instant and both leave the column on UTC.

    Args:
        bytes: The row.
        at: Where the zone starts.
        into: Where to put the offset.

    Returns:
        The position after the zone.

    Raises:
        Error: If what is there is neither a Z nor a signed offset.
    """
    if at < len(bytes) and (bytes[at] | 32) == UInt8(ord("z")):
        into.offset = 0
        into.has_offset = True
        return at + 1

    if at >= len(bytes):
        raise Error("temporal: the row ends where a time zone should be")
    var sign = Int64(1)
    if bytes[at] == UInt8(ord("-")):
        sign = -1
    elif bytes[at] != UInt8(ord("+")):
        raise Error(
            "temporal: a time zone starts with Z, + or -, and this one starts"
            " with '"
            + chr(Int(bytes[at]))
            + "'"
        )

    var hours = read_digits(bytes, at + 1, 2)
    if hours.count != 2:
        raise Error("temporal: a time zone offset needs two digits of hours")
    var cursor = hours.at
    if cursor < len(bytes) and bytes[cursor] == UInt8(ord(":")):
        cursor += 1
    var minutes = read_digits(bytes, cursor, 2)
    if minutes.count != 2:
        raise Error("temporal: a time zone offset needs two digits of minutes")

    into.offset = sign * (hours.value * 3600 + minutes.value * 60)
    into.has_offset = True
    return minutes.at


def read_row(
    bytes: Span[UInt8, _], steps: List[Step], fmt: StringSlice
) raises -> Fields:
    """Reads one row of text through an already parsed format.

    Args:
        bytes: The row.
        steps: The parsed format, the same list the renderer walks.
        fmt: The format string those steps point into.

    Returns:
        The fields the row carried.

    Raises:
        Error: If the row does not match the format, or if the format holds a
            directive that cannot be read back.
    """
    var got = Fields()
    var raw = fmt.as_bytes()
    var at = 0

    for step in steps:
        if step.code == STEP_SLICE:
            for k in range(step.start, step.stop):
                if at >= len(bytes) or bytes[at] != raw[k]:
                    raise Error(
                        "temporal: the row does not match the format '"
                        + String(fmt)
                        + "'"
                    )
                at += 1
            continue
        if step.code == STEP_BYTE:
            if at >= len(bytes) or bytes[at] != step.pad:
                raise Error(
                    "temporal: the row does not match the format '"
                    + String(fmt)
                    + "'"
                )
            at += 1
            continue

        var letter = step.code
        if not _readable(letter):
            raise Error(
                "temporal: '%"
                + chr(Int(letter))
                + "' can be written and not read, because it is computed from"
                " a date rather than being a part of one, so a format holding"
                " it cannot name an instant to read back"
            )

        if letter == UInt8(ord("z")) or letter == UInt8(ord("Z")):
            at = _read_offset(bytes, at, got)
            continue

        if letter == UInt8(ord("f")):
            var fraction = read_digits(bytes, at, MAX_FRACTION_DIGITS)
            if fraction.count == 0:
                raise Error("temporal: '%f' needs at least one digit after it")
            got.fraction_digits = fraction.count
            # Whatever was written is scaled up to nanoseconds, so that three
            # digits and nine digits land in the same field and only the count
            # of them says what resolution the column wants.
            var scaled = fraction.value
            for _ in range(MAX_FRACTION_DIGITS - fraction.count):
                scaled *= 10
            got.nanosecond = scaled
            at = fraction.at
            continue

        # `%e` and `%k` are the space padded spellings, so a leading space
        # belongs to the number rather than to the format around it.
        if letter == UInt8(ord("e")) or letter == UInt8(ord("k")):
            at = _skip_blanks(bytes, at)

        var width = 4 if letter == UInt8(ord("Y")) else 2
        var read = read_digits(bytes, at, width)
        if read.count == 0:
            raise Error(
                "temporal: the row does not match the format '"
                + String(fmt)
                + "'"
            )
        at = read.at

        if letter == UInt8(ord("Y")):
            got.year = read.value
        elif letter == UInt8(ord("y")):
            # Two digits with no century in front of them. The window is the
            # POSIX one that Python uses, so 69 is 1969 and 68 is 2068.
            if read.value >= 69:
                got.year = read.value + 1900
            else:
                got.year = read.value + 2000
        elif letter == UInt8(ord("m")):
            got.month = read.value
        elif letter == UInt8(ord("d")) or letter == UInt8(ord("e")):
            got.day = read.value
        elif letter == UInt8(ord("H")) or letter == UInt8(ord("k")):
            got.hour = read.value
        elif letter == UInt8(ord("M")):
            got.minute = read.value
        else:
            got.second = read.value

    if at != len(bytes):
        raise Error(
            "temporal: the row has "
            + String(len(bytes) - at)
            + " bytes left over after the format '"
            + String(fmt)
            + "'"
        )
    return got^


def _zone_suffix(bytes: Span[UInt8, _], at: Int, text: String) raises -> String:
    """Returns the format for whatever is left, which can only be a zone.

    Args:
        bytes: The row.
        at: Where the rest starts.
        text: The whole row, for the message.

    Returns:
        `%z`, or an empty string when there is nothing left.

    Raises:
        Error: If what is left is not a zone.
    """
    if at >= len(bytes):
        return String("")
    var byte = bytes[at]
    if (
        (byte | 32) == UInt8(ord("z"))
        or byte == UInt8(ord("+"))
        or byte == UInt8(ord("-"))
    ):
        return String("%z")
    raise Error(
        "temporal: '"
        + text
        + "' has '"
        + chr(Int(byte))
        + "' where firepanda expected a time zone or the end of the value, and"
        " it guesses ISO 8601 and nothing else, so pass a format for this one"
    )


def guess_format(bytes: Span[UInt8, _]) raises -> String:
    """Works out which ISO 8601 shape one row is written in.

    This is deliberately narrow. It recognises a date, a date and a time, an
    optional fraction and an optional zone, in the two separators ISO 8601
    allows between them, and it recognises nothing else.

    Args:
        bytes: The first row of the column that is not missing.

    Returns:
        A format string made of directives `parse_format` knows.

    Raises:
        Error: If the row is not ISO 8601.
    """
    var text = String(StringSlice(unsafe_from_utf8=bytes))
    var n = len(bytes)

    var year = read_digits(bytes, 0, 4)
    if year.count != 4:
        raise Error(
            "temporal: '"
            + text
            + "' does not start with a four digit year, and firepanda guesses"
            " ISO 8601 and nothing else, so pass a format for this one"
        )
    var cursor = year.at

    # `20260101` is the basic form, which has no separators anywhere in it and
    # is the one case where the whole shape is decided by the length.
    if cursor < n and is_digit(bytes[cursor]):
        if n == 8:
            return String("%Y%m%d")
        raise Error(
            "temporal: '"
            + text
            + "' has digits where a separator should be and is not the eight"
            " digit basic form either, so pass a format for this one"
        )

    var out = String("%Y")
    if cursor == n:
        return out^
    if bytes[cursor] != UInt8(ord("-")):
        raise Error(
            "temporal: '"
            + text
            + "' separates the year with '"
            + chr(Int(bytes[cursor]))
            + "' rather than a hyphen, and firepanda guesses ISO 8601 and"
            " nothing else, so pass a format for this one"
        )

    var month = read_digits(bytes, cursor + 1, 2)
    if month.count != 2:
        raise Error(
            "temporal: '" + text + "' has no two digit month after the year"
        )
    out += "-%m"
    cursor = month.at
    if cursor == n:
        return out^

    if bytes[cursor] != UInt8(ord("-")):
        raise Error(
            "temporal: '"
            + text
            + "' has no hyphen between the month and the day"
        )
    var day = read_digits(bytes, cursor + 1, 2)
    if day.count != 2:
        raise Error(
            "temporal: '" + text + "' has no two digit day after the month"
        )
    out += "-%d"
    cursor = day.at
    if cursor == n:
        return out^

    # ISO 8601 says the letter T between the date and the time and everyday
    # writing says a space, and pandas reads both. Which one it was belongs in
    # the format rather than being normalised away, because the format then
    # holds every other row to the same separator, which is what pandas does
    # with them too.
    if bytes[cursor] == UInt8(ord("T")):
        out += "T"
    elif bytes[cursor] == UInt8(ord(" ")):
        out += " "
    else:
        out += _zone_suffix(bytes, cursor, text)
        return out^
    cursor += 1

    var hour = read_digits(bytes, cursor, 2)
    if hour.count != 2:
        raise Error(
            "temporal: '" + text + "' has no two digit hour after the date"
        )
    out += "%H"
    cursor = hour.at
    if cursor == n:
        return out^

    if bytes[cursor] == UInt8(ord(":")):
        var minute = read_digits(bytes, cursor + 1, 2)
        if minute.count != 2:
            raise Error(
                "temporal: '"
                + text
                + "' has no two digit minute after the hour"
            )
        out += ":%M"
        cursor = minute.at
        if cursor == n:
            return out^

        if bytes[cursor] == UInt8(ord(":")):
            var second = read_digits(bytes, cursor + 1, 2)
            if second.count != 2:
                raise Error(
                    "temporal: '"
                    + text
                    + "' has no two digit second after the minute"
                )
            out += ":%S"
            cursor = second.at
            if cursor == n:
                return out^

            if bytes[cursor] == UInt8(ord(".")):
                var fraction = read_digits(
                    bytes, cursor + 1, MAX_FRACTION_DIGITS
                )
                if fraction.count == 0:
                    raise Error(
                        "temporal: '"
                        + text
                        + "' has a decimal point with no digits after it"
                    )
                out += ".%f"
                cursor = fraction.at
                if cursor == n:
                    return out^

    out += _zone_suffix(bytes, cursor, text)
    return out^


def is_missing_word(bytes: Span[UInt8, _]) -> Bool:
    """Says whether a value is one of the spellings that mean missing.

    Measured against pandas rather than guessed at, and it is a short list. The
    empty string, `NaT` and `nan` are missing in either case of either letter,
    and `None`, `null`, `NA`, `N/A` and a lone hyphen are all values pandas
    refuses outright. So a reader treating `null` as missing would be quietly
    accepting a file pandas rejects, which is a difference in the wrong
    direction for a library whose claim is that pandas code runs on it
    unchanged.

    Args:
        bytes: The row.

    Returns:
        True when the row means missing.
    """
    var n = len(bytes)
    if n == 0:
        return True
    if n != 3:
        return False
    var a = bytes[0] | 32
    var b = bytes[1] | 32
    var c = bytes[2] | 32
    if a != UInt8(ord("n")) or b != UInt8(ord("a")):
        return False
    return c == UInt8(ord("t")) or c == UInt8(ord("n"))


def _offset_name(offset: Int64) -> String:
    """Spells a fixed offset the way pandas prints one.

    pandas calls a column read against `+02:00` a `UTC+02:00`, and that is the
    spelling `TimeZone` reads back, so writing it here is what makes the
    column's own type say what its numbers mean.

    Args:
        offset: Seconds ahead of UTC.

    Returns:
        A name like `UTC+02:00`.
    """
    var sign = "+" if offset >= 0 else "-"
    var total = offset if offset >= 0 else -offset
    var hours = total // 3600
    var minutes = (total % 3600) // 60
    var out = String("UTC") + sign
    if hours < 10:
        out += "0"
    out += String(hours) + ":"
    if minutes < 10:
        out += "0"
    out += String(minutes)
    return out^


def parse_timestamps(
    text: StringArray,
    fmt: StringSlice,
    guess: Bool,
    coerce: Bool,
    utc: Bool,
) raises -> AnyArray:
    """Reads a column of text as a column of instants.

    The format is worked out once, from the first row that is not missing, and
    every other row is then read through it. That is what pandas does and it is
    why a column holding both `2026-01-02` and `2026-01-01 12:34:56` is an
    error in both libraries rather than a column with two shapes in it.

    Args:
        text: The column.
        fmt: The format, when the caller gave one.
        guess: Whether to work the format out from the first row instead.
        coerce: Whether a row that will not read becomes missing rather than an
            error.
        utc: Whether to put the answer on UTC, which is the only way a column
            carrying more than one offset can be read at all.

    Returns:
        A timestamp column, null wherever the input was null or unreadable.

    Raises:
        Error: If a row does not match the format, if the rows carry different
            offsets and `utc` was not asked for, or if the format holds a
            directive that cannot be read.
    """
    var n = len(text)
    var rows = List[Fields](capacity=n)
    var present = List[Bool](capacity=n)

    var first = -1
    for i in range(n):
        if text.is_valid(i) and not is_missing_word(text.unsafe_bytes(i)):
            first = i
            break

    # An empty column and a column of nothing but missing rows have no row to
    # read a format out of and nothing to apply one to. pandas answers seconds
    # for both, which is the coarsest unit there is, and so does this.
    if first < 0:
        var empty = Array[DType.int64](n)
        for i in range(n):
            empty.set_null(i)
        return AnyArray(
            empty^.into_data(), LogicalType.timestamp(TimeUnit.SECOND)
        )

    var chosen = String(fmt)
    if guess:
        chosen = guess_format(text.unsafe_bytes(first))
    var steps = parse_format(chosen)

    var finest = 0
    var offset = Int64(0)
    var zoned = False
    var seen_zone = False

    for i in range(n):
        if not text.is_valid(i) or is_missing_word(text.unsafe_bytes(i)):
            rows.append(Fields())
            present.append(False)
            continue

        # The fields are kept rather than folded here, because folding them
        # needs the column's unit and the column's unit is not known until every
        # row has been looked at. A row seven digits into the fraction takes the
        # whole column to nanoseconds, and nanoseconds is the one unit a date
        # can fall outside of, so folding early is what would decide a range
        # question before the range is known.
        var fields: Fields
        try:
            fields = read_row(text.unsafe_bytes(i), steps, chosen)
            # Folded once here and thrown away, so that a date nothing can hold
            # is refused on the row it is on rather than after the loop, where
            # the message could not say which row.
            _ = fields.seconds()
            rows.append(fields)
        except error:
            if not coerce:
                raise error
            rows.append(Fields())
            present.append(False)
            continue

        present.append(True)
        if fields.fraction_digits > finest:
            finest = fields.fraction_digits
        if fields.has_offset:
            if seen_zone and fields.offset != offset and not utc:
                raise Error(
                    "temporal: the rows carry different offsets from UTC, so"
                    " there is no one clock to read them all against; pass"
                    " utc=True to read them against UTC"
                )
            zoned = True
            seen_zone = True
            offset = fields.offset

    # pandas answers microseconds unless something in the column needed more,
    # measured rather than assumed, and one row carrying seven digits takes the
    # whole column to nanoseconds.
    var unit = TimeUnit.NANO if finest > 6 else TimeUnit.MICRO

    var out = Array[DType.int64](overwritten=n)
    for i in range(n):
        if present[i]:
            out.set_valid(i, rows[i].scaled(unit))
        else:
            out.set_null(i)

    var zone = TimeZone()
    if utc or (zoned and offset == 0):
        zone = TimeZone("UTC")
    elif zoned:
        zone = TimeZone(_offset_name(offset))
    return AnyArray(out^.into_data(), LogicalType.timestamp(unit, zone))


def numbers_to_timestamps(a: AnyArray, unit: TimeUnit) raises -> AnyArray:
    """Reads a column of whole numbers as a column of instants.

    This is `pandas.to_datetime` with a unit given, and like `to_timedelta` on
    an integer column it is a relabelling rather than a conversion: the
    integers are already the counts and the unit says what they are counts of.
    So the buffer is copied once and a null stays a null.

    A column with no rows in it is read as an empty column of instants whatever
    its type says, since there is nothing there to relabel and an empty column
    carries no evidence of what it was going to hold. Without that a caller
    handing over an empty list would be refused, because an empty list with
    nothing in it to look at infers as a column of floats, and pandas builds an
    empty `DatetimeIndex` out of the same empty list.

    Args:
        a: An integer column.
        unit: The resolution the integers are counts of.

    Returns:
        A timestamp column at that unit.

    Raises:
        Error: If the column is not one of whole numbers.
    """
    if len(a) == 0:
        return AnyArray(
            Array[DType.int64](0).into_data(), LogicalType.timestamp(unit)
        )
    if a.type.physical != DType.int64:
        raise Error(
            "temporal: to_datetime with a unit needs a column of whole numbers"
            " and this one is "
            + String(a.type)
        )
    return AnyArray(
        Array[DType.int64](copy=a.as_typed_view[DType.int64]()).into_data(),
        LogicalType.timestamp(unit),
    )
