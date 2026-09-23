"""The IANA time zone database, read from the files the system already has.

A zone like `America/New_York` is a rule rather than a number, and the rule is
a table of the instants at which the zone's offset changed, followed by a line
saying how it keeps changing after the table runs out. Every platform this
library targets ships that table as a directory of TZif files, the format RFC
8536 specifies, under `/usr/share/zoneinfo`. Reading those files is a few
hundred lines and no dependency, and it is the same data Python's `zoneinfo`
reads, which is what pandas answers from.

The table here is the file's transitions followed by the footer rule written
out year by year for four hundred years. Four hundred is not a guess. The
Gregorian calendar repeats exactly every four hundred years, weekdays included,
because 146097 days is a whole number of weeks, so a rule like "the second
Sunday in March" lands on the same day of the cycle every time round. An
instant past the written out part is folded back into it by whole cycles, which
keeps the table a few kilobytes and every lookup exact however far out a
timestamp in seconds reaches.

See docs/specs/03-dtype-dispatch.md.
"""

comptime ZONEINFO_ROOT = "/usr/share/zoneinfo/"
"""Where every platform this library targets keeps the database."""

comptime SECONDS_PER_DAY: Int64 = 86400
"""Seconds in a civil day, which the database never adds a leap second to."""

comptime CYCLE_YEARS = 400
"""Years after which the Gregorian calendar repeats itself exactly."""

comptime CYCLE_SECONDS: Int64 = 146097 * SECONDS_PER_DAY
"""The same four hundred years in seconds."""


def _days_from_civil(year: Int64, month: Int64, day: Int64) -> Int64:
    """Returns the day number of a date, counting from 1970-01-01.

    The kernel module has a vectorised version of this and cannot be imported
    from here, since it imports this module. This is the scalar form of the
    same algorithm, which is Howard Hinnant's.

    Args:
        year: The proleptic Gregorian year.
        month: The month, from 1.
        day: The day of the month, from 1.

    Returns:
        Days since the epoch, negative before it.
    """
    var y = year - 1 if month <= 2 else year
    var era = (y if y >= 0 else y - 399) // 400
    var yoe = y - era * 400
    var mp = month + 9 if month <= 2 else month - 3
    var doy = (153 * mp + 2) // 5 + day - 1
    var doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _year_of(seconds: Int64) -> Int64:
    """Returns the civil year a UTC instant falls in.

    Args:
        seconds: Seconds since the epoch.

    Returns:
        The proleptic Gregorian year.
    """
    var z = seconds // SECONDS_PER_DAY + 719468
    var era = (z if z >= 0 else z - 146096) // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var year = yoe + era * 400
    return year + 1 if mp >= 10 else year


def _is_leap(year: Int64) -> Bool:
    """Returns whether a year has a 29 February.

    Args:
        year: The proleptic Gregorian year.

    Returns:
        True for a leap year.
    """
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


@fieldwise_init
struct _Date(Copyable, ImplicitlyCopyable, Movable):
    """One end of a daylight saving period, as a POSIX TZ string writes it."""

    var form: Int
    """0 for `Jn`, 1 for a bare `n`, 2 for `Mm.w.d`."""

    var month: Int64
    """The month for the third form."""

    var week: Int64
    """The week of the month for the third form, where 5 means the last."""

    var day: Int64
    """The day of the year for the first two forms and of the week for the third."""

    var time: Int64
    """Seconds after local midnight at which the change happens."""

    def local_seconds(self, year: Int64) -> Int64:
        """Returns the local reading at which this change happens in a year.

        Args:
            year: The year.

        Returns:
            Seconds since the epoch on the local clock.
        """
        var first = _days_from_civil(year, 1, 1)
        var days: Int64
        if self.form == 0:
            days = first + self.day - 1
            if _is_leap(year) and self.day >= 60:
                days += 1
        elif self.form == 1:
            days = first + self.day
        else:
            var start = _days_from_civil(year, self.month, 1)
            # 1970-01-01 was a Thursday, which is day 4 counting from Sunday.
            var weekday = (start + 4) % 7
            days = start + (self.day - weekday + 7) % 7 + (self.week - 1) * 7
            var next_month = _days_from_civil(
                year + 1 if self.month == 12 else year,
                1 if self.month == 12 else self.month + 1,
                1,
            )
            while days >= next_month:
                days -= 7
        return days * SECONDS_PER_DAY + self.time


struct _Footer(Copyable, Movable):
    """What a POSIX TZ string says about every instant past the table."""

    var standard: Int64
    """Seconds the zone is ahead of UTC outside daylight saving time."""

    var daylight: Int64
    """Seconds it is ahead during it, meaningless when `has_rule` is false."""

    var has_rule: Bool
    """Whether the zone has daylight saving time at all."""

    var start: _Date
    """When daylight saving time begins, read on standard time."""

    var end: _Date
    """When it ends, read on daylight saving time."""

    def __init__(out self, standard: Int64):
        """Constructs a footer for a zone that never changes its clock.

        Args:
            standard: Its one offset.
        """
        self.standard = standard
        self.daylight = standard
        self.has_rule = False
        self.start = _Date(0, 0, 0, 0, 0)
        self.end = _Date(0, 0, 0, 0, 0)


struct _Cursor:
    """A position in a POSIX TZ string, and the parsing that moves it."""

    var text: List[UInt8]
    """The string."""

    var at: Int
    """The next unread byte."""

    def __init__(out self, var text: List[UInt8]):
        """Starts at the beginning of a string.

        Args:
            text: The string.
        """
        self.text = text^
        self.at = 0

    def done(self) -> Bool:
        """Returns whether every byte has been read."""
        return self.at >= len(self.text)

    def peek(self) -> UInt8:
        """Returns the next byte, or zero at the end."""
        return self.text[self.at] if self.at < len(self.text) else 0

    def name(mut self) raises:
        """Skips a zone abbreviation, quoted in angle brackets or not.

        Raises:
            Error: If there is none.
        """
        var begin = self.at
        if self.peek() == UInt8(ord("<")):
            while not self.done() and self.peek() != UInt8(ord(">")):
                self.at += 1
            if self.done():
                raise Error("an abbreviation opened with < is never closed")
            self.at += 1
            return
        while not self.done():
            var byte = self.peek()
            var upper = byte & 0xDF
            if upper < UInt8(ord("A")) or upper > UInt8(ord("Z")):
                break
            self.at += 1
        if self.at - begin < 3:
            raise Error("an abbreviation is shorter than three letters")

    def number(mut self) raises -> Int64:
        """Reads a run of digits.

        Returns:
            Their value.

        Raises:
            Error: If there are none.
        """
        var begin = self.at
        var value = Int64(0)
        while not self.done():
            var byte = self.peek()
            if byte < UInt8(ord("0")) or byte > UInt8(ord("9")):
                break
            value = value * 10 + Int64(byte - UInt8(ord("0")))
            self.at += 1
        if self.at == begin:
            raise Error("a number was expected")
        return value

    def clock(mut self) raises -> Int64:
        """Reads `[+-]hh[:mm[:ss]]`, which is both an offset and a time of day.

        Returns:
            Seconds, with the sign as written.

        Raises:
            Error: If there is no hour.
        """
        var sign = Int64(1)
        if self.peek() == UInt8(ord("+")):
            self.at += 1
        elif self.peek() == UInt8(ord("-")):
            sign = -1
            self.at += 1
        var seconds = self.number() * 3600
        if self.peek() == UInt8(ord(":")):
            self.at += 1
            seconds += self.number() * 60
            if self.peek() == UInt8(ord(":")):
                self.at += 1
                seconds += self.number()
        return sign * seconds

    def date(mut self) raises -> _Date:
        """Reads one end of a daylight saving period and its optional time.

        Returns:
            The date, at two in the morning unless the string says otherwise.

        Raises:
            Error: If it is none of the three forms.
        """
        var out = _Date(1, 0, 0, 0, 7200)
        if self.peek() == UInt8(ord("J")):
            self.at += 1
            out.form = 0
            out.day = self.number()
        elif self.peek() == UInt8(ord("M")):
            self.at += 1
            out.form = 2
            out.month = self.number()
            self.expect(".")
            out.week = self.number()
            self.expect(".")
            out.day = self.number()
            if out.month < 1 or out.month > 12 or out.week < 1 or out.week > 5:
                raise Error("a month rule is out of range")
        else:
            out.day = self.number()
        if self.peek() == UInt8(ord("/")):
            self.at += 1
            out.time = self.clock()
        return out

    def expect(mut self, byte: StringSlice) raises:
        """Reads one particular byte.

        Args:
            byte: The byte, as a one character string.

        Raises:
            Error: If the next byte is anything else.
        """
        if self.peek() != byte.as_bytes()[0]:
            raise Error(String("expected '", byte, "'"))
        self.at += 1


def _parse_footer(var text: List[UInt8]) raises -> _Footer:
    """Reads the POSIX TZ string at the end of a version 2 file.

    Offsets in that string are how far behind UTC the zone is, which is the
    opposite sign to every other offset in this library, and they are turned
    round here so that nothing past this function has to remember that.

    Args:
        text: The string, without the newlines around it.

    Returns:
        What it says.

    Raises:
        Error: If it is not a TZ string.
    """
    var cursor = _Cursor(text^)
    cursor.name()
    var out = _Footer(-cursor.clock())
    if cursor.done():
        return out^
    cursor.name()
    out.daylight = out.standard + 3600
    if not cursor.done() and cursor.peek() != UInt8(ord(",")):
        out.daylight = -cursor.clock()
    out.has_rule = True
    if cursor.done():
        # The rule is optional in the grammar and every zone that leaves it
        # out means the one the United States has used since 2007.
        out.start = _Date(2, 3, 2, 0, 7200)
        out.end = _Date(2, 11, 1, 0, 7200)
        return out^
    cursor.expect(",")
    out.start = cursor.date()
    cursor.expect(",")
    out.end = cursor.date()
    if not cursor.done():
        raise Error("a TZ string has something after its rule")
    return out^


struct _Reader:
    """Big endian integers out of the bytes of a file."""

    var data: List[UInt8]
    """The file."""

    var at: Int
    """The next unread byte."""

    def __init__(out self, var data: List[UInt8]):
        """Starts at the beginning of a file.

        Args:
            data: The file.
        """
        self.data = data^
        self.at = 0

    def need(self, count: Int) raises:
        """Checks that a number of bytes remain.

        Args:
            count: How many.

        Raises:
            Error: If the file ends first.
        """
        if count < 0 or self.at + count > len(self.data):
            raise Error("the file ends in the middle of its table")

    def int(mut self, width: Int) raises -> Int64:
        """Reads one signed big endian integer.

        Args:
            width: Its size in bytes, which is 4 or 8.

        Returns:
            The integer.

        Raises:
            Error: If the file ends first.
        """
        self.need(width)
        var value = UInt64(0)
        for i in range(width):
            value = (value << 8) | UInt64(self.data[self.at + i])
        self.at += width
        var signed = value.cast[DType.int64]()
        if width == 4 and signed >= 0x80000000:
            signed -= 0x100000000
        return signed

    def byte(mut self) raises -> UInt8:
        """Reads one byte.

        Returns:
            The byte.

        Raises:
            Error: If the file ends first.
        """
        self.need(1)
        var value = self.data[self.at]
        self.at += 1
        return value


struct ZoneRules(Copyable, Movable):
    """Every offset a zone has had and will have, and when each one starts."""

    var times: List[Int64]
    """UTC seconds at which each offset takes over, in increasing order."""

    var offsets: List[Int64]
    """The seconds the zone is ahead of UTC from the matching time onward."""

    var before: Int64
    """What it is ahead of UTC before the first time, or always if none."""

    var cycle_start: Int64
    """The start of the four hundred written out years, or the largest Int64
    when the zone has no rule to write out and needs no folding."""

    def __init__(out self, before: Int64):
        """Constructs a zone whose offset has never changed.

        Args:
            before: The offset.
        """
        self.times = List[Int64]()
        self.offsets = List[Int64]()
        self.before = before
        self.cycle_start = Int64.MAX

    def offset_at(self, seconds: Int64) -> Int64:
        """Returns how far ahead of UTC the zone is at an instant.

        Args:
            seconds: The instant, in UTC seconds since the epoch.

        Returns:
            The offset in seconds.
        """
        var t = seconds
        if self.cycle_start != Int64.MAX:
            var past = t - self.cycle_start
            if past >= CYCLE_SECONDS:
                t = self.cycle_start + past % CYCLE_SECONDS
        # The number of transitions at or before t, by bisection.
        var low = 0
        var high = len(self.times)
        while low < high:
            var middle = (low + high) // 2
            if self.times[middle] <= t:
                low = middle + 1
            else:
                high = middle
        if low == 0:
            return self.before
        return self.offsets[low - 1]

    def _push(mut self, time: Int64, offset: Int64):
        """Appends a transition, skipping one that would not move forward.

        Args:
            time: When it happens, in UTC seconds.
            offset: The offset from then on.
        """
        var count = len(self.times)
        if count > 0 and time <= self.times[count - 1]:
            return
        self.times.append(time)
        self.offsets.append(offset)

    def _extend(mut self, footer: _Footer):
        """Writes the footer rule out for one four hundred year cycle.

        Args:
            footer: The rule.
        """
        var count = len(self.times)
        if not footer.has_rule:
            if count == 0:
                self.before = footer.standard
            elif self.offsets[count - 1] != footer.standard:
                self._push(self.times[count - 1] + 1, footer.standard)
            return
        var first_year = Int64(1970)
        if count > 0:
            first_year = _year_of(self.times[count - 1]) + 1
        self.cycle_start = _days_from_civil(first_year, 1, 1) * SECONDS_PER_DAY
        # One year before the cycle as well, so that an instant early in its
        # first January finds the change that came before it.
        for y in range(Int(first_year) - 1, Int(first_year) + CYCLE_YEARS):
            var year = Int64(y)
            var start = footer.start.local_seconds(year) - footer.standard
            var end = footer.end.local_seconds(year) - footer.daylight
            if start < end:
                self._push(start, footer.daylight)
                self._push(end, footer.standard)
            else:
                self._push(end, footer.standard)
                self._push(start, footer.daylight)


def _valid_name(name: StringSlice) -> Bool:
    """Returns whether a zone name can be looked up without leaving the tree.

    Args:
        name: The name.

    Returns:
        False for anything that is not a path of letters, digits and the
        punctuation real zone names use, or that climbs out with `..`.
    """
    var raw = name.as_bytes()
    if len(raw) == 0 or raw[0] == UInt8(ord("/")):
        return False
    for i in range(len(raw)):
        var byte = raw[i]
        var upper = byte & 0xDF
        var letter = upper >= UInt8(ord("A")) and upper <= UInt8(ord("Z"))
        var digit = byte >= UInt8(ord("0")) and byte <= UInt8(ord("9"))
        var mark = (
            byte == UInt8(ord("/"))
            or byte == UInt8(ord("_"))
            or byte == UInt8(ord("-"))
            or byte == UInt8(ord("+"))
            or byte == UInt8(ord("."))
        )
        if not (letter or digit or mark):
            return False
        if (
            byte == UInt8(ord("."))
            and i + 1 < len(raw)
            and raw[i + 1] == UInt8(ord("."))
        ):
            return False
    return True


def parse_zone(var data: List[UInt8]) raises -> ZoneRules:
    """Reads the bytes of a TZif file.

    A version 1 file has 32 bit transition times, which stop in 2038, and a
    version 2 or later file repeats the table with 64 bit times and then gives
    the rule in force after it. The second table is the one read when there is
    one. Leap second records are skipped, since pandas counts no leap seconds
    and neither does this library.

    The offset before the first transition is the first standard time type,
    or the first type of any kind when a zone has no standard time, which is
    what Python's `zoneinfo` chooses and so what pandas answers.

    Args:
        data: The file.

    Returns:
        The zone.

    Raises:
        Error: If it is not a TZif file.
    """
    var reader = _Reader(data^)
    reader.need(44)
    if (
        reader.data[0] != UInt8(ord("T"))
        or reader.data[1] != UInt8(ord("Z"))
        or reader.data[2] != UInt8(ord("i"))
        or reader.data[3] != UInt8(ord("f"))
    ):
        raise Error("it does not start with TZif")
    var version = reader.data[4]
    reader.at = 20
    var isutcnt = Int(reader.int(4))
    var isstdcnt = Int(reader.int(4))
    var leapcnt = Int(reader.int(4))
    var timecnt = Int(reader.int(4))
    var typecnt = Int(reader.int(4))
    var charcnt = Int(reader.int(4))
    var width = 4
    if version != 0:
        var skip = (
            timecnt * 5
            + typecnt * 6
            + charcnt
            + leapcnt * 8
            + isstdcnt
            + isutcnt
        )
        reader.need(skip + 44)
        reader.at += skip + 20
        isutcnt = Int(reader.int(4))
        isstdcnt = Int(reader.int(4))
        leapcnt = Int(reader.int(4))
        timecnt = Int(reader.int(4))
        typecnt = Int(reader.int(4))
        charcnt = Int(reader.int(4))
        width = 8
    if typecnt < 1:
        raise Error("it has no time types")

    var times = List[Int64](capacity=timecnt)
    for _ in range(timecnt):
        times.append(reader.int(width))
    var kinds = List[Int](capacity=timecnt)
    for _ in range(timecnt):
        var kind = Int(reader.byte())
        if kind >= typecnt:
            raise Error("a transition names a time type that is not there")
        kinds.append(kind)
    var utoffs = List[Int64](capacity=typecnt)
    var dsts = List[Bool](capacity=typecnt)
    for _ in range(typecnt):
        utoffs.append(reader.int(4))
        dsts.append(reader.byte() != 0)
        _ = reader.byte()
    reader.need(charcnt + leapcnt * (width + 4) + isstdcnt + isutcnt)
    reader.at += charcnt + leapcnt * (width + 4) + isstdcnt + isutcnt

    var before = utoffs[0]
    for i in range(typecnt):
        if not dsts[i]:
            before = utoffs[i]
            break
    var out = ZoneRules(before)
    for i in range(timecnt):
        out._push(times[i], utoffs[kinds[i]])

    if version != 0 and reader.at < len(reader.data):
        if reader.data[reader.at] != UInt8(ord("\n")):
            raise Error("its footer does not start on a new line")
        var text = List[UInt8]()
        var at = reader.at + 1
        while at < len(reader.data) and reader.data[at] != UInt8(ord("\n")):
            text.append(reader.data[at])
            at += 1
        if len(text) > 0:
            out._extend(_parse_footer(text^))
    return out^


def load_zone(name: StringSlice) raises -> ZoneRules:
    """Reads a zone out of the system's database by its IANA name.

    Args:
        name: The name, such as `America/New_York`.

    Returns:
        The zone.

    Raises:
        Error: If there is no such zone, saying so in the words Python's
            `zoneinfo` uses, or if the file is not one this can read.
    """
    if not _valid_name(name):
        raise Error(String("No time zone found with key ", name))
    var data: List[UInt8]
    try:
        var handle = open(String(ZONEINFO_ROOT, name), "r")
        data = handle.read_bytes()
        handle.close()
    except:
        raise Error(String("No time zone found with key ", name))
    try:
        return parse_zone(data^)
    except e:
        raise Error(
            String("No time zone found with key ", name, ", because ", e)
        )
