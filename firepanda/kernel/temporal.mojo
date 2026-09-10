"""The calendar and clock fields of a temporal column.

A timestamp column is a column of integers and a count of them per second. Every
name on the pandas `dt` accessor that answers a number is a function of those two
things and of nothing else, so the whole of `dt.year`, `dt.month`, `dt.day`,
`dt.dayofweek`, `dt.quarter` and the six `is_` predicates is one calendar
conversion written once and read twenty two ways, the last three of those being
the ISO calendar, which is the same conversion applied to a different day of the
same week.

The conversion is Howard Hinnant's `civil_from_days`, which turns a count of days
since 1970-01-01 into a year, a month and a day with no table, no loop and no
branch. It works by moving the start of the year to March, which puts the leap
day at the end where it stops disturbing the month lengths, and then by dividing
into the four hundred year cycle over which the Gregorian calendar repeats
exactly. Everything after that is integer arithmetic on non negative numbers.

Three things about it are worth knowing before reading the code.

The first is that the whole of it depends on division rounding down rather than
towards zero, and that this is a real hazard rather than a pedantic one. The last
second of 1969 is timestamp -1. Rounded down, -1 divided by 86400 is -1, which is
the day before the epoch, which is right. Rounded towards zero it is 0, which is
the epoch itself, and the answer comes back as 1 January 1970. Mojo's `//` rounds
down on a scalar and on a register, which was measured rather than assumed, and
the hardware instruction underneath it does not, so this is correct because the
compiler inserts a correction and would stop being correct if it stopped. There
is a test on the 1969 row for exactly that reason and it is the first test in the
file.

The second is that after the first two lines nothing here divides a negative
number at all. `shifted // DAYS_PER_ERA` rounds down, so the remainder after it
is in `[0, 146096]` whatever the sign of the input was, and every division below
that point is on a value already known to be non negative. That is not an
accident of the algorithm, it is the reason the algorithm is shaped this way.

The third is that pandas answers these with int32 and not int64. `dt.year` on a
column of a million rows is an int32 column in pandas 3.0, measured, and the
conformance suite compares integer widths exactly, so a correct year in the wrong
width is a wrong answer. The predicates answer bool. There is no width here that
is a matter of taste.

The time zone comes in at one point and only one. Every field below reads the
integers as they are stored, which is the wall clock reading for a naive column
and UTC for a zoned one, so a zoned column is turned into the readings it stands
for before any of them sees it. That turn is possible when the zone names its own
offset and impossible when it names a rule, so `UTC` and `+05:30` are answered
and `America/New_York` is refused with a sentence saying it needs a database this
library does not have yet. Refusing beats answering in UTC, because an hour that
is silently seven off is worse than an hour that is missing.
"""

from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.logical import LogicalType, TypeKind
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.exec import parallel_morsels

from .mask import repair_range

comptime NANOS_PER_SECOND = Int64(1_000_000_000)
"""How many nanoseconds are in a second. The frequency parser works in these
because a nanosecond is the finest thing Arrow counts, so every frequency any of
the four resolutions can name is a whole number of them."""

comptime ROUND_DOWN = 0
"""Rounding mode for `dt.floor`, which moves an instant to the multiple at or
below it."""

comptime ROUND_UP = 1
"""Rounding mode for `dt.ceil`, which moves an instant to the multiple at or
above it."""

comptime ROUND_HALF_EVEN = 2
"""Rounding mode for `dt.round`, which moves an instant to whichever multiple is
nearer and settles a tie on the even one."""

comptime SECONDS_PER_DAY = 86400
"""How many seconds are in a day, with no leap seconds, which is what Arrow and
pandas both mean by a day and what the civil calendar is defined against."""

comptime DAYS_PER_ERA = 146097
"""How many days are in four hundred years of the Gregorian calendar, which is
the cycle over which the whole calendar repeats exactly."""

comptime DAYS_TO_MARCH = 719468
"""How many days are between 0000-03-01 and 1970-01-01. Adding this moves the
epoch to the start of a four hundred year cycle whose year begins in March."""

comptime FIELD_YEAR = 0
"""Field code for `dt.year`."""

comptime FIELD_MONTH = 1
"""Field code for `dt.month`."""

comptime FIELD_DAY = 2
"""Field code for `dt.day`."""

comptime FIELD_HOUR = 3
"""Field code for `dt.hour`."""

comptime FIELD_MINUTE = 4
"""Field code for `dt.minute`."""

comptime FIELD_SECOND = 5
"""Field code for `dt.second`."""

comptime FIELD_MICROSECOND = 6
"""Field code for `dt.microsecond`."""

comptime FIELD_NANOSECOND = 7
"""Field code for `dt.nanosecond`."""

comptime FIELD_DAY_OF_WEEK = 8
"""Field code for `dt.dayofweek`."""

comptime FIELD_DAY_OF_YEAR = 9
"""Field code for `dt.dayofyear`."""

comptime FIELD_QUARTER = 10
"""Field code for `dt.quarter`."""

comptime FIELD_DAYS_IN_MONTH = 11
"""Field code for `dt.days_in_month`."""

comptime FIELD_IS_LEAP_YEAR = 12
"""Field code for `dt.is_leap_year`, and the first of the predicates. Every code
from here up to `FIELD_IS_YEAR_END` answers bool and every code below it answers
int32."""

comptime FIELD_IS_MONTH_START = 13
"""Field code for `dt.is_month_start`."""

comptime FIELD_IS_MONTH_END = 14
"""Field code for `dt.is_month_end`."""

comptime FIELD_IS_QUARTER_START = 15
"""Field code for `dt.is_quarter_start`."""

comptime FIELD_IS_QUARTER_END = 16
"""Field code for `dt.is_quarter_end`."""

comptime FIELD_IS_YEAR_START = 17
"""Field code for `dt.is_year_start`."""

comptime FIELD_IS_YEAR_END = 18
"""Field code for `dt.is_year_end`, and the last of the predicates."""

comptime FIELD_ISO_YEAR = 19
"""Field code for the year column of `dt.isocalendar`, and the first of the
three. Every code at or above this one answers uint32, which is what pandas
gives that frame, and none of the three is a name on `dt`."""

comptime FIELD_ISO_WEEK = 20
"""Field code for the week column of `dt.isocalendar`, from 1 to 53."""

comptime FIELD_ISO_DAY = 21
"""Field code for the day column of `dt.isocalendar`, from 1 for Monday, which
is the ISO numbering and not the one `dt.dayofweek` uses."""

comptime FIELD_CODES = [
    FIELD_YEAR,
    FIELD_MONTH,
    FIELD_DAY,
    FIELD_HOUR,
    FIELD_MINUTE,
    FIELD_SECOND,
    FIELD_MICROSECOND,
    FIELD_NANOSECOND,
    FIELD_DAY_OF_WEEK,
    FIELD_DAY_OF_YEAR,
    FIELD_QUARTER,
    FIELD_DAYS_IN_MONTH,
    FIELD_IS_LEAP_YEAR,
    FIELD_IS_MONTH_START,
    FIELD_IS_MONTH_END,
    FIELD_IS_QUARTER_START,
    FIELD_IS_QUARTER_END,
    FIELD_IS_YEAR_START,
    FIELD_IS_YEAR_END,
    FIELD_ISO_YEAR,
    FIELD_ISO_WEEK,
    FIELD_ISO_DAY,
]
"""Every field code, in order, for the dispatch to walk at compile time."""


def field_dtype(field: Int) -> DType:
    """Returns the dtype a field answers with.

    Args:
        field: The field code.

    Returns:
        `DType.bool` for the seven predicates, `DType.uint32` for the three
        columns of `isocalendar` and `DType.int32` for the rest, which is what
        pandas answers and is compared exactly.
    """
    if field >= FIELD_ISO_YEAR:
        return DType.uint32
    return DType.bool if field >= FIELD_IS_LEAP_YEAR else DType.int32


@fieldwise_init
struct TemporalField(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which part of a timestamp is being asked for.

    Held as a code rather than as a function for the same reason `UnaryOp` is:
    the erased entry point takes it as an ordinary argument and the typed loop
    takes it as a parameter the compiler folds away.
    """

    var code: Int
    """The field, as one of the twenty two values below."""

    comptime YEAR = Self(FIELD_YEAR)
    """The calendar year, negative before the year zero."""

    comptime MONTH = Self(FIELD_MONTH)
    """The month, from 1 for January to 12 for December."""

    comptime DAY = Self(FIELD_DAY)
    """The day of the month, from 1."""

    comptime HOUR = Self(FIELD_HOUR)
    """The hour, from 0 to 23."""

    comptime MINUTE = Self(FIELD_MINUTE)
    """The minute, from 0 to 59."""

    comptime SECOND = Self(FIELD_SECOND)
    """The second, from 0 to 59. There are no leap seconds in this calendar."""

    comptime MICROSECOND = Self(FIELD_MICROSECOND)
    """The microseconds below the second, from 0 to 999999. Always zero on a
    column stored in seconds, and never the whole sub second part on a column
    stored in nanoseconds, where the last three digits are the nanosecond."""

    comptime NANOSECOND = Self(FIELD_NANOSECOND)
    """The nanoseconds below the microsecond, from 0 to 999. Zero on every
    column that is not stored in nanoseconds."""

    comptime DAY_OF_WEEK = Self(FIELD_DAY_OF_WEEK)
    """The day of the week, from 0 for Monday, which is pandas' numbering and
    not the ISO one and not C's."""

    comptime DAY_OF_YEAR = Self(FIELD_DAY_OF_YEAR)
    """The day of the year, from 1 for the first of January."""

    comptime QUARTER = Self(FIELD_QUARTER)
    """The quarter, from 1 to 4."""

    comptime DAYS_IN_MONTH = Self(FIELD_DAYS_IN_MONTH)
    """How many days are in this row's month, 28 to 31."""

    comptime IS_LEAP_YEAR = Self(FIELD_IS_LEAP_YEAR)
    """Whether this row's year has a 29 February in it."""

    comptime IS_MONTH_START = Self(FIELD_IS_MONTH_START)
    """Whether this row is the first day of its month."""

    comptime IS_MONTH_END = Self(FIELD_IS_MONTH_END)
    """Whether this row is the last day of its month."""

    comptime IS_QUARTER_START = Self(FIELD_IS_QUARTER_START)
    """Whether this row is the first day of January, April, July or October."""

    comptime IS_QUARTER_END = Self(FIELD_IS_QUARTER_END)
    """Whether this row is the last day of March, June, September or
    December."""

    comptime IS_YEAR_START = Self(FIELD_IS_YEAR_START)
    """Whether this row is the first of January."""

    comptime IS_YEAR_END = Self(FIELD_IS_YEAR_END)
    """Whether this row is the thirty first of December."""

    comptime ISO_YEAR = Self(FIELD_ISO_YEAR)
    """The year the row's ISO week belongs to, which is not always the calendar
    year: the last days of December can be in the next ISO year and the first
    days of January can be in the previous one."""

    comptime ISO_WEEK = Self(FIELD_ISO_WEEK)
    """The ISO week, from 1 to 53."""

    comptime ISO_DAY = Self(FIELD_ISO_DAY)
    """The ISO day of the week, from 1 for Monday to 7 for Sunday."""

    def __eq__(self, other: Self) -> Bool:
        """Compares two fields.

        Args:
            other: The field to compare against.

        Returns:
            True if they are the same field.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two fields for difference.

        Args:
            other: The field to compare against.

        Returns:
            True if they are different fields.
        """
        return self.code != other.code

    def write_to(self, mut writer: Some[Writer]):
        """Writes the field under the name pandas gives it.

        The lookup is unrolled rather than indexed because a `comptime` list of
        names does not materialize into a runtime value, and unrolling it is
        what a caller wants anyway: it collapses to a chain of comparisons
        against a value in a register.

        Args:
            writer: Where to write.
        """
        comptime for code in FIELD_CODES:
            if self.code == code:
                comptime name = NAMES[code]
                writer.write(name)
                return
        writer.write("field ", self.code)


comptime NAMES = [
    StaticString("year"),
    StaticString("month"),
    StaticString("day"),
    StaticString("hour"),
    StaticString("minute"),
    StaticString("second"),
    StaticString("microsecond"),
    StaticString("nanosecond"),
    StaticString("dayofweek"),
    StaticString("dayofyear"),
    StaticString("quarter"),
    StaticString("days_in_month"),
    StaticString("is_leap_year"),
    StaticString("is_month_start"),
    StaticString("is_month_end"),
    StaticString("is_quarter_start"),
    StaticString("is_quarter_end"),
    StaticString("is_year_start"),
    StaticString("is_year_end"),
    StaticString("isoyear"),
    StaticString("isoweek"),
    StaticString("isoday"),
]
"""The pandas name of each field, in code order, for error messages.

The last three are not pandas names. `isocalendar` returns a frame whose columns
are called year, week and day, and two of those already mean something else on
`dt`, so the three carry a spelling of their own here and `field_named` refuses
it. They are dispatchable and not nameable, which is what they are in pandas."""


@fieldwise_init
struct Civil[w: Int](ImplicitlyCopyable, Movable):
    """One register of day numbers, taken apart into calendar fields.

    Every field is computed whether or not the caller wants it, because the
    caller is a loop whose field is a compile time parameter and the compiler
    deletes what that loop does not read. Writing this as twelve small functions
    that each redo the conversion would cost twelve conversions on the paths
    that want two fields at once, which is what the `is_` predicates all are.

    Parameters:
        w: How many lanes.
    """

    var year: SIMD[DType.int64, Self.w]
    """The calendar year, so 1969 and not the March year 1969 belongs to."""

    var month: SIMD[DType.int64, Self.w]
    """The month from 1 to 12."""

    var day: SIMD[DType.int64, Self.w]
    """The day of the month from 1."""

    var day_of_year: SIMD[DType.int64, Self.w]
    """The day of the year from 1."""

    var days_in_month: SIMD[DType.int64, Self.w]
    """How long this row's month is."""

    var is_leap: SIMD[DType.bool, Self.w]
    """Whether this row's year is a leap year."""


def civil_from_days[w: Int](z: SIMD[DType.int64, w]) -> Civil[w]:
    """Turns days since 1970-01-01 into a civil date.

    Hinnant's algorithm, unrolled far enough to also answer the day of the year,
    the length of the month and whether the year is a leap year, all three of
    which fall out of intermediate values the conversion has already computed.

    Args:
        z: Days since the epoch, negative before it.

    Parameters:
        w: How many lanes.

    Returns:
        The date each lane names.
    """
    # These two lines are the only ones that touch a negative number. `//`
    # rounds down, so the remainder is in [0, 146096] on both sides of the
    # epoch, and everything below here divides a value that cannot be negative.
    var shifted = z + DAYS_TO_MARCH
    var era = shifted // DAYS_PER_ERA
    var day_of_era = shifted - era * DAYS_PER_ERA

    # The year within the era, found by removing the leap days: one every four
    # years, minus one every hundred, plus one every four hundred.
    var year_of_era = (
        day_of_era
        - day_of_era // 1460
        + day_of_era // 36524
        - day_of_era // 146096
    ) // 365
    var march_year = year_of_era + era * 400
    var day_of_march_year = day_of_era - (
        365 * year_of_era + year_of_era // 4 - year_of_era // 100
    )

    # The month lengths from March are 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    # 31, 28, which is close enough to a repeating pattern that this one
    # division recovers the month index from the day of the year. That is the
    # whole reason the year was moved to March.
    var march_month = (5 * day_of_march_year + 2) // 153
    var day = day_of_march_year - (153 * march_month + 2) // 5 + 1

    # March is month index 0, so index 10 and 11 are January and February and
    # belong to the next calendar year.
    var early = march_month.ge(10)
    var month = early.select(march_month - 9, march_month + 3)
    var year = march_year + early.cast[DType.int64]()

    var is_leap = (year % 4).eq(0) & ((year % 100).ne(0) | (year % 400).eq(0))
    var leap_day = is_leap.cast[DType.int64]()

    # January and February are the 306th day of the March year onwards, so
    # subtracting 305 turns 306 into 1. March onwards is 60 days into the
    # calendar year, 61 in a leap year, and the leap day is behind it either
    # way, which is why the correction is a constant rather than a table.
    var day_of_year = early.select(
        day_of_march_year - 305, day_of_march_year + 60 + leap_day
    )

    # The difference of two cumulative month lengths is the length of the month
    # between them, for every month except the last one of the March year, which
    # is February and is the one the cycle was arranged to leave until last.
    var length = (153 * (march_month + 1) + 2) // 5 - (
        153 * march_month + 2
    ) // 5
    var days_in_month = month.eq(2).select(28 + leap_day, length)

    return Civil[w](
        year=year,
        month=month,
        day=day,
        day_of_year=day_of_year,
        days_in_month=days_in_month,
        is_leap=is_leap,
    )


def extract_field[
    src: DType, field: Int, dst: DType
](a: Array[src], per_day: Int64, per_second: Int64) raises -> Array[dst]:
    """Reads one field out of every row of a temporal column.

    The same shape as the other elementwise kernels: compute across the whole
    values buffer including the null rows and repair them afterwards, because a
    branch per row costs more than the arithmetic it skips. A null row holds
    zero, which is the epoch, which converts to a real date and is then blanked.

    The two divisions by the unit are by a value that is a constant at run time
    and not at compile time, so they are real divide instructions rather than
    the multiply and shift a literal would become. Making the unit a parameter
    would fix that and would multiply the instantiations of this function by
    four, so it is left until a profile asks for it.

    Args:
        a: The column, holding whole units since the epoch.
        per_day: How many of that unit make a day.
        per_second: How many of that unit make a second.

    Parameters:
        src: The physical dtype of the column, int64 or int32.
        field: The field code.
        dst: The dtype that field answers with, which `field_dtype` decides.

    Returns:
        A column of `dst`, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[DType.int64]()

    var n = len(a)
    # Every row is written below, the null ones included, so the allocation does
    # not need the pass that zeroes it.
    var out = Array[dst](overwritten=n)
    var validity = Bitmap(copy=a.data.validity)

    def compute(start: Int, stop: Int) raises {mut out, imm}:
        var source = a.unsafe_ptr()
        var target = out.unsafe_ptr()
        var i = start
        while i < stop:
            var value = (
                source.unsafe_offset(i)
                .unsafe_load[width=width]()
                .cast[DType.int64]()
            )

            comptime if field >= FIELD_HOUR and field <= FIELD_NANOSECOND:
                # The clock fields never look at the calendar. The day number is
                # still needed to get the remainder below it, and that is the
                # one division here that has to round down.
                var days = value // per_day
                var rest = value - days * per_day

                comptime if field == FIELD_HOUR:
                    target.unsafe_offset(i).unsafe_store(
                        (rest // (per_second * 3600)).cast[dst]()
                    )
                elif field == FIELD_MINUTE:
                    target.unsafe_offset(i).unsafe_store(
                        ((rest // (per_second * 60)) % 60).cast[dst]()
                    )
                elif field == FIELD_SECOND:
                    target.unsafe_offset(i).unsafe_store(
                        ((rest // per_second) % 60).cast[dst]()
                    )
                else:
                    # Everything below the second, in nanoseconds, whatever the
                    # column is stored in. The scaling is written as a multiply
                    # then a divide so that one expression covers all four
                    # units, and it cannot overflow: the remainder is under a
                    # second, so the product is at most ten to the eighteenth.
                    var rate = max(per_second, 1)
                    var below = (rest % rate) * 1_000_000_000 // rate

                    comptime if field == FIELD_MICROSECOND:
                        # pandas caps this at 999999 and puts the last three
                        # digits in `nanosecond`, so a nanosecond column drops
                        # them here rather than answering a seven digit number.
                        target.unsafe_offset(i).unsafe_store(
                            (below // 1_000).cast[dst]()
                        )
                    else:
                        target.unsafe_offset(i).unsafe_store(
                            (below % 1_000).cast[dst]()
                        )
            else:
                var days = value // per_day

                comptime if field == FIELD_DAY_OF_WEEK:
                    # 1970-01-01 was a Thursday, which pandas numbers 3, and the
                    # remainder of a negative day number is still in [0, 6]
                    # because `%` rounds towards minus infinity as `//` does.
                    target.unsafe_offset(i).unsafe_store(
                        ((days + 3) % 7).cast[dst]()
                    )
                elif field >= FIELD_ISO_YEAR:
                    # The whole of the ISO calendar is one sentence: a week
                    # belongs to the year its Thursday falls in. So find that
                    # Thursday, convert it, and read the answer off the ordinary
                    # calendar fields of a different day. That is why the last
                    # day of 1969 is in ISO year 1970 without a single special
                    # case, and why none of this needs a table of year lengths.
                    var weekday = (days + 3) % 7
                    var thursday = civil_from_days(days - weekday + 3)

                    comptime if field == FIELD_ISO_YEAR:
                        target.unsafe_offset(i).unsafe_store(
                            thursday.year.cast[dst]()
                        )
                    elif field == FIELD_ISO_WEEK:
                        # The Thursday of week one is always in the first seven
                        # days of its year, so this division needs no offset.
                        target.unsafe_offset(i).unsafe_store(
                            ((thursday.day_of_year - 1) // 7 + 1).cast[dst]()
                        )
                    else:
                        target.unsafe_offset(i).unsafe_store(
                            (weekday + 1).cast[dst]()
                        )
                else:
                    var civil = civil_from_days(days)

                    comptime if field == FIELD_YEAR:
                        target.unsafe_offset(i).unsafe_store(
                            civil.year.cast[dst]()
                        )
                    elif field == FIELD_MONTH:
                        target.unsafe_offset(i).unsafe_store(
                            civil.month.cast[dst]()
                        )
                    elif field == FIELD_DAY:
                        target.unsafe_offset(i).unsafe_store(
                            civil.day.cast[dst]()
                        )
                    elif field == FIELD_DAY_OF_YEAR:
                        target.unsafe_offset(i).unsafe_store(
                            civil.day_of_year.cast[dst]()
                        )
                    elif field == FIELD_QUARTER:
                        target.unsafe_offset(i).unsafe_store(
                            ((civil.month - 1) // 3 + 1).cast[dst]()
                        )
                    elif field == FIELD_DAYS_IN_MONTH:
                        target.unsafe_offset(i).unsafe_store(
                            civil.days_in_month.cast[dst]()
                        )
                    elif field == FIELD_IS_LEAP_YEAR:
                        target.unsafe_offset(i).unsafe_store(
                            civil.is_leap.cast[dst]()
                        )
                    elif field == FIELD_IS_MONTH_START:
                        target.unsafe_offset(i).unsafe_store(
                            civil.day.eq(1).cast[dst]()
                        )
                    elif field == FIELD_IS_MONTH_END:
                        target.unsafe_offset(i).unsafe_store(
                            civil.day.eq(civil.days_in_month).cast[dst]()
                        )
                    elif field == FIELD_IS_QUARTER_START:
                        target.unsafe_offset(i).unsafe_store(
                            (
                                civil.day.eq(1) & ((civil.month - 1) % 3).eq(0)
                            ).cast[dst]()
                        )
                    elif field == FIELD_IS_QUARTER_END:
                        target.unsafe_offset(i).unsafe_store(
                            (
                                civil.day.eq(civil.days_in_month)
                                & (civil.month % 3).eq(0)
                            ).cast[dst]()
                        )
                    elif field == FIELD_IS_YEAR_START:
                        target.unsafe_offset(i).unsafe_store(
                            (civil.day.eq(1) & civil.month.eq(1)).cast[dst]()
                        )
                    else:
                        target.unsafe_offset(i).unsafe_store(
                            (civil.day.eq(31) & civil.month.eq(12)).cast[dst]()
                        )
            i += width

        # These rows are in this core's cache right now, so the repair is nearly
        # free here and is a second walk over the column anywhere else.
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)
    out.data.validity = validity^
    return out^


def _units_per_day(t: LogicalType) raises -> Int64:
    """Returns how many of a temporal column's integers make one day.

    Args:
        t: The column type.

    Returns:
        1 for a date column and 86400 times the unit's rate for a timestamp.

    Raises:
        Error: If the type is not one whose values are instants, or if it is a
            timestamp that carries a time zone.
    """
    if t.kind == TypeKind.DATE:
        return 1
    if t.kind != TypeKind.TIMESTAMP:
        raise Error(
            "temporal: the calendar fields are only defined on a date or a"
            " timestamp column, and this one is "
            + String(t)
        )
    _refuse_zone(t)
    return t.unit.per_second() * SECONDS_PER_DAY


def _refuse_zone(t: LogicalType) raises:
    """Refuses a column that is still on a clock.

    Nothing reaches this through a public entry point, because each of them
    turns a zoned column into the readings it stands for first and refuses the
    zones that cannot be turned. It stays because it is the backstop for the
    next entry point somebody writes, and the bug it catches is an hour that is
    silently seven off rather than a crash.

    Args:
        t: The column type.

    Raises:
        Error: If the column carries a zone at all.
    """
    if not t.zone.is_naive():
        raise Error(
            "temporal: a column in "
            + String(t.zone)
            + " reached a calendar kernel without being read against its own"
            " clock first, which is a bug in firepanda rather than in the call"
        )


def zone_offset(t: LogicalType) raises -> Int64:
    """Returns what separates a column's stored instants from its readings.

    A zoned column holds UTC and a name. The number a person reads off it is the
    instant plus whatever the zone is ahead of UTC at that moment, and for a
    zone that names its own offset that is one constant for the whole column.
    For a zone that names a rule it is not a constant at all, and there is no
    honest answer here without the IANA database.

    Args:
        t: The column type.

    Returns:
        How many of the column's own units to add to an instant to reach the
        reading, which is zero for a naive column and for UTC.

    Raises:
        Error: If the column is on a clock whose offset is a rule rather than a
            number.
    """
    if t.kind != TypeKind.TIMESTAMP or t.zone.is_naive():
        return 0
    var seconds = t.zone.fixed_offset()
    if not seconds:
        raise Error(
            "temporal: reading a wall clock in "
            + String(t.zone)
            + " needs a time zone database firepanda does not have yet, since"
            " what that zone is ahead of UTC changes twice a year and the"
            " stored instants are UTC"
        )
    return seconds.value() * t.unit.per_second()


def _shifted(a: Array[DType.int64], by: Int64) raises -> Array[DType.int64]:
    """Adds one constant to every row of a column of instants.

    Args:
        a: The column.
        by: How many of its own units to add.

    Returns:
        The shifted column, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[DType.int64]()

    var n = len(a)
    var out = Array[DType.int64](overwritten=n)
    var validity = Bitmap(copy=a.data.validity)

    def compute(start: Int, stop: Int) raises {mut out, imm}:
        var source = a.unsafe_ptr()
        var target = out.unsafe_ptr()
        var i = start
        while i < stop:
            var value = source.unsafe_offset(i).unsafe_load[width=width]()
            target.unsafe_offset(i).unsafe_store(value + by)
            i += width
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)
    out.data.validity = validity^
    return out^


def local_readings(a: AnyArray) raises -> AnyArray:
    """Returns a zoned column as the naive readings it stands for.

    This is what every calendar kernel below wants, since each of them reads the
    integers as they are stored and the integers are UTC. A naive column comes
    back unchanged, which is the case that matters for speed, and a zoned one
    pays one pass.

    Args:
        a: A timestamp column, on a clock or not.

    Returns:
        A naive timestamp column at the same resolution.

    Raises:
        Error: If the column is on a clock whose offset is a rule.
    """
    var shift = zone_offset(a.type)
    var naive = LogicalType.timestamp(a.type.unit, TimeZone())
    if shift == 0:
        var same = AnyArray(copy=a)
        same.type = naive
        return same^
    return AnyArray(
        _shifted(a.as_typed_view[DType.int64](), shift).into_data(), naive
    )


def _back_on_the_clock(
    var readings: AnyArray, t: LogicalType
) raises -> AnyArray:
    """Puts a column of readings back on the clock it was read from.

    Args:
        readings: A naive timestamp column, at the same resolution as `t`.
        t: The type it is going back to.

    Returns:
        A column of that type.

    Raises:
        Error: If the type is on a clock whose offset is a rule, which the
            caller has already met on the way in.
    """
    var shift = zone_offset(t)
    if shift == 0:
        readings.type = t
        return readings^
    return AnyArray(
        _shifted(readings.as_typed_view[DType.int64](), -shift).into_data(), t
    )


def temporal_tz_convert(a: AnyArray, zone: StringSlice) raises -> AnyArray:
    """Reads a zoned column against another clock.

    This is `dt.tz_convert` and it moves nothing. A zoned column holds UTC
    instants and a name saying which clock they are read against, so converting
    changes the name and leaves every integer where it was. The instant a row
    denotes is the same instant before and after, which is the one sentence that
    separates this from `tz_localize`.

    It follows that this works for every zone there is while localising works
    only for the zones that name their own offset, because converting never asks
    what the offset is. It also follows that the target name is not checked
    against anything, since there is nothing here to check it against. pandas
    would refuse a name its database does not hold and this does not.

    Args:
        a: A zoned timestamp column.
        zone: The name to read it against.

    Returns:
        A column of the same instants under the new name.

    Raises:
        Error: If the column is not a timestamp, if it carries no zone, or if
            the name is longer than any zone name is.
    """
    if a.type.kind != TypeKind.TIMESTAMP:
        raise Error(
            "temporal: tz_convert is only defined on a timestamp column, and"
            " this one is "
            + String(a.type)
        )
    if a.type.zone.is_naive():
        # pandas' own sentence first and ours after it. The first half is what
        # somebody pastes into a search box when they hit this, and a message
        # that describes our internals accurately and shares no words with the
        # pandas documentation sends them nowhere.
        raise Error(
            "temporal: Cannot convert tz-naive timestamps, use tz_localize to"
            " localize. This column carries no zone, so there is nothing to"
            " read it against, and tz_localize is the one that puts a clock on"
            " a column of readings"
        )
    var out = AnyArray(copy=a)
    out.type = LogicalType.timestamp(a.type.unit, TimeZone(zone))
    return out^


def temporal_tz_localize(a: AnyArray, zone: StringSlice) raises -> AnyArray:
    """Puts a clock on a column of readings.

    This is `dt.tz_localize` with a name, and it is the operation `tz_convert`
    is not. The readings stay what they were and the instants move, because a
    reading of nine o'clock is a different moment in each zone somebody might
    have taken it in.

    Moving them needs to know what the zone is ahead of UTC, so this works for
    the zones that name their own offset and refuses the ones that name a rule.
    The refused ones are also where the hard questions live, since a reading in
    the hour a clock skips forward denotes no instant at all and a reading in
    the hour it repeats denotes two, and neither can happen in a zone whose
    offset never changes.

    Args:
        a: A naive timestamp column.
        zone: The name to put on it.

    Returns:
        A column on that clock, denoting instants that are the readings less
        whatever the zone is ahead of UTC.

    Raises:
        Error: If the column is not a naive timestamp, or if the zone names a
            rule rather than a number.
    """
    if a.type.kind != TypeKind.TIMESTAMP:
        raise Error(
            "temporal: tz_localize is only defined on a timestamp column, and"
            " this one is "
            + String(a.type)
        )
    if not a.type.zone.is_naive():
        raise Error(
            "temporal: Already tz-aware, use tz_convert to convert. This column"
            " is already on "
            + String(a.type.zone)
            + ", and tz_convert is the one that reads it against another clock"
        )
    var wanted = LogicalType.timestamp(a.type.unit, TimeZone(zone))
    return _back_on_the_clock(AnyArray(copy=a), wanted)


def temporal_tz_localize_none(a: AnyArray) raises -> AnyArray:
    """Takes the clock off a zoned column, keeping the reading.

    This is `dt.tz_localize(None)`, and it is the exact opposite of what
    `tz_convert` does. The reading stays what it was and the instant moves,
    where converting moves the reading and keeps the instant. A column of New
    York afternoons becomes a column of naive afternoons rather than a column of
    the evenings in UTC that they were stored as.

    Args:
        a: A zoned timestamp column.

    Returns:
        A naive timestamp column holding the readings.

    Raises:
        Error: If the column is not a timestamp, if it carries no zone, or if
            the zone names a rule rather than a number.
    """
    if a.type.kind != TypeKind.TIMESTAMP:
        raise Error(
            "temporal: tz_localize is only defined on a timestamp column, and"
            " this one is "
            + String(a.type)
        )
    if a.type.zone.is_naive():
        raise Error(
            "temporal: this column carries no zone, so there is none to take"
            " off it"
        )
    return local_readings(a)


def _units_per_second(t: LogicalType) -> Int64:
    """Returns how many of a temporal column's integers make one second.

    Args:
        t: The column type, already checked by `_units_per_day`.

    Returns:
        The rate, or zero for a date column, which has no clock in it at all.
    """
    if t.kind == TypeKind.DATE:
        return 0
    return t.unit.per_second()


def field_named(name: StringSlice) raises -> TemporalField:
    """Looks a field up by the name pandas gives it.

    Here so that a caller holding a string does not have to keep its own copy of
    the table. The Python layer will hold one of those strings and the
    conformance driver holds another, and two copies of a nineteen row table is
    two chances to spell `days_in_month` differently. Nineteen rather than
    twenty two, because the three ISO fields below have no name on `dt`.

    The three `isocalendar` fields are deliberately not reachable from here.
    They are columns of a frame rather than names on `dt`, and two of the three
    are spelled the same as fields that already exist, so a caller that asked
    for `year` and got the ISO year would be wrong for eleven months of the
    year and right for the twelfth, which is the worst way to be wrong.

    Args:
        name: The pandas spelling, so `dayofweek` rather than `day_of_week`.

    Returns:
        The field.

    Raises:
        Error: If nothing is called that.
    """
    comptime for code in FIELD_CODES:
        comptime if code < FIELD_ISO_YEAR:
            comptime spelling = NAMES[code]
            if name == spelling:
                return TemporalField(code)
    raise Error(
        "temporal: there is no field called " + String(name) + " on a datetime"
    )


def temporal_field(a: AnyArray, field: TemporalField) raises -> AnyArray:
    """Reads one calendar or clock field out of a temporal column.

    Args:
        a: A date or a naive timestamp column.
        field: Which field.

    Returns:
        An int32 column for the twelve fields that are numbers, a bool column
        for the seven that are predicates and a uint32 column for the three that
        make up the ISO calendar, null wherever the input is null.

    Raises:
        Error: If the column is not temporal, if it is on a clock whose offset
            is a rule, or if the field code is not one of the twenty two.
    """
    if not a.type.zone.is_naive():
        # The turn happens once here rather than inside the five kernels below,
        # each of which would otherwise have to remember that the integers it
        # was handed are UTC.
        return temporal_field(local_readings(a), field)

    var per_day = _units_per_day(a.type)
    var per_second = _units_per_second(a.type)

    comptime for code in FIELD_CODES:
        if field.code == code:
            comptime result = field_dtype(code)
            if a.type.kind == TypeKind.DATE:
                ref days = a.as_typed_view[DType.int32]()
                return AnyArray(
                    extract_field[DType.int32, code, result](
                        days, per_day, per_second
                    )
                )
            ref stamps = a.as_typed_view[DType.int64]()
            return AnyArray(
                extract_field[DType.int64, code, result](
                    stamps, per_day, per_second
                )
            )
    raise Error("temporal: " + String(field.code) + " is not a field code")


def _days_of(
    a: Array[DType.int64], per_day: Int64
) raises -> Array[DType.int32]:
    """Turns a column of instants into a column of day numbers.

    Args:
        a: The column, holding whole units since the epoch.
        per_day: How many of that unit make a day.

    Returns:
        Days since the epoch, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[DType.int64]()

    var n = len(a)
    var out = Array[DType.int32](overwritten=n)
    var validity = Bitmap(copy=a.data.validity)

    def compute(start: Int, stop: Int) raises {mut out, imm}:
        var source = a.unsafe_ptr()
        var target = out.unsafe_ptr()
        var i = start
        while i < stop:
            var value = source.unsafe_offset(i).unsafe_load[width=width]()
            target.unsafe_offset(i).unsafe_store(
                (value // per_day).cast[DType.int32]()
            )
            i += width
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)
    out.data.validity = validity^
    return out^


def _midnights_of(
    a: Array[DType.int64], per_day: Int64
) raises -> Array[DType.int64]:
    """Rounds a column of instants down to the start of its day.

    Args:
        a: The column, holding whole units since the epoch.
        per_day: How many of that unit make a day.

    Returns:
        The same unit, with everything below the day removed, null wherever the
        input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[DType.int64]()

    var n = len(a)
    var out = Array[DType.int64](overwritten=n)
    var validity = Bitmap(copy=a.data.validity)

    def compute(start: Int, stop: Int) raises {mut out, imm}:
        var source = a.unsafe_ptr()
        var target = out.unsafe_ptr()
        var i = start
        while i < stop:
            var value = source.unsafe_offset(i).unsafe_load[width=width]()
            target.unsafe_offset(i).unsafe_store((value // per_day) * per_day)
            i += width
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)
    out.data.validity = validity^
    return out^


def temporal_date(a: AnyArray) raises -> AnyArray:
    """Drops the clock from a temporal column, leaving a date column.

    This is `dt.date`. pandas answers it with an object column of Python `date`
    values, which is the only shape it has for a date on the numpy backend, and
    firepanda answers it with a date32 column, which is the same information in
    a form that is not one Python object per row.

    Args:
        a: A date or a naive timestamp column.

    Returns:
        A date32 column, null wherever the input is null.

    Raises:
        Error: If the column is not temporal, or is on a clock whose offset is
            a rule.
    """
    if not a.type.zone.is_naive():
        return temporal_date(local_readings(a))

    var per_day = _units_per_day(a.type)
    if a.type.kind == TypeKind.DATE:
        return AnyArray(copy=a)
    return AnyArray(
        _days_of(a.as_typed_view[DType.int64](), per_day).into_data(),
        LogicalType.DATE32,
    )


def temporal_normalize(a: AnyArray) raises -> AnyArray:
    """Sets the clock of every row to midnight, keeping the column's type.

    This is `dt.normalize`, and it is `dt.date` followed by putting the days
    back into the original unit rather than a different operation. The type does
    not change, which is the difference between this and `dt.date` and is the
    reason both exist.

    Args:
        a: A date or a naive timestamp column.

    Returns:
        A column of the same type, null wherever the input is null.

    Raises:
        Error: If the column is not temporal, or is on a clock whose offset is
            a rule.
    """
    if not a.type.zone.is_naive():
        # The midnight is the local one, so the reading has to come off the
        # clock and the answer has to go back on it. That round trip is exact
        # for a zone whose offset never moves and is why the zones that are a
        # rule are refused rather than approximated.
        return _back_on_the_clock(temporal_normalize(local_readings(a)), a.type)

    var per_day = _units_per_day(a.type)
    if a.type.kind == TypeKind.DATE:
        return AnyArray(copy=a)
    return AnyArray(
        _midnights_of(a.as_typed_view[DType.int64](), per_day).into_data(),
        a.type,
    )


def _check_elapsed(t: LogicalType) raises:
    """Refuses a column that is not a length of time.

    Args:
        t: The column type.

    Raises:
        Error: If the type is anything other than a duration, with the two
            temporal types that are a point in time named separately, since
            asking a timestamp how many days it is is a different mistake from
            asking an integer.
    """
    if t.kind == TypeKind.DURATION:
        return
    if t.kind == TypeKind.TIMESTAMP or t.kind == TypeKind.DATE:
        raise Error(
            "temporal: "
            + String(t)
            + " is a point in time rather than a length of one, and how long a"
            " point in time is has no answer"
        )
    raise Error("temporal: " + String(t) + " is not a length of time")


def temporal_total_seconds(a: AnyArray) raises -> AnyArray:
    """Counts a column of elapsed times in seconds, fractions included.

    This is `dt.total_seconds`, and pandas answers it in float64 whatever the
    column's resolution is. That is a lossy answer on a long duration, since a
    float64 runs out of mantissa at 2**53 and a count of microseconds passes
    that at about 285 years, so a difference of one microsecond between two
    century long spans does not survive the division. The same answer is given
    here, because the number a caller compares against came out of pandas.

    Args:
        a: A duration column.

    Returns:
        A float64 column of seconds, null wherever the input is null.

    Raises:
        Error: If the column is not a duration, or what the morsel runtime
            raises.
    """
    _check_elapsed(a.type)
    comptime width = simd_width_of[DType.int64]()

    ref counts = a.as_typed_view[DType.int64]()
    var per_second = Float64(a.type.unit.per_second())
    var n = len(counts)
    var out = Array[DType.float64](overwritten=n)
    var validity = Bitmap(copy=counts.data.validity)

    def compute(start: Int, stop: Int) raises {mut out, imm}:
        var source = counts.unsafe_ptr()
        var target = out.unsafe_ptr()
        var i = start
        while i < stop:
            var value = source.unsafe_offset(i).unsafe_load[width=width]()
            target.unsafe_offset(i).unsafe_store(
                value.cast[DType.float64]() / per_second
            )
            i += width

        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)
    out.data.validity = validity^
    return AnyArray(out^)


def temporal_duration_days(a: AnyArray) raises -> AnyArray:
    """Counts the whole days in a column of elapsed times.

    This is `dt.days`, and the rounding is downward rather than toward zero,
    which is what pandas gives and is worth saying because the sign makes it
    visible: minus one microsecond is minus one day here and would be zero days
    under a truncating division. `//` in Mojo floors, so the loop does not have
    to correct anything.

    pandas answers int64 on a column with no nulls and float64 on a column with
    some, because a numpy int64 has no missing value to put in the gap. This
    answers int64 either way and puts the missing behind the validity bit. That
    is the package's general position on the question and it is written down in
    issue #170 rather than here.

    Args:
        a: A duration column.

    Returns:
        An int64 column of whole days, null wherever the input is null.

    Raises:
        Error: If the column is not a duration, or what the morsel runtime
            raises.
    """
    _check_elapsed(a.type)
    comptime width = simd_width_of[DType.int64]()

    ref counts = a.as_typed_view[DType.int64]()
    var per_day = a.type.unit.per_second() * SECONDS_PER_DAY
    var n = len(counts)
    var out = Array[DType.int64](overwritten=n)
    var validity = Bitmap(copy=counts.data.validity)

    def compute(start: Int, stop: Int) raises {mut out, imm}:
        var source = counts.unsafe_ptr()
        var target = out.unsafe_ptr()
        var i = start
        while i < stop:
            var value = source.unsafe_offset(i).unsafe_load[width=width]()
            target.unsafe_offset(i).unsafe_store(value // per_day)
            i += width

        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)
    out.data.validity = validity^
    return AnyArray(out^)


def temporal_to_duration(a: AnyArray, unit: TimeUnit) raises -> AnyArray:
    """Reads a column of whole numbers as a column of elapsed times.

    This is `pandas.to_timedelta` with a unit given, and on an integer column it
    is a relabelling and not a conversion: the integers are already the counts
    and the unit says what they are counts of. So this allocates nothing and
    copies the buffer once, and a null stays a null rather than becoming a zero
    length span.

    A column that is already a duration is answered unchanged rather than
    rescaled, because that is what pandas does with one: the unit argument is
    ignored when the column already carries a resolution of its own.

    Args:
        a: An integer column, or a duration column.
        unit: The resolution the integers are counts of.

    Returns:
        A duration column at that unit, null wherever the input is null.

    Raises:
        Error: If the column is neither an integer nor a duration, since text
            parsing is a different piece of work and float seconds are another.
    """
    if a.type.kind == TypeKind.DURATION:
        return AnyArray(copy=a)
    if a.type.physical != DType.int64:
        raise Error(
            "temporal: to_timedelta needs a column of whole numbers and this"
            " one is "
            + String(a.type)
        )
    return AnyArray(
        Array[DType.int64](copy=a.as_typed_view[DType.int64]()).into_data(),
        LogicalType.duration(unit),
    )


comptime MAX_COUNT = Int64(922_337_203_685_477_580)
"""The largest count a frequency can carry before one more digit would leave the
range of an int64. It is `Int64.MAX // 10`, written out so the check in front of
the multiply reads as a comparison rather than as arithmetic."""

comptime MAX_SCALE = Int64(100_000_000)
"""The largest power of ten the digits after a decimal point may reach before
one more of them would put the divisor out of range."""


def _nanos_per_unit(t: LogicalType) raises -> Int64:
    """Returns how many nanoseconds one of a timestamp column's integers is.

    Args:
        t: The column type.

    Returns:
        1000000000 for a second column, down to 1 for a nanosecond one.

    Raises:
        Error: If the column is not a timestamp, or if it carries a time zone.
    """
    if t.kind != TypeKind.TIMESTAMP:
        raise Error(
            "temporal: rounding to a frequency is only defined on a timestamp"
            " column, and this one is "
            + String(t)
        )
    _refuse_zone(t)
    return NANOS_PER_SECOND // t.unit.per_second()


def _alias_nanos(spelling: StringSlice, whole: String) raises -> Int64:
    """Returns how many nanoseconds long one of the fixed frequencies is.

    These seven are the whole list pandas will round to. Everything else it
    knows about, a week, a month end, a quarter, is a non fixed frequency whose
    length depends on where in the calendar it lands, and pandas refuses those
    rather than picking an average, so they are refused here too.

    Args:
        spelling: The letters after the count, so `h` rather than `2h`.
        whole: The frequency the caller wrote, count and all, which is what the
            message names. Naming the alias alone would tell somebody who wrote
            `2xyz` that `xyz` is the problem, which is true and is not the
            string they typed or the string they will search for. It is a copy
            rather than a slice because the alias is a slice of the same string
            and two slices of one origin cannot both be passed.

    Returns:
        The length in nanoseconds.

    Raises:
        Error: If the alias is not one of the seven.
    """
    if spelling == "D":
        return Int64(SECONDS_PER_DAY) * NANOS_PER_SECOND
    if spelling == "h":
        return Int64(3_600) * NANOS_PER_SECOND
    if spelling == "min":
        return Int64(60) * NANOS_PER_SECOND
    if spelling == "s":
        return NANOS_PER_SECOND
    if spelling == "ms":
        return 1_000_000
    if spelling == "us":
        return 1_000
    if spelling == "ns":
        return 1
    raise Error(
        "temporal: Invalid frequency: "
        + whole
        + ". The fixed frequencies, which are the ones that can be rounded to,"
        " are D, h, min, s, ms, us and ns, each with an optional count in front"
        " of it"
    )


def _is_blank(byte: UInt8) -> Bool:
    """Says whether a byte is one pandas trims off the ends of a frequency.

    Args:
        byte: The byte.

    Returns:
        True for a space and for a tab.
    """
    return byte == 32 or byte == 9


def _is_digit(byte: UInt8) -> Bool:
    """Says whether a byte is one of the ten decimal digits.

    Args:
        byte: The byte.

    Returns:
        True for `0` through `9`.
    """
    return byte >= 48 and byte <= 57


def frequency_period(freq: StringSlice, t: LogicalType) raises -> Int64:
    """Turns a pandas frequency string into a count of the column's own units.

    A frequency is an optional sign, an optional count, and one of the seven
    fixed aliases, so `h`, `15min`, `2D` and `1.5h` are all of them. Spaces at
    either end are trimmed, which is what pandas does, and nothing in the middle
    is allowed.

    The count is kept as a whole number over a power of ten rather than as a
    float, so `1.5h` on a second column is fifteen times an hour over ten and
    comes out as exactly 5400 rather than as whatever 1.5 times 3.6e12 rounds
    to. The last division is the one that turns nanoseconds into the column's
    unit and it rounds down, which is why a frequency finer than the column
    itself comes back as zero. That is not an error. pandas answers `floor('ms')`
    on a second column with the column unchanged, and so does the caller here,
    for the same reason: there is nothing below a second in it to remove.

    Args:
        freq: The frequency, as pandas spells it.
        t: The column type, which decides what unit the answer is in.

    Returns:
        The length of one period in the column's own integers, which may be zero
        and may be negative if the frequency carried a minus sign.

    Raises:
        Error: If the column is not a naive timestamp, if the alias is not one
            of the seven, or if the count is too long to be a number.
    """
    var per_unit = _nanos_per_unit(t)
    var raw = freq.as_bytes()
    var start = 0
    var stop = len(raw)
    while start < stop and _is_blank(raw[start]):
        start += 1
    while stop > start and _is_blank(raw[stop - 1]):
        stop -= 1

    var at = start
    var negative = False
    if at < stop and (raw[at] == 43 or raw[at] == 45):
        negative = raw[at] == 45
        at += 1

    var count = Int64(0)
    var scale = Int64(1)
    var digits = 0
    while at < stop and _is_digit(raw[at]):
        if count > MAX_COUNT:
            raise Error(
                "temporal: the count in frequency '"
                + String(freq)
                + "' has more digits than a whole number can hold"
            )
        count = count * 10 + Int64(Int(raw[at]) - 48)
        digits += 1
        at += 1
    if at < stop and raw[at] == 46:
        at += 1
        while at < stop and _is_digit(raw[at]):
            if scale > MAX_SCALE:
                raise Error(
                    "temporal: frequency '"
                    + String(freq)
                    + "' has more than nine digits after the point, and a"
                    " nanosecond is the smallest thing there is to divide"
                )
            count = count * 10 + Int64(Int(raw[at]) - 48)
            scale = scale * 10
            digits += 1
            at += 1
    if digits == 0:
        count = 1
    if negative:
        count = -count

    var nanos = _alias_nanos(
        freq[byte=at:stop], String(freq[byte=start:stop])
    )
    var magnitude = count if count >= 0 else -count
    if magnitude > Int64.MAX // nanos:
        raise Error(
            "temporal: frequency '"
            + String(freq)
            + "' is longer than the number of nanoseconds that fit in an int64"
        )
    return (count * nanos) // (scale * per_unit)


def round_to_period[
    mode: Int
](a: Array[DType.int64], period: Int64) raises -> Array[DType.int64]:
    """Moves every instant in a column onto a multiple of a period.

    All three modes start from the same floored quotient and differ only in
    whether they step it on by one, which is what lets them share a loop. Floor
    keeps it. Ceiling steps it whenever the multiple it names is not the value
    itself. Round steps it when twice the remainder is past the period, and on
    the exact halfway row it steps only from an odd quotient, which is what
    sends a tie to the even multiple.

    The comparison in the round case is against the signed period rather than
    against half of its size, which matters because pandas accepts a negative
    frequency and its own answer for one does not agree with its floor. The
    signed form reproduces both, so there is one rule here and not two.

    Args:
        a: The column, holding whole units since the epoch.
        period: The length of one period in those same units, never zero.

    Parameters:
        mode: `ROUND_DOWN`, `ROUND_UP` or `ROUND_HALF_EVEN`.

    Returns:
        The same unit, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[DType.int64]()

    var n = len(a)
    var out = Array[DType.int64](overwritten=n)
    var validity = Bitmap(copy=a.data.validity)

    # Broadcast once outside the loop, and compare with the named methods. `>`
    # and `!=` on a register are the whole-vector forms, which answer one `Bool`
    # for all the lanes at once and are not what any of this wants.
    var span = SIMD[DType.int64, width](period)
    var odd = SIMD[DType.int64, width](1)
    var none = SIMD[DType.int64, width](0)

    def compute(start: Int, stop: Int) raises {mut out, imm}:
        var source = a.unsafe_ptr()
        var target = out.unsafe_ptr()
        var i = start
        while i < stop:
            var value = source.unsafe_offset(i).unsafe_load[width=width]()
            var quotient = value // span
            comptime if mode == ROUND_UP:
                var landed = quotient * span
                quotient += landed.ne(value).cast[DType.int64]()
            elif mode == ROUND_HALF_EVEN:
                var rest = value - quotient * span
                var twice = rest + rest
                var past = twice.gt(span)
                var tie = twice.eq(span) & (quotient & odd).ne(none)
                quotient += (past | tie).cast[DType.int64]()
            target.unsafe_offset(i).unsafe_store(quotient * span)
            i += width
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)
    out.data.validity = validity^
    return out^


def temporal_round(
    a: AnyArray, freq: StringSlice, mode: Int
) raises -> AnyArray:
    """Rounds a timestamp column to a multiple of a frequency.

    This is `dt.floor`, `dt.ceil` and `dt.round`, which are one kernel at three
    settings. The type does not change, including the resolution, so rounding a
    millisecond column to the hour answers milliseconds that happen to be whole
    hours.

    Args:
        a: A naive timestamp column.
        freq: The frequency, as pandas spells it.
        mode: `ROUND_DOWN`, `ROUND_UP` or `ROUND_HALF_EVEN`.

    Returns:
        A column of the same type, null wherever the input is null.

    Raises:
        Error: If the column is not a timestamp, if it is on a clock whose
            offset is a rule, if the frequency does not parse, or if the mode is
            not one of the three.
    """
    if not a.type.zone.is_naive():
        # pandas rounds the reading on the local clock and the stored integers
        # are UTC, and in a zone that is not on a whole hour offset those are
        # two different answers rather than the same answer written twice.
        return _back_on_the_clock(
            temporal_round(local_readings(a), freq, mode), a.type
        )

    var period = frequency_period(freq, a.type)
    if period == 0:
        # A frequency finer than the column's own unit, and a count of zero,
        # both land here, and pandas answers both with the column unchanged.
        return AnyArray(copy=a)

    ref stamps = a.as_typed_view[DType.int64]()
    if mode == ROUND_DOWN:
        return AnyArray(
            round_to_period[ROUND_DOWN](stamps, period).into_data(), a.type
        )
    if mode == ROUND_UP:
        return AnyArray(
            round_to_period[ROUND_UP](stamps, period).into_data(), a.type
        )
    if mode == ROUND_HALF_EVEN:
        return AnyArray(
            round_to_period[ROUND_HALF_EVEN](stamps, period).into_data(), a.type
        )
    raise Error("temporal: " + String(mode) + " is not a rounding mode")


def unit_named(name: StringSlice) raises -> TimeUnit:
    """Looks a resolution up by the name pandas gives it.

    Args:
        name: One of `s`, `ms`, `us` and `ns`.

    Returns:
        The unit.

    Raises:
        Error: If nothing is called that.
    """
    if name == "s":
        return TimeUnit.SECOND
    if name == "ms":
        return TimeUnit.MILLI
    if name == "us":
        return TimeUnit.MICRO
    if name == "ns":
        return TimeUnit.NANO
    raise Error(
        "temporal: '"
        + String(name)
        + "' is not a resolution; the four Arrow has are s, ms, us and ns"
    )


def _any_beyond(a: Array[DType.int64], limit: Int64) -> Bool:
    """Says whether any row at all sits outside plus or minus a limit.

    This reads the null rows too, which is deliberate and is why the caller does
    not act on the answer directly. A null row usually holds zero and sometimes
    holds whatever the file that produced it left there, so this is a cheap
    vector scan that is allowed to be wrong in one direction and the caller
    settles the rows it flags one at a time. The point is that a column that is
    nowhere near the limit, which is every real column, pays one pass and no
    branches.

    Args:
        a: The column.
        limit: The distance from zero to stay inside.

    Returns:
        True if some row is outside it.
    """
    comptime width = simd_width_of[DType.int64]()

    var n = len(a)
    var source = a.unsafe_ptr()
    var high = SIMD[DType.int64, width](limit)
    var low = SIMD[DType.int64, width](-limit)
    var found = SIMD[DType.bool, width](fill=False)
    var i = 0
    while i + width <= n:
        var value = source.unsafe_offset(i).unsafe_load[width=width]()
        found |= value.gt(high) | value.lt(low)
        i += width
    if found.reduce_or():
        return True
    while i < n:
        if a[i] > limit or a[i] < -limit:
            return True
        i += 1
    return False


def _beyond_and_there(a: Array[DType.int64], limit: Int64) -> Bool:
    """Says whether any row that is really there sits outside a limit.

    Args:
        a: The column.
        limit: The distance from zero to stay inside.

    Returns:
        True if some row that is not null is outside it.
    """
    if not _any_beyond(a, limit):
        return False
    for i in range(len(a)):
        if a.is_valid(i) and (a[i] > limit or a[i] < -limit):
            return True
    return False


def _rescale[
    up: Bool
](a: Array[DType.int64], ratio: Int64) raises -> Array[DType.int64]:
    """Moves a column of instants between two resolutions.

    Args:
        a: The column, holding whole units since the epoch.
        ratio: How many of the finer unit make one of the coarser.

    Parameters:
        up: True to go to the finer unit, which multiplies, and False to go to
            the coarser one, which divides.

    Returns:
        The values in the other unit, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[DType.int64]()

    var n = len(a)
    var out = Array[DType.int64](overwritten=n)
    var validity = Bitmap(copy=a.data.validity)

    def compute(start: Int, stop: Int) raises {mut out, imm}:
        var source = a.unsafe_ptr()
        var target = out.unsafe_ptr()
        var i = start
        while i < stop:
            var value = source.unsafe_offset(i).unsafe_load[width=width]()
            comptime if up:
                target.unsafe_offset(i).unsafe_store(value * ratio)
            else:
                target.unsafe_offset(i).unsafe_store(value // ratio)
            i += width
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)
    out.data.validity = validity^
    return out^


def temporal_as_unit(a: AnyArray, unit: TimeUnit) raises -> AnyArray:
    """Restates a timestamp or a duration column at a different resolution.

    This is `dt.as_unit` on both column types, and it is the same arithmetic for
    both because both are a count of the type's own unit. Going down in
    precision throws away what will not fit and rounds down while doing it, so
    the last second of 1969 in nanoseconds is still the last second of 1969 in
    seconds and not the epoch, and minus one nanosecond is minus one second and
    not zero. Going up is a multiply and does not recover what an earlier trip
    down removed.

    The zone comes through untouched and a zoned column is allowed, because
    nothing here reads the calendar. Which second an instant is does not depend
    on which clock it is being read against, so there is no zone database in the
    way of this one. A duration has no zone to carry.

    The binary arithmetic uses this to reconcile two columns at the finer of
    their resolutions, which is why it takes a duration at all: `s - t` on a
    second column and a nanosecond one is a nanosecond answer and the second
    column has to be multiplied to get there.

    Args:
        a: A timestamp or duration column.
        unit: The resolution to restate it at.

    Returns:
        A column of the same kind at the new resolution, null wherever the input
        is null.

    Raises:
        Error: If the column is neither a timestamp nor a duration, or if going
            up in precision would put a value outside the range of an int64.
    """
    if a.type.kind != TypeKind.TIMESTAMP and a.type.kind != TypeKind.DURATION:
        raise Error(
            "temporal: a resolution is something only a timestamp or a duration"
            " column has, and this one is "
            + String(a.type)
        )

    var have = a.type.unit.per_second()
    var want = unit.per_second()
    if have == want:
        return AnyArray(copy=a)

    ref stamps = a.as_typed_view[DType.int64]()
    var result = LogicalType.timestamp(
        unit, a.type.zone
    ) if a.type.kind == TypeKind.TIMESTAMP else LogicalType.duration(unit)
    if want < have:
        return AnyArray(
            _rescale[False](stamps, have // want).into_data(), result
        )

    var ratio = want // have
    if _beyond_and_there(stamps, Int64.MAX // ratio):
        raise Error(
            "temporal: restating this column in "
            + String(unit)
            + " multiplies every value by "
            + String(ratio)
            + ", and at least one of them does not fit in an int64 afterwards,"
            " which is the range pandas calls out of bounds"
        )
    return AnyArray(_rescale[True](stamps, ratio).into_data(), result)


comptime DAY_NAMES = StaticString(
    "MondayTuesdayWednesdayThursdayFridaySaturdaySunday"
)
"""The seven day names, back to back, in the order `dt.dayofweek` numbers them.

Packed rather than held as a list because a `comptime` list cannot be indexed by
a runtime value, and a chain of seven comparisons per row is a chain of seven
comparisons per row. The offsets are in `day_name_bounds`."""

comptime DAY_ABBREVS = StaticString("MonTueWedThuFriSatSun")
"""The seven abbreviations, which are all three bytes, so this one needs no
offsets at all."""

comptime MONTH_NAMES = StaticString(
    "JanuaryFebruaryMarchAprilMayJuneJulyAugustSeptemberOctoberNovemberDecember"
)
"""The twelve month names, back to back, January first."""

comptime MONTH_ABBREVS = StaticString("JanFebMarAprMayJunJulAugSepOctNovDec")
"""The twelve abbreviations, all three bytes."""


def day_name_bounds() -> List[Int]:
    """Returns where each day name starts in `DAY_NAMES`.

    Eight numbers for seven names, so that a name is the span between two of
    them and the last one needs no special case.

    Returns:
        The offsets, Monday first.
    """
    return [0, 6, 13, 22, 30, 36, 44, 50]


def month_name_bounds() -> List[Int]:
    """Returns where each month name starts in `MONTH_NAMES`.

    Thirteen numbers for twelve names. Index zero is January, so a caller
    holding a month from 1 subtracts one first.

    Returns:
        The offsets, January first.
    """
    return [0, 7, 15, 20, 25, 28, 32, 36, 42, 51, 58, 66, 74]


def _check_nameable(t: LogicalType) raises:
    """Refuses a column that has no calendar in it, or has one in another zone.

    The same two refusals `temporal_field` makes, in one place, because the
    three text producing entry points below all make them and a fourth copy of
    the sentence is a fourth chance to word it differently.

    Args:
        t: The column type.

    Raises:
        Error: If the column is not a date or a timestamp, or if it is zoned.
    """
    if t.kind != TypeKind.TIMESTAMP and t.kind != TypeKind.DATE:
        raise Error(
            "temporal: a name or a format comes off a date or a timestamp"
            " column, and this one is "
            + String(t)
        )
    _refuse_zone(t)


def _named_column[
    src: DType, month: Bool
](
    a: Array[src], per_day: Int64, offsets: List[Int], packed: StringSlice
) raises -> StringArray:
    """Writes the day name or the month name of every row.

    One row at a time, which is what text is. The conversion underneath is the
    same vector one every other field uses, called on a register of one lane,
    because a second scalar copy of Hinnant's algorithm is a second place for it
    to be wrong.

    Args:
        a: The column.
        per_day: How many of the column's units make a day.
        offsets: Where each name starts in `packed`.
        packed: The names, back to back.

    Parameters:
        src: The physical dtype of the column.
        month: Whether the month name is wanted rather than the day name.

    Returns:
        A text column, null wherever the input is null.

    Raises:
        Error: Only what the builder raises.
    """
    var n = len(a)
    var out = StringBuilder(capacity=n)

    for i in range(n):
        if not a.is_valid(i):
            out.append_null()
            continue

        var days = Int64(a[i]) // per_day

        var at: Int
        comptime if month:
            at = Int(civil_from_days[1](days).month[0]) - 1
        else:
            # 1970-01-01 was a Thursday, which is index 3 counting from Monday.
            at = Int((days + 3) % 7)

        var start = offsets[at]
        var stop = offsets[at + 1]
        out.append(packed[byte=start:stop].as_bytes())

    return out^.finish()


def temporal_day_name(a: AnyArray, locale: StringSlice) raises -> StringArray:
    """Writes the name of the day of the week of every row.

    Args:
        a: A date or a naive timestamp column.
        locale: Which language. Only the empty string, meaning English, is
            accepted.

    Returns:
        A text column, null wherever the input is null.

    Raises:
        Error: If the column is not temporal, if it is on a clock whose offset
            is a rule, or if a locale other than English is asked for.
    """
    if not a.type.zone.is_naive():
        return temporal_day_name(local_readings(a), locale)

    _check_nameable(a.type)
    _check_locale(locale)

    var per_day = _units_per_day(a.type)
    var bounds = day_name_bounds()

    if a.type.kind == TypeKind.DATE:
        ref days = a.as_typed_view[DType.int32]()
        return _named_column[DType.int32, False](
            days, per_day, bounds, DAY_NAMES
        )
    ref stamps = a.as_typed_view[DType.int64]()
    return _named_column[DType.int64, False](stamps, per_day, bounds, DAY_NAMES)


def temporal_month_name(a: AnyArray, locale: StringSlice) raises -> StringArray:
    """Writes the name of the month of every row.

    Args:
        a: A date or a naive timestamp column.
        locale: Which language. Only the empty string, meaning English, is
            accepted.

    Returns:
        A text column, null wherever the input is null.

    Raises:
        Error: If the column is not temporal, if it is on a clock whose offset
            is a rule, or if a locale other than English is asked for.
    """
    if not a.type.zone.is_naive():
        return temporal_month_name(local_readings(a), locale)

    _check_nameable(a.type)
    _check_locale(locale)

    var per_day = _units_per_day(a.type)
    var bounds = month_name_bounds()

    if a.type.kind == TypeKind.DATE:
        ref days = a.as_typed_view[DType.int32]()
        return _named_column[DType.int32, True](
            days, per_day, bounds, MONTH_NAMES
        )
    ref stamps = a.as_typed_view[DType.int64]()
    return _named_column[DType.int64, True](
        stamps, per_day, bounds, MONTH_NAMES
    )


def _check_locale(locale: StringSlice) raises:
    """Refuses any locale but the default one.

    pandas takes a locale here and hands it to the C library, so what
    `day_name('de_DE')` answers depends on which locales the machine running it
    has installed. firepanda carries the English names and nothing else, and the
    honest way to say so is to refuse the argument rather than to accept it and
    answer in English anyway, which would be a wrong answer wearing the shape of
    a right one.

    Args:
        locale: What the caller asked for.

    Raises:
        Error: If it is not the empty string.
    """
    if locale.byte_length() != 0:
        raise Error(
            "temporal: firepanda has the English day and month names and no"
            " others, so the locale '"
            + String(locale)
            + "' is refused rather than answered in English"
        )


comptime STEP_SLICE = UInt8(0)
"""A step that copies a run of bytes out of the format string itself."""

comptime STEP_BYTE = UInt8(1)
"""A step that emits one byte held in the step, which is what the compound
directives expand into: `%F` is a year, a hyphen, a month, a hyphen and a day,
and the two hyphens are not anywhere in the format the caller passed."""


@fieldwise_init
struct Step(Copyable, ImplicitlyCopyable, Movable):
    """One thing to do once per row while formatting.

    The format is taken apart into these once, before any row is looked at, so
    that the row loop is a walk over a small list rather than a parse. On a
    column of a million rows that is one parse instead of a million, and it is
    also where an unknown directive is caught: the caller finds out that `%c` is
    refused before the first row is written rather than after the last one.
    """

    var code: UInt8
    """`STEP_SLICE`, `STEP_BYTE`, or the directive letter."""

    var pad: UInt8
    """The byte to emit for `STEP_BYTE`, or the padding flag for a directive,
    or zero for the directive's own default padding."""

    var start: Int
    """Where the slice starts, for `STEP_SLICE`."""

    var stop: Int
    """Where the slice ends, for `STEP_SLICE`."""


def _is_directive(letter: UInt8) -> Bool:
    """Says whether a letter is one of the directives this formatter knows.

    The list is the directives whose answer is the same on every machine.
    Everything else is refused, and what is refused is worth naming: `%c`, `%x`
    and `%X` are whatever the C library's locale says they are, `%s` is the
    epoch second and on a machine measured here it answered the local time
    rather than UTC, and `%P` is a GNU extension that answers the letter P on a
    machine that does not have it. pandas hands the format to the platform and
    inherits all of that. A conformance library that did the same would answer
    differently on two computers and call both of them correct.

    Args:
        letter: The byte after the percent sign.

    Returns:
        True if it is supported.
    """
    return (
        letter == UInt8(ord("Y"))
        or letter == UInt8(ord("y"))
        or letter == UInt8(ord("C"))
        or letter == UInt8(ord("G"))
        or letter == UInt8(ord("g"))
        or letter == UInt8(ord("m"))
        or letter == UInt8(ord("B"))
        or letter == UInt8(ord("b"))
        or letter == UInt8(ord("h"))
        or letter == UInt8(ord("d"))
        or letter == UInt8(ord("e"))
        or letter == UInt8(ord("j"))
        or letter == UInt8(ord("A"))
        or letter == UInt8(ord("a"))
        or letter == UInt8(ord("u"))
        or letter == UInt8(ord("w"))
        or letter == UInt8(ord("V"))
        or letter == UInt8(ord("U"))
        or letter == UInt8(ord("W"))
        or letter == UInt8(ord("H"))
        or letter == UInt8(ord("k"))
        or letter == UInt8(ord("I"))
        or letter == UInt8(ord("l"))
        or letter == UInt8(ord("M"))
        or letter == UInt8(ord("S"))
        or letter == UInt8(ord("f"))
        or letter == UInt8(ord("p"))
        or letter == UInt8(ord("z"))
        or letter == UInt8(ord("Z"))
    )


def _takes_pad_flag(letter: UInt8) -> Bool:
    """Says whether a padding flag in front of a directive changes anything.

    Thirteen directives write a number that has a width the flag can move.
    Everywhere else pandas ignores the flag, which is what this makes it do:
    `%-A` is `%A`, `%-Y` is `%Y` even for the year five, and `%-u` is `%u`
    because a weekday is one digit wide however it is padded. All of that was
    measured rather than assumed, directive by directive.

    Args:
        letter: The directive letter.

    Returns:
        True for the thirteen where the flag has work to do.
    """
    return (
        letter == UInt8(ord("m"))
        or letter == UInt8(ord("d"))
        or letter == UInt8(ord("e"))
        or letter == UInt8(ord("j"))
        or letter == UInt8(ord("V"))
        or letter == UInt8(ord("U"))
        or letter == UInt8(ord("W"))
        or letter == UInt8(ord("H"))
        or letter == UInt8(ord("k"))
        or letter == UInt8(ord("I"))
        or letter == UInt8(ord("l"))
        or letter == UInt8(ord("M"))
        or letter == UInt8(ord("S"))
    )


def _expand(letter: UInt8, mut steps: List[Step]):
    """Pushes the steps a compound directive stands for.

    POSIX defines four of these as abbreviations for other directives, and
    writing them out here rather than in the row loop keeps the row loop with
    one kind of step in it.

    Args:
        letter: `F`, `T`, `R` or `D`.
        steps: Where to push.
    """
    if letter == UInt8(ord("F")):
        steps.append(Step(UInt8(ord("Y")), 0, 0, 0))
        steps.append(Step(STEP_BYTE, UInt8(ord("-")), 0, 0))
        steps.append(Step(UInt8(ord("m")), 0, 0, 0))
        steps.append(Step(STEP_BYTE, UInt8(ord("-")), 0, 0))
        steps.append(Step(UInt8(ord("d")), 0, 0, 0))
    elif letter == UInt8(ord("T")):
        steps.append(Step(UInt8(ord("H")), 0, 0, 0))
        steps.append(Step(STEP_BYTE, UInt8(ord(":")), 0, 0))
        steps.append(Step(UInt8(ord("M")), 0, 0, 0))
        steps.append(Step(STEP_BYTE, UInt8(ord(":")), 0, 0))
        steps.append(Step(UInt8(ord("S")), 0, 0, 0))
    elif letter == UInt8(ord("R")):
        steps.append(Step(UInt8(ord("H")), 0, 0, 0))
        steps.append(Step(STEP_BYTE, UInt8(ord(":")), 0, 0))
        steps.append(Step(UInt8(ord("M")), 0, 0, 0))
    else:
        steps.append(Step(UInt8(ord("m")), 0, 0, 0))
        steps.append(Step(STEP_BYTE, UInt8(ord("/")), 0, 0))
        steps.append(Step(UInt8(ord("d")), 0, 0, 0))
        steps.append(Step(STEP_BYTE, UInt8(ord("/")), 0, 0))
        steps.append(Step(UInt8(ord("y")), 0, 0, 0))


def parse_format(fmt: StringSlice) raises -> List[Step]:
    """Takes a format string apart into the steps that render one row.

    Args:
        fmt: The format.

    Returns:
        The steps, in order.

    Raises:
        Error: If the format ends in a bare percent sign, if a directive is one
            this formatter refuses, or if a padding flag is put in front of
            `%f`.
    """
    var bytes = fmt.as_bytes()
    var n = len(bytes)
    var steps = List[Step]()
    var at = 0
    var run = 0

    while at < n:
        if bytes[at] != UInt8(ord("%")):
            at += 1
            continue

        # Everything since the last directive is one slice step, however long.
        if at > run:
            steps.append(Step(STEP_SLICE, 0, run, at))

        var cursor = at + 1
        if cursor >= n:
            raise Error(
                "temporal: the format '"
                + String(fmt)
                + "' ends in a percent sign with nothing after it"
            )

        var flag = UInt8(0)
        if (
            bytes[cursor] == UInt8(ord("-"))
            or bytes[cursor] == UInt8(ord("_"))
            or bytes[cursor] == UInt8(ord("0"))
        ):
            flag = bytes[cursor]
            cursor += 1
            if cursor >= n:
                raise Error(
                    "temporal: the format '"
                    + String(fmt)
                    + "' ends in a padding flag with no directive after it"
                )

        var letter = bytes[cursor]

        # A flag that cannot move a width is dropped rather than refused,
        # because that is what pandas does with it. `%-A` is `%A` there and it
        # is `%A` here. The exception is `%f`, below, where pandas answers the
        # letter f and loses the microseconds, which is a thing to refuse
        # rather than a thing to copy.
        if flag != 0 and not _takes_pad_flag(letter):
            if letter == UInt8(ord("f")):
                raise Error(
                    "temporal: pandas answers a padding flag on '%f' with the"
                    " letter f and no microseconds at all, so firepanda refuses"
                    " the format rather than writing a column of that"
                )
            flag = 0

        if (
            letter == UInt8(ord("%"))
            or letter == UInt8(ord("n"))
            or letter == UInt8(ord("t"))
        ):
            var literal = letter
            if letter == UInt8(ord("n")):
                literal = 10
            elif letter == UInt8(ord("t")):
                literal = 9
            steps.append(Step(STEP_BYTE, literal, 0, 0))
        elif (
            letter == UInt8(ord("F"))
            or letter == UInt8(ord("T"))
            or letter == UInt8(ord("R"))
            or letter == UInt8(ord("D"))
        ):
            _expand(letter, steps)
        elif _is_directive(letter):
            steps.append(Step(letter, flag, 0, 0))
        else:
            raise Error(
                "temporal: '%"
                + chr(Int(letter))
                + "' is not a directive firepanda knows; pandas passes the"
                " format to the C library and the ones left out here are the"
                " ones whose answer depends on the machine or on its locale"
            )

        at = cursor + 1
        run = at

    if n > run:
        steps.append(Step(STEP_SLICE, 0, run, n))
    return steps^


def _append_digits(mut out: List[UInt8], value: Int64, width: Int, pad: UInt8):
    """Appends one number, padded on the left to a minimum width.

    Args:
        out: Where to append.
        value: The number. A negative one gets its sign in front of the padding,
            which is where a year before the year one wants it.
        width: The minimum number of digits, before the sign.
        pad: The byte to pad with. Zero means no padding at all, which is what
            a `-` flag asks for.
    """
    var digits = List[UInt8](capacity=20)
    var rest = value if value >= 0 else -value

    if rest == 0:
        digits.append(UInt8(48))
    while rest > 0:
        digits.append(UInt8(48 + Int(rest % 10)))
        rest //= 10

    if value < 0:
        out.append(UInt8(45))
    if pad != 0:
        for _ in range(width - len(digits)):
            out.append(pad)

    for i in range(len(digits)):
        out.append(digits[len(digits) - 1 - i])


def _pad_for(flag: UInt8, default: UInt8) -> UInt8:
    """Turns a padding flag into the byte to pad with.

    Args:
        flag: What the format asked for, or zero for the directive's default.
        default: The directive's default.

    Returns:
        The byte, or zero for no padding.
    """
    if flag == 0:
        return default
    if flag == 45:
        return 0
    if flag == 95:
        return 32
    return 48


def _render_row(
    mut out: List[UInt8],
    steps: List[Step],
    fmt: StringSlice,
    value: Int64,
    per_day: Int64,
    per_second: Int64,
) raises:
    """Renders one row into a byte buffer.

    Both calendar conversions are done whether or not the format asks for them.
    One is the day this row falls in and the other is the Thursday of its ISO
    week, and doing them unconditionally costs two dozen integer instructions on
    a path that is about to write digits one at a time into a buffer. The
    formatting is the expensive part here, not the calendar.

    Args:
        out: The buffer, already cleared.
        steps: The parsed format.
        fmt: The format string the slice steps point into.
        value: The row, in whole units since the epoch.
        per_day: How many of the column's units make a day.
        per_second: How many of them make a second, or 1 for a date column.

    Raises:
        Error: Never, in practice. `parse_format` has already refused every
            directive this cannot render, so the trailing case below is
            unreachable and is here so that a directive added to one list and
            not the other fails loudly instead of writing nothing.
    """
    var days = value // per_day
    var rest = value - days * per_day
    var civil = civil_from_days[1](days)
    var weekday = Int((days + 3) % 7)
    var thursday = civil_from_days[1](days - Int64(weekday) + 3)

    var year = civil.year[0]
    var month = Int(civil.month[0])
    var day_of_year = civil.day_of_year[0]
    # A date column answers zero per second, because it has no clock in it, and
    # its remainder above is zero too. One rather than zero here is what stops
    # `%H` on a date column from dividing by zero on its way to the answer it
    # was always going to give.
    var rate = max(per_second, 1)
    var hour = rest // (rate * 3600)
    var minute = (rest // (rate * 60)) % 60
    var second = (rest // rate) % 60
    var micro = (rest % rate) * 1_000_000 // rate

    var days_bounds = day_name_bounds()
    var months_bounds = month_name_bounds()

    for i in range(len(steps)):
        ref step = steps[i]
        var code = step.code

        if code == STEP_SLICE:
            var bytes = fmt.as_bytes()
            for at in range(step.start, step.stop):
                out.append(bytes[at])
        elif code == STEP_BYTE:
            out.append(step.pad)
        elif code == UInt8(ord("Y")):
            _append_digits(out, year, 4, _pad_for(step.pad, 48))
        elif code == UInt8(ord("y")):
            _append_digits(out, year % 100, 2, _pad_for(step.pad, 48))
        elif code == UInt8(ord("C")):
            _append_digits(out, year // 100, 2, _pad_for(step.pad, 48))
        elif code == UInt8(ord("G")):
            _append_digits(out, thursday.year[0], 4, _pad_for(step.pad, 48))
        elif code == UInt8(ord("g")):
            _append_digits(
                out, thursday.year[0] % 100, 2, _pad_for(step.pad, 48)
            )
        elif code == UInt8(ord("m")):
            _append_digits(out, Int64(month), 2, _pad_for(step.pad, 48))
        elif code == UInt8(ord("d")):
            _append_digits(out, civil.day[0], 2, _pad_for(step.pad, 48))
        elif code == UInt8(ord("e")):
            _append_digits(out, civil.day[0], 2, _pad_for(step.pad, 32))
        elif code == UInt8(ord("j")):
            _append_digits(out, day_of_year, 3, _pad_for(step.pad, 48))
        elif code == UInt8(ord("H")):
            _append_digits(out, hour, 2, _pad_for(step.pad, 48))
        elif code == UInt8(ord("k")):
            _append_digits(out, hour, 2, _pad_for(step.pad, 32))
        elif code == UInt8(ord("I")):
            var half = hour % 12
            _append_digits(
                out, 12 if half == 0 else half, 2, _pad_for(step.pad, 48)
            )
        elif code == UInt8(ord("l")):
            var half = hour % 12
            _append_digits(
                out, 12 if half == 0 else half, 2, _pad_for(step.pad, 32)
            )
        elif code == UInt8(ord("M")):
            _append_digits(out, minute, 2, _pad_for(step.pad, 48))
        elif code == UInt8(ord("S")):
            _append_digits(out, second, 2, _pad_for(step.pad, 48))
        elif code == UInt8(ord("f")):
            # Six digits whatever the column's resolution is, which is what
            # pandas writes: a second column answers 000000 and a nanosecond
            # column drops its last three digits rather than writing nine.
            _append_digits(out, micro, 6, _pad_for(step.pad, 48))
        elif code == UInt8(ord("u")):
            _append_digits(out, Int64(weekday + 1), 1, _pad_for(step.pad, 48))
        elif code == UInt8(ord("w")):
            # C numbers the week from Sunday here and ISO numbers it from
            # Monday, and this is the one line where the two disagree.
            _append_digits(
                out, Int64((weekday + 1) % 7), 1, _pad_for(step.pad, 48)
            )
        elif code == UInt8(ord("V")):
            _append_digits(
                out,
                (thursday.day_of_year[0] - 1) // 7 + 1,
                2,
                _pad_for(step.pad, 48),
            )
        elif code == UInt8(ord("U")):
            _append_digits(
                out,
                (day_of_year + 6 - Int64((weekday + 1) % 7)) // 7,
                2,
                _pad_for(step.pad, 48),
            )
        elif code == UInt8(ord("W")):
            _append_digits(
                out,
                (day_of_year + 6 - Int64(weekday)) // 7,
                2,
                _pad_for(step.pad, 48),
            )
        elif code == UInt8(ord("p")):
            out.append(UInt8(65) if hour < 12 else UInt8(80))
            out.append(UInt8(77))
        elif code == UInt8(ord("A")):
            var bytes = DAY_NAMES.as_bytes()
            for at in range(days_bounds[weekday], days_bounds[weekday + 1]):
                out.append(bytes[at])
        elif code == UInt8(ord("a")):
            var bytes = DAY_ABBREVS.as_bytes()
            for at in range(weekday * 3, weekday * 3 + 3):
                out.append(bytes[at])
        elif code == UInt8(ord("B")):
            var bytes = MONTH_NAMES.as_bytes()
            for at in range(months_bounds[month - 1], months_bounds[month]):
                out.append(bytes[at])
        elif code == UInt8(ord("b")) or code == UInt8(ord("h")):
            var bytes = MONTH_ABBREVS.as_bytes()
            for at in range((month - 1) * 3, (month - 1) * 3 + 3):
                out.append(bytes[at])
        elif code == UInt8(ord("z")) or code == UInt8(ord("Z")):
            # A naive column has no offset and no zone name, and pandas writes
            # nothing for both. A zoned one never reaches here.
            pass
        else:
            raise Error(
                "temporal: '%"
                + chr(Int(code))
                + "' passed the format parser and has no renderer, which is a"
                " bug in firepanda rather than in the format"
            )


def _formatted_column[
    src: DType
](
    a: Array[src],
    steps: List[Step],
    fmt: StringSlice,
    per_day: Int64,
    per_second: Int64,
) raises -> StringArray:
    """Renders every row of a column through an already parsed format.

    Args:
        a: The column.
        steps: The parsed format.
        fmt: The format string the slice steps point into.
        per_day: How many of the column's units make a day.
        per_second: How many of them make a second.

    Parameters:
        src: The physical dtype of the column.

    Returns:
        A text column, null wherever the input is null.

    Raises:
        Error: Only what the renderer raises.
    """
    var n = len(a)
    var out = StringBuilder(capacity=n)
    var row = List[UInt8](capacity=64)

    for i in range(n):
        if not a.is_valid(i):
            out.append_null()
            continue
        row.clear()
        _render_row(row, steps, fmt, Int64(a[i]), per_day, per_second)
        out.append(Span(row))

    return out^.finish()


def temporal_strftime(a: AnyArray, fmt: StringSlice) raises -> StringArray:
    """Writes every row of a temporal column through a format string.

    The format is parsed once, here, and not once per row. That is worth saying
    out loud because it is also where the error comes from: a format with a
    directive this refuses fails before the first row is looked at, so the
    caller gets the message rather than a column that is wrong in a million
    places.

    Args:
        a: A date or a naive timestamp column.
        fmt: The format.

    Returns:
        A text column, null wherever the input is null.

    Raises:
        Error: If the column is not temporal, if it is on a clock whose offset
            is a rule, or if the format has something in it this cannot render.
    """
    if not a.type.zone.is_naive():
        return temporal_strftime(local_readings(a), fmt)

    _check_nameable(a.type)

    var steps = parse_format(fmt)
    var per_day = _units_per_day(a.type)
    var per_second = _units_per_second(a.type)

    if a.type.kind == TypeKind.DATE:
        ref days = a.as_typed_view[DType.int32]()
        return _formatted_column[DType.int32](
            days, steps, fmt, per_day, per_second
        )
    ref stamps = a.as_typed_view[DType.int64]()
    return _formatted_column[DType.int64](
        stamps, steps, fmt, per_day, per_second
    )


def _padded(value: Int64, width: Int) -> String:
    """Writes a non negative number with at least a given number of digits.

    Args:
        value: The number.
        width: The least number of digits.

    Returns:
        The digits, left padded with zeros.
    """
    var text = String(value)
    var out = String()
    for _ in range(width - text.byte_length()):
        out += "0"
    return out + text


def temporal_text(type: LogicalType, raw: Int64) raises -> String:
    """Renders one instant the way a file and a printed table should carry it.

    A date is stored as a day count and a timestamp as a tick count, so anything
    that renders a column by reading its physical layout writes the count. The
    bytes are right and the reader gets an integer where the schema promised a
    date, which is the same wrongness `AnyArray.retyped` was written for, one
    layer further out: the type is known and thrown away at the last step.

    The spelling is ISO 8601, which is what pandas prints, what Polars prints
    and what every CSV reader including this library's own parses back to the
    type it started as.

    Args:
        type: The column's logical type, which must be a date or a naive
            timestamp.
        raw: The stored value.

    Returns:
        `YYYY-MM-DD` for a date and `YYYY-MM-DD HH:MM:SS` for a timestamp, with
        a fractional part only when the value has one.

    Raises:
        Error: If the type is not one whose values are instants, or if it is a
            timestamp carrying a time zone, since rendering that from the stored
            instant would print UTC under the name of a local hour.
    """
    var per_day = _units_per_day(type)
    var per_second = _units_per_second(type)

    # Floor division, so a tick before the epoch belongs to the day it falls in
    # rather than to the one after it.
    var days = raw // per_day
    var civil = civil_from_days[1](days)

    var year = civil.year[0]
    var out = String()
    if year < 0:
        out += "-"
        year = -year
    out += _padded(year, 4)
    out += "-" + _padded(civil.month[0], 2)
    out += "-" + _padded(civil.day[0], 2)
    if per_second == 0:
        return out^

    var within = raw - days * per_day
    var seconds = within // per_second
    out += " " + _padded(seconds // 3600, 2)
    out += ":" + _padded((seconds // 60) % 60, 2)
    out += ":" + _padded(seconds % 60, 2)

    var fraction = within - seconds * per_second
    if fraction == 0:
        return out^
    var digits = 0
    var scale = per_second
    while scale > 1:
        scale //= 10
        digits += 1
    return out + "." + _padded(fraction, digits)


def instant_text(col: AnyArray, i: Int) -> Optional[String]:
    """Renders one row of a column when that column holds instants.

    This is what a renderer calls before it dispatches on the physical layout,
    and it answers nothing for every column that is not a date or a naive
    timestamp, which is the signal to carry on down the usual path. A timestamp
    carrying a time zone answers nothing too: the stored instants are UTC and
    printing an hour off them would be a wrong number under the right name, so
    the count is the more honest thing to show until there is a zone database
    to convert with.

    Args:
        col: The column.
        i: The row, which the caller has already found to be present.

    Returns:
        The text, or nothing when this column is not one this renders.
    """
    if col.type.kind == TypeKind.DATE:
        try:
            return temporal_text(
                col.type,
                Int64(
                    col.unsafe_ptr[DType.int32]().unsafe_offset(i).unsafe_load()
                ),
            )
        except:
            return None
    if col.type.kind == TypeKind.TIMESTAMP and col.type.zone.is_naive():
        try:
            return temporal_text(
                col.type,
                col.unsafe_ptr[DType.int64]().unsafe_offset(i).unsafe_load(),
            )
        except:
            return None
    return None
