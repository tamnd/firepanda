"""The calendar and clock fields of a temporal column.

A timestamp column is a column of integers and a count of them per second. Every
name on the pandas `dt` accessor that answers a number is a function of those two
things and of nothing else, so the whole of `dt.year`, `dt.month`, `dt.day`,
`dt.dayofweek`, `dt.quarter` and the six `is_` predicates is one calendar
conversion written once and read nineteen ways.

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

What is not here is the time zone. Every field below reads the integers as they
are stored, which is the wall clock reading for a naive column and UTC for a
zoned one, and the wall clock reading in the column's own zone is a different
number that needs a zone database this library does not have yet. A zoned column
is refused rather than answered in UTC, because an hour that is silently seven
off is worse than an hour that is missing.
"""

from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.logical import LogicalType, TypeKind
from firepanda.exec import parallel_morsels

from .mask import repair_range

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
at or above this one answers bool and every code below it answers int32."""

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
"""Field code for `dt.is_year_end`."""

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
]
"""Every field code, in order, for the dispatch to walk at compile time."""


def field_dtype(field: Int) -> DType:
    """Returns the dtype a field answers with.

    Args:
        field: The field code.

    Returns:
        `DType.bool` for the predicates and `DType.int32` for the rest, which is
        what pandas answers and is compared exactly.
    """
    return DType.bool if field >= FIELD_IS_LEAP_YEAR else DType.int32


@fieldwise_init
struct TemporalField(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which part of a timestamp is being asked for.

    Held as a code rather than as a function for the same reason `UnaryOp` is:
    the erased entry point takes it as an ordinary argument and the typed loop
    takes it as a parameter the compiler folds away.
    """

    var code: Int
    """The field, as one of the nineteen values below."""

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
]
"""The pandas name of each field, in code order, for error messages."""


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
    if not t.zone.is_naive():
        # Reading these off the stored integers would answer UTC under the name
        # of a local hour, which is a wrong number rather than a missing one.
        raise Error(
            "temporal: the calendar fields of a column in "
            + String(t.zone)
            + " need a time zone database firepanda does not have yet, and"
            " answering them from the stored instants would give UTC"
        )
    return t.unit.per_second() * SECONDS_PER_DAY


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
    two chances to spell `days_in_month` differently.

    Args:
        name: The pandas spelling, so `dayofweek` rather than `day_of_week`.

    Returns:
        The field.

    Raises:
        Error: If nothing is called that.
    """
    comptime for code in FIELD_CODES:
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
        An int32 column for the twelve fields that are numbers and a bool column
        for the seven that are predicates, null wherever the input is null.

    Raises:
        Error: If the column is not temporal, if it carries a time zone, or if
            the field code is not one of the nineteen.
    """
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
        Error: If the column is not temporal or carries a time zone.
    """
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
        Error: If the column is not temporal or carries a time zone.
    """
    var per_day = _units_per_day(a.type)
    if a.type.kind == TypeKind.DATE:
        return AnyArray(copy=a)
    return AnyArray(
        _midnights_of(a.as_typed_view[DType.int64](), per_day).into_data(),
        a.type,
    )
