"""Tests for the day and month names, the ISO calendar and the format string.

Every expected value in this file was read off a running pandas 3.0.3 rather
than worked out here, including the ones a reader can check on a wall calendar,
because the point of the file is agreement with pandas and not agreement with
arithmetic.

Three things in here are worth knowing before reading the tables.

The first is that the ISO calendar year is not the calendar year. A week belongs
to the year its Thursday falls in, so the last day of 1969 is in ISO year 1970
and the first day of 2021 is in ISO year 2020. Both of those rows are here, and
they are the two rows that catch an implementation that reads the ISO year off
the calendar year.

The second is that `%U` and `%W` are two different week numbers and neither of
them is the ISO one. `%U` starts its weeks on Sunday and `%W` starts them on
Monday, so 4 January 1970, which was a Sunday, is week 1 by the first and week 0
by the second. That row is here for that reason alone.

The third is that a padding flag does something in front of thirteen directives
and nothing in front of the rest, which is what pandas does with it. `%-A` is
`%A`, and `%-Y` is `%Y` even for a year below a thousand, where the year is
padded to four digits whatever the flag says. The one place pandas does not
ignore the flag is `%f`, where it answers the letter f and drops the
microseconds, and that is the one format this refuses.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.frame.frame import dt_isocalendar
from firepanda.frame.series import Series
from firepanda.kernel.temporal import (
    TemporalField,
    temporal_day_name,
    temporal_field,
    temporal_month_name,
    temporal_strftime,
)

comptime NULL_ROW = Int64.MIN
"""The marker `stamps` reads as a missing row rather than as an instant, chosen
for the same reason as in `test_temporal_round.mojo`: -1 is a real row this file
cares about."""


def calendar_rows() -> List[Int64]:
    """Returns the seconds every table in this file is indexed by.

    Twelve instants, each one chosen for something it alone shows. In order:
    the smallest date pandas can hold in nanoseconds, the start of the twentieth
    century, the second before the epoch, the epoch, the first Saturday and the
    first Sunday after it, a Sunday that is in the previous ISO year, a leap
    day, the Thursday that puts 2020 in a fifty three week year, the Friday
    after it that is still in ISO 2020, the largest date pandas can hold in
    nanoseconds, and the start of the twenty fourth century.

    Returns:
        Twelve counts of seconds since the epoch.
    """
    return [
        Int64(-9223286400),
        -2208988800,
        -1,
        0,
        172800,
        259200,
        946771200,
        1582934400,
        1609372800,
        1609459200,
        9223286400,
        10413792000,
    ]


def day_names() -> List[String]:
    """Returns what pandas answers for `dt.day_name()` on `calendar_rows`.

    Returns:
        One name per row.
    """
    return [
        String("Wednesday"),
        "Monday",
        "Wednesday",
        "Thursday",
        "Saturday",
        "Sunday",
        "Sunday",
        "Saturday",
        "Thursday",
        "Friday",
        "Friday",
        "Monday",
    ]


def month_names() -> List[String]:
    """Returns what pandas answers for `dt.month_name()` on `calendar_rows`.

    Returns:
        One name per row.
    """
    return [
        String("September"),
        "January",
        "December",
        "January",
        "January",
        "January",
        "January",
        "February",
        "December",
        "January",
        "April",
        "January",
    ]


def iso_years() -> List[Int64]:
    """Returns the `year` column of `dt.isocalendar()` on `calendar_rows`.

    Returns:
        One year per row.
    """
    return [
        Int64(1677),
        1900,
        1970,
        1970,
        1970,
        1970,
        1999,
        2020,
        2020,
        2020,
        2262,
        2300,
    ]


def iso_weeks() -> List[Int64]:
    """Returns the `week` column of `dt.isocalendar()` on `calendar_rows`.

    Returns:
        One week per row.
    """
    return [Int64(38), 1, 1, 1, 1, 1, 52, 9, 53, 53, 15, 1]


def iso_days() -> List[Int64]:
    """Returns the `day` column of `dt.isocalendar()` on `calendar_rows`.

    Returns:
        One weekday per row, Monday being one.
    """
    return [Int64(3), 1, 3, 4, 6, 7, 7, 6, 4, 5, 5, 1]


def stamps(values: List[Int64], unit: TimeUnit) raises -> AnyArray:
    """Builds a naive timestamp column, with `NULL_ROW` meaning a null.

    Args:
        values: The instants, in whole units since the epoch.
        unit: The resolution.

    Returns:
        The column.
    """
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        if values[i] == NULL_ROW:
            col.set_null(i)
        else:
            col.set_valid(i, values[i])
    return AnyArray(col^.into_data(), LogicalType.timestamp(unit))


def dates(values: List[Int64]) raises -> AnyArray:
    """Builds a date column from counts of days, with `NULL_ROW` for a null.

    Args:
        values: The days since the epoch.

    Returns:
        The column.
    """
    var col = Array[DType.int32](len(values))
    for i in range(len(values)):
        if values[i] == NULL_ROW:
            col.set_null(i)
        else:
            col.set_valid(i, Int32(values[i]))
    return AnyArray(col^.into_data(), LogicalType.DATE32)


def test_the_day_of_the_week_is_named_as_pandas_names_it() raises:
    """Twelve instants against twelve names, every one of them from pandas."""
    var got = temporal_day_name(stamps(calendar_rows(), TimeUnit.SECOND), "")
    var want = day_names()
    for i in range(len(want)):
        assert_equal(got[i], want[i], String("row ", i))


def test_the_month_is_named_as_pandas_names_it() raises:
    """The same twelve instants against the month names."""
    var got = temporal_month_name(stamps(calendar_rows(), TimeUnit.SECOND), "")
    var want = month_names()
    for i in range(len(want)):
        assert_equal(got[i], want[i], String("row ", i))


def test_a_name_reads_the_same_at_every_resolution() raises:
    """The instant decides the name, not the unit it is counted in."""
    var rows = calendar_rows()
    var milli = List[Int64](capacity=len(rows))
    for i in range(len(rows)):
        milli.append(rows[i] * 1000)
    var got = temporal_day_name(stamps(milli, TimeUnit.MILLI), "")
    var want = day_names()
    for i in range(len(want)):
        assert_equal(got[i], want[i], String("row ", i))


def test_a_date_column_is_named_too() raises:
    """A date has a day of the week the same way a timestamp does."""
    var got = temporal_day_name(dates([Int64(0), 1, 2, -1]), "")
    assert_equal(got[0], "Thursday", "the epoch")
    assert_equal(got[1], "Friday", "the day after")
    assert_equal(got[2], "Saturday", "the day after that")
    assert_equal(got[3], "Wednesday", "the day before")


def test_a_missing_row_has_no_name() raises:
    """A missing row has no name: pandas answers NaN here and this answers a null, which is the same claim
    written in the type system rather than in a float."""
    var col = stamps([Int64(0), NULL_ROW, 172800], TimeUnit.SECOND)
    var got = temporal_day_name(col, "")
    assert_true(got.is_valid(0), "row 0 is there")
    assert_true(not got.is_valid(1), "row 1 is missing")
    assert_equal(got[2], "Saturday", "row 2")


def test_a_locale_other_than_english_is_refused() raises:
    """Only English: pandas hands the locale to the C library, so its answer for a French one
    depends on the machine. Refusing is the only answer that does not."""
    var col = stamps([Int64(0)], TimeUnit.SECOND)
    with assert_raises(contains="English day and month names"):
        _ = temporal_day_name(col, "fr_FR")
    with assert_raises(contains="'de_DE'"):
        _ = temporal_month_name(col, "de_DE")


def test_a_column_with_no_calendar_in_it_has_no_names() raises:
    """A number is not a date and a zoned column is not a local reading."""
    var plain = Array[DType.int64](1)
    plain.set_valid(0, 0)
    with assert_raises(contains="comes off a date or a timestamp"):
        _ = temporal_day_name(AnyArray(plain^), "")

    var zoned = Array[DType.int64](1)
    zoned.set_valid(0, 0)
    var col = AnyArray(
        zoned^.into_data(),
        LogicalType.timestamp(TimeUnit.SECOND, TimeZone("Asia/Tokyo")),
    )
    with assert_raises(contains="time zone database"):
        _ = temporal_month_name(col, "")


def test_the_iso_year_is_not_the_calendar_year() raises:
    """The two rows either side of a year boundary that move, and the ten that
    do not."""
    var col = stamps(calendar_rows(), TimeUnit.SECOND)
    var years = temporal_field(col, TemporalField.ISO_YEAR)
    var weeks = temporal_field(col, TemporalField.ISO_WEEK)
    var days = temporal_field(col, TemporalField.ISO_DAY)
    ref y = years.as_typed_view[DType.uint32]()
    ref w = weeks.as_typed_view[DType.uint32]()
    ref d = days.as_typed_view[DType.uint32]()

    var want_years = iso_years()
    var want_weeks = iso_weeks()
    var want_days = iso_days()
    for i in range(len(want_years)):
        assert_equal(Int64(y[i]), want_years[i], String("year of row ", i))
        assert_equal(Int64(w[i]), want_weeks[i], String("week of row ", i))
        assert_equal(Int64(d[i]), want_days[i], String("day of row ", i))


def test_the_three_iso_fields_are_unsigned_and_the_others_are_not() raises:
    """The ISO fields are unsigned: pandas gives that frame three uint32 columns and gives every other
    field on the accessor an int32 one."""
    var col = stamps([Int64(0)], TimeUnit.SECOND)
    assert_equal(
        String(temporal_field(col, TemporalField.ISO_YEAR).type),
        "uint32",
        "the ISO year",
    )
    assert_equal(
        String(temporal_field(col, TemporalField.YEAR).type),
        "int32",
        "the calendar year",
    )


def test_a_missing_row_has_no_iso_calendar() raises:
    """A null goes through the ISO fields the way it goes through the others."""
    var col = stamps([Int64(0), NULL_ROW], TimeUnit.SECOND)
    var got = temporal_field(col, TemporalField.ISO_WEEK)
    assert_true(got.is_valid(0), "row 0 is there")
    assert_true(not got.is_valid(1), "row 1 is missing")


def test_the_iso_frame_has_three_named_columns_and_the_row_labels() raises:
    """`dt.isocalendar()` is the one member of the accessor that answers a frame
    rather than a column, and the names on it are pandas' names."""
    var s = Series("when", stamps([Int64(-1), 0], TimeUnit.SECOND))
    var got = dt_isocalendar(s)
    assert_equal(len(got), 2, "the height")
    assert_equal(got.schema.fields[0].name, "year", "the first name")
    assert_equal(got.schema.fields[1].name, "week", "the second name")
    assert_equal(got.schema.fields[2].name, "day", "the third name")
    assert_equal(
        got.column("year").values.as_typed_view[DType.uint32]()[0],
        1970,
        "the ISO year of the second before the epoch",
    )
    assert_equal(
        got.column("day").values.as_typed_view[DType.uint32]()[1],
        4,
        "the epoch was a Thursday",
    )


def test_a_format_writes_what_pandas_writes() raises:
    """The plainest format there is, over the twelve rows."""
    var col = stamps(calendar_rows(), TimeUnit.SECOND)
    var got = temporal_strftime(col, "%Y-%m-%d %H:%M:%S")
    assert_equal(got[0], "1677-09-22 00:00:00", "row 0")
    assert_equal(got[2], "1969-12-31 23:59:59", "the second before the epoch")
    assert_equal(got[3], "1970-01-01 00:00:00", "the epoch")
    assert_equal(got[7], "2020-02-29 00:00:00", "the leap day")
    assert_equal(got[10], "2262-04-11 00:00:00", "row 10")


def test_the_named_directives_write_the_names() raises:
    """The abbreviations, the twelve hour clock and the day of the year."""
    var col = stamps(calendar_rows(), TimeUnit.SECOND)
    var got = temporal_strftime(col, "%a %b %e %I%p %j")
    assert_equal(got[0], "Wed Sep 22 12AM 265", "row 0")
    assert_equal(got[1], "Mon Jan  1 12AM 001", "the space padded day")
    assert_equal(got[2], "Wed Dec 31 11PM 365", "the hour before the epoch")
    assert_equal(got[8], "Thu Dec 31 12AM 366", "the last day of a leap year")


def test_the_three_week_numbers_disagree_and_all_three_are_right() raises:
    """`%U` counts from Sunday, `%W` counts from Monday and `%V` is the ISO
    week, which is why 4 January 1970 is week 1, week 0 and week 1."""
    var col = stamps(calendar_rows(), TimeUnit.SECOND)
    var got = temporal_strftime(col, "%U %W %V")
    assert_equal(got[1], "00 01 01", "1 January 1900, a Monday")
    assert_equal(got[3], "00 00 01", "the epoch, a Thursday")
    assert_equal(got[5], "01 00 01", "4 January 1970, a Sunday")
    assert_equal(got[9], "00 00 53", "1 January 2021, still in ISO 2020")


def test_the_iso_directives_agree_with_the_iso_fields() raises:
    """`%G`, `%V` and `%u` are the three columns of the ISO frame written out,
    and they are computed by different code here, so this is two answers to the
    same question rather than one answer twice."""
    var col = stamps(calendar_rows(), TimeUnit.SECOND)
    var got = temporal_strftime(col, "%G|%V|%u")
    var years = temporal_field(col, TemporalField.ISO_YEAR)
    var weeks = temporal_field(col, TemporalField.ISO_WEEK)
    var days = temporal_field(col, TemporalField.ISO_DAY)
    ref y = years.as_typed_view[DType.uint32]()
    ref w = weeks.as_typed_view[DType.uint32]()
    ref d = days.as_typed_view[DType.uint32]()

    for i in range(len(calendar_rows())):
        var want = String(Int(y[i]))
        want += "|"
        if w[i] < 10:
            want += "0"
        want += String(Int(w[i]))
        want += "|"
        want += String(Int(d[i]))
        assert_equal(got[i], want, String("row ", i))


def test_a_year_below_a_thousand_is_padded_to_four_digits() raises:
    """Measured, and not what a reader of the C library manual would expect: the
    year five is written 0005 and a padding flag does not change it."""
    var col = dates([Int64(-718000)])
    assert_equal(temporal_strftime(col, "%Y")[0], "0004", "the year")
    assert_equal(temporal_strftime(col, "%-Y")[0], "0004", "the flag is idle")
    assert_equal(temporal_strftime(col, "%C")[0], "00", "the century")
    assert_equal(temporal_strftime(col, "%y")[0], "04", "the two digit year")


def test_the_padding_flags_move_the_thirteen_widths_they_can() raises:
    """A flag in front of a width does what it says and a flag in front of
    anything else is ignored, which is what pandas does with it."""
    var col = stamps([Int64(3723)], TimeUnit.SECOND)
    assert_equal(temporal_strftime(col, "%m")[0], "01", "the default")
    assert_equal(temporal_strftime(col, "%-m")[0], "1", "no padding")
    assert_equal(temporal_strftime(col, "%_m")[0], " 1", "a space")
    assert_equal(temporal_strftime(col, "%0e")[0], "01", "a zero")
    assert_equal(temporal_strftime(col, "%e")[0], " 1", "the space by default")
    assert_equal(temporal_strftime(col, "%-A")[0], "Thursday", "no width here")
    assert_equal(temporal_strftime(col, "%-u")[0], "4", "one digit either way")


def test_the_compound_directives_stand_for_the_others() raises:
    """Four POSIX abbreviations, each of them written out at parse time rather
    than in the row loop."""
    var col = stamps([Int64(3723)], TimeUnit.SECOND)
    assert_equal(temporal_strftime(col, "%F")[0], "1970-01-01", "the date")
    assert_equal(temporal_strftime(col, "%T")[0], "01:02:03", "the clock")
    assert_equal(temporal_strftime(col, "%R")[0], "01:02", "the short clock")
    assert_equal(temporal_strftime(col, "%D")[0], "01/01/70", "the American")


def test_the_literals_come_through() raises:
    """A percent sign, a newline, a tab and text either side of a directive."""
    var col = stamps([Int64(0)], TimeUnit.SECOND)
    assert_equal(temporal_strftime(col, "100%%")[0], "100%", "the percent")
    assert_equal(temporal_strftime(col, "a%nb")[0], "a\nb", "the newline")
    assert_equal(temporal_strftime(col, "a%tb")[0], "a\tb", "the tab")
    assert_equal(
        temporal_strftime(col, "year %Y end")[0], "year 1970 end", "the text"
    )
    assert_equal(temporal_strftime(col, "")[0], "", "nothing at all")


def test_the_sub_second_digits_are_always_six() raises:
    """`%f` is six digits whatever the column's resolution is, so a millisecond
    column pads and a nanosecond column loses its last three digits."""
    assert_equal(
        temporal_strftime(stamps([Int64(1500)], TimeUnit.MILLI), "%f")[0],
        "500000",
        "half a second in milliseconds",
    )
    assert_equal(
        temporal_strftime(stamps([Int64(1_123_456_789)], TimeUnit.NANO), "%f")[
            0
        ],
        "123456",
        "a nanosecond column truncates",
    )
    assert_equal(
        temporal_strftime(stamps([Int64(7)], TimeUnit.SECOND), "%f")[0],
        "000000",
        "a second column has none",
    )


def test_a_date_column_formats_its_clock_as_midnight() raises:
    """A date has no clock in it and the clock directives still work, because
    pandas answers them from a timestamp at midnight."""
    var got = temporal_strftime(dates([Int64(1)]), "%F %T %p")
    assert_equal(got[0], "1970-01-02 00:00:00 AM", "the day after the epoch")


def test_a_missing_row_is_not_formatted() raises:
    """The format never sees the row, so a null in cannot become text out."""
    var col = stamps([Int64(0), NULL_ROW], TimeUnit.SECOND)
    var got = temporal_strftime(col, "%Y")
    assert_true(got.is_valid(0), "row 0 is there")
    assert_true(not got.is_valid(1), "row 1 is missing")


def test_a_directive_that_depends_on_the_machine_is_refused() raises:
    """`%c`, `%x`, `%X` and `%s` are whatever the C library says they are, and
    on the machine this was written on `%s` answered the local time rather than
    the epoch second. pandas inherits all of that and this does not."""
    var col = stamps([Int64(0)], TimeUnit.SECOND)
    with assert_raises(contains="'%c' is not a directive"):
        _ = temporal_strftime(col, "%c")
    with assert_raises(contains="'%s' is not a directive"):
        _ = temporal_strftime(col, "%s")
    with assert_raises(contains="depends on the machine"):
        _ = temporal_strftime(col, "%X")


def test_a_format_that_cannot_be_parsed_is_refused_before_any_row() raises:
    """The parse happens once, so a bad format is one error rather than a column
    of them."""
    var col = stamps([Int64(0), 1], TimeUnit.SECOND)
    with assert_raises(contains="ends in a percent sign"):
        _ = temporal_strftime(col, "%Y-%m-%d %")
    with assert_raises(contains="ends in a padding flag"):
        _ = temporal_strftime(col, "%Y %-")
    with assert_raises(contains="the letter f"):
        _ = temporal_strftime(col, "%-f")


def test_a_series_names_and_formats_and_keeps_its_labels() raises:
    """The three of these that answer a column reach the frame layer."""
    var s = Series("when", stamps([Int64(0), 172800], TimeUnit.SECOND))
    assert_equal(s.dt_day_name().name, "when", "the name survives")
    assert_equal(len(s.dt_month_name()), 2, "the height survives")
    assert_equal(
        String(s.dt_strftime("%Y").values.type), "string", "the type is text"
    )
    ref names = s.dt_day_name().values
    assert_equal(names.strings()[0], "Thursday", "row 0")
    assert_equal(names.strings()[1], "Saturday", "row 1")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
