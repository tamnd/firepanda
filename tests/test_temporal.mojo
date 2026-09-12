"""Tests for the calendar and clock fields of a temporal column.

Every expected value in this file was read off a running pandas 3.0.3 rather than
worked out here, and the whole kernel was checked against pandas a second way
before any of it was written down: five hundred random timestamps between the
year 1 and the year 2262, at all four resolutions, put through all nineteen
named fields plus `dt.date` and `dt.normalize`, and the two outputs diffed. They were
identical. What is in this file is the subset of that a reader can check by eye
and that names why each row is here.

The rows were not picked for coverage of the arithmetic, which the random run
already has. They were picked for the places where a plausible implementation is
wrong:

The last second of 1969 is here because it is the one row that tells a division
that rounds down apart from a division that rounds towards zero, and getting that
wrong moves the date forward by a day and the year forward by one.

The 29th of February is here twice, in 2024 and in 2000, because the leap rule
has three clauses and a year divisible by four hundred exercises the one that
almost never runs. 1900 is here for the clause in the middle.

The 31st of December, the 31st of March and the 30th of June are here because
`is_month_end`, `is_quarter_end` and `is_year_end` are the three predicates that
need the length of the month rather than just the day number, and a month length
table with February wrong passes every test that does not land on one.

Timestamps at four resolutions carrying the same instant are here because pandas
2 put the resolution in the dtype, and an implementation that normalises
everything to nanoseconds answers all of these correctly and then loses every row
before 1677 and after 2262.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import strings_from_list
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.frame.series import Series
from firepanda.kernel.scalar import civil_scalar, temporal_field_scalar
from firepanda.kernel.temporal import (
    FIELD_CODES,
    FIELD_IS_LEAP_YEAR,
    FIELD_ISO_YEAR,
    TRUNC_CENTURY,
    TRUNC_DAY,
    TRUNC_DECADE,
    TRUNC_HOUR,
    TRUNC_MICROSECOND,
    TRUNC_MILLENNIUM,
    TRUNC_MILLISECOND,
    TRUNC_MINUTE,
    TRUNC_MONTH,
    TRUNC_QUARTER,
    TRUNC_SECOND,
    TRUNC_WEEK,
    TRUNC_YEAR,
    TemporalField,
    civil_from_days,
    days_from_civil,
    extract_field,
    field_dtype,
    sql_field_named,
    temporal_as_timestamp,
    temporal_date,
    temporal_field,
    temporal_normalize,
    temporal_truncate,
    trunc_unit_named,
)

comptime NULL_ROW = Int64.MIN
"""The marker `stamps` reads as a missing row rather than as an instant.

It is the smallest number there is rather than something readable like -1,
because -1 is the last second of 1969 and is one of the rows this file cares
most about."""


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


def numbers(col: AnyArray, field: TemporalField) raises -> List[Int]:
    """Reads one field out of a column as plain integers.

    Args:
        col: The column.
        field: The field, which must be one of the twelve that are numbers.

    Returns:
        One integer per row, with a null row reading as zero.
    """
    var answer = temporal_field(col, field)
    ref view = answer.as_typed_view[DType.int32]()
    var out = List[Int](capacity=len(view))
    for i in range(len(view)):
        out.append(Int(view[i]))
    return out^


def counts(col: AnyArray, field: TemporalField) raises -> List[Int]:
    """Reads one of the three ISO fields out of a column as plain integers.

    Args:
        col: The column.
        field: The field, which must be one of the three that are unsigned.

    Returns:
        One integer per row, with a null row reading as zero.
    """
    var answer = temporal_field(col, field)
    ref view = answer.as_typed_view[DType.uint32]()
    var out = List[Int](capacity=len(view))
    for i in range(len(view)):
        out.append(Int(view[i]))
    return out^


def flags(col: AnyArray, field: TemporalField) raises -> List[Bool]:
    """Reads one field out of a column as plain booleans.

    Args:
        col: The column.
        field: The field, which must be one of the seven that are predicates.

    Returns:
        One boolean per row, with a null row reading as false.
    """
    var answer = temporal_field(col, field)
    ref view = answer.as_typed_view[DType.bool]()
    var out = List[Bool](capacity=len(view))
    for i in range(len(view)):
        out.append(Bool(view[i]))
    return out^


def test_the_last_second_of_1969_is_not_the_first_of_1970() raises:
    """The one row that tells rounding down apart from rounding towards zero.

    Timestamp -1 is 1969-12-31 23:59:59. Divided by 86400 rounding down it is day
    -1, and rounding towards zero it is day 0, which is 1970-01-01. Every field
    below moves if that division stops rounding down, so this test is the guard
    on the property the whole file rests on."""
    var col = stamps([Int64(-1)], TimeUnit.SECOND)
    assert_equal(numbers(col, TemporalField.YEAR)[0], 1969, "year")
    assert_equal(numbers(col, TemporalField.MONTH)[0], 12, "month")
    assert_equal(numbers(col, TemporalField.DAY)[0], 31, "day")
    assert_equal(numbers(col, TemporalField.HOUR)[0], 23, "hour")
    assert_equal(numbers(col, TemporalField.MINUTE)[0], 59, "minute")
    assert_equal(numbers(col, TemporalField.SECOND)[0], 59, "second")
    assert_equal(numbers(col, TemporalField.DAY_OF_YEAR)[0], 365, "day of year")
    assert_true(flags(col, TemporalField.IS_YEAR_END)[0], "is year end")


def test_the_epoch_itself_is_the_first_of_january() raises:
    """The row either side of the one above, so that a fix to that one which
    moved everything by a day would not pass both."""
    var col = stamps([Int64(0)], TimeUnit.SECOND)
    assert_equal(numbers(col, TemporalField.YEAR)[0], 1970, "year")
    assert_equal(numbers(col, TemporalField.MONTH)[0], 1, "month")
    assert_equal(numbers(col, TemporalField.DAY)[0], 1, "day")
    assert_equal(numbers(col, TemporalField.HOUR)[0], 0, "hour")
    assert_equal(numbers(col, TemporalField.DAY_OF_WEEK)[0], 3, "a Thursday")
    assert_true(flags(col, TemporalField.IS_YEAR_START)[0], "is year start")


def test_a_day_deep_before_the_epoch_still_reads_the_right_way_round() raises:
    """1900-01-01, which is timestamp -2208988800, and is far enough back that a
    sign handled only at the first division would have drifted by now."""
    var col = stamps([Int64(-2208988800)], TimeUnit.SECOND)
    assert_equal(numbers(col, TemporalField.YEAR)[0], 1900, "year")
    assert_equal(numbers(col, TemporalField.MONTH)[0], 1, "month")
    assert_equal(numbers(col, TemporalField.DAY)[0], 1, "day")
    assert_equal(numbers(col, TemporalField.DAY_OF_WEEK)[0], 0, "a Monday")


def test_the_leap_rule_has_three_clauses_and_all_three_are_here() raises:
    """2024 is divisible by four, 1900 by a hundred and 2000 by four hundred.

    A leap rule written as `year % 4 == 0` gets 1900 wrong, and one written
    without the four hundred clause gets 2000 wrong, and both of those pass every
    test that only ever looks at an ordinary year."""
    var col = stamps(
        [Int64(1709164800), Int64(-2208988800), Int64(951782400)],
        TimeUnit.SECOND,
    )
    var got = flags(col, TemporalField.IS_LEAP_YEAR)
    assert_true(got[0], "2024 is a leap year")
    assert_true(not got[1], "1900 is not")
    assert_true(got[2], "2000 is")

    var lengths = numbers(col, TemporalField.DAYS_IN_MONTH)
    assert_equal(lengths[0], 29, "February 2024")
    assert_equal(lengths[1], 31, "January 1900")
    assert_equal(lengths[2], 29, "February 2000")


def test_the_twenty_ninth_of_february_is_the_end_of_its_month() raises:
    """A month end predicate built on a fixed table of month lengths says no
    here, which is the failure this row exists to catch."""
    var col = stamps([Int64(1709164800)], TimeUnit.SECOND)
    assert_equal(numbers(col, TemporalField.DAY)[0], 29, "day")
    assert_equal(numbers(col, TemporalField.MONTH)[0], 2, "month")
    assert_equal(numbers(col, TemporalField.DAY_OF_YEAR)[0], 60, "day of year")
    assert_true(flags(col, TemporalField.IS_MONTH_END)[0], "is month end")
    assert_true(
        not flags(col, TemporalField.IS_QUARTER_END)[0], "is not a quarter end"
    )


def test_a_quarter_ends_on_four_days_a_year_and_starts_on_four_others() raises:
    """31 March, 30 June, 1 April and 1 July, which between them need the month
    number and the length of the month and would both pass with either one of
    those wrong if only one date were checked."""
    var col = stamps(
        [
            Int64(1711843200),
            Int64(1719705600),
            Int64(1711929600),
            Int64(1719792000),
        ],
        TimeUnit.SECOND,
    )
    var ends = flags(col, TemporalField.IS_QUARTER_END)
    assert_true(ends[0], "31 March ends a quarter")
    assert_true(ends[1], "30 June ends a quarter")
    assert_true(not ends[2], "1 April does not")
    assert_true(not ends[3], "1 July does not")

    var starts = flags(col, TemporalField.IS_QUARTER_START)
    assert_true(not starts[0], "31 March does not start one")
    assert_true(starts[2], "1 April starts one")
    assert_true(starts[3], "1 July starts one")

    var quarters = numbers(col, TemporalField.QUARTER)
    assert_equal(quarters[0], 1, "March is the first quarter")
    assert_equal(quarters[1], 2, "June is the second")
    assert_equal(quarters[2], 2, "April is the second")
    assert_equal(quarters[3], 3, "July is the third")


def test_the_clock_fields_come_out_of_the_remainder_below_the_day() raises:
    """23:59:59 on the last day of February 2024, which is the largest each of
    the three can be and is the row where a division by the wrong constant is
    visible in all three at once."""
    var col = stamps([Int64(1709251199)], TimeUnit.SECOND)
    assert_equal(numbers(col, TemporalField.HOUR)[0], 23, "hour")
    assert_equal(numbers(col, TemporalField.MINUTE)[0], 59, "minute")
    assert_equal(numbers(col, TemporalField.SECOND)[0], 59, "second")


def test_the_same_instant_reads_the_same_at_all_four_resolutions() raises:
    """A library that normalises everything to nanoseconds answers these too.
    What it cannot do is hold the 1677 and 2300 rows of the corpus at the same
    time, which is why the resolution stays on the column."""
    var second = stamps([Int64(1709251199)], TimeUnit.SECOND)
    var milli = stamps([Int64(1709251199_000)], TimeUnit.MILLI)
    var micro = stamps([Int64(1709251199_000_000)], TimeUnit.MICRO)
    var nano = stamps([Int64(1709251199_000_000_000)], TimeUnit.NANO)

    comptime for code in FIELD_CODES:
        var name = String(TemporalField(code))
        comptime if code >= FIELD_ISO_YEAR:
            var want = counts(second, TemporalField(code))[0]
            assert_equal(
                counts(milli, TemporalField(code))[0],
                want,
                String("milli ", name),
            )
            assert_equal(
                counts(micro, TemporalField(code))[0],
                want,
                String("micro ", name),
            )
            assert_equal(
                counts(nano, TemporalField(code))[0],
                want,
                String("nano ", name),
            )
        elif code >= FIELD_IS_LEAP_YEAR:
            var want = flags(second, TemporalField(code))[0]
            assert_equal(
                flags(milli, TemporalField(code))[0],
                want,
                String("milli ", name),
            )
            assert_equal(
                flags(micro, TemporalField(code))[0],
                want,
                String("micro ", name),
            )
            assert_equal(
                flags(nano, TemporalField(code))[0], want, String("nano ", name)
            )
        else:
            var want = numbers(second, TemporalField(code))[0]
            assert_equal(
                numbers(milli, TemporalField(code))[0],
                want,
                String("milli ", name),
            )
            assert_equal(
                numbers(micro, TemporalField(code))[0],
                want,
                String("micro ", name),
            )
            assert_equal(
                numbers(nano, TemporalField(code))[0],
                want,
                String("nano ", name),
            )


def test_the_sub_second_fields_split_at_the_microsecond() raises:
    """The `microsecond` field caps at 999999 in pandas and the last three
    digits go in `nanosecond`, so a nanosecond column with 123456789 below the
    second answers 123456 and 789 and not 123456789 and 0."""
    var nano = stamps([Int64(123_456_789)], TimeUnit.NANO)
    assert_equal(numbers(nano, TemporalField.MICROSECOND)[0], 123456, "us")
    assert_equal(numbers(nano, TemporalField.NANOSECOND)[0], 789, "ns")

    var micro = stamps([Int64(123_456)], TimeUnit.MICRO)
    assert_equal(numbers(micro, TemporalField.MICROSECOND)[0], 123456, "us")
    assert_equal(numbers(micro, TemporalField.NANOSECOND)[0], 0, "ns")

    var milli = stamps([Int64(123)], TimeUnit.MILLI)
    assert_equal(numbers(milli, TemporalField.MICROSECOND)[0], 123000, "us")
    assert_equal(numbers(milli, TemporalField.NANOSECOND)[0], 0, "ns")

    var second = stamps([Int64(1)], TimeUnit.SECOND)
    assert_equal(numbers(second, TemporalField.MICROSECOND)[0], 0, "us")
    assert_equal(numbers(second, TemporalField.NANOSECOND)[0], 0, "ns")


def test_the_week_starts_on_monday_because_pandas_says_so() raises:
    """Numbered from zero for Monday, which is neither the ISO numbering nor
    C's, and is the one a caller comparing against pandas gets."""
    var col = stamps(
        [
            Int64(1709251199),
            Int64(1709251200),
            Int64(1709337600),
            Int64(0),
            Int64(-1),
        ],
        TimeUnit.SECOND,
    )
    var got = numbers(col, TemporalField.DAY_OF_WEEK)
    assert_equal(got[0], 3, "29 February 2024 was a Thursday")
    assert_equal(got[1], 4, "the Friday after it")
    assert_equal(got[2], 5, "the Saturday after that")
    assert_equal(got[3], 3, "the epoch was a Thursday")
    assert_equal(got[4], 2, "the day before it was a Wednesday")


def test_the_day_of_the_year_counts_the_leap_day_when_there_is_one() raises:
    """1 March is day 61 in a leap year and day 60 in an ordinary one, which is
    the only place the two calendars part company."""
    var leap = stamps([Int64(1709251200)], TimeUnit.SECOND)
    var plain = stamps([Int64(1677628800)], TimeUnit.SECOND)
    assert_equal(
        numbers(leap, TemporalField.DAY_OF_YEAR)[0], 61, "1 March 2024"
    )
    assert_equal(
        numbers(plain, TemporalField.DAY_OF_YEAR)[0], 60, "1 March 2023"
    )


def test_a_null_row_stays_null_and_does_not_read_as_the_epoch() raises:
    """A null holds a zero in the values buffer, and zero is a real instant that
    converts to a real date, so the repair after the loop is doing work here
    rather than tidying up."""
    var col = stamps([Int64(0), NULL_ROW, Int64(1709164800)], TimeUnit.SECOND)
    var years = temporal_field(col, TemporalField.YEAR)
    assert_true(years.is_valid(0), "row 0 is present")
    assert_true(not years.is_valid(1), "row 1 is missing")
    assert_true(years.is_valid(2), "row 2 is present")

    var leap = temporal_field(col, TemporalField.IS_LEAP_YEAR)
    assert_true(not leap.is_valid(1), "the predicate keeps the null too")


def test_a_field_is_int32_or_bool_and_never_int64() raises:
    """The numbers come out int32 in pandas and bool for the predicates, and the
    conformance suite compares integer widths exactly, so a year that is right
    in int64 is a failing case rather than a passing one."""
    var col = stamps([Int64(0)], TimeUnit.SECOND)
    assert_equal(
        String(temporal_field(col, TemporalField.YEAR).type), "int32", "year"
    )
    assert_equal(
        String(temporal_field(col, TemporalField.DAYS_IN_MONTH).type),
        "int32",
        "days in month",
    )
    assert_equal(
        String(temporal_field(col, TemporalField.IS_LEAP_YEAR).type),
        "bool",
        "is leap year",
    )
    assert_equal(field_dtype(0), DType.int32, "the first field code")
    assert_equal(field_dtype(18), DType.bool, "the last field code")


def test_the_date_of_a_timestamp_is_a_date_column() raises:
    """In pandas `dt.date` is an object column of Python dates, because
    the numpy backend has no date dtype. firepanda has one, and says so."""
    var col = stamps([Int64(-1), Int64(0), Int64(1709251199)], TimeUnit.SECOND)
    var got = temporal_date(col)
    assert_equal(String(got.type), "date32[day]", "the type")
    ref view = got.as_typed_view[DType.int32]()
    assert_equal(Int(view[0]), -1, "the day before the epoch")
    assert_equal(Int(view[1]), 0, "the epoch")
    assert_equal(Int(view[2]), 19782, "29 February 2024")


def test_normalising_keeps_the_type_and_moves_the_clock_to_midnight() raises:
    """The difference between this and `dt.date` is entirely the type, which is
    why both exist and why a test that only checked the values would pass with
    one implemented as the other."""
    var col = stamps([Int64(-1), Int64(1709251199)], TimeUnit.SECOND)
    var got = temporal_normalize(col)
    assert_equal(String(got.type), "datetime64[s]", "the type")
    ref view = got.as_typed_view[DType.int64]()
    assert_equal(Int(view[0]), -86400, "midnight before the epoch")
    assert_equal(Int(view[1]), 1709164800, "midnight on 29 February 2024")


def test_normalising_a_millisecond_column_answers_milliseconds() raises:
    """The unit travels through, so the midnight is in the column's own unit and
    not in seconds, which is the mistake a shared implementation would make."""
    var col = stamps([Int64(1709251199_000)], TimeUnit.MILLI)
    var got = temporal_normalize(col)
    assert_equal(String(got.type), "datetime64[ms]", "the type")
    assert_equal(
        Int(got.as_typed_view[DType.int64]()[0]),
        1709164800_000,
        "midnight in milliseconds",
    )


def test_a_date_column_answers_the_calendar_and_no_clock() raises:
    """A date32 column is already a count of days, so the same fields come off it
    with the day division doing nothing and every clock field reading zero."""
    var days = Array[DType.int32](1)
    days.set_valid(0, Int32(19782))
    var col = AnyArray(days^.into_data(), LogicalType.DATE32)
    assert_equal(numbers(col, TemporalField.YEAR)[0], 2024, "year")
    assert_equal(numbers(col, TemporalField.MONTH)[0], 2, "month")
    assert_equal(numbers(col, TemporalField.DAY)[0], 29, "day")
    assert_equal(numbers(col, TemporalField.HOUR)[0], 0, "hour")
    assert_equal(numbers(col, TemporalField.MICROSECOND)[0], 0, "microsecond")


def test_a_zoned_column_is_refused_rather_than_answered_in_utc() raises:
    """The stored integers are UTC, so answering from them would give an hour
    that is silently seven off in New York. Refusing is the honest answer until
    there is a zone database. See issue 287."""
    var col = Array[DType.int64](1)
    col.set_valid(0, Int64(0))
    var zoned = AnyArray(
        col^.into_data(),
        LogicalType.timestamp(TimeUnit.SECOND, TimeZone("America/New_York")),
    )
    with assert_raises(contains="time zone database"):
        _ = temporal_field(zoned, TemporalField.HOUR)
    with assert_raises(contains="time zone database"):
        _ = temporal_date(zoned)


def test_a_column_that_is_not_temporal_has_no_calendar_in_it() raises:
    """An integer column holds the same bits a timestamp column does, so this is
    refused on the type rather than on the layout."""
    var plain = Array[DType.int64](1)
    plain.set_valid(0, Int64(0))
    with assert_raises(contains="date or a timestamp"):
        _ = temporal_field(AnyArray(plain^), TemporalField.YEAR)

    var text = AnyArray(strings_from_list(["a"]))
    with assert_raises(contains="date or a timestamp"):
        _ = temporal_field(text, TemporalField.YEAR)


def test_a_field_code_that_is_not_a_field_is_refused() raises:
    """The dispatch is a compile time walk over twenty two codes, so a twenty
    third falls off the end of it rather than reaching a loop."""
    var col = stamps([Int64(0)], TimeUnit.SECOND)
    with assert_raises(contains="is not a field code"):
        _ = temporal_field(col, TemporalField(22))


def test_every_field_prints_under_the_name_pandas_gives_it() raises:
    """The names are what a caller sees in a message, and they are pandas'
    spellings rather than this library's: `dayofweek` and not `day_of_week`."""
    assert_equal(String(TemporalField.YEAR), "year", "year")
    assert_equal(String(TemporalField.DAY_OF_WEEK), "dayofweek", "dayofweek")
    assert_equal(String(TemporalField.DAY_OF_YEAR), "dayofyear", "dayofyear")
    assert_equal(
        String(TemporalField.DAYS_IN_MONTH), "days_in_month", "days_in_month"
    )
    assert_equal(
        String(TemporalField.IS_YEAR_END), "is_year_end", "is_year_end"
    )
    assert_equal(String(TemporalField.ISO_YEAR), "isoyear", "isoyear")
    assert_equal(
        String(TemporalField(22)), "field 22", "a code that is not one"
    )


def test_the_kernel_agrees_with_the_twin_on_every_field() raises:
    """The twin walks the calendar a year at a time from 1970 and the kernel
    divides into a four hundred year cycle, so the two agreeing is evidence
    rather than a tautology. `tests/fuzz/kernel.mojo` runs this on random data
    forever; what is here is the fixed set, which includes the rows the fuzzer
    would take a long time to draw."""
    var values = [
        Int64(-2208988800),
        Int64(-86401),
        Int64(-1),
        Int64(0),
        Int64(1),
        Int64(951782400),
        Int64(1709164800),
        Int64(1709251199),
        Int64(1711843200),
        Int64(4102444800),
    ]
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])

    comptime for code in FIELD_CODES:
        comptime result = field_dtype(code)
        var fast = extract_field[DType.int64, code, result](col, 86400, 1)
        var slow = temporal_field_scalar[code, result](col, 86400, 1)
        for i in range(len(values)):
            assert_equal(
                fast[i],
                slow[i],
                String("row ", i, " of ", TemporalField(code)),
            )


def test_the_twin_and_the_kernel_agree_on_a_date_far_from_1970() raises:
    """The twin's loop is one iteration per year, so this is the slowest thing
    either of them does and the one place the walk could run off the end."""
    assert_equal(civil_scalar(-719162).year, 1, "the first of January, year 1")
    assert_equal(civil_scalar(-719162).month, 1, "month")
    assert_equal(civil_scalar(-719162).day, 1, "day")

    var fast = civil_from_days[1](SIMD[DType.int64, 1](-719162))
    assert_equal(Int(fast.year[0]), 1, "the kernel says the same year")
    assert_equal(Int(fast.month[0]), 1, "month")
    assert_equal(Int(fast.day[0]), 1, "day")


def test_a_series_keeps_its_name_and_its_labels_through_a_field() raises:
    """pandas answers `s.dt.year` with the same index and the same name, so a
    field is not the place a label goes missing."""
    var col = stamps([Int64(0), Int64(1709164800)], TimeUnit.SECOND)
    var s = Series("when", col^)
    var got = s.dt(TemporalField.YEAR)
    assert_equal(got.name, "when", "the name")
    assert_equal(len(got), 2, "the height")
    assert_equal(String(got.values.type), "int32", "the dtype")


def test_a_series_looks_a_field_up_by_its_pandas_name() raises:
    """The twenty one names the Python layer will hold, resolved in one place so
    that there is only ever one table of them."""
    var col = stamps([Int64(1709251199)], TimeUnit.SECOND)
    var s = Series("when", col^)
    assert_equal(
        Int(s.dt("year").values.as_typed_view[DType.int32]()[0]), 2024, "year"
    )
    assert_equal(
        Int(s.dt("dayofweek").values.as_typed_view[DType.int32]()[0]),
        3,
        "dayofweek",
    )
    assert_equal(String(s.dt("date").values.type), "date32[day]", "date")
    assert_equal(
        String(s.dt("normalize").values.type), "datetime64[s]", "normalize"
    )
    with assert_raises(contains="no field called"):
        _ = s.dt("day_of_week")


def test_the_minute_field_agrees_with_duckdb_either_side_of_the_epoch() raises:
    """ClickBench q18 groups by `extract(minute FROM EventTime)`.

    Sixteen rows read off a running DuckDB, chosen around the three places the
    arithmetic could go wrong: the minute boundary, the hour boundary, and the
    epoch, where a division that truncated towards zero instead of flooring
    would put the last minute of 1969 into 1970. pandas answers all sixteen the
    same way and was checked against the same list.

    There is no leap second row and there cannot be one. A Unix timestamp is a
    count of seconds that pretends every day has 86,400 of them, so 23:59:60 has
    no integer to be, and neither DuckDB nor pandas nor this library can hold the
    instant the question would be about. The boundary a leap second row would be
    testing is the minute boundary, which is the first four rows here.
    """
    var rows: List[Int64] = [
        Int64(-5401),
        -3600,
        -1800,
        -61,
        -60,
        -59,
        -1,
        0,
        1,
        59,
        60,
        61,
        1800,
        3600,
        5401,
        1372636859,
    ]
    var want = [29, 0, 30, 58, 59, 59, 59, 0, 0, 0, 1, 1, 30, 0, 30, 0]
    var got = numbers(stamps(rows, TimeUnit.SECOND), TemporalField.MINUTE)
    for i in range(len(want)):
        assert_equal(got[i], want[i], "row " + String(i))


def test_the_sql_names_reach_the_fields_they_name() raises:
    # The specifiers DuckDB takes, against the codes here. Every one of these
    # was checked against a running DuckDB 1.5.1 before it was written down.
    assert_true(sql_field_named("year") == TemporalField.YEAR, "year")
    assert_true(sql_field_named("mon") == TemporalField.MONTH, "the short one")
    assert_true(sql_field_named("days") == TemporalField.DAY, "the plural")
    assert_true(sql_field_named("doy") == TemporalField.DAY_OF_YEAR, "doy")
    assert_true(sql_field_named("quarter") == TemporalField.QUARTER, "quarter")


def test_the_sql_week_is_the_iso_week() raises:
    # DuckDB's `week` is the ISO week, so 1 January 2021 is week 53 of 2020 and
    # not week 1 of 2021. Reading it off the calendar instead would be wrong for
    # a handful of days a year, which is the worst kind of wrong.
    assert_true(sql_field_named("week") == TemporalField.ISO_WEEK, "week")
    assert_true(
        sql_field_named("weekofyear") == TemporalField.ISO_WEEK, "the long one"
    )
    assert_true(sql_field_named("isoyear") == TemporalField.ISO_YEAR, "isoyear")
    assert_true(sql_field_named("isodow") == TemporalField.ISO_DAY, "isodow")


def test_the_names_that_mean_different_things_are_not_in_the_sql_table() raises:
    # `dayofweek` starts the week on a different day in the two systems and
    # `microsecond` counts from a different place, so neither is in here. A
    # table that answered them with the pandas field would be wrong and would
    # look right.
    with assert_raises(contains="no field SQL calls dayofweek"):
        _ = sql_field_named("dayofweek")
    with assert_raises(contains="no field SQL calls dow"):
        _ = sql_field_named("dow")
    with assert_raises(contains="no field SQL calls microsecond"):
        _ = sql_field_named("microsecond")


def test_a_field_neither_system_has_is_refused() raises:
    with assert_raises(contains="no field SQL calls epoch"):
        _ = sql_field_named("epoch")
    with assert_raises(contains="no field SQL calls nosuch"):
        _ = sql_field_named("nosuch")


def truncs(values: List[Int64], unit: Int) raises -> List[Int64]:
    """Truncates a microsecond column and reads the answer back.

    Args:
        values: The instants, in microseconds since the epoch, with `NULL_ROW`
            meaning a null.
        unit: One of the `TRUNC_` codes.

    Returns:
        One integer per row, with a null row reading as zero.
    """
    var answer = temporal_truncate(stamps(values, TimeUnit.MICRO), unit)
    ref view = answer.as_typed_view[DType.int64]()
    var out = List[Int64](capacity=len(view))
    for i in range(len(view)):
        out.append(view[i])
    return out^


def test_a_day_number_survives_the_round_trip_through_the_calendar() raises:
    # Four hundred years of days taken apart and put back together. That is a
    # whole Gregorian cycle, so every leap rule and every month length is in
    # here, and it runs either side of the epoch because the two lines that
    # divide a negative number are the ones most likely to be wrong.
    for day in range(-73049, 73049, 7):
        var one = SIMD[DType.int64, 1](day)
        var civil = civil_from_days[1](one)
        assert_equal(
            Int(days_from_civil[1](civil.year, civil.month, civil.day)[0]),
            day,
            String("day ", day),
        )


def test_the_first_day_of_a_month_is_what_truncating_to_one_gives() raises:
    # The reference numbers are DuckDB's, read as microseconds since the epoch,
    # off 2013-07-15 13:45:12.345678.
    var when = List[Int64](capacity=1)
    when.append(1373895912345678)
    assert_equal(truncs(when, TRUNC_YEAR)[0], 1356998400000000, "year")
    assert_equal(truncs(when, TRUNC_QUARTER)[0], 1372636800000000, "quarter")
    assert_equal(truncs(when, TRUNC_MONTH)[0], 1372636800000000, "month")
    assert_equal(truncs(when, TRUNC_WEEK)[0], 1373846400000000, "week")
    assert_equal(truncs(when, TRUNC_DAY)[0], 1373846400000000, "day")


def test_the_clock_units_are_the_column_divided_and_nothing_else() raises:
    var when = List[Int64](capacity=1)
    when.append(1373895912345678)
    assert_equal(truncs(when, TRUNC_HOUR)[0], 1373893200000000, "hour")
    assert_equal(truncs(when, TRUNC_MINUTE)[0], 1373895900000000, "minute")
    assert_equal(truncs(when, TRUNC_SECOND)[0], 1373895912000000, "second")
    assert_equal(
        truncs(when, TRUNC_MILLISECOND)[0], 1373895912345000, "millisecond"
    )
    assert_equal(
        truncs(when, TRUNC_MICROSECOND)[0], 1373895912345678, "microsecond"
    )


def test_the_three_long_units_all_end_on_a_year_of_zeros() raises:
    var when = List[Int64](capacity=1)
    when.append(1373895912345678)
    assert_equal(truncs(when, TRUNC_DECADE)[0], 1262304000000000, "decade")
    assert_equal(truncs(when, TRUNC_CENTURY)[0], 946684800000000, "century")
    assert_equal(
        truncs(when, TRUNC_MILLENNIUM)[0], 946684800000000, "millennium"
    )


def test_truncating_before_the_epoch_goes_back_and_never_forward() raises:
    # The last second of 1969, which is the row the whole file is built around.
    # Every one of these has to land before it and not on the epoch.
    var when = List[Int64](capacity=1)
    when.append(-1000000)
    assert_equal(truncs(when, TRUNC_YEAR)[0], -31536000000000, "year")
    assert_equal(truncs(when, TRUNC_MONTH)[0], -2678400000000, "month")
    assert_equal(truncs(when, TRUNC_WEEK)[0], -259200000000, "week")
    assert_equal(truncs(when, TRUNC_DAY)[0], -86400000000, "day")

    # 1965-02-28, whose decade starts in 1960 and whose century starts in 1900.
    var older = List[Int64](capacity=1)
    older.append(-152755200000000)
    assert_equal(truncs(older, TRUNC_DECADE)[0], -315619200000000, "decade")
    assert_equal(truncs(older, TRUNC_CENTURY)[0], -2208988800000000, "century")
    assert_equal(truncs(older, TRUNC_QUARTER)[0], -157766400000000, "quarter")


def test_a_week_runs_monday_to_sunday_wherever_it_lands() raises:
    # The 15th of July 2013 was a Monday and truncates to itself, the 14th was
    # the Sunday before it and goes back six days to the 8th.
    var week = List[Int64](capacity=2)
    week.append(1373846400000000)
    week.append(1373760000000000)
    var got = truncs(week, TRUNC_WEEK)
    assert_equal(got[0], 1373846400000000, "a Monday truncates to itself")
    assert_equal(got[1], 1373241600000000, "a Sunday goes back six days")


def test_a_leap_day_truncates_to_a_month_and_a_week_like_any_other() raises:
    # The 29th of February 2016, which is a Monday, so the week is itself.
    var leap = List[Int64](capacity=1)
    leap.append(1456747200000000)
    assert_equal(truncs(leap, TRUNC_MONTH)[0], 1454284800000000, "month")
    assert_equal(truncs(leap, TRUNC_WEEK)[0], 1456704000000000, "week")


def test_a_null_row_truncates_to_nothing() raises:
    var when = List[Int64](capacity=2)
    when.append(NULL_ROW)
    when.append(1373895912345678)
    var answer = temporal_truncate(stamps(when, TimeUnit.MICRO), TRUNC_MONTH)
    ref view = answer.as_typed_view[DType.int64]()
    assert_true(not view.is_valid(0), "the null row stays null")
    assert_true(view.is_valid(1), "and the one beside it does not")


def test_a_unit_finer_than_the_column_leaves_it_alone() raises:
    var when = List[Int64](capacity=1)
    when.append(1373895912345)
    var answer = temporal_truncate(
        stamps(when, TimeUnit.MILLI), TRUNC_MICROSECOND
    )
    ref view = answer.as_typed_view[DType.int64]()
    assert_equal(view[0], 1373895912345, "milliseconds are already whole")
    assert_true(
        answer.type == LogicalType.timestamp(TimeUnit.MILLI),
        "and the type is untouched",
    )


def test_truncating_keeps_the_resolution_it_was_given() raises:
    var when = List[Int64](capacity=1)
    when.append(1373895912)
    var answer = temporal_truncate(stamps(when, TimeUnit.SECOND), TRUNC_HOUR)
    ref view = answer.as_typed_view[DType.int64]()
    assert_equal(view[0], 1373893200, "seconds that are a whole hour")
    assert_true(
        answer.type == LogicalType.timestamp(TimeUnit.SECOND),
        "still seconds",
    )


def test_the_sql_unit_names_reach_the_units_they_name() raises:
    assert_equal(trunc_unit_named("year"), TRUNC_YEAR, "year")
    assert_equal(trunc_unit_named("years"), TRUNC_YEAR, "years")
    assert_equal(trunc_unit_named("y"), TRUNC_YEAR, "y")
    assert_equal(trunc_unit_named("mon"), TRUNC_MONTH, "mon")
    assert_equal(trunc_unit_named("quarters"), TRUNC_QUARTER, "quarters")
    assert_equal(trunc_unit_named("w"), TRUNC_WEEK, "w")
    assert_equal(trunc_unit_named("min"), TRUNC_MINUTE, "min")
    assert_equal(trunc_unit_named("us"), TRUNC_MICROSECOND, "us")
    assert_equal(trunc_unit_named("millennia"), TRUNC_MILLENNIUM, "millennia")


def test_a_field_name_is_not_a_unit_even_where_duckdb_takes_one() raises:
    # DuckDB folds these onto the unit the field lives in, so `dayofweek`
    # truncates to the day and `epoch` to the second. Reading them that way is
    # a guess about what somebody meant, so they are refused instead.
    with assert_raises(contains="nothing to truncate to called dayofweek"):
        _ = trunc_unit_named("dayofweek")
    with assert_raises(contains="nothing to truncate to called epoch"):
        _ = trunc_unit_named("epoch")
    with assert_raises(contains="nothing to truncate to called fortnight"):
        _ = trunc_unit_named("fortnight")


def test_a_date_becomes_midnight_on_the_day_it_named() raises:
    var when = List[Int64](capacity=3)
    when.append(1373895912345678)
    when.append(NULL_ROW)
    when.append(-1000000)
    var days = temporal_date(stamps(when, TimeUnit.MICRO))
    var back = temporal_as_timestamp(days, TimeUnit.MICRO)
    assert_true(
        back.type == LogicalType.timestamp(TimeUnit.MICRO),
        "a naive microsecond timestamp",
    )
    ref view = back.as_typed_view[DType.int64]()
    assert_equal(view[0], 1373846400000000, "the 15th of July at midnight")
    assert_true(not view.is_valid(1), "a missing day is still missing")
    assert_equal(view[2], -86400000000, "and the day before the epoch")


def test_a_timestamp_handed_to_the_same_call_is_just_restated() raises:
    var when = List[Int64](capacity=1)
    when.append(1373895912)
    var back = temporal_as_timestamp(
        stamps(when, TimeUnit.SECOND), TimeUnit.MICRO
    )
    ref view = back.as_typed_view[DType.int64]()
    assert_equal(view[0], 1373895912000000, "seconds multiplied up")


def test_a_date_column_is_sent_away_to_be_cast_first() raises:
    with assert_raises(contains="has to be cast to one first"):
        _ = temporal_truncate(
            temporal_date(stamps(List[Int64](), TimeUnit.MICRO)), TRUNC_YEAR
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
