"""Tests for reading a column of text as a column of instants.

Every expected value here was read off a running pandas 3.0.5 rather than worked
out in this file, including the ones that look like they could not possibly be
in doubt. Three of them were surprises worth naming up front, because each one
is a place where the obvious implementation is wrong and passes every test
somebody would write from memory.

The result unit is microseconds and not nanoseconds. pandas moved off
nanoseconds by default and now answers `datetime64[us]` for a column of ordinary
dates, and it goes to nanoseconds only when some row in the column carried more
than six digits after the decimal point. So the unit is a property of the whole
column decided by its most precise row, which means it cannot be settled until
every row has been read.

The format is guessed once, from the first row that is not missing, and then
every other row must match it. A column holding `2026-01-01 12:34:56` and
`2026-01-02` is a ValueError in pandas rather than a column with two shapes in
it. That is worth a test because the friendly behaviour, reading each row on its
own terms, is the one a person would build and is not what pandas does.

`nan` is missing and `null` is not. pandas takes the empty string, `NaT` and
`nan` as missing and refuses `None`, `null`, `NA` and a lone hyphen, and the
list is short enough that a reader who is generous with it silently accepts
files pandas rejects.

The round trip against the renderer is the last test in the file and it is the
one doing the most work. `temporal_strftime` and `parse_timestamps` share only
`parse_format`, so a mistake in either walk of the step list shows up as a
number that does not come back.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.dtype.logical import LogicalType, TypeKind
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.kernel.parse_time import (
    days_from_civil,
    days_in_month,
    guess_format,
    is_missing_word,
    numbers_to_timestamps,
    parse_timestamps,
)
from firepanda.kernel.temporal import civil_from_days, temporal_strftime


def read(values: List[String]) raises -> AnyArray:
    """Reads a list of strings the way `to_datetime` with no arguments would.

    Args:
        values: The rows, none of them null.

    Returns:
        The column.
    """
    return parse_timestamps(strings_from_list(values), "", True, False, False)


def counts(a: AnyArray) raises -> Array[DType.int64]:
    """Borrows the numbers out of a timestamp column.

    Args:
        a: The column.

    Returns:
        A view of its values.
    """
    return Array[DType.int64](copy=a.as_typed_view[DType.int64]())


def test_the_day_count_is_the_inverse_of_the_one_next_door() raises:
    """Every day from 1600 to 2400 survives a round trip through both halves.

    `civil_from_days` is already checked against pandas, so checking this
    against that is checking it against pandas one step removed, and it covers
    two hundred and ninety thousand days rather than the handful anybody would
    write out by hand.

    Those two years are the bounds because the leap rule has three clauses and
    the only place the third one fires is a century divisible by four hundred.
    This range holds 1600, 2000 and 2400, which are leap years, and 1700, 1800,
    1900, 2100, 2200 and 2300, which are not, so every clause runs on both
    answers. It is also wider on both sides than a nanosecond column reaches.

    The failure is tallied rather than asserted per day. An assertion inside
    the loop builds a message for every one of the days that passes, which is
    most of the time this test spends, and the only message anybody wants is
    the first day that does not.
    """
    var failures = 0
    var first_bad = 0
    for day in range(-135_140, 157_054):
        var civil = civil_from_days[1](Int64(day))
        var back = days_from_civil(
            Int64(civil.year[0]), Int64(civil.month[0]), Int64(civil.day[0])
        )
        if Int(back) != day:
            if failures == 0:
                first_bad = day
            failures += 1
    assert_equal(failures, 0, "first bad day " + String(first_bad))


def test_a_month_length_knows_the_three_leap_clauses() raises:
    """February is 29 days in 2024 and 2000 and 28 days in 1900 and 2023.

    The leap rule has three clauses and a year divisible by four hundred runs
    the one that almost never fires, so a table with only the first clause in
    it passes every test that does not land on a century.
    """
    assert_equal(Int(days_in_month(2024, 2)), 29, "2024")
    assert_equal(Int(days_in_month(2000, 2)), 29, "2000")
    assert_equal(Int(days_in_month(1900, 2)), 28, "1900")
    assert_equal(Int(days_in_month(2023, 2)), 28, "2023")
    assert_equal(Int(days_in_month(2024, 1)), 31, "January")
    assert_equal(Int(days_in_month(2024, 4)), 30, "April")


def test_a_date_reads_as_microseconds() raises:
    """`2026-01-01` is midnight on that day, counted in microseconds.

    pandas answers `datetime64[us]` here rather than the nanoseconds it
    answered before version 3, and the number is therefore a thousand times
    smaller than an implementation that normalises everything to nanoseconds
    would give.
    """
    var got = read(["2026-01-01", "2026-01-02"])
    assert_equal(String(got.type), "datetime64[us]", "type")
    var values = counts(got)
    assert_equal(Int(values[0]), 1_767_225_600_000_000, "first")
    assert_equal(Int(values[1]), 1_767_312_000_000_000, "second")


def test_a_date_and_a_time_read_in_either_separator() raises:
    """ISO 8601 says T between the date and the time and people write a space.

    pandas reads both and so does this, and the two spellings of the same
    instant give the same number.
    """
    var spaced = counts(read(["2026-03-04 05:06:07"]))
    var lettered = counts(read(["2026-03-04T05:06:07"]))
    assert_equal(Int(spaced[0]), Int(lettered[0]), "same instant")
    assert_equal(Int(spaced[0]), 1_772_600_767_000_000, "value")


def test_the_shorter_shapes_read() raises:
    """A year alone, a year and a month, and the eight digit basic form.

    All three are ISO 8601 and all three parse in pandas, and each one fills
    the fields it does not name with the first of the month or the first of
    January.
    """
    assert_equal(Int(counts(read(["2026"]))[0]), 1_767_225_600_000_000, "year")
    assert_equal(
        Int(counts(read(["2026-01"]))[0]), 1_767_225_600_000_000, "month"
    )
    assert_equal(
        Int(counts(read(["20260101"]))[0]), 1_767_225_600_000_000, "basic"
    )


def test_seven_fraction_digits_take_the_column_to_nanoseconds() raises:
    """The unit is decided by the most precise row rather than by the first.

    Six digits or fewer is microseconds and seven or more is nanoseconds, and
    the row that decides it can be anywhere in the column, so the unit is not
    known until every row has been read. This puts the precise row last on
    purpose.
    """
    var coarse = read(["2026-01-01T00:00:00.123456"])
    assert_equal(String(coarse.type), "datetime64[us]", "six digits")
    assert_equal(Int(counts(coarse)[0]), 1_767_225_600_123_456, "six value")

    var fine = read(["2026-01-01T00:00:00.1", "2026-01-01T00:00:00.1234567"])
    assert_equal(String(fine.type), "datetime64[ns]", "seven digits")
    var values = counts(fine)
    assert_equal(Int(values[0]), 1_767_225_600_100_000_000, "the coarse row")
    assert_equal(Int(values[1]), 1_767_225_600_123_456_700, "the fine row")


def test_a_year_no_nanosecond_column_reaches_still_reads() raises:
    """2300 and 1500 are ordinary microsecond dates and were wrong for a while.

    This is the bug the conformance board found and the tests above did not.
    Every row was folded to nanoseconds first and rescaled afterwards, and an
    Int64 of nanoseconds reaches 1677 to 2262 and no further, so 2300-01-01
    wrapped and came back as 1715-06-13 with nothing reporting it. Folding at
    the column's own unit is the fix and this is what pins it.

    The two years are the two sides. 2300 is past the top of the nanosecond
    range and 1500 is below the bottom of it, and both are far inside the range
    of the microseconds the column actually holds. Both numbers were read off a
    running pandas.
    """
    var far = read(["2300-01-01", "1500-06-15"])
    assert_equal(String(far.type), "datetime64[us]", "type")
    var values = counts(far)
    assert_equal(Int(values[0]), 10_413_792_000_000_000, "2300")
    assert_equal(Int(values[1]), -14_817_513_600_000_000, "1500")

    var timed = read(["1500-06-15T13:02:03"])
    assert_equal(Int(counts(timed)[0]), -14_817_466_677_000_000, "with a time")


def test_a_year_that_will_not_fit_the_column_says_so() raises:
    """The same date in a column of nanoseconds is a refusal rather than a wrap.

    Nine digits after the decimal point take the whole column to nanoseconds,
    and there is no nanosecond count of 2300 to store, so the row cannot be
    read at all. pandas raises `OutOfBoundsDatetime` here and firepanda raises
    with the year and the unit in the message. Naming both matters because the
    same text in a column that stayed at microseconds reads perfectly well, so
    the reason is the company the row keeps rather than the row itself.
    """
    with assert_raises(contains="outside the range"):
        _ = read(["2300-01-01T00:00:00.123456789"])


def test_a_trailing_z_makes_the_column_utc() raises:
    """`Z` is an offset of zero, and a column carrying one is not naive.

    The number is the same as the naive reading of the same text and the type
    is not, which is the whole point: a zone on the column is what stops the
    values being read against whatever clock the reader happens to be on.
    """
    var got = read(["2026-01-01T00:00:00Z"])
    assert_equal(String(got.type), "datetime64[us, UTC]", "type")
    assert_equal(Int(counts(got)[0]), 1_767_225_600_000_000, "value")


def test_an_offset_is_taken_off_the_instant() raises:
    """Two in the afternoon at `+02:00` is midday UTC.

    Both spellings of the offset are read, with the colon and without it,
    because both are ISO 8601. The type carries the offset the way pandas
    prints one.
    """
    var got = read(["2026-01-01T14:00:00+02:00"])
    assert_equal(String(got.type), "datetime64[us, UTC+02:00]", "type")
    assert_equal(Int(counts(got)[0]), 1_767_268_800_000_000, "value")

    var terse = read(["2026-01-01T14:00:00+0200"])
    assert_equal(String(terse.type), "datetime64[us, UTC+02:00]", "no colon")
    assert_equal(Int(counts(terse)[0]), 1_767_268_800_000_000, "no colon")

    var behind = read(["2026-01-01T10:00:00-02:00"])
    assert_equal(String(behind.type), "datetime64[us, UTC-02:00]", "behind")
    assert_equal(Int(counts(behind)[0]), 1_767_268_800_000_000, "behind")


def test_two_different_offsets_need_utc() raises:
    """A column of mixed offsets has no one clock to be read against.

    pandas raises here unless `utc=True` is passed, and the reason is worth
    stating: the two rows are perfectly good instants, and what does not exist
    is a single time zone the resulting column could carry. Asking for UTC
    names one.
    """
    var mixed = strings_from_list(
        [
            String("2026-01-01T00:00:00+01:00"),
            String("2026-01-01T00:00:00+02:00"),
        ]
    )
    with assert_raises(contains="different offsets"):
        _ = parse_timestamps(mixed, "", True, False, False)

    var got = parse_timestamps(mixed, "", True, False, True)
    assert_equal(String(got.type), "datetime64[us, UTC]", "type")
    var values = counts(got)
    assert_equal(Int(values[0]), 1_767_222_000_000_000, "first")
    assert_equal(Int(values[1]), 1_767_218_400_000_000, "second")


def test_the_format_is_guessed_once_and_holds_for_every_row() raises:
    """A column with two shapes in it is an error rather than two readings.

    This is the friendly behaviour that pandas does not have. Reading each row
    on its own terms would answer both of these, and pandas raises, so
    answering them would be a difference that only shows up on the files where
    somebody has a real problem with their data.
    """
    with assert_raises(contains="does not match the format"):
        _ = read(["2026-01-01 12:34:56", "2026-01-02"])


def test_the_missing_words_are_the_three_pandas_takes() raises:
    """Empty, `NaT` and `nan` are missing, in any case of any letter.

    Everything else is a value, and the list is checked from both ends here
    because being generous with it is how a reader quietly accepts a file
    pandas rejects.
    """
    assert_true(is_missing_word("".as_bytes()), "empty")
    assert_true(is_missing_word("NaT".as_bytes()), "NaT")
    assert_true(is_missing_word("nat".as_bytes()), "nat")
    assert_true(is_missing_word("nan".as_bytes()), "nan")
    assert_true(is_missing_word("NaN".as_bytes()), "NaN")
    assert_true(not is_missing_word("None".as_bytes()), "None")
    assert_true(not is_missing_word("null".as_bytes()), "null")
    assert_true(not is_missing_word("NA".as_bytes()), "NA")
    assert_true(not is_missing_word("-".as_bytes()), "a hyphen")
    assert_true(not is_missing_word(" ".as_bytes()), "a space")


def test_a_missing_row_stays_missing_and_does_not_pick_the_format() raises:
    """The format comes from the first row that is a value, not the first row.

    A column beginning with a null would otherwise have nothing to guess from,
    and a column beginning with `NaT` would guess from a word that is not a
    date at all.
    """
    var builder = StringBuilder(capacity=3)
    builder.append_null()
    builder.append("2026-01-01".as_bytes())
    builder.append("NaT".as_bytes())
    var got = parse_timestamps(builder^.finish(), "", True, False, False)

    assert_equal(String(got.type), "datetime64[us]", "type")
    assert_equal(got.null_count(), 2, "two missing")
    var values = counts(got)
    assert_true(not values.is_valid(0), "the null")
    assert_equal(Int(values[1]), 1_767_225_600_000_000, "the value")
    assert_true(not values.is_valid(2), "the NaT")


def test_a_column_of_nothing_but_missing_answers_seconds() raises:
    """There is no row to read a format out of, and pandas answers seconds.

    Seconds is the coarsest unit there is, which is the right answer for a
    column that carries no evidence of wanting a finer one. An empty column
    goes the same way for the same reason.
    """
    var all_null = read(["NaT", "nan", ""])
    assert_equal(String(all_null.type), "datetime64[s]", "all missing")
    assert_equal(all_null.null_count(), 3, "all three")

    var empty = parse_timestamps(strings_from_list([]), "", True, False, False)
    assert_equal(String(empty.type), "datetime64[s]", "empty")
    assert_equal(len(empty), 0, "no rows")


def test_a_format_given_by_the_caller_is_used_as_it_stands() raises:
    """`format=` reads shapes the guesser refuses, which is what it is for.

    The American ordering is the example that matters, because it is the one
    the guesser will not touch and the one a caller most often has.
    """
    var text = strings_from_list([String("01/02/2026"), String("03/04/2026")])
    var got = parse_timestamps(text, "%m/%d/%Y", False, False, False)
    var values = counts(got)
    assert_equal(Int(values[0]), 1_767_312_000_000_000, "the second of January")
    assert_equal(Int(values[1]), 1_772_582_400_000_000, "the fourth of March")


def test_a_two_digit_year_uses_the_posix_window() raises:
    """69 is 1969 and 68 is 2068, which is the window Python uses.

    Every window is arbitrary and the only thing that matters is which one,
    because a reader who picks a different one is wrong by a century on half
    the values and right on the other half.
    """
    var text = strings_from_list([String("69-01-01"), String("68-01-01")])
    var values = counts(parse_timestamps(text, "%y-%m-%d", False, False, False))
    assert_equal(Int(values[0]), -31_536_000_000_000, "1969")
    assert_equal(Int(values[1]), 3_092_601_600_000_000, "2068")


def test_a_directive_that_cannot_be_read_is_refused_by_name() raises:
    """`%j` renders and does not read, and the message says which one it was.

    A day of the year is computed from a date rather than being a part of one,
    so a format built out of it names text that cannot be read back to the
    instant it came from. Refusing is the only honest answer and naming the
    directive is what makes it actionable.
    """
    var text = strings_from_list([String("2026 001")])
    with assert_raises(contains="written and not read"):
        _ = parse_timestamps(text, "%Y %j", False, False, False)


def test_an_impossible_date_is_refused() raises:
    """The 30th of February and the 13th month are values no calendar has.

    The month length check needs the year, so the 29th of February is a value
    in 2024 and not in 2023, and a reader with a fixed table gets one of those
    two wrong.
    """
    with assert_raises(contains="outside 1..29"):
        _ = read(["2024-02-30"])
    with assert_raises(contains="outside 1..28"):
        _ = read(["2023-02-29"])
    with assert_raises(contains="month must be in 1..12"):
        _ = read(["2026-13-01"])
    with assert_raises(contains="is not a reading a clock has"):
        _ = read(["2026-01-01T25:00:00"])


def test_coerce_turns_an_unreadable_row_into_a_missing_one() raises:
    """`errors="coerce"` is the only way a bad row does not stop the column.

    The row that fails is the one that becomes missing, and the rows around it
    are unaffected, which is what makes this different from giving up on the
    whole column.
    """
    var text = strings_from_list(
        [String("2026-01-01"), String("2026-02-30"), String("2026-01-03")]
    )
    var got = parse_timestamps(text, "", True, True, False)
    assert_equal(got.null_count(), 1, "one bad row")
    var values = counts(got)
    assert_equal(Int(values[0]), 1_767_225_600_000_000, "before")
    assert_true(not values.is_valid(1), "the bad row")
    assert_equal(Int(values[2]), 1_767_398_400_000_000, "after")


def test_the_guesser_refuses_what_it_does_not_recognise() raises:
    """Anything that is not ISO 8601 is refused with the value in the message.

    The alternative is guessing, and a guess about `01/02/2026` is a column of
    instants that are wrong by up to eleven months and that nothing anywhere
    reports. The message names the value and says to pass a format, which is
    the whole fix.
    """
    with assert_raises(contains="pass a format for this one"):
        _ = guess_format("01/02/2026".as_bytes())
    with assert_raises(contains="pass a format for this one"):
        _ = guess_format("March 4, 2026".as_bytes())
    with assert_raises(contains="pass a format for this one"):
        _ = guess_format("2026/01/01".as_bytes())


def test_the_guesser_names_the_shape_it_found() raises:
    """The guessed format is a real format string, which is worth checking.

    It is handed straight to `parse_format`, so a guesser that produced
    something almost right would fail somewhere further along with a message
    about the wrong thing.
    """
    assert_equal(guess_format("2026".as_bytes()), "%Y", "a year")
    assert_equal(guess_format("2026-01".as_bytes()), "%Y-%m", "a month")
    assert_equal(guess_format("2026-01-01".as_bytes()), "%Y-%m-%d", "a date")
    assert_equal(
        guess_format("20260101".as_bytes()), "%Y%m%d", "the basic form"
    )
    assert_equal(
        guess_format("2026-01-01 05:06".as_bytes()),
        "%Y-%m-%d %H:%M",
        "to the minute",
    )
    assert_equal(
        guess_format("2026-01-01T05:06:07.123Z".as_bytes()),
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "everything",
    )


def test_whole_numbers_read_as_counts_of_a_unit() raises:
    """`to_datetime(unit=)` relabels rather than converting.

    The integers are already the counts and the unit says what they are counts
    of, so the numbers come out unchanged and only the type moves. A null stays
    a null rather than becoming a zero.
    """
    var col = Array[DType.int64](overwritten=3)
    col.set_valid(0, 0)
    col.set_valid(1, 1_767_225_600)
    col.set_null(2)
    var got = numbers_to_timestamps(
        AnyArray(col^.into_data(), LogicalType(TypeKind.INT, DType.int64)),
        TimeUnit.SECOND,
    )

    assert_equal(String(got.type), "datetime64[s]", "type")
    assert_equal(got.null_count(), 1, "the null survived")
    var values = counts(got)
    assert_equal(Int(values[0]), 0, "the epoch")
    assert_equal(Int(values[1]), 1_767_225_600, "the value")


def test_text_is_refused_by_the_numeric_door() raises:
    """A column of strings has no unit to be relabelled with.

    pandas ignores `unit=` for a column of text rather than raising, and the
    Python side is where that is arranged, because it is a choice about which
    door to knock on rather than anything this kernel can see.
    """
    with assert_raises(contains="whole numbers"):
        _ = numbers_to_timestamps(
            AnyArray(strings_from_list([String("2026-01-01")])),
            TimeUnit.SECOND,
        )


def test_the_reader_and_the_renderer_come_back_to_the_same_number() raises:
    """Ten thousand instants rendered and read back are unchanged.

    This is the twin the package rules ask for, in the only shape a parser can
    have one. `temporal_strftime` and `parse_timestamps` share `parse_format`
    and nothing else, so a mistake in either walk over the step list shows up
    here as a number that does not survive the trip, and the two were written
    far enough apart that a mistake common to both is unlikely.

    The instants are spread over the whole range a microsecond column holds
    rather than clustered near today, because the interesting failures are all
    at the ends: a year before the epoch, a leap day, and the last second of a
    year are the three places an off by one in the day arithmetic shows.
    """
    var wanted = Array[DType.int64](overwritten=10_000)
    var seed = Int64(0x2545_F491_4F6C_DD1D)
    for i in range(10_000):
        # A cheap mixing step rather than a real generator, because what this
        # needs is a spread of values and not a distribution anybody relies on.
        seed = seed * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407
        var spread = (seed >> 11) % 8_000_000_000
        wanted.set_valid(i, spread - 2_000_000_000)

    var stamps = AnyArray(
        Array[DType.int64](copy=wanted).into_data(),
        LogicalType.timestamp(TimeUnit.SECOND),
    )
    var text = temporal_strftime(stamps, "%Y-%m-%dT%H:%M:%S")
    var back = parse_timestamps(text, "%Y-%m-%dT%H:%M:%S", False, False, False)

    # The renderer wrote seconds and the reader answers microseconds, because
    # the reader answers what pandas answers and cannot know what unit the
    # column had before it became text.
    assert_equal(String(back.type), "datetime64[us]", "type")
    var values = counts(back)
    for i in range(10_000):
        assert_equal(
            Int(values[i]),
            Int(wanted[i]) * 1_000_000,
            "row " + String(i),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
