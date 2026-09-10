"""Tests for the zones that need no database.

A zone name is either a rule or a number. `America/New_York` is a rule, and what
it is ahead of UTC changes twice a year, changed on different days before 2007,
and will change again when somebody legislates, so reading a clock against it
needs the IANA database. `UTC` and `+05:30` are numbers. They state the whole
answer in themselves and no database can tell you anything about them the name
does not.

That is the line these tests draw. Everything on the number side works and
everything on the rule side is refused with a sentence saying why, and the split
is a property of the name rather than a list of zones somebody has to keep up to
date.

The other half is that converting and localising are opposite operations that
sound like the same one. Converting keeps the instant and moves the reading, so
it needs no offset at all and works for every zone there is. Localising keeps
the reading and moves the instant, so it needs the offset and is where the rule
zones are refused. Half the tests here are that pair being told apart.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.kernel.temporal import (
    ROUND_DOWN,
    field_named,
    temporal_date,
    temporal_field,
    temporal_normalize,
    temporal_round,
    temporal_strftime,
    temporal_tz_convert,
    temporal_tz_localize,
    temporal_tz_localize_none,
)


def offset_of(name: String) raises -> Int64:
    """Reads the offset a zone name states, and fails if it states none.

    Args:
        name: The zone name.

    Returns:
        The offset in seconds.

    Raises:
        Error: If the name is one this cannot read an offset out of.
    """
    var found = TimeZone(name).fixed_offset()
    if not found:
        raise Error("no fixed offset in " + name)
    return found.value()


def names_an_offset(name: String) raises -> Bool:
    """Reports whether a zone name states its own offset.

    Args:
        name: The zone name.

    Returns:
        True if the name is a number rather than a rule.
    """
    return Bool(TimeZone(name).fixed_offset())


def stamps(
    values: List[Int64], unit: TimeUnit, zone: String
) raises -> AnyArray:
    """Builds a timestamp column.

    Args:
        values: The counts since the epoch.
        unit: What they are counts of.
        zone: The clock they are read against, empty for none.

    Returns:
        The column, with every row present.
    """
    var out = Array[DType.int64](len(values))
    for i in range(len(values)):
        out[i] = values[i]
    return AnyArray(
        out^.into_data(), LogicalType.timestamp(unit, TimeZone(zone))
    )


def row(col: AnyArray, i: Int) raises -> Int64:
    """Reads one stored count out of a temporal column.

    Args:
        col: The column.
        i: Which row.

    Returns:
        The count, which for a zoned column is UTC.
    """
    return col.as_typed_view[DType.int64]()[i]


def field(col: AnyArray, name: String) raises -> Int32:
    """Reads one calendar field off the first row of a column.

    Args:
        col: The column.
        name: The pandas name of the field.

    Returns:
        The value.
    """
    return temporal_field(col, field_named(name)).as_typed_view[DType.int32]()[
        0
    ]


def noon() -> List[Int64]:
    """Midday UTC on the first of January 2024, and the two hours after it.

    Returns:
        Three counts of seconds since the epoch.
    """
    return [Int64(1704110400), 1704114000, 1704117600]


def test_a_name_is_a_rule_or_a_number() raises:
    """The whole slice rests on telling those two apart from the name alone."""
    assert_equal(offset_of("UTC"), 0, "the one everybody has")
    assert_equal(offset_of("utc"), 0, "and it is not case sensitive")
    assert_equal(offset_of("+05:30"), 19800, "the Arrow spelling")
    assert_equal(offset_of("-08:00"), -28800, "and the other direction")
    assert_equal(offset_of("+0530"), 19800, "the colon is optional")
    assert_equal(offset_of("+05"), 18000, "and so are the minutes")
    assert_equal(
        offset_of("UTC+05:30"),
        19800,
        "the form pandas prints when it made the zone itself",
    )

    assert_true(
        not names_an_offset("America/New_York"),
        "a rule, and the database is what reads it",
    )
    assert_true(
        not names_an_offset("Etc/GMT+5"),
        (
            "an IANA name whose sign runs the other way, and reading it as a"
            " number would be five hours out in the direction nobody checks"
        ),
    )
    assert_true(not names_an_offset("Europe/Paris"), "a rule")
    assert_true(not names_an_offset("+5:30"), "one digit is not the spelling")
    assert_true(not names_an_offset("+05:99"), "and ninety nine is not minutes")


def test_converting_moves_the_reading_and_not_the_instant() raises:
    """The sentence that separates convert from localize, as an assertion."""
    var zoned = stamps(noon(), TimeUnit.SECOND, "America/New_York")
    var moved = temporal_tz_convert(zoned, "Asia/Kolkata")
    assert_equal(
        String(moved.type),
        "datetime64[s, Asia/Kolkata]",
        "the name changed",
    )
    assert_equal(row(moved, 0), 1704110400, "and the instant did not")
    assert_equal(len(moved), 3, "on every row")


def test_converting_asks_no_database_and_so_takes_any_name() raises:
    """Converting never asks what the offset is, so a rule zone is as easy as a
    number one, which is the reverse of every other test here."""
    var zoned = stamps(noon(), TimeUnit.SECOND, "UTC")
    assert_equal(
        String(temporal_tz_convert(zoned, "Australia/Lord_Howe").type),
        "datetime64[s, Australia/Lord_Howe]",
        "a rule zone converts",
    )
    assert_equal(
        String(temporal_tz_convert(zoned, "+05:30").type),
        "datetime64[s, +05:30]",
        "and so does a number one",
    )

    with assert_raises(contains="carries no zone"):
        _ = temporal_tz_convert(
            stamps(noon(), TimeUnit.SECOND, ""), "Asia/Kolkata"
        )


def test_localising_moves_the_instant_and_not_the_reading() raises:
    """The other half of the pair, and the half that needs the offset."""
    var naive = stamps(noon(), TimeUnit.SECOND, "")
    var utc = temporal_tz_localize(naive, "UTC")
    assert_equal(String(utc.type), "datetime64[s, UTC]", "the name went on")
    assert_equal(row(utc, 0), 1704110400, "and UTC is no distance from UTC")

    var india = temporal_tz_localize(naive, "+05:30")
    assert_equal(
        String(india.type), "datetime64[s, +05:30]", "the name went on"
    )
    assert_equal(
        row(india, 0),
        1704110400 - 19800,
        (
            "and midday in a zone five and a half hours ahead is an earlier"
            " instant than midday in UTC"
        ),
    )

    with assert_raises(contains="already on"):
        _ = temporal_tz_localize(utc, "Asia/Kolkata")
    with assert_raises(contains="time zone database"):
        _ = temporal_tz_localize(naive, "America/New_York")


def test_taking_the_clock_off_keeps_the_reading() raises:
    """Dropping the zone is the exact opposite of converting, which is worth an
    assertion because both of them are spelled as a zone going away."""
    var india = stamps(noon(), TimeUnit.SECOND, "+05:30")
    var dropped = temporal_tz_localize_none(india)
    assert_equal(String(dropped.type), "datetime64[s]", "the name came off")
    assert_equal(
        row(dropped, 0),
        1704110400 + 19800,
        "and the reading is the one a person in that zone would have taken",
    )

    with assert_raises(contains="carries no zone"):
        _ = temporal_tz_localize_none(stamps(noon(), TimeUnit.SECOND, ""))
    with assert_raises(contains="time zone database"):
        _ = temporal_tz_localize_none(
            stamps(noon(), TimeUnit.SECOND, "America/New_York")
        )


def test_a_field_is_read_off_the_local_clock() raises:
    """The number a person actually reads, which is the one a wrong offset
    changes and the reason a rule zone is refused rather than answered."""
    assert_equal(field(stamps(noon(), TimeUnit.SECOND, ""), "hour"), 12)
    assert_equal(
        field(stamps(noon(), TimeUnit.SECOND, "UTC"), "hour"),
        12,
        "UTC is no distance from the stored instants",
    )
    assert_equal(
        field(stamps(noon(), TimeUnit.SECOND, "+05:30"), "hour"),
        17,
        "midday UTC is half past five in the evening five and a half on",
    )
    assert_equal(
        field(stamps(noon(), TimeUnit.SECOND, "+05:30"), "minute"),
        30,
        "and the half hour is why the offset is not kept in hours",
    )
    assert_equal(
        field(stamps(noon(), TimeUnit.SECOND, "-08:00"), "hour"),
        4,
        "and eight hours behind is four in the morning",
    )
    assert_equal(
        field(stamps(noon(), TimeUnit.SECOND, "-08:00"), "day"),
        1,
        "on the same day, which the next case is not",
    )
    assert_equal(
        field(stamps([Int64(1704067200)], TimeUnit.SECOND, "-08:00"), "day"),
        31,
        "midnight on New Year's Day is still December in California",
    )

    with assert_raises(contains="time zone database"):
        _ = field(stamps(noon(), TimeUnit.SECOND, "America/New_York"), "hour")


def test_the_other_readers_go_through_the_same_clock() raises:
    """One turn serves all of them, so this is a test that each one takes it."""
    var india = stamps(noon(), TimeUnit.SECOND, "+05:30")
    assert_equal(
        String(temporal_date(india).type),
        "date32[day]",
        "a date has no clock left to be on",
    )
    var text = temporal_strftime(india, "%H:%M")
    assert_equal(text[0], "17:30", "the format string reads the local clock")

    with assert_raises(contains="time zone database"):
        _ = temporal_strftime(
            stamps(noon(), TimeUnit.SECOND, "America/New_York"), "%H:%M"
        )


def test_a_result_that_is_still_a_time_goes_back_on_the_clock() raises:
    """Normalising and rounding read the local clock and answer an instant, so
    they are the two that make the round trip rather than half of it."""
    var india = stamps(noon(), TimeUnit.SECOND, "+05:30")
    var midnight = temporal_normalize(india)
    assert_equal(
        String(midnight.type),
        "datetime64[s, +05:30]",
        "the answer is still on the clock it was read from",
    )
    assert_equal(
        row(midnight, 0),
        1704067200 - 19800,
        "and it is the instant that local midnight was",
    )

    var floored = temporal_round(india, "h", ROUND_DOWN)
    assert_equal(String(floored.type), "datetime64[s, +05:30]")
    assert_equal(
        row(floored, 0),
        1704110400 - 1800,
        (
            "half past five floors to five, which is thirty minutes earlier,"
            " and flooring the stored instant would have moved nothing at all"
        ),
    )

    with assert_raises(contains="time zone database"):
        _ = temporal_normalize(
            stamps(noon(), TimeUnit.SECOND, "America/New_York")
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
