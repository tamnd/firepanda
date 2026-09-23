"""Tests for the zone database, read from the TZif files the system ships.

Every expected value here was measured with Python's `zoneinfo`, which reads the
same files and is what pandas answers from. The rows are chosen to reach each
part of the reader: the table a file carries, the time before the table starts,
the footer rule after it ends, a zone in the southern hemisphere whose summer
spans New Year, and a zone whose clock moves by thirty minutes.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.kernel.temporal import (
    IN_A_GAP_BACKWARD,
    IN_A_GAP_FORWARD,
    IN_A_GAP_NULL,
    IN_A_GAP_RAISE,
    IN_A_GAP_SHIFT,
    ON_A_FOLD_EARLIER,
    ON_A_FOLD_LATER,
    ON_A_FOLD_NULL,
    ON_A_FOLD_RAISE,
    ZonePolicy,
    temporal_tz_convert,
    temporal_tz_localize,
    temporal_tz_localize_none,
)
from firepanda.kernel.zoneinfo import load_zone, parse_zone


comptime NULL_ROW = Int64.MIN
"""A row the column builder leaves null."""


def column(
    values: List[Int64], unit: TimeUnit, zone: String
) raises -> AnyArray:
    """Builds a timestamp column, with `NULL_ROW` meaning a null.

    Args:
        values: The counts since the epoch.
        unit: What they are counts of.
        zone: The clock they are read against, empty for none.

    Returns:
        The column.
    """
    var out = Array[DType.int64](len(values))
    for i in range(len(values)):
        if values[i] == NULL_ROW:
            out.set_null(i)
        else:
            out.set_valid(i, values[i])
    return AnyArray(
        out^.into_data(), LogicalType.timestamp(unit, TimeZone(zone))
    )


def reading(utc: Int64, zone: String) raises -> Int64:
    """Returns what a zone's clock showed at one instant, in seconds.

    Args:
        utc: The instant, in seconds since the epoch.
        zone: The zone.

    Returns:
        The naive reading, in seconds.
    """
    var local = temporal_tz_localize_none(column([utc], TimeUnit.SECOND, zone))
    return local.as_typed_view[DType.int64]()[0]


def test_the_offset_follows_the_season() raises:
    """New York is five hours behind in January and four in July."""
    assert_equal(
        reading(1704110400, "America/New_York") - 1704110400, -18000, "winter"
    )
    assert_equal(
        reading(1719835200, "America/New_York") - 1719835200, -14400, "summer"
    )


def test_the_southern_summer_spans_new_year() raises:
    """Sydney is on daylight saving time in January and off it in July, the
    other way round from New York, so its footer rule starts in October and
    ends in April of the next year."""
    assert_equal(
        reading(1704067200, "Australia/Sydney") - 1704067200, 39600, "January"
    )
    assert_equal(
        reading(1719792000, "Australia/Sydney") - 1719792000, 36000, "July"
    )


def test_a_half_hour_zone_keeps_its_half_hour() raises:
    """Lord Howe Island moves its clock by thirty minutes rather than sixty."""
    assert_equal(
        reading(1717200000, "Australia/Lord_Howe") - 1717200000,
        37800,
        "ten and a half hours ahead in its winter",
    )
    assert_equal(
        reading(1704067200, "Australia/Lord_Howe") - 1704067200,
        39600,
        "and eleven in its summer",
    )


def test_before_the_first_transition_is_local_mean_time() raises:
    """In 1800 New York kept its own solar time, four hours fifty six minutes
    and two seconds behind Greenwich, which is the first type in the file."""
    assert_equal(
        reading(-5364619200, "America/New_York") - -5364619200,
        -(4 * 3600 + 56 * 60 + 2),
    )


def test_past_the_table_the_footer_rule_takes_over() raises:
    """A file's table stops somewhere around 2037, and every instant after that
    is answered from the rule in its footer, folded back by whole four hundred
    year cycles once it is past the part written out. July 2400 is well past
    both."""
    assert_equal(
        reading(13585233600, "America/New_York") - 13585233600,
        -14400,
        "daylight saving time in July 2400",
    )


def test_a_skipped_reading_is_refused_in_the_words_pandas_uses() raises:
    """Half past two on 10 March 2024 never happened in New York."""
    with assert_raises(
        contains=(
            "2024-03-10 02:30:00 is a nonexistent time due to daylight savings"
            " time. Try using the 'nonexistent' argument."
        )
    ):
        _ = temporal_tz_localize(
            column([Int64(1710037800)], TimeUnit.SECOND, ""),
            "America/New_York",
        )


def test_a_repeated_reading_is_refused_in_the_words_pandas_uses() raises:
    """Half past one on 3 November 2024 happened twice in New York."""
    with assert_raises(
        contains=(
            "Cannot infer dst time from 2024-11-03 01:30:00, try using the"
            " 'ambiguous' argument"
        )
    ):
        _ = temporal_tz_localize(
            column([Int64(1730597400)], TimeUnit.SECOND, ""),
            "America/New_York",
        )


def test_a_repeated_reading_past_the_table_is_refused_too() raises:
    """The fall back in 2124 comes from the footer rule, not the table."""
    with assert_raises(
        contains="Cannot infer dst time from 2124-11-05 01:30:00"
    ):
        _ = temporal_tz_localize(
            column([Int64(4886461800 - 18000)], TimeUnit.SECOND, ""),
            "America/New_York",
        )


def test_the_message_names_the_first_row_that_fails() raises:
    """The morsels run in parallel, so the first to fail is not always the
    first row, and pandas names the first row."""
    var rows = List[Int64]()
    for _ in range(200_000):
        rows.append(1704110400)
    rows.append(1710037800)
    rows.append(1730597400)
    for _ in range(200_000):
        rows.append(1730597400)
    with assert_raises(contains="2024-03-10 02:30:00 is a nonexistent time"):
        _ = temporal_tz_localize(
            column(rows, TimeUnit.SECOND, ""), "America/New_York"
        )


def test_a_null_row_is_never_asked_about() raises:
    """What sits under a null is not a reading, so it is never a skipped one."""
    var placed = temporal_tz_localize(
        column([NULL_ROW, 1704110400], TimeUnit.SECOND, ""), "America/New_York"
    )
    ref view = placed.as_typed_view[DType.int64]()
    assert_true(not view.is_valid(0), "the null stays null")
    assert_equal(view[1], 1704110400 + 18000, "and the reading is placed")


def test_localising_and_reading_back_is_the_identity() raises:
    """Every reading a clock shows once goes there and back unchanged, at
    nanoseconds as well as seconds, including the minute either side of both
    of 2024's changes."""
    var rows: List[Int64] = [
        1704110400,
        1710037800 - 1860,
        1710037800 + 1800,
        1730597400 - 3600,
        1730597400 + 3600,
    ]
    var nanos = List[Int64]()
    for i in range(len(rows)):
        nanos.append(rows[i] * 1_000_000_000 + 123)
    var placed = temporal_tz_localize(
        column(nanos, TimeUnit.NANO, ""), "America/New_York"
    )
    var back = temporal_tz_localize_none(placed)
    ref view = back.as_typed_view[DType.int64]()
    for i in range(len(nanos)):
        assert_equal(view[i], nanos[i], String("row ", i))


def test_a_name_the_database_does_not_hold_is_refused() raises:
    """pandas raises `ZoneInfoNotFoundError` with this sentence, and a name
    that climbs out of the database directory is refused before it is read."""
    var utc = column([Int64(0)], TimeUnit.SECOND, "UTC")
    with assert_raises(contains="No time zone found with key Nowhere/Land"):
        _ = temporal_tz_convert(utc, "Nowhere/Land")
    with assert_raises(contains="No time zone found with key"):
        _ = temporal_tz_convert(utc, "../../etc/passwd")
    with assert_raises(contains="No time zone found with key"):
        _ = load_zone("America")


def test_a_file_that_is_not_tzif_is_refused() raises:
    """The magic number is checked before anything else is read."""
    var junk: List[UInt8] = [0x50, 0x4B, 0x03, 0x04]
    for _ in range(60):
        junk.append(0)
    with assert_raises(contains="TZif"):
        _ = parse_zone(junk^)


def test_etc_zones_carry_their_inverted_sign() raises:
    """`Etc/GMT+5` is five hours behind UTC, the opposite of what its name
    reads as, which is why the fixed offset reader leaves it to the database."""
    assert_equal(reading(0, "Etc/GMT+5"), -18000)


comptime SKIPPED = Int64(1710037800)
"""Half past two on 10 March 2024, which New York skipped."""

comptime REPEATED = Int64(1730597400)
"""Half past one on 3 November 2024, which New York showed twice."""


def placed_under(
    value: Int64, zone: String, policy: ZonePolicy
) raises -> AnyArray:
    """Localises one reading in seconds under a policy.

    Args:
        value: The reading.
        zone: The zone.
        policy: The policy.

    Returns:
        The one row column.
    """
    return temporal_tz_localize(
        column([value], TimeUnit.SECOND, ""), zone, policy
    )


def instant(placed: AnyArray) raises -> Int64:
    """Returns the one row of a column, which the caller knows is present.

    Args:
        placed: The column.

    Returns:
        Its first row.
    """
    return placed.as_typed_view[DType.int64]()[0]


def test_a_skipped_reading_can_be_null() raises:
    var placed = placed_under(
        SKIPPED,
        "America/New_York",
        ZonePolicy(ON_A_FOLD_RAISE, IN_A_GAP_NULL, 0),
    )
    assert_true(not placed.as_typed_view[DType.int64]().is_valid(0))


def test_a_skipped_reading_shifts_to_the_hours_either_side() raises:
    """Forward lands on three o'clock daylight time and back on the last
    second of standard time, both measured against pandas."""
    var forward = ZonePolicy(ON_A_FOLD_RAISE, IN_A_GAP_FORWARD, 0)
    var backward = ZonePolicy(ON_A_FOLD_RAISE, IN_A_GAP_BACKWARD, 0)
    assert_equal(
        instant(placed_under(SKIPPED, "America/New_York", forward)),
        1710054000,
    )
    assert_equal(
        instant(placed_under(SKIPPED, "America/New_York", backward)),
        1710053999,
    )


def test_a_half_hour_gap_still_shifts_by_the_hour() raises:
    """Lord Howe skips from two to half past, and pandas shifts forward to
    three and back to one second before two, so this does too."""
    var forward = ZonePolicy(ON_A_FOLD_RAISE, IN_A_GAP_FORWARD, 0)
    var backward = ZonePolicy(ON_A_FOLD_RAISE, IN_A_GAP_BACKWARD, 0)
    assert_equal(
        instant(placed_under(1728180900, "Australia/Lord_Howe", forward)),
        1728144000,
    )
    assert_equal(
        instant(placed_under(1728180900, "Australia/Lord_Howe", backward)),
        1728142199,
    )


def test_a_skipped_reading_shifts_by_a_timedelta() raises:
    """An hour on from half past two is half past three daylight time."""
    var hour = ZonePolicy(ON_A_FOLD_RAISE, IN_A_GAP_SHIFT, 3_600_000_000_000)
    assert_equal(
        instant(placed_under(SKIPPED, "America/New_York", hour)), 1710055800
    )


def test_a_shift_that_stays_in_the_hour_is_refused() raises:
    var twenty = ZonePolicy(ON_A_FOLD_RAISE, IN_A_GAP_SHIFT, 1_200_000_000_000)
    with assert_raises(
        contains="The provided timedelta will relocalize on a nonexistent time"
    ):
        _ = placed_under(SKIPPED, "America/New_York", twenty)


def test_a_repeated_reading_takes_the_side_it_is_told() raises:
    """True is the first instant, still on daylight time, and False the
    second, back on standard time."""
    var earlier = ZonePolicy(ON_A_FOLD_EARLIER, IN_A_GAP_RAISE, 0)
    var later = ZonePolicy(ON_A_FOLD_LATER, IN_A_GAP_RAISE, 0)
    var null = ZonePolicy(ON_A_FOLD_NULL, IN_A_GAP_RAISE, 0)
    assert_equal(
        instant(placed_under(REPEATED, "America/New_York", earlier)),
        1730611800,
    )
    assert_equal(
        instant(placed_under(REPEATED, "America/New_York", later)), 1730615400
    )
    assert_true(
        not placed_under(REPEATED, "America/New_York", null)
        .as_typed_view[DType.int64]()
        .is_valid(0)
    )


def test_a_policy_for_one_leaves_the_other_raising() raises:
    """Telling it what to do with a fold says nothing about a gap."""
    with assert_raises(contains="is a nonexistent time"):
        _ = placed_under(
            SKIPPED,
            "America/New_York",
            ZonePolicy(ON_A_FOLD_NULL, IN_A_GAP_RAISE, 0),
        )


def test_the_nulls_a_policy_makes_land_on_their_own_rows() raises:
    """Across several morsels, every skipped row is null and every other one
    is placed, so no morsel wrote another's part of the bitmap."""
    var rows = List[Int64]()
    for i in range(300_000):
        rows.append(SKIPPED if i % 3 == 0 else Int64(1704110400))
    var placed = temporal_tz_localize(
        column(rows, TimeUnit.SECOND, ""),
        "America/New_York",
        ZonePolicy(ON_A_FOLD_RAISE, IN_A_GAP_NULL, 0),
    )
    ref view = placed.as_typed_view[DType.int64]()
    for i in range(300_000):
        if i % 3 == 0:
            assert_true(not view.is_valid(i), "a skipped row is null")
        else:
            assert_equal(view[i], 1704110400 + 18000, "and the rest placed")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
