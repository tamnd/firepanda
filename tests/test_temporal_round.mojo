"""Tests for rounding a timestamp column to a frequency, and for restating it.

Every expected value in this file was read off a running pandas 3.0.3 rather than
worked out here. The rows are seconds either side of the epoch, chosen so that
each of the three modes has one row it alone gets right, and the whole table was
run through `dt.floor('h')`, `dt.ceil('h')` and `dt.round('h')` at all four
resolutions before any of it was written down.

Four things in here are not what a reader would guess, and each of them is why
the row it belongs to is here.

The first is that `dt.round` breaks a tie towards the even multiple. Half past
midnight rounds back to midnight and half past one rounds forward to two, which
is the same rule the numeric round uses and is not the rule anyone has ever
wanted from a clock. Both of those rows are here.

The second is that a frequency finer than the column's own unit is not an error.
`dt.floor('ms')` on a column of whole seconds answers the column unchanged, for
the same reason that flooring an integer to the nearest integer does nothing, and
so does a count of zero. Both are here.

The third is that a negative frequency is accepted, and that pandas' own answer
for it does not follow from its answer for the positive one. `dt.floor('-1h')`
is the floored quotient by a negative period, which moves an instant forward, and
`dt.round('-1h')` sends the epoch itself to the hour before it. Reproducing that
takes one rule rather than two, and the negative rows are here to hold the code
to the rule that produces both.

The fourth is that going down in resolution rounds down rather than towards zero,
so a nanosecond column restated in seconds keeps its 1969 rows in 1969. That is
the same hazard the calendar fields have and it is here again because it is a
different division.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.frame.series import Series
from firepanda.kernel.scalar import rescale_scalar, round_to_period_scalar
from firepanda.kernel.temporal import (
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    ROUND_UP,
    frequency_period,
    round_to_period,
    temporal_as_unit,
    temporal_round,
    unit_named,
)

comptime NULL_ROW = Int64.MIN
"""The marker `stamps` reads as a missing row rather than as an instant, chosen
for the same reason as in `test_temporal.mojo`: -1 is a real row this file cares
about."""


def hour_rows() -> List[Int64]:
    """Returns the seconds every table in this file is indexed by.

    Either side of the epoch, at and around the hour and the half hour. Every
    one of them was put through pandas and the answers below are what came back.

    Returns:
        Twelve counts of seconds since the epoch.
    """
    return [
        Int64(-5401),
        -3600,
        -1800,
        -1,
        0,
        1,
        1800,
        3600,
        5400,
        5401,
        7200,
        9000,
    ]


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


def hours(unit: TimeUnit) raises -> AnyArray:
    """Builds the shared row list at one of the four resolutions.

    Args:
        unit: The resolution.

    Returns:
        A column holding the rows of `hour_rows`, counted in that unit.
    """
    var rate = unit.per_second()
    var rows = hour_rows()
    var scaled = List[Int64](capacity=len(rows))
    for i in range(len(rows)):
        scaled.append(rows[i] * rate)
    return stamps(scaled^, unit)


def instants(col: AnyArray) raises -> List[Int64]:
    """Reads a timestamp column back as plain integers.

    Args:
        col: The column.

    Returns:
        One integer per row, with a null row reading as whatever is stored.
    """
    ref view = col.as_typed_view[DType.int64]()
    var out = List[Int64](capacity=len(view))
    for i in range(len(view)):
        out.append(view[i])
    return out^


def seconds(col: AnyArray) raises -> List[Int64]:
    """Reads a timestamp column back as whole seconds.

    Args:
        col: The column, at any of the four resolutions.

    Returns:
        One count of seconds per row, so that the same expected list can be
        compared against all four.
    """
    var rate = col.type.unit.per_second()
    var raw = instants(col)
    var out = List[Int64](capacity=len(raw))
    for i in range(len(raw)):
        out.append(raw[i] // rate)
    return out^


def test_flooring_moves_an_instant_back_and_never_forward() raises:
    """The twelve rows through `dt.floor('h')`, as pandas answers them.

    The three rows below the epoch are the ones that matter. Flooring rounds
    down rather than towards zero, so -1, which is the last second of 1969,
    lands on the hour before it and not on the epoch."""
    var got = seconds(temporal_round(hours(TimeUnit.SECOND), "h", ROUND_DOWN))
    var want = [
        Int64(-7200),
        -3600,
        -3600,
        -3600,
        0,
        0,
        0,
        3600,
        3600,
        3600,
        7200,
        7200,
    ]
    for i in range(len(want)):
        assert_equal(got[i], want[i], String("row ", i))


def test_ceiling_leaves_a_row_that_is_already_whole_alone() raises:
    """The same rows through `dt.ceil('h')`.

    The rows on the hour, -3600 and 0 and 3600 and 7200, come back unchanged.
    An implementation that adds a period first and floors afterwards moves all
    four of them on by an hour and passes every other row here."""
    var got = seconds(temporal_round(hours(TimeUnit.SECOND), "h", ROUND_UP))
    var want = [
        Int64(-3600),
        -3600,
        0,
        0,
        0,
        3600,
        3600,
        3600,
        7200,
        7200,
        7200,
        10800,
    ]
    for i in range(len(want)):
        assert_equal(got[i], want[i], String("row ", i))


def test_rounding_settles_a_tie_on_the_even_hour() raises:
    """The same rows through `dt.round('h')`.

    Four of them are exactly on the half hour and each goes to whichever
    neighbour is an even number of hours from the epoch. -1800 and 1800 both go
    to 0, 5400 goes forward to 7200 and 9000 goes back to 7200, so a rule that
    always rounds a tie up passes two of the four and a rule that always rounds
    a tie away from zero passes three."""
    var got = seconds(
        temporal_round(hours(TimeUnit.SECOND), "h", ROUND_HALF_EVEN)
    )
    var want = [
        Int64(-7200),
        -3600,
        0,
        0,
        0,
        0,
        0,
        3600,
        7200,
        7200,
        7200,
        7200,
    ]
    for i in range(len(want)):
        assert_equal(got[i], want[i], String("row ", i))


def test_all_four_resolutions_round_the_same_instant_the_same_way() raises:
    """The same instants at four resolutions give the same answer in seconds.

    The period is computed in the column's own unit, so a millisecond column
    rounds to 3600000 and a nanosecond one to 3600000000000. Reading all four
    back as seconds is how the same expected list checks all four."""
    for unit in [
        TimeUnit.SECOND,
        TimeUnit.MILLI,
        TimeUnit.MICRO,
        TimeUnit.NANO,
    ]:
        var col = hours(unit)
        var floored = temporal_round(col, "h", ROUND_DOWN)
        assert_equal(
            String(floored.type),
            String("datetime64[", unit, "]"),
            "the type is kept",
        )
        assert_equal(seconds(floored)[0], -7200, String("floor at ", unit))
        assert_equal(
            seconds(temporal_round(col, "h", ROUND_UP))[0],
            -3600,
            String("ceil at ", unit),
        )
        assert_equal(
            seconds(temporal_round(col, "h", ROUND_HALF_EVEN))[2],
            0,
            String("round at ", unit),
        )


def test_a_frequency_finer_than_the_column_leaves_it_alone() raises:
    """A millisecond does not divide a column of whole seconds into anything.

    pandas answers the column unchanged rather than raising, which is the same
    thing a period of zero has to mean, and dividing by that period would be a
    trap rather than an answer."""
    var col = hours(TimeUnit.SECOND)
    for freq in ["ms", "us", "ns", "0h", "0min"]:
        var got = seconds(temporal_round(col, freq, ROUND_DOWN))
        assert_equal(got[0], -5401, String("floor by ", freq))
        assert_equal(
            seconds(temporal_round(col, freq, ROUND_UP))[0],
            -5401,
            String("ceil by ", freq),
        )


def test_a_count_in_front_of_the_alias_multiplies_the_period() raises:
    """`15min` and `2h` and `1D` are all frequencies and all mean what they say.
    """
    var stamp = stamps([Int64(5401)], TimeUnit.SECOND)
    assert_equal(
        seconds(temporal_round(stamp, "15min", ROUND_DOWN))[0], 5400, "15min"
    )
    assert_equal(seconds(temporal_round(stamp, "2h", ROUND_DOWN))[0], 0, "2h")
    assert_equal(seconds(temporal_round(stamp, "1D", ROUND_DOWN))[0], 0, "1D")
    assert_equal(
        seconds(temporal_round(stamp, "90s", ROUND_DOWN))[0], 5400, "90s"
    )
    assert_equal(
        seconds(temporal_round(stamp, "3600s", ROUND_DOWN))[0], 3600, "3600s"
    )


def test_a_frequency_with_a_decimal_point_is_exact() raises:
    """`1.5h` is 5400 seconds and not whatever 1.5 times 3.6e12 rounds to.

    The count is carried as fifteen over ten rather than as a float, so this
    holds however many digits are involved. `2500ms` on a column of seconds is
    two seconds, because the last division rounds down into the column's unit,
    and `1500ms` is one second, which makes it a no op."""
    var stamp = stamps([Int64(5401)], TimeUnit.SECOND)
    assert_equal(
        seconds(temporal_round(stamp, "1.5h", ROUND_DOWN))[0], 5400, "1.5h"
    )
    assert_equal(
        seconds(temporal_round(stamp, "2500ms", ROUND_DOWN))[0], 5400, "2500ms"
    )
    assert_equal(
        seconds(temporal_round(stamp, "1500ms", ROUND_DOWN))[0], 5401, "1500ms"
    )


def test_spaces_at_the_ends_of_a_frequency_are_trimmed() raises:
    """Space around a frequency is trimmed off, as pandas trims it."""
    var stamp = stamps([Int64(5401)], TimeUnit.SECOND)
    for freq in [" h", "h ", "  h  "]:
        assert_equal(
            seconds(temporal_round(stamp, freq, ROUND_DOWN))[0],
            3600,
            String("floor by '", freq, "'"),
        )


def test_a_negative_frequency_rounds_the_way_pandas_rounds_it() raises:
    """`-1h` is a period of minus one hour and pandas accepts it.

    Flooring by it is the floored quotient by a negative number, which moves an
    instant forward rather than back. Rounding by it sends the epoch to the hour
    before it, which does not follow from the floor at all. Both fall out of the
    same two rules the positive case uses, which is the reason this test is
    here: it is the one that fails if the tie is compared against half the size
    of the period rather than against the signed period itself."""
    var col = hours(TimeUnit.SECOND)
    var floored = seconds(temporal_round(col, "-1h", ROUND_DOWN))
    var want_floor = [
        Int64(-3600),
        -3600,
        0,
        0,
        0,
        3600,
        3600,
        3600,
        7200,
        7200,
        7200,
        10800,
    ]
    for i in range(len(want_floor)):
        assert_equal(floored[i], want_floor[i], String("floor row ", i))

    var rounded = seconds(temporal_round(col, "-1h", ROUND_HALF_EVEN))
    var want_round = [
        Int64(-3600),
        -7200,
        0,
        -3600,
        -3600,
        3600,
        0,
        0,
        7200,
        3600,
        3600,
        7200,
    ]
    for i in range(len(want_round)):
        assert_equal(rounded[i], want_round[i], String("round row ", i))


def test_a_frequency_that_has_no_fixed_length_is_refused() raises:
    """A week and a month end are frequencies pandas will not round to either.

    How long a month is depends on which month it is, so there is no period to
    divide by, and pandas raises rather than picking an average. So does this.
    """
    var stamp = stamps([Int64(0)], TimeUnit.SECOND)
    for freq in ["W", "ME", "QE", "YS", "B"]:
        with assert_raises(contains="is not a fixed frequency"):
            _ = temporal_round(stamp, freq, ROUND_DOWN)


def test_the_spellings_pandas_dropped_are_refused_too() raises:
    """`T` for a minute and `H` for an hour are gone from pandas 3 and raise.

    Accepting them here would be a library that takes input pandas rejects,
    which is a different kind of wrong from rejecting input pandas takes but is
    still a difference."""
    var stamp = stamps([Int64(0)], TimeUnit.SECOND)
    for freq in ["T", "H", "S", "L", "U", "N", "hour", "minutes", "", "hh"]:
        with assert_raises(contains="is not a fixed frequency"):
            _ = temporal_round(stamp, freq, ROUND_DOWN)


def test_a_null_row_stays_null_and_is_not_rounded_into_a_value() raises:
    """The validity comes through untouched and nothing reads the stored value.
    """
    var col = stamps([Int64(5401), NULL_ROW, 1], TimeUnit.SECOND)
    var got = temporal_round(col, "h", ROUND_DOWN)
    ref view = got.as_typed_view[DType.int64]()
    assert_true(view.is_valid(0), "row 0 is there")
    assert_true(not view.is_valid(1), "row 1 is not")
    assert_true(view.is_valid(2), "row 2 is there")
    assert_equal(view[0], 3600, "row 0")
    assert_equal(view[2], 0, "row 2")


def test_a_zoned_column_is_refused_rather_than_rounded_in_utc() raises:
    """The local reading is what pandas rounds and the stored instants are UTC.

    In a zone whose offset is not a whole number of hours those are two
    different answers, so answering from the stored instants would be wrong
    rather than approximate."""
    var col = Array[DType.int64](1)
    col.set_valid(0, 5401)
    var zoned = AnyArray(
        col^.into_data(),
        LogicalType.timestamp(TimeUnit.SECOND, TimeZone("Australia/Lord_Howe")),
    )
    with assert_raises(contains="time zone database"):
        _ = temporal_round(zoned, "h", ROUND_DOWN)


def test_a_column_that_is_not_a_timestamp_has_no_frequency_in_it() raises:
    """An int64 column carries no unit, so there is nothing to divide by."""
    var col = Array[DType.int64](1)
    col.set_valid(0, 5401)
    var plain = AnyArray(col^)
    with assert_raises(contains="only defined on a timestamp column"):
        _ = temporal_round(plain, "h", ROUND_DOWN)


def test_a_mode_that_is_not_a_mode_is_refused() raises:
    """The three modes are the whole list and a fourth is a caller bug."""
    var stamp = stamps([Int64(0)], TimeUnit.SECOND)
    with assert_raises(contains="is not a rounding mode"):
        _ = temporal_round(stamp, "h", 7)


def test_the_period_is_counted_in_the_columns_own_unit() raises:
    """One hour is 3600 in a second column and 3600000000000 in a nanosecond one.
    """
    assert_equal(
        frequency_period("h", LogicalType.timestamp(TimeUnit.SECOND)),
        3_600,
        "seconds",
    )
    assert_equal(
        frequency_period("h", LogicalType.timestamp(TimeUnit.MILLI)),
        3_600_000,
        "milliseconds",
    )
    assert_equal(
        frequency_period("us", LogicalType.timestamp(TimeUnit.NANO)),
        1_000,
        "nanoseconds",
    )
    assert_equal(
        frequency_period("us", LogicalType.timestamp(TimeUnit.SECOND)),
        0,
        "finer than the column",
    )


def test_going_down_in_precision_rounds_down_and_not_towards_zero() raises:
    """A nanosecond column restated in seconds keeps its 1969 rows in 1969.

    Half a second before the epoch is -500000000 nanoseconds, which is -1 second
    rounded down and 0 rounded towards zero, and the second of those puts the
    row in the wrong year. pandas answers -1."""
    var col = stamps(
        [
            Int64(-1_500_000_000),
            -1_000_000_000,
            -500_000_000,
            -1,
            0,
            1,
            1_500_000_000,
        ],
        TimeUnit.NANO,
    )
    var got = instants(temporal_as_unit(col, TimeUnit.SECOND))
    var want = [Int64(-2), -1, -1, -1, 0, 0, 1]
    for i in range(len(want)):
        assert_equal(got[i], want[i], String("row ", i))


def test_going_up_in_precision_multiplies_and_does_not_recover() raises:
    """Seconds restated in nanoseconds is a multiply and nothing more.

    What an earlier trip down removed stays removed, which is why `as_unit` down
    and then up is not the identity and why the case list has both directions in
    it."""
    var col = stamps([Int64(-2), -1, 0, 1], TimeUnit.SECOND)
    var up = temporal_as_unit(col, TimeUnit.NANO)
    assert_equal(String(up.type), "datetime64[ns]", "the type")
    var got = instants(up)
    var want = [Int64(-2_000_000_000), -1_000_000_000, 0, 1_000_000_000]
    for i in range(len(want)):
        assert_equal(got[i], want[i], String("row ", i))
    var back = instants(temporal_as_unit(up, TimeUnit.SECOND))
    for i in range(len(want)):
        assert_equal(back[i], instants(col)[i], String("back row ", i))


def test_restating_at_the_resolution_it_already_has_is_a_copy() raises:
    """The values and the type both come through unchanged."""
    var col = stamps([Int64(-1), 0, 1], TimeUnit.MILLI)
    var got = temporal_as_unit(col, TimeUnit.MILLI)
    assert_equal(String(got.type), "datetime64[ms]", "the type")
    assert_equal(instants(got)[0], -1, "row 0")


def test_going_up_out_of_range_is_refused_rather_than_wrapped() raises:
    """An instant that does not fit after the multiply raises here as it does
    there.

    The year 2300 is a perfectly good second column and is outside what an int64
    of nanoseconds can hold, which is the whole reason pandas 2 moved the
    resolution onto the dtype. Answering a wrapped number would be a date in the
    wrong millennium presented as though it were right."""
    var col = stamps([Int64(0), 100_000_000_000], TimeUnit.SECOND)
    with assert_raises(contains="does not fit in an int64"):
        _ = temporal_as_unit(col, TimeUnit.NANO)


def test_a_null_row_out_of_range_does_not_refuse_the_column() raises:
    """The out of range check reads the rows that are really there.

    A null row holds whatever the file that produced it left in the buffer, and
    a column whose only large value is under a cleared validity bit converts
    perfectly well. The cheap vector scan flags this column and the row by row
    check then clears it, which is the reason there are two of them."""
    var col = Array[DType.int64](2)
    col.set_valid(0, 0)
    col[1] = 100_000_000_000
    col.set_null(1)
    var stamped = AnyArray(
        col^.into_data(), LogicalType.timestamp(TimeUnit.SECOND)
    )
    var got = temporal_as_unit(stamped, TimeUnit.NANO)
    assert_equal(String(got.type), "datetime64[ns]", "the type")
    ref view = got.as_typed_view[DType.int64]()
    assert_true(view.is_valid(0), "row 0 is there")
    assert_true(not view.is_valid(1), "row 1 is not")


def test_a_zoned_column_keeps_its_zone_through_a_resolution_change() raises:
    """Which second an instant is does not depend on the clock reading it.

    So `as_unit` is the one thing on the accessor that a zoned column is allowed
    to do, and the zone has to survive it."""
    var col = Array[DType.int64](1)
    col.set_valid(0, 5401)
    var zoned = AnyArray(
        col^.into_data(),
        LogicalType.timestamp(TimeUnit.SECOND, TimeZone("America/New_York")),
    )
    var got = temporal_as_unit(zoned, TimeUnit.MILLI)
    assert_equal(
        String(got.type), "datetime64[ms, America/New_York]", "the type"
    )
    assert_equal(instants(got)[0], 5_401_000, "the value")


def test_a_resolution_that_is_not_one_of_the_four_is_refused() raises:
    """Arrow has four and a name outside them is a caller bug, not a rounding.
    """
    with assert_raises(contains="is not a resolution"):
        _ = unit_named("m")
    with assert_raises(contains="is not a resolution"):
        _ = unit_named("nanoseconds")


def test_the_kernel_agrees_with_the_twin_on_all_three_modes() raises:
    """The vector kernel against the one row at a time twin, across a tail.

    Nineteen rows is not a whole number of vectors on any width this runs on, so
    a kernel that handles its tail wrongly disagrees here. The twin divides with
    a truncating divide and a correction rather than with `//`, so agreement
    between the two is evidence about the answer rather than about the
    operator."""
    var col = Array[DType.int64](19)
    for i in range(19):
        col.set_valid(i, Int64(i * 907) - 8_000)
    col.set_null(5)
    col.set_null(11)

    for period in [Int64(3_600), 60, 907, -3_600, 1]:
        var down = round_to_period[ROUND_DOWN](col, period)
        var down_twin = round_to_period_scalar[ROUND_DOWN](col, period)
        var up = round_to_period[ROUND_UP](col, period)
        var up_twin = round_to_period_scalar[ROUND_UP](col, period)
        var near = round_to_period[ROUND_HALF_EVEN](col, period)
        var near_twin = round_to_period_scalar[ROUND_HALF_EVEN](col, period)
        for i in range(19):
            assert_equal(
                down.is_valid(i), down_twin.is_valid(i), String("down bit ", i)
            )
            if down.is_valid(i):
                assert_equal(down[i], down_twin[i], String("down row ", i))
                assert_equal(up[i], up_twin[i], String("up row ", i))
                assert_equal(near[i], near_twin[i], String("near row ", i))


def test_the_rescale_kernel_agrees_with_its_twin_both_ways() raises:
    """The same comparison for the multiply and the divide."""
    var col = Array[DType.int64](19)
    for i in range(19):
        col.set_valid(i, Int64(i * 907) - 8_000)
    col.set_null(3)

    var down = rescale_scalar[False](col, 1_000)
    var down_fast = temporal_as_unit(
        AnyArray(
            Array[DType.int64](copy=col).into_data(),
            LogicalType.timestamp(TimeUnit.MILLI),
        ),
        TimeUnit.SECOND,
    )
    ref down_view = down_fast.as_typed_view[DType.int64]()
    for i in range(19):
        assert_equal(down.is_valid(i), down_view.is_valid(i), String("bit ", i))
        if down.is_valid(i):
            assert_equal(down[i], down_view[i], String("down row ", i))

    var up = rescale_scalar[True](col, 1_000)
    var up_fast = temporal_as_unit(
        AnyArray(
            Array[DType.int64](copy=col).into_data(),
            LogicalType.timestamp(TimeUnit.SECOND),
        ),
        TimeUnit.MILLI,
    )
    ref up_view = up_fast.as_typed_view[DType.int64]()
    for i in range(19):
        if up.is_valid(i):
            assert_equal(up[i], up_view[i], String("up row ", i))


def test_a_series_rounds_and_keeps_its_name_and_its_labels() raises:
    """The three modes reach the frame layer under the names pandas gives them.
    """
    var col = stamps([Int64(5401), 1800], TimeUnit.SECOND)
    var s = Series("when", col^)
    assert_equal(s.dt_floor("h").name, "when", "the name")
    assert_equal(len(s.dt_ceil("h")), 2, "the height")
    assert_equal(
        s.dt_floor("h").values.as_typed_view[DType.int64]()[0], 3600, "floor"
    )
    assert_equal(
        s.dt_ceil("h").values.as_typed_view[DType.int64]()[0], 7200, "ceil"
    )
    assert_equal(
        s.dt_round("h").values.as_typed_view[DType.int64]()[1], 0, "round"
    )


def test_a_series_names_a_resolution_as_a_string() raises:
    """`as_unit` takes the name pandas takes rather than an enumerator."""
    var col = stamps([Int64(5401)], TimeUnit.SECOND)
    var s = Series("when", col^)
    var got = s.dt_as_unit("ms")
    assert_equal(String(got.values.type), "datetime64[ms]", "the type")
    assert_equal(
        got.values.as_typed_view[DType.int64]()[0], 5_401_000, "the value"
    )
    with assert_raises(contains="is not a resolution"):
        _ = s.dt_as_unit("minutes")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
