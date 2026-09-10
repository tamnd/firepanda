"""Tests for elapsed times: the arithmetic, the reductions and the two readers.

Every expected value here was read off a running pandas 3.0.3. That matters more
in this file than in most, because three of the rules below are ones nobody
would arrive at by reasoning and two of them look like bugs until you see where
they come from.

The first rule is that an expression between two temporal operands answers at
the finer of the two resolutions, and it is the same rule for all three shapes
of the expression. Second minus second is a second, second plus a millisecond
span is a millisecond, and a second span plus a microsecond span is a
microsecond span. There is no widening to a common unit first, because there is
no common unit that both operands could be cast to without changing one of them.

The second rule is that the resolution of a constant depends on how the constant
was spelled and not on how long it is. `pd.Timedelta(90, unit='s')` is a second
and `pd.Timedelta(hours=1)` is a microsecond, so adding the first to a second
column leaves it a second column and adding the second turns it into a
microsecond one. Same accessor, same amount of time, different answer type. The
tests below pin both spellings.

The third rule is the mean, which pandas computes in float64 and then truncates
toward zero at both signs. So the mean of one and two seconds is one second and
the mean of minus two and minus one is minus one second, neither of which is the
nearest answer and neither of which is a floor. Above 2**53 counts in the
column's own unit the float has run out of mantissa and the mean of two equal
durations stops being either of them. firepanda reproduces that rather than
fixing it, for the reason `binary.mojo` gives at the top of its own file: being
wrong the way people already expect costs less than being right in a way that
makes an answer depend on how large the values happened to be.

The sum is the other reduction with a rule worth writing down. The sum of an
empty duration column and the sum of an all null one are both a zero length
span rather than a missing one, which is the same answer pandas gives for the
sum of an empty numeric column and is arrived at the same way. The mean of
either is missing.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit, finer_unit
from firepanda.frame.series import Series
from firepanda.kernel.binary import BinaryOp, binary_any, binary_value_any
from firepanda.kernel.group import AggKind
from firepanda.kernel.reduce import reduce_any
from firepanda.kernel.temporal import (
    temporal_duration_days,
    temporal_to_duration,
    temporal_total_seconds,
)
from firepanda.kernel.unary import UnaryOp, unary_any


def counts(values: List[Int64]) -> Array[DType.int64]:
    """Builds an int64 column out of a list.

    Args:
        values: The counts.

    Returns:
        The column, with every row present.
    """
    var out = Array[DType.int64](len(values))
    for i in range(len(values)):
        out[i] = values[i]
    return out^


def span(values: List[Int64], unit: TimeUnit) -> AnyArray:
    """Builds a duration column.

    Args:
        values: The counts.
        unit: What they are counts of.

    Returns:
        The column.
    """
    return AnyArray(counts(values).into_data(), LogicalType.duration(unit))


def when(values: List[Int64], unit: TimeUnit) -> AnyArray:
    """Builds a naive timestamp column.

    Args:
        values: The counts since the epoch.
        unit: What they are counts of.

    Returns:
        The column.
    """
    return AnyArray(counts(values).into_data(), LogicalType.timestamp(unit))


def row(col: AnyArray, i: Int) raises -> Int64:
    """Reads one count out of a temporal or integer column.

    Args:
        col: The column.
        i: Which row.

    Returns:
        The stored count.
    """
    return col.as_typed_view[DType.int64]()[i]


def test_the_finer_unit_wins_whichever_side_it_is_on() raises:
    """The one rule behind every temporal expression, checked both ways round.
    """
    assert_true(
        finer_unit(TimeUnit.SECOND, TimeUnit.MICRO) == TimeUnit.MICRO,
        "the microsecond is finer than the second",
    )
    assert_true(
        finer_unit(TimeUnit.NANO, TimeUnit.MILLI) == TimeUnit.NANO,
        "and the order of the arguments does not matter",
    )
    assert_true(
        finer_unit(TimeUnit.MILLI, TimeUnit.MILLI) == TimeUnit.MILLI,
        "two of the same answer that one",
    )


def test_two_instants_subtract_to_an_elapsed_time() raises:
    """`s - s.iloc[0]` is the expression the conformance suite runs and it is
    the reason the duration type exists at all."""
    var a = when([Int64(0), 90, 3600], TimeUnit.SECOND)
    var b = when([Int64(0), 0, 0], TimeUnit.SECOND)
    var got = binary_any(a, b, BinaryOp.SUB)
    assert_equal(String(got.type), "timedelta64[s]", "the answer is a span")
    assert_equal(row(got, 1), 90, "ninety seconds apart")
    assert_equal(row(got, 2), 3600, "an hour apart")


def test_the_difference_of_two_instants_is_taken_at_the_finer_one() raises:
    """A second column and a millisecond column subtract in milliseconds, so
    the answer can hold a difference that is not a whole second."""
    var a = when([Int64(0), 1], TimeUnit.MILLI)
    var b = when([Int64(0), 0], TimeUnit.SECOND)
    var got = binary_any(a, b, BinaryOp.SUB)
    assert_equal(String(got.type), "timedelta64[ms]", "the finer unit wins")
    assert_equal(row(got, 1), 1, "one millisecond, which a second cannot hold")


def test_two_instants_cannot_be_added() raises:
    """The refusal pandas gives, because the sum of two points in time is not a
    point in time and is not a length of one either."""
    var a = when([Int64(0)], TimeUnit.SECOND)
    with assert_raises(contains="two points in time have a difference"):
        _ = binary_any(a, a, BinaryOp.ADD)


def test_an_instant_takes_a_span_and_stays_an_instant() raises:
    """Addition either way round and subtraction one way round, which is the
    whole of what a timestamp and a duration do together."""
    var t = when([Int64(0), 90], TimeUnit.SECOND)
    var d = span([Int64(1), 1], TimeUnit.SECOND)

    var later = binary_any(t, d, BinaryOp.ADD)
    assert_equal(String(later.type), "datetime64[s]", "still an instant")
    assert_equal(row(later, 1), 91, "a second later")

    var mirrored = binary_any(d, t, BinaryOp.ADD)
    assert_equal(String(mirrored.type), "datetime64[s]", "either way round")
    assert_equal(row(mirrored, 1), 91, "and the same answer")

    var earlier = binary_any(t, d, BinaryOp.SUB)
    assert_equal(row(earlier, 1), 89, "a second earlier")


def test_a_span_minus_an_instant_is_refused() raises:
    """The one asymmetry. Subtraction is the operation whose operands are not
    interchangeable and this is where that shows."""
    var t = when([Int64(0)], TimeUnit.SECOND)
    var d = span([Int64(1)], TimeUnit.SECOND)
    with assert_raises(contains="an elapsed time minus a point in time"):
        _ = binary_any(d, t, BinaryOp.SUB)


def test_a_timestamp_constant_carries_its_own_resolution() raises:
    """The rule that surprises people, taken straight from pandas: the type of
    `s + delta` depends on how the delta was written."""
    var t = when([Int64(0), 1], TimeUnit.SECOND)

    var coarse = binary_value_any(
        t, Value.duration(90, TimeUnit.SECOND), BinaryOp.ADD
    )
    assert_equal(String(coarse.type), "datetime64[s]", "a second stays seconds")
    assert_equal(row(coarse, 0), 90, "ninety seconds on")

    var fine = binary_value_any(
        t, Value.duration(3_600_000_000, TimeUnit.MICRO), BinaryOp.ADD
    )
    assert_equal(
        String(fine.type), "datetime64[us]", "a microsecond drags it finer"
    )
    assert_equal(row(fine, 0), 3_600_000_000, "an hour on, in microseconds")
    assert_equal(row(fine, 1), 3_601_000_000, "and the column came with it")


def test_a_number_is_not_a_length_of_time() raises:
    """An instant and a plain integer have no common type, which is what stops
    `s + 1` from meaning one of whatever the column happens to be stored in."""
    var t = when([Int64(0)], TimeUnit.SECOND)
    with assert_raises(contains="a point in time and a number"):
        _ = binary_value_any(t, Value(Int64(1)), BinaryOp.ADD)


def test_a_span_has_a_sign_and_an_instant_does_not() raises:
    """A duration takes `-s` and `abs(s)` in pandas and a datetime refuses both,
    and the reason is the same reason in both directions."""
    var d = span([Int64(-1), 2], TimeUnit.SECOND)

    var flipped = unary_any(d, UnaryOp.NEG)
    assert_equal(String(flipped.type), "timedelta64[s]", "still a span")
    assert_equal(row(flipped, 0), 1, "the sign turned over")

    var size = unary_any(d, UnaryOp.ABS)
    assert_equal(String(size.type), "timedelta64[s]", "a magnitude is a span")
    assert_equal(row(size, 0), 1, "one second either way")

    var t = when([Int64(0)], TimeUnit.SECOND)
    with assert_raises(contains="is not defined on"):
        _ = unary_any(t, UnaryOp.NEG)
    with assert_raises(contains="is not defined on"):
        _ = unary_any(d, UnaryOp.INVERT)


def test_the_reductions_keep_the_type_they_reduced() raises:
    """The whole point of the reduce change: a maximum of instants is an instant
    and not the int64 it is stored in."""
    var d = span([Int64(0), 1, -1, 86_400_000_000], TimeUnit.MICRO)

    var total = reduce_any(d, AggKind.SUM)
    assert_equal(String(total.type), "timedelta64[us]", "a total is a span")
    assert_equal(row(total, 0), 86_400_000_000, "the day, the rest cancelling")

    var least = reduce_any(d, AggKind.MIN)
    assert_equal(String(least.type), "timedelta64[us]", "so is a minimum")
    assert_equal(row(least, 0), -1, "the negative microsecond")

    var most = reduce_any(d, AggKind.MAX)
    assert_equal(row(most, 0), 86_400_000_000, "and a maximum")

    var t = when([Int64(10), 30, 20], TimeUnit.SECOND)
    var latest = reduce_any(t, AggKind.MAX)
    assert_equal(String(latest.type), "datetime64[s]", "an instant comes back")
    assert_equal(row(latest, 0), 30, "the last of the three")

    var middle = reduce_any(t, AggKind.MEAN)
    assert_equal(String(middle.type), "datetime64[s]", "and so does an average")
    assert_equal(row(middle, 0), 20, "twenty seconds in")


def test_the_mean_of_a_span_truncates_toward_zero() raises:
    """Not a round and not a floor. Both signs are here because they are what
    tells the three apart."""
    var up = reduce_any(span([Int64(1), 2], TimeUnit.SECOND), AggKind.MEAN)
    assert_equal(row(up, 0), 1, "one and a half seconds truncates down to one")

    var down = reduce_any(span([Int64(-2), -1], TimeUnit.SECOND), AggKind.MEAN)
    assert_equal(row(down, 0), -1, "and minus one and a half up to minus one")

    var across = reduce_any(span([Int64(-1), 0], TimeUnit.SECOND), AggKind.MEAN)
    assert_equal(row(across, 0), 0, "and half a second below zero to zero")


def test_a_total_of_nothing_is_zero_and_an_average_of_nothing_is_missing() raises:
    """The two reductions disagree about the empty column on purpose, because
    pandas has them disagree and the reason is that a sum has an identity."""
    var nothing = span(List[Int64](), TimeUnit.SECOND)

    var total = reduce_any(nothing, AggKind.SUM)
    assert_equal(String(total.type), "timedelta64[s]", "still a span")
    assert_true(total.data.validity.get(0), "and it is a real zero")
    assert_equal(row(total, 0), 0, "of no length")

    var average = reduce_any(nothing, AggKind.MEAN)
    assert_true(not average.data.validity.get(0), "the average is missing")

    var blank = span([Int64(0), 0], TimeUnit.SECOND)
    blank.data.validity.set(0, False)
    blank.data.validity.set(1, False)
    var summed = reduce_any(blank, AggKind.SUM)
    assert_true(summed.data.validity.get(0), "an all null column sums to zero")
    assert_equal(row(summed, 0), 0, "of no length either")


def test_an_instant_has_no_total() raises:
    """The `s.sum()` that pandas raises on for a datetime column, with a message
    here that says why rather than saying the dtype is unsupported."""
    var t = when([Int64(0), 1], TimeUnit.SECOND)
    with assert_raises(contains="adding two points in time"):
        _ = reduce_any(t, AggKind.SUM)


def test_whole_days_round_downward() raises:
    """`dt.days` floors, so the negative rows are the ones that matter and a
    truncating division would get both of them wrong."""
    var d = span(
        [
            Int64(0),
            -1,
            86_400_000_000,
            -86_400_000_000,
            31_557_600_000_000,
        ],
        TimeUnit.MICRO,
    )
    var days = temporal_duration_days(d)
    assert_equal(String(days.type), "int64", "a count of days is a number")
    assert_equal(row(days, 0), 0, "nothing is no days")
    assert_equal(row(days, 1), -1, "and a microsecond short of nothing is -1")
    assert_equal(row(days, 2), 1, "one day")
    assert_equal(row(days, 3), -1, "one day back")
    assert_equal(row(days, 4), 365, "a year and a quarter day")


def test_seconds_are_counted_with_their_fraction() raises:
    """`dt.total_seconds` is float64 whatever the column's unit is, so a
    millisecond does not vanish."""
    var d = span([Int64(0), 1, -1, 86_400_000_000], TimeUnit.MICRO)
    var secs = temporal_total_seconds(d)
    assert_equal(String(secs.type), "float64", "float even on whole seconds")
    ref view = secs.as_typed_view[DType.float64]()
    assert_equal(view[0], 0.0, "nothing is no seconds")
    assert_equal(view[1], 1e-06, "a microsecond survives the division")
    assert_equal(view[2], -1e-06, "and so does a negative one")
    assert_equal(view[3], 86400.0, "a day is that many seconds")


def test_neither_reader_will_take_a_point_in_time() raises:
    """How long an instant is has no answer, and saying so is better than
    answering how far it is from the epoch."""
    var t = when([Int64(0)], TimeUnit.SECOND)
    with assert_raises(contains="is a point in time rather than a length"):
        _ = temporal_duration_days(t)
    with assert_raises(contains="is a point in time rather than a length"):
        _ = temporal_total_seconds(t)


def test_whole_numbers_become_spans_without_losing_a_null() raises:
    """`to_timedelta` on an integer column is a relabelling, which is why the
    null has to be checked: a conversion that went through a value would have
    turned it into a zero length span."""
    var raw = counts([Int64(1), 2, 0])
    raw.set_null(2)
    var made = temporal_to_duration(AnyArray(raw^), TimeUnit.SECOND)
    assert_equal(String(made.type), "timedelta64[s]", "seconds, as asked")
    assert_equal(row(made, 0), 1, "one second")
    assert_equal(made.null_count(), 1, "and the missing row is still missing")

    var already = span([Int64(5)], TimeUnit.MILLI)
    var again = temporal_to_duration(already, TimeUnit.SECOND)
    assert_equal(
        String(again.type),
        "timedelta64[ms]",
        "a column that already has a unit keeps it and ignores the argument",
    )


def test_a_series_reaches_all_of_it_and_keeps_its_labels() raises:
    """The frame layer spellings, which are what the conformance driver calls.
    """
    var s = Series("value", span([Int64(0), 86_400_000_000], TimeUnit.MICRO))
    assert_equal(String(s.values.type), "timedelta64[us]", "the dtype prints")
    assert_equal(s.dt_days().name, "value", "the name survives")
    assert_equal(len(s.dt_total_seconds()), 2, "the height survives")
    assert_equal(
        String(s.dt_total_seconds().values.type), "float64", "and the dtype"
    )
    assert_equal(
        s.dt("days").values.as_typed_view[DType.int64]()[1],
        1,
        "the accessor reaches days by name as well",
    )

    var plain = Series("value", AnyArray(counts([Int64(90), 1])))
    assert_equal(
        String(plain.to_timedelta("s").values.type),
        "timedelta64[s]",
        "and the free function has a method here",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
