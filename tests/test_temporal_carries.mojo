"""Tests that a column of times stays a column of times.

Every kernel here is one that moves values around without changing any of them,
or reduces them to a value drawn from the same scale. So the type on the way out
is the type on the way in, and that is easy to say and easy to lose, because the
kernels underneath are written against the physical dtype and a timestamp is an
int64 to every one of them. A sorted column of instants came back as a column of
numbers before this file existed, and nothing complained, because the numbers
were right and the sort had done its job.

The reductions are the other half. Which of them exist on a column of times, and
what each one answers, was measured against a running pandas 3.0.3 rather than
worked out, and three of the answers are not what working it out would give.

A standard deviation of instants is an elapsed time, since the spread of a set
of points in time is a length of time and not a point in it. A variance is
refused while its own square root is given, because the variance is in units of
time multiplied by itself and there is no dtype to hold that while the square
root is back in units of time. And a standard error is refused for a whole
column and answered for a group, on the same column, which is an inconsistency
in pandas rather than a rule and is copied anyway, because a user comparing the
two libraries is comparing whichever spelling they wrote.

Concatenating and coalescing get the other kind of test here, which is that they
refuse. Two resolutions share a physical dtype, so stacking a second column onto
a millisecond one is a silent factor of a thousand on one of the two, and that
is the worst answer a library can give: right in shape, right in dtype, wrong by
three orders of magnitude in half the rows.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit
from firepanda.frame.series import Series
from firepanda.kernel.concat import concat_two_any
from firepanda.kernel.group import AggKind, aggregate_group_any
from firepanda.kernel.nulls import coalesce_any, fill_forward_any
from firepanda.kernel.reduce import reduce_any
from firepanda.kernel.select import filter_any, take_any


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


def when(values: List[Int64], unit: TimeUnit) -> AnyArray:
    """Builds a naive timestamp column.

    Args:
        values: The counts since the epoch.
        unit: What they are counts of.

    Returns:
        The column.
    """
    return AnyArray(counts(values).into_data(), LogicalType.timestamp(unit))


def span(values: List[Int64], unit: TimeUnit) -> AnyArray:
    """Builds a duration column.

    Args:
        values: The counts.
        unit: What they are counts of.

    Returns:
        The column.
    """
    return AnyArray(counts(values).into_data(), LogicalType.duration(unit))


def row(col: AnyArray, i: Int) raises -> Int64:
    """Reads one count out of a temporal or integer column.

    Args:
        col: The column.
        i: Which row.

    Returns:
        The stored count.
    """
    return col.as_typed_view[DType.int64]()[i]


def named(col: AnyArray) -> String:
    """Spells a column's type the way pandas spells its dtype.

    Args:
        col: The column.

    Returns:
        The type as text.
    """
    return String(col.type)


def sample() -> AnyArray:
    """The five second column every test below reduces.

    The values are deliberately not sorted and straddle the epoch, since a
    reduction that lost the sign or the order would still pass on five ascending
    positive numbers.

    Returns:
        A `datetime64[s]` column of five present rows.
    """
    return when([Int64(30), -10, 0, 20, -5], TimeUnit.SECOND)


def one_group(rows: Int) -> Array[DType.uint32]:
    """Puts every row in the same group.

    Args:
        rows: How many rows there are.

    Returns:
        A code per row, all of them zero.
    """
    return Array[DType.uint32](rows)


def test_sorting_a_column_of_instants_gives_back_instants() raises:
    """The case that started this file, since the numbers were already right."""
    var sorted = Series("t", sample()).sort_values()
    assert_equal(
        String(sorted.values.type),
        "datetime64[s]",
        "the sort moved the rows and did not reinterpret them",
    )
    assert_equal(row(sorted.values, 0), -10, "smallest first")
    assert_equal(row(sorted.values, 4), 30, "largest last")


def test_gathering_and_filtering_keep_the_type() raises:
    """`take` and `filter` are the two doors most of the frame layer goes
    through, so a type lost in either is a type lost nearly everywhere."""
    var positions = List[Int]()
    positions.append(2)
    positions.append(0)
    var picked = take_any(sample(), positions)
    assert_equal(named(picked), "datetime64[s]", "a gather keeps the type")
    assert_equal(row(picked, 0), 0, "and gathers the row it was asked for")

    var mask = Array[DType.bool](5)
    for i in range(5):
        mask[i] = i < 2
    var kept = filter_any(sample(), mask)
    assert_equal(named(kept), "datetime64[s]", "and so does a filter")
    assert_equal(len(kept), 2, "which kept the two rows the mask named")


def test_filling_a_gap_keeps_the_type() raises:
    """A forward fill copies a value it already had, so there is nothing in it
    that could change what the value means."""
    var gapped = span([Int64(7), 0], TimeUnit.MILLI)
    gapped.data.validity.set(1, False)
    var filled = fill_forward_any(gapped)
    assert_equal(named(filled), "timedelta64[ms]", "the fill keeps the type")
    assert_equal(row(filled, 1), 7, "and carried the value forward")


def test_stacking_two_of_the_same_resolution_keeps_it() raises:
    """The ordinary case, which has to keep working for the refusal below to be
    worth anything."""
    var both = concat_two_any(
        when([Int64(1)], TimeUnit.MICRO), when([Int64(2)], TimeUnit.MICRO)
    )
    assert_equal(named(both), "datetime64[us]", "the stack keeps the type")
    assert_equal(len(both), 2, "and holds both rows")


def test_two_resolutions_are_not_the_same_column() raises:
    """The silent factor of a thousand, refused in the two places it could
    happen.

    This was written against a message of its own, saying that the two are
    counts of different things and that stacking them would put one of them out
    by the ratio between their units. The message went away because the check
    it belonged to did. Both kernels now compare the whole logical type rather
    than the physical dtype, which catches a resolution mismatch on the way to
    catching everything else, so a second check behind it would never run. What
    is asserted here is the refusal, since that is the part a caller sees and
    the part that would be a wrong answer if it stopped happening.
    """
    with assert_raises(contains="same dtype"):
        _ = concat_two_any(
            when([Int64(1)], TimeUnit.SECOND), when([Int64(2)], TimeUnit.MILLI)
        )
    with assert_raises(contains="same dtype"):
        _ = coalesce_any(
            when([Int64(1)], TimeUnit.SECOND), when([Int64(2)], TimeUnit.MILLI)
        )
    with assert_raises(contains="same dtype"):
        _ = concat_two_any(
            when([Int64(1)], TimeUnit.SECOND), span([Int64(2)], TimeUnit.SECOND)
        )


def test_the_reductions_that_answer_a_time_answer_a_time() raises:
    """Four reductions, each of which draws its answer from the same scale it
    read, so each of them keeps the column's own type."""
    assert_equal(named(reduce_any(sample(), AggKind.MIN)), "datetime64[s]")
    assert_equal(row(reduce_any(sample(), AggKind.MIN), 0), -10)
    assert_equal(row(reduce_any(sample(), AggKind.MAX), 0), 30)
    assert_equal(
        named(reduce_any(sample(), AggKind.MEAN)),
        "datetime64[s]",
        "an average of instants is an instant",
    )
    assert_equal(row(reduce_any(sample(), AggKind.MEAN), 0), 7)
    assert_equal(
        named(reduce_any(sample(), AggKind.MEDIAN)),
        "datetime64[s]",
        "and so is a middle one",
    )
    assert_equal(row(reduce_any(sample(), AggKind.MEDIAN), 0), 0)


def test_a_spread_of_instants_is_a_length_of_time() raises:
    """The answer nobody would guess, and the refusal beside it that explains
    why it is the answer."""
    var spread = reduce_any(sample(), AggKind.STD)
    assert_equal(
        named(spread),
        "timedelta64[s]",
        "how far apart a set of instants are is a length of time",
    )
    assert_equal(row(spread, 0), 17, "truncated toward zero from 17.176")

    with assert_raises(contains="multiplied by itself"):
        _ = reduce_any(sample(), AggKind.VAR)
    with assert_raises(contains="multiplied by itself"):
        _ = reduce_any(sample(), AggKind.SKEW)


def test_the_standard_error_disagrees_with_itself_and_that_is_copied() raises:
    """A pandas inconsistency, reproduced in both directions on purpose."""
    with assert_raises(contains="refused for a whole column"):
        _ = reduce_any(sample(), AggKind.SEM)

    var grouped = aggregate_group_any(
        sample(), AggKind.SEM, one_group(5), 1, trusted=True
    )
    assert_equal(
        named(grouped),
        "timedelta64[s]",
        "the same reduction inside a group by answers a length of time",
    )


def test_counting_is_still_counting() raises:
    """Three reductions that are numbers about the column rather than values out
    of it, so nothing about a column of times changes them."""
    assert_equal(named(reduce_any(sample(), AggKind.COUNT)), "int64")
    assert_equal(row(reduce_any(sample(), AggKind.COUNT), 0), 5)
    assert_equal(named(reduce_any(sample(), AggKind.SIZE)), "int64")
    assert_equal(
        named(reduce_any(sample(), AggKind.NUNIQUE)),
        "int64",
        "how many distinct instants there are is a number and not an instant",
    )
    assert_equal(row(reduce_any(sample(), AggKind.NUNIQUE), 0), 5)


def test_a_group_answers_what_the_whole_column_does() raises:
    """The two paths read one table, so the only way they can disagree is if one
    of them stops reading it."""
    for kind in [AggKind.MIN, AggKind.MAX, AggKind.MEAN, AggKind.FIRST]:
        var grouped = aggregate_group_any(
            sample(), kind, one_group(5), 1, trusted=True
        )
        assert_equal(
            named(grouped),
            "datetime64[s]",
            "a group of instants reduces to an instant",
        )
    var totals = aggregate_group_any(
        span([Int64(1), 2], TimeUnit.MICRO),
        AggKind.SUM,
        one_group(2),
        1,
        trusted=True,
    )
    assert_equal(named(totals), "timedelta64[us]", "and a total of spans")
    assert_equal(row(totals, 0), 3, "which adds them up")

    with assert_raises(contains="a total is additions in a row"):
        _ = aggregate_group_any(
            sample(), AggKind.SUM, one_group(5), 1, trusted=True
        )


def test_a_date_has_no_scale_to_measure_a_spread_against() raises:
    """A date column keeps its type through the reductions that report a value
    it held, and is refused by the ones that would have to invent a unit."""
    var days = AnyArray(Array[DType.int32](3).into_data(), LogicalType.DATE32)
    days.as_typed_view[DType.int32]()[0] = 5
    days.as_typed_view[DType.int32]()[1] = -2
    days.as_typed_view[DType.int32]()[2] = 9
    var latest = reduce_any(days, AggKind.MAX)
    assert_equal(named(latest), "date32[day]", "the latest date is a date")
    assert_equal(
        latest.as_typed_view[DType.int32]()[0], 9, "and it is the right one"
    )

    with assert_raises(contains="no date dtype"):
        _ = reduce_any(days, AggKind.MEAN)
    with assert_raises(contains="no date dtype"):
        _ = reduce_any(days, AggKind.STD)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
