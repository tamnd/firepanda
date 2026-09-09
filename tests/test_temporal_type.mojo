"""A date stays a date through the kernels that only move rows around.

A date is stored as a signed thirty two bit day count and a timestamp as a
sixty four bit tick count, so the kernels that filter, gather, stack and fill
all build their output through `Array[DType.int32]` and `Array[DType.int64]`.
Erasing one of those gives back a column that says it is an integer, and the
bytes are right and the label is wrong. That is the quiet kind of wrong: the
frame's schema still says date, so the column and the schema disagree, and the
first thing to notice is whatever tries to put a filtered chunk back beside an
unfiltered one. On a Parquet file of several row groups that is the filter
itself, which raises about a chunk dtype in a message that names neither the
filter nor the date.

So every one of those kernels puts the input's type back on the output, and
these are the tests that say so. They are written against the frame API rather
than the kernels, because a caller reaches this through `df.filter` and a test
that reached past it would pass while the thing anybody types stayed broken.

The other half is refusing to mix. A date column and an int32 column have the
same layout, so before this the two would stack and the answer would be one of
the two, having quietly picked. Now that raises, and `astype` is how a caller
says which one it meant.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.value import Value
from firepanda.dtype import Field, LogicalType, Schema
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.kernel.concat import concat_two_any
from firepanda.kernel.nulls import coalesce_any, fill_forward_any
from firepanda.kernel.select import filter_any, take_any


def days(values: List[Int]) -> AnyArray:
    """Builds a date column from day counts.

    Args:
        values: The days since the epoch, one per row.

    Returns:
        The column, typed as a date.
    """
    var out = Array[DType.int32](len(values))
    for i in range(len(values)):
        out[i] = Int32(values[i])
    var erased = AnyArray(out^)
    erased.type = LogicalType.DATE32
    return erased^


def date_value(value: Int) -> Value:
    """Builds a date scalar from a day count.

    Args:
        value: The days since the epoch.

    Returns:
        The scalar, typed as a date.
    """
    var out = Value(Int32(value))
    out.type = LogicalType.DATE32
    return out^


def dated_frame(values: List[Int]) raises -> DataFrame:
    """Builds a one column frame of dates.

    Args:
        values: The days since the epoch.

    Returns:
        The frame.
    """
    var fields = List[Field](capacity=1)
    fields.append(Field("d", LogicalType.DATE32))
    var columns = List[AnyArray](capacity=1)
    columns.append(days(values))
    return DataFrame(Schema(fields^), columns^)


def test_a_filtered_date_is_still_a_date() raises:
    """The kernel most likely to be reached first, and the one that raised."""
    var df = dated_frame([10, 20, 30, 40])
    var mask = (df.column("d") >= date_value(20)).as_typed[DType.bool]()
    var kept = df.filter(mask)
    assert_equal(kept.rows, 3)
    assert_equal(String(kept.schema.fields[0].dtype), "date32[day]")
    assert_equal(
        String(filter_any(days([1, 2]), mask.slice(0, 2)).type), "date32[day]"
    )


def test_a_gathered_date_is_still_a_date() raises:
    """Which is `take`, and therefore also `sort_values`, `head` and a join."""
    var df = dated_frame([10, 20, 30])
    var order: List[Bool] = [True]
    var nulls: List[Bool] = [False]
    var by: List[String] = ["d"]
    var sorted = df.sort_values(by^, order^, nulls^)
    assert_equal(String(sorted.schema.fields[0].dtype), "date32[day]")
    var at: List[Int] = [2, 0]
    assert_equal(String(take_any(days([10, 20, 30]), at).type), "date32[day]")


def test_two_stacked_dates_are_still_dates() raises:
    """Which is `concat`, and therefore also the two sides of a join's output.
    """
    var stacked = concat_two_any(days([1, 2]), days([3]))
    assert_equal(len(stacked), 3)
    assert_equal(String(stacked.type), "date32[day]")


def test_a_filled_date_is_still_a_date() raises:
    """Forward fill and coalesce, which both rebuild the column they read."""
    var holed = days([5, 0, 9])
    holed.data.validity.set(1, False)
    assert_equal(String(fill_forward_any(holed).type), "date32[day]")
    var other = days([1, 7, 2])
    assert_equal(String(coalesce_any(holed, other).type), "date32[day]")


def test_a_date_and_an_integer_do_not_stack() raises:
    """The other half. They have the same layout and are not the same thing."""
    var plain = Array[DType.int32](2)
    plain[0] = 1
    plain[1] = 2
    var erased = AnyArray(plain^)
    with assert_raises(contains="same dtype"):
        _ = concat_two_any(days([1, 2]), erased)


def test_relabelling_needs_the_same_layout() raises:
    """`retyped` moves the label and never the bytes."""
    var column = days([1, 2, 3])
    assert_equal(
        String(column^.retyped(LogicalType.DATE32).type), "date32[day]"
    )
    with assert_raises(contains="laid out as"):
        _ = days([1, 2, 3]).retyped(LogicalType.FLOAT64)


def test_a_date_survives_a_round_trip_through_a_frame() raises:
    """The whole point, from the caller's side: filter, sort, and still a date.
    """
    var df = dated_frame([40, 10, 30, 20])
    var mask = (df.column("d") > date_value(15)).as_typed[DType.bool]()
    var by: List[String] = ["d"]
    var order: List[Bool] = [False]
    var nulls: List[Bool] = [False]
    var out = df.filter(mask).sort_values(by^, order^, nulls^)
    assert_equal(out.rows, 3)
    assert_equal(String(out.schema.fields[0].dtype), "date32[day]")
    var back = out.column("d").as_typed[DType.int32]()
    assert_equal(Int(back[0]), 20)
    assert_equal(Int(back[2]), 40)
    assert_true(out.column("d").dtype() == DType.int32)


def main() raises:
    """Runs the suite."""
    var suite = TestSuite()
    suite.test[test_a_filtered_date_is_still_a_date]()
    suite.test[test_a_gathered_date_is_still_a_date]()
    suite.test[test_two_stacked_dates_are_still_dates]()
    suite.test[test_a_filled_date_is_still_a_date]()
    suite.test[test_a_date_and_an_integer_do_not_stack]()
    suite.test[test_relabelling_needs_the_same_layout]()
    suite.test[test_a_date_survives_a_round_trip_through_a_frame]()
    suite^.run()
