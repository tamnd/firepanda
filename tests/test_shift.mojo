"""Tests for moving a column along its own rows, and the differences that follow.

A shift reads nothing and computes nothing, so almost everything that can go
wrong with it is an off by one at one end or the other. That is what most of
these assert: which rows moved, which rows are the gap, and which end the gap is
at, for a shift each way and for a shift further than the column is tall.

The other half is the type, and it is the half that is easy to miss because
every value is right when it is wrong. Moving the rows of a complete integer
column makes room that has nothing in it, and pandas has no integer that means
absent, so the answer is a float column. A shift of nothing and a shift with a
fill value make no room and stay integer columns. Those three are one rule
looked at from three sides and they are asserted separately for that reason.

The expected answers were taken from a running pandas 3.0.3 and are quoted in
the assertions that check them.
"""

from std.math import isnan
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.frame.series import Series
from firepanda.kernel.shift import shift_any


def counted[dt: DType](values: List[Scalar[dt]]) -> Array[dt]:
    """Builds a column with nothing missing in it.

    Args:
        values: The values.

    Parameters:
        dt: The dtype.

    Returns:
        The column.
    """
    var out = Array[dt](len(values))
    for i in range(len(values)):
        out[i] = values[i]
    return out^


def test_a_shift_forward_leaves_the_gap_at_the_start() raises:
    """Pandas gives `[nan, 1.0, 2.0, 3.0, 4.0]` for this one."""
    var moved = Series(
        "v", AnyArray(counted[DType.int64]([Int64(1), 2, 3, 4, 5]))
    ).shift()
    assert_equal(len(moved), 5, "a shift never changes the height")
    ref values = moved.values.as_typed_view[DType.float64]()
    assert_true(isnan(values[0]), "the first row had nothing to move into it")
    assert_equal(values[1], 1.0)
    assert_equal(values[4], 4.0, "and the last value fell off the end")


def test_a_shift_backward_leaves_the_gap_at_the_end() raises:
    """Pandas gives `[3.0, 4.0, 5.0, nan, nan]` for `shift(-2)`."""
    var moved = Series(
        "v", AnyArray(counted[DType.int64]([Int64(1), 2, 3, 4, 5]))
    ).shift(-2)
    ref values = moved.values.as_typed_view[DType.float64]()
    assert_equal(values[0], 3.0)
    assert_equal(values[2], 5.0)
    assert_true(isnan(values[3]), "two rows of gap")
    assert_true(isnan(values[4]))


def test_a_shift_past_the_end_is_all_gap() raises:
    """The case a slice of negative length would have got wrong quietly."""
    var column = Series("v", AnyArray(counted[DType.int64]([Int64(1), 2, 3])))
    for periods in [10, -10, 3, -3]:
        var moved = column.shift(periods)
        assert_equal(len(moved), 3)
        assert_equal(
            moved.null_count(), 3, "nothing reached any row, at either sign"
        )


def test_a_shift_of_nothing_keeps_the_type_it_had() raises:
    """No gap is made, so there is nothing to make room for, so no widening.

    pandas agrees and answers int64 here, which is the assertion that says the
    widening below is about the gap rather than about the shift.
    """
    var still = Series(
        "v", AnyArray(counted[DType.int64]([Int64(1), 2, 3]))
    ).shift(0)
    assert_equal(String(still.logical()), "int64")
    assert_equal(still.as_typed[DType.int64]()[1], 2)


def test_a_fill_value_keeps_the_type_it_had() raises:
    """Pandas answers int64 for `shift(1, fill_value=0)` and float64 without it.

    Same column, same shift, two types, and the argument that decides it is the
    one that says whether anything is missing.
    """
    var filled = Series(
        "v", AnyArray(counted[DType.int64]([Int64(1), 2, 3]))
    ).shift(1, Value(Int64(0)))
    assert_equal(String(filled.logical()), "int64", "nothing went missing")
    assert_equal(filled.null_count(), 0)
    ref values = filled.values.as_typed_view[DType.int64]()
    assert_equal(values[0], 0)
    assert_equal(values[1], 1)
    assert_equal(values[2], 2)


def test_a_float_column_keeps_its_width_and_gains_a_nan() raises:
    """A NaN already fits in a float column, so there is nothing to widen."""
    var moved = Series(
        "v", AnyArray(counted[DType.float32]([Float32(1.5), 2.5, 3.5]))
    ).shift()
    assert_equal(String(moved.logical()), "float32")
    ref values = moved.values.as_typed_view[DType.float32]()
    assert_true(isnan(values[0]))
    assert_equal(values[1], Float32(1.5))


def test_a_string_column_shifts_without_a_type_to_widen() raises:
    """The variable width path, which builds its gap rather than zeroing one."""
    var builder = StringBuilder()
    builder.append(String("a").as_bytes())
    builder.append(String("b").as_bytes())
    builder.append(String("c").as_bytes())
    var moved = shift_any(AnyArray(builder^.finish()), 1)
    assert_true(moved.is_string())
    assert_false(
        moved.is_valid(0), "the gap is a real null and not an empty string"
    )
    assert_equal(moved.strings()[1], "a")
    assert_equal(moved.strings()[2], "b")


def test_a_timestamp_column_is_still_a_timestamp_afterwards() raises:
    """The concat refuses to stack two temporal columns of different types, so
    this would raise rather than answer wrongly if the gap block lost the type.
    """
    var stamps = counted[DType.int64]([Int64(10), 20, 30])
    var when = Series(
        "t",
        AnyArray(
            stamps^.into_data(),
            LogicalType.timestamp(TimeUnit.SECOND, TimeZone()),
        ),
    ).shift()
    assert_equal(String(when.logical()), "datetime64[s]")
    assert_equal(
        when.null_count(), 1, "a timestamp keeps its NaT in the bitmap"
    )
    assert_equal(when.as_typed[DType.int64]()[1], 10)


def test_a_difference_is_a_subtraction_against_a_lag() raises:
    """Pandas gives `[nan, 1.0, 1.0, 1.0, 1.0]` for `diff()` on this column."""
    var change = Series(
        "v", AnyArray(counted[DType.int64]([Int64(1), 2, 3, 4, 5]))
    ).diff()
    assert_equal(String(change.logical()), "float64", "for the shift's reason")
    ref values = change.values.as_typed_view[DType.float64]()
    assert_true(isnan(values[0]), "there is no row before the first one")
    for i in range(1, 5):
        assert_equal(values[i], 1.0)


def test_a_difference_the_other_way_puts_the_gap_at_the_end() raises:
    """Pandas gives `[-1.0, -1.0, -1.0, -1.0, nan]` for `diff(-1)`."""
    var change = Series(
        "v", AnyArray(counted[DType.int64]([Int64(1), 2, 3, 4, 5]))
    ).diff(-1)
    ref values = change.values.as_typed_view[DType.float64]()
    assert_equal(values[0], -1.0, "compared against the row ahead")
    assert_equal(values[3], -1.0)
    assert_true(isnan(values[4]))


def test_a_difference_between_instants_is_a_length_of_time() raises:
    """The type rule comes out of the subtraction rather than being restated.

    pandas answers `timedelta64` for `diff()` on a datetime column, and it does
    so here for the same reason it does anywhere else in the library, which is
    that a timestamp minus a timestamp is a duration.
    """
    var stamps = counted[DType.int64]([Int64(0), 60, 180])
    var change = Series(
        "t",
        AnyArray(
            stamps^.into_data(),
            LogicalType.timestamp(TimeUnit.SECOND, TimeZone()),
        ),
    ).diff()
    assert_equal(String(change.logical()), "timedelta64[s]")
    assert_equal(change.null_count(), 1)
    ref values = change.values.as_typed_view[DType.int64]()
    assert_equal(values[1], 60, "a minute")
    assert_equal(values[2], 120, "and then two")


def test_a_change_as_a_fraction_divides_by_the_earlier_value() raises:
    """Pandas gives `[nan, 1.0, 0.5, 0.3333333333333333, 0.25]` here, which says
    it divides by where the column was and not by where it got to."""
    var change = Series(
        "v", AnyArray(counted[DType.int64]([Int64(1), 2, 3, 4, 5]))
    ).pct_change()
    ref values = change.values.as_typed_view[DType.float64]()
    assert_true(isnan(values[0]))
    assert_equal(values[1], 1.0, "one to two is a whole one more")
    assert_equal(values[2], 0.5, "two to three is a half more")
    assert_equal(values[4], 0.25)


def test_a_fraction_divides_before_it_subtracts() raises:
    """The order of the two steps is invisible until it is not.

    Minus zero after minus infinity is the row that tells them apart. Dividing
    first gives zero over minus infinity, which is minus zero, and then minus
    one. Subtracting first gives infinity over minus infinity, which is NaN.
    pandas gives minus one, and this is the case in the conformance corpus that
    caught it.
    """
    var edges = counted[DType.float64](
        [Float64(1.0), Float64("-inf"), Float64(-0.0)]
    )
    var change = Series("v", AnyArray(edges^)).pct_change()
    ref values = change.values.as_typed_view[DType.float64]()
    assert_equal(values[2], -1.0, "and not a NaN")


def test_the_labels_come_through_a_shift_unchanged() raises:
    """A shift moves the values past the labels rather than moving both.

    This is what makes `df["a"] - df["a"].shift(1)` mean anything, since the
    subtraction aligns on labels and would line the column back up with itself if
    the labels moved with it.
    """
    var column = Series("v", AnyArray(counted[DType.int64]([Int64(1), 2, 3])))
    var moved = column.shift()
    assert_equal(len(moved.index), 3)
    assert_true(moved.index.equals(column.index), "the same labels, in place")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
