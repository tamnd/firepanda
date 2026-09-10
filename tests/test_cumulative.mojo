"""Tests for the running totals, products and extremes.

Three things can go wrong here and they are independent of each other, so they
are asserted separately. The arithmetic can be wrong, which is what the small
columns of round numbers catch. The block seam can be wrong, which is what the
tall columns catch, because a column shorter than one register never folds a
carry in and a column whose height is a multiple of the register width never
runs the tail loop. And the type can be wrong, which is the half that looks
right in every value, so there is a test per rule rather than one test that
looks at four columns.

The expected answers were taken from a running pandas 3.0.3 and are quoted in
the assertions that check them.
"""

from std.math import isinf, isnan
from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.frame.series import Series
from firepanda.kernel.cumulative import CumulativeOp, cumulative_any


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


def column[dt: DType](values: List[Scalar[dt]]) -> Series:
    """Builds a series over `counted`, since almost every test below wants one.

    Args:
        values: The values.

    Parameters:
        dt: The dtype.

    Returns:
        The series, named `v`.
    """
    return Series("v", AnyArray(counted[dt](values)))


def test_a_running_total_adds_every_row_up_to_this_one() raises:
    """Pandas gives `[1, 3, 6, 10, 15]` for `cumsum` on this column."""
    var total = column[DType.int64]([Int64(1), 2, 3, 4, 5]).cumsum()
    assert_equal(len(total), 5, "a scan never changes the height")
    ref values = total.values.as_typed_view[DType.int64]()
    assert_equal(values[0], 1, "the first row is itself")
    assert_equal(values[1], 3)
    assert_equal(values[2], 6)
    assert_equal(values[4], 15, "and the last row is the whole column")


def test_a_running_product_multiplies_them_instead() raises:
    """Pandas gives `[1, 2, 6, 24, 120]` for `cumprod` on the same column.

    The identity changes with the operator and this is the test that would fail
    if it did not, since a product seeded with zero is zero forever.
    """
    var product = column[DType.int64]([Int64(1), 2, 3, 4, 5]).cumprod()
    ref values = product.values.as_typed_view[DType.int64]()
    assert_equal(values[0], 1)
    assert_equal(values[3], 24)
    assert_equal(values[4], 120)


def test_a_running_extreme_holds_the_best_row_so_far() raises:
    """Pandas gives `[3, 3, 4, 4, 9]` for `cummax` and `[3, 1, 1, 1, 1]` for
    `cummin` on `[3, 1, 4, 1, 9]`."""
    var source: List[Int64] = [Int64(3), 1, 4, 1, 9]
    var high = column[DType.int64](source).cummax()
    var low = column[DType.int64](source).cummin()
    ref highs = high.values.as_typed_view[DType.int64]()
    ref lows = low.values.as_typed_view[DType.int64]()
    assert_equal(highs[1], 3, "one is not an improvement on three")
    assert_equal(highs[2], 4)
    assert_equal(highs[4], 9)
    assert_equal(lows[0], 3)
    assert_equal(lows[1], 1)
    assert_equal(lows[4], 1, "and nothing beat it afterwards")


def test_a_narrow_integer_column_widens_to_the_widest_one() raises:
    """A running total overflows for the same reason a whole column total does,
    so pandas widens for the same reason, and it widens to the same two types.
    """
    var signed = column[DType.int8]([Int8(100), 100, 100]).cumsum()
    assert_equal(String(signed.logical()), "int64")
    assert_equal(signed.as_typed[DType.int64]()[2], 300, "and does not wrap")

    var unsigned = column[DType.uint32]([UInt32(4000000000), 4000000000])
    var running = unsigned.cumsum()
    assert_equal(String(running.logical()), "uint64", "unsigned stays unsigned")
    assert_equal(running.as_typed[DType.uint64]()[1], 8000000000)


def test_a_float_column_does_not_widen() raises:
    """The one place a scan and a reduction disagree, and it is on purpose.

    `Series.sum` over a float32 column accumulates in float64 because it answers
    one number and can afford the room. A scan answers a whole column and
    widening it would double the answer, so pandas leaves a float32 running
    total in float32 and so does this.
    """
    var total = column[DType.float32]([Float32(1.5), 2.5, 3.0]).cumsum()
    assert_equal(String(total.logical()), "float32")
    ref values = total.values.as_typed_view[DType.float32]()
    assert_equal(values[2], Float32(7.0))


def test_a_boolean_column_counts_when_it_is_added_up() raises:
    """Pandas gives int64 `[1, 1, 2, 3]` for `cumsum` on a bool column and bool
    `[True, True, True, True]` for `cummax`, which is a different question about
    the same column and gets a different type."""
    var source: List[Scalar[DType.bool]] = [True, False, True, True]
    var total = column[DType.bool](source).cumsum()
    assert_equal(String(total.logical()), "int64")
    ref counts = total.values.as_typed_view[DType.int64]()
    assert_equal(counts[1], 1, "a false adds nothing")
    assert_equal(counts[3], 3)

    var ever = column[DType.bool](source).cummax()
    assert_equal(String(ever.logical()), "bool", "an extreme keeps the type")
    ref flags = ever.values.as_typed_view[DType.bool]()
    assert_true(flags[1], "true so far, and a false does not undo it")
    var never = column[DType.bool](source).cummin()
    ref alls = never.values.as_typed_view[DType.bool]()
    assert_true(alls[0])
    assert_equal(alls[1], False, "and a running minimum is a running all")


def test_a_missing_row_is_skipped_and_then_put_back() raises:
    """Pandas gives `[1.0, nan, 4.0, nan, 9.0]` for `cumsum` on
    `[1.0, nan, 3.0, nan, 5.0]`, so the gap neither restarts the total nor
    poisons it. The four after the gap is the whole assertion."""
    var edges = counted[DType.float64](
        [Float64(1.0), Float64("nan"), 3.0, Float64("nan"), 5.0]
    )
    var total = Series("v", AnyArray(edges^)).cumsum()
    ref values = total.values.as_typed_view[DType.float64]()
    assert_equal(values[0], 1.0)
    assert_true(isnan(values[1]), "in place, where it was")
    assert_equal(values[2], 4.0, "and one plus three, not three")
    assert_true(isnan(values[3]))
    assert_equal(values[4], 9.0)


def test_a_missing_row_in_an_integer_column_stays_missing() raises:
    """The Arrow answer, which is what the kernel gives and what the pandas
    facing layer widens afterwards. An int64 column has no NaN to write, so the
    gap comes back as a cleared bit and the total steps over it."""
    var source = counted[DType.int64]([Int64(1), 2, 3, 4])
    source.set_null(1)
    var total = cumulative_any(AnyArray(source^), CumulativeOp.SUM)
    assert_equal(total.null_count(), 1, "exactly where the input was missing")
    var values = total.unsafe_ptr[DType.int64]()
    assert_equal(values[unsafe_offset=0], 1)
    assert_equal(values[unsafe_offset=2], 4, "one plus three")
    assert_equal(values[unsafe_offset=3], 8)


def test_a_nan_the_arithmetic_makes_is_a_value_and_is_carried() raises:
    """Pandas gives `[nan, inf, nan, nan]` for `cumsum` on
    `[nan, inf, -inf, -0.0]`, which looks like a bug and is not.

    The NaN in row zero arrived in the column, so it is missing and is skipped.
    The NaN in row two was produced by adding infinity to negative infinity, so
    it is a value, and every row after it is that value plus something.
    """
    var edges = counted[DType.float64](
        [Float64("nan"), Float64("inf"), Float64("-inf"), Float64(-0.0)]
    )
    var total = Series("v", AnyArray(edges^)).cumsum()
    ref values = total.values.as_typed_view[DType.float64]()
    assert_true(isnan(values[0]), "skipped")
    assert_true(isinf(values[1]))
    assert_true(isnan(values[2]), "and this one is arithmetic")
    assert_true(isnan(values[3]), "so it does not stop at the row that made it")


def test_a_column_taller_than_one_register_folds_the_carry_in() raises:
    """Two hundred rows of one, which is the row number at every row.

    A column shorter than one SIMD register never folds a block carry, so
    everything above this would pass with the carry deleted. This is the test
    that would not.
    """
    var ones = List[Int64]()
    for _ in range(200):
        ones.append(1)
    var total = column[DType.int64](ones).cumsum()
    ref values = total.values.as_typed_view[DType.int64]()
    for i in range(200):
        assert_equal(values[i], Int64(i + 1))


def test_a_height_that_is_not_a_whole_number_of_registers() raises:
    """Every height from nothing at all up to a bit over two registers.

    The block loop stops when fewer rows are left than a register holds and the
    tail loop takes the rest, so the two have to agree about the carry. A height
    of zero is in here as well, since an empty column is the shape most easily
    got wrong by a loop that assumes it runs at least once.
    """
    for rows in range(0, 35):
        var values = List[Int64]()
        for i in range(rows):
            values.append(Int64(i + 1))
        var total = column[DType.int64](values).cumsum()
        assert_equal(len(total), rows)
        if rows > 0:
            ref answers = total.values.as_typed_view[DType.int64]()
            var running = Int64(0)
            for i in range(rows):
                running += Int64(i + 1)
                assert_equal(answers[i], running, String("at height ", rows))


def test_a_missing_row_past_the_first_register_is_still_skipped() raises:
    """The bits for a block are read out of the validity word the block lies
    inside, and a block past the first one is at a non zero offset into that
    word. A gap in the second and third register is what tells a shift by the
    wrong amount from a correct one."""
    var values = List[Int64]()
    for _ in range(100):
        values.append(1)
    var source = counted[DType.int64](values)
    source.set_null(70)
    var total = cumulative_any(AnyArray(source^), CumulativeOp.SUM)
    var answers = total.unsafe_ptr[DType.int64]()
    assert_equal(answers[unsafe_offset=69], 70)
    assert_equal(total.is_valid(70), False, "the gap is where it was put")
    assert_equal(answers[unsafe_offset=71], 71, "and the total stepped over it")
    assert_equal(answers[unsafe_offset=99], 99)


def test_an_elapsed_time_adds_up_and_stays_an_elapsed_time() raises:
    """Two hours and then three is five hours, which is a length of time, so
    pandas keeps the type. This is the one temporal type a running total has an
    answer for."""
    var spans = counted[DType.int64]([Int64(60), 120, 180])
    var total = Series(
        "d", AnyArray(spans^.into_data(), LogicalType.duration(TimeUnit.SECOND))
    ).cumsum()
    assert_equal(String(total.logical()), "timedelta64[s]")
    assert_equal(total.as_typed[DType.int64]()[2], 360)


def test_an_instant_has_a_running_extreme_and_no_running_total() raises:
    """There is no point in time that is the sum of two points in time, and
    pandas raises rather than answering one. There is a latest instant so far,
    though, and it is an instant, so `cummax` keeps the type."""
    var stamps = counted[DType.int64]([Int64(30), 10, 20])
    var when = Series(
        "t",
        AnyArray(
            stamps^.into_data(),
            LogicalType.timestamp(TimeUnit.SECOND, TimeZone()),
        ),
    )
    var latest = when.cummax()
    assert_equal(String(latest.logical()), "datetime64[s]")
    assert_equal(latest.as_typed[DType.int64]()[2], 30, "still the first one")

    with assert_raises(contains="sum of two instants"):
        _ = when.cumsum()


def test_a_product_of_two_elapsed_times_has_no_answer() raises:
    """An area, in units of seconds squared, which is not a type this library
    has and not one pandas has either. It refuses and so does this."""
    var spans = counted[DType.int64]([Int64(60), 120])
    var elapsed = Series(
        "d", AnyArray(spans^.into_data(), LogicalType.duration(TimeUnit.SECOND))
    )
    assert_equal(String(elapsed.cummax().logical()), "timedelta64[s]")
    with assert_raises(contains="not an elapsed time"):
        _ = elapsed.cumprod()


def test_a_string_column_has_no_running_fold_here() raises:
    """Pandas concatenates strings for `cumsum` and orders them for `cummax`.
    Both are out of scope for now and both raise, which is a measured absence
    rather than a wrong answer."""
    var builder = StringBuilder()
    builder.append(String("a").as_bytes())
    builder.append(String("b").as_bytes())
    var text = Series("s", AnyArray(builder^.finish()))
    with assert_raises(contains="not defined on"):
        _ = text.cumsum()
    with assert_raises(contains="not defined on"):
        _ = text.cummax()


def test_the_labels_come_through_a_scan_unchanged() raises:
    """A running total is a value per row and the rows did not move."""
    var source = column[DType.int64]([Int64(1), 2, 3])
    var total = source.cumsum()
    assert_equal(total.name, "v", "and the name is the column's own")
    assert_true(total.index.equals(source.index), "the same labels, in place")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
