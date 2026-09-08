"""Tests for reading a frame the way pandas would have read it.

pandas on the numpy backend has one missing value for a number and it is NaN.
There is no integer that means absent, so an integer column with a missing row
does not survive as an integer column: pandas widens it to float64 when it reads
the data, and every operation after that is a float operation because the column
is a float column. firepanda keeps the Arrow answer, which is the type plus a
validity bitmap, and that is the better model and is not the one the pandas API
describes.

The turn between the two happens once, on the way in, and these tests are that
turn. What they mostly assert is where it does not happen, because a rule that
fires on everything is a rule that has eaten the library: a column with no
missing row is not touched, a string and a boolean and a timestamp keep their
own missing value, and a float32 column stays float32 because a NaN already fits
in it.

The measurements the rule is copied from were taken against pandas 3.0.3, one
Arrow type at a time, and they are quoted in the assertions that check them.
"""

from std.math import isnan
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.value import Value
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.nulls import widen_for_missing


def gapped[dt: DType](values: List[Scalar[dt]], missing: Int) -> Array[dt]:
    """Builds a column with one row cleared.

    Args:
        values: The values, every one of them written.
        missing: Which row to clear afterwards.

    Parameters:
        dt: The dtype.

    Returns:
        The column, with a null at that row.
    """
    var out = Array[dt](len(values))
    for i in range(len(values)):
        out[i] = values[i]
    out.set_null(missing)
    return out^


def value[dt: DType](col: AnyArray, i: Int) raises -> Scalar[dt]:
    """Reads one value out of an erased column.

    Args:
        col: The column.
        i: Which row.

    Parameters:
        dt: The dtype to read it as.

    Returns:
        The value.
    """
    return col.as_typed_view[dt]()[i]


def test_an_integer_column_widens_to_make_room() raises:
    """The measured pandas answer for every integer width, which is float64 and
    never float32 and never the width it started at."""
    var col = widen_for_missing(
        AnyArray(gapped[DType.int64]([Int64(1), 2, 3], 1))
    )
    assert_equal(String(col.type), "float64", "there is no missing int64")
    assert_equal(col.null_count(), 0, "the bitmap is gone")
    assert_equal(value[DType.float64](col, 0), 1.0)
    assert_true(isnan(value[DType.float64](col, 1)), "and the gap is a NaN")
    assert_equal(value[DType.float64](col, 2), 3.0)


def test_a_narrow_integer_widens_all_the_way() raises:
    """A narrow width goes to float64 and not to float32, which is worth its own
    case because float32 holds every int8 there is and pandas does not use it.
    """
    var col = widen_for_missing(
        AnyArray(gapped[DType.int8]([Int8(1), 2, 3], 0))
    )
    assert_equal(String(col.type), "float64")
    assert_true(isnan(value[DType.float64](col, 0)))
    assert_equal(value[DType.float64](col, 2), 3.0)

    var unsigned = widen_for_missing(
        AnyArray(gapped[DType.uint32]([UInt32(7), 8, 9], 2))
    )
    assert_equal(String(unsigned.type), "float64", "unsigned goes the same way")
    assert_true(isnan(value[DType.float64](unsigned, 2)))


def test_a_column_with_nothing_missing_is_not_touched() raises:
    """The common case, and the one that decides whether this is affordable."""
    var full = Array[DType.int64](3)
    full[0] = 1
    full[1] = 2
    full[2] = 3
    var col = widen_for_missing(AnyArray(full^))
    assert_equal(String(col.type), "int64", "an int64 column is still int64")
    assert_equal(value[DType.int64](col, 1), 2)


def test_a_float_column_keeps_its_width() raises:
    """A NaN already fits in a float32, so pandas has no reason to widen one and
    measurably does not."""
    var narrow = widen_for_missing(
        AnyArray(gapped[DType.float32]([Float32(1.5), 2.5, 3.5], 1))
    )
    assert_equal(String(narrow.type), "float32", "float32 stays float32")
    assert_equal(narrow.null_count(), 0)
    assert_true(isnan(value[DType.float32](narrow, 1)))

    var wide = widen_for_missing(
        AnyArray(gapped[DType.float64]([Float64(1.5), 2.5, 3.5], 2))
    )
    assert_equal(String(wide.type), "float64")
    assert_equal(wide.null_count(), 0, "the null became a NaN")
    assert_equal(value[DType.float64](wide, 0), 1.5)


def test_the_types_that_have_their_own_missing_value_keep_it() raises:
    """A number widens because pandas has nowhere else to put the gap. A date has
    NaT, a string has NA and a boolean becomes an object column, and none of
    those is this rule's business."""
    var stamps = Array[DType.int64](3)
    stamps[0] = 1
    stamps[1] = 2
    stamps[2] = 3
    stamps.set_null(1)
    var when = widen_for_missing(
        AnyArray(
            stamps^.into_data(),
            LogicalType.timestamp(TimeUnit.SECOND, TimeZone()),
        )
    )
    assert_equal(
        String(when.type),
        "datetime64[s]",
        "a timestamp is int64 underneath and is not a number here",
    )
    assert_equal(when.null_count(), 1, "and it keeps its NaT")

    var flags = Array[DType.bool](3)
    flags[0] = True
    flags[1] = False
    flags[2] = True
    flags.set_null(0)
    var boolean = widen_for_missing(AnyArray(flags^))
    assert_equal(String(boolean.type), "bool", "pandas makes this an object")
    assert_equal(boolean.null_count(), 1)

    var builder = StringBuilder()
    builder.append(String("a").as_bytes())
    builder.append_null()
    builder.append(String("c").as_bytes())
    var text = widen_for_missing(AnyArray(builder^.finish()))
    assert_true(text.is_string(), "a string column is left alone")
    assert_equal(text.null_count(), 1)


def test_a_frame_widens_the_columns_that_need_it_and_no_others() raises:
    """The entry point the read path actually calls, which has to leave the
    columns beside the widened one exactly as they were."""
    var keys = Array[DType.int64](3)
    keys[0] = 10
    keys[1] = 20
    keys[2] = 30
    var frame = DataFrame.from_series(
        [
            Series("row", AnyArray(keys^)),
            Series("value", AnyArray(gapped[DType.int64]([Int64(1), 2, 3], 1))),
        ]
    ).widen_for_missing()

    assert_equal(
        String(frame.column("row").logical()),
        "int64",
        "nothing was missing in this one",
    )
    assert_equal(
        String(frame.column("value").logical()),
        "float64",
        "and something was missing in this one",
    )
    assert_equal(frame.column("row").null_count(), 0)
    assert_equal(
        frame.column("value").null_count(),
        1,
        (
            "Series.null_count is on the pandas side of the line and counts a"
            " NaN, so the row is still missing after the bitmap stopped saying"
            " so, which is the whole reason this widening is safe"
        ),
    )


def test_what_the_widened_column_does_next_is_what_pandas_does() raises:
    """The point of the whole change, in one assertion: nothing downstream had
    to be told about any of this."""
    var frame = DataFrame.from_series(
        [Series("value", AnyArray(gapped[DType.int64]([Int64(1), 2, 3], 1)))]
    ).widen_for_missing()
    var column = frame.column("value")

    var doubled = column.binary(Value(Int64(2)).weakened(), BinaryOp.MUL)
    assert_equal(
        String(doubled.logical()),
        "float64",
        (
            "pandas answers float64 here because its input was float64, and so"
            " does this, for the same reason and with no rule about pandas"
            " anywhere in the multiply"
        ),
    )
    assert_equal(doubled.as_typed[DType.float64]()[0], 2.0)
    assert_true(
        isnan(doubled.as_typed[DType.float64]()[1]),
        "and a NaN comes out of an arithmetic the way it went in",
    )
    assert_equal(
        doubled.null_count(),
        1,
        "which is still a missing row to everything that asks, per #170",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
