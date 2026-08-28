"""Tests for casting between text and numbers.

The numeric cast is a loop with no branch in it and is tested elsewhere. This is
the pair that crosses the line between bytes and values, where the interesting
cases are all about what happens to something that is not a number: a null, an
empty string, a word, a number too large for the target, and a float written
where an integer was asked for.

The property that ties the two directions together is the round trip. A number
column rendered as text and read back has to be the same column, which is why the
writer spells a float at enough digits rather than at the digits a person wants
to read.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.dtype.logical import LogicalType
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.kernel.cast import cast_any, cast_strings_to, cast_to_strings


def text(var values: List[String]) -> StringArray:
    """Builds a text column from strings, none of them null.

    Args:
        values: The values.

    Returns:
        The column.
    """
    return strings_from_list(values)


def text_with_null(var values: List[String], null_at: Int) -> StringArray:
    """Builds a text column with one null in it.

    Args:
        values: The values, the one at `null_at` being ignored.
        null_at: The element to make null.

    Returns:
        The column.
    """
    var builder = StringBuilder(len(values))
    for i in range(len(values)):
        if i == null_at:
            builder.append_null()
        else:
            builder.append(values[i].as_bytes())
    return builder^.finish()


def words(*values: String) -> List[String]:
    """Collects strings into a list.

    Args:
        values: The strings.

    Returns:
        The list.
    """
    var out = List[String]()
    for value in values:
        out.append(value)
    return out^


def test_text_reads_as_integers() raises:
    var column = text(words("1", "-2", "300"))
    var out = cast_strings_to[DType.int64](column, True)
    assert_equal(len(out), 3)
    assert_equal(out.unsafe_ptr().unsafe_offset(0).unsafe_load(), 1)
    assert_equal(out.unsafe_ptr().unsafe_offset(1).unsafe_load(), -2)
    assert_equal(out.unsafe_ptr().unsafe_offset(2).unsafe_load(), 300)


def test_text_reads_as_floats() raises:
    var column = text(words("1.5", "-0.25", "1e3"))
    var out = cast_strings_to[DType.float64](column, True)
    assert_almost_equal(out.unsafe_ptr().unsafe_offset(0).unsafe_load(), 1.5)
    assert_almost_equal(out.unsafe_ptr().unsafe_offset(1).unsafe_load(), -0.25)
    assert_almost_equal(out.unsafe_ptr().unsafe_offset(2).unsafe_load(), 1000.0)


def test_text_reads_as_bools() raises:
    var column = text(words("true", "false", "TRUE"))
    var out = cast_strings_to[DType.bool](column, True)
    assert_true(out.unsafe_ptr().unsafe_offset(0).unsafe_load())
    assert_false(out.unsafe_ptr().unsafe_offset(1).unsafe_load())
    assert_true(out.unsafe_ptr().unsafe_offset(2).unsafe_load())


def test_a_null_stays_null_and_is_not_read() raises:
    var column = text_with_null(words("1", "x", "3"), 1)
    var out = cast_strings_to[DType.int64](column, True)
    assert_true(out.is_valid(0))
    # The value at 1 is never looked at, so the strict cast does not raise on
    # the "x" that is sitting under the null.
    assert_false(out.is_valid(1))
    assert_true(out.is_valid(2))


def test_a_word_is_refused_by_name() raises:
    var column = text(words("1", "banana", "3"))
    var message = String()
    try:
        _ = cast_strings_to[DType.int64](column, True)
    except error:
        message = String(error)
    assert_true("row 1" in message, message)
    assert_true("banana" in message, message)
    assert_true("int64" in message, message)


def test_a_word_becomes_a_null_when_not_strict() raises:
    var column = text(words("1", "banana", "3"))
    var out = cast_strings_to[DType.int64](column, False)
    assert_true(out.is_valid(0))
    assert_false(out.is_valid(1))
    assert_equal(out.unsafe_ptr().unsafe_offset(2).unsafe_load(), 3)


def test_the_empty_string_is_not_a_number() raises:
    # It is a value the column holds and it is not the null beside it, so
    # coercing it quietly would lose the distinction the column keeps.
    var column = text(words("1", "", "3"))
    var raised = False
    try:
        _ = cast_strings_to[DType.int64](column, True)
    except:
        raised = True
    assert_true(raised, "the empty string is not an integer")

    var lenient = cast_strings_to[DType.int64](column, False)
    assert_false(lenient.is_valid(1))


def test_a_float_written_where_an_integer_was_asked_for_is_refused() raises:
    var column = text(words("1", "2.5"))
    var raised = False
    try:
        _ = cast_strings_to[DType.int64](column, True)
    except:
        raised = True
    assert_true(raised, "2.5 is not an integer")


def test_a_number_too_large_for_the_target_is_refused() raises:
    # The digits are a perfectly good integer and merely do not fit, which the
    # parser reports as a failure rather than wrapping to 44.
    var column = text(words("300"))
    var raised = False
    try:
        _ = cast_strings_to[DType.int8](column, True)
    except:
        raised = True
    assert_true(raised, "300 does not fit in an int8")


def test_surrounding_space_is_not_stripped() raises:
    var column = text(words(" 1"))
    var raised = False
    try:
        _ = cast_strings_to[DType.int64](column, True)
    except:
        raised = True
    assert_true(raised, "a field that needs stripping has to say so")


def test_integers_write_as_text() raises:
    var column = Array[DType.int64](3)
    column.set_valid(0, 1)
    column.set_valid(1, -2)
    column.set_valid(2, 300)
    var out = cast_to_strings(column)
    assert_equal(out[0], "1")
    assert_equal(out[1], "-2")
    assert_equal(out[2], "300")


def test_bools_write_as_words() raises:
    var column = Array[DType.bool](2)
    column.set_valid(0, True)
    column.set_valid(1, False)
    var out = cast_to_strings(column)
    assert_equal(out[0], "true")
    assert_equal(out[1], "false")


def test_a_null_writes_as_a_null_and_not_as_the_empty_string() raises:
    var column = Array[DType.int64](2)
    column.set_valid(0, 7)
    column.set_null(1)
    var out = cast_to_strings(column)
    assert_true(out.validity.get(0))
    assert_false(out.validity.get(1))


def test_a_float_survives_the_round_trip() raises:
    # The display layer rounds to something a person can scan down a column,
    # which would lose the last few digits here. This spells enough of them.
    var column = Array[DType.float64](3)
    column.set_valid(0, 0.1)
    column.set_valid(1, 1.0 / 3.0)
    column.set_valid(2, 1.2345678901234567e100)
    var written = cast_to_strings(column)
    var read = cast_strings_to[DType.float64](written, True)
    for i in range(3):
        assert_equal(
            read.unsafe_ptr().unsafe_offset(i).unsafe_load(),
            column.unsafe_ptr().unsafe_offset(i).unsafe_load(),
            "element " + String(i),
        )


def test_an_integer_survives_the_round_trip() raises:
    var column = Array[DType.int64](3)
    column.set_valid(0, 0)
    column.set_valid(1, 9223372036854775807)
    column.set_valid(2, -9223372036854775808)
    var read = cast_strings_to[DType.int64](cast_to_strings(column), True)
    for i in range(3):
        assert_equal(
            read.unsafe_ptr().unsafe_offset(i).unsafe_load(),
            column.unsafe_ptr().unsafe_offset(i).unsafe_load(),
            "element " + String(i),
        )


def test_the_erased_cast_reads_text_as_numbers() raises:
    var column = AnyArray(text(words("1", "2")))
    var out = cast_any(column, DType.int64)
    assert_false(out.is_string())
    assert_equal(out.dtype(), DType.int64)
    assert_equal(
        out.unsafe_ptr[DType.int64]().unsafe_offset(1).unsafe_load(), 2
    )


def test_the_erased_cast_writes_numbers_as_text() raises:
    var values = Array[DType.int32](2)
    values.set_valid(0, 10)
    values.set_valid(1, 20)
    var out = cast_any(AnyArray(values^), LogicalType.STRING)
    assert_true(out.is_string())
    assert_equal(out.strings()[1], "20")


def test_casting_text_to_text_copies() raises:
    var column = AnyArray(text(words("a", "b")))
    var out = cast_any(column, LogicalType.STRING)
    assert_true(out.is_string())
    assert_equal(out.strings()[0], "a")
    assert_equal(len(out), 2)


def test_nothing_converts_to_the_null_type() raises:
    var column = AnyArray(text(words("a")))
    var raised = False
    try:
        _ = cast_any(column, LogicalType.NULL)
    except:
        raised = True
    assert_true(raised, "the null type is not a target")


def test_a_series_casts_both_ways() raises:
    var values = Array[DType.int64](3)
    values.set_valid(0, 1)
    values.set_valid(1, 2)
    values.set_valid(2, 3)
    var series = Series("id", values^)

    var as_text = series.cast(LogicalType.STRING)
    assert_true(as_text.values.is_string())
    assert_equal(as_text.name, "id")
    assert_equal(as_text.values.strings()[2], "3")

    var back = as_text.cast(DType.int64)
    assert_equal(
        back.values.unsafe_ptr[DType.int64]().unsafe_offset(2).unsafe_load(), 3
    )


def test_a_series_casts_leniently_when_told_to() raises:
    var series = Series("mixed", AnyArray(text(words("1", "n/a", "3"))))
    var out = series.cast(DType.int64, False)
    assert_true(out.values.is_valid(0))
    # "n/a" is a word the CSV reader would have treated as missing at read time.
    # Here it reaches the cast as a value the column holds, and it is the lenient
    # flag and nothing else that turns it into a null.
    assert_false(out.values.is_valid(1))
    assert_true(out.values.is_valid(2))


def test_a_frame_casts_one_column_and_leaves_the_rest() raises:
    var ids = Array[DType.int64](2)
    ids.set_valid(0, 1)
    ids.set_valid(1, 2)
    var columns = List[Series]()
    columns.append(Series("id", ids^))
    columns.append(Series("label", AnyArray(text(words("7", "8")))))
    var frame = DataFrame.from_series(columns^)

    var out = frame.cast("label", DType.int32)
    assert_equal(out.width(), 2)
    assert_equal(out.names()[1], "label")
    assert_equal(out.schema[1].dtype.physical, DType.int32)
    assert_equal(
        out[1].unsafe_ptr[DType.int32]().unsafe_offset(1).unsafe_load(), 8
    )
    # The other column is the one it was, not a copy that went through the cast.
    assert_equal(out.schema[0].dtype.physical, DType.int64)
    assert_equal(
        out[0].unsafe_ptr[DType.int64]().unsafe_offset(0).unsafe_load(), 1
    )


def test_a_frame_renders_a_number_column_as_text() raises:
    var ids = Array[DType.int64](2)
    ids.set_valid(0, 10)
    ids.set_null(1)
    var columns = List[Series]()
    columns.append(Series("id", ids^))
    var frame = DataFrame.from_series(columns^)

    var out = frame.cast("id", LogicalType.STRING)
    assert_true(out[0].is_string())
    assert_equal(out.schema[0].dtype, LogicalType.STRING)
    assert_equal(out[0].strings()[0], "10")
    assert_false(out[0].is_valid(1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
