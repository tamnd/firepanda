"""Tests for the CSV reader and writer.

Three things are worth being thorough about here and the rest is bookkeeping.

Inference, because a type decided from a sample is a type that can be wrong on a
row the sample did not include, and the ladder only climbing in one direction is
what makes that safe. A column that looks like an integer for a thousand rows and
then holds `3.5` has to come out a float column, not an error and not a truncated
three.

The three way distinction between a missing value, the empty string and the
string that spells a null word, because a reader that collapses any two of those
loses information the file was carrying.

The round trip, because a writer whose output its own reader misreads is worse
than no writer. A frame that survives `read_csv(write_csv(frame))` unchanged is
the only claim here that covers both files at once.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array import Array, AnyArray, StringBuilder
from firepanda.dtype import Field, LogicalType, Schema
from firepanda.frame import DataFrame, Series
from firepanda.io import (
    Dialect,
    ReadOptions,
    WriteOptions,
    default_dialect,
    infer_schema,
    read_csv_bytes,
    read_csv_bytes_as,
    scan_csv,
    write_csv_bytes,
)


def bytes_of(var text: String) -> List[UInt8]:
    """Copies a string's bytes into a list the reader can span.

    Args:
        text: The string.

    Returns:
        The bytes, without a terminator.
    """
    var out = List[UInt8](capacity=text.byte_length())
    out.extend(text.as_bytes())
    return out^


def frame_of(var text: String) raises -> DataFrame:
    """Reads a frame out of a string with the default options.

    Args:
        text: The file's contents.

    Returns:
        The frame.
    """
    var data = bytes_of(text)
    return read_csv_bytes(Span(data), ReadOptions())


def text_of(frame: DataFrame) raises -> String:
    """Writes a frame out to a string with the default options.

    Args:
        frame: The frame.

    Returns:
        The file's contents.
    """
    var bytes = write_csv_bytes(frame, WriteOptions())
    return String(StringSlice(unsafe_from_utf8=Span(bytes)))


def test_a_header_names_the_columns() raises:
    var frame = frame_of("a,b,c\n1,2,3\n")
    assert_equal(frame.width(), 3)
    assert_equal(len(frame), 1)
    assert_equal(frame.names()[0], "a")
    assert_equal(frame.names()[2], "c")


def test_a_file_without_a_header_gets_positional_names() raises:
    var data = bytes_of("1,2\n3,4\n")
    var frame = read_csv_bytes(
        Span(data), ReadOptions(default_dialect(), False, 0)
    )
    assert_equal(frame.names()[0], "column_0")
    assert_equal(frame.names()[1], "column_1")
    assert_equal(len(frame), 2)


def test_integers_infer_as_int64() raises:
    var frame = frame_of("n\n1\n2\n-3\n")
    assert_true(frame.schema[0].dtype == LogicalType.INT64)
    var values = frame[0].as_typed[DType.int64]()
    assert_equal(values[0], Int64(1))
    assert_equal(values[2], Int64(-3))


def test_one_float_makes_the_whole_column_a_float() raises:
    """The ladder climbs on the row that needs it and stays climbed."""
    var frame = frame_of("n\n1\n2\n3.5\n4\n")
    assert_true(frame.schema[0].dtype == LogicalType.FLOAT64)
    var values = frame[0].as_typed[DType.float64]()
    assert_equal(values[0], Float64(1.0))
    assert_equal(values[2], Float64(3.5))
    assert_equal(values[3], Float64(4.0))


def test_a_float_after_the_sample_is_still_seen_by_default() raises:
    """Inference reads the whole file unless the caller bounds it."""
    var text = String("n\n")
    for _ in range(200):
        text += "1\n"
    text += "3.5\n"
    var frame = frame_of(text^)
    assert_true(frame.schema[0].dtype == LogicalType.FLOAT64)


def test_a_bounded_sample_can_be_told_to_stop_looking() raises:
    """And then the value past the bound is an error rather than a wrong number.
    """
    var text = String("n\n")
    for _ in range(200):
        text += "1\n"
    text += "3.5\n"
    var data = bytes_of(text^)
    with assert_raises(contains="is not an integer"):
        _ = read_csv_bytes(Span(data), ReadOptions(default_dialect(), True, 10))


def split_csv(header: String, var row: String, last: String) -> String:
    """Builds a file big enough that the reader cuts it into blocks.

    Speculation only happens on the parallel path, and the parallel path only
    happens above `MIN_BLOCK` twice over, so a test of it is half a megabyte at
    the smallest. The body is doubled rather than appended to because a dozen
    doublings are a dozen allocations and a hundred thousand appends are not.

    Args:
        header: The header row, terminator included.
        row: The row to repeat, terminator included.
        last: The one row at the end that the sample never sees.

    Returns:
        The file.
    """
    var body = row^
    while body.byte_length() < 1 << 19:
        var doubled = String(body, body)
        body = doubled^
    return String(header, body, last)


def test_a_split_file_settles_on_the_types_full_inference_would_give() raises:
    """The sample is a guess, and every column here is a way of being wrong.

    A file above the block threshold picks its types from the first hundred and
    odd rows of each block and starts reading. What comes out has to be what
    looking at every row first would have given, so the last row of this one is
    a value none of the columns can hold: `b` climbs a rung to float, `c` climbs
    two to text, and `d` never sees a value anywhere and lands on text with its
    nulls intact, which is where the single threaded path leaves it too.

    Three columns in one file rather than three files because the file has to be
    half a megabyte and the test suite has to finish.
    """
    var frame = frame_of(split_csv("a,b,c,d\n", "1,2,2,\n", "1,3.5,x,\n"))
    var last = len(frame) - 1
    assert_true(frame.schema[0].dtype == LogicalType.INT64)
    assert_true(frame.schema[1].dtype == LogicalType.FLOAT64)
    assert_true(frame.schema[2].dtype == LogicalType.STRING)
    assert_true(frame.schema[3].dtype == LogicalType.STRING)
    var floats = frame[1].as_typed[DType.float64]()
    assert_equal(floats[0], Float64(2.0))
    assert_equal(floats[last], Float64(3.5))
    assert_equal(frame[2].strings()[0], "2")
    assert_equal(frame[2].strings()[last], "x")
    assert_false(frame[3].is_valid(0))
    assert_false(frame[3].is_valid(last))


def test_a_declared_schema_still_refuses_a_bad_value_in_a_split_file() raises:
    """Promotion is for a guess. A declared type is the caller's answer.

    The row the message names is the lowest failing row rather than whichever
    block reached one first, so the same file always produces the same message.
    """
    var fields = List[Field]()
    fields.append(Field("a", LogicalType.INT64))
    fields.append(Field("b", LogicalType.INT64))
    var data = bytes_of(split_csv("a,b\n", "1,2\n", "1,3.5\n"))
    with assert_raises(contains="is not an integer"):
        _ = read_csv_bytes_as(Span(data), Schema(fields^), ReadOptions())


def test_text_stops_the_ladder_at_the_top() raises:
    var frame = frame_of("v\n1\n2.5\nx\n")
    assert_true(frame.schema[0].dtype == LogicalType.STRING)
    assert_equal(frame[0].strings()[2], "x")


def test_true_and_false_infer_as_bool() raises:
    var frame = frame_of("flag\ntrue\nFALSE\nTrue\n")
    assert_true(frame.schema[0].dtype == LogicalType.BOOL)
    var values = frame[0].as_typed[DType.bool]()
    assert_true(values[0])
    assert_false(values[1])


def test_ones_and_zeros_are_integers_not_booleans() raises:
    """Inferring a boolean here would narrow a column on the strength of two
    values that are also perfectly good integers, and nothing could undo it."""
    var frame = frame_of("flag\n1\n0\n1\n")
    assert_true(frame.schema[0].dtype == LogicalType.INT64)


def test_missing_values_do_not_decide_the_type() raises:
    """An integer column with a null in it, which pandas cannot represent and
    has to widen to a float for."""
    var frame = frame_of("k,n\n1,1\n2,\n3,3\n4,NA\n")
    assert_true(frame.schema[1].dtype == LogicalType.INT64)
    assert_true(frame[1].is_valid(0))
    assert_false(frame[1].is_valid(1))
    assert_false(frame[1].is_valid(3))


def test_a_column_with_no_values_at_all_is_text() raises:
    """Nothing in the file argues for a number, and text can still be cast."""
    var frame = frame_of("a,b\n1,\n2,NA\n")
    assert_true(frame.schema[1].dtype == LogicalType.STRING)
    assert_false(frame[1].is_valid(0))


def test_a_quoted_empty_field_is_the_empty_string_not_a_null() raises:
    """The only way a CSV file has of telling those two apart."""
    var frame = frame_of('s\n""\nx\n')
    assert_true(frame.schema[0].dtype == LogicalType.STRING)
    assert_true(frame[0].is_valid(0))
    assert_equal(frame[0].strings()[0], "")


def test_a_quoted_field_holding_a_delimiter_is_one_value() raises:
    var frame = frame_of('s\n"1,2"\nx\n')
    assert_equal(frame.width(), 1)
    assert_true(frame.schema[0].dtype == LogicalType.STRING)
    assert_equal(frame[0].strings()[0], "1,2")


def test_quoting_does_not_change_the_inferred_type() raises:
    """Files that quote every field are common and they are not files of text.
    """
    var frame = frame_of('n\n"1"\n"2"\n')
    assert_true(frame.schema[0].dtype == LogicalType.INT64)
    assert_equal(frame[0].as_typed[DType.int64]()[1], Int64(2))


def test_a_quoted_empty_field_in_a_number_column_is_still_a_null() raises:
    """The empty string and the null are only different where a column can hold
    both, and a numeric column cannot."""
    var frame = frame_of('n\n1\n""\n')
    assert_true(frame.schema[0].dtype == LogicalType.INT64)
    assert_false(frame[0].is_valid(1))


def test_a_doubled_quote_is_one_literal_quote() raises:
    var frame = frame_of('s\n"he said ""hi"""\n')
    assert_equal(frame[0].strings()[0], 'he said "hi"')


def test_a_declared_schema_skips_inference() raises:
    """And is the only way to get a column read wider, or as text, than it
    looks."""
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.FLOAT64))
    fields.append(Field("s", LogicalType.STRING))
    var data = bytes_of("n,s\n1,2\n3,4\n")
    var frame = read_csv_bytes_as(Span(data), Schema(fields^), ReadOptions())
    assert_true(frame.schema[0].dtype == LogicalType.FLOAT64)
    assert_equal(frame[0].as_typed[DType.float64]()[0], Float64(1.0))
    assert_equal(frame[1].strings()[1], "4")


def test_a_declared_schema_of_the_wrong_width_is_refused() raises:
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    var data = bytes_of("a,b\n1,2\n")
    with assert_raises(contains="2 columns"):
        _ = read_csv_bytes_as(Span(data), Schema(fields^), ReadOptions())


def test_a_ragged_file_is_refused() raises:
    var data = bytes_of("a,b\n1,2\n3\n")
    with assert_raises(contains="ragged"):
        _ = read_csv_bytes(Span(data), ReadOptions())


def test_an_error_names_the_column_and_the_file_row() raises:
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.INT64))
    var data = bytes_of("n\n1\nx\n")
    with assert_raises(contains="column 'n' row 3"):
        _ = read_csv_bytes_as(Span(data), Schema(fields^), ReadOptions())


def test_an_empty_file_reads_as_an_empty_frame() raises:
    var frame = frame_of("")
    assert_equal(frame.width(), 0)
    assert_equal(len(frame), 0)


def test_a_header_with_no_rows_keeps_its_columns() raises:
    var frame = frame_of("a,b\n")
    assert_equal(frame.width(), 2)
    assert_equal(len(frame), 0)


def test_the_writer_only_quotes_what_needs_it() raises:
    var builder = StringBuilder(3)
    builder.append(String("plain").as_bytes())
    builder.append(String("has,comma").as_bytes())
    builder.append(String('has"quote').as_bytes())
    var columns = List[Series]()
    columns.append(Series("s", builder^.finish()))
    var frame = DataFrame.from_series(columns^)
    assert_equal(
        text_of(frame),
        's\nplain\n"has,comma"\n"has""quote"\n',
    )


def test_the_writer_spells_a_null_as_an_empty_field() raises:
    var values = Array[DType.int64](2)
    values.set_valid(0, Int64(7))
    values.set_null(1)
    var columns = List[Series]()
    columns.append(Series("n", values^))
    var frame = DataFrame.from_series(columns^)
    assert_equal(text_of(frame), "n\n7\n\n")


def test_the_writer_can_be_told_to_leave_the_header_off() raises:
    var values = Array[DType.int64](1)
    values.set_valid(0, Int64(7))
    var columns = List[Series]()
    columns.append(Series("n", values^))
    var frame = DataFrame.from_series(columns^)
    var bytes = write_csv_bytes(frame, WriteOptions(default_dialect(), False))
    assert_equal(String(StringSlice(unsafe_from_utf8=Span(bytes))), "7\n")


def test_a_frame_survives_a_round_trip() raises:
    """Types, values, nulls and the empty string, all the way out and back."""
    var numbers = Array[DType.int64](3)
    numbers.set_valid(0, Int64(1))
    numbers.set_null(1)
    numbers.set_valid(2, Int64(-30))

    var reals = Array[DType.float64](3)
    reals.set_valid(0, Float64(1.5))
    reals.set_valid(1, Float64(0.25))
    reals.set_null(2)

    var builder = StringBuilder(3)
    builder.append(String("a,b").as_bytes())
    builder.append_null()
    builder.append(String("").as_bytes())

    var columns = List[Series]()
    columns.append(Series("n", numbers^))
    columns.append(Series("r", reals^))
    columns.append(Series("s", builder^.finish()))
    var frame = DataFrame.from_series(columns^)

    var back = frame_of(text_of(frame))

    assert_equal(back.width(), 3)
    assert_equal(len(back), 3)
    assert_true(back.schema[0].dtype == LogicalType.INT64)
    assert_true(back.schema[1].dtype == LogicalType.FLOAT64)
    assert_true(back.schema[2].dtype == LogicalType.STRING)

    assert_equal(back[0].as_typed[DType.int64]()[2], Int64(-30))
    assert_false(back[0].is_valid(1))
    assert_equal(back[1].as_typed[DType.float64]()[1], Float64(0.25))
    assert_false(back[1].is_valid(2))
    assert_equal(back[2].strings()[0], "a,b")
    assert_false(back[2].is_valid(1))
    assert_true(back[2].is_valid(2))
    assert_equal(back[2].strings()[2], "")


def test_a_semicolon_file_round_trips_through_its_own_dialect() raises:
    var builder = StringBuilder(1)
    builder.append(String("a;b").as_bytes())
    var columns = List[Series]()
    columns.append(Series("s", builder^.finish()))
    var frame = DataFrame.from_series(columns^)
    var dialect = Dialect(UInt8(59), UInt8(34))
    var bytes = write_csv_bytes(frame, WriteOptions(dialect, True))
    var back = read_csv_bytes(Span(bytes), ReadOptions(dialect, True, 0))
    assert_equal(back[0].strings()[0], "a;b")


def test_inference_reads_the_header_as_a_name_not_a_value() raises:
    """A header of digits would otherwise turn its column into text."""
    var frame = frame_of("1\n2\n3\n")
    assert_equal(frame.names()[0], "1")
    assert_true(frame.schema[0].dtype == LogicalType.INT64)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
