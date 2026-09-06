"""Tests for the Arrow IPC writer.

Almost everything here is a round trip through the reader in the same package,
which is a weaker test than it looks and a stronger one than the alternative. It
is weaker because a writer and a reader that share a misunderstanding agree with
each other perfectly, and no round trip will ever notice. It is stronger because
the reader was written first, against fixtures produced by pyarrow 25 and checked
into `test_arrow_ipc.mojo`, so agreeing with it means agreeing with the bytes
pyarrow wrote.

The other half of the answer does not live here, because it cannot: every file
this writer produces was also handed to pyarrow, which read the values back
exactly, and that is recorded in the pull request rather than in a test, since
running pyarrow from a Mojo test is not something this repository can do. What
the tests here can do instead is pin the structural decisions the round trip
would happily let slide: how many batches a write makes, that a column with no
nulls writes an empty validity buffer rather than a bitmap of ones, that a bool
column goes out one bit per row, and that the file format wraps the stream in the
magic numbers and a footer rather than in something that merely reads back.

The frame under most of these is an int64 with a null in it, a float64 with
none, a bool, and a string column holding an inline string, a string too long to
inline, an empty string and a null, because those four are the string cases that
have ever been wrong.
"""

from std.collections.span import Span
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.data import ColumnData
from firepanda.array.strings import StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.io.arrow_ipc import (
    BUFFER_SIZE,
    MESSAGE_NONE,
    MESSAGE_RECORD_BATCH,
    read_arrow,
    read_arrow_bytes,
    read_ipc_file,
    read_ipc_stream,
    read_message,
)
from firepanda.io.arrow_ipc_write import (
    IpcWriteOptions,
    write_arrow,
    write_ipc_file_bytes,
    write_ipc_stream,
    write_ipc_stream_bytes,
)
from firepanda.io.flatbuf import field_vector, read_scalar, vector_element


def _sample() raises -> DataFrame:
    """Builds the frame most of these tests write.

    Returns:
        Five rows of an int64, a float64, a bool and a string column, with a
        null in the first and the last.
    """
    var ints = Array[DType.int64](5)
    var reals = Array[DType.float64](5)
    var flags = Array[DType.bool](5)
    for i in range(5):
        ints.set_valid(i, Int64(i * 3))
        reals.set_valid(i, Float64(i) + 0.5)
        flags.set_valid(i, i % 2 == 0)
    ints.set_null(2)

    var text = StringBuilder(capacity=5)
    text.append("alpha".as_bytes())
    text.append("a rather longer string than fits inside a view".as_bytes())
    text.append_null()
    text.append("".as_bytes())
    text.append("omega".as_bytes())

    var columns = List[Series]()
    columns.append(Series("i", ints^))
    columns.append(Series("f", reals^))
    columns.append(Series("b", flags^))
    columns.append(Series("s", text^.finish()))
    return DataFrame.from_series(columns^)


def _assert_matches_the_sample(frame: DataFrame) raises:
    """Asserts a frame is the one `_sample` built.

    Args:
        frame: The frame that came back from a round trip.
    """
    assert_equal(frame.width(), 4)
    assert_equal(len(frame), 5)
    assert_equal(frame.names()[0], "i")
    assert_equal(frame.names()[3], "s")

    ref ints = frame[0]
    assert_equal(ints.dtype(), DType.int64)
    assert_equal(ints.null_count(), 1)
    assert_equal(ints.as_typed[DType.int64]()[0], Int64(0))
    assert_equal(ints.as_typed[DType.int64]()[4], Int64(12))
    assert_false(ints.is_valid(2))

    assert_equal(frame[1].as_typed[DType.float64]()[3], Float64(3.5))
    assert_true(frame[2].as_typed[DType.bool]()[0])
    assert_false(frame[2].as_typed[DType.bool]()[1])

    ref text = frame[3].strings()
    assert_equal(text[0], "alpha")
    assert_equal(text[1], "a rather longer string than fits inside a view")
    assert_false(text.is_valid(2))
    assert_equal(text[3], "")
    assert_equal(text[4], "omega")


def _batches(data: Span[UInt8, _], start: Int) raises -> Int:
    """Counts the record batches in a stream by walking it.

    Args:
        data: The stream, or the file the stream is inside.
        start: Where the schema message begins, which is zero in a stream and
            eight in a file.

    Returns:
        How many record batch messages there are.
    """
    var pos = start
    var count = 0
    while True:
        var message = read_message(data, pos)
        if message.header_type == MESSAGE_NONE:
            return count
        if message.header_type == MESSAGE_RECORD_BATCH:
            count += 1
        pos = message.next


def _buffer_length(data: Span[UInt8, _], start: Int, index: Int) raises -> Int:
    """Returns the length of one buffer of the first record batch.

    Reading the metadata rather than the round trip is the only way to see a
    decision the round trip hides, such as a validity buffer that is not written
    at all being read back as a column with no nulls either way.

    Args:
        data: The stream, or the file the stream is inside.
        start: Where the schema message begins.
        index: Which buffer, counting across all columns in Arrow's order.

    Returns:
        The buffer's length in bytes.
    """
    var schema = read_message(data, start)
    var batch = read_message(data, schema.next)
    var buffers = field_vector(data, batch.header, 2)
    var at = vector_element(data, buffers, index, BUFFER_SIZE)
    return Int(read_scalar[DType.int64](data, at + 8))


def _temporal() raises -> DataFrame:
    """Builds a frame of the three temporal types the writer can spell.

    Returns:
        Three rows of a second resolution timestamp, a microsecond one in New
        York, and a date, with a null in the middle of each.
    """
    var seconds = Array[DType.int64](3)
    var micros = Array[DType.int64](3)
    var days = Array[DType.int32](3)
    seconds.set_valid(0, Int64(1710034245))
    seconds.set_null(1)
    seconds.set_valid(2, Int64(-1))
    micros.set_valid(0, Int64(1710034245000000))
    micros.set_null(1)
    micros.set_valid(2, Int64(-1000000))
    days.set_valid(0, Int32(19792))
    days.set_null(1)
    days.set_valid(2, Int32(-1))

    var second = AnyArray(seconds^)
    second.type = LogicalType.timestamp(TimeUnit.SECOND)
    var zoned = AnyArray(micros^)
    zoned.type = LogicalType.timestamp(
        TimeUnit.MICRO, TimeZone("America/New_York")
    )
    var date = AnyArray(days^)
    date.type = LogicalType.DATE32

    var columns = List[Series]()
    columns.append(Series("second", second^))
    columns.append(Series("zoned", zoned^))
    columns.append(Series("day", date^))
    return DataFrame.from_series(columns^)


def test_a_temporal_frame_round_trips_with_its_units_and_its_zone() raises:
    # The unit and the zone are the whole content of a timestamp type and
    # neither is in the buffer, so a writer that dropped either would produce a
    # file that reads back with the right integers and the wrong meaning. That
    # is why this asserts the types before it asserts the values.
    var bytes = write_ipc_stream_bytes(_temporal())
    var frame = read_ipc_stream(Span(bytes))
    assert_equal(frame.width(), 3)
    assert_equal(len(frame), 3)
    assert_equal(String(frame.schema[0].dtype), "datetime64[s]")
    assert_equal(
        String(frame.schema[1].dtype), "datetime64[us, America/New_York]"
    )
    assert_equal(String(frame.schema[2].dtype), "date32[day]")
    assert_equal(frame[0].as_typed[DType.int64]()[0], Int64(1710034245))
    assert_equal(frame[1].as_typed[DType.int64]()[2], Int64(-1000000))
    assert_equal(frame[2].as_typed[DType.int32]()[0], Int32(19792))
    assert_false(frame[0].is_valid(1))
    assert_false(frame[1].is_valid(1))
    assert_false(frame[2].is_valid(1))


def test_a_stream_round_trips() raises:
    var bytes = write_ipc_stream_bytes(_sample())
    _assert_matches_the_sample(read_ipc_stream(Span(bytes)))


def test_a_file_round_trips() raises:
    var bytes = write_ipc_file_bytes(_sample())
    _assert_matches_the_sample(read_ipc_file(Span(bytes)))


def test_the_reader_tells_what_was_written_apart() raises:
    # The writer's two formats go through the sniffing reader rather than the
    # one that was told which format to expect, because that is the call a
    # program that was handed a path makes.
    _assert_matches_the_sample(
        read_arrow_bytes(Span(write_ipc_stream_bytes(_sample())))
    )
    _assert_matches_the_sample(
        read_arrow_bytes(Span(write_ipc_file_bytes(_sample())))
    )


def test_a_file_begins_and_ends_with_the_magic_number() raises:
    var bytes = write_ipc_file_bytes(_sample())
    var magic = "ARROW1".as_bytes()
    for i in range(6):
        assert_equal(bytes[i], magic[i])
        assert_equal(bytes[len(bytes) - 6 + i], magic[i])
    # Two bytes of padding after the magic number, so that the first message
    # starts eight byte aligned.
    assert_equal(bytes[6], UInt8(0))
    assert_equal(bytes[7], UInt8(0))


def test_a_stream_starts_with_a_message() raises:
    var bytes = write_ipc_stream_bytes(_sample())
    for i in range(4):
        assert_equal(bytes[i], UInt8(0xFF))


def test_the_default_is_one_batch() raises:
    var bytes = write_ipc_stream_bytes(_sample())
    assert_equal(_batches(Span(bytes), 0), 1)


def test_rows_per_batch_splits_the_stream() raises:
    var bytes = write_ipc_stream_bytes(_sample(), IpcWriteOptions(2))
    # Five rows in batches of two is three batches, the last of them short.
    assert_equal(_batches(Span(bytes), 0), 3)
    _assert_matches_the_sample(read_ipc_stream(Span(bytes)))


def test_rows_per_batch_splits_a_file_and_the_footer_agrees() raises:
    var bytes = write_ipc_file_bytes(_sample(), IpcWriteOptions(2))
    assert_equal(_batches(Span(bytes), 8), 3)
    # `read_ipc_file` goes through the footer's blocks rather than walking the
    # messages, so this reading the same three batches is the footer being
    # right and not just the stream inside it.
    _assert_matches_the_sample(read_ipc_file(Span(bytes)))


def test_a_batch_per_row_still_round_trips() raises:
    var bytes = write_ipc_file_bytes(_sample(), IpcWriteOptions(1))
    assert_equal(_batches(Span(bytes), 8), 5)
    _assert_matches_the_sample(read_ipc_file(Span(bytes)))


def test_a_column_with_no_nulls_writes_an_empty_validity_buffer() raises:
    var bytes = write_ipc_stream_bytes(_sample())
    # Buffer 0 is the int column's validity, which has a null in it. Buffer 2
    # is the float column's, which does not.
    assert_equal(_buffer_length(Span(bytes), 0, 0), 1)
    assert_equal(_buffer_length(Span(bytes), 0, 2), 0)


def test_a_bool_column_goes_out_one_bit_per_row() raises:
    var flags = Array[DType.bool](1000)
    for i in range(1000):
        flags.set_valid(i, i % 3 == 0)
    var columns = List[Series]()
    columns.append(Series("b", flags^))
    var frame = DataFrame.from_series(columns^)

    var bytes = write_ipc_stream_bytes(frame)
    # A thousand bits is a hundred and twenty five bytes, which is what the
    # column costs in Arrow and an eighth of what it costs in firepanda.
    assert_equal(_buffer_length(Span(bytes), 0, 1), 125)

    var back = read_ipc_stream(Span(bytes))
    var values = back[0].as_typed[DType.bool]()
    for i in range(1000):
        assert_equal(values[i], i % 3 == 0)


def test_a_long_string_column_round_trips_its_payload() raises:
    var text = StringBuilder(capacity=300)
    for i in range(300):
        text.append(
            String(
                "row ", i, " with enough text to live outside the view"
            ).as_bytes()
        )
    var columns = List[Series]()
    columns.append(Series("s", text^.finish()))
    var frame = DataFrame.from_series(columns^)

    var back = read_ipc_stream(Span(write_ipc_stream_bytes(frame)))
    ref values = back[0].strings()
    assert_equal(len(values), 300)
    for i in range(300):
        assert_equal(
            values[i],
            String("row ", i, " with enough text to live outside the view"),
        )


def test_an_empty_frame_keeps_its_schema() raises:
    var ints = Array[DType.int64](0)
    var text = StringBuilder()
    var columns = List[Series]()
    columns.append(Series("i", ints^))
    columns.append(Series("s", text^.finish()))
    var frame = DataFrame.from_series(columns^)

    # No rows means no batches at all, so what comes back is built from the
    # schema message on its own.
    var bytes = write_ipc_file_bytes(frame)
    assert_equal(_batches(Span(bytes), 8), 0)

    var back = read_ipc_file(Span(bytes))
    assert_equal(len(back), 0)
    assert_equal(back.width(), 2)
    assert_equal(back.names()[1], "s")
    assert_equal(back[0].dtype(), DType.int64)
    assert_true(back[1].is_string())


def test_a_frame_of_no_columns_round_trips() raises:
    var back = read_ipc_stream(Span(write_ipc_stream_bytes(DataFrame())))
    assert_equal(back.width(), 0)
    assert_equal(len(back), 0)


def test_a_column_that_is_not_nullable_and_holds_nulls_is_written_nullable() raises:
    var ints = Array[DType.int64](3)
    for i in range(3):
        ints.set_valid(i, Int64(i))
    ints.set_null(1)

    var fields = List[Field]()
    fields.append(Field("i", LogicalType.INT64, False))
    var values = List[AnyArray]()
    values.append(AnyArray(ints^))
    var frame = DataFrame(Schema(fields^), values^)

    # The schema said the column cannot hold nulls and the column holds one.
    # Writing what the schema says would produce a file that describes itself
    # wrongly, so the values win.
    var back = read_ipc_stream(Span(write_ipc_stream_bytes(frame)))
    assert_true(back.schema.fields[0].nullable)
    assert_equal(back[0].null_count(), 1)


def test_a_null_column_cannot_be_written() raises:
    var values = List[AnyArray]()
    values.append(
        AnyArray(ColumnData(Buffer(0), Bitmap(0), 0), LogicalType.NULL)
    )
    var fields = List[Field]()
    fields.append(Field("n", LogicalType.NULL))
    var frame = DataFrame(Schema(fields^), values^)

    with assert_raises(contains="cannot write"):
        _ = write_ipc_stream_bytes(frame)


def test_a_file_on_disk_reads_back() raises:
    var path = String("/tmp/firepanda-arrow-ipc-write.arrow")
    write_arrow(_sample(), path)
    _assert_matches_the_sample(read_arrow(path))


def test_a_stream_on_disk_reads_back() raises:
    var path = String("/tmp/firepanda-arrow-ipc-write.arrows")
    write_ipc_stream(_sample(), path, IpcWriteOptions(3))
    _assert_matches_the_sample(read_arrow(path))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
