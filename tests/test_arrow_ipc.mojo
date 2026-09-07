"""Tests for the Arrow IPC reader.

Every byte in this file came out of pyarrow 25 and is checked in as it came off
the wire. That is the only way to test a reader honestly: a fixture this reader
also wrote would agree with itself about any misunderstanding it happens to have,
and the whole reason to read IPC at all is to read what somebody else wrote.

The fixtures are small on purpose, three or four rows each, because what is being
tested is the shape rather than the volume. Between them they cover a stream and
a file, one batch and several, an int64 column, a string column with a null and
with an element too long to inline, a large string column, a string view column,
a float column, a bool column, an int32 column, and a schema with no batches
behind it.

The rest of the tests are the refusals. A list column and a file whose magic
number is wrong both have to be named rather than misread, because either is a
shape this reader could otherwise walk into and produce numbers from. The
dictionary refusals are the sharpest of these: a categorical column read as its
codes is an integer column that looks completely ordinary and is wrong, so a
dictionary whose categories never arrived, whose codes point past the end of
them, or that arrives in pieces is refused by name.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit, TimeZone
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.io.arrow_ipc import (
    read_arrow,
    read_arrow_bytes,
    read_ipc_file,
    read_ipc_stream,
)
from firepanda.io.arrow_ipc_write import write_ipc_stream_bytes


def _from_hex(text: StringSlice) raises -> List[UInt8]:
    """Decodes a hex string into bytes, so a wire capture can be checked in."""
    var digits = text.as_bytes()
    var out = List[UInt8]()
    for i in range(0, len(digits), 2):
        out.append(UInt8(_nibble(digits[i]) * 16 + _nibble(digits[i + 1])))
    return out^


def _nibble(byte: UInt8) raises -> Int:
    var value = Int(byte)
    if value >= ord("0") and value <= ord("9"):
        return value - ord("0")
    if value >= ord("a") and value <= ord("f"):
        return value - ord("a") + 10
    raise Error(String("not a hex digit: ", value))


def test_a_stream_of_two_batches_reads_as_one_frame() raises:
    # Written with a chunk size of two, so the four rows arrive as two messages
    # and the reader has to concatenate them.
    var bytes = _two_batch_stream()
    var frame = read_ipc_stream(Span(bytes))
    assert_equal(frame.width(), 2)
    assert_equal(len(frame), 4)
    assert_equal(frame.names()[0], "a")
    assert_equal(frame.names()[1], "s")
    assert_true(frame.schema[0].dtype == LogicalType.INT64)
    assert_true(frame.schema[1].dtype == LogicalType.STRING)

    var numbers = frame[0].as_typed[DType.int64]()
    for i in range(4):
        assert_equal(numbers[i], Int64(i + 1))


def test_the_strings_of_a_two_batch_stream_survive_the_boundary() raises:
    # The null is in the second batch and the long string is the last row of it,
    # so this also checks that a payload built in one batch is not read with the
    # views of another.
    var bytes = _two_batch_stream()
    var frame = read_ipc_stream(Span(bytes))
    assert_equal(frame[1].strings()[0], "x")
    assert_equal(frame[1].strings()[1], "yy")
    assert_false(frame[1].is_valid(2))
    assert_equal(frame[1].strings()[3], "a much longer string than twelve")
    assert_equal(frame[1].null_count(), 1)


def test_a_file_reads_through_its_footer() raises:
    var bytes = _mixed_file()
    var frame = read_ipc_file(Span(bytes))
    assert_equal(frame.width(), 3)
    assert_equal(len(frame), 3)
    assert_true(frame.schema[0].dtype == LogicalType.FLOAT64)
    assert_true(frame.schema[1].dtype == LogicalType.BOOL)
    assert_true(frame.schema[2].dtype == LogicalType.INT32)

    var reals = frame[0].as_typed[DType.float64]()
    assert_equal(reals[0], Float64(1.5))
    assert_false(frame[0].is_valid(1))
    assert_equal(reals[2], Float64(3.25))


def test_a_bool_column_unpacks_from_bits() raises:
    # Arrow stores a bool as a bit and firepanda as a byte, so this is the one
    # column that is not a copy on the way in.
    var bytes = _mixed_file()
    var frame = read_ipc_file(Span(bytes))
    var flags = frame[1].as_typed[DType.bool]()
    assert_equal(flags[0], True)
    assert_equal(flags[1], False)
    assert_false(frame[1].is_valid(2))


def test_a_null_in_the_middle_keeps_the_values_around_it() raises:
    var bytes = _mixed_file()
    var frame = read_ipc_file(Span(bytes))
    var numbers = frame[2].as_typed[DType.int32]()
    assert_equal(numbers[0], Int32(1))
    assert_false(frame[2].is_valid(1))
    assert_equal(numbers[2], Int32(3))
    assert_equal(frame[2].null_count(), 1)


def test_a_file_of_three_batches_reads_in_order() raises:
    # Six rows written two at a time, so the footer names three blocks and the
    # order they come back in is the order the footer gives.
    var bytes = _three_batch_file()
    var frame = read_ipc_file(Span(bytes))
    assert_equal(len(frame), 6)
    var numbers = frame[0].as_typed[DType.int32]()
    for i in range(6):
        assert_equal(numbers[i], Int32(i))


def test_a_string_view_column_reads() raises:
    # IPC keeps the block lengths in the buffer entries rather than in a buffer
    # of their own, so the reader has to gather them and hand the importer the
    # sizes buffer it expects.
    var bytes = _view_stream()
    var frame = read_ipc_stream(Span(bytes))
    assert_equal(len(frame), 3)
    assert_true(frame.schema[0].dtype == LogicalType.STRING)
    assert_equal(frame[0].strings()[0], "short")
    assert_false(frame[0].is_valid(1))
    assert_equal(frame[0].strings()[2], "another rather long string value")


def test_a_large_string_column_reads() raises:
    # Sixty four bit offsets, which is a different buffer width and the same
    # everything else.
    var bytes = _large_string_stream()
    var frame = read_ipc_stream(Span(bytes))
    assert_equal(len(frame), 3)
    assert_true(frame.schema[0].dtype == LogicalType.STRING)
    assert_equal(frame[0].strings()[0], "alpha")
    assert_false(frame[0].is_valid(1))
    assert_equal(frame[0].strings()[2], "an even longer string value here")


def test_a_schema_with_no_batches_is_an_empty_frame_not_no_frame() raises:
    # What an empty result set looks like on the wire. The columns and their
    # types are the answer, and a frame of no columns would be a different one.
    var bytes = _empty_stream()
    var frame = read_ipc_stream(Span(bytes))
    assert_equal(frame.width(), 2)
    assert_equal(len(frame), 0)
    assert_equal(frame.names()[0], "a")
    assert_true(frame.schema[0].dtype == LogicalType.INT64)
    assert_true(frame.schema[1].dtype == LogicalType.STRING)


def test_a_view_column_split_across_batches_keeps_every_payload() raises:
    # The one case the reader gets wrong if the fill in place arithmetic is
    # wrong. A view column whose producer used one data buffer is copied
    # wholesale, buffer and all, and every long element's offset then has to move
    # by the amount of payload the batches before it contributed. The first batch
    # moves by nothing, which is why a fixture of one batch would pass either
    # way, and this one has three.
    var bytes = _view_batches_stream()
    var frame = read_ipc_stream(Span(bytes))
    assert_equal(len(frame), 6)
    assert_equal(frame.width(), 2)
    assert_equal(frame[0].strings()[0], "short one")
    assert_false(frame[0].is_valid(1))
    assert_equal(frame[0].strings()[2], "a rather long string value here")
    assert_equal(frame[0].strings()[3], "tiny")
    assert_equal(frame[0].strings()[4], "another quite long string value")
    assert_equal(frame[0].strings()[5], "")
    assert_equal(frame[0].null_count(), 1)


def test_the_rows_of_a_split_view_column_still_line_up_with_their_number() raises:
    # The columns of a batch are filled by separate tasks, so a column that ends
    # up one row out of step with the one beside it is a real failure mode and
    # not one a single column fixture can show.
    var bytes = _view_batches_stream()
    var frame = read_ipc_stream(Span(bytes))
    var numbers = frame[1].as_typed[DType.int32]()
    for i in range(6):
        assert_equal(numbers[i], Int32(i))


def test_a_batch_bigger_than_a_piece_is_copied_by_more_than_one_task() raises:
    # The one test here whose bytes are ours rather than pyarrow's, because a
    # fixture of two hundred thousand rows cannot be checked in as hex and what
    # this pins is arithmetic on row counts rather than anything about the
    # format. A batch is cut into pieces of sixty five thousand five hundred and
    # thirty six rows, so this batch becomes four, and the rows either side of
    # every cut are the ones a wrong offset moves.
    var rows = 200000
    var numbers = Array[DType.int64](rows)
    var text = StringBuilder(capacity=rows)
    for i in range(rows):
        numbers.set_valid(i, Int64(i))
        if i % 5 == 0:
            text.append_null()
        elif i % 3 == 0:
            text.append(
                String("a long enough string to need payload ", i).as_bytes()
            )
        else:
            text.append(String("s", i % 100).as_bytes())

    var columns = List[Series]()
    columns.append(Series("n", numbers^))
    columns.append(Series("s", text^.finish()))
    var written = write_ipc_stream_bytes(DataFrame.from_series(columns^))

    var frame = read_ipc_stream(Span(written))
    assert_equal(len(frame), rows)
    var back = frame[0].as_typed[DType.int64]()
    ref strings = frame[1].strings()
    for i in range(rows):
        assert_equal(back[i], Int64(i))
    var edges: List[Int] = [
        0,
        1,
        65535,
        65536,
        65537,
        131071,
        131072,
        196607,
        196608,
        199999,
    ]
    for edge in edges:
        if edge % 5 == 0:
            assert_false(frame[1].is_valid(edge))
        elif edge % 3 == 0:
            assert_equal(
                strings[edge],
                String("a long enough string to need payload ", edge),
            )
        else:
            assert_equal(strings[edge], String("s", edge % 100))
    assert_equal(frame[1].null_count(), rows // 5)


def test_the_reader_tells_a_file_from_a_stream() raises:
    # Both arrive with the same extension and people mean both by it.
    var file = read_arrow_bytes(Span(_mixed_file()))
    assert_equal(len(file), 3)
    var stream = read_arrow_bytes(Span(_two_batch_stream()))
    assert_equal(len(stream), 4)


def test_a_file_on_disk_reads_through_the_mapping() raises:
    # `read_arrow` maps rather than reads, so this is also the test that the
    # buffers handed to the importer as raw addresses can be a read only
    # mapping, which is the case every real file is.
    var path = String("/tmp/firepanda-arrow-ipc.arrow")
    var bytes = _mixed_file()
    var handle = open(path, "w")
    handle.write_bytes(Span(bytes))
    handle.close()

    var frame = read_arrow(path)
    assert_equal(frame.width(), 3)
    assert_equal(len(frame), 3)
    assert_equal(frame[0].as_typed[DType.float64]()[2], Float64(3.25))


def test_a_timestamp_column_reads_at_the_unit_the_file_says() raises:
    # The same instant written four times at four resolutions, which is the one
    # fixture that catches a reader that normalizes to nanoseconds on the way
    # in. Such a reader gets every value right and every dtype wrong, and pandas
    # 3 puts the resolution in the dtype, so wrong dtype is wrong answer.
    var frame = read_ipc_stream(Span(_timestamp_stream()))
    assert_equal(frame.width(), 4)
    assert_equal(len(frame), 3)
    assert_true(frame.schema[0].dtype == LogicalType.timestamp(TimeUnit.SECOND))
    assert_true(frame.schema[1].dtype == LogicalType.timestamp(TimeUnit.MILLI))
    assert_true(frame.schema[2].dtype == LogicalType.timestamp(TimeUnit.MICRO))
    assert_true(frame.schema[3].dtype == LogicalType.timestamp(TimeUnit.NANO))
    assert_equal(String(frame.schema[0].dtype), "datetime64[s]")
    assert_equal(String(frame.schema[3].dtype), "datetime64[ns]")


def test_a_timestamp_column_holds_the_integers_arrow_wrote() raises:
    # 2024-03-10 01:30:45 and one second before the epoch, so the negative row
    # checks that nothing here treats the count as unsigned.
    var frame = read_ipc_stream(Span(_timestamp_stream()))
    assert_equal(frame[0].as_typed[DType.int64]()[0], Int64(1710034245))
    assert_equal(frame[0].as_typed[DType.int64]()[2], Int64(-1))
    assert_equal(
        frame[3].as_typed[DType.int64]()[0], Int64(1710034245000000000)
    )
    assert_equal(frame[3].as_typed[DType.int64]()[2], Int64(-1000000000))
    assert_false(frame[0].is_valid(1))
    assert_equal(frame[0].null_count(), 1)


def test_a_zoned_timestamp_keeps_the_zone_and_a_naive_one_does_not_get_utc() raises:
    # Two columns holding the same integers, and they are not the same type. A
    # naive column is not UTC, it is a column with no answer to the question,
    # and the day this reader starts filling that in is the day a user in Sydney
    # gets different numbers out of the same file.
    var frame = read_ipc_stream(Span(_zoned_stream()))
    assert_equal(frame.width(), 2)
    assert_true(frame.schema[0].dtype == LogicalType.timestamp(TimeUnit.MICRO))
    assert_true(
        frame.schema[1].dtype
        == LogicalType.timestamp(TimeUnit.MICRO, TimeZone("America/New_York"))
    )
    var naive = frame.schema[0].dtype
    var zoned = frame.schema[1].dtype
    assert_true(naive != zoned)
    assert_equal(
        String(frame.schema[1].dtype), "datetime64[us, America/New_York]"
    )
    assert_equal(
        frame[0].as_typed[DType.int64]()[0],
        frame[1].as_typed[DType.int64]()[0],
    )


def test_a_date_column_reads_as_a_count_of_days() raises:
    # 19792 days after the epoch is 2024-03-10, and the negative row is the day
    # before it.
    var frame = read_ipc_stream(Span(_date_stream()))
    assert_equal(frame.width(), 1)
    assert_true(frame.schema[0].dtype == LogicalType.DATE32)
    assert_equal(String(frame.schema[0].dtype), "date32[day]")
    assert_equal(frame[0].as_typed[DType.int32]()[0], Int32(19792))
    assert_equal(frame[0].as_typed[DType.int32]()[2], Int32(-1))
    assert_false(frame[0].is_valid(1))


def test_a_date64_column_is_refused_by_name() raises:
    # A date64 counts milliseconds and firepanda's date counts days. The two are
    # a division apart and the reader will not do it silently.
    with assert_raises(contains="date64"):
        _ = read_ipc_stream(Span(_date64_stream()))


def test_a_duration_column_reads_as_an_elapsed_count() raises:
    # Arrow type 18, and the one temporal type whose unit field defaults to
    # millisecond rather than to zero, so a microsecond column only reads
    # correctly if the reader asks for the field instead of taking the default.
    var frame = read_ipc_stream(Span(_duration_stream()))
    assert_equal(frame.width(), 1)
    assert_true(frame.schema[0].dtype == LogicalType.duration(TimeUnit.MICRO))
    assert_equal(String(frame.schema[0].dtype), "timedelta64[us]")
    assert_equal(frame[0].as_typed[DType.int64]()[0], Int64(1))


def test_a_list_column_is_refused_by_name() raises:
    with assert_raises(contains="nested"):
        _ = read_ipc_stream(Span(_list_stream()))


def test_a_dictionary_encoded_column_reads_as_a_categorical() raises:
    # The categories are not in the schema message. They arrive in a message of
    # their own between the schema and the batch that uses them, which is the
    # only type in this file whose description is spread across two messages.
    var frame = read_ipc_stream(Span(_dictionary_stream()))
    assert_equal(frame.width(), 1)
    assert_equal(String(frame.schema[0].dtype), "category")
    assert_true(frame.schema[0].dtype == LogicalType.dictionary(DType.int32))
    assert_equal(len(frame[0]), 3)
    assert_equal(len(frame[0].categories()), 2)
    assert_equal(frame[0].categories()[0], "a")
    assert_equal(frame[0].categories()[1], "b")
    var codes = frame[0].codes[DType.int32]()
    assert_equal(codes[0], Int32(0))
    assert_equal(codes[1], Int32(1))
    assert_equal(codes[2], Int32(0))


def test_the_codes_of_a_categorical_are_not_reachable_as_values() raises:
    # The point of the whole type. The codes are a perfectly good int32 buffer
    # and reading them as one gives 0, 1, 0 for a column whose values are 'a',
    # 'b', 'a', so the way to them has to be a different call rather than the
    # ordinary one.
    var frame = read_ipc_stream(Span(_dictionary_stream()))
    with assert_raises(contains="positions into its categories"):
        _ = frame[0].as_typed[DType.int32]()
    with assert_raises(contains="not a string column"):
        _ = frame[0].strings()


def test_an_ordered_dictionary_in_a_file_keeps_its_order_and_its_index_width() raises:
    # A file rather than a stream, so the categories are found through the
    # footer's own list of dictionary blocks rather than by meeting the message
    # on the way past. The index is an int8 and the categories are large
    # strings, neither of which is what pyarrow writes by default, because the
    # reader reads the width the file states rather than the width it expects.
    var frame = read_ipc_file(Span(_ordered_dictionary_file()))
    assert_equal(frame.width(), 1)
    assert_true(
        frame.schema[0].dtype == LogicalType.dictionary(DType.int8, True)
    )
    assert_true(frame.schema[0].dtype.ordered)
    assert_equal(len(frame[0]), 4)
    assert_equal(len(frame[0].categories()), 3)
    assert_equal(frame[0].categories()[0], "low")
    assert_equal(frame[0].categories()[1], "high")
    assert_equal(frame[0].categories()[2], "medium")
    var codes = frame[0].codes[DType.int8]()
    assert_equal(codes[0], Int8(0))
    assert_equal(codes[3], Int8(2))


def test_an_ordered_dictionary_is_not_the_same_type_as_an_unordered_one() raises:
    # pandas puts the flag in the dtype and so does this, because it decides
    # whether `<` on two categorical columns is an answer or an error.
    assert_true(
        LogicalType.dictionary(DType.int32, True)
        != LogicalType.dictionary(DType.int32, False)
    )


def test_a_dictionary_over_something_other_than_strings_is_refused() raises:
    # Arrow allows a dictionary over any type. pandas' categories are an index
    # of objects and firepanda's are strings, so an integer dictionary is
    # refused rather than read as the integer column it decodes to, which would
    # be a different dtype from the one the file names.
    with assert_raises(contains="dictionary encoded over"):
        _ = read_ipc_stream(Span(_integer_dictionary_stream()))


def test_a_code_that_points_past_the_categories_is_refused() raises:
    # Nothing downstream can catch this. A code is read in order to reach a
    # category, and by then a bad one is a read off the end of the categories
    # rather than a question anybody is in a position to ask.
    with assert_raises(
        contains="has a code of 5 at row 1 against 2 categories"
    ):
        _ = read_ipc_stream(Span(_out_of_range_dictionary_stream()))


def test_categories_that_arrive_in_pieces_are_refused() raises:
    # A delta dictionary adds categories partway through a stream, so the second
    # batch reads against a longer list than the first. Every batch here becomes
    # one frame with one set of categories, so the two would have to be merged
    # and the codes of one side rewritten, which is not something to do quietly.
    with assert_raises(contains="arrive as a delta"):
        _ = read_ipc_stream(Span(_delta_dictionary_stream()))


def test_a_file_without_the_magic_number_is_refused() raises:
    var bytes = _mixed_file()
    bytes[0] = UInt8(ord("X"))
    with assert_raises(contains="magic"):
        _ = read_ipc_file(Span(bytes))


def test_a_file_truncated_at_the_footer_is_refused() raises:
    var bytes = _mixed_file()
    var short = List[UInt8]()
    for i in range(len(bytes) - 20):
        short.append(bytes[i])
    with assert_raises():
        _ = read_ipc_file(Span(short))


def test_a_stream_truncated_in_a_body_is_refused() raises:
    # The metadata says how long the body is, so a body that is not there is
    # caught by the framing rather than by reading whatever follows.
    var bytes = _two_batch_stream()
    var short = List[UInt8]()
    for i in range(len(bytes) - 40):
        short.append(bytes[i])
    with assert_raises(contains="body"):
        _ = read_ipc_stream(Span(short))


def test_a_stream_that_does_not_begin_with_a_schema_is_refused() raises:
    var bytes = _two_batch_stream()
    var without_schema = List[UInt8]()
    for i in range(8, len(bytes)):
        without_schema.append(bytes[i])
    with assert_raises():
        _ = read_ipc_stream(Span(without_schema))


def test_a_stream_of_nothing_is_refused() raises:
    var empty = List[UInt8]()
    with assert_raises(contains="schema"):
        _ = read_ipc_stream(Span(empty))


def _two_batch_stream() raises -> List[UInt8]:
    """760 bytes from pyarrow 25."""
    return _from_hex(
        "ffffffffa80000001000000000000a000c000600050008000a00000000010400"
        "0c00000008000800000004000800000004000000020000004000000004000000"
        "d8ffffff00000105100000001800000004000000000000000100000073000000"
        "0400040004000000100014000800060007000c00000010001000000000000102"
        "100000001c0000000400000000000000010000006100000008000c0008000700"
        "08000000000000014000000000000000ffffffffc80000001400000000000000"
        "0c0016000600050008000c000c00000000030400180000005800000000000000"
        "00000a0018000c00040008000a0000006c000000100000000200000000000000"
        "0000000005000000000000000000000000000000000000000000000000000000"
        "2000000000000000200000000000000000000000000000002000000000000000"
        "0c00000000000000300000000000000023000000000000000000000002000000"
        "0200000000000000000000000000000002000000000000000000000000000000"
        "0100000000000000020000000000000003000000000000000400000000000000"
        "0000000001000000030000000000000078797961206d756368206c6f6e676572"
        "20737472696e67207468616e207477656c76650000000000ffffffffc8000000"
        "14000000000000000c0016000600050008000c000c0000000003040018000000"
        "480000000000000000000a0018000c00040008000a0000006c00000010000000"
        "0200000000000000000000000500000000000000000000000000000000000000"
        "0000000000000000100000000000000010000000000000000100000000000000"
        "18000000000000000c0000000000000028000000000000002000000000000000"
        "0000000002000000020000000000000000000000000000000200000000000000"
        "0100000000000000030000000000000004000000000000000200000000000000"
        "0000000000000000200000000000000061206d756368206c6f6e676572207374"
        "72696e67207468616e207477656c7665ffffffff00000000"
    )


def _mixed_file() raises -> List[UInt8]:
    """818 bytes from pyarrow 25."""
    return _from_hex(
        "4152524f57310000ffffffffd80000001000000000000a000c00060005000800"
        "0a000000000104000c0000000800080000000400080000000400000003000000"
        "780000003c00000004000000a4ffffff00000102100000001c00000004000000"
        "00000000010000006900000008000c0008000700080000000000000120000000"
        "d8ffffff00000106100000001800000004000000000000000100000062000000"
        "0400040004000000100014000800060007000c00000010001000000000000103"
        "1000000018000000040000000000000001000000660006000800060006000000"
        "0000020000000000ffffffffe800000014000000000000000c00160006000500"
        "08000c000c0000000003040018000000480000000000000000000a0018000c00"
        "040008000a0000007c0000001000000003000000000000000000000006000000"
        "0000000000000000010000000000000008000000000000001800000000000000"
        "2000000000000000010000000000000028000000000000000100000000000000"
        "3000000000000000010000000000000038000000000000000c00000000000000"
        "0000000003000000030000000000000001000000000000000300000000000000"
        "0100000000000000030000000000000001000000000000000500000000000000"
        "000000000000f83f00000000000000000000000000000a400300000000000000"
        "0100000000000000050000000000000001000000000000000300000000000000"
        "ffffffff00000000100000000c001400060008000c0010000c00000000000400"
        "38000000280000000400000001000000e800000000000000f000000000000000"
        "4800000000000000000000000000000008000800000004000800000004000000"
        "03000000780000003c00000004000000a4ffffff00000102100000001c000000"
        "0400000000000000010000006900000008000c00080007000800000000000001"
        "20000000d8ffffff000001061000000018000000040000000000000001000000"
        "620000000400040004000000100014000800060007000c000000100010000000"
        "0000010310000000180000000400000000000000010000006600060008000600"
        "0600000000000200000100004152524f5731"
    )


def _view_batches_stream() raises -> List[UInt8]:
    """1216 bytes from pyarrow 25, six rows of string view in three batches."""
    return _from_hex(
        "ffffffffa80000001000000000000a000c000600050008000a00000000010400"
        "0c00000008000800000004000800000004000000020000004c00000004000000"
        "ccffffff00000102100000001c0000000400000000000000010000006e000000"
        "08000c0008000700080000000000000120000000100014000800060007000c00"
        "0000100010000000000001181000000018000000040000000000000001000000"
        "73000000040004000400000000000000ffffffffe00000001400000000000000"
        "0c0016000600050008000c000c000000000304001c0000008000000000000000"
        "00000e001c0010000400080000000c000e000000800000002400000010000000"
        "0200000000000000000000000100000001000000000000000000000005000000"
        "0000000000000000010000000000000008000000000000002000000000000000"
        "28000000000000003e0000000000000068000000000000000000000000000000"
        "6800000000000000180000000000000000000000020000000200000000000000"
        "0100000000000000020000000000000000000000000000003d00000000000000"
        "0900000073686f7274206f6e6500000000000000000000000000000000000000"
        "6120726174686572206c6f6e6720737472696e672076616c7565206865726561"
        "6e6f74686572207175697465206c6f6e6720737472696e672076616c75650000"
        "000000000100000002000000030000000400000005000000ffffffffe0000000"
        "14000000000000000c0016000600050008000c000c000000000304001c000000"
        "680000000000000000000e001c0010000400080000000c000e00000080000000"
        "2400000010000000020000000000000000000000010000000100000000000000"
        "0000000005000000000000000000000000000000000000000000000000000000"
        "200000000000000020000000000000003e000000000000006000000000000000"
        "0000000000000000600000000000000008000000000000000000000002000000"
        "0200000000000000000000000000000002000000000000000000000000000000"
        "1f0000006120726100000000000000000400000074696e790000000000000000"
        "6120726174686572206c6f6e6720737472696e672076616c7565206865726561"
        "6e6f74686572207175697465206c6f6e6720737472696e672076616c75650000"
        "0200000003000000ffffffffe000000014000000000000000c00160006000500"
        "08000c000c000000000304001c000000680000000000000000000e001c001000"
        "0400080000000c000e0000008000000024000000100000000200000000000000"
        "0000000001000000010000000000000000000000050000000000000000000000"
        "0000000000000000000000000000000020000000000000002000000000000000"
        "3e00000000000000600000000000000000000000000000006000000000000000"
        "0800000000000000000000000200000002000000000000000000000000000000"
        "020000000000000000000000000000001f000000616e6f74000000001f000000"
        "000000000000000000000000000000006120726174686572206c6f6e67207374"
        "72696e672076616c75652068657265616e6f74686572207175697465206c6f6e"
        "6720737472696e672076616c756500000400000005000000ffffffff00000000"
    )


def _view_stream() raises -> List[UInt8]:
    """400 bytes from pyarrow 25."""
    return _from_hex(
        "ffffffff700000001000000000000a000c000600050008000a00000000010400"
        "0c00000008000800000004000800000004000000010000001400000010001400"
        "0800060007000c00000010001000000000000118100000001800000004000000"
        "000000000100000076000000040004000400000000000000ffffffffb0000000"
        "14000000000000000c0016000600050008000c000c000000000304001c000000"
        "580000000000000000000e001c0010000400080000000c000e00000060000000"
        "2400000010000000030000000000000000000000010000000100000000000000"
        "0000000003000000000000000000000001000000000000000800000000000000"
        "3000000000000000380000000000000020000000000000000000000001000000"
        "0300000000000000010000000000000005000000000000000500000073686f72"
        "74000000000000000000000000000000000000000000000020000000616e6f74"
        "0000000000000000616e6f7468657220726174686572206c6f6e672073747269"
        "6e672076616c7565ffffffff00000000"
    )


def _empty_stream() raises -> List[UInt8]:
    """184 bytes from pyarrow 25."""
    return _from_hex(
        "ffffffffa80000001000000000000a000c000600050008000a00000000010400"
        "0c00000008000800000004000800000004000000020000004000000004000000"
        "d8ffffff00000105100000001800000004000000000000000100000073000000"
        "0400040004000000100014000800060007000c00000010001000000000000102"
        "100000001c0000000400000000000000010000006100000008000c0008000700"
        "08000000000000014000000000000000ffffffff00000000"
    )


def _three_batch_file() raises -> List[UInt8]:
    """834 bytes from pyarrow 25."""
    return _from_hex(
        "4152524f57310000ffffffff780000001000000000000a000c00060005000800"
        "0a000000000104000c0000000800080000000400080000000400000001000000"
        "14000000100014000800060007000c0000001000100000000000010210000000"
        "1c0000000400000000000000010000006e00000008000c000800070008000000"
        "0000000120000000ffffffff8800000014000000000000000c00160006000500"
        "08000c000c0000000003040018000000180000000000000000000a0018000c00"
        "040008000a0000003c0000001000000002000000000000000000000002000000"
        "0000000000000000000000000000000000000000000000001800000000000000"
        "0000000001000000020000000000000000000000000000000000000001000000"
        "02000000030000000400000005000000ffffffff880000001400000000000000"
        "0c0016000600050008000c000c00000000030400180000000800000000000000"
        "00000a0018000c00040008000a0000003c000000100000000200000000000000"
        "0000000002000000000000000000000000000000000000000000000000000000"
        "0800000000000000000000000100000002000000000000000000000000000000"
        "0200000003000000ffffffff8800000014000000000000000c00160006000500"
        "08000c000c0000000003040018000000080000000000000000000a0018000c00"
        "040008000a0000003c0000001000000002000000000000000000000002000000"
        "0000000000000000000000000000000000000000000000000800000000000000"
        "0000000001000000020000000000000000000000000000000400000005000000"
        "ffffffff00000000100000000c001400060008000c0010000c00000000000400"
        "6400000054000000040000000300000088000000000000009000000000000000"
        "1800000000000000300100000000000090000000000000000800000000000000"
        "c801000000000000900000000000000008000000000000000000000008000800"
        "0000040008000000040000000100000014000000100014000800060007000c00"
        "000010001000000000000102100000001c000000040000000000000001000000"
        "6e00000008000c0008000700080000000000000120000000d00000004152524f"
        "5731"
    )


def _list_stream() raises -> List[UInt8]:
    """416 bytes from pyarrow 25."""
    return _from_hex(
        "ffffffffa80000001000000000000a000c000600050008000a00000000010400"
        "0c000000080008000000040008000000040000000100000004000000d4ffffff"
        "0000010c140000001c000000040000000100000024000000010000006c000000"
        "0400040004000000100014000800060007000c00000010001000000000000102"
        "10000000200000000400000000000000040000006974656d0000000008000c00"
        "08000700080000000000000140000000ffffffffb80000001400000000000000"
        "0c0016000600050008000c000c00000000030400180000002800000000000000"
        "00000a0018000c00040008000a0000005c000000100000000200000000000000"
        "0000000004000000000000000000000000000000000000000000000000000000"
        "0c00000000000000100000000000000000000000000000001000000000000000"
        "1800000000000000000000000200000002000000000000000000000000000000"
        "0300000000000000000000000000000000000000020000000300000000000000"
        "010000000000000002000000000000000300000000000000ffffffff00000000"
    )


def _dictionary_stream() raises -> List[UInt8]:
    """520 bytes from pyarrow 25."""
    return _from_hex(
        "ffffffff900000001000000000000a000c000600050008000a00000000010400"
        "04000000bcffffff040000000100000014000000100018000800060007000c00"
        "10001400100000000000010514000000400000001c0000000400000000000000"
        "01000000640000000800080000000400080000000c00000008000c0008000700"
        "080000000000000120000000040004000400000000000000ffffffffa8000000"
        "14000000000000000c0014000600050008000c000c0000000002040014000000"
        "180000000000000008000a0000000400080000001000000000000a0018000c00"
        "040008000a0000004c0000001000000002000000000000000000000003000000"
        "0000000000000000000000000000000000000000000000000c00000000000000"
        "1000000000000000020000000000000000000000010000000200000000000000"
        "0000000000000000000000000100000002000000000000006162000000000000"
        "ffffffff8800000014000000000000000c0016000600050008000c000c000000"
        "0003040018000000100000000000000000000a0018000c00040008000a000000"
        "3c00000010000000030000000000000000000000020000000000000000000000"
        "000000000000000000000000000000000c000000000000000000000001000000"
        "0300000000000000000000000000000000000000010000000000000000000000"
        "ffffffff00000000"
    )


def _ordered_dictionary_file() raises -> List[UInt8]:
    """794 bytes from pyarrow 25."""
    return _from_hex(
        "4152524f57310000ffffffffa00000001000000000000a000c00060005000800"
        "0a000000000104000c0000000800080000000400080000000400000001000000"
        "14000000100018000800060007000c0010001400100000000000011414000000"
        "48000000200000000400000000000000050000006772616465000a000c000000"
        "080007000a000000000000010c00000008000c00080007000800000000000001"
        "08000000040004000400000000000000ffffffffa80000001400000000000000"
        "0c0014000600050008000c000c00000000020400140000003000000000000000"
        "08000a0000000400080000001000000000000a0018000c00040008000a000000"
        "4c00000010000000030000000000000000000000030000000000000000000000"
        "0000000000000000000000000000000020000000000000002000000000000000"
        "0d00000000000000000000000100000003000000000000000000000000000000"
        "0000000000000000030000000000000007000000000000000d00000000000000"
        "6c6f77686967686d656469756d000000ffffffff880000001400000000000000"
        "0c0016000600050008000c000c00000000030400180000000800000000000000"
        "00000a0018000c00040008000a0000003c000000100000000400000000000000"
        "0000000002000000000000000000000000000000000000000000000000000000"
        "0400000000000000000000000100000004000000000000000000000000000000"
        "0001000200000000ffffffff00000000100000000c001400060008000c001000"
        "0c00000000000400500000002800000004000000010000009001000000000000"
        "900000000000000008000000000000000000000001000000b000000000000000"
        "b000000000000000300000000000000008000800000004000800000004000000"
        "0100000014000000100018000800060007000c00100014001000000000000114"
        "1400000048000000200000000400000000000000050000006772616465000a00"
        "0c000000080007000a000000000000010c00000008000c000800070008000000"
        "00000001080000000400040004000000e00000004152524f5731"
    )


def _integer_dictionary_stream() raises -> List[UInt8]:
    """496 bytes from pyarrow 25."""
    return _from_hex(
        "ffffffff900000001000000000000a000c000600050008000a00000000010400"
        "04000000bcffffff040000000100000014000000100018000800060007000c00"
        "100014001000000000000102140000003c0000001c0000000400000000000000"
        "010000006e00000008000800000004000800000004000000f4ffffff00000001"
        "2000000008000c0008000700080000000000000140000000ffffffff98000000"
        "14000000000000000c0014000600050008000c000c0000000002040014000000"
        "100000000000000008000a0000000400080000001000000000000a0018000c00"
        "040008000a0000003c0000001000000002000000000000000000000002000000"
        "0000000000000000000000000000000000000000000000001000000000000000"
        "0000000001000000020000000000000000000000000000000a00000000000000"
        "1400000000000000ffffffff8800000014000000000000000c00160006000500"
        "08000c000c0000000003040018000000100000000000000000000a0018000c00"
        "040008000a0000003c0000001000000003000000000000000000000002000000"
        "0000000000000000000000000000000000000000000000000c00000000000000"
        "0000000001000000030000000000000000000000000000000000000001000000"
        "0000000000000000ffffffff00000000"
    )


def _out_of_range_dictionary_stream() raises -> List[UInt8]:
    """520 bytes from pyarrow 25."""
    return _from_hex(
        "ffffffff900000001000000000000a000c000600050008000a00000000010400"
        "04000000bcffffff040000000100000014000000100018000800060007000c00"
        "10001400100000000000010514000000400000001c0000000400000000000000"
        "01000000640000000800080000000400080000000c00000008000c0008000700"
        "080000000000000120000000040004000400000000000000ffffffffa8000000"
        "14000000000000000c0014000600050008000c000c0000000002040014000000"
        "180000000000000008000a0000000400080000001000000000000a0018000c00"
        "040008000a0000004c0000001000000002000000000000000000000003000000"
        "0000000000000000000000000000000000000000000000000c00000000000000"
        "1000000000000000020000000000000000000000010000000200000000000000"
        "0000000000000000000000000100000002000000000000006162000000000000"
        "ffffffff8800000014000000000000000c0016000600050008000c000c000000"
        "0003040018000000100000000000000000000a0018000c00040008000a000000"
        "3c00000010000000030000000000000000000000020000000000000000000000"
        "000000000000000000000000000000000c000000000000000000000001000000"
        "0300000000000000000000000000000000000000050000000100000000000000"
        "ffffffff00000000"
    )


def _delta_dictionary_stream() raises -> List[UInt8]:
    """872 bytes from pyarrow 25."""
    return _from_hex(
        "ffffffff900000001000000000000a000c000600050008000a00000000010400"
        "04000000bcffffff040000000100000014000000100018000800060007000c00"
        "10001400100000000000010514000000400000001c0000000400000000000000"
        "01000000640000000800080000000400080000000c00000008000c0008000700"
        "080000000000000120000000040004000400000000000000ffffffffa8000000"
        "14000000000000000c0014000600050008000c000c0000000002040014000000"
        "180000000000000008000a0000000400080000001000000000000a0018000c00"
        "040008000a0000004c0000001000000002000000000000000000000003000000"
        "0000000000000000000000000000000000000000000000000c00000000000000"
        "1000000000000000020000000000000000000000010000000200000000000000"
        "0000000000000000000000000100000002000000000000007879000000000000"
        "ffffffff8800000014000000000000000c0016000600050008000c000c000000"
        "0003040018000000080000000000000000000a0018000c00040008000a000000"
        "3c00000010000000020000000000000000000000020000000000000000000000"
        "0000000000000000000000000000000008000000000000000000000001000000"
        "020000000000000000000000000000000000000001000000ffffffffb0000000"
        "14000000000000000c0016000600050008000c000c0000000002040018000000"
        "100000000000000000000a000e000000080007000a0000000000000110000000"
        "00000a0018000c00040008000a0000004c000000100000000100000000000000"
        "0000000003000000000000000000000000000000000000000000000000000000"
        "0800000000000000080000000000000001000000000000000000000001000000"
        "0100000000000000000000000000000000000000010000007a00000000000000"
        "ffffffff8800000014000000000000000c0016000600050008000c000c000000"
        "0003040018000000100000000000000000000a0018000c00040008000a000000"
        "3c00000010000000030000000000000000000000020000000000000000000000"
        "000000000000000000000000000000000c000000000000000000000001000000"
        "0300000000000000000000000000000000000000010000000200000000000000"
        "ffffffff00000000"
    )


def _large_string_stream() raises -> List[UInt8]:
    """368 bytes from pyarrow 25."""
    return _from_hex(
        "ffffffff700000001000000000000a000c000600050008000a00000000010400"
        "0c00000008000800000004000800000004000000010000001400000010001400"
        "0800060007000c00000010001000000000000114100000001800000004000000"
        "000000000100000074000000040004000400000000000000ffffffff98000000"
        "14000000000000000c0016000600050008000c000c0000000003040018000000"
        "500000000000000000000a0018000c00040008000a0000004c00000010000000"
        "0300000000000000000000000300000000000000000000000100000000000000"
        "0800000000000000200000000000000028000000000000002500000000000000"
        "0000000001000000030000000000000001000000000000000500000000000000"
        "0000000000000000050000000000000005000000000000002500000000000000"
        "616c706861616e206576656e206c6f6e67657220737472696e672076616c7565"
        "2068657265000000ffffffff00000000"
    )


def _timestamp_stream() raises -> List[UInt8]:
    """680 bytes from pyarrow 25. Four columns, one instant, four units."""
    return _from_hex(
        "fffffffff80000001000000000000a000c000600050008000a00000000010400"
        "0c0000000800080000000400080000000400000004000000a00000005c000000"
        "300000000400000080ffffff0000010a10000000140000000400000000000000"
        "020000006e730000aeffffff00000300a8ffffff0000010a1000000014000000"
        "04000000000000000200000075730000d6ffffff00000200d0ffffff0000010a"
        "100000001c0000000400000000000000020000006d7300000000060008000600"
        "0600000000000100100014000800060007000c0000001000100000000000010a"
        "1000000018000000040000000000000001000000730000000400040004000000"
        "ffffffff1801000014000000000000000c0016000600050008000c000c000000"
        "0003040018000000800000000000000000000a0018000c00040008000a000000"
        "9c00000010000000030000000000000000000000080000000000000000000000"
        "0100000000000000080000000000000018000000000000002000000000000000"
        "0100000000000000280000000000000018000000000000004000000000000000"
        "0100000000000000480000000000000018000000000000006000000000000000"
        "0100000000000000680000000000000018000000000000000000000004000000"
        "0300000000000000010000000000000003000000000000000100000000000000"
        "0300000000000000010000000000000003000000000000000100000000000000"
        "0500000000000000450ded65000000000000000000000000ffffffffffffffff"
        "050000000000000088d5fb258e010000000000000000000018fcffffffffffff"
        "0500000000000000401bba5f441306000000000000000000c0bdf0ffffffffff"
        "05000000000000000072faee1543bb170000000000000000003665c4ffffffff"
        "ffffffff00000000"
    )


def _zoned_stream() raises -> List[UInt8]:
    """472 bytes from pyarrow 25. The same integers naive and in New York."""
    return _from_hex(
        "ffffffffc80000001000000000000a000c000600050008000a00000000010400"
        "0c00000008000800000004000800000004000000020000006800000004000000"
        "b0ffffff0000010a10000000200000000400000000000000050000007a6f6e65"
        "6400000008000c000600080008000000000002000400000010000000416d6572"
        "6963612f4e65775f596f726b00000000100014000800060007000c0000001000"
        "100000000000010a100000001c0000000400000000000000050000006e616976"
        "65000600080006000600000000000200ffffffffb80000001400000000000000"
        "0c0016000600050008000c000c00000000030400180000004000000000000000"
        "00000a0018000c00040008000a0000005c000000100000000300000000000000"
        "0000000004000000000000000000000001000000000000000800000000000000"
        "1800000000000000200000000000000001000000000000002800000000000000"
        "1800000000000000000000000200000003000000000000000100000000000000"
        "030000000000000001000000000000000500000000000000401bba5f44130600"
        "0000000000000000c0bdf0ffffffffff0500000000000000401bba5f44130600"
        "0000000000000000c0bdf0ffffffffffffffffff00000000"
    )


def _date_stream() raises -> List[UInt8]:
    """304 bytes from pyarrow 25. One date32 column with a null in it."""
    return _from_hex(
        "ffffffff780000001000000000000a000c000600050008000a00000000010400"
        "0c00000008000800000004000800000004000000010000001400000010001400"
        "0800060007000c00000010001000000000000108100000001c00000004000000"
        "0000000003000000646179000000060008000600060000000000000000000000"
        "ffffffff8800000014000000000000000c0016000600050008000c000c000000"
        "0003040018000000180000000000000000000a0018000c00040008000a000000"
        "3c00000010000000030000000000000000000000020000000000000000000000"
        "010000000000000008000000000000000c000000000000000000000001000000"
        "030000000000000001000000000000000500000000000000504d000000000000"
        "ffffffff00000000ffffffff00000000"
    )


def _date64_stream() raises -> List[UInt8]:
    """304 bytes from pyarrow 25. The same three days, counted in milliseconds.
    """
    return _from_hex(
        "ffffffff700000001000000000000a000c000600050008000a00000000010400"
        "0c00000008000800000004000800000004000000010000001400000010001400"
        "0800060007000c00000010001000000000000108100000001800000004000000"
        "000000000300000064617900040004000400000000000000ffffffff88000000"
        "14000000000000000c0016000600050008000c000c0000000003040018000000"
        "200000000000000000000a0018000c00040008000a0000003c00000010000000"
        "0300000000000000000000000200000000000000000000000100000000000000"
        "0800000000000000180000000000000000000000010000000300000000000000"
        "0100000000000000050000000000000000c0a8258e0100000000000000000000"
        "00a4d9faffffffffffffffff00000000"
    )


def _duration_stream() raises -> List[UInt8]:
    """288 bytes from pyarrow 25. One microsecond, as a duration."""
    return _from_hex(
        "ffffffff780000001000000000000a000c000600050008000a00000000010400"
        "0c00000008000800000004000800000004000000010000001400000010001400"
        "0800060007000c00000010001000000000000112100000001c00000004000000"
        "000000000500000076616c756500060008000600060000000000020000000000"
        "ffffffff8800000014000000000000000c0016000600050008000c000c000000"
        "0003040018000000080000000000000000000a0018000c00040008000a000000"
        "3c00000010000000010000000000000000000000020000000000000000000000"
        "0000000000000000000000000000000008000000000000000000000001000000"
        "010000000000000000000000000000000100000000000000ffffffff00000000"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
