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

The rest of the tests are the refusals. A list column, a dictionary encoded
column and a file whose magic number is wrong all have to be named rather than
misread, because every one of them is a shape this reader could otherwise walk
into and produce numbers from.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.dtype.logical import LogicalType
from firepanda.io.arrow_ipc import (
    read_arrow,
    read_arrow_bytes,
    read_ipc_file,
    read_ipc_stream,
)


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


def test_a_list_column_is_refused_by_name() raises:
    with assert_raises(contains="nested"):
        _ = read_ipc_stream(Span(_list_stream()))


def test_a_dictionary_encoded_column_is_refused_by_name() raises:
    with assert_raises(contains="dictionary"):
        _ = read_ipc_stream(Span(_dictionary_stream()))


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
