"""Tests for the FlatBuffers reader and builder.

Most of these build something and read it back, which proves the two halves
agree and nothing else. Two agreeing halves that are both wrong is exactly the
failure mode for a wire format, so the test that carries the weight here is
`test_a_real_arrow_schema_message_decodes`: 176 bytes produced by pyarrow 25,
checked in as they came off the wire, read with nothing but the functions in
this file. If the reader has misunderstood the layout, that test says so and the
round trips do not.

The rest is about the parts a round trip cannot reach. A field the writer left
out has to read back as its default rather than as whatever byte follows, and a
field past the end of a short vtable has to do the same, because that is the
case a reader hits when it is newer than the file it was given. And every read
of a truncated or self referential buffer has to raise rather than wander off,
because the buffer came from somewhere else.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.io.flatbuf import (
    Builder,
    field_scalar,
    field_string,
    field_table,
    field_vector,
    has_field,
    read_scalar,
    root_table,
    string_at,
    vector_element,
    vector_length,
    vector_table,
)


def _from_hex(text: StringSlice) raises -> List[UInt8]:
    """Decodes a hex string into bytes, so a wire capture can be checked in."""
    var digits = text.as_bytes()
    var out = List[UInt8]()
    for i in range(0, len(digits), 2):
        var high = _nibble(digits[i])
        var low = _nibble(digits[i + 1])
        out.append(UInt8(high * 16 + low))
    return out^


def _nibble(byte: UInt8) raises -> Int:
    var value = Int(byte)
    if value >= ord("0") and value <= ord("9"):
        return value - ord("0")
    if value >= ord("a") and value <= ord("f"):
        return value - ord("a") + 10
    raise Error(String("not a hex digit: ", value))


def test_a_table_of_scalars_round_trips() raises:
    var builder = Builder()
    builder.start_table(4)
    builder.add_scalar[DType.int32](0, 42, 0)
    builder.add_scalar[DType.int64](1, -9_000_000_000, 0)
    builder.add_scalar[DType.uint8](2, 200, 0)
    builder.add_scalar[DType.float64](3, 1.5, 0.0)
    var root = builder.end_table()
    var bytes = builder.finish(root)

    var data = Span(bytes)
    var table = root_table(data)
    assert_equal(field_scalar[DType.int32](data, table, 0, 0), 42)
    assert_equal(field_scalar[DType.int64](data, table, 1, 0), -9_000_000_000)
    assert_equal(field_scalar[DType.uint8](data, table, 2, 0), 200)
    assert_equal(field_scalar[DType.float64](data, table, 3, 0.0), 1.5)


def test_a_field_equal_to_its_default_is_not_written() raises:
    # The definition of a default rather than a size trick: the writer omits it
    # and the reader supplies it, and the two have to agree about which value
    # that is.
    var builder = Builder()
    builder.start_table(2)
    builder.add_scalar[DType.int32](0, 7, 7)
    builder.add_scalar[DType.int32](1, 8, 7)
    var root = builder.end_table()
    var bytes = builder.finish(root)

    var data = Span(bytes)
    var table = root_table(data)
    assert_false(has_field(data, table, 0))
    assert_true(has_field(data, table, 1))
    assert_equal(field_scalar[DType.int32](data, table, 0, 7), 7)
    assert_equal(field_scalar[DType.int32](data, table, 1, 7), 8)


def test_a_field_past_the_end_of_the_vtable_reads_as_its_default() raises:
    # This is the whole point of the format. A reader that knows about six
    # fields has to keep working against a file written by something that knew
    # about two, and it finds out by running off the end of the vtable.
    var builder = Builder()
    builder.start_table(2)
    builder.add_scalar[DType.int32](0, 1, 0)
    var root = builder.end_table()
    var bytes = builder.finish(root)

    var data = Span(bytes)
    var table = root_table(data)
    assert_false(has_field(data, table, 5))
    assert_equal(field_scalar[DType.int32](data, table, 5, 99), 99)


def test_trailing_empty_slots_are_trimmed() raises:
    # A table started with room for many fields and given one costs the same as
    # a table started with room for one, which matters for a schema with a long
    # tail of options nobody sets.
    var wide = Builder()
    wide.start_table(20)
    wide.add_scalar[DType.int32](0, 5, 0)
    var wide_bytes = wide.finish(wide.end_table())

    var narrow = Builder()
    narrow.start_table(1)
    narrow.add_scalar[DType.int32](0, 5, 0)
    var narrow_bytes = narrow.finish(narrow.end_table())

    assert_equal(len(wide_bytes), len(narrow_bytes))


def test_a_string_field_round_trips() raises:
    var builder = Builder()
    var name = builder.create_string("quantity")
    builder.start_table(2)
    builder.add_offset(0, name)
    builder.add_scalar[DType.bool](1, True, False)
    var bytes = builder.finish(builder.end_table())

    var data = Span(bytes)
    var table = root_table(data)
    assert_equal(field_string(data, table, 0), "quantity")
    assert_equal(field_scalar[DType.bool](data, table, 1, False), True)


def test_an_absent_string_reads_as_empty() raises:
    var builder = Builder()
    builder.start_table(1)
    var bytes = builder.finish(builder.end_table())

    var data = Span(bytes)
    assert_equal(field_string(data, root_table(data), 0), "")


def test_an_empty_string_is_written_and_read() raises:
    var builder = Builder()
    var empty = builder.create_string("")
    builder.start_table(1)
    builder.add_offset(0, empty)
    var bytes = builder.finish(builder.end_table())

    var data = Span(bytes)
    var table = root_table(data)
    assert_true(has_field(data, table, 0))
    assert_equal(field_string(data, table, 0), "")


def test_a_vector_of_scalars_round_trips() raises:
    var builder = Builder()
    builder.start_vector(8, 3, 8)
    # Written back to front, because the whole buffer is.
    builder.prepend[DType.int64](30)
    builder.prepend[DType.int64](20)
    builder.prepend[DType.int64](10)
    var vector = builder.end_vector(3)
    builder.start_table(1)
    builder.add_offset(0, vector)
    var bytes = builder.finish(builder.end_table())

    var data = Span(bytes)
    var found = field_vector(data, root_table(data), 0)
    assert_equal(vector_length(data, found), 3)
    for i in range(3):
        assert_equal(
            read_scalar[DType.int64](data, vector_element(data, found, i, 8)),
            Int64((i + 1) * 10),
        )


def test_a_vector_of_bytes_round_trips() raises:
    var payload: List[UInt8] = [1, 2, 3, 250]
    var builder = Builder()
    var vector = builder.create_bytes(Span(payload))
    builder.start_table(1)
    builder.add_offset(0, vector)
    var bytes = builder.finish(builder.end_table())

    var data = Span(bytes)
    var found = field_vector(data, root_table(data), 0)
    assert_equal(vector_length(data, found), 4)
    assert_equal(
        read_scalar[DType.uint8](data, vector_element(data, found, 3, 1)),
        UInt8(250),
    )


def test_a_vector_of_tables_round_trips() raises:
    # The shape a schema is: a table holding a vector of child tables, each with
    # a name and a number.
    var builder = Builder()
    var names: List[String] = ["alpha", "beta", "gamma"]
    var children = List[Int]()
    for i in range(len(names)):
        var text = builder.create_string(names[i])
        builder.start_table(2)
        builder.add_offset(0, text)
        builder.add_scalar[DType.int32](1, Int32(i + 1), 0)
        children.append(builder.end_table())
    var vector = builder.create_offsets(children)
    builder.start_table(1)
    builder.add_offset(0, vector)
    var bytes = builder.finish(builder.end_table())

    var data = Span(bytes)
    var found = field_vector(data, root_table(data), 0)
    assert_equal(vector_length(data, found), 3)
    for i in range(3):
        var child = vector_table(data, found, i)
        assert_equal(field_string(data, child, 0), names[i])
        assert_equal(field_scalar[DType.int32](data, child, 1, 0), Int32(i + 1))


def test_identical_tables_share_one_vtable() raises:
    # Not a size micro-optimization. A schema of a hundred columns writes a
    # hundred `Field` tables with the same shape, and the sharing is what keeps
    # the metadata proportional to the names rather than to the layout.
    var builder = Builder()
    builder.start_table(2)
    builder.add_scalar[DType.int32](0, 1, 0)
    builder.add_scalar[DType.int32](1, 2, 0)
    var first = builder.end_table()
    builder.start_table(2)
    builder.add_scalar[DType.int32](0, 3, 0)
    builder.add_scalar[DType.int32](1, 4, 0)
    var second = builder.end_table()
    builder.start_table(2)
    builder.add_offset(0, first)
    builder.add_offset(1, second)
    var bytes = builder.finish(builder.end_table())

    var data = Span(bytes)
    var root = root_table(data)
    var a = field_table(data, root, 0)
    var b = field_table(data, root, 1)
    assert_true(a != b)
    var vtable_a = a - Int(read_scalar[DType.int32](data, a))
    var vtable_b = b - Int(read_scalar[DType.int32](data, b))
    assert_equal(vtable_a, vtable_b)
    assert_equal(field_scalar[DType.int32](data, a, 0, 0), 1)
    assert_equal(field_scalar[DType.int32](data, b, 0, 0), 3)


def test_tables_with_different_shapes_do_not_share() raises:
    var builder = Builder()
    builder.start_table(2)
    builder.add_scalar[DType.int32](0, 1, 0)
    var first = builder.end_table()
    builder.start_table(2)
    builder.add_scalar[DType.int32](1, 1, 0)
    var second = builder.end_table()
    builder.start_table(2)
    builder.add_offset(0, first)
    builder.add_offset(1, second)
    var bytes = builder.finish(builder.end_table())

    var data = Span(bytes)
    var root = root_table(data)
    var a = field_table(data, root, 0)
    var b = field_table(data, root, 1)
    var vtable_a = a - Int(read_scalar[DType.int32](data, a))
    var vtable_b = b - Int(read_scalar[DType.int32](data, b))
    assert_true(vtable_a != vtable_b)
    assert_equal(field_scalar[DType.int32](data, a, 0, 0), 1)
    assert_equal(field_scalar[DType.int32](data, a, 1, 0), 0)
    assert_equal(field_scalar[DType.int32](data, b, 0, 0), 0)
    assert_equal(field_scalar[DType.int32](data, b, 1, 0), 1)


def test_a_table_cannot_be_opened_inside_another() raises:
    var builder = Builder()
    builder.start_table(1)
    with assert_raises(contains="cannot be started"):
        builder.start_table(1)


def test_finishing_with_a_table_open_is_refused() raises:
    var builder = Builder()
    builder.start_table(1)
    with assert_raises(contains="still open"):
        _ = builder.finish(4)


def test_a_slot_outside_the_table_is_refused() raises:
    var builder = Builder()
    builder.start_table(2)
    with assert_raises(contains="outside the 2 slots"):
        builder.add_scalar[DType.int32](5, 1, 0)


def test_a_truncated_buffer_raises() raises:
    var builder = Builder()
    builder.start_table(1)
    builder.add_scalar[DType.int64](0, 1234, 0)
    var bytes = builder.finish(builder.end_table())

    var short = List[UInt8]()
    for i in range(len(bytes) - 4):
        short.append(bytes[i])
    with assert_raises(contains="outside the"):
        var data = Span(short)
        var table = root_table(data)
        _ = field_scalar[DType.int64](data, table, 0, 0)


def test_a_root_offset_pointing_past_the_end_raises() raises:
    var bytes: List[UInt8] = [0xFF, 0xFF, 0x00, 0x00, 0, 0, 0, 0]
    with assert_raises(contains="outside the"):
        _ = root_table(Span(bytes))


def test_an_element_past_the_end_of_a_vector_raises() raises:
    var builder = Builder()
    builder.start_vector(4, 2, 4)
    builder.prepend[DType.int32](2)
    builder.prepend[DType.int32](1)
    var vector = builder.end_vector(2)
    builder.start_table(1)
    builder.add_offset(0, vector)
    var bytes = builder.finish(builder.end_table())

    var data = Span(bytes)
    var found = field_vector(data, root_table(data), 0)
    with assert_raises(contains="out of range"):
        _ = vector_element(data, found, 2, 4)


def test_a_real_arrow_schema_message_decodes() raises:
    # 176 bytes off the wire, produced by pyarrow 25 for a table of an int64
    # column "a" and a string column "b". The first eight are the IPC framing,
    # a continuation marker and the metadata length, so the FlatBuffer starts at
    # byte eight and every offset inside it is relative to there. That is the
    # only reason `root_table` takes a start at all.
    var bytes = _from_hex(
        "ffffffffa80000001000000000000a000c000600050008000a00000000010400"
        "0c00000008000800000004000800000004000000020000004000000004000000"
        "d8ffffff00000105100000001800000004000000000000000100000062000000"
        "0400040004000000100014000800060007000c00000010001000000000000102"
        "100000001c0000000400000000000000010000006100000008000c0008000700"
        "08000000000000014000000000000000"
    )
    assert_equal(len(bytes), 176)

    var data = Span(bytes)
    var message = root_table(data, 8)
    # MetadataVersion.V5 and MessageHeader.Schema.
    assert_equal(field_scalar[DType.int16](data, message, 0, 0), 4)
    assert_equal(field_scalar[DType.uint8](data, message, 1, 0), 1)
    assert_equal(field_scalar[DType.int64](data, message, 3, 0), 0)

    var schema = field_table(data, message, 2)
    # Endianness.Little.
    assert_equal(field_scalar[DType.int16](data, schema, 0, 0), 0)

    var fields = field_vector(data, schema, 1)
    assert_equal(vector_length(data, fields), 2)

    var first = vector_table(data, fields, 0)
    assert_equal(field_string(data, first, 0), "a")
    assert_equal(field_scalar[DType.bool](data, first, 1, False), True)
    # Type.Int, then bitWidth and is_signed inside it.
    assert_equal(field_scalar[DType.uint8](data, first, 2, 0), 2)
    var integer = field_table(data, first, 3)
    assert_equal(field_scalar[DType.int32](data, integer, 0, 0), 64)
    assert_equal(field_scalar[DType.bool](data, integer, 1, False), True)

    var second = vector_table(data, fields, 1)
    assert_equal(field_string(data, second, 0), "b")
    # Type.Utf8, which carries no members of its own.
    assert_equal(field_scalar[DType.uint8](data, second, 2, 0), 5)


def test_a_string_reads_the_same_through_both_paths() raises:
    var builder = Builder()
    var text = builder.create_string("lineitem")
    builder.start_table(1)
    builder.add_offset(0, text)
    var bytes = builder.finish(builder.end_table())

    var data = Span(bytes)
    var table = root_table(data)
    var vector = field_vector(data, table, 0)
    assert_equal(vector_length(data, vector), 8)
    assert_equal(string_at(data, vector), field_string(data, table, 0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
