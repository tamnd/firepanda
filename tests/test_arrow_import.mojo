"""Tests for taking a column from a C producer.

Most of these export a column and import it back, which is the only way to get a
conforming producer without linking one in, and it has the useful property that a
mistake made symmetrically in both halves would still fail: the assertions are
against the values that went in, not against what the exporter said about them.

The tests that matter most are the ones that do not round trip. A producer that
is not firepanda hands over slices with a nonzero offset, offset based string
buffers rather than views, and view arrays with more than one data buffer, and
none of those shapes can be produced by our own exporter. Those are built by hand
here, out of raw memory laid out the way the specification says, because a shape
firepanda cannot construct is exactly the shape a bug hides in.
"""

from std.ffi import c_char, external_call
from std.memory import unsafe_memcpy
from std.sys import size_of
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder, strings_from_list
from firepanda.array.strview import StringView, make_inline, make_long
from firepanda.dtype.logical import LogicalType
from firepanda.io.arrow_c import (
    ArrayPtr,
    ArrowArray,
    ArrowSchema,
    NullableVoidPtr,
    SchemaPtr,
    VoidPtr,
    array_release_callback,
    release_array,
    release_schema,
    schema_release_callback,
)
from firepanda.io.arrow_export import export_array, export_schema
from firepanda.io.arrow_import import import_array


def _numbers(count: Int) raises -> Array[DType.int64]:
    """Builds an int64 column of 100, 200, 300 and so on, with no nulls."""
    var out = Array[DType.int64](count)
    for i in range(count):
        out.set_valid(i, Int64((i + 1) * 100))
    return out^


def _round_trip(var column: AnyArray) raises -> AnyArray:
    """Exports a column and imports the result back."""
    var type = column.type
    var schema = export_schema(type)
    var array = export_array(column^)
    return import_array(schema, array)


def test_an_int64_column_round_trips() raises:
    var out = _round_trip(AnyArray(_numbers(9)))
    assert_equal(len(out), 9)
    assert_equal(out.type, LogicalType.INT64)
    assert_equal(out.null_count(), 0)
    var values = out.unsafe_ptr[DType.int64]()
    for i in range(9):
        assert_equal(
            values.unsafe_offset(i).unsafe_load(), Int64((i + 1) * 100)
        )


def test_the_imported_column_owns_its_memory() raises:
    # The import copies, so the values it hands back must not be the exporter's.
    # If they ever were, the release the importer performs would have freed them
    # and this test would be reading memory that no longer belongs to anyone.
    var column = _numbers(16)
    var before = Int(column.data.values.unsafe_ptr())

    var out = _round_trip(AnyArray(column^))
    assert_true(Int(out.data.values.unsafe_ptr()) != before)
    assert_equal(out.unsafe_ptr[DType.int64]().unsafe_load(), Int64(100))


def test_nulls_round_trip() raises:
    var column = _numbers(20)
    column.set_null(3)
    column.set_null(17)

    var out = _round_trip(AnyArray(column^))
    assert_equal(out.null_count(), 2)
    for i in range(20):
        assert_equal(out.is_valid(i), i != 3 and i != 17)


def test_every_fixed_width_type_round_trips() raises:
    var a = _round_trip(AnyArray(Array[DType.int8](4)))
    assert_equal(a.type, LogicalType.INT8)
    assert_equal(len(a), 4)
    var b = _round_trip(AnyArray(Array[DType.uint16](4)))
    assert_equal(b.type, LogicalType.UINT16)
    var c = _round_trip(AnyArray(Array[DType.float32](4)))
    assert_equal(c.type, LogicalType.FLOAT32)
    var d = _round_trip(AnyArray(Array[DType.uint64](4)))
    assert_equal(d.type, LogicalType.UINT64)


def test_an_empty_column_round_trips() raises:
    var out = _round_trip(AnyArray(_numbers(0)))
    assert_equal(len(out), 0)
    assert_equal(out.type, LogicalType.INT64)


def test_a_bool_column_round_trips() raises:
    # Packed on the way out, unpacked on the way in. The only type where both
    # halves do real work, so the only one where a shift in the wrong direction
    # would cancel out and still be wrong.
    var column = Array[DType.bool](19)
    for i in range(19):
        column.set_valid(i, i % 3 == 0)

    var out = _round_trip(AnyArray(column^))
    assert_equal(out.type, LogicalType.BOOL)
    assert_equal(len(out), 19)
    var values = out.unsafe_ptr[DType.bool]()
    for i in range(19):
        assert_equal(values.unsafe_offset(i).unsafe_load(), i % 3 == 0)


def test_a_string_column_round_trips() raises:
    var expected: List[String] = [
        String("hi"),
        String("a string much longer than twelve bytes"),
        String(""),
        String("also quite long, well past the inline limit"),
    ]
    var out = _round_trip(AnyArray(strings_from_list(expected)))
    assert_equal(out.type, LogicalType.STRING)
    assert_equal(len(out), 4)
    ref text = out.strings()
    for i in range(4):
        assert_equal(text[i], expected[i])


def test_a_string_column_with_nulls_round_trips() raises:
    var first = String("first")
    var third = String("a third element longer than the inline limit")
    var builder = StringBuilder(capacity=3)
    builder.append(first.as_bytes())
    builder.append_null()
    builder.append(third.as_bytes())

    var out = _round_trip(AnyArray(builder^.finish()))
    assert_equal(out.null_count(), 1)
    assert_false(out.is_valid(1))
    ref text = out.strings()
    assert_equal(text[0], "first")
    assert_equal(text[2], "a third element longer than the inline limit")


def test_a_released_structure_is_refused() raises:
    var schema = export_schema(LogicalType.INT64)
    var array = export_array(AnyArray(_numbers(2)))
    release_schema(schema)
    with assert_raises(contains="already been released"):
        _ = import_array(schema, array)
    release_array(array)


def test_a_type_firepanda_cannot_read_is_refused() raises:
    # And the structures are released on the way out, which valgrind is what
    # actually checks. What is checked here is that the refusal happens at all
    # and names the reason rather than the format string.
    var schema = export_schema(LogicalType.INT64)
    var array = export_array(AnyArray(_numbers(2)))
    array.n_children = 1
    with assert_raises(contains="nested columns"):
        _ = import_array(schema, array)


def test_a_wrong_buffer_count_is_refused() raises:
    var schema = export_schema(LogicalType.INT64)
    var array = export_array(AnyArray(_numbers(2)))
    array.n_buffers = 3
    with assert_raises(contains="needs 2 buffers"):
        _ = import_array(schema, array)


# The producers below are built by hand rather than by the exporter, because the
# shapes they produce are ones firepanda never emits. Each one owns nothing: the
# storage is a local in the test and the release callback only clears the
# structure, which is a conforming producer as far as the interface is concerned.


def _release_borrowed(array: ArrayPtr) abi("C") -> None:
    """Releases a hand built array whose buffers belong to the test."""
    array[].release = None


def _release_borrowed_schema(schema: SchemaPtr) abi("C") -> None:
    """Releases a hand built schema, freeing the format string it owns."""
    if schema[].format:
        external_call["free", NoneType](schema[].format.value())
        schema[].format = None
    schema[].release = None


def _as_void[o: MutOrigin](p: Pointer[UInt8, o]) -> VoidPtr:
    """Reinterprets a byte pointer as the `void*` a buffer array holds."""
    return p.unsafe_origin_cast[MutUntrackedOrigin]().unsafe_bitcast[NoneType]()


def test_a_slice_with_a_byte_aligned_offset_is_read_from_its_start() raises:
    # A producer that hands over rows eight onwards of its own column. Arrow says
    # the values and the validity bits are both indexed from `offset`, and a
    # reader that ignored it would return the first rows of somebody else's data
    # while reporting the right length.
    var values = InlineArray[Int64, 12](uninitialized=True)
    for i in range(12):
        values[i] = Int64(i)
    var valid = InlineArray[UInt8, 2](fill=0xFF)
    var buffers = InlineArray[NullableVoidPtr, 2](fill=None)
    buffers[0] = _as_void(
        Pointer(to=valid[0]).unsafe_origin_cast[MutUntrackedOrigin]()
    )
    buffers[1] = _as_void(
        Pointer(to=values[0])
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[UInt8]()
    )

    var array = ArrowArray()
    array.length = 4
    array.null_count = 0
    array.offset = 8
    array.n_buffers = 2
    array.buffers = Pointer(to=buffers[0]).unsafe_origin_cast[
        MutUntrackedOrigin
    ]()
    array.release = array_release_callback(_release_borrowed)

    var schema = export_schema(LogicalType.INT64)
    var out = import_array(schema, array)
    assert_equal(len(out), 4)
    var got = out.unsafe_ptr[DType.int64]()
    for i in range(4):
        assert_equal(got.unsafe_offset(i).unsafe_load(), Int64(8 + i))
    # Mojo destroys a value at its last use, and the last use of the storage
    # above is the line that took a pointer to it. Without these the string's
    # heap allocation is freed before the import walks it, which is a use after
    # free that a hand built producer runs into and a real one never would,
    # because a real one owns its buffers for as long as the structure lives.
    _ = values^
    _ = valid^
    _ = buffers^


def test_a_slice_with_an_unaligned_offset_shifts_the_validity_bits() raises:
    # Offset three, which is the case the byte copy cannot take. Rows three, five
    # and eight of the producer are null, so rows zero, two and five of what
    # comes out should be.
    var values = InlineArray[Int64, 16](uninitialized=True)
    for i in range(16):
        values[i] = Int64(i * 10)
    var valid = InlineArray[UInt8, 2](fill=0xFF)
    valid[0] = 0xFF & ~UInt8(1 << 3) & ~UInt8(1 << 5)
    valid[1] = 0xFF & ~UInt8(1 << 0)
    var buffers = InlineArray[NullableVoidPtr, 2](fill=None)
    buffers[0] = _as_void(
        Pointer(to=valid[0]).unsafe_origin_cast[MutUntrackedOrigin]()
    )
    buffers[1] = _as_void(
        Pointer(to=values[0])
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[UInt8]()
    )

    var array = ArrowArray()
    array.length = 9
    array.null_count = 3
    array.offset = 3
    array.n_buffers = 2
    array.buffers = Pointer(to=buffers[0]).unsafe_origin_cast[
        MutUntrackedOrigin
    ]()
    array.release = array_release_callback(_release_borrowed)

    var schema = export_schema(LogicalType.INT64)
    var out = import_array(schema, array)
    assert_equal(len(out), 9)
    assert_equal(out.null_count(), 3)
    var got = out.unsafe_ptr[DType.int64]()
    for i in range(9):
        assert_equal(got.unsafe_offset(i).unsafe_load(), Int64((3 + i) * 10))
        assert_equal(out.is_valid(i), i != 0 and i != 2 and i != 5)
    # Mojo destroys a value at its last use, and the last use of the storage
    # above is the line that took a pointer to it. Without these the string's
    # heap allocation is freed before the import walks it, which is a use after
    # free that a hand built producer runs into and a real one never would,
    # because a real one owns its buffers for as long as the structure lives.
    _ = values^
    _ = valid^
    _ = buffers^


def test_the_offset_based_string_layout_is_read() raises:
    # Format "u": one data buffer and an int32 offset per element plus one. This
    # is what pyarrow produces unless it is asked for views, so it is the import
    # path that will get the most use and the one firepanda can never produce.
    var data = String("alphabetaan element well past the inline limit")
    var offsets = InlineArray[Int32, 4](uninitialized=True)
    offsets[0] = 0
    offsets[1] = 5
    offsets[2] = 9
    offsets[3] = Int32(data.byte_length())

    var buffers = InlineArray[NullableVoidPtr, 3](fill=None)
    buffers[1] = _as_void(
        Pointer(to=offsets[0])
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[UInt8]()
    )
    buffers[2] = _as_void(
        data.as_bytes().unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )

    var array = ArrowArray()
    array.length = 3
    array.null_count = 0
    array.n_buffers = 3
    array.buffers = Pointer(to=buffers[0]).unsafe_origin_cast[
        MutUntrackedOrigin
    ]()
    array.release = array_release_callback(_release_borrowed)

    var schema = ArrowSchema()
    schema.format = _u_format()
    schema.release = schema_release_callback(_release_borrowed_schema)
    var out = import_array(schema, array)

    assert_equal(out.type, LogicalType.STRING)
    assert_equal(len(out), 3)
    ref text = out.strings()
    assert_equal(text[0], "alpha")
    assert_equal(text[1], "beta")
    assert_equal(text[2], "an element well past the inline limit")
    # Mojo destroys a value at its last use, and the last use of the storage
    # above is the line that took a pointer to it. Without these the string's
    # heap allocation is freed before the import walks it, which is a use after
    # free that a hand built producer runs into and a real one never would,
    # because a real one owns its buffers for as long as the structure lives.
    _ = data^
    _ = offsets^
    _ = buffers^


def test_a_view_array_with_two_data_buffers_is_read() raises:
    # firepanda has exactly one payload block and Arrow lets a producer have any
    # number, so a long element's view names which one. This is the shape that
    # forces the import to rebuild the views rather than copy them, and it is a
    # shape our own exporter can never produce, so it is built by hand.
    var first = String("the first data buffer holds this one")
    var second = String("and the second data buffer holds this other one")
    var views = InlineArray[StringView, 3](fill=StringView())
    views[0] = make_inline(String("short").as_bytes())
    views[1] = make_long(first.as_bytes(), 0, 0)
    views[2] = make_long(second.as_bytes(), 1, 0)
    var sizes = InlineArray[Int64, 2](uninitialized=True)
    sizes[0] = Int64(first.byte_length())
    sizes[1] = Int64(second.byte_length())

    var buffers = InlineArray[NullableVoidPtr, 5](fill=None)
    buffers[1] = _as_void(
        Pointer(to=views[0])
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[UInt8]()
    )
    buffers[2] = _as_void(
        first.as_bytes().unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )
    buffers[3] = _as_void(
        second.as_bytes().unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )
    buffers[4] = _as_void(
        Pointer(to=sizes[0])
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[UInt8]()
    )

    var array = ArrowArray()
    array.length = 3
    array.null_count = 0
    array.n_buffers = 5
    array.buffers = Pointer(to=buffers[0]).unsafe_origin_cast[
        MutUntrackedOrigin
    ]()
    array.release = array_release_callback(_release_borrowed)

    var schema = export_schema(LogicalType.STRING)
    var out = import_array(schema, array)
    ref text = out.strings()
    assert_equal(text[0], "short")
    assert_equal(text[1], first)
    assert_equal(text[2], second)
    # Everything now lives in one block, which is the whole point of the rebuild.
    assert_equal(
        len(out.text.value().payload),
        first.byte_length() + second.byte_length(),
    )
    _ = views^
    _ = sizes^
    _ = buffers^
    _ = first^
    _ = second^


def test_a_view_naming_a_data_buffer_that_is_not_there_is_refused() raises:
    var long = String("long enough to need a data buffer")
    var views = InlineArray[StringView, 1](fill=StringView())
    views[0] = make_long(long.as_bytes(), 3, 0)
    var sizes = InlineArray[Int64, 1](fill=Int64(long.byte_length()))

    var buffers = InlineArray[NullableVoidPtr, 4](fill=None)
    buffers[1] = _as_void(
        Pointer(to=views[0])
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[UInt8]()
    )
    buffers[2] = _as_void(
        long.as_bytes().unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )
    buffers[3] = _as_void(
        Pointer(to=sizes[0])
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[UInt8]()
    )

    var array = ArrowArray()
    array.length = 1
    array.null_count = 0
    array.n_buffers = 4
    array.buffers = Pointer(to=buffers[0]).unsafe_origin_cast[
        MutUntrackedOrigin
    ]()
    array.release = array_release_callback(_release_borrowed)

    var schema = export_schema(LogicalType.STRING)
    with assert_raises(contains="names data buffer 3"):
        _ = import_array(schema, array)
    _ = views^
    _ = sizes^
    _ = buffers^
    _ = long^


def test_a_view_that_runs_past_its_data_buffer_is_refused() raises:
    # A view is three numbers chosen by somebody else, and following one without
    # checking it is an out of bounds read waiting for a malformed file. The
    # sizes buffer is what makes the check possible.
    var long = String("long enough to need a data buffer")
    var views = InlineArray[StringView, 1](fill=StringView())
    views[0] = make_long(long.as_bytes(), 0, 8)
    var sizes = InlineArray[Int64, 1](fill=Int64(long.byte_length()))

    var buffers = InlineArray[NullableVoidPtr, 4](fill=None)
    buffers[1] = _as_void(
        Pointer(to=views[0])
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[UInt8]()
    )
    buffers[2] = _as_void(
        long.as_bytes().unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )
    buffers[3] = _as_void(
        Pointer(to=sizes[0])
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[UInt8]()
    )

    var array = ArrowArray()
    array.length = 1
    array.null_count = 0
    array.n_buffers = 4
    array.buffers = Pointer(to=buffers[0]).unsafe_origin_cast[
        MutUntrackedOrigin
    ]()
    array.release = array_release_callback(_release_borrowed)

    var schema = export_schema(LogicalType.STRING)
    with assert_raises(contains="runs past the end"):
        _ = import_array(schema, array)
    _ = views^
    _ = sizes^
    _ = buffers^
    _ = long^


def test_offsets_that_go_backwards_are_refused() raises:
    # A negative element length is an enormous one once it reaches a memcpy, so
    # this is the one check the offset layout allows and needs.
    var data = String("alphabeta")
    var offsets = InlineArray[Int32, 3](uninitialized=True)
    offsets[0] = 0
    offsets[1] = 9
    offsets[2] = 5

    var buffers = InlineArray[NullableVoidPtr, 3](fill=None)
    buffers[1] = _as_void(
        Pointer(to=offsets[0])
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[UInt8]()
    )
    buffers[2] = _as_void(
        data.as_bytes().unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )

    var array = ArrowArray()
    array.length = 2
    array.null_count = 0
    array.n_buffers = 3
    array.buffers = Pointer(to=buffers[0]).unsafe_origin_cast[
        MutUntrackedOrigin
    ]()
    array.release = array_release_callback(_release_borrowed)

    var schema = ArrowSchema()
    schema.format = _u_format()
    schema.release = schema_release_callback(_release_borrowed_schema)
    with assert_raises(contains="offsets go backwards"):
        _ = import_array(schema, array)
    _ = data^
    _ = offsets^
    _ = buffers^


def _u_format() -> Optional[Pointer[c_char, MutUntrackedOrigin]]:
    """Returns a static, null terminated "u" for a hand built schema.

    Heap allocated, and the schema's release callback frees it. Two bytes is a
    small enough leak to be tempting to ignore, and ignoring it would mean the
    leak check on this file could no longer be read as a plain zero.
    """
    var p = external_call["malloc", Pointer[c_char, MutUntrackedOrigin]](2)
    p.unsafe_write(c_char(ord("u")))
    p.unsafe_offset(1).unsafe_write(c_char(0))
    return p


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
