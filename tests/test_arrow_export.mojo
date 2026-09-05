"""Tests for handing a firepanda column to a C consumer.

The one that matters is `test_the_values_buffer_is_not_copied`. It records the
column's values pointer before the export, exports, and asserts that the address
Arrow's `buffers[1]` holds is that same address. Every other way of checking this
compares contents, and contents compare equal for a copy, which is the exact
failure being ruled out.

The rest is ownership. An exported array outlives the scope that made it, holds
pointers into memory firepanda allocated, and is freed by a callback a foreign
runtime invokes. So the tests release things twice, release things that were never
exported, and read values back through the raw `void*` the way a consumer would
rather than through the Mojo column, which would prove nothing about what the
consumer sees.
"""

from std.ffi import c_char
from std.memory import ArcPointer
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
from firepanda.dtype.logical import LogicalType
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.io.arrow_c import (
    ARROW_FLAG_NULLABLE,
    ArrowArray,
    NullableVoidPtr,
    release_array,
    release_schema,
)
from firepanda.io.arrow_export import (
    export_array,
    export_frame_array,
    export_frame_schema,
    export_schema,
)


def _numbers(count: Int) raises -> Array[DType.int64]:
    """Builds an int64 column of 100, 200, 300 and so on, with no nulls."""
    var out = Array[DType.int64](count)
    for i in range(count):
        out.set_valid(i, Int64((i + 1) * 100))
    return out^


def _c_string_at(p: Pointer[c_char, MutUntrackedOrigin]) -> String:
    """Reads a null terminated C string back, the way a consumer would."""
    var out = String()
    var i = 0
    while True:
        var byte = p.unsafe_offset(i).unsafe_load()
        if byte == 0:
            break
        out += chr(Int(byte))
        i += 1
    return out^


def _buffer_at(array: ArrowArray, i: Int) -> NullableVoidPtr:
    """Reads one entry of the exported buffer array."""
    return array.buffers.value().unsafe_offset(i)[]


def test_a_schema_carries_the_format_string() raises:
    var schema = export_schema(LogicalType.INT64, "amount")
    assert_equal(_c_string_at(schema.format.value()), "l")
    assert_equal(_c_string_at(schema.name.value()), "amount")
    assert_equal(schema.flags, ARROW_FLAG_NULLABLE)
    assert_equal(schema.n_children, 0)
    assert_false(schema.is_released())
    release_schema(schema)


def test_a_top_level_schema_has_a_null_name() raises:
    # Arrow distinguishes an absent name from an empty one, and the top level
    # array of a stream has an absent one.
    var schema = export_schema(LogicalType.FLOAT64)
    assert_equal(_c_string_at(schema.format.value()), "g")
    assert_false(Bool(schema.name))
    release_schema(schema)


def test_releasing_a_schema_clears_it() raises:
    var schema = export_schema(LogicalType.INT32, "n")
    release_schema(schema)
    assert_true(schema.is_released())
    assert_false(Bool(schema.private_data))
    assert_false(Bool(schema.format))
    # A consumer is allowed to release without checking first.
    release_schema(schema)
    assert_true(schema.is_released())


def test_an_exported_array_describes_itself() raises:
    var array = export_array(AnyArray(_numbers(7)))
    assert_equal(array.length, 7)
    assert_equal(array.null_count, 0)
    assert_equal(array.offset, 0)
    assert_equal(array.n_buffers, 2)
    assert_equal(array.n_children, 0)
    assert_false(array.is_released())
    assert_true(Bool(array.private_data))
    release_array(array)


def test_the_values_buffer_is_not_copied() raises:
    # The whole point of the file. Address identity, not content equality.
    var column = _numbers(64)
    var before = Int(column.data.values.unsafe_ptr())

    var array = export_array(AnyArray(column^))
    var exported = Int(_buffer_at(array, 1).value())

    assert_equal(exported, before)
    release_array(array)


def test_a_consumer_reads_the_right_values() raises:
    # Read back through the void pointer rather than through the column, because
    # what the column says is not evidence about what the consumer sees.
    var array = export_array(AnyArray(_numbers(5)))
    var values = _buffer_at(array, 1).value().unsafe_bitcast[Int64]()
    for i in range(5):
        assert_equal(
            values.unsafe_offset(i).unsafe_load(), Int64((i + 1) * 100)
        )
    release_array(array)


def test_no_nulls_means_no_validity_buffer() raises:
    # Arrow allows a null validity buffer when nothing is null, and consumers
    # use it to skip a branch per value.
    var array = export_array(AnyArray(_numbers(9)))
    assert_equal(array.null_count, 0)
    assert_false(Bool(_buffer_at(array, 0)))
    release_array(array)


def test_the_validity_bitmap_is_not_copied_either() raises:
    var column = _numbers(20)
    column.set_null(3)
    column.set_null(17)
    var before = Int(column.data.validity.unsafe_ptr())

    var array = export_array(AnyArray(column^))
    assert_equal(array.null_count, 2)
    assert_equal(Int(_buffer_at(array, 0).value()), before)
    release_array(array)


def test_the_validity_bits_mean_what_arrow_thinks_they_mean() raises:
    # firepanda and Arrow both pack one bit per row, least significant bit
    # first, with one meaning present. That coincidence is what makes the
    # validity buffer shareable, so it is asserted rather than assumed.
    var column = _numbers(12)
    column.set_null(0)
    column.set_null(5)
    column.set_null(11)

    var array = export_array(AnyArray(column^))
    var bits = _buffer_at(array, 0).value().unsafe_bitcast[UInt8]()
    for i in range(12):
        var byte = bits.unsafe_offset(i >> 3).unsafe_load()
        var present = ((byte >> UInt8(i & 7)) & 1) != 0
        var expected = i != 0 and i != 5 and i != 11
        assert_equal(present, expected)
    release_array(array)


def test_releasing_an_array_twice_is_a_no_op() raises:
    var array = export_array(AnyArray(_numbers(3)))
    release_array(array)
    assert_true(array.is_released())
    assert_false(Bool(array.private_data))
    assert_false(Bool(array.buffers))
    release_array(array)
    assert_true(array.is_released())


def test_an_empty_column_exports() raises:
    # Zero rows is a real answer, not an error, and a consumer that gets a
    # released structure instead would treat it as one.
    var array = export_array(AnyArray(_numbers(0)))
    assert_equal(array.length, 0)
    assert_equal(array.n_buffers, 2)
    assert_false(array.is_released())
    release_array(array)


def test_every_fixed_width_type_exports() raises:
    var a = export_array(AnyArray(Array[DType.int8](4)))
    assert_equal(a.n_buffers, 2)
    release_array(a)
    var b = export_array(AnyArray(Array[DType.uint16](4)))
    release_array(b)
    var c = export_array(AnyArray(Array[DType.float32](4)))
    release_array(c)
    var d = export_array(AnyArray(Array[DType.uint64](4)))
    release_array(d)


def test_a_bool_column_is_packed_to_a_bit_per_value() raises:
    # The one copy in the exporter, and it is not avoidable: firepanda stores a
    # bool as a byte because that is what a kernel loads, Arrow stores it as a
    # bit. Checked bit by bit through the void pointer.
    var column = Array[DType.bool](11)
    for i in range(11):
        column.set_valid(i, i % 3 == 0)

    var array = export_array(AnyArray(column^))
    assert_equal(array.length, 11)
    assert_equal(array.n_buffers, 2)
    var bits = _buffer_at(array, 1).value().unsafe_bitcast[UInt8]()
    for i in range(11):
        var byte = bits.unsafe_offset(i >> 3).unsafe_load()
        var bit = ((byte >> UInt8(i & 7)) & 1) != 0
        assert_equal(bit, i % 3 == 0)
    release_array(array)


def test_the_bool_values_buffer_is_the_packed_one_not_the_column() raises:
    # The negative of test_the_values_buffer_is_not_copied, and worth asserting
    # for the same reason: handing out the byte per value buffer here would look
    # right in every test that only reads index zero.
    var column = Array[DType.bool](8)
    for i in range(8):
        column.set_valid(i, True)
    var byte_values = Int(column.data.values.unsafe_ptr())

    var array = export_array(AnyArray(column^))
    assert_true(Int(_buffer_at(array, 1).value()) != byte_values)
    release_array(array)


def test_a_string_column_exports_four_buffers() raises:
    var array = export_array(AnyArray(strings_from_list(["a", "bb", "ccc"])))
    assert_equal(array.length, 3)
    assert_equal(array.n_buffers, 4)
    assert_equal(array.null_count, 0)
    release_array(array)


def test_the_string_views_are_not_copied() raises:
    var text = strings_from_list(["alpha", "beta", "gamma"])
    var views_before = Int(text.views.unsafe_ptr())
    var payload_before = Int(text.payload.unsafe_ptr())

    var array = export_array(AnyArray(text^))
    assert_equal(Int(_buffer_at(array, 1).value()), views_before)
    assert_equal(Int(_buffer_at(array, 2).value()), payload_before)
    release_array(array)


def test_the_view_layout_is_arrows_view_layout() raises:
    # firepanda picked this layout for its own reasons and landing on Arrow's was
    # not one of them, so the bytes are compared rather than the coincidence
    # trusted. A short string is a little endian uint32 length followed by the
    # data inline; a long one is the length, a four byte prefix, a uint32 buffer
    # index and a uint32 offset.
    var array = export_array(
        AnyArray(strings_from_list(["hi", "a string longer than twelve bytes"]))
    )
    var views = _buffer_at(array, 1).value().unsafe_bitcast[UInt8]()

    var short_length = views.unsafe_bitcast[UInt32]().unsafe_load()
    assert_equal(short_length, UInt32(2))
    assert_equal(views.unsafe_offset(4).unsafe_load(), UInt8(ord("h")))
    assert_equal(views.unsafe_offset(5).unsafe_load(), UInt8(ord("i")))

    var long_words = views.unsafe_offset(16).unsafe_bitcast[UInt32]()
    assert_equal(long_words.unsafe_load(), UInt32(33))
    # The prefix is the first four bytes of the string, stored as they appear
    # rather than byte swapped, which is what makes it Arrow's prefix and not
    # just a comparison key of our own.
    var prefix = String("a st")
    for i in range(4):
        assert_equal(
            views.unsafe_offset(20 + i).unsafe_load(), prefix.as_bytes()[i]
        )
    # Buffer index zero, because a finished column has exactly one payload block
    # and Arrow numbers the variadic buffers from zero.
    assert_equal(long_words.unsafe_offset(2).unsafe_load(), UInt32(0))
    assert_equal(long_words.unsafe_offset(3).unsafe_load(), UInt32(0))
    release_array(array)


def test_the_sizes_buffer_reports_the_payload_length() raises:
    # The last buffer of a view array is one int64 per variadic data buffer.
    var text = strings_from_list(["a string longer than twelve bytes"])
    var payload_length = len(text.payload)

    var array = export_array(AnyArray(text^))
    var sizes = _buffer_at(array, 3).value().unsafe_bitcast[Int64]()
    assert_equal(sizes.unsafe_load(), Int64(payload_length))
    assert_true(payload_length > 0)
    release_array(array)


def test_a_column_of_only_short_strings_still_has_four_buffers() raises:
    # Nothing goes to the payload when every string fits inside its view, and the
    # empty block is emitted anyway so that the buffer count is a constant. A
    # consumer never reads it, because the sizes buffer says it is zero long.
    var array = export_array(AnyArray(strings_from_list(["a", "bb"])))
    assert_equal(array.n_buffers, 4)
    var sizes = _buffer_at(array, 3).value().unsafe_bitcast[Int64]()
    assert_equal(sizes.unsafe_load(), Int64(0))
    release_array(array)


def test_the_format_string_of_a_string_column_is_the_view_one() raises:
    # "vu", not "u". A consumer that read "u" would expect an offsets buffer and
    # find sixteen byte views.
    var schema = export_schema(LogicalType.STRING, "label")
    assert_equal(_c_string_at(schema.format.value()), "vu")
    release_schema(schema)


def _frame() raises -> DataFrame:
    """Builds a two column frame of int64 and string, with one null."""
    var qty = Array[DType.int64](3)
    qty.set_valid(0, Int64(4))
    qty.set_null(1)
    qty.set_valid(2, Int64(25))
    var series = List[Series](capacity=2)
    series.append(Series("qty", AnyArray(qty^)))
    series.append(
        Series("name", AnyArray(strings_from_list(["rivet", "bolt", "nut"])))
    )
    return DataFrame.from_series(series^)


def _borrow(
    frame: ArcPointer[DataFrame],
) raises -> List[Pointer[AnyArray, MutAnyOrigin]]:
    """Points at every column of a shared frame, the way the binding does."""
    var columns = List[Pointer[AnyArray, MutAnyOrigin]](
        capacity=frame[].width()
    )
    for i in range(frame[].width()):
        columns.append(
            Pointer(to=frame[].columns[i].only()).unsafe_origin_cast[
                MutAnyOrigin
            ]()
        )
    return columns^


def test_a_frame_is_a_struct_with_one_child_per_column() raises:
    # Arrow has no table type at this level. A frame is an array of struct type,
    # which is why the protocol hands back one array and not one per column.
    var schema = export_frame_schema(
        [LogicalType.INT64, LogicalType.STRING], ["qty", "name"]
    )
    assert_equal(_c_string_at(schema.format.value()), "+s")
    assert_equal(schema.n_children, 2)
    var children = schema.children.value()
    assert_equal(
        _c_string_at(children.unsafe_offset(0)[][].name.value()),
        "qty",
    )
    assert_equal(
        _c_string_at(children.unsafe_offset(1)[][].format.value()),
        "vu",
    )
    release_schema(schema)


def test_a_schema_with_the_wrong_number_of_names_is_refused() raises:
    with assert_raises(contains="2 column types but 1 column names"):
        _ = export_frame_schema(
            [LogicalType.INT64, LogicalType.STRING], ["qty"]
        )


def test_a_frames_columns_are_not_copied_on_the_way_out() raises:
    # The whole point. The child's values buffer is the frame's own memory, so a
    # consumer holding it is reading the frame rather than a copy of it.
    var frame = ArcPointer(_frame())
    var values = Int(frame[].columns[0].only().data.values.unsafe_ptr())
    var array = export_frame_array(_borrow(frame), 3, frame)
    var child = array.children.value().unsafe_offset(0)[]
    assert_equal(Int(_buffer_at(child[], 1).value()), values)
    release_array(array)


def test_the_export_keeps_the_frame_alive_after_the_last_other_holder_goes() raises:
    # The ownership question zero copy creates. The frame is destroyed inside the
    # block, the consumer keeps reading, and the values it reads are freed memory
    # if the share in the box is not doing its job.
    var array: ArrowArray

    def build() raises -> ArrowArray:
        var frame = ArcPointer(_frame())
        return export_frame_array(_borrow(frame), 3, frame)

    array = build()
    var child = array.children.value().unsafe_offset(0)[]
    var values = _buffer_at(child[], 1).value().unsafe_bitcast[Int64]()
    assert_equal(values.unsafe_offset(0).unsafe_load(), Int64(4))
    assert_equal(values.unsafe_offset(2).unsafe_load(), Int64(25))
    release_array(array)


def test_a_struct_array_has_one_null_buffer_and_no_nulls() raises:
    # Arrow says a struct array has exactly one buffer, its validity. A frame has
    # no concept of a null row, so the slot is there and holds null, which is what
    # a consumer reads as every row present.
    var frame = ArcPointer(_frame())
    var array = export_frame_array(_borrow(frame), 3, frame)
    assert_equal(array.n_buffers, 1)
    assert_equal(array.null_count, 0)
    assert_false(Bool(_buffer_at(array, 0)))
    release_array(array)


def test_a_column_of_the_wrong_length_is_refused() raises:
    # Arrow requires every child of a struct to be the struct's length. This is
    # firepanda's own frame invariant too, but the export is where a consumer
    # would find out, and it finds out as an error rather than as short reads.
    var frame = ArcPointer(_frame())
    with assert_raises(contains="has 3 rows but the frame has 2"):
        _ = export_frame_array(_borrow(frame), 2, frame)


def test_releasing_a_struct_twice_is_a_no_op() raises:
    # The children are released with the parent, so a second call has to notice
    # that rather than walk a freed list of them.
    var frame = ArcPointer(_frame())
    var array = export_frame_array(_borrow(frame), 3, frame)
    release_array(array)
    release_array(array)
    assert_equal(array.n_children, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
