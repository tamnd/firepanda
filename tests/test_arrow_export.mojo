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
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import strings_from_list
from firepanda.dtype.logical import LogicalType
from firepanda.io.arrow_c import (
    ARROW_FLAG_NULLABLE,
    ArrowArray,
    NullableVoidPtr,
    release_array,
    release_schema,
)
from firepanda.io.arrow_export import export_array, export_schema


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


def test_bool_is_refused_with_its_reason() raises:
    # Not exported wrong, and not silently omitted. A firepanda bool column is a
    # byte per value where Arrow wants a bit per value.
    with assert_raises(contains="byte per value"):
        _ = export_array(AnyArray(Array[DType.bool](4)))


def test_strings_are_refused_with_their_reason() raises:
    var text = strings_from_list(["a", "b"])
    with assert_raises(contains="variadic data buffers"):
        _ = export_array(AnyArray(text^))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
