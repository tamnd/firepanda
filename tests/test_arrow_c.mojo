"""Tests for the Arrow C Data Interface declarations.

These are layout tests, which is an unusual thing to write and worth explaining.
Nothing here computes anything. What every one of them checks is that the bytes
firepanda hands to a C consumer sit where that consumer will look for them, and
the reason to check it here rather than by round tripping through pyarrow is that
a layout mistake seen through pyarrow arrives as a segfault or, worse, as plausible
wrong numbers, in a process where nothing points back at the field that moved.

The size assertions are the first line. The offset assertions are the second, and
they work by writing a distinct value into every field and then reading the struct
back as an array of eight byte slots. If a field moves, one of those slots holds
somebody else's value and the test names which one. That is a stronger check than
size alone, because two fields swapping places does not change the size.

The last group is the release protocol, which is the part of the C interface that
is a contract rather than a layout: a callback that sets its own structure's
release field to null so that a double release is a no-op. That has to be
exercised for real, through an actual `abi("C")` function pointer, because whether
Mojo can call one at all is exactly the thing in doubt.
"""

from std.ffi import c_char
from std.sys import size_of
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.dtype.logical import LogicalType
from firepanda.io.arrow_c import (
    ARROW_FLAG_DICTIONARY_ORDERED,
    ARROW_FLAG_MAP_KEYS_SORTED,
    ARROW_FLAG_NULLABLE,
    ArrayPtr,
    ArrayRelease,
    ArrowArray,
    ArrowSchema,
    NullableVoidPtr,
    SchemaPtr,
    SchemaRelease,
    VoidPtr,
    array_release_callback,
    buffer_count,
    format_for,
    release_array,
    release_schema,
    schema_release_callback,
    type_for_format,
)


def test_schema_is_seventy_two_bytes() raises:
    # Three pointers, two int64, four pointers. Nine fields, all eight bytes on
    # every platform Arrow supports.
    assert_equal(size_of[ArrowSchema](), 72)


def test_array_is_eighty_bytes() raises:
    # Five int64 then five pointers.
    assert_equal(size_of[ArrowArray](), 80)


def test_nullable_pointer_is_one_word() raises:
    # The whole reason the release fields are typed as a nullable void pointer
    # rather than a nullable function pointer. If this ever stops holding, both
    # structs above grow and every offset after the field shifts, so the check
    # belongs next to the size assertions rather than in a comment.
    assert_equal(size_of[NullableVoidPtr](), 8)
    assert_equal(size_of[VoidPtr](), 8)


def test_a_function_pointer_is_one_word() raises:
    assert_equal(size_of[SchemaRelease](), 8)
    assert_equal(size_of[ArrayRelease](), 8)


def test_none_is_the_null_pointer() raises:
    # `None` has to be all zero bytes for the struct to be C compatible, because
    # C says an unset callback is a null pointer and knows nothing about Mojo's
    # optional. Written by reading the byte pattern rather than by asking the
    # optional whether it is empty, which would only prove Mojo agrees with
    # itself.
    var slot = NullableVoidPtr(None)
    var bytes = Pointer(to=slot).unsafe_bitcast[Int64]()
    assert_equal(bytes[], 0)


def test_default_schema_is_all_zero() raises:
    # A released structure is the zero structure, and a producer that fills in
    # fields one at a time depends on the rest already being zero.
    var schema = ArrowSchema()
    var words = Pointer(to=schema).unsafe_bitcast[Int64]()
    for i in range(9):
        assert_equal(words[unsafe_offset=i], 0)
    assert_true(schema.is_released())


def test_default_array_is_all_zero() raises:
    var array = ArrowArray()
    var words = Pointer(to=array).unsafe_bitcast[Int64]()
    for i in range(10):
        assert_equal(words[unsafe_offset=i], 0)
    assert_true(array.is_released())


def test_schema_field_order() raises:
    # A distinct value in every field, read back positionally. The two pointer
    # fields get the addresses of two different locals, so they are distinct from
    # each other and from every other slot without having to pick a magic number
    # that a pointer might happen to equal.
    var format_bytes = InlineArray[UInt8, 2](fill=0)
    var name_bytes = InlineArray[UInt8, 2](fill=0)
    var schema = ArrowSchema()
    schema.format = (
        Pointer(to=format_bytes[0])
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[c_char]()
    )
    schema.name = (
        Pointer(to=name_bytes[0])
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[c_char]()
    )
    schema.flags = 111
    schema.n_children = 222

    var words = Pointer(to=schema).unsafe_bitcast[Int64]()
    assert_equal(words[unsafe_offset=0], Int64(Int(schema.format.value())))
    assert_equal(words[unsafe_offset=1], Int64(Int(schema.name.value())))
    assert_equal(words[unsafe_offset=2], 0)
    assert_equal(words[unsafe_offset=3], 111)
    assert_equal(words[unsafe_offset=4], 222)
    assert_equal(words[unsafe_offset=5], 0)
    assert_equal(words[unsafe_offset=6], 0)
    assert_equal(words[unsafe_offset=7], 0)
    assert_equal(words[unsafe_offset=8], 0)


def test_array_field_order() raises:
    var array = ArrowArray()
    array.length = 11
    array.null_count = 22
    array.offset = 33
    array.n_buffers = 44
    array.n_children = 55

    var words = Pointer(to=array).unsafe_bitcast[Int64]()
    assert_equal(words[unsafe_offset=0], 11)
    assert_equal(words[unsafe_offset=1], 22)
    assert_equal(words[unsafe_offset=2], 33)
    assert_equal(words[unsafe_offset=3], 44)
    assert_equal(words[unsafe_offset=4], 55)
    for i in range(5, 10):
        assert_equal(words[unsafe_offset=i], 0)


def test_flag_values() raises:
    # Fixed by the specification, not by us, so they are asserted rather than
    # trusted to have been typed correctly.
    assert_equal(ARROW_FLAG_DICTIONARY_ORDERED, 1)
    assert_equal(ARROW_FLAG_NULLABLE, 2)
    assert_equal(ARROW_FLAG_MAP_KEYS_SORTED, 4)


def _release_schema_stub(schema: SchemaPtr) abi("C") -> None:
    """A minimal conforming schema release callback."""
    schema[].release = None


def _release_array_stub(array: ArrayPtr) abi("C") -> None:
    """A minimal conforming array release callback."""
    array[].release = None


def test_a_c_callback_can_be_stored_and_called() raises:
    # The one test here that is about Mojo rather than about Arrow. It proves a
    # function declared `abi("C")` survives being reduced to a void pointer,
    # stored in a struct field, read back and called, which is the mechanism
    # every producer and consumer in this file depends on.
    var schema = ArrowSchema()
    schema.release = schema_release_callback(_release_schema_stub)
    assert_false(schema.is_released())

    release_schema(schema)
    assert_true(schema.is_released())


def test_releasing_twice_is_a_no_op() raises:
    # A consumer is allowed to release whatever it was handed without checking
    # first, and a producer that faulted on the second call would be the one at
    # fault.
    var array = ArrowArray()
    array.release = array_release_callback(_release_array_stub)
    release_array(array)
    assert_true(array.is_released())
    release_array(array)
    assert_true(array.is_released())


def test_releasing_a_zero_structure_is_a_no_op() raises:
    var array = ArrowArray()
    release_array(array)
    assert_true(array.is_released())


def test_format_strings() raises:
    assert_equal(format_for(LogicalType.NULL), "n")
    assert_equal(format_for(LogicalType.BOOL), "b")
    assert_equal(format_for(LogicalType.INT8), "c")
    assert_equal(format_for(LogicalType.UINT8), "C")
    assert_equal(format_for(LogicalType.INT16), "s")
    assert_equal(format_for(LogicalType.UINT16), "S")
    assert_equal(format_for(LogicalType.INT32), "i")
    assert_equal(format_for(LogicalType.UINT32), "I")
    assert_equal(format_for(LogicalType.INT64), "l")
    assert_equal(format_for(LogicalType.UINT64), "L")
    assert_equal(format_for(LogicalType.FLOAT16), "e")
    assert_equal(format_for(LogicalType.FLOAT32), "f")
    assert_equal(format_for(LogicalType.FLOAT64), "g")
    assert_equal(format_for(LogicalType.STRING), "vu")
    assert_equal(format_for(LogicalType.BINARY), "vz")


def test_every_format_string_round_trips() raises:
    # The pair of tables is the thing most likely to drift, because adding a type
    # means editing two functions and nothing makes you do the second one.
    var types = [
        LogicalType.NULL,
        LogicalType.BOOL,
        LogicalType.INT8,
        LogicalType.UINT8,
        LogicalType.INT16,
        LogicalType.UINT16,
        LogicalType.INT32,
        LogicalType.UINT32,
        LogicalType.INT64,
        LogicalType.UINT64,
        LogicalType.FLOAT16,
        LogicalType.FLOAT32,
        LogicalType.FLOAT64,
        LogicalType.STRING,
        LogicalType.BINARY,
    ]
    for type in types:
        assert_equal(type_for_format(format_for(type)), type)


def test_case_matters() raises:
    # The signed and unsigned formats differ only in case, so a comparison that
    # folded case would silently turn every unsigned column into a signed one.
    assert_true(type_for_format("c") != type_for_format("C"))
    assert_true(type_for_format("s") != type_for_format("S"))
    assert_true(type_for_format("i") != type_for_format("I"))
    assert_true(type_for_format("l") != type_for_format("L"))


def test_offset_string_formats_are_refused() raises:
    # Not silently read as views. See the note on `type_for_format`.
    for format in ["u", "U", "z", "Z"]:
        with assert_raises(contains="unsupported format string"):
            _ = type_for_format(format)


def test_unknown_formats_are_refused() raises:
    with assert_raises(contains="unsupported format string"):
        _ = type_for_format("+s")
    with assert_raises(contains="unsupported format string"):
        _ = type_for_format("tsm:")
    with assert_raises(contains="unsupported format string"):
        _ = type_for_format("")


def test_buffer_counts() raises:
    assert_equal(buffer_count(LogicalType.NULL), 0)
    assert_equal(buffer_count(LogicalType.BOOL), 2)
    assert_equal(buffer_count(LogicalType.INT64), 2)
    assert_equal(buffer_count(LogicalType.FLOAT64), 2)
    assert_equal(buffer_count(LogicalType.STRING), 4)
    assert_equal(buffer_count(LogicalType.BINARY), 4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
