"""Tests for the type lattice and the compile-time dtype lists.

The promotion table is the part worth being careful about. It is copied from
NumPy rather than from SQL, including the two rules that surprise people: signed
mixed with unsigned of the same width goes up a width rather than reinterpreting,
and uint64 mixed with any signed type goes to float64 and loses precision above
2^53. Both are checked against the real numpy in tests/differential/main.mojo;
what is checked here is that the answer does not depend on argument order and
that the raising cases raise.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.dtype.lists import (
    ALL,
    FLOAT,
    INTEGER,
    NUMERIC,
    SIGNED,
    UNSIGNED,
    contains,
    dtype_size,
)
from firepanda.dtype.logical import LogicalType, TypeKind, logical_for, promote


def test_list_membership() raises:
    assert_true(contains[SIGNED](DType.int32))
    assert_false(contains[SIGNED](DType.uint32))
    assert_true(contains[UNSIGNED](DType.uint32))
    assert_true(contains[INTEGER](DType.uint8))
    assert_false(contains[INTEGER](DType.float32))
    assert_true(contains[FLOAT](DType.float16))
    assert_true(contains[NUMERIC](DType.float64))
    assert_false(contains[NUMERIC](DType.bool))
    assert_true(contains[ALL](DType.bool))


def test_list_sizes() raises:
    # These counts are the multiplier on every dispatched kernel's code size, so
    # a change to them is a change worth noticing in review.
    assert_equal(comptime (len(SIGNED)), 4)
    assert_equal(comptime (len(UNSIGNED)), 4)
    assert_equal(comptime (len(INTEGER)), 8)
    assert_equal(comptime (len(FLOAT)), 3)
    assert_equal(comptime (len(NUMERIC)), 11)
    assert_equal(comptime (len(ALL)), 12)


def test_dtype_size() raises:
    assert_equal(dtype_size(DType.bool), 1)
    assert_equal(dtype_size(DType.int8), 1)
    assert_equal(dtype_size(DType.int16), 2)
    assert_equal(dtype_size(DType.int32), 4)
    assert_equal(dtype_size(DType.int64), 8)
    assert_equal(dtype_size(DType.uint64), 8)
    assert_equal(dtype_size(DType.float16), 2)
    assert_equal(dtype_size(DType.float32), 4)
    assert_equal(dtype_size(DType.float64), 8)


def test_logical_for() raises:
    assert_equal(logical_for(DType.bool), LogicalType.BOOL)
    assert_equal(logical_for(DType.int32), LogicalType.INT32)
    assert_equal(logical_for(DType.uint16), LogicalType.UINT16)
    assert_equal(logical_for(DType.float64), LogicalType.FLOAT64)


def test_predicates() raises:
    assert_true(LogicalType.INT64.is_numeric())
    assert_true(LogicalType.INT64.is_integer())
    assert_false(LogicalType.INT64.is_float())
    assert_true(LogicalType.INT64.is_signed())
    assert_false(LogicalType.UINT64.is_signed())
    assert_true(LogicalType.FLOAT32.is_float())
    assert_false(LogicalType.BOOL.is_numeric())
    assert_true(LogicalType.STRING.is_variable_width())
    assert_false(LogicalType.INT8.is_variable_width())


def test_bit_width() raises:
    assert_equal(LogicalType.INT8.bit_width(), 8)
    assert_equal(LogicalType.INT16.bit_width(), 16)
    assert_equal(LogicalType.UINT32.bit_width(), 32)
    assert_equal(LogicalType.FLOAT64.bit_width(), 64)


def test_promote_identical() raises:
    assert_equal(
        promote(LogicalType.INT32, LogicalType.INT32), LogicalType.INT32
    )
    assert_equal(
        promote(LogicalType.STRING, LogicalType.STRING), LogicalType.STRING
    )


def test_promote_null() raises:
    assert_equal(
        promote(LogicalType.NULL, LogicalType.INT16), LogicalType.INT16
    )
    assert_equal(
        promote(LogicalType.STRING, LogicalType.NULL), LogicalType.STRING
    )


def test_promote_same_signedness_takes_the_wider() raises:
    assert_equal(
        promote(LogicalType.INT8, LogicalType.INT64), LogicalType.INT64
    )
    assert_equal(
        promote(LogicalType.UINT16, LogicalType.UINT8), LogicalType.UINT16
    )


def test_promote_mixed_signedness_goes_up_a_width() raises:
    assert_equal(
        promote(LogicalType.UINT8, LogicalType.INT8), LogicalType.INT16
    )
    assert_equal(
        promote(LogicalType.UINT16, LogicalType.INT8), LogicalType.INT32
    )
    assert_equal(
        promote(LogicalType.UINT32, LogicalType.INT16), LogicalType.INT64
    )
    assert_equal(
        promote(LogicalType.UINT8, LogicalType.INT64), LogicalType.INT64
    )


def test_promote_uint64_with_signed_goes_to_float64() raises:
    assert_equal(
        promote(LogicalType.UINT64, LogicalType.INT8), LogicalType.FLOAT64
    )
    assert_equal(
        promote(LogicalType.UINT64, LogicalType.INT64), LogicalType.FLOAT64
    )


def test_promote_float_beats_integer() raises:
    assert_equal(
        promote(LogicalType.INT8, LogicalType.FLOAT16), LogicalType.FLOAT16
    )
    assert_equal(
        promote(LogicalType.INT16, LogicalType.FLOAT16), LogicalType.FLOAT32
    )
    assert_equal(
        promote(LogicalType.INT32, LogicalType.FLOAT16), LogicalType.FLOAT64
    )
    assert_equal(
        promote(LogicalType.INT16, LogicalType.FLOAT32), LogicalType.FLOAT32
    )
    assert_equal(
        promote(LogicalType.INT32, LogicalType.FLOAT32), LogicalType.FLOAT64
    )
    assert_equal(
        promote(LogicalType.INT64, LogicalType.FLOAT32), LogicalType.FLOAT64
    )
    assert_equal(
        promote(LogicalType.FLOAT32, LogicalType.FLOAT64), LogicalType.FLOAT64
    )


def test_promote_bool_with_numeric() raises:
    assert_equal(promote(LogicalType.BOOL, LogicalType.INT8), LogicalType.INT8)
    assert_equal(
        promote(LogicalType.FLOAT32, LogicalType.BOOL), LogicalType.FLOAT32
    )


def test_promote_is_symmetric() raises:
    var types = [
        LogicalType.BOOL,
        LogicalType.INT8,
        LogicalType.INT16,
        LogicalType.INT32,
        LogicalType.INT64,
        LogicalType.UINT8,
        LogicalType.UINT16,
        LogicalType.UINT32,
        LogicalType.UINT64,
        LogicalType.FLOAT16,
        LogicalType.FLOAT32,
        LogicalType.FLOAT64,
    ]
    for i in range(len(types)):
        for j in range(len(types)):
            assert_equal(
                promote(types[i], types[j]), promote(types[j], types[i])
            )


def test_promote_string_with_number_raises() raises:
    with assert_raises(contains="no common type"):
        _ = promote(LogicalType.STRING, LogicalType.INT64)


def test_type_kind_equality() raises:
    assert_true(TypeKind.INT == TypeKind.INT)
    assert_true(TypeKind.INT != TypeKind.FLOAT_KIND)


def test_types_print() raises:
    assert_equal(String(LogicalType.INT64), "int64")
    assert_equal(String(LogicalType.STRING), "string")
    assert_equal(String(LogicalType.NULL), "null")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
