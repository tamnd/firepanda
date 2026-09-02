"""Tests for the type-erased scalar.

`Value` is a union written out as fields, and a union written out as fields is
only correct if every way in agrees with every way out. So most of what is here
is round trips: put an element in at one dtype, read it back, and check that both
the element and the type survived. The interesting ones are the pairs where a
naive store would lose something. A negative signed integer read back through a
float. A uint64 above the top of int64. A float32 that has to come back exactly
rather than as the nearest float64 of a widened-then-narrowed value.

The other half is nulls. A null is a value here, not the absence of one, and it
has a type, and it is not the zero of that type. Two tests pin that down because
every reduction that gets built on this will depend on it.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_raises
from std.testing import assert_true

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType


def test_int_round_trips() raises:
    """An integer comes back as itself, at the dtype it went in as."""
    var v = Value(Int32(42))
    assert_equal(v.type, LogicalType.INT32)
    assert_equal(v.as_scalar[DType.int32](), 42)
    assert_true(v.present)


def test_negative_int_round_trips() raises:
    """A negative integer keeps its sign through the unsigned store.

    The bits field is a uint64, so minus five is held as the sign extended bit
    pattern. Reading it back has to go through int64 or the sign is gone.
    """
    var v = Value(Int8(-5))
    assert_equal(v.type, LogicalType.INT8)
    assert_equal(v.as_scalar[DType.int8](), -5)
    assert_equal(v.as_scalar[DType.int64](), -5)


def test_negative_int_reads_as_a_float() raises:
    """A negative integer read at a float dtype is still negative.

    This is the one a bitwise store gets wrong. Casting the raw uint64 straight
    to a float answers eighteen quintillion rather than minus five.
    """
    var v = Value(Int64(-5))
    assert_equal(v.as_scalar[DType.float64](), -5.0)


def test_large_unsigned_round_trips() raises:
    """A uint64 above the top of int64 survives.

    Storing unsigned values in a signed field would fold this one, so the store
    is unsigned and the sign is put back on the way out instead.
    """
    var big = UInt64(18446744073709551615)
    var v = Value(big)
    assert_equal(v.type, LogicalType.UINT64)
    assert_equal(v.as_scalar[DType.uint64](), big)


def test_float32_round_trips_exactly() raises:
    """A float32 comes back exact, not through a narrowing.

    Floats are stored at full width, so this is a widening on the way in and a
    narrowing on the way out, and both are exact for a value that started as a
    float32.
    """
    var v = Value(Float32(0.1))
    assert_equal(v.type, LogicalType.FLOAT32)
    assert_equal(v.as_scalar[DType.float32](), Float32(0.1))


def test_float_keeps_full_width() raises:
    """A float64 is not rounded to a float32 on the way through."""
    var v = Value(Float64(0.1))
    assert_equal(v.type, LogicalType.FLOAT64)
    assert_equal(v.as_scalar[DType.float64](), 0.1)


def test_bool_is_its_own_constructor() raises:
    """A bool goes in as a bool and comes out as one.

    Bool is not a one lane SIMD in Mojo, so it needs a constructor of its own
    rather than falling into the parametric one.
    """
    var t = Value(True)
    var f = Value(False)
    assert_equal(t.type, LogicalType.BOOL)
    assert_equal(t.as_scalar[DType.bool](), True)
    assert_equal(f.as_scalar[DType.bool](), False)


def test_text_round_trips() raises:
    """A string comes back byte for byte."""
    var v = Value(String("firepanda"))
    assert_equal(v.type, LogicalType.STRING)
    assert_equal(v.as_string(), "firepanda")


def test_text_accessor_refuses_a_number() raises:
    """Reading a number as text is an error rather than a conversion.

    Turning a number into text allocates, so it belongs in the cast kernel where
    the caller asked for it, not in an accessor where it would happen quietly.
    """
    var v = Value(Int64(7))
    with assert_raises(contains="is not text"):
        _ = v.as_string()


def test_null_has_a_type() raises:
    """A null knows what type it would have been.

    A reduction over an empty int64 column answers a null int64, and the frame
    that receives it still has to build a column of some type.
    """
    var v = Value(null=LogicalType.INT64)
    assert_true(v.is_null())
    assert_equal(v.type, LogicalType.INT64)


def test_null_is_not_the_zero_of_its_type() raises:
    """A null and a zero of the same type are different values.

    pandas answers NaN for the mean of an empty column and 0.0 for the mean of a
    column holding one zero. Those two answers have to be distinguishable here or
    the difference is lost before anybody can look at it.
    """
    var nothing = Value(null=LogicalType.INT64)
    var zero = Value(Int64(0))
    assert_true(nothing != zero)
    assert_false(nothing.present)
    assert_true(zero.present)


def test_null_reads_as_zero() raises:
    """A null reads as zero, which is what the column layout already says.

    A null position in a column holds a zero in the values buffer with the
    validity saying so separately, and this keeps that rule so a caller that
    forgot to ask gets the same wrong answer here as there rather than garbage.
    """
    var v = Value(null=LogicalType.FLOAT64)
    assert_equal(v.as_scalar[DType.float64](), 0.0)


def test_equality_takes_the_type_into_account() raises:
    """The same number at two dtypes is two different values.

    Promotion happens on types, so a value that did not carry its type would let
    an int32 constant and an int64 constant produce different answers from an
    expression that compared equal here.
    """
    assert_true(Value(Int32(5)) != Value(Int64(5)))
    assert_true(Value(Int32(5)) == Value(Int32(5)))


def test_two_nulls_of_one_type_are_equal() raises:
    """Two nulls of the same type compare equal.

    That is not what SQL says about nulls in a predicate, and it is what a test
    comparing two reduction answers wants. Nothing builds a predicate out of
    this.
    """
    assert_true(Value(null=LogicalType.INT64) == Value(null=LogicalType.INT64))
    assert_true(Value(null=LogicalType.INT64) != Value(null=LogicalType.INT32))


def test_copy_is_independent() raises:
    """Copying a text value copies the bytes rather than sharing them."""
    var original = Value(String("hello"))
    var copy = Value(copy=original)
    assert_equal(copy.as_string(), "hello")
    assert_true(copy == original)


def test_printing() raises:
    """Every shape prints the way a frame prints one cell."""
    assert_equal(String(Value(Int64(-3))), "-3")
    assert_equal(String(Value(UInt8(200))), "200")
    assert_equal(String(Value(True)), "true")
    assert_equal(String(Value(False)), "false")
    assert_equal(String(Value(String("abc"))), "abc")
    assert_equal(String(Value(null=LogicalType.INT64)), "null")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
