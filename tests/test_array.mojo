"""Tests for the column types.

The round trip test is the M0 exit criterion: for every dtype in `ALL`, build a
typed array, write values and nulls into it, erase the dtype, put it back, and get
the same values out. The erasure step is the one that has been rewritten most
often, because moving buffers out of a typed column into an untyped one without
copying them is the whole reason `ColumnData` exists.
"""

from std.builtin.rebind import rebind
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.array.chunked import ChunkedArray
from firepanda.dtype.lists import ALL, NUMERIC
from firepanda.dtype.logical import LogicalType, logical_for


def sample[dt: DType](i: Int) -> Scalar[dt]:
    """Returns a value of `dt` that varies with the position.

    Args:
        i: The position.

    Parameters:
        dt: The dtype.

    Returns:
        A value small enough to be exact in float16 and to fit in int8.
    """
    comptime if dt == DType.bool:
        # A `comptime if` does not narrow `dt`, so the bool scalar has to be
        # built at its own type and then rebound. Without the rebind the compiler
        # picks the Intable constructor, which rejects a non-integral dtype.
        return rebind[Scalar[dt]](Scalar[DType.bool](i % 3 == 0))
    else:
        return Scalar[dt]((i * 7) % 100)


def test_every_dtype_round_trips() raises:
    comptime for dt in ALL:
        var typed = Array[dt](64)
        assert_equal(len(typed), 64)
        assert_equal(typed.null_count(), 0)

        for i in range(64):
            typed[i] = sample[dt](i)
        for i in range(0, 64, 5):
            typed.set_null(i)

        var expected_nulls = typed.null_count()
        var erased = AnyArray(typed^)
        assert_equal(len(erased), 64)
        assert_equal(erased.dtype(), dt)
        assert_equal(erased.null_count(), expected_nulls)

        var back = erased^.into_typed[dt]()
        assert_equal(len(back), 64)
        assert_equal(back.null_count(), expected_nulls)
        for i in range(64):
            if i % 5 == 0:
                assert_false(back.is_valid(i))
                assert_equal(back[i], Scalar[dt](0))
            else:
                assert_true(back.is_valid(i))
                assert_equal(back[i], sample[dt](i))


def test_from_list_and_to_list() raises:
    var values: List[Int64] = [Int64(3), Int64(-1), Int64(0), Int64(99)]
    var array = from_list[DType.int64](values)
    assert_equal(len(array), 4)
    var out = array.to_list()
    for i in range(4):
        assert_equal(out[i], values[i])


def test_new_array_is_zero_and_valid() raises:
    var array = Array[DType.float64](10)
    for i in range(10):
        assert_equal(array[i], Float64(0))
        assert_true(array.is_valid(i))


def test_set_null_zeroes_the_value() raises:
    # Kernels read under nulls without masking in several places. That is only
    # correct if a null position holds zero.
    var array = Array[DType.int32](4)
    array[2] = Int32(1234)
    array.set_null(2)
    assert_false(array.is_valid(2))
    assert_equal(array[2], Int32(0))


def test_set_valid() raises:
    var array = Array[DType.int32](4)
    array.set_null(1)
    array.set_valid(1, Int32(77))
    assert_true(array.is_valid(1))
    assert_equal(array[1], Int32(77))


def test_simd_load_and_store() raises:
    var array = Array[DType.int32](16)
    array.store[4](0, SIMD[DType.int32, 4](1, 2, 3, 4))
    var loaded = array.load[4](0)
    assert_equal(loaded[0], Int32(1))
    assert_equal(loaded[3], Int32(4))


def test_load_may_run_past_the_end() raises:
    # The buffer is padded to a 64-byte multiple, so a tail load of a register
    # that straddles the logical end reads zeros rather than someone else's page.
    var array = Array[DType.int32](5)
    for i in range(5):
        array[i] = Int32(i + 1)
    var loaded = array.load[8](0)
    assert_equal(loaded[4], Int32(5))
    assert_equal(loaded[5], Int32(0))
    assert_equal(loaded[7], Int32(0))


def test_slice_copies_values_and_validity() raises:
    var array = Array[DType.int64](20)
    for i in range(20):
        array[i] = Int64(i)
    array.set_null(7)
    var part = array.slice(5, 15)
    assert_equal(len(part), 10)
    for i in range(10):
        if i == 2:
            assert_false(part.is_valid(i))
        else:
            assert_true(part.is_valid(i))
            assert_equal(part[i], Int64(i + 5))


def test_copy_is_deep() raises:
    var original = Array[DType.int64](8)
    original[0] = Int64(1)
    var duplicate = Array[DType.int64](copy=original)
    duplicate[0] = Int64(2)
    assert_equal(original[0], Int64(1))
    assert_equal(duplicate[0], Int64(2))


def test_any_array_as_typed_copies() raises:
    var typed = Array[DType.int64](4)
    typed[0] = Int64(5)
    var erased = AnyArray(typed^)
    var view = erased.as_typed[DType.int64]()
    view[0] = Int64(6)
    assert_equal(erased.unsafe_ptr[DType.int64]().unsafe_load(), Int64(5))


def test_as_typed_with_the_wrong_dtype_raises() raises:
    var erased = AnyArray(Array[DType.int64](4))
    with assert_raises(contains="dtype mismatch"):
        _ = erased.as_typed[DType.float64]()


def test_any_array_carries_its_logical_type() raises:
    var erased = AnyArray(Array[DType.float32](2))
    assert_equal(erased.type, LogicalType.FLOAT32)
    assert_equal(erased.dtype(), DType.float32)


def test_chunked_lengths_add_up() raises:
    var column = ChunkedArray(AnyArray(Array[DType.int64](10)))
    column.append(AnyArray(Array[DType.int64](5)))
    column.append(AnyArray(Array[DType.int64](7)))
    assert_equal(len(column), 22)
    assert_equal(column.num_chunks(), 3)
    assert_equal(column.dtype(), DType.int64)


def test_chunked_rejects_a_mismatched_chunk() raises:
    var column = ChunkedArray(AnyArray(Array[DType.int64](4)))
    with assert_raises(contains="does not match"):
        column.append(AnyArray(Array[DType.float64](4)))


def test_chunked_locate() raises:
    var column = ChunkedArray(AnyArray(Array[DType.int64](10)))
    column.append(AnyArray(Array[DType.int64](5)))

    var first = column.locate(3)
    assert_equal(first[0], 0)
    assert_equal(first[1], 3)

    var second = column.locate(12)
    assert_equal(second[0], 1)
    assert_equal(second[1], 2)

    with assert_raises(contains="out of range"):
        _ = column.locate(15)


def test_chunked_null_count_sums_chunks() raises:
    var first = Array[DType.int64](4)
    first.set_null(0)
    var second = Array[DType.int64](4)
    second.set_null(1)
    second.set_null(2)

    var column = ChunkedArray(AnyArray(first^))
    column.append(AnyArray(second^))
    assert_equal(column.null_count(), 3)


def test_empty_chunked_column() raises:
    var column = ChunkedArray(logical_for(DType.int64))
    assert_equal(len(column), 0)
    assert_equal(column.num_chunks(), 0)
    assert_equal(column.null_count(), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
