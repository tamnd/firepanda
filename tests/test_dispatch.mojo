"""Tests for the runtime dtype to compile-time dtype bridge.

`dispatch` is a chain of comparisons that ends in a monomorphized call. The two
things that matter are that it picks the right instantiation and that it raises,
rather than doing something arbitrary, when the column's dtype is not in the list
it was given. The second is what stops a kernel that only handles numbers from
silently treating a bool column as int8.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.dispatch import dispatch, dispatch_typed, list_names
from firepanda.dtype.lists import FLOAT, INTEGER, NUMERIC, SIGNED


def sum_of[dt: DType](col: AnyArray) raises -> Float64:
    """Adds up a column through the unchecked pointer path.

    Args:
        col: The column, whose dtype dispatch has already proved is `dt`.

    Parameters:
        dt: The dtype.

    Returns:
        The sum as a double.
    """
    var total = Float64(0)
    var ptr = col.unsafe_ptr[dt]()
    for i in range(len(col)):
        total += Float64(ptr.unsafe_offset(i).unsafe_load())
    return total


def name_of[dt: DType](col: AnyArray) raises -> String:
    """Returns the name of the dtype the dispatch selected.

    Args:
        col: The column.

    Parameters:
        dt: The dtype.

    Returns:
        The dtype name.
    """
    return String(dt)


def length_of[dt: DType](col: Array[dt]) raises -> Int:
    """Returns the length of an already typed column.

    Args:
        col: The typed column dispatch handed over.

    Parameters:
        dt: The dtype.

    Returns:
        The row count.
    """
    return len(col)


def test_dispatch_selects_the_right_instantiation() raises:
    comptime for dt in NUMERIC:
        var typed = Array[dt](3)
        typed[0] = Scalar[dt](1)
        typed[1] = Scalar[dt](2)
        typed[2] = Scalar[dt](3)
        var erased = AnyArray(typed^)
        assert_equal(dispatch[NUMERIC](erased, name_of), String(dt))
        assert_equal(dispatch[NUMERIC](erased, sum_of), Float64(6))


def test_dispatch_over_a_narrower_list_raises() raises:
    var erased = AnyArray(Array[DType.float64](4))
    with assert_raises(contains="unsupported dtype"):
        _ = dispatch[INTEGER](erased, sum_of)


def test_dispatch_rejects_a_dtype_outside_the_list() raises:
    # bool is deliberately not in NUMERIC. A kernel that asks for NUMERIC is
    # saying arithmetic, and arithmetic on a validity-shaped column is a bug in
    # the caller, not something to guess at.
    var erased = AnyArray(Array[DType.bool](4))
    with assert_raises(contains="unsupported dtype"):
        _ = dispatch[NUMERIC](erased, sum_of)


def test_the_error_lists_what_was_supported() raises:
    var erased = AnyArray(Array[DType.uint8](2))
    with assert_raises(contains="float16, float32, float64"):
        _ = dispatch[FLOAT](erased, sum_of)


def test_dispatch_typed() raises:
    var erased = AnyArray(Array[DType.int32](9))
    assert_equal(dispatch_typed[SIGNED](erased, length_of), 9)


def test_list_names() raises:
    assert_equal(list_names[FLOAT](), "float16, float32, float64")
    assert_equal(list_names[SIGNED](), "int8, int16, int32, int64")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
