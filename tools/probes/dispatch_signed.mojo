"""The same dispatched operation over four dtypes instead of eleven.

This probe is byte for byte `dispatch_numeric.mojo` with `SIGNED` in place of
`NUMERIC`. Nothing else differs, including the column construction, so the
difference between the two binaries is seven instantiations of `sum_of` and the
seven extra arms of the dispatch chain. Dividing by seven gives the marginal cost
of a dtype.

Keep the two files in step. If an edit lands in one and not the other the numbers
keep being produced and quietly stop meaning anything.
"""

from std.sys import argv

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.dispatch import dispatch
from firepanda.dtype.lists import SIGNED


def sum_of[dt: DType](col: AnyArray) raises -> Float64:
    """Adds up a column, vectorized by the compiler at the column's own dtype.

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


def opaque_column() -> AnyArray:
    """Builds a column whose dtype the compiler cannot know.

    Returns:
        A short column of whichever dtype the argument count selects.
    """
    var kind = len(argv())
    if kind == 1:
        return AnyArray(Array[DType.int8](16))
    elif kind == 2:
        return AnyArray(Array[DType.int16](16))
    elif kind == 3:
        return AnyArray(Array[DType.int32](16))
    elif kind == 4:
        return AnyArray(Array[DType.int64](16))
    elif kind == 5:
        return AnyArray(Array[DType.uint8](16))
    elif kind == 6:
        return AnyArray(Array[DType.uint16](16))
    elif kind == 7:
        return AnyArray(Array[DType.uint32](16))
    elif kind == 8:
        return AnyArray(Array[DType.uint64](16))
    elif kind == 9:
        return AnyArray(Array[DType.float16](16))
    elif kind == 10:
        return AnyArray(Array[DType.float32](16))
    elif kind == 11:
        return AnyArray(Array[DType.float64](16))
    return AnyArray(Array[DType.bool](16))


def main():
    """Dispatches once, and prints whatever comes back."""
    var column = opaque_column()
    try:
        print(dispatch[SIGNED](column, sum_of))
    except error:
        print(error)
