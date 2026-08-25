"""One dispatched operation over the eleven numeric dtypes.

The strategy in docs/specs/03-dtype-dispatch.md trades binary size for speed: an
operation written once is compiled once per dtype in the list it is dispatched
over. The number that matters is the marginal cost of a dtype, and the way to get
it is to compile the same operation over lists of different lengths and subtract.

This probe is the eleven-way version and `dispatch_signed.mojo` is the four-way
one. Everything in the two files is identical apart from the name of the list, so
the difference between the binaries is seven instantiations of `sum_of` and the
seven extra arms of the dispatch chain, which together are what one more dtype
costs. `baseline.mojo` is the floor that both are reported against.

The column's dtype comes from the argument count rather than from a literal, and
that detail is load bearing. The first version of this probe built an
`Array[DType.int32]` directly, the compiler saw that the dtype was a constant,
folded the whole dispatch chain down to the one arm that could match, and both
probes came out at exactly 45816 bytes. A probe that measures zero is worse than
no probe, because it looks like an answer.
"""

from std.sys import argv

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.dispatch import dispatch
from firepanda.dtype.lists import NUMERIC


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
        print(dispatch[NUMERIC](column, sum_of))
    except error:
        print(error)
