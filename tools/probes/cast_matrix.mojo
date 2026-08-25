"""One dispatch on two dtypes at once, which is a hundred and forty four copies.

Every other probe here measures a one-sided dispatch: a runtime dtype picks one
of a list of compiled instantiations and the count is the length of the list.
`cast_any` is the first operation in firepanda that dispatches on a source and a
target independently, so its count is the square, and the square of twelve is a
different order of thing from twelve.

This probe exists to make that a number rather than an argument. Its interesting
figure is `bytes_over_baseline`: `baseline` links the package and dispatches over
nothing, this one links the package and pulls in the whole matrix, and the
difference is what a two-sided dispatch costs to have. If a second one is ever
proposed, this is the row to look at first.

The column's dtype comes from the argument count for the same reason it does in
the other probes: it has to be a value the compiler cannot fold, or there is no
dispatch left to measure.
"""

from std.sys import argv

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.kernel.cast import cast_any


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


def opaque_target() -> DType:
    """Returns a target dtype the compiler cannot fold either.

    Returns:
        A dtype selected by the argument count.
    """
    var kind = len(argv())
    if kind == 1:
        return DType.float64
    elif kind == 2:
        return DType.int32
    elif kind == 3:
        return DType.uint8
    return DType.int64


def main():
    """Casts once, across a pair of dtypes neither of which is known."""
    var column = opaque_column()
    try:
        print(len(cast_any(column, opaque_target())))
    except error:
        print(error)
