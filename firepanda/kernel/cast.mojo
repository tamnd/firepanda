"""Converting a column from one dtype to another.

This is the shortest kernel in the package and the reason is the null-is-zero
invariant. A null holds a zero, zero converts to zero in every direction, so the
loop can run straight over the values buffer and the validity bitmap comes across
unchanged with no repair pass afterwards.

What this does not do is range checking. Casting 300 to int8 gives whatever the
hardware gives, which is 44, and casting a NaN to an integer is undefined in the
same way it is in C. pandas raises on the first and warns on the second. Deciding
which of those firepanda should copy is a question for the milestone that builds
the user-facing cast, and when it is decided the check goes in the layer above
this one. The kernel stays the raw conversion.

`cast_any` is the erased entry point a `DataFrame` calls, and it is the most
expensive function in the package to compile: it dispatches on both ends, so it
instantiates the loop once per ordered pair of the twelve physical dtypes, which
is a hundred and forty four copies. `tools/probes/cast_matrix.mojo` measures what
that is worth in a binary and the answer is 72 KB and 1.3 seconds of compile time
over a probe that links the package and dispatches over nothing, which is about
500 bytes per pair. It is affordable because the loop being copied is nine lines
with no branch in it, and the alternative, routing every cast through a common
widest type, would silently lose precision on exactly the int64 and uint64 round
trips a join key most needs to survive.
"""

from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL


def cast_to[src: DType, dst: DType](col: Array[src]) -> Array[dst]:
    """Converts a column to another dtype.

    Args:
        col: The column to convert.

    Parameters:
        src: The input dtype.
        dst: The output dtype.

    Returns:
        A column of the target dtype, null in the same places as the input.
    """
    # The narrower of the two register widths. int8 to int64 reads four bytes and
    # writes a full register; int64 to int8 does the reverse. Stepping by the
    # smaller lane count keeps both sides inside one register per iteration,
    # which is what the buffer padding guarantees is in bounds.
    comptime width = min(simd_width_of[src](), simd_width_of[dst]())

    var n = len(col)
    var out = Array[dst](n)
    var source = col.unsafe_ptr()
    var target = out.unsafe_ptr()

    var i = 0
    while i < n:
        target.unsafe_offset(i).unsafe_store(
            source.unsafe_offset(i).unsafe_load[width=width]().cast[dst]()
        )
        i += width

    out.data.validity = Bitmap(copy=col.data.validity)
    return out^


def cast_any(col: AnyArray, to: DType) raises -> AnyArray:
    """Converts a column whose dtype is a runtime value to another runtime dtype.

    A cast to the dtype the column already has still copies. Returning the input
    would be faster and would make the result share a buffer with something the
    caller still holds, which is the one thing an eager API cannot afford.

    Args:
        col: The column to convert.
        to: The target dtype.

    Returns:
        A column of dtype `to`, null in the same places as the input.

    Raises:
        If either dtype is not one firepanda has a physical layout for.
    """
    comptime for target in ALL:
        if to == target:
            return AnyArray(_cast_erased[target](col))
    raise Error("cast: unsupported target dtype")


def _cast_erased[dst: DType](col: AnyArray) raises -> Array[dst]:
    """Resolves the source dtype, the target having already been resolved."""
    comptime for source in ALL:
        if col.dtype() == source:
            comptime width = min(simd_width_of[source](), simd_width_of[dst]())
            var n = len(col)
            var out = Array[dst](n)
            var values = col.unsafe_ptr[source]()
            var target = out.unsafe_ptr()

            var i = 0
            while i < n:
                target.unsafe_offset(i).unsafe_store(
                    values.unsafe_offset(i)
                    .unsafe_load[width=width]()
                    .cast[dst]()
                )
                i += width

            out.data.validity = Bitmap(copy=col.data.validity)
            return out^
    raise Error("cast: unsupported source dtype")
