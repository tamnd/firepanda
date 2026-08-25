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
"""

from std.sys.info import simd_width_of

from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap


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
