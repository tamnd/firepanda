"""Restoring the null-is-zero invariant after a kernel has run.

An elementwise kernel computes over the whole values buffer, nulls included,
because branching per element would cost more than the arithmetic. That is fine
for the result at the positions that are present, and it is wrong at the ones
that are not: adding a valid 5 to a null gives 5, and the answer has to be a null
holding a zero, not a 5 that is merely flagged.

So the kernels compute first and repair afterwards. `apply_validity` is the
repair. It walks the output's validity a 64-bit word at a time, leaves the
all-present words alone, blanks the all-null words with vector stores, and only
touches individual elements in the words that are mixed.

The cost is proportional to the nulls, not to the length. On a column with no
nulls it is one comparison per sixty four rows.

The repair is not a pass of its own. A kernel that runs on every core repairs
each morsel inside the worker that just computed it, through `repair_range`, so
the rows are still in that core's cache and there is no second walk over the
column and no second parallel region. That is why `repair_range` takes a row
range rather than the kernel calling `apply_validity` at the end: a separate
parallel pass over the whole column measured slower than the serial one on a
column with no nulls, because the work it was splitting was already down to one
comparison per sixty four rows and the split cost more than it saved.

A morsel boundary is a multiple of sixty four rows, so the ranges fall between
words and no two workers ever touch the same one. `apply_validity` is the whole
column spelling, for the callers that are not splitting anything.
"""

from std.sys.info import simd_width_of

from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap


def apply_validity[dt: DType](mut col: Array[dt], var validity: Bitmap):
    """Zeroes the values under the nulls, then installs the bitmap.

    The bitmap is taken by value and moved into the column at the end. Doing the
    masking against a bitmap the column does not yet own is not an accident: it
    keeps the mutable pointer into the values buffer and the reads of the
    validity bits on two different objects.

    Args:
        col: The column to repair. Its length must match the bitmap's.
        validity: The validity the column should end up with.

    Parameters:
        dt: The column's dtype.
    """
    repair_range(col, validity, 0, len(col))
    col.data.validity = validity^


def repair_range[
    dt: DType
](mut col: Array[dt], validity: Bitmap, start: Int, stop: Int):
    """Zeroes the values under the nulls, over one range of rows.

    The bitmap is borrowed and not installed, because a caller that is splitting
    the column calls this once per morsel and can only hand the bitmap over once
    the last of them is done.

    Args:
        col: The column to repair. Its length must match the bitmap's.
        validity: The validity the column will end up with.
        start: The first row to repair. Must be a multiple of 64, which every
            morsel boundary is.
        stop: One past the last row to repair.

    Parameters:
        dt: The column's dtype.
    """
    comptime width = simd_width_of[dt]()
    var ptr = col.unsafe_ptr()
    var n = len(col)

    # Rounding the end up is what picks up the last partial word of the column;
    # `last` below is what keeps the writes inside the length.
    var first = start // 64
    var limit = min((stop + 63) // 64, validity.word_count())

    for w in range(first, limit):
        var word = validity.unsafe_word(w)
        if word == UInt64.MAX:
            continue

        var base = w * 64
        var last = base + 64
        if last > n:
            last = n

        if word == 0:
            var i = base
            while i + width <= last:
                ptr.unsafe_offset(i).unsafe_store(SIMD[dt, width](0))
                i += width
            while i < last:
                ptr.unsafe_offset(i).unsafe_write(Scalar[dt](0))
                i += 1
            continue

        for i in range(base, last):
            if (word >> UInt64(i - base)) & 1 == 0:
                ptr.unsafe_offset(i).unsafe_write(Scalar[dt](0))


def combined_validity(a: Bitmap, b: Bitmap) -> Bitmap:
    """Returns the validity a binary operation on two columns should produce.

    A result is present only where both inputs are, which is the Arrow rule and
    the pandas one.

    Args:
        a: The left column's validity.
        b: The right column's validity. Must be the same length as `a`.

    Returns:
        The intersection.
    """
    var out = Bitmap(copy=a)
    out.and_with(b)
    return out^
