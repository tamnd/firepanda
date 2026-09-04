"""Elementwise comparison of text columns, producing a boolean column.

`compare.mojo` holds the same six operations for the fixed width dtypes and does
them a register at a time. Text cannot be done that way. An element is a run of
bytes of its own length, so the comparison is a loop over bytes and the answer
for one row says nothing about how long the next row takes. That is why this is a
file of its own rather than another arm of the numeric dispatch.

The operation codes are the `CMP_` ones from `compare.mojo`, reused rather than
redefined, because the caller that picks between the two files picks the code
once and does not want to know which family it ended up in.

Equality and ordering take different routes on purpose. Equality can be settled
by the view alone whenever the two strings are short, since a short view holds
the whole string zero padded into sixteen bytes and two of them are equal exactly
when the strings are. Ordering cannot use that trick, because the bytes are
packed into words in an order that makes equality one compare and makes ordering
wrong, so it goes to the byte loop.

The constant form hoists what it can out of the loop. A short constant is turned
into a view once, before any row is read, and then each row of a column of short
strings costs four register compares and no memory traffic at all beyond the
views. That is the shape a filter on a status column or a country code has, and
it is the case worth being fast.

Comparison against a null is null, exactly as it is for numbers, and it is
handled the same way: the loop writes whatever falls out and the repair at the
end of each morsel clears the rows where either side was missing. The value
under a null is false either way, so nothing depends on which branch the loop
took there.

Both forms run on every core. A row's cost here depends on its own bytes and on
nothing else, so the column splits over morsels with no more care than a numeric
kernel needs, and the only thing that had to move is the constant's view, which
is now built once above the split rather than once per worker.
"""

from std.collections.span import Span

from firepanda.array.array import Array
from firepanda.array.strings import StringArray
from firepanda.array.strview import (
    INLINE_CAPACITY,
    StringView,
    make_inline,
    views_equal_short,
)
from firepanda.bitmap.bitmap import Bitmap
from firepanda.exec import parallel_morsels

from .compare import CMP_EQ, CMP_GE, CMP_GT, CMP_LE, CMP_LT, CMP_NE
from .mask import combined_validity, repair_range


def compare_text[
    op: Int
](a: StringArray, b: StringArray) raises -> Array[DType.bool]:
    """Compares two text columns elementwise.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        op: One of the `CMP_` codes from `compare.mojo`.

    Returns:
        A bool column, null wherever either input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    # Every row is written below, so the zeroing allocation is a wasted pass.
    var out = Array[DType.bool](overwritten=n)
    var validity = combined_validity(a.validity, b.validity)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_ptr()
        comptime if op == CMP_EQ or op == CMP_NE:
            for i in range(start, stop):
                # `equals` answers False for a null on the left, and the bytes
                # of a null on the right are empty, so a null row can come out
                # either way here. The repair pass below settles it.
                var same = a.equals(i, b.unsafe_bytes(i))
                comptime if op == CMP_EQ:
                    dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](same))
                else:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](not same)
                    )
        else:
            for i in range(start, stop):
                var order = a.compare(i, b.unsafe_bytes(i))
                comptime if op == CMP_LT:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order < 0)
                    )
                elif op == CMP_LE:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order <= 0)
                    )
                elif op == CMP_GT:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order > 0)
                    )
                else:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order >= 0)
                    )

        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def compare_text_const[
    op: Int
](a: StringArray, b: Span[UInt8, _]) raises -> Array[DType.bool]:
    """Compares a text column against one run of bytes.

    There is no flipped form. A constant on the left is the mirrored operation
    on the right, and the caller does that swap, which is the same arrangement
    the numeric constant kernel has.

    Args:
        a: The column.
        b: The constant's bytes. Borrowed for the length of the call and not
            stored.

    Parameters:
        op: One of the `CMP_` codes from `compare.mojo`.

    Returns:
        A bool column, null wherever the column is null. A null constant makes
        the whole answer null and never reaches here.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    # Every row is written below, so the zeroing allocation is a wasted pass.
    var out = Array[DType.bool](overwritten=n)
    var validity = Bitmap(copy=a.validity)

    # A short constant becomes a view once, before any row is read and before
    # the split, so the workers share it rather than each making their own. A
    # long one cannot: the view for a long string points into a payload this
    # constant does not live in, so those rows go to the byte loop.
    var short = len(b) <= INLINE_CAPACITY
    var probe = StringView()
    comptime if op == CMP_EQ or op == CMP_NE:
        if short:
            probe = make_inline(b)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_ptr()
        comptime if op == CMP_EQ or op == CMP_NE:
            for i in range(start, stop):
                var same: Bool
                if short:
                    same = views_equal_short(a.view(i), probe)
                else:
                    same = a.equals(i, b)
                comptime if op == CMP_EQ:
                    dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](same))
                else:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](not same)
                    )
        else:
            for i in range(start, stop):
                var order = a.compare(i, b)
                comptime if op == CMP_LT:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order < 0)
                    )
                elif op == CMP_LE:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order <= 0)
                    )
                elif op == CMP_GT:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order > 0)
                    )
                else:
                    dst.unsafe_offset(i).unsafe_write(
                        Scalar[DType.bool](order >= 0)
                    )

        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^
