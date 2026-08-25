"""Reordering and dropping rows: take and filter.

These two are where the vectorized style runs out. A gather reads a different
cache line per row and a compaction writes a variable number of them, so neither
loop has a shape the vector unit helps with on the targets firepanda builds for.
What can be done is to keep the branches out of the value loop, and both kernels
do that in the same way: read unconditionally, because a null holds a zero and
reading it is harmless, and build the validity bitmap separately.

`take_rows` treats a negative index as a null. That is not a convenience, it is
how a left join reports that the row on the right did not exist, and it is why
the index list is signed.

`filter_rows` drops the rows where the mask is null. The alternative, keeping
them, would mean `filter(m)` and `filter(not m)` both contain the same row, which
no query engine does and pandas does not either.
"""

from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap


def take_rows[dt: DType](col: Array[dt], indices: List[Int]) -> Array[dt]:
    """Gathers rows by position.

    Args:
        col: The column to gather from.
        indices: The positions to gather. Each is either a valid position in
            `col` or negative, which produces a null.

    Parameters:
        dt: The dtype.

    Returns:
        A column of length `len(indices)`.
    """
    var n = len(indices)
    var out = Array[dt](n)
    var source = col.unsafe_ptr()
    var target = out.unsafe_ptr()
    var validity = Bitmap(n, all_valid=False)

    # The output positions are consecutive, so the validity bits can be built in
    # a register and stored once every sixty four rows instead of read-modify-
    # writing a byte per row. The input side has no such luck; a gather is a
    # gather.
    var word = UInt64(0)
    for i in range(n):
        var at = indices[i]
        if at >= 0 and col.data.validity.get(at):
            target.unsafe_offset(i).unsafe_write(
                source.unsafe_offset(at).unsafe_load()
            )
            word |= UInt64(1) << UInt64(i & 63)
        if i & 63 == 63:
            validity.unsafe_set_word(i >> 6, word)
            word = 0

    if n & 63 != 0:
        validity.unsafe_set_word(n >> 6, word)

    out.data.validity = validity^
    return out^


def filter_rows[
    dt: DType
](col: Array[dt], mask: Array[DType.bool]) -> Array[dt]:
    """Keeps the rows where the mask is true.

    Two passes. The first counts the kept rows so the output can be allocated
    once at the right size, the second copies. Growing a buffer instead would
    save the counting pass and cost a reallocation and a copy of everything
    already written, several times, on a column that is usually large.

    The second pass comes in two versions and the split is on whether the column
    being filtered has any nulls. It usually does not, and that case is worth a
    lot: with no validity to carry across, the copy loop has no branch left in it.

    Args:
        col: The column to filter.
        mask: The mask. Must be the same length as `col`.

    Parameters:
        dt: The dtype.

    Returns:
        A column holding the kept rows, in their original order.
    """
    var n = len(mask)
    var mask_values = mask.unsafe_ptr()

    var kept = 0
    for i in range(n):
        if not mask.data.validity.get(i):
            continue
        if Bool(mask_values.unsafe_offset(i).unsafe_load()):
            kept += 1

    var out = Array[dt](kept)
    var source = col.unsafe_ptr()
    var target = out.unsafe_ptr()

    if col.null_count() == 0:
        # Nothing to record, because a filter of a column with no nulls has no
        # nulls, and `Array` starts out all present. That leaves a loop with no
        # branch in it at all: every row is written at the output cursor and the
        # cursor advances by the mask bit, so a dropped row is simply overwritten
        # by the next one. The mask is data and the branch predictor cannot learn
        # it, which is why removing the branch is worth writing a row nobody
        # keeps.
        var written = 0
        var i = 0
        while written < kept:
            var present = mask.data.validity.get(i)
            var truthy = Bool(mask_values.unsafe_offset(i).unsafe_load())
            target.unsafe_offset(written).unsafe_write(
                source.unsafe_offset(i).unsafe_load()
            )
            written += Int(present and truthy)
            i += 1
        return out^

    var validity = Bitmap(kept, all_valid=False)

    # As in `take_rows`, the output positions are consecutive and the validity
    # goes down a word at a time. Here it matters more, because the row being
    # written is not the row being read and the byte the bit lives in would be a
    # second unpredictable memory reference per kept row.
    var at = 0
    var word = UInt64(0)
    for i in range(n):
        if not mask.data.validity.get(i):
            continue
        if not Bool(mask_values.unsafe_offset(i).unsafe_load()):
            continue
        target.unsafe_offset(at).unsafe_write(
            source.unsafe_offset(i).unsafe_load()
        )
        if col.data.validity.get(i):
            word |= UInt64(1) << UInt64(at & 63)
        at += 1
        if at & 63 == 0:
            validity.unsafe_set_word((at >> 6) - 1, word)
            word = 0

    if at & 63 != 0:
        validity.unsafe_set_word(at >> 6, word)

    out.data.validity = validity^
    return out^
