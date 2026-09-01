"""Reordering and dropping rows: take and filter.

These two are where the vectorized style runs out. A gather reads a different
cache line per row and a compaction writes a variable number of them, so neither
loop has a shape the vector unit helps with on the targets firepanda builds for.
What can be done is to keep the branches out of the value loop, and both kernels
do that in the same way: read unconditionally, because a null holds a zero and
reading it is harmless, and build the validity bitmap separately.

What the vector unit will not do, the other cores will. A gather's output row
depends on its own index and on nothing else in the output, so `take_rows` splits
by output row, on boundaries rounded to a multiple of sixty four because the
validity bitmap is the one thing in there that is not per row. `filter_rows` has
no such shape: where a row lands depends on how many rows before it survived.

`take_rows` treats a negative index as a null. That is not a convenience, it is
how a left join reports that the row on the right did not exist, and it is why
the index list is signed.

`filter_rows` drops the rows where the mask is null. The alternative, keeping
them, would mean `filter(m)` and `filter(not m)` both contain the same row, which
no query engine does and pandas does not either.

Both kernels come in two spellings. The typed one takes an `Array[dt]` and is
what other kernels call. The erased one takes an `AnyArray` and is what a
`DataFrame` calls, because a frame holds a list of columns whose dtypes are only
known at runtime and differ from each other. They share a body: the typed entry
point hands its pointer and bitmap to the core, and the erased one walks `ALL`
and hands over the same two things. Routing the erased case through
`AnyArray.as_typed` instead would have been three lines shorter and would have
deep copied every column on the way in, which on a filter is the entire cost of
the operation paid twice.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.exec import parallel_morsels


comptime PARALLEL_TAKE_ROWS = 1 << 16
"""Below this many gathered rows the take stays on one thread.

A gather is a cache miss a row, so it is a great deal more than a handful of
nanoseconds and the split pays off sooner than it does for a loop over
consecutive memory. Half of `join`'s threshold, and picked the same way.
"""

comptime TAKE_MORSEL_ROWS = 1 << 16
"""Output rows a worker takes at a time once the gather is on every core.

A multiple of sixty four, which is what makes the morsel boundaries land on
validity word boundaries and is the only coupling between this number and the
loop below.

Small, because the cost of a gathered row is the cost of the cache miss it takes
and that is not the same for every row: indices that walk a small region are hot
and indices that walk the whole column are not, and a join or a sort produces
both in the same call. Eight thousand rows is around ten microseconds of work,
which is four orders of magnitude more than the atomic that hands it out.
"""


def take_rows[
    dt: DType
](col: Array[dt], indices: List[Int]) raises -> Array[dt]:
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
    return _take_core(
        col.unsafe_ptr(), col.data.validity, col.null_count() > 0, indices
    )


def take_any(col: AnyArray, indices: List[Int]) raises -> AnyArray:
    """Gathers rows by position from a column whose dtype is a runtime value.

    Args:
        col: The column to gather from.
        indices: The positions to gather, negative meaning null, as in
            `take_rows`.

    Returns:
        A column of length `len(indices)` with the same dtype as the input.

    Raises:
        If the column's dtype is not one firepanda has a physical layout for.
    """
    if col.is_string():
        return AnyArray(_take_strings(col.strings(), indices))
    comptime for candidate in ALL:
        if col.dtype() == candidate:
            return AnyArray(
                _take_core(
                    col.unsafe_ptr[candidate](),
                    col.data.validity,
                    col.null_count() > 0,
                    indices,
                )
            )
    raise Error("take: unsupported dtype")


def _take_strings(col: StringArray, indices: List[Int]) raises -> StringArray:
    """Gathers variable width rows by position.

    None of what `_take_core` does applies here. The output element is not a
    fixed number of bytes, so it cannot be written unconditionally at a computed
    offset, and the validity cannot be built a word at a time ahead of the
    values. What is left is the straightforward loop, and it is still the right
    shape: a short element is copied into its own view with no payload touched at
    all, which is the case that a gather of a key column actually hits.

    Args:
        col: The column to gather from.
        indices: The positions to gather. A negative index produces a null, as
            in `take_rows`, because that is how a left join reports a row that
            was not there.

    Returns:
        A column of length `len(indices)`.

    Raises:
        If an index is neither negative nor a position in the column.
    """
    var builder = StringBuilder(capacity=len(indices))
    for k in range(len(indices)):
        var at = indices[k]
        if at >= len(col):
            raise Error(
                String("take index ", at, " is outside a column of ", len(col))
            )
        if at < 0 or not col.is_valid(at):
            builder.append_null()
        else:
            builder.append(col.unsafe_bytes(at))
    return builder^.finish()


def _take_core[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_nulls: Bool,
    indices: List[Int],
) raises -> Array[dt]:
    """The gather loop, over a pointer and a bitmap rather than a column."""
    var n = len(indices)
    var out = Array[dt](n)
    var built = Bitmap(n, all_valid=False)

    # Output row `i` depends on `indices[i]` and on nothing else in the output,
    # so the gather splits by output row. The morsel size is a multiple of sixty
    # four so that no two workers write the same validity word, which is the only
    # thing here that is not per row.
    def gather(start: Int, stop: Int) raises {mut out, mut built, imm}:
        var target = out.unsafe_ptr()

        # The output positions are consecutive, so the validity bits can be
        # built in a register and stored once every sixty four rows instead of
        # read-modify-writing a byte per row. The input side has no such luck; a
        # gather is a gather.
        #
        # The validity probe is the second random read of the row, into a
        # different array from the values, and a column with no nulls does not
        # need it. A join gathers with a list that has negatives in it and a
        # source that usually does not have nulls, so the two halves of that
        # condition are worth keeping apart.
        var word = UInt64(0)
        for i in range(start, stop):
            var at = indices[i]
            if at >= 0 and (not has_nulls or validity.get(at)):
                target.unsafe_offset(i).unsafe_write(
                    source.unsafe_offset(at).unsafe_load()
                )
                word |= UInt64(1) << UInt64(i & 63)
            if i & 63 == 63:
                built.unsafe_set_word(i >> 6, word)
                word = 0

        # Only a morsel that ends part way through a word has anything left in
        # the register, and since the morsel is a multiple of sixty four that is
        # only ever the last one.
        if stop & 63 != 0 and stop > start:
            built.unsafe_set_word(stop >> 6, word)

    if n < PARALLEL_TAKE_ROWS:
        gather(0, n)
    else:
        parallel_morsels(gather, n, TAKE_MORSEL_ROWS)

    out.data.validity = built^
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
    return _filter_core(
        col.unsafe_ptr(), col.data.validity, col.null_count() > 0, mask
    )


def filter_any(col: AnyArray, mask: Array[DType.bool]) raises -> AnyArray:
    """Keeps the rows where the mask is true, for a runtime dtype.

    Args:
        col: The column to filter.
        mask: The mask. Must be the same length as `col`.

    Returns:
        A column holding the kept rows, with the same dtype as the input.

    Raises:
        If the column's dtype is not one firepanda has a physical layout for.
    """
    if col.is_string():
        return AnyArray(_filter_strings(col.strings(), mask))
    comptime for candidate in ALL:
        if col.dtype() == candidate:
            return AnyArray(
                _filter_core(
                    col.unsafe_ptr[candidate](),
                    col.data.validity,
                    col.null_count() > 0,
                    mask,
                )
            )
    raise Error("filter: unsupported dtype")


def _filter_strings(
    col: StringArray, mask: Array[DType.bool]
) raises -> StringArray:
    """Keeps the variable width rows where the mask is true.

    A null in the mask drops the row, the same rule `_filter_core` follows and
    for the same reason. There is no branch free version of this one: the
    trick in `_filter_core` is to write every row and advance the cursor by the
    mask bit, which only works when a row that nobody keeps costs a fixed number
    of bytes that the next row overwrites.

    Args:
        col: The column to filter.
        mask: The mask. Must be as tall as the column.

    Returns:
        A column holding the kept rows in their original order.

    Raises:
        If the mask is not as tall as the column.
    """
    if len(mask) != len(col):
        raise Error(
            String(
                "filter mask has ",
                len(mask),
                " rows and the column has ",
                len(col),
            )
        )
    var values = mask.unsafe_ptr()
    var builder = StringBuilder(capacity=len(col))
    for i in range(len(col)):
        if not mask.data.validity.get(i):
            continue
        if not Bool(values.unsafe_offset(i).unsafe_load()):
            continue
        if col.is_valid(i):
            builder.append(col.unsafe_bytes(i))
        else:
            builder.append_null()
    return builder^.finish()


def _filter_core[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    mask: Array[DType.bool],
) -> Array[dt]:
    """The compaction loop, over a pointer and a bitmap rather than a column."""
    var n = len(mask)
    var mask_values = mask.unsafe_ptr()

    var kept = 0
    for i in range(n):
        if not mask.data.validity.get(i):
            continue
        if Bool(mask_values.unsafe_offset(i).unsafe_load()):
            kept += 1

    var out = Array[dt](kept)
    var target = out.unsafe_ptr()

    if not has_null:
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

    var built = Bitmap(kept, all_valid=False)

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
        if validity.get(i):
            word |= UInt64(1) << UInt64(at & 63)
        at += 1
        if at & 63 == 0:
            built.unsafe_set_word((at >> 6) - 1, word)
            word = 0

    if at & 63 != 0:
        built.unsafe_set_word(at >> 6, word)

    out.data.validity = built^
    return out^
