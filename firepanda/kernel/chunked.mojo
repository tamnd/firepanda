"""Chunk walking entry points for the whole column kernels.

Every kernel in `firepanda/kernel` takes an `AnyArray` and returns one, and none
of them are going to stop doing that. They are the contiguous inner loops and they
are where the vectorization and the parallelism live. What was missing was the
layer above: a column is a `ChunkedArray` now, and something has to decide how to
apply a contiguous kernel to a column that is in pieces.

For the elementwise family that decision is simple, because a row's output depends
on that row and nothing else. Cut the operation the same way the column is cut,
call the existing kernel once per chunk, and keep the pieces. The chunking is
preserved rather than flattened, which is the whole point: a column read from a
file with sixteen row groups stays sixteen chunks through a filter and a cast, and
never has to exist as one eighty megabyte array at all.

`take` is the one that does not fit. Its output row `i` comes from input row
`indices[i]`, which may be in any chunk, so there is no way to cut it into
independent per chunk calls without also solving how to put the pieces back in
order, and putting them back in order is another `take`. It combines first and
says so below. The selection vector work in M2b is what fixes that properly, by
letting the output be a set of positions into the input rather than a new array.

Nothing here parallelizes across chunks. The kernels these call are already
parallel inside, and running them on all cores one after another is better than
running each on one core, because the last chunk of a column is usually smaller
than the rest and a per chunk task split would leave a core idle at the end of
every operation. If chunks get small enough for the per call overhead to show, the
fix is bigger chunks, not a second scheduler.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.dtype.logical import LogicalType, logical_for

from .cast import cast_any
from .select import filter_any, take_any


def _one_empty_chunk(
    var out: ChunkedArray, source: ChunkedArray
) raises -> ChunkedArray:
    """Gives a column that came out empty one empty chunk to be empty in.

    A column of no rows is either no chunks at all or one chunk of no rows, and
    the difference is invisible until something calls `only()`, which is the
    borrow every kernel written against `AnyArray` reaches through. A column of
    no chunks raises there. So a filter that matches nothing produces a frame
    that cannot be written, aggregated or printed, and it does it on the most
    ordinary line anybody writes, which is a predicate that happens to select
    nothing today.

    The rest of the package already answers this the other way. `take` with no
    indices returns one empty chunk, the Arrow reader gives one empty chunk to a
    file with no record batches, and `ChunkedArray.combine` builds one when it
    has none. This brings the two producers that did not agree into line rather
    than teaching a hundred call sites to expect either.

    Slicing an existing chunk to nothing rather than building an array from
    scratch is what keeps this correct for a type with children. An empty string
    view column is not an empty buffer, it has an offsets buffer and a data
    buffer of its own, and the way to get an empty one of exactly the right
    shape is to ask a full one for none of its rows.

    Args:
        out: The column that was built. Consumed.
        source: The column it was built from, for a chunk to take the shape of.

    Returns:
        `out` when it has any rows or when the source had no chunks to copy the
        shape of, and a column of one empty chunk otherwise.

    Raises:
        Error: If the source chunk cannot be sliced.
    """
    if out.num_chunks() > 0 or source.num_chunks() == 0:
        return out^
    return ChunkedArray(source.chunks[0].slice(0, 0))


def filter_chunked(
    col: ChunkedArray, mask: Array[DType.bool]
) raises -> ChunkedArray:
    """Keeps the rows a mask is true on, chunk by chunk.

    Each chunk gets the slice of the mask that covers it and the result keeps
    the same number of chunks, minus the ones that nothing survived, because
    `append` drops an empty chunk. A column that nothing survived at all would
    therefore be a column of no chunks, and that shape raises in `only()`, so a
    filter matching nothing produced a frame that could not be written or
    aggregated. It gets one empty chunk instead, which is what `take` and the
    Arrow reader already give a column of no rows.

    Args:
        col: The column to filter.
        mask: The mask. Must be the same length as the column.

    Returns:
        The kept rows, still in pieces.

    Raises:
        If the mask length differs from the column length, or a chunk's dtype is
        not one firepanda has a layout for.
    """
    if len(mask) != len(col):
        raise Error(
            "filter: mask has "
            + String(len(mask))
            + " rows and the column has "
            + String(len(col))
        )
    var out = ChunkedArray(col.type)
    for c in range(col.num_chunks()):
        var start = col.starts[c]
        var end = col.starts[c + 1]
        out.append(filter_any(col.chunks[c], mask.slice(start, end)))
    return _one_empty_chunk(out^, col)


def slice_chunked(
    col: ChunkedArray, start: Int, end: Int
) raises -> ChunkedArray:
    """Returns a half-open range of the column, keeping the chunk boundaries.

    A slice touches at most two chunks partially and takes the rest whole, and
    the whole ones are copied as they are rather than being cut and rejoined.
    That matters for `head` and `tail`, which are the two things anybody types at
    a prompt: at a hundred rows out of ten million this copies one chunk's worth
    of bytes at most, where flattening first would copy the column.

    Args:
        col: The column.
        start: The first row, inclusive.
        end: The last row, exclusive.

    Returns:
        The rows in the range, in as many pieces as they spanned.

    Raises:
        If the range is outside the column.
    """
    if start < 0 or end > len(col) or start > end:
        raise Error(
            "slice: range ["
            + String(start)
            + ", "
            + String(end)
            + ") is outside a column of "
            + String(len(col))
            + " rows"
        )
    var out = ChunkedArray(col.type)
    for c in range(col.num_chunks()):
        var at = col.starts[c]
        var stop = col.starts[c + 1]
        if stop <= start or at >= end:
            continue
        var lo = start - at if start > at else 0
        var hi = end - at if end < stop else stop - at
        out.append(col.chunks[c].slice(lo, hi))
    return _one_empty_chunk(out^, col)


def cast_chunked(
    col: ChunkedArray, to: LogicalType, strict: Bool = True
) raises -> ChunkedArray:
    """Converts every chunk to another type, keeping the chunk boundaries.

    Args:
        col: The column to convert.
        to: The target type.
        strict: Whether a text value that is not a number raises rather than
            becoming a null.

    Returns:
        The converted column, in the same number of pieces.

    Raises:
        If the conversion is not one firepanda has, or `strict` and a text value
        is not a number.
    """
    var out = ChunkedArray(to)
    for c in range(col.num_chunks()):
        out.append(cast_any(col.chunks[c], to, strict))
    return out^


def cast_chunked(
    col: ChunkedArray, to: DType, strict: Bool = True
) raises -> ChunkedArray:
    """Converts every chunk to another dtype, keeping the chunk boundaries.

    The overload that names a `LogicalType` can ask for text; this one cannot,
    text having no dtype of its own, and it is what `astype` calls.

    Args:
        col: The column to convert.
        to: The target dtype.
        strict: Whether a text value that is not a number raises rather than
            becoming a null.

    Returns:
        The converted column, in the same number of pieces.

    Raises:
        If the conversion is not one firepanda has, or `strict` and a text value
        is not a number.
    """
    var out = ChunkedArray(logical_for(to))
    for c in range(col.num_chunks()):
        out.append(cast_any(col.chunks[c], to, strict))
    return out^


def take_chunked(col: ChunkedArray, indices: List[Int]) raises -> ChunkedArray:
    """Gathers rows by position, flattening the column first if it has to.

    Output row `i` comes from input row `indices[i]`, which can be in any chunk,
    so unlike the elementwise kernels there is no cut of this that turns into
    independent per chunk calls. The pieces would come out grouped by which chunk
    they came from rather than in the order asked for, and putting them back in
    order is another gather.

    A column of one chunk, which is every column in the tree today, skips the
    flattening entirely and this is exactly `take_any`. A column of several pays
    one copy of the input, which is worth saying out loud because it is the one
    place chunking currently costs something. The fix is the selection vector
    work, where the output is a set of positions into the input and there is
    nothing to flatten.

    Args:
        col: The column to gather from.
        indices: The rows to take, in output order. A negative index is a null.

    Returns:
        The gathered rows, as one chunk.

    Raises:
        If the column's dtype is not one firepanda has a layout for.
    """
    if col.num_chunks() == 1:
        return ChunkedArray(take_any(col.only(), indices))
    var flat = ChunkedArray(copy=col).combine()
    return ChunkedArray(take_any(flat, indices))
