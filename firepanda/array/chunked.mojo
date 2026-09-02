"""A column stored as a sequence of arrays.

Reading a Parquet file gives one array per row group. Concatenating two frames
gives one array per input. Materializing those into a single contiguous array
costs a copy of the whole column, and for the operations that dominate a
dataframe workload, scan, filter, aggregate, that copy buys nothing because those
operations are happy to run per chunk.

`ChunkedArray` therefore defers the copy until something actually needs contiguous
memory, and `combine` is the explicit place that happens.

Every chunk has the same dtype. That invariant is checked on append and is what
lets a kernel dispatch once for the whole column rather than once per chunk.

Two things here exist for the frame that is about to be built on top of this
rather than for the reader that has been using it. The first is `starts`, a prefix
sum of the chunk lengths, so that finding the chunk a row lives in is a binary
search over a small array of integers rather than a walk that adds up lengths
again on every call. The second is `only`, which hands back the single chunk of a
column that has one, by reference. Almost every column in the system has exactly
one chunk, every kernel written before chunking wants an `AnyArray`, and the way
those two facts meet has to be a borrow. If it were a copy then putting this type
in front of the frame's columns would reintroduce the whole-column deep copy that
was taken out of `take` and `filter`, on every operation, which is the opposite of
the point.
"""

from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.logical import LogicalType
from firepanda.kernel.concat import concat_any

from .any import AnyArray
from .data import ColumnData


struct ChunkedArray(Movable, Sized):
    """A logical column made of one or more physical arrays."""

    var chunks: List[AnyArray]
    """The pieces, in row order. May be empty."""

    var starts: List[Int]
    """Where each chunk begins, plus the total. Always `len(chunks) + 1` long."""

    var type: LogicalType
    """The dtype every chunk has."""

    var nulls: Int
    """How many nulls the chunks hold between them, kept current on append."""

    def __init__(out self, type: LogicalType):
        """Constructs a column with no chunks.

        Args:
            type: The dtype chunks must have.
        """
        self.chunks = List[AnyArray]()
        self.starts = List[Int]()
        self.starts.append(0)
        self.type = type
        self.nulls = 0

    def __init__(out self, var first: AnyArray):
        """Constructs a column from one chunk, taking its dtype.

        Args:
            first: The first chunk. Consumed.
        """
        self.type = first.type
        self.nulls = first.null_count()
        self.starts = List[Int]()
        self.starts.append(0)
        self.starts.append(len(first))
        self.chunks = List[AnyArray]()
        self.chunks.append(first^)

    def __len__(self) -> Int:
        """Returns the total number of values across all chunks.

        Returns:
            The row count.
        """
        return self.starts[len(self.starts) - 1]

    def num_chunks(self) -> Int:
        """Returns the number of chunks.

        Returns:
            The chunk count.
        """
        return len(self.chunks)

    def dtype(self) -> DType:
        """Returns the physical dtype of every chunk.

        Returns:
            The dtype.
        """
        return self.type.physical

    def append(mut self, var chunk: AnyArray) raises:
        """Adds a chunk at the end.

        A chunk of no rows is dropped rather than kept. It would make `starts`
        hold two equal entries, which would mean a row position could name two
        chunks and the binary search in `locate` could return the empty one.

        Args:
            chunk: The chunk to add. Consumed.

        Raises:
            If the chunk's dtype differs from the column's.
        """
        if chunk.type != self.type:
            raise Error(
                "chunk dtype "
                + String(chunk.type)
                + " does not match column dtype "
                + String(self.type)
            )
        if len(chunk) == 0:
            return
        self.nulls += chunk.null_count()
        self.starts.append(self.starts[len(self.starts) - 1] + len(chunk))
        self.chunks.append(chunk^)

    def null_count(self) -> Int:
        """Returns the number of nulls across all chunks.

        Returns:
            The null count.
        """
        return self.nulls

    def only(ref self) raises -> ref[self.chunks[0]] AnyArray:
        """Returns the single chunk of a column that has exactly one.

        This is the borrow that lets every kernel written against `AnyArray`
        keep working with a chunked column in front of it. It has to be a
        reference: a column of ten million rows has one chunk in almost every
        case that matters, and handing back a copy of it would put a full column
        copy back into the cost of every operation.

        Returns:
            A reference to the chunk, valid as long as this column is.

        Raises:
            If the column does not have exactly one chunk.
        """
        if len(self.chunks) != 1:
            raise Error(
                "column has "
                + String(len(self.chunks))
                + " chunks, not one; call combine() first"
            )
        return self.chunks[0]

    def combine(deinit self) raises -> AnyArray:
        """Flattens the column into one contiguous array, consuming it.

        The single chunk case is a move and copies nothing, which is what makes
        this safe to call from an operator that has not been taught about chunks
        yet. Everything else stacks the chunks the way `concat` does.

        Returns:
            One array holding every row in order.

        Raises:
            If the chunks cannot be stacked.
        """
        var held = self.chunks^
        if len(held) == 1:
            return held.pop()
        if len(held) == 0:
            return AnyArray(
                ColumnData(Buffer(0), Bitmap(0), 0),
                self.type,
            )
        return concat_any(held)

    def locate(self, index: Int) raises -> Tuple[Int, Int]:
        """Maps a row position to a chunk and an offset within it.

        `starts` is sorted and no two entries are equal, because `append` drops
        an empty chunk, so the chunk holding a row is the last entry that is not
        past it and a binary search finds it. The walk this replaces added up the
        chunk lengths again on every call, which is fine for the handful of
        chunks a file produces and is not fine once a column is cut into pieces
        of a hundred and twenty eight thousand rows.

        Args:
            index: The row position.

        Returns:
            The chunk index and the offset inside that chunk.

        Raises:
            If the position is out of range.
        """
        if index < 0 or index >= len(self):
            raise Error(
                "row "
                + String(index)
                + " out of range for column of length "
                + String(len(self))
            )
        var low = 0
        var high = len(self.chunks) - 1
        while low < high:
            var mid = (low + high + 1) // 2
            if self.starts[mid] <= index:
                low = mid
            else:
                high = mid - 1
        return (low, index - self.starts[low])
