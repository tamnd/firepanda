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
"""

from firepanda.dtype.logical import LogicalType

from .any import AnyArray


struct ChunkedArray(Movable, Sized):
    """A logical column made of one or more physical arrays."""

    var chunks: List[AnyArray]
    """The pieces, in row order. May be empty."""

    var type: LogicalType
    """The dtype every chunk has."""

    var _length: Int

    def __init__(out self, type: LogicalType):
        """Constructs a column with no chunks.

        Args:
            type: The dtype chunks must have.
        """
        self.chunks = List[AnyArray]()
        self.type = type
        self._length = 0

    def __init__(out self, var first: AnyArray):
        """Constructs a column from one chunk, taking its dtype.

        Args:
            first: The first chunk. Consumed.
        """
        self.type = first.type
        self._length = len(first)
        self.chunks = List[AnyArray]()
        self.chunks.append(first^)

    def __len__(self) -> Int:
        """Returns the total number of values across all chunks.

        Returns:
            The row count.
        """
        return self._length

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
        self._length += len(chunk)
        self.chunks.append(chunk^)

    def null_count(self) -> Int:
        """Returns the number of nulls across all chunks.

        Returns:
            The null count.
        """
        var total = 0
        for i in range(len(self.chunks)):
            total += self.chunks[i].null_count()
        return total

    def locate(self, index: Int) raises -> Tuple[Int, Int]:
        """Maps a row position to a chunk and an offset within it.

        This is a linear walk. It is fine for the handful of chunks a file
        produces and it is the wrong thing to call in a loop; kernels iterate
        chunks directly instead of indexing rows.

        Args:
            index: The row position.

        Returns:
            The chunk index and the offset inside that chunk.

        Raises:
            If the position is out of range.
        """
        if index < 0 or index >= self._length:
            raise Error(
                "row "
                + String(index)
                + " out of range for column of length "
                + String(self._length)
            )
        var remaining = index
        for i in range(len(self.chunks)):
            var n = len(self.chunks[i])
            if remaining < n:
                return (i, remaining)
            remaining -= n
        raise Error("chunk lengths do not sum to column length")
