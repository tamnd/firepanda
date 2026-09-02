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
from firepanda.kernel.concat import concat_any, concat_two_any
from firepanda.kernel.sort import is_sorted_any

from .any import AnyArray
from .data import ColumnData


struct Sortedness(Equatable, ImplicitlyCopyable, Movable, Writable):
    """What is known about the order of a column's values.

    A runtime tag on the column rather than something rediscovered per query.
    Two operators want it. A group by on a key whose equal values are adjacent
    needs no hash table at all, it walks the column and closes a group when the
    value changes, and that is the whole of Polars' sorted group by. A join of
    two sorted inputs is a merge and needs no hash table either.

    Null placement is deliberately not part of this. A column that has any nulls
    is never marked, because the two ends a sort can put them at would need two
    more states and neither operator that wants the flag can use them: a merge
    join has to know, and a group by would rather ask `null_count` and take the
    ordinary path. `nulls == 0` is already a field, so the check is free.
    """

    var code: UInt8
    """Which order."""

    def __init__(out self, code: UInt8):
        """Constructs an order.

        Args:
            code: Which order.
        """
        self.code = code

    comptime UNKNOWN = Self(0)
    """Nothing is known. Every column starts here and returns here when written."""

    comptime ASCENDING = Self(1)
    """Values do not decrease from one row to the next, and there are no nulls."""

    comptime DESCENDING = Self(2)
    """Values do not increase from one row to the next, and there are no nulls."""

    comptime CONSTANT = Self(3)
    """Every value is the same, which is both of the above at once.

    Worth a state of its own rather than picking one arbitrarily, because pandas
    says a constant series is monotonic increasing and monotonic decreasing, and
    a scan that answered only the first question would have to be run twice to
    answer the second."""

    comptime UNORDERED = Self(4)
    """Checked, and in no order. Distinct from `UNKNOWN` so a scan runs once."""

    def __eq__(self, other: Self) -> Bool:
        """Compares two orders.

        Args:
            other: The order to compare against.

        Returns:
            True if they are the same.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two orders for inequality.

        Args:
            other: The order to compare against.

        Returns:
            True if they differ.
        """
        return self.code != other.code

    def is_known(self) -> Bool:
        """Reports whether the question has been settled either way.

        Returns:
            True for anything but `UNKNOWN`.
        """
        return self.code != 0

    def is_ascending(self) -> Bool:
        """Reports whether the values never decrease.

        Returns:
            True for `ASCENDING` and for `CONSTANT`.
        """
        return self == Self.ASCENDING or self == Self.CONSTANT

    def is_descending(self) -> Bool:
        """Reports whether the values never increase.

        Returns:
            True for `DESCENDING` and for `CONSTANT`.
        """
        return self == Self.DESCENDING or self == Self.CONSTANT

    def write_to(self, mut writer: Some[Writer]):
        """Writes the name a user would recognise.

        Args:
            writer: The sink.
        """
        if self == Self.ASCENDING:
            writer.write("ascending")
        elif self == Self.DESCENDING:
            writer.write("descending")
        elif self == Self.CONSTANT:
            writer.write("constant")
        elif self == Self.UNORDERED:
            writer.write("unordered")
        else:
            writer.write("unknown")


struct ChunkedArray(Copyable, Movable, Sized):
    """A logical column made of one or more physical arrays."""

    var chunks: List[AnyArray]
    """The pieces, in row order. May be empty."""

    var starts: List[Int]
    """Where each chunk begins, plus the total. Always `len(chunks) + 1` long."""

    var type: LogicalType
    """The dtype every chunk has."""

    var nulls: Int
    """How many nulls the chunks hold between them, kept current on append."""

    var order: Sortedness
    """What is known about the order of the values. Never guessed, only set."""

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
        self.order = Sortedness.UNKNOWN

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
        self.order = Sortedness.UNKNOWN

    def __init__(out self, *, copy: Self):
        """Deep-copies a column, chunk by chunk.

        Args:
            copy: The column to copy.
        """
        self.chunks = List[AnyArray](capacity=len(copy.chunks))
        for i in range(len(copy.chunks)):
            self.chunks.append(AnyArray(copy=copy.chunks[i]))
        self.starts = List[Int](copy.starts)
        self.type = copy.type
        self.nulls = copy.nulls
        self.order = copy.order

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
        # Appending can break an order that held over what was already here, and
        # checking whether it did costs a comparison across the seam plus a scan
        # of the new chunk. That is `prove_sorted`'s job and it is the caller's
        # decision to pay for it, so the flag is dropped rather than defended.
        self.order = Sortedness.UNKNOWN
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

    def mark_sorted(mut self, order: Sortedness):
        """Records an order the caller already knows, without checking it.

        This is what a sort calls on its own key column and what a reader calls
        on a column a file says is sorted. It is unchecked on purpose: the caller
        has just done the work that establishes the order and rescanning would
        double it. A column holding a null is left alone, because the flag says
        nothing about where nulls went.

        Args:
            order: What is known.
        """
        if self.nulls > 0:
            return
        self.order = order

    def prove_sorted(mut self) raises -> Sortedness:
        """Finds out whether the column is sorted, and remembers the answer.

        One pass over the values, which is the price of turning a group by on
        this column into a walk instead of a hash table, so it is worth paying
        when the column is about to be grouped and not worth paying otherwise.
        The answer is cached either way, including a negative one, so asking
        twice costs one scan.

        Returns:
            What is now known about the order.

        Raises:
            If the column's dtype is not one firepanda can order.
        """
        if self.order.is_known() or self.nulls > 0:
            return self.order
        if len(self) < 2:
            self.order = Sortedness.CONSTANT
            return self.order
        var up = self._is_sorted(descending=False)
        var down = self._is_sorted(descending=True)
        if up and down:
            self.order = Sortedness.CONSTANT
        elif up:
            self.order = Sortedness.ASCENDING
        elif down:
            self.order = Sortedness.DESCENDING
        else:
            self.order = Sortedness.UNORDERED
        return self.order

    def _is_sorted(self, descending: Bool) raises -> Bool:
        """Checks one direction without flattening the column.

        Each chunk is checked on its own and then each seam between two chunks
        is checked as a pair of rows, which is the whole of what a walk over the
        flattened column would have found. Flattening first would copy every
        byte of the column to answer a question about its order.

        Args:
            descending: Check for largest first.

        Returns:
            Whether the column is sorted that way.

        Raises:
            If the column's dtype is not one firepanda can order.
        """
        for c in range(len(self.chunks)):
            if not is_sorted_any(self.chunks[c], descending):
                return False
        for c in range(len(self.chunks) - 1):
            var last = len(self.chunks[c])
            var seam = concat_two_any(
                self.chunks[c].slice(last - 1, last),
                self.chunks[c + 1].slice(0, 1),
            )
            if not is_sorted_any(seam, descending):
                return False
        return True

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


def wrap_columns(var columns: List[AnyArray]) -> List[ChunkedArray]:
    """Wraps each array as a column of one chunk, moving rather than copying.

    Every operator in the tree still produces a `List[AnyArray]`, and the frame
    holds `ChunkedArray`, so this is the seam between the two. It has to move: a
    copy here would put a full copy of every column back into the cost of every
    operation, which is the thing `only` exists to avoid.

    The pop and reverse is the same trick `DataFrame.from_series` documents. A
    loop over `columns^` binds a value with no origin and cannot give up an
    element, so the only way to take the arrays without copying them is to pop
    from the back, and the second loop puts them back in order.

    Args:
        columns: The arrays. Consumed.

    Returns:
        One single chunk column per array, in the same order.
    """
    var backwards = List[ChunkedArray](capacity=len(columns))
    while len(columns) > 0:
        backwards.append(ChunkedArray(columns.pop()))
    var out = List[ChunkedArray](capacity=len(backwards))
    while len(backwards) > 0:
        out.append(backwards.pop())
    return out^
