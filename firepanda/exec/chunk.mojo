"""The unit of work an engine pushes.

A chunk is a horizontal slice of a frame: one array per column, all of the same
length, and nothing else. It has no schema and no column names, which is
deliberate. Names are a property of the plan and the plan is fixed before the
first row moves, so carrying them on every chunk would be copying a list of
strings a thousand times to learn something that was already known. `Pipeline`
holds the schema and each node knows what its own input looks like.

The size of a chunk is the same hundred and twenty eight thousand rows that
`firepanda/exec/morsel.mojo` hands out as a morsel, so a morsel is a chunk and a
kernel written to walk one walks the other. That number is not a guess. It comes
out of the same argument in DuckDB and in the Polars streaming engine: large
enough that the per call setup every kernel does is amortised over real work,
small enough that a chunk and its output fit in a core's L2 and never make the
trip to memory between one operator and the next. That second half is the whole
reason a chunked engine is faster than a whole column one, and it stops being
true the moment a chunk is sized in megabytes.

A chunk owns its arrays. There is no sharing and no reference counting, so an
operator that wants to keep a chunk keeps it and an operator that wants to
transform one consumes it. That is why every method here that produces a chunk
takes the old one by `var` rather than by reference.
"""

from firepanda.array.any import AnyArray


struct Chunk(Movable, Sized):
    """One horizontal slice of a frame, owned outright."""

    var columns: List[AnyArray]
    """One array per column, in plan order. All the same length."""

    var rows: Int
    """The number of rows, which every column agrees on.

    Carried rather than read off the first column because a chunk of no columns
    still has a row count, and a count of rows with nothing in them is what a
    `count(*)` over a projected away table is.
    """

    def __init__(out self, var columns: List[AnyArray]) raises:
        """Constructs a chunk, taking the row count from the columns.

        Args:
            columns: The arrays. Consumed.

        Raises:
            If the columns are not all the same length.
        """
        self.rows = 0 if len(columns) == 0 else len(columns[0])
        for i in range(len(columns)):
            if len(columns[i]) != self.rows:
                raise Error(
                    "chunk: column "
                    + String(i)
                    + " has "
                    + String(len(columns[i]))
                    + " rows and column 0 has "
                    + String(self.rows)
                )
        self.columns = columns^

    def __init__(out self, var columns: List[AnyArray], rows: Int):
        """Constructs a chunk whose row count the caller already knows.

        Unchecked, and that is the point: an operator that just built every
        column with the same loop bound does not need the lengths compared
        again on the hot path.

        Args:
            columns: The arrays. Consumed.
            rows: The number of rows.
        """
        self.columns = columns^
        self.rows = rows

    def __len__(self) -> Int:
        """Returns the number of rows.

        Returns:
            The row count.
        """
        return self.rows

    def width(self) -> Int:
        """Returns the number of columns.

        Returns:
            The column count.
        """
        return len(self.columns)

    def into_columns(deinit self) -> List[AnyArray]:
        """Gives up the arrays without copying them, consuming the chunk.

        Returns:
            The columns, in order.
        """
        return self.columns^
