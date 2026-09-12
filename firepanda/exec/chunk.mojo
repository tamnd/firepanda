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

## The selection

A chunk may carry a selection, which is a list of positions into its arrays.
Section 4 of `docs/specs/engine/02-execution-model.md` and issue #521. The point
is that filtering a column means writing a new one, and a filter that writes a
list of positions instead has moved one number per surviving row rather than one
value per surviving row per column.

That makes a chunk's columns two different things, and the `dense` list is what
says which is which. A dense column's element i is row i of the chunk. A column
that is not dense has its row i at element `picks[i]`. A chunk with no selection
has every column dense, which is what every chunk was before this existed and
what most chunks still are.

The two are both needed because an operator that computes something computes it
over the chunk's rows and not over the array positions underneath them, so its
output is dense while its inputs were not. Keeping the distinction per column is
what lets a filter leave the columns it did not touch exactly as the scan handed
them over.

`flatten` puts a chunk back to all dense, and every operator that has not been
taught to read through a selection has it called on its input first, so adding
this changes no answer anywhere.
"""

from firepanda.array.any import AnyArray
from firepanda.kernel.select import gather_any


struct Chunk(Movable, Sized):
    """One horizontal slice of a frame, owned outright."""

    var columns: List[AnyArray]
    """One array per column, in plan order.

    All the same length when there is no selection. When there is one, a dense
    column is as long as the chunk has rows and the rest are as long as whatever
    they were gathered from.
    """

    var rows: Int
    """The number of rows, which every column agrees on.

    Carried rather than read off the first column because a chunk of no columns
    still has a row count, and a count of rows with nothing in them is what a
    `count(*)` over a projected away table is.
    """

    var picks: List[UInt32]
    """The selection, or empty when there is none.

    Four bytes a position. A selection points into one chunk and a chunk is a
    hundred and twenty eight thousand rows, so three bytes would do, and the
    width is worth having because `gather_any` reads one position for every
    value it moves.
    """

    var dense: List[Bool]
    """Whether each column's element i is row i, or empty when there is no
    selection."""

    def __init__(out self, var columns: List[AnyArray]) raises:
        """Constructs a chunk with no selection, taking the row count from the
        columns.

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
        self.picks = List[UInt32]()
        self.dense = List[Bool]()

    def __init__(out self, var columns: List[AnyArray], rows: Int):
        """Constructs a chunk with no selection whose row count is known.

        Unchecked, and that is the point: an operator that just built every
        column with the same loop bound does not need the lengths compared
        again on the hot path.

        Args:
            columns: The arrays. Consumed.
            rows: The number of rows.
        """
        self.columns = columns^
        self.rows = rows
        self.picks = List[UInt32]()
        self.dense = List[Bool]()

    def __init__(
        out self,
        var columns: List[AnyArray],
        var picks: List[UInt32],
        var dense: List[Bool],
    ):
        """Constructs a chunk under a selection.

        Unchecked, like the constructor above and for the same reason. The
        caller is the operator that just built the selection and it knows the
        lengths line up.

        Args:
            columns: The arrays. Consumed.
            picks: The selection, one position per row. Consumed.
            dense: Whether each column is already at the chunk's rows, one per
                column. Consumed.
        """
        self.rows = len(picks)
        self.columns = columns^
        self.picks = picks^
        self.dense = dense^

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

    def selected(self) -> Bool:
        """Whether the chunk carries a selection.

        Returns:
            True if some column is read through one.
        """
        return len(self.picks) > 0

    def flatten(mut self, spread: Bool = True) raises:
        """Gathers every column that is not dense and drops the selection.

        Does nothing to a chunk that has no selection, which is the common case
        and is why this is cheap to call unconditionally.

        Args:
            spread: Whether a gather may use more than one core. False when the
                caller is already running on a worker.

        Raises:
            If a column's dtype is not one firepanda has a layout for.
        """
        if not self.selected():
            return
        for i in range(len(self.columns)):
            if i < len(self.dense) and self.dense[i]:
                continue
            self.columns[i] = gather_any(self.columns[i], self.picks, spread)
        self.picks = List[UInt32]()
        self.dense = List[Bool]()

    def column(self, at: Int, spread: Bool = True) raises -> AnyArray:
        """Returns one column at the chunk's rows, gathering it if it has to.

        A copy either way, since the caller wants something contiguous. What it
        saves over `flatten` is the columns it was not asked about.

        Args:
            at: The column.
            spread: Whether a gather may use more than one core.

        Returns:
            An array of `rows` elements.

        Raises:
            If the position is out of range, or the dtype has no layout.
        """
        if at < 0 or at >= len(self.columns):
            raise Error(
                "chunk: column "
                + String(at)
                + " is outside a chunk of "
                + String(len(self.columns))
                + " columns"
            )
        if not self.selected() or (at < len(self.dense) and self.dense[at]):
            return AnyArray(copy=self.columns[at])
        return gather_any(self.columns[at], self.picks, spread)

    def append(mut self, var column: AnyArray, dense: Bool):
        """Puts a column on the end, saying whether it is at the chunk's rows.

        What `Compute` does, and the reason it is a method here rather than a
        rebuild there: an operator that only adds a column should not have to
        take the selection apart and put it back to do it.

        Args:
            column: The array. Consumed.
            dense: Whether its element i is row i. Ignored when the chunk has no
                selection, since then every column is.
        """
        self.columns.append(column^)
        if self.selected():
            self.dense.append(dense)

    def replace(mut self, at: Int, var column: AnyArray, dense: Bool) raises:
        """Puts a column in the place of another, which `Cast` does.

        Args:
            at: The position to replace.
            column: The array. Consumed.
            dense: Whether its element i is row i.

        Raises:
            If the position is outside the chunk.
        """
        if at < 0 or at >= len(self.columns):
            raise Error(
                "chunk: column "
                + String(at)
                + " is outside a chunk of "
                + String(len(self.columns))
                + " columns"
            )
        self.columns[at] = column^
        if self.selected():
            self.dense[at] = dense

    def into_raw_columns(deinit self) -> List[AnyArray]:
        """Gives up the arrays as they are, without flattening, consuming the
        chunk.

        For an operator that reads selections and has already read `picks` and
        `dense`. Anything else wants `into_columns`, since what comes back here
        is not one array per column at the chunk's rows and is only meaningful
        beside the selection it was under.

        Returns:
            The columns, in order, dense and not.
        """
        return self.columns^

    def into_columns(deinit self) raises -> List[AnyArray]:
        """Gives up the arrays, consuming the chunk.

        Flattens first, so what comes back is always one array per column at the
        chunk's rows. Nothing is copied when there was no selection, which is
        what the callers of this all expect.

        Returns:
            The columns, in order.

        Raises:
            If a column's dtype is not one firepanda has a layout for.
        """
        self.flatten()
        return self.columns^
