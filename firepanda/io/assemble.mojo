"""Building one frame out of many Arrow arrays, allocating each column once.

Every producer that hands us more than one array of the same schema wants the
same thing done with them. An IPC file has record batches, a Parquet reader has
row groups, a query result has result chunks, and in all three cases the caller
wants a frame rather than a list of pieces. The obvious way to get there is to
build a frame per piece and concatenate, and that writes every byte a second
time, on one thread, after the read has already done the work.

Nothing about a file makes that necessary. The arrays are all located before any
of them is read, so the finished row count is known in advance, and the only
thing that is not is how much payload the string columns need. So this asks each
array for that number first, which is a pass the copy was going to make anyway,
adds the answers up, allocates each column once and then tells every array the
place its rows go. Both passes run on every core and there is nothing to
concatenate at the end.

The unit of work is a range of rows rather than an array, because how a producer
chunks its output is its own decision and says nothing about how many cores are
sitting here. A single array of ten million rows would otherwise be one task per
column. An Arrow array has carried an offset since the beginning, so a range of
an array is an ordinary array and the importer needed nothing new to read one.

Validity is the one part that does not fill in place. A range boundary need not
land on a byte of the bitmap, and two threads writing the byte either side of one
would lose each other's bits, so each range builds its own bitmap and they are
pasted in afterwards on one thread. That is a bit per row rather than a value per
row and does not show up next to the copy it follows.

The output can come back in one chunk or in morsels, and the copy is the same
either way. A frame in one chunk is what an eager caller needs, because
`DataFrame.__getitem__` borrows a column only when there is a single chunk to
borrow. A frame in morsels is what the pipeline needs, because it runs its
elementwise operators a chunk per worker, so a frame in one chunk is a query on
one core no matter how many the machine has. Cutting one up afterwards is a
second copy of every byte, and putting the boundaries in while the first copy is
happening is free: a piece is trimmed so that it never crosses one, and it then
writes into its own morsel's sink at an offset rather than into a long sink at a
larger one.

There is no default height here on purpose. Which height is fastest is a property
of the query, the machine's cache and how many cores are free, and measuring it
wants an idle machine, so the caller names a height and this file has no opinion
about it. The opinion belongs wherever the plan layer decides to run a query
through the pipeline, which is not written yet.

Nothing here owns the memory the arrays point into. The producer keeps it alive
across the call and releases it afterwards, by which time every byte that was
wanted has been copied.
"""

from firepanda.array.any import AnyArray
from firepanda.array.chunked import ChunkedArray
from firepanda.array.strings import StringArray
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.exec import parallel_for
from firepanda.frame.frame import DataFrame

from .arrow_c import ArrowArray
from .arrow_import import ColumnSink, fill_column, payload_for


@fieldwise_init
struct ArrowDictionary(Copyable, Movable):
    """The categories of one dictionary encoded column, once they are known."""

    var column: Int
    """Which column of the layout these belong to."""

    var ordered: Bool
    """Whether the categories have a meaningful order."""

    var values: StringArray
    """The categories themselves, one entry each rather than one per row."""


@fieldwise_init
struct ArrowNested(Movable):
    """One list or struct column, already built, waiting for its place.

    Everything else here is copied a row range at a time on whatever cores are
    free, and a nested column cannot be. A range of rows of a list is not a range
    of rows of the column its elements are in, and which range it is cannot be
    known without reading the offsets first, so the piecewise plan that makes the
    rest of the read parallel has nothing to say about one. The producer builds
    these whole and hands them over finished, and the assembly drops them into
    place at the end next to the columns it did copy.
    """

    var column: Int
    """Which column of the layout this is."""

    var value: AnyArray
    """The column, complete with its children."""

    def take(deinit self) -> AnyArray:
        """Hands the column over without copying it.

        Returns:
            The column.
        """
        return self.value^


struct ArrowLayout(Copyable, Movable, Sized):
    """What a producer says about its columns, which is all three things a frame
    needs and none of the data."""

    var names: List[String]
    """The column names, in order."""

    var formats: List[String]
    """One C Data Interface format string per column, in the same order.

    A dictionary encoded column puts its index type here and not its value type,
    which is what makes the whole assembly below work on one unchanged: the codes
    are an ordinary int32 column and are copied by the same path as any other,
    and the categories are attached at the end. Arrow's own C interface makes the
    same choice, where the schema's format is the index type and the values hang
    off a separate member.
    """

    var nullable: List[Bool]
    """Whether each column was declared nullable."""

    var dictionaries: List[ArrowDictionary]
    """The categories of whichever columns are dictionary encoded.

    One entry per dictionary column rather than one per column, because almost
    every layout has none and the ones that do have one or two. Looked up by
    scanning, which over a list this short beats a map by more than the code it
    saves.
    """

    def __init__(out self):
        """Constructs a layout of no columns."""
        self.names = List[String]()
        self.formats = List[String]()
        self.nullable = List[Bool]()
        self.dictionaries = List[ArrowDictionary]()

    def __len__(self) -> Int:
        """Returns the column count.

        Returns:
            How many columns the layout has.
        """
        return len(self.names)

    def dictionary_at(self, column: Int) -> Int:
        """Finds the categories belonging to a column.

        Args:
            column: The column position.

        Returns:
            The position in `dictionaries`, or -1 if the column is not
            dictionary encoded.
        """
        for i in range(len(self.dictionaries)):
            if self.dictionaries[i].column == column:
                return i
        return -1


comptime PIECE_ROWS = 65536
"""How many rows one task copies. A multiple of eight, so that a piece begins on
a byte of the validity bitmap and the bits are copied rather than shifted."""


@fieldwise_init
struct _Piece(ImplicitlyCopyable, Movable):
    """One task's share of the copy: a range of rows of one batch."""

    var batch: Int
    """Which batch."""

    var start: Int
    """The first row of the range, within that batch."""

    var rows: Int
    """How many rows the range holds."""

    var at: Int
    """The first row of the range in the finished column."""

    var whole: Bool
    """Whether the range is the whole batch, which decides whether a view column
    may take the producer's data buffer as it stands rather than compacting."""

    var morsel: Int
    """Which chunk of the output the range belongs to, and zero for a frame in
    one chunk. A range never crosses a chunk boundary, so there is one answer."""

    var within: Int
    """The first row of the range within that chunk, which is where the fill
    writes. The same number as `at` for a frame in one chunk."""


def _slice(array: ArrowArray, start: Int, rows: Int) -> ArrowArray:
    """Narrows an array to a range of its rows without touching its buffers.

    The copy carries a null pointer where the original has its release callback,
    because a slice is a view of somebody else's array and releasing it twice is
    the one way to turn this into a crash a long way from here. The null count is
    the whole array's, which overestimates a slice and is only ever compared
    against zero, so a slice of a column with no nulls still gets the cheap path.

    Args:
        array: The array to narrow.
        start: The first row of the range, counted from the array's own first row.
        rows: How many rows the range holds.

    Returns:
        An array over the same buffers, owning nothing.
    """
    var out = array.copy()
    out.offset = array.offset + Int64(start)
    out.length = Int64(rows)
    out.release = None
    out.private_data = None
    return out^


def _check_codes[
    dt: DType
](column: AnyArray, categories: Int, name: String) raises:
    """Raises if any code points outside the categories.

    Nothing downstream can do this check for itself. A code is read to reach a
    category, and by then a bad one is an out of bounds read rather than a
    question anybody asks, so the one pass to rule them out happens here where
    the column is built and once.

    A null row's code is not read and is not checked. Arrow says nothing about
    what a writer puts there and pandas writes a negative number, so a valid
    file would be rejected for a byte that means nothing.

    Args:
        column: The finished codes column.
        categories: How many categories there are.
        name: The column name, for the error message.

    Parameters:
        dt: The index dtype.

    Raises:
        Error: If a code of a non-null row is negative or too large.
    """
    ref codes = column.as_typed_view[dt]()
    for i in range(len(column)):
        if not column.is_valid(i):
            continue
        var code = Int(codes[i])
        if code < 0 or code >= categories:
            raise Error(
                String(
                    "arrow: column '",
                    name,
                    "' has a code of ",
                    code,
                    " at row ",
                    i,
                    " against ",
                    categories,
                    " categories",
                )
            )


def attach_dictionary(
    mut column: AnyArray, entry: ArrowDictionary, name: String
) raises:
    """Turns a finished codes column into a dictionary column.

    Args:
        column: The codes, as an ordinary integer column. Left as a dictionary
            column over the same buffer.
        entry: The categories and the ordered flag.
        name: The column name, for the error message.

    Raises:
        Error: If the index type is not an integer, or a code is out of range.
    """
    var size = len(entry.values)
    var index = column.type.physical
    # Arrow allows any signed integer width for an index and says unsigned ones
    # are permitted too, so all eight are read rather than the int32 that
    # everything in practice writes.
    if index == DType.int8:
        _check_codes[DType.int8](column, size, name)
    elif index == DType.int16:
        _check_codes[DType.int16](column, size, name)
    elif index == DType.int32:
        _check_codes[DType.int32](column, size, name)
    elif index == DType.int64:
        _check_codes[DType.int64](column, size, name)
    elif index == DType.uint8:
        _check_codes[DType.uint8](column, size, name)
    elif index == DType.uint16:
        _check_codes[DType.uint16](column, size, name)
    elif index == DType.uint32:
        _check_codes[DType.uint32](column, size, name)
    elif index == DType.uint64:
        _check_codes[DType.uint64](column, size, name)
    else:
        raise Error(
            String(
                "arrow: column '",
                name,
                "' is dictionary encoded with an index type of ",
                index,
                ", and an index is an integer",
            )
        )
    column.dict_values = StringArray(copy=entry.values)
    column.type = LogicalType.dictionary(index, entry.ordered)


def assemble(
    layout: ArrowLayout,
    batches: List[List[ArrowArray]],
    var nested: List[ArrowNested] = List[ArrowNested](),
    morsel_rows: Int = 0,
) raises -> DataFrame:
    """Copies every array into its place in a frame allocated once.

    Args:
        layout: The names, formats and nullability the producer declared.
        batches: One list of arrays per batch, each holding one array per column
            in the layout's order. Every array in a batch has the same length.
        nested: Whichever columns the producer built itself, because they are
            list or struct columns and do not come apart into row ranges. The
            arrays sitting in `batches` at those positions are not read.
        morsel_rows: The chunk height to come back in, or zero for a frame in one
            chunk. A frame in one chunk is what every eager caller wants, because
            `DataFrame.__getitem__` borrows a column only when it has a single
            chunk. A frame in morsels is what the pipeline wants, because it runs
            its elementwise operators a chunk per worker and a frame in one chunk
            is a query on one core. Nothing about the copy below changes either
            way: the same pieces are filled by the same tasks, into a sink each
            rather than into one long sink.

    Returns:
        The frame, holding its own copy of everything.

    Raises:
        Error: If a column holds a type firepanda cannot read, or an array is
            malformed. The first failure by task index is the one raised.
    """
    var width = len(layout)
    if width == 0:
        return DataFrame(Schema(List[Field]()), List[AnyArray]())

    # The columns that are copied here, which is all of them unless the producer
    # brought a nested one along. Everything below counts in these rather than in
    # columns, so that a task index still maps onto a piece of real work.
    var plain = List[Int](capacity=width)
    for c in range(width):
        var found = False
        for n in range(len(nested)):
            if nested[n].column == c:
                found = True
                break
        if not found:
            plain.append(c)
    var taken = len(plain)

    # A column the producer built whole cannot be cut into morsels, and a frame
    # whose columns disagree about their chunk boundaries is not a frame, so one
    # nested column puts the whole read back in one chunk.
    var height = 0 if len(nested) != 0 else morsel_rows

    var pieces = List[_Piece]()
    var rows = 0
    for b in range(len(batches)):
        var length = Int(batches[b][0].length)
        var start = 0
        while start < length:
            var take = min(PIECE_ROWS, length - start)
            if height > 0:
                # A piece is one task's share of one sink, so it may not cross a
                # morsel boundary. Cutting it short here is what keeps the fill
                # below a plain write at an offset.
                var room = height - rows % height
                if take > room:
                    take = room
            var morsel = 0 if height <= 0 else rows // height
            var within = rows if height <= 0 else rows % height
            pieces.append(
                _Piece(b, start, take, rows, take == length, morsel, within)
            )
            start += take
            rows += take
        if length == 0:
            # A batch of no rows copies nothing, so where it is pointed does not
            # matter, and the first chunk is the one that is always there.
            pieces.append(_Piece(b, 0, 0, rows, True, 0, 0))
    var count = len(pieces)

    var morsels = 1
    if height > 0 and rows > 0:
        morsels = (rows + height - 1) // height

    var payload = List[Int](length=taken * count, fill=0)

    def plan(task: Int) raises {mut payload, imm}:
        var c = plain[task // count]
        ref piece = pieces[task % count]
        payload[task] = payload_for(
            _slice(batches[piece.batch][c], piece.start, piece.rows),
            layout.formats[c],
            piece.whole,
        )

    parallel_for(plan, taken * count)

    # One sink per column per morsel, and the payload each piece planned turns
    # into that piece's offset within its own sink. In the one chunk case there
    # is a sink a column and this is the running total it always was.
    var sinks = List[ColumnSink](capacity=taken * morsels)
    for k in range(taken):
        var totals = List[Int](length=morsels, fill=0)
        for p in range(count):
            var m = pieces[p].morsel
            var here = payload[k * count + p]
            payload[k * count + p] = totals[m]
            totals[m] += here
        for m in range(morsels):
            var tall = rows
            if height > 0:
                tall = min(height, rows - m * height)
            sinks.append(ColumnSink(layout.formats[plain[k]], tall, totals[m]))

    var validity = List[Bitmap](length=taken * count, fill=Bitmap(0))

    def fill(task: Int) raises {mut sinks, mut validity, imm}:
        var k = task // count
        var c = plain[k]
        ref piece = pieces[task % count]
        validity[task] = fill_column(
            sinks[k * morsels + piece.morsel],
            piece.within,
            payload[task],
            _slice(batches[piece.batch][c], piece.start, piece.rows),
            layout.formats[c],
            piece.whole,
        )

    parallel_for(fill, taken * count)

    # Validity is pasted here rather than inside the fill, because a piece
    # boundary need not fall on a byte of the bitmap and two threads writing the
    # byte either side of one would lose each other's bits. One pass over a bit
    # per row does not show up next to the copy it follows.
    for k in range(taken):
        for p in range(count):
            ref piece = pieces[p]
            if batches[piece.batch][plain[k]].null_count == 0:
                continue
            sinks[k * morsels + piece.morsel].validity.paste(
                piece.within, validity[k * count + p], piece.rows
            )

    var fields = List[Field](capacity=width)
    var columns = List[ChunkedArray](capacity=width)
    for c in range(width):
        var brought = -1
        for n in range(len(nested)):
            if nested[n].column == c:
                brought = n
                break
        var at = layout.dictionary_at(c)
        var built: ChunkedArray
        if brought >= 0:
            var column = nested.pop(brought).take()
            if at >= 0:
                attach_dictionary(
                    column, layout.dictionaries[at], layout.names[c]
                )
            built = ChunkedArray(column^)
        else:
            # The sinks of one column are next to each other and in morsel
            # order, so the first one makes the column and the rest follow it.
            var first = sinks.pop(0).finish()
            if at >= 0:
                attach_dictionary(
                    first, layout.dictionaries[at], layout.names[c]
                )
            built = ChunkedArray(first^)
            for _ in range(morsels - 1):
                var chunk = sinks.pop(0).finish()
                if at >= 0:
                    attach_dictionary(
                        chunk, layout.dictionaries[at], layout.names[c]
                    )
                built.append(chunk^)
        var field = Field(layout.names[c], built.type)
        field.nullable = layout.nullable[c]
        fields.append(field^)
        columns.append(built^)
    return DataFrame(Schema(fields^), columns^)
