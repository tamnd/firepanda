"""Reading Parquet, by asking DuckDB and taking the answer as Arrow.

Parquet is not one format. It is a container with several encodings, three
compression codecs anybody uses and two more nobody does, a page index, bloom
filters, and a decade of writers that disagree about the details. Decoding it
well is a performance project, and document 08 puts that at M8. What this
milestone wants is the capability, now, so that a TPC-H file loads and everything
after this can be measured against something real.

So the file is handed to DuckDB, which decodes it into its own vectors and hands
them over as Arrow through `duckdb_data_chunk_to_arrow`. The copy into a
firepanda frame is the same `assemble` the IPC reader uses, so a Parquet file of
a hundred row groups costs what its bytes cost, the same as an IPC file of a
hundred record batches.

There used to be a second route beside that one. `duckvector.mojo` described
DuckDB's vectors as Arrow arrays directly, without asking DuckDB to convert them,
on the grounds that for most types they already are and the conversion was most
of what a read cost. That was true when it was written and is not true now.
Measured on DuckDB 1.5.5 against sf1 lineitem, six million rows and sixteen
columns, the direct route peaked at 4639 MB and took 1200 ms where the conversion
peaked at 3611 MB and took 1010 ms, and it lost on a projection of eleven numeric
and date columns with no strings in it at all, which is the case it should have
won most easily. Across TPC-H the relationship was monotonic: the more of the
suite that took the direct route the worse both numbers got. So the route is
gone, and the reason it is written down here is that the idea is a good one and
somebody will have it again.

The reason it is a SQL string rather than a reader object is that everything else
this milestone needs is already a SQL string. A Hive partitioned directory is a
path and a setting. Reading some of the columns is naming them instead of a star,
and it is a real projection pushdown rather than a filter after the fact, because
DuckDB never decodes the pages it was not asked for. None of that is code here,
and `ParquetOptions` at the bottom is four booleans and a list of names rather
than a reader with a life of its own.

What is here, and matters, is that the query is built rather than interpolated.
A path with an apostrophe in it is a path, not a syntax error and not an
injection, so the path is quoted the way SQL quotes a string and the caller never
writes SQL unless it wants to.
"""

from firepanda.array.chunked import ChunkedArray
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame

from .arrow_c import (
    ARROW_FLAG_NULLABLE,
    ArrowArray,
    ArrowSchema,
    CString,
    release_array,
    release_schema,
)
from .assemble import ArrowLayout, assemble
from .duckdb import (
    Cells,
    DuckResult,
    Handle,
    Library,
    MaybeHandle,
    ResultPtr,
    SUCCESS,
    terminated,
    text_of,
)

comptime DATABASE = 0
"""Which of a session's cells holds the database."""

comptime CONNECTION = 1
"""Which of a session's cells holds the connection."""

comptime RESULT_WORDS = 6
"""How many words a `duckdb_result` is. See `DuckResult`."""

comptime COLLECT_GROUP_ROWS = 1 << 19
"""Rows of Arrow chunks to hold before assembling them and letting them go.

A read used to fetch every chunk out of DuckDB, describe all of them as Arrow,
and only then copy the lot into a frame, because `assemble` sizes the sink it
allocates from every batch it is handed and so cannot begin until it has them
all. The Arrow copy of the whole answer and the frame it turns into were
therefore both resident, and the peak was the sum of the two.

Assembling a group at a time and dropping that group's Arrow arrays before
fetching the next holds one group instead of all of them. Measured on a 13900K
reading sf1 lineitem, six million rows and sixteen columns, the read peaks at
3498 MB with no grouping, 2896 MB at two million rows a group, 2625 MB at half
a million, and 2556 MB at a hundred and twenty eight thousand.

Half a million is where it sits because the bottom of that ladder is not free.
The read itself is 1040 to 1090 ms at half a million against 1140 to 1440 with
no grouping, since a group that fits in cache is assembled out of cache, and it
goes back up to 1220 ms at a hundred and twenty eight thousand where the fixed
cost of an assemble is being paid forty six times. What is left under the curve
at that end is not the Arrow copy any more, it is DuckDB's own materialised
result, which is about 1400 MB here and is not ours to release a piece at a
time.

The frame still comes back in one chunk, which `_combined` does afterwards and
which costs about 410 MB of peak, so the whole change is 3498 MB to 3037 MB
end to end with the read a little faster than it was. Dropping the stacking
step would be worth the other 410 MB and is not done here, because
`DataFrame.__getitem__` raises on a column with more than one chunk and the
whole library is downstream of that.
"""


def quote(text: StringSlice) -> String:
    """Wraps a string in single quotes the way SQL does, doubling any inside.

    Args:
        text: The string.

    Returns:
        A SQL string literal holding exactly those characters.
    """
    var out = String("'")
    for byte in text.as_bytes():
        if byte == UInt8(ord("'")):
            out += "'"
        out += chr(Int(byte))
    out += "'"
    return out^


def _layout_of(schema: ArrowSchema) raises -> ArrowLayout:
    """Reads a struct schema's children into the three lists a frame needs.

    Args:
        schema: The schema DuckDB produced, whose children are the columns.

    Returns:
        The names, format strings and nullability, in column order.

    Raises:
        Error: If a child is missing its name or its format string, which makes
            it not a schema.
    """
    var out = ArrowLayout()
    if not schema.children:
        return out^
    var children = schema.children.value()
    for i in range(Int(schema.n_children)):
        var slot = children.unsafe_offset(i)[]
        ref child = slot[]
        if not child.format:
            raise Error(
                String("parquet: column ", i, " came back with no type")
            )
        out.formats.append(text_of(child.format.value()))
        if child.name:
            out.names.append(text_of(child.name.value()))
        else:
            out.names.append(String("column_", i))
        out.nullable.append((child.flags & ARROW_FLAG_NULLABLE) != 0)
    return out^


def _array_box() -> List[ArrowArray]:
    """Makes one empty `ArrowArray` on the heap for DuckDB to fill in.

    On the heap for the reason `Cells` is on the heap, which duckdb.mojo says at
    length: a struct C fills in through a pointer has to live somewhere the
    compiler is not tracking as a value, or the fields read back as they were
    before the call.

    Returns:
        A list of one array, zeroed.
    """
    var box = List[ArrowArray](capacity=1)
    box.append(ArrowArray())
    return box^


def _schema_box() -> List[ArrowSchema]:
    """Makes one empty `ArrowSchema` on the heap, for the same reason.

    Returns:
        A list of one schema, zeroed.
    """
    var box = List[ArrowSchema](capacity=1)
    box.append(ArrowSchema())
    return box^


def _columns_of(array: ArrowArray) raises -> List[ArrowArray]:
    """Takes the children of a struct array as arrays in their own right.

    The copies carry a null release callback, because the parent owns them and
    releasing a child on its own is a double free waiting for the parent.

    Args:
        array: The struct array one chunk converted to.

    Returns:
        One array per column, in order.

    Raises:
        Error: If the array says it has children and does not.
    """
    var out = List[ArrowArray](capacity=Int(array.n_children))
    if array.n_children == 0:
        return out^
    if not array.children:
        raise Error(
            "parquet: a chunk says it has columns and carries no pointer to"
            " them"
        )
    var children = array.children.value()
    for i in range(Int(array.n_children)):
        var column = children.unsafe_offset(i)[][].copy()
        column.release = None
        column.private_data = None
        out.append(column^)
    return out^


def _stitch(
    mut schema: Schema, mut built: List[ChunkedArray], var group: DataFrame
) raises:
    """Adds one group's frame onto the end of the frame being read.

    The first group is the answer so far and every later one hands over its
    chunks, which is a pointer each rather than a copy, so a read of ten groups
    costs the same bytes as a read of one and differs only in how many chunks a
    column ends up holding.

    Args:
        schema: The names and types, taken from the first group and checked
            against the rest.
        built: The columns so far, one entry a column, extended in place.
        group: The frame the assembler just built, consumed here.

    Raises:
        Error: If a later group has a different width from the first, which
            would mean DuckDB changed the shape of a result mid stream.
    """
    if len(built) == 0:
        schema = group.schema.copy()
        built = group^.into_columns()
        return

    var columns = group^.into_columns()
    if len(columns) != len(built):
        raise Error(
            String(
                "parquet: a result began with ",
                len(built),
                " columns and went on with ",
                len(columns),
            )
        )
    for c in range(len(built)):
        var chunks = columns.pop(0).into_chunks()
        while len(chunks) != 0:
            built[c].append(chunks.pop(0))


def _combined(var frame: DataFrame) raises -> DataFrame:
    """Stacks each column's groups back into the one chunk callers expect.

    A column at a time rather than all of them at once, so that what is held
    twice is the column being stacked and not the frame. A read of one group,
    which is every read small enough not to care, moves its chunk and copies
    nothing.

    This is where the grouped read gives some of its saving back. It is a
    hundred milliseconds on sf1 lineitem, against the two hundred and fifty the
    grouping took off the read itself, and 410 MB of peak, against the 870 the
    grouping took off that. What it buys is that nothing downstream has to know
    the read happened in pieces: `DataFrame.__getitem__` borrows a column only
    when it has exactly one chunk and raises otherwise, so a frame in groups is
    not a frame most of this library can use.

    Args:
        frame: The frame the collector built, consumed here.

    Returns:
        The same rows, one chunk a column.

    Raises:
        Error: If the chunks of a column cannot be stacked.
    """
    var schema = frame.schema.copy()
    var pieces = frame^.into_columns()
    var out = List[ChunkedArray](capacity=len(pieces))
    while len(pieces) != 0:
        out.append(ChunkedArray(pieces.pop(0).combine()))
    return DataFrame(schema^, out^)


struct Session(Movable):
    """An in memory DuckDB, open for as long as one read takes.

    There is no database and no state worth keeping between reads, so this holds
    nothing across calls and closing it is unconditional. Opening one costs about
    a millisecond, which is not a number that matters next to reading a file, and
    a process wide connection would be a shared mutable thing with no lock around
    it.
    """

    var lib: Library
    """The loaded library and its entry points."""

    var cells: Cells
    """The database in cell zero and the connection in cell one, both written by
    DuckDB and both read back by address rather than by name."""

    def __init__(out self) raises:
        """Loads DuckDB, opens a database in memory and connects to it.

        Raises:
            Error: If the library cannot be loaded, or refuses to open.
        """
        self.lib = Library()
        self.cells = Cells(2)

        var path = terminated(":memory:")
        var state = self.lib.open(
            path.unsafe_ptr()
            .unsafe_bitcast[Int8]()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            self.cells.at(DATABASE),
        )
        _ = path^
        if state != SUCCESS or self.cells.word(DATABASE) == 0:
            raise Error("parquet: duckdb would not open a database in memory")

        state = self.lib.connect(
            self.cells.handle(DATABASE), self.cells.at(CONNECTION)
        )
        if state != SUCCESS or self.cells.word(CONNECTION) == 0:
            raise Error("parquet: duckdb would not open a connection")

    def __deinit__(deinit self):
        """Closes the connection and the database, in that order.

        The cells are taken out of the session first and let go of last. Left in
        place their last use would be the address handed to `duckdb_close`, and
        they would be freed before the call that reads them, which glibc reports
        as an invalid free from inside DuckDB.
        """
        var cells = self.cells^
        if cells.word(CONNECTION) != 0:
            self.lib.disconnect(cells.at(CONNECTION))
        if cells.word(DATABASE) != 0:
            self.lib.close(cells.at(DATABASE))
        _ = cells^

    def run(
        mut self,
        sql: StringSlice,
        morsel_rows: Int = 0,
        group_rows: Int = COLLECT_GROUP_ROWS,
    ) raises -> DataFrame:
        """Runs one query and returns the whole answer as a frame.

        The assembler wants to know the row count before it allocates, so a
        group of chunks is converted to Arrow and kept until the group is full
        and then copied into the frame in one go. Those Arrow arrays are let go
        of before the next group is fetched, so what is held twice is a group
        and not the answer.

        Args:
            sql: The query.
            morsel_rows: The chunk height to come back in, or zero for a frame in
                one chunk, which is what every eager caller wants.
            group_rows: How many rows to hold before assembling them. See
                `COLLECT_GROUP_ROWS`, which is the default and is the only value
                anything but a test passes.

        Returns:
            The result.

        Raises:
            Error: If the query fails, or the answer holds a type firepanda
                cannot read.
        """
        var result = Cells(RESULT_WORDS)
        return self._execute(sql, result, morsel_rows, group_rows)

    def _execute(
        mut self,
        sql: StringSlice,
        mut result: Cells,
        morsel_rows: Int = 0,
        group_rows: Int = COLLECT_GROUP_ROWS,
    ) raises -> DataFrame:
        """Runs one query into a result the caller owns.

        The result cells belong to the caller rather than to this method because
        a value dies at its last use, and if they were made here their last use
        would be the line that takes their address. DuckDB would then be writing
        into freed memory for the length of the read. Taking them by reference
        keeps them alive for exactly as long as the call.

        The query is the materialising one and not the streaming one, which is
        the opposite of what it looks like it should be. `duckdb_query` builds
        the whole answer inside DuckDB and then hands it over a chunk at a time,
        so the rows are written once by the scan and read once by us, and it is
        tempting to skip the first of those with
        `duckdb_execute_prepared_streaming`. Measured on a sixteen core machine
        over ten million rows of Parquet, the streaming read is half again as
        slow, four hundred milliseconds against two hundred and seventy. A
        streaming result is produced by one thread pulling on the pipeline,
        while a materialised one is produced by every thread DuckDB has. The
        copy out is cheaper than the parallelism it costs.

        Args:
            sql: The query.
            result: Six words for DuckDB's `duckdb_result`.
            morsel_rows: The chunk height, passed through to the assembler.
            group_rows: How many rows to hold before assembling them, passed
                through to the collector.

        Returns:
            The answer.

        Raises:
            Error: If the query fails, or the answer holds a type firepanda
                cannot read.
        """
        var text = terminated(sql)
        var slot = result.at(0).unsafe_bitcast[DuckResult]()
        var state = self.lib.query(
            self.cells.handle(CONNECTION),
            text.unsafe_ptr()
            .unsafe_bitcast[Int8]()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            slot,
        )
        _ = text^
        if state != SUCCESS:
            var message = String("duckdb: the query failed")
            var reason = self.lib.result_error(slot)
            if reason:
                message = String("duckdb: ", text_of(reason.value()))
            self.lib.destroy_result(slot)
            raise Error(message)

        var frame: DataFrame
        try:
            frame = self._collect(slot, morsel_rows, group_rows)
        except error:
            self.lib.destroy_result(slot)
            raise error
        self.lib.destroy_result(slot)

        # After the result is destroyed rather than before, so that DuckDB's
        # copy of the answer is gone by the time a second copy of a column
        # exists. A caller that asked for morsels asked for chunks and keeps
        # them.
        if morsel_rows == 0:
            return _combined(frame^)
        return frame^

    def _collect(
        self,
        slot: ResultPtr,
        morsel_rows: Int = 0,
        group_rows: Int = COLLECT_GROUP_ROWS,
    ) raises -> DataFrame:
        """Drains a result and turns it into a frame.

        Args:
            slot: The result, which the caller destroys either way.
            morsel_rows: The chunk height, passed through to the assembler.
            group_rows: How many rows of Arrow chunks to hold before assembling
                them and letting them go. See `COLLECT_GROUP_ROWS`.

        Returns:
            The frame.

        Raises:
            Error: If a conversion fails or a type cannot be read.
        """
        var width = Int(self.lib.column_count(slot))
        var options = self.lib.arrow_options(slot)
        if not options:
            raise Error("duckdb: the result carries no arrow options")
        var settings = options.value()

        var layout: ArrowLayout
        try:
            layout = self._layout(slot, settings, width)
        except error:
            self._drop_options(settings)
            raise error

        # A group boundary is a chunk boundary, so a caller that asked for
        # morsels has to get groups that are a whole number of them or the
        # frame comes back in morsels with a short one every group. Rounding up
        # rather than down, because a morsel taller than the group is a
        # perfectly reasonable thing to ask for and the answer to it is one
        # morsel a group.
        var wanted = group_rows
        if morsel_rows > 0 and wanted > 0:
            wanted = ((wanted + morsel_rows - 1) // morsel_rows) * morsel_rows

        var arrays = List[ArrowArray]()
        var batches = List[List[ArrowArray]]()
        var schema = Schema(List[Field]())
        var built = List[ChunkedArray]()
        var held_rows = 0
        try:
            while True:
                var chunk = self.lib.fetch_chunk(slot[])
                if not chunk:
                    break
                var box = _array_box()
                var failure = self.lib.chunk_to_arrow(
                    settings,
                    chunk.value(),
                    box.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                )
                var held = Cells(1)
                held.put(0, chunk.value())
                self.lib.destroy_chunk(held.at(0))
                _ = held^
                self.lib.check(failure^)
                held_rows += Int(box[0].length)
                batches.append(_columns_of(box[0]))
                arrays.append(box.pop())
                if width != 0 and wanted > 0 and held_rows >= wanted:
                    _stitch(
                        schema,
                        built,
                        assemble(layout, batches, morsel_rows=morsel_rows),
                    )
                    batches.clear()
                    self._drop(arrays^)
                    arrays = List[ArrowArray]()
                    held_rows = 0

            # The last group, and also the whole of a result that never reached
            # the line. An empty result goes through here as well, so that a
            # query matching no rows still comes back with its columns.
            if len(batches) != 0 or len(built) == 0:
                _stitch(
                    schema,
                    built,
                    assemble(layout, batches, morsel_rows=morsel_rows),
                )
        except error:
            self._drop(arrays^)
            self._drop_options(settings)
            raise error

        self._drop(arrays^)
        self._drop_options(settings)
        return DataFrame(schema^, built^)

    def _layout(
        self, slot: ResultPtr, settings: Handle, width: Int
    ) raises -> ArrowLayout:
        """Asks DuckDB what the result's columns are, in Arrow's own words.

        Args:
            slot: The result.
            settings: The arrow options the result was made with.
            width: How many columns it has.

        Returns:
            The layout.

        Raises:
            Error: If DuckDB cannot describe a column as Arrow.
        """
        var types = List[MaybeHandle](capacity=width)
        var names = List[CString](capacity=width)
        for i in range(width):
            var type = self.lib.column_type(slot, UInt64(i))
            if not type:
                raise Error(String("duckdb: column ", i, " has no type"))
            types.append(type)
            var name = self.lib.column_name(slot, UInt64(i))
            if not name:
                raise Error(String("duckdb: column ", i, " has no name"))
            names.append(name.value())

        var box = _schema_box()
        var failure = self.lib.to_arrow_schema(
            settings,
            types.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            names.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            UInt64(width),
            box.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        )
        _ = names^
        for i in range(len(types)):
            self.lib.destroy_type(
                types.unsafe_ptr()
                .unsafe_offset(i)
                .unsafe_bitcast[UInt64]()
                .unsafe_origin_cast[MutUntrackedOrigin]()
            )
        try:
            self.lib.check(failure^)
        except error:
            raise error

        var out: ArrowLayout
        try:
            out = _layout_of(box[0])
        except error:
            release_schema(box[0])
            raise error
        release_schema(box[0])
        return out^

    def _drop(self, var arrays: List[ArrowArray]):
        """Releases every struct array a read produced.

        Args:
            arrays: The arrays, which this consumes.
        """
        for i in range(len(arrays)):
            release_array(arrays[i])

    def _drop_options(self, settings: Handle):
        """Frees a result's arrow options.

        Args:
            settings: The options.
        """
        var held = Cells(1)
        held.put(0, settings)
        self.lib.destroy_arrow_options(held.at(0))
        _ = held^


@fieldwise_init
struct ParquetOptions(Copyable, Movable):
    """What to read, and what to read it as, beyond the path.

    Everything here is a decision about a set of files rather than about one
    file, which is why none of it is an argument to the one path form. A
    directory written by Spark or by pyarrow is a tree of directories named
    `key=value` holding files that do not contain those columns, and reading it
    correctly means reading the paths as data.
    """

    var columns: List[String]
    """The columns to read, in the order they should come back. Empty means all
    of them, which is not the same as none of them."""

    var hive_partitioning: Bool
    """Whether the directories on the way to each file are data. On, which is the
    default, a file at `sales/year=2024/month=03/part-0.parquet` contributes a
    `year` and a `month` column that are nowhere in the file, filled with 2024
    and 3 for every row that came out of it, and a query naming neither one never
    opens the directories that cannot match. The partition columns come back
    after the file's own columns, sorted by name rather than in the order the
    path visits them, which is DuckDB's choice and not ours. This defaults to on
    because DuckDB detects a `key=value` directory by itself, so off would be the
    surprising one: it is here to turn the detection off for a tree that has
    equals signs in its directory names and does not mean anything by them."""

    var union_by_name: Bool
    """Whether files with different schemas line up by column name rather than by
    position. Off, which is DuckDB's default too, every file must have the same
    columns in the same order and a mismatch is an error. On, a column missing
    from one file is null for that file's rows, which is what a directory that
    grew a column halfway through its life needs."""

    var filename: Bool
    """Whether to add a `filename` column holding the path each row came from.
    Useful for a dataset assembled out of files that are not interchangeable, and
    for finding the one file in ten thousand that has the bad row in it."""

    var decimals_as_double: Bool
    """Whether a decimal column is read as a float64 rather than refused.

    firepanda has no decimal column, so a file with one in it cannot be read
    without a conversion, and the conversion loses the exactness that is the
    entire reason somebody wrote a decimal. `DECIMAL(15,2)` holds an unscaled
    integer below 2^53, so the integer survives, and then it is divided by a
    hundred and a hundred is not a power of two. Money that was exact stops being
    exact, and a sum over six million rows of it is off by a number that depends
    on the order they were added in.

    So this is off, and a file with a decimal column is refused by name with a
    message that says this flag is here. It is the same decision the date64
    refusal in `arrow_c.type_for_format` makes, for the same reason: a
    conversion nobody asked for is worse than an error, because an error gets
    read.

    On, the cast happens in DuckDB before the bytes ever become Arrow, which is
    also what it would have cost to do it here and means the doubles arrive as
    doubles. Only a column whose own type is a decimal is cast. A decimal inside
    a list or a struct is left alone and is still refused, because the cast that
    would reach it has to name the shape it is in.
    """

    def __init__(out self):
        """The defaults: every column, partitioning detected, no unioning, no
        filename, and a decimal column refused rather than approximated."""
        self.columns = List[String]()
        self.hive_partitioning = True
        self.union_by_name = False
        self.filename = False
        self.decimals_as_double = False


def _scan_of(path: StringSlice, options: ParquetOptions) -> String:
    """Builds the `read_parquet(...)` call for a path and a set of options.

    Only the settings that are on are named, so the query stays the short thing
    it was for the ordinary case and the long one is what asked for it. The
    exception is `hive_partitioning`, which is written either way because leaving
    it out is not the same as writing false: DuckDB detects a partitioned
    directory on its own, so saying nothing means yes.

    Args:
        path: The file, glob or directory.
        options: What to read it as.

    Returns:
        The table function call, ready to go after a `FROM`.
    """
    var out = String("read_parquet(", quote(path))
    if options.hive_partitioning:
        out += ", hive_partitioning = true"
    else:
        out += ", hive_partitioning = false"
    if options.union_by_name:
        out += ", union_by_name = true"
    if options.filename:
        out += ", filename = true"
    out += ")"
    return out^


def _projection_of(columns: List[String]) raises -> String:
    """Builds the select list for a set of column names.

    Args:
        columns: The names, in the order they should come back. Empty is a star.

    Returns:
        The select list.

    Raises:
        Error: Never. The empty case is a star rather than an error here, because
            the caller that means it separates the two.
    """
    if len(columns) == 0:
        return String("*")
    var out = String()
    for i in range(len(columns)):
        if i != 0:
            out += ", "
        out += _quoted_name(columns[i])
    return out^


def read_parquet(
    path: StringSlice, options: ParquetOptions
) raises -> DataFrame:
    """Reads a Parquet file, a glob, or a partitioned directory of them.

    This is the form that reads a dataset rather than a file.
    `read_parquet("sales/**/*.parquet", options)` over a tree of
    `year=2024/month=03` directories comes back with `year` and `month` columns
    that are in no file, and a query that names neither of them never opens the
    directories that cannot match.

    It is also the form that reads a file with money in it, since a decimal
    column is refused unless `decimals_as_double` is on and the one path form has
    nowhere to say so.

    Args:
        path: The file, glob or directory.
        options: What to read, and what to read it as.

    Returns:
        The rows, with any partition columns on the end.

    Raises:
        Error: If DuckDB is not installed, the files cannot be read, a named
            column is not in them, or a column holds a type firepanda cannot
            read.
    """
    var session = Session()
    var scan = _scan_of(path, options)
    var projection: String
    if options.decimals_as_double:
        projection = _cast_projection(session, scan, options.columns)
    else:
        projection = _projection_of(options.columns)
    return session.run(String("SELECT ", projection, " FROM ", scan))


def _cast_projection(
    mut session: Session, scan: StringSlice, columns: List[String]
) raises -> String:
    """Builds the select list that casts every decimal column to a double.

    A star cannot say this, because there is no way to write "cast the decimals"
    in SQL without knowing which columns are decimals, so the types are asked for
    first. `DESCRIBE` answers from the file's footer and reads no pages, so this
    is a round trip to DuckDB and not a second scan.

    The match is on the type text beginning with `DECIMAL(`, which is how DuckDB
    prints the type and is exact for a column that is a decimal. A list of them
    prints as `DECIMAL(15,2)[]` and does not match, which is deliberate: casting
    it needs `DOUBLE[]` rather than `DOUBLE` and a struct needs the whole struct
    type written out, so those stay refused rather than being half handled.

    Args:
        session: The connection, used for the describe and then for the read.
        scan: The `read_parquet(...)` call, already built.
        columns: The columns the caller asked for, empty for all of them.

    Returns:
        The select list.

    Raises:
        Error: If the files cannot be read or a named column is not in them,
            which is raised here now rather than by the read below.
    """
    var described = session.run(
        String("DESCRIBE SELECT ", _projection_of(columns), " FROM ", scan)
    )
    var out = String()
    for i in range(described.rows):
        if i != 0:
            out += ", "
        var name = _quoted_name(described[0].strings()[i])
        if described[1].strings()[i].startswith("DECIMAL("):
            out += String("CAST(", name, " AS DOUBLE) AS ", name)
        else:
            out += name
    return out^


def read_parquet(path: StringSlice) raises -> DataFrame:
    """Reads a Parquet file, or a glob of them, into a frame.

    Args:
        path: The file. Anything DuckDB's `read_parquet` accepts works here,
            including a glob over a directory of files with the same schema.

    Returns:
        Every row, in file order.

    Raises:
        Error: If DuckDB is not installed, the file cannot be read, or it holds
            a type firepanda has no column for.
    """
    var session = Session()
    return session.run(String("SELECT * FROM read_parquet(", quote(path), ")"))


def read_parquet(path: StringSlice, columns: List[String]) raises -> DataFrame:
    """Reads some of the columns of a Parquet file.

    This is a projection pushdown rather than a read and a drop. DuckDB never
    decodes the column chunks it was not asked for, which on a wide file is the
    single largest thing that can be done for a read and is why the argument is
    here rather than left to a `select` afterwards.

    Args:
        path: The file, as in the one argument form.
        columns: The columns to read, in the order they should come back.

    Returns:
        Those columns, every row.

    Raises:
        Error: If DuckDB is not installed, the file cannot be read, a name is
            not in it, or a column holds a type firepanda cannot read.
    """
    if len(columns) == 0:
        raise Error("parquet: asked for no columns, which is no frame")
    var session = Session()
    return session.run(
        String(
            "SELECT ",
            _projection_of(columns),
            " FROM read_parquet(",
            quote(path),
            ")",
        )
    )


def _quoted_name(name: StringSlice) -> String:
    """Wraps a column name in double quotes, doubling any inside.

    A column named `order` or `group` is a column, not a keyword, and a Parquet
    file written by somebody else is full of names nobody chose for SQL.

    Args:
        name: The column name.

    Returns:
        A quoted identifier.
    """
    var out = String('"')
    for byte in name.as_bytes():
        if byte == UInt8(ord('"')):
            out += '"'
        out += chr(Int(byte))
    out += '"'
    return out^
