"""Reading Parquet, by asking DuckDB and taking the answer as Arrow.

Parquet is not one format. It is a container with several encodings, three
compression codecs anybody uses and two more nobody does, a page index, bloom
filters, and a decade of writers that disagree about the details. Decoding it
well is a performance project, and document 08 puts that at M8. What this
milestone wants is the capability, now, so that a TPC-H file loads and everything
after this can be measured against something real.

So the file is handed to DuckDB, which decodes it into its own vectors, converts
each one to an Arrow array, and hands those back over the C Data Interface that
firepanda already speaks in both directions. The copy out of Arrow and into a
firepanda frame is the same `assemble` the IPC reader uses, so a Parquet file of
a hundred row groups costs what its bytes cost, the same as an IPC file of a
hundred record batches.

The reason it is a SQL string rather than a reader object is that everything else
this milestone needs is already a SQL string. A Hive partitioned directory is
`read_parquet('dir/**/*.parquet', hive_partitioning = true)`. Reading some of the
columns is naming them instead of a star, and it is a real projection pushdown
rather than a filter after the fact, because DuckDB never decodes the pages it
was not asked for. None of that is code here.

What is here, and matters, is that the query is built rather than interpolated.
A path with an apostrophe in it is a path, not a syntax error and not an
injection, so the path is quoted the way SQL quotes a string and the caller never
writes SQL unless it wants to.
"""

from firepanda.exec import parallel_for
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
from .duckvector import Scratch, duck_array, duck_format

comptime DATABASE = 0
"""Which of a session's cells holds the database."""

comptime CONNECTION = 1
"""Which of a session's cells holds the connection."""

comptime RESULT_WORDS = 6
"""How many words a `duckdb_result` is. See `DuckResult`."""


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

    def run(mut self, sql: StringSlice) raises -> DataFrame:
        """Runs one query and returns the whole answer as a frame.

        Every chunk is converted to Arrow and kept, and none of them is copied
        into the frame until all of them have arrived, because the assembler
        wants to know the row count before it allocates. That is a copy of the
        answer held twice for the length of the read, which is the same trade
        every eager reader makes and is what an eager frame is.

        Args:
            sql: The query.

        Returns:
            The result.

        Raises:
            Error: If the query fails, or the answer holds a type firepanda
                cannot read.
        """
        var result = Cells(RESULT_WORDS)
        return self._execute(sql, result)

    def _execute(
        mut self, sql: StringSlice, mut result: Cells
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

        try:
            var frame = self._collect(slot)
            self.lib.destroy_result(slot)
            return frame^
        except error:
            self.lib.destroy_result(slot)
            raise error

    def _collect(self, slot: ResultPtr) raises -> DataFrame:
        """Drains a result and turns it into a frame.

        Args:
            slot: The result, which the caller destroys either way.

        Returns:
            The frame.

        Raises:
            Error: If a conversion fails or a type cannot be read.
        """
        var width = Int(self.lib.column_count(slot))
        var kinds = self._kinds(slot, width)
        var direct = self._direct_layout(slot, width, kinds)
        if len(direct) == width and width != 0:
            return self._collect_direct(slot, width, kinds, direct)

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

        var arrays = List[ArrowArray]()
        var batches = List[List[ArrowArray]]()
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
                batches.append(_columns_of(box[0]))
                arrays.append(box.pop())
        except error:
            self._drop(arrays^)
            self._drop_options(settings)
            raise error

        var frame: DataFrame
        try:
            frame = assemble(layout, batches)
        except error:
            self._drop(arrays^)
            self._drop_options(settings)
            raise error

        self._drop(arrays^)
        self._drop_options(settings)
        return frame^

    def _kinds(self, slot: ResultPtr, width: Int) raises -> List[Int32]:
        """Reads every column's DuckDB type id.

        The logical type is a handle that has to be destroyed, and the type id
        is a small integer, so this takes the integers once and lets the handles
        go rather than carrying them around a read.

        Args:
            slot: The result.
            width: How many columns it has.

        Returns:
            One `duckdb_type` per column.

        Raises:
            Error: If a column has no type, which would be a DuckDB bug.
        """
        var out = List[Int32](capacity=width)
        for i in range(width):
            var type = self.lib.column_type(slot, UInt64(i))
            if not type:
                raise Error(String("duckdb: column ", i, " has no type"))
            out.append(self.lib.type_id(type.value()))
            var held = Cells(1)
            held.put(0, type.value())
            self.lib.destroy_type(held.at(0))
            _ = held^
        return out^

    def _direct_layout(
        self, slot: ResultPtr, width: Int, kinds: List[Int32]
    ) raises -> ArrowLayout:
        """Builds the layout for reading the vectors directly, if we can.

        Returns a layout of fewer columns than the result has when any one of
        them is a type `duckvector` does not read, which is the caller's signal
        to go through `duckdb_data_chunk_to_arrow` instead. It is all or nothing
        because the assembler takes one layout, and a result read half by one
        route and half by another is two ways to be wrong instead of one.

        Args:
            slot: The result.
            width: How many columns it has.
            kinds: The type ids from `_kinds`.

        Returns:
            The layout, or a short one meaning no.

        Raises:
            Error: If a column has no name.
        """
        var out = ArrowLayout()
        for i in range(width):
            var format = duck_format(kinds[i])
            if format == "":
                return out^
            var name = self.lib.column_name(slot, UInt64(i))
            if not name:
                raise Error(String("duckdb: column ", i, " has no name"))
            out.names.append(String(text_of(name.value())))
            out.formats.append(format^)
            # Parquet says whether a column is nullable and DuckDB does not pass
            # that through a result, so this claims every column may hold a
            # null. Claiming more than is true costs a validity bitmap on a
            # column that has no nulls; claiming less would lose one.
            out.nullable.append(True)
        return out^

    def _collect_direct(
        self,
        slot: ResultPtr,
        width: Int,
        kinds: List[Int32],
        layout: ArrowLayout,
    ) raises -> DataFrame:
        """Drains a result by describing DuckDB's own vectors as Arrow arrays.

        Every chunk is pulled before any of it is read, because the assembler
        wants to know the finished row count before it allocates, and because a
        chunk of a materialized result is a handle rather than a copy so holding
        all of them costs what the result already cost.

        Args:
            slot: The result, which the caller destroys either way.
            width: How many columns it has.
            kinds: The type ids from `_kinds`.
            layout: The layout from `_direct_layout`.

        Returns:
            The frame.

        Raises:
            Error: If DuckDB will not hand over a vector, or the assembly fails.
        """
        var chunks = List[Handle]()
        while True:
            var chunk = self.lib.fetch_chunk(slot[])
            if not chunk:
                break
            chunks.append(chunk.value())
        var count = len(chunks)

        var scratch = Scratch(width * count)
        var batches = List[List[ArrowArray]](capacity=count)
        for _ in range(count):
            batches.append(List[ArrowArray](length=width, fill=ArrowArray()))

        def describe(task: Int) raises {mut scratch, mut batches, imm}:
            var c = task // count
            var b = task % count
            batches[b][c] = duck_array(
                self.lib, chunks[b], c, kinds[c], scratch, task
            )

        var frame: DataFrame
        try:
            if count != 0:
                parallel_for(describe, width * count)
            frame = assemble(layout, batches)
        except error:
            _ = scratch^
            self._drop_chunks(chunks)
            raise error
        _ = scratch^
        self._drop_chunks(chunks)
        return frame^

    def _drop_chunks(self, chunks: List[Handle]):
        """Destroys every data chunk a direct read pulled.

        Args:
            chunks: The chunks.
        """
        for i in range(len(chunks)):
            var held = Cells(1)
            held.put(0, chunks[i])
            self.lib.destroy_chunk(held.at(0))
            _ = held^

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
    var wanted = String()
    for i in range(len(columns)):
        if i != 0:
            wanted += ", "
        wanted += _quoted_name(columns[i])
    var session = Session()
    return session.run(
        String("SELECT ", wanted, " FROM read_parquet(", quote(path), ")")
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
