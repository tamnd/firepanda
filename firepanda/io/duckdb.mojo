"""DuckDB's C API, loaded at run time.

Firepanda reads Parquet by binding a library rather than by decoding pages
itself, which is the decision document 08 makes for M2 and which stands: a native
Parquet reader is a performance project and this is a capability that unblocks
everything now. What changed is which library. The plan named Arrow C++, and
Arrow C++ turns out to export no C symbols at all, so binding it means writing a
C++ shim and shipping a C++ compiler and the Arrow headers to anyone who wants to
read a Parquet file. DuckDB is a single shared library with a stable, documented
C API, it produces Arrow C Data Interface structures directly, and it brings Hive
partitioning, projection and predicate pushdown with it. The full argument is in
docs/specs/08-milestones.md.

It is loaded with `dlopen` rather than linked, and that is the important part.
Nothing in firepanda needs DuckDB to build or to run. A process that never calls
`read_parquet` never opens the library, and a machine without it gets an error
from `read_parquet` naming the package to install and nothing else changes. The
alternative, linking it, would make every firepanda binary carry sixty megabytes
so that some of them can read Parquet.

Every entry point is a function pointer resolved by name once and reinterpreted
the way `arrow_c` reinterprets a release callback. The signatures are transcribed
from duckdb.h and each one is written above its declaration, because a wrong
signature here is not a compile error in any language, it is a wrong stack at run
time.

Every out parameter goes through `Cells`, which is a list, which is the heap.
That is not a style choice. A stack local reached through
`Pointer(to=x).unsafe_origin_cast[MutUntrackedOrigin]()` does not survive the
round trip: erasing the origin also erases the compiler's reason to believe
anything ever wrote there, so C's write lands and the next read of the local
still sees what it held before the call. This showed up as `duckdb_connect`
returning `DuckDBSuccess` with a null connection, which reads like a DuckDB bug
and is not one. Heap memory behind a list is not a value the compiler is
tracking, and a write through a pointer into it is a write.

`duckdb_result` is passed to `duckdb_fetch_chunk` by value, which is why the
struct is declared field for field rather than kept behind a pointer. Six words,
all of them either an unsigned length or a pointer, and the deprecated fields
have to be there because the ones after them are at a fixed distance from the
front.

Reference: https://duckdb.org/docs/stable/clients/c/overview
"""

from std.ffi import c_char, external_call

from .arrow_c import ArrowArray, ArrowSchema, CString

comptime Handle = Pointer[UInt8, MutUntrackedOrigin]
"""One of DuckDB's opaque handles, which are all a pointer to a struct holding a
pointer. Never dereferenced here."""

comptime MaybeHandle = Optional[Handle]
"""A handle that may be null, which is how every one of them starts and how the
ones that carry no error end."""

comptime MaybeText = Optional[CString]
"""A `const char*` that may be null."""

comptime SUCCESS: Int32 = 0
"""`DuckDBSuccess`. Every other value is a failure."""

comptime RTLD_NOW: Int32 = 2
"""Resolve every symbol at load rather than at first call. A missing symbol
should be an error from the load, where we can say which library was wrong."""


struct DuckResult(ImplicitlyCopyable, Movable):
    """The C `duckdb_result`. Six words, this order.

    The first five fields are deprecated in duckdb.h and are still part of the
    layout, so they are here under their own names. Nothing reads them: the
    column count comes from `duckdb_column_count` and the error from
    `duckdb_result_error`, which are the calls that replaced them.

    Every field is a word rather than the type it holds, and three of them hold
    pointers. That is because this struct is passed to `duckdb_fetch_chunk` by
    value, and a struct of six plain words is passed the way C passes it while a
    struct of optionals is passed by reference and crashes inside DuckDB. Since
    nothing here is ever read, six words loses no meaning.
    """

    var deprecated_column_count: UInt64
    """Was the column count."""

    var deprecated_row_count: UInt64
    """Was the row count."""

    var deprecated_rows_changed: UInt64
    """Was the number of rows a statement changed."""

    var deprecated_columns: UInt64
    """Was a pointer to an array of materialized columns."""

    var deprecated_error_message: UInt64
    """Was a pointer to the error message."""

    var internal_data: UInt64
    """The result itself, opaque. Zero once the result has been destroyed."""

    def __init__(out self):
        """Constructs the all zero state, which is what an unused result is."""
        self.deprecated_column_count = 0
        self.deprecated_row_count = 0
        self.deprecated_rows_changed = 0
        self.deprecated_columns = 0
        self.deprecated_error_message = 0
        self.internal_data = 0


comptime ResultPtr = Pointer[DuckResult, MutUntrackedOrigin]
comptime Cell = Pointer[UInt64, MutUntrackedOrigin]
"""A pointer sized hole for C to write a handle into. See `Cells`."""

comptime HandlePtr = Pointer[MaybeHandle, MutUntrackedOrigin]
comptime TextPtr = Pointer[CString, MutUntrackedOrigin]
comptime ArrayPtr = Pointer[ArrowArray, MutUntrackedOrigin]
comptime SchemaPtr = Pointer[ArrowSchema, MutUntrackedOrigin]

comptime FnOpen = def(CString, Cell) thin abi("C") -> Int32
"""`duckdb_state duckdb_open(const char *path, duckdb_database *out)`."""

comptime FnClose = def(Cell) thin abi("C") -> None
"""`void duckdb_close(duckdb_database *database)`."""

comptime FnConnect = def(Handle, Cell) thin abi("C") -> Int32
"""`duckdb_state duckdb_connect(duckdb_database db, duckdb_connection *out)`."""

comptime FnDisconnect = def(Cell) thin abi("C") -> None
"""`void duckdb_disconnect(duckdb_connection *connection)`."""

comptime FnQuery = def(Handle, CString, ResultPtr) thin abi("C") -> Int32
"""`duckdb_state duckdb_query(duckdb_connection, const char *, duckdb_result *)`.
"""

comptime FnDestroyResult = def(ResultPtr) thin abi("C") -> None
"""`void duckdb_destroy_result(duckdb_result *result)`."""

comptime FnResultError = def(ResultPtr) thin abi("C") -> MaybeText
"""`const char *duckdb_result_error(duckdb_result *result)`."""

comptime FnColumnCount = def(ResultPtr) thin abi("C") -> UInt64
"""`idx_t duckdb_column_count(duckdb_result *result)`."""

comptime FnColumnName = def(ResultPtr, UInt64) thin abi("C") -> MaybeText
"""`const char *duckdb_column_name(duckdb_result *result, idx_t col)`."""

comptime FnColumnType = def(ResultPtr, UInt64) thin abi("C") -> MaybeHandle
"""`duckdb_logical_type duckdb_column_logical_type(duckdb_result *, idx_t)`."""

comptime FnDestroyType = def(Cell) thin abi("C") -> None
"""`void duckdb_destroy_logical_type(duckdb_logical_type *type)`."""

comptime FnArrowOptions = def(ResultPtr) thin abi("C") -> MaybeHandle
"""`duckdb_arrow_options duckdb_result_get_arrow_options(duckdb_result *)`."""

comptime FnDestroyArrowOptions = def(Cell) thin abi("C") -> None
"""`void duckdb_destroy_arrow_options(duckdb_arrow_options *options)`."""

comptime FnFetchChunk = def(DuckResult) thin abi("C") -> MaybeHandle
"""`duckdb_data_chunk duckdb_fetch_chunk(duckdb_result result)`, by value."""

comptime FnDestroyChunk = def(Cell) thin abi("C") -> None
"""`void duckdb_destroy_data_chunk(duckdb_data_chunk *chunk)`."""

comptime FnChunkToArrow = def(Handle, Handle, ArrayPtr) thin abi(
    "C"
) -> MaybeHandle
"""`duckdb_error_data duckdb_data_chunk_to_arrow(duckdb_arrow_options,
duckdb_data_chunk, struct ArrowArray *)`."""

comptime FnToArrowSchema = def(
    Handle, HandlePtr, TextPtr, UInt64, SchemaPtr
) thin abi("C") -> MaybeHandle
"""`duckdb_error_data duckdb_to_arrow_schema(duckdb_arrow_options,
duckdb_logical_type *, const char **, idx_t, struct ArrowSchema *)`."""

comptime FnErrorMessage = def(Handle) thin abi("C") -> MaybeText
"""`const char *duckdb_error_data_message(duckdb_error_data error_data)`."""

comptime FnDestroyError = def(Cell) thin abi("C") -> None
"""`void duckdb_destroy_error_data(duckdb_error_data *error_data)`."""

comptime FnVersion = def() thin abi("C") -> MaybeText
"""`const char *duckdb_library_version()`."""


struct Cells(Movable, Sized):
    """Pointer sized holes on the heap, which is where C writes its answers.

    Every DuckDB call that hands something back does it through a pointer to a
    handle, and this is that pointer. One allocation holds all of them for a
    session, so a read does not allocate once per call, and the addresses stay
    put for as long as the session does, which is what lets `__deinit__` hand
    the same address to `duckdb_disconnect` that `duckdb_connect` wrote into.

    Nothing here is optional, because a handle DuckDB has not written yet is
    zero and zero is the only thing null can be.
    """

    var slots: List[UInt64]
    """The holes, all zero until something fills one."""

    def __init__(out self, count: Int):
        """Makes some holes.

        Args:
            count: How many.
        """
        self.slots = List[UInt64](length=count, fill=0)

    def __len__(self) -> Int:
        """Returns how many holes there are."""
        return len(self.slots)

    def at(mut self, index: Int) -> Cell:
        """Points at one hole, for C to write into.

        Args:
            index: Which one.

        Returns:
            Its address.
        """
        return (
            self.slots.unsafe_ptr()
            .unsafe_offset(index)
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )

    def word(self, index: Int) -> UInt64:
        """Reads one hole as a number, which is how null is spotted.

        Args:
            index: Which one.

        Returns:
            What is in it, zero if nothing.
        """
        return self.slots[index]

    def handle(mut self, index: Int) -> Handle:
        """Reads one hole as the handle DuckDB put there.

        The caller checks `word` first if the call it came from is allowed to
        return nothing.

        Args:
            index: Which one.

        Returns:
            The handle.
        """
        return self.at(index).unsafe_bitcast[Handle]()[]

    def put(mut self, index: Int, value: Handle):
        """Writes a handle into a hole, so it can be passed to a destructor.

        Args:
            index: Which one.
            value: The handle.
        """
        self.at(index).unsafe_bitcast[Handle]()[] = value


def text_of(p: CString) -> String:
    """Copies a null terminated C string into memory firepanda owns.

    Args:
        p: The string.

    Returns:
        Its contents.
    """
    var out = String()
    var i = 0
    while True:
        var byte = p.unsafe_offset(i).unsafe_load()
        if byte == 0:
            break
        out += chr(Int(byte))
        i += 1
    return out^


def terminated(text: StringSlice) -> List[UInt8]:
    """Copies a string and puts a zero byte after it.

    Mojo's `String` holds a length and does not promise a zero after the last
    byte, and every one of these strings is about to be handed to a C function
    that looks for one. This was found the hard way: `dlsym` reads past the end
    of a name that has no terminator and looks up a symbol nobody asked for,
    which fails in a way that reads exactly like a missing symbol.

    Args:
        text: The string.

    Returns:
        Its bytes with a zero after them.

    The caller keeps it alive for as long as C is looking at the pointer, and
    that is not automatic. A Mojo value dies at its last use, and taking a
    pointer out of a list is a use, so `f(terminated(s).unsafe_ptr())` frees the
    bytes before `f` runs. Every call site here binds the result to a local and
    writes `_ = local^` after the call. This was found by calling `strlen`
    through the same machinery and getting zero back.
    """
    var out = List[UInt8](capacity=text.byte_length() + 1)
    for byte in text.as_bytes():
        out.append(byte)
    out.append(0)
    return out^


def _as[F: ImplicitlyCopyable](slot: Handle) -> F:
    """Reinterprets a resolved symbol as the function it is.

    Parameters:
        F: The function type, transcribed from duckdb.h.

    Args:
        slot: What `dlsym` returned.

    Returns:
        The function.
    """
    return Pointer(to=slot).unsafe_bitcast[F]()[]


struct Library(Movable):
    """An open libduckdb and every entry point firepanda calls.

    Resolved once and held together, so that a machine with the wrong version of
    the library fails at the open with the name of the symbol that is missing,
    rather than partway through a read with a segmentation fault.

    The handle is deliberately never closed. `dlclose` on a library whose
    allocations are still reachable is a way to turn a working program into a
    crash at exit, and the process is going to end anyway.
    """

    var open: FnOpen
    var close: FnClose
    var connect: FnConnect
    var disconnect: FnDisconnect
    var query: FnQuery
    var destroy_result: FnDestroyResult
    var result_error: FnResultError
    var column_count: FnColumnCount
    var column_name: FnColumnName
    var column_type: FnColumnType
    var destroy_type: FnDestroyType
    var arrow_options: FnArrowOptions
    var destroy_arrow_options: FnDestroyArrowOptions
    var fetch_chunk: FnFetchChunk
    var destroy_chunk: FnDestroyChunk
    var chunk_to_arrow: FnChunkToArrow
    var to_arrow_schema: FnToArrowSchema
    var error_message: FnErrorMessage
    var destroy_error: FnDestroyError
    var version: FnVersion

    def __init__(out self) raises:
        """Opens libduckdb and resolves every symbol.

        Raises:
            Error: If the library cannot be found, or is found and is missing
                something, which is the same sentence to a user either way and is
                said as two different sentences because the fixes differ.
        """
        var handle = _open_library()
        self.open = _as[FnOpen](_symbol(handle, "duckdb_open"))
        self.close = _as[FnClose](_symbol(handle, "duckdb_close"))
        self.connect = _as[FnConnect](_symbol(handle, "duckdb_connect"))
        self.disconnect = _as[FnDisconnect](
            _symbol(handle, "duckdb_disconnect")
        )
        self.query = _as[FnQuery](_symbol(handle, "duckdb_query"))
        self.destroy_result = _as[FnDestroyResult](
            _symbol(handle, "duckdb_destroy_result")
        )
        self.result_error = _as[FnResultError](
            _symbol(handle, "duckdb_result_error")
        )
        self.column_count = _as[FnColumnCount](
            _symbol(handle, "duckdb_column_count")
        )
        self.column_name = _as[FnColumnName](
            _symbol(handle, "duckdb_column_name")
        )
        self.column_type = _as[FnColumnType](
            _symbol(handle, "duckdb_column_logical_type")
        )
        self.destroy_type = _as[FnDestroyType](
            _symbol(handle, "duckdb_destroy_logical_type")
        )
        self.arrow_options = _as[FnArrowOptions](
            _symbol(handle, "duckdb_result_get_arrow_options")
        )
        self.destroy_arrow_options = _as[FnDestroyArrowOptions](
            _symbol(handle, "duckdb_destroy_arrow_options")
        )
        self.fetch_chunk = _as[FnFetchChunk](
            _symbol(handle, "duckdb_fetch_chunk")
        )
        self.destroy_chunk = _as[FnDestroyChunk](
            _symbol(handle, "duckdb_destroy_data_chunk")
        )
        self.chunk_to_arrow = _as[FnChunkToArrow](
            _symbol(handle, "duckdb_data_chunk_to_arrow")
        )
        self.to_arrow_schema = _as[FnToArrowSchema](
            _symbol(handle, "duckdb_to_arrow_schema")
        )
        self.error_message = _as[FnErrorMessage](
            _symbol(handle, "duckdb_error_data_message")
        )
        self.destroy_error = _as[FnDestroyError](
            _symbol(handle, "duckdb_destroy_error_data")
        )
        self.version = _as[FnVersion](_symbol(handle, "duckdb_library_version"))

    def check(self, var error: MaybeHandle) raises:
        """Raises whatever a `duckdb_error_data` is carrying, and frees it.

        DuckDB returns one of these from the Arrow conversions whether or not
        anything went wrong, and a null one means it did not.

        Args:
            error: What the call returned.

        Raises:
            Error: If the handle is not null, carrying DuckDB's own message.
        """
        if not error:
            return
        var handle = error.value()
        var text = self.error_message(handle)
        var message = String("duckdb: ")
        if text:
            message += text_of(text.value())
        else:
            message += "a call failed and said nothing about why"
        var slot = Cells(1)
        slot.put(0, handle)
        self.destroy_error(slot.at(0))
        _ = slot^
        raise Error(message)


def _open_library() raises -> Handle:
    """Finds and opens libduckdb.

    Looks at `FIREPANDA_DUCKDB` first, so that a machine with the library
    somewhere unusual has a way to say where without a rebuild, and then at the
    plain names, which is what finds the one a package manager installed
    alongside this binary.

    Returns:
        The library handle.

    Raises:
        Error: If none of the names opens, with all of them named.
    """
    var names = List[String]()
    var wanted = terminated("FIREPANDA_DUCKDB")
    var override = external_call["getenv", MaybeText](
        wanted.unsafe_ptr().unsafe_bitcast[c_char]()
    )
    _ = wanted^
    if override:
        names.append(text_of(override.value()))
    names.append(String("libduckdb.so"))
    names.append(String("libduckdb.dylib"))

    for name in names:
        var bytes = terminated(name)
        var opened = external_call["dlopen", MaybeHandle](
            bytes.unsafe_ptr(), RTLD_NOW
        )
        _ = bytes^
        if opened:
            return opened.value()

    var tried = String()
    for i in range(len(names)):
        if i != 0:
            tried += ", "
        tried += names[i]
    raise Error(
        String(
            (
                "parquet: firepanda reads Parquet through DuckDB and could not"
                " load it. Install the libduckdb package, or set"
                " FIREPANDA_DUCKDB to the file. Tried "
            ),
            tried,
        )
    )


def _symbol(handle: Handle, name: StringSlice) raises -> Handle:
    """Resolves one symbol.

    Args:
        handle: The open library.
        name: The symbol name.

    Returns:
        Its address.

    Raises:
        Error: If the library does not have it, which means it is a libduckdb
            older than the C Arrow interface this uses.
    """
    var bytes = terminated(name)
    var found = external_call["dlsym", MaybeHandle](handle, bytes.unsafe_ptr())
    _ = bytes^
    if not found:
        raise Error(
            String(
                "parquet: the libduckdb that loaded has no ",
                name,
                (
                    ", which means it is older than 1.4. Upgrade it or set"
                    " FIREPANDA_DUCKDB to a newer one"
                ),
            )
        )
    return found.value()
