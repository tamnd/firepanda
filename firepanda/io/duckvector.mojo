"""Reading DuckDB's data chunks without asking DuckDB to convert them first.

`duckdb_data_chunk_to_arrow` works and it is most of what a Parquet read costs.
Ten million rows of int64, float64 and a short string measured 83 ms of scan and
between 560 and 1100 ms of conversion, and the conversion is buying almost
nothing, because a DuckDB data chunk is already columnar. A flat vector of
BIGINT is a contiguous array of int64 with a validity bitmap beside it in the
same bit order Arrow uses, which is an Arrow array in everything but the struct
that describes it. So this file writes that struct instead.

For every fixed width type that is no copy and no allocation at all. The
`ArrowArray` points into DuckDB's own memory and the assembler copies from there
into the finished column, which is the one copy that was always going to happen.

Two types do not line up and get a small conversion into scratch memory that
this file allocates and the caller holds.

BOOLEAN is a byte per value in DuckDB and a bit per value in Arrow, so it is
packed here.

VARCHAR is closer than it looks. DuckDB's `string_t` and Arrow's string view are
both sixteen bytes, both start with a four byte length, and for a string of
twelve bytes or fewer both spell the rest as the data itself, so a short string
needs no conversion whatsoever. A longer one differs in the last eight bytes:
DuckDB keeps an absolute pointer where Arrow keeps a buffer index and an offset
into it. So the long ones are copied into a payload buffer and their views are
rewritten, and a column whose strings are all short costs one memcpy of the
vector.

Anything else, and there is a lot else, falls back to `duckdb_data_chunk_to_arrow`
in the reader that calls this. The decision is made once for the whole result
rather than per column, because the assembler takes one layout and a result that
is half converted by one route and half by another is two code paths to be wrong
in instead of one.
"""

from std.memory import unsafe_memcpy

from firepanda.buffer.buffer import Buffer

from .arrow_c import ArrowArray, ArrayBuffers, NullableVoidPtr, VoidPtr
from .duckdb import Handle, Library, MaybeHandle

comptime DUCK_BOOLEAN = 1
"""`DUCKDB_TYPE_BOOLEAN`, a byte per value."""

comptime DUCK_VARCHAR = 17
"""`DUCKDB_TYPE_VARCHAR`, a `string_t` per value."""

comptime DUCK_BLOB = 18
"""`DUCKDB_TYPE_BLOB`, also a `string_t` per value."""

comptime STRING_T_BYTES = 16
"""The size of a `duckdb_string_t`, which is also the size of an Arrow view."""

comptime STRING_T_INLINE = 12
"""The longest string DuckDB keeps inside the `string_t`. Arrow's number too,
which is the whole reason a short string needs no conversion."""


def duck_format(kind: Int32) -> String:
    """Names the Arrow type a DuckDB type maps onto, or the empty string.

    The empty string means this file cannot read the type directly and the
    caller should go through `duckdb_data_chunk_to_arrow`. It is not an error:
    DuckDB has around forty types and there is no reason to hand write the ones
    that are rare in a Parquet file.

    Args:
        kind: The `duckdb_type` enum value.

    Returns:
        The Arrow C Data Interface format string, or the empty string.
    """
    if kind == 1:
        return "b"
    if kind == 2:
        return "c"
    if kind == 3:
        return "s"
    if kind == 4:
        return "i"
    if kind == 5:
        return "l"
    if kind == 6:
        return "C"
    if kind == 7:
        return "S"
    if kind == 8:
        return "I"
    if kind == 9:
        return "L"
    if kind == 10:
        return "f"
    if kind == 11:
        return "g"
    if kind == 17:
        return "vu"
    if kind == 18:
        return "vz"
    return ""


def _width_of(kind: Int32) -> Int:
    """Returns the bytes one value of a fixed width DuckDB type occupies.

    Args:
        kind: The `duckdb_type` enum value.

    Returns:
        The width in bytes, or zero for a type this file converts instead.
    """
    if kind == 2 or kind == 6:
        return 1
    if kind == 3 or kind == 7:
        return 2
    if kind == 4 or kind == 8 or kind == 10:
        return 4
    if kind == 5 or kind == 9 or kind == 11:
        return 8
    return 0


struct Scratch(Movable):
    """The memory a converted chunk needs, allocated once for the whole read.

    Everything in here is pointed at by an `ArrowArray` that the assembler will
    read, so it has to outlive the assembly, and nothing in here may move after
    the pointers are taken. Both lists are given their full length up front and
    only ever have elements assigned in place, which is what stops a `List`
    growing under a pointer somebody already took.
    """

    var buffers: List[Buffer]
    """Two per column per chunk: the values, and the payload for a view type.
    Most of them stay empty, because a fixed width column converts to nothing."""

    var pointers: List[NullableVoidPtr]
    """Four per column per chunk, which is the `ArrowArray.buffers` array. It is
    one flat allocation rather than one per array so that no array's buffer list
    can be moved by another array being added."""

    var sizes: List[Int64]
    """One per column per chunk: the length of the payload buffer, which a view
    array carries in its last buffer."""

    def __init__(out self, count: Int):
        """Allocates room for a whole result.

        Args:
            count: The number of columns times the number of chunks.
        """
        self.buffers = List[Buffer](length=count * 2, fill=Buffer(0))
        self.pointers = List[NullableVoidPtr](length=count * 4, fill=None)
        self.sizes = List[Int64](length=count, fill=0)


def _erase[T: AnyType, o: MutOrigin](pointer: Pointer[T, o]) -> NullableVoidPtr:
    """Turns any pointer into the `void*` an Arrow buffer slot holds.

    Args:
        pointer: The pointer, to DuckDB's memory or to our own scratch.

    Returns:
        The same address, with its type and origin gone.
    """
    return pointer.unsafe_origin_cast[MutUntrackedOrigin]().unsafe_bitcast[
        NoneType
    ]()


def _valid(mask: MaybeHandle, i: Int) -> Bool:
    """Reads one bit of a DuckDB validity mask.

    A vector with no mask has no nulls, which is what DuckDB means by handing
    back a null pointer here rather than a mask of all ones.

    Args:
        mask: The mask, or nothing.
        i: The row within the chunk.

    Returns:
        True if the row is present.
    """
    if not mask:
        return True
    var word = mask.value().unsafe_bitcast[UInt64]().unsafe_offset(i // 64)[]
    return (word >> UInt64(i % 64)) & 1 == 1


def _pack_bools(data: Handle, mask: MaybeHandle, rows: Int) -> Buffer:
    """Packs DuckDB's byte per boolean into Arrow's bit per boolean.

    A null row's byte is not read, because DuckDB does not promise what is in it
    and a bit set from garbage is a value that was never in the file.

    Args:
        data: The vector's data pointer.
        mask: The vector's validity mask, or nothing.
        rows: How many rows the chunk holds.

    Returns:
        A buffer of `ceil(rows / 8)` bytes, set bit meaning true.
    """
    var out = Buffer((rows + 7) // 8)
    var bits = out.unsafe_ptr()
    for i in range(rows):
        if not _valid(mask, i):
            continue
        if data.unsafe_offset(i)[] != 0:
            bits.unsafe_offset(i // 8)[] |= UInt8(1) << UInt8(i % 8)
    return out^


def _payload_bytes(data: Handle, mask: MaybeHandle, rows: Int) -> Int:
    """Adds up how much room the long strings in a vector need.

    Args:
        data: The vector's data pointer, an array of `duckdb_string_t`.
        mask: The vector's validity mask, or nothing.
        rows: How many rows the chunk holds.

    Returns:
        The total bytes of every element longer than twelve.
    """
    var total = 0
    # A `string_t` is four words and the length is the first, so element `i * 4`
    # of a `UInt32` view of the vector is element `i`'s length.
    var lengths = data.unsafe_bitcast[UInt32]()
    for i in range(rows):
        if not _valid(mask, i):
            continue
        var length = Int(lengths.unsafe_offset(i * 4)[])
        if length > STRING_T_INLINE:
            total += length
    return total


def _views_of(
    data: Handle, mask: MaybeHandle, rows: Int, mut payload: Buffer
) -> Buffer:
    """Turns a vector of `string_t` into a buffer of Arrow string views.

    Args:
        data: The vector's data pointer.
        mask: The vector's validity mask, or nothing.
        rows: How many rows the chunk holds.
        payload: The buffer the long strings are copied into, already the size
            `_payload_bytes` asked for.

    Returns:
        A buffer of `rows * 16` bytes, one view per row, zero for a null.
    """
    var out = Buffer(rows * STRING_T_BYTES)
    var views = out.unsafe_ptr()
    var bytes = payload.unsafe_ptr()
    var cursor = 0
    for i in range(rows):
        if not _valid(mask, i):
            continue
        var src = data.unsafe_offset(i * STRING_T_BYTES)
        var dest = views.unsafe_offset(i * STRING_T_BYTES)
        var length = Int(src.unsafe_bitcast[UInt32]()[])
        if length <= STRING_T_INLINE:
            # The two layouts agree byte for byte here, length word included.
            unsafe_memcpy(dest=dest, src=src, count=STRING_T_BYTES)
            continue
        # Length and the four byte prefix are in the same place in both. What
        # differs is the last eight bytes, a pointer against a block and an
        # offset, and DuckDB's pointer is to the whole string rather than to the
        # part after the prefix.
        unsafe_memcpy(dest=dest, src=src, count=8)
        var words = dest.unsafe_bitcast[UInt32]()
        words.unsafe_offset(2)[] = 0
        words.unsafe_offset(3)[] = UInt32(cursor)
        var text = src.unsafe_offset(8).unsafe_bitcast[Handle]()[]
        unsafe_memcpy(dest=bytes.unsafe_offset(cursor), src=text, count=length)
        cursor += length
    return out^


def duck_array(
    lib: Library,
    chunk: Handle,
    column: Int,
    kind: Int32,
    mut scratch: Scratch,
    slot: Int,
) raises -> ArrowArray:
    """Describes one column of one data chunk as an Arrow array.

    The array points into DuckDB's memory for a fixed width type and into
    `scratch` for the two that need converting. Either way it owns nothing and
    carries no release callback, so the caller keeps the chunks and the scratch
    alive until everything that wanted the bytes has copied them.

    Args:
        lib: The open library.
        chunk: The data chunk.
        column: Which column of it.
        kind: The column's `duckdb_type`, which the caller read once from the
            result rather than once per chunk.
        scratch: Where a converted column's memory goes.
        slot: This column and chunk's index into the scratch, which is the
            column times the chunk count plus the chunk.

    Returns:
        The array.

    Raises:
        Error: If DuckDB will not hand over the vector, or the type is one this
            file does not read directly.
    """
    var vector = lib.chunk_vector(chunk, UInt64(column))
    if not vector:
        raise Error(String("duckdb: no vector for column ", column))
    var rows = Int(lib.chunk_size(chunk))
    var values = lib.vector_data(vector.value())
    if not values:
        raise Error(String("duckdb: no data for column ", column))
    var mask = lib.vector_validity(vector.value())

    var out = ArrowArray()
    out.length = Int64(rows)
    # Minus one is Arrow's "not computed". DuckDB hands out a mask when a vector
    # could hold a null rather than when it does, so counting here would mean a
    # pass over the bitmap to save the assembler a copy it makes anyway.
    out.null_count = -1 if mask else 0
    out.offset = 0

    var slots = scratch.pointers.unsafe_ptr().unsafe_offset(slot * 4)
    if mask:
        slots[] = _erase(mask.value())

    var width = _width_of(kind)
    if width != 0:
        slots.unsafe_offset(1)[] = _erase(values.value())
        out.n_buffers = 2
    elif kind == DUCK_BOOLEAN:
        scratch.buffers[slot * 2] = _pack_bools(values.value(), mask, rows)
        slots.unsafe_offset(1)[] = _erase(
            scratch.buffers[slot * 2].unsafe_ptr()
        )
        out.n_buffers = 2
    elif kind == DUCK_VARCHAR or kind == DUCK_BLOB:
        var size = _payload_bytes(values.value(), mask, rows)
        scratch.sizes[slot] = Int64(size)
        var payload = Buffer(size)
        scratch.buffers[slot * 2] = _views_of(
            values.value(), mask, rows, payload
        )
        scratch.buffers[slot * 2 + 1] = payload^
        slots.unsafe_offset(1)[] = _erase(
            scratch.buffers[slot * 2].unsafe_ptr()
        )
        slots.unsafe_offset(2)[] = _erase(
            scratch.buffers[slot * 2 + 1].unsafe_ptr()
        )
        # Validity, views, one payload buffer, and the buffer of sizes that
        # Arrow puts last on a view array because the number of data buffers is
        # the producer's choice.
        slots.unsafe_offset(3)[] = _erase(
            scratch.sizes.unsafe_ptr().unsafe_offset(slot)
        )
        out.n_buffers = 4
    else:
        raise Error(String("duckdb: type ", kind, " is not read directly"))

    out.buffers = slots.unsafe_origin_cast[MutUntrackedOrigin]()
    return out^
