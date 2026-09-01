"""Handing a firepanda column to a C consumer without copying it.

`arrow_c.mojo` says what the structs are. This says how to fill them in and, much
more importantly, who frees what afterwards.

Nothing here copies a value. The pointers in the exported `ArrowArray` are the
firepanda `Buffer`'s own pointers, and `test_the_values_buffer_is_not_copied`
asserts that by comparing addresses rather than contents, because comparing
contents would pass just as happily for a copy.

Two pieces of luck make that possible and both are worth writing down, because if
either stops holding the export becomes a copy and the tests should be the thing
that notices. firepanda's validity bitmap is one bit per row, least significant bit
first, with one meaning present. Arrow's is one bit per row, least significant bit
first, with one meaning present. And a firepanda values buffer for a fixed width
type is exactly the Arrow values buffer for that type, because both are the values
end to end with no header.

### Who frees what

Zero copy across a language boundary is an ownership problem wearing a performance
problem's clothes. The consumer gets pointers into memory firepanda allocated, and
it is going to outlive the Mojo scope that made it, so something has to keep the
column alive for exactly as long as the consumer holds it and then let go.

That something is a heap box behind `private_data`. `export_array` moves the column
into a box, points the `ArrowArray` at the box's own buffers, and installs a
release callback that destroys the box. The consumer calls release when it is done,
which is the contract it already has to honour for every other Arrow producer, and
the column dies then.

The box is allocated with `malloc` and freed with `free` rather than through Mojo's
allocator. It is freed from inside an `abi("C")` callback that a foreign runtime
invokes on a thread firepanda knows nothing about, and C's allocator is the one
that is unambiguously safe to call from there.

`export_array` therefore takes its column by value and consumes it. That is not a
convenience choice. firepanda columns are deep copied rather than refcounted today,
so sharing one between a `DataFrame` and a consumer is not something this layer can
offer, and a signature that borrowed would be promising it.

### Strings, which turned out to be free

A firepanda string column is already the Arrow view layout, byte for byte. Both
store sixteen bytes per element: a little endian uint32 length, then either the
data inlined when it is at most twelve bytes, or a four byte prefix followed by a
uint32 buffer index and a uint32 offset. firepanda chose that layout for its own
reasons, recorded in `array/strview.mojo`, and landing on Arrow's was not one of
them, so a test compares the bytes rather than trusting the coincidence to hold.

The consequence is that a string column exports without copying either, which is
the case that would have hurt most: the payload of a text column is usually the
largest buffer in the frame. A finished firepanda column has exactly one payload
block, so it exports as four buffers, and the block is emitted even when it is
empty so that the count is a constant. A consumer never reads an empty one,
because the sizes buffer says it is zero long.

### Bool, which is not

The one copy in this file. firepanda stores a bool as a byte, because that is what
a kernel wants to load, and Arrow stores it as a bit, so exporting one is a packing
pass. It shrinks by a factor of eight on the way, which is why the cost of it
hardly shows, and the packed buffer lives in the box like everything else.

### What is still not here

The null type, which firepanda has as a `LogicalType` but not as a column anything
constructs, so there is nothing to export. It raises with that reason.
"""

from std.ffi import c_char, external_call
from std.sys import size_of

from firepanda.array.any import AnyArray
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.logical import LogicalType

from .arrow_c import (
    ARROW_FLAG_NULLABLE,
    ArrayPtr,
    ArrowArray,
    ArrowSchema,
    NullableVoidPtr,
    SchemaPtr,
    VoidPtr,
    array_release_callback,
    format_for,
    schema_release_callback,
)


struct _SchemaBox(Movable):
    """What an exported schema's `private_data` points at.

    A schema hands out two `const char*` and does not own the storage behind
    either, so somebody has to. This is that somebody: the format string and the
    field name, null terminated, alive until the consumer releases the schema.
    """

    var format: List[UInt8]
    var name: List[UInt8]
    var has_name: Bool

    def __init__(
        out self, var format: List[UInt8], var name: List[UInt8], has_name: Bool
    ):
        """Constructs the box.

        Args:
            format: The null terminated format string.
            name: The null terminated field name, empty when there is none.
            has_name: Whether the field has a name at all, which is different
                from having an empty one.
        """
        self.format = format^
        self.name = name^
        self.has_name = has_name


struct _ArrayBox(Movable):
    """What an exported array's `private_data` points at.

    The column, which is what keeps the values and the validity alive, and the
    array of buffer pointers, which the C struct points at rather than owns.

    Two fields exist only for the types that need more than the column already
    has. `packed` is the bit packed values of a bool column, which is the one
    buffer this file ever builds. `sizes` is the lengths of a view column's
    variadic data buffers, which Arrow wants as a buffer rather than a field, so
    they cannot be a local.
    """

    var column: AnyArray
    var buffers: List[NullableVoidPtr]
    var packed: Optional[Bitmap]
    var sizes: List[Int64]

    def __init__(
        out self,
        var column: AnyArray,
        var buffers: List[NullableVoidPtr],
        var packed: Optional[Bitmap],
        var sizes: List[Int64],
    ):
        """Constructs the box.

        Args:
            column: The column being exported, moved in.
            buffers: The pointers, in Arrow's order.
            packed: The bit packed values of a bool column, empty otherwise.
            sizes: The lengths of the variadic data buffers, for a view column.
        """
        self.column = column^
        self.buffers = buffers^
        self.packed = packed^
        self.sizes = sizes^


def _c_string(text: StringSlice) -> List[UInt8]:
    """Copies a string into a null terminated byte list.

    Mojo string literals are not guaranteed to be null terminated and Arrow's
    `format` and `name` are both `const char*`, so the terminator is added here
    rather than assumed.

    Args:
        text: The string.

    Returns:
        Its bytes followed by a zero.
    """
    var out = List[UInt8](capacity=text.byte_length() + 1)
    for byte in text.as_bytes():
        out.append(byte)
    out.append(0)
    return out^


def _as_void[o: MutOrigin](p: Pointer[UInt8, o]) -> VoidPtr:
    """Reinterprets a byte pointer as the `void*` Arrow's buffer array holds.

    The origin is dropped rather than translated. What keeps the memory alive is
    the box, not a Mojo scope, and a tracked origin here would be a claim the
    compiler cannot check.

    Args:
        p: The pointer.

    Returns:
        The same address, as a `void*`.
    """
    return p.unsafe_origin_cast[MutUntrackedOrigin]().unsafe_bitcast[NoneType]()


def pack_bools(column: AnyArray) -> Bitmap:
    """Packs a byte per value bool column into a bit per value buffer.

    The one copy in this file, and it is not avoidable: firepanda stores a bool
    as a byte because that is what a kernel wants to load, and Arrow stores it as
    a bit. Eight rows per byte rather than one, so the copy also shrinks by a
    factor of eight, which is why the cost of it barely shows.

    A null row's value byte is not meaningful, and neither is its Arrow value
    bit, so nulls are packed as whatever the byte happened to hold rather than
    being special cased. Arrow says the values buffer is undefined where validity
    says null.

    Args:
        column: The bool column.

    Returns:
        A bitmap holding one bit per value.
    """
    var length = len(column)
    var out = Bitmap(length, all_valid=False)
    var values = column.data.values.unsafe_ptr()
    for i in range(length):
        if values.unsafe_offset(i).unsafe_load() != 0:
            out.set(i, True)
    return out^


def _release_exported_schema(schema: SchemaPtr) abi("C") -> None:
    """Frees an exported schema. Installed as `ArrowSchema.release`."""
    if not schema[].release:
        return
    if schema[].private_data:
        var box = schema[].private_data.value().unsafe_bitcast[_SchemaBox]()
        box.unsafe_deinit_pointee()
        external_call["free", NoneType](box)
        schema[].private_data = None
    schema[].format = None
    schema[].name = None
    schema[].release = None


def _release_exported_array(array: ArrayPtr) abi("C") -> None:
    """Frees an exported array. Installed as `ArrowArray.release`."""
    if not array[].release:
        return
    if array[].private_data:
        var box = array[].private_data.value().unsafe_bitcast[_ArrayBox]()
        box.unsafe_deinit_pointee()
        external_call["free", NoneType](box)
        array[].private_data = None
    array[].buffers = None
    array[].release = None


def export_schema(
    type: LogicalType, name: StringSlice = ""
) raises -> ArrowSchema:
    """Describes a firepanda type as an `ArrowSchema`.

    The returned schema owns a heap box and must be released by whoever ends up
    holding it, exactly once. Every column is marked nullable, because firepanda
    columns are: a column with no nulls today can be the result of an operation
    that produces them tomorrow, and the flag describes the type rather than the
    values.

    Args:
        type: The column type.
        name: The field name, or empty for a top level array, where Arrow wants
            a null name rather than an empty one.

    Returns:
        A schema the caller owns and must release.

    Raises:
        Error: If the type has no Arrow format string.
    """
    var format = _c_string(format_for(type))
    var has_name = name.byte_length() > 0
    var name_bytes = _c_string(name)

    var box = external_call["malloc", Pointer[_SchemaBox, MutUntrackedOrigin]](
        size_of[_SchemaBox]()
    )
    box.unsafe_write(_SchemaBox(format^, name_bytes^, has_name))

    var schema = ArrowSchema()
    schema.format = (
        box[]
        .format.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[c_char]()
    )
    if has_name:
        schema.name = (
            box[]
            .name.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
            .unsafe_bitcast[c_char]()
        )
    schema.flags = ARROW_FLAG_NULLABLE
    schema.private_data = box.unsafe_bitcast[NoneType]()
    schema.release = schema_release_callback(_release_exported_schema)
    return schema^


def export_array(var column: AnyArray) raises -> ArrowArray:
    """Hands a firepanda column to a C consumer without copying its values.

    The column is moved into a box that lives until the returned array is
    released. Nothing is copied for any type except bool: `buffers[1]` is the
    column's own values pointer, or its own views pointer for a string column,
    and `buffers[0]` is its own validity pointer.

    The validity buffer is null when the column has no nulls, which Arrow allows
    and consumers use to skip a branch per value. A column that has nulls hands
    out its bitmap as it stands, because firepanda's bitmap and Arrow's are the
    same bytes in the same order with the same meaning.

    Two buffers for a fixed width or bool column, four for a string or binary
    one: validity, the views, the single payload block, and the block's length.

    Args:
        column: The column, consumed.

    Returns:
        An array the caller owns and must release exactly once.

    Raises:
        Error: If the column is of the null type, which nothing constructs, or
            is a string column with no text storage, which nothing constructs
            either but which would be read as a column of garbage views.
    """
    var type = column.type
    if type == LogicalType.NULL:
        raise Error(
            "arrow: cannot export a null column yet, because firepanda has no"
            " column that carries the null type at run time"
        )
    var is_view = type == LogicalType.STRING or type == LogicalType.BINARY
    if is_view and not column.text:
        raise Error(
            "arrow: a string column with no text storage cannot be exported"
        )
    # Raises for anything else that has no format string, before the box exists
    # and there is something to leak.
    _ = format_for(type)

    var length = len(column)
    var null_count = column.data.validity.null_count()
    var packed = Optional[Bitmap](None)
    if type == LogicalType.BOOL:
        packed = pack_bools(column)

    var box = external_call["malloc", Pointer[_ArrayBox, MutUntrackedOrigin]](
        size_of[_ArrayBox]()
    )
    box.unsafe_write(
        _ArrayBox(
            column^,
            List[NullableVoidPtr](capacity=4),
            packed^,
            List[Int64](capacity=1),
        )
    )

    # Filled after the move, so the pointers are the box's copy of the column
    # rather than a local that is about to go away.
    if null_count == 0:
        box[].buffers.append(None)
    else:
        box[].buffers.append(_as_void(box[].column.data.validity.unsafe_ptr()))

    var n_buffers = 2
    if type == LogicalType.BOOL:
        box[].buffers.append(_as_void(box[].packed.value().unsafe_ptr()))
    elif is_view:
        n_buffers = 4
        ref text = box[].column.text.value()
        box[].sizes.append(Int64(len(text.payload)))
        box[].buffers.append(_as_void(text.views.unsafe_ptr()))
        box[].buffers.append(_as_void(text.payload.unsafe_ptr()))
        box[].buffers.append(
            box[]
            .sizes.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
            .unsafe_bitcast[NoneType]()
        )
    else:
        box[].buffers.append(_as_void(box[].column.data.values.unsafe_ptr()))

    var array = ArrowArray()
    array.length = Int64(length)
    array.null_count = Int64(null_count)
    array.offset = 0
    array.n_buffers = Int64(n_buffers)
    array.n_children = 0
    array.buffers = (
        box[].buffers.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )
    array.private_data = box.unsafe_bitcast[NoneType]()
    array.release = array_release_callback(_release_exported_array)
    return array^
