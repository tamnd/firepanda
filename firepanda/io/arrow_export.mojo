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

### A whole frame, which is a struct

A frame is not a list of arrays to Arrow, it is one array of struct type with one
child per column, which is why `__arrow_c_array__` on a table like object hands
back a single pair rather than a pair per column. `export_frame_schema` and
`export_frame_array` build that, and the only new idea in them is that a parent
owns its children: the box behind `private_data` holds the child structs by
value, the `children` field points at an array of pointers into that box, and the
parent's release callback releases every child before freeing itself.

Nothing extra is copied. The children are the same exports a column would have
produced on its own, and the struct on top of them is one null buffer and a list
of pointers.

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
    release_array,
    release_schema,
    schema_release_callback,
)

comptime STRUCT_FORMAT = "+s"
"""The Arrow format string for a struct, which is what a frame is."""


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


def _check_exportable(column: AnyArray) raises:
    """Refuses the two columns that cannot be exported.

    Called before anything is allocated, so a refusal leaks nothing.

    Args:
        column: The column about to be exported.

    Raises:
        Error: If the column is of the null type, which nothing constructs, or
            is a string column with no text storage, which nothing constructs
            either but which would be read as a column of garbage views, or is
            of any other type that has no Arrow format string.
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
    _ = format_for(type)


def _fill_buffers(
    column: Pointer[AnyArray, MutAnyOrigin],
    mut buffers: List[NullableVoidPtr],
    mut packed: Optional[Bitmap],
    mut sizes: List[Int64],
) raises -> Int:
    """Points a box's buffer array at a column's own memory.

    This is the whole of the zero copy, and it is shared by both export paths
    because the answer does not depend on who owns the column afterwards. The
    three lists it fills live in the box, so every pointer it writes stays valid
    for as long as the exported array does.

    The validity buffer is null when the column has no nulls, which Arrow allows
    and consumers use to skip a branch per value. A column that has nulls hands
    out its bitmap as it stands, because firepanda's bitmap and Arrow's are the
    same bytes in the same order with the same meaning.

    Args:
        column: The column to read. It has to outlive the exported array, which
            is the caller's problem and is the only real difference between the
            two paths.
        buffers: The box's buffer array, filled in Arrow's order.
        packed: The box's slot for a bool column's bit packed values, set here
            rather than passed in because the packed buffer has to live in the
            box like every other one.
        sizes: The box's slot for a view column's payload length.

    Returns:
        How many buffers the array has, which is two for a fixed width or a bool
        column and four for a view one.
    """
    var type = column[].type
    if column[].data.validity.null_count() == 0:
        buffers.append(None)
    else:
        buffers.append(_as_void(column[].data.validity.unsafe_ptr()))

    if type == LogicalType.BOOL:
        packed = pack_bools(column[])
        buffers.append(_as_void(packed.value().unsafe_ptr()))
        return 2

    if type == LogicalType.STRING or type == LogicalType.BINARY:
        ref text = column[].text.value()
        sizes.append(Int64(len(text.payload)))
        buffers.append(_as_void(text.views.unsafe_ptr()))
        buffers.append(_as_void(text.payload.unsafe_ptr()))
        buffers.append(
            sizes.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
            .unsafe_bitcast[NoneType]()
        )
        return 4

    buffers.append(_as_void(column[].data.values.unsafe_ptr()))
    return 2


def export_array(var column: AnyArray) raises -> ArrowArray:
    """Hands a firepanda column to a C consumer without copying its values.

    The column is moved into a box that lives until the returned array is
    released. Nothing is copied for any type except bool: `buffers[1]` is the
    column's own values pointer, or its own views pointer for a string column,
    and `buffers[0]` is its own validity pointer.

    Two buffers for a fixed width or bool column, four for a string or binary
    one: validity, the views, the single payload block, and the block's length.

    This is the path for a column nobody else wants. A column that is still in a
    frame the caller keeps goes through `export_array_borrowed` instead.

    Args:
        column: The column, consumed.

    Returns:
        An array the caller owns and must release exactly once.

    Raises:
        Error: If the column cannot be exported. See `_check_exportable`.
    """
    _check_exportable(column)
    var length = len(column)
    var null_count = column.data.validity.null_count()

    var box = external_call["malloc", Pointer[_ArrayBox, MutUntrackedOrigin]](
        size_of[_ArrayBox]()
    )
    box.unsafe_write(
        _ArrayBox(
            column^,
            List[NullableVoidPtr](capacity=4),
            Optional[Bitmap](None),
            List[Int64](capacity=1),
        )
    )

    # Filled after the move, so the pointers are the box's copy of the column
    # rather than a local that is about to go away.
    var n_buffers = _fill_buffers(
        Pointer(to=box[].column).unsafe_origin_cast[MutAnyOrigin](),
        box[].buffers,
        box[].packed,
        box[].sizes,
    )

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


struct _BorrowedArrayBox[K: Copyable & Deinitable](Movable):
    """What a borrowed array's `private_data` points at.

    The same three working lists as `_ArrayBox` and, instead of the column, a
    keep alive. The keep alive is whatever the caller decided is the thing that
    owns the memory the buffers point at, held by value so that the exported
    array is what keeps it from being destroyed.

    That is what makes a zero copy export of a frame Python is still holding
    possible at all. firepanda columns are deep copied rather than refcounted, so
    a box that held a column would be holding a copy, and the export would no
    longer be pointing at the frame the user has. A box that holds a shared
    pointer to the frame points at the real thing and costs one atomic increment.
    """

    var keep: Self.K
    var buffers: List[NullableVoidPtr]
    var packed: Optional[Bitmap]
    var sizes: List[Int64]

    def __init__(
        out self,
        var keep: Self.K,
        var buffers: List[NullableVoidPtr],
        var packed: Optional[Bitmap],
        var sizes: List[Int64],
    ):
        """Constructs the box.

        Args:
            keep: What owns the memory the buffers point at.
            buffers: The pointers, in Arrow's order.
            packed: The bit packed values of a bool column, empty otherwise.
            sizes: The lengths of the variadic data buffers, for a view column.
        """
        self.keep = keep^
        self.buffers = buffers^
        self.packed = packed^
        self.sizes = sizes^


def _release_borrowed_array[
    K: Copyable & Deinitable
](array: ArrayPtr) abi("C") -> None:
    """Frees a borrowed array. Installed as `ArrowArray.release`.

    Destroying the box drops the keep alive, which is the point: for a shared
    pointer that is an atomic decrement, and the frame goes away with the last
    consumer rather than with the Mojo scope that exported it.

    Parameters:
        K: The keep alive's type, which has to be the same one the matching
            `export_array_borrowed` used or the box is read as the wrong type.
    """
    if not array[].release:
        return
    if array[].private_data:
        var box = (
            array[].private_data.value().unsafe_bitcast[_BorrowedArrayBox[K]]()
        )
        box.unsafe_deinit_pointee()
        external_call["free", NoneType](box)
        array[].private_data = None
    array[].buffers = None
    array[].release = None


def export_array_borrowed[
    K: Copyable & Deinitable
](column: Pointer[AnyArray, MutAnyOrigin], var keep: K) raises -> ArrowArray:
    """Hands out a column that something else owns, without copying it.

    The difference from `export_array` is who is holding the column afterwards.
    There the array owns it, here the array owns a share of whatever does, and in
    both cases the buffers are the column's own memory and the consumer decides
    when it is finished by calling release.

    Parameters:
        K: The keep alive's type.

    Args:
        column: The column. It must be reachable through `keep`, because that is
            the only thing this export keeps alive.
        keep: A share of whatever owns the column.

    Returns:
        An array the caller owns and must release exactly once.

    Raises:
        Error: If the column cannot be exported. See `_check_exportable`.
    """
    _check_exportable(column[])
    var length = len(column[])
    var null_count = column[].data.validity.null_count()

    var box = external_call[
        "malloc", Pointer[_BorrowedArrayBox[K], MutUntrackedOrigin]
    ](size_of[_BorrowedArrayBox[K]]())
    box.unsafe_write(
        _BorrowedArrayBox[K](
            keep^,
            List[NullableVoidPtr](capacity=4),
            Optional[Bitmap](None),
            List[Int64](capacity=1),
        )
    )

    var n_buffers = _fill_buffers(
        column, box[].buffers, box[].packed, box[].sizes
    )

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
    array.release = array_release_callback(_release_borrowed_array[K])
    return array^


struct _StructSchemaBox(Movable):
    """What an exported struct schema's `private_data` points at.

    Three things that all have to outlive the parent and none of which can be a
    local. The format string, which is `+s` and is the same two bytes every time
    but still has to be null terminated storage somebody owns. The child schemas
    themselves, by value, because the parent's `children` field is an array of
    pointers and something has to be at the other end of them. And the array of
    pointers, because `children` points at it rather than owning it.

    The children are held before the pointers are taken, and the pointers are
    filled in after the box has been moved into its allocation, so that every one
    of them names the box's own copy rather than a list that is about to be moved
    out from under them.
    """

    var format: List[UInt8]
    var children: List[ArrowSchema]
    var pointers: List[SchemaPtr]

    def __init__(
        out self,
        var format: List[UInt8],
        var children: List[ArrowSchema],
        var pointers: List[SchemaPtr],
    ):
        """Constructs the box.

        Args:
            format: The null terminated `+s`.
            children: The child schemas, moved in.
            pointers: An empty list with room for one pointer per child, filled
                in by the caller after this has been moved into place.
        """
        self.format = format^
        self.children = children^
        self.pointers = pointers^


struct _StructArrayBox(Movable):
    """What an exported struct array's `private_data` points at.

    The same shape as `_StructSchemaBox`, plus the one buffer slot a struct array
    has. Arrow says a struct array has exactly one buffer, its validity, and a
    frame has no row level validity, so the slot is there and is null.
    """

    var buffers: List[NullableVoidPtr]
    var children: List[ArrowArray]
    var pointers: List[ArrayPtr]

    def __init__(
        out self,
        var buffers: List[NullableVoidPtr],
        var children: List[ArrowArray],
        var pointers: List[ArrayPtr],
    ):
        """Constructs the box.

        Args:
            buffers: The one element buffer array, holding null.
            children: The child arrays, moved in.
            pointers: An empty list with room for one pointer per child.
        """
        self.buffers = buffers^
        self.children = children^
        self.pointers = pointers^


def _release_struct_schema(schema: SchemaPtr) abi("C") -> None:
    """Frees an exported struct schema. Installed as `ArrowSchema.release`.

    The parent releases its children before freeing the box that holds them,
    which is what the C Data Interface requires of a parent and is also simply
    what has to happen: each child owns a box of its own and nothing else is
    going to release it.
    """
    if not schema[].release:
        return
    if schema[].private_data:
        var box = (
            schema[].private_data.value().unsafe_bitcast[_StructSchemaBox]()
        )
        for i in range(len(box[].children)):
            release_schema(box[].children[i])
        box.unsafe_deinit_pointee()
        external_call["free", NoneType](box)
        schema[].private_data = None
    schema[].format = None
    schema[].name = None
    schema[].children = None
    schema[].n_children = 0
    schema[].release = None


def _release_struct_array(array: ArrayPtr) abi("C") -> None:
    """Frees an exported struct array. Installed as `ArrowArray.release`."""
    if not array[].release:
        return
    if array[].private_data:
        var box = array[].private_data.value().unsafe_bitcast[_StructArrayBox]()
        for i in range(len(box[].children)):
            release_array(box[].children[i])
        box.unsafe_deinit_pointee()
        external_call["free", NoneType](box)
        array[].private_data = None
    array[].buffers = None
    array[].children = None
    array[].n_children = 0
    array[].release = None


def export_frame_schema(
    types: List[LogicalType], names: List[String]
) raises -> ArrowSchema:
    """Describes a frame as a struct schema with one child per column.

    A frame is a struct in Arrow's type system, which is why `__arrow_c_array__`
    on a table like object hands back one array rather than a list of them. The
    format string is `+s` and the field names live on the children, so this is
    where a frame's column names cross the boundary.

    Args:
        types: The column types, in order.
        names: The column names, in the same order.

    Returns:
        A schema the caller owns and must release exactly once, which releases
        every child with it.

    Raises:
        Error: If the two lists are different lengths, or if any column type has
            no Arrow format string.
    """
    if len(types) != len(names):
        raise Error(
            String(
                "arrow: ",
                len(types),
                " column types but ",
                len(names),
                " column names",
            )
        )
    var children = List[ArrowSchema](capacity=len(types))
    for i in range(len(types)):
        children.append(export_schema(types[i], names[i]))

    var box = external_call[
        "malloc", Pointer[_StructSchemaBox, MutUntrackedOrigin]
    ](size_of[_StructSchemaBox]())
    box.unsafe_write(
        _StructSchemaBox(
            _c_string(STRUCT_FORMAT),
            children^,
            List[SchemaPtr](capacity=len(types)),
        )
    )
    # After the move, so every pointer names the box's own child rather than a
    # list that has just been moved out from under it.
    for i in range(len(box[].children)):
        box[].pointers.append(
            Pointer(to=box[].children[i]).unsafe_origin_cast[
                MutUntrackedOrigin
            ]()
        )

    var schema = ArrowSchema()
    schema.format = (
        box[]
        .format.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[c_char]()
    )
    schema.flags = ARROW_FLAG_NULLABLE
    schema.n_children = Int64(len(box[].children))
    schema.children = (
        box[].pointers.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )
    schema.private_data = box.unsafe_bitcast[NoneType]()
    schema.release = schema_release_callback(_release_struct_schema)
    return schema^


def export_frame_array[
    K: Copyable & Deinitable
](
    columns: List[Pointer[AnyArray, MutAnyOrigin]], length: Int, var keep: K
) raises -> ArrowArray:
    """Hands a whole frame to a C consumer as a struct array.

    Nothing is copied that a single column export would not have copied, which is
    to say nothing at all except a bool column's bit packing. The struct on top
    is one null buffer and a list of pointers.

    A struct array carries no validity of its own here, because a frame has no
    concept of a null row. Arrow requires the buffer slot to exist and allows it
    to be null, which is what a consumer reads as every row present.

    Every child gets its own share of the keep alive rather than relying on the
    parent's. A consumer is allowed to move a child out and release the parent,
    and a child whose lifetime depended on its parent's box would then be reading
    memory nobody was holding.

    Parameters:
        K: The keep alive's type.

    Args:
        columns: One pointer per column, in schema order. Each must be reachable
            through `keep`, and each must have exactly the length given, which is
            Arrow's requirement for a struct rather than firepanda's.
        length: The row count.
        keep: A share of whatever owns the columns.

    Returns:
        An array the caller owns and must release exactly once, which releases
        every child with it.

    Raises:
        Error: If any column is a length other than `length`, or if any column
            cannot be exported.
    """
    for i in range(len(columns)):
        if len(columns[i][]) != length:
            raise Error(
                String(
                    "arrow: column ",
                    i,
                    " has ",
                    len(columns[i][]),
                    " rows but the frame has ",
                    length,
                )
            )

    var children = List[ArrowArray](capacity=len(columns))
    # A failure part way through leaves the children already built with nobody
    # to release them, so they are released here before the error goes up.
    try:
        for i in range(len(columns)):
            children.append(export_array_borrowed(columns[i], keep.copy()))
    except cause:
        for i in range(len(children)):
            release_array(children[i])
        raise cause

    var box = external_call[
        "malloc", Pointer[_StructArrayBox, MutUntrackedOrigin]
    ](size_of[_StructArrayBox]())
    var buffers = List[NullableVoidPtr](capacity=1)
    buffers.append(None)
    box.unsafe_write(
        _StructArrayBox(
            buffers^, children^, List[ArrayPtr](capacity=len(columns))
        )
    )
    for i in range(len(box[].children)):
        box[].pointers.append(
            Pointer(to=box[].children[i]).unsafe_origin_cast[
                MutUntrackedOrigin
            ]()
        )

    var array = ArrowArray()
    array.length = Int64(length)
    array.null_count = 0
    array.offset = 0
    array.n_buffers = 1
    array.n_children = Int64(len(box[].children))
    array.buffers = (
        box[].buffers.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )
    array.children = (
        box[].pointers.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )
    array.private_data = box.unsafe_bitcast[NoneType]()
    array.release = array_release_callback(_release_struct_array)
    return array^
