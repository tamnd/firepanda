"""The Arrow C Data Interface, declared.

This file is the boundary. `ArrowSchema` and `ArrowArray` are two of the structs
every Arrow producer and consumer in the world agrees on, and getting their layout
wrong is not a compile error anywhere, in any language. It is a wrong pointer read
at run time in somebody else's process.

So this file declares them and nothing else. The export and import directions
build on it and are separate, and `ArrowArrayStream` arrives with whichever of
them needs it first. What is here is the part that has to be exactly right and can
be checked without allocating anything: the field order, the field widths, and the
format string for every type firepanda has.

Three things about Mojo 1.0 decided how these are written, and all three were
found by trying the alternatives.

A C function pointer field is `def (args) thin abi("C") -> None`. The `thin`
effect is what makes it a bare pointer rather than a callable that might carry a
capture context, and without it the field is a trait type, which a struct cannot
hold. `abi("C")` is the calling convention and it goes in the same effects
position, both on the type and on the declaration of any function assigned to it.

Every pointer field is `Optional[Pointer[...]]` rather than `Pointer[...]`. Mojo's
`Pointer` is not nullable and has no way to spell null, which sounds like an
obstacle and is actually the right answer, because in C most of these fields are
allowed to be null and the ones that are not still have to be null in the released
state. `Optional[Pointer[...]]` is eight bytes, since a null pointer is the niche
the discriminant packs into, so the layout is unchanged and the nullability is now
in the type rather than in a comment.

That does not extend to function pointers. `Optional[def (...) thin abi("C") ->
None]` is sixteen bytes, because Mojo has no niche for a function pointer, and
sixteen bytes in the middle of the struct moves every field after it. So the
release fields are typed as nullable void pointers, and `schema_release_callback`,
`array_release_callback`, `release_schema` and `release_array` do the
reinterpretation at the two ends. `test_a_function_pointer_is_one_word` and
`test_nullable_pointer_is_one_word` are what keep both halves of that claim
honest.

Origins are untracked rather than external. The memory on the other side of these
pointers belongs to whichever library produced the structure, its lifetime is
governed by the release callback rather than by any Mojo scope, and asking the
compiler to track an origin it cannot see would be a claim rather than a check.

Reference: https://arrow.apache.org/docs/format/CDataInterface.html
"""

from std.ffi import c_char

from firepanda.dtype.logical import LogicalType


comptime CString = Pointer[c_char, ImmUntrackedOrigin]
"""A borrowed, null terminated C string."""

comptime NullableCString = Optional[CString]
"""A `const char*` that may be null."""

comptime VoidPtr = Pointer[NoneType, MutUntrackedOrigin]
"""A `void*`."""

comptime NullableVoidPtr = Optional[VoidPtr]
"""A `void*` that may be null. Eight bytes, with null as the empty case."""

comptime SchemaPtr = Pointer[ArrowSchema, MutUntrackedOrigin]
"""A `struct ArrowSchema*`."""

comptime ArrayPtr = Pointer[ArrowArray, MutUntrackedOrigin]
"""A `struct ArrowArray*`."""

comptime SchemaChildren = Optional[Pointer[SchemaPtr, MutUntrackedOrigin]]
"""A `struct ArrowSchema**` that may be null."""

comptime ArrayChildren = Optional[Pointer[ArrayPtr, MutUntrackedOrigin]]
"""A `struct ArrowArray**` that may be null."""

comptime ArrayBuffers = Optional[Pointer[NullableVoidPtr, MutUntrackedOrigin]]
"""A `const void**` that may be null."""

comptime SchemaRelease = def(SchemaPtr) thin abi("C") -> None
"""The type of `ArrowSchema.release`."""

comptime ArrayRelease = def(ArrayPtr) thin abi("C") -> None
"""The type of `ArrowArray.release`."""

comptime ARROW_FLAG_DICTIONARY_ORDERED: Int64 = 1
"""The dictionary indices are meaningfully ordered."""

comptime ARROW_FLAG_NULLABLE: Int64 = 2
"""The field may hold nulls."""

comptime ARROW_FLAG_MAP_KEYS_SORTED: Int64 = 4
"""A map's keys are sorted."""


@fieldwise_init
struct ArrowSchema(Copyable, Movable):
    """The C `struct ArrowSchema`. Seventy two bytes, nine fields, this order.
    """

    var format: NullableCString
    """The type, as a format string. Never null except in the released state."""

    var name: NullableCString
    """The field name. May be null, and is null for a top level array."""

    var metadata: NullableCString
    """Key value metadata in Arrow's own packed encoding. May be null."""

    var flags: Int64
    """A bitmask of the `ARROW_FLAG_` values."""

    var n_children: Int64
    """How many child schemas."""

    var children: SchemaChildren
    """`n_children` pointers to child schemas. Null when there are none."""

    var dictionary: NullableVoidPtr
    """The value schema, for a dictionary encoded type. Null otherwise."""

    var release: NullableVoidPtr
    """The callback that frees this schema. Null once it has been called."""

    var private_data: NullableVoidPtr
    """The producer's own state. Opaque to everyone else."""

    def __init__(out self):
        """Constructs a released schema, which is the all zero state."""
        self.format = None
        self.name = None
        self.metadata = None
        self.flags = 0
        self.n_children = 0
        self.children = None
        self.dictionary = None
        self.release = None
        self.private_data = None

    def is_released(self) -> Bool:
        """Reports whether this schema has been released.

        Returns:
            True if the release callback is null, which is what the C interface
            defines released to mean.
        """
        return not self.release


@fieldwise_init
struct ArrowArray(Copyable, Movable):
    """The C `struct ArrowArray`. Eighty bytes, ten fields, this order."""

    var length: Int64
    """How many values."""

    var null_count: Int64
    """How many are null, or -1 for not yet computed."""

    var offset: Int64
    """Where the logical first value starts inside the buffers."""

    var n_buffers: Int64
    """How many entries `buffers` has."""

    var n_children: Int64
    """How many child arrays."""

    var buffers: ArrayBuffers
    """`n_buffers` pointers, any of which may be null."""

    var children: ArrayChildren
    """`n_children` pointers to child arrays. Null when there are none."""

    var dictionary: NullableVoidPtr
    """The values array, for a dictionary encoded type. Null otherwise."""

    var release: NullableVoidPtr
    """The callback that frees this array. Null once it has been called."""

    var private_data: NullableVoidPtr
    """The producer's own state. Opaque to everyone else."""

    def __init__(out self):
        """Constructs a released array, which is the all zero state."""
        self.length = 0
        self.null_count = 0
        self.offset = 0
        self.n_buffers = 0
        self.n_children = 0
        self.buffers = None
        self.children = None
        self.dictionary = None
        self.release = None
        self.private_data = None

    def is_released(self) -> Bool:
        """Reports whether this array has been released.

        Returns:
            True if the release callback is null.
        """
        return not self.release


def schema_release_callback(f: SchemaRelease) -> VoidPtr:
    """Reinterprets a schema release function as the void pointer the field holds.

    Args:
        f: The function.

    Returns:
        Its address.
    """
    return Pointer(to=f).unsafe_bitcast[VoidPtr]()[]


def array_release_callback(f: ArrayRelease) -> VoidPtr:
    """Reinterprets an array release function as the void pointer the field holds.

    Args:
        f: The function.

    Returns:
        Its address.
    """
    return Pointer(to=f).unsafe_bitcast[VoidPtr]()[]


def release_schema(mut schema: ArrowSchema):
    """Calls a schema's release callback, if it has one.

    Releasing an already released schema is defined to be a no-op rather than an
    error, because a consumer that has to check first is a consumer that will
    forget to.

    Args:
        schema: The schema to release.
    """
    if not schema.release:
        return
    var slot = schema.release.value()
    var f = Pointer(to=slot).unsafe_bitcast[SchemaRelease]()[]
    f(Pointer(to=schema).unsafe_origin_cast[MutUntrackedOrigin]())


def release_array(mut array: ArrowArray):
    """Calls an array's release callback, if it has one.

    Args:
        array: The array to release.
    """
    if not array.release:
        return
    var slot = array.release.value()
    var f = Pointer(to=slot).unsafe_bitcast[ArrayRelease]()[]
    f(Pointer(to=array).unsafe_origin_cast[MutUntrackedOrigin]())


def format_for(type: LogicalType) raises -> StaticString:
    """Returns the Arrow format string for a firepanda type.

    The two string cases are the view formats rather than the offset ones,
    because firepanda's string columns are the variable size binary view layout
    and a producer that claimed `u` here would be handing out a buffer shaped
    nothing like what a consumer reading `u` expects.

    Args:
        type: The column type.

    Returns:
        The format string, without a null terminator. Callers that need a C
        string add one.

    Raises:
        Error: If the type has no format string, which today means only that it
            is a type this function has not been taught.
    """
    if type == LogicalType.NULL:
        return "n"
    if type == LogicalType.BOOL:
        return "b"
    if type == LogicalType.INT8:
        return "c"
    if type == LogicalType.UINT8:
        return "C"
    if type == LogicalType.INT16:
        return "s"
    if type == LogicalType.UINT16:
        return "S"
    if type == LogicalType.INT32:
        return "i"
    if type == LogicalType.UINT32:
        return "I"
    if type == LogicalType.INT64:
        return "l"
    if type == LogicalType.UINT64:
        return "L"
    if type == LogicalType.FLOAT16:
        return "e"
    if type == LogicalType.FLOAT32:
        return "f"
    if type == LogicalType.FLOAT64:
        return "g"
    if type == LogicalType.STRING:
        return "vu"
    if type == LogicalType.BINARY:
        return "vz"
    raise Error(String("arrow: no format string for ", type))


def type_for_format(format: StringSlice) raises -> LogicalType:
    """Returns the firepanda type for an Arrow format string.

    The inverse of `format_for` over the types firepanda has, and a refusal
    everywhere else. It deliberately refuses the offset based string formats `u`,
    `U`, `z` and `Z` rather than quietly treating them as views, because the two
    layouts are different memory and accepting one as the other is the failure
    this whole file exists to prevent. Reading them is a conversion, and a
    conversion belongs in the import path where it can allocate.

    Args:
        format: The format string.

    Returns:
        The matching type.

    Raises:
        Error: If the format is one firepanda cannot represent.
    """
    if format == "n":
        return LogicalType.NULL
    if format == "b":
        return LogicalType.BOOL
    if format == "c":
        return LogicalType.INT8
    if format == "C":
        return LogicalType.UINT8
    if format == "s":
        return LogicalType.INT16
    if format == "S":
        return LogicalType.UINT16
    if format == "i":
        return LogicalType.INT32
    if format == "I":
        return LogicalType.UINT32
    if format == "l":
        return LogicalType.INT64
    if format == "L":
        return LogicalType.UINT64
    if format == "e":
        return LogicalType.FLOAT16
    if format == "f":
        return LogicalType.FLOAT32
    if format == "g":
        return LogicalType.FLOAT64
    if format == "vu":
        return LogicalType.STRING
    if format == "vz":
        return LogicalType.BINARY
    raise Error(String("arrow: unsupported format string '", format, "'"))


def buffer_count(type: LogicalType) raises -> Int:
    """Returns how many buffers an array of a type has, as the C interface counts.

    A null array has none. A fixed width or boolean array has two, the validity
    bitmap and the values. A string view array has three or more: validity, the
    sixteen byte views, then one buffer per variadic data buffer, then a buffer
    of their sizes. Three is the count when there are no long strings at all, and
    the export path is what knows the real number.

    Args:
        type: The column type.

    Returns:
        The buffer count, which for a view type is the minimum.

    Raises:
        Error: If the type is not one this file knows.
    """
    if type == LogicalType.NULL:
        return 0
    if type == LogicalType.STRING or type == LogicalType.BINARY:
        return 3
    _ = format_for(type)
    return 2
