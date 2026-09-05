"""Handing firepanda's Arrow exports to Python, as the capsules the protocol wants.

`arrow_export.mojo` produces an `ArrowSchema` and an `ArrowArray`, which are two C
structs. The Arrow PyCapsule interface says how those cross into Python: each one
goes in a `PyCapsule` whose name is exactly `arrow_schema` or `arrow_array`, and
whose destructor releases the struct if the consumer never got round to it. Every
library that speaks this protocol agrees on those two names, and a capsule with
any other name is not recognised, so they are written down once here and nowhere
else.

### The struct has to move to the heap

An `ArrowSchema` returned from `export_schema` is a Mojo local. A capsule holds a
`void*` and outlives the function that made it, so the struct is copied into a
`malloc` allocation and the capsule points at that. The copy is of the struct
itself, which is a handful of pointers, and not of anything it points at.

The allocation is `malloc` rather than Mojo's allocator for the same reason the
boxes in `arrow_export.mojo` are: it is freed from inside a destructor CPython
calls, and C's allocator is the one that is unambiguously safe from there.

### The destructor has to cope with either state

A consumer that takes the data moves the struct out and sets `release` to null,
which is how the C Data Interface says "this one is spent". A consumer that
changes its mind, or a capsule that is simply garbage collected, leaves it
untouched. So the destructor calls `release_schema` or `release_array`, which are
already no-ops on a spent struct, and then frees the allocation either way. Both
paths end with nothing leaked and nothing freed twice.

The destructors swallow their errors, because there is no way to report one from a
`PyCapsule` destructor: CPython calls it while it is already tearing an object
down, and the only thing an exception raised there can do is surface somewhere
unrelated. Nothing in the body raises for any reason other than the capsule being
the wrong kind, which cannot happen because CPython only ever hands each one the
capsule it was installed on.

### Reading somebody else's capsule is the same handshake backwards

`take_schema` and `take_array` are the other direction, and they are what makes
`firepanda.from_arrow` work on a pyarrow table or a Polars frame. The protocol
says a consumer takes ownership by moving the struct out of the capsule and
leaving the released state behind, so that is what they do: copy the struct, then
overwrite the producer's copy with the all zero state that `release` being null
means. After that the capsule's own destructor finds a spent struct and does
nothing, and the release is owed by this side.

Overwriting rather than merely reading matters. A capsule read but not blanked
would be released twice, once by whoever we hand the struct to and once by the
producer's destructor when Python collects the capsule, and the second release
runs a callback over memory the first one freed.
"""

from std.ffi import external_call
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.sys import size_of

from firepanda.io.arrow_c import (
    ArrayPtr,
    ArrowArray,
    ArrowArrayStream,
    ArrowSchema,
    SchemaPtr,
    release_array,
    release_schema,
)

comptime SCHEMA_CAPSULE = "arrow_schema"
"""The capsule name the Arrow PyCapsule interface reserves for a schema."""

comptime ARRAY_CAPSULE = "arrow_array"
"""The capsule name the Arrow PyCapsule interface reserves for an array."""

comptime STREAM_CAPSULE = "arrow_array_stream"
"""The capsule name the Arrow PyCapsule interface reserves for a stream."""


def _drop_schema(capsule: PyObjectPtr) abi("C") -> None:
    """Releases and frees a schema capsule's struct. Called by CPython."""
    try:
        ref cpython = Python().cpython()
        var schema = cpython.PyCapsule_GetPointer(
            capsule, String(SCHEMA_CAPSULE)
        ).unsafe_bitcast[ArrowSchema]()
        release_schema(schema[])
        external_call["free", NoneType](schema)
    except:
        pass


def _drop_array(capsule: PyObjectPtr) abi("C") -> None:
    """Releases and frees an array capsule's struct. Called by CPython."""
    try:
        ref cpython = Python().cpython()
        var array = cpython.PyCapsule_GetPointer(
            capsule, String(ARRAY_CAPSULE)
        ).unsafe_bitcast[ArrowArray]()
        release_array(array[])
        external_call["free", NoneType](array)
    except:
        pass


def schema_capsule(var schema: ArrowSchema) raises -> PythonObject:
    """Wraps an exported schema in the capsule the protocol expects.

    Args:
        schema: The schema, consumed. Ownership of it passes to the capsule and
            from there to whoever holds the capsule.

    Returns:
        A `PyCapsule` named `arrow_schema`.
    """
    var box = external_call["malloc", SchemaPtr](size_of[ArrowSchema]())
    box.unsafe_write(schema^)
    return PythonObject(
        from_owned=Python()
        .cpython()
        .PyCapsule_New(
            box.unsafe_bitcast[NoneType](), SCHEMA_CAPSULE, _drop_schema
        )
    )


def array_capsule(var array: ArrowArray) raises -> PythonObject:
    """Wraps an exported array in the capsule the protocol expects.

    Args:
        array: The array, consumed. Ownership of it passes to the capsule and
            from there to whoever holds the capsule.

    Returns:
        A `PyCapsule` named `arrow_array`.
    """
    var box = external_call["malloc", ArrayPtr](size_of[ArrowArray]())
    box.unsafe_write(array^)
    return PythonObject(
        from_owned=Python()
        .cpython()
        .PyCapsule_New(
            box.unsafe_bitcast[NoneType](), ARRAY_CAPSULE, _drop_array
        )
    )


def take_schema(capsule: PythonObject) raises -> ArrowSchema:
    """Takes ownership of the schema inside somebody else's capsule.

    See the note above on what taking means. The struct is copied out and the
    producer's copy is overwritten with the released state, so the capsule's own
    destructor finds nothing to do and this side owes the release.

    Args:
        capsule: A `PyCapsule` named `arrow_schema`.

    Returns:
        The schema, which the caller must release.

    Raises:
        Error: If the object is not a capsule of that name.
    """
    var box = (
        Python()
        .cpython()
        .PyCapsule_GetPointer(capsule._obj_ptr, String(SCHEMA_CAPSULE))
        .unsafe_bitcast[ArrowSchema]()
    )
    var out = box[].copy()
    box[] = ArrowSchema()
    return out^


def take_array(capsule: PythonObject) raises -> ArrowArray:
    """Takes ownership of the array inside somebody else's capsule.

    Args:
        capsule: A `PyCapsule` named `arrow_array`.

    Returns:
        The array, which the caller must release.

    Raises:
        Error: If the object is not a capsule of that name.
    """
    var box = (
        Python()
        .cpython()
        .PyCapsule_GetPointer(capsule._obj_ptr, String(ARRAY_CAPSULE))
        .unsafe_bitcast[ArrowArray]()
    )
    var out = box[].copy()
    box[] = ArrowArray()
    return out^


def take_stream(capsule: PythonObject) raises -> ArrowArrayStream:
    """Takes ownership of the stream inside somebody else's capsule.

    Args:
        capsule: A `PyCapsule` named `arrow_array_stream`.

    Returns:
        The stream, which the caller must release.

    Raises:
        Error: If the object is not a capsule of that name.
    """
    var box = (
        Python()
        .cpython()
        .PyCapsule_GetPointer(capsule._obj_ptr, String(STREAM_CAPSULE))
        .unsafe_bitcast[ArrowArrayStream]()
    )
    var out = box[].copy()
    box[] = ArrowArrayStream()
    return out^
