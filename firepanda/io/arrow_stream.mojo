"""Reading a whole table from an Arrow producer, rather than a single column.

`arrow_import.mojo` reads one column. This reads a frame, and the two are
separate files because they answer to different structures: a column is an
`ArrowArray`, and a table is either a struct array with one child per column or
an `ArrowArrayStream` handing out a run of them.

### The stream is the one that matters, which was a surprise

The obvious reading of the PyCapsule protocol is that `__arrow_c_array__` is the
common case and `__arrow_c_stream__` is for the awkward one. It is the other way
round at the table level, and it is worth writing the measurement down because it
decided the shape of this file. Of the objects a user would actually pass, only
`pyarrow.RecordBatch` offers `__arrow_c_array__`. `pyarrow.Table`, `polars`
frames and series and `pandas` frames all offer only `__arrow_c_stream__`.

So an importer that read only arrays would work with pyarrow record batches and
with nothing else anybody has. Both are read here, the stream first, and the
array path is the fallback rather than the main road.

### A stream is many batches and one allocation

A stream hands out a schema once and then a run of arrays that all match it. The
naive read is a frame per batch followed by a concatenate, which writes every
byte twice and does the second write on one thread. `assemble.mojo` already
solves that problem for the IPC and Parquet readers: it asks every array how much
payload it will contribute, adds the answers up, allocates each column once, and
then copies every range in parallel. A stream is exactly the input it wants, so
this file collects the batches, turns each one into a list of per column arrays,
and hands the lot over.

That is why the batches are all pulled before any of them is copied. A stream can
only be walked once and the total row count is not known until it ends, and the
alternative to holding the structures is to give up the single allocation. What is
held is the `ArrowArray` structs, which are a few dozen bytes each, and not
copies of the data they point at.

### Where the offsets are

Arrow lets a producer record a slice either on a struct array or on its children,
and pyarrow puts it on the children, so a consumer that reads one and not the
other reads the wrong rows for somebody. Both are added. The catch is that a
child's own `length` counts from its own offset rather than from the front of its
buffers, so the length check compares against the parent's offset while the read
starts from the sum of the two. Getting that the wrong way round refuses every
sliced table pyarrow produces.

### Who frees what

The same rule the column importer follows, for the same reason: a consumer that
has been handed a structure owns it and owes it a release call, including when it
refuses the contents. Each batch a stream hands out is a structure of its own, so
each one is released, and so is the stream.

When is the interesting part. The windows a batch is turned into own nothing and
point straight at the producer's buffers, so a batch has to outlive `assemble`,
which is where the bytes are finally copied. Releasing a batch as soon as it has
been read would free memory the rest of the assembly is still going to read. So
every batch is held until the frame is built and then all of them are released
together, on the way out and on the way to a raise alike.
"""

from firepanda.array.any import AnyArray
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame

from .arrow_c import (
    STRUCT_FORMAT,
    ArrayPtr,
    ArrowArray,
    ArrowArrayStream,
    ArrowSchema,
    SchemaPtr,
    release_array,
    release_schema,
    release_stream,
    stream_error,
    stream_next,
    stream_schema,
)
from .arrow_import import ColumnSink, format_string
from .assemble import ArrowLayout, assemble

comptime MAX_BATCHES = 1 << 24
"""How many batches a stream may hand out before this gives up on it.

A producer with a bug in its end of stream condition is an infinite loop here,
and an infinite loop that allocates is a machine that stops responding rather
than a program that fails. Sixteen million batches is far past anything real and
far short of that."""


def _name_at(schema: SchemaPtr, i: Int) raises -> String:
    """Names one field of a struct, inventing a name when the producer gave none.

    Arrow allows a field to have no name. Every producer that matters names its
    fields, so the fallback is for schemas built by hand, and inventing `f0` is
    better than refusing a frame over a field label. It is deterministic and it
    cannot collide with another invented name, though it can collide with a real
    one, which `from_series` catches along with every other duplicate.

    Args:
        schema: The child schema.
        i: Its position, which is what the invented name is built from.

    Returns:
        The field name.
    """
    if not schema[].name:
        return String("f", i)
    var p = schema[].name.value()
    var out = String()
    var at = 0
    while True:
        var byte = p.unsafe_offset(at).unsafe_load()
        if byte == 0:
            break
        out += chr(Int(byte))
        at += 1
    return out^


def _child_schema(schema: ArrowSchema, i: Int) raises -> SchemaPtr:
    """Reads one entry of a struct schema's child array.

    Args:
        schema: The parent.
        i: The child index, already checked against `n_children`.

    Returns:
        The pointer to the child.

    Raises:
        Error: If the schema claims children but does not have the array of them.
    """
    if not schema.children:
        raise Error("arrow: struct schema has children but no child array")
    return schema.children.value().unsafe_offset(i)[]


def _child_array(array: ArrowArray, i: Int) raises -> ArrayPtr:
    """Reads one entry of a struct array's child array.

    Args:
        array: The parent.
        i: The child index, already checked against `n_children`.

    Returns:
        The pointer to the child.

    Raises:
        Error: If the array claims children but does not have the array of them.
    """
    if not array.children:
        raise Error("arrow: struct array has children but no child array")
    return array.children.value().unsafe_offset(i)[]


def frame_layout(schema: ArrowSchema) raises -> ArrowLayout:
    """Reads the names, formats and nullability out of a struct schema.

    Everything the frame needs and none of the data, which is what lets the
    layout be read once for a stream and used for every batch in it.

    Args:
        schema: The schema, which must describe a struct.

    Returns:
        The layout, one entry per field, in schema order.

    Raises:
        Error: If the schema is not a struct, which is what a consumer gets when
            it is handed a single column and asked to make a table of it.
    """
    var format = format_string(schema)
    if format != STRUCT_FORMAT:
        raise Error(
            String(
                "arrow: a frame is a struct and this is format '",
                format,
                (
                    "'; pass a table, a record batch or a stream of them rather"
                    " than one column"
                ),
            )
        )
    var out = ArrowLayout()
    for i in range(Int(schema.n_children)):
        var child = _child_schema(schema, i)
        out.names.append(_name_at(child, i))
        out.formats.append(format_string(child[]))
        out.nullable.append(True)
    return out^


def _check_struct(array: ArrowArray, width: Int) raises:
    """Rejects the struct arrays that cannot be a batch of a frame.

    Args:
        array: The struct array.
        width: How many fields the schema declared.

    Raises:
        Error: If the array disagrees with the schema about the field count, if
            its length or offset is negative, or if any row is null.
    """
    if array.length < 0:
        raise Error(String("arrow: negative length ", array.length))
    if array.offset < 0:
        raise Error(String("arrow: negative offset ", array.offset))
    if Int(array.n_children) != width:
        raise Error(
            String(
                "arrow: the schema has ",
                width,
                " fields and a batch has ",
                array.n_children,
                " children",
            )
        )
    if array.null_count == 0 or array.n_buffers == 0:
        return
    if not array.buffers:
        raise Error("arrow: struct array has buffers but no buffer array")
    if array.buffers.value().unsafe_offset(0)[]:
        raise Error(
            "arrow: a struct array with null rows cannot become a frame,"
            " because a frame has rows that are present and columns that are"
            " null, and never the other way round"
        )


def struct_columns(array: ArrowArray, width: Int) raises -> List[ArrowArray]:
    """Narrows every child of a struct array to the rows the parent names.

    See the note at the top of this file on where the offsets are. The arrays
    that come back own nothing: the release callback is cleared so that a copy
    can never release a structure the producer is still holding.

    Args:
        array: The struct array.
        width: How many fields the schema declared.

    Returns:
        One array per field, in schema order, over the producer's own buffers.

    Raises:
        Error: If the array is not a well formed struct, or if a child is too
            short for the rows the parent claims.
    """
    _check_struct(array, width)
    var out = List[ArrowArray](capacity=width)
    for i in range(width):
        var child = _child_array(array, i)
        if array.offset + array.length > child[].length:
            raise Error(
                String(
                    "arrow: field ",
                    i,
                    " has ",
                    child[].length,
                    " rows, which is not enough for ",
                    array.length,
                    " rows starting at ",
                    array.offset,
                )
            )
        var window = child[].copy()
        window.offset = child[].offset + array.offset
        window.length = array.length
        window.release = None
        window.private_data = None
        out.append(window^)
    return out^


def import_frame(
    mut schema: ArrowSchema, mut array: ArrowArray
) raises -> DataFrame:
    """Takes one struct array from a C producer and returns a firepanda frame.

    The mirror of `export_frame_array`, for a producer that offers a single array
    rather than a stream. Everything is copied, for the reasons
    `arrow_import.mojo` gives, so the frame owns all of its memory and nothing in
    it points at the producer.

    Consumes both structures and releases them, on the way out of a failure as
    well as on success, because a producer that has handed a structure over has
    no way to reclaim it.

    Args:
        schema: The schema, which must describe a struct.
        array: The array, holding one child per field.

    Returns:
        The frame, owning all of its memory.

    Raises:
        Error: If the array is not a struct, if a field has a type firepanda
            cannot read, or if two fields have the same name. Both structures are
            released before the error propagates.
    """
    if schema.is_released():
        raise Error("arrow: the schema has already been released")

    var out: DataFrame
    try:
        if array.is_released():
            raise Error("arrow: the array has already been released")
        var layout = frame_layout(schema)
        var batch = struct_columns(array, len(layout))
        var batches = List[List[ArrowArray]]()
        batches.append(batch^)
        out = assemble(layout, batches)
    except error:
        release_array(array)
        release_schema(schema)
        raise error

    release_array(array)
    release_schema(schema)
    return out^


def import_stream(mut stream: ArrowArrayStream) raises -> DataFrame:
    """Drains a stream of struct arrays into one frame.

    This is the path nearly every producer takes, for the reason at the top of
    this file. The batches are all pulled before any is copied, so that every
    column can be allocated once and filled in parallel rather than built per
    batch and concatenated.

    Consumes the stream and releases it, on the way out of a failure as well as
    on success. Releasing a stream releases everything it handed out, so the
    batches are not released one by one.

    Args:
        stream: The stream, which must hand out struct arrays.

    Returns:
        The frame, owning all of its memory.

    Raises:
        Error: If the stream reports an error, if it does not hand out structs,
            or if a field has a type firepanda cannot read. The stream is
            released before the error propagates.
    """
    if stream.is_released():
        raise Error("arrow: the stream has already been released")

    var out: DataFrame
    try:
        out = _drain(stream)
    except error:
        release_stream(stream)
        raise error

    release_stream(stream)
    return out^


def _drain(mut stream: ArrowArrayStream) raises -> DataFrame:
    """Reads a stream to its end and assembles the frame, releasing nothing.

    Split out so that `import_stream` has one release on the failure path and one
    on the success path, rather than a release before every raise in here.

    Args:
        stream: The stream.

    Returns:
        The frame.

    Raises:
        Error: If the stream reports an error or hands out something that is not
            a struct.
    """
    var schema = ArrowSchema()
    var code = stream_schema(stream, schema)
    if code != 0:
        raise Error(
            String(
                "arrow: the stream refused to describe itself: ",
                _why(stream, code),
            )
        )

    var layout: ArrowLayout
    try:
        layout = frame_layout(schema)
    except error:
        release_schema(schema)
        raise error
    release_schema(schema)

    var width = len(layout)
    var handed_out = List[ArrowArray]()
    var batches = List[List[ArrowArray]]()
    var frame: DataFrame
    try:
        while True:
            if len(batches) >= MAX_BATCHES:
                raise Error(
                    String(
                        "arrow: the stream handed out ",
                        MAX_BATCHES,
                        (
                            " batches without ending, which is a producer that"
                            " never stops rather than a table"
                        ),
                    )
                )
            var array = ArrowArray()
            code = stream_next(stream, array)
            if code != 0:
                raise Error(
                    String(
                        "arrow: the stream failed part way: ",
                        _why(stream, code),
                    )
                )
            if array.is_released():
                break
            handed_out.append(array^)
            batches.append(
                struct_columns(handed_out[len(handed_out) - 1], width)
            )

        if len(batches) == 0:
            frame = _empty_frame(layout)
        else:
            frame = assemble(layout, batches)
    except error:
        _release_batches(handed_out)
        raise error

    _release_batches(handed_out)
    return frame^


def _release_batches(mut handed_out: List[ArrowArray]) -> None:
    """Releases every batch the stream handed out.

    It has to happen here rather than in the loop that reads them. A window from
    `struct_columns` points at the producer's buffers and owns none of them, so
    the batch a window came from has to outlive `assemble`, which is where the
    bytes are finally copied. Releasing a batch as soon as it was read would free
    the memory the next batch's assembly is still going to read.

    Args:
        handed_out: Every batch, in the order it arrived.
    """
    for ref array in handed_out:
        release_array(array)


def _empty_frame(layout: ArrowLayout) raises -> DataFrame:
    """Builds the frame a stream that handed out no batches describes.

    It still has a schema, so it still has its columns, and every one of them has
    no rows. This does not go through `assemble` because there is no array to
    give it: a released `ArrowArray` has no buffer count and would be refused as
    malformed, which it is, rather than read as empty.

    Args:
        layout: The names, formats and nullability the stream declared.

    Returns:
        A frame with the right columns and no rows.

    Raises:
        Error: If a column holds a type firepanda cannot read.
    """
    var width = len(layout)
    var fields = List[Field](capacity=width)
    var columns = List[AnyArray](capacity=width)
    for c in range(width):
        var column = ColumnSink(layout.formats[c], 0, 0).finish()
        var field = Field(layout.names[c], column.type)
        field.nullable = layout.nullable[c]
        fields.append(field^)
        columns.append(column^)
    return DataFrame(Schema(fields^), columns^)


def _why(mut stream: ArrowArrayStream, code: Int32) raises -> String:
    """Describes a stream's failure, falling back to the errno it returned.

    Args:
        stream: The stream, immediately after the call that failed.
        code: What that call returned.

    Returns:
        The producer's message, or the code if it offered none.
    """
    var message = stream_error(stream)
    if message.byte_length() != 0:
        return message^
    return String("error ", code)
