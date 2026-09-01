"""Reading Arrow IPC, the format an `.arrow` file is written in.

IPC is what Arrow calls its serialized form, and it is the same bytes whether it
arrives over a socket or out of a file. A stream is a schema message followed by
record batch messages. A file is that same sequence wrapped in a magic number at
each end and followed by a footer listing where every batch begins, so a reader
can take the last batch without walking the first thousand.

A message is a length, a FlatBuffers table saying what follows, and a body. The
table gives the row count, one node per column holding its length and null count,
and one entry per buffer holding an offset into the body and a length. So the
body is the column data laid end to end, and the metadata is the map. Reading it
is arithmetic and one copy per buffer, with nothing to parse in the usual sense.

The copy is the same copy `arrow_import` makes and for the same reasons, which is
why this file makes an `ArrowArray` pointing into the message body and hands it
to `build_column` rather than reading the buffers itself. Everything that file
learned about shifting a validity bitmap, turning offsets into views and checking
a view against the buffer it names applies here unchanged, and the alternative
was a second implementation of all of it that would drift.

What is not read yet is anything nested, dictionary encoded or compressed. A
record batch that says its body is compressed is refused by name rather than by
reading it as raw bytes, because lz4 framed data read as int64 is not an error
that shows up anywhere near here. Dictionary batches are refused for now too:
supporting them means holding a dictionary across messages and rewriting the
codes, and firepanda has no dictionary encoded column to put the result in.

The reader is strict about the parts a file could lie about. Every buffer is
checked against the length of the body it is supposed to sit in, every node
against the row count, and the buffer entries are consumed in the order the
schema says, so a batch with too few of them is caught rather than read into the
next column's data.
"""

from std.collections.span import Span

from firepanda.array.any import AnyArray
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.concat import concat
from firepanda.frame.frame import DataFrame

from .arrow_c import ArrowArray, NullableVoidPtr, VoidPtr
from .arrow_import import build_column
from .flatbuf import (
    field_scalar,
    field_string,
    field_table,
    field_vector,
    read_scalar,
    root_table,
    vector_element,
    vector_length,
    vector_table,
)
from .mapped import map_file

comptime CONTINUATION: UInt32 = 0xFFFFFFFF
"""What a message starts with, since Arrow 0.15. Older streams start with the
length itself, and both are read here because reading the old one is a branch."""

comptime MAGIC = "ARROW1"
"""The six bytes at each end of a file. At the start they are followed by two
bytes of padding, so that the first message begins eight byte aligned."""

comptime NODE_SIZE = 16
"""A FieldNode is a length and a null count, both int64 and both inline."""

comptime BUFFER_SIZE = 16
"""A Buffer is an offset and a length, both int64 and both inline."""

comptime BLOCK_SIZE = 24
"""A Block is an int64 offset, an int32 metadata length, four bytes of padding
and an int64 body length."""

comptime MESSAGE_NONE = 0
comptime MESSAGE_SCHEMA = 1
comptime MESSAGE_DICTIONARY_BATCH = 2
comptime MESSAGE_RECORD_BATCH = 3

comptime TYPE_NULL = 1
comptime TYPE_INT = 2
comptime TYPE_FLOATING_POINT = 3
comptime TYPE_BINARY = 4
comptime TYPE_UTF8 = 5
comptime TYPE_BOOL = 6
comptime TYPE_LARGE_BINARY = 19
comptime TYPE_LARGE_UTF8 = 20
comptime TYPE_BINARY_VIEW = 23
comptime TYPE_UTF8_VIEW = 24


@fieldwise_init
struct IpcMessage(Copyable, Movable):
    """One framed message: where its header is, where its body is, what follows.
    """

    var header_type: Int
    """Which of the `MESSAGE_` kinds this is. `MESSAGE_NONE` ends the stream."""

    var header: Int
    """The position of the header table, or -1 when there is none."""

    var start: Int
    """Where the FlatBuffer begins, which every offset inside it is relative to.
    """

    var body: Int
    """Where the body begins."""

    var body_length: Int
    """How long the body is."""

    var next: Int
    """Where the message after this one begins."""


def _void_at(data: Span[UInt8, _], pos: Int) -> VoidPtr:
    """Returns a C pointer to a position in the buffer.

    The address goes through an integer because the C structure's fields are
    typed mutable, which is what C says when it means nothing in particular, and
    the span this came from may be a read only mapping. Nothing downstream
    writes through it: the importer copies every buffer it is given, which is
    the property that makes handing out the address of a mapped file reasonable
    at all.

    Args:
        data: The buffer.
        pos: The position.

    Returns:
        The address as the pointer type the C structures use.
    """
    return VoidPtr(
        unsafe_from_address=Int(data.unsafe_ptr().unsafe_offset(pos))
    )


def read_message(data: Span[UInt8, _], pos: Int) raises -> IpcMessage:
    """Reads one message header at `pos`.

    Args:
        data: The whole stream or file.
        pos: Where the message begins. Must be eight byte aligned, which every
            position this returns is.

    Returns:
        The message. A `header_type` of `MESSAGE_NONE` means the stream ended,
        either at the end of stream marker or at the end of the bytes.

    Raises:
        Error: If the framing is malformed or a length runs off the end.
    """
    var out = IpcMessage(MESSAGE_NONE, -1, pos, pos, 0, pos)
    if pos + 4 > len(data):
        return out^

    var first = read_scalar[DType.uint32](data, pos)
    var start: Int
    var size: Int
    if first == CONTINUATION:
        if pos + 8 > len(data):
            raise Error(
                "arrow ipc: a continuation marker with no length after it"
            )
        size = Int(read_scalar[DType.int32](data, pos + 4))
        start = pos + 8
    else:
        # Pre 0.15 framing, which put the length first with no marker. Files
        # written that way are still around and cost one branch to read.
        size = Int(first)
        start = pos + 4
    if size == 0:
        return out^
    if size < 0 or start > len(data) - size:
        raise Error(
            String(
                "arrow ipc: a message at ",
                pos,
                " claims ",
                size,
                " bytes of metadata, and only ",
                len(data) - start,
                " are left",
            )
        )

    var message = root_table(data, start)
    var version = Int(field_scalar[DType.int16](data, message, 0, 0))
    if version < 3:
        raise Error(
            String(
                "arrow ipc: metadata version ",
                version,
                " predates V4 and is not read here",
            )
        )
    var body_length = Int(field_scalar[DType.int64](data, message, 3, 0))
    var body = start + size
    if body_length < 0 or body > len(data) - body_length:
        raise Error(
            String(
                "arrow ipc: a message at ",
                pos,
                " claims a body of ",
                body_length,
                " bytes, and only ",
                len(data) - body,
                " are left",
            )
        )

    out.header_type = Int(field_scalar[DType.uint8](data, message, 1, 0))
    out.header = field_table(data, message, 2)
    out.start = start
    out.body = body
    out.body_length = body_length
    out.next = body + body_length
    return out^


def _format_for(
    data: Span[UInt8, _], field: Int, name: String
) raises -> String:
    """Turns one schema field into the format string the importer speaks.

    Args:
        data: The buffer holding the schema message.
        field: The position of the `Field` table.
        name: The column name, for the error message.

    Returns:
        The Arrow C Data Interface format string.

    Raises:
        Error: If the type is one firepanda has no column for.
    """
    var children = field_vector(data, field, 5)
    if children >= 0 and vector_length(data, children) != 0:
        raise Error(
            String(
                "arrow ipc: column '",
                name,
                "' is a nested type, and nested columns are not read yet",
            )
        )
    if field_table(data, field, 4) >= 0:
        raise Error(
            String(
                "arrow ipc: column '",
                name,
                (
                    "' is dictionary encoded, and firepanda has no dictionary"
                    " encoded column to read it into"
                ),
            )
        )

    var kind = Int(field_scalar[DType.uint8](data, field, 2, 0))
    var type = field_table(data, field, 3)
    if kind == TYPE_NULL:
        return "n"
    if kind == TYPE_BOOL:
        return "b"
    if kind == TYPE_BINARY:
        return "z"
    if kind == TYPE_UTF8:
        return "u"
    if kind == TYPE_LARGE_BINARY:
        return "Z"
    if kind == TYPE_LARGE_UTF8:
        return "U"
    if kind == TYPE_BINARY_VIEW:
        return "vz"
    if kind == TYPE_UTF8_VIEW:
        return "vu"
    if type < 0:
        raise Error(
            String(
                "arrow ipc: column '",
                name,
                "' says it has type ",
                kind,
                " but carries no type table",
            )
        )
    if kind == TYPE_INT:
        var bits = Int(field_scalar[DType.int32](data, type, 0, 0))
        var signed = Bool(field_scalar[DType.bool](data, type, 1, False))
        if bits == 8:
            return "c" if signed else "C"
        if bits == 16:
            return "s" if signed else "S"
        if bits == 32:
            return "i" if signed else "I"
        if bits == 64:
            return "l" if signed else "L"
        raise Error(
            String("arrow ipc: column '", name, "' is a ", bits, " bit integer")
        )
    if kind == TYPE_FLOATING_POINT:
        var precision = Int(field_scalar[DType.int16](data, type, 0, 0))
        if precision == 0:
            return "e"
        if precision == 1:
            return "f"
        if precision == 2:
            return "g"
        raise Error(
            String(
                "arrow ipc: column '",
                name,
                "' has floating point precision ",
                precision,
            )
        )
    raise Error(
        String(
            "arrow ipc: column '",
            name,
            "' has Arrow type ",
            kind,
            ", which firepanda cannot read yet",
        )
    )


struct IpcSchema(Copyable, Movable, Sized):
    """The column names and format strings a stream's schema message gave."""

    var names: List[String]
    """The column names, in order."""

    var formats: List[String]
    """One format string per column, in the same order."""

    var nullable: List[Bool]
    """Whether each column was declared nullable."""

    def __init__(out self):
        """Constructs an empty schema."""
        self.names = List[String]()
        self.formats = List[String]()
        self.nullable = List[Bool]()

    def __len__(self) -> Int:
        """Returns the column count.

        Returns:
            How many columns the schema has.
        """
        return len(self.names)


def read_schema(data: Span[UInt8, _], schema: Int) raises -> IpcSchema:
    """Reads a schema message's header table.

    Args:
        data: The buffer holding the message.
        schema: The position of the `Schema` table.

    Returns:
        The names and format strings.

    Raises:
        Error: If a column has a type firepanda cannot read, or the stream is
            big endian.
    """
    if schema < 0:
        raise Error("arrow ipc: a schema message with no schema in it")
    var endianness = Int(field_scalar[DType.int16](data, schema, 0, 0))
    if endianness != 0:
        raise Error(
            "arrow ipc: the stream is big endian, and every buffer would have"
            " to be byte swapped rather than copied"
        )

    var out = IpcSchema()
    var fields = field_vector(data, schema, 1)
    if fields < 0:
        return out^
    for i in range(vector_length(data, fields)):
        var field = vector_table(data, fields, i)
        var name = field_string(data, field, 0)
        out.names.append(name)
        out.formats.append(_format_for(data, field, name))
        out.nullable.append(
            Bool(field_scalar[DType.bool](data, field, 1, True))
        )
    return out^


def _fixed_buffers(format: StringSlice) -> Int:
    """Returns how many buffers a column of this type has, before the variadic
    ones a view column adds.

    Args:
        format: The format string.

    Returns:
        The buffer count.
    """
    if format == "n":
        return 0
    if format == "u" or format == "U" or format == "z" or format == "Z":
        return 3
    return 2


def _read_batch(
    data: Span[UInt8, _], schema: IpcSchema, message: IpcMessage
) raises -> DataFrame:
    """Turns one record batch message into a frame.

    Args:
        data: The buffer holding the message.
        schema: The schema the stream declared.
        message: The framed message, already located.

    Returns:
        A frame of the batch's rows.

    Raises:
        Error: If the batch does not match the schema, or a buffer runs outside
            the body.
    """
    var batch = message.header
    if batch < 0:
        raise Error("arrow ipc: a record batch message with no batch in it")
    if field_table(data, batch, 3) >= 0:
        raise Error(
            "arrow ipc: the body is compressed, and compressed bodies are not"
            " read yet"
        )

    var rows = Int(field_scalar[DType.int64](data, batch, 0, 0))
    if rows < 0:
        raise Error(String("arrow ipc: a batch of ", rows, " rows"))
    var nodes = field_vector(data, batch, 1)
    var buffers = field_vector(data, batch, 2)
    var variadic = field_vector(data, batch, 4)
    var node_count = 0 if nodes < 0 else vector_length(data, nodes)
    if node_count != len(schema):
        raise Error(
            String(
                "arrow ipc: a batch with ",
                node_count,
                " columns against a schema of ",
                len(schema),
            )
        )

    var fields = List[Field](capacity=len(schema))
    var columns = List[AnyArray](capacity=len(schema))
    var taken = 0
    var variadic_taken = 0
    for i in range(len(schema)):
        ref format = schema.formats[i]
        var node = vector_element(data, nodes, i, NODE_SIZE)
        var length = Int(read_scalar[DType.int64](data, node))
        var nulls = Int(read_scalar[DType.int64](data, node + 8))
        if length != rows:
            raise Error(
                String(
                    "arrow ipc: column '",
                    schema.names[i],
                    "' has ",
                    length,
                    " rows in a batch of ",
                    rows,
                )
            )
        if nulls < 0 or nulls > length:
            raise Error(
                String(
                    "arrow ipc: column '",
                    schema.names[i],
                    "' claims ",
                    nulls,
                    " nulls in ",
                    length,
                    " rows",
                )
            )

        var count = _fixed_buffers(format)
        var blocks = 0
        if format == "vu" or format == "vz":
            if variadic < 0:
                raise Error(
                    "arrow ipc: a view column in a batch with no variadic"
                    " buffer counts"
                )
            blocks = Int(
                read_scalar[DType.int64](
                    data, vector_element(data, variadic, variadic_taken, 8)
                )
            )
            variadic_taken += 1
            if blocks < 0:
                raise Error(
                    String("arrow ipc: a view column with ", blocks, " blocks")
                )
            count += blocks

        var pointers = List[NullableVoidPtr](capacity=count + 1)
        var sizes = List[Int64](capacity=blocks + 1)
        for j in range(count):
            var entry = vector_element(data, buffers, taken + j, BUFFER_SIZE)
            var offset = Int(read_scalar[DType.int64](data, entry))
            var size = Int(read_scalar[DType.int64](data, entry + 8))
            if offset < 0 or size < 0 or offset > message.body_length - size:
                raise Error(
                    String(
                        "arrow ipc: buffer ",
                        taken + j,
                        " runs from ",
                        offset,
                        " for ",
                        size,
                        " bytes in a body of ",
                        message.body_length,
                    )
                )
            if j == 0 and size == 0:
                # Arrow writes an entry for the validity bitmap even when there
                # is nothing null, and gives it a length of zero. A null pointer
                # is what the importer reads as all present.
                if nulls != 0:
                    raise Error(
                        String(
                            "arrow ipc: column '",
                            schema.names[i],
                            "' claims ",
                            nulls,
                            " nulls and carries no validity bitmap",
                        )
                    )
                pointers.append(NullableVoidPtr(None))
            else:
                pointers.append(
                    NullableVoidPtr(_void_at(data, message.body + offset))
                )
            if j >= 2:
                sizes.append(Int64(size))
        taken += count

        if blocks > 0:
            # A view array over the C interface ends with a buffer of block
            # lengths. IPC carries those lengths in the buffer entries instead,
            # so they are gathered above and handed over as the buffer the
            # importer expects to find.
            pointers.append(
                NullableVoidPtr(
                    VoidPtr(unsafe_from_address=Int(sizes.unsafe_ptr()))
                )
            )

        var array = ArrowArray()
        array.length = Int64(length)
        array.null_count = Int64(nulls)
        array.offset = 0
        array.n_buffers = Int64(len(pointers))
        array.buffers = pointers.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        var column = build_column(array, format)
        # The buffer array and the block lengths are read through raw pointers,
        # and Mojo would otherwise destroy them at the line that took those
        # pointers rather than at the line that used them.
        _ = pointers^
        _ = sizes^

        var field = Field(schema.names[i], column.type)
        field.nullable = schema.nullable[i]
        fields.append(field^)
        columns.append(column^)

    return DataFrame(Schema(fields^), columns^)


def _empty_frame(data: Span[UInt8, _], schema: IpcSchema) raises -> DataFrame:
    """Builds a frame of no rows with the schema's types.

    A stream may carry a schema and no batches, which is what an empty query
    result looks like, and the answer to that is a frame with the right columns
    rather than a frame with none. The columns are built by the ordinary import
    path with every buffer pointing at the start of the message and a length of
    zero, so there is one construction of a column rather than two.

    Args:
        data: Any buffer at least eight bytes long, used only for an address no
            read will reach.
        schema: The schema.

    Returns:
        An empty frame with the schema's columns.
    """
    var fields = List[Field](capacity=len(schema))
    var columns = List[AnyArray](capacity=len(schema))
    for i in range(len(schema)):
        ref format = schema.formats[i]
        var count = _fixed_buffers(format)
        var view = format == "vu" or format == "vz"
        if view:
            count += 1
        var sizes = List[Int64](length=1, fill=0)
        var pointers = List[NullableVoidPtr](capacity=count + 1)
        pointers.append(NullableVoidPtr(None))
        for _ in range(1, count):
            pointers.append(NullableVoidPtr(_void_at(data, 0)))
        if view:
            pointers.append(
                NullableVoidPtr(
                    VoidPtr(unsafe_from_address=Int(sizes.unsafe_ptr()))
                )
            )

        var array = ArrowArray()
        array.length = 0
        array.null_count = 0
        array.offset = 0
        array.n_buffers = Int64(len(pointers))
        array.buffers = pointers.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        var column = build_column(array, format)
        _ = pointers^
        _ = sizes^

        var field = Field(schema.names[i], column.type)
        field.nullable = schema.nullable[i]
        fields.append(field^)
        columns.append(column^)
    return DataFrame(Schema(fields^), columns^)


def read_ipc_stream(data: Span[UInt8, _]) raises -> DataFrame:
    """Reads an Arrow IPC stream.

    Args:
        data: The whole stream.

    Returns:
        Every batch, concatenated.

    Raises:
        Error: If the stream is malformed or holds a type that is not read yet.
    """
    var message = read_message(data, 0)
    if message.header_type != MESSAGE_SCHEMA:
        raise Error(
            "arrow ipc: a stream begins with a schema message, and this one"
            " does not"
        )
    var schema = read_schema(data, message.header)

    var batches = List[DataFrame]()
    var pos = message.next
    while True:
        var next = read_message(data, pos)
        if next.header_type == MESSAGE_NONE:
            break
        if next.header_type == MESSAGE_DICTIONARY_BATCH:
            raise Error(
                "arrow ipc: the stream carries a dictionary batch, and"
                " dictionary encoded columns are not read yet"
            )
        if next.header_type != MESSAGE_RECORD_BATCH:
            raise Error(
                String(
                    "arrow ipc: a message of kind ",
                    next.header_type,
                    " where a record batch was expected",
                )
            )
        batches.append(_read_batch(data, schema, next))
        pos = next.next

    if len(batches) == 0:
        return _empty_frame(data, schema)
    if len(batches) == 1:
        return batches.pop()
    return concat(batches)


def read_ipc_file(data: Span[UInt8, _]) raises -> DataFrame:
    """Reads an Arrow IPC file.

    The footer is read rather than the messages walked, which is the point of
    the file format: a batch's position is a lookup rather than a traversal of
    everything before it.

    Args:
        data: The whole file.

    Returns:
        Every batch, concatenated.

    Raises:
        Error: If the file is malformed or holds a type that is not read yet.
    """
    var magic = MAGIC.byte_length()
    if len(data) < 2 * magic + 2 + 4:
        raise Error(
            String("arrow ipc: a file of ", len(data), " bytes is too short")
        )
    for i in range(magic):
        if (
            data[i] != MAGIC.as_bytes()[i]
            or data[len(data) - magic + i] != MAGIC.as_bytes()[i]
        ):
            raise Error(
                "arrow ipc: the file does not begin and end with the Arrow"
                " magic number"
            )

    var length_at = len(data) - magic - 4
    var footer_length = Int(read_scalar[DType.int32](data, length_at))
    if footer_length <= 0 or footer_length > length_at - magic - 2:
        raise Error(
            String(
                "arrow ipc: the footer claims to be ",
                footer_length,
                " bytes, which does not fit in the file",
            )
        )
    var footer = root_table(data, length_at - footer_length)
    var schema = read_schema(data, field_table(data, footer, 1))

    var blocks = field_vector(data, footer, 3)
    var count = 0 if blocks < 0 else vector_length(data, blocks)
    if count == 0:
        return _empty_frame(data, schema)

    var batches = List[DataFrame](capacity=count)
    for i in range(count):
        var block = vector_element(data, blocks, i, BLOCK_SIZE)
        var offset = Int(read_scalar[DType.int64](data, block))
        var message = read_message(data, offset)
        if message.header_type != MESSAGE_RECORD_BATCH:
            raise Error(
                String(
                    "arrow ipc: the footer points at a message of kind ",
                    message.header_type,
                    " where a record batch should be",
                )
            )
        batches.append(_read_batch(data, schema, message))

    if len(batches) == 1:
        return batches.pop()
    return concat(batches)


def read_arrow_bytes(data: Span[UInt8, _]) raises -> DataFrame:
    """Reads either an Arrow IPC file or a stream, whichever it is.

    The two are told apart by the magic number, which only the file has. A
    caller usually knows, but the extension is the same and so is what people
    mean by it.

    Args:
        data: The bytes.

    Returns:
        The frame.

    Raises:
        Error: If the bytes are neither.
    """
    var magic = MAGIC.byte_length()
    if len(data) >= magic:
        var looks_like_a_file = True
        for i in range(magic):
            if data[i] != MAGIC.as_bytes()[i]:
                looks_like_a_file = False
                break
        if looks_like_a_file:
            return read_ipc_file(data)
    return read_ipc_stream(data)


def read_arrow(path: String) raises -> DataFrame:
    """Reads an Arrow IPC file or stream from disk.

    The file is mapped rather than read, the same way `read_csv` maps, and for a
    stronger reason: the reader copies each buffer into a firepanda buffer
    exactly once, so mapping means the bytes are touched once on the way in
    rather than copied into a `List` and then copied again.

    Args:
        path: The file to read.

    Returns:
        The frame.

    Raises:
        Error: If the file cannot be read or is not Arrow IPC.
    """
    var mapped = map_file(path)
    if mapped:
        ref file = mapped.value()
        return read_arrow_bytes(file.bytes())
    var handle = open(path, "r")
    var data = handle.read_bytes()
    handle.close()
    return read_arrow_bytes(Span(data))
