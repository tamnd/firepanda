"""Writing a frame out as an Arrow IPC stream or file.

The other half of `arrow_ipc.mojo`, and the reason `flatbuf.mojo` has a builder
in it at all. Everything the reader knows how to read, this writes: a schema
message, a record batch per chunk, and for the file format a footer that says
where each batch begins.

Three decisions are worth stating, because none of them is forced.

**Strings are written as views, not as offsets.** A firepanda string column is
already the Arrow variable size binary view layout, byte for byte, and the
exporter in `arrow_export.mojo` relies on that to hand a column to a C consumer
without copying it. Writing `u` instead would mean building an offsets array and
a compacted payload for every string column on the way out, which is a full copy
of the largest buffer in the frame to produce a layout nothing here has. So the
writer emits Utf8View and BinaryView, the reader reads both, and pyarrow has
read them since it learned the types.

**One batch unless the caller asks for more.** A record batch is a unit of
reading, not a unit of correctness, and a reader that wants the whole frame pays
for every extra batch: the batches arrive as separate columns and have to be
stacked. Splitting is still worth having, because a consumer that streams wants
to start work before the last row is written, so `IpcWriteOptions.rows_per_batch`
splits and the default does not. Splitting slices the frame, and slicing a string
column compacts its payload, so a chunked write copies the text and a single
batch write does not.

**Nothing is buffered that does not have to be.** The metadata is a few hundred
bytes and is built in memory. The body is written straight from the column's own
buffers to the file, so writing a 300 MB frame allocates a few kilobytes rather
than a second copy of the frame. `_Sink` is the one place that knows whether the
destination is a file or a list of bytes, and the `_bytes` spellings exist for
tests and for callers holding a socket rather than a path.

### Alignment, which is the only fiddly part

Every message starts on an eight byte boundary and so does every buffer inside a
body, because a reader is going to point a typed pointer at those bytes and an
unaligned load is undefined where it is not merely slow. That is why the file
format writes two bytes of padding after its six byte magic number, and why the
sink tracks how much it has written rather than trusting the caller to count.
Buffer offsets are computed once, when the batch is planned, and checked again
as the body goes out, because an offset in the metadata that disagrees with
where the bytes actually landed is a corrupt file that reads back as garbage
rather than as an error.
"""

from std.collections.span import Span
from std.memory import unsafe_memcpy

from firepanda.array.strview import VIEW_SIZE
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer, round_up
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.lists import dtype_size
from firepanda.frame.frame import DataFrame

from .arrow_export import pack_bools
from .arrow_ipc import (
    BLOCK_SIZE,
    BUFFER_SIZE,
    CONTINUATION,
    MAGIC,
    MESSAGE_RECORD_BATCH,
    MESSAGE_SCHEMA,
    NODE_SIZE,
    TYPE_BINARY_VIEW,
    TYPE_BOOL,
    TYPE_FLOATING_POINT,
    TYPE_INT,
    TYPE_UTF8_VIEW,
)
from .flatbuf import Builder

comptime METADATA_V5 = 4
"""The metadata version every message and the footer carry. V4 and V5 differ in
how a message is framed rather than in the tables, and the framing written here
is the V5 one."""

comptime ALIGNMENT = 8
"""What a message and a buffer are aligned to."""

comptime PRECISION_HALF = 0
comptime PRECISION_SINGLE = 1
comptime PRECISION_DOUBLE = 2


struct IpcWriteOptions(ImplicitlyCopyable, Movable):
    """Everything about a write that is not the frame itself."""

    var rows_per_batch: Int
    """How many rows to put in each record batch. Zero writes one batch."""

    def __init__(out self):
        """Constructs the defaults, which are one batch of everything."""
        self.rows_per_batch = 0

    def __init__(out self, rows_per_batch: Int):
        """Constructs options explicitly.

        Args:
            rows_per_batch: Rows per record batch, or zero for one batch.
        """
        self.rows_per_batch = rows_per_batch


struct _Sink(Movable):
    """Where the bytes go, and how many of them have gone.

    A file when there is a path and a list when the caller wants the bytes, with
    one `put` either way so that nothing above here has to care. The count is
    kept rather than derived, because the list is only there in one of the two
    cases and alignment has to be decided in both.
    """

    var buf: List[UInt8]
    """The bytes, when the destination is memory."""

    var file: Optional[FileHandle]
    """The destination, when it is a file."""

    var written: Int
    """How many bytes have been written so far."""

    def __init__(out self):
        """Constructs a sink that collects the bytes in memory."""
        self.buf = List[UInt8]()
        self.file = None
        self.written = 0

    def __init__(out self, path: String) raises:
        """Constructs a sink that writes to a file, truncating it.

        Args:
            path: The file to write.

        Raises:
            Error: If the file cannot be opened.
        """
        self.buf = List[UInt8]()
        self.file = open(path, "w")
        self.written = 0

    def put(mut self, data: Span[UInt8, _]) raises:
        """Writes bytes.

        Args:
            data: The bytes.

        Raises:
            Error: If the file cannot be written.
        """
        if self.file:
            self.file.value().write_bytes(data)
        else:
            var at = len(self.buf)
            self.buf.resize(at + len(data), 0)
            if len(data) > 0:
                unsafe_memcpy(
                    dest=self.buf.unsafe_ptr().unsafe_offset(at),
                    src=data.unsafe_ptr(),
                    count=len(data),
                )
        self.written += len(data)

    def put_word(mut self, value: UInt32) raises:
        """Writes one little endian four byte word.

        Every number in the framing is a four byte word and none of them is ever
        negative, so the signed and unsigned spellings are the same bytes and
        one of them is enough. They are assembled a byte at a time rather than
        copied out of the value, which costs nothing at five words per file and
        says the byte order rather than inheriting it from the host.

        Args:
            value: The value.

        Raises:
            Error: If the file cannot be written.
        """
        var bytes = List[UInt8](capacity=4)
        for i in range(4):
            bytes.append(UInt8((value >> UInt32(8 * i)) & 0xFF))
        self.put(Span(bytes))

    def pad(mut self) raises:
        """Writes zeros until the next eight byte boundary.

        Raises:
            Error: If the file cannot be written.
        """
        var extra = (-self.written) & (ALIGNMENT - 1)
        if extra == 0:
            return
        var zeros = List[UInt8](length=extra, fill=0)
        self.put(Span(zeros))

    def finish(deinit self) raises -> List[UInt8]:
        """Closes the file, if there is one, and returns whatever was collected.

        Returns:
            The bytes, or an empty list when the destination was a file.

        Raises:
            Error: If the file cannot be closed.
        """
        if self.file:
            self.file.value().close()
        return self.buf^


@fieldwise_init
struct _Buf(ImplicitlyCopyable, Movable):
    """One buffer of one column of one batch, and where it goes in the body."""

    var at: Pointer[UInt8, ImmUntrackedOrigin]
    """The bytes. Untracked because the column outlives the plan by
    construction: nothing between planning a batch and writing it can touch the
    frame."""

    var length: Int
    """How many bytes to write, which is not the capacity of the buffer behind
    the pointer."""

    var offset: Int
    """Where the bytes go, counted from the start of the body."""


struct _Batch(Movable):
    """A record batch worked out but not yet written.

    Arrow puts the layout of a batch in the metadata and the bytes in the body,
    so the offsets have to be known before a single byte of either goes out. This
    is that answer, computed once and then used twice.
    """

    var lengths: List[Int64]
    """The row count of each column, which for a flat schema is the same number
    repeated, and is still per column because Arrow says nodes, not columns."""

    var nulls: List[Int64]
    """The null count of each column."""

    var buffers: List[_Buf]
    """Every buffer of every column, in Arrow's order."""

    var variadic: List[Int64]
    """How many data buffers each view column has, which is always one here."""

    var packed: List[Bitmap]
    """Bit packed bool columns, which are the only bytes this file builds. The
    list is allocated with room for every column up front, because a `_Buf`
    points into it and a reallocation would move what it points at."""

    var body: Int
    """The length of the body, padding included."""

    def __init__(out self, width: Int):
        """Constructs an empty plan for a frame of a given width.

        Args:
            width: The number of columns, used to size the lists that must not
                move.
        """
        self.lengths = List[Int64](capacity=width)
        self.nulls = List[Int64](capacity=width)
        self.buffers = List[_Buf](capacity=3 * width)
        self.variadic = List[Int64](capacity=width)
        self.packed = List[Bitmap](capacity=width)
        self.body = 0


@fieldwise_init
struct _Block(ImplicitlyCopyable, Movable):
    """Where a batch begins in a file, for the footer."""

    var offset: Int
    """The position of the message, counted from the start of the file."""

    var meta: Int
    """The length of the metadata, including the eight byte prefix in front of
    it. That is what other writers put here, so it is what readers expect."""

    var body: Int
    """The length of the body."""


def _type_code(type: LogicalType) raises -> Int:
    """Returns which Arrow type table a firepanda type is described by.

    Args:
        type: The column type.

    Returns:
        The union tag for the type table.

    Raises:
        Error: If the type has no Arrow spelling here yet.
    """
    if type == LogicalType.BOOL:
        return TYPE_BOOL
    if type == LogicalType.STRING:
        return TYPE_UTF8_VIEW
    if type == LogicalType.BINARY:
        return TYPE_BINARY_VIEW
    if (
        type == LogicalType.FLOAT16
        or type == LogicalType.FLOAT32
        or type == LogicalType.FLOAT64
    ):
        return TYPE_FLOATING_POINT
    if (
        type == LogicalType.INT8
        or type == LogicalType.INT16
        or type == LogicalType.INT32
        or type == LogicalType.INT64
        or type == LogicalType.UINT8
        or type == LogicalType.UINT16
        or type == LogicalType.UINT32
        or type == LogicalType.UINT64
    ):
        return TYPE_INT
    raise Error(String("arrow ipc: cannot write a column of type ", type))


def _type_table(mut b: Builder, type: LogicalType) raises -> Int:
    """Builds the Arrow type table for a firepanda type.

    Args:
        b: The builder.
        type: The column type.

    Returns:
        The offset of the table.

    Raises:
        Error: If the type has no Arrow spelling here yet.
    """
    var code = _type_code(type)
    if code == TYPE_INT:
        var signed = (
            type == LogicalType.INT8
            or type == LogicalType.INT16
            or type == LogicalType.INT32
            or type == LogicalType.INT64
        )
        b.start_table(2)
        b.add_scalar[DType.int32](0, Int32(8 * dtype_size(type.physical)), 0)
        b.add_scalar[DType.bool](1, signed, False)
        return b.end_table()
    if code == TYPE_FLOATING_POINT:
        var precision = PRECISION_HALF
        if type == LogicalType.FLOAT32:
            precision = PRECISION_SINGLE
        elif type == LogicalType.FLOAT64:
            precision = PRECISION_DOUBLE
        b.start_table(1)
        b.add_scalar[DType.int16](0, Int16(precision), 0)
        return b.end_table()
    # Bool, Utf8View and BinaryView are tables with no fields at all. The type
    # tag is the whole description, and the table exists so that a later version
    # of the format has somewhere to put a field.
    b.start_table(0)
    return b.end_table()


def _schema_table(mut b: Builder, frame: DataFrame) raises -> Int:
    """Builds the Schema table describing a frame's columns.

    Endianness is left out, which says little endian, because that is the
    default the field carries and every host this builds on is little endian.

    Args:
        b: The builder.
        frame: The frame.

    Returns:
        The offset of the table.

    Raises:
        Error: If a column has a type that cannot be written.
    """
    var fields = List[Int](capacity=frame.width())
    for c in range(frame.width()):
        ref column = frame[c]
        var name = b.create_string(frame.schema.fields[c].name)
        var type = _type_table(b, column.type)
        # A column that holds nulls is written nullable whatever the schema
        # says, because the two disagreeing is a file that describes itself
        # wrongly, and the values are the part a reader cannot argue with.
        var nullable = (
            frame.schema.fields[c].nullable or column.null_count() > 0
        )
        b.start_table(4)
        b.add_offset(0, name)
        b.add_scalar[DType.bool](1, nullable, False)
        b.add_scalar[DType.uint8](2, UInt8(_type_code(column.type)), 0)
        b.add_offset(3, type)
        fields.append(b.end_table())

    var vector = b.create_offsets(fields)
    b.start_table(2)
    b.add_offset(1, vector)
    return b.end_table()


def _schema_message(frame: DataFrame) raises -> List[UInt8]:
    """Builds the schema message that begins every stream and every file.

    Args:
        frame: The frame.

    Returns:
        The FlatBuffer, without the framing in front of it.

    Raises:
        Error: If a column has a type that cannot be written.
    """
    var b = Builder(1024)
    var schema = _schema_table(b, frame)
    b.start_table(4)
    b.add_scalar[DType.int16](0, Int16(METADATA_V5), 0)
    b.add_scalar[DType.uint8](1, UInt8(MESSAGE_SCHEMA), 0)
    b.add_offset(2, schema)
    var message = b.end_table()
    return b.finish(message)


def _bytes_of(imm buffer: Buffer) -> Pointer[UInt8, ImmUntrackedOrigin]:
    """Points at a buffer's bytes, dropping the origin.

    What keeps the bytes alive is the frame, which the caller holds across the
    whole write, and the plan cannot say that in the type system because it is a
    struct rather than a borrow.

    Args:
        buffer: The buffer.

    Returns:
        Its address with an untracked origin.
    """
    return buffer.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()


def _bits_of(imm bits: Bitmap) -> Pointer[UInt8, ImmUntrackedOrigin]:
    """Points at a bitmap's bytes, dropping the origin.

    Args:
        bits: The bitmap.

    Returns:
        Its address with an untracked origin.
    """
    return bits.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()


def _plan(frame: DataFrame) raises -> _Batch:
    """Works out every buffer of a batch and where it goes in the body.

    Args:
        frame: The frame, or the slice of it this batch covers.

    Returns:
        The plan.

    Raises:
        Error: If a column has a type that cannot be written, or is a string
            column with no text storage.
    """
    var out = _Batch(frame.width())
    for c in range(frame.width()):
        ref column = frame[c]
        var rows = len(column)
        var nulls = column.data.validity.null_count()
        out.lengths.append(Int64(rows))
        out.nulls.append(Int64(nulls))

        # The validity buffer, which is written only when there is something in
        # it to say. Arrow reads a zero length buffer here as every value being
        # present, which is the same thing a bitmap of all ones says and is what
        # every other writer emits.
        ref validity = column.data.validity
        var bits = _bits_of(validity)
        if nulls == 0:
            out.buffers.append(_Buf(bits, 0, out.body))
        else:
            out.buffers.append(_Buf(bits, validity.byte_length(), out.body))
        out.body += round_up(
            out.buffers[len(out.buffers) - 1].length, ALIGNMENT
        )

        var code = _type_code(column.type)
        if code == TYPE_BOOL:
            # The one copy in this file. firepanda stores a bool as a byte and
            # Arrow stores it as a bit, so this is the same packing pass the
            # exporter does, and it shrinks the column by a factor of eight on
            # the way out.
            out.packed.append(pack_bools(column))
            ref packed = out.packed[len(out.packed) - 1]
            var length = packed.byte_length() if rows > 0 else 0
            out.buffers.append(_Buf(_bits_of(packed), length, out.body))
            out.body += round_up(length, ALIGNMENT)
        elif code == TYPE_UTF8_VIEW or code == TYPE_BINARY_VIEW:
            if not column.text:
                raise Error(
                    "arrow ipc: a string column with no text storage cannot be"
                    " written"
                )
            ref text = column.text.value()
            out.buffers.append(
                _Buf(_bytes_of(text.views), rows * VIEW_SIZE, out.body)
            )
            out.body += round_up(rows * VIEW_SIZE, ALIGNMENT)
            # One payload block, always written even when it is empty, so that
            # the buffer count of a string column is a constant and the variadic
            # count that describes it is always one.
            var payload = len(text.payload)
            out.buffers.append(_Buf(_bytes_of(text.payload), payload, out.body))
            out.body += round_up(payload, ALIGNMENT)
            out.variadic.append(1)
        else:
            var length = rows * dtype_size(column.type.physical)
            out.buffers.append(
                _Buf(_bytes_of(column.data.values), length, out.body)
            )
            out.body += round_up(length, ALIGNMENT)
    return out^


def _batch_message(batch: _Batch, rows: Int) raises -> List[UInt8]:
    """Builds the record batch message describing a plan.

    Args:
        batch: The plan.
        rows: How many rows the batch holds.

    Returns:
        The FlatBuffer, without the framing in front of it.

    Raises:
        Error: If the builder rejects what it is given, which would be a bug
            here rather than anything about the frame.
    """
    var b = Builder(1024)

    # Both of these are vectors of structs, which FlatBuffers writes inline and
    # in reverse: the last element first, and within an element the last field
    # first, because the builder fills the buffer from the end.
    var count = len(batch.lengths)
    b.start_vector(NODE_SIZE, count, ALIGNMENT)
    for i in range(count - 1, -1, -1):
        b.prepend(batch.nulls[i])
        b.prepend(batch.lengths[i])
    var nodes = b.end_vector(count)

    var total = len(batch.buffers)
    b.start_vector(BUFFER_SIZE, total, ALIGNMENT)
    for i in range(total - 1, -1, -1):
        b.prepend(Int64(batch.buffers[i].length))
        b.prepend(Int64(batch.buffers[i].offset))
    var buffers = b.end_vector(total)

    var variadic = 0
    if len(batch.variadic) > 0:
        var counts = len(batch.variadic)
        b.start_vector(8, counts, ALIGNMENT)
        for i in range(counts - 1, -1, -1):
            b.prepend(batch.variadic[i])
        variadic = b.end_vector(counts)

    b.start_table(5)
    b.add_scalar[DType.int64](0, Int64(rows), 0)
    b.add_offset(1, nodes)
    b.add_offset(2, buffers)
    b.add_offset(4, variadic)
    var record = b.end_table()

    b.start_table(4)
    b.add_scalar[DType.int16](0, Int16(METADATA_V5), 0)
    b.add_scalar[DType.uint8](1, UInt8(MESSAGE_RECORD_BATCH), 0)
    b.add_offset(2, record)
    b.add_scalar[DType.int64](3, Int64(batch.body), 0)
    var message = b.end_table()
    return b.finish(message)


def _footer(frame: DataFrame, blocks: List[_Block]) raises -> List[UInt8]:
    """Builds the footer of a file.

    Args:
        frame: The frame, for the schema the footer repeats.
        blocks: Where each record batch begins.

    Returns:
        The FlatBuffer, which a file carries without any framing in front of it.

    Raises:
        Error: If a column has a type that cannot be written.
    """
    var b = Builder(1024)
    var schema = _schema_table(b, frame)

    # An empty vector rather than a missing field, so that the two block vectors
    # a footer has are written the same way whether or not there is anything in
    # them.
    b.start_vector(BLOCK_SIZE, 0, ALIGNMENT)
    var dictionaries = b.end_vector(0)

    var count = len(blocks)
    b.start_vector(BLOCK_SIZE, count, ALIGNMENT)
    for i in range(count - 1, -1, -1):
        b.prepend(Int64(blocks[i].body))
        # A Block is not sixteen bytes and a multiple of eight by accident: the
        # four byte metadata length is followed by four bytes of padding so that
        # the body length lands aligned.
        b.prepend(Int32(0))
        b.prepend(Int32(blocks[i].meta))
        b.prepend(Int64(blocks[i].offset))
    var batches = b.end_vector(count)

    b.start_table(4)
    b.add_scalar[DType.int16](0, Int16(METADATA_V5), 0)
    b.add_offset(1, schema)
    b.add_offset(2, dictionaries)
    b.add_offset(3, batches)
    var footer = b.end_table()
    return b.finish(footer)


def _put_message(mut sink: _Sink, meta: Span[UInt8, _]) raises -> Int:
    """Writes a message's framing and its metadata.

    Args:
        sink: Where the bytes go. Must be eight byte aligned already.
        meta: The FlatBuffer.

    Returns:
        How many bytes were written, which is what a `Block` records.

    Raises:
        Error: If the destination cannot be written.
    """
    var padded = round_up(len(meta), ALIGNMENT)
    sink.put_word(CONTINUATION)
    sink.put_word(UInt32(padded))
    sink.put(meta)
    sink.pad()
    return padded + 8


def _put_body(mut sink: _Sink, batch: _Batch) raises:
    """Writes a batch's buffers, padded to where the metadata says they are.

    Args:
        sink: Where the bytes go. Must be eight byte aligned already.
        batch: The plan, already written as a message.

    Raises:
        Error: If the destination cannot be written, or if a buffer did not land
            where the metadata said it would, which is a bug in the planner and
            is checked here because the alternative is a file that reads back as
            plausible garbage.
    """
    var start = sink.written
    for i in range(len(batch.buffers)):
        ref buf = batch.buffers[i]
        if sink.written - start != buf.offset:
            raise Error(
                String(
                    "arrow ipc: buffer ",
                    i,
                    " was planned at ",
                    buf.offset,
                    " and landed at ",
                    sink.written - start,
                )
            )
        sink.put(
            Span[UInt8, ImmUntrackedOrigin](
                unsafe_ptr=buf.at, length=buf.length
            )
        )
        sink.pad()
    if sink.written - start != batch.body:
        raise Error(
            String(
                "arrow ipc: a body planned at ",
                batch.body,
                " bytes came out ",
                sink.written - start,
            )
        )


def _put_batch(mut sink: _Sink, frame: DataFrame) raises -> _Block:
    """Writes one record batch, metadata and body.

    Args:
        sink: Where the bytes go.
        frame: The frame, or the slice of it this batch covers.

    Returns:
        Where the batch began and how long its two halves are.

    Raises:
        Error: If a column cannot be written or the destination cannot be.
    """
    var at = sink.written
    var batch = _plan(frame)
    var meta = _batch_message(batch, len(frame))
    var length = _put_message(sink, Span(meta))
    _put_body(sink, batch)
    return _Block(at, length, batch.body)


def _put_end(mut sink: _Sink) raises:
    """Writes the end of stream marker, which is a message of no length.

    Args:
        sink: Where the bytes go.

    Raises:
        Error: If the destination cannot be written.
    """
    sink.put_word(CONTINUATION)
    sink.put_word(0)


def _write(
    mut sink: _Sink, frame: DataFrame, options: IpcWriteOptions, as_file: Bool
) raises:
    """Writes a whole stream or file.

    Args:
        sink: Where the bytes go.
        frame: The frame.
        options: How many rows go in each batch.
        as_file: Whether to wrap the stream in the file format's magic numbers
            and footer.

    Raises:
        Error: If a column cannot be written or the destination cannot be.
    """
    if as_file:
        sink.put(MAGIC.as_bytes())
        # Two bytes, which is what the padding rule works out to after a six
        # byte magic number and is why the format has them.
        sink.pad()

    var schema = _schema_message(frame)
    _ = _put_message(sink, Span(schema))

    var rows = len(frame)
    var step = options.rows_per_batch
    if step <= 0:
        step = rows
    var blocks = List[_Block]()
    if rows > 0 and step >= rows:
        # The whole frame in one batch, and the case worth keeping off the
        # slicing path: a slice copies, and copying the frame to write it would
        # double what a write costs.
        blocks.append(_put_batch(sink, frame))
    else:
        var at = 0
        while at < rows:
            var end = at + step
            if end > rows:
                end = rows
            var part = frame.slice(at, end)
            blocks.append(_put_batch(sink, part))
            at = end

    _put_end(sink)

    if as_file:
        var footer = _footer(frame, blocks)
        sink.put(Span(footer))
        sink.put_word(UInt32(len(footer)))
        sink.put(MAGIC.as_bytes())


def write_ipc_stream_bytes(
    frame: DataFrame, options: IpcWriteOptions
) raises -> List[UInt8]:
    """Writes a frame as an Arrow IPC stream and returns the bytes.

    Args:
        frame: The frame.
        options: How many rows go in each batch.

    Returns:
        The stream.

    Raises:
        Error: If the frame holds a type that cannot be written.
    """
    var sink = _Sink()
    _write(sink, frame, options, False)
    return sink^.finish()


def write_ipc_stream_bytes(frame: DataFrame) raises -> List[UInt8]:
    """Writes a frame as an Arrow IPC stream of one batch.

    Args:
        frame: The frame.

    Returns:
        The stream.

    Raises:
        Error: If the frame holds a type that cannot be written.
    """
    return write_ipc_stream_bytes(frame, IpcWriteOptions())


def write_ipc_file_bytes(
    frame: DataFrame, options: IpcWriteOptions
) raises -> List[UInt8]:
    """Writes a frame as an Arrow IPC file and returns the bytes.

    Args:
        frame: The frame.
        options: How many rows go in each batch.

    Returns:
        The file.

    Raises:
        Error: If the frame holds a type that cannot be written.
    """
    var sink = _Sink()
    _write(sink, frame, options, True)
    return sink^.finish()


def write_ipc_file_bytes(frame: DataFrame) raises -> List[UInt8]:
    """Writes a frame as an Arrow IPC file of one batch.

    Args:
        frame: The frame.

    Returns:
        The file.

    Raises:
        Error: If the frame holds a type that cannot be written.
    """
    return write_ipc_file_bytes(frame, IpcWriteOptions())


def write_ipc_stream(
    frame: DataFrame, path: String, options: IpcWriteOptions
) raises:
    """Writes a frame to disk as an Arrow IPC stream.

    Args:
        frame: The frame.
        path: The file to write.
        options: How many rows go in each batch.

    Raises:
        Error: If the frame holds a type that cannot be written or the file
            cannot be written.
    """
    var sink = _Sink(path)
    _write(sink, frame, options, False)
    _ = sink^.finish()


def write_ipc_stream(frame: DataFrame, path: String) raises:
    """Writes a frame to disk as an Arrow IPC stream of one batch.

    Args:
        frame: The frame.
        path: The file to write.

    Raises:
        Error: If the frame holds a type that cannot be written or the file
            cannot be written.
    """
    write_ipc_stream(frame, path, IpcWriteOptions())


def write_arrow(
    frame: DataFrame, path: String, options: IpcWriteOptions
) raises:
    """Writes a frame to disk as an Arrow IPC file.

    The file format rather than the stream format, because a file on disk is
    the thing people mean by an Arrow file and the footer is what lets a reader
    find a batch without walking everything in front of it. `write_ipc_stream`
    is the other spelling.

    Args:
        frame: The frame.
        path: The file to write.
        options: How many rows go in each batch.

    Raises:
        Error: If the frame holds a type that cannot be written or the file
            cannot be written.
    """
    var sink = _Sink(path)
    _write(sink, frame, options, True)
    _ = sink^.finish()


def write_arrow(frame: DataFrame, path: String) raises:
    """Writes a frame to disk as an Arrow IPC file of one batch.

    Args:
        frame: The frame.
        path: The file to write.

    Raises:
        Error: If the frame holds a type that cannot be written or the file
            cannot be written.
    """
    write_arrow(frame, path, IpcWriteOptions())
