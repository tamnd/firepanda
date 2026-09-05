"""Taking a column from a C producer and making it a firepanda column.

The export half hands out pointers. This half cannot. It copies, and since that
is the opposite of what the file next door is for, the reason belongs at the top
rather than buried in a function.

### Why the import copies

Three separate reasons, any one of which would be enough on its own.

A firepanda `Buffer` is 64-byte aligned and allocated in whole 64-byte blocks,
and every kernel in the engine is written against that: a loop may read one full
SIMD register past the last value it cares about, because the bytes are there and
they are zero. Arrow requires an 8-byte alignment and recommends 64, and it says
nothing at all about what follows the last byte. A borrowed foreign buffer would
therefore be a buffer that most kernels may not touch, which is not a buffer.

An Arrow array carries an `offset`, so what arrives is often a slice of somebody
else's column. firepanda columns start at element zero. Honouring an offset means
shifting the validity bits, which is a rewrite whatever else happens.

And a foreign string column is not shaped like ours even when it is a view array.
Arrow allows any number of variadic data buffers; a firepanda column has exactly
one. Every long element's view names a buffer index and that index has to be
rewritten, so the views are rebuilt no matter what.

The last one is worth sitting with, because it means the zero copy that made the
export interesting was never available here. What is available is a copy that
happens once, at memory bandwidth, at the boundary, instead of the format
conversion per element that reading someone else's file would otherwise cost.

### The offset based string formats

`u`, `U`, `z` and `Z` are accepted here, which is the promise `type_for_format`
made when it refused them. Those are the layouts where each element is a slice of
one data buffer named by a pair of offsets, and they matter far more than the
symmetry with our export does: `u` is what pyarrow produces by default, so an
import that took only `vu` would be an import that works with almost nothing.
Reading one is a walk that appends each element to a `StringBuilder`, which is
the same path the CSV reader takes and gets short string inlining for free.

### Who frees what

`import_array` takes ownership, which is what the C interface says a consumer
does, and releases both structures before it returns. It releases them when it
raises, too. A producer that has handed a structure over has no way to reclaim
it, so a consumer that refuses the type still owes it the release call, and a
refusal that leaked would make every unsupported column a slow leak in a loop
that is probably reading a directory of files.

Because everything is copied, the release happens immediately rather than being
deferred to the lifetime of the column that came out. Nothing firepanda returns
points into the producer's memory.
"""

from std.memory import unsafe_memcpy

from firepanda.array.any import AnyArray
from firepanda.array.data import ColumnData
from firepanda.array.strings import StringArray
from firepanda.array.strview import (
    INLINE_CAPACITY,
    VIEW_SIZE,
    StringView,
    make_inline_at,
    make_long_at,
)
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import dtype_size
from firepanda.dtype.logical import LogicalType

from .arrow_c import (
    ArrowArray,
    ArrowSchema,
    NullableVoidPtr,
    release_array,
    release_schema,
    type_for_format,
)


def format_string(schema: ArrowSchema) raises -> String:
    """Reads a schema's format string into memory firepanda owns.

    Copied rather than borrowed because the schema is released before the
    imported column is returned, and the format string dies with it.

    Public because `arrow_stream.mojo` reads the format string of every child of
    a struct schema before any of them reaches this file.

    Args:
        schema: The schema.

    Returns:
        The format string.

    Raises:
        Error: If the schema has no format string, which makes it not a schema.
    """
    if not schema.format:
        raise Error("arrow: schema has no format string")
    var p = schema.format.value()
    var out = String()
    var i = 0
    while True:
        var byte = p.unsafe_offset(i).unsafe_load()
        if byte == 0:
            break
        out += chr(Int(byte))
        i += 1
    return out^


def _buffer_at(array: ArrowArray, i: Int) raises -> NullableVoidPtr:
    """Reads one entry of an imported array's buffer array.

    Args:
        array: The array.
        i: The buffer index, which the caller has already checked against
            `n_buffers`.

    Returns:
        The pointer, which may be null.

    Raises:
        Error: If the array claims buffers but does not have the array of them.
    """
    if not array.buffers:
        raise Error("arrow: array has buffers but no buffer array")
    return array.buffers.value().unsafe_offset(i)[]


def _bytes_at(
    array: ArrowArray, i: Int
) raises -> Pointer[UInt8, MutUntrackedOrigin]:
    """Reads one entry of the buffer array as bytes, refusing a null.

    Args:
        array: The array.
        i: The buffer index.

    Returns:
        The pointer.

    Raises:
        Error: If the buffer is null, which for a values buffer means the
            producer built the structure wrong rather than that the column is
            empty.
    """
    var slot = _buffer_at(array, i)
    if not slot:
        raise Error(String("arrow: buffer ", i, " is null and must not be"))
    return slot.value().unsafe_bitcast[UInt8]()


def _sizes_at(
    array: ArrowArray, variadic: Int
) raises -> Pointer[Int64, MutUntrackedOrigin]:
    """Reads the lengths of a view array's data buffers.

    That list is what makes it possible to check a view before following it. A
    view is three numbers a stranger chose, and following one without checking is
    an out of bounds read waiting for a malformed file.

    A producer with no data buffers is allowed to leave the list null, and Polars
    does, so the pointer returned in that case is the views buffer instead. It is
    never read: with no data buffers no view can legally be long, so every caller
    raises on the buffer index before it reaches the length. Mojo's `Pointer` has
    no null to return instead.

    Args:
        array: The array.
        variadic: How many data buffers it has, which the caller has computed.

    Returns:
        One length per data buffer.

    Raises:
        Error: If the array has data buffers but no list of their lengths.
    """
    if variadic == 0:
        return _bytes_at(array, 1).unsafe_bitcast[Int64]()
    return _bytes_at(array, Int(array.n_buffers) - 1).unsafe_bitcast[Int64]()


def _import_validity(array: ArrowArray, length: Int) raises -> Bitmap:
    """Builds a firepanda validity bitmap from Arrow's.

    The two conventions agree bit for bit, so with no offset this is a byte copy
    and a tail mask. With an offset it is a shift, and the bit at a time loop is
    kept for that case rather than a word shifted one, because a nonzero offset
    means the producer handed over a slice and a slice is not the shape a bulk
    reader arrives in.

    Args:
        array: The array, for its validity buffer and offset.
        length: The number of rows.

    Returns:
        A bitmap with one meaning present, starting at row zero.

    Raises:
        Error: If the buffer array is missing.
    """
    if array.null_count == 0:
        return Bitmap(length)
    var slot = _buffer_at(array, 0)
    if not slot:
        # Arrow allows a null validity buffer, and then nothing is null however
        # the null count reads. A count of minus one means unknown, which is
        # also nothing null once the buffer is absent.
        return Bitmap(length)

    var bits = slot.value().unsafe_bitcast[UInt8]()
    var offset = Int(array.offset)
    var out = Bitmap(length, all_valid=False)

    if offset % 8 == 0:
        var source = bits.unsafe_offset(offset // 8)
        var nbytes = out.byte_length()
        unsafe_memcpy(dest=out.unsafe_ptr(), src=source, count=nbytes)
        # The last byte may carry bits belonging to rows past the end of this
        # slice, and `null_count` counts whole bytes, so they are cleared here
        # rather than left to be counted as present values that do not exist.
        var used = length & 7
        if used != 0:
            var last = nbytes - 1
            var slot_ptr = out.unsafe_ptr().unsafe_offset(last)
            slot_ptr.unsafe_write(
                slot_ptr.unsafe_load() & UInt8((1 << used) - 1)
            )
        return out^

    for i in range(length):
        var bit = offset + i
        var byte = bits.unsafe_offset(bit >> 3).unsafe_load()
        if ((byte >> UInt8(bit & 7)) & 1) != 0:
            out.set(i, True)
    return out^


def _import_fixed(
    array: ArrowArray, type: LogicalType, length: Int
) raises -> AnyArray:
    """Copies a fixed width values buffer.

    Args:
        array: The array.
        type: The column type.
        length: The number of rows.

    Returns:
        The column.

    Raises:
        Error: If the values buffer is missing.
    """
    var width = dtype_size(type.physical)
    var values = Buffer(overwritten=length * width)
    if length > 0:
        unsafe_memcpy(
            dest=values.unsafe_ptr(),
            src=_bytes_at(array, 1).unsafe_offset(Int(array.offset) * width),
            count=length * width,
        )
    var validity = _import_validity(array, length)
    return AnyArray(ColumnData(values^, validity^, length), type)


def _import_bool(array: ArrowArray, length: Int) raises -> AnyArray:
    """Unpacks Arrow's bit per value into firepanda's byte per value.

    The mirror of the packing the export does, and the same trade: a kernel wants
    to load a bool without shifting for it.

    Args:
        array: The array.
        length: The number of rows.

    Returns:
        The column.

    Raises:
        Error: If the values buffer is missing.
    """
    var values = Buffer(overwritten=length)
    if length > 0:
        var bits = _bytes_at(array, 1)
        var offset = Int(array.offset)
        var out = values.unsafe_ptr()
        for i in range(length):
            var bit = offset + i
            var byte = bits.unsafe_offset(bit >> 3).unsafe_load()
            out.unsafe_offset(i).unsafe_write((byte >> UInt8(bit & 7)) & 1)
    var validity = _import_validity(array, length)
    return AnyArray(ColumnData(values^, validity^, length), LogicalType.BOOL)


def _import_views(
    array: ArrowArray, type: LogicalType, length: Int
) raises -> AnyArray:
    """Rebuilds a view column so that every long element names one payload.

    A short element's view is already exactly ours and is copied through as
    sixteen bytes. A long one is not, because it names a data buffer by index and
    the producer may have any number of them where we have one, so its bytes are
    moved into the one payload and its view is rewritten to say where they
    landed.

    Written as two passes rather than through a `StringBuilder`. The first pass
    adds up how much payload the long elements need, so the second allocates once
    and writes each view straight into place. A builder would double its payload
    as it went and then copy every view a second time on the way out of `finish`,
    which on a ten million row column is a second pass over a hundred and sixty
    megabytes for no reason: the element count is known before the first byte is
    read, which is the thing a builder cannot assume and this can.

    There is a case where even that is too much work. A producer with exactly one
    data buffer, no offset and no nulls has handed over a column already shaped
    the way ours is, and then the views and the payload are each one memcpy. The
    first pass still runs, because it is also where every view is checked against
    the length of the buffer it names, and a view is three numbers a stranger
    chose.

    Args:
        array: The array.
        type: The column type, string or binary.
        length: The number of rows.

    Returns:
        The column.

    Raises:
        Error: If a view names a data buffer the array does not have.
    """
    var variadic = Int(array.n_buffers) - 3
    var validity = _import_validity(array, length)
    var source = (
        _bytes_at(array, 1)
        .unsafe_bitcast[StringView]()
        .unsafe_offset(Int(array.offset))
    )

    var sizes = _sizes_at(array, variadic)

    var needed = 0
    for i in range(length):
        if not validity.get(i):
            continue
        ref view = source.unsafe_offset(i)[]
        if view.is_inline():
            continue
        var block = view.block()
        if block >= variadic:
            raise Error(
                String(
                    "arrow: view at row ",
                    i,
                    " names data buffer ",
                    block,
                    " but the array has ",
                    variadic,
                )
            )
        var count = len(view)
        if (
            view.offset() < 0
            or Int64(view.offset() + count)
            > sizes.unsafe_offset(block).unsafe_load()
        ):
            raise Error(
                String(
                    "arrow: view at row ",
                    i,
                    " runs past the end of data buffer ",
                    block,
                )
            )
        needed += count

    var views = Buffer(length * VIEW_SIZE)
    var target = views.unsafe_ptr().unsafe_bitcast[StringView]()

    if variadic == 1 and array.offset == 0:
        # Every view already names buffer zero and every offset is already
        # relative to it, so nothing needs rewriting and the two buffers are one
        # memcpy each. The whole block is taken rather than the bytes the views
        # actually reach, because a producer is allowed to leave gaps in it and
        # closing them would mean rewriting the offsets, which is the work this
        # path exists to skip.
        var total = Int(sizes.unsafe_load())
        var payload = Buffer(overwritten=total)
        if length > 0:
            unsafe_memcpy(
                dest=views.unsafe_ptr(),
                src=_bytes_at(array, 1),
                count=length * VIEW_SIZE,
            )
        if total > 0:
            unsafe_memcpy(
                dest=payload.unsafe_ptr(),
                src=_bytes_at(array, 2),
                count=total,
            )
        # A null element's view is whatever the producer left there, which the
        # pass above deliberately did not check and which nothing downstream
        # should follow. Overwriting them with the view of the empty string is
        # what the rest of the library expects of a null.
        if array.null_count != 0:
            for i in range(length):
                if not validity.get(i):
                    target.unsafe_offset(i)[] = StringView()
        return _as_text(views^, payload^, validity^, length, type)

    var payload = Buffer(overwritten=needed)
    var written = 0
    for i in range(length):
        if not validity.get(i):
            target.unsafe_offset(i)[] = StringView()
            continue
        ref view = source.unsafe_offset(i)[]
        if view.is_inline():
            target.unsafe_offset(i)[] = view
            continue
        var count = len(view)
        var dest = payload.unsafe_ptr().unsafe_offset(written)
        unsafe_memcpy(
            dest=dest,
            src=_bytes_at(array, 2 + view.block()).unsafe_offset(view.offset()),
            count=count,
        )
        target.unsafe_offset(i)[] = make_long_at(dest, count, 0, written)
        written += count

    return _as_text(views^, payload^, validity^, length, type)


def _import_offsets[
    index: DType
](array: ArrowArray, type: LogicalType, length: Int) raises -> AnyArray:
    """Reads the offset based layout, where an element is a slice of one buffer.

    This is what pyarrow produces unless it is asked for views, so it is the
    import that will get the most use even though firepanda never writes it.

    Two passes, for the reason given in `_import_views`: the offsets say how long
    every element is before any of them are read, so the payload is sized once
    and each view is written straight into its place.

    An element that fits in twelve bytes never reaches the payload at all, which
    is the whole point of our layout and is why this is worth doing rather than
    keeping the data buffer as it arrived. A column of country codes arrives as a
    data buffer plus an offset per element and leaves as views alone.

    Parameters:
        index: The offset type, int32 for `u` and `z` and int64 for `U` and `Z`.

    Args:
        array: The array.
        type: The column type, string or binary.
        length: The number of rows.

    Returns:
        The column.

    Raises:
        Error: If a buffer is missing.
    """
    var validity = _import_validity(array, length)
    var views = Buffer(length * VIEW_SIZE)
    if length == 0:
        return _as_text(views^, Buffer(0), validity^, length, type)

    var target = views.unsafe_ptr().unsafe_bitcast[StringView]()
    var offsets = (
        _bytes_at(array, 1)
        .unsafe_bitcast[Scalar[index]]()
        .unsafe_offset(Int(array.offset))
    )
    var data = _bytes_at(array, 2)

    # There is no separate length for the data buffer in this layout: the last
    # offset is its length by definition. So the only thing that can be checked
    # is that the offsets go forwards, and it has to be checked, because a
    # backwards pair is a negative element length and a negative length passed to
    # a memcpy is an enormous one.
    var needed = 0
    for i in range(length):
        if not validity.get(i):
            continue
        var count = Int(
            offsets.unsafe_offset(i + 1).unsafe_load()
            - offsets.unsafe_offset(i).unsafe_load()
        )
        if count < 0:
            raise Error(String("arrow: offsets go backwards at row ", i))
        if count > INLINE_CAPACITY:
            needed += count

    var payload = Buffer(overwritten=needed)
    var written = 0
    for i in range(length):
        if not validity.get(i):
            target.unsafe_offset(i)[] = StringView()
            continue
        var start = Int(offsets.unsafe_offset(i).unsafe_load())
        var count = Int(offsets.unsafe_offset(i + 1).unsafe_load()) - start
        var src = data.unsafe_offset(start)
        if count <= INLINE_CAPACITY:
            target.unsafe_offset(i)[] = make_inline_at(src, count)
            continue
        var dest = payload.unsafe_ptr().unsafe_offset(written)
        unsafe_memcpy(dest=dest, src=src, count=count)
        target.unsafe_offset(i)[] = make_long_at(dest, count, 0, written)
        written += count

    return _as_text(views^, payload^, validity^, length, type)


def _as_text(
    var views: Buffer,
    var payload: Buffer,
    var validity: Bitmap,
    length: Int,
    type: LogicalType,
) -> AnyArray:
    """Wraps finished string storage as a column of the given type.

    String and binary share a representation and differ only in what the type
    says the bytes mean, so the type is set rather than the storage rebuilt.

    Args:
        views: The views buffer.
        payload: The payload buffer.
        validity: The validity bitmap.
        length: The number of rows.
        type: String or binary.

    Returns:
        The column.
    """
    var out = AnyArray(StringArray(views^, payload^, validity^, length))
    out.type = type
    return out^


def _build(
    array: ArrowArray, format: StringSlice, length: Int
) raises -> AnyArray:
    """Dispatches on the format string.

    Args:
        array: The array.
        format: The schema's format string.
        length: The number of rows.

    Returns:
        The column.

    Raises:
        Error: If the format is one firepanda cannot represent.
    """
    if format == "u" or format == "z":
        return _import_offsets[DType.int32](
            array,
            LogicalType.STRING if format == "u" else LogicalType.BINARY,
            length,
        )
    if format == "U" or format == "Z":
        return _import_offsets[DType.int64](
            array,
            LogicalType.STRING if format == "U" else LogicalType.BINARY,
            length,
        )

    var type = type_for_format(format)
    if type == LogicalType.NULL:
        raise Error(
            "arrow: cannot import a null column, because firepanda has no"
            " column that carries the null type at run time"
        )
    if type == LogicalType.BOOL:
        return _import_bool(array, length)
    if type == LogicalType.STRING or type == LogicalType.BINARY:
        return _import_views(array, type, length)
    return _import_fixed(array, type, length)


def _check(array: ArrowArray, format: StringSlice) raises:
    """Rejects the shapes this file cannot read, before anything is allocated.

    Args:
        array: The array.
        format: The schema's format string.

    Raises:
        Error: If the array is nested, dictionary encoded, or has a buffer count
            that does not match its type.
    """
    if array.length < 0:
        raise Error(String("arrow: negative length ", array.length))
    if array.offset < 0:
        raise Error(String("arrow: negative offset ", array.offset))
    if array.n_children != 0:
        raise Error(
            "arrow: nested columns are not supported yet, and a child cannot be"
            " read as anything else"
        )
    if array.dictionary:
        raise Error("arrow: dictionary encoded columns are not supported yet")

    var expected: Int64
    if format == "vu" or format == "vz":
        # Validity, views and the sizes, plus a data buffer per variadic block. A
        # producer may have any number of those and is allowed to have none, which
        # is what Polars hands over for a column whose every string is short
        # enough to sit inside its view. So three is the floor rather than four,
        # and a column that arrives with three buffers is one that cannot legally
        # contain a long view.
        if array.n_buffers < 3:
            raise Error(
                String(
                    "arrow: a view column needs at least three buffers, got ",
                    array.n_buffers,
                )
            )
        return
    if format == "u" or format == "U" or format == "z" or format == "Z":
        expected = 3
    elif format == "n":
        expected = 0
    else:
        expected = 2
    if array.n_buffers != expected:
        raise Error(
            String(
                "arrow: format ",
                format,
                " needs ",
                expected,
                " buffers, got ",
                array.n_buffers,
            )
        )


def build_column(array: ArrowArray, format: StringSlice) raises -> AnyArray:
    """Builds a firepanda column out of an Arrow array, checking it first.

    This is `import_array` without the ownership half, for a caller that did not
    get its buffers from a C producer and so has nothing to release. The Arrow
    IPC reader is that caller: it has a message body and a map of where each
    buffer sits inside it, which is an `ArrowArray` in everything but where it
    came from, and everything this file knows about shifting a validity bitmap
    or turning offsets into views applies to it unchanged.

    Args:
        array: The array. Nothing is released and nothing is retained, so the
            buffers only have to outlive the call.
        format: The Arrow C Data Interface format string for the type.

    Returns:
        The column, owning all of its memory.

    Raises:
        Error: If the type or shape is one firepanda cannot read.
    """
    _check(array, format)
    return _build(array, format, Int(array.length))


def _type_for(format: StringSlice) raises -> LogicalType:
    """Maps a format string to the column type the import produces.

    The offset based string formats are the reason this is not just
    `type_for_format`: firepanda has no offset based column, so `u` and `U`
    become a string column and `z` and `Z` a binary one, which is a decision this
    file makes rather than one the format string states.

    Args:
        format: The format string.

    Returns:
        The type.

    Raises:
        Error: If the format is one firepanda cannot represent.
    """
    if format == "u" or format == "U":
        return LogicalType.STRING
    if format == "z" or format == "Z":
        return LogicalType.BINARY
    var type = type_for_format(format)
    if type == LogicalType.NULL:
        raise Error(
            "arrow: cannot import a null column, because firepanda has no"
            " column that carries the null type at run time"
        )
    return type


struct ColumnSink(Movable):
    """A column allocated once and filled by several arrays at once.

    `build_column` answers the question a single array asks. This answers the one
    a file asks, which is different: a reader holding twenty record batches knows
    the finished column's length before it copies a byte, and building twenty
    columns and concatenating them means every byte is written twice and the
    second write is single file through one thread.

    So the reader plans first. It asks each array how much payload it needs, adds
    the answers up, allocates each column once, and then hands every array the
    place its rows go. The fills touch disjoint ranges, so they run on every core,
    and there is nothing to concatenate at the end.

    Validity is the one part that does not fill in place. A batch boundary lands
    wherever the writer put it, so two batches share a byte of the bitmap and two
    threads writing it would race over the six bits neither of them owns. Each
    fill returns its own bitmap and the caller pastes them in order, which is a
    pass over one bit per row and does not show up in a measurement.
    """

    var type: LogicalType
    """The column type."""

    var values: Buffer
    """Values for a fixed width column, a byte per value for a bool one, and the
    views for a string or binary one."""

    var payload: Buffer
    """The string payload, and empty for every other type."""

    var validity: Bitmap
    """One bit per row, one meaning present, filled by the caller."""

    var rows: Int
    """The finished length."""

    def __init__(out self, format: StringSlice, rows: Int, payload: Int) raises:
        """Allocates a column of `rows` rows with room for `payload` bytes.

        Args:
            format: The format string the arrays will arrive in.
            rows: The finished length.
            payload: The total payload bytes, from `payload_for`.

        Raises:
            Error: If the format is one firepanda cannot represent.
        """
        self.type = _type_for(format)
        self.rows = rows
        self.validity = Bitmap(rows)
        if self.type == LogicalType.STRING or self.type == LogicalType.BINARY:
            self.values = Buffer(rows * VIEW_SIZE)
            self.payload = Buffer(overwritten=payload)
        elif self.type == LogicalType.BOOL:
            self.values = Buffer(overwritten=rows)
            self.payload = Buffer(0)
        else:
            self.values = Buffer(
                overwritten=rows * dtype_size(self.type.physical)
            )
            self.payload = Buffer(0)

    def finish(deinit self) -> AnyArray:
        """Returns the finished column.

        Returns:
            The column, owning all of its memory.
        """
        if self.type == LogicalType.STRING or self.type == LogicalType.BINARY:
            return _as_text(
                self.values^,
                self.payload^,
                self.validity^,
                self.rows,
                self.type,
            )
        return AnyArray(
            ColumnData(self.values^, self.validity^, self.rows), self.type
        )


def _plan_views(array: ArrowArray, length: Int, whole: Bool) raises -> Int:
    """Adds up the payload a view array needs, checking every view on the way.

    This is the first of the two passes `_import_views` makes, lifted out so that
    the second one can write into a column that is already allocated. A view is
    three numbers a stranger chose, so the checks live here and the fill that
    follows can follow a view without looking at it again.

    Args:
        array: The array.
        length: The number of rows.
        whole: Whether this array is the producer's whole array rather than a
            slice of it. A slice cannot take the producer's data buffer
            wholesale, because every other slice of the same array would take it
            too and the column would hold the payload once per slice.

    Returns:
        How many payload bytes this array contributes.

    Raises:
        Error: If a view names a data buffer the array does not have, or runs
            past the end of it.
    """
    var variadic = Int(array.n_buffers) - 3
    var validity = _import_validity(array, length)
    var source = (
        _bytes_at(array, 1)
        .unsafe_bitcast[StringView]()
        .unsafe_offset(Int(array.offset))
    )
    var sizes = _sizes_at(array, variadic)

    var needed = 0
    for i in range(length):
        if not validity.get(i):
            continue
        ref view = source.unsafe_offset(i)[]
        if view.is_inline():
            continue
        var block = view.block()
        if block >= variadic:
            raise Error(
                String(
                    "arrow: view at row ",
                    i,
                    " names data buffer ",
                    block,
                    " but the array has ",
                    variadic,
                )
            )
        var count = len(view)
        if (
            view.offset() < 0
            or Int64(view.offset() + count)
            > sizes.unsafe_offset(block).unsafe_load()
        ):
            raise Error(
                String(
                    "arrow: view at row ",
                    i,
                    " runs past the end of data buffer ",
                    block,
                )
            )
        needed += count

    if whole and variadic == 1 and array.offset == 0:
        # The block goes over whole, gaps and all, because rewriting the offsets
        # to close them is exactly the work the fill's fast path exists to skip.
        return Int(sizes.unsafe_load())
    return needed


def _plan_offsets[index: DType](array: ArrowArray, length: Int) raises -> Int:
    """Adds up the payload an offset based array needs.

    Parameters:
        index: The offset type.

    Args:
        array: The array.
        length: The number of rows.

    Returns:
        How many payload bytes this array contributes. An element of twelve
        bytes or fewer contributes none, because it lives inside its own view.

    Raises:
        Error: If a pair of offsets goes backwards, which is a negative element
            length and would be an enormous one passed to a copy.
    """
    if length == 0:
        return 0
    var validity = _import_validity(array, length)
    var offsets = (
        _bytes_at(array, 1)
        .unsafe_bitcast[Scalar[index]]()
        .unsafe_offset(Int(array.offset))
    )
    var needed = 0
    for i in range(length):
        if not validity.get(i):
            continue
        var count = Int(
            offsets.unsafe_offset(i + 1).unsafe_load()
            - offsets.unsafe_offset(i).unsafe_load()
        )
        if count < 0:
            raise Error(String("arrow: offsets go backwards at row ", i))
        if count > INLINE_CAPACITY:
            needed += count
    return needed


def payload_for(
    array: ArrowArray, format: StringSlice, whole: Bool = True
) raises -> Int:
    """Checks an array and returns the payload bytes it will contribute.

    The planning half of a fill in place read. Nothing is written and nothing is
    allocated except a bitmap the size of the array's own rows, so a caller can
    ask this of every batch in a file before deciding how big anything is.

    Args:
        array: The array.
        format: The schema's format string.
        whole: Whether this array is the producer's whole array rather than a
            slice of it, which decides whether a view column may take the
            producer's data buffer as it stands.

    Returns:
        The payload bytes, and zero for every type that has no payload.

    Raises:
        Error: If the type or shape is one firepanda cannot read.
    """
    _check(array, format)
    var length = Int(array.length)
    if format == "u" or format == "z":
        return _plan_offsets[DType.int32](array, length)
    if format == "U" or format == "Z":
        return _plan_offsets[DType.int64](array, length)
    var type = _type_for(format)
    if type == LogicalType.STRING or type == LogicalType.BINARY:
        return _plan_views(array, length, whole)
    return 0


def _fill_views(
    mut sink: ColumnSink,
    at: Int,
    payload_at: Int,
    array: ArrowArray,
    whole: Bool,
) raises -> Bitmap:
    """Copies a view array into a sink at a row and a payload offset.

    Args:
        sink: The column being filled.
        at: The first row this array's rows occupy.
        payload_at: The first payload byte this array's payload occupies.
        array: The array.
        whole: Whether this array is the producer's whole array rather than a
            slice of it. `payload_for` was asked the same question and the two
            answers have to agree, because one sized the payload and the other
            fills it.

    Returns:
        This array's validity, for the caller to paste.

    Raises:
        Error: If a buffer is missing.
    """
    var length = Int(array.length)
    var validity = _import_validity(array, length)
    var variadic = Int(array.n_buffers) - 3
    var target = (
        sink.values.unsafe_ptr().unsafe_bitcast[StringView]().unsafe_offset(at)
    )
    var sizes = _sizes_at(array, variadic)

    if whole and variadic == 1 and array.offset == 0:
        # The producer's column is already shaped like ours, so the views and the
        # payload are one copy each. What is left is arithmetic: a view's offset
        # was relative to the start of the producer's block and the block has
        # landed at `payload_at`, so every long view moves by that much. The
        # first batch of a file has a `payload_at` of zero and skips even that.
        var total = Int(sizes.unsafe_load())
        if length > 0:
            unsafe_memcpy(
                dest=sink.values.unsafe_ptr().unsafe_offset(at * VIEW_SIZE),
                src=_bytes_at(array, 1),
                count=length * VIEW_SIZE,
            )
        if total > 0:
            unsafe_memcpy(
                dest=sink.payload.unsafe_ptr().unsafe_offset(payload_at),
                src=_bytes_at(array, 2),
                count=total,
            )
        if payload_at != 0 or array.null_count != 0:
            for i in range(length):
                if not validity.get(i):
                    # A null element's view is whatever the producer left there,
                    # which nothing downstream should follow.
                    target.unsafe_offset(i)[] = StringView()
                    continue
                if payload_at == 0:
                    continue
                var view = target.unsafe_offset(i)[]
                if view.is_inline():
                    continue
                var spot = payload_at + view.offset()
                target.unsafe_offset(i)[] = make_long_at(
                    sink.payload.unsafe_ptr().unsafe_offset(spot),
                    len(view),
                    0,
                    spot,
                )
        return validity^

    var source = (
        _bytes_at(array, 1)
        .unsafe_bitcast[StringView]()
        .unsafe_offset(Int(array.offset))
    )
    var written = 0
    for i in range(length):
        if not validity.get(i):
            target.unsafe_offset(i)[] = StringView()
            continue
        ref view = source.unsafe_offset(i)[]
        if view.is_inline():
            target.unsafe_offset(i)[] = view
            continue
        var count = len(view)
        var spot = payload_at + written
        var dest = sink.payload.unsafe_ptr().unsafe_offset(spot)
        unsafe_memcpy(
            dest=dest,
            src=_bytes_at(array, 2 + view.block()).unsafe_offset(view.offset()),
            count=count,
        )
        target.unsafe_offset(i)[] = make_long_at(dest, count, 0, spot)
        written += count
    return validity^


def _fill_offsets[
    index: DType
](
    mut sink: ColumnSink, at: Int, payload_at: Int, array: ArrowArray
) raises -> Bitmap:
    """Copies an offset based array into a sink at a row and a payload offset.

    Parameters:
        index: The offset type.

    Args:
        sink: The column being filled.
        at: The first row this array's rows occupy.
        payload_at: The first payload byte this array's payload occupies.
        array: The array.

    Returns:
        This array's validity, for the caller to paste.

    Raises:
        Error: If a buffer is missing.
    """
    var length = Int(array.length)
    var validity = _import_validity(array, length)
    if length == 0:
        return validity^

    var target = (
        sink.values.unsafe_ptr().unsafe_bitcast[StringView]().unsafe_offset(at)
    )
    var offsets = (
        _bytes_at(array, 1)
        .unsafe_bitcast[Scalar[index]]()
        .unsafe_offset(Int(array.offset))
    )
    var data = _bytes_at(array, 2)

    var written = 0
    for i in range(length):
        if not validity.get(i):
            target.unsafe_offset(i)[] = StringView()
            continue
        var start = Int(offsets.unsafe_offset(i).unsafe_load())
        var count = Int(offsets.unsafe_offset(i + 1).unsafe_load()) - start
        var src = data.unsafe_offset(start)
        if count <= INLINE_CAPACITY:
            target.unsafe_offset(i)[] = make_inline_at(src, count)
            continue
        var spot = payload_at + written
        var dest = sink.payload.unsafe_ptr().unsafe_offset(spot)
        unsafe_memcpy(dest=dest, src=src, count=count)
        target.unsafe_offset(i)[] = make_long_at(dest, count, 0, spot)
        written += count
    return validity^


def fill_column(
    mut sink: ColumnSink,
    at: Int,
    payload_at: Int,
    array: ArrowArray,
    format: StringSlice,
    whole: Bool = True,
) raises -> Bitmap:
    """Copies one array into its place in a column that is already allocated.

    The other half of `payload_for`, and it must be given the offsets that
    planning produced: `at` is the running row count and `payload_at` the running
    payload total, both taken before this array. Nothing here allocates the
    column, so several calls against one sink may run at once as long as their
    ranges do not overlap.

    Args:
        sink: The column being filled.
        at: The first row this array's rows occupy.
        payload_at: The first payload byte this array's payload occupies.
        array: The array.
        format: The schema's format string.
        whole: Whether this array is the producer's whole array rather than a
            slice of it. It has to be the same answer `payload_for` was given.

    Returns:
        This array's validity, one bit per row of this array. The caller pastes
        it at `at`, because a batch boundary falls inside a byte and two threads
        writing that byte would lose each other's bits.

    Raises:
        Error: If a buffer is missing or the type is one firepanda cannot read.
    """
    var length = Int(array.length)
    if format == "u" or format == "z":
        return _fill_offsets[DType.int32](sink, at, payload_at, array)
    if format == "U" or format == "Z":
        return _fill_offsets[DType.int64](sink, at, payload_at, array)
    if sink.type == LogicalType.STRING or sink.type == LogicalType.BINARY:
        return _fill_views(sink, at, payload_at, array, whole)
    if sink.type == LogicalType.BOOL:
        if length > 0:
            var bits = _bytes_at(array, 1)
            var offset = Int(array.offset)
            var out = sink.values.unsafe_ptr().unsafe_offset(at)
            for i in range(length):
                var bit = offset + i
                var byte = bits.unsafe_offset(bit >> 3).unsafe_load()
                out.unsafe_offset(i).unsafe_write((byte >> UInt8(bit & 7)) & 1)
        return _import_validity(array, length)
    var width = dtype_size(sink.type.physical)
    if length > 0:
        unsafe_memcpy(
            dest=sink.values.unsafe_ptr().unsafe_offset(at * width),
            src=_bytes_at(array, 1).unsafe_offset(Int(array.offset) * width),
            count=length * width,
        )
    return _import_validity(array, length)


def import_array(
    mut schema: ArrowSchema, mut array: ArrowArray
) raises -> AnyArray:
    """Takes a column from a C producer and returns a firepanda column.

    Consumes both structures and releases them, which is what the C interface
    says a consumer does, and does it on the way out of a failure as well. See
    the module docstring for why nothing here is borrowed.

    Args:
        schema: The schema, describing the type.
        array: The array, holding the data.

    Returns:
        The imported column, owning all of its memory.

    Raises:
        Error: If the type or shape is one firepanda cannot read. Both
            structures are released before the error propagates.
    """
    if schema.is_released():
        raise Error("arrow: the schema has already been released")

    var format: String
    try:
        format = format_string(schema)
    except error:
        release_array(array)
        release_schema(schema)
        raise error

    var out: AnyArray
    try:
        if array.is_released():
            raise Error("arrow: the array has already been released")
        out = build_column(array, format)
    except error:
        release_array(array)
        release_schema(schema)
        raise error

    release_array(array)
    release_schema(schema)
    return out^
