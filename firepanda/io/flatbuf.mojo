"""FlatBuffers, which is the format Arrow writes its metadata in.

Every Arrow IPC message begins with a FlatBuffers table describing what follows,
so a reader for this format is the thing standing between firepanda and every
`.arrow` file in the world. It is a small format and the whole of what firepanda
needs is here: no code generation, no schema compiler, no dependency.

The layout is worth stating once, because the rest of this file is unreadable
without it. A buffer holds a root table, and the first four bytes are an unsigned
offset to it, counted forwards from where the buffer starts. A table is a signed
offset to its vtable followed by its inline fields, and the vtable is what makes
the format extensible: two byte lengths, then one two byte entry per field
holding where in the table that field sits, or zero when the field was never
written. A field the writer left out costs nothing at all and reads back as its
default, which is how a reader written against one version of a schema keeps
working against a later one.

Everything larger than a scalar is stored by offset rather than inline, and those
offsets count forwards from the four bytes that hold them rather than from the
start of the buffer, so following one is an addition and never a lookup. Strings
and vectors are a four byte count followed by the elements. Structs are inline
and have no vtable, which is why `vector_element` exists next to `vector_table`.

Every read here is bounds checked and every one of them raises. That is not the
usual firepanda posture, where a kernel trusts the invariants of the buffer it
was handed, and the difference is that this data came from somebody else. A
FlatBuffer is a graph of offsets, and following an offset from a corrupt or
hostile file lands wherever that file says. The metadata of a record batch is a
few hundred bytes next to a body of megabytes, so the checks are free in any
sense that matters.

The writer builds back to front, which is what lets a table know where its
children are before it writes the fields pointing at them. So an offset in the
builder is a distance from the end of the buffer rather than the start, and it
stays valid while everything in front of it moves. Vtables are shared between
tables that use the same fields, which a schema with many columns gets a lot of
use out of, since every column writes a `Field` with the same shape.

What is not here is the parts of FlatBuffers that Arrow does not use: file
identifiers, size prefixes, shared strings, and the unions and enums that
generated code would give names to. Enum values are plain integers at this layer
and get their names one level up, in the IPC reader.
"""

from std.collections.span import Span
from std.memory import unsafe_memcpy
from std.sys import size_of

comptime OFFSET_SIZE = 4
"""Bytes in a uoffset, the forward reference to a table, string or vector."""

comptime VOFFSET_SIZE = 2
"""Bytes in a vtable entry."""


def _bounds(data: Span[UInt8, _], pos: Int, size: Int) raises:
    """Raises unless `size` bytes starting at `pos` are inside the buffer.

    Args:
        data: The buffer being read.
        pos: The byte position the read starts at.
        size: The number of bytes the read covers.
    """
    if pos < 0 or size < 0 or pos > len(data) - size:
        raise Error(
            String(
                "flatbuffers: a read of ",
                size,
                " bytes at ",
                pos,
                " is outside the ",
                len(data),
                " byte buffer",
            )
        )


def read_scalar[dt: DType](data: Span[UInt8, _], pos: Int) raises -> Scalar[dt]:
    """Reads one little-endian scalar out of the buffer.

    The read goes through a copy rather than a cast because a FlatBuffer is
    aligned relative to its own start and this one may sit at any offset in a
    file. A misaligned load is a fault on some targets and merely slow on the
    rest, and the copy of at most eight bytes costs less than finding out which.

    Parameters:
        dt: The type to read. Its size decides how many bytes are read.

    Args:
        data: The buffer being read.
        pos: The byte position to read from.

    Returns:
        The value at that position.
    """
    var size = size_of[Scalar[dt]]()
    _bounds(data, pos, size)
    var out = Scalar[dt]()
    unsafe_memcpy(
        dest=Pointer(to=out).unsafe_bitcast[UInt8](),
        src=data.unsafe_ptr().unsafe_offset(pos),
        count=size,
    )
    return out


def follow(data: Span[UInt8, _], pos: Int) raises -> Int:
    """Follows a uoffset at `pos` and returns where it points.

    Args:
        data: The buffer being read.
        pos: The position of the four byte offset.

    Returns:
        The absolute position of the table, string or vector it names.
    """
    var target = pos + Int(read_scalar[DType.uint32](data, pos))
    _bounds(data, target, OFFSET_SIZE)
    return target


def root_table(data: Span[UInt8, _], start: Int = 0) raises -> Int:
    """Returns the position of the root table.

    Args:
        data: The buffer being read.
        start: Where the FlatBuffer begins, for a buffer embedded in a larger
            one. Offsets inside are relative to this, not to zero.

    Returns:
        The absolute position of the root table.
    """
    _bounds(data, start, OFFSET_SIZE)
    return follow(data, start)


def field_position(data: Span[UInt8, _], table: Int, field: Int) raises -> Int:
    """Returns where a field sits, or -1 when the writer left it out.

    Args:
        data: The buffer being read.
        table: The position of the table.
        field: The field's slot number, counting from zero in schema order. A
            union takes two slots, one for the tag and one for the value.

    Returns:
        The absolute position of the field, or -1 if it is absent.
    """
    if field < 0:
        raise Error(String("flatbuffers: field ", field, " is negative"))
    var vtable = table - Int(read_scalar[DType.int32](data, table))
    var vtable_bytes = Int(read_scalar[DType.uint16](data, vtable))
    if vtable_bytes < 4:
        raise Error(
            String(
                "flatbuffers: the vtable at ",
                vtable,
                " claims to be ",
                vtable_bytes,
                " bytes, and the smallest legal one is 4",
            )
        )
    _bounds(data, vtable, vtable_bytes)
    var slot = 4 + field * VOFFSET_SIZE
    if slot + VOFFSET_SIZE > vtable_bytes:
        return -1
    var voffset = Int(read_scalar[DType.uint16](data, vtable + slot))
    if voffset == 0:
        return -1
    var table_bytes = Int(read_scalar[DType.uint16](data, vtable + 2))
    if voffset >= table_bytes:
        raise Error(
            String(
                "flatbuffers: field ",
                field,
                " sits at ",
                voffset,
                " in a table the vtable says is ",
                table_bytes,
                " bytes",
            )
        )
    return table + voffset


def has_field(data: Span[UInt8, _], table: Int, field: Int) raises -> Bool:
    """Returns whether a field was written.

    A scalar that happens to equal its default is not written, so this answers a
    question about the encoding rather than about the value.

    Args:
        data: The buffer being read.
        table: The position of the table.
        field: The field's slot number.

    Returns:
        True when the field is present.
    """
    return field_position(data, table, field) >= 0


def field_scalar[
    dt: DType
](
    data: Span[UInt8, _], table: Int, field: Int, default: Scalar[dt]
) raises -> Scalar[dt]:
    """Reads a scalar field, or its default when the writer left it out.

    Parameters:
        dt: The field's type.

    Args:
        data: The buffer being read.
        table: The position of the table.
        field: The field's slot number.
        default: What the schema says an absent field means.

    Returns:
        The field's value.
    """
    var pos = field_position(data, table, field)
    if pos < 0:
        return default
    return read_scalar[dt](data, pos)


def field_table(data: Span[UInt8, _], table: Int, field: Int) raises -> Int:
    """Returns the position of a table valued field, or -1 when absent.

    Args:
        data: The buffer being read.
        table: The position of the table.
        field: The field's slot number.

    Returns:
        The absolute position of the child table, or -1 if it is absent.
    """
    var pos = field_position(data, table, field)
    if pos < 0:
        return -1
    return follow(data, pos)


def field_vector(data: Span[UInt8, _], table: Int, field: Int) raises -> Int:
    """Returns the position of a vector's count word, or -1 when absent.

    Args:
        data: The buffer being read.
        table: The position of the table.
        field: The field's slot number.

    Returns:
        The absolute position of the four byte element count, or -1 if the field
        is absent.
    """
    var pos = field_position(data, table, field)
    if pos < 0:
        return -1
    return follow(data, pos)


def vector_length(data: Span[UInt8, _], vector: Int) raises -> Int:
    """Returns how many elements a vector holds.

    Args:
        data: The buffer being read.
        vector: The position of the vector's count word.

    Returns:
        The element count.
    """
    return Int(read_scalar[DType.uint32](data, vector))


def vector_element(
    data: Span[UInt8, _], vector: Int, index: Int, element_size: Int
) raises -> Int:
    """Returns where an element of a vector sits.

    For a vector of scalars or of inline structs this is the element itself. For
    a vector of tables or strings it is the offset naming one, and
    `vector_table` follows it.

    Args:
        data: The buffer being read.
        vector: The position of the vector's count word.
        index: Which element, counting from zero.
        element_size: The size of one element in bytes.

    Returns:
        The absolute position of the element.
    """
    var count = vector_length(data, vector)
    if index < 0 or index >= count:
        raise Error(
            String(
                "flatbuffers: element ",
                index,
                " of a vector of ",
                count,
                " is out of range",
            )
        )
    var pos = vector + OFFSET_SIZE + index * element_size
    _bounds(data, pos, element_size)
    return pos


def vector_table(data: Span[UInt8, _], vector: Int, index: Int) raises -> Int:
    """Returns the position of one table in a vector of tables.

    Args:
        data: The buffer being read.
        vector: The position of the vector's count word.
        index: Which element, counting from zero.

    Returns:
        The absolute position of that table.
    """
    return follow(data, vector_element(data, vector, index, OFFSET_SIZE))


def string_at(data: Span[UInt8, _], pos: Int) raises -> String:
    """Reads a string given the position of its count word.

    Args:
        data: The buffer being read.
        pos: The position of the four byte length.

    Returns:
        A copy of the string. FlatBuffers stores it null terminated, and the
        terminator is not part of the value.
    """
    var count = Int(read_scalar[DType.uint32](data, pos))
    _bounds(data, pos + OFFSET_SIZE, count)
    return String(
        StringSlice(
            unsafe_from_utf8=Span(
                unsafe_ptr=data.unsafe_ptr().unsafe_offset(pos + OFFSET_SIZE),
                length=count,
            )
        )
    )


def field_string(data: Span[UInt8, _], table: Int, field: Int) raises -> String:
    """Reads a string field, or the empty string when the writer left it out.

    Arrow uses an absent name and an empty name interchangeably, so the two are
    not distinguished here. `has_field` answers that question for a caller who
    needs it.

    Args:
        data: The buffer being read.
        table: The position of the table.
        field: The field's slot number.

    Returns:
        The string, or "" if the field is absent.
    """
    var pos = field_position(data, table, field)
    if pos < 0:
        return String()
    return string_at(data, follow(data, pos))


struct Builder(Movable):
    """Writes a FlatBuffer, back to front.

    A table cannot be written until its children exist, because it stores
    offsets to them, so the builder fills a buffer from the end towards the
    start and every offset it hands out is a distance from that end. Those stay
    valid when the buffer is reallocated, which a distance from the start would
    not.

    The order of calls matters and is checked. A table is opened with
    `start_table`, its fields are added in any order, and `end_table` returns
    the offset that names it. Nothing may be built inside an open table, because
    the builder is writing that table's fields into the same run of bytes.
    """

    var _buf: List[UInt8]
    """The whole allocation. The written bytes are the tail of it."""

    var _space: Int
    """How many bytes at the front are still free."""

    var _vtable: List[Int]
    """Where each field of the open table went, or zero for the ones left out."""

    var _vtables: List[Int]
    """Every vtable written so far, for sharing."""

    var _table_end: Int
    """Where the open table's fields start, used for the table's own length."""

    var _min_align: Int
    """The widest thing written, which is what the finished buffer aligns to."""

    var _nested: Bool
    """Whether a table or vector is open."""

    def __init__(out self, capacity: Int = 1024):
        """Constructs an empty builder.

        Args:
            capacity: Bytes to allocate up front. The builder grows by doubling,
                so this only decides how many times that happens.
        """
        var initial = capacity if capacity > 0 else 1024
        self._buf = List[UInt8](length=initial, fill=0)
        self._space = initial
        self._vtable = List[Int]()
        self._vtables = List[Int]()
        self._table_end = 0
        self._min_align = 1
        self._nested = False

    def position(self) -> Int:
        """Returns how many bytes have been written, which is also the offset
        of the next thing to be written.

        Returns:
            The distance from the end of the buffer to the last byte written.
        """
        return len(self._buf) - self._space

    def _grow(mut self):
        """Doubles the allocation, keeping the written bytes at the end."""
        var old = len(self._buf)
        var fresh = List[UInt8](length=old * 2, fill=0)
        unsafe_memcpy(
            dest=fresh.unsafe_ptr().unsafe_offset(old),
            src=self._buf.unsafe_ptr(),
            count=old,
        )
        self._buf = fresh^
        self._space += old

    def _prep(mut self, size: Int, additional: Int):
        """Makes room for `size` bytes, padded so they land aligned.

        Args:
            size: The size of the thing about to be written, which is also what
                it must be aligned to.
            additional: Bytes that will be written in front of it and must not
                disturb its alignment, such as the count word of a vector.
        """
        if size > self._min_align:
            self._min_align = size
        var pad = (-(self.position() + additional)) & (size - 1)
        while self._space < pad + size + additional:
            self._grow()
        for _ in range(pad):
            self._space -= 1
            self._buf[self._space] = 0

    def _place[dt: DType](mut self, value: Scalar[dt]):
        """Writes a scalar into space `_prep` already made.

        Parameters:
            dt: The type being written.

        Args:
            value: The value.
        """
        var size = size_of[Scalar[dt]]()
        var local = value
        self._space -= size
        unsafe_memcpy(
            dest=self._buf.unsafe_ptr().unsafe_offset(self._space),
            src=Pointer(to=local).unsafe_bitcast[UInt8](),
            count=size,
        )

    def prepend[dt: DType](mut self, value: Scalar[dt]):
        """Writes one aligned scalar.

        Parameters:
            dt: The type being written.

        Args:
            value: The value.
        """
        self._prep(size_of[Scalar[dt]](), 0)
        self._place(value)

    def _write_at(mut self, offset: Int, value: Int32):
        """Overwrites a four byte word that was left as a placeholder.

        Args:
            offset: The offset of the word, counted from the end.
            value: What to put there.
        """
        var local = value
        unsafe_memcpy(
            dest=self._buf.unsafe_ptr().unsafe_offset(len(self._buf) - offset),
            src=Pointer(to=local).unsafe_bitcast[UInt8](),
            count=4,
        )

    def _read_u16_at(self, offset: Int) -> UInt16:
        """Reads a two byte word already written.

        Args:
            offset: The offset of the word, counted from the end.

        Returns:
            The value there.
        """
        var out = UInt16()
        unsafe_memcpy(
            dest=Pointer(to=out).unsafe_bitcast[UInt8](),
            src=self._buf.unsafe_ptr().unsafe_offset(len(self._buf) - offset),
            count=2,
        )
        return out

    def prepend_offset(mut self, offset: Int) raises:
        """Writes a forward reference to something already built.

        Args:
            offset: The offset the reference names.
        """
        self._prep(OFFSET_SIZE, 0)
        if offset > self.position():
            raise Error(
                String(
                    "flatbuffers: an offset of ",
                    offset,
                    " points past the ",
                    self.position(),
                    " bytes written so far",
                )
            )
        self._place(UInt32(self.position() - offset + OFFSET_SIZE))

    def start_table(mut self, fields: Int) raises:
        """Opens a table.

        Args:
            fields: How many slots the table's schema has. Fields left out cost
                nothing, so this is the count in the schema rather than the
                count being written.
        """
        if self._nested:
            raise Error(
                "flatbuffers: a table cannot be started while another table or"
                " vector is open"
            )
        self._nested = True
        self._vtable = List[Int](length=fields, fill=0)
        self._table_end = self.position()

    def _slot(mut self, field: Int) raises:
        """Records that the field just written belongs in a given slot.

        Args:
            field: The slot number.
        """
        if field < 0 or field >= len(self._vtable):
            raise Error(
                String(
                    "flatbuffers: slot ",
                    field,
                    " is outside the ",
                    len(self._vtable),
                    " slots this table was started with",
                )
            )
        self._vtable[field] = self.position()

    def add_scalar[
        dt: DType
    ](mut self, field: Int, value: Scalar[dt], default: Scalar[dt]) raises:
        """Adds a scalar field, unless it equals its default.

        A field that equals its default is not written at all, which is not a
        space optimization so much as the definition of the default: a reader
        that finds the slot empty returns it.

        Parameters:
            dt: The field's type.

        Args:
            field: The slot number.
            value: The value.
            default: What the schema says an absent field means.
        """
        if value == default:
            return
        self.prepend(value)
        self._slot(field)

    def add_offset(mut self, field: Int, offset: Int) raises:
        """Adds a field naming a table, string or vector.

        Args:
            field: The slot number.
            offset: The offset of the thing being named. Zero means absent, the
                way a null reference does.
        """
        if offset == 0:
            return
        self.prepend_offset(offset)
        self._slot(field)

    def end_table(mut self) raises -> Int:
        """Closes the open table and writes its vtable.

        Returns:
            The offset naming the table.
        """
        if not self._nested:
            raise Error("flatbuffers: no table is open")
        self.prepend[DType.int32](0)
        var table = self.position()

        var used = len(self._vtable)
        while used > 0 and self._vtable[used - 1] == 0:
            used -= 1
        for i in range(used - 1, -1, -1):
            var slot = self._vtable[i]
            self.prepend(UInt16(0) if slot == 0 else UInt16(table - slot))
        self.prepend(UInt16(table - self._table_end))
        var vtable_bytes = (used + 2) * VOFFSET_SIZE
        self.prepend(UInt16(vtable_bytes))

        var vtable = self.position()
        var shared = -1
        for i in range(len(self._vtables)):
            var candidate = self._vtables[i]
            if self._same_vtable(candidate, vtable, vtable_bytes):
                shared = candidate
                break

        if shared >= 0:
            # Drop the vtable just written and point at the identical one. Every
            # `Field` in a schema has the same shape, so a wide frame writes one
            # vtable rather than one per column.
            self._space = len(self._buf) - table
            self._write_at(table, Int32(shared - table))
        else:
            self._vtables.append(vtable)
            self._write_at(table, Int32(vtable - table))

        self._nested = False
        return table

    def _same_vtable(self, candidate: Int, vtable: Int, size: Int) -> Bool:
        """Returns whether an earlier vtable is byte for byte the new one.

        Args:
            candidate: The offset of the earlier vtable.
            vtable: The offset of the one just written.
            size: The size of the one just written.

        Returns:
            True when they are interchangeable.
        """
        if Int(self._read_u16_at(candidate)) != size:
            return False
        for i in range(2, size, 2):
            if self._read_u16_at(candidate - i) != self._read_u16_at(
                vtable - i
            ):
                return False
        return True

    def start_vector(
        mut self, element_size: Int, count: Int, alignment: Int
    ) raises:
        """Opens a vector.

        Args:
            element_size: The size of one element in bytes.
            count: How many elements will be written.
            alignment: What one element must be aligned to, which is its size
                for a scalar and the widest member for a struct.
        """
        if self._nested:
            raise Error(
                "flatbuffers: a vector cannot be started while another table or"
                " vector is open"
            )
        self._nested = True
        self._prep(OFFSET_SIZE, element_size * count)
        self._prep(alignment, element_size * count)

    def end_vector(mut self, count: Int) raises -> Int:
        """Closes the open vector by writing its count.

        Args:
            count: How many elements were written.

        Returns:
            The offset naming the vector.
        """
        if not self._nested:
            raise Error("flatbuffers: no vector is open")
        self._nested = False
        self.prepend(UInt32(count))
        return self.position()

    def create_string(mut self, text: StringSlice) raises -> Int:
        """Writes a string.

        Args:
            text: The string.

        Returns:
            The offset naming it.
        """
        var bytes = text.as_bytes()
        var count = len(bytes)
        if self._nested:
            raise Error(
                "flatbuffers: a string cannot be written while a table or"
                " vector is open"
            )
        self._nested = True
        # One more than the length, for the null terminator FlatBuffers writes
        # so that a C reader can use the bytes in place.
        self._prep(OFFSET_SIZE, count + 1)
        self._place(UInt8(0))
        self._space -= count
        if count > 0:
            unsafe_memcpy(
                dest=self._buf.unsafe_ptr().unsafe_offset(self._space),
                src=bytes.unsafe_ptr(),
                count=count,
            )
        return self.end_vector(count)

    def create_bytes(mut self, data: Span[UInt8, _]) raises -> Int:
        """Writes a vector of bytes.

        Args:
            data: The bytes.

        Returns:
            The offset naming the vector.
        """
        var count = len(data)
        self.start_vector(1, count, 1)
        self._space -= count
        if count > 0:
            unsafe_memcpy(
                dest=self._buf.unsafe_ptr().unsafe_offset(self._space),
                src=data.unsafe_ptr(),
                count=count,
            )
        return self.end_vector(count)

    def create_offsets(mut self, offsets: List[Int]) raises -> Int:
        """Writes a vector of references to tables or strings already built.

        Args:
            offsets: The offsets, in order.

        Returns:
            The offset naming the vector.
        """
        var count = len(offsets)
        self.start_vector(OFFSET_SIZE, count, OFFSET_SIZE)
        for i in range(count - 1, -1, -1):
            self.prepend_offset(offsets[i])
        return self.end_vector(count)

    def finish(mut self, root: Int) raises -> List[UInt8]:
        """Closes the buffer and returns its bytes.

        Args:
            root: The offset of the root table.

        Returns:
            The finished FlatBuffer, starting at its root offset.
        """
        if self._nested:
            raise Error("flatbuffers: a table or vector is still open")
        self._prep(self._min_align, OFFSET_SIZE)
        self.prepend_offset(root)
        var size = self.position()
        var out = List[UInt8](unsafe_uninit_length=size)
        unsafe_memcpy(
            dest=out.unsafe_ptr(),
            src=self._buf.unsafe_ptr().unsafe_offset(self._space),
            count=size,
        )
        return out^
