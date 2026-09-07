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
to that file rather than reading the buffers itself. Everything it learned about
shifting a validity bitmap, turning offsets into views and checking a view
against the buffer it names applies here unchanged, and the alternative was a
second implementation of all of it that would drift.

Every batch is located before any of it is copied, which is what lets a file cost
the same however its writer chunked it. Decoding a batch produces pointers into
the body rather than data, so the whole file can be looked at before anything is
allocated, and then the columns are allocated once and filled in place.

The copy itself is `assemble`, next door, because a Parquet reader and a query
result want exactly the same thing done with the arrays they produce.

What is not read yet is anything nested or compressed. A record batch that says
its body is compressed is refused by name rather than by reading it as raw bytes,
because lz4 framed data read as int64 is not an error that shows up anywhere near
here.

Dictionary encoded columns are read, and they are the one type whose description
does not fit in the schema message. The schema says which columns are encoded and
under what id, the categories arrive later in dictionary batch messages of their
own, and the record batches carry codes. So the schema pass records what to
expect, the dictionary batches fill it in, and the codes are copied by the same
path as any other integer column with the categories attached at the end. A
column whose categories never arrive is refused rather than handed back as the
integers it physically is, because that is the one way this path could produce an
answer instead of an error.

The reader is strict about the parts a file could lie about. Every buffer is
checked against the length of the body it is supposed to sit in, every node
against the row count, and the buffer entries are consumed in the order the
schema says, so a batch with too few of them is caught rather than read into the
next column's data.
"""

from std.collections.span import Span

from firepanda.array.any import AnyArray
from firepanda.array.data import ColumnData
from firepanda.array.nested import ITEM, ROOT, NestedNode
from firepanda.array.strings import StringArray
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.dtype.temporal import unit_for_code
from firepanda.frame.frame import DataFrame

from .arrow_c import STRUCT_FORMAT, ArrowArray, NullableVoidPtr, VoidPtr
from .arrow_import import (
    build_column,
    import_list_offsets,
    import_validity,
)
from .assemble import (
    ArrowDictionary,
    ArrowLayout,
    ArrowNested,
    assemble,
    attach_dictionary,
)
from .flatbuf import (
    field_scalar,
    field_string,
    field_table,
    field_vector,
    has_field,
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
comptime TYPE_DATE = 8
comptime TYPE_TIMESTAMP = 10
comptime TYPE_LIST = 12
comptime TYPE_STRUCT = 13
comptime TYPE_DURATION = 18
comptime TYPE_LARGE_BINARY = 19
comptime TYPE_LARGE_UTF8 = 20
comptime TYPE_LARGE_LIST = 21
comptime TYPE_BINARY_VIEW = 23
comptime TYPE_UTF8_VIEW = 24

comptime LIST_FORMAT = "+l"
"""What the C Data Interface calls a list, whose offsets are int32."""

comptime LARGE_LIST_FORMAT = "+L"
"""What it calls a large list, whose offsets are int64. The corpus the
conformance suite reads is written this way, because pyarrow builds a large list
from Python lists by default."""


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


def _int_format(data: Span[UInt8, _], type: Int, name: String) raises -> String:
    """Turns an `Int` type table into a format string.

    Read from two places, because a dictionary's index type is an `Int` table
    exactly like an integer column's own type and neither is allowed to accept a
    width the other refuses.

    Args:
        data: The buffer holding the schema message.
        type: The position of the `Int` table.
        name: The column name, for the error message.

    Returns:
        The format string.

    Raises:
        Error: If the width is not one of 8, 16, 32 and 64.
    """
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


@fieldwise_init
struct _Child(Copyable, Movable):
    """One field node below a nested column, in the order Arrow writes them.

    A schema is a tree and a record batch is a flat vector of nodes walked in
    pre-order, so the reader keeps the tree the same way the batch does: a flat
    list, each entry naming the entry it hangs off. That is also the shape a
    finished nested column holds its children in, so building one is a copy of
    this list with the buffers filled in rather than a shape change.
    """

    var column: Int
    """Which top level column this belongs to."""

    var parent: Int
    """The position in this list of the node above, or `ROOT` when the node hangs
    off the column itself."""

    var name: String
    """The field name, and `item` for a list's child."""

    var format: String
    """The C Data Interface format string for this node's own type."""


def _nested_children(
    data: Span[UInt8, _],
    field: Int,
    name: String,
    kind: Int,
    column: Int,
    parent: Int,
    mut children: List[_Child],
) raises -> String:
    """Reads the children of a list or a struct field into the flat list.

    Each child is appended before its own children are read, which is what makes
    the list pre-order and lets a grandchild name its parent by a position that
    already exists.

    Args:
        data: The buffer holding the schema message.
        field: The position of the `Field` table.
        name: The field's name, for the error message.
        kind: Which Arrow type table the field carries.
        column: The top level column this field belongs to.
        parent: The position of this field in `children`, or `ROOT` when it is
            the column itself.
        children: Filled with one entry per node below this field.

    Returns:
        The format string for this field's own type.

    Raises:
        Error: If a list does not have exactly one child, or a child holds a
            type firepanda cannot read.
    """
    var kids = field_vector(data, field, 5)
    var count = 0 if kids < 0 else vector_length(data, kids)
    if kind != TYPE_STRUCT and count != 1:
        raise Error(
            String(
                "arrow ipc: column '",
                name,
                "' is a list with ",
                count,
                " children, and a list has one",
            )
        )

    for i in range(count):
        var kid = vector_table(data, kids, i)
        var kid_name = field_string(data, kid, 0)
        if field_table(data, kid, 4) >= 0:
            # A dictionary encoded child would need its categories attached
            # inside the tree, and the categories arrive in messages of their own
            # against an id the assembler matches against a top level column.
            # Refused by name rather than read as the codes it physically is.
            raise Error(
                String(
                    "arrow ipc: field '",
                    kid_name,
                    "' inside column '",
                    name,
                    (
                        "' is dictionary encoded, and firepanda reads a"
                        " dictionary at the top level of a frame only"
                    ),
                )
            )
        children.append(_Child(column, parent, kid_name, String()))
        var at = len(children) - 1
        var format = _format_for(data, kid, kid_name, column, at, children)
        children[at].format = format^

    if kind == TYPE_STRUCT:
        return String(STRUCT_FORMAT)
    if kind == TYPE_LARGE_LIST:
        return String(LARGE_LIST_FORMAT)
    return String(LIST_FORMAT)


def _format_for(
    data: Span[UInt8, _],
    field: Int,
    name: String,
    column: Int,
    parent: Int,
    mut children: List[_Child],
) raises -> String:
    """Turns one schema field's own type into the format string the importer
    speaks.

    A dictionary encoded field's own type is the type of its categories rather
    than of what the record batches carry, so this is not on its own the answer
    for one of those. `_dictionary_for` is, and calls this for the half of the
    answer that lives here.

    A nested field's own type says only that it is a list or a struct. What is
    inside it goes into `children`, which is why this takes the position of the
    field being read and a list to add to rather than returning a type on its
    own.

    Args:
        data: The buffer holding the schema message.
        field: The position of the `Field` table.
        name: The column name, for the error message.
        column: The top level column this field belongs to.
        parent: The position of this field in `children`, or `ROOT` when it is
            the column itself.
        children: Filled with one entry per node below this field, if it has any.

    Returns:
        The Arrow C Data Interface format string.

    Raises:
        Error: If the type is one firepanda has no column for.
    """
    var kind = Int(field_scalar[DType.uint8](data, field, 2, 0))
    if kind == TYPE_LIST or kind == TYPE_LARGE_LIST or kind == TYPE_STRUCT:
        return _nested_children(
            data, field, name, kind, column, parent, children
        )

    var kids = field_vector(data, field, 5)
    if kids >= 0 and vector_length(data, kids) != 0:
        raise Error(
            String(
                "arrow ipc: column '",
                name,
                "' has Arrow type ",
                kind,
                ", which has children and is not a list or a struct",
            )
        )
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
        return _int_format(data, type, name)
    if kind == TYPE_TIMESTAMP:
        # Field 0 is the unit and field 1 is the zone name, absent when the
        # column is naive. The unit is passed through rather than normalized to
        # nanoseconds, which is the whole point: pandas 3 carries the resolution
        # on the dtype, a file written at second resolution is a second column
        # in pandas too, and a reader that multiplied on the way in would answer
        # a different dtype for every temporal file anybody has.
        var unit = unit_for_code(
            Int(field_scalar[DType.int16](data, type, 0, 0))
        )
        var zone = String()
        if has_field(data, type, 1):
            zone = field_string(data, type, 1)
        return String("ts", unit.code_letter(), ":", zone)
    if kind == TYPE_DATE:
        # The unit defaults to MILLISECOND when the field is absent, which is
        # the flatbuffer default and not a choice this reader gets to make.
        var date_unit = Int(field_scalar[DType.int16](data, type, 0, 1))
        if date_unit != 0:
            raise Error(
                String(
                    "arrow ipc: column '",
                    name,
                    (
                        "' is a date64, and firepanda's date column counts days"
                        " rather than milliseconds"
                    ),
                )
            )
        return "tdD"
    if kind == TYPE_DURATION:
        # One field, the unit, and its flatbuffer default is MILLISECOND rather
        # than the zero the other tables default to, which is why the default
        # argument here is 1 the way the date's is.
        var span = unit_for_code(
            Int(field_scalar[DType.int16](data, type, 0, 1))
        )
        return String("tD", span.code_letter())
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


@fieldwise_init
struct _Encoding(Copyable, Movable):
    """What a schema says about one dictionary encoded column, before the
    categories themselves have arrived."""

    var column: Int
    """Which column of the layout this belongs to."""

    var id: Int64
    """The number the dictionary batch carrying the categories will name."""

    var values: String
    """The format string of the categories, which is what the dictionary batch
    carries and not what the record batches do."""

    var index: String
    """The format string of the codes, which is what the record batches carry and
    what the layout holds for this column."""

    var ordered: Bool
    """Whether the categories have a meaningful order."""


def _dictionary_for(
    data: Span[UInt8, _], field: Int, name: String, column: Int
) raises -> _Encoding:
    """Reads one field's `DictionaryEncoding` table.

    Args:
        data: The buffer holding the schema message.
        field: The position of the `Field` table.
        name: The column name, for the error message.
        column: The column position, carried into the result.

    Returns:
        What the schema says about the encoding.

    Raises:
        Error: If the categories are not strings, or the index type is not one
            firepanda reads.

    Notes:
        The three string spellings accepted are the three firepanda has a string
        column for. The view one is there because firepanda's own writer emits
        it, for the same reason it emits it for an ordinary string column: the
        view layout is what a string column already is, and writing offsets
        would mean building a second copy of every category on the way out.
    """
    var encoding = field_table(data, field, 4)
    # A dictionary's own type is the type of its categories, and those are a
    # string column. A nested one comes back as `+l` or `+s` and is refused by
    # the check below along with every other type that is not text, so the
    # children this fills in are thrown away with it.
    var inside = List[_Child]()
    var format = _format_for(data, field, name, column, ROOT, inside)
    if format != "u" and format != "U" and format != "vu":
        raise Error(
            String(
                "arrow ipc: column '",
                name,
                "' is dictionary encoded over ",
                format,
                (
                    " values, and firepanda's categories are strings the way"
                    " pandas' are"
                ),
            )
        )
    # Field 3 is the dictionary kind, and DenseArray is the only one Arrow has
    # ever defined. Checking it costs nothing and means a file written against a
    # later spec is refused rather than read as though the extra kind did not
    # change anything.
    var kind = Int(field_scalar[DType.int16](data, encoding, 3, 0))
    if kind != 0:
        raise Error(
            String(
                "arrow ipc: column '",
                name,
                "' has dictionary kind ",
                kind,
                ", and the dense array is the only kind there is",
            )
        )
    var index = field_table(data, encoding, 1)
    # Absent, the index type is a signed int32, which is the flatbuffer comment's
    # default rather than a table default, so it has to be spelled here.
    var codes = String("i")
    if index >= 0:
        codes = _int_format(data, index, name)
    return _Encoding(
        column,
        field_scalar[DType.int64](data, encoding, 0, 0),
        format,
        codes,
        Bool(field_scalar[DType.bool](data, encoding, 2, False)),
    )


def read_schema(
    data: Span[UInt8, _],
    schema: Int,
    mut encodings: List[_Encoding],
    mut children: List[_Child],
) raises -> ArrowLayout:
    """Reads a schema message's header table.

    Args:
        data: The buffer holding the message.
        schema: The position of the `Schema` table.
        encodings: Filled with one entry per dictionary encoded column. The
            categories are not in the schema and arrive in their own messages,
            so what the caller gets here is what to expect rather than what to
            use.
        children: Filled with one entry per node below a nested column, in
            Arrow's own order, and left empty by a schema of flat columns.

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

    var out = ArrowLayout()
    var fields = field_vector(data, schema, 1)
    if fields < 0:
        return out^
    for i in range(vector_length(data, fields)):
        var field = vector_table(data, fields, i)
        var name = field_string(data, field, 0)
        out.names.append(name)
        if field_table(data, field, 4) >= 0:
            var encoding = _dictionary_for(data, field, name, i)
            out.formats.append(encoding.index)
            encodings.append(encoding^)
        else:
            out.formats.append(
                _format_for(data, field, name, i, ROOT, children)
            )
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
    if format == STRUCT_FORMAT:
        # A struct owns nothing but the bitmap saying which of its rows are
        # there. Every value is in a field, and every field is a node of its own
        # further along the batch.
        return 1
    if format == "u" or format == "U" or format == "z" or format == "Z":
        return 3
    return 2


struct _ColumnRef(Movable):
    """Where one column of one record batch keeps its buffers.

    Decoding a batch is arithmetic over the metadata and it produces pointers
    into the body, not data. Holding those separately from the copy is what makes
    it possible to look at every batch in a file before allocating anything.

    The buffer array is kept as a raw pointer as well as as a list, because an
    `ArrowArray` wants an address and the list's own pointer is only reachable
    through a mutable reference, which a task running beside fifteen others does
    not have. The address is taken once, while the list is still a local, and the
    list's heap block does not move when the list does.
    """

    var pointers: List[NullableVoidPtr]
    """One entry per buffer, in the order the importer expects."""

    var sizes: List[Int64]
    """The data buffer lengths a view column needs, which IPC keeps in its
    buffer entries where the C interface keeps them in a buffer of their own."""

    var buffers: Pointer[NullableVoidPtr, MutUntrackedOrigin]
    """The address of `pointers`, taken before the list was moved in."""

    var length: Int
    """The batch's row count, which the schema says is every column's."""

    var nulls: Int
    """How many of those rows are null."""

    def __init__(
        out self,
        var pointers: List[NullableVoidPtr],
        var sizes: List[Int64],
        length: Int,
        nulls: Int,
    ):
        """Takes ownership of one column's buffer pointers.

        Args:
            pointers: The buffer array.
            sizes: The data buffer lengths, empty for a column with no views.
            length: The row count.
            nulls: The null count.
        """
        self.buffers = pointers.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        self.pointers = pointers^
        self.sizes = sizes^
        self.length = length
        self.nulls = nulls

    def array(self) raises -> ArrowArray:
        """Builds the `ArrowArray` the importer reads for this column.

        Returns:
            An array pointing into the message body, owning nothing. The
            assembler narrows it to a range of rows and never releases it, which
            is why the release callback stays null.
        """
        var out = ArrowArray()
        out.length = Int64(self.length)
        out.null_count = Int64(self.nulls)
        out.n_buffers = Int64(len(self.pointers))
        out.buffers = self.buffers
        return out^


@fieldwise_init
struct _Step(Copyable, Movable):
    """One field node of the schema, in the order a record batch writes them.

    Arrow's node vector is the schema walked depth first, so a frame of a plain
    int64 column and a struct of two fields has four nodes and not two. This is
    that walk, worked out once from the schema and then used to read every batch.
    """

    var name: String
    """The name, for an error message."""

    var format: String
    """The C Data Interface format string for this node's own type."""

    var column: Int
    """The top level column this node belongs to."""

    var child: Int
    """The position in the children list, or -1 when the node is the column
    itself."""


def _steps(schema: ArrowLayout, children: List[_Child]) -> List[_Step]:
    """Works out the node order a batch of this schema is written in.

    Args:
        schema: The schema the stream declared.
        children: The nodes below the nested columns, in Arrow's order.

    Returns:
        One entry per field node, columns in order and each column's own nodes
        following it.
    """
    var out = List[_Step]()
    for c in range(len(schema)):
        out.append(
            _Step(String(schema.names[c]), String(schema.formats[c]), c, -1)
        )
        for j in range(len(children)):
            if children[j].column == c:
                out.append(
                    _Step(
                        String(children[j].name),
                        String(children[j].format),
                        c,
                        j,
                    )
                )
    return out^


def _decode_batch(
    data: Span[UInt8, _], steps: List[_Step], message: IpcMessage
) raises -> List[_ColumnRef]:
    """Reads one record batch's metadata and locates every buffer in its body.

    Nothing is copied here. What comes out is one `_ColumnRef` per field node,
    each an `ArrowArray` in everything but where it came from.

    Args:
        data: The buffer holding the message.
        steps: The node order the schema implies.
        message: The framed message, already located.

    Returns:
        One entry per field node, in the same order as `steps`.

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
    if node_count != len(steps):
        raise Error(
            String(
                "arrow ipc: a batch with ",
                node_count,
                " field nodes against a schema of ",
                len(steps),
            )
        )

    var out = List[_ColumnRef](capacity=len(steps))
    var taken = 0
    var variadic_taken = 0
    for i in range(len(steps)):
        ref format = steps[i].format
        var node = vector_element(data, nodes, i, NODE_SIZE)
        var length = Int(read_scalar[DType.int64](data, node))
        var nulls = Int(read_scalar[DType.int64](data, node + 8))
        # A column has the batch's row count and a node inside one need not. A
        # list's element column holds every element of every row and is as long
        # as the last offset says, so the check is on the columns alone and the
        # length of a child is checked against its parent's offsets instead,
        # where the two can be compared.
        if steps[i].child < 0 and length != rows:
            raise Error(
                String(
                    "arrow ipc: column '",
                    steps[i].name,
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
                    steps[i].name,
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
                            steps[i].name,
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

        out.append(_ColumnRef(pointers^, sizes^, length, nulls))

    return out^


def _read_dictionary(
    data: Span[UInt8, _],
    message: IpcMessage,
    encodings: List[_Encoding],
    mut layout: ArrowLayout,
) raises:
    """Reads one dictionary batch and gives its categories to their columns.

    A dictionary batch is a record batch of one column wearing an id, so the
    ordinary batch decoder does the work and the only thing added here is
    finding out whose categories these are.

    Args:
        data: The buffer holding the message.
        message: The framed dictionary batch message.
        encodings: What the schema said about every dictionary column.
        layout: The layout the categories are attached to.

    Raises:
        Error: If the batch is a delta, if its id belongs to no column, if the
            categories arrive twice for one id, or if the batch is malformed.
    """
    var header = message.header
    if header < 0:
        raise Error("arrow ipc: a dictionary batch message with no batch in it")
    var id = field_scalar[DType.int64](data, header, 0, 0)
    if Bool(field_scalar[DType.bool](data, header, 2, False)):
        raise Error(
            String(
                "arrow ipc: the categories of dictionary ",
                id,
                (
                    " arrive as a delta, and firepanda reads a dictionary that"
                    " comes whole"
                ),
            )
        )

    var wanted = List[Int]()
    var format = String()
    var ordered = False
    for i in range(len(encodings)):
        if encodings[i].id != id:
            continue
        # Two columns may name one dictionary and then they share it, so this
        # gathers every column rather than stopping at the first.
        if layout.dictionary_at(encodings[i].column) >= 0:
            raise Error(
                String(
                    "arrow ipc: the categories of dictionary ",
                    id,
                    (
                        " arrive twice, and every batch of one frame has to"
                        " read against the same categories"
                    ),
                )
            )
        wanted.append(encodings[i].column)
        format = encodings[i].values
        ordered = encodings[i].ordered
    if len(wanted) == 0:
        raise Error(
            String(
                "arrow ipc: a dictionary batch carries id ",
                id,
                ", which no column in the schema asked for",
            )
        )

    var mini = ArrowLayout()
    mini.names.append(String("categories"))
    mini.formats.append(format)
    mini.nullable.append(True)

    # The message header is the `DictionaryBatch` table and the batch decoder
    # wants the `RecordBatch` inside it, which is field 1.
    var inner = message.copy()
    inner.header = field_table(data, header, 1)
    var refs = _decode_batch(data, _steps(mini, List[_Child]()), inner)
    var column = build_column(refs[0].array(), format)
    _ = refs^

    var categories = column^.into_strings()
    for i in range(len(wanted)):
        layout.dictionaries.append(
            ArrowDictionary(wanted[i], ordered, StringArray(copy=categories))
        )


def _check_dictionaries(encodings: List[_Encoding], layout: ArrowLayout) raises:
    """Raises unless every dictionary column got its categories.

    A column whose categories never arrived would otherwise assemble as a plain
    integer column of codes, which is the one failure mode of this whole path
    that produces an answer rather than an error.

    Args:
        encodings: What the schema said about every dictionary column.
        layout: The layout after every dictionary batch has been read.

    Raises:
        Error: If a declared dictionary column has no categories.
    """
    for i in range(len(encodings)):
        if layout.dictionary_at(encodings[i].column) < 0:
            raise Error(
                String(
                    "arrow ipc: column '",
                    layout.names[encodings[i].column],
                    "' is dictionary ",
                    encodings[i].id,
                    ", and no dictionary batch carried its categories",
                )
            )


def _is_nested_format(format: StringSlice) -> Bool:
    """Says whether a format string is one of the two nested types.

    Args:
        format: A C Data Interface format string.

    Returns:
        True for a list, a large list or a struct.
    """
    return (
        format == STRUCT_FORMAT
        or format == LIST_FORMAT
        or format == LARGE_LIST_FORMAT
    )


def _column_step(steps: List[_Step], column: Int) -> Int:
    """Finds the node that is a column itself rather than something inside one.

    Args:
        steps: The node order the schema implies.
        column: The column position.

    Returns:
        The position in `steps`, or -1 if the schema has no such column.
    """
    for i in range(len(steps)):
        if steps[i].column == column and steps[i].child < 0:
            return i
    return -1


def _child_steps(
    steps: List[_Step], children: List[_Child], at: Int
) -> List[Int]:
    """Finds the nodes hanging directly off one node.

    Args:
        steps: The node order the schema implies.
        children: The nodes below the nested columns, in Arrow's order.
        at: The position in `steps` of the node to look under.

    Returns:
        The positions in `steps` of its children, left to right.
    """
    var out = List[Int]()
    for i in range(len(steps)):
        if steps[i].column != steps[at].column or steps[i].child < 0:
            continue
        if children[steps[i].child].parent == steps[at].child:
            out.append(i)
    return out^


def _build_nodes(
    refs: List[_ColumnRef],
    steps: List[_Step],
    children: List[_Child],
    at: Int,
    parent: Int,
    mut nodes: List[NestedNode],
) raises:
    """Copies one node of a nested column and then everything below it.

    The node is appended before its children are, so the list comes out in the
    same pre-order the batch wrote its field nodes in, which is the order a
    finished column holds its children in and the order a writer walks them in
    again.

    Args:
        refs: The decoded batch, one entry per field node.
        steps: The node order the schema implies.
        children: The nodes below the nested columns, in Arrow's order.
        at: The position in `steps` of the node to build.
        parent: The position in `nodes` of the node it hangs off, or `ROOT`.
        nodes: Filled with this node and every node below it.

    Raises:
        Error: If a list does not have exactly one child, if a struct's field
            has a different row count from the struct, or if a buffer inside is
            malformed.
    """
    var kids = _child_steps(steps, children, at)
    var here = len(nodes)
    ref format = steps[at].format
    var rows = refs[at].length

    if format == STRUCT_FORMAT:
        for i in range(len(kids)):
            if refs[kids[i]].length != rows:
                raise Error(
                    String(
                        "arrow ipc: field '",
                        steps[kids[i]].name,
                        "' has ",
                        refs[kids[i]].length,
                        " rows in a struct of ",
                        rows,
                    )
                )
        nodes.append(
            NestedNode(
                String(steps[at].name),
                LogicalType.STRUCT,
                parent,
                ColumnData(
                    Buffer(0), import_validity(refs[at].array(), rows), rows
                ),
            )
        )
    elif format == LIST_FORMAT or format == LARGE_LIST_FORMAT:
        if len(kids) != 1:
            raise Error(
                String(
                    "arrow ipc: '",
                    steps[at].name,
                    "' is a list with ",
                    len(kids),
                    " children, and a list has one",
                )
            )
        var elements = refs[kids[0]].length
        var index = DType.int32 if format == LIST_FORMAT else DType.int64
        var data: ColumnData
        if format == LIST_FORMAT:
            data = import_list_offsets[DType.int32](
                refs[at].array(), rows, elements
            )
        else:
            data = import_list_offsets[DType.int64](
                refs[at].array(), rows, elements
            )
        nodes.append(
            NestedNode(
                String(steps[at].name),
                LogicalType.list_of(index),
                parent,
                data^,
            )
        )
    else:
        var leaf = build_column(refs[at].array(), format)
        nodes.append(leaf^.into_node(String(steps[at].name), parent))

    for i in range(len(kids)):
        _build_nodes(refs, steps, children, kids[i], here, nodes)


def _nested_column(
    refs: List[_ColumnRef],
    steps: List[_Step],
    children: List[_Child],
    at: Int,
) raises -> AnyArray:
    """Builds one whole list or struct column out of a decoded batch.

    Args:
        refs: The decoded batch, one entry per field node.
        steps: The node order the schema implies.
        children: The nodes below the nested columns, in Arrow's order.
        at: The position in `steps` of the column.

    Returns:
        The column, children and all.

    Raises:
        Error: If the batch does not describe the shape the schema declared, or
            a buffer inside is malformed.
    """
    var nodes = List[NestedNode]()
    _build_nodes(refs, steps, children, at, ROOT, nodes)
    return AnyArray.nested_from(nodes^)


def _arrays(
    refs: List[List[_ColumnRef]], steps: List[_Step]
) raises -> List[List[ArrowArray]]:
    """Turns the decoded batches into the arrays the assembler takes.

    The refs are one per field node and the assembler works in columns, so this
    is also where the nodes inside a nested column are left behind. Those are
    read whole elsewhere and the array standing in for one here is only ever
    asked its row count.

    The refs have to outlive the call this feeds, because each array points at
    the buffer list its ref owns and nothing in the array says so.

    Args:
        refs: One decoded batch per record batch, in order.
        steps: The node order the schema implies.

    Returns:
        The same thing as Arrow arrays, one per column of each batch.
    """
    var roots = List[Int]()
    for i in range(len(steps)):
        if steps[i].child < 0:
            roots.append(i)

    var out = List[List[ArrowArray]](capacity=len(refs))
    for b in range(len(refs)):
        var row = List[ArrowArray](capacity=len(roots))
        for c in range(len(roots)):
            row.append(refs[b][roots[c]].array())
        out.append(row^)
    return out^


def _assemble(
    schema: ArrowLayout,
    steps: List[_Step],
    children: List[_Child],
    var refs: List[List[_ColumnRef]],
) raises -> DataFrame:
    """Builds the frame from every batch of a file.

    Args:
        schema: The schema the stream declared.
        steps: The node order the schema implies.
        children: The nodes below the nested columns, in Arrow's order.
        refs: One decoded batch per record batch message, in order.

    Returns:
        The frame.

    Raises:
        Error: If a column holds a type firepanda cannot read, if a nested
            column arrives in more than one batch, or if a buffer is malformed.
    """
    var nested = List[ArrowNested]()
    for c in range(len(schema)):
        var at = _column_step(steps, c)
        if not _is_nested_format(steps[at].format):
            continue
        if len(refs) != 1:
            # Joining two batches of a nested column means shifting the second
            # one's offsets by the first one's element count and stacking the
            # children under them, at every level. Refused by name until that is
            # written, rather than read as a frame missing most of its rows.
            raise Error(
                String(
                    "arrow ipc: column '",
                    schema.names[c],
                    "' is a nested type spread across ",
                    len(refs),
                    (
                        " record batches, and firepanda reads a nested column"
                        " that arrives in one"
                    ),
                )
            )
        nested.append(
            ArrowNested(c, _nested_column(refs[0], steps, children, at))
        )

    var arrays = _arrays(refs, steps)
    var frame = assemble(schema, arrays, nested^)
    # The refs own the buffer lists the arrays point at, and their last use above
    # is where they would otherwise be destroyed.
    _ = refs^
    return frame^


def _empty_refs(
    data: Span[UInt8, _], steps: List[_Step]
) raises -> List[_ColumnRef]:
    """Builds a batch of no rows, one entry per field node.

    Every buffer points at the start of the message and every length is zero, so
    nothing here is ever read. It exists so that a stream with no batches goes
    through the same column builder as a stream with one, which is what keeps an
    empty frame's types the types a full one would have had.

    Args:
        data: Any buffer at least eight bytes long, used only for an address no
            read will reach.
        steps: The node order the schema implies.

    Returns:
        One empty ref per field node.
    """
    var out = List[_ColumnRef](capacity=len(steps))
    for i in range(len(steps)):
        ref format = steps[i].format
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
        out.append(_ColumnRef(pointers^, sizes^, 0, 0))
    return out^


def _empty_frame(
    data: Span[UInt8, _],
    schema: ArrowLayout,
    steps: List[_Step],
    children: List[_Child],
) raises -> DataFrame:
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
        steps: The node order the schema implies.
        children: The nodes below the nested columns, in Arrow's order.

    Returns:
        An empty frame with the schema's columns.
    """
    var refs = _empty_refs(data, steps)
    var fields = List[Field](capacity=len(schema))
    var columns = List[AnyArray](capacity=len(schema))
    for i in range(len(schema)):
        var at = _column_step(steps, i)
        var column: AnyArray
        if _is_nested_format(steps[at].format):
            column = _nested_column(refs, steps, children, at)
        else:
            column = build_column(refs[at].array(), steps[at].format)

        # A stream may carry its categories and then no rows at all, and the
        # answer to that is a categorical column of no rows rather than an
        # integer one. There are no codes to range check, so this is only the
        # attachment.
        var have = schema.dictionary_at(i)
        if have >= 0:
            attach_dictionary(
                column, schema.dictionaries[have], schema.names[i]
            )

        var field = Field(schema.names[i], column.type)
        field.nullable = schema.nullable[i]
        fields.append(field^)
        columns.append(column^)

    # The refs own the buffer lists the arrays above pointed at.
    _ = refs^
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
    var encodings = List[_Encoding]()
    var children = List[_Child]()
    var schema = read_schema(data, message.header, encodings, children)
    var steps = _steps(schema, children)

    var batches = List[List[_ColumnRef]]()
    var pos = message.next
    while True:
        var next = read_message(data, pos)
        if next.header_type == MESSAGE_NONE:
            break
        if next.header_type == MESSAGE_DICTIONARY_BATCH:
            _read_dictionary(data, next, encodings, schema)
            pos = next.next
            continue
        if next.header_type != MESSAGE_RECORD_BATCH:
            raise Error(
                String(
                    "arrow ipc: a message of kind ",
                    next.header_type,
                    " where a record batch was expected",
                )
            )
        batches.append(_decode_batch(data, steps, next))
        pos = next.next

    _check_dictionaries(encodings, schema)
    if len(batches) == 0:
        return _empty_frame(data, schema, steps, children)
    return _assemble(schema, steps, children, batches^)


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
    var encodings = List[_Encoding]()
    var children = List[_Child]()
    var schema = read_schema(
        data, field_table(data, footer, 1), encodings, children
    )
    var steps = _steps(schema, children)

    # Field 2 is the dictionaries, in their own list of blocks ahead of the
    # record batches in field 3. A file is not walked in order, so the categories
    # are fetched by the footer saying where they are rather than by meeting them
    # on the way past, and a file with no dictionary columns has an empty vector
    # here and does no work.
    var dictionaries = field_vector(data, footer, 2)
    var dictionary_count = 0 if dictionaries < 0 else vector_length(
        data, dictionaries
    )
    for i in range(dictionary_count):
        var block = vector_element(data, dictionaries, i, BLOCK_SIZE)
        var at = Int(read_scalar[DType.int64](data, block))
        var message = read_message(data, at)
        if message.header_type != MESSAGE_DICTIONARY_BATCH:
            raise Error(
                String(
                    (
                        "arrow ipc: the footer's dictionaries point at a"
                        " message of kind "
                    ),
                    message.header_type,
                    " where a dictionary batch should be",
                )
            )
        _read_dictionary(data, message, encodings, schema)
    _check_dictionaries(encodings, schema)

    var blocks = field_vector(data, footer, 3)
    var count = 0 if blocks < 0 else vector_length(data, blocks)
    if count == 0:
        return _empty_frame(data, schema, steps, children)

    var batches = List[List[_ColumnRef]](capacity=count)
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
        batches.append(_decode_batch(data, steps, message))

    return _assemble(schema, steps, children, batches^)


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
