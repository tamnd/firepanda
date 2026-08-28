"""Turning a frame back into CSV bytes.

The writer's only real decisions are when to quote and how to spell a float, and
both are decided by what a reader will do with the output rather than by what
looks tidy.

A field is quoted when leaving it bare would change where the boundaries are,
which is when it contains the delimiter, a quote, a newline or a carriage
return, and not otherwise. Quoting everything is legal and is what a few writers
do, and it costs two bytes per field on a file that is mostly numbers.

A float is written at enough digits to read back as the same float. The display
layer rounds to something a person can scan down a column, which is right there
and wrong here: a file is going to be read by a machine, and a value that does
not survive the round trip is a value the writer lost.

A null is an empty field. That is what pandas writes and what this reader reads
back as a null, so `read_csv(write_csv(frame))` returns the nulls it started
with. A frame holding the empty string in a string column writes `""`, which is
the only way to tell the two apart in a CSV file, and the reader honours it.
"""

from std.collections.span import Span

from firepanda.array import AnyArray
from firepanda.dtype import ALL
from firepanda.frame import DataFrame

from .scan import Dialect, default_dialect, NEWLINE


struct WriteOptions(ImplicitlyCopyable, Movable):
    """Everything about a write that is not the frame itself."""

    var dialect: Dialect
    """The delimiter and quote character."""

    var write_header: Bool
    """Whether to write the column names as the first row."""

    def __init__(out self):
        """Constructs the defaults: comma, double quote, header."""
        self.dialect = default_dialect()
        self.write_header = True

    def __init__(out self, dialect: Dialect, write_header: Bool):
        """Constructs options explicitly.

        Args:
            dialect: The delimiter and quote character.
            write_header: Whether to write a header row.
        """
        self.dialect = dialect
        self.write_header = write_header


def needs_quoting(text: String, dialect: Dialect) -> Bool:
    """Returns whether writing a value bare would change the field boundaries.

    Args:
        text: The value as it would be written.
        dialect: The delimiter and quote character.

    Returns:
        True if the value has to be quoted.
    """
    # A null is written as nothing and never reaches here, so an empty value at
    # this point is the empty string, and `""` is the only spelling of it that
    # reads back as itself.
    if text.byte_length() == 0:
        return True
    for byte in text.as_bytes():
        if (
            byte == dialect.delimiter
            or byte == dialect.quote
            or byte == NEWLINE
            or byte == 13
        ):
            return True
    return False


def quote_into(mut out: List[UInt8], text: String, dialect: Dialect):
    """Appends one field, quoting and escaping it only if it needs it.

    Args:
        out: The buffer to append to.
        text: The value as it should be written.
        dialect: The delimiter and quote character.
    """
    if not needs_quoting(text, dialect):
        out.extend(text.as_bytes())
        return
    out.append(dialect.quote)
    for byte in text.as_bytes():
        if byte == dialect.quote:
            out.append(dialect.quote)
        out.append(byte)
    out.append(dialect.quote)


def cell_text(column: AnyArray, i: Int) raises -> String:
    """Renders one value the way a file should carry it.

    Args:
        column: The column.
        i: The row.

    Returns:
        The value's text, empty for a null.

    Raises:
        Error: If the column holds a type the writer cannot spell.
    """
    if not column.is_valid(i):
        return String("")
    if column.is_string():
        return column.strings()[i]
    if column.dtype() == DType.bool:
        var value = (
            column.unsafe_ptr[DType.bool]().unsafe_offset(i).unsafe_load()
        )
        return String("true") if value else String("false")
    comptime for candidate in ALL:
        if column.dtype() == candidate:
            return String(
                column.unsafe_ptr[candidate]().unsafe_offset(i).unsafe_load()
            )
    raise Error(String("no way to write a ", column.type, " to CSV"))


def write_csv_bytes(
    frame: DataFrame, options: WriteOptions
) raises -> List[UInt8]:
    """Renders a whole frame as CSV bytes.

    Args:
        frame: The frame.
        options: The write options.

    Returns:
        The bytes.

    Raises:
        Error: If the frame holds a type the writer cannot spell.
    """
    var out = List[UInt8]()
    if options.write_header:
        for column in range(frame.width()):
            if column > 0:
                out.append(options.dialect.delimiter)
            quote_into(out, frame.names()[column], options.dialect)
        out.append(NEWLINE)

    for row in range(len(frame)):
        for column in range(frame.width()):
            if column > 0:
                out.append(options.dialect.delimiter)
            # A null writes as nothing at all, so it skips the quoting path
            # rather than passing an empty string through it, because a quoted
            # empty field reads back as the empty string and not as a null.
            if not frame[column].is_valid(row):
                continue
            quote_into(out, cell_text(frame[column], row), options.dialect)
        out.append(NEWLINE)
    return out^


def write_csv(frame: DataFrame, path: String, options: WriteOptions) raises:
    """Writes a frame to a file.

    Args:
        frame: The frame.
        path: The file to write.
        options: The write options.

    Raises:
        Error: If the file cannot be written, or the frame holds a type the
            writer cannot spell.
    """
    var bytes = write_csv_bytes(frame, options)
    var handle = open(path, "w")
    handle.write_bytes(Span(bytes))
    handle.close()


def write_csv(frame: DataFrame, path: String) raises:
    """Writes a frame to a file with the default options.

    Args:
        frame: The frame.
        path: The file to write.

    Raises:
        Error: If the file cannot be written, or the frame holds a type the
            writer cannot spell.
    """
    write_csv(frame, path, WriteOptions())
