"""Turning CSV bytes into a frame.

The two halves underneath this one, `scan.mojo` and `parse.mojo`, do not know
what a column is. This file is where that knowledge lives: which type each column
should be, and the loop that fills it.

Inference is a ladder and it only ever climbs. A column starts at boolean, and
the first field that is not a boolean moves it to integer, then to float, then to
string, and it never moves back down. That ordering is the whole design. A column
read as an integer because the first thousand rows happened to be integers is a
read that fails on row one thousand and one, and the alternatives to failing are
both bad: raise, and a file that pandas reads becomes a file firepanda refuses,
or silently null the value, and the frame quietly loses data. So the climb
happens during inference, over a sample the caller chooses the size of, and the
default sample is the whole file. Reading twice is the price of never being
wrong about a type, and the second pass is over offsets rather than over text.

A caller who knows the types can say so and skip inference entirely, which is
both faster and the only way to force a column wider than its values need.

Missing fields never constrain the type. A column of a hundred integers and one
`NA` is an integer column with one null in it, which is a thing this frame can
represent and a pandas frame cannot, and the difference is the reason the
validity bitmap exists.
"""

from std.collections.span import Span

from firepanda.array import Array, AnyArray, StringArray, StringBuilder
from firepanda.dtype import Field, LogicalType, Schema
from firepanda.frame import DataFrame

from .parse import is_missing, parse_bool, parse_float, parse_int
from .scan import (
    Dialect,
    FieldSpan,
    Scan,
    default_dialect,
    field_bytes,
    scan_csv,
    unescape,
)


comptime INFER_ALL = 0
"""Passed as `infer_rows` to look at every row before deciding a type."""


struct ReadOptions(ImplicitlyCopyable, Movable):
    """Everything about a read that is not the bytes themselves."""

    var dialect: Dialect
    """The delimiter and quote character."""

    var has_header: Bool
    """Whether the first row names the columns rather than carrying values."""

    var infer_rows: Int
    """How many value rows to look at before deciding types, 0 for all of them.

    A file whose types are stable in the first few thousand rows reads faster
    with a bound here, and a file that is not stable is exactly the file the
    bound gets wrong. The default is all of them for that reason.
    """

    def __init__(out self):
        """Constructs the defaults: comma, quote, header, infer over everything.
        """
        self.dialect = default_dialect()
        self.has_header = True
        self.infer_rows = INFER_ALL

    def __init__(out self, dialect: Dialect, has_header: Bool, infer_rows: Int):
        """Constructs options explicitly.

        Args:
            dialect: The delimiter and quote character.
            has_header: Whether row zero is a header.
            infer_rows: Value rows to sample, or `INFER_ALL`.
        """
        self.dialect = dialect
        self.has_header = has_header
        self.infer_rows = infer_rows


def infer_column(
    data: Span[UInt8, _],
    scan: Scan,
    column: Int,
    first_row: Int,
    last_row: Int,
) -> LogicalType:
    """Decides one column's type by climbing the ladder over a row range.

    Args:
        data: The buffer the scan points into.
        scan: The scanned fields.
        column: Which column to look at.
        first_row: The first row to sample, past any header.
        last_row: One past the last row to sample.

    Returns:
        The narrowest type every sampled value fits in, or string if none does.
    """
    var kind = 0
    var saw_value = False
    for row in range(first_row, last_row):
        if column >= scan.width(row):
            continue
        var span = scan.at(row, column)
        var bytes = field_bytes(data, span)
        if is_missing(bytes):
            continue
        saw_value = True
        # An escaped field has a quote in it, so it is text whatever its
        # remaining bytes look like, and unescaping it here to find that out
        # would allocate on a path that runs once per value in the file.
        if span.escaped:
            return LogicalType.STRING
        if kind == 0 and parse_bool(bytes).ok:
            continue
        if kind <= 1 and parse_int[DType.int64](bytes).ok:
            kind = 1
            continue
        if kind <= 2 and parse_float[DType.float64](bytes).ok:
            kind = 2
            continue
        return LogicalType.STRING
    if not saw_value:
        # No value was observed, so nothing argues for a number. A string column
        # of all nulls can be cast to anything later; a float column of all
        # nulls has already thrown the text away.
        return LogicalType.STRING
    if kind == 0:
        return LogicalType.BOOL
    if kind == 1:
        return LogicalType.INT64
    return LogicalType.FLOAT64


def infer_schema(
    data: Span[UInt8, _], scan: Scan, options: ReadOptions
) raises -> Schema:
    """Names and types every column of a scanned buffer.

    Args:
        data: The buffer the scan points into.
        scan: The scanned fields.
        options: The read options.

    Returns:
        The schema.

    Raises:
        Error: If the rows disagree on how many fields they have.
    """
    if scan.is_ragged():
        raise Error(
            "rows have different numbers of fields; a ragged file has no schema"
        )
    if len(scan) == 0:
        return Schema()

    var width = scan.width(0)
    var first_row = 1 if options.has_header else 0
    var last_row = len(scan)
    if options.infer_rows != INFER_ALL:
        var bound = first_row + options.infer_rows
        if bound < last_row:
            last_row = bound

    var fields = List[Field](capacity=width)
    for column in range(width):
        var name = String("column_", column)
        if options.has_header:
            var span = scan.at(0, column)
            if span.escaped:
                var literal = unescape(data, span, options.dialect.quote)
                name = String(StringSlice(unsafe_from_utf8=Span(literal)))
            else:
                name = String(
                    StringSlice(unsafe_from_utf8=field_bytes(data, span))
                )
        var dtype = infer_column(data, scan, column, first_row, last_row)
        fields.append(Field(name, dtype))
    return Schema(fields^)


def read_csv_bytes(
    data: Span[UInt8, _], options: ReadOptions
) raises -> DataFrame:
    """Reads a frame out of a buffer of CSV bytes.

    Args:
        data: The whole file.
        options: The read options.

    Returns:
        The frame.

    Raises:
        Error: If the file is ragged, or if a value does not fit the type the
            schema declares for its column.
    """
    var scan = scan_csv(data, options.dialect)
    var schema = infer_schema(data, scan, options)
    return build(data, scan, schema^, options)


def read_csv_bytes_as(
    data: Span[UInt8, _], var schema: Schema, options: ReadOptions
) raises -> DataFrame:
    """Reads a frame out of CSV bytes using types the caller already knows.

    Skips inference, which is the whole second pass over the file, and is also
    the only way to get a column wider than its values need or a numeric column
    read as text.

    Args:
        data: The whole file.
        schema: The types to read as, in column order.
        options: The read options.

    Returns:
        The frame.

    Raises:
        Error: If the file is ragged, if the schema does not have one field per
            column, or if a value does not fit its declared type.
    """
    var scan = scan_csv(data, options.dialect)
    if scan.is_ragged():
        raise Error(
            "rows have different numbers of fields; a ragged file has no schema"
        )
    if len(scan) > 0 and len(schema) != scan.width(0):
        raise Error(
            String(
                "schema has ",
                len(schema),
                " fields but the file has ",
                scan.width(0),
                " columns",
            )
        )
    return build(data, scan, schema^, options)


def build(
    data: Span[UInt8, _], scan: Scan, var schema: Schema, options: ReadOptions
) raises -> DataFrame:
    """Fills one column per field of the schema.

    Args:
        data: The buffer the scan points into.
        scan: The scanned fields.
        schema: The types to read as.
        options: The read options.

    Returns:
        The frame.

    Raises:
        Error: If a value does not fit its column's declared type.
    """
    var first_row = 1 if options.has_header else 0
    var rows = len(scan) - first_row
    if rows < 0:
        rows = 0

    var columns = List[AnyArray](capacity=len(schema))
    for column in range(len(schema)):
        var dtype = schema[column].dtype
        var name = schema[column].name
        if dtype == LogicalType.BOOL:
            columns.append(
                AnyArray(read_bool(data, scan, column, first_row, rows, name))
            )
        elif dtype == LogicalType.INT64:
            columns.append(
                AnyArray(
                    read_number[DType.int64, True](
                        data, scan, column, first_row, rows, name
                    )
                )
            )
        elif dtype == LogicalType.FLOAT64:
            columns.append(
                AnyArray(
                    read_number[DType.float64, False](
                        data, scan, column, first_row, rows, name
                    )
                )
            )
        elif dtype == LogicalType.STRING:
            columns.append(
                AnyArray(
                    read_text(data, scan, column, first_row, rows, options)
                )
            )
        else:
            raise Error(
                String(
                    "column '",
                    name,
                    "' asks for ",
                    dtype,
                    ", which the CSV reader does not produce yet",
                )
            )
    return DataFrame(schema^, columns^)


def read_bool(
    data: Span[UInt8, _],
    scan: Scan,
    column: Int,
    first_row: Int,
    rows: Int,
    name: String,
) raises -> Array[DType.bool]:
    """Fills one boolean column.

    Args:
        data: The buffer the scan points into.
        scan: The scanned fields.
        column: Which column to read.
        first_row: The first value row.
        rows: How many value rows there are.
        name: The column name, for the error message.

    Returns:
        The column.

    Raises:
        Error: If a value is neither missing nor a boolean.
    """
    var out = Array[DType.bool](rows)
    for i in range(rows):
        var bytes = field_bytes(data, scan.at(first_row + i, column))
        if is_missing(bytes):
            out.set_null(i)
            continue
        var parsed = parse_bool(bytes)
        if not parsed.ok:
            raise Error(bad_value(name, first_row + i, "a boolean"))
        out.set_valid(i, parsed.value)
    return out^


def read_number[
    dt: DType, is_int: Bool
](
    data: Span[UInt8, _],
    scan: Scan,
    column: Int,
    first_row: Int,
    rows: Int,
    name: String,
) raises -> Array[dt]:
    """Fills one numeric column.

    Args:
        data: The buffer the scan points into.
        scan: The scanned fields.
        column: Which column to read.
        first_row: The first value row.
        rows: How many value rows there are.
        name: The column name, for the error message.

    Returns:
        The column.

    Raises:
        Error: If a value is neither missing nor a number of this type.

    Parameters:
        dt: The dtype to read as.
        is_int: Whether to read integers rather than floats.
    """
    var out = Array[dt](rows)
    for i in range(rows):
        var bytes = field_bytes(data, scan.at(first_row + i, column))
        if is_missing(bytes):
            out.set_null(i)
            continue

        comptime if is_int:
            var parsed = parse_int[dt](bytes)
            if not parsed.ok:
                raise Error(bad_value(name, first_row + i, "an integer"))
            out.set_valid(i, parsed.value)
        else:
            var parsed = parse_float[dt](bytes)
            if not parsed.ok:
                raise Error(bad_value(name, first_row + i, "a number"))
            out.set_valid(i, parsed.value)
    return out^


def read_text(
    data: Span[UInt8, _],
    scan: Scan,
    column: Int,
    first_row: Int,
    rows: Int,
    options: ReadOptions,
) raises -> StringArray:
    """Fills one string column.

    Args:
        data: The buffer the scan points into.
        scan: The scanned fields.
        column: Which column to read.
        first_row: The first value row.
        rows: How many value rows there are.
        options: The read options, for the quote character.

    Returns:
        The column.
    """
    var builder = StringBuilder(rows)
    for i in range(rows):
        var span = scan.at(first_row + i, column)
        var bytes = field_bytes(data, span)
        if span.escaped:
            var literal = unescape(data, span, options.dialect.quote)
            builder.append(Span(literal))
        # A quoted field is a value the writer meant literally, including the
        # quoted empty string and the quoted word `NA`. That is the one place
        # quoting changes a value rather than a boundary, and it is the only way
        # a file has of saying which of the two it meant.
        elif not span.quoted and is_missing(bytes):
            builder.append_null()
        else:
            builder.append(bytes)
    return builder^.finish()


def bad_value(name: String, row: Int, wanted: String) -> String:
    """Builds the message for a value that does not fit its column.

    The row number is the file's, counting the header, because that is the
    number an editor shows.

    Args:
        name: The column name.
        row: The row in the file.
        wanted: What the column wanted, as a noun phrase.

    Returns:
        The message.
    """
    return String("column '", name, "' row ", row + 1, " is not ", wanted)


def read_csv(path: String, options: ReadOptions) raises -> DataFrame:
    """Reads a whole CSV file into a frame.

    The file is read into memory in one piece. A reader that streams blocks has
    to hold a partial row across a block boundary and cannot know the row count
    before it allocates, and neither is worth it until there is a file here that
    does not fit.

    Args:
        path: The file to read.
        options: The read options.

    Returns:
        The frame.

    Raises:
        Error: If the file cannot be read, or is not readable as CSV.
    """
    var handle = open(path, "r")
    var data = handle.read_bytes()
    handle.close()
    return read_csv_bytes(Span(data), options)


def read_csv(path: String) raises -> DataFrame:
    """Reads a whole CSV file with the default options.

    Args:
        path: The file to read.

    Returns:
        The frame.

    Raises:
        Error: If the file cannot be read, or is not readable as CSV.
    """
    return read_csv(path, ReadOptions())
