"""Reading newline delimited JSON into a frame.

One JSON object per line, which is what `read_json(lines=True)` means in pandas
and what almost every log file and every export from a document store looks
like. The scanning is `jsonscan.mojo`'s and the number parsing is `parse.mojo`'s,
so what is here is the part that turns a pile of spans into columns: working out
what the columns are, working out what type each one is, and filling them.

Three things make this reader simpler than the CSV one next door, and all three
come from JSON being self describing.

The first is that splitting the file is trivial. A literal newline inside a JSON
string is illegal, it has to be written as an escape, so every 0x0A byte in a
valid document is a record separator and there is no quote state to carry across
a block boundary. The CSV reader has a whole file, `split.mojo`, arguing about
that problem, and here the argument does not exist.

The second is that inference does not have to guess. A CSV field is bytes and
whether `1` is a number or a string is a decision; a JSON value says which one
it is. So the ladder from bool to int to float to string is climbed by looking
at each value's kind rather than by trying to parse it, and a column of mixed
kinds falls to string, which is the only type that can hold all of them.

The third is that a missing key and a null are both nulls, and neither one is an
error. A CSV row with the wrong number of fields is a broken row. An NDJSON line
that does not mention a key is an ordinary line, so the column set is the union
of every key in the file and a line that says nothing about a column gets a null
there.

Nested values are read as the text they were written as, into a string column.
firepanda has no nested column type yet, and the alternatives are worse: dropping
the column loses data silently, and flattening it invents a schema. When the
nested types arrive this is the one place that changes.

The one thing worth knowing about the shape of the work is that a column is
filled by walking its block's members and taking the ones whose ordinal matches,
which is a pass over a compact array of Int32 per column rather than a hash
lookup per value. That is the right trade up to a few dozen columns and the wrong
one for a file of five hundred, where the fill would want a slot array per row
instead. The files this is for have tens of columns, and the comment is here so
the day one of them does not, the reason is written down.
"""

from std.collections.span import Span

from firepanda.array import Array, AnyArray, StringArray, StringBuilder
from firepanda.dtype import Field, LogicalType, Schema
from firepanda.exec import parallel_for
from firepanda.frame import DataFrame
from firepanda.kernel.concat import column_ref, concat_refs_any

from .jsonscan import (
    JSON_FALSE,
    JSON_NULL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_TRUE,
    Member,
    NEWLINE,
    RETURN,
    SPACE,
    TAB,
    Value,
    scan_object,
    text_of,
    unescape,
)
from .mapped import map_file
from .parse import parse_float, parse_int
from .read import TypeGuess, block_count, combine, logical_of

comptime INFER_ALL = 0
"""Look at every line before deciding what the columns are."""


@fieldwise_init
struct NdjsonOptions(ImplicitlyCopyable, Movable):
    """What to do about the parts of an NDJSON file that are not stated."""

    var infer_rows: Int
    """How many lines to look at before fixing the columns and their types.

    `INFER_ALL` looks at all of them, which is the default because a key that
    first appears on the last line is a key, and a reader that dropped it would
    be losing data quietly. A number here is a promise the caller is making
    about the file and buys back a pass over it.
    """

    def __init__(out self):
        """Infer from the whole file."""
        self.infer_rows = INFER_ALL


struct Lines(Movable, Sized):
    """One block's worth of scanned lines.

    The members of every line in the block sit in one list, and `starts` says
    where each line's run of them begins. One list rather than a list per line
    because a list per line is an allocation per line, and a file is a lot of
    lines.
    """

    var members: List[Member]
    """Every member of every line in the block, in order."""

    var starts: List[Int]
    """Where each line's members begin, with a final entry that is the end."""

    var ordinals: List[Int32]
    """Which column each member belongs to, filled in once the columns of the
    whole file are known. Empty until then."""

    def __init__(out self):
        """An empty block."""
        self.members = List[Member]()
        self.starts = List[Int]()
        self.starts.append(0)
        self.ordinals = List[Int32]()

    def __len__(self) -> Int:
        """How many lines the block holds.

        Returns:
            The line count.
        """
        return len(self.starts) - 1


def _skip_space_to(bytes: Span[UInt8, _], var at: Int, end: Int) -> Int:
    """Moves past whitespace without leaving the block.

    Args:
        bytes: The buffer.
        at: Where to start.
        end: One past the last byte of this block.

    Returns:
        The first non whitespace offset, or `end`.
    """
    var ptr = bytes.unsafe_ptr()
    while at < end:
        var c = ptr.unsafe_offset(at).unsafe_load()
        if c != SPACE and c != TAB and c != NEWLINE and c != RETURN:
            return at
        at += 1
    return at


def scan_lines(bytes: Span[UInt8, _], start: Int, end: Int) raises -> Lines:
    """Scans every line of one block.

    Blank lines are skipped, which is what a file ending in a newline needs in
    order not to produce a phantom last row.

    Args:
        bytes: The whole file.
        start: The first byte of this block, which is a line start.
        end: One past the last byte, which is also a line start.

    Returns:
        The block's members and line boundaries.

    Raises:
        Error: If a line is not a well formed JSON object.
    """
    var out = Lines()
    var at = start
    while True:
        at = _skip_space_to(bytes, at, end)
        if at >= end:
            break
        at = scan_object(bytes, at, out.members)
        out.starts.append(len(out.members))
    return out^


def line_bounds(bytes: Span[UInt8, _], blocks: Int) -> List[Int]:
    """Cuts a buffer into blocks that each begin at a line start.

    A newline is a record separator everywhere in valid NDJSON, so this is a
    search for one byte from each target offset forward. No quote state crosses
    a boundary, which is the whole difference between this and `split.mojo`.

    Args:
        bytes: The whole file.
        blocks: How many blocks to aim for.

    Returns:
        `blocks + 1` offsets, the first zero and the last the file's length.
        Blocks may come out empty when the lines are long.
    """
    var length = len(bytes)
    var out = List[Int](capacity=blocks + 1)
    out.append(0)
    var ptr = bytes.unsafe_ptr()
    for b in range(1, blocks):
        var at = (length // blocks) * b
        if at < out[len(out) - 1]:
            at = out[len(out) - 1]
        while at < length and ptr.unsafe_offset(at).unsafe_load() != NEWLINE:
            at += 1
        if at < length:
            at += 1
        out.append(at)
    out.append(length)
    return out^


def _same(bytes: Span[UInt8, _], key: Value, name: StringSlice) raises -> Bool:
    """Compares a key's bytes to a name without allocating.

    A key is compared rather than turned into a `String` because a file is one
    key per column per line, and a `String` per member would be an allocation
    per member for an answer that is nearly always no.

    Args:
        bytes: The buffer the key points into.
        key: The key.
        name: The name to compare against.

    Returns:
        Whether they are the same text.

    Raises:
        Error: If the key holds an escape that is not one JSON has.
    """
    if not key.escaped:
        var wanted = name.as_bytes()
        if key.end - key.start != len(wanted):
            return False
        var ptr = bytes.unsafe_ptr()
        for i in range(len(wanted)):
            if ptr.unsafe_offset(key.start + i).unsafe_load() != wanted[i]:
                return False
        return True
    var literal = unescape(bytes, key)
    var wanted = name.as_bytes()
    if len(literal) != len(wanted):
        return False
    for i in range(len(wanted)):
        if literal[i] != wanted[i]:
            return False
    return True


def local_names(
    bytes: Span[UInt8, _], lines: Lines, limit: Int
) raises -> List[String]:
    """Collects the keys one block uses, in the order it first uses them.

    Args:
        bytes: The buffer the members point into.
        lines: The block.
        limit: How many lines to look at, or `INFER_ALL` for all of them.

    Returns:
        The names.

    Raises:
        Error: If a key holds an escape that is not one JSON has.
    """
    var out = List[String]()
    var rows = len(lines)
    if limit != INFER_ALL and limit < rows:
        rows = limit
    for row in range(rows):
        for m in range(lines.starts[row], lines.starts[row + 1]):
            var key = lines.members[m].key
            var known = False
            for c in range(len(out)):
                if _same(bytes, key, out[c]):
                    known = True
                    break
            if not known:
                out.append(text_of(bytes, key))
    return out^


def _rung_of(kind: Int, bytes: Span[UInt8, _], value: Value) -> TypeGuess:
    """Says how far up the ladder one value pushes its column.

    A JSON value says what kind of thing it is, so this looks rather than tries.
    The one judgement left is between an integer and a float, and that is made
    on how the number was written: a number with a point or an exponent in it is
    a float even when it happens to land on a whole number, because the file
    said so.

    Args:
        kind: The value's kind.
        bytes: The buffer.
        value: The value.

    Returns:
        The rung, and whether it counts as having seen anything.
    """
    if kind == JSON_NULL:
        return TypeGuess(0, False)
    if kind == JSON_TRUE or kind == JSON_FALSE:
        return TypeGuess(0, True)
    if kind == JSON_NUMBER:
        var ptr = bytes.unsafe_ptr()
        for i in range(value.start, value.end):
            var c = ptr.unsafe_offset(i).unsafe_load()
            if c == UInt8(46) or c == UInt8(101) or c == UInt8(69):
                return TypeGuess(2, True)
        return TypeGuess(1, True)
    return TypeGuess(3, True)


def _ordinal(
    bytes: Span[UInt8, _], key: Value, names: List[String]
) raises -> Int:
    """Finds which column a key belongs to.

    Args:
        bytes: The buffer.
        key: The key.
        names: The columns, in order.

    Returns:
        The column's position, or minus one if it is not one of them.

    Raises:
        Error: If the key holds an escape that is not one JSON has.
    """
    for c in range(len(names)):
        if _same(bytes, key, names[c]):
            return c
    return -1


def _fill_ordinals(
    bytes: Span[UInt8, _], mut lines: Lines, names: List[String]
) raises:
    """Records which column each of a block's members belongs to.

    Done once per member rather than once per member per column, because the
    fill walks the members of a block for every column it builds and comparing
    an Int32 is not comparing a name.

    Args:
        bytes: The buffer.
        lines: The block.
        names: The columns of the whole file.

    Raises:
        Error: If a key holds an escape that is not one JSON has.
    """
    lines.ordinals = List[Int32](capacity=len(lines.members))
    for m in range(len(lines.members)):
        lines.ordinals.append(
            Int32(_ordinal(bytes, lines.members[m].key, names))
        )


def _bad(name: String, row: Int, wanted: String) -> String:
    """Builds the message for a value that does not fit its column.

    Args:
        name: The column name.
        row: The line number in the file, counting from zero.
        wanted: What the column wanted, as a noun phrase.

    Returns:
        The message.
    """
    return String("column '", name, "' line ", row + 1, " is not ", wanted)


def _bools(
    bytes: Span[UInt8, _],
    lines: Lines,
    column: Int,
    name: String,
    first_line: Int,
) raises -> Array[DType.bool]:
    """Fills one boolean column for one block.

    A line that does not mention the key gets a null, which is the difference
    between this and the CSV reader: there every row has every field and a row
    that does not is a broken row.

    Args:
        bytes: The buffer.
        lines: The block.
        column: Which column to fill.
        name: The column name, for the error message.
        first_line: The line number in the file of the block's first line.

    Returns:
        The column.

    Raises:
        Error: If a value is neither null nor a boolean.
    """
    var rows = len(lines)
    var out = Array[DType.bool](rows)
    for row in range(rows):
        var found = False
        for m in range(lines.starts[row], lines.starts[row + 1]):
            if Int(lines.ordinals[m]) != column:
                continue
            var value = lines.members[m].value
            if value.kind == JSON_TRUE:
                out.set_valid(row, True)
            elif value.kind == JSON_FALSE:
                out.set_valid(row, False)
            elif value.kind == JSON_NULL:
                out.set_null(row)
            else:
                raise Error(_bad(name, first_line + row, "a boolean"))
            found = True
            break
        if not found:
            out.set_null(row)
    return out^


def _numbers[
    dt: DType, is_int: Bool
](
    bytes: Span[UInt8, _],
    lines: Lines,
    column: Int,
    name: String,
    first_line: Int,
) raises -> Array[dt]:
    """Fills one numeric column for one block.

    Args:
        bytes: The buffer.
        lines: The block.
        column: Which column to fill.
        name: The column name, for the error message.
        first_line: The line number in the file of the block's first line.

    Returns:
        The column.

    Raises:
        Error: If a value is neither null nor a number of this type.

    Parameters:
        dt: The dtype to read as.
        is_int: Whether to read integers rather than floats.
    """
    var rows = len(lines)
    var out = Array[dt](rows)
    for row in range(rows):
        var found = False
        for m in range(lines.starts[row], lines.starts[row + 1]):
            if Int(lines.ordinals[m]) != column:
                continue
            var value = lines.members[m].value
            if value.kind == JSON_NULL:
                out.set_null(row)
            elif value.kind != JSON_NUMBER:
                raise Error(
                    _bad(
                        name,
                        first_line + row,
                        "a whole number" if is_int else "a number",
                    )
                )
            else:
                var span = bytes[value.start : value.end]

                comptime if is_int:
                    var parsed = parse_int[dt](span)
                    if not parsed.ok:
                        raise Error(
                            _bad(name, first_line + row, "a whole number")
                        )
                    out.set_valid(row, parsed.value)
                else:
                    var parsed = parse_float[dt](span)
                    if not parsed.ok:
                        raise Error(_bad(name, first_line + row, "a number"))
                    out.set_valid(row, parsed.value)
            found = True
            break
        if not found:
            out.set_null(row)
    return out^


def _texts(
    bytes: Span[UInt8, _], lines: Lines, column: Int
) raises -> StringArray:
    """Fills one string column for one block.

    Everything that is not a string is taken as the text it was written as,
    which is how a column of mixed kinds and a column of nested values both
    come out readable rather than empty. A null is a null.

    Args:
        bytes: The buffer.
        lines: The block.
        column: Which column to fill.

    Returns:
        The column.

    Raises:
        Error: If a string holds an escape that is not one JSON has.
    """
    var rows = len(lines)
    var builder = StringBuilder(rows)
    for row in range(rows):
        var found = False
        for m in range(lines.starts[row], lines.starts[row + 1]):
            if Int(lines.ordinals[m]) != column:
                continue
            var value = lines.members[m].value
            if value.kind == JSON_NULL:
                builder.append_null()
            elif value.kind == JSON_STRING and value.escaped:
                var literal = unescape(bytes, value)
                builder.append(Span(literal))
            else:
                builder.append(bytes[value.start : value.end])
            found = True
            break
        if not found:
            builder.append_null()
    return builder^.finish()


def _column_of(
    bytes: Span[UInt8, _],
    lines: Lines,
    column: Int,
    field: Field,
    first_line: Int,
) raises -> AnyArray:
    """Fills one column of one block, as whatever type it turned out to be.

    Args:
        bytes: The buffer.
        lines: The block.
        column: Which column to fill.
        field: The column's name and type.
        first_line: The line number in the file of the block's first line.

    Returns:
        The column.

    Raises:
        Error: If a value does not fit the type, or if the type is one the
            NDJSON reader cannot produce.
    """
    if field.dtype == LogicalType.BOOL:
        return AnyArray(_bools(bytes, lines, column, field.name, first_line))
    if field.dtype == LogicalType.INT64:
        return AnyArray(
            _numbers[DType.int64, True](
                bytes, lines, column, field.name, first_line
            )
        )
    if field.dtype == LogicalType.FLOAT64:
        return AnyArray(
            _numbers[DType.float64, False](
                bytes, lines, column, field.name, first_line
            )
        )
    if field.dtype == LogicalType.STRING:
        return AnyArray(_texts(bytes, lines, column))
    raise Error(
        String(
            "column '",
            field.name,
            "' asks for ",
            field.dtype,
            ", which the NDJSON reader does not produce yet",
        )
    )


def read_ndjson_bytes(
    bytes: Span[UInt8, _], options: NdjsonOptions
) raises -> DataFrame:
    """Reads a buffer of newline delimited JSON into a frame.

    Args:
        bytes: The whole document.
        options: How much of it to look at before deciding the columns.

    Returns:
        The frame, one row per line and one column per key anywhere in the file.

    Raises:
        Error: If a line is not a well formed JSON object, or a value does not
            fit the type its column turned out to be.
    """
    var blocks = block_count(len(bytes))
    var bounds = line_bounds(bytes, blocks)

    var scans = List[Lines](capacity=blocks)
    for _ in range(blocks):
        scans.append(Lines())

    def one(b: Int) raises {mut scans, imm}:
        scans[b] = scan_lines(bytes, bounds[b], bounds[b + 1])

    parallel_for(one, blocks)

    # Where each block's lines land in the file, which is what an error message
    # quotes and what bounds the sample when the caller asked for one.
    var firsts = List[Int](capacity=blocks)
    var rows = 0
    for b in range(blocks):
        firsts.append(rows)
        rows += len(scans[b])

    var limits = List[Int](capacity=blocks)
    for b in range(blocks):
        if options.infer_rows == INFER_ALL:
            limits.append(INFER_ALL)
            continue
        var left = options.infer_rows - firsts[b]
        limits.append(left if left > 0 else -1)

    # The columns of the file, in the order the file first uses them. Found per
    # block in parallel because it is a pass over every member, and merged here
    # because the merge is a few dozen names per block rather than one per
    # member.
    var locals = List[List[String]](length=blocks, fill=List[String]())

    def names_of(b: Int) raises {mut locals, imm}:
        locals[b] = local_names(bytes, scans[b], limits[b])

    parallel_for(names_of, blocks)

    var names = List[String]()
    for b in range(blocks):
        for c in range(len(locals[b])):
            var known = False
            for k in range(len(names)):
                if names[k] == locals[b][c]:
                    known = True
                    break
            if not known:
                names.append(locals[b][c].copy())
    var width = len(names)

    def ordinals_of(b: Int) raises {mut scans, imm}:
        _fill_ordinals(bytes, scans[b], names)

    parallel_for(ordinals_of, blocks)

    # One guess per column per block, merged by taking the higher rung, which is
    # what makes inferring a block at a time give the same answer as inferring
    # the whole file at once.
    var tasks = width * blocks
    var guesses = List[TypeGuess](length=tasks, fill=TypeGuess(0, False))

    def guess(task: Int) raises {mut guesses, imm}:
        var column = task // blocks
        var b = task % blocks
        ref lines = scans[b]
        var seen = len(lines)
        if limits[b] != INFER_ALL and limits[b] < seen:
            seen = limits[b] if limits[b] > 0 else 0
        var here = TypeGuess(0, False)
        for row in range(seen):
            for m in range(lines.starts[row], lines.starts[row + 1]):
                if Int(lines.ordinals[m]) != column:
                    continue
                ref value = lines.members[m].value
                here = combine(here, _rung_of(value.kind, bytes, value))
                break
        guesses[task] = here

    if tasks != 0:
        parallel_for(guess, tasks)

    var fields = List[Field](capacity=width)
    for c in range(width):
        var merged = TypeGuess(0, False)
        for b in range(blocks):
            merged = combine(merged, guesses[c * blocks + b])
        fields.append(Field(names[c].copy(), logical_of(merged)))
    var schema = Schema(fields^)

    # One task per column per block rather than one per column, because a file
    # with two columns would otherwise leave most of the machine idle for the
    # part of the read that touches every value.
    var pieces = List[AnyArray](
        length=tasks, fill=AnyArray(Array[DType.bool](0))
    )

    def fill(task: Int) raises {mut pieces, imm}:
        var column = task // blocks
        var b = task % blocks
        pieces[task] = _column_of(
            bytes, scans[b], column, schema[column], firsts[b]
        )

    if tasks != 0:
        parallel_for(fill, tasks)

    var columns = List[AnyArray](capacity=width)
    for c in range(width):
        if blocks == 1:
            # Moved rather than copied, and the pieces of column c are the front
            # of the list once the columns before it have been taken.
            columns.append(pieces.pop(0))
            continue
        # By reference, because a copy here is the same size as the concat and
        # would double what the read costs in memory.
        var refs = List[Pointer[AnyArray, ImmUntrackedOrigin]](capacity=blocks)
        for b in range(blocks):
            refs.append(column_ref(pieces[c * blocks + b]))
        columns.append(concat_refs_any(refs))
    return DataFrame(schema^, columns^)


def read_ndjson_bytes(bytes: Span[UInt8, _]) raises -> DataFrame:
    """Reads a buffer of newline delimited JSON with the default options.

    Args:
        bytes: The whole document.

    Returns:
        The frame.

    Raises:
        Error: If a line is not a well formed JSON object.
    """
    return read_ndjson_bytes(bytes, NdjsonOptions())


def read_ndjson(path: String, options: NdjsonOptions) raises -> DataFrame:
    """Reads a whole NDJSON file into a frame.

    The file is mapped rather than copied, for the reason `read_csv` gives at
    length: the parser reads the page cache where it lies and the pages arrive
    on demand, on the same cores that are already parsing. A file that cannot be
    mapped, and an empty one counts, is read the old way, which is also where
    the error for a file that is not there comes from.

    Args:
        path: The file to read.
        options: How much of it to look at before deciding the columns.

    Returns:
        The frame.

    Raises:
        Error: If the file cannot be read, or is not readable as NDJSON.
    """
    var mapped = map_file(path)
    if mapped:
        ref file = mapped.value()
        return read_ndjson_bytes(file.bytes(), options)
    var handle = open(path, "r")
    var data = handle.read_bytes()
    handle.close()
    return read_ndjson_bytes(Span(data), options)


def read_ndjson(path: String) raises -> DataFrame:
    """Reads a whole NDJSON file with the default options.

    Args:
        path: The file to read.

    Returns:
        The frame.

    Raises:
        Error: If the file cannot be read, or is not readable as NDJSON.
    """
    return read_ndjson(path, NdjsonOptions())
