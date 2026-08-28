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

A file large enough to be worth it is read on every core. `split.mojo` cuts the
buffer into blocks that each start on a row boundary, and from there the scan,
the inference and the conversion all run one task per block. Each block scans
into its own `Scan` and guesses its own types.

Four things are worth knowing about that.

**The types are decided globally, not per block.** Each block climbs the ladder
over its own rows and reports how far it got, and the column takes the highest
rung any block reached. A block of all nulls reports that it saw no value at all,
which is not the same as reporting string, because otherwise one empty block
would force a numeric column to text.

**A fixed width column is written once, in place.** The column is allocated at
its full height before any block runs and each block parses its rows straight
into its own range of it, so there are no per block columns to allocate and
nothing to stack afterwards. Validity is the exception: two blocks meeting
inside a byte would each have to read, modify and write the same word, so each
block fills a bitmap of its own and those are pasted in afterwards, which is a
pass over a bit per row rather than over a value per row. A block with no nulls
is not pasted at all.

**A string column still builds pieces and stacks them.** A block's payload size
is not known until the block has been read, so there is nothing to write into.
The stacking is two memcpys per block since 0.6.10, which is most of the way to
the same place.

**Columns are done one at a time.** Doing every column's blocks at once would
hold the whole file's worth of pieces at the peak. Each column still fills every
core, because the parallelism is over blocks rather than over columns.

**A wrong split cannot pass silently.** `split.mojo` documents the check; if it
fires, or if any block fails to scan, the whole read is done again on one thread
and that is the answer the caller gets.
"""

from std.collections.span import Span

from firepanda.array import Array, AnyArray, StringArray, StringBuilder
from firepanda.bitmap import Bitmap
from firepanda.dtype import Field, LogicalType, Schema
from firepanda.exec import parallel_for, worker_count
from firepanda.frame import DataFrame
from firepanda.kernel.concat import concat_any

from .mapped import map_file
from .parse import is_missing, parse_bool, parse_float, parse_int
from .scan import (
    Dialect,
    FieldSpan,
    Scan,
    default_dialect,
    field_bytes,
    scan_block,
    scan_csv,
    unescape,
)
from .split import Split, split_buffer


comptime INFER_ALL = 0
"""Passed as `infer_rows` to look at every row before deciding a type."""

comptime MIN_BLOCK = 1 << 18
"""Bytes a block must be worth before the buffer is cut into more of them.

A quarter of a megabyte. Below this the split, the task and the stacking cost
more than the block saves, and a small file is fast either way.
"""


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


comptime _BOOL = 0
"""The bottom rung of the inference ladder."""

comptime _INT = 1
"""One rung up from boolean."""

comptime _FLOAT = 2
"""One rung up from integer."""

comptime _STRING = 3
"""The top rung, which everything fits and nothing climbs out of."""


@fieldwise_init
struct TypeGuess(ImplicitlyCopyable, Movable):
    """How far up the ladder one part of a column got.

    Two parts of the same column are combined by taking the higher rung, which is
    what makes inferring a block at a time give the same answer as inferring the
    whole column at once.
    """

    var rung: Int
    """The highest rung any value needed."""

    var saw_value: Bool
    """Whether any value was looked at at all.

    A part with no values does not vote. It is not evidence for string, which is
    what a bare rung would make it, and one block of nulls in a numeric column is
    common enough that getting this wrong would be noticed immediately.
    """


def combine(a: TypeGuess, b: TypeGuess) -> TypeGuess:
    """Merges two parts of the same column.

    Args:
        a: One part's guess.
        b: The other's.

    Returns:
        The guess for the two together.
    """
    return TypeGuess(
        a.rung if a.rung > b.rung else b.rung, a.saw_value or b.saw_value
    )


def logical_of(guess: TypeGuess) -> LogicalType:
    """Turns a finished guess into the type a column gets.

    Args:
        guess: The guess.

    Returns:
        The type.
    """
    if not guess.saw_value:
        # No value was observed, so nothing argues for a number. A string column
        # of all nulls can be cast to anything later; a float column of all
        # nulls has already thrown the text away.
        return LogicalType.STRING
    if guess.rung == _BOOL:
        return LogicalType.BOOL
    if guess.rung == _INT:
        return LogicalType.INT64
    if guess.rung == _FLOAT:
        return LogicalType.FLOAT64
    return LogicalType.STRING


def guess_column(
    data: Span[UInt8, _],
    scan: Scan,
    column: Int,
    first_row: Int,
    last_row: Int,
) -> TypeGuess:
    """Climbs the ladder over one column of one row range.

    Args:
        data: The buffer the scan points into.
        scan: The scanned fields.
        column: Which column to look at.
        first_row: The first row to sample, past any header.
        last_row: One past the last row to sample.

    Returns:
        The highest rung needed, and whether any value was seen.
    """
    var rung = _BOOL
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
            return TypeGuess(_STRING, True)
        if rung == _BOOL and parse_bool(bytes).ok:
            continue
        if rung <= _INT and parse_int[DType.int64](bytes).ok:
            rung = _INT
            continue
        if rung <= _FLOAT and parse_float[DType.float64](bytes).ok:
            rung = _FLOAT
            continue
        return TypeGuess(_STRING, True)
    return TypeGuess(rung, saw_value)


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
    return logical_of(guess_column(data, scan, column, first_row, last_row))


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
        var name = column_name(data, scan, column, options)
        var dtype = infer_column(data, scan, column, first_row, last_row)
        fields.append(Field(name, dtype))
    return Schema(fields^)


def column_name(
    data: Span[UInt8, _], scan: Scan, column: Int, options: ReadOptions
) -> String:
    """Returns what one column is called.

    Args:
        data: The buffer the scan points into.
        scan: A scan whose row zero is the file's first row.
        column: Which column to name.
        options: The read options.

    Returns:
        The header cell, or a positional name for a file with no header.
    """
    if not options.has_header:
        return String("column_", column)
    var span = scan.at(0, column)
    if span.escaped:
        var literal = unescape(data, span, options.dialect.quote)
        return String(StringSlice(unsafe_from_utf8=Span(literal)))
    return String(StringSlice(unsafe_from_utf8=field_bytes(data, span)))


def block_count(size: Int) -> Int:
    """Returns how many blocks a buffer of this size should be cut into.

    Args:
        size: The buffer's length in bytes.

    Returns:
        One for a file not worth splitting, otherwise one block per worker at
        most and never a block smaller than `MIN_BLOCK`.
    """
    var affordable = size // MIN_BLOCK
    var workers = worker_count()
    var count = workers if affordable > workers else affordable
    return count if count > 1 else 1


def scan_blocks(
    data: Span[UInt8, _], dialect: Dialect, blocks: Int
) raises -> List[Scan]:
    """Scans a buffer in parallel, or raises if the split cannot be trusted.

    Args:
        data: The whole file.
        dialect: The delimiter and quote character.
        blocks: How many blocks to cut the buffer into.

    Returns:
        One scan per block, with offsets into `data`.

    Raises:
        Error: If any block fails to scan, or if the blocks did not account for
            every quote byte in the file. Either means the caller should read
            the file on one thread instead. `split.mojo` has the argument.
    """
    var split = split_buffer(data, dialect, blocks)
    var scans = List[Scan](capacity=blocks)
    for _ in range(blocks):
        scans.append(Scan())

    def one(b: Int) raises {mut scans, imm}:
        scans[b] = scan_block(
            data, dialect, split.bounds[b], split.bounds[b + 1]
        )

    parallel_for(one, blocks)

    var accounted = 0
    for b in range(blocks):
        accounted += scans[b].quotes
    if accounted != split.quotes:
        raise Error(
            "csv: a quote byte was read as data, so the block split is not"
            " trustworthy"
        )
    return scans^


@fieldwise_init
struct BlockRows(ImplicitlyCopyable, Movable):
    """Where one block's rows sit, in its own scan and in the file."""

    var first: Int
    """The block's first value row, which is one for a block zero with a header.
    """

    var count: Int
    """How many value rows the block has."""

    var before: Int
    """How many value rows the whole file has before this block."""


def block_rows(scans: List[Scan], options: ReadOptions) -> List[BlockRows]:
    """Works out each block's share of the file's rows.

    Args:
        scans: One scan per block.
        options: The read options.

    Returns:
        One entry per block.
    """
    var out = List[BlockRows](capacity=len(scans))
    var before = 0
    for b in range(len(scans)):
        var first = 1 if b == 0 and options.has_header else 0
        var count = len(scans[b]) - first
        if count < 0:
            count = 0
        out.append(BlockRows(first, count, before))
        before += count
    return out^


def block_width(scans: List[Scan]) raises -> Int:
    """Returns how many columns the file has, checking that every block agrees.

    Args:
        scans: One scan per block.

    Returns:
        The column count, or zero for a file with no rows.

    Raises:
        Error: If any block is ragged, or if two blocks disagree on the width.
    """
    var width = -1
    for b in range(len(scans)):
        if len(scans[b]) == 0:
            continue
        if scans[b].is_ragged():
            raise Error(
                "rows have different numbers of fields; a ragged file has no"
                " schema"
            )
        if width < 0:
            width = scans[b].width(0)
        elif scans[b].width(0) != width:
            raise Error(
                "rows have different numbers of fields; a ragged file has no"
                " schema"
            )
    return width if width > 0 else 0


def infer_schema_blocks(
    data: Span[UInt8, _],
    scans: List[Scan],
    layout: List[BlockRows],
    width: Int,
    options: ReadOptions,
) raises -> Schema:
    """Names and types every column of a buffer that was scanned in blocks.

    Every column of every block is guessed at once and the guesses are combined
    afterwards, rather than one task per column, because a file with four columns
    would otherwise use four cores.

    Args:
        data: The buffer the scans point into.
        scans: One scan per block.
        layout: Each block's share of the rows.
        width: How many columns the file has.
        options: The read options.

    Returns:
        The schema.

    Raises:
        Error: Never directly. The signature is inherited from `parallel_for`.
    """
    var blocks = len(scans)
    var guesses = List[TypeGuess](
        length=width * blocks, fill=TypeGuess(_BOOL, False)
    )

    def one(task: Int) raises {mut guesses, imm}:
        var column = task // blocks
        var b = task % blocks
        var last = len(scans[b])
        if options.infer_rows != INFER_ALL:
            # The sample is the file's first `infer_rows` value rows, so a block
            # that starts past the end of it looks at nothing at all.
            var left = options.infer_rows - layout[b].before
            var bound = layout[b].first + (left if left > 0 else 0)
            if bound < last:
                last = bound
        guesses[task] = guess_column(
            data, scans[b], column, layout[b].first, last
        )

    parallel_for(one, width * blocks)

    var fields = List[Field](capacity=width)
    for column in range(width):
        var merged = TypeGuess(_BOOL, False)
        for b in range(blocks):
            merged = combine(merged, guesses[column * blocks + b])
        fields.append(
            Field(
                column_name(data, scans[0], column, options), logical_of(merged)
            )
        )
    return Schema(fields^)


def build_blocks(
    data: Span[UInt8, _],
    scans: List[Scan],
    layout: List[BlockRows],
    var schema: Schema,
    options: ReadOptions,
) raises -> DataFrame:
    """Fills one column at a time, every block of it at once.

    Args:
        data: The buffer the scans point into.
        scans: One scan per block.
        layout: Each block's share of the rows.
        schema: The types to read as.
        options: The read options.

    Returns:
        The frame.

    Raises:
        Error: If a value does not fit its column's declared type.
    """
    var blocks = len(scans)
    var header = 1 if options.has_header else 0
    var rows = 0
    for b in range(blocks):
        rows += layout[b].count
    var columns = List[AnyArray](capacity=len(schema))

    for column in range(len(schema)):
        var dtype = schema[column].dtype
        # A fixed width column is allocated once at its full height and every
        # block writes its own rows into it, so there is no per block column to
        # allocate and nothing to stack afterwards. A string column cannot do
        # that, because a block's payload size is not known until it has been
        # read, so it still builds pieces and concatenates them.
        if dtype == LogicalType.BOOL:
            var bools = fill_column[DType.bool, _BOOL](
                data, scans, layout, column, header, schema[column], rows
            )
            columns.append(AnyArray(bools^))
        elif dtype == LogicalType.INT64:
            var ints = fill_column[DType.int64, _INT](
                data, scans, layout, column, header, schema[column], rows
            )
            columns.append(AnyArray(ints^))
        elif dtype == LogicalType.FLOAT64:
            var floats = fill_column[DType.float64, _FLOAT](
                data, scans, layout, column, header, schema[column], rows
            )
            columns.append(AnyArray(floats^))
        else:
            columns.append(
                stack_column(
                    data, scans, layout, column, header, schema[column], options
                )
            )
    return DataFrame(schema^, columns^)


def stack_column(
    data: Span[UInt8, _],
    scans: List[Scan],
    layout: List[BlockRows],
    column: Int,
    header: Int,
    field: Field,
    options: ReadOptions,
) raises -> AnyArray:
    """Reads one column as a piece per block and stacks the pieces.

    Args:
        data: The buffer the scans point into.
        scans: One scan per block.
        layout: Each block's share of the rows.
        column: Which column to read.
        header: One if the file has a header row, zero if not.
        field: The column's name and type.
        options: The read options.

    Returns:
        The column.

    Raises:
        Error: If a value does not fit the column's declared type.
    """
    var blocks = len(scans)
    var pieces = List[AnyArray](capacity=blocks)
    for _ in range(blocks):
        # A placeholder rather than an `Optional`, because every slot is written
        # before any of them is read and the empty column costs an allocation of
        # nothing.
        pieces.append(AnyArray(Array[DType.int8](0)))

    def one(b: Int) raises {mut pieces, imm}:
        pieces[b] = read_column(
            data,
            scans[b],
            column,
            layout[b].first,
            layout[b].count,
            layout[b].before + header,
            field,
            options,
        )

    parallel_for(one, blocks)
    return concat_any(pieces)


def fill_column[
    dt: DType, rung: Int
](
    data: Span[UInt8, _],
    scans: List[Scan],
    layout: List[BlockRows],
    column: Int,
    header: Int,
    field: Field,
    rows: Int,
) raises -> Array[dt]:
    """Reads one fixed width column, every block writing straight into it.

    The values go into the finished column's own buffer, because a block's rows
    are a contiguous range of it and no two blocks share an element. Validity
    cannot go the same way: two blocks meeting inside a byte would each have to
    read, modify and write the same word, and that is a race whichever order
    they run in. So each block fills a bitmap of its own and the bitmaps are
    pasted afterwards, which is one pass over a bit per row against a pass over
    a value per row and does not show up next to the parse.

    A block whose rows are all present does not get pasted at all, and a column
    with no nulls anywhere therefore costs nothing.

    Args:
        data: The buffer the scans point into.
        scans: One scan per block.
        layout: Each block's share of the rows.
        column: Which column to read.
        header: One if the file has a header row, zero if not.
        field: The column's name and type.
        rows: The height of the finished column.

    Returns:
        The column.

    Raises:
        Error: If a value does not fit the column's declared type.

    Parameters:
        dt: The dtype to read as.
        rung: Which of `_BOOL`, `_INT` and `_FLOAT` the values are parsed as.
    """
    var blocks = len(scans)
    var out = Array[dt](rows)
    # The origin is dropped deliberately. Every block writes a range of its own
    # and the ranges partition the column, which is the fact that makes this
    # safe, and it is not a fact the origin system can be told.
    var values = out.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var masks = List[Bitmap](capacity=blocks)
    for b in range(blocks):
        masks.append(Bitmap(layout[b].count))

    def one(b: Int) raises {mut masks, imm}:
        masks[b] = fill_block[dt, rung](
            data,
            scans[b],
            column,
            layout[b].first,
            layout[b].count,
            layout[b].before + header,
            field.name,
            values.unsafe_offset(layout[b].before),
        )

    parallel_for(one, blocks)

    for b in range(blocks):
        if not masks[b].all_valid():
            out.data.validity.paste(layout[b].before, masks[b], layout[b].count)
    return out^


def fill_block[
    dt: DType, rung: Int
](
    data: Span[UInt8, _],
    scan: Scan,
    column: Int,
    first_row: Int,
    rows: Int,
    file_row: Int,
    name: String,
    values: Pointer[Scalar[dt], MutUntrackedOrigin],
) raises -> Bitmap:
    """Parses one block's rows of one column into somebody else's buffer.

    Args:
        data: The buffer the scan points into.
        scan: The scanned fields.
        column: Which column to read.
        first_row: The first value row within `scan`.
        rows: How many value rows to read.
        file_row: The row number in the whole file of `first_row`, which is what
            an error message quotes.
        name: The column name, for the error message.
        values: Where this block's first value goes. The caller guarantees room
            for `rows` of them.

    Returns:
        This block's validity, which the caller pastes into the column's.

    Raises:
        Error: If a value does not fit the type.

    Parameters:
        dt: The dtype to read as.
        rung: Which of `_BOOL`, `_INT` and `_FLOAT` the values are parsed as.
    """
    var valid = Bitmap(rows)
    for i in range(rows):
        var bytes = field_bytes(data, scan.at(first_row + i, column))
        if is_missing(bytes):
            # A null is a zero in the values buffer, and the buffer arrived
            # zeroed, so marking the bit is the whole of it.
            valid.set(i, False)
            continue

        comptime if rung == _BOOL:
            var parsed = parse_bool(bytes)
            if not parsed.ok:
                raise Error(bad_value(name, file_row + i, "a boolean"))
            values.unsafe_offset(i)[] = rebind[Scalar[dt]](parsed.value)
        elif rung == _INT:
            var parsed = parse_int[dt](bytes)
            if not parsed.ok:
                raise Error(bad_value(name, file_row + i, "an integer"))
            values.unsafe_offset(i)[] = parsed.value
        else:
            var parsed = parse_float[dt](bytes)
            if not parsed.ok:
                raise Error(bad_value(name, file_row + i, "a number"))
            values.unsafe_offset(i)[] = parsed.value
    return valid^


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
    var scans = List[Scan]()
    if split_scan(data, options.dialect, scans):
        var width = block_width(scans)
        var layout = block_rows(scans, options)
        var schema = infer_schema_blocks(data, scans, layout, width, options)
        return build_blocks(data, scans, layout, schema^, options)

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
    var scans = List[Scan]()
    if split_scan(data, options.dialect, scans):
        check_width(len(schema), block_width(scans))
        var layout = block_rows(scans, options)
        return build_blocks(data, scans, layout, schema^, options)

    var scan = scan_csv(data, options.dialect)
    if scan.is_ragged():
        raise Error(
            "rows have different numbers of fields; a ragged file has no schema"
        )
    if len(scan) > 0:
        check_width(len(schema), scan.width(0))
    return build(data, scan, schema^, options)


def check_width(declared: Int, found: Int) raises:
    """Checks that a declared schema has one field per column of the file.

    Args:
        declared: How many fields the schema has.
        found: How many columns the file has, or zero for an empty file.

    Raises:
        Error: If they disagree.
    """
    if found > 0 and declared != found:
        raise Error(
            String(
                "schema has ",
                declared,
                " fields but the file has ",
                found,
                " columns",
            )
        )


def split_scan(
    data: Span[UInt8, _], dialect: Dialect, mut scans: List[Scan]
) -> Bool:
    """Tries to scan a buffer in blocks and says whether it worked.

    The failure is swallowed rather than reported, and that is deliberate. There
    are two reasons a block scan fails: the split was not trustworthy, or the
    file is malformed. Only the sequential scan can tell them apart, and it has
    to run either way, so the answer the caller gets comes from the one pass that
    knows which row the problem is on.

    Args:
        data: The whole file.
        dialect: The delimiter and quote character.
        scans: Filled with one scan per block if this returns true, and left
            alone if it does not.

    Returns:
        True if the blocks can be used.
    """
    var blocks = block_count(len(data))
    if blocks <= 1:
        return False
    try:
        scans = scan_blocks(data, dialect, blocks)
        return True
    except:
        return False


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
        columns.append(
            read_column(
                data,
                scan,
                column,
                first_row,
                rows,
                first_row,
                schema[column],
                options,
            )
        )
    return DataFrame(schema^, columns^)


def read_column(
    data: Span[UInt8, _],
    scan: Scan,
    column: Int,
    first_row: Int,
    rows: Int,
    file_row: Int,
    field: Field,
    options: ReadOptions,
) raises -> AnyArray:
    """Fills one column, or one block's piece of one.

    Args:
        data: The buffer the scan points into.
        scan: The scanned fields.
        column: Which column to read.
        first_row: The first value row within `scan`.
        rows: How many value rows to read.
        file_row: The row number in the whole file of `first_row`, which is what
            an error message quotes. It differs from `first_row` only when the
            scan is one block of a file.
        field: The column's name and type.
        options: The read options.

    Returns:
        The column.

    Raises:
        Error: If a value does not fit the type, or if the type is one the CSV
            reader cannot produce.
    """
    if field.dtype == LogicalType.BOOL:
        return AnyArray(
            read_bool(data, scan, column, first_row, rows, file_row, field.name)
        )
    if field.dtype == LogicalType.INT64:
        return AnyArray(
            read_number[DType.int64, True](
                data, scan, column, first_row, rows, file_row, field.name
            )
        )
    if field.dtype == LogicalType.FLOAT64:
        return AnyArray(
            read_number[DType.float64, False](
                data, scan, column, first_row, rows, file_row, field.name
            )
        )
    if field.dtype == LogicalType.STRING:
        return AnyArray(read_text(data, scan, column, first_row, rows, options))
    raise Error(
        String(
            "column '",
            field.name,
            "' asks for ",
            field.dtype,
            ", which the CSV reader does not produce yet",
        )
    )


def read_bool(
    data: Span[UInt8, _],
    scan: Scan,
    column: Int,
    first_row: Int,
    rows: Int,
    file_row: Int,
    name: String,
) raises -> Array[DType.bool]:
    """Fills one boolean column.

    Args:
        data: The buffer the scan points into.
        scan: The scanned fields.
        column: Which column to read.
        first_row: The first value row.
        rows: How many value rows there are.
        file_row: The row number in the file of `first_row`.
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
            raise Error(bad_value(name, file_row + i, "a boolean"))
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
    file_row: Int,
    name: String,
) raises -> Array[dt]:
    """Fills one numeric column.

    Args:
        data: The buffer the scan points into.
        scan: The scanned fields.
        column: Which column to read.
        first_row: The first value row.
        rows: How many value rows there are.
        file_row: The row number in the file of `first_row`.
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
                raise Error(bad_value(name, file_row + i, "an integer"))
            out.set_valid(i, parsed.value)
        else:
            var parsed = parse_float[dt](bytes)
            if not parsed.ok:
                raise Error(bad_value(name, file_row + i, "a number"))
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

    In one piece does not mean in one copy. The file is mapped, so the parser
    reads the page cache where it lies and the pages arrive on demand, on the
    same cores that are already parsing. Copying the file first cost 394 ms on
    a 375 MB file and a second whole file of resident memory.

    A file that cannot be mapped, and an empty one counts, is read the old way.
    That path is also where the error for a file that does not exist comes
    from, which is why the mapping failure itself is swallowed.

    Args:
        path: The file to read.
        options: The read options.

    Returns:
        The frame.

    Raises:
        Error: If the file cannot be read, or is not readable as CSV.
    """
    var mapped = map_file(path)
    if mapped:
        ref file = mapped.value()
        return read_csv_bytes(file.bytes(), options)
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
