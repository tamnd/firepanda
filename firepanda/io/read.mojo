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
or silently null the value, and the frame quietly loses data.

What the ladder also buys is a way not to read the file twice. Deciding a type
by looking at every value means parsing every value in the file before parsing
every value in the file, and on a wide file that second pass was a third of the
read. So the type is guessed from the first `SPECULATE_ROWS` rows of each block
and the column is filled straight away, and a value that does not fit moves the
column one rung up and fills it again. A rung read off a sample is never higher
than the rung the whole column needs, and the ladder only climbs, so the type
that comes out is the type looking at everything first would have given. It is
the same answer, arrived at without the pass that proved it.

Refilling is the uncommon case and it costs one more pass over one column, not
over the file. A caller who would rather not risk it can bound `infer_rows`,
which goes back to deciding first and filling second, or declare the schema and
skip the question, which is also the only way to force a column wider than its
values need.

Missing fields never constrain the type. A column of a hundred integers and one
`NA` is an integer column with one null in it, which is a thing this frame can
represent and a pandas frame cannot, and the difference is the reason the
validity bitmap exists.

A file large enough to be worth it is read on every core. `split.mojo` cuts the
buffer into blocks that each start on a row boundary, and from there the scan,
the sampling and the conversion all run one task per block. Each block scans into
its own `Scan` and samples its own rows.

Four things are worth knowing about that.

**The types are decided globally, not per block.** Each block climbs the ladder
over the rows it looked at and reports how far it got, and the column starts at
the highest rung any block reached. A block of all nulls reports that it saw no
value at all, which is not the same as reporting string, because otherwise one
empty block would force a numeric column to text. Sampling every block rather
than the head of the file costs nothing extra and starts the file whose first
megabyte is integers and whose last is not off at the right rung.

**A fixed width column is written once, in place.** The column is allocated at
its full height before any block runs and each block parses its rows straight
into its own range of it, so there are no per block columns to allocate and
nothing to stack afterwards. Validity is the exception: two blocks meeting
inside a byte would each have to read, modify and write the same word, so each
block fills a bitmap of its own and those are pasted in afterwards, which is a
pass over a bit per row rather than over a value per row. A block with no nulls
is not pasted at all.

**A string column is written once too, since 0.6.15.** The obstacle used to be
that a block's payload size is not known until the block has been read. It is
knowable without reading it: the field index already records where every field
starts and ends, so a pass over the index adds up the lengths, and the only
fields that need the file are the escaped ones, whose doubled quotes have to be
counted before they collapse. Those block totals are prefix summed, the column
is allocated once at its full height and payload size, and each block writes its
own slice of views and its own slice of payload at absolute offsets. Nothing is
stacked and nothing is rebased.

**Columns are done one at a time.** Doing every column's blocks at once would
hold the whole file's worth of pieces at the peak. Each column still fills every
core, because the parallelism is over blocks rather than over columns.

**A wrong split cannot pass silently.** `split.mojo` documents the check; if it
fires, or if any block fails to scan, the whole read is done again on one thread
and that is the answer the caller gets.
"""

from std.collections.span import Span
from std.memory import unsafe_memcpy

from firepanda.array import (
    Array,
    AnyArray,
    INLINE_CAPACITY,
    PREFIX_LENGTH,
    StringArray,
    StringBuilder,
    StringView,
    VIEW_SIZE,
    collapse_into,
    make_inline_at,
    make_long_at,
)
from firepanda.bitmap import Bitmap
from firepanda.buffer import Buffer
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
    collapsed_length,
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


comptime SPECULATE_ROWS = 128
"""Rows of each block a read looks at before it picks a type and commits.

Small on purpose. The point of the sample is not to be right, it is to be right
almost always and cheap always: a wrong guess costs one extra pass over one
column, and a sample large enough to never be wrong costs a pass over the whole
file every time.
"""


@fieldwise_init
struct BlockFill(Movable):
    """What one block made of one column."""

    var valid: Bitmap
    """Which of the block's rows held a value."""

    var ok: Bool
    """Whether every value parsed as the type asked for."""

    var row: Int
    """The file row of the first value that did not, when `ok` is false."""

    var saw_value: Bool
    """Whether any row of the block held a value at all."""


@fieldwise_init
struct TextFill(Movable):
    """What one block made of one text column.

    Text has no type to get wrong, so the only thing a block can report is
    whether it got to the end, which under `inline_only` it does not when it
    meets an element that needs a payload.
    """

    var valid: Bitmap
    """Which of the block's rows held a value."""

    var done: Bool
    """Whether the block filled every row it was given."""


@fieldwise_init
struct FillReport(ImplicitlyCopyable, Movable):
    """What every block together made of one column, apart from the values.

    It is filled in through a `mut` argument rather than returned beside the
    column, because a caller that has to decide whether to keep the column
    cannot decide it while holding the two in one value.
    """

    var ok: Bool
    """Whether every value in the file parsed as the type asked for."""

    var row: Int
    """The lowest file row that did not, when `ok` is false."""

    var saw_value: Bool
    """Whether any row of the column held a value at all."""


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


def sample_columns(
    data: Span[UInt8, _],
    scans: List[Scan],
    layout: List[BlockRows],
    width: Int,
    options: ReadOptions,
) raises -> List[TypeGuess]:
    """Guesses every column's type from the first few rows of every block.

    The same climb as full inference over a hundredth of a percent of the rows.
    A rung read off a sample is always at or below the rung the whole column
    needs, because a rung only ever goes up, which is what makes the guess safe
    to build on: too low is a column read twice and too high cannot happen.

    Sampling every block rather than the head of the file costs nothing extra
    and catches the file whose first megabyte is integers and whose last is not.

    Args:
        data: The buffer the scans point into.
        scans: One scan per block.
        layout: Each block's share of the rows.
        width: How many columns the file has.
        options: The read options.

    Returns:
        One guess per column.

    Raises:
        Error: Never directly. The signature is inherited from `parallel_for`.
    """
    var blocks = len(scans)
    var parts = List[TypeGuess](
        length=width * blocks, fill=TypeGuess(_BOOL, False)
    )

    def one(task: Int) raises {mut parts, imm}:
        var column = task // blocks
        var b = task % blocks
        var last = len(scans[b])
        var bound = layout[b].first + SPECULATE_ROWS
        if bound < last:
            last = bound
        parts[task] = guess_column(
            data, scans[b], column, layout[b].first, last
        )

    parallel_for(one, width * blocks)

    var merged = List[TypeGuess](capacity=width)
    for column in range(width):
        var guess = TypeGuess(_BOOL, False)
        for b in range(blocks):
            guess = combine(guess, parts[column * blocks + b])
        merged.append(guess)
    return merged^


def rung_of(dtype: LogicalType) -> Int:
    """Returns which rung of the ladder a declared type sits on.

    Anything that is not one of the three types inference can produce is
    `_STRING`, which here means "not a fixed width column this file fills in
    place" rather than "text". A declared `INT32` goes down the piece and stack
    path with everything else, which is what it did before any of this existed.

    Args:
        dtype: The column's type.

    Returns:
        One of `_BOOL`, `_INT`, `_FLOAT` and `_STRING`.
    """
    if dtype == LogicalType.BOOL:
        return _BOOL
    if dtype == LogicalType.INT64:
        return _INT
    if dtype == LogicalType.FLOAT64:
        return _FLOAT
    return _STRING


def wanted_of(rung: Int) -> String:
    """Names a rung the way an error message names it.

    Args:
        rung: One of `_BOOL`, `_INT` and `_FLOAT`.

    Returns:
        The phrase that goes after "is not".
    """
    if rung == _BOOL:
        return "a boolean"
    if rung == _INT:
        return "an integer"
    return "a number"


def merge_reports(
    parts: List[BlockFill], blocks: Int, width: Int, k: Int
) -> FillReport:
    """Combines one column's per block reports into one.

    Args:
        parts: One report per block per fixed width column.
        blocks: How many blocks there are.
        width: How many fixed width columns there are.
        k: Which of them to combine.

    Returns:
        Whether every value fitted, the lowest file row that did not, and
        whether any value was seen anywhere in the column.
    """
    var out = FillReport(True, 0, False)
    for b in range(blocks):
        ref part = parts[b * width + k]
        out.saw_value = out.saw_value or part.saw_value
        if not part.ok and (out.ok or part.row < out.row):
            # The lowest failing row rather than whichever block happened to
            # finish first, so a malformed file always names the same line.
            out.ok = False
            out.row = part.row
    return out


def paste_validity[
    dt: DType
](
    mut column: Array[dt],
    parts: List[BlockFill],
    layout: List[BlockRows],
    blocks: Int,
    width: Int,
    k: Int,
):
    """Copies one column's per block bitmaps into the column's own.

    Args:
        column: The column to paste into.
        parts: One report per block per fixed width column.
        layout: Each block's share of the rows.
        blocks: How many blocks there are.
        width: How many fixed width columns there are.
        k: Which of them to paste.

    Parameters:
        dt: The column's dtype.
    """
    for b in range(blocks):
        ref part = parts[b * width + k]
        if not part.valid.all_valid():
            column.data.validity.paste(
                layout[b].before, part.valid, layout[b].count
            )


comptime TILE_BYTES = 1 << 14
"""How much of the field index one tile of a fixed width sweep works over.

Sixteen kilobytes, which is a third of a data cache and leaves the rest to the
file bytes and the columns being written. What it buys is on
`sweep_fixed`.
"""


def sweep_fixed(
    data: Span[UInt8, _],
    scans: List[Scan],
    layout: List[BlockRows],
    header: Int,
    width: Int,
    fixed: List[Int],
    ints: List[Pointer[Scalar[DType.int64], MutUntrackedOrigin]],
    floats: List[Pointer[Scalar[DType.float64], MutUntrackedOrigin]],
    bools: List[Pointer[Scalar[DType.bool], MutUntrackedOrigin]],
    mut parts: List[BlockFill],
) raises:
    """Fills every fixed width column, one parallel pass, a tile of rows at a
    time.

    A column at a time is the wrong way round for a wide file. A column's fields
    sit `width` words apart in the index, so reading one column of a fifty column
    file touches one word of every cache line and leaves the other seven, and
    then the next column comes back for those. The index is read fifty times
    over to be used once, and it is far too big to still be there on the second
    pass. Filling the fixed width columns of a fifty column file cost 7.6 ns a
    value against 1.6 ns a value on a four column file of the same types, and
    nothing about a value costs four times more because it has more neighbours.

    Going by row instead reads the index once, in order, and fixes the fifty
    column file. It also makes the four column file twelve per cent slower,
    which is the other half of the problem: a column's running state, whether
    every value has fitted and where the first that did not was, cannot sit in a
    register when the loop moves to a different column on every value.

    So the rows are done in tiles. A tile is chosen to make its slice of the
    index about `TILE_BYTES`, and within a tile the columns are still done one
    at a time. The index slice is read once per column as before, but the reads
    after the first are out of the data cache rather than out of memory, and the
    per column state is a register again for the length of the tile. Neither
    file pays for the other.

    A string column is not in here. It cannot be, since a block's payload size
    is not known until it has been read, so those are still done one at a time.

    Args:
        data: The buffer the scans point into.
        scans: One scan per block.
        layout: Each block's share of the rows.
        header: One if the file has a header row, zero if not.
        width: How many fields a row of the file has.
        fixed: The schema positions of the fixed width columns, integer columns
            first, then float, then boolean, in schema order within each group.
        ints: Where each integer column's values go, one per column of the
            first group.
        floats: The same for the second group.
        bools: The same for the third.
        parts: One report per block per fixed width column, in the order `fixed`
            is in. Filled in.

    Raises:
        Error: Never directly. The signature is inherited from `parallel_for`.
    """
    var count = len(fixed)
    var blocks = len(scans)
    var n_int = len(ints)
    var n_float = len(floats)
    var tile = TILE_BYTES // (width * 8)
    if tile < 16:
        tile = 16

    def one(b: Int) raises {mut parts, imm}:
        ref scan = scans[b]
        var rows = layout[b].count
        var before = layout[b].before
        var first = layout[b].first
        var file_row = before + header
        var base = b * count
        var valids = List[Bitmap](capacity=count)
        var ok = List[Bool](length=count, fill=True)
        var bad = List[Int](length=count, fill=0)
        var saw = List[Bool](length=count, fill=False)
        for _ in range(count):
            valids.append(Bitmap(rows))

        for start in range(0, rows, tile):
            var stop = start + tile
            if stop > rows:
                stop = rows
            for k in range(count):
                var column = fixed[k]
                ref valid = valids[k]
                var column_ok = ok[k]
                var column_bad = bad[k]
                var column_saw = saw[k]
                for i in range(start, stop):
                    var bytes = field_bytes(data, scan.at(first + i, column))
                    if is_missing(bytes):
                        # A null is a zero in the values buffer, and the buffer
                        # arrived zeroed, so marking the bit is the whole of it.
                        valid.set(i, False)
                        continue
                    column_saw = True
                    var fits: Bool
                    if k < n_int:
                        var parsed = parse_int[DType.int64](bytes)
                        fits = parsed.ok
                        if fits:
                            ints[k].unsafe_offset(before + i)[] = parsed.value
                    elif k < n_int + n_float:
                        var parsed = parse_float[DType.float64](bytes)
                        fits = parsed.ok
                        if fits:
                            floats[k - n_int].unsafe_offset(
                                before + i
                            )[] = parsed.value
                    else:
                        var parsed = parse_bool(bytes)
                        fits = parsed.ok
                        if fits:
                            bools[k - n_int - n_float].unsafe_offset(
                                before + i
                            )[] = parsed.value
                    if not fits and column_ok:
                        # The column is wrong for the file and every value in it
                        # will be read again, but the other columns of this
                        # block are not, so this cannot stop at the first bad
                        # value the way one column at a time could.
                        column_ok = False
                        column_bad = file_row + i
                ok[k] = column_ok
                bad[k] = column_bad
                saw[k] = column_saw

        for k in range(count):
            parts[base + k] = BlockFill(valids.pop(0), ok[k], bad[k], saw[k])

    parallel_for(one, blocks)


def build_blocks(
    data: Span[UInt8, _],
    scans: List[Scan],
    layout: List[BlockRows],
    var schema: Schema,
    options: ReadOptions,
    guesses: List[TypeGuess],
) raises -> DataFrame:
    """Fills every fixed width column in one pass and the text ones after.

    A fixed width column is allocated at its full height and written in place,
    so all of them can be filled together, and they are: one walk per block over
    the rows, every column of a row done while the index for that row is in
    cache. A string column cannot join in, because a block's payload size is not
    known until it has been read, so it builds a piece per block and stacks
    them, one column at a time.

    When `guesses` is not empty the schema is a guess from a sample rather than
    a fact, and a column with a value that does not fit is filled again one rung
    higher. The ladder only goes up, so a column is read at most four times and
    in every file anyone has it is read once. What that buys is the whole of
    inference: deciding the type properly means parsing every field in the file
    before parsing every field in the file.

    When `guesses` is empty the schema came from the caller and a value that
    does not fit is an error, which is the only thing a declared type can mean.

    Args:
        data: The buffer the scans point into.
        scans: One scan per block.
        layout: Each block's share of the rows.
        schema: The types to read as, which speculation may raise.
        options: The read options.
        guesses: One sampled guess per column, or empty for a declared schema.

    Returns:
        The frame.

    Raises:
        Error: If a value does not fit a declared column's type.
    """
    var blocks = len(scans)
    var header = 1 if options.has_header else 0
    var rows = 0
    for b in range(blocks):
        rows += layout[b].count
    var speculating = len(guesses) > 0

    var starts = List[Int](capacity=len(schema))
    for column in range(len(schema)):
        var rung = rung_of(schema[column].dtype)
        if speculating and not guesses[column].saw_value:
            # The sample saw no value at all, so it said string, and that is
            # the right answer only if the whole column turns out to be null.
            # Start at the bottom and let the first value that turns up decide.
            rung = _BOOL
        starts.append(rung)

    # Grouped by rung rather than left in schema order, because the sweep walks
    # a row once per group and a group with its rung known in advance is a
    # straight loop rather than a loop with a three way test in it.
    var fixed = List[Int]()
    var place = List[Int](length=len(schema), fill=-1)
    var ints = List[Array[DType.int64]]()
    var floats = List[Array[DType.float64]]()
    var bools = List[Array[DType.bool]]()
    for column in range(len(schema)):
        if starts[column] == _INT:
            place[column] = len(fixed)
            fixed.append(column)
            ints.append(Array[DType.int64](rows))
    for column in range(len(schema)):
        if starts[column] == _FLOAT:
            place[column] = len(fixed)
            fixed.append(column)
            floats.append(Array[DType.float64](rows))
    for column in range(len(schema)):
        if starts[column] == _BOOL:
            place[column] = len(fixed)
            fixed.append(column)
            bools.append(Array[DType.bool](rows))

    var count = len(fixed)
    # Placeholders. Every entry is overwritten by the sweep, which builds its
    # own bitmap of the right size, and `BlockFill` is not copyable so the list
    # cannot be filled any more cheaply than this.
    var parts = List[BlockFill](capacity=blocks * count)
    for _ in range(blocks * count):
        parts.append(BlockFill(Bitmap(0), True, 0, False))
    if count > 0:
        # The origins are dropped deliberately. Every block writes a range of
        # its own and the ranges partition each column, which is the fact that
        # makes this safe, and it is not a fact the origin system can be told.
        var int_at = List[Pointer[Scalar[DType.int64], MutUntrackedOrigin]]()
        for i in range(len(ints)):
            int_at.append(
                ints[i].unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            )
        var float_at = List[
            Pointer[Scalar[DType.float64], MutUntrackedOrigin]
        ]()
        for i in range(len(floats)):
            float_at.append(
                floats[i].unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            )
        var bool_at = List[Pointer[Scalar[DType.bool], MutUntrackedOrigin]]()
        for i in range(len(bools)):
            bool_at.append(
                bools[i].unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            )
        sweep_fixed(
            data,
            scans,
            layout,
            header,
            len(schema),
            fixed,
            int_at,
            float_at,
            bool_at,
            parts,
        )

    var columns = List[AnyArray](capacity=len(schema))
    for column in range(len(schema)):
        var rung = starts[column]
        if rung == _STRING:
            columns.append(
                stack_column(
                    data,
                    scans,
                    layout,
                    column,
                    header,
                    schema[column],
                    options,
                )
            )
            continue

        # Every group holds its columns in schema order, so the one wanted
        # here is always the one at the front of its list.
        var k = place[column]
        var report = merge_reports(parts, blocks, count, k)
        var accepted = report.ok and (report.saw_value or not speculating)
        if rung == _INT:
            var values = ints.pop(0)
            if accepted:
                paste_validity(values, parts, layout, blocks, count, k)
                columns.append(AnyArray(values^))
        elif rung == _FLOAT:
            var values = floats.pop(0)
            if accepted:
                paste_validity(values, parts, layout, blocks, count, k)
                columns.append(AnyArray(values^))
        else:
            var values = bools.pop(0)
            if accepted:
                paste_validity(values, parts, layout, blocks, count, k)
                columns.append(AnyArray(values^))
        if accepted:
            continue

        if not report.ok and not speculating:
            raise Error(
                bad_value(schema[column].name, report.row, wanted_of(rung))
            )
        # Either a value did not fit, which is one rung up, or the column turned
        # out to hold no value at all, which is text.
        var next = _STRING if report.ok else rung + 1
        columns.append(
            refill_column(
                data,
                scans,
                layout,
                column,
                header,
                rows,
                next,
                speculating,
                schema,
                options,
            )
        )

    return DataFrame(schema^, columns^)


def refill_column(
    data: Span[UInt8, _],
    scans: List[Scan],
    layout: List[BlockRows],
    column: Int,
    header: Int,
    rows: Int,
    var rung: Int,
    speculating: Bool,
    mut schema: Schema,
    options: ReadOptions,
) raises -> AnyArray:
    """Reads one column again, a rung at a time, until it fits.

    Only a column the sweep guessed too narrow for gets here, which is a column
    whose first hundred and odd rows of every block disagreed with the rest of
    it. On the way it keeps the schema in step with the rung, because the text
    path reads the type out of the schema rather than being told it.

    Args:
        data: The buffer the scans point into.
        scans: One scan per block.
        layout: Each block's share of the rows.
        column: Which column to read.
        header: One if the file has a header row, zero if not.
        rows: The height of the finished column.
        rung: The rung to try first.
        speculating: Whether a value that does not fit is a wrong guess rather
            than an error.
        schema: Updated to the type the column settles on.
        options: The read options.

    Returns:
        The column.

    Raises:
        Error: If a value does not fit a declared column's type.
    """
    var report = FillReport(True, 0, False)
    while True:
        if speculating:
            var settled = logical_of(TypeGuess(rung, True))
            if settled != schema[column].dtype:
                schema.fields[column] = Field(schema[column].name, settled)
        if rung == _STRING:
            return stack_column(
                data, scans, layout, column, header, schema[column], options
            )

        var accepted: Bool
        var values: AnyArray
        if rung == _BOOL:
            var got = fill_column[DType.bool, _BOOL](
                data,
                scans,
                layout,
                column,
                header,
                schema[column],
                rows,
                report,
            )
            accepted = report.ok and (report.saw_value or not speculating)
            values = AnyArray(got^)
        elif rung == _INT:
            var got = fill_column[DType.int64, _INT](
                data,
                scans,
                layout,
                column,
                header,
                schema[column],
                rows,
                report,
            )
            accepted = report.ok and (report.saw_value or not speculating)
            values = AnyArray(got^)
        else:
            var got = fill_column[DType.float64, _FLOAT](
                data,
                scans,
                layout,
                column,
                header,
                schema[column],
                rows,
                report,
            )
            accepted = report.ok and (report.saw_value or not speculating)
            values = AnyArray(got^)
        if accepted:
            return values^
        if not report.ok and not speculating:
            raise Error(
                bad_value(schema[column].name, report.row, wanted_of(rung))
            )
        rung = _STRING if report.ok else rung + 1


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
    if field.dtype == LogicalType.STRING:
        var rows = 0
        for b in range(len(scans)):
            rows += layout[b].count
        return AnyArray(fill_text(data, scans, layout, column, rows, options))

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


def text_payload_bytes(
    data: Span[UInt8, _],
    scan: Scan,
    column: Int,
    first_row: Int,
    rows: Int,
    options: ReadOptions,
) -> Int:
    """Returns how many payload bytes one block's rows of a text column need.

    Only elements longer than twelve bytes reach the payload, and a missing
    value writes nothing at all, so this is usually a walk over the index that
    touches none of the file. It has to look at an escaped field's bytes,
    because collapsing doubled quotes is the one thing about a field's length
    that the index does not record, and those are rare.

    A field longer than twelve bytes is never a missing value, since the longest
    token `is_missing` accepts is four, so the null test is skipped for exactly
    the fields whose length would have to be counted.

    Args:
        data: The buffer the scan points into.
        scan: The scanned fields.
        column: Which column to measure.
        first_row: The first value row within `scan`.
        rows: How many value rows to measure.
        options: The read options, for the quote character.

    Returns:
        The byte count.
    """
    var total = 0
    for i in range(rows):
        var span = scan.at(first_row + i, column)
        var length = span.end - span.start
        if length <= INLINE_CAPACITY:
            continue
        if span.escaped:
            length = collapsed_length(data, span, options.dialect.quote)
            if length <= INLINE_CAPACITY:
                continue
        total += length
    return total


def fill_text(
    data: Span[UInt8, _],
    scans: List[Scan],
    layout: List[BlockRows],
    column: Int,
    rows: Int,
    options: ReadOptions,
) raises -> StringArray:
    """Reads one text column, every block writing straight into it.

    The reason a string column used to build a piece per block and stack them is
    that a block's payload size is not known until the block has been read. It
    is knowable without reading it, though: an element's payload cost is its
    length, the index already holds every length, and the only length the index
    gets wrong is an escaped field's, which is a rare enough case to measure
    directly. So the block sizes are added up first and the fill writes into one
    column, and the stacking is gone rather than made faster.

    That is worth more than the copy it removes. The concat allocated a second
    payload as large as the first and then touched every page of it, and on a
    ten million row text column the page faults alone were most of the cost.

    Measuring is a second walk over the index, though, and a column whose
    elements all fit inside their views has no payload for that walk to find.
    That is the ordinary case: country codes, labels, identifiers. So the fill
    is tried first on the guess that nothing reaches the payload, and a block
    that meets an element too long to inline stops where it stands and says so.
    Then, and only then, the column is measured and filled again. The guess is
    the whole read when it holds, and when it does not the block that disproves
    it usually does so within a few rows, because a column with long elements in
    it rarely hides them at the end. This is the same bargain the type ladder
    makes: be right almost always and cheap always, and pay one extra pass over
    one column when wrong.

    Validity cannot be written in place, for the reason `fill_column` gives, so
    each block fills a bitmap of its own and they are pasted afterwards.

    Args:
        data: The buffer the scans point into.
        scans: One scan per block.
        layout: Each block's share of the rows.
        column: Which column to read.
        rows: The height of the finished column.
        options: The read options, for the quote character.

    Returns:
        The column.

    Raises:
        Error: Never directly. The signature is inherited from `parallel_for`.
    """
    var blocks = len(scans)
    var bases = List[Int](length=blocks, fill=0)
    var payload_bytes = 0

    # Every view and every payload byte below is written by the fill, a null
    # included, so neither buffer needs zeroing first.
    var views = Buffer(overwritten=rows * VIEW_SIZE)
    # The origins are dropped for the reason `fill_column` gives: the blocks
    # partition both buffers and that is not a fact the origin system can hold.
    var slots = views.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var payload = Buffer(overwritten=0)
    var parts = List[TextFill](capacity=blocks)
    for b in range(blocks):
        parts.append(TextFill(Bitmap(layout[b].count), True))

    var inline_only = True
    for _ in range(2):
        var bytes = payload.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()

        def one(b: Int) raises {mut parts, imm}:
            parts[b] = fill_text_block(
                data,
                scans[b],
                column,
                layout[b].first,
                layout[b].count,
                slots.unsafe_offset(layout[b].before * VIEW_SIZE),
                bytes,
                bases[b],
                inline_only,
                options,
            )

        parallel_for(one, blocks)
        if not inline_only:
            break
        var held = True
        for b in range(blocks):
            if not parts[b].done:
                held = False
                break
        if held:
            break

        inline_only = False
        var sizes = List[Int](length=blocks, fill=0)

        def measure(b: Int) raises {mut sizes, imm}:
            sizes[b] = text_payload_bytes(
                data,
                scans[b],
                column,
                layout[b].first,
                layout[b].count,
                options,
            )

        parallel_for(measure, blocks)
        for b in range(blocks):
            bases[b] = payload_bytes
            payload_bytes += sizes[b]
        payload = Buffer(overwritten=payload_bytes)

    var validity = Bitmap(rows)
    validity.set_all()
    for b in range(blocks):
        if layout[b].count > 0 and not parts[b].valid.all_valid():
            validity.paste(layout[b].before, parts[b].valid, layout[b].count)
    return StringArray(views^, payload^, validity^, rows)


def fill_text_block(
    data: Span[UInt8, _],
    scan: Scan,
    column: Int,
    first_row: Int,
    rows: Int,
    slots: Pointer[UInt8, MutUntrackedOrigin],
    payload: Pointer[UInt8, MutUntrackedOrigin],
    base: Int,
    inline_only: Bool,
    options: ReadOptions,
) -> TextFill:
    """Writes one block's rows of a text column into the finished buffers.

    An escaped field that collapses to twelve bytes or fewer is built in a
    register rather than in the payload, because the payload has room for
    exactly what `text_payload_bytes` counted and that element was counted as
    costing nothing.

    Under `inline_only` there is no payload at all, so an element that does not
    fit inside its view ends the block. What has been written so far is left
    where it is and overwritten by the second attempt, which is why the caller
    keeps one views buffer across both.

    Args:
        data: The buffer the scan points into.
        scan: The scanned fields.
        column: Which column to read.
        first_row: The first value row within `scan`.
        rows: How many value rows to read.
        slots: The column's views, offset to this block's first row.
        payload: The column's payload, from its first byte.
        base: This block's first payload byte.
        inline_only: Whether the payload is known to be empty, in which case an
            element needing one stops the block rather than writing to it.
        options: The read options, for the quote character.

    Returns:
        A bitmap of this block's rows set where the value is present, and
        whether the block ran to the end.
    """
    var valid = Bitmap(rows)
    valid.set_all()
    var views = slots.unsafe_bitcast[StringView]()
    var quote = options.dialect.quote
    var offset = base
    for i in range(rows):
        var span = scan.at(first_row + i, column)
        var field = field_bytes(data, span)
        var length = len(field)
        if span.escaped:
            if collapsed_length(data, span, quote) <= INLINE_CAPACITY:
                var scratch = InlineArray[UInt8, INLINE_CAPACITY](fill=0)
                var short = collapse_into(Pointer(to=scratch[0]), field, quote)
                views.unsafe_offset(i)[] = make_inline_at(
                    Pointer(to=scratch[0]), short
                )
                continue
            if inline_only:
                return TextFill(valid^, False)
            var dest = payload.unsafe_offset(offset)
            var written = collapse_into(dest, field, quote)
            views.unsafe_offset(i)[] = make_long_at(dest, written, 0, offset)
            offset += written
            continue
        # A quoted field is a value the writer meant literally, including the
        # quoted empty string and the quoted word `NA`. That is the one place
        # quoting changes a value rather than a boundary, and it is the only way
        # a file has of saying which of the two it meant.
        if length <= INLINE_CAPACITY:
            if not span.quoted and is_missing(field):
                views.unsafe_offset(i)[] = StringView()
                valid.set(i, False)
                continue
            # Built in place rather than in a register and stored, because a
            # sixteen byte load of what a four byte store and a short memcpy
            # just wrote does not forward, and that stall is per row.
            var slot = views.unsafe_offset(i)
            slot[] = StringView(UInt32(length), 0, 0, 0)
            unsafe_memcpy(
                dest=slot.unsafe_bitcast[UInt8]().unsafe_offset(PREFIX_LENGTH),
                src=field.unsafe_ptr(),
                count=length,
            )
            continue
        if inline_only:
            return TextFill(valid^, False)
        var dest = payload.unsafe_offset(offset)
        unsafe_memcpy(dest=dest, src=field.unsafe_ptr(), count=length)
        views.unsafe_offset(i)[] = make_long_at(dest, length, 0, offset)
        offset += length
    return TextFill(valid^, True)


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
    mut report: FillReport,
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
        report: Set to whether every value fitted, the first row that did not,
            and whether any value was seen at all.

    Returns:
        The column, which is complete only if the report says so.

    Raises:
        Error: Never directly. The signature is inherited from `parallel_for`.

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
    var parts = List[BlockFill](capacity=blocks)
    for b in range(blocks):
        parts.append(BlockFill(Bitmap(layout[b].count), True, 0, False))

    def one(b: Int) raises {mut parts, imm}:
        parts[b] = fill_block[dt, rung](
            data,
            scans[b],
            column,
            layout[b].first,
            layout[b].count,
            layout[b].before + header,
            values.unsafe_offset(layout[b].before),
        )

    parallel_for(one, blocks)

    report = FillReport(True, 0, False)
    for b in range(blocks):
        report.saw_value = report.saw_value or parts[b].saw_value
        if not parts[b].ok:
            # The lowest failing row rather than whichever block happened to
            # finish first, so a malformed file always names the same line.
            if report.ok or parts[b].row < report.row:
                report.ok = False
                report.row = parts[b].row
            continue
        if not parts[b].valid.all_valid():
            out.data.validity.paste(
                layout[b].before, parts[b].valid, layout[b].count
            )
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
    values: Pointer[Scalar[dt], MutUntrackedOrigin],
) raises -> BlockFill:
    """Parses one block's rows of one column into somebody else's buffer.

    Args:
        data: The buffer the scan points into.
        scan: The scanned fields.
        column: Which column to read.
        first_row: The first value row within `scan`.
        rows: How many value rows to read.
        file_row: The row number in the whole file of `first_row`, which is what
            an error message quotes.
        values: Where this block's first value goes. The caller guarantees room
            for `rows` of them.

    Returns:
        This block's validity, whether every value fitted, and the file row of
        the first that did not.

    Raises:
        Error: Never. A value that does not fit is reported rather than raised,
            because whether it is an error at all is the caller's question: a
            declared type says it is and a guessed one says the guess was too
            narrow. The signature is inherited from `parallel_for`.

    Parameters:
        dt: The dtype to read as.
        rung: Which of `_BOOL`, `_INT` and `_FLOAT` the values are parsed as.
    """
    var valid = Bitmap(rows)
    var saw_value = False
    for i in range(rows):
        var bytes = field_bytes(data, scan.at(first_row + i, column))
        if is_missing(bytes):
            # A null is a zero in the values buffer, and the buffer arrived
            # zeroed, so marking the bit is the whole of it.
            valid.set(i, False)
            continue
        saw_value = True

        comptime if rung == _BOOL:
            var parsed = parse_bool(bytes)
            if not parsed.ok:
                return BlockFill(valid^, False, file_row + i, saw_value)
            values.unsafe_offset(i)[] = rebind[Scalar[dt]](parsed.value)
        elif rung == _INT:
            var parsed = parse_int[dt](bytes)
            if not parsed.ok:
                return BlockFill(valid^, False, file_row + i, saw_value)
            values.unsafe_offset(i)[] = parsed.value
        else:
            var parsed = parse_float[dt](bytes)
            if not parsed.ok:
                return BlockFill(valid^, False, file_row + i, saw_value)
            values.unsafe_offset(i)[] = parsed.value
    return BlockFill(valid^, True, 0, saw_value)


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
        if options.infer_rows != INFER_ALL:
            # A bound on the sample is a promise about what gets looked at, so
            # it keeps the two pass read: speculating would quietly look past it
            # and give a type the caller asked not to be given.
            var bounded = infer_schema_blocks(
                data, scans, layout, width, options
            )
            return build_blocks(
                data, scans, layout, bounded^, options, List[TypeGuess]()
            )
        var guesses = sample_columns(data, scans, layout, width, options)
        var fields = List[Field](capacity=width)
        for column in range(width):
            fields.append(
                Field(
                    column_name(data, scans[0], column, options),
                    logical_of(guesses[column]),
                )
            )
        return build_blocks(
            data, scans, layout, Schema(fields^), options, guesses
        )

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
        return build_blocks(
            data, scans, layout, schema^, options, List[TypeGuess]()
        )

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
            builder.append_escaped(bytes, options.dialect.quote)
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


def read_csv_as(
    path: String, var schema: Schema, options: ReadOptions
) raises -> DataFrame:
    """Reads a whole CSV file using types the caller already knows.

    The same mapping as `read_csv`, and the same reason for the fallback. What
    it saves over `read_csv` is inference, which is a whole pass over every
    field in the file before a single value is parsed.

    Args:
        path: The file to read.
        schema: The types to read as, in column order.
        options: The read options.

    Returns:
        The frame.

    Raises:
        Error: If the file cannot be read, if the schema does not have one
            field per column, or if a value does not fit its declared type.
    """
    var mapped = map_file(path)
    if mapped:
        ref file = mapped.value()
        return read_csv_bytes_as(file.bytes(), schema^, options)
    var handle = open(path, "r")
    var data = handle.read_bytes()
    handle.close()
    return read_csv_bytes_as(Span(data), schema^, options)


def read_csv_as(path: String, var schema: Schema) raises -> DataFrame:
    """Reads a whole CSV file as a known schema with the default options.

    Args:
        path: The file to read.
        schema: The types to read as, in column order.

    Returns:
        The frame.

    Raises:
        Error: If the file cannot be read, if the schema does not have one
            field per column, or if a value does not fit its declared type.
    """
    return read_csv_as(path, schema^, ReadOptions())


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
