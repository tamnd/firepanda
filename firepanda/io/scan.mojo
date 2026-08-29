"""Finding the fields in a block of CSV bytes.

The scan is separate from the parse and runs first over the whole buffer. That
is one more pass than a character at a time reader that parses as it goes, and
it is the right trade for three reasons: the second pass is over a compact array
of offsets rather than over text, the parse of each column then knows its dtype
and can be a tight typed loop instead of a switch, and the row count is known
before a single column is allocated, so nothing has to grow.

The scan itself is a search for four bytes in a haystack, and the search has two
halves for a reason worth stating plainly, because the obvious version of it is
slower than no vectorization at all.

A register can be tested against the delimiter, the newline and the carriage
return in three instructions, and if none of them are in it the loop moves on by
a whole register. If one of them is, the loop has to find out which lane, and
there is no packed movemask reachable from this stdlib: `to_bits` on a boolean
vector gives a vector of zeros and ones back rather than an integer with one bit
per lane. So a register that hits has to be walked byte by byte anyway, and a
hit therefore costs the byte walk plus the vector compare rather than instead of
it.

In a file whose fields are five bytes long, every register hits. Which is why
the search walks the first eight bytes one at a time and only starts testing
registers once a field has proved it is longer than a word. Short fields never
pay for the vector path, long fields still skip most of themselves, and the
`csv/scan_narrow` and `csv/scan_long_text` rows in the benchmark suite are that
decision measured from both ends.

Quoting is RFC 4180. A field is quoted if its first byte is the quote character,
a doubled quote inside a quoted field is one literal quote, and a newline inside
a quoted field is data. The span this returns excludes the outer quotes and
carries a flag saying whether a doubled quote is in it, so the reader only pays
for unescaping on the fields that need it, which in practice is close to none of
them. It also records whether the field was quoted at all, which the reader needs
for exactly one thing: an empty field is a missing value and a quoted empty field
is the empty string, and those are different values.

Two things are refused rather than guessed. A quote that never closes before the
end of the buffer is an error with the row number in it, and so are bytes
between a closing quote and the next delimiter. Both of those are files that
some readers accept by silently inventing a value, and a silently invented value
in a data file is worse than a failed read.

Blank lines are skipped, which is pandas' default and is what a file ending in a
newline needs in order not to produce a phantom last row.

What comes out is one 64-bit word per field, not a struct of two offsets and two
flags. The index is the largest thing a read allocates and it is written once
and read once, so its size is nearly all of its cost, and three times smaller is
three times less memory traffic in both directions. `Scan.push` and `Scan.field`
are the only two functions that know the layout.
"""

from std.collections.span import Span
from std.sys.info import simd_width_of


comptime NEWLINE = UInt8(10)
"""ASCII line feed."""

comptime RETURN = UInt8(13)
"""ASCII carriage return."""

comptime COMMA = UInt8(44)
"""ASCII comma, the default delimiter."""

comptime DOUBLE_QUOTE = UInt8(34)
"""ASCII double quote, the default quote character."""

comptime PROLOGUE = 8
"""Bytes a search walks one at a time before it starts testing registers.

One 64-bit word. Almost every field in almost every file is shorter than this,
and for those the vector path is pure overhead. A whole register would be the
tidier number, but it costs the long fields a register's worth of skipping for
nothing, since the walk after a hit is bounded by the register width either way.
"""


comptime START_BITS = 40
"""Bits a packed field gives to its start offset, so a buffer up to a terabyte.

A field is packed into one 64-bit word because the index is the largest thing a
read allocates and it is read exactly once after it is written, so its size is
its cost almost entirely. The old layout was two `Int` offsets and two flags,
which padded to twenty four bytes: for a four column file of ten million rows
that is 960 MB of index over a 357 MB file.

Forty bits for the start and twenty two for the length is the split that makes
the common case exact. The start has to address the whole buffer, and a buffer
larger than a terabyte is refused with a message that says so rather than
truncated. The length only has to cover one field, and a field longer than four
megabytes is kept in `Scan.long` instead, so nothing is refused for being wide.
"""

comptime LENGTH_BITS = 22
"""Bits a packed field gives to its length, so four megabytes less one."""

comptime MAX_START = 1 << START_BITS
"""One past the largest byte offset a scan can record."""

comptime LONG_FIELD = (1 << LENGTH_BITS) - 1
"""The length that means the real length is in `Scan.long`.

A field of exactly this length is stored there too, which costs one entry and
saves a special case.
"""

comptime START_MASK = UInt64(MAX_START - 1)
"""Where the start sits in a packed field."""

comptime LENGTH_MASK = UInt64(LONG_FIELD)
"""Where the length sits, once shifted down."""

comptime ESCAPED_BIT = UInt64(1) << 62
"""Set when the field's content contains a doubled quote."""

comptime QUOTED_BIT = UInt64(1) << 63
"""Set when the field was written inside quotes."""


@fieldwise_init
struct Dialect(ImplicitlyCopyable, Movable):
    """The two bytes that decide how a file is cut up.

    Everything else a dialect could carry, comment characters and escape
    characters and skipped leading rows, is a reader option rather than a
    scanner one, because none of it changes where the field boundaries are.
    """

    var delimiter: UInt8
    """The byte between fields."""

    var quote: UInt8
    """The byte that opens and closes a quoted field."""


def default_dialect() -> Dialect:
    """Returns the comma and double quote dialect.

    Returns:
        The dialect almost every file uses.
    """
    return Dialect(COMMA, DOUBLE_QUOTE)


@fieldwise_init
struct FieldSpan(ImplicitlyCopyable, Movable):
    """Where one field's bytes are, and whether they are literal."""

    var start: Int
    """First byte of the field's content, past an opening quote."""

    var end: Int
    """One past the last byte of the content, before a closing quote."""

    var escaped: Bool
    """Whether the content contains a doubled quote needing an unescape."""

    var quoted: Bool
    """Whether the field was written inside quotes.

    Only one thing depends on this and it is worth stating, because otherwise
    this looks like a flag kept for its own sake. An empty field and a quoted
    empty field are the two ways a CSV file has of writing a missing value and
    the empty string, and they are different values. Without this the reader has
    to guess, and whichever way it guesses, one of the two is unrepresentable.
    """


@fieldwise_init
struct LongField(ImplicitlyCopyable, Movable):
    """The real length of a field too long to fit in a packed one."""

    var at: Int
    """The field's position in `Scan.fields`."""

    var length: Int
    """How many bytes of content it has."""


struct Scan(Movable, Sized):
    """Every field in a buffer, grouped into rows.

    The fields are one flat list and the rows are offsets into it, which is the
    same shape as an Arrow offsets buffer and for the same reason: one
    allocation for the whole file rather than one per row.

    A field is one 64-bit word rather than a `FieldSpan`, which is three times
    smaller and is the difference between an index that fits in cache and one
    that does not. `push` and `field` are the only two places that know the
    layout, and `at` still hands back an ordinary `FieldSpan`, so nothing above
    this struct has to think about it.

    The rows are not offsets unless they have to be. Almost every CSV file is
    rectangular, and for one of those the offset of row r is r times the width
    and the list holding it is eight bytes a row of pure redundancy. On a ten
    million row file that list was eighty megabytes written during the scan and
    read back during the fill, and it cost more than the fields themselves: the
    scan's time tracked the row count rather than the field count or the byte
    count. So the width is kept as a number and the offsets are built only when
    a row turns up that disagrees, which is a file this reader refuses anyway.
    """

    var fields: List[UInt64]
    """Every field, in file order, packed."""

    var _rows: Int
    """How many rows have been closed."""

    var _uniform: Int
    """Fields per row, while every row has agreed. Zero before the first row."""

    var _open: Int
    """Where the row being pushed began in `fields`."""

    var _starts: List[Int]
    """Where each row begins in `fields`, once a row has disagreed.

    Empty while the file is rectangular. When one is not, it is filled in for
    every row up to that point, since they all had `_uniform` fields, and
    appended to from then on.
    """

    var quotes: Int
    """How many quote bytes this scan read as structure rather than as data.

    An opening quote, a closing quote and both halves of a doubled quote are
    structure. A bare quote in the middle of an unquoted field is data, and this
    reader accepts one because pandas does. The parallel reader needs to know
    whether the file has any, because a data quote flips the parity that the
    block split is found by, so it compares this against the number of quote
    bytes in the file. `firepanda/io/split.mojo` has the argument in full.
    """

    var long: List[LongField]
    """Fields longer than `LONG_FIELD`, in increasing position order.

    Almost always empty. A file of four megabyte fields is a real file, though,
    and refusing it to save two bytes of packing would be the wrong trade.
    """

    def __init__(out self):
        """Constructs an empty scan."""
        self.fields = List[UInt64]()
        self._rows = 0
        self._uniform = 0
        self._open = 0
        self._starts = List[Int]()
        self.long = List[LongField]()
        self.quotes = 0

    def end_row(mut self):
        """Closes the row whose fields have just been pushed."""
        var here = len(self.fields)
        var got = here - self._open
        self._open = here
        self._rows += 1
        if len(self._starts) > 0:
            self._starts.append(here)
            return
        if self._rows == 1:
            self._uniform = got
            return
        if got == self._uniform:
            return
        # This row disagrees, so the offsets have to exist after all. Every row
        # before it had the same width, so they are an arithmetic sequence and
        # writing them out is the whole of the fallback.
        self._starts = List[Int](capacity=self._rows + 1)
        for r in range(self._rows):
            self._starts.append(r * self._uniform)
        self._starts.append(here)

    def push(mut self, start: Int, end: Int, escaped: Bool, quoted: Bool):
        """Records one field.

        Args:
            start: First byte of the content.
            end: One past the last byte of the content.
            escaped: Whether the content has a doubled quote in it.
            quoted: Whether the field was written inside quotes.
        """
        var length = end - start
        var stored = length
        if length >= LONG_FIELD:
            self.long.append(LongField(len(self.fields), length))
            stored = LONG_FIELD
        var bits = UInt64(start) | (UInt64(stored) << START_BITS)
        if escaped:
            bits |= ESCAPED_BIT
        if quoted:
            bits |= QUOTED_BIT
        self.fields.append(bits)

    def field(self, i: Int) -> FieldSpan:
        """Returns one field's span by its position in `fields`.

        Args:
            i: The field's position.

        Returns:
            The span.
        """
        var bits = self.fields[i]
        var start = Int(bits & START_MASK)
        var length = Int((bits >> START_BITS) & LENGTH_MASK)
        if length == LONG_FIELD:
            length = self._long_length(i)
        return FieldSpan(
            start,
            start + length,
            (bits & ESCAPED_BIT) != 0,
            (bits & QUOTED_BIT) != 0,
        )

    def _long_length(self, i: Int) -> Int:
        """Returns the recorded length of a field that did not fit.

        `long` is appended to in field order, so this is a binary search rather
        than a walk, which matters only for the pathological file where most
        fields are long.

        Args:
            i: The field's position in `fields`.

        Returns:
            The length, or `LONG_FIELD` if the entry is somehow missing, which
            cannot happen and is not worth raising over.
        """
        var lo = 0
        var hi = len(self.long)
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.long[mid].at < i:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(self.long) and self.long[lo].at == i:
            return self.long[lo].length
        return LONG_FIELD

    def __len__(self) -> Int:
        """Returns the number of rows.

        Returns:
            The row count.
        """
        return self._rows

    def width(self, row: Int) -> Int:
        """Returns how many fields a row has.

        Args:
            row: The row number.

        Returns:
            The field count.
        """
        if len(self._starts) > 0:
            return self._starts[row + 1] - self._starts[row]
        return self._uniform

    def row_start(self, row: Int) -> Int:
        """Returns where a row's fields begin in `fields`.

        Args:
            row: The row number.

        Returns:
            The position of the row's first field.
        """
        if len(self._starts) > 0:
            return self._starts[row]
        return row * self._uniform

    def at(self, row: Int, column: Int) -> FieldSpan:
        """Returns one field's span.

        Args:
            row: The row number.
            column: The field's position within the row.

        Returns:
            The span.
        """
        return self.field(self.row_start(row) + column)

    def is_ragged(self) -> Bool:
        """Returns whether the rows disagree on how many fields they have.

        Returns:
            True if any row has a different width from the first.
        """
        return len(self._starts) > 0


def scan_csv(
    data: Span[UInt8, _], dialect: Dialect, first: Int = 0
) raises -> Scan:
    """Cuts a buffer into rows of fields.

    Args:
        data: The whole file, or a prefix of it ending on a row boundary.
        dialect: The delimiter and quote character.
        first: Where to start, which must be the first byte of a row. Field
            offsets are into `data` either way, so a block scanned this way can
            be read against the whole buffer without rebasing anything.

    Returns:
        The fields and the row offsets.

    Raises:
        If a quoted field is never closed, or if a closing quote is followed by
        anything other than a delimiter or the end of the row.
    """
    var n = len(data)
    if n > MAX_START:
        raise Error(
            "csv: a buffer larger than a terabyte cannot be scanned in one"
            " piece"
        )
    var out = Scan()
    var ptr = data.unsafe_ptr()
    var at = first

    while at < n:
        # A row of nothing is not a row. This is what makes a trailing newline
        # mean the end of the last row rather than the start of an empty one.
        var lead = ptr.unsafe_offset(at).unsafe_load()
        if lead == NEWLINE:
            at += 1
            continue
        if lead == RETURN:
            # A carriage return can only be the first byte of a line if the line
            # is empty, because it terminates any field it appears in.
            at += 1
            if at < n and ptr.unsafe_offset(at).unsafe_load() == NEWLINE:
                at += 1
            continue

        while True:
            if at < n and ptr.unsafe_offset(at).unsafe_load() == dialect.quote:
                at = _quoted_field(out, data, at, dialect, out._rows)
            else:
                at = _plain_field(out, data, at, dialect)
            if at >= n:
                break
            var here = ptr.unsafe_offset(at).unsafe_load()
            if here == dialect.delimiter:
                at += 1
                # A delimiter at the end of the buffer or the line still opens
                # one more field, and that field is empty.
                if at >= n or _at_row_end(data, at):
                    out.push(at, at, False, False)
                    break
                continue
            if here == RETURN:
                at += 1
                if at < n and ptr.unsafe_offset(at).unsafe_load() == NEWLINE:
                    at += 1
                break
            at += 1
            break

        out.end_row()

    return out^


def scan_block[
    origin: ImmOrigin
](
    data: Span[UInt8, origin], dialect: Dialect, start: Int, end: Int
) raises -> Scan:
    """Cuts one block of a buffer into rows of fields.

    Narrowing the span rather than passing a length around is what keeps the
    field offsets absolute: the block's span has the buffer's own pointer, so
    every offset the scan records is an offset into the whole file and the
    reader never rebases anything.

    Args:
        data: The whole file.
        dialect: The delimiter and quote character.
        start: The block's first byte, which must begin a row.
        end: One past the block's last byte, which must end a row.

    Returns:
        The fields and the row offsets, with offsets into `data`.

    Raises:
        If the block is not cut on row boundaries, or if the file is malformed.

    Parameters:
        origin: Where the buffer lives.
    """
    return scan_csv(
        Span[UInt8, origin](unsafe_ptr=data.unsafe_ptr(), length=end),
        dialect,
        start,
    )


def _at_row_end(data: Span[UInt8, _], at: Int) -> Bool:
    """Returns whether a position is the end of a line.

    Args:
        data: The buffer.
        at: The position.

    Returns:
        True at a newline, a carriage return or the end of the buffer.
    """
    if at >= len(data):
        return True
    var c = data.unsafe_ptr().unsafe_offset(at).unsafe_load()
    return c == NEWLINE or c == RETURN


def _plain_field(
    mut out: Scan, data: Span[UInt8, _], start: Int, dialect: Dialect
) -> Int:
    """Records one unquoted field and returns the position of its terminator.

    Args:
        out: The scan being built.
        data: The buffer.
        start: The field's first byte.
        dialect: The delimiter and quote character.

    Returns:
        The index of the delimiter or line ending that closed the field, or the
        end of the buffer.
    """
    var stop = _next_special(data, start, dialect.delimiter)
    out.push(start, stop, False, False)
    return stop


def _quoted_field(
    mut out: Scan,
    data: Span[UInt8, _],
    start: Int,
    dialect: Dialect,
    row: Int,
) raises -> Int:
    """Records one quoted field and returns the position of its terminator.

    Args:
        out: The scan being built.
        data: The buffer.
        start: The opening quote.
        dialect: The delimiter and quote character.
        row: The row number, for the error message.

    Returns:
        The index of the delimiter or line ending that closed the field, or the
        end of the buffer.

    Raises:
        If the quote never closes, or if the closing quote is followed by
        anything other than a delimiter or a line ending.
    """
    var n = len(data)
    var ptr = data.unsafe_ptr()
    var at = start + 1
    var escaped = False

    while at < n:
        at = _next_quote(data, at, dialect.quote)
        if at >= n:
            break
        if (
            at + 1 < n
            and ptr.unsafe_offset(at + 1).unsafe_load() == dialect.quote
        ):
            escaped = True
            out.quotes += 2
            at += 2
            continue
        # The opening quote and this closing one, counted together now that the
        # field is known to have both.
        out.quotes += 2
        out.push(start + 1, at, escaped, True)
        var after = at + 1
        if after >= n or _at_row_end(data, after):
            return after
        if ptr.unsafe_offset(after).unsafe_load() == dialect.delimiter:
            return after
        raise Error(
            "csv: row "
            + String(row)
            + " has a closing quote at byte "
            + String(at)
            + " followed by a value instead of a delimiter"
        )

    raise Error(
        "csv: row "
        + String(row)
        + " opens a quoted field at byte "
        + String(start)
        + " that is never closed"
    )


def _next_special(data: Span[UInt8, _], start: Int, delimiter: UInt8) -> Int:
    """Returns the next delimiter or line ending at or after a position.

    Args:
        data: The buffer.
        start: Where to begin.
        delimiter: The byte between fields.

    Returns:
        The index, or the end of the buffer if there is none.
    """
    comptime width = simd_width_of[DType.uint8]()
    var n = len(data)
    var ptr = data.unsafe_ptr()
    var at = start

    # One register's worth of byte loop before the block loop starts, and this
    # is load bearing rather than a warm up. There is no packed movemask
    # reachable from this stdlib, so a register that contains a boundary has to
    # be walked byte by byte anyway to find which lane it is in. That makes a
    # hit cost the scalar walk plus the vector compare rather than instead of
    # it, and in a file whose fields are five bytes long every register hits.
    # Waiting until a field has proved it is longer than a register means the
    # block loop only runs where it can actually skip something.
    var prologue = start + PROLOGUE
    if prologue > n:
        prologue = n
    while at < prologue:
        var c = ptr.unsafe_offset(at).unsafe_load()
        if c == delimiter or c == NEWLINE or c == RETURN:
            return at
        at += 1

    var wanted = SIMD[DType.uint8, width](delimiter)
    var newlines = SIMD[DType.uint8, width](NEWLINE)
    var returns = SIMD[DType.uint8, width](RETURN)

    while at + width <= n:
        var chunk = ptr.unsafe_offset(at).unsafe_load[width=width]()
        var hit = chunk.eq(wanted) | chunk.eq(newlines) | chunk.eq(returns)
        if hit.reduce_or():
            break
        at += width

    while at < n:
        var c = ptr.unsafe_offset(at).unsafe_load()
        if c == delimiter or c == NEWLINE or c == RETURN:
            return at
        at += 1
    return n


def _next_quote(data: Span[UInt8, _], start: Int, quote: UInt8) -> Int:
    """Returns the next quote character at or after a position.

    Same shape as `_next_special` and for the same reason. A quoted field is
    usually a short label, and the ones that are not are usually long enough for
    the block loop to skip most of them.

    Args:
        data: The buffer.
        start: Where to begin.
        quote: The quote character.

    Returns:
        The index, or the end of the buffer if there is none.
    """
    comptime width = simd_width_of[DType.uint8]()
    var n = len(data)
    var ptr = data.unsafe_ptr()
    var at = start

    var prologue = start + PROLOGUE
    if prologue > n:
        prologue = n
    while at < prologue:
        if ptr.unsafe_offset(at).unsafe_load() == quote:
            return at
        at += 1

    var wanted = SIMD[DType.uint8, width](quote)
    while at + width <= n:
        var chunk = ptr.unsafe_offset(at).unsafe_load[width=width]()
        if chunk.eq(wanted).reduce_or():
            break
        at += width

    while at < n:
        if ptr.unsafe_offset(at).unsafe_load() == quote:
            return at
        at += 1
    return n


def field_bytes[
    origin: ImmOrigin
](data: Span[UInt8, origin], field: FieldSpan) -> Span[UInt8, origin]:
    """Narrows a buffer to one field's content.

    This is the whole reason the scan records offsets rather than copying: a
    field that needs no unescaping is read straight out of the buffer, and only
    the ones that do get a `List` of their own.

    Args:
        data: The buffer the span points into.
        field: The span.

    Returns:
        A view of the field's bytes, borrowed from the buffer.

    Parameters:
        origin: Where the buffer lives.
    """
    return Span[UInt8, origin](
        unsafe_ptr=data.unsafe_ptr().unsafe_offset(field.start),
        length=field.end - field.start,
    )


def collapsed_length(
    data: Span[UInt8, _], field: FieldSpan, quote: UInt8
) -> Int:
    """Returns how many bytes a field is once its doubled quotes are collapsed.

    A reader that wants to write a field straight into a column it has already
    allocated has to know this before it writes anything, and it is the only
    thing about a field that cannot be read off the index. Only worth calling on
    a field whose span says it is escaped; everything else is its own length.

    Args:
        data: The buffer the span points into.
        field: The span.
        quote: The quote character.

    Returns:
        The literal byte count.
    """
    var ptr = data.unsafe_ptr()
    var length = 0
    var at = field.start
    while at < field.end:
        length += 1
        if (
            ptr.unsafe_offset(at).unsafe_load() == quote
            and at + 1 < field.end
            and ptr.unsafe_offset(at + 1).unsafe_load() == quote
        ):
            at += 2
            continue
        at += 1
    return length


def unescape(
    data: Span[UInt8, _], field: FieldSpan, quote: UInt8
) -> List[UInt8]:
    """Copies a field's bytes out, collapsing doubled quotes.

    Only worth calling on a field whose span says it is escaped. Everything else
    can be read in place.

    Args:
        data: The buffer the span points into.
        field: The span.
        quote: The quote character.

    Returns:
        The literal bytes.
    """
    var out = List[UInt8](capacity=field.end - field.start)
    var ptr = data.unsafe_ptr()
    var at = field.start
    while at < field.end:
        var c = ptr.unsafe_offset(at).unsafe_load()
        out.append(c)
        if (
            c == quote
            and at + 1 < field.end
            and ptr.unsafe_offset(at + 1).unsafe_load() == quote
        ):
            at += 2
            continue
        at += 1
    return out^
