"""Cutting a CSV buffer into blocks that each start on a row boundary.

The reason this is not trivial is that a newline inside a quoted field is data
rather than a row separator, so a block boundary cannot be found by looking for a
newline near the target offset. Whether a given newline separates rows depends on
whether the offset is inside a quoted field, and that depends on everything
before it.

It depends on everything before it in exactly one way, though, and that is what
makes this cheap. Under RFC 4180 the quote character appears only as the opening
quote of a field, the closing quote of a field, or one half of a doubled quote
inside one. An opening quote and a closing quote each flip the state, and a
doubled quote is two quotes and so leaves it alone. So the state at any offset is
the parity of the number of quote bytes before it, and nothing else about the
preceding text matters.

Counting bytes is embarrassingly parallel and a prefix sum over one number per
block is free, so every worker learns its own starting state without reading
anything the workers before it read. From there it walks forward to the first
newline at even parity, and the block starts after it.

There is one way this can be wrong. This reader accepts a bare quote in the middle
of an unquoted field, because pandas does and because refusing it would fail on
files that are otherwise fine. Such a quote is data rather than structure, it
does not come in a pair, and it flips the parity of every offset after it.

So the split is not trusted, it is checked, and the check is two things that cost
nothing. `Scan` counts the quote bytes it consumed as structure, and `Split`
counts the quote bytes in the file. If no block raised and the two totals agree,
the parse is exactly the parse one thread would have produced, and the argument
is short enough to give in full.

A block's structural count can never exceed the number of quote bytes in its own
byte range, because every quote it counted is one of them. The blocks partition
the buffer. So if the totals agree over the whole file, they agree over every
block, which means every quote byte in every block was read as structure and none
was data. Block zero starts at offset zero, which is a row boundary outside quotes
by definition, so block zero's parse is the correct one. Its last row therefore
ends where the block ends, which makes the next block's start a row boundary
outside quotes too, and the induction runs to the end of the file.

The raise is the other half of it. If a boundary's parity were wrong, the block
before it would be cut inside a quoted field, and a quoted field with no closing
quote before the end of the block is the error `scan_csv` already raises. Between
the two, a wrong split cannot pass silently. When either fires the whole read is
done again on one thread, which is also what produces the right row number in the
message for a file that really is malformed.
"""

from std.collections.span import Span
from std.sys.info import simd_width_of

from firepanda.exec import parallel_for

from .scan import Dialect, NEWLINE


comptime _FLUSH = 255
"""Registers accumulated before the byte counters are widened.

A lane holds a count of zero or one per register, so 255 registers is the most
that can be added into a `uint8` lane without wrapping.
"""


struct Split(Movable):
    """Where the blocks of a buffer begin and end."""

    var bounds: List[Int]
    """One offset per block plus a final one, so block `i` is `[bounds[i], bounds[i + 1])`.

    Bounds are non-decreasing rather than increasing. A file with fewer row
    boundaries than blocks asked for produces empty blocks rather than an error,
    and an empty block scans to nothing.
    """

    var quotes: Int
    """How many quote bytes the whole buffer holds.

    The reader compares this against the number the blocks accounted for as
    structure. Equal means the split was exact; the module docstring has the
    argument for why.
    """

    def __init__(out self, var bounds: List[Int], quotes: Int):
        """Constructs a split from its parts.

        Args:
            bounds: The block offsets.
            quotes: The buffer's quote byte count.
        """
        self.bounds = bounds^
        self.quotes = quotes

    def blocks(self) -> Int:
        """Returns how many blocks the buffer was cut into.

        Returns:
            The block count.
        """
        return len(self.bounds) - 1


def count_bytes(
    data: Span[UInt8, _], start: Int, end: Int, wanted: UInt8
) -> Int:
    """Counts how many times a byte occurs in a range.

    Args:
        data: The buffer.
        start: The first byte to look at.
        end: One past the last.
        wanted: The byte to count.

    Returns:
        The count.
    """
    comptime width = simd_width_of[DType.uint8]()
    var ptr = data.unsafe_ptr()
    var at = start
    var total = 0

    # The counter lives in the register between flushes. Reducing every
    # iteration would put a horizontal add in a loop whose body is otherwise two
    # instructions, and the horizontal add is the expensive one.
    var needle = SIMD[DType.uint8, width](wanted)
    var counts = SIMD[DType.uint8, width](0)
    var since_flush = 0
    while at + width <= end:
        var chunk = ptr.unsafe_offset(at).unsafe_load[width=width]()
        counts += chunk.eq(needle).cast[DType.uint8]()
        at += width
        since_flush += 1
        if since_flush == _FLUSH:
            total += Int(counts.cast[DType.uint64]().reduce_add())
            counts = SIMD[DType.uint8, width](0)
            since_flush = 0
    total += Int(counts.cast[DType.uint64]().reduce_add())

    while at < end:
        if ptr.unsafe_offset(at).unsafe_load() == wanted:
            total += 1
        at += 1
    return total


def _next_quote_or_newline(
    data: Span[UInt8, _], start: Int, quote: UInt8
) -> Int:
    """Returns the next quote or line feed at or after a position.

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

    var quotes = SIMD[DType.uint8, width](quote)
    var newlines = SIMD[DType.uint8, width](NEWLINE)
    while at + width <= n:
        var chunk = ptr.unsafe_offset(at).unsafe_load[width=width]()
        if (chunk.eq(quotes) | chunk.eq(newlines)).reduce_or():
            break
        at += width

    while at < n:
        var c = ptr.unsafe_offset(at).unsafe_load()
        if c == quote or c == NEWLINE:
            return at
        at += 1
    return n


def row_start_at_or_after(
    data: Span[UInt8, _], start: Int, inside_quotes: Bool, quote: UInt8
) -> Int:
    """Returns the first row boundary at or after a position.

    Walks forward tracking the quote state, which the caller supplies for the
    starting position, and stops after the first line feed that is not inside a
    quoted field.

    A carriage return needs no special case. A file with CRLF endings has the
    return before the feed, so stopping after the feed still lands on the first
    byte of the next row. A file with bare carriage returns and no feeds at all
    has no boundary this can find, and gets one block.

    Args:
        data: The buffer.
        start: Where to begin.
        inside_quotes: Whether `start` sits inside a quoted field.
        quote: The quote character.

    Returns:
        The offset of the next row's first byte, or the end of the buffer.
    """
    var n = len(data)
    var ptr = data.unsafe_ptr()
    var inside = inside_quotes
    var at = start
    while at < n:
        at = _next_quote_or_newline(data, at, quote)
        if at >= n:
            return n
        if ptr.unsafe_offset(at).unsafe_load() == quote:
            inside = not inside
            at += 1
            continue
        at += 1
        if not inside:
            return at
    return n


def split_buffer(
    data: Span[UInt8, _], dialect: Dialect, blocks: Int
) raises -> Split:
    """Cuts a buffer into the requested number of blocks, each starting a row.

    Args:
        data: The whole file.
        dialect: The delimiter and quote character.
        blocks: How many blocks to aim for. One or fewer gives the whole buffer
            as a single block.

    Returns:
        The block offsets and the buffer's quote byte count.

    Raises:
        Error: Never directly. The signature is inherited from `parallel_for`.
    """
    var n = len(data)
    if blocks <= 1 or n == 0:
        var whole = List[Int](capacity=2)
        whole.append(0)
        whole.append(n)
        return Split(whole^, count_bytes(data, 0, n, dialect.quote))

    # The targets are where the blocks would begin if quoting did not exist.
    # Each real boundary is at or after its target, so the blocks stay in order
    # and the last one always ends at the end of the buffer.
    var targets = List[Int](capacity=blocks + 1)
    for i in range(blocks + 1):
        targets.append(n * i // blocks)

    var quotes = List[Int](length=blocks, fill=0)

    def count(i: Int) raises {mut quotes, imm}:
        quotes[i] = count_bytes(data, targets[i], targets[i + 1], dialect.quote)

    parallel_for(count, blocks)

    # The parity of the count before a target is the quote state at it, so one
    # sequential pass over one number per block gives every block its state.
    var before = List[Bool](length=blocks, fill=False)
    var total = 0
    for i in range(blocks):
        before[i] = (total & 1) == 1
        total += quotes[i]

    var bounds = List[Int](length=blocks + 1, fill=0)
    bounds[blocks] = n

    def boundary(i: Int) raises {mut bounds, imm}:
        bounds[i + 1] = row_start_at_or_after(
            data, targets[i + 1], before[i + 1], dialect.quote
        )

    parallel_for(boundary, blocks - 1)

    # A block whose boundary ran past the next target leaves the blocks after it
    # empty rather than negative.
    for i in range(1, blocks + 1):
        if bounds[i] < bounds[i - 1]:
            bounds[i] = bounds[i - 1]

    return Split(bounds^, total)
