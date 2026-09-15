"""The replacement string, and the scan that walks a row applying it.

Two things live here and they are here together because neither is the engine.
The engine answers where the leftmost first match is and what each of its groups
held. What is left is a grammar for the replacement, which is RE2's and not
Python's, and a loop that decides where to look next after a match, which is
Arrow's and is not the loop `count` runs.

### The replacement is RE2's rewrite string

`\\0` is the whole match, `\\1` through `\\9` are the groups, `\\\\` is one
backslash, and every other byte after a backslash is an error rather than
itself. So `\\n` in a replacement is refused where Python's `re` would read a
newline, and a replacement asking for a group the pattern does not have is
refused where Python would raise a different error at a different time. There
are only nine groups a replacement can name, and `\\10` is group one followed by
the character zero, which is RE2's reading and is the one place where the
grammar cannot express something the caller might reasonably want.

A group that did not take part contributes nothing rather than being an error,
which is the one generous rule in the grammar.

### The scan is not the scan that counts

Both scans are a loop that runs the pattern, does something with the match and
looks again. They disagree about both halves, in the same library, on the same
pattern, and document 80 has the measurements that say so.

The text is not cut. Counting hands RE2 what is left of the row after each
match, so `^` becomes the start of the remainder. Replacing hands RE2 the whole
row and a position to start looking from, so `^` stays the start of the row and
`str.replace("(?m)^", "#")` on a row holding a newline puts a marker at the
start and after the newline and nowhere else.

The cursor moves in characters. Counting moves it in bytes, so an empty pattern
counts the bytes of a row and one more. Replacing moves it a character at a
time, so the same empty pattern puts a marker between the characters and there
is one fewer of them on a row with an accented letter in it.

A match of no width that lands exactly where the last match ended is refused.
The scan then copies one character across and moves on, which is why
`str.replace("a*", "#")` on `abc` is `#b#c#` and not `##b#c#`: the empty match
after the `a` is thrown away and the `b` is written out instead. Python's
`re.sub` keeps that match, which is the single most visible difference between
the two and is the reason this loop is written out rather than described.

### What is not here

A bounded replace. `str.replace(pat, repl, n)` with `n` zero or more goes down a
different path in Arrow, one that finds a match and then asks RE2 to replace
inside the text it found, with no rule about empty matches and no advance at
all. `str.replace("a*", "#", n=5)` on a row gives five markers at the front of
the untouched row, and `str.replace(r"\\b", "#", n=1)` raises on every row
including a row of plain ASCII. Copying that would put wrong answers on the
board where a refusal puts a gap, so the binding refuses `n` with a pattern and
says why.
"""

from std.collections.span import Span

from firepanda.kernel.regex.parse import decode_into
from firepanda.kernel.regex.pike import Machine, byte_width
from firepanda.kernel.regex.program import Program


struct Rewrite(Movable):
    """A replacement string, read once, ready to be applied to every row.

    The parts are held as three lists rather than as a list of parts because a
    part is a piece of literal text and then at most one group, and splitting it
    that way lets the literal text of the whole replacement be one buffer. A
    replacement is applied once per match, which on a long row is thousands of
    times, so what it costs to apply is worth more than what it looks like.
    """

    var ok: Bool
    """Whether the replacement could be read at all."""

    var problem: String
    """Why not, in this library's words, and empty when it could."""

    var literal: List[UInt8]
    """Every literal byte of the replacement, end to end, with the escapes
    already resolved so that `\\\\` is one byte here."""

    var start: List[Int32]
    """Where each part's literal text begins in `literal`."""

    var stop: List[Int32]
    """Where it ends."""

    var group: List[Int32]
    """Which group to write after each part's literal text, and -1 for the last
    part, which has nothing after it."""

    def __init__(out self):
        """An empty replacement, which writes nothing and is not an error."""
        self.ok = True
        self.problem = String("")
        self.literal = []
        self.start = []
        self.stop = []
        self.group = []


def parse_rewrite(replacement: String, groups: Int) -> Rewrite:
    """Reads a replacement string the way RE2 reads a rewrite string.

    A refusal is a value rather than a raise for the reason `compile_program`
    gives: the caller is deciding what to do about a replacement, and what a
    refusal becomes in Python is a fact about the binding. Every refusal here is
    one RE2 makes too, so there is no flag saying whose it is.

    Args:
        replacement: The replacement as the caller wrote it.
        groups: How many capturing groups the pattern opened, which is what
            decides whether a number in the replacement names anything.

    Returns:
        The replacement, or the reason it cannot be read.
    """
    var out = Rewrite()
    var bytes = replacement.as_bytes()
    var i = 0
    var begins = 0
    while i < len(bytes):
        var b = bytes[i]
        if b != UInt8(ord("\\")):
            out.literal.append(b)
            i += 1
            continue
        if i + 1 >= len(bytes):
            out.ok = False
            out.problem = String("a replacement cannot end in a backslash")
            return out^
        var after = bytes[i + 1]
        if after == UInt8(ord("\\")):
            out.literal.append(after)
            i += 2
            continue
        if after < UInt8(ord("0")) or after > UInt8(ord("9")):
            out.ok = False
            out.problem = String(
                "a backslash in a replacement is followed by a digit or"
                " another backslash"
            )
            return out^
        var which = Int(after) - ord("0")
        if which > groups:
            out.ok = False
            out.problem = String(
                "the replacement asks for group ",
                which,
                " and the pattern has ",
                groups,
            )
            return out^
        out.start.append(Int32(begins))
        out.stop.append(Int32(len(out.literal)))
        out.group.append(Int32(which))
        begins = len(out.literal)
        i += 2
    # The tail, which every replacement has and which is the whole of one that
    # names no group.
    out.start.append(Int32(begins))
    out.stop.append(Int32(len(out.literal)))
    out.group.append(-1)
    return out^


def replaced(
    program: Program,
    rewrite: Rewrite,
    bytes: Span[UInt8, _],
    points: Span[UInt32, _],
    mut machine: Machine,
    mut offsets: List[Int],
    mut found: List[Int32],
    mut out: List[UInt8],
    limit: Int = -1,
):
    """Replaces the first `limit` matches in one row, or all of them.

    The three rules this loop follows are on the module, and the one worth
    repeating beside the code is the last: an empty match that lands where the
    last match ended is thrown away and one character is copied across instead.
    That rule is the reason `p` and `lastend` are two variables rather than one,
    since `p` is where to look from and `lastend` is where the last match
    really ended, and after an empty match that was kept they are the same
    number and after a character was copied across they are not.

    The scan works in characters and the row is bytes, so `offsets` turns a
    character position back into a byte position. It is the caller's list so
    that a column pays for it once rather than once per row, and so are the
    machine, the slots and the output.

    The limit is not the bounded replace the module says is not here. That one
    is Arrow's `n`, which finds a match and then replaces inside the text it
    found, and copying it would put wrong answers where a refusal puts a gap.
    This is the same scan with a counter on it, which is what SQL's
    `regexp_replace` wants: the first match and then the rest of the row
    untouched, unless the call asked for `g`. Stopping early is the only
    difference, so a limit of one and a pattern that can only match once give
    exactly what no limit gives.

    Args:
        program: The pattern, compiled with captures.
        rewrite: The replacement, already read.
        bytes: The row as it is written.
        points: The same row as code points.
        machine: The engine's buffers, reused across rows.
        offsets: Scratch, refilled here.
        found: Scratch for the slots of a match, refilled by every search.
        out: Where the answer goes. Emptied first.
        limit: How many matches to replace, or a negative number for all of
            them. Zero writes the row out unchanged.
    """
    out.clear()
    offsets.clear()
    var at = 0
    for i in range(len(points)):
        offsets.append(at)
        at += byte_width(points[i])
    offsets.append(at)

    var n = len(points)
    var p = 0
    var lastend = -1
    var done = 0
    while p <= n and (limit < 0 or done < limit):
        var end = machine.search(program, points, p, found)
        if end < 0:
            break
        var start = Int(found[0])
        for k in range(offsets[p], offsets[start]):
            out.append(bytes[k])
        if start == lastend and start == end:
            # The refused empty match. There is nothing to copy across when the
            # cursor is already at the end of the row, and the cursor still
            # moves, which is what ends the loop.
            if p < n:
                for k in range(offsets[p], offsets[p + 1]):
                    out.append(bytes[k])
            p += 1
            continue
        for part in range(len(rewrite.group)):
            for k in range(Int(rewrite.start[part]), Int(rewrite.stop[part])):
                out.append(rewrite.literal[k])
            var g = Int(rewrite.group[part])
            if g >= 0:
                var opened = Int(found[g * 2])
                var closed = Int(found[g * 2 + 1])
                if opened >= 0 and closed >= opened:
                    for k in range(offsets[opened], offsets[closed]):
                        out.append(bytes[k])
        p = end
        lastend = p
        done += 1
    # A cursor that walked off the end of the row has nothing left to copy, and
    # it gets there by refusing an empty match at the last position. A cursor
    # stopped by the limit is inside the row, and the same copy is what carries
    # the rest of it across untouched.
    if p <= n:
        for k in range(offsets[p], len(bytes)):
            out.append(bytes[k])


def replaced_text(
    program: Program, rewrite: Rewrite, text: StringSlice
) -> String:
    """Replaces every match in one piece of text.

    The one shot form, which builds everything a scan needs, uses it once and
    drops it. A caller with a column to walk wants `replaced` and its own
    buffers.

    Args:
        program: The pattern, compiled with captures.
        rewrite: The replacement, already read.
        text: The text.

    Returns:
        The text with every match replaced.
    """
    var points = List[UInt32]()
    var bytes = text.as_bytes()
    decode_into(bytes, points)
    var machine = Machine(program)
    var offsets = List[Int]()
    var found = List[Int32]()
    var out = List[UInt8]()
    replaced(
        program, rewrite, bytes, Span(points), machine, offsets, found, out
    )
    return String(StringSlice(unsafe_from_utf8=Span(out)))
