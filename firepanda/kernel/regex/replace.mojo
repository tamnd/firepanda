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

### Python's engine brings a second grammar and a second scan

A call that carries flags, or a `case=False`, or a replacement naming a group by
name, or an empty pattern, is answered upstream by `re.sub` rather than by
Arrow. That changes both halves of this file at once, which is why there are
four functions here rather than two.

The replacement is read by a different grammar. `\\n` is a newline where RE2
refuses it, `\\0` is a NUL byte where RE2 means the whole match, `\\g<1>` and
`\\g<name>` exist at all, three octal digits are a character, and an unknown
letter after a backslash is an error while an unknown punctuation mark is itself
with the backslash still in front of it. There is no reading of a replacement
that satisfies both, so `parse_rewrite_python` is a second function and not a
flag on the first.

The scan is a different loop. Python never cuts the row and never steps the
cursor over a character. It keeps a match of no width wherever it finds one, and
then looks at that same position a second time with the end of the pattern
refused, so an arm that reads a character gets its turn where an arm that reads
nothing has already answered, and the search moves itself along only when that
second look finds nothing. That is one rule where Arrow has two, which is the
whole reason `str.replace("a*", "#")` is `#b#c#` upstream today and `##b#c#` the
moment a flag is added. Document 93 section 10 has why the second look is not
the same thing as a step, and what it costs to write it as one.

### What is not here

A bounded replace on Arrow's path. `str.replace(pat, repl, n)` with `n` zero or
more goes down a different path in Arrow, one that finds a match and then asks
RE2 to replace inside the text it found, with no rule about empty matches and no
advance at all. `str.replace("a*", "#", n=5)` on a row gives five markers at the
front of the untouched row, and `str.replace(r"\\b", "#", n=1)` raises on every
row including a row of plain ASCII. Copying that would put wrong answers on the
board where a refusal puts a gap, so the binding refuses `n` with a pattern and
says why. Python's path has no such problem, since `re.sub` takes a count and
stops after it, so `n` is answered there rather than refused.
"""

from std.collections.span import Span

from firepanda.kernel.regex.backtrack import Bounded, searched
from firepanda.kernel.regex.parse import decode_into
from firepanda.kernel.regex.pike import Machine, byte_of, fill_byte_offsets
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


def _is_digit(b: UInt8) -> Bool:
    """Whether a byte is one of the ten digits.

    Args:
        b: The byte.

    Returns:
        True for `0` through `9`.
    """
    return b >= UInt8(ord("0")) and b <= UInt8(ord("9"))


def _is_octal(b: UInt8) -> Bool:
    """Whether a byte is one of the eight octal digits.

    Args:
        b: The byte.

    Returns:
        True for `0` through `7`.
    """
    return b >= UInt8(ord("0")) and b <= UInt8(ord("7"))


def _is_letter(b: UInt8) -> Bool:
    """Whether a byte is an ASCII letter.

    The one test that decides whether an escape nobody recognises is an error or
    is itself. Python reserves the letters for escapes it may grow later and
    leaves the punctuation alone, so `\\s` in a replacement is refused and `\\-`
    is a backslash followed by a minus sign.

    Args:
        b: The byte.

    Returns:
        True for `a` through `z` and `A` through `Z`.
    """
    return (b >= UInt8(ord("a")) and b <= UInt8(ord("z"))) or (
        b >= UInt8(ord("A")) and b <= UInt8(ord("Z"))
    )


def _control(b: UInt8) -> Int:
    """Which character a letter after a backslash stands for, or minus one.

    The eight Python resolves in a replacement, which are the seven control
    characters and the backslash itself. They are the same eight Python resolves
    in a pattern, minus the ones that take an argument: `\\x41` is a pattern
    escape and is refused in a replacement, which was measured rather than
    assumed.

    Args:
        b: The byte after the backslash.

    Returns:
        The character it stands for, or -1 when it stands for nothing.
    """
    if b == UInt8(ord("a")):
        return 7
    if b == UInt8(ord("b")):
        return 8
    if b == UInt8(ord("f")):
        return 12
    if b == UInt8(ord("n")):
        return 10
    if b == UInt8(ord("r")):
        return 13
    if b == UInt8(ord("t")):
        return 9
    if b == UInt8(ord("v")):
        return 11
    if b == UInt8(ord("\\")):
        return 92
    return -1


def _put_point(mut into: List[UInt8], point: Int):
    """Writes one code point out as UTF-8.

    Only an octal escape reaches this and Python masks one to a byte, so nothing
    above 255 arrives and two branches are all there are. Writing it as UTF-8
    rather than as the byte itself is the point: `\\377` is the character U+00FF
    in a replacement and not the byte 0xFF, and a row is text.

    Args:
        into: The bytes, appended.
        point: The code point, which is 0 through 255.
    """
    if point < 0x80:
        into.append(UInt8(point))
        return
    into.append(UInt8(0xC0 | (point >> 6)))
    into.append(UInt8(0x80 | (point & 0x3F)))


def parse_rewrite_python(
    replacement: String, groups: Int, labels: List[String]
) -> Rewrite:
    """Reads a replacement string the way Python's `re` reads a template.

    The other reading is above and the module says why there are two. The short
    of it is that the two grammars agree on `\\\\` and on nothing else that
    matters: a digit after a backslash may be a group here or may be three octal
    digits, a letter may be a control character or may be an error, and a name
    in angle brackets is a group here and is a syntax error there.

    Four kinds of thing can follow the backslash and they are tried in Python's
    own order. A `g` opens a name in angle brackets, which may hold a number as
    well as a name. A `0` opens an octal escape of up to three digits. Any other
    digit is a group number of one or two digits, unless all three of it, the
    digit after it and the digit after that are octal, in which case it was an
    octal escape all along. Anything else is a control character if Python has
    one for it, an error if it is a letter, and itself with the backslash still
    in front of it otherwise.

    Args:
        replacement: The replacement as the caller wrote it.
        groups: How many capturing groups the pattern opened.
        labels: What each of those groups is called, one entry per group and
            empty for an unnamed one, which is what `\\g<name>` is resolved
            against.

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
        if after == UInt8(ord("g")):
            if i + 2 >= len(bytes) or bytes[i + 2] != UInt8(ord("<")):
                out.ok = False
                out.problem = String(
                    "a backslash g in a replacement opens a group name with <"
                )
                return out^
            var shut = i + 3
            while shut < len(bytes) and bytes[shut] != UInt8(ord(">")):
                shut += 1
            if shut >= len(bytes):
                out.ok = False
                out.problem = String(
                    "a group name in a replacement is not closed with >"
                )
                return out^
            var name = String(replacement[byte = i + 3 : shut])
            if name.byte_length() == 0:
                out.ok = False
                out.problem = String("a replacement names a group with no name")
                return out^
            var numeric = True
            for c in name.as_bytes():
                if not _is_digit(c):
                    numeric = False
                    break
            var which = -1
            if numeric:
                which = 0
                for c in name.as_bytes():
                    which = which * 10 + (Int(c) - ord("0"))
                if which > groups:
                    out.ok = False
                    out.problem = String(
                        "the replacement asks for group ",
                        which,
                        " and the pattern has ",
                        groups,
                    )
                    return out^
            else:
                for k in range(len(labels)):
                    if labels[k] == name:
                        which = k + 1
                        break
                if which < 0:
                    out.ok = False
                    out.problem = String(
                        "the replacement asks for a group called ",
                        name,
                        " and the pattern opens none by that name",
                    )
                    return out^
            out.start.append(Int32(begins))
            out.stop.append(Int32(len(out.literal)))
            out.group.append(Int32(which))
            begins = len(out.literal)
            i = shut + 1
            continue
        if after == UInt8(ord("0")):
            var value = 0
            var read = i + 2
            var taken = 0
            while taken < 2 and read < len(bytes) and _is_octal(bytes[read]):
                value = value * 8 + (Int(bytes[read]) - ord("0"))
                read += 1
                taken += 1
            _put_point(out.literal, value & 0xFF)
            i = read
            continue
        if _is_digit(after):
            var second = i + 2
            if second < len(bytes) and _is_digit(bytes[second]):
                var third = i + 3
                if (
                    _is_octal(after)
                    and _is_octal(bytes[second])
                    and third < len(bytes)
                    and _is_octal(bytes[third])
                ):
                    var value = (
                        (Int(after) - ord("0")) * 64
                        + (Int(bytes[second]) - ord("0")) * 8
                        + (Int(bytes[third]) - ord("0"))
                    )
                    if value > 0o377:
                        out.ok = False
                        out.problem = String(
                            "an octal escape in a replacement runs past 377"
                        )
                        return out^
                    _put_point(out.literal, value)
                    i = third + 1
                    continue
                var pair = (Int(after) - ord("0")) * 10 + (
                    Int(bytes[second]) - ord("0")
                )
                if pair > groups:
                    out.ok = False
                    out.problem = String(
                        "the replacement asks for group ",
                        pair,
                        " and the pattern has ",
                        groups,
                    )
                    return out^
                out.start.append(Int32(begins))
                out.stop.append(Int32(len(out.literal)))
                out.group.append(Int32(pair))
                begins = len(out.literal)
                i = second + 1
                continue
            var one = Int(after) - ord("0")
            if one > groups:
                out.ok = False
                out.problem = String(
                    "the replacement asks for group ",
                    one,
                    " and the pattern has ",
                    groups,
                )
                return out^
            out.start.append(Int32(begins))
            out.stop.append(Int32(len(out.literal)))
            out.group.append(Int32(one))
            begins = len(out.literal)
            i += 2
            continue
        var stands = _control(after)
        if stands >= 0:
            out.literal.append(UInt8(stands))
            i += 2
            continue
        if _is_letter(after):
            out.ok = False
            out.problem = String(
                "a backslash in a replacement is followed by a letter Python"
                " does not know"
            )
            return out^
        out.literal.append(b)
        out.literal.append(after)
        i += 2
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
    mut bounded: Bounded,
    mut offsets: List[Int],
    mut found: List[Int32],
    mut out: List[UInt8],
    limit: Int = -1,
) raises:
    """Replaces the first `limit` matches in one row, or all of them.

    The three rules this loop follows are on the module, and the one worth
    repeating beside the code is the last: an empty match that lands where the
    last match ended is thrown away and one character is copied across instead.
    That rule is the reason `p` and `lastend` are two variables rather than one,
    since `p` is where to look from and `lastend` is where the last match
    really ended, and after an empty match that was kept they are the same
    number and after a character was copied across they are not.

    The scan works in characters and the row is bytes, so `offsets` turns a
    character position back into a byte position, and every read of it goes
    through `byte_of` because a row of one byte characters is left with an
    empty table and the two positions are then the same number. It is the
    caller's list so that a column pays for it once rather than once per row,
    and so are the machine, the slots and the output.

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
        machine: The machine's buffers, reused across rows.
        bounded: The backtracker's buffers, the same way. It answers most rows
            and the machine answers the ones it hands back.
        offsets: Scratch, refilled here.
        found: Scratch for the slots of a match, refilled by every search.
        out: Where the answer goes. Emptied first.
        limit: How many matches to replace, or a negative number for all of
            them. Zero writes the row out unchanged.

    Raises:
        Error: Only what the engines raise, which is a row a backreference ran
            out of steps on.
    """
    out.clear()
    fill_byte_offsets(bytes, points, offsets)

    var n = len(points)
    var p = 0
    var lastend = -1
    var done = 0
    while p <= n and (limit < 0 or done < limit):
        var end = searched(program, points, p, machine, bounded, found)
        if end < 0:
            break
        var start = Int(found[0])
        for k in range(byte_of(offsets, p), byte_of(offsets, start)):
            out.append(bytes[k])
        if start == lastend and start == end:
            # The refused empty match. There is nothing to copy across when the
            # cursor is already at the end of the row, and the cursor still
            # moves, which is what ends the loop.
            if p < n:
                for k in range(byte_of(offsets, p), byte_of(offsets, p + 1)):
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
                    for k in range(
                        byte_of(offsets, opened), byte_of(offsets, closed)
                    ):
                        out.append(bytes[k])
        p = end
        lastend = p
        done += 1
    # A cursor that walked off the end of the row has nothing left to copy, and
    # it gets there by refusing an empty match at the last position. A cursor
    # stopped by the limit is inside the row, and the same copy is what carries
    # the rest of it across untouched.
    if p <= n:
        for k in range(byte_of(offsets, p), len(bytes)):
            out.append(bytes[k])


def replaced_python(
    program: Program,
    rewrite: Rewrite,
    bytes: Span[UInt8, _],
    points: Span[UInt32, _],
    mut machine: Machine,
    mut bounded: Bounded,
    mut offsets: List[Int],
    mut found: List[Int32],
    mut out: List[UInt8],
    limit: Int = -1,
) raises:
    """Replaces the first `limit` matches in one row the way `re.sub` does.

    The loop above is Arrow's and this one is Python's, and Python's is the
    shorter of the two because it has one rule where Arrow has three. Look from
    the cursor. Copy across whatever sits between the end of the last match and
    the start of this one. Write the replacement. Put the cursor where the match
    ended.

    The cursor never steps over a character, which is the part of this that is
    easy to get wrong and was wrong here until document 93. After a match of no
    width the cursor stays where it is and the next search is told to refuse the
    end of the pattern at that one position, so the same place is looked at
    again and a wider match can come out of it. The search moves along by itself
    when the second look finds nothing there, and the character it walks over is
    copied across by the next round's copy rather than by a rule of its own.

    That is also what makes a limit come out right. A scan stopped by its count
    writes out the rest of the row from the cursor, and the cursor is the end of
    the last match, so nothing has been consumed that was not replaced.
    `str.replace("a*", "#", n=2, case=False)` on `abc` is `##bc` upstream and
    that is the line that gets it.

    Args:
        program: The pattern, compiled with captures and for Python's engine.
        rewrite: The replacement, read by Python's grammar.
        bytes: The row as it is written.
        points: The same row as code points.
        machine: The machine's buffers, reused across rows.
        bounded: The backtracker's buffers, the same way. It answers most rows
            and the machine answers the ones it hands back.
        offsets: Scratch, refilled here.
        found: Scratch for the slots of a match, refilled by every search.
        out: Where the answer goes. Emptied first.
        limit: How many matches to replace, or a negative number for all of
            them. Zero writes the row out unchanged, which is not what `n=0`
            means to pandas on this path and is seen to by the binding.

    Raises:
        Error: Only what the engines raise, which is a row a backreference ran
            out of steps on.
    """
    out.clear()
    fill_byte_offsets(bytes, points, offsets)

    var n = len(points)
    var p = 0
    var done = 0
    var advance = False
    while p <= n and (limit < 0 or done < limit):
        var end = searched(program, points, p, machine, bounded, found, advance)
        if end < 0:
            break
        var start = Int(found[0])
        for k in range(byte_of(offsets, p), byte_of(offsets, start)):
            out.append(bytes[k])
        for part in range(len(rewrite.group)):
            for k in range(Int(rewrite.start[part]), Int(rewrite.stop[part])):
                out.append(rewrite.literal[k])
            var g = Int(rewrite.group[part])
            if g >= 0:
                var opened = Int(found[g * 2])
                var closed = Int(found[g * 2 + 1])
                if opened >= 0 and closed >= opened:
                    for k in range(
                        byte_of(offsets, opened), byte_of(offsets, closed)
                    ):
                        out.append(bytes[k])
        advance = start == end
        p = end
        done += 1
    for k in range(byte_of(offsets, p), len(bytes)):
        out.append(bytes[k])


def replaced_text(
    program: Program, rewrite: Rewrite, text: StringSlice
) raises -> String:
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

    Raises:
        Error: Only what the engines raise.
    """
    var points = List[UInt32]()
    var bytes = text.as_bytes()
    decode_into(bytes, points)
    var machine = Machine(program)
    var bounded = Bounded(program)
    var offsets = List[Int]()
    var found = List[Int32]()
    var out = List[UInt8]()
    replaced(
        program,
        rewrite,
        bytes,
        Span(points),
        machine,
        bounded,
        offsets,
        found,
        out,
    )
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def replaced_python_text(
    program: Program, rewrite: Rewrite, text: StringSlice, limit: Int = -1
) raises -> String:
    """Replaces matches in one piece of text the way `re.sub` does.

    The one shot form of `replaced_python`, which is where the rule is.

    Args:
        program: The pattern, compiled with captures and for Python's engine.
        rewrite: The replacement, read by Python's grammar.
        text: The text.
        limit: How many matches to replace, or a negative number for all.

    Returns:
        The text with the matches replaced.

    Raises:
        Error: Only what the engines raise.
    """
    var points = List[UInt32]()
    var bytes = text.as_bytes()
    decode_into(bytes, points)
    var machine = Machine(program)
    var bounded = Bounded(program)
    var offsets = List[Int]()
    var found = List[Int32]()
    var out = List[UInt8]()
    replaced_python(
        program,
        rewrite,
        bytes,
        Span(points),
        machine,
        bounded,
        offsets,
        found,
        out,
        limit,
    )
    return String(StringSlice(unsafe_from_utf8=Span(out)))
