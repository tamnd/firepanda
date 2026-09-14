"""Positions in a text column, counted in characters rather than in bytes.

Every kernel already in the library that looks inside a string looks at bytes.
`text_substring` cuts a byte range, `find_bytes` returns a byte offset, and the
comparison kernels order by byte. That is right for most of them, because the
question each one is asked is about bytes: a `LIKE` pattern is a run of bytes and
a sort order over UTF-8 bytes is code point order. The docstring on `substr.mojo`
adds that a code point variant is a different kernel that should be written when
something asks for one.

Two things ask. The `str` accessor is the first. `s.str.len()` on a column
holding an accented letter is 1 in pandas and 2 in bytes, `s.str[:2]` takes two
characters and can take three or four or six bytes doing it, and `s.str.find("a")`
answers a position a caller can hand straight back to `s.str.slice`. So every
position in this file is a character, and the two words are not interchangeable
anywhere in it.

SQL is the second, which was a surprise and is worth writing down. The standard
defines `SUBSTRING` on characters and DuckDB computes it on characters, so
`substring('héllo', 1, 2)` is `hé` there and would be `h` and half a letter if it
were cut by byte. `text_character_substring` below is what a query reaches, and
`text_substring` in `substr.mojo` is the byte cut the frame API's `str_slice`
reaches and is a good deal quicker.

### A character here is a code point

Python counts code points and pandas inherits that, so this file does too. A
combining accent is its own character, which means the same visible letter is one
character when it arrived composed and two when it arrived decomposed, and
`str.normalize` exists precisely because that is a difference somebody has to be
able to remove. Grapheme clusters would be the other defensible answer and they
are not the answer pandas gives, so a library aiming at pandas cannot pick them.

### Bytes that are not UTF-8

Nothing in this library has ever promised that a text column holds valid UTF-8.
The reader copies what the file had. So the walk here counts the bytes that are
not continuation bytes, which is a character count for anything well formed and a
deterministic number for anything else. It never reads past the end of an element
and it never splits a well formed character, which are the two properties that
matter, and a column of invalid bytes gets an answer rather than an error because
every other kernel in the library gives that column an answer too.

The case kernels below cannot do their own walking, and the walk they borrow does
read past the end of a truncated element, which in a text column means into the
row after it, since the payload is one buffer with the elements end to end. So
they ask first: an element that is not well formed is copied through unchanged by
the two that write text and answers False to the three that ask a question. The
promise is kept, at the cost of a scan.

### Why this builds its output one row at a time

`text_substring` sizes every morsel's share of the payload before any bytes move,
which lets it fill a text column in parallel, and the comment there says the same
shape would work for `filter` and `take`. It does not work here. That trick rests
on the output length of an element being derivable from its input length, which
is in the view and costs no indirection to read. A character slice has no such
property: how many bytes two characters are depends on which characters they are,
so the sizing pass would have to walk the payload, which is the same work the
copying pass does. Doing it in parallel is still worth something, and it is worth
measuring before it is worth writing, so this file uses `StringBuilder` and says
so rather than copying a shape that no longer earns its complexity.

### Case is the other thing that cannot be done a byte at a time

Changing case and asking about case both live here for the reason the positions
do. A byte is not a unit of case: the upper case of `ß` is two characters, the
upper case of one Greek letter with an iota written under it is three, and
neither answer can come out of a table indexed by byte. So the case kernels hand
each element to the standard library as a `StringSlice` and let it walk the
characters, which is the one place in this file where the walk is not ours.

Document 64 measures where that library's answer and Python's differ, and the
one difference of the hundred odd that reaches a Latin alphabet is corrected
here rather than left for the reader to find.
"""

from std.collections.span import Span

from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.exec import parallel_morsels
from firepanda.kernel.pattern import find_bytes, rfind_bytes

from .mask import repair_range

comptime DOTTED_CAPITAL_I_LEAD = Byte(0xC4)
"""The first byte of U+0130, the capital I with a dot over it."""

comptime DOTTED_CAPITAL_I_TAIL = Byte(0xB0)
"""The second byte of U+0130. The pair can be tested anywhere in a run of
bytes, because 0xC4 is never a continuation byte and so can only begin a
character."""

comptime SMALL_I = Byte(0x69)
"""The letter `i`, which is the first half of what U+0130 lowers to."""

comptime DOT_ABOVE_LEAD = Byte(0xCC)
"""The first byte of U+0307, the combining dot above."""

comptime DOT_ABOVE_TAIL = Byte(0x87)
"""The second byte of U+0307."""

comptime EVERY_SPACE = 0
"""Ask `_every_character` whether every character is whitespace."""

comptime EVERY_LOWER = 1
"""Ask `_every_character` whether the text is lower case."""

comptime EVERY_UPPER = 2
"""Ask `_every_character` whether the text is upper case."""


def starts_character(b: UInt8) -> Bool:
    """Whether a byte begins a character rather than continuing one.

    Args:
        b: The byte.

    Returns:
        True for anything that is not a UTF-8 continuation byte. Continuation
        bytes are the ones whose top two bits are `10`, and every other byte
        starts something, including the bytes of a sequence that is malformed.
    """
    return (b & 0xC0) != 0x80


def character_count(bytes: Span[UInt8, _]) -> Int:
    """How many characters a run of bytes holds.

    Args:
        bytes: The bytes.

    Returns:
        The number of bytes that are not continuation bytes.
    """
    var seen = 0
    for i in range(len(bytes)):
        seen += Int(starts_character(bytes[i]))
    return seen


def character_at(bytes: Span[UInt8, _], at: Int) -> Int:
    """The byte offset where a character starts.

    Args:
        bytes: The bytes.
        at: Which character, counting from zero. Anything at or past the end
            gives the byte length, so a caller slicing between two of these gets
            the empty string rather than an error.

    Returns:
        The offset.
    """
    if at <= 0:
        return 0
    var seen = 0
    for i in range(len(bytes)):
        if starts_character(bytes[i]):
            if seen == at:
                return i
            seen += 1
    return len(bytes)


def characters_before(bytes: Span[UInt8, _], offset: Int) -> Int:
    """How many characters come before a byte offset.

    This is what turns a byte position from a search back into the character
    position a caller asked a question in.

    Args:
        bytes: The bytes.
        offset: The byte offset.

    Returns:
        The number of characters starting before `offset`.
    """
    var seen = 0
    for i in range(min(offset, len(bytes))):
        seen += Int(starts_character(bytes[i]))
    return seen


@fieldwise_init
struct Bounds(ImplicitlyCopyable, Movable):
    """A resolved slice, in characters, with all three parts made concrete."""

    var start: Int
    """The first character taken."""

    var stop: Int
    """One step past the last character taken, in the direction of the step."""

    var step: Int
    """How far to move between characters. Never zero."""


def resolve(
    n: Int, start: Optional[Int], stop: Optional[Int], step: Int
) raises -> Bounds:
    """Turns a Python slice into concrete character positions.

    These are Python's rules and not a simplification of them, because
    `s.str.slice` is documented as being Python slicing and a caller who has
    reached for it has reached for the rules they already know. A missing end
    means the far end in whichever direction the step goes, a negative position
    counts back from the end, and everything is clamped rather than refused, so
    a slice off the end of a short string is empty.

    Args:
        n: How many characters the string has.
        start: The first position, or nothing for the near end.
        stop: The position to stop before, or nothing for the far end.
        step: How far to move between characters.

    Returns:
        The resolved bounds.

    Raises:
        Error: If the step is zero, which Python refuses too.
    """
    if step == 0:
        raise Error("slice step cannot be zero")

    # The end a missing bound means depends on which way the step goes, which is
    # the part of Python's rules that is easiest to write down backwards. Going
    # forwards a slice runs from 0 up to n, and going backwards it runs from the
    # last character down past the first, so the pair below is the same pair
    # reversed and every clamp works off it.
    var near = n - 1 if step < 0 else 0
    var far = -1 if step < 0 else n
    var lowest = min(near, far)
    var highest = max(near, far)

    var begin = near
    if start:
        begin = start.value()
        if begin < 0:
            begin += n
        begin = min(max(begin, lowest), highest)

    var end = far
    if stop:
        end = stop.value()
        if end < 0:
            end += n
        end = min(max(end, lowest), highest)

    return Bounds(begin, end, step)


def text_character_length(a: StringArray) raises -> Array[DType.int64]:
    """How many characters each element holds.

    Args:
        a: The column.

    Returns:
        An int64 column, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var out = Array[DType.int64](overwritten=n)
    var validity = Bitmap(copy=a.validity)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            dst.unsafe_offset(i).unsafe_write(
                Int64(character_count(a.unsafe_bytes(i)))
            )
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def text_find(
    a: StringArray,
    needle: Span[UInt8, _],
    start: Optional[Int],
    stop: Optional[Int],
    from_end: Bool,
) raises -> Array[DType.int64]:
    """Where a run of bytes sits in each element, as a character position.

    The search itself is over bytes, because a needle that is well formed UTF-8
    cannot match anything except at a character boundary and the byte searcher is
    the fast one. Only the answer is converted, which costs one walk of the bytes
    before the hit and nothing at all on a row that has no hit.

    Args:
        a: The column.
        needle: The bytes to look for.
        start: The first character the match may start at, or nothing for the
            beginning.
        stop: The character to stop searching before, or nothing for the end.
        from_end: Whether to answer the last match rather than the first, which
            is the difference between `find` and `rfind`.

    Returns:
        An int64 column holding the character position of the match, or -1 where
        there is none, and null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var out = Array[DType.int64](overwritten=n)
    var validity = Bitmap(copy=a.validity)

    def compute(first: Int, last: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        for i in range(first, last):
            var bytes = a.unsafe_bytes(i)
            var from_ = 0
            if start:
                from_ = character_at(bytes, _clamped(start.value(), bytes))
            var until = len(bytes)
            if stop:
                until = character_at(bytes, _clamped(stop.value(), bytes))
            var at = -1
            if from_ <= until:
                if from_end:
                    at = rfind_bytes(bytes, needle, from_, until)
                else:
                    at = find_bytes(bytes[:until], needle, from_)
            var answer = -1 if at < 0 else characters_before(bytes, at)
            dst.unsafe_offset(i).unsafe_write(Int64(answer))
        repair_range(out, validity, first, last)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def _clamped(at: Int, bytes: Span[UInt8, _]) -> Int:
    """Reads a search bound as a character position against one element.

    Args:
        at: The position, which counts back from the end when it is negative.
        bytes: The element.

    Returns:
        A position between zero and the character count.
    """
    if at >= 0:
        return at
    var n = character_count(bytes)
    return max(n + at, 0)


def text_character_slice(
    a: StringArray, start: Optional[Int], stop: Optional[Int], step: Int
) raises -> StringArray:
    """Takes a range of characters out of every element.

    Args:
        a: The column.
        start: The first character, or nothing for the near end.
        stop: The character to stop before, or nothing for the far end.
        step: How far to move between characters.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: If the step is zero.
    """
    var n = len(a)
    var built = StringBuilder(capacity=n)
    var scratch = List[UInt8]()
    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        var bounds = resolve(character_count(bytes), start, stop, step)
        if step == 1:
            var from_ = character_at(bytes, bounds.start)
            var until = character_at(bytes, max(bounds.stop, bounds.start))
            built.append(bytes[from_:until])
            continue
        scratch.clear()
        _gather(bytes, bounds, scratch)
        built.append(Span(scratch))
    return built^.finish()


def text_character_substring(
    a: StringArray, start: Int, length: Optional[Int]
) raises -> StringArray:
    """Cuts the range of characters SQL's `substring` names out of every element.

    A kernel of its own rather than a call into `text_character_slice`, because
    these are not Python's rules and the difference is not cosmetic. SQL counts
    from one, and a position off either end clips the window rather than moving
    it: `substring('hello', 0, 3)` is `he` and not `hel`, since the window covers
    positions 0, 1 and 2 and no string has a position 0. A start far enough back
    that the whole window lands before the string gives the empty string for the
    same reason, where Python would clamp the start to the front and hand back
    the first characters.

    A negative length is legal and runs the window backwards from the start, so
    `substring('hello', 2, -1)` is `h`. That is DuckDB's answer and it falls out
    of treating the two numbers as the ends of a range rather than as an offset
    and a count.

    Characters and not bytes. That is what the SQL standard says and what DuckDB
    computes, and it is the one place the library counts characters for a reason
    other than pandas. `text_substring` in `substr.mojo` cuts the same shape in
    bytes and is a good deal quicker at it, so a column known to hold nothing but
    ASCII could be sent there instead, and that is worth measuring before it is
    worth writing.

    Args:
        a: The column.
        start: The first character, counting from one. Negative counts back from
            the end, so -3 starts at the third character from the end.
        length: How many characters to take, or nothing for everything to the
            end of the element. Negative runs backwards from the start.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: Only what the builder raises.
    """
    var n = len(a)
    var built = StringBuilder(capacity=n)
    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        var count = character_count(bytes)

        # Where the window starts and stops, both counting from one, before
        # anything is clipped. A zero start stays a zero, which is the whole of
        # what makes the clipping visible.
        var first = start
        if start < 0:
            first = count + start + 1
        var last = count + 1
        if length:
            last = first + length.value()
            if last < first:
                var back = first
                first = last
                last = back

        var from_ = min(max(first - 1, 0), count)
        var until = min(max(last - 1, 0), count)
        if until < from_:
            until = from_
        built.append(
            bytes[character_at(bytes, from_) : character_at(bytes, until)]
        )
    return built^.finish()


def _gather(bytes: Span[UInt8, _], bounds: Bounds, mut into: List[UInt8]):
    """Copies the characters a stepped slice selects, in the order it wants them.

    A step of one is handled by the caller as a single byte range, which is the
    case worth being quick about. Everything else lands here and is copied one
    character at a time, because a step of two takes half the characters and a
    step of minus one takes them all backwards, and neither is a contiguous run
    of bytes in the input.

    Args:
        bytes: The element.
        bounds: The resolved bounds.
        into: Where to put the bytes. Cleared by the caller and reused across
            rows, so that a column of a million short strings does not do a
            million allocations.
    """
    var at = bounds.start
    while (at > bounds.stop) if bounds.step < 0 else (at < bounds.stop):
        var from_ = character_at(bytes, at)
        var until = character_at(bytes, at + 1)
        for k in range(from_, until):
            into.append(bytes[k])
        at += bounds.step


def text_character_get(a: StringArray, at: Int) raises -> StringArray:
    """Takes one character out of every element.

    Args:
        a: The column.
        at: Which character, counting back from the end when negative.

    Returns:
        A text column of the same height. Null wherever the input is null and
        also wherever the element is too short to have that character, which is
        what pandas answers and is not what Python's indexing does.

    Raises:
        Error: If the builder cannot allocate.
    """
    var n = len(a)
    var built = StringBuilder(capacity=n)
    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        var count = character_count(bytes)
        var which = at + count if at < 0 else at
        if which < 0 or which >= count:
            built.append_null()
            continue
        built.append(
            bytes[character_at(bytes, which) : character_at(bytes, which + 1)]
        )
    return built^.finish()


def text_slice_replace(
    a: StringArray,
    start: Optional[Int],
    stop: Optional[Int],
    replacement: Span[UInt8, _],
) raises -> StringArray:
    """Puts a run of bytes where a range of characters used to be.

    Args:
        a: The column.
        start: The first character replaced, or nothing for the beginning.
        stop: The character to stop replacing before, or nothing for the end.
        replacement: The bytes to put there.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: If the builder cannot allocate.
    """
    var n = len(a)
    var built = StringBuilder(capacity=n)
    var scratch = List[UInt8]()
    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        var bounds = resolve(character_count(bytes), start, stop, 1)
        var from_ = character_at(bytes, bounds.start)
        # A stop before the start replaces nothing and inserts, which is what
        # Python's slice assignment does and what pandas inherits.
        var until = character_at(bytes, max(bounds.stop, bounds.start))
        scratch.clear()
        for k in range(from_):
            scratch.append(bytes[k])
        for k in range(len(replacement)):
            scratch.append(replacement[k])
        for k in range(until, len(bytes)):
            scratch.append(bytes[k])
        built.append(Span(scratch))
    return built^.finish()


def text_remove_prefix(
    a: StringArray, prefix: Span[UInt8, _]
) raises -> StringArray:
    """Takes a run of bytes off the front of every element that has it.

    Args:
        a: The column.
        prefix: The bytes to remove.

    Returns:
        A text column of the same height, null wherever the input is null and
        unchanged wherever the element does not begin with the prefix.

    Raises:
        Error: If the builder cannot allocate.
    """
    var n = len(a)
    var m = len(prefix)
    var built = StringBuilder(capacity=n)
    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        var cut = m if _matches_at(bytes, prefix, 0) else 0
        built.append(bytes[cut:])
    return built^.finish()


def text_remove_suffix(
    a: StringArray, suffix: Span[UInt8, _]
) raises -> StringArray:
    """Takes a run of bytes off the back of every element that has it.

    Args:
        a: The column.
        suffix: The bytes to remove.

    Returns:
        A text column of the same height, null wherever the input is null and
        unchanged wherever the element does not end with the suffix.

    Raises:
        Error: If the builder cannot allocate.
    """
    var n = len(a)
    var m = len(suffix)
    var built = StringBuilder(capacity=n)
    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        var room = len(bytes) - m
        var until = room if room >= 0 and _matches_at(
            bytes, suffix, room
        ) else len(bytes)
        built.append(bytes[:until])
    return built^.finish()


def _matches_at(hay: Span[UInt8, _], needle: Span[UInt8, _], at: Int) -> Bool:
    """Whether a run of bytes sits at an offset, bounds checked.

    `_starts_at` in `pattern.mojo` is the same test and leaves the bounds to its
    caller, because it is called from inside a loop that has already worked them
    out. Here there is no such loop and a length test either way is one compare.

    Args:
        hay: The bytes being searched.
        needle: The bytes being looked for.
        at: Where in `hay` to look.

    Returns:
        True if every byte matches and the needle fits.
    """
    var m = len(needle)
    if at < 0 or at + m > len(hay):
        return False
    for k in range(m):
        if hay[at + k] != needle[k]:
            return False
    return True


def _well_formed(bytes: Span[UInt8, _]) -> Bool:
    """Whether a run of bytes is valid UTF-8.

    Asked before any case work, and only there. The counting kernels above walk
    the bytes themselves and stop at the end of the element whatever the bytes
    say, but the standard library's case walk trusts a lead byte: a truncated
    two byte sequence at the end of an element takes the first byte of the next
    element with it, because the payload of a text column is one buffer and the
    elements sit in it end to end. That is the one thing the header of this file
    promises never happens, so an element that is not well formed never reaches
    the walk.

    Args:
        bytes: The element.

    Returns:
        True if the standard library is willing to read it as text.
    """
    try:
        _ = StringSlice(from_utf8=bytes)
    except:
        return False
    return True


def _holds_dotted_capital_i(bytes: Span[UInt8, _]) -> Bool:
    """Whether a run of bytes holds U+0130 anywhere in it.

    A pair of bytes rather than a character walk, which is sound because the
    first byte of U+0130 is 0xC4 and no continuation byte is that, so the pair
    cannot straddle a character boundary or sit inside another character.

    Args:
        bytes: The element.

    Returns:
        True if the two bytes are there, one after the other.
    """
    for k in range(len(bytes) - 1):
        if (
            bytes[k] == DOTTED_CAPITAL_I_LEAD
            and bytes[k + 1] == DOTTED_CAPITAL_I_TAIL
        ):
            return True
    return False


def _spell_the_dot(bytes: Span[UInt8, _], mut into: List[UInt8]):
    """Writes out every U+0130 as the two characters it lowers to.

    Done before the lowering rather than after, because after it the dot is
    gone and there is nothing left to put back. Lowering the two characters
    this leaves is the identity on both of them, so the answer is the same as
    if the standard library had the rule.

    Args:
        bytes: The element.
        into: Where to write, cleared first.
    """
    into.clear()
    var k = 0
    while k < len(bytes):
        if (
            k + 1 < len(bytes)
            and bytes[k] == DOTTED_CAPITAL_I_LEAD
            and bytes[k + 1] == DOTTED_CAPITAL_I_TAIL
        ):
            into.append(SMALL_I)
            into.append(DOT_ABOVE_LEAD)
            into.append(DOT_ABOVE_TAIL)
            k += 2
        else:
            into.append(bytes[k])
            k += 1


def text_case(a: StringArray, upper: Bool) raises -> StringArray:
    """Rewrites every element in one case or the other.

    An element that is not valid UTF-8 comes back exactly as it went in, which
    is the only answer available: there is nothing to change the case of, and
    the alternative of handing the bytes to a walk that will read past them is
    worse than leaving them alone.

    Args:
        a: The column.
        upper: Whether to write it upper case rather than lower case.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: If the builder cannot allocate.
    """
    var n = len(a)
    var built = StringBuilder(capacity=n)
    var scratch = List[UInt8]()
    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        if not _well_formed(bytes):
            built.append(bytes)
            continue
        if upper:
            var raised = StringSlice(unsafe_from_utf8=bytes).upper()
            built.append(raised.as_bytes())
            continue
        if _holds_dotted_capital_i(bytes):
            _spell_the_dot(bytes, scratch)
            var spelled = StringSlice(unsafe_from_utf8=Span(scratch)).lower()
            built.append(spelled.as_bytes())
            continue
        var dropped = StringSlice(unsafe_from_utf8=bytes).lower()
        built.append(dropped.as_bytes())
    return built^.finish()


def _every_character(a: StringArray, kind: Int) raises -> Array[DType.bool]:
    """Asks one question of every character of every element.

    The three questions share everything except the call in the middle, and
    they share the two rules that come with it. An element answers True only if
    it has a character in it, so the empty string is False for all three, and an
    element with no cased character in it is False for the two that are about
    case, so a row holding a digit is neither lower nor upper. An element that
    is not valid UTF-8 answers False to all three for the reason `text_case`
    gives, which is that it is not text to be asked about.

    Args:
        a: The column.
        kind: Which question, one of the three `EVERY_` words above.

    Returns:
        A bool column, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var out = Array[DType.bool](overwritten=n)
    var validity = Bitmap(copy=a.validity)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            var bytes = a.unsafe_bytes(i)
            var text = StringSlice(unsafe_from_utf8=bytes)
            var answer: Bool
            if not _well_formed(bytes):
                answer = False
            elif kind == EVERY_SPACE:
                answer = text.isspace()
            elif kind == EVERY_LOWER:
                answer = text.islower()
            else:
                answer = text.isupper()
            dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](answer))
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def text_is_space(a: StringArray) raises -> Array[DType.bool]:
    """Whether every character of each element is whitespace.

    Args:
        a: The column.

    Returns:
        A bool column, null wherever the input is null and False wherever the
        element is empty.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _every_character(a, EVERY_SPACE)


def text_is_lower(a: StringArray) raises -> Array[DType.bool]:
    """Whether each element is lower case.

    Args:
        a: The column.

    Returns:
        A bool column, null wherever the input is null and False wherever the
        element has no cased character in it.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _every_character(a, EVERY_LOWER)


def text_is_upper(a: StringArray) raises -> Array[DType.bool]:
    """Whether each element is upper case.

    Args:
        a: The column.

    Returns:
        A bool column, null wherever the input is null and False wherever the
        element has no cased character in it.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _every_character(a, EVERY_UPPER)
