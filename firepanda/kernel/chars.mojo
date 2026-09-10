"""Positions in a text column, counted in characters rather than in bytes.

Every kernel already in the library that looks inside a string looks at bytes.
`text_substring` cuts a byte range, `find_bytes` returns a byte offset, and the
comparison kernels order by byte. That is right for all of them, because the
question each one is asked is about bytes: a `LIKE` pattern is a run of bytes, a
sort order over UTF-8 bytes is code point order, and SQL's `substring` is defined
on bytes. The docstring on `substr.mojo` says so and adds that a code point
variant is a different kernel that should be written when something asks for one.

The `str` accessor is what asks. `s.str.len()` on a column holding an accented
letter is 1 in pandas and 2 in bytes, `s.str[:2]` takes two characters and can
take three or four or six bytes doing it, and `s.str.find("a")` answers a
position a caller can hand straight back to `s.str.slice`. So every position in
this file is a character, and the two words are not interchangeable anywhere in
it.

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
"""

from std.collections.span import Span

from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.exec import parallel_morsels
from firepanda.kernel.pattern import find_bytes, rfind_bytes

from .mask import repair_range


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
