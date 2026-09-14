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
the five that write text and answers False to the three that ask a question. The
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
do. A byte is not a unit of case, since the character a byte belongs to is what
has a case and the two cases of a character are not always the same length. So
the case kernels hand each element to the standard library as a `StringSlice`
and let it walk the characters, which is the one place in this file where the
walk is not ours.

That library's case data is not the case data pandas answers out of. pandas
holds text in Arrow and its case methods are Arrow kernels, which use the simple
mappings and never make a row longer, while the library here uses the full
mappings for thirty nine code points and has never heard of a hundred and ten
others. `casefix.mojo` is the list of all hundred and forty nine with Arrow's
answer for each, an element is tested for holding one before it is handed over,
and one that does is written out a code point at a time instead. Document 64
measures the difference and says why the list is carried here.

`capitalize` and `swapcase` are the two case kernels the standard library has no
call for, so they are walked here rather than borrowed. Neither needs any case
data past the corrections. Capitalising is the first character raised and the
rest dropped, and swapping is decided by the mappings themselves, since a
character that lowers to something else was upper and one that raises to
something else was lower. The thirty one titlecase characters are the only ones
the mappings cannot classify, because they are in neither case while both of
their mappings would move them, and `KEPT_BY_SWAP` is that list. Both rules are
checked against Arrow over every code point there is by the generator, which
refuses to write a table if either one stops holding.

`casefold` is the one case kernel here that is not answering Arrow. pyarrow has
no casefold kernel, so pandas falls back to Python for that one method whatever
dtype the column is held as, which makes the full mappings right for it and
wrong for everything else in this file. It is also the only kernel here that can
give back a row longer in characters than the one it was given, since `ß` folds
to two letters. `casefold.mojo` is the 353 code points whose fold is not their
lower case, and everything else falls through to the lower case path.

### The questions do not borrow anything

`isspace`, `islower`, `isupper` and `istitle` are not about a mapping at all,
they are about a class, and the classes in `charclass.mojo` are Arrow's read
straight out of pyarrow. They used to be the standard library's, which
disagreed with Arrow about 1384 code points across the first three, mostly by
not having heard of the spaces above ASCII and by counting a titlecase
character as both cases at once. None of them is the loop a reader expects
either. A row is lower case when one of its characters is lower and none of
them is upper or titlecase, so a row of digits is neither case, an empty row is
neither, and the titlecase class has to be read by both questions even though
neither is named after it.

`isascii` is the exception to all of that and to most of this file. It needs no
class, no mapping and no decoding, it is a pass over the bytes looking for a
top bit, and it is the one question here that answers True on an empty row.

### A word is the thing a case table cannot tell you about

`title` and `istitle` need to know where a word starts, which pandas and Arrow
both decide by asking whether the character before was cased. That is the one
question the mappings cannot answer, because a cased character need not have
another case to be mapped to, and `ĸ` and 1294 others are exactly that. So both
of these waited for the classes rather than for the mappings, and `_is_cased`
is the three case classes together, which is Arrow's cased set exactly.

The character a word starts with is Arrow's upper case and not its titlecase,
which sounds wrong and is measured: Arrow's titlecase mapping equals its upper
case mapping for every code point in Unicode, so there is no third mapping
table here and `ǅungla` titles to `Ǆungla`.

### The five that are one rule

`isalpha`, `isnumeric`, `isdigit`, `isdecimal` and `isalnum` are the simple
shape the case questions are not: the row has a character in it and every
character it has is in the class, so the loop stops at the first character that
is not a member. Four of them read one class each and `isalnum` reads two,
because Arrow has no alphanumeric class and a character is alphanumeric exactly
when it is a letter or a number. That identity and the nesting of the three
number classes are measured against Arrow over every code point by the
generator rather than taken from the standard.

The three number questions narrow, and not where a Python programmer expects.
An ASCII four is all three. A superscript two and a half sign are numeric and
digits and not decimal, because Arrow calls anything written as a single number
sign a digit, where Python calls `½` numeric and not a digit. A Roman numeral is
numeric and neither of the others, because it is a number and a letter at once.
Arrow and Python disagree about 877 code points on the digit question alone and
pandas answers Arrow, so Arrow is what these tables hold.

The three are held as three separate classes rather than one with two range
tests inside it, because the whole of the three is under three kilobytes and a
search that answers directly beats a search that answers a question you then
have to ask again.
"""

from std.collections.span import Span
from std.collections.string import Codepoint

from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.exec import parallel_morsels
from firepanda.kernel.pattern import find_bytes, rfind_bytes

from .casefix import (
    CORRECTED_DOWN,
    CORRECTED_FROM,
    CORRECTED_UP,
    KEPT_BY_SWAP,
    LOWEST_CORRECTED_LEAD,
)
from .casefold import FOLDED_AT, FOLDED_FROM, FOLDED_TO
from .charclass import (
    ALPHA_ASCII_HIGH,
    ALPHA_ASCII_LOW,
    ALPHA_EDGES,
    DECIMAL_ASCII_HIGH,
    DECIMAL_ASCII_LOW,
    DECIMAL_EDGES,
    DIGIT_ASCII_HIGH,
    DIGIT_ASCII_LOW,
    DIGIT_EDGES,
    LOWER_ASCII_HIGH,
    LOWER_ASCII_LOW,
    LOWER_EDGES,
    NUMERIC_ASCII_HIGH,
    NUMERIC_ASCII_LOW,
    NUMERIC_EDGES,
    SPACE_ASCII_HIGH,
    SPACE_ASCII_LOW,
    SPACE_EDGES,
    TITLE_ONLY_ASCII_HIGH,
    TITLE_ONLY_ASCII_LOW,
    TITLE_ONLY_EDGES,
    UPPER_ASCII_HIGH,
    UPPER_ASCII_LOW,
    UPPER_EDGES,
)
from .mask import repair_range

comptime EVERY_SPACE = 0
"""Ask `_every_character` whether every character is whitespace."""

comptime EVERY_LOWER = 1
"""Ask `_every_character` whether the text is lower case."""

comptime EVERY_UPPER = 2
"""Ask `_every_character` whether the text is upper case."""

comptime EVERY_TITLE = 3
"""Ask `_every_character` whether the text is in title case."""

comptime ALL_ALPHA = 0
"""Ask `_all_in_class` whether every character is a letter."""

comptime ALL_NUMERIC = 1
"""Ask `_all_in_class` whether every character is a number."""

comptime ALL_DIGIT = 2
"""Ask `_all_in_class` whether every character is a digit."""

comptime ALL_DECIMAL = 3
"""Ask `_all_in_class` whether every character is a decimal digit."""

comptime ALL_ALNUM = 4
"""Ask `_all_in_class` whether every character is a letter or a number."""


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


def _listed_at(keys: Span[UInt32, _], point: UInt32) -> Int:
    """Where a code point sits in one of the tables, or minus one.

    A binary search over a hundred and fifty entries at most, so eight
    comparisons at worst. Both tables in `casefix.mojo` are in order and both
    are searched with this, the correction table to find what Arrow answers for
    a code point and the swapcase table to find whether there is anything to
    answer at all.

    Args:
        keys: A table from `casefix.mojo`, which is in order.
        point: The code point being asked about.

    Returns:
        The index, or minus one when the table does not have it.
    """
    var lo = 0
    var hi = len(keys)
    while lo < hi:
        var mid = (lo + hi) // 2
        if keys[mid] < point:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(keys) and keys[lo] == point:
        return lo
    return -1


def _in_class(
    point: UInt32, low: UInt64, high: UInt64, edges: Span[UInt32, _]
) -> Bool:
    """Whether a code point is in one of the classes in `charclass.mojo`.

    A class is held as the ranges it covers, written flat as a start, one past
    an end, the next start and so on, so a code point is in the class when the
    number of entries at or below it is odd. That is what the search leaves in
    `lo`, which is why there is no second comparison at the end the way
    `_listed_at` has one.

    The first 128 code points never reach the search. Each class carries its
    ASCII half as two words and nearly all real text is answered by a shift and
    a mask, which matters here rather than in the mapping kernels because this
    one asks per character with nothing to skip ahead on.

    Args:
        point: The code point being asked about.
        low: The class's bits for code points 0 to 63.
        high: The class's bits for code points 64 to 127.
        edges: The class's ranges from `charclass.mojo`, flat and in order.

    Returns:
        True when the code point is in the class.
    """
    if point < 64:
        return (low >> UInt64(point)) & 1 == 1
    if point < 128:
        return (high >> UInt64(point - 64)) & 1 == 1
    var lo = 0
    var hi = len(edges)
    while lo < hi:
        var mid = (lo + hi) // 2
        if edges[mid] <= point:
            lo = mid + 1
        else:
            hi = mid
    return lo % 2 == 1


def _may_need_correcting(bytes: Span[UInt8, _]) -> Bool:
    """Whether an element could hold a code point the table knows about.

    One pass over the bytes with no decoding in it. The lowest code point in
    the table is U+00DF, so a code point in the table has a lead byte of at
    least 0xC3, and an element with no such byte cannot hold one. Every ASCII
    element leaves here, which is the point.

    Args:
        bytes: The element.

    Returns:
        True if the element has to be walked a code point at a time.
    """
    for k in range(len(bytes)):
        if bytes[k] >= LOWEST_CORRECTED_LEAD:
            return True
    return False


def text_case(a: StringArray, upper: Bool) raises -> StringArray:
    """Rewrites every element in one case or the other.

    pandas holds text in Arrow and answers this out of an Arrow kernel, which
    uses the simple case mappings. The standard library here uses the full ones
    for thirty nine code points and does not know a hundred and ten others, so
    an element carrying any of the hundred and forty nine is written out a code
    point at a time with the table in `casefix.mojo` consulted first. Document
    64 measures the difference and says why it is corrected rather than left.

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
    var keys = materialize[CORRECTED_FROM]()
    var raised = materialize[CORRECTED_UP]()
    var dropped = materialize[CORRECTED_DOWN]()
    var scratch = String()
    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        if not _well_formed(bytes):
            built.append(bytes)
            continue
        var text = StringSlice(unsafe_from_utf8=bytes)
        if _may_need_correcting(bytes) and _holds_a_correction(
            text, Span(keys)
        ):
            _one_at_a_time(
                text, Span(keys), Span(raised), Span(dropped), upper, scratch
            )
            built.append(scratch.as_bytes())
            continue
        if upper:
            var one = text.upper()
            built.append(one.as_bytes())
        else:
            var one = text.lower()
            built.append(one.as_bytes())
    return built^.finish()


def _holds_a_correction(text: StringSlice, keys: Span[UInt32, _]) -> Bool:
    """Whether an element really holds one of the code points in the table.

    The byte test above says an element could, because it has a byte large
    enough somewhere in it, and most elements that pass it are ordinary
    accented text with nothing in the table at all. This is the second test,
    which decodes and searches, and it is what keeps those elements on the one
    call per element path.

    Args:
        text: The element.
        keys: `CORRECTED_FROM`.

    Returns:
        True if a code point of the element is in the table.
    """
    for point in text.codepoints():
        if _listed_at(keys, point.to_u32()) >= 0:
            return True
    return False


def _one_at_a_time(
    text: StringSlice,
    keys: Span[UInt32, _],
    raised: Span[UInt32, _],
    dropped: Span[UInt32, _],
    upper: Bool,
    mut into: String,
) raises:
    """Writes an element out a code point at a time, correcting as it goes.

    The slow path, and the only place in this file that builds a string per
    character rather than per element. A code point the table knows about is
    written as the table says, and every other one is handed to the standard
    library on its own, which gives the same answer it would have given inside
    a longer walk because it has no rule that looks at a neighbour.

    Args:
        text: The element, already known to be valid UTF-8.
        keys: `CORRECTED_FROM`.
        raised: `CORRECTED_UP`.
        dropped: `CORRECTED_DOWN`.
        upper: Which way this is going.
        into: Where to write, cleared first.

    Raises:
        Error: If a string cannot be allocated.
    """
    into = String()
    for point in text.codepoints():
        into += _case_one(point, keys, raised, dropped, upper)


def _case_one(
    point: Codepoint,
    keys: Span[UInt32, _],
    raised: Span[UInt32, _],
    dropped: Span[UInt32, _],
    upper: Bool,
) raises -> String:
    """One character in the other case, spelled the way Arrow spells it.

    The table is asked first and the standard library is asked second, which is
    the order that makes the answer always one code point long: every mapping
    the library would have expanded is in the table, so what comes back from
    here is a character rather than a run of them. The three kernels that walk
    a code point at a time all go through this, and the two that pick a case
    per character rely on the length as well as on the answer.

    Args:
        point: The character.
        keys: `CORRECTED_FROM`.
        raised: `CORRECTED_UP`.
        dropped: `CORRECTED_DOWN`.
        upper: Which way this is going.

    Returns:
        The character in the asked for case, unchanged when it has no other
        case to be in.

    Raises:
        Error: If a string cannot be allocated.
    """
    var at = _listed_at(keys, point.to_u32())
    if at >= 0:
        var answer = raised[at] if upper else dropped[at]
        return String(Codepoint(unsafe_unchecked_codepoint=answer))
    var one = String(point)
    if upper:
        return one.upper()
    return one.lower()


def text_capitalize(a: StringArray) raises -> StringArray:
    """Raises the first character of every element and drops the rest.

    That is the whole of what Arrow's kernel does, measured rather than read
    off the documentation: `tools/gen_casefix.py` runs the rule here against
    Arrow over every code point on its own and over four thousand words built
    out of the cased ones, and refuses to write a table if a single answer
    differs. So this needs no case data of its own beyond the corrections that
    `upper` and `lower` already need, and it is exactly right wherever those
    two are.

    Worth saying what it does not do, because the name suggests otherwise. It
    has no idea what a word is, so a row of several words comes back with one
    capital in it, and it does not leave a character that is already capital
    alone, so a row that arrived shouting comes back whispering after the
    first letter. Both are what pandas does.

    Args:
        a: The column.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: If the builder cannot allocate.
    """
    var n = len(a)
    var built = StringBuilder(capacity=n)
    var keys = materialize[CORRECTED_FROM]()
    var raised = materialize[CORRECTED_UP]()
    var dropped = materialize[CORRECTED_DOWN]()
    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        if len(bytes) == 0 or not _well_formed(bytes):
            built.append(bytes)
            continue
        var text = StringSlice(unsafe_from_utf8=bytes)
        if _may_need_correcting(bytes) and _holds_a_correction(
            text, Span(keys)
        ):
            var out = String()
            var first = True
            for point in text.codepoints():
                out += _case_one(
                    point, Span(keys), Span(raised), Span(dropped), first
                )
                first = False
            built.append(out.as_bytes())
            continue
        var cut = 1
        while cut < len(bytes) and not starts_character(bytes[cut]):
            cut += 1
        var whole = text[byte=0:cut].upper()
        whole += text[byte = cut : len(bytes)].lower()
        built.append(whole.as_bytes())
    return built^.finish()


def text_swapcase(a: StringArray) raises -> StringArray:
    """Writes every upper case character lower and every lower case one upper.

    This is the one case kernel the standard library cannot be asked for, so it
    is walked here in full, and walking it needs a way to tell which case a
    character is already in. The mappings answer that on their own almost
    everywhere: a character that lowers to something else was upper, one that
    raises to something else was lower, and one that neither raises nor lowers
    has no case to swap. The exceptions are the thirty one titlecase
    characters, which are in neither case and which Arrow therefore leaves
    alone even though both of their mappings would move them, and they are
    `KEPT_BY_SWAP`. With that list the rule is exact against Arrow over every
    code point there is, which the generator checks each time it runs.

    What it deliberately does not use is `islower` and `isupper`, which are the
    two questions in this file that are still answered out of the standard
    library's category data and are wrong about more than a thousand code
    points. Asking the mappings instead is both cheaper and right, and it is
    why this name is exact while `title`, which really does need to know
    whether a character is cased, is not written yet.

    Args:
        a: The column.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: If the builder cannot allocate.
    """
    var n = len(a)
    var built = StringBuilder(capacity=n)
    var keys = materialize[CORRECTED_FROM]()
    var raised = materialize[CORRECTED_UP]()
    var dropped = materialize[CORRECTED_DOWN]()
    var kept = materialize[KEPT_BY_SWAP]()
    var swapped = List[UInt8]()
    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        if not _well_formed(bytes):
            built.append(bytes)
            continue
        if _is_ascii(bytes):
            _swap_ascii(bytes, swapped)
            built.append(Span(swapped))
            continue
        var text = StringSlice(unsafe_from_utf8=bytes)
        var out = String()
        for point in text.codepoints():
            var here = String(point)
            if _listed_at(Span(kept), point.to_u32()) >= 0:
                out += here
                continue
            var down = _case_one(
                point, Span(keys), Span(raised), Span(dropped), False
            )
            if down != here:
                out += down
                continue
            out += _case_one(
                point, Span(keys), Span(raised), Span(dropped), True
            )
        built.append(out.as_bytes())
    return built^.finish()


def text_casefold(a: StringArray) raises -> StringArray:
    """Writes every element in the form two equal rows agree on.

    Folding is the third case operation and it is not a case. Nobody reads a
    folded row: its one job is that two rows a reader would call the same come
    out as the same bytes, so `Straße` and `STRASSE` both fold to `strasse`,
    and the price of that is that a row can come out longer in characters than
    it went in. `upper` and `lower` here never do, because pandas answers those
    out of Arrow and Arrow uses the simple mappings.

    This one is different and the difference is pandas', not ours. pyarrow has
    no casefold kernel, so a pandas text column falls back to Python for this
    one method whatever dtype it is held as, and both pandas backends give the
    same answer as a result. So the full mappings are the right answer here and
    the wrong answer three kernels up, which looks like an inconsistency until
    you see that each one is copying whatever pandas actually does.

    `casefold.mojo` holds the 353 code points that fold to something other
    than their lower case, which is what makes this a small table rather than a
    copy of the whole case database: everything else folds to exactly what it
    lowers to, corrections and all, so the fold table is asked first and the
    lower case path answers the rest.

    Args:
        a: The column.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: If the builder cannot allocate.
    """
    var n = len(a)
    var built = StringBuilder(capacity=n)
    var keys = materialize[CORRECTED_FROM]()
    var raised = materialize[CORRECTED_UP]()
    var dropped = materialize[CORRECTED_DOWN]()
    var folds = materialize[FOLDED_FROM]()
    var starts = materialize[FOLDED_AT]()
    var answers = materialize[FOLDED_TO]()
    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        if not _well_formed(bytes):
            built.append(bytes)
            continue
        var text = StringSlice(unsafe_from_utf8=bytes)
        if _is_ascii(bytes):
            var one = text.lower()
            built.append(one.as_bytes())
            continue
        var out = String()
        for point in text.codepoints():
            var seat = _listed_at(Span(folds), point.to_u32())
            if seat < 0:
                out += _case_one(
                    point, Span(keys), Span(raised), Span(dropped), False
                )
                continue
            for k in range(Int(starts[seat]), Int(starts[seat + 1])):
                out += String(Codepoint(unsafe_unchecked_codepoint=answers[k]))
        built.append(out.as_bytes())
    return built^.finish()


def _is_ascii(bytes: Span[UInt8, _]) -> Bool:
    """Whether a run of bytes is all ASCII.

    Args:
        bytes: The element.

    Returns:
        True if no byte has its top bit set.
    """
    for k in range(len(bytes)):
        if bytes[k] >= 0x80:
            return False
    return True


def _swap_ascii(bytes: Span[UInt8, _], mut into: List[UInt8]):
    """Swaps the case of an element that is all ASCII, a byte at a time.

    The common case by a wide margin, and the one place case really is a byte,
    since the two cases of an ASCII letter differ in one bit and nothing else
    in the range has a case at all.

    Args:
        bytes: The element, already known to be ASCII.
        into: Where to write, cleared first and reused across elements.
    """
    into.clear()
    into.reserve(len(bytes))
    for k in range(len(bytes)):
        var b = bytes[k]
        if (b >= 0x41 and b <= 0x5A) or (b >= 0x61 and b <= 0x7A):
            into.append(b ^ 0x20)
        else:
            into.append(b)


def _title_ascii(bytes: Span[UInt8, _], mut into: List[UInt8]):
    """Titles an element that is all ASCII, a byte at a time.

    The same shortcut `_swap_ascii` is, and it carries the word boundary as
    well as the case, which it can because within ASCII a cased character is
    exactly a letter and nothing else needs asking.

    Args:
        bytes: The element, already known to be ASCII.
        into: Where to write, cleared first and reused across elements.
    """
    into.clear()
    into.reserve(len(bytes))
    var after_cased = False
    for k in range(len(bytes)):
        var b = bytes[k]
        var letter = (b >= 0x41 and b <= 0x5A) or (b >= 0x61 and b <= 0x7A)
        if letter:
            into.append(b | 0x20 if after_cased else b & 0xDF)
        else:
            into.append(b)
        after_cased = letter


def _is_cased(
    point: UInt32,
    lowers: Span[UInt32, _],
    uppers: Span[UInt32, _],
    titles: Span[UInt32, _],
) -> Bool:
    """Whether a character has a case at all, which is not whether it is a letter.

    This is the question `title` and `istitle` turn on and the one the case
    mappings cannot answer, because a cased character need not have another
    case to be mapped to. `ĸ` is lower case and raises to itself, and there are
    1295 more like it, so a word boundary decided by asking whether a mapping
    moves a character is wrong on every one of them.

    It is the three classes together rather than a fourth table, since Arrow's
    cased set is exactly its lower, upper and titlecase sets put side by side,
    which was measured over every code point rather than assumed.

    Args:
        point: The code point being asked about.
        lowers: `LOWER_EDGES`.
        uppers: `UPPER_EDGES`.
        titles: `TITLE_ONLY_EDGES`.

    Returns:
        True when the character is in one of the three cases.
    """
    return (
        _in_class(point, LOWER_ASCII_LOW, LOWER_ASCII_HIGH, lowers)
        or _in_class(point, UPPER_ASCII_LOW, UPPER_ASCII_HIGH, uppers)
        or _in_class(point, TITLE_ONLY_ASCII_LOW, TITLE_ONLY_ASCII_HIGH, titles)
    )


def text_title(a: StringArray) raises -> StringArray:
    """Raises the first character of every word and drops the rest.

    A word starts at a cased character that does not follow another one, which
    means the apostrophe in `o'brien` starts a word and the row comes back
    `O'Brien`, and it means a digit does too, so `a1b` comes back `A1B`. Both
    are pandas, and both fall out of the rule rather than being written into
    it.

    The character a word starts with is Arrow's upper case for it and not its
    titlecase, which is the one surprise here and was measured rather than
    assumed: Arrow's titlecase mapping is its upper case mapping for every code
    point in Unicode, so `ǅungla` titles to `Ǆungla` with the full capital
    rather than to the digraph's own title form. That is why this needs no
    third mapping table beside `CORRECTED_UP` and `CORRECTED_DOWN`, and it is
    the whole reason this name arrived one slice after the classes did rather
    than one slice after the mappings.

    Args:
        a: The column.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: If the builder cannot allocate.
    """
    var n = len(a)
    var built = StringBuilder(capacity=n)
    var keys = materialize[CORRECTED_FROM]()
    var raised = materialize[CORRECTED_UP]()
    var dropped = materialize[CORRECTED_DOWN]()
    var lowers = materialize[LOWER_EDGES]()
    var uppers = materialize[UPPER_EDGES]()
    var titles = materialize[TITLE_ONLY_EDGES]()
    var titled = List[UInt8]()
    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        if len(bytes) == 0 or not _well_formed(bytes):
            built.append(bytes)
            continue
        if _is_ascii(bytes):
            _title_ascii(bytes, titled)
            built.append(Span(titled))
            continue
        var text = StringSlice(unsafe_from_utf8=bytes)
        var out = String()
        var after_cased = False
        for point in text.codepoints():
            out += _case_one(
                point, Span(keys), Span(raised), Span(dropped), not after_cased
            )
            after_cased = _is_cased(
                point.to_u32(), Span(lowers), Span(uppers), Span(titles)
            )
        built.append(out.as_bytes())
    return built^.finish()


def text_is_ascii(a: StringArray) raises -> Array[DType.bool]:
    """Whether every byte of each element is below 128.

    The one question in this file that needs no character data and no decoding,
    and the one that does not ask for a character to be there: the empty row is
    True here and is False for every other question in the file, because there
    is no byte in it that is not ASCII. pandas answers the same way.

    It is also the one pandas does not answer out of Arrow, since pyarrow has
    no kernel for it at all, and it does not matter here for once. Whether a
    byte has its top bit set is not a thing two Unicode tables can disagree
    about.

    Args:
        a: The column.

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
            var answer = _is_ascii(a.unsafe_bytes(i))
            dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](answer))
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def _every_character(a: StringArray, kind: Int) raises -> Array[DType.bool]:
    """Asks one question of every character of every element.

    The four questions share the two rules that come with them. An element
    answers True only if it has a character in it, so the empty string is False
    for all four, and an element with no cased character in it is False for the
    three that are about case, so a row holding a digit is none of them. An
    element that is not valid UTF-8 answers False to all four for the reason
    `text_case` gives, which is that it is not text to be asked about.

    The three about case are not each other's opposites and none is a loop that
    stops at the first character. A row is lower case when one of its characters
    is lower and none of them is upper or titlecase, so both of the first two
    read the titlecase class as well as their own, and a row can fail either by
    holding a character that is in none of the three classes.

    Title case is the only one of the four that is a question about a pair of
    characters rather than about a character. Every character that starts a word
    has to be upper or titlecase and every character that continues one has to
    be lower, and what tells those two apart is whether the character before was
    cased, which is why this loop carries a flag the other three ignore.

    Args:
        a: The column.
        kind: Which question, one of the four `EVERY_` words above.

    Returns:
        A bool column, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var out = Array[DType.bool](overwritten=n)
    var validity = Bitmap(copy=a.validity)
    var spaces = materialize[SPACE_EDGES]()
    var lowers = materialize[LOWER_EDGES]()
    var uppers = materialize[UPPER_EDGES]()
    var titles = materialize[TITLE_ONLY_EDGES]()

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            var bytes = a.unsafe_bytes(i)
            var answer = False
            if _well_formed(bytes):
                var text = StringSlice(unsafe_from_utf8=bytes)
                var seen = False
                var spoiled = False
                var after_cased = False
                for point in text.codepoints():
                    var cp = point.to_u32()
                    if kind == EVERY_SPACE:
                        if not _in_class(
                            cp, SPACE_ASCII_LOW, SPACE_ASCII_HIGH, Span(spaces)
                        ):
                            spoiled = True
                            break
                        seen = True
                        continue
                    var lower = _in_class(
                        cp, LOWER_ASCII_LOW, LOWER_ASCII_HIGH, Span(lowers)
                    )
                    var upper = _in_class(
                        cp, UPPER_ASCII_LOW, UPPER_ASCII_HIGH, Span(uppers)
                    )
                    if kind == EVERY_TITLE:
                        var third = False
                        if not lower and not upper:
                            third = _in_class(
                                cp,
                                TITLE_ONLY_ASCII_LOW,
                                TITLE_ONLY_ASCII_HIGH,
                                Span(titles),
                            )
                        if upper or third:
                            if after_cased:
                                spoiled = True
                                break
                            seen = True
                        elif lower:
                            if not after_cased:
                                spoiled = True
                                break
                            seen = True
                        after_cased = lower or upper or third
                        continue
                    if lower:
                        if kind == EVERY_UPPER:
                            spoiled = True
                            break
                        seen = True
                    elif upper:
                        if kind == EVERY_LOWER:
                            spoiled = True
                            break
                        seen = True
                    elif _in_class(
                        cp,
                        TITLE_ONLY_ASCII_LOW,
                        TITLE_ONLY_ASCII_HIGH,
                        Span(titles),
                    ):
                        spoiled = True
                        break
                answer = seen and not spoiled
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


def text_is_title(a: StringArray) raises -> Array[DType.bool]:
    """Whether each element reads as a title, word by word.

    Which is every word starting with an upper or titlecase character and
    continuing in lower case, with at least one cased character somewhere. A
    word starts after anything that is not cased, so `a1b` is not in title case
    and `A1B` is, and `O'Brien` is because the apostrophe ends a word as surely
    as a space does.

    Args:
        a: The column.

    Returns:
        A bool column, null wherever the input is null and False wherever the
        element has no cased character in it.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _every_character(a, EVERY_TITLE)


def _all_in_class(a: StringArray, kind: Int) raises -> Array[DType.bool]:
    """Asks whether every character of every element is in one class.

    The five questions that are not about case are all this one rule, which is
    that the element has a character in it and every character it has is in the
    class. There is no flag to carry and no second class to rule a row out, so
    the loop stops at the first character that is not a member rather than
    walking to the end to find out whether one of them was.

    Alphanumeric is the one of the five that reads two classes, because Arrow
    has no such class and a character is alphanumeric exactly when it is a
    letter or a number. The two have nothing in common, so which one is read
    first only decides how quickly a letter is accepted.

    An element that is not valid UTF-8 answers False for the reason `text_case`
    gives, which is that it is not text to be asked about.

    Args:
        a: The column.
        kind: Which question, one of the five `ALL_` words above.

    Returns:
        A bool column, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var out = Array[DType.bool](overwritten=n)
    var validity = Bitmap(copy=a.validity)
    var alphas = materialize[ALPHA_EDGES]()
    var numerics = materialize[NUMERIC_EDGES]()
    var digits = materialize[DIGIT_EDGES]()
    var decimals = materialize[DECIMAL_EDGES]()

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            var bytes = a.unsafe_bytes(i)
            var answer = False
            if len(bytes) > 0 and _well_formed(bytes):
                var text = StringSlice(unsafe_from_utf8=bytes)
                answer = True
                for point in text.codepoints():
                    var cp = point.to_u32()
                    var here = False
                    if kind == ALL_DIGIT:
                        here = _in_class(
                            cp,
                            DIGIT_ASCII_LOW,
                            DIGIT_ASCII_HIGH,
                            Span(digits),
                        )
                    elif kind == ALL_DECIMAL:
                        here = _in_class(
                            cp,
                            DECIMAL_ASCII_LOW,
                            DECIMAL_ASCII_HIGH,
                            Span(decimals),
                        )
                    else:
                        if kind != ALL_NUMERIC:
                            here = _in_class(
                                cp,
                                ALPHA_ASCII_LOW,
                                ALPHA_ASCII_HIGH,
                                Span(alphas),
                            )
                        if not here and kind != ALL_ALPHA:
                            here = _in_class(
                                cp,
                                NUMERIC_ASCII_LOW,
                                NUMERIC_ASCII_HIGH,
                                Span(numerics),
                            )
                    if not here:
                        answer = False
                        break
            dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](answer))
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def text_is_alpha(a: StringArray) raises -> Array[DType.bool]:
    """Whether every character of each element is a letter.

    Args:
        a: The column.

    Returns:
        A bool column, null wherever the input is null and False wherever the
        element is empty.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _all_in_class(a, ALL_ALPHA)


def text_is_numeric(a: StringArray) raises -> Array[DType.bool]:
    """Whether every character of each element is a number.

    Which is the widest of the three number questions, and takes in the Roman
    numerals and the Runic counting marks as well as everything the digit
    question takes in.

    Args:
        a: The column.

    Returns:
        A bool column, null wherever the input is null and False wherever the
        element is empty.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _all_in_class(a, ALL_NUMERIC)


def text_is_digit(a: StringArray) raises -> Array[DType.bool]:
    """Whether every character of each element is a digit.

    Narrower than the numeric question by the 239 characters that are a number
    and a letter at once, which are the Roman numerals and the Runic counting
    marks. A superscript two and a half sign are both digits here, which is
    Arrow's answer and not Python's, and pandas gives Arrow's.

    Args:
        a: The column.

    Returns:
        A bool column, null wherever the input is null and False wherever the
        element is empty.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _all_in_class(a, ALL_DIGIT)


def text_is_decimal(a: StringArray) raises -> Array[DType.bool]:
    """Whether every character of each element is a decimal digit.

    The narrowest of the three number questions and the only one whose members
    can all be a place in a base ten number, so a superscript two and a half
    sign are digits and are not decimal ones.

    Args:
        a: The column.

    Returns:
        A bool column, null wherever the input is null and False wherever the
        element is empty.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _all_in_class(a, ALL_DECIMAL)


def text_is_alnum(a: StringArray) raises -> Array[DType.bool]:
    """Whether every character of each element is a letter or a number.

    Args:
        a: The column.

    Returns:
        A bool column, null wherever the input is null and False wherever the
        element is empty.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _all_in_class(a, ALL_ALNUM)
