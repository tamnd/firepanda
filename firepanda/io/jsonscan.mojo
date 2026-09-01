"""Finding the members of a JSON object in a block of bytes.

This is the JSON equivalent of `scan.mojo` and it is separate from the reader
for the same reasons: it takes a span of bytes and knows nothing about a
`Schema`, so it can be tested and fuzzed on its own, and the reader gets to make
one pass over an array of offsets rather than a second pass over text.

What it produces is a span per key and a span per value, pointing into the
buffer. Nothing is copied and nothing is decoded. A number is a range of bytes
that looks like a number, and turning it into an Int64 is `parse.mojo`'s job on
the column that turned out to be an Int64 rather than this file's job on every
value it walks past. A string is a range of bytes with a flag saying whether a
backslash is in it, so the reader only pays for unescaping on the strings that
need it, which in a real file is close to none of them.

Nested values are found and skipped rather than descended into. `scan_object`
reports a member whose value is an object or an array as one span covering the
whole thing, because firepanda has no nested column type yet and a reader that
silently flattened one would be inventing a schema nobody asked for. When the
nested types arrive the span is already the right span to hand them.

Two things about NDJSON in particular are worth stating because they make the
reader much simpler than the CSV one. A literal newline inside a JSON string is
illegal, it has to be written `\\n`, so in valid NDJSON every 0x0A byte is a
record separator and splitting a buffer into lines is a search for one byte with
no quote state to carry across a block boundary. And a member's key is always a
string, so the key of a member is a `Value` of kind `JSON_STRING` rather than a
type of its own.

Everything is refused rather than guessed. A string that does not close, an
escape that is not one of the eight JSON allows, a `\\u` that is not four hex
digits, a number with no digits in it, a colon that is missing, a bracket that
does not match its brace: all of them are errors with the byte offset in them. A
JSON reader that repairs what it was given is a JSON reader that turns a corrupt
file into a plausible table, which is worse than failing.
"""

from std.collections.span import Span

comptime QUOTE = UInt8(34)
"""ASCII double quote, which begins and ends every JSON string."""

comptime BACKSLASH = UInt8(92)
"""ASCII backslash, the only escape character JSON has."""

comptime LEFT_BRACE = UInt8(123)
"""ASCII `{`."""

comptime RIGHT_BRACE = UInt8(125)
"""ASCII `}`."""

comptime LEFT_BRACKET = UInt8(91)
"""ASCII `[`."""

comptime RIGHT_BRACKET = UInt8(93)
"""ASCII `]`."""

comptime COLON = UInt8(58)
"""ASCII `:`, between a key and its value."""

comptime COMMA = UInt8(44)
"""ASCII `,`, between one member and the next."""

comptime SPACE = UInt8(32)
"""ASCII space, one of the four bytes JSON calls whitespace."""

comptime TAB = UInt8(9)
"""ASCII tab, another."""

comptime NEWLINE = UInt8(10)
"""ASCII line feed, another, and in NDJSON also the record separator."""

comptime RETURN = UInt8(13)
"""ASCII carriage return, the last."""

comptime JSON_NULL = 0
"""A `null`."""

comptime JSON_FALSE = 1
"""A `false`."""

comptime JSON_TRUE = 2
"""A `true`."""

comptime JSON_NUMBER = 3
"""A number, still as the bytes it was written as."""

comptime JSON_STRING = 4
"""A string, without its quotes."""

comptime JSON_OBJECT = 5
"""An object, as one span from its brace to past its matching one."""

comptime JSON_ARRAY = 6
"""An array, likewise."""

comptime MAX_DEPTH = 128
"""How deep a nested value may be before the scan gives up.

A limit is here because skipping a nested value is a loop over a depth counter
and a file can always claim more nesting than the machine has patience for. The
number is far past anything a real document does and far short of anything that
takes a noticeable time to walk.
"""


@fieldwise_init
struct Value(ImplicitlyCopyable, Movable):
    """Where one JSON value is, and what kind of thing it is.

    The three offsets look like one too many and are not. `start` and `end` are
    the content, so a string's are inside its quotes and can be handed straight
    to a parser, and `after` is where the scan continues, which is one past the
    closing quote. For every other kind the two agree.
    """

    var kind: Int
    """One of the `JSON_` constants."""

    var start: Int
    """The first byte of the content."""

    var end: Int
    """One past the last byte of the content."""

    var after: Int
    """One past the whole token, which differs from `end` only for a string."""

    var escaped: Bool
    """Whether a string holds a backslash, so the reader knows whether it can
    read the bytes in place. Always false for anything else."""


@fieldwise_init
struct Member(ImplicitlyCopyable, Movable):
    """One key and its value."""

    var key: Value
    """The name, which JSON requires to be a string."""

    var value: Value
    """What it was set to."""


def skip_space(bytes: Span[UInt8, _], var at: Int) -> Int:
    """Moves past space, tab, carriage return and line feed.

    Args:
        bytes: The buffer.
        at: Where to start.

    Returns:
        The first offset at or after `at` that is not whitespace, or the length
        of the buffer if there is none.
    """
    var length = len(bytes)
    var ptr = bytes.unsafe_ptr()
    while at < length:
        var c = ptr.unsafe_offset(at).unsafe_load()
        if c != SPACE and c != TAB and c != NEWLINE and c != RETURN:
            return at
        at += 1
    return at


def _is_digit(c: UInt8) -> Bool:
    return c >= UInt8(48) and c <= UInt8(57)


def _hex_value(c: UInt8) raises -> Int:
    if c >= UInt8(48) and c <= UInt8(57):
        return Int(c) - 48
    if c >= UInt8(97) and c <= UInt8(102):
        return Int(c) - 97 + 10
    if c >= UInt8(65) and c <= UInt8(70):
        return Int(c) - 65 + 10
    raise Error(String("json: ", chr(Int(c)), " is not a hex digit"))


def _string_at(bytes: Span[UInt8, _], at: Int) raises -> Value:
    """Scans a string whose opening quote is at `at`.

    Args:
        bytes: The buffer.
        at: The offset of the opening quote.

    Returns:
        The value, spanning the content between the quotes.

    Raises:
        Error: If the string does not close before the end of the buffer.
    """
    var length = len(bytes)
    var ptr = bytes.unsafe_ptr()
    var i = at + 1
    var escaped = False
    while i < length:
        var c = ptr.unsafe_offset(i).unsafe_load()
        if c == BACKSLASH:
            escaped = True
            i += 2
            continue
        if c == QUOTE:
            return Value(JSON_STRING, at + 1, i, i + 1, escaped)
        i += 1
    raise Error(String("json: the string at byte ", at, " never closes"))


def _number_at(bytes: Span[UInt8, _], at: Int) raises -> Value:
    """Scans a number whose first byte is at `at`.

    The grammar is JSON's, which is stricter than most parsers enforce: no
    leading plus, no leading zero on a multi digit integer, at least one digit
    after a decimal point and at least one after an exponent.

    Args:
        bytes: The buffer.
        at: The offset of the first byte.

    Returns:
        The value, spanning the bytes as written.

    Raises:
        Error: If what is there is not a number.
    """
    var length = len(bytes)
    var ptr = bytes.unsafe_ptr()
    var i = at
    if i < length and ptr.unsafe_offset(i).unsafe_load() == UInt8(45):
        i += 1
    var digits = i
    while i < length and _is_digit(ptr.unsafe_offset(i).unsafe_load()):
        i += 1
    if i == digits:
        raise Error(String("json: the number at byte ", at, " has no digits"))
    if i - digits > 1 and ptr.unsafe_offset(digits).unsafe_load() == UInt8(48):
        raise Error(
            String("json: the number at byte ", at, " has a leading zero")
        )
    if i < length and ptr.unsafe_offset(i).unsafe_load() == UInt8(46):
        i += 1
        var fraction = i
        while i < length and _is_digit(ptr.unsafe_offset(i).unsafe_load()):
            i += 1
        if i == fraction:
            raise Error(
                String(
                    "json: the number at byte ",
                    at,
                    " has nothing after its decimal point",
                )
            )
    if i < length:
        var c = ptr.unsafe_offset(i).unsafe_load()
        if c == UInt8(101) or c == UInt8(69):
            i += 1
            if i < length:
                var sign = ptr.unsafe_offset(i).unsafe_load()
                if sign == UInt8(43) or sign == UInt8(45):
                    i += 1
            var exponent = i
            while i < length and _is_digit(ptr.unsafe_offset(i).unsafe_load()):
                i += 1
            if i == exponent:
                raise Error(
                    String(
                        "json: the number at byte ",
                        at,
                        " has nothing after its exponent",
                    )
                )
    return Value(JSON_NUMBER, at, i, i, False)


def _word_at(bytes: Span[UInt8, _], at: Int, word: StringSlice) -> Bool:
    var length = len(bytes)
    var ptr = bytes.unsafe_ptr()
    var wanted = word.as_bytes()
    if at + len(wanted) > length:
        return False
    for i in range(len(wanted)):
        if ptr.unsafe_offset(at + i).unsafe_load() != wanted[i]:
            return False
    return True


def _nested_at(bytes: Span[UInt8, _], at: Int) raises -> Value:
    """Finds where a nested object or array ends, without looking inside it.

    The only thing that has to be understood in there is a string, because a
    brace inside a string is a brace and not a nesting level. Everything else is
    counted.

    Args:
        bytes: The buffer.
        at: The offset of the opening brace or bracket.

    Returns:
        The value, spanning the whole thing including its brackets.

    Raises:
        Error: If it does not close, or nests deeper than `MAX_DEPTH`.
    """
    var length = len(bytes)
    var ptr = bytes.unsafe_ptr()
    var opening = ptr.unsafe_offset(at).unsafe_load()
    var kind = JSON_OBJECT if opening == LEFT_BRACE else JSON_ARRAY
    var depth = 0
    var i = at
    while i < length:
        var c = ptr.unsafe_offset(i).unsafe_load()
        if c == QUOTE:
            i = _string_at(bytes, i).after
            continue
        if c == LEFT_BRACE or c == LEFT_BRACKET:
            depth += 1
            if depth > MAX_DEPTH:
                raise Error(
                    String(
                        "json: the value at byte ",
                        at,
                        " nests deeper than ",
                        MAX_DEPTH,
                    )
                )
        elif c == RIGHT_BRACE or c == RIGHT_BRACKET:
            depth -= 1
            if depth == 0:
                return Value(kind, at, i + 1, i + 1, False)
        i += 1
    raise Error(String("json: the value at byte ", at, " never closes"))


def scan_value(bytes: Span[UInt8, _], at: Int) raises -> Value:
    """Scans whatever value begins at `at`.

    Args:
        bytes: The buffer.
        at: The offset of the first byte of the value, whitespace already
            skipped.

    Returns:
        Where it is and what it is.

    Raises:
        Error: If there is no value there, or it is malformed.
    """
    if at >= len(bytes):
        raise Error(String("json: a value was expected at byte ", at))
    var c = bytes.unsafe_ptr().unsafe_offset(at).unsafe_load()
    if c == QUOTE:
        return _string_at(bytes, at)
    if c == LEFT_BRACE or c == LEFT_BRACKET:
        return _nested_at(bytes, at)
    if _word_at(bytes, at, "null"):
        return Value(JSON_NULL, at, at + 4, at + 4, False)
    if _word_at(bytes, at, "true"):
        return Value(JSON_TRUE, at, at + 4, at + 4, False)
    if _word_at(bytes, at, "false"):
        return Value(JSON_FALSE, at, at + 5, at + 5, False)
    if c == UInt8(45) or _is_digit(c):
        return _number_at(bytes, at)
    raise Error(
        String(
            "json: ", chr(Int(c)), " at byte ", at, " does not begin a value"
        )
    )


def scan_object(
    bytes: Span[UInt8, _], at: Int, mut out: List[Member]
) raises -> Int:
    """Scans one object, appending a member per key.

    Members are appended in the order the document writes them, and a repeated
    key is appended twice rather than deduplicated, because which of the two a
    reader should take is the reader's decision and not this one's.

    Args:
        bytes: The buffer.
        at: The offset of the opening brace, whitespace already skipped.
        out: Where the members go. Not cleared, so one list can collect a run of
            objects.

    Returns:
        One past the closing brace.

    Raises:
        Error: If it is not an object, or is not a well formed one.
    """
    var length = len(bytes)
    var ptr = bytes.unsafe_ptr()
    if at >= length or ptr.unsafe_offset(at).unsafe_load() != LEFT_BRACE:
        raise Error(String("json: an object was expected at byte ", at))
    var i = skip_space(bytes, at + 1)
    if i < length and ptr.unsafe_offset(i).unsafe_load() == RIGHT_BRACE:
        return i + 1
    while True:
        if i >= length or ptr.unsafe_offset(i).unsafe_load() != QUOTE:
            raise Error(String("json: a key was expected at byte ", i))
        var key = _string_at(bytes, i)
        i = skip_space(bytes, key.after)
        if i >= length or ptr.unsafe_offset(i).unsafe_load() != COLON:
            raise Error(
                String("json: a colon was expected after the key at byte ", i)
            )
        i = skip_space(bytes, i + 1)
        var value = scan_value(bytes, i)
        out.append(Member(key, value))
        i = skip_space(bytes, value.after)
        if i >= length:
            break
        var c = ptr.unsafe_offset(i).unsafe_load()
        if c == RIGHT_BRACE:
            return i + 1
        if c != COMMA:
            raise Error(
                String(
                    "json: a comma or a closing brace was expected at byte ", i
                )
            )
        i = skip_space(bytes, i + 1)
    raise Error(String("json: the object at byte ", at, " never closes"))


def _push_utf8(mut out: List[UInt8], code: Int):
    """Appends one code point as UTF-8."""
    if code < 0x80:
        out.append(UInt8(code))
    elif code < 0x800:
        out.append(UInt8(0xC0 | (code >> 6)))
        out.append(UInt8(0x80 | (code & 0x3F)))
    elif code < 0x10000:
        out.append(UInt8(0xE0 | (code >> 12)))
        out.append(UInt8(0x80 | ((code >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (code & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (code >> 18)))
        out.append(UInt8(0x80 | ((code >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((code >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (code & 0x3F)))


def _hex4(bytes: Span[UInt8, _], at: Int) raises -> Int:
    if at + 4 > len(bytes):
        raise Error(
            String("json: the escape at byte ", at, " wants four hex digits")
        )
    var ptr = bytes.unsafe_ptr()
    var value = 0
    for i in range(4):
        value = value * 16 + _hex_value(ptr.unsafe_offset(at + i).unsafe_load())
    return value


def unescape(bytes: Span[UInt8, _], value: Value) raises -> List[UInt8]:
    """Copies a string's bytes out, turning its escapes into what they mean.

    Only worth calling on a value whose `escaped` flag is set. Everything else
    can be read in place.

    A `\\u` escape becomes UTF-8, and a surrogate pair becomes the one character
    it stands for rather than two broken halves, because a pair is how JSON
    writes anything outside the basic plane and an emoji that survives a round
    trip through a reader as two replacement characters is a bug that surfaces a
    long way from here.

    Args:
        bytes: The buffer the value points into.
        value: The string.

    Returns:
        The literal bytes.

    Raises:
        Error: If an escape is not one JSON has, or a `\\u` is not four hex
            digits.
    """
    var out = List[UInt8](capacity=value.end - value.start)
    var ptr = bytes.unsafe_ptr()
    var at = value.start
    while at < value.end:
        var c = ptr.unsafe_offset(at).unsafe_load()
        if c != BACKSLASH:
            out.append(c)
            at += 1
            continue
        if at + 1 >= value.end:
            raise Error(
                String("json: the string ends in a backslash at byte ", at)
            )
        var what = ptr.unsafe_offset(at + 1).unsafe_load()
        at += 2
        if what == QUOTE or what == BACKSLASH or what == UInt8(47):
            out.append(what)
        elif what == UInt8(98):
            out.append(UInt8(8))
        elif what == UInt8(102):
            out.append(UInt8(12))
        elif what == UInt8(110):
            out.append(NEWLINE)
        elif what == UInt8(114):
            out.append(RETURN)
        elif what == UInt8(116):
            out.append(TAB)
        elif what == UInt8(117):
            var code = _hex4(bytes, at)
            at += 4
            if code >= 0xD800 and code <= 0xDBFF and at + 6 <= value.end:
                if ptr.unsafe_offset(
                    at
                ).unsafe_load() == BACKSLASH and ptr.unsafe_offset(
                    at + 1
                ).unsafe_load() == UInt8(
                    117
                ):
                    var low = _hex4(bytes, at + 2)
                    if low >= 0xDC00 and low <= 0xDFFF:
                        code = (
                            0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        )
                        at += 6
            _push_utf8(out, code)
        else:
            raise Error(
                String(
                    "json: \\",
                    chr(Int(what)),
                    " at byte ",
                    at - 2,
                    " is not an escape",
                )
            )
    return out^


def text_of(bytes: Span[UInt8, _], value: Value) raises -> String:
    """Reads a string value as a `String`, unescaping only if it has to.

    Args:
        bytes: The buffer the value points into.
        value: The string.

    Returns:
        What it says.

    Raises:
        Error: If an escape in it is not one JSON has.
    """
    if not value.escaped:
        return String(
            StringSlice(unsafe_from_utf8=bytes[value.start : value.end])
        )
    var raw = unescape(bytes, value)
    return String(StringSlice(unsafe_from_utf8=Span(raw)))
