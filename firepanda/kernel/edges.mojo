"""The two ends of a row: what to take off them and what to put on them.

Six of the `str` methods do not look inside a row at all. `strip` and its two
one sided spellings take characters off the ends until they reach one they were
not asked to remove, and `pad`, `center`, `ljust`, `rjust` and `zfill` put
characters on the ends until the row is as wide as it was asked to be. Nothing
in that group parses, searches or compares the middle of a row, which is why
they are one file rather than being scattered among the kernels that do.

`text_repeat` is here too, for a duller reason: it is the other method that
changes how long a row is without looking at what is in it.

### A width is a count of characters

Everything here counts characters, for the reason `chars.mojo` gives at length:
Python counts code points, pandas inherits that, and `s.str.center(12)` on a
column of accented letters has to agree with what Python would have done. So a
row that is twelve bytes and eight characters is padded to twelve characters and
comes out longer than twelve bytes, and the byte oriented kernels next door are
no help.

### What counts as whitespace

`strip` with nothing to strip removes whitespace, and Python's idea of
whitespace is not ASCII's. It is the Unicode White_Space property plus the four
C1 file and group separators, twenty nine code points in all, and it is the same
set `str.isspace` answers for. The table is written out below rather than being
derived, because it is twenty nine entries that have not changed since Unicode
4.1 and a derivation would need a property table this library does not carry.

### SQL means something narrower by whitespace

`TRIM` with nothing to trim removes whitespace too, and DuckDB's idea of it is
not Python's. It is the Unicode Zs category and nothing else, seventeen code
points, and it leaves the tab, the newline, the carriage return and the form feed
where they are. `trim(e'\tx\t')` in DuckDB comes back with both tabs on it. That
is not an oversight on their side, it is what the standard's `<trim character>`
defaults to, and a column of text that arrived from a file with a trailing tab
keeps it.

So the two callers want two different tables and the difference is not one that
can be papered over. `is_python_space` stays what pandas reaches for and
`is_sql_space` is what `TRIM` reaches for, and `_strip` takes which one to ask.

### Why the odd character goes where it goes

Padding both sides of an odd gap leaves one character over, and which side gets
it is not written down in pandas, in Python's documentation or in the Unicode
standard. It is written down in CPython's `unicode_center`, as
`left = margin / 2 + (margin & width & 1)`, and that expression is reproduced
here rather than approximated, because `"a".center(4, ".")` is `".a.."` and any
of the three obvious readings of "split it evenly" gets it wrong.
"""

from std.collections.span import Span

from firepanda.array.strings import StringArray, StringBuilder

from .chars import character_count, starts_character


def _code_point(bytes: Span[UInt8, _], at: Int) -> Int:
    """Reads the character beginning at a byte offset.

    Args:
        bytes: The element.
        at: A byte offset that starts a character.

    Returns:
        The code point, or the lead byte itself for a sequence that is malformed
        or is cut off by the end of the element. Nothing here ever reads past
        the end, and a column of bytes that are not UTF-8 gets an answer for the
        reason `chars.mojo` gives, which is that every other kernel gives one.
    """
    var lead = Int(bytes[at])
    if lead < 0x80:
        return lead
    var wide: Int
    var value: Int
    if lead >= 0xF0:
        wide = 4
        value = lead & 0x07
    elif lead >= 0xE0:
        wide = 3
        value = lead & 0x0F
    elif lead >= 0xC0:
        wide = 2
        value = lead & 0x1F
    else:
        return lead
    if at + wide > len(bytes):
        return lead
    for k in range(at + 1, at + wide):
        if starts_character(bytes[k]):
            return lead
        value = (value << 6) | Int(bytes[k] & 0x3F)
    return value


def is_python_space(code: Int) -> Bool:
    """Whether a code point is whitespace the way Python means it.

    Args:
        code: The code point.

    Returns:
        True for the twenty nine code points `str.isspace` answers True for.
    """
    if code < 0x80:
        return (code >= 0x09 and code <= 0x0D) or (
            code >= 0x1C and code <= 0x20
        )
    if code < 0x2000:
        return code == 0x85 or code == 0xA0 or code == 0x1680
    if code <= 0x200A:
        return code >= 0x2000
    return (
        code == 0x2028
        or code == 0x2029
        or code == 0x202F
        or code == 0x205F
        or code == 0x3000
    )


def is_sql_space(code: Int) -> Bool:
    """Whether a code point is whitespace the way SQL means it.

    The Unicode Zs category, which is the space, the no break space, the Ogham
    space mark, the eleven general punctuation spaces, the narrow no break
    space, the medium mathematical space and the ideographic space. Seventeen
    in all, and a strict subset of the twenty nine `is_python_space` answers for.

    What is not in it is the part worth knowing. The tab, the newline, the
    carriage return and the form feed are all whitespace to Python and none of
    them is a Zs, so `TRIM` leaves them where they are, which is what DuckDB
    does and what the standard asks for. So does the zero width space, which is
    not a space by any reading but the name.

    Args:
        code: The code point.

    Returns:
        True for the seventeen code points in Zs.
    """
    if code < 0x1680:
        return code == 0x20 or code == 0xA0
    if code <= 0x200A:
        return code == 0x1680 or code >= 0x2000
    return code == 0x202F or code == 0x205F or code == 0x3000


def _wanted(
    bytes: Span[UInt8, _],
    at: Int,
    until: Int,
    set: Span[UInt8, _],
    by_set: Bool,
    sql: Bool,
) -> Bool:
    """Whether the character at a byte offset is one the caller asked to remove.

    Args:
        bytes: The element.
        at: The character's first byte.
        until: One past the character's last byte.
        set: The characters to remove, when there are any.
        by_set: Whether to use the set rather than the whitespace table.
        sql: Which whitespace table to use when there is no set.

    Returns:
        True if it should come off.
    """
    if not by_set:
        var code = _code_point(bytes, at)
        if sql:
            return is_sql_space(code)
        return is_python_space(code)
    var width = until - at
    var k = 0
    while k < len(set):
        var end = k + 1
        while end < len(set) and not starts_character(set[end]):
            end += 1
        if end - k == width:
            var same = True
            for step in range(width):
                if set[k + step] != bytes[at + step]:
                    same = False
                    break
            if same:
                return True
        k = end
    return False


def _strip(
    a: StringArray,
    set: Span[UInt8, _],
    by_set: Bool,
    from_left: Bool,
    from_right: Bool,
    sql: Bool,
) raises -> StringArray:
    """Takes characters off one end of every element, or off both.

    The two ends are walked in byte offsets rather than in character ordinals,
    and that is the whole of what makes this cheap. `character_at` answers which
    byte a character starts at by scanning from the front of the element, so a
    loop that asks for the last character and then the one before it scans the
    element twice to move three bytes. Counting the characters first is a third
    scan and slicing at the end is two more, which came to about five passes to
    take nothing off a forty byte element and was three times the cost of the
    substring beside it in the benchmarks.

    Nothing here needs an ordinal. Going left, the next character starts at the
    next byte that is not a continuation byte, which is at most three bytes on.
    Going right, the previous character starts at the previous such byte, found
    the same way. Both ends stop the moment a character is not wanted, so an
    element with nothing to strip is two lead byte tests and the element is
    never read in the middle at all.

    Measured on the i9-13900K over four million rows of forty byte text. Rows
    with nothing to trim went from 53.9 milliseconds to 26.4 and rows with three
    spaces on each end went from 135.1 to 28.0, which is two times and 4.8
    times. The second of those is the `exec/text_trim_padded` benchmark row and
    it exists so that making the common case cheap at the cost of the real work
    cannot pass for a win.

    Args:
        a: The column.
        set: The characters to remove, when there are any.
        by_set: Whether to use the set rather than a whitespace table.
        from_left: Whether to work on the near end.
        from_right: Whether to work on the far end.
        sql: Which whitespace table to use when there is no set.

    Returns:
        A text column of the same height, null wherever the input is null.

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
        var first = 0
        var last = len(bytes)
        if from_left:
            while first < last:
                var until = first + 1
                while until < last and not starts_character(bytes[until]):
                    until += 1
                if not _wanted(bytes, first, until, set, by_set, sql):
                    break
                first = until
        if from_right:
            while last > first:
                var at = last - 1
                while at > first and not starts_character(bytes[at]):
                    at -= 1
                if not _wanted(bytes, at, last, set, by_set, sql):
                    break
                last = at
        built.append(bytes[first:last])
    return built^.finish()


def text_strip(
    a: StringArray,
    set: Span[UInt8, _],
    by_set: Bool,
    from_left: Bool,
    from_right: Bool,
) raises -> StringArray:
    """Takes characters off one end of every element, or off both, for pandas.

    Args:
        a: The column.
        set: The characters to remove, as a run of bytes holding each of them
            once or more. Read as a set of characters and not as a prefix, which
            is the thing everybody has been bitten by at least once.
        by_set: Whether to use the set. False means whitespace, which is what
            pandas does when no characters are named.
        from_left: Whether to work on the near end.
        from_right: Whether to work on the far end.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: If the builder cannot allocate.
    """
    return _strip(a, set, by_set, from_left, from_right, sql=False)


def text_trim(
    a: StringArray,
    set: Span[UInt8, _],
    by_set: Bool,
    from_left: Bool,
    from_right: Bool,
) raises -> StringArray:
    """Takes characters off one end of every element, or off both, for SQL.

    The same work `text_strip` does and the same reading of the set, which is a
    set of characters in SQL as well: `trim('abcxcba', 'abc')` is `x` in DuckDB
    and not `xcba`. The one difference is which characters come off when no set
    is named, and that difference is the whole reason this is a second entry
    point rather than a flag on the first one.

    Args:
        a: The column.
        set: The characters to remove, as a run of bytes holding each of them
            once or more.
        by_set: Whether to use the set. False means the Zs characters, which is
            what SQL removes when none are named.
        from_left: Whether to work on the near end.
        from_right: Whether to work on the far end.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: If the builder cannot allocate.
    """
    return _strip(a, set, by_set, from_left, from_right, sql=True)


def text_pad(
    a: StringArray,
    width: Int,
    fill: Span[UInt8, _],
    on_left: Bool,
    on_right: Bool,
) raises -> StringArray:
    """Puts a character on the ends of every element until it is wide enough.

    Args:
        a: The column.
        width: How many characters the answer should hold. An element that is
            already that wide or wider is handed back as it is, which is Python
            and is not what a fixed width field would do.
        fill: The character to pad with, as its bytes.
        on_left: Whether to pad the near end.
        on_right: Whether to pad the far end.

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
        var margin = width - character_count(bytes)
        if margin <= 0:
            built.append(bytes)
            continue
        var before = 0
        var after = 0
        if on_left and on_right:
            before = margin // 2 + (margin & width & 1)
            after = margin - before
        elif on_left:
            before = margin
        else:
            after = margin
        scratch.clear()
        for _ in range(before):
            scratch.extend(fill)
        scratch.extend(bytes)
        for _ in range(after):
            scratch.extend(fill)
        built.append(Span(scratch))
    return built^.finish()


def text_zfill(a: StringArray, width: Int) raises -> StringArray:
    """Puts zeros on the near end of every element, after any sign.

    The sign rule is inherited from Python and is not about strings: `-5` zero
    filled to six characters is `-00005` and not `00000-5`, because the thing
    being filled is read as a number even though it is being handled as text.

    Args:
        a: The column.
        width: How many characters the answer should hold.

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
        var margin = width - character_count(bytes)
        if margin <= 0:
            built.append(bytes)
            continue
        var signed = len(bytes) > 0 and (
            bytes[0] == UInt8(ord("-")) or bytes[0] == UInt8(ord("+"))
        )
        scratch.clear()
        if signed:
            scratch.append(bytes[0])
        for _ in range(margin):
            scratch.append(UInt8(ord("0")))
        scratch.extend(bytes[1:] if signed else bytes)
        built.append(Span(scratch))
    return built^.finish()


def text_repeat(a: StringArray, times: Int) raises -> StringArray:
    """Writes every element out several times, end to end.

    Args:
        a: The column.
        times: How many copies. Zero or fewer is the empty string, which is what
            multiplying a Python string by a number that is not positive does.

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
        if times <= 0:
            built.append(bytes[0:0])
            continue
        if times == 1:
            built.append(bytes)
            continue
        scratch.clear()
        for _ in range(times):
            scratch.extend(bytes)
        built.append(Span(scratch))
    return built^.finish()
