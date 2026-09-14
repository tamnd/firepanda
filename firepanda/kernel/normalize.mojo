"""The four Unicode normalization forms over a text column.

Every other name in the `str` accessor asks what a character is or what it maps
to. This one asks what a character is equivalent to, which is a different kind of
question and the only one on the accessor that is about sequences rather than
about characters one at a time.

Unicode lets the same text be spelled more than one way. An e with an acute
accent is one character, U+00E9, or it is two, a plain e followed by a combining
acute. Both are correct, both print the same, and they are not the same bytes, so
they do not compare equal and they do not hash alike. Normalization picks one
spelling. Two strings that a reader would call the same string come out of it as
the same bytes.

There are four forms because there are two independent choices. The first is
which equivalence is meant. Canonical equivalence says two spellings are the same
text, which is the e above. Compatibility equivalence is wider and says two
spellings are the same content with the formatting thrown away, which turns the
fi ligature into the two letters, the circled one into a one, and the fullwidth
Latin letters into ordinary ones. The second choice is whether to finish by
taking things apart or by putting them back together. So NFD is canonical and
taken apart, NFC is canonical and put back together, and NFKD and NFKC are the
same two with the wider relation.

NFC is the one worth knowing about because it is what nearly everything else
assumes. It is what the web platform normalizes to, it is what most text arrives
in already, and it is the form in which the accented e is one character.

### Where the answers come from

CPython, through the tables `tools/gen_normalize.py` writes into
`normalize_data.mojo`. That is worth being explicit about because the file next
door takes its answers from Arrow instead, and the two sit in the same accessor.

pandas holds text in Arrow, so most of the accessor is answered by an Arrow
kernel and Arrow is the thing to copy. `str.normalize` is not. It is defined once
in `ObjectStringArrayMixin` as `unicodedata.normalize` applied a row at a time,
nothing overrides it, and there is no Arrow normalization kernel for anything to
call even if something wanted to. So on every backend pandas has, this name is
CPython's answer, and CPython's is the one reproduced here.

Which library is the authority is therefore decided per name by reading what
pandas actually calls, rather than by a rule about the accessor. Documents 64 and
74 are the two halves of that: 64 is a name where Arrow and the standard library
disagree and Arrow wins, and 74 is a name where the standard library is the only
one in the room.

### Four steps, of which two are conditional

Decompose. Every character is replaced by its decomposition, which the generator
has already expanded all the way, so one lookup gives the final sequence and
nothing here recurses. Which table is consulted is the only place the K in the
name shows up.

Order. A run of combining marks is sorted by combining class. This is what makes
the form canonical rather than merely decomposed: an acute and a cedilla on the
same letter can be written in either order and mean the same thing, so one order
has to be picked, and the standard picks lowest class first. The sort has to be
stable, because two marks of the same class are not interchangeable and swapping
them changes the text.

Compose, for NFC and NFKC only. Walk forward holding the last starter and put
back every pair that the table says goes back together and that nothing blocks.

Encode. The code points are written back out as UTF-8.

### The two things that cost no table

Hangul is arithmetic. A Korean syllable is a lead, a vowel and an optional trail
packed into one code point by a formula, so taking one apart and putting it back
together are both a division and a multiplication. Eleven thousand one hundred
and seventy two syllables that would otherwise be the largest thing in the
generated file are three lines each here instead.

An element that is entirely ASCII is already in all four forms and is copied
straight through. No ASCII character has a decomposition of either kind, none has
a combining class, and no pair of them composes, so the answer is the input and
the check is one pass looking for a byte at or above 0x80. That is the same
bargain `text_case` makes next door and it is worth more here, because the
general path allocates two lists and walks the element four times.

An element that is not valid UTF-8 is copied straight through as well, for the
same reason the case kernels do it: there is nothing to normalize, and the
alternative is a decode that reads past the end of the element into the next one.
"""

from std.collections.span import Span

from firepanda.array.strings import StringArray, StringBuilder

from firepanda.kernel.normalize_data import (
    CCC_KEYS,
    CCC_VALUES,
    COMPOSE_PAIRS,
    COMPOSE_VALUES,
    NFD_KEYS,
    NFD_STARTS,
    NFD_VALUES,
    NFKD_KEYS,
    NFKD_STARTS,
    NFKD_VALUES,
)

comptime SBASE: UInt32 = 0xAC00
"""The first Hangul syllable."""

comptime LBASE: UInt32 = 0x1100
"""The first Hangul lead consonant."""

comptime VBASE: UInt32 = 0x1161
"""The first Hangul vowel."""

comptime TBASE: UInt32 = 0x11A7
"""One below the first Hangul trailing consonant, which is how the standard
writes it so that a trail index of zero means there is no trailing consonant."""

comptime LCOUNT: UInt32 = 19
"""How many Hangul lead consonants there are."""

comptime VCOUNT: UInt32 = 21
"""How many Hangul vowels there are."""

comptime TCOUNT: UInt32 = 28
"""How many trailing consonants there are, counting the absence of one."""

comptime NCOUNT: UInt32 = VCOUNT * TCOUNT
"""How many syllables share a lead consonant."""

comptime SCOUNT: UInt32 = LCOUNT * NCOUNT
"""How many Hangul syllables there are, which is eleven thousand one hundred and
seventy two characters this file answers without a table."""


def _keyed_at(keys: Span[UInt32, _], point: UInt32) -> Int:
    """Where a code point sits in one of the sorted key lists, or minus one.

    A bisection over at most five thousand eight hundred and fifty seven
    entries, so thirteen comparisons at worst. All three key lists in
    `normalize_data.mojo` are in code point order and all three are searched
    with this.

    Args:
        keys: A key list from `normalize_data.mojo`, which is in order.
        point: The code point being asked about.

    Returns:
        The index, or minus one when the list does not have it.
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


def _paired_at(pairs: Span[UInt64, _], key: UInt64) -> Int:
    """Where a packed pair sits in the composition table, or minus one.

    The same bisection over a different width. A pair is the starter shifted up
    by twenty one bits with the following character below it, which fits in one
    number because a code point needs twenty one bits, and holding it that way
    means the composition table is one sorted list rather than a list of lists.

    Args:
        pairs: `COMPOSE_PAIRS`, which is in order.
        key: The two code points packed together.

    Returns:
        The index, or minus one when the pair does not compose.
    """
    var lo = 0
    var hi = len(pairs)
    while lo < hi:
        var mid = (lo + hi) // 2
        if pairs[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(pairs) and pairs[lo] == key:
        return lo
    return -1


def _class_of(
    keys: Span[UInt32, _], values: Span[UInt8, _], point: UInt32
) -> UInt8:
    """A character's combining class, which is zero for nearly everything.

    The class is what canonical ordering sorts on. Zero means the character is a
    starter, which is to say something a mark can attach to rather than a mark
    itself, and the nine hundred and twenty two characters with a class of their
    own are the marks.

    Args:
        keys: `CCC_KEYS`.
        values: `CCC_VALUES`.
        point: The character.

    Returns:
        The class, or zero.
    """
    var at = _keyed_at(keys, point)
    if at < 0:
        return 0
    return values[at]


def _decompose_hangul(point: UInt32, mut into: List[UInt32]) -> Bool:
    """Takes a Hangul syllable apart by arithmetic, or says it is not one.

    The whole of the Hangul decomposition table, which is why there is no Hangul
    decomposition table. A syllable's index below `SBASE` divides into a lead, a
    vowel and a trail, and a trail index of zero means the syllable has two
    parts rather than three.

    Args:
        point: The character.
        into: Where the parts go, appended.

    Returns:
        True if it was a syllable and parts were written.
    """
    if point < SBASE or point >= SBASE + SCOUNT:
        return False
    var index = point - SBASE
    into.append(LBASE + index // NCOUNT)
    into.append(VBASE + (index % NCOUNT) // TCOUNT)
    var trail = index % TCOUNT
    if trail != 0:
        into.append(TBASE + trail)
    return True


def _compose_hangul(first: UInt32, second: UInt32) -> UInt32:
    """Puts two Hangul jamo together by arithmetic, or answers zero.

    Two cases, because a syllable is built in two steps. A lead and a vowel make
    a syllable with no trailing consonant, and a syllable with no trailing
    consonant takes one. Zero is safe as the answer meaning no, because U+0000
    is not a Hangul anything.

    Args:
        first: The character on the left.
        second: The character on the right.

    Returns:
        The syllable, or zero when the two do not make one.
    """
    if (
        first >= LBASE
        and first < LBASE + LCOUNT
        and second >= VBASE
        and second < VBASE + VCOUNT
    ):
        return SBASE + ((first - LBASE) * VCOUNT + (second - VBASE)) * TCOUNT
    if (
        first >= SBASE
        and first < SBASE + SCOUNT
        and (first - SBASE) % TCOUNT == 0
        and second > TBASE
        and second < TBASE + TCOUNT
    ):
        return first + (second - TBASE)
    return 0


def _order(mut points: List[UInt32], mut classes: List[UInt8]):
    """Sorts each run of combining marks by combining class, stably.

    An insertion sort that only ever swaps a pair where the left one has the
    higher class and the right one is a mark. Swapping only on a strict
    difference is what makes it stable, and stability is the point: two marks of
    the same class are ordered by where they were written and exchanging them
    changes the text. A starter has class zero and is never the right hand side
    of a swap, so a starter is a wall and each run is sorted on its own without
    the loop ever having to find where a run begins.

    The classes travel alongside the code points rather than being looked up
    again, because a lookup is a bisection and a sort asks for the same one
    repeatedly.

    Args:
        points: The decomposed code points, reordered in place.
        classes: Their combining classes, reordered the same way.
    """
    var i = 1
    while i < len(points):
        var here = classes[i]
        if here != 0 and classes[i - 1] > here:
            var point = points[i - 1]
            var klass = classes[i - 1]
            points[i - 1] = points[i]
            classes[i - 1] = here
            points[i] = point
            classes[i] = klass
            i = 1 if i == 1 else i - 1
        else:
            i += 1


def _compose(
    mut points: List[UInt32],
    mut classes: List[UInt8],
    pairs: Span[UInt64, _],
    composed: Span[UInt32, _],
):
    """Puts back together every pair that goes back together, in place.

    The algorithm from the standard. Walk forward holding the position of the
    last starter. A character joins that starter when the pair is in the table
    and nothing between them blocks it, and a character blocks when it sits
    between the two with a combining class that is not lower than the
    candidate's.

    The blocking rule is the part worth stating rather than just writing.
    Without it, two marks on one letter would each be able to reach the letter
    and the answer would depend on which one was looked at first, so a letter
    with an acute and a cedilla would come out differently depending on the
    order they were written in, which is exactly what the previous step just
    finished removing.

    Args:
        points: The ordered code points, rewritten in place and shortened.
        classes: Their combining classes, kept alongside.
        pairs: `COMPOSE_PAIRS`.
        composed: `COMPOSE_VALUES`.
    """
    var kept = 0
    var starter = -1
    var last_class: Int = -1
    for i in range(len(points)):
        var point = points[i]
        var here = Int(classes[i])
        if starter >= 0 and last_class < here:
            var joined = _compose_hangul(points[starter], point)
            if joined == 0:
                var at = _paired_at(
                    pairs, (UInt64(points[starter]) << 21) | UInt64(point)
                )
                if at >= 0:
                    joined = composed[at]
            if joined != 0:
                points[starter] = joined
                continue
        if here == 0:
            starter = kept
            last_class = -1
        else:
            last_class = here
        points[kept] = point
        classes[kept] = UInt8(here)
        kept += 1
    while len(points) > kept:
        _ = points.pop()
        _ = classes.pop()


def _put(mut into: List[UInt8], point: UInt32):
    """Writes one code point out as UTF-8.

    Here rather than through a `String` per character because this is the only
    thing standing between a decomposed element and the builder, and building a
    heap string for each of the four or five code points an accented letter
    turns into is the kind of cost that only shows up once the kernel is in a
    loop over millions of rows.

    Args:
        into: The byte scratch, appended.
        point: The character, which came out of a table and is known to be a
            code point.
    """
    if point < 0x80:
        into.append(point.cast[DType.uint8]())
    elif point < 0x800:
        into.append((0xC0 | (point >> 6)).cast[DType.uint8]())
        into.append((0x80 | (point & 0x3F)).cast[DType.uint8]())
    elif point < 0x10000:
        into.append((0xE0 | (point >> 12)).cast[DType.uint8]())
        into.append((0x80 | ((point >> 6) & 0x3F)).cast[DType.uint8]())
        into.append((0x80 | (point & 0x3F)).cast[DType.uint8]())
    else:
        into.append((0xF0 | (point >> 18)).cast[DType.uint8]())
        into.append((0x80 | ((point >> 12) & 0x3F)).cast[DType.uint8]())
        into.append((0x80 | ((point >> 6) & 0x3F)).cast[DType.uint8]())
        into.append((0x80 | (point & 0x3F)).cast[DType.uint8]())


def _is_ascii(bytes: Span[UInt8, _]) -> Bool:
    """Whether an element is entirely ASCII, and so already normalized.

    The claim behind the fast path, checked in the generator as well as argued
    here: no ASCII character has a canonical or a compatibility decomposition,
    none has a combining class other than zero, and no pair of ASCII characters
    composes into anything. An ASCII element is therefore its own answer in all
    four forms.

    Args:
        bytes: The element.

    Returns:
        True if no byte is 0x80 or above.
    """
    for k in range(len(bytes)):
        if bytes[k] >= 0x80:
            return False
    return True


def _well_formed(bytes: Span[UInt8, _]) -> Bool:
    """Whether a run of bytes is valid UTF-8.

    An element that is not gets copied through untouched. The decode below walks
    by lead byte and would run off the end of a truncated sequence into the next
    element, because the payload of a text column is one buffer with the
    elements laid end to end.

    Args:
        bytes: The element.

    Returns:
        True if it can be read as text.
    """
    try:
        _ = StringSlice(from_utf8=bytes)
        return True
    except:
        return False


def _copied(table: Span[UInt32, _]) -> List[UInt32]:
    """One of the generated tables copied into a list.

    The canonical and the compatibility tables are different lengths, and an
    `InlineArray` carries its length in its type, so the two are different types
    and cannot be picked between with an expression. Copying both candidates
    down to a list makes the choice an ordinary one. It costs a single pass over
    eighty odd kilobytes per call to the kernel, which is per column rather than
    per element, and it is what keeps the loop below from being written twice.

    Args:
        table: A materialized table.

    Returns:
        The same numbers in a list.
    """
    var out = List[UInt32](capacity=len(table))
    for k in range(len(table)):
        out.append(table[k])
    return out^


def text_normalize(
    a: StringArray, full: Bool, compose: Bool
) raises -> StringArray:
    """Rewrites every element in one of the four normalization forms.

    The two arguments are the two independent choices rather than a form name,
    because that is what the four forms are: `full` picks the compatibility
    relation over the canonical one, which is the K, and `compose` picks putting
    things back together over leaving them apart, which is the C. NFD is False
    and False, NFC is False and True, NFKD is True and False, NFKC is True and
    True. Reading the name and turning it into these two happens once in the
    Python layer, where the error for a name that is not one of the four also
    lives.

    A null element stays null and an empty element stays empty, which is the one
    thing about this kernel nobody has to look up: normalization never invents a
    character and never removes the last one.

    Args:
        a: The column.
        full: Whether to use the compatibility decompositions as well as the
            canonical ones.
        compose: Whether to finish by composing.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: If the builder cannot allocate.
    """
    var n = len(a)
    var built = StringBuilder(capacity=n)
    var keys: List[UInt32]
    var starts: List[UInt32]
    var values: List[UInt32]
    if full:
        var table_keys = materialize[NFKD_KEYS]()
        var table_starts = materialize[NFKD_STARTS]()
        var table_values = materialize[NFKD_VALUES]()
        keys = _copied(Span(table_keys))
        starts = _copied(Span(table_starts))
        values = _copied(Span(table_values))
    else:
        var table_keys = materialize[NFD_KEYS]()
        var table_starts = materialize[NFD_STARTS]()
        var table_values = materialize[NFD_VALUES]()
        keys = _copied(Span(table_keys))
        starts = _copied(Span(table_starts))
        values = _copied(Span(table_values))
    var ccc_keys = materialize[CCC_KEYS]()
    var ccc_values = materialize[CCC_VALUES]()
    var pairs = materialize[COMPOSE_PAIRS]()
    var composed = materialize[COMPOSE_VALUES]()
    var points = List[UInt32]()
    var classes = List[UInt8]()
    var scratch = List[UInt8]()
    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        if _is_ascii(bytes) or not _well_formed(bytes):
            built.append(bytes)
            continue
        points.clear()
        classes.clear()
        scratch.clear()
        var text = StringSlice(unsafe_from_utf8=bytes)
        for one in text.codepoints():
            var point = one.to_u32()
            if _decompose_hangul(point, points):
                continue
            var at = _keyed_at(Span(keys), point)
            if at < 0:
                points.append(point)
                continue
            for k in range(Int(starts[at]), Int(starts[at + 1])):
                points.append(values[k])
        for k in range(len(points)):
            classes.append(
                _class_of(Span(ccc_keys), Span(ccc_values), points[k])
            )
        _order(points, classes)
        if compose:
            _compose(points, classes, Span(pairs), Span(composed))
        for k in range(len(points)):
            _put(scratch, points[k])
        built.append(Span(scratch))
    return built^.finish()
