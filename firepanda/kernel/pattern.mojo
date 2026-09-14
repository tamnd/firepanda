"""Substring search over a text column: contains, starts, ends, equals, count,
replace.

These are what a `LIKE` pattern turns into once the wildcards are read. `LIKE
'%green%'` is a contains, `LIKE 'forest%'` is a starts with, `LIKE '%BRASS'` is
an ends with, and `LIKE '%special%requests%'` is a contains followed by another
contains in what is left. Those four cover every pattern TPC-H uses and most of
what a filter on a text column is asked for outside a benchmark, which is why
they are four named kernels rather than a pattern compiler.

Anything else is the general matcher at the end of the file, which is a sixth
search and not a replacement for the five. That order was deliberate. A matcher
walks the pattern and the row together and it cannot do any of the things that
make a prefix cheap, so building it first would have made every pattern anybody
actually writes pay for the ones nobody does. Built second it costs the five
nothing: a pattern that reads as one of them still gets its kernel, and the
matcher only ever sees what used to be refused.

Two of the four cost almost nothing and one of them is the whole file. Starts
with and ends with are a length test and one run of bytes compared at a known
offset, so they are as cheap as an equality against a constant. Contains has to
look at every position, and the position that matches is different for every row,
so this is where the work is.

The search is a two ended filter with a vectorized skip. The needle's first and
last bytes are each broadcast across a register, two blocks of the haystack are
loaded a needle apart, and a candidate survives only where both agree. A block
with no survivor moves the cursor by the whole block, and a block with one falls
back to comparing the middle bytes of that candidate. Filtering on one end lets
through every position holding the needle's first byte, which on ordinary text is
one in twenty odd; filtering on both lets through one in five hundred, and the
second load is free next to what it saves. On the shapes here, needles of five to
twenty bytes against strings of ten to eighty, this beats a two way or a Boyer
Moore search because those spend their setup building tables that a short needle
never earns back.

Comparison against a null is null, the same as it is for the ordering kernels,
and it is handled the same way: the loop writes whatever falls out and the repair
at the end of each morsel clears the rows where the input was missing.

### The four the str accessor added

`str.contains`, `str.match`, `str.fullmatch` and `str.count` are the same four
questions this file already answered for `LIKE`, asked by a different caller.
Contains is contains, match is starts with, fullmatch is equality, and only the
count needed a kernel, so three of the four cost nothing and the fourth is the
search in a loop with the cursor moved past each hit.

pandas reads all four arguments as regular expressions, and this file has no
regular expression engine and does not want one yet. What makes the four useful
anyway is that a pattern with no metacharacter in it means the same thing to a
regular expression engine as it does to a byte search, so the Python layer reads
the pattern, and a literal one comes here while the rest are refused by name.
That is a smaller promise than pandas makes and it is a true one, which is the
trade document 07 asks for.

### The fifth, which needs no promise at all

`str.replace` came in after the four and is the one name of the group where
pandas asks for a literal by default: its `regex` argument defaults to False in
pandas 3, so the ordinary call is a byte search and a rewrite and there is
nothing to refuse. It is also the first kernel here whose answer is text, which
means it is the first that cannot write into a column allocated up front,
because how long a row comes out is not known until the search has run.
"""

from std.collections.span import Span

from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.exec import parallel_morsels

from .mask import repair_range


comptime SCAN_WIDTH = 16
"""Candidate positions tested at once.

Sixteen and not thirty two, and the reason is the row length rather than the
register width. A block starting at `i` reads the needle's last byte for every
candidate in it, so it reads through `i + SCAN_WIDTH + m - 2`, and a row shorter
than `SCAN_WIDTH + m - 1` has no room for a single block and falls to the byte
loop. At thirty two that threshold is thirty six bytes for a five byte needle,
which is longer than most of the columns anybody searches: TPC-H's part type is
about twenty five bytes and its name about forty. The first version of this file
used thirty two and measured twelve nanoseconds a row on a thirty two byte
column, three times DuckDB, because it never entered the block loop once.

Sixteen puts the threshold at twenty bytes, which those columns clear, and it is
one SSE register.
"""

comptime WORD = 8
"""Bytes compared at once when verifying a candidate.

The same number `_bytes_equal` uses in `strings.mojo` and for the same reason.
"""


struct MatchKind(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which of the searches a `LIKE` pattern turned out to be.

    Equality is in here with the other five although it is not a search at all,
    because what reads a pattern hands back one thing and the caller decides
    what to do with it. A pattern with no wildcard in it is an equality against
    a constant, and saying so here is one place rather than one per caller.

    `GENERAL` is the one that answers everything, so it is last on purpose. A
    pattern is only read as that one once the shapes above it have been tried
    and none of them fit.
    """

    var code: UInt8
    """The search, as a small integer."""

    comptime EQUALS = Self(0)
    """No wildcard, so the whole element has to be the pattern."""

    comptime CONTAINS = Self(1)
    """A run wrapped in wildcards, so `%green%`."""

    comptime STARTS_WITH = Self(2)
    """A run with a wildcard after it, so `forest%`."""

    comptime ENDS_WITH = Self(3)
    """A run with a wildcard before it, so `%BRASS`."""

    comptime IN_ORDER = Self(4)
    """Two runs each wrapped in wildcards, so `%special%requests%`."""

    comptime GENERAL = Self(5)
    """Any other pattern, so `a_c` or `a%e`, walked against the row."""

    def __init__(out self, code: UInt8):
        """Constructs a search from its code.

        Args:
            code: The search.
        """
        self.code = code

    def __eq__(self, other: Self) -> Bool:
        """Compares two searches.

        Args:
            other: The search to compare against.

        Returns:
            True if they are the same one.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two searches.

        Args:
            other: The search to compare against.

        Returns:
            True if they are different ones.
        """
        return self.code != other.code

    def write_to(self, mut writer: Some[Writer]):
        """Writes the search as the name of the kernel that answers it.

        Args:
            writer: Where the text goes.
        """
        if self == Self.EQUALS:
            writer.write("equals")
        elif self == Self.CONTAINS:
            writer.write("contains")
        elif self == Self.STARTS_WITH:
            writer.write("starts with")
        elif self == Self.ENDS_WITH:
            writer.write("ends with")
        elif self == Self.IN_ORDER:
            writer.write("contains in order")
        else:
            writer.write("matches")


struct Pattern(ImplicitlyCopyable, Movable):
    """A `LIKE` pattern with its wildcards read off it.

    The runs are what is left once the wildcards are gone, so the pattern is not
    kept. `second` is empty for every search but the one that reads two runs,
    and an empty run is a real answer for the others: `%` reads as an ends with
    against nothing, which every element ends with and no null does, and that is
    what `LIKE '%'` means.

    The general search is the exception, because it has no runs to take out. Its
    `first` is the pattern exactly as the query wrote it, wildcards and all,
    since the matcher reads them itself.
    """

    var kind: MatchKind
    """Which search answers it."""

    var first: String
    """The run of bytes to look for, the whole element for an equality, or the
    pattern itself for the general search."""

    var second: String
    """The run that has to follow the first one, and empty otherwise."""

    def __init__(
        out self, kind: MatchKind, var first: String, var second: String
    ):
        """Constructs a read pattern.

        Args:
            kind: Which search answers it.
            first: The first run. Consumed.
            second: The second run, empty unless the search reads two. Consumed.
        """
        self.kind = kind
        self.first = first^
        self.second = second^


def read_pattern(pattern: StringSlice) raises -> Pattern:
    """Reads a `LIKE` pattern into the search that answers it.

    The five shapes above `GENERAL` are the ones with a kernel of their own, and
    they are tried first because each of them is much cheaper than walking the
    pattern. A pattern that is none of them reads as the general search, which
    answers every pattern there is and is the only one that has to look at a
    wildcard while the rows go past.

    An underscore anywhere sends the pattern straight to the general search
    without the split below being tried at all. That is not an optimisation, it
    is the correctness of the four: they are found by counting the runs between
    the `%` signs, and an underscore sitting inside one of those runs would be
    compared as an ordinary byte. `a_c` would read as an equality and `%a_b%` as
    a substring, and both would quietly answer the wrong rows.

    There is no escape character. DuckDB has none by default either, so a
    backslash in a pattern is an ordinary byte to both, and `ESCAPE` is refused
    by the SQL front end before anything gets here.

    Args:
        pattern: The pattern as the query wrote it, wildcards and all.

    Returns:
        The search and the runs of bytes it reads.

    Raises:
        Error: Nothing here raises. The signature keeps it because every caller
            is already in a context that can, and because reading a pattern is
            where an escape character would be refused if one is ever added.
    """
    var text = String(pattern)
    var bytes = text.as_bytes()
    var general = False
    for i in range(len(bytes)):
        if bytes[i] == UInt8(ord("_")):
            general = True
            break

    if not general:
        var runs = List[String]()
        for piece in text.split("%"):
            runs.append(String(piece))

        if len(runs) == 1:
            return Pattern(MatchKind.EQUALS, runs[0].copy(), String(""))
        if len(runs) == 2:
            if runs[0] == "":
                return Pattern(MatchKind.ENDS_WITH, runs[1].copy(), String(""))
            if runs[1] == "":
                return Pattern(
                    MatchKind.STARTS_WITH, runs[0].copy(), String("")
                )
        elif len(runs) == 3:
            if runs[0] == "" and runs[2] == "":
                return Pattern(MatchKind.CONTAINS, runs[1].copy(), String(""))
        elif len(runs) == 4:
            if (
                runs[0] == ""
                and runs[3] == ""
                and runs[1] != ""
                and runs[2] != ""
            ):
                return Pattern(
                    MatchKind.IN_ORDER, runs[1].copy(), runs[2].copy()
                )

    return Pattern(MatchKind.GENERAL, String(pattern), String(""))


def _match_at(
    hay: Span[UInt8, _], needle: Span[UInt8, _], at: Int, count: Int
) -> Bool:
    """Whether the needle sits at a given offset in the haystack.

    The caller has already checked the first and last bytes and that the needle
    fits, so this is the middle only.

    Args:
        hay: The bytes being searched.
        needle: The bytes being looked for.
        at: Where in `hay` the candidate starts.
        count: How long the needle is.

    Returns:
        True if every byte of the needle matches from `at`.
    """
    var left = hay.unsafe_ptr().unsafe_offset(at)
    var right = needle.unsafe_ptr()
    var i = 1
    while i + WORD <= count - 1:
        var chunk = left.unsafe_offset(i).unsafe_load[width=WORD]()
        var other = right.unsafe_offset(i).unsafe_load[width=WORD]()
        if chunk.ne(other).reduce_or():
            return False
        i += WORD
    while i < count - 1:
        if left.unsafe_offset(i)[] != right.unsafe_offset(i)[]:
            return False
        i += 1
    return True


def find_bytes(hay: Span[UInt8, _], needle: Span[UInt8, _], from_: Int) -> Int:
    """Finds the first occurrence of the needle at or after an offset.

    Args:
        hay: The bytes being searched.
        needle: The bytes being looked for.
        from_: The first position that may be returned.

    Returns:
        The offset of the match, or -1 if there is none. An empty needle matches
        at `from_`, which is what every other language's `find` says and what
        makes `LIKE '%%'` true rather than false.
    """
    var n = len(hay)
    var m = len(needle)
    if m == 0:
        return from_ if from_ <= n else -1
    # The last position a match can start at. Everything below is relative to
    # this rather than to the length, because a candidate that starts inside the
    # last `m - 1` bytes cannot fit and must never be loaded.
    var limit = n - m
    if limit < from_:
        return -1

    var base = hay.unsafe_ptr()
    var first = SIMD[DType.uint8, SCAN_WIDTH](needle[0])
    var last = SIMD[DType.uint8, SCAN_WIDTH](needle[m - 1])

    if limit + 1 - from_ >= SCAN_WIDTH:
        # Sixteen candidates at a time, each one filtered on both ends before
        # anything reads the middle. Two loads, one at the candidate and one at
        # where its last byte would be, and a position survives only if both
        # agree. Checking one end lets through every position holding the
        # needle's first byte, which on ordinary text is one in twenty odd;
        # checking both lets through one in five hundred.
        #
        # The last block overlaps the one before it rather than giving up and
        # handing the remainder to a byte loop. Positions get tested twice that
        # way, which costs one block and cannot change the answer: a block
        # returns the first match inside itself, and a position that was already
        # tested was already found not to match.
        var stop = limit + 1 - SCAN_WIDTH
        var i = from_
        while True:
            var front = base.unsafe_offset(i).unsafe_load[width=SCAN_WIDTH]()
            var back = base.unsafe_offset(i + m - 1).unsafe_load[
                width=SCAN_WIDTH
            ]()
            var hits = front.eq(first) & back.eq(last)
            if hits.reduce_or():
                # Unrolled, because `k` indexes a SIMD lane. A runtime index
                # into a register is a store and a reload on most targets, and
                # a compile time one is a single extract. That is worth more
                # here than it looks: this loop runs on every row that matches,
                # and with it rolled the row that finds its needle cost twice
                # what the row that has none did.
                comptime for k in range(SCAN_WIDTH):
                    if hits[k] and _match_at(hay, needle, i + k, m):
                        return i + k
            if i >= stop:
                return -1
            i = min(i + SCAN_WIDTH, stop)

    # A row too short to hold one block. The bounds are the same as above and
    # the filter is the same filter, one position at a time.
    var head = needle[0]
    var tail = needle[m - 1]
    var at = from_
    while at <= limit:
        if (
            base.unsafe_offset(at)[] == head
            and base.unsafe_offset(at + m - 1)[] == tail
            and _match_at(hay, needle, at, m)
        ):
            return at
        at += 1
    return -1


def rfind_bytes(
    hay: Span[UInt8, _], needle: Span[UInt8, _], from_: Int, until: Int
) -> Int:
    """Finds the last occurrence of the needle inside a byte range.

    The forward search above filters sixteen candidates at a time and this one
    walks backwards a byte at a time. That is a deliberate gap rather than an
    oversight: the same two ended filter mirrors onto a backwards scan without
    changing an idea in it, and the only caller of this today is `str.rfind` on
    a column of short strings, so writing the fast version now would be adding
    a page of index arithmetic in exchange for a number nobody has asked for.
    It should be mirrored the day something measures it.

    Args:
        hay: The bytes being searched.
        needle: The bytes being looked for.
        from_: The first position that may be returned.
        until: The position the match must end at or before.

    Returns:
        The offset of the match, or -1 if there is none. An empty needle matches
        at `until`, which is where Python's `rfind` puts it.

    """
    var m = len(needle)
    var limit = min(until, len(hay)) - m
    if limit < from_:
        return -1
    if m == 0:
        return limit
    var at = limit
    while at >= from_:
        if _starts_at(hay, needle, at):
            return at
        at -= 1
    return -1


def _starts_at(hay: Span[UInt8, _], needle: Span[UInt8, _], at: Int) -> Bool:
    """Whether the needle sits at a given offset, first and last byte included.

    Args:
        hay: The bytes being searched.
        needle: The bytes being looked for.
        at: Where in `hay` to look.

    Returns:
        True if every byte matches. The caller guarantees the needle fits.
    """
    var m = len(needle)
    if m == 0:
        return True
    if hay[at] != needle[0]:
        return False
    if hay[at + m - 1] != needle[m - 1]:
        return False
    return _match_at(hay, needle, at, m)


def text_contains(
    a: StringArray, needle: Span[UInt8, _]
) raises -> Array[DType.bool]:
    """Whether each element contains a run of bytes.

    Args:
        a: The column.
        needle: The bytes to look for. Borrowed for the length of the call and
            not stored.

    Returns:
        A bool column, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    # Every row is written below, so the zeroing allocation is a wasted pass.
    var out = Array[DType.bool](overwritten=n)
    var validity = Bitmap(copy=a.validity)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            var found = find_bytes(a.unsafe_bytes(i), needle, 0) >= 0
            dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](found))
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def text_contains_in_order(
    a: StringArray, first: Span[UInt8, _], second: Span[UInt8, _]
) raises -> Array[DType.bool]:
    """Whether each element contains two runs of bytes, in order and disjoint.

    This is `LIKE '%a%b%'`. The second run has to start after the first one ends,
    which is what the SQL pattern means and is not what two independent contains
    calls would say: `'abc'` matches `'%bc%a%'` under two contains and does not
    match it under `LIKE`.

    Args:
        a: The column.
        first: The bytes that must come first.
        second: The bytes that must follow them.

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
            var at = find_bytes(bytes, first, 0)
            var found = False
            if at >= 0:
                found = find_bytes(bytes, second, at + len(first)) >= 0
            dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](found))
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def text_starts_with(
    a: StringArray, prefix: Span[UInt8, _]
) raises -> Array[DType.bool]:
    """Whether each element begins with a run of bytes.

    Args:
        a: The column.
        prefix: The bytes to look for at the front.

    Returns:
        A bool column, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var out = Array[DType.bool](overwritten=n)
    var validity = Bitmap(copy=a.validity)
    var m = len(prefix)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            var bytes = a.unsafe_bytes(i)
            var found = len(bytes) >= m and _starts_at(bytes, prefix, 0)
            dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](found))
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def text_equals(
    a: StringArray, other: Span[UInt8, _]
) raises -> Array[DType.bool]:
    """Whether each element is exactly a run of bytes and nothing more.

    This is what `str.fullmatch` is once the pattern is known to be literal, and
    it is the cheapest of the five: the lengths have to agree before a byte is
    read, so a column of forty byte rows against a three byte pattern answers
    without touching the data at all.

    Args:
        a: The column.
        other: The bytes the whole element has to be.

    Returns:
        A bool column, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var out = Array[DType.bool](overwritten=n)
    var validity = Bitmap(copy=a.validity)
    var m = len(other)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            var bytes = a.unsafe_bytes(i)
            var same = len(bytes) == m and _starts_at(bytes, other, 0)
            dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](same))
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def text_count(
    a: StringArray, needle: Span[UInt8, _]
) raises -> Array[DType.int64]:
    """How many times a run of bytes appears in each element, without overlap.

    Without overlap means the cursor moves by the whole needle after a hit, so
    `aa` appears twice in `aaaa` and not three times. That is what a regular
    expression engine answers for a literal pattern and it is what pandas
    answers, and it is the only rule here that a caller is likely to have an
    opinion about.

    An empty needle is counted in bytes and not in characters, which is the one
    place in this file where the two differ and is worth saying out loud. Arrow
    counts a match at every byte offset and one past the end, so an empty needle
    against a five character word holding one accented letter is seven and not
    six. pandas answers Arrow here, this library answers pandas, and Python's
    own `re` module answers six. Document 66 has the measurement.

    Args:
        a: The column.
        needle: The bytes to look for.

    Returns:
        An int64 column, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var out = Array[DType.int64](overwritten=n)
    var validity = Bitmap(copy=a.validity)
    var m = len(needle)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            var bytes = a.unsafe_bytes(i)
            var seen = 0
            if m == 0:
                seen = len(bytes) + 1
            else:
                var from_ = 0
                while from_ + m <= len(bytes):
                    var at = find_bytes(bytes, needle, from_)
                    if at < 0:
                        break
                    seen += 1
                    from_ = at + m
            dst.unsafe_offset(i).unsafe_write(Int64(seen))
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def text_replace(
    a: StringArray, needle: Span[UInt8, _], repl: Span[UInt8, _], limit: Int
) raises -> StringArray:
    """Writes every element out with a run of bytes swapped for another.

    This is the first kernel in the file whose answer is text rather than a
    number or a flag, and it is the reason it cannot be one: how long a row comes
    out is not known until the search has run, so the rows are built one at a
    time into a builder rather than written into a column allocated up front.
    Nothing else here needed that and nothing else here pays for it.

    Matches do not overlap, which is the rule `text_count` explains and is the
    same rule for the same reason: the cursor moves by the whole needle after a
    hit, so replacing `aa` in `aaaa` swaps twice and not three times.

    An empty needle is the one place this counts characters. Python inserts the
    replacement before every character and once at the end, so replacing nothing
    in `hello` with a dash gives six dashes, and pandas answers Python here
    rather than Arrow because Arrow does not terminate on an empty pattern at
    all. That is `str.count` of an empty pattern counted in bytes and
    `str.replace` of an empty pattern counted in characters, inside one accessor,
    and document 66 has both measurements and the reason they differ.

    Args:
        a: The column.
        needle: The bytes to look for.
        repl: The bytes to put in their place.
        limit: How many matches in each row to replace. Negative means all of
            them, and zero means none, which copies the row unchanged. pandas
            spells this `n` and gives it the same three readings.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: If the builder cannot allocate.
    """
    var n = len(a)
    var built = StringBuilder(capacity=n)
    var m = len(needle)
    var scratch = List[UInt8]()

    for i in range(n):
        if not a.is_valid(i):
            built.append_null()
            continue
        var bytes = a.unsafe_bytes(i)
        if limit == 0:
            built.append(bytes)
            continue

        scratch.clear()
        var left = limit
        if m == 0:
            # A character boundary is any byte that is not a UTF-8 continuation
            # byte, which is the test `starts_character` in chars.mojo makes. It
            # is written out rather than imported because chars.mojo reads the
            # search out of this file and the two cannot read each other.
            for k in range(len(bytes)):
                if (bytes[k] & 0xC0) != 0x80 and left != 0:
                    scratch.extend(repl)
                    if left > 0:
                        left -= 1
                scratch.append(bytes[k])
            if left != 0:
                scratch.extend(repl)
        else:
            var from_ = 0
            while left != 0 and from_ + m <= len(bytes):
                var at = find_bytes(bytes, needle, from_)
                if at < 0:
                    break
                scratch.extend(bytes[from_:at])
                scratch.extend(repl)
                from_ = at + m
                if left > 0:
                    left -= 1
            scratch.extend(bytes[from_ : len(bytes)])
        built.append(Span(scratch))

    return built^.finish()


def _character_width(bytes: Span[UInt8, _], at: Int) -> Int:
    """How many bytes the character starting at an offset takes up.

    The length is counted rather than read off the lead byte, because a run of
    bytes that is not well formed UTF-8 still has to make progress. Anything
    whose top two bits are not `10` starts something, so the width is one plus
    however many continuation bytes follow it, and a stray continuation byte on
    its own comes out as one.

    Written here rather than imported from `chars.mojo`, which has the same
    test, because that file reads the search out of this one and the two cannot
    read each other.

    Args:
        bytes: The bytes.
        at: Where the character starts. The caller guarantees it is inside.

    Returns:
        The width in bytes, always at least one.
    """
    var end = at + 1
    while end < len(bytes) and (bytes[end] & 0xC0) == 0x80:
        end += 1
    return end - at


def matches_pattern(bytes: Span[UInt8, _], pattern: Span[UInt8, _]) -> Bool:
    """Whether a run of bytes matches a `LIKE` pattern, wildcards and all.

    The pattern language is two wildcards and everything else. `%` stands for
    any run of characters including none, `_` stands for exactly one character,
    and every other byte stands for itself.

    Characters and not bytes, which is the one thing about `_` that is easy to
    get wrong and that DuckDB is clear about: `'héllo' LIKE 'h_llo'` is true
    there and `'héllo' LIKE 'h__llo'` is false, so an underscore steps over the
    two bytes of the accented letter as one thing. `%` has to move a character
    at a time for the same reason, since a `%` that stopped halfway into a
    letter would let the pattern after it compare against the tail of one.

    The walk carries one remembered `%` and no stack. Both pointers move
    forward, and the only way to go back is to the last `%` seen, whose match
    then grows by one character. That is enough because a pattern has no
    alternation in it, so an earlier `%` never needs to be reconsidered once a
    later one has been reached: whatever the earlier one gave up would have to
    be taken by the later one anyway. It costs the length of the row times the
    length of the pattern in the worst case and nothing like that in practice,
    and it means there is no depth to limit and no recursion in a plan.

    Literal bytes are compared as bytes even though the wildcards count
    characters, which is safe and not a shortcut: two characters are equal
    exactly when their bytes are, and a literal run in a well formed pattern is
    whole characters, so the cursor is only ever left on a character boundary
    when a wildcard is reached.

    Args:
        bytes: The element being matched.
        pattern: The pattern as the query wrote it.

    Returns:
        True if the whole element matches the whole pattern.
    """
    var n = len(bytes)
    var m = len(pattern)
    comptime PERCENT = UInt8(ord("%"))
    comptime UNDERSCORE = UInt8(ord("_"))

    var s = 0
    var p = 0
    # Where the last `%` sits in the pattern, and how much of the row it has
    # been given so far. Minus one for the first means none has been seen, which
    # is what makes a mismatch final rather than something to go back from.
    var star_p = -1
    var star_s = 0

    while s < n:
        if p < m and pattern[p] == UNDERSCORE:
            s += _character_width(bytes, s)
            p += 1
        elif p < m and pattern[p] == PERCENT:
            star_p = p
            star_s = s
            p += 1
        elif p < m and pattern[p] == bytes[s]:
            p += 1
            s += 1
        elif star_p >= 0:
            # The last `%` takes one more character and the pattern after it
            # starts again from there. `star_s` is on a boundary, because it was
            # set where a wildcard was reached and only ever moves by a whole
            # character.
            star_s += _character_width(bytes, star_s)
            s = star_s
            p = star_p + 1
        else:
            return False

    # The row is used up. What is left of the pattern can only match nothing,
    # which `%` does and neither `_` nor a literal byte does.
    while p < m and pattern[p] == PERCENT:
        p += 1
    return p == m


def text_like(
    a: StringArray, pattern: Span[UInt8, _]
) raises -> Array[DType.bool]:
    """Whether each element matches a `LIKE` pattern with wildcards in it.

    The sixth search, and the one nothing reaches unless the five above it were
    tried first. `read_pattern` does that trying, so a prefix never arrives here
    and never pays for the walk.

    Args:
        a: The column.
        pattern: The pattern as the query wrote it, wildcards and all. Borrowed
            for the length of the call and not stored.

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
            var hit = matches_pattern(a.unsafe_bytes(i), pattern)
            dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](hit))
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def text_ends_with(
    a: StringArray, suffix: Span[UInt8, _]
) raises -> Array[DType.bool]:
    """Whether each element ends with a run of bytes.

    Args:
        a: The column.
        suffix: The bytes to look for at the back.

    Returns:
        A bool column, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var out = Array[DType.bool](overwritten=n)
    var validity = Bitmap(copy=a.validity)
    var m = len(suffix)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            var bytes = a.unsafe_bytes(i)
            var room = len(bytes) - m
            var found = room >= 0 and _starts_at(bytes, suffix, room)
            dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](found))
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^
