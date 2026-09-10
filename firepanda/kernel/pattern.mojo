"""Substring search over a text column: contains, starts with, ends with.

These are what a `LIKE` pattern turns into once the wildcards are read. `LIKE
'%green%'` is a contains, `LIKE 'forest%'` is a starts with, `LIKE '%BRASS'` is
an ends with, and `LIKE '%special%requests%'` is a contains followed by another
contains in what is left. Those four cover every pattern TPC-H uses and most of
what a filter on a text column is asked for outside a benchmark, which is why
they are four named kernels rather than a pattern compiler. A general matcher
with a wildcard alphabet is a different piece of work and it should not be built
first: it would be slower on all four of these and would answer questions nobody
has asked yet.

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
"""

from std.collections.span import Span

from firepanda.array.array import Array
from firepanda.array.strings import StringArray
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
