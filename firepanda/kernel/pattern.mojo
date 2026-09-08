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

The search is the classic first and last byte filter with a vectorized skip. The
needle's first byte is broadcast across a register, a block of the haystack is
compared against it, and a block with no hit moves the cursor by the whole block.
A block with a hit falls back to checking each candidate position in it, and a
candidate is only checked in full when its last byte matches too, which throws
out almost everything the first byte let through. On the shapes here, needles of
five to twenty bytes against strings of ten to eighty, that beats a two way or a
Boyer Moore search because those spend their setup building tables that a short
needle never earns back.

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


comptime SCAN_WIDTH = 32
"""Bytes of haystack compared against the needle's first byte at once.

Wide enough that a string of forty bytes is two blocks rather than five, and no
wider, because the tail below the block loop is walked a byte at a time and a
wider block leaves a longer tail. Thirty two is one AVX2 register and two SSE
ones, and on a machine with neither the compiler splits it into whatever it has.
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
        if hay[at + i] != needle[i]:
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
    var last = needle[m - 1]
    var i = from_

    while i + SCAN_WIDTH <= limit + 1:
        var block = base.unsafe_offset(i).unsafe_load[width=SCAN_WIDTH]()
        var hits = block.eq(first)
        if hits.reduce_or():
            for k in range(SCAN_WIDTH):
                if (
                    hits[k]
                    and hay[i + k + m - 1] == last
                    and _match_at(hay, needle, i + k, m)
                ):
                    return i + k
        i += SCAN_WIDTH

    while i <= limit:
        if (
            hay[i] == needle[0]
            and hay[i + m - 1] == last
            and _match_at(hay, needle, i, m)
        ):
            return i
        i += 1
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
        var dst = out.unsafe_ptr()
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
        var dst = out.unsafe_ptr()
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
        var dst = out.unsafe_ptr()
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
        var dst = out.unsafe_ptr()
        for i in range(start, stop):
            var bytes = a.unsafe_bytes(i)
            var room = len(bytes) - m
            var found = room >= 0 and _starts_at(bytes, suffix, room)
            dst.unsafe_offset(i).unsafe_write(Scalar[DType.bool](found))
        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^
