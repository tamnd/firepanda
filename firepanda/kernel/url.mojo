"""The hostname out of a URL, by hand, because the regex engine is not here yet.

ClickBench q28 groups by `REGEXP_REPLACE(Referer,
'^https?://(?:www\\.)?([^/]+)/.*$', '\\1')`, which is a hostname extractor
written as a regular expression. It is the only query in the suite whose group
by key is a computed expression rather than a column, and the expression
happens to need the one thing this library does not have: an anchored
alternation with an optional group, a negated class and a back reference into a
capture. `pattern.mojo` says in its own first paragraph that four named kernels
are not a pattern compiler, and it is right.

So this is the extractor and not the engine. The point of writing it is that
everything else q28 needs, the group by on a computed key, the average length,
the count, the smallest string per group, the having and the ordered limit, can
be built and tested now, and the day RE2 lands q28 becomes one substitution
rather than a new query. What this file must not do is be approximately the
regex, since then the substitution would change answers. So the rule below is
the regex read byte by byte, including the two parts of it that are easy to get
wrong.

The first is the optional `www.`. A regex prefers to take an optional group and
backtracks only when what follows fails, so on `http://www./x` the preferred
reading consumes `www.` and then `[^/]+` has nothing left to match before the
slash. The engine backs up and reads the host as `www.` instead. This does the
same thing in the same order rather than stripping a `www.` unconditionally.

The second is that `.` does not match a newline in RE2 by default and `$` is
the end of the text rather than the end of a line. So a URL whose path holds a
newline does not match the pattern at all, and `regexp_replace` hands back the
string it was given. A negated class is not `.`, though, so a newline inside
the hostname is fine. Both of those are asserted in the tests, because they are
exactly the kind of thing a hand written extractor gets wrong and nobody
notices until the engine arrives and the numbers move.

No match means the input, unchanged, which is what `regexp_replace` does with a
subject it cannot match. That matters more here than it sounds: `Referer` is
empty on most rows of the hits table, q28 filters those out, and everything
left that is not a URL comes through as itself and becomes its own group.

The build is the two pass shape `substr.mojo` uses, for the same reason: an
output element's length is known before any bytes move, so every morsel can be
told where its payload starts and then write views and bytes with nothing
shared. The difference is that the sizing pass here has to follow the pointer
into the payload to find the slash, where a substring can read the length out
of the view. That is one pass over the bytes to size and one to copy, and the
alternative is a serial builder.

So the cut is worked out twice on the rows that need the payload, once to size
and once to copy. Keeping it instead would mean a scratch entry per row, and at
a hundred million rows that is eight hundred megabytes to save one scan of a
column that is already being read twice. The rows that fit in their own view
skip the first pass entirely, since an element of twelve bytes or fewer cannot
produce a longer one.
"""

from std.collections.span import Span
from std.memory import unsafe_memcpy

from firepanda.array.strings import StringArray
from firepanda.array.strview import (
    INLINE_CAPACITY,
    StringView,
    VIEW_SIZE,
    make_inline_at,
    make_long_at,
)
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.array.array import Array
from firepanda.exec import parallel_morsels
from firepanda.exec.morsel import MORSEL_ROWS

from .substr import _Cut


comptime SLASH = UInt8(ord("/"))
"""The byte the hostname stops at, which is also the one it may not contain."""

comptime NEWLINE = UInt8(ord("\n"))
"""The byte `.` refuses to match, so a path holding one has no host."""


def _scheme_end(bytes: Span[UInt8, _]) -> Int:
    """Where the scheme and its separator stop, or -1 if there is no scheme.

    Args:
        bytes: One element.

    Returns:
        7 for `http://`, 8 for `https://`, and -1 for anything else, which
        includes an uppercase scheme, since the pattern is case sensitive.
    """
    var n = len(bytes)
    if n < 7:
        return -1
    if (
        bytes[0] != UInt8(ord("h"))
        or bytes[1] != UInt8(ord("t"))
        or bytes[2] != UInt8(ord("t"))
        or bytes[3] != UInt8(ord("p"))
    ):
        return -1
    if bytes[4] == UInt8(ord(":")) and bytes[5] == SLASH and bytes[6] == SLASH:
        return 7
    if n < 8:
        return -1
    if (
        bytes[4] == UInt8(ord("s"))
        and bytes[5] == UInt8(ord(":"))
        and bytes[6] == SLASH
        and bytes[7] == SLASH
    ):
        return 8
    return -1


def _host_from(bytes: Span[UInt8, _], start: Int) -> Int:
    """Where the host starting at an offset ends, or -1 if it does not.

    The host is one or more bytes that are not a slash, followed by a slash,
    followed by anything that is not a newline through to the end of the text.

    Args:
        bytes: One element.
        start: The first byte of the host.

    Returns:
        The offset of the slash that ends the host, or -1 if the rest of the
        element cannot be read that way.
    """
    var n = len(bytes)
    var at = start
    while at < n and bytes[at] != SLASH:
        at += 1
    if at == start or at == n:
        # An empty host, or no slash after it at all. The pattern needs both.
        return -1
    for i in range(at + 1, n):
        if bytes[i] == NEWLINE:
            return -1
    return at


def _host_cut(bytes: Span[UInt8, _]) -> _Cut:
    """The byte range the hostname occupies, or the whole element without one.

    Args:
        bytes: One element.

    Returns:
        The range to copy out.
    """
    var n = len(bytes)
    var after = _scheme_end(bytes)
    if after < 0:
        return _Cut(0, n)

    # The optional group is preferred, so `www.` is consumed first and only
    # given back when what follows it cannot be read.
    if (
        n - after >= 4
        and bytes[after] == UInt8(ord("w"))
        and bytes[after + 1] == UInt8(ord("w"))
        and bytes[after + 2] == UInt8(ord("w"))
        and bytes[after + 3] == UInt8(ord("."))
    ):
        var stop = _host_from(bytes, after + 4)
        if stop >= 0:
            return _Cut(after + 4, stop - after - 4)

    var stop = _host_from(bytes, after)
    if stop < 0:
        return _Cut(0, n)
    return _Cut(after, stop - after)


def text_hostname(a: StringArray) raises -> StringArray:
    """Extracts the hostname from every element, leaving the ones without one.

    This is ClickBench q28's group key, which is `regexp_replace` against
    `^https?://(?:www\\.)?([^/]+)/.*$` with a back reference to the capture. The
    module docstring has the rule and the two places it is subtler than it
    looks.

    An element that does not match comes back whole, which is what
    `regexp_replace` does rather than answering an empty string, and an empty
    element is one of those. A null stays null.

    Args:
        a: The column.

    Returns:
        A text column of the same height, null wherever the input is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    var validity = Bitmap(copy=a.validity)
    var views = Buffer(n * VIEW_SIZE)
    if n == 0:
        return StringArray(views^, Buffer(0), validity^, 0)

    var morsels = (n + MORSEL_ROWS - 1) // MORSEL_ROWS
    # One entry per morsel, holding that morsel's payload bytes on the way in
    # and the offset it writes them at on the way out.
    var bases = Array[DType.int64](morsels)
    var payload_bytes = 0

    def size(start: Int, stop: Int) {mut bases, imm}:
        var here = 0
        for i in range(start, stop):
            if not a.is_valid(i):
                continue
            # A result that fits inside its own view costs no payload, and a
            # hostname is short, so most rows of a real column land here.
            if a.byte_length(i) > INLINE_CAPACITY:
                var cut = _host_cut(a.unsafe_bytes(i))
                if cut.count > INLINE_CAPACITY:
                    here += cut.count
        bases.unsafe_mut_ptr().unsafe_offset(start // MORSEL_ROWS).unsafe_write(
            Int64(here)
        )

    parallel_morsels(size, n, MORSEL_ROWS)

    # An exclusive prefix, in place. One pass over one entry per morsel.
    for k in range(morsels):
        var here = Int(bases[k])
        bases[k] = Int64(payload_bytes)
        payload_bytes += here

    var payload = Buffer(payload_bytes if payload_bytes > 0 else 1)

    def fill(start: Int, stop: Int) {mut views, mut payload, imm}:
        var dst = views.unsafe_mut_ptr().unsafe_bitcast[StringView]()
        var out = payload.unsafe_mut_ptr()
        var at = Int(bases[start // MORSEL_ROWS])
        for i in range(start, stop):
            # A null's view is written rather than left alone, because a view
            # that was never written is whatever the allocation held and every
            # read of this column would follow it.
            if not a.is_valid(i):
                dst.unsafe_offset(i)[] = StringView()
                continue
            var bytes = a.unsafe_bytes(i)
            var cut = _host_cut(bytes)
            var src = bytes.unsafe_ptr().unsafe_offset(cut.at)
            if cut.count <= INLINE_CAPACITY:
                dst.unsafe_offset(i)[] = make_inline_at(src, cut.count)
            else:
                unsafe_memcpy(
                    dest=out.unsafe_offset(at), src=src, count=cut.count
                )
                dst.unsafe_offset(i)[] = make_long_at(src, cut.count, 0, at)
                at += cut.count

    parallel_morsels(fill, n, MORSEL_ROWS)

    payload.set_size(payload_bytes)
    return StringArray(views^, payload^, validity^, n)
