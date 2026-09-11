"""The byte range of an element of a text column, cut out of it or measured.

This is SQL's `substring` and it is what TPC-H q22 takes the first two characters
of a phone number with. It cuts by bytes and not by code points, which is the
same thing the rest of the column does: `StringView.__len__` is a byte length,
`byte_length` is a byte length, and the reader that filled the column never
promised the bytes were UTF-8 in the first place. That is a real difference from
pandas, where `.str[:2]` slices code points, and it is worth knowing about before
this is pointed at text that is not ASCII. A code point variant is a different
kernel and it should be written when something asks for one rather than now.

Building a text column in parallel is the interesting part, and this is the first
kernel in the library that does it. `StringArray.filter` and `StringArray.take`
both go through `StringBuilder` one element at a time, because a payload offset
is a running total and a running total is serial. The way out is to know the
totals before any bytes move, and here that is easy: the output length of an
element depends on its input length and nothing else, and an input length is in
the view. So the views are read once to size every morsel's share of the payload,
the shares are summed into a base per morsel, and then every morsel writes its
own views and copies into its own stretch of payload with nothing shared at all.

Two things fall out of that which are worth knowing. The sizing pass reads the
views buffer and never touches the payload, so it costs sixteen bytes a row and
no indirection. And it is skipped entirely when the requested length is twelve or
less, because then every result fits inside its own view, the payload is provably
empty, and there is nothing to size.

The same shape would work for `filter` and for `take`, and those two are hotter
than this one. This file is the smallest place to get it right first.

`text_byte_length` is here for the same reason the cut is: it answers a question
about bytes. The character counting twin is `text_character_length` in
`chars.mojo`, and that file exists because the `str` accessor asks every question
in characters. The two are a real divergence and not an accident, and which one a
caller gets depends on who is asking rather than on which was easier to write.
"""

from std.collections.span import Span
from std.memory import unsafe_memcpy

from firepanda.array.array import Array
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
from firepanda.exec import parallel_morsels
from firepanda.exec.morsel import MORSEL_ROWS

from .mask import repair_range


comptime TO_END = -1
"""A length meaning everything from the offset to the end of the element."""


@fieldwise_init
struct _Cut(ImplicitlyCopyable, Movable):
    """The byte range one element's substring occupies."""

    var at: Int
    """Where the substring starts, an offset from the front of the element."""

    var count: Int
    """How many bytes it is."""


def _cut(n: Int, offset: Int, length: Int) -> _Cut:
    """Resolves an offset and a length against one element's byte length.

    Both ends are clamped rather than refused. A substring that runs off the end
    of a short element is the empty string and not an error, which is what SQL
    says and what saves the caller from having to know the longest element in
    the column before it can ask a question about all of them.

    Args:
        n: The element's length in bytes.
        offset: Where to start. Negative counts back from the end, so -3 is the
            last three bytes.
        length: How many bytes to take. Negative means to the end.

    Returns:
        The range, already clamped to the element.
    """
    var at = offset
    if at < 0:
        at = n + at
        if at < 0:
            at = 0
    elif at > n:
        at = n
    var room = n - at
    var count = room if length < 0 else min(length, room)
    return _Cut(at, count)


def text_substring(
    a: StringArray, offset: Int, length: Int = TO_END
) raises -> StringArray:
    """Cuts a byte range out of every element.

    Args:
        a: The column.
        offset: Where each substring starts, in bytes. Negative counts back from
            the end of the element.
        length: How many bytes to take. Negative, which is the default, means
            everything to the end of the element.

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

    if length < 0 or length > INLINE_CAPACITY:
        # Every element's output length comes out of its view, so this pass
        # reads sixteen bytes a row in order and follows no pointers.

        def size(start: Int, stop: Int) {mut bases, imm}:
            var here = 0
            for i in range(start, stop):
                if not a.is_valid(i):
                    continue
                var cut = _cut(a.byte_length(i), offset, length)
                if cut.count > INLINE_CAPACITY:
                    here += cut.count
            bases.unsafe_mut_ptr().unsafe_offset(
                start // MORSEL_ROWS
            ).unsafe_write(Int64(here))

        parallel_morsels(size, n, MORSEL_ROWS)

        # An exclusive prefix, in place. One pass over one entry per morsel,
        # which is eight numbers on a million rows.
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
            var cut = _cut(len(bytes), offset, length)
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


def text_byte_length(a: StringArray) raises -> Array[DType.int64]:
    """How many bytes each element holds.

    This is DuckDB's `strlen`, which ClickBench q27 and q28 average over `URL`
    and `Referer`, and it is a read of a field rather than a pass over anything.
    Every element's length is the first four bytes of its view, so the whole
    kernel touches sixteen bytes a row in order and follows no pointer into the
    payload. `text_character_length` in `chars.mojo` has to walk the bytes of
    every element and this does not, which is why the two are a long way apart in
    cost as well as in meaning.

    Bytes and not characters, which is a real divergence from pandas'
    `Series.str.len` and is deliberate. Bytes are what DuckDB computes, what the
    rest of this library already means by a length, and what the reader that
    filled the column can actually promise, since nothing here has ever claimed
    the payload is valid UTF-8. The `str` accessor keeps the character counting
    one because that is what a caller writing pandas asked for.

    A null's length is null and not zero. `byte_length` answers zero for a null,
    which is the right answer for a caller sizing a buffer and the wrong one
    here, so the validity comes across and the repair at the end of each morsel
    clears those rows. An empty string is a length of zero and is present, and
    that is the pair worth keeping straight on the hits table, where the missing
    text is spelled as an empty string.

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

    def measure(start: Int, stop: Int) {mut out, imm}:
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            dst.unsafe_offset(i).unsafe_write(Int64(a.byte_length(i)))
        repair_range(out, validity, start, stop)

    parallel_morsels(measure, n)

    out.data.validity = validity^
    return out^
