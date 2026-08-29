"""The 16-byte string view.

Strings in a dataframe are dominated by short strings: country codes, currency
symbols, category labels, enum-like status columns. The classic Arrow layout puts
every one of them behind an offsets array, so comparing two strings costs two
dependent loads before the first byte is read.

The view layout inlines the short ones. Every element is exactly 16 bytes:

    short, length <= 12:
      bytes  0..3   length, little endian uint32
      bytes  4..15  the string data, zero padded

    long, length > 12:
      bytes  0..3   length, little endian uint32
      bytes  4..7   the first four bytes of the string
      bytes  8..11  index of the payload block holding the data
      bytes 12..15  offset of the data within that block

The prefix in bytes 4..7 is the reason this layout wins on more than short
strings. Two long strings that differ in their first four bytes compare unequal
without either payload being touched, and that is the common case for a join key
or a sort.

The discriminant is the length field and nothing else. There is no flag bit. A
view is short if and only if `length <= 12`, so a corrupted length does not
silently reinterpret the other twelve bytes as a pointer, it reads out of range
and trips the payload bounds check.

This file is written against the layout tests in tests/test_strview.mojo, which
were written first. See docs/specs/02-architecture.md.
"""

from std.collections.span import Span
from std.memory import unsafe_memcpy

comptime VIEW_SIZE = 16
"""Bytes per view. Two views fit in one 32-byte SIMD register."""

comptime INLINE_CAPACITY = 12
"""Longest string stored inside the view itself."""

comptime PREFIX_LENGTH = 4
"""Bytes of a long string duplicated into the view for prefix comparison."""


@fieldwise_init
struct StringView(ImplicitlyCopyable, Movable, Sized):
    """A 16-byte fixed-size handle to a string.

    The four fields below are the four little-endian words of the layout above.
    For a short string, `_w1` through `_w3` hold data bytes rather than the
    prefix, block and offset they are named for in the long case.
    """

    var _length: UInt32
    var _w1: UInt32
    var _w2: UInt32
    var _w3: UInt32

    def __init__(out self):
        """Constructs the view of the empty string."""
        self._length = 0
        self._w1 = 0
        self._w2 = 0
        self._w3 = 0

    def __len__(self) -> Int:
        """Returns the length of the string in bytes.

        Returns:
            The UTF-8 byte length, not the code point or grapheme count.
        """
        return Int(self._length)

    def is_inline(self) -> Bool:
        """Reports whether the data lives inside this view.

        Returns:
            True if the string is at most 12 bytes long.
        """
        return self._length <= INLINE_CAPACITY

    def block(self) -> Int:
        """Returns the payload block index for a long string.

        Returns:
            The block index. Meaningless for a short string.
        """
        return Int(self._w2)

    def offset(self) -> Int:
        """Returns the byte offset within the payload block for a long string.

        Returns:
            The offset. Meaningless for a short string.
        """
        return Int(self._w3)

    def prefix(self) -> UInt32:
        """Returns the first four bytes of the string as a word.

        For a short string this is the first four data bytes, which are the same
        four bytes a long string would carry, so prefix comparison is uniform
        across both cases and needs no branch.

        Returns:
            The first four bytes, zero padded for strings shorter than four.
        """
        return self._w1

    def shift_offset(mut self, by: UInt32):
        """Moves a long string's payload offset along by a fixed amount.

        Stacking two columns' payloads end to end leaves the second column's
        views pointing at where its bytes used to be, and this is how they are
        moved onto where the bytes are now. It is only meaningful on a long
        view; on a short one `_w3` holds data bytes and adding to it corrupts
        the string.

        Args:
            by: The number of bytes the payload moved.
        """
        self._w3 += by

    def inline_bytes(ref self) -> Pointer[UInt8, origin_of(self)]:
        """Returns a pointer to the twelve inline data bytes.

        Returns:
            A pointer to byte 4 of the view.
        """
        return (
            Pointer(to=self)
            .unsafe_bitcast[UInt8]()
            .unsafe_offset(PREFIX_LENGTH)
        )


def make_inline_at(src: Pointer[UInt8, _], length: Int) -> StringView:
    """Builds a view for a string that fits inside the view.

    Takes a pointer rather than a span, for the caller that has just written the
    bytes into a buffer it owns and would have to name that buffer's origin to
    hand over a span of them.

    Args:
        src: The string bytes. Must be readable for `length` bytes.
        length: How long the string is. Must be at most 12.

    Returns:
        A short view holding the bytes.
    """
    var view = StringView()
    view._length = UInt32(length)
    var dest = (
        Pointer(to=view).unsafe_bitcast[UInt8]().unsafe_offset(PREFIX_LENGTH)
    )
    unsafe_memcpy(dest=dest, src=src, count=length)
    return view


def make_long_at(
    src: Pointer[UInt8, _], length: Int, block: Int, offset: Int
) -> StringView:
    """Builds a view for a string held in a payload block.

    The pointer counterpart of `make_long`, for the same reason as
    `make_inline_at`.

    Args:
        src: The string bytes. Must be readable for at least four bytes. Only
            the prefix is copied into the view.
        length: How long the string is. Must be more than 12.
        block: The index of the payload block holding the data.
        offset: The byte offset of the data within that block.

    Returns:
        A long view pointing at the payload.
    """
    var view = StringView()
    view._length = UInt32(length)
    var prefix_dest = (
        Pointer(to=view).unsafe_bitcast[UInt8]().unsafe_offset(PREFIX_LENGTH)
    )
    unsafe_memcpy(dest=prefix_dest, src=src, count=PREFIX_LENGTH)
    view._w2 = UInt32(block)
    view._w3 = UInt32(offset)
    return view


def make_inline(data: Span[UInt8, _]) -> StringView:
    """Builds a view for a string that fits inside the view.

    Args:
        data: The string bytes. Must be at most 12 bytes long.

    Returns:
        A short view holding the bytes.
    """
    return make_inline_at(data.unsafe_ptr(), len(data))


def make_long(data: Span[UInt8, _], block: Int, offset: Int) -> StringView:
    """Builds a view for a string held in a payload block.

    Args:
        data: The string bytes. Must be more than 12 bytes long. Only the prefix
            is copied into the view.
        block: The index of the payload block holding the data.
        offset: The byte offset of the data within that block.

    Returns:
        A long view pointing at the payload.
    """
    return make_long_at(data.unsafe_ptr(), len(data), block, offset)


def views_equal_short(a: StringView, b: StringView) -> Bool:
    """Compares two views that are both known to be short.

    Args:
        a: The left view.
        b: The right view.

    Returns:
        True if the strings are byte-identical.
    """
    # Short views are zero-padded on construction, so equal strings have equal
    # words and the comparison is four register compares with no loads.
    return (
        a._length == b._length
        and a._w1 == b._w1
        and a._w2 == b._w2
        and a._w3 == b._w3
    )
