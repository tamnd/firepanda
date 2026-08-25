"""Tests for the 16-byte string view.

The layout is fixed by the Arrow StringView spec, and the reason to test it byte
by byte rather than through the accessors is that a buffer written by this code
gets read by Arrow consumers that have their own struct definition. `size_of` and
the placement of the prefix are therefore assertions about an external contract,
not about our own accessors.

Zero padding of the inline bytes is the other load-bearing property: it is what
lets a short comparison be two 64-bit word comparisons with no length branch.
"""

from std.sys.info import size_of
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
)

from firepanda.array.strview import (
    INLINE_CAPACITY,
    PREFIX_LENGTH,
    VIEW_SIZE,
    StringView,
    make_inline,
    make_long,
    views_equal_short,
)


def bytes_of(text: String) -> List[UInt8]:
    """Copies a string's bytes into a list the view builders can span.

    Args:
        text: The string.

    Returns:
        The bytes, without a terminator.
    """
    var out = List[UInt8]()
    for i in range(text.byte_length()):
        out.append(text.unsafe_ptr().unsafe_offset(i).unsafe_load())
    return out^


def test_the_view_is_exactly_sixteen_bytes() raises:
    assert_equal(size_of[StringView](), VIEW_SIZE)
    assert_equal(VIEW_SIZE, 16)
    assert_equal(INLINE_CAPACITY, 12)
    assert_equal(PREFIX_LENGTH, 4)


def test_default_view_is_the_empty_string() raises:
    var view = StringView()
    assert_equal(len(view), 0)
    assert_true(view.is_inline())
    assert_equal(view.prefix(), UInt32(0))


def test_short_string() raises:
    var data = bytes_of("hello")
    var view = make_inline(data)
    assert_equal(len(view), 5)
    assert_true(view.is_inline())

    var inline = view.inline_bytes()
    assert_equal(inline.unsafe_offset(0).unsafe_load(), UInt8(ord("h")))
    assert_equal(inline.unsafe_offset(4).unsafe_load(), UInt8(ord("o")))


def test_short_string_is_zero_padded() raises:
    var view = make_inline(bytes_of("abc"))
    var inline = view.inline_bytes()
    for i in range(3, INLINE_CAPACITY):
        assert_equal(inline.unsafe_offset(i).unsafe_load(), UInt8(0))


def test_twelve_bytes_still_fits_inline() raises:
    var view = make_inline(bytes_of("012345678901"))
    assert_equal(len(view), 12)
    assert_true(view.is_inline())
    assert_equal(
        view.inline_bytes().unsafe_offset(11).unsafe_load(), UInt8(ord("1"))
    )


def test_long_string_carries_a_prefix_and_a_location() raises:
    var view = make_long(bytes_of("this string does not fit inline"), 3, 128)
    assert_equal(len(view), 31)
    assert_false(view.is_inline())
    assert_equal(view.block(), 3)
    assert_equal(view.offset(), 128)

    var inline = view.inline_bytes()
    assert_equal(inline.unsafe_offset(0).unsafe_load(), UInt8(ord("t")))
    assert_equal(inline.unsafe_offset(3).unsafe_load(), UInt8(ord("s")))


def test_prefix_is_the_same_word_for_both_forms() raises:
    # A filter on a string column compares prefixes before touching the payload.
    # That is only sound if a short and a long view of strings with the same first
    # four bytes produce the same prefix word.
    var short = make_inline(bytes_of("abcd"))
    var long = make_long(bytes_of("abcdefghijklmnop"), 0, 0)
    assert_equal(short.prefix(), long.prefix())


def test_short_equality() raises:
    assert_true(
        views_equal_short(
            make_inline(bytes_of("abc")), make_inline(bytes_of("abc"))
        )
    )
    assert_false(
        views_equal_short(
            make_inline(bytes_of("abc")), make_inline(bytes_of("abd"))
        )
    )
    assert_false(
        views_equal_short(
            make_inline(bytes_of("abc")), make_inline(bytes_of("ab"))
        )
    )
    assert_true(views_equal_short(StringView(), make_inline(bytes_of(""))))


def test_length_alone_decides_the_form() raises:
    # There is no flag bit. Anything at or below twelve is inline and anything
    # above is not, which is what the Arrow spec says and what lets a consumer
    # written against the spec read our buffers.
    assert_true(make_inline(bytes_of("012345678901")).is_inline())
    assert_false(make_long(bytes_of("0123456789012"), 0, 0).is_inline())


def test_prefix_of_a_string_shorter_than_the_prefix() raises:
    var view = make_inline(bytes_of("ab"))
    assert_not_equal(view.prefix(), UInt32(0))
    assert_equal(view.prefix() >> UInt32(16), UInt32(0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
