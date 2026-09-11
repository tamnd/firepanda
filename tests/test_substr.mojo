"""Tests for cutting a byte range out of a text column.

Every test here is the kernel against the scalar twin in `scalar.mojo`, which
copies one byte at a time out of a `String` and cannot get an offset arithmetic
wrong because it does not do any.

The lengths matter more than usual in this file, because the kernel has two
routes and the switch between them is the requested length rather than anything
about the data. A length of at most twelve takes the route that writes every
element into its own view, and anything longer goes through the builder and the
payload. Both are exercised on the same columns, and the columns hold elements
either side of twelve so that a short input and a long input go down each.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.exec.morsel import MORSEL_ROWS
from firepanda.kernel.concat import concat_strings
from firepanda.kernel.agg import sum_of
from firepanda.kernel.compare import CMP_EQ
from firepanda.kernel.scalar import text_substring_scalar
from firepanda.kernel.chars import text_character_length
from firepanda.kernel.substr import TO_END, text_byte_length, text_substring
from firepanda.kernel.text import compare_text


def sample() -> StringArray:
    """Builds the column every test below cuts.

    The rows straddle the twelve byte line the view layout splits on, and two
    of them are null.

    Returns:
        The column.
    """
    var rows = List[String]()
    rows.append("")
    rows.append("a")
    rows.append("abcdefghijkl")
    rows.append("ignored")
    rows.append("abcdefghijklm")
    rows.append("the quick brown fox jumps over the lazy dog")
    rows.append("ignored")
    rows.append("12345678901234567890")

    var builder = StringBuilder(capacity=len(rows))
    for i in range(len(rows)):
        if i == 3 or i == 6:
            builder.append_null()
        else:
            builder.append(rows[i].as_bytes())
    return builder^.finish()


def agrees(col: StringArray, offset: Int, length: Int) raises:
    """Cuts a column both ways and asserts the two answers are the same.

    Args:
        col: The column.
        offset: Where to start.
        length: How much to take.

    Raises:
        AssertionError: On the first row that differs.
    """
    var label = String("[") + String(offset) + ", " + String(length) + "] "
    var got = text_substring(col, offset, length)
    var want = text_substring_scalar(col, offset, length)
    assert_equal(len(got), len(want), label + "lengths differ")
    for i in range(len(got)):
        assert_equal(got.is_valid(i), want.is_valid(i), label + "validity")
        if got.is_valid(i):
            assert_equal(got[i], want[i], label + "row " + String(i))


def test_a_short_cut_matches_the_twin() raises:
    # Everything here comes out at twelve bytes or fewer, which is the route
    # that never allocates a payload.
    var col = sample()
    agrees(col, 0, 2)
    agrees(col, 0, 12)
    agrees(col, 5, 3)
    agrees(col, 1, 0)
    agrees(col, 0, 0)


def test_a_long_cut_matches_the_twin() raises:
    # And here at least one row comes out longer than twelve, which is the
    # route through the builder.
    var col = sample()
    agrees(col, 0, 13)
    agrees(col, 0, 40)
    agrees(col, 2, 30)


def test_taking_everything_from_an_offset_matches_the_twin() raises:
    var col = sample()
    agrees(col, 0, TO_END)
    agrees(col, 4, TO_END)
    agrees(col, 100, TO_END)


def test_a_negative_offset_counts_back_from_the_end() raises:
    var col = sample()
    agrees(col, -1, TO_END)
    agrees(col, -4, 2)
    agrees(col, -13, TO_END)
    agrees(col, -100, TO_END)

    var got = text_substring(strings_from_list(["abcdef"]), -3, TO_END)
    assert_equal(got[0], "def")


def test_an_offset_past_the_end_is_empty_and_not_an_error() raises:
    # SQL clamps rather than refusing, and a column with rows of different
    # lengths would otherwise need the caller to know the shortest one before
    # asking a question about all of them.
    var col = strings_from_list(["ab", "abcdefghijklmnop"])
    var got = text_substring(col, 8, 4)
    assert_equal(got[0], "")
    assert_equal(got[1], "ijkl")


def test_a_cut_running_off_the_end_stops_at_the_end() raises:
    var col = strings_from_list(["abc"])
    assert_equal(text_substring(col, 1, 99)[0], "bc")
    assert_equal(text_substring(col, 1, TO_END)[0], "bc")


def test_a_null_row_stays_null_on_both_routes() raises:
    var col = sample()
    var short = text_substring(col, 0, 2)
    var long = text_substring(col, 0, 20)
    for i in [3, 6]:
        assert_false(short.is_valid(i))
        assert_false(long.is_valid(i))
    assert_true(short.is_valid(0))
    assert_true(long.is_valid(0))


def test_a_cut_that_crosses_the_inline_boundary_keeps_its_bytes() raises:
    # Thirteen bytes in and twelve bytes out is the pair the view layout is
    # most likely to get wrong: the input lives in the payload and the output
    # has to end up inside a view, prefix and all.
    var col = strings_from_list(["abcdefghijklm"])
    assert_equal(text_substring(col, 0, 12)[0], "abcdefghijkl")
    assert_equal(text_substring(col, 1, 12)[0], "bcdefghijklm")
    assert_equal(text_substring(col, 1, TO_END)[0], "bcdefghijklm")


def test_the_phone_prefix_tpch_asks_for() raises:
    # Query 22 takes the country code off the front of a phone number and
    # groups on it. The column is fifteen bytes, so the input is long and the
    # answer is inline.
    var col = strings_from_list(["23-768-687-3665", "13-750-942-6364"])
    var got = text_substring(col, 0, 2)
    assert_equal(got[0], "23")
    assert_equal(got[1], "13")


def test_a_column_either_side_of_the_morsel_split_matches_the_twin() raises:
    # The payload route sizes each morsel's share of the output, sums the
    # shares into a base per morsel, and then lets every morsel copy into its
    # own stretch. Nothing about that can go wrong inside one morsel, which is
    # all any test above this one has, so the column here is one row past the
    # split and every element is long enough to land in the payload. A morsel
    # that used the running total from the start of the column, or its own
    # offset rather than its base, writes over its neighbour and this is where
    # it shows.
    #
    # Every row is checked and not one of them is read from this file. The twin
    # is asked for the whole column in one call, the two answers are compared
    # by a kernel and reduced by another, and the only thing crossing back here
    # is a count.
    def unit() -> StringArray:
        return strings_from_list(
            [
                "the quick brown fox jumps over the lazy dog",
                "pack my box with five dozen liquor jugs",
                "short",
                "",
            ]
        )

    var col = unit()
    while len(col) < MORSEL_ROWS:
        var pair = List[StringArray]()
        pair.append(col.copy())
        pair.append(col.copy())
        col = concat_strings(pair)
    assert_equal(len(col), MORSEL_ROWS)
    var tail = List[StringArray]()
    tail.append(col^)
    tail.append(unit())
    col = concat_strings(tail)

    var got = text_substring(col, 4, 30)
    var want = text_substring_scalar(col, 4, 30)
    assert_equal(len(got), MORSEL_ROWS + 4)
    var same = compare_text[CMP_EQ](got, want)
    assert_equal(Int(sum_of(same).value), len(col), "rows disagreeing")

    # And that the payload was actually used, which a column that came out all
    # inline would let this test pass without. Two of the four rows are long
    # enough to cut to thirty bytes, one has a byte left after the offset, and
    # one is empty before it starts.
    assert_equal(got[0], "quick brown fox jumps over the")
    assert_equal(got[1], " my box with five dozen liquor")
    assert_equal(got[2], "t")
    assert_equal(got[3], "")


def test_a_byte_length_reads_the_view() raises:
    var got = text_byte_length(sample())
    assert_equal(got[0], 0)
    assert_equal(got[1], 1)
    assert_equal(got[2], 12)
    assert_equal(got[4], 13)
    assert_equal(got[5], 43)
    assert_equal(got[7], 20)


def test_a_byte_length_is_null_where_the_input_is() raises:
    """`byte_length` answers zero for a null, which is right for a caller sizing
    a buffer and wrong for a column. An empty string is also zero and is present,
    and those two rows are next to each other here on purpose."""
    var got = text_byte_length(sample())
    assert_true(got.is_valid(0))
    assert_equal(got[0], 0)
    assert_false(got.is_valid(3))
    assert_false(got.is_valid(6))


def test_a_byte_length_is_not_a_character_length() raises:
    """The divergence from pandas, asserted rather than described. A two byte
    letter is one character, and this is the kernel that says two."""
    var rows = List[String]()
    rows.append("café")
    rows.append("naive")
    var col = strings_from_list(rows)
    var bytes = text_byte_length(StringArray(copy=col))
    var chars = text_character_length(col^)
    assert_equal(bytes[0], 5)
    assert_equal(chars[0], 4)
    assert_equal(bytes[1], 5)
    assert_equal(chars[1], 5)


def test_a_byte_length_of_nothing_is_nothing() raises:
    var got = text_byte_length(strings_from_list(List[String]()))
    assert_equal(len(got), 0)


def test_a_byte_length_crosses_a_morsel() raises:
    var n = MORSEL_ROWS + 1000
    var short = String("ab")
    var long = String("a byte length longer than a view holds")
    var builder = StringBuilder(capacity=n)
    for i in range(n):
        if i % 3 == 0:
            builder.append_null()
        elif i % 3 == 1:
            builder.append(short.as_bytes())
        else:
            builder.append(long.as_bytes())
    var got = text_byte_length(builder^.finish())
    for i in range(n - 6, n):
        if i % 3 == 0:
            assert_false(got.is_valid(i))
        elif i % 3 == 1:
            assert_equal(got[i], 2)
        else:
            assert_equal(got[i], 38)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
