"""Tests that the two ends of a row are trimmed and padded the way Python does.

The interesting assertions here are the three that look like details and are
not. A strip set is a set of characters rather than a prefix, so `aXbab` with
`ab` stripped is `X` and not `Xbab`. Padding counts characters, so a row of
accented letters is padded to the width that was asked for and not to a byte
count that happens to be larger. And an odd amount of padding on both sides goes
to a side that CPython picked and nobody wrote down, so `a` centred in four with
a dot is `.a..` and getting that backwards would be wrong on every second row.

The whitespace table gets a test of its own, because it is twenty nine code
points copied out by hand and a table copied out by hand is a table with a typo
in it until something checks.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.strings import StringArray, StringBuilder
from firepanda.kernel.edges import (
    is_python_space,
    is_sql_space,
    text_pad,
    text_repeat,
    text_strip,
    text_trim,
    text_zfill,
)


def made(var values: List[String]) raises -> StringArray:
    """Builds a text column, using `null` for a missing row.

    Args:
        values: The values, one per row.

    Returns:
        The column.

    Raises:
        Error: If it cannot be built.
    """
    var built = StringBuilder(capacity=len(values))
    for value in values:
        if value == "null":
            built.append_null()
            continue
        built.append(value.as_bytes())
    return built^.finish()


def rows(column: StringArray) raises -> List[String]:
    """Reads a text column back as a list.

    Args:
        column: The column.

    Returns:
        One string per row, and `null` for a missing one.

    Raises:
        Error: If a row cannot be read.
    """
    var read: List[String] = []
    for i in range(len(column)):
        if not column.is_valid(i):
            read.append(String("null"))
            continue
        read.append(
            String(StringSlice(unsafe_from_utf8=column.unsafe_bytes(i)))
        )
    return read^


def test_stripping_nothing_in_particular_removes_whitespace() raises:
    var column = made(["  hi  ", "\tab\n", "no", "   ", "null"])
    var answer = rows(text_strip(column, "".as_bytes(), False, True, True))
    assert_equal(answer[0], "hi")
    assert_equal(answer[1], "ab")
    assert_equal(answer[2], "no")
    assert_equal(answer[3], "")
    assert_equal(answer[4], "null")


def test_stripping_one_end_leaves_the_other_alone() raises:
    var column = made(["  hi  "])
    assert_equal(
        rows(text_strip(column, "".as_bytes(), False, True, False))[0], "hi  "
    )
    assert_equal(
        rows(text_strip(column, "".as_bytes(), False, False, True))[0], "  hi"
    )


def test_a_strip_set_is_a_set_and_not_a_prefix() raises:
    # The thing everybody has been bitten by at least once. Every leading and
    # trailing character that is in the set comes off, in any order and any
    # number of times, so this is `X` and not `Xbab`.
    var column = made(["aXbab", "abab", "Xab"])
    var answer = rows(text_strip(column, "ab".as_bytes(), True, True, True))
    assert_equal(answer[0], "X")
    assert_equal(answer[1], "")
    assert_equal(answer[2], "X")


def test_an_empty_strip_set_removes_nothing() raises:
    # `"  hi  ".strip("")` is `"  hi  "` in Python, which is why an absent set
    # and an empty one have to reach the kernel as two different requests.
    var column = made(["  hi  "])
    assert_equal(
        rows(text_strip(column, "".as_bytes(), True, True, True))[0], "  hi  "
    )


def test_a_strip_set_counts_characters_and_not_bytes() raises:
    var column = made(["ééxéé", "ééxab"])
    var answer = rows(text_strip(column, "é".as_bytes(), True, True, True))
    assert_equal(answer[0], "x")
    assert_equal(answer[1], "xab")


def test_a_trim_removes_the_spaces_sql_calls_spaces() raises:
    var column = made(["  hi  ", "　hi", "null"])
    var answer = rows(text_trim(column, "".as_bytes(), False, True, True))
    assert_equal(answer[0], "hi")
    assert_equal(answer[1], "hi")
    assert_equal(answer[2], "null")


def test_a_trim_leaves_the_whitespace_sql_does_not_call_a_space() raises:
    # Three of the four C0 characters `str.strip` removes and `TRIM` does not,
    # checked against DuckDB rather than against a reading of the standard. The
    # form feed is the fourth and is left out of here rather than written into
    # the source of a test.
    var column = made(["\tab\t", "\nab\n", "\rab\r"])
    var answer = rows(text_trim(column, "".as_bytes(), False, True, True))
    assert_equal(answer[0], "\tab\t")
    assert_equal(answer[1], "\nab\n")
    assert_equal(answer[2], "\rab\r")


def test_a_strip_removes_what_a_trim_leaves() raises:
    # The same rows through the other entry point, which is the pair of tests
    # that says the two tables are not the same table.
    var column = made(["\tab\t"])
    assert_equal(
        rows(text_strip(column, "".as_bytes(), False, True, True))[0], "ab"
    )


def test_a_trim_set_is_a_set_and_not_a_prefix_either() raises:
    # `trim('abcxcba', 'abc')` is `x` in DuckDB, so SQL reads the second
    # argument the way pandas reads it and not as a prefix to remove once.
    var column = made(["abcxcba"])
    assert_equal(
        rows(text_trim(column, "abc".as_bytes(), True, True, True))[0], "x"
    )


def test_a_trim_of_one_end_leaves_the_other_alone() raises:
    var column = made(["  hi  "])
    assert_equal(
        rows(text_trim(column, "".as_bytes(), False, True, False))[0], "hi  "
    )
    assert_equal(
        rows(text_trim(column, "".as_bytes(), False, False, True))[0], "  hi"
    )


def test_every_sql_space_is_a_python_space_and_not_the_other_way() raises:
    var zs = [
        0x20,
        0xA0,
        0x1680,
        0x2000,
        0x2005,
        0x200A,
        0x202F,
        0x205F,
        0x3000,
    ]
    for code in zs:
        assert_true(is_sql_space(code), "a Zs is a space to SQL")
        assert_true(is_python_space(code), "and to Python as well")
    var others = [0x09, 0x0A, 0x0C, 0x0D, 0x85, 0x2028, 0x2029]
    for code in others:
        assert_false(is_sql_space(code), "and these are not spaces to SQL")
        assert_true(is_python_space(code), "though they are to Python")
    assert_false(is_sql_space(0x200B), "a zero width space is not one")
    assert_false(is_sql_space(0x180E), "and neither is a Mongolian separator")


def test_padding_counts_characters_and_not_bytes() raises:
    var column = made(["café", "ab", "null"])
    var answer = rows(text_pad(column, 6, " ".as_bytes(), True, False))
    # Four characters and five bytes, so two spaces and not one.
    assert_equal(answer[0], "  café")
    assert_equal(answer[1], "    ab")
    assert_equal(answer[2], "null")


def test_a_row_that_is_already_wide_enough_is_handed_back() raises:
    var column = made(["abcdef", "abcdefgh"])
    var answer = rows(text_pad(column, 6, ".".as_bytes(), True, True))
    assert_equal(answer[0], "abcdef")
    assert_equal(answer[1], "abcdefgh")


def test_the_odd_character_of_a_both_sided_pad_goes_where_cpython_puts_it() raises:
    # `"a".center(4, ".")` is `".a.."` and `"a".center(3, ".")` is `".a."`, which
    # is not what any of the obvious readings of splitting a gap in two gives.
    var column = made(["a", "a", "ab", "ab"])
    assert_equal(
        rows(text_pad(column, 4, ".".as_bytes(), True, True))[0], ".a.."
    )
    assert_equal(
        rows(text_pad(column, 3, ".".as_bytes(), True, True))[0], ".a."
    )
    assert_equal(
        rows(text_pad(column, 5, ".".as_bytes(), True, True))[2], "..ab."
    )
    assert_equal(
        rows(text_pad(column, 6, ".".as_bytes(), True, True))[2], "..ab.."
    )


def test_zero_filling_puts_the_zeros_after_a_sign() raises:
    var column = made(["-5", "+5", "5", "-", "null"])
    var answer = rows(text_zfill(column, 6))
    assert_equal(answer[0], "-00005")
    assert_equal(answer[1], "+00005")
    assert_equal(answer[2], "000005")
    # A lone sign is a sign, and Python fills after it just the same.
    assert_equal(answer[3], "-00000")
    assert_equal(answer[4], "null")


def test_zero_filling_a_row_that_is_wide_enough_changes_nothing() raises:
    var column = made(["123456", "1234567"])
    var answer = rows(text_zfill(column, 6))
    assert_equal(answer[0], "123456")
    assert_equal(answer[1], "1234567")


def test_repeating_writes_a_row_out_end_to_end() raises:
    var column = made(["ab", "", "null"])
    var answer = rows(text_repeat(column, 3))
    assert_equal(answer[0], "ababab")
    assert_equal(answer[1], "")
    assert_equal(answer[2], "null")


def test_repeating_a_row_no_times_empties_it() raises:
    var column = made(["ab", "ab"])
    assert_equal(rows(text_repeat(column, 0))[0], "")
    assert_equal(rows(text_repeat(column, -2))[0], "")
    assert_equal(rows(text_repeat(column, 1))[0], "ab")


def test_the_whitespace_table_says_what_python_says() raises:
    # The twenty nine that `str.isspace` answers True for, and six that look
    # like they should be in and are not. The Mongolian vowel separator stopped
    # being whitespace in Unicode 6.3 and the zero width space never was one.
    for code in range(0x09, 0x0E):
        assert_true(is_python_space(code))
    for code in range(0x1C, 0x21):
        assert_true(is_python_space(code))
    for code in range(0x2000, 0x200B):
        assert_true(is_python_space(code))
    assert_true(is_python_space(0x85))
    assert_true(is_python_space(0xA0))
    assert_true(is_python_space(0x1680))
    assert_true(is_python_space(0x2028))
    assert_true(is_python_space(0x2029))
    assert_true(is_python_space(0x202F))
    assert_true(is_python_space(0x205F))
    assert_true(is_python_space(0x3000))
    assert_false(is_python_space(ord("a")))
    assert_false(is_python_space(0x08))
    assert_false(is_python_space(0x0E))
    assert_false(is_python_space(0x180E))
    assert_false(is_python_space(0x200B))
    assert_false(is_python_space(0x2060))


def test_an_empty_column_stays_empty() raises:
    var column = made(List[String]())
    assert_equal(len(text_strip(column, "".as_bytes(), False, True, True)), 0)
    assert_equal(len(text_pad(column, 4, " ".as_bytes(), True, True)), 0)
    assert_equal(len(text_zfill(column, 4)), 0)
    assert_equal(len(text_repeat(column, 4)), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
