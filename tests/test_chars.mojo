"""Tests that a position in a text column is a character and not a byte.

Every assertion here that involves an accented letter or an emoji is the same
assertion twice: the answer in characters, and the fact that the answer in bytes
would have been different. That is the whole content of the file it tests, so a
test that only ever used ASCII would pass against a byte kernel and prove
nothing.

The other half is the slice rules, which are Python's and are full of corners
that look like edge cases and are not: a slice off the end is empty rather than
an error, a negative start counts back from the end, and a step of minus one
reverses. Those are the rules a caller already knows, so getting one of them
subtly wrong is worse than not having the method.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.strings import StringArray, StringBuilder
from firepanda.kernel.chars import (
    character_at,
    character_count,
    characters_before,
    text_character_get,
    text_character_length,
    text_character_slice,
    text_character_substring,
    text_find,
    text_remove_prefix,
    text_remove_suffix,
    text_slice_replace,
)
from firepanda.kernel.pattern import rfind_bytes


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
            read.append("null")
            continue
        read.append(column[i])
    return read^


def sliced(
    var values: List[String],
    start: Optional[Int],
    stop: Optional[Int],
    step: Int = 1,
) raises -> List[String]:
    """Slices a column built from a list and reads it back.

    Args:
        values: The values.
        start: The first character.
        stop: The character to stop before.
        step: How far to move between characters.

    Returns:
        One string per row.

    Raises:
        Error: If the slice is refused.
    """
    return rows(text_character_slice(made(values^), start, stop, step))


def cut(
    var values: List[String], start: Int, length: Optional[Int] = None
) raises -> List[String]:
    """Cuts a column built from a list the way SQL cuts one and reads it back.

    Args:
        values: The values.
        start: The first character, counting from one.
        length: How many characters, or nothing for everything to the end.

    Returns:
        One string per row.

    Raises:
        Error: If the cut is refused.
    """
    return rows(text_character_substring(made(values^), start, length))


def test_a_character_count_is_not_a_byte_count() raises:
    var col = made(["abc", "héllo", "日本"])
    var lengths = text_character_length(col)
    assert_equal(lengths[0], 3)
    assert_equal(lengths[1], 5)
    assert_equal(lengths[2], 2)
    # The same three rows in bytes, which is what every other kernel counts.
    assert_equal(col.byte_length(0), 3)
    assert_equal(col.byte_length(1), 6)
    assert_equal(col.byte_length(2), 6)


def test_a_missing_row_has_no_length() raises:
    var lengths = text_character_length(made(["abc", "null"]))
    assert_true(lengths.is_valid(0))
    assert_false(lengths.is_valid(1))


def test_an_empty_string_has_length_zero() raises:
    assert_equal(text_character_length(made([""]))[0], 0)


def test_a_slice_takes_characters() raises:
    var read = sliced(["abcdef", "héllo", "日本語です"], 1, 4)
    assert_equal(read[0], "bcd")
    assert_equal(read[1], "éll")
    assert_equal(read[2], "本語で")


def test_a_slice_past_the_end_is_empty_rather_than_an_error() raises:
    var read = sliced(["ab", ""], 5, 9)
    assert_equal(read[0], "")
    assert_equal(read[1], "")


def test_a_negative_start_counts_back_from_the_end() raises:
    var read = sliced(["abcdef", "日本語です"], -3, None)
    assert_equal(read[0], "def")
    assert_equal(read[1], "語です")


def test_a_negative_start_longer_than_the_string_starts_at_the_front() raises:
    assert_equal(sliced(["ab"], -9, None)[0], "ab")


def test_a_step_takes_every_nth_character() raises:
    var read = sliced(["abcdef", "héllo"], None, None, 2)
    assert_equal(read[0], "ace")
    assert_equal(read[1], "hlo")


def test_a_negative_step_reverses() raises:
    var read = sliced(["abc", "héllo"], None, None, -1)
    assert_equal(read[0], "cba")
    assert_equal(read[1], "olléh")


def test_a_step_of_zero_is_refused() raises:
    var said = String("")
    try:
        _ = sliced(["abc"], None, None, 0)
    except error:
        said = String(error)
    assert_true("step cannot be zero" in said)


def test_a_slice_keeps_the_nulls() raises:
    var read = sliced(["abc", "null"], 0, 2)
    assert_equal(read[0], "ab")
    assert_equal(read[1], "null")


def test_getting_one_character_is_not_getting_one_byte() raises:
    var read = rows(text_character_get(made(["abc", "héllo", "日本"]), 1))
    assert_equal(read[0], "b")
    assert_equal(read[1], "é")
    assert_equal(read[2], "本")


def test_getting_past_the_end_gives_a_null_rather_than_raising() raises:
    # Which is pandas and is not Python, where the same index is an IndexError.
    var read = rows(text_character_get(made(["abc", "a"]), 2))
    assert_equal(read[0], "c")
    assert_equal(read[1], "null")


def test_getting_a_negative_index_counts_back() raises:
    assert_equal(rows(text_character_get(made(["héllo"]), -1))[0], "o")


def test_a_find_answers_a_character_position() raises:
    var col = made(["abc", "héllo", "xyz"])
    var found = text_find(col, "l".as_bytes(), None, None, False)
    assert_equal(found[0], -1)
    # Byte three would be the answer if this counted bytes, because the accented
    # letter is two of them.
    assert_equal(found[1], 2)
    assert_equal(found[2], -1)


def test_a_find_that_misses_is_minus_one_and_not_an_error() raises:
    assert_equal(
        text_find(made(["abc"]), "q".as_bytes(), None, None, False)[0], -1
    )


def test_a_find_starts_where_it_is_told() raises:
    var col = made(["abcabc"])
    assert_equal(text_find(col, "a".as_bytes(), None, None, False)[0], 0)
    assert_equal(text_find(col, "a".as_bytes(), 1, None, False)[0], 3)


def test_a_find_stops_where_it_is_told() raises:
    var col = made(["abcabc"])
    assert_equal(text_find(col, "a".as_bytes(), 1, 3, False)[0], -1)


def test_a_reverse_find_answers_the_last_one() raises:
    var col = made(["abcabc", "héllo"])
    var found = text_find(col, "a".as_bytes(), None, None, True)
    assert_equal(found[0], 3)
    assert_equal(found[1], -1)
    assert_equal(text_find(col, "l".as_bytes(), None, None, True)[1], 3)


def test_a_find_keeps_the_nulls() raises:
    var found = text_find(
        made(["abc", "null"]), "a".as_bytes(), None, None, False
    )
    assert_true(found.is_valid(0))
    assert_false(found.is_valid(1))


def test_replacing_a_slice_puts_bytes_where_characters_were() raises:
    var read = rows(
        text_slice_replace(made(["abcdef", "héllo"]), 1, 3, "XX".as_bytes())
    )
    assert_equal(read[0], "aXXdef")
    assert_equal(read[1], "hXXlo")


def test_replacing_an_empty_slice_inserts() raises:
    var read = rows(text_slice_replace(made(["abc"]), 1, 1, "X".as_bytes()))
    assert_equal(read[0], "aXbc")


def test_replacing_to_the_end_truncates() raises:
    var read = rows(
        text_slice_replace(made(["abcdef"]), 2, None, "!".as_bytes())
    )
    assert_equal(read[0], "ab!")


def test_removing_a_prefix_that_is_there_and_one_that_is_not() raises:
    var read = rows(
        text_remove_prefix(made(["abc", "xbc", "null"]), "a".as_bytes())
    )
    assert_equal(read[0], "bc")
    assert_equal(read[1], "xbc")
    assert_equal(read[2], "null")


def test_removing_a_prefix_removes_only_one() raises:
    # `removeprefix` is not a strip, and this is the difference people trip on.
    assert_equal(
        rows(text_remove_prefix(made(["aaab"]), "a".as_bytes()))[0], "aab"
    )


def test_removing_a_suffix_that_is_there_and_one_that_is_not() raises:
    var read = rows(
        text_remove_suffix(made(["abc", "abx", "null"]), "c".as_bytes())
    )
    assert_equal(read[0], "ab")
    assert_equal(read[1], "abx")
    assert_equal(read[2], "null")


def test_removing_a_prefix_longer_than_the_string_changes_nothing() raises:
    assert_equal(
        rows(text_remove_prefix(made(["ab"]), "abcd".as_bytes()))[0], "ab"
    )


def test_the_helpers_agree_about_where_a_character_starts() raises:
    var text = String("héllo")
    var bytes = text.as_bytes()
    assert_equal(character_count(bytes), 5)
    assert_equal(character_at(bytes, 0), 0)
    assert_equal(character_at(bytes, 1), 1)
    # The accented letter is two bytes, so every character after it is one byte
    # further along than its position.
    assert_equal(character_at(bytes, 2), 3)
    assert_equal(character_at(bytes, 9), 6)
    assert_equal(characters_before(bytes, 3), 2)


def test_a_reverse_byte_search_finds_the_last_match() raises:
    var hay = String("abcabc")
    var bytes = hay.as_bytes()
    assert_equal(rfind_bytes(bytes, "bc".as_bytes(), 0, 6), 4)
    assert_equal(rfind_bytes(bytes, "bc".as_bytes(), 0, 5), 1)
    assert_equal(rfind_bytes(bytes, "q".as_bytes(), 0, 6), -1)


def test_a_sql_substring_counts_from_one_and_counts_characters() raises:
    # Every answer in this file's SQL half was read out of DuckDB 1.5.1 first.
    # The accented row is the one that says this is not the byte kernel: cut by
    # byte, two characters of `héllo` would be one letter and half of another.
    var read = cut(["hello", "héllo", "日本語です"], 1, 2)
    assert_equal(read[0], "he")
    assert_equal(read[1], "hé")
    assert_equal(read[2], "日本")


def test_a_sql_substring_with_no_length_runs_to_the_end() raises:
    var read = cut(["hello", "日本語です"], 2)
    assert_equal(read[0], "ello")
    assert_equal(read[1], "本語です")


def test_a_start_of_zero_loses_the_first_character_of_the_window() raises:
    # The window covers positions 0, 1 and 2, and no string has a position 0,
    # so what comes back is two characters and not three. This is the corner
    # that makes these rules different from Python's rather than a rewriting of
    # them, and clamping the start to the front would answer `hel`.
    assert_equal(cut(["hello"], 0, 3)[0], "he")


def test_a_start_of_zero_with_no_length_is_the_whole_string() raises:
    assert_equal(cut(["hello"], 0)[0], "hello")


def test_a_negative_start_counts_back_from_the_end_of_the_element() raises:
    var read = cut(["hello", "héllo"], -2, 2)
    assert_equal(read[0], "lo")
    assert_equal(read[1], "lo")


def test_a_start_far_enough_back_leaves_the_window_off_the_front() raises:
    # Python would clamp this to the front and answer `hel`. SQL clips the
    # window instead, and the window is positions -4, -3 and -2, none of which
    # the string has.
    assert_equal(cut(["hello"], -10, 3)[0], "")
    # With no length the window has no far end, so the same start keeps
    # everything.
    assert_equal(cut(["hello"], -10)[0], "hello")


def test_a_negative_length_runs_the_window_backwards() raises:
    # `substring('hello', 2, -1)` is the one character before position 2, which
    # falls out of reading the two numbers as the ends of a range.
    assert_equal(cut(["hello"], 2, -1)[0], "h")
    assert_equal(cut(["hello"], 4, -2)[0], "el")


def test_a_window_off_the_end_is_empty_rather_than_an_error() raises:
    var read = cut(["hello", ""], 10, 2)
    assert_equal(read[0], "")
    assert_equal(read[1], "")


def test_a_length_of_zero_takes_nothing() raises:
    assert_equal(cut(["hello"], 2, 0)[0], "")


def test_a_length_past_the_end_stops_at_the_end() raises:
    assert_equal(cut(["hello"], 1, 100)[0], "hello")


def test_a_missing_row_is_missing_in_the_answer() raises:
    var read = cut(["hello", "null"], 1, 2)
    assert_equal(read[0], "he")
    assert_equal(read[1], "null")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
