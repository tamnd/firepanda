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

from std.collections.string import Codepoint
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.kernel.casefix import KEPT_BY_SWAP
from firepanda.kernel.charclass import TITLE_ONLY_EDGES
from firepanda.kernel.chars import (
    character_at,
    character_count,
    characters_before,
    text_capitalize,
    text_case,
    text_casefold,
    text_character_get,
    text_character_length,
    text_character_slice,
    text_character_substring,
    text_find,
    text_is_ascii,
    text_is_lower,
    text_is_space,
    text_is_title,
    text_is_upper,
    text_remove_prefix,
    text_remove_suffix,
    text_slice_replace,
    text_swapcase,
    text_title,
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


def cased(var values: List[String], upper: Bool) raises -> List[String]:
    """Changes the case of a column built from a list and reads it back.

    Args:
        values: The values.
        upper: Whether to raise the case rather than lower it.

    Returns:
        One string per row.

    Raises:
        Error: If the column cannot be built.
    """
    return rows(text_case(made(values^), upper))


def capitalised(var values: List[String]) raises -> List[String]:
    """Capitalises a column built from a list and reads it back.

    Args:
        values: The values.

    Returns:
        One string per row.

    Raises:
        Error: If the column cannot be built.
    """
    return rows(text_capitalize(made(values^)))


def folded(var values: List[String]) raises -> List[String]:
    """Folds a column built from a list and reads it back.

    Args:
        values: The values.

    Returns:
        One string per row.

    Raises:
        Error: If the column cannot be built.
    """
    return rows(text_casefold(made(values^)))


def swapped(var values: List[String]) raises -> List[String]:
    """Swaps the case of a column built from a list and reads it back.

    Args:
        values: The values.

    Returns:
        One string per row.

    Raises:
        Error: If the column cannot be built.
    """
    return rows(text_swapcase(made(values^)))


def asked(column: Array[DType.bool]) raises -> List[String]:
    """Reads a mask back as words, so that a missing row can be named.

    Args:
        column: The mask.

    Returns:
        `yes`, `no` or `null`, one per row.

    Raises:
        Error: If a row cannot be read.
    """
    var read: List[String] = []
    for i in range(len(column)):
        if not column.is_valid(i):
            read.append("null")
        elif column[i]:
            read.append("yes")
        else:
            read.append("no")
    return read^


def test_a_case_change_rewrites_every_row() raises:
    var raised = cased(["abc", "Mixed Case", ""], True)
    assert_equal(raised[0], "ABC")
    assert_equal(raised[1], "MIXED CASE")
    assert_equal(raised[2], "")
    var dropped = cased(["ABC", "Mixed Case", ""], False)
    assert_equal(dropped[0], "abc")
    assert_equal(dropped[1], "mixed case")
    assert_equal(dropped[2], "")


def test_a_case_change_keeps_a_missing_row_missing() raises:
    var raised = cased(["abc", "null"], True)
    assert_equal(raised[0], "ABC")
    assert_equal(raised[1], "null")


def test_an_accented_letter_changes_case_like_a_plain_one() raises:
    assert_equal(cased(["café"], True)[0], "CAFÉ")
    assert_equal(cased(["CAFÉ"], False)[0], "café")


def test_a_sharp_s_raises_to_one_letter_and_not_two() raises:
    # The standard library gives `SS` here, which is Python's answer. pandas
    # holds text in Arrow and Arrow gives the capital sharp s, so the table in
    # casefix.mojo overrides the library and the row keeps its length.
    var raised = cased(["straße"], True)
    assert_equal(raised[0], "STRAẞE")
    assert_equal(len(raised[0].as_bytes()), 8)


def test_a_ligature_is_left_alone_where_python_would_split_it() raises:
    # The same difference in a different alphabet, and the second row of the
    # conformance corpus that depends on it.
    assert_equal(cased(["ﬁance"], True)[0], "ﬁANCE")


def test_a_capital_i_with_a_dot_lowers_to_a_plain_letter() raises:
    # Python writes U+0130 out as `i` followed by a combining dot above and
    # Arrow writes a plain `i`, so the row loses the dot and keeps its length.
    # This is the row of the corpus that says which of the two is being copied.
    var dropped = cased(["İstanbul"], False)
    assert_equal(dropped[0], "istanbul")
    assert_equal(len(dropped[0].as_bytes()), 8)


def test_a_correction_is_made_wherever_the_character_sits() raises:
    # Not just at the front, and not just once, and a row with nothing in the
    # table goes through the fast path beside them unchanged.
    var raised = cased(["aßb", "ßß", "plain"], True)
    assert_equal(raised[0], "AẞB")
    assert_equal(raised[1], "ẞẞ")
    assert_equal(raised[2], "PLAIN")


def test_a_dotless_i_raises_to_a_plain_capital() raises:
    assert_equal(cased(["ıstanbul"], True)[0], "ISTANBUL")


def test_an_accented_row_with_nothing_to_correct_keeps_its_accents() raises:
    # This row has a byte large enough to reach the table's first test and no
    # character that is in the table, which is the path most non ASCII text
    # takes, so it is worth a row of its own.
    assert_equal(cased(["éèê"], True)[0], "ÉÈÊ")
    assert_equal(cased(["ÉÈÊ"], False)[0], "éèê")


def test_bytes_that_are_not_utf8_come_back_as_they_went_in() raises:
    # The element is a letter and then a lead byte with nothing after it. The
    # standard library's walk would take the first byte of the next row to
    # finish the character, which is the one thing this file promises not to
    # do, so the element is copied instead.
    var truncated = List[UInt8]()
    truncated.append(0x61)
    truncated.append(0xC4)
    var built = StringBuilder(capacity=2)
    built.append(Span(truncated))
    built.append("b".as_bytes())
    var raised = text_case(built^.finish(), True)
    var first = raised.unsafe_bytes(0)
    assert_equal(len(first), 2)
    assert_equal(Int(first[0]), 0x61)
    assert_equal(Int(first[1]), 0xC4)
    assert_equal(raised[1], "B")


def test_whitespace_is_a_question_about_every_character() raises:
    var read = asked(text_is_space(made([" ", " \t\n", "a b", "ab"])))
    assert_equal(read[0], "yes")
    assert_equal(read[1], "yes")
    assert_equal(read[2], "no")
    assert_equal(read[3], "no")


def test_an_empty_row_answers_no_to_all_three() raises:
    # Python's rule, which is that all of nothing is not enough: there has to
    # be a character for the question to be about.
    assert_equal(asked(text_is_space(made([""])))[0], "no")
    assert_equal(asked(text_is_lower(made([""])))[0], "no")
    assert_equal(asked(text_is_upper(made([""])))[0], "no")


def test_the_two_case_questions_are_not_opposites() raises:
    var lower = asked(text_is_lower(made(["abc", "ABC", "aBc", "42", "a1"])))
    var upper = asked(text_is_upper(made(["abc", "ABC", "aBc", "42", "a1"])))
    assert_equal(lower[0], "yes")
    assert_equal(upper[0], "no")
    assert_equal(lower[1], "no")
    assert_equal(upper[1], "yes")
    assert_equal(lower[2], "no")
    assert_equal(upper[2], "no")
    # A row with no cased character in it is neither, rather than both.
    assert_equal(lower[3], "no")
    assert_equal(upper[3], "no")
    # And one cased character is enough to answer, whatever it is sitting next
    # to.
    assert_equal(lower[4], "yes")
    assert_equal(upper[4], "no")


def test_a_case_question_keeps_a_missing_row_missing() raises:
    assert_equal(asked(text_is_lower(made(["abc", "null"])))[1], "null")
    assert_equal(asked(text_is_upper(made(["ABC", "null"])))[1], "null")
    assert_equal(asked(text_is_space(made([" ", "null"])))[1], "null")


def test_a_case_question_about_bytes_that_are_not_utf8_is_no() raises:
    var truncated = List[UInt8]()
    truncated.append(0x61)
    truncated.append(0xC4)
    var built = StringBuilder(capacity=2)
    built.append(Span(truncated))
    built.append("b".as_bytes())
    var col = built^.finish()
    assert_equal(asked(text_is_lower(col))[0], "no")
    assert_equal(asked(text_is_lower(col))[1], "yes")


def one(cp: UInt32) -> String:
    """The one character row for a code point, written as a number.

    Most of the characters the class questions turn on are invisible or are
    indistinguishable from an ordinary space in a source file, and a test whose
    reader cannot tell which character it is about is not much of a test.

    Args:
        cp: The code point.

    Returns:
        A string of that one character.
    """
    return String(Codepoint(unsafe_unchecked_codepoint=cp))


def test_the_spaces_above_ascii_are_spaces() raises:
    # The larger half of issue 748. The standard library's data knew about the
    # six ASCII spaces and almost nothing else, so a row holding the non
    # breaking space, which is the one ordinary text is full of, answered no.
    # These are the non breaking space, the ogham mark, the em space, the
    # narrow no break space and the ideographic space.
    var marks: List[String] = [
        one(0x00A0),
        one(0x1680),
        one(0x2003),
        one(0x202F),
        one(0x3000),
    ]
    var read = asked(text_is_space(made(marks^)))
    for i in range(5):
        assert_equal(read[i], "yes")


def test_a_control_character_arrow_calls_a_space_is_one() raises:
    # U+001F is a space to Arrow and is not one to Python, and pandas holds
    # text in Arrow, so this is a row where following the oracle rather than
    # the language means answering yes.
    assert_equal(asked(text_is_space(made(["\x1f"])))[0], "yes")


def test_a_space_question_is_asked_of_every_character() raises:
    var read = asked(text_is_space(made(["  ", " x", "x "])))
    assert_equal(read[0], "yes")
    assert_equal(read[1], "no")
    assert_equal(read[2], "no")


def test_a_titlecase_character_is_neither_lower_nor_upper() raises:
    # The third case, and the reason both questions have to read a class
    # neither of them is named after. The standard library counted these as
    # both cases at once, which made a row holding one answer yes twice.
    var digraphs: List[String] = ["ǅ", "ǈ", "ǋ", "ǲ"]
    var lower = asked(text_is_lower(made(digraphs.copy())))
    var upper = asked(text_is_upper(made(digraphs^)))
    for i in range(4):
        assert_equal(lower[i], "no")
        assert_equal(upper[i], "no")


def test_a_titlecase_character_spoils_the_row_it_sits_in() raises:
    var lower = asked(text_is_lower(made(["ǅa", "aǅ", "abc"])))
    assert_equal(lower[0], "no")
    assert_equal(lower[1], "no")
    assert_equal(lower[2], "yes")


def test_a_letter_that_looks_lower_case_and_is_in_no_case_at_all() raises:
    # A modifier letter and the feminine ordinal. Both are letters, neither is
    # cased, so both behave here exactly like a digit does: they answer no on
    # their own and they do not stop the row around them answering yes.
    var alone = asked(text_is_lower(made(["ª", "ᵃ"])))
    assert_equal(alone[0], "no")
    assert_equal(alone[1], "no")
    assert_equal(asked(text_is_lower(made(["ªa"])))[0], "yes")


def test_the_case_questions_read_an_alphabet_that_is_not_latin() raises:
    var lower = asked(text_is_lower(made(["αβγ", "ΑΒΓ", "мир", "МИР"])))
    var upper = asked(text_is_upper(made(["αβγ", "ΑΒΓ", "мир", "МИР"])))
    assert_equal(lower[0], "yes")
    assert_equal(upper[0], "no")
    assert_equal(lower[1], "no")
    assert_equal(upper[1], "yes")
    assert_equal(lower[2], "yes")
    assert_equal(upper[3], "yes")


def test_the_titlecase_class_is_the_list_swapcase_already_had() raises:
    # The same 31 code points arrived at from opposite directions. `casefix`
    # derives them from the mappings, because a character both of whose
    # mappings move it and which is in neither case can only be the third one,
    # and `charclass` reads the class straight from Arrow. The two agreeing is
    # a check on both, and it is cheap enough to make rather than to claim.
    var kept = materialize[KEPT_BY_SWAP]()
    var edges = materialize[TITLE_ONLY_EDGES]()
    var counted = 0
    for i in range(0, len(edges), 2):
        for cp in range(Int(edges[i]), Int(edges[i + 1])):
            var found = False
            for k in range(len(kept)):
                if Int(kept[k]) == cp:
                    found = True
            assert_true(found)
            counted += 1
    assert_equal(counted, len(kept))


def test_a_character_above_the_basic_plane_has_a_case_too() raises:
    # Deseret, which is the far end of the table and the one place an edge
    # search that overflowed somewhere would show.
    assert_equal(asked(text_is_lower(made(["𐐨"])))[0], "yes")
    assert_equal(asked(text_is_upper(made(["𐐀"])))[0], "yes")
    assert_equal(asked(text_is_lower(made(["𐐀"])))[0], "no")


def test_capitalising_raises_the_first_character_and_drops_the_rest() raises:
    var out = capitalised(["hello world", "HELLO WORLD", "hello WORLD"])
    assert_equal(out[0], "Hello world")
    assert_equal(out[1], "Hello world")
    assert_equal(out[2], "Hello world")


def test_capitalising_starts_at_the_first_character_whatever_it_is() raises:
    # A row starting with something that has no case is not skipped over in
    # search of something that has one, so the letter after a digit stays
    # lower even though it is the first letter in the row.
    assert_equal(capitalised(["1abc def"])[0], "1abc def")
    assert_equal(capitalised(["  spaced  "])[0], "  spaced  ")


def test_capitalising_an_empty_row_and_a_missing_row() raises:
    var out = capitalised(["", "null", "a"])
    assert_equal(out[0], "")
    assert_equal(out[1], "null")
    assert_equal(out[2], "A")


def test_capitalising_corrects_the_first_character_and_the_rest() raises:
    # The sharp s is one of the hundred and forty nine, in the head here and in
    # the tail in the second row, so both paths through the element go through
    # the table.
    assert_equal(capitalised(["ßa"])[0], "ẞa")
    assert_equal(capitalised(["Aß"])[0], "Aß")
    assert_equal(capitalised(["ΟΔΟΣ"])[0], "Οδοσ")


def test_swapping_case_exchanges_the_two_cases() raises:
    var out = swapped(["hello WORLD", "MiXeD", "o'neill"])
    assert_equal(out[0], "HELLO world")
    assert_equal(out[1], "mIxEd")
    assert_equal(out[2], "O'NEILL")


def test_swapping_case_leaves_a_character_with_no_case_alone() raises:
    assert_equal(swapped(["1 2 3 !?"])[0], "1 2 3 !?")


def test_swapping_case_leaves_a_titlecase_character_alone() raises:
    # A titlecase character is in neither case, so there is no other case to
    # write it in, and both of its mappings would move it if the mappings were
    # all this had to go on.
    assert_equal(swapped(["ǅungla"])[0], "ǅUNGLA")
    assert_equal(swapped(["ᾈα"])[0], "ᾈΑ")


def test_swapping_case_keeps_a_row_the_same_length_in_characters() raises:
    var out = swapped(["straße", "İstanbul", "ﬁance"])
    assert_equal(out[0], "STRAẞE")
    assert_equal(out[1], "iSTANBUL")
    assert_equal(out[2], "ﬁANCE")


def test_swapping_case_of_an_empty_row_and_a_missing_row() raises:
    var out = swapped(["", "null", "Ab"])
    assert_equal(out[0], "")
    assert_equal(out[1], "null")
    assert_equal(out[2], "aB")


def test_the_two_new_names_leave_bytes_that_are_not_utf8_alone() raises:
    var truncated = List[UInt8]()
    truncated.append(0x61)
    truncated.append(0xC4)
    var built = StringBuilder(capacity=2)
    built.append(Span(truncated))
    built.append("ab".as_bytes())
    var col = built^.finish()
    assert_equal(len(rows(text_swapcase(col))[0].as_bytes()), 2)
    assert_equal(rows(text_swapcase(col))[1], "AB")
    assert_equal(len(rows(text_capitalize(col))[0].as_bytes()), 2)
    assert_equal(rows(text_capitalize(col))[1], "Ab")


def test_folding_is_lowering_for_anything_ordinary() raises:
    var out = folded(["ABC", "Ab", "café", "CAFÉ", "123", "ΑΒΓ"])
    assert_equal(out[0], "abc")
    assert_equal(out[1], "ab")
    assert_equal(out[2], "café")
    assert_equal(out[3], "café")
    assert_equal(out[4], "123")
    assert_equal(out[5], "αβγ")


def test_folding_makes_a_row_longer_where_lowering_never_does() raises:
    # The one rewrite in this file that can add characters. Lowering a sharp s
    # leaves it alone and folding it writes two letters, which is the whole
    # reason folding is a separate name rather than a spelling of lower.
    var out = folded(["ß", "Straße", "ﬁ"])
    assert_equal(out[0], "ss")
    assert_equal(out[1], "strasse")
    assert_equal(out[2], "fi")


def test_folding_brings_two_rows_a_reader_calls_equal_together() raises:
    # The point of the name. Neither of these lowers to the other and both of
    # them fold to the same thing, which is what folding is for.
    var out = folded(["Straße", "STRASSE"])
    assert_equal(out[0], out[1])


def test_folding_is_not_lowering_for_the_micro_sign() raises:
    # The lowest code point in the table, and the reason there is no byte test
    # on this path: its lead byte is the lowest a non ASCII character can have.
    var out = folded(["µ", "ſ", "İ"])
    assert_equal(out[0], "μ")
    assert_equal(out[1], "s")
    assert_equal(len(out[2].as_bytes()), 3)


def test_folding_corrects_the_same_code_points_the_other_names_do() raises:
    # A code point the standard library lowers wrongly still has to be lowered
    # rightly here, since the fold table only holds the ones that fold to
    # something other than their lower case and everything else falls through.
    var out = folded(["ẞ", "ϴ"])
    assert_equal(out[0], "ss")
    assert_equal(out[1], "θ")


def test_folding_a_titlecase_character_takes_it_all_the_way_down() raises:
    # `swapcase` leaves these alone because they are in neither case. Folding
    # does not care what case anything is in, so all three of a titlecase
    # character, the capital beside it and the small one fold to the same thing.
    var out = folded(["ǅ", "Ǆ", "ǆ"])
    assert_equal(out[0], out[2])
    assert_equal(out[1], out[2])


def test_folding_an_empty_row_and_a_missing_row() raises:
    var out = folded(["", "null", "AB"])
    assert_equal(out[0], "")
    assert_equal(out[1], "null")
    assert_equal(out[2], "ab")


def test_folding_leaves_bytes_that_are_not_utf8_alone() raises:
    var truncated = List[UInt8]()
    truncated.append(0x41)
    truncated.append(0xC4)
    var built = StringBuilder(capacity=2)
    built.append(Span(truncated))
    built.append("AB".as_bytes())
    var col = built^.finish()
    assert_equal(len(rows(text_casefold(col))[0].as_bytes()), 2)
    assert_equal(rows(text_casefold(col))[1], "ab")


def titled(var values: List[String]) raises -> List[String]:
    """Titles a column built from a list and reads it back.

    Args:
        values: The values.

    Returns:
        One string per row.

    Raises:
        Error: If the column cannot be built.
    """
    return rows(text_title(made(values^)))


def test_titling_raises_the_first_character_of_every_word() raises:
    var out = titled(["hello world", "ABC DEF", "a b  c"])
    assert_equal(out[0], "Hello World")
    assert_equal(out[1], "Abc Def")
    assert_equal(out[2], "A B  C")


def test_a_word_starts_after_anything_that_is_in_no_case() raises:
    # A word does not end at whitespace, it ends at any character in no case at
    # all, so an apostrophe and a digit both start a new word and the letter
    # after them is raised. That is the rule pandas has and it surprises people.
    var out = titled(["don't", "abc1def", "_ab"])
    assert_equal(out[0], "Don'T")
    assert_equal(out[1], "Abc1Def")
    assert_equal(out[2], "_Ab")


def test_titling_raises_a_titlecase_character_all_the_way() raises:
    # The obvious guess is that a digraph at the start of a word becomes the
    # titlecase form, and it does not. Arrow's titlecase mapping is its upper
    # case mapping everywhere, so the whole capital is what comes out, and the
    # second digraph is inside the word and drops instead.
    var out = titled(["ǆx", "ǅ", "ǄǄ"])
    assert_equal(out[0], "Ǆx")
    assert_equal(out[1], "Ǆ")
    assert_equal(out[2], "Ǆǆ")


def test_titling_corrects_the_same_code_points_the_other_names_do() raises:
    # A sharp s inside a word stays a sharp s because lowering leaves it alone,
    # and one at the start becomes the capital the correction table carries
    # rather than the two letters the standard library would give.
    var out = titled(["straße", "ßx"])
    assert_equal(out[0], "Straße")
    assert_equal(out[1], "ẞx")


def test_titling_an_empty_row_and_a_missing_row() raises:
    var out = titled(["", "null", "çA"])
    assert_equal(out[0], "")
    assert_equal(out[1], "null")
    assert_equal(out[2], "Ça")


def test_titling_leaves_bytes_that_are_not_utf8_alone() raises:
    var truncated = List[UInt8]()
    truncated.append(0x61)
    truncated.append(0xC4)
    var built = StringBuilder(capacity=2)
    built.append(Span(truncated))
    built.append("ab cd".as_bytes())
    var col = built^.finish()
    assert_equal(len(rows(text_title(col))[0].as_bytes()), 2)
    assert_equal(rows(text_title(col))[1], "Ab Cd")


def test_the_title_question_is_asked_of_every_word() raises:
    var out = asked(text_is_title(made(["Hello World", "Hello world", "A "])))
    assert_equal(out[0], "yes")
    assert_equal(out[1], "no")
    assert_equal(out[2], "yes")


def test_a_row_with_no_cased_character_is_not_titled() raises:
    var out = asked(text_is_title(made(["1", "", " ", "Abc Def"])))
    assert_equal(out[0], "no")
    assert_equal(out[1], "no")
    assert_equal(out[2], "no")
    assert_equal(out[3], "yes")


def test_the_title_question_counts_a_digit_as_a_word_break() raises:
    # `A1b` is not titled, because the digit ends the word and the `b` after it
    # is the start of a new one and is not raised. `A1B` is titled.
    var out = asked(text_is_title(made(["A1b", "A1B", "Don'T", "Don't"])))
    assert_equal(out[0], "no")
    assert_equal(out[1], "yes")
    assert_equal(out[2], "yes")
    assert_equal(out[3], "no")


def test_a_titlecase_character_starts_a_word_and_does_not_continue_one() raises:
    var out = asked(text_is_title(made(["ǅx", "ǅX", "ǅ", "Ǆ"])))
    assert_equal(out[0], "yes")
    assert_equal(out[1], "no")
    assert_equal(out[2], "yes")
    assert_equal(out[3], "yes")


def test_the_title_question_keeps_a_missing_row_missing() raises:
    var out = asked(text_is_title(made(["Ab", "null"])))
    assert_equal(out[0], "yes")
    assert_equal(out[1], "null")


def test_the_ascii_question_reads_bytes_and_not_characters() raises:
    var out = asked(text_is_ascii(made(["abc", "café", "~", one(0x0080)])))
    assert_equal(out[0], "yes")
    assert_equal(out[1], "no")
    assert_equal(out[2], "yes")
    assert_equal(out[3], "no")


def test_the_ascii_question_is_the_one_that_says_yes_to_an_empty_row() raises:
    # Every other question about a row wants a character to answer yes. This one
    # is about what a row does not contain, so a row containing nothing passes.
    var out = asked(text_is_ascii(made(["", "null", "x"])))
    assert_equal(out[0], "yes")
    assert_equal(out[1], "null")
    assert_equal(out[2], "yes")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
