"""The ten `str` methods that are about case, checked against pandas.

Six of them write a row out in some case or other and four of them ask a
question about the case a row is already in. They are one group because they all
rest on the same piece of data, which is the table that says what the other case
of a character is, and because the interesting rows are the same rows for all
ten. `isascii` is here too, which is not about case at all and is here because it
is the eleventh name the accessor gained in the same week and because it is the
one question in the group that a row of nothing answers yes to.

Which table that is turns out to matter more than anything else here. pandas 3
holds a text column in Arrow and answers `upper` and `lower` out of an Arrow
kernel, which uses the simple case mappings, so a row is never longer coming out
than it was going in. The same column held as object goes through Python's own
string methods, which use the full mappings, and the two disagree: a sharp s
raises to one letter in the first and two in the second, and a Turkish capital I
keeps its dot in the second and loses it in the first. So these tests compare
against the default dtype, which is what a caller gets without asking, and the
test at the end of the file writes both pandas answers out side by side so that
the choice is visible rather than implied.

The three that ask a question used to have a test asserting three differences
rather than working around them, because they answered out of the standard
library's older copy of the Unicode data and differed from Arrow on 1384 code
points. They do not any more. `charclass.mojo` carries Arrow's classes and the
test that named those three rows is now a sweep of every code point there is
with nothing left over.

`capitalize` and `swapcase` need nothing beyond the correction table and are
exact, which was measured rather than hoped for: both were run against pandas
over every code point in Unicode on its own and over sixty thousand random
words, with no row differing. `title` and `istitle` are the two names here that
need something the tables do not hold, which is where a word starts, and that is
a question about what comes before a character rather than about the character.
The answer turns out to be that a word starts after any character in no case at
all, so an apostrophe and a digit both begin a new one, and the rule is checked
here over the same sweep the rest of the group gets.

`casefold` is the exception to everything the paragraph above says about which
pandas to follow, and it is pandas' exception rather than this library's. pyarrow
has no casefold kernel, so a pandas text column falls back to Python for that one
method whatever dtype it is held as, both backends agree, and the full mappings
are the right answer. It is the only name here that can give back a row longer
than the row it was given, and the tests for it are the only ones in this file
that do not have to name a dtype to be meaningful.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ROWS = [
    "hello",
    "HÉLLO",
    "Mixed Case",
    "café",
    "straße",
    "İstanbul",
    "\u0131stanbul",
    "",
    "42",
    None,
]
"""Ten rows, each of them there to catch a different way of being wrong.

The second and the fourth are the rows where a byte walk would change the letter
and leave the accent, the fifth is the row where Python and Arrow disagree about
how many letters the answer has, and the sixth and the seventh are the Turkish
pair, which is the one place in a Latin alphabet where the two cases are not a
pair at all. The empty string and the row of digits are the two rows where a
question about case has no cased character to answer about, and the None is here
because a case change on a missing row is a missing row rather than an empty
string.
"""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds it by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def like(mine: list[Any], them: list[Any]) -> bool:
    """Compares two columns of values, reading a NaN as a None."""
    if len(mine) != len(them):
        return False
    for one, other in zip(mine, them, strict=True):
        if one is None:
            if other is None or other != other:
                continue
            return False
        if one != other:
            return False
    return True


@needs_pandas
def test_upper_matches_pandas_row_for_row(firepanda: ModuleType) -> None:
    """Every row, including the sharp s and the two Turkish ones."""
    assert like(made(firepanda).str.upper().tolist(), theirs().str.upper().tolist())


@needs_pandas
def test_lower_matches_pandas_row_for_row(firepanda: ModuleType) -> None:
    """The same ten rows the other way, which is the harder direction of the two."""
    assert like(made(firepanda).str.lower().tolist(), theirs().str.lower().tolist())


@needs_pandas
def test_the_two_pandas_string_backends_do_not_agree_with_each_other(
    firepanda: ModuleType,
) -> None:
    """Which is why the tests above name a dtype, and this is the name they chose.

    Held as pandas holds it by default the answer comes out of Arrow, and held
    as object it comes out of Python. This library follows the first, because
    that is what a caller gets from `pd.Series([...])` without asking for
    anything, and because a case change that can make a row longer is a
    different kind of operation from one that cannot.
    """
    import pandas as pd

    rows = ["straße", "İstanbul"]
    assert pd.Series(rows, dtype="str").str.upper().tolist() == ["STRAẞE", "İSTANBUL"]
    assert pd.Series(rows, dtype="object").str.upper().tolist() == ["STRASSE", "İSTANBUL"]
    assert pd.Series(rows, dtype="str").str.lower().tolist() == ["straße", "istanbul"]
    assert pd.Series(rows, dtype="object").str.lower().tolist() == [
        "straße",
        "i̇stanbul",
    ]
    assert made(firepanda, rows).str.upper().tolist() == ["STRAẞE", "İSTANBUL"]
    assert made(firepanda, rows).str.lower().tolist() == ["straße", "istanbul"]


def test_a_row_keeps_its_length_through_a_case_change(firepanda: ModuleType) -> None:
    """The simple mappings are one character in and one character out, always."""
    for row in ("straße", "İstanbul", "ﬁance", "ŉ"):
        assert len(made(firepanda, [row]).str.upper().tolist()[0]) == len(row)
        assert len(made(firepanda, [row]).str.lower().tolist()[0]) == len(row)


def test_changing_case_twice_does_not_come_back(firepanda: ModuleType) -> None:
    """Which is a fact about Unicode rather than about this library."""
    assert made(firepanda, ["İstanbul"]).str.lower().str.upper().tolist() == ["ISTANBUL"]


def test_a_case_change_keeps_a_missing_row_missing(firepanda: ModuleType) -> None:
    """And does not turn it into the empty string, which is the tempting mistake."""
    assert made(firepanda, ["a", None]).str.upper().tolist() == ["A", None]
    assert made(firepanda, ["A", None]).str.lower().tolist() == ["a", None]


@needs_pandas
def test_the_three_questions_match_pandas_row_for_row(firepanda: ModuleType) -> None:
    """On every row of the ten except the missing one, which the next test is about."""
    for name in ("isspace", "islower", "isupper"):
        mine = getattr(made(firepanda).str, name)().tolist()[:-1]
        assert mine == getattr(theirs().str, name)().tolist()[:-1], name


def test_a_row_with_no_cased_character_is_neither_lower_nor_upper(
    firepanda: ModuleType,
) -> None:
    """Python's rule, and the one that makes the two questions not opposites."""
    column = made(firepanda, ["42", "", "abc", "ABC"])
    assert column.str.islower().tolist() == [False, False, True, False]
    assert column.str.isupper().tolist() == [False, False, False, True]


def test_whitespace_is_asked_of_every_character(firepanda: ModuleType) -> None:
    """Including the empty row, where there is no character to ask about."""
    column = made(firepanda, [" ", " \t\n", "a b", "", "ab"])
    assert column.str.isspace().tolist() == [True, True, False, False, False]


@needs_pandas
def test_a_question_about_a_missing_row_is_missing_here_and_false_there(
    firepanda: ModuleType,
) -> None:
    """The asserted difference, which is `engine/string-predicate-null` in the registry.

    pandas holding the column the way it holds it by default has nowhere to put
    a missing answer, because the answer is a numpy array of bools, so a missing
    row comes back False and cannot be told from a row that was really not upper
    case. Held as object it comes back None, which is what this library answers
    whatever the column is made of.
    """
    import pandas as pd

    assert made(firepanda, ["A", None]).str.isupper().tolist() == [True, None]
    assert pd.Series(["A", None], dtype="str").str.isupper().tolist() == [True, False]
    assert pd.Series(["A", None], dtype="object").str.isupper().tolist() == [True, None]


def test_a_question_answers_a_column_of_bools(firepanda: ModuleType) -> None:
    """The shape of the answer, which is what picks the door these three come through."""
    assert str(made(firepanda, ["a"]).str.isupper().dtype) == "bool"
    assert str(made(firepanda, ["a"]).str.upper().dtype) == "string"


def test_the_accessor_refuses_a_column_that_is_not_text(firepanda: ModuleType) -> None:
    """At the accessor rather than at the method, which is where pandas refuses it."""
    with pytest.raises(AttributeError):
        firepanda.Series([1, 2, 3]).str.upper()


@needs_pandas
def test_the_three_questions_agree_on_every_code_point_there_is(
    firepanda: ModuleType,
) -> None:
    """The measurement that used to be a list of differences, run as a test.

    These three were answered out of the Mojo standard library's character data
    until the classes in `charclass.mojo` replaced it, and that data disagreed
    with Arrow about 1384 code points across the three. The rows that
    disagreement was most likely to be met on had their own test here, written
    to fail the day the data was replaced. It was, so the test is this one
    instead: every code point in Unicode, one to a row, through both sides.

    A million rows through either library is a fraction of a second, so there is
    no reason to assert a sample of something that can be asserted whole.
    """
    rows = [chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    mine, them = made(firepanda, rows), theirs(rows)
    for name in ("isspace", "islower", "isupper"):
        assert getattr(mine.str, name)().tolist() == getattr(them.str, name)().tolist(), name


@needs_pandas
def test_the_three_rows_that_used_to_be_the_measured_differences(
    firepanda: ModuleType,
) -> None:
    """The non breaking space, the kra and the double struck capital C.

    One row for each question, kept from the test that came before because they
    are the rows a caller is most likely to meet and because a sweep that fails
    names a code point rather than a reason.
    """
    assert made(firepanda, ["\u00a0"]).str.isspace().tolist() == [True]
    assert theirs(["\u00a0"]).str.isspace().tolist() == [True]
    assert made(firepanda, ["\u0138"]).str.islower().tolist() == [True]
    assert theirs(["\u0138"]).str.islower().tolist() == [True]
    assert made(firepanda, ["\u2102"]).str.isupper().tolist() == [True]
    assert theirs(["\u2102"]).str.isupper().tolist() == [True]


@needs_pandas
def test_a_titlecase_character_is_neither_lower_nor_upper(
    firepanda: ModuleType,
) -> None:
    """The third case, which is why the two questions are not opposites twice over.

    A row of digits is neither because it has no cased character in it at all. A
    row holding one of these is neither for a different reason, which is that
    the character is cased and is in a case that is neither of the two being
    asked about, and it carries that answer to any row it sits in.
    """
    rows = ["\u01c5", "\u01c5a", "a\u01c5"]
    assert made(firepanda, rows).str.islower().tolist() == [False, False, False]
    assert made(firepanda, rows).str.isupper().tolist() == [False, False, False]
    assert theirs(rows).str.islower().tolist() == [False, False, False]
    assert theirs(rows).str.isupper().tolist() == [False, False, False]


@needs_pandas
def test_a_letter_that_looks_lower_case_and_is_in_no_case_at_all(
    firepanda: ModuleType,
) -> None:
    """The feminine ordinal and a modifier letter, which are letters and are not cased.

    They behave in a row exactly as a digit does, which is to answer no on their
    own and to leave the row around them free to answer yes.
    """
    rows = ["\u00aa", "\u1d43", "\u00aaa"]
    assert made(firepanda, rows).str.islower().tolist() == [False, False, True]
    assert theirs(rows).str.islower().tolist() == [False, False, True]


@needs_pandas
def test_the_spaces_nobody_lists_are_spaces_too(firepanda: ModuleType) -> None:
    """Four separators and the non breaking space, none of them the six obvious ones.

    Arrow counts 29 code points as spaces and so does Python, exactly the same
    29, which is worth knowing because it means the disagreement this class
    table was built to settle was never between those two. It was the Mojo
    standard library that had the shorter list, and the four ASCII separators at
    U+001C to U+001F were nowhere on it.
    """
    rows = ["\u001c", "\u001d", "\u001e", "\u001f", "\u00a0"]
    assert made(firepanda, rows).str.isspace().tolist() == [True] * 5
    assert theirs(rows).str.isspace().tolist() == [True] * 5


@needs_pandas
def test_capitalize_matches_pandas_row_for_row(firepanda: ModuleType) -> None:
    """The same ten rows, including the two where Arrow and Python disagree."""
    assert like(made(firepanda).str.capitalize().tolist(), theirs().str.capitalize().tolist())


@needs_pandas
def test_swapcase_matches_pandas_row_for_row(firepanda: ModuleType) -> None:
    """And the same ten again, the only one of the seven with no library call under it."""
    assert like(made(firepanda).str.swapcase().tolist(), theirs().str.swapcase().tolist())


def test_capitalize_drops_everything_after_the_first_character(
    firepanda: ModuleType,
) -> None:
    """Which is the thing about this method that surprises people."""
    column = made(firepanda, ["hello world", "HELLO WORLD", "hello WORLD"])
    assert column.str.capitalize().tolist() == ["Hello world"] * 3


def test_capitalize_starts_at_the_first_character_and_not_the_first_letter(
    firepanda: ModuleType,
) -> None:
    """A row that opens with a digit or a space keeps its first letter lower."""
    column = made(firepanda, ["1abc def", "  spaced", "", None])
    assert column.str.capitalize().tolist() == ["1abc def", "  spaced", "", None]


def test_swapcase_leaves_a_character_in_neither_case_alone(
    firepanda: ModuleType,
) -> None:
    """Digits and punctuation, and the titlecase characters, which look like capitals."""
    column = made(firepanda, ["1 2 !?", "\u01c5ungla", "\u1f88\u03b1"])
    assert column.str.swapcase().tolist() == ["1 2 !?", "\u01c5UNGLA", "\u1f88\u0391"]


@needs_pandas
def test_the_titlecase_rule_is_what_pandas_does_and_not_an_invention(
    firepanda: ModuleType,
) -> None:
    """Both pandas backends agree here, which is worth pinning since so few rows do."""
    import pandas as pd

    rows = ["\u01c5ungla"]
    assert pd.Series(rows, dtype="str").str.swapcase().tolist() == ["\u01c5UNGLA"]
    assert pd.Series(rows, dtype="object").str.swapcase().tolist() == ["\u01c5UNGLA"]
    assert made(firepanda, rows).str.swapcase().tolist() == ["\u01c5UNGLA"]


def test_swapping_twice_gives_the_row_back(firepanda: ModuleType) -> None:
    """True for these rows and not for every row, which is why the next test exists."""
    for row in ("Hello World", "o'neill", "MiXeD 42"):
        assert made(firepanda, [row]).str.swapcase().str.swapcase().tolist() == [row]


def test_swapping_twice_does_not_always_give_the_row_back(
    firepanda: ModuleType,
) -> None:
    """A Turkish capital I loses its dot on the way down and does not get it back.

    A sharp s does come back, which is worth having next to it: it swaps up to a
    capital sharp s and down again to the small one it started as, because Arrow
    holds both halves of that pair. The Turkish letter has no pair to hold, since
    the lower case of a capital I with a dot is a plain i to Arrow, and a plain i
    raises to a plain I. Both are facts about Unicode rather than about this
    library, and both are what pandas answers.
    """
    assert made(firepanda, ["stra\u00dfe"]).str.swapcase().tolist() == ["STRA\u1e9eE"]
    assert made(firepanda, ["stra\u00dfe"]).str.swapcase().str.swapcase().tolist() == [
        "stra\u00dfe"
    ]
    assert made(firepanda, ["\u0130stanbul"]).str.swapcase().tolist() == ["iSTANBUL"]
    assert made(firepanda, ["\u0130stanbul"]).str.swapcase().str.swapcase().tolist() == ["Istanbul"]


def test_the_two_new_names_keep_a_missing_row_missing(firepanda: ModuleType) -> None:
    """The same rule as the other five, asserted again because it is easy to lose."""
    assert made(firepanda, ["a", None]).str.capitalize().tolist() == ["A", None]
    assert made(firepanda, ["a", None]).str.swapcase().tolist() == ["A", None]


@needs_pandas
def test_casefold_matches_pandas_row_for_row(firepanda: ModuleType) -> None:
    """All ten rows, and the sharp s in the middle of them is the point."""
    assert like(made(firepanda).str.casefold().tolist(), theirs().str.casefold().tolist())


@needs_pandas
def test_casefold_is_the_one_name_here_both_pandas_backends_agree_on(
    firepanda: ModuleType,
) -> None:
    """Because neither of them is Arrow. pyarrow has no kernel for this method.

    Every other rewrite in this file answers differently depending on how the
    column is held, which is what the test above this group is about. This one
    cannot, since both backends end up in the same Python function, and that is
    why the full mappings are right here and wrong three names over.
    """
    import pandas as pd

    rows = ["stra\u00dfe", "\u0130stanbul", "\ufb01ance"]
    out = ["strasse", "i\u0307stanbul", "fiance"]
    assert pd.Series(rows, dtype="str").str.casefold().tolist() == out
    assert pd.Series(rows, dtype="object").str.casefold().tolist() == out
    assert made(firepanda, rows).str.casefold().tolist() == out


def test_folding_makes_a_row_longer_where_lowering_leaves_it_alone(
    firepanda: ModuleType,
) -> None:
    """The difference between the two names, in one row.

    Lowering a sharp s leaves it as it is, because it is already the lower case
    one and Arrow has no mapping that would make it anything else. Folding it
    writes two letters, because the question folding answers is which rows a
    reader would call the same rather than what the row looks like in lower case.
    """
    assert made(firepanda, ["stra\u00dfe"]).str.lower().tolist() == ["stra\u00dfe"]
    assert made(firepanda, ["stra\u00dfe"]).str.casefold().tolist() == ["strasse"]


def test_two_rows_a_reader_calls_equal_fold_to_the_same_bytes(
    firepanda: ModuleType,
) -> None:
    """The whole reason the name exists, and the reason lowering cannot replace it."""
    folded = made(firepanda, ["Stra\u00dfe", "STRASSE"]).str.casefold().tolist()
    assert folded[0] == folded[1]
    lowered = made(firepanda, ["Stra\u00dfe", "STRASSE"]).str.lower().tolist()
    assert lowered[0] != lowered[1]


def test_folding_takes_a_titlecase_character_all_the_way_down(
    firepanda: ModuleType,
) -> None:
    """Where `swapcase` leaves the same character exactly as it found it.

    A titlecase character is in neither case, which is why swapping has nothing
    to do to it. Folding is not about a case at all, so all three of the letters
    in that family come out as the same one.
    """
    rows = ["\u01c4", "\u01c5", "\u01c6"]
    assert made(firepanda, rows).str.casefold().tolist() == ["\u01c6"] * 3
    assert made(firepanda, ["\u01c5"]).str.swapcase().tolist() == ["\u01c5"]


@needs_pandas
def test_folding_corrects_the_code_points_the_lower_case_path_corrects(
    firepanda: ModuleType,
) -> None:
    """The fold table is a difference table, so most characters fall through it.

    A capital sharp s is in the fold table because it folds to two letters. A
    capital theta symbol is not, because folding it is lowering it, so it is
    answered by the same corrected lower case path `str.lower` uses, and the
    standard library underneath gets that particular character wrong on its own.
    """
    import pandas as pd

    rows = ["\u1e9e", "\u03f4"]
    out = ["ss", "\u03b8"]
    assert pd.Series(rows, dtype="str").str.casefold().tolist() == out
    assert made(firepanda, rows).str.casefold().tolist() == out


def test_casefold_keeps_a_missing_row_missing(firepanda: ModuleType) -> None:
    """The same rule as the other seven, asserted again because it is easy to lose."""
    assert made(firepanda, ["A", None]).str.casefold().tolist() == ["a", None]


@needs_pandas
def test_title_matches_pandas_row_for_row(firepanda: ModuleType) -> None:
    """The same ten rows again, with the Turkish pair the interesting part.

    A capital I with a dot at the start of a row is already the raised form and
    stays put, and the dotless i beside it raises to a plain capital, which is
    the pair that makes a case change not a round trip.
    """
    assert like(made(firepanda).str.title().tolist(), theirs().str.title().tolist())


@needs_pandas
def test_a_word_starts_after_anything_that_is_in_no_case(firepanda: ModuleType) -> None:
    """Not at whitespace, which is the thing about this method that surprises people.

    A word ends at the first character that is in no case at all, so a digit and
    an apostrophe and an underscore each start a new one and the letter after
    them is raised. `don't` comes out as `Don'T` and that is pandas' answer as
    much as it is this library's.
    """
    rows = ["don't", "abc1def", "_ab", "a b  c"]
    out = ["Don'T", "Abc1Def", "_Ab", "A B  C"]
    assert made(firepanda, rows).str.title().tolist() == out
    assert theirs(rows).str.title().tolist() == out


@needs_pandas
def test_titling_raises_a_digraph_all_the_way_and_not_to_the_titlecase_form(
    firepanda: ModuleType,
) -> None:
    """The guess the name invites, and it is wrong, and it is wrong in pandas too.

    Arrow's titlecase mapping is its upper case mapping for every code point in
    Unicode, so a Croatian digraph at the start of a word becomes the whole
    capital rather than the titlecase form that exists for exactly this purpose.
    The second one in the third row is inside a word and drops instead.
    """
    rows = ["ǆx", "ǅ", "ǄǄ"]
    out = ["Ǆx", "Ǆ", "Ǆǆ"]
    assert made(firepanda, rows).str.title().tolist() == out
    assert theirs(rows).str.title().tolist() == out


@needs_pandas
def test_title_and_istitle_agree_on_every_code_point_there_is(
    firepanda: ModuleType,
) -> None:
    """Both of them, one code point to a row, with nothing left over.

    A single character is a whole word, so this sweep is the part of the rule
    that is about the character rather than about what comes before it. The test
    after this one is the other half.
    """
    rows = [chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    mine, them = made(firepanda, rows), theirs(rows)
    assert mine.str.title().tolist() == them.str.title().tolist()
    assert mine.str.istitle().tolist() == them.str.istitle().tolist()


@needs_pandas
def test_title_and_istitle_agree_on_words_built_to_break_them(
    firepanda: ModuleType,
) -> None:
    """Where a word starts is the half a sweep of single characters cannot reach.

    The alphabet here is chosen rather than random: both cases of a letter, a
    titlecase digraph, a letter in no case at all, a digit, an apostrophe and a
    space, which are the seven characters this rule has anything different to
    say about. Every arrangement of four of them is 2401 rows.
    """
    from itertools import product

    alphabet = ["a", "A", "ǅ", "ª", "1", "'", " "]
    rows = ["".join(word) for word in product(alphabet, repeat=4)]
    mine, them = made(firepanda, rows), theirs(rows)
    assert mine.str.title().tolist() == them.str.title().tolist()
    assert mine.str.istitle().tolist() == them.str.istitle().tolist()


def test_a_row_with_no_cased_character_is_not_titled(firepanda: ModuleType) -> None:
    """The same rule the other two case questions have, and the empty row falls under it."""
    column = made(firepanda, ["1", "", " ", "Abc Def"])
    assert column.str.istitle().tolist() == [False, False, False, True]


@needs_pandas
def test_isascii_reads_bytes_and_not_characters(firepanda: ModuleType) -> None:
    """The one question in this file that needs no table, and Arrow has no kernel for it.

    pyarrow has no `utf8_is_ascii`, so pandas answers this one somewhere else,
    and the answer is the same either way because there is nothing to disagree
    about. A row is ASCII when none of its bytes has the top bit set, which is a
    pass over the bytes with no decoding at all.
    """
    rows = ["abc", "café", "~", "\t"]
    out = [True, False, True, True]
    assert made(firepanda, rows).str.isascii().tolist() == out
    assert theirs(rows).str.isascii().tolist() == out


@needs_pandas
def test_isascii_is_the_one_question_an_empty_row_answers_yes_to(
    firepanda: ModuleType,
) -> None:
    """Every other question here wants a character before it will say yes, and this one does not.

    It is a question about what a row does not contain, so a row containing
    nothing passes it. pandas answers the same, which is worth pinning because
    the four questions next to it all answer the other way.
    """
    assert made(firepanda, [""]).str.isascii().tolist() == [True]
    assert theirs([""]).str.isascii().tolist() == [True]
    for name in ("isspace", "islower", "isupper", "istitle"):
        assert getattr(made(firepanda, [""]).str, name)().tolist() == [False], name


@needs_pandas
def test_isascii_agrees_on_every_code_point_there_is(firepanda: ModuleType) -> None:
    """Which is a sweep of one comparison, and cheap enough to run whole anyway."""
    rows = [chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    assert made(firepanda, rows).str.isascii().tolist() == theirs(rows).str.isascii().tolist()


def test_the_three_new_names_keep_a_missing_row_missing(firepanda: ModuleType) -> None:
    """Including `isascii`, which says yes to an empty row and still not to a missing one."""
    assert made(firepanda, ["a b", None]).str.title().tolist() == ["A B", None]
    assert made(firepanda, ["Ab", None]).str.istitle().tolist() == [True, None]
    assert made(firepanda, ["ab", None]).str.isascii().tolist() == [True, None]
