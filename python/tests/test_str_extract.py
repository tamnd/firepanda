"""`str.extract`, checked against pandas.

The third answer shape on this accessor that no single column can hold, after the three columns of
`partition` and the however many of `get_dummies`, and the first whose width is read off the pattern
rather than off the data. Two groups answer two columns whatever the column holds, including a
column where nothing matched at all.

Four things are worth knowing before reading the assertions.

The match is the leftmost one anywhere in the row and not one anchored at the front, because pandas
runs `regex.search` here and `regex.match` for `str.match`. A pattern with no anchor in it therefore
finds its match in the middle of a row, which is the opposite of what the three mask methods on this
accessor do with the same pattern.

A row with no match is missing in every column, and a group that took no part in a match that did
happen is missing on its own. Those two look like one rule until a row disagrees with itself, which
is what `(a)(x)?` does to a row holding `a`, and they are the reason the answer is not a match flag
repeated across the width.

The class letters are read in Python's alphabet. `\\w` is 138558 code points here and 63 for
`str.count` on the same accessor, because this is one of the three names that never reach Arrow, and
the test at the bottom of this file is the one that would fail if the pattern had quietly been
compiled for the other engine.

There is one disagreement and it is the label one `partition` already carries. pandas labels an
unnamed group with its own position as an integer and a firepanda frame holds text column labels, so
a group that would be `1` upstream is `"1"` here. A named group is labelled with its name on both
sides, which is why the tests compare the labels through `str` rather than directly.

`flags` is the fifth thing worth knowing and it is the one argument on this accessor that carries no
route with it. Everywhere else a flags argument says both what the letters mean and that the call
has left Arrow, and the second of those is what costs a word at the door the call crosses by. This
name is on Python's engine whatever anybody passes, so the letters cross on their own. It is also
the one pattern method with no `case` argument, so it is the one where the two spellings of a fold
cannot disagree with each other.
"""

from __future__ import annotations

import importlib.util
import re
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ROWS = [
    "ab1",
    "zz",
    None,
    "",
    "xxc3",
    "d4e5",
    "ω7",
    "F8",
]
"""Eight rows, each of them there to catch a different way of being wrong.

The first matches at the front and the second matches nothing, which are the two ordinary answers.
The third is missing and the fourth is empty, which are two different kinds of nothing and reach the
same answer by different routes. The fifth matches in the middle, so an implementation that anchored
would answer nothing for it. The sixth holds two matches and only the first is wanted. The seventh
is a letter that is a word character to Python and not to RE2. The last is a capital, so a pattern
written in lowercase does not accidentally match it.
"""

PATTERNS = (
    r"([a-z])(\d)",
    r"([a-z])\d",
    r"(\w)(\d)",
    r"(?P<letter>[a-z])(\d)",
    r"(a)(x)?",
    r"(a*)b",
    r"(.)",
    r"( \w ) (\d)",
)
"""Eight patterns: two groups, one group, a class, a name, an optional group, an empty match, a
group that matches anything, and one written with spaces in it so that verbose mode has something
to throw away and every other flag has a pattern that needs the spaces matched."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds it by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def cell(value: Any) -> Any:
    """One value with every spelling of missing written the same way.

    pandas answers `nan` for a group that has nothing in it and firepanda answers `None`, and
    neither of those is a fact about `str.extract`, so the comparisons below go through here.
    """
    if value is None or value != value:
        return None
    return value


def mine_rows(frame: Any) -> list[list[Any]]:
    """A firepanda frame read out row by row."""
    labels = list(frame.columns)
    return [[cell(frame[label][i]) for label in labels] for i in range(frame.shape[0])]


def their_rows(frame: Any) -> list[list[Any]]:
    """A pandas frame read out the same way."""
    return [[cell(value) for value in row] for row in frame.values.tolist()]


@needs_pandas
def test_the_columns_match_pandas_on_every_pattern(firepanda: ModuleType) -> None:
    """Same values under the same labels, for seven patterns over eight rows."""
    for pattern in PATTERNS:
        assert mine_rows(made(firepanda).str.extract(pattern)) == their_rows(
            theirs().str.extract(pattern)
        )


@needs_pandas
def test_the_labels_match_pandas_once_an_integer_is_written_out(firepanda: ModuleType) -> None:
    """An unnamed group is labelled with its own position, which pandas writes as an integer.

    This is the same divergence `partition` carries and not a new one, so the comparison goes
    through `str` rather than being written out as an expected list.
    """
    for pattern in PATTERNS:
        assert list(made(firepanda).str.extract(pattern).columns) == [
            str(label) for label in theirs().str.extract(pattern).columns
        ]


@needs_pandas
def test_a_named_group_is_labelled_with_its_name(firepanda: ModuleType) -> None:
    """Which is the same label on both sides, because a name is already text."""
    pattern = r"(?P<letter>[a-z])(?P<digit>\d)"
    assert list(made(firepanda).str.extract(pattern).columns) == ["letter", "digit"]
    assert list(theirs().str.extract(pattern).columns) == ["letter", "digit"]


@needs_pandas
def test_a_name_and_a_position_can_sit_side_by_side(firepanda: ModuleType) -> None:
    """Naming one group of two leaves the other labelled by its position and not by its order.

    The second group is the second one opened and is labelled 1, counting from zero, which is a
    position in the answer rather than a group number.
    """
    pattern = r"(?P<letter>[a-z])(\d)"
    assert list(made(firepanda).str.extract(pattern).columns) == ["letter", "1"]
    assert list(theirs().str.extract(pattern).columns) == ["letter", 1]


@needs_pandas
def test_the_match_is_looked_for_anywhere_rather_than_at_the_front(firepanda: ModuleType) -> None:
    """`extract` searches where `match` anchors, on the same accessor and with the same pattern."""
    rows = ["xxa1"]
    assert mine_rows(made(firepanda, rows).str.extract(r"([a-z])(\d)")) == [["a", "1"]]
    assert their_rows(theirs(rows).str.extract(r"([a-z])(\d)")) == [["a", "1"]]
    assert made(firepanda, rows).str.match(r"([a-z])(\d)")[0] is False


@needs_pandas
def test_a_row_with_no_match_is_missing_in_every_column(firepanda: ModuleType) -> None:
    """The columns of one row agree about whether there was a match at all."""
    rows = ["a1", "zz"]
    assert mine_rows(made(firepanda, rows).str.extract(r"([a-z])(\d)")) == [
        ["a", "1"],
        [None, None],
    ]
    assert their_rows(theirs(rows).str.extract(r"([a-z])(\d)")) == [["a", "1"], [None, None]]


@needs_pandas
def test_a_group_that_took_no_part_is_missing_on_its_own(firepanda: ModuleType) -> None:
    """The one case where the columns of a row disagree, and the reason the rule above is not a
    match flag repeated across the width."""
    rows = ["a", "ax"]
    assert mine_rows(made(firepanda, rows).str.extract(r"(a)(x)?")) == [["a", None], ["a", "x"]]
    assert their_rows(theirs(rows).str.extract(r"(a)(x)?")) == [["a", None], ["a", "x"]]


@needs_pandas
def test_a_group_can_hold_nothing_without_being_missing(firepanda: ModuleType) -> None:
    """A group that matched the empty string is not a group that was left out."""
    rows = ["b", "ab"]
    assert mine_rows(made(firepanda, rows).str.extract(r"(a*)b")) == [[""], ["a"]]
    assert their_rows(theirs(rows).str.extract(r"(a*)b")) == [[""], ["a"]]


@needs_pandas
def test_a_missing_row_is_missing_in_every_column(firepanda: ModuleType) -> None:
    """The same answer a row with no match gets, arrived at without reading any text."""
    rows = ["a1", None]
    assert mine_rows(made(firepanda, rows).str.extract(r"([a-z])(\d)")) == [
        ["a", "1"],
        [None, None],
    ]
    assert their_rows(theirs(rows).str.extract(r"([a-z])(\d)")) == [["a", "1"], [None, None]]


@needs_pandas
def test_one_group_answers_a_frame_by_default(firepanda: ModuleType) -> None:
    """`expand` defaults to True, so the narrow answer is the one a caller has to ask for."""
    frame = made(firepanda).str.extract(r"([a-z])\d")
    assert frame.shape == (8, 1)
    assert theirs().str.extract(r"([a-z])\d").shape == (8, 1)


@needs_pandas
def test_one_group_with_expand_false_answers_a_column(firepanda: ModuleType) -> None:
    """The only shape on this accessor a caller can pick with an argument."""
    mine = made(firepanda).str.extract(r"([a-z])\d", expand=False)
    assert [cell(value) for value in mine.tolist()] == [
        cell(value) for value in theirs().str.extract(r"([a-z])\d", expand=False).tolist()
    ]


@needs_pandas
def test_the_column_is_named_after_the_group_when_the_group_has_a_name(
    firepanda: ModuleType,
) -> None:
    """Which is the one place on this accessor the answer's name is not the name it was called on.

    A group with no name leaves the name alone rather than clearing it, so the two calls below on
    the same column answer two different names.
    """
    column = firepanda.Series(ROWS, name="words")
    assert column.str.extract(r"(?P<letter>[a-z])\d", expand=False).name == "letter"
    assert column.str.extract(r"([a-z])\d", expand=False).name == "words"

    import pandas as pd

    theirs_named = pd.Series(ROWS, dtype="str", name="words")
    assert theirs_named.str.extract(r"(?P<letter>[a-z])\d", expand=False).name == "letter"
    assert theirs_named.str.extract(r"([a-z])\d", expand=False).name == "words"


@needs_pandas
def test_two_groups_answer_a_frame_whatever_expand_says(firepanda: ModuleType) -> None:
    """There is nowhere to put the second group, so `expand=False` is read and has no effect."""
    frame = made(firepanda).str.extract(r"([a-z])(\d)", expand=False)
    assert frame.shape == (8, 2)
    assert theirs().str.extract(r"([a-z])(\d)", expand=False).shape == (8, 2)


@needs_pandas
def test_the_class_is_read_in_pythons_alphabet_and_not_arrows(firepanda: ModuleType) -> None:
    """The one assertion here about which engine ran.

    `str.count` reads `\\w` as 63 characters of ASCII because it goes to Arrow, and this method
    reads it as 138558 code points because it does not go to Arrow at all. A row of Greek matches
    for one reading and not for the other, on the same accessor and in the same session.
    """
    rows = ["ωx", "-x"]
    assert mine_rows(made(firepanda, rows).str.extract(r"(\w)x")) == [["ω"], [None]]
    assert their_rows(theirs(rows).str.extract(r"(\w)x")) == [["ω"], [None]]
    assert made(firepanda, ["ω"]).str.count(r"\w").tolist() == [0]
    assert theirs(["ω"]).str.count(r"\w").tolist() == [0]


@needs_pandas
def test_a_group_can_hold_more_than_one_byte_a_character(firepanda: ModuleType) -> None:
    """The engine walks characters and the answer is cut in bytes, so a row outside ASCII is where
    an offset that was not converted shows up as the wrong half of the row. Every character below is
    three bytes wide."""
    rows = ["日日本語"]
    assert mine_rows(made(firepanda, rows).str.extract(r"(日+)(本)")) == [["日日", "本"]]
    assert their_rows(theirs(rows).str.extract(r"(日+)(本)")) == [["日日", "本"]]


@needs_pandas
def test_a_pattern_with_no_groups_is_refused_in_pandas_own_sentence(firepanda: ModuleType) -> None:
    """Upstream refuses this after compiling the pattern and so does this library."""
    with pytest.raises(ValueError, match="pattern contains no capture groups"):
        made(firepanda).str.extract(r"[a-z]")

    import pandas as pd

    with pytest.raises(ValueError, match="pattern contains no capture groups"):
        theirs().str.extract(r"[a-z]")
    assert pd is not None


@needs_pandas
def test_expand_is_checked_before_the_pattern_is(firepanda: ModuleType) -> None:
    """A caller who wrote `expand=None` and a pattern with no groups is told about `expand`, which
    is the order pandas checks the two in."""
    with pytest.raises(ValueError, match="expand must be True or False"):
        made(firepanda).str.extract(r"[a-z]", expand=None)
    with pytest.raises(ValueError, match="expand must be True or False"):
        theirs().str.extract(r"[a-z]", expand=None)


FLAGGED_ROWS = ["1\nab", "ab", "1", None, "a\nb1", "\u212a1", "F8", ""]
"""Eight rows chosen by running both libraries over candidates rather than from memory.

Each of the first three flags has a row here that answers one way without it and another way with
it, and those rows were found by sweeping rather than picked because they sound discriminating. The
first is the multiline row, where the front of the text is a digit and the front of the second line
is a letter. The fifth is the dot row, where the only thing between an `a` and a `b` is a newline.
The sixth and seventh are the fold rows, a Kelvin sign and a capital F.
"""

FLAGS = (
    0,
    re.IGNORECASE,
    re.MULTILINE,
    re.DOTALL,
    re.UNICODE,
    re.IGNORECASE | re.DOTALL,
    re.ASCII,
    re.IGNORECASE | re.ASCII,
    re.VERBOSE,
)
"""The six letters that go through, alone and in two pairs, with no flags at the front of the list
so that every sweep below also runs the unflagged call it is being compared against."""


def flagged(firepanda: ModuleType, pattern: str, flags: int = 0) -> list[list[Any]]:
    """One flagged extract, read out of a firepanda frame row by row."""
    return mine_rows(made(firepanda, FLAGGED_ROWS).str.extract(pattern, flags=flags))


def their_flagged(pattern: str, flags: int = 0) -> list[list[Any]]:
    """The same call on the same rows in pandas."""
    return their_rows(theirs(FLAGGED_ROWS).str.extract(pattern, flags=flags))


@needs_pandas
def test_every_pattern_under_every_flag_matches_pandas(firepanda: ModuleType) -> None:
    """Eight patterns under nine flag settings over eight rows, which is 576 cells.

    The sweep is the assertion that matters here. Each test below it names one flag and one row
    where that flag changes the answer, and those are worth reading, but they are three cells out
    of the same 336 and every one of them would pass against an implementation that read the flags
    for some patterns and dropped them for the rest.
    """
    for pattern in PATTERNS:
        for flags in FLAGS:
            assert flagged(firepanda, pattern, flags) == their_flagged(pattern, flags), (
                pattern,
                flags,
            )


@needs_pandas
def test_multiline_moves_the_front_of_the_row_to_the_front_of_a_line(
    firepanda: ModuleType,
) -> None:
    """`extract` runs `search` rather than `match`, so an anchor is the only way a flag about
    anchors can show at all. A row whose first line is a digit answers nothing for `^([a-z])` and
    answers the second line's first letter once the flag is beside it."""
    assert flagged(firepanda, r"^([a-z])")[0] == [None]
    assert flagged(firepanda, r"^([a-z])", re.MULTILINE)[0] == ["a"]
    assert their_flagged(r"^([a-z])")[0] == [None]
    assert their_flagged(r"^([a-z])", re.MULTILINE)[0] == ["a"]


@needs_pandas
def test_dotall_lets_the_dot_cover_the_newline(firepanda: ModuleType) -> None:
    """The one flag of the four whose effect is on a single character rather than on an anchor or
    on a table. `a.b` finds nothing in a row where the only thing between the two letters is a
    newline, and finds the whole of it with the flag."""
    assert flagged(firepanda, r"(a.b)")[4] == [None]
    assert flagged(firepanda, r"(a.b)", re.DOTALL)[4] == ["a\nb"]
    assert their_flagged(r"(a.b)")[4] == [None]
    assert their_flagged(r"(a.b)", re.DOTALL)[4] == ["a\nb"]


@needs_pandas
def test_ignorecase_folds_the_engines_way_and_there_is_no_other_way_here(
    firepanda: ModuleType,
) -> None:
    """A Kelvin sign is a `k` to this fold and a capital F is an `f`, and both of those are the
    engine's fold rather than the one to one one RE2 uses. That distinction decides nothing on this
    name, because there is no `case` argument here to reach the other fold by, which is what makes
    this the one pattern method where the two spellings cannot disagree."""
    assert flagged(firepanda, r"(k)")[5] == [None]
    assert flagged(firepanda, r"(k)", re.IGNORECASE)[5] == ["\u212a"]
    assert flagged(firepanda, r"([a-z])(\d)", re.IGNORECASE)[6] == ["F", "8"]
    assert their_flagged(r"(k)")[5] == [None]
    assert their_flagged(r"(k)", re.IGNORECASE)[5] == ["\u212a"]
    assert their_flagged(r"([a-z])(\d)", re.IGNORECASE)[6] == ["F", "8"]


@needs_pandas
def test_the_locale_flag_is_a_value_error_on_both_sides(firepanda: ModuleType) -> None:
    """Python turns `re.LOCALE` down on text and pandas hands text to `re`, so this is upstream's
    refusal rather than one of ours, and it is a `ValueError` here because it is one there."""
    with pytest.raises(ValueError):
        flagged(firepanda, r"([a-z])", re.LOCALE)
    with pytest.raises(ValueError):
        their_flagged(r"([a-z])", re.LOCALE)


@needs_pandas
def test_the_last_two_letters_answer_here_now(firepanda: ModuleType) -> None:
    """Verbose mode and the ascii flag were the two this method could not read, and they were the
    two nothing on the accessor could read. Each gets the row where it changes the answer: a
    pattern written with spaces in it under one, and a Kelvin sign that stops being a `k` under the
    other."""
    assert flagged(firepanda, r"( [a-z] )", re.VERBOSE)[1] == ["a"]
    assert flagged(firepanda, r"( [a-z] )")[1] == [None]
    assert flagged(firepanda, r"([a-z])", re.IGNORECASE)[5] == ["\u212a"]
    assert flagged(firepanda, r"([a-z])", re.IGNORECASE | re.ASCII)[5] == [None]
    assert their_flagged(r"( [a-z] )", re.VERBOSE)[1] == ["a"]
    assert their_flagged(r"([a-z])", re.IGNORECASE | re.ASCII)[5] == [None]


def test_a_flag_value_that_names_no_letter_is_still_refused(firepanda: ModuleType) -> None:
    """A bit belonging to none of the seven letters is refused rather than dropped, which is the
    rule document 85 set and the one a defect found in document 87 had been breaking."""
    with pytest.raises(firepanda.errors.UnsupportedError, match="flag value"):
        flagged(firepanda, r"([a-z])", 1024)


def test_a_pattern_that_is_not_text_is_refused_beside_a_flag_too(firepanda: ModuleType) -> None:
    """The check that used to live on the path a pattern took rather than at the entrance, which is
    correct only while there is one path. This name takes a second one now."""
    for flags in (0, re.IGNORECASE):
        with pytest.raises(TypeError):
            made(firepanda, FLAGGED_ROWS).str.extract(1, flags=flags)


@needs_pandas
def test_a_column_with_no_rows_answers_a_frame_of_the_right_width(firepanda: ModuleType) -> None:
    """The width comes off the pattern, so it survives a column with nothing in it to read.

    Both columns are built with the dtype spelled out, because neither library reads text out of a
    list that holds none.
    """
    frame = firepanda.Series([], dtype="string").str.extract(r"([a-z])(\d)")
    assert frame.shape == (0, 2)
    assert theirs([]).str.extract(r"([a-z])(\d)").shape == (0, 2)


@needs_pandas
def test_a_column_of_only_missing_rows_answers_missing(firepanda: ModuleType) -> None:
    """Which is the other column that has nothing to read, and unlike the one above it has rows."""
    rows = [None, None]
    mine = firepanda.Series(rows, dtype="string").str.extract(r"([a-z])(\d)")
    assert mine_rows(mine) == [[None, None], [None, None]]
    assert their_rows(theirs(rows).str.extract(r"([a-z])(\d)")) == [[None, None], [None, None]]
