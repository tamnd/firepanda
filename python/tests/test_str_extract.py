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
)
"""Seven patterns: two groups, one group, a class, a name, an optional group, an empty match and a
group that matches anything."""


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


def test_flags_are_refused(firepanda: ModuleType) -> None:
    """The same refusal the four names above this one make, for the same reason."""
    with pytest.raises(firepanda.errors.UnsupportedError, match="flags"):
        made(firepanda).str.extract(r"([a-z])", flags=2)


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
