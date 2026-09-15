"""`str.replace` with a literal pattern, checked against pandas.

This is the fifth `str` name about a pattern and the only one of the five where
pandas asks for a literal by default. Its `regex` argument defaults to False in
pandas 3, so the ordinary call is a byte search and a rewrite, and there is
nothing to refuse and no smaller promise to make. `regex=True` goes through the
same check the other four use, which answers a pattern with no metacharacter in
it and refuses the rest.

One thing about the group is worth knowing before reading the assertions, and it
is the reason this file exists separately from `test_str_pattern.py`.

An empty pattern is counted in characters here and in bytes in `count`, on the
same data, in the same accessor. `pandas.Series(["héllo"]).str.count("")` is 7,
which is the six bytes of the row plus one, and
`pandas.Series(["héllo"]).str.replace("", "-")` puts six dashes in, which is the
five characters of the row plus one. The reason is not a decision anybody made:
`pyarrow.compute.replace_substring` does not terminate on an empty pattern, so
pandas has a guard for that one case that hands the row to Python's own
`str.replace` instead, and Python counts characters. `count` has no such guard
and stays in Arrow, and Arrow counts bytes. Both answers are pandas' answers, so
this library gives both.
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
    "abcabc",
    "ab",
    "",
    "ABC",
    "aaaa",
    "a b c",
    "héllo",
    "日本語",
    None,
]
"""The same nine rows the other pattern file uses, for the same reasons."""

PATTERNS = ("bc", "a", "abc", "", "aa", "z", "héllo", "本")
"""Eight literal patterns, none of them holding a metacharacter."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds it by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def same(mine: Any, them: Any) -> bool:
    """Whether two columns agree, with pandas' missing row read as None.

    pandas hands a missing row back as a NaN float and this library hands back
    None, which is `engine/missing-spelling` and is not what any test here is
    about.
    """
    theirs_rows = [None if one is None or one != one else one for one in them.tolist()]
    return mine.tolist() == theirs_rows


@needs_pandas
def test_replace_matches_pandas_on_every_literal_pattern(firepanda: ModuleType) -> None:
    """Both spellings, since `regex=True` on a literal is the same question."""
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        want = them.str.replace(pattern, "X")
        assert same(mine.str.replace(pattern, "X"), want), pattern
        assert same(mine.str.replace(pattern, "X", regex=True), want), pattern


@needs_pandas
def test_replace_matches_pandas_at_every_count(firepanda: ModuleType) -> None:
    """Including zero, which hands the row back, and a negative, which means all."""
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        for n in (-2, -1, 0, 1, 2, 3, 100):
            got = mine.str.replace(pattern, "X", n=n)
            assert same(got, them.str.replace(pattern, "X", n=n)), (pattern, n)


@needs_pandas
def test_an_empty_replacement_deletes(firepanda: ModuleType) -> None:
    """Which is the shortest way to write a delete and is what pandas does."""
    mine, them = made(firepanda), theirs()
    for pattern in ("a", "abc", "héllo"):
        assert same(mine.str.replace(pattern, ""), them.str.replace(pattern, "")), pattern


@needs_pandas
def test_an_empty_pattern_is_replaced_between_characters_and_counted_in_bytes(
    firepanda: ModuleType,
) -> None:
    """The two halves of the same argument, answered in two different units.

    Both numbers are pandas' and the difference is a workaround in pandas for a
    pyarrow kernel that does not terminate. Written out together because either
    one on its own reads like a mistake.
    """
    rows = ["héllo", "日本", "hello", ""]
    assert made(firepanda, rows).str.replace("", "-").tolist() == [
        "-h-é-l-l-o-",
        "-日-本-",
        "-h-e-l-l-o-",
        "-",
    ]
    assert theirs(rows).str.replace("", "-").tolist() == [
        "-h-é-l-l-o-",
        "-日-本-",
        "-h-e-l-l-o-",
        "-",
    ]
    assert made(firepanda, rows).str.count("").tolist() == [7, 7, 6, 1]
    assert theirs(rows).str.count("").tolist() == [7, 7, 6, 1]


@needs_pandas
def test_an_empty_pattern_obeys_the_count(firepanda: ModuleType) -> None:
    """A row shorter than the count stops when it runs out of places to insert."""
    rows = ["abcabc", "ab", ""]
    for n in (1, 2, 3):
        got = made(firepanda, rows).str.replace("", "-", n=n)
        assert got.tolist() == theirs(rows).str.replace("", "-", n=n).tolist(), n


@needs_pandas
def test_replace_does_not_let_matches_overlap(firepanda: ModuleType) -> None:
    """Four a's hold two runs of two, and pandas agrees."""
    rows = ["aaaa", "aaaaa", "abab", "aa", "a"]
    want = ["XX", "XXa", "abab", "X", "a"]
    assert made(firepanda, rows).str.replace("aa", "X").tolist() == want
    assert theirs(rows).str.replace("aa", "X").tolist() == want


@needs_pandas
def test_a_metacharacter_is_the_character_itself_by_default(
    firepanda: ModuleType,
) -> None:
    """Which is the whole reason this name needs no promise made for it."""
    rows = ["a.c", "abc", "a+c", "a|b", ""]
    for pattern in (".", "a.c", "+", "|", "[", "$", "\\"):
        got = made(firepanda, rows).str.replace(pattern, "Z")
        assert got.tolist() == theirs(rows).str.replace(pattern, "Z").tolist(), pattern


@needs_pandas
def test_a_metacharacter_with_regex_on_is_read_as_a_pattern(
    firepanda: ModuleType,
) -> None:
    """Which is the other half of the test above, and used to be a refusal.

    Turning `regex` on changes more than the pattern here. It changes the
    replacement as well, because Arrow reads a replacement with a grammar of its
    own in which a pair of backslashes is one backslash and a backslash and a
    digit is a group. `test_str_replace_regex.py` is the test of all of that.
    """
    rows = ["a.c", "abc", "a+c", "a|b", "", "aaaa"]
    for pattern in ("a.c", "^a", "a+", "a|b", "[ab]", "a*"):
        got = made(firepanda, rows).str.replace(pattern, "X", regex=True).tolist()
        assert got == theirs(rows).str.replace(pattern, "X", regex=True).tolist(), pattern


@needs_pandas
def test_a_dictionary_is_several_replacements_in_order(firepanda: ModuleType) -> None:
    """Which is what pandas does with one, and the order is the dictionary's."""
    rows = ["abcabc", "ab", "", "cba"]
    swaps = {"a": "X", "b": "Y", "c": "Z"}
    got = made(firepanda, rows).str.replace(swaps)
    assert got.tolist() == theirs(rows).str.replace(swaps).tolist()


@needs_pandas
def test_flags_are_refused_rather_than_ignored(
    firepanda: ModuleType,
) -> None:
    """The other half of this pair is answered now, and this half needs an engine.

    `case=False` landed with the folded search and is covered by
    `test_str_case_insensitive.py`, which is also where the odd fact lives that
    pandas answers this one name out of `re.IGNORECASE` rather than out of Arrow.
    """
    import re

    mine = made(firepanda)
    with pytest.raises(firepanda.errors.UnsupportedError):
        mine.str.replace("a", "X", flags=re.IGNORECASE)


@needs_pandas
def test_a_callable_replacement_is_refused(firepanda: ModuleType) -> None:
    """pandas refuses this too when regex is off, and needs an engine when it is on."""
    mine = made(firepanda)
    with pytest.raises(firepanda.errors.UnsupportedError):
        mine.str.replace("a", lambda match: "X")


@needs_pandas
def test_arguments_of_the_wrong_type_are_type_errors(firepanda: ModuleType) -> None:
    """A pattern, a replacement and a count, all three checked the way pandas checks."""
    mine = made(firepanda)
    for bad in (1, None, 1.5):
        with pytest.raises(TypeError):
            mine.str.replace(bad, "X")
    for bad in (1, None, 1.5):
        with pytest.raises(TypeError):
            mine.str.replace("a", bad)
    for bad in ("1", 1.5, True):
        with pytest.raises(TypeError):
            mine.str.replace("a", "X", n=bad)


@needs_pandas
def test_replace_keeps_a_missing_row_missing(firepanda: ModuleType) -> None:
    """As pandas does, which makes this the one name of the five that agrees."""
    mine = made(firepanda)
    assert mine.str.replace("a", "X").tolist()[-1] is None
    assert theirs().str.replace("a", "X").isna().tolist()[-1] is True
