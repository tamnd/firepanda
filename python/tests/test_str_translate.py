"""`str.translate`, checked against pandas.

pandas hands the table straight to Python's `str.translate`, which looks a
character up by its ordinal and leaves it alone whenever the lookup raises a
`LookupError`. So the rule is Python's rule and there is nothing about Arrow in
it, which makes this the first name in a while whose answer needed no argument
about which backend pandas was reading out of.

Two things separate this from `replace` and both are asserted below. A key is
always exactly one character, so nothing is searched for and no match can
overlap another. And every key is applied in the same pass, so a table that
swaps two characters for each other really swaps them, where the same pair given
to `replace` one after the other would collapse both into the same letter.
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
"""The same nine rows the other two string files use."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds it by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def same(mine: Any, them: Any) -> bool:
    """Whether two columns agree, with pandas' missing row read as None."""
    theirs_rows = [None if one is None or one != one else one for one in them.tolist()]
    return mine.tolist() == theirs_rows


TABLES: list[dict[int, Any]] = [
    {ord("a"): ord("X")},
    {ord("a"): "X"},
    {ord("a"): "XY"},
    {ord("a"): None},
    {ord("a"): ""},
    {ord("a"): "X", ord("b"): "Y", ord("c"): "Z"},
    {ord("a"): "b", ord("b"): "a"},
    {ord("é"): "E"},
    {ord("日"): None, ord("語"): "GO"},
    {ord("a"): "é", ord("é"): "a"},
    {ord(" "): "_"},
    {ord("z"): "Q"},
    {},
]
"""Thirteen tables, covering every shape a value is allowed to be."""


@needs_pandas
def test_translate_matches_pandas_on_every_table(firepanda: ModuleType) -> None:
    """Including the empty one, which hands every row back."""
    mine, them = made(firepanda), theirs()
    for table in TABLES:
        assert same(mine.str.translate(table), them.str.translate(table)), table


@needs_pandas
def test_maketrans_is_the_usual_way_to_build_one(firepanda: ModuleType) -> None:
    """Which is why a mapping is the shape this takes and a general object is not."""
    mine, them = made(firepanda), theirs()
    for table in (
        str.maketrans("ab", "XY"),
        str.maketrans("", "", "ab"),
        str.maketrans("aé", "éa"),
        str.maketrans({"a": "XY", "b": None}),
    ):
        assert same(mine.str.translate(table), them.str.translate(table)), table


@needs_pandas
def test_every_key_is_applied_in_the_same_pass(firepanda: ModuleType) -> None:
    """The difference between this and two replaces, spelled out beside them."""
    rows = ["ab", "abab", "ba"]
    swap = {ord("a"): "b", ord("b"): "a"}
    assert made(firepanda, rows).str.translate(swap).tolist() == ["ba", "baba", "ab"]
    assert theirs(rows).str.translate(swap).tolist() == ["ba", "baba", "ab"]
    # The same pair one after the other, which is what a dict does to replace.
    stepwise = made(firepanda, rows).str.replace({"a": "b", "b": "a"})
    assert stepwise.tolist() == ["aa", "aaaa", "aa"]
    assert theirs(rows).str.replace({"a": "b", "b": "a"}).tolist() == ["aa", "aaaa", "aa"]


@needs_pandas
def test_what_a_key_maps_to_is_never_looked_at_again(firepanda: ModuleType) -> None:
    """So a key that maps to itself is a no-op and a chain does not chain."""
    rows = ["ab", "aa"]
    for table in ({ord("a"): "aa"}, {ord("a"): "a"}, {ord("a"): "ba", ord("b"): "c"}):
        got = made(firepanda, rows).str.translate(table)
        assert got.tolist() == theirs(rows).str.translate(table).tolist(), table


@needs_pandas
def test_deleting_and_mapping_to_nothing_are_the_same_request(
    firepanda: ModuleType,
) -> None:
    """Which is why the crossing carries no third case for a delete."""
    rows = ["abcabc", "aaa", ""]
    by_none = made(firepanda, rows).str.translate({ord("a"): None})
    by_empty = made(firepanda, rows).str.translate({ord("a"): ""})
    assert by_none.tolist() == by_empty.tolist()
    assert by_none.tolist() == theirs(rows).str.translate({ord("a"): None}).tolist()


@needs_pandas
def test_a_key_that_could_never_have_matched_is_dropped(firepanda: ModuleType) -> None:
    """pandas looks a character up by its ordinal, so a text key never matches."""
    rows = ["abc"]
    for table in ({"a": "X"}, {-1: "X"}, {0x110000: "X"}, {"a": "X", ord("b"): "Y"}):
        got = made(firepanda, rows).str.translate(table)
        assert got.tolist() == theirs(rows).str.translate(table).tolist(), table


@needs_pandas
def test_a_value_out_of_range_or_of_the_wrong_type_is_refused(
    firepanda: ModuleType,
) -> None:
    """With pandas' own two sentences, which say which of the two went wrong."""
    mine = made(firepanda)
    for bad in (0x110000, -1):
        with pytest.raises(ValueError) as caught:
            mine.str.translate({ord("a"): bad})
        assert "range(0x110000)" in str(caught.value)
    for wrong in (1.5, [], {}):
        with pytest.raises(TypeError) as raised:
            mine.str.translate({ord("a"): wrong})
        assert "integer, None or str" in str(raised.value)


@needs_pandas
def test_a_table_that_is_not_a_mapping_is_refused(firepanda: ModuleType) -> None:
    """pandas takes anything subscriptable and this takes a mapping, by choice.

    Serving the general case means asking a Python object about every character
    of every row, which is the one thing crossing into a kernel is for avoiding.
    A refusal reads as a gap on the board; answering it slowly would not.
    """
    mine = made(firepanda)
    for table in ([1, 2, 3], "abc", object()):
        with pytest.raises(firepanda.errors.UnsupportedError):
            mine.str.translate(table)


@needs_pandas
def test_translate_keeps_a_missing_row_missing(firepanda: ModuleType) -> None:
    """As pandas does, which makes this the second text name in a row that agrees."""
    mine = made(firepanda)
    assert mine.str.translate({ord("a"): "X"}).tolist()[-1] is None
    assert theirs().str.translate({ord("a"): "X"}).isna().tolist()[-1] is True


@needs_pandas
def test_the_accessor_still_refuses_a_column_that_is_not_text(
    firepanda: ModuleType,
) -> None:
    """The check every name here shares, and the accessor makes it before this does.

    pandas raises an `AttributeError` when the accessor is reached for rather than
    when a method on it is called, because the problem is with the column and not
    with the name, and this does the same.
    """
    with pytest.raises(AttributeError) as caught:
        firepanda.Series([1, 2, 3]).str.translate({ord("a"): "X"})
    assert "string values" in str(caught.value)
