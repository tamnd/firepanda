"""The four pattern methods with `case=False`, checked against pandas.

The fold a search compares through is not the fold `casefold` does, and that is
the whole slice. `casefold` is allowed to make a row longer, so `ß` becomes `ss`
and `ﬁ` becomes `fi`. A search cannot afford that, and pandas does not do it:
`contains`, `match` and `fullmatch` are answered by `pyarrow.compute` with
`ignore_case=True`, which folds one character to exactly one character, so
`STRASSE` does not hold `straße` there even though the two casefold to the same
word. Every claim in this file is asserted against live pandas rather than
against a rule, because the rule was recovered by measuring pandas in the first
place.

`replace` is the odd one of the four. pandas refuses `case=False` in its Arrow
path for that name alone and falls back to the object path, which escapes the
pattern and runs it with `re.IGNORECASE`, so a different implementation in a
different language decides its answer. The two agree on every pair, which did
not have to be true and is why one table serves all four here, and the tests
below check the agreement rather than assuming it.
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
    "AbCaBc",
    "aaaa",
    "a b c",
    "héllo",
    "HÉLLO",
    "日本語",
    None,
]
"""The nine rows the other pattern files use, with two case variants added."""

FOLDING = [
    "straße",
    "STRASSE",
    "Straße",
    "STRASSE",
    "ſtraße",
    "ﬁance",
    "FIANCE",
    "fiance",
    "KELVIN",
    "Kelvin",
    "ΣΟΦΟΣ",
    "σοφος",
    "σοφoς",
    "İstanbul",
    "ıstanbul",
    "istanbul",
    "µm",
    "μm",
    "ẞ",
    "ß",
    None,
]
"""Rows picked so that each one separates two rules that agree everywhere else."""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds it by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def without_the_missing(values: list[Any]) -> list[Any]:
    """Drops the last row, which is the null every mask here disagrees about.

    pandas hands back a bool column and a missing row becomes False in it, and
    this library keeps the row missing. That is the divergence the board records
    as `engine/string-predicate-null`, it is the same one the case sensitive
    file drops this row for, and folding the search changes nothing about it.
    """
    return values[:-1]


def mask_of(them: Any) -> list[Any]:
    """pandas' mask as a list, with every flavour of missing read as None."""
    return [None if one is None or one != one else bool(one) for one in them.tolist()]


def text_of(them: Any) -> list[Any]:
    """pandas' text column as a list, with its `nan` read as None."""
    return [None if one is None or one != one else one for one in them.tolist()]


PATTERNS = [
    "a",
    "A",
    "abc",
    "ABC",
    "aBc",
    "",
    "b c",
    "é",
    "É",
    "日",
    "zzz",
    "abcabc",
    "ABCABC",
]
"""Thirteen patterns, half of them differing from the rows only in case."""


@needs_pandas
def test_the_three_questions_match_pandas_on_every_pattern(firepanda: ModuleType) -> None:
    """contains, match and fullmatch, folded, against the ordinary rows."""
    mine, them = made(firepanda), theirs()
    for name in ("contains", "match", "fullmatch"):
        for pattern in PATTERNS:
            got = getattr(mine.str, name)(pattern, case=False)
            want = getattr(them.str, name)(pattern, case=False)
            assert without_the_missing(got.tolist()) == without_the_missing(mask_of(want)), (
                name,
                pattern,
            )


@needs_pandas
def test_the_fold_is_one_character_to_one_character(firepanda: ModuleType) -> None:
    """The finding the whole slice rests on, asserted against pandas and not a rule.

    `ß` casefolds to `ss` and `ﬁ` casefolds to `fi`, and neither of those folds
    happens in a search. If this test ever starts failing because pandas began
    answering True, the table under `searchfold.mojo` is the wrong table and not
    merely out of date.
    """
    mine, them = made(firepanda, FOLDING), theirs(FOLDING)
    for pattern in ("straße", "STRASSE", "ﬁance", "fiance"):
        got = mine.str.contains(pattern, case=False)
        assert without_the_missing(got.tolist()) == without_the_missing(
            mask_of(them.str.contains(pattern, case=False))
        ), pattern
    # Spelled out, so that the reason the two tables exist is readable here.
    rows = ["STRASSE"]
    assert made(firepanda, rows).str.contains("straße", case=False).tolist() == [False]
    assert theirs(rows).str.contains("straße", case=False).tolist() == [False]
    assert "STRASSE".casefold() == "strasse" == "straße".casefold()


@needs_pandas
def test_the_fold_is_not_the_lower_case_either(firepanda: ModuleType) -> None:
    """Four characters that lowercasing gets wrong and simple folding gets right.

    Final sigma against medial sigma, the micro sign against Greek mu, long s
    against s, and the Kelvin sign against k. A search calls each pair the same
    character and `str.lower` does not, which is why the table is derived from
    the fold rather than from the lowercase.
    """
    rows = ["ΣΟΦΟΣ", "σοφος", "µm", "μm", "ſtraße", "straße", "KELVIN", "Kelvin"]
    mine, them = made(firepanda, rows), theirs(rows)
    for pattern in ("σοφος", "ΣΟΦΟΣ", "μm", "µm", "s", "ſ", "kelvin", "Klvin"):
        got = mine.str.contains(pattern, case=False)
        assert got.tolist() == mask_of(them.str.contains(pattern, case=False)), pattern


@needs_pandas
def test_a_match_can_cover_a_different_number_of_bytes(firepanda: ModuleType) -> None:
    """`ſ` is two bytes and is compared as the one byte `s`, which fullmatch feels.

    A case sensitive `fullmatch` can decide most rows by comparing lengths and a
    folded one cannot, so this is the assertion that the kernel walks both sides
    instead of taking the shortcut its sibling takes.
    """
    rows = ["ſtraße", "straße", "STRASSE", "ſ", "s", "S"]
    mine, them = made(firepanda, rows), theirs(rows)
    for pattern in ("straße", "ſtraße", "s", "ſ"):
        got = mine.str.fullmatch(pattern, case=False)
        assert got.tolist() == mask_of(them.str.fullmatch(pattern, case=False)), pattern


@needs_pandas
def test_replace_folds_the_same_way_the_three_questions_do(firepanda: ModuleType) -> None:
    """Which pandas gets out of `re.IGNORECASE` and the other three out of Arrow.

    There is no reason in principle for two implementations in two languages to
    agree about this, so the agreement is checked rather than assumed: wherever
    `contains` says a row holds the pattern, `replace` has to change that row.
    """
    mine, them = made(firepanda, FOLDING), theirs(FOLDING)
    for pattern in ("straße", "ſ", "σ", "µ", "s"):
        got = mine.str.replace(pattern, "#", case=False)
        assert got.tolist() == text_of(them.str.replace(pattern, "#", case=False)), pattern
        changed = [
            None if row is None else row != answer
            for row, answer in zip(FOLDING, got.tolist(), strict=True)
        ]
        assert changed == mine.str.contains(pattern, case=False).tolist(), pattern


@needs_pandas
def test_replace_matches_pandas_on_every_pattern(firepanda: ModuleType) -> None:
    """The ordinary rows, every pattern, and the three interesting counts."""
    mine, them = made(firepanda), theirs()
    for pattern in PATTERNS:
        for count in (-1, 0, 1, 2):
            got = mine.str.replace(pattern, "#", n=count, case=False)
            want = them.str.replace(pattern, "#", n=count, case=False)
            assert got.tolist() == text_of(want), (pattern, count)


@needs_pandas
def test_replace_keeps_the_case_of_what_it_did_not_touch(firepanda: ModuleType) -> None:
    """Only the comparison is folded, and the rest of the row is copied through."""
    rows = ["AbCaBc", "ABC", "abc"]
    mine, them = made(firepanda, rows), theirs(rows)
    got = mine.str.replace("b", "#", case=False)
    assert got.tolist() == ["A#Ca#c", "A#C", "a#c"]
    assert got.tolist() == them.str.replace("b", "#", case=False).tolist()


@needs_pandas
def test_an_empty_pattern_has_nothing_to_do_with_case(firepanda: ModuleType) -> None:
    """So the folded kernels hand it to the ones beside them rather than branching."""
    mine, them = made(firepanda), theirs()
    for name in ("contains", "match", "fullmatch"):
        folded = getattr(mine.str, name)("", case=False)
        assert folded.tolist() == getattr(mine.str, name)("").tolist(), name
        assert without_the_missing(folded.tolist()) == without_the_missing(
            mask_of(getattr(them.str, name)("", case=False))
        ), name
    assert mine.str.replace("", "#", case=False).tolist() == mine.str.replace("", "#").tolist()


@needs_pandas
def test_a_count_of_zero_means_all_of_them_once_the_search_is_folded(
    firepanda: ModuleType,
) -> None:
    """Which is pandas, is measured, and is not a rule anybody would have guessed.

    `n=0` is a request for no replacements on the case sensitive path and a
    request for every replacement on the folded one. pandas answers the first
    out of Arrow, which takes the number at its word, and the second out of
    `re.sub`, where a count of zero has meant unlimited since long before pandas
    existed. Same method, same column, same argument, two answers depending on
    another argument entirely.
    """
    rows = ["abcabc", "ABCabc"]
    mine, them = made(firepanda, rows), theirs(rows)
    assert mine.str.replace("a", "#", n=0).tolist() == rows
    assert them.str.replace("a", "#", n=0).tolist() == rows
    assert mine.str.replace("a", "#", n=0, case=False).tolist() == ["#bc#bc", "#BC#bc"]
    assert them.str.replace("a", "#", n=0, case=False).tolist() == ["#bc#bc", "#BC#bc"]


@needs_pandas
def test_a_folded_search_keeps_a_missing_row_missing(firepanda: ModuleType) -> None:
    """Which pandas does for the replacement and does not for the three masks.

    The masks are the `engine/string-predicate-null` divergence, unchanged by
    the fold and recorded on the board rather than worked around here, and it is
    asserted in both directions so that a change on either side shows up.
    """
    mine = made(firepanda)
    for name in ("contains", "match", "fullmatch"):
        assert getattr(mine.str, name)("a", case=False).tolist()[-1] is None, name
        assert theirs().str.contains("a", case=False).tolist()[-1] is False
    assert mine.str.replace("a", "#", case=False).tolist()[-1] is None
    assert text_of(theirs().str.replace("a", "#", case=False))[-1] is None


@needs_pandas
def test_case_true_is_the_search_that_was_already_there(firepanda: ModuleType) -> None:
    """The default, spelled out, which has to reach the unfolded kernel unchanged."""
    mine = made(firepanda, FOLDING)
    for name in ("contains", "match", "fullmatch"):
        for pattern in ("straße", "s", "ΣΟΦΟΣ"):
            asked = getattr(mine.str, name)(pattern, case=True)
            assert asked.tolist() == getattr(mine.str, name)(pattern).tolist(), (name, pattern)


@needs_pandas
def test_flags_are_still_refused_with_case_off(firepanda: ModuleType) -> None:
    """`case=False` is served on all four of these now and `flags` is not, except
    on the one name upstream lets through.

    `re.IGNORECASE` as a flag asks for the same fold `case=False` asks for, and
    that is not enough to make it the same call: pandas routes a pattern that
    was handed any flag to Python's engine, and the one exception is `match`,
    which is measured in `test_str_regex_case_and_flags.py` rather than here.
    """
    import re

    mine = made(firepanda)
    for name in ("contains", "fullmatch"):
        with pytest.raises(firepanda.errors.UnsupportedError):
            getattr(mine.str, name)("a", case=False, flags=re.IGNORECASE)
    with pytest.raises(firepanda.errors.UnsupportedError):
        mine.str.replace("a", "#", case=False, flags=re.IGNORECASE)


@needs_pandas
def test_a_metacharacter_is_still_refused_with_case_off(firepanda: ModuleType) -> None:
    """Folding a pattern does not make it a literal, and it no longer has to.

    These three used to refuse a metacharacter beside `case=False` because the
    engine had no fold to run it with. It has one now, so the pattern goes to
    the engine and the row is searched rather than the refusal being raised, and
    what is left to assert here is that `regex=False` still means the characters
    themselves. The folded engine paths are measured in
    `test_str_regex_case_and_flags.py`.
    """
    mine = made(firepanda)
    for name in ("contains", "match", "fullmatch"):
        assert getattr(mine.str, name)("a.c", case=False).tolist()[0] is not None
    assert mine.str.contains("a.c", case=False, regex=False).tolist()[0] is False
