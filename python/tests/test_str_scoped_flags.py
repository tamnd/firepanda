"""`(?i:a)` and the rest of the scoped flag groups, against pandas.

A scoped group is a letter with a reach. Every other way of writing a flag
applies to the whole pattern, whether it was written as `(?i)` at the front or
passed beside the pattern as a number, and this is the one form where `a` and
`b` in the same pattern can be read under different rules. That is why it took a
node of its own rather than a wider field on the parse: a set of letters with no
place attached is enough to refuse a pattern on and not enough to answer one.

Both engines have some of it and neither has all of it. RE2 reads `(?i:`, `(?m:`
and `(?s:` and refuses `(?x:` and `(?a:` with the same sentence it refuses `(?x)`
and `(?a)` with, so a scoped letter RE2 never had is an error upstream as much as
a global one is. Python reads all of them but `(?L:`, which it turns down on a
pattern made of text wherever it appears.

The rows below are picked so that a dropped letter and a letter that reached too
far are two different wrong answers rather than one. `(?i:a)bc` on `aBc` is False
and a global `(?i)` would make it True, and `(?i:a)bc` on `Abc` is True and a
tree with the letters thrown away would make it False. Every pattern here is run
against live pandas rather than against a table, because the table would be a
second reading of the same rule.
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
    "abc",
    "ABC",
    "aBc",
    "Abc",
    "ab",
    "a b",
    "abcd",
    "abc d",
    "a\nb",
    "\u00e9",
    "a\u00e9",
    "\u00e9a",
    "\u212a",
    "\u017f",
    "k",
    "\x0b",
    "",
    None,
]
"""Rows picked so that each letter has somewhere its reach shows.

The four spellings of `abc` are for ignore case, the two rows holding a space
are for verbose mode, the row holding a newline is for dotall and multiline, and
the accented letter with a plain one on either side of it is for the ascii flag,
which is the only letter whose scope can be read off a two character row. The
Kelvin sign and the long s are there because the narrow fold and the wide one
part company on them and a scope is where the two can be asked in one pattern.
"""

SHARED = [
    "(?i:a)bc",
    "(?i:ab)c",
    "a(?i:b)c",
    "(?i:(?-i:a)b)c",
    "(?i:a(?-i:b))c",
    "(?-i:a)bc",
    "(?s:.)b",
    "(?m:^b)",
    "(?i:a)+b",
    "(?:(?i:a))bc",
    "(?i:[a-c])x",
    "(?i:k)",
]
"""The scoped groups both engines read, which are the three letters RE2 has.

Run twice over, once with no flags so that the call goes to Arrow the way
pandas sends it there, and once with a flag beside it so that the same pattern
is answered by Python's engine. The two engines are supposed to disagree about
the Kelvin sign and to agree about everything else in this list, and that is
pandas' disagreement rather than this library's, so the comparison is against
pandas each way round.
"""

PYTHON_ONLY = [
    "(?x:a b)c",
    "(?x:a b)c d",
    "(?-x:a b)",
    "(?a:\\w)",
    "(?a:\\w)\\w",
    "(?a:\\s)",
    "(?a:\\b)x",
    "(?ia:k)",
    "(?i:(?a:k))",
    "(?u:\\w)",
    "(?a:\\w)(?u:\\w)",
]
"""The two letters RE2 never had, which reach an answer only once a flag has
moved the call off Arrow. `(?a:\\w)(?u:\\w)` is the row that says the two
alphabets are a conflict at the top of a pattern and nowhere else, since the
same two letters written globally are a `ValueError` in Python."""

FLAGS = [
    re.IGNORECASE,
    re.MULTILINE,
    re.DOTALL,
    re.VERBOSE,
    re.ASCII,
    re.IGNORECASE | re.MULTILINE,
]
"""The flag beside the pattern, whose first job here is to move the call. What
it turns on is a second question and the scoped group may turn the same letter
back off, which is what `(?-i:a)bc` under `re.IGNORECASE` is for."""

HELD_OUT = {("(?u:\\w)", re.ASCII)}
"""The one cell of the sweep where this library and pandas part company.

It is a CPython defect rather than a difference of opinion and it has a test of
its own further down, which asserts both sides of it so that the day it is fixed
upstream is a day this file fails rather than a day nobody notices.
"""


def made(firepanda: ModuleType, values: list[Any] = ROWS) -> Any:
    """A firepanda text column."""
    return firepanda.Series(values)


def theirs(values: list[Any] = ROWS) -> Any:
    """The same column in pandas, held the way pandas holds one by default."""
    import pandas as pd

    return pd.Series(values, dtype="str")


def mask_of(them: Any) -> list[Any]:
    """pandas' mask as a list, with every flavour of missing read as None."""
    return [None if one is None or one != one else bool(one) for one in them.tolist()]


def counts_of(them: Any) -> list[Any]:
    """pandas' counts as a list, with every flavour of missing read as None."""
    return [None if one is None or one != one else int(one) for one in them.tolist()]


def texts_of(them: Any) -> list[Any]:
    """pandas' text answers as a list, with every flavour of missing read as None."""
    return [None if one is None or one != one else one for one in them.tolist()]


def without_the_missing(values: list[Any]) -> list[Any]:
    """Drops the last row, which is the null every mask here disagrees about.

    pandas hands back a bool column and a missing row becomes False in it, and
    this library keeps the row missing. That is the divergence the board records
    as `engine/string-predicate-null` and a scoped flag changes nothing about it.
    """
    return values[:-1]


@needs_pandas
def test_a_scoped_group_answers_the_same_out_of_arrow(firepanda: ModuleType) -> None:
    """The three letters RE2 has, with no flag beside them, which is the call
    pandas sends to Arrow. Five methods and twelve patterns, and nothing is held
    out but the null row the two libraries have always disagreed about."""
    mine, them = made(firepanda), theirs()
    for pattern in SHARED:
        for name in ("contains", "match", "fullmatch"):
            got = getattr(mine.str, name)(pattern).tolist()
            want = mask_of(getattr(them.str, name)(pattern))
            assert without_the_missing(got) == without_the_missing(want), (name, pattern)
        assert mine.str.count(pattern).tolist() == counts_of(them.str.count(pattern)), pattern
        assert mine.str.replace(pattern, "#", regex=True).tolist() == texts_of(
            them.str.replace(pattern, "#", regex=True)
        ), pattern


@needs_pandas
def test_a_scoped_group_answers_the_same_on_pythons_engine(firepanda: ModuleType) -> None:
    """The same patterns and the two letters RE2 never had, with a flag beside
    them so the call lands on Python's engine instead.

    `match` is not asked here. Upstream keeps that one name's flags on Arrow and
    raises `Cannot pass flags that do not match pat.flags` for every pattern
    below, which is a fact about pandas recorded in
    `test_str_regex_case_and_flags.py` rather than anything to compare.
    """
    mine, them = made(firepanda), theirs()
    for pattern in SHARED + PYTHON_ONLY:
        for flags in FLAGS:
            if (pattern, flags) in HELD_OUT:
                continue
            for name in ("contains", "fullmatch"):
                got = getattr(mine.str, name)(pattern, flags=flags).tolist()
                want = mask_of(getattr(them.str, name)(pattern, flags=flags))
                assert without_the_missing(got) == without_the_missing(want), (
                    name,
                    pattern,
                    flags,
                )
            got_counts = mine.str.count(pattern, flags=flags).tolist()
            assert got_counts == counts_of(them.str.count(pattern, flags=flags)), (
                pattern,
                flags,
            )
            got_texts = mine.str.replace(pattern, "#", regex=True, flags=flags).tolist()
            assert got_texts == texts_of(them.str.replace(pattern, "#", regex=True, flags=flags)), (
                pattern,
                flags,
            )


@needs_pandas
def test_the_scope_ends_where_the_bracket_does(firepanda: ModuleType) -> None:
    """The one claim the whole node exists for, written out rather than swept.

    Both wrong answers are here. A tree that dropped the letters answers the
    first row False, and a letter that reached past its bracket answers the
    second row True, and neither would show up in a test that only asked whether
    a scoped pattern was accepted.
    """
    mine, them = made(firepanda), theirs()
    got = mine.str.contains("(?i:a)bc").tolist()
    want = mask_of(them.str.contains("(?i:a)bc"))
    for row, answer in (("Abc", True), ("aBc", False), ("abc", True), ("ABC", False)):
        assert got[ROWS.index(row)] is answer, row
        assert want[ROWS.index(row)] is answer, row


@needs_pandas
def test_the_inner_group_puts_back_what_the_outer_one_had(firepanda: ModuleType) -> None:
    """Nesting, which says the letters are saved and restored rather than
    cleared. If the restore wrote the pattern's own flags back instead of the
    enclosing group's, `aBc` would be False here and `Abc` would be True."""
    mine, them = made(firepanda), theirs()
    got = mine.str.contains("(?i:(?-i:a)b)c").tolist()
    want = mask_of(them.str.contains("(?i:(?-i:a)b)c"))
    for row, answer in (("aBc", True), ("Abc", False), ("ABC", False), ("abc", True)):
        assert got[ROWS.index(row)] is answer, row
        assert want[ROWS.index(row)] is answer, row


@needs_pandas
def test_verbose_mode_is_spent_while_the_pattern_is_read(firepanda: ModuleType) -> None:
    """The odd letter out. The other six are questions the compiler asks about a
    character and this one decides which characters there are, so it is turned on
    and off around the reading of the group rather than around the running of it.
    The space after the closing bracket is outside the group and stays."""
    mine, them = made(firepanda), theirs()
    for pattern, matched, missed in (("(?x:a b)c", "abc", "a b"), ("(?x:a b)c d", "abc d", "abcd")):
        got = mine.str.contains(pattern, flags=re.MULTILINE).tolist()
        want = mask_of(them.str.contains(pattern, flags=re.MULTILINE))
        assert got[ROWS.index(matched)] is True, pattern
        assert got[ROWS.index(missed)] is False, pattern
        assert want[ROWS.index(matched)] is True, pattern
        assert want[ROWS.index(missed)] is False, pattern


@needs_pandas
def test_the_alphabet_narrows_for_part_of_a_pattern(firepanda: ModuleType) -> None:
    """The ascii flag, which is the letter that can be read off a two character
    row because the scope covers one character and not the other. `é` is a word
    character to Python and not to the narrow reading, so which side of the
    bracket it falls on is the whole answer."""
    mine, them = made(firepanda), theirs()
    got = mine.str.contains("(?a:\\w)\\w", flags=re.MULTILINE).tolist()
    want = mask_of(them.str.contains("(?a:\\w)\\w", flags=re.MULTILINE))
    for row, answer in (("a\u00e9", True), ("\u00e9a", False), ("ab", True)):
        assert got[ROWS.index(row)] is answer, row
        assert want[ROWS.index(row)] is answer, row


@needs_pandas
def test_the_letters_re2_never_had_are_refused_in_a_scoped_group_too(
    firepanda: ModuleType,
) -> None:
    """Agreement rather than a shortfall. Arrow reads `(?i:` and has never heard
    of `(?x:` or `(?a:` in any position, so a call with no flag on it is an error
    upstream and an error here, and the refusal is reproduced in kind rather than
    in wording because pandas' wording names a library nobody called."""
    mine, them = made(firepanda), theirs()
    for pattern in ("(?x:a b)c", "(?a:\\w)", "(?u:\\w)", "(?ia:k)"):
        with pytest.raises(ValueError):
            them.str.contains(pattern)
        with pytest.raises(ValueError):
            mine.str.contains(pattern)


@needs_pandas
def test_a_widening_scope_at_the_front_of_a_pattern_is_a_divergence(
    firepanda: ModuleType,
) -> None:
    """The one cell of the sweep that is held out, asserted from both sides.

    `re.search` on a pattern that opens with a scoped group widening the
    alphabet skips positions the pattern matches. `re.fullmatch` on the same
    pattern and the same row does not, and neither does the same pattern with an
    alternation bar in it, which is what says this is the first character
    optimisation reading the outer flags rather than the combined ones.

    So `contains`, `count` and `replace` inherit a wrong answer upstream and
    `fullmatch` does not, and the two sit in one accessor. `match` is not asked
    because upstream refuses a flag beside that one name. This
    library answers the same question the same way whichever name asked it,
    which is the divergence, and reproducing the other behaviour would mean
    putting an optimisation into the semantics where a defect could reach it.
    """
    rows = ["\u00e9", "a\u00e9"]
    mine, them = made(firepanda, rows), theirs(rows)
    assert mine.str.contains("(?u:\\w)", flags=re.ASCII).tolist() == [True, True]
    assert mask_of(them.str.contains("(?u:\\w)", flags=re.ASCII)) == [False, True]
    assert mine.str.count("(?u:\\w)", flags=re.ASCII).tolist() == [1, 2]
    assert counts_of(them.str.count("(?u:\\w)", flags=re.ASCII)) == [0, 1]
    assert mine.str.fullmatch("(?u:\\w)", flags=re.ASCII).tolist() == [True, False]
    assert mask_of(them.str.fullmatch("(?u:\\w)", flags=re.ASCII)) == [True, False]
    assert bool(re.search("(?u:\\w)", "\u00e9", re.ASCII)) is False
    assert bool(re.fullmatch("(?u:\\w)", "\u00e9", re.ASCII)) is True
    assert bool(re.search("(?u:\\w)|z", "\u00e9", re.ASCII)) is True


@needs_pandas
def test_the_locale_letter_is_still_turned_down_inside_a_bracket(
    firepanda: ModuleType,
) -> None:
    """The one letter neither engine takes. Python turns it down on a pattern
    made of text wherever it is written and Arrow has never heard of it, so both
    libraries refuse both routes, and a scoped group that carried its letters
    could have been the moment somebody stopped checking."""
    mine, them = made(firepanda), theirs()
    with pytest.raises(NotImplementedError):
        mine.str.contains("(?L:a)")
    with pytest.raises(ValueError):
        them.str.contains("(?L:a)")
    with pytest.raises(ValueError):
        mine.str.contains("(?L:a)", flags=re.MULTILINE)
    with pytest.raises(re.PatternError):
        them.str.contains("(?L:a)", flags=re.MULTILINE)


@needs_pandas
def test_a_scope_is_not_a_group_anybody_can_refer_to(firepanda: ModuleType) -> None:
    """The numbering, which is the reason this is a node of its own rather than
    a capturing group whose number happens to be zero. A scope around a capture
    leaves the capture where it was, and `extract` is where that shows."""
    mine = made(firepanda, ["abc", "ABC"])
    assert mine.str.extract("(?i:(a))")["0"].tolist() == ["a", "A"]
    assert mine.str.extract("(?i:(a))(b)").shape[1] == 2


@needs_pandas
def test_a_quantifier_repeats_the_scope_and_not_the_letter(firepanda: ModuleType) -> None:
    """A repeat above a scope, which is where a letter left switched on after
    the last pass would show up. `(?i:a)+b` folds every `a` it takes and never
    the `b` after them, however many passes it made."""
    rows = ["Aab", "AaB", "aab", "aaB"]
    mine, them = made(firepanda, rows), theirs(rows)
    assert mine.str.contains("(?i:a)+b").tolist() == mask_of(them.str.contains("(?i:a)+b"))
    assert mine.str.contains("(?i:a)+b").tolist() == [True, False, True, False]
