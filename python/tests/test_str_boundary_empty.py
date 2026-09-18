"""`\\B` on a row with nothing in it, which is a question about which Python.

CPython up to 3.13 fails a `\\B` on an empty subject. 3.14 took the case out and
made `\\B` the plain negation of `\\b`, which is what RE2 has always had and
what every other engine has always had. `pixi.toml` says this project supports
3.12 and up, so both answers are live and neither of them is the answer, and a
pattern is compiled for a version of Python rather than for Python. Document 90
is the whole of it.

That is why nothing below spells the expected value for the empty row. It is
computed by asking the running `re` the same question, which is the only honest
way to write a row whose answer depends on which interpreter opened the file,
and it is the same thing this library does at its own door.

Every pattern here carries a flag, because `\\B` with no flag beside it goes to
Arrow and RE2 asks the word boundary question between bytes rather than between
characters, so this library refuses it there rather than answering it wrongly.
The refusal is asserted in a row of its own. `re.MULTILINE` is the flag used to
move the call, since it says nothing about a word boundary and nothing about an
empty row, so it moves the route and nothing else.
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
    "",
    " ",
    "a",
    "ab",
    "  a  ",
    "aé",
    "éa",
    "é",
    "a\nb",
    None,
]
"""Rows picked so that the empty one is not the only thing being read.

The single character and the pair are the two shapes a boundary can be asked
about at all, the row of spaces around a letter is four non boundaries in five
characters and is what a count has to get right, and the accented letter beside
a plain one is where the two word classes part company, which is the difference
the ascii flag moves and the empty row's answer does not.
"""

EMPTY_IS_A_MATCH = re.search("\\B", "") is not None
"""What the running interpreter says about the row this file is named after.

False up to 3.13 and True from 3.14. Asked rather than spelled, and asked once
at import rather than in each row, so that a reader who wants to know which
interpreter a run was made on has one place to look.
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
    as `engine/string-predicate-null` and a word boundary changes nothing about
    it.
    """
    return values[:-1]


@needs_pandas
def test_the_empty_row_is_whatever_the_interpreter_says(firepanda: ModuleType) -> None:
    """The row the slice is about. pandas answers it with `re` and this library
    answers it with its own engine, and the two agree because this library asks
    the interpreter it is running in rather than the one it was written on."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains("\\B", flags=re.MULTILINE).tolist()
    assert without_the_missing(ours) == without_the_missing(
        mask_of(them.str.contains("\\B", flags=re.MULTILINE))
    )
    assert ours[0] == EMPTY_IS_A_MATCH
    assert ours[1:4] == [True, False, True]


@needs_pandas
def test_the_positive_half_never_had_the_case(firepanda: ModuleType) -> None:
    """`\\b` fails on an empty row in every version, because there is nothing
    there for a boundary to be between. The case that changed was only ever on
    the negative half, which is what kept the two from being a pair."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains("\\b", flags=re.MULTILINE).tolist()
    assert without_the_missing(ours) == without_the_missing(
        mask_of(them.str.contains("\\b", flags=re.MULTILINE))
    )
    assert ours[0] is False
    assert ours[2] is True


@needs_pandas
def test_the_alphabet_has_nothing_to_say_about_an_empty_row(
    firepanda: ModuleType,
) -> None:
    """`(?a)` narrows which characters count as word characters, and a row that
    holds none of any kind is not a question about characters at all. Both
    spellings of the letter are asked, since one is read by the parser and the
    other arrives as a number beside the pattern."""
    mine, them = made(firepanda), theirs()
    for pattern, flags in [
        ("\\B", re.MULTILINE | re.ASCII),
        ("(?a)\\B", re.MULTILINE),
    ]:
        ours = mine.str.contains(pattern, flags=flags).tolist()
        assert without_the_missing(ours) == without_the_missing(
            mask_of(them.str.contains(pattern, flags=flags))
        )
        assert ours[0] == EMPTY_IS_A_MATCH


@needs_pandas
def test_the_alphabet_still_moves_the_rows_that_hold_something(
    firepanda: ModuleType,
) -> None:
    """The two alphabets are a real difference and writing the empty row as a
    question about the row has not flattened it. An e-acute is a word character
    to the wide reading and is not to the narrow one, so the position between an
    `a` and one is a non boundary under the first and a boundary under the
    second."""
    mine, them = made(firepanda), theirs()
    wide = mine.str.contains("a\\B", flags=re.MULTILINE).tolist()
    narrow = mine.str.contains("a\\B", flags=re.MULTILINE | re.ASCII).tolist()
    assert without_the_missing(wide) == without_the_missing(
        mask_of(them.str.contains("a\\B", flags=re.MULTILINE))
    )
    assert without_the_missing(narrow) == without_the_missing(
        mask_of(them.str.contains("a\\B", flags=re.MULTILINE | re.ASCII))
    )
    assert wide[5] is True
    assert narrow[5] is False


@needs_pandas
def test_the_counting_scan_asks_the_empty_row_once(firepanda: ModuleType) -> None:
    """An empty row has one position in it and the scan asks about that one, so
    the count there is whatever a single `\\B` says. The row of spaces around a
    letter is the one worth reading twice: four non boundaries in five
    characters, because the two spaces on each side of the letter are one word
    character away from a run of spaces."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.count("\\B", flags=re.MULTILINE).tolist()
    assert ours == counts_of(them.str.count("\\B", flags=re.MULTILINE))
    assert ours[0] == int(EMPTY_IS_A_MATCH)
    assert ours[4] == 4


@needs_pandas
def test_the_replacing_scan_follows_the_counting_one(firepanda: ModuleType) -> None:
    """One rule for both, which is `re.finditer`, so the marks land where the
    count said they would and the empty row gets a mark or does not by the same
    answer the mask gave."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.replace("\\B", "#", flags=re.MULTILINE, regex=True).tolist()
    assert ours == texts_of(them.str.replace("\\B", "#", flags=re.MULTILINE, regex=True))
    assert ours[0] == ("#" if EMPTY_IS_A_MATCH else "")
    assert ours[3] == "a#b"


@needs_pandas
def test_fullmatch_reads_the_empty_row_the_same_way(firepanda: ModuleType) -> None:
    """`\\B` consumes nothing, so the only row a `fullmatch` of it can match is
    the row with nothing in it, and that makes this method the narrowest place
    the difference shows: every cell of the column is False under one
    interpreter and exactly one cell is True under the next.

    `match` is not asked beside it. pandas raises `Cannot pass flags that do not
    match pat.flags` for `match` under any flag at all, which is an upstream
    defect of its own and is not this one.
    """
    mine, them = made(firepanda), theirs()
    ours = mine.str.fullmatch("\\B", flags=re.MULTILINE).tolist()
    assert without_the_missing(ours) == without_the_missing(
        mask_of(them.str.fullmatch("\\B", flags=re.MULTILINE))
    )
    assert ours[0] == EMPTY_IS_A_MATCH
    assert ours[1:9] == [False] * 8


@needs_pandas
def test_the_boundary_is_not_the_only_thing_in_the_pattern(
    firepanda: ModuleType,
) -> None:
    """The case is attached to the boundary rather than to the pattern, so a
    `\\B` that is not the first thing read still only asks about the row it is
    in, and an alternative beside it is answered without ever reaching it."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains("a|\\B", flags=re.MULTILINE).tolist()
    assert without_the_missing(ours) == without_the_missing(
        mask_of(them.str.contains("a|\\B", flags=re.MULTILINE))
    )
    assert ours[0] == EMPTY_IS_A_MATCH
    assert ours[2] is True


@needs_pandas
def test_re2_has_not_got_a_version_of_python(firepanda: ModuleType) -> None:
    """With no flag beside it the call goes to Arrow, and RE2 asks the word
    boundary question between bytes rather than between characters, so this
    library refuses it there. The refusal is asserted so that the day it becomes
    an answer this test says so, and pandas' own answer is asserted beside it
    because RE2 has always agreed with 3.14 here and disagreed with everything
    before it."""
    mine, them = made(firepanda), theirs()
    assert them.str.contains("\\B").tolist()[0] is True
    with pytest.raises(NotImplementedError):
        mine.str.contains("\\B")
