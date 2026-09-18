"""A lookahead, which is the first construct RE2 has not got that this has.

Every flag question this accessor can ask has been answered for a while and the
constructs were what was left. There are five of them, RE2 refuses all five, and
pandas knows it: a pattern holding one never reaches Arrow at all, it is
compiled with `re` and looped over in Python, and the answer comes back a
perfectly ordinary column. This library routed those patterns the same way and
then refused them, so a caller writing `(?=...)` got an exception where pandas
gave a column.

The half that lands here is the lookahead. A lookbehind is still refused, and it
is a different question rather than a harder version of the same one, since
reading one means knowing the width of the body and raising for a body that has
not got a fixed width. Document 93.

Nothing here passes a flag. That is the point worth reading: the route is
decided by what the pattern holds rather than by what was passed beside it, so
these calls reach Python's engine with an empty `flags` and would have reached
Arrow without the brackets. The row that says so directly is the last one.

The rows compare against pandas rather than against a written down answer,
because the claim of the slice is agreement and not correctness in the abstract.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ROWS = ["ab", "ac", "a", "ba", "abab", "", None]
"""Rows picked so that a lookahead decides most of them.

The first two are the pair a lookahead was invented for, one where the letter
ahead is the one asked about and one where it is not. The single letter is the
row where there is nothing ahead at all, which is the case a negative lookahead
gets right and a plain pattern cannot say anything about. The reversed pair puts
the letter in front rather than behind. The repeated pair is the row that counts
more than once. The empty row and the missing one are the two every text file
here carries.
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
    as `engine/string-predicate-null` and a lookahead changes nothing about it.
    """
    return values[:-1]


@needs_pandas
def test_the_positive_form_agrees_with_pandas_row_by_row(
    firepanda: ModuleType,
) -> None:
    """The call that used to raise. `a(?=b)` is an `a` with a `b` after it and
    the `b` is not consumed, which is visible in the count rather than in the
    mask and is why the count is here beside it."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains("a(?=b)").tolist()
    assert without_the_missing(ours) == without_the_missing(mask_of(them.str.contains("a(?=b)")))
    assert without_the_missing(ours) == [True, False, False, False, True, False]
    assert mine.str.count("a(?=b)").tolist() == counts_of(them.str.count("a(?=b)"))


@needs_pandas
def test_the_negative_form_agrees_as_well(firepanda: ModuleType) -> None:
    """`a(?!b)` holds on the row where there is nothing after the `a` at all,
    which is the case that separates a negative lookahead from a plain `a[^b]`
    and is why the single letter row is in the list."""
    mine, them = made(firepanda), theirs()
    ours = mine.str.contains("a(?!b)").tolist()
    assert without_the_missing(ours) == without_the_missing(mask_of(them.str.contains("a(?!b)")))
    assert without_the_missing(ours) == [False, True, True, True, False, False]


@needs_pandas
def test_the_width_of_one_is_nothing_so_the_replacement_says_so(
    firepanda: ModuleType,
) -> None:
    """A lookahead is a question about a position rather than a piece of text,
    so what it asked about is still there afterwards. `a(?=b)` replaced by a
    hash turns `ab` into `#b` and not into `#`."""
    mine, them = made(firepanda), theirs()
    assert mine.str.replace("a(?=b)", "#", regex=True).tolist() == texts_of(
        them.str.replace("a(?=b)", "#", regex=True)
    )
    assert mine.str.replace("a(?=b)", "#", regex=True).tolist()[0] == "#b"


@needs_pandas
def test_the_anchored_pair_take_it_too(firepanda: ModuleType) -> None:
    """`match` and `fullmatch` are answered by writing anchors around the
    caller's pattern, and a lookahead inside those anchors is the same
    instruction in a different place."""
    mine, them = made(firepanda), theirs()
    assert without_the_missing(mine.str.match("a(?=b)").tolist()) == without_the_missing(
        mask_of(them.str.match("a(?=b)"))
    )
    assert without_the_missing(mine.str.fullmatch("a(?=b)b").tolist()) == without_the_missing(
        mask_of(them.str.fullmatch("a(?=b)b"))
    )


@needs_pandas
def test_extract_takes_a_group_beside_one_but_not_inside_one(
    firepanda: ModuleType,
) -> None:
    """A group inside a lookahead keeps what it matched upstream, and the second
    machine this engine runs the body with carries no slots, so that shape is
    refused rather than answered wrongly. A group beside one is ordinary."""
    mine, them = made(firepanda), theirs()
    assert mine.str.extract("(a)(?=b)").iloc[:, 0].tolist() == texts_of(
        them.str.extract("(a)(?=b)")[0]
    )
    assert them.str.extract("(?=(a))a")[0].tolist()[0] == "a"
    with pytest.raises(NotImplementedError):
        mine.str.extract("(?=(a))a")


@needs_pandas
def test_an_empty_negative_lookaround_is_a_pattern_and_not_an_error(
    firepanda: ModuleType,
) -> None:
    """`(?!)` never matches, which upstream reads as a pattern rather than as a
    mistake. Only the methods that never reach Arrow can say so, because the
    parser collapses it into a node the router does not recognise as an
    assertion, so `contains` goes to Arrow on both sides and raises on both.
    Both of those raises are a `ValueError` underneath, pandas' through
    `ArrowInvalid` and this library's through `InvalidArgumentError`, which is
    as close as the two get and is not an accident on either side."""
    mine, them = made(firepanda), theirs()
    assert mine.str.extract("((?!))").iloc[:, 0].tolist() == texts_of(them.str.extract("((?!))")[0])
    with pytest.raises(ValueError):
        them.str.contains("(?!)")
    with pytest.raises(ValueError):
        mine.str.contains("(?!)")


@needs_pandas
def test_a_lookbehind_is_the_half_that_is_still_refused(
    firepanda: ModuleType,
) -> None:
    """Named as a gap rather than as a refusal, because pandas answers it. The
    row is here so that the day it lands is a day this file fails."""
    mine, them = made(firepanda), theirs()
    assert without_the_missing(mask_of(them.str.contains("(?<=a)b"))) == [
        True,
        False,
        False,
        False,
        True,
        False,
    ]
    with pytest.raises(NotImplementedError):
        mine.str.contains("(?<=a)b")


@needs_pandas
def test_the_brackets_are_what_moved_the_call_off_arrow(
    firepanda: ModuleType,
) -> None:
    """No flag is passed anywhere in this file and every call above reached
    Python's engine anyway, because the route is decided by what the pattern
    holds. The same pattern without the brackets is a pattern Arrow answers, and
    the two give the same column here for a reason that has nothing to do with
    either engine: they are asking different questions that agree on these
    rows."""
    mine, them = made(firepanda), theirs()
    assert mine.str.contains("ab").tolist() == mine.str.contains("a(?=b)b").tolist()
    assert without_the_missing(mine.str.contains("a(?=b)b").tolist()) == (
        without_the_missing(mask_of(them.str.contains("a(?=b)b")))
    )
