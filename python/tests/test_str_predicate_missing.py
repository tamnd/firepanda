"""A string question asked of a missing row, checked against a running pandas.

The kernel answers null there, since the question has no answer, and pandas
answers False. pandas' default `str` dtype carries a missing row as NaN and every
predicate on the accessor fills that row with False, even with `na=None` written
out, and fills it with `na` when `na` is something else. The accessor gives
pandas' answer, so the tests are every predicate, the prefix tuple, the three
pattern questions down both the byte search and the engine, and `na` in each
spelling.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ROWS = ["Ab", None, "cd", "12", " ", None, "Title Case"]
CASES = [
    "isspace",
    "islower",
    "isupper",
    "istitle",
    "isascii",
    "isalpha",
    "isnumeric",
    "isdigit",
    "isdecimal",
    "isalnum",
]


def both(firepanda: ModuleType) -> tuple[Any, Any]:
    """The same column in both libraries."""
    import pandas as pd

    return firepanda.Series(ROWS), pd.Series(ROWS)


@pytest.mark.parametrize("name", CASES)
def test_a_question_about_case_answers_false(firepanda: ModuleType, name: str) -> None:
    """The ten questions that take no argument."""
    mine, theirs = both(firepanda)
    assert getattr(mine.str, name)().tolist() == getattr(theirs.str, name)().tolist()


@pytest.mark.parametrize("name", ["startswith", "endswith"])
@pytest.mark.parametrize("pat", ["A", "d", ("A", "c"), ()])
@pytest.mark.parametrize("na", [None, True, False])
def test_a_prefix_or_a_suffix(firepanda: ModuleType, name: str, pat: Any, na: Any) -> None:
    """One string or a tuple of them, with `na` left alone and written out."""
    mine, theirs = both(firepanda)
    want = getattr(theirs.str, name)(pat, na=na).tolist()
    assert getattr(mine.str, name)(pat, na=na).tolist() == want
    if na is None:
        assert getattr(mine.str, name)(pat).tolist() == want


@pytest.mark.parametrize("name", ["contains", "match", "fullmatch"])
@pytest.mark.parametrize("pat", ["A", "a", "[Ac]", "d$"])
@pytest.mark.parametrize("case", [True, False])
@pytest.mark.parametrize("na", [None, True])
def test_a_pattern(firepanda: ModuleType, name: str, pat: str, case: bool, na: Any) -> None:
    """A literal goes to the byte search and a class to the engine, and both fill."""
    mine, theirs = both(firepanda)
    want = getattr(theirs.str, name)(pat, case=case, na=na).tolist()
    assert getattr(mine.str, name)(pat, case=case, na=na).tolist() == want


def test_a_mask_with_nothing_missing_is_unchanged(firepanda: ModuleType) -> None:
    """The fill is skipped when there is nothing to fill."""
    assert firepanda.Series(["a", "B"]).str.isupper().tolist() == [False, True]


def test_a_filled_mask_is_bool_and_selects_rows(firepanda: ModuleType) -> None:
    """The answer has no gap left in it, so it can pick rows out of the column."""
    mine, theirs = both(firepanda)
    picked = mine[mine.str.startswith("A")].tolist()
    assert picked == theirs[theirs.str.startswith("A")].tolist()
    assert mine.str.startswith("A").hasnans is False
