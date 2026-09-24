"""`str.split`, `str.rsplit`, `str.join` and `str.wrap`, checked against pandas.

A split with `expand=True` is a frame with one column per piece, as wide as the
row cut into the most pieces, with gaps padding the shorter rows. pandas labels
those columns with integers and a frame here labels them with the text of the
integer, so the labels are compared as text. `join` on text puts the separator
between characters and `wrap` hands the row to `textwrap`.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def answer(value: Any) -> Any:
    """Labels as text, the index, and the values with gaps as None."""

    def plain(cell: Any) -> Any:
        return None if cell is None or type(cell).__name__ == "NAType" or cell != cell else cell

    if hasattr(value, "columns"):
        return (
            [str(c) for c in value.columns],
            value.index.tolist(),
            [[plain(v) for v in value[c].tolist()] for c in value.columns],
        )
    return value.index.tolist(), value.name, [plain(v) for v in value.tolist()]


def dashed(m: ModuleType) -> Any:
    """Text cut by dashes, with a gap, an empty row and a doubled dash."""
    return m.Series(["a-b-c", "d", None, "", "x--y"], index=[5, 6, 7, 8, 9], name="t")


def spaced(m: ModuleType) -> Any:
    """Text with runs of whitespace at the ends and between words."""
    return m.Series(["a  b c", " lead", None, "tab\there"])


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: dashed(m).str.split("-", expand=True),
    lambda m: dashed(m).str.split("-", n=1, expand=True),
    lambda m: dashed(m).str.split("-", n=0, expand=True),
    lambda m: dashed(m).str.split("-", n=None, expand=True),
    lambda m: dashed(m).str.split("--", expand=True),
    lambda m: dashed(m).str.split("-", expand=True, regex=True),
    lambda m: dashed(m).str.rsplit("-", expand=True),
    lambda m: dashed(m).str.rsplit("-", n=1, expand=True),
    lambda m: dashed(m).str.rsplit(pat="-", n=0, expand=True),
    lambda m: spaced(m).str.split(expand=True),
    lambda m: spaced(m).str.split(n=1, expand=True),
    lambda m: spaced(m).str.rsplit(n=1, expand=True),
    lambda m: m.Series(["a1b22c", "zz"]).str.split(r"\d+", expand=True),
    lambda m: m.Series(["a.b", "ab"]).str.split(".", expand=True),
    lambda m: m.Series(["a.b", "ab"]).str.split(".", expand=True, regex=True),
    lambda m: m.Series(["a.b", "a.*b"]).str.split(".*", expand=True, regex=False),
    lambda m: m.Series(["aXbxc"]).str.split(re.compile("x", re.I), expand=True),
    lambda m: m.Series(["a-b", "c"]).str.split("", expand=True),
    lambda m: m.Series([None, None], dtype="string").str.split("-", expand=True),
    lambda m: m.Series([], dtype="string").str.split("-", expand=True),
    lambda m: dashed(m).str.join("+"),
    lambda m: dashed(m).str.join(""),
    lambda m: m.Series(["the quick brown fox", None, "ab", ""]).str.wrap(5),
    lambda m: m.Series(["the quick brown fox"], name="w").str.wrap(9, break_long_words=False),
    lambda m: m.Series(["a\tb  c"]).str.wrap(4),
    lambda m: m.Series(["one two three four"]).str.wrap(8, max_lines=2, placeholder=".."),
    lambda m: m.Series(["hyphen-ated words"]).str.wrap(7, break_on_hyphens=False),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every pattern kind, count, gap and wrapping option."""
    import pandas as pd

    assert answer(build(firepanda)) == answer(build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: dashed(m).str.split(3, expand=True),
    lambda m: dashed(m).str.split(re.compile("-"), expand=True, regex=False),
    lambda m: dashed(m).str.split("-", n=1.5, expand=True),
    lambda m: dashed(m).str.split("-", expand="yes"),
    lambda m: dashed(m).str.rsplit("", expand=True),
    lambda m: dashed(m).str.rsplit(3, expand=True),
    lambda m: dashed(m).str.join(3),
    lambda m: dashed(m).str.wrap(0),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_mistakes_raise_as_pandas_raises(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The same error type and words as pandas."""
    import pandas as pd

    with pytest.raises(Exception) as expected:
        build(pd)
    with pytest.raises(type(expected.value), match="^" + re.escape(str(expected.value)) + "$"):
        build(firepanda)


@pytest.mark.parametrize("name", ["split", "rsplit"])
def test_a_column_of_lists_is_refused_by_name(firepanda: ModuleType, name: str) -> None:
    """Without expand the answer is a column of lists, which has no column type yet."""
    with pytest.raises(NotImplementedError, match="expand=False"):
        getattr(dashed(firepanda).str, name)("-")
