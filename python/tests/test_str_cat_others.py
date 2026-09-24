"""`str.cat` with `others`, joining columns row by row, checked against pandas.

pandas lines the others up against the column first, by position when every
label list is the column's own and by label under `join` otherwise, and then
joins each row with `sep`. A row with a missing piece is missing, unless
`na_rep` stands in for the piece.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

WORDS = ["a", None, "c", "d"]
LABELS = [3, 1, 7, 5]
OTHER = ["x", "y", None]
OTHER_LABELS = [5, 9, 1]


def plain(values: list[Any]) -> list[Any]:
    """The values with every spelling of missing as one word."""
    return ["missing" if value is None or value != value else value for value in values]


def agrees(got: Any, want: Any) -> None:
    """The same values, labels, index name, name and text type."""
    assert plain(got.tolist()) == plain(want.tolist())
    assert list(got.index) == list(want.index)
    assert got.index.name == want.index.name
    assert got.name == want.name
    assert str(got.dtype) in ("string", "str")


def words(m: ModuleType) -> Any:
    """The column the others are joined to."""
    return m.Series(WORDS, index=LABELS, name="w")


def other(m: ModuleType) -> Any:
    """A column labelled partly like `words`, in another order."""
    return m.Series(OTHER, index=OTHER_LABELS)


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: words(m).str.cat(words(m).str.upper(), sep="/"),
    lambda m: words(m).str.cat(words(m), na_rep="-"),
    lambda m: words(m).str.cat(["1", "2", "3", "4"], sep=","),
    lambda m: words(m).str.cat(m.DataFrame({"p": ["1", "2", "3", "4"], "q": list("wxyz")})),
    lambda m: words(m).str.cat(m.Index(["1", "2", "3", "4"])),
    lambda m: words(m).str.cat(other(m)),
    lambda m: words(m).str.cat(other(m), na_rep="-"),
    lambda m: words(m).str.cat(other(m), join="right", na_rep="-"),
    lambda m: words(m).str.cat(other(m), join="inner", na_rep="-"),
    lambda m: words(m).str.cat(other(m), join="outer", na_rep="-"),
    lambda m: words(m).str.cat([other(m), m.Series(["q"], index=[8])], join="right", na_rep="-"),
    lambda m: words(m).str.cat([other(m), m.Series(["q"], index=[5])], join="inner", na_rep="-"),
    lambda m: words(m).str.cat([other(m), m.Series(["q"], index=[8])], join="outer", na_rep="-"),
    lambda m: m.Series(["a", "b"], index=m.Index([0, 1], name="ix"), name=4).str.cat(
        m.Series(["c"], index=[1]), na_rep="-"
    ),
    lambda m: words(m).str.cat(words(m), join="nonsense is not read when the labels agree"),
    lambda m: m.Series(["a", "b"]).str.cat(m.Series([None, None], dtype="float64")),
    lambda m: m.Series([], dtype="string").str.cat(m.Series([], dtype="string")),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Lists, frames, an index, the four joins, several others, gaps and na_rep."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_numpy_arrays_are_pieces(firepanda: ModuleType) -> None:
    """A one dimensional array is one column and a two dimensional one is several."""
    import numpy as np
    import pandas as pd

    flat = np.array(["1", "2", "3", "4"])
    square = np.array([["1", "2"], ["3", "4"], ["5", "6"], ["7", "8"]])
    for pieces in (flat, square, [flat, flat]):
        agrees(words(firepanda).str.cat(pieces, na_rep="-"), words(pd).str.cat(pieces, na_rep="-"))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: words(m).str.cat("x"),
    lambda m: words(m).str.cat(["a"]),
    lambda m: words(m).str.cat([["a", "b", "c", "d"]]),
    lambda m: words(m).str.cat([words(m), ["1", "2", "3", "4"]]),
    lambda m: words(m).str.cat(5),
    lambda m: words(m).str.cat(other(m), join="bogus"),
    lambda m: words(m).str.cat(m.Series([1, 2, 3, 4], index=LABELS)),
    lambda m: words(m).str.cat(m.Series([1.5, None, 2.5, 3.5], index=LABELS)),
    lambda m: words(m).str.cat(m.Series([1.5, None, 2.5, 3.5], index=LABELS), na_rep="-"),
    lambda m: words(m).str.cat(m.Series([True, False, True, True], index=LABELS)),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class and the same message."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value) == str(theirs.value)


def test_the_fold_is_unchanged(firepanda: ModuleType) -> None:
    """With no others the column still folds into one string."""
    assert words(firepanda).str.cat(sep="-") == "a-c-d"
    assert words(firepanda).str.cat(sep="-", na_rep="?") == "a-?-c-d"


@pytest.mark.parametrize("name", [4, 2.5, ("a", 1), None, "w"])
def test_a_name_given_to_the_constructor_is_kept(firepanda: ModuleType, name: Any) -> None:
    """A number or a tuple comes back as itself, as it does from `rename`."""
    import pandas as pd

    assert firepanda.Series(["a"], name=name).name == pd.Series(["a"], name=name).name
    assert type(firepanda.Series(["a"], index=[3], name=name).name) is type(name)
