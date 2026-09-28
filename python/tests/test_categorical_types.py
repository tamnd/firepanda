"""`Categorical`, `CategoricalDtype` and `CategoricalIndex`, compared with pandas.

Each is a thin shell over a category column: the type carries the categories
and the order flag, the array prints as pandas prints it, and the index is an
index of category.
"""

from __future__ import annotations

import copy
import importlib.util
import pickle
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def column(m: ModuleType) -> Any:
    return m.Series(["b", "a", None]).astype(m.CategoricalDtype(["b", "a", "c"], ordered=True))


ANSWERS: dict[str, Callable[[ModuleType], Any]] = {
    "a dtype": lambda m: repr(m.CategoricalDtype(["a", "b"])),
    "an ordered dtype": lambda m: repr(m.CategoricalDtype(["b", "a"], ordered=True)),
    "a dtype with no categories": lambda m: repr(m.CategoricalDtype()),
    "a dtype's categories": lambda m: m.CategoricalDtype(["b", "a"]).categories.tolist(),
    "a dtype's name": lambda m: (str(m.CategoricalDtype()), m.CategoricalDtype().name),
    "a dtype is the word": lambda m: m.CategoricalDtype(["a"]) == "category",
    "unordered in any order": lambda m: (
        m.CategoricalDtype(["a", "b"]) == m.CategoricalDtype(["b", "a"])
    ),
    "ordered in its order": lambda m: (
        m.CategoricalDtype(["a", "b"], True) == m.CategoricalDtype(["b", "a"], True)
    ),
    "ordered and not": lambda m: m.CategoricalDtype(["a"], True) == m.CategoricalDtype(["a"]),
    "a column's dtype": lambda m: repr(column(m).dtype),
    "a column's categories": lambda m: column(m).dtype.categories.tolist(),
    "a column's order": lambda m: column(m).dtype.ordered,
    "a column cast": lambda m: repr(column(m)),
    "a frame cast": lambda m: repr(
        m.DataFrame({"k": ["a", "b"], "v": [1, 2]})
        .astype({"k": m.CategoricalDtype(["b", "a"])})["k"]
        .dtype
    ),
    "a categorical": lambda m: repr(m.Categorical(["b", "a", "b", None], ordered=True)),
    "a categorical with categories": lambda m: repr(
        m.Categorical(["b", "a", "z"], categories=["b", "a", "c"])
    ),
    "a categorical of a dtype": lambda m: repr(
        m.Categorical(["b", "a"], dtype=m.CategoricalDtype(["a", "b"], ordered=True))
    ),
    "a long categorical": lambda m: repr(m.Categorical(list("abcdefghijkl"))),
    "an empty categorical": lambda m: repr(m.Categorical([])),
    "codes": lambda m: repr(m.Categorical(["b", "a", None], categories=["a", "b"]).codes),
    "from codes": lambda m: repr(m.Categorical.from_codes([0, 1, -1], ["x", "y"])),
    "from codes and a dtype": lambda m: repr(
        m.Categorical.from_codes([1, 0], dtype=m.CategoricalDtype(["x", "y"], ordered=True))
    ),
    "a categorical's parts": lambda m: (
        m.Categorical(["b", "a"]).categories.tolist(),
        m.Categorical(["b", "a"]).ordered,
        len(m.Categorical(["b", "a"])),
    ),
    "a categorical changed": lambda m: repr(m.Categorical(["b", "a"]).add_categories(["c"])),
    "a categorical compared": lambda m: (m.Categorical(["b", "a"]) == "a").tolist(),
    "a series of a categorical": lambda m: repr(
        m.Series(m.Categorical(["b", "a"], categories=["b", "a", "c"]))
    ),
    "a categorical index": lambda m: repr(m.CategoricalIndex(["b", "a"], name="n")),
    "a categorical index with categories": lambda m: repr(
        m.CategoricalIndex(["b", "a"], categories=["a", "b", "c"], ordered=True)
    ),
    "a categorical index's parts": lambda m: (
        m.CategoricalIndex(["b", "a", "b"]).categories.tolist(),
        m.CategoricalIndex(["b", "a", "b"]).codes.tolist(),
    ),
    "a copied dtype": lambda m: repr(copy.deepcopy(m.CategoricalDtype(["a"]))),
    "a pickled dtype": lambda m: repr(
        pickle.loads(pickle.dumps(m.CategoricalDtype(["a"], ordered=True)))
    ),
}

MISTAKES: dict[str, Callable[[ModuleType], Any]] = {
    "repeated categories": lambda m: m.CategoricalDtype(["a", "a"]),
    "a missing category": lambda m: m.CategoricalDtype(["a", None]),
    "a code past the categories": lambda m: m.Categorical.from_codes([0, 2], ["x", "y"]),
    "codes with no categories": lambda m: m.Categorical.from_codes([0]),
    "a dtype and categories": lambda m: m.Categorical(
        ["a"], categories=["a"], dtype=m.CategoricalDtype(["a"])
    ),
}


def spelled(answer: Any) -> Any:
    """firepanda's text type is `string` where pandas prints `str`."""
    return answer.replace("dtype: string", "dtype: str") if isinstance(answer, str) else answer


@needs_pandas
@pytest.mark.parametrize("name", list(ANSWERS))
def test_an_answer_matches_pandas(firepanda: ModuleType, name: str) -> None:
    """The same value, or the same text where the answer is printed."""
    import pandas as pd

    assert spelled(ANSWERS[name](firepanda)) == ANSWERS[name](pd)


@needs_pandas
@pytest.mark.parametrize("name", list(MISTAKES))
def test_a_mistake_raises_what_pandas_raises(firepanda: ModuleType, name: str) -> None:
    """The same kind of error and the same words."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        MISTAKES[name](pd)
    with pytest.raises(theirs.type) as mine:
        MISTAKES[name](firepanda)
    assert str(mine.value) == str(theirs.value)


def test_the_dtype_is_still_the_word(firepanda: ModuleType) -> None:
    """Code that compares a column's type with `"category"` keeps working."""
    kind = column(firepanda).dtype
    assert isinstance(kind, str)
    assert kind == "category"
    assert str(kind) == "category"
