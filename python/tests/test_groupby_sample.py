"""`groupby(...).sample`, checked against pandas for the same seeds.

pandas walks the groups in order and draws each one's positions from one
random state with numpy's `choice`, so for the same seed both libraries draw
the same rows in the same order, and refuse the same sizes with the same words.
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

WEIGHTS = [1, 1, 1, 0, 0, 1, 2, 5]
UNEVEN = [0, 1, 0, 1, 1, 0, 0, 1]


def _frame(module: ModuleType) -> Any:
    return module.DataFrame(
        {
            "a": ["red", "blue", "red", "black", "blue", "black", "red", None],
            "b": [0, 1, 2, 3, 4, 5, 6, 7],
            "w": [1.0, 1, 1, 0, 0, 1, 2, 5],
        }
    )


CALLS: dict[str, Callable[[Any], Any]] = {
    "one each": lambda df: df.groupby("a").sample(random_state=1),
    "with replacement": lambda df: df.groupby("a").sample(n=2, replace=True, random_state=7),
    "a share": lambda df: df.groupby("a").sample(frac=0.5, random_state=3),
    "more than all": lambda df: df.groupby("a").sample(frac=2, replace=True, random_state=3),
    "one column": lambda df: df.groupby("a")["b"].sample(n=1, random_state=11),
    "weights": lambda df: df.groupby("a").sample(n=1, weights=WEIGHTS, random_state=1),
    "weights by name": lambda df: df.groupby("a").sample(n=1, weights="w", random_state=1),
    "unsorted": lambda df: df.groupby("a", sort=False).sample(random_state=2),
    "missing key kept": lambda df: df.groupby("a", dropna=False).sample(random_state=2),
    "none": lambda df: df.groupby("a").sample(n=0),
    "empty frame": lambda df: df.iloc[:0].groupby("a").sample(),
}

MISTAKES: dict[str, Callable[[Any], Any]] = {
    "too many": lambda df: df.groupby("a").sample(n=3, random_state=1),
    "both sizes": lambda df: df.groupby("a").sample(n=1, frac=0.5),
    "negative": lambda df: df.groupby("a").sample(n=-1),
    "a fraction of a row": lambda df: df.groupby("a").sample(n=1.5),
    "upsampled": lambda df: df.groupby("a").sample(frac=1.5),
    "no weight": lambda df: df.groupby("a").sample(n=1, weights=UNEVEN, random_state=1),
    "too heavy": lambda df: df.groupby("a").sample(n=2, weights=UNEVEN, random_state=1),
}


@pytest.mark.parametrize("call", CALLS)
def test_the_rows_drawn_are_the_ones_pandas_draws(firepanda: ModuleType, call: str) -> None:
    """The same rows, in the same order, with the same labels."""
    import pandas as pd

    ours, theirs = CALLS[call](_frame(firepanda)), CALLS[call](_frame(pd))
    assert ours.index.tolist() == theirs.index.tolist()
    assert repr(ours) == repr(theirs)


@pytest.mark.parametrize("call", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, call: str) -> None:
    """The same class of error, with the same words."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        MISTAKES[call](_frame(pd))
    with pytest.raises(type(theirs.value)) as ours:
        MISTAKES[call](_frame(firepanda))
    assert str(ours.value) == str(theirs.value)
