"""`searchsorted` on a column and an index, checked against pandas.

pandas asks numpy where each value would have to go for the values to stay in
order. One value answers one position and several answer one each, whether
they come as a list, an index, a series or a numpy array, and `sorter` reads
the values in its order first.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

KEYS = [1, 3, 5, 7]


def agrees(got: Any, want: Any) -> None:
    """The same position, or the same positions."""
    if hasattr(want, "tolist") and getattr(want, "ndim", 0) > 0:
        assert list(got) == want.tolist()
    else:
        assert got == int(want)


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.Series(KEYS).searchsorted(5),
    lambda m: m.Series(KEYS).searchsorted(5, side="right"),
    lambda m: m.Series(KEYS).searchsorted([0, 5, 1000]),
    lambda m: m.Series(KEYS).searchsorted((0, 5)),
    lambda m: m.Series(KEYS).searchsorted(m.Series([0, 5])),
    lambda m: m.Series(KEYS).searchsorted(m.Index([0, 5])),
    lambda m: m.Series(KEYS).searchsorted(4.5),
    lambda m: m.Series(KEYS, index=[9, 8, 7, 6]).searchsorted(2),
    lambda m: m.Series([1.5, 3.5]).searchsorted([1, 2, 4]),
    lambda m: m.Series(["a", "c", "e"]).searchsorted("d"),
    lambda m: m.Series(["a", "c", "e"]).searchsorted(["d", "z"], side="right"),
    lambda m: m.Series([7, 1, 5]).searchsorted(5, sorter=[1, 2, 0]),
    lambda m: m.Series([7, 1, 5]).searchsorted([0, 6], sorter=[1, 2, 0]),
    lambda m: m.Series([], dtype="int64").searchsorted(5),
    lambda m: m.Index(KEYS).searchsorted(m.Series([0, 5])),
    lambda m: m.Index([7, 1, 5]).searchsorted(5, side="right", sorter=[1, 2, 0]),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """One value and several, both sides, text, a sorter and nothing to search."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_a_numpy_array_is_several_values(firepanda: ModuleType) -> None:
    """A numpy array of values answers a position for each."""
    import numpy as np

    wanted = np.array([0, 5])
    assert list(firepanda.Series(KEYS).searchsorted(wanted)) == [0, 2]
    assert list(firepanda.Index(KEYS).searchsorted(wanted)) == [0, 2]


def test_several_values_answer_an_array(firepanda: ModuleType) -> None:
    """An array of positions with no labels, as pandas answers a numpy array."""
    answer = firepanda.Series(KEYS).searchsorted([0, 5])
    assert type(answer).__name__ == "FirepandaArray"
    assert answer.ndim == 1
    assert str(answer.dtype) == "int64"


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.Series(KEYS).searchsorted(5, side="x"),
    lambda m: m.Index(KEYS).searchsorted(5, side="middle"),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class or a subclass of it, and the same message."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize("owner", ["Series", "Index"])
def test_the_signature_is_pandas_signature(firepanda: ModuleType, owner: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(firepanda, owner).searchsorted).parameters
    yours = inspect.signature(getattr(pd, owner).searchsorted).parameters
    assert [(p.name, p.kind, p.default) for p in ours.values()] == [
        (p.name, p.kind, p.default) for p in yours.values()
    ]
