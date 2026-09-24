"""`pandas.get_dummies`, checked against pandas.

Every test builds the same answer in both libraries and compares the labels,
the row labels, the types and the flags.
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

KEYS = ["a", "b", None, "", "a", None, "", "b", "c", None]
FRAME = {
    "k": ["x", "y", "x", None, "z"],
    "n": [1, 2, 3, 2, 1],
    "b": [True, False, True, True, False],
    "f": [0.5, 1.5, 0.5, None, 2.0],
}


def agrees(got: Any, want: Any, kept: tuple[str, ...] = tuple(FRAME)) -> None:
    """The same labels, row labels and values as pandas, and the same type for every flag.

    A column the frame had before is compared for its values only, since a text
    column's type is named differently in the two libraries.
    """
    assert list(got.columns) == [str(label) for label in want.columns]
    assert list(got.index) == list(want.index)
    for label in want.columns:
        if label not in kept:
            assert str(got[str(label)].dtype) == str(want[label].dtype), label
        mine = [None if value != value else value for value in got[str(label)].tolist()]
        theirs = [None if value != value else value for value in want[label].tolist()]
        assert mine == theirs, label


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.get_dummies(m.Series(KEYS, name="key")),
    lambda m: m.get_dummies(m.Series(KEYS), dummy_na=True, prefix="p"),
    lambda m: m.get_dummies(m.Series(KEYS), drop_first=True),
    lambda m: m.get_dummies(m.Series(KEYS), drop_first=True, dummy_na=True, prefix="p"),
    lambda m: m.get_dummies(m.Series(KEYS), prefix="x", prefix_sep="-"),
    lambda m: m.get_dummies(m.Series(KEYS), dtype=int),
    lambda m: m.get_dummies(m.Series(KEYS), dtype="int8"),
    lambda m: m.get_dummies(m.Series(KEYS), dtype="float32"),
    lambda m: m.get_dummies(m.Series(KEYS), dtype=float),
    lambda m: m.get_dummies(m.Series(KEYS, index=range(10, 20))),
    lambda m: m.get_dummies(m.Series(KEYS).astype("category")),
    lambda m: m.get_dummies(m.Series(KEYS).astype("category"), dummy_na=True, prefix="c"),
    lambda m: m.get_dummies(m.Series(KEYS).astype("category"), drop_first=True),
    lambda m: m.get_dummies(m.Series(["only"])),
    lambda m: m.get_dummies(m.Series(["only"]), drop_first=True),
    lambda m: m.get_dummies(m.Series([1.5, 2.0, 1.5]), prefix="v"),
    lambda m: m.get_dummies(m.Series([True, False]), prefix="t"),
    lambda m: m.get_dummies(["a", "b", "a"]),
    lambda m: m.get_dummies(m.DataFrame(FRAME)),
    lambda m: m.get_dummies(m.DataFrame(FRAME), dummy_na=True),
    lambda m: m.get_dummies(m.DataFrame(FRAME), drop_first=True),
    lambda m: m.get_dummies(m.DataFrame(FRAME), columns=["k", "n"]),
    lambda m: m.get_dummies(m.DataFrame(FRAME), columns=["n"], prefix="N"),
    lambda m: m.get_dummies(m.DataFrame(FRAME), columns=["n", "f"], prefix=["N", "F"]),
    lambda m: m.get_dummies(m.DataFrame(FRAME), prefix={"k": "K"}),
    lambda m: m.get_dummies(m.DataFrame(FRAME), prefix_sep="-"),
    lambda m: m.get_dummies(m.DataFrame(FRAME), columns=["k", "n"], prefix_sep=["-", "+"]),
    lambda m: m.get_dummies(m.DataFrame(FRAME), columns=["k", "n"], prefix={"k": "K", "n": "N"}),
    lambda m: m.get_dummies(m.DataFrame(FRAME), dtype="int64"),
    lambda m: m.get_dummies(m.DataFrame(FRAME)[["n", "b"]]),
    lambda m: m.get_dummies(m.DataFrame(FRAME).assign(c=m.Series(KEYS[:5]).astype("category"))),
    lambda m: m.get_dummies(m.DataFrame(FRAME).set_index("n")),
    lambda m: m.get_dummies(m.DataFrame(FRAME), columns=[]),
]


@pytest.mark.parametrize("build", BUILDS)
def test_dummies_are_pandas_dummies(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Text, categories, numbers with a prefix, frames, and every parameter."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.get_dummies(m.DataFrame(FRAME), columns="k"),
    lambda m: m.get_dummies(m.DataFrame(FRAME), prefix=["P", "Q", "R"]),
    lambda m: m.get_dummies(m.DataFrame(FRAME), prefix_sep=["-", "+"]),
    lambda m: m.get_dummies(m.Series(KEYS), dtype=object),
    lambda m: m.get_dummies(m.DataFrame(FRAME), columns=["k", "n"], prefix_sep={"k": "."}),
    lambda m: m.get_dummies(m.DataFrame(FRAME), columns=["k", "n"], prefix={"k": "K", "f": "F"}),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class, or a subclass of it, and the same first line."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value).split("\n")[0] == str(theirs.value).split("\n")[0]


REFUSED: list[Callable[[Any], Any]] = [
    lambda m: m.get_dummies(m.Series(KEYS), sparse=True),
    lambda m: m.get_dummies(m.Series([1.5, 2.0])),
    lambda m: m.get_dummies(m.Series(KEYS), dummy_na=True),
    lambda m: m.get_dummies(m.Series([True, False])),
]


@pytest.mark.parametrize("build", REFUSED)
def test_what_is_not_written_is_refused(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """A sparse answer, and a column label that would not be text."""
    with pytest.raises(NotImplementedError):
        build(firepanda)


def test_the_signature_is_pandas_signature(firepanda: ModuleType) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(firepanda.get_dummies).parameters
    yours = inspect.signature(pd.get_dummies).parameters
    assert [(p.name, p.kind) for p in ours.values()] == [(p.name, p.kind) for p in yours.values()]
    for each in ours:
        assert ours[each].default == yours[each].default, each
