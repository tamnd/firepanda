"""`sample`, `case_when`, `dot` and `compare`, checked against pandas.

`sample` asks numpy's `choice` for positions as pandas does, so the same seed
draws the same rows. `case_when` moves the column to the type it has in common
with every replacement first. `dot` lines the two sides up by label, and
`compare` sets the values that differ side by side.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def plain(values: list[Any]) -> list[Any]:
    """The values with every spelling of missing as one word."""
    return ["missing" if value is None or value != value else value for value in values]


def shown(answer: Any) -> Any:
    """An answer as plain Python, with its labels and types."""
    if hasattr(answer, "columns"):
        return (
            [str(name) for name in answer.columns],
            [str(kind).replace("string", "str") for kind in answer.dtypes],
            [plain(answer[name].tolist()) for name in answer.columns],
            answer.index.tolist(),
        )
    if hasattr(answer, "index"):
        return (
            plain(answer.tolist()),
            str(answer.dtype).replace("string", "str"),
            answer.index.tolist(),
        )
    return answer.tolist() if hasattr(answer, "tolist") else answer


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.Series([1, 2, 3, 4, 5]).sample(3, random_state=1),
    lambda m: m.Series([1, 2, 3, 4, 5]).sample(frac=0.4, random_state=7, ignore_index=True),
    lambda m: m.Series([1, 2, 3]).sample(5, replace=True, random_state=2),
    lambda m: m.Series([1, 2, 3]).sample(2, weights=[0, 1, 1], random_state=3),
    lambda m: m.Series([1, 2, 3]).sample(1, weights=m.Series([5, 0, 0]), random_state=3),
    lambda m: m.DataFrame({"a": [1, 2, 3], "w": [0.0, None, 1.0]}).sample(
        1, weights="w", random_state=0
    ),
    lambda m: m.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).sample(1, axis=1, random_state=4),
    lambda m: m.Series([1, 2, 3]).sample(random_state=np.random.default_rng(5)),
    lambda m: m.Series([1, 2, 3]).case_when([(m.Series([True, False, False]), 9)]),
    lambda m: m.Series([1, 2, 3]).case_when([(lambda s: s > 1, 0.5), (lambda s: s > 0, 7)]),
    lambda m: m.Series([1, 2, 3]).case_when([(lambda s: s > 5, 0.5)]),
    lambda m: m.Series([1.0, 2, 3]).case_when([(lambda s: s > 1, m.Series([7, 8, 9]))]),
    lambda m: m.Series(["a", "b"]).case_when([(lambda s: s == "a", "z")]),
    lambda m: m.Series([1, 2]).case_when([([True, False], 5), ([True, True], 6)]),
    lambda m: m.Series([1, 2, 3]).dot(m.Series([1, 5, 3])),
    lambda m: m.Series([1, 2, 3]).dot(m.Series([1, 5, 3], index=[2, 1, 0])),
    lambda m: m.Series([1.5, 2]).dot([2, 2]),
    lambda m: m.Series([1, 2]) @ m.DataFrame({"a": [1, 2], "b": [3, 4]}),
    lambda m: [1, 2, 3] @ m.Series([1, 2, 3]),
    lambda m: m.DataFrame({"a": [1, 2], "b": [3, 4]}).dot(m.Series([1, 1], index=["b", "a"])),
    lambda m: (
        m.DataFrame({"a": [1, 2], "b": [3, 4]})
        @ m.DataFrame({"x": [1, 0], "y": [2, 1]}, index=["a", "b"])
    ),
    lambda m: m.DataFrame({"a": [1, 2], "b": [3, 4]}).dot([1, 2]),
    lambda m: m.Series([1, 2, 3]).compare(m.Series([1, 5, 3])),
    lambda m: m.Series([1, 2, 3]).compare(m.Series([1, 5, 3]), keep_shape=True),
    lambda m: m.Series([1, 2, 3]).compare(
        m.Series([1, 5, 3]), keep_equal=True, result_names=("a", "b")
    ),
    lambda m: m.Series([1, 2, 3]).compare(m.Series([1, 5, 3]), keep_equal=True, keep_shape=True),
    lambda m: m.Series([1, 2]).compare(m.Series([3, 4])),
    lambda m: m.Series([1.0, None, 2]).compare(m.Series([1.0, None, 5])),
    lambda m: m.Series(["a", "b"], index=["x", "y"]).compare(
        m.Series(["a", "c"], index=["x", "y"])
    ),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Seeds, weights, sizes, conditions, labels lined up and values set side by side."""
    import pandas as pd

    assert shown(build(firepanda)) == shown(build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.Series([1, 2, 3]).sample(2, frac=0.5),
    lambda m: m.Series([1, 2, 3]).sample(-1),
    lambda m: m.Series([1, 2, 3]).sample(1.5),
    lambda m: m.Series([1, 2, 3]).sample(frac=2),
    lambda m: m.Series([1, 2, 3]).sample(1, weights="x"),
    lambda m: m.Series([1, 2, 3]).sample(1, weights=[1, 1]),
    lambda m: m.Series([1, 2, 3]).sample(1, weights=[0, 0, 0]),
    lambda m: m.Series([1, 2, 3]).sample(3, weights=[1, 0, 0]),
    lambda m: m.Series([1, 2, 3]).sample(1, weights=[1, -1, 0]),
    lambda m: m.Series([1, 2, 3]).sample(1, random_state="x"),
    lambda m: m.DataFrame({"a": [1]}).sample(1, weights="zz"),
    lambda m: m.Series([1, 2]).case_when((1, 2)),
    lambda m: m.Series([1, 2]).case_when([]),
    lambda m: m.Series([1, 2]).case_when([[1, 2]]),
    lambda m: m.Series([1, 2]).case_when([(1, 2, 3)]),
    lambda m: m.Series([1, 2, 3]).dot(m.Series([1, 2], index=[0, 5])),
    lambda m: m.Series([1, 2, 3]).dot([1, 2]),
    lambda m: m.DataFrame({"a": [1, 2], "b": [3, 4]}).dot([1, 2, 3]),
    lambda m: m.DataFrame({"a": [1, 2], "b": [3, 4]}).dot(m.Series([1, 1], index=["a", "c"])),
    lambda m: m.Series([1, 2, 3]).compare(m.Series([1, 2])),
    lambda m: m.Series([1, 2, 3]).compare(m.Series([1, 2, 3]), result_names=["a", "b"]),
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


@pytest.mark.parametrize(
    "build",
    [
        lambda m: m.Series([1, 2]).case_when([(lambda s: s > 1, "x")]),
        lambda m: m.Series([True, False]).compare(m.Series([True, True])),
        lambda m: m.Series([1, 2]).compare(m.Series([1, 3]), align_axis=0),
        lambda m: m.DataFrame({"a": [1, 2]}).dot([[1, 2]]),
    ],
)
def test_what_pandas_answers_with_objects_or_levels_is_refused(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """Objects, a MultiIndex and columns labelled with numbers are not firepanda's."""
    with pytest.raises(NotImplementedError):
        build(firepanda)
