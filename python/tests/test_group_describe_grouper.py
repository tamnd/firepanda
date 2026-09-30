"""`SeriesGroupBy.describe` and `pd.Grouper`, each compared with pandas.

`describe` answers the numbers `Series.describe` gives for each group, as
float64 columns on the keys. A `Grouper` with a key groups by that column, and
on its own it brings its own `sort` and `dropna`, so a plain one leaves the
groups in the order they are first seen. One with a frequency bins a column of
moments as `resample(freq, on=key)` does.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

MOMENTS = [
    "2024-01-01 00:00",
    "2024-01-01 07:00",
    "2024-01-01 14:00",
    "2024-01-01 21:00",
    "2024-01-02 04:00",
    "2024-01-02 11:00",
]


def data(m: ModuleType) -> Any:
    return m.DataFrame(
        {
            "k": ["b", "a", "b", None, "a", "c"],
            "v": [1, 2, 3, 4, 6, 7],
            "w": [1.5, 2.0, 3.0, 4.0, 5.0, None],
            "t": m.to_datetime(MOMENTS),
        }
    )


def keyed(m: ModuleType) -> Any:
    return data(m)[["k", "v"]]


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "describe": lambda m: data(m).groupby("k")["v"].describe(),
    "describe with a group of gaps": lambda m: data(m).groupby("k")["w"].describe(),
    "describe keys as columns": lambda m: data(m).groupby("k", as_index=False)["v"].describe(),
    "describe percentiles": lambda m: data(m).groupby("k")["v"].describe(percentiles=[0.1, 0.9]),
    "describe unsorted with the missing key": lambda m: (
        data(m).groupby("k", sort=False, dropna=False)["v"].describe()
    ),
    "describe two keys as columns": lambda m: (
        m.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"], "v": [1.0, 2.0, 3.0]})
        .groupby(["a", "b"], as_index=False)["v"]
        .describe()
    ),
    "grouper": lambda m: keyed(m).groupby(m.Grouper(key="k")).sum(),
    "grouper sorted": lambda m: keyed(m).groupby(m.Grouper(key="k", sort=True), sort=False).sum(),
    "grouper keeps the missing key": lambda m: (
        data(m).groupby(m.Grouper(key="k", dropna=False))["v"].sum()
    ),
    "grouper in a list": lambda m: data(m).groupby([m.Grouper(key="k")])["v"].sum(),
    "grouper beside a name": lambda m: (
        data(m).groupby([m.Grouper(key="k"), "v"], as_index=False)["w"].sum()
    ),
    "grouper with a frequency": lambda m: data(m).groupby(m.Grouper(key="t", freq="D"))["v"].sum(),
    "grouper with closed bins": lambda m: (
        data(m).groupby(m.Grouper(key="t", freq="12h", closed="right"))["v"].sum()
    ),
}


def shown(answer: Any) -> Any:
    """The kind, the types, the labels and the values, with text named `str`."""
    if hasattr(answer, "dtypes") and not hasattr(answer, "dtype"):
        types = [str(kind) for kind in answer.dtypes.tolist()]
        values = {name: answer[name].tolist() for name in answer.columns}
        name = None
    else:
        types = [str(answer.dtype)]
        values = answer.tolist()
        name = answer.name
    types = [kind.replace("string", "str") for kind in types]
    labels = repr(answer.index.tolist())
    return type(answer).__name__, types, labels, answer.index.name, name, repr(values)


@needs_pandas
@pytest.mark.parametrize("name", list(CASES))
def test_an_answer_matches_pandas(firepanda: ModuleType, name: str) -> None:
    """The kind, the types, the labels and every value."""
    import pandas as pd

    assert shown(CASES[name](firepanda)) == shown(CASES[name](pd))


@needs_pandas
@pytest.mark.parametrize(
    "arguments",
    [
        {"key": "k"},
        {"key": "k", "sort": True, "dropna": False},
        {"key": "t", "freq": "ME"},
        {"key": "t", "freq": "W"},
        {"key": "t", "freq": "3min"},
        {"key": "t", "freq": "D", "closed": "right", "label": "right", "origin": "epoch"},
        {"key": "t", "freq": "h", "offset": "10min"},
    ],
)
def test_a_grouper_prints_as_pandas_prints_it(
    firepanda: ModuleType, arguments: dict[str, Any]
) -> None:
    """The class pandas names, the frequency as an offset, and the ends of the bins."""
    import pandas as pd

    assert repr(firepanda.Grouper(**arguments)) == repr(pd.Grouper(**arguments))


MISTAKES: dict[str, Callable[[ModuleType], Any]] = {
    "a grouper on a missing column": lambda m: data(m).groupby(m.Grouper(key="zz")),
    "a bin keyword with no frequency": lambda m: m.Grouper(key="k", closed="left"),
}


@needs_pandas
@pytest.mark.parametrize("name", list(MISTAKES))
def test_a_mistake_raises_what_pandas_raises(firepanda: ModuleType, name: str) -> None:
    """The same kind of error with the same words."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        MISTAKES[name](pd)
    with pytest.raises(type(theirs.value)) as mine:
        MISTAKES[name](firepanda)
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize(
    "build",
    [
        lambda m: data(m).groupby("v")["k"].describe(),
        lambda m: data(m).groupby("k")["t"].describe(),
        lambda m: data(m).groupby([m.Grouper(key="t", freq="D"), "k"]).sum(),
    ],
)
def test_what_is_not_written_is_refused(firepanda: ModuleType, build: Any) -> None:
    """Text and moments have no numbers to describe, and the row labels are not a key yet."""
    with pytest.raises(NotImplementedError):
        build(firepanda)
