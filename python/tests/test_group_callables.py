"""A Python function run once a group, through `agg`, `transform` and `apply`.

Each is compared with pandas on the types, the labels, the name and every
value. `agg` hands the function a column's values in the group, named after the
column, and reads the type from what comes back. `transform` and `apply` name a
column's group after its key. A row whose key is missing is in no group, so
`transform` answers missing there. `apply` answers one row a group for one value
or one series a group, and the rows in the frame's order for answers that keep
each group's rows when `group_keys=False`, stacking any other answers group
after group. With `group_keys=True` the group's key goes in front of each
label, which the tests of labels with levels compare.
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


def data(m: ModuleType) -> Any:
    return m.DataFrame(
        {"k": ["b", "a", "b", None, "a"], "v": [1, 2, 3, 4, 6], "w": [1.5, 2.0, 3.0, 4.0, 5.0]}
    )


def two(m: ModuleType) -> Any:
    return m.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "y", "x", "x"], "v": [1.0, 2, 3, 4]})


def spread(value: Any) -> Callable[[Any], Any]:
    return lambda s: [value] * len(s)


def keyed(m: ModuleType) -> Callable[[Any], Any]:
    return lambda s: m.Series([str(s.name)] * len(s), index=s.index)


def plus(s: Any, a: int, b: int = 0) -> Any:
    return s.max() + a + b


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "agg mean of whole numbers": lambda m: data(m).groupby("k")["v"].agg(lambda s: s.mean()),
    "agg count of floats": lambda m: data(m).groupby("k")["w"].agg(lambda s: len(s)),
    "agg flags": lambda m: data(m).groupby("k")["v"].agg(lambda s: s.sum() > 3),
    "agg text": lambda m: data(m).groupby("k")["v"].agg(lambda s: "x" * len(s)),
    "agg sees the column name": lambda m: data(m).groupby("k")["v"].agg(lambda s: s.name),
    "aggregate": lambda m: data(m).groupby("k")["v"].aggregate(lambda s: s.iloc[-1]),
    "agg over a frame": lambda m: data(m).groupby("k").agg(lambda s: s.max() - s.min()),
    "agg over a selection": lambda m: data(m).groupby("k")[["w"]].agg(lambda s: s.sum()),
    "agg keys as columns": lambda m: data(m).groupby("k", as_index=False).agg(lambda s: s.max()),
    "agg column keys as columns": lambda m: (
        data(m).groupby("k", as_index=False)["v"].agg(lambda s: s.min())
    ),
    "agg list with lambdas": lambda m: (
        data(m).groupby("k")["v"].agg([lambda s: s.max(), "sum", lambda s: s.min()])
    ),
    "agg list with a builtin": lambda m: data(m).groupby("k")["v"].agg(["sum", len]),
    "agg mapping": lambda m: data(m).groupby("k").agg({"v": lambda s: s.max(), "w": "sum"}),
    "agg named": lambda m: data(m).groupby("k").agg(top=("v", lambda s: s.max())),
    "agg arguments": lambda m: data(m).groupby("k")["v"].agg(plus, 10, b=5),
    "agg unsorted": lambda m: data(m).groupby("k", sort=False)["v"].agg(lambda s: s.iloc[0]),
    "agg missing key kept": lambda m: (
        data(m).groupby("k", dropna=False)["v"].agg(lambda s: s.sum())
    ),
    "agg two keys as columns": lambda m: (
        two(m).groupby(["a", "b"], as_index=False)["v"].agg(lambda s: s.sum())
    ),
    "transform centred": lambda m: data(m).groupby("k")["v"].transform(lambda s: s - s.mean()),
    "transform whole numbers": lambda m: data(m).groupby("k")["v"].transform(lambda s: s * 2),
    "transform one value": lambda m: data(m).groupby("k")["v"].transform(lambda s: s.max()),
    "transform sees the key": lambda m: data(m).groupby("k")["v"].transform(keyed(m)),
    "transform a list": lambda m: data(m).groupby("k")["v"].transform(spread(7)),
    "transform arguments": lambda m: data(m).groupby("k")["v"].transform(lambda s, n: s + n, 3),
    "transform missing key kept": lambda m: (
        data(m).groupby("k", dropna=False)["v"].transform(lambda s: s * 2)
    ),
    "transform unsorted": lambda m: (
        data(m).groupby("k", sort=False)["v"].transform(lambda s: s.cumsum())
    ),
    "transform two keys": lambda m: (
        two(m).groupby(["a", "b"])["v"].transform(lambda s: len(str(s.name)))
    ),
    "transform a frame": lambda m: data(m).groupby("k").transform(lambda d: d - d.mean()),
    "transform a frame to one value": lambda m: data(m).groupby("k").transform(lambda d: 1),
    "transform a frame to a row": lambda m: data(m).groupby("k").transform(lambda d: d.sum()),
    "apply one value": lambda m: data(m).groupby("k")["v"].apply(lambda s: s.max()),
    "apply sees the key": lambda m: data(m).groupby("k")["v"].apply(lambda s: s.name),
    "apply a row": lambda m: data(m).groupby("k")[["v"]].apply(lambda d: d.sum()),
    "apply a row of mixed columns": lambda m: data(m).groupby("k").apply(lambda d: d.sum()),
    "apply one value a frame": lambda m: data(m).groupby("k").apply(lambda d: len(d)),
    "apply keeping rows": lambda m: (
        data(m).groupby("k", group_keys=False)["v"].apply(lambda s: s * 2)
    ),
    "apply keeping a frame's rows": lambda m: (
        data(m).groupby("k", group_keys=False).apply(lambda d: d * 2)
    ),
    "apply keeping a selection's rows": lambda m: (
        data(m).groupby("k", group_keys=False)[["w"]].apply(lambda d: d.head(1))
    ),
    "apply a column of a frame's rows": lambda m: (
        data(m).groupby("k", group_keys=False).apply(lambda d: d["v"] * 2)
    ),
    "apply the first row of each group": lambda m: (
        data(m).groupby("k", group_keys=False).apply(lambda d: d.head(1))
    ),
    "apply the first value of each group": lambda m: (
        data(m).groupby("k", group_keys=False)["v"].apply(lambda s: s.head(1))
    ),
    "apply keys as columns": lambda m: (
        data(m).groupby("k", as_index=False)["v"].apply(lambda s: s.max())
    ),
    "apply a row keys as columns": lambda m: (
        data(m).groupby("k", as_index=False).apply(lambda d: d.sum())
    ),
    "apply by name": lambda m: data(m).groupby("k").apply("sum"),
    "apply arguments": lambda m: data(m).groupby("k")["v"].apply(plus, 1, b=2),
}


def shown(answer: Any) -> Any:
    """The kind, the types, the labels, the name and the values, with text named `str`."""
    if hasattr(answer, "dtypes") and not hasattr(answer, "dtype"):
        types = [str(kind) for kind in answer.dtypes.tolist()]
        values = {name: answer[name].tolist() for name in answer.columns}
        name = None
    else:
        types = [str(answer.dtype)]
        values = answer.tolist()
        name = answer.name
    types = [kind.replace("string", "str") for kind in types]
    return (
        type(answer).__name__,
        types,
        repr(answer.index.tolist()),
        answer.index.name,
        name,
        repr(values),
    )


@needs_pandas
@pytest.mark.parametrize("name", list(CASES))
def test_a_function_over_the_groups_matches_pandas(firepanda: ModuleType, name: str) -> None:
    """The kind, the types, the labels, the name and every value."""
    import pandas as pd

    assert shown(CASES[name](firepanda)) == shown(CASES[name](pd))


MISTAKES: dict[str, Callable[[ModuleType], Any]] = {
    "agg answering a series": lambda m: data(m).groupby("k")["v"].agg(lambda s: s * 2),
    "transform list of the wrong length": lambda m: (
        data(m).groupby("k")["v"].transform(lambda s: [1, 2, 3])
    ),
    "apply of something not callable": lambda m: data(m).groupby("k").apply(5),
    "apply including the keys": lambda m: (
        data(m).groupby("k").apply(lambda d: d, include_groups=True)
    ),
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


@needs_pandas
@pytest.mark.parametrize(
    "build",
    [
        lambda m: data(m).groupby("k", as_index=False).apply(lambda d: len(d)),
        lambda m: data(m).groupby("k", as_index=False).apply(lambda d: d["w"].sum()),
        lambda m: data(m).groupby("k", as_index=False, sort=False).apply(lambda d: "x" * len(d)),
    ],
)
def test_one_value_a_group_goes_in_a_column_named_none(firepanda: ModuleType, build: Any) -> None:
    """pandas puts one value a group beside the keys, in a column it names None."""
    import pandas as pd

    assert repr(build(firepanda)) == repr(build(pd))


def test_the_groups_are_handed_out_without_the_keys(firepanda: ModuleType) -> None:
    """pandas 3 leaves the keys out of what `apply` and `transform` hand the function."""
    seen: list[list[str]] = []

    def look(group: Any) -> int:
        seen.append(list(group.columns))
        return 0

    data(firepanda).groupby("k").apply(look)
    assert seen == [["v", "w"], ["v", "w"]]
