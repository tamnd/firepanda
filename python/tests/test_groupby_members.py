"""The grouping made visible, checked against pandas.

`ngroups`, `ndim`, `len`, iterating, `groups`, `indices`, `get_group` and
`pipe`, on a frame's group by and on a column's. A frame is compared by its
columns, labels and values, and a dictionary of groups by its keys and the
labels or positions in each.
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


def plain(value: Any) -> Any:
    """A value with every missing one as None, so nan equals nan."""
    if isinstance(value, tuple):
        return tuple(plain(v) for v in value)
    if value is None or value != value:
        return None
    return value


def facts(answer: Any) -> Any:
    """What is compared, whatever shape the answer has."""
    if isinstance(answer, dict):
        return [(plain(k), [int(v) for v in list(v)]) for k, v in answer.items()]
    if isinstance(answer, list):
        return [facts(v) for v in answer]
    if isinstance(answer, tuple):
        return tuple(facts(v) for v in answer)
    if isinstance(answer, str):
        return answer
    if hasattr(answer, "columns"):
        return (
            list(answer.columns),
            answer.index.tolist(),
            [[plain(v) for v in answer[c].tolist()] for c in answer.columns],
        )
    if hasattr(answer, "index"):
        return answer.name, answer.index.tolist(), [plain(v) for v in answer.tolist()]
    return plain(answer)


def frame(m: ModuleType) -> Any:
    """Two keys, one of them with a gap, over labels that are not positions."""
    return m.DataFrame(
        {
            "k": ["b", "a", "b", None, "a"],
            "j": [1, 1, 2, 2, 1],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=[10, 11, 12, 13, 14],
    )


def by(m: ModuleType, keys: Any = "k", **kw: Any) -> Any:
    """The frame grouped."""
    return frame(m).groupby(keys, **kw)


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: by(m).ngroups,
    lambda m: by(m, ["k", "j"]).ngroups,
    lambda m: by(m, dropna=False).ngroups,
    lambda m: by(m).ndim,
    lambda m: by(m)["v"].ndim,
    lambda m: by(m)[["v"]].ndim,
    lambda m: len(by(m)),
    lambda m: len(by(m)["v"]),
    lambda m: len(frame(m).iloc[:0].groupby("k")),
    lambda m: by(m).groups,
    lambda m: by(m, ["k", "j"]).groups,
    lambda m: by(m, dropna=False).groups,
    lambda m: by(m, sort=False).groups,
    lambda m: by(m)["v"].groups,
    lambda m: by(m).indices,
    lambda m: by(m, ["k", "j"], sort=False).indices,
    lambda m: (
        m.DataFrame({"k": ["b", "c", "a", "c"], "j": [2, 1, 1, 0]})
        .groupby(["k", "j"], sort=False)
        .indices
    ),
    lambda m: by(m, dropna=False).indices,
    lambda m: [k for k, _ in by(m)],
    lambda m: [k for k, _ in by(m, ["k"])],
    lambda m: list(by(m)),
    lambda m: list(by(m, ["k", "j"])),
    lambda m: list(by(m, ["k", "j"], sort=False)),
    lambda m: list(by(m, dropna=False)),
    lambda m: list(by(m)["v"]),
    lambda m: list(by(m)[["v"]]),
    lambda m: list(by(m, as_index=False)),
    lambda m: by(m).get_group("a"),
    lambda m: by(m, ["k", "j"]).get_group(("b", 2)),
    lambda m: by(m, ["k"]).get_group(("a",)),
    lambda m: by(m, dropna=False).get_group(float("nan")),
    lambda m: by(m)["v"].get_group("b"),
    lambda m: by(m)[["v"]].get_group("a"),
    lambda m: by(m, as_index=False).get_group("a"),
    lambda m: by(m).pipe(lambda g: g.ngroups),
    lambda m: by(m).pipe(lambda g, n: g.sum() * n, 2),
    lambda m: by(m).pipe(lambda g, n=1: g.max() + n, n=3),
    lambda m: by(m)["v"].pipe(lambda g: g.max()),
    lambda m: by(m).pipe((lambda n, g: g.min() - n, "g"), 1),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every member on one key, two keys, a key with a gap, and both orders."""
    import pandas as pd

    assert facts(build(firepanda)) == facts(build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: by(m).get_group("z"),
    lambda m: by(m, ["k", "j"]).get_group(("b", 9)),
    lambda m: by(m, ["k"]).get_group("a"),
    lambda m: by(m).get_group(["a"]),
    lambda m: by(m)["v"].get_group("z"),
    lambda m: by(m).pipe((lambda g: g, "g"), g=1),
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


def test_the_groups_are_labels_and_the_indices_positions(firepanda: ModuleType) -> None:
    """`groups` holds an index of labels and `indices` an array of positions."""
    grouped = by(firepanda)
    assert isinstance(grouped.groups["a"], firepanda.Index)
    assert grouped.groups["a"].tolist() == [11, 14]
    assert list(grouped.indices["a"]) == [1, 4]
