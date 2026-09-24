"""`pivot` and `pivot_table`, checked against pandas.

Each value of one column becomes a column of its own and each value of another
a row, both sorted. `pivot` refuses two rows with the same pair, and
`pivot_table` aggregates them, by name or with a Python function called on each
group. A pair no row holds is missing, so whole numbers with a gap become
floats unless `fill_value` fills it.
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


def base(m: ModuleType) -> Any:
    """Five rows with three row keys, three column keys and one of each type."""
    return m.DataFrame(
        {
            "r": ["b", "a", "b", "a", "c"],
            "c": ["x", "x", "y", "z", "y"],
            "v": [1, 2, 3, 4, 5],
            "w": [1.5, 2.5, 3.5, 4.5, 5.5],
            "t": ["p", "q", "r", "s", "u"],
        }
    )


def twice(m: ModuleType) -> Any:
    """Every row of `base` twice, so each pair has two rows."""
    return m.concat([base(m), base(m)])


def shown(frame: Any) -> Any:
    """A frame as plain Python, with its types and labels."""
    return (
        [str(name) for name in frame.columns],
        [str(kind).replace("string", "str") for kind in frame.dtypes],
        [
            ["missing" if value is None or value != value else value for value in frame[name]]
            for name in frame.columns
        ],
        frame.index.tolist(),
        frame.index.name,
    )


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: base(m).pivot(index="r", columns="c", values="v"),
    lambda m: base(m).pivot(index="r", columns="c", values="w"),
    lambda m: base(m).pivot(index="r", columns="c", values="t"),
    lambda m: base(m).pivot(columns="c", values="v"),
    lambda m: base(m).pivot(index="c", columns="r", values="v"),
    lambda m: base(m).pivot(index=["r"], columns=["c"], values="v"),
    lambda m: m.pivot(base(m), index="r", columns="c", values="v"),
    lambda m: base(m).pivot_table(index="r", columns="c", values="v"),
    lambda m: base(m).pivot_table(index="r", columns="c", values="v", aggfunc="sum"),
    lambda m: base(m).pivot_table(index="r", columns="c", values="v", aggfunc="sum", fill_value=0),
    lambda m: twice(m).pivot_table(index="r", columns="c", values="v", aggfunc="sum"),
    lambda m: twice(m).pivot_table(index="r", columns="c", values="w", aggfunc="count"),
    lambda m: twice(m).pivot_table(index="r", columns="c", values="w", aggfunc="max"),
    lambda m: twice(m).pivot_table(index="r", columns="c", values="v", fill_value=0),
    lambda m: base(m).pivot_table(index="r", values="v"),
    lambda m: base(m).pivot_table(index="r", values=["v", "w"], aggfunc="sum"),
    lambda m: base(m).pivot_table(index="r", columns="c", values="v", sort=False),
    lambda m: m.pivot_table(base(m), index="r", columns="c", values="v", aggfunc="min"),
    lambda m: m.DataFrame(
        {"r": ["a", None, "a"], "c": ["x", "x", None], "v": [1, 2, 3]}
    ).pivot_table(index="r", columns="c", values="v"),
    lambda m: base(m).pivot_table(
        index="r", columns="c", values="v", aggfunc=lambda s: s.max() - s.min()
    ),
    lambda m: base(m).pivot_table(index="r", values="v", aggfunc=lambda s: s.max() - s.min()),
    lambda m: base(m).pivot_table(index="r", values=["v", "w"], aggfunc=len),
    lambda m: twice(m).pivot_table(
        index="r", columns="c", values="w", aggfunc=lambda s: s.sum() / 2
    ),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Each type, gaps, fills, repeats, names, functions, order and missing keys."""
    import pandas as pd

    assert shown(build(firepanda)) == shown(build(pd))


def test_a_repeated_pair_is_pandas_mistake(firepanda: ModuleType) -> None:
    """pivot does not aggregate, so two rows with one pair are refused."""
    import pandas as pd

    with pytest.raises(ValueError) as theirs:
        twice(pd).pivot(index="r", columns="c", values="v")
    with pytest.raises(ValueError) as mine:
        twice(firepanda).pivot(index="r", columns="c", values="v")
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize(
    "build",
    [
        lambda m: base(m).pivot(index="r", columns="c"),
        lambda m: base(m).pivot(index="r", columns="c", values=["v", "w"]),
        lambda m: base(m).pivot(index=["r", "t"], columns="c", values="v"),
        lambda m: base(m).pivot(index="c", columns="v", values="w"),
        lambda m: base(m).pivot_table(index="r", columns="c", values="v", margins=True),
        lambda m: base(m).pivot_table(index="r", columns="c", values="v", aggfunc=["sum"]),
        lambda m: base(m).pivot_table(index="r", columns="c", values=["v", "w"]),
    ],
)
def test_what_pandas_labels_with_levels_or_numbers_is_refused(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """A MultiIndex, margins and columns named by numbers are not firepanda's."""
    with pytest.raises(NotImplementedError):
        build(firepanda)


@pytest.mark.parametrize(
    ("mine", "yours"),
    [
        ("DataFrame.pivot", "DataFrame.pivot"),
        ("DataFrame.pivot_table", "DataFrame.pivot_table"),
        ("pivot", "pivot"),
        ("pivot_table", "pivot_table"),
    ],
)
def test_the_signature_is_pandas_signature(firepanda: ModuleType, mine: str, yours: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    def found(module: Any, path: str) -> Any:
        for part in path.split("."):
            module = getattr(module, part)
        return inspect.signature(module).parameters

    ours = found(firepanda, mine)
    theirs = found(pd, yours)
    assert [(p.name, p.kind, repr(p.default)) for p in ours.values()] == [
        (p.name, p.kind, repr(p.default)) for p in theirs.values()
    ]
