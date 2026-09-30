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


def gappy(m: ModuleType) -> Any:
    """`base` with gaps in `w`, which the totals leave out and the body keeps."""
    return base(m).assign(w=[1.5, None, 3.5, None, 5.5])


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
    lambda m: base(m).pivot_table(index="r", values=["w", "v"], aggfunc="sum"),
    lambda m: base(m).pivot_table(index="r", values=["w", "v"], aggfunc="sum", sort=False),
    lambda m: base(m).pivot_table(index="r", columns="c", values="v", margins=True),
    lambda m: twice(m).pivot_table(
        index="r", columns="c", values="v", aggfunc="sum", fill_value=0, margins=True
    ),
    lambda m: gappy(m).pivot_table(
        index="r", columns="c", values="w", aggfunc="count", margins=True, margins_name="T"
    ),
    lambda m: gappy(m).pivot_table(index="r", values=["v", "w"], aggfunc="sum", margins=True),
    lambda m: gappy(m).pivot_table(index="r", values="w", margins=True, sort=False),
    lambda m: gappy(m).pivot_table(index="r", values="w"),
    lambda m: gappy(m).pivot_table(index="r", columns="c", values="w"),
    lambda m: gappy(m).pivot_table(index="r", values="w", dropna=False),
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


LEVELS: list[Callable[[Any], Any]] = [
    lambda m: base(m).pivot(index="r", columns="c"),
    lambda m: base(m).pivot(index="r", columns="c", values=["v", "w"]),
    lambda m: base(m).pivot(columns="c", values=["w", "v"]),
    lambda m: base(m).pivot_table(index="r", columns="c", values="v", aggfunc=["sum"]),
    lambda m: base(m).pivot_table(index="r", columns="c", values="v", aggfunc=["sum", "max"]),
    lambda m: base(m).pivot_table(index="r", columns="c", values=["w", "v"]),
    lambda m: base(m).pivot_table(index="r", columns="c", values=["v"], aggfunc="sum"),
    lambda m: base(m).pivot_table(index="r", columns="c", values=["w", "v"], sort=False),
    lambda m: base(m)[["r", "c", "v", "w"]].pivot_table(index="r", columns="c", aggfunc="sum"),
    lambda m: base(m).pivot_table(index="r", values=["v", "w"], aggfunc=["sum", "mean"]),
    lambda m: base(m).pivot_table(index="r", columns="c", values="v", aggfunc={"v": "sum"}),
    lambda m: base(m).pivot_table(
        index="r", columns="c", aggfunc={"v": "sum", "w": ["min", "max"]}
    ),
    lambda m: base(m).pivot_table(
        index="r", columns="c", values="v", aggfunc=[lambda s: s.sum()], fill_value=0
    ),
    lambda m: base(m).pivot_table(
        index="r", columns="c", values="v", aggfunc=["sum", "max"], margins=True
    ),
    lambda m: base(m).pivot_table(index="r", columns="c", values="v", aggfunc=["sum"]).columns,
]


@pytest.mark.parametrize("build", LEVELS)
def test_columns_of_several_levels_are_pandas(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """Several values or functions put the key's columns under each, with the level names."""
    import pandas as pd

    assert repr(build(firepanda)) == repr(build(pd))


@pytest.mark.parametrize(
    "build",
    [
        lambda m: (
            base(m)
            .assign(r=[1, 2, 1, 2, 3])
            .pivot_table(index="r", columns="c", values="v", margins=True)
        ),
        lambda m: (
            base(m)
            .assign(c=[1, 1, 2, 3, 2])
            .pivot_table(index="r", columns="c", values="v", margins=True)
        ),
        lambda m: base(m).pivot_table(index="r", columns="c", values=["v", "w"], margins=True),
        lambda m: base(m).pivot(index="r", columns="c", values=["v", "t"]),
    ],
)
def test_what_pandas_labels_with_numbers_or_objects_is_refused(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """Totals beside numbers, columns named by numbers and values read as objects."""
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
