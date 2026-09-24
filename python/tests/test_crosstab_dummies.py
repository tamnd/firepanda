"""`crosstab`, `from_dummies` and `lreshape`, checked against pandas.

`crosstab` counts each pair of keys, or runs a function over the values of
each pair, with totals and normalising on request. `from_dummies` reads the
columns `get_dummies` makes back into categories, and `lreshape` stacks groups
of columns into one column each.
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


def plain(values: list[Any]) -> list[Any]:
    """The values with every spelling of missing as one word, and floats rounded."""
    return [
        "missing"
        if value is None or value != value
        else round(value, 12)
        if isinstance(value, float)
        else value
        for value in values
    ]


def agrees(got: Any, want: Any) -> None:
    """The same labels, index name, columns, values and types."""
    assert plain(list(got.index)) == plain(list(want.index))
    assert got.index.name == want.index.name
    assert list(got.columns) == list(want.columns)
    for name in want.columns:
        assert plain(got[name].tolist()) == plain(want[name].tolist()), name
        assert str(got[name].dtype).replace("string", "str") == str(want[name].dtype), name


def keys(m: ModuleType) -> tuple[Any, Any, Any]:
    """Two keys with a gap each and values to aggregate."""
    return (
        m.Series(["x", "y", "x", "x", None, "z"], name="a"),
        m.Series(["p", "q", "q", "p", "p", None], name="b"),
        m.Series([1, 2, 3, 4, 5, 6]),
    )


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.crosstab(*keys(m)[:2]),
    lambda m: m.crosstab(*keys(m)[:2], margins=True),
    lambda m: m.crosstab(*keys(m)[:2], margins=True, margins_name="Total"),
    lambda m: m.crosstab(*keys(m)[:2], normalize=True),
    lambda m: m.crosstab(*keys(m)[:2], normalize="index"),
    lambda m: m.crosstab(*keys(m)[:2], normalize="columns"),
    lambda m: m.crosstab(*keys(m)[:2], normalize=0),
    lambda m: m.crosstab(*keys(m)[:2], normalize=1, margins=True),
    lambda m: m.crosstab(*keys(m)[:2], normalize="index", margins=True),
    lambda m: m.crosstab(*keys(m)[:2], normalize="all", margins=True),
    lambda m: m.crosstab(m.Series(["x", None, "x"]), m.Series(["p", "q", "q"]), dropna=False),
    lambda m: m.crosstab(*keys(m)[:2], rownames=["R"], colnames=["C"]),
    lambda m: m.crosstab(*keys(m), aggfunc="sum"),
    lambda m: m.crosstab(*keys(m), aggfunc="mean", margins=True),
    lambda m: m.crosstab(*keys(m), aggfunc="sum", margins=True),
    lambda m: m.crosstab(*keys(m), aggfunc="max", margins=True, normalize="columns"),
    lambda m: m.crosstab(*keys(m), aggfunc=len),
    lambda m: m.crosstab(*keys(m), aggfunc="count"),
    lambda m: m.crosstab(keys(m)[0], keys(m)[0]),
    lambda m: m.crosstab([1, 2, 1, 1], ["p", "q", "q", "p"]),
    lambda m: m.crosstab([[1, 2, 1]], [["p", "q", "q"]]),
    lambda m: m.crosstab(
        m.Series([1, 2, 1]), m.Series(["p", "q", "q"]), values=[1.5, 2, 3], aggfunc="sum"
    ),
    lambda m: m.crosstab(m.Series(["x", "y"], index=[1, 2]), m.Series(["p", "q"], index=[2, 3])),
    lambda m: m.crosstab(m.Series(["x", "y"], index=[5, 6]), ["p", "q"]),
    lambda m: m.crosstab(m.Series([True, False, True]), m.Series(["p", "p", "q"])),
]


@pytest.mark.parametrize("build", BUILDS)
def test_crosstab_is_pandas_crosstab(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Counts, functions, totals, the four normalisings, gaps and names."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


DUMMIES: list[Callable[[Any], Any]] = [
    lambda m: m.from_dummies(m.DataFrame({"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]})),
    lambda m: m.from_dummies(m.DataFrame({"a": [True, False], "b": [False, True]})),
    lambda m: m.from_dummies(m.DataFrame({"a": [1.0, 0.0], "b": [0.0, 1.0]})),
    lambda m: m.from_dummies(m.DataFrame({"a": [1, 0], "b": [0, 0]}), default_category="z"),
    lambda m: m.from_dummies(m.DataFrame({"a": [1, 0], "b": [0, 0]}), default_category=1),
    lambda m: m.from_dummies(m.DataFrame({"a": [1, 0], "b": [0, 1]}), default_category=1),
    lambda m: m.from_dummies(m.DataFrame({"a": [1, 0], "b": [0, 1]}, index=[5, 6])),
    lambda m: m.from_dummies(
        m.DataFrame({"x_a": [1, 0], "x_b": [0, 1], "y_c": [1, 0], "y_d": [0, 1]}), sep="_"
    ),
    lambda m: m.from_dummies(
        m.DataFrame({"x_a": [1, 0], "x_b": [0, 0], "y_c": [1, 0], "y_d": [0, 1]}),
        sep="_",
        default_category={"x": "q", "y": "r"},
    ),
    lambda m: m.from_dummies(m.DataFrame({"x_a_1": [1, 0], "x_b": [0, 1]}), sep="_"),
    lambda m: m.from_dummies(m.get_dummies(m.Series(["a", "b", "a"]))),
    lambda m: m.from_dummies(
        m.get_dummies(m.DataFrame({"k": ["u", "v"]}))[["k_u", "k_v"]], sep="_"
    ),
]


@pytest.mark.parametrize("build", DUMMIES)
def test_from_dummies_is_pandas_from_dummies(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """Ones and zeros, true and false, defaults, separators and labels."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def wide(m: ModuleType) -> Any:
    """A frame with two groups of columns to stack and two to repeat."""
    return m.DataFrame(
        {
            "k": ["u", "v"],
            "id": [1, 2],
            "a1": [1.0, None],
            "a2": [3.0, 4.0],
            "b1": [5, 6],
            "b2": [7, 8],
        }
    )


LONG: list[Callable[[Any], Any]] = [
    lambda m: m.lreshape(wide(m), {"a": ["a1", "a2"], "b": ["b1", "b2"]}),
    lambda m: m.lreshape(wide(m), {"a": ["a1", "a2"], "b": ["b1", "b2"]}, dropna=False),
    lambda m: m.lreshape(wide(m), {"b": ["b1", "b2"]}),
    lambda m: m.lreshape(wide(m), {"n": ["b1", "a2"]}),
]


@pytest.mark.parametrize("build", LONG)
def test_lreshape_is_pandas_lreshape(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Stacked groups, repeated columns in sorted order, and gaps left out or kept."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.crosstab(*keys(m)[:2], aggfunc="sum"),
    lambda m: m.crosstab(*keys(m)),
    lambda m: m.crosstab(*keys(m)[:2], normalize=2),
    lambda m: m.crosstab(*keys(m)[:2], normalize="rows"),
    lambda m: m.crosstab(*keys(m)[:2], rownames=["x", "y"]),
    lambda m: m.from_dummies(m.DataFrame({"a": [1, 0], "b": [0, 0]})),
    lambda m: m.from_dummies(m.DataFrame({"a": [1, 1], "b": [0, 1]})),
    lambda m: m.from_dummies(m.DataFrame({"a": [1.0, None]})),
    lambda m: m.from_dummies(m.DataFrame({"a": ["x", "y"]})),
    lambda m: m.from_dummies(m.DataFrame({"a": [2, 0], "b": [0, 1]})),
    lambda m: m.from_dummies(m.DataFrame({"a": [1, 0], "b": [0, 0]}, index=["r", "s"])),
    lambda m: m.from_dummies(m.DataFrame({"x_a": [1, 0], "b": [0, 1]}), sep="_"),
    lambda m: m.from_dummies(m.DataFrame({"a": [1, 0], "b": [0, 1]}), sep=1),
    lambda m: m.from_dummies(m.DataFrame({"a": [1, 0], "b": [0, 1]}), default_category=["q"]),
    lambda m: m.from_dummies(
        m.DataFrame({"x_a": [1, 0], "y_b": [0, 1]}), sep="_", default_category={"x": "q"}
    ),
    lambda m: m.from_dummies([1, 0]),
    lambda m: m.lreshape(wide(m), {"a": ["a1", "a2"], "b": ["b1"]}),
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


REFUSED: list[Callable[[Any], Any]] = [
    lambda m: m.crosstab([keys(m)[0], keys(m)[1]], keys(m)[1]),
    lambda m: m.crosstab(keys(m)[1], keys(m)[0], dropna=False),
    lambda m: m.crosstab(m.Series([1, 2]), m.Series(["p", "q"]), margins=True),
    lambda m: m.crosstab(m.Series(["p", "q"]), m.Series([1, 2])),
]


@pytest.mark.parametrize("build", REFUSED)
def test_what_firepanda_cannot_hold_is_refused(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """Several keys, a missing or numeric column key, and totals among numeric labels."""
    with pytest.raises(NotImplementedError, match="crosstab"):
        build(firepanda)


@pytest.mark.parametrize("name", ["crosstab", "from_dummies", "lreshape"])
def test_the_signature_is_pandas_signature(firepanda: ModuleType, name: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(firepanda, name)).parameters
    yours = inspect.signature(getattr(pd, name)).parameters
    assert [(p.name, p.kind, p.default) for p in ours.values()] == [
        (p.name, p.kind, p.default) for p in yours.values()
    ]
