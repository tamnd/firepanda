"""`ffill`, `bfill`, `pct_change` and `filter` on a group by, checked against pandas.

A fill is a running largest position within each group and a gather, a change
is a division by the group's own shift, and a filter runs the function once a
group and keeps the rows of the groups it answers True for. Every test builds
the same answer in both libraries and compares the labels, the types and the
rows, with labels that repeat and keys that are missing among them.
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

NAN = float("nan")
FRAME = {
    "k": ["a", "b", "a", "b", "a", None, "b", "a", "a", "b"],
    "x": [NAN, 1.0, 3.0, NAN, NAN, 5.0, NAN, NAN, 2.0, 4.0],
    "y": [1.5, NAN, NAN, 2.5, 3.5, NAN, 6.0, NAN, NAN, 1.0],
    "n": [4, 4, 1, 2, 2, 7, 9, 4, 3, 8],
}


def repeated(m: Any) -> Any:
    """The frame on labels that repeat."""
    labels = ["p", "q", "p", "r", "q", "p", "s", "r", "p", "q"]
    return m.DataFrame({"i": labels, **FRAME}).set_index("i")


def same(got: list[Any], want: list[Any]) -> bool:
    """Equal row by row to a relative 1e-12, with NaN equal to NaN."""

    def missing(value: Any) -> bool:
        return value is None or value != value

    return len(got) == len(want) and all(
        (missing(a) and missing(b)) or (a == b or abs(a - b) <= 1e-12 * max(abs(a), abs(b)))
        for a, b in zip(got, want, strict=True)
    )


def agrees(got: Any, want: Any) -> None:
    """The same labels, types and rows as pandas, frame or series."""
    assert list(got.index) == list(want.index)
    if hasattr(want, "columns"):
        assert list(got.columns) == list(want.columns)
        for name in want.columns:
            printed = str(want[name].dtype)
            assert got[name].dtype == ("string" if printed == "str" else printed), name
            assert same(got[name].tolist(), want[name].tolist()), name
    else:
        assert got.name == want.name
        assert got.dtype == str(want.dtype)
        assert same(got.tolist(), want.tolist())


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.DataFrame(FRAME).groupby("k").ffill(),
    lambda m: m.DataFrame(FRAME).groupby("k").bfill(),
    lambda m: m.DataFrame(FRAME).groupby("k").ffill(limit=1),
    lambda m: m.DataFrame(FRAME).groupby("k").bfill(limit=1),
    lambda m: m.DataFrame(FRAME).groupby("k").ffill(limit=0),
    lambda m: m.DataFrame(FRAME).groupby("k").ffill(limit=-1),
    lambda m: m.DataFrame(FRAME).groupby("k").bfill(limit=2.0),
    lambda m: m.DataFrame(FRAME).groupby("k")["x"].ffill(),
    lambda m: m.DataFrame(FRAME).groupby("k")["y"].bfill(limit=1),
    lambda m: m.DataFrame(FRAME).groupby("k", dropna=False).ffill(),
    lambda m: m.DataFrame(FRAME).groupby("k", dropna=False)["x"].bfill(),
    lambda m: m.DataFrame(FRAME).groupby("k", sort=False).ffill(),
    lambda m: m.DataFrame(FRAME).groupby(["k", "n"]).ffill(),
    lambda m: m.DataFrame(FRAME).groupby("n")[["x", "y"]].bfill(),
    lambda m: repeated(m).groupby("k").ffill(),
    lambda m: repeated(m).groupby("k")["x"].bfill(limit=1),
    lambda m: m.DataFrame(FRAME).head(0).groupby("k").ffill(),
    lambda m: m.DataFrame(FRAME).groupby("k").pct_change(),
    lambda m: m.DataFrame(FRAME).groupby("k").pct_change(periods=2),
    lambda m: m.DataFrame(FRAME).groupby("k").pct_change(periods=-1),
    lambda m: m.DataFrame(FRAME).groupby("k")["n"].pct_change(),
    lambda m: m.DataFrame(FRAME).groupby("k", dropna=False)["x"].pct_change(),
    lambda m: repeated(m).groupby("k").pct_change(),
    lambda m: m.DataFrame(FRAME).groupby("k").filter(lambda g: len(g) > 4),
    lambda m: m.DataFrame(FRAME).groupby("k").filter(lambda g: g["n"].sum() > 20),
    lambda m: m.DataFrame(FRAME).groupby("k").filter(lambda g: False),
    lambda m: m.DataFrame(FRAME).groupby("k").filter(lambda g: NAN),
    lambda m: m.DataFrame(FRAME).groupby("k", sort=False).filter(lambda g: g["x"].count() > 2),
    lambda m: m.DataFrame(FRAME).groupby("k", dropna=False).filter(lambda g: len(g) < 5),
    lambda m: m.DataFrame(FRAME).groupby("n").filter(lambda g, at: len(g) > at, at=1),
    lambda m: m.DataFrame(FRAME).groupby("k")["n"].filter(lambda s: s.max() > 8),
    lambda m: m.DataFrame(FRAME).groupby("k")["x"].filter(lambda s: 1),
    lambda m: m.DataFrame(FRAME).groupby("k")["x"].filter(lambda s: s.mean()),
    lambda m: m.DataFrame(FRAME).groupby("k")["x"].filter(lambda s: "yes"),
    lambda m: m.DataFrame(FRAME).groupby("k")["x"].filter(lambda s: ""),
    lambda m: repeated(m).groupby("k").filter(lambda g: len(g) > 4),
    lambda m: repeated(m).groupby("k")["n"].filter(lambda s: s.iloc[0] == 4),
]


@pytest.mark.parametrize("build", BUILDS)
def test_a_group_answer_is_pandas_answer(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """Limits, directions, missing keys, repeated labels, empties and filters."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.DataFrame(FRAME).groupby("k").ffill(limit="1"),
    lambda m: m.DataFrame(FRAME).groupby("k").pct_change(fill_method="ffill"),
    lambda m: m.DataFrame(FRAME).groupby("k").pct_change(periods=1.5),
    lambda m: m.DataFrame(FRAME).groupby("k").filter(lambda g: 1),
    lambda m: m.DataFrame(FRAME).groupby("k").filter(lambda g: "yes"),
    lambda m: m.DataFrame(FRAME).groupby("k").filter(5),
    lambda m: m.DataFrame(FRAME).groupby("k")["x"].filter(lambda s: s > 0),
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
    lambda m: m.DataFrame(FRAME).groupby("k").filter(lambda g: True, dropna=False),
    lambda m: m.DataFrame(FRAME).groupby("k").pct_change(freq="D"),
]


@pytest.mark.parametrize("build", REFUSED)
def test_what_is_not_written_is_refused(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Keeping the lost rows as blanks, and a shift by a frequency."""
    with pytest.raises(NotImplementedError):
        build(firepanda)


OWNERS: list[Callable[[Any], Any]] = [
    lambda m: type(m.DataFrame(FRAME).groupby("k")),
    lambda m: type(m.DataFrame(FRAME).groupby("k")["x"]),
]


@pytest.mark.parametrize("name", ["ffill", "bfill", "pct_change", "filter"])
@pytest.mark.parametrize("owner", OWNERS)
def test_the_signature_is_pandas_signature(
    firepanda: ModuleType, owner: Callable[[Any], Any], name: str
) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(owner(firepanda), name)).parameters
    yours = inspect.signature(getattr(owner(pd), name)).parameters
    assert [(p.name, p.kind) for p in ours.values()] == [(p.name, p.kind) for p in yours.values()]
    for each in ours:
        assert ours[each].default == yours[each].default, each
