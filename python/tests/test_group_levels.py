"""Grouping by the levels of a MultiIndex, and a `Grouper` with no key.

pandas' `level=` on labels of several levels groups by the values of each
level named, by name or by number, and labels the groups after them. A
`Grouper` with no key groups by the row labels, or by the level it names, and
one with a frequency bins them as `resample` does, on its own or beside
other keys, where each row is keyed by the label of the bin it falls in. Each test here runs the
same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def levels(lib: Any) -> Any:
    labels = lib.MultiIndex.from_tuples([("x", 1), ("y", 1), ("x", 2), ("y", 2)], names=["p", "q"])
    return lib.DataFrame({"v": [1, 2, 3, 4], "w": [0.5, 1.5, 2.5, 3.5]}, index=labels)


def instants(lib: Any) -> Any:
    when = lib.date_range("2024-01-01", periods=5, freq="12h", name="t")
    return lib.DataFrame({"v": [1, 2, 3, 4, 5]}, index=when)


def named(lib: Any) -> Any:
    return lib.DataFrame({"v": [1, 2, 3, 4]}, index=lib.Index(["b", "a", "b", "a"], name="k"))


BUILDS = {
    "name": lambda lib: levels(lib).groupby(level="q").sum(),
    "number": lambda lib: levels(lib).groupby(level=1).sum(),
    "negative": lambda lib: levels(lib).groupby(level=-2).mean(),
    "list of one": lambda lib: levels(lib).groupby(level=["q"]).sum(),
    "list of two": lambda lib: levels(lib).groupby(level=["q", "p"]).sum(),
    "unsorted": lambda lib: levels(lib).groupby(level="p", sort=False).sum(),
    "series": lambda lib: levels(lib)["v"].groupby(level="p").sum(),
    "series two": lambda lib: levels(lib)["v"].groupby(level=[0, 1]).max(),
    "transform": lambda lib: levels(lib).groupby(level="p").transform("sum"),
    "cumsum": lambda lib: levels(lib)["v"].groupby(level="q").cumsum(),
    "size": lambda lib: levels(lib).groupby(level="p").size(),
    "as_index": lambda lib: levels(lib).groupby(level="p", as_index=False).sum(),
    "grouper level": lambda lib: levels(lib).groupby(lib.Grouper(level="q")).sum(),
    "grouper level 0": lambda lib: named(lib).groupby(lib.Grouper(level=0)).sum(),
    "grouper unsorted": lambda lib: named(lib).groupby(lib.Grouper(level="k", sort=False)).sum(),
    "grouper freq": lambda lib: instants(lib).groupby(lib.Grouper(freq="D")).sum(),
    "grouper freq mean": lambda lib: instants(lib).groupby(lib.Grouper(freq="2D")).mean(),
    "grouper freq closed": lambda lib: (
        instants(lib).groupby(lib.Grouper(freq="D", closed="right", label="right")).sum()
    ),
    "grouper level in a list": lambda lib: (
        levels(lib).assign(k=["a", "b", "a", "b"]).groupby([lib.Grouper(level="p"), "k"]).sum()
    ),
    "level after a column": lambda lib: (
        levels(lib).assign(k=["a", "b", "a", "b"]).groupby(["k", lib.Grouper(level=1)]).sum()
    ),
    "level name in a list": lambda lib: (
        levels(lib).assign(k=["a", "b", "a", "b"]).groupby(["p", "k"]).sum()
    ),
    "grouper freq on labels in a list": lambda lib: (
        instants(lib)
        .assign(k=["a", "b", "a", "b", "a"])
        .groupby([lib.Grouper(freq="D"), "k"])
        .sum()
    ),
    "grouper freq closed in a list": lambda lib: (
        instants(lib)
        .assign(k=["a", "b", "a", "b", "a"])
        .groupby([lib.Grouper(freq="D", closed="right", label="right"), "k"])
        .sum()
    ),
    "grouper freq closed right label left": lambda lib: (
        instants(lib)
        .assign(k=["a", "b", "a", "b", "a"])
        .groupby([lib.Grouper(freq="D", closed="right"), "k"])
        .sum()
    ),
    "grouper freq on a column in a list": lambda lib: (
        instants(lib)
        .assign(k=["a", "b", "a", "b", "a"])
        .reset_index()
        .groupby([lib.Grouper(key="t", freq="D"), "k"])
        .sum()
    ),
    "grouper month end in a list": lambda lib: (
        instants(lib)
        .assign(k=["a", "b", "a", "b", "a"])
        .reset_index()
        .groupby([lib.Grouper(key="t", freq="ME"), "k"])
        .v.sum()
    ),
    "grouper hours after a column": lambda lib: (
        instants(lib)
        .assign(k=["a", "b", "a", "b", "a"])
        .reset_index()
        .groupby(["k", lib.Grouper(key="t", freq="7h")])
        .v.count()
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_group_levels_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


MISTAKES = {
    "no such name": lambda lib: levels(lib).groupby(level="z").sum(),
    "no such number": lambda lib: levels(lib).groupby(level=5).sum(),
}


@pytest.mark.parametrize("make", MISTAKES.values(), ids=MISTAKES.keys())
def test_a_level_not_there_is_refused_as_pandas(firepanda: Any, make: Any) -> None:
    with pytest.raises(Exception) as theirs:
        make(pd)
    with pytest.raises(Exception) as mine:
        make(firepanda)
    assert type(mine.value).__name__ == type(theirs.value).__name__ or isinstance(
        mine.value, type(theirs.value)
    )
    assert str(mine.value) == str(theirs.value)
