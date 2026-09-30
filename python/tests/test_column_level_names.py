"""What a frame's reductions label their answer with, and how `nunique` counts NaN.

pandas labels a reduction of a frame by the frame's columns, so the names of a
named column axis, one level or several, come along, and `idxmax` over columns
of several levels answers a MultiIndex rather than tuples. `nunique` counts a
NaN as missing, never as a value. Each test runs the same code on both
libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

NAN = float("nan")


def named(lib: Any) -> Any:
    return lib.DataFrame({"x": [1, 2], "y": [3.0, 4.0]}).rename_axis(columns="c")


def levels(lib: Any) -> Any:
    frame = lib.DataFrame({"a": ["x", "y"], "b": ["p", "q"], "v": [1, 2]})
    return frame.pivot(index="a", columns="b", values=["v"])


BUILDS = {
    "sum": lambda lib: levels(lib).sum(),
    "mean": lambda lib: levels(lib).mean(),
    "count": lambda lib: levels(lib).count(),
    "max": lambda lib: levels(lib).max(),
    "dtypes": lambda lib: levels(lib).dtypes.astype(str),
    "gaps counted": lambda lib: levels(lib).isna().sum(),
    "any": lambda lib: levels(lib).any(),
    "idxmax": lambda lib: levels(lib).idxmax(),
    "row read across": lambda lib: levels(lib).iloc[0],
    "quantile": lambda lib: levels(lib).quantile(0.5),
    "memory without index": lambda lib: levels(lib).memory_usage(index=False),
    "memory named": lambda lib: named(lib).memory_usage(index=False),
    "memory with index": lambda lib: named(lib).memory_usage(),
    "nunique levels": lambda lib: levels(lib).nunique(),
    "nunique series": lambda lib: lib.Series([1.0, NAN]).nunique(),
    "nunique series kept": lambda lib: lib.Series([1.0, NAN]).nunique(dropna=False),
    "nunique both spellings": lambda lib: lib.Series([1.0, None, NAN, 2.0]).nunique(dropna=False),
    "nunique all gaps": lambda lib: lib.Series([NAN, NAN]).nunique(),
    "nunique frame": lambda lib: lib.DataFrame(
        {"p": [1.0, NAN, NAN], "q": [1, 2, 2], "r": ["a", None, "a"]}
    ).nunique(),
    "nunique frame kept": lambda lib: lib.DataFrame(
        {"p": [1.0, NAN, NAN], "q": [1, 2, 2], "r": ["a", None, "a"]}
    ).nunique(dropna=False),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_column_level_names_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_memory_with_index_labels_levels_as_tuples(firepanda: Any) -> None:
    usage = levels(firepanda).memory_usage()
    assert usage.index.tolist() == ["Index", ("v", "p"), ("v", "q")]
    assert usage.tolist()[1:] == levels(pd).memory_usage().tolist()[1:]
