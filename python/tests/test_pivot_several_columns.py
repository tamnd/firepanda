"""`pivot` and `pivot_table` over several columns keys, and frames of several column levels.

pandas answers several columns keys with a MultiIndex on the columns, by
setting the keys as row labels, or aggregating over them, and unstacking the
columns keys. The same frames need `fillna`, `dropna`, `astype` and the
reductions to read a column name that is a tuple. Each test runs the same code
on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def base(lib: Any) -> Any:
    return lib.DataFrame(
        {
            "a": ["x", "x", "y", "y", "y"],
            "b": ["p", "q", "p", "q", "q"],
            "c": ["u", "u", "v", "v", "v"],
            "d": ["m", "n", "m", "n", "n"],
            "v": [1, 2, 3, 4, 5],
            "w": [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )


def unique(lib: Any) -> Any:
    return base(lib).iloc[:4]


def levels(lib: Any) -> Any:
    return unique(lib).groupby(["a", "b", "c"])[["v"]].sum().unstack(["b", "c"])


BUILDS = {
    "pivot": lambda lib: unique(lib).pivot(index="a", columns=["b", "c"], values="v"),
    "pivot values": lambda lib: unique(lib).pivot(index="a", columns=["b", "c"], values=["v", "w"]),
    "pivot every value": lambda lib: unique(lib)[["a", "b", "c", "v", "w"]].pivot(
        index="a", columns=["b", "c"]
    ),
    "pivot row labels": lambda lib: unique(lib).pivot(columns=["b", "c"], values="v"),
    "pivot two row keys": lambda lib: unique(lib).pivot(
        index=["a", "d"], columns=["b", "c"], values="w"
    ),
    "table sum": lambda lib: base(lib).pivot_table(
        index="a", columns=["b", "c"], values="v", aggfunc="sum"
    ),
    "table mean": lambda lib: base(lib).pivot_table(index="a", columns=["b", "c"], values="v"),
    "table filled": lambda lib: base(lib).pivot_table(
        index="a", columns=["b", "c"], values="v", aggfunc="sum", fill_value=0
    ),
    "table mean filled": lambda lib: base(lib).pivot_table(
        index="a", columns=["b", "c"], values="w", fill_value=0
    ),
    "table two row keys": lambda lib: base(lib).pivot_table(
        index=["a", "d"], columns=["b", "c"], values="v", aggfunc="sum"
    ),
    "table values": lambda lib: base(lib).pivot_table(
        index="a", columns=["b", "c"], values=["v", "w"], aggfunc="max"
    ),
    "table every value": lambda lib: base(lib).pivot_table(
        index="a", columns=["b", "c"], aggfunc="count"
    ),
    "table function": lambda lib: base(lib).pivot_table(
        index="a", columns=["b", "c"], values="v", aggfunc=lambda s: s.max() - s.min()
    ),
    "table unsorted": lambda lib: base(lib).pivot_table(
        index="a", columns=["c", "b"], values="v", aggfunc="sum", sort=False
    ),
    "table no row keys": lambda lib: base(lib).pivot_table(
        columns=["b", "c"], values="v", aggfunc="sum"
    ),
    "fillna": lambda lib: levels(lib).fillna(-1),
    "dropna columns": lambda lib: levels(lib).dropna(axis=1),
    "dropna all columns": lambda lib: levels(lib).dropna(how="all", axis=1),
    "astype": lambda lib: [
        str(kind) for kind in levels(lib).astype({("v", "p", "u"): "float32"}).dtypes
    ],
    "sum of flags": lambda lib: levels(lib).notna().sum().tolist(),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_several_columns_keys_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_a_repeated_pair_is_refused(firepanda: Any) -> None:
    with pytest.raises(ValueError, match="Index contains duplicate entries, cannot reshape"):
        base(firepanda).pivot(index="a", columns=["b", "c"], values="v")
