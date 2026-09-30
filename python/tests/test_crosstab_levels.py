"""`crosstab` with several keys on an axis, labelled by a MultiIndex.

pandas labels the rows or the columns of a cross table with the pairs of keys
it saw, sorted, one level a key, named by the columns or by `rownames` and
`colnames`. Each test here runs the same code on both libraries and compares
what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def keys(lib: Any) -> tuple[Any, Any, Any, Any]:
    return (
        lib.Series(["x", "x", "y", "x", "y"], name="a"),
        lib.Series(["p", "q", "q", "p", "p"], name="b"),
        lib.Series(["u", "v", "u", "u", "v"], name="c"),
        lib.Series([1, 2, 3, 4, 5], name="v"),
    )


def gapped(lib: Any) -> tuple[Any, Any, Any]:
    return (
        lib.Series(["x", None, "y", "x"], name="a"),
        lib.Series(["p", "q", None, "p"], name="b"),
        lib.Series(["u", "v", "u", "v"], name="c"),
    )


BUILDS = {
    "rows": lambda lib: lib.crosstab(list(keys(lib)[:2]), keys(lib)[2]),
    "columns": lambda lib: lib.crosstab(keys(lib)[0], list(keys(lib)[1:3])),
    "both": lambda lib: lib.crosstab(list(keys(lib)[:2]), [keys(lib)[2], keys(lib)[0]]),
    "named": lambda lib: lib.crosstab(
        list(keys(lib)[:2]), keys(lib)[2], rownames=["r1", "r2"], colnames=["cc"]
    ),
    "by row": lambda lib: lib.crosstab(list(keys(lib)[:2]), keys(lib)[2], normalize="index"),
    "by column": lambda lib: lib.crosstab(list(keys(lib)[:2]), keys(lib)[2], normalize="columns"),
    "share": lambda lib: lib.crosstab(keys(lib)[0], list(keys(lib)[1:3]), normalize=True),
    "values": lambda lib: lib.crosstab(
        list(keys(lib)[:2]), keys(lib)[2], values=keys(lib)[3], aggfunc="sum"
    ),
    "lists": lambda lib: lib.crosstab(
        [keys(lib)[0].tolist(), keys(lib)[1].tolist()], keys(lib)[2].tolist()
    ),
    "gaps": lambda lib: lib.crosstab(list(gapped(lib)[:2]), gapped(lib)[2]),
    "level names": lambda lib: (
        lib.crosstab(list(keys(lib)[:2]), keys(lib)[2]).index.names,
        lib.crosstab(keys(lib)[0], list(keys(lib)[1:3])).columns.names,
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_crosstab_levels_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_a_name_for_each_key_or_pandas_complains(firepanda: Any) -> None:
    for lib in (pd, firepanda):
        with pytest.raises(AssertionError, match="arrays and names must have the same length"):
            lib.crosstab(list(keys(lib)[:2]), keys(lib)[2], rownames=["r1"])
