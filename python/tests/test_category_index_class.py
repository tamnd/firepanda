"""`CategoricalIndex` as a class of its own, with pandas' lookups and category methods.

pandas answers an index of categories with a `CategoricalIndex` that finds a
label, drops it, takes each label once and changes its categories the way a
`Categorical` does, all without losing the categories. Each test here runs
the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def plain(lib: Any) -> Any:
    return lib.CategoricalIndex(["a", "b", "a", "c"], name="k")


def ordered(lib: Any) -> Any:
    return lib.CategoricalIndex(["b", "a", "c"], categories=["c", "b", "a"], ordered=True)


BUILDS = {
    "get_loc": lambda lib: plain(lib).get_loc("b"),
    "get_loc ordered": lambda lib: ordered(lib).get_loc("c"),
    "get_loc repeated": lambda lib: [bool(x) for x in plain(lib).get_loc("a")],
    "get_indexer": lambda lib: [int(x) for x in ordered(lib).get_indexer(["a", "z"])],
    "in": lambda lib: ("b" in plain(lib), "z" in plain(lib)),
    "unique": lambda lib: plain(lib).unique(),
    "values": lambda lib: plain(lib).values,
    "drop": lambda lib: plain(lib).drop("b"),
    "drop ignored": lambda lib: plain(lib).drop(["z"], errors="ignore"),
    "duplicated": lambda lib: [bool(x) for x in plain(lib).duplicated()],
    "is_unique": lambda lib: (plain(lib).is_unique, ordered(lib).is_unique),
    "monotonic": lambda lib: (
        ordered(lib).is_monotonic_increasing,
        ordered(lib).sort_values().is_monotonic_increasing,
        ordered(lib).sort_values(ascending=False).is_monotonic_decreasing,
    ),
    "union": lambda lib: plain(lib).union(lib.CategoricalIndex(["d"])),
    "rename_categories": lambda lib: plain(lib).rename_categories(["x", "y", "z"]),
    "add_categories": lambda lib: plain(lib).add_categories(["d"]),
    "remove_categories": lambda lib: plain(lib).remove_categories(["c"]),
    "remove_unused": lambda lib: lib.CategoricalIndex(
        ["a"], categories=["a", "b"]
    ).remove_unused_categories(),
    "reorder": lambda lib: plain(lib).reorder_categories(["c", "b", "a"]),
    "as_ordered": lambda lib: plain(lib).as_ordered(),
    "as_unordered": lambda lib: ordered(lib).as_unordered(),
    "set_categories": lambda lib: plain(lib).set_categories(["a", "b"]),
    "map": lambda lib: plain(lib).map({"a": "A", "b": "B", "c": "C"}),
    "map function": lambda lib: plain(lib).map(str.upper),
    "map merging": lambda lib: plain(lib).map({"a": "x", "b": "x", "c": "y"}),
    "class": lambda lib: (type(plain(lib)).__name__, type(plain(lib)[1:]).__name__),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_category_index_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_an_index_of_categories_is_one(firepanda: Any) -> None:
    grouped = firepanda.DataFrame({"k": ["b", "a", "b"], "v": [1, 2, 3]}).astype({"k": "category"})
    labels = grouped.groupby("k", observed=True).sum().index
    assert isinstance(plain(firepanda), firepanda.CategoricalIndex)
    assert isinstance(labels, firepanda.CategoricalIndex)
    assert isinstance(firepanda.Index(firepanda.Categorical(["x"])), firepanda.CategoricalIndex)


def test_a_missing_label_is_refused(firepanda: Any) -> None:
    with pytest.raises(KeyError):
        plain(firepanda).get_loc("z")
    with pytest.raises(KeyError, match="not found in axis"):
        plain(firepanda).drop("z")
    with pytest.raises(ValueError, match="duplicate labels"):
        plain(firepanda).reindex(["b"])
