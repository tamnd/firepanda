"""A MultiIndex as a frame, counted, sorted by a key, and looked up by several levels.

pandas labels the frame `to_frame` makes with the index itself, counts each
row with `value_counts`, calls a sort key on the values of each level, and
looks a key up across several levels at once, dropping the levels it looked
up unless none would be left. `reindex(level=)` takes each row holding a value
of one level in turn, and `get_indexer` fills forward or back along sorted
rows. Each test here runs the same code on both libraries and compares what
they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def pairs(lib: Any) -> Any:
    return lib.MultiIndex.from_arrays([["a", "a", "b", "b"], [1, 2, 1, 2]], names=["k", "n"])


def triples(lib: Any) -> Any:
    return lib.MultiIndex.from_arrays(
        [["a", "a", "b", "b"], [1, 2, 1, 2], ["x", "y", "x", "x"]], names=["k", "n", "c"]
    )


def repeated(lib: Any) -> Any:
    return lib.MultiIndex.from_arrays([["a", "b", "a"], [1, 2, 1]])


BUILDS = {
    "to_frame": lambda lib: pairs(lib).to_frame(),
    "to_frame names": lambda lib: pairs(lib).to_frame(name=["x", "y"]),
    "to_frame index": lambda lib: pairs(lib).to_frame(name=["x", "y"]).index,
    "value_counts": lambda lib: repeated(lib).value_counts(),
    "value_counts named": lambda lib: pairs(lib).value_counts(),
    "value_counts normalize": lambda lib: pairs(lib).value_counts(normalize=True),
    "value_counts ascending": lambda lib: repeated(lib).value_counts(ascending=True),
    "sort key": lambda lib: pairs(lib).sort_values(key=lambda level: level),
    "sort key descending": lambda lib: pairs(lib).sort_values(
        key=lambda level: level.map(lambda v: -v if isinstance(v, int) else v), ascending=False
    ),
    "levels all": lambda lib: pairs(lib).get_loc_level((2, "a"), level=[1, 0]),
    "levels two of three": lambda lib: triples(lib).get_loc_level(("a", "x"), level=[0, 2]),
    "levels kept": lambda lib: triples(lib).get_loc_level(
        ("b", "x"), level=[0, 2], drop_level=False
    ),
    "levels by name": lambda lib: triples(lib).get_loc_level(("b", 1), level=["k", "n"]),
    "levels one": lambda lib: triples(lib).get_loc_level(["b"], level=[0]),
    "reindex level": lambda lib: pairs(lib).reindex(["b", "z", "a"], level=0),
    "reindex level name": lambda lib: pairs(lib).reindex([2], level="n"),
    "reindex level empty": lambda lib: pairs(lib).reindex([], level=0),
    "ffill": lambda lib: pairs(lib).get_indexer([("a", 3), ("b", 0), ("0", 1)], method="ffill"),
    "bfill": lambda lib: pairs(lib).get_indexer([("a", 3), ("b", 0), ("c", 1)], method="bfill"),
    "pad exact": lambda lib: pairs(lib).get_indexer([("a", 2)], method="pad"),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_multi_lookups_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_refusals_are_pandas_refusals(firepanda: Any) -> None:
    with pytest.raises(KeyError, match="z"):
        triples(firepanda).get_loc_level(("z", 1), level=[0, 1])
    unsorted = firepanda.MultiIndex.from_arrays([["b", "a", "c"], [1, 2, 3]])
    with pytest.raises(ValueError, match="index must be monotonic increasing or decreasing"):
        unsorted.get_indexer([("a", 1)], method="ffill")
    with pytest.raises(NotImplementedError, match="method='nearest' not implemented yet"):
        pairs(firepanda).get_indexer([("a", 1)], method="nearest")
