"""`reindex(level=)` from a MultiIndex onto flat labels, and the order of the rows.

On the outer level pandas takes the rows in the order the labels are asked for.
On an inner level it keeps the rows in the order they stand and only drops the
ones whose label is not asked for. Each test runs the same code on both
libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def labels(lib: Any) -> Any:
    pairs = [("b", 1), ("a", 2), ("b", 2), ("a", 1), ("c", 3)]
    return lib.MultiIndex.from_tuples(pairs, names=["x", "y"])


def frame(lib: Any) -> Any:
    return lib.DataFrame({"v": range(5)}, index=labels(lib))


def series(lib: Any) -> Any:
    return lib.Series(range(5), index=labels(lib))


def deep(lib: Any) -> Any:
    triples = [("b", 1, "q"), ("a", 2, "p"), ("a", 1, "q"), ("b", 1, "p")]
    return lib.Series(range(4), index=lib.MultiIndex.from_tuples(triples))


BUILDS = {
    "inner by name": lambda lib: frame(lib).reindex([2, 1], level="y"),
    "inner reversed": lambda lib: frame(lib).reindex([3, 2], level=1),
    "outer": lambda lib: frame(lib).reindex(["c", "b"], level=0),
    "series inner": lambda lib: series(lib).reindex([2, 1, 3], level=1),
    "third level": lambda lib: deep(lib).reindex(["q", "p"], level=2),
    "axis spelled": lambda lib: frame(lib).reindex(labels=[2, 1], axis="index", level="y"),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_reindex_on_a_level_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
