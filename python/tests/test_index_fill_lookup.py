"""`Index.get_indexer` and `Index.reindex` with `method`, `limit` and `tolerance`, as in pandas.

A label the index does not hold reads from the label before it, after it or
nearest it, a `limit` caps how many labels one neighbour fills, and a
`tolerance` takes back a neighbour too far away. Each test runs the same code
on both libraries and compares the positions.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

TARGET = [5, 10, 14, 26, 35, 50]


def tens(lib: Any) -> Any:
    return lib.Index([10, 20, 30, 40])


def days(lib: Any) -> Any:
    return lib.date_range("2024-01-01", periods=3, freq="D")


def moments(lib: Any) -> Any:
    return lib.to_datetime(["2024-01-01 12:00", "2024-01-05 00:00"])


BUILDS = {
    "pad": lambda lib: tens(lib).get_indexer(TARGET, method="pad"),
    "ffill": lambda lib: tens(lib).get_indexer(TARGET, method="ffill"),
    "bfill": lambda lib: tens(lib).get_indexer(TARGET, method="bfill"),
    "nearest": lambda lib: tens(lib).get_indexer(TARGET, method="nearest"),
    "nearest tie": lambda lib: tens(lib).get_indexer([15, 25], method="nearest"),
    "tolerance": lambda lib: tens(lib).get_indexer(TARGET, method="nearest", tolerance=3),
    "limit": lambda lib: tens(lib).get_indexer([11, 12, 13, 21], method="pad", limit=1),
    "falling": lambda lib: lib.Index([40, 30, 20, 10]).get_indexer(TARGET, method="pad"),
    "text": lambda lib: lib.Index(["a", "c"]).get_indexer(["b", "d"], method="pad"),
    "moments": lambda lib: days(lib).get_indexer(moments(lib), method="pad"),
    "moments tolerance": lambda lib: days(lib).get_indexer(
        moments(lib), method="nearest", tolerance="6h"
    ),
    "reindex": lambda lib: tens(lib).reindex(TARGET, method="pad")[1],
    "reindex tolerance": lambda lib: tens(lib).reindex(TARGET, method="nearest", tolerance=3)[1],
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_filled_lookup_is_pandas(firepanda: Any, make: Any) -> None:
    assert [int(x) for x in make(firepanda)] == [int(x) for x in make(pd)]


def test_the_same_labels_need_no_positions(firepanda: Any) -> None:
    made, positions = tens(firepanda).reindex([10, 20, 30, 40], method="pad")
    assert positions is None
    assert repr(made) == repr(tens(pd).reindex([10, 20, 30, 40], method="pad")[0])


@pytest.mark.parametrize(
    ("build", "error"),
    [
        (lambda lib: lib.Index([1, 1, 2]).get_indexer([1], method="pad"), "InvalidIndexError"),
        (lambda lib: lib.Index([1, 3, 2]).get_indexer([2], method="pad"), "ValueError"),
        (lambda lib: tens(lib).get_indexer(TARGET, method="foo"), "ValueError"),
        (lambda lib: tens(lib).get_indexer(TARGET, tolerance=1), "ValueError"),
        (lambda lib: lib.Index(["a", "c"]).get_indexer(["b"], method="nearest"), "TypeError"),
    ],
)
def test_what_pandas_refuses_is_refused(firepanda: Any, build: Any, error: str) -> None:
    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert type(theirs.value).__name__ == error
    assert error in {c.__name__ for c in type(mine.value).__mro__}
    assert str(mine.value) == str(theirs.value)
