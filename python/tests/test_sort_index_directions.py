"""`sort_index` with a list of directions, one a level.

pandas sorts the labels of a MultiIndex on each level in the direction given
for it, and with `level=` sorts on those levels alone, leaving the rest in the
order they came in. On labels of one level it reads the list as one flag: the
labels stay as they are when every flag agrees with the order they are in, and
are otherwise sorted upward unless the list is empty. Each test here runs the
same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def keyed(lib: Any) -> Any:
    labels = lib.MultiIndex.from_tuples([("b", 1), ("a", 2), ("b", 3), ("a", 1)], names=["x", "y"])
    return lib.DataFrame({"v": [1, 2, 3, 4]}, index=labels)


def flat(lib: Any, labels: list[Any]) -> Any:
    return lib.DataFrame([list(range(len(labels)))], columns=labels)


BUILDS = {
    "up then down": lambda lib: keyed(lib).sort_index(ascending=[True, False]),
    "down then up": lambda lib: keyed(lib).sort_index(ascending=[False, True]),
    "tuple": lambda lib: keyed(lib).sort_index(ascending=(False, False)),
    "series": lambda lib: keyed(lib)["v"].sort_index(ascending=[True, False]),
    "levels": lambda lib: keyed(lib).sort_index(level=["y", "x"], ascending=[False, True]),
    "one level": lambda lib: keyed(lib).sort_index(level="y", ascending=[False]),
    "labels dropped": lambda lib: keyed(lib).sort_index(ascending=[False, True], ignore_index=True),
    "in place": lambda lib: keyed(lib).sort_index(ascending=[False, True], inplace=True),
    "rows of one level": lambda lib: lib.DataFrame({"v": [1, 2, 3]}, index=[2, 1, 3]).sort_index(
        ascending=[False]
    ),
    "empty list": lambda lib: lib.Series([1, 2, 3], index=[2, 1, 3]).sort_index(ascending=[]),
    "columns falling": lambda lib: flat(lib, ["c", "b", "a"]).sort_index(axis=1, ascending=[False]),
    "columns mixed": lambda lib: flat(lib, ["b", "a", "c"]).sort_index(axis=1, ascending=[False]),
    "columns two flags": lambda lib: flat(lib, ["c", "b", "a"]).sort_index(
        axis=1, ascending=[False, True]
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_directions_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_too_few_directions_are_refused(firepanda: Any) -> None:
    with pytest.raises(ValueError):
        keyed(firepanda).sort_index(ascending=[True])
