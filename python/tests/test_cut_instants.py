"""`cut` and `qcut` of instants and spans, and of values that are not a column.

pandas bins instants and spans by their counts of their unit and answers
intervals of instants or spans, with the zone kept, and it answers a
`Categorical` for a list, an array or an index. Each test runs the same code
on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def stamps(lib: Any) -> Any:
    return lib.Series(lib.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", None]), name="d")


def spans(lib: Any) -> Any:
    return lib.Series(lib.to_timedelta(["1h", "3h", "10h", None]))


BUILDS = {
    "count": lambda lib: lib.cut(stamps(lib), 3),
    "edges": lambda lib: lib.cut(
        stamps(lib), lib.to_datetime(["2023-12-31", "2024-01-04", "2024-01-11"])
    ),
    "labels": lambda lib: lib.cut(stamps(lib), 2, labels=["a", "b"]),
    "positions": lambda lib: lib.cut(stamps(lib), 2, labels=False),
    "edges back": lambda lib: lib.cut(stamps(lib), 2, retbins=True)[1],
    "left": lambda lib: lib.cut(stamps(lib), 2, right=False),
    "lowest": lambda lib: lib.cut(
        stamps(lib),
        lib.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10"]),
        include_lowest=True,
    ),
    "qcut": lambda lib: lib.qcut(stamps(lib), 2),
    "qcut edges back": lambda lib: lib.qcut(stamps(lib), 2, retbins=True)[1],
    "qcut positions": lambda lib: lib.qcut(stamps(lib), 3, labels=False),
    "seconds": lambda lib: lib.qcut(stamps(lib).dt.as_unit("s"), 2),
    "nanoseconds": lambda lib: lib.cut(stamps(lib).dt.as_unit("ns"), 2),
    "zone": lambda lib: lib.cut(stamps(lib).dt.tz_localize("UTC"), 2),
    "spans": lambda lib: lib.cut(spans(lib), 3),
    "spans qcut": lambda lib: lib.qcut(spans(lib), 2),
    "spans edges": lambda lib: lib.cut(spans(lib), lib.to_timedelta(["0h", "2h", "12h"])),
    "spans intervals": lambda lib: lib.cut(
        spans(lib), lib.IntervalIndex.from_breaks(lib.to_timedelta(["0h", "2h", "12h"]))
    ),
    "spans qcut edges back": lambda lib: lib.qcut(spans(lib), [0, 0.3, 1], retbins=True)[1],
    "mid": lambda lib: lib.cut(stamps(lib), 2).cat.categories.mid,
    "left ends": lambda lib: lib.cut(stamps(lib), 2).cat.categories.left,
    "length": lambda lib: lib.cut(spans(lib), 2).cat.categories.length,
    "one": lambda lib: lib.cut(stamps(lib), 2)[0],
    "counted": lambda lib: lib.cut(stamps(lib), 2).value_counts(),
    "index": lambda lib: lib.cut(lib.DatetimeIndex(stamps(lib)), 2),
    "array": lambda lib: list(lib.cut(stamps(lib).to_numpy(), 2, labels=False)),
    "list": lambda lib: lib.cut([1, 5, 9, 3], 2),
    "list labels": lambda lib: lib.cut([1, 5, 9, 3], 2, labels=["lo", "hi"]),
    "list positions": lambda lib: list(lib.cut([1, 5, 9, 3], 2, labels=False)),
    "list intervals": lambda lib: lib.cut([1, 5, 9], lib.IntervalIndex.from_breaks([0, 4, 10])),
    "index qcut": lambda lib: lib.qcut(lib.Index([1.0, 2.0, 3.0, 4.0]), 2),
    "breaks": lambda lib: lib.IntervalIndex.from_breaks(
        lib.to_datetime(["2023-12-31", "2024-01-04", "2024-01-11"])
    ),
    "long label": lambda lib: lib.Series(["a" * 60, "b"], dtype="category"),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_cut_of_instants_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_edges_of_another_kind_are_refused(firepanda: Any) -> None:
    with pytest.raises(ValueError, match="bins must be of datetime64 dtype"):
        firepanda.cut(stamps(firepanda), [1, 2, 3])
    with pytest.raises(ValueError, match="bins must be of timedelta64 dtype"):
        firepanda.cut(spans(firepanda), [1, 2, 3])
