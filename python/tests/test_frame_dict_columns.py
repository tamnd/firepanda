"""A frame built from a mapping whose columns are mappings of row label to value.

pandas reads each inner mapping by its keys: the rows are every key the
mappings hold, in the order first seen, and a key one mapping lacks is a gap.
With rows named, each mapping is read by those labels. A list beside a mapping
has nothing to line up by, so pandas refuses it. Each test runs the same code
on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

BUILDS = {
    "same keys": lambda lib: lib.DataFrame({"p": {"x": 1}, "q": {"x": 2}}),
    "uneven keys": lambda lib: lib.DataFrame({"p": {"x": 1, "y": 3}, "q": {"x": 2}}),
    "first seen order": lambda lib: lib.DataFrame({"p": {"y": 1, "x": 3}, "q": {"z": 2}}),
    "empty": lambda lib: lib.DataFrame({"p": {}, "q": {}}),
    "number keys": lambda lib: lib.DataFrame({"p": {1: "a", 0: "b"}}),
    "with a scalar": lambda lib: lib.DataFrame({"p": {"y": 1}, "q": 7}),
    "rows named": lambda lib: lib.DataFrame({"p": {"y": 1}}, index=["y", "w"]),
    "mixed types": lambda lib: lib.DataFrame({"p": {"x": 1.5}, "q": {"x": "s"}}).dtypes,
    "from_dict": lambda lib: lib.DataFrame.from_dict({"p": {"x": 1, "y": 3}, "q": {"x": 2}}),
    "from_dict index": lambda lib: lib.DataFrame.from_dict(
        {"p": {"x": 1, "y": 3}, "q": {"x": 2}}, orient="index"
    ),
    "from_dict dtype": lambda lib: lib.DataFrame.from_dict(
        {"p": {"x": 1}, "q": {"x": 2}}, orient="columns", dtype="float64"
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_dict_columns_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_a_mapping_beside_a_list_is_refused(firepanda: Any) -> None:
    with pytest.raises(ValueError, match="Mixing dicts with non-Series"):
        firepanda.DataFrame({"p": {"y": 1, "x": 3}, "q": [5, 6]})
