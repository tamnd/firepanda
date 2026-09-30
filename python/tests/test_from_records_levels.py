"""`DataFrame.from_records` with several columns for the row labels.

pandas takes a list of column names as `index` and labels the rows by a level
for each, named after it, whether the records are tuples or dicts, with
`exclude` leaving other columns out. Each test here runs the same code on both
libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

ROWS = [("a", 1, 2.5), ("b", 2, 3.5), ("a", 3, 4.5)]
DICTS = [{"k": "a", "n": 1, "v": 2.5}, {"k": "b", "n": 2, "v": 3.5}]

BUILDS = {
    "tuples": lambda lib: lib.DataFrame.from_records(
        ROWS, columns=["k", "n", "v"], index=["k", "n"]
    ),
    "dicts": lambda lib: lib.DataFrame.from_records(DICTS, index=["k", "n"]),
    "excluded": lambda lib: lib.DataFrame.from_records(DICTS, index=["k", "n"], exclude=["v"]),
    "names": lambda lib: list(lib.DataFrame.from_records(DICTS, index=["k", "n"]).index.names),
    "one": lambda lib: lib.DataFrame.from_records(DICTS, index=["k"]),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_several_index_columns_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
