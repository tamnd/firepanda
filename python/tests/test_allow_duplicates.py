"""`allow_duplicates=True` on `reset_index` and `insert` when no label repeats.

pandas' flag only matters when the new column's label is already there, and
otherwise the answer is the one without it. A firepanda frame holds each
column label once, so only a real clash is refused. A nameless index lands
under `level_0` when `index` is already a column, which is pandas' fallback.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def named(lib: Any) -> Any:
    return lib.DataFrame({"v": [1, 2]}, index=lib.Index([5, 6], name="k"))


def inserted(lib: Any) -> Any:
    frame = named(lib)
    frame.insert(0, "z", [7, 8], allow_duplicates=True)
    return frame


BUILDS = {
    "named": lambda lib: named(lib).reset_index(allow_duplicates=True),
    "nameless": lambda lib: lib.DataFrame({"v": [1, 2]}).reset_index(allow_duplicates=True),
    "index taken": lambda lib: lib.DataFrame({"index": [1, 2]}).reset_index(),
    "index taken allowed": lambda lib: lib.DataFrame({"index": [1, 2]}).reset_index(
        allow_duplicates=True
    ),
    "dropped": lambda lib: named(lib).reset_index(drop=True, allow_duplicates=True),
    "renamed": lambda lib: named(lib).reset_index(names="w", allow_duplicates=True),
    "insert": inserted,
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_allow_duplicates_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_a_real_clash_is_refused(firepanda: Any) -> None:
    clashing = firepanda.DataFrame({"v": [1, 2]}, index=firepanda.Index([5, 6], name="v"))
    with pytest.raises(NotImplementedError, match="allow_duplicates=True onto the column 'v'"):
        clashing.reset_index(allow_duplicates=True)
    with pytest.raises(NotImplementedError, match="insert with allow_duplicates=True"):
        named(firepanda).insert(0, "v", [1, 2], allow_duplicates=True)
