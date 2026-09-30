"""`corr` and `cov` of every pair of columns over a window of a frame.

pandas answers `pairwise=True`, the default with no other frame, with each row
label once for every column of the other frame, under a second level of row
labels, and the columns of the first frame across. Each test here runs the
same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def frame(lib: Any) -> Any:
    return lib.DataFrame({"a": [1.0, 2.0, 4.0, 3.0], "b": [4, 3, 2, 5]}, index=list("wxyz"))


BUILDS = {
    "rolling corr": lambda lib: frame(lib).rolling(3).corr(),
    "rolling cov": lambda lib: frame(lib).rolling(2).cov(),
    "other frame": lambda lib: frame(lib).rolling(2).cov(frame(lib), pairwise=True),
    "other columns": lambda lib: (
        frame(lib).rolling(3).corr(frame(lib).rename(columns={"a": "p", "b": "q"}), pairwise=True)
    ),
    "expanding corr": lambda lib: frame(lib).expanding().corr(),
    "ewm corr": lambda lib: frame(lib).ewm(span=3).corr(),
    "ewm cov": lambda lib: frame(lib).ewm(span=3).cov(),
    "labels": lambda lib: frame(lib).rolling(3).corr().index,
    "names": lambda lib: list(frame(lib).rename_axis("r").rolling(3).corr().index.names),
    "by name": lambda lib: frame(lib).rolling(2).cov(frame(lib), pairwise=False),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_every_pair_over_a_window_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
