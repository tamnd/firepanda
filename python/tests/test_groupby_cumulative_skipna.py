"""`skipna=False` on a group by's running folds, `cumsum`, `cumprod`, `cummin` and `cummax`.

pandas takes `skipna` through the keywords these folds pass on, and a gap that
may not be skipped leaves every later row of its group missing. `cummin` and
`cummax` read nothing else from the keywords and pass over the rest. Each test
runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

KINDS = ["cumsum", "cumprod", "cummin", "cummax"]


def gapped(lib: Any) -> Any:
    return lib.DataFrame(
        {
            "k": ["a", "a", "b", "a", "b", "c"],
            "v": [1.0, None, 2.0, 5.0, 3.0, None],
            "w": [1, 2, 3, 4, 5, 6],
            "f": [True, False, True, True, False, True],
        }
    )


def shown(answer: Any) -> str:
    kinds = answer.dtypes if hasattr(answer, "columns") else [answer.dtype]
    return repr((answer.to_string(), [str(kind) for kind in kinds]))


BUILDS = {
    "frame": lambda lib, kind: getattr(gapped(lib).groupby("k"), kind)(skipna=False),
    "column": lambda lib, kind: getattr(gapped(lib).groupby("k")["v"], kind)(skipna=False),
    "skipped": lambda lib, kind: getattr(gapped(lib).groupby("k")["v"], kind)(skipna=True),
    "numbers only": lambda lib, kind: getattr(
        gapped(lib).assign(t=list("pqrstu")).groupby("k"), kind
    )(numeric_only=True, skipna=False),
    "as_index": lambda lib, kind: getattr(gapped(lib).groupby("k", as_index=False), kind)(
        skipna=False
    ),
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_running_fold_skipna_is_pandas(firepanda: Any, make: Any, kind: str) -> None:
    assert shown(make(firepanda, kind)) == shown(make(pd, kind))


@pytest.mark.parametrize("kind", ["cummin", "cummax"])
def test_other_keywords_are_passed_over(firepanda: Any, kind: str) -> None:
    def make(lib: Any) -> Any:
        return getattr(gapped(lib).groupby("k"), kind)(axis=0)

    assert shown(make(firepanda)) == shown(make(pd))


@pytest.mark.parametrize("kind", ["cumsum", "cumprod"])
def test_other_keywords_are_refused(firepanda: Any, kind: str) -> None:
    with pytest.raises(ValueError) as theirs:
        getattr(gapped(pd).groupby("k"), kind)(axis=0)
    with pytest.raises(ValueError) as mine:
        getattr(gapped(firepanda).groupby("k"), kind)(axis=0)
    assert str(mine.value) == str(theirs.value)
