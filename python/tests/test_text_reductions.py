"""Reductions and scans over text, measured against pandas.

pandas sums text by joining it in row order, over a column, a frame and each
group, and runs a join, least or greatest down a column. Every other numeric
reduction is refused with a `TypeError` whose words depend on where it ran.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pandas as pd
import pytest


def _frame(module: Any, dtype: str = "str") -> Any:
    return module.DataFrame(
        {
            "k": ["a", "b", "a", None],
            "s": module.Series(["x", None, "z", "w"], dtype=dtype),
            "v": [1, 2, 3, 4],
        }
    )


CASES = {
    "sum": lambda f: f["s"].sum(),
    "sum-skipna-false": lambda f: f["s"].sum(skipna=False),
    "sum-min-count": lambda f: f["s"].sum(min_count=4),
    "sum-empty": lambda f: f["s"].head(0).sum(),
    "cumsum": lambda f: f["s"].cumsum(),
    "cumsum-skipna-false": lambda f: f["s"].cumsum(skipna=False),
    "cummin": lambda f: f["s"].cummin(),
    "cummax": lambda f: f["s"].cummax(),
    "frame-sum": lambda f: f.sum(),
    "frame-sum-min-count": lambda f: f.sum(min_count=4),
    "group-sum": lambda f: f.groupby("k")["s"].sum(),
    "group-frame-sum": lambda f: f.groupby("k").sum(),
    "group-sum-as-index-false": lambda f: f.groupby("k", as_index=False).sum(),
    "group-sum-selection": lambda f: f.groupby("k")[["s", "v"]].sum(),
    "group-sum-null-key": lambda f: f.groupby("k", dropna=False).sum(),
    "group-sum-min-count": lambda f: f.groupby("k").sum(min_count=2),
    "group-sum-unsorted": lambda f: f.groupby("k", sort=False)["s"].sum(),
    "group-agg-sum": lambda f: f.groupby("k")["s"].agg("sum"),
}


@pytest.mark.parametrize("dtype", ["str", "string"])
@pytest.mark.parametrize("case", list(CASES.values()), ids=list(CASES))
def test_text_reduces_as_pandas(firepanda: ModuleType, case: Any, dtype: str) -> None:
    """Each answer prints as pandas prints it."""
    if dtype == "string" and case is CASES["group-sum-min-count"]:
        pytest.skip("a group of nothing but gaps loses the string type, see note 125")
    if dtype == "string" and case is CASES["frame-sum-min-count"]:
        pytest.skip("an object column cannot hold NA yet")
    assert repr(case(_frame(firepanda, dtype))) == repr(case(_frame(pd, dtype)))


REFUSED = {
    "mean": lambda f: f["s"].mean(),
    "median": lambda f: f["s"].median(),
    "std": lambda f: f["s"].std(),
    "prod": lambda f: f["s"].prod(),
    "cumprod": lambda f: f["s"].cumprod(),
    "frame-mean": lambda f: f.mean(),
    "group-mean": lambda f: f.groupby("k")["s"].mean(),
    "group-frame-prod": lambda f: f.groupby("k").prod(),
    "group-cumsum": lambda f: f.groupby("k").cumsum(),
    "group-cummin": lambda f: f.groupby("k")["s"].cummin(),
}


def _refusal(module: Any, case: Any, dtype: str) -> str:
    with pytest.raises(TypeError) as caught:
        case(_frame(module, dtype))
    return str(caught.value)


@pytest.mark.parametrize("dtype", ["str", "string"])
@pytest.mark.parametrize("case", list(REFUSED.values()), ids=list(REFUSED))
def test_text_refuses_as_pandas(firepanda: ModuleType, case: Any, dtype: str) -> None:
    """A reduction text has no meaning for raises pandas' `TypeError` in pandas' words."""
    assert _refusal(firepanda, case, dtype) == _refusal(pd, case, dtype)
