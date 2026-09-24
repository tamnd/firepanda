"""`value_counts` and `mode` on a series, against pandas.

Both are a group by on the column itself in the order each value first
appears, which is the order pandas' hash table hands its keys back in. A tie
in the counts keeps that order, because the sort on the count is stable.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def plain(values: list[Any]) -> list[Any]:
    """NaN as None, so two lists with NaN in the same places compare equal."""
    return [None if isinstance(v, float) and math.isnan(v) else v for v in values]


COLUMNS: list[Callable[[Any], Any]] = [
    lambda m: m.Series(["b", "a", "b", None, "c", "a", "d"], name="x"),
    lambda m: m.Series([3, 1, 3, 2]),
    lambda m: m.Series([1.5, None, 1.5, None, None, 2.0], name="f"),
    lambda m: m.Series([True, False, True]),
    lambda m: m.Series([], dtype="float64"),
    lambda m: m.Series([None, None], dtype="float64"),
]

SETTINGS: list[dict[str, Any]] = [
    {},
    {"dropna": False},
    {"normalize": True},
    {"ascending": True},
    {"sort": False},
    {"sort": False, "dropna": False},
]


@pytest.mark.parametrize("settings", SETTINGS)
@pytest.mark.parametrize("build", COLUMNS)
def test_value_counts(firepanda: ModuleType, build: Any, settings: dict[str, Any]) -> None:
    """The counts, their order, both names and the type."""
    import pandas as pd

    got, want = build(firepanda).value_counts(**settings), build(pd).value_counts(**settings)
    assert plain(got.tolist()) == plain(want.tolist())
    assert plain(got.index.tolist()) == plain(want.index.tolist())
    assert got.name == want.name
    assert got.index.name == want.index.name
    assert got.dtype == str(want.dtype)


@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("build", COLUMNS)
def test_mode(firepanda: ModuleType, build: Any, dropna: bool) -> None:
    """Every value tied at the top, sorted, under a fresh index."""
    import pandas as pd

    got, want = build(firepanda).mode(dropna=dropna), build(pd).mode(dropna=dropna)
    assert plain(got.tolist()) == plain(want.tolist())
    assert got.index.tolist() == want.index.tolist()
    assert got.name == want.name


def test_bins_are_refused(firepanda: ModuleType) -> None:
    """They need `cut`, which firepanda has not written."""
    with pytest.raises(NotImplementedError):
        firepanda.Series([1.0, 2.0]).value_counts(bins=2)
