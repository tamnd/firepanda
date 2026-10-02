"""Masked columns through frame reductions, rank, round, clip, isin, counts and describe.

Each answer is compared with pandas by its printed form, which carries the
masked type and how its floats are written.
"""

from types import ModuleType
from typing import Any

import pandas as pd
import pytest


def _ints(lib: Any) -> Any:
    return lib.Series([1, None, 3, 3, None, 1, 2], dtype="Int64")


def _floats(lib: Any) -> Any:
    return lib.Series([1.5, None, -2.5, 0.125], dtype="Float64")


def _frame(lib: Any) -> Any:
    return lib.DataFrame(
        {
            "a": lib.Series([1, None, 3], dtype="Int64"),
            "b": lib.Series([1.5, None, -2.5], dtype="Float64"),
        }
    )


CASES = {
    "frame-sum": lambda lib: _frame(lib).sum(),
    "frame-prod": lambda lib: _frame(lib).prod(),
    "frame-mean": lambda lib: _frame(lib).mean(),
    "frame-min": lambda lib: _frame(lib).min(),
    "frame-std": lambda lib: _frame(lib).std(),
    "frame-var-ddof": lambda lib: _frame(lib).var(ddof=0),
    "frame-whole-sum": lambda lib: lib.DataFrame(
        {"a": lib.Series([1, None, 3], dtype="Int64"), "n": [1, 2, 3]}
    ).sum(),
    "frame-flags-sum": lambda lib: lib.DataFrame(
        {
            "a": lib.Series([1, None, 3], dtype="Int64"),
            "c": lib.Series([True, None, True], dtype="boolean"),
        }
    ).sum(),
    "frame-describe": lambda lib: _frame(lib).describe(),
    "negative-floats": lambda lib: lib.DataFrame({"a": [-1.5, 2.0]}, dtype="Float64"),
    "negative-whole": lambda lib: lib.Series([1, -3, None], dtype="Int64"),
    "negative-whole-frame": lambda lib: lib.DataFrame(
        {"a": lib.Series([1, -30], dtype="Int64"), "b": [-1, 2]}
    ),
    "round": lambda lib: _floats(lib).round(),
    "round-two": lambda lib: _floats(lib).round(2),
    "clip": lambda lib: _ints(lib).clip(2, 3),
    "rank": lambda lib: _ints(lib).rank(),
    "rank-min": lambda lib: _ints(lib).rank(method="min", ascending=False),
    "rank-dense-pct": lambda lib: _ints(lib).rank(method="dense", pct=True),
    "isin": lambda lib: _ints(lib).isin([1, 2]),
    "isin-flags": lambda lib: lib.Series([True, None], dtype="boolean").isin([True]),
    "isin-text": lambda lib: lib.Series(["a", None], dtype="string").isin(["a"]),
    "counts-gap": lambda lib: _ints(lib).value_counts(dropna=False),
    "counts-gap-ascending": lambda lib: _ints(lib).value_counts(dropna=False, ascending=True),
    "counts-gap-unsorted": lambda lib: _ints(lib).value_counts(dropna=False, sort=False),
}


@pytest.mark.parametrize("name", list(CASES))
def test_masked_answers_match_pandas(firepanda: ModuleType, name: str) -> None:
    assert repr(CASES[name](firepanda)) == repr(CASES[name](pd))


def test_a_settled_truth_is_a_numpy_flag(firepanda: ModuleType) -> None:
    flags = firepanda.Series([True, None, False], dtype="boolean")
    assert repr(flags.any(skipna=False)) == repr(pd.Series([True, None], dtype="boolean").any())
