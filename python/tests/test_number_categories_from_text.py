"""Categories that are not text, given to a column of text categories, as in pandas.

`rename_categories`, `set_categories`, `map` and `from_codes` with numbers make
the column again from its codes, and a decided `CategoricalDtype` of numbers
matches the values by equality, so 1.0 lands in the category 1.
"""

from types import ModuleType
from typing import Any

import pandas as pd
import pytest


def _shown(out: Any) -> str:
    return repr(out)


CASES = {
    "rename-list": lambda lib: lib.Categorical(["a", "b", "a"]).rename_categories([0, 1]),
    "rename-dict": lambda lib: lib.Categorical(["a", "b", "a"]).rename_categories({"a": 5, "b": 7}),
    "rename-series": lambda lib: lib.Series(
        ["x", "y", "x"], dtype="category", index=[4, 5, 6], name="k"
    ).cat.rename_categories([10, 20]),
    "set-values": lambda lib: lib.Series(["x", "y"], dtype="category").cat.set_categories([1, 2]),
    "set-rename-short": lambda lib: lib.Series(
        ["x", "y", "x"], dtype="category"
    ).cat.set_categories([1], rename=True),
    "map-dict": lambda lib: lib.Series(["x", "y", "x"], dtype="category").map({"x": 1.5, "y": 2.5}),
    "map-meeting": lambda lib: lib.Series(["x", "y", "x"], dtype="category").map({"x": 1, "y": 1}),
    "map-gap": lambda lib: lib.Series(["x", None, "y"], dtype="category").map({"x": 3, "y": 4}),
    "map-missing-key": lambda lib: lib.Series(["x", "y", "x"], dtype="category").map({"x": 3}),
    "categorical-map": lambda lib: lib.Categorical(["a", "b", "a"]).map({"a": 3, "b": 2}),
    "from-codes": lambda lib: lib.Categorical.from_codes([0, 1, 0, -1], categories=[10, 20]),
    "from-codes-floats": lambda lib: lib.Categorical.from_codes([1, 1], categories=[0.5, 1.5]),
    "decided-floats": lambda lib: lib.Series([0, 1, 0, None]).astype(
        lib.CategoricalDtype([0, 1], ordered=True)
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_number_categories_match_pandas(firepanda: ModuleType, name: str) -> None:
    assert _shown(CASES[name](firepanda)) == _shown(CASES[name](pd))


def test_renamed_categories_are_numbers(firepanda: ModuleType) -> None:
    column = firepanda.Series(["x", "y", "x"], dtype="category")
    out = column.cat.rename_categories([10, 20])
    assert out.cat.categories.tolist() == [10, 20]
    assert str(out.cat.categories.dtype) == "int64"
    assert out.cat.codes.tolist() == [0, 1, 0]
