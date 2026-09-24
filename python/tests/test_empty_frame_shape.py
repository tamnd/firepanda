"""A frame built from labels or names with no values, checked against pandas.

pandas keeps `index=` and `columns=` when there is no data, so the frame has
those rows and those columns and every cell is missing. pandas makes the
columns objects, which firepanda does not have, so they are floats here, and
`dtype=` makes them what it names.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.DataFrame(index=[0, 1]),
    lambda m: m.DataFrame(index=m.Index(["x", "y"], name="k")),
    lambda m: m.DataFrame(columns=["a", "b"], dtype="float64"),
    lambda m: m.DataFrame(columns=["a"], index=[3, 1], dtype="float64"),
    lambda m: m.DataFrame(columns=["a", "b"], index=m.Index(["x"], name="k"), dtype="str"),
    lambda m: m.DataFrame(index=[], columns=[]),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_shape_is_pandas_shape(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Rows alone, columns alone, both, a named index and a type."""
    import pandas as pd

    got, want = build(firepanda), build(pd)
    assert got.shape == want.shape
    assert list(got.columns) == list(want.columns)
    assert list(got.index) == list(want.index)
    assert got.index.name == want.index.name
    for name in want.columns:
        assert str(got[name].dtype).replace("string", "str") == str(want[name].dtype)
        assert all(value is None or value != value for value in got[name].tolist())


def test_columns_alone_are_floats(firepanda: ModuleType) -> None:
    """pandas' object columns are floats, the type that holds a gap and takes numbers."""
    frame = firepanda.DataFrame(columns=["a", "b"])
    assert list(frame.columns) == ["a", "b"]
    assert frame.shape == (0, 2)
    assert [str(kind) for kind in frame.dtypes] == ["float64", "float64"]
