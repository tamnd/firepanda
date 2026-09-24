"""Arithmetic between two labelled operands holds each gap as pandas does.

A row only one side has is a gap in the answer. The kernel answers it with a
null and pandas answers NaN, widening an integer column to float64 to have
room for it, so the pandas facing layer turns the one into the other.

`**` is left out. numpy answers `1 ** nan` and `nan ** 0` with 1, so a gap
under a power is not always a NaN in pandas, and the kernel answers the gap
before either value is looked at.
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


def same(got: list[Any], want: list[Any]) -> bool:
    """Equal row by row, with NaN equal to NaN."""
    return len(got) == len(want) and all(
        a == b or (a != a and b != b) for a, b in zip(got, want, strict=True)
    )


SERIES: list[Callable[[Any], Any]] = [
    lambda m: m.Series([1, 2, 3]).iloc[:2] + m.Series([10, 20, 30]).iloc[1:],
    lambda m: m.Series([1, 2, 3]).iloc[:2] * m.Series([10, 20, 30]).iloc[1:],
    lambda m: m.Series([1, 2, 3]).iloc[:2] // m.Series([10, 20, 30]).iloc[1:],
    lambda m: m.Series([1, 2, 3]).iloc[:2] % m.Series([10, 20, 30]).iloc[1:],
    lambda m: m.Series([1.5, 2.0, 3.0]).iloc[:2].sub(m.Series([1.0, 2.0, 4.0]).iloc[1:]),
    lambda m: m.Series([1.5, 2.0, 3.0]).iloc[:2] / m.Series([1.0, 2.0, 4.0]).iloc[1:],
    lambda m: m.Series([1.5, 2.0, 3.0]).iloc[:2].add(m.Series([1.0, 4.0]).iloc[1:], fill_value=0),
    lambda m: m.Series([1, 2]) + m.Series([3, 4]),
    lambda m: m.Series([1, 2]).iloc[:1].eq(m.Series([1, 2])),
]


@pytest.mark.parametrize("build", SERIES)
def test_a_series_gap_is_a_nan(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The type and every row, against pandas."""
    import pandas as pd

    got, want = build(firepanda), build(pd)
    assert got.dtype == str(want.dtype)
    assert same(got.tolist(), want.tolist())


def test_a_frame_gap_is_a_nan(firepanda: ModuleType) -> None:
    """Both axes, and an integer column widens."""
    import pandas as pd

    data = {"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}
    got = firepanda.DataFrame(data)
    want = pd.DataFrame(data)
    got, want = got + got.iloc[1:], want + want.iloc[1:]
    for name in ("a", "b"):
        assert got[name].dtype == str(want[name].dtype)
        assert same(got[name].tolist(), want[name].tolist())


def test_a_frame_minus_a_series_with_a_column_missing(firepanda: ModuleType) -> None:
    """A column the series has no label for is all gaps."""
    import pandas as pd

    data = {"a": [1, 2], "b": [3, 4]}
    row = {"k": ["a"], "v": [1]}
    got = firepanda.DataFrame(data) - firepanda.DataFrame(row).set_index("k")["v"]
    want = pd.DataFrame(data) - pd.DataFrame(row).set_index("k")["v"]
    for name in ("a", "b"):
        assert got[name].dtype == str(want[name].dtype)
        assert same(got[name].tolist(), want[name].tolist())
