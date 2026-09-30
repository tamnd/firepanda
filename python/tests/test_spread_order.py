"""`var`, `std`, `sem` and `skew` of a series added up in pandas' order.

pandas measures each value from the mean numpy sums and leaves the error in
that mean alone, so on five values near two to the fifty two it answers a
variance of 37.25 where exact arithmetic gives 37.2, and a skewness twelve
percent off. firepanda takes the same sums in the same order and answers the
same floats, which is what these compare.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

SHIFTED = [2.0**52 + value for value in (1, 3, 7, 15, 20)]
COLUMNS = [
    ("shifted", SHIFTED, None),
    ("whole", [1, 2, 3, 10], None),
    ("gaps", [1.0, None, 3.0, 7.0, 2.5], "float64"),
    ("large whole", [2**62 + i * 7919 for i in range(50)], None),
    ("constant", [5, 5, 5], None),
    ("one", [1.0], None),
    ("float32", [1.5, 2.25, 3.0, None], "float32"),
]


@needs_pandas
@pytest.mark.parametrize("ddof", [0, 1, 2])
@pytest.mark.parametrize("method", ["var", "std", "sem"])
@pytest.mark.parametrize(("label", "values", "dtype"), COLUMNS, ids=[c[0] for c in COLUMNS])
def test_a_spread_is_pandas_float(
    firepanda: ModuleType, label: str, values: list, dtype: str | None, method: str, ddof: int
) -> None:
    """The same float, NaN where pandas has too few values for the degrees taken out."""
    import pandas as pd

    mine = getattr(firepanda.Series(values, dtype=dtype), method)(ddof=ddof)
    theirs = float(getattr(pd.Series(values, dtype=dtype), method)(ddof=ddof))
    assert mine == theirs or (mine != mine and theirs != theirs)


@needs_pandas
@pytest.mark.parametrize(("label", "values", "dtype"), COLUMNS, ids=[c[0] for c in COLUMNS])
def test_a_skewness_is_pandas_float(
    firepanda: ModuleType, label: str, values: list, dtype: str | None
) -> None:
    """The shifted column is the one where the two orders part by twelve percent."""
    import pandas as pd

    mine = firepanda.Series(values, dtype=dtype).skew()
    theirs = float(pd.Series(values, dtype=dtype).skew())
    assert mine == theirs or (mine != mine and theirs != theirs)


def test_skipna_false_with_a_gap_is_nan(firepanda: ModuleType) -> None:
    """A gap the caller asked to keep makes the answer missing, as in pandas."""
    column = firepanda.Series([1.0, None, 3.0])
    assert column.var(skipna=False) != column.var(skipna=False)
