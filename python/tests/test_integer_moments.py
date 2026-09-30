"""`skew` and `kurt` of whole numbers far from zero, against pandas.

Near two to the sixty two neighbouring floats are 1024 apart, so a column cast
to float before its moments are taken has moved each value by up to 512.
pandas casts first, and so does firepanda now, so both answer the same float
even where it is a few parts in ten million away from exact arithmetic.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

BASE = -(2**62)
SPREADS = [0, 4_000_000_000, 17, 3_999_999_999, 123_456_789, 2_000_000_000, 7, 3_000_000_001]


@needs_pandas
@pytest.mark.parametrize("method", ["skew", "kurt"])
@pytest.mark.parametrize("gap", [False, True])
def test_a_moment_far_from_zero_is_pandas_answer(
    firepanda: ModuleType, method: str, gap: bool
) -> None:
    """The same float pandas gives, with or without a gap."""
    import pandas as pd

    values = [BASE + spread for spread in SPREADS]
    values = [*values, None] if gap else values
    mine = getattr(firepanda.Series(values), method)()
    theirs = getattr(pd.Series(values, dtype="float64" if gap else "int64"), method)()
    assert mine == float(theirs)


@needs_pandas
@pytest.mark.parametrize("method", ["skew", "kurt"])
def test_a_column_spanning_every_whole_number_still_answers(
    firepanda: ModuleType, method: str
) -> None:
    """The widest column there is, cast as pandas casts it."""
    import pandas as pd

    values = [-(2**63), 2**63 - 1, 0, 5]
    mine = getattr(firepanda.Series(values), method)()
    assert mine == float(getattr(pd.Series(values), method)())
