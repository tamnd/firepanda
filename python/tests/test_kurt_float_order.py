"""`kurt` of a float column far from zero, against pandas to the last bits.

At values near 4.6e18 with a spread of 4e9 the fourth moment moves by 2e-7
with the order of adding, and pandas' answer is 1.2e-7 off the exact one. So
a float column is summed as pandas sums it, whole, with its gaps as zeros, in
numpy's order.
"""

from __future__ import annotations

import importlib.util
import random
from types import ModuleType

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def far(seed: int, gaps: bool) -> list[float | None]:
    """64 floats near -4.6e18 spread over 4e9, every other one a gap if asked."""
    draw = random.Random(seed)
    values: list[float | None] = [
        float(-(2**62) + draw.randrange(4_000_000_000)) for _ in range(64)
    ]
    if gaps:
        values[::2] = [None] * 32
    return values


COLUMNS = {
    "far with gaps": far(1, True),
    "far": far(2, False),
    "long with gaps": far(3, True) * 9,
    "small": [0.5, 1.5, None, 2.25, -3.0, 8.0],
    "three": [1.0, None, 2.0, 3.0],
    "constant": [1e18, 1e18, 1e18, 1e18, None],
}


@needs_pandas
@pytest.mark.parametrize("name", list(COLUMNS))
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_kurt_of_floats_is_pandas_to_the_bit(firepanda: ModuleType, name: str, dtype: str) -> None:
    """The same float, NaN and zero included."""
    import pandas as pd

    mine = firepanda.Series(COLUMNS[name], dtype=dtype).kurt()
    them = pd.Series(COLUMNS[name], dtype=dtype).kurt()
    assert repr(float(mine)) == repr(float(them))
