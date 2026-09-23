"""A median and a linear quantile agree with pandas to the last bit.

pandas hands a whole column to numpy, whose median adds the two middle values
and halves them and whose linear quantile is the lower value plus the fraction
of the gap below a half and the upper value less the rest of it from a half up.
A group goes to pandas' own Cython, which halves the sum for a median as well.
The two ways of writing the same number part in the last bit often enough that
a few hundred random columns find it every time, so these compare with `==`.

A grouped quantile is left out on purpose. pandas writes it as one multiply and
one add, which a C compiler fuses on arm64 and not on x86, so pandas' own answer
depends on the machine it runs on.
"""

from __future__ import annotations

import importlib.util
import random
from types import ModuleType

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

SCALES = [1.0, 100.0, 1e6, 1e300]


def columns(seed: int, count: int = 300) -> list[list[float]]:
    """Random columns from one to twelve rows, over a spread of magnitudes."""
    rng = random.Random(seed)
    return [
        [rng.random() * rng.choice(SCALES) - rng.random() * 10 for _ in range(rng.randint(1, 12))]
        for _ in range(count)
    ]


def test_a_median_is_exactly_pandas(firepanda: ModuleType) -> None:
    """Every column, even and odd."""
    import pandas as pd

    for rows in columns(1):
        assert float(firepanda.Series(rows).median()) == float(pd.Series(rows).median())


def test_a_linear_quantile_is_exactly_pandas(firepanda: ModuleType) -> None:
    """At a random fraction and at the ones that land exactly on a half."""
    import pandas as pd

    rng = random.Random(2)
    for rows in columns(2):
        q = rng.choice([rng.random(), 0.5, 1 / 3, 0.25, 0.75, 0.9])
        assert float(firepanda.Series(rows).quantile(q)) == float(pd.Series(rows).quantile(q))


def test_a_frame_quantile_and_median_are_exactly_pandas(firepanda: ModuleType) -> None:
    """Down each column, through the same door as a series."""
    import pandas as pd

    for rows in columns(3, 100):
        data = {"a": rows, "b": list(reversed(rows))}
        mine = firepanda.DataFrame(data)
        theirs = pd.DataFrame(data)
        assert mine.median().tolist() == theirs.median().tolist()
        assert mine.quantile(0.4).tolist() == theirs.quantile(0.4).tolist()


def test_a_grouped_median_is_exactly_pandas(firepanda: ModuleType) -> None:
    """Three groups over each column."""
    import pandas as pd

    for rows in columns(4, 100):
        data = {"k": [i % 3 for i in range(len(rows))], "v": rows}
        mine = firepanda.DataFrame(data).groupby("k")["v"].median().tolist()
        assert mine == pd.DataFrame(data).groupby("k")["v"].median().tolist()
