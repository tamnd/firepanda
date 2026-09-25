"""`Interval`, checked against pandas.

The repr, the fields, containment, overlap, ordering and arithmetic are compared
for ends that are numbers, instants and spans, and the mistakes by type and
words.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.Interval(0, 1),
    lambda m: str(m.Interval(0, 1)),
    lambda m: str(m.Interval(0.5, 1.5, "both")),
    lambda m: m.Interval(m.Timestamp("2020-01-01"), m.Timestamp("2021-01-01")),
    lambda m: str(m.Interval(m.Timestamp("2020-01-01"), m.Timestamp("2021-01-01"), "left")),
    lambda m: m.Interval(0, 2).mid,
    lambda m: m.Interval(0, 3).mid,
    lambda m: m.Interval(0, 2).length,
    lambda m: m.Interval(m.Timestamp("2020-01-01"), m.Timestamp("2021-01-01")).length,
    lambda m: m.Interval(m.Timestamp("2020-01-01"), m.Timestamp("2021-01-01")).mid,
    lambda m: m.Interval(m.Timedelta("1D"), m.Timedelta("3D")).mid,
    lambda m: m.Interval(m.Timedelta("1D"), m.Timedelta("3D")),
    lambda m: m.Interval(1, 1).is_empty,
    lambda m: m.Interval(1, 1, "both").is_empty,
    lambda m: (1 in m.Interval(0, 1), 0 in m.Interval(0, 1), 0 in m.Interval(0, 1, "left")),
    lambda m: m.Interval(0, 1).overlaps(m.Interval(1, 2)),
    lambda m: m.Interval(0, 1, "both").overlaps(m.Interval(1, 2, "both")),
    lambda m: m.Interval(0, 1, "neither").overlaps(m.Interval(0.5, 0.7)),
    lambda m: m.Interval(0, 1) + 1,
    lambda m: 2 * m.Interval(0, 1),
    lambda m: m.Interval(0, 1) / 2,
    lambda m: m.Interval(0, 1) // 2,
    lambda m: m.Interval(0, 1) - 0.5,
    lambda m: m.Interval(m.Timestamp("2020-01-01"), m.Timestamp("2021-01-01")) + m.Timedelta("1D"),
    lambda m: m.Interval(0, 1) == m.Interval(0, 1),
    lambda m: m.Interval(0, 1) == m.Interval(0, 1, "left"),
    lambda m: m.Interval(0, 1) < m.Interval(0, 2),
    lambda m: m.Interval(0, 1, "both") < m.Interval(0, 1, "left"),
    lambda m: m.Interval(0, 1) == 1,
    lambda m: hash(m.Interval(0, 1)) == hash(m.Interval(0, 1)),
    lambda m: m.Interval(0, 1) in m.Interval(0, 2),
    lambda m: (
        m.Interval(0, 2).closed_left,
        m.Interval(0, 2).closed_right,
        m.Interval(0, 2).open_left,
        m.Interval(0, 2).open_right,
        m.Interval(0, 2, "both").closed,
    ),
    lambda m: sorted([m.Interval(2, 3), m.Interval(0, 1), m.Interval(0, 5)]),
    lambda m: {m.Interval(0, 1): "a"}[m.Interval(0, 1)],
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every field and operation, compared by repr."""
    import pandas as pd

    assert repr(build(firepanda)) == repr(build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.Interval(0, 1, "foo"),
    lambda m: m.Interval(2, 1),
    lambda m: m.Interval(0, "a"),
    lambda m: m.Interval("a", "b"),
    lambda m: m.Interval(0, 1).overlaps(1),
    lambda m: "a" in m.Interval(0, 1),
    lambda m: m.Interval(0, 1, closed=None),
    lambda m: m.Interval(0.0, float("nan")),
    lambda m: m.Interval(True, 2),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_mistakes_raise_as_pandas_raises(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The same error type and words as pandas."""
    import pandas as pd

    with pytest.raises(Exception) as expected:
        build(pd)
    with pytest.raises(type(expected.value), match="^" + re.escape(str(expected.value)) + "$"):
        build(firepanda)


PYTHON_WORDED: list[Callable[[Any], Any]] = [
    lambda m: m.Interval(0, 1) + m.Interval(0, 1),
    lambda m: m.Interval(0, 1) * "a",
    lambda m: m.Interval(0, 1) < 1,
    lambda m: m.Interval(m.Timestamp("2020-01-01", tz="UTC"), m.Timestamp("2021-01-01")),
]


@pytest.mark.parametrize("build", PYTHON_WORDED)
def test_python_worded_mistakes_raise_the_same_type(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """Python writes these, naming the class by its module, so only the type is compared."""
    import pandas as pd

    with pytest.raises(Exception) as expected:
        build(pd)
    with pytest.raises(type(expected.value)):
        build(firepanda)
