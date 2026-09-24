"""`to_timedelta` against pandas, for single values, lists and columns.

pandas reads text, whole numbers in a unit, floats and elapsed times, and
picks the unit of the answer from what it read: microseconds for text, the
unit itself or the coarsest that holds it for whole numbers, and nanoseconds
for floats and for anything mixed. A list answers a column in firepanda where
pandas answers a `TimedeltaIndex`, so lists are compared by value and dtype.
"""

from __future__ import annotations

import datetime
import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def facts(obj: Any) -> Any:
    """The values, dtype, labels and name of an answer, whichever shape it has."""
    if not hasattr(obj, "dtype"):
        return None if obj is None or str(obj) == "NaT" else (obj.value, obj.unit)
    values = [None if str(value) == "NaT" or value is None else value.value for value in obj]
    index = getattr(obj, "index", None)
    labels = None if index is None else index.tolist()
    if labels == list(range(len(values))):
        labels = None
    return values, str(obj.dtype), labels, getattr(obj, "name", None)


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.to_timedelta(["1h", "2 days 00:00:01.5", None, "1d2h3m4s", "-1 days +23:00:00"]),
    lambda m: m.to_timedelta(["00:01:02", "1.5h", "3us", "1 day", "10ns"]),
    lambda m: m.to_timedelta(["x", "1h"], errors="coerce"),
    lambda m: m.to_timedelta(["1h", 5]),
    lambda m: m.to_timedelta([1, 2]),
    lambda m: m.to_timedelta([1.5]),
    lambda m: m.to_timedelta([1, None], unit="s"),
    lambda m: m.to_timedelta((1, 2), unit="ms"),
    lambda m: m.to_timedelta([1, 2], unit="us"),
    lambda m: m.to_timedelta([1, 2], unit="ns"),
    lambda m: m.to_timedelta([1, 2], unit="h"),
    lambda m: m.to_timedelta([1, 2], unit="D"),
    lambda m: m.to_timedelta([1, 2], unit="min"),
    lambda m: m.to_timedelta([1, 2], unit="W"),
    lambda m: m.to_timedelta([1.5, None], unit="s"),
    lambda m: m.to_timedelta([1.5, None], unit="D"),
    lambda m: m.to_timedelta([datetime.timedelta(hours=1), None]),
    lambda m: m.to_timedelta(m.Series(["1h", None], index=[5, 6], name="x")),
    lambda m: m.to_timedelta(m.Series([1, 2], name="n"), unit="s"),
    lambda m: m.to_timedelta(m.Series([1, 2])),
    lambda m: m.to_timedelta(m.Series([1.5, None]), unit="ms"),
    lambda m: m.to_timedelta("1h"),
    lambda m: m.to_timedelta("1 days 02:00:00.5"),
    lambda m: m.to_timedelta(5),
    lambda m: m.to_timedelta(1.5, unit="s"),
    lambda m: m.to_timedelta(2, unit="D"),
    lambda m: m.to_timedelta(None),
    lambda m: m.to_timedelta(float("nan")),
    lambda m: m.to_timedelta("x", errors="coerce"),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Text, whole numbers in every unit, floats, gaps, columns and single values."""
    import pandas as pd

    mine, theirs = facts(build(firepanda)), facts(build(pd))
    assert mine == theirs


def test_a_column_of_elapsed_times_is_itself(firepanda: ModuleType) -> None:
    """A column that already holds elapsed times comes back as it is."""
    spans = firepanda.to_timedelta(["1h", "2h"])
    assert firepanda.to_timedelta(spans).tolist() == spans.tolist()


@pytest.mark.parametrize(
    "build",
    [
        lambda m: m.to_timedelta(["x"]),
        lambda m: m.to_timedelta(["1"], unit="s"),
        lambda m: m.to_timedelta([1], errors="ignore"),
        lambda m: m.to_timedelta(m.DataFrame({"a": [1]})),
        lambda m: m.to_timedelta([True]),
        lambda m: m.to_timedelta([1], unit="bogus"),
    ],
)
def test_mistakes_fail_as_pandas_fails(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Unreadable text, a unit with text, a bad `errors`, a frame, a flag, a bad unit."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value) == str(theirs.value)
