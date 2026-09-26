"""`at_time`, `between_time` and `asof` on a column and a frame, checked against pandas.

The first two choose rows by the time of day of labels that are instants.
`asof` finds the last row at or before a label that holds values, skipping
missing values on a column and rows with a missing value in `subset` on a
frame, for one label or several.
"""

from __future__ import annotations

import datetime
import importlib.util
import re
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

STAMPS = [
    "2024-01-01 09:30:00",
    "2024-01-02 12:00:00",
    "2024-01-06 18:45:10",
    "2024-01-07 12:00:00",
]
LOOKUPS = ["2024-01-03 00:00:00", "2024-01-06 19:00:00", "2023-01-01 00:00:00"]


def facts(answer: Any) -> Any:
    """What is compared: a frame by column, a column by values, labels and name."""
    if hasattr(answer, "columns"):
        return (
            "frame",
            list(answer.columns),
            [[plain(v) for v in answer[column].tolist()] for column in answer.columns],
            [plain(v) for v in answer.index.tolist()],
        )
    if hasattr(answer, "index") and hasattr(answer, "tolist"):
        return (
            "column",
            [plain(v) for v in answer.tolist()],
            [plain(v) for v in answer.index.tolist()],
            plain(answer.name),
            str(answer.dtype).replace("float64", "float").replace("int64", "int"),
        )
    return plain(answer)


def plain(value: Any) -> Any:
    """A value as compared: a gap as None, an instant as text, a numpy number as Python."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if type(value).__name__ == "Timestamp":
        return str(value)
    return value.item() if hasattr(value, "item") and not hasattr(value, "__len__") else value


def column(m: ModuleType) -> Any:
    """A column of floats with a gap, on labels that are instants."""
    return m.Series([1.0, None, 3.0, 4.0], index=m.DatetimeIndex(STAMPS, name="w"), name="v")


def frame(m: ModuleType) -> Any:
    """A frame of whole numbers and floats with gaps, on labels that are instants."""
    return m.DataFrame(
        {"a": [1, 2, 3, 4], "b": [1.5, None, None, 4.5]}, index=m.DatetimeIndex(STAMPS, name="w")
    )


def lookups(m: ModuleType) -> Any:
    """Three instants to look up, one before every label."""
    return m.DatetimeIndex(LOOKUPS)


def numbered(m: ModuleType) -> Any:
    """A column of floats with a gap at the end, on whole number labels."""
    return m.Series([1.0, 2.0, None], index=[10, 20, 30])


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: column(m).at_time("12:00"),
    lambda m: column(m).at_time(datetime.time(9, 30)),
    lambda m: column(m).at_time("12:00", axis=0),
    lambda m: frame(m).at_time("12:00"),
    lambda m: column(m).between_time("09:00", "13:00"),
    lambda m: column(m).between_time("13:00", "10:00"),
    lambda m: frame(m).between_time("09:30", "12:00", inclusive="neither"),
    lambda m: frame(m).between_time("09:30", "12:00", inclusive="left"),
    lambda m: frame(m).between_time("09:30", "12:00", inclusive="right"),
    lambda m: frame(m).between_time("09:00", "13:00", axis="index"),
    lambda m: column(m).asof("2024-01-03"),
    lambda m: column(m).asof("2024-01-02 13:00"),
    lambda m: column(m).asof("2024-01-02 12:00"),
    lambda m: column(m).asof("2023-01-01"),
    lambda m: column(m).asof(m.Timestamp("2024-01-07 12:00")),
    lambda m: column(m).asof(lookups(m)),
    lambda m: frame(m).asof("2024-01-03"),
    lambda m: frame(m).asof("2023-01-01"),
    lambda m: frame(m).asof(m.Timestamp("2024-01-07")),
    lambda m: frame(m).asof(lookups(m)),
    lambda m: frame(m).asof(lookups(m), subset=["a"]),
    lambda m: frame(m).asof(lookups(m), subset="a"),
    lambda m: numbered(m).asof(25),
    lambda m: numbered(m).asof(35),
    lambda m: numbered(m).asof([5, 15, 35]),
    lambda m: numbered(m).asof((10, 20)),
    lambda m: m.Series([1, 2, 3], index=[10, 20, 30]).asof(25),
    lambda m: m.Series([1, 2, 3], index=[10, 20, 30]).asof([15, 30]),
    lambda m: m.Series([None, None], index=[1, 2], dtype=float).asof(5),
    lambda m: m.Series([None, None], index=[1, 2], dtype=float).asof([5]),
    lambda m: m.DataFrame({"a": [1, 2]}, index=[1, 3]).asof(2),
    lambda m: m.DataFrame({"a": [1, 2]}, index=[1, 3]).asof([0, 2, 4]),
    lambda m: m.DataFrame({"a": [None, None]}, index=[1, 3], dtype=float).asof([2]),
    lambda m: m.DataFrame({"a": [None, None]}, index=[1, 3], dtype=float).asof(2),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Rows, labels, names and whether whole numbers were widened, against pandas."""
    import pandas as pd

    assert facts(build(firepanda)) == facts(build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: frame(m).at_time("12:00", asof=True),
    lambda m: frame(m).between_time("09:30", "12:00", inclusive="x"),
    lambda m: m.Series([1, 2]).at_time("12:00"),
    lambda m: m.Series([1, 2]).between_time("1:00", "2:00"),
    lambda m: frame(m).at_time("12:00", axis=1),
    lambda m: column(m).at_time("12:00", axis=1),
    lambda m: column(m).between_time("noon", "13:00"),
    lambda m: column(m).asof(lookups(m), subset=["a"]),
    lambda m: column(m).asof(["2024-01-03"]),
    lambda m: frame(m).asof(["2024-01-03"]),
    lambda m: frame(m).asof(lookups(m), subset=["zz"]),
    lambda m: m.Series([1.0, 2.0], index=[20, 10]).asof(15),
    lambda m: m.Series([1.0, 2.0, 3.0], index=[3, 1, 2]).asof(2),
    lambda m: m.Series([0.0])[:0].asof(1),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_mistakes_raise_as_pandas_raises(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The same error type as pandas, compared by name, and the same words."""
    import pandas as pd

    with pytest.raises(Exception) as expected:
        build(pd)
    with pytest.raises(Exception, match="^" + re.escape(str(expected.value)) + "$") as found:
        build(firepanda)
    named = type(found.value).__name__.replace("InvalidArgumentError", "ValueError")
    assert named == type(expected.value).__name__


def test_text_that_is_no_instant_is_a_value_error(firepanda: ModuleType) -> None:
    """pandas reads text as an instant whatever the labels are, and raises a ValueError."""
    with pytest.raises(ValueError):
        firepanda.Series([1.0, 2.0], index=[1, 2]).asof("x")
