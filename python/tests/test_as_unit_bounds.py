"""`as_unit("ns")` on instants a nanosecond count cannot reach, checked against pandas.

pandas raises `OutOfBoundsDatetime` naming the first instant outside the
range, on the UTC wall clock and to the second. Instants just inside either
end restate without an error.
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


def index_of(values: list[Any], **options: Any) -> Callable[[Any], Any]:
    return lambda m: m.to_datetime(values, **options).as_unit("ns")


def column_of(values: list[Any], **options: Any) -> Callable[[Any], Any]:
    return lambda m: m.to_datetime(
        m.Series(values, index=[4 + i for i in range(len(values))]), **options
    ).dt.as_unit("ns")


BEYOND: list[Callable[[Any], Any]] = [
    index_of(["1500-01-01"]),
    index_of(["2024-01-01", "1500-01-01", "1400-01-01"]),
    index_of(["2024-01-01", None, "2500-01-01"]),
    index_of(["1500-01-01T10:00+01:00"], utc=True),
    column_of(["1500-01-01 10:30:15.123456"]),
    column_of(["2262-04-11 23:47:16.854776"]),
    column_of(["1677-09-21 00:12:43.145224"]),
]


@pytest.mark.parametrize("build", BEYOND)
def test_the_first_instant_beyond_is_named(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The same type as pandas, compared by name, and the same words."""
    import pandas as pd

    with pytest.raises(Exception) as expected:
        build(pd)
    with pytest.raises(
        firepanda.errors.OutOfBoundsDatetime, match="^" + re.escape(str(expected.value)) + "$"
    ):
        build(firepanda)
    assert type(expected.value).__name__ == "OutOfBoundsDatetime"


def test_the_two_ends_restate(firepanda: ModuleType) -> None:
    """The last microsecond at either end still fits in nanoseconds."""
    build = column_of(["2262-04-11 23:47:16.854775", "1677-09-21 00:12:43.145225", None])
    answer = build(firepanda)
    assert str(answer.dtype) == "datetime64[ns]"
    assert [None if value is None else str(value) for value in answer.tolist()] == [
        "2262-04-11 23:47:16.854775",
        "1677-09-21 00:12:43.145225",
        None,
    ]


def test_another_unit_is_not_checked_here(firepanda: ModuleType) -> None:
    """Milliseconds reach the sixteenth century, so nothing is raised."""
    answer = firepanda.to_datetime(["1500-01-01"]).as_unit("ms")
    assert str(answer.dtype) == "datetime64[ms]"
