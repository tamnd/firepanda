"""Instants and spans cast to text, checked against pandas.

pandas writes them the way it prints them: the date alone when every instant
is at midnight, as many fraction digits as the finest instant needs, a zone's
offset at the end, and each span as `1 days 02:03:04`. Missing values stay
missing.
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


def plain(values: list[Any]) -> list[Any]:
    """The values with every spelling of missing as one word."""
    return ["missing" if value is None or value != value else value for value in values]


def agrees(got: Any, want: Any) -> None:
    """The same labels, name, text and type, column by column for a frame."""
    assert list(got.index) == list(want.index)
    if hasattr(want, "columns"):
        assert list(got.columns) == list(want.columns)
        for name in want.columns:
            agrees(got[name], want[name])
        return
    assert plain(got.tolist()) == plain(want.tolist())
    assert str(got.dtype).replace("string", "str") == str(want.dtype)
    assert got.name == want.name


def instants(m: ModuleType, values: list[Any]) -> Any:
    """A named column of instants with labels of its own."""
    return m.to_datetime(m.Series(values, index=[3, 1, 2][: len(values)], name="t"))


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: instants(m, ["2020-01-01", "2020-01-02", None]).astype(str),
    lambda m: instants(m, ["2020-01-01 10:00:00", "2020-01-02 00:00:00"]).astype("str"),
    lambda m: instants(m, ["2020-01-01 10:00:00.000", "2020-01-02 00:00:00.500"]).astype(str),
    lambda m: instants(m, ["2020-01-01 00:00:00.000000", "2020-01-02 00:00:00.000001"]).astype(str),
    lambda m: (
        instants(m, ["2020-01-01 00:00", "2020-01-02 03:00"])
        .dt.tz_localize("Europe/Paris")
        .astype(str)
    ),
    lambda m: m.Series([m.Timedelta(hours=1), None, m.Timedelta(days=2, microseconds=5)]).astype(
        str
    ),
    lambda m: m.Series([m.Timedelta(0), m.Timedelta(days=-1, hours=3)], name="s").astype(str),
    lambda m: m.DataFrame({"t": m.to_datetime(["2020-01-01", "2020-01-02"]), "x": [1, 2]}).astype(
        str
    ),
    lambda m: m.DataFrame({"t": m.to_datetime(["2020-01-01", "2020-01-02"]), "x": [1, 2]}).astype(
        {"t": str}
    ),
    lambda m: m.Index(m.to_datetime(["2020-01-01 10:00:00", "2020-01-02 00:00:00"])).astype(str),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_text_is_pandas_text(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Dates, times, fractions, zones, spans, gaps, frames, mappings and an index."""
    import pandas as pd

    got, want = build(firepanda), build(pd)
    if not hasattr(want, "index"):
        assert got.tolist() == want.tolist()
        assert got.name == want.name
        return
    agrees(got, want)
