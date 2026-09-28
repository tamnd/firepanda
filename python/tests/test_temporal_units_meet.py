"""Instants and spans in different units, stacked or converted, against pandas.

pandas keeps the finest unit when `concat` meets two, so seconds and
microseconds make microseconds, for columns and for row labels alike. A mix of
zones or of instants and spans is its object column, which firepanda refuses.
`astype` to the same kind in another unit is `as_unit`.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def coarse(m: ModuleType) -> Any:
    return m.Series([m.Timestamp("2020-01-02"), None]).dt.as_unit("s")


def fine(m: ModuleType) -> Any:
    return m.Series([m.Timestamp("2021-01-02 03:04:05.123456")]).dt.as_unit("us")


def spans(m: ModuleType, unit: str) -> Any:
    return m.Series([m.Timedelta("1.5s"), None]).dt.as_unit(unit)


def shown(value: Any) -> Any:
    """The type and the values of a column, or of each column of a frame."""
    if hasattr(value, "columns"):
        return {name: shown(value[name]) for name in value.columns}
    return str(value.dtype), [repr(v) for v in value.tolist()]


CALLS: dict[str, Callable[[ModuleType], Any]] = {
    "series": lambda m: m.concat([coarse(m), fine(m)], ignore_index=True),
    "finer first": lambda m: m.concat([fine(m), coarse(m).dt.as_unit("ns")], ignore_index=True),
    "zoned": lambda m: m.concat(
        [coarse(m).dt.tz_localize("UTC"), fine(m).dt.tz_localize("UTC")], ignore_index=True
    ),
    "spans": lambda m: m.concat([spans(m, "s"), spans(m, "ms")], ignore_index=True),
    "frames": lambda m: m.concat(
        [m.DataFrame({"x": coarse(m)}), m.DataFrame({"x": fine(m), "y": [1]})], ignore_index=True
    ),
    "labels": lambda m: m.concat(
        [
            m.Series([1, 2], index=m.DatetimeIndex(coarse(m).fillna(m.Timestamp("2020-05-05")))),
            m.Series([3], index=m.DatetimeIndex(fine(m))),
        ]
    ).index.tolist(),
    "astype": lambda m: coarse(m).astype("datetime64[ms]"),
    "astype coarser": lambda m: fine(m).astype("datetime64[s]"),
    "astype zoned": lambda m: fine(m).dt.tz_localize("UTC").astype("datetime64[ms, UTC]"),
    "astype spans": lambda m: spans(m, "us").astype("timedelta64[ms]"),
    "astype frame": lambda m: m.DataFrame({"a": fine(m), "b": [1]}).astype(
        {"a": "datetime64[s]", "b": "float64"}
    ),
}


@needs_pandas
@pytest.mark.parametrize("name", list(CALLS))
def test_units_meet_as_in_pandas(firepanda: ModuleType, name: str) -> None:
    """The unit, the values and the gaps all match."""
    import pandas as pd

    mine = CALLS[name](firepanda)
    them = CALLS[name](pd)
    if isinstance(them, list):
        assert [repr(v) for v in mine] == [repr(v) for v in them]
    else:
        assert shown(mine) == shown(them)


def test_a_zoned_column_is_not_made_naive_by_astype(firepanda: ModuleType) -> None:
    """pandas refuses to drop a zone through `astype`, with a `TypeError`."""
    with pytest.raises(TypeError, match="timezone-aware dtype to timezone-naive"):
        fine(firepanda).dt.tz_localize("UTC").astype("datetime64[ms]")


def test_a_mix_of_zones_is_refused(firepanda: ModuleType) -> None:
    """pandas answers an object column, which firepanda has no type for."""
    parts = [coarse(firepanda).dt.tz_localize("UTC"), fine(firepanda).dt.tz_localize("Asia/Tokyo")]
    with pytest.raises(firepanda.errors.UnsupportedError):
        firepanda.concat(parts, ignore_index=True)
