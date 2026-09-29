"""Upsampling a resampler against pandas, compared by repr.

pandas keeps the rule on the index it builds as its freq and prints it as a
`Freq:` line, while an index here does not carry a freq yet, so the
comparison drops that line and the `freq=` part of an index repr.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")

STAMPS = ["2026-01-01T00:00", "2026-01-01T12:00", "2026-01-03T00:00", "2026-01-04T06:00"]
TWICE = [STAMPS[0], STAMPS[0], STAMPS[2], STAMPS[3]]


def series(lib: ModuleType, stamps: list[str] = STAMPS) -> Any:
    return lib.Series([1.0, 2.0, 3.0, 4.0], index=lib.to_datetime(stamps), name="x")


def frame(lib: ModuleType) -> Any:
    return lib.DataFrame(
        {"a": [1, 2, 3, 4], "b": ["p", "q", "r", "s"]}, index=lib.to_datetime(STAMPS)
    )


def gappy(lib: ModuleType) -> Any:
    return lib.DataFrame({"a": [1.0, None, 3, 4], "b": [1, 2, 3, 4]}, index=lib.to_datetime(STAMPS))


def keyed(lib: ModuleType) -> Any:
    return frame(lib).assign(k=["x", "y", "x", "y"])


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "ffill": lambda lib: series(lib).resample("D").ffill(),
    "ffill-6h": lambda lib: series(lib).resample("6h").ffill(),
    "ffill-limit": lambda lib: series(lib).resample("6h").ffill(limit=1),
    "bfill": lambda lib: series(lib).resample("6h").bfill(),
    "nearest": lambda lib: series(lib).resample("6h").nearest(),
    "nearest-limit": lambda lib: series(lib).resample("6h").nearest(limit=1),
    "asfreq": lambda lib: series(lib).resample("6h").asfreq(),
    "asfreq-fill": lambda lib: series(lib).resample("6h").asfreq(fill_value=0),
    "asfreq-label-right": lambda lib: series(lib).resample("D", label="right").asfreq(),
    "ffill-closed-right": lambda lib: series(lib).resample("D", closed="right").ffill(),
    "bfill-both-right": lambda lib: (
        series(lib).resample("D", closed="right", label="right").bfill()
    ),
    "frame-ffill": lambda lib: frame(lib).resample("12h").ffill(),
    "frame-column-ffill": lambda lib: frame(lib).resample("12h")["a"].ffill(),
    "frame-asfreq": lambda lib: frame(lib).resample("12h").asfreq(),
    "interpolate": lambda lib: series(lib).resample("6h").interpolate(),
    "interpolate-coarse": lambda lib: series(lib).resample("2D").interpolate(),
    "interpolate-time": lambda lib: series(lib).resample("5h").interpolate(method="time"),
    "interpolate-inplace": lambda lib: series(lib).resample("6h").interpolate(inplace=True),
    "on": lambda lib: frame(lib).rename_axis("t").reset_index().resample("D", on="t").ffill(),
    "repeated-asfreq": lambda lib: series(lib, TWICE).resample("6h").asfreq(),
    "repeated-ffill": lambda lib: series(lib, TWICE).resample("6h").ffill(),
    "empty": lambda lib: series(lib).iloc[:0].resample("D").ffill(),
    "index": lambda lib: series(lib).resample("12h").ffill().index,
    "grouped-ffill": lambda lib: keyed(lib).groupby("k").resample("D").ffill(),
    "grouped-interpolate": lambda lib: (
        keyed(lib)[["a", "k"]].groupby("k").resample("D").interpolate()
    ),
    "picked-limit": lambda lib: gappy(lib).resample("D")["a"].ffill(limit=1),
    "sum-index": lambda lib: series(lib).resample("12h").sum().index,
    "ohlc": lambda lib: series(lib).resample("D").ohlc(),
    "agg": lambda lib: frame(lib).resample("D").agg({"a": "sum", "b": "first"}),
    "where": lambda lib: series(lib).where(series(lib) > 2),
    "interpolate-over-dates": lambda lib: lib.Series(
        [1.0, None, 4.0], index=lib.to_datetime(STAMPS[:3])
    ).interpolate(),
    "time-over-dates": lambda lib: lib.Series(
        [1.0, None, 4.0], index=lib.to_datetime(STAMPS[:3])
    ).interpolate(method="time"),
    "index-over-dates": lambda lib: lib.Series(
        [1.0, None, 4.0], index=lib.to_datetime(STAMPS[:3])
    ).interpolate(method="index"),
    "time-over-spans": lambda lib: lib.Series(
        [1.0, None, 4.0], index=lib.to_timedelta(["1D", "2D", "5D"])
    ).interpolate(method="time"),
    "time-over-numbers": lambda lib: lib.Series([1.0, None, 4.0], index=[1, 2, 3]).interpolate(
        method="time"
    ),
}

for method in ("ffill", "bfill", "nearest", "asfreq", "interpolate"):
    CASES[f"picked-{method}"] = lambda lib, method=method: getattr(
        gappy(lib).resample("D")["a"], method
    )()
    CASES[f"picked-list-{method}"] = lambda lib, method=method: getattr(
        gappy(lib).resample("D")[["a"]], method
    )()


def mistake(error: Exception) -> str:
    kind = next(kind.__name__ for kind in type(error).__mro__ if kind.__module__ == "builtins")
    return f"{kind}: {error}"


def outcome(build: Callable[[], Any]) -> str:
    try:
        shown = repr(build()).replace("string", "str")
    except Exception as error:
        return mistake(error)
    return re.sub(r"Freq: [^,]*, |, freq=.*\)$", "", shown)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))
