"""`isin` on instant and span columns, compared with pandas.

pandas finds a row of datetimes or timedeltas by its value: an instant finds
a row when both have a zone or neither does, text and numbers find nothing,
and a gap is found by `NaT` or `None`. Each test runs the same code on both
libraries and compares the answers.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def naive(lib: Any) -> Any:
    return lib.Series(lib.to_datetime(["2024-01-02", "2024-01-01"]))


def spans(lib: Any) -> Any:
    return lib.Series(lib.to_timedelta(["1D", "2h"]))


BUILDS = {
    "timestamp": lambda lib: naive(lib).isin([lib.Timestamp("2024-01-01")]),
    "text": lambda lib: naive(lib).isin(["2024-01-01"]),
    "index": lambda lib: naive(lib).isin(lib.to_datetime(["2024-01-01"])),
    "series": lambda lib: naive(lib).isin(naive(lib).iloc[:1]),
    "aware against naive": lambda lib: naive(lib).isin([lib.Timestamp("2024-01-01", tz="UTC")]),
    "same zone": lambda lib: (
        naive(lib).dt.tz_localize("UTC").isin([lib.Timestamp("2024-01-01", tz="UTC")])
    ),
    "other zone": lambda lib: (
        naive(lib).dt.tz_localize("UTC").isin([lib.Timestamp("2024-01-01 09:00", tz="Asia/Tokyo")])
    ),
    "naive against aware": lambda lib: (
        naive(lib).dt.tz_localize("UTC").isin([lib.Timestamp("2024-01-01")])
    ),
    "span": lambda lib: spans(lib).isin([lib.Timedelta("2h")]),
    "span text": lambda lib: spans(lib).isin(["1D"]),
    "span number": lambda lib: spans(lib).isin([1]),
    "number": lambda lib: naive(lib).isin([1704067200000000000]),
    "empty": lambda lib: naive(lib).isin([]),
    "nat": lambda lib: lib.Series(lib.to_datetime(["2024-01-02", None])).isin([lib.NaT]),
    "none": lambda lib: lib.Series(lib.to_datetime(["2024-01-02", None])).isin([None]),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_isin_timed_is_pandas(firepanda: Any, make: Any) -> None:
    assert make(firepanda).tolist() == make(pd).tolist()


def test_frame_isin_timed_is_pandas(firepanda: Any) -> None:
    def build(lib: Any) -> Any:
        frame = lib.DataFrame({"d": naive(lib), "v": [1, 2]})
        return frame.isin([lib.Timestamp("2024-01-02"), 2])

    assert repr(build(firepanda)) == repr(build(pd))
