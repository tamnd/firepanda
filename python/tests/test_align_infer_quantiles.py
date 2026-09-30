"""`align` by one level, `ambiguous="infer"`, quantiles as an index, and instants compared.

pandas lines a flat index up with one level of a MultiIndex, the join picking
only which rows of the MultiIndex stay. It infers which side of a repeated
hour each reading is on from where the readings go back, takes quantiles as
an index or a series as well as a list, and compares two timestamps by the
instant, so the two sides of a repeated hour are not equal. Each test here
runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

WALL = [
    "2024-11-03 00:30",
    "2024-11-03 01:00",
    "2024-11-03 01:30",
    "2024-11-03 01:00",
    "2024-11-03 01:30",
    "2024-11-03 02:30",
]


def layered(lib: Any) -> Any:
    index = lib.MultiIndex.from_arrays([["b", "a", "b", "c"], [1, 2, 2, 1]], names=["k", "n"])
    return lib.DataFrame({"v": [1, 2, 3, 4]}, index=index)


def flat(lib: Any) -> Any:
    return lib.DataFrame({"w": [10, 20, 30]}, index=lib.Index(["a", "b", "z"], name="k"))


def by_n(lib: Any) -> Any:
    return lib.Series([7, 8], index=lib.Index([2, 1], name="n"))


def readings(lib: Any) -> Any:
    return lib.Series(lib.to_datetime(WALL))


BUILDS: dict[str, Any] = {
    f"align {join}{' flipped' if flipped else ''}": (
        lambda lib, join=join, flipped=flipped: (
            flat(lib).align(layered(lib), join=join, level="k")
            if flipped
            else layered(lib).align(flat(lib), join=join, level="k")
        )
    )
    for join in ("outer", "inner", "left", "right")
    for flipped in (False, True)
}
BUILDS |= {
    "align series": lambda lib: layered(lib)["v"].align(by_n(lib), level=1),
    "align frame and series": lambda lib: layered(lib).align(
        by_n(lib), level="n", axis=0, fill_value=0
    ),
    "infer": lambda lib: readings(lib).dt.tz_localize("US/Eastern", ambiguous="infer"),
    "infer index": lambda lib: lib.DatetimeIndex(lib.to_datetime(WALL)).tz_localize(
        "US/Eastern", ambiguous="infer"
    ),
    "infer twice": lambda lib: lib.Series(lib.to_datetime(WALL + WALL[1:5])).dt.tz_localize(
        "US/Eastern", ambiguous="infer"
    ),
    "infer nothing repeated": lambda lib: lib.Series(
        lib.to_datetime(["2024-06-01", None, "2024-07-01"])
    ).dt.tz_localize("US/Eastern", ambiguous="infer"),
    "quantiles index": lambda lib: lib.Series([1.0, 4.0, 2.0, 8.0, 5.0]).quantile(
        lib.Index([0.5, 0.9])
    ),
    "quantiles series frame": lambda lib: lib.DataFrame({"a": [1, 2, 3, 4]}).quantile(
        lib.Series([0.25, 0.75])
    ),
    "repeated hour unequal": lambda lib: [
        a == b
        for a, b in zip(
            readings(lib).dt.tz_localize("US/Eastern", ambiguous=True).tolist(),
            readings(lib).dt.tz_localize("US/Eastern", ambiguous=False).tolist(),
            strict=True,
        )
    ],
    "repeated hour order": lambda lib: sorted(
        readings(lib).dt.tz_localize("US/Eastern", ambiguous="infer").tolist()
    ),
    "nanoseconds unequal": lambda lib: (
        lib.Timestamp("2024-01-01 00:00:00.000000001")
        == lib.Timestamp("2024-01-01 00:00:00.000000002"),
        lib.Timestamp("2024-01-01 00:00:00.000000001")
        < lib.Timestamp("2024-01-01 00:00:00.000000002"),
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_align_infer_and_quantiles_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


REFUSED = {
    "no fall back": lambda lib: lib.Series(lib.to_datetime(WALL[:3])),
    "one reading": lambda lib: lib.Series(lib.to_datetime(["2024-11-03 01:30"])),
    "apart": lambda lib: (
        readings(lib)
        .dt.tz_localize("US/Eastern", ambiguous="infer")
        .dt.tz_localize(None)
        .dt.round("h")
    ),
}


@pytest.mark.parametrize("make", REFUSED.values(), ids=REFUSED.keys())
def test_what_cannot_be_inferred_is_refused_in_pandas_words(firepanda: Any, make: Any) -> None:
    with pytest.raises(ValueError) as theirs:
        make(pd).dt.tz_localize("US/Eastern", ambiguous="infer")
    with pytest.raises(ValueError, match=str(theirs.value)):
        make(firepanda).dt.tz_localize("US/Eastern", ambiguous="infer")
