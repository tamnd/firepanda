"""`resample` on a calendar offset, whose bins are not all one length.

Weeks, months, quarters and years are laid bin by bin the way pandas lays
them: from the first day rolled back onto the offset, or a step before it when
the bins close on the right, to a step past the last day. An offset that lands
on the end of a period closes and names its bins on the right, and a bin
closed on the right holds every moment of its end day. Each test here runs the
same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

STAMPS = [
    "2024-01-01 00:00",
    "2024-01-15 10:00",
    "2024-02-01 00:00",
    "2024-02-20 00:00",
    "2024-03-05 00:00",
    "2024-03-31 23:00",
    "2024-06-30 00:00",
]


def series(lib: Any) -> Any:
    return lib.Series([1, 2, 3, 4, 5, 6, 7], index=lib.to_datetime(STAMPS), name="x")


def frame(lib: Any) -> Any:
    return lib.DataFrame(
        {"d": lib.to_datetime(STAMPS), "v": [1.5, 2, 3, 4, 5, 6, 7], "w": [1, 2, 3, 4, 5, 6, 7]}
    )


RULES = ["ME", "MS", "W", "W-MON", "2ME", "3MS", "2W", "QE", "QS", "QE-JAN", "YE", "YS"]
SIDES = [{}, {"closed": "left"}, {"closed": "right", "label": "left"}, {"label": "right"}]


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("sides", SIDES, ids=["default", "left", "right-left", "label-right"])
def test_calendar_bins_are_pandas(firepanda: Any, rule: str, sides: dict[str, str]) -> None:
    mine = series(firepanda).resample(rule, **sides)
    theirs = series(pd).resample(rule, **sides)
    assert repr(mine.sum()) == repr(theirs.sum())
    assert repr(mine.mean().index.freq) == repr(theirs.mean().index.freq)


BUILDS = {
    "on": lambda lib: frame(lib).resample("ME", on="d").sum(),
    "asfreq": lambda lib: frame(lib).set_index("d").resample("MS").asfreq(),
    "ffill": lambda lib: frame(lib).set_index("d").resample("W").ffill(),
    "ohlc": lambda lib: frame(lib).set_index("d")["v"].resample("QE").ohlc(),
    "size": lambda lib: series(lib).resample("ME").size(),
    "binner": lambda lib: series(lib).resample("ME").binner,
    "ngroups": lambda lib: series(lib).resample("ME").ngroups,
    "offset": lambda lib: series(lib).resample(lib.offsets.MonthEnd()).sum(),
    "picked": lambda lib: frame(lib).set_index("d").resample("ME")["w"].agg(["first", "last"]),
    "series list": lambda lib: series(lib).resample("ME").agg(["sum", "count"]),
    "frame list": lambda lib: frame(lib).set_index("d").resample("ME").agg(["sum", "max"]),
    "on list": lambda lib: frame(lib).resample("MS", on="d").agg(["mean"]),
    "mapping of lists": lambda lib: (
        frame(lib).set_index("d").resample("ME").agg({"v": ["sum", "min"], "w": "max"})
    ),
    "fixed list": lambda lib: frame(lib).set_index("d").resample("20D").agg(["sum", "count"]),
    "fixed binner": lambda lib: series(lib).resample("2h").binner[:3],
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_calendar_resamples_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_a_calendar_rule_warns_that_origin_does_nothing(firepanda: Any) -> None:
    with pytest.warns(RuntimeWarning, match="'origin' keyword does not take effect"):
        series(firepanda).resample("ME", origin="epoch")
