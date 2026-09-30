"""An exponentially weighted mean that decays against real instants, against pandas."""

from __future__ import annotations

import datetime
import math
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

DAYS = ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05", "2024-01-09"]


def _times(module: Any) -> Any:
    return module.Series(module.to_datetime(DAYS))


def _values(module: Any) -> Any:
    return module.Series([1.0, None, 3.0, 4.0, 10.0], name="v")


CASES = {
    "mean": lambda m: _values(m).ewm(halflife="1D", times=_times(m)).mean(),
    "ignore-na": lambda m: _values(m).ewm(halflife="2D", times=_times(m), ignore_na=True).mean(),
    "min-periods": lambda m: _values(m).ewm(halflife="1D", times=_times(m), min_periods=2).mean(),
    "timedelta": lambda m: _values(m).ewm(halflife=m.Timedelta(hours=36), times=_times(m)).mean(),
    "python-timedelta": lambda m: (
        _values(m).ewm(halflife=datetime.timedelta(days=1), times=_times(m)).mean()
    ),
    "adjust-false": lambda m: _values(m).ewm(halflife="1D", times=_times(m), adjust=False).mean(),
    "com": lambda m: _values(m).ewm(com=3, halflife="1D", times=_times(m)).mean(),
    "frame": lambda m: (
        m.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5.0, 4.0, None, 2.0, 1.0]})
        .ewm(halflife="1D", times=_times(m))
        .mean()
    ),
    "unsorted": lambda m: (
        _values(m).ewm(halflife="1D", times=_times(m)[::-1].reset_index(drop=True)).mean()
    ),
    "grouped": lambda m: (
        m.DataFrame({"k": ["x", "y", "x", "y", "x"], "a": [1, 2, 3, 4, 5]})
        .groupby("k")["a"]
        .ewm(halflife="1D", times=_times(m))
        .mean()
    ),
    "unadjusted-gaps": lambda m: (
        m.Series([1.0, None, 3.0, None, None, 4.0]).ewm(alpha=0.5, adjust=False).mean()
    ),
    "repr": lambda m: _values(m).ewm(halflife="1D", times=_times(m)),
    "index": lambda m: _values(m).ewm(halflife="1D", times=m.to_datetime(DAYS)).mean(),
    "infinite": lambda m: (
        m.Series([-math.inf, math.inf, 2.0, None, 4.0])
        .ewm(halflife="1D", times=_times(m), adjust=False)
        .mean()
    ),
    "seconds": lambda m: (
        _values(m).ewm(halflife="2s", times=m.to_datetime([0, 1, 4, 9, 16], unit="s")).mean()
    ),
}


@pytest.mark.parametrize("case", list(CASES.values()), ids=list(CASES))
def test_a_timed_decay_answers_as_pandas(firepanda: ModuleType, case: Any) -> None:
    """Each answer prints as pandas prints it."""
    assert repr(case(firepanda)) == repr(case(pd))


REFUSED = {
    "sum": (NotImplementedError, lambda m: _values(m).ewm(halflife="1D", times=_times(m)).sum()),
    "std": (NotImplementedError, lambda m: _values(m).ewm(halflife="1D", times=_times(m)).std()),
    "alpha": (ValueError, lambda m: _values(m).ewm(alpha=0.5, times=_times(m))),
    "short": (ValueError, lambda m: _values(m).ewm(halflife="1D", times=_times(m)[:3])),
    "list": (ValueError, lambda m: _values(m).ewm(halflife="1D", times=list(DAYS))),
    "gap": (
        ValueError,
        lambda m: _values(m).ewm(halflife="1D", times=_times(m).where(_times(m).index != 2)),
    ),
    "com-unadjusted": (
        NotImplementedError,
        lambda m: _values(m).ewm(com=1, halflife="1D", times=_times(m), adjust=False),
    ),
    "duration-alone": (ValueError, lambda m: _values(m).ewm(halflife="1D")),
}


@pytest.mark.parametrize("case", list(REFUSED.values()), ids=list(REFUSED))
def test_a_timed_decay_refuses_as_pandas(firepanda: ModuleType, case: Any) -> None:
    """The class and the words are pandas'."""
    kind, call = case
    with pytest.raises(kind) as ours:
        call(firepanda)
    with pytest.raises(kind) as theirs:
        call(pd)
    assert str(ours.value) == str(theirs.value)
