"""`pct_change` with a `freq`, which moves the labels rather than the values, as pandas does.

pandas works each value out over the one `freq` earlier, lining the two up by
label, and reads the answer back onto the labels it started with. A group by
does that group by group. Each test runs the same code on both libraries and
compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def days(lib: Any) -> Any:
    return lib.date_range("2024-01-01", periods=5, freq="D")


def series(lib: Any) -> Any:
    return lib.Series([1.0, 2.0, 4.0, 5.0, 10.0], index=days(lib))


def frame(lib: Any) -> Any:
    return lib.DataFrame({"a": [1.0, 2.0, 4.0, 5.0, 10.0], "b": [2, 4, 4, 8, 8]}, index=days(lib))


def keyed(lib: Any) -> Any:
    return lib.DataFrame({"k": [1, 2, 1, 2, 1], "v": [1.0, 2.0, 4.0, 5.0, 10.0]}, index=days(lib))


BUILDS = {
    "series": lambda lib: series(lib).pct_change(freq="D"),
    "series two days": lambda lib: series(lib).pct_change(freq="2D"),
    "series periods": lambda lib: series(lib).pct_change(periods=2, freq="D"),
    "series gap": lambda lib: series(lib).drop(days(lib)[2]).pct_change(freq="D"),
    "frame": lambda lib: frame(lib).pct_change(freq="D"),
    "frame two days": lambda lib: frame(lib).pct_change(freq="2D"),
    "group by": lambda lib: keyed(lib).groupby("k").pct_change(freq="2D"),
    "group by picked": lambda lib: keyed(lib).groupby("k")[["v"]].pct_change(freq="2D"),
    "group by column": lambda lib: list(keyed(lib).groupby("k")["v"].pct_change(freq="2D")),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_pct_change_by_freq_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_labels_that_are_not_times_are_refused(firepanda: Any) -> None:
    with pytest.raises(NotImplementedError) as theirs:
        pd.Series([1.0, 2.0]).pct_change(freq="D")
    with pytest.raises(NotImplementedError) as mine:
        firepanda.Series([1.0, 2.0]).pct_change(freq="D")
    assert str(mine.value) == str(theirs.value)
