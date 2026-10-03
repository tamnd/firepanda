"""The settings a group by's window and resample hold, read as pandas reads them.

`window`, `min_periods`, `com` and the like are the same on every group's window,
so they are read off one window over no rows. A resample of a group by names its
bins the way pandas does, and both list what they have for `dir`.
"""

from __future__ import annotations

from types import ModuleType

import pytest


@pytest.fixture
def frame(firepanda: ModuleType) -> object:
    return firepanda.DataFrame({"a": [1, 1, 2], "b": [1.5, 2.5, 3.5]})


def test_rolling_settings(frame: object) -> None:
    """The window's own settings, and the rows it moves over without the keys."""
    window = frame.groupby("a").rolling(2, min_periods=1, center=True)
    assert window.window == 2 and window.min_periods == 1 and window.center
    assert window.closed is None and window.method == "single" and window.win_type is None
    assert window.obj.columns.tolist() == ["b"]
    assert window.ndim == 2 and window.exclusions == frozenset()
    assert {"sum", "window", "obj", "exclusions"} <= set(dir(window))
    assert window.sum()["b"].tolist()[1] == 4.0


def test_expanding_and_ewm_settings(frame: object) -> None:
    """An expanding window has no width, and an ewm keeps its decay."""
    growing = frame.groupby("a").expanding()
    assert growing.window is None and growing.min_periods == 1
    decaying = frame.groupby("a").ewm(com=1)
    assert decaying.com == 1 and decaying.adjust and not decaying.ignore_na
    assert decaying.span is None and decaying.times is None


def test_a_picked_column(frame: object) -> None:
    """Over one column the rows are that column and the count of axes is one."""
    window = frame.groupby("a")["b"].rolling(2)
    assert window.ndim == 1 and window.obj.name == "b"


def test_resample_settings(firepanda: ModuleType) -> None:
    """The bins' sides, width, origin and offset, as pandas names them."""
    stamps = firepanda.date_range("2024-01-01", periods=2)
    frame = firepanda.DataFrame({"a": [1, 1], "b": [1.0, 2.0]}, index=stamps)
    daily = frame.groupby("a").resample("D")
    assert (daily.closed, daily.label, daily.origin, daily.offset) == (
        "left",
        "left",
        "start_day",
        None,
    )
    assert daily.freq == firepanda.offsets.Day()
    assert daily.key is None and daily.convention == "e"
    assert {"a", "b", "sum", "closed", "freq"} <= set(dir(daily))
    monthly = frame.groupby("a").resample("ME")
    assert (monthly.closed, monthly.label) == ("right", "right")
    moved = frame.groupby("a").resample("2h", origin="epoch", offset="30min")
    assert moved.origin == "epoch" and moved.offset == firepanda.Timedelta("30min")
