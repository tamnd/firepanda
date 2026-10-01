"""`tz_convert` and `tz_localize` on a frame's or a series' labels, and their step.

pandas gives the moved labels the frequency the index method gives them: a
clock put on keeps it, a day moved to another clock or taken off one drops it.
Each test runs the same code on both libraries and compares the frequency and
what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def zoned(lib: Any) -> Any:
    return lib.Series([1, 2], index=lib.date_range("2024-01-01", periods=2, tz="UTC"))


def naive(lib: Any) -> Any:
    return lib.Series([1, 2], index=lib.date_range("2024-01-01", periods=2))


MOVES = {
    "series convert": lambda lib: zoned(lib).tz_convert("US/Eastern"),
    "series take off": lambda lib: zoned(lib).tz_localize(None),
    "series put on": lambda lib: naive(lib).tz_localize("UTC"),
    "frame convert": lambda lib: zoned(lib).to_frame("v").tz_convert("Asia/Tokyo"),
    "frame put on": lambda lib: naive(lib).to_frame("v").tz_localize("UTC"),
}


@pytest.mark.parametrize("move", MOVES.values(), ids=MOVES.keys())
def test_moved_labels_keep_the_pandas_step(firepanda: Any, move: Any) -> None:
    mine, theirs = move(firepanda), move(pd)
    assert repr(mine.index.freq) == repr(theirs.index.freq)
    assert repr(mine) == repr(theirs)
