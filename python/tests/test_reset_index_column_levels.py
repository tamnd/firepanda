"""`DataFrame.reset_index` on a frame whose columns are a `MultiIndex`.

pandas names each column the old labels land in by a tuple, the label's name
at `col_level` and `col_fill` at every other level, or the name at every
level when `col_fill` is None. On columns of one level both are passed over.
Each test runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def tiers(lib: Any) -> Any:
    return lib.MultiIndex.from_tuples([("a", "x"), ("a", "y")], names=["u", "w"])


def keyed(lib: Any) -> Any:
    rows = lib.Index(["p", "q"], name="k")
    return lib.DataFrame([[1, 2], [3, 4]], index=rows, columns=tiers(lib))


def deep(lib: Any) -> Any:
    rows = lib.MultiIndex.from_tuples([("p", 1), ("q", 2)], names=["k", "n"])
    return lib.DataFrame([[1, 2], [3, 4]], index=rows, columns=tiers(lib))


def shown(frame: Any) -> str:
    return repr((frame.to_string(), list(frame.columns), list(frame.columns.names)))


BUILDS = {
    "default": lambda lib: keyed(lib).reset_index(),
    "level": lambda lib: keyed(lib).reset_index(col_level=1),
    "level by name": lambda lib: keyed(lib).reset_index(col_level="w"),
    "fill": lambda lib: keyed(lib).reset_index(col_fill="z"),
    "fill with the name": lambda lib: keyed(lib).reset_index(col_fill=None),
    "name everywhere": lambda lib: keyed(lib).reset_index(col_level=1, col_fill=None),
    "several levels": lambda lib: deep(lib).reset_index(),
    "several filled": lambda lib: deep(lib).reset_index(col_level=1, col_fill="f"),
    "one of several": lambda lib: deep(lib).reset_index(level="n", col_level=1),
    "unnamed": lambda lib: keyed(lib).rename_axis(None).reset_index(col_level=1),
    "dropped": lambda lib: keyed(lib).reset_index(drop=True, col_level=1),
    "names": lambda lib: keyed(lib).reset_index(names="r", col_level=1),
    "flat level": lambda lib: lib.DataFrame({"v": [1]}).reset_index(col_level=1),
    "flat fill": lambda lib: lib.DataFrame({"v": [1]}).reset_index(col_fill="z"),
}

MISTAKES = {
    "level past the last": (IndexError, lambda lib: keyed(lib).reset_index(col_level=5)),
    "level not a name": (KeyError, lambda lib: keyed(lib).reset_index(col_level="nope")),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_reset_index_under_levels_is_pandas(firepanda: Any, make: Any) -> None:
    assert shown(make(firepanda)) == shown(make(pd))


def test_in_place_keeps_the_levels(firepanda: Any) -> None:
    frame = keyed(firepanda)
    frame.reset_index(col_level=1, inplace=True)
    assert list(frame.columns) == [("", "k"), ("a", "x"), ("a", "y")]


@pytest.mark.parametrize(("kind", "make"), MISTAKES.values(), ids=MISTAKES.keys())
def test_what_pandas_refuses_is_refused(firepanda: Any, kind: type, make: Any) -> None:
    with pytest.raises(kind) as theirs:
        make(pd)
    with pytest.raises(kind) as mine:
        make(firepanda)
    assert str(mine.value) == str(theirs.value)
