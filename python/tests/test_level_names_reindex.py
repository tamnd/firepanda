"""Level names through `stack` and `rename_axis`, `reindex(level=)`, `asfreq` and `info`.

`stack` carries the names of the column levels it moves and takes a level by
name. `rename_axis(index=)` with a mapping or a function renames the names the
levels have. `reindex(level=)` reads flat labels against one level of a
MultiIndex. `asfreq` keeps the step of its range and moves periods, and `info`
without its memory line ends on its last line. Each test runs the same code on
both libraries and compares what they print.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import pytest


def levels(lib: Any) -> Any:
    return lib.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("c", 3)], names=["k", "n"])


def layered(lib: Any) -> Any:
    return lib.DataFrame([[1, 5]], columns=levels(lib)[:2])


def flat(lib: Any) -> Any:
    return lib.Series([1, 2], index=lib.Index(["a", "b"], name="z"))


def stepped(lib: Any) -> Any:
    index = lib.date_range("2024-01-01 00:07", periods=4, freq="17min")
    return lib.Series(range(4), index=index)


def yearly(lib: Any) -> Any:
    return lib.Series([1, 2], index=lib.PeriodIndex(["2024", "2025"], freq="Y"))


def report(lib: Any, obj: Any) -> str:
    buf = io.StringIO()
    obj.info(buf=buf, memory_usage=False)
    return buf.getvalue()


BUILDS = {
    "stack first": lambda lib: layered(lib).stack(level=0),
    "stack by name": lambda lib: layered(lib).stack(level="n"),
    "stack both": lambda lib: layered(lib).stack(level=[0, 1]),
    "rename_axis mapping": lambda lib: (
        lib.Series(range(4), index=levels(lib)).rename_axis(index={"k": "K"}).index.names
    ),
    "rename_axis function": lambda lib: flat(lib).rename_axis(index=str.upper).index.name,
    "frame rename_axis": lambda lib: (
        lib.DataFrame({"v": range(4)}, index=levels(lib)).rename_axis(index={"n": "N"}).index.names
    ),
    "reindex onto levels": lambda lib: flat(lib).reindex(levels(lib), level=0),
    "reindex onto a named level": lambda lib: flat(lib).reindex(levels(lib), level="k"),
    "reindex from levels": lambda lib: lib.Series([1.0, 2, 3, 4], index=levels(lib)).reindex(
        ["b", "a", "z"], level=0
    ),
    "frame reindex from levels": lambda lib: lib.DataFrame(
        {"v": [1, 2, 3, 4]}, index=levels(lib)
    ).reindex(["c", "a"], level=0),
    "reindex fill": lambda lib: flat(lib).reindex(levels(lib), level=0, fill_value=0),
    "asfreq ffill": lambda lib: stepped(lib).asfreq("10min", method="ffill"),
    "asfreq index": lambda lib: stepped(lib).to_frame().asfreq("10min", method="bfill").index,
    "reindex keeps freq": lambda lib: (
        stepped(lib)
        .reindex(lib.date_range("2024-01-01", periods=3, freq="h"), method="ffill")
        .index
    ),
    "asfreq periods": lambda lib: yearly(lib).asfreq("M", how="start"),
    "asfreq periods end": lambda lib: yearly(lib).to_frame("v").asfreq("Q"),
    "info series": lambda lib: report(lib, lib.Series([1, 2], name="x")).replace(lib.__name__, ""),
    "info frame": lambda lib: report(lib, lib.DataFrame({"a": [1]})).replace(lib.__name__, ""),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_level_names_and_reindex_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_refusals_are_pandas(firepanda: Any) -> None:
    with pytest.raises(TypeError, match="Fill method not supported if level passed"):
        flat(firepanda).reindex(levels(firepanda), level=0, method="ffill")
    with pytest.raises(ValueError, match="to alter labels with a mapper"):
        flat(firepanda).rename_axis({"z": "Z"})
    with pytest.raises(NotImplementedError, match="'method' argument is not supported"):
        yearly(firepanda).asfreq("M", method="ffill")
    with pytest.raises(KeyError, match="Level q not found"):
        layered(firepanda).stack(level="q")
