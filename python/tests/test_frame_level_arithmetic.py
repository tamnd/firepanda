"""Arithmetic between two frames with `level=`, the flat frame read by one level.

pandas aligns two frames on both axes, and with `level` a frame on flat row
labels is read out once for every row of the other's MultiIndex whose label
on that level it holds, a label it lacks standing as a gap that `fill_value`
fills when the other side has a value. Each test runs the same code on both
libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def layered(lib: Any) -> Any:
    labels = lib.MultiIndex.from_arrays([list("aabb"), [1, 2, 1, 2]], names=["x", "y"])
    return lib.DataFrame({"p": [1.0, 2.0, 3.0, 4.0], "q": [5.0, None, 7.0, 8.0]}, index=labels)


def flat(lib: Any) -> Any:
    return lib.DataFrame(
        {"p": [1.0, None, 9.0], "r": [2.0, 3.0, 4.0]}, index=lib.Index(["a", "c", "b"], name="x")
    )


OPS = ["add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "radd", "rsub", "rtruediv"]


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("fill", [None, 1.0])
def test_frame_level_arithmetic_is_pandas(firepanda: Any, op: str, fill: Any) -> None:
    def make(lib: Any) -> Any:
        return getattr(layered(lib), op)(flat(lib), level="x", fill_value=fill)

    assert repr(make(firepanda)) == repr(make(pd))


BUILDS = {
    "level by number": lambda lib: layered(lib).mul(flat(lib), level=0),
    "second level": lambda lib: layered(lib).add(
        lib.DataFrame({"p": [10.0, 20.0]}, index=lib.Index([1, 2], name="y")), level="y"
    ),
    "eq": lambda lib: layered(lib).eq(flat(lib)[["p"]], level="x"),
    "lt": lambda lib: layered(lib).lt(flat(lib).reindex(columns=["p", "q"]), level="x"),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_frame_level_forms_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
