"""`concat` of parts labelled by a MultiIndex and parts labelled flat.

pandas meets the two as objects: each row keeps its label, a tuple for the
layered rows and the plain label for the others. Each test runs the same code
on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def layered(lib: Any) -> Any:
    labels = lib.MultiIndex.from_arrays([list("ab"), [1, 2]], names=["x", "y"])
    return lib.Series([1.0, 2.0], index=labels)


def flat(lib: Any) -> Any:
    return lib.Series([3.0], index=[0])


BUILDS = {
    "layered first": lambda lib: lib.concat([layered(lib), flat(lib)]),
    "flat first": lambda lib: lib.concat([flat(lib), layered(lib)]),
    "frames": lambda lib: lib.concat([layered(lib).to_frame("v"), flat(lib).to_frame("v")]),
    "text labels": lambda lib: lib.concat([layered(lib), lib.Series([4.0], index=["q"])]),
    "both layered": lambda lib: lib.concat([layered(lib), layered(lib)]),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_concat_layered_flat_is_pandas(firepanda: Any, make: Any) -> None:
    mine, theirs = make(firepanda), make(pd)
    assert repr(mine) == repr(theirs)
    assert repr(mine.index) == repr(theirs.index)
