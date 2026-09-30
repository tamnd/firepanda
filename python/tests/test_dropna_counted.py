"""`dropna` counting values: `thresh`, the other axis and numbering again.

pandas keeps a row with at least `thresh` values among the columns looked at,
with every one of them under `how="any"` and with one under `how="all"`, and
with `axis=1` drops columns the same way, `subset` then naming rows.
`ignore_index` numbers what is left from zero. Each test here runs the same
code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def frame(lib: Any) -> Any:
    return lib.DataFrame(
        {"a": [1.0, None, None, 4.0], "b": [None, None, 3.0, 4.0], "c": ["x", None, "z", "w"]},
        index=[10, 11, 12, 13],
    )


def emptied(lib: Any, **options: Any) -> Any:
    held = frame(lib)
    held.dropna(inplace=True, **options)
    return held


BUILDS = {
    "thresh": lambda lib: frame(lib).dropna(thresh=2),
    "thresh one": lambda lib: frame(lib).dropna(thresh=1),
    "thresh subset": lambda lib: frame(lib).dropna(thresh=1, subset=["a", "b"]),
    "thresh none": lambda lib: frame(lib).dropna(thresh=0),
    "thresh half": lambda lib: frame(lib).dropna(thresh=1.5),
    "subset text": lambda lib: frame(lib).dropna(subset="a", thresh=1),
    "columns": lambda lib: frame(lib).dropna(axis=1),
    "columns all": lambda lib: frame(lib).dropna(axis="columns", how="all"),
    "columns thresh": lambda lib: frame(lib).dropna(axis=1, thresh=3),
    "columns subset": lambda lib: frame(lib).dropna(axis=1, subset=[12, 13]),
    "numbered": lambda lib: frame(lib).dropna(ignore_index=True),
    "numbered all": lambda lib: frame(lib).dropna(how="all", ignore_index=True),
    "in place": lambda lib: emptied(lib, thresh=2),
    "no columns": lambda lib: frame(lib)[[]].dropna(),
    "no columns all": lambda lib: frame(lib)[[]].dropna(how="all"),
    "series numbered": lambda lib: frame(lib)["a"].dropna(ignore_index=True),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_counted_drops_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


@pytest.mark.parametrize("axis", [0, 1])
def test_a_label_the_other_axis_lacks_is_pandas_key_error(firepanda: Any, axis: int) -> None:
    with pytest.raises(KeyError) as mine:
        frame(firepanda).dropna(axis=axis, subset=["q"])
    assert mine.value.args == (["q"],)
