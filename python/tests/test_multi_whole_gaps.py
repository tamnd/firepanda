"""A level of whole numbers with a gap in it, which pandas keeps whole.

pandas codes a gap in a level as -1 rather than holding it among the level's
values, so a level of whole numbers stays int64 beside a gap and prints its
numbers whole, with NaN for the gap. Reading the level back row by row gives
floats, as NaN needs. Each test here runs the same code on both libraries and
compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def tuples(lib: Any) -> Any:
    return lib.MultiIndex.from_tuples(
        [("b", 2), ("a", 1), ("b", 1), ("a", 3), (None, 2), ("a", None)], names=["p", "q"]
    )


def frame(lib: Any) -> Any:
    return lib.DataFrame({"v": [1, 2, 3, 4, 5, 6]}, index=tuples(lib))


BUILDS = {
    "level from tuples": lambda lib: tuples(lib).levels[1],
    "level from arrays": lambda lib: lib.MultiIndex.from_arrays([["b", "a"], [2, None]]).levels[1],
    "level of floats": lambda lib: lib.MultiIndex.from_arrays([["b", "a"], [2.0, None]]).levels[1],
    "level from product": lambda lib: lib.MultiIndex.from_product([["b"], [2, None]]).levels[1],
    "values read back": lambda lib: tuples(lib).get_level_values(1),
    "index": lambda lib: tuples(lib),
    "frame": frame,
    "series": lambda lib: lib.Series(
        [1, 2], index=lib.MultiIndex.from_tuples([("b", 10), ("a", None)])
    ),
    "product frame": lambda lib: lib.DataFrame(
        {"v": [1, 2]}, index=lib.MultiIndex.from_product([["b"], [2, None]])
    ),
    "sorted per level": lambda lib: frame(lib).sort_index(ascending=[True, False]),
    "sorted the other way": lambda lib: frame(lib).sort_index(ascending=[False, True]),
    "sorted on a level": lambda lib: frame(lib).sort_index(level=1, ascending=[False]),
    "sorted on levels": lambda lib: frame(lib).sort_index(level=[1, 0], ascending=[False, True]),
    "series sorted per level": lambda lib: frame(lib)["v"].sort_index(ascending=[True, False]),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_whole_levels_with_gaps_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
