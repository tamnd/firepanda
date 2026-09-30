"""`DataFrame.join` with a list of frames, which pandas joins by a road of its own.

When every frame labels its rows once, pandas lays them side by side and puts
the rows in the order of this frame for a left join, of the last frame for a
right one, sorted for an outer one. When a label repeats, it merges them one
after another. Each test runs the same code on both libraries and compares
what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def frames(lib: Any) -> dict[str, Any]:
    return {
        "a": lib.DataFrame({"x": [1, 2, 3]}, index=["a", "b", "c"]),
        "b": lib.DataFrame({"y": [4, 5]}, index=["b", "d"]),
        "c": lib.DataFrame({"z": [6.0, 7.0]}, index=["c", "a"]),
        "d": lib.DataFrame({"w": [8, 9, 10]}, index=["a", "a", "b"]),
        "s": lib.Series([1, 2], index=["a", "b"], name="s"),
    }


HOWS = ["left", "right", "outer", "inner"]


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("how", HOWS)
def test_unique_labels_are_laid_side_by_side(firepanda: Any, how: str, sort: bool) -> None:
    def build(lib: Any) -> Any:
        made = frames(lib)
        return made["c"].join([made["b"], made["a"]], how=how, sort=sort)

    assert repr(build(firepanda)) == repr(build(pd))


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("how", HOWS)
def test_repeated_labels_are_merged_in_turn(firepanda: Any, how: str, sort: bool) -> None:
    def build(lib: Any) -> Any:
        made = frames(lib)
        return made["a"].join([made["d"], made["c"]], how=how, sort=sort)

    assert repr(build(firepanda)) == repr(build(pd))


@pytest.mark.parametrize("members", [["s", "c"], ["b"]])
def test_series_and_single_frames_join_too(firepanda: Any, members: list[str]) -> None:
    def build(lib: Any) -> Any:
        made = frames(lib)
        return made["a"].join([made[name] for name in members])

    assert repr(build(firepanda)) == repr(build(pd))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"on": "x"}, "Joining multiple DataFrames only supported for joining on index"),
        ({"rsuffix": "_r"}, "Suffixes not supported when joining multiple DataFrames"),
    ],
)
def test_what_pandas_refuses_for_a_list(
    firepanda: Any, arguments: dict[str, Any], message: str
) -> None:
    for lib in (firepanda, pd):
        made = frames(lib)
        with pytest.raises(ValueError, match=message):
            made["a"].join([made["b"], made["c"]], **arguments)


def test_a_repeated_column_is_refused(firepanda: Any) -> None:
    for lib in (firepanda, pd):
        made = frames(lib)
        with pytest.raises(ValueError, match="Indexes have overlapping values"):
            made["a"].join([made["a"], made["b"]])
