"""Rolling windows measured as a span of time, compared with pandas.

A window given as text, a `Timedelta` or a fixed offset reaches back that far
along the row labels, or along the column `on` names, and `on` leaves its
column out of the reduction and puts it back into the answer. Picking columns
out of a window keeps the window.
"""

from __future__ import annotations

import datetime
import importlib.util
import random
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

TIMES = [
    "2024-01-01 00:00:00",
    "2024-01-01 00:00:10",
    "2024-01-01 00:00:10",
    "2024-01-01 00:00:45",
    "2024-01-01 00:01:00",
]


def frame(m: ModuleType) -> Any:
    t = m.to_datetime(TIMES).as_unit("s")
    return m.DataFrame({"a": [1.0, 2, float("nan"), 4, 5], "t": t, "b": [1, 2, 3, 4, 5]})


def column(m: ModuleType) -> Any:
    return m.Series([1.0, 2, 3, 4, 5], index=m.to_datetime(TIMES).as_unit("s"), name="x")


REDUCTIONS = ["sum", "mean", "count", "min", "max", "std", "var", "median", "skew"]
ENDS = ["right", "left", "both", "neither"]


def reduced(kind: str, closed: str, center: bool) -> Callable[[ModuleType], Any]:
    """One reduction over the frame's window of thirty seconds on `t`."""
    return lambda m: getattr(frame(m).rolling("30s", on="t", closed=closed, center=center), kind)()


ANSWERS: dict[str, Callable[[ModuleType], Any]] = {
    f"{kind} {closed}{' centred' if center else ''}": reduced(kind, closed, center)
    for kind in REDUCTIONS
    for closed in ENDS
    for center in (False, True)
}
ANSWERS.update(
    {
        "first": lambda m: column(m).rolling("30s").first(),
        "nunique": lambda m: column(m).rolling("30s").nunique(),
        "apply": lambda m: column(m).rolling("30s").apply(lambda x: x.max() - x.min()),
        "quantile": lambda m: column(m).rolling("30s").quantile(0.3),
        "sem": lambda m: column(m).rolling("30s").sem(),
        "corr": lambda m: column(m).rolling("30s").corr(column(m) ** 2),
        "rank": lambda m: column(m).rolling("30s").rank(),
        "min_periods 0": lambda m: column(m).rolling("30s", closed="left", min_periods=0).sum(),
        "min_periods 3": lambda m: frame(m).rolling("30s", on="t", min_periods=3).sum(),
        "falling": lambda m: column(m)[::-1].rolling("25s").mean(),
        "falling centred": lambda m: column(m)[::-1].rolling("25s", center=True).mean(),
        "a Timedelta": lambda m: column(m).rolling(m.Timedelta("20s")).sum(),
        "a timedelta": lambda m: column(m).rolling(datetime.timedelta(seconds=30)).sum(),
        "an offset": lambda m: column(m).rolling(m.offsets.Second(30)).sum(),
        "two units": lambda m: column(m).rolling("1min30s").sum(),
        "no span": lambda m: column(m).rolling("0s").sum(),
        "row labels of a frame": lambda m: frame(m).set_index("t").rolling("30s").sum(),
        "durations": lambda m: (
            m.Series([1.0, 2, 3], index=m.to_timedelta([0, 1, 5], unit="s")).rolling("2s").sum()
        ),
        "one column": lambda m: frame(m).rolling("30s", on="t")["b"].sum(),
        "some columns": lambda m: frame(m).rolling("30s", on="t")[["a"]].sum(),
        "some columns with on": lambda m: frame(m).rolling("30s", on="t")[["b", "t", "a"]].max(),
        "on with rows": lambda m: frame(m).drop(columns="t").rolling(2, on="b").sum(),
        "a column of rows": lambda m: frame(m).rolling(2)["a"].sum(),
        "a column expanding": lambda m: frame(m).expanding()["b"].sum(),
        "what it holds": lambda m: [
            (r.window, r.on, r.min_periods, list(r.obj.columns))
            for r in [frame(m).rolling("30s", on="t")]
        ],
        "reprs": lambda m: (
            repr(frame(m).rolling("30s", on="t")),
            repr(column(m).rolling(2, min_periods=1, closed="both")),
            repr(frame(m).rolling(2)["a"]),
            repr(column(m).expanding()),
        ),
    }
)

MISTAKES: dict[str, Callable[[ModuleType], Any]] = {
    "a month": lambda m: column(m).rolling("1ME"),
    "not a frequency": lambda m: column(m).rolling("xx"),
    "rows that are not times": lambda m: column(m).reset_index(drop=True).rolling("30s"),
    "times out of order": lambda m: column(m).iloc[[1, 0, 2]].rolling("30s"),
    "a step": lambda m: column(m).rolling("30s", step=2),
    "on a column not there": lambda m: frame(m).rolling("30s", on="zz"),
    "on over a column": lambda m: column(m).rolling("30s", on="t"),
    "on a column of numbers": lambda m: frame(m).rolling("30s", on="a"),
    "a gap in the times": lambda m: m.Series(
        [1.0, 2, 3], index=m.DatetimeIndex(["2024-01-01", None, "2024-01-02"])
    ).rolling("1D"),
    "a column not there": lambda m: frame(m).rolling(2)["zz"],
    "columns not there": lambda m: frame(m).rolling(2)[["a", "zz"]],
    "picked twice": lambda m: frame(m).rolling(2)["a"]["a"],
    "a column of a column": lambda m: frame(m)["a"].rolling(2)["a"],
}


def shown(answer: Any) -> Any:
    """An answer's columns, labels and values, spelled the same in both."""

    def values(part: Any) -> list[Any]:
        return [
            None if value != value else round(value, 9) if isinstance(value, float) else str(value)
            for value in part.tolist()
        ]

    if hasattr(answer, "columns"):
        labels = [str(label) for label in answer.index.tolist()]
        return list(answer.columns), labels, {name: values(answer[name]) for name in answer.columns}
    if hasattr(answer, "tolist"):
        return answer.name, [str(label) for label in answer.index.tolist()], values(answer)
    return answer


@needs_pandas
@pytest.mark.parametrize("name", list(ANSWERS))
def test_an_answer_matches_pandas(firepanda: ModuleType, name: str) -> None:
    """The columns, the labels and every value."""
    import pandas as pd

    assert shown(ANSWERS[name](firepanda)) == shown(ANSWERS[name](pd))


@needs_pandas
@pytest.mark.parametrize("name", list(MISTAKES))
def test_a_mistake_raises_what_pandas_raises(firepanda: ModuleType, name: str) -> None:
    """The same kind of error and the same words."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        MISTAKES[name](pd)
    with pytest.raises(theirs.type) as mine:
        MISTAKES[name](firepanda)
    assert str(mine.value) == str(theirs.value)


@needs_pandas
@pytest.mark.parametrize("seed", range(40))
def test_uneven_times_match_pandas(firepanda: ModuleType, seed: int) -> None:
    """Times with repeats and long gaps, under every end and a centred window."""
    import pandas as pd

    rng = random.Random(seed)
    rows = rng.randint(1, 40)
    seconds = sorted(rng.choice([rng.randint(0, 300), rng.randint(0, 20)]) for _ in range(rows))
    values = [rng.choice([None, rng.uniform(-5, 5), float(rng.randint(0, 3))]) for _ in range(rows)]
    span = rng.choice(["1s", "3s", "7s", "30s", "90s"])
    closed = rng.choice(ENDS)
    center = rng.random() < 0.5
    least = rng.choice([None, 0, 1, 2, 4])
    kind = rng.choice(["sum", "mean", "count", "max", "var", "median", "min"])

    def answer(m: ModuleType) -> Any:
        data = [float("nan") if value is None else value for value in values]
        index = m.to_datetime(seconds, unit="s")
        window = m.Series(data, index=index).rolling(
            span, closed=closed, center=center, min_periods=least
        )
        return getattr(window, kind)()

    assert shown(answer(firepanda)) == shown(answer(pd))


@needs_pandas
def test_the_bounds_are_pandas_bounds(firepanda: ModuleType) -> None:
    """Where each window starts and stops, against pandas' own function, rising and falling."""
    import numpy as np
    from pandas._libs.window.indexers import calculate_variable_window_bounds

    rng = random.Random(7)
    window = firepanda.Series([1.0]).rolling(1)
    for _ in range(500):
        rows = rng.randint(1, 30)
        axis = np.array(sorted(rng.randint(-50, 300) for _ in range(rows)), dtype="int64")
        if rng.random() < 0.3:
            axis = axis[::-1].copy()
        span = rng.randint(0, 40)
        closed = rng.choice(ENDS)
        center = rng.random() < 0.5
        starts, stops = calculate_variable_window_bounds(rows, span, None, center, closed, axis)
        window._axis, window._span, window._closed, window._center = axis, span, closed, center
        mine = window._span_bounds()
        assert mine[0].tolist() == starts.tolist()
        assert mine[1].tolist() == stops.tolist()
