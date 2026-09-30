"""`numeric_only=True` on a series reduction and on a group by's folds and picks.

pandas reads a series' one column whatever the flag says and refuses it only
on a column of Python objects. A group by keeps its number and flag columns
for `cumsum`, `cummax`, `idxmax`, `quantile` and the like, while a group by
over one column folds it whatever the flag says. Each case runs the same code
on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def mixed(lib: Any) -> Any:
    return lib.DataFrame(
        {
            "k": ["a", "b", "a"],
            "v": [1, 5, 3],
            "s": ["x", "y", "z"],
            "t": lib.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02"]),
            "b": [True, False, True],
        }
    )


def grouped(lib: Any) -> Any:
    return mixed(lib).groupby("k")


BUILDS = {
    "series sum": lambda lib: int(mixed(lib)["v"].sum(numeric_only=True)),
    "series text sum": lambda lib: mixed(lib)["s"].sum(numeric_only=True),
    "series flags mean": lambda lib: float(mixed(lib)["b"].mean(numeric_only=True)),
    "series kurt": lambda lib: mixed(lib)["v"].kurt(numeric_only=True),
    "series dates max": lambda lib: mixed(lib)["t"].max(numeric_only=True),
    "series dates mean": lambda lib: mixed(lib)["t"].mean(numeric_only=True),
    "idxmax": lambda lib: grouped(lib).idxmax(numeric_only=True),
    "idxmin": lambda lib: grouped(lib).idxmin(numeric_only=True),
    "idxmax selected": lambda lib: grouped(lib)[["v", "t"]].idxmax(numeric_only=True),
    "cumsum": lambda lib: grouped(lib).cumsum(numeric_only=True),
    "cummax": lambda lib: grouped(lib).cummax(numeric_only=True),
    "cumprod": lambda lib: grouped(lib).cumprod(numeric_only=True),
    "cumsum selected": lambda lib: grouped(lib)[["v", "s"]].cumsum(numeric_only=True),
    "one column cumsum": lambda lib: grouped(lib)["v"].cumsum(numeric_only=True),
    "quantile list": lambda lib: (
        mixed(lib)[["k", "v"]].groupby("k").quantile([0.25, 0.5], numeric_only=True)
    ),
    "quantile nearest": lambda lib: (
        mixed(lib)[["k", "v", "s"]]
        .groupby("k")
        .quantile(0.5, interpolation="nearest", numeric_only=True)
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_numeric_only_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


REFUSED = {
    "objects sum": (
        lambda lib: lib.Series(["x", "y"], dtype=object).sum(numeric_only=True),
        "Series.sum does not allow numeric_only=True",
    ),
    "objects kurt": (
        lambda lib: lib.Series(["x", "y"], dtype=object).kurt(numeric_only=True),
        "Series.kurt does not allow numeric_only=True",
    ),
    "objects max": (
        lambda lib: lib.Series(["x", "y"], dtype=object).max(numeric_only=True),
        "Series.max does not allow numeric_only=True",
    ),
    "one text column fold": (
        lambda lib: grouped(lib)["s"].cummax(numeric_only=True),
        "cummax is not supported for str dtype",
    ),
    "one text column quantile": (
        lambda lib: grouped(lib)["s"].quantile(numeric_only=True),
        "Cannot use numeric_only=True with SeriesGroupBy.quantile",
    ),
}


@pytest.mark.parametrize(("make", "words"), REFUSED.values(), ids=REFUSED.keys())
def test_numeric_only_refuses_as_pandas(firepanda: Any, make: Any, words: str) -> None:
    with pytest.raises(TypeError, match=words):
        make(pd)
    with pytest.raises(TypeError, match=words):
        make(firepanda)
