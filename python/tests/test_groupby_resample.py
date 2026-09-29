"""`GroupBy.resample` against pandas, compared by repr."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")

STAMPS = ["2026-01-01", "2026-01-01", "2026-01-03", "2026-01-02", "2026-01-04"]


def frame(lib: ModuleType) -> Any:
    return lib.DataFrame(
        {"k": ["a", "b", "a", "b", "a"], "v": [1, 2, 3, 4, 5], "w": [1.5, 2.5, 3.5, 4.5, 5.5]},
        index=lib.to_datetime(STAMPS),
    )


def grouped(lib: ModuleType, rule: str = "D", **options: Any) -> Any:
    return frame(lib).groupby("k", **options).resample(rule)


def stamped(lib: ModuleType) -> Any:
    """The frame with its times in a column named `t`, for `on=`."""
    return frame(lib).rename_axis("t").reset_index()


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "sum": lambda lib: grouped(lib).sum(),
    "mean-two-days": lambda lib: grouped(lib, "2D").mean(),
    "size": lambda lib: grouped(lib).size(),
    "count": lambda lib: grouped(lib).count(),
    "asfreq": lambda lib: grouped(lib).asfreq(),
    "agg": lambda lib: grouped(lib).agg({"v": "sum", "w": "max"}),
    "column": lambda lib: grouped(lib)["v"].sum(),
    "column-attribute": lambda lib: grouped(lib).v.sum(),
    "columns": lambda lib: grouped(lib)[["v"]].sum(),
    "key-column": lambda lib: grouped(lib)["k"].count(),
    "key-column-first": lambda lib: grouped(lib)["k"].first(),
    "ohlc": lambda lib: grouped(lib)["v"].ohlc(),
    "series": lambda lib: frame(lib).groupby("k")["v"].resample("D").sum(),
    "series-no-keys": lambda lib: (
        frame(lib).groupby("k", group_keys=False)["w"].resample("D").max()
    ),
    "on": lambda lib: stamped(lib).groupby("k").resample("D", on="t").sum(),
    "unsorted": lambda lib: grouped(lib, sort=False).sum(),
    "no-keys": lambda lib: grouped(lib, group_keys=False).sum(),
    "not-as-index": lambda lib: grouped(lib, as_index=False).sum(),
    "two-keys": lambda lib: frame(lib).assign(j=1).groupby(["k", "j"]).resample("D").sum(),
    "ndim": lambda lib: (grouped(lib).ndim, grouped(lib)["v"].ndim, grouped(lib)[["v"]].ndim),
    "include-groups": lambda lib: frame(lib).groupby("k").resample("D", include_groups=True),
    "no-such-column": lambda lib: grouped(lib)["zz"],
    "column-of-column": lambda lib: frame(lib).groupby("k")["v"].resample("D")["v"],
    "no-such-attribute": lambda lib: grouped(lib).nope,
    "not-dates": lambda lib: lib.DataFrame({"k": [1], "v": [2]}).groupby("k").resample("D").sum(),
}


def mistake(error: Exception) -> str:
    kind = next(kind.__name__ for kind in type(error).__mro__ if kind.__module__ == "builtins")
    return f"{kind}: {error}"


def outcome(build: Callable[[], Any]) -> str:
    try:
        return repr(build()).replace("string", "str")
    except Exception as error:
        return mistake(error)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


def test_transform_is_refused_for_now() -> None:
    with pytest.raises(NotImplementedError, match="transform"):
        grouped(fp).transform("sum")
