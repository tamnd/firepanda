"""`PeriodIndex` and `period_range` against pandas, compared by repr."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def labels(lib: ModuleType) -> Any:
    return lib.PeriodIndex(["2026-01", "2026-03", None, "2025-12"], freq="M", name="p")


def labelled(lib: ModuleType) -> Any:
    return lib.Series([1, 2, 3, 4], index=labels(lib))


def months(lib: ModuleType) -> Any:
    return lib.Series([1.5, 2.5, 3.5], index=lib.period_range("2026-01", periods=3, freq="M"))


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "make-periods": lambda lib: lib.PeriodIndex(
        [lib.Period("2026-01", "M"), lib.Period("2026-02", "M")]
    ),
    "make-noname": lambda lib: lib.PeriodIndex(["2026-01"], freq="M"),
    "make-dtype": lambda lib: lib.PeriodIndex(["2026-01"], dtype="period[Q]"),
    "make-nofreq": lambda lib: lib.PeriodIndex(["2026-01"]),
    "make-mixed": lambda lib: lib.PeriodIndex(
        [lib.Period("2026-01", "M"), lib.Period("2026-01-01", "D")]
    ),
    "make-empty": lambda lib: lib.PeriodIndex([], freq="D"),
    "dtype": lambda lib: labels(lib).dtype,
    "freq": lambda lib: labels(lib).freq,
    "freqstr": lambda lib: labels(lib).freqstr,
    "name": lambda lib: labels(lib).name,
    "len": lambda lib: len(labels(lib)),
    "list": lambda lib: labels(lib).tolist(),
    "get0": lambda lib: labels(lib)[0],
    "get2": lambda lib: labels(lib)[2],
    "slice": lambda lib: labels(lib)[1:3],
    "year": lambda lib: labels(lib).year,
    "month": lambda lib: labels(lib).month,
    "quarter": lambda lib: labels(lib).quarter,
    "day": lambda lib: labels(lib).day,
    "dim": lambda lib: labels(lib).days_in_month,
    "leap": lambda lib: labels(lib).is_leap_year,
    "start": lambda lib: labels(lib).start_time,
    "end": lambda lib: labels(lib).end_time,
    "asfreq": lambda lib: labels(lib).asfreq("D"),
    "asfreq-s": lambda lib: labels(lib).asfreq("D", how="start"),
    "to_ts": lambda lib: labels(lib).to_timestamp(),
    "strftime": lambda lib: labels(lib).strftime("%Y/%m"),
    "shift": lambda lib: labels(lib).shift(2),
    "add": lambda lib: labels(lib) + 1,
    "sub": lambda lib: labels(lib) - 1,
    "eq": lambda lib: labels(lib) == lib.Period("2026-01", "M"),
    "min": lambda lib: labels(lib).min(),
    "max": lambda lib: labels(lib).max(),
    "mono": lambda lib: labels(lib).is_monotonic_increasing,
    "uniq": lambda lib: labels(lib).is_unique,
    "getloc": lambda lib: labels(lib).get_loc(lib.Period("2026-03", "M")),
    "getloc-str": lambda lib: labels(lib).get_loc("2026-03"),
    "contains": lambda lib: lib.Period("2026-03", "M") in labels(lib),
    "equals": lambda lib: labels(lib).equals(labels(lib)),
    "sort": lambda lib: labels(lib).sort_values(),
    "unique": lambda lib: labels(lib).unique(),
    "dropna": lambda lib: labels(lib).dropna(),
    "to_series": lambda lib: labels(lib).to_series(),
    "series-idx": lambda lib: lib.Series([1, 2, 3, 4], index=labels(lib)),
    "series-idx-type": lambda lib: type(lib.Series([1, 2, 3, 4], index=labels(lib)).index).__name__,
    "index-of": lambda lib: lib.Index([lib.Period("2026-01", "M"), lib.Period("2026-02", "M")]),
    "index-type": lambda lib: type(lib.Index([lib.Period("2026-01", "M")])).__name__,
    "isinst": lambda lib: isinstance(labels(lib), lib.Index),
    "range": lambda lib: lib.period_range("2026-01", periods=4, freq="M"),
    "range-end": lambda lib: lib.period_range("2026-01-01", "2026-01-05", freq="D", name="d"),
    "range-q": lambda lib: lib.period_range("2026Q1", "2027Q2", freq="Q"),
    "range-nofreq": lambda lib: lib.period_range("2026-01", "2026-04"),
    "range-p": lambda lib: lib.period_range(lib.Period("2026-01", "M"), periods=3),
    "range-bad": lambda lib: lib.period_range("2026-01", periods=3),
    "range-end-p": lambda lib: lib.period_range(end="2026-05", periods=3, freq="M"),
    "range-2m": lambda lib: lib.period_range("2026-01", periods=3, freq="2M"),
    "fields": lambda lib: lib.PeriodIndex.from_fields(year=[2026, 2027], month=[1, 5], freq="M"),
    "ordinals": lambda lib: lib.PeriodIndex.from_ordinals([0, 12], freq="M"),
    "hours": lambda lib: lib.period_range("2026-01-01", periods=3, freq="h"),
    "hours-hour": lambda lib: lib.period_range("2026-01-01", periods=3, freq="h").hour,
    "long": lambda lib: lib.period_range("2026-01-01", periods=120, freq="D"),
    "dtype-eq": lambda lib: labels(lib).dtype == "period[M]",
    "values": lambda lib: labels(lib).values,
    "array": lambda lib: labels(lib).array,
    "to_numpy": lambda lib: labels(lib).to_numpy(),
    "astype-str": lambda lib: labels(lib).astype(str),
    "inferred": lambda lib: labels(lib).inferred_type,
    "weekday": lambda lib: lib.period_range("2026-01-01", periods=3, freq="D").weekday,
    "dayofyear": lambda lib: lib.period_range("2026-01-01", periods=3, freq="D").dayofyear,
    "qyear": lambda lib: lib.period_range("2026Q1", periods=2, freq="Q-NOV").qyear,
    "week": lambda lib: lib.period_range("2026-01-01", periods=3, freq="D").week,
    "ndim": lambda lib: labels(lib).ndim,
    "hasnans": lambda lib: labels(lib).hasnans,
    "copy": lambda lib: labels(lib).copy(),
    "rename": lambda lib: labels(lib).rename("q"),
    "repr-freq": lambda lib: str(labels(lib)),
    "ser-type": lambda lib: type(labelled(lib).index).__name__,
    "loc-slice": lambda lib: months(lib).loc["2026-01":"2026-02"],
    "frame": lambda lib: lib.DataFrame(
        {"a": [1, 2, 3]}, index=lib.period_range("2026Q1", periods=3, freq="Q")
    ),
    "sort-idx": lambda lib: labelled(lib).sort_index(),
    "reset": lambda lib: months(lib).reset_index(),
    "vc": lambda lib: lib.Series(
        [lib.Period("2026-01", "M"), lib.Period("2026-03", "M"), lib.Period("2026-01", "M")]
    ).value_counts(),
    "gb": lambda lib: (
        lib.DataFrame(
            {
                "p": [
                    lib.Period("2026-01", "M"),
                    lib.Period("2026-03", "M"),
                    lib.Period("2026-01", "M"),
                ],
                "v": [1, 2, 3],
            }
        )
        .groupby("p")
        .sum()
    ),
    "gb-idx": lambda lib: (
        type(
            lib.DataFrame({"p": [lib.Period("2026-01", "M")], "v": [1]}).groupby("p").sum().index
        ).__name__
    ),
    "set-index": lambda lib: (
        type(
            lib.DataFrame({"p": [lib.Period("2026-01", "M")], "v": [1]}).set_index("p").index
        ).__name__
    ),
    "cat": lambda lib: lib.Series(
        [lib.Period("2026-01", "M"), lib.Period("2026-03", "M"), None]
    ).astype("category"),
    "cat-cats": lambda lib: (
        lib.Series([lib.Period("2026-01", "M"), lib.Period("2026-03", "M")])
        .astype("category")
        .cat.categories
    ),
    "ser-list-idx": lambda lib: lib.Series(
        [1, 2], index=[lib.Period("2026-01", "M"), lib.Period("2026-02", "M")]
    ),
    "concat": lambda lib: lib.concat([months(lib), months(lib)]).index,
    "reindex": lambda lib: months(lib).reindex(lib.period_range("2025-12", periods=3, freq="M")),
    "dtype-idx": lambda lib: lib.PeriodDtype("M").index_class,
    "to_frame": lambda lib: labels(lib).to_frame(),
    "union": lambda lib: lib.period_range("2026-01", periods=2, freq="M").union(
        lib.period_range("2026-02", periods=2, freq="M")
    ),
    "append": lambda lib: lib.period_range("2026-01", periods=2, freq="M").append(
        lib.period_range("2026-05", periods=1, freq="M")
    ),
    "map": lambda lib: labels(lib).map(lambda p: p.month if p is not lib.NaT else 0),
    "pickle": lambda lib: pickle.loads(pickle.dumps(labels(lib))),
    "loc-value": lambda lib: int(labelled(lib).loc[lib.Period("2026-03", "M")]),
    "loc-text": lambda lib: float(months(lib).loc["2026-02"]),
    "getitem-text": lambda lib: float(months(lib)["2026-02"]),
    "getitem-period": lambda lib: float(months(lib)[lib.Period("2026-02", "M")]),
    "isna-list": lambda lib: [bool(gap) for gap in labels(lib).isna()],
    "isin-list": lambda lib: [bool(hit) for hit in labels(lib).isin([lib.Period("2026-01", "M")])],
    "cat-gap": lambda lib: lib.Series([lib.Period("2026-01", "M"), None]).astype("category"),
}


def mistake(error: Exception) -> str:
    kind = next(kind.__name__ for kind in type(error).__mro__ if kind.__module__ == "builtins")
    return f"{kind}: {error}"


def outcome(build: Callable[[], Any]) -> str:
    try:
        return repr(build()).replace("firepanda._period_index", "pandas")
    except Exception as error:
        return mistake(error)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


def test_an_empty_index_keeps_its_type() -> None:
    empty = fp.PeriodIndex([], freq="Q")
    assert (str(empty.dtype), len(empty), isinstance(empty, fp.Index)) == ("period[Q-DEC]", 0, True)
