"""`to_period`, `to_timestamp` and the `dt` accessor on periods against pandas, by repr."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def instants(lib: ModuleType) -> Any:
    return lib.to_datetime(["2026-01-15", "2026-03-02 10:30", None, "2025-12-31"], format="ISO8601")


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "dt-to_period": lambda lib: lib.Series(instants(lib), name="d").dt.to_period("M"),
    "dt-to_period-D": lambda lib: lib.Series(instants(lib)).dt.to_period("D"),
    "dt-to_period-Q": lambda lib: lib.Series(instants(lib)).dt.to_period("Q"),
    "dt-to_period-h": lambda lib: lib.Series(instants(lib)).dt.to_period("h"),
    "dt-to_period-none": lambda lib: lib.Series(instants(lib)).dt.to_period(),
    "dt-to_period-none-reg": lambda lib: lib.Series(
        lib.date_range("2026-01-01", periods=3, freq="D")
    ).dt.to_period(),
    "dti-to_period": lambda lib: instants(lib).to_period("M"),
    "dti-to_period-none": lambda lib: lib.date_range(
        "2026-01-01", periods=3, freq="MS"
    ).to_period(),
    "dti-to_period-none2": lambda lib: lib.date_range(
        "2026-01-01", periods=3, freq="D"
    ).to_period(),
    "dti-to_period-bad": lambda lib: instants(lib).to_period(),
    "ser-to_period": lambda lib: lib.Series(
        [1, 2, 3], index=lib.date_range("2026-01-01", periods=3, freq="D")
    ).to_period(),
    "ser-to_period-M": lambda lib: lib.Series(
        [1, 2, 3], index=lib.date_range("2026-01-01", periods=3, freq="D")
    ).to_period("M"),
    "ser-to_period-bad": lambda lib: lib.Series([1, 2, 3]).to_period(),
    "ser-to_ts-end": lambda lib: lib.Series(
        [1, 2, 3], index=lib.period_range("2026-01", periods=3, freq="M")
    ).to_timestamp(how="end"),
    "ser-to_ts-bad": lambda lib: lib.Series([1, 2, 3]).to_timestamp(),
    "frame-to_ts": lambda lib: lib.DataFrame(
        {"a": [1, 2]}, index=lib.period_range("2026Q1", periods=2, freq="Q")
    ).to_timestamp(),
    "frame-to_ts-bad": lambda lib: lib.DataFrame({"a": [1, 2]}).to_timestamp(),
    "dt-to_ts-col": lambda lib: lib.Series(
        lib.period_range("2026-01", periods=2, freq="M")
    ).dt.to_timestamp(),
    "dt-year-col": lambda lib: lib.Series(lib.period_range("2026-01", periods=2, freq="M")).dt.year,
    "dt-month-col": lambda lib: lib.Series([lib.Period("2026-01", "M"), None]).dt.month,
    "dt-start-col": lambda lib: (
        lib.Series(lib.period_range("2026-01", periods=2, freq="M")).dt.start_time
    ),
    "dt-end-col": lambda lib: (
        lib.Series(lib.period_range("2026-01", periods=2, freq="M")).dt.end_time
    ),
    "dt-freq-col": lambda lib: lib.Series(lib.period_range("2026-01", periods=2, freq="M")).dt.freq,
    "dt-asfreq-col": lambda lib: lib.Series(
        lib.period_range("2026-01", periods=2, freq="M")
    ).dt.asfreq("D"),
    "dt-strftime-col": lambda lib: lib.Series(
        lib.period_range("2026-01", periods=2, freq="M")
    ).dt.strftime("%Y/%m"),
    "dt-qyear-col": lambda lib: (
        lib.Series(lib.period_range("2026-01", periods=2, freq="M")).dt.qyear
    ),
    "dt-dim-col": lambda lib: (
        lib.Series(lib.period_range("2026-01", periods=2, freq="M")).dt.days_in_month
    ),
    "dt-leap-col": lambda lib: (
        lib.Series(lib.period_range("2026-01", periods=2, freq="M")).dt.is_leap_year
    ),
    "pi-ts-2": lambda lib: lib.period_range("2026-01", periods=2, freq="M").to_timestamp().freq,
    "pi-ts-3": lambda lib: lib.period_range("2026-01", periods=3, freq="M").to_timestamp().freq,
    "pi-ts-q": lambda lib: lib.period_range("2026Q1", periods=3, freq="Q").to_timestamp().freq,
    "pi-ts-end": lambda lib: (
        lib.period_range("2026-01", periods=3, freq="M").to_timestamp(how="end").freq
    ),
    "pi-ts-gap": lambda lib: (
        lib.PeriodIndex(["2026-01", "2026-03", "2026-04"], freq="M").to_timestamp().freq
    ),
    "pi-ts-D": lambda lib: lib.period_range("2026-01-01", periods=3, freq="D").to_timestamp().freq,
    "pi-ts-D2": lambda lib: lib.period_range("2026-01-01", periods=2, freq="D").to_timestamp().freq,
    "pi-ts-M-D": lambda lib: (
        lib.period_range("2026-01", periods=3, freq="M").to_timestamp("D").freq
    ),
    "ser-to_period-copy": lambda lib: lib.Series(
        [1.5], index=lib.DatetimeIndex(["2026-01-05"])
    ).to_period("W"),
    "ser-to_period-name": lambda lib: (
        lib.Series([1], index=lib.DatetimeIndex(["2026-01-05"], name="when")).to_period("M").index
    ),
    "frame-to_ts-cols-bad": lambda lib: lib.DataFrame([[1, 2]]).to_timestamp(axis=1),
    "frame-to_period-cols-bad": lambda lib: lib.DataFrame([[1, 2]]).to_period(axis=1),
    "frame-to_period-axis-str": lambda lib: lib.DataFrame(
        {"a": [1]}, index=lib.DatetimeIndex(["2026-01-05"])
    ).to_period("M", axis="index"),
    "dt-to_period-tz": lambda lib: lib.Series(
        lib.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    ).dt.to_period("D"),
    "dt-to_period-td": lambda lib: lib.Series(lib.to_timedelta([1, 2], unit="D")).dt.to_period("D"),
    "dt-period-dayofweek": lambda lib: (
        lib.Series(lib.period_range("2026-01-01", periods=2, freq="D")).dt.dayofweek
    ),
    "dt-period-week": lambda lib: (
        lib.Series(lib.period_range("2026-01-01", periods=2, freq="D")).dt.week
    ),
    "dt-period-quarter": lambda lib: (
        lib.Series(
            lib.period_range("2026-01-01", periods=2, freq="D"), index=[5, 6], name="p"
        ).dt.quarter
    ),
    "dt-period-to_ts-end": lambda lib: lib.Series(
        lib.period_range("2026-01", periods=2, freq="M")
    ).dt.to_timestamp(how="end"),
    "dt-period-tz": lambda lib: lib.Series(lib.period_range("2026-01", periods=2, freq="M")).dt.tz,
    "dt-period-date": lambda lib: (
        lib.Series(lib.period_range("2026-01", periods=2, freq="M")).dt.date
    ),
    "dt-period-type": lambda lib: (
        type(lib.Series(lib.period_range("2026-01", periods=2, freq="M")).dt).__name__
    ),
    "dt-period-leap-gap": lambda lib: (
        lib.Series([lib.Period("2024-01", "M"), None]).dt.is_leap_year
    ),
    "dt-period-start-gap": lambda lib: lib.Series([lib.Period("2024-01", "M"), None]).dt.start_time,
    "dt-period-strf-gap": lambda lib: lib.Series([lib.Period("2024-01", "M"), None]).dt.strftime(
        "%Y"
    ),
    "dt-period-asfreq-s": lambda lib: lib.Series(
        lib.period_range("2026-01", periods=2, freq="M")
    ).dt.asfreq("D", how="s"),
    "frame-to_period-3": lambda lib: lib.DataFrame(
        {"a": [1, 2, 3]}, index=lib.date_range("2026-01-31", periods=3, freq="ME")
    ).to_period(),
    "ser-to_ts-labels": lambda lib: list(
        lib.Series([1, 2, 3], index=lib.period_range("2026-01", periods=3, freq="M"))
        .to_timestamp()
        .index
    ),
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


def test_a_period_column_answers_period_properties() -> None:
    column = fp.Series(fp.period_range("2026-01", periods=2, freq="M"))
    assert type(column.dt).__name__ == "PeriodProperties"
    assert type(fp.Series(instants(fp)).dt).__name__ == "DatetimeProperties"
