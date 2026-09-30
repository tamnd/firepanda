"""The row labels keep their frequency through the steps pandas keeps it through."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def days(lib: ModuleType) -> Any:
    return lib.Series([1.0, 2, 3, 4], index=lib.date_range("2026-01-01", periods=4), name="v")


def frame(lib: ModuleType) -> Any:
    return lib.DataFrame({"a": [1, 2, 3, 4]}, index=lib.date_range("2026-01-01", periods=4))


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "series": lambda lib: days(lib),
    "frame": lambda lib: frame(lib),
    "head": lambda lib: days(lib).head(2),
    "iloc-step": lambda lib: days(lib).iloc[::2],
    "iloc-pick": lambda lib: days(lib).iloc[[0, 3]],
    "loc-slice": lambda lib: days(lib).loc["2026-01-02":],
    "arith": lambda lib: days(lib) * 2,
    "frame-assign": lambda lib: frame(lib).assign(b=1),
    "concat": lambda lib: lib.concat([days(lib).head(2), days(lib).tail(2)]),
    "resample": lambda lib: days(lib).resample("2D").sum(),
    "sort-index-down": lambda lib: days(lib).sort_index(ascending=False),
    "sort-values-down": lambda lib: days(lib).sort_values(ascending=False),
    "reset": lambda lib: days(lib).reset_index(drop=True),
    "shift-freq": lambda lib: days(lib).shift(1, freq="D"),
    "shift-hours": lambda lib: days(lib).shift(-2, freq="h"),
    "shift-infer": lambda lib: days(lib).shift(1, freq="infer"),
    "shift-frame": lambda lib: frame(lib).shift(2, freq="2D"),
    "shift-spans": lambda lib: lib.Series(
        [1, 2], index=lib.timedelta_range("1h", periods=2, freq="h")
    ).shift(1, freq="h"),
    "shift-periods": lambda lib: lib.Series(
        [1, 2], index=lib.period_range("2026-01", periods=2, freq="M")
    ).shift(1, freq="M"),
}


def outcome(build: Callable[[], Any]) -> str:
    try:
        return repr(build())
    except Exception as error:
        return f"{type(error).__name__}: {error}"


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_prints_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


def test_inferring_needs_a_frequency_to_read() -> None:
    labels = fp.DatetimeIndex(["2026-01-01", "2026-01-03", "2026-01-04"])
    with pytest.raises(ValueError, match="cannot be inferred"):
        fp.Series([1, 2, 3], index=labels).shift(1, freq="infer")


def test_shifting_plain_labels_by_a_frequency_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="only implemented for DatetimeIndex"):
        fp.Series([1, 2]).shift(1, freq="D")
