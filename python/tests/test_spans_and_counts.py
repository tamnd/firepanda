"""Spans and instants read as counts, and operands given as lists, compared with pandas.

Each case runs in both libraries. A series is compared by its type, its name,
its labels and its index, an index by its class, type, labels and frequency, a
scalar by its text, and a mistake by its class name, where firepanda's own
classes stand in for pandas' by being subclasses of them.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def spans(lib: ModuleType) -> Any:
    return lib.Series(lib.to_timedelta(["1h", "90min", None]), name="s")


def hours(lib: ModuleType) -> Any:
    return lib.timedelta_range("1h", periods=3, freq="h")


def days(lib: ModuleType) -> Any:
    return lib.date_range("2024-01-01", periods=5, freq="D")


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        got = call(lib)
    except Exception as error:
        return "error", error
    if isinstance(got, (fp.Index, pd.Index)):
        return (
            type(got).__name__,
            str(got.dtype),
            str(got.tolist()),
            str(getattr(got, "freq", None)),
        )
    if isinstance(got, (fp.DataFrame, pd.DataFrame)):
        return str(got.to_dict()), [str(one) for one in got.dtypes]
    if isinstance(got, (fp.Series, pd.Series)):
        return str(got.dtype), got.name, str(got.tolist()), got.index.tolist()
    return str(got)


def same(call: Callable[[ModuleType], Any]) -> None:
    ours, theirs = outcome(call, fp), outcome(call, pd)
    if theirs[0] == "error":
        assert ours[0] == "error"
        assert isinstance(ours[1], type(theirs[1])) or (
            type(ours[1]).__name__ == type(theirs[1]).__name__
        )
        assert str(ours[1]) == str(theirs[1])
    else:
        assert ours == theirs


CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: lib.Series([1, 2, 3], index=[5, 6, 7], name="a") + [1, 2, 3],  # noqa: RUF005
    lambda lib: [1, 2, 3] - lib.Series([1, 2, 3], name="a"),
    lambda lib: lib.Series([1, 2, 3]) * (1, 2, 3),
    lambda lib: lib.Series([1, 2, 3]) * np.array([1, 2, 3]),
    lambda lib: lib.Series([1, 2, 3]) + [1, 2],  # noqa: RUF005
    lambda lib: lib.Series([1, 2, 3]) == [1, 0, 3],
    lambda lib: lib.Series([1, 2, 3]) == [1, 0],
    lambda lib: lib.Series([1, 2, 3]).add([1, 2, 3]),
    lambda lib: spans(lib) / [1, 2, 3],
    lambda lib: lib.Series([3600000000, None]).astype("timedelta64[us]"),
    lambda lib: lib.Series([3600, 5], name="z").astype("timedelta64[s]"),
    lambda lib: lib.Series([3.6e9, float("nan")]).astype("timedelta64[us]"),
    lambda lib: lib.Series([5], dtype="uint8").astype("timedelta64[ms]"),
    lambda lib: lib.Series([0, 86400]).astype("datetime64[s]"),
    lambda lib: lib.Series([86400, 5]).astype("datetime64[ms]"),
    lambda lib: lib.Series([86400.5, float("nan")]).astype("datetime64[s]"),
    lambda lib: lib.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}).astype({"a": "timedelta64[s]"}),
    lambda lib: hours(lib).shift(2),
    lambda lib: lib.timedelta_range("1h", periods=3).shift(1, freq="h"),
    lambda lib: lib.to_timedelta(["1h"]).shift(1),
    lambda lib: days(lib).symmetric_difference(days(lib)[:2]),
    lambda lib: days(lib).symmetric_difference(days(lib)[1:3]),
    lambda lib: days(lib).symmetric_difference(lib.DatetimeIndex(["2024-01-03"]), sort=False),
    lambda lib: lib.DatetimeIndex(["2024-01-01", "2024-01-02"]).symmetric_difference(
        lib.DatetimeIndex(["2024-01-03"])
    ),
    lambda lib: lib.to_timedelta(["1h29min", "2h31min", "90min"]).round("h"),
    lambda lib: lib.to_timedelta(["1h29min", "-2h31min"]).floor("h"),
    lambda lib: lib.to_timedelta(["1h29min", "-2h31min"]).ceil("h"),
    lambda lib: spans(lib).dt.floor("h"),
    lambda lib: spans(lib).dt.ceil("h"),
    lambda lib: lib.Series(lib.to_timedelta(["30min", "90min", "-30min", "150min"])).dt.round("h"),
    lambda lib: lib.Series(lib.to_timedelta(["1500ms", "-1500ms", None])).dt.ceil("s"),
    lambda lib: lib.Series(lib.to_timedelta(["1h7min"])).dt.floor(lib.offsets.Minute(15)),
    lambda lib: lib.Series(lib.to_timedelta(["1h"])).dt.floor("ns"),
    lambda lib: spans(lib).quantile(0.5),
    lambda lib: spans(lib).quantile([0.25, 0.5]),
    lambda lib: lib.Series(lib.to_datetime(["2024-01-01", "2024-01-02"])).quantile([0.25, 0.5]),
    lambda lib: lib.Series(lib.to_datetime(["2024-01-01", "2024-01-03"])).std(),
    lambda lib: lib.Series(lib.to_timedelta(["1h", "3h"])).std(),
    lambda lib: spans(lib).clip(upper=lib.Timedelta("80min")),
    lambda lib: spans(lib).clip(lower=dt.timedelta(minutes=80)),
    lambda lib: lib.Series(lib.to_datetime(["2024-01-01", "2024-01-05"])).clip(
        lower=lib.Timestamp("2024-01-02")
    ),
    lambda lib: lib.Series(
        lib.to_timedelta([0, None, None, 2, None, None, -2], unit="us").as_unit("us"), name="x"
    ).interpolate(),
    lambda lib: lib.Series(
        lib.to_datetime(["2024-01-01", None, "2024-01-02"]).as_unit("s")
    ).interpolate(),
    lambda lib: lib.Series(lib.to_datetime(["2024-01-01", None, None, "2024-01-04"])).interpolate(
        limit=1
    ),
    lambda lib: str(hours(lib).to_pytimedelta()[0]),
    lambda lib: str(spans(lib).dropna().dt.to_pytimedelta()[1]),
]


@pytest.mark.parametrize("call", CASES)
def test_the_answer_is_pandas_answer(call: Callable[[ModuleType], Any]) -> None:
    same(call)


def test_a_fixed_frequency_is_needed_to_round_spans() -> None:
    for lib in (fp, pd):
        with pytest.raises(ValueError, match="non-fixed frequency"):
            lib.Series(lib.to_timedelta(["1h"])).dt.floor("ME")


def test_a_span_over_a_missing_span_is_a_nan_rather_than_a_null() -> None:
    pa = pytest.importorskip("pyarrow")
    spans = fp.Series(fp.to_timedelta(["1h", None]))
    for answer in (spans / fp.Timedelta("1h"), spans // fp.Timedelta("1h")):
        held = pa.RecordBatchReader.from_stream(answer.to_frame("v")).read_all().column(0)
        assert held.null_count == 0
        assert str(held.to_pylist()) == "[1.0, nan]"
