"""Operators on an index, and the class of the index an operation answers, compared with pandas.

Each case runs in both libraries. An index is compared by its class, its type,
its name and its labels, a series by its labels and name, and a comparison by
its bools, which pandas gives as a numpy array and firepanda as a list. A
mistake is compared by its class and its message.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def instants(lib: ModuleType) -> Any:
    return lib.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-04"], name="n")


def spans(lib: ModuleType) -> Any:
    return lib.to_timedelta(["1h", "2h", "4h"])


def numbers(lib: ModuleType) -> Any:
    return lib.Index([1, 2, 3], name="a")


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        got = call(lib)
    except Exception as error:
        return type(error).__name__, str(error)
    if isinstance(got, (fp.Index, pd.Index)):
        return type(got).__name__, str(got.dtype), got.name, str(got.tolist())
    if isinstance(got, (fp.Series, pd.Series)):
        return "Series", str(got.dtype), got.name, str(got.tolist()), got.index.tolist()
    return [bool(one) for one in got]


CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: numbers(lib) + 1,
    lambda lib: 1 - numbers(lib),
    lambda lib: numbers(lib) * lib.Index([1.5, 2.0, 0.5], name="b"),
    lambda lib: numbers(lib) / 2,
    lambda lib: numbers(lib) // 2,
    lambda lib: numbers(lib) % 2,
    lambda lib: numbers(lib) ** 2,
    lambda lib: 2 ** numbers(lib),
    lambda lib: -numbers(lib),
    lambda lib: abs(lib.Index([-1.5, 2.0])),
    lambda lib: numbers(lib) + numbers(lib),
    lambda lib: numbers(lib) + [1, 2, 3],  # noqa: RUF005
    lambda lib: numbers(lib) + [1],  # noqa: RUF005
    lambda lib: numbers(lib) + [1, 2],  # noqa: RUF005
    lambda lib: numbers(lib) + lib.Series([1, 1, 1]),
    lambda lib: numbers(lib) < lib.Index([1.5, 2.0, 0.5]),
    lambda lib: numbers(lib) >= 2,
    lambda lib: numbers(lib) == [1, 0, 3],
    lambda lib: numbers(lib) < [1, 2],
    lambda lib: lib.Index(["a", "b"]) < "b",
    lambda lib: instants(lib) > instants(lib)[1],
    lambda lib: instants(lib) == instants(lib)[1],
    lambda lib: instants(lib) != instants(lib)[1],
    lambda lib: instants(lib) > "2024-01-02",
    lambda lib: instants(lib) >= instants(lib),
    lambda lib: instants(lib) + lib.Timedelta("1h"),
    lambda lib: lib.Timedelta("1h") + instants(lib),
    lambda lib: instants(lib) - lib.Timedelta("1D"),
    lambda lib: instants(lib) - instants(lib)[0],
    lambda lib: instants(lib) - instants(lib),
    lambda lib: instants(lib) + dt.timedelta(hours=1),
    lambda lib: instants(lib) + lib.offsets.Day(2),
    lambda lib: instants(lib).tz_localize("UTC") < instants(lib).tz_localize("UTC")[1],
    lambda lib: spans(lib) > spans(lib)[1],
    lambda lib: spans(lib) == lib.Timedelta("2h"),
    lambda lib: spans(lib) + spans(lib),
    lambda lib: -spans(lib),
    lambda lib: abs(-spans(lib)),
]


@pytest.mark.parametrize("call", CASES)
def test_the_operator_is_pandas_operator(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)


CLASSES: list[Callable[[ModuleType], Any]] = [
    lambda lib: instants(lib).unique(),
    lambda lib: instants(lib).delete(0),
    lambda lib: instants(lib).take([2, 0]),
    lambda lib: instants(lib)[:2].union(instants(lib)[1:]),
    lambda lib: instants(lib).intersection(instants(lib)[1:]),
    lambda lib: instants(lib).difference(instants(lib)[2:]),
    lambda lib: instants(lib).sort_values(ascending=False),
    lambda lib: instants(lib).append(instants(lib)),
    lambda lib: instants(lib).diff(),
    lambda lib: spans(lib).unique(),
    lambda lib: spans(lib).delete(-1),
    lambda lib: lib.Index(lib.Series(lib.to_datetime(["2024-01-01"]))),
    lambda lib: lib.Index(lib.Series(lib.to_timedelta(["1h"]), name="s")),
    lambda lib: lib.to_timedelta(["1h", None]),
    lambda lib: lib.to_timedelta(("1D",)),
    lambda lib: lib.to_timedelta(lib.Index(["1h"], name="x")),
]


@pytest.mark.parametrize("call", CLASSES)
def test_the_class_is_pandas_class(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)


LABELS: list[Callable[[ModuleType], Any]] = [
    lambda lib: instants(lib).drop(instants(lib)[0]),
    lambda lib: instants(lib).drop("2024-01-02"),
    lambda lib: instants(lib).drop(
        [lib.Timestamp("2025-01-01"), instants(lib)[1]], errors="ignore"
    ),
    lambda lib: instants(lib).drop(lib.Timestamp("2025-01-01")),
    lambda lib: spans(lib).drop(spans(lib)[1]),
    lambda lib: spans(lib).drop(["9h"]),
    lambda lib: instants(lib).insert(1, lib.Timestamp("2024-01-01 12:00")),
    lambda lib: instants(lib).insert(0, "2023-12-31"),
    lambda lib: instants(lib).insert(-1, lib.NaT),
    lambda lib: instants(lib).insert(3, dt.datetime(2024, 2, 1)),
    lambda lib: instants(lib).tz_localize("UTC").insert(3, lib.Timestamp("2024-02-01", tz="UTC")),
    lambda lib: spans(lib).insert(3, lib.Timedelta("5h")),
    lambda lib: spans(lib).insert(-1, "90min"),
    lambda lib: instants(lib).putmask([True, False, False], instants(lib)[2]),
    lambda lib: spans(lib).putmask([False, True, False], lib.Timedelta("9h")),
    lambda lib: instants(lib).isin([instants(lib)[0]]),
    lambda lib: instants(lib).isin(["2024-01-02"]),
    lambda lib: instants(lib).insert(1, lib.NaT).isin([lib.NaT, None]),
    lambda lib: instants(lib).insert(1, lib.NaT).isin([None]),
    lambda lib: instants(lib).isin([dt.datetime(2024, 1, 1)]),
    lambda lib: spans(lib).isin(["1h", lib.Timedelta("4h")]),
    lambda lib: instants(lib).tz_localize("UTC").isin(["2024-01-01 00:00:00+00:00"]),
]


@pytest.mark.parametrize("call", LABELS)
def test_labels_of_instants_and_spans_are_pandas_labels(
    call: Callable[[ModuleType], Any],
) -> None:
    assert outcome(call, fp) == outcome(call, pd)


FILLS: list[Callable[[ModuleType], Any]] = [
    lambda lib: lib.Series(spans(lib)).where([True, False, True], lib.Timedelta("9h")),
    lambda lib: lib.Series(spans(lib)).where([True, False, True], dt.timedelta(minutes=5)),
    lambda lib: lib.Series(spans(lib)).where([True, False, True], "9h"),
    lambda lib: lib.Series(spans(lib)).mask([True, False, False], lib.Timedelta("1s")),
    lambda lib: lib.Series(lib.to_timedelta(["1h", None])).fillna(lib.Timedelta("1D")),
    lambda lib: lib.Series(lib.to_timedelta(["1h", None])).fillna("2h"),
]


@pytest.mark.parametrize("call", FILLS)
def test_a_column_of_spans_takes_a_span(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)


def test_a_label_pandas_keeps_as_an_object_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="index of objects"):
        instants(fp).insert(0, 5)
    assert pd.api.types.is_object_dtype(instants(pd).insert(0, 5))
