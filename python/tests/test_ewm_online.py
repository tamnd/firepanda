"""`ewm(...).online()`, the decay that can be fed more rows later.

pandas computes the online mean only with numba, which the test environment
does not have, so the numbers are checked against what pandas' online kernel
must agree with: a plain call matches `ewm(...).mean()` over the same rows, and
an update matches `ewm(...).mean()` over the rows so far with the new ones
appended. The repr, the refusals and the engine check need no numba in pandas
and are compared with it live.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

DECAYS: list[dict[str, Any]] = [
    {"com": 0.5},
    {"span": 3, "adjust": False},
    {"alpha": 0.3, "ignore_na": True},
    {"halflife": 2, "min_periods": 3},
    {"alpha": 0.6, "adjust": False, "ignore_na": True},
]

A = [1.0, 2.0, None, 4.0, 3.0]
B = [5.0, 6.0, 7.0, 8.0, 9.0]


def outcome(call: Any) -> str:
    try:
        return repr(call())
    except Exception as error:
        kind = next(k for k in type(error).__mro__ if k.__module__ == "builtins")
        return f"{kind.__name__}: {error}"


@pytest.mark.parametrize("decay", DECAYS)
def test_a_plain_call_is_the_ordinary_mean(firepanda: ModuleType, decay: dict[str, Any]) -> None:
    frame = firepanda.DataFrame({"a": A, "b": B})
    assert frame.ewm(**decay).online().mean().equals(frame.ewm(**decay).mean())
    column = frame["a"]
    assert column.ewm(**decay).online().mean().equals(column.ewm(**decay).mean())


@pytest.mark.parametrize("decay", [decay for decay in DECAYS if "min_periods" not in decay])
def test_an_update_carries_on_from_the_last_row(
    firepanda: ModuleType, decay: dict[str, Any]
) -> None:
    frame = firepanda.DataFrame({"a": A, "b": B})
    more = firepanda.DataFrame({"a": [5.0, None, 2.0], "b": [1.0, 2.0, None]}, index=[7, 8, 9])
    online = frame.ewm(**decay).online()
    online.mean()
    whole = firepanda.concat([frame, more]).ewm(**decay).mean()
    assert online.mean(update=more).equals(whole.iloc[5:])

    column = firepanda.Series(A, name="a")
    online = column.ewm(**decay).online()
    online.mean()
    step = firepanda.Series([10.0, None], index=[5, 6], name="z")
    whole = firepanda.concat([column, step]).ewm(**decay).mean()
    got = online.mean(update=step)
    assert got.name == "z"
    assert got.tolist() == whole.iloc[5:].tolist()


def test_an_update_counts_its_rows_afresh(firepanda: ModuleType) -> None:
    # pandas counts the observations for min_periods from the first row of each
    # call, which for an update is the last row of the call before it.
    online = firepanda.Series([1.0, 2.0, 3.0]).ewm(com=1, min_periods=3).online()
    online.mean()
    got = online.mean(update=firepanda.Series([4.0, 5.0]))
    assert [value != value for value in got.tolist()] == [True, False]


def test_reset_forgets_the_rows(firepanda: ModuleType) -> None:
    online = firepanda.Series(A).ewm(com=1).online()
    first = online.mean()
    online.reset()
    assert online.mean().equals(first)
    online.reset()
    with pytest.raises(ValueError, match="Must call mean with update=None first"):
        online.mean(update=firepanda.Series([1.0]))


CALLS: list[Any] = [
    lambda pd: repr(pd.DataFrame({"a": A}).ewm(com=0.5).online()),
    lambda pd: repr(pd.Series(A).ewm(span=3, adjust=False, ignore_na=True).online()),
    lambda pd: repr(pd.Series(A).ewm(halflife=2, min_periods=2)),
    lambda pd: pd.Series(A).ewm(com=1).online(engine="cython"),
    lambda pd: pd.Series(A).ewm(com=1).online().var(),
    lambda pd: pd.Series(A).ewm(com=1).online().std(),
    lambda pd: pd.Series(A).ewm(com=1).online().cov(),
    lambda pd: pd.Series(A).ewm(com=1).online().corr(),
    lambda pd: pd.Series(A).ewm(com=1).online().aggregate("mean"),
    lambda pd: pd.Series(A).ewm(com=1).online().mean(update_times=pd.Series([1])),
    lambda pd: pd.Series(A).ewm(com=1).online().engine,
    lambda pd: pd.Series(A).ewm(com=1).online(engine_kwargs={"nogil": True}).engine_kwargs,
]


@needs_pandas
@pytest.mark.parametrize("call", CALLS)
def test_what_needs_no_numba_matches_pandas(firepanda: ModuleType, call: Any) -> None:
    import pandas

    assert outcome(lambda: call(firepanda)) == outcome(lambda: call(pandas))
