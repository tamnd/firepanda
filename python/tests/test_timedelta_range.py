"""`TimedeltaIndex`, `timedelta_range` and the span fields, checked against pandas.

The labels, their unit and the index name are compared, and so are the fields
a span is read by, on the index and on the `dt` accessor of a column of spans.
A frequency is not held, so `freq` is not compared.
"""

from __future__ import annotations

import datetime
import importlib.util
import re
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def facts(found: Any) -> tuple[Any, ...]:
    """What is compared: the type, the name and the values with gaps as None."""

    def plain(value: Any) -> Any:
        if value is None or value != value:
            return None
        return str(value) if hasattr(value, "value") else value

    if hasattr(found, "columns"):
        return (
            list(found.columns),
            [str(found[c].dtype) for c in found.columns],
            [[plain(v) for v in found[c].tolist()] for c in found.columns],
        )
    return str(found.dtype), found.name, [plain(v) for v in found.tolist()]


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.timedelta_range("1s", periods=3),
    lambda m: m.timedelta_range(1, periods=2, freq="ns"),
    lambda m: m.timedelta_range(m.Timedelta(1, "s"), periods=2),
    lambda m: m.timedelta_range(datetime.timedelta(seconds=1), periods=2, name="n"),
    lambda m: m.timedelta_range("1s", "3s", freq="s"),
    lambda m: m.timedelta_range("1s", "3s", freq="s", closed="left"),
    lambda m: m.timedelta_range("1s", "3s", freq="s", closed="right"),
    lambda m: m.timedelta_range("1s", "2s", periods=5),
    lambda m: m.timedelta_range(end="3s", periods=3, freq="s"),
    lambda m: m.timedelta_range("3s", "1s", freq="-1s"),
    lambda m: m.timedelta_range("3s", "1s", freq="s"),
    lambda m: m.timedelta_range("1s", periods=0),
    lambda m: m.timedelta_range("1s", periods=3, freq="h"),
    lambda m: m.timedelta_range("1s", periods=3, freq="2D3h"),
    lambda m: m.timedelta_range("-1D", "0D", freq="12h"),
    lambda m: m.timedelta_range("1s", periods=2, unit="s"),
    lambda m: m.timedelta_range("1s", periods=2, freq="ms"),
    lambda m: m.TimedeltaIndex(["1s", "2D"], name="n"),
    lambda m: m.TimedeltaIndex(["1s", None]),
    lambda m: m.TimedeltaIndex([1, 2]),
    lambda m: m.TimedeltaIndex(m.timedelta_range("1s", periods=2), name="n"),
    lambda m: m.TimedeltaIndex(["1s", "2D"]).as_unit("s"),
    lambda m: m.TimedeltaIndex(["1s", "-2D3h", None], name="n").days,
    lambda m: m.TimedeltaIndex(["1s", "-2D3h"], name="n").seconds,
    lambda m: m.TimedeltaIndex(["1.5ms", "-1us"]).microseconds,
    lambda m: m.TimedeltaIndex(["1ns", "-7ns"]).nanoseconds,
    lambda m: m.TimedeltaIndex(["1s", "2D"], name="n").total_seconds(),
    lambda m: m.TimedeltaIndex(["1s", "-2D3h1ns"]).components,
    lambda m: m.TimedeltaIndex(["1s", None]).components,
    lambda m: m.to_timedelta(m.Series(["1s", "-2D3h"], name="n")).dt.seconds,
    lambda m: m.to_timedelta(m.Series(["1.5ms", None])).dt.microseconds,
    lambda m: m.to_timedelta(m.Series(["1ns", "3us7ns"])).dt.nanoseconds,
    lambda m: m.to_timedelta(m.Series(["1s", "-2D3h1ns"])).dt.components,
    lambda m: m.to_timedelta(m.Series(["1s", None])).dt.components,
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every way of giving the ends and the step, and every field of a span."""
    import pandas as pd

    assert facts(build(firepanda)) == facts(build(pd))


def test_the_unit_and_numbers_are_pandas(firepanda: ModuleType) -> None:
    """`unit` and `asi8` read the storage the way pandas does."""
    import pandas as pd

    for m in (firepanda, pd):
        found = m.TimedeltaIndex(["1s", "2D"]).as_unit("ms")
        assert (found.unit, list(found.asi8)) == ("ms", [1000, 172800000])


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.timedelta_range("1s"),
    lambda m: m.timedelta_range("1s", "2s", periods=2, freq="s"),
    lambda m: m.timedelta_range("1s", periods=1.5),
    lambda m: m.timedelta_range("1s", periods=2, closed="both"),
    lambda m: m.timedelta_range("1s", periods=2, unit="h"),
    lambda m: m.timedelta_range("1s", periods=2, freq="ms", unit="s"),
    lambda m: m.timedelta_range(True, periods=2),
    lambda m: m.TimedeltaIndex(),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_mistakes_raise_as_pandas_raises(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The same error type and words as pandas."""
    import pandas as pd

    with pytest.raises(Exception) as expected:
        build(pd)
    with pytest.raises(type(expected.value), match="^" + re.escape(str(expected.value)) + "$"):
        build(firepanda)


def test_a_held_frequency_is_refused(firepanda: ModuleType) -> None:
    """Holding a frequency means inferring one, which is not supported yet."""
    with pytest.raises(NotImplementedError, match="freq= is not supported"):
        firepanda.TimedeltaIndex(["1s"], freq="s")
