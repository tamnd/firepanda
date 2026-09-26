"""`tz_convert(None)` and `tz_convert` with a `tzinfo`, checked against pandas.

`tz_convert(None)` moves the instants to UTC and takes the clock off, and a
`zoneinfo.ZoneInfo` or `datetime.timezone.utc` is read as the zone it names.
Both work on a column through `.dt` and on a `DatetimeIndex`.
"""

from __future__ import annotations

import datetime
import importlib.util
import re
import zoneinfo
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

TEXTS = ["2024-01-01 10:00", None, "2024-07-01 23:30:00.5"]


def column(m: Any) -> Any:
    return m.Series(m.to_datetime(TEXTS, format="mixed"), name="t").dt.tz_localize("Europe/Paris")


def index(m: Any) -> Any:
    return m.DatetimeIndex(["2024-01-01 10:00", "2024-07-01 23:30"], name="d").tz_localize(
        "Asia/Tokyo"
    )


def facts(answer: Any) -> Any:
    """The instants as text, the type and the name."""
    values = [None if value is None or str(value) == "NaT" else str(value) for value in answer]
    return values, str(answer.dtype), answer.name


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: column(m).dt.tz_convert(None),
    lambda m: index(m).tz_convert(None),
    lambda m: column(m).dt.tz_convert(zoneinfo.ZoneInfo("America/New_York")),
    lambda m: index(m).tz_convert(zoneinfo.ZoneInfo("Europe/London")),
    lambda m: column(m).dt.tz_convert(datetime.UTC),
    lambda m: index(m).tz_convert(datetime.UTC),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The instants, the unit, the zone and the name, against pandas."""
    import pandas as pd

    assert facts(build(firepanda)) == facts(build(pd))


NAIVE: list[Callable[[Any], Any]] = [
    lambda m: m.Series(m.to_datetime(["2024-01-01"])).dt.tz_convert(None),
    lambda m: m.to_datetime(["2024-01-01"]).tz_convert(None),
    lambda m: m.Series(m.to_datetime(["2024-01-01"])).dt.tz_convert("UTC"),
    lambda m: m.to_datetime(["2024-01-01"]).tz_convert(zoneinfo.ZoneInfo("Asia/Tokyo")),
]


@pytest.mark.parametrize("build", NAIVE)
def test_instants_with_no_zone_raise_as_pandas_raises(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """A `TypeError` with pandas' words, since there is no zone to convert from."""
    import pandas as pd

    with pytest.raises(TypeError) as expected:
        build(pd)
    with pytest.raises(TypeError, match="^" + re.escape(str(expected.value)) + "$"):
        build(firepanda)
