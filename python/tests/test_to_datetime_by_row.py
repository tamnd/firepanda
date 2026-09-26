"""`to_datetime` with `format="ISO8601"` and `format="mixed"`, checked against pandas.

Both read every row with its own format. `ISO8601` holds each row to ISO 8601
and `mixed` also reads the shapes dateutil reads. The instants, the unit, the
zone and the errors are compared.
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

ISO = [
    ["2024-01-03", "2024-01-06 19:00:00"],
    ["2024-01-03", "2024-01-06T19:00:00.5", None, "NaT", "nan", ""],
    ["2024-01-03", "2024-01-06 19:00:00.123456789"],
    ["2024-01-03 10:00:00.1234567"],
    ["2024-01", "2024", "2024-1-3", "20240103", " 2024-01-04"],
    ["2024-01-03T10", "2024-01-03 10", "2024-01-03T1000", "20240103T100000"],
    ["2024-01-03T10:00+01:00", "2024-01-03T10:00:00+0100", "2024-01-03T10:00:00+01"],
    ["2024-01-03 10:00:00 +01:00", "2024-01-04T10:00:00.25+01:00"],
    ["2024-01-03T10:00Z", "2024-01-03T11:00+00:00"],
    ["2024-01-03T10:00:00-05:30"],
    ["1500-01-03T10:00:00"],
    [None, "NaT"],
]

LOOSE = [
    ["01/02/2024", "13/02/2024", "1/2/24", "31/12/99", "2024/01/02"],
    ["Jan 2, 2024", "2 January 2024", "January 2024", "Tue Jan 2 2024"],
    ["01/02/2024 3:04 PM", "01/02/2024 15:04:05", "12/31/2024 12:00 AM"],
    ["01-02-2024", "01.02.2024", "02/2024", "2024-01-03 3pm"],
    ["Jan 2 2024 10:00:00.25", "2024-01-03t10:00", "2024-01-03  10:00"],
    ["2024-01-03T10:00:00 UTC", "2024-01-03T11:00:00Z"],
    ["2024-01-03", "01/02/2024", None, " 2024-01-04 "],
]


def facts(answer: Any) -> Any:
    """The instants as text and the type, which carries the unit and the zone."""
    return [None if value is None or str(value) == "NaT" else str(value) for value in answer], str(
        answer.dtype
    )


def by_format(fmt: str, values: list[Any], **options: Any) -> Callable[[Any], Any]:
    return lambda m: m.to_datetime(values, format=fmt, **options)


BUILDS: list[Callable[[Any], Any]] = [
    *(by_format("ISO8601", values) for values in ISO),
    *(by_format("mixed", values) for values in ISO + LOOSE),
    by_format("ISO8601", ["2024-01-03T10:00+01:00", "2024-01-03T10:00+02:00"], utc=True),
    by_format("mixed", ["2024-01-03T10:00+01:00", "2024-01-03"], utc=True),
    by_format("ISO8601", ["2024-01-03", "x", "2024-13-01"], errors="coerce"),
    by_format("mixed", ["2024-01-03", "x", "2024-13-01", "10/40/2024"], errors="coerce"),
    by_format("ISO8601", ["x"], errors="coerce"),
    by_format("ISO8601", []),
    lambda m: m.to_datetime(m.Series(["2024-01-03", "2024-01-06 19:00"], name="s"), format="mixed"),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The instants, the unit and the zone, against pandas."""
    import pandas as pd

    assert facts(build(firepanda)) == facts(build(pd))


def test_a_column_keeps_its_labels_and_name(firepanda: ModuleType) -> None:
    """A column read row by row answers a column on the same labels with the same name."""
    import pandas as pd

    for m in (firepanda, pd):
        answer = m.to_datetime(
            m.Series(["2024-01", "2024-01-06 19:00"], index=[5, 7], name="s"), format="ISO8601"
        )
        assert answer.index.tolist() == [5, 7] and answer.name == "s"


def test_a_time_alone_is_today(firepanda: ModuleType) -> None:
    """dateutil reads a time with no date as that time today, and so does `mixed`."""
    answer = firepanda.to_datetime(["3:04 PM"], format="mixed")
    today = datetime.date.today()
    assert str(answer[0]) == f"{today.isoformat()} 15:04:00"


MISTAKES: list[Callable[[Any], Any]] = [
    by_format("ISO8601", ["2024-01-03", "x"]),
    by_format("ISO8601", ["2024-01-03", "2024-13-01"]),
    by_format("ISO8601", ["2024-01-03", "2024-02-30"]),
    by_format("ISO8601", ["2024-01-03 25:00"]),
    by_format("ISO8601", ["01/02/2024"]),
    by_format("ISO8601", ["2024-01-03T10:00:00 UTC"]),
    by_format("ISO8601", ["2024-001"]),
    by_format("ISO8601", ["202401"]),
    by_format("ISO8601", ["2024-01-04 "]),
    by_format("ISO8601", ["2024-01-03T10:00+01:00", "2024-01-03T10:00+02:00"]),
    by_format("ISO8601", ["2024-01-03T10:00Z", "2024-01-03"]),
    by_format("ISO8601", ["2024-01-03", "2024-01-03 10:00:00 +01:00"], errors="coerce"),
    by_format("mixed", ["2024-01-03", "x"]),
    by_format("mixed", ["2024-01-03", "2024-13-01"]),
    by_format("mixed", ["2024-01-03", "2024-02-30"]),
    by_format("mixed", ["2024-01-03 25:00"]),
    by_format("mixed", ["2024-01-03 10:00:60"]),
    by_format("mixed", ["2024-001"]),
    by_format("mixed", ["2024-W01-1"]),
    by_format("mixed", ["202401"]),
    by_format("mixed", ["2024-01-03 10:00:00 EST"]),
    by_format("mixed", ["2024-01-03T10:00+01:00", "2024-01-03"]),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_mistakes_raise_as_pandas_raises(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The same error type as pandas, compared by name, and the same words."""
    import pandas as pd

    with pytest.raises(Exception) as expected:
        build(pd)
    with pytest.raises(Exception, match="^" + re.escape(str(expected.value)) + "$") as found:
        build(firepanda)
    named = type(found.value).__name__.replace("InvalidArgumentError", "ValueError")
    assert named == type(expected.value).__name__
