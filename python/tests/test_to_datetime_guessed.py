"""`to_datetime` of text with no format, and with a strptime format, checked against pandas.

With no format pandas guesses one from the first value that is not missing,
holds every row to it, and reads row by row when it cannot guess. The core
reads ISO 8601, so everything else goes through the reading written here,
and the answer, the type and the words of any error have to be pandas'.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

CASES: list[tuple[list[Any], dict[str, Any]]] = [
    (["2024-01-01", "2024-1-1"], {}),
    (["2024-01-01 10:00", "2024-01-01  10:00"], {}),
    (["2024-01-01 10:00", "2024-01-01T10:00"], {}),
    (["01/02/2024", "02/30/2024"], {}),
    (["01/02/2024", "1/2/2024"], {}),
    (["01/02/2024", "13/02/2024"], {}),
    (["2024-01-01T10:00:00", "2024-01-01t10:00:00"], {}),
    (["2024-01-01 10:00:00 UTC", "2024-01-01 10:00:00 GMT"], {}),
    (["2024-01-01 10:00:00 UTC", "2024-01-01 10:00:00 Europe/Paris"], {}),
    (["Jan 2, 2024", "feb 3, 2024"], {}),
    (["Jan 2, 2024", "February 3, 2024"], {}),
    (["2 January 2024", "3 feb 2024"], {}),
    (["2024-01-01", "2024-02-30"], {}),
    (["01/02/2024", "02/29/2023"], {}),
    (["20240101", "2024011"], {}),
    (["2024-01-01", " 2024-01-02"], {}),
    (["2024-01-01T10:00:00Z", "2024-01-01T10:00:00+01:00"], {}),
    (["2024-01-01T10:00:00+01:00", "2024-01-01T10:00:00+0100"], {}),
    (["2024-01-01 10:00:00.5", "2024-01-01 10:00:00.1234567891"], {}),
    (["2024-01-01 10:00:00.5", "2024-01-01 10:00:00"], {}),
    (["2024-01-01 10", "2024-01-01 9"], {}),
    (["2024-01-01", "2024-01-01\t"], {}),
    (["2024", "2025"], {}),
    (["2024-01", "2024-13"], {}),
    (["01/02/2024 10:00", "01/02/2024 10:00:05"], {}),
    (["2024-01-01", None, "nan", "NaT", ""], {}),
    (["2024/01/02", "2024/1/2"], {}),
    (["1/2/2024", "1/2/2024 10:00"], {}),
    (["2024-01-01T10:00:00Z", "2024-01-01T10:00:00"], {}),
    (["2024-01-01", "2024-01-01 00:00"], {}),
    (["x"], dict(format="%Y-%m-%d")),
    (["2024-01-01", "x", "2024-02-30"], dict(errors="coerce")),
    (["x", "2024-01-01"], dict(errors="coerce")),
    (["2 January 2024", "3 March 2024"], dict(format="%d %B %Y")),
    (["01/02/2024 03:04 PM"], dict(format="%m/%d/%Y %I:%M %p")),
    (["2024-01-01 10:00:60"], {}),
    (["2024-01-01 10:00:00 GMT"], {}),
    (["2024-01-01 10:00:00 Europe/Paris"], dict(format="%Y-%m-%d %H:%M:%S %Z")),
    (["2024-01-01T10:00:00Z", "2024-01-01T10:00:00+00:00"], {}),
    (["2024-01-01T10:00:00+01:00", "2024-01-01T10:00:00+02:00"], dict(utc=True)),
    (["2024-01-01T10:00:00+01:00", "x"], dict(errors="coerce")),
    (["2024-01-01T10:00:00+01:00", "2024-01-01T10:00:00+02:00"], dict(errors="coerce")),
    (["24-01-02"], dict(format="%y-%m-%d")),
    (["69-01-02"], dict(format="%y-%m-%d")),
    (["01/02/2024"], {}),
    (["13/02/2024", "01/02/2024"], {}),
    (["Jan 2, 2024", "Mar 3, 2024"], {}),
    (
        ["2024-01-01 10:00:00 Europe/Paris", "2024-07-01 10:00:00 Europe/Paris"],
        dict(format="%Y-%m-%d %H:%M:%S %Z", utc=True),
    ),
    ([" 2024-01-01", "2024-01-02"], {}),
    (["2024-01-01"], dict(format="%Y-%m-%d %H")),
    (["2024-01-01 10:00:00.1234567"], dict(format="%Y-%m-%d %H:%M:%S.%f")),
    (["2024-01-01 10:00:60"], dict(format="%Y-%m-%d %H:%M:%S")),
    (["2024-01-01 10:00:00 GMT", "2024-01-01 10:00:00 GMT"], dict(format="%Y-%m-%d %H:%M:%S %Z")),
    (["NaT", "01/02/2024"], {}),
    (["", "01/02/2024 10:00"], {}),
    (["2024-01-01T10:00:00+05:45"], {}),
    (["2024-01-01T10:00:00-0000"], {}),
    (["2024-001"], dict(format="%Y-%j")),
    (["Monday 2024-01-01"], dict(format="%A %Y-%m-%d")),
    (["2024-01-01 12:30 am"], dict(format="%Y-%m-%d %I:%M %p")),
    (["2024-01-01", "2024-01-01 junk"], dict(errors="coerce")),
    ([None, "01/02/2024"], {}),
    (["2024-01-01", "x"], {}),
    (["2024-01-01", "2024-13-01"], {}),
    (["2024-01-01", "01/02/2024"], {}),
    (["x", "2024-01-01"], {}),
    (["2024-01-01T10:00:00", "2024-01-01T25:00:00"], {}),
    (["2024-01-01", "20240101"], {}),
    (["Sat, 06 Jan 2024 10:00:00 +0000", "Sun, 07 Jan 2024 11:00:00 +0000"], {}),
    (["2024 01 02", "2024 01 03"], {}),
    (["June 15, 2024 2:30 PM", "June 16, 2024 3:30 PM"], {}),
    (["12/31/2024 11:59:59 PM"], {}),
    (["31.12.2024", "30.12.2024"], {}),
    (["6/15/2024 2:30"], {}),
    (["15 June 2024 14:30"], {}),
    (["01/02/24", "01/03/24"], {}),
    (["January 2024"], {}),
    (["2024-01-03 3pm"], {}),
    (["10:00"], {}),
    (["x"], dict(errors="coerce")),
    (["01/02/2024", "2024-01-01"], dict(errors="coerce")),
    (["2024-01-01 10:00:00 EST"], {}),
]


def outcome(m: Any, values: list[Any], options: dict[str, Any]) -> Any:
    """The instants as text and the type, or the kind of error and its first line."""
    try:
        read = m.to_datetime(values, **options)
    except Exception as error:
        kind = type(error).__name__.replace("InvalidArgumentError", "ValueError")
        return kind, str(error).split("\n")[0]
    texts = [None if value is None or str(value) == "NaT" else str(value) for value in read]
    return texts, str(read.dtype).replace("UTC+", "+").replace("UTC-", "-")


@pytest.mark.parametrize(("values", "options"), CASES)
def test_the_reading_is_pandas_reading(
    firepanda: ModuleType, values: list[Any], options: dict[str, Any]
) -> None:
    """The same instants and type, or the same error with the same words."""
    import pandas as pd

    assert outcome(firepanda, values, options) == outcome(pd, values, options)


GUESSES = [
    "2024-01-01",
    "2024-01-01 10:00",
    "2024-01-01T10:00:00",
    "2024-01-01 10:00:00.5",
    "2024-01-01T10:00:00.123456789",
    "2024-01-01T10:00:00Z",
    "2024-01-01T10:00:00+01:00",
    "2024-01-01 10:00:00+0100",
    "20240101",
    "2024-1-1",
    "2024-01",
    "2024",
    "01/02/2024",
    "13/02/2024",
    "1/2/2024",
    "01/02/24",
    "2024/01/02",
    "01-02-2024",
    "01.02.2024",
    "01/02/2024 10:00",
    "01/02/2024 3:04 PM",
    "Jan 2, 2024",
    "2 January 2024",
    "January 2024",
    "Tue Jan 2 2024",
    "2024-01-01 10",
    "2024-01-01T1000",
    "10:00",
    "2024-01-01 10:00:00 UTC",
    "x",
    "2024-01-01  10:00",
    "02/2024",
    "2024-01-03 3pm",
    "2024-01-03t10:00",
    " 2024-01-01",
    "2024-01-01 ",
    "20240101T100000",
    "2024-01-01 10:00:00.5+01:00",
    "2024-01-01 10:00:00 GMT",
    "2024-01-01T10:00:00-0000",
    "2024-01-01T10:00:00+05:45",
    "NaT",
    "nan",
    "Monday, January 1, 2024",
    "1 Jan 2024 10:00",
    "2024-01-01 10:00:60",
    "12/31/2024 11:59:59 PM",
    "2024-01-01 10:00:00.000",
    "Jan 2024",
    "2024 01 02",
    "2024-01-01T10:00:00.5Z",
    "31.12.2024",
    "2024.12.31",
    "01/02/2024 10:00:00.123",
    "Sat, 06 Jan 2024 10:00:00 +0000",
    "2024-06-15 14:30:00-05:00",
    "6/15/2024 2:30",
    "15 June 2024 14:30",
    "June 15, 2024 2:30 PM",
    "2024-W01",
    "2024-001",
]


@pytest.mark.parametrize("text", GUESSES)
def test_the_guess_is_pandas_guess(text: str) -> None:
    """The format guessed from one value, or no guess, as pandas' guesser answers."""
    from firepanda._row_formats import guess
    from pandas.tseries.api import guess_datetime_format

    assert guess(text) == guess_datetime_format(text)
