"""`to_datetime` with `format="ISO8601"`, checked against pandas' own ISO 8601 reader.

pandas lets the date be split by a slash, a dot, a space or a backslash as
well as a hyphen, as long as both splits are the same, and it refuses a comma
before the fraction, which dateutil and `format="mixed"` accept.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

TEXTS = [
    "2024/01/02",
    "2024.01.02",
    "2024 01 02",
    "2024\\01\\02",
    "2024/01",
    "2024-01/02",
    "2024/1/2",
    "2024/01/02 10:00",
    "2024/01/02T10:00",
    "2024-01-02 10",
    "2024-01-02T10",
    "2024-01-02 10:00:00.5",
    "2024-01-02T10:00:00,5",
    "2024-01-02T1000",
    "2024-01-02 10:00Z",
    "2024-01-02 10:00 +01:00",
    "2024-01-02T10:00:00+0100",
    "2024-01-02T10:00:00-01",
    "20240102",
    "20240102T1000",
    "2024-W01",
    "2024-002",
    "2024-1-2",
    "2024-01-02T10:00:00.123456789123",
    " 2024-01-02",
    "2024-01-02 ",
    "2024-01-02  10:00",
    "2024",
    "+2024-01-02",
    "2024-01-02t10:00",
    "2024-01-02 10:00:60",
    "2024-01-02T24:00",
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


@pytest.mark.parametrize("text", TEXTS)
def test_one_value_reads_as_pandas_reads_it(firepanda: ModuleType, text: str) -> None:
    """The instant and the type, or the same error with the same words."""
    import pandas as pd

    options = {"format": "ISO8601"}
    assert outcome(firepanda, [text], options) == outcome(pd, [text], options)


COLUMNS: list[tuple[list[Any], dict[str, Any]]] = [
    (["2024-01-02", "2024/01/03", "2024.01.04", None], {}),
    (["2024/01/02 10:00", "2024-01-02T10:00:00,5"], {}),
    (["2024/01/02 10:00", "2024-01-02T10:00:00,5"], {"errors": "coerce"}),
    (["2024-01-02T10:00:00,5", "2024-01-02T10:00:00,5"], {"format": "mixed"}),
]


@pytest.mark.parametrize(("values", "options"), COLUMNS)
def test_a_column_reads_as_pandas_reads_it(
    firepanda: ModuleType, values: list[Any], options: dict[str, Any]
) -> None:
    """Each row is read on its own, and a comma is only refused under `ISO8601`."""
    import pandas as pd

    options = {"format": "ISO8601", **options}
    assert outcome(firepanda, values, options) == outcome(pd, values, options)
