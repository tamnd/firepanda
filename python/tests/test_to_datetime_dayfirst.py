"""`dayfirst` and `yearfirst` in `to_datetime` and `DatetimeIndex`, checked against pandas.

pandas hands both to dateutil. `dayfirst` reaches the guesser too, so the
format guessed from the first value is day first, and even `2024-01-02` is
read as the first of February. `yearfirst` only matters when each row is read
on its own. An explicit format and `ISO8601` ignore both.
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
    "2024-01-02",
    "01/02/2024",
    "13/02/2024",
    "02/13/2024",
    "01/02/24",
    "10/11/12",
    "2024/01/02",
    "01.02.2024",
    "01-02-2024",
    "1 2 2024",
    "2024 01 02",
    "Jan 02 2024",
    "02 Jan 24",
    "10-Jan-12",
    "01/02",
    "01/2024",
    "20240102",
    "240102",
    "2024-01-02 10:00",
    "01/02/2024 10:00",
    "31/12/2024",
    "12/31/2024",
    "Jan 2",
    "1 2",
    "12 13",
    "13 12",
    "2024 13 01",
    "99 01 01",
    "01 02 03",
    "24 Jan 02",
    "2024-13-01",
    "32/01/02",
]

ORDERS = [
    {"dayfirst": True},
    {"yearfirst": True},
    {"dayfirst": True, "yearfirst": True},
    {"dayfirst": False},
]


def outcome(m: Any, values: list[Any], options: dict[str, Any]) -> Any:
    """The instants as text and the type, or the kind of error and its first line."""
    try:
        read = m.to_datetime(values, **options)
    except Exception as error:
        kind = type(error).__name__.replace("InvalidArgumentError", "ValueError")
        return kind, str(error).split("\n")[0]
    texts = [None if value is None or str(value) == "NaT" else str(value) for value in read]
    return texts, str(read.dtype)


@pytest.mark.parametrize("options", ORDERS)
@pytest.mark.parametrize("text", TEXTS)
def test_one_value_reads_as_pandas_reads_it(
    firepanda: ModuleType, text: str, options: dict[str, Any]
) -> None:
    """The guessed format, or the row read on its own, in the order asked for."""
    import pandas as pd

    assert outcome(firepanda, [text], options) == outcome(pd, [text], options)


@pytest.mark.parametrize("options", ORDERS)
@pytest.mark.parametrize("text", TEXTS)
def test_every_row_on_its_own_reads_as_pandas_reads_it(
    firepanda: ModuleType, text: str, options: dict[str, Any]
) -> None:
    """`format="mixed"`, and a first row that is missing, which leaves nothing to guess from."""
    import pandas as pd

    mixed = {**options, "format": "mixed"}
    assert outcome(firepanda, [text, "13/01/2024"], mixed) == outcome(
        pd, [text, "13/01/2024"], mixed
    )
    assert outcome(firepanda, [None, text], options) == outcome(pd, [None, text], options)


@pytest.mark.parametrize("text", TEXTS)
def test_the_rest_of_the_column_is_held_to_the_first_row(firepanda: ModuleType, text: str) -> None:
    """A later row read against the format guessed day first from the first one."""
    import pandas as pd

    values = [text, "13/01/2024"]
    assert outcome(firepanda, values, {"dayfirst": True}) == outcome(pd, values, {"dayfirst": True})


@pytest.mark.parametrize(
    "options",
    [
        {"format": "ISO8601", "dayfirst": True},
        {"format": "%Y-%m-%d", "dayfirst": True},
        {"format": "%d/%m/%Y", "yearfirst": True},
    ],
)
@pytest.mark.parametrize("text", ["2024-01-02", "02/01/2024", "2024-13-01"])
def test_a_format_ignores_the_order(
    firepanda: ModuleType, text: str, options: dict[str, Any]
) -> None:
    """The format says where the day is, so the order asked for changes nothing."""
    import pandas as pd

    assert outcome(firepanda, [text], options) == outcome(pd, [text], options)


@pytest.mark.parametrize("text", ["01/02/2024", "2024-01-02", "10/11/12"])
def test_an_index_reads_in_the_order_asked_for(firepanda: ModuleType, text: str) -> None:
    """`DatetimeIndex` takes the same two arguments and reads each label on its own."""
    import pandas as pd

    for options in ORDERS:
        ours = [str(value) for value in firepanda.DatetimeIndex([text], **options)]
        assert ours == [str(value) for value in pd.DatetimeIndex([text], **options)]


def test_a_year_before_1000_is_padded_in_print_and_not_in_the_echo(firepanda: ModuleType) -> None:
    """pandas prints `0001-01-02 00:00:00` and echoes `Timestamp('1-01-02 00:00:00')`."""
    import pandas as pd

    ours = firepanda.to_datetime(["01/02"], format="mixed")[0]
    theirs = pd.to_datetime(["01/02"], format="mixed")[0]
    assert (str(ours), repr(ours)) == (str(theirs), repr(theirs))
