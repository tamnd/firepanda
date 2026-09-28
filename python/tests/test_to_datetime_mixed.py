"""`to_datetime` of a list mixing moments with text, against pandas.

pandas takes each `Timestamp`, `datetime` or `date` as it is and reads the text
with the format it works out from the first piece of text, so a list mixing the
two reads as one column of instants. A number among moments is a count of
nanoseconds since 1970, and then every instant is held in nanoseconds.
`to_timedelta` reads text beside spans at microseconds, and at nanoseconds when
a number is there too.
"""

from __future__ import annotations

import datetime
import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def stamp(m: ModuleType) -> Any:
    return m.Timestamp("2020-01-02 03:04:05")


CALLS: dict[str, Callable[[ModuleType], Any]] = {
    "stamp and text": lambda m: m.to_datetime([stamp(m), "2020-05-05"]),
    "gaps": lambda m: m.to_datetime([stamp(m), m.NaT, "2020-05-05", None]),
    "datetime": lambda m: m.to_datetime([datetime.datetime(2020, 1, 2, 3), "2020-05-05 01:02"]),
    "date": lambda m: m.to_datetime([datetime.date(2020, 1, 2), "2020-05-05"]),
    "tuple": lambda m: m.to_datetime((datetime.datetime(2020, 1, 1), "2020-01-02T10:00")),
    "format guessed": lambda m: m.to_datetime([stamp(m), "05/06/2020", "07/08/2021"]),
    "format given": lambda m: m.to_datetime([stamp(m), "05/06/2020"], format="%d/%m/%Y"),
    "dayfirst": lambda m: m.to_datetime([stamp(m), "05/06/2020"], dayfirst=True),
    "coerce": lambda m: m.to_datetime([stamp(m), "nope"], errors="coerce"),
    "utc": lambda m: m.to_datetime([stamp(m), "2020-05-05"], utc=True),
    "zoned": lambda m: m.to_datetime([stamp(m).tz_localize("UTC"), "2020-05-05 00:00+00:00"]),
    "count": lambda m: m.to_datetime([stamp(m), 5]),
    "count at seconds": lambda m: m.to_datetime([stamp(m).as_unit("s"), 2**62]),
    "count and float": lambda m: m.to_datetime([datetime.date(2020, 1, 2), 5, 1.5e18]),
    "count and text": lambda m: m.to_datetime([stamp(m), 5, "2020-05-05"]),
    "count and gaps": lambda m: m.to_datetime([stamp(m), 5, m.NaT, None, float("nan")]),
    "count utc": lambda m: m.to_datetime([stamp(m).tz_localize("Asia/Tokyo"), 5], utc=True),
    "nan": lambda m: m.to_datetime([stamp(m), float("nan")]),
    "spans and text": lambda m: m.to_timedelta([m.Timedelta("1s").as_unit("s"), "2h", None]),
    "timedelta and text": lambda m: m.to_timedelta([datetime.timedelta(1), "2h"]),
    "spans and count": lambda m: m.to_timedelta([m.Timedelta("1s"), 5, "2h"]),
}


@needs_pandas
@pytest.mark.parametrize("name", list(CALLS))
def test_moments_among_text_read_as_in_pandas(firepanda: ModuleType, name: str) -> None:
    """The type and every value match."""
    import pandas as pd

    mine = CALLS[name](firepanda)
    them = CALLS[name](pd)
    assert str(mine.dtype) == str(them.dtype)
    assert [repr(v) for v in mine.tolist()] == [repr(v) for v in them.tolist()]


def test_text_that_will_not_read_raises(firepanda: ModuleType) -> None:
    """The text is read by the same rules as without a moment beside it."""
    with pytest.raises(ValueError):
        firepanda.to_datetime([stamp(firepanda), "nope"])


@needs_pandas
@pytest.mark.parametrize("values", [[1, True], [1, 5, "UTC"]], ids=["flag", "zoned count"])
def test_a_moment_beside_what_pandas_refuses_raises_its_error(
    firepanda: ModuleType, values: list[Any]
) -> None:
    """A flag is a `TypeError`, and a zoned moment beside a count a `ValueError`."""
    import pandas as pd

    def build(m: ModuleType) -> list[Any]:
        moment = stamp(m).tz_localize(values[2]) if len(values) == 3 else stamp(m)
        return [moment, values[1]]

    with pytest.raises(Exception) as theirs:
        pd.to_datetime(build(pd))
    with pytest.raises(type(theirs.value), match=str(theirs.value).split(".")[0]):
        firepanda.to_datetime(build(firepanda))
