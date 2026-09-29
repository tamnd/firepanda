"""The `tseries` namespace and the `pandas` self name, compared with pandas.

`firepanda.tseries` carries `api`, `frequencies` and `offsets` the way
`pandas.tseries` does, and `firepanda.pandas` is the package itself.
"""

from __future__ import annotations

import datetime
import importlib
import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

FREQUENCIES = [
    *("D", "d", "2D", "2 D", "h", "3h", "min", "s", "ms", "us", "ns", "-2D", "1.5h", "2D3h"),
    *("ME", "MS", "QE", "QE-JAN", "YE", "YS-MAR", "W", "W-SUN", "B", "BME", "SME"),
    *("", " ", "3", "-", "bad", "2bad", "M", "2M", "Q", "Q-JAN", "Y", "Y-DEC", "BM", "BQ"),
    *("BY", "SM", "CBM", "H", "T", "S", "L", "U", "N", "A", "BH"),
    *("D-FOO", "W-FOO", "MS-JAN", "ME-JAN"),
]

GUESSES = ["2026-01-02", "01/02/2026", "2026-01-02 10:11:12", "2026-01-02T10:11:12.123"]
GUESSES += ["Jan 2 2026", "nonsense", "20260102"]

ALIASES = ["ME", "QE-DEC", "YE", "D", "h", "MS", "W-SUN", "B", "nope", "QS-FEB", "BYE-MAR"]


def outcome(call: Callable[[], Any]) -> str:
    """What a call gives, as its repr or as the builtin kind and words of its error."""
    try:
        return repr(call())
    except Exception as error:
        kind = next(k for k in type(error).__mro__ if k.__module__ == "builtins")
        return f"{kind.__name__}: {error}"


def parts(pd: ModuleType) -> tuple[ModuleType, ModuleType, ModuleType]:
    name = pd.__name__
    return (
        importlib.import_module(name + ".tseries.frequencies"),
        importlib.import_module(name + ".tseries.api"),
        importlib.import_module(name + ".tseries.offsets"),
    )


@needs_pandas
@pytest.mark.parametrize("text", FREQUENCIES)
def test_to_offset_matches_pandas(firepanda: ModuleType, text: str) -> None:
    """Text reads as the same offset, or fails with the same words."""
    import pandas as pd

    mine, theirs = parts(firepanda)[0], parts(pd)[0]
    assert outcome(lambda: mine.to_offset(text)) == outcome(lambda: theirs.to_offset(text))


@needs_pandas
def test_to_offset_takes_spans_offsets_and_none(firepanda: ModuleType) -> None:
    """A span, an offset and None pass through the way pandas passes them."""
    import pandas as pd

    for pandas in (firepanda, pd):
        assert parts(pandas)[0].to_offset(None) is None
    mine, _, my_offsets = parts(firepanda)
    theirs, _, their_offsets = parts(pd)
    day = datetime.timedelta(days=1)
    assert repr(mine.to_offset(day)) == repr(theirs.to_offset(day))
    assert repr(mine.to_offset(firepanda.Timedelta("90min"))) == repr(
        theirs.to_offset(pd.Timedelta("90min"))
    )
    assert repr(mine.to_offset(my_offsets.Day(3))) == repr(theirs.to_offset(their_offsets.Day(3)))
    assert outcome(lambda: mine.to_offset(5)) == outcome(lambda: theirs.to_offset(5))


@needs_pandas
@pytest.mark.parametrize("text", GUESSES)
def test_guess_datetime_format_matches_pandas(firepanda: ModuleType, text: str) -> None:
    """The format guessed for a date string is pandas' guess."""
    import pandas as pd

    mine, theirs = parts(firepanda)[1], parts(pd)[1]
    assert mine.guess_datetime_format(text) == theirs.guess_datetime_format(text)


def test_guess_datetime_format_reads_the_day_first(firepanda: ModuleType) -> None:
    api = parts(firepanda)[1]
    assert api.guess_datetime_format("02/01/2026", dayfirst=True) == "%d/%m/%Y"


@needs_pandas
@pytest.mark.parametrize("text", ALIASES)
def test_get_period_alias_matches_pandas(firepanda: ModuleType, text: str) -> None:
    import pandas as pd

    mine, theirs = parts(firepanda)[0], parts(pd)[0]
    assert mine.get_period_alias(text) == theirs.get_period_alias(text)


@needs_pandas
def test_infer_freq_is_in_both_places(firepanda: ModuleType) -> None:
    import pandas as pd

    frequencies, api, _ = parts(firepanda)
    weekly = firepanda.date_range("2026-01-01", periods=4, freq="W")
    assert frequencies.infer_freq(weekly) == pd.infer_freq(
        pd.date_range("2026-01-01", periods=4, freq="W")
    )
    assert api.infer_freq(firepanda.date_range("2026-01-01", periods=4)) == "D"


def test_offsets_are_the_same_classes(firepanda: ModuleType) -> None:
    offsets = parts(firepanda)[2]
    assert offsets.MonthEnd is firepanda.offsets.MonthEnd
    assert set(offsets.__all__) == set(firepanda.offsets.__all__)
    assert firepanda.tseries.offsets is offsets


def test_the_package_names_itself(firepanda: ModuleType) -> None:
    assert firepanda.pandas is firepanda
    assert firepanda.pandas.DataFrame is firepanda.DataFrame
