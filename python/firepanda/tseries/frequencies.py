"""pandas' `pandas.tseries.frequencies`: a frequency read as an offset, inferred, or named.

`to_offset` reads text, a span or an offset the way every `freq` argument is
read. `get_period_alias` is pandas' table from an offset's alias to the name
the same step has as a period frequency, so `ME` is `M` and `QE-JAN` is
`Q-JAN`, with None for an alias that has no period form.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

from .._frequency import _offset_of, infer_freq

__all__ = ["get_period_alias", "infer_freq", "to_offset"]

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_DAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")

_PERIOD_ALIASES = {
    "WEEKDAY": "D",
    "EOM": "M",
    "BME": "M",
    "SME": "M",
    "BMS": "M",
    "CBME": "M",
    "CBMS": "M",
    "SMS": "M",
    "BQS": "Q",
    "QS": "Q",
    "BQE": "Q",
    "MS": "M",
    "D": "D",
    "B": "B",
    "min": "min",
    "s": "s",
    "ms": "ms",
    "us": "us",
    "ns": "ns",
    "h": "h",
    "QE": "Q",
    "YE": "Y",
    "W": "W",
    "ME": "M",
    "BYE": "Y",
    "YS": "Y",
    "BYS": "Y",
}
for _month in _MONTHS:
    _PERIOD_ALIASES.update(
        {
            f"Q-{_month}": f"Q-{_month}",
            f"QE-{_month}": f"Q-{_month}",
            f"Y-{_month}": f"Y-{_month}",
            f"YE-{_month}": f"Y-{_month}",
            f"QS-{_month}": "Q",
            f"BQE-{_month}": "Q",
            f"BQS-{_month}": "Q",
            f"YS-{_month}": "Y",
            f"BYE-{_month}": "Y",
            f"BYS-{_month}": "Y",
        }
    )
for _day in _DAYS:
    _PERIOD_ALIASES[f"W-{_day}"] = f"W-{_day}"


def get_period_alias(offset_str: str) -> str | None:
    """The period frequency an offset alias stands for, or None when there is none."""
    return _PERIOD_ALIASES.get(offset_str)


_RENAMED = {
    "M": "ME",
    "Q": "QE",
    "Y": "YE",
    "BM": "BME",
    "BQ": "BQE",
    "BY": "BYE",
    "SM": "SME",
    "CBM": "CBME",
}
"""The aliases pandas 3 no longer takes for an offset, and the ones it asks for instead."""

_MEANT = {"H": "h", "T": "min", "S": "s", "L": "ms", "U": "us", "N": "ns", "A": "Y", "BH": "bh"}
"""The retired aliases pandas answers with a suggestion instead of a rename."""

_ALIAS = re.compile(r"\s*-?\d*\.?\d*\s*([A-Za-z]+)(-\w+)?\s*")


def _mistake(freq: str) -> str:
    """The words pandas' `to_offset` uses for text it cannot read, around the reason."""
    found = _ALIAS.fullmatch(freq)
    if found is None:
        inner = ValueError("last element must be blank")
        return f"Invalid frequency: {freq}. Failed to parse with error message: {inner!r}"
    name, suffix = found.group(1), found.group(2) or ""
    if name in _RENAMED:
        inner = ValueError(
            f"'{name}{suffix}' is no longer supported for offsets. Please use"
            f" '{_RENAMED[name]}{suffix}' instead."
        )
        return f"Invalid frequency: {freq}. Failed to parse with error message: {inner!r}"
    if name in _MEANT:
        hint = f" Did you mean {_MEANT[name]}?"
        inner = ValueError(
            f"Invalid frequency: {name}. Failed to parse with error message: KeyError('{name}')."
            + hint
        )
        return f"Invalid frequency: {freq}. Failed to parse with error message: {inner!r}{hint}"
    if suffix and name in ("W", "WOM"):
        inner = ValueError(
            f"Invalid frequency: {freq}. Failed to parse with error message:"
            f" KeyError('{suffix[1:]}')."
        )
        return f"Invalid frequency: {freq}. Failed to parse with error message: {inner!r}"
    if suffix:
        inner = ValueError(
            f"Invalid frequency: {freq}. Failed to parse with error message:"
            f" ValueError('Bad freq suffix {suffix[1:]}')."
        )
        return f"Invalid frequency: {freq}. Failed to parse with error message: {inner!r}"
    inner = ValueError(
        f"Invalid frequency: {name}. Failed to parse with error message: KeyError('{name}')."
    )
    return f"Invalid frequency: {freq}. Failed to parse with error message: {inner!r}"


def to_offset(freq: Any, is_period: bool = False) -> Any:
    """The offset `freq` stands for: text like `2D` or `QE-JAN`, a span, an offset, or None.

    Raises:
        ValueError: For text that is not a frequency, in pandas' words.
        TypeError: For anything that is not text, a span or an offset.
    """
    from .. import offsets
    from .._scalars import Timedelta
    from ..errors import InvalidArgumentError

    known = (str, datetime.timedelta, Timedelta, offsets.BaseOffset)
    if freq is not None and not isinstance(freq, known):
        raise TypeError(
            f"Argument 'freq' has incorrect type (expected str, got {type(freq).__name__})"
        )
    del is_period
    if isinstance(freq, str):
        if not freq.strip():
            raise InvalidArgumentError(f"Invalid frequency: {freq}.")
        found = _ALIAS.fullmatch(freq)
        if found is not None:
            count = freq[: found.start(1)].replace(" ", "")
            name = "D" if found.group(1) == "d" else found.group(1)
            freq_text = count + name + (found.group(2) or "")
        else:
            freq_text = freq
        try:
            return _offset_of(freq_text)
        except ValueError:
            raise InvalidArgumentError(_mistake(freq)) from None
    return _offset_of(freq)
