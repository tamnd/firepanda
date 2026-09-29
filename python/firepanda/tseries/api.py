"""pandas' `pandas.tseries.api`: guessing a format, inferring a frequency, and the offsets."""

from __future__ import annotations

from .. import offsets
from .._frequency import infer_freq
from .._row_formats import guess

__all__ = ["guess_datetime_format", "infer_freq", "offsets"]


def guess_datetime_format(dt_str: str, dayfirst: bool | None = False) -> str | None:
    """The `strptime` format pandas guesses from one value, or None when it cannot guess one."""
    return guess(dt_str, dayfirst=bool(dayfirst))
