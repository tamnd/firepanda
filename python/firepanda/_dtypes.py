"""pandas' `StringDtype` and `DatetimeTZDtype`, each equal to the text firepanda names its type by.

firepanda names a column's type with text, and pandas' dtype objects compare
equal to theirs, so each class here is that text with pandas' attributes on
it. Passing one as `dtype=` is passing the text.
"""

from __future__ import annotations

import datetime
import zoneinfo
from typing import Any

from ._na import NA
from ._scalars import NaT, Timestamp
from .errors import InvalidArgumentError

_STORAGES = ("python", "pyarrow")
_UNITS = ("s", "ms", "us", "ns")


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and value != value


def _default_storage() -> str:
    """`pyarrow` when it can be imported, as pandas picks, and `python` otherwise."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return "python"
    return "pyarrow"


class StringDtype(str):
    """pandas' type for a column of text, `string` with `NA` for a gap and `str` with NaN."""

    def __new__(cls, storage: str | None = None, na_value: Any = NA) -> StringDtype:
        if storage is None:
            storage = _default_storage()
        if storage not in _STORAGES:
            raise InvalidArgumentError(
                f"Storage must be 'python' or 'pyarrow'. Got {storage} instead."
            )
        if na_value is not NA and not _is_nan(na_value):
            raise InvalidArgumentError(f"'na_value' must be np.nan or pd.NA, got {na_value}")
        made = super().__new__(cls, "string" if na_value is NA else "str")
        made._storage = storage
        made._na_value = na_value
        return made

    @property
    def storage(self) -> str:
        """Where the text is held, `python` or `pyarrow`."""
        return self._storage

    @property
    def na_value(self) -> Any:
        """What a gap reads as, `NA` or NaN."""
        return self._na_value

    @property
    def name(self) -> str:
        """`string` for the type with `NA` and `str` for the one with NaN."""
        return str.__str__(self)

    @property
    def type(self) -> type:
        """The class of each value, `str`."""
        return str

    @property
    def kind(self) -> str:
        """numpy's letter for the type, `O`."""
        return "O"

    def __repr__(self) -> str:
        storage = "" if self._storage == "pyarrow" else "storage='python', "
        return f"<StringDtype({storage}na_value={self._na_value})>"

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(("StringDtype", self._storage, _is_nan(self._na_value)))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, StringDtype):
            return self._storage == other._storage and _is_nan(self._na_value) == _is_nan(
                other._na_value
            )
        if isinstance(other, str):
            return other in ("string", self.name, f"{self.name}[{self._storage}]")
        return False

    def __ne__(self, other: object) -> bool:
        return not self == other


def _zone(tz: Any) -> datetime.tzinfo:
    """A zone given by name or as a tzinfo, as the tzinfo pandas holds."""
    if isinstance(tz, datetime.tzinfo):
        return tz
    if tz == "UTC":
        return datetime.UTC
    return zoneinfo.ZoneInfo(tz)


class DatetimeTZDtype(str):
    """pandas' type for instants read on a clock, equal to its text, like `datetime64[ns, UTC]`."""

    def __new__(cls, unit: Any = "ns", tz: Any = None) -> DatetimeTZDtype:
        if isinstance(unit, DatetimeTZDtype):
            unit, tz = unit.unit, unit.tz
        if unit not in _UNITS:
            if isinstance(unit, str) and unit.startswith("datetime64["):
                raise InvalidArgumentError(
                    f"Passing a dtype alias like '{unit}' to DatetimeTZDtype is no longer"
                    " supported. Use 'DatetimeTZDtype.construct_from_string()' instead."
                )
            raise InvalidArgumentError("DatetimeTZDtype only supports s, ms, us, ns units")
        if tz is None:
            raise TypeError("A 'tz' is required.")
        zone = _zone(tz)
        made = super().__new__(cls, f"datetime64[{unit}, {zone}]")
        made._unit = unit
        made._tz = zone
        return made

    @classmethod
    def construct_from_string(cls, string: str) -> DatetimeTZDtype:
        """The type named by text like `datetime64[ns, UTC]`.

        Raises:
            TypeError: For text that does not name one, in pandas' words.
        """
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        inside = string[len("datetime64[") : -1] if string.startswith("datetime64[") else ""
        unit, _, tz = inside.partition(", ")
        if not string.endswith("]") or unit not in _UNITS or not tz:
            raise TypeError(f"Cannot construct a 'DatetimeTZDtype' from '{string}'")
        return cls(unit, tz)

    @property
    def unit(self) -> str:
        """The resolution, one of s, ms, us and ns."""
        return self._unit

    @property
    def tz(self) -> datetime.tzinfo:
        """The clock, as a tzinfo."""
        return self._tz

    @property
    def name(self) -> str:
        """The text, like `datetime64[ns, UTC]`."""
        return str.__str__(self)

    @property
    def type(self) -> type:
        """The class of each value, `Timestamp`."""
        return Timestamp

    @property
    def kind(self) -> str:
        """numpy's letter for the type, `M`."""
        return "M"

    @property
    def na_value(self) -> Any:
        """What a gap reads as, `NaT`."""
        return NaT

    @property
    def base(self) -> Any:
        """The numpy type of the instants without the clock."""
        import numpy

        return numpy.dtype(f"datetime64[{self._unit}]")

    @property
    def str(self) -> str:
        """numpy's short text for `base`, like `|M8[ns]`."""
        return f"|M8[{self._unit}]"

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and str.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self == other
