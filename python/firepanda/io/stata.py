"""The Stata reader, where pandas keeps it as `pandas.io.stata`."""

from __future__ import annotations

from .._stata import StataMissingValue, StataReader, read_stata

__all__ = ["StataMissingValue", "StataReader", "read_stata"]
