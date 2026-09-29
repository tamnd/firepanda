"""pandas' CSS reading, where pandas keeps it as `pandas.io.formats.css`."""

from __future__ import annotations

from ..._css import CSSResolver
from ...errors import CSSWarning

__all__ = ["CSSResolver", "CSSWarning"]
