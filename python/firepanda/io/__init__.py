"""The readers and writers, which pandas groups under `pandas.io`.

`firepanda.io.api` names the same readers that `firepanda` itself exports,
the way `pandas.io.api` does.
"""

from __future__ import annotations

from . import api

__all__ = ["api"]
