"""The core objects, which pandas groups under `pandas.core`.

`firepanda.core.api` names the frames, indexes, dtypes, scalars and top level
functions that `pandas.core.api` names, as the same objects `firepanda`
exports at the top.
"""

from __future__ import annotations

from . import api

__all__ = ["api"]
