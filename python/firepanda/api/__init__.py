"""`firepanda.api`, which is the part of pandas that other libraries import.

`pandas.api` is the namespace pandas promises is public. Everything else in
pandas is public by convention and this is public by contract, which is why the
code that reads it is almost never a notebook. It is scikit-learn deciding
whether a column is numeric before it fits, seaborn deciding whether to draw a
histogram or a bar chart, and every wrapper that takes a frame from somebody
else and has to ask what is in it.

That makes it a strange shape for a compatibility project. None of it computes
anything. All of it is questions about types, asked by code that never called
firepanda and does not know it is talking to firepanda. If `import firepanda as
pd` is the claim, then `pd.api.types.is_numeric_dtype(df["a"])` has to answer,
and it has to answer the same thing pandas would.

Only `types` is here. `pandas.api` also holds `extensions`, `indexers`,
`interchange` and `typing`, and those are absent rather than empty, because each
of them hands out machinery for extending pandas rather than for reading it and
firepanda has no extension mechanism to hand out.
"""

from __future__ import annotations

from . import types

__all__ = ["types"]
