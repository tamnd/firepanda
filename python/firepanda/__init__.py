"""firepanda, seen from Python.

This package is the thin half. Everything it does eventually happens in
`_firepanda`, the compiled extension built from `firepanda/py/module.mojo`, and
the Python here exists to give that extension a name people can import, a place
for type stubs to live, and somewhere to put the pure Python conveniences that
are not worth crossing the boundary for.

It is more than a name, though, and document 13 is why. `PythonTypeBuilder` can
attach methods to a type and nothing else, so `df["a"]`, `len(df)`, `df.shape`
and every operator are not expressible in Mojo, and 28 percent of the pandas
surface is properties and operators. A bound type cannot be subclassed and has no
`__dict__` either, so neither of the obvious ways round that is available. The
pandas API therefore lives here, in `_frame.py`, and each class holds an
extension object and delegates to it. The Mojo side is a private calling
convention with names like `length` and no dunders at all.

Both halves are generated from the table in `tools/bindings.py`, because they are
the two files in this project most likely to drift apart and they sit on opposite
sides of a language boundary where nothing would notice.

The import of `_firepanda` below is deliberately at module level and
deliberately not guarded. A firepanda install with no extension in it is not a
degraded install, it is a broken one, and an ImportError naming the missing
extension is a better thing to hand someone than an AttributeError from
whichever function they called first.
"""

from __future__ import annotations

from . import _firepanda
from ._frame import DataFrame, read_csv

__all__ = ["DataFrame", "__version__", "read_csv"]

__version__: str = _firepanda.version()
"""The version, asked of the extension rather than written down here.

There are already three copies of this string, in `pixi.toml`, in
`pyproject.toml` and in `firepanda/version.mojo`, and they are kept in step by
hand during a release. A fourth copy here would be the one most likely to drift,
because it is the one furthest from the build, so this asks the extension what
it was compiled from instead. When the wheel metadata and this disagree, the
wheel was assembled out of two different builds, and `python/tests` is where
that gets caught.
"""
