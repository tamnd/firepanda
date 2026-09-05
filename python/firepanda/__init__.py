"""firepanda, seen from Python.

This package is the thin half. Everything it does eventually happens in
`_firepanda`, the compiled extension built from `firepanda/py/module.mojo`, and
the Python here exists to give that extension a name people can import, a place
for type stubs to live, and somewhere to put the pure Python conveniences that
are not worth crossing the boundary for.

Right now it does one thing, which is report the version. That is not a feature
and is not meant to become one. It is the smallest end to end path through the
whole boundary, from a `pixi run build-extension` to a working import, and
having it land on its own means the interesting parts arrive on top of something
already known to work rather than alongside it.

The import of `_firepanda` below is deliberately at module level and
deliberately not guarded. A firepanda install with no extension in it is not a
degraded install, it is a broken one, and an ImportError naming the missing
extension is a better thing to hand someone than an AttributeError from
whichever function they called first.
"""

from __future__ import annotations

from . import _firepanda

__all__ = ["__version__"]

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
