"""The platform and version flags pandas keeps in `pandas.compat`.

Libraries built on pandas read these to decide what the running interpreter,
machine and pyarrow can do. The pyarrow flags are worked out the first time
one is read, so importing firepanda does not import pyarrow.
"""

from __future__ import annotations

import os
import platform
import sys
import sysconfig
from typing import Any

__all__ = [
    "CHAINED_WARNING_DISABLED",
    "HAS_PYARROW",
    "IS64",
    "ISMUSL",
    "PY312",
    "PY314",
    "PYARROW_INSTALLED",
    "PYARROW_MIN_VERSION",
    "PYPY",
    "WASM",
    "is_numpy_dev",
    "pa_version_under14p0",
    "pa_version_under14p1",
    "pa_version_under16p0",
    "pa_version_under17p0",
    "pa_version_under18p0",
    "pa_version_under19p0",
    "pa_version_under20p0",
    "pa_version_under21p0",
    "pa_version_under22p0",
    "pa_version_under23p0",
]

IS64 = sys.maxsize > 2**32
PY312 = sys.version_info >= (3, 12)
PY314 = sys.version_info >= (3, 14)
PYPY = platform.python_implementation() == "PyPy"
WASM = sys.platform == "emscripten" or platform.machine() in ["wasm32", "wasm64"]
ISMUSL = "musl" in (sysconfig.get_config_var("HOST_GNU_TYPE") or "")
CHAINED_WARNING_DISABLED = PYPY
PYARROW_MIN_VERSION = "13.0.0"

_UNDER = {
    "pa_version_under14p0": "14.0.0",
    "pa_version_under14p1": "14.0.1",
    "pa_version_under16p0": "16.0.0",
    "pa_version_under17p0": "17.0.0",
    "pa_version_under18p0": "18.0.0",
    "pa_version_under19p0": "19.0.0",
    "pa_version_under20p0": "20.0.0",
    "pa_version_under21p0": "21.0.0",
    "pa_version_under22p0": "22.0.0",
    "pa_version_under23p0": "23.0.0",
}


def _flags() -> dict[str, Any]:
    """The pyarrow flags, all True with pyarrow missing, as pandas sets them."""
    from ..util.version import Version

    try:
        import pyarrow
    except ImportError:
        return {**dict.fromkeys(_UNDER, True), "PYARROW_INSTALLED": False, "HAS_PYARROW": False}
    installed = Version(Version(pyarrow.__version__).base_version)
    flags: dict[str, Any] = {name: installed < Version(at) for name, at in _UNDER.items()}
    flags["PYARROW_INSTALLED"] = True
    flags["HAS_PYARROW"] = installed >= Version(PYARROW_MIN_VERSION)
    return flags


def _numpy_dev() -> bool:
    from ..util.version import Version

    try:
        import numpy
    except ImportError:
        return False
    return Version(numpy.__version__).dev is not None


def __getattr__(name: str) -> Any:
    if name in _UNDER or name in ("PYARROW_INSTALLED", "HAS_PYARROW"):
        globals().update(_flags())
        return globals()[name]
    if name == "is_numpy_dev":
        globals()["is_numpy_dev"] = _numpy_dev()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def set_function_name(f: Any, name: str, cls: type) -> Any:
    """The function renamed as a method of the class, for its name, qualified name and module."""
    f.__name__ = name
    f.__qualname__ = f"{cls.__name__}.{name}"
    f.__module__ = cls.__module__
    return f


def is_platform_little_endian() -> bool:
    return sys.byteorder == "little"


def is_platform_windows() -> bool:
    return sys.platform in ["win32", "cygwin"]


def is_platform_linux() -> bool:
    return sys.platform == "linux"


def is_platform_mac() -> bool:
    return sys.platform == "darwin"


def is_platform_arm() -> bool:
    return platform.machine() in ("arm64", "aarch64") or platform.machine().startswith("armv")


def is_platform_power() -> bool:
    return platform.machine() in ("ppc64", "ppc64le")


def is_platform_riscv64() -> bool:
    return platform.machine() == "riscv64"


def is_ci_environment() -> bool:
    """Whether the PANDAS_CI variable is 1, which is how pandas' own CI marks itself."""
    return os.environ.get("PANDAS_CI", "0") == "1"
