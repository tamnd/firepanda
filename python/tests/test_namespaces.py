"""`compat`, `core`, `io` and the `errors` submodules, compared with pandas.

These are the module names pandas exposes next to its functions. Libraries
reach into them for platform flags, for the core names and the readers, and
for the chained assignment messages, so each one answers the same way here.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

FLAGS = [
    *("CHAINED_WARNING_DISABLED", "HAS_PYARROW", "IS64", "ISMUSL", "PY312", "PY314"),
    *("PYARROW_INSTALLED", "PYARROW_MIN_VERSION", "PYPY", "WASM", "is_numpy_dev"),
    *(f"pa_version_under{v}" for v in ("14p0", "14p1", "16p0", "17p0", "18p0", "19p0")),
    *(f"pa_version_under{v}" for v in ("20p0", "21p0", "22p0", "23p0")),
]

PLATFORMS = [
    *("is_platform_little_endian", "is_platform_windows", "is_platform_linux"),
    *("is_platform_mac", "is_platform_arm", "is_platform_power", "is_platform_riscv64"),
    "is_ci_environment",
]

MESSAGES = [
    "_chained_assignment_msg",
    "_chained_assignment_method_msg",
    "_chained_assignment_method_update_msg",
]


def module(pd: ModuleType, name: str) -> ModuleType:
    return importlib.import_module(f"{pd.__name__}.{name}")


def public(mod: ModuleType) -> set[str]:
    return {name for name in dir(mod) if not name.startswith("_") and name != "annotations"}


@needs_pandas
@pytest.mark.parametrize("name", FLAGS)
def test_compat_flags_match_pandas(firepanda: ModuleType, name: str) -> None:
    import pandas as pd

    assert getattr(module(firepanda, "compat"), name) == getattr(module(pd, "compat"), name)


@needs_pandas
@pytest.mark.parametrize("name", PLATFORMS)
def test_compat_platform_checks_match_pandas(firepanda: ModuleType, name: str) -> None:
    import pandas as pd

    assert getattr(module(firepanda, "compat"), name)() == getattr(module(pd, "compat"), name)()


@needs_pandas
def test_compat_lists_the_same_names(firepanda: ModuleType) -> None:
    import pandas as pd

    assert module(firepanda, "compat").__all__ == module(pd, "compat").__all__
    with pytest.raises(AttributeError, match="has no attribute 'pa_version_under15p0'"):
        module(firepanda, "compat").pa_version_under15p0  # noqa: B018


def test_set_function_name_names_a_method(firepanda: ModuleType) -> None:
    class Holder:
        pass

    def f() -> None:
        pass

    named = module(firepanda, "compat").set_function_name(f, "method", Holder)
    assert named is f
    assert (f.__name__, f.__qualname__, f.__module__) == ("method", "Holder.method", __name__)


def test_ci_environment_reads_pandas_ci(firepanda: ModuleType, monkeypatch: Any) -> None:
    compat = module(firepanda, "compat")
    monkeypatch.setenv("PANDAS_CI", "1")
    assert compat.is_ci_environment() is True
    monkeypatch.setenv("PANDAS_CI", "0")
    assert compat.is_ci_environment() is False


@needs_pandas
@pytest.mark.parametrize("name", MESSAGES)
def test_chained_assignment_messages_match_pandas(firepanda: ModuleType, name: str) -> None:
    import pandas as pd

    assert getattr(module(firepanda, "errors.cow"), name) == getattr(module(pd, "errors.cow"), name)


def test_errors_carries_its_submodules(firepanda: ModuleType) -> None:
    import abc
    import ctypes

    errors = firepanda.errors
    assert errors.abc is abc
    assert errors.ctypes is ctypes
    assert errors.cow is module(firepanda, "errors.cow")


@needs_pandas
def test_core_api_is_the_top_level_objects(firepanda: ModuleType) -> None:
    import pandas as pd

    api = module(firepanda, "core.api")
    assert public(api) == public(module(pd, "core.api"))
    for name in public(api):
        assert getattr(api, name) is getattr(firepanda, name), name
    assert firepanda.core.api is api


@needs_pandas
def test_io_api_is_the_top_level_readers(firepanda: ModuleType) -> None:
    import pandas as pd

    api = module(firepanda, "io.api")
    assert public(api) <= public(module(pd, "io.api"))
    for name in public(api):
        assert getattr(api, name) is getattr(firepanda, name), name
    assert firepanda.io.api is api
    missing = public(module(pd, "io.api")) - public(api)
    assert all(not hasattr(firepanda, name) for name in missing)
