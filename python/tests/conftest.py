"""One place that knows how an installed firepanda is laid out.

The tree and the wheel do not agree about where the extension lives. In the tree
the Python is in `python/firepanda/` and the compiled `_firepanda.so` is in
`build/extension/`, because one is checked in and the other is a build product.
In a wheel they are the same directory, and they have to be, since
`python/firepanda/__init__.py` does `from . import _firepanda` and the runtime
libraries are found through `@loader_path` and `$ORIGIN`, which mean the
directory the extension was loaded from.

So every test that wants to import firepanda has to build that directory first.
This module does it once, and it is the only place in the test suite that knows
the layout, which matters because the layout is exactly what
`tools/build_extension.sh` and the wheel packaging have to keep agreeing about.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "python" / "firepanda"
BUILT = REPO / "build" / "extension"


def stage(root: Path) -> Path:
    """Lays out the package the way an installed wheel has it, under `root`.

    The extension and the runtime libraries go inside the package directory
    rather than beside it, which is what makes `@loader_path` and `$ORIGIN` the
    right answer: the library looks for its neighbours in the directory it was
    loaded from, and in a wheel that directory is the package.
    """
    package = root / "firepanda"
    package.mkdir()
    for source in PACKAGE.iterdir():
        if source.is_file():
            shutil.copy2(source, package / source.name)
    for source in BUILT.iterdir():
        shutil.copy2(source, package / source.name)
    return package


@pytest.fixture
def staged() -> Callable[[Path], Path]:
    """Hands back the staging function, for tests that want their own copy."""
    return stage


@pytest.fixture(scope="session")
def firepanda(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    """Imports a staged firepanda, once for the whole run.

    Once, rather than per test, because a second import of the same extension
    into the same interpreter is a different question from the one any of these
    tests is asking. `test_extension.py` deliberately does its import in a child
    interpreter with the environment stripped, and that is where the question of
    whether the artifact stands alone belongs. Here we just want the module.
    """
    if not (BUILT / "_firepanda.so").exists():
        pytest.skip("no extension built, run `pixi run build-extension` first")

    root = tmp_path_factory.mktemp("staged")
    stage(root)
    sys.path.insert(0, str(root))
    import firepanda

    return firepanda
