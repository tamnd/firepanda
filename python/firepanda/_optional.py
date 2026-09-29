"""pandas' sentence for an optional dependency that is not installed.

pandas imports every optional package through one helper, which names the
package by what is installed rather than by what is imported, so `odf` is
asked for as `odfpy`, and words the failure the same way everywhere. The
readers and writers that lean on such a package use this to say it the same
way.
"""

from __future__ import annotations

import importlib
from typing import Any

# The packages whose name to install differs from the module imported.
_INSTALL = {
    "bs4": "beautifulsoup4",
    "bottleneck": "Bottleneck",
    "jinja2": "Jinja2",
    "lxml.etree": "lxml",
    "odf": "odfpy",
    "python_calamine": "python-calamine",
    "sqlalchemy": "SQLAlchemy",
    "tables": "pytables",
}


def missing(name: str, extra: str = "") -> ImportError:
    """The ImportError pandas raises when `name` cannot be imported."""
    install = _INSTALL.get(name, name)
    return ImportError(
        f"`Import {install}` failed. {extra} Use pip or conda to install the {install} package."
    )


def imported(name: str, extra: str = "") -> Any:
    """The module `name`, or pandas' ImportError when it is not installed."""
    try:
        return importlib.import_module(name)
    except ImportError:
        raise missing(name, extra) from None
