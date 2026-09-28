"""`show_versions`, which prints the system and the versions of what firepanda uses.

The system part holds the same keys as pandas' report. The dependency part
names firepanda and the libraries it reads or writes through, with the
version each has installed, or None when one is not installed. firepanda's
own version comes from its extension, and every other one is read from the
installed package's metadata, so nothing is imported.
"""

from __future__ import annotations

import json
import locale
import os
import platform
import struct
import sys
from importlib import metadata
from typing import Any

_DEPENDENCIES = (
    "firepanda",
    "numpy",
    "pyarrow",
    "pandas",
    "python-dateutil",
    "tzdata",
    "pip",
    "IPython",
)
"""The packages whose versions are shown, firepanda first."""


def _version(package: str) -> str | None:
    if package == "firepanda":
        from . import _firepanda

        return _firepanda.version()
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _system() -> dict[str, Any]:
    uname = platform.uname()
    language_code, encoding = locale.getlocale()
    return {
        "commit": None,
        "python": platform.python_version(),
        "python-bits": struct.calcsize("P") * 8,
        "OS": uname.system,
        "OS-release": uname.release,
        "Version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "byteorder": sys.byteorder,
        "LC_ALL": os.environ.get("LC_ALL"),
        "LANG": os.environ.get("LANG"),
        "LOCALE": {"language-code": language_code, "encoding": encoding},
    }


def show_versions(as_json: str | bool = False) -> None:
    """Prints the system and the version of each dependency, for a bug report.

    Args:
        as_json: False to print a table, True to print the same as JSON, or a
            path to write the JSON to that file instead.
    """
    system = _system()
    dependencies = {package: _version(package) for package in _DEPENDENCIES}
    if as_json:
        report = {"system": system, "dependencies": dependencies}
        if as_json is True:
            sys.stdout.writelines(json.dumps(report, indent=2))
        else:
            with open(as_json, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        return
    locale_of = system["LOCALE"]
    system["LOCALE"] = f"{locale_of['language-code']}.{locale_of['encoding']}"
    widest = max(len(package) for package in dependencies)
    print("\nINSTALLED VERSIONS")
    print("------------------")
    for key, value in system.items():
        print(f"{key:<{widest}}: {value}")
    print("")
    for key, value in dependencies.items():
        print(f"{key:<{widest}}: {value}")
