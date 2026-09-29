"""The formatters, which pandas groups under `pandas.io.formats`.

`firepanda.io.formats.style` holds the Styler. It is not imported here, since
it imports jinja2 and a package import should not fail for want of an optional
dependency.
"""

from __future__ import annotations
