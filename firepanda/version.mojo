"""Version identity.

The Mojo ABI is not stable across compiler releases, so a firepanda build is
identified by its own version and by the toolchain that produced it. Both are
reported by `firepanda.version()`.
"""

comptime VERSION = "0.0.0-dev"
"""The library version. Bumped by the release workflow, not by hand."""


def version() -> String:
    """Returns the library version.

    Returns:
        The version string.
    """
    return String(VERSION)
