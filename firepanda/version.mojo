"""Version identity.

The Mojo ABI is not stable across compiler releases, so a firepanda build is
identified by its own version and by the toolchain that produced it. Both are
reported by `firepanda.version()`.
"""

comptime VERSION = "0.6.45"
"""The library version.

Bumped in the release pull request, alongside `pixi.toml` and `pyproject.toml`,
because nothing in the build derives it from the tag and three copies that can
disagree are worse than three copies that change together.
"""


def version() -> String:
    """Returns the library version.

    Returns:
        The version string.
    """
    return String(VERSION)
