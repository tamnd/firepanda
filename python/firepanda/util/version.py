"""Version strings read and ordered by PEP 440, which is `pandas.util.version`.

A version is an optional epoch, a dotted release, and optional pre, post, dev
and local parts. Two versions compare by a key in which a missing part sorts
where PEP 440 says it should, so `1.0.dev1 < 1.0a1 < 1.0 < 1.0.post1`.
"""

from __future__ import annotations

import functools
import re
from typing import Any

from ..errors import InvalidVersion

__all__ = ["VERSION_PATTERN", "InvalidVersion", "Version", "parse"]

VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""

_LETTERS = {"alpha": "a", "beta": "b", "c": "rc", "pre": "rc", "preview": "rc", "rev": "post"}
_LETTERS["r"] = "post"


@functools.total_ordering
class _Edge:
    """A bound that sorts above everything, or below everything when `low` is set."""

    def __init__(self, low: bool) -> None:
        self.low = low

    def __repr__(self) -> str:
        return "-Infinity" if self.low else "Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Edge) and other.low == self.low

    def __lt__(self, other: object) -> bool:
        return self.low and not self == other


_TOP = _Edge(low=False)
_BOTTOM = _Edge(low=True)


def _lettered(letter: str | None, number: str | None) -> tuple[str, int] | None:
    if letter:
        letter = letter.lower()
        return (_LETTERS.get(letter, letter), int(number or 0))
    if number:
        return ("post", int(number))
    return None


def parse(version: str) -> Version:
    """The version the string spells.

    Raises:
        InvalidVersion: When the string is not a PEP 440 version.
    """
    return Version(version)


class Version:
    """A PEP 440 version that prints in normal form and compares by PEP 440 order."""

    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)

    def __init__(self, version: str) -> None:
        found = self._regex.search(version)
        if not found:
            raise InvalidVersion(f"Invalid version: '{version}'")
        self.epoch: int = int(found.group("epoch") or 0)
        self.release: tuple[int, ...] = tuple(int(i) for i in found.group("release").split("."))
        self.pre: tuple[str, int] | None = _lettered(found.group("pre_l"), found.group("pre_n"))
        self._post = _lettered(
            found.group("post_l"), found.group("post_n1") or found.group("post_n2")
        )
        self._dev = _lettered(found.group("dev_l"), found.group("dev_n"))
        local = found.group("local")
        self._local = (
            tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"[\._-]", local))
            if local is not None
            else None
        )
        self._key = self._ordering()

    def _ordering(self) -> tuple[Any, ...]:
        release = list(self.release)
        while release and release[-1] == 0:
            release.pop()
        if self.pre is None and self._post is None and self._dev is not None:
            pre: Any = _BOTTOM
        else:
            pre = _TOP if self.pre is None else self.pre
        post = _BOTTOM if self._post is None else self._post
        dev = _TOP if self._dev is None else self._dev
        if self._local is None:
            local: Any = _BOTTOM
        else:
            local = tuple((p, "") if isinstance(p, int) else (_BOTTOM, p) for p in self._local)
        return (self.epoch, tuple(release), pre, post, dev, local)

    def __repr__(self) -> str:
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        text = self.base_version
        if self.pre is not None:
            text += "".join(str(x) for x in self.pre)
        if self.post is not None:
            text += f".post{self.post}"
        if self.dev is not None:
            text += f".dev{self.dev}"
        if self.local is not None:
            text += f"+{self.local}"
        return text

    def __hash__(self) -> int:
        return hash(self._key)

    def _compared(self, other: object, test: Any) -> Any:
        if not isinstance(other, Version):
            return NotImplemented
        return test(self._key, other._key)

    def __eq__(self, other: object) -> Any:
        return self._compared(other, lambda a, b: a == b)

    def __ne__(self, other: object) -> Any:
        return self._compared(other, lambda a, b: a != b)

    def __lt__(self, other: object) -> Any:
        return self._compared(other, lambda a, b: a < b)

    def __le__(self, other: object) -> Any:
        return self._compared(other, lambda a, b: a <= b)

    def __gt__(self, other: object) -> Any:
        return self._compared(other, lambda a, b: a > b)

    def __ge__(self, other: object) -> Any:
        return self._compared(other, lambda a, b: a >= b)

    @property
    def post(self) -> int | None:
        return self._post[1] if self._post else None

    @property
    def dev(self) -> int | None:
        return self._dev[1] if self._dev else None

    @property
    def local(self) -> str | None:
        return ".".join(str(x) for x in self._local) if self._local else None

    @property
    def public(self) -> str:
        return str(self).split("+", 1)[0]

    @property
    def base_version(self) -> str:
        epoch = f"{self.epoch}!" if self.epoch != 0 else ""
        return epoch + ".".join(str(x) for x in self.release)

    @property
    def is_prerelease(self) -> bool:
        return self.dev is not None or self.pre is not None

    @property
    def is_postrelease(self) -> bool:
        return self.post is not None

    @property
    def is_devrelease(self) -> bool:
        return self.dev is not None

    @property
    def major(self) -> int:
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    def minor(self) -> int:
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    def micro(self) -> int:
        return self.release[2] if len(self.release) >= 3 else 0
