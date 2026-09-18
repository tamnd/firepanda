"""Checks the version rules in the regular expression compiler against reality.

One of this library's two regular expression engines copies CPython's `re`, and
`re` is not the same module in every version of Python this project supports. Two
rules have already moved inside the supported range. `\\B` on an empty subject
failed up to 3.13 and matches from 3.14, and `\\z` was a bad escape up to 3.13
and is the end of the string from 3.14. Each of those is written down in
`firepanda/kernel/regex/program.mojo` as a `comptime PYTHON_<name>: Int`, and the
compiler compares the interpreter it was handed against one of them.

A rule copied off a running interpreter is a measurement with a date on it, and
the failure mode of a compiler holding one is quiet. Nothing raises when the
world moves past the measurement. The library keeps answering, with the newest
rules it knows, which are the rules of whatever version was current when somebody
last looked. Documents 90 and 91 both end by naming this script as the thing that
turns the next release into a red light on purpose rather than into one by luck.

Three things are checked and all three are about the constants rather than about
any pattern.

The running interpreter is not newer than `PYTHON_NEWEST`. That is the whole
point of the script. A newer one means a release has happened that nobody has
measured, and the answer is to run the corpus sweep across versions, write down
what moved, and raise the constant.

No rule constant is above `PYTHON_NEWEST`. A threshold above the newest measured
version is a rule nobody could have measured, so it is a typo or a guess.

No rule constant is at or below the floor `pixi.toml` declares. A threshold at or
below the oldest supported interpreter can never fire, because every supported
interpreter is already at or above it, so the branch behind it is dead and the
rule has retired. That is how these constants are meant to leave: the floor rises
and the rules that predate it come out. Nothing else will notice, which is why
this is a failure and not a note.

The floor is read from `pixi.toml` rather than spelled here, because the two
would drift and the file that declares which interpreters are supported is the
one that should say so.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROGRAM = ROOT / "firepanda" / "kernel" / "regex" / "program.mojo"
PIXI = ROOT / "pixi.toml"

NEWEST = "PYTHON_NEWEST"

CONSTANT = re.compile(r"^comptime (PYTHON_[A-Z_]+): Int = (\d+)$", re.MULTILINE)
FLOOR = re.compile(r'^python = ">=3\.(\d+)"$', re.MULTILINE)


def rules(text: str) -> dict[str, int]:
    """Every `PYTHON_` constant in the compiler, by name.

    Args:
        text: The compiler's source.

    Returns:
        The name of each constant against the minor version number it holds.
    """
    return {name: int(value) for name, value in CONSTANT.findall(text)}


def floor(text: str) -> int:
    """The oldest minor version the project says it supports.

    Args:
        text: The contents of `pixi.toml`.

    Returns:
        The minor version number.

    Raises:
        SystemExit: If the file does not declare one in the shape this reads.
    """
    found = FLOOR.search(text)
    if found is None:
        raise SystemExit(f'{PIXI} does not declare a python floor as `python = ">=3.N"`')
    return int(found.group(1))


def complaints(found: dict[str, int], oldest: int, running: int) -> list[str]:
    """Everything wrong with the constants, as sentences.

    Args:
        found: The constants, by name.
        oldest: The oldest supported minor version.
        running: The minor version of the interpreter asking.

    Returns:
        One sentence per problem, empty when there is nothing to say.
    """
    if NEWEST not in found:
        return [f"{PROGRAM} has no {NEWEST}, which is the constant everything here is about"]
    newest = found[NEWEST]
    said = []
    if running > newest:
        said.append(
            f"this interpreter is 3.{running} and {NEWEST} is {newest}, so a release has"
            " happened that nobody has measured. Run the corpus sweep across versions,"
            " write down what moved, add a constant for each rule that did, and raise"
            f" {NEWEST}. Documents 90 and 91 are what that looks like."
        )
    for name, value in sorted(found.items()):
        if name == NEWEST:
            continue
        if value > newest:
            said.append(
                f"{name} is {value} and {NEWEST} is {newest}, so it names a rule in a"
                " version nobody has measured"
            )
        if value <= oldest:
            said.append(
                f"{name} is {value} and the oldest supported interpreter is 3.{oldest},"
                " so the branch behind it can never run and the rule has retired. Delete"
                " the constant, delete the branch, and delete the rows that asked for it."
            )
    return said


def main() -> int:
    """Says what it found and whether it is a problem.

    Returns:
        The exit status, which is 0 when there is nothing to say.
    """
    found = rules(PROGRAM.read_text(encoding="utf-8"))
    oldest = floor(PIXI.read_text(encoding="utf-8"))
    running = sys.version_info.minor
    said = complaints(found, oldest, running)
    print(f"supported 3.{oldest} and up, running 3.{running}")
    for name, value in sorted(found.items()):
        print(f"  {name} = {value}")
    if not said:
        return 0
    for one in said:
        print(f"error: {one}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
