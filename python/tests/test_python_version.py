"""The version rules in the compiler, checked against the interpreter running.

One of this library's two regular expression engines copies CPython's `re`, and
two of `re`'s rules have moved inside the range of interpreters this project
supports. Each one is a `comptime PYTHON_<name>: Int` in the compiler, and the
compiler compares the interpreter it was handed against it.

A constant like that fails quietly. When a release moves a rule, nothing raises.
The library keeps answering, with the rules of whatever version was current when
somebody last looked, and the wrong answers are ordinary looking columns.
Documents 90 and 91 each end by saying that the next release has to be a red
light on purpose rather than one by luck, and `tools/python_version.py` is that
light.

The tool is wired into CI under the pixi interpreter, which is the one the
differential sweeps measure against. This file runs the same check under the
interpreter the accessor suite is using, which is a different one and is the
whole reason document 90 exists: those two suites have been standing in two
different Pythons all along and neither of them said so. A contributor who
upgrades one of them finds out here.

The rows below are about the checking rather than about this run, apart from the
last one. A check that only ever sees a healthy repository is a check nobody
knows the failure message of, so each way of being wrong is asked for directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


def tool() -> ModuleType:
    """The checker, loaded from `tools/` by path.

    `tools/` is not a package and is not on the path, and putting it there for
    one test would change what every other test in this directory can import.

    Returns:
        The module.
    """
    where = ROOT / "tools" / "python_version.py"
    spec = importlib.util.spec_from_file_location("python_version", where)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_constants_are_read_out_of_the_compiler() -> None:
    """Read rather than spelled, because a copy of a number in a second file is a
    number that goes stale. The names are not asserted one by one for the same
    reason: this file would have to be edited every time a rule lands or
    retires, and the whole point of the check is that it needs no maintenance
    between releases."""
    found = tool().rules((ROOT / "firepanda" / "kernel" / "regex" / "program.mojo").read_text())
    assert "PYTHON_NEWEST" in found
    assert all(isinstance(value, int) for value in found.values())
    assert len(found) > 1


def test_the_floor_is_read_out_of_the_file_that_declares_it() -> None:
    """`pixi.toml` says which interpreters this project supports, so it is what
    the check reads. Spelling the floor in the checker would mean two places to
    change and one of them silently wrong."""
    assert tool().floor((ROOT / "pixi.toml").read_text()) >= 12


def test_an_interpreter_newer_than_the_newest_measured_one_is_the_red_light() -> None:
    """The row the whole thing is for. A release has happened that nobody has
    measured, the library is compiling with the rules of an older one, and
    nothing else anywhere will say so."""
    said = tool().complaints({"PYTHON_NEWEST": 14, "PYTHON_ZED_ESCAPE": 14}, 12, 15)
    assert len(said) == 1
    assert "nobody has measured" in said[0]


def test_a_rule_nobody_could_have_measured_is_refused() -> None:
    """A threshold above the newest measured version names a rule in a version
    that has not been looked at, so it is a guess or a typo. Cheap to check and
    it costs nothing to keep."""
    said = tool().complaints({"PYTHON_NEWEST": 14, "PYTHON_ZED_ESCAPE": 16}, 12, 13)
    assert len(said) == 1
    assert "PYTHON_ZED_ESCAPE" in said[0]


def test_a_rule_that_can_never_fire_again_is_refused() -> None:
    """How these constants are meant to leave. When the floor rises past a
    threshold, every supported interpreter is already above it, the branch behind
    it is dead, and the rule has retired. Nothing else would ever notice, so this
    is a failure rather than a note."""
    said = tool().complaints({"PYTHON_NEWEST": 14, "PYTHON_ZED_ESCAPE": 12}, 12, 13)
    assert len(said) == 1
    assert "retired" in said[0]


def test_a_compiler_with_no_newest_constant_at_all_is_refused() -> None:
    """The one failure the rest of the check cannot be run without, so it is
    answered on its own and the others are not asked."""
    said = tool().complaints({"PYTHON_ZED_ESCAPE": 14}, 12, 13)
    assert len(said) == 1
    assert "PYTHON_NEWEST" in said[0]


def test_this_interpreter_is_one_the_rules_were_measured_against() -> None:
    """The row about this run. If it is red, read the message rather than this
    docstring: it says what to measure and where the last two of these were
    written down."""
    module = tool()
    found = module.rules((ROOT / "firepanda" / "kernel" / "regex" / "program.mojo").read_text())
    oldest = module.floor((ROOT / "pixi.toml").read_text())
    said = module.complaints(found, oldest, sys.version_info.minor)
    if said:
        pytest.fail("\n".join(said))
