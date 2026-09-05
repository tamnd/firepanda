"""Does the built extension work somewhere that has never seen a Mojo toolchain.

`docs/specs/12-the-python-front-door-measured.md` section 2 says the answer is
yes, and says it on the strength of one afternoon on one laptop. This file is
that measurement turned into something that runs on every change, because the
ways it could stop being true are all quiet ones. A toolchain upgrade adds a
runtime library and the build script does not notice. A dependency starts being
named by absolute path and the copy nobody meant to ship comes along. An rpath
into the build environment survives and the extension keeps working on the
machine that built it, which is the only machine that will not catch it.

The check is a child interpreter with the environment taken away from it. Not a
subprocess with a tidied PATH, an interpreter with no pixi variables, no
DYLD_LIBRARY_PATH or LD_LIBRARY_PATH, and nothing on PATH but the system
directories, so that anything the extension manages to load it loaded from
beside itself. The child asserts `shutil.which("mojo") is None` before it does
anything else, because a test that silently ran inside the build environment
would pass and mean nothing.

These tests skip when there is no build to look at. Running them needs
`pixi run build-extension` first, and CI does exactly that.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from conftest import BUILT, REPO, stage

needs_a_build = pytest.mark.skipif(
    not (BUILT / "_firepanda.so").exists(),
    reason="no extension built, run `pixi run build-extension` first",
)

# What the whole self contained set weighs, with room above the measured figure
# but not room enough to hide a mistake. The number this guards against is not a
# gradual creep, it is the single wrong library: the smallest thing the build
# script could wrongly pick up is over a megabyte, and the first version of the
# script picked up exactly that, the C++ standard library, because it matched
# dependencies by file name and the toolchain names that one by absolute path.
# Vendoring a second libc++ into a process that already has one is worse than
# the wasted megabyte suggests.
SIZE_BUDGET = 8 * 1024 * 1024


def _stripped_environment() -> dict[str, str]:
    """The environment with everything that could be helping taken out of it."""
    keep = {"HOME", "TMPDIR", "SYSTEMROOT"}
    environment = {name: os.environ[name] for name in keep if name in os.environ}
    environment["PATH"] = "/usr/bin:/bin"
    return environment


CHILD = """
import shutil
assert shutil.which("mojo") is None, "a toolchain is on PATH, this proves nothing"
assert not any("pixi" in name.lower() for name in __import__("os").environ), "pixi is still here"
import firepanda
print(firepanda.__version__)
"""


@needs_a_build
def test_the_extension_imports_with_no_toolchain_anywhere(tmp_path: Path) -> None:
    stage(tmp_path)
    finished = subprocess.run(
        [sys.executable, "-c", CHILD],
        cwd=tmp_path,
        env=_stripped_environment(),
        capture_output=True,
        text=True,
    )
    # Report the status as well as the output. The interesting failure here does
    # not go through Python at all: on Apple silicon an edited load command
    # invalidates a binary's signature and the loader answers by killing the
    # process, so the whole of the evidence is a 137 and two empty streams.
    assert finished.returncode == 0, (
        f"exit {finished.returncode}\nstdout: {finished.stdout}\nstderr: {finished.stderr}"
    )

    metadata = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert finished.stdout.strip() == metadata["project"]["version"]


@needs_a_build
def test_the_build_directory_stays_the_size_it_was_measured_at() -> None:
    sizes = {path.name: path.stat().st_size for path in BUILT.iterdir() if path.is_file()}
    total = sum(sizes.values())
    assert total < SIZE_BUDGET, f"{total} bytes across {sorted(sizes)}"


@needs_a_build
@pytest.mark.skipif(sys.platform != "darwin", reason="signatures are a macOS concern")
def test_every_binary_is_still_validly_signed() -> None:
    """Because the alternative diagnosis is a bare 137 and two empty streams.

    The import test above catches this too, but it catches it as an exit status
    with nothing attached to it. This says which file and why.
    """
    for path in sorted(BUILT.iterdir()):
        if not path.is_file():
            continue
        finished = subprocess.run(
            ["codesign", "--verify", str(path)], capture_output=True, text=True
        )
        assert finished.returncode == 0, f"{path.name}: {finished.stderr.strip()}"


def _search_paths(binary: Path) -> list[str]:
    """Every directory the loader will look in for this binary's dependencies."""
    if sys.platform == "darwin":
        listing = subprocess.run(
            ["otool", "-l", str(binary)], capture_output=True, text=True, check=True
        ).stdout
        # `path` appears in an LC_RPATH and nowhere else; a dependency is a `name`.
        return [
            line.split()[1] for line in listing.splitlines() if line.strip().startswith("path ")
        ]
    written = subprocess.run(
        ["patchelf", "--print-rpath", str(binary)], capture_output=True, text=True, check=True
    ).stdout.strip()
    return [entry for entry in written.split(":") if entry]


@needs_a_build
def test_nothing_points_back_at_the_machine_that_built_it() -> None:
    """No search path is absolute, so none of them can name the build environment.

    This is the failure that running the thing cannot catch, because the machine
    that built it is the one machine where a path into the build environment
    still resolves.

    Relative is the whole test, rather than an exact expected list. The runtime
    libraries carry several `@loader_path` entries each, most of them Bazel
    leftovers naming directories that do not exist in a pixi environment, and
    they are harmless: a search path that resolves to nowhere costs a failed
    stat. Insisting on an exact set would mean editing those libraries to remove
    entries that do no harm, and on Apple silicon editing a library invalidates
    its signature, which is the one thing here that is genuinely fatal.
    """
    if sys.platform not in ("darwin", "linux"):
        pytest.skip(f"no load command reader wired up for {sys.platform}")

    for path in sorted(BUILT.iterdir()):
        if not path.is_file():
            continue
        absolute = [entry for entry in _search_paths(path) if entry.startswith("/")]
        assert not absolute, f"{path.name} still searches {absolute}"
