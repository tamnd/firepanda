#!/usr/bin/env python3
"""Measures what the monomorphization strategy costs in compile time and bytes.

Every operation in firepanda is written once and compiled once per dtype in the
list it is dispatched over, which is the whole speed argument and also the whole
code size risk. docs/specs/03-dtype-dispatch.md puts a number on the risk: an
engine with a few hundred operations over a dozen dtypes is a few thousand
instantiations, and nobody notices that going wrong until a build takes twenty
minutes.

So the numbers are collected from the first commit and graphed, not thresholded.
A threshold now would be a guess. A graph now means that when the threshold does
land at M8 it lands on evidence.

What is measured:

- how long `mojo precompile` takes on the package, and how big the artifact is;
- three probe programs under tools/probes, each linking the package and doing a
  different amount of dispatch, built to real executables and stripped.

The probes are the interesting part. `baseline` dispatches over nothing,
`dispatch_signed` over four dtypes and `dispatch_numeric` over eleven, with the
same kernel in both dispatching probes, so the difference between the last two is
seven instantiations of one kernel and nothing else.

Usage:
    python tools/compile_budget.py [--output compile-budget.json] [--repeat N]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBES = ["baseline", "dispatch_signed", "dispatch_numeric"]

# The two dispatching probes differ only in the length of the dtype list.
SPREAD_PROBES = ("dispatch_signed", "dispatch_numeric")
SPREAD_DTYPES = 11 - 4


def run(command: list[str], cwd: Path) -> float:
    """Runs a command and returns how long it took in seconds.

    Args:
        command: The command and its arguments.
        cwd: The working directory.

    Returns:
        The elapsed wall clock time.

    Raises:
        SystemExit: If the command fails, with its output on stderr.
    """
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        sys.stderr.write(f"command failed: {' '.join(command)}\n")
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return elapsed


def best_of(command: list[str], cwd: Path, repeat: int) -> float:
    """Runs a command several times and keeps the fastest run.

    Compile time on a shared CI runner is noisy in one direction only: something
    else on the machine can steal time, nothing can give it back. The minimum is
    therefore a better estimate of the compiler's cost than the mean.

    There is also a first-invocation cost that is not compile time at all. The
    very first `mojo` on a machine pages the toolchain and the standard library in
    off disk, which measured close to a second on the development box while every
    later build of the same file took half that. Two runs is enough to leave that
    behind, which is why the default repeat is two rather than one.

    Args:
        command: The command and its arguments.
        cwd: The working directory.
        repeat: How many times to run it.

    Returns:
        The fastest elapsed time in seconds.
    """
    return min(run(command, cwd) for _ in range(repeat))


def strip_size(binary: Path) -> int | None:
    """Returns the size of a stripped copy of a binary, if strip is available.

    Args:
        binary: The executable to measure.

    Returns:
        The stripped size in bytes, or None if no strip tool was found.
    """
    tool = shutil.which("llvm-strip") or shutil.which("strip")
    if tool is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / binary.name
        shutil.copy2(binary, copy)
        result = subprocess.run(
            [tool, str(copy)], capture_output=True, text=True
        )
        if result.returncode != 0:
            return None
        return copy.stat().st_size


def toolchain_version() -> str:
    """Returns the mojo version string.

    Returns:
        Whatever `mojo --version` prints, on one line.
    """
    result = subprocess.run(
        ["mojo", "--version"], capture_output=True, text=True, cwd=ROOT
    )
    return result.stdout.strip() or "unknown"


def main() -> int:
    """Collects the measurements and writes them out.

    Returns:
        A process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help="write the measurements to this file as JSON as well as to stdout",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="build each target this many times and keep the fastest",
    )
    args = parser.parse_args()

    build_dir = ROOT / "build"
    build_dir.mkdir(exist_ok=True)

    package = build_dir / "firepanda.mojoc"
    package.unlink(missing_ok=True)
    package_seconds = best_of(
        ["mojo", "precompile", "firepanda", "-o", str(package)], ROOT, args.repeat
    )

    report: dict[str, object] = {
        "schema": 1,
        "toolchain": toolchain_version(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "cpus": os.cpu_count(),
        },
        "package": {
            "seconds": round(package_seconds, 3),
            "bytes": package.stat().st_size,
        },
        "probes": {},
    }

    probes: dict[str, dict[str, object]] = {}
    for name in PROBES:
        source = ROOT / "tools" / "probes" / f"{name}.mojo"
        binary = build_dir / name
        binary.unlink(missing_ok=True)
        seconds = best_of(
            ["mojo", "build", "-I", ".", str(source), "-o", str(binary)],
            ROOT,
            args.repeat,
        )
        probes[name] = {
            "seconds": round(seconds, 3),
            "bytes": binary.stat().st_size,
            "stripped_bytes": strip_size(binary),
        }
    report["probes"] = probes

    baseline = probes["baseline"]
    for name in PROBES:
        if name == "baseline":
            continue
        probes[name]["bytes_over_baseline"] = (
            probes[name]["bytes"] - baseline["bytes"]
        )
        probes[name]["seconds_over_baseline"] = round(
            probes[name]["seconds"] - baseline["seconds"], 3
        )

    narrow, wide = SPREAD_PROBES
    report["per_dtype"] = {
        "measured_between": list(SPREAD_PROBES),
        "extra_dtypes": SPREAD_DTYPES,
        "bytes": round(
            (probes[wide]["bytes"] - probes[narrow]["bytes"]) / SPREAD_DTYPES, 1
        ),
        "seconds": round(
            (probes[wide]["seconds"] - probes[narrow]["seconds"]) / SPREAD_DTYPES,
            4,
        ),
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
