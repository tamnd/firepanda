#!/usr/bin/env python3
"""Compares two microbenchmark runs and fails the build on a regression.

`benchmarks/main.mojo` writes a result file. This reads two of them, lines the
benchmarks up by name, and decides whether anything got slower. It is the gate
docs/specs/09-quality-bar.md asks for on every pull request.

The decision is deliberately conservative, because a benchmark gate that cries
wolf gets disabled within a month and then nobody notices the real regression.
Two conditions have to hold before a change is called a regression:

- the median got worse by more than the threshold, and
- the change is larger than the noise in the measurement itself, taken as the
  wider of the two interquartile ranges.

The second condition is what a plain percentage comparison is missing. A
benchmark whose own spread across ten repetitions is eight percent cannot tell
you anything about a five percent change, and pretending otherwise produces a red
build with no bug behind it.

Both files must come from the same machine. Comparing a run on a laptop against a
run on a CI runner is meaningless, so a mismatch is refused rather than warned
about, unless `--allow-machine-mismatch` says the operator knows better.

Usage:
    python tools/bench_compare.py --baseline old.json --candidate new.json
    python tools/bench_compare.py --candidate new.json          # just print it
"""

from __future__ import annotations

import argparse
import json
import sys
from fnmatch import fnmatch
from pathlib import Path

# Anything slower than this fraction of the baseline is a regression, provided it
# also clears the noise floor below. Ten percent is loose enough to survive a CI
# runner and tight enough to catch an accidental copy in a hot loop.
DEFAULT_THRESHOLD = 0.10

# The smallest interquartile range we will believe. A run that reports two percent
# spread on a shared runner is lucky rather than precise, so the noise floor never
# drops below this no matter what the measurement claims.
MIN_NOISE = 0.03


def load(path: Path) -> dict:
    """Reads a result file.

    Args:
        path: The file.

    Returns:
        The parsed document.

    Raises:
        SystemExit: If the file is missing or not a result file.
    """
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"no such result file: {path}")
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path} is not valid JSON: {error}")
    if document.get("schema") != 1:
        raise SystemExit(f"{path} has schema {document.get('schema')}, expected 1")
    return document


def by_name(document: dict) -> dict[str, dict]:
    """Indexes a result file's benchmarks by name.

    Args:
        document: A parsed result file.

    Returns:
        A mapping from benchmark name to its record.
    """
    return {entry["name"]: entry for entry in document.get("benchmarks", [])}


def spread(entry: dict) -> float:
    """Returns a benchmark's interquartile range as a fraction of its median.

    Args:
        entry: A benchmark record.

    Returns:
        The spread, or zero if the median is zero.
    """
    median = entry.get("median_secs", 0.0)
    if median <= 0.0:
        return 0.0
    return (entry.get("q3_secs", median) - entry.get("q1_secs", median)) / median


def machine_key(document: dict) -> tuple:
    """Returns the part of the machine description that has to match.

    The label is excluded on purpose. Two runs on the same runner with different
    labels are still comparable, and refusing them because somebody typed the
    label differently helps nobody.

    Args:
        document: A parsed result file.

    Returns:
        The os, architecture and physical core count.
    """
    machine = document.get("machine", {})
    return (
        machine.get("os"),
        machine.get("arch"),
        machine.get("physical_cores"),
    )


def describe(document: dict) -> str:
    """Returns a one line description of where a run happened.

    Args:
        document: A parsed result file.

    Returns:
        The label, os, architecture and core count.
    """
    machine = document.get("machine", {})
    label = machine.get("label") or "unlabelled"
    return (
        f"{label} ({machine.get('os')}/{machine.get('arch')}, "
        f"{machine.get('physical_cores')} cores, "
        f"rows={document.get('config', {}).get('rows')})"
    )


def format_seconds(value: float) -> str:
    """Formats a duration in whichever unit keeps it readable.

    Args:
        value: The duration in seconds.

    Returns:
        The duration with its unit.
    """
    if value >= 1.0:
        return f"{value:.3f} s"
    if value >= 1e-3:
        return f"{value * 1e3:.3f} ms"
    if value >= 1e-6:
        return f"{value * 1e6:.3f} us"
    return f"{value * 1e9:.1f} ns"


def print_single(document: dict) -> int:
    """Prints one result file as a table.

    Args:
        document: A parsed result file.

    Returns:
        A process exit status.
    """
    print(describe(document))
    print()
    print(f"{'benchmark':<26}{'median':>14}{'iqr':>9}{'per item':>14}")
    for entry in document.get("benchmarks", []):
        print(
            f"{entry['name']:<26}"
            f"{format_seconds(entry['median_secs']):>14}"
            f"{spread(entry) * 100:>8.1f}%"
            f"{entry['per_item_ns']:>11.3f} ns"
        )
    return 0


def main() -> int:
    """Compares the two files and reports.

    Returns:
        A process exit status, non zero if anything regressed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        default="bench.json",
        help="the run being judged",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="the run to judge it against; without this the candidate is just printed",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="fraction slower that counts as a regression",
    )
    parser.add_argument(
        "--allow-machine-mismatch",
        action="store_true",
        help="compare runs from different machines anyway",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help=(
            "glob for benchmarks that exist to be compared against rather than "
            "to be defended, such as a standard library implementation. They are "
            "reported and never gate. Repeatable."
        ),
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="also write a markdown table here, for a pull request comment",
    )
    args = parser.parse_args()

    candidate = load(Path(args.candidate))
    if args.baseline is None:
        return print_single(candidate)

    baseline = load(Path(args.baseline))
    if machine_key(baseline) != machine_key(candidate):
        message = (
            f"machine mismatch: baseline is {describe(baseline)} "
            f"and candidate is {describe(candidate)}"
        )
        if not args.allow_machine_mismatch:
            raise SystemExit(message + "\npass --allow-machine-mismatch to override")
        sys.stderr.write(message + "\n")

    old = by_name(baseline)
    new = by_name(candidate)

    rows: list[tuple[str, str, str, float, str]] = []
    regressions: list[str] = []
    improvements: list[str] = []

    for name, entry in new.items():
        if name not in old:
            rows.append((name, "-", format_seconds(entry["median_secs"]), 0.0, "new"))
            continue
        before = old[name]["median_secs"]
        after = entry["median_secs"]
        if before <= 0.0:
            continue
        change = (after - before) / before
        noise = max(MIN_NOISE, spread(old[name]), spread(entry))
        # A reference row measures somebody else's code. It moving tells us
        # nothing about this change and everything about the machine, so it is
        # printed and then left alone.
        if any(fnmatch(name, pattern) for pattern in args.reference):
            rows.append(
                (name, format_seconds(before), format_seconds(after), change, "ref")
            )
            continue
        if change > args.threshold and change > noise:
            verdict = "REGRESSED"
            regressions.append(
                f"{name}: {format_seconds(before)} to {format_seconds(after)}, "
                f"{change * 100:+.1f}% against a noise floor of {noise * 100:.1f}%"
            )
        elif change < -args.threshold and -change > noise:
            verdict = "faster"
            improvements.append(f"{name}: {change * 100:+.1f}%")
        else:
            verdict = "ok"
        rows.append(
            (name, format_seconds(before), format_seconds(after), change, verdict)
        )

    missing = [name for name in old if name not in new]

    print(f"baseline  {describe(baseline)}")
    print(f"candidate {describe(candidate)}")
    print()
    print(f"{'benchmark':<26}{'baseline':>14}{'candidate':>14}{'change':>10}  verdict")
    for name, before_text, after_text, change, verdict in rows:
        change_text = "-" if verdict == "new" else f"{change * 100:+.1f}%"
        print(
            f"{name:<26}{before_text:>14}{after_text:>14}{change_text:>10}  {verdict}"
        )

    if missing:
        print()
        print("gone from the candidate: " + ", ".join(sorted(missing)))

    print()
    if regressions:
        print(f"{len(regressions)} regression(s):")
        for line in regressions:
            print("  " + line)
    else:
        print("no regressions")
    if improvements:
        print(f"{len(improvements)} improvement(s):")
        for line in improvements:
            print("  " + line)

    if args.markdown:
        lines = [
            "### Microbenchmarks",
            "",
            f"baseline `{describe(baseline)}`",
            "",
            "| benchmark | baseline | candidate | change | |",
            "| --- | --: | --: | --: | --- |",
        ]
        for name, before_text, after_text, change, verdict in rows:
            change_text = "-" if verdict == "new" else f"{change * 100:+.1f}%"
            mark = {
                "REGRESSED": "slower",
                "faster": "faster",
                "ok": "",
                "new": "new",
                "ref": "reference",
            }[verdict]
            lines.append(
                f"| `{name}` | {before_text} | {after_text} | {change_text} | {mark} |"
            )
        if missing:
            lines += ["", "Gone from the candidate: " + ", ".join(sorted(missing))]
        Path(args.markdown).write_text("\n".join(lines) + "\n")

    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
