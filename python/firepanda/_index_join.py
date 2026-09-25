"""The row pairing behind `Index.join`, written over plain lists.

pandas answers a join of two indexes along one of three paths, and which one it
takes decides more than speed. Two sorted indexes where one side is unique go
through the sorted join and answer `None` for a side that comes back in its
own order. Indexes with repeated labels go through the merge join, which
counts the labels into groups and pairs the rows of each group. Everything
else looks every label up in the other index. The paths answer the same rows
but not always in the same order, and code written against pandas can see the
difference, so each one is kept.

The merge join is pandas' `libjoin` read line by line, including one shortcut
in the unsorted inner join that assumes a left row matched at most once
whenever the answer has as many rows as the left side. That is not always
true, and when it is not, the rows come back in an order that looks random but
is exactly pandas' order. The rules below were measured against pandas 3.0.
"""

from __future__ import annotations

import itertools
from typing import Any

__all__ = ["increasing", "keyed", "merged", "missing", "ordered", "unique"]

_GAP = object()
"""The one key every missing label shares, so a gap joins a gap."""


def missing(value: Any) -> bool:
    """Whether a label is a gap, which is None or a NaN."""
    return value is None or (isinstance(value, float) and value != value)


def keyed(value: Any) -> Any:
    """A label as a dictionary key, with every gap made the same key."""
    return _GAP if missing(value) else value


def unique(values: list[Any]) -> bool:
    """Whether no label repeats, a gap counted as one label."""
    return len({keyed(value) for value in values}) == len(values)


def increasing(values: list[Any]) -> bool:
    """Whether the labels never go down, which a gap never is, as in pandas."""
    if any(missing(value) for value in values):
        return False
    try:
        return all(a <= b for a, b in itertools.pairwise(values))
    except TypeError:
        return False


def ordered(values: list[Any]) -> list[int]:
    """The positions that sort the labels, stably, with the gaps at the end."""
    return sorted(
        range(len(values)),
        key=lambda at: (missing(values[at]), None if missing(values[at]) else values[at]),
    )


def _labels(
    left: list[Any], right: list[Any], sort: bool, right_first: bool
) -> tuple[list[int], list[int], int]:
    """Both sides counted into one set of groups, the gaps in a group of their own.

    The groups are numbered in the order the labels are first seen, reading
    the right side first when `right_first` is set, which is what pandas' inner
    join of numbers does. With `sort` the groups are renumbered in label
    order. The gap group comes last whichever way.
    """
    codes: dict[Any, int] = {}
    seen: list[Any] = []

    def label(values: list[Any]) -> list[int]:
        found = []
        for value in values:
            if missing(value):
                found.append(-1)
                continue
            if value not in codes:
                codes[value] = len(seen)
                seen.append(value)
            found.append(codes[value])
        return found

    if right_first:
        rlab = label(right)
        llab = label(left)
    else:
        llab = label(left)
        rlab = label(right)
    count = len(seen)
    if sort:
        try:
            rank = {old: new for new, old in enumerate(sorted(range(count), key=seen.__getitem__))}
        except TypeError:
            rank = {old: old for old in range(count)}
        llab = [rank.get(code, -1) for code in llab]
        rlab = [rank.get(code, -1) for code in rlab]
    if -1 in llab or -1 in rlab:
        llab = [count if code == -1 else code for code in llab]
        rlab = [count if code == -1 else code for code in rlab]
        count += 1
    return llab, rlab, count


def _grouped(labels: list[int], groups: int) -> tuple[list[int], list[int]]:
    """The positions sorted stably by group, and how many rows each group holds."""
    counts = [0] * groups
    for label in labels:
        counts[label] += 1
    starts = [0] * groups
    for group in range(1, groups):
        starts[group] = starts[group - 1] + counts[group - 1]
    sorter = [0] * len(labels)
    for at, label in enumerate(labels):
        sorter[starts[label]] = at
        starts[label] += 1
    return sorter, counts


def _paired(
    left: list[int], right: list[int], groups: int, how: str
) -> tuple[list[int], list[int], list[int]]:
    """Every pair of rows that share a group, in group order.

    Answers the left positions, the right positions and the left sorter,
    which the unsorted joins need to put the rows back.
    """
    lsort, lcount = _grouped(left, groups)
    rsort, rcount = _grouped(right, groups)
    lidx: list[int] = []
    ridx: list[int] = []
    lpos = rpos = 0
    for group in range(groups):
        lc, rc = lcount[group], rcount[group]
        if lc and rc:
            for j in range(lc):
                for k in range(rc):
                    lidx.append(lsort[lpos + j])
                    ridx.append(rsort[rpos + k])
        elif lc and how in ("left", "outer"):
            lidx.extend(lsort[lpos : lpos + lc])
            ridx.extend([-1] * lc)
        elif rc and how == "outer":
            lidx.extend([-1] * rc)
            ridx.extend(rsort[rpos : rpos + rc])
        lpos += lc
        rpos += rc
    return lidx, ridx, lsort


def _reverted(lidx: list[int], ridx: list[int], lsort: list[int], size: int) -> tuple[list, list]:
    """The pairs put back in the left side's order, the way pandas does it."""
    if len(lsort) == len(lidx):
        back = [0] * size
        for at, row in enumerate(lsort):
            back[row] = at
    else:
        back, _ = _grouped(lidx, size)
    return [lidx[at] for at in back], [ridx[at] for at in back]


def merged(
    left: list[Any], right: list[Any], how: str, sort: bool, numeric: bool
) -> tuple[list[int], list[int]]:
    """The left and right positions of pandas' merge join of two lists of labels.

    Args:
        left: The left labels.
        right: The right labels.
        how: One of left, right, inner and outer.
        sort: Whether the answer comes in label order.
        numeric: Whether the labels are numbers, which changes the group order
            of an unsorted inner join.
    """
    if how == "right":
        ridx, lidx = merged(right, left, "left", sort, numeric)
        return lidx, ridx
    llab, rlab, groups = _labels(left, right, sort, how == "inner" and not sort and numeric)
    lidx, ridx, lsort = _paired(llab, rlab, groups, how)
    if how != "outer" and not sort:
        return _reverted(lidx, ridx, lsort, len(left))
    return lidx, ridx
