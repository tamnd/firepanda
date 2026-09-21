"""What pandas answers for the methods it does not send to Arrow.

The oracle next door asks `contains`, `match` and `fullmatch`, and all three of
those go through `pyarrow.compute` and get RE2's reading of the pattern. This
one asks `findall`, which does not go near Arrow at all: pandas compiles the
pattern with `re` and loops in Python, and so do `extract` and `extractall`.
That is the whole reason this file exists rather than a `method` argument on the
other one, and document 81 has where the split was measured.

So the same accessor answers `count(r"\\w")` of 3 and `findall(r"\\w")` of 4 on a
row of `café`, and neither of those numbers is wrong. The pattern means two
different things depending on which method was called.

`findall` is asked rather than `extract`, for two reasons. It takes a pattern
with no groups in it, which most of the corpus is, where `extract` raises for
one. And its answer for a row is a list rather than a frame, so whether the
pattern matched at all is `len` of it, which is the same yes or no the other
oracle reports and lets the two reports be read side by side.

The texts are imported rather than written again, because comparing two engines
on two different lists of text compares nothing.

Answers come back one line per pattern so that a batch crosses the boundary
once.
"""

from __future__ import annotations

import warnings

import pandas as pd
from regex_texts import TEXTS


def _row(pattern: str) -> str:
    """What pandas answers for one pattern over every text.

    Returns:
        One character per text, `y` when the pattern was found somewhere in it
        and `n` when it was not, or a single `x` when the call raised.
    """
    column = pd.Series(TEXTS, dtype="str")
    try:
        found = column.str.findall(pattern)
    except Exception:
        return "x"
    return "".join("y" if len(value) > 0 else "n" for value in found)


def answers(patterns: list[str]) -> str:
    """What pandas answers for every pattern, in order.

    Args:
        patterns: The patterns, as the caller would have written them.

    Returns:
        One line per pattern, each either a single `x` or one character per
        text.
    """
    out = []
    with warnings.catch_warnings():
        # The corpus writes `[[:alpha:]]` and `[a--b]` on purpose, and Python
        # warns about both while reading them perfectly well.
        warnings.simplefilter("ignore")
        for pattern in patterns:
            out.append(_row(pattern))
    return "\n".join(out)
