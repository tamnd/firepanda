"""What pandas answers when a pattern is actually run, for the differential next
door.

The other oracle in this directory asks which engine pandas picks. This one asks
what comes back once it has picked, which is a different question with different
failure modes: a pattern can be routed correctly and then answered wrongly, and
the only way to see that is to run it on text and compare.

Every call here goes through `Series.str.contains` on pandas' own string dtype
rather than through `pyarrow.compute` directly. Going through the accessor is
the point. pandas rewrites a trailing `\\Z` on the way past, decides for itself
whether Arrow gets the pattern at all, and turns Arrow's refusal into whatever
exception it turns it into, and a differential that skipped all that would be
testing RE2 rather than testing pandas.

The texts are here rather than in the caller because they have to be the same
texts on both sides and a list written down twice is a list that drifts.

Answers come back one line per pattern so that a batch crosses the boundary
once.
"""

from __future__ import annotations

import warnings

import pandas as pd

TEXTS = [
    "",
    "a",
    "b",
    "ab",
    "ba",
    "aab",
    "abc",
    "ABC",
    "a\n",
    "\na",
    "a\nb",
    "a1_ b2",
    "007",
    "٣٤",
    "héllo ΑΒΓ",
    "a\tb\x0bc\xa0d",
]
"""The text every pattern is run against.

Chosen so that each of the measured differences between the two engines has
something to bite on. The Arabic Indic digits are digits to Python's `\\d` and
not to RE2's. The last line holds a tab, which both engines call whitespace, and
a vertical tab and a non breaking space, which only Python does. The three with
newlines in them are there for `$`, which matches before a final newline in
Python and does not in RE2, and for the full stop, which excludes a newline in
both. The rest are ordinary enough that an ordinary pattern has somewhere to
match.
"""


def texts() -> list[str]:
    """The text the patterns are run against.

    Returns:
        The list, in the order the answers use.
    """
    return list(TEXTS)


def _row(pattern: str) -> str:
    """What pandas answers for one pattern over every text.

    Returns:
        One character per text, `y` for a match and `n` for none, or a single
        `x` when the call raised. The refusal is a property of the pattern
        rather than of a text, so it is one character rather than sixteen.
    """
    column = pd.Series(TEXTS, dtype="str")
    try:
        found = column.str.contains(pattern, regex=True)
    except Exception:
        return "x"
    return "".join("y" if bool(value) else "n" for value in found)


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
        # A pattern with a group in it warns that it has one, which is advice
        # for somebody who meant to call `extract` and says nothing about the
        # answer. The nested set and set operation warnings come from the
        # corpus writing `[[:alpha:]]` and `[a--b]` on purpose.
        warnings.simplefilter("ignore")
        for pattern in patterns:
            out.append(_row(pattern))
    return "\n".join(out)
