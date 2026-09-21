"""How many times pandas says a pattern matches, for the differential next door.

The oracle beside this one asks whether a pattern matches. This one asks how
many times, which sounds like the same question asked again and is not. Whether
is one bit and stops at the first match. How many is a loop, and the loop is
Arrow's rather than either engine's: it runs the pattern, counts the match, cuts
the text at a position it works out from where the match ended, and runs the
pattern again on what is left. Document 79 has the three rules that come out of
that and why none of them is what a reader would guess.

So this file exists because the answer has a different shape. A count is a
number per text rather than a bit, the numbers can be larger than the text is
long once a pattern matches nothing at all, and comparing them as `y` and `n`
would hide every one of the three rules.

The texts are imported from the other oracle rather than written again. They
were chosen for what the two engines disagree about when matching, and every one
of those disagreements still bites when counting, plus two more: the row with
characters wider than a byte, because the loop steps in bytes, and the empty
row, because a pattern that can match nothing matches it once.

Answers come back one line per pattern so that a batch crosses the boundary
once.
"""

from __future__ import annotations

import warnings

import pandas as pd

from regex_texts import TEXTS


def _row(pattern: str) -> str:
    """How many times pandas says a pattern matches in every text.

    Returns:
        One number per text separated by spaces, or a single `x` when the call
        raised. The refusal is a property of the pattern rather than of a text,
        so it is one letter rather than one number per text.
    """
    column = pd.Series(TEXTS, dtype="str")
    try:
        found = column.str.count(pattern)
    except Exception:
        return "x"
    return " ".join(str(int(value)) for value in found)


def answers(patterns: list[str]) -> str:
    """How many times every pattern matches, in order.

    Args:
        patterns: The patterns, as the caller would have written them.

    Returns:
        One line per pattern, each either a single `x` or one number per text.
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
