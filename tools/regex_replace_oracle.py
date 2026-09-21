"""What pandas writes out when a pattern is replaced, for the differential next
door.

The two oracles beside this one ask whether a pattern matches and how many times
it matches. This one asks what the row looks like afterwards, which is the first
of the three whose answer is text rather than a bit or a number, and that is why
it is a third file rather than a fourth sweep in one of the others.

Two things are being asked at once here and both of them are new. The scan is
not the scan that counts, in three separate ways that document 80 sets out, so
the same pattern over the same row can be right in the count oracle and wrong
here. And the replacement has a grammar, which is RE2's rewrite grammar rather
than Python's, so a replacement can be read wrongly by a library that ran the
pattern perfectly.

Three replacements are asked for every pattern, for three different reasons. A
plain marker says where the matches were and nothing else, which is the scan on
its own with the grammar kept out of the way. A marker holding the whole match
says how much of the row each match covered. And a bare group reference says
where the first group matched, which is the only one of the three that can fail
because of a capture slot, and is also the only one that a pattern without a
group is refused for, so the refusals are compared as well as the answers.

The text comes out as hexadecimal because it is text. A row can hold a newline,
a space or a tab, all three of which a reader would otherwise have to guess
about, and the answers have to survive being sent back across the boundary as
one string. Hexadecimal costs two characters a byte and settles all of it.

The texts are imported from the first oracle rather than written again, the same
way the count oracle imports them, so that a pattern named in one report can be
looked up in the others.

Answers come back one line per pattern so that a batch crosses the boundary
once.
"""

from __future__ import annotations

import warnings

import pandas as pd

from regex_texts import TEXTS

REPLACEMENTS = ("#", "[\\0]", "\\1")
"""The three replacements every pattern is asked for, in the order the answers
use.

The differential writes the same three down again on its side, because a list
sent across the boundary is a list the two sides can disagree about the meaning
of, and these three are short enough that writing them twice is safer than
agreeing about them once.
"""


def replacements() -> list[str]:
    """The replacements every pattern is asked for.

    Returns:
        The list, in the order the answers use.
    """
    return list(REPLACEMENTS)


def _sweep(pattern: str, repl: str) -> str:
    """What pandas writes out for one pattern and one replacement.

    Returns:
        One hexadecimal string per text separated by spaces, or a single `x`
        when the call raised, or a single `u` when it did not raise and the rows
        it wrote are not text. A refusal is a property of the pattern and the
        replacement together rather than of any one text, so it is one letter
        rather than one string per text.

        The third of those looks impossible and is not. RE2 reads a zero width
        assertion between bytes rather than between characters, so a pattern
        holding one can cut a character in half and write the halves out either
        side of the replacement. pyarrow catches most of those and raises, which
        is the `x` above, and the ones it does not catch arrive here as a column
        holding bytes that are not UTF-8 and blow up on the way into Python. The
        differential sets those aside and says how many, because there is no
        right answer on the other side to compare with.
    """
    column = pd.Series(TEXTS, dtype="str")
    try:
        written = column.str.replace(pattern, repl, regex=True)
    except Exception:
        return "x"
    try:
        return " ".join(str(value).encode().hex() for value in written)
    except Exception:
        return "u"


def _row(pattern: str) -> str:
    """Every replacement for one pattern.

    Returns:
        The three sweeps separated by vertical bars.
    """
    return "|".join(_sweep(pattern, repl) for repl in REPLACEMENTS)


def answers(patterns: list[str]) -> str:
    """What pandas writes out for every pattern, in order.

    Args:
        patterns: The patterns, as the caller would have written them.

    Returns:
        One line per pattern, each holding three sweeps separated by vertical
        bars.
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
