"""Whether RE2 reads a pattern, and what it says when it does not.

The other four oracles in this directory go through pandas' accessor, because
the question each of them asks is about what a caller of pandas gets back and
pandas' own routing is part of the answer. This one does not, and the reason is
worth writing down rather than discovering.

The question here is about RE2's grammar and nothing else. Going through the
accessor would put pandas' router in front of it, and the router sends a pattern
to Arrow only when Python's parser has already refused it, so the accessor can
never be asked about the patterns Python reads and RE2 does not. Half the
grammar would be unreachable. `pyarrow.compute.match_substring_regex` is the
same RE2, built into the same library, reached by the same call pandas makes
once it has decided, so asking it directly is asking the thing the accessor
would have asked with one fewer opinion in the way.

Nothing is matched. The array has one short row in it because the compute
function wants a column, and the row is never looked at: a pattern that compiles
answers and a pattern that does not raises, and which of the two happened is the
whole of what comes back.

RE2 has eleven ways of refusing and each of them is a sentence beginning with a
kind and then a colon and then the piece of the pattern it stopped at. The piece
is not useful here and is not comparable across two libraries that quote
different amounts of it, so only the kind comes back. A refusal with no colon in
it, which is the trailing backslash, comes back whole.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

COLUMN = pa.array(["a"])
"""One row, never read. The compute function wants a column and the pattern is
the only input that matters."""

PREFIX = "Invalid regular expression: "
"""What Arrow puts in front of RE2's own sentence."""


def answers(patterns: list[str]) -> str:
    """What RE2 says about every pattern.

    Args:
        patterns: The corpus.

    Returns:
        One line per pattern, each either `ok` or an `x` and a space and the
        kind of refusal.
    """
    out = []
    for pattern in patterns:
        out.append(asked(pattern))
    return "\n".join(out)


def asked(pattern: str) -> str:
    """What RE2 says about one pattern.

    Args:
        pattern: The pattern.

    Returns:
        `ok`, or an `x` and a space and the kind of refusal.
    """
    try:
        pc.match_substring_regex(COLUMN, pattern)
    except Exception as error:
        said = str(error).split("\n")[0]
        if said.startswith(PREFIX):
            said = said[len(PREFIX) :]
        return "x " + said.split(":")[0].strip()
    return "ok"
