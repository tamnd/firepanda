"""What RE2 says each `\\p{...}` name covers, for the differential next door.

The other six oracles in this directory ask pandas, because the question they
are asking is what a caller gets, and what a caller gets is decided by pandas'
router before either engine sees the pattern. This one asks pyarrow directly,
because the question is narrower: not what a caller gets but whether the table
this library carries still says the same thing as the engine it was copied
from. pandas would add a router and an accessor to that and answer the same.

One call per name, over a column holding one code point per row. That is the
only way to get a set out of an engine that will only say yes or no about a
whole string, and it is the same shape `tools/gen_regexunicode.py` used to write
the table in the first place. The difference is what is in the column: the
generator asks about every code point there is and this asks about the ends of
the ranges it wrote down, which is a few thousand rather than a million, because
a range table goes wrong at its edges or not at all.

A name RE2 no longer reads comes back as a single `x`, which is a disagreement
rather than an error here: the table has the name, so RE2 having lost it is
exactly the kind of drift this is looking for.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc


def _row(name: str, points: list[int]) -> str:
    """Whether RE2 says the name covers each of the code points.

    Args:
        name: The name, as it is written between the braces.
        points: The code points to ask about.

    Returns:
        One `y` or `n` per code point, or a single `x` when RE2 would not read
        the name at all.
    """
    column = pa.array([chr(point) for point in points])
    try:
        found = pc.match_substring_regex(column, "\\p{%s}" % name).to_pylist()
    except Exception:
        return "x"
    return "".join("y" if held else "n" for held in found)


def answers(rows: list[str]) -> str:
    """What RE2 covers, for every name in order.

    Args:
        rows: One per name, each the name and then the code points to ask about
            in decimal, separated by spaces.

    Returns:
        One line per name, each either a single `x` or one letter per code
        point.
    """
    out = []
    for row in rows:
        pieces = row.split(" ")
        out.append(_row(pieces[0], [int(piece) for piece in pieces[1:]]))
    return "\n".join(out)
