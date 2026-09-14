"""The two answers pandas gets out of a pattern, for the differential next door.

pandas asks exactly two questions of a pattern before it decides anything: can
Python's own parser read it, and does what comes back hold a lookaround or a
backreference where a partial walk can see it. The second question is asked by
`ArrowStringArrayMixin._has_unsupported_regex`, which is called here rather than
reimplemented, because a reimplementation of the oracle is not an oracle. It is
a second copy of the thing under test wearing the same name, and it agrees with
the first copy for exactly the reason that makes the agreement worthless.

The first question has to be asked separately because pandas folds it into the
second: a pattern Python cannot read answers False to `_has_unsupported_regex`,
which is the same False a plain pattern gives, and the two are different
situations with the same routing. `firepanda` tells them apart, so the
differential needs them told apart here too.

Answers come back one line per pattern, two characters each, so that a batch of
fifty thousand crosses the boundary once.
"""

from __future__ import annotations

import re
import warnings

from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin


def _read(pattern: str) -> str:
    """What happens when Python's parser is handed a pattern.

    Three answers rather than two. pandas catches `re.error` and nothing else,
    and there is at least one pattern that makes the parser raise something
    else: turning on both the ASCII flag and the Unicode flag globally is
    reported at the end of the parse as a `ValueError`, which goes straight out
    through `str.contains` with a message naming Python's parser. That is not a
    routing decision at all, so it is reported as its own answer rather than
    being squeezed into one of the two.

    Returns:
        `y` when it reads, `n` when it refuses with a parse error, and `x` when
        pandas does not survive the pattern.
    """
    try:
        re._parser.parse(pattern)  # type: ignore[attr-defined]
    except re.error:
        return "n"
    except Exception:
        return "x"
    return "y"


def answers(patterns: list[str]) -> str:
    """Both answers for every pattern, in order.

    Args:
        patterns: The patterns, as the caller would have written them.

    Returns:
        One line per pattern. The first character is `y` when Python's parser
        reads it, `n` when it refuses it and `x` when pandas does not survive
        it, and the second is `y` when pandas would route it to Python's `re`
        and `n` when it would route it to Arrow. The second character means
        nothing when the first is `x`.
    """
    out = []
    with warnings.catch_warnings():
        # The nested set warning fires for `[[:alpha:]]` and the set operation
        # one for `[a--b]`, both of which the corpus writes on purpose, and
        # neither says anything about whether the pattern parses.
        warnings.simplefilter("ignore")
        for pattern in patterns:
            first = _read(pattern)
            if first == "x":
                out.append("xn")
                continue
            second = ArrowStringArrayMixin._has_unsupported_regex(pattern)
            out.append(first + ("y" if second else "n"))
    return "\n".join(out)
