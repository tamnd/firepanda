"""The text every pattern in the regular expression differentials is run
against.

One list rather than four. The four oracles beside this file ask pandas four
different questions about the same patterns, and a text that finds a mistake in
one of them is exactly the kind of text likely to find a mistake in another, so
a list written down four times would be four lists that drift and the one that
drifted would be the one nobody was looking at.

The first sixteen are handpicked and are the whole list this file used to be.
Each of them was chosen because a measured difference between the two engines
had something to bite on: the Arabic Indic digits are digits to Python and not
to RE2, the vertical tab and the non breaking space are whitespace to Python and
not to RE2, and the three with newlines are there for the dollar sign, which
matches before a final newline in Python and does not in RE2.

The rest are generated, and the reason they exist is that sixteen texts is a
sample rather than a corpus. Every disagreement the differentials have found so
far was found by a pattern, because the patterns are thirty thousand and the
texts were sixteen, and a reading that is wrong only on a text nobody picked is
a reading the differential calls right. The generator picks from an alphabet of
fragments chosen the same way the sixteen were, which is that each fragment is
something the two engines are known or suspected to disagree about, and it puts
between one and five of them together.

The fragments that matter most are the ones outside ASCII. RE2 runs on bytes and
Python runs on code points, so a word boundary can fall between the two bytes of
one character to one engine and nowhere at all to the other, and the only way to
see that is to put a character like that next to a letter and ask. The same goes
for folding, where the Kelvin sign folds to `k` and the dotless i does not fold
to `i`, and for the combining mark, which is a character to both engines and
half of a grapheme to neither.

Deterministic, because a differential whose input moves is one where a fix
cannot be told from a reshuffle. The generator is a few lines of arithmetic
rather than the standard library's, for the same reason the pattern corpus has
its own.
"""

from __future__ import annotations

HANDPICKED = [
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
"""The sixteen this list used to be, kept in the order they were in.

Kept verbatim and kept first, so that every answer the differentials have agreed
on so far is still being asked in the same place. A text corpus that replaced
these would be a different measurement wearing the same name.
"""

FRAGMENTS = [
    "a",
    "Z",
    "1",
    "_",
    " ",
    "-",
    ".",
    "\n",
    "\t",
    "\x0b",
    "\xa0",
    " ",
    "é",
    "Ω",
    "中",
    "𝔸",
    "٣",
    "д",
    "א",
    "ก",
    "あ",
    "한",
    "Ⅵ",
    "²",
    "́",
    "ß",
    "K",
    "İ",
    "ı",
    "ς",
    "Σ",
    "{",
    "}",
    "[",
    "]",
    ":",
    "(",
    ")",
    "*",
    "\\",
    "+",
    "|",
    "^",
    "$",
    "?",
]
"""What a generated text is built out of.

Four groups. The first eight are ASCII and are there so an ordinary pattern has
somewhere ordinary to match and so a word boundary has both of its sides. The
next four are whitespace the two engines disagree about, the vertical tab and
the non breaking space being whitespace to Python only and the line separator
being a line to neither engine's dollar sign.

The middle group is the one the generator exists for, and it is in two halves.
The first is about bytes: `é` and `Ω` and `٣` are two bytes, `中` and the Thai and
the Hiragana and the Hangul are three and `𝔸` is four, so any of them beside a
letter puts a byte boundary inside a character, and the combining mark puts a
code point inside what a reader would call one letter. The second is about
names: document 103 measured a table of 196 of them and said plainly that the
corpus exercised two of the 163 scripts in it, and these nine are Greek, Arabic,
Cyrillic, Hebrew, Thai, Hiragana, Hangul, Han and the mathematical alphabet,
with a Roman numeral and a superscript two beside them because those are letter
numbers and other numbers and are the categories a reader is most likely to get
wrong.

The five after them are the folding corners: the sharp s, the Kelvin sign which
folds to a plain `k`, the dotted and dotless i, and the two sigmas.

The last group is punctuation that means something in a pattern, which is there
because the pattern corpus writes a great many patterns that are read as literal
text and a literal brace has to be found in a row before anybody can say it was
read right.
"""

COUNT = 96
"""How many texts there are in all.

Ninety six rather than sixteen, and not very much more than that. Every one of
the six differentials runs every pattern against every text, so the cost is the
product, and at thirty thousand patterns the difference between a run somebody
waits for and a run somebody abandons is about this wide. Sixteen was too few to
catch anything the patterns did not catch first and a thousand would buy the
same coverage six times over, since the fragments are forty five and a text is
at most five of them.
"""

SEED = 0x9E3779B97F4A7C15
"""Where the generator starts.

A fixed number with no meaning beyond being well mixed. It is written down so
that a corpus can be regenerated exactly, which is the whole reason the
generator is not the standard library's.
"""


def _generated(count: int) -> list[str]:
    """The texts that are not handpicked.

    Args:
        count: How many to make.

    Returns:
        That many texts, each between one and five fragments long, with no two
        the same and none of them equal to one of the handpicked sixteen.
    """
    out: list[str] = []
    seen = set(HANDPICKED)
    state = SEED
    while len(out) < count:
        state = (state * 6364136223846793005 + 1442695040888963407) % (2**64)
        picked = state >> 11
        length = 1 + picked % 5
        picked //= 5
        text = ""
        for _ in range(length):
            text += FRAGMENTS[picked % len(FRAGMENTS)]
            picked //= len(FRAGMENTS)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


TEXTS = HANDPICKED + _generated(COUNT - len(HANDPICKED))
"""The corpus, handpicked first and generated after.

Built once at import so that every oracle that asks for it gets the same list in
the same order, which is what lets an answer be one character per text.
"""


def texts() -> list[str]:
    """The corpus, for a caller that is not a Python one.

    A function rather than the list itself, because the differentials reach this
    module across the Python boundary and a call is the one thing that boundary
    carries the same way every time.

    Returns:
        The texts, in the order the answers use.
    """
    return list(TEXTS)
