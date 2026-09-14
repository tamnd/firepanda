"""Writes the one to one fold a case insensitive search compares through.

`str.casefold` and a case insensitive search do not fold the same way, and the
difference is not a detail. Folding a row is allowed to make it longer, so `ß`
folds to `ss` and `ﬁ` folds to `fi`. A search cannot afford that: pandas answers
`contains(pat, case=False)` out of `pyarrow.compute.match_substring` with
`ignore_case=True`, and that kernel folds each code point to exactly one code
point, so `STRASSE` does not contain `straße` and `FIANCE` does not contain
`ﬁance`. Both of those were measured rather than assumed.

So this library needs two fold tables and not one. `casefold.mojo` is the full
mapping, which is what `str.casefold` gives back and what a reader of that
method wants. This is the simple mapping, which is what all four of the case
insensitive searches compare through, and the two disagree on 104 code points.

The rule was recovered by measurement rather than read out of a Unicode data
file, because what this has to match is what pandas does and not what Unicode
recommends. It is: the full fold when the full fold is a single code point,
which covers 1426 of them; the lower case when the full fold is longer and the
lower case is a single code point that is not the character itself, which covers
28 more and is the branch that gets `ẞ` to `ß`; and the character itself
otherwise, which is the 76 that a search leaves alone. Three pairs sit outside
that rule and are listed by hand below.

The same table serves `str.replace(pat, repl, case=False)`, and that is worth
one sentence because it did not have to be true. pandas refuses `case=False` in
the Arrow path for `replace` and falls back to the object path, which escapes
the pattern and runs it with `re.IGNORECASE`, so a different implementation in a
different language decides the answer for that one method. Every pair of code
points that simple folding calls equal was checked against `re.IGNORECASE` and
every one of them agrees, and the three pairs the other way are the ones written
out below. One rule serves all four methods.

    uv run --no-project --python 3.13 --with pyarrow python tools/gen_searchfold.py

The generator verifies its own table against pyarrow rather than trusting the
rule it was derived from. Every entry is checked in both directions, and the
whole 1.1 million code point space is checked for a fold this table does not
know about, which takes a couple of minutes and is the reason this is committed
rather than built.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXTRA = {0x1FD3: 0x0390, 0x1FE3: 0x03B0, 0xFB06: 0xFB05}
"""The three pairs the rule below does not reach, found by measurement.

Two of them are Greek iota and upsilon with dialytika and tonos, which Unicode
encodes twice and whose full folds are both three code points long, and Unicode
gives the higher of each pair a simple fold to the lower one. The third is the
pair of `st` ligatures, which have no simple fold written down anywhere and
which pyarrow and `re.IGNORECASE` both fold together anyway. All three were
found by checking every pair inside a full fold class rather than by reading a
file, and all three are real: pandas answers True for each of them.
"""

HEADER = '''"""The one to one fold a case insensitive search compares through.

This is not `casefold.mojo` and the difference is the whole reason it exists.
Folding a row for a reader is allowed to make it longer, so `ß` folds to `ss`.
A search cannot do that, because a match would then cover a number of bytes
that has nothing to do with the number of bytes it was found in, and pandas
does not do it either: `contains(pat, case=False)` is answered by Arrow's
`match_substring` with `ignore_case=True`, which folds one code point to one
code point, and so `STRASSE` does not contain `straße` there.

{count} code points fold to something other than themselves under that rule and
every one of them folds to exactly one code point. The two tables are the code
points in order and their answers in the same order, which is the shape a binary
search reads, and the order is what makes the search legal.

The lowest code point here is the micro sign at U+00B5, whose lead byte is
0xC2, and 0xC2 is the lowest lead byte a non ASCII character can have at all,
so asking whether an element could hold one of these and asking whether it is
not ASCII are the same question. The ASCII half of the fold is the 26 upper
case letters and is not in this table, because a byte test is cheaper than a
binary search and the search kernels do that test first.

The same table serves all four of the case insensitive methods. Three of them
are Arrow's and the fourth, `replace`, is Python's `re.IGNORECASE`, because
pandas refuses `case=False` in its Arrow path for that one method and falls
back. The two rules were checked against each other over every pair of code
points inside a fold class and they agree, which did not have to be true and is
the reason `pattern.mojo` has one folded search rather than two.

Generated by tools/gen_searchfold.py against pyarrow {pyarrow} and Python
{python}, and verified against both rather than against the rule it was derived
from. Committed rather than built, because the check needs pyarrow and takes a
couple of minutes and the Mojo build has neither.
"""

'''


def table(name: str, values: list[int], doc: str) -> str:
    """One comptime array, a code point to a line.

    A line each rather than a row of eight because that is the shape the Mojo
    formatter puts a long literal in, and a generator whose output the formatter
    rewrites is a generator nobody can tell has been run.
    """
    body = "\n".join(f"    0x{v:04X}," for v in values)
    head = f"comptime {name}: InlineArray[UInt32, {len(values)}] = [\n{body}\n]\n"
    return head + f'"""{doc}"""\n\n'


def folded(cp: int) -> int:
    """The single code point a case insensitive search compares this one as.

    Args:
        cp: The code point.

    Returns:
        The code point it folds to, which is itself when nothing folds it.
    """
    if cp in EXTRA:
        return EXTRA[cp]
    one = chr(cp)
    full = one.casefold()
    if len(full) == 1:
        return ord(full)
    lower = one.lower()
    if len(lower) == 1 and lower != one:
        return ord(lower)
    return cp


def check_against_arrow(keys: list[int], answers: list[int]) -> None:
    """Asks pyarrow whether every entry in the table is real, both ways round.

    A table derived from a rule is only as good as the rule, and the rule here
    was recovered from measurements rather than read out of a file, so it gets
    measured back. Every pair is checked in both directions, because a fold that
    held one way and not the other would be a fold nobody could use.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    rows = pa.chunked_array([[chr(key) for key in keys]])
    needles = [chr(answer) for answer in answers]
    for key, answer in zip(keys, answers, strict=True):
        one = pa.chunked_array([[chr(key)]])
        other = pa.chunked_array([[chr(answer)]])
        if not pc.match_substring(one, chr(answer), ignore_case=True)[0].as_py():
            raise SystemExit(f"U+{key:04X} does not fold to U+{answer:04X} in arrow")
        if not pc.match_substring(other, chr(key), ignore_case=True)[0].as_py():
            raise SystemExit(f"U+{answer:04X} does not fold to U+{key:04X} in arrow")
    del rows, needles


def check_no_fold_is_missing(points: list[int], keys: list[int], answers: list[int]) -> None:
    """Asks pyarrow whether any code point folds in a way this table does not know.

    The check above proves every entry is real and this one proves the table is
    complete. For each code point something folds to, it asks pyarrow which of
    the 1.1 million code points match it, and refuses to write anything unless
    that set is exactly the set this table says it should be. It is slow and it
    is the reason the file is committed.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    table_of = dict(zip(keys, answers, strict=True))
    classes: dict[int, set[int]] = {}
    for point in points:
        classes.setdefault(table_of.get(point, point), set()).add(point)
    every = pa.chunked_array([[chr(point) for point in points]])
    for target in sorted(set(answers)):
        got = pc.match_substring(every, chr(target), ignore_case=True).to_pylist()
        found = {points[i] for i, hit in enumerate(got) if hit}
        want = classes[target]
        if found != want:
            extra = sorted(found - want)[:4]
            missing = sorted(want - found)[:4]
            raise SystemExit(
                f"U+{target:04X} folds a different set in arrow,"
                f" extra {[hex(c) for c in extra]} missing {[hex(c) for c in missing]}"
            )


def check_against_python(keys: list[int], answers: list[int]) -> None:
    """Asks Python whether `re.IGNORECASE` folds the same way pyarrow does.

    `replace` is the one of the four methods pandas does not answer out of
    Arrow, so its rule comes from a different implementation in a different
    language and there is no reason in principle for the two to agree. They do,
    on every entry here, which is what lets one table serve all four.
    """
    import re

    for key, answer in zip(keys, answers, strict=True):
        if not re.fullmatch(re.escape(chr(key)), chr(answer), re.IGNORECASE):
            raise SystemExit(f"re.IGNORECASE does not fold U+{key:04X} to U+{answer:04X}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("firepanda/kernel/searchfold.mojo"),
        help="where to write the table",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip the sweep over every code point, which is the slow check",
    )
    args = parser.parse_args()

    import pyarrow as pa

    points = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    keys: list[int] = []
    answers: list[int] = []
    for point in points:
        answer = folded(point)
        if answer != point:
            keys.append(point)
            answers.append(answer)

    check_against_arrow(keys, answers)
    check_against_python(keys, answers)
    if not args.quick:
        check_no_fold_is_missing(points, keys, answers)

    body = HEADER.format(
        count=len(keys),
        pyarrow=pa.__version__,
        python=".".join(str(part) for part in sys.version_info[:3]),
    )
    body += table(
        "SEARCHED_FROM",
        keys,
        "The code points a case insensitive search does not compare as themselves.",
    )
    body += table(
        "SEARCHED_TO",
        answers,
        "What each one is compared as, in the order `SEARCHED_FROM` holds them.",
    )
    args.out.write_text(body)
    print(f"{len(keys)} folds, {len(set(answers))} distinct answers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
