"""Writes the tables the four normalization forms are built out of.

`str.normalize` is the one name on the accessor that is a question about what a
character is equivalent to rather than about what it is or what it maps to.
Unicode says the same text can be spelled several ways, an e with an acute
accent being one character or two, and normalization is the operation that picks
one spelling so that two strings meaning the same thing compare equal.

pandas answers this out of `unicodedata.normalize` row by row on every backend,
so CPython is the authority here and not Arrow. That is the opposite of the case
classes, where Arrow is the authority because pandas asks Arrow. It is worth
being explicit about which way round it is for each name rather than having a
house rule, because the two names sit next to each other in the same accessor and
are answered by different libraries.

The four forms are two decompositions and a composition on top of each. NFD
replaces every character by its canonical decomposition and then puts the
combining marks in canonical order. NFC does that and then puts back together
every pair that can be put back together. NFKD and NFKC are the same two with the
compatibility decompositions as well, which is the wider relation that turns a
ligature into its letters and a circled digit into a digit.

Four tables come out of here.

The two decomposition tables are fully expanded at generation time, so a lookup
answers the final sequence and the kernel never has to recurse. That costs
nothing in space worth counting, 3406 code points for the canonical table and
9112 for the compatibility one, and it removes the one part of the algorithm
where a depth limit would otherwise have to be argued about.

The combining class table is 922 entries and is a plain sorted key and value
pair, not ranges, because combining marks are scattered rather than blocked and
the runs would be longer than the list.

The composition table is derived rather than read. Unicode excludes some pairs
from recomposition, through the composition exclusion list, through singleton
decompositions and through non starter decompositions, and the rules are fiddly
enough that reading the list and applying the rules is a second implementation to
get wrong. Instead every canonical pair is tried: decompose it, ask CPython to
compose it back, and keep the pair only if CPython agrees. What comes out is by
construction the set of pairs CPython will compose, which is the only set that
matters here.

    uv run --no-project --python 3.13 python tools/gen_normalize.py

Before it writes anything it runs the whole algorithm in Python out of nothing
but the tables it is about to emit, over every code point on its own and over a
hundred thousand strings built to be awkward, and refuses to write if CPython
disagrees about any of them. The awkward strings matter more than the code
points: normalization is a rule about sequences, and every interesting bug in it
is a bug about what happens when two combining marks of the same class meet, or
when a decomposition ends in a mark that the next character's decomposition also
starts with.
"""

from __future__ import annotations

import argparse
import random
import unicodedata
from pathlib import Path

MAX = 0x110000

SBASE, LBASE, VBASE, TBASE = 0xAC00, 0x1100, 0x1161, 0x11A7
LCOUNT, VCOUNT, TCOUNT = 19, 21, 28
NCOUNT = VCOUNT * TCOUNT
SCOUNT = LCOUNT * NCOUNT

HEADER = '''"""The tables the four normalization forms are built out of.

Unicode lets the same text be spelled more than one way. An e with an acute
accent is one character or it is two, a ligature is one character or it is the
letters it is made of, and two strings that mean the same thing do not have to
be the same bytes. Normalization picks one spelling so that they do.

These tables are CPython's, written by tools/gen_normalize.py out of
`unicodedata`, because pandas answers `str.normalize` out of
`unicodedata.normalize` row by row on every one of its backends. That is the
other way round from the case classes next door, which are Arrow's because
pandas asks Arrow for those. Which library is the authority is decided per
name by which one pandas actually calls, rather than by a house rule.

The two decomposition tables are fully expanded, so one lookup gives the final
sequence and nothing here recurses. The canonical table is {canon_n} characters
coming to {canon_v} code points, and the compatibility table is {compat_n}
characters coming to {compat_v}. Both are sorted by code point and searched by
bisection, with a start offset per entry into one flat run of values.

The combining class table is {ccc_n} entries held as a sorted key list and a
value list rather than as ranges, because combining marks are scattered and the
ranges would outnumber the entries.

The composition table is {comp_n} pairs, held as one sorted list of a starter
and a following character packed into a single number, and a list of what the
pair composes to. It was derived by trying every canonical pair and keeping the
ones CPython actually composes, rather than by reading the exclusion list and
reimplementing the three rules that feed it.

Hangul is not in any of these. Its decomposition and its composition are
arithmetic on the code point, so the syllables are handled by the formulas in
`normalize.mojo` and cost no table at all, which is {hangul} characters not
stored.
"""

'''


def decompositions() -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, int]]:
    """The raw canonical and compatibility mappings and the combining classes."""
    canon: dict[int, list[int]] = {}
    compat: dict[int, list[int]] = {}
    ccc: dict[int, int] = {}
    for cp in range(MAX):
        ch = chr(cp)
        klass = unicodedata.combining(ch)
        if klass:
            ccc[cp] = klass
        written = unicodedata.decomposition(ch)
        if not written:
            continue
        if written.startswith("<"):
            compat[cp] = [int(x, 16) for x in written.split(">", 1)[1].split()]
        else:
            canon[cp] = [int(x, 16) for x in written.split()]
    return canon, compat, ccc


def expanded(
    canon: dict[int, list[int]], compat: dict[int, list[int]], full: bool
) -> dict[int, list[int]]:
    """Every mapping applied until nothing maps any more.

    Doing this here rather than in the kernel is the whole reason the kernel has
    no recursion in it. The longest canonical expansion is four code points and
    the longest compatibility one is eighteen, so nothing here is deep, but a
    depth that is not deep is still a depth somebody has to bound.
    """
    memo: dict[int, list[int]] = {}

    def walk(cp: int) -> list[int]:
        if cp in memo:
            return memo[cp]
        pieces = canon.get(cp)
        if pieces is None and full:
            pieces = compat.get(cp)
        if pieces is None:
            memo[cp] = [cp]
            return memo[cp]
        out: list[int] = []
        for piece in pieces:
            out.extend(walk(piece))
        memo[cp] = out
        return out

    keys = set(canon) if not full else set(canon) | set(compat)
    return {cp: walk(cp) for cp in sorted(keys)}


def pairs(canon: dict[int, list[int]], ccc: dict[int, int]) -> dict[tuple[int, int], int]:
    """The pairs CPython will actually put back together.

    Every character with a canonical decomposition of exactly two is a candidate.
    The candidate is kept when CPython, given the two characters, hands back the
    one. That filters out the composition exclusions, the singletons and the non
    starter decompositions all at once, without this file knowing what any of
    those three are, which is the point: the exclusion rules are the part of
    normalization most likely to be reimplemented slightly wrong.
    """
    kept: dict[tuple[int, int], int] = {}
    for cp, pieces in canon.items():
        if len(pieces) != 2:
            continue
        first, second = pieces
        if ccc.get(first, 0) != 0:
            continue
        if unicodedata.normalize("NFC", chr(first) + chr(second)) == chr(cp):
            kept[(first, second)] = cp
    return kept


def hangul_decompose(cp: int) -> list[int] | None:
    """A Hangul syllable taken apart by arithmetic, or None if it is not one."""
    if not SBASE <= cp < SBASE + SCOUNT:
        return None
    index = cp - SBASE
    lead = LBASE + index // NCOUNT
    vowel = VBASE + (index % NCOUNT) // TCOUNT
    trail = TBASE + index % TCOUNT
    return [lead, vowel] if trail == TBASE else [lead, vowel, trail]


def model(
    text: str,
    nfd: dict[int, list[int]],
    nfkd: dict[int, list[int]],
    ccc: dict[int, int],
    composes: dict[tuple[int, int], int],
    full: bool,
    compose: bool,
) -> str:
    """The whole algorithm in Python, out of nothing but the emitted tables.

    This exists so the generator can check its own tables rather than checking
    CPython against itself. Every step here has a counterpart in `normalize.mojo`
    and the two are meant to be read side by side.
    """
    table = nfkd if full else nfd
    out: list[int] = []
    for ch in text:
        cp = ord(ch)
        pieces = hangul_decompose(cp)
        if pieces is not None:
            out.extend(pieces)
        else:
            out.extend(table.get(cp, [cp]))

    # Canonical ordering. A run of marks is sorted by combining class and the
    # sort has to be stable, because two marks of the same class are ordered by
    # where they were written and swapping them changes the text.
    i = 1
    while i < len(out):
        here = ccc.get(out[i], 0)
        if here != 0 and ccc.get(out[i - 1], 0) > here:
            out[i - 1], out[i] = out[i], out[i - 1]
            i = max(1, i - 1)
        else:
            i += 1

    if not compose:
        return "".join(chr(c) for c in out)

    return "".join(chr(c) for c in composed(out, ccc, composes))


def composed(
    out: list[int], ccc: dict[int, int], composes: dict[tuple[int, int], int]
) -> list[int]:
    """Canonical composition, which is the algorithm from the standard.

    Walk forward holding the last starter. A character combines with that
    starter when the pair is in the table and nothing blocks it, and a character
    blocks when it sits between the starter and the candidate with a combining
    class that is not lower than the candidate's. That blocking rule is the part
    worth stating, because without it two marks on one letter would compose in
    either order and the answer would depend on which one was written first.
    """
    result: list[int] = []
    starter = -1
    last_class = -1
    for cp in out:
        here = ccc.get(cp, 0)
        if starter >= 0 and last_class < here:
            joined = hangul_compose(result[starter], cp)
            if joined is None:
                joined = composes.get((result[starter], cp))
            if joined is not None:
                result[starter] = joined
                continue
        if here == 0:
            starter = len(result)
            last_class = -1
        else:
            last_class = here
        result.append(cp)
    return result


def hangul_compose(first: int, second: int) -> int | None:
    """Two Hangul jamo put together by arithmetic, or None if they are not."""
    if LBASE <= first < LBASE + LCOUNT and VBASE <= second < VBASE + VCOUNT:
        return SBASE + ((first - LBASE) * VCOUNT + (second - VBASE)) * TCOUNT
    if (
        SBASE <= first < SBASE + SCOUNT
        and (first - SBASE) % TCOUNT == 0
        and TBASE < second < TBASE + TCOUNT
    ):
        return first + (second - TBASE)
    return None


def awkward(ccc: dict[int, int], nfkd: dict[int, list[int]], seed: int) -> list[str]:
    """Strings built to break a normalizer rather than to exercise it.

    Random text over the whole of Unicode is nearly all characters that
    normalize to themselves, and a normalizer that did nothing at all would pass
    on almost every row of it. What separates a right implementation from a
    plausible one is sequences: several marks of the same class in a row, marks
    of different classes needing a reorder, a decomposition that ends in a mark
    followed by a character that starts with one, and Hangul both spelled out and
    put together.
    """
    rng = random.Random(seed)
    marks = sorted(ccc)
    same_class: dict[int, list[int]] = {}
    for cp, klass in ccc.items():
        same_class.setdefault(klass, []).append(cp)
    crowded = [sorted(v) for v in same_class.values() if len(v) > 1]
    starters = [cp for cp in nfkd if ccc.get(cp, 0) == 0]
    jamo = list(range(LBASE, LBASE + LCOUNT)) + list(range(VBASE, VBASE + VCOUNT))
    jamo += list(range(TBASE + 1, TBASE + TCOUNT))
    syllables = [SBASE + rng.randrange(SCOUNT) for _ in range(200)]

    rows: list[str] = []
    for _ in range(40000):
        base = chr(rng.choice(starters))
        rows.append(base + "".join(chr(rng.choice(marks)) for _ in range(rng.randint(1, 4))))
    for _ in range(20000):
        group = rng.choice(crowded)
        base = chr(rng.choice(starters))
        rows.append(base + "".join(chr(rng.choice(group)) for _ in range(rng.randint(2, 4))))
    for _ in range(20000):
        rows.append("".join(chr(rng.choice(jamo)) for _ in range(rng.randint(1, 6))))
    for _ in range(10000):
        rows.append("".join(chr(rng.choice(syllables)) for _ in range(rng.randint(1, 3))))
    for _ in range(10000):
        pool = starters + marks + jamo + syllables
        rows.append("".join(chr(rng.choice(pool)) for _ in range(rng.randint(0, 6))))
    return rows


def check(
    nfd: dict[int, list[int]],
    nfkd: dict[int, list[int]],
    ccc: dict[int, int],
    composes: dict[tuple[int, int], int],
) -> None:
    """Runs the four forms out of the tables and refuses to write on a mismatch."""
    forms = {
        "NFD": (False, False),
        "NFC": (False, True),
        "NFKD": (True, False),
        "NFKC": (True, True),
    }
    rows = [chr(cp) for cp in range(MAX) if not 0xD800 <= cp < 0xE000]
    rows += awkward(ccc, nfkd, seed=11)
    for name, (full, compose) in forms.items():
        for row in rows:
            want = unicodedata.normalize(name, row)
            got = model(row, nfd, nfkd, ccc, composes, full, compose)
            if got != want:
                raise SystemExit(
                    f"{name} disagrees with CPython on {row!r}: "
                    f"want {[hex(ord(c)) for c in want]} got {[hex(ord(c)) for c in got]}"
                )
    print(f"checked four forms over {len(rows)} rows")


def emit_table(name: str, table: dict[int, list[int]], doc: str) -> str:
    """One decomposition table as keys, start offsets and one flat run of values."""
    keys = sorted(table)
    starts: list[int] = []
    values: list[int] = []
    for cp in keys:
        starts.append(len(values))
        values.extend(table[cp])
    starts.append(len(values))
    out = f"comptime {name}_KEYS: InlineArray[UInt32, {len(keys)}] = [\n"
    out += "".join(f"    0x{cp:04X},\n" for cp in keys) + "]\n"
    out += f'"""{doc} The characters that have one, in order."""\n\n'
    out += f"comptime {name}_STARTS: InlineArray[UInt32, {len(starts)}] = [\n"
    out += "".join(f"    {v},\n" for v in starts) + "]\n"
    out += '"""Where each key\'s sequence begins, with one past the last on the end."""\n\n'
    out += f"comptime {name}_VALUES: InlineArray[UInt32, {len(values)}] = [\n"
    out += "".join(f"    0x{v:04X},\n" for v in values) + "]\n"
    out += '"""Every sequence laid end to end."""\n\n'
    return out


def emit_ccc(ccc: dict[int, int]) -> str:
    """The combining classes as a sorted key list and a value list."""
    keys = sorted(ccc)
    out = f"comptime CCC_KEYS: InlineArray[UInt32, {len(keys)}] = [\n"
    out += "".join(f"    0x{cp:04X},\n" for cp in keys) + "]\n"
    out += '"""The characters whose combining class is not zero, in order."""\n\n'
    out += f"comptime CCC_VALUES: InlineArray[UInt8, {len(keys)}] = [\n"
    out += "".join(f"    {ccc[cp]},\n" for cp in keys) + "]\n"
    out += '"""Each one\'s class, which is what canonical ordering sorts on."""\n\n'
    return out


def emit_composition(composes: dict[tuple[int, int], int]) -> str:
    """The composable pairs, packed one pair to a number so the search is one list."""
    packed = sorted((first << 21) | second for first, second in composes)
    lookup = {(first << 21) | second: cp for (first, second), cp in composes.items()}
    out = f"comptime COMPOSE_PAIRS: InlineArray[UInt64, {len(packed)}] = [\n"
    out += "".join(f"    0x{key:012X},\n" for key in packed) + "]\n"
    out += (
        '"""Each composable pair as the starter shifted up by twenty one bits and'
        ' the following character below it, sorted so the search is a bisection."""\n\n'
    )
    out += f"comptime COMPOSE_VALUES: InlineArray[UInt32, {len(packed)}] = [\n"
    out += "".join(f"    0x{lookup[key]:04X},\n" for key in packed) + "]\n"
    out += '"""What each pair composes to."""\n\n'
    return out


def main() -> None:
    """Checks the tables against CPython and then writes them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("firepanda/kernel/normalize_data.mojo"),
        help="where the tables go",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="write without running the whole algorithm against CPython first",
    )
    args = parser.parse_args()

    canon, compat, ccc = decompositions()
    nfd = expanded(canon, compat, full=False)
    nfkd = expanded(canon, compat, full=True)
    composes = pairs(canon, ccc)

    if not args.skip_check:
        check(nfd, nfkd, ccc, composes)

    body = HEADER.format(
        canon_n=len(nfd),
        canon_v=sum(len(v) for v in nfd.values()),
        compat_n=len(nfkd),
        compat_v=sum(len(v) for v in nfkd.values()),
        ccc_n=len(ccc),
        comp_n=len(composes),
        hangul=SCOUNT,
    )
    body += emit_table("NFD", nfd, "The canonical decompositions, every mapping already applied.")
    body += emit_table(
        "NFKD",
        nfkd,
        "The compatibility decompositions, which include the canonical ones.",
    )
    body += emit_ccc(ccc)
    body += emit_composition(composes)
    args.out.write_text(body)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
