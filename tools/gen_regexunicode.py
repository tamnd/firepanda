"""Writes the Unicode name table that `\\p{...}` and `\\P{...}` are read against.

This generator reads pyarrow rather than CPython, which is the opposite of
tools/gen_regexclass.py next door, and the reason is the whole of why this table
exists. `\\p` is not Python syntax at all. CPython's `re` calls it `bad escape
\\p` in every version, so pandas hands every pattern holding one to Arrow, and
the only engine that will ever run it is the RE2 inside Arrow. Asking CPython
what `\\p{L}` means would be asking a library that has no opinion, and asking the
published Unicode data files would be asking a third party that neither of them
consults. `pyarrow.compute.match_substring_regex` is the exact engine the
caller's pattern will reach, so it is not an approximation of the right answer,
it is the right answer.

That matters more here than it did for the Perl classes, because RE2 and CPython
are built against different Unicode releases and this table is large enough that
the skew shows. Measured while this was written, `\\p{L}` and CPython's
`unicodedata` disagree about 4302 code points and `\\p{Lo}` about 4243, every one
of them a code point one release has assigned and the other has not.

Two of RE2's rules are worth knowing before reading the output, because neither
is guessable and both were measured. `\\p{C}` does not hold the unassigned code
points: it is exactly `Cc` and `Cf` and `Co` and `Cs`, and `\\p{Cn}` is refused
outright, which is why the one letter names here are generated as unions of the
two letter ones and then checked against RE2 rather than taken from it. And
`\\p{Cs}` is a real name that can never match anything, because a surrogate has
no UTF-8 encoding and every row reaching the engine arrived as UTF-8.

The measurement is one compute call per name over every code point, which is
around two hundred calls over a column of 1112064 rows, so it takes a few
minutes and is not something to run in a loop.

    pixi run -e differential python tools/gen_regexunicode.py
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

POINTS = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
"""Every code point a row can actually hold. A surrogate is left out because
`chr` of one cannot be encoded as UTF-8 and a row that arrived as UTF-8 can
never contain one."""

COLUMN = pa.array([chr(c) for c in POINTS])
"""One row per code point. Asking RE2 whether a name matches each of them in one
call is the only way to get a set out of an engine that will only answer yes or
no about a whole string."""

PARTS = {
    "L": ("Lu", "Ll", "Lt", "Lm", "Lo"),
    "M": ("Mn", "Mc", "Me"),
    "N": ("Nd", "Nl", "No"),
    "P": ("Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"),
    "S": ("Sm", "Sc", "Sk", "So"),
    "C": ("Cc", "Cf", "Co", "Cs"),
}
"""The one letter general categories and what each is the union of. Checked
against RE2 after the parts are measured rather than trusted, because the one
that would have been got wrong is `C`, which leaves out the unassigned code
points that a reader would expect a category called other to hold."""

PAIRS = tuple(name for names in PARTS.values() for name in names)
"""The two letter general categories, which are the ones actually measured."""

SCRIPTS = """
Adlam Ahom Anatolian_Hieroglyphs Arabic Armenian Avestan Balinese Bamum Bassa_Vah
Batak Bengali Bhaiksuki Bopomofo Brahmi Braille Buginese Buhid Canadian_Aboriginal
Carian Caucasian_Albanian Chakma Cham Cherokee Chorasmian Common Coptic Cuneiform
Cypriot Cypro_Minoan Cyrillic Deseret Devanagari Dives_Akuru Dogra Duployan
Egyptian_Hieroglyphs Elbasan Elymaic Ethiopic Georgian Glagolitic Gothic Grantha
Greek Gujarati Gunjala_Gondi Gurmukhi Han Hangul Hanifi_Rohingya Hanunoo Hatran
Hebrew Hiragana Imperial_Aramaic Inherited Inscriptional_Pahlavi
Inscriptional_Parthian Javanese Kaithi Kannada Katakana Kawi Kayah_Li Kharoshthi
Khitan_Small_Script Khmer Khojki Khudawadi Lao Latin Lepcha Limbu Linear_A Linear_B
Lisu Lycian Lydian Mahajani Makasar Malayalam Mandaic Manichaean Marchen
Masaram_Gondi Medefaidrin Meetei_Mayek Mende_Kikakui Meroitic_Cursive
Meroitic_Hieroglyphs Miao Modi Mongolian Mro Multani Myanmar Nabataean Nag_Mundari
Nandinagari New_Tai_Lue Newa Nko Nushu Nyiakeng_Puachue_Hmong Ogham Ol_Chiki
Old_Hungarian Old_Italic Old_North_Arabian Old_Permic Old_Persian Old_Sogdian
Old_South_Arabian Old_Turkic Old_Uyghur Oriya Osage Osmanya Pahawh_Hmong Palmyrene
Pau_Cin_Hau Phags_Pa Phoenician Psalter_Pahlavi Rejang Runic Samaritan Saurashtra
Sharada Shavian Siddham SignWriting Sinhala Sogdian Sora_Sompeng Soyombo Sundanese
Syloti_Nagri Syriac Tagalog Tagbanwa Tai_Le Tai_Tham Tai_Viet Takri Tamil Tangsa
Tangut Telugu Thaana Thai Tibetan Tifinagh Tirhuta Toto Ugaritic Vai Vithkuqi
Wancho Warang_Citi Yezidi Yi Zanabazar_Square
""".split()
"""Every script name RE2 takes. There is no way to ask an engine for the list of
names it knows, so this is a written list and the generator refuses to write
anything if RE2 turns one of them down, which is what turns a release that drops
a name into a failure here rather than a wrong table later."""

HEADER = '''"""Every name `\\\\p{{...}}` may hold, and the code points each of them covers.

`\\\\p` is not Python syntax. CPython's `re` calls it `bad escape \\\\p`, so pandas
hands every pattern holding one to Arrow and the RE2 inside Arrow is the only
engine that will ever run it. That is why this table was measured against
pyarrow and why the three next door in `classdata.mojo` were measured against
CPython: those are what a class means to Python's engine and these are what a
name means to RE2's, and neither library has an opinion about the other's.

Two of RE2's rules are in the numbers rather than in any sentence. `\\\\p{{C}}` is
exactly `Cc` and `Cf` and `Co` and `Cs` and holds no unassigned code point at
all, and `\\\\p{{Cn}}` is not a name RE2 has. And `\\\\p{{Cs}}` is a name that can
never match, because a surrogate has no UTF-8 encoding and every row reaching
the engine arrived as UTF-8, so it is here with no ranges under it.

The names are sorted, so a lookup is a binary search over `UNICODE_NAMES` using
`UNICODE_NAME_AT` for where each one starts. `UNICODE_RANGE_AT` then says which
slice of `UNICODE_RANGES` belongs to the name, as low and high pairs with both
ends inside the set, which is the shape `_category_ranges` in `program.mojo`
already hands back.

{summary}

Generated by tools/gen_regexunicode.py against the RE2 inside pyarrow
{pyarrow}, committed rather than built because the Mojo build has no Python in
it. The generator checks that every script name it was told about is still a
name RE2 takes, and that each one letter category is exactly the union of its
two letter parts, before it writes anything, so a release that moves either one
fails here rather than drifting.
"""

'''


def runs(flags: list[bool]) -> list[tuple[int, int]]:
    """The code points a name matched, folded into the runs they make up.

    Args:
        flags: One answer per entry of `POINTS`, in order.

    Returns:
        Low and high pairs, both ends inside the set.
    """
    out: list[list[int]] = []
    for i, on in enumerate(flags):
        if not on:
            continue
        cp = POINTS[i]
        if out and cp == out[-1][1] + 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return [(low, high) for low, high in out]


def asked(name: str) -> list[tuple[int, int]]:
    """What RE2 says `\\p{name}` covers.

    Args:
        name: The name, as it is written between the braces.

    Returns:
        The ranges, in order.
    """
    return runs(pc.match_substring_regex(COLUMN, "\\p{%s}" % name).to_pylist())


def reads(pattern: str) -> bool:
    """Whether RE2 reads a pattern at all.

    Args:
        pattern: The pattern.

    Returns:
        Whether it compiled.
    """
    try:
        pc.match_substring_regex(pa.array(["a"]), pattern)
    except Exception:
        return False
    return True


def points(ranges: list[tuple[int, int]]) -> set[int]:
    """The ranges written back out as the set they stand for."""
    out: set[int] = set()
    for low, high in ranges:
        out.update(range(low, high + 1))
    return out


def measured() -> dict[str, list[tuple[int, int]]]:
    """Every name and its ranges, asked of RE2 one name at a time.

    The one letter categories are built as unions rather than measured, and then
    checked against RE2, because a union that turns out to be wrong is a fact
    worth failing on. `Any` is written by hand as the whole space, since asking
    for it would produce one range that reads as if the surrogates were in it,
    which they are and which no row can hold.

    Returns:
        The names and their ranges.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    for name in PAIRS:
        out[name] = asked(name)
    for name, parts in PARTS.items():
        whole: set[int] = set()
        for part in parts:
            whole |= points(out[part])
        if whole != points(asked(name)):
            raise SystemExit(r"\p{%s} is no longer the union of its parts" % name)
        out[name] = runs([cp in whole for cp in POINTS])
    out["Any"] = [(0, 0x10FFFF)]
    for name in SCRIPTS:
        if not reads("\\p{%s}" % name):
            raise SystemExit(r"RE2 no longer reads \p{%s}" % name)
        out[name] = asked(name)
    return out


def check() -> None:
    """The rules about which names exist, asserted before anything is written.

    Every one of these is a sentence somebody would write from memory and get
    wrong. `Cn` is a real Unicode general category and is not a name RE2 has.
    The `Is` prefix that several other engines take is not one RE2 takes. And a
    name is case sensitive, so `latin` is not `Latin`.
    """
    if reads("\\p{Cn}"):
        raise SystemExit(r"RE2 now reads \p{Cn}, so the C union has moved")
    if reads("\\p{IsLatin}") or reads("\\p{Is_Latin}"):
        raise SystemExit(r"RE2 now takes an Is prefix")
    if reads("\\p{latin}"):
        raise SystemExit(r"\p names are no longer case sensitive")
    if not reads("\\pL") or not reads("\\p{^L}"):
        raise SystemExit(r"RE2 no longer reads \pL or \p{^L}")


def emit(table: dict[str, list[tuple[int, int]]]) -> str:
    """The whole file body below the docstring.

    Args:
        table: The names and their ranges.

    Returns:
        The four comptime values, in the order a reader wants them.
    """
    names = sorted(table)
    joined = "".join(names)
    name_at, at = [0], 0
    for name in names:
        at += len(name.encode())
        name_at.append(at)
    flat: list[int] = []
    range_at = [0]
    for name in names:
        for low, high in table[name]:
            flat.append(low)
            flat.append(high)
        range_at.append(len(flat))

    out = f'comptime UNICODE_NAMES: StaticString = "{joined}"\n'
    out += (
        '"""Every name, sorted and run together. A lookup is a binary search\n'
        "over this using `UNICODE_NAME_AT` to find where each one starts, which\n"
        "is a few string comparisons once while a pattern is being compiled.\n"
        'There are %d of them."""\n\n' % len(names)
    )
    out += _array("UNICODE_NAME_AT", name_at, 0)
    out += (
        '"""Where each name starts in `UNICODE_NAMES`, with one more entry on\n'
        'the end so that the last name has somewhere to stop."""\n\n'
    )
    out += _array("UNICODE_RANGE_AT", range_at, 0)
    out += (
        '"""Where each name\'s ranges start in `UNICODE_RANGES`, counted in\n'
        "single values rather than in pairs, with one more entry on the end. A\n"
        'name with nothing under it, which is `Cs`, has the two equal."""\n\n'
    )
    out += _array("UNICODE_RANGES", flat, 4)
    out += (
        '"""Every name\'s ranges, one after another, as low and high pairs with\n'
        "both ends inside the set. `UNICODE_RANGE_AT` says which slice belongs\n"
        'to which name."""\n'
    )
    return out


def _array(name: str, values: list[int], width: int) -> str:
    """One comptime array, one value to a line the way the tables next door are.

    Args:
        name: What to call it.
        values: The values.
        width: How many hex digits to pad a code point to, or zero for decimal.

    Returns:
        The declaration, without its docstring.
    """
    if width:
        body = "\n".join(f"    0x{v:0{width}X}," for v in values)
    else:
        body = "\n".join(f"    {v}," for v in values)
    return (
        f"comptime {name}: InlineArray[Int32, {len(values)}] = [\n{body}\n]\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("firepanda/kernel/regex/unicodedata.mojo"),
        help="where to write the table",
    )
    args = parser.parse_args()

    check()
    table = measured()

    pairs = sum(len(table[name]) for name in table)
    covered = len(points(table["L"]))
    sentence = (
        "%d names, of which %d are general categories and %d are scripts and "
        "one is `Any`, in %d ranges between them. `L` alone is %d code points "
        "and `Han` is %d, and `Cs` is the only name with nothing under it."
        % (
            len(table),
            len(PAIRS) + len(PARTS),
            len(SCRIPTS),
            pairs,
            covered,
            len(points(table["Han"])),
        )
    )
    body = HEADER.format(
        summary="\n".join(textwrap.wrap(sentence, width=75)),
        pyarrow=pa.__version__,
    )
    args.out.write_text(body + emit(table))
    print(sentence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
