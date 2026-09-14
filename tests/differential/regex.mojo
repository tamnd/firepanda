"""Patterns nobody wrote, through both front ends.

Every regular expression call in pandas passes through one decision before it
reaches an engine: Python's parser is handed the pattern, and what comes back
decides whether the call is answered by Python's `re` or by Arrow's RE2. The two
engines give different answers for `\\d`, for `$` and for several other things,
so that decision is visible in results rather than only in refusals, and a
library copying pandas has to make the same decision on the same patterns for
the same reasons.

This asks both questions of both front ends over patterns built out of the
grammar rather than collected by hand. A hand written corpus is bounded by what
somebody thought to write, and what somebody thinks to write is the constructs
they already know are interesting, which is the wrong end of the problem: the
routing mistakes that matter are the ones on patterns nobody would think to try,
because those are the ones that reach a user before they reach a test.

Two answers are compared per pattern, and they are separate on purpose.

Whether Python's grammar reads it at all. pandas catches the parse error and
gives the pattern to Arrow, so an unreadable pattern and a plain one route the
same way, and folding the two together would hide every disagreement where this
parser refuses a pattern Python reads and then routes it correctly by accident.

Whether pandas routes it to Python. This is the walk over the parsed tokens
looking for a lookaround or a backreference, and the walk is incomplete upstream
in a way document 76 sets out. The incompleteness is reproduced, which makes it
a thing this differential has to confirm rather than a thing it would report as
a difference.

Both ceilings are zero. There is no bounded disagreement worth carrying here,
because a pattern routed to the wrong engine is not a refusal, it is an answer
computed by the engine pandas would not have used.

Two kinds of pattern are counted and set aside rather than compared, and both
are named in the report so that setting them aside is a number somebody can
watch rather than a silence.

A pattern that takes pandas down. Turning on the ASCII flag and the Unicode flag
in the same pattern is reported by Python's parser as a `ValueError`, pandas
catches only `re.error`, and the exception comes out of `str.contains` naming a
module the caller never imported. There is no routing decision to agree with, so
comparing one would be inventing an answer for pandas.

A pattern naming a character rather than writing it. Resolving `\\N{GREEK SMALL
LETTER ALPHA}` needs the Unicode name table, which this library does not carry
yet, so it reads the braces and not what is between them. Every name that exists
routes the same way as a result, and every name that does not is read here and
refused by Python, which is a real difference with one fix and the fix is a
table. Document 76 section 9 says why the table is not worth it yet, and until
it exists these are held out of the agreement figure rather than reported as the
same difference a thousand times.

The oracle is `tools/regex_oracle.py`, which calls pandas' own predicate rather
than a copy of it.

Usage:
    pixi run differential-regex
    pixi run differential-regex -- --cases 40000 --seed 7
"""

from std.python import Python, PythonObject
from std.sys import argv

from firepanda.kernel.regex.route import (
    ENGINE_PYTHON,
    engine_for,
    reads_as_python,
)

comptime CASES = 30000
"""How many generated patterns when nobody says otherwise.

Large enough that the awkward corners of the grammar come up several times each,
and small enough that the whole thing is well under a minute, most of which is
Python's parser rather than this one.
"""

comptime SEED = 1
"""The default seed. Fixed rather than taken from the clock, so that a failure
reported by this program can be reproduced by the person reading the report."""

comptime SHOWN = 30
"""How many disagreements of each kind are printed before the rest are counted.

Thirty because a systematic mistake in a grammar shows up in the first few
examples and the rest of the list is the same mistake wearing different
characters, and a report nobody scrolls to the end of is a report with its
conclusion off the screen.
"""


struct _Rng(Movable):
    """A small deterministic generator.

    Not the standard library's, because the corpus has to be the same corpus on
    every machine and on every run: a differential whose input moves is one
    where a fix cannot be told from a reshuffle.
    """

    var state: UInt64
    """Everything it knows."""

    def __init__(out self, seed: UInt64):
        """Starts at a seed.

        Args:
            seed: Where to start. Zero is turned into one, since this generator
                has nowhere to go from zero.
        """
        self.state = seed if seed != 0 else 1

    def next(mut self) -> UInt64:
        """The next number.

        Returns:
            A number, well mixed enough for choosing among a few dozen
            fragments, which is all it is asked to do.
        """
        var x = self.state
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.state = x
        return x * 0x2545F4914F6CDD1D

    def below(mut self, bound: Int) -> Int:
        """A number under a bound.

        Args:
            bound: One past the largest wanted.

        Returns:
            Something from zero up to the bound.
        """
        return Int(self.next() >> 33) % bound


def _atoms() -> List[String]:
    """The pieces with nothing underneath them.

    Three quarters of these are things a person would call malformed, which is
    the point. A grammar is defined as much by what it refuses as by what it
    reads, and pandas turns a refusal into a routing decision, so a parser that
    is right about every valid pattern and wrong about `\\x4` sends that pattern
    to the wrong engine.

    Returns:
        The list, in no order that means anything.
    """
    var out: List[String] = [
        "a",
        "b",
        "z",
        "0",
        "9",
        "_",
        " ",
        "é",
        "漢",
        ".",
        "^",
        "$",
        "\\d",
        "\\D",
        "\\s",
        "\\S",
        "\\w",
        "\\W",
        "\\b",
        "\\B",
        "\\A",
        "\\Z",
        "\\z",
        "\\G",
        "\\1",
        "\\2",
        "\\9",
        "\\0",
        "\\12",
        "\\123",
        "\\400",
        "\\x41",
        "\\x4",
        "\\xzz",
        "\\u0041",
        "\\u00",
        "\\U00000041",
        "\\U0011FFFF",
        "\\N{BULLET}",
        "\\N{NOT A CHARACTER NAME}",
        "\\N",
        "\\Nx",
        "\\p{L}",
        "\\P{L}",
        "\\k<n>",
        "\\Q",
        "\\e",
        "\\a",
        "\\f",
        "\\n",
        "\\r",
        "\\t",
        "\\v",
        "\\-",
        "\\ ",
        "\\.",
        "\\\\",
        "\\",
        "[abc]",
        "[^abc]",
        "[a-z]",
        "[z-a]",
        "[]",
        "[^]",
        "[]]",
        "[a-]",
        "[-a]",
        "[\\d]",
        "[\\w-]",
        "[[:alpha:]]",
        "[a\\]b]",
        "[\\x41-\\x5a]",
        "[^\\D]",
        "[a",
        "[a-\\d]",
        "[\\d-a]",
        "[é-漢]",
        "(?#note)",
        "(?i)",
        "(?s)",
        "(?m)",
        "(?x)",
        "(?a)",
        "(?L)",
        "(?u)",
        "(?-i)",
        "(?i-s)",
        "(?y)",
        "(?P=n)",
        "(?P=missing)",
        "(?P>n)",
        "*",
        "+",
        "?",
        "{2}",
        "{2,3}",
        "{,3}",
        "{2,}",
        "{}",
        "{2",
        "}",
        "]",
        "(",
        ")",
        "|",
        "-",
        "",
    ]
    return out^


def _quantifiers() -> List[String]:
    """The suffixes that repeat whatever is in front of them.

    The possessive ones are here even though RE2 refuses them, because refusing
    them is RE2's job and reading them is Python's, and the whole point of this
    front end is that the reading happens before the choice of engine.

    Returns:
        The list.
    """
    var out: List[String] = [
        "*",
        "+",
        "?",
        "*?",
        "+?",
        "??",
        "*+",
        "++",
        "?+",
        "{0}",
        "{2}",
        "{2,}",
        "{,2}",
        "{1,3}",
        "{1,3}?",
        "{1,3}+",
        "{3,1}",
        "{2}+",
        "{99999}",
    ]
    return out^


def _wrappers() -> List[String]:
    """The brackets that take something else inside them.

    Each is written with a `%` where the body goes, which is a stand in rather
    than a format string because there is exactly one hole and a real formatter
    would be more machinery than the thing it replaces. The conditional forms
    have two holes and are handled apart from these.

    Returns:
        The list.
    """
    var out: List[String] = [
        "(%)",
        "(?:%)",
        "(?i:%)",
        "(?-i:%)",
        "(?im:%)",
        "(?P<n>%)",
        "(?P<1n>%)",
        "(?P<>%)",
        "(?'n'%)",
        "(?=%)",
        "(?!%)",
        "(?<=%)",
        "(?<!%)",
        "(?>%)",
        "(?#%)",
        "(%",
        "%)",
    ]
    return out^


def _filled(shape: StringSlice, body: StringSlice) -> String:
    """Puts a body into a wrapper.

    Args:
        shape: The wrapper, holding one percent sign where the body goes.
        body: What to put there.

    Returns:
        The two joined, or the shape unchanged when it holds no percent sign.
    """
    var out = String()
    var seen = False
    for point in shape.codepoints():
        if not seen and point.to_u32() == UInt32(ord("%")):
            out += body
            seen = True
        else:
            out += String(point)
    return out^


def _grown(mut rng: _Rng, depth: Int) -> String:
    """One item, which at any depth above zero may hold more items.

    Args:
        rng: The generator.
        depth: How much further down it may go. At zero it returns an atom,
            which is what stops this.

    Returns:
        A fragment of a pattern.
    """
    var atoms = _atoms()
    if depth <= 0:
        return atoms[rng.below(len(atoms))]

    var roll = rng.below(100)
    if roll < 30:
        return atoms[rng.below(len(atoms))]
    if roll < 50:
        var wrappers = _wrappers()
        return _filled(
            wrappers[rng.below(len(wrappers))], _grown(rng, depth - 1)
        )
    if roll < 62:
        var quantifiers = _quantifiers()
        return _grown(rng, depth - 1) + quantifiers[rng.below(len(quantifiers))]
    if roll < 74:
        var out = String()
        var parts = 2 + rng.below(2)
        for at in range(parts):
            if at != 0:
                out += "|"
            out += _grown(rng, depth - 1)
        return out^
    if roll < 82:
        var out = String("(?(1)")
        out += _grown(rng, depth - 1)
        if rng.below(2) == 0:
            out += "|"
            out += _grown(rng, depth - 1)
        out += ")"
        return out^
    if roll < 88:
        var out = String("(?(n)")
        out += _grown(rng, depth - 1)
        out += ")"
        return out^
    if roll < 94:
        # A named group and a reference to it, which is the only way a
        # backreference by name is ever valid, and is therefore the only way to
        # reach the routing decision through one.
        var out = String("(?P<n>")
        out += _grown(rng, depth - 1)
        out += ")(?P=n)"
        return out^
    var out = String()
    var parts = 2 + rng.below(3)
    for _ in range(parts):
        out += _grown(rng, depth - 1)
    return out^


def _measured() -> List[String]:
    """The patterns the measurements in document 76 were taken on.

    Kept apart from the generated ones and always asked, because a generated
    corpus is a statement about the grammar as a whole and these are the
    individual facts the design rests on. A corpus that stops covering one of
    them after a seed changes would take a documented finding with it silently.

    Returns:
        The list.
    """
    var out: List[String] = [
        "a",
        "(?=a)",
        "(?=a)*",
        "(?=a)?",
        "(?:(?=a))*",
        "(?:(?=a))+",
        "(?:(?=a)){2}",
        "(?:(?=a))?",
        "(?:(?=a))*+",
        "(?>(?=a))",
        "(a)(?:\\1)*",
        "(a)(?:\\1)?",
        "[a](?:(?=x))*",
        "(?:(?!a))*",
        "(?:(?<=a))*",
        "((?=a))*",
        "(?:a|(?:(?=b))*)",
        "(a)(?(1)(?=b)|c)",
        "(?!)",
        "(?<!)",
        "(?=)",
        "(?<=)",
        "(?:)",
        "()",
        "(?#hi)a",
        "(?i)a",
        "(?i)(?s)a",
        "a(?i)b",
        "(?a)(?u)",
        "(?a)(?u)(?=a)",
        "(?#\\)",
        "(?#a\\\\)b",
        "\\N{GREEK SMALL LETTER ALPHA}",
        "[^a]",
        "[a]",
        "a|b",
        "a{}",
        "a{2",
        "\\p{L}",
        "(a)\\1",
        "\\1(a)",
        "(?P<n>a)(?P=n)",
        "(?P=n)",
        "(?i:(?=a))",
        "(?:ab)",
        "(ab)",
        "^*",
        "^?",
        "**",
        "a**",
        "(?:a)*",
        "\\b*",
    ]
    return out^


def corpus(cases: Int, seed: UInt64) -> List[String]:
    """The patterns to ask about.

    Args:
        cases: How many to generate on top of the measured ones.
        seed: Where the generator starts.

    Returns:
        The measured patterns first, then the generated ones.
    """
    var out = _measured()
    var rng = _Rng(seed)
    for _ in range(cases):
        out.append(_grown(rng, 1 + rng.below(3)))
    return out^


def ask_pandas(patterns: List[String]) raises -> List[String]:
    """Both of pandas' answers for every pattern.

    One call across the boundary for the whole batch, because the work per
    pattern is a few microseconds of parsing and the crossing is not.

    Args:
        patterns: The corpus.

    Returns:
        One two character answer per pattern, in order.

    Raises:
        Error: If the helper could not be reached, or answered the wrong number
            of times.
    """
    var batch = Python.list()
    for pattern in patterns:
        batch.append(PythonObject(pattern))

    var python_path = Python.import_module("sys").path
    python_path.insert(0, "tools")
    var helper = Python.import_module("regex_oracle")

    var out = List[String]()
    for line in String(helper.answers(batch)).split("\n"):
        if line.byte_length() != 0:
            out.append(String(line))
    if len(out) != len(patterns):
        raise Error(
            String(
                "pandas answered about ",
                len(out),
                " patterns and there are ",
                len(patterns),
            )
        )
    return out^


def report(title: StringSlice, patterns: List[String], total: Int):
    """Prints one kind of disagreement.

    Args:
        title: What the list is.
        patterns: The patterns in it.
        total: How many were compared, for the share.
    """
    if len(patterns) == 0:
        return
    print()
    print(len(patterns), "of", total, title)
    var shown = len(patterns) if len(patterns) < SHOWN else SHOWN
    for at in range(shown):
        print("   ", patterns[at])
    if len(patterns) > shown:
        print("   ", len(patterns) - shown, "more")


def _names_a_character(pattern: StringSlice) -> Bool:
    """Whether a pattern holds a `\\N{...}` escape.

    The backslashes are counted rather than looked at one at a time, because
    `\\\\N{` is an escaped backslash followed by a plain letter and is not the
    escape at all.

    Args:
        pattern: The pattern.

    Returns:
        True when the parser will meet a named character escape.
    """
    var at = 0
    var bytes = pattern.as_bytes()
    while at < len(bytes):
        if bytes[at] != 0x5C:
            at += 1
            continue
        var run = 0
        while at < len(bytes) and bytes[at] == 0x5C:
            run += 1
            at += 1
        if run % 2 == 0:
            continue
        if at + 1 < len(bytes) and bytes[at] == 0x4E and bytes[at + 1] == 0x7B:
            return True
    return False


def main() raises:
    var cases = CASES
    var seed = UInt64(SEED)
    var args = argv()
    for at in range(1, len(args)):
        if args[at] == "--cases" and at + 1 < len(args):
            cases = Int(args[at + 1])
        elif args[at] == "--seed" and at + 1 < len(args):
            seed = UInt64(Int(args[at + 1]))

    var patterns = corpus(cases, seed)
    print("asking pandas about", len(patterns), "patterns")
    var answers = ask_pandas(patterns)

    var we_read = List[String]()
    var they_read = List[String]()
    var we_route = List[String]()
    var they_route = List[String]()
    var killed = 0
    var deferred = 0
    var compared = 0

    for at in range(len(patterns)):
        ref pattern = patterns[at]
        ref answer = answers[at]
        if answer[byte=0] == "x":
            killed += 1
            continue
        if _names_a_character(pattern):
            deferred += 1
            continue
        compared += 1
        var theirs_reads = answer[byte=0] == "y"
        var theirs_routes = answer[byte=1] == "y"
        var ours_reads = reads_as_python(pattern)
        var ours_routes = engine_for(pattern) == ENGINE_PYTHON

        if ours_reads and not theirs_reads:
            we_read.append(pattern)
        elif theirs_reads and not ours_reads:
            they_read.append(pattern)
        if ours_routes and not theirs_routes:
            we_route.append(pattern)
        elif theirs_routes and not ours_routes:
            they_route.append(pattern)

    var disagreements = (
        len(we_read) + len(they_read) + len(we_route) + len(they_route)
    )
    print("compared", compared, "patterns on both answers")
    print(
        "held out",
        killed,
        "pandas does not survive,",
        deferred,
        "name a character",
    )

    report("firepanda reads and Python does not:", we_read, compared)
    report("Python reads and firepanda does not:", they_read, compared)
    report(
        "firepanda routes to Python and pandas routes to Arrow:",
        we_route,
        compared,
    )
    report(
        "pandas routes to Python and firepanda routes to Arrow:",
        they_route,
        compared,
    )

    print()
    print(
        "agreement",
        (compared * 2 - disagreements) * 10000 // (compared * 2),
        "in ten thousand,",
        disagreements,
        "disagreements",
    )

    if disagreements != 0:
        raise Error(
            String(
                disagreements,
                (
                    " disagreements about which engine answers, against a"
                    " ceiling of zero"
                ),
            )
        )
