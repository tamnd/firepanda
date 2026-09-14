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
from regex_corpus import SHOWN, corpus, names_a_character, report


comptime CASES = 30000
"""How many generated patterns when nobody says otherwise.

Large enough that the awkward corners of the grammar come up several times each,
and small enough that the whole thing is well under a minute, most of which is
Python's parser rather than this one.
"""

comptime SEED = 1
"""The default seed. Fixed rather than taken from the clock, so that a failure
reported by this program can be reproduced by the person reading the report."""


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
        if names_a_character(pattern):
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
