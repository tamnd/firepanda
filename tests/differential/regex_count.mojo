"""Patterns nobody wrote, counted in text, against pandas' own numbers.

The differential beside this one asks whether a pattern matches. This one asks
how many times, and it is a separate program rather than a fourth sweep in that
one because the answer is a different shape. Whether is a bit and the comparison
is a bit. How many is a number, it can be larger than the text has characters,
and the three ways it can be wrong all produce numbers that look reasonable.

What is being compared is the loop rather than the engine. The engine is the
same one the other differential already ran thirty thousand patterns through, so
a pattern that matches wrongly fails there first and fails here as a consequence.
What only this program can see is the loop Arrow wraps around the engine: where
it starts the pattern again, what the text looks like when it does, and how far
it moves when a match had no width. `firepanda/kernel/regex/pike.mojo` states
the three rules those come to and document 79 says where each was measured.

Two things are compared for every pattern, the same two as next door. Whether it
runs at all, since RE2 refuses constructs Python reads and pandas hands the
pattern to RE2 anyway, so the refusal is the specification rather than a gap.
And the sixteen numbers.

`count` is asked once rather than three times because there is nothing to anchor.
Upstream reads the pattern as the caller wrote it, with no rewrite and no
anchoring, which is what makes it the plainest of the four and is also why a
disagreement here is a disagreement about the loop and not about a rewrite.

The ceiling is zero. A wrong count is not a refusal a caller can see, it is a
column of numbers that looks exactly like a right one.

A pattern is set aside when firepanda's own compiler says the refusal is a gap
here rather than something RE2 refuses too, and the reasons are tallied in the
report, exactly as next door and for the reasons that file gives at length.

Usage:
    pixi run differential-regex-count
    pixi run differential-regex-count -- --cases 40000 --seed 7
"""

from std.collections.span import Span
from std.python import Python, PythonObject
from std.sys import argv

from firepanda.kernel.regex.method import METHOD_COUNT, program_for
from firepanda.kernel.regex.parse import decoded
from firepanda.kernel.regex.pike import Machine
from regex_corpus import corpus, report

comptime CASES = 30000
"""How many generated patterns when nobody says otherwise.

The same number the other two regular expression differentials use, on the same
corpus, so that a pattern named in one report can be looked up in the others.
"""

comptime SEED = 1
"""The default seed, fixed so that a failure can be reproduced by whoever reads
the report."""


def ask_pandas(patterns: List[String]) raises -> List[String]:
    """How many times pandas says every pattern matches in every text.

    One call across the boundary for the whole batch, for the reason the other
    differential gives: the work per pattern is a parse, a routing decision and
    sixteen short scans, and the crossing costs more than any of that.

    Args:
        patterns: The corpus.

    Returns:
        One line per pattern, each either a single `x` for a call that raised or
        one number per text.

    Raises:
        Error: If the helper could not be reached, or answered the wrong number
            of times.
    """
    var batch = Python.list()
    for pattern in patterns:
        batch.append(PythonObject(pattern))

    var python_path = Python.import_module("sys").path
    python_path.insert(0, "tools")
    var helper = Python.import_module("regex_count_oracle")

    var out = List[String]()
    for line in String(helper.answers(batch)).split("\n"):
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


def ask_texts() raises -> List[String]:
    """The text every pattern is counted in.

    Read from the oracle rather than written here, so that the two sides cannot
    disagree about what they are comparing.

    Returns:
        The texts, in the order the answers use.
    """
    var python_path = Python.import_module("sys").path
    python_path.insert(0, "tools")
    var helper = Python.import_module("regex_count_oracle")
    var out = List[String]()
    var given = helper.texts()
    for text in given:
        out.append(String(text))
    return out^


def numbers(line: String) raises -> List[Int]:
    """Reads one line of the oracle's answer.

    Args:
        line: The line, which is one number per text separated by spaces.

    Returns:
        The numbers.

    Raises:
        Error: If a piece of the line is not a number, which would mean the two
            sides had stopped agreeing about the format.
    """
    var out = List[Int]()
    for piece in line.split(" "):
        out.append(Int(piece))
    return out^


def tally(mut reasons: List[String], mut counts: List[Int], reason: String):
    """Counts one held out pattern under the reason it was held out for.

    A list and a linear scan rather than a dictionary, because there are about
    eight reasons and the whole point of the tally is that somebody reads it.

    Args:
        reasons: The reasons seen so far.
        counts: How many patterns each has, in the same order.
        reason: This pattern's reason.
    """
    for at in range(len(reasons)):
        if reasons[at] == reason:
            counts[at] += 1
            return
    reasons.append(reason.copy())
    counts.append(1)


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
    var texts = ask_texts()
    print(
        "asking pandas about",
        len(patterns),
        "patterns over",
        len(texts),
        "texts",
    )

    var points = List[List[UInt32]]()
    for text in texts:
        points.append(decoded(text))

    var answers = ask_pandas(patterns)

    var we_refuse = List[String]()
    var they_refuse = List[String]()
    var differ = List[String]()
    var reasons = List[String]()
    var counts = List[Int]()
    var held = 0
    var compared = 0

    for at in range(len(patterns)):
        ref pattern = patterns[at]
        ref answer = answers[at]

        var program = program_for(METHOD_COUNT, pattern)
        if not program.ok and program.gap:
            held += 1
            tally(reasons, counts, program.problem)
            continue

        compared += 1
        var theirs_refuses = answer[byte=0] == "x"
        if program.ok and theirs_refuses:
            we_refuse.append(pattern)
            continue
        if not program.ok:
            if not theirs_refuses:
                they_refuse.append(pattern)
            continue

        var want = numbers(answer)
        var machine = Machine(program)
        for which in range(len(points)):
            var ours = machine.counts(program, Span(points[which]))
            if ours != want[which]:
                differ.append(pattern)
                break

    var disagreements = len(we_refuse) + len(they_refuse) + len(differ)
    print()
    print("str.count")
    print("compared", compared, "patterns")
    print("held out", held, "patterns firepanda cannot answer yet")
    for at in range(len(reasons)):
        print("   ", counts[at], reasons[at])

    report("firepanda answers and pandas raises:", we_refuse, compared)
    report("pandas answers and firepanda refuses:", they_refuse, compared)
    report("both answer and the counts differ:", differ, compared)

    print(
        "agreement",
        (compared - disagreements) * 10000 // compared,
        "in ten thousand,",
        disagreements,
        "disagreements",
    )

    if disagreements != 0:
        raise Error(
            String(
                disagreements,
                (
                    " disagreements about how many times a pattern matches,"
                    " against a ceiling of zero"
                ),
            )
        )
