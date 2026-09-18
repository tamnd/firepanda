"""Patterns nobody wrote, run against text, against Python's own answers.

The match differential asks what RE2 says, because for six of the accessor's
nine pattern methods RE2 is what pandas asks. This one asks the other three.
`extract`, `extractall` and `findall` never reach Arrow: pandas compiles the
pattern with `re` and loops in Python, so the same letter in the same accessor
covers 63 characters or 138558 of them depending on the method name. Document 81
is where that was measured.

What is compared is the same two things the match differential compares, whether
the pattern runs and what it matches, and the second one is the point. A pattern
holding none of `\\w`, `\\d`, `\\s`, `\\b` or `$` answers the same for both
engines, and the corpus is full of patterns holding one of them, so a compiler
that quietly used RE2's tables would agree here on most of the corpus and
disagree on the part that matters.

`findall` is the method asked, because it takes a pattern with no groups and
answers a list whose length says whether anything matched. The other two want a
pattern with a group in it and answer a frame, which is a different comparison
and belongs with the accessor work rather than with the engine.

The ceiling is zero, for the reason it is zero next door. A wrong answer here is
not a refusal a caller can see, it is a list that looks exactly like a right one.

A pattern is set aside when firepanda's own compiler says the refusal is a gap
here rather than something Python refuses too, and the reasons are tallied so
that setting a pattern aside is a number somebody watches. The reasons are
shorter than the RE2 ones and every one of them is this library falling short of
an engine that reads the pattern: a lookaround, a backreference, a conditional,
an atomic group, a possessive quantifier and a named character.

Usage:
    pixi run differential-regex-python
    pixi run differential-regex-python -- --cases 40000 --seed 7
"""

from std.collections.span import Span
from std.python import Python, PythonObject
from std.sys import argv

from firepanda.kernel.regex.parse import decoded, parse_pattern
from firepanda.kernel.regex.pike import Machine
from firepanda.kernel.regex.program import compile_program
from firepanda.kernel.regex.route import ENGINE_PYTHON
from regex_corpus import corpus, report

comptime CASES = 30000
"""How many generated patterns when nobody says otherwise.

The same number and the same corpus the other two regex differentials use, so
that a pattern named in one report can be looked up in the others.
"""

comptime SEED = 1
"""The default seed, fixed so that a failure can be reproduced by whoever reads
the report."""


def ask_pandas(patterns: List[String]) raises -> List[String]:
    """What pandas answers for every pattern over every text.

    One call across the boundary for the whole batch, because the work per
    pattern is a parse and sixteen short matches and the crossing costs more
    than any of that.

    Args:
        patterns: The corpus.

    Returns:
        One line per pattern, each either a single `x` for a call that raised or
        one character per text.

    Raises:
        Error: If the helper could not be reached, or answered the wrong number
            of times.
    """
    var batch = Python.list()
    for pattern in patterns:
        batch.append(PythonObject(pattern))

    var python_path = Python.import_module("sys").path
    python_path.insert(0, "tools")
    var helper = Python.import_module("regex_python_oracle")

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
    """The text every pattern is run against.

    Read from the oracle rather than written here, so that the two sides cannot
    disagree about what they are comparing.

    Returns:
        The texts, in the order the answers use.
    """
    var python_path = Python.import_module("sys").path
    python_path.insert(0, "tools")
    var helper = Python.import_module("regex_python_oracle")
    var out = List[String]()
    var given = helper.texts()
    for text in given:
        out.append(String(text))
    return out^


def tally(mut reasons: List[String], mut counts: List[Int], reason: String):
    """Counts one held out pattern under the reason it was held out for.

    A list and a linear scan rather than a dictionary, because there are about
    ten reasons and the whole point of the tally is that somebody reads it.

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
        "texts, through str.findall",
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

        # No rewrite and no anchoring on this side. `str.match` and
        # `str.fullmatch` are `str.contains` with the pattern rewritten, and
        # none of the three methods this engine answers does anything of the
        # kind, so the pattern reaches the compiler as the caller wrote it.
        var program = compile_program(parse_pattern(pattern), ENGINE_PYTHON)
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

        var machine = Machine(program)
        for which in range(len(points)):
            var ours = machine.matches(program, Span(points[which]))
            var theirs = answer[byte=which] == "y"
            if ours != theirs:
                differ.append(pattern)
                break

    var disagreements = len(we_refuse) + len(they_refuse) + len(differ)
    print()
    print("str.findall")
    print("compared", compared, "patterns")
    print("held out", held, "patterns firepanda cannot answer yet")
    for at in range(len(reasons)):
        print("   ", counts[at], reasons[at])

    report("firepanda answers and pandas raises:", we_refuse, compared)
    report("pandas answers and firepanda refuses:", they_refuse, compared)
    report("both answer and the answers differ:", differ, compared)

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
                    " disagreements about what a pattern matches for Python's"
                    " engine, against a ceiling of zero"
                ),
            )
        )
