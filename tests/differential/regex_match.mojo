"""Patterns nobody wrote, run against text, against pandas' own answers.

The differential next door asks which of the two engines pandas picks. This one
picks up where that leaves off and asks what the engine then says, which is a
question with a different shape: routing is one bit per pattern and matching is
one bit per pattern and text, and a pattern can be routed perfectly and then
answered wrongly in a way no amount of routing agreement would show.

Two things are compared for every pattern.

Whether the pattern runs at all. RE2 refuses six constructs Python reads, and
pandas hands those patterns to RE2 anyway, so the caller gets an Arrow error out
of `str.contains`. Those refusals are not an implementation gap here, they are
the specification, and a compiler that quietly ran a possessive quantifier would
be answering where pandas raises. The refusal is compared as its own answer.

What it matches. Sixteen pieces of text, chosen in `tools/regex_match_oracle.py`
so that every measured difference between the two engines has something to bite
on: Arabic Indic digits for `\\d`, a vertical tab for `\\s`, a trailing newline
for `$`.

All of it three times over, once for each of `contains`, `match` and
`fullmatch`, because upstream answers the last two by rewriting the pattern and
asking the first, and a rewrite is exactly the kind of thing that is right on
the patterns somebody thought to write down and wrong on the ones nobody did. A
pattern opening with a flag group, a pattern already carrying its own anchor, a
pattern ending in a backslash: each of those goes down a different arm of the
rewrite, and the corpus produces all three in quantity. The rewrite also moves
patterns across the line between answered and refused in both directions, so the
two refusal comparisons are worth as much here as the match comparison is.

The ceiling is zero. A wrong answer here is not a refusal a caller can see, it
is a column of booleans that looks exactly like a right one.

A pattern is set aside when firepanda's own compiler says the refusal is a gap
here rather than something RE2 refuses too, and the reasons are tallied in the
report so that setting a pattern aside is a number somebody watches rather than
a silence. Asking the compiler rather than looking at the pattern text is the
part worth copying: a harness that held out every pattern with `(?i)` in it
would also hold out the ones where `(?i)` failed to parse and the pattern was
therefore an ordinary RE2 pattern, and it would have to be kept in step with the
compiler by hand.

The reasons are six. Python's grammar cannot read it, which is `\\p{L}` and
everything else that is RE2 syntax and not Python syntax, and which document 77
section 8 has as the largest single gap in the component. It routes to Python's
`re`, whose engine is not written. It reads a character class or a count the way
Python does and RE2 reads the same text differently. It names a character and
the Unicode name table is missing. It folds case and the folding table is
missing. It asks for `\\B`, which RE2 answers between bytes rather than between
characters.

Usage:
    pixi run differential-regex-match
    pixi run differential-regex-match -- --cases 40000 --seed 7
"""

from std.collections.span import Span
from std.python import Python, PythonObject
from std.sys import argv

from firepanda.kernel.regex.method import (
    METHOD_CONTAINS,
    METHOD_FULLMATCH,
    METHOD_MATCH,
    program_for,
)
from firepanda.kernel.regex.parse import decoded
from firepanda.kernel.regex.pike import Machine
from regex_corpus import corpus, report

comptime CASES = 30000
"""How many generated patterns when nobody says otherwise.

The same number the routing differential uses, on the same corpus, so that the
two reports are about the same set of patterns and a pattern named in one can be
looked up in the other.
"""

comptime SEED = 1
"""The default seed, fixed so that a failure can be reproduced by whoever reads
the report."""


def ask_pandas(patterns: List[String], method: String) raises -> List[String]:
    """What pandas answers for every pattern over every text.

    One call across the boundary for the whole batch. The work per pattern is a
    parse, a routing decision and sixteen short matches, and the crossing costs
    more than any of that.

    Args:
        patterns: The corpus.
        method: Which accessor method to ask, of the three that ask the engine
            this question.

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
    var helper = Python.import_module("regex_match_oracle")

    var out = List[String]()
    for line in String(helper.answers(batch, PythonObject(method))).split("\n"):
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
    var helper = Python.import_module("regex_match_oracle")
    var out = List[String]()
    var given = helper.texts()
    for text in given:
        out.append(String(text))
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


def sweep(
    method: UInt8,
    name: String,
    patterns: List[String],
    points: List[List[UInt32]],
) raises -> Int:
    """Runs the whole corpus through one of the three methods and reports it.

    The firepanda side goes through `program_for`, which is the same call the
    accessor makes, so what is compared is the rewrite and the routing order as
    well as the engine. Reaching past it and compiling the pattern here would
    leave the rewrite untested, and the rewrite is the only thing that differs
    between the three sweeps.

    Args:
        method: Which of the three, as the code the rewrite takes.
        name: The same one as pandas spells it, for the oracle and the report.
        patterns: The corpus.
        points: Every text, already read as code points, since each of them is
            read once for the whole run rather than once per pattern.

    Returns:
        How many patterns disagreed, of any of the three kinds.

    Raises:
        Error: If pandas could not be asked.
    """
    var answers = ask_pandas(patterns, name)

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

        var program = program_for(method, pattern)
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
    print("str.", name, sep="")
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
    return disagreements


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
        "texts, three ways",
    )

    var points = List[List[UInt32]]()
    for text in texts:
        points.append(decoded(text))

    var names = List[String]()
    names.append(String("contains"))
    names.append(String("match"))
    names.append(String("fullmatch"))
    var methods = List[UInt8]()
    methods.append(METHOD_CONTAINS)
    methods.append(METHOD_MATCH)
    methods.append(METHOD_FULLMATCH)

    var disagreements = 0
    for at in range(len(names)):
        disagreements += sweep(methods[at], names[at], patterns, points)

    if disagreements != 0:
        raise Error(
            String(
                disagreements,
                (
                    " disagreements about what a pattern matches, against a"
                    " ceiling of zero"
                ),
            )
        )
