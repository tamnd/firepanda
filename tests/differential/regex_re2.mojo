"""Patterns nobody wrote, read with RE2's grammar, against RE2's own opinion.

The four differentials next door all ask what an engine answers once a pattern
has compiled. This one asks the question underneath those: would RE2 compile it
at all. That is a question worth its own program because for most of the corpus
it is the only question there is. Nineteen thousand six hundred and fifteen of
the thirty thousand patterns are ones Python's grammar cannot read, which is how
they reach Arrow in pandas in the first place, and seventeen thousand nine
hundred and seventy six of those are ones RE2 will not read either. For every
one of them the entire answer a caller gets is a refusal, and until
`firepanda/kernel/regex/re2.mojo` existed firepanda gave the wrong one.

Two things are compared for every pattern.

Whether RE2 reads it. Answering no to a pattern RE2 reads is a new wrong
refusal, which is a column somebody could have had turned into an exception, and
the ceiling for that is zero. Answering yes to a pattern RE2 refuses is the
other direction and is not a failure: the reader is deliberately biased that way
and says so, and a pattern it is unsure about leaves the caller exactly where
they already were. Those are counted and printed and the number is expected to
fall as later slices land, but they do not fail the run.

Why it refuses it. RE2 has eleven ways of saying no and the reader has a
sentence of its own for each, so the sentences are mapped back to RE2's kinds
and compared. Getting the sentence wrong is not as bad as getting the verdict
wrong, but it is a refusal that tells somebody the wrong thing about their own
pattern, and it is just as cheap to be right, so the ceiling for that is zero as
well.

The two Unicode class spellings are the one construct the reader will not judge,
because telling a script name RE2 knows from one it does not needs a table this
library has not got yet. Every pattern holding one comes back as a pattern RE2
reads, which lands in the unsure bucket rather than in either ceiling.

Usage:
    pixi run differential-regex-re2
    pixi run differential-regex-re2 -- --cases 40000 --seed 7
"""

from std.python import Python, PythonObject
from std.sys import argv

from firepanda.kernel.regex.re2 import re2_reads

from regex_corpus import corpus, report

comptime CASES = 30000
"""How many generated patterns when nobody says otherwise.

The same number the other four use, on the same corpus, so that a pattern named
in one report can be looked up in another.
"""

comptime SEED = 1
"""The default seed, fixed so that a failure can be reproduced by whoever reads
the report."""


def ask_arrow(patterns: List[String]) raises -> List[String]:
    """What RE2 says about every pattern.

    One call across the boundary for the whole batch, because the work per
    pattern is a compile of a few dozen characters and the crossing costs more
    than the compile.

    Args:
        patterns: The corpus.

    Returns:
        One line per pattern, each either `ok` or an `x` and the kind of
        refusal.

    Raises:
        Error: If the helper could not be reached, or answered the wrong number
            of times.
    """
    var batch = Python.list()
    for pattern in patterns:
        batch.append(PythonObject(pattern))

    var python_path = Python.import_module("sys").path
    python_path.insert(0, "tools")
    var helper = Python.import_module("regex_re2_oracle")

    var out = List[String]()
    for line in String(helper.answers(batch)).split("\n"):
        out.append(String(line))
    if len(out) != len(patterns):
        raise Error(
            String(
                "RE2 answered about ",
                len(out),
                " patterns and there are ",
                len(patterns),
            )
        )
    return out^


def kind_of(said: String) -> String:
    """Which of RE2's eleven complaints one of the reader's sentences is.

    The reader says things in this library's own voice, because a caller reading
    a message is a caller of firepanda and `invalid perl operator` is not a
    sentence about anything they wrote. That leaves a mapping to keep, and this
    is it, and it is here rather than in the reader so that the reader owes
    nothing to the harness.

    Two of the sentences go to the same kind, because RE2 uses one complaint for
    a class name it does not know and for a range written the wrong way round,
    and telling those two apart is worth doing even though RE2 does not.

    Args:
        said: What the reader said.

    Returns:
        RE2's kind, or the sentence itself when it is one this mapping has not
        been told about, which will then fail loudly rather than quietly match.
    """
    if said == "RE2 has no group written that way":
        return String("invalid perl operator")
    if said == "RE2 has no such escape":
        return String("invalid escape sequence")
    if said == "there is nothing here for that repeat to repeat":
        return String("no argument for repetition operator")
    if said == "RE2 will not repeat a repeat":
        return String("bad repetition operator")
    if said == "a class is never closed":
        return String("missing ]")
    if said == "RE2 will not take that group name":
        return String("invalid named capture group")
    if said == "a bracket is closed that nothing opened":
        return String("unexpected )")
    if said == "a bracket is opened that nothing closes":
        return String("missing )")
    if said == "RE2 will not repeat that many times":
        return String("invalid repetition size")
    if said == "RE2 has no such character class":
        return String("invalid character class range")
    if said == "a range in a class runs backwards":
        return String("invalid character class range")
    if said == "a backslash is the last thing in the pattern":
        return String("trailing \\")
    return said.copy()


def tally(mut names: List[String], mut counts: List[Int], name: String):
    """Counts one thing under the name it goes under.

    A list and a linear scan rather than a dictionary, because there are eleven
    names at most and the whole point of the tally is that somebody reads it.

    Args:
        names: The names seen so far.
        counts: How many of each, in the same order.
        name: This one's name.
    """
    for at in range(len(names)):
        if names[at] == name:
            counts[at] += 1
            return
    names.append(name.copy())
    counts.append(1)


def main() raises:
    """Reads the corpus with RE2's grammar and compares against RE2.

    Raises:
        Error: If RE2 could not be asked, or if anything disagreed.
    """
    var cases = CASES
    var seed = UInt64(SEED)
    var args = argv()
    for at in range(1, len(args)):
        if args[at] == "--cases" and at + 1 < len(args):
            cases = Int(args[at + 1])
        elif args[at] == "--seed" and at + 1 < len(args):
            seed = UInt64(Int(args[at + 1]))

    var patterns = corpus(cases, seed)
    var answers = ask_arrow(patterns)

    var wrongly_refused = List[String]()
    var wrong_reason = List[String]()
    var unsure = 0
    var missed = 0
    var reads = 0
    var refuses = 0
    var kinds = List[String]()
    var counts = List[Int]()
    var misses = List[String]()
    var miss_counts = List[Int]()

    for at in range(len(patterns)):
        ref pattern = patterns[at]
        ref answer = answers[at]
        var they_read = answer == "ok"
        var read = re2_reads(pattern)

        if read.ok and they_read:
            reads += 1
            continue
        if read.ok:
            # The safe direction, and it has two halves worth telling apart. A
            # pattern the reader declined to judge is one construct away from
            # being answerable and needs a name table. A pattern it read and RE2
            # did not is a rule it has not been taught, which is a smaller and
            # more embarrassing thing. Neither leaves the caller worse off than
            # they were, so neither fails the run.
            if read.unsure:
                unsure += 1
            else:
                missed += 1
                tally(misses, miss_counts, String(answer[byte=2:]))
            continue
        if they_read:
            wrongly_refused.append(pattern)
            continue
        refuses += 1
        tally(kinds, counts, kind_of(read.problem))
        if kind_of(read.problem) != String(answer[byte=2:]):
            wrong_reason.append(pattern)

    print("corpus", len(patterns), "patterns")
    print("RE2 reads and the reader reads", reads)
    print("RE2 refuses and the reader refuses", refuses)
    for at in range(len(kinds)):
        print("   ", counts[at], kinds[at])
    print("RE2 refuses and the reader declined to judge", unsure)
    print("RE2 refuses and the reader read it", missed)
    for at in range(len(misses)):
        print("   ", miss_counts[at], misses[at])

    report("RE2 reads and the reader refuses:", wrongly_refused, len(patterns))
    report("both refuse and the reason differs:", wrong_reason, len(patterns))

    var disagreements = len(wrongly_refused) + len(wrong_reason)
    print(
        "agreement",
        (len(patterns) - disagreements) * 10000 // len(patterns),
        "in ten thousand,",
        disagreements,
        "disagreements",
    )
    if disagreements != 0:
        raise Error(
            String(
                "the RE2 reader disagreed with RE2 about ",
                disagreements,
                " patterns",
            )
        )
