"""Patterns nobody wrote, replaced in text, against pandas' own rows.

The third of the regular expression differentials and the first whose answer is
text. The other two compare a bit and a number, and both of them stop at the
engine: a pattern that matches wrongly fails there and a pattern that matches
rightly passes. This one carries on past the engine into two pieces of machinery
neither of the others touches.

The first is the scan. Arrow replaces down a row with a different loop from the
one it counts with, in the same library, on the same pattern. It does not cut the
row after a match, so an anchor keeps seeing the row it started with. It moves
its cursor a character at a time where the count moves a byte at a time. And when
it refuses a match of no width it copies one character across rather than simply
stepping. Document 80 says where each of those was measured and
`firepanda/kernel/regex/replace.mojo` states them beside the loop that follows
them.

The second is the replacement. Arrow reads it with RE2's rewrite grammar, in
which a pair of backslashes is one backslash and a backslash and a digit is a
group, and a group reference is the only thing in this library that asks the
engine where a match started rather than only whether it ended. A pattern can be
compiled correctly, scanned correctly and still written out wrongly if a capture
slot is kept for the wrong thread.

So three replacements are asked for every pattern. A plain marker is the scan on
its own. A marker holding the whole match is how much of the row each match
covered. A bare group reference is the capture slots, and it is also the one
that is refused when the pattern has no group, which makes the refusals worth
comparing as well as the answers.

The ceiling is zero, for the reason the count differential gives: a row written
out wrongly is not a refusal a caller can see, it is text that looks exactly like
text.

A pattern is set aside when firepanda's own compiler says the refusal is a gap
here rather than something RE2 refuses too, and the reasons are tallied in the
report, exactly as in the other two.

Usage:
    pixi run differential-regex-replace
    pixi run differential-regex-replace -- --cases 40000 --seed 7
"""

from std.collections.span import Span
from std.python import Python, PythonObject
from std.sys import argv

from firepanda.kernel.regex.method import METHOD_REPLACE, program_for
from firepanda.kernel.regex.parse import decoded
from firepanda.kernel.regex.pike import Machine
from firepanda.kernel.regex.replace import parse_rewrite, replaced
from regex_corpus import corpus, report

comptime CASES = 30000
"""How many generated patterns when nobody says otherwise.

The same number the other two regular expression differentials use, on the same
corpus, so that a pattern named in one report can be looked up in the others.
"""

comptime SEED = 1
"""The default seed, fixed so that a failure can be reproduced by whoever reads
the report."""


def replacements() -> List[String]:
    """The three replacements every pattern is asked for.

    Written here as well as in the oracle rather than sent across the boundary,
    because the two sides have to mean the same three things by them and a list
    that travels is a list that can arrive changed. They are short enough that
    writing them twice is the cheaper safety.

    Returns:
        The replacements, in the order the answers use.
    """
    var out = List[String]()
    out.append(String("#"))
    out.append(String("[\\0]"))
    out.append(String("\\1"))
    return out^


def ask_pandas(patterns: List[String]) raises -> List[String]:
    """What pandas writes out for every pattern and every replacement.

    One call across the boundary for the whole batch, for the reason the other
    two differentials give: the work per pattern is a parse, a routing decision
    and a handful of short scans, and the crossing costs more than any of that.

    Args:
        patterns: The corpus.

    Returns:
        One line per pattern, each holding three sweeps separated by vertical
        bars, each sweep either a single `x` for a call that raised or one
        hexadecimal string per text.

    Raises:
        Error: If the helper could not be reached, or answered the wrong number
            of times.
    """
    var batch = Python.list()
    for pattern in patterns:
        batch.append(PythonObject(pattern))

    var python_path = Python.import_module("sys").path
    python_path.insert(0, "tools")
    var helper = Python.import_module("regex_replace_oracle")

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
    """The text every pattern is replaced in.

    Read from the oracle rather than written here, so that the two sides cannot
    disagree about what they are comparing.

    Returns:
        The texts, in the order the answers use.
    """
    var python_path = Python.import_module("sys").path
    python_path.insert(0, "tools")
    var helper = Python.import_module("regex_replace_oracle")
    var out = List[String]()
    var given = helper.texts()
    for text in given:
        out.append(String(text))
    return out^


def split_on(line: String, separator: StringSlice) raises -> List[String]:
    """Cuts one line of the oracle's answer into its pieces.

    Args:
        line: The line, or one sweep of it.
        separator: The vertical bar between sweeps, or the space between texts.

    Returns:
        The pieces, in order.
    """
    var out = List[String]()
    for piece in line.split(separator):
        out.append(String(piece))
    return out^


def hexed(bytes: Span[UInt8, _]) -> String:
    """Writes a row the way the oracle writes it.

    The answers cross the boundary as one string and a row can hold a newline, a
    space or a tab, so both sides write hexadecimal rather than text and the
    comparison is a comparison of two strings of digits.

    Args:
        bytes: The row.

    Returns:
        Two lowercase digits per byte.
    """
    var table = String("0123456789abcdef")
    var digits = table.as_bytes()
    var out = List[UInt8]()
    for i in range(len(bytes)):
        out.append(digits[Int(bytes[i] >> 4)])
        out.append(digits[Int(bytes[i] & 15)])
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def holds_a_backslash(text: String) -> Bool:
    """Whether a replacement has anything in it the two grammars read apart.

    Args:
        text: The replacement.

    Returns:
        True if there is a backslash in it.
    """
    for byte in text.as_bytes():
        if byte == UInt8(ord("\\")):
            return True
    return False


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
    var repls = replacements()
    print(
        "asking pandas about",
        len(patterns),
        "patterns over",
        len(texts),
        "texts and",
        len(repls),
        "replacements",
    )

    var points = List[List[UInt32]]()
    var bytes = List[List[UInt8]]()
    for text in texts:
        points.append(decoded(text))
        var row = List[UInt8]()
        for byte in text.as_bytes():
            row.append(byte)
        bytes.append(row^)

    var answers = ask_pandas(patterns)

    var we_refuse = List[String]()
    var they_refuse = List[String]()
    var differ = List[String]()
    var reasons = List[String]()
    var counts = List[Int]()
    var held = 0
    var compared = 0
    var unreadable = 0
    var elsewhere = 0

    var offsets = List[Int]()
    var found = List[Int32]()
    var out = List[UInt8]()

    for at in range(len(patterns)):
        ref pattern = patterns[at]

        var program = program_for(METHOD_REPLACE, pattern)
        if not program.ok and program.gap:
            held += 1
            tally(reasons, counts, program.problem)
            continue

        compared += 1
        var sweeps = split_on(answers[at], "|")
        # The first replacement is a marker holding no backslash, so the only
        # thing that can refuse it is the pattern, which makes that one sweep
        # the answer to whether pandas ran the pattern at all.
        var they_refuse_it = sweeps[0] == "x"
        if program.ok and they_refuse_it:
            we_refuse.append(pattern)
            continue
        if not program.ok:
            if not they_refuse_it:
                they_refuse.append(pattern)
            continue

        var machine = Machine(program)
        var wrong = False
        var empty = pattern.byte_length() == 0
        for which in range(len(repls)):
            if empty and holds_a_backslash(repls[which]):
                # An empty pattern is the one shape pandas sends to Python's
                # engine rather than to Arrow, so the replacement is read by
                # Python's grammar there, in which a backslash and a zero is a
                # null character rather than the whole match. The two grammars
                # agree while there is no backslash to disagree about, which is
                # why only the sweeps holding one are set aside. The binding
                # refuses this shape rather than answering it, and
                # `python/firepanda/_pandas.py` says so where it does.
                elsewhere += 1
                continue
            if sweeps[which] == "u":
                # pandas answered and what it answered is not text, which RE2
                # can do because it reads a zero width assertion between bytes.
                # There is nothing on the other side to compare with, so the
                # sweep is set aside and counted.
                unreadable += 1
                continue
            var rewrite = parse_rewrite(repls[which], program.groups)
            var they_refuse_this = sweeps[which] == "x"
            if not rewrite.ok:
                if not they_refuse_this:
                    they_refuse.append(pattern)
                    break
                continue
            if they_refuse_this:
                we_refuse.append(pattern)
                break

            var want = split_on(sweeps[which], " ")
            for row in range(len(points)):
                replaced(
                    program,
                    rewrite,
                    Span(bytes[row]),
                    Span(points[row]),
                    machine,
                    offsets,
                    found,
                    out,
                )
                if hexed(Span(out)) != want[row]:
                    differ.append(pattern)
                    wrong = True
                    break
            if wrong:
                break

    var disagreements = len(we_refuse) + len(they_refuse) + len(differ)
    print()
    print("str.replace")
    print("compared", compared, "patterns")
    print("held out", held, "patterns firepanda cannot answer yet")
    print(
        "set aside",
        unreadable,
        "sweeps pandas wrote as something other than text and",
        elsewhere,
        "it answered out of the other engine",
    )
    for at in range(len(reasons)):
        print("   ", counts[at], reasons[at])

    report("firepanda answers and pandas raises:", we_refuse, compared)
    report("pandas answers and firepanda refuses:", they_refuse, compared)
    report("both answer and the rows differ:", differ, compared)

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
                    " disagreements about what a row looks like after a pattern"
                    " has been replaced in it, against a ceiling of zero"
                ),
            )
        )
