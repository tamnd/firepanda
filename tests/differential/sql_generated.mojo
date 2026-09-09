"""Statements nobody wrote, through both parsers.

The corpus differential in `sql.mojo` is bounded by what DuckDB's test suite
happens to contain. Whole corners of the grammar are in there once or not at
all, and a rule with no test behind it is a rule where a mistake sits until a
user finds it. This one closes that gap from the other side: it walks the
grammar itself to build statements, then asks DuckDB and firepanda about each
one and reports where they disagree.

The generator is in `firepanda/sql/generator.mojo` and the idea is old. A PEG
grammar is a recognizer read forwards and a generator read backwards, so the
same 1,187 rules that decide whether a statement parses will also write one.

Two things make the output unfair to firepanda in a way that is worth knowing
about before reading the numbers. The generator ignores negative lookahead,
because satisfying `!X` in general means solving the thing the parser is for,
so it can write a statement the grammar itself would refuse. And the vendored
grammar is not DuckDB's parser, it is a second description of the same
language that DuckDB ships for tooling, so the two do not agree everywhere.
Both show up as firepanda accepting what DuckDB rejects, which is the harmless
direction, and both are why that ceiling is loose.

The direction that matters is the other one. A statement the grammar wrote and
DuckDB accepts and firepanda rejects is a bug in the matcher, and the ceiling
for those is zero.

Usage:
    pixi run differential-sql-generated
    pixi run differential-sql-generated -- --cases 20000 --seed 7
"""

from std.python import Python, PythonObject
from std.sys import argv

from firepanda.sql import Grammar
from firepanda.sql.generator import Generator
from firepanda.sql.matcher import parse

comptime CASES = 25000
"""How many statements to generate when nobody says otherwise.

Ten seconds end to end, which is nothing next to the corpus differential this
sits beside, and enough statements to reach the rules that the corpus has one
test for or none.
"""

comptime SEED = 0x5EED
"""The default seed, so an unadorned run is the same run every time.

A fuzzer that reports something different on every commit is a fuzzer people
learn to ignore. The seed moves when somebody moves it.
"""

comptime SHOWN = 25
"""How many disagreements of each kind to print before summarizing the rest."""

comptime DUCKDB_ONLY_CEILING = 0
"""How many unexplained statements DuckDB may parse that firepanda does not.

Zero, and it should stay zero. The generator only writes what the grammar
allows, so anything DuckDB agrees with and firepanda refuses is either the
matcher reading its own grammar wrong or a defect in the grammar itself. The
second kind is listed in `known` and counted separately, so this number is the
first kind alone.
"""

comptime FIREPANDA_ONLY_CEILING = 6200
"""How many in ten thousand firepanda may parse that DuckDB does not.

A rate rather than a count, because the case count is a command line option and
a ceiling that only holds at one setting is a ceiling nobody can use.

It is high and that is expected. The generator samples the grammar rather than
the language people write, so it spends most of its time in rules DuckDB's
released parser has never been asked about, where the corpus spends almost
none. `DROP EXTENSION REPOSITORY` and `DISCONNECT` are both in the vendored
development grammar and neither is in the oracle. Add the statements the
generator writes that the grammar's own negative lookahead would have refused
and this is most of the gap.

So it is a ceiling and not a target. It catches the number moving and it should
come down on its own the next time the oracle catches up with the grammar.
"""


def known(sql: StringSlice, why: StringSlice) -> Bool:
    """Says whether a rejection is a defect in the grammar rather than in here.

    The grammar is vendored byte for byte from DuckDB and it is not the parser
    DuckDB runs, it is a second description of the same language kept for
    tooling. The two have drifted in a few places, and where they have, a
    faithful reading of the grammar is the right behaviour for this repository
    even though it disagrees with the oracle. Those cases are listed here with
    the reason, so that the ceiling above can stay at zero and mean something.

    Args:
        sql: The generated statement.
        why: The parser error firepanda gave for it.

    Returns:
        Whether this is one of the known ones.
    """
    # `CopyFileName <- ... / CopyFileNameIdentifier / CopyFileNameIdentifierColId`
    # puts a bare `Identifier` ahead of `Identifier '.' ColId`, and PEG choice
    # is ordered, so the bare one always wins and the qualified alternative can
    # never be reached. `COPY t TO a.b` is a statement DuckDB's own parser takes
    # and its own grammar cannot. Reported upstream, see
    # docs/specs/sql/13-open-questions.md question 12.
    #
    # `COPY` anywhere rather than a leading one, because `PREPARE x AS COPY ...`
    # is the same defect one rule further in. Matching on the statement and the
    # blamed token together is loose enough to catch that and tight enough that
    # it would have to be a copy statement failing at a dot to be missed.
    if "COPY " in sql and 'near "."' in why:
        return True
    return False


def generate(count: Int, seed: UInt64, grammar: Grammar) raises -> List[String]:
    """Writes some statements out of the grammar.

    Args:
        count: How many to write.
        seed: What to seed the generator with.
        grammar: A loaded grammar.

    Returns:
        The statements, in the order they were written.

    Raises:
        Error: If the generator failed on one of them.
    """
    var out = List[String]()
    var generator = Generator(grammar, seed)
    for _ in range(count):
        out.append(generator.statement())
    return out^


def ask_duckdb(statements: List[String]) raises -> String:
    """Asks DuckDB about every statement at once.

    One call across the whole batch rather than one call each, because the
    bridge costs more per crossing than DuckDB costs per statement.

    Args:
        statements: The statements.

    Returns:
        One byte per statement: "1" parsed, "0" was a syntax error, anything
        else means DuckDB refused it for some other reason and is not an
        oracle for this question.

    Raises:
        Error: If the helper could not be reached.
    """
    var batch = Python.list()
    for item in statements:
        batch.append(PythonObject(item))
    var python_path = Python.import_module("sys").path
    python_path.insert(0, "tools")
    var helper = Python.import_module("corpus")
    return String(helper.verdicts_for(batch))


def report(kind: StringSlice, cases: List[String], of: Int) -> None:
    """Prints one side of the disagreement list.

    Args:
        kind: What this list is, for the heading.
        cases: The statements that disagreed this way.
        of: How many statements were compared in total.
    """
    if len(cases) == 0:
        return

    print()
    print(kind, len(cases), "of", of)
    print()
    for i in range(min(SHOWN, len(cases))):
        var sql = cases[i]
        if sql.byte_length() > 140:
            sql = String(StringSlice(sql)[byte=0:140], " ...")
        print("   ", sql)
    if len(cases) > SHOWN:
        print("   ", len(cases) - SHOWN, "more")


def option(name: StringSlice, fallback: Int) raises -> Int:
    """Reads a numeric command line option.

    Args:
        name: The option, leading dashes and all.
        fallback: What to use when it was not given.

    Returns:
        The value.

    Raises:
        Error: If the option was given without one.
    """
    var arguments = argv()
    for i in range(len(arguments) - 1):
        if arguments[i] == name:
            return Int(arguments[i + 1])
    for i in range(len(arguments)):
        if arguments[i] == name:
            raise Error(String(name, " needs a value"))
    return fallback


def main() raises:
    var cases = option("--cases", CASES)
    var seed = UInt64(option("--seed", Int(SEED)))

    var grammar = Grammar()
    var statements = generate(cases, seed, grammar)
    print("generated", len(statements), "statements from seed", seed)

    var theirs = ask_duckdb(statements)
    if theirs.byte_length() != len(statements):
        raise Error(
            String(
                "DuckDB answered for ",
                theirs.byte_length(),
                " statements and there are ",
                len(statements),
            )
        )
    var answers = theirs.as_bytes()

    var compared = 0
    var undecided = 0
    var explained = 0
    var we_reject = List[String]()
    var we_accept = List[String]()

    for i in range(len(statements)):
        var answer = answers[i]
        if answer != Byte(ord("1")) and answer != Byte(ord("0")):
            undecided += 1
            continue

        var ours = True
        var why = String()
        try:
            _ = parse(statements[i], grammar)
        except error:
            ours = False
            why = String(error)

        compared += 1
        if ours and answer == Byte(ord("0")):
            we_accept.append(statements[i])
        elif not ours and answer == Byte(ord("1")):
            if known(statements[i], why):
                explained += 1
            else:
                we_reject.append(statements[i])

    print(
        "compared",
        compared,
        "statements,",
        undecided,
        "had no parser answer from DuckDB",
    )
    if explained != 0:
        print(explained, "were known defects in the grammar itself")

    report("DuckDB parses these and firepanda does not:", we_reject, compared)
    report("firepanda parses these and DuckDB does not:", we_accept, compared)

    print()
    var disagreements = len(we_reject) + len(we_accept)
    var permissive = len(we_accept) * 10000 // compared
    print(
        "agreement",
        (compared - disagreements) * 10000 // compared,
        "in ten thousand,",
        disagreements,
        "disagreements",
    )

    if len(we_reject) > DUCKDB_ONLY_CEILING:
        raise Error(
            String(
                "DuckDB parses ",
                len(we_reject),
                (
                    " generated statements firepanda rejects, against a"
                    " ceiling of "
                ),
                DUCKDB_ONLY_CEILING,
            )
        )
    if permissive > FIREPANDA_ONLY_CEILING:
        raise Error(
            String(
                "firepanda parses ",
                permissive,
                " in ten thousand that DuckDB rejects, against a ceiling of ",
                FIREPANDA_ONLY_CEILING,
            )
        )
