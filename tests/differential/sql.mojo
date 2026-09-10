"""Every statement in DuckDB's test corpus, through both parsers.

The compatibility claim is that firepanda accepts what DuckDB accepts. This is
the thing that turns that sentence into a number. It reads the 71,438 statements
in DuckDB's own `test/sql` tree at the commit the vendored grammar came from,
parses each one with firepanda and with DuckDB, and reports every statement the
two disagree about.

Only DuckDB's parser runs. `extract_statements` splits a string into statements
and builds nothing else, so a missing column or a function with no overload is
not an error on that side, which is exactly where the line has to be. firepanda
would refuse those in the binder, and a binder refusal is not a compatibility
failure.

Some divergence is expected and is not a bug in this repository. The grammar is
vendored from DuckDB's development branch and the oracle is whatever DuckDB the
differential environment resolved, so a statement upstream taught its parser
last month is one this build has never heard of. Those show up here as firepanda
accepting something DuckDB rejects, which is the harmless direction, and they are
listed rather than hidden.

Usage:
    pixi run corpus            # once, to fetch and extract
    pixi run differential-sql
"""

from std.python import Python, PythonObject
from std.sys import argv

from firepanda.sql import Grammar
from firepanda.sql.matcher import parse

comptime ACCEPTED = Byte(ord("1"))
"""DuckDB parsed it."""

comptime REJECTED = Byte(ord("0"))
"""DuckDB called it a syntax error."""

comptime SHOWN = 40
"""How many disagreements of each kind to print before summarizing the rest."""

comptime LONGEST = 120
"""How much of a statement to print before cutting it off.

The corpus has a 57 KB generated expression in it, and a log nobody can scroll
through is a log nobody reads.
"""

comptime DUCKDB_ONLY_CEILING = 2
"""How many statements DuckDB may parse that firepanda does not.

This is the direction that matters and the target for it is zero. The two that
are left are both the recursion depth guard in `firepanda/sql/matcher.mojo`
firing: one is a 57 KB generated expression and the other nests a subquery a
hundred deep. Closing them means turning the matcher into an explicit stack
machine, which is a piece of work of its own, so the number sits here where it
can be lowered and not raised.
"""

comptime FIREPANDA_ONLY_CEILING = 1200
"""How many statements firepanda may parse that DuckDB does not.

This direction is not a compatibility failure on its own. The grammar comes from
DuckDB's development branch and the oracle is the last release conda-forge has,
so a statement upstream taught its parser after that release is one this build
has never heard of. `CREATE TRIGGER` alone is 357 of them. The rest are mostly
DuckDB moving an error out of the parser and into the binder when it replaced
Bison with the PEG grammar, which is the same thing firepanda will do.

So this one is a ceiling rather than a target. It exists to catch the parser
going slack, not to be driven to zero, and it should come down on its own the
next time the oracle catches up with the grammar.
"""


struct Statement(ImplicitlyCopyable, Movable):
    """One statement out of the corpus, and where it came from."""

    var origin: String
    """The path relative to `test/sql`, and the line the statement starts on."""

    var sql: String
    """The statement itself, newlines and all."""

    def __init__(out self, var origin: String, var sql: String):
        """Builds one.

        Args:
            origin: The path and line.
            sql: The statement.
        """
        self.origin = origin^
        self.sql = sql^


def read_statements(path: String) raises -> List[Statement]:
    """Reads the flat statements file that `tools/corpus.py` writes.

    The format is a header line holding the byte length and the location,
    separated by a tab, then that many bytes of SQL, then a newline. The length
    is there so that a statement spanning several lines needs no escaping and no
    scanning, which matters at seventy thousand of them.

    Args:
        path: The statements file.

    Returns:
        Every statement, in the order the file lists them.

    Raises:
        Error: If the file is not in that format.
    """
    var handle = open(path, "r")
    var data = handle.read_bytes()
    handle.close()

    var bytes = Span(data)
    var out = List[Statement]()
    var at = 0
    while at < len(bytes):
        var newline = at
        while newline < len(bytes) and bytes[newline] != Byte(ord("\n")):
            newline += 1
        if newline == len(bytes):
            raise Error("the statements file ends in the middle of a header")

        var tab = at
        while tab < newline and bytes[tab] != Byte(ord("\t")):
            tab += 1
        if tab == newline:
            raise Error(String("a header at byte ", at, " has no tab in it"))

        var length = atol(StringSlice(unsafe_from_utf8=bytes[at:tab]))
        var start = newline + 1
        var stop = start + length
        if stop > len(bytes):
            raise Error(
                String("a statement at byte ", start, " runs past the end")
            )

        out.append(
            Statement(
                String(StringSlice(unsafe_from_utf8=bytes[tab + 1 : newline])),
                String(StringSlice(unsafe_from_utf8=bytes[start:stop])),
            )
        )
        at = stop + 1

    return out^


def directory_of(origin: StringSlice) -> String:
    """Returns the top level corpus directory a location is in.

    The per directory breakdown is the useful shape of a failure list, because
    the corpus is organised by feature and a hundred failures in one directory is
    one missing feature rather than a hundred bugs.

    Args:
        origin: A path relative to `test/sql`, and a line number.

    Returns:
        The first path component.
    """
    var slash = origin.find("/")
    if slash < 0:
        return String(".")
    return String(origin[byte=0:slash])


def report(kind: StringSlice, cases: List[Statement], of: Int) -> None:
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

    var directories = List[String]()
    var counts = List[Int]()
    for item in cases:
        var directory = directory_of(item.origin)
        var known = -1
        for i in range(len(directories)):
            if directories[i] == directory:
                known = i
                break
        if known < 0:
            directories.append(directory)
            counts.append(1)
        else:
            counts[known] += 1

    for i in range(len(directories)):
        print("   ", counts[i], "in", directories[i])

    print()
    for i in range(min(SHOWN, len(cases))):
        ref item = cases[i]
        # Built once from the statement rather than sliced and assigned back over
        # itself. The short version took a slice of `sql` and put the result into
        # `sql`, so the value being overwritten was the value being read.
        var sql: String
        if item.sql.byte_length() > LONGEST:
            sql = String(StringSlice(item.sql)[byte=0:LONGEST], " ...")
        else:
            sql = item.sql
        print("   ", item.origin, "  ", sql.replace("\n", " "))
    if len(cases) > SHOWN:
        print("   ", len(cases) - SHOWN, "more")


def main() raises:
    var python_path = Python.import_module("sys").path
    python_path.insert(0, "tools")
    var helper = Python.import_module("corpus")

    var path = String(helper.location())
    var statements = read_statements(path)
    print("read", len(statements), "statements from", path)

    var theirs = String(helper.verdicts(path))
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

    var grammar = Grammar()
    var compared = 0
    var undecided = 0
    var we_reject = List[Statement]()
    var we_accept = List[Statement]()

    for i in range(len(statements)):
        var answer = answers[i]
        if answer != ACCEPTED and answer != REJECTED:
            # DuckDB refused it for a reason that was not a syntax error, so it
            # is not an oracle for this question.
            undecided += 1
            continue

        ref statement = statements[i]
        var ours = True
        try:
            _ = parse(statement.sql, grammar)
        except:
            ours = False

        compared += 1
        if ours and answer == REJECTED:
            we_accept.append(statement)
        elif not ours and answer == ACCEPTED:
            we_reject.append(statement)

    print(
        "compared",
        compared,
        "statements,",
        undecided,
        "had no parser answer from DuckDB",
    )

    report("DuckDB parses these and firepanda does not:", we_reject, compared)
    report("firepanda parses these and DuckDB does not:", we_accept, compared)

    var arguments = argv()
    if len(arguments) > 1:
        var handle = open(String(arguments[1]), "w")
        for item in we_reject:
            handle.write(
                String("duckdb-only ", item.origin, "\t", item.sql, "\n")
            )
        for item in we_accept:
            handle.write(
                String("firepanda-only ", item.origin, "\t", item.sql, "\n")
            )
        handle.close()
        print("wrote the full list to", arguments[1])

    print()
    var disagreements = len(we_reject) + len(we_accept)
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
                " statements firepanda rejects, against a ceiling of ",
                DUCKDB_ONLY_CEILING,
            )
        )
    if len(we_accept) > FIREPANDA_ONLY_CEILING:
        raise Error(
            String(
                "firepanda parses ",
                len(we_accept),
                " statements DuckDB rejects, against a ceiling of ",
                FIREPANDA_ONLY_CEILING,
            )
        )
