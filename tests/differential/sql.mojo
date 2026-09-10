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

Every statement firepanda parses also goes through the transformer and the
printer, twice, and the two printings have to agree. That is the round trip the
printer exists for, and running it here rather than on a handful of hand written
queries is the point: a precedence mistake or a clause the transformer drops
shows up as text that says something else, on real SQL, at corpus scale. A
statement outside the surface firepanda covers refuses by name and is counted
rather than failed, which is what the table in `firepanda/sql/unsupported.mojo`
is for.

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

from firepanda.sql import (
    Ast,
    Grammar,
    NO_REFUSAL,
    Refusal,
    Transform,
    feature_of,
    print_stmt,
    sql_support,
)
from firepanda.sql.matcher import parse

comptime ACCEPTED = Byte(ord("1"))
"""DuckDB parsed it."""

comptime REJECTED = Byte(ord("0"))
"""DuckDB called it a syntax error."""

comptime REFUSAL = "Not Implemented Error:"
"""What a refusal starts with, and what tells one from a crash.

The whole point of the table in `firepanda/sql/unsupported.mojo` is that a
harness can make this distinction. A statement outside the surface firepanda
covers is supposed to refuse, and a statement that refuses is not a failure
here. One that fails any other way is.
"""

comptime SYNTAX = "Parser Error:"
"""What the matcher says when a rule did not match the text.

Now that the transformer is aimed at the whole statement rule, what is left in
this bucket is text the grammar itself will not take. The corpus is a test suite
and part of what it tests is that bad SQL is rejected, so a hundred or so of
these are the corpus checking DuckDB's error messages and firepanda agreeing
with it. That is not a bug and it is not a refusal either, so it keeps a count
of its own.
"""

comptime EXHAUSTED = "memory exhausted"
"""What the matcher's depth guard says, in DuckDB's own words.

Told apart from an ordinary syntax error because the two mean different things
about the printer. A syntax error in printed text means the printer wrote
something wrong. This means it wrote something right and too deeply nested for
our own matcher to read back, which is a limit of the matcher.
"""

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

comptime BROKEN_CEILING = 0
"""How many statements the transformer may fail on for a reason that is not a
refusal.

Zero, and it stays zero. A refusal is a sentence naming a feature, and the
refusal table is what makes one recognisable from the outside. Anything else out
of the transformer is a bug here: a parse tree shape nobody expected, or a
message written where a table entry belongs.
"""

comptime UNSTABLE_CEILING = 0
"""How many statements may print back to something that is not themselves.

Also zero. Print the AST, parse the text, print it again: if the two texts
differ then one of the two passes lost something, and a printer that loses
something is a wrong answer with no error attached to it. This is the property
document 05 section 6 says the printer exists for.
"""

comptime TOO_DEEP_CEILING = 3
"""How many statements may print to text the matcher will not read back.

The printer parenthesizes every operand, so a chain of two thousand additions
prints with two thousand nested parentheses in front of it, and the matcher's
own depth guard stops at about twenty two. The three that are left are the eight
kilobyte expression in `overflow/expression_tree_depth.test` and two wide
generated CTEs in `cte/materialized/test_materialized_cte.test`, and they are the
same `MAX_DEPTH` that `DUCKDB_ONLY_CEILING` above is about. All of them go when
the matcher becomes an explicit stack machine.

It is a bucket of its own rather than an unstable one because nothing was lost.
The AST is right, the text is right, and the only thing that cannot read it is
our own matcher.
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


def breakdown(table: List[Refusal], counts: List[Int], unnamed: Int) -> None:
    """Prints which refusals the corpus landed on, commonest first.

    One number for every refusal is the number that says what to build next. A
    thousand statements on one entry is one feature and a thousand spread over
    twenty is twenty, and the totals above cannot tell those apart.

    `no-case` is the row to watch. Every other entry is a decision somebody
    wrote down, and that one is the transformer running out of cases, so it is
    the count that says how far the jump table is from covering the grammar.

    Args:
        table: The refusal table, in its own order.
        counts: How many statements landed on each entry, parallel to it.
        unnamed: Refusals whose text matched no entry, which should be zero.
    """
    print()
    print("refusals by feature:")

    # Selection sort over a few dozen entries, because the alternative is
    # sorting a list of pairs and there is no pair type here worth adding.
    var shown = List[Bool](length=len(table), fill=False)
    for _ in range(len(table)):
        var best = -1
        for i in range(len(table)):
            if shown[i] or counts[i] == 0:
                continue
            if best < 0 or counts[i] > counts[best]:
                best = i
        if best < 0:
            break
        shown[best] = True
        print("   ", counts[best], table[best].feature)

    if unnamed > 0:
        print("   ", unnamed, "matched no entry, which is a bug in the table")


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


comptime STABLE: UInt8 = 0
"""It transformed, printed, parsed and printed to the same text twice."""

comptime REFUSED_BY_TABLE: UInt8 = 1
"""The transformer refused it by name, which is the expected answer for
everything outside the surface it covers."""

comptime NOT_A_QUERY: UInt8 = 2
"""It is a statement the transformer is not aimed at yet, so the query rule did
not match it and the matcher said so."""

comptime BROKEN: UInt8 = 3
"""The transformer failed for a reason that was not either of those."""

comptime UNSTABLE: UInt8 = 4
"""It printed back to text that says something else."""

comptime TOO_DEEP: UInt8 = 5
"""It printed to text the matcher's depth guard will not read back."""


def round_trip(
    sql: StringSlice, grammar: Grammar, rules: Transform, mut feature: UInt16
) -> UInt8:
    """Puts one statement through the transformer and the printer twice.

    Once is not enough. The printer parenthesizes every operand, so a
    transformer that read its own output one level deeper each time would still
    print something that parses, and only the second pass shows the text
    drifting. Comparing the two prints rather than comparing a print against the
    query is also what lets the printer normalize: `FETCH FIRST 10 ROWS ONLY` is
    allowed to come back as `LIMIT 10`, and it is not allowed to come back as
    something different again the next time round.

    Args:
        sql: The statement, which the parser has already accepted.
        grammar: A loaded grammar.
        rules: A loaded transformer.
        feature: Set to the table entry a refusal came from, and to
            `NO_REFUSAL` for every other outcome.

    Returns:
        One of `STABLE`, `REFUSED_BY_TABLE`, `NOT_A_QUERY`, `BROKEN`,
        `UNSTABLE` or `TOO_DEEP`.
    """
    feature = NO_REFUSAL
    var once: String
    try:
        var ast = Ast()
        var node = rules.parse_statement(sql, grammar, ast)
        once = print_stmt(ast, node, grammar)
    except error:
        var message = String(error)
        if message.startswith(REFUSAL):
            feature = feature_of(message)
            return REFUSED_BY_TABLE
        if message.startswith(SYNTAX):
            return NOT_A_QUERY
        return BROKEN

    try:
        var ast = Ast()
        var node = rules.parse_statement(once, grammar, ast)
        if print_stmt(ast, node, grammar) != once:
            return UNSTABLE
    except error:
        if String(error).find(EXHAUSTED) >= 0:
            return TOO_DEEP
        return UNSTABLE
    return STABLE


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
    var rules = Transform(grammar)
    var compared = 0
    var undecided = 0
    var we_reject = List[Statement]()
    var we_accept = List[Statement]()
    var transformed = 0
    var refused = 0
    var other_statements = 0
    var broken = List[Statement]()
    var unstable = List[Statement]()
    var too_deep = List[Statement]()
    var table = sql_support()
    var by_feature = List[Int](length=len(table), fill=0)
    var unnamed = 0

    for i in range(len(statements)):
        var answer = answers[i]
        ref statement = statements[i]
        var ours = True
        try:
            _ = parse(statement.sql, grammar)
        except:
            ours = False

        if ours:
            var feature = NO_REFUSAL
            var outcome = round_trip(statement.sql, grammar, rules, feature)
            if feature != NO_REFUSAL:
                by_feature[Int(feature)] += 1
            elif outcome == REFUSED_BY_TABLE:
                unnamed += 1
            if outcome == STABLE:
                transformed += 1
            elif outcome == REFUSED_BY_TABLE:
                refused += 1
            elif outcome == NOT_A_QUERY:
                other_statements += 1
            elif outcome == BROKEN:
                broken.append(statement)
            elif outcome == TOO_DEEP:
                too_deep.append(statement)
            else:
                unstable.append(statement)

        if answer != ACCEPTED and answer != REJECTED:
            # DuckDB refused it for a reason that was not a syntax error, so it
            # is not an oracle for this question.
            undecided += 1
            continue

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

    var round_tripped = (
        transformed
        + refused
        + other_statements
        + len(broken)
        + len(unstable)
        + len(too_deep)
    )
    print()
    print(
        "round trip:",
        transformed,
        "stable,",
        refused,
        "refused by name,",
        other_statements,
        "not a query yet,",
        len(broken),
        "broken,",
        len(unstable),
        "unstable,",
        len(too_deep),
        "too deep to read back, of",
        round_tripped,
    )
    breakdown(table, by_feature, unnamed)
    report(
        "the transformer failed on these without refusing:",
        broken,
        round_tripped,
    )
    report("these print back to something else:", unstable, round_tripped)
    report(
        "these print to text the matcher will not read back:",
        too_deep,
        round_tripped,
    )

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
    if len(broken) > BROKEN_CEILING:
        raise Error(
            String(
                "the transformer failed on ",
                len(broken),
                " statements without refusing, against a ceiling of ",
                BROKEN_CEILING,
            )
        )
    if len(unstable) > UNSTABLE_CEILING:
        raise Error(
            String(
                len(unstable),
                (
                    " statements print back to something else, against a"
                    " ceiling of "
                ),
                UNSTABLE_CEILING,
            )
        )
    if len(too_deep) > TOO_DEEP_CEILING:
        raise Error(
            String(
                len(too_deep),
                (
                    " statements print to text the matcher will not read back,"
                    " against a ceiling of "
                ),
                TOO_DEEP_CEILING,
            )
        )
