"""What an expression evaluates to, asked of both engines.

The semantics differential next door asks DuckDB what type an expression has.
This asks the question after that one, and it is the question a type comparison
cannot reach. `strlen` shipped counting characters where DuckDB counts bytes,
and every stage above it agreed: the type was `BIGINT` on both sides, the plan
printed the way it was meant to, and the unit tests asserted the wrong number
because they had been written from the same misreading. Nothing in the
repository could have found it, and this is the thing that would have.

The probe is one table with eight rows in it, described once in `fixture()` and
built twice from that description. firepanda parses the literals into a frame
and DuckDB inserts them, so the two sides cannot drift the way two fixtures
maintained separately do. The rows are chosen so that every reading a kernel can
get wrong is in the table: a row of ASCII, a row that is not ASCII, a row whose
character count and byte count differ, an empty row, a row with spaces on its
ends, a null in every column, a negative number, a zero, a date before the epoch
and a leap day.

Every answer is compared as text. That is the one rendering both sides can be
asked for without either having an opinion about formatting, and a whole number
and a boolean render identically in both. Floating point does not, so nothing
here answers one, and the decimals and the timestamps are the obvious next
thing to add rather than something this is pretending to cover.

Three things can happen to an expression. Both sides answer and the answers
agree, which is the point. Both sides refuse, which is agreement of a weaker
kind and is counted rather than failed. Or they disagree, and there are two ways
to do that: the same rows with different values in them, which is the defect
this exists to catch and whose ceiling is zero, and one side answering where the
other refuses, which is the gap list in executable form and whose ceiling is
zero too.

Both ceilings are zero and both mean something, because a disagreement that is
allowed to stand is one `recorded` names with the reason it stands. Either
firepanda answers differently on purpose, or there is a gap with an issue number
against it. Either way the expression stays in the list and both answers are
printed every run with the reason underneath it.

Usage:
    pixi run differential-answers
"""

from std.python import Python, PythonObject
from std.testing import TestSuite

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.array.strings import StringBuilder
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame
from firepanda.kernel.parse_time import parse_instant
from firepanda.sql.catalog import Catalog
from firepanda.sql.run import run

comptime REFUSED = "!"
"""What DuckDB's answer starts with when it would not run the expression.

`tools/answers.py` writes it, and the rest of the line is the first line of
DuckDB's own message, so a disagreement reads without running the query again by
hand.
"""

comptime PRESENT = "="
"""What a row's answer starts with when there is a value in it."""

comptime MISSING = "-"
"""What a row's answer is when the expression came out null."""

comptime BETWEEN = "\x1f"
"""What separates one row's answer from the next within a line."""

comptime OURS_REFUSED = "!"
"""What firepanda's answer starts with when it would not run the expression."""

comptime NOT_COMPARED = "?"
"""What firepanda's answer is when the expression came out a type this does not
compare yet.

Only a float, a decimal or a temporal answer produces one. Text is what both
sides are asked for and the two do not agree about how to write those, so
comparing them here would report formatting as a wrong answer. Counted and
listed rather than hidden, so that the number is visible and comes down when
the renderings are settled.
"""

comptime SHOWN = 25
"""How many disagreements of each kind to print before summarizing the rest."""

comptime REFUSAL_CEILING = 0
"""How many expressions DuckDB answers and firepanda refuses, at most.

Zero because the list below was written to the surface firepanda covers, and
anything that does not agree is written down in `recorded` by name. A new entry
that refuses is either a gap worth recording or an expression that does not
belong in the list yet, and either way the decision is made when it is added
rather than absorbed by a number that drifts up.
"""


def recorded(expression: StringSlice) -> String:
    """Why a disagreement about this expression is one already written down.

    The list the type differential keeps, in the same form and for the same
    reason: a ceiling of zero only means something if every case that is not
    zero carries the reason it is not, in the file, next to the number. A case
    here is either a decision, meaning the two engines answer differently and
    firepanda's answer is the one it means to give, or a gap with an issue
    against it. Both are named rather than deleted from the list, because an
    expression quietly dropped is a comparison nobody makes again.

    Args:
        expression: The expression that disagreed.

    Returns:
        The reason, or an empty string if this disagreement is a new one.
    """
    # Empty, and it has been empty three times over now. The integer division
    # under issue #770, the LIKE pattern under issue #776 and the cast of a
    # double to an integer under issue #786 were each found here and then fixed
    # rather than written down and left. The list is worth keeping for the next
    # one whose fix is a decision rather than a patch.
    _ = expression
    return ""


def fixture() -> List[List[String]]:
    """The probe table, as SQL literals, one list per row.

    The one description both sides are built from. Each row is the three
    columns in the order `declarations()` names them. `NULL` written bare is a
    missing value, which is why no text row in the table is the word null.

    Returns:
        Eight rows of three literals.
    """
    var rows = List[List[String]]()
    rows.append(["'abc'", "3", "DATE '2020-01-01'"])
    rows.append(["'héllo'", "-2", "DATE '2013-06-30'"])
    rows.append(["'日本語です'", "0", "DATE '1969-12-31'"])
    rows.append(["''", "7", "DATE '2024-02-29'"])
    rows.append(["NULL", "NULL", "NULL"])
    rows.append(["'  pad  '", "100", "DATE '2000-03-15'"])
    rows.append(["'ABC'", "-100", "DATE '1970-01-01'"])
    rows.append(["'banana'", "5", "DATE '2013-08-01'"])
    return rows^


def declarations() -> List[String]:
    """What the probe table's columns are called and what they hold.

    Returns:
        One declaration per column, spelled the way DuckDB spells one.
    """
    return ["s VARCHAR", "n BIGINT", "d DATE"]


def unquoted(literal: StringSlice) -> String:
    """Reads the text out of a single quoted SQL literal.

    Args:
        literal: The literal, quotes included.

    Returns:
        What is between the quotes, with a doubled quote read as one.
    """
    var inner = literal[byte = 1 : literal.byte_length() - 1]
    return String(inner).replace("''", "'")


def probe() raises -> DataFrame:
    """Builds the probe table as a frame, from the literals in `fixture()`.

    The date column goes through `parse_instant`, which is the same reader a
    `DATE '2020-01-01'` in a query goes through, so the frame holds what the
    query would have put there.

    Returns:
        A frame with a row number, a text column, a whole number column and a
        date column.

    Raises:
        Error: If a literal cannot be read.
    """
    var rows = fixture()

    var i = Array[DType.int64](len(rows))
    var text = StringBuilder(capacity=len(rows))
    var n = Array[DType.int64](len(rows))
    var d = Array[DType.int32](len(rows))

    for at in range(len(rows)):
        i.set_valid(at, Int64(at))

        ref written = rows[at][0]
        if written == "NULL":
            text.append_null()
        else:
            text.append(unquoted(written).as_bytes())

        ref counted = rows[at][1]
        if counted == "NULL":
            n.set_null(at)
        else:
            n.set_valid(at, Int64(atol(counted)))

        ref dated = rows[at][2]
        if dated == "NULL":
            d.set_null(at)
        else:
            # `DATE '...'`, so the literal inside it starts at the quote.
            var quoted = dated[byte = dated.find("'") :]
            var value = parse_instant(
                unquoted(quoted).as_bytes(), LogicalType.DATE32
            )
            d.set_valid(at, value.as_scalar[DType.int32]())

    var columns = List[ChunkedArray]()
    var row = ChunkedArray(LogicalType.INT64)
    row.append(AnyArray(i^))
    columns.append(row^)
    var word = ChunkedArray(LogicalType.STRING)
    word.append(AnyArray(text^.finish()))
    columns.append(word^)
    var count = ChunkedArray(LogicalType.INT64)
    count.append(AnyArray(n^))
    columns.append(count^)
    var day = ChunkedArray(LogicalType.DATE32)
    # A date is an int32 count of days, so the logical type has to be said
    # rather than inferred, or the column comes back an `INT32`.
    day.append(AnyArray(d^.into_data(), LogicalType.DATE32))
    columns.append(day^)

    var fields = List[Field]()
    fields.append(Field("i", LogicalType.INT64))
    fields.append(Field("s", LogicalType.STRING, True))
    fields.append(Field("n", LogicalType.INT64, True))
    fields.append(Field("d", LogicalType.DATE32, True))
    return DataFrame(Schema(fields^), columns^)


def expressions() -> List[String]:
    """Every expression both engines are asked about.

    Written by hand rather than generated, because what is worth asking is the
    surface firepanda actually runs, and a generator over the tier 1 catalog
    would spend the whole run on names nothing is wired to. The order is the
    order the functions were wired in, which makes a diff against a later
    version of this list read as the list of what was added.

    Returns:
        The expressions, in terms of `s`, `n` and `d`.
    """
    var out = List[String]()

    # The two lengths, which are two questions and not one.
    out.append("length(s)")
    out.append("len(s)")
    out.append("strlen(s)")

    # Case.
    out.append("upper(s)")
    out.append("lower(s)")
    out.append("ucase(s)")
    out.append("lcase(s)")

    # The ends.
    out.append("trim(s)")
    out.append("ltrim(s)")
    out.append("rtrim(s)")
    out.append("trim(s, 'ab')")
    out.append("ltrim(s, 'a')")
    out.append("rtrim(s, 'a')")

    # Pieces.
    out.append("substring(s, 2)")
    out.append("substring(s, 2, 3)")
    out.append("substr(s, 1, 2)")
    out.append("substring(s FROM 2 FOR 2)")
    out.append("substring(s, -2, 3)")
    out.append("substring(s, 1, 0)")

    # Searching.
    out.append("instr(s, 'a')")
    out.append("strpos(s, 'an')")
    out.append("position('a' IN s)")
    out.append("instr(s, '')")

    # Presence.
    out.append("coalesce(s, 'x')")
    out.append("ifnull(s, 'x')")
    out.append("nullif(s, 'abc')")
    out.append("s IS NULL")
    out.append("s IS NOT NULL")
    out.append("n IS NULL")

    # Matching.
    out.append("s LIKE 'a%'")
    out.append("s LIKE '%a%'")
    out.append("s LIKE '_b_'")
    out.append("s NOT LIKE 'a%'")
    # The wildcards against text that is not one byte a character, which is the
    # part of the pattern language a byte counter gets wrong and a row of ASCII
    # can never show. `héllo` is five characters and six bytes and `日本語です`
    # is five characters and fifteen.
    out.append("s LIKE 'h_llo'")
    out.append("s LIKE '_____'")
    out.append("s LIKE '%本%です'")
    out.append("s LIKE 'a%c'")
    out.append("s LIKE '%a%a%'")

    # Comparing text.
    out.append("s = 'abc'")
    out.append("s <> 'abc'")
    out.append("s < 'b'")

    # Arithmetic.
    out.append("n + 1")
    out.append("n - 1")
    out.append("n * 2")
    out.append("n // 3")
    out.append("n % 3")
    out.append("-n")

    # Comparing numbers.
    out.append("n > 0")
    out.append("n <= 0")
    out.append("n BETWEEN -2 AND 5")
    out.append("n IN (0, 3, 5)")
    out.append("n NOT IN (0, 3, 5)")

    # Connectives, including what a null does to one.
    out.append("n > 0 AND s IS NOT NULL")
    out.append("n > 0 OR s IS NULL")
    out.append("NOT (n > 0)")

    # Filling.
    out.append("coalesce(n, -1)")
    out.append("nullif(n, 0)")

    # Choosing.
    out.append(
        "CASE WHEN n > 0 THEN 'pos' WHEN n = 0 THEN 'zero' ELSE 'neg' END"
    )
    out.append("CASE WHEN s IS NULL THEN 'none' ELSE s END")
    out.append("CASE n WHEN 0 THEN 'zero' WHEN 3 THEN 'three' END")

    # Dates, read as whole numbers so that the comparison is about the reading
    # and not about how a date is written out.
    out.append("date_part('year', d)")
    out.append("date_part('month', d)")
    out.append("date_part('day', d)")
    out.append("datepart('year', d)")
    out.append("extract(YEAR FROM d)")
    out.append("extract(MONTH FROM d)")
    out.append("extract(DAY FROM d)")
    out.append("d = DATE '2020-01-01'")
    out.append("d > DATE '2013-01-01'")
    out.append("d IS NULL")

    # The number literals that are doubles to DuckDB, over a column so that
    # neither engine folds the expression before it is run. Read back as a
    # whole number or a yes and no, because this compares three types and a
    # double is not one of them. The decimal literal that fits a decimal is not
    # here, because firepanda refuses it.
    out.append("CAST(n * 1e3 AS BIGINT)")
    out.append("CAST(n + 1.5e3 AS BIGINT)")
    out.append("n * 1.1e-2 < 1")
    out.append("CAST(n * 1.5000000000000000000000000000000000000000 AS BIGINT)")

    # Nested, because a kernel that is right on a column can still be wrong on
    # what another kernel just built.
    out.append("length(trim(s))")
    out.append("upper(substring(s, 1, 2))")
    out.append("instr(upper(s), 'A')")
    out.append("length(coalesce(s, 'xx'))")
    out.append("trim(upper(s))")

    return out^


def rendered(frame: DataFrame) raises -> String:
    """Writes a frame's answer column out the way `tools/answers.py` writes one.

    The answer is the second column, because the query asks for the row number
    in front of it so that both sides can order by the same thing.

    Args:
        frame: What the query returned.

    Returns:
        The rows joined by `BETWEEN`, or `NOT_COMPARED` for a type this does
        not compare.

    Raises:
        Error: If the column cannot be read.
    """
    var type = frame.schema[1].dtype
    var answer = frame.column("a")
    var pieces = List[String]()

    if type == LogicalType.STRING:
        var col = answer.as_strings()
        for at in range(len(col)):
            if not col.is_valid(at):
                pieces.append(MISSING)
            else:
                pieces.append(PRESENT + String(col[at]))
    elif type == LogicalType.INT64:
        var col = answer.as_typed[DType.int64]()
        for at in range(len(col)):
            if not col.is_valid(at):
                pieces.append(MISSING)
            else:
                pieces.append(PRESENT + String(col[at]))
    elif type == LogicalType.BOOL:
        var col = answer.as_typed[DType.bool]()
        for at in range(len(col)):
            if not col.is_valid(at):
                pieces.append(MISSING)
            else:
                pieces.append(PRESENT + ("true" if col[at] else "false"))
    else:
        return NOT_COMPARED

    return BETWEEN.join(pieces)


def ours(catalog: Catalog, expression: StringSlice) -> String:
    """Runs one expression through firepanda and renders what came back.

    Args:
        catalog: The session holding the probe table.
        expression: The expression.

    Returns:
        The rendered answer, or `OURS_REFUSED` followed by the first line of
        the message.
    """
    try:
        var out = run(
            String("SELECT i, ", expression, " AS a FROM probe ORDER BY i"),
            catalog,
        )
        return rendered(out)
    except error:
        return OURS_REFUSED + String(error).split("\n")[0]


def ask_duckdb(batch: List[String]) raises -> List[String]:
    """Asks DuckDB about every expression at once.

    One crossing into Python for the whole run, for the reason the semantics
    differential does the same: the bridge costs more per crossing than DuckDB
    costs per query.

    Args:
        batch: The expressions.

    Returns:
        One answer per expression, in order.

    Raises:
        Error: If the helper could not be reached, or answered the wrong number
            of times.
    """
    var declared = declarations()
    var columns = Python.list()
    for name in declared:
        columns.append(PythonObject(name))

    var table = fixture()
    var rows = Python.list()
    for row in table:
        var written = Python.list()
        for literal in row:
            written.append(PythonObject(literal))
        rows.append(written)

    var asked = Python.list()
    for expression in batch:
        asked.append(PythonObject(expression))

    var python_path = Python.import_module("sys").path
    python_path.insert(0, "tools")
    var helper = Python.import_module("answers")

    var out = List[String]()
    for line in String(helper.answers_of(columns, rows, asked)).split("\n"):
        if line.byte_length() != 0:
            out.append(String(line))
    if len(out) != len(batch):
        raise Error(
            String(
                "DuckDB answered about ",
                len(out),
                " expressions and there are ",
                len(batch),
            )
        )
    return out^


def report(
    title: StringSlice,
    written: List[String],
    theirs: List[String],
    mine: List[String],
):
    """Prints a bucket of disagreements, at most `SHOWN` of them.

    Args:
        title: What the bucket is.
        written: The expressions in it.
        theirs: What DuckDB answered for each.
        mine: What firepanda answered for each.
    """
    if len(written) == 0:
        return
    print()
    print(len(written), title)
    for at in range(min(len(written), SHOWN)):
        print("   ", written[at])
        print("      duckdb:   ", theirs[at])
        print("      firepanda:", mine[at])
    if len(written) > SHOWN:
        print("    and", len(written) - SHOWN, "more")


def explain(
    written: List[String],
    theirs: List[String],
    mine: List[String],
    why: List[String],
):
    """Prints the disagreements that are already written down, with reasons.

    Separate from `report` because these carry a fourth line each, and because
    the point of printing them at all is the reason rather than the values.

    Args:
        written: The expressions in the bucket.
        theirs: What DuckDB answered for each.
        mine: What firepanda answered for each.
        why: The reason each one is on the list.
    """
    if len(written) == 0:
        return
    print()
    print(len(written), "disagree for a reason already written down:")
    for at in range(len(written)):
        print("   ", written[at])
        print("      duckdb:   ", theirs[at])
        print("      firepanda:", mine[at])
        print("      why:      ", why[at])


def test_every_expression_agrees_with_duckdb() raises:
    """Runs the comparison and raises if any ceiling is passed."""
    var batch = expressions()
    print(
        "asking both engines about",
        len(batch),
        "expressions over",
        len(fixture()),
        "rows",
    )

    var catalog = Catalog()
    catalog.register("probe", probe())

    var theirs = ask_duckdb(batch)

    var agreed = 0
    var both_refused = 0
    var skipped = 0

    var wrong_value = List[String]()
    var wrong_value_theirs = List[String]()
    var wrong_value_mine = List[String]()
    var we_refuse = List[String]()
    var we_refuse_theirs = List[String]()
    var we_refuse_mine = List[String]()
    var we_answer = List[String]()
    var we_answer_theirs = List[String]()
    var we_answer_mine = List[String]()
    var known = List[String]()
    var known_theirs = List[String]()
    var known_mine = List[String]()
    var known_why = List[String]()

    for at in range(len(batch)):
        ref expression = batch[at]
        ref them = theirs[at]
        var us = ours(catalog, expression)

        var they_refused = them.startswith(REFUSED)
        var we_refused = us.startswith(OURS_REFUSED)

        if they_refused and we_refused:
            both_refused += 1
            continue
        if us == NOT_COMPARED:
            skipped += 1
            continue
        if not they_refused and not we_refused and us == them:
            agreed += 1
            continue

        # Everything from here down is a disagreement of some shape, so the
        # list of the ones already written down is consulted once rather than
        # in each of the three buckets below.
        var why = recorded(expression)
        if why != "":
            known.append(expression)
            known_theirs.append(them)
            known_mine.append(us)
            known_why.append(why)
            continue

        if not they_refused and not we_refused:
            wrong_value.append(expression)
            wrong_value_theirs.append(them)
            wrong_value_mine.append(us)
        elif we_refused:
            we_refuse.append(expression)
            we_refuse_theirs.append(them)
            we_refuse_mine.append(us)
        else:
            we_answer.append(expression)
            we_answer_theirs.append(them)
            we_answer_mine.append(us)

    print(
        "compared",
        len(batch),
        "expressions:",
        agreed,
        "agreed,",
        both_refused,
        "were refused by both,",
        len(known),
        "disagree for a written reason,",
        skipped,
        "answered a type this does not compare",
    )

    explain(known, known_theirs, known_mine, known_why)
    report(
        "come out with different values:",
        wrong_value,
        wrong_value_theirs,
        wrong_value_mine,
    )
    report(
        "DuckDB answers and firepanda refuses:",
        we_refuse,
        we_refuse_theirs,
        we_refuse_mine,
    )
    report(
        "firepanda answers and DuckDB refuses:",
        we_answer,
        we_answer_theirs,
        we_answer_mine,
    )

    print()
    if len(wrong_value) != 0:
        raise Error(
            String(
                "firepanda and DuckDB answer ",
                len(wrong_value),
                " expressions differently, against a ceiling of 0",
            )
        )
    if len(we_refuse) > REFUSAL_CEILING:
        raise Error(
            String(
                "DuckDB answers ",
                len(we_refuse),
                " expressions firepanda refuses, against a ceiling of ",
                REFUSAL_CEILING,
            )
        )
    if len(we_answer) != 0:
        raise Error(
            String(
                "firepanda answers ",
                len(we_answer),
                " expressions DuckDB refuses, against a ceiling of 0",
            )
        )
    if len(known) == 0:
        print("every expression agrees")
    else:
        print(
            "every expression agrees except the",
            len(known),
            "written down above",
        )


def main() raises:
    # The comparison is reached through the suite's table of functions rather
    # than called, and that indirection is load bearing rather than tidiness. A
    # `main` that calls anything reaching `firepanda.sql.run` by a direct call
    # hangs the compiler: it parks with no CPU a few seconds after the import
    # phase and never comes back, at every optimization level, at any thread
    # count, and whether the program is built or run. The same body behind this
    # indirection compiles in seconds. Reduced to twenty lines, the difference
    # is those two lines and nothing else, so it is the compiler and not this
    # file. See docs/specs/sql/13-open-questions.md question 13.
    TestSuite.discover_tests[__functions_in_module()]().run()
