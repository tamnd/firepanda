"""TPC-H, asked of both engines over the same bytes.

The harnesses next door ask about one expression at a time, which is the right
size for a kernel and the wrong size for a query. A query has a plan in it, and
a plan has a join order, a group, a sort and a limit, and every one of those is
a place an answer can go wrong in a way no expression comparison reaches. So
this one asks whole queries, and it asks the three S4 names as its exit
criteria, TPC-H q1, q3 and q6, together with every other query that answers what
DuckDB answers. Ten of the twenty two are in the list today.

The data is DuckDB's own `tpch` generator, exported to Parquet, and both engines
read the same files. Two generators seeded the same way is a claim about two
programs, and a claim about two programs is what a differential harness exists
to stop making.

The queries are DuckDB's own `tpch_queries()`, written out beside the data, so
there is no second copy of the query text here to drift from the first one.

One thing is not the reference data and is said out loud in `tools/tpch.py`
too. TPC-H declares eight columns as `DECIMAL(15,2)`, firepanda has no decimal
type, and those columns are cast to `DOUBLE` on the way out. DuckDB is asked the
same question over the same doubles, so the comparison stays exact, but it is a
comparison against DuckDB over this data and not against the published answer
set. The published answers come back when there is a decimal type.

A double is compared with a relative tolerance and everything else exactly,
because neither engine adds a column in the same order as the other and two
orders of addition do not agree in the last bits. An exact comparison would be a
test of the summation order.

A query firepanda refuses is reported rather than failed when the refusal has a
reason written down in `recorded`, which is the gap list in executable form. A
refusal with no reason against it fails, and so does a query that answers where a
refusal was recorded, because a stale record is how a gap that closed goes on
looking open.

The scale factor is `FIREPANDA_TPCH_SCALE` and defaults to 0.01, which is about
sixty thousand lineitem rows and three megabytes. The exit criterion is written
at 1, which is a hundred times that in both, and is a number to run by hand
rather than on every push.

Usage:
    pixi run tpch
    FIREPANDA_TPCH_SCALE=1 pixi run tpch
"""

from std.python import Python, PythonObject
from std.testing import TestSuite

from firepanda.frame.frame import DataFrame
from firepanda.io.parquet import read_parquet
from firepanda.io.write import cell_text
from firepanda.sql.catalog import Catalog
from firepanda.sql.run import run

comptime BETWEEN = "\x1f"
"""What separates one cell from the next within a row, both ways across the
Python boundary."""

comptime MISSING = "\x1e"
"""What a cell holds when the value is null."""


def tables() -> List[String]:
    """The eight tables TPC-H defines.

    All eight are registered whatever the query reads, because the catalog is
    what a query is asked against and a catalog that changes from one query to
    the next is a fixture that can disagree with itself.

    Returns:
        The table names, which are the Parquet file names too.
    """
    return [
        String("customer"),
        String("lineitem"),
        String("nation"),
        String("orders"),
        String("part"),
        String("partsupp"),
        String("region"),
        String("supplier"),
    ]


def queries() -> List[Int]:
    """Which queries this asks about.

    The three S4 names as its exit criteria, and every query that answers what
    DuckDB answers. The rest want a decimal, a join order, a cross product, a
    dependent join or a name resolved across a self join, and each one is added
    here the day it runs rather than sitting in a list of pending failures.

    This list and `QUERIES` in `tools/tpch.py` are the same list written twice,
    because the Python side is what writes an answer out and the Mojo side is
    what asks for one. A query in one and not the other is caught on the run
    after it is added: a query here and not there has no answer file to read,
    and a query there and not here is an answer nobody asks for.

    Returns:
        The query numbers.
    """
    return [1, 3, 4, 5, 6, 10, 12, 15, 16, 18]


def recorded(number: Int) -> String:
    """Why a query firepanda refuses is allowed to be refused.

    A refusal here is a gap with a reason, and the reason is printed every run
    beside the refusal itself. Anything not named here is a failure, so a query
    that stops running is noticed the run after it stops.

    Args:
        number: The query number.

    Returns:
        The reason, or the empty string if a refusal is not expected.
    """
    if number == 6:
        return String(
            "the decimal literals in `l_discount BETWEEN 0.05 AND 0.07`, which"
            " a plan cannot hold exactly and which a double in their place"
            " would answer wrongly. Issue #309"
        )
    return String()


def _rendered(frame: DataFrame) raises -> Tuple[String, String]:
    """Writes an answer out the way the reference is written.

    Args:
        frame: What the query answered.

    Returns:
        The column names joined by `BETWEEN`, and the rows, each one joined by
        `BETWEEN` and separated by newlines.

    Raises:
        Error: If the answer holds a type there is no text for.
    """
    var header = String()
    for column in range(frame.width()):
        if column > 0:
            header += BETWEEN
        header += frame.names()[column]

    var body = String()
    for row in range(len(frame)):
        if row > 0:
            body += "\n"
        for column in range(frame.width()):
            if column > 0:
                body += BETWEEN
            if not frame[column].is_valid(row):
                body += MISSING
            else:
                body += cell_text(frame[column], row)
    return (header^, body^)


def test_the_queries_answer_what_duckdb_answers() raises:
    var python_path = Python.import_module("sys").path
    python_path.insert(0, "tools")
    var helper = Python.import_module("tpch")
    var environ = Python.import_module("os").environ

    var scale = String(environ.get("FIREPANDA_TPCH_SCALE", "0.01"))
    var directory = String("build/tpch/sf", scale)
    print("TPC-H at scale", scale, "in", directory)
    _ = helper.prepare(PythonObject(scale), PythonObject(directory))

    var named = tables()
    var catalog = Catalog()
    for name in named:
        catalog.register(
            name, read_parquet(String(directory, "/", name, ".parquet"))
        )
    print("read", len(named), "tables")
    print()

    var wrong = List[Int]()
    var unexplained = List[Int]()
    var stale = List[Int]()
    var agreed = 0

    var asked = queries()
    for number in asked:
        var sql = String(helper.question(PythonObject(directory), number))
        var answer = DataFrame()
        var refusal = String()
        try:
            answer = run(sql, catalog)
        except e:
            refusal = String(e)

        var reason = recorded(number)
        if refusal:
            if reason:
                print(String("q", number, " is refused, for a written reason:"))
                print("   ", refusal)
                print("    because", reason)
            else:
                print(String("q", number, " is refused:"))
                print("   ", refusal)
                unexplained.append(number)
            print()
            continue

        if reason:
            print(String("q", number, " answers, and this says it does not:"))
            print("   ", reason)
            print()
            stale.append(number)
            continue

        var written = _rendered(answer)
        var differs = String(
            helper.compare(
                PythonObject(directory),
                number,
                PythonObject(written[0]),
                PythonObject(written[1]),
            )
        )
        if differs:
            print(String("q", number, " differs: ", differs))
            print()
            wrong.append(number)
        else:
            print(String("q", number, " agrees, over ", len(answer), " rows"))
            agreed += 1

    print()
    print(
        agreed,
        "of",
        len(asked),
        "queries agree with DuckDB over the same rows",
    )

    if len(wrong) != 0:
        raise Error(
            String(
                "firepanda answers ",
                len(wrong),
                (
                    " TPC-H queries differently from DuckDB, against a ceiling"
                    " of 0"
                ),
            )
        )
    if len(unexplained) != 0:
        raise Error(
            String(
                len(unexplained),
                (
                    " TPC-H queries are refused with no reason written down,"
                    " and a refusal without one is a regression rather than"
                    " a gap"
                ),
            )
        )
    if len(stale) != 0:
        raise Error(
            String(
                len(stale),
                (
                    " TPC-H queries answer where a refusal is recorded, so the"
                    " record is stale and `recorded` has to drop them"
                ),
            )
        )


def main() raises:
    # Through the suite's table of functions rather than called, for the reason
    # written at the bottom of `answers.mojo`: a `main` that reaches
    # `firepanda.sql.run` by a direct call hangs the compiler.
    # See docs/specs/sql/13-open-questions.md question 13.
    TestSuite.discover_tests[__functions_in_module()]().run()
