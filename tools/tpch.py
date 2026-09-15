"""TPC-H data and reference answers, generated once and read by both engines.

The other harnesses in this directory ask about one expression at a time. This
one asks about a whole query, which is the question a benchmark asks and the
one an exit criterion is written in: does firepanda answer a TPC-H query the
way DuckDB answers it over the same rows. `QUERIES` says which ones it asks.

The data comes from DuckDB's own `tpch` extension, which is the reference
generator rather than something written here, and it is exported to Parquet so
that both engines read the same bytes. Reading the same bytes is the whole
point. Two generators seeded the same way is a claim about two programs, and a
claim about two programs is what a differential harness exists to stop making.

The queries come from DuckDB's `tpch_queries()` for the same reason. They are
written out beside the data rather than checked in, so there is no second copy
of the query text to drift from the first one.

One thing is not the reference data and has to be said out loud. TPC-H declares
eight columns as `DECIMAL(15,2)` and firepanda has no decimal type, so those
columns are cast to `DOUBLE` on the way out. DuckDB is then asked the same
question over the same doubles, which keeps the comparison exact, but it is a
comparison against DuckDB over this data rather than against the published
answer set: a sum of doubles is not a sum of decimals and the last digits
differ. The published answers come back when there is a decimal type, and issue
#309 is where that is tracked.

A float is compared with a relative tolerance and everything else exactly.
Neither engine adds a column of doubles in the same order as the other, and two
orders of addition do not agree in the last bits, so an exact comparison would
be a test of the summation order rather than of the answer.

The one entry point a caller needs is `prepare`, which builds the data if it is
not already there and hands back the directory. `question` and `compare` are the
two the Mojo side calls afterwards and neither of them touches DuckDB, for the
reason `answers.py` gives: `import duckdb` inside the interpreter embedded in a
Mojo binary registers exit handlers that run after that interpreter is gone, and
the process then dies in a destructor with the report already printed.

See docs/specs/sql/11-conformance.md.
"""

from __future__ import annotations

import math
import os
import sys

TABLES = (
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
)
"""The eight tables TPC-H defines, in the order the schema declares them."""

QUERIES = (1, 3, 4, 5, 6, 7, 10, 12, 15, 16, 18, 19)
"""Which queries the harness covers.

The three S4 names as its exit criteria, and every query that answers what
DuckDB answers. The rest want a decimal, a join order, a dependent join or
a binder gap closed, and each one goes in here the day it runs rather than being
listed as a pending failure.
"""

BETWEEN = "\x1f"
"""What separates one cell from the next within a row.

A unit separator, so a value holding a comma or a tab is written as it is. TPC-H
writes no newline into a value and neither side escapes one.
"""

MISSING = "\x1e"
"""What a cell holds when the value is null.

A separate character rather than an empty field, because an empty string is a
value a TPC-H column can really hold and the two have to stay apart.
"""

TOLERANCE = 1e-9
"""How far apart two doubles may be, relatively, and still be the same answer.

Wide enough to cover a different order of addition over six million rows and far
narrower than any difference a wrong expression makes. A wrong `1 - l_discount`
is out by percent, not by parts in a billion.
"""


def _exported(connection, table):
    """What to select out of a table so that firepanda can read it back.

    Args:
        connection: A DuckDB connection with the TPC-H tables in it.
        table: The table name.

    Returns:
        The select list, as text, with every decimal column cast to a double
        under its own name and every other column named plainly.
    """
    described = connection.execute(f"describe {table}").fetchall()
    out = []
    for row in described:
        name, declared = row[0], row[1]
        if declared.startswith("DECIMAL"):
            out.append(f"{name}::DOUBLE AS {name}")
        else:
            out.append(name)
    return ", ".join(out)


def _rendered(value):
    """Writes one cell the way both sides agree to read it.

    Args:
        value: What DuckDB handed back for the cell.

    Returns:
        The cell's text, or `MISSING` for a null.
    """
    if value is None:
        return MISSING
    return str(value)


def _stamped(marker, stamp):
    """Whether a marker file exists and already says this.

    Args:
        marker: The path.
        stamp: What it has to hold for the work behind it to be skipped.

    Returns:
        True when the file is there and holds exactly that.
    """
    if not os.path.exists(marker):
        return False
    with open(marker, encoding="utf-8") as handle:
        return handle.read().strip() == stamp


def prepare_here(scale, directory):
    """Builds the data and the reference answers, in this process.

    Does nothing when the directory already holds both at the same scale, which
    is what makes the harness cheap to run twice. Each marker is written after
    what it stands for, so an interrupted run leaves a directory that gets built
    again rather than one that looks finished.

    There are two markers because the two halves go stale for different
    reasons. The data goes stale when the scale changes and nothing else, and
    at scale 1 it is a quarter of a gigabyte to regenerate. The answers go stale
    when a query is added to `QUERIES` too, since only the queries in it get an
    answer written, and one marker for both would either regenerate the data
    every time a query was added or leave the new query with no answer to be
    compared against.

    Call this from a plain Python process. `prepare` is what an embedded
    interpreter should call, and it runs this in a child.

    Args:
        scale: The TPC-H scale factor. 0.01 is about 60 thousand lineitem rows
            and 3 MB, and 1 is about 6 million and 250 MB.
        directory: Where the Parquet files, the queries and the answers go.
    """
    import duckdb

    marker = os.path.join(directory, "ready")
    stamp = str(scale) + " " + " ".join(str(number) for number in QUERIES)
    if _stamped(marker, stamp):
        return

    built = os.path.join(directory, "generated")
    os.makedirs(directory, exist_ok=True)
    connection = duckdb.connect()
    if not _stamped(built, str(scale)):
        connection.execute(f"call dbgen(sf={scale})")
        for table in TABLES:
            path = os.path.join(directory, f"{table}.parquet")
            selected = _exported(connection, table)
            connection.execute(
                f"copy (select {selected} from {table}) to '{path}'"
                " (format parquet)"
            )
        with open(built, "w", encoding="utf-8") as handle:
            handle.write(str(scale) + "\n")
    written = dict(connection.execute("select * from tpch_queries()").fetchall())
    connection.close()

    # A second connection, reading the Parquet the first one wrote rather than
    # the tables it generated. The answers have to come off the bytes firepanda
    # reads, not off the decimals they were cast from, or the reference is the
    # answer to a different question.
    reading = duckdb.connect()
    for table in TABLES:
        path = os.path.join(directory, f"{table}.parquet")
        reading.execute(
            f"create view {table} as select * from read_parquet('{path}')"
        )
    for number in QUERIES:
        text = written[number].strip().rstrip(";")
        with open(
            os.path.join(directory, f"q{number:02d}.sql"), "w", encoding="utf-8"
        ) as handle:
            handle.write(text + "\n")
        found = reading.execute(text)
        names = [column[0] for column in found.description]
        lines = [BETWEEN.join(names)]
        for row in found.fetchall():
            lines.append(BETWEEN.join(_rendered(cell) for cell in row))
        with open(
            os.path.join(directory, f"a{number:02d}.tsv"), "w", encoding="utf-8"
        ) as handle:
            handle.write("\n".join(lines) + "\n")
    reading.close()

    with open(marker, "w", encoding="utf-8") as handle:
        handle.write(stamp + "\n")


def prepare(scale, directory) -> str:
    """Builds the data in a child process and hands back the directory.

    Args:
        scale: The TPC-H scale factor.
        directory: Where the Parquet files, the queries and the answers go.

    Returns:
        The directory, so that a caller can hold one value rather than two.

    Raises:
        RuntimeError: If the child could not build the data.
    """
    import subprocess

    child = subprocess.run(
        [sys.executable, os.path.abspath(__file__), str(scale), directory],
        capture_output=True,
        text=True,
        check=False,
    )
    if child.returncode != 0:
        raise RuntimeError(
            f"generating TPC-H at scale {scale} exited {child.returncode}:"
            f" {child.stderr.strip()}"
        )
    return directory


def question(directory, number) -> str:
    """Reads back the text of one query.

    Args:
        directory: What `prepare` handed back.
        number: The query number.

    Returns:
        The query, as DuckDB publishes it, with no trailing semicolon.
    """
    with open(
        os.path.join(directory, f"q{number:02d}.sql"), encoding="utf-8"
    ) as handle:
        return handle.read().strip()


def _same(mine, theirs) -> bool:
    """Whether two cells hold the same value.

    Numeric when both sides parse as a number, which covers the whole numbers
    exactly and gives the doubles the tolerance they need, and text otherwise,
    which is what a date, a flag and a comment want.

    Args:
        mine: firepanda's cell.
        theirs: DuckDB's cell.

    Returns:
        True when they agree.
    """
    if mine == theirs:
        return True
    try:
        return math.isclose(float(mine), float(theirs), rel_tol=TOLERANCE)
    except ValueError:
        return False


def compare(directory, number, header, body) -> str:
    """Compares firepanda's answer against the reference, and says how it differs.

    Args:
        directory: What `prepare` handed back.
        number: The query number.
        header: firepanda's column names, joined by `BETWEEN`.
        body: firepanda's rows, each one joined by `BETWEEN`, rows separated by
            newlines. Empty for an answer with no rows in it.

    Returns:
        The empty string when the two agree, and one line saying where they
        first differ otherwise.
    """
    with open(
        os.path.join(directory, f"a{number:02d}.tsv"), encoding="utf-8"
    ) as handle:
        lines = handle.read().rstrip("\n").split("\n")

    wanted_names = lines[0].split(BETWEEN)
    got_names = header.split(BETWEEN)
    if got_names != wanted_names:
        return (
            f"columns {', '.join(got_names)} against"
            f" {', '.join(wanted_names)}"
        )

    wanted = [line.split(BETWEEN) for line in lines[1:]]
    got = [line.split(BETWEEN) for line in body.split("\n") if line]
    if len(got) != len(wanted):
        return f"{len(got)} rows against {len(wanted)}"

    for at, (ours, theirs) in enumerate(zip(got, wanted)):
        for column, (mine, reference) in enumerate(zip(ours, theirs)):
            if not _same(mine, reference):
                return (
                    f"row {at} column {wanted_names[column]}: {mine} against"
                    f" {reference}"
                )
    return ""


def main() -> int:
    """Builds the data for one scale factor, which is what `prepare` runs.

    Returns:
        A process exit status.
    """
    prepare_here(float(sys.argv[1]), sys.argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
