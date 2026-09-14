"""What DuckDB says an expression evaluates to, row by row.

`tools/semantics.py` asks DuckDB what type an expression has. This asks the
question after that one, and it is the question a type comparison cannot reach:
given the same rows, does the expression come out with the same values. A binder
that agrees with DuckDB about every type and a `strlen` that counts characters
where DuckDB counts bytes is a library that returns wrong answers of exactly the
right type, and nothing above this notices.

The probe is a table the caller describes: column declarations, and rows written
out as SQL literals. Both sides build the same table from the same description,
so the comparison is about the expressions rather than about two fixtures that
drifted.

Each expression is evaluated as `CAST(expression AS VARCHAR)`, because text is
the one rendering both sides can be asked for without either of them having an
opinion about formatting. A whole number casts to its digits and a boolean casts
to `true` or `false` in DuckDB, and firepanda writes both the same way, so the
comparison is exact rather than approximate. Floating point is the one this
cannot do honestly, and the caller is the one that leaves it out.

The one entry point is `answers_of`, which takes the table description and the
expressions and hands back one line per expression. It runs DuckDB in a child
process for the reason `corpus.py` and `semantics.py` do: `import duckdb` inside
the interpreter embedded in a Mojo binary registers exit handlers that run after
that interpreter is gone, and the process then dies in a destructor with the
report already printed.

See docs/specs/sql/11-conformance.md.
"""

from __future__ import annotations

import os
import sys

REFUSED = "!"
"""What an answer starts with when DuckDB would not run the expression.

The rest of the line is the first line of DuckDB's own message, which is what
makes a disagreement readable without running the query again by hand.
"""

PRESENT = "="
"""What a row's answer starts with when there is a value in it."""

MISSING = "-"
"""What a row's answer is when the expression came out null.

A separate character rather than an empty field, because an empty string is a
value an expression can really answer and the two have to stay apart.
"""

BETWEEN = "\x1f"
"""What separates one row's answer from the next within a line.

A unit separator, so that a value holding a comma or a tab is written out as it
is rather than escaped. A value holding a newline would still break the format,
and the caller is asked not to write one.
"""


def answers_for(columns, rows, expressions):
    """Asks DuckDB for the value of every expression over every row.

    One connection and one table for the whole batch. The table has a column
    called `i` in front of the ones the caller named, holding the row's position,
    which is what both sides order by so that the answers line up.

    Call this from a plain Python process. `answers_of` is what an embedded
    interpreter should call, and it runs this in a child.

    Args:
        columns: The column declarations, spelled as DuckDB spells them, each
            one a name and a type together.
        rows: The rows, each one a list of SQL literals, one per column.
        expressions: The expressions to evaluate, written in terms of the
            column names.

    Returns:
        One answer per expression, in order. Either the rows joined by
        `BETWEEN`, each one `PRESENT` or `MISSING` followed by the value, or
        `REFUSED` followed by the first line of DuckDB's message.
    """
    import duckdb

    connection = duckdb.connect()
    declared = ", ".join(["i BIGINT", *columns])
    connection.execute(f"create table probe({declared})")
    for at, row in enumerate(rows):
        written = ", ".join([str(at), *row])
        connection.execute(f"insert into probe values ({written})")

    out = []
    for expression in expressions:
        try:
            found = connection.execute(
                f"select cast(({expression}) as varchar) from probe order by i"
            ).fetchall()
            out.append(
                BETWEEN.join(
                    MISSING if value[0] is None else PRESENT + value[0]
                    for value in found
                )
            )
        except Exception as error:
            out.append(REFUSED + str(error).split("\n")[0].strip())
    connection.close()
    return out


def answers_of(columns, rows, expressions) -> str:
    """Asks DuckDB about a batch the caller already has.

    The batch goes over a temporary file rather than over the command line, for
    the reason `semantics.py` does the same: a few thousand expressions is past
    what an argument list holds.

    Args:
        columns: The column declarations, spelled as DuckDB spells them.
        rows: The rows, each one a list of SQL literals.
        expressions: The expressions to evaluate.

    Returns:
        The answers, one per line, in order.

    Raises:
        RuntimeError: If the child could not answer.
    """
    import subprocess
    import tempfile

    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".txt", encoding="utf-8", delete=False
    )
    try:
        with handle:
            handle.write(f"{len(columns)}\n")
            for name in columns:
                handle.write(f"{name}\n")
            handle.write(f"{len(rows)}\n")
            for row in rows:
                handle.write(BETWEEN.join(row) + "\n")
            for expression in expressions:
                handle.write(f"{expression}\n")
        child = subprocess.run(
            [sys.executable, os.path.abspath(__file__), handle.name],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.unlink(handle.name)

    if child.returncode != 0:
        raise RuntimeError(
            f"asking DuckDB about {len(expressions)} expressions exited"
            f" {child.returncode}: {child.stderr.strip()}"
        )
    return child.stdout


def main() -> int:
    """Answers one batch written to a file, which is what `answers_of` runs.

    Returns:
        A process exit status.
    """
    with open(sys.argv[1], encoding="utf-8") as handle:
        lines = handle.read().split("\n")

    at = 0
    count = int(lines[at])
    at += 1
    columns = lines[at : at + count]
    at += count
    height = int(lines[at])
    at += 1
    rows = [lines[at + which].split(BETWEEN) for which in range(height)]
    at += height
    expressions = [line for line in lines[at:] if line]

    for answer in answers_for(columns, rows, expressions):
        sys.stdout.write(answer.replace("\n", " ") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
