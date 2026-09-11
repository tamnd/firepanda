"""What DuckDB says the type of an expression is.

The corpus differential in `tests/differential/sql.mojo` asks DuckDB whether a
statement parses. This asks it something the parser cannot answer: given columns
of known types, what type does an expression over them have. That is the binder
speaking rather than the parser, and it is the oracle for everything
`firepanda/sql/arith.mojo`, `cast.mojo` and `resolve.mojo` decide.

The probe is a table with one column per type in the matrix and a single row of
nulls in it. A column is needed rather than a literal because DuckDB folds a
call whose arguments are all constant before anything can be read off it, and a
row is needed because `select typeof(...) from probe` over an empty table
returns no rows to read. The row being null costs nothing here, since `typeof`
is answered by the binder and never looks at a value.

The one entry point is `types_of`, which takes the column types and the
expressions and hands back one answer per expression. It runs DuckDB in a child
process for the reason `corpus.py` does: `import duckdb` inside the interpreter
embedded in a Mojo binary registers exit handlers that run after that
interpreter is gone, and the process then dies in a destructor with the report
already printed.

See docs/specs/sql/11-conformance.md.
"""

from __future__ import annotations

import os
import sys

REFUSED = "!"
"""What an answer starts with when DuckDB would not bind the expression.

The rest of the line is the first line of DuckDB's own message, which is what
makes a disagreement readable without running the query again by hand.
"""


def types_for(columns, expressions):
    """Asks DuckDB for the type of every expression over a set of columns.

    One connection and one table for the whole batch. Each expression is
    written in terms of `c0`, `c1` and so on, which are the columns in the order
    they were given.

    Call this from a plain Python process. `types_of` is what an embedded
    interpreter should call, and it runs this in a child.

    Args:
        columns: The column types, spelled as DuckDB spells them.
        expressions: The expressions to ask about.

    Returns:
        One answer per expression, in order: the type name, or `REFUSED`
        followed by the first line of DuckDB's message.
    """
    import duckdb

    connection = duckdb.connect()
    declared = ", ".join(f"c{at} {name}" for at, name in enumerate(columns))
    connection.execute(f"create table probe({declared})")
    connection.execute(
        f"insert into probe values ({', '.join(['NULL'] * len(columns))})"
    )

    out = []
    for expression in expressions:
        try:
            rows = connection.execute(
                f"select typeof({expression}) from probe"
            ).fetchall()
            out.append(rows[0][0])
        except Exception as error:
            out.append(REFUSED + str(error).split("\n")[0].strip())
    connection.close()
    return out


def types_of(columns, expressions) -> str:
    """Asks DuckDB about a batch the caller already has.

    The batch goes over a temporary file rather than over the command line,
    because a few thousand expressions is past what an argument list holds, and
    the answers come back over the child's output one to a line.

    Args:
        columns: The column types, spelled as DuckDB spells them.
        expressions: The expressions to ask about.

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
    """Answers one batch written to a file, which is what `types_of` runs.

    Returns:
        A process exit status.
    """
    with open(sys.argv[1], encoding="utf-8") as handle:
        lines = handle.read().split("\n")

    count = int(lines[0])
    columns = lines[1 : 1 + count]
    expressions = [line for line in lines[1 + count :] if line]
    for answer in types_for(columns, expressions):
        sys.stdout.write(answer.replace("\n", " ") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
