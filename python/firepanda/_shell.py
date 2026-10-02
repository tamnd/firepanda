"""`firepanda`, the SQL shell: a query in DuckDB's dialect in, a table out.

Run with no arguments at a terminal and it is interactive: a statement runs
when a line ends it with a semicolon, so one statement can take several lines,
and the history is kept between runs. `-c SQL` runs its statements and exits,
and statements piped in on standard input run the same way. Those two are not
interactive, and they start with `enable_external_access` turned off, because
the text may have been put together by a script rather than typed by the
person running the command. The setting cannot be turned back on from inside,
by DuckDB's rule.

Every statement runs in one session, so a `CREATE TABLE` is there for the next
one, a `SET` holds until a `RESET`, and a `PREPARE` is there to `EXECUTE`.

A line starting with a dot is a command to the shell rather than SQL, as it is
in DuckDB's shell. `.help` lists them.

Results print as DuckDB's shell prints them by default: a box, the column names
over their types, numbers to the right, and `NULL` for a missing value. `.mode`
switches to `csv`, `list` or `line`.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from collections.abc import Callable, Iterator
from typing import TextIO

from . import _firepanda
from ._frame import DataFrame, SqlSession

MODES = ("duckbox", "box", "csv", "list", "line")
"""The output modes `.mode` takes. `box` is `duckbox` without the type row."""

SHOWN = 40
"""How many rows a box shows before it shows the first and last halves."""

PROMPT = "F "
"""The prompt at the start of a statement, which is DuckDB's `D ` for firepanda."""

MORE = "· "
"""The prompt on a line that continues a statement."""

_HELP = """\
.exit                 Leaves the shell. .quit is the same.
.help                 Shows this.
.import FILE TABLE    Reads a CSV file into a table.
.mode MODE            Prints results as duckbox, box, csv, list or line.
.read FILE            Runs the statements in a file.
.tables               Lists the tables and views.
.timer on|off         Prints how long each statement took.
"""

# The type row says what DuckDB would call each column, which is not always what
# the dtype is called.
_TYPES = {
    "float64": "double",
    "float32": "float",
    "bool": "boolean",
    "boolean": "boolean",
    "object": "varchar",
    "str": "varchar",
    "string": "varchar",
}


def _type_of(dtype: object) -> str:
    """What DuckDB's shell calls a column of this dtype."""
    text = str(dtype)
    if text in _TYPES:
        return _TYPES[text]
    if text.startswith("datetime64"):
        return "timestamp with time zone" if "," in text else "timestamp"
    if text.startswith("timedelta64"):
        return "interval"
    return text.lower()


def _cell(value: object) -> str:
    """One value as DuckDB's shell prints it."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and math.isnan(value):
        return "NULL"
    return str(value)


def _numeric(dtype: object) -> bool:
    """Whether a column's values sit to the right of their box."""
    text = str(dtype)
    return text.startswith(("int", "uint", "float", "decimal"))


def statements(text: str) -> tuple[list[str], str]:
    """Splits text into whole statements and whatever is left unfinished.

    A semicolon ends a statement unless it is inside a string, a quoted name or
    a comment. The rest after the last one is handed back, so a shell can keep
    reading lines until it is finished too.

    Args:
        text: SQL, possibly several statements.

    Returns:
        The statements, each without its semicolon and with blank ones dropped,
        and the unfinished rest.
    """
    found: list[str] = []
    start = 0
    at = 0
    while at < len(text):
        char = text[at]
        if char in "'\"":
            close = text.find(char, at + 1)
            # A doubled quote is the quote itself, so the search goes on past it.
            while close != -1 and text[close + 1 : close + 2] == char:
                close = text.find(char, close + 2)
            if close == -1:
                return found, text[start:]
            at = close + 1
            continue
        if text.startswith("--", at):
            end = text.find("\n", at)
            at = len(text) if end == -1 else end + 1
            continue
        if text.startswith("/*", at):
            end = text.find("*/", at + 2)
            if end == -1:
                return found, text[start:]
            at = end + 2
            continue
        if char == ";":
            statement = text[start:at].strip()
            if statement:
                found.append(statement)
            start = at + 1
        at += 1
    rest = text[start:]
    return found, rest if rest.strip() else ""


class Shell:
    """One session and the settings of the shell around it."""

    def __init__(self, out: TextIO, *, interactive: bool) -> None:
        """Starts a session.

        Args:
            out: Where results go.
            interactive: Whether a person is typing. Otherwise the session
                starts without access to the file system.
        """
        self.session = SqlSession(_firepanda.SqlSession())
        self.out = out
        self.mode = "duckbox"
        self.timer = False
        self.failed = False
        if not interactive:
            self.session.execute("SET enable_external_access = false")

    def run(self, text: str) -> None:
        """Runs the statements and dot commands in some text, in order.

        A dot command is a line of its own between statements. A line inside
        an unfinished statement that happens to start with a dot is SQL.
        """
        pending = ""
        for line in text.splitlines(keepends=True):
            if not pending.strip() and line.lstrip().startswith("."):
                self.command(line.strip())
                continue
            pending += line
            found, pending = statements(pending)
            for statement in found:
                self.statement(statement)
        if pending.strip():
            self.statement(pending.strip())

    def statement(self, sql: str) -> None:
        """Runs one statement and prints what it answers, or why it did not."""
        began = time.perf_counter()
        try:
            answer = self.session.execute(sql)
        except (ValueError, NotImplementedError) as error:
            self.failed = True
            print(f"Error: {error}", file=sys.stderr)
            return
        took = time.perf_counter() - began
        if len(answer.columns) > 0:
            self.out.write(render(answer, self.mode))
        if self.timer:
            self.out.write(f"Run Time (s): real {took:.3f}\n")

    def command(self, line: str) -> None:
        """Runs one dot command."""
        words = line.split()
        name, rest = words[0], words[1:]
        if name in (".exit", ".quit"):
            raise SystemExit(1 if self.failed else 0)
        if name == ".help":
            self.out.write(_HELP)
        elif name == ".mode" and len(rest) == 1 and rest[0] in MODES:
            self.mode = rest[0]
        elif name == ".timer" and len(rest) == 1 and rest[0] in ("on", "off"):
            self.timer = rest[0] == "on"
        elif name == ".tables":
            for table in sorted(self.session.names(), key=str.lower):
                self.out.write(table + "\n")
        elif name == ".read" and len(rest) == 1:
            with open(rest[0]) as file:
                self.run(file.read())
        elif name == ".import" and len(rest) == 2:
            from ._pandas import read_csv

            self.session.register(rest[1], read_csv(rest[0]))
        else:
            self.failed = True
            print(f"Error: unknown command or wrong arguments: {line}", file=sys.stderr)


def render(frame: DataFrame, mode: str) -> str:
    """A result as the shell prints it.

    Args:
        frame: The result.
        mode: One of `MODES`.

    Returns:
        The text, ending in a newline.
    """
    names = [str(name) for name in frame.columns]
    dtypes = [frame[name].dtype for name in frame.columns]
    columns = [[_cell(value) for value in frame[name].tolist()] for name in frame.columns]
    rows = len(frame)
    if mode == "csv":
        lines = [",".join(_csv(name) for name in names)]
        lines += [",".join(_csv(column[i]) for column in columns) for i in range(rows)]
        return "\n".join(lines) + "\n"
    if mode == "list":
        lines = ["|".join(names)]
        lines += ["|".join(column[i] for column in columns) for i in range(rows)]
        return "\n".join(lines) + "\n"
    if mode == "line":
        width = max(len(name) for name in names)
        blocks = [
            "\n".join(
                f"{name:>{width}} = {column[i]}"
                for name, column in zip(names, columns, strict=True)
            )
            for i in range(rows)
        ]
        return "\n\n".join(blocks) + "\n" if blocks else ""
    return _box(
        names, [_type_of(dtype) for dtype in dtypes], dtypes, columns, rows, mode == "duckbox"
    )


def _csv(text: str) -> str:
    """One field of a CSV line, quoted when it has to be."""
    if any(char in text for char in ',"\n\r'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _box(
    names: list[str],
    types: list[str],
    dtypes: list[object],
    columns: list[list[str]],
    rows: int,
    typed: bool,
) -> str:
    """The box DuckDB's shell draws, with its first and last rows when long."""
    shown = list(range(rows))
    cut = rows > SHOWN
    if cut:
        shown = list(range(SHOWN // 2)) + list(range(rows - SHOWN // 2, rows))
    widths = []
    for name, kind, column in zip(names, types, columns, strict=True):
        width = max([len(name), len(kind) if typed else 0] + [len(column[i]) for i in shown])
        widths.append(width)

    def line(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def row(cells: Iterator[str]) -> str:
        return "│" + "│".join(f" {cell} " for cell in cells) + "│"

    out = [line("┌", "┬", "┐")]
    out.append(row(name.center(width) for name, width in zip(names, widths, strict=True)))
    if typed:
        out.append(row(kind.center(width) for kind, width in zip(types, widths, strict=True)))
    out.append(line("├", "┼", "┤"))
    for at, i in enumerate(shown):
        if cut and at == SHOWN // 2:
            for _ in range(3):
                out.append(row("·".center(width) for width in widths))
        out.append(
            row(
                column[i].rjust(width) if _numeric(dtype) else column[i].ljust(width)
                for column, width, dtype in zip(columns, widths, dtypes, strict=True)
            )
        )
    out.append(line("└", "┴", "┘"))
    if cut:
        out.append(f"{rows} rows ({SHOWN} shown)")
    return "\n".join(out) + "\n"


def _complete(shell: Shell) -> Callable[[str, int], str | None]:
    """A readline completer over the session's names, the functions and the keywords."""
    keywords = [
        "SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "JOIN",
        "LEFT", "RIGHT", "FULL", "INNER", "OUTER", "ON", "USING", "AS", "AND", "OR",
        "NOT", "NULL", "IS", "IN", "BETWEEN", "LIKE", "CASE", "WHEN", "THEN", "ELSE",
        "END", "DISTINCT", "UNION", "ALL", "EXCEPT", "INTERSECT", "WITH", "CREATE",
        "TABLE", "VIEW", "INSERT", "INTO", "VALUES", "DROP", "SET", "RESET", "PRAGMA",
        "PREPARE", "EXECUTE", "DEALLOCATE", "BEGIN", "COMMIT", "ROLLBACK", "QUALIFY",
        "WINDOW", "OVER", "PARTITION", "PIVOT", "UNPIVOT", "DESC", "ASC",
    ]  # fmt: skip
    functions = [name for name in shell.session.functions() if re.fullmatch(r"\w+", name)]
    matches: list[str] = []

    def complete(text: str, state: int) -> str | None:
        nonlocal matches
        if state == 0:
            folded = text.lower()
            pool = shell.session.names() + functions
            matches = [word for word in pool if word.lower().startswith(folded)]
            matches += [word for word in keywords if word.lower().startswith(folded)]
        return matches[state] if state < len(matches) else None

    return complete


def _interact(shell: Shell) -> int:
    """Reads statements from a person until they leave."""
    history = os.path.expanduser("~/.firepanda_history")
    try:
        import readline

        readline.set_completer(_complete(shell))
        readline.set_completer_delims(" \t\n(),;.")
        readline.parse_and_bind("tab: complete")
        if os.path.exists(history):
            readline.read_history_file(history)
    except ImportError:
        readline = None  # type: ignore[assignment]
    print(f"firepanda {_firepanda.version()}")
    print('Enter ".help" for usage hints.')
    pending = ""
    try:
        while True:
            try:
                line = input(MORE if pending else PROMPT)
            except KeyboardInterrupt:
                print()
                pending = ""
                continue
            if not pending and line.lstrip().startswith("."):
                shell.command(line.strip())
                continue
            pending += line + "\n"
            found, pending = statements(pending)
            for statement in found:
                shell.statement(statement)
    except EOFError:
        print()
        return 0
    except SystemExit as leaving:
        return int(leaving.code or 0)
    finally:
        if readline is not None:
            readline.write_history_file(history)


def main(argv: list[str] | None = None) -> int:
    """Runs the shell.

    Args:
        argv: The arguments after the program name, or the process's own.

    Returns:
        The exit status: 1 if a statement failed, 0 otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="firepanda", description="Runs SQL in DuckDB's dialect over firepanda."
    )
    parser.add_argument(
        "-c", dest="commands", action="append", default=[], metavar="SQL",
        help="run the statements and exit, without file system access",
    )  # fmt: skip
    for mode in MODES:
        parser.add_argument(
            f"-{mode}", dest="mode", action="store_const", const=mode,
            help=f"print results in {mode} mode",
        )  # fmt: skip
    arguments = parser.parse_args(argv)
    interactive = not arguments.commands and sys.stdin.isatty()
    shell = Shell(sys.stdout, interactive=interactive)
    if arguments.mode:
        shell.mode = arguments.mode
    if interactive:
        return _interact(shell)
    try:
        if arguments.commands:
            for text in arguments.commands:
                shell.run(text)
        else:
            shell.run(sys.stdin.read())
    except SystemExit as leaving:
        return int(leaving.code or 0)
    return 1 if shell.failed else 0
