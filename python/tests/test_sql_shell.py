"""`firepanda`, the SQL shell, run the way a script runs it.

The boxes are DuckDB 1.5's shell's own output for the same statements.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest


def _shell(firepanda: ModuleType) -> ModuleType:
    """The shell module, from the staged package."""
    return importlib.import_module(f"{firepanda.__name__}._shell")


def _run(
    firepanda: ModuleType, capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int, str, str]:
    """Runs the shell with some arguments and hands back its status and output."""
    status = _shell(firepanda).main(list(args))
    out, err = capsys.readouterr()
    return status, out, err


def test_a_semicolon_ends_a_statement_outside_quotes_and_comments(firepanda: ModuleType) -> None:
    """A semicolon in a string, a quoted name or a comment is not an end."""
    statements = _shell(firepanda).statements
    assert statements("SELECT 1; SELECT 2;") == (["SELECT 1", "SELECT 2"], "")
    assert statements("SELECT ';' AS s; SELECT") == (["SELECT ';' AS s"], " SELECT")
    assert statements("SELECT 'it''s;' AS s;") == (["SELECT 'it''s;' AS s"], "")
    assert statements('SELECT 1 AS ";";') == (['SELECT 1 AS ";"'], "")
    assert statements("SELECT 1 -- a;\n;") == (["SELECT 1 -- a;"], "")
    assert statements("SELECT /* ; */ 1;") == (["SELECT /* ; */ 1"], "")
    assert statements("SELECT 'open;") == ([], "SELECT 'open;")
    assert statements(" ; ;\n") == ([], "")


def test_a_result_prints_as_duckdbs_box(
    firepanda: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """Names over types, numbers to the right, text to the left, NULL for a hole."""
    status, out, err = _run(
        firepanda,
        capsys,
        "-c",
        "CREATE TABLE t(a BIGINT, b VARCHAR); INSERT INTO t VALUES (2, NULL), (10, 'x');"
        " SELECT a, b FROM t ORDER BY a",
    )
    assert (status, err) == (0, "")
    assert out == (
        "┌───────┐\n"
        "│ Count │\n"
        "│ int64 │\n"
        "├───────┤\n"
        "│     2 │\n"
        "└───────┘\n"
        "┌───────┬─────────┐\n"
        "│   a   │    b    │\n"
        "│ int64 │ varchar │\n"
        "├───────┼─────────┤\n"
        "│     2 │ NULL    │\n"
        "│    10 │ x       │\n"
        "└───────┴─────────┘\n"
    )


def test_every_statement_runs_in_one_session(
    firepanda: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """A table, a view and a setting made by one `-c` are there for the next."""
    status, out, err = _run(
        firepanda,
        capsys,
        "-csv",
        "-c",
        "CREATE TABLE t AS SELECT * FROM range(3) r(i)",
        "-c",
        "CREATE VIEW v AS SELECT i * 10 AS j FROM t",
        "-c",
        "SELECT sum(j) AS s FROM v",
    )
    assert (status, out, err) == (0, "s\n30\n", "")


def test_the_modes(firepanda: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    """`csv` quotes what has to be, `list` joins with bars, `line` is one per line."""
    query = "SELECT 1 AS n, 'a,b' AS s"
    assert _run(firepanda, capsys, "-csv", "-c", query) == (0, 'n,s\n1,"a,b"\n', "")
    assert _run(firepanda, capsys, "-list", "-c", query) == (0, "n|s\n1|a,b\n", "")
    assert _run(firepanda, capsys, "-line", "-c", query) == (0, "n = 1\ns = a,b\n", "")


def test_a_dot_command_between_statements(
    firepanda: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """A line starting with a dot is the shell's, and runs in its place in the text."""
    status, out, err = _run(
        firepanda,
        capsys,
        "-c",
        "CREATE TABLE b(x INTEGER);\nCREATE TABLE a(x INTEGER);\n"
        ".tables\n.mode csv\nSELECT 7 AS x;",
    )
    assert (status, out, err) == (0, "a\nb\nx\n7\n", "")


def test_a_failed_statement_is_reported_and_the_rest_run(
    firepanda: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """The error goes to standard error, the next statement still runs, and the status is 1."""
    status, out, err = _run(
        firepanda, capsys, "-csv", "-c", "SELECT nope FROM missing; SELECT 2 AS two"
    )
    assert status == 1
    assert out == "two\n2\n"
    assert err.startswith("Error: Catalog Error: Table with name missing does not exist!")


def test_a_script_cannot_give_itself_file_access(
    firepanda: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """`-c` starts with external access off, and it cannot be turned back on."""
    status, out, err = _run(firepanda, capsys, "-c", "SET enable_external_access = true")
    assert status == 1
    assert out == ""
    assert "Cannot enable external access" in err


def test_a_long_result_shows_its_ends(
    firepanda: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """Past forty rows the box keeps the first and last twenty and says how many there were."""
    status, out, _ = _run(firepanda, capsys, "-box", "-c", "SELECT * FROM range(100) r(i)")
    lines = out.splitlines()
    assert status == 0
    assert lines[1] == "│ i  │"
    assert lines[3] == "│  0 │"
    assert lines[22:26] == ["│ 19 │", "│ ·  │", "│ ·  │", "│ ·  │"]
    assert lines[26] == "│ 80 │"
    assert lines[-1] == "100 rows (40 shown)"
