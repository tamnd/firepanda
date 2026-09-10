"""DuckDB's SQL test corpus, turned into a flat list of statements.

The corpus is 4,796 files in sqllogictest format, which is a plain text format
where a directive line introduces a block and a run of `-` characters separates
the block's SQL from its expected output. Only the SQL matters here. The
expected output does not, and neither does the ok against error annotation,
because the oracle for accept against reject is DuckDB's own parser and not what
somebody wrote in a test file years ago. A `statement error` is very often a
binder error, which means the parser was perfectly happy with it.

Two entry points.

As a script this writes `statements.txt` next to the corpus, one record per
statement, in a format the Mojo harness can walk without a parser of its own:

    <byte length>\\t<path>:<line>\\n
    <the SQL>\\n

As a module `verdicts` and `verdicts_of` hand the Mojo harness DuckDB's answer
for every one of those statements in a single string, one character each. That
is one crossing of the CPython boundary rather than seventy thousand of them.
Both of them run DuckDB in a child process, which is deliberate and is explained
on `verdicts`.

See docs/specs/sql/11-conformance.md.
"""

from __future__ import annotations

import os
import sys

# The directives that introduce SQL. `statement maybe` is DuckDB's marker for a
# statement whose result depends on the build, and it is still SQL.
STARTS = ("statement ", "query ")

# A separator line between a block's SQL and its expected output. The format
# writes exactly four, but nothing is lost by accepting a longer run.
SEPARATOR = "----"

ACCEPTED = "1"
REJECTED = "0"
UNDECIDED = "x"


def statements(root: str):
    """Yields every SQL statement in a corpus tree.

    Args:
        root: The directory holding the `.test` files.

    Yields:
        A path relative to root, a one based line number, and the SQL.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.startswith(".") and ".test" in name:
                path = os.path.join(dirpath, name)
                for line, sql in _in_file(path):
                    yield os.path.relpath(path, root), line, sql


def _in_file(path: str):
    """Yields the statements in one test file.

    Args:
        path: The file.

    Yields:
        A one based line number and the SQL that starts on it.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().split("\n")
    except (OSError, UnicodeDecodeError):
        # A file that is not UTF-8 is not a file DuckDB's own runner reads
        # either, so it is not a corpus statement we are missing.
        return

    at = 0
    skipping = False
    while at < len(lines):
        line = lines[at]
        # `mode skip` disables everything after it until `mode unskip` or the end
        # of the file. The statements in there are ones DuckDB itself has turned
        # off, usually as flaky, and running them would be asking a question
        # upstream has already declined to answer.
        if line.startswith("mode "):
            word = line[5:].strip()
            if word.startswith("skip"):
                skipping = True
            elif word.startswith("unskip"):
                skipping = False
            at += 1
            continue
        if skipping or not line.startswith(STARTS):
            at += 1
            continue

        body = []
        at += 1
        first = at
        while at < len(lines):
            if not lines[at].strip() or lines[at].startswith(SEPARATOR):
                break
            body.append(lines[at])
            at += 1
        # Walk off the end of the expected output too, so that a result row that
        # happens to begin with the word query is never read as a directive.
        if at < len(lines) and lines[at].startswith(SEPARATOR):
            at += 1
            while at < len(lines) and lines[at].strip():
                at += 1

        sql = "\n".join(body).strip()
        # `${name}` is a loop variable. Substituting it would mean running the
        # loop, and leaving it in feeds the tokenizer a `$` that is not a
        # parameter, so the honest thing is to leave these out and say so.
        if sql and "${" not in sql:
            yield first + 1, sql


def write(root: str, out: str) -> int:
    """Writes every statement in a corpus tree to one file.

    Args:
        root: The directory holding the `.test` files.
        out: The file to write.

    Returns:
        How many statements were written.
    """
    count = 0
    with open(out, "w", encoding="utf-8") as handle:
        for path, line, sql in statements(root):
            body = sql.encode("utf-8")
            handle.write(f"{len(body)}\t{path}:{line}\n{sql}\n")
            count += 1
    return count


def read(path: str):
    """Reads back what `write` wrote.

    Args:
        path: The statements file.

    Returns:
        A list of location and SQL pairs, in the order they were written.
    """
    with open(path, "rb") as handle:
        data = handle.read()

    out = []
    at = 0
    while at < len(data):
        end = data.index(b"\n", at)
        length, _, where = data[at:end].partition(b"\t")
        start = end + 1
        stop = start + int(length)
        out.append((where.decode("utf-8"), data[start:stop].decode("utf-8")))
        at = stop + 1
    return out


def cache() -> str:
    """Returns the directory the pinned corpus is fetched into.

    The commit comes out of the grammar's VENDOR file, so the corpus and the
    grammar cannot drift apart, and the directory is named after it, so two
    commits can sit side by side and a bisect over a grammar bump still works.

    Returns:
        An absolute path, which may not exist yet.

    Raises:
        FileNotFoundError: If there is no vendored grammar to read a commit from.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vendor = os.path.join(root, "firepanda", "sql", "grammar", "VENDOR")
    sha = ""
    with open(vendor, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("commit:"):
                sha = line.split(":", 1)[1].strip()
                break
    if not sha:
        raise FileNotFoundError(f"{vendor} has no commit line")

    outer = os.environ.get("FIREPANDA_CORPUS") or os.path.join(
        root, ".cache", "duckdb-corpus"
    )
    return os.path.join(outer, sha)


def location() -> str:
    """Returns the statements file for the corpus the grammar is pinned to.

    Returns:
        An absolute path.

    Raises:
        FileNotFoundError: If the corpus has not been fetched and extracted, with
            the command that does it.
    """
    path = os.path.join(cache(), "statements.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} is not there. Run `pixi run corpus` to fetch DuckDB's test"
            " corpus at the pinned commit and extract the statements from it."
        )
    return path


def verdicts_for(sqls) -> str:
    """Asks DuckDB to parse every statement it is given and reports what it said.

    Only the parser runs. `extract_statements` splits a string into statements
    and builds nothing else, so a column that does not exist and a function with
    no overload are not errors here, which is exactly the line the compatibility
    claim is drawn along.

    One connection for the whole batch, because opening one costs more than
    parsing a statement does.

    Call this from a plain Python process. `verdicts` and `verdicts_of` are what
    an embedded interpreter should call, and they run this in a child.

    Args:
        sqls: An iterable of statements.

    Returns:
        One character per statement, in order: `1` accepted, `0` rejected with a
        parser error, `x` refused for some other reason.
    """
    import duckdb

    connection = duckdb.connect()
    out = []
    for sql in sqls:
        try:
            connection.extract_statements(sql)
            out.append(ACCEPTED)
        except duckdb.ParserException:
            out.append(REJECTED)
        except Exception:
            out.append(UNDECIDED)
    connection.close()
    return "".join(out)


def verdicts_of(sqls) -> str:
    """Asks DuckDB about a batch of statements the caller already has.

    Same child process as `verdicts` and for the same reason. The statements go
    over a temporary file in the format `write` uses rather than over a pipe,
    because SQL has newlines in it and a length prefix is the one framing that
    does not need escaping.

    Args:
        sqls: An iterable of statements.

    Returns:
        One character per statement, in order.

    Raises:
        RuntimeError: If the child could not answer.
    """
    import tempfile

    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".txt", encoding="utf-8", delete=False
    )
    try:
        with handle:
            for sql in sqls:
                handle.write(f"{len(sql.encode('utf-8'))}\tbatch\n{sql}\n")
        return verdicts(handle.name)
    finally:
        os.unlink(handle.name)


def verdicts(path: str) -> str:
    """Asks DuckDB about every statement in a file and reports what it said.

    The asking happens in a child process. The caller is a Mojo binary with
    CPython embedded in it, and `import duckdb` in that interpreter loads an
    extension module that registers process exit handlers. Those handlers run
    after the embedded interpreter has been torn down, the connection they go
    looking for is gone, and the process dies in a destructor with `corrupted
    double-linked list` on Linux and a heap trace on macOS. The report has
    already been printed by then, so what is lost is the exit status, and on CI
    the job hangs until the runner kills it. See
    https://github.com/tamnd/firepanda/issues/359.

    A child process does the same work, prints the answer, and exits before
    anything of ours has started shutting down. It costs one fork and one pipe
    for the whole corpus.

    Args:
        path: The statements file written by `write`.

    Returns:
        One character per statement, in the order they were written.

    Raises:
        RuntimeError: If the child could not answer.
    """
    import subprocess

    child = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "verdicts", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if child.returncode != 0:
        raise RuntimeError(
            f"asking DuckDB about {path} exited {child.returncode}:"
            f" {child.stderr.strip()}"
        )
    return child.stdout.strip()


def main() -> int:
    """Writes the statements file for a fetched corpus.

    With `verdicts <path>` it instead prints DuckDB's answer for every statement
    in a file, which is what `verdicts` above runs in a child.

    Returns:
        A process exit status.
    """
    if len(sys.argv) > 2 and sys.argv[1] == "verdicts":
        sys.stdout.write(verdicts_for(sql for _, sql in read(sys.argv[2])))
        return 0

    root = cache()
    tests = os.path.join(root, "test", "sql")
    if not os.path.isdir(tests):
        print(
            f"{tests} is not there, so run tools/fetch_corpus.sh first",
            file=sys.stderr,
        )
        return 1

    out = os.path.join(root, "statements.txt")
    count = write(tests, out)
    print(f"wrote {count} statements to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
