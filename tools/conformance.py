"""The execution harness: DuckDB's `.test` files, run through firepanda.

`tools/corpus.py` takes the SQL out of the corpus and asks whether it parses.
This asks whether it runs and answers what the file says it answers, which is
the number document 11 publishes. It reads each file the way DuckDB's own
runner does, hands the statements to `build/differential/conformance` (the
program in `tests/differential/conformance.mojo`), and compares what comes back
with what the file expects.

A file passes when every statement in it does. A file fails at its first
statement that does not, and that statement puts the file in one of five
buckets:

    unsupported   firepanda refused by name, a document 05 refusal
    function      a function firepanda does not have, tier 2 or tier 3
    divergence    an error where DuckDB answered, or none where DuckDB
                  refused, or the wrong error, or the wrong number of columns
    wrong         an answer that is not the one the file expects
    crash         the program died or stopped answering

The last two are bugs. The first two are the roadmap. A refusal is told apart
from a divergence by whether the message names firepanda: DuckDB's messages
never do and every message firepanda writes for itself does.

Some files are not run, and are reported as not run rather than as failures:
a file that needs an extension or a setting of DuckDB's build, one that reads a
database file or restarts the database, one that reads the corpus's `data/`
directory, which the fetch does not take, and one using a directive this does
not know. They are counted beside each directory's rate so the denominator is
never a mystery.

Usage:

    pixi run conformance                   the target directories, checked
                                           against the floor
    python tools/conformance.py --all      every directory, with the report
    python tools/conformance.py --update   raise the floor to today's counts
    python tools/conformance.py --readme   write the table into README.md

See docs/specs/sql/11-conformance.md.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import corpus  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "build", "differential", "conformance")
FLOOR = os.path.join(ROOT, "tests", "conformance_floor.txt")
REPORT = os.path.join(ROOT, "build", "conformance")

# The directories the SELECT surface lives in. The floor is kept for these, and
# a change that lowers the count of passing files in any of them fails.
TARGETS = (
    "aggregate",
    "cte",
    "filter",
    "join",
    "limit",
    "order",
    "pivot",
    "projection",
    "select",
    "setops",
    "subquery",
    "topn",
    "window",
)

# The name the files' `onlyif` and `skipif` are read against. firepanda is
# checked against DuckDB's answers, so it reads the ones written for DuckDB.
ENGINE = "duckdb"

# A `require` that asks for a property of DuckDB's build rather than an
# extension. Each either holds here or has nothing to hold against.
HELD = {
    "64bit",
    "allow_unsigned_extensions",
    "core_functions",
    "no_alternative_verify",
    "no_block_verification",
    "no_extension_autoloading",
    "no_vector_verification",
    "noforcestorage",
    "notmingw",
    "notwindows",
    "skip_reload",
    "vector_size",
}

# The value lists `foreach` spells with angle brackets, as DuckDB's runner
# expands them.
_SIGNED = ["tinyint", "smallint", "integer", "bigint", "hugeint"]
_UNSIGNED = ["utinyint", "usmallint", "uinteger", "ubigint", "uhugeint"]
_INTEGRAL = _SIGNED + _UNSIGNED
_NUMERIC = _INTEGRAL + ["float", "double"]
EXPANSIONS = {
    "<signed>": _SIGNED,
    "<unsigned>": _UNSIGNED,
    "<integral>": _INTEGRAL,
    "<numeric>": _NUMERIC,
    "<alltypes>": _NUMERIC + ["bool", "interval", "varchar"],
}

# Directives that change nothing a run here can see.
IGNORED = ("hash-threshold", "tags", "reconnect", "set", "reset", "test-env")

_HASHED = re.compile(r"^(\d+) values hashing to ([0-9a-f]+)$")

# The phrases firepanda writes when a function is missing. Checked before the
# general rule, since each of them names firepanda too.
_MISSING = (
    "there is no function named",
    "has no kernel for the function",
    "has no fold for the aggregate",
    "Function with name",
)

BUCKETS = ("unsupported", "function", "divergence", "wrong", "crash")


class Step:
    """One statement to run and what the file says about it."""

    def __init__(self, line, kind, sql, expected, types="", sort="nosort", label=""):
        self.line = line
        self.kind = kind  # ok, error, maybe or query
        self.sql = sql
        self.expected = expected
        self.types = types
        self.sort = sort
        self.label = label


class Skipped(Exception):
    """A file this does not run, and why."""


def read_file(path: str) -> list[Step]:
    """Reads one test file into the statements to run.

    Args:
        path: The file.

    Returns:
        The steps, in order, with loops unrolled.

    Raises:
        Skipped: If the file is one this does not run.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().split("\n")
    except (OSError, UnicodeDecodeError):
        raise Skipped("not UTF-8") from None
    steps, _ = _block(lines, 0, {}, False)
    return steps


def _block(lines, at, bound, inner):
    """Reads records until the end of the file or the `endloop` that closes
    the loop being read.

    Returns:
        The steps and the line after the last one read.
    """
    steps = []
    skipping = False
    gate = None
    while at < len(lines):
        raw = lines[at]
        line = _bound(raw, bound).strip()
        at += 1
        if not line or line.startswith("#"):
            continue
        words = line.split()
        word = words[0]
        if word == "mode":
            if len(words) > 1 and words[1] == "skip":
                skipping = True
            elif len(words) > 1 and words[1] == "unskip":
                skipping = False
            continue
        if word == "endloop":
            if not inner:
                raise Skipped("an endloop with no loop")
            return steps, at
        if word in ("loop", "concurrentloop", "foreach", "concurrentforeach"):
            values = _values(words)
            start = at
            end = at
            for value in values:
                more = dict(bound)
                more[words[1]] = value
                body, end = _block(lines, start, more, True)
                if not skipping:
                    steps.extend(body)
            if not values:
                _, end = _block(lines, start, bound, True)
            at = end
            continue
        if skipping:
            if word in ("statement", "query"):
                at = _past(lines, at)
            continue
        if word in ("skipif", "onlyif"):
            engine = words[1] if len(words) > 1 else ""
            gate = (engine == ENGINE) == (word == "onlyif")
            continue
        if word == "halt":
            if gate is False:
                gate = None
                continue
            break
        if word == "require":
            wanted = words[1] if len(words) > 1 else ""
            if wanted not in HELD:
                raise Skipped("requires " + wanted)
            continue
        if word in ("require-env", "load", "restart", "unzip", "sleep"):
            raise Skipped(word)
        if word in IGNORED:
            continue
        if word == "include":
            # A path from the root of DuckDB's tree, whose records run here as
            # if they had been written here.
            if len(words) < 2:
                raise Skipped("an include with no file")
            try:
                with open(os.path.join(_corpus_root(), words[1]), encoding="utf-8") as handle:
                    included = handle.read().split("\n")
            except (OSError, UnicodeDecodeError):
                raise Skipped("includes " + words[1]) from None
            body, _ = _block(included, 0, bound, False)
            steps.extend(body)
            continue
        if word not in ("statement", "query"):
            raise Skipped("the directive " + word)

        run = gate is not False
        gate = None
        body = []
        first = at
        while at < len(lines) and lines[at].strip() and not lines[at].startswith("----"):
            body.append(_bound(lines[at], bound))
            at += 1
        expected = []
        if at < len(lines) and lines[at].startswith("----"):
            at += 1
            while at < len(lines) and lines[at].strip():
                expected.append(_bound(lines[at], bound))
                at += 1
        if not run:
            continue
        sql = "\n".join(body).strip()
        if "${" in sql:
            raise Skipped("a loop variable outside a loop")
        if "data/" in sql or "{DATA_DIR}" in sql:
            raise Skipped("reads data/")
        sql = sql.replace("__TEST_DIR__", tempfile.gettempdir())
        if word == "statement":
            kind = words[1] if len(words) > 1 else "ok"
            if kind not in ("ok", "error", "maybe"):
                raise Skipped("statement " + kind)
            steps.append(Step(first, kind, sql, expected))
            continue
        types = words[1] if len(words) > 1 else ""
        sort = "nosort"
        label = ""
        for extra in words[2:]:
            if extra in ("nosort", "rowsort", "valuesort", "sort"):
                sort = extra
            else:
                label = extra
        steps.append(Step(first, "query", sql, expected, types, sort, label))
    if inner:
        raise Skipped("a loop with no endloop")
    return steps, at


def _corpus_root():
    return corpus.cache()


def _past(lines, at):
    """The line after a record's SQL and expected output."""
    while at < len(lines) and lines[at].strip() and not lines[at].startswith("----"):
        at += 1
    if at < len(lines) and lines[at].startswith("----"):
        at += 1
        while at < len(lines) and lines[at].strip():
            at += 1
    return at


def _values(words):
    """What a loop's variable takes, in order."""
    if words[0].endswith("loop"):
        if len(words) < 4:
            raise Skipped("a loop without bounds")
        return [str(i) for i in range(int(words[2]), int(words[3]))]
    values = []
    for word in words[2:]:
        if word.startswith("<"):
            if word not in EXPANSIONS:
                raise Skipped("the value list " + word)
            values.extend(EXPANSIONS[word])
        else:
            values.append(word)
    return values


def _bound(line, bound):
    for name, value in bound.items():
        line = line.replace("${" + name + "}", value)
    return line


class Outcome:
    """What one file came to."""

    def __init__(self, path):
        self.path = path
        self.skipped = ""
        self.bucket = ""
        self.line = 0
        self.detail = ""

    @property
    def passed(self):
        return not self.skipped and not self.bucket


def run_files(paths, root, jobs, stall):
    """Runs files through the runner, several runners at once.

    Returns:
        One outcome per path, in the order given.
    """
    outcomes = {}
    work = []
    for path in paths:
        outcome = Outcome(path)
        outcomes[path] = outcome
        try:
            steps = read_file(os.path.join(root, path))
        except Skipped as why:
            outcome.skipped = str(why)
            continue
        if not steps:
            outcome.skipped = "nothing to run"
            continue
        work.append((path, steps))

    shards = [work[i::jobs] for i in range(jobs)]
    threads = [
        threading.Thread(target=_shard, args=(shard, outcomes, stall))
        for shard in shards
        if shard
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [outcomes[path] for path in paths]


def _shard(work, outcomes, stall):
    """Runs one shard's files, starting the runner again after a crash."""
    with tempfile.NamedTemporaryFile("wb", suffix=".steps", delete=False) as handle:
        for path, steps in work:
            handle.write(f"F {path}\n".encode())
            for step in steps:
                body = step.sql.encode("utf-8")
                handle.write(f"S {len(body)}\n".encode() + body + b"\n")
        steps_path = handle.name
    try:
        done = 0
        while done < len(work):
            done = _one_run(work, done, steps_path, outcomes, stall)
    finally:
        os.unlink(steps_path)


def _one_run(work, start, steps_path, outcomes, stall):
    """Runs the runner from file `start` until it finishes or dies.

    Returns:
        The index of the first file not yet settled.
    """
    environment = dict(os.environ)
    environment["FIREPANDA_CONFORMANCE_STEPS"] = steps_path
    environment["FIREPANDA_CONFORMANCE_SKIP"] = str(start)
    process = subprocess.Popen(
        [RUNNER],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    records = queue.Queue()
    reader = threading.Thread(target=_records, args=(process.stdout, records))
    reader.daemon = True
    reader.start()

    current = start - 1
    results = []
    while True:
        try:
            record = records.get(timeout=stall)
        except queue.Empty:
            process.kill()
            record = ("dead", "stopped answering for %d seconds" % stall)
        if record[0] == "file":
            if current >= start:
                _settle(work[current], results, outcomes)
            current += 1
            results = []
            continue
        if record[0] in ("rows", "error"):
            results.append(record)
            continue
        # The stream ended, or stopped, or said something it should not have.
        code = process.wait()
        if record[0] == "eof" and code == 0 and current == len(work) - 1:
            _settle(work[current], results, outcomes)
            return len(work)
        why = record[1] if record[0] == "dead" else "the runner exited with %d" % code
        if current < start:
            current = start
            results = []
        _settle(work[current], results, outcomes)
        outcome = outcomes[work[current][0]]
        if outcome.bucket == "crash":
            outcome.detail = why
        return current + 1


def _records(stream, records):
    """Reads the runner's records into a queue, ending with `eof` or `dead`."""
    while True:
        head = stream.readline()
        if not head:
            records.put(("eof",))
            return
        head = head.decode("utf-8", "replace").rstrip("\n")
        if head.startswith("F "):
            records.put(("file", head[2:]))
            continue
        if head.startswith("R "):
            _, columns, size = head.split(" ")
            body = stream.read(int(size))
            stream.read(1)
            records.put(("rows", int(columns), body.decode("utf-8", "replace")))
            continue
        if head.startswith("E "):
            body = stream.read(int(head[2:]))
            stream.read(1)
            records.put(("error", body.decode("utf-8", "replace")))
            continue
        # The suite around the runner writes its own header and summary, which
        # are not records and say nothing the exit status does not.


def _settle(item, results, outcomes):
    """Compares one file's results with what it expects."""
    path, steps = item
    outcome = outcomes[path]
    if outcome.bucket:
        return
    labels = {}
    for i, step in enumerate(steps):
        if i >= len(results):
            outcome.bucket = "crash"
            outcome.line = step.line
            outcome.detail = "no answer"
            return
        verdict = _judge(step, results[i], labels)
        if verdict:
            outcome.bucket, outcome.detail = verdict
            outcome.line = step.line
            return


def _judge(step, result, labels):
    """Returns nothing when a statement did what the file says, and otherwise
    the bucket and a line saying what happened."""
    if result[0] == "error":
        message = result[1]
        if step.kind in ("error", "maybe"):
            if step.kind == "error" and step.expected:
                wanted = "\n".join(step.expected)
                if not _matches(message, wanted):
                    return "divergence", "the error was %r, not %r" % (
                        _first(message),
                        _first(wanted),
                    )
            return None
        return _refused(message), _first(message)
    if step.kind == "error":
        return "divergence", "answered where DuckDB raises"
    if step.kind != "query":
        return None
    columns, text = result[1], result[2]
    rows = [row.split("\t") for row in text.split("\n")] if text else []
    if rows and columns == 0:
        rows = []
    if step.types and columns != len(step.types):
        return "divergence", "%d columns where DuckDB has %d" % (
            columns,
            len(step.types),
        )
    values = _ordered(rows, step.sort)
    if not step.expected:
        if step.label:
            if step.label in labels and labels[step.label] != values:
                return "wrong", "not the rows of the earlier query " + step.label
            labels[step.label] = values
        return None
    if step.label:
        labels[step.label] = values
    hashed = _HASHED.match(step.expected[0].strip()) if len(step.expected) == 1 else None
    if hashed:
        count, digest = int(hashed.group(1)), hashed.group(2)
        flat = [value for row in values for value in row]
        if len(flat) != count:
            return "wrong", "%d values where DuckDB has %d" % (len(flat), count)
        mine = hashlib.md5("".join(v + "\n" for v in flat).encode()).hexdigest()
        if mine != digest:
            return "wrong", "the values hash to something else"
        return None
    wanted = _expected(step.expected, columns, len(rows))
    wanted = _ordered(wanted, step.sort)
    if len(wanted) != len(values):
        return "wrong", "%d rows where DuckDB has %d" % (len(values), len(wanted))
    for got, want in zip(values, wanted):
        if len(got) != len(want) or not all(map(_same, got, want)):
            return "wrong", "%s where DuckDB has %s" % (
                "\t".join(got)[:120],
                "\t".join(want)[:120],
            )
    return None


def _expected(lines, columns, rows):
    """The expected rows, whether the file wrote a row per line or a value."""
    if columns > 1 and len(lines) == rows * columns and all("\t" not in l for l in lines):
        return [lines[i : i + columns] for i in range(0, len(lines), columns)]
    return [line.split("\t") for line in lines]


def _ordered(rows, sort):
    if sort == "rowsort" or sort == "sort":
        return sorted(rows)
    if sort == "valuesort":
        flat = sorted(value for row in rows for value in row)
        return [[value] for value in flat]
    return rows


def _same(got, want):
    """DuckDB's runner's rule for two values: the same text, or two numbers
    within its tolerance of each other."""
    if got == want:
        return True
    try:
        left = float(got)
        right = float(want)
    except ValueError:
        return False
    if left == right:
        return True
    return abs(left - right) <= abs(right) * 0.01 + 0.00000001


def _matches(message, wanted):
    if wanted.startswith("<REGEX>:"):
        return re.search(wanted[len("<REGEX>:") :], message, re.S) is not None
    if wanted.startswith("<!REGEX>:"):
        return re.search(wanted[len("<!REGEX>:") :], message, re.S) is None
    return wanted in message


def _refused(message):
    if any(phrase in message for phrase in _MISSING):
        return "function"
    if "firepanda" in message:
        return "unsupported"
    return "divergence"


def _first(text):
    return text.split("\n", 1)[0][:160]


def directory(path):
    parts = path.split(os.sep)
    # test/sql/<dir>/...
    return parts[2] if len(parts) > 3 else "(top)"


def tally(outcomes):
    """Per directory: passed, run, not run, and failures by bucket."""
    table = {}
    for outcome in outcomes:
        row = table.setdefault(
            directory(outcome.path),
            {"passed": 0, "run": 0, "skipped": 0, **{b: 0 for b in BUCKETS}},
        )
        if outcome.skipped:
            row["skipped"] += 1
            continue
        row["run"] += 1
        if outcome.passed:
            row["passed"] += 1
        else:
            row[outcome.bucket] += 1
    return table


def rendered(table):
    """The per directory table, as markdown."""
    out = [
        "| Directory | Passed | Rate | Not run | Unsupported | Function | Divergence | Wrong | Crash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(table):
        row = table[name]
        rate = "%.1f%%" % (100.0 * row["passed"] / row["run"]) if row["run"] else "-"
        # Truncated rather than rounded, so a rate never reads higher than it is.
        if row["run"]:
            rate = "%.1f%%" % (int(1000.0 * row["passed"] / row["run"]) / 10.0)
        out.append(
            "| test/sql/%s | %d/%d | %s | %d | %d | %d | %d | %d | %d |"
            % (
                name,
                row["passed"],
                row["run"],
                rate,
                row["skipped"],
                row["unsupported"],
                row["function"],
                row["divergence"],
                row["wrong"],
                row["crash"],
            )
        )
    return "\n".join(out)


def read_floor():
    floor = {}
    if os.path.exists(FLOOR):
        with open(FLOOR, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    name, count = line.split()
                    floor[name] = int(count)
    return floor


def write_floor(table):
    with open(FLOOR, "w", encoding="utf-8") as handle:
        handle.write(
            "# Files passing per target directory of DuckDB's corpus. Written by\n"
            "# `python tools/conformance.py --update` and checked by `pixi run\n"
            "# conformance`, which fails when any count here goes down.\n"
        )
        for name in sorted(table):
            if name in TARGETS:
                handle.write("%s %d\n" % (name, table[name]["passed"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--all", action="store_true", help="every directory")
    parser.add_argument("--update", action="store_true", help="raise the floor")
    parser.add_argument("--readme", action="store_true", help="write README.md")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--stall", type=int, default=60)
    parser.add_argument("paths", nargs="*", help="files or directories under test/sql")
    args = parser.parse_args(argv)

    root = corpus.cache()
    if not os.path.isdir(os.path.join(root, "test", "sql")):
        print("no corpus at %s; run pixi run corpus first" % root, file=sys.stderr)
        return 2
    if not os.path.exists(RUNNER):
        print("no runner at %s; run pixi run conformance-build" % RUNNER, file=sys.stderr)
        return 2

    wanted = args.paths or (
        ["test/sql"] if args.all or args.readme else ["test/sql/" + t for t in TARGETS]
    )
    paths = []
    for want in wanted:
        full = os.path.join(root, want)
        if os.path.isfile(full):
            paths.append(want)
            continue
        for dirpath, dirnames, filenames in os.walk(full):
            dirnames.sort()
            for name in sorted(filenames):
                if name.endswith(".test") or name.endswith(".test_slow"):
                    paths.append(os.path.relpath(os.path.join(dirpath, name), root))

    outcomes = run_files(paths, root, args.jobs, args.stall)
    table = tally(outcomes)
    text = rendered(table)
    print(text)

    os.makedirs(REPORT, exist_ok=True)
    with open(os.path.join(REPORT, "failures.txt"), "w", encoding="utf-8") as handle:
        for outcome in outcomes:
            if outcome.skipped:
                handle.write("skip\t%s\t%s\n" % (outcome.path, outcome.skipped))
            elif outcome.bucket:
                handle.write(
                    "%s\t%s:%d\t%s\n"
                    % (outcome.bucket, outcome.path, outcome.line, outcome.detail)
                )
    with open(os.path.join(REPORT, "table.md"), "w", encoding="utf-8") as handle:
        handle.write(text + "\n")

    if args.readme:
        _readme(text)
    if args.update:
        write_floor(table)
        return 0

    status = 0
    floor = read_floor()
    for name, count in sorted(floor.items()):
        have = table.get(name, {}).get("passed", 0)
        if have < count:
            print("test/sql/%s: %d files pass, below the floor of %d" % (name, have, count))
            status = 1
        elif have > count:
            print(
                "test/sql/%s: %d files pass, above the floor of %d; raise it with --update"
                % (name, have, count)
            )
    return status


def _readme(text):
    path = os.path.join(ROOT, "README.md")
    with open(path, encoding="utf-8") as handle:
        readme = handle.read()
    opens, closes = "<!-- sql-conformance -->", "<!-- end sql-conformance -->"
    start = readme.index(opens) + len(opens)
    end = readme.index(closes)
    readme = readme[:start] + "\n\n" + text + "\n\n" + readme[end:]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(readme)


if __name__ == "__main__":
    sys.exit(main())
