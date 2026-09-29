"""Runs the statements of DuckDB's `.test` files through firepanda.

This is the half of the execution harness that runs SQL. `tools/conformance.py`
is the other half: it reads the files, works out which statements to run and
what DuckDB says each should answer, hands this program the statements, and
compares what comes back. The comparison is in Python because the file format
is text all the way down and its answers are checked with md5, and keeping all
of that on one side leaves this side with one job.

The input is the steps file the driver writes, one record per line of header:

    F <path>            a new file, run against a fresh catalog
    S <bytes>           a statement, whose text follows on the next line

and the output is one record per record in:

    F <path>            the file about to run
    R <columns> <bytes> the statement answered, then a letter per column
                        for its type and a line per row
    E <bytes>           the statement raised, then the message

A row is its values joined by tabs, a null as `NULL` and empty text as
`(empty)`, which is how DuckDB's own runner writes them. The values are cast to
text by SQL's own `CAST(... AS VARCHAR)`, so a number prints the way the dialect
prints it rather than the way this program would.

Every record is flushed as it is written. A statement that takes the process
down leaves the driver the last file it started, which is how a crash is told
apart from a failure, and the second argument says how many files to pass over
so the driver can start again after it. Both arrive in the environment, as
`FIREPANDA_CONFORMANCE_STEPS` and `FIREPANDA_CONFORMANCE_SKIP`, rather than on
the command line, because the suite that runs the work reads the command line
as its own.

See docs/specs/sql/11-conformance.md.
"""

from std.os import getenv
from std.testing import TestSuite

from firepanda.array.any import AnyArray
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame
from firepanda.io.write import cell_text
from firepanda.sql.catalog import Catalog
from firepanda.sql.ddl import execute
from firepanda.sql.run import Dialect


def _column(frame: DataFrame, i: Int) raises -> AnyArray:
    """One column as a single array."""
    return frame.columns[i].copy().combine()


def _cast(dialect: Dialect, frame: DataFrame, i: Int) raises -> AnyArray:
    """Column `i` cast to text by SQL's own cast.

    The column goes into a catalog of its own under a name that cannot clash,
    so a result with two columns of one name, or a name no query could write,
    is still read one column at a time.
    """
    var fields = List[Field]()
    fields.append(Field("c", frame.schema[i].dtype))
    var columns = List[AnyArray]()
    columns.append(_column(frame, i))
    var scratch = Catalog()
    scratch.register("r", DataFrame(Schema(fields^), columns^))
    var cast = dialect.run("SELECT CAST(c AS VARCHAR) AS c FROM r", scratch)
    return _column(cast, 0)


def _kinds(frame: DataFrame) -> String:
    """One letter per column for the driver's comparison.

    DuckDB's runner compares a value that is not the text it expects under the
    column's type, so `true` answers a `1` it expects of a boolean. `B` is a
    boolean, `I` an integer, `R` a float and `T` anything else.
    """
    var out = String()
    for i in range(frame.width()):
        var dtype = frame.schema[i].dtype
        if dtype == LogicalType.BOOL:
            out += "B"
        elif dtype.is_integer():
            out += "I"
        elif dtype.is_float():
            out += "R"
        else:
            out += "T"
    return out^


def _rendered(dialect: Dialect, frame: DataFrame) raises -> String:
    """The rows, a tab between values and a newline between rows.

    A column SQL cannot cast to text yet is written the way a file would carry
    it, which spells a date as a date rather than as the day count it is stored
    as, and a row that prints is more use to the report than one that could
    not.
    """
    var width = frame.width()
    var texts = List[AnyArray]()
    var cast = List[Bool]()
    for i in range(width):
        try:
            texts.append(_cast(dialect, frame, i))
            cast.append(True)
        except:
            texts.append(_column(frame, i))
            cast.append(False)
    var rows = len(frame)
    var out = String()
    for r in range(rows):
        if r > 0:
            out += "\n"
        for c in range(width):
            if c > 0:
                out += "\t"
            if not texts[c].is_valid(r):
                out += "NULL"
                continue
            var text: String
            if cast[c]:
                text = texts[c].text_at(r)
            else:
                text = cell_text(texts[c], r)
            if text.byte_length() == 0:
                out += "(empty)"
            else:
                out += text
    return out^


def test_conformance() raises:
    var steps = getenv("FIREPANDA_CONFORMANCE_STEPS")
    if steps == "":
        raise Error("FIREPANDA_CONFORMANCE_STEPS names no steps file")
    var skip = 0
    var passed = getenv("FIREPANDA_CONFORMANCE_SKIP")
    if passed != "":
        skip = Int(passed)
    var data: String
    with open(steps, "r") as handle:
        data = handle.read()

    var dialect = Dialect()
    var catalog = Catalog()
    var files = -1
    var active = False
    var at = 0
    var size = data.byte_length()
    while at < size:
        var end = data.find("\n", at)
        if end < 0:
            end = size
        var head = String(data[byte=at:end])
        at = end + 1
        if head.startswith("F "):
            files += 1
            catalog = Catalog()
            active = files >= skip
            if active:
                print(head, flush=True)
            continue
        var length = Int(String(head[byte = 2 : head.byte_length()]))
        var sql = String(data[byte = at : at + length])
        at += length + 1
        if not active:
            continue
        try:
            var frame = execute(dialect, sql, catalog)
            var text = _kinds(frame) + "\n" + _rendered(dialect, frame)
            print(
                "R ",
                frame.width(),
                " ",
                text.byte_length(),
                "\n",
                text,
                sep="",
                flush=True,
            )
        except e:
            var message = String(e)
            print(
                "E ", message.byte_length(), "\n", message, sep="", flush=True
            )


def main() raises:
    # Through the suite's table of functions rather than called, for the reason
    # written at the bottom of `answers.mojo`: a `main` that reaches
    # `firepanda.sql.run` by a direct call hangs the compiler.
    # See docs/specs/sql/13-open-questions.md question 13.
    TestSuite.discover_tests[__functions_in_module()]().run()
