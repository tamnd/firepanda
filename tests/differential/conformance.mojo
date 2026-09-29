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
    R <columns> <bytes> the statement answered, then its rows
    E <bytes>           the statement raised, then the message

A row is its values joined by tabs, a null as `NULL` and empty text as
`(empty)`, which is how DuckDB's own runner writes them. The values are cast to
text by SQL's own `CAST(... AS VARCHAR)`, so a number prints the way the dialect
prints it rather than the way this program would.

Every record is flushed as it is written. A statement that takes the process
down leaves the driver the last file it started, which is how a crash is told
apart from a failure, and the second argument says how many files to pass over
so the driver can start again after it.

See docs/specs/sql/11-conformance.md.
"""

from std.sys import argv

from firepanda.array.any import AnyArray
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame
from firepanda.kernel.cast import cast_any
from firepanda.sql.catalog import Catalog
from firepanda.sql.ddl import execute
from firepanda.sql.run import Dialect


def _column(frame: DataFrame, i: Int) raises -> AnyArray:
    """One column as a single array."""
    return frame.columns[i].copy().combine()


def _texts(dialect: Dialect, frame: DataFrame) raises -> List[AnyArray]:
    """Every column cast to text, by SQL's cast where it has one.

    The frame goes into a catalog of its own under names that cannot clash, so
    a result with two columns of one name, or a name no query could write, is
    still read one column at a time.
    """
    var fields = List[Field]()
    var columns = List[AnyArray]()
    var select = String("SELECT ")
    for i in range(frame.width()):
        var name = String("c", i)
        fields.append(Field(name, frame.schema[i].dtype))
        columns.append(_column(frame, i))
        if i > 0:
            select += ", "
        select += String("CAST(", name, " AS VARCHAR) AS ", name)
    select += " FROM r"
    var scratch = Catalog()
    scratch.register("r", DataFrame(Schema(fields^), columns^))
    var out = List[AnyArray]()
    try:
        var cast = dialect.run(select, scratch)
        for i in range(cast.width()):
            out.append(_column(cast, i))
        return out^
    except:
        pass
    # A type SQL cannot cast to text yet still has a rendering, and a row that
    # prints is more use to the report than a statement that could not.
    for i in range(frame.width()):
        out.append(
            cast_any(_column(frame, i), LogicalType.STRING, strict=False)
        )
    return out^


def _rendered(dialect: Dialect, frame: DataFrame) raises -> String:
    """The rows, a tab between values and a newline between rows."""
    var texts = _texts(dialect, frame)
    var rows = len(frame)
    var out = String()
    for r in range(rows):
        if r > 0:
            out += "\n"
        for c in range(len(texts)):
            if c > 0:
                out += "\t"
            if not texts[c].is_valid(r):
                out += "NULL"
                continue
            var text = texts[c].text_at(r)
            if text.byte_length() == 0:
                out += "(empty)"
            else:
                out += text
    return out^


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: conformance <steps file> [files to pass over]")
    var skip = 0
    if len(args) > 2:
        skip = Int(String(args[2]))
    var data: String
    with open(String(args[1]), "r") as handle:
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
            var text = _rendered(dialect, frame)
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
