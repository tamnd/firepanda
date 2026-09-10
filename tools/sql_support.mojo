"""Writes the SQL support table into the README.

The README's list of what firepanda's SQL front end refuses is generated from
`sql_support()` rather than kept beside it, because a hand written list is right
on the day it is written and wrong a week later. This program rewrites
everything between the two markers in `README.md`. `tests/test_sql_support.mojo`
fails when the file and the table disagree, so forgetting to run this is caught
by the test suite rather than by a reader.

Run it with `pixi run sql-support`, from the repository root.
"""

from std.sys import argv

from firepanda.sql.unsupported import support_table

comptime OPEN: StaticString = "<!-- sql-support -->"
"""The line the generated block starts after."""

comptime CLOSE: StaticString = "<!-- end sql-support -->"
"""The line the generated block stops before."""


def spliced(readme: String, table: String) raises -> String:
    """Puts a fresh table between the markers.

    Args:
        readme: The whole file.
        table: What `support_table` returned.

    Returns:
        The whole file, with the block between the markers replaced.

    Raises:
        Error: If either marker is missing, or they are the wrong way round.
    """
    var opens = readme.find(OPEN)
    if opens < 0:
        raise Error("README.md has no ", OPEN, " marker")
    var closes = readme.find(CLOSE)
    if closes < 0:
        raise Error("README.md has no ", CLOSE, " marker")
    if closes < opens:
        raise Error("README.md has ", CLOSE, " before ", OPEN)
    var head = readme[byte = 0 : opens + OPEN.byte_length()]
    var tail = readme[byte = closes : readme.byte_length()]
    return String(head, "\n\n", table, "\n", tail)


def main() raises:
    var path = "README.md"
    var handle = open(path, "r")
    var readme = handle.read()
    handle.close()

    var wanted = spliced(readme, support_table())
    if readme == wanted:
        print("the SQL support table in README.md is current")
        return

    var check = False
    var args = argv()
    for i in range(1, len(args)):
        if args[i] == "--check":
            check = True

    if check:
        raise Error(
            "the SQL support table in README.md is stale, run 'pixi run"
            " sql-support' and commit the result"
        )

    var out = open(path, "w")
    out.write(wanted)
    out.close()
    print("wrote the SQL support table into README.md")
