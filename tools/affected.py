"""Names the test files a set of changed files can possibly have broken.

The suite costs 12086 seconds and a test file spends nearly all of that
compiling the library again, so the only two levers on it are more machines and
fewer files. `tools/run_tests_remote.sh` is the first one. This is the second.

A test file cannot break unless something it imports changed. Mojo imports are
static and every one of them is a line at the top of a file, so the set of
library modules a test pulls in is readable straight off the source, and a test
whose set does not contain any changed file does not need running.

How much that is worth was measured over all 199 library files, each one asked
how much of the suite a change to it selects. The median file selects 16.5 per
cent, the SQL front end files select 11.7, and a quarter of the files select 2.7
or less. The other quarter select more than 93, and those are the ones
everything needs: `dtype/logical.mojo`, `array/array.mojo`, `frame/frame.mojo`.
So this is worth six times on a typical change, eight on the SQL work that most
of the current milestone is, and nothing at all on a change to the bottom of the
library. That last case is the point rather than a flaw. It is the same answer
the suite would give, arrived at without running it.

What it does not do is decide. It prints a list and `tools/run_tests.sh
--changed` runs it, and a branch is still merged on a full run. This is for the
loop somebody is in while they are writing code, which is the same thing
`--fast` is for and is why it says so in its own output.

Anything it does not understand means the whole suite. A changed file that is
not under `firepanda/` or `tests/` could be the pixi manifest or the runner
itself or a generator whose output is checked in, and none of those are visible
in an import line. An import it cannot resolve to a file on disk means the same.
Being wrong here costs a test that should have run and did not, which is the one
failure mode worth spending the whole suite to avoid.

Anything that is not a `.mojo` file means the whole suite too, and that one is
load bearing rather than cautious. Tests open files at run time that no import
line mentions: `tests/test_sql_unsupported.mojo` reads `README.md` and fails if
the table in it is stale, and the differential tests read the corpus. An import
graph cannot see any of that, so a change to anything it cannot reason about
gets the answer it can be sure of.

Exit codes are the interface. Nothing on standard output and a zero exit means
no test is affected. A list and a zero exit means run those. A one exit means it
could not tell, so run all of them.
"""

import os
import re
import sys

# Both forms appear in this tree. Tests spell the whole path, and the library
# spells its own modules relative, which is why the second one exists and why
# leaving it out made the first measurement of all this say that a change to
# `sql/transform.mojo` broke nothing at all.
ABSOLUTE = re.compile(r"^\s*(?:from|import)\s+(firepanda[A-Za-z0-9_.]*)", re.M)
RELATIVE = re.compile(r"^\s*(?:from|import)\s+(\.+)([A-Za-z0-9_.]*)", re.M)


def module_file(parts):
    """Finds the file a dotted module name refers to, or nothing.

    Args:
        parts: The name, already split on dots.

    Returns:
        The path, or None if neither spelling of it is on disk.
    """
    joined = "/".join(parts)
    for candidate in (joined + ".mojo", joined + "/__init__.mojo"):
        if os.path.exists(candidate):
            return candidate
    return None


def imports_of(path, text):
    """Reads the library modules one file imports.

    A relative import counts its leading dots: one means the directory the file
    is in, and each one after that is a step up from there.

    Args:
        path: Where the file is, which is what the dots are relative to.
        text: Its source.

    Returns:
        A set of paths, and a flag saying whether some import could not be
        resolved to a file.
    """
    found = set()
    unresolved = False
    for name in ABSOLUTE.findall(text):
        resolved = module_file(name.split("."))
        if resolved is None:
            unresolved = True
        else:
            found.add(resolved)
    here = os.path.dirname(path).split(os.sep)
    for dots, rest in RELATIVE.findall(text):
        base = here[: len(here) - len(dots) + 1]
        resolved = module_file(base + [part for part in rest.split(".") if part])
        if resolved is None:
            unresolved = True
        else:
            found.add(resolved)
    return found, unresolved


def read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def library():
    """Reads what every library file imports.

    Returns:
        A map from path to the set of paths it imports, and a flag saying
        whether anything could not be resolved.
    """
    edges = {}
    unresolved = False
    for directory, _, names in os.walk("firepanda"):
        for name in names:
            if not name.endswith(".mojo"):
                continue
            path = os.path.join(directory, name)
            edges[path], missed = imports_of(path, read(path))
            unresolved = unresolved or missed
    return edges, unresolved


def reaches(edges, start):
    """Walks the import graph from a starting set.

    Args:
        edges: What each file imports.
        start: The files to start from.

    Returns:
        Every file reachable, including the ones started from.
    """
    seen = set()
    stack = list(start)
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        stack.extend(edges.get(path, ()))
    return seen


def affected(changed):
    """Names the tests a set of changed files can reach.

    Args:
        changed: The changed paths, relative to the repository root.

    Returns:
        A sorted list of test files, or None meaning run all of them.
    """
    interesting = [path for path in changed if path.endswith(".mojo")]
    if len(interesting) != len(changed):
        return None
    for path in interesting:
        if not path.startswith("firepanda/") and not path.startswith("tests/"):
            return None

    edges, unresolved = library()
    if unresolved:
        return None

    wanted = set(interesting)
    picked = []
    for name in sorted(os.listdir("tests")):
        if not name.startswith("test_") or not name.endswith(".mojo"):
            continue
        path = "tests/" + name
        # A changed test file runs whatever else it does or does not import.
        if path in wanted:
            picked.append(path)
            continue
        imports, missed = imports_of(path, read(path))
        if missed:
            return None
        if reaches(edges, imports) & wanted:
            picked.append(path)
    return picked


def main():
    changed = sys.argv[1:]
    if not changed:
        return 0
    picked = affected(changed)
    if picked is None:
        return 1
    for path in picked:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
