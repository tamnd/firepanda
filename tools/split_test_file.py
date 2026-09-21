"""Splits one test file into several, with the fixtures moved to a shared module.

A test file is a program, so every one of them compiles the slice of the library
its imports reach. For the three biggest files that slice is the whole SQL and
execution stack and the compile is most of the cost, which is why the shards in
CI are floored by whichever shard holds tests/test_sql_run.mojo. Cutting the file
up is the only thing that moves that floor.

The fixtures go to tests/support/<stem>.mojo rather than being copied into each
part, so the frames and helpers the tests share still live in one place.

    python3 tools/split_test_file.py tests/test_sql_run.mojo 3
"""

import os
import re
import sys

TESTDEF = re.compile(r"(def|fn) test_")
ANYDEF = re.compile(r"(def|fn) ")
DEFNAME = re.compile(r"(?:def|fn) ([A-Za-z_][A-Za-z_0-9]*)")


def parse(path):
    src = open(path).read().split("\n")
    first = next(i for i, l in enumerate(src) if TESTDEF.match(l))
    header = src[:first]
    blocks, cur = [], []
    for l in src[first:]:
        if ANYDEF.match(l) and cur:
            blocks.append(cur)
            cur = [l]
        else:
            cur.append(l)
    if cur:
        blocks.append(cur)
    tests, helpers = [], []
    for b in blocks:
        if TESTDEF.match(b[0]):
            tests.append(b)
        elif b[0].startswith("def main"):
            pass
        else:
            helpers.append(b)
    return header, helpers, tests


def trim(lines):
    while lines and lines[-1].strip() == "":
        lines.pop()
    return lines


def main():
    path, n = sys.argv[1], int(sys.argv[2])
    stem = os.path.basename(path)[:-5]
    header, helpers, tests = parse(path)

    # The header is a docstring, then imports, then the fixtures the tests share.
    # All of it goes to the support module; the parts import what they name.
    doc_end = 0
    if header and header[0].startswith('"""'):
        doc_end = next(i for i, l in enumerate(header) if l.rstrip().endswith('"""') and i > 0) + 1
    doc = header[:doc_end]
    rest = header[doc_end:]

    fixture_start = next(i for i, l in enumerate(rest) if ANYDEF.match(l))
    imports = trim(rest[:fixture_start])
    fixtures = rest[fixture_start:]

    support_dir = "tests/support"
    os.makedirs(support_dir, exist_ok=True)
    support = os.path.join(support_dir, stem[len("test_"):] + ".mojo")

    names = []
    for l in fixtures + [h for b in helpers for h in b]:
        m = DEFNAME.match(l)
        if m:
            names.append(m.group(1))

    body = []
    body.append('"""The frames and helpers the %s tests share.' % stem[len("test_"):].replace("_", " "))
    body.append("")
    body.append("These were the top of one file before it was cut into %d, which was done" % n)
    body.append("because a test file is a program and every one of them compiles the slice")
    body.append("of the library its imports reach. That slice is the whole stack here, so")
    body.append("the file was the longest thing in its CI shard and the shard could not")
    body.append("finish faster than it did.")
    body.append('"""')
    body.append("")
    body.extend(imports)
    body.append("")
    body.extend(trim(fixtures))
    body.append("")
    body.extend(trim([h for b in helpers for h in b]))
    body.append("")
    open(support, "w").write(re.sub(r"\n{3,}", "\n\n\n", "\n".join(body)))

    module = support[:-5].replace("/", ".")
    import_line = "from %s import (\n%s\n)" % (
        module,
        "\n".join("    %s," % s for s in sorted(names)),
    )

    per = (len(tests) + n - 1) // n
    written = []
    for k in range(n):
        part = tests[k * per:(k + 1) * per]
        if not part:
            continue
        out = "tests/%s_%d.mojo" % (stem, k + 1)
        b = []
        b.extend(doc[:-1] if doc else [])
        if doc:
            b.append("")
            b.append("Part %d of %d. The fixtures are in %s." % (k + 1, n, support))
            b.append('"""')
        b.append("")
        b.extend(imports)
        b.append("")
        b.append(import_line)
        b.append("")
        for t in part:
            b.extend(t)
        b.extend(["def main() raises:", "    TestSuite.discover_tests[__functions_in_module()]().run()", ""])
        open(out, "w").write(re.sub(r"\n{3,}", "\n\n\n", "\n".join(b)))
        written.append((out, len(part)))

    os.remove(path)
    print("support:", support, len(names), "names")
    for o, c in written:
        print(" ", o, c, "tests")


main()
