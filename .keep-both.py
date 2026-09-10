"""Resolves a conflict where both sides appended, by keeping both appendings.

Only for files where the two sides added different things next to each other and
neither edited what the other wrote, which is what a changelog and a sorted import
list do to every branch that touches them. It is not a general resolver and it says
nothing about files where the sides disagree about the same line.
"""

import sys

for path in sys.argv[1:]:
    kept: list[str] = []
    ours: list[str] = []
    theirs: list[str] = []
    where = "outside"
    for line in open(path).read().splitlines(keepends=True):
        if line.startswith("<<<<<<<"):
            where, ours, theirs = "ours", [], []
        elif line.startswith("=======") and where == "ours":
            where = "theirs"
        elif line.startswith(">>>>>>>") and where == "theirs":
            kept.extend(ours)
            kept.extend(theirs)
            where = "outside"
        elif where == "ours":
            ours.append(line)
        elif where == "theirs":
            theirs.append(line)
        else:
            kept.append(line)
    if where != "outside":
        raise SystemExit(f"{path}: a conflict marker was left open")
    open(path, "w").write("".join(kept))
    print(f"kept both sides in {path}")
