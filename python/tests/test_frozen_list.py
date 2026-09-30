"""`names`, `levels` and `codes` as pandas' `FrozenList`.

pandas answers these with a list that refuses to be written into and prints
under its own name, an index or an array inside it printed as the list of its
values. Each test here runs the same code on both libraries and compares what
they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def multi(lib: Any) -> Any:
    return lib.MultiIndex.from_tuples([("b", 2), ("a", 1)], names=["p", "q"])


def refusal(make: Any) -> str:
    try:
        make()
    except TypeError as error:
        return str(error)
    return "allowed"


BUILDS = {
    "levels": lambda lib: repr(multi(lib).levels),
    "levels printed": lambda lib: str(multi(lib).levels),
    "names": lambda lib: repr(multi(lib).names),
    "names printed": lambda lib: str(multi(lib).names),
    "codes": lambda lib: repr(multi(lib).codes),
    "class": lambda lib: (type(multi(lib).names).__name__, type(multi(lib).levels).__name__),
    "frame labels": lambda lib: repr(lib.DataFrame({"v": [1, 2]}, index=multi(lib)).index.names),
    "flat names": lambda lib: repr(lib.Index([1], name="k").names),
    "equal to a list": lambda lib: (multi(lib).names == ["p", "q"], multi(lib).names != ["p"]),
    "added": lambda lib: repr(multi(lib).names + ["r"]),  # noqa: RUF005
    "sliced": lambda lib: repr(multi(lib).names[1:]),
    "one item": lambda lib: multi(lib).names[0],
    "hashed": lambda lib: hash(multi(lib).names) == hash(("p", "q")),
    "difference": lambda lib: repr(multi(lib).names.difference(["p"])),
    "union": lambda lib: repr(multi(lib).names.union(["z"])),
    "set item": lambda lib: refusal(lambda: multi(lib).names.__setitem__(0, "z")),
    "append": lambda lib: refusal(lambda: multi(lib).names.append("z")),
    "pop": lambda lib: refusal(lambda: multi(lib).names.pop()),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_frozen_lists_are_pandas(firepanda: Any, make: Any) -> None:
    assert make(firepanda) == make(pd)
