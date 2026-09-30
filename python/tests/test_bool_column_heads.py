"""Column names that are booleans, as a crosstab or an unstack of a flag makes them.

pandas prints booleans flush left to one width, so beside `False` the name
`True` reads `True ` and its column is a character wider. Each test here runs
the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def flagged(lib: Any) -> Any:
    return lib.DataFrame({"k": ["x", "y"], "f": [True, False]})


BUILDS = {
    "crosstab": lambda lib: lib.crosstab(flagged(lib)["k"], flagged(lib)["f"]),
    "normalized": lambda lib: lib.crosstab(flagged(lib)["k"], flagged(lib)["f"], normalize="index"),
    "unstacked": lambda lib: flagged(lib).groupby(["k", "f"]).size().unstack(),
    "given": lambda lib: lib.DataFrame({True: [1], False: [2]}),
    "true alone": lambda lib: lib.DataFrame({True: [1]}),
    "false alone": lambda lib: lib.DataFrame({False: ["a"]}),
    "transposed": lambda lib: lib.Series([1.0, 2.0], index=[True, False]).to_frame().T,
    "no index": lambda lib: lib.DataFrame({True: [1.5], False: [2]}).to_string(index=False),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_boolean_column_names_print_as_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
