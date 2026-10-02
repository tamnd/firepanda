"""Floor division and the remainder of masked whole numbers by zero, as pandas answers.

pandas answers 0 where an `Int64` or `UInt8` column is floored or taken the
remainder of by zero, keeping the type and the gaps, where the lower case kernel
answers a gap.
"""

from types import ModuleType
from typing import Any

import pandas as pd
import pytest


def _ints(lib: Any) -> Any:
    return lib.Series([1, None, -3, 0], dtype="Int64")


CASES = {
    "floordiv-scalar": lambda lib: _ints(lib) // 0,
    "mod-scalar": lambda lib: _ints(lib) % 0,
    "floordiv-column": lambda lib: _ints(lib) // lib.Series([0, 1, 2, 0], dtype="Int64"),
    "mod-column": lambda lib: _ints(lib) % lib.Series([0, 1, 2, 0], dtype="Int64"),
    "unsigned": lambda lib: lib.Series([1, 2], dtype="UInt8") // 0,
    "fill-value": lambda lib: _ints(lib).floordiv(0, fill_value=1),
    "plain-divisor": lambda lib: _ints(lib) // lib.Series([2, 2, 0, 1]),
    "no-zero": lambda lib: _ints(lib) // 2,
}


@pytest.mark.parametrize("name", list(CASES))
def test_masked_whole_numbers_over_zero_match_pandas(firepanda: ModuleType, name: str) -> None:
    mine, theirs = CASES[name](firepanda), CASES[name](pd)
    assert str(mine.dtype) == str(theirs.dtype)
    assert repr(mine) == repr(theirs)
