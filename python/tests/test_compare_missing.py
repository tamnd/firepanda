"""A comparison on a row with nothing in it, checked against a running pandas.

The kernel answers null there, and pandas answers False for every comparison and
True for `!=`, because its answer is a numpy bool array with no third value. The
pandas facing operators give pandas' answer, so the tests are every operator in
both spellings, against a constant, against another column and across a frame.
"""

from __future__ import annotations

import importlib.util
import operator
from types import ModuleType

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

OPS = ["eq", "ne", "lt", "le", "gt", "ge"]
SYMBOLS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}
COLUMNS = {
    "number": [1.0, None, 3.0, float("nan")],
    "text": ["a", None, "c", "b"],
}


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize(("column", "value"), [("number", 2.0), ("text", "b")])
def test_against_a_constant(firepanda: ModuleType, op: str, column: str, value: object) -> None:
    """The named form and the operator, which are two routes to the same kernel."""
    import pandas as pd

    mine = firepanda.Series(COLUMNS[column])
    theirs = pd.Series(COLUMNS[column])
    assert getattr(mine, op)(value).tolist() == getattr(theirs, op)(value).tolist()
    assert SYMBOLS[op](mine, value).tolist() == SYMBOLS[op](theirs, value).tolist()


@pytest.mark.parametrize("op", OPS)
def test_against_another_column(firepanda: ModuleType, op: str) -> None:
    """A gap on either side is a gap in the answer."""
    import pandas as pd

    left, right = [1.0, None, 3.0, 4.0], [1.0, 2.0, None, 5.0]
    mine = SYMBOLS[op](firepanda.Series(left), firepanda.Series(right))
    theirs = SYMBOLS[op](pd.Series(left), pd.Series(right))
    assert mine.tolist() == theirs.tolist()


@pytest.mark.parametrize("op", OPS)
def test_across_a_frame(firepanda: ModuleType, op: str) -> None:
    """Every column filled, and only the rows that had a gap."""
    import pandas as pd

    data = {"a": [1.0, None, 3.0], "b": [None, 2.0, 5.0]}
    mine = SYMBOLS[op](firepanda.DataFrame(data), 2.0)
    theirs = SYMBOLS[op](pd.DataFrame(data), 2.0)
    for name in data:
        assert mine[name].tolist() == theirs[name].tolist(), name
