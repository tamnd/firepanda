"""Arithmetic between text and numbers, refused with the `TypeError` pandas raises."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pandas as pd
import pytest

CASES = [
    lambda m: m.Series([1, 2]) + m.Series(["x", "y"]),
    lambda m: m.Series([1, 2]) - m.Series(["x", "y"]),
    lambda m: m.DataFrame({"a": [1], "c": ["x"]}) + 1,
    lambda m: m.DataFrame({"a": [1], "c": ["x"]}).add(1),
    lambda m: 2 / m.DataFrame({"c": ["x"]}),
]


def _message(module: Any, case: Any) -> str:
    with pytest.raises(TypeError) as caught:
        case(module)
    return str(caught.value)


@pytest.mark.parametrize("case", CASES, ids=["radd", "rsub", "frame-add", "frame-named", "rdiv"])
def test_a_number_and_text_raise_pandas_words(firepanda: ModuleType, case: Any) -> None:
    """The message names the reflected operator when the number is on the left, as pandas does."""
    assert _message(firepanda, case) == _message(pd, case)
