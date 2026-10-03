"""Arithmetic text cannot do, refused with pandas' words wherever it is reached.

The operators already said what pandas says. These are the routes that reach
the same failure from further away, `diff`, `pct_change`, a frame's `cumprod`,
negation and `clip`, which used to let the core's sentence through.
"""

from __future__ import annotations

from types import ModuleType

import pytest

_BOTH_TEXT = "not supported for dtype 'str' with dtype 'str'$"


def _text(firepanda: ModuleType) -> object:
    return firepanda.Series(["a", "b"])


def test_differences_and_changes_of_text(firepanda: ModuleType) -> None:
    """`diff` subtracts and `pct_change` divides, and each is named."""
    with pytest.raises(TypeError, match=rf"^operation 'sub' {_BOTH_TEXT}"):
        _text(firepanda).diff()
    with pytest.raises(TypeError, match=rf"^operation 'truediv' {_BOTH_TEXT}"):
        _text(firepanda).pct_change()
    with pytest.raises(TypeError, match=rf"^operation 'sub' {_BOTH_TEXT}"):
        firepanda.DataFrame({"s": ["a", "b"]}).diff()


def test_running_product_of_text_in_a_frame(firepanda: ModuleType) -> None:
    """The frame says what the column says."""
    frame = firepanda.DataFrame({"s": ["a", "b"], "n": [1, 2]})
    with pytest.raises(TypeError, match=r"^operation 'cumprod' not supported for dtype 'str'$"):
        frame.cumprod()


def test_negated_text(firepanda: ModuleType) -> None:
    """Unary minus names itself the way pandas does."""
    with pytest.raises(TypeError, match=r"^unary '-' not supported for dtype 'str'$"):
        -_text(firepanda)


def test_text_times_what_is_not_a_whole_number(firepanda: ModuleType) -> None:
    """Text repeats by whole numbers only, whether the other side is text or floats."""
    text = _text(firepanda)
    for other in (text, firepanda.Series([1.5, 2.0])):
        with pytest.raises(TypeError, match=r"^Can only string multiply by an integer\.$"):
            text * other
    assert (text * firepanda.Series([2, 1])).tolist() == ["aa", "b"]


def test_clipping_text_by_numbers(firepanda: ModuleType) -> None:
    """A numeric bound cannot be compared with text, and text bounds still clip."""
    text = _text(firepanda)
    with pytest.raises(TypeError, match=r"^Invalid comparison between dtype=str and int$"):
        text.clip(0, 1)
    with pytest.raises(TypeError, match=r"^Invalid comparison between dtype=str and int$"):
        text.clip(lower=0)
    assert text.clip("b", "c").tolist() == ["b", "b"]
