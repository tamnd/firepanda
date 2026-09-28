"""`astype("category")` of a column that is not text is refused as not written yet.

pandas keeps the categories in the values' own type. firepanda holds them as a
text column, so a category of numbers, flags or instants is refused as a
feature that is not there, rather than failing inside the core.
"""

from __future__ import annotations

from types import ModuleType

import pytest


@pytest.mark.parametrize(
    "values", [[1, 2], [True, False], [1.5, None]], ids=["int", "bool", "float"]
)
def test_a_category_of_numbers_is_not_supported_yet(firepanda: ModuleType, values: list) -> None:
    """A series and a frame both refuse, naming the column's type."""
    column = firepanda.Series(values)
    with pytest.raises(NotImplementedError, match=str(column.dtype)):
        column.astype("category")
    with pytest.raises(NotImplementedError):
        firepanda.DataFrame({"a": values}).astype({"a": "category"})


def test_a_category_of_text_still_works(firepanda: ModuleType) -> None:
    """Text and a category already are the columns the core encodes."""
    column = firepanda.Series(["b", "a", None]).astype("category")
    assert column.cat.categories.tolist() == ["a", "b"]
    assert str(column.astype("category").dtype) == "category"
