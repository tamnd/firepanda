"""`astype("category")` of instants and spans is refused as not written yet.

pandas keeps the categories in the values' own type. firepanda writes numbers
and flags into cells to hold them, which is document 97, and a category of
instants or spans is still refused as a feature that is not there, rather than
failing inside the core.
"""

from __future__ import annotations

from types import ModuleType

import pytest


def test_a_category_of_instants_is_not_supported_yet(firepanda: ModuleType) -> None:
    """A series and a frame both refuse, naming the column's type."""
    column = firepanda.to_datetime(firepanda.Series(["2024-01-01", "2024-01-02"]))
    with pytest.raises(NotImplementedError, match="datetime64"):
        column.astype("category")
    with pytest.raises(NotImplementedError):
        firepanda.DataFrame({"a": column}).astype({"a": "category"})


def test_a_category_of_text_still_works(firepanda: ModuleType) -> None:
    """Text and a category already are the columns the core encodes."""
    column = firepanda.Series(["b", "a", None]).astype("category")
    assert column.cat.categories.tolist() == ["a", "b"]
    assert str(column.astype("category").dtype) == "category"
