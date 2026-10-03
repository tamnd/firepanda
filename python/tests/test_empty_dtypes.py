"""The type a column of no values answers, as pandas answers it.

An extension column writes its type into each value it holds, so one with no
values, or only gaps, kept no type and read as text. The type asked for is kept
beside the column instead, and no values at all read as objects, as in pandas.
"""

from __future__ import annotations

from types import ModuleType

import pytest

KINDS = ["Int64", "Float64", "boolean", "UInt8", "string", "int64[pyarrow]", "object"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("data", [[], [None], None])
def test_asked_type_is_kept(firepanda: ModuleType, kind: str, data: object) -> None:
    """The constructor's `dtype` and `astype` both answer the type asked for."""
    assert str(firepanda.Series(data, dtype=kind).dtype) == kind
    assert str(firepanda.Series(data).astype(kind).dtype) == kind


def test_period_and_sparse(firepanda: ModuleType) -> None:
    """A period type and a sparse type with no values keep their names."""
    assert str(firepanda.Series([None], dtype="period[M]").dtype) == "period[M]"
    assert str(firepanda.Series([], dtype="Sparse[int64]").dtype) == "Sparse[int64, 0]"


@pytest.mark.parametrize("data", [[], [None], None])
def test_no_values_are_objects(firepanda: ModuleType, data: object) -> None:
    """No values, or only None, are objects, and floats stay floats."""
    assert str(firepanda.Series(data).dtype) == "object"
    assert str(firepanda.Series([float("nan")]).dtype) == "float64"
    assert str(firepanda.Series(data, dtype="float32").dtype) == "float32"


def test_labels_alone_are_floats(firepanda: ModuleType) -> None:
    """Labels with no data are NaN in each row, a float column."""
    column = firepanda.Series(index=[0, 1])
    assert str(column.dtype) == "float64" and len(column) == 2


@pytest.mark.parametrize("data", [[], [None, None]])
def test_a_window_reads_objects_as_floats(firepanda: ModuleType, data: list) -> None:
    """A window over objects with no values reads them as floats, as pandas does."""
    column = firepanda.Series(data)
    assert len(column.ewm(alpha=0.3).mean().dropna()) == 0
    assert len(column.rolling(2).sum().dropna()) == 0


def test_an_empty_masked_column_reduces(firepanda: ModuleType) -> None:
    """A sum over a masked column with no values is nothing added up."""
    column = firepanda.Series([], dtype="Int64")
    assert column.sum() == 0 and column.count() == 0
    assert firepanda.Series([None], dtype="Int64").isna().tolist() == [True]


def test_writing_into_a_masked_column(firepanda: ModuleType) -> None:
    """A value, a gap or a refused value written into a masked column, as pandas writes."""
    column = firepanda.Series([None], dtype="Int64")
    column[0] = 3
    assert str(column.dtype) == "Int64" and column.tolist() == [3]
    column = firepanda.Series([1, None], dtype="Int64")
    column[0] = None
    assert str(column.dtype) == "Int64" and column.isna().tolist() == [True, True]
    flags = firepanda.Series([True, False], dtype="boolean")
    flags[0] = None
    assert str(flags.dtype) == "boolean" and flags.isna().tolist() == [True, False]
    whole = firepanda.Series([1, 2, 3], dtype="Int64")
    whole[0:2] = [7, 8]
    assert whole.tolist() == [7, 8, 3]
    with pytest.raises(TypeError, match=r"Invalid value '1.5' for dtype 'Int64'"):
        whole[0] = 1.5
