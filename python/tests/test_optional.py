"""The sentence for an optional dependency that is not installed, compared with pandas."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pytest

import firepanda as fp
from firepanda import _optional

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


@needs_pandas
@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("xarray", ""),
        ("odf", ""),
        ("python_calamine", ""),
        ("lxml.etree", ""),
        ("xlrd", "Install xlrd >= 2.0.1 for xls Excel support"),
    ],
)
def test_the_sentence_is_pandas_sentence(name: str, extra: str, monkeypatch: Any) -> None:
    from pandas.compat._optional import import_optional_dependency

    monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError) as theirs:
        import_optional_dependency(name, extra=extra)
    with pytest.raises(ImportError) as ours:
        _optional.imported(name, extra)
    assert str(ours.value) == str(theirs.value)


def test_a_missing_pytest_is_named(monkeypatch: Any) -> None:
    monkeypatch.setitem(sys.modules, "pytest", None)
    with pytest.raises(ImportError, match="`Import pytest` failed"):
        fp.test()
