"""`DataFrame.to_xml`, compared with pandas.

Each case builds the same frame in both libraries and compares the document,
with the standard library's parser, which is always there, and with lxml when
it is installed. A written file is compared by its bytes once it is read back
out of its compression, and a mistake by its class and its message.
"""

from __future__ import annotations

import gzip
import io
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def frame(lib: ModuleType) -> Any:
    return lib.DataFrame(
        {
            "i": [1, 2],
            "f": [1.5, None],
            "s": ["x<&", ""],
            "b": [True, False],
            "t": lib.to_datetime(["2024-01-01", None]),
            "d": lib.to_timedelta(["1h", "2D"]),
        }
    )


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        return call(lib)
    except Exception as error:
        return type(error), str(error)


OPTIONS: list[dict[str, Any]] = [
    {},
    {"pretty_print": False, "xml_declaration": False, "index": False},
    {"na_rep": "NA", "attr_cols": ["i", "s"], "elem_cols": ["f"]},
    {"attr_cols": ["i", "t"]},
    {"elem_cols": ["d", "b"], "index": False},
    {"namespaces": {"doc": "https://example.com"}, "prefix": "doc"},
    {"namespaces": {"": "https://example.com", "o": "https://o"}, "root_name": "r"},
    {"row_name": "w", "encoding": "latin-1", "pretty_print": False},
    {"parser": "x"},
    {"attr_cols": "i"},
    {"elem_cols": ["zz"]},
    {"attr_cols": ["zz"]},
    {"namespaces": {"q": "u"}, "prefix": "p"},
    {"encoding": "zz"},
]


@pytest.mark.parametrize("options", OPTIONS)
def test_the_document_is_pandas_document(options: dict[str, Any]) -> None:
    def call(lib: ModuleType) -> Any:
        return frame(lib).to_xml(**{"parser": "etree", **options})

    assert outcome(call, fp) == outcome(call, pd)


def test_a_stylesheet_needs_lxml() -> None:
    def call(lib: ModuleType) -> Any:
        return frame(lib).to_xml(parser="etree", stylesheet="style.xsl")

    assert outcome(call, fp) == outcome(call, pd)


def test_a_named_index_is_written_first() -> None:
    def call(lib: ModuleType) -> Any:
        labels = lib.Index(["k", "m"], name="key")
        return lib.DataFrame({"a": [1, 2]}, index=labels).to_xml(parser="etree", attr_cols=["a"])

    assert outcome(call, fp) == outcome(call, pd)


def test_a_handle_gets_the_encoded_bytes() -> None:
    ours, theirs = io.BytesIO(), io.BytesIO()
    assert frame(fp).to_xml(ours, parser="etree") is None
    frame(pd).to_xml(theirs, parser="etree")
    assert ours.getvalue() == theirs.getvalue()


def test_a_file_is_compressed_as_its_name_says(tmp_path: Path) -> None:
    ours, theirs = tmp_path / "ours.xml.gz", tmp_path / "theirs.xml.gz"
    frame(fp).to_xml(ours, parser="etree")
    frame(pd).to_xml(theirs, parser="etree")
    assert gzip.decompress(ours.read_bytes()) == gzip.decompress(theirs.read_bytes())


STYLE = (
    '<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">'
    '<xsl:output method="xml" indent="yes"/><xsl:template match="/data"><out>'
    '<xsl:for-each select="row"><r n="{i}"/></xsl:for-each></out></xsl:template>'
    "</xsl:stylesheet>"
)

LXML: list[dict[str, Any]] = [
    {},
    {"pretty_print": False, "na_rep": "-", "attr_cols": ["i"], "elem_cols": ["f"]},
    {"namespaces": {"": "https://example.com", "o": "https://o"}},
    {"namespaces": {"doc": "https://example.com"}, "prefix": "doc", "index": False},
]


@pytest.mark.parametrize("options", LXML)
def test_the_lxml_document_is_pandas_document(options: dict[str, Any]) -> None:
    pytest.importorskip("lxml")
    assert frame(fp).to_xml(**options) == frame(pd).to_xml(**options)


def test_lxml_runs_a_stylesheet() -> None:
    pytest.importorskip("lxml")
    ours = frame(fp).to_xml(stylesheet=io.StringIO(STYLE))
    assert ours == frame(pd).to_xml(stylesheet=io.StringIO(STYLE))
    assert frame(fp).to_xml(stylesheet=io.BytesIO(STYLE.encode())) == ours


def test_the_default_parser_needs_lxml() -> None:
    try:
        import lxml  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("lxml is installed")
    assert outcome(lambda lib: frame(lib).to_xml(), fp) == outcome(
        lambda lib: frame(lib).to_xml(), pd
    )
