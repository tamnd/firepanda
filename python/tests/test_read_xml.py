"""`read_xml` with the etree parser against pandas, compared by repr."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")

DOC = """<?xml version='1.0' encoding='utf-8'?>
<data>
  <row shape="square">
    <degrees>360</degrees><sides>4.0</sides><ok>True</ok><when>2026-01-02</when>
  </row>
  <row shape="circle">
    <degrees>360</degrees><sides/><ok>False</ok><when>2026-01-03</when>
  </row>
  <row shape="triangle">
    <degrees>180</degrees><sides>3.0</sides><ok>True</ok><when>2026-01-04</when>
  </row>
</data>"""

SPACED = """<doc xmlns="http://x"><r a="1"><b>x</b></r><r a="2"><b>y, "z"</b></r></doc>"""

TEXT = """<data><row>alpha</row><row>beta</row></data>"""


def read(lib: ModuleType, source: str = DOC, **options: Any) -> Any:
    options.setdefault("parser", "etree")
    return lib.read_xml(io.StringIO(source), **options)


def typed(frame: Any) -> Any:
    """The frame with its column types, with text spelled `str` as pandas 3 spells it."""
    kinds = [str(kind).replace("string", "str") for kind in frame.dtypes]
    return frame, kinds


def cells(frame: Any) -> Any:
    """Values and column types, for masked and Arrow columns whose printed widths differ."""
    kinds = [str(kind).replace("string", "str") for kind in frame.dtypes]
    values = {
        name: [None if value is None or value != value else value for value in frame[name].tolist()]
        for name in frame.columns
    }
    return values, kinds


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "plain": lambda lib: typed(read(lib)),
    "lxml-missing": lambda lib: lib.read_xml(io.StringIO(DOC)),
    "elems-only": lambda lib: typed(read(lib, elems_only=True)),
    "attrs-only": lambda lib: typed(read(lib, attrs_only=True)),
    "elems-and-attrs": lambda lib: read(lib, elems_only=True, attrs_only=True),
    "names": lambda lib: typed(read(lib, names=["s", "d", "n", "o", "w"])),
    "names-short": lambda lib: read(lib, names=["s"]),
    "names-text": lambda lib: read(lib, names="s"),
    "dtype": lambda lib: typed(read(lib, dtype={"degrees": "float64"})),
    "converters": lambda lib: typed(read(lib, converters={"degrees": str})),
    "parse-dates": lambda lib: typed(read(lib, parse_dates=["when"])),
    "xpath": lambda lib: typed(read(lib, xpath=".//row[@shape='circle']")),
    "xpath-nothing": lambda lib: read(lib, xpath=".//nothing"),
    "xpath-unsupported": lambda lib: read(lib, xpath="//row[position()=1]"),
    "namespaces": lambda lib: typed(read(lib, SPACED, xpath="x:r", namespaces={"x": "http://x"})),
    "namespace-undeclared": lambda lib: read(lib, SPACED, xpath="y:r"),
    "own-text": lambda lib: typed(read(lib, TEXT)),
    "parser-unknown": lambda lib: read(lib, parser="x"),
    "stylesheet": lambda lib: read(lib, stylesheet="a.xsl"),
    "bytes": lambda lib: typed(lib.read_xml(io.BytesIO(DOC.encode()), parser="etree")),
    "iterparse-buffer": lambda lib: read(lib, iterparse={"row": ["shape"]}),
    "backend-masked": lambda lib: cells(read(lib, dtype_backend="numpy_nullable")),
    "backend-arrow": lambda lib: cells(read(lib, dtype_backend="pyarrow")),
    "backend-unknown": lambda lib: read(lib, dtype_backend="x"),
    "missing-file": lambda lib: lib.read_xml("/no/such/file.xml", parser="etree"),
    "round-trip": lambda lib: typed(
        read(lib, lib.DataFrame({"a": [1, 2], "b": ["x", None]}).to_xml(parser="etree"))
    ),
}


def mistake(error: Exception) -> str:
    """A mistake as its builtin class and message, since each library raises its own subclass."""
    kind = next(kind.__name__ for kind in type(error).__mro__ if kind.__module__ == "builtins")
    return f"{kind}: {error}"


def outcome(build: Callable[[], Any]) -> str:
    try:
        return repr(build())
    except Exception as error:
        return mistake(error)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


@pytest.mark.parametrize(
    "iterparse",
    [{"row": ["shape", "degrees", "sides"]}, ["row"], {"nothing": ["a"]}],
    ids=["rows", "not-a-dict", "no-rows"],
)
def test_iterparse_reads_a_file_as_pandas(tmp_path: Path, iterparse: Any) -> None:
    path = tmp_path / "doc.xml"
    path.write_text(DOC)
    assert outcome(lambda: typed(fp.read_xml(path, parser="etree", iterparse=iterparse))) == (
        outcome(lambda: typed(pd.read_xml(path, parser="etree", iterparse=iterparse)))
    )
