"""`read_html` against what pandas reads with lxml, recorded since pandas needs lxml for it."""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import pytest

import firepanda as fp

S = io.StringIO
S = io.StringIO
BASIC = (
    "<table><thead><tr><th>a</th><th>b</th></tr></thead>"
    "<tbody><tr><td>1</td><td>x</td></tr><tr><td>2</td><td>y</td></tr>"
    "</tbody></table>"
)
NOHEAD = "<table><tr><td>1</td><td>2.5</td></tr><tr><td>3</td><td>4.5</td></tr></table>"
THROW = "<table><tr><th>k</th><th>v</th></tr><tr><td>a</td><td>1</td></tr></table>"
TWO = BASIC + ("<p>mid</p><table id='t2' class='c'><tr><th>z</th></tr><tr><td>9</td></tr></table>")
SPAN = (
    "<table><tr><th colspan='2'>ab</th><th>c</th></tr>"
    "<tr><td rowspan='2'>1</td><td>2</td><td>3</td></tr>"
    "<tr><td>5</td><td>6</td></tr></table>"
)
RAG = (
    "<table><tr><th>a</th><th>b</th><th>c</th></tr><tr><td>1</td></tr>"
    "<tr><td>2</td><td>3</td><td>4</td></tr></table>"
)
FOOT = (
    "<table><thead><tr><th>a</th></tr></thead><tbody><tr><td>1</td></tr>"
    "</tbody><tfoot><tr><td>total</td></tr></tfoot></table>"
)
WS = (
    "<table><tr><th> a  b </th></tr><tr><td>line<br>two</td></tr><tr><td>  x\n y </td></tr></table>"
)
NUM = (
    "<table><tr><th>n</th><th>d</th></tr>"
    "<tr><td>1,234</td><td>2026-01-02</td></tr>"
    "<tr><td>5,678</td><td>2026-02-03</td></tr></table>"
)
NA = (
    "<table><tr><th>a</th><th>b</th></tr><tr><td></td><td>NA</td></tr>"
    "<tr><td>1</td><td>-</td></tr></table>"
)
HID = (
    "<table><tr><th>a</th><th style='display: none'>h</th></tr>"
    "<tr><td>1</td><td style='display:none'>2</td></tr></table>"
    "<table style='display:none'><tr><td>9</td></tr></table>"
)
UNCLOSED = "<table><tr><th>a<th>b<tr><td>1<td>2<tr><td>3<td>4</table>"
BOOL = (
    "<table><tr><th>a</th><th>b</th></tr><tr><td>True</td><td>1.5</td></tr>"
    "<tr><td>False</td><td>nan</td></tr></table>"
)
LINK = "<table><tr><th><a href='h'>A</a></th></tr><tr><td><a href='/x'>X</a></td></tr></table>"
MULTIH = (
    "<table><thead><tr><th>a</th><th>b</th></tr>"
    "<tr><th>c</th><th>d</th></tr></thead><tr><td>1</td><td>2</td></tr>"
    "</table>"
)
NESTED = "<table><tr><th>o</th></tr><tr><td><table><tr><td>in</td></tr></table></td></tr></table>"


CASES: dict[str, tuple[Callable[[], Any], tuple[str, ...]]] = {
    "basic": (lambda: fp.read_html(S(BASIC)), ("[   a  b", "0  1  x", "1  2  y]")),
    "nohead": (lambda: fp.read_html(S(NOHEAD)), ("[   0    1", "0  1  2.5", "1  3  4.5]")),
    "th-row": (lambda: fp.read_html(S(THROW)), ("[   k  v", "0  a  1]")),
    "two": (lambda: fp.read_html(S(TWO)), ("[   a  b", "0  1  x", "1  2  y,    z", "0  9]")),
    "match": (lambda: fp.read_html(S(TWO), match="9"), ("[   z", "0  9]")),
    "attrs": (lambda: fp.read_html(S(TWO), attrs={"id": "t2"}), ("[   z", "0  9]")),
    "attrs-class": (lambda: fp.read_html(S(TWO), attrs={"class": "c"}), ("[   z", "0  9]")),
    "header0": (lambda: fp.read_html(S(NOHEAD), header=0), ("[   1  2.5", "0  3  4.5]")),
    "header-none-th": (lambda: fp.read_html(S(THROW), header=None), ("[   k  v", "0  a  1]")),
    "index_col": (lambda: fp.read_html(S(BASIC), index_col=0), ("[   b", "a   ", "1  x", "2  y]")),
    "skiprows": (lambda: fp.read_html(S(NOHEAD), skiprows=1), ("[   0    1", "0  3  4.5]")),
    "skiprows-list": (lambda: fp.read_html(S(BASIC), skiprows=[1]), ("[   a  b", "0  2  y]")),
    "skiprows-slice": (
        lambda: fp.read_html(S(NOHEAD), skiprows=slice(0, 1)),
        ("[   0    1", "0  3  4.5]"),
    ),
    "skiprows-neg": (
        lambda: fp.read_html(S(NOHEAD), skiprows=-1),
        (
            (
                "ValueError: cannot skip rows starting from the end of the data (you passed a"
                " negative value)"
            ),
        ),
    ),
    "skiprows-bad": (
        lambda: fp.read_html(S(NOHEAD), skiprows=1.5),
        ("TypeError: float is not a valid type for skipping rows",),
    ),
    "span": (
        lambda: fp.read_html(S(SPAN)),
        ("[   ab  ab.1  c", "0   1     2  3", "1   1     5  6]"),
    ),
    "ragged": (
        lambda: fp.read_html(S(RAG)),
        ("[   a    b    c", "0  1  NaN  NaN", "1  2  3.0  4.0]"),
    ),
    "foot": (lambda: fp.read_html(S(FOOT)), ("[       a", "0      1", "1  total]")),
    "ws": (lambda: fp.read_html(S(WS)), ("[        a b", "0  line two", "1      x  y]")),
    "thousands": (
        lambda: fp.read_html(S(NUM)),
        ("[      n           d", "0  1234  2026-01-02", "1  5678  2026-02-03]"),
    ),
    "thousands-none": (
        lambda: fp.read_html(S(NUM), thousands=None),
        ("[       n           d", "0  1,234  2026-01-02", "1  5,678  2026-02-03]"),
    ),
    "na": (lambda: fp.read_html(S(NA)), ("[     a    b", "0  NaN  NaN", "1  1.0    -]")),
    "na_values": (
        lambda: fp.read_html(S(NA), na_values=["-"]),
        ("[     a   b", "0  NaN NaN", "1  1.0 NaN]"),
    ),
    "keep_default_na": (
        lambda: fp.read_html(S(NA), keep_default_na=False),
        ("[   a   b", "0     NA", "1  1   -]"),
    ),
    "hidden": (lambda: fp.read_html(S(HID)), ("[   a", "0  1]")),
    "hidden-shown": (
        lambda: fp.read_html(S(HID), displayed_only=False),
        ("[   a  h", "0  1  2,    0", "0  9]"),
    ),
    "unclosed": (lambda: fp.read_html(S(UNCLOSED)), ("[   a  b", "0  1  2", "1  3  4]")),
    "bool-values": (
        lambda: fp.read_html(S(BOOL)),
        ("[       a    b", "0   True  1.5", "1  False  NaN]"),
    ),
    "decimal": (
        lambda: fp.read_html(
            S("<table><tr><th>a</th></tr><tr><td>1,5</td></tr></table>"), decimal=",", thousands="."
        ),
        ("[     a", "0  1.5]"),
    ),
    "converters": (
        lambda: fp.read_html(S(BASIC), converters={"b": str.upper}),
        ("[   a  b", "0  1  X", "1  2  Y]"),
    ),
    "links-body": (
        lambda: fp.read_html(S(LINK), extract_links="body"),
        ("[         A", "0  (X, /x)]"),
    ),
    "links-bad": (
        lambda: fp.read_html(S(LINK), extract_links="x"),
        (
            (
                'ValueError: `extract_links` must be one of {None, "header", "footer", "body",'
                ' "all"}, got "x"'
            ),
        ),
    ),
    "multi-header": (lambda: fp.read_html(S(MULTIH)), ("[   a  b", "   c  d", "0  1  2]")),
    "nested": (lambda: fp.read_html(S(NESTED)), ("[    o", "0  in,     0", "0  in]")),
    "dtype-backend-bad": (
        lambda: fp.read_html(S(BASIC), dtype_backend="x"),
        (
            (
                "ValueError: dtype_backend x is invalid, only 'numpy_nullable' and 'pyarrow' are"
                " allowed."
            ),
        ),
    ),
    "header-bad": (
        lambda: fp.read_html(S(BASIC), header=-1),
        (
            (
                "ValueError: Passing negative integer to header is invalid. For no header, use"
                " header=None instead"
            ),
        ),
    ),
    "to_html-roundtrip": (
        lambda: fp.read_html(
            S(fp.DataFrame({"a": [1, 2], "b": ["x", None]}).to_html()), index_col=0
        ),
        ("[   a    b", "0  1    x", "1  2  NaN]"),
    ),
    "path-missing": (
        lambda: fp.read_html("/nonexistent/file.html"),
        ("FileNotFoundError: [Errno 2] No such file or directory: /nonexistent/file.html",),
    ),
    "bytes": (
        lambda: fp.read_html(io.BytesIO(BASIC.encode())),
        ("[   a  b", "0  1  x", "1  2  y]"),
    ),
    "empty-cells": (lambda: fp.read_html(S("<table><tr><td> </td></tr></table>")), ("[]",)),
    "multi-header-types": (lambda: fp.read_html(S(MULTIH))[0].iloc[0].tolist(), ("[1, 2]",)),
    "flavor-lxml-none": (
        lambda: fp.read_html(S("<p>hi</p>"), flavor="lxml"),
        ("ValueError: No tables found matching regex '.+'",),
    ),
    "flavor-lxml-match": (
        lambda: fp.read_html(S(TWO), flavor="lxml", match="nothing"),
        ("ValueError: No tables found matching regex 'nothing'",),
    ),
    "header-bool": (
        lambda: fp.read_html(S(BASIC), header=True),
        (
            (
                "TypeError: Passing a bool to header is invalid. Use header=None for no header or"
                " header=int or list-like of ints to specify the row(s) making up the column names"
            ),
        ),
    ),
    "links-body-values": (
        lambda: fp.read_html(S(LINK), extract_links="body")[0]["A"].tolist(),
        ("[('X', '/x')]",),
    ),
    "style-dropped": (
        lambda: fp.read_html(
            S("<table><tr><th>a<style>x{}</style></th></tr><tr><td>1</td></tr></table>")
        ),
        ("[   a", "0  1]"),
    ),
    "entity": (
        lambda: fp.read_html(
            S("<table><tr><th>a&amp;b</th></tr><tr><td>&lt;1&gt;</td></tr></table>")
        ),
        ("[   a&b", "0  <1>]"),
    ),
    "caption": (
        lambda: fp.read_html(
            S("<table><caption>Cap</caption><tr><th>a</th></tr><tr><td>1</td></tr></table>")
        ),
        ("[   a", "0  1]"),
    ),
    "two-th-rows": (
        lambda: fp.read_html(
            S("<table><tr><th>a</th></tr><tr><th>b</th></tr><tr><td>1</td></tr></table>")
        ),
        ("[   a", "   b", "0  1]"),
    ),
    "th-index": (
        lambda: fp.read_html(
            S("<table><tr><th></th><th>v</th></tr><tr><th>r</th><td>1</td></tr></table>")
        ),
        ("[  Unnamed: 0  v", "0          r  1]"),
    ),
}


def outcome(case: Callable[[], Any]) -> str:
    try:
        found = case()
    except Exception as mistake:
        kind = next(c.__name__ for c in type(mistake).__mro__ if c.__module__ == "builtins")
        return f"{kind}: {mistake}"
    return found if isinstance(found, str) else repr(found)


@pytest.mark.parametrize("name", list(CASES))
def test_reads_what_pandas_reads(name: str) -> None:
    case, expected = CASES[name]
    assert outcome(case) == "\n".join(expected)


def test_no_table_is_the_lxml_mistake() -> None:
    with pytest.raises(ValueError, match=r"No tables found matching regex '\.\+'"):
        fp.read_html(S("<p>hi</p>"))


def test_a_path_is_read(tmp_path: Any) -> None:
    path = tmp_path / "page.html"
    path.write_text("<table><tr><th>p</th></tr><tr><td>7</td></tr></table>")
    assert fp.read_html(path)[0]["p"].tolist() == [7]
    assert fp.read_html(str(path))[0]["p"].tolist() == [7]


def test_markup_in_a_string_is_a_path_as_in_pandas_3() -> None:
    with pytest.raises(FileNotFoundError):
        fp.read_html("<table><tr><td>1</td></tr></table>")


def test_a_link_in_the_body_is_kept_with_its_text() -> None:
    frame = fp.read_html(S(LINK), extract_links="body")[0]
    assert frame["A"].tolist() == [("X", "/x")]
