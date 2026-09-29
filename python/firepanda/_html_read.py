"""`read_html`, the tables of an HTML document as frames, a port of pandas' HTML reader.

pandas parses the document with lxml, or with BeautifulSoup and html5lib, and
then works the same way on either tree. This module builds a small tree with
the standard library's `html.parser`, closing the cells, rows and sections a
page leaves open the way an HTML parser does, and then follows pandas' steps:
pick the tables whose text matches and whose attributes agree, split each into
header, body and footer rows, copy a cell across its `colspan` and down its
`rowspan`, and hand the rows to the parser `read_csv` uses, so the columns are
typed as pandas types them.
"""

from __future__ import annotations

import numbers
import os
import re
from html.parser import HTMLParser
from typing import Any

from ._pandas import NO_DEFAULT
from .errors import InvalidArgumentError

__all__ = ["read_html"]

_WHITESPACE = re.compile(r"[\r\n]+|\s{2,}")

_VOID = frozenset(
    [
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    ]
)

_SECTIONS = ("thead", "tbody", "tfoot")

_FLAVORS = ("lxml", None, "html5lib", "bs4")

_LINKS = (None, "header", "footer", "body", "all")


class _Node:
    """An element: its tag, its attributes, and its children, which are nodes and text."""

    __slots__ = ("attrs", "children", "tag")

    def __init__(self, tag: str, attrs: dict[str, str]) -> None:
        self.tag = tag
        self.attrs = attrs
        self.children: list[Any] = []

    def elements(self) -> list[_Node]:
        return [child for child in self.children if isinstance(child, _Node)]

    def descendants(self) -> list[_Node]:
        """Every element under this one, in document order."""
        found = []
        for child in self.elements():
            found.append(child)
            found.extend(child.descendants())
        return found

    def texts(self) -> list[str]:
        """Every run of text under this one, as lxml's `.//text()` lists them."""
        found = []
        for child in self.children:
            if isinstance(child, str):
                found.append(child)
            else:
                found.extend(child.texts())
        return found

    def text(self) -> str:
        """The text of the element and everything in it, with a line break for `br`."""
        if self.tag == "br":
            return "\n"
        return "".join(child if isinstance(child, str) else child.text() for child in self.children)


class _Builder(HTMLParser):
    """Builds the tree, closing what a table leaves open as an HTML parser would."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("#document", {})
        self.stack = [self.root]

    def _open(self, tags: tuple[str, ...], stop: tuple[str, ...]) -> int | None:
        """Where the nearest open element among `tags` is, looking no further than `stop`."""
        for place in range(len(self.stack) - 1, 0, -1):
            tag = self.stack[place].tag
            if tag in tags:
                return place
            if tag in stop:
                return None
        return None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("td", "th"):
            place = self._open(("td", "th"), ("tr", "table"))
        elif tag == "tr":
            place = self._open(("tr",), ("table",))
        elif tag in _SECTIONS:
            place = self._open(_SECTIONS, ("table",))
        else:
            place = None
        if place is not None:
            del self.stack[place:]
        node = _Node(tag, {name: value or "" for name, value in attrs})
        self.stack[-1].children.append(node)
        if tag not in _VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID and self.stack[-1].tag == tag:
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        stop = () if tag == "table" else ("table",)
        place = self._open((tag,), stop)
        if place is not None:
            del self.stack[place:]

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def _document(io: Any, encoding: Any, storage_options: Any) -> str:
    """The text of the document, read from a path, a URL or anything with `read`."""
    if isinstance(io, os.PathLike):
        io = os.fspath(io)
    if hasattr(io, "read"):
        raw = io.read()
    elif isinstance(io, str) and re.match(r"^(https?|ftp|file)://", io):
        from urllib.request import Request, urlopen

        headers = dict(storage_options or {})
        with urlopen(Request(io, headers=headers)) as response:
            raw = response.read()
    elif isinstance(io, (str, bytes)):
        try:
            with open(io, "rb") as handle:
                raw = handle.read()
        except OSError:
            shown = io.decode() if isinstance(io, bytes) else io
            raise FileNotFoundError(f"[Errno 2] No such file or directory: {shown}") from None
    else:
        raise TypeError(f"Cannot read object of type '{type(io).__name__}'")
    if isinstance(raw, bytes):
        raw = raw.decode(encoding or "utf-8", errors="replace")
    return raw


def _hidden(node: _Node) -> bool:
    return "display:none" in node.attrs.get("style", "").replace(" ", "")


def _drop_hidden(node: _Node) -> None:
    """Takes out the `style` elements and the elements styled `display:none`."""
    node.children = [
        child
        for child in node.children
        if not (isinstance(child, _Node) and (child.tag == "style" or _hidden(child)))
    ]
    for child in node.elements():
        _drop_hidden(child)


def _tables(root: _Node, match: re.Pattern[str], attrs: Any, displayed_only: bool) -> list[_Node]:
    """The tables with a run of text `match` finds and every attribute in `attrs`."""
    tables = [
        node
        for node in root.descendants()
        if node.tag == "table"
        and any(match.search(text) for text in node.texts())
        and all(node.attrs.get(name) == value for name, value in (attrs or {}).items())
    ]
    if displayed_only:
        tables = [table for table in tables if not _hidden(table)]
        for table in tables:
            _drop_hidden(table)
    if not tables:
        raise InvalidArgumentError(f"No tables found matching regex {match.pattern!r}")
    return tables


def _cells(row: _Node) -> list[_Node]:
    return [child for child in row.elements() if child.tag in ("td", "th")]


def _sections(table: _Node) -> tuple[list[_Node], list[_Node], list[_Node]]:
    """The header, body and footer rows, found as pandas' lxml parser finds them."""
    inside = table.descendants()
    head: list[_Node] = []
    for thead in (node for node in inside if node.tag == "thead"):
        head.extend(child for child in thead.elements() if child.tag == "tr")
        if _cells(thead):
            head.append(thead)
    body = [
        row
        for section in inside
        if section.tag == "tbody"
        for row in section.descendants()
        if row.tag == "tr"
    ]
    body += [child for child in table.elements() if child.tag == "tr"]
    foot = [
        row
        for section in inside
        if section.tag == "tfoot"
        for row in section.descendants()
        if row.tag == "tr"
    ]
    return head, body, foot


def _link(cell: _Node) -> Any:
    for node in cell.descendants():
        if node.tag == "a" and "href" in node.attrs:
            return node.attrs["href"]
    return None


def _expanded(
    rows: list[_Node], section: str, links: Any, remainder: list[Any], overflow: bool
) -> tuple[list[list[Any]], list[Any]]:
    """The rows as lists of text, each cell repeated across its colspan and down its rowspan."""
    texts_of_rows = []
    for row in rows:
        texts: list[Any] = []
        following = []
        index = 0
        for cell in _cells(row):
            while remainder and remainder[0][0] <= index:
                where, text, span = remainder.pop(0)
                texts.append(text)
                if span > 1:
                    following.append((where, text, span - 1))
                index += 1
            text: Any = _WHITESPACE.sub(" ", cell.text().strip())
            if links in ("all", section):
                text = (text, _link(cell))
            rowspan = int(cell.attrs.get("rowspan") or 1)
            colspan = int(cell.attrs.get("colspan") or 1)
            for _ in range(colspan):
                texts.append(text)
                if rowspan > 1:
                    following.append((index, text, rowspan - 1))
                index += 1
        for where, text, span in remainder:
            texts.append(text)
            if span > 1:
                following.append((where, text, span - 1))
        texts_of_rows.append(texts)
        remainder = following
    if not overflow:
        while remainder:
            following = []
            texts = []
            for where, text, span in remainder:
                texts.append(text)
                if span > 1:
                    following.append((where, text, span - 1))
            texts_of_rows.append(texts)
            remainder = following
    return texts_of_rows, remainder


def _parsed(table: _Node, links: Any) -> tuple[list[Any], list[Any], list[Any]]:
    head_rows, body_rows, foot_rows = _sections(table)
    if not head_rows:
        while body_rows and all(cell.tag == "th" for cell in _cells(body_rows[0])):
            head_rows.append(body_rows.pop(0))
    head, rest = _expanded(head_rows, "header", links, [], True)
    body, rest = _expanded(body_rows, "body", links, rest, bool(foot_rows))
    foot, _ = _expanded(foot_rows, "footer", links, rest, False)
    return head, body, foot


def _skiprows(skiprows: Any) -> Any:
    if isinstance(skiprows, slice):
        start, step = skiprows.start or 0, skiprows.step or 1
        return list(range(start, skiprows.stop, step))
    if isinstance(skiprows, numbers.Integral) or skiprows is None:
        return skiprows or 0
    if isinstance(skiprows, (list, tuple, set, range)) or hasattr(skiprows, "__iter__"):
        return skiprows
    raise TypeError(f"{type(skiprows).__name__} is not a valid type for skipping rows")


def _framed(
    parsed: tuple[list[Any], list[Any], list[Any]], header: Any, options: dict[str, Any]
) -> Any:
    """The rows as a frame, through `read_csv` as pandas goes through its text parser."""
    import csv
    import io

    from ._pandas import read_csv

    head, body, foot = parsed
    if head:
        body = head + body
        if header is None:
            header = 0 if len(head) == 1 else [i for i, row in enumerate(head) if any(row)]
    body = body + foot
    pairs: list[tuple[str, Any]] = []
    for row in body:
        for place, cell in enumerate(row):
            if isinstance(cell, tuple):
                row[place] = f"\x00{len(pairs)}"
                pairs.append(cell)
    width = max((len(row) for row in body), default=0)
    body = [row + [""] * (width - len(row)) if row != [""] else row for row in body]
    labels, index_col = None, options["index_col"]
    if isinstance(header, list) and len(header) > 1:
        labels = [tuple(body[place][column] for place in header) for column in range(width)]
        body, header = body[max(header) + 1 :], None
        options = {**options, "index_col": None}
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    for row in body:
        if row == [""]:
            text.write("\n")
        else:
            writer.writerow(row)
    text.seek(0)
    frame = read_csv(text, header=header, **options)
    if labels is not None:
        frame = _relabelled(frame, labels, index_col)
    return _linked(frame, pairs) if pairs else frame


def _relabelled(frame: Any, labels: list[tuple[Any, ...]], index_col: Any) -> Any:
    """The frame with its columns labelled by the tuples a header of more than one row makes."""
    from ._frame import DataFrame

    built = DataFrame(
        {label: frame[column].tolist() for label, column in zip(labels, frame.columns, strict=True)}
    )
    if index_col is not None:
        built = built.set_index(built.columns[index_col])
    return built


def _linked(frame: Any, pairs: list[tuple[str, Any]]) -> Any:
    """The frame with each stand-in swapped back for the (text, link) pair it stood for."""
    from ._frame import DataFrame

    def restored(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("\x00"):
            return pairs[int(value[1:])]
        return value

    columns = {}
    for label in frame.columns:
        values = frame[label].tolist()
        if any(isinstance(value, str) and value.startswith("\x00") for value in values):
            values = [restored(value) for value in values]
        columns[restored(label)] = values
    return DataFrame(columns, index=frame.index)


def read_html(
    io: Any,
    *,
    match: Any = ".+",
    flavor: Any = None,
    header: Any = None,
    index_col: Any = None,
    skiprows: Any = None,
    attrs: Any = None,
    parse_dates: Any = False,
    thousands: Any = ",",
    encoding: Any = None,
    decimal: str = ".",
    converters: Any = None,
    na_values: Any = None,
    keep_default_na: bool = True,
    displayed_only: bool = True,
    extract_links: Any = None,
    dtype_backend: Any = NO_DEFAULT,
    storage_options: Any = None,
) -> list[Any]:
    """Every table in an HTML document that `match` and `attrs` pick, each as a frame.

    `io` is a path, a URL or anything with a `read` method. A string is a path,
    as in pandas 3, never the markup itself, so markup is passed in a
    `StringIO`. A row of only `th` cells at the top of a table with no `thead`
    is the header, a cell is copied across its `colspan` and down its
    `rowspan`, and short rows are padded with gaps. The cells are then typed
    the way `read_csv` types text, with `thousands`, `decimal`, `na_values`,
    `converters` and `parse_dates` passed on to it.

    Raises:
        ValueError: When no table matches, and for the other mistakes pandas
            names in its words.
        FileNotFoundError: For a path that does not exist.
    """
    from ._pandas import _backend
    from .errors import EmptyDataError

    if isinstance(skiprows, numbers.Integral) and skiprows < 0:
        raise InvalidArgumentError(
            "cannot skip rows starting from the end of the data (you passed a negative value)"
        )
    if extract_links not in _LINKS:
        raise InvalidArgumentError(
            '`extract_links` must be one of {None, "header", "footer", "body", "all"}, got '
            f'"{extract_links}"'
        )
    if isinstance(header, bool):
        raise TypeError(
            "Passing a bool to header is invalid. Use header=None for no header or "
            "header=int or list-like of ints to specify the row(s) making up the column names"
        )
    if isinstance(header, int) and header < 0:
        raise InvalidArgumentError(
            "Passing negative integer to header is invalid. For no header, use header=None instead"
        )
    if dtype_backend is not NO_DEFAULT:
        _backend(dtype_backend)
    flavors = (flavor,) if flavor is None or isinstance(flavor, str) else tuple(flavor)
    if not set(flavors) & set(_FLAVORS):
        shown = "{" + ", ".join(str(one) for one in flavors) + "}"
        raise InvalidArgumentError(
            f"{shown} is not a valid set of flavors, valid flavors are "
            "{lxml, None, html5lib, bs4}"
        )
    builder = _Builder()
    builder.feed(_document(io, encoding, storage_options))
    builder.close()
    pattern = re.compile(match)
    options = {
        "index_col": index_col,
        "skiprows": _skiprows(skiprows),
        "parse_dates": parse_dates,
        "thousands": thousands,
        "decimal": decimal,
        "converters": converters,
        "na_values": na_values,
        "keep_default_na": keep_default_na,
    }
    frames = []
    for table in _tables(builder.root, pattern, attrs, displayed_only):
        try:
            frame = _framed(_parsed(table, extract_links), header, options)
        except EmptyDataError:
            continue
        if dtype_backend is not NO_DEFAULT:
            from ._xml import _backend_type

            frame = frame.astype(
                {
                    name: _backend_type(str(kind), dtype_backend)
                    for name, kind in frame.dtypes.items()
                }
            )
        frames.append(frame)
    return frames
