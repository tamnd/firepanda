"""`DataFrame.to_xml` and `read_xml`, a port of pandas' two XML formatters and two XML parsers.

The frame is read row by row through `to_dict(orient="index")`, after
`reset_index` when the index is written, and a missing value is swapped for its
stand-in when there is one, so every value is written as `str` writes it, the
way pandas does.

Each row becomes an element under the root, and each column becomes an
attribute of it or an element inside it. The `etree` parser uses the standard
library, pretty printed through `minidom`, and the `lxml` parser uses lxml
when it is installed, which is also the only one that runs a stylesheet.
"""

from __future__ import annotations

import codecs
from typing import Any

from ._pandas import NO_DEFAULT
from .errors import InvalidArgumentError

__all__ = ["read_xml", "to_xml"]


def _flat(prefix_uri: str, col: Any) -> str:
    """The name a column is written under, with a tuple joined as pandas joins it."""
    if isinstance(col, tuple):
        parts = [str(one) for one in col]
        col = ("".join(parts) if "" in col else "_".join(parts)).strip()
    return f"{prefix_uri}{col}"


class _Formatter:
    """What both parsers share: the rows, the columns and the namespace prefix."""

    def __init__(self, frame: Any, options: dict[str, Any]) -> None:
        from .api.types import is_list_like

        self.__dict__.update(options)
        self.orig_cols = list(frame.columns)
        if self.index:
            frame = frame.reset_index()
        self.rows = frame.to_dict(orient="index")
        if self.na_rep is not None:
            # pandas fills the frame, which needs a column of objects for text in a
            # column of numbers, so the stand-in goes into the rows instead.
            from ._pandas import isna

            for row in self.rows.values():
                row.update((key, self.na_rep) for key, value in row.items() if isna(value))
        for name in ("attr_cols", "elem_cols"):
            cols = getattr(self, name)
            if cols and not is_list_like(cols):
                raise TypeError(f"{type(cols).__name__} is not a valid type for {name}")
        codecs.lookup(self.encoding)
        self.prefix_uri = self._prefix_uri()
        if self.index:
            first = next(iter(self.rows))
            indexes = [key for key in self.rows[first] if key not in self.orig_cols]
            if self.attr_cols:
                self.attr_cols = indexes + list(self.attr_cols)
            if self.elem_cols:
                self.elem_cols = indexes + list(self.elem_cols)

    def _prefix_uri(self) -> str:
        if not self.namespaces:
            return ""
        if self.prefix:
            try:
                return f"{{{self.namespaces[self.prefix]}}}"
            except KeyError as err:
                raise KeyError(f"{self.prefix} is not included in namespaces") from err
        if "" in self.namespaces:
            return f"{{{self.namespaces['']}}}"
        return ""

    def _rows(self, root: Any, sub_element: Any) -> None:
        from ._pandas import isna

        for row in self.rows.values():
            element = sub_element(root, f"{self.prefix_uri}{self.row_name}")
            if not self.attr_cols and not self.elem_cols:
                self.elem_cols = list(row.keys())
            for col in self.attr_cols or ():
                try:
                    if not isna(row[col]):
                        element.attrib[_flat(self.prefix_uri, col)] = str(row[col])
                except KeyError as err:
                    raise KeyError(f"no valid column, {col}") from err
            for col in self.elem_cols or ():
                try:
                    value = None if isna(row[col]) or row[col] == "" else str(row[col])
                except KeyError as err:
                    raise KeyError(f"no valid column, {col}") from err
                sub_element(element, _flat(self.prefix_uri, col)).text = value


class _Etree(_Formatter):
    def _prefix_uri(self) -> str:
        from xml.etree.ElementTree import register_namespace

        for prefix, uri in (self.namespaces or {}).items():
            if isinstance(prefix, str) and isinstance(uri, str):
                register_namespace(prefix, uri)
        return super()._prefix_uri()

    def tree(self) -> bytes:
        from xml.etree.ElementTree import Element, SubElement, tostring

        others = {
            f"xmlns{'' if prefix == '' else f':{prefix}'}": uri
            for prefix, uri in (self.namespaces or {}).items()
            if uri != self.prefix_uri[1:-1]
        }
        root = Element(f"{self.prefix_uri}{self.root_name}", attrib=others)
        self._rows(root, SubElement)
        out = tostring(
            root, method="xml", encoding=self.encoding, xml_declaration=self.xml_declaration
        )
        if self.pretty_print:
            from xml.dom.minidom import parseString

            out = parseString(out).toprettyxml(indent="  ", encoding=self.encoding)
        if self.stylesheet is not None:
            raise ValueError("To use stylesheet, you need lxml installed and selected as parser.")
        return out


class _Lxml(_Formatter):
    def __init__(self, frame: Any, options: dict[str, Any]) -> None:
        super().__init__(frame, options)
        # lxml takes no empty prefix, so the default namespace is keyed by None.
        if self.namespaces and "" in self.namespaces:
            self.namespaces = dict(self.namespaces)
            self.namespaces[None] = self.namespaces.pop("")

    def tree(self) -> bytes:
        from lxml.etree import Element, SubElement, tostring

        root = Element(f"{self.prefix_uri}{self.root_name}", nsmap=self.namespaces)
        self._rows(root, SubElement)
        out = tostring(
            root,
            pretty_print=self.pretty_print,
            method="xml",
            encoding=self.encoding,
            xml_declaration=self.xml_declaration,
        )
        if self.stylesheet is None:
            return out
        return self._transformed(root)

    def _transformed(self, root: Any) -> bytes:
        import io
        import os

        from lxml.etree import XSLT, XMLParser, fromstring, parse

        parser = XMLParser(encoding=self.encoding)
        style = self.stylesheet
        if isinstance(style, (str, os.PathLike)):
            with open(os.path.expanduser(os.fspath(style)), "rb") as handle:
                xsl = parse(handle, parser=parser)
        elif isinstance(style, io.StringIO):
            xsl = fromstring(style.getvalue().encode(self.encoding), parser=parser)
        else:
            xsl = parse(style, parser=parser)
        return bytes(XSLT(xsl)(root))


def to_xml(frame: Any, path_or_buffer: Any, parser: Any, options: dict[str, Any]) -> str | None:
    """The document as text, or None once it is written to a path or a binary handle.

    Raises:
        ImportError: For the lxml parser when lxml is not installed.
        ValueError: For a parser other than lxml and etree, or a stylesheet
            without lxml.
        TypeError: For columns given as something that is not a list.
        KeyError: For a column the frame lacks, or a prefix with no namespace.
        LookupError: For an encoding Python does not know.
    """
    from ._pickle import _bytes_written

    if parser == "lxml":
        try:
            import lxml.etree  # noqa: F401
        except ImportError:
            raise ImportError("lxml not found, please install or use the etree parser.") from None
        formatter: _Formatter = _Lxml(frame, options)
    elif parser == "etree":
        formatter = _Etree(frame, options)
    else:
        raise ValueError("Values for parser can only be lxml or etree.")
    document = formatter.tree()  # type: ignore[attr-defined]
    if path_or_buffer is None:
        return document.decode(options["encoding"]).rstrip()
    _bytes_written(
        document, path_or_buffer, options.pop("compression"), options.pop("storage_options")
    )
    return None


_NO_NODES = (
    "xpath does not return any nodes or attributes. Be sure to specify in `xpath` the parent"
    " nodes of children and attributes to parse. If document uses namespaces denoted with"
    " xmlns, be sure to define namespaces and use them in xpath."
)


def _local(tag: str) -> str:
    """A tag or an attribute without the namespace URI ElementTree writes in braces."""
    return tag.split("}")[1] if "}" in tag else tag


def _text_of(element: Any) -> dict[str, Any]:
    """The element's own text as a column named by its tag, when it has any that is not space."""
    text = element.text
    return {element.tag: text} if text and not text.isspace() else {}


class _Reader:
    """What both parsers share: the options, and turning the chosen elements into rows."""

    def __init__(self, source: Any, options: dict[str, Any]) -> None:
        self.source = source
        self.__dict__.update(options)

    def _document(self, source: Any) -> Any:
        """The text of a path or a handle, decompressed, as a handle to parse."""
        import io

        from ._pandas import _json_source

        if self.storage_options is not None:
            # Only an fsspec URL takes storage options, never a local path or a handle.
            raise InvalidArgumentError(
                "storage_options passed with file object or non-fsspec file path"
            )
        if not hasattr(source, "read"):
            import errno
            import os

            path = os.path.expanduser(os.fspath(source))
            if not os.path.isfile(path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        return io.StringIO(_json_source(source, self.encoding, "strict", self.compression))

    def _rows(self, elements: list[Any]) -> list[dict[str, Any]]:
        """Each chosen element as a row: its attributes, its own text and its children's text."""
        names = self.names
        if self.elems_only and self.attrs_only:
            raise InvalidArgumentError("Either element or attributes can be parsed not both.")
        if self.elems_only:
            if names:
                rows = [
                    {
                        **_text_of(element),
                        **{
                            name: child.text if child.text else None
                            for name, child in zip(names, self._children(element), strict=True)
                        },
                    }
                    for element in elements
                ]
            else:
                rows = [
                    {child.tag: child.text if child.text else None for child in self._children(e)}
                    for e in elements
                ]
        elif self.attrs_only:
            rows = [{k: v if v else None for k, v in e.attrib.items()} for e in elements]
        elif names:
            rows = [
                {
                    **e.attrib,
                    **_text_of(e),
                    **{
                        name: child.text if child.text else None
                        for name, child in zip(names, self._children(e), strict=False)
                    },
                }
                for e in elements
            ]
        else:
            rows = [
                {
                    **e.attrib,
                    **_text_of(e),
                    **{
                        child.tag: child.text if child.text else None for child in self._children(e)
                    },
                }
                for e in elements
            ]
        rows = [{_local(key): value for key, value in row.items()} for row in rows]
        return self._aligned(rows)

    def _aligned(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Every row with every key any row has, None where it lacks one, renamed by `names`."""
        keys = list(dict.fromkeys(key for row in rows for key in row))
        rows = [{key: row.get(key) for key in keys} for row in rows]
        if self.names:
            rows = [dict(zip(self.names, row.values(), strict=True)) for row in rows]
        return rows

    def _streamed(self, iterparse: Any) -> list[dict[str, Any]]:
        """The rows `iterparse` names, read from a file on disk one element at a time."""
        import os

        from ._pandas import _CSV_ZIPPED
        from .api.types import is_list_like
        from .errors import ParserError

        wanted = self.iterparse
        if not isinstance(wanted, dict):
            raise TypeError(f"{type(wanted).__name__} is not a valid type for iterparse")
        row_node = next(iter(wanted.keys())) if wanted else ""
        if not is_list_like(wanted[row_node]):
            raise TypeError(f"{type(wanted[row_node])} is not a valid type for value in iterparse")
        source = self.source
        if not hasattr(source, "read") and (
            not isinstance(source, (str, os.PathLike))
            or str(source).startswith(("<?xml", "<", "http://", "https://", "ftp://", "s3://"))
            or "://" in str(source)
            or (self.compression == "infer" and str(source).lower().endswith(tuple(_CSV_ZIPPED)))
            or self.compression not in ("infer", None)
        ):
            raise ParserError(
                "iterparse is designed for large XML files that are fully extracted on local"
                " disk and not as compressed files or online sources."
            )
        columns = wanted[row_node]
        self._checked_names(columns)
        repeats = len(columns) != len(set(columns))
        rows: list[dict[str, Any]] = []
        row: dict[str, Any] | None = None
        for event, element in iterparse(source, events=("start", "end")):
            current = _local(element.tag)
            if event == "start" and current == row_node:
                row = {}
            if row is not None:
                if self.names and repeats:
                    for column, name in zip(columns, self.names, strict=True):
                        if current == column:
                            value = element.text if element.text else None
                            if value not in row.values() and name not in row:
                                row[name] = value
                        held = element.attrib.get(column)
                        if (
                            column in element.attrib
                            and held not in row.values()
                            and name not in row
                        ):
                            row[name] = held
                else:
                    for column in columns:
                        if current == column:
                            row[column] = element.text if element.text else None
                        if column in element.attrib:
                            row[column] = element.attrib[column]
            if event == "end":
                if current == row_node and row is not None:
                    rows.append(row)
                    row = None
                element.clear()
                if hasattr(element, "getprevious"):
                    while element.getprevious() is not None and element.getparent() is not None:
                        del element.getparent()[0]
        if not rows:
            raise ParserError("No result from selected items in iterparse.")
        return self._aligned(rows)

    def _checked_names(self, children: list[Any]) -> None:
        from .api.types import is_list_like

        if not self.names:
            return
        if not is_list_like(self.names):
            raise TypeError(f"{type(self.names).__name__} is not a valid type for names")
        if len(self.names) < len(children):
            raise InvalidArgumentError("names does not match length of child elements in xpath.")

    def _checked_nodes(self, elements: list[Any]) -> None:
        children = [child for e in elements for child in self._children(e)]
        attributes = {k: v for e in elements for k, v in e.attrib.items()}
        if (
            (self.elems_only and not children)
            or (self.attrs_only and not attributes)
            or (not children and not attributes)
        ):
            raise InvalidArgumentError(_NO_NODES)


class _EtreeReader(_Reader):
    """pandas' `etree` parser: the standard library, and the part of XPath it knows."""

    def _children(self, element: Any) -> list[Any]:
        return element.findall("*")

    def rows(self) -> list[dict[str, Any]]:
        from xml.etree.ElementTree import XMLParser, iterparse, parse

        if self.stylesheet is not None:
            raise InvalidArgumentError(
                "To use stylesheet, you need lxml installed and selected as parser."
            )
        if self.iterparse is not None:
            return self._streamed(iterparse)
        with self._document(self.source) as handle:
            root = parse(handle, parser=XMLParser(encoding=self.encoding)).getroot()
        try:
            elements = root.findall(self.xpath, namespaces=self.namespaces)
            self._checked_nodes(elements)
        except (KeyError, SyntaxError) as error:
            raise SyntaxError(
                "You have used an incorrect or unsupported XPath expression for etree library"
                " or you used an undeclared namespace prefix."
            ) from error
        parent = root.find(self.xpath, namespaces=self.namespaces)
        self._checked_names(parent.findall("*") if parent is not None else [])
        return self._rows(elements)


class _LxmlReader(_Reader):
    """pandas' `lxml` parser: full XPath 1.0, and XSLT 1.0 for a stylesheet."""

    def _children(self, element: Any) -> list[Any]:
        return element.xpath("*")

    def _tree(self, source: Any) -> Any:
        import io

        from lxml.etree import XMLParser, fromstring, parse

        parser = XMLParser(encoding=self.encoding)
        with self._document(source) as handle:
            if isinstance(handle, io.StringIO):
                if self.encoding is None:
                    raise TypeError("Can not pass encoding None when input is StringIO.")
                return fromstring(handle.getvalue().encode(self.encoding), parser=parser)
            return parse(handle, parser=parser)

    def rows(self) -> list[dict[str, Any]]:
        from lxml.etree import XSLT, iterparse

        if self.iterparse is not None:
            return self._streamed(iterparse)
        tree = self._tree(self.source)
        if self.stylesheet:
            tree = XSLT(self._tree(self.stylesheet))(tree)
        elements = tree.xpath(self.xpath, namespaces=self.namespaces)
        if not elements:
            raise InvalidArgumentError(_NO_NODES)
        self._checked_nodes(elements)
        self._checked_names(tree.xpath(self.xpath + "[1]/*", namespaces=self.namespaces))
        return self._rows(elements)


def _framed(
    rows: list[dict[str, Any]], dtype: Any, converters: Any, parse_dates: Any, dtype_backend: Any
) -> Any:
    """The rows as a frame, each column typed the way pandas' text parser types it.

    pandas hands the rows to the parser `read_csv` uses, so they are written as
    CSV text, with a missing value as an empty field, and read back by
    `read_csv`, which types a column of text the same way.
    """
    import csv
    import io

    from ._pandas import NO_DEFAULT, read_csv

    names = list(rows[0])
    text = io.StringIO()
    csv.writer(text, lineterminator="\n").writerows(
        ["" if value is None else value for value in row.values()] for row in rows
    )
    text.seek(0)
    frame = read_csv(
        text, header=None, names=names, dtype=dtype, converters=converters, parse_dates=parse_dates
    )
    if dtype_backend is NO_DEFAULT:
        return frame
    return frame.astype(
        {column: _backend_type(str(kind), dtype_backend) for column, kind in frame.dtypes.items()}
    )


_ARROW_TYPES = {"int64": "int64", "float64": "float64", "bool": "bool_", "string": "string"}
"""The Arrow type pandas' text parser reads each plain type as under `dtype_backend="pyarrow"`."""

_MASKED_TYPES = {"int64": "Int64", "float64": "Float64", "bool": "boolean", "string": "string"}
"""The masked type it reads each as under `dtype_backend="numpy_nullable"`."""


def _backend_type(kind: str, dtype_backend: str) -> Any:
    """The type a column of `kind` is read as under `dtype_backend`, or `kind` when it has none."""
    kind = "string" if kind == "str" else kind
    if kind not in _MASKED_TYPES:
        return kind
    if dtype_backend == "numpy_nullable":
        return _MASKED_TYPES[kind]
    import pyarrow as pa

    from ._arrowtyped import ArrowDtype

    return ArrowDtype(getattr(pa, _ARROW_TYPES[kind])())


def _read(path_or_buffer: Any, parser: Any, options: dict[str, Any]) -> Any:
    """A frame of the elements `xpath` picks, read by the parser pandas would use.

    Raises:
        ImportError: For the lxml parser when lxml is not installed.
        ValueError: For a parser other than lxml and etree, an `xpath` that
            picks nothing, and the other mistakes pandas names in its words.
        SyntaxError: For an `xpath` the etree parser cannot follow.
    """
    from ._pandas import NO_DEFAULT, _backend

    backend = options.pop("dtype_backend")
    if backend is not NO_DEFAULT:
        _backend(backend)
    if parser == "lxml":
        try:
            import lxml.etree  # noqa: F401
        except ImportError:
            raise ImportError("lxml not found, please install or use the etree parser.") from None
        reader: _Reader = _LxmlReader(path_or_buffer, options)
    elif parser == "etree":
        reader = _EtreeReader(path_or_buffer, options)
    else:
        raise InvalidArgumentError("Values for parser can only be lxml or etree.")
    rows = reader.rows()  # type: ignore[attr-defined]
    return _framed(rows, options["dtype"], options["converters"], options["parse_dates"], backend)


def read_xml(
    path_or_buffer: Any,
    *,
    xpath: str = "./*",
    namespaces: dict[str, str] | None = None,
    elems_only: bool = False,
    attrs_only: bool = False,
    names: Any = None,
    dtype: Any = None,
    converters: Any = None,
    parse_dates: Any = None,
    encoding: str | None = "utf-8",
    parser: str = "lxml",
    stylesheet: Any = None,
    iterparse: dict[str, list[str]] | None = None,
    compression: Any = "infer",
    storage_options: Any = None,
    dtype_backend: Any = NO_DEFAULT,
) -> Any:
    """A frame of the elements `xpath` picks from an XML document, as pandas reads it.

    Each element is a row, and its attributes, its own text and the text of
    its children are the columns, typed the way `read_csv` types text. The
    default `lxml` parser needs lxml, as in pandas, and `parser="etree"` uses
    the standard library and the part of XPath it follows.

    Raises:
        ImportError: For the lxml parser when lxml is not installed.
        ValueError: For an `xpath` that picks nothing, and the other mistakes
            pandas names in its words.
    """
    return _read(
        path_or_buffer,
        parser,
        {
            "xpath": xpath,
            "namespaces": namespaces,
            "elems_only": elems_only,
            "attrs_only": attrs_only,
            "names": names,
            "dtype": dtype,
            "converters": converters,
            "parse_dates": parse_dates,
            "encoding": encoding,
            "stylesheet": stylesheet,
            "iterparse": iterparse,
            "compression": compression,
            "storage_options": storage_options,
            "dtype_backend": dtype_backend,
        },
    )
