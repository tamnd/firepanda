"""`DataFrame.to_xml`, a port of pandas' two XML formatters.

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

__all__ = ["to_xml"]


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
