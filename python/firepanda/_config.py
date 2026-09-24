"""pandas' options, `get_option`, `set_option`, `reset_option`, `describe_option`,
`option_context` and `options`.

Every one of the 73 options pandas 3.0 registers is registered here too, under
the same name, with the same default and the same check on a new value, so a
script that reads or sets one runs unchanged. The rules below were measured
against pandas 3.0.

- A name is a pattern. An exact name is that option, and otherwise the name is
  searched for in every option's name, ignoring case, so `max_colwidth` finds
  `display.max_colwidth`. Getting or setting needs exactly one match, and
  resetting takes every match but wants at least four characters when there
  are several, with `all` for every option.
- A value is checked before it is stored, and a pair that fails leaves the pairs
  before it set.
- `mode.copy_on_write` and `future.no_silent_downcasting` still answer, with a
  `Pandas4Warning` each time they are named.

Most options change nothing here. The printed form of a frame is made by the
extension, which knows nothing of these options, and there is no plotting, no
Excel and no HDF5. `display.max_info_columns` is read by `info`, which is the
one place an option has an effect.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, NamedTuple

from .errors import OptionError, Pandas4Warning

__all__ = [
    "describe_option",
    "get_option",
    "option_context",
    "options",
    "reset_option",
    "set_option",
]


def _type_is(kind: type) -> Callable[[Any], None]:
    """A check that the value is exactly of one type, so `True` is not an int."""

    def check(value: Any) -> None:
        if type(value) is not kind:
            raise ValueError(f"Value must have type '{kind}'")

    return check


def _instance_of(*kinds: type) -> Callable[[Any], None]:
    """A check that the value is an instance of one of several types."""
    printed = "|".join(str(kind) for kind in kinds)

    def check(value: Any) -> None:
        if not isinstance(value, kinds):
            raise ValueError(f"Value must be an instance of {printed}")

    return check


def _one_of(*legal: Any) -> Callable[[Any], None]:
    """A check that the value is one of a few, compared with `==` as pandas compares."""
    printed = "|".join(str(value) for value in legal)

    def check(value: Any) -> None:
        if value not in legal:
            raise ValueError(f"Value must be one of {printed}")

    return check


def _nonnegative(value: Any) -> None:
    """A count of zero or more, or None for no limit."""
    if value is None or (isinstance(value, int) and value >= 0):
        return
    raise ValueError("Value must be a nonnegative integer or None")


def _none_or_callable(value: Any) -> None:
    """A function to write floats with, or None for the usual way."""
    if value is not None and not callable(value):
        raise ValueError("Value must be a callable")


def _string_storage(value: Any) -> None:
    """Where text is kept, with `auto` allowed though pandas' message leaves it out."""
    if value not in ("auto", "python", "pyarrow"):
        raise ValueError("Value must be one of python|pyarrow")


def _plotting_backend(value: Any) -> None:
    """`matplotlib`, or a module with a `plot` at the top of it, which is where pandas looks."""
    if value == "matplotlib":
        return
    import importlib

    try:
        module = importlib.import_module(value)
    except ImportError:
        module = None
    if module is None or not hasattr(module, "plot"):
        raise ValueError(
            f"Could not find plotting backend '{value}'. Ensure that you've installed the"
            f" package providing the '{value}' entrypoint, or that the package has a"
            " top-level `.plot` method."
        )


def _anything(value: Any) -> None:
    """No check, for the options pandas registers without one."""


class _Option(NamedTuple):
    """One registered option: its default, its description and the check on a new value."""

    default: Any
    kind: str
    doc: str
    check: Callable[[Any], None]


_BOOL = _type_is(bool)
_INT = _type_is(int)
_STR = _type_is(str)
_TEXT = _instance_of(str, bytes)
_WIDTH = _instance_of(type(None), int)
_OPTIONAL_TEXT = _instance_of(type(None), str)
_EXCEL = ("calamine", "auto")

_REGISTERED: dict[str, _Option] = {
    "compute.use_bottleneck": _Option(
        True, "bool", "Use the bottleneck library to accelerate computation if installed.", _BOOL
    ),
    "compute.use_numba": _Option(
        False, "bool", "Use the numba engine for operations that support it.", _BOOL
    ),
    "compute.use_numexpr": _Option(
        True, "bool", "Use the numexpr library to accelerate computation if installed.", _BOOL
    ),
    "display.chop_threshold": _Option(
        None, "float or None", "Floats below this in absolute value print as zero.", _anything
    ),
    "display.colheader_justify": _Option(
        "right", "'left'/'right'", "How column headers are justified.", _TEXT
    ),
    "display.date_dayfirst": _Option(
        False, "boolean", "Read and print dates with the day first.", _BOOL
    ),
    "display.date_yearfirst": _Option(
        False, "boolean", "Read and print dates with the year first.", _BOOL
    ),
    "display.encoding": _Option(
        "utf-8", "str/unicode", "The encoding for text written to the console.", _TEXT
    ),
    "display.expand_frame_repr": _Option(
        True, "boolean", "Print a wide frame across several lines.", _anything
    ),
    "display.float_format": _Option(
        None, "callable", "A function that writes a float as text.", _none_or_callable
    ),
    "display.html.border": _Option(1, "int", "The border attribute of a printed HTML table.", _INT),
    "display.html.table_schema": _Option(
        False, "boolean", "Publish a Table Schema representation for frontends.", _BOOL
    ),
    "display.html.use_mathjax": _Option(
        True, "boolean", "Let MathJax render text in dollar signs in HTML tables.", _BOOL
    ),
    "display.large_repr": _Option(
        "truncate",
        "'truncate'/'info'",
        "What a frame too large to print whole prints as.",
        _one_of("truncate", "info"),
    ),
    "display.max_categories": _Option(
        8, "int", "How many categories a categorical column prints.", _INT
    ),
    "display.max_columns": _Option(
        0, "int", "How many columns print before the rest are elided.", _nonnegative
    ),
    "display.max_colwidth": _Option(
        50, "int or None", "How many characters a cell prints.", _nonnegative
    ),
    "display.max_dir_items": _Option(
        100, "int", "How many columns tab completion offers.", _nonnegative
    ),
    "display.max_info_columns": _Option(
        100, "int", "How many columns `info` lists one by one.", _INT
    ),
    "display.max_info_rows": _Option(1690785, "int", "How many rows `info` counts gaps in.", _INT),
    "display.max_rows": _Option(
        60, "int", "How many rows print before the rest are elided.", _nonnegative
    ),
    "display.max_seq_items": _Option(
        100, "int or None", "How many items a long sequence prints.", _anything
    ),
    "display.memory_usage": _Option(
        True,
        "bool, string or None",
        "Whether `info` prints the memory a frame uses.",
        _one_of(None, True, False, "deep"),
    ),
    "display.min_rows": _Option(10, "int", "How many rows a truncated frame prints.", _WIDTH),
    "display.multi_sparse": _Option(
        True, "boolean", "Print repeated outer labels of a hierarchical index once.", _BOOL
    ),
    "display.notebook_repr_html": _Option(
        True, "boolean", "Print frames as HTML in a notebook.", _BOOL
    ),
    "display.pprint_nest_depth": _Option(3, "int", "How deep nested sequences print.", _INT),
    "display.precision": _Option(6, "int", "How many decimals a float prints.", _nonnegative),
    "display.show_dimensions": _Option(
        "truncate",
        "boolean or 'truncate'",
        "Whether a printed frame ends with its shape.",
        _one_of(True, False, "truncate"),
    ),
    "display.unicode.ambiguous_as_wide": _Option(
        False, "boolean", "Count characters of ambiguous width as two columns.", _BOOL
    ),
    "display.unicode.east_asian_width": _Option(
        False, "boolean", "Measure text by East Asian width when aligning columns.", _BOOL
    ),
    "display.width": _Option(80, "int", "The width of the console in characters.", _WIDTH),
    "future.distinguish_nan_and_na": _Option(
        False, "bool", "Keep NaN apart from NA in nullable float columns.", _one_of(True, False)
    ),
    "future.infer_string": _Option(
        True, "bool", "Read a sequence of text as the string type.", _one_of(True, False)
    ),
    "future.no_silent_downcasting": _Option(
        False,
        "bool",
        "This option is deprecated and will be removed in a future version. It has no effect.",
        _one_of(True, False),
    ),
    "future.python_scalars": _Option(
        False, "bool", "Answer Python scalars rather than numpy ones.", _one_of(True, False)
    ),
    "io.excel.ods.reader": _Option(
        "auto", "string", "The engine that reads ods files.", _one_of("odf", *_EXCEL)
    ),
    "io.excel.ods.writer": _Option("auto", "string", "The engine that writes ods files.", str),
    "io.excel.xls.reader": _Option(
        "auto", "string", "The engine that reads xls files.", _one_of("xlrd", *_EXCEL)
    ),
    "io.excel.xlsb.reader": _Option(
        "auto", "string", "The engine that reads xlsb files.", _one_of("pyxlsb", *_EXCEL)
    ),
    "io.excel.xlsm.reader": _Option(
        "auto",
        "string",
        "The engine that reads xlsm files.",
        _one_of("xlrd", "openpyxl", *_EXCEL),
    ),
    "io.excel.xlsm.writer": _Option("auto", "string", "The engine that writes xlsm files.", str),
    "io.excel.xlsx.reader": _Option(
        "auto",
        "string",
        "The engine that reads xlsx files.",
        _one_of("xlrd", "openpyxl", *_EXCEL),
    ),
    "io.excel.xlsx.writer": _Option("auto", "string", "The engine that writes xlsx files.", str),
    "io.hdf.default_format": _Option(
        None, "format", "The format HDF5 files are written in.", _one_of("fixed", "table", None)
    ),
    "io.hdf.dropna_table": _Option(
        False, "boolean", "Leave out rows that are all missing when writing HDF5.", _BOOL
    ),
    "io.parquet.engine": _Option(
        "auto",
        "string",
        "The engine that reads and writes Parquet.",
        _one_of("auto", "pyarrow", "fastparquet"),
    ),
    "io.sql.engine": _Option(
        "auto", "string", "The engine that reads and writes SQL.", _one_of("auto", "sqlalchemy")
    ),
    "mode.chained_assignment": _Option(
        "warn",
        "string",
        "What a chained assignment does.",
        _one_of(None, "warn", "raise"),
    ),
    "mode.copy_on_write": _Option(
        True,
        "bool",
        "No longer used, a copy is always made on write. This option will be removed.",
        _one_of(True, False, "warn"),
    ),
    "mode.performance_warnings": _Option(
        True, "boolean", "Whether a PerformanceWarning is raised.", _BOOL
    ),
    "mode.sim_interactive": _Option(
        False, "boolean", "Behave as if the console were interactive.", _anything
    ),
    "mode.string_storage": _Option(
        "auto", "string", "Where the string type keeps its text.", _string_storage
    ),
    "plotting.backend": _Option(
        "matplotlib", "str", "The module that draws plots.", _plotting_backend
    ),
    "plotting.matplotlib.register_converters": _Option(
        "auto",
        "bool or 'auto'.",
        "Register pandas' converters with matplotlib.",
        _one_of("auto", True, False),
    ),
    "styler.format.decimal": _Option(".", "str", "The decimal separator a Styler writes.", _STR),
    "styler.format.escape": _Option(
        None,
        "str, optional",
        "How a Styler escapes text.",
        _one_of(None, "html", "latex", "latex-math"),
    ),
    "styler.format.formatter": _Option(
        None,
        "str, callable, dict, optional",
        "The formatter a Styler uses by default.",
        _instance_of(type(None), dict, Callable, str),  # type: ignore[arg-type]
    ),
    "styler.format.na_rep": _Option(
        None, "str, optional", "What a Styler writes for a gap.", _OPTIONAL_TEXT
    ),
    "styler.format.precision": _Option(
        6, "int", "How many decimals a Styler writes.", _nonnegative
    ),
    "styler.format.thousands": _Option(
        None, "str, optional", "The thousands separator a Styler writes.", _OPTIONAL_TEXT
    ),
    "styler.html.mathjax": _Option(True, "bool", "Let MathJax render a Styler's HTML.", _BOOL),
    "styler.latex.environment": _Option(
        None, "str", "The LaTeX environment a Styler writes.", _OPTIONAL_TEXT
    ),
    "styler.latex.hrules": _Option(
        False, "bool", "Draw rules at the top, bottom and header in LaTeX.", _BOOL
    ),
    "styler.latex.multicol_align": _Option(
        "r",
        '{"r", "c", "l", "naive-l", "naive-r"}',
        "How sparsified column labels align in LaTeX.",
        _one_of(
            *(
                f"{left}{side}{right}"
                for side in "rcl"
                for left, right in (("", ""), ("|", "|"), ("|", ""), ("", "|"))
            ),
            "naive-l",
            "naive-r",
        ),
    ),
    "styler.latex.multirow_align": _Option(
        "c",
        '{"c", "t", "b"}',
        "How sparsified row labels align in LaTeX.",
        _one_of("c", "t", "b", "naive"),
    ),
    "styler.render.encoding": _Option(
        "utf-8", "str", "The encoding a Styler writes its output in.", _STR
    ),
    "styler.render.max_columns": _Option(
        None, "int, optional", "How many columns a Styler renders.", _nonnegative
    ),
    "styler.render.max_elements": _Option(
        262144, "int", "How many cells a Styler renders.", _nonnegative
    ),
    "styler.render.max_rows": _Option(
        None, "int, optional", "How many rows a Styler renders.", _nonnegative
    ),
    "styler.render.repr": _Option(
        "html", "str", "What a Styler renders as in a notebook.", _one_of("html", "latex")
    ),
    "styler.sparse.columns": _Option(
        True, "bool", "Print repeated column labels of a hierarchy once.", _BOOL
    ),
    "styler.sparse.index": _Option(
        True, "bool", "Print repeated row labels of a hierarchy once.", _BOOL
    ),
}
"""Every option pandas 3.0 registers, by its full name."""

_DEPRECATED: dict[str, str] = {
    "future.no_silent_downcasting": (
        "'future.no_silent_downcasting' is deprecated, please refrain from using it."
    ),
    "mode.copy_on_write": (
        "The 'mode.copy_on_write' option is deprecated. Copy-on-Write can no longer be"
        " disabled (it is always enabled with pandas >= 3.0), and setting the option has"
        " no impact. This option will be removed in pandas 4.0."
    ),
}
"""The options that warn when named, with the warning pandas gives."""

_VALUES: dict[str, Any] = {key: option.default for key, option in _REGISTERED.items()}
"""The value each option holds now."""


def _matches(pat: str) -> list[str]:
    """The options a pattern names: itself, all of them, or every name it is found in."""
    if pat in _REGISTERED:
        return [pat]
    keys = sorted(_REGISTERED)
    if pat == "all":
        return keys
    return [key for key in keys if re.search(pat, key, re.IGNORECASE)]


def _warn_if_deprecated(key: str) -> None:
    """A `Pandas4Warning` for an option pandas is about to drop."""
    if key in _DEPRECATED:
        warnings.warn(_DEPRECATED[key], Pandas4Warning, stacklevel=4)


def _one_key(pat: str) -> str:
    """The one option a pattern names, or the error pandas raises when it names none or several."""
    keys = _matches(pat)
    if not keys:
        _warn_if_deprecated(pat)
        raise OptionError(f"No such keys(s): {pat!r}")
    if len(keys) > 1:
        raise OptionError("Pattern matched multiple keys")
    _warn_if_deprecated(keys[0])
    return keys[0]


def get_option(pat: str) -> Any:
    """The value of the one option `pat` names.

    Args:
        pat: An option's full name, or text found in exactly one option's name.

    Returns:
        The value the option holds.

    Raises:
        OptionError: No option or several options match.
    """
    return _VALUES[_one_key(pat)]


def set_option(*args: Any) -> None:
    """Set options from pairs of a pattern and a value, one pair after another.

    Args:
        *args: A pattern, its value, and as many more pairs as wanted.

    Raises:
        ValueError: The arguments are not pairs, or a value fails its option's check.
        OptionError: A pattern matches no option or several.
    """
    if not args or len(args) % 2:
        raise ValueError("Must provide an even number of non-keyword arguments")
    for pat, value in zip(args[::2], args[1::2], strict=True):
        key = _one_key(pat)
        _REGISTERED[key].check(value)
        _VALUES[key] = value


def reset_option(pat: str) -> None:
    """Put every option `pat` matches back to its default.

    Args:
        pat: A pattern, which must be four characters or more when it matches several
            options, or `all` for every option.

    Raises:
        OptionError: No option matches.
        ValueError: A short pattern matches several options.
    """
    keys = _matches(pat)
    if not keys:
        raise OptionError(f"No such keys(s) for pat={pat!r}")
    if len(keys) > 1 and len(pat) < 4 and pat != "all":
        raise ValueError(
            "You must specify at least 4 characters when resetting multiple keys,"
            ' use the special keyword "all" to reset all the options to their default value'
        )
    for key in keys:
        set_option(key, _REGISTERED[key].default)


def _description(key: str) -> str:
    """One option described the way pandas describes it, with its default and value."""
    option = _REGISTERED[key]
    text = f"{key} : {option.kind}\n    {option.doc}"
    text += f"\n    [default: {option.default}] [currently: {_VALUES[key]}]"
    if key in _DEPRECATED:
        text += "\n    (Deprecated, use `` instead.)"
    return text


def describe_option(pat: str = "", _print_desc: bool = True) -> str | None:
    """Describe every option `pat` matches, printed or answered as text.

    Args:
        pat: A pattern, and every option when it is empty.
        _print_desc: Print the description and answer None, rather than answer it.

    Returns:
        The description when `_print_desc` is false, and None otherwise.

    Raises:
        OptionError: No option matches.
    """
    keys = _matches(pat)
    if not keys:
        raise OptionError(f"No such keys(s) for pat={pat!r}")
    text = "\n".join(_description(key) for key in keys)
    if _print_desc:
        print(text)
        return None
    return text


@contextmanager
def option_context(*args: Any) -> Generator[None]:
    """Set options for the length of a `with` block and put them back after it.

    Args:
        *args: Pairs of a pattern and a value, or one dict of them.

    Raises:
        ValueError: The arguments are not pairs.
    """
    if len(args) == 1 and isinstance(args[0], dict):
        args = tuple(part for item in args[0].items() for part in item)
    if len(args) % 2 or len(args) < 2:
        raise ValueError(
            "Provide an even amount of arguments as option_context(pat, val, pat, val...)."
        )
    pairs = tuple(zip(args[::2], args[1::2], strict=True))
    undo: tuple[tuple[Any, Any], ...] = ()
    try:
        undo = tuple((pat, get_option(pat)) for pat, _ in pairs)
        for pat, value in pairs:
            set_option(pat, value)
        yield
    finally:
        for pat, value in undo:
            set_option(pat, value)


def _tree() -> dict[str, Any]:
    """The option names as nested dicts, a name's last part holding its full name."""
    root: dict[str, Any] = {}
    for key in _REGISTERED:
        *path, last = key.split(".")
        node = root
        for part in path:
            node = node.setdefault(part, {})
        node[last] = key
    return root


class _Options:
    """`options`, the options as attributes, `options.display.max_rows` for one."""

    def __init__(self, tree: dict[str, Any], prefix: str = "") -> None:
        object.__setattr__(self, "_tree", tree)
        object.__setattr__(self, "_prefix", prefix)

    def __getattr__(self, name: str) -> Any:
        tree = object.__getattribute__(self, "_tree")
        if name not in tree:
            raise OptionError("No such option")
        found = tree[name]
        if isinstance(found, dict):
            return _Options(found, f"{object.__getattribute__(self, '_prefix')}{name}.")
        return get_option(found)

    def __setattr__(self, name: str, value: Any) -> None:
        found = object.__getattribute__(self, "_tree").get(name)
        if not isinstance(found, str):
            raise OptionError("You can only set the value of existing options")
        set_option(found, value)

    def __dir__(self) -> list[str]:
        return list(object.__getattribute__(self, "_tree"))


options = _Options(_tree())
"""Every option as an attribute, read and set the way `get_option` and `set_option` do."""
