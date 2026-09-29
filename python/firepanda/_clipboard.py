"""`to_clipboard` and `read_clipboard`, through the system's own copy and paste commands.

pandas carries a copy of pyperclip, which finds a way to reach the clipboard
on each system: `pbcopy` and `pbpaste` on macOS, `wl-copy`, `xsel`, `xclip` or
Klipper on Linux with a display, `clip.exe` and PowerShell under WSL, and the
`/dev/clipboard` file under Cygwin. This finds the same commands in the same
order. On Windows pyperclip calls the Win32 clipboard functions itself, and
this uses `clip.exe` and PowerShell there too, which reach the same clipboard.
With none of them there, pandas raises pyperclip's error, and so does this.

`to_clipboard` writes what `to_csv` writes, tab separated by default, and
`read_clipboard` reads the text back with `read_csv`, guessing tabs the way
pandas guesses them.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import warnings
from collections.abc import Callable
from io import StringIO
from typing import Any

from ._pandas import NO_DEFAULT
from .errors import InvalidArgumentError, PyperclipException

_MISSING = """
    Pyperclip could not find a copy/paste mechanism for your system.
    For more information, please visit
    https://pyperclip.readthedocs.io/en/latest/index.html#not-implemented-error
    """
"""pyperclip's message when a system has no clipboard it can reach, spaces and all."""


def _text(value: Any) -> str:
    """A value as the text copied, which pyperclip takes only from text, numbers and flags."""
    if not isinstance(value, (str, int, float, bool)):
        raise PyperclipException(
            "only str, int, float, and bool values can be copied to the clipboard,"
            f" not {type(value).__name__}"
        )
    return str(value)


def _piped(command: list[str], text: str) -> None:
    with subprocess.Popen(command, stdin=subprocess.PIPE, close_fds=True) as process:
        process.communicate(input=text.encode("utf-8"))


def _read(command: list[str]) -> str:
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True
    ) as process:
        return process.communicate()[0].decode("utf-8")


def _klipper_paste() -> str:
    text = _read(["qdbus", "org.kde.klipper", "/klipper", "getClipboardContents"])
    return text[:-1] if text.endswith("\n") else text


def _dev_copy(text: str) -> None:
    with open("/dev/clipboard", "w", encoding="utf-8") as handle:
        handle.write(text)


def _dev_paste() -> str:
    with open("/dev/clipboard", encoding="utf-8") as handle:
        return handle.read()


def _windows_paste() -> str:
    text = _read(["powershell.exe", "-command", "Get-Clipboard"])
    return text[:-2] if text.endswith("\r\n") else text


def _mechanism() -> tuple[Callable[[str], None], Callable[[], str]]:
    """The copy and the paste this system has, found in pyperclip's order."""
    system = platform.system()
    if "cygwin" in system.lower() and os.path.exists("/dev/clipboard"):
        return _dev_copy, _dev_paste
    if (
        os.name == "nt"
        or system == "Windows"
        or (system == "Linux" and shutil.which("wslconfig.exe"))
    ):
        return (lambda text: _piped(["clip.exe"], text)), _windows_paste
    if system == "Darwin":
        return (lambda text: _piped(["pbcopy", "w"], text)), lambda: _read(["pbpaste", "r"])
    if os.getenv("DISPLAY"):
        if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
            return (lambda text: _piped(["wl-copy"], text)), lambda: _read(["wl-paste", "-n"])
        if shutil.which("xsel"):
            return (lambda text: _piped(["xsel", "-b", "-i"], text)), lambda: _read(
                ["xsel", "-b", "-o"]
            )
        if shutil.which("xclip"):
            return (lambda text: _piped(["xclip", "-selection", "c"], text)), lambda: _read(
                ["xclip", "-selection", "c", "-o"]
            )
        if shutil.which("klipper") and shutil.which("qdbus"):
            return (
                lambda text: _piped(
                    ["qdbus", "org.kde.klipper", "/klipper", "setClipboardContents", text], ""
                )
            ), _klipper_paste
    raise PyperclipException(_MISSING)


def clipboard_set(value: Any) -> None:
    """Puts text on the system clipboard."""
    text = _text(value)
    _mechanism()[0](text)


def clipboard_get() -> str:
    """The text on the system clipboard."""
    return _mechanism()[1]()


def read_clipboard(sep: str = r"\s+", dtype_backend: Any = NO_DEFAULT, **kwargs: Any) -> Any:
    """A frame of the text on the clipboard, read by `read_csv` as pandas reads it.

    Text copied from a spreadsheet is tab separated, so when the first lines
    all hold the same number of tabs, and more than none, the separator is a
    tab and the leading blank columns are the row labels, as pandas guesses.

    Raises:
        NotImplementedError: For an encoding other than UTF-8, as in pandas.
        ValueError: For a separator that is not text.
    """
    from ._pandas import _backend, read_csv

    encoding = kwargs.pop("encoding", "utf-8")
    if encoding is not None and encoding.lower().replace("-", "") != "utf8":
        raise NotImplementedError("reading from clipboard only supports utf-8 encoding")
    if dtype_backend is not NO_DEFAULT:
        _backend(dtype_backend)
    text = clipboard_get()
    lines = text[:10000].split("\n")[:-1][:10]
    counts = {line.lstrip(" ").count("\t") for line in lines}
    if len(lines) > 1 and len(counts) == 1 and counts.pop() != 0:
        sep = "\t"
        index_length = len(lines[0]) - len(lines[0].lstrip(" \t"))
        if index_length != 0:
            kwargs.setdefault("index_col", list(range(index_length)))
    elif not isinstance(sep, str):
        raise InvalidArgumentError(f"{sep=} must be a string")
    if len(sep) > 1 and kwargs.get("engine") is None:
        kwargs["engine"] = "python"
    elif len(sep) > 1 and kwargs.get("engine") == "c":
        warnings.warn(
            "read_clipboard with regex separator does not work properly with c engine.",
            stacklevel=2,
        )
    return read_csv(StringIO(text), sep=sep, dtype_backend=dtype_backend, **kwargs)


def to_clipboard(
    obj: Any, excel: bool | None = True, sep: str | None = None, **kwargs: Any
) -> None:
    """Puts a frame or a column on the clipboard, as `to_csv` writes it or as it prints.

    Raises:
        ValueError: For an encoding other than UTF-8, as in pandas.
    """
    from ._config import option_context
    from ._frame import DataFrame

    encoding = kwargs.pop("encoding", "utf-8")
    if encoding is not None and encoding.lower().replace("-", "") != "utf8":
        raise InvalidArgumentError("clipboard only supports utf-8 encoding")
    if excel is None:
        excel = True
    if excel:
        try:
            buffer = StringIO()
            obj.to_csv(buffer, sep="\t" if sep is None else sep, encoding="utf-8", **kwargs)
            clipboard_set(buffer.getvalue())
            return
        except TypeError:
            warnings.warn(
                "to_clipboard in excel mode requires a single character separator.", stacklevel=3
            )
    elif sep is not None:
        warnings.warn("to_clipboard with excel=False ignores the sep argument.", stacklevel=3)
    if isinstance(obj, DataFrame):
        with option_context("display.max_colwidth", None):
            text = obj.to_string(**kwargs)
    else:
        text = str(obj)
    clipboard_set(text)
