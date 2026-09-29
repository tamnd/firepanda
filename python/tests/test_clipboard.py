"""`to_clipboard` and `read_clipboard` against pandas, over an in-memory clipboard.

Both libraries have their clipboard swapped for a list here, so the tests
never touch the clipboard of the machine they run on.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp
from firepanda import _clipboard

pd = pytest.importorskip("pandas")
clipboard = pytest.importorskip("pandas.io.clipboard")

BOARD: list[str] = [""]


def put(text: str) -> None:
    BOARD[0] = text


def board() -> str:
    return BOARD[0]


@pytest.fixture(autouse=True)
def memory(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for module in (_clipboard, clipboard):
        monkeypatch.setattr(module, "clipboard_set", put)
        monkeypatch.setattr(module, "clipboard_get", board)
    BOARD[0] = ""
    yield


def frame(lib: ModuleType) -> Any:
    return lib.DataFrame({"a": [1, 2], "b": ["x", "long words"]}, index=["p", "q"])


def copied(lib: ModuleType, **options: Any) -> str:
    """What `to_clipboard` leaves on the clipboard for the frame."""
    frame(lib).to_clipboard(**options)
    return board()


def pasted(lib: ModuleType, text: str, **options: Any) -> Any:
    """The frame `read_clipboard` reads from the text, with text spelled `str`."""
    put(text)
    read = lib.read_clipboard(**options)
    return read, [str(kind).replace("string", "str") for kind in read.dtypes]


SPACED = "a b\n1 x\n2 y\n"
TABBED = "a\tb\n1\tx\n2\ty\n"
LABELLED = "\ta\tb\np\t1\tx\nq\t2\ty\n"
LABELLED_SPACED = "  a  b\np  1  x\nq  2  y\n"

CASES: dict[str, Callable[[ModuleType], Any]] = {
    "copy-excel": lambda lib: copied(lib),
    "copy-comma": lambda lib: copied(lib, sep=","),
    "copy-no-index": lambda lib: copied(lib, index=False),
    "copy-plain": lambda lib: copied(lib, excel=False),
    "copy-plain-sep": lambda lib: copied(lib, excel=False, sep=","),
    "copy-excel-none": lambda lib: copied(lib, excel=None),
    "copy-long-sep": lambda lib: copied(lib, sep="::"),
    "copy-encoding": lambda lib: copied(lib, encoding="latin-1"),
    "copy-utf8": lambda lib: copied(lib, encoding="UTF-8"),
    "copy-series": lambda lib: (lib.Series([1, 2], name="n").to_clipboard(), board())[1],
    "copy-series-plain": lambda lib: (
        lib.Series([1.5, 2.5], name="n").to_clipboard(excel=False),
        board(),
    )[1],
    "paste-spaced": lambda lib: pasted(lib, SPACED),
    "paste-tabbed": lambda lib: pasted(lib, TABBED),
    "paste-labelled": lambda lib: pasted(lib, LABELLED),
    "paste-labelled-spaced": lambda lib: pasted(lib, LABELLED_SPACED),
    "paste-comma": lambda lib: pasted(lib, "a,b\n1,x\n2,y\n", sep=","),
    "paste-header-none": lambda lib: pasted(lib, SPACED, header=None),
    "paste-encoding": lambda lib: pasted(lib, SPACED, encoding="latin-1"),
    "paste-bad-sep": lambda lib: pasted(lib, "a;b\n1;2\n", sep=3),
    "paste-bad-backend": lambda lib: pasted(lib, SPACED, dtype_backend="numpi"),
    "round-trip": lambda lib: (frame(lib).to_clipboard(), pasted(lib, board()))[1],
    "round-trip-no-index": lambda lib: (
        frame(lib).to_clipboard(index=False),
        pasted(lib, board()),
    )[1],
}


def mistake(error: Exception) -> str:
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
    ("options", "message"),
    [({"sep": "::"}, "single character"), ({"excel": False, "sep": ","}, "ignores the sep")],
    ids=["long-sep", "plain-sep"],
)
def test_warns_as_pandas(options: dict[str, Any], message: str) -> None:
    with pytest.warns(UserWarning, match=message):
        frame(fp).to_clipboard(**options)


def test_a_value_that_is_not_text_is_refused() -> None:
    with pytest.raises(fp.errors.PyperclipException, match="not list"):
        _clipboard._text([1])


def test_no_mechanism_raises_pyperclips_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_clipboard.platform, "system", lambda: "Plan9")
    monkeypatch.setattr(_clipboard.os, "name", "posix")
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(fp.errors.PyperclipException, match="could not find a copy/paste"):
        _clipboard._mechanism()
