"""`display.precision`, `display.float_format`, `set_eng_float_format` and
`show_versions`, compared with pandas.

Each case prints the same frame or column in both libraries under the same
options, set with `option_context` so nothing is left set after the case.
"""

from __future__ import annotations

import importlib.util
import io
import json
from collections.abc import Callable
from contextlib import nullcontext, redirect_stdout
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def mixed(m: ModuleType) -> Any:
    return m.DataFrame(
        {"a": [1.123456789, 22.5, 1234567.0], "b": [1, 2, 3], "c": [0.5, None, -2.25]}
    )


def tiny(m: ModuleType) -> Any:
    return m.DataFrame({"a": [0.000012345, 1.5, 250.0]})


def plain(m: ModuleType) -> Any:
    return m.DataFrame({"a": [1.0, 2.5, -3.125], "s": ["x", "y", "z"]})


FRAMES: dict[str, Callable[[ModuleType], Any]] = {
    "mixed": mixed,
    "tiny": tiny,
    "plain": plain,
}

OPTIONS: dict[str, tuple[Any, ...]] = {
    "the defaults": (),
    "precision 2": ("display.precision", 2),
    "precision 0": ("display.precision", 0),
    "precision 9": ("display.precision", 9),
    "a float format": ("display.float_format", lambda v: f"<{v:.1f}>"),
    "a format method": ("display.float_format", "{:,.3f}".format),
    "both": ("display.precision", 3, "display.float_format", lambda v: f"{v:.2e}"),
}


def printed(m: ModuleType, frame: str, options: str) -> tuple[str, ...]:
    """The repr of the frame and its first column, and `to_string` of both, under the options."""
    shown = FRAMES[frame](m)
    with m.option_context(*OPTIONS[options]) if OPTIONS[options] else nullcontext():
        return (
            repr(shown),
            repr(shown.iloc[:, 0]),
            shown.to_string(),
            shown.iloc[:, 0].to_string(),
        )


@needs_pandas
@pytest.mark.parametrize("options", list(OPTIONS))
@pytest.mark.parametrize("frame", list(FRAMES))
def test_floats_print_by_the_options(firepanda: ModuleType, frame: str, options: str) -> None:
    """A repr and `to_string` read the precision and the float format as pandas does."""
    import pandas as pd

    assert printed(firepanda, frame, options) == printed(pd, frame, options)


@needs_pandas
def test_a_float_format_given_wins_over_the_option(firepanda: ModuleType) -> None:
    """`to_string(float_format=...)` is used rather than `display.float_format`."""
    import pandas as pd

    def shown(m: ModuleType) -> str:
        with m.option_context("display.float_format", lambda v: "no"):
            return mixed(m).to_string(float_format=lambda v: f"{v:.2f}")

    assert shown(firepanda) == shown(pd)


ENGINEERING: dict[str, tuple[Any, ...]] = {
    "the defaults": (),
    "accuracy 1": (1,),
    "with prefixes": (2, True),
    "accuracy 0 with prefixes": (0, True),
}


@needs_pandas
@pytest.mark.parametrize("name", list(ENGINEERING))
def test_engineering_floats_print_as_pandas_prints_them(firepanda: ModuleType, name: str) -> None:
    """`set_eng_float_format` sets `display.float_format`, which every repr then reads."""
    import pandas as pd

    def shown(m: ModuleType) -> tuple[str, ...]:
        try:
            m.set_eng_float_format(*ENGINEERING[name])
            frame = m.DataFrame({"a": [1.5e-7, 0.0, -2500.0, 3.25e9, 1e30], "b": [1, 2, 3, 4, 5]})
            return repr(frame), repr(frame["a"]), repr(m.get_option("display.float_format")(1e6))
        finally:
            m.reset_option("display.float_format")

    assert shown(firepanda) == shown(pd)


@needs_pandas
@pytest.mark.parametrize("value", [0, 1_000_000, "-1e-6", 1e-30, 1e40, float("nan"), float("inf")])
def test_the_engineering_formatter_writes_what_pandas_writes(
    firepanda: ModuleType, value: Any
) -> None:
    """The formatter `set_eng_float_format` installs, on numbers and text, at each accuracy."""
    import pandas as pd

    def shown(m: ModuleType) -> list[str]:
        written = []
        for accuracy, prefix in ((None, False), (0, True), (1, True), (2, False)):
            m.set_eng_float_format(accuracy, prefix)
            written.append(m.get_option("display.float_format")(value))
        m.reset_option("display.float_format")
        return written

    assert shown(firepanda) == shown(pd)


@needs_pandas
def test_show_versions_reports_the_same_system(firepanda: ModuleType) -> None:
    """The table and the JSON hold pandas' system keys, and the dependencies name firepanda."""
    import pandas as pd

    def report(m: ModuleType) -> dict[str, Any]:
        out = io.StringIO()
        with redirect_stdout(out):
            m.show_versions(as_json=True)
        return json.loads(out.getvalue())

    mine, theirs = report(firepanda), report(pd)
    assert list(mine) == list(theirs) == ["system", "dependencies"]
    assert list(mine["system"]) == list(theirs["system"])
    del mine["system"]["commit"], theirs["system"]["commit"]
    assert mine["system"] == theirs["system"]
    assert mine["dependencies"]["firepanda"] == firepanda.__version__
    assert mine["dependencies"]["numpy"] == theirs["dependencies"]["numpy"]


def test_show_versions_prints_a_table_or_writes_a_file(firepanda: ModuleType, tmp_path) -> None:
    """The table starts as pandas' does, and a path writes the JSON there instead."""
    out = io.StringIO()
    with redirect_stdout(out):
        firepanda.show_versions()
    lines = out.getvalue().splitlines()
    assert lines[:3] == ["", "INSTALLED VERSIONS", "------------------"]
    assert any(line.startswith("firepanda") for line in lines)
    target = tmp_path / "versions.json"
    with redirect_stdout(io.StringIO()) as quiet:
        firepanda.show_versions(as_json=str(target))
    assert quiet.getvalue() == ""
    assert json.loads(target.read_text())["dependencies"]["firepanda"] == firepanda.__version__
