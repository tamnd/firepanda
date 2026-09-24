"""`to_string` on a frame and a column, checked against pandas.

pandas writes a frame as text with a formatter of its own: floats get six
digits after the point with the zeros trimmed across the column, or scientific
form when a value is tiny or huge, instants are dates alone when all are at
midnight, and every column is padded to its widest cell. The text is compared
character for character.
"""

from __future__ import annotations

import importlib.util
import io
import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def mixed(m: ModuleType) -> Any:
    """Floats with gaps, integers, text with gaps, and flags."""
    return m.DataFrame(
        {
            "f": [1.5, None, -2.25, 1000.0],
            "i": [1, 22, -333, 4],
            "s": ["a", None, "long text", "tab\there"],
            "b": [True, False, True, False],
        }
    )


def floats(m: ModuleType, values: list[Any]) -> Any:
    """One float column."""
    return m.DataFrame({"x": values})


def timed(m: ModuleType, texts: list[Any]) -> Any:
    """Instants read from text, and the spans from the first of them."""
    instants = m.to_datetime(m.Series(texts))
    return m.DataFrame({"when": instants, "since": instants - instants.min()})


def labelled(m: ModuleType) -> Any:
    """A column with a name and text labels under a named index."""
    index = m.Index(["a", "bb", "ccc"], name="key")
    return m.Series([1.5, None, 3.0], index=index, name="v")


MIDNIGHTS = ["2020-01-01 00:00:00", "2020-01-03 00:00:00", None]
SECONDS = ["2020-01-01 00:00:00", "2020-01-02 03:04:05", None]
MILLIS = ["2020-01-01 00:00:00.000000", "2020-01-02 03:04:05.250000"]
MICROS = ["2020-01-01 00:00:00.000000", "2020-01-02 03:04:05.123456"]

BUILDS: list[Callable[[Any], Any]] = [
    lambda m: mixed(m).to_string(),
    lambda m: mixed(m).to_string(index=False),
    lambda m: mixed(m).to_string(header=False),
    lambda m: mixed(m).to_string(header=["F", "I", "S", "B"]),
    lambda m: mixed(m).to_string(columns=["s", "f"]),
    lambda m: mixed(m).to_string(na_rep="-"),
    lambda m: mixed(m).to_string(float_format="%.3f"),
    lambda m: mixed(m).to_string(float_format="{:.1f}".format),
    lambda m: mixed(m).to_string(float_format="{:,.2f}"),
    lambda m: mixed(m).to_string(decimal=","),
    lambda m: mixed(m).to_string(justify="left"),
    lambda m: mixed(m).to_string(col_space=12),
    lambda m: mixed(m).to_string(col_space={"f": 10, "s": 3}),
    lambda m: mixed(m).to_string(col_space=[1, 2, 3, 20]),
    lambda m: mixed(m).to_string(formatters={"i": lambda v: f"<{v}>"}),
    lambda m: mixed(m).to_string(formatters=[str, str, str.upper, str]),
    lambda m: mixed(m).to_string(show_dimensions=True),
    lambda m: mixed(m).to_string(max_colwidth=5),
    lambda m: mixed(m).set_index("s").to_string(),
    lambda m: mixed(m).set_index("s").to_string(index_names=False),
    lambda m: mixed(m).set_index("s").to_string(header=False),
    lambda m: mixed(m).set_index("i").to_string(),
    lambda m: mixed(m).set_index("f").to_string(),
    lambda m: floats(m, [1.0, 2.0, 3.0]).to_string(),
    lambda m: floats(m, [0.1, 0.25, 1 / 3]).to_string(),
    lambda m: floats(m, [1e-9, 1.0, None]).to_string(),
    lambda m: floats(m, [1e7, 1.5, 2.125]).to_string(),
    lambda m: floats(m, [1e7, 2.0]).to_string(),
    lambda m: floats(m, [123456789.123, 1.0]).to_string(),
    lambda m: floats(m, [float("inf"), -1.5, float("-inf")]).to_string(),
    lambda m: floats(m, [float("nan"), float("nan")]).to_string(),
    lambda m: floats(m, [0.0, -0.0, 5e-7]).to_string(index=False),
    lambda m: timed(m, MIDNIGHTS).to_string(),
    lambda m: timed(m, SECONDS).to_string(),
    lambda m: timed(m, MILLIS).to_string(),
    lambda m: timed(m, MICROS).to_string(index=False),
    lambda m: timed(m, SECONDS).set_index("when").to_string(),
    lambda m: timed(m, MIDNIGHTS[:2]).set_index("when").to_string(),
    lambda m: m.DataFrame({"a": list(range(12))}).to_string(),
    lambda m: m.DataFrame({"a_long_header": [1, 2], "b": [-0.5, 1e-8]}).to_string(),
    lambda m: m.DataFrame({"c": ["x", "yy", "x"]}).astype("category").to_string(),
    lambda m: m.DataFrame({"x": [1.5, 2.0]}, index=["one", "three"]).to_string(),
    lambda m: m.DataFrame({"x": [1, 2]}, index=[0.5, -10.25]).to_string(),
    lambda m: m.Series([-1.5, 2.0, None]).to_string(),
    lambda m: m.Series(["c", "a"], dtype="category").to_string(dtype=True),
    lambda m: m.Series(["c", "a"], dtype="category").to_string(),
    lambda m: m.Series([f"level {i}" for i in range(20)], dtype="category").to_string(),
    lambda m: m.Series([f"category number {i}" for i in range(6)], dtype="category").to_string(),
    lambda m: m.Series(list("cab"), dtype="category").cat.as_ordered().to_string(),
    lambda m: m.DataFrame({"a": [], "b": []}).to_string(),
    lambda m: m.DataFrame({"a": [], "b": []}).to_string(show_dimensions=True),
    lambda m: labelled(m).to_string(),
    lambda m: labelled(m).to_string(name=True, dtype=True, length=True),
    lambda m: labelled(m).to_string(index=False),
    lambda m: labelled(m).to_string(header=False),
    lambda m: labelled(m).to_string(na_rep="?", float_format="%.2f"),
    lambda m: labelled(m).to_string(float_format=lambda v: f"[{v}]"),
    lambda m: m.Series(["x", None, "a much longer piece of text"], name="s").to_string(),
    lambda m: m.Series(["x" * 60, "y"]).to_string(),
    lambda m: m.Series(["x", "y"], name="s").to_string(dtype=True, name=True),
    lambda m: m.Series([True, False]).to_string(),
    lambda m: m.Series(list(range(-3, 9))).to_string(),
    lambda m: m.Series([1.25, 3.5]).to_string(index=False),
    lambda m: m.Series([], dtype="float64").to_string(),
    lambda m: m.Series([], dtype="float64", name="e").to_string(name=True, dtype=True),
    lambda m: timed(m, SECONDS)["since"].to_string(),
    lambda m: timed(m, MIDNIGHTS)["since"].to_string(),
    lambda m: timed(m, MIDNIGHTS)["when"].to_string(dtype=True),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_text_is_pandas_text(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every type, option, label and width."""
    import pandas as pd

    assert build(firepanda) == build(pd)


def test_the_text_goes_to_a_buffer_or_a_path(firepanda: ModuleType, tmp_path: Path) -> None:
    """A buffer is written to, a path is created, and both answer None."""
    import pandas as pd

    for m in (firepanda, pd):
        buffer = io.StringIO()
        assert mixed(m).to_string(buffer) is None
        assert labelled(m).to_string(buf=buffer) is None
        (tmp_path / m.__name__).write_text(buffer.getvalue())
        assert mixed(m).to_string(tmp_path / f"{m.__name__}.txt") is None
    assert (tmp_path / "firepanda").read_text() == (tmp_path / "pandas").read_text()
    assert (tmp_path / "firepanda.txt").read_text() == (tmp_path / "pandas.txt").read_text()


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: mixed(m).to_string(header=["x"]),
    lambda m: mixed(m).to_string(formatters=[str]),
    lambda m: mixed(m).to_string(col_space={"nope": 3}),
    lambda m: mixed(m).to_string(col_space=[1]),
    lambda m: mixed(m).to_string(float_format="{:q}"),
    lambda m: mixed(m).to_string(float_format=3),
    lambda m: mixed(m).to_string(encoding="utf-8"),
    lambda m: mixed(m).to_string(buf=3),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_mistakes_raise_as_pandas_raises(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The same error type and words as pandas."""
    import pandas as pd

    with pytest.raises(Exception) as expected:
        build(pd)
    with pytest.raises(type(expected.value), match="^" + re.escape(str(expected.value))):
        build(firepanda)


@pytest.mark.parametrize("keyword", ["max_rows", "max_cols", "line_width", "min_rows"])
def test_the_limits_are_refused_by_name(firepanda: ModuleType, keyword: str) -> None:
    """A limit would leave rows or columns out, which is not supported yet."""
    with pytest.raises(NotImplementedError, match=keyword):
        mixed(firepanda).to_string(**{keyword: 2})
