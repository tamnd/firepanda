"""`to_csv` on a frame and a column, checked against pandas.

pandas writes floats the shortest way that reads back the same, missing values
as `na_rep`, and instants with one shape for the whole column: the date alone
when every instant is at midnight, and otherwise as many fraction digits as the
finest one needs. The text is the same whether it is answered or written to a
file, compressed or not.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import importlib.util
import io
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def mixed(m: ModuleType) -> Any:
    """Floats, integers, text with commas and quotes, flags, gaps and instants."""
    return m.DataFrame(
        {
            "f": [1.0, None, 1e20, 0.1],
            "i": [1, 2, 3, 4],
            "s": ["a", "b,c", None, 'q"x'],
            "b": [True, False, True, False],
            "d": m.to_datetime(["2020-01-01", "2020-01-02", None, "2020-01-04"]),
        }
    )


def instants(m: ModuleType, values: list[Any]) -> Any:
    """A frame with one column of instants."""
    return m.DataFrame({"t": m.to_datetime(values)})


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: mixed(m).to_csv(),
    lambda m: mixed(m).to_csv(index=False, na_rep="NA", sep=";"),
    lambda m: mixed(m).to_csv(float_format="%.2f", decimal=","),
    lambda m: mixed(m).to_csv(decimal=",", sep=";"),
    lambda m: mixed(m).to_csv(float_format=lambda value: f"<{value}>"),
    lambda m: mixed(m).to_csv(quoting=csv.QUOTE_NONNUMERIC),
    lambda m: mixed(m).to_csv(quoting=csv.QUOTE_NONNUMERIC, float_format="%.1f"),
    lambda m: mixed(m).to_csv(quoting=csv.QUOTE_ALL, index_label="k"),
    lambda m: mixed(m).to_csv(quoting=csv.QUOTE_NONE, escapechar="\\"),
    lambda m: mixed(m).to_csv(doublequote=False, escapechar="\\"),
    lambda m: mixed(m).to_csv(quotechar="'", lineterminator="\r\n"),
    lambda m: mixed(m).to_csv(header=["A", "B", "C", "D", "E"]),
    lambda m: mixed(m).to_csv(header=False),
    lambda m: mixed(m).to_csv(columns=["s", "f"]),
    lambda m: mixed(m).to_csv(index_label=False),
    lambda m: mixed(m).to_csv(index_label=["k"]),
    lambda m: mixed(m).to_csv(date_format="%d/%m/%Y"),
    lambda m: mixed(m).rename_axis("ix").to_csv(),
    lambda m: mixed(m).set_index("s").to_csv(),
    lambda m: mixed(m).set_index("d").to_csv(),
    lambda m: m.DataFrame({"x": [1.5]}, index=m.Index([2.5])).to_csv(float_format="%.3f"),
    lambda m: m.DataFrame({"x": [0.1 + 0.2, 1 / 3, 1e-7, 123456789.123, float("inf")]}).to_csv(),
    lambda m: m.DataFrame({"x": [1.25, 3.0, 0.1]}, dtype="float32").to_csv(),
    lambda m: m.DataFrame({"x": [1, -2]}, dtype="int32").to_csv(),
    lambda m: m.DataFrame({"c": ["x", "y", None]}).astype("category").to_csv(),
    lambda m: m.DataFrame({"a": [], "b": []}).to_csv(),
    lambda m: m.DataFrame().to_csv(),
    lambda m: instants(m, ["2020-01-01 10:00:00", "2020-01-02 00:00:00"]).to_csv(),
    lambda m: instants(m, ["2020-01-01 10:00:00.000", "2020-01-02 00:00:00.500"]).to_csv(),
    lambda m: instants(m, ["2020-01-01 00:00:00.000000", "2020-01-02 00:00:00.000001"]).to_csv(),
    lambda m: instants(m, ["2020-01-01 00:00", "2020-01-02 03:00"]).to_csv(date_format="%H"),
    lambda m: m.DataFrame(
        {
            "t": m.Series(m.to_datetime(["2020-01-01 00:00", "2020-01-02 03:00"])).dt.tz_localize(
                "Europe/Paris"
            )
        }
    ).to_csv(),
    lambda m: m.DataFrame(
        {"t": [m.Timedelta(hours=1), None, m.Timedelta(days=2, microseconds=5)]}
    ).to_csv(),
    lambda m: m.DataFrame({"t": [m.Timedelta(0), m.Timedelta(days=-1, hours=3)]}).to_csv(),
    lambda m: m.Series([1, 2]).to_csv(),
    lambda m: m.Series([1.5, None], name="x").to_csv(index=False),
    lambda m: m.Series([1, 2], name="n").to_csv(header=False),
    lambda m: m.Series(["a", "b"], index=m.Index([3, 4], name="k"), name="v").to_csv(),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_text_is_pandas_text(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every type, quoting, separators, headers, index labels and float formats."""
    import pandas as pd

    assert build(firepanda) == build(pd)


@pytest.mark.parametrize(
    ("name", "read"),
    [
        ("plain.csv", lambda path: path.read_text()),
        ("packed.csv.gz", lambda path: gzip.decompress(path.read_bytes()).decode()),
        ("packed.csv.bz2", lambda path: bz2.decompress(path.read_bytes()).decode()),
        (
            "packed.csv.zip",
            lambda path: zipfile.ZipFile(path).read("packed.csv").decode(),
        ),
    ],
)
def test_a_file_holds_the_same_text(
    firepanda: ModuleType, tmp_path: Path, name: str, read: Callable[[Path], str]
) -> None:
    """A path, compressed by its name, holds what pandas writes and answers None."""
    import pandas as pd

    mine, theirs = tmp_path / "mine", tmp_path / "theirs"
    mine.mkdir()
    theirs.mkdir()
    assert mixed(firepanda).to_csv(mine / name) is None
    mixed(pd).to_csv(theirs / name)
    assert read(mine / name) == read(theirs / name)


def test_appending_and_handles(firepanda: ModuleType, tmp_path: Path) -> None:
    """`mode="a"` adds to a file, and a text or byte handle gets the text."""
    path = tmp_path / "out.csv"
    frame = firepanda.DataFrame({"x": [1, 2]})
    frame.to_csv(path, index=False)
    frame.to_csv(path, mode="a", header=False, index=False)
    assert path.read_text() == "x\n1\n2\n1\n2\n"
    text, raw = io.StringIO(), io.BytesIO()
    frame.to_csv(text)
    frame.to_csv(raw, encoding="utf-8")
    assert text.getvalue() == raw.getvalue().decode() == ",x\n0,1\n1,2\n"


@pytest.mark.parametrize(
    "build",
    [
        lambda m: m.DataFrame({"x": [1]}).to_csv(header=["a", "b"]),
        lambda m: m.DataFrame({"x": [1]}).to_csv(sep="ab"),
        lambda m: m.DataFrame({"x": ["a,b"]}).to_csv(quoting=csv.QUOTE_NONE),
    ],
)
def test_mistakes_fail_as_pandas_fails(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Too many aliases, a long separator, and a comma with nothing to escape it."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert type(mine.value) is type(theirs.value)
    assert str(mine.value) == str(theirs.value)
