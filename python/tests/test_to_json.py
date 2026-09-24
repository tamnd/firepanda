"""`to_json` on a frame and a column, checked against pandas.

pandas writes floats with its own encoder, which keeps at most
`double_precision` digits after the point and rounds them its own way, so the
text is compared character for character. Instants are counts since the epoch
unless `date_format="iso"` asks for text, and every orient shapes the same
values differently.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def mixed(m: ModuleType) -> Any:
    """Floats, integers, text with quotes and slashes, flags and gaps."""
    return m.DataFrame(
        {
            "f": [1 / 3, None, 1e20, -0.5, 5e-11, 123456789.123456789, float("inf"), 2.0],
            "i": [1, 2, 3, 4, -5, 6, 7, 8],
            "s": ["a", 'q"x', None, "a/b\\c", "tab\there", "café", "\U0001f600", "\x01"],
            "b": [True, False, True, False, True, False, True, False],
        },
        index=[10, 11, 12, 13, 14, 15, 16, 17],
    )


def timed(m: ModuleType) -> Any:
    """Instants, instants with a zone, and spans."""
    instants = m.to_datetime(m.Series(["2020-01-01 00:00:00.000000", "2020-01-02 03:04:05.123456"]))
    zoned = m.to_datetime(m.Series(["2020-01-01 00:00:00", "2020-06-01 12:00:00"]))
    return m.DataFrame(
        {
            "d": instants,
            "z": zoned.dt.tz_localize("US/Eastern"),
            "t": instants - m.Timestamp("2019-12-31 23:00:00"),
        }
    )


def labelled(m: ModuleType) -> Any:
    """A column with a name and text labels."""
    return m.Series([1.5, None, 3.0], index=["a", "b", "c"], name="v")


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: mixed(m).to_json(),
    lambda m: mixed(m).to_json(orient="index"),
    lambda m: mixed(m).to_json(orient="records"),
    lambda m: mixed(m).to_json(orient="split"),
    lambda m: mixed(m).to_json(orient="split", index=False),
    lambda m: mixed(m).to_json(orient="values"),
    lambda m: mixed(m).to_json(orient="records", lines=True),
    lambda m: mixed(m).to_json(double_precision=3),
    lambda m: mixed(m).to_json(double_precision=0),
    lambda m: mixed(m).to_json(double_precision=15),
    lambda m: mixed(m).to_json(force_ascii=False),
    lambda m: mixed(m).to_json(indent=2),
    lambda m: mixed(m).to_json(orient="split", indent=4),
    lambda m: m.DataFrame().to_json(),
    lambda m: m.DataFrame().to_json(indent=2),
    lambda m: m.DataFrame({"x": [1.5, 2.25]}, index=[0.5, 1.0]).to_json(),
    lambda m: m.DataFrame({"x": ["a", "b"]}).astype("category").to_json(),
    lambda m: timed(m).to_json(date_format="iso"),
    lambda m: timed(m).to_json(date_format="iso", date_unit="s"),
    lambda m: timed(m).to_json(date_format="iso", date_unit="us"),
    lambda m: timed(m).to_json(date_format="iso", date_unit="ns", orient="records"),
    lambda m: timed(m).to_json(),
    lambda m: timed(m).to_json(date_unit="s"),
    lambda m: timed(m).to_json(date_unit="ns", orient="values"),
    lambda m: timed(m).set_index("d").to_json(),
    lambda m: timed(m).set_index("d").to_json(date_format="iso"),
    lambda m: labelled(m).to_json(),
    lambda m: labelled(m).to_json(orient="records"),
    lambda m: labelled(m).to_json(orient="split"),
    lambda m: labelled(m).to_json(orient="split", index=False),
    lambda m: labelled(m).to_json(orient="values"),
    lambda m: labelled(m).to_json(orient="columns"),
    lambda m: labelled(m).to_json(orient="records", lines=True),
    lambda m: labelled(m).to_json(indent=1),
    lambda m: m.Series([1, 2]).to_json(orient="split"),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_text_is_pandas_text(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every orient, precision, escape, indent, and date shape."""
    import pandas as pd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert build(firepanda) == build(pd)


def test_epoch_dates_warn_as_pandas_warns(firepanda: ModuleType) -> None:
    """Epoch dates, asked for or taken by default, warn with pandas' words."""
    from firepanda.errors import Pandas4Warning

    frame = firepanda.DataFrame({"d": firepanda.to_datetime(firepanda.Series(["2020-01-01"]))})
    with pytest.warns(Pandas4Warning, match="The default 'epoch' date format"):
        frame.to_json()
    with pytest.warns(Pandas4Warning, match="^'epoch' date format is deprecated"):
        firepanda.DataFrame({"x": [1]}).to_json(date_format="epoch")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        firepanda.DataFrame({"x": [1]}).to_json()
        frame.to_json(date_format="iso")


def test_files_and_handles(firepanda: ModuleType, tmp_path: Path) -> None:
    """A path, compressed by its name, a handle, and appending lines."""
    import pandas as pd

    for name in ("plain.json", "packed.json.gz"):
        mine, theirs = tmp_path / f"mine-{name}", tmp_path / f"theirs-{name}"
        assert mixed(firepanda).to_json(mine) is None
        mixed(pd).to_json(theirs)
        read = gzip.decompress if name.endswith(".gz") else (lambda raw: raw)
        assert read(mine.read_bytes()) == read(theirs.read_bytes())
    path = tmp_path / "lines.json"
    frame = firepanda.DataFrame({"x": [1, 2]})
    frame.to_json(path, orient="records", lines=True)
    frame.to_json(path, orient="records", lines=True, mode="a")
    assert path.read_text() == '{"x":1}\n{"x":2}\n' * 2
    text = io.StringIO()
    frame.to_json(text)
    assert text.getvalue() == '{"x":{"0":1,"1":2}}'


@pytest.mark.parametrize(
    "build",
    [
        lambda m: m.DataFrame({"x": [1]}).to_json(orient="records", index=True),
        lambda m: m.DataFrame({"x": [1]}).to_json(orient="index", index=False),
        lambda m: m.DataFrame({"x": [1]}).to_json(lines=True),
        lambda m: m.DataFrame({"x": [1]}).to_json(mode="x"),
        lambda m: m.DataFrame({"x": [1]}).to_json(mode="a"),
        lambda m: m.DataFrame({"x": [1]}).to_json(date_unit="h"),
        lambda m: m.DataFrame({"x": [1]}).to_json(orient="bogus"),
        lambda m: m.DataFrame({"x": [1, 2]}, index=[0, 0]).to_json(),
        lambda m: m.Series([1, 2], index=[0, 0]).to_json(),
    ],
)
def test_mistakes_fail_as_pandas_fails(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Index flags and lines with the wrong orient, bad modes and units, repeated labels."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert type(mine.value) is type(theirs.value)
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize("precision", [0, 1, 5, 10, 15])
def test_many_floats_round_as_pandas_rounds(firepanda: ModuleType, precision: int) -> None:
    """Floats across every magnitude, halves included, keep pandas' digits."""
    import random

    import pandas as pd

    draw = random.Random(precision)
    values = [draw.uniform(-1, 1) * 10.0 ** draw.randint(-18, 18) for _ in range(2000)]
    values += [0.5, 1.5, 2.5, -0.5, 0.125, 0.0, -0.0, 1e16, 1e16 - 1, 9999999999999998.0]
    mine = firepanda.Series(values).to_json(orient="values", double_precision=precision)
    assert mine == pd.Series(values).to_json(orient="values", double_precision=precision)
