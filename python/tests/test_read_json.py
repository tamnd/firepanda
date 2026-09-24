"""`read_json` against pandas, orient by orient and inference by inference.

pandas lays the decoded JSON out by `orient` and then infers each column:
text that is all numbers becomes numbers, whole floats become integers, and
columns named like dates become instants. The row labels go through the same
inference. Floats come from pandas' own decoder, which is not always the
nearest float, so values are compared exactly.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import random
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def facts(obj: Any) -> Any:
    """What a frame or a column holds, in plain Python, with dtypes by kind."""

    def kind(dtype: Any) -> str:
        printed = str(dtype)
        return "datetime" if printed.startswith("datetime") else printed.replace("string", "str")

    def cells(column: Any) -> list[Any]:
        return [None if value != value else value for value in column.tolist()]

    labels = [str(label) for label in obj.index.tolist()]
    if hasattr(obj, "columns"):
        return (
            [str(name) for name in obj.columns],
            labels,
            kind(obj.index.dtype),
            {str(name): kind(dtype) for name, dtype in zip(obj.columns, obj.dtypes, strict=True)},
            [cells(obj.iloc[:, place]) for place in range(len(obj.columns))],
        )
    return obj.name, labels, kind(obj.index.dtype), kind(obj.dtype), cells(obj)


TEXTS: list[tuple[str, dict[str, Any]]] = [
    ('{"a":{"0":1,"1":2},"b":{"0":"x","1":null}}', {}),
    ('{"a":{"0":1.5,"1":null},"b":{"0":1,"1":null}}', {}),
    ('{"a":{"1":1,"0":2}}', {}),
    ('{"a":{"b":1},"c":{"a":2}}', {}),
    ('{"a":[1,2],"b":[3.5,4]}', {}),
    ('[{"a":1,"b":2.5},{"a":3,"b":null}]', {"orient": "records"}),
    ('[{"b":1},{"a":2}]', {}),
    ('{"columns":["a","b"],"index":[5,6],"data":[[1,"x"],[2,"y"]]}', {"orient": "split"}),
    ('{"columns":["a"],"index":["1","2"],"data":[[1],[2]]}', {"orient": "split"}),
    ('{"columns":["a"],"data":[[1],[2]]}', {"orient": "split"}),
    ('{"y":{"b":1},"x":{"a":2}}', {"orient": "index"}),
    ('{"x":{"a":1.25},"y":{"a":2}}', {"orient": "index"}),
    ('{"a":1}\n{"a":2}\n\n{"a":3}\n', {"lines": True}),
    ('{"a":1}\n{"a":2}\n{"a":3}\n', {"lines": True, "nrows": 2}),
    ('{"a":{"0":true,"1":false}}', {}),
    ('{"a":{"0":true,"1":null}}', {}),
    ('{"a":{"0":"1","1":"2"},"b":{"0":"1.5","1":"2"},"c":{"0":"1","1":null}}', {}),
    ('{"a":{"0":1.0,"1":2.0}}', {}),
    ('{"a":{"0":"1","1":"2"}}', {"dtype": False}),
    ('{"a":{"0":1,"1":2}}', {"dtype": {"a": "float32"}}),
    ('{"a":{"0":1,"1":2}}', {"dtype": "float64"}),
    ('{"a":{"0":1}}', {"convert_axes": False}),
    ('{"date":{"0":1577836800000},"x_at":{"0":1577836800},"v":{"0":1577836800000}}', {}),
    ('{"date":{"0":1577836800000}}', {"keep_default_dates": False}),
    ('{"date":{"0":1577836800000}}', {"convert_dates": False}),
    ('{"v":{"0":1577836800000}}', {"convert_dates": ["v"]}),
    ('{"date":{"0":1577836800,"1":null}}', {}),
    ('{"date":{"0":1577836800000}}', {"date_unit": "ms"}),
    ('{"modified":{"0":"2020-01-01T00:00:00.000"},"v":{"0":"2020-01-01T00:00:00.000"}}', {}),
    ('{"date":{"0":"2020-01-01","1":"x"}}', {}),
    ('{"date":{"0":"1577836800"}}', {}),
    ('{"date":{"0":-5},"timestamp_x":{"0":true}}', {}),
    ('{"a":{"1577836800000":1}}', {}),
    ('{"a":{"2020-01-01T00:00:00.000":1}}', {}),
    ('{"a":{"1.5":1,"2.5":2}}', {}),
    ('{"a":{"x":1,"y":2}}', {}),
    ('{"a":{"0":NaN,"1":Infinity,"2":-Infinity}}', {}),
    ('{"a":{"0":12.5e3,"1":-0.0}}', {}),
    ("{}", {}),
    ("[]", {}),
    ('{"x":1,"y":2}', {"typ": "series"}),
    ('{"0":1577836800000}', {"typ": "series"}),
    ('{"0":"3"}', {"typ": "series"}),
    ('{"name":"v","index":[0,1],"data":[1.5,2]}', {"typ": "series", "orient": "split"}),
    ("[1,2]", {"typ": "series", "orient": "records"}),
    ('{"a":1.5,"b":null}', {"typ": "series"}),
    ('{"a":1}', {"typ": "series", "dtype": False}),
]


@pytest.mark.parametrize(("text", "options"), TEXTS)
def test_frames_and_columns_are_pandas_frames(
    firepanda: ModuleType, text: str, options: dict[str, Any]
) -> None:
    """Every orient, the inference of values and labels, dates, lines and dtypes."""
    import pandas as pd

    mine = firepanda.read_json(io.StringIO(text), **options)
    theirs = pd.read_json(io.StringIO(text), **options)
    assert facts(mine) == facts(theirs)


def test_floats_read_as_pandas_reads_them(firepanda: ModuleType) -> None:
    """Floats in many shapes keep the digits pandas' decoder gives them."""
    import pandas as pd

    draw = random.Random(7)
    texts = []
    for _ in range(3000):
        value = draw.uniform(-1, 1) * 10.0 ** draw.randint(-25, 25)
        texts += [repr(value), f"{value:.10g}", f"{value:.17g}"]
        if abs(value) < 1e15:
            texts.append(f"{value:.4f}")
    text = "[" + ",".join(texts) + "]"
    for precise in (False, True):
        mine = firepanda.read_json(io.StringIO(text), typ="series", precise_float=precise)
        theirs = pd.read_json(io.StringIO(text), typ="series", precise_float=precise)
        assert mine.tolist() == theirs.tolist()


def test_what_to_json_writes_reads_back(firepanda: ModuleType) -> None:
    """Each orient `to_json` writes reads back to the same values."""
    frame = firepanda.DataFrame({"a": [1, 2], "b": [0.5, None], "c": ["x", "y"]})
    for orient in ("columns", "index", "records", "split", "values"):
        back = firepanda.read_json(io.StringIO(frame.to_json(orient=orient)), orient=orient)
        assert back.iloc[:, 0].tolist() == [1, 2]
        assert back.iloc[:, 2].tolist() == ["x", "y"]


def test_files(firepanda: ModuleType, tmp_path: Path) -> None:
    """A path, a compressed path, bytes in a handle, and text that is not a path."""
    plain, packed = tmp_path / "plain.json", tmp_path / "packed.json.gz"
    plain.write_text('{"a":{"0":1}}')
    packed.write_bytes(gzip.compress(b'{"a":{"0":2}}'))
    assert firepanda.read_json(plain)["a"].tolist() == [1]
    assert firepanda.read_json(str(packed))["a"].tolist() == [2]
    assert firepanda.read_json(io.BytesIO(b'{"a":{"0":3}}'))["a"].tolist() == [3]


@pytest.mark.parametrize(
    "build",
    [
        lambda m: m.read_json('{"a":{"0":1}}'),
        lambda m: m.read_json(io.StringIO('{"a":1}'), nrows=2),
        lambda m: m.read_json(io.StringIO('{"a":1}'), chunksize=2),
        lambda m: m.read_json(io.StringIO('{"a":1}')),
        lambda m: m.read_json(io.StringIO('{"columns":[],"x":1}'), orient="split"),
        lambda m: m.read_json(io.StringIO("{}"), date_unit="h"),
        lambda m: m.read_json(io.StringIO("{}"), typ="x"),
        lambda m: m.read_json(io.StringIO("{}"), engine="x"),
        lambda m: m.read_json(io.StringIO("[99999999999999999999]"), typ="series"),
        lambda m: m.read_json(io.StringIO("[-99999999999999999999.5]"), typ="series"),
    ],
)
def test_mistakes_fail_as_pandas_fails(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Text that is not a path, options that need lines, bad keys, units and numbers."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value) == str(theirs.value)
