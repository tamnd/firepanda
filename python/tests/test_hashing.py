"""Row hashes and the version parser in `firepanda.util`, compared with pandas.

pandas' `hash_pandas_object` numbers are stable across releases, so the same
inputs must give the same 64 bit numbers here, gaps, keys and index included.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

SERIES: dict[str, Callable[[Any], Any]] = {
    "int": lambda pd: pd.Series([1, -2, 3]),
    "int8": lambda pd: pd.Series([1, -1], dtype="int8"),
    "uint8": lambda pd: pd.Series([1, 255], dtype="uint8"),
    "int32": lambda pd: pd.Series([-5, 7], dtype="int32"),
    "uint64": lambda pd: pd.Series([-1, 1]).astype("uint64"),
    "float": lambda pd: pd.Series([1.5, None, -0.0, 0.0]),
    "float32": lambda pd: pd.Series([1.5, -2.25], dtype="float32"),
    "bool": lambda pd: pd.Series([True, False]),
    "str": lambda pd: pd.Series(["a", None, "b", "a"]),
    "object": lambda pd: pd.Series(["a", 1, None, 2.5], dtype=object),
    "bytes": lambda pd: pd.Series([b"a", "a", None], dtype=object),
    "tuple": lambda pd: pd.Series([(1, 2), (1, 2), None], dtype=object),
    "string": lambda pd: pd.Series(["a", None], dtype="string"),
    "Int64": lambda pd: pd.Series([1, None, 3], dtype="Int64"),
    "Float64": lambda pd: pd.Series([1.5, None], dtype="Float64"),
    "boolean": lambda pd: pd.Series([True, None], dtype="boolean"),
    "UInt8": lambda pd: pd.Series([1, None], dtype="UInt8"),
    "category": lambda pd: pd.Series(["b", "a", None, "b"], dtype="category"),
    "datetime": lambda pd: pd.Series(pd.to_datetime(["2026-01-02", None])),
    "seconds": lambda pd: pd.Series(pd.date_range("2026-01-01", periods=2, unit="s")),
    "zoned": lambda pd: pd.Series(pd.date_range("2026-01-01", periods=2, tz="Asia/Tokyo")),
    "timedelta": lambda pd: pd.Series(pd.to_timedelta(["1s", None])),
    "complex": lambda pd: pd.Series([1 + 2j, 3j]),
    "period": lambda pd: pd.Series(pd.period_range("2026-01", periods=2, freq="M")),
    "labelled": lambda pd: pd.Series([1, 2], index=["x", "y"]),
    "empty": lambda pd: pd.Series([], dtype="float64"),
}

OTHERS: dict[str, Callable[[Any], Any]] = {
    "frame": lambda pd: pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}, index=["p", "q"]),
    "no columns": lambda pd: pd.DataFrame(index=[1, 2]),
    "multi": lambda pd: pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("a", 2)]),
    "multi series": lambda pd: pd.Series(
        [1, 2, 3], index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("a", 2)])
    ),
    "index": lambda pd: pd.Index(["a", "b"]),
    "range": lambda pd: pd.RangeIndex(3),
}

OPTIONS: dict[str, dict[str, Any]] = {
    "default": {},
    "no index": {"index": False},
    "no categorize": {"index": False, "categorize": False},
    "key": {"hash_key": "abcdefghijklmnop"},
    "no key": {"hash_key": None},
    "latin": {"encoding": "latin-1"},
}


def outcome(call: Callable[[], Any]) -> Any:
    """What a call gives, as plain values, or the builtin kind and words of its error."""
    try:
        result = call()
    except Exception as error:
        kind = next(k for k in type(error).__mro__ if k.__module__ == "builtins")
        return f"{kind.__name__}: {error}"
    labels = list(result.index) if hasattr(result, "index") else None
    return result.tolist(), str(result.dtype), labels


def util_of(pd: ModuleType) -> ModuleType:
    return importlib.import_module(pd.__name__ + ".util")


@needs_pandas
@pytest.mark.parametrize("options", list(OPTIONS))
@pytest.mark.parametrize("case", list(SERIES) + list(OTHERS))
def test_hash_pandas_object_matches_pandas(firepanda: ModuleType, case: str, options: str) -> None:
    import pandas as pd

    make = {**SERIES, **OTHERS}[case]
    keywords = OPTIONS[options]
    mine = outcome(lambda: util_of(firepanda).hash_pandas_object(make(firepanda), **keywords))
    theirs = outcome(lambda: util_of(pd).hash_pandas_object(make(pd), **keywords))
    assert mine == theirs


@needs_pandas
@pytest.mark.parametrize(
    "call",
    [
        lambda util, pd: util.hash_pandas_object([1, 2]),
        lambda util, pd: util.hash_pandas_object(pd.DataFrame(index=[1, 2]), index=False),
        lambda util, pd: util.hash_pandas_object(pd.Series(["a"]), hash_key="short"),
        lambda util, pd: util.hash_pandas_object(pd.Series([1]), hash_key="short"),
        lambda util, pd: util.hash_array([1, 2]),
        lambda util, pd: util.hash_array(pd.Series([1, 2])),
        lambda util, pd: util.hash_array(pd.array([1, None], dtype="Int64")),
        lambda util, pd: util.hash_array(pd.Categorical(["a", "b", "a"])),
        lambda util, pd: util.hash_array(pd.array(["a", None])),
    ],
)
def test_the_edges_match_pandas(firepanda: ModuleType, call: Callable[..., Any]) -> None:
    import pandas as pd

    mine = outcome(lambda: call(util_of(firepanda), firepanda))
    theirs = outcome(lambda: call(util_of(pd), pd))
    assert mine == theirs


@needs_pandas
@pytest.mark.parametrize("categorize", [True, False])
@pytest.mark.parametrize(
    "make",
    [
        lambda np: np.array([1, -2], dtype="i8"),
        lambda np: np.array([-1], dtype="i1"),
        lambda np: np.array([1.5, np.nan]),
        lambda np: np.array([1.5], dtype="f4"),
        lambda np: np.array([True, False]),
        lambda np: np.array(["a", None, 1], dtype=object),
        lambda np: np.array(["2026-01-01", "NaT"], dtype="M8[s]"),
        lambda np: np.array([5, "NaT"], dtype="m8[ns]"),
        lambda np: np.array([1 + 2j]),
        lambda np: np.array([1 + 2j], dtype="c8"),
        lambda np: np.array([[1, 2], [3, 4]]),
        lambda np: np.array([], dtype="f8"),
    ],
)
def test_hash_array_matches_pandas(
    firepanda: ModuleType, make: Callable[[Any], Any], categorize: bool
) -> None:
    np = pytest.importorskip("numpy")
    import pandas as pd

    mine = util_of(firepanda).hash_array(make(np), categorize=categorize)
    theirs = util_of(pd).hash_array(make(np), categorize=categorize)
    assert mine.dtype == theirs.dtype
    assert mine.tolist() == theirs.tolist()


def test_known_numbers(firepanda: ModuleType) -> None:
    """The numbers pandas gives, written down, for a text, a gap and a whole number."""
    util = util_of(firepanda)
    hashed = util.hash_pandas_object(firepanda.Series(["a", None, "a"]), index=False)
    assert hashed.tolist() == [13950350942979735504, 18446744073709551615, 13950350942979735504]
    assert str(hashed.dtype) == "uint64"
    assert util.hash_pandas_object(firepanda.Series([1]), index=False).tolist() == [
        6238072747940578789
    ]


VERSIONS = ["1.0", "1.0.0", "v2.1rc1", "1.0.dev3", "1.0a1", "1.0.post2", "1!2.0", "1.0+local.7"]
VERSIONS += ["2.0-1", "1.0.0b2.post3.dev4", "bad", " 1.2 ", "1.0PREVIEW1", "1.0r"]


def described(version: Any) -> tuple[Any, ...]:
    return (
        *(repr(version), str(version), version.epoch, version.release, version.pre),
        *(version.post, version.dev, version.local, version.public, version.base_version),
        *(version.is_prerelease, version.is_postrelease, version.is_devrelease),
        *(version.major, version.minor, version.micro),
    )


@needs_pandas
@pytest.mark.parametrize("text", VERSIONS)
def test_version_reads_like_pandas(firepanda: ModuleType, text: str) -> None:
    import pandas as pd

    mine = util_of(firepanda).version
    theirs = util_of(pd).version
    assert outcome_of(lambda: described(mine.parse(text))) == outcome_of(
        lambda: described(theirs.parse(text))
    )


def outcome_of(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except ValueError as error:
        return f"ValueError: {error}"


def test_versions_sort_by_pep_440(firepanda: ModuleType) -> None:
    version = util_of(firepanda).version
    order = ["1.0.dev1", "1.0a1.dev1", "1.0a1", "1.0b1", "1.0rc1", "1.0", "1.0+abc", "1.0+1"]
    order += ["1.0.post1.dev1", "1.0.post1", "1.1", "1!0.5"]
    assert [str(v) for v in sorted(version.Version(s) for s in reversed(order))] == order
    assert version.Version("1.0") == version.Version("1.0.0")
    assert hash(version.Version("1.0")) == hash(version.Version("1.0.0"))
    assert version.Version("1.0") != "1.0"
    with pytest.raises(TypeError):
        version.Version("1.0") < "1.0"  # noqa: B015
    assert issubclass(version.InvalidVersion, ValueError)
    assert version.__all__ == ["VERSION_PATTERN", "InvalidVersion", "Version", "parse"]
