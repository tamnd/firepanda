"""`firepanda.testing` against `pandas.testing`, pass or fail and the message.

Each case builds two objects in a library and hands them to that library's
assertion. The outcome is None when the assertion passes and the message when
it raises, and the two libraries' outcomes are compared, which is document 100.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")
pdt = pytest.importorskip("pandas.testing")


def series(lib: ModuleType, left: Any, right: Any, **keywords: Any) -> None:
    lib.testing.assert_series_equal(left(lib), right(lib), **keywords)


CASES: dict[str, Callable[[ModuleType], None]] = {
    "same": lambda lib: series(lib, lambda m: m.Series([1, 2]), lambda m: m.Series([1, 2])),
    "whole-values": lambda lib: series(
        lib, lambda m: m.Series([1, 2, 3]), lambda m: m.Series([1, 2, 4])
    ),
    "dtype": lambda lib: series(lib, lambda m: m.Series([1, 2]), lambda m: m.Series([1.0, 2.0])),
    "dtype-unchecked": lambda lib: series(
        lib, lambda m: m.Series([1, 2]), lambda m: m.Series([1.0, 2.0]), check_dtype=False
    ),
    "length": lambda lib: series(lib, lambda m: m.Series([1, 2]), lambda m: m.Series([1, 2, 3])),
    "name": lambda lib: series(
        lib, lambda m: m.Series([1], name="a"), lambda m: m.Series([1], name="b")
    ),
    "name-unchecked": lambda lib: series(
        lib, lambda m: m.Series([1], name="a"), lambda m: m.Series([1], name="b"), check_names=False
    ),
    "close-floats": lambda lib: series(
        lib, lambda m: m.Series([1.0, 2.0]), lambda m: m.Series([1.0000001, 2.0])
    ),
    "far-floats": lambda lib: series(
        lib, lambda m: m.Series([1.0, 2.0]), lambda m: m.Series([1.1, 2.0])
    ),
    "rtol": lambda lib: series(
        lib, lambda m: m.Series([1.0, 2.0]), lambda m: m.Series([1.1, 2.0]), rtol=0.2
    ),
    "exact-floats": lambda lib: series(
        lib, lambda m: m.Series([1.0, 2.0]), lambda m: m.Series([1.1, 2.0]), check_exact=True
    ),
    "float-gap": lambda lib: series(
        lib, lambda m: m.Series([1.5, None]), lambda m: m.Series([1.5, 2.25])
    ),
    "gaps-equal": lambda lib: series(
        lib, lambda m: m.Series([1.5, None]), lambda m: m.Series([1.5, None])
    ),
    "flags": lambda lib: series(
        lib, lambda m: m.Series([True, False]), lambda m: m.Series([True, True])
    ),
    "text": lambda lib: series(lib, lambda m: m.Series(["a", "b"]), lambda m: m.Series(["a", "c"])),
    "text-gap": lambda lib: series(
        lib, lambda m: m.Series(["a", None]), lambda m: m.Series(["a", "b"])
    ),
    "text-same": lambda lib: series(
        lib, lambda m: m.Series(["a", None]), lambda m: m.Series(["a", None])
    ),
    "index": lambda lib: series(
        lib, lambda m: m.Series([1, 2]), lambda m: m.Series([1, 2], index=[1, 2])
    ),
    "like": lambda lib: series(
        lib,
        lambda m: m.Series([1, 2], index=["a", "b"]),
        lambda m: m.Series([2, 1], index=["b", "a"]),
        check_like=True,
    ),
    "dates": lambda lib: series(
        lib,
        lambda m: m.Series(m.to_datetime(["2020-01-01"])),
        lambda m: m.Series(m.to_datetime(["2020-01-02"])),
    ),
    "masked-gap": lambda lib: series(
        lib,
        lambda m: m.Series([1, 2], dtype="Int64"),
        lambda m: m.Series([1, None], dtype="Int64"),
    ),
    "masked-values": lambda lib: series(
        lib,
        lambda m: m.Series([1, 2], dtype="Int64"),
        lambda m: m.Series([1, 3], dtype="Int64"),
    ),
    "categories": lambda lib: series(
        lib,
        lambda m: m.Series(["a", "b"], dtype="category"),
        lambda m: m.Series(["a", "c"], dtype="category"),
    ),
    "categories-same": lambda lib: series(
        lib,
        lambda m: m.Series(["a", "b"], dtype="category"),
        lambda m: m.Series(["a", "b"], dtype="category"),
    ),
    "floats-masked": lambda lib: series(
        lib,
        lambda m: m.Series([1.5], dtype="Float64"),
        lambda m: m.Series([1.6], dtype="Float64"),
    ),
    "durations": lambda lib: series(
        lib,
        lambda m: m.Series(m.to_timedelta(["1D"])),
        lambda m: m.Series(m.to_timedelta(["2D"])),
    ),
    "long": lambda lib: series(
        lib, lambda m: m.Series(range(150)), lambda m: m.Series(range(1, 151))
    ),
    "obj": lambda lib: series(lib, lambda m: m.Series([1]), lambda m: m.Series([2]), obj="thing"),
    "not-a-series": lambda lib: lib.testing.assert_series_equal(lib.Series([1]), lib.DataFrame()),
    "frame-same": lambda lib: lib.testing.assert_frame_equal(
        lib.DataFrame({"a": [1], "b": ["x"]}), lib.DataFrame({"a": [1], "b": ["x"]})
    ),
    "frame-columns": lambda lib: lib.testing.assert_frame_equal(
        lib.DataFrame({"a": [1]}), lib.DataFrame({"b": [1]})
    ),
    "frame-values": lambda lib: lib.testing.assert_frame_equal(
        lib.DataFrame({"a": [1]}), lib.DataFrame({"a": [2]})
    ),
    "frame-shape": lambda lib: lib.testing.assert_frame_equal(
        lib.DataFrame({"a": [1]}), lib.DataFrame({"a": [1], "b": [2]})
    ),
    "frame-dtype": lambda lib: lib.testing.assert_frame_equal(
        lib.DataFrame({"a": [1]}), lib.DataFrame({"a": [1.0]})
    ),
    "frame-order": lambda lib: lib.testing.assert_frame_equal(
        lib.DataFrame({"a": [1], "b": [2]}), lib.DataFrame({"b": [2], "a": [1]})
    ),
    "frame-like": lambda lib: lib.testing.assert_frame_equal(
        lib.DataFrame({"a": [1], "b": [2]}),
        lib.DataFrame({"b": [2], "a": [1]}),
        check_like=True,
    ),
    "frame-index": lambda lib: lib.testing.assert_frame_equal(
        lib.DataFrame({"a": [1, 2]}), lib.DataFrame({"a": [1, 2]}, index=[0, 2])
    ),
    "frame-text": lambda lib: lib.testing.assert_frame_equal(
        lib.DataFrame({"a": ["x", "y"]}), lib.DataFrame({"a": ["x", "z"]})
    ),
    "frame-index-name": lambda lib: lib.testing.assert_frame_equal(
        lib.DataFrame({"a": [1]}, index=lib.Index([0], name="k")), lib.DataFrame({"a": [1]})
    ),
    "levels": lambda lib: lib.testing.assert_index_equal(
        lib.MultiIndex.from_tuples([(1, "a")]), lib.MultiIndex.from_tuples([(1, "b")])
    ),
    "levels-same": lambda lib: lib.testing.assert_index_equal(
        lib.MultiIndex.from_tuples([(1, "a")]), lib.MultiIndex.from_tuples([(1, "a")])
    ),
    "index-values": lambda lib: lib.testing.assert_index_equal(
        lib.Index([1, 2]), lib.Index([1, 3])
    ),
    "index-type": lambda lib: lib.testing.assert_index_equal(
        lib.Index([1, 2]), lib.Index([1.0, 2.0])
    ),
    "index-names": lambda lib: lib.testing.assert_index_equal(
        lib.Index([1, 2], name="x"), lib.Index([1, 2], name="y")
    ),
    "index-length": lambda lib: lib.testing.assert_index_equal(
        lib.Index([1, 2]), lib.Index([1, 2, 3])
    ),
    "index-equiv": lambda lib: lib.testing.assert_index_equal(lib.RangeIndex(2), lib.Index([0, 1])),
    "index-exact": lambda lib: lib.testing.assert_index_equal(
        lib.RangeIndex(2), lib.Index([0, 1]), exact=True
    ),
    "index-order": lambda lib: lib.testing.assert_index_equal(
        lib.Index([2, 1]), lib.Index([1, 2]), check_order=False
    ),
}


def outcome(case: Callable[[ModuleType], None], lib: ModuleType) -> str | None:
    try:
        case(lib)
    except AssertionError as error:
        return str(error)
    return None


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_an_assertion_answers_as_pandas(case: Callable[[ModuleType], None]) -> None:
    expected = outcome(case, pd)
    if expected is not None:
        expected = expected.replace("pandas.core.frame.", "").replace("pandas.", "")
    answer = outcome(case, fp)
    if answer is not None:
        answer = answer.replace("firepanda._frame.", "")
    assert answer == expected


def test_the_module_has_pandas_names() -> None:
    assert sorted(name for name in dir(pdt) if not name.startswith("_")) == fp.testing.__all__
