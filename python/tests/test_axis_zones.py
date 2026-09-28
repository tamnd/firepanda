"""`tz_localize` and `tz_convert` on a frame or a column, compared with pandas.

Both move the row labels between zones and leave the values alone. Each case builds
the same object in both libraries and compares the labels that come back, or the
mistake raised, by class and message.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")

MOMENTS = ["2024-01-01 10:00", "2024-03-10 02:30", "2024-07-01 00:00"]


def built(lib: ModuleType, kind: str, labels: list[Any], tz: str | None) -> Any:
    """A column or a frame of counts under the given row labels."""
    if labels and isinstance(labels[0], str):
        index = lib.DatetimeIndex(labels, name="when")
    else:
        index = lib.Index(labels)
    if tz:
        index = index.tz_localize(tz)
    values = list(range(len(labels)))
    if kind == "series":
        return lib.Series(values, index=index, name="v")
    return lib.DataFrame({"a": values}, index=index)


def labels_of(result: Any) -> tuple[Any, ...]:
    """What the call did to the labels, and what it kept."""
    index = result.index
    return str(index.dtype), index.name, [str(x) for x in index], result.shape


CASES: list[tuple[str, list[Any], str | None, Callable[[Any], Any]]] = [
    ("series", MOMENTS, None, lambda o: o.tz_localize("UTC")),
    ("frame", MOMENTS, None, lambda o: o.tz_localize("America/New_York", nonexistent="NaT")),
    ("frame", MOMENTS, None, lambda o: o.tz_localize("Asia/Tokyo", axis="index")),
    ("series", MOMENTS, "UTC", lambda o: o.tz_convert("Asia/Tokyo")),
    ("frame", MOMENTS, "UTC", lambda o: o.tz_convert(None)),
    ("frame", MOMENTS, "UTC", lambda o: o.tz_localize(None)),
    ("frame", MOMENTS, None, lambda o: o.tz_localize("UTC", level="when")),
    ("frame", MOMENTS, None, lambda o: o.tz_localize("UTC", level=0)),
    ("series", [], None, lambda o: o.tz_localize("UTC")),
]


@pytest.mark.parametrize(("kind", "labels", "tz", "call"), CASES)
def test_the_labels_move_as_in_pandas(
    kind: str, labels: list[Any], tz: str | None, call: Callable[[Any], Any]
) -> None:
    ours = call(built(fp, kind, labels, tz))
    theirs = call(built(pd, kind, labels, tz))
    assert labels_of(ours) == labels_of(theirs)
    if kind == "frame":
        ours, theirs = ours["a"], theirs["a"]
    assert ours.tolist() == theirs.tolist()


MISTAKES: list[tuple[str, list[Any], str | None, Callable[[Any], Any]]] = [
    ("series", MOMENTS, None, lambda o: o.tz_convert("UTC")),
    ("series", [1, 2], None, lambda o: o.tz_localize("UTC")),
    ("frame", [1, 2], None, lambda o: o.tz_localize("UTC", axis=1)),
    ("frame", MOMENTS, None, lambda o: o.tz_localize("UTC", level=1)),
    ("series", MOMENTS, None, lambda o: o.tz_localize("UTC", axis=1)),
    ("frame", MOMENTS, "UTC", lambda o: o.tz_localize("Asia/Tokyo")),
]


@pytest.mark.parametrize(("kind", "labels", "tz", "call"), MISTAKES)
def test_the_mistakes_are_pandas_mistakes(
    kind: str, labels: list[Any], tz: str | None, call: Callable[[Any], Any]
) -> None:
    with pytest.raises(Exception) as theirs:
        call(built(pd, kind, labels, tz))
    with pytest.raises(Exception) as ours:
        call(built(fp, kind, labels, tz))
    assert isinstance(ours.value, type(theirs.value))
    # The core names its layer and may add a hint after pandas' own words.
    assert str(ours.value).removeprefix("temporal: ").startswith(str(theirs.value))


def test_a_copy_warns_as_in_pandas() -> None:
    # pandas raises its own subclass of the deprecation warning.
    for lib in (fp, pd):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            built(lib, "frame", MOMENTS, None).tz_localize("UTC", copy=True)
        assert len(caught) == 1
        assert issubclass(caught[0].category, DeprecationWarning)


def test_the_signatures_are_pandas_signatures() -> None:
    import inspect

    for owner in ("DataFrame", "Series"):
        for name in ("tz_localize", "tz_convert"):
            ours = inspect.signature(getattr(getattr(fp, owner), name))
            theirs = inspect.signature(getattr(getattr(pd, owner), name))
            assert list(ours.parameters) == list(theirs.parameters)
