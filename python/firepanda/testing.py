"""pandas' `pandas.testing`: the assertions a test suite written against pandas imports.

Each function either returns None or raises `AssertionError` with the message
pandas raises, the lines in the same order and the numbers written the same
way, so a test that fails points at the same place under either library. The
checks read each side through the public interface, the dtype, the index, the
name and the values as a list, so they measure what a program sees. Document
100 of the compat notes describes the rules.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "assert_extension_array_equal",
    "assert_frame_equal",
    "assert_index_equal",
    "assert_series_equal",
]


class _NoDefault:
    """The marker for a keyword left out, since `check_exact` means something as None."""

    def __repr__(self) -> str:
        return "<no_default>"


_NO_DEFAULT: Any = _NoDefault()

_SEQUENCE_ITEMS = 100
"""How many values a message shows before `...`, which is pandas' `display.max_seq_items`."""

_UNITS = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}

_MISSING_DATE = -(2**63)
"""The whole number pandas writes a missing date as when it compares dates by their value."""


def _is_gap(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return value != value
    return type(value).__name__ in {"NAType", "NaTType"}


def _text(value: Any) -> str:
    """One value as pandas prints it inside a message."""
    if value is None:
        return "None"
    if type(value).__name__ == "NAType":
        return "<NA>"
    return str(value)


def _listed(values: list[Any]) -> str:
    """Values as pandas prints an array inside a message, `[1, 2, 3]`."""
    shown = [_text(value) for value in values[:_SEQUENCE_ITEMS]]
    tail = ", ..." if len(values) > _SEQUENCE_ITEMS else ""
    return "[" + ", ".join(shown) + tail + "]"


def _percent(differ: int, total: int) -> str:
    return str(round(differ * 100.0 / total, 5))


def _detail(
    obj: str,
    message: str,
    left: Any,
    right: Any,
    index_values: list[Any] | None = None,
    first_diff: str | None = None,
) -> AssertionError:
    """The error pandas raises: what differs, then each side, in pandas' layout."""
    text = f"{obj} are different\n\n{message}"
    if index_values is not None:
        text += f"\n[index]: {_listed(index_values)}"
    text += f"\n[left]:  {left}\n[right]: {right}"
    if first_diff is not None:
        text += f"\n{first_diff}"
    return AssertionError(text)


def _check_isinstance(left: Any, right: Any, cls: type) -> None:
    for side in (left, right):
        if not isinstance(side, cls):
            raise AssertionError(f"{cls.__name__} Expected type {cls}, found {type(side)} instead")


def _check_attr(attr: str, left: Any, right: Any, obj: str) -> None:
    """One attribute on both sides, where two gaps are equal."""
    if left is right or (_is_gap(left) and _is_gap(right) and type(left) is type(right)):
        return
    try:
        same = left == right
    except TypeError:
        same = False
    if (type(left).__name__ == "NAType") != (type(right).__name__ == "NAType"):
        same = False
    if not isinstance(same, bool):
        same = bool(all(same))
    if not same:
        raise _detail(obj, f'Attribute "{attr}" are different', left, right)


def _numeric(dtype: str) -> bool:
    """Whether pandas compares a dtype's values exactly by default: whole numbers and flags."""
    lower = dtype.lower()
    return lower.startswith(("int", "uint", "bool")) and "[" not in lower


def _extension(dtype: str) -> bool:
    """Whether pandas holds a dtype in an extension array, which it compares gaps first."""
    from ._masked import masked_name

    return (
        dtype in {"str", "string", "category"}
        or masked_name(dtype) is not None
        or "[pyarrow]" in dtype
        or _dated(dtype)
    )


def _dated(dtype: str) -> bool:
    return dtype.startswith(("datetime64", "timedelta64"))


def _whole_values(values: list[Any], dtype: str) -> list[int]:
    """Dates or durations as the whole numbers of their unit, which pandas compares."""
    unit = dtype.split("[", 1)[1].split(",", 1)[0].rstrip("]").strip() if "[" in dtype else "ns"
    step = _UNITS.get(unit, 1)
    return [_MISSING_DATE if _is_gap(value) else value.value // step for value in values]


def _same(left: Any, right: Any) -> bool:
    if _is_gap(left) and _is_gap(right):
        return True
    if _is_gap(left) or _is_gap(right):
        return False
    try:
        return bool(left == right)
    except Exception:
        return False


def _close(left: Any, right: Any, rtol: float, atol: float) -> bool:
    if _is_gap(left) and _is_gap(right):
        return True
    if _is_gap(left) or _is_gap(right):
        return False
    numbers = (int, float)
    if (
        isinstance(left, numbers)
        and isinstance(right, numbers)
        and not isinstance(left, bool)
        and not isinstance(right, bool)
    ):
        return math.isclose(left, right, rel_tol=rtol, abs_tol=atol)
    return _same(left, right)


def _compare_exactly(
    left: list[Any],
    right: list[Any],
    obj: str,
    index_values: list[Any] | None = None,
    shown: tuple[Any, Any] | None = None,
) -> None:
    """Values compared exactly, as pandas' `assert_numpy_array_equal` does."""
    if len(left) != len(right):
        raise _detail(obj, f"{obj} shapes are different", (len(left),), (len(right),))
    differ = sum(not _same(a, b) for a, b in zip(left, right, strict=True))
    if differ:
        message = f"{obj} values are different ({_percent(differ, len(left))} %)"
        lobj, robj = shown or (_listed(left), _listed(right))
        raise _detail(obj, message, lobj, robj, index_values)


def _compare_closely(
    left: list[Any],
    right: list[Any],
    rtol: float,
    atol: float,
    obj: str,
    index_values: list[Any] | None = None,
    shown: tuple[Any, Any] | None = None,
) -> None:
    """Values compared within a tolerance, as pandas' `assert_almost_equal` does."""
    if len(left) != len(right):
        raise _detail(obj, f"{obj} length are different", len(left), len(right))
    differ = 0
    first_diff = None
    for at, (a, b) in enumerate(zip(left, right, strict=True)):
        if not _close(a, b, rtol, atol):
            differ += 1
            if first_diff is None:
                first_diff = f"At positional index {at}, first diff: {_text(a)} != {_text(b)}"
    if differ:
        message = f"{obj} values are different ({_percent(differ, len(left))} %)"
        lobj, robj = shown or (_listed(left), _listed(right))
        raise _detail(obj, message, lobj, robj, index_values, first_diff)


def _check_class(left: Any, right: Any, exact: Any, obj: str) -> None:
    from . import Index, RangeIndex

    if type(left) is type(right):
        return

    def equiv(side: Any) -> bool:
        return type(side) is Index or isinstance(side, RangeIndex)

    if exact == "equiv" and equiv(left) and equiv(right):
        return

    def shown(side: Any) -> Any:
        return side if isinstance(side, Index) else type(side).__name__

    raise _detail(obj, f"{obj} classes are different", shown(left), shown(right))


def _check_index_types(left: Any, right: Any, exact: Any, obj: str) -> None:
    if not exact:
        return
    _check_class(left, right, exact, obj)
    _check_attr("inferred_type", left.inferred_type, right.inferred_type, obj)
    _check_attr("dtype", str(left.dtype), str(right.dtype), obj)


def assert_index_equal(
    left: Any,
    right: Any,
    exact: bool | str = "equiv",
    check_names: bool = True,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_order: bool = True,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    obj: str | None = None,
) -> None:
    """Checks that two indexes are equal, as pandas' `assert_index_equal` does.

    Raises:
        AssertionError: At the first check that differs, with pandas' message.
    """
    from . import Index, MultiIndex

    __tracebackhide__ = True
    obj = obj or ("MultiIndex" if isinstance(left, MultiIndex) else "Index")
    if not isinstance(left, MultiIndex) or not isinstance(right, MultiIndex):
        _check_isinstance(left, right, Index)
    _check_index_types(left, right, exact, obj)
    if left.nlevels != right.nlevels:
        raise _detail(
            obj,
            f"{obj} levels are different",
            f"{left.nlevels}, {left}",
            f"{right.nlevels}, {right}",
        )
    if len(left) != len(right):
        raise _detail(
            obj, f"{obj} length are different", f"{len(left)}, {left}", f"{len(right)}, {right}"
        )
    if not check_order:
        left = left.sort_values()
        right = right.sort_values()
    if left.nlevels > 1:
        for level in range(left.nlevels):
            assert_index_equal(
                left.get_level_values(level),
                right.get_level_values(level),
                exact=exact,
                check_names=check_names,
                check_exact=check_exact,
                check_categorical=check_categorical,
                rtol=rtol,
                atol=atol,
                obj=f"{obj} level [{level}]",
            )
    elif check_exact and check_categorical:
        _compare_exactly(list(left), list(right), obj, shown=(left, right))
    else:
        _compare_closely(list(left), list(right), rtol, atol, obj, shown=(left, right))
    if check_names:
        _check_attr("names", list(left.names), list(right.names), obj)


def _exact_by_default(left: Any, right: Any) -> bool:
    return _numeric(str(left)) or _numeric(str(right))


def assert_extension_array_equal(
    left: Any,
    right: Any,
    check_dtype: bool | str = True,
    index_values: Any = None,
    check_exact: Any = _NO_DEFAULT,
    rtol: Any = _NO_DEFAULT,
    atol: Any = _NO_DEFAULT,
    obj: str = "ExtensionArray",
) -> None:
    """Checks that two arrays are equal, as pandas' `assert_extension_array_equal` does.

    The gaps are compared first, under the name `NA mask`, and then the values
    that are there. Dates and durations are compared as the whole numbers of
    their unit, as pandas compares them.

    Raises:
        AssertionError: At the first check that differs, with pandas' message.
    """
    __tracebackhide__ = True
    if check_exact is _NO_DEFAULT and rtol is _NO_DEFAULT and atol is _NO_DEFAULT:
        check_exact = _exact_by_default(left.dtype, right.dtype)
    elif check_exact is _NO_DEFAULT:
        check_exact = False
    rtol = 1.0e-5 if rtol is _NO_DEFAULT else rtol
    atol = 1.0e-8 if atol is _NO_DEFAULT else atol
    labels = None if index_values is None else list(index_values)
    if check_dtype:
        _check_attr("dtype", str(left.dtype), str(right.dtype), f"Attributes of {obj}")
    lvalues, rvalues = _values(left), _values(right)
    ltype, rtype = str(left.dtype), str(right.dtype)
    if _dated(ltype) and _dated(rtype):
        _compare_exactly(_whole_values(lvalues, ltype), _whole_values(rvalues, rtype), obj, labels)
        return
    lgaps = [_is_gap(value) for value in lvalues]
    rgaps = [_is_gap(value) for value in rvalues]
    _compare_exactly(lgaps, rgaps, f"{obj} NA mask", labels)
    lvalid = [value for value, gap in zip(lvalues, lgaps, strict=True) if not gap]
    rvalid = [value for value, gap in zip(rvalues, rgaps, strict=True) if not gap]
    if check_exact:
        _compare_exactly(lvalid, rvalid, obj, labels)
    else:
        _compare_closely(lvalid, rvalid, rtol, atol, obj, labels)


def _values(column: Any) -> list[Any]:
    return column.tolist() if hasattr(column, "tolist") else list(column)


def _float_gaps(values: list[Any], dtype: str) -> list[Any]:
    """A float column's gaps as NaN, which is how pandas prints them in a message."""
    if dtype.startswith("float"):
        return [math.nan if _is_gap(value) else value for value in values]
    return values


def assert_series_equal(
    left: Any,
    right: Any,
    check_dtype: bool | str = True,
    check_index_type: bool | str = "equiv",
    check_series_type: bool = True,
    check_names: bool = True,
    check_exact: Any = _NO_DEFAULT,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_category_order: bool = True,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: Any = _NO_DEFAULT,
    atol: Any = _NO_DEFAULT,
    obj: str = "Series",
    *,
    check_index: bool = True,
    check_like: bool = False,
) -> None:
    """Checks that two series are equal, as pandas' `assert_series_equal` does.

    The checks run in pandas' order: the kind of object, the length, the flags,
    the index, the dtype, the values, the name and, for categories, the
    categories and codes. Whole numbers and flags are compared exactly unless
    told otherwise, and anything else within `rtol` and `atol`.

    Raises:
        AssertionError: At the first check that differs, with pandas' message.
    """
    from . import Series

    __tracebackhide__ = True
    _check_isinstance(left, right, Series)
    if check_series_type:
        _check_class(left, right, True, obj)
    ltype, rtype = str(left.dtype), str(right.dtype)
    if check_exact is _NO_DEFAULT and rtol is _NO_DEFAULT and atol is _NO_DEFAULT:
        check_exact = _exact_by_default(ltype, rtype)
        exact_index = _whole_index(left.index) or _whole_index(right.index)
    elif check_exact is _NO_DEFAULT:
        check_exact = exact_index = False
    else:
        exact_index = check_exact
    rtol = 1.0e-5 if rtol is _NO_DEFAULT else rtol
    atol = 1.0e-8 if atol is _NO_DEFAULT else atol
    if len(left) != len(right):
        raise _detail(
            obj,
            "Series length are different",
            f"{len(left)}, {left.index}",
            f"{len(right)}, {right.index}",
        )
    if check_flags:
        assert left.flags == right.flags, f"{left.flags!r} != {right.flags!r}"
    if check_index:
        assert_index_equal(
            left.index,
            right.index,
            exact=check_index_type,
            check_names=check_names,
            check_exact=exact_index,
            check_categorical=check_categorical,
            check_order=not check_like,
            rtol=rtol,
            atol=atol,
            obj=f"{obj}.index",
        )
    if check_like:
        left = left.reindex(right.index)
    if check_freq and hasattr(left.index, "freq") and hasattr(right.index, "freq"):
        assert left.index.freq == right.index.freq, (left.index.freq, right.index.freq)
    categorical = "category" in (ltype, rtype)
    if check_dtype and not (ltype == rtype == "category" and not check_categorical):
        _check_attr("dtype", _dtype_of(left), _dtype_of(right), f"Attributes of {obj}")
    labels = list(left.index)
    lvalues, rvalues = left.tolist(), right.tolist()
    both = _extension(ltype) and _extension(rtype)
    if check_exact and both:
        assert_extension_array_equal(
            left, right, check_dtype=check_dtype, index_values=labels, obj=obj
        )
    elif check_exact:
        _compare_exactly(_float_gaps(lvalues, ltype), _float_gaps(rvalues, rtype), obj, labels)
    elif both and "category" not in (ltype, rtype):
        dated = _dated(ltype) and _dated(rtype)
        assert_extension_array_equal(
            left,
            right,
            check_dtype=check_dtype,
            index_values=labels,
            rtol=_NO_DEFAULT if dated else rtol,
            atol=_NO_DEFAULT if dated else atol,
            obj=obj,
        )
    else:
        lvalues, rvalues = _float_gaps(lvalues, ltype), _float_gaps(rvalues, rtype)
        _compare_closely(lvalues, rvalues, rtol, atol, obj, labels)
    if check_names:
        _check_attr("name", left.name, right.name, obj)
    if check_categorical and categorical:
        _check_categories(left, right, check_category_order, f"{obj} category")


def _dtype_of(column: Any) -> str:
    """A dtype as pandas compares and prints it, a category's with its categories."""
    return repr(column.dtype) if str(column.dtype) == "category" else str(column.dtype)


def _whole_index(index: Any) -> bool:
    return all(str(dtype).startswith(("int", "uint")) for dtype in _index_dtypes(index))


def _index_dtypes(index: Any) -> list[Any]:
    if index.nlevels == 1:
        return [index.dtype]
    return [index.get_level_values(level).dtype for level in range(index.nlevels)]


def _check_categories(left: Any, right: Any, check_order: bool, obj: str) -> None:
    """The categories, then the codes, then whether they are ordered, as pandas checks them."""
    if check_order:
        assert_index_equal(left.cat.categories, right.cat.categories, obj=f"{obj}.categories")
        _compare_exactly(left.cat.codes.tolist(), right.cat.codes.tolist(), f"{obj}.codes")
    else:
        assert_index_equal(
            left.cat.categories.sort_values(),
            right.cat.categories.sort_values(),
            obj=f"{obj}.categories",
        )
        _compare_exactly(left.tolist(), right.tolist(), f"{obj}.values")
    _check_attr("ordered", left.cat.ordered, right.cat.ordered, obj)


def assert_frame_equal(
    left: Any,
    right: Any,
    check_dtype: bool | str = True,
    check_index_type: bool | str = "equiv",
    check_column_type: bool | str = "equiv",
    check_frame_type: bool = True,
    check_names: bool = True,
    by_blocks: bool = False,
    check_exact: Any = _NO_DEFAULT,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_like: bool = False,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: Any = _NO_DEFAULT,
    atol: Any = _NO_DEFAULT,
    obj: str = "DataFrame",
) -> None:
    """Checks that two frames are equal, as pandas' `assert_frame_equal` does.

    The shape, the index and the columns are checked first, and then each
    column in turn as a series named by its position and label.

    Raises:
        AssertionError: At the first check that differs, with pandas' message.
    """
    from . import DataFrame

    __tracebackhide__ = True
    close = False if check_exact is _NO_DEFAULT else check_exact
    near = 1.0e-5 if rtol is _NO_DEFAULT else rtol
    far = 1.0e-8 if atol is _NO_DEFAULT else atol
    _check_isinstance(left, right, DataFrame)
    if check_frame_type:
        assert isinstance(left, type(right))
    if left.shape != right.shape:
        raise _detail(obj, f"{obj} shape mismatch", f"{left.shape!r}", f"{right.shape!r}")
    if check_flags:
        assert left.flags == right.flags, f"{left.flags!r} != {right.flags!r}"
    for axis, exact in (("index", check_index_type), ("columns", check_column_type)):
        assert_index_equal(
            getattr(left, axis),
            getattr(right, axis),
            exact=exact,
            check_names=check_names,
            check_exact=close,
            check_categorical=check_categorical,
            check_order=not check_like,
            rtol=near,
            atol=far,
            obj=f"{obj}.{axis}",
        )
    if check_like:
        left = left.reindex(index=right.index, columns=right.columns)
    for at, label in enumerate(left.columns):
        assert_series_equal(
            left.iloc[:, at],
            right.iloc[:, at],
            check_dtype=check_dtype,
            check_index_type=check_index_type,
            check_exact=check_exact,
            check_names=check_names,
            check_datetimelike_compat=check_datetimelike_compat,
            check_categorical=check_categorical,
            check_freq=check_freq,
            obj=f'{obj}.iloc[:, {at}] (column name="{label}")',
            rtol=rtol,
            atol=atol,
            check_index=False,
            check_flags=False,
        )
