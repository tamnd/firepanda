"""pandas' masked types, `Int64`, `Float64`, `boolean` and the rest.

pandas spells these with a capital letter, and a gap in one reads as `NA`
without changing the type. The extension's own numeric columns already keep a
gap that way, and what it has nowhere to hold is the bit saying a column is
masked. So a masked column is an object column whose cells carry the type's
name, which `_objects` writes, and everything that computes reads it into the
lower case column of the same width, runs the extension's kernel, and writes
the answer back in the masked type pandas answers. Document 99 of the compat
notes describes the design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _objects

if TYPE_CHECKING:
    from ._frame import Series

_LOWER: dict[str, str] = {
    **{
        f"{kind}{bits}": f"{kind.lower()}{bits}"
        for kind in ("Int", "UInt")
        for bits in (8, 16, 32, 64)
    },
    "Float32": "float32",
    "Float64": "float64",
    "boolean": "bool",
    "string": "str",
}
"""Each masked type's name and the lower case type of the same width, `str` for text."""

_STRING_SPELLINGS = ("string", "string[python]", "string[pyarrow]")
"""The names of pandas' masked text type, with and without where it is stored."""

_UPPER: dict[str, str] = {lower: name for name, lower in _LOWER.items()}

_COMPARISONS = ("eq", "ne", "lt", "le", "gt", "ge")


class MaskedDtype(str):
    """One of pandas' masked types, equal to its name, such as `Int64` or `boolean`."""

    _name = ""

    def __new__(cls) -> MaskedDtype:
        return super().__new__(cls, cls._name)

    @property
    def name(self) -> str:
        """The name, which is also what the type prints as."""
        return self._name

    @property
    def na_value(self) -> Any:
        """What a gap reads as, which is `NA`."""
        from ._na import NA

        return NA

    @property
    def type(self) -> type:
        """The Python type of one value."""
        lower = _LOWER[self._name]
        return bool if lower == "bool" else float if lower.startswith("float") else int

    @property
    def kind(self) -> str:
        """numpy's letter for the values' kind."""
        return {"b": "b", "u": "u", "f": "f"}.get(_LOWER[self._name][0], "i")

    def __repr__(self) -> str:
        # pandas prints the one flag type without the parentheses.
        return "BooleanDtype" if self._name == "boolean" else f"{type(self).__name__}()"

    def __hash__(self) -> int:
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        return str.__eq__(self, masked_name(other) or other) is True

    def __ne__(self, other: object) -> bool:
        return not self == other


def _dtype_class(name: str) -> type[MaskedDtype]:
    title = "Boolean" if name == "boolean" else name
    return type(
        f"{title}Dtype",
        (MaskedDtype,),
        {"_name": name, "__doc__": f"pandas' `{name}` type.", "__module__": __name__},
    )


Int8Dtype = _dtype_class("Int8")
Int16Dtype = _dtype_class("Int16")
Int32Dtype = _dtype_class("Int32")
Int64Dtype = _dtype_class("Int64")
UInt8Dtype = _dtype_class("UInt8")
UInt16Dtype = _dtype_class("UInt16")
UInt32Dtype = _dtype_class("UInt32")
UInt64Dtype = _dtype_class("UInt64")
Float32Dtype = _dtype_class("Float32")
Float64Dtype = _dtype_class("Float64")
BooleanDtype = _dtype_class("boolean")

_CLASSES = {cls._name: cls for cls in MaskedDtype.__subclasses__()}


def masked_dtype(name: str) -> Any:
    """The dtype object for a masked type's name, `StringDtype` with `NA` for `string`."""
    if name == "string":
        from ._dtypes import StringDtype

        return StringDtype()
    return _CLASSES[name]()


def masked_name(dtype: Any) -> str | None:
    """The masked type a dtype argument asks for, in either library's spelling, or None."""
    if isinstance(dtype, str):
        text = str.__str__(dtype)
        if text in _STRING_SPELLINGS:
            return "string"
        return text if text in _LOWER else None
    if type(dtype).__name__ in {cls.__name__ for cls in _CLASSES.values()}:
        return getattr(dtype, "name", None)
    return None


def masked_for(lower: str) -> str:
    """The masked type of the same width as a lower case type, such as `Int32` for int32."""
    return _UPPER[lower]


def upper_of(lower: str) -> str:
    """The masked type a numpy type stands as, such as `Int64` for `int64`, else `Float64`."""
    return _UPPER.get(lower, "Float64")


def masked_of(column: Any) -> str | None:
    """The masked type of a series, or None for any other column.

    A column with no values has nowhere to write its type, and answers the one
    its wrapper was asked for, as `_kept_kind` keeps it.
    """
    name = _objects.masked_name_of(column._inner)
    if name is None:
        asked = getattr(column, "_asked_kind", None)
        name = None if asked is None else masked_name(asked)
    return name


def _gapless(value: Any) -> Any:
    """A value with every spelling of a gap as None."""
    return (
        None
        if value is None or _objects.is_gap(value) or type(value).__name__ == "NAType"
        else value
    )


def _converted(value: Any, name: str, source: str) -> Any:
    """One value as a masked type holds it.

    Raises:
        TypeError: For a float with a fraction cast to whole numbers, in pandas' words.
    """
    lower = _LOWER[name]
    if lower == "str":
        return value if isinstance(value, str) else str(value)
    if lower == "bool":
        return bool(value)
    if lower.startswith("float"):
        return float(value)
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"cannot safely cast non-equivalent {source} to {lower}")
    return int(value)


def as_masked(column: Series, name: str) -> Series:
    """A column cast to a masked type, a gap of any spelling becoming `NA`."""
    from ._frame import Series
    from ._pandas import _values_of

    source = str(column.dtype)
    values = [_gapless(value) for value in _values_of(column._inner)]
    held = [None if value is None else _converted(value, name, source) for value in values]
    cells = _objects.masked_cells(held, name)
    answer = Series(cells, dtype="str", index=column.index, name=column.name)
    if all(value is None for value in held):
        # No value carries the type, so the wrapper does, as `_kept_kind` says.
        answer._asked_kind = masked_dtype(name)
    return answer


def plain(column: Series) -> Series:
    """A masked column as the lower case column of the same width, its gaps the extension's."""
    from ._frame import Series
    from ._pandas import _gaps_at, _held_values

    lower = _LOWER[masked_of(column) or "Int64"]
    values = _held_values(column._inner)
    if lower == "str":
        return Series(values, dtype="str", index=column.index, name=column.name)
    if not values:
        empty = Series([], index=column.index, name=column.name)
        answer = Series._wrap(empty._inner.cast(lower, False))
    elif all(value is None for value in values):
        answer = Series([0.0] * len(values), index=column.index, name=column.name)
        answer = answer.where(Series([False] * len(values), index=column.index))
    elif lower == "bool" and None in values:
        # Flags beside a gap are objects in pandas, so the gaps go back after.
        filled = [False if value is None else value for value in values]
        answer = Series(filled, index=column.index, name=column.name)
        answer = _gaps_at(
            answer, Series([value is not None for value in values], index=column.index)
        )
    else:
        answer = Series(values, index=column.index, name=column.name)
    if str(answer._inner.dtype()) != lower:
        # A narrower width is cast without its gaps, which the cast refuses, and they go back after.
        present = answer.notna()
        answer = Series._wrap(answer.fillna(0)._inner.cast(lower, False))
        if not all(present.tolist()):
            answer = _gaps_at(answer, present)
    return answer


def rewrap(answer: Any, name: str | None = None) -> Any:
    """A lower case answer written back as the masked type pandas answers.

    Without a `name` the family is read off the answer, so whole numbers answer
    `Int64` of their width, floats `Float64` and flags `boolean`. Anything that
    is not a column, or is a column of another type, is handed back as it is.
    """
    from ._frame import Series
    from ._pandas import SeriesMixin, _held_values

    if not isinstance(answer, SeriesMixin):
        return answer
    name = name or _family(answer)
    if name is None:
        return answer
    values = [_gapless(value) for value in _held_values(answer._inner)]
    held = [None if value is None else _converted(value, name, "float64") for value in values]
    cells = _objects.masked_cells(held, name)
    return Series(cells, dtype="str", index=answer.index, name=answer.name)


def _family(answer: Any) -> str | None:
    """The masked type a lower case column is written back as, or None for another column."""
    lower = str(answer._inner.dtype())
    if lower == "string":
        # The core calls text `string`, and a column of cells is not text.
        return None if _objects.is_object(answer._inner) else "string"
    return _UPPER.get(lower)


def text_answer(answer: Any, source: Any, method: str) -> Any:
    """What a `str` method answers on a `string` column, in the masked types pandas answers.

    Text comes back `string`, flags `boolean` and whole numbers `Int64`, and a row
    that was a gap is `NA` whatever the method made of it, since pandas does not
    ask the question of a missing row. A frame, from `split` or `extract`, is read
    a column at a time, and anything else, such as lists or one joined value, is
    handed back as it is.
    """
    from ._frame import DataFrame
    from ._pandas import DataFrameMixin, SeriesMixin

    if method == "get_dummies":
        return answer
    gaps = _gaps(source)
    if isinstance(answer, SeriesMixin):
        return _with_gaps(answer, gaps)
    if isinstance(answer, DataFrameMixin):
        columns = {name: _with_gaps(answer[name], gaps) for name in answer.columns}
        return DataFrame(columns, index=answer.index)
    return answer


def _with_gaps(answer: Any, gaps: list[bool]) -> Any:
    """One answered column in its masked type, with `NA` on the rows that were gaps."""
    from ._frame import Series

    name = _family(answer)
    if name is None:
        return answer
    values = _held_values_of(answer)
    if len(values) == len(gaps):
        values = [None if gap else value for gap, value in zip(gaps, values, strict=True)]
    held = [None if value is None else _converted(value, name, "float64") for value in values]
    cells = _objects.masked_cells(held, name)
    return Series(cells, dtype="str", index=answer.index, name=answer.name)


def _gaps(column: Any) -> list[bool]:
    return [value is None for value in _held_values_of(column)]


def _held_values_of(column: Any) -> list[Any]:
    from ._pandas import _held_values

    return [_gapless(value) for value in _held_values(column._inner)]


def operated(column: Series, other: Any, op: str, run: Any, flip: bool = False) -> Any:
    """An operator with a masked column on either side, run over the lower case columns.

    A comparison answers `boolean` with a gap wherever either side has one, which
    is the one rule the lower case kernels do not already keep.
    """
    from ._pandas import SeriesMixin

    whole = _whole_over_zero(column, other, op, flip)
    if whole is not None:
        return whole

    left = plain(column) if masked_of(column) else column
    right = other
    if isinstance(other, SeriesMixin) and masked_of(other):
        right = plain(other)
    if type(right).__name__ == "NAType":
        return rewrap(left.where(left != left) if op not in _COMPARISONS else left != left, None)
    answer = run(left, right)
    if op in _COMPARISONS and isinstance(answer, SeriesMixin):
        gaps = _gaps(left)
        if isinstance(right, SeriesMixin) and len(right) == len(left):
            gaps = [a or b for a, b in zip(gaps, _gaps(right), strict=True)]
        flags = [
            None if gap else flag for gap, flag in zip(gaps, _held_values_of(answer), strict=True)
        ]
        return _boolean(flags, answer)
    return rewrap(answer)


def _whole_over_zero(column: Series, other: Any, op: str, flip: bool) -> Series | None:
    """Floor division or the remainder of masked whole numbers, a zero divisor giving 0.

    pandas answers 0 where a masked integer column is floored or taken the
    remainder of by zero, keeping the type, where the lower case kernel answers
    a gap. A divisor that is not a whole number, a column on the right, or two
    masked types that differ go the usual way, and None says so.
    """
    from ._frame import Series
    from ._pandas import SeriesMixin

    name = masked_of(column)
    if flip or op not in ("floordiv", "mod") or not name or "Int" not in name:
        return None
    rows = _held_values_of(column)
    if isinstance(other, SeriesMixin):
        if not other.index.equals(column.index):
            return None
        theirs = masked_of(other)
        if theirs not in (None, name) or not str(other.dtype).startswith(("int", "uint", name)):
            return None
        divisors = _held_values_of(other)
    elif (isinstance(other, int) and not isinstance(other, bool)) or _is_whole(other):
        divisors = [int(other)] * len(rows)
    else:
        return None
    take = (lambda a, b: a // b) if op == "floordiv" else (lambda a, b: a % b)
    held = [
        None if a is None or b is None else 0 if b == 0 else take(int(a), int(b))
        for a, b in zip(rows, divisors, strict=True)
    ]
    cells = _objects.masked_cells(held, name)
    return Series(cells, dtype="str", index=column.index, name=column.name)


def _is_whole(value: Any) -> bool:
    """Whether a value is a numpy whole number."""
    return type(value).__module__ == "numpy" and type(value).__name__.startswith(("int", "uint"))


def _boolean(flags: list[Any], like: Any) -> Series:
    """Flags as a `boolean` column under another column's labels and name."""
    from ._frame import Series

    return Series(
        _objects.masked_cells(flags, "boolean"), dtype="str", index=like.index, name=like.name
    )


def _kleene(op: str, left: Any, right: Any) -> Any:
    """One step of Kleene's logic, where None is not known."""
    if op == "and":
        if left is False or right is False:
            return False
        return None if left is None or right is None else True
    if op == "or":
        if left is True or right is True:
            return True
        return None if left is None or right is None else False
    return None if left is None or right is None else left != right


def logical(column: Series, other: Any, op: str) -> Series:
    """`&`, `|` or `^` with a masked column on either side, by Kleene's logic, as pandas does."""
    from ._pandas import SeriesMixin

    left = [None if value is None else bool(value) for value in _held_values_of(column)]
    if isinstance(other, SeriesMixin):
        right = [None if value is None else bool(value) for value in _held_values_of(other)]
    else:
        flag = _gapless(other)
        right = [None if flag is None else bool(flag)] * len(left)
    flags = [_kleene(op, a, b) for a, b in zip(left, right, strict=True)]
    return _boolean(flags, column)


def reduced(answer: Any) -> Any:
    """A reduction's answer over a masked column, where nothing to answer is `NA`."""
    from ._na import NA

    return NA if _gapless(answer) is None else answer


def float_text(value: float, precision: int) -> str:
    """One value of a masked float column as pandas prints it, on its own.

    pandas writes each value of a masked float column apart, to the display
    precision with the trailing zeros taken off, where a lower case float
    column shares one number of decimals across its values.
    """
    if value != value or value in (float("inf"), float("-inf")):
        return {"inf": "inf", "-inf": "-inf"}.get(str(value), "NaN")
    text = f"{value:.{precision}f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def truth(values: list[Any], kind: str, skipna: bool) -> Any:
    """`any` or `all` over a masked column's values, a gap as None, by Kleene's logic.

    With `skipna` the gaps are left out. Without it a gap answers `NA` unless a
    value settles the answer anyway, a True for `any` or a False for `all`.
    """
    from ._na import NA
    from ._pandas import _numpy

    present = [bool(value) for value in values if value is not None]
    settled = any(present) if kind == "any" else all(present)
    if skipna or len(present) == len(values) or settled == (kind == "any"):
        # A settled answer is a numpy flag, as pandas answers it.
        return _numpy().bool_(settled)
    return NA
