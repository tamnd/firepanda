"""pandas' sparse columns: `SparseDtype`, and the `sparse` accessors.

pandas keeps a sparse column as the values that differ from a fill value and
their positions. What a caller sees of that is the type, which names the
subtype and the fill value, the accessor, which reports the density and the
stored values, and the fill value each operation answers, which is the
operation applied to the fill value. firepanda keeps every value of a sparse
column as an object column whose cells carry the type, which `_objects` writes,
and everything that computes reads the column into the dense column of its
subtype, runs the extension's kernel there, and writes the answer back with the
fill value pandas answers. Document 114 of the compat notes describes the design.
"""

from __future__ import annotations

import math
import operator
import re
from typing import TYPE_CHECKING, Any

from . import _objects
from .errors import InvalidArgumentError

if TYPE_CHECKING:
    from ._frame import DataFrame, Series

_KINDS = {"int": "i", "uint": "u", "float": "f", "bool": "b", "object": "O"}
_TYPES: dict[str, type] = {"i": int, "u": int, "f": float, "b": bool, "O": object}
_SUBTYPE = re.compile(r"Sparse\[(?P<subtype>[^,]*)(, )?(?P<fill_value>.*?)?\]$")
_MISSING = "Missing optional dependency '{}'.  Use pip or conda to install {}."


def _is_na(value: Any) -> bool:
    return _objects.is_gap(value) or type(value).__name__ == "NAType"


def _subtype_of(dtype: Any) -> str:
    """The name of a subtype argument, with text as object, as pandas has it.

    Raises:
        TypeError: For a type numpy has no dtype for, in pandas' words.
    """
    from ._pandas import _is_object_dtype, _named_dtype

    if _is_object_dtype(dtype) or (isinstance(dtype, str) and dtype == "str"):
        return "object"
    name = _named_dtype(dtype)
    if name == "string":
        return "object"
    if _kind(name) is None:
        raise TypeError("SparseDtype subtype must be a numpy dtype")
    return name


def _kind(subtype: str) -> str | None:
    """numpy's one letter kind of a subtype, or None for one firepanda cannot hold sparse."""
    if subtype in ("bool", "object"):
        return _KINDS[subtype]
    for prefix in ("uint", "int", "float"):
        if subtype.startswith(prefix) and subtype[len(prefix) :].isdigit():
            return _KINDS[prefix]
    return None


def _default_fill(kind: str) -> Any:
    return {"i": 0, "u": 0, "b": False}.get(kind, math.nan)


def _holds(kind: str, value: Any) -> bool:
    """Whether a column of the kind can hold a fill value, as numpy's casting rules say."""
    if _is_na(value):
        return kind in ("f", "O") and not (kind == "f" and value is None)
    if kind == "O":
        return True
    if isinstance(value, bool) or type(value).__name__ in ("bool_", "bool"):
        return kind == "b" or kind == "O"
    if kind == "b":
        return False
    if isinstance(value, int) or type(value).__name__.startswith(("int", "uint")):
        return kind != "u" or int(value) >= 0
    if isinstance(value, float) or type(value).__name__.startswith("float"):
        return kind == "f" or float(value).is_integer()
    return False


class SparseDtype(str):
    """The type of a sparse column, named as pandas names it, such as `Sparse[int64, 0]`.

    Equal to its name, and to another `SparseDtype` of the same subtype and fill
    value, any two missing fill values being equal.
    """

    _subtype: str
    _fill_value: Any
    _numpy: bool

    def __new__(cls, dtype: Any = "float64", fill_value: Any = None) -> SparseDtype:
        if isinstance(dtype, SparseDtype):
            if fill_value is None:
                fill_value = dtype.fill_value
            dtype = dtype.subtype
        subtype = _subtype_of(dtype)
        kind = _kind(subtype) or "O"
        if fill_value is None:
            fill_value = _default_fill(kind)
        if isinstance(fill_value, (list, tuple, dict, set)):
            raise InvalidArgumentError(f"fill_value must be a scalar. Got {fill_value} instead")
        if not _holds(kind, fill_value):
            raise InvalidArgumentError(
                "fill_value must be a valid value for the SparseDtype.subtype"
            )
        return cls._made(subtype, fill_value, False)

    @classmethod
    def _made(cls, subtype: str, fill_value: Any, numpy: bool) -> SparseDtype:
        """A type with no checks, `numpy` saying the fill value prints as a numpy scalar."""
        shown = _numpy_repr(subtype, fill_value) if numpy else repr(fill_value)
        self = super().__new__(cls, f"Sparse[{subtype}, {shown}]")
        self._subtype = subtype
        self._fill_value = fill_value
        self._numpy = numpy
        return self

    @property
    def fill_value(self) -> Any:
        """The value a sparse column does not store."""
        return self._fill_value

    @property
    def subtype(self) -> str:
        """The type of the values, as a dense column of them would print it."""
        return self._subtype

    @property
    def name(self) -> str:
        """The name, which is also what the type prints as."""
        return str.__str__(self)

    @property
    def kind(self) -> str:
        """numpy's one letter kind of the subtype."""
        return _kind(self._subtype) or "O"

    @property
    def type(self) -> type:
        """The Python type of one value."""
        return _TYPES[self.kind]

    @property
    def na_value(self) -> float:
        return math.nan

    @property
    def _is_na_fill_value(self) -> bool:
        return _is_na(self._fill_value)

    @property
    def _is_numeric(self) -> bool:
        return self._subtype != "object"

    @property
    def _is_boolean(self) -> bool:
        return self.kind == "b"

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str) and not isinstance(other, SparseDtype):
            try:
                other = SparseDtype.construct_from_string(other)
            except TypeError:
                return False
        if not isinstance(other, SparseDtype):
            return False
        if self._is_na_fill_value or other._is_na_fill_value:
            mine, theirs = self._fill_value, other._fill_value
            same = isinstance(mine, type(theirs)) or isinstance(theirs, type(mine))
        else:
            same = bool(self._fill_value == other._fill_value)
        return self._subtype == other._subtype and same

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __reduce__(self) -> Any:
        return (SparseDtype._made, (self._subtype, self._fill_value, self._numpy))

    @classmethod
    def construct_from_string(cls, string: Any) -> SparseDtype:
        """The type a name such as `Sparse[int64]` or `Sparse[int64, 0]` spells.

        Raises:
            TypeError: If it is not a sparse type's name, or names a fill value
                other than the subtype's default, in pandas' words.
        """
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        message = f"Cannot construct a 'SparseDtype' from '{string}'"
        if not string.startswith("Sparse"):
            raise TypeError(message)
        try:
            subtype, has_fill_value = cls._parse_subtype(string)
            result = SparseDtype(subtype)
        except (ValueError, TypeError, NotImplementedError) as error:
            raise TypeError(message) from error
        if has_fill_value and str.__str__(result) != string:
            raise TypeError(
                f"{message}.\n\nIt looks like the fill_value in the string is not the default"
                " for the dtype. Non-default fill_values are not supported. Use the"
                " 'SparseDtype()' constructor instead."
            )
        return result

    @staticmethod
    def _parse_subtype(dtype: str) -> tuple[str, bool]:
        found = _SUBTYPE.match(dtype)
        if found:
            return found.group("subtype"), bool(found.group("fill_value"))
        if dtype == "Sparse":
            return "float64", False
        raise InvalidArgumentError(f"Cannot parse {dtype}")

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """Whether an argument is a sparse type or names one."""
        dtype = getattr(dtype, "dtype", dtype)
        if isinstance(dtype, SparseDtype):
            return True
        if isinstance(dtype, str) and dtype.startswith("Sparse"):
            subtype, _ = cls._parse_subtype(dtype)
            try:
                return _kind(_subtype_of(subtype)) is not None
            except (TypeError, NotImplementedError, ValueError):
                return False
        return False

    def update_dtype(self, dtype: Any) -> SparseDtype:
        """This type with another subtype, the fill value cast to it as numpy casts it.

        Raises:
            TypeError: If the new subtype is an extension type, in pandas' words.
        """
        if isinstance(dtype, SparseDtype):
            return dtype
        if isinstance(dtype, str) and dtype.startswith("Sparse"):
            return SparseDtype.construct_from_string(dtype)
        try:
            subtype = _subtype_of(dtype)
        except (TypeError, NotImplementedError):
            raise TypeError("sparse arrays of extension dtypes not supported") from None
        return SparseDtype._made(subtype, _cast(self._fill_value, subtype), True)


def _cast(value: Any, subtype: str) -> Any:
    """One value cast to a subtype, as numpy's `astype` casts it."""
    kind = _kind(subtype) or "O"
    if kind == "O":
        return value
    if kind == "b":
        return bool(value)
    if kind == "f":
        return float(value)
    if _is_na(value):
        raise InvalidArgumentError("Cannot convert non-finite values (NA or inf) to integer")
    return int(value)


def _numpy_repr(subtype: str, value: Any) -> str:
    """A fill value as numpy prints the scalar of its subtype, such as `np.int64(1)`."""
    kind = _kind(subtype)
    if kind == "b":
        return "np.True_" if value else "np.False_"
    if kind == "O":
        return repr(value)
    return f"np.{subtype}({value!r})"


def sparse_dtype(dtype: Any) -> SparseDtype | None:
    """The sparse type a dtype argument asks for, or None when it asks for another.

    Raises:
        TypeError: For a sparse name with a fill value other than the default,
            in numpy's words, since pandas hands such a name on to numpy.
    """
    if isinstance(dtype, SparseDtype):
        return dtype
    if isinstance(dtype, str) and str.__str__(dtype).startswith("Sparse"):
        name = str.__str__(dtype)
        try:
            return SparseDtype.construct_from_string(name)
        except TypeError:
            raise TypeError(f"data type '{name}' not understood") from None
    return None


def sparse_of(column: Any) -> SparseDtype | None:
    """The sparse type of a series, or None for any other column."""
    return _objects.sparse_type_of(column._inner)


def plain(column: Series) -> Series:
    """A sparse column as the dense column of its subtype, with the same labels and name."""
    from ._frame import Series
    from ._pandas import _values_of

    dtype = sparse_of(column)
    subtype = dtype.subtype if dtype is not None else "float64"
    values = [math.nan if value is None else value for value in _values_of(column._inner)]
    if subtype == "object":
        return Series(values, dtype="object", index=column.index, name=column.name)
    return Series(values, index=column.index, name=column.name).astype(subtype)


def _filled(value: Any, fill_value: Any) -> bool:
    if _is_na(fill_value):
        return _is_na(value)
    return not _is_na(value) and value == fill_value


def as_sparse(column: Series, dtype: SparseDtype) -> Series:
    """A column cast to a sparse type, its values cast to the subtype first."""
    from ._frame import Series
    from ._pandas import _values_of

    old = sparse_of(column)
    filled: list[bool] = []
    if old is not None:
        column = plain(column)
        # pandas casts the stored values only, so a slot of the old fill value takes the new one.
        filled = [_filled(value, old.fill_value) for value in _values_of(column._inner)]
    if dtype.subtype != "object" and str(column.dtype) != dtype.subtype:
        column = column.astype(dtype.subtype)
    values = [None if _is_na(value) else value for value in _values_of(column._inner)]
    if filled:
        fill = None if _is_na(dtype.fill_value) else dtype.fill_value
        values = [fill if hit else value for value, hit in zip(values, filled, strict=True)]
    cells = _objects.sparse_cells(values, dtype)
    return Series(cells, dtype="str", index=column.index, name=column.name)


def rewrap(answer: Any, fill_value: Any, numpy: bool = False) -> Any:
    """A dense answer written back as sparse, its subtype read off the answer.

    Anything that is not a column is handed back as it is.
    """
    from ._pandas import SeriesMixin

    if not isinstance(answer, SeriesMixin):
        return answer
    subtype = str(answer.dtype)
    if _kind(subtype) in ("i", "u") and answer.hasnans:
        # A gap in whole numbers is a NaN in numpy, which makes the column one of floats.
        subtype = "float64"
        answer = answer.astype(subtype)
    if _kind(subtype) is None or (subtype == "bool" and answer.hasnans):
        subtype = "object"
        answer = answer.astype("object")
    return as_sparse(answer, SparseDtype._made(subtype, fill_value, numpy))


_ARITHMETIC = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
    "floordiv": operator.floordiv,
    "mod": operator.mod,
    "pow": operator.pow,
}
_COMPARISONS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}
_LOGICAL = {"and": operator.and_, "or": operator.or_, "xor": operator.xor}


def _applied(run: Any, left: Any, right: Any) -> Any:
    """An operator on two fill values, with numpy's answer where Python would raise."""
    if _is_na(left) or _is_na(right):
        return math.nan
    try:
        return run(left, right)
    except ZeroDivisionError:
        return math.nan
    except TypeError:
        return math.nan


def _fill_side(other: Any, fill_value: Any) -> Any:
    """The fill value the other side of an operator brings, as pandas reads it.

    A sparse column brings its own, a scalar is its own fill value, and a dense
    column is made sparse with the fill value of the sparse side.
    """
    from ._pandas import SeriesMixin

    if isinstance(other, SeriesMixin):
        dtype = sparse_of(other)
        return dtype.fill_value if dtype is not None else fill_value
    return other


def operated(column: Any, other: Any, op: str, flip: bool, run: Any) -> Any:
    """An operator with a sparse column on either side, run over the dense columns.

    The answer is sparse, and its fill value is the operator applied to the two
    fill values, which prints as a numpy scalar after arithmetic, as pandas'
    does.
    """
    from ._pandas import SeriesMixin

    left_dtype = sparse_of(column)
    fill_value = left_dtype.fill_value if left_dtype is not None else None
    if left_dtype is None:
        right_dtype = sparse_of(other)
        fill_value = right_dtype.fill_value if right_dtype is not None else None
        mine, theirs = _fill_side(column, fill_value), fill_value
    else:
        mine, theirs = fill_value, _fill_side(other, fill_value)
    left = plain(column) if left_dtype is not None else column
    right = plain(other) if isinstance(other, SeriesMixin) and sparse_of(other) else other
    answer = run(left, right)
    first, second = (theirs, mine) if flip else (mine, theirs)
    if op in _COMPARISONS:
        return rewrap(answer, bool(_applied(_COMPARISONS[op], first, second)))
    if op in _LOGICAL:
        return rewrap(answer, _applied(_LOGICAL[op], first, second))
    fill = _applied(_ARITHMETIC[op], first, second)
    subtype = str(answer.dtype) if isinstance(answer, SeriesMixin) else ""
    if subtype.startswith("float") and not _is_na(fill):
        fill = float(fill)
    return rewrap(answer, fill, numpy=True)


def unary(column: Series, op: str, run: Any) -> Any:
    """`-`, `+`, `abs` and `~` on a sparse column, the fill value going the same way."""
    dtype = sparse_of(column)
    assert dtype is not None
    fill = dtype.fill_value
    if not _is_na(fill):
        if op == "neg":
            fill = -fill
        elif op == "abs":
            fill = abs(fill)
        elif op == "invert":
            fill = not fill if isinstance(fill, bool) else ~fill
    return rewrap(run(plain(column)), fill)


_REDUCIBLE = frozenset(("sum", "mean", "max", "min", "count", "size", "first", "last", "nunique"))


def reduced(column: Series, kind: str, run: Any) -> Any:
    """A reduction of a sparse column, over the dense column where pandas has one.

    Raises:
        TypeError: For a reduction pandas does not run on a sparse column, in its words.
    """
    dtype = sparse_of(column)
    if kind not in _REDUCIBLE:
        raise TypeError(f"cannot perform {kind} with type {dtype}")
    return run(plain(column))


def scanned(column: Series, kind: str) -> Any:
    """A running total or extreme, which pandas refuses on a sparse column."""
    raise NotImplementedError(f"cannot perform {kind} with type {sparse_of(column)}")


def transformed(column: Series, kind: str, run: Any) -> Any:
    """`isna`, `notna`, `diff`, `shift` and `pct_change` of a sparse column."""
    dtype = sparse_of(column)
    assert dtype is not None
    fill = dtype.fill_value
    answer = run(plain(column))
    if kind == "isna":
        return rewrap(answer, _is_na(fill))
    if kind == "notna":
        return rewrap(answer, not _is_na(fill))
    if kind in ("diff", "pct_change") and not _is_na(fill):
        return rewrap(answer, float(fill - fill) if kind == "diff" else math.nan)
    return rewrap(answer, fill)


def kept(column: Series, run: Any, fill: Any = None) -> Any:
    """A method whose answer keeps the column's fill value, or takes `fill` when it had none."""
    dtype = sparse_of(column)
    assert dtype is not None
    fill_value = dtype.fill_value
    if fill is not None and _is_na(fill_value) and not _is_na(fill):
        fill_value = fill
    return rewrap(run(plain(column)), fill_value)


def mapped(column: Series, run: Any, func: Any) -> Any:
    """`map` and `apply`, whose fill value is the function applied to the fill value.

    A mapping rather than a function answers dense, since its fill value is looked up.
    """
    dtype = sparse_of(column)
    assert dtype is not None
    answer = run(plain(column))
    if not callable(func):
        return answer
    try:
        fill_value = func(dtype.fill_value)
    except Exception:
        return answer
    return rewrap(answer, fill_value)


def _numpy() -> Any:
    try:
        import numpy
    except ImportError:
        raise ImportError(_MISSING.format("numpy", "numpy")) from None
    return numpy


def _stored(values: list[Any], fill_value: Any) -> list[Any]:
    """The values a sparse column stores: the ones that differ from its fill value."""
    if _is_na(fill_value):
        return [value for value in values if not _is_na(value)]
    return [value for value in values if _is_na(value) or value != fill_value]


class SparseAccessor:
    """`Series.sparse`, the fill value, the stored values and the density of a sparse column."""

    __slots__ = ("_column", "_dtype")

    def __init__(self, data: Series) -> None:
        dtype = sparse_of(data)
        if dtype is None:
            raise AttributeError("Can only use the '.sparse' accessor with Sparse data.")
        self._column = data
        self._dtype = dtype

    def _values(self) -> list[Any]:
        return plain(self._column).tolist()

    @property
    def fill_value(self) -> Any:
        """The value the column does not store."""
        return self._dtype.fill_value

    @property
    def npoints(self) -> int:
        """How many values differ from the fill value."""
        return len(_stored(self._values(), self._dtype.fill_value))

    @property
    def density(self) -> float:
        """The share of the values that differ from the fill value."""
        rows = len(self._column)
        return self.npoints / rows if rows else math.nan

    @property
    def sp_values(self) -> Any:
        """The values that differ from the fill value, as a numpy array."""
        numpy = _numpy()
        subtype = self._dtype.subtype
        return numpy.array(_stored(self._values(), self._dtype.fill_value), dtype=subtype)

    def to_dense(self) -> Series:
        """The column as the dense column of its subtype."""
        return plain(self._column)

    @classmethod
    def from_coo(cls, A: Any, dense_index: bool = False) -> Series:
        """A sparse column from a scipy COO matrix, labelled by row and column.

        Raises:
            ImportError: If scipy is not installed, with pandas' sentence.
        """
        from ._frame import Series
        from ._multi import MultiIndex

        _scipy()
        matrix = A.tocoo()
        rows, columns = matrix.row.tolist(), matrix.col.tolist()
        values = matrix.data.tolist()
        if dense_index:
            pairs = [(r, c) for r in range(matrix.shape[0]) for c in range(matrix.shape[1])]
            found = dict(zip(zip(rows, columns, strict=True), values, strict=True))
            values = [found.get(pair, math.nan) for pair in pairs]
        else:
            pairs = list(zip(rows, columns, strict=True))
        index = MultiIndex.from_tuples(pairs)
        subtype = str(matrix.dtype)
        return as_sparse(Series(values, index=index), SparseDtype(subtype, math.nan))

    def to_coo(
        self, row_levels: Any = (0,), column_levels: Any = (1,), sort_labels: bool = False
    ) -> Any:
        """The column as a scipy COO matrix, rows and columns from the index's levels.

        Raises:
            ImportError: If scipy is not installed, with pandas' sentence.
            InvalidArgumentError: If the index is not a MultiIndex of at least two levels.
        """
        scipy = _scipy()
        index = self._column.index
        if getattr(index, "nlevels", 1) < 2:
            raise InvalidArgumentError("to_coo requires MultiIndex with nlevels >= 2.")
        row_levels, column_levels = list(row_levels), list(column_levels)
        stored = [
            (label, value)
            for label, value in zip(index.tolist(), self._values(), strict=True)
            if value in _stored([value], self._dtype.fill_value)
        ]
        row_keys = [_key(label, row_levels) for label, _ in stored]
        column_keys = [_key(label, column_levels) for label, _ in stored]
        row_labels = _labels(row_keys, sort_labels)
        column_labels = _labels(column_keys, sort_labels)
        row_at = {key: at for at, key in enumerate(row_labels)}
        column_at = {key: at for at, key in enumerate(column_labels)}
        matrix = scipy.sparse.coo_matrix(
            (
                [value for _, value in stored],
                ([row_at[key] for key in row_keys], [column_at[key] for key in column_keys]),
            ),
            shape=(len(row_labels), len(column_labels)),
        )
        return matrix, row_labels, column_labels


def _key(label: tuple[Any, ...], levels: list[int]) -> Any:
    return label[levels[0]] if len(levels) == 1 else tuple(label[at] for at in levels)


def _labels(keys: list[Any], sort_labels: bool) -> list[Any]:
    unique = list(dict.fromkeys(keys))
    return sorted(unique) if sort_labels else unique


def _scipy() -> Any:
    try:
        import scipy.sparse
    except ImportError:
        raise ImportError(_MISSING.format("scipy", "scipy")) from None
    return scipy


class SparseFrameAccessor:
    """`DataFrame.sparse`, for a frame whose every column is sparse."""

    __slots__ = ("_frame",)

    def __init__(self, data: DataFrame) -> None:
        names = list(data.columns)
        if not all(sparse_of(data.iloc[:, at]) is not None for at in range(len(names))):
            raise AttributeError("Can only use the '.sparse' accessor with Sparse data.")
        self._frame = data

    def _columns(self) -> list[Series]:
        frame = self._frame
        return [frame.iloc[:, at] for at in range(len(frame.columns))]

    @property
    def density(self) -> float:
        """The share of all the values that differ from their column's fill value."""
        columns = self._columns()
        cells = sum(len(column) for column in columns)
        stored = sum(SparseAccessor(column).npoints for column in columns)
        return stored / cells if cells else math.nan

    def to_dense(self) -> DataFrame:
        """The frame with every column dense."""
        from ._frame import DataFrame

        frame = self._frame
        columns = [plain(column).tolist() for column in self._columns()]
        dtypes = [sparse_of(column).subtype for column in self._columns()]  # type: ignore[union-attr]
        rows = [list(row) for row in zip(*columns, strict=True)] if columns else []
        dense = DataFrame(rows, index=frame.index, columns=frame.columns)
        return dense.astype(dict(zip(frame.columns, dtypes, strict=True))) if columns else dense

    @classmethod
    def from_spmatrix(cls, data: Any, index: Any = None, columns: Any = None) -> DataFrame:
        """A frame of sparse columns from a scipy sparse matrix, filled with zero.

        Raises:
            ImportError: If scipy is not installed, with pandas' sentence.
        """
        from ._frame import DataFrame

        _scipy()
        matrix = data.tocsc()
        rows, width = matrix.shape
        subtype = str(matrix.dtype)
        dtype = SparseDtype(subtype, _cast(0, subtype))
        dense = matrix.toarray().tolist()
        frame = DataFrame(
            dense,
            index=index if index is not None else range(rows),
            columns=columns if columns is not None else range(width),
        )
        return frame.astype(dtype)

    def to_coo(self) -> Any:
        """The frame as a scipy COO matrix, every fill value left out.

        Raises:
            ImportError: If scipy is not installed, with pandas' sentence.
        """
        scipy = _scipy()
        numpy = _numpy()
        rows, columns, values = [], [], []
        for at, column in enumerate(self._columns()):
            fill = sparse_of(column).fill_value  # type: ignore[union-attr]
            for row, value in enumerate(plain(column).tolist()):
                if value in _stored([value], fill):
                    rows.append(row)
                    columns.append(at)
                    values.append(value)
        shape = (len(self._frame), len(self._frame.columns))
        return scipy.sparse.coo_matrix(
            (numpy.array(values), (numpy.array(rows), numpy.array(columns))), shape=shape
        )
