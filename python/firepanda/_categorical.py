"""`CategoricalDtype` and `Categorical`, two of pandas' three names for categories.

A category column in firepanda is an Arrow dictionary column: a code for each
row and a text column of categories. pandas spells the same thing three ways,
and each is a thin shell over that column here.

`CategoricalDtype` is a `str` equal to `"category"`, which is how firepanda
spells every other type, so everything that compares a column's type against
the word keeps working, and it carries `categories` and `ordered` the way
pandas' does. `Categorical` is a `FirepandaArray` of categories, and
`CategoricalIndex`, the index of categories, is in `_category_index`.

The categories are text, as they are in every category column here, so values
of any other type are refused where the column is made.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._array import FirepandaArray
from .errors import DTypeError, InvalidArgumentError

if TYPE_CHECKING:
    from ._frame import Index, Series


def _kind_of(categories: Index) -> str:
    """The type pandas prints for categories: `str` for text, and the index's type otherwise."""
    kind = str(categories.dtype)
    return "str" if kind == "string" else kind


class CategoricalDtype(str):
    """The type of a category column: its categories and whether their order means anything.

    Equal to the word `"category"`, and to another `CategoricalDtype` with the
    same categories and order flag, where unordered categories are the same in
    any order, as in pandas. The categories are read from the column only when
    asked for, so reading a column's type costs nothing.
    """

    def __new__(cls, categories: Any = None, ordered: Any = False) -> CategoricalDtype:
        """Checks the categories, in pandas' words.

        Raises:
            ValueError: For a category that repeats, or one that is missing.
        """
        self = super().__new__(cls, "category")
        if categories is not None:
            held = list(categories.tolist() if hasattr(categories, "tolist") else categories)
            if any(label is None or label != label for label in held):
                raise InvalidArgumentError("Categorical categories cannot be null")
            if len(set(held)) != len(held):
                raise InvalidArgumentError("Categorical categories must be unique")
            categories = held
        self._held = categories
        self._ordered = ordered
        self._source: Series | None = None
        return self

    @classmethod
    def _of(cls, column: Series) -> CategoricalDtype:
        """The type of a category column, reading its categories when they are asked for."""
        self = cls()
        self._source = column
        return self

    def _read(self) -> None:
        """Reads the categories and the order flag off the column this came from."""
        if self._source is not None:
            accessor = self._source.cat
            self._held = accessor.categories.tolist()
            self._ordered = accessor.ordered
            self._source = None

    @property
    def categories(self) -> Index | None:
        """The categories, as an index, or None when they are not decided."""
        from ._frame import Index

        self._read()
        return None if self._held is None else Index(self._held, dtype=None)

    @property
    def ordered(self) -> Any:
        """Whether the categories are in order."""
        self._read()
        return self._ordered

    @property
    def name(self) -> str:
        """`category`, the name pandas gives the type."""
        return "category"

    @property
    def kind(self) -> str:
        """`O`, the kind pandas gives the type."""
        return "O"

    @property
    def _decided(self) -> bool:
        """Whether this says more than the word does: its categories, or that they are in order."""
        self._read()
        return self._held is not None or bool(self._ordered)

    def __repr__(self) -> str:
        self._read()
        if self._held is None:
            return (
                f"CategoricalDtype(categories=None, ordered={self._ordered}, categories_dtype=None)"
            )
        shown = ", ".join(repr(label) for label in self._held)
        kind = _kind_of(self.categories) if self._held else "str"
        return (
            f"CategoricalDtype(categories=[{shown}], ordered={self._ordered},"
            f" categories_dtype={kind})"
        )

    def __str__(self) -> str:
        return "category"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CategoricalDtype):
            return isinstance(other, str) and str.__eq__(other, "category")
        self._read()
        other._read()
        if self._held is None or other._held is None:
            return self._held is other._held and bool(self._ordered) == bool(other._ordered)
        if bool(self._ordered) != bool(other._ordered):
            return False
        if self._ordered:
            return self._held == other._held
        return set(self._held) == set(other._held)

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = str.__hash__

    def __reduce__(self) -> Any:
        self._read()
        return (CategoricalDtype, (self._held, self._ordered))

    def __copy__(self) -> CategoricalDtype:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> CategoricalDtype:
        return self


def _code_width(count: int) -> str:
    """The narrowest integer type pandas holds codes for `count` categories in."""
    if count < 2**7:
        return "int8"
    if count < 2**15:
        return "int16"
    return "int32" if count < 2**31 else "int64"


def _codes_array(column: Series) -> Any:
    """A category column's codes as a numpy array as narrow as pandas makes it."""
    import numpy as np

    codes = column.cat.codes.tolist()
    return np.array(codes, dtype=_code_width(len(column.cat.categories)))


def _categorical_column(
    values: Any, categories: Any, ordered: Any, dtype: Any, where: str
) -> Series:
    """The category column pandas makes of `values` under `categories`, `ordered` and `dtype`.

    Raises:
        ValueError: For `dtype` given with `categories` or `ordered`, as in pandas.
    """
    from ._frame import Index, Series

    if dtype is not None and dtype != "category":
        raise InvalidArgumentError(f"{where}: dtype has to be category, not {dtype!r}")
    if isinstance(dtype, CategoricalDtype):
        if categories is not None or ordered is not None:
            raise InvalidArgumentError(
                "Cannot specify `categories` or `ordered` together with `dtype`."
            )
        dtype._read()
        categories, ordered = dtype._held, dtype._ordered
    if isinstance(values, FirepandaArray):
        column = values._column
    elif isinstance(values, Series):
        column = values.reset_index(drop=True).rename(None)
    elif isinstance(values, Index):
        column = values.to_series().reset_index(drop=True).rename(None)
    else:
        values = list(values)
        column = Series(values, dtype="str" if not values else None)
    if str(column.dtype) != "category":
        column = column.astype("category")
    if categories is not None:
        wanted = CategoricalDtype(categories)._held
        column = column.cat.set_categories(wanted, ordered=bool(ordered))
    elif ordered is not None:
        column = column.cat.set_categories(column.cat.categories.tolist(), ordered=bool(ordered))
    return column


class Categorical(FirepandaArray):
    """Values drawn from a list of categories, as pandas' `Categorical`.

    With no categories given they are the values seen, in sorted order, as in
    pandas. Categories that are given leave a gap for each value not in them.
    """

    __slots__ = ()

    def __init__(
        self,
        values: Any,
        categories: Any = None,
        ordered: Any = None,
        dtype: Any = None,
        copy: bool = True,
    ) -> None:
        super().__init__(_categorical_column(values, categories, ordered, dtype, "Categorical"))

    @classmethod
    def _held_by(cls, column: Series) -> Categorical:
        """A categorical over a category column that is already made."""
        self = object.__new__(cls)
        self._column = column.reset_index(drop=True).rename(None)
        return self

    @classmethod
    def from_codes(
        cls,
        codes: Any,
        categories: Any = None,
        ordered: Any = None,
        dtype: Any = None,
        validate: bool = True,
    ) -> Categorical:
        """The categorical whose values are `categories` picked by `codes`, -1 for a gap.

        Raises:
            ValueError: For a code past the categories, or for no categories at all,
                in pandas' words.
        """
        from ._pandas import _coded_categories

        if isinstance(dtype, CategoricalDtype):
            if categories is not None or ordered is not None:
                raise InvalidArgumentError(
                    "Cannot specify `categories` or `ordered` together with `dtype`."
                )
            dtype._read()
            categories, ordered = dtype._held, dtype._ordered
        if categories is None:
            raise InvalidArgumentError(
                "The categories must be provided in 'categories' or 'dtype'. Both were None."
            )
        held = CategoricalDtype(categories)._held
        positions = [int(code) for code in (codes.tolist() if hasattr(codes, "tolist") else codes)]
        if validate and any(code < -1 or code >= len(held) for code in positions):
            raise InvalidArgumentError("codes need to be between -1 and len(categories)-1")
        return cls._held_by(_coded_categories(positions, list(held), bool(ordered)))

    @property
    def dtype(self) -> CategoricalDtype:
        """The categories and the order flag, as a `CategoricalDtype`."""
        return CategoricalDtype._of(self._column)

    @property
    def categories(self) -> Index:
        """The categories, as an index."""
        return self._column.cat.categories

    @property
    def ordered(self) -> bool:
        """Whether the categories are in order."""
        return self._column.cat.ordered

    @property
    def codes(self) -> Any:
        """The position of each value in the categories, -1 for a gap, as a numpy array."""
        return _codes_array(self._column)

    def __getitem__(self, key: Any) -> Any:
        """One value for a position, and a categorical for a slice, a list or a mask."""
        if isinstance(key, int):
            return self._column.iloc[key]
        if isinstance(key, FirepandaArray):
            key = key.tolist()
        return Categorical._held_by(self._column.iloc[key])

    def __setitem__(self, key: Any, value: Any) -> None:
        """Values put in place, each one of the categories.

        Raises:
            TypeError: For a value not among the categories, in pandas' words.
        """
        from ._pandas import _missing

        values = self._column.tolist()
        places = range(len(values))
        if isinstance(key, int):
            chosen, new = [places[key]], [value]
        else:
            if isinstance(key, slice):
                chosen = list(places[key])
            elif all(isinstance(item, bool) for item in list(key)):
                chosen = [at for at, flag in zip(places, list(key), strict=True) if flag]
            else:
                chosen = [places[at] for at in list(key)]
            many = isinstance(value, list | tuple | FirepandaArray)
            new = list(value) if many else [value] * len(chosen)
        categories = self.categories.tolist()
        for item in new:
            if not _missing(item) and item not in categories:
                raise DTypeError(
                    f"Cannot setitem on a Categorical with a new category ({item}), "
                    "set the categories first"
                )
        for at, item in zip(chosen, new, strict=True):
            values[at] = item
        self._column = Categorical(values, categories=categories, ordered=self.ordered)._column

    def _again(self, column: Series) -> Categorical:
        return Categorical._held_by(column)

    def map(self, mapper: Any, na_action: Any = None) -> Any:
        """`mapper` applied to each category, a categorical when no two meet, else an index."""
        from ._frame import Index

        pick = mapper.get if isinstance(mapper, dict) else mapper
        renamed = [pick(category) for category in self.categories.tolist()]
        if len(set(renamed)) == len(renamed):
            return self._changed("rename_categories", renamed)
        return Index(self._mapped(mapper, "ignore"))

    def sort_values(
        self, *, inplace: bool = False, ascending: bool = True, na_position: str = "last"
    ) -> Categorical | None:
        """The values in the order of the categories."""
        order = self.argsort(ascending=ascending, na_position=na_position).tolist()
        ordered = Categorical._held_by(self._column.iloc[order])
        if not inplace:
            return ordered
        self._column = ordered._column
        return None

    def _changed(self, method: str, *args: Any, **kwargs: Any) -> Categorical:
        """The categorical after one of the column's `cat` methods."""
        return Categorical._held_by(getattr(self._column.cat, method)(*args, **kwargs))

    def as_ordered(self) -> Categorical:
        """The same values with the categories in order."""
        return self._changed("as_ordered")

    def as_unordered(self) -> Categorical:
        """The same values with the categories in no order."""
        return self._changed("as_unordered")

    def add_categories(self, new_categories: Any) -> Categorical:
        """New categories on the end of the ones there are."""
        return self._changed("add_categories", new_categories)

    def remove_categories(self, removals: Any) -> Categorical:
        """Without the named categories, each value in them a gap."""
        return self._changed("remove_categories", removals)

    def remove_unused_categories(self) -> Categorical:
        """Without the categories no value is in."""
        return self._changed("remove_unused_categories")

    def rename_categories(self, new_categories: Any) -> Categorical:
        """The categories under new names, each value where it was."""
        return self._changed("rename_categories", new_categories)

    def reorder_categories(self, new_categories: Any, ordered: Any = None) -> Categorical:
        """The same categories in another order."""
        return self._changed("reorder_categories", new_categories, ordered=ordered)

    def set_categories(
        self, new_categories: Any, ordered: Any = None, rename: bool = False
    ) -> Categorical:
        """Another list of categories, each value not in it a gap."""
        return self._changed("set_categories", new_categories, ordered=ordered, rename=rename)

    def __eq__(self, other: object) -> Any:
        import numpy as np

        values = self._column.tolist()
        if isinstance(other, FirepandaArray):
            other = other.tolist()
        if isinstance(other, list | tuple):
            return np.array([a == b for a, b in zip(values, other, strict=True)])
        return np.array([value == other for value in values])

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        values = self._column.tolist()
        held = self.categories.tolist()
        timed = _kind_of(self.categories).startswith(("datetime64", "timedelta64"))
        names = [repr(label) for label in held]
        if timed:
            # Instants and spans print as their index does, midnights as dates, no quotes.
            from ._pandas import _text_values

            names = [text.strip() for text in _text_values(self.categories, justify="left")]
            codes = [int(code) for code in self._column.cat.codes.tolist()]
            values = [names[code] if code >= 0 else None for code in codes]

        spans = _kind_of(self.categories).startswith("interval")
        # pandas lists whole number intervals beside a gap as floats, since the
        # gap is a NaN their endpoints widen to hold.
        widened = _kind_of(self.categories).startswith(("interval[int", "interval[uint"))
        widened = widened and _holds_gap(values)

        def one(value: Any) -> str:
            if value is None or value != value:
                return "NaT" if timed else "NaN"
            if widened:
                value = type(value)(float(value.left), float(value.right), value.closed)
            return value if timed else str(value) if spans else repr(value)

        if len(values) > 10:
            shown = ", ".join([*map(one, values[:5]), "...", *map(one, values[-5:])])
        else:
            shown = ", ".join(map(one, values))
        link = " < " if self.ordered else ", "
        if len(names) > 10:
            names = [*names[:4], "...", *names[-4:]]
        levels = link.join(names)
        kind = _kind_of(self.categories) if held else "object"
        if not values:
            return f"[], Categories (0, {kind}): [{levels}]"
        tail = f"\nLength: {len(values)}" if len(values) > 10 else ""
        if spans:
            # Intervals are listed and wrapped as under a column of them.
            from ._pandas import _text_categories

            return f"[{shown}]{tail}\n{_text_categories(self._column)}"
        return f"[{shown}]{tail}\nCategories ({len(held)}, {kind}): [{levels}]"


def _holds_gap(values: list[Any]) -> bool:
    """Whether any of these values read out of a column is missing."""
    return any(value is None or value != value for value in values)
