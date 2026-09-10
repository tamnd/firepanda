"""The members that are not a plain delegation.

`tools/bindings.py` generates a class per bound type out of one table, and every
member in that table is one expression written against `self._inner`. That is the
right shape for the surface it covers and it is not a shape logic fits into. The
note at the top of the generator says so, and asks that the first member needing
real logic go in a hand written base class rather than be smuggled into the table
as a longer expression, because the moment the table carries code it stops being
reviewable as a table.

This is that file. Everything here is inherited by a generated class, so what a
user holds is still the generated one, and the parity tests still walk the table
rather than this.

The rule for what belongs here is narrow on purpose. A member goes here when what
it does depends on its argument, and nowhere else. Anything that is one call with
a different name belongs in the table where it can be checked against pandas.

`to_datetime` at the bottom is the one thing here that is not a member, and it is
here for the same reason the members are. It is a module level function in pandas
rather than a method, so nothing inherits it and `__init__.py` exports it
directly, but it is ten parameters of which five are refused by name, which is
exactly the shape the generator cannot write and exactly the shape `_refuse` and
`_held_at` exist for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _firepanda
from .errors import InvalidArgumentError, translate

if TYPE_CHECKING:
    from ._frame import DataFrame, DataFrameGroupBy, Index, Series, SeriesGroupBy


def _refuse(name: str, value: object, why: str) -> None:
    """Complains about a constructor argument that is declared and not honoured.

    The pandas signature is declared in full and four of its five parameters are
    not implemented, which is a deliberate choice document 18 section 4 argues
    for: a caller who passes `columns=` gets a message about `columns` rather
    than a TypeError about an unexpected keyword, and the day one of them is
    implemented no signature changes. This is what makes that honest, because a
    declared parameter that is silently ignored is worse than one that is
    missing.

    Args:
        name: The parameter name.
        value: What was passed.
        why: What it would take to support it.

    Raises:
        NotImplementedError: If anything other than None was passed.
    """
    if value is not None:
        raise NotImplementedError(f"{name}= is not supported yet, because {why}")


def _no_level(level: Any) -> None:
    """Refuses the `level` argument every named arithmetic form declares.

    pandas takes a level to say which level of a MultiIndex to align on, and
    firepanda has no MultiIndex yet, so there is nothing for it to name. It is
    declared rather than left out for the reason `_refuse` gives: the signature
    parity test compares the whole parameter list against a running pandas, and a
    caller who passes one is told what is missing.

    Args:
        level: What was passed.

    Raises:
        NotImplementedError: If anything other than None was passed.
    """
    if level is not None:
        raise NotImplementedError(
            "level= is not supported yet, because aligning on one level of a"
            " MultiIndex needs a MultiIndex, and there is not one yet"
        )


def _axis_number(axis: Any, owner: str, default: int, allowed: tuple[int, ...]) -> int:
    """Turns the five spellings of an axis into the number the boundary takes.

    pandas accepts `0`, `1`, `"index"`, `"columns"` and `"rows"` interchangeably,
    and a series accepts only the ones that mean the rows since it has one axis.
    What is allowed is passed in rather than decided here, because that is the
    difference between the two classes and it is one line at each call site.

    `None` is not an error and does not mean the first axis: pandas takes it as
    the default for the call, which is the columns on a frame, and the message
    and the default are therefore both passed in too. A `True` is not an error
    either, which is measured rather than assumed. Python makes a bool an int and
    pandas does nothing to undo that, so `axis=True` means the columns and this
    says so as well, since a compatibility layer that is stricter than the thing
    it copies is still incompatible with it.

    Args:
        axis: What was passed.
        owner: The class name, for the message.
        default: The axis a `None` means.
        allowed: The axis numbers this class has.

    Returns:
        0 for the rows, 1 for the columns.

    Raises:
        InvalidArgumentError: If it is not one of the five, or is an axis this
            class does not have.
    """
    if axis is None:
        return default
    numbers: dict[Any, int] = {0: 0, 1: 1, "index": 0, "columns": 1, "rows": 0}
    try:
        number = numbers[axis]
    except (KeyError, TypeError):
        raise InvalidArgumentError(f"No axis named {axis} for object type {owner}") from None
    if number not in allowed:
        raise InvalidArgumentError(f"No axis named {axis} for object type {owner}") from None
    return number


def _no_fill_against_a_series(fill_value: Any) -> None:
    """Refuses a fill value between a frame and a series.

    This looks like a gap and it is measured. A series is broadcast across the
    rows rather than aligned against them cell by cell, so there is no second side
    for a fill to stand in for, and pandas says `NotImplementedError: fill_value 0
    not supported.` rather than picking a meaning for it. This says the same thing
    at more length, because the pandas message names the value and not the reason.

    Args:
        fill_value: What was passed.

    Raises:
        NotImplementedError: If anything other than None was passed.
    """
    if fill_value is not None:
        raise NotImplementedError(
            f"fill_value {fill_value!r} not supported when the other operand is a"
            " Series, because a Series is broadcast across the rows rather than"
            " aligned against them cell by cell, so there is no second side for"
            " the fill to stand in for"
        )


def _held_at(name: str, value: Any, default: Any, why: str) -> None:
    """Refuses a declared argument that is only implemented at its default.

    `_refuse` is this for the arguments whose default is None, which is most of
    them. The twelve reductions are where that stopped being enough: `skipna`
    defaults to True and `min_count` to zero, so there is no None to compare
    against and the question is whether what arrived is the one value there is an
    implementation for.

    The reason it refuses rather than ignores is the one `_refuse` gives. A
    caller who writes `skipna=False` is asking for a different answer, and
    handing back the answer for `skipna=True` would be wrong in the way that
    takes longest to find.

    Args:
        name: The parameter name.
        value: What was passed.
        default: The one value that is implemented.
        why: What it would take to support the rest.

    Raises:
        NotImplementedError: If it is not the default.
    """
    if value != default:
        raise NotImplementedError(f"{name}={value!r} is not supported yet, because {why}")


def _reducing_axis(axis: Any, owner: str) -> None:
    """Refuses a reduction along the second axis.

    A series has one axis and this is a check on the spelling. A frame has two
    and only one of them reduces: `df.sum()` runs down each column, which is a
    pass over contiguous memory per column, and `df.sum(axis=1)` runs across each
    row, which reads one value out of every column and is a different kernel
    rather than the same one transposed.

    Args:
        axis: What was passed.
        owner: The class name, for the message.

    Raises:
        InvalidArgumentError: If it is not an axis this class has.
        NotImplementedError: If it is the second axis of a frame.
    """
    allowed = (0, 1) if owner == "DataFrame" else (0,)
    if _axis_number(axis, owner, 0, allowed) == 1:
        raise NotImplementedError(
            "axis=1 is not supported yet, because reducing across a row reads one"
            " value out of every column and is a different kernel from the one"
            " that runs down a column"
        )


def _quantile_wanted(q: Any, interpolation: str) -> float:
    """Reads the one quantile a reduction can answer.

    pandas takes a list of quantiles as well as one, and answers a series for a
    series and a frame for a frame. That is a different shape rather than a
    longer loop, so it is refused by shape here and the scalar is what crosses.

    Args:
        q: The quantile, between zero and one.
        interpolation: How to land between two values.

    Returns:
        The quantile as a float.

    Raises:
        InvalidArgumentError: If it is not a number between zero and one.
        NotImplementedError: If it is a list, or if the interpolation is one of
            the four that are not linear.
    """
    _held_at(
        "interpolation",
        interpolation,
        "linear",
        "the reduction lands between two values by weighting them and the other"
        " four rules pick one of them instead",
    )
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        raise NotImplementedError(
            "q has to be a single quantile for now, because a list of them"
            " answers a Series rather than a value and that is a different shape"
        )
    if not 0.0 <= float(q) <= 1.0:
        raise InvalidArgumentError(f"percentiles should all be in the interval [0, 1]. Try {q!r}")
    return float(q)


class _NoDefault:
    """The sentinel pandas puts where a default has to mean "nothing was passed".

    pandas has one of these, `pandas._libs.lib.no_default`, and it appears as the
    default of `shift(fill_value=)`, `dropna(how=)`, `dropna(thresh=)` and
    several more. It exists because `None` is a value a caller might mean, so a
    parameter that treats a missing argument differently from an explicit `None`
    needs a third thing, and pandas made one.

    firepanda needs the same third thing on the same parameters and cannot
    borrow theirs, because importing pandas to be compatible with pandas would
    make the library depend on the thing it replaces. So there is one here, and
    the signature parity test knows the two sentinels are the same idea wearing
    different names. That is the only place the difference is visible, since a
    caller can neither construct one nor tell them apart from the outside.
    """

    def __repr__(self) -> str:
        """Reads the way the pandas one does, so a signature prints the same."""
        return "<no_default>"


NO_DEFAULT = _NoDefault()
"""The one instance, compared by identity the way the pandas one is."""


def _limit_wanted(limit: Any) -> int:
    """Reads a fill limit, where the absence of one means as far as it goes.

    pandas spells no limit as `None` and the core spells it as zero, and this is
    the one line where the two meet. A caller who writes `limit=0` meant
    something and it is not "no limit", so it is rejected here with the pandas
    message rather than being read as its own opposite.

    Args:
        limit: What was passed.

    Returns:
        The limit, or zero for no limit.

    Raises:
        InvalidArgumentError: If it is not a positive whole number.
    """
    if limit is None:
        return 0
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise InvalidArgumentError("Limit must be an integer")
    if limit <= 0:
        raise InvalidArgumentError("Limit must be greater than 0")
    return limit


def _transforming_axis(axis: Any, owner: str) -> None:
    """Refuses a transformation along the second axis.

    Same shape as `_reducing_axis` and the same reason, one step further along.
    `df.cumsum()` totals down each column, which is a pass over contiguous
    memory, and `df.cumsum(axis=1)` totals across each row, which touches every
    column once per row. A frame is stored as columns, so those are two
    different kernels and not one kernel pointed the other way.

    Args:
        axis: What was passed.
        owner: The class name, for the message.

    Raises:
        InvalidArgumentError: If it is not an axis this class has.
        NotImplementedError: If it is the second axis of a frame.
    """
    allowed = (0, 1) if owner == "DataFrame" else (0,)
    if _axis_number(axis, owner, 0, allowed) == 1:
        raise NotImplementedError(
            "axis=1 is not supported yet, because a frame is stored as columns"
            " and running across a row touches every one of them per row, which"
            " is a different kernel from the one that runs down a column"
        )


__all__ = ["NO_DEFAULT", "DataFrameMixin", "IndexMixin", "SeriesMixin"]


class DataFrameMixin:
    """The hand written half of `DataFrame`."""

    __slots__ = ("_inner",)
    """The one piece of state, declared here rather than on the generated class.

    It has to be here because the constructor is here, and a class cannot assign
    to a slot it does not own. The generated subclass declares an empty
    `__slots__`, so an instance still has no `__dict__` and there is still
    exactly one place the extension object lives."""

    _inner: _firepanda.DataFrame

    def __init__(
        self,
        data: Any = None,
        index: Any = None,
        columns: Any = None,
        dtype: Any = None,
        copy: bool | None = None,
    ) -> None:
        """Builds a frame from a mapping of column name to values.

        The signature is the pandas one in full and only the first parameter is
        implemented. The other four are refused by name, which is a stronger
        statement than leaving them out: the signature parity test compares five
        parameters against pandas instead of one, and a caller who passes one of
        them is told what is missing rather than that the keyword is unexpected.
        """
        _refuse("index", index, "putting labels on a frame as it is built is not written")
        _refuse("columns", columns, "selecting and reordering on the way in is not written")
        _refuse("dtype", dtype, "casting on the way in needs the cast machinery")
        _refuse("copy", copy, "there is exactly one behaviour and it always copies")
        try:
            self._inner = _firepanda.DataFrame(data)
        except Exception as error:
            raise translate(error) from None

    def __getitem__(self, key: Any) -> DataFrame | Series:
        """One column as a series, or several as a frame.

        This is the most written expression in pandas and it is two operations
        wearing one name, which is why it cannot be a row in the table. A string
        key takes a column out and a list of strings takes a frame out, and the
        difference is the argument rather than the method.

        pandas reads several other kinds of key here, including a boolean mask, a
        slice and a callable. Those are refused rather than approximated, with a
        message that says what is read today, because a key that quietly means
        something else is worse than one that does not work at all.
        """
        from ._frame import DataFrame, Series

        try:
            if isinstance(key, str):
                return Series._wrap(self._inner.column(key))
            if isinstance(key, (list, tuple)) and all(isinstance(k, str) for k in key):
                return DataFrame._wrap(self._inner.select(list(key)))
        except Exception as error:
            raise translate(error) from None
        raise TypeError(
            f"cannot select with a {type(key).__name__}; df[key] reads a column"
            " name or a list of column names"
        )

    def _operator(self, other: Any, op: str, flip: bool, strict: bool) -> Any:
        """Runs one of the twenty operators, on whichever of three operands it got.

        What the operand is decides which of three calls this makes, which is the
        rule for what belongs in this file rather than in the table. A frame
        aligns on both axes, a series is broadcast along the columns because that
        is the axis an operator has no way to choose, and anything else is a
        constant.

        `strict` is the difference between `==` and `eq`. Two frames that are not
        labelled the same are refused by the operator and aligned by the named
        form, which is pandas' rule and is not arbitrary: a cell only one side has
        has no true or false answer, and only the named form has a `fill_value` to
        say one with.
        """
        from ._frame import DataFrame

        try:
            if isinstance(other, DataFrameMixin):
                if strict:
                    return DataFrame._wrap(self._inner.compare_frame(other._inner, op))
                return DataFrame._wrap(self._inner.binary_frame(other._inner, op, flip, None))
            if isinstance(other, SeriesMixin):
                return DataFrame._wrap(self._inner.binary_series(other._inner, op, 1, flip))
            return DataFrame._wrap(self._inner.binary_value(other, op, flip))
        except Exception as error:
            raise translate(error) from None

    def _named(
        self, other: Any, op: str, axis: Any, level: Any, fill_value: Any, flip: bool
    ) -> Any:
        """Runs one of the twenty named forms, which is the operator plus two arguments.

        `axis` is the one the operators cannot reach. An operator has to pick an
        axis and pandas picks the columns, so `df.add(s, axis=0)` is the only
        spelling of adding a series down the rows there is. Between two frames it
        is read, checked and then ignored, because two frames align on both axes
        whatever it says.

        `fill_value` has three behaviours and all three are pandas'. Between two
        frames it stands in for the side a row or a column is missing from.
        Against a constant it is accepted and ignored, since a constant is never
        the missing side. Against a series it raises.
        """
        from ._frame import DataFrame

        _no_level(level)
        number = _axis_number(axis, "DataFrame", 1, (0, 1))
        try:
            if isinstance(other, DataFrameMixin):
                return DataFrame._wrap(self._inner.binary_frame(other._inner, op, flip, fill_value))
            if isinstance(other, SeriesMixin):
                _no_fill_against_a_series(fill_value)
                return DataFrame._wrap(self._inner.binary_series(other._inner, op, number, flip))
            return DataFrame._wrap(self._inner.binary_value(other, op, flip))
        except Exception as error:
            raise translate(error) from None

    def _unary(self, op: str) -> Any:
        """Runs one of the four unary operations over every column."""
        from ._frame import DataFrame

        try:
            return DataFrame._wrap(self._inner.unary(op))
        except Exception as error:
            raise translate(error) from None

    def _reduce(
        self,
        kind: str,
        param: float,
        axis: Any,
        skipna: bool,
        numeric_only: bool,
        min_count: int,
    ) -> Series:
        """Runs one of the twelve reductions down every column.

        The answer is a series labelled by the column names, which is a
        different shape from the one row frame the core produces, and the turn
        between them is in `PyDataFrame.reduce` where the type the answers have
        to share is picked.
        """
        from ._frame import Series

        _reducing_axis(axis, "DataFrame")
        _held_at(
            "skipna",
            skipna,
            True,
            "a missing value is skipped by every reduction in the library and"
            " there is no second pass that lets one through",
        )
        _held_at(
            "numeric_only",
            numeric_only,
            False,
            "dropping the columns a reduction cannot read is a choice about the"
            " shape of the answer rather than about the reduction",
        )
        _held_at(
            "min_count",
            min_count,
            0,
            "a floor on how many values a sum needs before it answers at all is"
            " a rule about the result rather than about the sum",
        )
        try:
            return Series._wrap(self._inner.reduce(kind, param))
        except Exception as error:
            raise translate(error) from None

    def _quantile(
        self, q: Any, axis: Any, numeric_only: bool, interpolation: str, method: str
    ) -> Series:
        """Runs the quantile down every column."""
        _held_at(
            "method",
            method,
            "single",
            "computing one quantile over the whole frame at once rather than"
            " over each column is a different reduction",
        )
        return self._reduce(
            "quantile", _quantile_wanted(q, interpolation), axis, True, numeric_only, 0
        )

    def _nunique(self, axis: Any, dropna: bool) -> Series:
        """Counts the distinct values in every column."""
        _held_at(
            "dropna",
            dropna,
            True,
            "counting a missing value as one more distinct value needs the count"
            " to know it saw one, and the kernel skips them before it counts",
        )
        return self._reduce("nunique", 0.0, axis, True, False, 0)

    def _transform(
        self, kind: str, periods: int, axis: Any, inplace: bool, ignore_index: bool
    ) -> DataFrame:
        """Runs one named transformation down every column.

        Eleven of the twelve come through here. `dropna` on a frame does not,
        because it removes rows rather than transforming columns, and it has its
        own method below.
        """
        from ._frame import DataFrame

        _transforming_axis(axis, "DataFrame")
        _held_at(
            "inplace",
            inplace,
            False,
            "every operation here answers a new frame and the Arrow buffers"
            " underneath are shared rather than owned, so writing into one would"
            " change frames the caller never mentioned",
        )
        _held_at(
            "ignore_index",
            ignore_index,
            False,
            "throwing the labels away and numbering the rows again is a change to"
            " the index rather than to the values",
        )
        try:
            return DataFrame._wrap(self._inner.transform(kind, periods))
        except Exception as error:
            raise translate(error) from None

    def _fill(self, kind: str, axis: Any, inplace: bool, limit: Any, limit_area: Any) -> DataFrame:
        """Fills each column's missing values from its neighbours."""
        _refuse(
            "limit_area",
            limit_area,
            "filling only the gaps between two present values, or only the ones"
            " outside them, needs the fill to know where the ends are and it"
            " walks the column without looking",
        )
        return self._transform(kind, _limit_wanted(limit), axis, inplace, False)

    def _shift(self, periods: Any, freq: Any, axis: Any, fill_value: Any, suffix: Any) -> DataFrame:
        """Moves every column's rows along, leaving the gap missing."""
        _refuse(
            "freq",
            freq,
            "shifting by a frequency moves the labels rather than the values and"
            " needs the offset vocabulary, which is the resampling milestone",
        )
        _refuse("suffix", suffix, "it only names the columns a list of periods produces")
        _held_at(
            "fill_value",
            fill_value,
            NO_DEFAULT,
            "filling the gap keeps a column of whole numbers whole, and the value"
            " has to reach the kernel as a typed one rather than as a Python"
            " object",
        )
        if not isinstance(periods, int) or isinstance(periods, bool):
            raise NotImplementedError(
                "periods has to be a single number for now, because a list of them"
                " answers a frame with one set of columns per period"
            )
        return self._transform("shift", periods, axis, False, False)

    def _pct_change(self, periods: int, fill_method: Any, freq: Any) -> DataFrame:
        """The fractional change between each row and the one before it."""
        _refuse("fill_method", fill_method, "pandas removed it in 3.0 and only accepts None")
        _refuse("freq", freq, "it needs the offset vocabulary, which is the resampling milestone")
        return self._transform("pct_change", periods, 0, False, False)

    def _scan(self, kind: str, axis: Any, skipna: bool, numeric_only: bool) -> DataFrame:
        """Runs one of the four scans down every column."""
        _held_at(
            "skipna",
            skipna,
            True,
            "a missing row is stepped over and put back where it was, and letting"
            " one through would poison every row after it",
        )
        _held_at(
            "numeric_only",
            numeric_only,
            False,
            "dropping the columns a scan cannot read is a choice about the shape"
            " of the answer rather than about the scan",
        )
        return self._transform(kind, 0, axis, False, False)

    def _dropna(
        self,
        axis: Any,
        how: Any,
        thresh: Any,
        subset: Any,
        inplace: bool,
        ignore_index: bool,
    ) -> DataFrame:
        """Removes the rows that have a missing value in them.

        The one name on the transformation list that means something different
        to a frame than to a column, so it does not go through `_transform`.
        """
        from ._frame import DataFrame

        _transforming_axis(axis, "DataFrame")
        _held_at(
            "how",
            how,
            NO_DEFAULT,
            "dropping a row only when every column is missing is the other rule and"
            " the kernel implements the one where any column disqualifies it",
        )
        _held_at(
            "thresh",
            thresh,
            NO_DEFAULT,
            "keeping a row that has at least so many values counts per row, and"
            " the mask says present or absent rather than how many",
        )
        _held_at(
            "inplace",
            inplace,
            False,
            "the answer is a new frame over shared Arrow buffers, so writing into"
            " one would change frames the caller never mentioned",
        )
        _held_at(
            "ignore_index",
            ignore_index,
            False,
            "numbering the surviving rows again is a change to the index rather"
            " than to which rows survive",
        )
        names: list[str] = []
        if subset is not None:
            names = [subset] if isinstance(subset, str) else [str(one) for one in subset]
        try:
            return DataFrame._wrap(self._inner.dropna(names))
        except Exception as error:
            raise translate(error) from None


class SeriesMixin:
    """The hand written half of `Series`."""

    __slots__ = ("_inner",)
    """The one piece of state, for the reason `DataFrameMixin` gives."""

    _inner: _firepanda.Series

    def __init__(
        self,
        data: Any = None,
        index: Any = None,
        dtype: Any = None,
        name: Any = None,
        copy: bool | None = None,
    ) -> None:
        """Builds a series from a sequence of values, or copies another one.

        Same shape as the frame constructor and refusing the same way, with the
        one difference that `name` is honoured, since a series carries its name
        and there is nothing to implement.

        A series arriving as the data is unwrapped and handed across as the
        extension object it holds, so the extension can copy the column instead
        of iterating it. Going out through a Python list and back would infer
        the type again off the values, which loses a column of instants
        entirely and is slow for the columns it does not lose. It has to be
        unwrapped here because what a user holds is the generated wrapper and
        the extension can only recognise its own type.
        """
        _refuse("index", index, "putting labels on a series as it is built is not written")
        _refuse("dtype", dtype, "casting on the way in needs the cast machinery")
        _refuse("copy", copy, "there is exactly one behaviour and it always copies")
        source = data._inner if isinstance(data, SeriesMixin) else data
        try:
            self._inner = _firepanda.Series(source, "" if name is None else str(name))
        except Exception as error:
            raise translate(error) from None

    def _operator(self, other: Any, op: str, flip: bool, strict: bool) -> Any:
        """Runs one of the twenty operators, on whichever of three operands it got.

        A frame on the right is handed back rather than handled. `s + df` is a
        frame in pandas and the frame is the side that knows how to build one, so
        returning `NotImplemented` is what makes Python turn the expression round
        and ask `DataFrame.__radd__` instead. Refusing here would break an
        expression that has a perfectly good answer.

        `strict` is the difference between `==` and `eq`, for the reason
        `DataFrameMixin._operator` gives.
        """
        from ._frame import Series

        if isinstance(other, DataFrameMixin):
            return NotImplemented
        try:
            if isinstance(other, SeriesMixin):
                if strict:
                    return Series._wrap(self._inner.compare_series(other._inner, op))
                return Series._wrap(self._inner.binary_series(other._inner, op, flip, None))
            return Series._wrap(self._inner.binary_value(other, op, flip))
        except Exception as error:
            raise translate(error) from None

    def _named(
        self, other: Any, op: str, axis: Any, level: Any, fill_value: Any, flip: bool
    ) -> Any:
        """Runs one of the twenty two named forms, which is the operator plus two arguments.

        A series has one axis, so `axis` is read and checked and can only be the
        rows. It is declared because pandas declares it and the signature parity
        test compares the whole parameter list.

        `fill_value` is honoured between two series and ignored against a
        constant, which are two of the three behaviours the frame has. The third
        does not arise, because a series has nothing to broadcast against.
        """
        from ._frame import Series

        _no_level(level)
        _axis_number(axis, "Series", 0, (0,))
        try:
            if isinstance(other, SeriesMixin):
                return Series._wrap(self._inner.binary_series(other._inner, op, flip, fill_value))
            return Series._wrap(self._inner.binary_value(other, op, flip))
        except Exception as error:
            raise translate(error) from None

    def _divmod(self, other: Any, axis: Any, level: Any, fill_value: Any, flip: bool) -> Any:
        """The floor division and the remainder, as the pair Python asks for.

        Two passes rather than one, here and in pandas both, because the pair
        comes out of two operations run separately and a single pass would need a
        kernel that writes two columns, which nothing in `firepanda/kernel` does.
        """
        return (
            self._named(other, "floordiv", axis, level, fill_value, flip),
            self._named(other, "mod", axis, level, fill_value, flip),
        )

    def _unary(self, op: str) -> Any:
        """Runs one of the four unary operations over every row."""
        from ._frame import Series

        try:
            return Series._wrap(self._inner.unary(op))
        except Exception as error:
            raise translate(error) from None

    def _reduce(
        self,
        kind: str,
        param: float,
        axis: Any,
        skipna: bool,
        numeric_only: bool,
        min_count: int,
    ) -> Any:
        """Runs one of the twelve reductions over the whole column.

        The answer is an ordinary Python number rather than a one row series,
        because that is what pandas hands back and because a caller who wrote
        `s.sum() > 10` is holding it in a Python expression a moment later.

        A reduction with no answer comes back from the boundary as `None` and
        leaves here as a float NaN, because that is what pandas gives for the
        mean of nothing and a caller comparing against it will be using `isnan`
        rather than `is None`.
        """
        _reducing_axis(axis, "Series")
        _held_at(
            "skipna",
            skipna,
            True,
            "a missing value is skipped by every reduction in the library and"
            " there is no second pass that lets one through",
        )
        _held_at(
            "numeric_only",
            numeric_only,
            False,
            "refusing a column a reduction cannot read is what the reduction"
            " already does, and it says so with the dtype in the message",
        )
        _held_at(
            "min_count",
            min_count,
            0,
            "a floor on how many values a sum needs before it answers at all is"
            " a rule about the result rather than about the sum",
        )
        try:
            answer = self._inner.reduce(kind, param)
        except Exception as error:
            raise translate(error) from None
        return float("nan") if answer is None else answer

    def _quantile(self, q: Any, interpolation: str) -> Any:
        """Runs the quantile over the whole column."""
        return self._reduce("quantile", _quantile_wanted(q, interpolation), 0, True, False, 0)

    def _nunique(self, axis: Any, dropna: bool) -> Any:
        """Counts the distinct values in the column."""
        _held_at(
            "dropna",
            dropna,
            True,
            "counting a missing value as one more distinct value needs the count"
            " to know it saw one, and the kernel skips them before it counts",
        )
        return self._reduce("nunique", 0.0, axis, True, False, 0)

    def _transform(
        self, kind: str, periods: int, axis: Any, inplace: bool, ignore_index: bool
    ) -> Series:
        """Runs one named transformation over the whole column.

        All twelve come through here, including `dropna`, because a column
        `dropna` removes values and is a transformation like the rest. The frame
        one removes rows and is not.
        """
        from ._frame import Series

        _transforming_axis(axis, "Series")
        _held_at(
            "inplace",
            inplace,
            False,
            "the answer is a new series over shared Arrow buffers, so writing into"
            " one would change columns the caller never mentioned",
        )
        _held_at(
            "ignore_index",
            ignore_index,
            False,
            "throwing the labels away and numbering the rows again is a change to"
            " the index rather than to the values",
        )
        try:
            return Series._wrap(self._inner.transform(kind, periods))
        except Exception as error:
            raise translate(error) from None

    def _fill(self, kind: str, axis: Any, inplace: bool, limit: Any, limit_area: Any) -> Series:
        """Fills the column's missing values from its neighbours."""
        _refuse(
            "limit_area",
            limit_area,
            "filling only the gaps between two present values, or only the ones"
            " outside them, needs the fill to know where the ends are and it"
            " walks the column without looking",
        )
        return self._transform(kind, _limit_wanted(limit), axis, inplace, False)

    def _shift(self, periods: Any, freq: Any, axis: Any, fill_value: Any, suffix: Any) -> Series:
        """Moves the column's rows along, leaving the gap missing."""
        _refuse(
            "freq",
            freq,
            "shifting by a frequency moves the labels rather than the values and"
            " needs the offset vocabulary, which is the resampling milestone",
        )
        _refuse("suffix", suffix, "it only names the columns a list of periods produces")
        _held_at(
            "fill_value",
            fill_value,
            NO_DEFAULT,
            "filling the gap keeps a column of whole numbers whole, and the value"
            " has to reach the kernel as a typed one rather than as a Python"
            " object",
        )
        if not isinstance(periods, int) or isinstance(periods, bool):
            raise NotImplementedError(
                "periods has to be a single number for now, because a list of them"
                " answers a frame with one column per period"
            )
        return self._transform("shift", periods, axis, False, False)

    def _pct_change(self, periods: int, fill_method: Any, freq: Any) -> Series:
        """The fractional change between each row and the one before it."""
        _refuse("fill_method", fill_method, "pandas removed it in 3.0 and only accepts None")
        _refuse("freq", freq, "it needs the offset vocabulary, which is the resampling milestone")
        return self._transform("pct_change", periods, 0, False, False)

    def _scan(self, kind: str, axis: Any, skipna: bool, numeric_only: bool) -> Series:
        """Runs one of the four scans over the whole column."""
        _held_at(
            "skipna",
            skipna,
            True,
            "a missing row is stepped over and put back where it was, and letting"
            " one through would poison every row after it",
        )
        return self._transform(kind, 0, axis, False, False)


class Namespace:
    """One accessor, reached through an attribute the way pandas reaches one.

    `s.dt` builds a `DatetimeProperties` around the series and `Series.dt` is the
    class itself. That second half is the reason this is a descriptor rather than
    a property. A property answers itself when it is read off the class, so
    `Series.dt` would be a `property` object, and a program that reads the
    accessor's members off the class rather than off an instance would find
    nothing. pandas answers the accessor class there, and the conformance board
    reads the class, so this answers the accessor class too.

    Nothing is cached on the instance. pandas does cache, and a firepanda wrapper
    has `__slots__` and therefore nowhere to cache into, so the object is built
    per lookup. It holds one reference and building it is cheaper than the
    dictionary lookup finding a cached one would need. `s.dt is s.dt` is False in
    pandas as well, for a different reason, so nothing observable moves.
    """

    __slots__ = ("_accessor",)
    """The class to build, which is the whole of the state."""

    def __init__(self, accessor: type) -> None:
        """Holds the accessor class.

        Args:
            accessor: The class to build around a series.
        """
        self._accessor = accessor

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        """Answers the accessor class off the class and an accessor off an instance."""
        if obj is None:
            return self._accessor
        return self._accessor(obj)


class DatetimeMixin:
    """The hand written half of `DatetimeProperties`.

    The one mixin whose state is not an extension object. An accessor holds the
    series it was reached from, which is already a wrapper, and the reason it is
    the wrapper rather than the wrapper's `_inner` is that everything the
    accessor hands back is a `Series` and the wrapping has to happen somewhere.
    Holding the wrapper means the accessor never builds one from nothing.

    Nothing here decides what a part means. Every helper turns whatever pandas
    lets a caller write into the word and the one string the boundary takes, and
    the word is read in `firepanda/py/temporal.mojo`.
    """

    __slots__ = ("_series",)
    """The series the accessor was reached from. Slotted for the reason
    `DataFrameMixin` gives, and there is no `_inner` because there is no
    extension object for an accessor to hold."""

    _series: Series

    def __init__(self, data: Series) -> None:
        """Holds the series. Not a public entry point.

        The parameter is called `data` because pandas calls it `data`, and the
        signature board compares the two. It is the only hand written signature
        in this file that a conformance case reads, since every other member here
        is reached through a generated one.
        """
        self._series = data

    def _part(self, kind: str, arg: str) -> Series:
        """Reads one part of the column, and hands back a column.

        Thirty of the names come through here, and the string is the frequency,
        the unit, the format or the zone for the seven that take one and empty
        for the rest.
        """
        from ._frame import Series

        try:
            return Series._wrap(self._series._inner.temporal_part(kind, arg))
        except Exception as error:
            raise translate(error) from None

    def _zone(self) -> str | None:
        """Reads the clock the column is read against.

        None in pandas when the column carries no zone, and the boundary answers
        an empty string, because a Mojo `String` has no absent value and
        inventing one for one caller would be worse than turning it back here.
        """
        try:
            found = self._series._inner.temporal_word("tz")
        except Exception as error:
            raise translate(error) from None
        return found or None

    def _resolution(self) -> str:
        """Reads how many of the column's integers make a second.

        A separate helper from `_zone` rather than one that takes the word,
        which is the only place the accessor splits a door two ways. The reason
        is the return type: a zone can be absent and a resolution cannot, and one
        helper answering both would have to be typed as though either could be.
        """
        try:
            return self._series._inner.temporal_word("unit")
        except Exception as error:
            raise translate(error) from None

    def _rounded(self, kind: str, freq: Any, ambiguous: Any, nonexistent: Any) -> Series:
        """Moves every clock to a frequency, one of three ways."""
        _held_at(
            "ambiguous",
            ambiguous,
            "raise",
            "picking which of the two readings a repeated wall clock hour means"
            " needs the zone's transition table, which is the same work"
            " tz_localize over a fold needs",
        )
        _held_at(
            "nonexistent",
            nonexistent,
            "raise",
            "shifting a wall clock time that a spring forward skipped needs the"
            " zone's transition table",
        )
        if not isinstance(freq, str):
            raise NotImplementedError(
                "freq has to be a string for now, because an offset object carries"
                " the whole frequency vocabulary and firepanda parses the string"
                " spelling only"
            )
        return self._part(kind, freq)

    def _as_unit(self, unit: str, round_ok: bool) -> Series:
        """Restates the column in another resolution."""
        _held_at(
            "round_ok",
            round_ok,
            True,
            "refusing a cast that would lose precision rather than rounding it"
            " needs the cast to look at the values first, and it looks at the"
            " types only",
        )
        return self._part("as_unit", unit)

    def _named(self, kind: str, locale: Any) -> Series:
        """Writes out the name of the day or the month."""
        if locale is not None and not isinstance(locale, str):
            raise TypeError(f"locale has to be a string, not {type(locale).__name__}")
        return self._part(kind, "" if locale is None else locale)

    def _tz_convert(self, tz: Any) -> Series:
        """Reads the same instants against another clock."""
        if tz is None:
            raise NotImplementedError(
                "tz_convert(None) moves the column to UTC and then takes the clock"
                " off, and taking the clock off is tz_localize(None), so this is"
                " two operations pandas spells as one"
            )
        if not isinstance(tz, str):
            raise NotImplementedError(
                "tz has to be a zone name for now, because a tzinfo object is a"
                " Python object and the kernel reads the zone out of a string"
            )
        return self._part("tz_convert", tz)

    def _tz_localize(self, tz: Any, ambiguous: Any, nonexistent: Any) -> Series:
        """Puts the readings on a clock, or takes them off one.

        `None` is a different operation rather than an absent argument, which is
        why it crosses as its own word. Naming a zone keeps the readings and
        changes what they mean, and passing None keeps the instants and drops
        what they were read against.
        """
        _held_at(
            "ambiguous",
            ambiguous,
            "raise",
            "a wall clock hour that a fall back repeats is two instants and"
            " choosing between them needs the zone's transition table",
        )
        _held_at(
            "nonexistent",
            nonexistent,
            "raise",
            "a wall clock time that a spring forward skipped is no instant at"
            " all and shifting it needs the zone's transition table",
        )
        if tz is None:
            return self._part("tz_localize_none", "")
        if not isinstance(tz, str):
            raise NotImplementedError(
                "tz has to be a zone name for now, because a tzinfo object is a"
                " Python object and the kernel reads the zone out of a string"
            )
        return self._part("tz_localize", tz)

    def _isocalendar(self) -> DataFrame:
        """The ISO 8601 week date fields, as a frame of three columns.

        The one part of the accessor that goes through a module level function
        rather than a method on the series. The reason is the import graph and it
        is written where the function is.
        """
        from ._frame import _isocalendar

        return _isocalendar(self._series._inner)


def _grouped(
    frame: DataFrame,
    by: Any,
    level: Any,
    as_index: bool,
    sort: bool,
    group_keys: bool,
    observed: bool,
    dropna: bool,
) -> DataFrameGroupBy:
    """Builds the group by object `df.groupby(...)` hands back.

    Written rather than generated because it is the one member whose body is not
    a call on something the class already holds. It builds a different class,
    and the seven arguments have to be read before there is an object to read
    them into, so a generated one line delegation has nothing to delegate to.

    Two of the seven are declared and refused. `group_keys` decides whether the
    key comes back in the result of an `apply`, and there is no `apply` here for
    it to decide about. `observed` decides whether a categorical key contributes
    the groups it has no rows for, and pandas made True the default in version
    3, which is the behaviour here, so it is refused only at False.

    Args:
        frame: The frame being grouped.
        by: The key column name or names.
        level: Declared and refused, since there is no MultiIndex to have one.
        as_index: Whether the key becomes the row labels.
        sort: Whether the groups come out in key order.
        group_keys: Declared and refused.
        observed: Declared and held at True.
        dropna: Whether a missing key is a group.

    Returns:
        A `DataFrameGroupBy`.
    """
    from ._frame import DataFrameGroupBy

    _held_at(
        "group_keys",
        group_keys,
        True,
        "it says whether the key comes back in the result of an apply, and there"
        " is no apply here for it to say anything about",
    )
    _held_at(
        "observed",
        observed,
        True,
        "it says whether a categorical key contributes the groups it has no rows"
        " for, and a group with no rows in it is a row this has nothing to put in",
    )
    keys = GroupByMixin._keys(frame, by, level)
    return DataFrameGroupBy(frame, keys, as_index, sort, dropna)


def _relabelled(frame: DataFrame, name: str, label: str) -> Series:
    """Takes one column out of a frame and puts a different name on it.

    A group by reduction comes back as a frame, and turning it into the series
    pandas answers means taking one column out, at which point the column is
    called whatever the frame called it. `size` is called `size` there and has no
    name at all in pandas, and a narrowed group by answers a column called after
    the one the caller asked for rather than after the last key. Both are one
    rename and neither is a member a user reaches, so it is a function here
    rather than a method on the wrapper.

    The column is taken through the extension rather than through `frame[name]`,
    which reads a name and a list of names and therefore answers either shape.
    There is one name here and the answer is a column, so going the short way
    says that rather than leaving it to be narrowed afterwards.

    Args:
        frame: The frame the reduction produced.
        name: The column to take out.
        label: The new name, and empty for none.

    Returns:
        The column, carrying the new name.
    """
    from ._frame import Series

    try:
        return Series._wrap(frame._inner.column(name).relabel(label))
    except Exception as error:
        raise translate(error) from None


class GroupByMixin[Answer]:
    """What `DataFrameGroupBy` and `SeriesGroupBy` share, which is all the state.

    `Answer` is what the reductions hand back. A frame's group by answers a frame
    and a column's answers a column, and the fifteen reductions are otherwise the
    same call written twice, so the shared half is written once against the
    parameter and each subclass says which it is. That is not decoration: without
    it every reduction on both classes is declared to answer one thing and
    returns something a type checker only knows as `Any`, which is the one shape
    of mistake a wrapper this thin exists to make impossible.

    A group by object holds a frame, some key column names and three flags, and
    it computes nothing until a reduction is asked for. That is pandas' own
    arrangement and it is the reason the two classes have so little in them: the
    fifteen reductions are one call each with a different word in it, and the
    word is the pandas method name, which is the same string the boundary reads.

    The keys are checked here rather than at the first reduction. pandas raises
    `KeyError` out of `df.groupby("nope")` and not out of the `.sum()` after it,
    and a program that catches the wrong line is a program whose error handling
    does not run. Checking early costs one pass over the column names.

    What is deliberately absent is any notion of the groups themselves. pandas
    can hand back `g.groups`, `g.indices` and `g.get_group(k)`, which are the
    grouping made visible, and firepanda computes the grouping inside the
    reduction and throws it away. Keeping it would mean the object holds an
    index per group whether or not anybody asks, which is the cost pandas pays
    and is the wrong default for a library whose claim is the other one. Those
    three names are absent rather than wrong.
    """

    __slots__ = ("_as_index", "_by", "_dropna", "_frame", "_sort")
    """The frame, the key names, and the three flags that survive to the call.
    Slotted for the reason `DataFrameMixin` gives. There is no `_inner`, because
    a group by object has no extension object of its own: it is a frame and a
    plan for what to do to it."""

    _frame: DataFrame
    _by: list[str]
    _as_index: bool
    _sort: bool
    _dropna: bool

    def __init__(self, frame: DataFrame, by: list[str], as_index: bool, sort: bool, dropna: bool):
        """Holds the frame and the plan. Not a public entry point.

        Args:
            frame: The frame being grouped.
            by: The key column names, already read out of whatever pandas shape
                the caller wrote them in.
            as_index: Whether the key becomes the row labels.
            sort: Whether the groups come out in key order.
            dropna: Whether a missing key is a group.
        """
        self._frame = frame
        self._by = by
        self._as_index = as_index
        self._sort = sort
        self._dropna = dropna

    @staticmethod
    def _keys(frame: DataFrame, by: Any, level: Any) -> list[str]:
        """Reads the key columns out of whatever pandas lets a caller write.

        pandas takes a name, a list of names, a column, a list of columns, a
        function, a dictionary and a level, and turns all seven into the same
        thing. Two of them are here: a name and a list of names. The rest are
        refused by shape with the reason, because each one is a different piece
        of work rather than a longer list.

        Args:
            frame: The frame, so the names can be checked against it.
            by: What the caller passed.
            level: The level, which there is no MultiIndex to have.

        Returns:
            The key column names.

        Raises:
            KeyError: If a name is not a column, which is what pandas raises.
            TypeError: If neither a key nor a level was given.
            InvalidArgumentError: If the list of keys is empty, which is a
                `ValueError` and is what pandas raises for it.
            NotImplementedError: If the shape is one of the five that are not
                written, or if a level was asked for.
        """
        _no_level(level)
        if by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        wanted = [by] if isinstance(by, str) else by
        if not isinstance(wanted, list) or not all(isinstance(name, str) for name in wanted):
            raise NotImplementedError(
                "by= has to be a column name or a list of them for now, because"
                " grouping by a column that is not in the frame, by a function or"
                " by a mapping all need somewhere to put the key that came from"
                " outside it"
            )
        if len(wanted) == 0:
            # `InvalidArgumentError` rather than a bare `ValueError`, which is
            # what pandas raises and what this used to raise, because `translate`
            # turns an untagged `ValueError` into a `RuntimeError` on the way out
            # of the generated method. The class here is a `ValueError` too, so a
            # caller who catches the pandas one still catches this.
            raise InvalidArgumentError("No group keys passed!")
        known = frame.columns
        for name in wanted:
            if name not in known:
                raise KeyError(name)
        return list(wanted)

    def _reduced(self, kind: str, param: float, columns: list[str] | None = None) -> DataFrame:
        """Runs one reduction over the groups and hands back the frame it makes.

        Both classes come through here and they differ in what they do with the
        answer, which is the whole of the difference between them.

        Args:
            kind: The reduction, as pandas spells the method.
            param: The delta degrees of freedom or the quantile, and zero for
                the rest.
            columns: The columns to reduce, or None for every one that is not a
                key. A `SeriesGroupBy` names one.

        Returns:
            The frame the core produced, one row per group.
        """
        from ._frame import DataFrame

        try:
            source = self._frame._inner
            if columns is not None:
                source = source.select(self._by + columns)
            return DataFrame._wrap(
                source.group_agg(self._by, kind, param, self._dropna, self._sort, self._as_index)
            )
        except Exception as error:
            raise translate(error) from None

    def _reduce(
        self,
        kind: str,
        param: float = 0.0,
        numeric_only: bool = False,
        skipna: bool = True,
        min_count: int | None = None,
        engine: Any = None,
        engine_kwargs: Any = None,
    ) -> Answer:
        """Runs one reduction over the groups, after refusing what is declared.

        Every one of the fifteen comes through here and the five arguments below
        are the ones pandas puts on their signatures and firepanda does not
        honour. They are declared rather than left out because the parity test
        compares the whole parameter list against a running pandas, and they
        raise rather than being ignored for the reason `_held_at` gives.

        The shaping is what the two classes differ in, so it is one call at the
        end and is written in each of them.

        Args:
            kind: The reduction, as pandas spells the method.
            param: The delta degrees of freedom or the quantile, and zero for
                the rest.
            numeric_only: Declared and held at False.
            skipna: Declared and held at True.
            min_count: Declared and held at its default, which is zero for `sum`
                and minus one for the four that pick a value rather than combine
                them. None for the reductions that do not have it.
            engine: Declared and refused.
            engine_kwargs: Declared and refused.

        Returns:
            The frame or the series pandas answers.
        """
        _held_at(
            "numeric_only",
            numeric_only,
            False,
            "dropping the columns a reduction cannot read is a decision about"
            " which columns come back, and firepanda reduces the ones it was"
            " given or says which one it could not",
        )
        _held_at(
            "skipna",
            skipna,
            True,
            "a group's answer is computed from the values that are there, and"
            " taking one missing value as a reason to answer nothing at all is a"
            " second pass the kernels do not make",
        )
        if min_count is not None:
            # pandas defaults this to zero for `sum` and to minus one for the
            # four that pick a value rather than combining them, so the value
            # that means "nobody asked for anything" depends on the reduction.
            _held_at(
                "min_count",
                min_count,
                0 if kind == "sum" else -1,
                "answering nothing for a group that is too small is a check on"
                " the count after the reduction, and the count is not kept",
            )
        _refuse(
            "engine",
            engine,
            "there is one implementation and it is the compiled one, so there is"
            " nothing here for this to choose between",
        )
        _refuse(
            "engine_kwargs",
            engine_kwargs,
            "there is nothing to configure while there is nothing to choose",
        )
        return self._shape(kind, param)

    def _spread(
        self,
        kind: str,
        ddof: int,
        numeric_only: bool,
        skipna: bool,
        engine: Any,
        engine_kwargs: Any,
    ) -> Answer:
        """Runs `std` or `var`, which carry a delta degrees of freedom.

        A door of its own only because the generated body has to fit on one
        line, and `float(ddof)` written there rather than here pushed the two
        widest of the fifteen past the line limit. The conversion is the whole
        of what it adds: the kinds store the parameter as a float because it is
        a number in a formula rather than a length.

        Args:
            kind: `std` or `var`.
            ddof: The delta degrees of freedom.
            numeric_only: Declared and held at False.
            skipna: Declared and held at True.
            engine: Declared and refused.
            engine_kwargs: Declared and refused.

        Returns:
            The frame or the series pandas answers.
        """
        return self._reduce(
            kind,
            float(ddof),
            numeric_only,
            skipna,
            None,
            engine,
            engine_kwargs,
        )

    def _nunique(self, dropna: bool) -> Answer:
        """Counts the distinct values in each group.

        Its own entry point because its `dropna` is not the group by's `dropna`
        and the two are easy to read as one. The group by's says whether a
        missing key is a group. This one says whether a missing value counts as
        one of the distinct values inside a group, and it is held at True, which
        is pandas' default and the kernel's behaviour.

        Args:
            dropna: Declared and held at True.

        Returns:
            The frame or the series pandas answers.
        """
        _held_at(
            "dropna",
            dropna,
            True,
            "counting a missing value as one of the distinct values in a group"
            " is a second thing for the kernel to carry and nothing asks for it",
        )
        return self._shape("nunique", 0.0)

    def _quantile(self, q: Any, interpolation: str, numeric_only: bool) -> Answer:
        """The value at one quantile within each group.

        `_quantile_wanted` is the same reader the whole column reductions use,
        so a list of quantiles is refused with the same sentence in both places
        and the four interpolation rules that are not linear are refused once.

        Args:
            q: The quantile, between zero and one.
            interpolation: How to land between two values.
            numeric_only: Declared and held at False.

        Returns:
            The frame or the series pandas answers.
        """
        return self._reduce(
            "quantile", _quantile_wanted(q, interpolation), numeric_only=numeric_only
        )

    def _shape(self, kind: str, param: float) -> Answer:
        """Runs the reduction and puts the answer in the shape pandas gives.

        Overridden in both subclasses and never called on this one.

        Args:
            kind: The reduction, as pandas spells the method.
            param: The delta degrees of freedom or the quantile.

        Returns:
            The frame or the series pandas answers.
        """
        raise NotImplementedError(kind)


class DataFrameGroupByMixin(GroupByMixin["DataFrame"]):
    """The hand written half of `DataFrameGroupBy`.

    A reduction here answers a frame, except `size`, which counts rows rather
    than reducing a column and therefore answers one number per group. pandas
    makes that a Series when the key is in the index and a two column frame when
    it is not, and both of those are here, because the shape is the answer as
    much as the numbers are.
    """

    __slots__ = ()

    def __getitem__(self, key: Any) -> DataFrameGroupBy | SeriesGroupBy:
        """Narrows the group by to one column or to several.

        `df.groupby("k")["v"]` is a `SeriesGroupBy` and `df.groupby("k")[["v"]]`
        is a `DataFrameGroupBy` over fewer columns, which is the same split
        `df[...]` makes and is why this is written rather than generated.

        Args:
            key: A column name, or a list of them.

        Returns:
            A `SeriesGroupBy` for a name and a `DataFrameGroupBy` for a list.

        Raises:
            KeyError: If a name is not a column.
        """
        from ._frame import DataFrame, DataFrameGroupBy, SeriesGroupBy

        names = [key] if isinstance(key, str) else list(key)
        for name in names:
            if name not in self._frame.columns:
                raise KeyError(name)
        narrowed = DataFrame._wrap(self._frame._inner.select(self._by + names))
        if isinstance(key, str):
            return SeriesGroupBy(narrowed, self._by, self._as_index, self._sort, self._dropna, key)
        return DataFrameGroupBy(narrowed, self._by, self._as_index, self._sort, self._dropna)

    def _shape(self, kind: str, param: float) -> DataFrame:
        """One reduction over every column that is not a key.

        Args:
            kind: The reduction, as pandas spells the method.
            param: The delta degrees of freedom or the quantile.

        Returns:
            The frame of one row per group.
        """
        return self._reduced(kind, param)

    def _size(self) -> DataFrame | Series:
        """Counts the rows in each group, which is the one reduction with two shapes.

        A door of its own rather than a branch in `_shape`, because it is the one
        of the fifteen that does not answer what the other fourteen answer.
        pandas makes it a series when the key is in the index, since the answer is
        one column either way, and leaves it a two column frame when the key is
        not, and the shape is as much the answer as the numbers are.

        Returns:
            A series of counts, or a frame of the keys and the counts.
        """
        out = self._reduced("size", 0.0)
        if not self._as_index:
            return out
        # The name goes because a pandas `size` has none, and the frame's one
        # column is called `size` here only because a column has to be called
        # something.
        return _relabelled(out, out.columns[0], "")


class SeriesGroupByMixin(GroupByMixin["DataFrame | Series"]):
    """The hand written half of `SeriesGroupBy`.

    The same reductions over one column, answering a series rather than a frame.
    The extra piece of state is the column's name, which the answer carries and
    the frame underneath does not: after `df.groupby("k")["v"].sum()` the series
    is called `v`, and by then the frame it came out of has one column called
    `v` and one called `k` and no way to say which of them the caller asked for.
    """

    __slots__ = ("_column",)
    """The column the reductions run over."""

    _column: str

    def __init__(
        self,
        frame: DataFrame,
        by: list[str],
        as_index: bool,
        sort: bool,
        dropna: bool,
        column: str,
    ):
        """Holds the frame, the plan and the column. Not a public entry point.

        Args:
            frame: The frame, already narrowed to the keys and the column.
            by: The key column names.
            as_index: Whether the key becomes the row labels.
            sort: Whether the groups come out in key order.
            dropna: Whether a missing key is a group.
            column: The column the reductions run over.
        """
        super().__init__(frame, by, as_index, sort, dropna)
        self._column = column

    def _shape(self, kind: str, param: float) -> DataFrame | Series:
        """One reduction over the one column.

        Args:
            kind: The reduction, as pandas spells the method.
            param: The delta degrees of freedom or the quantile.

        Returns:
            A series named after the column, or a frame when the key was asked
            for as a column rather than as labels.
        """
        out = self._reduced(kind, param, [self._column])
        if not self._as_index:
            return out
        # `size` answers a column called `size` rather than one called after the
        # column it counted, because it did not read that column. Either way
        # there is exactly one column left once the keys have gone into the
        # labels, so the answer is the column that is there.
        return _relabelled(out, out.columns[-1], "" if kind == "size" else self._column)


class IndexMixin:
    """The hand written half of `Index`."""

    __slots__ = ("_inner",)
    """The one piece of state, for the reason `DataFrameMixin` gives."""

    _inner: _firepanda.Index

    def __init__(
        self,
        data: Any = None,
        dtype: Any = None,
        copy: bool | None = None,
        name: Any = None,
        tupleize_cols: bool = True,
    ) -> None:
        """Builds an index from a sequence of labels.

        The pandas signature in full, with `data` and `name` honoured and the
        rest refused by name. `tupleize_cols` is the one parameter here that is
        not refused, because refusing it would mean refusing its default, and
        what it turns on is the MultiIndex that does not exist yet.
        """
        _refuse("dtype", dtype, "casting on the way in needs the cast machinery")
        _refuse("copy", copy, "there is exactly one behaviour and it always copies")
        if not tupleize_cols:
            raise NotImplementedError(
                "tupleize_cols=False is not supported yet, because there is no"
                " MultiIndex for it to turn off"
            )
        try:
            self._inner = _firepanda.Index(data, None if name is None else str(name))
        except Exception as error:
            raise translate(error) from None

    def __getitem__(self, key: Any) -> Any:
        """One label, or an index of several.

        Three keys wearing one name, which is why this is here rather than in
        the table. An integer takes a label out and gives back a Python value, a
        slice and a list both give back an index, and a list is read as
        positions or as a mask depending on what is in it.

        A slice is resolved against the length here rather than in Mojo, because
        `slice.indices` is the definition of what a Python slice means and
        writing a second one that agrees with it is work with nothing to gain.
        """
        from ._frame import Index

        try:
            if isinstance(key, bool):
                raise TypeError("cannot index an index with a bool; pass a list of them")
            if isinstance(key, int):
                return self._inner.at(key)
            if isinstance(key, slice):
                start, stop, step = key.indices(self._inner.length())
                if step == 1:
                    return Index._wrap(self._inner.slice_rows(start, max(start, stop)))
                return Index._wrap(self._inner.take(list(range(start, stop, step))))
            if isinstance(key, (list, tuple)):
                picks = list(key)
                if picks and all(isinstance(k, bool) for k in picks):
                    return Index._wrap(
                        self._inner.take([i for i, keep in enumerate(picks) if keep])
                    )
                return Index._wrap(self._inner.take([int(k) for k in picks]))
        except Exception as error:
            raise translate(error) from None
        raise TypeError(
            f"cannot index an index with a {type(key).__name__}; index[key] reads"
            " a position, a slice, a list of positions or a list of bools"
        )

    def __iter__(self) -> Any:
        """The labels, one at a time.

        A copy of the whole list rather than a cursor into the index, because a
        cursor would have to keep the index alive across the loop body and the
        list already does that. An index being iterated is an index small enough
        for a person to look at.
        """
        try:
            return iter(self._inner.to_list())
        except Exception as error:
            raise translate(error) from None

    def __contains__(self, key: Any) -> bool:
        """Whether a label is in the index.

        pandas answers False for a key of the wrong type rather than raising,
        because `in` is a question and not a lookup, and this does the same.
        """
        try:
            return self._inner.contains(key)
        except Exception as error:
            raise translate(error) from None

    def __bool__(self) -> bool:
        """Refuses, in the same words pandas refuses in.

        An index of three labels is neither true nor false, and Python's default
        would make it true because it has a length. pandas raises rather than
        letting `if index:` mean something the writer did not intend, and the
        message is copied exactly because it is the message people search for.
        """
        raise ValueError(
            "The truth value of a Index is ambiguous. Use a.empty, a.bool(),"
            " a.item(), a.any() or a.all()."
        )

    def __eq__(self, other: Any) -> Any:
        """Elementwise comparison, against a scalar or against a sequence.

        pandas gives back a numpy array of bools here and this gives back a list
        of them, which is the same divergence `values` has and is recorded in
        document 21. What it is not is `Index.equals`, which asks whether two
        indexes are the same and gives back one bool.

        Defining this makes the class unhashable, which is what Python does when
        a class defines `__eq__` and not `__hash__`, and is what pandas does too.
        """
        mine = self._inner.to_list()
        if isinstance(other, IndexMixin):
            other = other._inner.to_list()
        if isinstance(other, (list, tuple)):
            if len(other) != len(mine):
                raise ValueError(f"lengths must match to compare: {len(mine)} and {len(other)}")
            return [a == b for a, b in zip(mine, other, strict=True)]
        return [a == other for a in mine]

    def __ne__(self, other: Any) -> Any:
        """Elementwise inequality, which is `__eq__` turned over."""
        return [not answer for answer in self.__eq__(other)]

    def copy(self, name: Any = None, deep: bool = False) -> Index:
        """The index again, under a new name if one is given.

        `deep` is accepted and ignored, which is the one place in the library
        that happens. An index is immutable once built and a copy of it can only
        be observed through `is_`, so the deep copy and the shallow one are the
        same object as far as anything a caller can write is concerned. pandas
        documents `deep` as having no effect on an index for the same reason.
        """
        from ._frame import Index

        try:
            wanted = self._inner.label() if name is None else str(name)
            return Index._wrap(self._inner.renamed(wanted))
        except Exception as error:
            raise translate(error) from None

    def get_loc(self, key: Any) -> Any:
        """Where a label is, as an integer, a slice or a mask.

        pandas returns three different types from this one method and which one
        it returns depends on the labels rather than on the argument: one hit is
        an integer, several hits in a row on a sorted index are a slice, and
        anything else is a boolean mask. Callers branch on the type, so getting
        the rule right matters more than it looks.
        """
        try:
            found = list(self._inner.get_loc(key))
        except Exception as error:
            raise translate(error) from None
        if len(found) == 1:
            return found[0]
        run = found[-1] - found[0] + 1 == len(found)
        if run and self._inner.is_monotonic_increasing():
            return slice(found[0], found[-1] + 1, None)
        return [i in set(found) for i in range(self._inner.length())]

    def get_indexer(
        self,
        target: Any,
        method: Any = None,
        limit: Any = None,
        tolerance: Any = None,
    ) -> list[int]:
        """Where each of a set of labels sits, with -1 for the ones that are not there.

        The three parameters after `target` are the ones that fill a missing
        label in from a neighbour, and they are refused rather than ignored. This
        is only defined on a unique index, which pandas also insists on, because
        one position per label asked for is not an answer an index with
        duplicates has.
        """
        _refuse("method", method, "filling a missing label from a neighbour is not written")
        _refuse("limit", limit, "there is no filling for it to limit")
        _refuse("tolerance", tolerance, "there is no filling for it to bound")
        try:
            return list(self._inner.get_indexer(target))
        except Exception as error:
            raise translate(error) from None

    def equals(self, other: Any) -> bool:
        """Whether two indexes hold the same labels in the same order.

        The name is ignored, which is what pandas does and is the difference
        between this and `identical`. Anything that is not an index is not equal
        to one, and that is False rather than an error.
        """
        if not isinstance(other, IndexMixin):
            return False
        try:
            return self._inner.equals(other._inner)
        except Exception as error:
            raise translate(error) from None

    def identical(self, other: Any) -> bool:
        """Whether the labels and the name both match."""
        if not isinstance(other, IndexMixin):
            return False
        try:
            return self._inner.identical(other._inner)
        except Exception as error:
            raise translate(error) from None

    def is_(self, other: Any) -> bool:
        """Whether two indexes are the same object underneath.

        Not `is`, which compares the wrappers, and not `equals`, which compares
        the labels. This is the question of whether a copy was taken, and the
        answer comes from the address of the shared index rather than from
        anything visible on this side.
        """
        if not isinstance(other, IndexMixin):
            return False
        try:
            return self._inner.same_as(other._inner)
        except Exception as error:
            raise translate(error) from None

    def append(self, other: Any) -> Index:
        """One index, or several, put on the end of this one."""
        from ._frame import Index

        others = other if isinstance(other, (list, tuple)) else [other]
        try:
            return Index._wrap(self._inner.append([_unwrap(o, "other") for o in others]))
        except Exception as error:
            raise translate(error) from None

    def delete(self, loc: Any) -> Index:
        """The index without the labels at one position, or at several."""
        from ._frame import Index

        picks = list(loc) if isinstance(loc, (list, tuple)) else [loc]
        try:
            return Index._wrap(self._inner.delete([int(i) for i in picks]))
        except Exception as error:
            raise translate(error) from None

    def drop(self, labels: Any, errors: str = "raise") -> Index:
        """The index without every row carrying one of a set of labels.

        A scalar label is wrapped in a list here rather than in Mojo, because a
        string is a sequence in Python and telling a label apart from a list of
        them is a Python question.
        """
        from ._frame import Index

        wanted = labels if isinstance(labels, (list, tuple)) else [labels]
        if isinstance(labels, IndexMixin):
            wanted = labels._inner.to_list()
        try:
            return Index._wrap(self._inner.drop(list(wanted), errors))
        except Exception as error:
            raise translate(error) from None

    def putmask(self, mask: Any, value: Any) -> Index:
        """The index with the labels a mask picks out replaced.

        The replacement is one label or a whole column of them, and a scalar is
        wrapped in a list here rather than in Mojo for the reason `drop` gives:
        a string is a sequence in Python and telling a label apart from a list
        of them is a Python question.
        """
        from ._frame import Index

        replacement = value if isinstance(value, (list, tuple)) else [value]
        try:
            return Index._wrap(self._inner.putmask([bool(m) for m in mask], list(replacement)))
        except Exception as error:
            raise translate(error) from None

    def slice_indexer(self, start: Any = None, end: Any = None, step: Any = None) -> slice:
        """The slice a pair of labels describes, with both ends included.

        The one range in the library that is not half open, because label based
        slicing in pandas includes its end and a caller who writes
        `df.loc["b":"d"]` means through d rather than up to it.
        """
        try:
            first, last, stride = self._inner.slice_indexer(
                start, end, 1 if step is None else int(step)
            )
        except Exception as error:
            raise translate(error) from None
        return slice(first, last, None if step is None else stride)

    def union(self, other: Any, sort: bool | None = None) -> Index:
        """Every label either side has.

        `sort=None` means sort, which is the pandas default here and is not the
        pandas default for `intersection`. The three way argument is turned into
        a bool on this side so that the core takes a bool and means it.
        """
        return self._set_operation("union", other, True if sort is None else bool(sort))

    def intersection(self, other: Any, sort: bool = False) -> Index:
        """Every label both sides have.

        Defaults to not sorting, which keeps this index's order, because an
        intersection is a filter of the left side and has an order to inherit.
        """
        return self._set_operation("intersection", other, False if sort is None else bool(sort))

    def difference(self, other: Any, sort: bool | None = None) -> Index:
        """Every label this index has and the other does not."""
        return self._set_operation("difference", other, True if sort is None else bool(sort))

    def symmetric_difference(
        self, other: Any, result_name: Any = None, sort: bool | None = None
    ) -> Index:
        """Every label exactly one side has.

        The only one of the four that names its result, because there is no
        left side for the name to come from when both sides contributed equally.
        """
        from ._frame import Index

        try:
            return Index._wrap(
                self._inner.symmetric_difference(
                    _unwrap(other, "other"),
                    True if sort is None else bool(sort),
                    None if result_name is None else str(result_name),
                )
            )
        except Exception as error:
            raise translate(error) from None

    def _set_operation(self, which: str, other: Any, sort: bool) -> Index:
        """Runs one of the three set operations that share a signature.

        Not public, and here rather than repeated three times, because the only
        thing that differs between them is the name of the call.
        """
        from ._frame import Index

        try:
            return Index._wrap(getattr(self._inner, which)(_unwrap(other, "other"), sort))
        except Exception as error:
            raise translate(error) from None


def _unwrap(value: Any, name: str) -> Any:
    """Takes the extension object out of an index, building one if it has to.

    Every method that takes another index takes the extension object rather
    than the wrapper, because the Mojo side can only downcast to a type it
    knows. pandas accepts a plain list wherever it accepts an index, so a list
    is turned into an index here rather than being refused.

    Args:
        value: The index, or something an index can be made of.
        name: The parameter name, for the message.

    Returns:
        The extension object.

    Raises:
        TypeError: If it is neither.
    """
    if isinstance(value, IndexMixin):
        return value._inner
    if isinstance(value, (list, tuple)):
        return _firepanda.Index(list(value), None)
    raise TypeError(f"{name} must be an Index or a list of labels, not a {type(value).__name__}")


def to_datetime(
    arg: Any,
    errors: str = "raise",
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: Any = NO_DEFAULT,
    unit: str | None = None,
    origin: Any = "unix",
    cache: bool = True,
) -> Any:
    """Reads text or whole numbers as instants, which is `pandas.to_datetime`.

    Hand written rather than generated for the reason the top of this file
    gives: what it does depends on its arguments. Ten of them are declared,
    four are implemented, one is accepted and has no effect, and five are
    refused by name, which is the pattern `_refuse` exists for.

    What it answers is a `Series` and pandas answers a `DatetimeIndex` when it
    is handed a list. That is the one difference a caller will meet on the
    first line they write, and it is not hidden: firepanda's `Index` is a
    labels object with none of the calendar members a `DatetimeIndex` carries,
    so answering one would be a name that resolves and then has nothing on it,
    which document 07 argues is worse than a name that resolves to something
    honest. See #354.

    The format is worked out from the first row that is not missing, and only
    ISO 8601 is recognised. pandas guesses more than that, including
    `01/02/2026`, and decides for itself which of the two numbers is the month.
    A wrong guess there is a column of instants that are wrong by up to eleven
    months and that nothing anywhere reports, so firepanda refuses the shapes
    it does not recognise and names the value in the message. Passing `format`
    reads anything, including the shapes the guesser will not touch.

    Args:
        arg: The values. A firepanda series, or anything a series is built
            from, such as a list of strings or a list of whole numbers.
        errors: `raise` to stop on the first row that will not read, or
            `coerce` to turn that row into a missing one.
        dayfirst: Refused. Only ISO 8601 is guessed and it has one order.
        yearfirst: Refused, for the same reason.
        utc: Whether to read every row against UTC, which is the only way a
            column carrying more than one offset can be read at all.
        format: The format the text is written in, or None to work it out.
        exact: Refused. It is a question about a regular expression search that
            this parser does not do.
        unit: What whole numbers are counts of, as one of `s`, `ms`, `us` and
            `ns`. Ignored for text, which pandas ignores it for too.
        origin: Refused at anything other than `unix`.
        cache: Accepted and has no effect. pandas caches repeated values to go
            faster and the answer is the same either way, so honouring the
            parameter means not changing the answer.

    Returns:
        A series of instants, null where the input was null and, under
        `errors="coerce"`, wherever a row would not read.

    Raises:
        NotImplementedError: For the five refused arguments and for a format
            firepanda's guesser does not recognise.
        ValueError: For a row that does not match the format, for a column
            carrying more than one offset with no `utc`, and for an `errors`
            that is neither of the two words.
    """
    from ._frame import Series

    _held_at("dayfirst", dayfirst, False, "firepanda guesses ISO 8601 and nothing else")
    _held_at("yearfirst", yearfirst, False, "firepanda guesses ISO 8601 and nothing else")
    _held_at("origin", origin, "unix", "an epoch other than 1970 has to move every value")
    if exact is not NO_DEFAULT:
        raise NotImplementedError(
            "exact= is not supported yet, because it asks whether the format may match"
            " part of the value, and this parser reads the whole of it or none of it"
        )
    if format in ("mixed", "ISO8601"):
        raise NotImplementedError(
            f"format={format!r} is not supported yet, because it asks for the format to"
            " be worked out per row, and firepanda works one out from the first row and"
            " holds every other row to it"
        )
    if errors not in ("raise", "coerce"):
        raise ValueError(f"errors must be one of 'raise' or 'coerce', not {errors!r}")

    column = arg if isinstance(arg, SeriesMixin) else Series(arg)
    try:
        return Series._wrap(
            column._inner.to_datetime(
                "" if format is None else format,
                "ns" if unit is None else unit,
                errors == "coerce",
                utc,
            )
        )
    except Exception as error:
        raise translate(error) from None
