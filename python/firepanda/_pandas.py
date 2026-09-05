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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _firepanda
from .errors import translate

if TYPE_CHECKING:
    from ._frame import DataFrame, Index, Series


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


__all__ = ["DataFrameMixin", "IndexMixin", "SeriesMixin"]


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
        """Builds a series from a sequence of values.

        Same shape as the frame constructor and refusing the same way, with the
        one difference that `name` is honoured, since a series carries its name
        and there is nothing to implement.
        """
        _refuse("index", index, "putting labels on a series as it is built is not written")
        _refuse("dtype", dtype, "casting on the way in needs the cast machinery")
        _refuse("copy", copy, "there is exactly one behaviour and it always copies")
        try:
            self._inner = _firepanda.Series(data, "" if name is None else str(name))
        except Exception as error:
            raise translate(error) from None


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
