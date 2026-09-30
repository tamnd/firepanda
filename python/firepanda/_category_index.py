"""An index of categories, which is pandas' `CategoricalIndex`.

The core holds category labels as codes into a list of categories, and it
cannot yet look a label up among them, find its duplicates or drop it. What is
here answers those by the labels the codes stand for, and takes rows by
position so the categories stay as they were. The methods that change the
categories are the ones `Categorical` has, and give back an index.

An index made by the core from a category column becomes a `CategoricalIndex`
by `_class_of`, so `isinstance` and the category methods work on the row
labels of a frame grouped by a category as well.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

from ._categorical import Categorical, _categorical_column
from ._frame import Index

__all__ = ["CategoricalIndex"]


def _listed(labels: Any) -> list[Any]:
    """Labels handed in as one label, a list or an index, as a list."""
    if isinstance(labels, str) or not hasattr(labels, "__iter__"):
        return [labels]
    return list(labels.tolist() if hasattr(labels, "tolist") else labels)


class CategoricalIndex(Index):
    """An `Index` whose labels are drawn from a list of categories."""

    __slots__ = ()

    def __init__(
        self,
        data: Any = None,
        categories: Any = None,
        ordered: Any = None,
        dtype: Any = None,
        copy: bool = False,
        name: Any = None,
    ) -> None:
        """Builds an index of categories, as pandas' `CategoricalIndex` does.

        Raises:
            ValueError: For `dtype` given with `categories` or `ordered`, as in pandas.
        """
        if data is None:
            data = []
        column = _categorical_column(data, categories, ordered, dtype, "CategoricalIndex")
        if name is None:
            name = getattr(data, "name", None)
        Index.__init__(self, column, name=name)
        self.__class__ = CategoricalIndex

    def _held(self) -> Categorical:
        """The labels as a `Categorical`, categories and order kept."""
        return Categorical._held_by(self.to_series())

    def _of(self, values: Categorical) -> CategoricalIndex:
        """An index of these categorical values under this index's name."""
        return CategoricalIndex(values, name=self.name)

    def _decoded(self) -> Index:
        """The labels the codes stand for, as a plain index."""
        return Index(self.tolist(), name=self.name)

    def _taken(self, positions: list[int]) -> CategoricalIndex:
        return self._of(self._held()[positions])

    @property
    def values(self) -> Categorical:
        """The labels as a `Categorical`."""
        return self._held()

    def get_loc(self, key: Any) -> Any:
        """Where a label is: a position, or a mask when it is there more than once.

        Raises:
            KeyError: For a label that is not there.
        """
        return self._decoded().get_loc(key)

    def get_indexer(
        self, target: Any, method: Any = None, limit: Any = None, tolerance: Any = None
    ) -> Any:
        """Where each of a set of labels sits, -1 for the ones that are not there."""
        return self._decoded().get_indexer(_listed(target), method, limit, tolerance)

    def __contains__(self, key: Any) -> bool:
        try:
            self.get_loc(key)
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def duplicated(self, keep: Any = "first") -> Any:
        """Whether each label was seen before, by the label rather than its code."""
        return self._decoded().duplicated(keep=keep)

    @property
    def is_unique(self) -> bool:
        """Whether no label is there twice."""
        return self._decoded().is_unique

    @property
    def has_duplicates(self) -> bool:
        """Whether some label is there twice."""
        return not self.is_unique

    def _codes_rise(self, codes: list[int]) -> bool:
        return all(a <= b for a, b in itertools.pairwise(codes))

    @property
    def is_monotonic_increasing(self) -> bool:
        """Whether the labels are in the order of their categories."""
        return self._codes_rise(self.codes.tolist())

    @property
    def is_monotonic_decreasing(self) -> bool:
        """Whether the labels are in the reverse order of their categories."""
        return self._codes_rise(self.codes.tolist()[::-1])

    def unique(self, level: Any = None) -> CategoricalIndex:
        """Each label once, in the order first seen, the categories kept."""
        seen = list(self.duplicated())
        return self._taken([at for at, again in enumerate(seen) if not again])

    def drop(self, labels: Any, errors: str = "raise") -> CategoricalIndex:
        """Without the given labels, the categories kept.

        Raises:
            KeyError: For a label that is not there, unless `errors="ignore"`.
        """
        listed = _listed(labels)
        values = self.tolist()
        if errors != "ignore":
            missing = [label for label in listed if label not in values]
            if missing:
                raise KeyError(f"{missing} not found in axis")
        return self._taken([at for at, value in enumerate(values) if value not in listed])

    def union(self, other: Any, sort: bool | None = None) -> Index:
        """The labels of both, a category index when both have the same categories."""
        joined = self._decoded().union(Index(_listed(other)), sort=sort)
        if isinstance(other, CategoricalIndex) and other.dtype == self.dtype:
            return CategoricalIndex(joined.tolist(), dtype=self.dtype, name=self.name)
        return joined

    def reindex(self, target: Any, method: Any = None, level: Any = None, **kwargs: Any) -> Any:
        """The target labels and where each sits here, -1 for the ones that are not.

        Raises:
            ValueError: For an index that has a label twice, in pandas' words.
        """
        return self._decoded().reindex(target, method=method, level=level, **kwargs)

    def map(self, mapper: Any, na_action: Any = None) -> Index:
        """Each category through a function or a mapping.

        When the new categories are all different the answer is still a category
        index, in the same order, as in pandas; otherwise it is a plain index.
        """
        held = self.categories.tolist()
        if callable(mapper):
            renamed = [mapper(category) for category in held]
        else:
            renamed = [mapper.get(category, math.nan) for category in held]
        if len(set(renamed)) == len(renamed) and all(value == value for value in renamed):
            return self.rename_categories(renamed)
        codes = self.codes.tolist()
        return Index([math.nan if at < 0 else renamed[at] for at in codes], name=self.name)

    def as_ordered(self) -> CategoricalIndex:
        """The same labels with the categories in order."""
        return self._of(self._held().as_ordered())

    def as_unordered(self) -> CategoricalIndex:
        """The same labels with the categories in no order."""
        return self._of(self._held().as_unordered())

    def add_categories(self, new_categories: Any) -> CategoricalIndex:
        """New categories on the end of the ones there are."""
        return self._of(self._held().add_categories(new_categories))

    def remove_categories(self, removals: Any) -> CategoricalIndex:
        """Without the named categories, each label in them a gap."""
        return self._of(self._held().remove_categories(removals))

    def remove_unused_categories(self) -> CategoricalIndex:
        """Without the categories no label is in."""
        return self._of(self._held().remove_unused_categories())

    def rename_categories(self, new_categories: Any) -> CategoricalIndex:
        """The categories under new names, each label where it was."""
        return self._of(self._held().rename_categories(new_categories))

    def reorder_categories(self, new_categories: Any, ordered: Any = None) -> CategoricalIndex:
        """The same categories in another order."""
        return self._of(self._held().reorder_categories(new_categories, ordered=ordered))

    def set_categories(
        self, new_categories: Any, ordered: Any = None, rename: bool = False
    ) -> CategoricalIndex:
        """Another list of categories, each label not in it a gap."""
        return self._of(self._held().set_categories(new_categories, ordered=ordered, rename=rename))
