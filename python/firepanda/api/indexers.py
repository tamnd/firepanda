"""`firepanda.api.indexers`, the window bounds a rolling window can be given.

A rolling window is usually a count of rows behind each row. pandas also takes
an object that says where each window starts and ends, which is how a window
that looks forward, or one that follows a calendar, is written. The object is a
`BaseIndexer`, and its one method, `get_window_bounds`, answers two arrays of
row numbers: where each window starts and where it stops, the stop left out.
`DataFrame.rolling` and `Series.rolling` read those bounds and reduce each
window, which is what pandas does with them.
"""

from __future__ import annotations

from typing import Any

from ..errors import InvalidArgumentError

__all__ = [
    "BaseIndexer",
    "FixedForwardWindowIndexer",
    "VariableOffsetWindowIndexer",
    "check_array_indexer",
]


class BaseIndexer:
    """Window bounds a rolling window reads, which is `pandas.api.indexers.BaseIndexer`.

    Every keyword given is kept as an attribute, as in pandas, so that a
    subclass's `get_window_bounds` can read it.
    """

    def __init__(self, index_array: Any = None, window_size: int = 0, **kwargs: Any) -> None:
        self.index_array = index_array
        self.window_size = window_size
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[Any, Any]:
        """Where each window starts and stops, as two numpy arrays; a subclass writes it.

        Raises:
            NotImplementedError: Always, as in pandas.
        """
        raise NotImplementedError


class FixedForwardWindowIndexer(BaseIndexer):
    """Each window the row and the `window_size` minus one rows after it."""

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[Any, Any]:
        """Each row's window start and stop, as numpy arrays.

        Raises:
            ValueError: For `center` or `closed`, in pandas' words.
        """
        import numpy

        if center:
            raise InvalidArgumentError("Forward-looking windows can't have center=True")
        if closed is not None:
            raise InvalidArgumentError(
                "Forward-looking windows don't support setting the closed argument"
            )
        start = numpy.arange(0, num_values, step or 1, dtype=numpy.int64)
        end = start + self.window_size
        if self.window_size:
            end = numpy.clip(end, 0, num_values)
        return start, end


class VariableOffsetWindowIndexer(BaseIndexer):
    """Each window the rows whose label is within `offset` before the row's label."""

    def __init__(
        self,
        index_array: Any = None,
        window_size: int = 0,
        index: Any = None,
        offset: Any = None,
        **kwargs: Any,
    ) -> None:
        from .._datetime import DatetimeIndex
        from ..offsets import BaseOffset

        super().__init__(index_array, window_size, **kwargs)
        if not isinstance(index, DatetimeIndex):
            raise InvalidArgumentError("index must be a DatetimeIndex.")
        if not isinstance(offset, BaseOffset):
            raise InvalidArgumentError("offset must be a DateOffset-like object.")
        self.index = index
        self.offset = offset

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[Any, Any]:
        """Each row's window start and stop along a rising index, as numpy arrays.

        Raises:
            ValueError: For `center` or a `step`, in pandas' words.
        """
        import numpy

        if center:
            raise InvalidArgumentError("We do not support center=True with offsets")
        if step is not None:
            raise InvalidArgumentError("VariableOffsetWindowIndexer does not support step")
        labels = list(self.index)[:num_values]
        closed = closed or "right"
        start = numpy.zeros(num_values, dtype=numpy.int64)
        end = numpy.zeros(num_values, dtype=numpy.int64)
        for at, label in enumerate(labels):
            edge = label - self.offset
            first = 0
            while first < at and (
                labels[first] < edge or (labels[first] == edge and closed in ("right", "neither"))
            ):
                first += 1
            start[at] = first
            end[at] = at + 1 if closed in ("right", "both") else at
        return start, end


def check_array_indexer(array: Any, indexer: Any) -> Any:
    """`indexer` checked against `array` and given as a numpy array, as pandas checks it.

    A missing flag is read as False. A scalar comes back as it is.

    Raises:
        IndexError: For flags of the wrong length, or values neither whole nor flags.
        ValueError: For whole numbers with a gap.
    """
    import numpy

    from .._array import FirepandaArray
    from .._frame import Index, Series
    from .._pandas import _missing

    if isinstance(indexer, str | bytes) or not hasattr(indexer, "__iter__"):
        return indexer
    if isinstance(indexer, FirepandaArray | Series | Index):
        kind = str(indexer.dtype)
        values = indexer.tolist()
        gaps = [_missing(value) for value in values]
        if kind == "boolean":
            flags = [False if gap else value for value, gap in zip(values, gaps, strict=True)]
            indexer = numpy.array(flags, dtype=bool)
        elif any(gaps):
            if kind.startswith(("Int", "UInt")):
                raise InvalidArgumentError(
                    "Cannot index with an integer indexer containing NA values"
                )
            indexer = numpy.array(values)
        else:
            indexer = numpy.asarray(values)
    else:
        indexer = numpy.asarray(list(indexer))
        if len(indexer) == 0:
            return numpy.array([], dtype=numpy.intp)
    if indexer.dtype.kind == "b":
        if len(indexer) != len(array):
            raise IndexError(
                f"Boolean index has wrong length: {len(indexer)} instead of {len(array)}"
            )
        return indexer
    if indexer.dtype.kind in "iu":
        return indexer
    raise IndexError("arrays used as indices must be of integer or boolean type")
