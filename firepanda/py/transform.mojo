"""The twelve transformations that answer a whole column, behind one name.

Same argument as `reduce.mojo` and the same shape of answer. A reduction folds a
column down to one number and a transformation hands back a column of the same
kind, and in both cases pandas spells a dozen of them as a dozen method names
while the core spells them as a dozen functions that differ by a word. Twelve
bound methods across the language boundary would have been twelve things that
can drift from each other, so there is one door and the word crosses it.

### Why an integer rides along beside the name

`shift`, `diff` and `pct_change` take a `periods`, and `ffill` and `bfill` take a
`limit`. Both are a whole number, neither is ever anything else, and no
transformation in this list takes two of them, so one integer beside the name
carries every argument there is. The seven that take neither are handed a zero
and ignore it.

That the two arguments have different names on the pandas side is deliberately
not modelled here. This module is the crossing and not the API, and the crossing
cares that a number arrived rather than what a caller was invited to call it.
The Python layer is where `periods` and `limit` are two separate parameters with
their own defaults and their own documentation, which is the right place for a
distinction only a caller can see.

### Why a limit of zero means no limit

The core reads `fill_forward(limit=0)` as no limit and pandas reads
`ffill(limit=None)` as no limit, and zero is a perfectly good spelling of "as
far as it goes" as long as somebody says so once. This is that sentence. The
Python layer turns a `None` into a zero on the way in, so a caller never has to
know, and `limit=0` from a caller who meant it literally is refused up there
rather than being silently read as its opposite.

### What is not here

`fillna` takes a value rather than a number, `astype` takes a type, `rename`
takes a name, `sort_values` takes an order and a null position, and `take` takes
a list. None of them fits an integer and each is its own door. The two monotonic
questions answer a bool rather than a column and are a different door again, for
the same reason a reduction is not a transformation: the shape of the answer is
what decides which door something goes through, not how similar the words are.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.frame.index import Index
from firepanda.frame.series import Series
from firepanda.py.errors import VALUE, tagged


def transformation(name: String) raises -> String:
    """Checks that a word names a transformation, and hands it back.

    Separate from `transformed` for the reason `reduce.mojo` keeps `reduction`
    separate from the reduction: the caller checks the word before it opens the
    handler that turns a kernel complaint into a dtype error, so a name nobody
    implements comes back tagged as the value error it is rather than as a
    complaint about the column's type. A word that crossed the boundary is a
    value the caller chose and a type is not.

    Args:
        name: The word that crossed, as pandas spells the method.

    Returns:
        The same word.

    Raises:
        Error: Tagged `value` if it is not one of the twelve.
    """
    if (
        name == "dropna"
        or name == "isna"
        or name == "notna"
        or name == "ffill"
        or name == "bfill"
        or name == "shift"
        or name == "diff"
        or name == "pct_change"
        or name == "cumsum"
        or name == "cumprod"
        or name == "cummax"
        or name == "cummin"
    ):
        return name
    raise tagged(VALUE, String("unknown transformation ", name))


def transformed(column: Series, kind: String, periods: Int) raises -> Series:
    """Applies one named transformation to a column.

    Args:
        column: The column to transform.
        kind: The transformation, as pandas spells the method.
        periods: The `periods` or the `limit`, and zero for the seven that take
            neither.

    Returns:
        A new column. Every one of these but `dropna` is as tall as the one it
        read, and `dropna` is shorter by the number of missing rows.

    Raises:
        Error: Tagged `value` if the name is not one of the twelve, and whatever
            the kernel raises for a column whose type the transformation cannot
            read.
    """
    if kind == "dropna":
        return column.drop_nulls()
    if kind == "isna":
        return masked(column, column.is_null())
    if kind == "notna":
        return masked(column, column.is_not_null())
    if kind == "ffill":
        return column.fill_forward(periods)
    if kind == "bfill":
        return column.fill_backward(periods)
    if kind == "shift":
        return column.shift(periods)
    if kind == "diff":
        return column.diff(periods)
    if kind == "pct_change":
        return column.pct_change(periods)
    if kind == "cumsum":
        return column.cumsum()
    if kind == "cumprod":
        return column.cumprod()
    if kind == "cummax":
        return column.cummax()
    if kind == "cummin":
        return column.cummin()
    raise tagged(VALUE, String("unknown transformation ", kind))


def masked(column: Series, var mask: Array[DType.bool]) raises -> Series:
    """Puts a bool mask back on the column's name and labels.

    `is_null` answers a bare `Array` on purpose, because a mask is almost always
    handed straight to `filter` and a round trip through a named column would be
    in the way. `isna` is the one caller that wants the column back, since a
    pandas program writes `s.isna()` to look at the answer rather than to feed
    it to something, and a mask with no labels cannot be lined up against the
    rows it describes.

    This had an underscore on it while `isna` and `notna` were the only two
    callers, and it lost the underscore when `isin` became the third. `isin`
    lives in `series.mojo` rather than here because it takes a second column and
    everything in this file reads one, so the helper had to cross a module and a
    name with an underscore on it is a name that says not to.

    Args:
        column: The column the mask was taken from, read for its name and
            labels.
        mask: The mask.

    Returns:
        A bool column with the name and labels of the one it describes.

    Raises:
        Error: Only what building a series raises.
    """
    var out = Series(column.name, AnyArray(mask^))
    out.index = Index(copy=column.index)
    return out^
