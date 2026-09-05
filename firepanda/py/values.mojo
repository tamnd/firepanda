"""Reading Arrow values back out into Python objects.

`build.mojo` is this file in the other direction and the two are a pair. What
crosses here is one value or a whole column, always copied, always as ordinary
Python objects rather than as anything a consumer would need a library to read.

This is the slow way out of a column and it is not apologised for, because it is
the only way out for anything that is not another Arrow consumer. `s.tolist()`,
`index[0]` and `list(index)` all land here, and all three are how a person reads
a value at a prompt. The fast way is `__arrow_c_array__` and it hands over
pointers rather than objects.

### A missing value is `None`

pandas would have widened an integer column with a missing value to float64 and
put a NaN in the hole, because a numpy int64 array has nowhere to record absence.
A firepanda column is Arrow and has a validity bitmap, so the value is missing
rather than approximated, and `None` is what says that. It is also what
`pyarrow.Array.to_pylist` and `polars.Series.to_list` both return, so the
surprising answer here would be the pandas one.
"""

from std.python import Python, PythonObject

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.dispatch import dispatch_typed
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import TypeKind


def _numbers[dt: DType](values: Array[dt]) raises -> PythonObject:
    """Turns one typed column into a Python list.

    Args:
        values: The column, borrowed rather than copied.

    Parameters:
        dt: The column's dtype, bound by the dispatch.

    Returns:
        A list with one element per row.
    """
    var out = Python.list()
    for i in range(len(values)):
        if not values.is_valid(i):
            out.append(Python.none())
        elif dt == DType.bool:
            out.append(PythonObject(Bool(values[i])))
        elif dt.is_floating_point():
            out.append(PythonObject(Float64(values[i].cast[DType.float64]())))
        else:
            out.append(PythonObject(Int(values[i].cast[DType.int64]())))
    return out


def python_list(column: AnyArray) raises -> PythonObject:
    """Copies every value of a column out into a Python list.

    Three kinds of column are handled separately. Text is not dispatched over
    `ALL` because a string column is physically uint8 and would come back as a
    list of bytes. A column of the null type has no buffer at all, so there is
    nothing to read and every row is missing. Everything else is one typed pass.

    Args:
        column: The values.

    Returns:
        A list with one element per row, with `None` for the missing ones.

    Raises:
        Error: If the column's dtype cannot be read.
    """
    if column.type.kind == TypeKind.NULL:
        var out = Python.list()
        for _ in range(len(column)):
            out.append(Python.none())
        return out
    if column.is_string():
        var out = Python.list()
        for i in range(len(column)):
            if column.is_valid(i):
                out.append(PythonObject(column.strings()[i]))
            else:
                out.append(Python.none())
        return out
    return dispatch_typed[ALL](column, _numbers)


def python_value(column: AnyArray, i: Int) raises -> PythonObject:
    """Copies one value of a column out as a Python object.

    Written as a slice and a list rather than as a fourth copy of the dispatch
    above. A one row slice of an Arrow array is a view with an offset on it and
    allocates nothing, so what this costs over a hand written accessor is a
    `PythonObject` for the list that holds the answer, and what it buys is that
    the null rule and the string rule are written once.

    Args:
        column: The values.
        i: The row, which the caller has already bounds checked.

    Returns:
        The value, or `None` if it is missing.

    Raises:
        Error: If the column's dtype cannot be read.
    """
    return python_list(column.slice(i, i + 1))[0]
