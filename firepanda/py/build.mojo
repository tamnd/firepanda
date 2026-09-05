"""Turning Python values into Arrow columns.

`pd.DataFrame({"a": [1, 2, 3]})` is the first line of every pandas tutorial, and
until this file existed the answer to it was that firepanda can read a frame off
disk or take one from another library and cannot make one. That is a strange
thing to require of a library whose pitch is compatibility, because it means
nobody can try it without a CSV in hand.

The whole file is one pass to classify and one to fill, and it is slow in the way
that reading a Python object per element is always slow. It is not the path that
has to be fast: a caller with real data has it in a file, a database or an Arrow
buffer, and all three of those already have a path that does not come through
here. This is for the first ten lines of a program.

Document 18 is the reasoning, including why every case pandas resolves by
widening to an object column has to be resolved here by picking a real type or
by refusing.
"""

from std.python import Python, PythonObject

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.py.errors import DTYPE, VALUE, tagged

comptime _NONE = 0
"""A `None`, which is missing and does not vote for a type."""

comptime _BOOL = 1
"""A `bool`, which has to be tested for before `int` because it is one."""

comptime _INT = 2
"""An `int`."""

comptime _FLOAT = 3
"""A `float`."""

comptime _TEXT = 4
"""A `str`."""

comptime _OTHER = 5
"""Anything else, which is refused with the row and the type in the message."""


def _kind(value: PythonObject, builtins: PythonObject) raises -> Int:
    """Says which of the five things a Python value is.

    `bool` is checked first because a Python `bool` is an `int` and the order is
    the only thing that keeps `True` out of an integer column.

    Args:
        value: The value.
        builtins: The `builtins` module, passed in rather than imported per
            element, since this runs once per value in the frame.

    Returns:
        One of the `_NONE` through `_OTHER` constants.
    """
    if value is Python.none():
        return _NONE
    if Bool(builtins.isinstance(value, builtins.bool)):
        return _BOOL
    if Bool(builtins.isinstance(value, builtins.int)):
        return _INT
    if Bool(builtins.isinstance(value, builtins.float)):
        return _FLOAT
    if Bool(builtins.isinstance(value, builtins.str)):
        return _TEXT
    return _OTHER


def _bools(items: PythonObject, count: Int) raises -> AnyArray:
    """Fills a bool column.

    Args:
        items: The values, already a list.
        count: How many there are.

    Returns:
        The column.
    """
    var out = Array[DType.bool](count)
    for i in range(count):
        if items[i] is Python.none():
            out.set_null(i)
        else:
            out[i] = Bool(py=items[i])
    return AnyArray(out^)


def _ints(name: String, items: PythonObject, count: Int) raises -> AnyArray:
    """Fills an int64 column.

    Args:
        name: The column name, for the message if a value does not fit.
        items: The values, already a list.
        count: How many there are.

    Returns:
        The column.

    Raises:
        Error: If a value is too large for int64. pandas widens to an object
            column and keeps it, which is not available here, so the remaining
            choices are to lose the value silently or to say so.
    """
    var out = Array[DType.int64](count)
    for i in range(count):
        if items[i] is Python.none():
            out.set_null(i)
        else:
            try:
                out[i] = Int64(Int(py=items[i]))
            except:
                raise tagged(
                    VALUE,
                    String(
                        "column '",
                        name,
                        "' has an integer at row ",
                        i,
                        (
                            " that does not fit in int64, and firepanda has no"
                            " column that holds an arbitrary precision integer"
                        ),
                    ),
                )
    return AnyArray(out^)


def _floats(items: PythonObject, count: Int) raises -> AnyArray:
    """Fills a float64 column.

    An integer in a column that also holds floats is converted, since float64 is
    the only type that holds both. A `float("nan")` is a value here and not a
    hole, which is issue #170 arriving at the front door and is the difference
    document 18 section 3 is about.

    Args:
        items: The values, already a list.
        count: How many there are.

    Returns:
        The column.
    """
    var out = Array[DType.float64](count)
    for i in range(count):
        if items[i] is Python.none():
            out.set_null(i)
        else:
            out[i] = Float64(py=items[i])
    return AnyArray(out^)


def _text(items: PythonObject, count: Int) raises -> AnyArray:
    """Fills a string column.

    Args:
        items: The values, already a list.
        count: How many there are.

    Returns:
        The column.
    """
    var builder = StringBuilder(capacity=count)
    for i in range(count):
        if items[i] is Python.none():
            builder.append_null()
        else:
            builder.append(String(items[i]).as_bytes())
    return AnyArray(builder^.finish())


def empty_column(count: Int) raises -> AnyArray:
    """Fills a column that has no values to learn a type from.

    A column of nothing but `None`, or of nothing at all, has no type in it.
    pandas answers with an object column and pyarrow answers with the Arrow null
    type, and neither is available: an object column is the representation this
    library exists to not have, and firepanda has no column that carries the null
    type at run time, which `arrow_import.mojo` says in as many words when it
    refuses to import one.

    So it is float64 with every row missing. That is the type that widens most
    freely if the column is later given numbers, and it is what pandas itself
    produces for a column of NaNs, which is the same list written the other way.

    Args:
        count: How many rows, all of them missing.

    Returns:
        The column.
    """
    var out = Array[DType.float64](count)
    for i in range(count):
        out.set_null(i)
    return AnyArray(out^)


def column_from(name: String, values: PythonObject) raises -> Series:
    """Builds one named column out of a Python sequence.

    This is `array_from` with a name put on it. The two are separate because an
    index has labels and no column name, and inferring the dtype twice by two
    routes is exactly the drift this file exists to prevent.

    Args:
        name: The column name.
        values: The values.

    Returns:
        The column, as a series.

    Raises:
        Error: Whatever `array_from` raises.
    """
    return Series(name, array_from(name, values))


def array_from(name: String, values: PythonObject) raises -> AnyArray:
    """Builds one column out of a Python sequence.

    The sequence goes through `list` first, so a tuple, a `range` and a generator
    all arrive as the same thing, which removes a whole category of works with a
    list and not with a range report. A bare string is refused rather than
    iterated, because a string is a scalar everywhere in pandas and a column of
    its characters is never what anybody meant.

    Args:
        name: What to call this in a message. A column name, or the word labels
            when the caller is an index and there is no column.
        values: The values.

    Returns:
        The column.

    Raises:
        Error: If the values are not a sequence, or hold a type that is not one
            of boolean, integer, floating point and text, or mix boolean with
            integer.
    """
    var builtins = Python.import_module("builtins")
    if Bool(builtins.isinstance(values, builtins.str)):
        raise tagged(
            VALUE,
            String(
                "column '",
                name,
                (
                    "' is a string, which pandas would broadcast to every row."
                    " firepanda does not broadcast, so pass a list of values"
                ),
            ),
        )
    var items: PythonObject
    try:
        items = builtins.list(values)
    except:
        raise tagged(
            VALUE,
            String(
                "column '",
                name,
                "' is a ",
                Python.type(values).__name__,
                ", and a column is built from a sequence of values",
            ),
        )

    var count = Int(len(items))
    var seen = 0
    for i in range(count):
        var kind = _kind(items[i], builtins)
        if kind == _OTHER:
            raise tagged(
                DTYPE,
                String(
                    "column '",
                    name,
                    "' has a ",
                    Python.type(items[i]).__name__,
                    " at row ",
                    i,
                    (
                        ", and a firepanda column holds booleans, integers,"
                        " floating point numbers or text"
                    ),
                ),
            )
        seen |= 1 << kind

    var boolean = (seen & (1 << _BOOL)) != 0
    var integer = (seen & (1 << _INT)) != 0
    var floating = (seen & (1 << _FLOAT)) != 0
    var text = (seen & (1 << _TEXT)) != 0

    if not (boolean or integer or floating or text):
        return empty_column(count)
    if boolean and not (integer or floating or text):
        return _bools(items, count)
    if text and not (boolean or integer or floating):
        return _text(items, count)
    if floating and not (boolean or text):
        return _floats(items, count)
    if integer and not (boolean or text):
        return _ints(name, items, count)
    raise tagged(
        DTYPE,
        String(
            "column '",
            name,
            "' mixes ",
            _describe(seen),
            (
                ", and a firepanda column holds one type. pandas would make"
                " this an object column, which is the representation firepanda"
                " exists to not have"
            ),
        ),
    )


def _describe(seen: Int) -> String:
    """Names the kinds a column turned out to hold, for the refusal message.

    Args:
        seen: The bits set by the classification pass.

    Returns:
        A comma separated list, in the order the constants are declared.
    """
    var names = List[String]()
    if (seen & (1 << _BOOL)) != 0:
        names.append("booleans")
    if (seen & (1 << _INT)) != 0:
        names.append("integers")
    if (seen & (1 << _FLOAT)) != 0:
        names.append("floating point numbers")
    if (seen & (1 << _TEXT)) != 0:
        names.append("text")
    var out = String()
    for i in range(len(names)):
        if i > 0:
            out += " and " if i == len(names) - 1 else ", "
        out += names[i]
    return out^


def frame_from(data: PythonObject) raises -> DataFrame:
    """Builds a frame out of a mapping of column name to sequence.

    The columns come out in the mapping's own order, which is what pandas does
    and is the reason dictionaries being ordered stopped being a detail.

    Args:
        data: The mapping, or `None` for an empty frame.

    Returns:
        The frame.

    Raises:
        Error: If the argument is not a mapping, or the columns are not all the
            same length, or any column cannot be built.
    """
    if data is Python.none():
        return DataFrame()
    var builtins = Python.import_module("builtins")
    if not Bool(builtins.hasattr(data, "keys")):
        raise tagged(
            VALUE,
            String(
                (
                    "a frame is built from a mapping of column name to values,"
                    " and this is a "
                ),
                Python.type(data).__name__,
                (
                    ". Reading a list of records, a list of rows or an array is"
                    " not written yet"
                ),
            ),
        )
    var columns = List[Series]()
    var rows = -1
    for key in data:
        var name = String(key)
        var column = column_from(name, data[key])
        if rows < 0:
            rows = len(column)
        elif len(column) != rows:
            raise tagged(
                VALUE,
                String(
                    "all columns must be the same length, and '",
                    name,
                    "' has ",
                    len(column),
                    " rows where the ones before it have ",
                    rows,
                ),
            )
        columns.append(column^)
    return DataFrame.from_series(columns^)
