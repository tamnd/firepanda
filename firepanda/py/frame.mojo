"""The `DataFrame` binding, which is the narrow half of the front door.

What is here is deliberately not the pandas API. It is a private calling
convention that the Python layer in `python/firepanda/` is written against, and
the reason for the split is measured in
`docs/specs/13-the-bound-type-is-not-a-dataframe.md`. `PythonTypeBuilder` can
attach methods to a type and nothing else, so `df["a"]`, `len(df)`, `df.shape`
and `for row in df` are not expressible here by any route, and 28 percent of the
pandas surface is properties and operators. Those live in Python and call flat
named methods on this type.

So the naming below reads oddly on purpose. `length` rather than `__len__`,
`shape` returning a tuple that Python turns into a property, and no `__getitem__`
at all. Nothing in this file is public API and nothing in it should be shaped to
look like pandas.
"""

from std.os import abort
from std.memory import Pointer
from std.python import Python, PythonObject
from std.python.bindings import check_arguments_arity

from firepanda.frame import DataFrame
from firepanda.io.read import read_csv
from firepanda.py.errors import (
    CANCELLED,
    COLUMN,
    DTYPE,
    IO,
    UNSUPPORTED,
    VALUE,
    retagged,
    tagged,
)


def _int(value: PythonObject, name: String) raises -> Int:
    """Reads a Python integer, and says which argument was wrong if it is not one.

    `Int(py=value)` raises `invalid literal for int() with base 10: 'x'`, which
    is the right complaint and names neither the argument nor the function. A
    user with three integer arguments cannot tell from it which one they got
    wrong.

    Args:
        value: What Python passed.
        name: The parameter name, for the message.

    Returns:
        The integer.
    """
    try:
        return Int(py=value)
    except:
        raise tagged(
            DTYPE,
            String(
                name,
                " must be an integer, got ",
                Python.type(value).__name__,
                " ",
                value.__repr__(),
            ),
        )


@fieldwise_init
struct PyDataFrame(Movable, Writable):
    """A firepanda `DataFrame` with a CPython object wrapped around it."""

    var frame: DataFrame
    """The frame itself. Owned by the Python object that holds this value."""

    @staticmethod
    def py_init(
        out self: Self, args: PythonObject, kwargs: PythonObject
    ) raises:
        """Refuses to build a frame from Python.

        There is no argument shape that makes sense yet. A frame arrives through
        `read_csv` or `read_parquet` and the constructor that takes columns needs
        the Python to Arrow conversion that is not written. Raising here is
        better than accepting nothing and handing back an empty frame, which
        would look like it worked.

        Args:
            args: Positional arguments, of which none are accepted.
            kwargs: Keyword arguments, of which none are accepted.
        """
        check_arguments_arity(0, args, "DataFrame")
        raise tagged(
            UNSUPPORTED,
            (
                "construct a frame with firepanda.read_csv or"
                " firepanda.read_parquet; building one from columns is not"
                " wired up yet"
            ),
        )

    @staticmethod
    def _frame(py_self: PythonObject) -> Pointer[Self, MutAnyOrigin]:
        """Recovers the Mojo value out of the Python object holding it.

        A failure here means the object is not a `PyDataFrame`, which the binding
        layer has already checked by the time a method body runs, so it is a bug
        in this file rather than a thing a caller can cause.

        Args:
            py_self: The Python object.

        Returns:
            A pointer to the wrapped value.
        """
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("not a firepanda DataFrame: ", e))

    @staticmethod
    def length(py_self: PythonObject) raises -> PythonObject:
        """Reports the row count.

        Args:
            py_self: The frame.

        Returns:
            The number of rows.
        """
        return PythonObject(len(Self._frame(py_self)[].frame))

    @staticmethod
    def width(py_self: PythonObject) raises -> PythonObject:
        """Reports the column count.

        Args:
            py_self: The frame.

        Returns:
            The number of columns.
        """
        return PythonObject(Self._frame(py_self)[].frame.width())

    @staticmethod
    def names(py_self: PythonObject) raises -> PythonObject:
        """Reports the column names, in order.

        Args:
            py_self: The frame.

        Returns:
            A list of strings.
        """
        var out = Python.list()
        for name in Self._frame(py_self)[].frame.names():
            out.append(PythonObject(name))
        return out

    @staticmethod
    def head(py_self: PythonObject, n: PythonObject) raises -> PythonObject:
        """Takes the first `n` rows.

        Args:
            py_self: The frame.
            n: How many rows to take.

        Returns:
            A new frame.
        """
        return PythonObject(
            alloc=Self(Self._frame(py_self)[].frame.head(_int(n, "n")))
        )

    @staticmethod
    def tail(py_self: PythonObject, n: PythonObject) raises -> PythonObject:
        """Takes the last `n` rows.

        Args:
            py_self: The frame.
            n: How many rows to take.

        Returns:
            A new frame.
        """
        return PythonObject(
            alloc=Self(Self._frame(py_self)[].frame.tail(_int(n, "n")))
        )

    def write_to(self, mut writer: Some[Writer]):
        """Writes the frame the way `describe` does.

        Both `__str__` and `__repr__` on the Python side come from
        `write_repr_to`, and `write_to` is never reached, which is recorded in
        document 13 section 2. It is written anyway so that the Mojo value
        behaves like every other `Writable` in the tree.

        Args:
            writer: Where to write.
        """
        writer.write(self.frame.describe())

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the frame. This is what Python sees for both `str` and `repr`.

        Args:
            writer: Where to write.
        """
        writer.write(self.frame.describe())


def open_csv(path: PythonObject) raises -> PythonObject:
    """Reads a CSV file into a frame.

    The reader's own message is kept, because it says which file and what the
    operating system said about it, which is more than this function knows. All
    that is added is the classification, which this function knows and the
    reader does not: everything that goes wrong reading a file is an `OSError`
    to a Python caller.

    Args:
        path: The path to read.

    Returns:
        A new frame.
    """
    try:
        return PythonObject(alloc=PyDataFrame(read_csv(String(path))))
    except cause:
        raise retagged(IO, cause)


def raise_for_test(kind: PythonObject) raises -> PythonObject:
    """Raises one classified error of each kind, so the table can be tested.

    Every row of the mapping in `python/firepanda/errors.py` has to be exercised
    from Python, and the bound surface is five methods, none of which can reach
    most of the rows. This is the way in. It is registered as `_raise_for_test`
    and it is the one entry point in the extension that exists for the tests
    rather than for a user.

    The alternative was to test the mapping against messages written in the test
    file, which would have tested the Python half against itself and left the
    thing that actually matters, that the two halves agree on the wire format,
    unchecked.

    Args:
        kind: The bare kind, such as `column`.

    Returns:
        Never. It always raises.
    """
    var which = String(kind)
    if which == "column":
        raise tagged(COLUMN, "no such column 'regoin'")
    if which == "dtype":
        raise tagged(DTYPE, "cannot add int64 and float64")
    if which == "value":
        raise tagged(VALUE, "n must not be negative")
    if which == "io":
        raise tagged(IO, "no such file '/nowhere'")
    if which == "unsupported":
        raise tagged(UNSUPPORTED, "object dtype is not supported")
    if which == "cancelled":
        raise tagged(CANCELLED, "interrupted")
    if which == "untagged":
        raise Error("something went wrong a long way down")
    raise tagged(VALUE, String("no such kind ", kind.__repr__()))
