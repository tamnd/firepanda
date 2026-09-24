"""What firepanda raises, and how a Mojo error becomes one of them.

pandas users write `except KeyError` around a column lookup, and they have
written it in code that predates this project by fifteen years. An error class
is as much a part of an API as a signature is, so this module is public and it
is in the compatibility policy: `firepanda.errors.ColumnNotFoundError` is a
`KeyError` and will not stop being one.

### Why there are named classes at all

Every class here subclasses the builtin that `docs/specs/07-python-bindings.md`
section 5 maps its kind onto, so `except KeyError` and `except TypeError` catch
them and nothing that works against pandas has to change. The names exist for
the other direction, which is reading a traceback. `DTypeError: cannot add
int64 and float64` says more at a glance than `TypeError` does, and document 04
section 8 is a promise about exactly that, so the names carry it rather than
being decoration. `pandas.errors` is laid out the same way for the same reason.

### Why the message carries the class

Nothing in the Mojo binding API can raise a typed exception. A `raise
Error(...)` from a bound function arrives as a bare `Exception` and setting a
type by hand first does not survive the wrapper, which is measured in document
12 section 5. What does survive is the message, so the message is what carries
the classification: `firepanda/py/errors.mojo` puts a short prefix on it and
`translate` below takes it off.

That is a wire format between two halves of one library, and its whole cost is
that both halves have to agree. `python/tests/test_errors.py` is what holds them
to that, by asking the Mojo side to raise one of each kind and checking what
comes out over here.

If Modular later exposes a way to raise a typed error from a bound function,
this module keeps its classes, `translate` goes away, and no user visible thing
moves. That is why it is built this way round.
"""

from __future__ import annotations

__all__ = [
    "AbstractMethodError",
    "AttributeConflictWarning",
    "CSSWarning",
    "CancelledError",
    "CategoricalConversionWarning",
    "ChainedAssignmentError",
    "ClosedFileError",
    "ColumnNotFoundError",
    "DTypeError",
    "DataError",
    "DatabaseError",
    "DtypeWarning",
    "DuplicateLabelError",
    "EmptyDataError",
    "FirepandaError",
    "IncompatibilityWarning",
    "IncompatibleFrequency",
    "IndexingError",
    "IntCastingNaNError",
    "InvalidArgumentError",
    "InvalidColumnName",
    "InvalidComparison",
    "InvalidIndexError",
    "InvalidVersion",
    "LossySetitemError",
    "MergeError",
    "NoBufferPresent",
    "NullFrequencyError",
    "NumExprClobberingError",
    "NumbaUtilError",
    "NumericOverflowError",
    "OptionError",
    "OutOfBoundsDatetime",
    "OutOfBoundsError",
    "OutOfBoundsTimedelta",
    "Pandas4Warning",
    "Pandas5Warning",
    "PandasChangeWarning",
    "PandasDeprecationWarning",
    "PandasFutureWarning",
    "PandasPendingDeprecationWarning",
    "ParserError",
    "ParserWarning",
    "PerformanceWarning",
    "PossibleDataLossError",
    "PossiblePrecisionLoss",
    "PyperclipException",
    "PyperclipWindowsException",
    "ReaderError",
    "SpecificationError",
    "UndefinedVariableError",
    "UnsortedIndexError",
    "UnsupportedError",
    "UnsupportedFunctionCall",
    "ValueLabelTypeMismatch",
]


class FirepandaError(Exception):
    """The marker every firepanda error carries.

    It is a mixin rather than a base, so that each class below can inherit from
    the builtin its kind maps onto and still be recognisable as ours. Catching
    this is a way of saying that firepanda failed rather than that the code
    around it did, and it is the only thing here that does not correspond to a
    row of the table.
    """


class ColumnNotFoundError(FirepandaError, KeyError):
    """A column was asked for and the frame does not have it.

    A `KeyError`, because that is what pandas raises and what fifteen years of
    code is already catching.
    """


class DTypeError(FirepandaError, TypeError):
    """A dtype mismatch, a cast that cannot be made, an argument of the wrong type.

    A `TypeError`. firepanda does not upcast silently, so this one is reached
    more often here than the equivalent is in pandas, and the message is
    expected to say what to write instead.
    """


class InvalidArgumentError(FirepandaError, ValueError):
    """An argument of the right type and the wrong value.

    A `ValueError`, on the same distinction Python itself draws: the type was
    acceptable and the value was not.
    """


class IndexingError(FirepandaError):
    """A row key that cannot be lined up with the rows it is selecting.

    pandas has a class with this name in `pandas.errors` and raises it for a
    boolean series whose labels miss some of the rows it is masking. It is a
    plain `Exception` there, so it is one here, and the name is carried for the
    reason `IntCastingNaNError` gives.
    """


class InvalidIndexError(FirepandaError):
    """Labels that repeat where a lookup needs each label once.

    pandas has a class with this name in `pandas.errors` and raises it when a
    column with repeated labels is used as a mapping, since a label would then
    name more than one value. It is a plain `Exception` there, so it is one
    here, and the name is carried for the reason `IntCastingNaNError` gives.
    """


class IntCastingNaNError(InvalidArgumentError):
    """A missing value, a NaN or an infinity where an integer column was asked for.

    pandas has a class with this exact name and this exact purpose, in
    `pandas.errors`, and it is the error that exists to explain why the nullable
    integer dtypes exist. A program that catches it by name is asking a specific
    question and gets nothing if the answer arrives as a plain `ValueError`, so
    the name is worth carrying.

    It subclasses `InvalidArgumentError` rather than sitting beside it, which
    makes it a `ValueError` by two routes: the pandas one is a `ValueError` too,
    and a caller who catches the broad one still catches this either way.
    """


class MergeError(InvalidArgumentError):
    """A merge asked for in a way that cannot be carried out.

    pandas has a class with this name in `pandas.errors` and raises it for keys
    that were not given or not found, and for a `validate` the keys fail. A
    program that catches it by name gets nothing from a plain `ValueError`, so
    the name is carried, for the reason `IntCastingNaNError` gives.
    """


class SpecificationError(FirepandaError):
    """An aggregation asked for in a shape pandas does not read.

    pandas has a class with this name in `pandas.errors` and raises it for a
    mapping handed to a column's `agg`, which would name a column inside a
    column. It is a plain `Exception` there, so it is one here, and the name is
    carried for the reason `IntCastingNaNError` gives.
    """


class UndefinedVariableError(FirepandaError, NameError):
    """A name in a `query` expression that is not a column, a label or a variable.

    pandas has a class with this name in `pandas.errors` and it is a `NameError`
    there, which is what Python raises for a name it cannot find, so it is one
    here, and the name is carried for the reason `IntCastingNaNError` gives.
    It is built the way pandas builds it, from the name and whether the name was
    a variable of the caller, and it writes the message from those.
    """

    def __init__(self, name: str, is_local: bool | None = None) -> None:
        said = f"{name!r} is not defined"
        super().__init__(f"local variable {said}" if is_local else f"name {said}")


class NumericOverflowError(FirepandaError, OverflowError):
    """A number that does not fit the dtype it was asked to fit.

    An `OverflowError`, which is what pandas raises when a scalar is too large
    for the column it is being combined with, and which is not a `ValueError`,
    so it needs a row of its own rather than sharing one with
    `InvalidArgumentError`. The name says numeric to keep it apart from
    `OutOfBoundsError`, which is about a position and is an `IndexError`.
    """


class OutOfBoundsError(FirepandaError, IndexError):
    """A row number outside the thing it was addressing.

    An `IndexError`, which is what a Python sequence raises and what `except
    IndexError` around `index[i]` is already catching. pandas raises the plain
    builtin here, so the only thing this adds is a name in the traceback.
    """


class ReaderError(FirepandaError, OSError):
    """A file that is missing, unreadable, or not what it claimed to be.

    An `OSError`, which is what `open` raises, and which `FileNotFoundError`
    already inherits from, so `except OSError` around a read keeps working.
    """


class UnsupportedError(FirepandaError, NotImplementedError):
    """Something firepanda has not implemented yet.

    A `NotImplementedError`. This is the honest answer while the surface is five
    members out of a thousand, and it is worth being a distinct class rather
    than a `RuntimeError` because a caller can reasonably branch on it and fall
    back to pandas.
    """


class CancelledError(KeyboardInterrupt):
    """The user interrupted the work.

    A `KeyboardInterrupt`, and the one class here that is not a
    `FirepandaError`. It cannot be, because `FirepandaError` is an `Exception`
    and inheriting from both would put `Exception` in this class's ancestry,
    after which a bare `except Exception` around a firepanda call would swallow
    a Ctrl-C. That is the exact behaviour `KeyboardInterrupt` exists to avoid,
    so the marker is what gets dropped.
    """


# The rest of `pandas.errors`, by the same name and on the same builtins, so
# that `except pandas.errors.X` written against pandas has a class to name here.
# Most are raised by readers and writers firepanda does not have, and are
# carried so that code naming them imports and catches as it did.


class AbstractMethodError(FirepandaError, NotImplementedError):
    """A method a subclass was meant to write and did not."""


class ClosedFileError(FirepandaError):
    """An operation on a store file that has been closed."""


class DataError(FirepandaError):
    """An operation that needs numbers, asked of values that are not numbers."""


class DatabaseError(FirepandaError, OSError):
    """SQL that did not run, from a bad statement or the database itself."""


class DuplicateLabelError(FirepandaError, ValueError):
    """An operation that would repeat a label where labels must not repeat."""


class EmptyDataError(FirepandaError, ValueError):
    """A file with nothing in it to read, not even a header."""


class IncompatibleFrequency(FirepandaError, TypeError):
    """Two periods or offsets of frequencies that cannot be combined."""


class InvalidComparison(FirepandaError):
    """A value that cannot be compared with the values on the other side."""


class InvalidVersion(FirepandaError, ValueError):
    """A version string that is not a version in the sense of PEP 440."""


class LossySetitemError(FirepandaError):
    """A value that cannot be put into a column without changing it."""


class NoBufferPresent(FirepandaError):
    """A buffer asked for in the interchange protocol that a column does not have."""


class NullFrequencyError(FirepandaError, ValueError):
    """An operation that needs a frequency, on labels that have none."""


class NumExprClobberingError(FirepandaError, NameError):
    """A variable in an expression named after one of numexpr's own names."""


class NumbaUtilError(FirepandaError):
    """A routine the numba engine does not support."""


class OptionError(FirepandaError, AttributeError, KeyError):
    """An option name that is unknown or names more than one option."""


class OutOfBoundsDatetime(FirepandaError, ValueError):
    """An instant outside the range its unit can count."""


class OutOfBoundsTimedelta(FirepandaError, ValueError):
    """A span outside the range its unit can count."""


class ParserError(FirepandaError, ValueError):
    """File contents that could not be parsed."""


class PossibleDataLossError(FirepandaError):
    """A store file opened again while it is still open."""


class PyperclipException(FirepandaError, RuntimeError):
    """The clipboard is not available on this machine."""


class PyperclipWindowsException(PyperclipException):
    """The clipboard is not available on this Windows machine."""


class UnsortedIndexError(FirepandaError, KeyError):
    """A slice of labels in several levels that are not sorted."""


class UnsupportedFunctionCall(FirepandaError, ValueError):
    """A numpy function called with arguments firepanda does not take."""


class PandasChangeWarning(Warning):
    """A change that is coming in a later version."""


class PandasDeprecationWarning(PandasChangeWarning, DeprecationWarning):
    """A coming change that is a deprecation."""


class PandasPendingDeprecationWarning(PandasChangeWarning, PendingDeprecationWarning):
    """A coming change that will become a deprecation."""


class PandasFutureWarning(PandasChangeWarning, FutureWarning):
    """A coming change in what a call answers."""


class Pandas4Warning(PandasDeprecationWarning):
    """A change pandas makes in its version 4."""


class Pandas5Warning(PandasPendingDeprecationWarning):
    """A change pandas makes in its version 5."""


class AttributeConflictWarning(Warning):
    """Index attributes that disagree between a store and what is written."""


class CSSWarning(UserWarning):
    """A style that could not be turned into the format asked for."""


class CategoricalConversionWarning(Warning):
    """A partly labelled Stata file read a piece at a time."""


class ChainedAssignmentError(Warning):
    """A value set through two selections, which sets it on a copy."""


class DtypeWarning(Warning):
    """A column of a file read with values of different types."""


class IncompatibilityWarning(Warning):
    """A where condition on a store file that cannot take one."""


class InvalidColumnName(Warning):
    """A column name Stata cannot hold, changed on the way out."""


class ParserWarning(Warning):
    """A file read with a parser other than the one asked for."""


class PerformanceWarning(Warning):
    """A call that works and is likely to be slow."""


class PossiblePrecisionLoss(Warning):
    """A whole number too large for Stata, written as a float."""


class ValueLabelTypeMismatch(Warning):
    """A category column with labels that are not text, written to Stata."""


# The table from document 07 section 5, and the only place the mapping is
# written down. A kind is what crosses the boundary; a class is what a user
# catches. Changing which class a kind raises is a change to the public API and
# is one line here.
BY_KIND: dict[str, type[BaseException]] = {
    "column": ColumnNotFoundError,
    "dtype": DTypeError,
    "value": InvalidArgumentError,
    "nonfinite": IntCastingNaNError,
    "overflow": NumericOverflowError,
    "position": OutOfBoundsError,
    "io": ReaderError,
    "unsupported": UnsupportedError,
    "cancelled": CancelledError,
}

PREFIX = "firepanda:"

# The binding layer's own errors, which are not ours and cannot be tagged at
# source. An arity mismatch arrives as `Exception: TypeError: <mojo function>()
# takes 1 positional argument but 2 were given`, with the right words in the
# wrong place: it is an `Exception` with `TypeError` written at the front of its
# message, so `except TypeError` does not catch it. Document 13 section 5 has
# the measurement. This is the most common mistake a user can make and it is
# worth the two lines it costs to put the class back.
BY_BUILTIN_NAME: dict[str, type[BaseException]] = {
    "TypeError": TypeError,
    "ValueError": ValueError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "OverflowError": OverflowError,
}


def translate(error: BaseException) -> BaseException:
    """Turns an error that crossed the boundary into the class it should be.

    Args:
        error: What the extension raised.

    Returns:
        The exception to raise instead. An error that is already one of ours, or
        that is already a `BaseException` Python raised on its own account such
        as a `MemoryError`, comes back unchanged, so this is safe to apply to
        anything.
    """
    if isinstance(error, FirepandaError):
        return error

    message = str(error)

    if message.startswith(PREFIX):
        kind, _, rest = message[len(PREFIX) :].partition(": ")
        # An unknown kind is a version skew between the two halves, which can
        # happen in a development tree with a stale extension in it. Keeping the
        # whole message including the prefix is deliberate: the prefix is the
        # evidence of what went wrong.
        wanted = BY_KIND.get(kind)
        return wanted(rest) if wanted is not None else RuntimeError(message)

    # `Exception: TypeError: ...` and the double wrapped `ValueError: TypeError:
    # ...` that a constructor produces, which document 13 section 5 records.
    name, separator, rest = message.partition(": ")
    if separator and name in BY_BUILTIN_NAME:
        return BY_BUILTIN_NAME[name](rest)

    # An error that reached Python untagged came out of the core without a
    # binding classifying it, which is the last row of the table and also, in
    # practice, a note that the binding it came through wants a `try` around it.
    if type(error) in (Exception, ValueError):
        return RuntimeError(message)
    return error
