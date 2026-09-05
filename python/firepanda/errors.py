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
    "CancelledError",
    "ColumnNotFoundError",
    "DTypeError",
    "FirepandaError",
    "InvalidArgumentError",
    "ReaderError",
    "UnsupportedError",
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


# The table from document 07 section 5, and the only place the mapping is
# written down. A kind is what crosses the boundary; a class is what a user
# catches. Changing which class a kind raises is a change to the public API and
# is one line here.
BY_KIND: dict[str, type[BaseException]] = {
    "column": ColumnNotFoundError,
    "dtype": DTypeError,
    "value": InvalidArgumentError,
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
