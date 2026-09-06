"""The wire format for errors crossing into Python.

Mojo has one `Error` and it carries a string. Python users write `except
KeyError`. Nothing in the binding API bridges that: a `raise Error(...)` from a
`def_function` or a `def_method` arrives as a bare `Exception`, and setting a
typed exception by hand beforehand does not survive the wrapper, which is
measured in `docs/specs/12-the-python-front-door-measured.md` section 5. So the
class has to be recovered on the Python side, and the only thing that crosses is
the message.

This module is the agreement about what that message looks like. A classified
error starts with `firepanda:` then a kind then `: `, and everything after that
is the message a user reads. `python/firepanda/errors.py` reads the prefix off,
raises the matching class and passes the rest along untouched.

The kinds are deliberately few and they are not the exception classes. Mapping a
kind onto a class is the Python side's business and it is written down in one
table there, so that changing which class an unknown column raises is one line in
one file rather than a search through Mojo.

### Whose job the tag is

The core raises errors in nearly three hundred places and none of them carries a
tag. That is on purpose and it should stay that way. A `raise Error(...)` deep in
a kernel is a diagnostic for somebody reading Mojo, and it does not know whether
it was reached because a user asked for a column that is not there or because
the planner built something wrong. The binding does know, because the binding is
the call the user made.

So the rule is that the tag is applied at the boundary, by the binding, on the
way out. A binding either raises a tagged error itself or catches an untagged one
and tags it with the kind that matches what the call was trying to do. Anything
that arrives in Python untagged is a bug in a binding rather than a bug in the
core, and it lands on `RuntimeError`, which is the row the table ends with.
"""


comptime COLUMN = "firepanda:column: "
"""A column that was asked for and is not there. Becomes `KeyError`."""

comptime DTYPE = "firepanda:dtype: "
"""A dtype mismatch, a bad cast, an argument of the wrong type. Becomes
`TypeError`."""

comptime POSITION = "firepanda:position: "
"""A row number outside the thing it was addressing. Becomes `IndexError`.

Separate from `value` because Python separates them, and because `except
IndexError` around an indexing expression is the idiom this exists to keep
working. It is `value` narrowed to one shape rather than a new idea."""

comptime VALUE = "firepanda:value: "
"""An argument of the right type and the wrong value, or something unparseable.
Becomes `ValueError`."""

comptime OVERFLOW = "firepanda:overflow: "
"""A number that does not fit the dtype it was asked to fit. Becomes
`OverflowError`.

Its own kind rather than a shape of `value`, because `OverflowError` is not a
`ValueError` in Python and an `except ValueError` written around a pandas call
does not catch one. Folding it into `value` would be a difference nobody would
find until it mattered."""

comptime IO = "firepanda:io: "
"""A file that is missing, unreadable or malformed. Becomes `OSError`."""

comptime UNSUPPORTED = "firepanda:unsupported: "
"""Something firepanda has not implemented. Becomes `NotImplementedError`."""

comptime CANCELLED = "firepanda:cancelled: "
"""The user interrupted the work. Becomes `KeyboardInterrupt`, which is not an
`Exception` in Python and so has to be raised rather than returned by anything
catching one."""


def tagged(kind: String, message: String) -> Error:
    """Builds a classified error.

    It returns rather than raises so that the call site keeps the word `raise`
    in it, which is what a reader scans for.

    Args:
        kind: One of the prefixes above.
        message: What the user reads, with no prefix on it.

    Returns:
        The error to raise.
    """
    return Error(kind + message)


def retagged(kind: String, cause: Error) -> Error:
    """Tags an error that came back untagged from somewhere further down.

    The message is kept exactly, because the core writes better messages than
    anything a binding could invent about a call it only knows the shape of.

    Args:
        kind: One of the prefixes above.
        cause: The error to classify.

    Returns:
        The error to raise.
    """
    return Error(kind + String(cause))
