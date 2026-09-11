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

`to_datetime` at the bottom is the one thing here that is not a member, and it is
here for the same reason the members are. It is a module level function in pandas
rather than a method, so nothing inherits it and `__init__.py` exports it
directly, but it is ten parameters of which five are refused by name, which is
exactly the shape the generator cannot write and exactly the shape `_refuse` and
`_held_at` exist for.
"""

from __future__ import annotations

import datetime
import math
import operator
import re
import warnings
from typing import TYPE_CHECKING, Any

from . import _firepanda
from .errors import (
    ColumnNotFoundError,
    DTypeError,
    InvalidArgumentError,
    OutOfBoundsError,
    UnsupportedError,
    translate,
)

if TYPE_CHECKING:
    from ._frame import (
        DataFrame,
        DataFrameGroupBy,
        Expanding,
        ExponentialMovingWindow,
        Index,
        Rolling,
        Series,
        SeriesGroupBy,
    )


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


def _no_level(level: Any) -> None:
    """Refuses the `level` argument every named arithmetic form declares.

    pandas takes a level to say which level of a MultiIndex to align on, and
    firepanda has no MultiIndex yet, so there is nothing for it to name. It is
    declared rather than left out for the reason `_refuse` gives: the signature
    parity test compares the whole parameter list against a running pandas, and a
    caller who passes one is told what is missing.

    Args:
        level: What was passed.

    Raises:
        NotImplementedError: If anything other than None was passed.
    """
    if level is not None:
        raise NotImplementedError(
            "level= is not supported yet, because aligning on one level of a"
            " MultiIndex needs a MultiIndex, and there is not one yet"
        )


def _reindex_filling(kind: str, method: Any, limit: Any, tolerance: Any) -> None:
    """Refuses the three parameters of reindex that are about filling.

    `method` fills a label the caller does not have from the label beside it,
    which needs the labels in order to mean anything and is a different
    operation from putting a value in the row, so it is refused rather than
    approximated. `limit` and `tolerance` belong to `method`, and passing either
    without it is pandas' own error, given back word for word because the caller
    made pandas' mistake and document 22 says the wording is theirs in that
    case.

    Args:
        kind: The word for what is being reindexed, for the message.
        method: What was passed for `method`.
        limit: What was passed for `limit`.
        tolerance: What was passed for `tolerance`.

    Raises:
        UnsupportedError: If a `method` was asked for.
        InvalidArgumentError: If a `limit` or a `tolerance` arrived without one.
    """
    if method is not None:
        raise UnsupportedError(
            f"reindex with method= fills a label the {kind} does not have from"
            " the label beside it, which needs the labels in order and is a"
            " different operation from putting a value in the row"
        )
    if limit is not None or tolerance is not None:
        raise InvalidArgumentError(
            "limit argument only valid if doing pad, backfill or nearest reindexing"
        )


def _axis_number(axis: Any, owner: str, default: int, allowed: tuple[int, ...]) -> int:
    """Turns the five spellings of an axis into the number the boundary takes.

    pandas accepts `0`, `1`, `"index"`, `"columns"` and `"rows"` interchangeably,
    and a series accepts only the ones that mean the rows since it has one axis.
    What is allowed is passed in rather than decided here, because that is the
    difference between the two classes and it is one line at each call site.

    `None` is not an error and does not mean the first axis: pandas takes it as
    the default for the call, which is the columns on a frame, and the message
    and the default are therefore both passed in too. A `True` is not an error
    either, which is measured rather than assumed. Python makes a bool an int and
    pandas does nothing to undo that, so `axis=True` means the columns and this
    says so as well, since a compatibility layer that is stricter than the thing
    it copies is still incompatible with it.

    Args:
        axis: What was passed.
        owner: The class name, for the message.
        default: The axis a `None` means.
        allowed: The axis numbers this class has.

    Returns:
        0 for the rows, 1 for the columns.

    Raises:
        InvalidArgumentError: If it is not one of the five, or is an axis this
            class does not have.
    """
    if axis is None:
        return default
    numbers: dict[Any, int] = {0: 0, 1: 1, "index": 0, "columns": 1, "rows": 0}
    try:
        number = numbers[axis]
    except (KeyError, TypeError):
        raise InvalidArgumentError(f"No axis named {axis} for object type {owner}") from None
    if number not in allowed:
        raise InvalidArgumentError(f"No axis named {axis} for object type {owner}") from None
    return number


def _no_fill_against_a_series(fill_value: Any) -> None:
    """Refuses a fill value between a frame and a series.

    This looks like a gap and it is measured. A series is broadcast across the
    rows rather than aligned against them cell by cell, so there is no second side
    for a fill to stand in for, and pandas says `NotImplementedError: fill_value 0
    not supported.` rather than picking a meaning for it. This says the same thing
    at more length, because the pandas message names the value and not the reason.

    Args:
        fill_value: What was passed.

    Raises:
        NotImplementedError: If anything other than None was passed.
    """
    if fill_value is not None:
        raise NotImplementedError(
            f"fill_value {fill_value!r} not supported when the other operand is a"
            " Series, because a Series is broadcast across the rows rather than"
            " aligned against them cell by cell, so there is no second side for"
            " the fill to stand in for"
        )


def _held_at(name: str, value: Any, default: Any, why: str) -> None:
    """Refuses a declared argument that is only implemented at its default.

    `_refuse` is this for the arguments whose default is None, which is most of
    them. The twelve reductions are where that stopped being enough: `skipna`
    defaults to True and `min_count` to zero, so there is no None to compare
    against and the question is whether what arrived is the one value there is an
    implementation for.

    The reason it refuses rather than ignores is the one `_refuse` gives. A
    caller who writes `skipna=False` is asking for a different answer, and
    handing back the answer for `skipna=True` would be wrong in the way that
    takes longest to find.

    Args:
        name: The parameter name.
        value: What was passed.
        default: The one value that is implemented.
        why: What it would take to support the rest.

    Raises:
        NotImplementedError: If it is not the default.
    """
    if value != default:
        raise NotImplementedError(f"{name}={value!r} is not supported yet, because {why}")


def _keep_word(keep: Any) -> str:
    """Turns pandas' `keep` into the word the boundary carries.

    pandas writes two of the three rules as strings and the third as the bool
    `False`, which is a spelling and not a distinction: all three are answers to
    the same question about which row of a repeated key is the one that is not a
    repeat. The boundary carries a word for all three so that one kind of thing
    crosses it, and this is where the spelling is undone.

    The check is membership in a tuple, which is what pandas does, and copying
    the form rather than the intent matters here. `0 == False` in Python, so a
    zero is in that tuple, so pandas takes `keep=0` and reads it as the third
    rule. A layer that checked with `is False` would be tidier and would refuse
    a call that the library it is copying accepts.

    Args:
        keep: What the caller wrote.

    Returns:
        `"first"`, `"last"` or `"none"`.

    Raises:
        InvalidArgumentError: If it is none of the three. The message is pandas'
            own, since a caller reading it is reading it out of a traceback and
            has no way to tell which library wrote it.
    """
    if keep not in ("first", "last", False):
        raise InvalidArgumentError("keep must be either 'first', 'last' or False")
    return keep if isinstance(keep, str) else "none"


# The thirteen interpolations pandas takes, in the order numpy lists them, since
# pandas hands the name straight to `numpy.quantile` and the message it raises
# prints numpy's own dictionary. firepanda has written `linear`. The other twelve
# are a schedule and are refused as one, which is why this list is longer than
# anything firepanda answers: it is what pandas accepts, and a name that is not
# on it is a typo rather than a gap.
_INTERPOLATIONS = (
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
)

# The four spellings of `nonexistent`, and the sentence pandas refuses a fifth
# with. pandas takes a timedelta here as well, which is not a string and would
# fail this check, so the call sites let one through to `_held_at` and it comes
# back a NotImplementedError. That is the right class for it: a timedelta is a
# value pandas accepts and firepanda has not written, which is the schedule and
# not the typo.
#
# `ambiguous` has no list like this and is deliberately not given one. pandas
# does not check the argument's vocabulary on a column at all: it carries
# whatever arrived down to the point where a wall clock hour turns out to be two
# instants, and `.dt.tz_localize('UTC', ambiguous=3)` comes back with an answer
# in it. Adding a check here would be firepanda refusing input pandas accepts,
# which is the direction of difference this library does not get to have. The
# scalar in `_scalars.py` does check it, because `Timestamp.tz_localize` does,
# and the two files differ there because pandas differs there.
_NONEXISTENT = ("raise", "NaT", "shift_forward", "shift_backward")
_NONEXISTENT_REFUSAL = (
    "The nonexistent argument must be one of 'raise', 'NaT', 'shift_forward',"
    " 'shift_backward' or a timedelta object"
)


def _spelled(value: Any, allowed: tuple[str, ...], message: str) -> None:
    """Refuses a value that is not in the argument's vocabulary at all.

    This is the question `_held_at` is not asking, and running the two together
    is what put the wrong class on three refusals.

    An argument like `interpolation` has a fixed vocabulary and two ways to be
    wrong. `interpolation="lower"` is a value pandas accepts and firepanda has
    not implemented, which is `NotImplementedError` and is a schedule. And
    `interpolation="not a method"` is a value pandas does not accept either,
    which is a typo, and pandas answers it with a `ValueError` naming the words
    that would have worked. Answering both with `NotImplementedError` tells
    somebody who misspelled `midpoint` that firepanda has not got round to their
    spelling yet, which is not true and sends them to the changelog instead of to
    their own line.

    So this is asked first and `_held_at` second. The message is built by the
    caller rather than assembled here, because each of these is pandas' own
    sentence and pandas words every one of them differently: `Invalid method: x.
    Method must be in {'table', 'single'}.` for one, `'x' is not a valid method.
    Use one of:` for another. Document 22 is why the wording follows pandas and
    where it deliberately does not.

    Not every argument with a fixed vocabulary gets one of these, and the test
    is whether pandas checks. `ambiguous` reads like the same shape and is not:
    pandas carries whatever arrived down to the point where an hour turns out to
    be two instants and complains about the hour rather than about the argument.
    A check here would refuse input pandas accepts, which is the one direction of
    difference this library does not get to have.

    Args:
        value: What was passed.
        allowed: The values pandas accepts, which is not the same list as the
            values firepanda implements and is usually longer.
        message: The refusal, worded the way pandas words it.

    Raises:
        InvalidArgumentError: If the value is not one of them.
    """
    if value not in allowed:
        raise InvalidArgumentError(message)


def _reducing_axis(axis: Any, owner: str) -> None:
    """Refuses a reduction along the second axis.

    A series has one axis and this is a check on the spelling. A frame has two
    and only one of them reduces: `df.sum()` runs down each column, which is a
    pass over contiguous memory per column, and `df.sum(axis=1)` runs across each
    row, which reads one value out of every column and is a different kernel
    rather than the same one transposed.

    Args:
        axis: What was passed.
        owner: The class name, for the message.

    Raises:
        InvalidArgumentError: If it is not an axis this class has.
        NotImplementedError: If it is the second axis of a frame.
    """
    allowed = (0, 1) if owner == "DataFrame" else (0,)
    if _axis_number(axis, owner, 0, allowed) == 1:
        raise NotImplementedError(
            "axis=1 is not supported yet, because reducing across a row reads one"
            " value out of every column and is a different kernel from the one"
            " that runs down a column"
        )


def _quantile_wanted(q: Any, interpolation: str) -> float:
    """Reads the one quantile a reduction can answer.

    pandas takes a list of quantiles as well as one, and answers a series for a
    series and a frame for a frame. That is a different shape rather than a
    longer loop, so it is refused by shape here and the scalar is what crosses.

    Args:
        q: The quantile, between zero and one.
        interpolation: How to land between two values.

    Returns:
        The quantile as a float.

    Raises:
        InvalidArgumentError: If it is not a number between zero and one, or if
            the interpolation is not one of the thirteen pandas takes.
        NotImplementedError: If it is a list, or if the interpolation is one
            pandas takes and firepanda has not written.
    """
    _spelled(
        interpolation,
        _INTERPOLATIONS,
        f"{interpolation!r} is not a valid method. Use one of: " + ", ".join(_INTERPOLATIONS),
    )
    _held_at(
        "interpolation",
        interpolation,
        "linear",
        "the reduction lands between two values by weighting them and the other"
        " twelve rules pick one of them or reweight the whole sample instead",
    )
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        raise NotImplementedError(
            "q has to be a single quantile for now, because a list of them"
            " answers a Series rather than a value and that is a different shape"
        )
    if not 0.0 <= float(q) <= 1.0:
        raise InvalidArgumentError(f"percentiles should all be in the interval [0, 1]. Try {q!r}")
    return float(q)


# Every spelling of a type firepanda has, and the one name firepanda prints for
# it. The table is long because pandas resolves a dtype through numpy and numpy
# has spent thirty years collecting names, so `int64` is also `int`, `int_`,
# `intp`, `long`, `longlong`, `l`, `q`, `p` and `i8`, and a program written by
# somebody who learned numpy first uses whichever of those they learned. Every
# row was measured against a running pandas 3.0.3 rather than read, which is how
# the surprises got in: `i` is int32 and not int64, `u` is not a name at all
# though `i`, `f` and `b` are, `long` is int64 while `longdouble` is not a
# float64 anywhere but on arm, and `unicode`, `str_`, `U` and `O` are all the
# object dtype rather than text.
#
# Two rows are platform dependent and are written as they measure on the
# machines this is built for, which is what pandas would report there too, since
# both libraries are asking the same C compiler how wide a `long` is: `long` and
# `uint` are sixty four bits. A third, `longdouble`, is platform dependent in a
# way that cannot be written down as one row and is refused below instead.
_DTYPE_NAMES: dict[str, str] = {
    "bool": "bool",
    "bool_": "bool",
    "?": "bool",
    "b1": "bool",
    "int8": "int8",
    "byte": "int8",
    "b": "int8",
    "i1": "int8",
    "int16": "int16",
    "short": "int16",
    "h": "int16",
    "i2": "int16",
    "int32": "int32",
    "intc": "int32",
    "i": "int32",
    "i4": "int32",
    "int64": "int64",
    "int": "int64",
    "int_": "int64",
    "intp": "int64",
    "long": "int64",
    "longlong": "int64",
    "l": "int64",
    "q": "int64",
    "p": "int64",
    "i8": "int64",
    "uint8": "uint8",
    "ubyte": "uint8",
    "B": "uint8",
    "u1": "uint8",
    "uint16": "uint16",
    "ushort": "uint16",
    "H": "uint16",
    "u2": "uint16",
    "uint32": "uint32",
    "uintc": "uint32",
    "I": "uint32",
    "u4": "uint32",
    "uint64": "uint64",
    "uint": "uint64",
    "uintp": "uint64",
    "ulong": "uint64",
    "ulonglong": "uint64",
    "L": "uint64",
    "Q": "uint64",
    "P": "uint64",
    "u8": "uint64",
    "float16": "float16",
    "half": "float16",
    "e": "float16",
    "f2": "float16",
    "float32": "float32",
    "single": "float32",
    "f": "float32",
    "f4": "float32",
    "float64": "float64",
    "double": "float64",
    "float": "float64",
    "d": "float64",
    "f8": "float64",
    "str": "string",
    "string": "string",
    # The only name here that is not a layout. It builds the categories as well
    # as the codes, which is a pass over the column rather than a conversion of
    # one, and it is spelled as a name anyway because that is how pandas asks
    # for it. What comes back is unordered over int32 codes, and a caller who
    # wants an ordering or a fixed set of categories asks with a
    # `CategoricalDtype` rather than with a word.
    "category": "category",
}

_NO_OBJECT = (
    "Arrow has no type that holds anything at all, so there is no column for an"
    " object dtype to be. `unicode`, `str_`, `U` and `O` are all spellings of it,"
    " which is a numpy wart rather than a firepanda one: the type that holds text"
    " is `str`"
)
"""Six spellings share this, so it is written once. The last sentence is there
because a caller who wrote `U` and got told there is no object column has been
answered accurately and not usefully, since what they wanted was text."""

_NO_BYTES = (
    "the fixed width byte string pads every value out to the longest one, so"
    " `bytes` is really `|S21` on a column of small integers, and firepanda's"
    " binary column is variable width"
)
"""Four spellings share this one. The width in the message is not a typo: pandas
picks it from the widest value the column would render to."""

_NO_LONGDOUBLE = (
    "longdouble is whatever extended precision float the machine has, which is"
    " an eighty bit float stored in sixteen bytes on x86 and a plain float64 on"
    " arm, and firepanda has no float wider than float64 on either. Asking for"
    " it and getting float64 back would be half the precision on the machines"
    " where the name means something. The float64 is `double`"
)
"""Two spellings share this one, and it is the only name in the table whose
answer changes with the machine rather than with the argument. Every other
platform dependent row here, `long` and `uint`, is the same width everywhere
firepanda is built for, so it can be written down. This one cannot be, and
mapping it to float64 was correct on arm and a silent narrowing on x86."""

# The types pandas has and firepanda does not, each with the reason it is
# refused rather than converted. Every one of these raises today by being
# absent, and raises tomorrow by being declared and refused, which is the
# difference between a caller finding out and a caller getting a number.
#
# The four temporal and binary rows are the ones worth reading twice, because
# each of them names a type firepanda has and is refused anyway.
# `datetime64[ns]`, `timedelta64[ns]` and `date32[day]` are refused because the
# cast underneath falls through to the physical layout and hands back the
# integers the instants, spans and days are stored as. A column of instants that
# came back as a column of nanosecond counts is the kind of wrong answer that
# looks right in a repl. `binary` is refused because the cast to it hands back
# text. All four are worth fixing in the kernel and none of them is worth
# shipping as a silent wrong answer in the meantime.
_REFUSED_DTYPES: dict[str, str] = {
    "object": _NO_OBJECT,
    "object_": _NO_OBJECT,
    "unicode": _NO_OBJECT,
    "str_": _NO_OBJECT,
    "O": _NO_OBJECT,
    "U": _NO_OBJECT,
    "datetime64": (
        "the cast underneath converts layouts and these are int64 underneath,"
        " so the answer would be a column of counts rather than of instants."
        " Reading a column as instants is to_datetime and changing the unit of"
        " one that already is, is dt.as_unit"
    ),
    "timedelta64": (
        "the cast underneath converts layouts and these are int64 underneath,"
        " so the answer would be a column of counts rather than of spans"
    ),
    "date32[day]": (
        "the same reason the two above are refused, one type narrower: a date is"
        " an int32 day number underneath and the cast would hand back the day"
        " numbers"
    ),
    "binary": (
        "the cast to it hands back a column of text rather than a column of"
        " bytes, and nothing in the pandas facing layer can build a binary column"
        " to convert from, so the name has nothing to do here yet"
    ),
    "complex": "there is no complex column",
    "cdouble": "there is no complex column",
    "csingle": "there is no complex column",
    "clongdouble": "there is no complex column",
    "longdouble": _NO_LONGDOUBLE,
    "g": _NO_LONGDOUBLE,
    "period": "there is no period column",
    "interval": "there is no interval column",
    "bytes": _NO_BYTES,
    "bytes_": _NO_BYTES,
    "S": _NO_BYTES,
    "void": "there is no void column",
    "V": "there is no void column",
}

_NULLABLE_DTYPES: frozenset[str] = frozenset(
    ["boolean"]
    + [f"{kind}{bits}" for kind in ("Int", "UInt") for bits in (8, 16, 32, 64)]
    + [f"Float{bits}" for bits in (32, 64)]
)
"""The extension types pandas spells with a capital letter, which are a second
missing value model rather than a second set of widths. They are listed rather
than matched on the capital, because `Int64` and `int64` differing only in a
letter is the kind of thing a reader has to be able to see the whole of."""

_NO_NULLABLE = (
    "the nullable extension types are a second way of spelling a missing value"
    " and firepanda has one way, which is the Arrow one. The lower case name is"
    " the same width and already holds missing values"
)

# What a python type means as a dtype, measured the same way. These are not
# strings and cannot be looked up in the table above, and pandas takes them, so
# `s.astype(float)` and `s.astype(int)` both work here too. `complex` and
# `bytes` are in the refused table by their own names, and `object` is refused
# under its own name as well, so all three arrive at the same message whether
# the caller wrote the word or the type.
_DTYPE_TYPES: dict[type, str] = {
    bool: "bool",
    int: "int64",
    float: "float64",
    str: "str",
    complex: "complex",
    bytes: "bytes",
    object: "object",
}


def _named_dtype(dtype: Any) -> str:
    """Resolves whatever a caller wrote into the one name firepanda prints.

    Four shapes arrive here and pandas takes all four: a string, a python type,
    a numpy dtype object, and a numpy scalar type. The last two are read by
    duck typing rather than by importing numpy, since firepanda does not depend
    on numpy and a compatibility layer that imported it in order to read a name
    off it would have acquired the dependency for one attribute.

    Args:
        dtype: What the caller passed.

    Returns:
        The canonical name, which is what `dtype` prints and what the extension
        reads.

    Raises:
        NotImplementedError: If the name is a type pandas has and firepanda
            does not.
        DTypeError: If it is not a type name at all. pandas raises a `TypeError`
            here rather than a `ValueError`, which reads oddly for a bad string
            and is what a caller catching pandas already catches.
    """
    if isinstance(dtype, type):
        named = _DTYPE_TYPES.get(dtype)
        # A numpy scalar type is a type whose name is already in the table,
        # which is what makes `s.astype(numpy.int8)` work without numpy being
        # importable here.
        name = named if named is not None else getattr(dtype, "__name__", "")
    elif isinstance(dtype, str):
        name = dtype
    else:
        # A numpy dtype object carries its spelling on `.name`, and so does a
        # pandas extension dtype. Anything else is rendered and will fail the
        # lookup below with what the caller wrote in the message.
        carried = getattr(dtype, "name", None)
        name = carried if isinstance(carried, str) else str(dtype)
    # Arrow is little endian everywhere, so the three byte order marks that mean
    # native, little or not applicable are dropped and the one that means big is
    # refused. A program that wrote `>i8` meant it.
    #
    # The mark only comes off when what is left is a name, which matters because
    # an object that is not a dtype at all was rendered above and `str(object())`
    # begins with a `<`. Stripping that unconditionally would put the angle
    # bracket in the message on one side and not the other.
    if name[:1] in {"<", "=", "|"} and name[1:] in _DTYPE_NAMES:
        name = name[1:]
    if name[:1] == ">":
        raise NotImplementedError(f"{name!r} is big endian and every Arrow buffer is little endian")
    known = _DTYPE_NAMES.get(name)
    if known is not None:
        return known
    if name in _NULLABLE_DTYPES:
        raise NotImplementedError(f"dtype {name!r} is not supported, because {_NO_NULLABLE}")
    # Not an exact match, because four of these carry something after the name:
    # a unit in `datetime64[ns]`, a frequency in `period[D]`, a width in
    # `complex128` and a length in `S21`. What may follow is spelled out rather
    # than left to `startswith` alone, since `S` is a key and a bare prefix test
    # would answer that every name beginning with an S is a byte string.
    for key, why in _REFUSED_DTYPES.items():
        rest = name[len(key) :] if name.startswith(key) else None
        if rest is None or not (rest == "" or rest[:1] == "[" or rest.isdigit()):
            continue
        raise NotImplementedError(f"dtype {name!r} is not supported, because {why}")
    raise DTypeError(f"data type {name!r} not understood")


# The words that name a set of types rather than one type, which is numpy's type
# tree with the branches that matter written out. `select_dtypes` is the only
# caller, and every other place a type is named takes one type and goes through
# `_named_dtype`, which is why these are not in `_DTYPE_NAMES`: `float` there
# means float64 and `float` here means every float, and both readings are right
# for their own caller.
#
# Two rows are worth reading twice. `int` and `float` are the concrete names
# numpy resolves them to and pandas widens them back out by hand, so `int` takes
# int32 and int64 and not the unsigned ones, while `integer` takes all of them.
# And `timedelta64` is under `signedinteger`, which is why a column of spans is
# selected by `number`. That is a numpy fact rather than a pandas one, pandas
# inherits it without comment, and a compatibility layer that tidied it up would
# answer a different question from the one the caller's pandas answers.
_DTYPE_GROUPS: dict[str, tuple[str, ...]] = {
    "number": ("signed", "unsigned", "floating", "span"),
    "integer": ("signed", "unsigned", "span"),
    "signedinteger": ("signed", "span"),
    "unsignedinteger": ("unsigned",),
    "inexact": ("floating",),
    "floating": ("floating",),
    "datetime": ("naive",),
    "datetime64": ("naive",),
    "datetimetz": ("aware",),
    "datetime64tz": ("aware",),
    "timedelta": ("span",),
    "timedelta64": ("span",),
}
"""Each word, and the family names a column has to carry to answer to it."""

_DTYPE_WIDENED: dict[str, tuple[str, ...]] = {
    "int": ("int32", "int64"),
    "float": ("float32", "float64"),
}
"""The two words that become several types rather than a branch of the tree.

numpy resolves a bare `int` to one concrete type whose width depends on the
platform, and pandas widens it back out to both signed widths by hand so that
the same code selects the same columns everywhere. `float` is widened the same
way. They are here rather than in the table above because they become concrete
names, and that is visible: `include="int"` against `exclude="int64"` is an
overlap and `include="integer"` against `exclude="int64"` is not."""

_SIGNED: frozenset[str] = frozenset({"int8", "int16", "int32", "int64"})
"""The four signed widths, which are `signed` and everything above it."""

_UNSIGNED: frozenset[str] = frozenset({"uint8", "uint16", "uint32", "uint64"})
"""The four unsigned widths, which are not under `int` and are under `integer`."""

_FLOATING: frozenset[str] = frozenset({"float16", "float32", "float64"})
"""The three float widths."""


def _dtype_family(printed: str) -> frozenset[str]:
    """Every word a column's type answers to, which is its branch of the tree.

    Args:
        printed: The type as `dtype` spells it, such as `int64` or
            `datetime64[ns, UTC]`.

    Returns:
        The type's own name and the family names above it.
    """
    if printed in _SIGNED:
        return frozenset({printed, "signed"})
    if printed in _UNSIGNED:
        return frozenset({printed, "unsigned"})
    if printed in _FLOATING:
        return frozenset({printed, "floating"})
    if printed.startswith("datetime64["):
        # The zone is written into the name after a comma, so a column with one
        # is `aware` and a column without one is `naive`, and the two are
        # different branches rather than one branch and a flag. pandas does the
        # same and that is why `datetime64` does not select a column that has a
        # zone on it.
        return frozenset({printed, "aware" if "," in printed else "naive"})
    if printed.startswith("timedelta64["):
        return frozenset({printed, "span"})
    return frozenset({printed})


def _dtype_words(spec: Any) -> frozenset[str]:
    """Reads one side of `select_dtypes` into the words the caller named.

    These are the words and not yet the branches, because the two sides are
    compared for an overlap before either is expanded and pandas compares them
    in this form. `number` and `floating` are two words and do not overlap even
    though every float answers to both.

    Args:
        spec: What the caller passed, which is a name, a type, or a list of
            either, or `None` for the side they did not pass.

    Returns:
        One word per type or family named, empty when the side was not passed.

    Raises:
        NotImplementedError: If a name is a type pandas has and firepanda does
            not.
        DTypeError: If a name is not a type name at all.
    """
    if spec is None:
        return frozenset()
    given = spec if isinstance(spec, (list, tuple, set, frozenset)) else [spec]
    out: set[str] = set()
    for one in given:
        # A python type and a numpy type class both carry their spelling on
        # `__name__`, which is how `numpy.number` is read without numpy being
        # importable here. A string is already the spelling.
        word = one if isinstance(one, str) else getattr(one, "__name__", "")
        widened = _DTYPE_WIDENED.get(word)
        if widened is not None:
            out.update(widened)
        elif word in _DTYPE_GROUPS:
            out.add(word)
        else:
            out.add(_named_dtype(one))
    return frozenset(out)


def _dtype_branches(words: frozenset[str]) -> frozenset[str]:
    """Expands the words one side named into the branches they select.

    Args:
        words: What `_dtype_words` answered.

    Returns:
        Every branch name and every concrete type name the side selects.
    """
    out: set[str] = set()
    for word in words:
        out.update(_DTYPE_GROUPS.get(word, (word,)))
    return frozenset(out)


class _NoDefault:
    """The sentinel pandas puts where a default has to mean "nothing was passed".

    pandas has one of these, `pandas._libs.lib.no_default`, and it appears as the
    default of `shift(fill_value=)`, `dropna(how=)`, `dropna(thresh=)` and
    several more. It exists because `None` is a value a caller might mean, so a
    parameter that treats a missing argument differently from an explicit `None`
    needs a third thing, and pandas made one.

    firepanda needs the same third thing on the same parameters and cannot
    borrow theirs, because importing pandas to be compatible with pandas would
    make the library depend on the thing it replaces. So there is one here, and
    the signature parity test knows the two sentinels are the same idea wearing
    different names. That is the only place the difference is visible, since a
    caller can neither construct one nor tell them apart from the outside.
    """

    def __repr__(self) -> str:
        """Reads the way the pandas one does, so a signature prints the same."""
        return "<no_default>"


NO_DEFAULT = _NoDefault()
"""The one instance, compared by identity the way the pandas one is."""


def _cast_keywords(copy: Any, errors: Any) -> bool:
    """Reads the two keywords `astype` carries besides the type itself.

    `copy` is deprecated in pandas 3 and does nothing there, because copy on
    write already decides when a copy happens. It does nothing here either, for
    a different reason: an answer is always a new column over new buffers. So it
    warns and is ignored, which is what pandas does, rather than being refused,
    because refusing it would break code that passes it and gets no complaint
    from pandas.

    `errors` is the one that changes behaviour. It says what a value that will
    not convert means, and it is read here rather than sent to the kernel,
    because the kernel's flag is a different question: it asks whether a bad
    value becomes missing, and pandas never asks for that.

    Args:
        copy: What the caller passed, or `NO_DEFAULT`.
        errors: Either `"raise"` or `"ignore"`.

    Returns:
        True if a value that will not convert should raise.

    Raises:
        InvalidArgumentError: If `errors` is neither of the two words.
    """
    if copy is not NO_DEFAULT:
        warnings.warn(
            "The copy keyword is deprecated and will be removed in a future"
            " version. Copy-on-Write is active in pandas since 3.0 which utilizes"
            " a lazy copy mechanism that defers copies until necessary. Use .copy()"
            " to make an eager copy if necessary.",
            DeprecationWarning,
            stacklevel=3,
        )
    if errors not in ("raise", "ignore"):
        raise InvalidArgumentError(
            "Expected value of kwarg 'errors' to be one of ['raise', 'ignore']."
            f" Supplied value is '{errors}'"
        )
    return bool(errors == "raise")


def _limit_wanted(limit: Any) -> int:
    """Reads a fill limit, where the absence of one means as far as it goes.

    pandas spells no limit as `None` and the core spells it as zero, and this is
    the one line where the two meet. A caller who writes `limit=0` meant
    something and it is not "no limit", so it is rejected here with the pandas
    message rather than being read as its own opposite.

    Args:
        limit: What was passed.

    Returns:
        The limit, or zero for no limit.

    Raises:
        InvalidArgumentError: If it is not a positive whole number.
    """
    if limit is None:
        return 0
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise InvalidArgumentError("Limit must be an integer")
    if limit <= 0:
        raise InvalidArgumentError("Limit must be greater than 0")
    return limit


def _transforming_axis(axis: Any, owner: str) -> None:
    """Refuses a transformation along the second axis.

    Same shape as `_reducing_axis` and the same reason, one step further along.
    `df.cumsum()` totals down each column, which is a pass over contiguous
    memory, and `df.cumsum(axis=1)` totals across each row, which touches every
    column once per row. A frame is stored as columns, so those are two
    different kernels and not one kernel pointed the other way.

    Args:
        axis: What was passed.
        owner: The class name, for the message.

    Raises:
        InvalidArgumentError: If it is not an axis this class has.
        NotImplementedError: If it is the second axis of a frame.
    """
    allowed = (0, 1) if owner == "DataFrame" else (0,)
    if _axis_number(axis, owner, 0, allowed) == 1:
        raise NotImplementedError(
            "axis=1 is not supported yet, because a frame is stored as columns"
            " and running across a row touches every one of them per row, which"
            " is a different kernel from the one that runs down a column"
        )


class _Every:
    """The stand in for an axis the caller did not mention.

    `df.iloc[2:5]` names one axis and means every column, and `df.iloc[:, 1]`
    names both and means every row, so the two cases have to be told apart from
    `df.iloc[None]`, which is neither. A sentinel does that and `slice(None)`
    would not, because a caller can write `slice(None)` and mean it.
    """

    def __repr__(self) -> str:
        """Reads as what it stands for, since it can reach a message."""
        return "<every>"


EVERY = _Every()
"""The one instance, compared by identity."""


def _two_axes(key: Any) -> tuple[Any, Any]:
    """Splits a subscript into a row key and a column key.

    Args:
        key: Whatever went inside the square brackets.

    Returns:
        The row key, and the column key or `EVERY`.

    Raises:
        InvalidArgumentError: If more than two axes were named.
    """
    if isinstance(key, tuple):
        if len(key) > 2:
            raise InvalidArgumentError("Too many indexers")
        if len(key) == 2:
            return key[0], key[1]
        return (key[0] if key else EVERY), EVERY
    return key, EVERY


def _is_mask(key: Any) -> bool:
    """Whether a key is a run of booleans rather than of positions or labels.

    `True` is an `int` in Python, so a list of booleans is also a list of
    integers by the loose reading and the two mean completely different rows.
    The exact check is the only one that tells them apart.

    Args:
        key: A list or tuple.

    Returns:
        True if every element is a bool and there is at least one.
    """
    return bool(key) and all(isinstance(one, bool) for one in key)


def _flattened(found: Any) -> list[int]:
    """Turns what `Index.get_loc` answers into a list of positions.

    `get_loc` has three return types and which one it uses depends on the
    labels rather than on the argument, which is pandas' rule. Every caller
    here wants positions, so this is where the three become one.

    Args:
        found: An integer, a slice or a boolean mask.

    Returns:
        The positions, in index order.
    """
    if isinstance(found, int):
        return [found]
    if isinstance(found, slice):
        return list(range(found.start, found.stop))
    return [i for i, hit in enumerate(found) if hit]


def _row_positions(index: Any, labels: Any) -> list[int]:
    """Where each of a list of labels sits, with every hit for a repeated one.

    Args:
        index: The frame's index.
        labels: The labels asked for, in the order the caller wrote them.

    Returns:
        The positions, in the order the labels were asked for.

    Raises:
        KeyError: If any label is not in the index, naming all of them.
    """
    found: list[int] = []
    missing: list[Any] = []
    for label in labels:
        try:
            found.extend(_flattened(index.get_loc(label)))
        except KeyError:
            missing.append(label)
    if missing:
        raise KeyError(f"{missing} not in index")
    return found


def _column_positions(names: list[str], key: Any) -> list[int]:
    """Which columns a key names, by position, for a key that is not one column.

    Args:
        names: The column names, in order.
        key: A slice, a list of positions or a list of booleans.

    Returns:
        The column positions.

    Raises:
        InvalidArgumentError: If the key is none of those shapes.
    """
    if isinstance(key, slice):
        return list(range(*key.indices(len(names))))
    if isinstance(key, (list, tuple)):
        if _is_mask(key):
            if len(key) != len(names):
                raise InvalidArgumentError(f"Item wrong length {len(key)} instead of {len(names)}")
            return [i for i, hit in enumerate(key) if hit]
        return [int(one) for one in key]
    raise InvalidArgumentError(f"cannot select columns with a {type(key).__name__}")


def _named_at(names: list[str], positions: list[int]) -> list[str]:
    """Turns column positions into column names, counting from the end.

    Args:
        names: The column names, in order.
        positions: The positions.

    Returns:
        The names.

    Raises:
        OutOfBoundsError: If a position is off either end.
    """
    out = []
    for one in positions:
        at = one + len(names) if one < 0 else one
        if at < 0 or at >= len(names):
            raise OutOfBoundsError(
                f"index {one} is out of bounds for axis 0 with size {len(names)}"
            )
        out.append(names[at])
    return out


def _counts_rather_than_names(key: slice) -> bool:
    """Whether `s[key]` reads its slice as positions rather than as labels.

    pandas decides this from the bounds and not from the index, and a bound
    that is `None` says nothing either way, so an open slice counts. `True` is
    an integer in Python and is not a position, for the reason `_is_mask`
    exists.

    Args:
        key: The slice.

    Returns:
        True if every bound that is written is a whole number.
    """
    written = [one for one in (key.start, key.stop) if one is not None]
    return all(isinstance(one, int) and not isinstance(one, bool) for one in written)


def _by_position(key: Any, height: int) -> tuple[Any, ...]:
    """Reads a row key as positions, which is what `iloc` does on either type.

    A slice of step one is a range, which the core answers by sharing buffers
    rather than gathering, and any other step is a gather over the positions
    the slice walks. Building the walk in Python is right rather than a
    shortcut: a step is rare, the positions are what the gather takes anyway,
    and a second kernel for it would be the gather again.

    A boolean key is a mask even here. `iloc` is about positions and a mask is
    not one, but pandas takes it and reads it as the positions it is true at,
    which is what it means and is what anybody writing `s.iloc[s > 0]` wants.

    Args:
        key: Whatever went inside the square brackets, for the row axis.
        height: How many rows there are.

    Returns:
        A tag and its arguments, which `_narrowed` applies.

    Raises:
        OutOfBoundsError: If a boolean key is not as long as the axis.
    """
    if key is EVERY:
        return ("every",)
    if isinstance(key, SeriesMixin) and key._inner.dtype() == "bool":
        return ("mask", key._inner)
    if isinstance(key, slice):
        start, stop, step = key.indices(height)
        if step == 1:
            return ("range", start, max(start, stop))
        return ("gather", list(range(start, stop, step)))
    if isinstance(key, (list, tuple)):
        if _is_mask(key):
            if len(key) != height:
                raise OutOfBoundsError(
                    f"Boolean index has wrong length: {len(key)} instead of {height}"
                )
            return ("gather", [i for i, hit in enumerate(key) if hit])
        return ("gather", [int(one) for one in key])
    if isinstance(key, SeriesMixin):
        held: list[Any] = key._inner.to_list()
        return ("gather", [int(one) for one in held])
    return ("one", int(key))


def _by_label(index: Any, key: Any, height: int) -> tuple[Any, ...]:
    """Reads a row key as labels, which is what `loc` does on either type.

    A boolean key is the one shape that is not a label at all, and it is
    checked for first because a column of booleans used as a mask is the most
    written `loc` there is. A mask that arrived as a column stays a column and
    crosses as one, and a mask that arrived as a list of Python bools becomes
    positions here, because it was already objects and building a column out of
    it to ask the kernel would be a conversion each way to answer a question a
    comprehension answers.

    Args:
        index: The labels of the thing being indexed.
        key: Whatever went inside the square brackets, for the row axis.
        height: How many rows there are.

    Returns:
        A tag and its arguments, which `_narrowed` applies.

    Raises:
        KeyError: If a label is not in the index.
        OutOfBoundsError: If a boolean key is not as long as the axis.
    """
    if key is EVERY:
        return ("every",)
    if isinstance(key, SeriesMixin) and key._inner.dtype() == "bool":
        return ("mask", key._inner)
    if isinstance(key, slice):
        walked = index.slice_indexer(key.start, key.stop, key.step)
        start, stop, step = walked.indices(height)
        if step == 1:
            return ("range", start, max(start, stop))
        return ("gather", list(range(start, stop, step)))
    if isinstance(key, (list, tuple)):
        if _is_mask(key):
            if len(key) != height:
                raise OutOfBoundsError(
                    f"Boolean index has wrong length: {len(key)} instead of {height}"
                )
            return ("gather", [i for i, hit in enumerate(key) if hit])
        return ("gather", _row_positions(index, key))
    if isinstance(key, IndexMixin):
        return ("gather", _row_positions(index, key._inner.to_list()))
    try:
        found = index.get_loc(key)
    except KeyError:
        # The label itself and nothing else, which is what pandas raises and
        # what a person grepping their own traceback is looking for. The
        # message underneath names the index rather than the label, because it
        # was written for a caller who already has the label in hand.
        raise KeyError(key) from None
    if isinstance(found, int):
        return ("one", found)
    return ("gather", _flattened(found))


class _Selection:
    """What `loc` and `iloc` have in common, which is everything after the key.

    The two differ only in how a key becomes a set of rows and a set of
    columns. Once a key has been read, what is done with the answer is the
    same on both, and it is this class: narrow to the columns, then narrow to
    the rows, then decide whether what comes out is a frame, a column or a
    single value.

    That order is not arbitrary. Narrowing the columns first means the row
    gather copies only the columns the caller asked for, which on a wide frame
    is the difference between copying five columns and copying five hundred.
    """

    __slots__ = ("_owner",)

    def __init__(self, owner: Any) -> None:
        """Holds the frame the accessor was reached through.

        Args:
            owner: The frame.
        """
        self._owner = owner

    def _rows(self, key: Any, height: int) -> tuple[Any, ...]:
        """Reads a row key, which is the half the two accessors do differently."""
        raise NotImplementedError

    def _columns(self, key: Any, names: list[str]) -> Any:
        """Reads a column key, which is the other half they do differently."""
        raise NotImplementedError

    def __getitem__(self, key: Any) -> Any:
        """Answers a frame, a column or a value, depending on the key's shape.

        The rule for which is pandas' and it is about the key rather than
        about the data: an axis named by one thing collapses and an axis named
        by a set of things does not. Two collapsed axes are a value, one is a
        column, and none is a frame.

        A single row with more than one column would be a row as a series, and
        that is refused rather than approximated. A row read across the columns
        has to find one type that all of them fit, which is a different
        operation from anything in this file and is written up in section 6 of
        document 36.
        """
        from ._frame import DataFrame, Series

        rows, columns = _two_axes(key)
        inner = self._owner._inner
        names = inner.names()
        picked = self._columns(columns, names)
        where = self._rows(rows, inner.length())
        try:
            if isinstance(picked, str):
                if where[0] == "one":
                    return inner.cell(where[1], names.index(picked))
                narrowed = _narrowed(inner.select([picked]), where)
                return Series._wrap(narrowed.column(picked))
            if where[0] == "one":
                raise NotImplementedError(
                    "reading one row across several columns is not supported yet,"
                    " because a row has to find one type that every column fits"
                    " and nothing here computes that type"
                )
            if picked is not EVERY:
                inner = inner.select(picked)
            return DataFrame._wrap(_narrowed(inner, where))
        except NotImplementedError:
            raise
        except Exception as error:
            raise translate(error) from None


def _narrowed(inner: Any, where: tuple[Any, ...]) -> Any:
    """Applies a row selection that has already been read to a frame.

    Args:
        inner: The extension frame.
        where: What `_rows` answered.

    Returns:
        A new extension frame, or the same one when nothing was selected.
    """
    if where[0] == "every":
        return inner
    if where[0] == "range":
        return inner.slice_rows(where[1], where[2])
    if where[0] == "mask":
        return inner.filter_rows(where[1])
    return inner.take(where[1])


class _Positional(_Selection):
    """`df.iloc`, where every key is a position and a slice excludes its end."""

    __slots__ = ()

    def _rows(self, key: Any, height: int) -> tuple[Any, ...]:
        """Reads a row key as positions."""
        return _by_position(key, height)

    def _columns(self, key: Any, names: list[str]) -> Any:
        """Reads a column key as positions."""
        if key is EVERY:
            return EVERY
        if isinstance(key, int) and not isinstance(key, bool):
            return _named_at(names, [int(key)])[0]
        return _named_at(names, _column_positions(names, key))


class _Labelled(_Selection):
    """`df.loc`, where every key is a label and a slice includes its end."""

    __slots__ = ()

    def _rows(self, key: Any, height: int) -> tuple[Any, ...]:
        """Reads a row key as labels."""
        return _by_label(self._owner.index, key, height)

    def _columns(self, key: Any, names: list[str]) -> Any:
        """Reads a column key as names.

        A slice of names includes the column it stops at, for the same reason
        a slice of row labels does, and it is resolved here rather than through
        the index because the column names are a plain list on this side.
        """
        if key is EVERY:
            return EVERY
        if isinstance(key, str):
            if key not in names:
                raise KeyError(key)
            return key
        if isinstance(key, slice):
            first = 0 if key.start is None else names.index(key.start)
            last = len(names) if key.stop is None else names.index(key.stop) + 1
            return names[first:last]
        if isinstance(key, (list, tuple)):
            if _is_mask(key):
                return _named_at(names, _column_positions(names, key))
            missing = [one for one in key if one not in names]
            if missing:
                raise KeyError(f"{missing} not in index")
            return [str(one) for one in key]
        raise InvalidArgumentError(f"cannot select columns with a {type(key).__name__}")


class _Cell:
    """`df.at` and `df.iat`, which are one value and nothing else.

    They are one class rather than two because the only difference is whether
    the pair of coordinates are labels or positions, and pandas has two classes
    for the same reason it has `loc` and `iloc`, which is that a caller has to
    say which they mean. Saying it once, at the property, is enough.

    What they are for is speed. `df.loc[label, name]` reaches the same value
    through the same lookup, and the reason to write `df.at[label, name]`
    instead is that it promises never to answer anything but a value, so it can
    skip every branch that decides what shape the answer has.
    """

    __slots__ = ("_labelled", "_owner")

    def __init__(self, owner: Any, labelled: bool) -> None:
        """Holds the frame and which of the two this is.

        Args:
            owner: The frame.
            labelled: True for `at`, False for `iat`.
        """
        self._owner = owner
        self._labelled = labelled

    def __getitem__(self, key: Any) -> Any:
        """Reads one value, by a pair of coordinates.

        Both coordinates are required, which is pandas' rule and is the whole
        point: a single coordinate would leave a shape to decide and deciding
        a shape is what these two exist to avoid.
        """
        name = "at" if self._labelled else "iat"
        if not isinstance(key, tuple) or len(key) != 2:
            raise InvalidArgumentError(f"{name} takes a row and a column, and got one key")
        row, column = key
        inner = self._owner._inner
        if not self._labelled:
            try:
                return inner.cell(int(row), int(column))
            except Exception as error:
                raise translate(error) from None
        names = inner.names()
        if column not in names:
            raise KeyError(column)
        found = self._owner.index.get_loc(row)
        if not isinstance(found, int):
            raise NotImplementedError(
                "at on a repeated label is not supported yet, because pandas"
                " answers several values there and this answers one"
            )
        try:
            return inner.cell(found, names.index(column))
        except Exception as error:
            raise translate(error) from None


class _Along:
    """`s.loc` and `s.iloc`, which are the frame's pair with one axis gone.

    One class rather than two, with a flag, because everything they do after
    the key has been read is the same and the key reading is two functions
    they share with the frame. A series has no column axis, so the whole of
    `_Selection` that decides between a frame, a column and a value collapses
    to deciding between a series and a value, and that decision is the shape of
    the key and nothing else: a key that names one row is a value and a key
    that names a set of them is a series, even a set of one.
    """

    __slots__ = ("_labelled", "_owner")

    def __init__(self, owner: Any, labelled: bool) -> None:
        """Holds the series the accessor was reached through.

        Args:
            owner: The series.
            labelled: True for `loc`, False for `iloc`.
        """
        self._owner = owner
        self._labelled = labelled

    def __getitem__(self, key: Any) -> Any:
        """Answers a value or a series, depending on the key's shape."""
        from ._frame import Series

        if isinstance(key, tuple):
            # A tuple is how a caller names a second axis, and there is not one
            # to name. pandas raises its own IndexingError here, which this
            # cannot be a subclass of without importing pandas, so the sentence
            # is pandas' and the class is not.
            raise InvalidArgumentError("Too many indexers")
        inner = self._owner._inner
        height = inner.length()
        if self._labelled:
            where = _by_label(self._owner.index, key, height)
        else:
            where = _by_position(key, height)
        try:
            if where[0] == "one":
                if not self._labelled and not -height <= where[1] < height:
                    # pandas says this rather than naming the position, and it
                    # says something else when the same position is handed to
                    # `iat`, so the two messages are raised by their two
                    # callers rather than by the one binding underneath.
                    raise OutOfBoundsError("single positional indexer is out-of-bounds")
                return inner.cell(where[1])
            return Series._wrap(_narrowed(inner, where))
        except Exception as error:
            raise translate(error) from None


class _Point:
    """`s.at` and `s.iat`, which are one value and one coordinate.

    The frame's pair take two coordinates because a frame has two axes. This
    takes one for the same reason, and the reason to write it rather than
    `s.loc[label]` is the same as it is over there: it promises to answer a
    value, so it can skip the branch that works out what shape the answer is.
    """

    __slots__ = ("_labelled", "_owner")

    def __init__(self, owner: Any, labelled: bool) -> None:
        """Holds the series and which of the two this is.

        Args:
            owner: The series.
            labelled: True for `at`, False for `iat`.
        """
        self._owner = owner
        self._labelled = labelled

    def __getitem__(self, key: Any) -> Any:
        """Reads one value, by one label or by one position."""
        inner = self._owner._inner
        if not self._labelled:
            try:
                return inner.cell(int(key))
            except Exception as error:
                raise translate(error) from None
        try:
            found = self._owner.index.get_loc(key)
        except KeyError:
            raise KeyError(key) from None
        if not isinstance(found, int):
            raise NotImplementedError(
                "at on a repeated label is not supported yet, because pandas"
                " answers several values there and this answers one"
            )
        try:
            return inner.cell(found)
        except Exception as error:
            raise translate(error) from None


__all__ = ["NO_DEFAULT", "DataFrameMixin", "IndexMixin", "SeriesMixin"]


class DataFrameMixin:
    """The hand written half of `DataFrame`."""

    __slots__ = ("_inner",)
    """The one piece of state, declared here rather than on the generated class.

    It has to be here because the constructor is here, and a class cannot assign
    to a slot it does not own. The generated subclass declares an empty
    `__slots__`, so an instance still has no `__dict__` and there is still
    exactly one place the extension object lives."""

    _inner: _firepanda.DataFrame

    if TYPE_CHECKING:

        @property
        def index(self) -> Index:
            """The row labels, declared here and defined by the generated class.

            Three members in this file reach the labels, and the property that
            answers them is in the table rather than here because it is one
            call. This says so for the type checker and is not compiled, which
            is why it is a property rather than an annotation: the generated
            one is read only and an annotation would promise a setter.
            """

    def __init__(
        self,
        data: Any = None,
        index: Any = None,
        columns: Any = None,
        dtype: Any = None,
        copy: bool | None = None,
    ) -> None:
        """Builds a frame from a mapping of column name to values.

        The signature is the pandas one in full and two of the five parameters
        are implemented. The other three are refused by name, which is a stronger
        statement than leaving them out: the signature parity test compares five
        parameters against pandas instead of one, and a caller who passes one of
        them is told what is missing rather than that the keyword is unexpected.

        `dtype=` is inference followed by a cast, which is what it is in pandas
        too. Building the frame first and converting it after is one pass more
        than reading the values into the asked for type directly would be, and it
        is the reading that decides what a value means, so the two agree on every
        answer and differ only in how much work they do.
        """
        _refuse("index", index, "putting labels on a frame as it is built is not written")
        _refuse("columns", columns, "selecting and reordering on the way in is not written")
        _refuse("copy", copy, "there is exactly one behaviour and it always copies")
        try:
            self._inner = _firepanda.DataFrame(data)
        except Exception as error:
            raise translate(error) from None
        if dtype is not None:
            names = self._inner.names()
            wanted = [_named_dtype(dtype)] * len(names)
            try:
                self._inner = self._inner.cast(names, wanted, True)
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

    def _get(self, key: Any, default: Any) -> Any:
        """One column, or a value of the caller's choosing when there is none.

        Square brackets with the failure turned into a value, and that is the
        whole method. It is the only difference between the two and it is the
        only reason this one exists, since a caller who already has a default
        ready does not want a traceback on the way to it.

        The failures caught here are one wider than the set pandas catches.
        pandas reads `df[0]` as a column it does not have and this reads it as a
        key of a type square brackets do not take, because square brackets here
        want a name or a list of names. Both objects answer the default, which
        is all a caller of this can see, and the class of the exception neither
        of them raises is where the two differ.
        """
        try:
            return self[key]
        except (IndexError, KeyError, TypeError, ValueError):
            return default

    def _squeeze(self, axis: Any) -> Any:
        """The frame with an axis of length one taken off it.

        Three shapes of answer out of one method, which is why it is here rather
        than in the table. A frame of one column is that column, a frame of one
        row and one column is the value in it, and a frame that is neither of
        those is itself. The axis names which of the two may be dropped, and
        `None`, the default, lets either go.

        The one answer refused is the one where the row axis goes and the
        column axis stays, which pandas gives as the frame's one row read
        across its columns. That row is a set of values of several types coming
        back as a series, which has one, and pandas' rule for choosing it sends
        a bool beside a number, or a string beside anything, to the object
        dtype, which this library does not have. Its name is the row label
        rather than a column name, which is the second thing in the way, since
        a series here is named by a string. Saying so is better than answering
        that row under some other type and some other name.
        """
        from ._frame import DataFrame

        names = self._inner.names()
        rows = self._inner.length()
        wanted = None if axis is None else _axis_number(axis, "DataFrame", 0, (0, 1))
        one_row = rows == 1 and wanted in (None, 0)
        one_column = len(names) == 1 and wanted in (None, 1)
        if one_row and one_column:
            try:
                return self._inner.cell(0, 0)
            except Exception as error:
                raise translate(error) from None
        if one_column:
            return self[names[0]]
        if one_row:
            raise UnsupportedError(
                "squeeze that drops the row axis and keeps the column axis is"
                " not written, because the answer is the row read across the"
                " columns, which is a series of one type where the columns it"
                " covers need not share one, under the row's label for a name"
                " where a series here is named by a string"
            )
        # A fresh wrapper around the same columns rather than `self`, because
        # pandas hands back something that is not the frame it was given even
        # when there was nothing to drop, and a caller who checks that is
        # checking something real.
        return DataFrame._wrap(self._inner)

    def _operator(self, other: Any, op: str, flip: bool, strict: bool) -> Any:
        """Runs one of the twenty operators, on whichever of three operands it got.

        What the operand is decides which of three calls this makes, which is the
        rule for what belongs in this file rather than in the table. A frame
        aligns on both axes, a series is broadcast along the columns because that
        is the axis an operator has no way to choose, and anything else is a
        constant.

        `strict` is the difference between `==` and `eq`. Two frames that are not
        labelled the same are refused by the operator and aligned by the named
        form, which is pandas' rule and is not arbitrary: a cell only one side has
        has no true or false answer, and only the named form has a `fill_value` to
        say one with.
        """
        from ._frame import DataFrame

        try:
            if isinstance(other, DataFrameMixin):
                if strict:
                    return DataFrame._wrap(self._inner.compare_frame(other._inner, op))
                return DataFrame._wrap(self._inner.binary_frame(other._inner, op, flip, None))
            if isinstance(other, SeriesMixin):
                return DataFrame._wrap(self._inner.binary_series(other._inner, op, 1, flip))
            return DataFrame._wrap(self._inner.binary_value(other, op, flip))
        except Exception as error:
            raise translate(error) from None

    def _named(
        self, other: Any, op: str, axis: Any, level: Any, fill_value: Any, flip: bool
    ) -> Any:
        """Runs one of the twenty named forms, which is the operator plus two arguments.

        `axis` is the one the operators cannot reach. An operator has to pick an
        axis and pandas picks the columns, so `df.add(s, axis=0)` is the only
        spelling of adding a series down the rows there is. Between two frames it
        is read, checked and then ignored, because two frames align on both axes
        whatever it says.

        `fill_value` has three behaviours and all three are pandas'. Between two
        frames it stands in for the side a row or a column is missing from.
        Against a constant it is accepted and ignored, since a constant is never
        the missing side. Against a series it raises.
        """
        from ._frame import DataFrame

        _no_level(level)
        number = _axis_number(axis, "DataFrame", 1, (0, 1))
        try:
            if isinstance(other, DataFrameMixin):
                return DataFrame._wrap(self._inner.binary_frame(other._inner, op, flip, fill_value))
            if isinstance(other, SeriesMixin):
                _no_fill_against_a_series(fill_value)
                return DataFrame._wrap(self._inner.binary_series(other._inner, op, number, flip))
            return DataFrame._wrap(self._inner.binary_value(other, op, flip))
        except Exception as error:
            raise translate(error) from None

    def _unary(self, op: str) -> Any:
        """Runs one of the four unary operations over every column."""
        from ._frame import DataFrame

        try:
            return DataFrame._wrap(self._inner.unary(op))
        except Exception as error:
            raise translate(error) from None

    def _reduce(
        self,
        kind: str,
        param: float,
        axis: Any,
        skipna: bool,
        numeric_only: bool,
        min_count: int,
    ) -> Series:
        """Runs one of the twelve reductions down every column.

        The answer is a series labelled by the column names, which is a
        different shape from the one row frame the core produces, and the turn
        between them is in `PyDataFrame.reduce` where the type the answers have
        to share is picked.
        """
        from ._frame import Series

        _reducing_axis(axis, "DataFrame")
        _held_at(
            "skipna",
            skipna,
            True,
            "a missing value is skipped by every reduction in the library and"
            " there is no second pass that lets one through",
        )
        _held_at(
            "numeric_only",
            numeric_only,
            False,
            "dropping the columns a reduction cannot read is a choice about the"
            " shape of the answer rather than about the reduction",
        )
        _held_at(
            "min_count",
            min_count,
            0,
            "a floor on how many values a sum needs before it answers at all is"
            " a rule about the result rather than about the sum",
        )
        try:
            return Series._wrap(self._inner.reduce(kind, param))
        except Exception as error:
            raise translate(error) from None

    def _quantile(
        self, q: Any, axis: Any, numeric_only: bool, interpolation: str, method: str
    ) -> Series:
        """Runs the quantile down every column."""
        _spelled(
            method,
            ("single", "table"),
            f"Invalid method: {method}. Method must be in {{'table', 'single'}}.",
        )
        _held_at(
            "method",
            method,
            "single",
            "computing one quantile over the whole frame at once rather than"
            " over each column is a different reduction",
        )
        return self._reduce(
            "quantile", _quantile_wanted(q, interpolation), axis, True, numeric_only, 0
        )

    def _nunique(self, axis: Any, dropna: bool) -> Series:
        """Counts the distinct values in every column."""
        _held_at(
            "dropna",
            dropna,
            True,
            "counting a missing value as one more distinct value needs the count"
            " to know it saw one, and the kernel skips them before it counts",
        )
        return self._reduce("nunique", 0.0, axis, True, False, 0)

    def _set_index(
        self, keys: Any, drop: bool, append: bool, inplace: bool, verify_integrity: Any
    ) -> DataFrame:
        """Moves one column into the row labels.

        `keys` is a label, or a list holding one. A list holding two or more is
        a MultiIndex and is refused by saying so rather than by taking the first
        of them, because taking the first would answer a frame that looks right
        and is indexed by half of what was asked for.
        """
        from ._frame import DataFrame

        _held_at("append", append, False, "keeping the old labels as well needs a MultiIndex")
        _held_at(
            "inplace",
            inplace,
            False,
            "every operation here answers a new frame and the Arrow buffers"
            " underneath are shared rather than owned",
        )
        if verify_integrity is not NO_DEFAULT and verify_integrity:
            raise NotImplementedError(
                "verify_integrity=True is not supported yet, because checking"
                " that the new labels are unique is a pass over them that"
                " nothing else here needs"
            )
        wanted = list(keys) if isinstance(keys, (list, tuple)) else [keys]
        if len(wanted) != 1:
            raise NotImplementedError(
                "set_index on more than one column is not supported yet, because"
                " the result is a MultiIndex and there is not one yet"
            )
        try:
            return DataFrame._wrap(self._inner.set_index(str(wanted[0]), bool(drop)))
        except Exception as error:
            raise translate(error) from None

    def _reset_index(
        self,
        level: Any,
        drop: bool,
        inplace: bool,
        col_level: Any,
        col_fill: Any,
        allow_duplicates: Any,
        names: Any,
    ) -> DataFrame:
        """Puts the row labels back to a count from zero.

        With `drop` the old labels are thrown away and without it they become the
        frame's first column, under the index's name or under `index` when it
        does not have one. That is pandas' rule and the name it picks when there
        is none is the string `index` rather than anything cleverer.
        """
        from ._frame import DataFrame

        _no_level(level)
        _refuse("names", names, "naming the columns the old labels land in needs a MultiIndex")
        _held_at(
            "inplace",
            inplace,
            False,
            "every operation here answers a new frame and the Arrow buffers"
            " underneath are shared rather than owned",
        )
        _held_at("col_level", col_level, 0, "there is one level of columns and it is that one")
        _held_at("col_fill", col_fill, "", "there is nothing above the columns to fill")
        if allow_duplicates is not NO_DEFAULT and allow_duplicates:
            raise NotImplementedError(
                "allow_duplicates=True is not supported yet, because two columns"
                " under one name is a shape the schema does not carry"
            )
        try:
            return DataFrame._wrap(self._inner.reset_index(bool(drop)))
        except Exception as error:
            raise translate(error) from None

    def _sort_index(
        self,
        axis: Any,
        level: Any,
        ascending: Any,
        inplace: bool,
        na_position: str,
        sort_remaining: bool,
        ignore_index: bool,
        key: Any,
    ) -> DataFrame:
        """Puts the rows in the order of their labels.

        `kind` is the one argument in the library that is accepted and never
        looked at, and it does not even reach here. The four names it takes are
        numpy's sort algorithms, the sort underneath is stable whichever of them
        is asked for, and a stable order is a correct answer to all four. Every
        other declared argument that is not implemented raises rather than being
        ignored, and this is the exception because there is nothing a caller
        could observe about it.
        """
        from ._frame import DataFrame

        _no_level(level)
        _refuse("key", key, "running a function over the labels before sorting is not written")
        _axis_number(axis, "DataFrame", 0, (0,))
        _held_at(
            "inplace",
            inplace,
            False,
            "every operation here answers a new frame and the Arrow buffers"
            " underneath are shared rather than owned",
        )
        _held_at(
            "na_position",
            na_position,
            "last",
            "where a missing label sits is decided by the sort kernel and it puts them at the end",
        )
        _held_at("sort_remaining", sort_remaining, True, "there is one level to sort")
        _held_at(
            "ignore_index",
            ignore_index,
            False,
            "numbering the rows again after sorting them by their labels throws"
            " away the thing that was just sorted",
        )
        if isinstance(ascending, (list, tuple)):
            raise NotImplementedError(
                "a direction per level is not supported yet, because there is one"
                " level for it to describe"
            )
        try:
            return DataFrame._wrap(self._inner.sort_index(bool(ascending)))
        except Exception as error:
            raise translate(error) from None

    def _take(self, indices: Any, axis: Any, kwargs: dict[str, Any]) -> DataFrame:
        """Gathers rows or columns by position, in the order asked for.

        `**kwargs` is in the signature because it is in pandas', where it
        exists only so that `take` can be called with the arguments numpy's
        `take` has and ignore the ones that do not apply. Passing one here
        raises, because pandas has nothing left that it accepts through it and
        a keyword that is quietly dropped is worse than one that is refused.

        A negative position counts from the end, which the binding does rather
        than this, because the core reads a negative index as a row that was
        not there and the checking has to happen on the side that knows the
        height.
        """
        from ._frame import DataFrame

        if kwargs:
            raise NotImplementedError(
                f"take does not read {sorted(kwargs)}, because pandas accepts them"
                " only to ignore them and a dropped keyword is worse than a refused one"
            )
        wanted = [int(one) for one in indices]
        try:
            if _axis_number(axis, "DataFrame", 0, (0, 1)) == 1:
                names = self._inner.names()
                return DataFrame._wrap(self._inner.select(_named_at(names, wanted)))
            return DataFrame._wrap(self._inner.take(wanted))
        except Exception as error:
            raise translate(error) from None

    def _filter(self, items: Any, like: str | None, regex: str | None, axis: Any) -> DataFrame:
        """Keeps the labels one of three rules names, on either axis.

        The three rules are exclusive and pandas says so with a `TypeError`
        rather than by preferring one, which is right: a call that passed two
        of them meant something the signature cannot express and the caller has
        to say which.

        `items` keeps the order it was written in and drops what is not there,
        which is the one rule of the three that is not a filter of the frame.
        The other two keep the frame's own order, because a substring and a
        pattern describe a set and not a sequence.
        """
        from ._frame import DataFrame

        rules = [
            name
            for name, value in (("items", items), ("like", like), ("regex", regex))
            if value is not None
        ]
        if len(rules) > 1:
            raise TypeError("Keyword arguments `items`, `like`, or `regex` are mutually exclusive")
        if not rules:
            raise TypeError("Must pass either `items`, `like`, or `regex`")
        over_rows = _axis_number(axis, "DataFrame", 1, (0, 1)) == 0
        labels = [str(one) for one in self.index] if over_rows else self._inner.names()
        if items is not None:
            wanted = [str(one) for one in items]
            held = set(labels)
            kept = [one for one in wanted if one in held]
        elif like is not None:
            kept = [one for one in labels if str(like) in one]
        else:
            pattern = re.compile(regex if isinstance(regex, str) else str(regex))
            kept = [one for one in labels if pattern.search(one) is not None]
        try:
            if not over_rows:
                return DataFrame._wrap(self._inner.select(kept))
            # The labels were rendered to compare them, so they cannot be looked
            # up again as labels. The positions come from walking the rendered
            # list, which also gives every row of a repeated label rather than
            # the first, and that is what pandas answers here.
            wherever = {one: i for i, one in enumerate(kept)}
            found = [i for i, one in enumerate(labels) if one in wherever]
            if items is not None:
                found.sort(key=lambda i: (wherever[labels[i]], i))
            return DataFrame._wrap(self._inner.take(found))
        except Exception as error:
            raise translate(error) from None

    def _select_dtypes(self, include: Any, exclude: Any) -> DataFrame:
        """Keeps the columns whose type is in one set of types and not in another.

        The vocabulary is numpy's type tree, which is why `number` takes a
        column of spans as well as the integers and the floats: numpy makes
        `timedelta64` a kind of signed integer, pandas inherits that, and a
        compatibility layer that tidied it up would be answering a different
        question from the one the caller's pandas answers. The tree is written
        out in `_DTYPE_FAMILIES` rather than computed, because firepanda does
        not import numpy and there is nothing here to ask.
        """
        from ._frame import DataFrame

        wanted = _dtype_words(include)
        unwanted = _dtype_words(exclude)
        if not wanted and not unwanted:
            raise InvalidArgumentError("at least one of include or exclude must be nonempty")
        shared = wanted & unwanted
        if shared:
            raise InvalidArgumentError(f"include and exclude overlap on {sorted(shared)}")
        kept_in = _dtype_branches(wanted)
        held_out = _dtype_branches(unwanted)
        names = self._inner.names()
        try:
            families = [_dtype_family(one) for one in self._inner.dtypes()]
        except Exception as error:
            raise translate(error) from None
        kept = [
            name
            for name, family in zip(names, families, strict=True)
            if (not kept_in or family & kept_in) and not family & held_out
        ]
        try:
            return DataFrame._wrap(self._inner.select(kept))
        except Exception as error:
            raise translate(error) from None

    def _truncate(self, before: Any, after: Any, axis: Any, copy: Any) -> DataFrame:
        """Keeps everything between two labels, with both of them kept.

        This is `loc[before:after]` with two rules on top. The labels have to be
        in order, which pandas checks by asking whether the index is sorted and
        refusing outright when it is not, because a truncation of an unsorted
        index would answer the rows that happen to lie between two positions
        and a caller who wrote two labels meant the values between them. And
        the pair has to be the right way round, which is checked here rather
        than left to answer nothing, since an empty frame is a plausible answer
        to a correct call and a useless one to a reversed pair.
        """
        from ._frame import DataFrame

        _held_at(
            "copy",
            copy,
            NO_DEFAULT,
            "an answer is always a new frame over buffers that are shared rather"
            " than owned, so there is no copy to ask for",
        )
        over_columns = _axis_number(axis, "DataFrame", 0, (0, 1)) == 1
        names = self._inner.names()
        if over_columns:
            rising = names == sorted(names)
            falling = names == sorted(names, reverse=True)
        else:
            rising = self.index.is_monotonic_increasing
            falling = self.index.is_monotonic_decreasing
        if not rising and not falling:
            raise InvalidArgumentError("truncate requires a sorted index")
        if before is not None and after is not None and before > after:
            raise InvalidArgumentError(f"Truncate: {after} must be after {before}")
        # A falling axis reads the pair the other way round, because the label
        # nearer the top of the frame is the larger one. The check above happens
        # first either way, which is pandas' order and means that on a falling
        # index the pair is still written smaller first even though the rows come
        # back in the other direction.
        if falling and not rising:
            before, after = after, before
        try:
            if over_columns:
                first = 0 if before is None else names.index(str(before))
                last = len(names) if after is None else names.index(str(after)) + 1
                return DataFrame._wrap(self._inner.select(names[first:last]))
            walked = self.index.slice_indexer(before, after)
            start, stop, _ = walked.indices(self._inner.length())
            return DataFrame._wrap(self._inner.slice_rows(start, max(start, stop)))
        except Exception as error:
            raise translate(error) from None

    def _duplicate_subset(self, subset: Any) -> list[str]:
        """Works out which columns decide whether two rows are the same.

        `None` means every column, and it is resolved here rather than sent
        across as an absence to be filled in on the other side, because which
        columns a frame has is a question this side can already ask and a
        default that lives in two places is a default that will disagree with
        itself.

        A bare name rather than a list of names is one column, which is pandas'
        rule and comes from `is_list_like` answering False for a string. It is
        worth having, because `subset="key"` is what a caller writes first and
        iterating the string would ask for a column per letter.

        A name written twice is dropped to one. The core refuses a repeat, on
        the grounds that a caller who wrote it meant something else, and that is
        the right answer for a Mojo caller writing the subset out by hand. It is
        the wrong answer here, because pandas accepts the repeat and answers as
        if it were written once, and this layer exists to answer what pandas
        answers. Nothing is lost either way: a key column compared against
        itself twice tells the same rows apart as a key column compared once.

        Args:
            subset: What the caller wrote.

        Returns:
            Column names, in the order given, with no repeats.
        """
        if subset is None:
            return self._inner.names()
        if isinstance(subset, str) or not hasattr(subset, "__iter__"):
            written = [subset]
        else:
            written = list(subset)
        seen: dict[str, None] = {}
        for one in written:
            seen[str(one)] = None
        return list(seen)

    def _duplicated(self, subset: Any, keep: Any) -> Series:
        """Marks the rows that repeat a key another row already carries.

        A frame with no columns has no rows to tell apart and the core says so
        rather than answering. pandas answers an empty mask, which is the same
        statement made quietly, and that is what comes back here: the alternative
        is an error raised on a frame where nothing went wrong.
        """
        from ._frame import Series

        word = _keep_word(keep)
        names = self._duplicate_subset(subset)
        if not names:
            return Series([], dtype="bool")
        try:
            return Series._wrap(self._inner.duplicated(names, word))
        except Exception as error:
            raise translate(error) from None

    def _drop_duplicates(
        self, subset: Any, keep: Any, inplace: bool, ignore_index: bool
    ) -> DataFrame:
        """Removes the rows that repeat a key another row already carries.

        `ignore_index` is honoured rather than refused, because the labels of the
        rows that survived are the one thing a drop leaves behind that a caller
        may not want: they are the positions the rows held in the frame before
        the drop, so they have gaps in them wherever a row went. Numbering them
        again is `reset_index` and there is nothing to write.
        """
        from ._frame import DataFrame

        _held_at(
            "inplace",
            inplace,
            False,
            "every operation here answers a new frame and the Arrow buffers"
            " underneath are shared rather than owned",
        )
        word = _keep_word(keep)
        names = self._duplicate_subset(subset)
        try:
            kept = self._inner if not names else self._inner.drop_duplicates(names, word)
            if ignore_index:
                kept = kept.reset_index(True)
            return DataFrame._wrap(kept)
        except Exception as error:
            raise translate(error) from None

    def _top_rows(self, n: Any, columns: Any, keep: Any, largest: bool) -> DataFrame:
        """The `n` rows holding the best values in one column.

        Four of pandas' rules are settled here rather than under the boundary,
        and each of them is a rule about what a caller is allowed to write
        rather than about which rows come back.

        A count that is not a whole number is refused by asking Python to make
        an index out of it, which is what pandas does and which gives the same
        sentence back for free. A negative count is not refused, because
        pandas answers an empty frame for it and a count of zero and a count of
        minus one are the same request.

        A bare name is one column and a list is a list of them, and a name
        written twice is read once, all as `duplicated` does it. An empty list
        is an empty frame with the columns kept, which is pandas' answer and
        is not obviously right but is not ours to change. More than one name
        is a refusal: pandas ranks by the first column and breaks its ties with
        the second, and the kernel here holds one value per slot and has no
        second value to break anything with.

        `keep="all"` is the other refusal. It answers more than `n` rows when
        the `n`th value is tied, and a fixed table of `n` slots per group
        cannot hold an answer whose height depends on the data in it. It needs
        a second pass that finds the cut value and then takes every row equal
        to it, which is a different piece of work rather than a flag.

        Args:
            n: How many rows to keep.
            columns: The column to rank by, or a one item list holding it.
            keep: Which row of a tie survives.
            largest: True for the top of the column, False for the bottom.

        Returns:
            A new frame of the kept rows, best first.
        """
        from ._frame import DataFrame

        wanted = operator.index(n)
        if keep not in ("first", "last", "all"):
            raise InvalidArgumentError('keep must be either "first", "last" or "all"')
        who = "nlargest" if largest else "nsmallest"
        if keep == "all":
            raise UnsupportedError(
                f"{who} with keep='all' answers more than n rows when the last"
                " value is tied, which the kernel underneath cannot express"
            )

        if isinstance(columns, str) or not hasattr(columns, "__iter__"):
            written = [columns]
        else:
            written = list(columns)
        names = list(dict.fromkeys(str(one) for one in written))

        try:
            if not names:
                return DataFrame._wrap(self._inner.slice_rows(0, 0))
            if len(names) > 1:
                raise UnsupportedError(
                    f"{who} ranks by one column here, and {len(names)} were"
                    " given, because breaking a tie with a second column is"
                    " work the kernel underneath does not do"
                )
            return DataFrame._wrap(self._inner.top_rows(names[0], wanted, largest, keep))
        except Exception as error:
            raise translate(error) from None

    def _reindex(
        self,
        labels: Any,
        index: Any,
        columns: Any,
        axis: Any,
        method: Any,
        copy: Any,
        level: Any,
        fill_value: Any,
        limit: Any,
        tolerance: Any,
    ) -> DataFrame:
        """The frame on a set of row labels or column names, or both.

        Ten parameters and two of them do the work. The rest are here because
        pandas has them and a caller who passes one should get an answer about
        that parameter rather than an answer about a parameter it does not have,
        which is why they are named in the signature and answered one at a time
        below rather than swept into a `**kwargs` nobody reads.

        `labels` is the axis `axis` names when it is the only one given, and it
        is the axis nobody named when one of `index=` and `columns=` is. That
        second rule is pandas' and it is not the rule a reader expects:
        `df.reindex([1], index=[2])` does not complain about being told twice
        and does not take the `index=` over the labels, it reads the labels as
        the columns, because the columns are the axis left over. Naming both of
        them and passing labels as well is the error, and so is naming an axis
        twice by writing `axis=` next to `index=`.

        `method` is the refusal. It fills a row the frame does not have from the
        row beside it, which is a different operation from putting a value in it
        and wants the labels sorted to mean anything. `limit` and `tolerance`
        belong to `method`, so passing either without it is the error pandas
        gives, word for word, rather than a refusal of our own.

        `copy` and `level` are accepted and ignored, which is also what pandas
        does. `copy` is deprecated there and everything here is immutable
        anyway, and `level` selects one level of a MultiIndex, of which a flat
        index has exactly one.

        A `fill_value` of NaN is read as no fill value at all. pandas' own
        default for the parameter is NaN, so a caller writing it out has asked
        for the rows to be missing, which is what happens when nothing is
        passed.

        Args:
            labels: The labels for whichever axis `axis` names.
            index: The row labels.
            columns: The column names.
            axis: Which axis `labels` is for. The rows by default.
            method: Refused.
            copy: Ignored.
            level: Ignored.
            fill_value: What to put in a row or column the frame does not have.
            limit: Refused, since it only means something with `method`.
            tolerance: Refused, for the same reason.

        Returns:
            A new frame on the labels asked for.
        """
        from ._frame import DataFrame

        _reindex_filling("frame", method, limit, tolerance)

        if index is not None or columns is not None:
            if axis is not None:
                raise TypeError("Cannot specify both 'axis' and any of 'index' or 'columns'")
            if labels is not None:
                if index is not None and columns is not None:
                    raise TypeError("Cannot specify all of 'labels', 'index', 'columns'.")
                if index is None:
                    index = labels
                else:
                    columns = labels
        elif labels is not None:
            if _axis_number(axis, "DataFrame", 0, (0, 1)) == 0:
                index = labels
            else:
                columns = labels

        value = None if isinstance(fill_value, float) and math.isnan(fill_value) else fill_value

        inner = self._inner
        try:
            if columns is not None:
                if isinstance(columns, str) or not hasattr(columns, "__iter__"):
                    raise DTypeError(
                        "Index(...) must be called with a collection of some"
                        f" kind, {columns!r} was passed"
                    )
                inner = inner.reindex_columns([str(one) for one in columns], value)
            if index is not None:
                inner = inner.reindex(index, value)
            return DataFrame._wrap(inner)
        except Exception as error:
            raise translate(error) from None

    def _reindex_like(
        self, other: Any, method: Any, copy: Any, limit: Any, tolerance: Any
    ) -> DataFrame:
        """The frame shaped the way another frame is shaped.

        Both axes of `reindex` with the labels read off `other` rather than
        written out, and the reason it is a method of its own rather than two
        keyword arguments is the third thing it carries: the name of the other
        frame's index. A caller who took the labels out and passed them as a
        sequence would get the answer back under this frame's own index name,
        which is not what asking for another frame's shape means.

        `other` has to be a frame. pandas refuses a series here with a sentence
        about there being no axis named columns on one, which is true and is
        copied rather than improved for the reason document 22 gives.

        There is no `fill_value`, because pandas does not offer one: a row or a
        column the other frame has and this one does not comes back missing, so
        an integer column that gains a row widens the way it does everywhere
        else.

        Args:
            other: The frame whose labels and column names to take.
            method: Refused, as it is on `reindex`.
            copy: Ignored, as it is on `reindex`.
            limit: Refused, since it only means something with `method`.
            tolerance: Refused, for the same reason.

        Returns:
            A new frame with the other frame's labels and columns.
        """
        from ._frame import DataFrame

        _reindex_filling("frame", method, limit, tolerance)
        if not isinstance(other, DataFrameMixin):
            raise InvalidArgumentError(
                f"No axis named columns for object type {type(other).__name__}"
            )
        try:
            return DataFrame._wrap(self._inner.reindex_like(other._inner))
        except Exception as error:
            raise translate(error) from None

    def _transform(
        self, kind: str, periods: int, axis: Any, inplace: bool, ignore_index: bool
    ) -> DataFrame:
        """Runs one named transformation down every column.

        Eleven of the twelve come through here. `dropna` on a frame does not,
        because it removes rows rather than transforming columns, and it has its
        own method below.
        """
        from ._frame import DataFrame

        _transforming_axis(axis, "DataFrame")
        _held_at(
            "inplace",
            inplace,
            False,
            "every operation here answers a new frame and the Arrow buffers"
            " underneath are shared rather than owned, so writing into one would"
            " change frames the caller never mentioned",
        )
        _held_at(
            "ignore_index",
            ignore_index,
            False,
            "throwing the labels away and numbering the rows again is a change to"
            " the index rather than to the values",
        )
        try:
            return DataFrame._wrap(self._inner.transform(kind, periods))
        except Exception as error:
            raise translate(error) from None

    def _fill(self, kind: str, axis: Any, inplace: bool, limit: Any, limit_area: Any) -> DataFrame:
        """Fills each column's missing values from its neighbours."""
        _refuse(
            "limit_area",
            limit_area,
            "filling only the gaps between two present values, or only the ones"
            " outside them, needs the fill to know where the ends are and it"
            " walks the column without looking",
        )
        return self._transform(kind, _limit_wanted(limit), axis, inplace, False)

    def _shift(self, periods: Any, freq: Any, axis: Any, fill_value: Any, suffix: Any) -> DataFrame:
        """Moves every column's rows along, leaving the gap missing."""
        _refuse(
            "freq",
            freq,
            "shifting by a frequency moves the labels rather than the values and"
            " needs the offset vocabulary, which is the resampling milestone",
        )
        _refuse("suffix", suffix, "it only names the columns a list of periods produces")
        _held_at(
            "fill_value",
            fill_value,
            NO_DEFAULT,
            "filling the gap keeps a column of whole numbers whole, and the value"
            " has to reach the kernel as a typed one rather than as a Python"
            " object",
        )
        if not isinstance(periods, int) or isinstance(periods, bool):
            raise NotImplementedError(
                "periods has to be a single number for now, because a list of them"
                " answers a frame with one set of columns per period"
            )
        return self._transform("shift", periods, axis, False, False)

    def _pct_change(self, periods: int, fill_method: Any, freq: Any) -> DataFrame:
        """The fractional change between each row and the one before it."""
        _refuse("fill_method", fill_method, "pandas removed it in 3.0 and only accepts None")
        _refuse("freq", freq, "it needs the offset vocabulary, which is the resampling milestone")
        return self._transform("pct_change", periods, 0, False, False)

    def _scan(self, kind: str, axis: Any, skipna: bool, numeric_only: bool) -> DataFrame:
        """Runs one of the four scans down every column."""
        _held_at(
            "skipna",
            skipna,
            True,
            "a missing row is stepped over and put back where it was, and letting"
            " one through would poison every row after it",
        )
        _held_at(
            "numeric_only",
            numeric_only,
            False,
            "dropping the columns a scan cannot read is a choice about the shape"
            " of the answer rather than about the scan",
        )
        return self._transform(kind, 0, axis, False, False)

    def _dropna(
        self,
        axis: Any,
        how: Any,
        thresh: Any,
        subset: Any,
        inplace: bool,
        ignore_index: bool,
    ) -> DataFrame:
        """Removes the rows that have a missing value in them.

        The one name on the transformation list that means something different
        to a frame than to a column, so it does not go through `_transform`.
        """
        from ._frame import DataFrame

        _transforming_axis(axis, "DataFrame")
        _held_at(
            "how",
            how,
            NO_DEFAULT,
            "dropping a row only when every column is missing is the other rule and"
            " the kernel implements the one where any column disqualifies it",
        )
        _held_at(
            "thresh",
            thresh,
            NO_DEFAULT,
            "keeping a row that has at least so many values counts per row, and"
            " the mask says present or absent rather than how many",
        )
        _held_at(
            "inplace",
            inplace,
            False,
            "the answer is a new frame over shared Arrow buffers, so writing into"
            " one would change frames the caller never mentioned",
        )
        _held_at(
            "ignore_index",
            ignore_index,
            False,
            "numbering the surviving rows again is a change to the index rather"
            " than to which rows survive",
        )
        names: list[str] = []
        if subset is not None:
            names = [subset] if isinstance(subset, str) else [str(one) for one in subset]
        try:
            return DataFrame._wrap(self._inner.dropna(names))
        except Exception as error:
            raise translate(error) from None

    def _astype(self, dtype: Any, copy: Any, errors: Any) -> DataFrame:
        """Converts some or all of the columns and hands back a new frame.

        Two shapes arrive. One type converts every column, and a dict naming
        some of them converts those and leaves the rest alone. The dict is the
        interesting one, because a key that is not a column is a mistake worth
        catching before anything converts, so the keys are all checked first and
        the frame is either converted whole or not touched.
        """
        from ._frame import DataFrame

        strictly = _cast_keywords(copy, errors)
        if isinstance(dtype, dict):
            present = set(self._inner.names())
            for one in dtype:
                if one not in present:
                    raise ColumnNotFoundError(
                        "Only a column name can be used for the key in a dtype"
                        f" mappings argument. '{one}' not found in columns."
                    )
            names = [str(one) for one in dtype]
            dtypes = [_named_dtype(one) for one in dtype.values()]
        else:
            names = self._inner.names()
            dtypes = [_named_dtype(dtype)] * len(names)
        try:
            return DataFrame._wrap(self._inner.cast(names, dtypes, True))
        except Exception as error:
            if strictly:
                raise translate(error) from None
            return DataFrame._wrap(self._inner)


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
        """Builds a series from a sequence of values, or copies another one.

        Same shape as the frame constructor and refusing the same way, with the
        one difference that `name` is honoured, since a series carries its name
        and there is nothing to implement.

        A series arriving as the data is unwrapped and handed across as the
        extension object it holds, so the extension can copy the column instead
        of iterating it. Going out through a Python list and back would infer
        the type again off the values, which loses a column of instants
        entirely and is slow for the columns it does not lose. It has to be
        unwrapped here because what a user holds is the generated wrapper and
        the extension can only recognise its own type.

        `dtype=` is inference followed by a cast, for the reason the frame
        constructor gives.
        """
        _refuse("index", index, "putting labels on a series as it is built is not written")
        _refuse("copy", copy, "there is exactly one behaviour and it always copies")
        source = data._inner if isinstance(data, SeriesMixin) else data
        try:
            self._inner = _firepanda.Series(source, "" if name is None else str(name))
        except Exception as error:
            raise translate(error) from None
        if dtype is not None:
            wanted = _named_dtype(dtype)
            try:
                self._inner = self._inner.cast(wanted, True)
            except Exception as error:
                raise translate(error) from None

    def __getitem__(self, key: Any) -> Any:
        """Reads by label, except for a slice of numbers, which is by position.

        The exception is the whole difficulty of this method and it is pandas'
        rather than an invention. `s[2]` is the label two even on an index whose
        labels are strings, where it raises, and `s[2:5]` is the rows two to
        five counting from the front even on an index whose labels are the
        numbers in another order. Nobody would design that, and code that
        relies on it is everywhere, so a library that reads the slice by label
        is not the library people have.

        What decides is the slice's own bounds rather than the index's type. A
        bound that is a whole number means positions and anything else means
        labels, which is how `s["a":"c"]` on a string index stays a closed
        slice of labels while `s[0:2]` on the same index is the first two rows.

        Everything that is not a slice goes through `loc`, which is exactly
        what pandas does with it.
        """
        from ._frame import Series

        if isinstance(key, slice) and _counts_rather_than_names(key):
            inner = self._inner
            where = _by_position(key, inner.length())
            try:
                return Series._wrap(_narrowed(inner, where))
            except Exception as error:
                raise translate(error) from None
        return _Along(self, True)[key]

    def _get(self, key: Any, default: Any) -> Any:
        """The value at a label, or a value of the caller's choosing.

        The series' half of the frame's method and the same one line. Square
        brackets on a series read a label, so a key that is not a label of this
        series is what brings the default back, and a slice of numbers is read
        as positions here for the same reason it is read that way there.
        """
        try:
            return self[key]
        except (IndexError, KeyError, TypeError, ValueError):
            return default

    def _squeeze(self, axis: Any) -> Any:
        """The series' one value when it has one row, and otherwise the series.

        A series has one axis, so there is one length that can be one and the
        parameter can only name the axis the series already has. Naming the
        other one is refused in pandas' words, which is worth having because a
        frame and a series are passed to the same function often enough that
        `axis=1` reaching this is a real mistake rather than a made up one.
        """
        from ._frame import Series

        _axis_number(axis, "Series", 0, (0,))
        if self._inner.length() != 1:
            return Series._wrap(self._inner)
        try:
            return self._inner.cell(0)
        except Exception as error:
            raise translate(error) from None

    def _operator(self, other: Any, op: str, flip: bool, strict: bool) -> Any:
        """Runs one of the twenty operators, on whichever of three operands it got.

        A frame on the right is handed back rather than handled. `s + df` is a
        frame in pandas and the frame is the side that knows how to build one, so
        returning `NotImplemented` is what makes Python turn the expression round
        and ask `DataFrame.__radd__` instead. Refusing here would break an
        expression that has a perfectly good answer.

        `strict` is the difference between `==` and `eq`, for the reason
        `DataFrameMixin._operator` gives.
        """
        from ._frame import Series

        if isinstance(other, DataFrameMixin):
            return NotImplemented
        try:
            if isinstance(other, SeriesMixin):
                if strict:
                    return Series._wrap(self._inner.compare_series(other._inner, op))
                return Series._wrap(self._inner.binary_series(other._inner, op, flip, None))
            return Series._wrap(self._inner.binary_value(other, op, flip))
        except Exception as error:
            raise translate(error) from None

    def _named(
        self, other: Any, op: str, axis: Any, level: Any, fill_value: Any, flip: bool
    ) -> Any:
        """Runs one of the twenty two named forms, which is the operator plus two arguments.

        A series has one axis, so `axis` is read and checked and can only be the
        rows. It is declared because pandas declares it and the signature parity
        test compares the whole parameter list.

        `fill_value` is honoured between two series and ignored against a
        constant, which are two of the three behaviours the frame has. The third
        does not arise, because a series has nothing to broadcast against.
        """
        from ._frame import Series

        _no_level(level)
        _axis_number(axis, "Series", 0, (0,))
        try:
            if isinstance(other, SeriesMixin):
                return Series._wrap(self._inner.binary_series(other._inner, op, flip, fill_value))
            return Series._wrap(self._inner.binary_value(other, op, flip))
        except Exception as error:
            raise translate(error) from None

    def _divmod(self, other: Any, axis: Any, level: Any, fill_value: Any, flip: bool) -> Any:
        """The floor division and the remainder, as the pair Python asks for.

        Two passes rather than one, here and in pandas both, because the pair
        comes out of two operations run separately and a single pass would need a
        kernel that writes two columns, which nothing in `firepanda/kernel` does.
        """
        return (
            self._named(other, "floordiv", axis, level, fill_value, flip),
            self._named(other, "mod", axis, level, fill_value, flip),
        )

    def _unary(self, op: str) -> Any:
        """Runs one of the four unary operations over every row."""
        from ._frame import Series

        try:
            return Series._wrap(self._inner.unary(op))
        except Exception as error:
            raise translate(error) from None

    def _reduce(
        self,
        kind: str,
        param: float,
        axis: Any,
        skipna: bool,
        numeric_only: bool,
        min_count: int,
    ) -> Any:
        """Runs one of the twelve reductions over the whole column.

        The answer is an ordinary Python number rather than a one row series,
        because that is what pandas hands back and because a caller who wrote
        `s.sum() > 10` is holding it in a Python expression a moment later.

        A reduction with no answer comes back from the boundary as `None` and
        leaves here as a float NaN, because that is what pandas gives for the
        mean of nothing and a caller comparing against it will be using `isnan`
        rather than `is None`.
        """
        _reducing_axis(axis, "Series")
        _held_at(
            "skipna",
            skipna,
            True,
            "a missing value is skipped by every reduction in the library and"
            " there is no second pass that lets one through",
        )
        _held_at(
            "numeric_only",
            numeric_only,
            False,
            "refusing a column a reduction cannot read is what the reduction"
            " already does, and it says so with the dtype in the message",
        )
        _held_at(
            "min_count",
            min_count,
            0,
            "a floor on how many values a sum needs before it answers at all is"
            " a rule about the result rather than about the sum",
        )
        try:
            answer = self._inner.reduce(kind, param)
        except Exception as error:
            raise translate(error) from None
        return float("nan") if answer is None else answer

    def _quantile(self, q: Any, interpolation: str) -> Any:
        """Runs the quantile over the whole column."""
        return self._reduce("quantile", _quantile_wanted(q, interpolation), 0, True, False, 0)

    def _nunique(self, axis: Any, dropna: bool) -> Any:
        """Counts the distinct values in the column."""
        _held_at(
            "dropna",
            dropna,
            True,
            "counting a missing value as one more distinct value needs the count"
            " to know it saw one, and the kernel skips them before it counts",
        )
        return self._reduce("nunique", 0.0, axis, True, False, 0)

    def _transform(
        self, kind: str, periods: int, axis: Any, inplace: bool, ignore_index: bool
    ) -> Series:
        """Runs one named transformation over the whole column.

        All twelve come through here, including `dropna`, because a column
        `dropna` removes values and is a transformation like the rest. The frame
        one removes rows and is not.
        """
        from ._frame import Series

        _transforming_axis(axis, "Series")
        _held_at(
            "inplace",
            inplace,
            False,
            "the answer is a new series over shared Arrow buffers, so writing into"
            " one would change columns the caller never mentioned",
        )
        _held_at(
            "ignore_index",
            ignore_index,
            False,
            "throwing the labels away and numbering the rows again is a change to"
            " the index rather than to the values",
        )
        try:
            return Series._wrap(self._inner.transform(kind, periods))
        except Exception as error:
            raise translate(error) from None

    def _fill(self, kind: str, axis: Any, inplace: bool, limit: Any, limit_area: Any) -> Series:
        """Fills the column's missing values from its neighbours."""
        _refuse(
            "limit_area",
            limit_area,
            "filling only the gaps between two present values, or only the ones"
            " outside them, needs the fill to know where the ends are and it"
            " walks the column without looking",
        )
        return self._transform(kind, _limit_wanted(limit), axis, inplace, False)

    def _shift(self, periods: Any, freq: Any, axis: Any, fill_value: Any, suffix: Any) -> Series:
        """Moves the column's rows along, leaving the gap missing."""
        _refuse(
            "freq",
            freq,
            "shifting by a frequency moves the labels rather than the values and"
            " needs the offset vocabulary, which is the resampling milestone",
        )
        _refuse("suffix", suffix, "it only names the columns a list of periods produces")
        _held_at(
            "fill_value",
            fill_value,
            NO_DEFAULT,
            "filling the gap keeps a column of whole numbers whole, and the value"
            " has to reach the kernel as a typed one rather than as a Python"
            " object",
        )
        if not isinstance(periods, int) or isinstance(periods, bool):
            raise NotImplementedError(
                "periods has to be a single number for now, because a list of them"
                " answers a frame with one column per period"
            )
        return self._transform("shift", periods, axis, False, False)

    def _pct_change(self, periods: int, fill_method: Any, freq: Any) -> Series:
        """The fractional change between each row and the one before it."""
        _refuse("fill_method", fill_method, "pandas removed it in 3.0 and only accepts None")
        _refuse("freq", freq, "it needs the offset vocabulary, which is the resampling milestone")
        return self._transform("pct_change", periods, 0, False, False)

    def _scan(self, kind: str, axis: Any, skipna: bool, numeric_only: bool) -> Series:
        """Runs one of the four scans over the whole column."""
        _held_at(
            "skipna",
            skipna,
            True,
            "a missing row is stepped over and put back where it was, and letting"
            " one through would poison every row after it",
        )
        return self._transform(kind, 0, axis, False, False)

    def _astype(self, dtype: Any, copy: Any, errors: Any) -> Series:
        """Converts the column and hands back a new one.

        The name is resolved outside the `try`, because `errors="ignore"` means
        a value that will not convert, not a type that does not exist. pandas
        raises on the name whichever way `errors` is set, and so does this.
        """
        from ._frame import Series

        strictly = _cast_keywords(copy, errors)
        wanted = _named_dtype(dtype)
        try:
            return Series._wrap(self._inner.cast(wanted, True))
        except Exception as error:
            if strictly:
                raise translate(error) from None
            return Series._wrap(self._inner)

    def _reindex(
        self,
        index: Any,
        axis: Any,
        method: Any,
        copy: Any,
        level: Any,
        fill_value: Any,
        limit: Any,
        tolerance: Any,
    ) -> Series:
        """The series on a set of labels, whether it has them or not.

        The row half of the frame's method with one column under it, and the
        same four answers to the parameters that do no work. `method` is
        refused, `limit` and `tolerance` give pandas' own sentence when they
        arrive without it, and `copy` and `level` are taken and ignored.

        `axis` is the one that differs from the frame, and it is taken and
        ignored too. A series has one axis, so naming it is not a choice, and
        pandas accepts any value here rather than checking it, including values
        there is no axis for.

        A `fill_value` of NaN is read as no fill value at all, which is the same
        rule the frame uses. pandas' default for the parameter here is `None`
        rather than NaN, and both of them mean leave the row missing.

        Args:
            index: The labels the result should have, in order.
            axis: Ignored.
            method: Refused.
            copy: Ignored.
            level: Ignored.
            fill_value: What to put in a row whose label was not found.
            limit: Refused, since it only means something with `method`.
            tolerance: Refused, for the same reason.

        Returns:
            A new series on the labels asked for.
        """
        from ._frame import Series

        _reindex_filling("series", method, limit, tolerance)
        if index is None:
            return Series._wrap(self._inner)

        value = None if isinstance(fill_value, float) and math.isnan(fill_value) else fill_value
        try:
            return Series._wrap(self._inner.reindex(index, value))
        except Exception as error:
            raise translate(error) from None

    def _reindex_like(
        self, other: Any, method: Any, copy: Any, limit: Any, tolerance: Any
    ) -> Series:
        """The series labelled the way another thing is labelled.

        `reindex` with the labels read off whatever was handed over, and the
        name of its index coming across with them, which is the difference
        between this and taking the labels out and passing them as a list.

        `other` can be a frame or a series, since pandas asks it for its index
        and both have one. Unlike the frame's version there is no second axis to
        disagree about, so a frame is accepted here where a series is refused
        there.

        Args:
            other: The frame or series whose labels to take.
            method: Refused, as it is on `reindex`.
            copy: Ignored, as it is on `reindex`.
            limit: Refused, since it only means something with `method`.
            tolerance: Refused, for the same reason.

        Returns:
            A new series on the other thing's labels.
        """
        from ._frame import Series

        _reindex_filling("series", method, limit, tolerance)
        if not isinstance(other, (DataFrameMixin, SeriesMixin)):
            raise DTypeError(
                "reindex_like takes a frame or a series, since it reads the"
                f" labels off one, and {type(other).__name__} has none"
            )
        shape: Any = other
        try:
            return Series._wrap(self._inner.reindex_like(shape.index._inner))
        except Exception as error:
            raise translate(error) from None


class Namespace:
    """One accessor, reached through an attribute the way pandas reaches one.

    `s.dt` builds a `DatetimeProperties` around the series and `Series.dt` is the
    class itself. That second half is the reason this is a descriptor rather than
    a property. A property answers itself when it is read off the class, so
    `Series.dt` would be a `property` object, and a program that reads the
    accessor's members off the class rather than off an instance would find
    nothing. pandas answers the accessor class there, and the conformance board
    reads the class, so this answers the accessor class too.

    Nothing is cached on the instance. pandas does cache, and a firepanda wrapper
    has `__slots__` and therefore nowhere to cache into, so the object is built
    per lookup. It holds one reference and building it is cheaper than the
    dictionary lookup finding a cached one would need. `s.dt is s.dt` is False in
    pandas as well, for a different reason, so nothing observable moves.
    """

    __slots__ = ("_accessor",)
    """The class to build, which is the whole of the state."""

    def __init__(self, accessor: type) -> None:
        """Holds the accessor class.

        Args:
            accessor: The class to build around a series.
        """
        self._accessor = accessor

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        """Answers the accessor class off the class and an accessor off an instance."""
        if obj is None:
            return self._accessor
        return self._accessor(obj)


class DatetimeMixin:
    """The hand written half of `DatetimeProperties`.

    The one mixin whose state is not an extension object. An accessor holds the
    series it was reached from, which is already a wrapper, and the reason it is
    the wrapper rather than the wrapper's `_inner` is that everything the
    accessor hands back is a `Series` and the wrapping has to happen somewhere.
    Holding the wrapper means the accessor never builds one from nothing.

    Nothing here decides what a part means. Every helper turns whatever pandas
    lets a caller write into the word and the one string the boundary takes, and
    the word is read in `firepanda/py/temporal.mojo`.
    """

    __slots__ = ("_series",)
    """The series the accessor was reached from. Slotted for the reason
    `DataFrameMixin` gives, and there is no `_inner` because there is no
    extension object for an accessor to hold."""

    _series: Series

    def __init__(self, data: Series) -> None:
        """Holds the series. Not a public entry point.

        The parameter is called `data` because pandas calls it `data`, and the
        signature board compares the two. It is the only hand written signature
        in this file that a conformance case reads, since every other member here
        is reached through a generated one.
        """
        self._series = data

    def _part(self, kind: str, arg: str) -> Series:
        """Reads one part of the column, and hands back a column.

        Thirty of the names come through here, and the string is the frequency,
        the unit, the format or the zone for the seven that take one and empty
        for the rest.
        """
        from ._frame import Series

        try:
            return Series._wrap(self._series._inner.temporal_part(kind, arg))
        except Exception as error:
            raise translate(error) from None

    def _zone(self) -> str | None:
        """Reads the clock the column is read against.

        None in pandas when the column carries no zone, and the boundary answers
        an empty string, because a Mojo `String` has no absent value and
        inventing one for one caller would be worse than turning it back here.
        """
        try:
            found = self._series._inner.temporal_word("tz")
        except Exception as error:
            raise translate(error) from None
        return found or None

    def _resolution(self) -> str:
        """Reads how many of the column's integers make a second.

        A separate helper from `_zone` rather than one that takes the word,
        which is the only place the accessor splits a door two ways. The reason
        is the return type: a zone can be absent and a resolution cannot, and one
        helper answering both would have to be typed as though either could be.
        """
        try:
            return self._series._inner.temporal_word("unit")
        except Exception as error:
            raise translate(error) from None

    def _rounded(self, kind: str, freq: Any, ambiguous: Any, nonexistent: Any) -> Series:
        """Moves every clock to a frequency, one of three ways.

        The two zone policies are read here only when the column already carries
        a zone, which is measured rather than reasoned about: pandas hands a naive
        column back rounded with a misspelled `nonexistent` in its arguments
        unread, because there is no daylight saving to have a policy about.
        Refusing that would be firepanda turning away input pandas takes, which is
        the direction of difference this library does not get to have. The zone is
        asked for only when one of the two is not the default, since asking is a
        boundary crossing and the default changes no answer either way.
        """
        if (ambiguous != "raise" or nonexistent != "raise") and self._zone() is not None:
            _held_at(
                "ambiguous",
                ambiguous,
                "raise",
                "picking which of the two readings a repeated wall clock hour means"
                " needs the zone's transition table, which is the same work"
                " tz_localize over a fold needs",
            )
            if not isinstance(nonexistent, datetime.timedelta):
                _spelled(nonexistent, _NONEXISTENT, _NONEXISTENT_REFUSAL)
            _held_at(
                "nonexistent",
                nonexistent,
                "raise",
                "shifting a wall clock time that a spring forward skipped needs the"
                " zone's transition table",
            )
        if not isinstance(freq, str):
            raise NotImplementedError(
                "freq has to be a string for now, because an offset object carries"
                " the whole frequency vocabulary and firepanda parses the string"
                " spelling only"
            )
        return self._part(kind, freq)

    def _as_unit(self, unit: str, round_ok: bool) -> Series:
        """Restates the column in another resolution."""
        _held_at(
            "round_ok",
            round_ok,
            True,
            "refusing a cast that would lose precision rather than rounding it"
            " needs the cast to look at the values first, and it looks at the"
            " types only",
        )
        return self._part("as_unit", unit)

    def _named(self, kind: str, locale: Any) -> Series:
        """Writes out the name of the day or the month."""
        if locale is not None and not isinstance(locale, str):
            raise TypeError(f"locale has to be a string, not {type(locale).__name__}")
        return self._part(kind, "" if locale is None else locale)

    def _tz_convert(self, tz: Any) -> Series:
        """Reads the same instants against another clock."""
        if tz is None:
            raise NotImplementedError(
                "tz_convert(None) moves the column to UTC and then takes the clock"
                " off, and taking the clock off is tz_localize(None), so this is"
                " two operations pandas spells as one"
            )
        if not isinstance(tz, str):
            raise NotImplementedError(
                "tz has to be a zone name for now, because a tzinfo object is a"
                " Python object and the kernel reads the zone out of a string"
            )
        return self._part("tz_convert", tz)

    def _tz_localize(self, tz: Any, ambiguous: Any, nonexistent: Any) -> Series:
        """Puts the readings on a clock, or takes them off one.

        `None` is a different operation rather than an absent argument, which is
        why it crosses as its own word. Naming a zone keeps the readings and
        changes what they mean, and passing None keeps the instants and drops
        what they were read against.
        """
        _held_at(
            "ambiguous",
            ambiguous,
            "raise",
            "a wall clock hour that a fall back repeats is two instants and"
            " choosing between them needs the zone's transition table",
        )
        if not isinstance(nonexistent, datetime.timedelta):
            _spelled(nonexistent, _NONEXISTENT, _NONEXISTENT_REFUSAL)
        _held_at(
            "nonexistent",
            nonexistent,
            "raise",
            "a wall clock time that a spring forward skipped is no instant at"
            " all and shifting it needs the zone's transition table",
        )
        if tz is None:
            return self._part("tz_localize_none", "")
        if not isinstance(tz, str):
            raise NotImplementedError(
                "tz has to be a zone name for now, because a tzinfo object is a"
                " Python object and the kernel reads the zone out of a string"
            )
        return self._part("tz_localize", tz)

    def _isocalendar(self) -> DataFrame:
        """The ISO 8601 week date fields, as a frame of three columns.

        The one part of the accessor that goes through a module level function
        rather than a method on the series. The reason is the import graph and it
        is written where the function is.
        """
        from ._frame import _isocalendar

        return _isocalendar(self._series._inner)


class CategoricalMixin:
    """The hand written half of `CategoricalAccessor`.

    Eleven names in pandas and three doors under them. A rename is decided by
    position, setting the categories is decided by value, and dropping the unused
    ones is decided by the codes, and everything else on the accessor is one of
    those three with a list worked out first. That arithmetic is here rather than
    in the extension because it is about the pandas surface: which disagreement
    is a `ValueError`, what a `dict` passed to `rename_categories` means, and
    whether a missing `ordered` keeps what the column had are all questions with
    a pandas answer rather than a kernel answer.

    The checks come before the call in every case, so a refusal names what the
    caller passed rather than what the boundary made of it. The messages are the
    pandas ones word for word, because a program catching a `ValueError` off
    pandas and matching on its text is a program that exists.
    """

    __slots__ = ("_series",)
    """The series the accessor was reached from, held for the reason
    `DatetimeMixin` gives."""

    _series: Series

    def __init__(self, data: Series) -> None:
        """Holds the series, and refuses one that is not a categorical.

        The one accessor that checks the column's type when it is built rather
        than when it is used. pandas does the same and raises an `AttributeError`
        with this text, which reads oddly for a type complaint and is right: a
        caller who wrote `s.cat` on a column of numbers asked for an attribute
        the object does not have, and code that guards with `hasattr` should get
        a False rather than an exception.
        """
        if data.dtype != "category":
            raise AttributeError("Can only use .cat accessor with a 'category' dtype")
        self._series = data

    def _levels(self) -> Index:
        """The categories, as an index."""
        from ._frame import Index

        try:
            return Index._wrap(self._series._inner.categories())
        except Exception as error:
            raise translate(error) from None

    def _ordered(self) -> bool:
        """Whether the category order means anything."""
        try:
            return self._series._inner.ordered()
        except Exception as error:
            raise translate(error) from None

    def _codes(self) -> Series:
        """The codes, as a column of positions.

        int32 here and int8 in pandas, which is a difference a caller can see
        through `dtype` and is the one deliberate divergence on this accessor.
        The width is what the encoder writes and the reason it writes it is in
        document 26.
        """
        from ._frame import Series

        try:
            return Series._wrap(self._series._inner.codes())
        except Exception as error:
            raise translate(error) from None

    def _with_order(self, ordered: bool) -> Series:
        """The same column under a type that says whether the order matters."""
        from ._frame import Series

        try:
            return Series._wrap(self._series._inner.set_ordered(ordered))
        except Exception as error:
            raise translate(error) from None

    def _thinned(self) -> Series:
        """The same values over only the categories that appear in them."""
        from ._frame import Series

        try:
            return Series._wrap(self._series._inner.drop_unused_categories())
        except Exception as error:
            raise translate(error) from None

    def _held(self) -> list[str]:
        """The categories the column has, as a plain list.

        A category that is missing is refused here. Arrow permits one and pandas
        does not, so a column that arrived over the C data interface can have one
        and nothing on this accessor has an answer for it: a list of labels with
        a hole in it cannot be compared against what the caller passed.
        """
        out: list[str] = []
        for label in self._levels().tolist():
            if not isinstance(label, str):
                raise NotImplementedError(
                    "one of this column's categories is missing, which Arrow permits"
                    " and pandas does not, and a category that is nothing cannot be"
                    " added to, removed or renamed"
                )
            out.append(label)
        return out

    def _wanted(self, value: Any, name: str) -> list[str]:
        """Reads what a caller passed as a list of category labels.

        A single label rather than a list is accepted, which pandas does too, so
        `add_categories("z")` means what it looks like it means. Everything else
        is iterated, and a label that is not a string is refused here rather than
        at the boundary, because firepanda holds categories as text and the
        message about that should name the value.
        """
        one = [value] if isinstance(value, str) else list(value)
        for label in one:
            if not isinstance(label, str):
                raise NotImplementedError(
                    f"{name} has to be text for now, because firepanda holds a"
                    f" category column's categories in a text column, and {label!r} is"
                    f" a {type(label).__name__}"
                )
        return one

    def _distinct(self, names: list[str]) -> None:
        """Refuses a category list with a repeat in it.

        The kernel refuses one too and says so in its own words. This is here so
        the message is the pandas one, since a caller catching the `ValueError`
        and matching on its text is doing an ordinary thing. It sits on the two
        doors rather than on the argument reader, because `reorder_categories`
        reports a repeat as a disagreement with the old categories and has to get
        its own answer in first.
        """
        if len(set(names)) != len(names):
            raise InvalidArgumentError("Categorical categories must be unique")

    def _relabel(self, names: list[str], ordered: bool) -> Series:
        """The positional door, which leaves every code where it is."""
        from ._frame import Series

        self._distinct(names)
        try:
            return Series._wrap(self._series._inner.relabel_categories(names, ordered))
        except Exception as error:
            raise translate(error) from None

    def _against(self, names: list[str], ordered: bool) -> Series:
        """The value door, which nulls the rows whose category is not in the list."""
        from ._frame import Series

        self._distinct(names)
        try:
            return Series._wrap(self._series._inner.recategorize(names, ordered))
        except Exception as error:
            raise translate(error) from None

    def _added(self, new_categories: Any) -> Series:
        """Adds categories nothing uses yet, on the end of the ones there are."""
        wanted = self._wanted(new_categories, "new_categories")
        held = self._held()
        clash = {label for label in wanted if label in held}
        if clash:
            raise InvalidArgumentError(f"new categories must not include old categories: {clash}")
        return self._against(held + wanted, self._ordered())

    def _removed(self, removals: Any) -> Series:
        """Drops named categories, and the rows that were in them become missing."""
        wanted = self._wanted(removals, "removals")
        held = self._held()
        missing = {label for label in wanted if label not in held}
        if missing:
            raise InvalidArgumentError(f"removals must all be in old categories: {missing}")
        return self._against([label for label in held if label not in wanted], self._ordered())

    def _renamed(self, new_categories: Any) -> Series:
        """Gives the categories new labels, keeping every row where it is.

        Three shapes arrive here and pandas takes all three. A list is one label
        per category in order. A dict names the ones that change and leaves the
        rest, and a key that is not a category is ignored rather than refused,
        which is pandas and is worth knowing. A callable is applied to each.
        """
        held = self._held()
        if callable(new_categories):
            return self._relabel([new_categories(label) for label in held], self._ordered())
        if isinstance(new_categories, dict):
            return self._relabel(
                [new_categories.get(label, label) for label in held], self._ordered()
            )
        wanted = self._wanted(new_categories, "new_categories")
        if len(wanted) != len(held):
            raise InvalidArgumentError(
                "new categories need to have the same number of items as the old categories!"
            )
        return self._relabel(wanted, self._ordered())

    def _reordered(self, new_categories: Any, ordered: Any) -> Series:
        """Puts the same categories in another order, moving the codes to match."""
        wanted = self._wanted(new_categories, "new_categories")
        if set(wanted) != set(self._held()) or len(wanted) != len(self._held()):
            raise InvalidArgumentError(
                "items in new_categories are not the same as in old categories"
            )
        return self._against(wanted, self._ordered() if ordered is None else bool(ordered))

    def _set(self, new_categories: Any, ordered: Any, rename: bool) -> Series:
        """Sets the categories outright, by value or by position.

        `rename=True` is the one place the positional door is reachable with a
        count that does not match, and pandas is explicit about what that means:
        a shorter list drops the categories off the end and the rows that were in
        them go missing, and a longer one leaves the extra labels there with
        nothing in them.
        """
        wanted = self._wanted(new_categories, "new_categories")
        wants = self._ordered() if ordered is None else bool(ordered)
        if rename:
            return self._relabel(wanted, wants)
        return self._against(wanted, wants)


def _grouped(
    frame: DataFrame,
    by: Any,
    level: Any,
    as_index: bool,
    sort: bool,
    group_keys: bool,
    observed: bool,
    dropna: bool,
) -> DataFrameGroupBy:
    """Builds the group by object `df.groupby(...)` hands back.

    Written rather than generated because it is the one member whose body is not
    a call on something the class already holds. It builds a different class,
    and the seven arguments have to be read before there is an object to read
    them into, so a generated one line delegation has nothing to delegate to.

    Two of the seven are declared and refused. `group_keys` decides whether the
    key comes back in the result of an `apply`, and there is no `apply` here for
    it to decide about. `observed` decides whether a categorical key contributes
    the groups it has no rows for, and pandas made True the default in version
    3, which is the behaviour here, so it is refused only at False.

    Args:
        frame: The frame being grouped.
        by: The key column name or names.
        level: Declared and refused, since there is no MultiIndex to have one.
        as_index: Whether the key becomes the row labels.
        sort: Whether the groups come out in key order.
        group_keys: Declared and refused.
        observed: Declared and held at True.
        dropna: Whether a missing key is a group.

    Returns:
        A `DataFrameGroupBy`.
    """
    from ._frame import DataFrameGroupBy

    _held_at(
        "group_keys",
        group_keys,
        True,
        "it says whether the key comes back in the result of an apply, and there"
        " is no apply here for it to say anything about",
    )
    _held_at(
        "observed",
        observed,
        True,
        "it says whether a categorical key contributes the groups it has no rows"
        " for, and a group with no rows in it is a row this has nothing to put in",
    )
    keys = GroupByMixin._keys(frame, by, level)
    return DataFrameGroupBy(frame, keys, as_index, sort, dropna)


def _relabelled(frame: DataFrame, name: str, label: str) -> Series:
    """Takes one column out of a frame and puts a different name on it.

    A group by reduction comes back as a frame, and turning it into the series
    pandas answers means taking one column out, at which point the column is
    called whatever the frame called it. `size` is called `size` there and has no
    name at all in pandas, and a narrowed group by answers a column called after
    the one the caller asked for rather than after the last key. Both are one
    rename and neither is a member a user reaches, so it is a function here
    rather than a method on the wrapper.

    The column is taken through the extension rather than through `frame[name]`,
    which reads a name and a list of names and therefore answers either shape.
    There is one name here and the answer is a column, so going the short way
    says that rather than leaving it to be narrowed afterwards.

    Args:
        frame: The frame the reduction produced.
        name: The column to take out.
        label: The new name, and empty for none.

    Returns:
        The column, carrying the new name.
    """
    from ._frame import Series

    try:
        return Series._wrap(frame._inner.column(name).relabel(label))
    except Exception as error:
        raise translate(error) from None


_CLOSED = ("right", "left", "both", "neither")
"""The four words pandas takes for which ends a window keeps, in the order its
own error message lists them."""

_BETWEEN = ("linear", "lower", "higher", "midpoint", "nearest")
"""The five words pandas takes for what a quantile does when its position lands
between two values."""

_TIED = ("average", "min", "max")
"""The three words a window rank takes for what to do with values that are equal.
`Series.rank` takes `dense` and `first` as well and a window does not, which is
pandas' own difference and is kept."""


class WindowMixin:
    """What `Rolling` and `Expanding` share, which is everything after the width.

    A window object holds some data and five numbers saying where each window
    sits, and computes nothing until a reduction is asked for. That is pandas'
    arrangement and it is why these two classes are so nearly empty: the five
    reductions are one call each with a different word in it, and the word is
    the pandas method name, which is the same string the boundary reads.

    The five numbers are checked in the constructor and not at the reduction,
    because pandas checks them there. `s.rolling(-1)` raises out of the
    `rolling` call and not out of the `.sum()` after it, and a program that
    catches the wrong line is a program whose error handling does not run. The
    kernel checks them again on the way in, which is not duplication worth
    removing: one of the two checks exists to be reached from Python and the
    other exists because the Mojo API is a public entry point of its own.

    The data is a column or a frame, and almost nothing here looks at which. A
    window is a pair of row numbers, every column of a frame has the same rows,
    so a frame window is the columns windowed one at a time. The two places that
    do look are the reduction, which has a different class to hand back, and
    `numeric_only`, which is a different question on each and is argued at
    `_reduce`.

    The two subclasses differ in their constructors only. A rolling window takes
    a width and defaults `min_periods` to it, and an expanding window takes no
    width and defaults `min_periods` to one. Everything below that is the same
    call.
    """

    __slots__ = ("_center", "_closed", "_data", "_min_periods", "_step", "_window")
    """The data and the five numbers that survive to the reduction. Slotted for
    the reason `DataFrameMixin` gives. There is no `_inner`, because a window
    object has no extension object of its own: it is some data and a plan for
    which rows to read together."""

    _data: Series | DataFrame
    _window: int | None
    _min_periods: int | None
    _center: bool
    _closed: str | None
    _step: int | None

    def _hold(
        self,
        data: Series | DataFrame,
        window: int | None,
        min_periods: int | None,
        center: bool,
        closed: str | None,
        step: int | None,
    ) -> None:
        """Checks the five and keeps them. Not a public entry point.

        The five are kept exactly as they arrived, including the two that have a
        default the caller did not write. `closed` stays None rather than
        becoming `right` and `min_periods` stays None rather than becoming the
        width, because a window object in pandas reports back what it was given
        and not what it resolved to, and `df.rolling(2).closed` is None there.
        The defaults are applied on the way to the kernel instead, which is one
        line later and one lie fewer.

        Args:
            data: The column or the frame.
            window: How many rows wide, and None for an expanding window.
            min_periods: How many values a window needs, and None for the
                default of whichever window type this is.
            center: Whether the window sits around its row.
            closed: Which ends the window keeps, and None for `right`.
            step: How many rows apart the answered rows are, and None for every
                row.

        Raises:
            InvalidArgumentError: If the five do not describe a window, with the
                sentence pandas raises for each. A `ValueError`, which is what
                pandas raises and what a caller's except clause is looking for.
        """
        if center is not True and center is not False:
            raise InvalidArgumentError("center must be a boolean")
        if window is not None and (not isinstance(window, int) or isinstance(window, bool)):
            # pandas says the same thing for a float, for a string and for an
            # offset, because all three reach it as a window it cannot count
            # rows with. A window given as a duration needs a datetime index to
            # measure against and that is its own piece of work.
            raise InvalidArgumentError("window must be an integer 0 or greater")
        if window is not None and window < 0:
            raise InvalidArgumentError("window must be an integer 0 or greater")
        if min_periods is not None:
            if not isinstance(min_periods, int) or isinstance(min_periods, bool):
                raise InvalidArgumentError("min_periods must be an integer")
            if min_periods < 0:
                raise InvalidArgumentError("min_periods must be >= 0")
            if window is not None and min_periods > window:
                raise InvalidArgumentError(f"min_periods {min_periods} must be <= window {window}")
        if closed is not None and closed not in _CLOSED:
            raise InvalidArgumentError("closed must be 'right', 'left', 'both' or 'neither'")
        if step is not None:
            if not isinstance(step, int) or isinstance(step, bool):
                raise InvalidArgumentError("step must be an integer")
            # pandas accepts a step of zero here and divides by it later, which
            # is a ZeroDivisionError out of the reduction rather than a sentence
            # about the argument that caused it. A step of zero asks for the same
            # row forever, so it is refused where the rest of them are.
            if step < 1:
                raise InvalidArgumentError("step must be >= 1")
        self._data = data
        self._window = window
        self._min_periods = min_periods
        self._center = center
        self._closed = closed
        self._step = step

    def _over_frame(self) -> bool:
        """Whether this window is over a frame rather than a column.

        Asked in three places and written once, because the import has to be
        deferred: `_frame` imports this module to get the mixins it inherits, so
        this module cannot import `_frame` at the top of the file.

        Returns:
            True for a frame.
        """
        from ._frame import DataFrame

        return isinstance(self._data, DataFrame)

    def _spread_settings(self, ddof: int) -> tuple[Any, ...]:
        """Checks the one parameter the three spreads read and packs it.

        pandas takes a float here and truncates it, so `ddof=1.5` quietly answers
        the `ddof=1` column, and a caller who wrote that meant something and did
        not get it. A whole number is asked for and anything else is a sentence.
        A negative one is allowed, as it is in pandas, because it is a divisor
        larger than the count rather than a mistake.

        Args:
            ddof: Subtracted from the count of values to give the divisor.

        Returns:
            The one value, as the tuple `_reduce` passes on.

        Raises:
            InvalidArgumentError: If it is not a whole number.
        """
        if not isinstance(ddof, int) or isinstance(ddof, bool):
            raise InvalidArgumentError("ddof must be an integer")
        return (ddof,)

    def _quantile_settings(self, q: float, interpolation: str) -> tuple[Any, ...]:
        """Checks the two parameters a quantile reads and packs them.

        pandas asks whether the fraction is below nought or above one, which a
        NaN is neither of, so `quantile(float("nan"))` is accepted there and
        answers a column of NaN. The question here is whether the fraction is
        between nought and one, which a NaN is not, so it gets the same sentence
        a fraction of two gets. A caller who wrote that asked for a position in
        the window and there is no position to give them.

        Both of these are checked again in the kernel, which is the arrangement
        `closed` already has: the sentences pandas raises for them live on this
        side, and the kernel checks as well because it is also the door the Mojo
        API comes through.

        Args:
            q: How far through the sorted window to read, from nought to one.
            interpolation: Which rule to use when the position lands between two
                values.

        Returns:
            The two values, in the order pandas declares them.

        Raises:
            InvalidArgumentError: If the fraction is not a number between nought
                and one, or the rule is not one of the five.
        """
        if isinstance(q, bool) or not isinstance(q, (int, float)):
            raise InvalidArgumentError("must be real number, not " + type(q).__name__)
        if not 0.0 <= float(q) <= 1.0:
            raise InvalidArgumentError(f"quantile value {q} not in [0, 1]")
        if interpolation not in _BETWEEN:
            raise InvalidArgumentError(f"Interpolation '{interpolation}' is not supported")
        return (float(q), interpolation)

    def _rank_settings(self, method: str, ascending: bool, pct: bool) -> tuple[Any, ...]:
        """Checks the three parameters a rank reads and packs them.

        `Series.rank` accepts five tie rules and `Rolling.rank` accepts three of
        them, which is pandas' own difference and is kept here, because the two
        that are missing are the two a window cannot answer out of a count of
        ranks alone. The sentence is pandas' sentence.

        Args:
            method: What to do with values that are equal.
            ascending: Whether to count from the smallest value.
            pct: Whether to divide the rank by how many values the window holds.

        Returns:
            The three values, in the order pandas declares them.

        Raises:
            InvalidArgumentError: If the rule is not one of the three, or either
                flag is not a boolean.
        """
        if method not in _TIED:
            raise InvalidArgumentError(f"Method '{method}' is not supported")
        if ascending is not True and ascending is not False:
            raise InvalidArgumentError("ascending must be a boolean")
        if pct is not True and pct is not False:
            raise InvalidArgumentError("pct must be a boolean")
        return (method, ascending, pct)

    def _reduce(
        self,
        kind: str,
        numeric_only: bool = False,
        engine: Any = None,
        engine_kwargs: Any = None,
        settings: tuple[Any, ...] = (),
    ) -> Series | DataFrame:
        """Runs one reduction over every window.

        `numeric_only` is one name asking two questions, and it gets a different
        answer on each. On a column it says to refuse a column that is not a
        number, and every reduction here already refuses one, so both values
        agree everywhere this library has an answer and both are accepted. On a
        frame it says to drop the columns that cannot be reduced rather than
        refuse them, which is a decision about which columns come back, so it is
        held at False and True is refused. That is the rule the group by path
        already follows and the sentence there is the same sentence.

        `engine` is the numba path and is refused, since there is no second
        implementation for it to pick. `cython` is the default path spelled out
        and is accepted.

        Args:
            kind: The reduction, as pandas spells the method.
            numeric_only: Held at False over a frame, and accepted at both
                values over a column, for the reason above.
            engine: Declared and refused, except at `cython`.
            engine_kwargs: Declared and refused. Last of the positional ones, so
                that `settings` can sit after it and the eight reductions that
                send nothing keep the call they already had.
            settings: The parameters the reduction reads and the window does not,
                already checked and in the order pandas declares them. Empty for
                the eight that read none of it. The three helpers above build it
                and are the only things that should.

        Returns:
            Whichever of the two was windowed, of float64, as tall as what it
            read unless a step made it shorter.

        Raises:
            NotImplementedError: If a numba engine was asked for, or if a frame
                was asked to drop the columns it cannot reduce.
        """
        from ._frame import DataFrame, Series

        if isinstance(self._data, DataFrame):
            _held_at(
                "numeric_only",
                numeric_only,
                False,
                "dropping the columns a window cannot read is a decision about"
                " which columns come back, and firepanda windows the ones it was"
                " given or says which one it could not",
            )
        if engine is not None and engine != "cython":
            raise NotImplementedError(
                f"engine={engine!r} is not supported yet, because there is one"
                " implementation here and it is the one cython names"
            )
        _refuse(
            "engine_kwargs",
            engine_kwargs,
            "it configures the numba engine, and there is no numba engine here for it to configure",
        )
        # The one default `_hold` did not apply is applied here. `right` is
        # pandas' word for a window that keeps the row it is answering and not
        # the one that fell off the far end. The absent `min_periods` is left
        # absent and crosses that way, because its default is the width on one of
        # these two classes and one on the other, and the side that knows which
        # is the kernel.
        plan = (
            kind,
            self._window,
            self._min_periods,
            self._center,
            self._closed or "right",
            self._step,
            settings,
        )
        try:
            if isinstance(self._data, DataFrame):
                return DataFrame._wrap(self._data._inner.window_agg(*plan))
            return Series._wrap(self._data._inner.window_agg(*plan))
        except Exception as error:
            raise translate(error) from None


class RollingMixin(WindowMixin):
    """The hand written half of `Rolling`, which is its constructor."""

    __slots__ = ()
    """All the state is `WindowMixin`'s."""

    def __init__(
        self,
        data: Series | DataFrame,
        window: int,
        min_periods: int | None,
        center: bool,
        closed: str | None,
        step: int | None,
    ) -> None:
        """Holds the data and the window. Not a public entry point.

        Args:
            data: The column or the frame.
            window: How many rows wide.
            min_periods: How many values a window needs, and None for the width.
            center: Whether the window sits around its row.
            closed: Which ends the window keeps, and None for `right`.
            step: How many rows apart the answered rows are, and None for every
                row.
        """
        self._hold(data, window, min_periods, center, closed, step)


class ExpandingMixin(WindowMixin):
    """The hand written half of `Expanding`, which is its constructor."""

    __slots__ = ()
    """All the state is `WindowMixin`'s."""

    def __init__(self, data: Series | DataFrame, min_periods: int) -> None:
        """Holds the data and how long it waits. Not a public entry point.

        The width is None rather than the height of the data, and that is the
        one place the absence is written down on the Python side: the data can
        grow between here and the reduction in pandas and the width is resolved
        against whatever it is when the reduction runs, so filling it in now
        would be answering a question that has not been asked yet.

        Args:
            data: The column or the frame.
            min_periods: How many values a window needs before it answers.
        """
        self._hold(data, None, min_periods, False, None, None)


def _rolling(
    data: Series | DataFrame,
    window: Any,
    min_periods: int | None,
    center: bool,
    win_type: str | None,
    on: str | None,
    closed: str | None,
    step: int | None,
    method: str,
) -> Rolling:
    """Builds the window object `s.rolling(...)` and `df.rolling(...)` hand back.

    Written rather than generated for the reason `_grouped` gives, which is that
    it builds a different class and the arguments have to be read before there
    is an object to read them into. One function for both owners, because the
    nine arguments mean the same thing on each and the class that comes out is
    the same class.

    Three of the nine are declared and refused. `win_type` asks for a weighted
    window, which is a different kernel and not a parameter of this one, and
    pandas needs scipy for it. `on` says to take the window's ordering from
    another column, and on a frame it also carries that column through into the
    answer unreduced. The carrying is the easy half and the ordering is the
    whole point, and ordering by a column means a window given as a duration,
    which needs a calendar first. Writing the half that copies a column would be
    a `rolling("2D", on="t")` that silently counted rows. `method` chooses
    between reducing each column separately and reducing them together, and this
    library reduces them separately.

    Args:
        data: The column or the frame.
        window: How many rows wide.
        min_periods: How many values a window needs.
        center: Whether the window sits around its row.
        win_type: Declared and refused.
        on: Declared and refused.
        closed: Which ends the window keeps.
        step: How many rows apart the answered rows are.
        method: Declared and held at `single`.

    Returns:
        A `Rolling`.
    """
    from ._frame import Rolling

    _refuse(
        "win_type",
        win_type,
        "it asks for a weighted window, which is a different kernel from an"
        " unweighted one rather than a parameter of this one",
    )
    _refuse(
        "on",
        on,
        "it says to order the window by another column, and a window measured in"
        " rows is already ordered by rows, so it would only mean something once a"
        " window can be given as a duration",
    )
    _held_at(
        "method",
        method,
        "single",
        "it says whether the columns are reduced together, and here they are"
        " reduced one at a time, which is what a window over a pair of row"
        " numbers can do without holding the whole frame at once",
    )
    return Rolling(data, window, min_periods, center, closed, step)


def _expanding(data: Series | DataFrame, min_periods: int, method: str) -> Expanding:
    """Builds the window object `s.expanding(...)` and `df.expanding(...)` hand back.

    Args:
        data: The column or the frame.
        min_periods: How many values a window needs before it answers.
        method: Declared and held at `single`, for the reason `_rolling` gives.

    Returns:
        An `Expanding`.
    """
    from ._frame import Expanding

    _held_at(
        "method",
        method,
        "single",
        "it says whether the columns are reduced together, and here they are"
        " reduced one at a time, which is what a window over a pair of row"
        " numbers can do without holding the whole frame at once",
    )
    return Expanding(data, min_periods)


class EwmMixin:
    """The hand written half of `ExponentialMovingWindow`.

    A window object that holds what it was told and computes nothing until a
    reduction is asked for, which is the arrangement `WindowMixin` has and is
    pandas' arrangement for both. It is a separate mixin rather than a third
    subclass of that one because it shares none of the state: there is no width,
    no centring, no closed rule and no step, and in their place there are four
    spellings of one decay and two flags that change the recurrence.
    `firepanda/kernel/ewm.mojo` argues why that is a different window and not a
    narrower one.

    The decay is checked here and collapsed to one number here, and it is checked
    again at the kernel's door. That is not duplication worth removing, for the
    reason `WindowMixin` gives about the five numbers: one of the two checks
    exists to be reached from Python with pandas' own sentence, and the other
    exists because the Mojo API is a public entry point of its own.

    The four are reported back uncollapsed. `ewm(span=5).com` is None in pandas
    and answering 2.0 here would be reporting a conversion rather than an
    argument, which is the same rule that keeps `closed` at None on a rolling
    window.
    """

    __slots__ = (
        "_adjust",
        "_alpha",
        "_com",
        "_data",
        "_factor",
        "_halflife",
        "_ignore_na",
        "_min_periods",
        "_span",
    )
    """The data, the four spellings, the one number they collapsed to, and the
    three things that change the answer. Slotted for the reason `DataFrameMixin`
    gives. There is no `_inner`, because a window object has no extension object
    of its own."""

    _data: Series | DataFrame
    _com: float | None
    _span: float | None
    _halflife: float | None
    _alpha: float | None
    _factor: float
    _min_periods: int
    _adjust: bool
    _ignore_na: bool

    def __init__(
        self,
        data: Series | DataFrame,
        com: float | None,
        span: float | None,
        halflife: float | None,
        alpha: float | None,
        min_periods: int | None,
        adjust: bool,
        ignore_na: bool,
    ) -> None:
        """Checks the decay, collapses it, and keeps the rest. Not a public entry
        point.

        `min_periods` is the one argument that is not kept as it arrived, because
        pandas does not keep it either: it resolves the default of nought to one
        in its own constructor, so `ewm(span=5, min_periods=0).min_periods` is 1
        there. A negative one also resolves to one rather than being refused,
        which is pandas' behaviour and is kept, because a count of values that
        cannot be below one is not a range a caller can be wrong about.

        Args:
            data: The column or the frame.
            com: The centre of mass, or None.
            span: The span, or None.
            halflife: The half life in rows, or None.
            alpha: The smoothing factor, or None.
            min_periods: How many values a row needs, and None or nought for one.
            adjust: Whether every row weighs one rather than the factor.
            ignore_na: Whether a missing row is skipped.

        Raises:
            InvalidArgumentError: If none of the four spellings arrived, if more
                than one did, if the one that did is outside the range pandas
                allows for it, or if either flag is not a boolean. A
                `ValueError`, which is what pandas raises for the decay.
        """
        if adjust is not True and adjust is not False:
            raise InvalidArgumentError("adjust must be a boolean")
        if ignore_na is not True and ignore_na is not False:
            raise InvalidArgumentError("ignore_na must be a boolean")
        if min_periods is not None and (
            not isinstance(min_periods, int) or isinstance(min_periods, bool)
        ):
            raise InvalidArgumentError("min_periods must be an integer")
        self._data = data
        self._com = com
        self._span = span
        self._halflife = halflife
        self._alpha = alpha
        self._factor = _smoothing(com, span, halflife, alpha)
        self._min_periods = max(int(min_periods), 1) if min_periods else 1
        self._adjust = adjust
        self._ignore_na = ignore_na

    def _over_frame(self) -> bool:
        """Whether this decay is over a frame rather than a column.

        Written again rather than shared with `WindowMixin`, because the two
        classes share no state and inheriting one method from a base that holds
        five numbers none of these objects has would be the wrong shape for one
        line.

        Returns:
            True for a frame.
        """
        from ._frame import DataFrame

        return isinstance(self._data, DataFrame)

    def _bias_settings(self, bias: bool) -> tuple[Any, ...]:
        """Checks the one parameter the two spreads read and packs it.

        pandas takes anything here and reads it for truth, so `bias="no"` quietly
        answers the biased column, which is the opposite of what the caller who
        wrote it meant. A boolean is asked for and anything else is a sentence,
        which is what `_spread_settings` does with `ddof` and for the same
        reason.

        Args:
            bias: Whether to answer the second moment itself, uncorrected.

        Returns:
            The one value, as the tuple `_reduce` passes on.

        Raises:
            InvalidArgumentError: If it is not a boolean.
        """
        if bias is not True and bias is not False:
            raise InvalidArgumentError("bias must be a boolean")
        return (bias,)

    def _reduce(
        self,
        kind: str,
        numeric_only: bool = False,
        engine: Any = None,
        engine_kwargs: Any = None,
        settings: tuple[Any, ...] = (),
    ) -> Series | DataFrame:
        """Runs one reduction under the decay.

        `numeric_only`, `engine` and `engine_kwargs` are read exactly the way
        `WindowMixin._reduce` reads them and the arguments there are the same
        arguments here, so they are not made again.

        The one thing this does that the window one does not is refuse a total
        under the unadjusted recurrence. pandas raises `NotImplementedError` with
        the sentence below rather than choosing one of the two things such a
        total could mean, and the refusal is made here rather than left to the
        kernel so that the class of the exception is the class pandas raises. The
        kernel refuses it as well, because the Mojo API does not come through
        here.

        Args:
            kind: The reduction, as pandas spells the method.
            numeric_only: Held at False over a frame and accepted at both values
                over a column.
            engine: Declared and refused, except at `cython`.
            engine_kwargs: Declared and refused.
            settings: The parameters the reduction reads and the decay does not,
                already checked. Empty for the mean and the total, and `bias` for
                the two spreads.

        Returns:
            Whichever of the two was decayed, of float64, as tall as what it
            read.

        Raises:
            NotImplementedError: If a numba engine was asked for, if a frame was
                asked to drop the columns it cannot reduce, or if a total was
                asked for with `adjust` off, which pandas also refuses.
        """
        from ._frame import DataFrame, Series

        if isinstance(self._data, DataFrame):
            _held_at(
                "numeric_only",
                numeric_only,
                False,
                "dropping the columns a decay cannot read is a decision about"
                " which columns come back, and firepanda decays the ones it was"
                " given or says which one it could not",
            )
        if engine is not None and engine != "cython":
            raise NotImplementedError(
                f"engine={engine!r} is not supported yet, because there is one"
                " implementation here and it is the one cython names"
            )
        _refuse(
            "engine_kwargs",
            engine_kwargs,
            "it configures the numba engine, and there is no numba engine here for it to configure",
        )
        if kind == "sum" and not self._adjust:
            raise NotImplementedError("sum is not implemented with adjust=False")
        plan = (
            kind,
            self._factor,
            self._min_periods,
            self._adjust,
            self._ignore_na,
            settings,
        )
        try:
            if isinstance(self._data, DataFrame):
                return DataFrame._wrap(self._data._inner.ewm_agg(*plan))
            return Series._wrap(self._data._inner.ewm_agg(*plan))
        except Exception as error:
            raise translate(error) from None


def _smoothing(
    com: float | None,
    span: float | None,
    halflife: float | None,
    alpha: float | None,
) -> float:
    """Turns whichever spelling of the decay arrived into one smoothing factor.

    The four are four ways of writing one number and pandas takes exactly one of
    them, which is the right rule: a caller who gives two has said something
    contradictory rather than something redundant. The sentences are pandas'
    sentences, including its spelling of the centre of mass as `comass`, which is
    what its own message says whatever the argument is called.

    Args:
        com: The centre of mass, or None.
        span: The span, or None.
        halflife: The half life in rows, or None.
        alpha: The smoothing factor, or None.

    Returns:
        The factor, above nought and at most one.

    Raises:
        InvalidArgumentError: If none of the four arrived, if more than one did,
            or if the one that did is outside the range pandas allows for it. A
            `ValueError`, which is what pandas raises for all three.
    """
    given = [
        (name, value)
        for name, value in (
            ("com", com),
            ("span", span),
            ("halflife", halflife),
            ("alpha", alpha),
        )
        if value is not None
    ]
    if not given:
        raise InvalidArgumentError("Must pass one of comass, span, halflife, or alpha")
    if len(given) > 1:
        raise InvalidArgumentError("comass, span, halflife, and alpha are mutually exclusive")
    # The one that arrived is carried out of the list rather than tested for
    # again, because the four branches below are four conversions of one number
    # and asking which of the four is present twice is how the two halves get
    # out of step.
    name, arrived = given[0]
    _real(name, arrived)
    number = float(arrived)
    if name == "span":
        if number < 1.0:
            raise InvalidArgumentError("span must satisfy: span >= 1")
        return 2.0 / (number + 1.0)
    if name == "com":
        if number < 0.0:
            raise InvalidArgumentError("comass must satisfy: comass >= 0")
        return 1.0 / (1.0 + number)
    if name == "halflife":
        if number <= 0.0:
            raise InvalidArgumentError("halflife must satisfy: halflife > 0")
        return 1.0 - math.exp(-math.log(2.0) / number)
    if not 0.0 < number <= 1.0:
        raise InvalidArgumentError("alpha must satisfy: 0 < alpha <= 1")
    return number


def _real(name: str, value: Any) -> None:
    """Refuses a decay that is not a number at all.

    pandas lets a half life arrive as a duration, which is the one of the four
    that has a second reading, and that reading needs a calendar first. So
    anything that is not a plain number is refused here rather than being
    converted to one and quietly counted as rows.

    Args:
        name: The argument, for the sentence.
        value: What arrived.

    Raises:
        InvalidArgumentError: If it is not a real number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidArgumentError(f"{name} must be a real number, not {type(value).__name__}")


def _ewm(
    data: Series | DataFrame,
    com: float | None,
    span: float | None,
    halflife: float | None,
    alpha: float | None,
    min_periods: int | None,
    adjust: bool,
    ignore_na: bool,
    times: Any,
    method: str,
) -> ExponentialMovingWindow:
    """Builds the object `s.ewm(...)` and `df.ewm(...)` hand back.

    Written rather than generated for the reason `_rolling` gives. Two of the
    nine arguments are declared and refused. `times` says to measure the decay
    against real instants rather than against row positions, so a half life of
    two days means two days however many rows fell in them, and that needs a
    calendar first. `method` chooses between decaying down the columns and
    decaying across them, and across them is a different computation rather than
    a different arrangement of this one.

    Args:
        data: The column or the frame.
        com: The centre of mass.
        span: The span.
        halflife: The half life in rows.
        alpha: The smoothing factor.
        min_periods: How many values a row needs before it is answered.
        adjust: Whether every row weighs one rather than the factor.
        ignore_na: Whether a missing row is skipped.
        times: Declared and refused.
        method: Declared and held at `single`.

    Returns:
        An `ExponentialMovingWindow`.
    """
    from ._frame import ExponentialMovingWindow

    _refuse(
        "times",
        times,
        "it says to measure the decay against real instants rather than against"
        " row positions, which means a half life given as a duration, and that"
        " needs a calendar first",
    )
    _held_at(
        "method",
        method,
        "single",
        "it says whether the columns decay together, and here they decay one at a"
        " time, which is the reading pandas calls single and defaults to",
    )
    return ExponentialMovingWindow(data, com, span, halflife, alpha, min_periods, adjust, ignore_na)


class StringMixin:
    """The hand written half of `StringAccessor`.

    Three doors under it, picked by the shape of the answer rather than by the
    shape of the arguments, which is the rule `firepanda/py/text.mojo` states and
    argues for. Everything here turns what pandas lets a caller write into the
    word and the four values a door takes.

    Several things are decided here rather than in the kernel, and all of them
    are decided here because they are about the pandas surface. A width that is
    not a whole number, a fill character that is not one character, and a side
    that is not one of three words are all refused with the message pandas would
    have given, before anything crosses. `strip(None)` and `strip("")` are told
    apart here as well, because the absence has to pick the word the crossing
    carries and there is no way to spell an absent string on the other side.

    `index` and `rindex` are
    `find` and `rfind` that raise when the substring is missing from any row, and
    the exception is a `ValueError` whose text pandas copied from Python's
    `str.index`. And `startswith` accepts a tuple of prefixes, which is Python's
    signature rather than anything a column kernel should know about, so it is a
    fold of the one prefix answer over the tuple.
    """

    __slots__ = ("_series",)
    """The series the accessor was reached from, held for the reason
    `DatetimeMixin` gives."""

    _series: Series

    def __init__(self, data: Series) -> None:
        """Holds the series, and refuses one that is not text.

        The second accessor that checks the column when it is built rather than
        when it is used, and pandas checks this one the same way and with this
        text. A caller who wrote `s.str` on a column of numbers asked for an
        attribute the object does not have, so `hasattr` should answer False
        rather than raise.

        pandas ends the message with a name for what the column holds instead,
        and the name it uses is the one `infer_dtype` gives rather than the
        dtype: a column of int64 is `integer` there. The sentence is pandas' and
        the last word is ours, because inventing a second vocabulary for types
        so that one message can read like pandas would be the wrong trade.
        """
        if not data._inner.string_is_text():
            raise AttributeError(f"Can only use .str accessor with string values, not {data.dtype}")
        self._series = data

    def _text(
        self,
        kind: str,
        arg: str = "",
        start: int | None = None,
        stop: int | None = None,
        step: int = 1,
    ) -> Series:
        """Runs a method that answers text."""
        from ._frame import Series

        try:
            return Series._wrap(self._series._inner.string_text(kind, arg, start, stop, step))
        except Exception as error:
            raise translate(error) from None

    def _flag(self, kind: str, arg: str) -> Series:
        """Runs a method that answers a mask."""
        from ._frame import Series

        try:
            return Series._wrap(self._series._inner.string_flag(kind, arg))
        except Exception as error:
            raise translate(error) from None

    def _number(
        self,
        kind: str,
        arg: str = "",
        start: int | None = None,
        stop: int | None = None,
    ) -> Series:
        """Runs a method that answers a number."""
        from ._frame import Series

        try:
            return Series._wrap(self._series._inner.string_number(kind, arg, start, stop))
        except Exception as error:
            raise translate(error) from None

    def _sliced(self, start: Any, stop: Any, step: Any) -> Series:
        """A range of characters out of every row.

        `step=None` means one, which is Python's rule and not a default this
        library chose, and it is the only one of the three that can be filled in
        without knowing how long the row is.
        """
        return self._text("slice", "", start, stop, 1 if step is None else step)

    def _replaced_slice(self, start: Any, stop: Any, repl: Any) -> Series:
        """Every row with a range of characters swapped for a string.

        pandas lets `repl` be left out and means the empty string by it, which
        makes `slice_replace(1, 3)` a deletion rather than an error.
        """
        return self._text("slice_replace", "" if repl is None else repl, start, stop)

    def _at(self, i: Any) -> Series:
        """One character out of every row, by position."""
        return self._text("get", "", i)

    def _trimmed(self, kind: str, to_strip: Any) -> Series:
        """Every row with characters taken off one end or both.

        `None` and the empty string are two different requests and pandas keeps
        them apart, because it hands both straight to Python: `strip()` removes
        whitespace and `strip("")` removes nothing at all. So the absence picks
        the word rather than being filled in with a default set, and the two
        words reach two different calls in the kernel.
        """
        if to_strip is None:
            return self._text(kind)
        return self._text(f"{kind}_chars", str(to_strip))

    def _padded(self, width: Any, side: Any, fillchar: Any) -> Series:
        """Every row filled out to a width with a character.

        All four checks here are pandas' own, in pandas' order and with pandas'
        text. The kernel checks the fill as well, because it is reachable from
        Mojo too, but the message a Python caller reads should be the one they
        would have read from pandas and it says `str` where the kernel would
        have counted the characters.
        """
        if not isinstance(fillchar, str):
            raise DTypeError(
                f"firepanda:dtype: fillchar must be a character, not {type(fillchar).__name__}"
            )
        if len(fillchar) != 1:
            raise DTypeError("firepanda:dtype: fillchar must be a character, not str")
        width = self._width(width)
        if side not in ("left", "right", "both"):
            raise InvalidArgumentError(
                f"firepanda:value: Invalid side: {side}. Side must be one of"
                " 'left', 'right', 'both'"
            )
        return self._text(f"pad_{side}", fillchar, width)

    def _filled(self, width: Any) -> Series:
        """Every row filled out to a width with zeros, after any leading sign."""
        return self._text("zfill", "", self._width(width))

    def _repeated(self, repeats: Any) -> Series:
        """Every row written out several times, end to end.

        pandas also takes one count per row here, which is a second method
        wearing the same name: it answers a different column for every row and
        needs a column of counts crossing rather than a number. That form is not
        written yet and says so, rather than quietly repeating by the first count
        it can find.
        """
        if isinstance(repeats, (str, bytes)) or not isinstance(repeats, int):
            try:
                iter(repeats)
            except TypeError:
                pass
            else:
                raise UnsupportedError(
                    "firepanda:unsupported: str.repeat takes one count for the"
                    " whole column, and a count per row is not written yet"
                )
        return self._text("repeat", "", self._width(repeats, "repeats"))

    @staticmethod
    def _width(value: Any, name: str = "width") -> int:
        """Reads a count, and refuses anything that is not whole.

        pandas checks this itself and says so in pandas' words, which is worth
        the four lines: a caller who wrote `zfill("10")` gets told that the
        argument is the wrong type rather than getting a message about the
        column, and a bool is refused as well because `zfill(True)` is a mistake
        every time even though Python is happy to call it an integer.
        """
        if isinstance(value, bool) or not isinstance(value, int):
            raise DTypeError(
                f"firepanda:dtype: {name} must be of integer type, not {type(value).__name__}"
            )
        return value

    def _found(self, kind: str, sub: Any, start: Any, end: Any) -> Series:
        """Where a substring sits in every row, or -1 where it is not there."""
        return self._number(kind, sub, start, end)

    def _demanded(self, kind: str, sub: Any, start: Any, end: Any) -> Series:
        """The same, but a row that does not contain the substring is an error.

        pandas checks every row and raises once, which means the answer is
        computed in full before it is thrown away. That is what it costs to give
        the caller the exception they asked for, and doing it any other way would
        mean stopping at the first missing row and reporting a position for the
        rows before it, which is not an answer anybody can use.

        The class raised is ours rather than a plain `ValueError`, because the
        generated method around this one runs every error through `translate`
        and an untagged `ValueError` is exactly what `translate` turns into a
        `RuntimeError`. `InvalidArgumentError` is a `ValueError` as well, so a
        caller catching what pandas raises still catches it.
        """
        found = self._found(kind, sub, start, end)
        for value in found.tolist():
            if isinstance(value, int) and value < 0:
                raise InvalidArgumentError("substring not found")
        return found

    def _begins(self, kind: str, pat: Any, na: Any) -> Series:
        """Whether every row starts or ends with a string, or with any of several.

        A tuple is Python's signature for these two and it is the only place in
        the accessor where one argument stands for several questions. An empty
        tuple is False on every row that is not missing, which is what Python's
        `startswith(())` says.

        The fold over the tuple and the filling of `na` both happen over plain
        lists here, because `Series` has neither `|` nor `fillna` yet. Both are
        one column operation each once it does, and moving them down is worth
        doing the day it exists rather than writing two kernels nothing else
        would call.
        """
        if not isinstance(pat, tuple):
            answer = self._flag(kind, pat)
            if na is None:
                return answer
            return self._as_mask([na if one is None else one for one in answer.tolist()])
        rows = self._series._inner.length()
        held: list[Any] = [False] * rows
        for one in pat:
            for at, value in enumerate(self._flag(kind, one).tolist()):
                if value is None:
                    held[at] = None
                elif value:
                    held[at] = True
        if na is not None:
            held = [na if one is None else one for one in held]
        return self._as_mask(held)

    def _as_mask(self, values: list[Any]) -> Series:
        """Builds a boolean column out of a plain list, carrying the name across."""
        from ._frame import Series

        return Series(values, name=self._series.name)


class GroupByMixin[Answer]:
    """What `DataFrameGroupBy` and `SeriesGroupBy` share, which is all the state.

    `Answer` is what the reductions hand back. A frame's group by answers a frame
    and a column's answers a column, and the fifteen reductions are otherwise the
    same call written twice, so the shared half is written once against the
    parameter and each subclass says which it is. That is not decoration: without
    it every reduction on both classes is declared to answer one thing and
    returns something a type checker only knows as `Any`, which is the one shape
    of mistake a wrapper this thin exists to make impossible.

    A group by object holds a frame, some key column names and three flags, and
    it computes nothing until a reduction is asked for. That is pandas' own
    arrangement and it is the reason the two classes have so little in them: the
    fifteen reductions are one call each with a different word in it, and the
    word is the pandas method name, which is the same string the boundary reads.

    The keys are checked here rather than at the first reduction. pandas raises
    `KeyError` out of `df.groupby("nope")` and not out of the `.sum()` after it,
    and a program that catches the wrong line is a program whose error handling
    does not run. Checking early costs one pass over the column names.

    What is deliberately absent is any notion of the groups themselves. pandas
    can hand back `g.groups`, `g.indices` and `g.get_group(k)`, which are the
    grouping made visible, and firepanda computes the grouping inside the
    reduction and throws it away. Keeping it would mean the object holds an
    index per group whether or not anybody asks, which is the cost pandas pays
    and is the wrong default for a library whose claim is the other one. Those
    three names are absent rather than wrong.
    """

    __slots__ = ("_as_index", "_by", "_dropna", "_frame", "_sort")
    """The frame, the key names, and the three flags that survive to the call.
    Slotted for the reason `DataFrameMixin` gives. There is no `_inner`, because
    a group by object has no extension object of its own: it is a frame and a
    plan for what to do to it."""

    _frame: DataFrame
    _by: list[str]
    _as_index: bool
    _sort: bool
    _dropna: bool

    def __init__(self, frame: DataFrame, by: list[str], as_index: bool, sort: bool, dropna: bool):
        """Holds the frame and the plan. Not a public entry point.

        Args:
            frame: The frame being grouped.
            by: The key column names, already read out of whatever pandas shape
                the caller wrote them in.
            as_index: Whether the key becomes the row labels.
            sort: Whether the groups come out in key order.
            dropna: Whether a missing key is a group.
        """
        self._frame = frame
        self._by = by
        self._as_index = as_index
        self._sort = sort
        self._dropna = dropna

    @staticmethod
    def _keys(frame: DataFrame, by: Any, level: Any) -> list[str]:
        """Reads the key columns out of whatever pandas lets a caller write.

        pandas takes a name, a list of names, a column, a list of columns, a
        function, a dictionary and a level, and turns all seven into the same
        thing. Two of them are here: a name and a list of names. The rest are
        refused by shape with the reason, because each one is a different piece
        of work rather than a longer list.

        Args:
            frame: The frame, so the names can be checked against it.
            by: What the caller passed.
            level: The level, which there is no MultiIndex to have.

        Returns:
            The key column names.

        Raises:
            KeyError: If a name is not a column, which is what pandas raises.
            TypeError: If neither a key nor a level was given.
            InvalidArgumentError: If the list of keys is empty, which is a
                `ValueError` and is what pandas raises for it.
            NotImplementedError: If the shape is one of the five that are not
                written, or if a level was asked for.
        """
        _no_level(level)
        if by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        wanted = [by] if isinstance(by, str) else by
        if not isinstance(wanted, list) or not all(isinstance(name, str) for name in wanted):
            raise NotImplementedError(
                "by= has to be a column name or a list of them for now, because"
                " grouping by a column that is not in the frame, by a function or"
                " by a mapping all need somewhere to put the key that came from"
                " outside it"
            )
        if len(wanted) == 0:
            # `InvalidArgumentError` rather than a bare `ValueError`, which is
            # what pandas raises and what this used to raise, because `translate`
            # turns an untagged `ValueError` into a `RuntimeError` on the way out
            # of the generated method. The class here is a `ValueError` too, so a
            # caller who catches the pandas one still catches this.
            raise InvalidArgumentError("No group keys passed!")
        known = frame.columns
        for name in wanted:
            if name not in known:
                raise KeyError(name)
        return list(wanted)

    def _reduced(self, kind: str, param: float, columns: list[str] | None = None) -> DataFrame:
        """Runs one reduction over the groups and hands back the frame it makes.

        Both classes come through here and they differ in what they do with the
        answer, which is the whole of the difference between them.

        Args:
            kind: The reduction, as pandas spells the method.
            param: The delta degrees of freedom or the quantile, and zero for
                the rest.
            columns: The columns to reduce, or None for every one that is not a
                key. A `SeriesGroupBy` names one.

        Returns:
            The frame the core produced, one row per group.
        """
        from ._frame import DataFrame

        try:
            source = self._frame._inner
            if columns is not None:
                source = source.select(self._by + columns)
            return DataFrame._wrap(
                source.group_agg(self._by, kind, param, self._dropna, self._sort, self._as_index)
            )
        except Exception as error:
            raise translate(error) from None

    def _reduce(
        self,
        kind: str,
        param: float = 0.0,
        numeric_only: bool = False,
        skipna: bool = True,
        min_count: int | None = None,
        engine: Any = None,
        engine_kwargs: Any = None,
    ) -> Answer:
        """Runs one reduction over the groups, after refusing what is declared.

        Every one of the fifteen comes through here and the five arguments below
        are the ones pandas puts on their signatures and firepanda does not
        honour. They are declared rather than left out because the parity test
        compares the whole parameter list against a running pandas, and they
        raise rather than being ignored for the reason `_held_at` gives.

        The shaping is what the two classes differ in, so it is one call at the
        end and is written in each of them.

        Args:
            kind: The reduction, as pandas spells the method.
            param: The delta degrees of freedom or the quantile, and zero for
                the rest.
            numeric_only: Declared and held at False.
            skipna: Declared and held at True.
            min_count: Declared and held at its default, which is zero for `sum`
                and minus one for the four that pick a value rather than combine
                them. None for the reductions that do not have it.
            engine: Declared and refused.
            engine_kwargs: Declared and refused.

        Returns:
            The frame or the series pandas answers.
        """
        _held_at(
            "numeric_only",
            numeric_only,
            False,
            "dropping the columns a reduction cannot read is a decision about"
            " which columns come back, and firepanda reduces the ones it was"
            " given or says which one it could not",
        )
        _held_at(
            "skipna",
            skipna,
            True,
            "a group's answer is computed from the values that are there, and"
            " taking one missing value as a reason to answer nothing at all is a"
            " second pass the kernels do not make",
        )
        if min_count is not None:
            # pandas defaults this to zero for `sum` and to minus one for the
            # four that pick a value rather than combining them, so the value
            # that means "nobody asked for anything" depends on the reduction.
            _held_at(
                "min_count",
                min_count,
                0 if kind == "sum" else -1,
                "answering nothing for a group that is too small is a check on"
                " the count after the reduction, and the count is not kept",
            )
        _refuse(
            "engine",
            engine,
            "there is one implementation and it is the compiled one, so there is"
            " nothing here for this to choose between",
        )
        _refuse(
            "engine_kwargs",
            engine_kwargs,
            "there is nothing to configure while there is nothing to choose",
        )
        return self._shape(kind, param)

    def _spread(
        self,
        kind: str,
        ddof: int,
        numeric_only: bool,
        skipna: bool,
        engine: Any,
        engine_kwargs: Any,
    ) -> Answer:
        """Runs `std` or `var`, which carry a delta degrees of freedom.

        A door of its own only because the generated body has to fit on one
        line, and `float(ddof)` written there rather than here pushed the two
        widest of the fifteen past the line limit. The conversion is the whole
        of what it adds: the kinds store the parameter as a float because it is
        a number in a formula rather than a length.

        Args:
            kind: `std` or `var`.
            ddof: The delta degrees of freedom.
            numeric_only: Declared and held at False.
            skipna: Declared and held at True.
            engine: Declared and refused.
            engine_kwargs: Declared and refused.

        Returns:
            The frame or the series pandas answers.
        """
        return self._reduce(
            kind,
            float(ddof),
            numeric_only,
            skipna,
            None,
            engine,
            engine_kwargs,
        )

    def _nunique(self, dropna: bool) -> Answer:
        """Counts the distinct values in each group.

        Its own entry point because its `dropna` is not the group by's `dropna`
        and the two are easy to read as one. The group by's says whether a
        missing key is a group. This one says whether a missing value counts as
        one of the distinct values inside a group, and it is held at True, which
        is pandas' default and the kernel's behaviour.

        Args:
            dropna: Declared and held at True.

        Returns:
            The frame or the series pandas answers.
        """
        _held_at(
            "dropna",
            dropna,
            True,
            "counting a missing value as one of the distinct values in a group"
            " is a second thing for the kernel to carry and nothing asks for it",
        )
        return self._shape("nunique", 0.0)

    def _quantile(self, q: Any, interpolation: str, numeric_only: bool) -> Answer:
        """The value at one quantile within each group.

        `_quantile_wanted` is the same reader the whole column reductions use,
        so a list of quantiles is refused with the same sentence in both places
        and the four interpolation rules that are not linear are refused once.

        Args:
            q: The quantile, between zero and one.
            interpolation: How to land between two values.
            numeric_only: Declared and held at False.

        Returns:
            The frame or the series pandas answers.
        """
        return self._reduce(
            "quantile", _quantile_wanted(q, interpolation), numeric_only=numeric_only
        )

    def _shape(self, kind: str, param: float) -> Answer:
        """Runs the reduction and puts the answer in the shape pandas gives.

        Overridden in both subclasses and never called on this one.

        Args:
            kind: The reduction, as pandas spells the method.
            param: The delta degrees of freedom or the quantile.

        Returns:
            The frame or the series pandas answers.
        """
        raise NotImplementedError(kind)


class DataFrameGroupByMixin(GroupByMixin["DataFrame"]):
    """The hand written half of `DataFrameGroupBy`.

    A reduction here answers a frame, except `size`, which counts rows rather
    than reducing a column and therefore answers one number per group. pandas
    makes that a Series when the key is in the index and a two column frame when
    it is not, and both of those are here, because the shape is the answer as
    much as the numbers are.
    """

    __slots__ = ()

    def __getitem__(self, key: Any) -> DataFrameGroupBy | SeriesGroupBy:
        """Narrows the group by to one column or to several.

        `df.groupby("k")["v"]` is a `SeriesGroupBy` and `df.groupby("k")[["v"]]`
        is a `DataFrameGroupBy` over fewer columns, which is the same split
        `df[...]` makes and is why this is written rather than generated.

        Args:
            key: A column name, or a list of them.

        Returns:
            A `SeriesGroupBy` for a name and a `DataFrameGroupBy` for a list.

        Raises:
            KeyError: If a name is not a column.
        """
        from ._frame import DataFrame, DataFrameGroupBy, SeriesGroupBy

        names = [key] if isinstance(key, str) else list(key)
        for name in names:
            if name not in self._frame.columns:
                raise KeyError(name)
        narrowed = DataFrame._wrap(self._frame._inner.select(self._by + names))
        if isinstance(key, str):
            return SeriesGroupBy(narrowed, self._by, self._as_index, self._sort, self._dropna, key)
        return DataFrameGroupBy(narrowed, self._by, self._as_index, self._sort, self._dropna)

    def _shape(self, kind: str, param: float) -> DataFrame:
        """One reduction over every column that is not a key.

        Args:
            kind: The reduction, as pandas spells the method.
            param: The delta degrees of freedom or the quantile.

        Returns:
            The frame of one row per group.
        """
        return self._reduced(kind, param)

    def _size(self) -> DataFrame | Series:
        """Counts the rows in each group, which is the one reduction with two shapes.

        A door of its own rather than a branch in `_shape`, because it is the one
        of the fifteen that does not answer what the other fourteen answer.
        pandas makes it a series when the key is in the index, since the answer is
        one column either way, and leaves it a two column frame when the key is
        not, and the shape is as much the answer as the numbers are.

        Returns:
            A series of counts, or a frame of the keys and the counts.
        """
        out = self._reduced("size", 0.0)
        if not self._as_index:
            return out
        # The name goes because a pandas `size` has none, and the frame's one
        # column is called `size` here only because a column has to be called
        # something.
        return _relabelled(out, out.columns[0], "")


class SeriesGroupByMixin(GroupByMixin["DataFrame | Series"]):
    """The hand written half of `SeriesGroupBy`.

    The same reductions over one column, answering a series rather than a frame.
    The extra piece of state is the column's name, which the answer carries and
    the frame underneath does not: after `df.groupby("k")["v"].sum()` the series
    is called `v`, and by then the frame it came out of has one column called
    `v` and one called `k` and no way to say which of them the caller asked for.
    """

    __slots__ = ("_column",)
    """The column the reductions run over."""

    _column: str

    def __init__(
        self,
        frame: DataFrame,
        by: list[str],
        as_index: bool,
        sort: bool,
        dropna: bool,
        column: str,
    ):
        """Holds the frame, the plan and the column. Not a public entry point.

        Args:
            frame: The frame, already narrowed to the keys and the column.
            by: The key column names.
            as_index: Whether the key becomes the row labels.
            sort: Whether the groups come out in key order.
            dropna: Whether a missing key is a group.
            column: The column the reductions run over.
        """
        super().__init__(frame, by, as_index, sort, dropna)
        self._column = column

    def _shape(self, kind: str, param: float) -> DataFrame | Series:
        """One reduction over the one column.

        Args:
            kind: The reduction, as pandas spells the method.
            param: The delta degrees of freedom or the quantile.

        Returns:
            A series named after the column, or a frame when the key was asked
            for as a column rather than as labels.
        """
        out = self._reduced(kind, param, [self._column])
        if not self._as_index:
            return out
        # `size` answers a column called `size` rather than one called after the
        # column it counted, because it did not read that column. Either way
        # there is exactly one column left once the keys have gone into the
        # labels, so the answer is the column that is there.
        return _relabelled(out, out.columns[-1], "" if kind == "size" else self._column)


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

        A series and another index are both taken as well as a plain sequence,
        and neither goes through the reader that a sequence goes through. Both
        already hold a column of a known type, so reading them back out into
        Python values and inferring a type from those again would be slower and
        would be a second chance to land somewhere else, which is the same
        argument `DatetimeIndex` makes for its own shortcut.

        A name that is not given comes off whatever the labels came from, which
        is pandas' rule and is the reason this is not one line. An index built
        out of a named series is named after it, and a name written in the call
        wins over one the data was carrying.
        """
        _refuse("dtype", dtype, "casting on the way in needs the cast machinery")
        _refuse("copy", copy, "there is exactly one behaviour and it always copies")
        if not tupleize_cols:
            raise NotImplementedError(
                "tupleize_cols=False is not supported yet, because there is no"
                " MultiIndex for it to turn off"
            )
        label = _label_of(data) if name is None else str(name)
        try:
            if isinstance(data, IndexMixin):
                self._inner = data._inner.renamed(label)
            elif isinstance(data, SeriesMixin):
                self._inner = data._inner.to_index(label)
            else:
                self._inner = _firepanda.Index(data, label)
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

    def reindex(
        self,
        target: Any,
        method: Any = None,
        level: Any = None,
        limit: Any = None,
        tolerance: Any = None,
    ) -> tuple[Index, list[int] | None]:
        """The labels asked for, and where each of them sits in this index.

        The strangest member on the type, because an index carries no values of
        its own: reindexing one moves nothing and the new index is the target,
        so the half a caller wanted is the second one, which is the lookup it
        can gather something else with.

        The second half is `None` when the target holds the labels this index
        already holds. pandas says so that way rather than with the range the
        gather would have been, and it means the caller can skip the gather.

        The level name follows the target when the target is an index and this
        index's own when it is a list of labels, which reads backwards until you
        notice it is one rule: the name belongs to whoever was in a position to
        say what it was.

        Written by hand rather than generated because the answer is a pair, and
        pandas hands the positions back as a numpy array where this hands back a
        list, which is the same difference `get_indexer` above already has.

        Args:
            target: The labels the result should carry, as an index or as a
                sequence of them.
            method: Refused, for the reason `get_indexer` gives.
            level: Ignored, since a flat index has exactly the one level.
            limit: Refused, since it only means something with `method`.
            tolerance: Refused, for the same reason.

        Returns:
            The new index and the positions, or the new index and `None`.

        Raises:
            ValueError: If this index holds a label more than once.
        """
        from ._frame import Index

        _refuse("method", method, "filling a missing label from a neighbour is not written")
        _refuse("limit", limit, "there is no filling for it to limit")
        _refuse("tolerance", tolerance, "there is no filling for it to bound")
        wanted = target._inner if isinstance(target, IndexMixin) else target
        try:
            answer: Any = self._inner.reindex(wanted)
        except Exception as error:
            raise translate(error) from None
        made, positions = answer
        return Index._wrap(made), None if positions is None else list(positions)

    def searchsorted(self, value: Any, side: str = "left", sorter: Any = None) -> Any:
        """Where a label would have to go for the labels to stay in order.

        One label gives back one position and a sequence of labels gives back a
        list of them, which is pandas' rule and is the reason this is written by
        hand: the shape of the answer is decided by the shape of the argument.

        Nothing here checks that the index is sorted, and that is deliberate.
        numpy does not check, pandas does not check, and an unsorted index gets
        an answer that is meaningless in exactly the way it is in both of them.
        Checking would cost a pass over the labels on every call to protect
        against a mistake the callers of this method do not make.
        """
        _refuse("sorter", sorter, "sorting the index on the way past needs the sort to be carried")
        if side not in ("left", "right"):
            raise InvalidArgumentError(
                f"firepanda:value: Invalid side: {side}. Side must be one of 'left', 'right'"
            )
        try:
            if isinstance(value, IndexMixin):
                value = value._inner.to_list()
            if isinstance(value, (list, tuple)):
                return [self._inner.searchsorted(one, side) for one in value]
            return self._inner.searchsorted(value, side)
        except Exception as error:
            raise translate(error) from None

    def isin(self, values: Any, level: Any = None) -> Any:
        """Whether each label is one of a set of values.

        pandas gives back a numpy array of bools and this gives back a list of
        them, which is the divergence `values` and `__eq__` already have and
        which document 21 records once for all three.

        A set the column's type cannot hold falls back to comparing in Python,
        and that is not a shortcut. pandas compares by value rather than by type,
        so `pd.Index([1, 2]).isin([1.0])` finds the one and `isin(["a", 2])`
        finds the two without complaining about the string. The kernel looks up
        one type in a set of one type, which is the fast answer and is the right
        one whenever it applies, so it is tried first and the slow path exists
        for the calls it refuses. An empty set never reaches it at all, because a
        list with nothing in it has no type to build a column from.
        """
        _no_level(level)
        if isinstance(values, IndexMixin):
            values = values._inner.to_list()
        wanted = list(values)
        if wanted:
            try:
                return list(self._inner.isin(wanted))
            except Exception:
                pass
        try:
            mine = self._inner.to_list()
        except Exception as error:
            raise translate(error) from None
        return [any(label == one for one in wanted) for label in mine]

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

    def slice_locs(self, start: Any = None, end: Any = None, step: Any = None) -> tuple[int, int]:
        """The half open row range a pair of labels describes, both ends inclusive.

        A forward step is the plain question and the Mojo answers it. A negative
        step is a different question wearing the same name: the caller means to
        read the range backwards, so the labels arrive in the order they will be
        walked in rather than in index order, and the pair that comes back is
        the pair a backward Python slice wants.

        The way pandas gets there is to swap the two labels, ask the forward
        question, and then shift both answers down by one, which turns a half
        open range that excludes its right end into one that excludes its left.
        A bound that lands on -1 is shifted down by the length of the index as
        well, because -1 in a Python slice means the last row rather than the
        row before the first and a bound that means nothing before the start
        has to say so in a way a slice will read.
        """
        forward = step is None or int(step) >= 0
        try:
            if forward:
                first, last = self._inner.slice_locs(start, end)
                return (first, last)
            height = self._inner.length()
            first = 0 if end is None else self._inner.get_slice_bound(end, "left")
            last = height if start is None else self._inner.get_slice_bound(start, "right")
        except Exception as error:
            raise translate(error) from None
        first, last = last - 1, first - 1
        if last == -1:
            last -= height
        if first == -1:
            first -= height
        return (first, last)

    def take(
        self,
        indices: Any,
        axis: Any = 0,
        allow_fill: bool = True,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Index:
        """The labels at a set of positions, in the order given.

        The two arguments that look like one question are `allow_fill` and
        `fill_value`, and pandas reads them together: filling only happens when
        both `allow_fill` is on and a `fill_value` was actually passed, so the
        default pair means no filling at all and a negative position counts back
        from the end the way it does everywhere else in Python. With filling on,
        -1 means a row that is not there and anything below -1 is a mistake
        rather than a position.

        The `fill_value` itself is read for whether it is there and not for what
        it is, which is pandas' own behaviour and is worth stating because it
        surprises people: the label that lands in the gap is a missing label
        whatever value was named. `axis` is accepted and ignored, as pandas
        accepts and ignores it, since an index has one axis to take along.

        Args:
            indices: The positions.
            axis: Accepted and ignored.
            allow_fill: Whether -1 is allowed to mean a missing label.
            fill_value: Whether to fill at all. Its value is not used.
            **kwargs: Accepted and ignored, as in pandas.

        Returns:
            An index of the labels at those positions.
        """
        from ._frame import Index

        wanted = [int(i) for i in indices]
        if allow_fill and fill_value is not None:
            if any(i < -1 for i in wanted):
                raise InvalidArgumentError(
                    "firepanda:value: when allow_fill=True and fill_value is not"
                    " None, all indices must be >= -1"
                )
        else:
            height = self._inner.length()
            wanted = [i + height if i < 0 else i for i in wanted]
            for was, now in zip(indices, wanted, strict=True):
                if now < 0:
                    raise OutOfBoundsError(
                        f"firepanda:position: index {int(was)} is out of bounds"
                        f" for axis 0 with size {height}"
                    )
        try:
            return Index._wrap(self._inner.take(wanted))
        except Exception as error:
            raise translate(error) from None

    def unique(self, level: Any = None) -> Index:
        """The index with each label kept once, in first seen order.

        `level` is here because a MultiIndex has levels and this shares the
        signature with it. On a flat index the only level is the one that is
        there, so `None`, `0` and `-1` all name it and the index's own name
        names it too. Anything else is an error rather than a refusal, because
        the caller asked for a level that does not exist rather than for a
        feature that is not written.
        """
        from ._frame import Index

        self._only_level(level)
        try:
            return Index._wrap(self._inner.unique())
        except Exception as error:
            raise translate(error) from None

    def to_series(self, index: Any = None, name: Any = None) -> Series:
        """The labels as a column, which carries the labels twice.

        The one door between the two types and the reason several of the members
        below are a line each. A column has the reductions, a column has the
        transforms, and an index that can turn into one gets all of them without
        a second copy of any of them living over here.

        `index` names the labels the answer carries and defaults to the ones it
        was read from, which is what makes the labels come back twice and is
        what pandas does. `name` names the column and defaults to the index's
        own name, which is the empty string when the index has none, where
        pandas leaves the series unnamed.
        """
        from ._frame import _index_to_series

        labels = None if index is None else _unwrap(index, "index")
        return _index_to_series(self._inner, labels, None if name is None else str(name))

    def isna(self) -> Any:
        """Whether each label is missing.

        pandas gives back a numpy array of bools and this gives back a list of
        them, which is the divergence `values`, `__eq__` and `isin` already have
        and which document 21 records once for all of them.

        Most indexes answer a list of `False`, because a range has no missing
        label and neither has an index read from a list with nothing missing in
        it. `take` is where a missing label comes from.
        """
        return self.to_series().isna().tolist()

    def isnull(self) -> Any:
        """Whether each label is missing. The older spelling of `isna`."""
        return self.isna()

    def notna(self) -> Any:
        """Whether each label is present, which is `isna` turned over."""
        return self.to_series().notna().tolist()

    def notnull(self) -> Any:
        """Whether each label is present. The older spelling of `notna`."""
        return self.notna()

    def dropna(self, how: str = "any") -> Index:
        """The index with the missing labels taken out.

        `how` is `any` or `all` and on a flat index the two mean the same thing,
        since a label is one value and there is nothing for the two of them to
        disagree about. The parameter is here for the reason `level` is here on
        `unique`, which is that a MultiIndex row holds several values and one
        signature covers both. Any other word is an error.

        What comes back is the class the index already was, so dropping a
        missing instant leaves a `DatetimeIndex` rather than a plain index. The
        labels that are left are the labels that were there, so the type they
        had is the type they keep.
        """
        if how not in ("any", "all"):
            raise InvalidArgumentError(f"firepanda:value: invalid how option: {how}")
        column = self.to_series().dropna()
        # The class the index already was rather than `Index`, because a
        # `DatetimeIndex` with a missing instant dropped is still a set of
        # instants. The mixin cannot see `_wrap`, which the generated half
        # writes, so the class goes through a name the checker leaves alone.
        made: Any = type(self)
        try:
            kept: Index = made._wrap(column._inner.to_index(self._inner.label()))
        except Exception as error:
            raise translate(error) from None
        return kept

    def min(self, axis: Any = None, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        """The smallest label.

        The column's reduction reached through `to_series`, which is the whole
        method. `axis` and the two catch alls after it are numpy's, since numpy
        calls these on an index and pandas takes what it passes.
        """
        self._numpy_only(axis, args, kwargs)
        return self.to_series().min(skipna=skipna)

    def max(self, axis: Any = None, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        """The largest label, which is `min` the other way round."""
        self._numpy_only(axis, args, kwargs)
        return self.to_series().max(skipna=skipna)

    def _numpy_only(self, axis: Any, args: Any, kwargs: Any) -> None:
        """Holds the numpy compatibility arguments of `min` and `max` at rest.

        numpy calls `min` and `max` on whatever it is handed with an axis and a
        few keywords of its own, so pandas takes them and checks that they say
        nothing. This checks the axis by pandas' rule, which lets `None`, `0`
        and `-1` through because an index has one dimension, and refuses the
        rest of them outright rather than dropping them, since a caller who
        passed `out=` meant something by it.
        """
        if axis is not None and (axis >= 1 or axis < -1):
            raise InvalidArgumentError(
                "firepanda:value: `axis` must be fewer than the number of dimensions (1)"
            )
        if args or kwargs:
            raise UnsupportedError(
                "the numpy compatibility arguments of min and max are not taken,"
                " because the only value any of them can hold that means"
                " anything here is the default it already has"
            )

    def nunique(self, dropna: bool = True) -> int:
        """How many distinct labels there are.

        The column's count, which skips the missing labels, plus one when the
        caller asked for a missing label to count as a distinct one and there
        was one. The column refuses `dropna=False` outright, because its kernel
        drops the missing values before it counts and cannot tell afterwards
        whether it saw any. An index can tell, since it is asked how many
        labels are missing often enough to keep the answer, so the two lines
        here are the whole of the difference.
        """
        answer = int(self.to_series().nunique())
        if dropna or self._inner.null_count() == 0:
            return answer
        return answer + 1

    def _only_level(self, level: Any) -> None:
        """Holds that `level` names the one level a flat index has.

        Written once because `unique` is the first of several members that take
        a level and mean nothing by it until there is a MultiIndex. The two
        errors are pandas' own, an `IndexError` for a number that is out of
        range and a `KeyError` for a name that is not this index's name.
        """
        if level is None:
            return
        if isinstance(level, int) and not isinstance(level, bool):
            if level in (0, -1):
                return
            if level < 0:
                raise IndexError(
                    f"Too many levels: Index has only 1 level, {level} is not a valid level number"
                )
            raise IndexError(f"Too many levels: Index has only 1 level, not {level + 1}")
        if level != self._inner.label():
            raise KeyError(
                f"Requested level ({level}) does not match index name ({self._inner.label()})"
            )

    def rename(self, name: Any, *, inplace: bool = False) -> Index | None:
        """The index under a different level name.

        `inplace` is the one place an index is mutable here, and it is mutable
        in pandas too, because a level name is not a label and changing it does
        not change what the index holds. The index underneath is still rebuilt
        rather than edited, since a name lives in Mojo beside the labels, and
        what `inplace` changes is which object the caller is left holding.
        """
        from ._frame import Index

        try:
            renamed = self._inner.renamed(None if name is None else str(name))
        except Exception as error:
            raise translate(error) from None
        if not inplace:
            return Index._wrap(renamed)
        self._inner = renamed
        return None

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


def _label_of(data: Any) -> str | None:
    """The name a set of labels arrives already carrying, or None for neither.

    pandas takes the name off the data when the call did not write one, so an
    index built out of a named series is named after the series and an index
    built out of another index keeps the name that one had. Anything else, a
    list or a tuple or a range, has no name to take and answers None.

    A series here is named by a string and the empty string is what unnamed
    looks like on one, so a series named that way gives an unnamed index rather
    than a level called nothing. That is the same rule read backwards that
    `Index.to_series` follows going the other way.

    Args:
        data: Whatever the constructor was handed.

    Returns:
        The name, or None.
    """
    if isinstance(data, IndexMixin):
        return data._inner.label()
    if isinstance(data, SeriesMixin):
        return data._inner.label() or None
    return None


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


def to_datetime(
    arg: Any,
    errors: str = "raise",
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: Any = NO_DEFAULT,
    unit: str | None = None,
    origin: Any = "unix",
    cache: bool = True,
) -> Any:
    """Reads text or whole numbers as instants, which is `pandas.to_datetime`.

    Hand written rather than generated for the reason the top of this file
    gives: what it does depends on its arguments. Ten of them are declared,
    four are implemented, one is accepted and has no effect, and five are
    refused by name, which is the pattern `_refuse` exists for.

    What it answers is a `Series` and pandas answers a `DatetimeIndex` when it
    is handed a list. That is the one difference a caller will meet on the
    first line they write, and it is not hidden: firepanda's `Index` is a
    labels object with none of the calendar members a `DatetimeIndex` carries,
    so answering one would be a name that resolves and then has nothing on it,
    which document 07 argues is worse than a name that resolves to something
    honest. See #354.

    The format is worked out from the first row that is not missing, and only
    ISO 8601 is recognised. pandas guesses more than that, including
    `01/02/2026`, and decides for itself which of the two numbers is the month.
    A wrong guess there is a column of instants that are wrong by up to eleven
    months and that nothing anywhere reports, so firepanda refuses the shapes
    it does not recognise and names the value in the message. Passing `format`
    reads anything, including the shapes the guesser will not touch.

    Args:
        arg: The values. A firepanda series, or anything a series is built
            from, such as a list of strings or a list of whole numbers.
        errors: `raise` to stop on the first row that will not read, or
            `coerce` to turn that row into a missing one.
        dayfirst: Refused. Only ISO 8601 is guessed and it has one order.
        yearfirst: Refused, for the same reason.
        utc: Whether to read every row against UTC, which is the only way a
            column carrying more than one offset can be read at all.
        format: The format the text is written in, or None to work it out.
        exact: Refused. It is a question about a regular expression search that
            this parser does not do.
        unit: What whole numbers are counts of, as one of `s`, `ms`, `us` and
            `ns`. Ignored for text, which pandas ignores it for too.
        origin: Refused at anything other than `unix`.
        cache: Accepted and has no effect. pandas caches repeated values to go
            faster and the answer is the same either way, so honouring the
            parameter means not changing the answer.

    Returns:
        A series of instants, null where the input was null and, under
        `errors="coerce"`, wherever a row would not read.

    Raises:
        NotImplementedError: For the five refused arguments and for a format
            firepanda's guesser does not recognise.
        ValueError: For a row that does not match the format, for a column
            carrying more than one offset with no `utc`, and for an `errors`
            that is neither of the two words.
    """
    from ._frame import Series

    _held_at("dayfirst", dayfirst, False, "firepanda guesses ISO 8601 and nothing else")
    _held_at("yearfirst", yearfirst, False, "firepanda guesses ISO 8601 and nothing else")
    _held_at("origin", origin, "unix", "an epoch other than 1970 has to move every value")
    if exact is not NO_DEFAULT:
        raise NotImplementedError(
            "exact= is not supported yet, because it asks whether the format may match"
            " part of the value, and this parser reads the whole of it or none of it"
        )
    if format in ("mixed", "ISO8601"):
        raise NotImplementedError(
            f"format={format!r} is not supported yet, because it asks for the format to"
            " be worked out per row, and firepanda works one out from the first row and"
            " holds every other row to it"
        )
    if errors not in ("raise", "coerce"):
        raise ValueError(f"errors must be one of 'raise' or 'coerce', not {errors!r}")

    column = arg if isinstance(arg, SeriesMixin) else Series(arg)
    try:
        return Series._wrap(
            column._inner.to_datetime(
                "" if format is None else format,
                "ns" if unit is None else unit,
                errors == "coerce",
                utc,
            )
        )
    except Exception as error:
        raise translate(error) from None
