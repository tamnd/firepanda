"""The `str` accessor, which is the largest namespace pandas has.

Same argument as `temporal.mojo`, arrived at for a fourth time. pandas puts
fifty seven names on `s.str` and every one of them is a column in and a column
out, so the crossing is a word rather than fifty seven methods, and the Python
layer is where each word has a name a caller was invited to type.

### Why there are three doors and not one

The rule `transform.mojo` set is that the shape of the answer picks the door. On
this accessor there are three shapes: `startswith` answers a mask, `len` and
`find` answer a number, and everything else answers text. A name sent through
the wrong door is refused rather than being quietly served, which is what makes
these worth being three functions instead of one with a branch at the end.

Argument shape does not pick a door and deliberately does not, because it varies
almost per method and grouping by it would give a door per method. So the widest
of the three carries the arguments the others do not need and hands them along:
one string, two positions that are allowed to be absent, and a step.

### Why a position crosses as an absence rather than as a number

`None` is a value in a Python slice and not a missing argument.
`s.str.slice(None, None, -1)` reverses every row and `s.str.slice(0, 0, -1)`
empties them, so an absent bound has to stay absent all the way to the kernel
that resolves it against the row's own length. Every row has a different length,
which is why neither side of the crossing can turn it into a number first.
`maybe_whole` in `args.mojo` is that, and it is the only reader there that
answers an `Optional`.

### What is not here

`get` rides in the same door as `slice` and takes its index in the `start`
position, because an index is a position and that is where positions go. It is
not a slice of one character: a slice past the end of a short row is the empty
string and `get` past the end is a null, which is pandas and is worth the
separate kernel it takes.

The forty odd names this file does not spell do not resolve at all rather than
resolving and refusing, for the reason document 07 gives: an absent name reads as
unimplemented on the board and a refusing one reads as a failure, and the second
is a lie about a method nobody has written yet.
"""

from firepanda.frame.series import Series
from firepanda.kernel.chars import character_count
from firepanda.py.errors import DTYPE, VALUE, tagged


def _text_name(name: String) raises -> String:
    """Checks that a word names one of the methods that answers text.

    Args:
        name: The word.

    Returns:
        The same word.

    Raises:
        Error: Tagged `value` if it is not one of them.
    """
    if (
        name == "slice"
        or name == "slice_replace"
        or name == "get"
        or name == "removeprefix"
        or name == "removesuffix"
        or name == "strip"
        or name == "lstrip"
        or name == "rstrip"
        or name == "strip_chars"
        or name == "lstrip_chars"
        or name == "rstrip_chars"
        or name == "pad_left"
        or name == "pad_right"
        or name == "pad_both"
        or name == "zfill"
        or name == "repeat"
    ):
        return name
    raise tagged(VALUE, String("str: ", name, " does not answer a text column"))


def _flag_name(name: String) raises -> String:
    """Checks that a word names one of the methods that answers a mask.

    Args:
        name: The word.

    Returns:
        The same word.

    Raises:
        Error: Tagged `value` if it is not one of them.
    """
    if name == "startswith" or name == "endswith":
        return name
    raise tagged(VALUE, String("str: ", name, " does not answer a mask"))


def _number_name(name: String) raises -> String:
    """Checks that a word names one of the methods that answers a number.

    Args:
        name: The word.

    Returns:
        The same word.

    Raises:
        Error: Tagged `value` if it is not one of them.
    """
    if name == "len" or name == "find" or name == "rfind":
        return name
    raise tagged(VALUE, String("str: ", name, " does not answer a number"))


def _text_column(column: Series) raises:
    """Refuses a column that is not text, before any method looks at it.

    pandas raises here rather than at the method, with a message about the
    accessor and not about whichever name was reached for, and it is the better
    message: a caller who typed `s.str.upper()` on a column of numbers has a
    problem with the column and not with `upper`.

    Args:
        column: The column.

    Raises:
        Error: Tagged `value` if the column does not hold text.
    """
    if column.chars_is_text():
        return
    raise tagged(
        VALUE,
        String(
            (
                "str: can only use the str accessor with string values, and"
                " this column is "
            ),
            column.values.type,
        ),
    )


def _whole(value: Optional[Int], name: String) raises -> Int:
    """Reads a count that the method cannot do without.

    Six of the names in the text door need a number rather than a position, and
    they borrow the `start` slot to carry it so that the door keeps the six
    arguments it already had. A slot that is allowed to be absent is the wrong
    shape for a count, so the absence is refused here instead of being read as a
    zero, which would have made `s.str.zfill()` quietly answer the column back.

    Args:
        value: What came across in the position slot.
        name: The word to put in the message, as pandas spells the argument.

    Returns:
        The number.

    Raises:
        Error: Tagged `value` if it is absent.
    """
    if value:
        return value.value()
    raise tagged(VALUE, String("str: ", name, " is required"))


def _one_character(fill: String) raises -> String:
    """Checks that a fill character is one character and not several.

    Args:
        fill: What the caller passed.

    Returns:
        The same string.

    Raises:
        Error: Tagged `dtype`, which reaches Python as the `TypeError` pandas
            raises here, if it is not exactly one character. pandas counts
            characters and not bytes, so a single accented letter is fine and
            two ASCII ones are not.
    """
    if character_count(fill.as_bytes()) == 1:
        return fill
    raise tagged(
        DTYPE,
        String(
            "str: fillchar must be a character, not a string of ",
            character_count(fill.as_bytes()),
        ),
    )


def text(
    column: Series,
    kind: String,
    arg: String,
    start: Optional[Int],
    stop: Optional[Int],
    step: Int,
) raises -> Series:
    """Runs one of the methods that answers text, and hands back a column.

    Args:
        column: The column to read.
        kind: The method, as pandas spells it.
        arg: The prefix, suffix or replacement, the characters to strip, or the
            character to pad with, and the empty string for the ones that take
            none.
        start: The first position, where the method has one, the index for
            `get`, and the width or the repeat count for the ones that take a
            number rather than a position.
        stop: The position to stop before, where the method has one.
        step: How far to move between characters, for `slice` alone.

    Returns:
        A new column, as tall as the one it read.

    Raises:
        Error: Tagged `value` if the name is not one that answers text or the
            column is not text, and whatever the kernel raises otherwise.
    """
    _text_column(column)
    var wanted = _text_name(kind)
    if wanted == "slice":
        # The kernel refuses this as well, since it has to and since it is
        # reachable from the Mojo API too. It is refused again here so that the
        # error carries a kind and reaches Python as the `ValueError` pandas
        # raises, rather than as the untagged error a bare kernel raise becomes.
        if step == 0:
            raise tagged(VALUE, String("slice step cannot be zero"))
        return column.chars_slice(start, stop, step)
    if wanted == "slice_replace":
        return column.chars_slice_replace(start, stop, arg)
    if wanted == "get":
        return column.chars_get(start.value() if start else 0)
    if wanted == "removeprefix":
        return column.chars_remove_prefix(arg)
    if wanted == "removesuffix":
        return column.chars_remove_suffix(arg)
    if wanted == "strip":
        return column.chars_strip("", False, True, True)
    if wanted == "lstrip":
        return column.chars_strip("", False, True, False)
    if wanted == "rstrip":
        return column.chars_strip("", False, False, True)
    if wanted == "strip_chars":
        return column.chars_strip(arg, True, True, True)
    if wanted == "lstrip_chars":
        return column.chars_strip(arg, True, True, False)
    if wanted == "rstrip_chars":
        return column.chars_strip(arg, True, False, True)
    if wanted == "pad_left":
        return column.chars_pad(
            _whole(start, "width"), _one_character(arg), True, False
        )
    if wanted == "pad_right":
        return column.chars_pad(
            _whole(start, "width"), _one_character(arg), False, True
        )
    if wanted == "pad_both":
        return column.chars_pad(
            _whole(start, "width"), _one_character(arg), True, True
        )
    if wanted == "zfill":
        return column.chars_zfill(_whole(start, "width"))
    return column.chars_repeat(_whole(start, "repeats"))


def flag(column: Series, kind: String, arg: String) raises -> Series:
    """Runs one of the methods that answers a mask, and hands back a column.

    Args:
        column: The column to read.
        kind: The method, as pandas spells it.
        arg: The prefix or the suffix.

    Returns:
        A bool column, as tall as the one it read.

    Raises:
        Error: Tagged `value` if the name is not one that answers a mask or the
            column is not text.
    """
    _text_column(column)
    var wanted = _flag_name(kind)
    if wanted == "startswith":
        return column.chars_starts_with(arg)
    return column.chars_ends_with(arg)


def number(
    column: Series,
    kind: String,
    arg: String,
    start: Optional[Int],
    stop: Optional[Int],
) raises -> Series:
    """Runs one of the methods that answers a number, and hands back a column.

    Args:
        column: The column to read.
        kind: The method, as pandas spells it.
        arg: The substring to look for, and the empty string for `len`.
        start: The first position a match may start at, where the method takes
            one.
        stop: The position to stop searching before, where the method takes one.

    Returns:
        An int64 column, as tall as the one it read.

    Raises:
        Error: Tagged `value` if the name is not one that answers a number or
            the column is not text.
    """
    _text_column(column)
    var wanted = _number_name(kind)
    if wanted == "len":
        return column.chars_length()
    return column.chars_find(arg, start, stop, wanted == "rfind")
