"""The `str` accessor, which is the largest namespace pandas has.

Same argument as `temporal.mojo`, arrived at for a fourth time. pandas puts
fifty seven names on `s.str` and every one of them is a column in and a column
out, so the crossing is a word rather than fifty seven methods, and the Python
layer is where each word has a name a caller was invited to type.

### Why there are three doors and not one

The rule `transform.mojo` set is that the shape of the answer picks the door. On
this accessor there are three shapes: `startswith`, the questions about case and
the questions about a pattern answer a mask, `len`, `find` and `count` answer a
number, and everything else answers text. A name sent through the wrong door is
refused rather than being quietly served, which is what makes these worth being
three functions instead of one with a branch at the end.

Argument shape does not pick a door and deliberately does not, because it varies
almost per method and grouping by it would give a door per method. So the widest
of the three carries the arguments the others do not need and hands them along:
two strings, two positions that are allowed to be absent, and a step.

The second string arrived with `replace`, which is the only name on the accessor
that takes two of them, and it was worth widening the door rather than opening a
fourth one. A fourth door would have been picked by argument shape, which is the
one rule this file has.

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

### The one answer that is wider than a column

`partition` and `rpartition` hand back three columns rather than one, which is
the first answer shape on this accessor that no door carried. They get a
function of their own and that is the rule working rather than an exception to
it: the doors are picked by the shape of the answer, and three columns is a
shape. `translate` beside them is the exception, because what makes it separate
is the shape of its argument.

### The answer whose width comes out of the data

`get_dummies` splits each row at a separator and answers one column per distinct
token. Its width is not three and it is not one: it is however many distinct
tokens the column turned out to hold, which is not knowable from the name or
from the arguments.

That is still the same rule, because a frame of unknown width is a shape and no
door carries it. What is different is that it takes two functions rather than
one. Nothing can be allocated until the column has been read once, so the first
reads it and answers the labels and the second fills the columns in, and the
Python layer is what holds the two together and turns them into a frame.

### The one answer that is narrower than a column

`cat` with no other column to concatenate against folds the whole thing into a
single string, which is the other direction off the same rule and gets a
function of its own for the same reason. Every door here answers a column and a
scalar is not one.

The other half of that name, the one that takes a column and works row by row,
is refused in the Python layer and is not represented here. It aligns two
columns on their labels first, which is a piece of machinery this library does
not have yet, and serving it by position instead would answer a different
question to the one that was asked.

The thirty odd names this file does not spell do not resolve at all rather than
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
        name == "upper"
        or name == "lower"
        or name == "capitalize"
        or name == "title"
        or name == "swapcase"
        or name == "casefold"
        or name == "slice"
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
        or name == "replace"
        or name == "replace_folded"
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
    if (
        name == "startswith"
        or name == "endswith"
        or name == "isspace"
        or name == "islower"
        or name == "isupper"
        or name == "istitle"
        or name == "isascii"
        or name == "isalpha"
        or name == "isnumeric"
        or name == "isdigit"
        or name == "isdecimal"
        or name == "isalnum"
        or name == "contains"
        or name == "match"
        or name == "fullmatch"
        or name == "contains_folded"
        or name == "match_folded"
        or name == "fullmatch_folded"
    ):
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
    if name == "len" or name == "find" or name == "rfind" or name == "count":
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
    other: String,
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
        other: The second string, for `replace` alone, which is the only name in
            the accessor that takes two. Every other name here leaves it empty.
        start: The first position, where the method has one, the index for
            `get`, the width or the repeat count for the ones that take a number
            rather than a position, and how many matches to replace.
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
    if wanted == "upper":
        return column.chars_upper()
    if wanted == "lower":
        return column.chars_lower()
    if wanted == "capitalize":
        return column.chars_capitalize()
    if wanted == "title":
        return column.chars_title()
    if wanted == "swapcase":
        return column.chars_swapcase()
    if wanted == "casefold":
        return column.chars_casefold()
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
    if wanted == "replace":
        # The only name here that takes two strings, and the only one in the
        # accessor that pandas reads as a literal by default, since its `regex`
        # argument defaults to False in pandas 3. So nothing is refused on the
        # way in and `n` rides in the position slot the way a width does.
        return column.chars_replace(arg, other, _whole(start, "n"))
    if wanted == "replace_folded":
        # `case=False`, and the one of the four folded names whose answer is
        # text. pandas answers this one out of Python rather than out of Arrow
        # and the two fold the same way anyway, which document 69 measures.
        return column.chars_replace_folded(arg, other, _whole(start, "n"))
    return column.chars_repeat(_whole(start, "repeats"))


def translate(column: Series, keys: Series, values: Series) raises -> Series:
    """Swaps single characters one for one, out of a table.

    The one name in this accessor that does not come through `text`, and the
    reason is not the shape of its answer, which is text like twenty two others.
    It is that its argument is not a scalar. A table has as many entries as it
    has, the same way `isin`'s set does, and the three doors carry two strings
    and two positions between them because that is what a scalar argument looks
    like. There is no way to fold a table into a string: a replacement can hold
    any character, so no character is available to separate one entry from the
    next, and an encoding that got around that would be a format nobody asked
    for in a place nobody would look for it.

    So the rule the three doors follow is unchanged and this is outside it
    rather than an exception to it. A fourth door for a fourth argument shape
    would have been the thing document 07 warns about; a door for the one
    argument in the namespace that is column sized is a different claim.

    Args:
        column: The column to read.
        keys: The characters to replace, one character per row.
        values: What to put in their place, in the same order.

    Returns:
        A new column, as tall as the one it read.

    Raises:
        Error: Tagged `value` if the column is not text, and whatever the
            kernel raises about the table otherwise.
    """
    _text_column(column)
    return column.chars_translate(keys, values)


def partition(
    column: Series, sep: String, from_right: Bool
) raises -> List[Series]:
    """Cuts every row at a separator and hands back three columns.

    The second name in this accessor that does not come through the three doors,
    and unlike `translate` this one is outside them for the reason the rule
    names. The rule is that the shape of the answer picks the door, and the
    shape of this answer is three columns rather than one. That is a new shape
    and not a new argument, so a door for it follows the rule instead of
    standing beside it.

    It is one call and not three because the search is what this costs. Asking
    for the head, then the separator, then the tail would run the same search
    over the same column three times and throw two thirds of each answer away.

    The other half of pandas, `expand=False`, would answer one column of
    three element tuples, and there is no column type here that can hold one. It
    is refused in the Python layer rather than approximated.

    Args:
        column: The column to read.
        sep: The separator to cut at, which the Python layer has already
            checked is not empty because pandas refuses that.
        from_right: Whether to cut at the last occurrence rather than the first,
            which is `rpartition` rather than `partition`.

    Returns:
        Three columns, each as tall as the one they read, in the order pandas
        labels 0, 1 and 2.

    Raises:
        Error: Tagged `value` if the column is not text.
    """
    _text_column(column)
    return column.chars_partition(sep, from_right)


def dummy_tokens(column: Series, sep: String) raises -> List[String]:
    """Works out what columns a dummy frame will have.

    The fourth shape outside the three doors, and the strangest of them.
    `partition` answers three columns and `cat` answers a scalar, and both of
    those are widths a reader could work out from the name. This one answers a
    frame whose width is a property of the data, so the caller cannot allocate
    anything until the column has been read once.

    That is why this is two functions and not one. This half reads the column
    and hands back the labels, and the Python layer uses them both to name the
    columns and to ask for them.

    Args:
        column: The column to read.
        sep: The text to split each row at, which the Python layer has already
            checked is not empty because pandas refuses that.

    Returns:
        The distinct tokens in the order the columns go in.

    Raises:
        Error: Tagged `value` if the column is not text.
    """
    _text_column(column)
    return column.chars_dummy_tokens(sep)


def dummies(
    column: Series, sep: String, tokens: List[String]
) raises -> List[Series]:
    """Fills in the columns of a dummy frame.

    The other half of `dummy_tokens`, which has to have run first. It is split
    that way rather than answering both at once because the two halves cross to
    Python separately: a list of labels is a list of strings and a list of
    columns is a list of wrapped series, and pairing them up on the Mojo side
    would mean inventing a shape for the pair.

    Args:
        column: The column to read.
        sep: The text to split each row at.
        tokens: The tokens, as `dummy_tokens` answered them.

    Returns:
        One int64 column per token, in the same order.

    Raises:
        Error: Tagged `value` if the column is not text.
    """
    _text_column(column)
    return column.chars_dummies(sep, tokens)


def join(
    column: Series, sep: String, na_rep: String, skip_missing: Bool
) raises -> String:
    """Folds the whole column into one string.

    The third name outside the three doors and the second one that is outside
    them for the reason the rule gives. A scalar is a shape, no door carries it,
    and every door here answers a column. `str.cat` with nothing to concatenate
    against is a reduction rather than a transformation, so a function of its
    own is the rule rather than a break from it.

    The half of `str.cat` that takes another column and works row by row is not
    here at all. That one aligns two columns on their labels before it does
    anything, and alignment is not written yet, so the Python layer refuses it
    by name rather than approximating it by position.

    Args:
        column: The column to read.
        sep: The text to put between neighbouring rows, which the Python layer
            has already turned an absent argument into the empty string for.
        na_rep: The text to stand in for a missing row.
        skip_missing: Whether a missing row is dropped rather than replaced,
            which is what pandas decides from whether `na_rep` was given.

    Returns:
        One string.

    Raises:
        Error: Tagged `value` if the column is not text.
    """
    _text_column(column)
    return column.chars_join(sep, na_rep, skip_missing)


def flag(column: Series, kind: String, arg: String) raises -> Series:
    """Runs one of the methods that answers a mask, and hands back a column.

    Args:
        column: The column to read.
        kind: The method, as pandas spells it.
        arg: The prefix, the suffix or the pattern, and the empty string for
            every question about what the characters are, which takes no
            argument at all.

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
    if wanted == "isspace":
        return column.chars_is_space()
    if wanted == "islower":
        return column.chars_is_lower()
    if wanted == "isupper":
        return column.chars_is_upper()
    if wanted == "istitle":
        return column.chars_is_title()
    if wanted == "isascii":
        return column.chars_is_ascii()
    if wanted == "isalpha":
        return column.chars_is_alpha()
    if wanted == "isnumeric":
        return column.chars_is_numeric()
    if wanted == "isdigit":
        return column.chars_is_digit()
    if wanted == "isdecimal":
        return column.chars_is_decimal()
    if wanted == "isalnum":
        return column.chars_is_alnum()
    # The three pattern questions take a literal and pandas takes a regular
    # expression. Which patterns are allowed to arrive here is decided in the
    # Python layer, because that is where the pattern is still a Python string
    # and where a refusal can name the metacharacter that caused it.
    if wanted == "contains":
        return column.chars_contains(arg)
    if wanted == "match":
        return column.chars_match(arg)
    if wanted == "fullmatch":
        return column.chars_full_match(arg)
    # The same three with `case=False`, which is a word of its own rather than a
    # seventh argument on the door, for the reason `strip` and `strip_chars` are
    # two words: the name and what it does with its argument are what a caller
    # picked, and a flag beside the name would put that choice in two places.
    if wanted == "contains_folded":
        return column.chars_contains_folded(arg)
    if wanted == "match_folded":
        return column.chars_match_folded(arg)
    if wanted == "fullmatch_folded":
        return column.chars_full_match_folded(arg)
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
        arg: The substring to look for or to count, and the empty string for
            `len`.
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
    if wanted == "count":
        return column.chars_count(arg)
    return column.chars_find(arg, start, stop, wanted == "rfind")
