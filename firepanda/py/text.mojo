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

The second string arrived with `replace`, which with its regular expression
form is the only name on the accessor that takes two of them, and it was worth
widening the door rather than opening a fourth one. A fourth door would have
been picked by argument shape, which is the one rule this file has.

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

from std.python import Python

from firepanda.frame.series import Series
from firepanda.kernel.chars import character_count
from firepanda.kernel.regex.method import (
    ALPHABET_ROWS,
    METHOD_CONTAINS,
    METHOD_COUNT,
    METHOD_EXTRACT,
    METHOD_FULLMATCH,
    METHOD_MATCH,
    METHOD_REPLACE,
    program_for,
)
from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.program import Program
from firepanda.kernel.regex.tokens import FLAG_IGNORECASE
from firepanda.kernel.regex.replace import (
    Rewrite,
    parse_rewrite,
    parse_rewrite_python,
)
from firepanda.py.errors import DTYPE, UNSUPPORTED, VALUE, tagged


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
        or name == "normalize"
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
        or name == "replace_regex"
        or name == "replace_regex_python"
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
        or name == "contains_regex"
        or name == "match_regex"
        or name == "fullmatch_regex"
        or name == "contains_folded"
        or name == "match_folded"
        or name == "fullmatch_folded"
        or name == "contains_regex_folded"
        or name == "match_regex_folded"
        or name == "fullmatch_regex_folded"
        or name == "contains_regex_python"
        or name == "match_regex_python"
        or name == "fullmatch_regex_python"
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
    if (
        name == "len"
        or name == "find"
        or name == "rfind"
        or name == "count"
        or name == "count_regex"
        or name == "count_regex_python"
    ):
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


def _python_minor() raises -> Int:
    """Which CPython this call arrived in, as the minor number alone.

    The thing this library copies on one of its two engines is the `re` module,
    and `re` is not the same module in every version of Python this project
    supports. `\\B` on a row with nothing in it is the rule that changed: three
    versions fail it and 3.14 matches it. pandas answers a `findall` or a call
    carrying a `flags` argument by running `re` in this process, so the answer
    that agrees with pandas is the answer this interpreter would have given and
    not the answer the newest one would have.

    Read here rather than passed in from Python, because it is not a thing a
    caller chose. The door takes a word for the engine and a number for the flag
    letters and both of those are the caller speaking. This is the room the call
    is standing in.

    Read on every compile rather than once and kept, because a module level
    value that is filled in on first use is a thing to get right in a language
    with no module level mutable state, and the cost is one attribute read
    beside the building of a program.

    Returns:
        The minor version, so 13 for CPython 3.13.

    Raises:
        Error: If `sys` cannot be reached, which it can.
    """
    return Int(py=Python.import_module("sys").version_info.minor)


def _compiled(
    kind: String, pattern: String, flags: Int32 = 0, rows: Int = 0
) raises -> Program:
    """Compiles what a method would run, or raises the refusal that belongs to it.

    Everything about which pattern and which engine is in
    `firepanda/kernel/regex/method.mojo`. What is decided here, and can only be
    decided here, is what a refusal becomes in Python.

    The two kinds the compiler tells apart become two different exceptions, and
    telling them apart is the whole point of the flag it carries. A pattern RE2
    refuses is refused by pandas today with an Arrow error, which is a
    `ValueError` in Python, so `value` is the tag that keeps a program written
    against pandas working. A pattern this library has not learned yet is
    `unsupported`, which reaches Python as `NotImplementedError`, because it is
    a shortfall and a caller who catches `ValueError` around a pattern they know
    to be good should not be told they wrote a bad one.

    The wording is this library's rather than RE2's. Document 76 made that
    decision for the routing refusal and the argument is the same one: pandas'
    message names Arrow, a caller here did not call Arrow, and reproducing a
    refusal in kind is what matters rather than reproducing it to the letter.

    A word ending in `_regex_folded` is the same method with `case=False`, and
    the fold it asks for is the one `(?i)` asks for rather than the one the byte
    search folds through. Those two are not the same rule: the search maps one
    character to one character and the engine takes every code point that folds
    onto the one written down, so `STRASSE` holds `straße` to neither and a row
    holding a Kelvin sign matches `(?i)k` to the engine alone. Upstream reaches
    the same place by compiling the pattern with `re.IGNORECASE` and handing the
    compiled object on, which is why `case=False` and a written `(?i)` answer
    alike there and have to answer alike here.

    A word ending in `_python` is the same method on the other engine, which is
    where upstream sends a call carrying a `flags` argument, a `case=False` on
    `replace`, a replacement naming a group by name, or an empty pattern. That
    is a different fact from the word ending in `_regex_folded` even when the
    bits the two carry are the same one. `case=False` on `contains` folds on RE2
    and `flags=re.IGNORECASE` folds on Python's engine, because upstream routes
    on how the caller spelled it rather than on what they asked for, so the two
    cross this door by two different routes and can answer differently. They do
    answer differently, on the four Turkish I code points, which is the whole of
    the measured gap between the two engines' fold tables and is the thing this
    arrangement exists to keep visible.

    The engine being a word and the letters being a number is the division the
    door settled on. A route is a choice a caller made by name, in the sense that
    every caller who lands on Python's engine got there by writing something
    upstream looks for, so it reads as a word like `_folded` does. The seven flag
    letters are not: they arrive as a number in any of 128 combinations, and
    there is no list of words for that anybody would want to read. Document 85
    had the routing riding in the number being nonzero, which held while only
    `contains` and `fullmatch` were served and stopped holding the moment
    `replace` arrived, since a replacement holding `\\g<` moves a call with no
    flags at all.

    Args:
        kind: The word the Python layer sent, which is one of the six that end
            in `_regex`, one of the three that end in `_regex_folded`, or one of
            the five that end in `_regex_python`.
        pattern: The pattern as the caller wrote it.
        flags: The flags the caller passed beside the pattern, as `FLAG_` bits,
            and zero when they passed none. They mean what the letters mean and
            they do not decide the engine.
        rows: How tall the column is, which decides nothing about the answer and
            only whether the pattern is compiled with the alphabet the state
            cache runs on. Zero is the answer for a call that has no column to
            speak of or does not reach the cache, and it means the alphabet is
            left out.

    Returns:
        The compiled program.

    Raises:
        Error: Tagged `value` when RE2 would refuse the pattern too, and
            `unsupported` when the refusal is this library's own.
    """
    var python = kind.endswith("_python")
    var folded = kind.endswith("_folded")
    var name = (
        String(kind[byte = 0 : kind.byte_length() - 7]) if python
        or folded else kind
    )
    var method = METHOD_CONTAINS
    if name == "match_regex":
        method = METHOD_MATCH
    elif name == "fullmatch_regex":
        method = METHOD_FULLMATCH
    elif name == "count_regex":
        method = METHOD_COUNT
    elif name == "replace_regex":
        method = METHOD_REPLACE
    elif name == "extract_regex":
        method = METHOD_EXTRACT
    var seeded = flags | (FLAG_IGNORECASE if folded else 0)
    var program = program_for(
        method,
        pattern,
        seeded,
        argued=python,
        minor=_python_minor(),
        alphabet=rows >= ALPHABET_ROWS,
    )
    if program.ok:
        return program^
    var said = String("str: ", program.problem, ", in the pattern ", pattern)
    raise tagged(UNSUPPORTED if program.gap else VALUE, said)


def _rewritten(program: Program, replacement: String) raises -> Rewrite:
    """Reads a replacement string, or raises the way pandas raises.

    There are two grammars and the program picks which one, because upstream
    picks with the same decision that picked the engine: a call answered by
    `replace_substring_regex` has its rewrite read RE2's way and a call answered
    by `re.sub` has its template read Python's way. So the replacement is not
    read until the pattern has been compiled, and a caller who changed nothing
    but the pattern can change what their replacement means.

    Every refusal either grammar makes is one upstream makes too, so there is one
    tag here rather than the two the pattern needs. It is not the same class in
    both places: Arrow's refusal is a `ValueError` in Python and `re`'s is a
    `re.PatternError`, which is not one, and a `ValueError` is what a program
    written against this library already catches. The wording is this library's
    for the reason `_compiled` gives.

    Args:
        program: The compiled pattern, which says which engine will run and
            therefore which grammar the replacement is written in, and how many
            groups and what they are called.
        replacement: The replacement as the caller wrote it.

    Returns:
        The replacement, read.

    Raises:
        Error: Tagged `value` when it cannot be read.
    """
    var rewrite = parse_rewrite_python(
        replacement, program.groups, program.labels
    ) if program.python else parse_rewrite(replacement, program.groups)
    if rewrite.ok:
        return rewrite^
    raise tagged(
        VALUE,
        String("str: ", rewrite.problem, ", in the replacement ", replacement),
    )


def text(
    column: Series,
    kind: String,
    arg: String,
    other: String,
    start: Optional[Int],
    stop: Optional[Int],
    step: Int,
    flags: Int = 0,
) raises -> Series:
    """Runs one of the methods that answers text, and hands back a column.

    Args:
        column: The column to read.
        kind: The method, as pandas spells it.
        arg: The prefix, suffix or replacement, the characters to strip, the
            character to pad with, or the normalization form, and the empty
            string for the ones that take none.
        other: The second string, for `replace` alone, which is the only name in
            the accessor that takes two. Every other name here leaves it empty.
        start: The first position, where the method has one, the index for
            `get`, the width or the repeat count for the ones that take a number
            rather than a position, and how many matches to replace.
        stop: The position to stop before, where the method has one.
        step: How far to move between characters, for `slice` alone.
        flags: The flags the caller passed beside the pattern, as `FLAG_` bits,
            for the one name here that has a pattern and zero for the other
            twenty three. They say what the letters say and the word says which
            engine, which is the division `_compiled` explains.

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
    if wanted == "normalize":
        # The form rides in the `arg` slot, because it is a string and because a
        # door picked by the shape of the answer has no other place to put one.
        # It is read here rather than in Python for the same reason the slice
        # step is refused twice: this door is reachable from the Mojo API, so
        # the four names have to be known on this side as well.
        if arg == "NFC":
            return column.chars_normalize(False, True)
        if arg == "NFD":
            return column.chars_normalize(False, False)
        if arg == "NFKC":
            return column.chars_normalize(True, True)
        if arg == "NFKD":
            return column.chars_normalize(True, False)
        raise tagged(VALUE, String("invalid normalization form"))
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
    if wanted == "replace_regex" or wanted == "replace_regex_python":
        # The two names in this door that reach the engine, and the only ones
        # anywhere that have two things to refuse: the pattern, which is refused
        # the way the other regular expression names refuse theirs, and the
        # replacement, which has a grammar of its own and has a different one on
        # each engine.
        #
        # The count is the other difference between the two. Arrow's bounded
        # replace is a scan this library will not copy, for the reasons
        # `replace.mojo` gives, so the binding refuses `n` on that path and
        # nothing here ever sees one. Python's is the same scan with a counter
        # on it, which is what `re.sub` does, so the number rides in the
        # position slot the way a width does and the absence of one means all
        # of them.
        var program = _compiled(wanted, arg, Int32(flags))
        var limit = start.value() if start else -1
        return column.chars_replace_regex(
            program, _rewritten(program, other), limit
        )
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


def extract(
    column: Series, pattern: String, flags: Int = 0
) raises -> Tuple[List[String], List[Series]]:
    """Pulls the groups of the first match out of every row, as columns.

    The fifth shape outside the three doors and the second whose width comes
    out of something rather than being written down. `dummies` reads the column
    to find out how wide its answer is, and this one reads the pattern, which
    is why this is one call where that one is two: the labels and the columns
    are both known as soon as the pattern has been compiled, and compiling it
    twice to hand them over separately would be compiling it twice.

    The labels are the group names, with an empty string for a group that was
    not named. pandas labels an unnamed group with its position counting from
    zero, and a firepanda frame holds text labels, so the Python layer writes
    the position out as text and the divergence is the one `partition` already
    has. Naming the groups is how a caller avoids it.

    A pattern that reads and opens no group is refused here with pandas' own
    sentence, and it is refused before the pattern is compiled because that is
    upstream's order. `re.compile` runs first there, so a pattern whose syntax
    is broken is a syntax error and a pattern that is merely groupless is the
    other message, and a pattern that is groupless and also holds a construct
    this library has not written is the other message too. Compiling first would
    get the last of those three backwards.

    A `flags` argument moves nothing here, which is what makes this the one
    name on the accessor where the number arrives without a word beside it.
    The other five cross a door that also has to say which engine is meant,
    because they have two and the caller's spelling picks one. This one has
    always been on Python's engine and has no second engine to be moved to, so
    the letters are read for what they mean and for nothing else. They still
    have to be read before the groups are counted, since a group is opened by
    a bracket that verbose mode does not change and a flag this library cannot
    carry should be refused over the pattern rather than over its groups.

    Args:
        column: The column to read.
        pattern: The pattern as the caller wrote it, which is compiled for
            Python's engine because this is one of the three names pandas never
            sends to Arrow.
        flags: The flags the caller passed beside the pattern, as `FLAG_` bits,
            and zero when they passed none.

    Returns:
        The group labels and the columns, in the order the groups were opened
        and the same length as each other.

    Raises:
        Error: Tagged `value` if the column is not text or RE2 would refuse the
            pattern too, and `unsupported` when the refusal is this library's
            own.
    """
    _text_column(column)
    var tree = parse_pattern(pattern, Int32(flags))
    if tree.ok and tree.groups == 0:
        raise tagged(VALUE, String("pattern contains no capture groups"))
    var program = _compiled(String("extract_regex"), pattern, Int32(flags))
    var labels = program.labels.copy()
    return (labels^, column.chars_extract_regex(program))


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


def flag(
    column: Series, kind: String, arg: String, flags: Int
) raises -> Series:
    """Runs one of the methods that answers a mask, and hands back a column.

    The door takes a number as well as a word, which is the arrangement the text
    door has too. The word says which method, whether it folds and which engine
    runs it, because all three of those are choices a caller made by name. The
    flags are not a choice made by name: seven letters in any combination is not
    a list of words anybody wants to write down. So the letters ride in a slot
    and everything else rides in the word.

    Args:
        column: The column to read.
        kind: The method, as pandas spells it.
        arg: The prefix, the suffix or the pattern, and the empty string for
            every question about what the characters are, which takes no
            argument at all.
        flags: The flags the caller passed beside the pattern, as `FLAG_` bits,
            and zero for every call that passed none and for every name in this
            door that has no pattern to pass them about.

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
    # expression. A pattern with no metacharacter in it means the same thing
    # either way and gets the byte search, which is what these three are, and
    # the Python layer is what decides that because that is where the pattern is
    # still a Python string.
    if wanted == "contains":
        return column.chars_contains(arg)
    if wanted == "match":
        return column.chars_match(arg)
    if wanted == "fullmatch":
        return column.chars_full_match(arg)
    # Everything else goes to the engine. Three words rather than one, because
    # the pattern arrives as the caller wrote it and which of the three asked is
    # what decides both the rewrite and, for a pattern that cannot be answered,
    # nothing at all: the routing decision is made before the rewrite.
    if (
        wanted == "contains_regex"
        or wanted == "match_regex"
        or wanted == "fullmatch_regex"
        or wanted == "contains_regex_folded"
        or wanted == "match_regex_folded"
        or wanted == "fullmatch_regex_folded"
        or wanted == "contains_regex_python"
        or wanted == "match_regex_python"
        or wanted == "fullmatch_regex_python"
    ):
        return column.chars_matches_regex(
            _compiled(wanted, arg, Int32(flags), len(column))
        )
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
    flags: Int = 0,
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
        flags: The flags the caller passed beside the pattern, as `FLAG_` bits,
            for the one name here that has a pattern and zero for the other
            four.

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
    # The two names in this door that reach the engine, and they reach it
    # through the same call the ones in the door above do, because what a
    # refusal becomes in Python is a fact about the binding rather than about
    # the question being asked. Which of the two words arrived says which engine
    # runs, and the program carries that on to the scan.
    if wanted == "count_regex" or wanted == "count_regex_python":
        return column.chars_count_regex(_compiled(wanted, arg, Int32(flags)))
    return column.chars_find(arg, start, stop, wanted == "rfind")
