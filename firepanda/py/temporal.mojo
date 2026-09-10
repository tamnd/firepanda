"""The `dt` accessor, which is one namespace and three shapes of answer.

Same argument as `reduce.mojo` and `transform.mojo`, arrived at for a third
time. pandas puts about forty names on `s.dt` and the core already answers
twenty two of them through one `Series.dt(name)` that takes the pandas spelling,
so the crossing is a word rather than forty methods.

### Why a string rides along beside the name

`floor`, `ceil`, `round` take a frequency, `as_unit` takes a unit, `strftime`
takes a format, `tz_convert` and `tz_localize` take a zone, and `day_name` and
`month_name` take a locale. Every one of those is a string, none of them takes
two, and the twenty five that take nothing are handed an empty one and ignore
it. That is the same shape `transform.mojo` uses for its integer and it is the
same reasoning: the crossing cares that a string arrived and the Python layer is
where it has a name a caller was invited to type.

### Why there are three doors and not one

`tz` and `unit` answer a word rather than a column, and `isocalendar` answers a
frame of three columns rather than one column. The rule from `transform.mojo`
decides it: the shape of the answer picks the door, not how similar the words
are. So `part` hands back a series, `word` hands back a string and
`iso_calendar` hands back a frame, and a caller that sends a name through the
wrong one is refused rather than being quietly served.

### What is not here

`time` and `timetz` need a time of day column type, which Arrow has and
firepanda does not yet. `to_period` needs a period type, `to_pydatetime` needs a
column of Python objects, and `freq` needs frequency inference over an index.
None of the four is a line in a table and none of them is spelled here, so the
name does not resolve at all rather than resolving and refusing, because an
absent name reads as unimplemented on the board and a refusing one reads as a
failure. That distinction is the whole of document 07.

`tz_localize(None)` arrives as its own word, `tz_localize_none`, rather than as
an absent argument. The Python layer is where `None` means "take the clock off"
and this is where it is a different operation from naming a zone, which it is,
since one reinterprets the readings and the other keeps them.
"""

from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit
from firepanda.frame import DataFrame
from firepanda.frame.index import Index
from firepanda.frame.series import Series
from firepanda.kernel.temporal import (
    TemporalField,
    field_named,
    frequency_period,
    temporal_field,
)
from firepanda.py.errors import VALUE, tagged


def temporal(name: String) raises -> String:
    """Checks that a word names something on the `dt` accessor, and hands it back.

    Separate from the three doors for the reason `transform.mojo` keeps
    `transformation` separate from `transformed`: the caller checks the word
    before it opens the handler that turns a kernel complaint into a dtype
    error, so a name nobody implements comes back tagged as the value error it
    is rather than as a complaint about the column's type.

    It checks the word against all three doors rather than against one, because
    a caller that sent `tz` to `part` has made a mistake about the shape of the
    answer and should hear about that rather than hearing that `tz` does not
    exist, which is not true. Which door a word belongs to is what `column_part`
    and `word_part` below decide.

    Args:
        name: The word that crossed, as pandas spells the attribute.

    Returns:
        The same word.

    Raises:
        Error: Tagged `value` if it is not one of the names the accessor has.
    """
    if (
        name == "tz"
        or name == "unit"
        or name == "isocalendar"
        or name == "date"
        or name == "normalize"
        or name == "days"
        or name == "total_seconds"
        or name == "floor"
        or name == "ceil"
        or name == "round"
        or name == "as_unit"
        or name == "day_name"
        or name == "month_name"
        or name == "strftime"
        or name == "tz_convert"
        or name == "tz_localize"
        or name == "tz_localize_none"
    ):
        return name
    # Everything else has to be one of the nineteen calendar and clock fields,
    # and the core owns that table. Asking it rather than keeping a copy is the
    # point `field_named` makes in its own docstring: two copies of a nineteen
    # row table is two chances to spell `days_in_month` differently.
    try:
        _ = field_named(name)
    except:
        raise tagged(VALUE, String("unknown datetime part ", name))
    return name


def column_part(name: String) raises -> String:
    """Checks that a word names a part that answers a column.

    Every door has one of these and the boundary calls it before it opens the
    handler that retags a kernel complaint, which is what keeps a word a caller
    chose from coming back as a complaint about the column's type. `part` calls
    it too, so a Mojo caller that reaches the function directly is checked as
    well and the message lives in one place.

    Args:
        name: The word that crossed.

    Returns:
        The same word.

    Raises:
        Error: Tagged `value` if the accessor has no such name, or if it has one
            and the answer is not a column.
    """
    _ = temporal(name)
    if name == "tz" or name == "unit" or name == "isocalendar":
        raise tagged(
            VALUE,
            String(
                name,
                " does not answer a column, so it does not come through here",
            ),
        )
    return name


def word_part(name: String) raises -> String:
    """Checks that a word names a part that answers a word.

    Two of them, and the reason there are only two is that `tz` and `unit` are
    the only things pandas puts on the accessor that are a property of the
    column's type rather than of its rows.

    Args:
        name: The word that crossed.

    Returns:
        The same word.

    Raises:
        Error: Tagged `value` if the accessor has no such name, or if it has one
            and the answer is not a word.
    """
    _ = temporal(name)
    if name != "tz" and name != "unit":
        raise tagged(VALUE, String(name, " does not answer a word"))
    return name


def frequency(spelling: String) raises -> String:
    """Checks that a word is a frequency something can be rounded to.

    Separate from the rounding itself for the reason `temporal` above is separate
    from `part`: the caller checks the argument before it opens the handler that
    turns a kernel complaint into a dtype error, so a frequency nobody can round
    to comes back as the value error pandas raises rather than as a complaint
    about the column's type, which is not what went wrong.

    It asks the kernel rather than keeping its own list, because the vocabulary
    is seven aliases with a count and a sign in front and two copies of that is
    two chances to accept `1.5h` in one place and refuse it in the other. The
    nanosecond type it asks against is a stand in and nothing about the answer
    depends on it: the type decides what unit the period comes back in, and the
    period is thrown away here.

    Args:
        spelling: The frequency, as pandas spells it.

    Returns:
        The same word.

    Raises:
        Error: Tagged `value` if it is not a fixed frequency, keeping whatever
            the kernel said about why.
    """
    try:
        _ = frequency_period(spelling, LogicalType.timestamp(TimeUnit.NANO))
    except e:
        raise tagged(VALUE, String(e))
    return spelling


def part(column: Series, kind: String, arg: String) raises -> Series:
    """Reads one part of a temporal column, and hands back a column.

    Args:
        column: The column to read.
        kind: The part, as pandas spells the attribute.
        arg: The frequency, unit, format, zone or locale, and the empty string
            for the twenty five that take none.

    Returns:
        A new column, as tall as the one it read.

    Raises:
        Error: Tagged `value` if the name is not one that answers a column, and
            whatever the kernel raises for a column whose type has no such part.
    """
    _ = column_part(kind)
    if kind == "floor" or kind == "ceil" or kind == "round":
        _ = frequency(arg)
    if kind == "date":
        return column.dt_date()
    if kind == "normalize":
        return column.dt_normalize()
    if kind == "days":
        return column.dt_days()
    if kind == "total_seconds":
        return column.dt_total_seconds()
    if kind == "floor":
        return column.dt_floor(arg)
    if kind == "ceil":
        return column.dt_ceil(arg)
    if kind == "round":
        return column.dt_round(arg)
    if kind == "as_unit":
        return column.dt_as_unit(arg)
    if kind == "day_name":
        return column.dt_day_name(arg)
    if kind == "month_name":
        return column.dt_month_name(arg)
    if kind == "strftime":
        return column.dt_strftime(arg)
    if kind == "tz_convert":
        return column.dt_tz_convert(arg)
    if kind == "tz_localize":
        return column.dt_tz_localize(arg)
    if kind == "tz_localize_none":
        return column.dt_tz_localize_none()
    return column.dt(kind)


def word(column: Series, kind: String) raises -> String:
    """Reads the one part of a temporal column that is a word rather than a column.

    Two of them, and they are together because they answer the same shape.
    `tz` is the clock the column is read against and `unit` is how many of its
    integers make a second, and pandas spells both as a string on the accessor.

    Args:
        column: The column to read.
        kind: Either `tz` or `unit`.

    Returns:
        The zone name, empty when the column carries none, or the unit spelled
        the way pandas spells it inside a dtype.

    Raises:
        Error: Tagged `value` if the name is not one of the two, and whatever
            the core raises for a column that is not temporal.
    """
    _ = word_part(kind)
    if kind == "tz":
        return column.dt_tz()
    if not column.logical().is_temporal():
        raise Error(
            String(
                "temporal: only a temporal column has a unit, and this one is ",
                column.logical(),
            )
        )
    return String(column.logical().unit)


def iso_calendar(column: Series) raises -> DataFrame:
    """Reads the three ISO 8601 week date fields, as a frame.

    The one part of the accessor that answers a frame, which is why it has a
    door rather than a name in `part`. pandas returns a frame of `year`, `week`
    and `day` and so does this, with the labels of the column it read.

    The three fields are deliberately unreachable through `Series.dt(name)` in
    the core, and the reason is written there: two of the three are spelled the
    same as ordinary fields, so a caller who asked for `year` and got the ISO
    year would be wrong for eleven months and right for the twelfth, which is
    the worst way to be wrong. They are reached here through their field codes,
    which is a door nobody can arrive at by typing a name.

    Args:
        column: The column to read.

    Returns:
        A frame of three columns, `year`, `week` and `day`.

    Raises:
        Error: Whatever the field kernel raises for a column that is not a date
            or a naive timestamp.
    """
    var parts = List[Series](capacity=3)
    parts.append(
        Series("year", temporal_field(column.values, TemporalField.ISO_YEAR))
    )
    parts.append(
        Series("week", temporal_field(column.values, TemporalField.ISO_WEEK))
    )
    parts.append(
        Series("day", temporal_field(column.values, TemporalField.ISO_DAY))
    )
    var out = DataFrame.from_series(parts^)
    out.index = Index(copy=column.index)
    return out^
