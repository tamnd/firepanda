"""Rendering a frame as text.

A dataframe you cannot look at is not a dataframe, and every one of the tests
written before this file existed had to reach through `as_typed` and read values
back one at a time to say what it saw. So this is worth having early even though
nothing depends on it.

The output follows pandas closely enough that a Python reader will not have to
think about it: a header row, the row labels down the left, values right aligned
in their columns, and both the rows and the columns elided in the middle when
there are too many to print.

The labels down the left are the frame's own labels and not the row positions,
and they are left aligned rather than right aligned, both of which are pandas'
layout. A named index gets its name printed too, on a line of its own above the
listing on a column and inside the table under the header on a frame. None of
that happens in here, because this file cannot reach an `Index` without a cycle.
The labels arrive already rendered as an `IndexCells`, and a caller that passes
none still gets the positions, which is what every caller got before labels
existed.

The spacing is pandas' spacing down to the byte. Every value is written one
place in from the separator and a negative number spends that place on its
minus, so a column of `1.5` and `-0.5` lines up on the dot and sits one place
to the left of where a column of `1.5` and `0.5` sits, and a column's name is
held in by the same place when the column is one pandas calls numeric. The
elision is three dots, or two in a column of three or fewer, centred on a
column and right aligned on a frame. None of that is worth arguing about and
all of it was measured rather than reasoned about, because the point of
matching is that somebody putting the two outputs side by side sees either
nothing or something worth reading.

Three decisions in here are ours rather than inherited.

**Nulls print as `<NA>`.** pandas has two spellings depending on whether a column
is a numpy float, where a null is a `NaN`, or a masked or Arrow backed dtype,
where it is `<NA>`. In firepanda every dtype is nullable through the validity
bitmap and `NaN` is a float value that a column can genuinely hold, so the two
have to look different. A float column with a null in row 3 and a `NaN` in row 4
prints `<NA>` and `NaN`, and they are not the same thing.

**Floats print to six decimals and drop trailing zeros.** Mojo prints a `Float64`
at the shortest representation that round trips, which for one third is
seventeen characters and makes a table unreadable. Six decimals is the pandas
default. The rest of what a float column looks like is not ours either: the
format is picked for the column rather than for the value, so a column goes to
scientific notation as a whole and the zeros come off it as a whole, and
`float_column` is where that happens.

**The shape line is always printed.** pandas prints it only when it truncated
something, so the absence of it means one thing when the frame is small and
another when it is not. Printing it always costs one line and removes the
question.

This file deliberately does not import `DataFrame` or `Series`. It renders a
`Schema` and a list of columns, which is what those two are made of, and that is
what lets `frame.mojo` and `series.mojo` both call it without a cycle.
"""

from firepanda.array.any import AnyArray, ColumnRefs
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Schema
from firepanda.kernel.temporal import instant_text

comptime DEFAULT_MAX_ROWS = 10
"""Rows printed before the middle is elided. Half from each end."""

comptime DEFAULT_MAX_COLUMNS = 20
"""Columns printed before the middle is elided. Half from each end."""

comptime DEFAULT_FLOAT_PRECISION = 6
"""Decimal places a float is rounded to, trailing zeros removed after."""

comptime ELLIPSIS = "..."
"""What stands in for the rows and columns that were not printed."""

comptime SHORT_ELLIPSIS = ".."
"""What stands in for them in a column too narrow to hold three dots."""

comptime MINUS = Byte(ord("-"))
"""The one byte that can begin a rendered number besides a digit."""

comptime POINT = Byte(ord("."))
"""The decimal point, which is what makes a rendering strippable."""

comptime ZERO = Byte(ord("0"))
"""The low end of a digit."""

comptime NINE = Byte(ord("9"))
"""The high end of one."""


struct DisplayOptions(Copyable, Movable):
    """How much of a frame to print and how to spell what is in it."""

    var max_rows: Int
    """Rows to print before eliding the middle. At least two."""

    var max_columns: Int
    """Columns to print before eliding the middle. At least two."""

    var float_precision: Int
    """Decimal places for a float, before trailing zeros are removed."""

    var null_text: String
    """What a null looks like. Not `NaN`, which is a value a float can hold."""

    def __init__(out self):
        """Constructs the defaults."""
        self.max_rows = DEFAULT_MAX_ROWS
        self.max_columns = DEFAULT_MAX_COLUMNS
        self.float_precision = DEFAULT_FLOAT_PRECISION
        self.null_text = String("<NA>")

    def __init__(
        out self,
        *,
        max_rows: Int,
        max_columns: Int = DEFAULT_MAX_COLUMNS,
        float_precision: Int = DEFAULT_FLOAT_PRECISION,
    ):
        """Constructs options with the limits changed.

        Args:
            max_rows: Rows to print before eliding the middle.
            max_columns: Columns to print before eliding the middle.
            float_precision: Decimal places for a float.
        """
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.float_precision = float_precision
        self.null_text = String("<NA>")


def pad_left(text: String, width: Int) -> String:
    """Right aligns a cell in a column.

    Args:
        text: The cell.
        width: The column width in bytes.

    Returns:
        The padded cell, or the original if it is already wider.
    """
    var out = text
    while out.byte_length() < width:
        out = String(" ", out)
    return out^


def pad_right(text: String, width: Int) -> String:
    """Left aligns a cell in a column.

    Only the index column is aligned this way. pandas left aligns the labels
    whatever they are, so a column of labels `10`, `200` and `3` prints with all
    three hard against the left edge rather than lined up on their last digit,
    and copying that is the whole reason this exists beside `pad_left`.

    Args:
        text: The cell.
        width: The column width in bytes.

    Returns:
        The padded cell, or the original if it is already wider.
    """
    var out = text
    while out.byte_length() < width:
        out += " "
    return out^


def pad_middle(text: String, width: Int) -> String:
    """Centres a cell in a column.

    Only the elision on a column is aligned this way, and it is here rather than
    written out at the one call site because where the odd space goes is not
    obvious. Python's own `str.center` gives the odd space to the right except
    when both the padding and the width are odd, where it gives it to the left,
    and pandas centres the dots by calling exactly that. So a two dot elision in
    a column three wide prints as ` ..` and not as `.. `, which is a difference
    a reader would never predict and would see immediately.

    Args:
        text: The cell.
        width: The column width in bytes.

    Returns:
        The padded cell, or the original if it is already wider.
    """
    var margin = width - text.byte_length()
    if margin <= 0:
        return text
    var left = margin // 2 + (margin & width & 1)
    var out = String("")
    for _ in range(left):
        out += " "
    out += text
    while out.byte_length() < width:
        out += " "
    return out^


def keeps_a_place(type: LogicalType) -> Bool:
    """Whether a value is written one place in from the separator.

    pandas writes every value one place in and lets a negative number spend that
    place on its minus, which is why a column of `1.5` and `-0.5` lines up on
    the dot and sits one place to the left of where a column of `1.5` and `0.5`
    sits. Copying it matters more than it sounds, because somebody diffing this
    library's output against pandas' should see either nothing or something
    worth reading, and a whitespace difference on every numeric column is
    neither.

    A timestamp is the exception. pandas formats the temporal types through a
    path of their own that never keeps the place, and nothing they print could
    begin with a minus anyway.

    Args:
        type: The column's type.

    Returns:
        True for everything but a temporal column.
    """
    return not type.is_temporal()


def keeps_a_sign(type: LogicalType) -> Bool:
    """Whether that place belongs to the sign.

    Where it does, a negative value writes its minus into it rather than beside
    it, and the column's name is written one place in as well so that the name
    and the values line up. It is true for the types pandas calls numeric, which
    is the integers, the floats and the booleans, and the boolean being in that
    list is visible in exactly one thing: a boolean column named `available` is
    one wider than a text column named the same, because the name is held in by
    a sign that no boolean will ever print.

    Args:
        type: The column's type.

    Returns:
        True for integers, floats and booleans.
    """
    return type.is_numeric() or type == LogicalType.BOOL


def dots_for(width: Int) -> String:
    """The elision that fits in a column of the given width.

    pandas drops to two dots in a column of three or fewer, which is how a
    column of single digits comes to be elided by `..` while the column beside
    it is elided by `...`. The width it asks is the width the column was already
    padded to rather than the width of the widest value, so a column made wide
    by a long name gets three dots even though nothing in it is long.

    Args:
        width: The column width in bytes.

    Returns:
        Three dots, or two.
    """
    return String(ELLIPSIS) if width > 3 else String(SHORT_ELLIPSIS)


struct IndexCells(Copyable, Movable):
    """The labels a renderer prints down the left, already rendered.

    The renderers in this file cannot reach an `Index`, because `index.mojo`
    imports this file and the cycle would not resolve. So the index arrives
    already turned into text, which also keeps the cost right: only the labels
    that will be printed are ever rendered, so a million row frame renders
    eleven of them and not a million.

    An empty `cells` means the caller did not supply any and the renderer falls
    back to the row positions, which is what every caller used to get and is
    what the tests and the benchmarks still ask for.
    """

    var name: Optional[String]
    """What the level is called. `None` prints no name line at all, which is
    what an index that was never named has."""

    var cells: List[String]
    """One entry per printed row, in order, with the elision already standing in
    for the rows that were left out."""

    def __init__(out self):
        """Constructs the absence: no name, and positions down the side."""
        self.name = None
        self.cells = List[String]()

    def __init__(out self, var name: Optional[String], var cells: List[String]):
        """Constructs the labels for one rendering.

        Args:
            name: The level name, or `None` for unnamed.
            cells: The rendered labels, one per printed row.
        """
        self.name = name^
        self.cells = cells^


def not_a_number(value: Float64) -> Bool:
    """Whether a float is a NaN.

    Read out of the bit pattern rather than compared against, because a NaN
    compares equal to nothing including itself.

    Args:
        value: The number.

    Returns:
        True for a NaN of either sign.
    """
    return (
        value.to_bits[DType.uint64]() & 0x7FFF_FFFF_FFFF_FFFF
    ) > 0x7FF0_0000_0000_0000


def infinite(value: Float64) -> Bool:
    """Whether a float is an infinity.

    Read out of the bit pattern for the same reason, since an infinity has no
    literal to compare against that does not itself need constructing.

    Args:
        value: The number.

    Returns:
        True for an infinity of either sign.
    """
    return (
        value.to_bits[DType.uint64]() & 0x7FFF_FFFF_FFFF_FFFF
    ) == 0x7FF0_0000_0000_0000


def negative(value: Float64) -> Bool:
    """Whether a float carries a minus.

    The sign bit rather than a comparison against zero, so a negative zero is
    negative here and prints as `-0.0`, which is what pandas prints.

    Args:
        value: The number.

    Returns:
        True when the sign bit is set.
    """
    return (value.to_bits[DType.uint64]() >> 63) != 0


def fixed_text(value: Float64, precision: Int) -> String:
    """One value in fixed notation, at exactly the precision asked for.

    Nothing is taken off the end in here, because whether a place comes off is
    a decision about the column rather than about the value. `strip_places` is
    where that happens.

    The rounding is to nearest and ties go to the even digit, which is what C's
    own formatting does and therefore what pandas prints. It shows up on the
    values whose halves are exact, so `0.0078125` at six places is `0.007812`
    and not `0.007813`, and `2.5` at none is `2` and not `3`.

    Args:
        value: The number.
        precision: Decimal places to write. None writes no point either, which
            is what `%.0f` does.

    Returns:
        The formatted number.
    """
    if not_a_number(value):
        return String("NaN")
    if infinite(value):
        return String("-inf") if negative(value) else String("inf")
    var sign = String("-") if negative(value) else String("")
    var magnitude = -value if negative(value) else value
    var whole = Int(magnitude)
    if precision < 1:
        var carry = magnitude - Float64(whole)
        if carry > 0.5 or (carry == 0.5 and whole % 2 == 1):
            whole += 1
        return String(sign, whole)

    var scale = 1
    for _ in range(precision):
        scale *= 10
    var rest = (magnitude - Float64(whole)) * Float64(scale)
    var scaled = Int(rest)
    var carry = rest - Float64(scaled)
    if carry > 0.5 or (carry == 0.5 and scaled % 2 == 1):
        scaled += 1
    if scaled >= scale:
        whole += 1
        scaled -= scale

    var digits = String(scaled)
    while digits.byte_length() < precision:
        digits = String("0", digits)
    return String(sign, whole, ".", digits)


def scientific_text(value: Float64, precision: Int) -> String:
    """One value in scientific notation, at exactly the precision asked for.

    The mantissa is brought into the range it prints in by dividing or
    multiplying a power of ten at a time rather than by one factor worked out
    in advance, because a single factor of 1e310 is not a number and because
    each step on its own is exact. The powers halve, so nine of them cover
    every exponent a float has.

    Args:
        value: The number.
        precision: Decimal places in the mantissa.

    Returns:
        The formatted number, with the exponent signed and at least two digits
        wide, which is the width C prints and pandas inherits.
    """
    if not_a_number(value):
        return String("NaN")
    if infinite(value):
        return String("-inf") if negative(value) else String("inf")
    var sign = String("-") if negative(value) else String("")
    var magnitude = -value if negative(value) else value
    var exponent = 0
    if magnitude > 0.0:
        var scales = [
            1.0e256,
            1.0e128,
            1.0e64,
            1.0e32,
            1.0e16,
            1.0e8,
            1.0e4,
            1.0e2,
            1.0e1,
        ]
        var steps = [256, 128, 64, 32, 16, 8, 4, 2, 1]
        for k in range(len(scales)):
            if magnitude >= scales[k]:
                magnitude /= scales[k]
                exponent += steps[k]
        for k in range(len(scales)):
            if magnitude * scales[k] < 10.0:
                magnitude *= scales[k]
                exponent -= steps[k]
        # A subnormal loses digits on the way up and the steps above can land
        # a hair outside the range on either side, so the last place is walked.
        while magnitude >= 10.0:
            magnitude /= 10.0
            exponent += 1
        while magnitude < 1.0:
            magnitude *= 10.0
            exponent -= 1

    var digits = fixed_text(magnitude, precision)
    # The rounding can carry past ten, where 9.9999999 becomes 10.000000 and
    # the exponent takes the extra place instead.
    if digits.startswith("10"):
        exponent += 1
        digits = fixed_text(magnitude / 10.0, precision)

    var mark = String("e+") if exponent >= 0 else String("e-")
    var size = -exponent if exponent < 0 else exponent
    var power = String(size)
    if power.byte_length() < 2:
        power = String("0", power)
    return String(sign, digits, mark, power)


def plain_number(text: String) -> Bool:
    """Whether a rendering is a number with a decimal point in it.

    Which is what decides whether a place comes off the end of it when the
    column is stripped. A null, a NaN and an infinity are not numbers by this
    test and neither is anything in scientific notation, which is why a
    scientific column keeps all six of its places where a fixed one does not.

    Args:
        text: A rendered value.

    Returns:
        True for an optional minus, digits, a point, and digits or nothing.
    """
    var bytes = text.as_bytes()
    var at = 0
    if at < len(bytes) and bytes[at] == MINUS:
        at += 1
    var seen = 0
    while at < len(bytes) and bytes[at] >= ZERO and bytes[at] <= NINE:
        at += 1
        seen += 1
    if seen == 0 or at >= len(bytes) or bytes[at] != POINT:
        return False
    at += 1
    while at < len(bytes) and bytes[at] >= ZERO and bytes[at] <= NINE:
        at += 1
    return at == len(bytes)


def strip_places(mut cells: List[String]):
    """Takes the trailing zeros off a whole column at once.

    A place comes off every number in the column or off none of them, which is
    why a column of `1234567.125` and `2.0` prints the second as `2.000`. Doing
    it per value would line the two up on nothing and is not what pandas does.
    One place always survives, so an integral value prints as `2.0` rather than
    `2.` or `2`, which is what keeps a float column visibly a float column.

    Args:
        cells: The column, rewritten in place.
    """
    while True:
        var numbers = 0
        var kept = False
        for i in range(len(cells)):
            if plain_number(cells[i]):
                numbers += 1
                if not cells[i].endswith("0"):
                    kept = True
        if numbers == 0 or kept:
            break
        for i in range(len(cells)):
            if plain_number(cells[i]):
                var shorter = String(
                    StringSlice(
                        unsafe_from_utf8=cells[i].as_bytes()[
                            : cells[i].byte_length() - 1
                        ]
                    )
                )
                cells[i] = shorter^
    for i in range(len(cells)):
        if cells[i].endswith(".") and plain_number(cells[i]):
            cells[i] += "0"


def float_column(
    values: List[Float64],
    present: List[Bool],
    precision: Int,
    null_text: String,
) -> List[String]:
    """Renders a float column, which is the only way pandas renders one.

    The format is picked for the column rather than for the value. Everything
    is written fixed first, the zeros come off the column as a whole, and then
    the column is measured: if the longest of them is more than six characters
    past the precision and anything in it is larger than 1e6, or if anything in
    it is smaller than the last place being printed, the whole column is
    written again in scientific notation and nothing comes off it.

    The length that gets measured counts the place kept in front of the value,
    so a negative and a positive of the same width count the same.

    A magnitude at or above 1e15 takes the scientific branch without being
    written fixed first. That is not a rule of its own: a number that large has
    sixteen digits in front of the point, which is past the length that sends a
    column scientific anyway, and it is also past where an `Int` can hold the
    digits to write it with.

    Args:
        values: The numbers, one per printed row.
        present: Whether each of them is a value rather than a null.
        precision: Decimal places.
        null_text: What a null prints as.

    Returns:
        One rendering per value.
    """
    var scale = 1.0
    for _ in range(precision):
        scale *= 10.0
    var smallest = 1.0 / scale

    var scientific = False
    var large = False
    for i in range(len(values)):
        if not present[i] or not_a_number(values[i]) or infinite(values[i]):
            continue
        var magnitude = -values[i] if negative(values[i]) else values[i]
        if magnitude > 1.0e6:
            large = True
        if magnitude >= 1.0e15 or (magnitude > 0.0 and magnitude < smallest):
            scientific = True

    var cells = List[String](capacity=len(values))
    if not scientific:
        for i in range(len(values)):
            cells.append(
                fixed_text(values[i], precision) if present[
                    i
                ] else null_text.copy()
            )
        strip_places(cells)
        var longest = 0
        for i in range(len(cells)):
            var width = cells[i].byte_length()
            if not cells[i].startswith("-"):
                width += 1
            if width > longest:
                longest = width
        if longest > precision + 6 and large:
            scientific = True
        if not scientific:
            return cells^

    cells = List[String](capacity=len(values))
    for i in range(len(values)):
        cells.append(
            scientific_text(values[i], precision) if present[
                i
            ] else null_text.copy()
        )
    return cells^


def format_float(value: Float64, precision: Int) -> String:
    """Formats one float the way a column holding only it would be formatted.

    Which is the honest way to write this: the rule belongs to the column, and
    a lone value is a column of one. It is what the places that really do have
    one value call, and everything that has a column calls `float_column`.

    Args:
        value: The number.
        precision: Decimal places to round to. Trailing zeros are removed
            afterwards, so 1.5 at six places is `1.5` and not `1.500000`.

    Returns:
        The formatted number.
    """
    return float_column([value], [True], precision, String(""))[0]


def render_value(col: AnyArray, i: Int, options: DisplayOptions) -> String:
    """Renders one cell.

    Args:
        col: The column.
        i: The row. Must be less than the column's length.
        options: How to spell a null and how to round a float.

    Returns:
        The cell text.
    """
    if not col.is_valid(i):
        return options.null_text
    # A string's physical dtype is uint8, so falling through to the dispatch
    # below would print the first byte of the value as a number.
    #
    # The text is not quoted and not escaped. pandas does not quote either, and a
    # table is read by a person rather than parsed, so a value containing a
    # newline is a display problem that quoting would not fix anyway. `to_csv` is
    # where escaping belongs and is where it happens.
    if col.is_string():
        try:
            return col.strings()[i]
        except:
            return String("<", col.type, ">")
    if col.type.is_variable_width():
        return String("<", col.type, ">")
    # Before the layout dispatch, because a date is laid out as an int32 and a
    # timestamp as an int64, and falling through would print the day count
    # instead of the day.
    var instant = instant_text(col, i)
    if instant:
        return instant.take()
    comptime for candidate in ALL:
        if col.dtype() == candidate:
            var value = (
                col.unsafe_ptr[candidate]().unsafe_offset(i).unsafe_load()
            )
            comptime if candidate.is_floating_point():
                return format_float(
                    value.cast[DType.float64](), options.float_precision
                )
            return String(value)
    return String("?")


def float_at(col: AnyArray, i: Int) -> Float64:
    """The value in a float column, widened to the widest float there is.

    Args:
        col: The column. Nothing else in here asks whether it holds floats, so
            a caller that passes something else gets a zero.
        i: The row. Must be less than the column's length.

    Returns:
        The value.
    """
    comptime for candidate in ALL:
        if col.dtype() == candidate:
            comptime if candidate.is_floating_point():
                return (
                    col.unsafe_ptr[candidate]()
                    .unsafe_offset(i)
                    .unsafe_load()
                    .cast[DType.float64]()
                )
    return 0.0


def render_values(
    col: AnyArray, rows: List[Int], options: DisplayOptions
) -> List[String]:
    """Renders the cells of one column, which is the unit a float is decided in.

    Every other type is rendered a value at a time and this is a loop over
    `render_value`. A float column is not: which format it takes is a fact
    about the column, and it is a fact about the part of the column that will
    be printed, so a value in the elided middle cannot push the values around
    it into scientific notation. That is pandas' order too, which elides first
    and formats second.

    Args:
        col: The column.
        rows: The rows about to be printed, with -1 standing for the elision.
        options: How to spell a null and how to round a float.

    Returns:
        One cell per row, with an empty string where the elision goes.
    """
    var cells = List[String](capacity=len(rows))
    if not col.dtype().is_floating_point():
        for i in range(len(rows)):
            if rows[i] < 0:
                cells.append(String(""))
            else:
                cells.append(render_value(col, rows[i], options))
        return cells^

    var values = List[Float64]()
    var present = List[Bool]()
    for i in range(len(rows)):
        if rows[i] < 0:
            continue
        var valid = col.is_valid(rows[i])
        values.append(float_at(col, rows[i]) if valid else 0.0)
        present.append(valid)

    var made = float_column(
        values, present, options.float_precision, options.null_text
    )
    var at = 0
    for i in range(len(rows)):
        if rows[i] < 0:
            cells.append(String(""))
            continue
        cells.append(made[at].copy())
        at += 1
    return cells^


def visible(n: Int, limit: Int) -> List[Int]:
    """Chooses which positions to print, with -1 standing for the elision.

    Args:
        n: How many there are.
        limit: How many may be printed.

    Returns:
        Positions in order, with a single -1 in the middle if anything was left
        out.
    """
    var out = List[Int]()
    if limit < 2 or n <= limit:
        for i in range(n):
            out.append(i)
        return out^

    var head = limit // 2
    var tail = limit - head
    for i in range(head):
        out.append(i)
    out.append(-1)
    for i in range(n - tail, n):
        out.append(i)
    return out^


def render_table[
    o: ImmOrigin
](
    schema: Schema,
    columns: ColumnRefs[o],
    rows: Int,
    options: DisplayOptions,
    index: IndexCells = IndexCells(),
) -> String:
    """Renders a frame as a table with a header, an index and a shape line.

    The cells are built into a column major grid first and the widths measured
    off it, because a column's width is the widest thing in it including its own
    name, and that is not known until every cell that will be printed exists.
    Only the cells that will be printed are ever built, so a million row frame
    renders eleven values per column and not a million.

    Args:
        schema: The column names, in order.
        columns: The data, one per schema field.
        rows: The frame's height.
        options: How much to print.
        index: The row labels, rendered. Left out means the row positions.

    Returns:
        The table, with no trailing newline.
    """
    if len(columns) == 0:
        return String("Empty DataFrame\n\n[", rows, " rows x 0 columns]")

    var shown_rows = visible(rows, options.max_rows)
    var shown_columns = visible(len(columns), options.max_columns)
    # A caller that got the count wrong would index past the end of its own
    # list, so the positions are used instead. Nothing in the library does this
    # and the check costs one comparison.
    var labelled = len(index.cells) == len(shown_rows)
    var named = Bool(index.name)
    # The elided row is left blank on the way through and filled in at the end,
    # because how many dots a cell gets and where they sit in it both depend on
    # how wide the column turned out to be, which is not known yet.
    var dots_row = -1
    var dots_column = -1

    var grid = List[List[String]]()

    var labels = List[String]()
    labels.append(String(""))
    if named:
        labels.append(index.name.value())
    for i in range(len(shown_rows)):
        if shown_rows[i] < 0:
            dots_row = len(labels)
            labels.append(String(""))
        elif labelled:
            labels.append(index.cells[i])
        else:
            labels.append(String(shown_rows[i]))
    var height = len(labels)
    grid.append(labels^)

    for c in range(len(shown_columns)):
        var at = shown_columns[c]
        var cells = List[String]()
        if at < 0:
            # The elided column is three dots in every row including the header
            # and the name, and it is four wide rather than three, both of which
            # are what pandas prints.
            dots_column = len(grid)
            for _ in range(height):
                cells.append(String(ELLIPSIS))
            grid.append(cells^)
            continue
        var type = columns[at][].type
        var indented = keeps_a_place(type)
        var signed = keeps_a_sign(type)
        cells.append(
            String(" ", schema[at].name) if signed else schema[at].name
        )
        if named:
            cells.append(String(""))
        var made = render_values(columns[at][], shown_rows, options)
        for i in range(len(shown_rows)):
            if shown_rows[i] < 0:
                cells.append(String(""))
                continue
            var value = made[i].copy()
            if not indented or (signed and value.startswith("-")):
                cells.append(value^)
            else:
                cells.append(String(" ", value))
        grid.append(cells^)

    var widths = List[Int]()
    for c in range(len(grid)):
        var width = 4 if c == dots_column else 0
        for r in range(len(grid[c])):
            if grid[c][r].byte_length() > width:
                width = grid[c][r].byte_length()
        widths.append(width)

    if dots_row >= 0:
        for c in range(len(grid)):
            grid[c][dots_row] = dots_for(widths[c])

    var out = String("")
    for r in range(height):
        for c in range(len(grid)):
            if c == 0:
                out += pad_right(grid[c][r], widths[c])
            else:
                # The separator is one space and the place kept in front of the
                # value is the other, which is why this pads to one more than
                # the width rather than writing a gap and then padding.
                out += pad_left(grid[c][r], widths[c] + 1)
        out += "\n"

    out += String("\n[", rows, " rows x ", len(columns), " columns]")
    return out^


def render_column(
    name: String,
    col: AnyArray,
    options: DisplayOptions,
    index: IndexCells = IndexCells(),
) -> String:
    """Renders a single column as a two column listing with a footer.

    Args:
        name: The column name. An empty name leaves the `Name:` part out, as
            pandas does.
        col: The data.
        options: How much to print.
        index: The row labels, rendered. Left out means the row positions.

    Returns:
        The listing, with no trailing newline.
    """
    var dtype = String(col.type)
    var footer = String("")
    # The name comes before the length, which is pandas' order and reads as the
    # sentence it is: what this column is called, then how much of it there is.
    if name.byte_length() > 0:
        footer += String("Name: ", name, ", ")
    if len(col) > options.max_rows:
        footer += String("Length: ", len(col), ", ")
    footer += String("dtype: ", dtype)

    if len(col) == 0:
        return String("Series([], ", footer, ")")

    var shown = visible(len(col), options.max_rows)
    var labelled = len(index.cells) == len(shown)
    var indented = keeps_a_place(col.type)
    var signed = keeps_a_sign(col.type)
    var dots_at = -1
    var labels = List[String]()
    var cells = List[String]()
    var made = render_values(col, shown, options)
    for i in range(len(shown)):
        if shown[i] < 0:
            # The label on an elided row is blank on a column, where a frame
            # puts dots in it. Neither is a decision of ours.
            dots_at = len(labels)
            labels.append(String(""))
            cells.append(String(""))
            continue
        labels.append(index.cells[i] if labelled else String(shown[i]))
        var value = made[i].copy()
        if not indented or (signed and value.startswith("-")):
            cells.append(value^)
        else:
            cells.append(String(" ", value))

    var index_width = 0
    var cell_width = 0
    for i in range(len(labels)):
        if labels[i].byte_length() > index_width:
            index_width = labels[i].byte_length()
        if cells[i].byte_length() > cell_width:
            cell_width = cells[i].byte_length()

    # Centred rather than right aligned, which is what pandas does here and not
    # what it does on a frame.
    if dots_at >= 0:
        cells[dots_at] = pad_middle(dots_for(cell_width), cell_width)

    # The name goes on a line of its own above the listing and is not padded to
    # the width of the labels under it, which is pandas' layout. On a frame the
    # same name goes inside the table instead, because there is a header row
    # there for it to sit under and here there is not.
    var out = String(index.name.value(), "\n") if index.name else String("")
    for i in range(len(labels)):
        out += pad_right(labels[i], index_width)
        # Three spaces and the place kept in front of the value, where a frame
        # keeps one and the place.
        out += pad_left(cells[i], cell_width + 3)
        out += "\n"
    out += footer
    return out^
