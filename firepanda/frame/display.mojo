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

**Floats print to six significant decimals and drop trailing zeros.** Mojo prints
a `Float64` at the shortest representation that round trips, which for one third
is seventeen characters and makes a table unreadable. Six decimals is the pandas
default. Anything at or above 1e15, or below 1e-4 without being zero, falls back
to Mojo's own formatting, which switches to an exponent, again as pandas does.

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


def format_float(value: Float64, precision: Int) -> String:
    """Formats a float the way a table wants it rather than the way it is stored.

    The special values are read out of the bit pattern rather than compared
    against, because a NaN compares equal to nothing including itself and an
    infinity has no literal to compare against that does not itself need
    constructing.

    Args:
        value: The number.
        precision: Decimal places to round to. Trailing zeros are removed
            afterwards, so 1.5 at six places is `1.5` and not `1.500000`.

    Returns:
        The formatted number.
    """
    var bits = value.to_bits[DType.uint64]()
    var negative = (bits >> 63) != 0
    var rest = bits & 0x7FFF_FFFF_FFFF_FFFF
    if rest > 0x7FF0_0000_0000_0000:
        return String("NaN")
    if rest == 0x7FF0_0000_0000_0000:
        return String("-inf") if negative else String("inf")

    var magnitude = -value if negative else value
    # Outside this range a fixed point rendering is either wrong or useless: the
    # integer part stops fitting in an Int on one side, and on the other the
    # first significant digit is past the last place being printed. Mojo's own
    # formatting switches to an exponent, which is what pandas does here too.
    if magnitude >= 1.0e15 or (magnitude > 0.0 and magnitude < 1.0e-4):
        return String(value)
    if precision < 1:
        return String("-", Int(magnitude + 0.5)) if negative else String(
            Int(magnitude + 0.5)
        )

    var scale = 1
    for _ in range(precision):
        scale *= 10

    var whole = Int(magnitude)
    var scaled = Int((magnitude - Float64(whole)) * Float64(scale) + 0.5)
    if scaled >= scale:
        whole += 1
        scaled -= scale

    # Strip trailing zeros by dividing them out, which also tells us how many
    # digits are left to pad to. One place always survives, so an integral value
    # prints as `2.0` rather than `2.` or `2`, which is what keeps a float column
    # visibly a float column.
    var places = precision
    while places > 1 and scaled % 10 == 0:
        scaled //= 10
        places -= 1

    var digits = String(scaled)
    while digits.byte_length() < places:
        digits = String("0", digits)

    var sign = String("-") if negative else String("")
    return String(sign, whole, ".", digits)


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
        for i in range(len(shown_rows)):
            if shown_rows[i] < 0:
                cells.append(String(""))
                continue
            var value = render_value(columns[at][], shown_rows[i], options)
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
    for i in range(len(shown)):
        if shown[i] < 0:
            # The label on an elided row is blank on a column, where a frame
            # puts dots in it. Neither is a decision of ours.
            dots_at = len(labels)
            labels.append(String(""))
            cells.append(String(""))
            continue
        labels.append(index.cells[i] if labelled else String(shown[i]))
        var value = render_value(col, shown[i], options)
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
