"""Rendering a frame as text.

A dataframe you cannot look at is not a dataframe, and every one of the tests
written before this file existed had to reach through `as_typed` and read values
back one at a time to say what it saw. So this is worth having early even though
nothing depends on it.

The output follows pandas closely enough that a Python reader will not have to
think about it: a header row, an integer index down the left, values right
aligned in their columns, and both the rows and the columns elided in the middle
when there are too many to print. What it does not copy is the parts of pandas
display that exist because pandas has an index made of arbitrary labels. There is
no index name line, and the left column is always the row position.

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

from firepanda.array.any import AnyArray
from firepanda.dtype.lists import ALL
from firepanda.dtype.schema import Schema

comptime DEFAULT_MAX_ROWS = 10
"""Rows printed before the middle is elided. Half from each end."""

comptime DEFAULT_MAX_COLUMNS = 20
"""Columns printed before the middle is elided. Half from each end."""

comptime DEFAULT_FLOAT_PRECISION = 6
"""Decimal places a float is rounded to, trailing zeros removed after."""

comptime ELLIPSIS = "..."
"""What stands in for the rows and columns that were not printed."""


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


def render_table(
    schema: Schema,
    columns: List[AnyArray],
    rows: Int,
    options: DisplayOptions,
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

    Returns:
        The table, with no trailing newline.
    """
    if len(columns) == 0:
        return String("Empty DataFrame\n\n[", rows, " rows x 0 columns]")

    var shown_rows = visible(rows, options.max_rows)
    var shown_columns = visible(len(columns), options.max_columns)

    var grid = List[List[String]]()

    var index = List[String]()
    index.append(String(""))
    for i in range(len(shown_rows)):
        if shown_rows[i] < 0:
            index.append(String(ELLIPSIS))
        else:
            index.append(String(shown_rows[i]))
    grid.append(index^)

    for c in range(len(shown_columns)):
        var at = shown_columns[c]
        var cells = List[String]()
        if at < 0:
            for _ in range(len(shown_rows) + 1):
                cells.append(String(ELLIPSIS))
            grid.append(cells^)
            continue
        cells.append(schema[at].name)
        for i in range(len(shown_rows)):
            if shown_rows[i] < 0:
                cells.append(String(ELLIPSIS))
            else:
                cells.append(render_value(columns[at], shown_rows[i], options))
        grid.append(cells^)

    var widths = List[Int]()
    for c in range(len(grid)):
        var width = 0
        for r in range(len(grid[c])):
            if grid[c][r].byte_length() > width:
                width = grid[c][r].byte_length()
        widths.append(width)

    var out = String("")
    for r in range(len(grid[0])):
        for c in range(len(grid)):
            if c > 0:
                out += "  "
            out += pad_left(grid[c][r], widths[c])
        out += "\n"

    out += String("\n[", rows, " rows x ", len(columns), " columns]")
    return out^


def render_column(
    name: String, col: AnyArray, options: DisplayOptions
) -> String:
    """Renders a single column as a two column listing with a footer.

    Args:
        name: The column name. An empty name leaves the `Name:` part out, as
            pandas does.
        col: The data.
        options: How much to print.

    Returns:
        The listing, with no trailing newline.
    """
    var dtype = String(col.type)
    var footer = String("")
    if len(col) > options.max_rows:
        footer += String("Length: ", len(col), ", ")
    if name.byte_length() > 0:
        footer += String("Name: ", name, ", ")
    footer += String("dtype: ", dtype)

    if len(col) == 0:
        return String("Series([], ", footer, ")")

    var shown = visible(len(col), options.max_rows)
    var index = List[String]()
    var cells = List[String]()
    for i in range(len(shown)):
        if shown[i] < 0:
            index.append(String(ELLIPSIS))
            cells.append(String(ELLIPSIS))
        else:
            index.append(String(shown[i]))
            cells.append(render_value(col, shown[i], options))

    var index_width = 0
    var cell_width = 0
    for i in range(len(index)):
        if index[i].byte_length() > index_width:
            index_width = index[i].byte_length()
        if cells[i].byte_length() > cell_width:
            cell_width = cells[i].byte_length()

    var out = String("")
    for i in range(len(index)):
        out += pad_left(index[i], index_width)
        out += "    "
        out += pad_left(cells[i], cell_width)
        out += "\n"
    out += footer
    return out^
