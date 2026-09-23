"""How a column's values are laid out, apart from what they mean.

`LogicalType` answers what a value is: a string, an int64, a date. This answers
how the column holds it. Today there is one answer, flat, meaning the values sit
in their buffers one per row in the order the rows are in, which is what every
column in the library has always been. Issue #979 is the reason there is a field
for it at all.

The plan there is that a string column with seventeen distinct values can be
held as four byte codes into a dictionary and still say `dtype` string, and that
a filter can hand back the positions it kept rather than a copy of the rows. Both
of those are a different layout of the same logical type, and neither fits in
`LogicalType` without the user seeing a dtype change, which is what
`astype("category")` does today and is exactly the thing a storage decision must
not do. So the layout gets its own field, and a kernel that has only been taught
the flat layout can ask for it and decode anything else first.

Nothing sets anything but flat yet. The field comes first on its own so that the
change adding the second encoding is about that encoding and not also about the
struct it lives in.
"""


struct Encoding(Equatable, ImplicitlyCopyable, Movable, Writable):
    """The physical layout of a column, separate from its logical type."""

    var code: UInt64
    """Which layout.

    Eight bytes for what would fit in one, and the width is not an accident.
    This field took the place of eight bytes of padding on `AnyArray` that are
    there to stop the struct ending short of its alignment, which Mojo
    miscompiles when the struct is held in an `Optional`. A one byte code would
    put seven bytes of that padding straight back. `AnyArray._slack`'s old
    docstring, now on `AnyArray.encoding`, has the whole story, and
    `test_a_column_has_no_trailing_padding` checks the sum.
    """

    comptime FLAT = Self(0)
    """One value per row in row order, in the buffers `LogicalType` describes.
    Every column is this today."""

    def __init__(out self, code: UInt64):
        """Names a layout by its code.

        Args:
            code: The code.
        """
        self.code = code

    def __eq__(self, other: Self) -> Bool:
        """Compares two encodings.

        Args:
            other: The encoding to compare against.

        Returns:
            True if they are the same layout.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two encodings for inequality.

        Args:
            other: The encoding to compare against.

        Returns:
            True if they are different layouts.
        """
        return self.code != other.code

    def write_to(self, mut writer: Some[Writer]):
        """Writes the encoding's name.

        Args:
            writer: The destination.
        """
        if self == Self.FLAT:
            writer.write("flat")
        else:
            writer.write("encoding ", self.code)
