"""Field and Schema.

A `Schema` is the ordered list of columns a frame has, with their names and
types. It is a value: taking a projection of a schema produces a new schema and
leaves the original alone. Nothing in this file allocates per row, so a schema is
cheap to copy around a query plan.
"""

from .logical import LogicalType


@fieldwise_init
struct Field(Copyable, Equatable, Movable, Writable):
    """One named, typed column position in a schema."""

    var name: String
    """The column name. Duplicates are legal; pandas allows them."""

    var dtype: LogicalType
    """The column type."""

    var nullable: Bool
    """Whether the column is allowed to contain nulls."""

    def __init__(out self, name: String, dtype: LogicalType):
        """Constructs a nullable field, which is the pandas default.

        Args:
            name: The column name.
            dtype: The column type.
        """
        self.name = name
        self.dtype = dtype
        self.nullable = True

    def __eq__(self, other: Self) -> Bool:
        """Compares two fields.

        Args:
            other: The field to compare against.

        Returns:
            True if name, type and nullability all match.
        """
        return (
            self.name == other.name
            and self.dtype == other.dtype
            and self.nullable == other.nullable
        )

    def __ne__(self, other: Self) -> Bool:
        """Compares two fields for inequality.

        Args:
            other: The field to compare against.

        Returns:
            True if the fields differ.
        """
        return not (self == other)

    def write_to(self, mut writer: Some[Writer]):
        """Writes the field as `name: type`.

        Args:
            writer: The destination.
        """
        writer.write(self.name, ": ", self.dtype)
        if not self.nullable:
            writer.write(" not null")


struct Schema(Copyable, Equatable, Movable, Sized, Writable):
    """The ordered, named, typed shape of a frame."""

    var fields: List[Field]
    """The columns, in position order."""

    def __init__(out self):
        """Constructs an empty schema."""
        self.fields = List[Field]()

    def __init__(out self, var fields: List[Field]):
        """Constructs a schema from a list of fields.

        Args:
            fields: The columns, in position order.
        """
        self.fields = fields^

    def __init__(out self, *, copy: Self):
        """Copies a schema.

        Args:
            copy: The schema to copy.
        """
        self.fields = copy.fields.copy()

    def __len__(self) -> Int:
        """Returns the number of columns.

        Returns:
            The column count.
        """
        return len(self.fields)

    def __getitem__(ref self, i: Int) -> ref[self.fields[i]] Field:
        """Returns the field at a position.

        Args:
            i: The column position.

        Returns:
            A reference to the field.
        """
        return self.fields[i]

    def __eq__(self, other: Self) -> Bool:
        """Compares two schemas.

        Args:
            other: The schema to compare against.

        Returns:
            True if the schemas have the same fields in the same order.
        """
        if len(self.fields) != len(other.fields):
            return False
        for i in range(len(self.fields)):
            if self.fields[i] != other.fields[i]:
                return False
        return True

    def __ne__(self, other: Self) -> Bool:
        """Compares two schemas for inequality.

        Args:
            other: The schema to compare against.

        Returns:
            True if the schemas differ.
        """
        return not (self == other)

    def append(mut self, var field: Field):
        """Adds a column at the end.

        Args:
            field: The column to add.
        """
        self.fields.append(field^)

    def index_of(self, name: String) raises -> Int:
        """Returns the position of the first column with a name.

        Args:
            name: The column name.

        Returns:
            The position.

        Raises:
            If no column has that name.
        """
        for i in range(len(self.fields)):
            if self.fields[i].name == name:
                return i
        raise Error("no column named '" + name + "'")

    def index_of_all(self, names: List[String]) raises -> List[Int]:
        """Returns the positions of several columns, in the order asked for.

        The same answer `index_of` gives for each name and the same error when a
        name is missing. This exists so a caller that needs the positions can
        look them up once and then work in positions, not because one lookup
        here is cheaper than one `index_of`. It is still a scan per name.

        A hash of the names was the obvious thing to do instead and it is much
        slower. Building a map of the hits table's 105 column names costs more
        than the eleven thousand string comparisons a full width projection does
        without one, because almost every comparison stops on the first byte and
        every insertion hashes a whole name. The quadratic is real and it is
        cheap, so the saving worth having is doing the scan once rather than
        making it faster, which is what `select` and `drop` below use this for.

        Args:
            names: The column names to look up.

        Returns:
            One position per name, in the order asked for.

        Raises:
            If any name is not present.
        """
        var out = List[Int](capacity=len(names))
        for i in range(len(names)):
            out.append(self.index_of(names[i]))
        return out^

    def has(self, name: String) -> Bool:
        """Reports whether any column has a name.

        Args:
            name: The column name.

        Returns:
            True if at least one column matches.
        """
        for i in range(len(self.fields)):
            if self.fields[i].name == name:
                return True
        return False

    def select(self, names: List[String]) raises -> Self:
        """Returns a new schema with only the named columns, in the order given.

        Args:
            names: The column names to keep.

        Returns:
            The projected schema.

        Raises:
            If any name is not present.
        """
        return self.select_at(self.index_of_all(names))

    def select_at(self, at: List[Int]) raises -> Self:
        """Returns a new schema with only the columns at those positions.

        The positional half of `select`, so a caller that has already resolved
        the names does not resolve them a second time on the way through.

        Args:
            at: The positions to keep, in the order to keep them.

        Returns:
            The projected schema.

        Raises:
            If a position is outside the schema.
        """
        var out = List[Field](capacity=len(at))
        for i in range(len(at)):
            if at[i] < 0 or at[i] >= len(self.fields):
                raise Error("no column at position " + String(at[i]))
            out.append(self.fields[at[i]].copy())
        return Self(out^)

    def write_to(self, mut writer: Some[Writer]):
        """Writes the schema one field per line.

        Args:
            writer: The destination.
        """
        for i in range(len(self.fields)):
            if i > 0:
                writer.write("\n")
            writer.write(self.fields[i])
