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
        var out = List[Field]()
        for name in names:
            out.append(self.fields[self.index_of(name)].copy())
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
