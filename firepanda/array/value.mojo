"""One element, carrying the type it is.

A column is a `Value` repeated, so this is the type-erased scalar the same way
`AnyArray` is the type-erased column, and it exists for the same reason: at the
frame boundary a dtype stops being a parameter and starts being a field.

Three things want it and all three were blocked on it. A reduction over a column
answers one element and has to say what type that element is and whether it is
present at all, because the sum of an empty column is not zero, it is nothing.
An expression with a constant in it, which is every `x > 5` anybody writes,
needs the constant to arrive at a kernel as a value with a type. And filling
nulls needs a value to fill them with.

The payload is a union spelled out as fields, the tag being the type field that
is there anyway. An integer or a bool lives in `bits` in the bit pattern of its
own dtype, and the sign is put back by the reader rather than kept in the store,
so a uint64 above the top of int64 round trips and so does a negative int64. A
float lives in `real` at full width, so a float32 that went in exact comes out
exact. Text lives in `text`, and it is the one that allocates, which is why it
is an `Optional` rather than an always-present empty string.

`present` is not a nicety. A null is a real answer and it is not a zero: pandas
answers NaN for the mean of an empty column and 0.0 for the mean of a column
holding one zero, and those two have to be different values here or the
difference is lost before anybody can look at it.
"""

from firepanda.dtype.lists import FLOAT, SIGNED, contains
from firepanda.dtype.logical import LogicalType, TypeKind, logical_for


struct Value(Copyable, Equatable, Movable, Writable):
    """One element of a column, with its type carried as a field."""

    var bits: UInt64
    """An integer or a bool, in the bit pattern of its own dtype."""

    var real: Float64
    """A float, at full width whatever width it came in at."""

    var text: Optional[String]
    """The bytes, for a string or binary value."""

    var type: LogicalType
    """The type of the value, which says which of the fields above to read."""

    var present: Bool
    """Whether there is a value at all. False is a null and it is an answer."""

    def __init__[dt: DType](out self, value: Scalar[dt]):
        """Constructs a value from an element whose dtype the caller has.

        The dtype is kept rather than widened, so a float32 stays a float32 and
        an expression built on it promotes the way it would have promoted
        against a float32 column.

        Args:
            value: The element.

        Parameters:
            dt: The dtype.
        """
        comptime if contains[FLOAT](dt):
            self.bits = 0
            self.real = value.cast[DType.float64]()
        else:
            self.bits = value.cast[DType.uint64]()
            self.real = 0.0
        self.text = None
        self.type = logical_for(dt)
        self.present = True

    def __init__(out self, value: Bool):
        """Constructs a bool value.

        `Bool` is its own type rather than a one lane SIMD, so it needs its own
        constructor rather than falling into the one above.

        Args:
            value: The flag.
        """
        self.bits = 1 if value else 0
        self.real = 0.0
        self.text = None
        self.type = LogicalType.BOOL
        self.present = True

    def __init__(out self, var value: String):
        """Constructs a text value.

        Args:
            value: The bytes. Consumed.
        """
        self.bits = 0
        self.real = 0.0
        self.text = value^
        self.type = LogicalType.STRING
        self.present = True

    def __init__(out self, *, null: LogicalType):
        """Constructs the absent value of a type.

        Args:
            null: The type the missing element would have had.
        """
        self.bits = 0
        self.real = 0.0
        self.text = None
        self.type = null
        self.present = False

    def __init__(out self, *, copy: Self):
        """Copies a value.

        Args:
            copy: The value to copy.
        """
        self.bits = copy.bits
        self.real = copy.real
        self.text = Optional[String](copy=copy.text)
        self.type = copy.type
        self.present = copy.present

    def __eq__(self, other: Self) -> Bool:
        """Compares two values, type and all.

        Two nulls of the same type are equal here, which is not what SQL says
        about nulls in a predicate and is what a test comparing two answers
        wants. Nothing builds a predicate out of this.

        Args:
            other: The value to compare against.

        Returns:
            True if the types match and either both are absent or both hold the
            same element.
        """
        if self.type != other.type or self.present != other.present:
            return False
        if not self.present:
            return True
        if self.type.is_variable_width():
            return self.text.value() == other.text.value()
        if self.type.is_float():
            return self.real == other.real
        return self.bits == other.bits

    def __ne__(self, other: Self) -> Bool:
        """Compares two values for inequality.

        Args:
            other: The value to compare against.

        Returns:
            True if they differ.
        """
        return not (self == other)

    def is_null(self) -> Bool:
        """Reports whether the value is absent.

        Returns:
            True for a null.
        """
        return not self.present

    def as_scalar[dt: DType](self) -> Scalar[dt]:
        """Reads the element at a dtype, converting across families if it must.

        A null reads as zero, which is the rule the column layout already
        follows: a null position holds a zero in the values buffer and the
        validity says so separately. A caller that cares asks `is_null` first.

        The signed case is the one that has to be spelled out. The store keeps
        an int8 of minus five as the bit pattern of minus five widened, so
        reading it straight out as a float would answer eighteen quintillion.
        Going back through int64 first is what puts the sign back.

        Parameters:
            dt: The dtype to read it as.

        Returns:
            The element, converted the way a cast would convert it.
        """
        if self.type.is_float():
            return self.real.cast[dt]()
        if contains[SIGNED](self.type.physical):
            return self.bits.cast[DType.int64]().cast[dt]()
        return self.bits.cast[dt]()

    def as_string(self) raises -> String:
        """Reads the element as text.

        Returns:
            The bytes.

        Raises:
            If the value is not text. There is no conversion here, because a
            number becomes text through the cast kernel, which allocates and so
            has to be asked for rather than happening in an accessor.
        """
        if not self.type.is_variable_width():
            raise Error("value: " + String(self.type) + " is not text")
        if not self.present:
            return String()
        return String(self.text.value())

    def write_to(self, mut writer: Some[Writer]):
        """Writes the element the way a frame prints one.

        Args:
            writer: The destination.
        """
        if not self.present:
            writer.write("null")
        elif self.type.is_variable_width():
            writer.write(self.text.value())
        elif self.type.kind == TypeKind.BOOL:
            writer.write("true" if self.bits != 0 else "false")
        elif self.type.is_float():
            writer.write(self.real)
        else:
            writer.write(self.as_scalar[DType.int64]())
