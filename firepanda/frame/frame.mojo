"""The eager `DataFrame`.

A frame is a `Schema` and a list of `AnyArray`, and the invariant that makes it a
frame rather than a bag of columns is that every column has the same length and
every name is distinct. Both are checked once, at construction, and then relied on
everywhere else.

The schema is not a summary of the columns, it is a separate authority. A column
knows its physical dtype and nothing else; the schema knows the name, the logical
type and whether nulls are allowed. Keeping both means the two can disagree, so
the constructor checks that they do not, and every operation that changes one
changes the other in the same expression.

There is no index. `docs/specs/04-python-dx.md` lists that as the first
deliberate divergence from pandas and this is where it shows up: rows are
identified by position, `filter` and `take` are the only ways to address them, and
nothing aligns on labels because there are no labels to align on. The second
divergence is that nothing mutates. Every method returns a new frame.

The third thing worth knowing is what a copy costs. `select` copies the columns it
keeps, `filter` and `take` copy by construction because they build new columns
anyway, and `slice` copies bytes. `__getitem__` on a position hands back a
borrowed reference and does not copy; `column` on a name hands back a `Series` and
does. That split is not elegant and it is honest: a borrowing accessor needs the
index in the return type, which a name lookup cannot provide until the plan layer
at M4 turns column references into something resolved ahead of time.

None of this is lazy. `docs/specs/04-python-dx.md` describes the eager surface as
a facade over a plan, which is true from M4 onwards. At M1 the facade is the
implementation and every method runs when it is called.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.kernel.cast import cast_any
from firepanda.kernel.select import filter_any, take_any
from firepanda.kernel.sort import argsort_any_into, identity_permutation

from .series import Series, _check_range, _clamp, _to_positions


struct DataFrame(Copyable, Movable, Sized, Writable):
    """A set of equal length named columns, addressed by position."""

    var schema: Schema
    """The column names, logical types and nullability."""

    var columns: List[AnyArray]
    """The data, one entry per schema field and in the same order."""

    var rows: Int
    """The row count. Every column has this length."""

    def __init__(out self):
        """Constructs a frame with no columns and no rows."""
        self.schema = Schema()
        self.columns = List[AnyArray]()
        self.rows = 0

    def __init__(out self, var schema: Schema, var columns: List[AnyArray]):
        """Constructs a frame from parts that have already been checked.

        This is the unchecked constructor and it is what every method here uses
        to build its result, because a frame derived from a valid frame by a
        length preserving or column preserving operation is valid by
        construction. Code outside this file should go through `from_series`.

        Args:
            schema: The schema. Must match `columns` in length and order.
            columns: The data. Must all be the same length.
        """
        self.rows = 0 if len(columns) == 0 else len(columns[0])
        self.schema = schema^
        self.columns = columns^

    def __init__(out self, *, copy: Self):
        """Deep-copies a frame.

        Args:
            copy: The frame to copy.
        """
        self.schema = Schema(copy=copy.schema)
        self.columns = List[AnyArray](capacity=len(copy.columns))
        for i in range(len(copy.columns)):
            self.columns.append(AnyArray(copy=copy.columns[i]))
        self.rows = copy.rows

    @staticmethod
    def from_series(var series: List[Series]) raises -> Self:
        """Constructs a frame from named columns, checking the invariants.

        Args:
            series: The columns. Must all be the same length and have distinct
                names.

        Returns:
            A frame holding them.

        Raises:
            If two columns share a name or the lengths differ.
        """
        var fields = List[Field](capacity=len(series))
        var columns = List[AnyArray](capacity=len(series))
        var rows = 0 if len(series) == 0 else len(series[0])

        for i in range(len(series)):
            if len(series[i]) != rows:
                raise Error(
                    "every column must have the same number of rows; column 0"
                    " has "
                    + String(rows)
                    + " and column '"
                    + series[i].name
                    + "' has "
                    + String(len(series[i]))
                )
            for j in range(i):
                if series[j].name == series[i].name:
                    raise Error(
                        "duplicate column name '" + series[i].name + "'"
                    )
            fields.append(Field(series[i].name, series[i].logical()))

        # Popping from the back and then reversing, rather than iterating, so
        # that each column's buffers move rather than being copied. A loop over
        # `series^` binds a value with no origin and cannot give up a field, and
        # copying instead would double the memory of every frame at the moment it
        # is built. The quadratic shape does not matter: this is once per frame,
        # over columns, and a frame with a thousand columns has other problems.
        var backwards = List[AnyArray](capacity=len(series))
        while len(series) > 0:
            backwards.append(series.pop().into_values())
        while len(backwards) > 0:
            columns.append(backwards.pop())
        return Self(Schema(fields^), columns^)

    def __len__(self) -> Int:
        """Returns the number of rows.

        Returns:
            The row count.
        """
        return self.rows

    def width(self) -> Int:
        """Returns the number of columns.

        Returns:
            The column count.
        """
        return len(self.columns)

    def shape(self) -> Tuple[Int, Int]:
        """Returns the row and column counts, as pandas does.

        Returns:
            A pair of rows and columns.
        """
        return (self.rows, len(self.columns))

    def __getitem__(ref self, i: Int) -> ref[self.columns[i]] AnyArray:
        """Borrows a column by position, without copying it.

        Args:
            i: The column position. Must be less than the width.

        Returns:
            A reference to the column.
        """
        return self.columns[i]

    def index_of(self, name: String) raises -> Int:
        """Returns the position of a named column.

        Args:
            name: The column name.

        Returns:
            Its position.

        Raises:
            If no column has that name.
        """
        return self.schema.index_of(name)

    def has(self, name: String) -> Bool:
        """Reports whether a column exists.

        Args:
            name: The column name.

        Returns:
            True if a column has that name.
        """
        return self.schema.has(name)

    def names(self) -> List[String]:
        """Returns the column names in order.

        Returns:
            One name per column.
        """
        var out = List[String](capacity=len(self.schema))
        for i in range(len(self.schema)):
            out.append(self.schema[i].name)
        return out^

    def column(self, name: String) raises -> Series:
        """Returns a named column as a `Series`.

        This copies. Use `__getitem__` with a position when the copy matters and
        a borrow will do.

        Args:
            name: The column name.

        Returns:
            A copy of the column, carrying its name.

        Raises:
            If no column has that name.
        """
        var at = self.schema.index_of(name)
        return Series(name, AnyArray(copy=self.columns[at]))

    def select(self, names: List[String]) raises -> Self:
        """Returns a frame with only the named columns, in the order given.

        Naming a column twice is an error rather than a duplication, because the
        result would have two columns with one name and nothing downstream could
        address the second.

        Args:
            names: The columns to keep.

        Returns:
            A frame of the same height and the named columns.

        Raises:
            If a name is missing or repeated.
        """
        var columns = List[AnyArray](capacity=len(names))
        for i in range(len(names)):
            for j in range(i):
                if names[j] == names[i]:
                    raise Error(
                        "select names a column twice: '" + names[i] + "'"
                    )
            columns.append(
                AnyArray(copy=self.columns[self.schema.index_of(names[i])])
            )
        var out = Self(self.schema.select(names), columns^)
        out.rows = self.rows
        return out^

    def drop(self, names: List[String]) raises -> Self:
        """Returns a frame without the named columns.

        Args:
            names: The columns to remove. Each must exist.

        Returns:
            A frame with the remaining columns in their original order.

        Raises:
            If a name is missing.
        """
        for i in range(len(names)):
            _ = self.schema.index_of(names[i])

        var keep = List[String]()
        for i in range(len(self.schema)):
            var name = self.schema[i].name
            var dropped = False
            for j in range(len(names)):
                if names[j] == name:
                    dropped = True
            if not dropped:
                keep.append(name)
        return self.select(keep)

    def rename(self, old: String, new: String) raises -> Self:
        """Returns a frame with one column renamed.

        Args:
            old: The current name.
            new: The replacement. Must not already be in use.

        Returns:
            A renamed copy of the frame.

        Raises:
            If `old` is missing or `new` is taken by a different column.
        """
        var at = self.schema.index_of(old)
        if new != old and self.schema.has(new):
            raise Error("cannot rename to '" + new + "', that name is taken")

        var out = Self(copy=self)
        out.schema.fields[at].name = new
        return out^

    def with_column(self, var column: Series) raises -> Self:
        """Returns a frame with a column added or replaced.

        Replacing is by name and keeps the column's position, which is what
        makes `df.with_column(df.column("a").cast(...))` leave the frame looking
        the way it did.

        Args:
            column: The column. Must match the frame's height unless the frame
                has no columns yet.

        Returns:
            A frame carrying it.

        Raises:
            If the length does not match the frame's height.
        """
        if len(self.columns) > 0 and len(column) != self.rows:
            raise Error(
                "cannot add column '"
                + column.name
                + "' with "
                + String(len(column))
                + " rows to a frame of "
                + String(self.rows)
                + " rows"
            )

        var out = Self(copy=self)
        var field = Field(column.name, column.logical())
        var replacing = out.schema.has(column.name)
        var at = out.schema.index_of(column.name) if replacing else 0
        if replacing:
            out.schema.fields[at] = field^
            out.columns[at] = column^.into_values()
            return out^

        out.schema.append(field^)
        out.columns.append(column^.into_values())
        out.rows = len(out.columns[0])
        return out^

    def cast(self, name: String, to: DType) raises -> Self:
        """Returns a frame with one column converted to another dtype.

        Args:
            name: The column to convert.
            to: The target dtype.

        Returns:
            A frame with that column converted and the rest untouched.

        Raises:
            If the name is missing or either dtype has no physical layout.
        """
        var at = self.schema.index_of(name)
        var converted = cast_any(self.columns[at], to)

        var out = Self(copy=self)
        out.schema.fields[at] = Field(name, converted.type)
        out.columns[at] = converted^
        return out^

    def filter(self, mask: Array[DType.bool]) raises -> Self:
        """Returns the rows where the mask is true.

        A null in the mask drops the row, as in `Series.filter`.

        Args:
            mask: The mask. Must be as long as the frame is tall.

        Returns:
            A frame of the kept rows.

        Raises:
            If the mask length does not match, or a dtype has no physical
            layout.
        """
        if len(mask) != self.rows:
            raise Error(
                "filter mask must be as long as the frame is tall; frame has "
                + String(self.rows)
                + " rows and mask has "
                + String(len(mask))
            )
        var columns = List[AnyArray](capacity=len(self.columns))
        for i in range(len(self.columns)):
            columns.append(filter_any(self.columns[i], mask))
        return Self(Schema(copy=self.schema), columns^)

    def take(self, indices: List[Int]) raises -> Self:
        """Returns rows gathered by position.

        Args:
            indices: The rows to gather. A negative index produces a null row,
                which is how an outer join reports a row that was not there.

        Returns:
            A frame of `len(indices)` rows.

        Raises:
            If a dtype has no physical layout.
        """
        var columns = List[AnyArray](capacity=len(self.columns))
        for i in range(len(self.columns)):
            columns.append(take_any(self.columns[i], indices))
        var out = Self(Schema(copy=self.schema), columns^)
        out.rows = len(indices)
        return out^

    def slice(self, start: Int, end: Int) raises -> Self:
        """Returns a half-open range of rows.

        Args:
            start: The first row, inclusive.
            end: The last row, exclusive.

        Returns:
            A frame of `end - start` rows.

        Raises:
            If the range is reversed or runs past either end.
        """
        _check_range(start, end, self.rows, "frame")
        var columns = List[AnyArray](capacity=len(self.columns))
        for i in range(len(self.columns)):
            columns.append(self.columns[i].slice(start, end))
        var out = Self(Schema(copy=self.schema), columns^)
        out.rows = end - start
        return out^

    def head(self, n: Int = 5) raises -> Self:
        """Returns the first rows.

        Args:
            n: How many, clamped to the height.

        Returns:
            A frame of at most `n` rows.
        """
        return self.slice(0, _clamp(n, self.rows))

    def tail(self, n: Int = 5) raises -> Self:
        """Returns the last rows.

        Args:
            n: How many, clamped to the height.

        Returns:
            A frame of at most `n` rows.
        """
        return self.slice(self.rows - _clamp(n, self.rows), self.rows)

    def argsort(
        self,
        by: List[String],
        descending: List[Bool],
        nulls_first: List[Bool],
    ) raises -> Array[DType.uint32]:
        """Returns the row order that sorts the frame on a set of keys.

        The keys are applied from the last to the first, each refining what the
        one after it produced. Every pass is stable, so the earlier key stays
        dominant. This is `argsort_multi` with the columns borrowed out of the
        frame instead of copied into a list, which on a wide frame is the
        difference between reading the keys and duplicating them.

        Args:
            by: The key columns, most significant first.
            descending: One flag per key.
            nulls_first: One flag per key.

        Returns:
            A permutation of `[0, len(self))`.

        Raises:
            If the lists disagree in length, if `by` is empty, if a name is
            missing, or if a key dtype is not sortable.
        """
        if len(by) == 0:
            raise Error("sort needs at least one key column")
        if len(descending) != len(by) or len(nulls_first) != len(by):
            raise Error(
                "sort needs one descending and one nulls_first flag per key;"
                " got "
                + String(len(by))
                + " keys, "
                + String(len(descending))
                + " descending and "
                + String(len(nulls_first))
                + " nulls_first"
            )

        var at = List[Int](capacity=len(by))
        for i in range(len(by)):
            at.append(self.schema.index_of(by[i]))

        var order = identity_permutation(self.rows)
        for i in range(len(at) - 1, -1, -1):
            argsort_any_into(
                self.columns[at[i]], order, descending[i], nulls_first[i]
            )
        return order^

    def sort_values(
        self,
        by: List[String],
        descending: List[Bool],
        nulls_first: List[Bool],
    ) raises -> Self:
        """Returns the frame in sorted order.

        The permutation is widened to a take list once and every column is
        gathered with the same one, so a frame with twenty columns pays for the
        conversion once rather than twenty times.

        Args:
            by: The key columns, most significant first.
            descending: One flag per key.
            nulls_first: One flag per key.

        Returns:
            A sorted frame.

        Raises:
            As `argsort` does.
        """
        return self.take(
            _to_positions(self.argsort(by, descending, nulls_first))
        )

    def sort_by(
        self,
        by: String,
        descending: Bool = False,
        nulls_first: Bool = False,
    ) raises -> Self:
        """Returns the frame sorted on a single column.

        Args:
            by: The key column.
            descending: Largest first.
            nulls_first: Put the nulls at the front rather than the back.

        Returns:
            A sorted frame.

        Raises:
            If the name is missing or the dtype is not sortable.
        """
        var keys = List[String]()
        keys.append(by)
        var down = List[Bool]()
        down.append(descending)
        var front = List[Bool]()
        front.append(nulls_first)
        return self.sort_values(keys, down, front)

    def write_to(self, mut writer: Some[Writer]):
        """Writes the shape and the schema, one column per line.

        This is not the display layer either. A frame that cannot be printed at
        all is not debuggable, and a frame that prints its values needs column
        widths, truncation and a null spelling, which is a change of its own.

        Args:
            writer: The sink.
        """
        writer.write(
            "DataFrame ",
            self.rows,
            " rows x ",
            len(self.columns),
            " columns",
        )
        for i in range(len(self.schema)):
            writer.write(
                "\n  ",
                self.schema[i].name,
                ": ",
                self.schema[i].dtype,
                ", ",
                self.columns[i].null_count(),
                " null",
            )
