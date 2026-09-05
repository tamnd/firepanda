"""The eager `DataFrame`.

A frame is a `Schema` and a list of `ChunkedArray`, and the invariant that makes
it a frame rather than a bag of columns is that every column has the same length
and every name is distinct. Both are checked once, at construction, and then
relied on everywhere else.

The columns are chunked because the things that produce them are. A Parquet file
gives one array per row group and a concat gives one per input, and flattening
those into contiguous memory costs a copy of the whole column that most of a
workload does not need. Nothing in this file produces a column of more than one
chunk yet, so in practice every column here has exactly one, and every method
reaches through `only()`, which borrows that chunk and copies nothing. A column
that does have more than one raises there rather than being silently read as its
first chunk, and teaching the operators to loop over chunks is what removes those
calls one at a time.

The schema is not a summary of the columns, it is a separate authority. A column
knows its physical dtype and nothing else; the schema knows the name, the logical
type and whether nulls are allowed. Keeping both means the two can disagree, so
the constructor checks that they do not, and every operation that changes one
changes the other in the same expression.

There are row labels but nothing aligns on them. A frame carries an `Index`, and
every operation that chooses rows carries the labels of the rows it chose, so
`tail` and `sort_values` and `drop_nulls` report which rows they kept the way
pandas does. What is still missing is the other half: nothing looks a row up by
its label, `loc` does not exist, and two frames put together do not match their
indexes first, so rows are still addressed by position and `filter` and `take`
are still the only ways to address them. That half is the `Index` API in
https://github.com/tamnd/firepanda/issues/154. The second divergence is that
nothing mutates. Every method returns a new frame.

The third thing worth knowing is what a copy costs. `select` copies the columns it
keeps, `filter` and `take` copy by construction because they build new columns
anyway, and `slice` copies bytes. `__getitem__` on a position hands back a
borrowed reference and does not copy; `column` on a name hands back a `Series` and
does. That split is not elegant and it is honest: a borrowing accessor needs the
index in the return type, which a name lookup cannot provide until the plan layer
at M4 turns column references into something resolved ahead of time.

`join` is the one method here that does real work of its own rather than calling
a kernel with the columns unpacked. Which rows pair with which is
`firepanda/join`; what the output columns are called and where a shared key
column's values come from is this file, because both of those are questions about
schemas and nothing below the frame layer has one.

None of this is lazy. `docs/specs/04-python-dx.md` describes the eager surface as
a facade over a plan, which is true from M4 onwards. At M1 the facade is the
implementation and every method runs when it is called.
"""

from firepanda.array.any import AnyArray, ColumnRefs, borrow_columns
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray, Sortedness, wrap_columns
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.display import DisplayOptions, render_table
from firepanda.frame.index import Index
from firepanda.hash.grouping import group_ordinals
from firepanda.join.pairs import JoinKind, join_indices, take_pair
from firepanda.kernel.cast import cast_any
from firepanda.kernel.chunked import (
    cast_chunked,
    filter_chunked,
    slice_chunked,
    take_chunked,
)
from firepanda.kernel.group import (
    AggKind,
    aggregate_group_any,
    aggregate_group_pair_any,
)
from firepanda.kernel.nulls import all_valid_mask, coalesce_any
from firepanda.kernel.reduce import reduce_any
from firepanda.kernel.select import filter_any, take_any
from firepanda.kernel.sort import argsort_any_into, identity_permutation
from firepanda.kernel.topn import group_top_rows_any

from .groupby import AggSpec
from .series import Series, _check_range, _head_end, _tail_start, _to_positions


struct DataFrame(Copyable, Movable, Sized, Writable):
    """A set of equal length named columns, addressed by position."""

    var schema: Schema
    """The column names, logical types and nullability."""

    var columns: List[ChunkedArray]
    """The data, one entry per schema field and in the same order.

    Chunked rather than contiguous, because a Parquet file gives one array per
    row group and a concat gives one per input, and flattening those costs a copy
    of the whole column for nothing. Almost every column here has exactly one
    chunk, and every operator below the frame is written against `AnyArray`, so
    `only()` is how they meet: it borrows the single chunk and copies nothing.
    An operator that has not been taught about chunks calls it and raises on a
    column that has more than one, which is the honest failure rather than a
    silent partial answer."""

    var rows: Int
    """The row count. Every column has this length."""

    var index: Index
    """The row labels. A frame that has not been gathered, filtered or sliced
    from anywhere but the front carries the default range, which is two integers
    and no memory, so the field costs nothing on the path that does not use it.
    See `firepanda/frame/index.mojo`."""

    def __init__(out self):
        """Constructs a frame with no columns and no rows."""
        self.schema = Schema()
        self.columns = List[ChunkedArray]()
        self.rows = 0
        self.index = Index(0)

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
        self.columns = wrap_columns(columns^)
        self.index = Index(self.rows)

    def __init__(out self, var schema: Schema, var columns: List[ChunkedArray]):
        """Constructs a frame from columns that are already chunked.

        The overload above is the one nearly everything uses, because nearly
        everything produces a `List[AnyArray]`. This one is for the operators
        that have been taught to walk chunks and so produce chunked columns of
        their own, and it exists so that they do not have to flatten their result
        just to hand it over.

        Args:
            schema: The schema. Must match `columns` in length and order.
            columns: The data. Must all be the same length.
        """
        self.rows = 0 if len(columns) == 0 else len(columns[0])
        self.schema = schema^
        self.columns = columns^
        self.index = Index(self.rows)

    def __init__(out self, *, copy: Self):
        """Deep-copies a frame.

        Args:
            copy: The frame to copy.
        """
        self.schema = Schema(copy=copy.schema)
        self.columns = List[ChunkedArray](capacity=len(copy.columns))
        for i in range(len(copy.columns)):
            self.columns.append(ChunkedArray(copy=copy.columns[i]))
        self.rows = copy.rows
        self.index = Index(copy=copy.index)

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

    def __getitem__(
        ref self, i: Int
    ) raises -> ref[self.columns[i].chunks[0]] AnyArray:
        """Borrows a column by position, without copying it.

        Args:
            i: The column position. Must be less than the width.

        Returns:
            A reference to the column.

        Raises:
            If the column has more than one chunk.
        """
        return self.columns[i].only()

    def column_refs[o: ImmOrigin](ref[o] self) raises -> ColumnRefs[o]:
        """Borrows every column, for handing to something that reads several.

        A group by, a join and the table renderer all want to look at a set of
        the frame's columns without owning them. They used to be handed the
        frame's own list, which worked only because the list holds the columns
        themselves; once it holds something else, or once a caller has only a
        borrow of the frame, that stops being possible and the alternative is a
        copy of every column.

        The origin travels with the references, so the frame cannot be destroyed
        while they are alive. Erasing it instead compiles and is wrong: the
        frame's last use is the argument, so it is destroyed before the callee
        runs, and the callee reads freed memory that still looks enough like a
        column to give an answer.

        Parameters:
            o: The frame's origin, which the references inherit.

        Returns:
            One reference per column, in schema order.

        Raises:
            If any column has more than one chunk.
        """
        var refs = ColumnRefs[o](capacity=len(self.columns))
        for i in range(len(self.columns)):
            refs.append(
                Pointer(to=self.columns[i].only()).unsafe_origin_cast[o]()
            )
        return refs^

    def into_columns(deinit self) -> List[ChunkedArray]:
        """Gives up the columns without copying them, consuming the frame.

        The schema is dropped, so this is for a caller that already has it or
        does not need it. The engine is the caller: a scan is handed a frame,
        takes the columns apart into chunks and pushes them, and the frame it
        came from is gone by the time the first chunk moves.

        Returns:
            The columns, in schema order.
        """
        return self.columns^

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

        This copies, and it flattens: a `Series` is one contiguous array, so a
        column in pieces is stacked on the way out. Use `__getitem__` with a
        position when the copy matters and a borrow will do.

        Args:
            name: The column name.

        Returns:
            A copy of the column, carrying its name.

        Raises:
            If no column has that name.
        """
        var at = self.schema.index_of(name)
        return Series(name, ChunkedArray(copy=self.columns[at]).combine())

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
        var columns = List[ChunkedArray](capacity=len(names))
        for i in range(len(names)):
            for j in range(i):
                if names[j] == names[i]:
                    raise Error(
                        "select names a column twice: '" + names[i] + "'"
                    )
            columns.append(
                ChunkedArray(copy=self.columns[self.schema.index_of(names[i])])
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
            out.columns[at] = ChunkedArray(column^.into_values())
            return out^

        out.schema.append(field^)
        out.columns.append(ChunkedArray(column^.into_values()))
        out.rows = len(out.columns[0])
        return out^

    def cast(self, name: String, to: DType, strict: Bool = True) raises -> Self:
        """Returns a frame with one column converted to another dtype.

        Args:
            name: The column to convert.
            to: The target dtype.
            strict: Whether a text value that is not a number raises rather than
                becoming a null.

        Returns:
            A frame with that column converted and the rest untouched.

        Raises:
            If the name is missing, either dtype has no physical layout, or the
            column is text and strict and some value is not a number.
        """
        return self._cast(
            name,
            cast_chunked(self.columns[self.schema.index_of(name)], to, strict),
        )

    def cast(
        self, name: String, to: LogicalType, strict: Bool = True
    ) raises -> Self:
        """Returns a frame with one column converted to another logical type.

        This is the overload that can name text. `frame.cast("id",
        LogicalType.STRING)` renders a number column as text, and the reverse
        reads it back.

        Args:
            name: The column to convert.
            to: The target type.
            strict: Whether a text value that is not a number raises rather than
                becoming a null.

        Returns:
            A frame with that column converted and the rest untouched.

        Raises:
            If the name is missing, the type has no conversion from this
            column's, or the column is text and strict and some value is not a
            number.
        """
        return self._cast(
            name,
            cast_chunked(self.columns[self.schema.index_of(name)], to, strict),
        )

    def _cast(self, name: String, var converted: ChunkedArray) raises -> Self:
        """Puts a converted column back in place of the one it came from."""
        var at = self.schema.index_of(name)
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
        var columns = List[ChunkedArray](capacity=len(self.columns))
        for i in range(len(self.columns)):
            columns.append(filter_chunked(self.columns[i], mask))
        var out = Self(Schema(copy=self.schema), columns^)
        # A frame of no columns cannot read its height off column zero, and a
        # filter that keeps nothing leaves every column with no chunks at all,
        # so the height is counted here rather than inferred.
        out.rows = 0 if len(out.columns) == 0 else len(out.columns[0])
        out.index = self.index.filter(mask)
        return out^

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
        var columns = List[ChunkedArray](capacity=len(self.columns))
        for i in range(len(self.columns)):
            columns.append(take_chunked(self.columns[i], indices))
        var out = Self(Schema(copy=self.schema), columns^)
        out.rows = len(indices)
        out.index = self.index.take(indices)
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
        var columns = List[ChunkedArray](capacity=len(self.columns))
        for i in range(len(self.columns)):
            columns.append(slice_chunked(self.columns[i], start, end))
        var out = Self(Schema(copy=self.schema), columns^)
        out.rows = end - start
        out.index = self.index.slice(start, end)
        return out^

    def head(self, n: Int = 5) raises -> Self:
        """Returns the first rows.

        Args:
            n: How many, clamped to the height. A negative `n` means all but the
                last `n` of them, as in pandas and as in `s[:-2]`.

        Returns:
            A frame of at most `n` rows.
        """
        return self.slice(0, _head_end(n, self.rows))

    def tail(self, n: Int = 5) raises -> Self:
        """Returns the last rows.

        Args:
            n: How many, clamped to the height. A negative `n` means all but the
                first `n` of them, as in pandas and as in `s[2:]`.

        Returns:
            A frame of at most `n` rows.
        """
        return self.slice(_tail_start(n, self.rows), self.rows)

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
                self.columns[at[i]].only(),
                order,
                descending[i],
                nulls_first[i],
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
        var out = self.take(
            _to_positions(self.argsort(by, descending, nulls_first))
        )
        # The most significant key is the one that came out sorted. The rest are
        # only sorted inside a run of equal values above them, which is not what
        # the flag means, so they are left alone. `mark_sorted` refuses a column
        # holding a null, which is the reason `nulls_first` is not consulted.
        out.columns[self.schema.index_of(by[0])].mark_sorted(
            Sortedness.DESCENDING if descending[0] else Sortedness.ASCENDING
        )
        return out^

    def sortedness(mut self, name: String) raises -> Sortedness:
        """Returns what is known about a column's order, scanning if it has to.

        A column that a sort produced knows already and answers for free. Any
        other column costs one pass the first time it is asked and remembers the
        answer, including a negative one.

        Args:
            name: The column.

        Returns:
            What is now known about the order.

        Raises:
            If the name is missing or the dtype is not one firepanda can order.
        """
        return self.columns[self.schema.index_of(name)].prove_sorted()

    def is_monotonic_increasing(mut self, name: String) raises -> Bool:
        """Reports whether a column never decreases, as pandas does.

        A column holding a null is never monotonic here. pandas says the same,
        and the reason on this side is that the flag deliberately records nothing
        about where a sort put the nulls.

        Args:
            name: The column.

        Returns:
            True if every value is at least the one before it.

        Raises:
            If the name is missing or the dtype is not one firepanda can order.
        """
        return self.sortedness(name).is_ascending()

    def is_monotonic_decreasing(mut self, name: String) raises -> Bool:
        """Reports whether a column never increases, as pandas does.

        Args:
            name: The column.

        Returns:
            True if every value is at most the one before it.

        Raises:
            If the name is missing or the dtype is not one firepanda can order.
        """
        return self.sortedness(name).is_descending()

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

    def agg(self, specs: List[AggSpec]) raises -> Self:
        """Reduces the whole frame to one row.

        This is `group_by` with nothing to group by, and it is a separate method
        rather than an empty key list because the two do different work. A group
        by hashes every row to find out which group it belongs to. There is
        nothing to find out here, so the reductions read the columns straight
        through and the hashing never happens. On ten million rows that is the
        difference between eighty five milliseconds and five.

        The output columns are named by the same rule `group_by` uses, so asking
        for two reductions of one column gives `x_sum` and `x_mean` rather than a
        collision.

        Args:
            specs: What to compute. At least one.

        Returns:
            A frame of exactly one row and one column per spec.

        Raises:
            If no specs were given, if a name is missing, if two outputs would
            collide, or if a dtype involved has no physical layout.
        """
        if len(specs) == 0:
            raise Error("agg: at least one reduction is required")

        var fields = List[Field](capacity=len(specs))
        var columns = List[AnyArray](capacity=len(specs))
        for s in range(len(specs)):
            var name = specs[s].output_name()
            for f in range(len(fields)):
                if fields[f].name == name:
                    raise Error(
                        "agg: two output columns would both be called " + name
                    )
            var produced: AnyArray
            if specs[s].kind.reads_two_columns():
                # A correlation over a whole frame has no fast path of its own
                # yet, so it goes the grouped way with every row in one group.
                # The codes are all zero because a fresh column is, and one
                # allocation on the rarest pair of reductions is not worth a
                # second implementation of the pairwise loop.
                var codes = Array[DType.uint32](self.rows)
                produced = aggregate_group_pair_any(
                    self.columns[self.schema.index_of(specs[s].column)].only(),
                    self.columns[self.schema.index_of(specs[s].other)].only(),
                    specs[s].kind,
                    codes^,
                    1,
                )
            else:
                produced = reduce_any(
                    self.columns[self.schema.index_of(specs[s].column)].only(),
                    specs[s].kind,
                )
            fields.append(Field(name, produced.type))
            columns.append(produced^)

        var out = Self(Schema(fields^), columns^)
        out.rows = 1
        return out^

    def agg_all(self, kind: AggKind) raises -> Self:
        """Applies one reduction to every column.

        This is `df.sum()` and its siblings. The output columns keep their
        original names, because there is only one reduction and no collision to
        disambiguate.

        Args:
            kind: The reduction to apply to everything.

        Returns:
            A frame of one row with the same column names.

        Raises:
            As `agg` does, and if the frame has no columns.
        """
        var specs = List[AggSpec](capacity=self.width())
        for i in range(len(self.schema)):
            specs.append(
                AggSpec(self.schema[i].name, kind, self.schema[i].name)
            )
        return self.agg(specs^)

    def group_by(
        self,
        by: List[String],
        specs: List[AggSpec],
        dropna: Bool = True,
        sort: Bool = True,
    ) raises -> Self:
        """Groups rows by one or more key columns and reduces each group.

        The result has one row per distinct key tuple, the key columns first in
        the order they were given, then one column per spec. Nothing about the
        input is mutated and the input's row order is never disturbed: the
        grouping is a scatter over the rows, and the only sort that happens is on
        the result, which is one row per group.

        Args:
            by: The key columns. At least one, no repeats.
            specs: What to compute. May be empty, which gives the distinct key
                tuples and nothing else, the way `drop_duplicates` would.
            dropna: Drop the groups whose key contains a null, as pandas does.
            sort: Order the result by the key columns ascending, as pandas does.
                False leaves the groups in first-seen order and costs nothing.

        Returns:
            The aggregated frame.

        Raises:
            If a name is missing or repeated, if two outputs would collide, or if
            a dtype involved has no physical layout.
        """
        var at = List[Int](capacity=len(by))
        for i in range(len(by)):
            var idx = self.schema.index_of(by[i])
            for j in range(len(at)):
                if at[j] == idx:
                    raise Error(
                        "group by: key column " + by[i] + " was given twice"
                    )
            at.append(idx)

        var grouping = group_ordinals(self.column_refs(), at, self.rows)

        var fields = List[Field]()
        var columns = List[AnyArray]()
        for k in range(len(at)):
            fields.append(Field(by[k], self.columns[at[k]].type))
            columns.append(
                take_any(self.columns[at[k]].only(), grouping.rows_at)
            )

        for s in range(len(specs)):
            var name = specs[s].output_name()
            for f in range(len(fields)):
                if fields[f].name == name:
                    raise Error(
                        "group by: two output columns would both be called "
                        + name
                    )
            var produced: AnyArray
            if specs[s].kind.reads_two_columns():
                produced = aggregate_group_pair_any(
                    self.columns[self.schema.index_of(specs[s].column)].only(),
                    self.columns[self.schema.index_of(specs[s].other)].only(),
                    specs[s].kind,
                    grouping.codes,
                    grouping.groups,
                )
            else:
                produced = aggregate_group_any(
                    self.columns[self.schema.index_of(specs[s].column)].only(),
                    specs[s].kind,
                    grouping.codes,
                    grouping.groups,
                )
            fields.append(Field(name, produced.type))
            columns.append(produced^)

        var out = Self(Schema(fields^), columns^)

        if dropna:
            # A group's key values are whatever its representative row holds, so
            # asking whether the key is null is asking about that one row rather
            # than about the gathered column.
            #
            # A key column with no nulls in it cannot drop anything, so ask each
            # one that question first and walk the groups only for the ones that
            # can answer yes. The question is a popcount over the column's
            # validity, which is one bit a row, and the walk it saves is a bit
            # lookup per group per key at four bytes a row of representative row
            # indexes. On a group by whose six keys have ten million tuples
            # between them that is sixty million bit lookups traded for six
            # passes over a megabyte and a quarter, and the usual answer is that
            # none of the keys have nulls and the whole thing goes away.
            var risky = List[Int]()
            for k in range(len(at)):
                if self.columns[at[k]].null_count() > 0:
                    risky.append(at[k])

            if len(risky) > 0:
                var keep = Array[DType.bool](grouping.groups)
                var dropping = False
                for g in range(grouping.groups):
                    var ok = True
                    for k in range(len(risky)):
                        if (
                            not self.columns[risky[k]]
                            .only()
                            .is_valid(grouping.rows_at[g])
                        ):
                            ok = False
                            break
                    keep.set_valid(g, ok)
                    if not ok:
                        dropping = True
                if dropping:
                    out = out.filter(keep)

        if sort and len(out) > 1:
            var down = List[Bool]()
            var front = List[Bool]()
            for _ in range(len(by)):
                down.append(False)
                front.append(False)
            out = out.sort_values(by, down, front)

        return out^

    def _group_top(
        self,
        by: List[String],
        column: String,
        n: Int,
        largest: Bool,
        dropna: Bool,
    ) raises -> Self:
        """The body behind `group_nlargest` and `group_nsmallest`.

        Args:
            by: The key columns.
            column: The column being ranked.
            n: How many rows to keep per group.
            largest: True to keep the highest values, False the lowest.
            dropna: Drop the groups whose key contains a null.

        Returns:
            The kept rows of every column.

        Raises:
            As `group_nlargest` does.
        """
        var at = List[Int](capacity=len(by))
        for i in range(len(by)):
            var idx = self.schema.index_of(by[i])
            for j in range(len(at)):
                if at[j] == idx:
                    raise Error(
                        "group top: key column " + by[i] + " was given twice"
                    )
            at.append(idx)

        var grouping = group_ordinals(self.column_refs(), at, self.rows)
        var top = group_top_rows_any(
            self.columns[self.schema.index_of(column)].only(),
            grouping.codes,
            grouping.groups,
            n,
            largest,
        )

        # A group's key values are whatever its representative row holds, so
        # dropping the null keys is a question about that one row. Ask each key
        # column whether it has any nulls at all first, the way `group_by` does,
        # because the usual answer is no and then the walk goes away entirely.
        var risky = List[Int]()
        if dropna:
            for k in range(len(at)):
                if self.columns[at[k]].null_count() > 0:
                    risky.append(at[k])

        if len(risky) == 0:
            return self.take(top.rows_at)

        var wanted = List[Int](capacity=len(top.rows_at))
        var cursor = 0
        for g in range(grouping.groups):
            var ok = True
            for k in range(len(risky)):
                if (
                    not self.columns[risky[k]]
                    .only()
                    .is_valid(grouping.rows_at[g])
                ):
                    ok = False
                    break
            if ok:
                for j in range(top.counts[g]):
                    wanted.append(top.rows_at[cursor + j])
            cursor += top.counts[g]
        return self.take(wanted)

    def group_nlargest(
        self,
        by: List[String],
        column: String,
        n: Int,
        dropna: Bool = True,
    ) raises -> Self:
        """Keeps each group's `n` rows with the largest values in one column.

        This is `df.groupby(by)[column].nlargest(n)` with the rest of the frame
        carried along, which is what a query wanting the top few rows of every
        group actually needs. The result keeps every column and every one of the
        input's dtypes, because it is a selection of rows and not a reduction.

        The groups come back in first-seen order and the rows inside a group come
        back best first. A row with a null or a NaN in the ranked column is never
        kept, so a group with fewer than `n` present values contributes fewer
        than `n` rows. Two rows with the same value are separated by their
        position, the earlier one winning, which is what pandas does and is what
        makes the answer independent of how the work was split across cores.

        Args:
            by: The key columns. At least one, no repeats.
            column: The column to rank by. Must be a numeric one.
            n: How many rows to keep per group. At least one.
            dropna: Drop the groups whose key contains a null, as pandas does.

        Returns:
            A frame of the kept rows, with the same columns as the input.

        Raises:
            If a name is missing or repeated, if the ranked column is not
            numeric, if `n` is not positive, or if a key dtype has no physical
            layout.
        """
        return self._group_top(by, column, n, True, dropna)

    def group_nsmallest(
        self,
        by: List[String],
        column: String,
        n: Int,
        dropna: Bool = True,
    ) raises -> Self:
        """Keeps each group's `n` rows with the smallest values in one column.

        `group_nlargest` read the other way round, and the same in every other
        respect, ties included: the earlier row still wins.

        Args:
            by: The key columns. At least one, no repeats.
            column: The column to rank by. Must be a numeric one.
            n: How many rows to keep per group. At least one.
            dropna: Drop the groups whose key contains a null, as pandas does.

        Returns:
            A frame of the kept rows, with the same columns as the input.

        Raises:
            As `group_nlargest` does.
        """
        return self._group_top(by, column, n, False, dropna)

    def group_agg(
        self,
        by: List[String],
        kind: AggKind,
        dropna: Bool = True,
        sort: Bool = True,
    ) raises -> Self:
        """Applies one reduction to every column that is not a key.

        This is `df.groupby(keys).sum()` and its siblings. The output columns keep
        their original names, because there is only one reduction and no
        collision to disambiguate.

        Args:
            by: The key columns.
            kind: The reduction to apply to everything else.
            dropna: As `group_by`.
            sort: As `group_by`.

        Returns:
            The aggregated frame.

        Raises:
            As `group_by` does.
        """
        var specs = List[AggSpec]()
        for i in range(len(self.schema)):
            var name = self.schema[i].name
            var is_key = False
            for k in range(len(by)):
                if by[k] == name:
                    is_key = True
                    break
            if not is_key:
                specs.append(AggSpec(name, kind, name))
        return self.group_by(by, specs, dropna, sort)

    def group_count(
        self, by: List[String], dropna: Bool = True, sort: Bool = True
    ) raises -> Self:
        """Counts the rows in each group.

        `size` rather than `count`: the number is per group, not per column, so a
        null in some other column does not change it. pandas spells this
        `df.groupby(keys).size()`.

        Args:
            by: The key columns.
            dropna: As `group_by`.
            sort: As `group_by`.

        Returns:
            The key columns plus an int64 column called `size`.

        Raises:
            As `group_by` does.
        """
        if len(by) == 0:
            raise Error("group by: at least one key column is required")
        var specs = List[AggSpec]()
        specs.append(AggSpec(by[0], AggKind.SIZE, "size"))
        return self.group_by(by, specs, dropna, sort)

    def join(
        self,
        other: Self,
        on: List[String],
        kind: JoinKind = JoinKind.INNER,
        suffix: String = "_right",
        columns: List[String] = List[String](),
    ) raises -> Self:
        """Joins two frames on columns that have the same name in both.

        The result has the left frame's columns in their original order, then the
        right frame's columns except the keys, which the two frames shared and of
        which the output keeps one. A right column whose name is already taken
        gets `suffix` appended.

        Args:
            other: The right frame.
            on: The key columns. Must exist in both frames with the same dtype.
            kind: Which rows to keep.
            suffix: Appended to a right column whose name collides.
            columns: Which output columns to build, or empty for all of them.

        Returns:
            The joined frame.

        Raises:
            As `join_on` does.
        """
        return self.join_on(other, on, on, kind, suffix, columns)

    def join_on(
        self,
        other: Self,
        left_on: List[String],
        right_on: List[String],
        kind: JoinKind = JoinKind.INNER,
        suffix: String = "_right",
        columns: List[String] = List[String](),
    ) raises -> Self:
        """Joins two frames on keys that are named differently on each side.

        Where a key pair shares a name the output keeps one column, and where it
        does not it keeps both, which is what pandas does and is the only thing it
        can do: the two columns have different names and dropping one would lose a
        name the caller asked for.

        The kept column of a shared pair is filled from whichever side had the
        row. That only matters for a right or outer join, where an output row can
        have no left row at all, and getting it wrong would put a null in the
        column the row was matched on.

        `columns` is the projection, and it is here rather than left to a `select`
        afterwards because of what a join costs. Pairing ten million rows against
        a thousand takes eight milliseconds and gathering the columns takes
        twenty four, so a column the caller is going to drop is not a small waste
        at the end of the query, it is most of the query. Naming the wanted
        columns skips their gathers entirely, and it also saves the caller
        narrowing the input first, which is a copy of a whole side to avoid a
        gather of part of it. The names are the output's names, worked out exactly
        as they would be without a projection, so asking for a column does not
        change what it is called.

        Args:
            other: The right frame.
            left_on: The left key columns, most significant first.
            right_on: The right key columns, matched positionally.
            kind: Which rows to keep.
            suffix: Appended to a right column whose name collides.
            columns: Which output columns to build, in the order wanted, or
                empty for all of them in their natural order.

        Returns:
            The joined frame.

        Raises:
            If the key lists disagree in length, if a name is missing or
            repeated, if a key pair has different dtypes, if a right column name
            still collides after the suffix, if a projected name is not one the
            result has or is asked for twice, or if a dtype involved has no
            physical layout.
        """
        if len(left_on) != len(right_on):
            raise Error(
                "join: needs the same number of keys on each side; got "
                + String(len(left_on))
                + " on the left and "
                + String(len(right_on))
                + " on the right"
            )

        var left_at = List[Int](capacity=len(left_on))
        var right_at = List[Int](capacity=len(right_on))
        for k in range(len(left_on)):
            var here = self.schema.index_of(left_on[k])
            var there = other.schema.index_of(right_on[k])
            for j in range(len(left_at)):
                if left_at[j] == here:
                    raise Error(
                        "join: key column " + left_on[k] + " was given twice"
                    )
                if right_at[j] == there:
                    raise Error(
                        "join: key column " + right_on[k] + " was given twice"
                    )
            left_at.append(here)
            right_at.append(there)

        var pairs = join_indices(
            self.column_refs(),
            left_at,
            self.rows,
            other.column_refs(),
            right_at,
            other.rows,
            kind,
        )

        # Only these two can produce an output row that no left row backs, so
        # only these two need the shared key columns filled from both sides. The
        # others gather the left column straight through, which is one pass
        # instead of one pass with a branch per row.
        var coalescing = kind == JoinKind.RIGHT or kind == JoinKind.OUTER

        # The whole output is worked out before any of it is built. The names
        # have to be settled first because a projection names the result and the
        # result's names are not known until the suffixing has been decided, and
        # planning first is also what makes a dropped column free rather than
        # built and then thrown away.
        var fields = List[Field]()
        var from_right = List[Bool]()
        var source_at = List[Int]()
        var pair_with = List[Int]()

        for i in range(len(self.columns)):
            var shared = -1
            for k in range(len(left_at)):
                if left_at[k] == i and left_on[k] == right_on[k]:
                    shared = k
                    break
            fields.append(Field(self.schema[i].name, self.schema[i].dtype))
            from_right.append(False)
            source_at.append(i)
            pair_with.append(
                right_at[shared] if shared >= 0 and coalescing else -1
            )

        if kind.keeps_right_columns():
            for j in range(len(other.columns)):
                var folded = False
                for k in range(len(right_at)):
                    if right_at[k] == j and left_on[k] == right_on[k]:
                        folded = True
                        break
                if folded:
                    continue

                var name = other.schema[j].name
                if _has_name(fields, name):
                    name = name + suffix
                    if _has_name(fields, name):
                        raise Error(
                            "join: the right frame's column '"
                            + other.schema[j].name
                            + "' collides and so does '"
                            + name
                            + "'; pass a different suffix"
                        )
                fields.append(Field(name, other.schema[j].dtype))
                from_right.append(True)
                source_at.append(j)
                pair_with.append(-1)

        var wanted = List[Int]()
        if len(columns) == 0:
            for i in range(len(fields)):
                wanted.append(i)
        else:
            for c in range(len(columns)):
                var found = -1
                for i in range(len(fields)):
                    if fields[i].name == columns[c]:
                        found = i
                        break
                if found < 0:
                    raise Error(
                        "join: the result has no column '"
                        + columns[c]
                        + "' to keep"
                    )
                for w in range(len(wanted)):
                    if wanted[w] == found:
                        raise Error(
                            "join: column '"
                            + columns[c]
                            + "' was asked for twice"
                        )
                wanted.append(found)

        var kept = List[Field](capacity=len(wanted))
        var built = List[AnyArray](capacity=len(wanted))
        for w in range(len(wanted)):
            var i = wanted[w]
            if from_right[i]:
                built.append(
                    take_any(other.columns[source_at[i]].only(), pairs.right_at)
                )
            elif pair_with[i] >= 0:
                built.append(
                    take_pair(
                        self.columns[source_at[i]].only(),
                        other.columns[pair_with[i]].only(),
                        pairs.left_at,
                        pairs.right_at,
                    )
                )
            else:
                built.append(
                    take_any(self.columns[source_at[i]].only(), pairs.left_at)
                )
            kept.append(Field(fields[i].name, fields[i].dtype))

        var out = Self(Schema(kept^), built^)
        out.rows = len(pairs)
        return out^

    def cross_join(self, other: Self, suffix: String = "_right") raises -> Self:
        """Pairs every row of this frame with every row of another.

        The result is as tall as the product of the two heights, so this is for a
        small frame on at least one side. Nothing here refuses a large one,
        because any threshold would be arbitrary and would be in the way of the
        case the operation exists for.

        Args:
            other: The right frame.
            suffix: Appended to a right column whose name collides.

        Returns:
            A frame of `len(self) * len(other)` rows.

        Raises:
            If a right column name still collides after the suffix, or if a dtype
            has no physical layout.
        """
        return self.join_on(
            other, List[String](), List[String](), JoinKind.CROSS, suffix
        )

    def drop_nulls(self, subset: List[String] = List[String]()) raises -> Self:
        """Returns the rows where every column in the subset is present.

        pandas spells the choice between "drop a row if any column is missing"
        and "drop it only if all of them are" as `how=`. This has only the first
        one, because `how="all"` on a subset of one column means the same thing
        as `how="any"` and on a subset of several it answers a question nobody
        has asked in the wild. Narrowing `subset` is the useful control and it is
        here.

        Args:
            subset: Which columns must be present. An empty list means all of
                them, which is the pandas default.

        Returns:
            A frame with the same columns and the offending rows removed.

        Raises:
            If a named column does not exist, or if a dtype has no physical
            layout.
        """
        # The mask reads validity and nothing else, so these are borrowed. They
        # used to be deep copies, which on a frame of ten million rows meant
        # copying every column in the frame to find out which rows to keep.
        if len(subset) == 0:
            return self.filter(all_valid_mask(self.column_refs(), self.rows))
        var picked = List[AnyArray](capacity=len(subset))
        for c in range(len(subset)):
            picked.append(
                AnyArray(copy=self.columns[self.index_of(subset[c])].only())
            )
        return self.filter(all_valid_mask(borrow_columns(picked), self.rows))

    def fill_null(self, name: String, var value: Series) raises -> Self:
        """Returns the frame with one column's missing rows taken from another column.

        The fallback may be one row, which is used for every missing row and is
        how filling with a scalar is spelled.

        Args:
            name: Which column to fill.
            value: The fallback, of the same dtype and either as tall as the
                frame or one row. Its name is ignored.

        Returns:
            A frame of the same shape, with the named column filled.

        Raises:
            If the column does not exist, if the dtypes differ, if the fallback
            is neither one row nor as tall as the frame, or if the dtype has no
            physical layout.
        """
        var at = self.index_of(name)
        var filled = Series(
            name, coalesce_any(self.columns[at].only(), value.values)
        )
        return self.with_column(filled^)

    def write_to(self, mut writer: Some[Writer]):
        """Writes the frame as a table.

        The default limits apply, which is ten rows and twenty columns with the
        middle elided. `render_table` in `firepanda.frame.display` takes a
        `DisplayOptions` for anything else.

        Args:
            writer: The sink.
        """
        # `Writable.write_to` cannot raise and borrowing the columns can, on a
        # column of more than one chunk. Printing is the one place where failing
        # is worse than saying so, because it is normally the thing you reached
        # for to find out what went wrong, so the reason goes where the table
        # would have been.
        try:
            writer.write(
                render_table(
                    self.schema, self.column_refs(), self.rows, DisplayOptions()
                )
            )
        except e:
            writer.write("DataFrame cannot be shown: ", String(e))

    def describe(self) -> String:
        """Returns the shape and the schema without rendering any values.

        `write_to` prints the data, which is what you want at a prompt and not
        what you want when a column is a million rows long and the question is
        what dtype it ended up as. This is the answer to that question.

        Returns:
            One line for the shape and one per column.
        """
        var out = String(
            "DataFrame ", self.rows, " rows x ", len(self.columns), " columns"
        )
        for i in range(len(self.schema)):
            out += String(
                "\n  ",
                self.schema[i].name,
                ": ",
                self.schema[i].dtype,
                ", ",
                self.columns[i].null_count(),
                " null",
            )
        return out^


def _has_name(fields: List[Field], name: String) -> Bool:
    """Reports whether a field list already uses a name.

    Args:
        fields: The fields built so far.
        name: The name to look for.

    Returns:
        True if some field has it.
    """
    for i in range(len(fields)):
        if fields[i].name == name:
            return True
    return False
