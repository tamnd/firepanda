"""Three valued and, or and not over boolean columns.

A comparison against a null answers null, so a predicate built out of
comparisons holds nulls, and what a connective does with one is not what two
valued logic would do. A false and a null is false, because nothing the null
could have been would have made the pair true. A true or a null is true, the
same reason read the other way round. Everything else a null touches is a null.

That is Kleene's three valued logic, it is what SQL says and what pandas says,
and it is why these three are a file of their own rather than three more entries
in `binary.mojo`. The other operations there answer null wherever either input
is null, which is one shared repair over the intersected validity. These answer
where one input is null and the other one settles it, so the validity has to be
read off the values and not off the two bitmaps.

The values come out of the same kind of morsel loop the comparisons use, one
byte per row, because a boolean column here is bytes and not bits. The validity
is the part that differs. When neither input holds a null, and a predicate over
a table without nulls is most of them, the answer is the two bitmaps intersected
and there is no more to do. When one of them does, only the words the
intersection marks are walked, and in those words only the rows a null reaches,
so the cost of the rule is paid by the columns that have nulls in them and by no
others.

The repair runs once at the end rather than inside each morsel, which is the one
place this departs from `compare.mojo`. It has to: the validity a row ends up
with is not known until that row's inputs have been looked at, and a worker
cannot repair against a bitmap that is still being written.
"""

from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray, ColumnRefs
from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.logical import LogicalType
from firepanda.exec import parallel_morsels

from .mask import apply_validity, combined_validity, repair_range


comptime MaskRefs[o: ImmOrigin] = List[Pointer[Array[DType.bool], o]]
"""A borrowed set of boolean columns, for the connectives that read many.

`ColumnRefs` for masks, and there for the same reason. A conjunction of five
predicates in an execution plan reads five columns that are already sitting in a
chunk, and a `List[Array[DType.bool]]` argument would mean copying every one of
them out of the chunk first. The copy is a byte a row per column, which is the
whole of what the conjunction was going to cost.

The origin is carried rather than erased, so a chunk cannot be destroyed between
the argument being built and the callee reading it.
"""

comptime LOGIC_AND = 0
"""Operation code for the conjunction."""

comptime LOGIC_OR = 1
"""Operation code for the disjunction."""


struct LogicOp(Equatable, ImplicitlyCopyable, Movable, Writable):
    """One of the three connectives.

    Negation is in here with the other two although it reads one column rather
    than two, because what a caller has is a name out of a plan, and deciding
    the arity from the connective is one place rather than one per caller.
    """

    var code: UInt8
    """The connective, as a small integer."""

    comptime AND = Self(0)
    """The conjunction."""

    comptime OR = Self(1)
    """The disjunction."""

    comptime NOT = Self(2)
    """The negation, which reads one column."""

    def __init__(out self, code: UInt8):
        """Constructs a connective from its code.

        Args:
            code: The connective.
        """
        self.code = code

    def __eq__(self, other: Self) -> Bool:
        """Compares two connectives.

        Args:
            other: The connective to compare against.

        Returns:
            True if they are the same one.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two connectives.

        Args:
            other: The connective to compare against.

        Returns:
            True if they are different ones.
        """
        return self.code != other.code

    def reads_two_columns(self) -> Bool:
        """Reports whether the connective takes a pair.

        Returns:
            True for and and or, false for not.
        """
        return self != Self.NOT

    def write_to(self, mut writer: Some[Writer]):
        """Writes the connective as the word it is spelled with.

        Args:
            writer: Where the text goes.
        """
        if self == Self.AND:
            writer.write("and")
        elif self == Self.OR:
            writer.write("or")
        else:
            writer.write("not")


def logic_op(name: StringSlice) raises -> LogicOp:
    """Returns the connective a name stands for.

    Args:
        name: The name, lowercase, as a plan call carries it.

    Returns:
        The connective.

    Raises:
        Error: If the name is not one of the three.
    """
    if name == "and":
        return LogicOp.AND
    if name == "or":
        return LogicOp.OR
    if name == "not":
        return LogicOp.NOT
    raise Error(String("logic: ", name, " is not a connective"))


def is_logic_name(name: StringSlice) -> Bool:
    """Reports whether a name is one of the three connectives.

    Args:
        name: The name, lowercase, as a plan call carries it.

    Returns:
        True for and, or and not.
    """
    return name == "and" or name == "or" or name == "not"


def logical_and(
    a: Array[DType.bool], b: Array[DType.bool]
) raises -> Array[DType.bool]:
    """Conjoins two boolean columns, three valued.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Returns:
        A bool column, null only on the rows where neither side settles it.

    Raises:
        Error: If the two columns are different lengths.
    """
    return _connect[LOGIC_AND](a, b)


def logical_or(
    a: Array[DType.bool], b: Array[DType.bool]
) raises -> Array[DType.bool]:
    """Disjoins two boolean columns, three valued.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Returns:
        A bool column, null only on the rows where neither side settles it.

    Raises:
        Error: If the two columns are different lengths.
    """
    return _connect[LOGIC_OR](a, b)


def conjoin(columns: List[Array[DType.bool]]) raises -> Array[DType.bool]:
    """Conjoins any number of boolean columns in one pass, three valued.

    `logical_and` twice is two passes and two answers, and the middle one is a
    whole column written down for the next call to read back. A query that ands
    five predicates together writes four of those, and the four columns nobody
    asked for cost more than the predicates did: four writes and eight reads of
    a byte a row, against the one write and five reads this does.

    Args:
        columns: The columns to conjoin. Must all be the same length, and there
            must be at least one.

    Returns:
        A bool column, null only on the rows no column settles.

    Raises:
        Error: If no columns were given or they are not all the same length.
    """
    return _connect_all[LOGIC_AND](borrow_masks(columns))


def disjoin(columns: List[Array[DType.bool]]) raises -> Array[DType.bool]:
    """Disjoins any number of boolean columns in one pass, three valued.

    `conjoin` read the other way round, and the same in every other respect.
    Worth having beside it rather than only the conjunction because the
    disjunctive predicates are the wide ones: TPC-H q19 is three groups of four
    comparisons, and written as pairwise calls that is eleven intermediate
    columns for twelve predicates.

    Args:
        columns: The columns to disjoin. Must all be the same length, and there
            must be at least one.

    Returns:
        A bool column, null only on the rows no column settles.

    Raises:
        Error: If no columns were given or they are not all the same length.
    """
    return _connect_all[LOGIC_OR](borrow_masks(columns))


def borrow_masks[
    o: ImmOrigin
](ref[o] masks: List[Array[DType.bool]]) -> MaskRefs[o]:
    """Borrows every mask in a list, for handing to a connective.

    Parameters:
        o: The origin of the list, which the references inherit.

    Args:
        masks: The masks.

    Returns:
        One reference per mask, in order.
    """
    var out = MaskRefs[o](capacity=len(masks))
    for i in range(len(masks)):
        out.append(Pointer(to=masks[i]).unsafe_origin_cast[o]())
    return out^


def connect_all[
    o: ImmOrigin
](columns: MaskRefs[o], op: LogicOp) raises -> Array[DType.bool]:
    """Applies a connective to borrowed masks, the connective chosen at runtime.

    What `conjoin` and `disjoin` are underneath, for the callers that have the
    columns somewhere they cannot give up and do not know until they run which
    of the two they are applying. An execution plan is both of those: the masks
    are columns of a chunk it is holding, and the node carries its connective as
    a value.

    Parameters:
        o: The origin the borrowed masks come from.

    Args:
        columns: The masks, borrowed. Must all be the same length, and there
            must be at least one.
        op: The connective, which must be one that reads more than one column.

    Returns:
        A bool column, null only on the rows no mask settles.

    Raises:
        Error: If the connective is negation, no masks were given, or they are
            not all the same length.
    """
    if not op.reads_two_columns():
        raise Error("logic: not reads one column and was given a list")
    if op == LogicOp.AND:
        return _connect_all[LOGIC_AND](columns)
    return _connect_all[LOGIC_OR](columns)


def logical_not(a: Array[DType.bool]) raises -> Array[DType.bool]:
    """Negates a boolean column.

    The one connective with nothing three valued about it. A null negated is a
    null, because there is no second operand to settle it, so the validity comes
    across as it is and only the values turn over.

    Args:
        a: The column.

    Returns:
        A bool column, null wherever the input is.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[DType.bool]()

    var n = len(a)
    var out = Array[DType.bool](overwritten=n)
    var validity = Bitmap(copy=a.data.validity)
    var ones = SIMD[DType.bool, width](fill=True)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var src = a.unsafe_ptr()
        var dst = out.unsafe_mut_ptr()
        var i = start
        while i < stop:
            var x = src.unsafe_offset(i).unsafe_load[width=width]()
            # The lanewise not, which `^` with all ones is on a mask register.
            # There is no unary `not` on a vector for the same reason there is
            # no vector `<`: the operators that read as one answer are held to
            # width one.
            dst.unsafe_offset(i).unsafe_store(x ^ ones)
            i += width

        repair_range(out, validity, start, stop)

    parallel_morsels(compute, n)

    out.data.validity = validity^
    return out^


def _connect[
    op: Int
](a: Array[DType.bool], b: Array[DType.bool]) raises -> Array[DType.bool]:
    """Applies one of the two pairwise connectives elementwise.

    Args:
        a: The left column.
        b: The right column. Must be the same length as `a`.

    Parameters:
        op: One of the `LOGIC_` codes.

    Returns:
        A bool column.

    Raises:
        Error: If the two columns are different lengths.
    """
    comptime width = simd_width_of[DType.bool]()

    var n = len(a)
    if len(b) != n:
        raise Error(
            String(
                "logic: the left column has ",
                n,
                " rows and the right has ",
                len(b),
            )
        )

    # Every row is written below, so the zeroing constructor would be a wasted
    # pass. The rows that end up null are blanked by the repair at the end.
    var out = Array[DType.bool](overwritten=n)

    def compute(start: Int, stop: Int) {mut out, imm}:
        var lhs = a.unsafe_ptr()
        var rhs = b.unsafe_ptr()
        var dst = out.unsafe_mut_ptr()
        var i = start
        while i < stop:
            var x = lhs.unsafe_offset(i).unsafe_load[width=width]()
            var y = rhs.unsafe_offset(i).unsafe_load[width=width]()
            comptime if op == LOGIC_AND:
                dst.unsafe_offset(i).unsafe_store(x & y)
            else:
                dst.unsafe_offset(i).unsafe_store(x | y)
            i += width

    parallel_morsels(compute, n)

    var validity = combined_validity(a.data.validity, b.data.validity)
    if a.null_count() != 0 or b.null_count() != 0:
        _settle[op](a, b, out, validity, n)

    apply_validity(out, validity^)
    return out^


def _connect_all[
    op: Int, o: ImmOrigin
](columns: MaskRefs[o]) raises -> Array[DType.bool]:
    """Applies one of the two connectives across any number of columns.

    Args:
        columns: The columns, borrowed.

    Parameters:
        op: One of the `LOGIC_` codes.
        o: The origin the borrowed columns come from.

    Returns:
        A bool column.

    Raises:
        Error: If no columns were given or they are not all the same length.
    """
    comptime width = simd_width_of[DType.bool]()

    var count = len(columns)
    if count == 0:
        raise Error("logic: a connective over no columns has no answer")
    if count == 1:
        return columns[0][].copy()
    if count == 2:
        return _connect[op](columns[0][], columns[1][])

    var n = len(columns[0][])
    for k in range(1, count):
        if len(columns[k][]) != n:
            raise Error(
                String(
                    "logic: column 0 has ",
                    n,
                    " rows and column ",
                    k,
                    " has ",
                    len(columns[k][]),
                )
            )

    # Every row is written below, so the zeroing constructor would be a wasted
    # pass, the same as in the pairwise case.
    var out = Array[DType.bool](overwritten=n)

    def compute(start: Int, stop: Int) {mut out, imm}:
        # The value pointers are derived once per morsel rather than once per
        # vector. `columns[k].unsafe_ptr()` in the inner loop walks the column
        # to its data to its buffer for a load that is otherwise one
        # instruction, and the walk is the same answer every time.
        var srcs = List[Pointer[Scalar[DType.bool], ImmUntrackedOrigin]](
            capacity=count
        )
        for k in range(count):
            srcs.append(
                columns[k][]
                .unsafe_ptr()
                .unsafe_origin_cast[ImmUntrackedOrigin]()
            )

        var dst = out.unsafe_mut_ptr()
        var i = start
        while i < stop:
            var acc = srcs[0].unsafe_offset(i).unsafe_load[width=width]()
            for k in range(1, count):
                var y = srcs[k].unsafe_offset(i).unsafe_load[width=width]()
                comptime if op == LOGIC_AND:
                    acc = acc & y
                else:
                    acc = acc | y
            dst.unsafe_offset(i).unsafe_store(acc)
            i += width

    parallel_morsels(compute, n)

    var validity = Bitmap(copy=columns[0][].data.validity)
    var any_null = columns[0][].null_count() != 0
    for k in range(1, count):
        validity.and_with(columns[k][].data.validity)
        if columns[k][].null_count() != 0:
            any_null = True
    if any_null:
        _settle_all[op](columns, out, validity, n)

    apply_validity(out, validity^)
    return out^


def _settle_all[
    op: Int, o: ImmOrigin
](
    columns: MaskRefs[o],
    mut out: Array[DType.bool],
    mut validity: Bitmap,
    n: Int,
):
    """Turns back on the rows a null reaches that some other column decides.

    `_settle` generalised. Under an and a single false decides the row whatever
    the nulls beside it hold, and under an or a single true does, so a row is
    settled by the first present column holding the decisive value and the
    search stops there.

    Args:
        columns: The columns, borrowed.
        out: The values, already computed for the rows where every column is
            present.
        validity: The intersection on the way in, the answer's validity on the
            way out.
        n: The number of rows.

    Parameters:
        op: One of the `LOGIC_` codes.
        o: The origin the borrowed columns come from.
    """
    comptime decisive = op == LOGIC_OR

    var count = len(columns)
    for w in range(validity.word_count()):
        var word = validity.unsafe_word(w)
        if word == UInt64.MAX:
            continue

        var base = w * 64
        var last = min(base + 64, n)
        for row in range(base, last):
            if (word >> UInt64(row - base)) & 1 == 1:
                continue

            var settled = False
            for k in range(count):
                if (
                    columns[k][].is_valid(row)
                    and Bool(columns[k][][row]) == decisive
                ):
                    settled = True
                    break

            if settled:
                validity.set(row, True)
                out[row] = decisive


def _settle[
    op: Int
](
    a: Array[DType.bool],
    b: Array[DType.bool],
    mut out: Array[DType.bool],
    mut validity: Bitmap,
    n: Int,
):
    """Turns back on the rows a null reaches that the other side decides.

    This is the whole of the three valued rule. It walks the intersected
    validity a word at a time and skips the words with no null in them, so a
    column whose nulls are in one place costs one comparison per sixty four rows
    everywhere else.

    The decided rows are written here rather than left to the loop above, which
    got them right only if the null rows of both inputs hold a zero under the
    null. That is an invariant the kernels keep, and it is not one worth resting
    an answer on when the cost of not resting on it is a store on a row that is
    already being looked at.

    Args:
        a: The left column.
        b: The right column.
        out: The values, already computed for the rows where both inputs are
            present.
        validity: The intersection on the way in, the answer's validity on the
            way out.
        n: The number of rows.

    Parameters:
        op: One of the `LOGIC_` codes.
    """
    comptime decisive = op == LOGIC_OR

    for w in range(validity.word_count()):
        var word = validity.unsafe_word(w)
        if word == UInt64.MAX:
            continue

        var base = w * 64
        var last = min(base + 64, n)
        for row in range(base, last):
            if (word >> UInt64(row - base)) & 1 == 1:
                continue

            # A row that settles it is one where the operand that is present
            # already has the answer: a false under an and, a true under an or.
            var settled: Bool
            if a.is_valid(row):
                settled = Bool(a[row]) == decisive
            elif b.is_valid(row):
                settled = Bool(b[row]) == decisive
            else:
                settled = False

            if settled:
                validity.set(row, True)
                out[row] = decisive


def logic_any(a: AnyArray, b: AnyArray, op: LogicOp) raises -> AnyArray:
    """Applies a pairwise connective to two columns of any type.

    Args:
        a: The left column.
        b: The right column.
        op: The connective, which must be one that reads two columns.

    Returns:
        A bool column.

    Raises:
        Error: If either column is not boolean, or the connective is negation.
    """
    if not op.reads_two_columns():
        raise Error("logic: not reads one column and was given two")
    if a.dtype() != DType.bool or b.dtype() != DType.bool:
        raise Error(
            String(
                "logic: ",
                op,
                " reads two boolean columns, and was given ",
                a.type,
                " and ",
                b.type,
            )
        )
    if op == LogicOp.AND:
        return AnyArray(
            logical_and(a.as_typed[DType.bool](), b.as_typed[DType.bool]())
        )
    return AnyArray(
        logical_or(a.as_typed[DType.bool](), b.as_typed[DType.bool]())
    )


def logic_any(a: AnyArray, op: LogicOp) raises -> AnyArray:
    """Applies the one column connective, which is negation.

    Args:
        a: The column.
        op: The connective, which must be negation.

    Returns:
        A bool column.

    Raises:
        Error: If the column is not boolean, or the connective reads two.
    """
    if op.reads_two_columns():
        raise Error(
            String("logic: ", op, " reads two columns and was given one")
        )
    if a.dtype() != DType.bool:
        raise Error(
            String("logic: not reads a boolean column, and was given ", a.type)
        )
    return AnyArray(logical_not(a.as_typed[DType.bool]()))


def logic_all_any[
    o: ImmOrigin
](columns: ColumnRefs[o], at: List[Int], op: LogicOp) raises -> AnyArray:
    """Applies a connective to several of a borrowed set of columns at once.

    The entry point an execution plan uses, and the reason the positions are an
    argument rather than the caller slicing the list first: what it has is every
    column of a chunk, and the operands are some of them.

    Nothing is copied. The columns are read through their own storage, so a
    conjunction of five predicates over a million rows allocates the one answer
    and nothing else.

    Parameters:
        o: The origin the borrowed columns come from.

    Args:
        columns: Every column available, borrowed.
        at: The positions of the operands, in the order the query wrote them.
        op: The connective, which must be one that reads more than one column.

    Returns:
        A bool column.

    Raises:
        Error: If a position is outside the list, an operand is not boolean,
            the connective is negation, or the operands are different lengths.
    """
    if not op.reads_two_columns():
        raise Error("logic: not reads one column and was given a list")
    if len(at) == 0:
        raise Error("logic: a connective over no columns has no answer")

    var masks = MaskRefs[o](capacity=len(at))
    for i in range(len(at)):
        var k = at[i]
        if k < 0 or k >= len(columns):
            raise Error(
                String(
                    "logic: column ",
                    k,
                    " is outside a set of ",
                    len(columns),
                    " columns",
                )
            )
        if columns[k][].dtype() != DType.bool:
            raise Error(
                String(
                    "logic: ",
                    op,
                    " reads boolean columns, and was given ",
                    columns[k][].type,
                )
            )
        masks.append(
            Pointer(
                to=columns[k][].as_typed_view[DType.bool]()
            ).unsafe_origin_cast[o]()
        )

    return AnyArray(connect_all(masks, op))


def logic_type(op: LogicOp, operand: LogicalType) raises -> LogicalType:
    """Returns what a connective answers, which is a bool whatever it reads.

    It checks all the same, and the check is the point of asking: a connective
    over a column that is not boolean is a query that means nothing, and saying
    so before any row moves is better than saying it on the first chunk.

    Args:
        op: The connective.
        operand: The type of an operand.

    Returns:
        The boolean type.

    Raises:
        Error: If the operand is not boolean.
    """
    if operand != LogicalType.BOOL:
        raise Error(
            String(
                "logic: ",
                op,
                " reads a boolean column, and was given ",
                operand,
            )
        )
    return LogicalType.BOOL
