"""Windows with an order, a frame, or a function that is not a fold.

`Window` in `firepanda/exec/node.mojo` reduces a partition to one value and
writes it on every row, which is the whole of `sum(x) OVER (PARTITION BY k)`.
Everything else a window can say needs the rows of a partition in an order, and
this file is what answers it once they are: the running fold, the moving frame,
the ranks and the functions that read another row.

### The rows are sorted once, and everything is a position after that

The partition ordinal goes first and the window's own `ORDER BY` after it, and
one stable sort over those puts every partition in one run with its rows in
order. From there a partition is a pair of positions, a peer group, meaning a run
of rows the `ORDER BY` cannot tell apart, is another pair inside it, and a frame
is a third pair worked out from those two and the frame clause. Every answer is
computed in that sorted order and put back in the order the rows arrived in by
the inverse of the sort, which is one gather per window.

With no `ORDER BY` every row of a partition is a peer of every other, which is
what SQL says, and the rows keep the order they arrived in because the sort is
stable. So `row_number() OVER ()` numbers the rows as they came.

### A frame is two ends, and an exclusion makes it at most three runs

`ROWS`, `RANGE` and `GROUPS` differ only in how an end is found: by counting
rows, by searching for a value of the one order key, and by counting peer
groups. Both ends are then clipped to the partition and an end before the start
is an empty frame, which is a real answer rather than a mistake, since
`ROWS BETWEEN 1 FOLLOWING AND 1 FOLLOWING` on the last row has nothing in it.

`EXCLUDE` cuts a hole in the middle. `CURRENT ROW` cuts the row, `GROUP` cuts
its peer group, and `TIES` cuts the peer group and puts the row back. So a frame
is up to three runs in position order, the one before the hole, the row itself
when `TIES` keeps it, and the one after, and every function below reads a frame
as those three runs.

A `RANGE` bound with an offset reads the order key's value, as a double. A row
whose order key is null has no value to count from, so its offset bounds are
its peer group, which is the other nulls, and a row with a value never reaches
a null by an offset. That is Postgres' rule and DuckDB's.

### A fold over a frame is a fold over a handful of blocks

Summing each frame row by row is the width of the frame per row, which on the
default frame, everything from the start of the partition to the current row,
is the square of the partition. So a sum, a count, a minimum, a maximum and an
average are built out of blocks instead. The column is folded in pairs, then the
pairs in pairs, and so on, one column per level, and any run of rows is the
blocks it covers exactly, never more than two per level. A frame is then a fold
over at most twice the number of levels of already folded values, and the whole
window is one gather and one grouped fold over all of them at once, done by the
same kernel a `GROUP BY` uses so the types and the nulls come out the way a
group's would.

Those five are the ones that can be put together out of the same fold over
pieces, a count being a sum of counts and an average being a sum over a count.
Every other fold is asked over the frame's own rows, which costs the width of the
frame per row and is the right thing for the small frames those are written
with.

A sum reads no validity, so a frame of nothing but nulls and a frame that added
to zero are the same zero until a count separates them. The count is a second
tree beside the sum for that reason, and it is the same count an average divides
by.
"""

from firepanda.array.any import AnyArray, borrow_columns, empty_any
from firepanda.array.array import Array
from firepanda.dtype.logical import LogicalType
from firepanda.hash.grouping import group_ordinals
from firepanda.kernel.cast import cast_any
from firepanda.kernel.concat import concat_any, concat_two_any
from firepanda.kernel.group import AggKind, aggregate_group_any
from firepanda.kernel.select import take_any
from firepanda.kernel.sort import argsort_multi


comptime WINDOW_FOLD = 0
"""A fold over the frame, which is the aggregate the window's `op` names."""

comptime WINDOW_ROW_NUMBER = 1
"""`row_number()`, the row's place in its partition counting from one."""

comptime WINDOW_RANK = 2
"""`rank()`, the place of the row's peer group's first row."""

comptime WINDOW_DENSE_RANK = 3
"""`dense_rank()`, which peer group the row is in, counting from one."""

comptime WINDOW_PERCENT_RANK = 4
"""`percent_rank()`, the rank less one over the partition's rows less one."""

comptime WINDOW_CUME_DIST = 5
"""`cume_dist()`, the share of the partition up to the end of the row's peers."""

comptime WINDOW_NTILE = 6
"""`ntile(n)`, which of `n` nearly equal buckets the row falls in."""

comptime WINDOW_LAG = 7
"""`lag(x, n, d)`, the value `n` rows back in the partition."""

comptime WINDOW_LEAD = 8
"""`lead(x, n, d)`, the value `n` rows on in the partition."""

comptime WINDOW_FIRST_VALUE = 9
"""`first_value(x)`, the value on the frame's first row."""

comptime WINDOW_LAST_VALUE = 10
"""`last_value(x)`, the value on the frame's last row."""

comptime WINDOW_NTH_VALUE = 11
"""`nth_value(x, n)`, the value on the frame's `n`th row."""


comptime SPAN_DEFAULT = 0
"""No frame was written. The whole partition with no `ORDER BY`, and everything
up to the end of the row's peers with one."""

comptime SPAN_ROWS = 1
"""`ROWS`, which counts rows."""

comptime SPAN_RANGE = 2
"""`RANGE`, which counts by the value of the one order key."""

comptime SPAN_GROUPS = 3
"""`GROUPS`, which counts peer groups."""


comptime EDGE_NONE = 0
"""No bound written, which is an end bound when there is no `BETWEEN`, and
means the current row."""

comptime EDGE_PRECEDING = 1
"""`n PRECEDING`."""

comptime EDGE_FOLLOWING = 2
"""`n FOLLOWING`."""

comptime EDGE_UNBOUNDED_PRECEDING = 3
"""`UNBOUNDED PRECEDING`, the start of the partition."""

comptime EDGE_UNBOUNDED_FOLLOWING = 4
"""`UNBOUNDED FOLLOWING`, the end of the partition."""

comptime EDGE_CURRENT_ROW = 5
"""`CURRENT ROW`, which for `RANGE` and `GROUPS` is the row's peer group."""


comptime LEAVE_NONE = 0
"""No `EXCLUDE`, or `EXCLUDE NO OTHERS`, which says the same thing."""

comptime LEAVE_CURRENT_ROW = 1
"""`EXCLUDE CURRENT ROW`."""

comptime LEAVE_GROUP = 2
"""`EXCLUDE GROUP`, the row and its peers."""

comptime LEAVE_TIES = 3
"""`EXCLUDE TIES`, the row's peers but not the row."""


def window_function_named(name: StringSlice) -> Int:
    """Which window function a name is, or -1 if it is not one.

    The folds are not in here. A fold is an aggregate with an `OVER` on it and
    the aggregate registry already knows it; these are the names that are only
    ever a window.

    Args:
        name: The function name, already folded to lower case.

    Returns:
        One of the `WINDOW_` constants other than `WINDOW_FOLD`, or -1.
    """
    if name == "row_number":
        return WINDOW_ROW_NUMBER
    if name == "rank":
        return WINDOW_RANK
    if name == "dense_rank":
        return WINDOW_DENSE_RANK
    if name == "percent_rank":
        return WINDOW_PERCENT_RANK
    if name == "cume_dist":
        return WINDOW_CUME_DIST
    if name == "ntile":
        return WINDOW_NTILE
    if name == "lag":
        return WINDOW_LAG
    if name == "lead":
        return WINDOW_LEAD
    if name == "first_value":
        return WINDOW_FIRST_VALUE
    if name == "last_value":
        return WINDOW_LAST_VALUE
    if name == "nth_value":
        return WINDOW_NTH_VALUE
    return -1


def window_function_name(function: Int) -> String:
    """The name a window function is written with, for printing a plan.

    Args:
        function: One of the `WINDOW_` constants.

    Returns:
        The name, or `fold` for `WINDOW_FOLD`, whose name is its aggregate's.
    """
    if function == WINDOW_ROW_NUMBER:
        return "row_number"
    if function == WINDOW_RANK:
        return "rank"
    if function == WINDOW_DENSE_RANK:
        return "dense_rank"
    if function == WINDOW_PERCENT_RANK:
        return "percent_rank"
    if function == WINDOW_CUME_DIST:
        return "cume_dist"
    if function == WINDOW_NTILE:
        return "ntile"
    if function == WINDOW_LAG:
        return "lag"
    if function == WINDOW_LEAD:
        return "lead"
    if function == WINDOW_FIRST_VALUE:
        return "first_value"
    if function == WINDOW_LAST_VALUE:
        return "last_value"
    if function == WINDOW_NTH_VALUE:
        return "nth_value"
    return "fold"


def window_type(function: Int, argument: LogicalType) -> LogicalType:
    """The type a window function that is not a fold answers.

    Args:
        function: One of the `WINDOW_` constants other than `WINDOW_FOLD`.
        argument: The type of its first argument, and anything when it has none.

    Returns:
        A `BIGINT` for the four that count, a `DOUBLE` for the two that are a
        share, and the argument's own type for the five that read a row.
    """
    if (
        function == WINDOW_ROW_NUMBER
        or function == WINDOW_RANK
        or function == WINDOW_DENSE_RANK
        or function == WINDOW_NTILE
    ):
        return LogicalType.INT64
    if function == WINDOW_PERCENT_RANK or function == WINDOW_CUME_DIST:
        return LogicalType.FLOAT64
    return argument


def _edge_text(edge: Int, by: Float64) -> String:
    """Writes one frame bound as SQL spells it."""
    var amount = String(Int(by)) if by == Float64(Int(by)) else String(by)
    if edge == EDGE_PRECEDING:
        return String(amount, " preceding")
    if edge == EDGE_FOLLOWING:
        return String(amount, " following")
    if edge == EDGE_UNBOUNDED_PRECEDING:
        return "unbounded preceding"
    if edge == EDGE_UNBOUNDED_FOLLOWING:
        return "unbounded following"
    return "current row"


struct WindowFrame(Copyable, Movable, Writable):
    """Everything about a window beyond its partition keys and its order keys.

    What function it is, how many of the expression's operands are its
    arguments, the directions the order keys sort in, and the frame. It rides on
    the plan's window expression and is handed to the operator as it is, so the
    plan and the operator read one description rather than two that could
    drift.

    The default is a fold over one argument with no frame written, which is the
    only window there was before this, so a caller that builds a window without
    saying anything about a frame gets the window it always got.
    """

    var function: Int
    """One of the `WINDOW_` constants."""

    var args: Int
    """How many of the window expression's operands are the function's
    arguments. They come first, before the partition keys. One for a fold, none
    for `row_number()`, and two for `lag(x, 1, d)`, whose second is the
    default."""

    var amount: Int
    """The count a function takes as a constant: how far `lag` and `lead` reach,
    the buckets of `ntile` and which row `nth_value` reads. One elsewhere."""

    var mode: Int
    """One of the `SPAN_` constants."""

    var start: Int
    """The start bound, one of the `EDGE_` constants."""

    var end: Int
    """The end bound, one of the `EDGE_` constants."""

    var start_by: Float64
    """The start bound's offset, when it is `PRECEDING` or `FOLLOWING`."""

    var end_by: Float64
    """The end bound's offset, when it is `PRECEDING` or `FOLLOWING`."""

    var exclude: Int
    """One of the `LEAVE_` constants."""

    var ignore_nulls: Bool
    """Whether a function that reads another row skips the rows where its
    argument is null, which is `IGNORE NULLS`."""

    var descending: List[Bool]
    """Per order key, whether it sorts downwards."""

    var nulls_last: List[Bool]
    """Per order key, whether its nulls go at the end."""

    def __init__(out self):
        """A fold over one argument with no frame written."""
        self.function = WINDOW_FOLD
        self.args = 1
        self.amount = 1
        self.mode = SPAN_DEFAULT
        self.start = EDGE_NONE
        self.end = EDGE_NONE
        self.start_by = 0
        self.end_by = 0
        self.exclude = LEAVE_NONE
        self.ignore_nulls = False
        self.descending = List[Bool]()
        self.nulls_last = List[Bool]()

    def is_whole(self) -> Bool:
        """Whether this is a fold over the whole partition when nothing orders
        it, which is the one shape the partition only operator answers."""
        return (
            self.function == WINDOW_FOLD
            and self.mode == SPAN_DEFAULT
            and self.exclude == LEAVE_NONE
        )

    def order_text(self, at: Int) -> String:
        """The direction words for one order key, empty for the default.

        Args:
            at: Which order key.

        Returns:
            ` desc`, ` nulls first` and so on, each with a leading space.
        """
        var down = at < len(self.descending) and self.descending[at]
        var last = self.nulls_last[at] if at < len(self.nulls_last) else True
        var out = String(" desc") if down else String()
        if not last:
            out += " nulls first"
        return out^

    def write_to[W: Writer](self, mut writer: W):
        """Writes the frame and the modifiers as SQL spells them, or nothing.

        Everything that decides what the window computes goes in, because the
        common subexpression pass keys a window on what this writes as well as
        on its operands, and two windows that differ only in their frame are
        two windows.

        Args:
            writer: Where it goes.
        """
        if self.function == WINDOW_NTILE or self.function == WINDOW_NTH_VALUE:
            writer.write(" #", self.amount)
        elif self.function == WINDOW_LAG or self.function == WINDOW_LEAD:
            writer.write(" #", self.amount)
        if self.ignore_nulls:
            writer.write(" ignore nulls")
        if self.mode != SPAN_DEFAULT:
            if self.mode == SPAN_ROWS:
                writer.write(" rows")
            elif self.mode == SPAN_RANGE:
                writer.write(" range")
            else:
                writer.write(" groups")
            if self.end == EDGE_NONE:
                writer.write(" ", _edge_text(self.start, self.start_by))
            else:
                writer.write(
                    " between ",
                    _edge_text(self.start, self.start_by),
                    " and ",
                    _edge_text(self.end, self.end_by),
                )
        if self.exclude == LEAVE_CURRENT_ROW:
            writer.write(" exclude current row")
        elif self.exclude == LEAVE_GROUP:
            writer.write(" exclude group")
        elif self.exclude == LEAVE_TIES:
            writer.write(" exclude ties")

    def key(self) -> String:
        """Everything this holds, written down, for comparing two windows.

        Returns:
            The frame as `write_to` writes it with the function and the order
            directions in front.
        """
        var out = String(self.function, "/", self.args)
        for i in range(len(self.descending)):
            out += self.order_text(i)
            out += ";"
        out += String(self)
        return out^


struct _Layout(Movable):
    """Where each sorted row's partition and peer group start and end.

    Every list is indexed by sorted position, apart from `group_lo` and
    `group_hi`, which are indexed by a peer group's number."""

    var perm: List[Int]
    """The input row at each sorted position."""

    var part_lo: List[Int]
    """The first position of the row's partition."""

    var part_hi: List[Int]
    """One past the last position of the row's partition."""

    var peer_lo: List[Int]
    """The first position of the row's peer group."""

    var peer_hi: List[Int]
    """One past the last position of the row's peer group."""

    var group: List[Int]
    """The number of the row's peer group, counted over the whole column."""

    var group_lo: List[Int]
    """Per peer group, its first position."""

    var group_hi: List[Int]
    """Per peer group, one past its last position."""

    def __init__(out self):
        self.perm = List[Int]()
        self.part_lo = List[Int]()
        self.part_hi = List[Int]()
        self.peer_lo = List[Int]()
        self.peer_hi = List[Int]()
        self.group = List[Int]()
        self.group_lo = List[Int]()
        self.group_hi = List[Int]()


def _layout(
    flat: List[AnyArray],
    keys: List[Int],
    order: List[Int],
    descending: List[Bool],
    nulls_last: List[Bool],
    rows: Int,
) raises -> _Layout:
    """Sorts the rows by partition and then by the order keys, and finds the
    runs.

    Args:
        flat: The input columns, whole.
        keys: The partition columns.
        order: The order columns.
        descending: Per order column, whether it sorts downwards.
        nulls_last: Per order column, whether its nulls go last.
        rows: The row count, which is more than zero.

    Returns:
        The sort and the runs.
    """
    var codes = Array[DType.uint32](rows)
    if len(keys) > 0:
        var refs = borrow_columns(flat)
        codes = group_ordinals(refs, keys, rows).into_codes()

    var cols = List[AnyArray](capacity=1 + len(order))
    var down = List[Bool](capacity=1 + len(order))
    var first = List[Bool](capacity=1 + len(order))
    cols.append(AnyArray(codes.copy()))
    down.append(False)
    first.append(False)
    for k in range(len(order)):
        cols.append(flat[order[k]].copy())
        down.append(descending[k])
        first.append(not nulls_last[k])
    var sorted = argsort_multi(cols, down, first)

    # A peer group is the partition and the order keys together, so it is a
    # grouping over the same columns the sort just read. Equal tuples sort next
    # to each other, so a change of ordinal between neighbours is a boundary.
    var peers = codes.copy()
    if len(order) > 0:
        var every = List[Int](capacity=len(cols))
        for k in range(len(cols)):
            every.append(k)
        var refs = borrow_columns(cols)
        peers = group_ordinals(refs, every, rows).into_codes()

    var out = _Layout()
    out.perm = List[Int](capacity=rows)
    for p in range(rows):
        out.perm.append(Int(sorted[p]))
    out.part_lo = List[Int](length=rows, fill=0)
    out.part_hi = List[Int](length=rows, fill=0)
    out.peer_lo = List[Int](length=rows, fill=0)
    out.peer_hi = List[Int](length=rows, fill=0)
    out.group = List[Int](length=rows, fill=0)

    var part_at = 0
    var peer_at = 0
    for p in range(rows):
        var row = out.perm[p]
        var fresh_part = p == 0 or codes[row] != codes[out.perm[p - 1]]
        var fresh_peer = fresh_part or peers[row] != peers[out.perm[p - 1]]
        if fresh_part:
            part_at = p
        if fresh_peer:
            peer_at = p
            if p > 0:
                out.group_hi.append(p)
            out.group_lo.append(p)
        out.part_lo[p] = part_at
        out.peer_lo[p] = peer_at
        out.group[p] = len(out.group_lo) - 1
    out.group_hi.append(rows)

    var part_end = rows
    var peer_end = rows
    for p in range(rows - 1, -1, -1):
        out.part_hi[p] = part_end
        out.peer_hi[p] = peer_end
        if out.part_lo[p] == p:
            part_end = p
        if out.peer_lo[p] == p:
            peer_end = p
    return out^


def _first_at_least(
    values: List[Float64], lo: Int, hi: Int, target: Float64
) -> Int:
    """The first position in `[lo, hi)` whose value is at least `target`."""
    var a = lo
    var b = hi
    while a < b:
        var mid = (a + b) // 2
        if values[mid] < target:
            a = mid + 1
        else:
            b = mid
    return a


def _first_above(
    values: List[Float64], lo: Int, hi: Int, target: Float64
) -> Int:
    """The first position in `[lo, hi)` whose value is more than `target`."""
    var a = lo
    var b = hi
    while a < b:
        var mid = (a + b) // 2
        if values[mid] <= target:
            a = mid + 1
        else:
            b = mid
    return a


struct _Runs(Movable):
    """A frame per sorted row, as the three runs an exclusion leaves.

    Each run is `[lo, hi)` in sorted positions and is empty when the two are
    equal. The runs are in position order, so reading `a` then `b` then `c` is
    reading the frame front to back."""

    var a_lo: List[Int]
    var a_hi: List[Int]
    var b_lo: List[Int]
    var b_hi: List[Int]
    var c_lo: List[Int]
    var c_hi: List[Int]

    def __init__(out self, rows: Int):
        self.a_lo = List[Int](length=rows, fill=0)
        self.a_hi = List[Int](length=rows, fill=0)
        self.b_lo = List[Int](length=rows, fill=0)
        self.b_hi = List[Int](length=rows, fill=0)
        self.c_lo = List[Int](length=rows, fill=0)
        self.c_hi = List[Int](length=rows, fill=0)

    def lo(self, run: Int, p: Int) -> Int:
        if run == 0:
            return self.a_lo[p]
        if run == 1:
            return self.b_lo[p]
        return self.c_lo[p]

    def hi(self, run: Int, p: Int) -> Int:
        if run == 0:
            return self.a_hi[p]
        if run == 1:
            return self.b_hi[p]
        return self.c_hi[p]


def _runs(
    frame: WindowFrame,
    lay: _Layout,
    flat: List[AnyArray],
    order: List[Int],
    ordered: Bool,
    rows: Int,
) raises -> _Runs:
    """Works out every sorted row's frame.

    Args:
        frame: The frame clause.
        lay: The sort and its runs.
        flat: The input columns, for a `RANGE` offset's order key.
        order: The order columns.
        ordered: Whether the window has an `ORDER BY`.
        rows: The row count.

    Returns:
        The frame of each row, as runs.

    Raises:
        If a `RANGE` offset is asked of a window that does not order by exactly
        one key, or of a key that is not a number.
    """
    var start = frame.start
    var end = frame.end
    var mode = frame.mode
    if mode == SPAN_DEFAULT:
        # No frame is the whole partition without an ORDER BY and everything up
        # to the row's last peer with one, which is RANGE UNBOUNDED PRECEDING.
        mode = SPAN_RANGE
        start = EDGE_UNBOUNDED_PRECEDING
        end = EDGE_CURRENT_ROW if ordered else EDGE_UNBOUNDED_FOLLOWING
    elif end == EDGE_NONE:
        end = EDGE_CURRENT_ROW

    # A RANGE bound that counts reads the order key as a number, and a row
    # whose key is null has no number. Those go to the peer group instead,
    # and a row with a number only searches the part of the partition that
    # has numbers, which is the partition less its run of nulls at one end.
    var values = List[Float64]()
    var present = List[Bool]()
    var counted = mode == SPAN_RANGE and (
        start == EDGE_PRECEDING
        or start == EDGE_FOLLOWING
        or end == EDGE_PRECEDING
        or end == EDGE_FOLLOWING
    )
    if counted:
        if len(order) != 1:
            raise Error(
                String(
                    "a RANGE frame with an offset orders by one key, and this"
                    " window orders by ",
                    len(order),
                )
            )
        var key = flat[order[0]].copy()
        if not key.type.is_numeric():
            raise Error(
                String(
                    "a RANGE frame with an offset counts along a number, and"
                    " this window orders by a ",
                    key.type,
                )
            )
        var sorted = take_any(key, lay.perm)
        var asked = cast_any(sorted, LogicalType.FLOAT64, strict=False)
        var typed = asked.as_typed[DType.float64]()
        var down = len(frame.descending) > 0 and frame.descending[0]
        values = List[Float64](capacity=rows)
        present = List[Bool](capacity=rows)
        for p in range(rows):
            var v = typed[p]
            values.append(-v if down else v)
            present.append(sorted.is_valid(p))

    # The run of values in each row's partition, for the search.
    var known_lo = List[Int]()
    var known_hi = List[Int]()
    if counted:
        known_lo = List[Int](length=rows, fill=0)
        known_hi = List[Int](length=rows, fill=0)
        var p = 0
        while p < rows:
            var lo = lay.part_lo[p]
            var hi = lay.part_hi[p]
            var a = lo
            var b = hi
            while a < b and not present[a]:
                a += 1
            while b > a and not present[b - 1]:
                b -= 1
            for q in range(lo, hi):
                known_lo[q] = a
                known_hi[q] = b
            p = hi

    var out = _Runs(rows)
    for p in range(rows):
        var part_lo = lay.part_lo[p]
        var part_hi = lay.part_hi[p]
        var lo: Int
        var hi: Int

        # The start.
        if start == EDGE_UNBOUNDED_PRECEDING:
            lo = part_lo
        elif start == EDGE_UNBOUNDED_FOLLOWING:
            lo = part_hi
        elif start == EDGE_CURRENT_ROW:
            lo = p if mode == SPAN_ROWS else lay.peer_lo[p]
        elif mode == SPAN_ROWS:
            var n = Int(frame.start_by)
            lo = p - n if start == EDGE_PRECEDING else p + n
        elif mode == SPAN_GROUPS:
            var n = Int(frame.start_by)
            var g = lay.group[p] - n if start == EDGE_PRECEDING else lay.group[
                p
            ] + n
            var first = lay.group[part_lo]
            var last = lay.group[part_hi - 1]
            if g < first:
                lo = part_lo
            elif g > last:
                lo = part_hi
            else:
                lo = lay.group_lo[g]
        elif not present[p]:
            lo = lay.peer_lo[p]
        else:
            var by = frame.start_by if start == EDGE_FOLLOWING else -frame.start_by
            lo = _first_at_least(
                values, known_lo[p], known_hi[p], values[p] + by
            )

        # The end, one past the last row in the frame.
        if end == EDGE_UNBOUNDED_FOLLOWING:
            hi = part_hi
        elif end == EDGE_UNBOUNDED_PRECEDING:
            hi = part_lo
        elif end == EDGE_CURRENT_ROW:
            hi = p + 1 if mode == SPAN_ROWS else lay.peer_hi[p]
        elif mode == SPAN_ROWS:
            var n = Int(frame.end_by)
            hi = (p - n if end == EDGE_PRECEDING else p + n) + 1
        elif mode == SPAN_GROUPS:
            var n = Int(frame.end_by)
            var g = lay.group[p] - n if end == EDGE_PRECEDING else lay.group[
                p
            ] + n
            var first = lay.group[part_lo]
            var last = lay.group[part_hi - 1]
            if g < first:
                hi = part_lo
            elif g > last:
                hi = part_hi
            else:
                hi = lay.group_hi[g]
        elif not present[p]:
            hi = lay.peer_hi[p]
        else:
            var by = frame.end_by if end == EDGE_FOLLOWING else -frame.end_by
            hi = _first_above(values, known_lo[p], known_hi[p], values[p] + by)

        lo = max(part_lo, min(lo, part_hi))
        hi = max(part_lo, min(hi, part_hi))
        if hi < lo:
            hi = lo

        # The hole an exclusion cuts, and the row TIES puts back.
        var cut_lo = hi
        var cut_hi = hi
        if frame.exclude == LEAVE_CURRENT_ROW:
            cut_lo = p
            cut_hi = p + 1
        elif frame.exclude == LEAVE_GROUP or frame.exclude == LEAVE_TIES:
            cut_lo = lay.peer_lo[p]
            cut_hi = lay.peer_hi[p]
        out.a_lo[p] = lo
        out.a_hi[p] = max(lo, min(hi, cut_lo))
        out.c_lo[p] = max(min(hi, cut_hi), lo)
        out.c_hi[p] = hi
        if frame.exclude == LEAVE_NONE:
            out.a_hi[p] = hi
            out.c_lo[p] = hi
        if frame.exclude == LEAVE_TIES and p >= lo and p < hi:
            out.b_lo[p] = p
            out.b_hi[p] = p + 1
        else:
            out.b_lo[p] = out.a_hi[p]
            out.b_hi[p] = out.a_hi[p]
    return out^


def _identity(rows: Int) -> Array[DType.uint32]:
    """Codes that put every row in a group of its own."""
    var codes = Array[DType.uint32](rows)
    for i in range(rows):
        codes[i] = UInt32(i)
    return codes^


def _tree(
    base: AnyArray, merge: AggKind, as_float: Bool
) raises -> List[AnyArray]:
    """Folds a column in pairs, then those in pairs, one column per level.

    Level `k` holds one value per block of `2**k` rows, block `j` being the rows
    from `j * 2**k`, and the last block of a level may be short.

    Args:
        base: The values, one per row, already in the type every level has.
        merge: The fold that puts two blocks together.
        as_float: Whether a sum is taken in doubles.

    Returns:
        The levels, the first being `base`.
    """
    var levels = List[AnyArray]()
    levels.append(base.copy())
    var n = len(base)
    while n > 1:
        var up = (n + 1) // 2
        var codes = Array[DType.uint32](n)
        for i in range(n):
            codes[i] = UInt32(i >> 1)
        var next = aggregate_group_any(
            levels[len(levels) - 1],
            merge,
            codes,
            up,
            trusted=True,
            as_float=as_float,
        )
        levels.append(next^)
        n = up
    return levels^


def _ask(
    levels: List[AnyArray],
    merge: AggKind,
    runs: _Runs,
    rows: Int,
    as_float: Bool,
) raises -> AnyArray:
    """Folds every row's frame out of the blocks that cover it exactly.

    Args:
        levels: The tree.
        merge: The fold that puts blocks together.
        runs: The frames.
        rows: The row count.
        as_float: Whether a sum is taken in doubles.

    Returns:
        One value per sorted row.
    """
    var depth = len(levels)
    var picks = List[List[Int]]()
    var whose = List[List[Int]]()
    for _ in range(depth):
        picks.append(List[Int]())
        whose.append(List[Int]())
    for p in range(rows):
        for run in range(3):
            var lo = runs.lo(run, p)
            var hi = runs.hi(run, p)
            while lo < hi:
                # The widest block that starts here and does not run past the
                # end, which is the whole of the decomposition.
                var k = 0
                while k + 1 < depth:
                    var wide = 1 << (k + 1)
                    if lo % wide != 0 or lo + wide > hi:
                        break
                    k += 1
                picks[k].append(lo >> k)
                whose[k].append(p)
                lo += 1 << k

    var parts = List[AnyArray]()
    var total = 0
    for k in range(depth):
        if len(picks[k]) > 0:
            parts.append(take_any(levels[k], picks[k]))
            total += len(picks[k])
    var codes = Array[DType.uint32](total)
    var at = 0
    for k in range(depth):
        for i in range(len(whose[k])):
            codes[at] = UInt32(whose[k][i])
            at += 1
    var column: AnyArray
    if len(parts) == 0:
        column = empty_any(levels[0].type)
    elif len(parts) == 1:
        column = parts[0].copy()
    else:
        column = concat_any(parts)
    return aggregate_group_any(
        column, merge, codes, rows, trusted=True, as_float=as_float
    )


def _counts(source: AnyArray, runs: _Runs, rows: Int) raises -> AnyArray:
    """How many values in each row's frame are not null."""
    var base = aggregate_group_any(
        source, AggKind.COUNT, _identity(rows), rows, trusted=True
    )
    return _ask(_tree(base, AggKind.SUM, False), AggKind.SUM, runs, rows, False)


def _fold(
    source: AnyArray,
    kind: AggKind,
    marked: Bool,
    runs: _Runs,
    rows: Int,
) raises -> AnyArray:
    """Folds each sorted row's frame.

    Args:
        source: The argument, in sorted order.
        kind: The fold.
        marked: Whether a sum over nothing but nulls answers null.
        runs: The frames.
        rows: The row count.

    Returns:
        One value per sorted row.
    """
    if kind == AggKind.MIN or kind == AggKind.MAX:
        var base = aggregate_group_any(
            source, kind, _identity(rows), rows, trusted=True
        )
        return _ask(_tree(base, kind, False), kind, runs, rows, False)
    if kind == AggKind.COUNT:
        return _counts(source, runs, rows)
    if kind == AggKind.SUM or kind == AggKind.MEAN:
        var mean = kind == AggKind.MEAN
        var base = aggregate_group_any(
            source, AggKind.SUM, _identity(rows), rows, trusted=True,
            as_float=mean,
        )
        var sums = _ask(
            _tree(base, AggKind.SUM, mean), AggKind.SUM, runs, rows, mean
        )
        if not mean and not marked:
            return sums^
        var counts = _counts(source, runs, rows).as_typed[DType.int64]()
        if mean:
            var total = sums.as_typed[DType.float64]()
            var out = Array[DType.float64](rows)
            for p in range(rows):
                if counts[p] == 0:
                    out.set_null(p)
                else:
                    out[p] = total[p] / Float64(counts[p])
            return AnyArray(out^)
        var keep = List[Int](capacity=rows)
        for p in range(rows):
            keep.append(p if counts[p] != 0 else -1)
        return take_any(sums, keep)

    # Everything else is asked of the frame's own rows.
    var picks = List[Int]()
    var whose = List[Int]()
    for p in range(rows):
        for run in range(3):
            for q in range(runs.lo(run, p), runs.hi(run, p)):
                picks.append(q)
                whose.append(p)
    var codes = Array[DType.uint32](len(whose))
    for i in range(len(whose)):
        codes[i] = UInt32(whose[i])
    var column = take_any(source, picks) if len(
        picks
    ) > 0 else empty_any(source.type)
    return aggregate_group_any(column, kind, codes, rows, trusted=True)


def _present_before(source: AnyArray, rows: Int) -> List[Int]:
    """How many values before each sorted position are not null, with one more
    entry at the end for the whole column."""
    var out = List[Int](capacity=rows + 1)
    out.append(0)
    for p in range(rows):
        out.append(out[p] + (1 if source.is_valid(p) else 0))
    return out^


def _nth_present(seen: List[Int], lo: Int, hi: Int, n: Int) -> Int:
    """The position of the `n`th value in `[lo, hi)` that is not null, counting
    from one, or -1 when there are fewer."""
    var want = seen[lo] + n
    if n < 1 or seen[hi] < want:
        return -1
    # The first q with seen[q + 1] >= want.
    var a = lo
    var b = hi
    while a < b:
        var mid = (a + b) // 2
        if seen[mid + 1] < want:
            a = mid + 1
        else:
            b = mid
    return a


def _pick(
    runs: _Runs,
    p: Int,
    n: Int,
    backwards: Bool,
    seen: List[Int],
    skip: Bool,
) -> Int:
    """The position of the `n`th row of a frame, from the front or the back.

    Args:
        runs: The frames.
        p: The row whose frame it is.
        n: Which row, counting from one.
        backwards: Whether to count from the back.
        seen: The count of values that are not null before each position.
        skip: Whether a null row is passed over rather than counted.

    Returns:
        The sorted position, or -1 when the frame has fewer rows than that.
    """
    var left = n
    for i in range(3):
        var run = 2 - i if backwards else i
        var lo = runs.lo(run, p)
        var hi = runs.hi(run, p)
        if hi <= lo:
            continue
        var size = (seen[hi] - seen[lo]) if skip else hi - lo
        if left <= size:
            if not skip:
                return hi - left if backwards else lo + left - 1
            if backwards:
                return _nth_present(seen, lo, hi, size - left + 1)
            return _nth_present(seen, lo, hi, left)
        left -= size
    return -1


def framed_windows(
    flat: List[AnyArray],
    keys: List[Int],
    order: List[Int],
    frames: List[WindowFrame],
    sources: List[Int],
    seconds: List[Int],
    kinds: List[AggKind],
    marked: List[Bool],
    types: List[LogicalType],
) raises -> List[AnyArray]:
    """Computes windows that share a partitioning and an ordering.

    Args:
        flat: The input columns, whole.
        keys: The partition columns.
        order: The order columns, the same for every window here.
        frames: Per window, the function, the frame and the directions the
            order columns sort in. The directions are read off the first.
        sources: Per window, the column of its first argument, or -1.
        seconds: Per window, the column of its second argument, or -1.
        kinds: Per window, the fold, read only when the function is one.
        marked: Per window, whether a sum over nothing answers null.
        types: Per window, the type it answers in.

    Returns:
        One column per window, in the order the rows arrived.
    """
    var rows = len(flat[0]) if len(flat) > 0 else 0
    var out = List[AnyArray](capacity=len(frames))
    if rows == 0:
        for a in range(len(frames)):
            out.append(empty_any(types[a]))
        return out^

    var descending = List[Bool]()
    var nulls_last = List[Bool]()
    for k in range(len(order)):
        var down = k < len(frames[0].descending) and frames[0].descending[k]
        descending.append(down)
        nulls_last.append(
            frames[0].nulls_last[k] if k < len(frames[0].nulls_last) else True
        )
    var lay = _layout(flat, keys, order, descending, nulls_last, rows)

    var back = List[Int](length=rows, fill=0)
    for p in range(rows):
        back[lay.perm[p]] = p

    for a in range(len(frames)):
        ref frame = frames[a]
        var function = frame.function
        var made: AnyArray

        if (
            function == WINDOW_ROW_NUMBER
            or function == WINDOW_RANK
            or function == WINDOW_DENSE_RANK
            or function == WINDOW_NTILE
        ):
            var counted = Array[DType.int64](rows)
            for p in range(rows):
                var lo = lay.part_lo[p]
                var answer: Int
                if function == WINDOW_ROW_NUMBER:
                    answer = p - lo + 1
                elif function == WINDOW_RANK:
                    answer = lay.peer_lo[p] - lo + 1
                elif function == WINDOW_DENSE_RANK:
                    answer = lay.group[p] - lay.group[lo] + 1
                else:
                    # DuckDB's split: the first `size % buckets` buckets take
                    # one row more than the rest.
                    var size = lay.part_hi[p] - lo
                    var buckets = min(frame.amount, size)
                    var small = size // buckets
                    var large = size - small * buckets
                    var into = large * (small + 1)
                    var at = p - lo
                    if at < into:
                        answer = 1 + at // (small + 1)
                    else:
                        answer = 1 + large + (at - into) // small
                counted[p] = Int64(answer)
            made = AnyArray(counted^)
        elif function == WINDOW_PERCENT_RANK or function == WINDOW_CUME_DIST:
            var share = Array[DType.float64](rows)
            for p in range(rows):
                var lo = lay.part_lo[p]
                var size = lay.part_hi[p] - lo
                if function == WINDOW_CUME_DIST:
                    share[p] = Float64(lay.peer_hi[p] - lo) / Float64(size)
                elif size > 1:
                    share[p] = Float64(lay.peer_lo[p] - lo) / Float64(size - 1)
                else:
                    share[p] = 0
            made = AnyArray(share^)
        else:
            var source = take_any(flat[sources[a]], lay.perm)
            if function == WINDOW_FOLD:
                var runs = _runs(frame, lay, flat, order, len(order) > 0, rows)
                made = _fold(source, kinds[a], marked[a], runs, rows)
            elif function == WINDOW_LAG or function == WINDOW_LEAD:
                var reach = frame.amount
                if function == WINDOW_LEAD:
                    reach = -reach
                var seen = List[Int]()
                if frame.ignore_nulls:
                    seen = _present_before(source, rows)
                var fallback = seconds[a] >= 0
                var picks = List[Int](capacity=rows)
                for p in range(rows):
                    var lo = lay.part_lo[p]
                    var hi = lay.part_hi[p]
                    var at = -1
                    if reach == 0:
                        at = p
                    elif not frame.ignore_nulls:
                        var q = p - reach
                        if q >= lo and q < hi:
                            at = q
                    elif reach > 0:
                        var have = seen[p] - seen[lo]
                        if have >= reach:
                            at = _nth_present(seen, lo, p, have - reach + 1)
                    else:
                        at = _nth_present(seen, p + 1, hi, -reach)
                    if at < 0 and fallback:
                        at = rows + p
                    picks.append(at)
                if fallback:
                    var other = take_any(flat[seconds[a]], lay.perm)
                    if other.type != source.type:
                        other = cast_any(other, source.type, strict=False)
                    made = take_any(concat_two_any(source, other), picks)
                else:
                    made = take_any(source, picks)
            else:
                var runs = _runs(frame, lay, flat, order, len(order) > 0, rows)
                var seen = _present_before(source, rows)
                var picks = List[Int](capacity=rows)
                for p in range(rows):
                    if function == WINDOW_FIRST_VALUE:
                        picks.append(
                            _pick(runs, p, 1, False, seen, frame.ignore_nulls)
                        )
                    elif function == WINDOW_LAST_VALUE:
                        picks.append(
                            _pick(runs, p, 1, True, seen, frame.ignore_nulls)
                        )
                    else:
                        picks.append(
                            _pick(
                                runs,
                                p,
                                frame.amount,
                                False,
                                seen,
                                frame.ignore_nulls,
                            )
                        )
                made = take_any(source, picks)
        out.append(take_any(made, back))
    return out^
