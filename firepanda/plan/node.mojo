"""The nine logical nodes a query is, and the arena they live in.

The same collision as `exec/node.mojo` and the right one, because these are the
same concept at two levels. A node here says what the query wants. A node there
says how one chunk of it is computed. The list here is shorter than the list
there, since several physical nodes are alternative implementations of one
logical node, and it is shorter than it looks: nine kinds cover every TPC-H
query.

Scan, Filter, Project, Aggregate, Join, Sort, Limit, Distinct and Union. That is
the list in `docs/specs/planner/01-what-a-plan-is.md` and the spec says to refuse
to add to it without an argument, so it is written down in `NodeKind` and
nowhere else.

## The arena, and why the expressions are in it

A `Plan` holds the nodes and the expression arena together. A node holds
expressions by index and an index only means something against an arena, so the
two travel as one value rather than as two that a caller has to keep paired. A
plan is then one thing to pass, one thing to rewrite and one thing to print.

Node indices are handed out in creation order like expression indices are, so an
input is always at a lower index than the node that reads it and a pass can walk
the list backwards to visit every node after its inputs.

## What is checked here and what is not

Everything that can be checked without knowing the schema. That a node's inputs
exist, that a project has a name for each output, that a sort has a direction
for each key, that a join has as many keys on one side as on the other, and that
the expressions in the positions that have to be elementwise are.

That last one is the only interesting check and it is worth naming. A filter's
predicate is evaluated a row at a time, so an aggregate in it is not a filter
with an aggregate in it, it is a query the caller has not written yet. Same for
a group key, a join key and a sort key. Catching it here makes it a plan error
with the node named rather than a kernel error about a length mismatch several
layers down.

What is not checked here is anything that needs a schema. Whether a name
resolves, whether a predicate is a boolean, whether the two sides of a union
have the same width. Those are binding's, and binding is `bind.mojo`.
"""

from firepanda.join.pairs import JoinKind
from firepanda.plan.expr import UNBOUND, ExprKind, Expressions


@fieldwise_init
struct NodeKind(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which of the nine kinds a logical node is."""

    var code: Int
    """The kind, as one of the nine values below."""

    comptime SCAN = Self(0)
    """A table, a file or an in memory frame. The only node with no input, and
    the one that pushdown pushes into."""

    comptime FILTER = Self(1)
    """One predicate over its input."""

    comptime PROJECT = Self(2)
    """A list of output expressions. `select` and `with_columns` are both this,
    and merging adjacent ones is a rewrite."""

    comptime AGGREGATE = Self(3)
    """Group keys and aggregate expressions. An empty key list is a whole frame
    reduction, which is a different physical operator and the same logical
    node."""

    comptime JOIN = Self(4)
    """Two inputs, a key pair list and a kind."""

    comptime SORT = Self(5)
    """Keys, directions and null placement."""

    comptime LIMIT = Self(6)
    """An offset and a length. Separate from `SORT` because a limit on an
    unsorted input is a different operator and because slice pushdown treats it
    separately."""

    comptime DISTINCT = Self(7)
    """Keys, or the whole row when the key list is empty."""

    comptime UNION = Self(8)
    """Several inputs stacked. Covers concat as well, since the difference is
    whether duplicates survive and that is a flag rather than a node."""

    def __eq__(self, other: Self) -> Bool:
        """Compares two kinds.

        Args:
            other: The kind to compare against.

        Returns:
            True if they are the same kind.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two kinds for difference.

        Args:
            other: The kind to compare against.

        Returns:
            True if they are different kinds.
        """
        return self.code != other.code

    def write_to(self, mut writer: Some[Writer]):
        """Writes the kind as the word an explain output uses for it.

        Args:
            writer: Where to write.
        """
        if self == Self.SCAN:
            writer.write("SCAN")
        elif self == Self.FILTER:
            writer.write("FILTER")
        elif self == Self.PROJECT:
            writer.write("PROJECT")
        elif self == Self.AGGREGATE:
            writer.write("AGGREGATE")
        elif self == Self.JOIN:
            writer.write("JOIN")
        elif self == Self.SORT:
            writer.write("SORT")
        elif self == Self.LIMIT:
            writer.write("LIMIT")
        elif self == Self.DISTINCT:
            writer.write("DISTINCT")
        else:
            writer.write("UNION")


comptime NO_LIMIT = -1
"""What a `LIMIT` holds for its length when it only skips. An offset with no
length is a real thing to write and zero is a real length, so the absence needs a
value of its own."""


struct PlanNode(Copyable, Movable):
    """One node of a logical plan.

    One struct for all nine kinds, on the same grounds as `Expr`: the arena
    holds them in one list and a list has one element type. Which fields a kind
    uses is documented on the builder that makes it.
    """

    var kind: NodeKind
    """Which of the nine this is."""

    var inputs: List[Int]
    """The nodes this one reads, as plan arena indices. Empty on a `SCAN`, one
    on the four in the middle, two on a `JOIN`, and any number on a `UNION`."""

    var exprs: List[Int]
    """The expressions, as expression arena indices, in the order the builder
    documents. Where a kind has two lists of them, `parts` says where the first
    one ends."""

    var parts: Int
    """How many of `exprs` belong to the first of two lists. The group key count
    on an `AGGREGATE` and the left key count on a `JOIN`, and zero elsewhere."""

    var names: List[String]
    """The output names on a `PROJECT` and an `AGGREGATE`, and the column names
    read on a `SCAN`. Empty elsewhere."""

    var flags: List[Bool]
    """The directions on a `SORT`, descending first and nulls last after, each
    as long as the key list. One entry on a `UNION`, true when duplicates
    survive. Empty elsewhere."""

    var op: Int
    """The `JoinKind` code on a `JOIN`. Zero elsewhere, which is a real code, so
    read it only after checking the kind."""

    var offset: Int
    """The rows a `LIMIT` skips. Zero elsewhere."""

    var length: Int
    """The rows a `LIMIT` keeps, or `NO_LIMIT`. Zero elsewhere."""

    var table: Int
    """Which relation a `SCAN` is, as the bit index a bound column reference
    carries. `UNBOUND` elsewhere, and a single input plan leaves it at zero."""

    var source: String
    """What a `SCAN` reads, as a table name or a path. Empty elsewhere."""

    def __init__(
        out self,
        kind: NodeKind,
        var inputs: List[Int],
        var exprs: List[Int],
        parts: Int,
        var names: List[String],
        var flags: List[Bool],
        op: Int,
        offset: Int,
        length: Int,
        table: Int,
        var source: String,
    ):
        """Builds a node.

        Args:
            kind: Which of the nine.
            inputs: The nodes this one reads.
            exprs: The expressions.
            parts: Where the first expression list ends.
            names: The output or column names.
            flags: The sort directions, or the union's duplicate flag.
            op: The join kind code.
            offset: The rows a limit skips.
            length: The rows a limit keeps.
            table: Which relation a scan is.
            source: What a scan reads.
        """
        self.kind = kind
        self.inputs = inputs^
        self.exprs = exprs^
        self.parts = parts
        self.names = names^
        self.flags = flags^
        self.op = op
        self.offset = offset
        self.length = length
        self.table = table
        self.source = source^


struct Plan(Movable, Sized):
    """A logical plan: the nodes, and the expression arena they index into.

    The two are one value because a node holds expressions by index and an index
    only means something against an arena. Keeping them apart would make every
    caller responsible for keeping the pair together, and a pass that rewrote one
    without the other would produce a plan that read the wrong subtree rather
    than one that failed.
    """

    var nodes: List[PlanNode]
    """Every node built so far, in creation order, so an input always sits below
    the node that reads it."""

    var exprs: Expressions
    """The expressions the nodes index into."""

    def __init__(out self):
        """Builds an empty plan."""
        self.nodes = List[PlanNode]()
        self.exprs = Expressions()

    def __len__(self) -> Int:
        """Counts the nodes.

        Returns:
            How many nodes have been built.
        """
        return len(self.nodes)

    def _add(mut self, var node: PlanNode) -> Int:
        """Appends a node and hands back its index.

        Args:
            node: The node.

        Returns:
            Its index.
        """
        var at = len(self.nodes)
        self.nodes.append(node^)
        return at

    def _input(self, at: Int) raises:
        """Refuses an input that is not in the plan.

        Args:
            at: The index.

        Raises:
            If the index names no node.
        """
        if at < 0 or at >= len(self.nodes):
            raise Error(
                String(
                    "plan node ",
                    at,
                    " is not in a plan of ",
                    len(self.nodes),
                )
            )

    def _rowwise(self, expr: Int, var role: String) raises:
        """Refuses an expression that cannot be evaluated a row at a time.

        A predicate, a group key, a join key and a sort key are all read once
        per row, so an aggregate in one of them is not an expression that needs
        a different kernel, it is a query the caller has not written. Saying so
        here names the node and the position. Letting it through says nothing
        useful several layers down.

        Args:
            expr: The expression.
            role: What the expression is being used as, for the message.

        Raises:
            If the expression is not in the arena or is not elementwise.
        """
        if not self.exprs.elementwise(expr):
            raise Error(
                String(
                    role,
                    (
                        " has to be read a row at a time, and this one folds"
                        " rows together"
                    ),
                )
            )

    def scan(
        mut self, var source: String, var columns: List[String], table: Int
    ) -> Int:
        """Builds a scan.

        Uses `source`, `names` for the columns read, and `table`. No inputs and
        no expressions until pushdown puts a predicate on one, which is a later
        change to this node rather than a different node.

        Args:
            source: The table name or path.
            columns: The columns read.
            table: Which relation this is.

        Returns:
            The index of the new node.
        """
        return self._add(
            PlanNode(
                NodeKind.SCAN,
                List[Int](),
                List[Int](),
                0,
                columns^,
                List[Bool](),
                0,
                0,
                0,
                table,
                source^,
            )
        )

    def filter(mut self, input: Int, predicate: Int) raises -> Int:
        """Builds a filter.

        One input and one expression, the predicate.

        Args:
            input: The node filtered.
            predicate: The predicate.

        Returns:
            The index of the new node.

        Raises:
            If the input is not in the plan, or the predicate is not in the
            arena or is not elementwise.
        """
        self._input(input)
        self._rowwise(predicate, "a filter predicate")
        return self._add(
            PlanNode(
                NodeKind.FILTER,
                [input],
                [predicate],
                0,
                List[String](),
                List[Bool](),
                0,
                0,
                0,
                UNBOUND,
                String(),
            )
        )

    def project(
        mut self, input: Int, var outputs: List[Int], var names: List[String]
    ) raises -> Int:
        """Builds a projection.

        One input, one expression per output column, and a name for each.

        Args:
            input: The node projected.
            outputs: The output expressions.
            names: What each one is called.

        Returns:
            The index of the new node.

        Raises:
            If the input is not in the plan, an output is not in the arena, or
            the two lists are different lengths.
        """
        self._input(input)
        if len(outputs) != len(names):
            raise Error(
                String(
                    "a projection has ",
                    len(outputs),
                    " outputs and ",
                    len(names),
                    " names for them",
                )
            )
        for i in range(len(outputs)):
            self.exprs.check(outputs[i])
        return self._add(
            PlanNode(
                NodeKind.PROJECT,
                [input],
                outputs^,
                0,
                names^,
                List[Bool](),
                0,
                0,
                0,
                UNBOUND,
                String(),
            )
        )

    def aggregate(
        mut self,
        input: Int,
        var keys: List[Int],
        var aggs: List[Int],
        var names: List[String],
    ) raises -> Int:
        """Builds an aggregation.

        One input. The expressions are the group keys and then the aggregates,
        with `parts` at the key count, and the names cover both in that order,
        which is the order the output columns come out in.

        An empty key list is a whole frame reduction. It is the same node
        because it means the same thing, and the difference between hashing
        every row to find its group and knowing there is one group is a physical
        choice rather than a logical one.

        Args:
            input: The node aggregated.
            keys: The group keys.
            aggs: The aggregates.
            names: What the key and aggregate columns are called.

        Returns:
            The index of the new node.

        Raises:
            If the input is not in the plan, an expression is not in the arena,
            a group key is not elementwise, or the names do not cover the
            output.
        """
        self._input(input)
        if len(names) != len(keys) + len(aggs):
            raise Error(
                String(
                    "an aggregation produces ",
                    len(keys) + len(aggs),
                    " columns and has ",
                    len(names),
                    " names for them",
                )
            )
        for i in range(len(keys)):
            self._rowwise(keys[i], "a group key")
        for i in range(len(aggs)):
            self.exprs.check(aggs[i])
        var parts = len(keys)
        var exprs = keys^
        for i in range(len(aggs)):
            exprs.append(aggs[i])
        return self._add(
            PlanNode(
                NodeKind.AGGREGATE,
                [input],
                exprs^,
                parts,
                names^,
                List[Bool](),
                0,
                0,
                0,
                UNBOUND,
                String(),
            )
        )

    def join(
        mut self,
        left: Int,
        right: Int,
        var left_keys: List[Int],
        var right_keys: List[Int],
        kind: JoinKind,
    ) raises -> Int:
        """Builds a join.

        Two inputs, left then right. The expressions are the left keys and then
        the right keys, with `parts` at the left key count, so the pair at
        position `i` is `exprs[i]` against `exprs[parts + i]`.

        Args:
            left: The left input.
            right: The right input.
            left_keys: The keys on the left.
            right_keys: The keys on the right.
            kind: Which rows to keep.

        Returns:
            The index of the new node.

        Raises:
            If either input is not in the plan, a key is not in the arena or is
            not elementwise, or the two key lists are different lengths.
        """
        self._input(left)
        self._input(right)
        if len(left_keys) != len(right_keys):
            raise Error(
                String(
                    "a join has ",
                    len(left_keys),
                    " keys on the left and ",
                    len(right_keys),
                    " on the right",
                )
            )
        for i in range(len(left_keys)):
            self._rowwise(left_keys[i], "a join key")
        for i in range(len(right_keys)):
            self._rowwise(right_keys[i], "a join key")
        var parts = len(left_keys)
        var exprs = left_keys^
        for i in range(len(right_keys)):
            exprs.append(right_keys[i])
        return self._add(
            PlanNode(
                NodeKind.JOIN,
                [left, right],
                exprs^,
                parts,
                List[String](),
                List[Bool](),
                Int(kind.code),
                0,
                0,
                UNBOUND,
                String(),
            )
        )

    def sort(
        mut self,
        input: Int,
        var keys: List[Int],
        var descending: List[Bool],
        var nulls_last: List[Bool],
    ) raises -> Int:
        """Builds a sort.

        One input, one expression per key, and two flags per key held as one
        list with the descending flags first.

        Args:
            input: The node sorted.
            keys: The sort keys, most significant first.
            descending: Whether each key sorts downwards.
            nulls_last: Whether the missing values of each key go at the end.

        Returns:
            The index of the new node.

        Raises:
            If the input is not in the plan, a key is not in the arena or is not
            elementwise, or either flag list is a different length from the
            keys.
        """
        self._input(input)
        if len(descending) != len(keys) or len(nulls_last) != len(keys):
            raise Error(
                String(
                    "a sort on ",
                    len(keys),
                    " keys has ",
                    len(descending),
                    " directions and ",
                    len(nulls_last),
                    " null placements",
                )
            )
        for i in range(len(keys)):
            self._rowwise(keys[i], "a sort key")
        var flags = descending^
        for i in range(len(nulls_last)):
            flags.append(nulls_last[i])
        return self._add(
            PlanNode(
                NodeKind.SORT,
                [input],
                keys^,
                0,
                List[String](),
                flags^,
                0,
                0,
                0,
                UNBOUND,
                String(),
            )
        )

    def limit(mut self, input: Int, offset: Int, length: Int) raises -> Int:
        """Builds a limit.

        One input and no expressions.

        Args:
            input: The node limited.
            offset: How many rows to skip.
            length: How many to keep, or `NO_LIMIT` to keep the rest.

        Returns:
            The index of the new node.

        Raises:
            If the input is not in the plan, or either number is negative and
            is not `NO_LIMIT`.
        """
        self._input(input)
        if offset < 0:
            raise Error(String("a limit cannot skip ", offset, " rows"))
        if length < 0 and length != NO_LIMIT:
            raise Error(String("a limit cannot keep ", length, " rows"))
        return self._add(
            PlanNode(
                NodeKind.LIMIT,
                [input],
                List[Int](),
                0,
                List[String](),
                List[Bool](),
                0,
                offset,
                length,
                UNBOUND,
                String(),
            )
        )

    def distinct(mut self, input: Int, var keys: List[Int]) raises -> Int:
        """Builds a distinct.

        One input. An empty key list means the whole row.

        Args:
            input: The node deduplicated.
            keys: The columns that decide, or empty for all of them.

        Returns:
            The index of the new node.

        Raises:
            If the input is not in the plan, or a key is not in the arena or is
            not elementwise.
        """
        self._input(input)
        for i in range(len(keys)):
            self._rowwise(keys[i], "a distinct key")
        return self._add(
            PlanNode(
                NodeKind.DISTINCT,
                [input],
                keys^,
                0,
                List[String](),
                List[Bool](),
                0,
                0,
                0,
                UNBOUND,
                String(),
            )
        )

    def union(mut self, var inputs: List[Int], all: Bool) raises -> Int:
        """Builds a union.

        Any number of inputs, in order. `all` true is a concatenation and false
        drops the duplicates, which is the whole difference between the two and
        is why there is one node rather than two.

        Args:
            inputs: The nodes stacked, in order.
            all: Whether duplicate rows survive.

        Returns:
            The index of the new node.

        Raises:
            If an input is not in the plan, or there are none.
        """
        if len(inputs) == 0:
            raise Error("a union needs something to stack")
        for i in range(len(inputs)):
            self._input(inputs[i])
        return self._add(
            PlanNode(
                NodeKind.UNION,
                inputs^,
                List[Int](),
                0,
                List[String](),
                [all],
                0,
                0,
                0,
                UNBOUND,
                String(),
            )
        )
