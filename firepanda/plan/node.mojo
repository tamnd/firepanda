"""The twelve logical nodes a query is, and the arena they live in.

The same collision as `exec/node.mojo` and the right one, because these are the
same concept at two levels. A node here says what the query wants. A node there
says how one chunk of it is computed. The list here is shorter than the list
there, since several physical nodes are alternative implementations of one
logical node, and it is shorter than it looks: nine of them cover every TPC-H
query and the rest are how the queries that TPC-H does not ask are written down.

Scan, Filter, Project, Aggregate, Join, Sort, Limit, Distinct and Union. That is
the list in `docs/specs/planner/01-what-a-plan-is.md` and the spec says to refuse
to add to it without an argument, so it is written down in `NodeKind` and
nowhere else. Values is the tenth and the argument for it is that `SELECT 1` and
`VALUES (1), (2)` have no node at all otherwise, and the alternative is a
projection over a node that is not there.

Union is the one that carries more than its name says. `EXCEPT` and `INTERSECT`
are the same node with a different code in `op`, because all three line their
inputs up by position, take the first input's names, and have the same question
about duplicates hanging off them. Three kinds for that would be three copies of
every pass that touches one.

Window is the twelfth and the argument for it is that a window is the one thing
a projection cannot hold. Every other expression a project computes reads one
row, and a window reads the partition the row is in, so putting one in a project
would mean every pass that treats a project as elementwise now has to check. The
node keeps that check in one place, and it adds its columns to the ones below it
rather than replacing them, because `SELECT x, sum(x) OVER ()` wants both and a
node that replaced them would need a project above it saying so.

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
    """Which of the twelve kinds a logical node is."""

    var code: Int
    """The kind, as one of the twelve values below."""

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
    """Several inputs combined by position: a union, an intersection or a
    difference. Covers concat as well, since the difference is whether
    duplicates survive and that is a flag rather than a node. Which of the three
    is in `op` and the kind keeps the name the common case has."""

    comptime VALUES = Self(9)
    """Rows written out rather than read from anywhere. The second node with no
    input, and the one that makes a query with no `FROM` a plan rather than a
    special case."""

    comptime TABLE_FUNCTION = Self(10)
    """A function called where a table goes, producing rows out of its
    arguments. The third node with no input. What it is called is in `source`
    and the arguments are the expressions, and like a `VALUES` every one of them
    has to read nothing, because there is nothing under it to read."""

    comptime WINDOW = Self(11)
    """A list of window expressions over one input, added to the columns that
    input already produces. The one node whose output is wider than what it was
    asked for, and the only place a window expression is allowed to sit."""

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
        elif self == Self.VALUES:
            writer.write("VALUES")
        elif self == Self.TABLE_FUNCTION:
            writer.write("TABLE FUNCTION")
        elif self == Self.WINDOW:
            writer.write("WINDOW")
        else:
            writer.write("UNION")


comptime NO_LIMIT = -1
"""What a `LIMIT` holds for its length when it only skips. An offset with no
length is a real thing to write and zero is a real length, so the absence needs a
value of its own."""

comptime SET_UNION = 0
"""Every row of every input, which is the one a `UNION` node had before the other
two arrived. Zero, so a node built before this existed still says union."""

comptime SET_EXCEPT = 1
"""The rows of the first input that the second does not have."""

comptime SET_INTERSECT = 2
"""The rows the first input and the second both have."""


struct PlanNode(Copyable, Movable):
    """One node of a logical plan.

    One struct for all twelve kinds, on the same grounds as `Expr`: the arena
    holds them in one list and a list has one element type. Which fields a kind
    uses is documented on the builder that makes it.
    """

    var kind: NodeKind
    """Which of the twelve this is."""

    var inputs: List[Int]
    """The nodes this one reads, as plan arena indices. Empty on a `SCAN`, on a
    `VALUES` and on a `TABLE_FUNCTION`, one on the seven in the middle, two on a
    `JOIN`, and any number on a `UNION`."""

    var exprs: List[Int]
    """The expressions, as expression arena indices, in the order the builder
    documents. Where a kind has two lists of them, `parts` says where the first
    one ends."""

    var parts: Int
    """How many of `exprs` belong to the first of two lists. The group key count
    on an `AGGREGATE` and the left key count on a `JOIN`. On a `VALUES` it is the
    width instead, since the expressions there are rows rather than two lists.
    Zero elsewhere."""

    var names: List[String]
    """The output names on a `PROJECT`, an `AGGREGATE`, a `VALUES` and a
    `TABLE_FUNCTION`, the names of the columns a `WINDOW` adds, and the column
    names read on a `SCAN`. Empty elsewhere."""

    var flags: List[Bool]
    """The directions on a `SORT`, descending first and nulls last after, each
    as long as the key list. One entry on a `UNION`, true when duplicates
    survive. Empty elsewhere."""

    var op: Int
    """The `JoinKind` code on a `JOIN` and one of the `SET_` codes on a `UNION`.
    Zero elsewhere, which is a real code in both, so read it only after checking
    the kind."""

    var offset: Int
    """The rows a `LIMIT` skips. Zero elsewhere."""

    var length: Int
    """The rows a `LIMIT` keeps, or `NO_LIMIT`. Zero elsewhere."""

    var table: Int
    """Which relation a `SCAN` is, as the bit index a bound column reference
    carries. `UNBOUND` elsewhere, and a single input plan leaves it at zero."""

    var source: String
    """What a `SCAN` reads, as a table name or a path, and what a
    `TABLE_FUNCTION` is called. Empty elsewhere."""

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
            kind: Which of the twelve.
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

    def check(self, at: Int) raises:
        """Refuses an index that names no node in the plan.

        Named the same as `Expressions.check` and public for the same reason:
        a pass over a plan takes indices from somewhere and has to be able to
        say so when one of them is not a node.

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
        self.check(input)
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
        self.check(input)
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

    def window(
        mut self, input: Int, var outputs: List[Int], var names: List[String]
    ) raises -> Int:
        """Builds a window node.

        One input, one window expression per column added, and a name for each.
        The columns the input already produces come out first and these come
        after them, which is the order `SELECT x, sum(x) OVER ()` reads in and
        is why the node does not carry the input's names anywhere.

        Every expression has to be a window. An aggregate here is a query that
        wanted a `GROUP BY`, and anything elementwise is a projection, and both
        of those have a node already. Keeping the rule this tight is what lets
        every other pass go on treating a project as elementwise.

        The frame is not here yet. Every window this builds reads its whole
        partition, which is what `OVER (PARTITION BY ...)` with no frame clause
        means and is the only frame the expression arena can spell.

        Args:
            input: The node the windows are computed over.
            outputs: The window expressions.
            names: What each one is called.

        Returns:
            The index of the new node.

        Raises:
            If the input is not in the plan, an expression is not in the arena
            or is not a window, the two lists are different lengths, or there
            are no windows at all.
        """
        self.check(input)
        if len(outputs) != len(names):
            raise Error(
                String(
                    "a window node computes ",
                    len(outputs),
                    " columns and has ",
                    len(names),
                    " names for them",
                )
            )
        if len(outputs) == 0:
            raise Error(
                "a window node that computes no window is the node below it"
            )
        for i in range(len(outputs)):
            self.exprs.check(outputs[i])
            if self.exprs.nodes[outputs[i]].kind != ExprKind.WINDOW:
                raise Error(
                    String(
                        "column ",
                        i + 1,
                        " of a window node is of kind ",
                        self.exprs.nodes[outputs[i]].kind,
                        (
                            ", and a window node holds windows because"
                            " everything else has a node of its own"
                        ),
                    )
                )
        return self._add(
            PlanNode(
                NodeKind.WINDOW,
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
        self.check(input)
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
        var mark: String = String(),
    ) raises -> Int:
        """Builds a join.

        Two inputs, left then right. The expressions are the left keys and then
        the right keys, with `parts` at the left key count, so the pair at
        position `i` is `exprs[i]` against `exprs[parts + i]`.

        A mark join adds a column to the left side rather than taking rows away
        from it, so it is the one kind that has an output name to settle, and
        that name is `mark`. It goes in `names`, which every other kind leaves
        empty.

        Args:
            left: The left input.
            right: The right input.
            left_keys: The keys on the left.
            right_keys: The keys on the right.
            kind: Which rows to keep.
            mark: What the mark join's boolean column is called. Consumed.
                Required for a mark join and refused for every other kind.

        Returns:
            The index of the new node.

        Raises:
            If either input is not in the plan, a key is not in the arena or is
            not elementwise, the two key lists are different lengths, or the
            mark name is missing on a mark join or given on anything else.
        """
        self.check(left)
        self.check(right)
        if kind == JoinKind.MARK:
            if mark.byte_length() == 0:
                raise Error(
                    "a mark join hands out a boolean column and the column has"
                    " to be called something, so the name is not optional"
                )
        elif mark.byte_length() != 0:
            raise Error(
                String(
                    "a ",
                    kind,
                    (
                        " join was given the name of a mark column, and the"
                        " mark column is the mark join's"
                    ),
                )
            )
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
        var names = List[String]()
        if kind == JoinKind.MARK:
            names.append(mark^)
        return self._add(
            PlanNode(
                NodeKind.JOIN,
                [left, right],
                exprs^,
                parts,
                names^,
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

        The length comes out `NO_LIMIT` and stays there unless `limits` puts a
        bound on it, which is how a top n is written down: a sort that only has
        to get the first n rows right. Nothing is obliged to honour the bound,
        because the limit that produced it is still sitting above the sort and
        still doing the cutting, so an operator that ignores it is slow rather
        than wrong.

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
        self.check(input)
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
                NO_LIMIT,
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
        self.check(input)
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
        self.check(input)
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
        return self.setop(inputs^, SET_UNION, all)

    def setop(
        mut self, var inputs: List[Int], op: Int, all: Bool
    ) raises -> Int:
        """Builds a union, a difference or an intersection.

        One node for the three because they differ in which rows of the inputs
        survive and in nothing else. Every one of them lines its inputs up by
        position, produces the first input's names, and has the same question
        about duplicates hanging off it.

        A union takes any number of inputs, since stacking is associative and a
        chain of them is one node. A difference and an intersection take exactly
        two, because SQL writes them between two queries and a chain is a nest
        of nodes rather than a list. `a EXCEPT b EXCEPT c` and `a EXCEPT (b
        EXCEPT c)` are different answers, so flattening one into a list would
        lose the thing that tells them apart.

        `all` means the same three things it means in SQL. A union keeps every
        row, a difference subtracts one copy of a row for each copy on the right
        rather than every copy, and an intersection keeps as many copies as the
        thinner side has.

        Args:
            inputs: The nodes combined, in order.
            op: One of `SET_UNION`, `SET_EXCEPT` and `SET_INTERSECT`.
            all: Whether duplicate rows survive.

        Returns:
            The index of the new node.

        Raises:
            If an input is not in the plan, if there are none, if a difference
            or an intersection does not have exactly two, or if the operation is
            not one of the three.
        """
        if op != SET_UNION and op != SET_EXCEPT and op != SET_INTERSECT:
            raise Error(String("set operation ", op, " is not one of three"))
        if len(inputs) == 0:
            raise Error("a union needs something to stack")
        if op != SET_UNION and len(inputs) != 2:
            var word = "a difference" if op == SET_EXCEPT else "an intersection"
            raise Error(
                String(
                    word, " is between two inputs, and this has ", len(inputs)
                )
            )
        for i in range(len(inputs)):
            self.check(inputs[i])
        return self._add(
            PlanNode(
                NodeKind.UNION,
                inputs^,
                List[Int](),
                0,
                List[String](),
                [all],
                op,
                0,
                0,
                UNBOUND,
                String(),
            )
        )

    def values(
        mut self, var rows: List[Int], var names: List[String]
    ) raises -> Int:
        """Builds a literal table, written out rather than read from anywhere.

        The rows are one flat list in row order, which is how the expressions
        come off a `VALUES` and is the only layout where adding a row is
        appending. How wide the table is comes from `names`, and `parts` carries
        it so that a pass can find the row boundaries without the schema.

        Every expression has to read nothing at all, which is a stronger rule
        than the one a filter predicate or a sort key gets. There is no input
        here, so a column reference has nothing to resolve against and an
        aggregate has no rows to fold. Saying so with the row and the column in
        the message beats binding against an empty schema and reporting a name
        that is not there.

        Args:
            rows: The expressions, row by row, each row as wide as `names`.
            names: What the columns are called.

        Returns:
            The index of the new node.

        Raises:
            If there are no columns, if the expressions do not divide into whole
            rows, if there are no rows, or if an expression reads anything.
        """
        if len(names) == 0:
            raise Error("a table of no columns is not a table")
        if len(rows) % len(names) != 0:
            raise Error(
                String(
                    "a table ",
                    len(names),
                    " columns wide cannot be made of ",
                    len(rows),
                    " values",
                )
            )
        if len(rows) == 0:
            raise Error("a table of no rows still has to say it has none")
        for i in range(len(rows)):
            if not self.exprs.input_independent(rows[i]):
                raise Error(
                    String(
                        (
                            "a VALUES is written out rather than read from"
                            " anywhere, and column "
                        ),
                        i % len(names),
                        " of row ",
                        i // len(names),
                        " reads something",
                    )
                )
        var parts = len(names)
        return self._add(
            PlanNode(
                NodeKind.VALUES,
                List[Int](),
                rows^,
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

    def table_function(
        mut self,
        var source: String,
        var args: List[Int],
        var names: List[String],
    ) raises -> Int:
        """Builds a call to a function that produces rows.

        Uses `source` for what the function is called, `exprs` for its arguments
        and `names` for the columns it produces. No inputs, the same way a
        `VALUES` has none and for the same reason: the rows come out of the node
        rather than out of something below it.

        Every argument has to read nothing at all, which is the rule a `VALUES`
        gets and is stronger than the one a filter predicate gets. `range(n)`
        where `n` is a column is a lateral call, which is a table function
        joined to the rows it was called for, and that is a different node with
        an input on it rather than this one with a looser rule.

        Which functions exist and what each one produces is binding's business
        rather than this one's. The plan is a shape, and which names an engine
        answers to is a question about the engine.

        Args:
            source: What the function is called.
            args: Its arguments, in the order they were written.
            names: What the columns it produces are called.

        Returns:
            The index of the new node.

        Raises:
            If the function has no name, if it produces no columns, if an
            argument is not in the plan, or if an argument reads something.
        """
        if source.byte_length() == 0:
            raise Error("a table function with no name is not a call")
        if len(names) == 0:
            raise Error("a table of no columns is not a table")
        for i in range(len(args)):
            self.exprs.check(args[i])
            if not self.exprs.input_independent(args[i]):
                raise Error(
                    String(
                        "argument ",
                        i + 1,
                        " of ",
                        source,
                        (
                            " reads something, and a table function is called"
                            " where a table goes, so there is nothing under it"
                            " to read"
                        ),
                    )
                )
        return self._add(
            PlanNode(
                NodeKind.TABLE_FUNCTION,
                List[Int](),
                args^,
                0,
                names^,
                List[Bool](),
                0,
                0,
                0,
                UNBOUND,
                source^,
            )
        )
