"""The expression tree a plan node hangs off, and the three analyses over it.

An expression is a tree and a plan node is a tree, and they are kept apart. A
`Filter` holds one expression, a `Project` holds a list of them, an `Aggregate`
holds group keys and aggregate expressions. None of them holds another plan node
inside an expression, which is what makes both trees easy to walk.

The tree lives in an arena. `Expressions` is a list of nodes and an expression
is an index into it, so a child reference is an `Int` and a whole subtree is one
number. That is how Polars holds its `AExpr` and it is worth copying for the
three reasons it is copied for: a self referencing struct is awkward in a
language with value semantics and no cycles, an arena makes a rewrite that
replaces a node in place a one word write rather than a rebuild of everything
above it, and a subtree can be shared by two parents without being copied, which
is what common subexpression elimination will need when it arrives.

The cost is that an index is not typed. Nothing stops a caller passing the index
of a literal where a column was meant. That is the same trade the exec node
union makes and it is the same answer: the set is closed, the builders below are
the only way in, and each one checks what it can.

## The nine kinds

Column reference, literal, unary, binary, cast, call, aggregate, conditional and
window. The list is from `docs/specs/planner/01-what-a-plan-is.md` and the spec
says to write it down and refuse to add to it without an argument, so it is
written down here and in `ExprKind`.

## The three analyses

Every pass in `docs/specs/planner/02-the-pass-pipeline.md` is written in terms of
these three, which is why they are here in the first file rather than arriving
with the first pass that wants one.

`elementwise` asks whether the value at row `i` depends only on row `i`. It is
what makes a projection free in a streaming engine, because an elementwise
expression can be evaluated on a morsel with nothing carried between morsels.
`a + b * 2` is elementwise, `a.sum()` is not, and a window is not.

`input_independent` asks whether the expression reads the input at all. TPC-H q1
has `date '1998-12-01' - interval '90 days'` in its predicate and an engine
without this analysis computes it six million times, so the payoff is real and
it arrives before there is anything else to be pleased about.

`tables` asks which inputs an expression reads from, as a bitmask. Predicate
pushdown and predicate transfer are both written on it, because a predicate can
only be pushed into a subtree that provides every column it references. It is
the one analysis that needs binding to have happened, and it says so by raising
rather than by returning an empty set, because an empty set means "reads
nothing" and would tell pushdown that a predicate is safe to push anywhere.
"""

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType, TypeKind
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.group import AggKind
from firepanda.kernel.unary import UnaryOp


@fieldwise_init
struct ExprKind(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which of the nine kinds an expression node is.

    Held as a code rather than as a variant for the same reason the exec node
    set is a closed union: a `List` of trait objects is not yet expressible, and
    a closed set is the truth anyway.
    """

    var code: Int
    """The kind, as one of the nine values below."""

    comptime COLUMN = Self(0)
    """A reference to a column of the input, by name before binding and by
    position after it."""

    comptime LITERAL = Self(1)
    """A constant. The only kind that is input independent on its own."""

    comptime UNARY = Self(2)
    """One operand and a `UnaryOp`."""

    comptime BINARY = Self(3)
    """Two operands and a `BinaryOp`. Comparison, arithmetic and the logical
    connectives are all this."""

    comptime CAST = Self(4)
    """One operand and a target type. Explicit in the plan even when it was
    implicit in what the user wrote, because binding is where an implicit cast
    is made visible."""

    comptime CALL = Self(5)
    """A named function over zero or more arguments.

    Wider than it looks, because it is where anything that is not one of the
    other eight goes. The logical connectives are here rather than among the
    binary operations, since `BinaryOp` has arithmetic and comparison and the
    conjunction is a free function in `kernel/arith.mojo` rather than a code, so
    `a AND b` is a call named `and` over two arguments."""

    comptime AGGREGATE = Self(6)
    """A fold over many rows into one. Never elementwise and never input
    independent, whatever it is folding."""

    comptime CONDITIONAL = Self(7)
    """A predicate, a value when it holds and a value when it does not."""

    comptime WINDOW = Self(8)
    """An aggregate evaluated over a frame of rows around each row, with
    partition keys and order keys."""

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
        if self == Self.COLUMN:
            writer.write("column")
        elif self == Self.LITERAL:
            writer.write("literal")
        elif self == Self.UNARY:
            writer.write("unary")
        elif self == Self.BINARY:
            writer.write("binary")
        elif self == Self.CAST:
            writer.write("cast")
        elif self == Self.CALL:
            writer.write("call")
        elif self == Self.AGGREGATE:
            writer.write("aggregate")
        elif self == Self.CONDITIONAL:
            writer.write("conditional")
        else:
            writer.write("window")


comptime UNBOUND = -1
"""What a column reference holds for its position and its table until binding
has run. Binding replaces both, and `tables` refuses to answer while either is
still this, because a wrong answer there is a pushdown that moves a predicate
past the only node that could provide its columns."""


struct Expr(Copyable, Movable):
    """One node of an expression tree.

    One struct for all nine kinds rather than nine structs, because the arena
    holds them in one list and a list has one element type. The fields a kind
    does not use are left at the defaults the builders set, and which fields a
    kind uses is documented on each builder rather than left to be discovered.
    """

    var kind: ExprKind
    """Which of the nine this is."""

    var type: LogicalType
    """What the node produces. `LogicalType.NULL` until binding has computed it,
    which is what makes a type error a plan error rather than a kernel error."""

    var name: String
    """The column name on a `COLUMN` before binding, the function name on a
    `CALL`, and empty elsewhere. It is kept on a bound column as well as the
    position, because an explain output that prints positions is not something a
    person can read."""

    var at: Int
    """The column position on a bound `COLUMN`, and `UNBOUND` otherwise."""

    var table: Int
    """Which input relation a bound `COLUMN` reads from, as a bit index rather
    than a mask, and `UNBOUND` otherwise. A single input plan leaves every
    column on table zero and nothing notices."""

    var value: Value
    """The constant on a `LITERAL`. Absent elsewhere."""

    var op: Int
    """The `BinaryOp`, `UnaryOp` or `AggKind` code, on the three kinds that have
    one, and on a `WINDOW`, which carries the aggregate it is evaluating. Zero
    elsewhere, which is a real code on every one of them, so read it only after
    checking the kind."""

    var rowwise: Bool
    """Whether a `CALL` produces its row from its own input row. True on every
    other kind that is elementwise and False on the two that are not.

    A call is the one kind whose answer is a property of the function rather
    than of the tree, and there is no function registry to ask yet, so the
    builder takes it and records it here. When the registry arrives this field
    goes away and the analysis asks the registry, which is the right place for
    it. Until then a caller that gets it wrong gets a cumulative sum evaluated
    per morsel, so the builder has no default and makes the caller say."""

    var parts: Int
    """How many of the children after the first are partition keys, on a
    `WINDOW`. The rest are the order keys. Zero on every other kind."""

    var children: List[Int]
    """The operands, as arena indices, in the order the kind documents."""

    def __init__(
        out self,
        kind: ExprKind,
        var name: String,
        at: Int,
        table: Int,
        var value: Value,
        op: Int,
        rowwise: Bool,
        parts: Int,
        var children: List[Int],
    ):
        """Builds a node with no type on it yet.

        Args:
            kind: Which of the nine.
            name: The column or function name, or empty.
            at: The bound position, or `UNBOUND`.
            table: The bound table, or `UNBOUND`.
            value: The constant, on a literal.
            op: The operation code, on a kind that has one.
            rowwise: Whether a call is elementwise.
            parts: The partition key count, on a window.
            children: The operands, as arena indices.
        """
        self.kind = kind
        self.type = LogicalType.NULL
        self.name = name^
        self.at = at
        self.table = table
        self.value = value^
        self.op = op
        self.rowwise = rowwise
        self.parts = parts
        self.children = children^


struct Expressions(Movable, Sized):
    """The arena the expression nodes live in.

    An expression is an index into `nodes`. Indices are handed out in creation
    order, so a child is always at a lower index than its parent and a walk from
    the end of the list to the beginning visits every node after its children.
    Several passes want that order and get it without a traversal.
    """

    var nodes: List[Expr]
    """Every node built so far. Never shrinks, because an index handed out
    earlier has to keep meaning what it meant."""

    def __init__(out self):
        """Builds an empty arena."""
        self.nodes = List[Expr]()

    def __len__(self) -> Int:
        """Counts the nodes.

        Returns:
            How many nodes have been built.
        """
        return len(self.nodes)

    def _add(mut self, var node: Expr) -> Int:
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
        """Refuses an index that is not in the arena.

        Args:
            at: The index.

        Raises:
            If the index names no node.
        """
        if at < 0 or at >= len(self.nodes):
            raise Error(
                String(
                    "expression ",
                    at,
                    " is not in an arena of ",
                    len(self.nodes),
                )
            )

    def column(mut self, var name: String) -> Int:
        """Builds an unbound reference to a column by name.

        Uses `name`. Leaves `at` and `table` at `UNBOUND` for binding to fill
        in, and the type with them.

        Args:
            name: The column name.

        Returns:
            The index of the new node.
        """
        return self._add(
            Expr(
                ExprKind.COLUMN,
                name^,
                UNBOUND,
                UNBOUND,
                Value(null=LogicalType.NULL),
                0,
                True,
                0,
                List[Int](),
            )
        )

    def literal(mut self, var value: Value) -> Int:
        """Builds a constant.

        Uses `value`, and takes its type from the value, since a constant is the
        one kind that knows what it produces before binding runs.

        Args:
            value: The constant.

        Returns:
            The index of the new node.
        """
        var type = value.type
        var at = self._add(
            Expr(
                ExprKind.LITERAL,
                String(),
                UNBOUND,
                UNBOUND,
                value^,
                0,
                True,
                0,
                List[Int](),
            )
        )
        self.nodes[at].type = type
        return at

    def unary(mut self, op: UnaryOp, over: Int) raises -> Int:
        """Builds a unary operation.

        One child, the operand.

        Args:
            op: The operation.
            over: The operand.

        Returns:
            The index of the new node.

        Raises:
            If the operand is not in the arena.
        """
        self.check(over)
        return self._add(
            Expr(
                ExprKind.UNARY,
                String(),
                UNBOUND,
                UNBOUND,
                Value(null=LogicalType.NULL),
                Int(op.code),
                True,
                0,
                [over],
            )
        )

    def binary(mut self, op: BinaryOp, left: Int, right: Int) raises -> Int:
        """Builds a binary operation.

        Two children, the left operand then the right.

        Args:
            op: The operation.
            left: The left operand.
            right: The right operand.

        Returns:
            The index of the new node.

        Raises:
            If either operand is not in the arena.
        """
        self.check(left)
        self.check(right)
        return self._add(
            Expr(
                ExprKind.BINARY,
                String(),
                UNBOUND,
                UNBOUND,
                Value(null=LogicalType.NULL),
                Int(op.code),
                True,
                0,
                [left, right],
            )
        )

    def cast(mut self, to: LogicalType, over: Int) raises -> Int:
        """Builds a cast.

        One child, the operand. The target type is known here rather than
        computed at binding, so it is written straight onto the node.

        Args:
            to: The target type.
            over: The operand.

        Returns:
            The index of the new node.

        Raises:
            If the operand is not in the arena.
        """
        self.check(over)
        var at = self._add(
            Expr(
                ExprKind.CAST,
                String(),
                UNBOUND,
                UNBOUND,
                Value(null=LogicalType.NULL),
                0,
                True,
                0,
                [over],
            )
        )
        self.nodes[at].type = to
        return at

    def call(
        mut self, var name: String, var args: List[Int], rowwise: Bool
    ) raises -> Int:
        """Builds a call to a named function.

        Uses `name` and `rowwise`. The children are the arguments in order.

        Args:
            name: The function name.
            args: The arguments.
            rowwise: Whether the function produces each output row from the
                matching input row alone. See `Expr.rowwise` for why the caller
                is the one that has to know.

        Returns:
            The index of the new node.

        Raises:
            If any argument is not in the arena.
        """
        for i in range(len(args)):
            self.check(args[i])
        return self._add(
            Expr(
                ExprKind.CALL,
                name^,
                UNBOUND,
                UNBOUND,
                Value(null=LogicalType.NULL),
                0,
                rowwise,
                0,
                args^,
            )
        )

    def aggregate(mut self, op: AggKind, over: Int) raises -> Int:
        """Builds an aggregate.

        One child, the expression being folded.

        Args:
            op: Which fold.
            over: What is being folded.

        Returns:
            The index of the new node.

        Raises:
            If the operand is not in the arena.
        """
        self.check(over)
        return self._add(
            Expr(
                ExprKind.AGGREGATE,
                String(),
                UNBOUND,
                UNBOUND,
                Value(null=LogicalType.NULL),
                Int(op.code),
                True,
                0,
                [over],
            )
        )

    def conditional(
        mut self, when: Int, then: Int, otherwise: Int
    ) raises -> Int:
        """Builds a conditional.

        Three children, the predicate, the value when it holds and the value
        when it does not.

        Args:
            when: The predicate.
            then: The value when the predicate holds.
            otherwise: The value when it does not.

        Returns:
            The index of the new node.

        Raises:
            If any of the three is not in the arena.
        """
        self.check(when)
        self.check(then)
        self.check(otherwise)
        return self._add(
            Expr(
                ExprKind.CONDITIONAL,
                String(),
                UNBOUND,
                UNBOUND,
                Value(null=LogicalType.NULL),
                0,
                True,
                0,
                [when, then, otherwise],
            )
        )

    def window(
        mut self,
        op: AggKind,
        over: Int,
        var partition: List[Int],
        var order: List[Int],
    ) raises -> Int:
        """Builds a window aggregate.

        The children are the expression being aggregated, then the partition
        keys, then the order keys, and `parts` records where the first list ends
        so that the second can begin.

        Args:
            op: Which aggregate.
            over: What is being aggregated.
            partition: The partition keys.
            order: The order keys.

        Returns:
            The index of the new node.

        Raises:
            If any of them is not in the arena.
        """
        self.check(over)
        var children: List[Int] = [over]
        for i in range(len(partition)):
            self.check(partition[i])
            children.append(partition[i])
        for i in range(len(order)):
            self.check(order[i])
            children.append(order[i])
        return self._add(
            Expr(
                ExprKind.WINDOW,
                String(),
                UNBOUND,
                UNBOUND,
                Value(null=LogicalType.NULL),
                Int(op.code),
                False,
                len(partition),
                children^,
            )
        )

    def elementwise(self, root: Int) raises -> Bool:
        """Whether the value at each row depends only on that row.

        An elementwise expression can be evaluated on one morsel with nothing
        carried between morsels, which is what makes a projection free in a
        streaming engine and what a fusion pass checks before it folds two
        operators into one loop.

        An aggregate is not elementwise and a window is not, whatever they are
        aggregating over, and everything else is elementwise exactly when all of
        its children are. A call is elementwise when the function is, which the
        node was told at build time.

        Args:
            root: The expression.

        Returns:
            True if it is elementwise.

        Raises:
            If the expression is not in the arena.
        """
        self.check(root)
        ref node = self.nodes[root]
        if node.kind == ExprKind.AGGREGATE or node.kind == ExprKind.WINDOW:
            return False
        if node.kind == ExprKind.CALL and not node.rowwise:
            return False
        for i in range(len(node.children)):
            if not self.elementwise(node.children[i]):
                return False
        return True

    def input_independent(self, root: Int) raises -> Bool:
        """Whether the expression can be evaluated without reading a row.

        A literal is, arithmetic over literals is, and a column reference is
        not. An expression that is gets computed once at plan time and replaced
        with its answer, which is what stops q1 computing the same date
        subtraction six million times.

        An aggregate over a constant is not input independent, and neither is a
        window or a call that is not elementwise, because all three read the
        height of the input even when they read none of its values. `sum(1)` is
        the row count.

        Args:
            root: The expression.

        Returns:
            True if it reads nothing.

        Raises:
            If the expression is not in the arena.
        """
        self.check(root)
        ref node = self.nodes[root]
        if node.kind == ExprKind.COLUMN:
            return False
        if node.kind == ExprKind.AGGREGATE or node.kind == ExprKind.WINDOW:
            return False
        if node.kind == ExprKind.CALL and not node.rowwise:
            return False
        for i in range(len(node.children)):
            if not self.input_independent(node.children[i]):
                return False
        return True

    def tables(self, root: Int) raises -> UInt64:
        """Which inputs the expression reads, as a bitmask.

        Bit `t` is set when the expression reads a column bound to table `t`. A
        predicate can be pushed into a subtree only when that subtree provides
        every table in this set, which is the whole of what pushdown and
        predicate transfer need to know about an expression.

        Refuses an unbound column rather than treating it as reading nothing.
        The empty set is a real answer here, since an input independent
        expression has one, and it is the answer that tells pushdown a predicate
        is safe to move anywhere. Handing that back for a column whose table is
        simply not known yet would move a predicate past the only node that
        could evaluate it.

        Args:
            root: The expression.

        Returns:
            The mask.

        Raises:
            If the expression is not in the arena, or holds a column that
            binding has not reached, or one bound to a table too far out to fit
            in the mask.
        """
        self.check(root)
        ref node = self.nodes[root]
        if node.kind == ExprKind.COLUMN:
            if node.table == UNBOUND:
                raise Error(
                    String(
                        "column '",
                        node.name,
                        "' has no table until binding has run",
                    )
                )
            if node.table < 0 or node.table >= 64:
                raise Error(
                    String(
                        "column '",
                        node.name,
                        "' is bound to table ",
                        node.table,
                        ", which does not fit in the mask",
                    )
                )
            return UInt64(1) << UInt64(node.table)
        var seen = UInt64(0)
        for i in range(len(node.children)):
            seen |= self.tables(node.children[i])
        return seen

    def positions(self, root: Int) raises -> List[Int]:
        """Which columns of its input the expression reads, as positions.

        The question projection pushdown asks of every expression in a plan,
        and the reason it is here beside `tables` rather than in the pass is
        that it is the same analysis one level finer. `tables` says which
        relations an expression needs and this says which of their columns, so
        a pass that has both can decide what a subtree has to produce.

        The answer comes back sorted and without repeats, because it is a set
        and a caller that has to sort it is a caller that will forget to.

        Args:
            root: The expression.

        Returns:
            The positions, ascending, each once.

        Raises:
            If the expression is not in the arena, or holds a column that
            binding has not reached.
        """
        var found = List[Int]()
        self._positions(root, found)
        return found^

    def _positions(self, root: Int, mut found: List[Int]) raises:
        """Adds every column position under one expression to a sorted set.

        Inserted in order rather than appended and sorted afterwards, because
        an expression reads a handful of columns and a linear insertion into a
        handful is cheaper than reaching for a sort.

        Args:
            root: The expression.
            found: The set, ascending and without repeats, added to.

        Raises:
            If the expression is not in the arena, or holds a column that
            binding has not reached.
        """
        self.check(root)
        ref node = self.nodes[root]
        if node.kind == ExprKind.COLUMN:
            if node.at == UNBOUND:
                raise Error(
                    String(
                        "column '",
                        node.name,
                        "' has no position until binding has run",
                    )
                )
            var to = 0
            while to < len(found) and found[to] < node.at:
                to += 1
            if to == len(found):
                found.append(node.at)
            elif found[to] != node.at:
                found.insert(to, node.at)
            return
        for i in range(len(node.children)):
            self._positions(node.children[i], found)

    def graft(
        mut self, root: Int, names: List[String], onto: List[Int]
    ) raises -> Int:
        """Returns the expression with named columns replaced by expressions.

        What projection merging needs and what any pass that folds one node's
        outputs into the node above it will need. A project over a project reads
        the lower one's outputs by name, so collapsing the two into one means
        putting the lower one's expression where the name was.

        Nothing is rewritten in place, because an expression index may be read
        by more than one node and an arena that let a caller change one out from
        under another would be a different data structure. Only the nodes on the
        path from the root to a replaced column are copied, so an expression
        with nothing to replace in it comes back as the index that went in.

        Args:
            root: The expression.
            names: The column names to replace.
            onto: The expression to put in place of each, in the same order.

        Returns:
            The new expression, or `root` when nothing matched.

        Raises:
            If an expression is not in the arena, or the two lists are different
            lengths.
        """
        self.check(root)
        if len(names) != len(onto):
            raise Error(
                String(
                    "a graft has ",
                    len(names),
                    " names and ",
                    len(onto),
                    " expressions to put in their place",
                )
            )
        for i in range(len(onto)):
            self.check(onto[i])
        return self._graft(root, names, onto)

    def _graft(
        mut self, root: Int, names: List[String], onto: List[Int]
    ) raises -> Int:
        """Copies one node if anything under it was replaced, and not if not.

        Args:
            root: The expression.
            names: The column names to replace.
            onto: The expression to put in place of each.

        Returns:
            The new expression, or `root` when nothing under it matched.

        Raises:
            If an expression is not in the arena.
        """
        var kind = self.nodes[root].kind
        if kind == ExprKind.COLUMN:
            for i in range(len(names)):
                if names[i] == self.nodes[root].name:
                    return onto[i]
            return root

        var kids = self.nodes[root].children.copy()
        var grown = List[Int]()
        var same = True
        for i in range(len(kids)):
            var at = self._graft(kids[i], names, onto)
            if at != kids[i]:
                same = False
            grown.append(at)
        if same:
            return root
        return self.rebuild(root, grown^)

    def rebuild(mut self, root: Int, var kids: List[Int]) raises -> Int:
        """Returns the same node over different operands.

        Nothing is rewritten in place, for the reason `graft` gives: an index
        may be read by more than one node and an arena that let a caller change
        one out from under another would be a different data structure. So this
        adds a node rather than editing one, and a pass that rebuilds a whole
        expression only pays for the path it changed.

        The type is not carried over. A node over new operands produces whatever
        the new operands produce, and the only honest answer here is to let
        binding say so, which is what every caller does next anyway.

        Args:
            root: The node to copy.
            kids: The operands the copy gets. Consumed.

        Returns:
            The new expression.

        Raises:
            If the node is not in the arena.
        """
        self.check(root)
        var name = self.nodes[root].name.copy()
        var value = self.nodes[root].value.copy()
        return self._add(
            Expr(
                self.nodes[root].kind,
                name^,
                self.nodes[root].at,
                self.nodes[root].table,
                value^,
                self.nodes[root].op,
                self.nodes[root].rowwise,
                self.nodes[root].parts,
                kids^,
            )
        )

    def names(self, root: Int) raises -> List[String]:
        """Which columns the expression reads, by name.

        The third of the three and the one predicate pushdown asks. Moving a
        predicate down the plan means asking whether the subtree below can
        answer it, and a position cannot be asked that question, because a
        position only means something against one schema and the whole point of
        moving the predicate is that it will be read against a different one.
        A name survives the move, so the question becomes whether the schema
        below has a column of each name, which is a question a schema can
        answer.

        Unbound is fine here, and that is the difference from the other two. A
        name is what the caller wrote and it is on the expression from the
        moment it is built, so this is the one analysis of the three that is
        worth asking before binding has run.

        Args:
            root: The expression.

        Returns:
            The names, in the order they were first met, each once.

        Raises:
            If the expression is not in the arena.
        """
        var found = List[String]()
        self._names(root, found)
        return found^

    def _names(self, root: Int, mut found: List[String]) raises:
        """Adds every column name under one expression to a set.

        Kept in the order they were met rather than sorted, because there is no
        order on names that means anything to a reader and the order a predicate
        mentions its columns in is the order the person who wrote it chose.

        Args:
            root: The expression.
            found: The set, added to, without repeats.

        Raises:
            If the expression is not in the arena.
        """
        self.check(root)
        ref node = self.nodes[root]
        if node.kind == ExprKind.COLUMN:
            for i in range(len(found)):
                if found[i] == node.name:
                    return
            found.append(node.name)
            return
        for i in range(len(node.children)):
            self._names(node.children[i], found)
