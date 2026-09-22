"""A query text in, a frame out.

Every stage of the SQL front end has been testable on its own since it was
written. The tokenizer is checked against token streams, the transform against
SQL printed back, the lowering against the text `explain` produces, and the
pipeline against frames built by hand. None of that asserts that the stages fit
together, and a seam that does not fit is exactly the kind of defect each stage
passing its own tests will not show.

This is the seam, written down once. It parses, lowers, binds, optimizes, finds
the frames the scans named, lowers the plan to a pipeline and runs it. There is
no decision in here that another stage has not already made, which is the point:
if this file ever grows a rule about what a query means then a stage above it
has left something unfinished.

### It is not the front door

`sql()`, `df.sql()` and `query()` are S7, and they own the things a front door
owns: the prepared statement cache, parameters, the capability flag, and the
latency budget that makes a cache hit cost two microseconds. This does none of
that.

What it does own is `Dialect`, which holds the grammar, the jump table built
from it and the function catalog. Those three are read out of the generated
tables and never written to, and building them is about two milliseconds, which
is a lot next to a statement that parses in a fifth of one. `run` below builds
one per call and throws it away, which is the right cost for a function whose
job is to be obviously correct and the wrong cost for anything running a second
statement. Anything running a second statement holds a `Dialect`.

### The frames come from the catalog and the plan says which

A scan carries the name it was written with and the relation id the lowering
gave it, and `firepanda.plan.lower` wants one frame per relation id in that
order. So the two are matched through the plan rather than through the order the
caller registered anything, and a query that names one table twice is refused
here for the same reason the plan lowering refuses it.

Reading the scans back off the optimized plan rather than off the one the SQL
lowering produced matters. Predicate pushdown rebuilds the node list and pruning
takes nodes out, so the scan that survives is the one whose relation id the
pipeline will ask for.
"""

from firepanda.frame import DataFrame
from firepanda.plan.bind import bind
from firepanda.plan.lower import lower as lower_plan
from firepanda.plan.node import NodeKind, Plan
from firepanda.plan.optimize import optimize

from .ast import Ast
from .catalog import Catalog
from .plan import lower
from .registry import Registry
from .table import Grammar
from .transform import Transform


def _scans(plan: Plan, root: Int) raises -> List[Int]:
    """Which node is the scan of each relation, walking down from the root.

    Args:
        plan: The plan, after optimization.
        root: The node whose output is the answer.

    Returns:
        One node index per relation id, in relation id order.

    Raises:
        Error: If two scans read the same relation, which would make a frame
            mean two things at once.
    """
    var found = List[Int]()
    var stack = List[Int](capacity=len(plan.nodes))
    stack.append(root)
    while len(stack) > 0:
        var at = stack.pop()
        if plan.nodes[at].kind == NodeKind.SCAN:
            var rel = plan.nodes[at].table
            while len(found) <= rel:
                found.append(-1)
            if found[rel] != -1:
                raise Error(
                    String(
                        "two scans of this query both read relation ",
                        rel,
                        ", and a relation is one frame",
                    )
                )
            found[rel] = at
        for i in range(len(plan.nodes[at].inputs)):
            stack.append(plan.nodes[at].inputs[i])
    return found^


def _frames(plan: Plan, root: Int, catalog: Catalog) raises -> List[DataFrame]:
    """The frame each relation reads, in the order the scans number them.

    Args:
        plan: The plan, after optimization.
        root: The node whose output is the answer.
        catalog: What the scans' names are resolved against.

    Returns:
        One frame per relation id.

    Raises:
        Error: If a scan names something the catalog no longer holds, or if a
            relation has no scan, which would leave the pipeline reading a
            frame nobody named.
    """
    var scans = _scans(plan, root)
    var held = List[DataFrame](capacity=len(scans))
    for rel in range(len(scans)):
        if scans[rel] == -1:
            raise Error(
                String(
                    "relation ",
                    rel,
                    (
                        " has no scan in the optimized plan, so there is no"
                        " table name to find its frame under"
                    ),
                )
            )
        var name = plan.nodes[scans[rel]].source
        var found = catalog.find(name)
        if found < 0:
            raise Error(catalog.missing(name))
        held.append(catalog.frame_at(found).copy())
    return held^


struct Dialect(Movable):
    """The three tables the front end reads and never writes.

    The grammar, the jump table built from it and the function catalog. None of
    the three depends on the statement, the catalog or the data: they are read
    out of the generated tables and then only looked at. Between them they are
    about two milliseconds, and `run` below was paying that on every call.

    That was the right trade while `run` was the seam test and nothing else.
    It stopped being the right trade when the ClickBench SQL route started
    calling it 43 times, because two milliseconds of table reading was showing
    up in a measurement of what a planner costs. Holding one of these is what
    makes the second statement in a process cheaper than the first.

    It is still not the front door. A front door owns the prepared statement
    cache, parameters, the capability flag and a latency budget, and this owns
    none of them. It owns the tables, which is the part that was measurably
    wrong.

    Costs a few hundred kilobytes and is read only once built, so one per
    process is the intended number and one per thread is also fine.
    """

    var grammar: Grammar
    """The dialect: every rule, every node, every literal, every keyword."""

    var rules: Transform
    """The jump table, one action per grammar rule."""

    var registry: Registry
    """The function catalog the lowering checks every call name against."""

    def __init__(out self) raises:
        """Reads the three tables.

        Raises:
            Error: If a table is malformed or the grammar has no rule by a name
                the transform expects, which is a build problem rather than
                anything a caller can do something about.
        """
        self.grammar = Grammar()
        self.rules = Transform(self.grammar)
        self.registry = Registry()

    def run(self, sql: StringSlice, catalog: Catalog) raises -> DataFrame:
        """Runs one `SELECT` against the frames a catalog holds.

        Args:
            sql: The whole statement, with or without a trailing semicolon.
            catalog: The names the query is allowed to say.

        Returns:
            The frame the query produces.

        Raises:
            Error: As the free function below does, and for the same reasons.
        """
        var ast = Ast()
        var statement = self.rules.parse_statement(sql, self.grammar, ast)
        var built = lower(ast, statement, catalog, self.grammar, self.registry)

        # Binding before the optimizer rather than leaving it to the passes, so
        # that a name that resolves against nothing is an error about the query
        # somebody wrote and not about a node a pass built.
        _ = bind(built.plan, built.root, built.sources)
        var root = optimize(built.plan, built.root, built.sources)

        var frames = _frames(built.plan, root, catalog)
        var pipe = lower_plan(built.plan, root, frames^)
        return pipe^.run()


def run(sql: StringSlice, catalog: Catalog) raises -> DataFrame:
    """Runs one `SELECT` against the frames a catalog holds.

    Reads the three tables, runs the statement and throws the tables away. A
    caller with more than one statement to run should hold a `Dialect` and call
    its `run` instead, which is the same code without the two milliseconds.

    Args:
        sql: The whole statement, with or without a trailing semicolon.
        catalog: The names the query is allowed to say.

    Returns:
        The frame the query produces.

    Raises:
        Error: If the text is not one statement firepanda runs, if it is a
            shape the SQL lowering does not lower yet, if it does not bind, or
            if the plan holds a node the pipeline has no operator for. The error
            says which of those it was, because the four have different answers
            and a caller that cannot tell them apart cannot act on any of them.
    """
    return Dialect().run(sql, catalog)
