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
that. It builds a grammar per call, which is the wrong cost for a front door and
the right cost for a function whose job is to be obviously correct.

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


def run(sql: StringSlice, catalog: Catalog) raises -> DataFrame:
    """Runs one `SELECT` against the frames a catalog holds.

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
    var grammar = Grammar()
    var rules = Transform(grammar)
    var ast = Ast()
    var statement = rules.parse_statement(sql, grammar, ast)
    var built = lower(ast, statement, catalog)

    # Binding before the optimizer rather than leaving it to the passes, so that
    # a name that resolves against nothing is an error about the query somebody
    # wrote and not about a node a pass built.
    _ = bind(built.plan, built.root, built.sources)
    var root = optimize(built.plan, built.root, built.sources)

    var frames = _frames(built.plan, root, catalog)
    var pipe = lower_plan(built.plan, root, frames^)
    return pipe^.run()
