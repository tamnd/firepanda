"""Projection pushdown: a column nothing reads is never read.

The third pass in `docs/specs/planner/02-the-pass-pipeline.md` and the one the
spec calls the single largest by a wide margin. In a column store the difference
between reading three columns and reading forty is most of the query, and the
first version of the TPC-H driver in `firepanda-bench` filtered all sixteen
columns of `lineitem` to answer questions that read two. Naming the needed
columns by hand is most of the distance from 6.671 seconds to about two across
the twenty two queries. This is that done by the planner instead of by the
person writing the query.

## What it works out

One set per node, of the output positions anything above it reads. The root
needs all of its own columns because they are the answer. Every other node needs
what the node above asked of it, plus whatever its own expressions read, and
those two are not the same question: a filter's predicate reads a column that
nothing above the filter wants, and that column still has to arrive.

The walk is one pass from the root downwards. Node indices are handed out in
creation order, so an input always sits at a lower index than the node reading
it, and counting down reaches every node after everything that demands anything
of it. No recursion and no visited set, the same shape `bind` already uses.

## What it rewrites

Three of the nine kinds carry a list of columns and the other six do not.

A scan gets its column list narrowed, which is the point of the pass and where
the reading stops happening. A scan with no list means the whole table, so a
scan that has been through here comes out with a list even when nothing was
dropped from it.

A project and an aggregate get their output lists narrowed, which matters
because a node that still computes a column has to read what that column is
made of. An unused output of a project is an expression evaluated once per row
for nobody. An aggregate keeps every group key whatever anything above thinks,
because dropping a key is not a projection, it is a different query with
different rows in it.

A filter, a sort, a limit, a distinct, a join and a union hand their input's
columns through unchanged, so a position above one of them is the same position
below it and there is nothing on the node itself to narrow.

Two of the nine ask for more than anything above them wants. A distinct with no
keys compares whole rows, and so does a union that drops duplicates, so a column
nothing above reads is still a column that decides whether two rows are one. Both
of them demand every column of their input whatever the node above asked for,
because narrowing either one deletes rows rather than reads. A difference and an
intersection are the same node as the union and they compare whole rows whether
duplicates survive or not, since being on both sides is what they are asking.

## Why it rebinds

A position only means something against a schema, and this pass changes the
schemas. Rather than remap every bound position it hands the plan back to
`bind`, which resolves by name and therefore gets the new positions right for
free. Binding a plan twice at plan time costs nothing next to reading one column
that nothing wanted.

That is also why a node whose output names are not all distinct is left alone.
Binding resolves a name to the first column that has it, so dropping the first
of two columns called the same thing would silently point an expression above at
the other one. A plan like that is ambiguous before this pass touches it, and the
pass declining to make it worse is cheaper than the pass having an opinion about
which of the two was meant.
"""

from firepanda.dtype.schema import Schema
from firepanda.plan.bind import Bound, bind, bind_all
from firepanda.plan.node import (
    SET_UNION,
    NodeKind,
    Plan,
)


def prune(mut plan: Plan, root: Int, sources: List[Schema]) raises -> Schema:
    """Narrows every scan, project and aggregate to the columns above them.

    Args:
        plan: The plan, rewritten in place.
        root: The node whose output is the answer.
        sources: The schema of each relation, indexed by the id a scan carries.

    Returns:
        The schema of the root, which is what it was before.

    Raises:
        Whatever binding raises, since this binds the plan before it rewrites
        anything and again afterwards.
    """
    var bound = bind_all(plan, root, sources)

    var need = List[List[Int]]()
    for _ in range(root + 1):
        need.append(List[Int]())
    for i in range(len(bound[root].schema)):
        need[root].append(i)

    # Downwards from the root. A node not reached is one an earlier rewrite
    # detached, and leaving it alone is the same thing `bind` does with it.
    var wanted = List[Bool](length=root + 1, fill=False)
    wanted[root] = True
    for at in range(root, -1, -1):
        if not wanted[at]:
            continue
        for i in range(len(plan.nodes[at].inputs)):
            wanted[plan.nodes[at].inputs[i]] = True
        _demand(plan, at, bound, need)

    for at in range(root + 1):
        if not wanted[at]:
            continue
        _narrow(plan, at, need[at], sources)

    return bind(plan, root, sources)


def _demand(
    mut plan: Plan, at: Int, bound: List[Bound], mut need: List[List[Int]]
) raises:
    """Adds to each input of a node what that node needs from it.

    Args:
        plan: The plan.
        at: The node, whose own demand is already complete.
        bound: What every node produces, for the join's left width.
        need: The demand on every node, added to.

    Raises:
        If an expression holds a column that binding has not reached.
    """
    var kind = plan.nodes[at].kind
    if kind == NodeKind.SCAN:
        return

    # Copied out once, because every branch below hands it to a routine that
    # writes into the same list of sets it came from.
    var here = need[at].copy()

    if kind == NodeKind.UNION:
        # Stacked, so every input answers the same positions, and asking all of
        # them for the same set is what keeps them the same width.
        #
        # A union that drops duplicates is the same case as a distinct with no
        # keys and needs the same answer. Two rows that differ only in a column
        # nothing above reads are two rows, and a union narrowed to the columns
        # above it are one, so narrowing that union deletes a row rather than a
        # read. It asks for the whole of every input for that reason.
        #
        # A difference and an intersection compare whole rows whatever the
        # duplicate flag says, since deciding whether a row is on both sides is
        # the operation rather than a step in it. They ask for the whole of every
        # input always.
        var whole = (
            not plan.nodes[at].flags[0] or plan.nodes[at].op != SET_UNION
        )
        for i in range(len(plan.nodes[at].inputs)):
            var input = plan.nodes[at].inputs[i]
            _want_all(need[input], here)
            if whole:
                for p in range(len(bound[input].schema)):
                    _want(need[input], p)
        return

    if kind == NodeKind.JOIN:
        var left = plan.nodes[at].inputs[0]
        var right = plan.nodes[at].inputs[1]
        var width = len(bound[left].schema)
        for i in range(len(here)):
            var p = here[i]
            if p < width:
                _want(need[left], p)
            else:
                _want(need[right], p - width)
        var keys = plan.nodes[at].parts
        for i in range(len(plan.nodes[at].exprs)):
            var side = left if i < keys else right
            _want_all(need[side], plan.exprs.positions(plan.nodes[at].exprs[i]))
        return

    var input = plan.nodes[at].inputs[0]

    if kind == NodeKind.PROJECT or kind == NodeKind.AGGREGATE:
        # The only two that rename their columns, so a position above says
        # nothing about a position below and the expressions are the whole of
        # the answer. An output nobody kept is an output whose columns nobody
        # has to read, which is where most of what this pass saves comes from.
        var keep = _kept(plan, at, here)
        for i in range(len(keep)):
            _want_all(
                need[input],
                plan.exprs.positions(plan.nodes[at].exprs[keep[i]]),
            )
        return

    # Filter, sort, limit and distinct change which rows survive and in what
    # order, never which columns, so a position above is the same position
    # below and their own expressions are extra.
    _want_all(need[input], here)
    for i in range(len(plan.nodes[at].exprs)):
        _want_all(need[input], plan.exprs.positions(plan.nodes[at].exprs[i]))
    if kind == NodeKind.DISTINCT and len(plan.nodes[at].exprs) == 0:
        # A distinct with no keys is a distinct on the whole row, and the whole
        # row is every column of it whatever anything above wants back.
        for p in range(len(bound[input].schema)):
            _want(need[input], p)


def _narrow(
    mut plan: Plan, at: Int, need: List[Int], sources: List[Schema]
) raises:
    """Narrows one node to the columns demanded of it.

    Args:
        plan: The plan, written through.
        at: The node.
        need: The output positions anything above it reads.
        sources: The schema of each relation.

    Raises:
        If a scan names a relation with no schema.
    """
    var kind = plan.nodes[at].kind

    if kind == NodeKind.SCAN:
        var names = plan.nodes[at].names.copy()
        if len(names) == 0:
            var table = plan.nodes[at].table
            if table < 0 or table >= len(sources):
                raise Error(
                    String(
                        "scan of ",
                        plan.nodes[at].source,
                        " is relation ",
                        table,
                        " and ",
                        len(sources),
                        " schemas were given",
                    )
                )
            for i in range(len(sources[table])):
                names.append(sources[table][i].name)
        if not _distinct(names):
            return
        var keep = List[String]()
        for i in range(len(need)):
            keep.append(names[need[i]])
        if len(keep) == 0:
            # An empty list on a scan means every column, which is the opposite
            # of what an empty demand asked for, so the narrowest list this can
            # honestly write is one column long. `SELECT count(*)` is the query
            # that gets here.
            keep.append(names[0])
        plan.nodes[at].names = keep^
        return

    if kind != NodeKind.PROJECT and kind != NodeKind.AGGREGATE:
        return

    var keep = _kept(plan, at, need)
    if len(keep) == len(plan.nodes[at].exprs):
        return
    var exprs = List[Int]()
    var names = List[String]()
    for i in range(len(keep)):
        exprs.append(plan.nodes[at].exprs[keep[i]])
        names.append(plan.nodes[at].names[keep[i]])
    plan.nodes[at].exprs = exprs^
    plan.nodes[at].names = names^


def _kept(plan: Plan, at: Int, need: List[Int]) raises -> List[Int]:
    """Which outputs of a project or an aggregate survive.

    Args:
        plan: The plan.
        at: The node.
        need: The output positions anything above it reads.

    Returns:
        The surviving outputs, as offsets into the node's expression list, in
        order.
    """
    var out = List[Int]()
    if not _distinct(plan.nodes[at].names):
        # Ambiguous already, and narrowing it would decide which of two columns
        # of one name an expression above meant. Answered here rather than in
        # the rewrite so that the demand and the rewrite cannot disagree: a
        # node that keeps an output has to keep reading what that output is
        # made of.
        for i in range(len(plan.nodes[at].exprs)):
            out.append(i)
        return out^
    var keys = plan.nodes[at].parts
    for i in range(keys):
        # A group key is not a projection. Dropping one changes which rows come
        # out, so it stays whatever anything above thinks of it.
        out.append(i)
    for i in range(len(need)):
        if need[i] >= keys:
            out.append(need[i])
    if len(out) == 0 and len(plan.nodes[at].exprs) > 0:
        # A node with no columns is not a node. Nothing above wants anything
        # from this one, so which output stays is arbitrary and the first is as
        # good as any.
        out.append(0)
    return out^


def _distinct(names: List[String]) -> Bool:
    """Whether every name in a list is different from every other.

    Args:
        names: The names.

    Returns:
        True when no two of them are the same.
    """
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if names[i] == names[j]:
                return False
    return True


def _want(mut set: List[Int], at: Int):
    """Adds one position to an ascending set, if it is not there already.

    Args:
        set: The set, ascending and without repeats.
        at: The position.
    """
    var to = 0
    while to < len(set) and set[to] < at:
        to += 1
    if to == len(set):
        set.append(at)
    elif set[to] != at:
        set.insert(to, at)


def _want_all(mut set: List[Int], more: List[Int]):
    """Adds every position of one set to another.

    Args:
        set: The set, ascending and without repeats.
        more: The positions to add.
    """
    for i in range(len(more)):
        _want(set, more[i])
