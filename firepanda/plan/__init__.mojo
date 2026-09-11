"""The plan: what a query says, before anything decides how to run it.

Tier: unstable, documented. docs/specs/11-package-layout.md, and the design
is docs/specs/planner/01-what-a-plan-is.md.

Three representations, kept apart, and this package holds the first two. The
tree the user's calls build, the tree that survives binding and rewriting, and
the physical plan, which is `firepanda/exec/`.

`expr.mojo` has the nine expression kinds, the arena they live in, and the
elementwise, input independent and table set analyses. Every pass in
docs/specs/planner/02-the-pass-pipeline.md is written in terms of those three,
which is why they arrive in the first file rather than with the first pass that
wants one.

`node.mojo` has the nine logical node kinds and the arena, which holds the
expression arena inside it so that a plan is one value rather than a pair a
caller has to keep together. The name collision with `exec/node.mojo` is the
right collision, because they are the same concept at two levels: a node here
says what the query wants and a node there says how one chunk of it is
computed.

`print.mojo` writes a plan out as an indented tree with the expressions in the
notation they were written in.

`bind.mojo` turns every name into a position and gives every expression a type,
which is what stops execution looking columns up by name and what makes a type
error a plan error rather than a kernel error.

`simplify.mojo` is the first pass, and the cheapest. It folds an expression
that reads no column down to its answer, turns a comparison round so the
constant is on the right, flattens the connectives and applies the boolean
identities, all to a fixed point.

`prune.mojo` is projection pushdown, the pass the spec calls the largest of the
thirteen. It works out which columns anything above a node reads, narrows every
scan, project and aggregate to those, and hands the plan back to `bind` so that
the positions come out right without any of them being remapped by hand.

`push.mojo` is predicate pushdown, which moves every filter as far toward the
scans as it can go, splitting it at its `and` nodes first so that the halves can
end up in different places. It is the one pass that rebuilds the node list rather
than rewriting it, because moving a filter down makes new parents for old
children and the arena's order forbids that in place.

`transit.mojo` is transitive predicates, which pushdown calls when it reaches an
inner join. A query that says `a.k = b.k` and also says something about `a.k` is
saying the same thing about `b.k` for every row the join will produce, so the
predicate is copied onto the other side and pushdown places the copy the way it
places everything else. It is a module rather than a pass because there is no
index between a node and its parent, so only the pass that already rebuilds the
node list can put a new filter above an arm.

`merge.mojo` folds a line of projections down to one, substituting the lower
one's expressions into the upper one's so that the rows are walked once instead
of once per node. It is the pass that pays for the two before it, since narrowing
a node to the columns above it is done by putting a projection there.

`limits.mojo` is slice pushdown and top n. It combines a limit with the limit
below it, swaps a limit past a projection so that the projection evaluates n
rows rather than all of them, and turns a limit above a sort into a bound on
the sort, which is what lets a sort keep the best n rows as they go past
instead of ordering the whole thing.

`cse.mojo` is common subexpression elimination. Within one plan node, two
expressions that are the same shape become one index, which lowering then
computes once because it remembers where it put an index it has already met. It
works a node at a time because binding writes a position onto a column and the
same name under two nodes can bind to two different positions.

`empty.mojo` is empty and constant pruning. A filter whose predicate folded to
false becomes a limit of zero rows, which is an empty relation with the schema
it had, and a filter that folded to true is spliced out, as is any filter, sort,
distinct or limit sitting over something empty. It is the only pass that makes
the plan smaller rather than different.

`subplan.mojo` is common subplan elimination, the other half of the section that
gave us the expression one and a level up from it. Two plan nodes of the same
shape over the same inputs become one node, so `df.filter(cond).select(a)` and
`df.filter(cond).select(b)` on two adjacent lines share their filter. It is the
only pass that leaves the plan a graph rather than a tree, which is why it runs
once at the end rather than inside the loop.

`optimize.mojo` is the pipeline: every pass above, in the order the spec fixes,
run again if a run changed anything and up to a small bound. It is the one entry
point, and the passes that are not written yet slot into it and nowhere else.

`lower.mojo` turns a bound plan into a `Pipeline` of physical operators, which
is what makes any of the rest of it reachable from a running query. It lowers a
line of scan, filter, projection and limit, and it raises by name on everything
nobody has written an operator for yet, so a caller can try it and fall back to
what it did before at no cost.

The rest of the passes follow. Nothing calls any of this yet and the eager API
does not change when they do.
"""

from .bind import Bound, bind, bind_all, bind_expr
from .cse import cse
from .empty import empty
from .expr import UNBOUND, Expr, ExprKind, Expressions
from .limits import limits
from .lower import lower
from .merge import merge
from .node import NO_LIMIT, NodeKind, Plan, PlanNode
from .optimize import SWEEPS, optimize
from .print import explain, render_expr
from .prune import prune
from .push import push
from .simplify import ROUNDS, simplify, simplify_expr
from .subplan import subplan
from .transit import derive
