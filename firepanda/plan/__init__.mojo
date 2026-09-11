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

The rest of the passes and the lowering into `exec` follow. Nothing calls any
of this yet and the eager API does not change when they do.
"""

from .bind import Bound, bind, bind_expr
from .expr import UNBOUND, Expr, ExprKind, Expressions
from .node import NO_LIMIT, NodeKind, Plan, PlanNode
from .print import explain, render_expr
from .simplify import ROUNDS, simplify, simplify_expr
