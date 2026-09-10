"""The plan: what a query says, before anything decides how to run it.

Tier: unstable, documented. docs/specs/11-package-layout.md, and the design
is docs/specs/planner/01-what-a-plan-is.md.

Three representations, kept apart, and this package holds the first two. The
tree the user's calls build, the tree that survives binding and rewriting, and
the physical plan, which is `firepanda/exec/`.

Only the expression layer is here so far. `expr.mojo` has the nine expression
kinds, the arena they live in, and the elementwise, input independent and table
set analyses. Every pass in docs/specs/planner/02-the-pass-pipeline.md is
written in terms of those three, which is why they arrive in the first file
rather than with the first pass that wants one.

The logical nodes, binding, the passes and the lowering into `exec` follow, and
the name collision between `plan/node.mojo` and `exec/node.mojo` when the first
of those lands is the right collision, because they are the same concept at two
levels.
"""

from .expr import UNBOUND, Expr, ExprKind, Expressions
