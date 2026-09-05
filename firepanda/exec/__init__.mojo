"""The engine: what a unit of work is, who does it, and on how many cores.

Tier: unstable, documented. docs/specs/11-package-layout.md.

Three layers, and each one is useful without the one above it.

`Chunk`, `Node` and `Pipeline` are the engine. A chunk is a horizontal slice of
a frame, a node is one operator, and a pipeline is a source, a line of nodes and
a sink, run by pushing chunks along the line. Every operator not yet written for
chunks goes through the `Materialize` fallback, which collects, calls today's
whole frame function and hands the answer back in chunks, so the engine can be
filled in one operator at a time rather than all at once.

Underneath that are two ways of using the machine, and the difference between
them is when the work is divided up.

`parallel_for` runs a body once per index and returns when every index has
finished. The division happens before any work is done, so it is right when the
pieces cost the same and it leaves cores idle when they do not.

`parallel_morsels` runs a body over a range of rows in fixed size pieces, handed
out by a shared counter to whichever worker asks next. Nothing is decided in
advance, so a piece that turns out to be expensive costs the job one morsel of
tail rather than one worker's whole share. That is the scheduler the engine in
docs/specs/engine is built on, and the operators move onto it one at a time.

The atomic that the counter needs is the reason this package is the one place
allowed to have one. A grep in CI asserts that `Atomic` and mutable globals
appear nowhere outside `morsel.mojo`, and in there they are one field of
`MorselQueue`. `parallel_for` still needs neither: each index writes its own slot
and nothing is read until every task has been waited on.
"""

from .chunk import Chunk
from .morsel import MORSEL_ROWS, Morsel, MorselQueue, parallel_morsels
from .node import Cast, Compute, Filter, Group, GroupAgg, Limit, Materialize
from .node import Node, NodeStatus, Project
from .node import node_apply, node_bind, node_computes_per_row
from .node import node_ends_early, node_finish, node_is_breaker
from .node import node_is_row_local, node_process, node_status
from .parallel import parallel_for, worker_count
from .pipeline import Collect, Pipeline, Scan
