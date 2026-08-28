"""Running work on more than one core.

Tier: unstable, documented. docs/specs/11-package-layout.md.

Today this is one file and one function. `parallel_for` runs a body once per
index and returns when every index has finished, and every other package that
wants a second core goes through it. The CSV reader is the first caller and the
group by and the join are the ones that will care most.

The morsel scheduler, the selection vectors and the operator graph that the
package layout document puts here are M6 work and are not here yet. What is here
is deliberately the smallest thing that lets a kernel use the machine, so that
the scheduler when it arrives replaces something measured rather than something
imagined.

`shared.mojo` does not exist yet either, and that is the point of the CI grep
that asserts no `Atomic` and no mutable global appears outside this package.
`parallel_for` needs neither: each index writes its own slot and nothing is read
until every task has been waited on.
"""

from .parallel import parallel_for, worker_count
