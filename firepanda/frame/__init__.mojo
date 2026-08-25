"""The frame layer: `Series`, `DataFrame`, and the eager surface.

This is the first package in firepanda that a user is meant to import directly.
Everything below it exists to make this work and everything above it, starting
with the plan layer at M4, exists to make it fast on more than one operation at a
time.

The layering rule from `docs/specs/11-package-layout.md` runs in one direction
only: the frame calls the kernels and the kernels have never heard of a frame. It
holds without exception in this package, and the place it was almost broken is
worth recording. `DataFrame.filter` needs to filter columns whose dtypes differ
from each other, which means a runtime dispatch, which lives in the kernel layer
next to the loop it dispatches to rather than here. That is why `take_any`,
`filter_any` and `cast_any` are in `firepanda.kernel` and not in this file.

`display.mojo` sits under both `frame.mojo` and `series.mojo` rather than inside
either. It renders a `Schema` and a list of columns, which is what the two have in
common, and taking that rather than a `DataFrame` is what lets both import it
without a cycle.
"""

from .display import DisplayOptions, render_column, render_table
from .frame import DataFrame
from .series import Series
