"""`firepanda.plotting`, pandas' `pandas.plotting`.

`df.plot`, `s.hist` and `df.boxplot` draw with matplotlib the way pandas draws,
and this namespace holds the rest of pandas' plotting: the scatter matrix, the
lag and autocorrelation plots, the Andrews curves and parallel coordinates, the
table, and `plot_params`. Nothing imports matplotlib until something is drawn,
so `import firepanda` does not need it.
"""

from __future__ import annotations

from .._plotting import (
    PlotAccessor,
    andrews_curves,
    autocorrelation_plot,
    bootstrap_plot,
    boxplot,
    boxplot_frame,
    boxplot_frame_groupby,
    hist_frame,
    hist_series,
    lag_plot,
    parallel_coordinates,
    plot_params,
    radviz,
    scatter_matrix,
    table,
)
from .._plotting import deregister as deregister_matplotlib_converters
from .._plotting import register as register_matplotlib_converters

__all__ = [
    "PlotAccessor",
    "andrews_curves",
    "autocorrelation_plot",
    "bootstrap_plot",
    "boxplot",
    "boxplot_frame",
    "boxplot_frame_groupby",
    "deregister_matplotlib_converters",
    "hist_frame",
    "hist_series",
    "lag_plot",
    "parallel_coordinates",
    "plot_params",
    "radviz",
    "register_matplotlib_converters",
    "scatter_matrix",
    "table",
]
