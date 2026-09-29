"""`read_spss`, which reads an SPSS file through pyreadstat as pandas does.

pyreadstat reads the file into a pandas frame and a record of its metadata,
and pandas returns that frame with the metadata as `attrs`. pyreadstat always
makes a pandas frame, so pandas is installed wherever it runs, and the frame
is carried over to firepanda through a pyarrow table with the pandas metadata,
which keeps the labels as categories and the moments as moments.
"""

from __future__ import annotations

import os
from typing import Any

from . import _columnar, _optional
from ._pandas import NO_DEFAULT


def read_spss(
    path: Any,
    usecols: Any = None,
    convert_categoricals: bool = True,
    dtype_backend: Any = NO_DEFAULT,
    **kwargs: Any,
) -> Any:
    """Read an SPSS file into a frame, with the file's metadata as `attrs`."""
    pyreadstat = _optional.imported("pyreadstat")
    if dtype_backend is not NO_DEFAULT:
        from ._pandas import _backend

        _backend(dtype_backend)
    if usecols is not None:
        if not hasattr(usecols, "__iter__") or isinstance(usecols, (str, bytes)):
            raise TypeError("usecols must be list-like.")
        usecols = list(usecols)
    frame, metadata = pyreadstat.read_sav(
        os.fspath(path) if isinstance(path, os.PathLike) else path,
        usecols=usecols,
        apply_value_formats=convert_categoricals,
        **kwargs,
    )
    pa = _optional.imported("pyarrow")
    result = _columnar.frame_of(pa.Table.from_pandas(frame))
    result.attrs = metadata.__dict__
    if dtype_backend is not NO_DEFAULT:
        result = result.convert_dtypes(dtype_backend=dtype_backend)
    return result
