"""The public readers, the same objects `firepanda` exports at the top."""

from __future__ import annotations

from .._clipboard import read_clipboard
from .._columnar import read_feather, read_orc, read_parquet
from .._excel import ExcelFile, read_excel
from .._excel_write import ExcelWriter
from .._html_read import read_html
from .._iceberg import read_iceberg
from .._pandas import read_csv, read_json
from .._pickle import read_pickle, to_pickle
from .._spss import read_spss
from .._sql import read_sql, read_sql_query, read_sql_table
from .._stata import read_stata
from .._textread import read_fwf, read_table
from .._xml import read_xml

__all__ = [
    "ExcelFile",
    "ExcelWriter",
    "read_clipboard",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_html",
    "read_iceberg",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_spss",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_stata",
    "read_table",
    "read_xml",
    "to_pickle",
]
