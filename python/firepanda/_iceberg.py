"""`read_iceberg` and `DataFrame.to_iceberg`, which hand a table to pyiceberg as pandas does.

pandas loads the catalog, scans or creates the table, and moves the rows
through pyarrow: a scan's `to_pandas` on the way in, and `Table.from_pandas`
on the way out. firepanda makes the same calls to pyiceberg and moves the
rows through the same pyarrow table, read and written with the pandas
metadata by `_columnar`, so a table reads back with pandas' types and labels.
"""

from __future__ import annotations

from typing import Any

from . import _optional


def read_iceberg(
    table_identifier: str,
    catalog_name: str | None = None,
    *,
    catalog_properties: dict[str, Any] | None = None,
    columns: list[str] | None = None,
    row_filter: str | None = None,
    case_sensitive: bool = True,
    snapshot_id: int | None = None,
    limit: int | None = None,
    scan_properties: dict[str, Any] | None = None,
) -> Any:
    """Read an Apache Iceberg table into a frame."""
    from . import _columnar

    catalogs = _optional.imported("pyiceberg.catalog")
    expressions = _optional.imported("pyiceberg.expressions")
    if catalog_properties is None:
        catalog_properties = {}
    catalog = catalogs.load_catalog(catalog_name, **catalog_properties)
    table = catalog.load_table(table_identifier)
    if row_filter is None:
        row_filter = expressions.AlwaysTrue()
    selected_fields = ("*",) if columns is None else tuple(columns)
    if scan_properties is None:
        scan_properties = {}
    result = table.scan(
        row_filter=row_filter,
        selected_fields=selected_fields,
        case_sensitive=case_sensitive,
        snapshot_id=snapshot_id,
        options=scan_properties,
        limit=limit,
    )
    return _columnar.frame_of(result.to_arrow())


def to_iceberg(
    self: Any,
    table_identifier: str,
    catalog_name: str | None = None,
    *,
    catalog_properties: dict[str, Any] | None = None,
    location: str | None = None,
    append: bool = False,
    snapshot_properties: dict[str, str] | None = None,
) -> None:
    """Write the frame to an Apache Iceberg table, replacing its rows or appending to them."""
    from . import _columnar

    _optional.imported("pyarrow")
    catalogs = _optional.imported("pyiceberg.catalog")
    if catalog_properties is None:
        catalog_properties = {}
    catalog = catalogs.load_catalog(catalog_name, **catalog_properties)
    arrow_table = _columnar.table_of(self, None)
    table = catalog.create_table_if_not_exists(
        identifier=table_identifier,
        schema=arrow_table.schema,
        location=location,
    )
    if snapshot_properties is None:
        snapshot_properties = {}
    if append:
        table.append(arrow_table, snapshot_properties=snapshot_properties)
    else:
        table.overwrite(arrow_table, snapshot_properties=snapshot_properties)
