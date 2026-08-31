# DuckDB, what changed recently

Where the project is as of August 2026, so that the rest of this folder can be dated.

## Versions

**1.4 LTS**, the long term support line, supported until September 2026. This is what conservative deployments are on and what a compatibility claim should target.

**1.5.5**, July 2026, the current release. The 1.5 line opened in March.

**2.0** is expected in the autumn of 2026. No public feature list yet worth writing down.

## 1.5.0, "Variegata", March 2026

The big one.

**VARIANT.** A self describing binary encoding for semi structured data, in the core engine rather than an extension. The point is that JSON stored as text is reparsed on every access. VARIANT parses once. Combined with shredding, which pulls frequently accessed fields into their own physical columns, DuckDB reports up to a hundredfold on JSON analysis workloads. `VariantColumnData` in the storage layer is where the shredding lives.

**GEOMETRY.** Spatial primitives promoted into the core type system.

**Checkpoint concurrency.** Checkpointing no longer requires the exclusive access it used to.

**Reworked CLI**, **Azure write support**, and **DuckLake v0.4**, which is their lakehouse format layered on a catalog database.

## 1.5.1, March 2026

**Lance** format support. **Iceberg v3.** ART index fixes.

## 1.5.3, May 2026

**Quack**, a client server protocol, shipped as a core extension. This is the one that changes what DuckDB is. Until now the answer to "can two processes share a database" was no, or MotherDuck. Quack makes a DuckDB process serve remote clients over a wire protocol.

It does not make the engine distributed. There is still one process doing the work.

## 1.5.4 and 1.5.5, June and July 2026

Bug fixes, plus an experimental `vacuum_rebuild_indexes` for reclaiming space from ART indexes after heavy deletes.

## What this means for us

Three things.

The comparison target is 1.5.5. Any benchmark we publish should name the version, and firepanda-bench should be pinned to the latest release and refreshed when it moves. It currently is.

VARIANT and GEOMETRY are out of scope. firepanda is a dataframe library and the roadmap does not have a semi structured type in it. If a JSON reader lands in M2 it produces struct and list columns, not a variant.

Quack is a signal about where the whole category is going, which is toward the engine being reachable from more than one process, and toward the lakehouse formats being first class. Iceberg v3, Lance and DuckLake in one release line is not an accident. Polars shipped `sink_iceberg()` in 1.39 in the same period. If firepanda gets to M5 and beyond, table format readers are the thing users will ask for, not another aggregation.
