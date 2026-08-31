# Polars, what changed recently

## Pace

Twelve releases between December 2025 and April 2026. 778 merged pull requests, 95 contributors. Current version in August 2026 is 1.39.

The practical consequence is that any blog post or talk about Polars internals from before 2025 describes an engine that no longer exists, and even the 1.31 material is out of date on the operator set. The source is the documentation of record.

## 1.37

A new sink pipeline for NDJSON, CSV and IPC. The sinks were rewritten to fit the streaming model properly rather than buffering.

## 1.38

**Streaming merge join.** A join on two sorted inputs that needs no hash table, so it is bounded memory by construction.

**`sink_delta()`.** Writing Delta Lake tables from a streaming query.

## 1.39

**Streaming AsOf join.** The time series join, in the streaming engine. This is a notable one because AsOf is inherently ordered and getting it into a morsel engine requires the sequence number machinery from `03-streaming-engine.md` to actually work.

**`sink_iceberg()`.**

**Cloud file streaming for `scan_ndjson()` and `scan_lines()`**, so a remote file is consumed as it arrives rather than downloaded first.

**`arg_min` and `arg_max` lowered to streaming**, which is the kind of entry that shows up in every release: one more expression removed from the fallback list.

## The categorical rewrite

Pull request #23016, covered in `01-data-model.md`. Explicit `Categories` objects replacing per Series mappings and the global `StringCache`. A breaking change to a core type, made because the streaming engine required it.

## The out of core work

Group by, equi join and sort wired to the lock free memory manager. The multiplexer made fully out of core in #26774. Covered in `04-out-of-core.md`.

## Benchmarks published in this window

Streaming engine three to seven times the in memory engine on PDS-H, growing with data size.

Distributed, June 2026: up to 7.7 times Spark, 3.2 times average. Single node up to 38 times, 6.4 times average.

cudf-polars 26.06 on RapidsMPF: 3.2 times on one GPU at PDS-H SF1000, 23.2 times on eight GPUs at SF3000.

## What this means for us

The release pattern is the lesson. Almost every release in this window is either one more operator lowered to streaming, one more sink made streaming, or one more thing made out of core. The engine was not delivered, it was migrated into, over eighteen months of releases, with the old engine as the fallback the whole time.

That is the plan `05-m2b.md` should copy. Not a rewrite with a flag day. A fallback path that is correct from day one, and then a sequence of small pull requests each of which removes one operator from it, each individually benchmarkable, each shippable on its own.

The other thing to take is which operators they thought were worth the effort and in what order. Sinks first, then joins, then the ordered joins, then the long tail of expressions. Sinks first because ETL is the workload that cannot be done any other way, not because sinks are the slowest.
