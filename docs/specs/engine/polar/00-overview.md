# Polars, the shape of the whole thing

## What it is

A dataframe library written in Rust with a Python front end, built on Arrow memory, with a lazy API that compiles to a query plan. The eager API exists and is a thin wrapper over the lazy one for most operations.

Current version in August 2026 is 1.39. Between December 2025 and April 2026 there were twelve releases, 778 merged pull requests and 95 contributors, so anything written about Polars internals more than a few months old is describing a different engine.

## Why it matters more to us than DuckDB does

DuckDB is a database with a dataframe API bolted to the side. Polars is the thing firepanda is trying to be, in a different language, five years ahead. Every design decision they made is a decision we will have to make, and most of them they have already changed their minds about at least once.

The specific reason to read this now: Polars had exactly firepanda's architecture and abandoned it. The old Polars engine ran a kernel over a whole column, the way our kernels do. They built a streaming engine to replace it and their own numbers say it is three to seven times faster on the PDS-H benchmark as data grows. Not because the kernels got faster. Because the engine stopped materializing between operators.

## The four layers

**Memory.** Arrow arrays, chunked. A `Series` is a chunked array of one type, a `DataFrame` is a set of Series. Document 01.

**The plan.** A Python or Rust DSL builds an unoptimized logical plan, the optimizer rewrites it into an IR, and a physical planner turns the IR into an executable graph. Document 02.

**The engines.** Two of them. The in memory engine, which is the original, and the new streaming engine, which is morsel driven with async state machines. Document 03. Out of core behaviour, which is a property of the streaming engine, is document 04.

**Beyond one machine.** Polars Cloud for distributed and cudf-polars for GPU, both of which reuse the logical plan and replace the physical layer. Document 05.

Document 06 is the release history, so the rest can be dated.

## The streaming engine is opt in, still

This is the single most surprising fact in this folder and it is worth stating plainly.

As of 1.39, the streaming engine is not the default. You turn it on with `pl.Config.set_engine_affinity("streaming")` or per call with `engine="streaming"`. It is the default only for partitioned sinks. The in memory engine remains the fallback for anything the streaming engine cannot lower, and a plan can be part streaming and part not, with `visualize_plan()` colouring the fallback nodes red.

It has been present since 1.31.1 and the tracking issue is #20947.

What that tells us is that the transition is genuinely hard. Polars has a full time team, five years of code, and it still took them eighteen months of releases to get an operator set complete enough to make the new engine the default, and they have not made it the default yet. When `05-m2b.md` estimates what a firepanda streaming engine costs, this is the calibration.

## What Polars is not

It is not a storage engine. There is no Polars file format, no MVCC, no transactions. It reads Parquet, IPC, CSV, NDJSON, Delta and Iceberg and writes most of those. This is the same position firepanda is in and it is a good one.

It is not a database. No SQL as the primary interface, though there is a SQL front end that lowers to the same plan.

## Why we care

We are behind Polars by two to eight times on every db-benchmark query, and the gap is largest exactly where the engine matters most, which is the joins. j1 through j3 have us at 155, 191 and 190 milliseconds against their 27, 29 and 41. That is a five to six times gap on queries where our join kernel is individually reasonable, and the reason is that we do six full passes over the data where they do one.
