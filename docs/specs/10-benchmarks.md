# Benchmarking, and the firepanda-bench repository

## Why this is a separate repository

`tamnd/firepanda-bench` exists because comparing against pandas, Polars, DuckDB and cuDF means installing them, and that must never become a condition of building firepanda. The library's pixi environment stays minimal and the benchmark harness carries the Python, the Docker images, the datasets and the result history.

The second reason is cadence. The library changes when someone writes code; the benchmarks need to re-run when pandas releases, when Polars releases, when Mojo releases, when a new machine type appears, and on a schedule regardless. Tying those two rhythms together makes both worse.

Note that this is a different split from the Go sibling spec, where a `-py` repository was needed just to load both libraries into one process. Here Mojo imports pandas directly, so the *correctness* comparison lives in the main repository and only the *performance* comparison needs its own.

## The rule that makes this worth doing

Benchmarks that run once, at the end, before an announcement, are marketing. Benchmarks that run continuously, from the first milestone, are engineering. The difference is that the second kind tells you which commit made things slow while you still remember what you were doing.

So the harness exists from M1, when there is barely anything to measure. At M1 it can only compare CSV reading and a few aggregations, and that is fine, because by M8 when the optimization work lands there is a year of history to check the claims against.

The second rule is that we publish results we lose. A suite that only shows wins is not information, and anyone experienced reads it as an advertisement and discounts everything in it. Where pandas, Polars, DuckDB or cuDF is faster, the number goes in the table with a note about why. This is not modesty, it is the only way the numbers we win stay credible.

For a project whose entire pitch is performance, in a language whose marketing has been criticized for exactly this, the credibility of the numbers is the asset.

## What we are measured against

| | Why it is in the table |
|---|---|
| pandas 3.0.5 | The API we are replacing and the audience we are addressing. 3.0 is faster than the pandas people remember, so this is not a free win. |
| Polars 1.43 | The performance bar and the reference implementation of this design. |
| DuckDB | The correctness oracle as well as a competitor, and the thing that wins TPC-H. |
| MojoFrame | The academic prior art in the same language. A production library that cannot beat a research prototype has not justified itself. |
| cuDF | The GPU comparison from M9. Mature, and the honest baseline for any GPU claim. |

MojoFrame being in that table is a discipline rather than a courtesy. It supports all 22 TPC-H queries and reports up to 4.60x over dataframe libraries in other languages, and its authors published where it falls behind, specifically high cardinality aggregation, for a reason they diagnosed as Mojo's dictionary. We claim to have fixed that with our own hash table. The claim is only meaningful if we measure the query they lost on, which is TPC-H Q18.

## What gets measured

### The db-benchmark suite

The h2oai database benchmark is the standard comparison for this class of library. h2oai stopped maintaining it in 2021 and DuckDB Labs revived it as `duckdblabs/db-benchmark`, which is the version everyone now cites. The frozen `h2oai.github.io` page is from 2021 and must never be used for comparison.

Ten group by queries and five join queries at 0.5 GB, 5 GB and 50 GB. The first two sizes in CI, the 50 GB size on demand, since it needs a machine most runners are not.

One methodological note matters more than the rest. ClickHouse and DuckDB use `CREATE TABLE ans AS SELECT` so that lazy engines are forced to materialize. Our harness does the same, because benchmarking a lazy engine without materializing the result measures nothing at all. This is not a theoretical concern for us: after M4 the *eager* API is lazy underneath, so a benchmark that calls `df.groupby(...).sum()` and never looks at the answer measures plan construction.

### TPC-H

Twenty two queries at scale factor 1 and 10. Where db-benchmark stresses group by and join in isolation, TPC-H stresses the optimizer, because the wins there come from projection pushdown, predicate pushdown, join ordering and partition pruning rather than kernel speed.

This is the suite that catches optimizer regressions and the one that will hurt early, since DuckDB and Polars have had years on exactly these queries.

Q18 gets called out specifically for the MojoFrame comparison above. Q13 gets called out because it is UDF heavy and it is where the language argument in document 00 should show up as a number.

### The UDF benchmark, which is ours to define

No standard suite measures the thing this project is actually about, so we define one and we define it in a way that is fair rather than flattering.

Six operations of increasing complexity, each expressed four ways: a pandas vectorized expression where one exists, `pandas.apply` with a Python lambda, `firepanda` as an expression, and `firepanda` with a Mojo function compiled into the pipeline. Plus, from M3, `firepanda` with a Python lambda through the binding, which is the slow path from document 04 section 7 and which goes in the table at its real cost.

That fifth column is the one that keeps this honest. It is easy to build a UDF benchmark where the Mojo column wins by 100x and quietly omit that a Python user calling `df.apply(lambda ...)` gets the fourth-place number.

### Ingestion

CSV and Parquet read throughput, cold and warm cache, against pandas with the pyarrow engine, Polars and DuckDB. First impressions are made here: `read_csv` is the first line of code almost every user writes.

Parquet gets measured twice after M8, once through the M2 Arrow C++ binding and once through the native reader, because that is how we know whether the native reader was worth writing.

### Microbenchmarks

In the main repository rather than here, because they run on every pull request and must not require Python. Individual kernels, the dense path against the masked path, chunk size sweeps, the hash table against `Dict` for the record, and the fused expression path against the interpreted one after M8.

A regression gate on every pull request, with a threshold that fails the build.

### Compile time and binary size

Not conventionally a benchmark and it belongs here anyway, because under the monomorphization strategy in document 03 it is the resource most likely to blow up, and the thresholds in document 08 M8 need a year of data behind them.

Recorded per commit from M0: clean build wall time, incremental build wall time, stripped library size, and the count of instantiated kernel bodies.

### Memory

Peak RSS on every db-benchmark and TPC-H query, alongside the timing. A library that is 2x faster and uses 4x the memory has not won and the table should be able to say so.

## Methodology

Fixed instance types, recorded in the results. Same machine for every engine in a given run, always.

Ten runs, report the median, publish the interquartile range. A single number with no spread is not a measurement.

Cold and warm cache reported separately for anything touching a file.

Version pinning for every engine, including our own Mojo toolchain version, recorded in the results file. A benchmark result that does not say which Mojo compiled it is not reproducible, given that the ABI is unstable and the codegen is changing release to release.

Docker for the comparison engines so the environment is reproducible.

`_utils/repro.sh` equivalent, so anyone can rerun the whole thing.

## Publishing

A static site from the results repository, updated on every scheduled run, with time series per query so a regression is visible as a step in a graph rather than as a number somebody has to compare by hand.

Every table includes the losses. Where we lose, a one line note says why, and if the reason is "not optimized yet" it says that rather than being omitted.

The README quotes numbers with a date and a link to the run that produced them. A performance claim with no date is a claim about a version that no longer exists.

## What not to do

Do not benchmark against pandas 2.x. It is what most people are running and it is not what we are competing with, and using it would inflate every number in a way that is indefensible the moment somebody notices.

Do not benchmark the eager API without forcing materialization. See above.

Do not report a GPU number against a CPU baseline without also reporting cuDF. The interesting comparison for a GPU dataframe is another GPU dataframe.

Do not quote the Mojo marketing numbers. Whatever the language's headline figures are, ours are the ones we measured on this library, and borrowing someone else's is how a project loses the benefit of the doubt on all of its own numbers.
