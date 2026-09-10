# The planner folder

Written 10 September 2026, against DuckDB 1.5.5, Polars 2.0 rc1 (2 September 2026) and the TPC-H work in tamnd/firepanda-bench#22.

## Why this exists

The `engine/` folder next door is about how a query runs: morsels, pipelines, chunked operators, spilling. It is right about all of that and it says almost nothing about how the shape of the query is decided in the first place. `engine/duckdb/05-optimizer.md` closes with a sentence saying the pass pipeline and DPhyp are M3 material and belong in the lazy frame spec. This is that spec.

It exists now rather than later because we just measured what it is worth, by hand, twenty two times.

The TPC-H driver in firepanda-bench started at 6.671 seconds for the twenty two queries at sf1 and is at about 1.4 seconds. One kernel changed in that time. Everything else was plan work done by a person: pushing predicates to the table they read, projecting columns away before a join instead of after it, reordering joins so the selective one runs first, and picking a different physical operator for the same logical group by depending on how many groups there were. That is a factor of four point eight from decisions a planner makes automatically and firepanda currently makes not at all.

The conclusion is uncomfortable and worth stating plainly. More kernel optimization is the wrong next move. Our kernels are within a small factor of Polars' and DuckDB's, and on several queries they win. What we do not have is anything that decides what to run.

## What is in here

`01-what-a-plan-is.md` is the intermediate representation. What a logical plan node is, what an expression is, how schemas get resolved, and the three representations Polars keeps and why.

`02-the-pass-pipeline.md` is the rewrite passes, in order, each with what it buys measured against the hand tuning we did on TPC-H.

`03-join-ordering.md` is the classical problem. DPccp, DPhyp, the Join Order Benchmark result, and why we are deliberately not starting here.

`04-predicate-transfer.md` is the paper that changes the answer. Yannakakis, Predicate Transfer, Robust Predicate Transfer, and the argument that a good enough plan reached robustly beats an optimal plan reached by guessing.

`05-cardinality-and-cost.md` is what we need to know about the data to make any of these decisions, how wrong the estimates are, and how little of it we are going to build.

`06-runtime-filters.md` is the decisions made after the query starts. Join filter pushdown, min and max transfer, bloom filters, and where the line is between this and the previous document.

`07-operator-selection.md` is the physical layer. Same logical operation, several implementations, and the rule for choosing. This is the one we can do first because we already have the implementations.

`08-the-plan.md` is what firepanda builds, in what order, and what each step is expected to be worth.

`09-refactoring.md` is how the existing code changes to accommodate it, and what stays exactly as it is.

`10-checklist.md` is the milestone issue.

## How to read this if you are short of time

Read `04-predicate-transfer.md` and `07-operator-selection.md`.

The first one is the strategy. Forty years of query optimization say the way to make joins fast is to find the right order, and finding the right order requires cardinality estimates, and the estimates are wrong. The 2024 and 2025 work says you can sidestep most of that: filter every table by what every other table implies about it before you join anything, and then the join order stops mattering very much. That result is measured, on DuckDB, on all of TPC-H, JOB and TPC-DS, and it is a much better fit for a young engine than a cost model is.

The second one is what we can ship next week, because it needs no plan at all.

## A note on sources

Everything here is from public papers, documentation and source, read in September 2026. Where a number is quoted a primary source states it, and the source is named. Where a number is ours it says so and names the machine. Where something is an inference it says so.

The measurements attributed to firepanda are from gamingpc, an i9-13900K with sixteen physical and thirty two logical cores, thirty six megabytes of L3 and thirty one gigabytes of memory, running TPC-H sf1 against pandas 3.0.5, polars 1.44.2 and duckdb 1.5.5.
