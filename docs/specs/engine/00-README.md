# The engine folder

Written 31 August 2026, against DuckDB 1.5.5 (July 2026) and Polars 1.39 (April 2026).

## Why this exists

The first twelve documents in this spec describe firepanda as a library: a memory layout, a set of kernels, a pandas shaped surface, and a milestone list. They are correct about all of that and they are thin about the part that actually decides whether firepanda is competitive, which is the engine. Documents 02 and 08 name a lazy layer, a morsel scheduler and a memory manager, and then say very little about what any of those three are.

That gap showed up in the work. The last several releases have been kernel level optimizations, and they have been good ones, but they are optimizations to a shape that both DuckDB and Polars abandoned years ago. firepanda today runs one kernel over one whole column at a time, materializes the result, and moves to the next kernel. That is the pandas execution model. It beats pandas because Mojo beats Cython, and it will keep losing to DuckDB by five to eight times on joins no matter how good the individual kernels get, because the gap is not in the kernels.

So before more kernel work, this folder writes down how the two engines we are chasing actually execute a query, and then what firepanda is going to do about it.

## What is in here

`duckdb/` is six documents on DuckDB. Data model, storage, execution, operators, optimizer, and what changed in 2025 and 2026. DuckDB matters to us because it is the fastest thing on the benchmarks we run and because it is a database, which means its answers to memory pressure and spilling are the mature ones.

`polar/` is six documents on Polars. Data model, the lazy IR, the streaming engine, out of core execution, the distributed and GPU backends, and what changed recently. Polars matters to us because it is a dataframe library rather than a database, so its problems are our problems: an eager surface people expect to behave like pandas, a lazy surface underneath, and a type system that has to survive contact with Arrow.

`01-what-we-take.md` through `04-operator-plan.md` are the firepanda plan. What we copy, what we deliberately do not, and in what order.

`05-m2b.md` is the checklist that became milestone issue M2b.

## How to read this if you are short of time

Read `duckdb/03-execution.md` and `polar/03-streaming-engine.md`. Between them they contain the whole argument. Both engines converged on the same design from opposite directions, and the design is: chop the data into morsels of a few thousand to a few hundred thousand rows, run a whole chain of operators over one morsel while it is still in cache, and let a scheduler hand morsels to threads instead of splitting the work up front.

Then read `02-execution-model.md`, which is what firepanda is going to do.

## A note on sources

Everything here is from public documentation, blog posts, papers and source code, read in August 2026. Where a number is quoted it is because a primary source states it, and the source is named in the text. Where something is an inference it says so.
