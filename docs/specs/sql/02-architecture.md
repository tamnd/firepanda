# Architecture

## 1. The stages

```
text
  |  tokenizer           document 04
tokens
  |  PEG matcher         document 04, grammar from document 03
parse tree
  |  transformer         document 05
AST
  |  binder              document 05, types from document 06
LogicalPlan + BoundExpr  document 08
  |  optimizer           document 08
LogicalPlan
  |  physical planner    document 09
pipeline graph           firepanda/exec/pipeline.mojo, which exists
  |  morsel execution    firepanda/exec/, kernels in firepanda/kernel/
columns, and a DataFrame
```

Seven stages, and only three of them are work that is specific to SQL. The tokenizer, matcher and transformer are the SQL front end. The binder and the logical plan are shared with the dataframe surface and with `query()`, and they are the part of this specification that has value even if the SQL surface were cancelled. The optimizer and the physical planner are the lazy engine, which firepanda needs anyway. The execution layer already exists.

This layering is the one DuckDB uses and describes, minus the catalog and the storage engine, and it is the one Polars uses if you read the DSL, IR and physical plan split in `docs/specs/engine/polar/02-lazy-ir.md`. Two engines converged on it. We are not going to be the third that tries something else.

## 2. Where this sits in the existing stack

`docs/specs/02-architecture.md` names five layers, L0 runtime through L4 Python, with L2 described as LazyFrame, Expr, plan and optimizer, the engine. L2 is the layer that does not exist. `firepanda/frame/frame.mojo` is eager, `firepanda/exec/pipeline.mojo` is a push driver that a frame method constructs inline, and there is nothing between them that could be called a plan.

SQL enters at L2, and it is the reason L2 gets built.

```
L4  Python     firepanda.sql(), df.sql(), df.query(), the ADBC driver
L3  Eager      DataFrame, Series, a facade over L2
L2  Plan       SQL front end lands here. LogicalPlan, BoundExpr, binder,
               optimizer, physical planner
L1  Storage    Array, ChunkedArray, Bitmap, kernels, unchanged
L0  Runtime    morsel scheduler, hash tables, pipelines, unchanged
```

The constraint that keeps this honest is that the SQL front end may not call into L1 or L0. Its only output is a `LogicalPlan`. If a SQL feature needs a kernel that does not exist, the answer is a kernel in L1 that the dataframe surface can also reach, never a special path from the binder into the execution layer. The `CASE WHEN` in TPC-H q8 becomes the same conditional column kernel that #299 already asks for on the dataframe side, and both surfaces get it at once.

The mirror constraint is the one that will actually be violated if it is not written down: the logical plan may not contain anything that only SQL can produce. No node whose only constructor is the transformer. If SQL can express something the dataframe API cannot, that is a gap in the dataframe API to be filed, not a private extension to the plan. This is what makes the plan equality test in document 01 possible.

## 3. The three artifacts that cross stage boundaries

**The parse tree is generic.** A tree of nodes, each carrying the grammar rule that matched, its token span and its children. It knows nothing about SQL semantics. This is DuckDB's design and the reason for it is that the matcher can then be fully generated from the grammar with no per rule code, which is what makes the grammar the single source of truth rather than a document that drifts from a hand written parser.

**The AST is ours.** It mirrors DuckDB's shapes closely enough that the transformer is a rule by rule translation, and it is not a copy: no catalog references, no storage hints, no extension hooks. It is arena allocated and index referenced, because a tree of owning pointers in a language with strict ownership is a fight and an index into a `List` is not.

**The logical plan is the contract**, and document 08 specifies it. Everything above it is replaceable and everything below it is shared.

Each of the three has a textual form and a parser for that form. This is not decoration. It is what makes every stage testable in isolation, diffable against DuckDB's `EXPLAIN` output, fuzzable with round trip as the invariant, and bisectable when a query gives a wrong answer.

## 4. What runs on which thread

Parsing, transformation and binding are single threaded and short. Document 04 gives the budget: tens of microseconds for a TPC-H query and single digit microseconds for a small one. There is no case for parallelizing any of it and there is a strong case for it being allocation light, because the latency axis is measured over a loop of small statements and an allocator is what such a loop actually spends its time in.

Execution is the existing morsel engine. Nothing here changes `firepanda/exec/parallel.mojo` or the chunk size, and document 09 is explicit that the SQL path produces the same pipelines the dataframe path produces.

The GIL matters at the Python door and only there. Issue #204 already tracks releasing it around execution. A SQL call is a longer unit of work than a kernel call, so it is the case that benefits most, and document 10 says where the release goes.

## 5. Errors

`docs/specs/14-errors-across-the-boundary.md` sets the error model, and this specification adds one requirement to it: every SQL error carries a byte offset into the original query text, and the front door renders it the way DuckDB does, with the line, a caret and the offending token.

There are four error classes and they are different things to a user.

**Syntax.** The matcher failed. Under the compatibility rule this means either the input is genuinely invalid or we have a grammar bug, and the two are distinguished by the differential harness rather than by the user. The message names the position and what was expected, from the rule stack at the failure point.

**Unsupported.** The matcher succeeded and the transformer or the binder refuses. Names the feature, the position and the tracking issue. This is the class that must never be reported as syntax, and document 05 gives the mechanism.

**Binding.** A column does not exist, a function has no matching overload, a type cannot be cast. DuckDB's messages for these are good, they include candidate suggestions, and matching their text is worth doing because `statement error` in the corpus matches on substring.

**Runtime.** Overflow, division semantics, a cast that fails on a value. Document 06 fixes which of these raise and which produce null, because DuckDB's answer differs by operation and by setting.

## 6. What SQL is allowed to touch outside the process

Table functions. `read_csv`, `read_parquet`, `read_json`, `glob`, and the replacement scan that turns `FROM 'file.parquet'` into one of them. These reach the filesystem and, with the right path, the network.

That is a capability boundary and it gets a flag from the first commit rather than after the first incident. `enable_external_access` defaults to true for the in process library call and false for any front door that accepts a query from somewhere else, which today means the ADBC driver and the CLI's non interactive mode. DuckDB has had the setting for years. Document 10 specifies the surface and document 13 records the open question about what the default should be for `df.sql()` inside a user defined function.

## 7. The parts we are explicitly not building

**A catalog with persistence.** Names registered in a session, and the session dies with the process.

**A transaction manager.** No `BEGIN`, no isolation, no MVCC. A statement sees the frames as they are when it starts.

**A storage layer.** No native file format. Everything is read through the existing readers and written through the existing writers.

**An extension mechanism.** DuckDB v2.0 ships grammar hooks so extensions can add syntax. That is the right design for DuckDB and it is a promise about a stable interface that we are not making yet. Document 13 keeps it open.

**A second execution engine.** The physical planner emits the pipelines that already exist. If a SQL feature cannot be expressed as those pipelines, the answer is a new operator in `firepanda/exec/`, available to both surfaces, or a refusal.
