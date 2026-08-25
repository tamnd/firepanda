# firepanda

A dataframe engine for Mojo. Columnar, vectorized, GPU capable, and importable from Python.

`github.com/tamnd/firepanda`

Written 25 August 2026, against Mojo 1.0 (Modular 26.5, released 11 August 2026), pandas 3.0.5, Polars 1.43 and Arrow 25.0.0.

## What this is

pandas, rewritten so that the fast path is the language rather than a C extension. Arrow memory layout, kernels that vectorize because the type system vectorizes them, a lazy expression API with a real optimizer, parallel execution with no GIL in the way, and the whole thing importable from Python as `import firepanda as pd`.

The one line pitch is that it is the first dataframe library where the library, the kernels and the user's own hot loop are all the same language.

That is the thing pandas cannot do. In pandas the fast parts are C and Cython, the slow parts are Python, and the boundary between them is where every performance cliff in the library lives. `df.apply(lambda r: ...)` is 100x slower than `df["a"] + df["b"]` for reasons that have nothing to do with the operation and everything to do with which side of that boundary it landed on. In firepanda there is no boundary, because a user defined function written in Mojo is compiled into the same kernel pipeline as the built in ones.

## What this is not

It is not a database. No storage engine, no transactions, no server.

It is not a machine learning library. Hand the frame to one through Arrow, at zero copy.

It is not a Mojo wrapper around pandas. `Python.import_module("pandas")` already exists and is not interesting. Nothing in this design calls into pandas at runtime except the conformance suite, which calls into it on purpose.

It is not finished. These documents describe twelve milestones before there is a defensible 1.0, and document 08 says where the decision points are.

## Relationship to Spec/2125

`Spec/2125` is `kuma`, the same idea in Go, written the same day. The two specs share the parity checklist structure, the Arrow layout, the optimizer pass list and the milestone discipline, and they diverge everywhere the language forces them to.

Three divergences are worth knowing before reading further, because they are what make this a different project rather than a translation.

**kuma's centrepiece is compile time column names. firepanda's is compile time dtypes.** Go 1.27 generic methods let kuma make a typo in a column name a compile error. Mojo's parameter system lets firepanda make every kernel monomorphic in its dtype with no dispatch cost at all. Different language, different thing worth being smug about. Document 03.

**kuma needed a dedicated SIMD milestone. firepanda does not have one.** In Go, a vectorized kernel is three implementations behind a build tagged dispatch table plus a runtime feature check, and the tail handling is where the bugs live. In Mojo it is one generic function and `vectorize` writes the tail. That milestone dissolves into M1. It is the single largest structural difference between the two specs.

**kuma put its Python bindings at B2, after the engine. firepanda puts them at M3, before the lazy engine.** kuma's audience is Go programmers who already have Go. firepanda's audience is Python programmers who have never installed a Mojo toolchain and are not going to. The front door has to exist before there is anything impressive behind it.

## The documents

| | | |
|---|---|---|
| 00 | this file | index, decisions, prerequisites |
| 01 | `01-research-2026.md` | the state of Mojo, pandas, Polars, Arrow and the Mojo ecosystem, and what each fact forces |
| 02 | `02-architecture.md` | layers, memory layout, types, plan, optimizer, execution |
| 03 | `03-dtype-dispatch.md` | comptime monomorphization and the generated dispatch table, the centrepiece |
| 04 | `04-python-dx.md` | the two front doors, and what "same DX as pandas" is allowed to mean |
| 05 | `05-kernels.md` | the kernel layer, `vectorize`, `parallelize`, the hash table, the GPU path |
| 06 | `06-pandas-parity.md` | the conformance checklist, every pandas feature, one to one |
| 07 | `07-python-bindings.md` | `PythonModuleBuilder`, Arrow PyCapsule, wheels, and the ABI problem |
| 08 | `08-milestones.md` | M0 through M11, exit criteria, the four places to stop |
| 09 | `09-quality-bar.md` | what stdlib quality means in a language with no race detector |
| 10 | `10-benchmarks.md` | `tamnd/firepanda-bench`, against pandas, Polars, DuckDB, cuDF and MojoFrame |
| 11 | `11-package-layout.md` | the tree, Mojo 1.0's re-export rules, stability tiers |

Read 04 first if you only read one, because the product is the developer experience and everything else is in service of it. Read 03 second, because it is where the engine differs from every other dataframe library.

## The decisions already made

Everything below is settled. Where a decision has a cost, the cost is named.

**Two front doors, one API.** The same query reads identically in Mojo and in Python, and there is exactly one implementation. Python gets a compiled extension module built with `PythonModuleBuilder`, not a JIT import hook and not a subprocess. Costs a binding layer that has to be maintained in lockstep with every public API change. Buys the entire audience. Document 04.

**Arrow layout, our own kernels, C ABI bridge.** There is no Arrow implementation in Mojo, so the layout is ours to build. The bridge is the Arrow C Data Interface over `abi("C")` function types, which is how we get zero copy interchange with pandas, Polars, DuckDB and pyarrow without writing any of them. Document 01 section 5.

**Kernels are comptime parameterized over `DType` and dispatched through a generated table.** No trait objects, no boxing, no vtable. A runtime dtype tag is an integer compare against a `comptime for` unrolled candidate list, followed by a `rebind` to the concrete type. Document 03.

**Never `Dict`, never `List[String]`.** The MojoFrame paper's two named findings are that Mojo's dictionary is unoptimized and that individual `String` objects carry per object overhead. Both are load bearing for a dataframe. We write our own open addressing hash table and we store strings as Arrow StringView in raw buffers. Document 05.

**StringView is the default string representation.** A 16 byte view with an inline prefix, so most comparisons never touch the heap. Same call as kuma made, for the same reasons, and additionally because it is the only representation that avoids the `String` overhead the MojoFrame authors measured.

**Copy-on-Write is free and therefore mandatory.** pandas spent a decade and a major version arriving at Copy-on-Write semantics. Mojo's ownership model gives us the same semantics as a compile time property with no reference counting and no defensive copying. Frames are values. There is no `inplace=`.

**No automatic index alignment.** An index is optional and explicit. When you set one you get `loc`, `reindex`, `align` and the rest. What never happens is two frames silently aligning during arithmetic, which is the actual footgun rather than the index itself. Same position as kuma, arrived at the same way.

**Every pandas feature is covered, in a shape a pandas user recognizes.** Document 06 is a tickable checklist of the entire pandas 3.0 surface. There are exactly seven genuine omissions and each one is listed with what to use instead.

**GPU is a backend, not a product.** The kernel layer is written so that the same source can target `max.gpu`, and the operator interface holds that open from M5. It ships at M9 and not before, and document 10 is honest that cuDF got there first and is mature.

**Linux and macOS. Windows through WSL.** This is Mojo's platform coverage, not a choice we get to make, and it goes on the README rather than being discovered at install time.

**Two repositories.** `firepanda` for the library including the Python module, `firepanda-bench` for the comparisons and the conformance suite. The split is because the benchmark repository has to install pandas, Polars, DuckDB and Docker, and that must never become a condition of building the library.

## Prerequisites

Mojo 1.0 or newer, which means Modular 26.5 or newer. Anything on the 25.x line will not compile a single file in this project, because 1.0 renamed enough of the standard library that the diff is not worth maintaining a compatibility shim for. Document 01 section 1 lists the specific renames.

`pixi` for dependency management, with the `https://conda.modular.com/max`, `https://repo.prefix.dev/modular-community` and `conda-forge` channels. `pixi.toml` with a `[workspace]` table, not the legacy `mojoproject.toml`.

`pixi-build-mojo` as the build backend for publishing to the community channel.

A pinned toolchain. Mojo's source compatibility is guaranteed within 1.x but its **ABI is not stable**, so `pixi.lock` is committed and the CI matrix tests against the pinned version plus nightly. This is a real constraint on shipping binary artifacts and document 07 is about nothing else.

For `firepanda-bench` and the conformance suite: Python 3.13 or newer, pandas 3.0.5, Polars 1.43, DuckDB, pyarrow 25, and Docker. None of this is needed to build or test the library, which is the entire reason it lives in another repository.

For the GPU path at M9: an NVIDIA or AMD accelerator. `sys.has_accelerator()` gates every test in that directory and the whole library builds and passes without one.

## A note on the name

Fire for Mojo, panda for the API it replaces. It says what the project is in one word to somebody who has never heard of it, which is the entire job of a library name.

`pypi.org/project/firepanda` is unregistered as of today, checked directly against the JSON API rather than by search. `github.com/tamnd/firepanda` is free. There is a GitHub user account at `github.com/firepanda`, which takes the organization name but not the repository path and is not a conflict in any context that matters.

`import firepanda as pd` is the intended Python spelling and it is what document 04 is built around. `fpd` is taken on PyPI by an unrelated package, so the short alias is not available as an install name and should not appear in documentation as though it were.

The name is the cheapest thing in this specification to change and does not deserve more thought than this section.

## Honesty about scope

These documents describe a large project, and three parts of it are genuinely hard.

The optimizer at M4 and the parallel executor at M5 are hard for the same reasons they are hard in any engine, and the estimates there are the least trustworthy.

The Python packaging story at M3 is hard for a reason specific to this project: Mojo's ABI is unstable, calling Mojo from Python is documented by Modular as being in early development, and between them they mean the front door rests on the two least settled things in the toolchain. If that turns out to be unworkable the project does not have an audience, which is exactly why it is at M3 and not at M9.

The part that looks hard and is not is the kernel layer. In Go this was an entire milestone of build tags, feature detection and tail bugs. In Mojo it is generic functions over `DType`, and the compiler does the part that used to be dangerous.

The part most likely to be wrong is the assumption that a `comptime for` over fifteen dtypes across eighty kernels produces a binary and a compile time that anyone will tolerate. Document 03 section 6 says what to measure and when to fall back to a narrower dtype set.

The four checkpoints where stopping is a reasonable answer are after M2, after M3, after M5 and after M6, and document 08 says what question each one asks.
