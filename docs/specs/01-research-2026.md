# Landscape research, August 2026

Everything here was checked against primary sources on 25 August 2026. Links are at the bottom.

The point of this document is not to survey the field. It is that about a dozen specific facts about Mojo 1.0, pandas 3.0, Polars, Arrow and the Mojo package ecosystem determine most of the architecture in the rest of these documents, and it is worth writing down what those facts are so that when one of them changes we know which decisions to revisit.

Where a claim came from a secondary source and could not be confirmed against Modular's own documentation, it is marked **[verify]**. Those are the ones to check against the toolchain you actually install, and there are more of them in this document than in the equivalent document for a ten year old language, which is itself a finding.

## 1. Mojo 1.0, released 11 August 2026

This is the release that makes the project worth starting, and it landed two weeks ago.

Mojo 1.0 shipped as part of Modular 26.5. Modular's framing is that the language has reached the point where changes during the 1.x line will be primarily additive, and where APIs explicitly marked stable follow semantic versioning. Nearly 200 contributors landed more than 1,100 pull requests changing over 200,000 lines of code in the run up to it.

The three year churn is what kept anyone from building a library this size in Mojo before now. That is the whole reason this specification is dated August 2026 rather than August 2025.

### What "stable" covers, and what it does not

The stability marker system is deliberately narrow at 1.0.

Fully stable: the traits `Deinitable`, `Movable`, `Copyable` and `ImplicitlyCopyable`.

Partially stable, meaning some APIs carry the marker and others do not: `Array`, `List`, `Span`, `String`, `Bool` and `Optional`.

Everything else, which includes `SIMD` beyond the basics, `Dict`, the `algorithm` package, the whole `max` package and all of the GPU surface, carries no guarantee at all.

**Source compatibility is guaranteed within 1.x. The ABI is not.**

What this means for firepanda:

The public API of the library must be expressible in terms of the stable subset wherever possible, and every use of an unstable stdlib API has to be behind our own wrapper so that a break costs an afternoon rather than a rewrite. Concretely: `Dict` is banned outright for other reasons anyway, the `algorithm` package is only reachable through `firepanda/kernel`, and no `max` package type appears in a public signature.

The unstable ABI is the single most consequential fact in this document for distribution, and document 07 exists because of it. A binary artifact built against 1.0.0 is not guaranteed to load against 1.1.0. That makes the Python wheel story a version pinning problem rather than a build problem.

### The 1.0 renames, which invalidate every code sample written before August

This matters more than it sounds like it should, because every Mojo tutorial, blog post and AI generated code sample in existence right now predates it.

| Old | New |
|---|---|
| `__del__()` | `__deinit__()` |
| `ImplicitlyDestructible` | `Deinitable` |
| `InlineArray` | `Array` |
| `StringSlice` | `StringSpan` |
| `size` parameter on arrays, SIMD and ranges | `length` |
| `read` argument convention | `imm` |
| `alias` | `comptime`, with `alias` still accepted |
| `@parameter if` and `@parameter for` | `comptime if` and `comptime for` **[verify]** |
| `std.gpu.*` | `max.gpu.*` |
| `SIMDSize` | `SIMDLength` |

Plus: all variable declarations use `var` and implicit declaration is deprecated; relative imports must use `from`; bare `**kwargs` requires a `var` prefix; methods declare `self` with type `Self`; `class`, `del`, `match` and `yield` are reserved and cannot name a free function; `Int` is now an alias for `Scalar[DType.int]` and integer literals materialize to it; `size_of()` returns allocation size rather than type size, which fixed a memory corruption bug for over aligned types in `List`.

The `match` reservation is worth noticing. It is reserved because pattern matching is on the roadmap and has not landed, which is section 1.4.

### Collections, pointers and memory safety

`List`, `Span` and `String` now bind element references to an **interior origin** rather than the whole collection origin, which means mutating the collection invalidates an outstanding element reference and the compiler now diagnoses the common cases, including the classic `List.append` invalidating a reference into the list.

Invalid contiguous slices exit the program rather than silently clamping. **Negative indexing is removed entirely.**

`Pointer` and `UnsafePointer` are unified into a single type, with unsafe operations prefixed `unsafe_` or requiring an `unsafe_` prefixed keyword argument: `unsafe_load()`, `unsafe_store()`, `[unsafe_offset=i]`.

`UnsafePointer` is non-null by design. The default null constructor and `__bool__()` are deprecated and it no longer conforms to `Defaultable` or `Boolable`. Nullability is expressed as `Optional[UnsafePointer[...]]`, which shares the pointer's layout with the null address as the `None` niche, so it stays zero overhead and FFI safe.

String iteration yields grapheme clusters by default, so `for c in s` gives what a user perceives as a character on screen.

What this means for firepanda:

The removal of negative indexing is a **direct user visible parity problem**, because `df.iloc[-1]` and `s[-3:]` are things pandas users type every day. The parity layer has to normalize negative indices itself at the API boundary, and document 06 flags every place this bites. It is not hard, it just has to be deliberate rather than inherited.

Interior origins are exactly right for a column library and we should lean on them. A `Span` over a chunk's data buffer that is invalidated by an append to that chunk is a bug the compiler now catches, and that is a class of bug the Go version had to test for.

Grapheme cluster iteration is correct for a text library and wrong for a dataframe kernel, where the hot loop wants bytes. Every string kernel operates on `Span[UInt8]` over the StringView buffer and never iterates a `String`.

`Optional[UnsafePointer[T]]` being layout compatible with a nullable C pointer is what makes the Arrow C Data Interface implementable cleanly, which is section 5.

### What is not in 1.0

**No pattern matching and no unions.** Both are on the roadmap and neither has shipped. This is the constraint that shapes document 03, because the obvious way to represent "a column of one of fifteen dtypes" in a modern language is a sum type, and we do not have one. `Variant` from the standard library is a tagged union over a fixed type list and it is the closest thing available.

**No asynchronous programming model.** A robust one is on the roadmap. There is no async IO, so parallel IO is a thread pool over `parallelize`, and every operator interface in document 02 is synchronous with cancellation checked at morsel boundaries rather than at await points.

**The compiler and toolchain are not open source.** Modular has restated a commitment to open sourcing them during 2026 and it has not happened yet. The standard library is open; the compiler is not.

**Windows is not supported natively.** Linux and macOS, with Windows through WSL, and a native port described as a mid term project.

### The corporate fact

Qualcomm completed its acquisition of Modular on 29 July 2026 **[verify: sources give 28 and 29 July]**, at a valuation reported around 3.1 billion dollars.

What this means for firepanda: nothing technical today and everything strategic eventually. A single vendor language whose vendor was just acquired by a hardware company is a different risk profile than one whose vendor was an independent AI infrastructure startup. The open sourcing commitment matters more now, not less. This is a reason to keep the FFI boundary at `firepanda/ffi` narrow and documented, and it is a reason not to build anything on the `max` package that we could not replace, but it is not a reason to wait.

## 2. The Mojo facts that make a dataframe engine practical

Separating these out because they are the positive case, and the previous section is mostly constraints.

### `DType` is a compile time value

`SIMD[dtype, length]` is the core numeric type in the standard library and `DType` is a parameter, not a runtime tag. `Int` is `Scalar[DType.int]`, which is `SIMD[DType.int, 1]`. Every numeric type in the language is already the vector type at width one.

This is the single best thing about writing a dataframe engine in Mojo and document 03 is about exploiting it. A kernel written once as `fn sum[dt: DType](...)` compiles to a separate optimal implementation per dtype, with no dispatch, no boxing and no interface. In Go the equivalent is either code generation or `any` with a type switch in the inner loop.

### `vectorize` and `parallelize` are library functions, not compiler magic

The `algorithm` package provides `vectorize` for data parallel processing over a width and `parallelize` for task parallel execution across cores. `simdwidthof[dtype]()` gives the native vector width for the target.

`vectorize` writes the remainder loop. The number one source of silent corruption in hand written vector code is tail handling, and in this language it is not hand written.

What this means for firepanda: there is no SIMD milestone. Kernels are vectorized from the first one, on every architecture the compiler targets, with the tail handled by the standard library. The Go sibling spec allocates a whole milestone and a three way build tagged dispatch table to this. Here it is a `comptime` parameter.

The cost is that the portable path is now the only path. Where a hand written intrinsic sequence would beat the generic one, we do not have a mechanism as direct as Go's `archsimd` to reach for, and the answer is to specialize the generic function with `comptime if` on the target rather than to write assembly.

### There is no GIL

Real threads, real shared memory parallelism, in a language whose syntax a Python programmer can read. This is the thing pandas cannot fix without breaking every extension in the ecosystem, and it is the reason the parallel executor in document 02 can be simple.

Python's own answer, the free threaded build that became officially supported in 3.14, is a real development and it does not close this gap, because pandas' internals and every C extension around it are not thread safe in the way that would be required.

### GPU support is in the language

The `gpu` package, now `max.gpu`, exposes `DeviceContext` with device allocation, host to device transfer and `enqueue_function()` for compiling and launching a kernel written in Mojo. `sys.has_accelerator()` reports whether one is present. Execution is stream ordered and asynchronous.

The relevant detail for us: `Int` and `UInt` are no longer `DevicePassable` as of 1.0, so kernel signatures use fixed width types. That is a small thing that will otherwise be discovered as a confusing error message at M9.

What this means for firepanda: a GPU backend is a backend rather than a separate product, because the kernel source is the same language. That is genuinely novel for a dataframe library. It is also M9 and not earlier, for the reasons in document 08.

### Python interop got 12x faster on the hot path

Mojo 1.0 rewrote `PythonObject` operator dispatch to go through the CPython abstract protocols, `PyNumber_Add`, `PyObject_RichCompare` and `PySequence_Contains`, rather than through attribute lookup, reported at roughly 12x on the interop hot path. A `std.python.numpy` module was added with `copy_to_numpy_array()` and `from_numpy_array()`.

What this means for firepanda: the conformance suite in document 09 can run pandas **in the same process** as firepanda and compare results directly, with no serialization and no subprocess. In the Go sibling spec this required an entire bindings milestone to exist first. Here it is available from M1, which moves differential testing against pandas from a late luxury to an early default.

## 3. Calling Mojo from Python, the mechanism the product depends on

Python→Mojo bindings arrived as a preview in the 25.5 release in August 2025 and Modular's documentation still describes calling Mojo from Python as **in early development, with a lot of expected change to the API and ergonomics**.

The mechanism is the ordinary CPython extension module protocol. A dynamic library exporting `PyInit_<modulename>()` is what Python loads. On the Mojo side, an `@export fn PyInit_firepanda()` constructs a `PythonModuleBuilder`, registers functions with `def_function[f]("name", docstring=...)`, registers types with `add_type[T]("Name")` and `PythonTypeBuilder`, and returns `finalize()`. Keyword arguments are supported through `OwnedKwargsDict[PythonObject]`.

The import path for the builder has moved. Older documentation says `python.bindings`, newer says `std.python.bindings`. **[verify against the installed toolchain.]**

On the Python side, the development workflow imports `max.mojo.importer`, puts the directory on `sys.path` and imports the `.mojo` file directly, which JIT compiles it.

What this means for firepanda, and there are four things:

**The JIT import hook is a development convenience and must not be the distribution mechanism.** Requiring an end user to have a Mojo toolchain installed to `import firepanda` would eliminate the entire audience. We build a shared library exporting `PyInit_firepanda` and ship it as an ordinary wheel. Document 07.

**The binding API is the least stable thing in the stack and it is load bearing.** This is a stated, documented instability underneath the product's main feature. The mitigation is that the binding layer is generated from a single declarative table rather than hand written per function, so an API change is one file. Document 07 section 3.

**Everything crossing the boundary should cross as Arrow, not as `PythonObject`.** Per element `PythonObject` traffic is the thing that makes pandas slow and it would make firepanda slow for exactly the same reason. Data crosses once, as an Arrow C stream through the PyCapsule protocol. Only the plan and the scalars cross as objects.

**A published benchmark of the binding path reported roughly 2x over already fast NumPy and 12x over pure Python**, which is a useful calibration against headline pure Mojo numbers. Crossing the boundary costs something and the design should minimize the number of crossings rather than assume they are free.

## 4. pandas 3.0, released 21 January 2026

The most important finding for scoping, because it changes what parity means.

3.0.5 is the current stable release, from 22 July 2026. Python 3.11 or newer is required.

**Copy-on-Write is the default and the only mode.** The decade old ambiguity about views and copies is gone. `df[cond]["col"] = value` raises `ChainedAssignmentError`. Reported at roughly 1.3x on operations that previously required defensive copying.

**The PyArrow backed string dtype is the default.** String columns infer as a native `str` dtype rather than `object`. Reported 5x to 10x on string operations and up to 50 percent less memory on text heavy columns. pyarrow remains an optional dependency, with a non Arrow fallback, contrary to several blog posts.

**`pd.col()` expressions exist.** pandas has an expression builder and column operations no longer require lambdas.

**Datetime data is microsecond resolution by default**, expanding the representable date range.

What this means for firepanda:

Parity is more coherent than it would have been a year ago, because pandas has moved toward the immutable, columnar, expression oriented model this spec was going to build anyway. `pd.col()` maps directly onto `firepanda.col()`, and the lazy API in document 02 stops being a foreign concept bolted onto a pandas shaped surface.

Copy-on-Write means our value semantics are not a divergence any more. They are the same semantics, obtained for free from the ownership model instead of from a reference counting scheme.

The checklist in document 06 targets 3.0. Several 2.x behaviours, meaning `inplace=`, chained assignment, `object` dtype strings and NaN as the missing value sentinel, are removed or deprecated upstream, which retroactively justifies not reproducing them.

The honest positioning changed. pandas is no longer slow in the way it was in 2023. What it still cannot do is run user code at native speed, use every core without a process pool, or tell you why a query is slow.

## 5. Arrow, and the fact that Mojo has none of it

Arrow 25.0.0 for C++ and Python was released on 10 July 2026.

The Arrow C Data Interface, meaning the `ArrowSchema`, `ArrowArray` and `ArrowArrayStream` structs, is the zero copy in process exchange standard. The PyCapsule interface, `__arrow_c_array__` and `__arrow_c_stream__`, is what pandas 3.0, Polars, DuckDB and pyarrow have all agreed on at the Python level.

**There is no Arrow implementation for Mojo.** No layout types, no IPC, no Parquet, no compute. This is the largest single difference in scope between this spec and the Go sibling, where `arrow-go` and `parquet-go` existed and the decision was how much of them to use.

What this means for firepanda, and there are three consequences.

**The layout is ours to write, which is a cost and an advantage.** The cost is M0 and part of M2. The advantage is that there is no existing implementation whose design decisions we have to work around, and no `Retain`/`Release` discipline inherited from a C++ port.

**The C Data Interface is the bridge and it is buildable today.** Mojo 1.0 added `abi("C")` as a function effect declaring the platform C calling convention for struct arguments and returns, and `DLHandle.get_function()` now enforces that the type parameter carries it. Combined with `Optional[UnsafePointer[T]]` for nullable C pointers and `MutExternalOrigin`/`ImmutExternalOrigin` for foreign memory, the three Arrow structs are expressible correctly rather than approximately. Before `abi("C")` this would have been guesswork about struct passing.

**Parquet is the biggest unbudgeted item in the plan.** Nothing exists. The options are to write a reader in Mojo, which is thrift metadata plus a dictionary, RLE, bit packing and delta decoder stack plus snappy, zstd, gzip and lz4, or to bind Arrow C++ through the C Data Interface and read Parquet through it. Document 08 takes the second option first and the first option later, because a bound reader at M2 unblocks everything and a native reader is a performance project rather than a capability project.

## 6. Polars, the performance bar

Polars is the reference implementation of the design firepanda is pursuing and it is what we measure against.

py-1.43.0 shipped on 21 July 2026 with Hive partition awareness as the headline: group bys and joins organize work around partition keys, and inner joins can be rewritten as a union of partition filtered joins. It also added `POLARS_OOC_DISK_BUDGET_MB` for out of core disk budgeting.

The streaming engine is morsel driven and pull based, with spillable sinks for joins, group bys and sorts. It is opt in through `pl.Config.set_engine_affinity("streaming")`, operators without a streaming implementation fall back transparently, and all major formats including CSV now have streaming scans. 1.37 added the new sink pipeline, 1.38 a streaming merge join, 1.39 a streaming as of join.

Polars 2.0 has not shipped. The roadmap issue notes 45 blocking issues and leaves open whether streaming becomes the default.

Also in the 2026 releases: a categoricals overhaul, stabilized `Decimal` and `Int128`, and a runtime selection mechanism where Polars loads the most conservative runtime available and alternatives install as extras such as `polars[rtcompat]`.

What this means for firepanda:

The three year arc says the ordering is a correct in memory vectorized engine first and streaming second. Streaming is M10 and nothing earlier depends on it.

Transparent per operator fallback is the pattern that lets an incomplete streaming engine ship at all. Copy it.

Partition aware planning is a large real world win and cheap to design for early, so scan nodes carry Hive partition metadata from M2 even though the optimizer ignores it until M10.

`Int128` and stabilized decimals set the type system bar. Decimal128 is not optional.

## 7. The Mojo dataframe and library ecosystem

### MojoFrame is the prior art and it is a research prototype

Shengya Huang, Zhaoheng Li, Derek Warner and Yongjoo Park, arXiv 2505.04080, submitted May 2025, revised May 2026, accepted to ICDE 2026.

It is the first native dataframe library for Mojo. It supports all 22 TPC-H queries and a selection of TPC-DS queries and reports up to 4.60x over existing dataframe libraries in other languages, with the wins concentrated in UDF heavy queries and low cardinality group by aggregation.

The design uses Mojo's tensor types for numeric columns and a cardinality aware approach for non numeric data, factorizing non numeric join keys into a shared integer space and then running an integer hash join.

Three findings from that paper are load bearing for this spec.

**Mojo's native dictionary is unoptimized**, and it is why MojoFrame falls short on high cardinality aggregation such as TPC-H Q18. We write our own hash table. This is not an optimization to do later; the group by and the join are built on it from M1.

**There is no native string alternative**, so they used individual `String` objects at 20 bytes of overhead each. Better than pandas' 49 bytes of Python string metadata, and still catastrophic for a hundred million row string column. Arrow StringView in raw buffers, from M0, no exceptions.

**The wins are where the UDF boundary is.** Their largest advantages are on UDF heavy queries, which is exactly the argument in document 00 about pandas' C-to-Python boundary. That is empirical support for the positioning rather than a claim we are making unaided.

MojoFrame is also a benchmark baseline. A production library that cannot beat a research prototype has not justified itself, and document 10 puts it in the comparison table.

### The community ecosystem is thin and 1.0 will reshape it

`NuMojo` provides an N-dimensional array type and numerical algorithms, roughly NumPy shaped. `ExtraMojo` fills standard library gaps with explicitly unstable APIs, ordering itself correctness, then performance, then ergonomics. `EmberJson` is the de facto JSON library. `Mojmelo` is machine learning algorithms in pure Mojo. `DeciMojo` covers decimals.

`lightbug_http`, the flagship HTTP framework, was **archived on 12 May 2026** and continues as a maintained fork. Several recent write ups still cite it as active, which is a good calibration on how much secondary source material about this ecosystem is stale.

Packages are distributed as conda packages through the `modular-community` channel on prefix.dev, with recipes submitted as pull requests to `modular/modular-community`. `pixi` replaced Modular's own `magic` tool; `mojoproject.toml` is legacy and `pixi.toml` with a `[workspace]` table is current. `pixi-build-mojo` is the build backend and discovers the package from `<name>/__init__.mojo`.

What this means for firepanda:

**Depend on almost nothing.** The ecosystem is young and its most prominent library was archived three months ago. `EmberJson` is the one dependency worth considering, for NDJSON, and even that should be behind an interface so it can be replaced. Everything else is ours.

The flip side is that there is no incumbent. There is no Mojo dataframe anybody is using in production, which means the bar for v0.1 is usefulness rather than beating Polars, and ergonomics and correctness matter more than the last 30 percent of throughput.

Publishing to `modular-community` is a pull request to somebody else's repository, which is a slower release cadence than `go get`. Plan releases accordingly and keep the git install path working.

### cuDF exists and is the honest comparison for the GPU story

RAPIDS cuDF is a mature GPU dataframe library with a pandas compatible API and a `cudf.pandas` accelerator mode. Any claim firepanda makes about GPU dataframes is a claim against cuDF, not against pandas, and document 10 benchmarks against it rather than pretending the category is empty.

Where firepanda can differ is that the CPU and GPU paths are the same source in the same language, so a user defined function written once runs in both places. cuDF cannot offer that, because a UDF in cuDF is either Numba compiled Python with its own restrictions or it is CUDA C++.

## 8. Benchmarking, briefly

The H2O.ai db-benchmark went dormant in July 2021 and DuckDB Labs revived it. The live suite is `duckdblabs/db-benchmark`, published at `duckdblabs.github.io/db-benchmark`. The frozen `h2oai.github.io` page must not be used for comparison.

Ten group by and five join queries at 0.5 GB, 5 GB and 50 GB, on a `c6id.metal` with 250 GB of RAM and 128 cores.

One methodological note matters more than the rest: ClickHouse and DuckDB use `CREATE TABLE ans AS SELECT` so that lazy engines are forced to materialize. A lazy engine benchmarked without materializing measures nothing. Our harness does the same.

Full treatment is document 10.

## 9. Consolidated design consequences

| Finding | Consequence |
|---|---|
| Mojo 1.0 stabilizes source but not ABI | Wheels pin a toolchain version. The CI matrix tests pinned plus nightly. Document 07 exists for this. |
| `DType` is a compile time parameter | Kernels are monomorphic per dtype with zero dispatch cost. Document 03. |
| No pattern matching, no unions | Heterogeneous columns are a runtime tag plus a `comptime for` generated dispatch table plus `rebind`, not a sum type. |
| `vectorize` writes the tail | No SIMD milestone. Kernels vectorize from M1. The largest schedule difference from the Go sibling spec. |
| No GIL | The parallel executor is ordinary shared memory parallelism. |
| `algorithm`, `Dict`, `max` are unstable | All reachable only through our own wrappers. No unstable type in a public signature. |
| Negative indexing removed from `List`, `Span`, `String` | The parity layer normalizes negative indices itself. `iloc[-1]` must keep working. |
| String iteration yields graphemes | Kernels operate on `Span[UInt8]`, never on `String`. |
| `abi("C")` and `Optional[UnsafePointer[T]]` exist | The Arrow C Data Interface is implementable correctly. It is the only data path across every boundary. |
| No Arrow, no Parquet in Mojo | Write the layout. Bind Arrow C++ for Parquet at M2, write a native reader later. |
| Python interop is 12x faster and in-process | Differential testing against pandas 3.0 is available from M1, not after a bindings milestone. |
| Calling Mojo from Python is "early development" | The binding layer is generated from one declarative table so a breaking change is one file. |
| MojoFrame: `Dict` is unoptimized | Our own open addressing hash table, from M1, for group by and join. |
| MojoFrame: `String` costs 20 bytes each | Arrow StringView in raw buffers, from M0. |
| MojoFrame wins are UDF heavy queries | The pitch is the absent C-to-Python boundary, and there is a paper supporting it. |
| pandas 3.0 is CoW, Arrow strings, expressions | Parity targets 3.0. Our value semantics are no longer a divergence. |
| Polars ordering was in memory then streaming | Streaming is M10. |
| Polars added Hive partition awareness | Scan nodes carry partition metadata from M2. |
| Polars stabilized `Decimal` and `Int128` | Decimal128 is in the type lattice at M0. |
| `lightbug_http` was archived in May 2026 | Depend on almost nothing. Assume secondary sources about this ecosystem are stale. |
| cuDF is mature | Benchmark the GPU path against cuDF, not against pandas. |
| db-benchmark moved to DuckDB Labs | Use the current suite, force materialization, gate in CI from M1. |
| Modular acquired by Qualcomm, compiler still closed | Keep the FFI boundary narrow. Not a reason to wait. |
| Linux and macOS only | Say so on the README. |

## Sources

**Mojo 1.0 and Modular 26.5**

- Modular 26.5: Mojo 1.0 is here, https://www.modular.com/blog/modular-26-5-mojo-1-0-is-here
- Mojo v1.0.0 release notes, https://mojolang.org/releases/v1.0.0/
- Mojo release notes and changelog, https://mojolang.org/releases/
- Mojo standard library index, https://mojolang.org/docs/std/
- Mojo (programming language), Wikipedia, https://en.wikipedia.org/wiki/Mojo_(programming_language)
- Mojo Hits 1.0: A Technical Look, https://dev.to/shresthapandey/mojo-hits-10-a-technical-look-50ga
- Mojo 1.0 Officially Arrives with Stability Guarantees, XenoSpectrum, https://xenospectrum.com/en/mojo-1-0-stability-open-source/

**Language features**

- Parameterization, https://mojolang.org/docs/manual/parameters/
- Compile-time evaluation, https://mojolang.org/docs/manual/metaprogramming/comptime-evaluation/
- Proposal: replace `@parameter` with a `comptime` statement modifier, https://forum.modular.com/t/proposal-replace-parameter-with-comptime-statement-modifier/2713
- RFC: rename the `alias` keyword, https://github.com/modular/modular/issues/171
- `simd` builtin module, https://mojolang.org/docs/std/builtin/simd/
- Using pointers, https://mojolang.org/docs/manual/pointers/using-pointers/
- UnsafePointer v2 proposal, https://github.com/modular/modular/blob/main/Mojo/proposals/unsafe-pointer-v2.md
- Trait objects and dynamic dispatch discussion, https://github.com/modular/modular/discussions/3169
- liuzhishan/mojo-dynamic-dispatch, https://github.com/liuzhishan/mojo-dynamic-dispatch

**GPU**

Note on where these live. When Mojo was open sourced the documentation split in
two: the language, manual and standard library moved to `mojolang.org`, and the
GPU host API moved to `max.modular.com`. That is not only a URL change. The
kernel-side primitives (`std.gpu`, `thread_idx`, `sys.info.has_accelerator`) are
Mojo standard library and covered by the 1.x source compatibility promise; the
host side that firepanda would need to launch anything (`max.gpu.host`,
`DeviceContext`) ships with MAX and carries no such promise. M9 therefore takes a
dependency on MAX, not just on the Mojo toolchain, and the GPU backend has to be
an optional build.

- GPU programming fundamentals, https://max.modular.com/gpu/fundamentals/
- Get started with GPU programming, https://max.modular.com/gpu/intro-tutorial/
- `DeviceContext` API, https://max.modular.com/api/mojo/max/gpu/host/device_context/DeviceContext/
- Build custom ops for GPUs, https://max.modular.com/develop/build-custom-ops/

**Python interop**

- Calling Mojo from Python, https://mojolang.org/docs/manual/python/mojo-from-python/
- Calling Python from Mojo, https://mojolang.org/docs/manual/python/python-from-mojo/
- Python Can Now Call Mojo, Towards Data Science, https://towardsdatascience.com/python-can-now-call-mojo/
- Giving Mojo a spin, koaning.io, https://koaning.io/posts/giving-mojo-a-spin/
- Free threading in Python, https://docs.python.org/3/howto/free-threading-python.html

**Packaging**

- Pixi basics for Mojo, https://mojolang.org/docs/pixi/
- pixi-build-mojo, https://pixi.prefix.dev/latest/build/backends/pixi-build-mojo/
- modular/modular-community, https://github.com/modular/modular-community
- prefix-dev/pixi PR 3942, legacy `mojoproject.toml` support, https://github.com/prefix-dev/pixi/pull/3942

**pandas**

- pandas 3.0 released, https://pandas.pydata.org/community/blog/pandas-3.0.html
- What's new in 3.0.0, https://pandas.pydata.org/docs/whatsnew/v3.0.0.html

**Polars**

- Polars in Aggregate, April 2026, https://pola.rs/posts/polars-in-aggregate-apr26/
- Polars in Aggregate, December 2025, https://pola.rs/posts/polars-in-aggregate-dec25/
- pola-rs/polars issue 26148, the 2.0 roadmap, https://github.com/pola-rs/polars/issues/26148
- pola-rs/polars releases, https://github.com/pola-rs/polars/releases

**Arrow**

- Apache Arrow 25.0.0 release, https://arrow.apache.org/blog/2026/07/10/25.0.0-release/
- The Arrow C Data Interface, https://arrow.apache.org/docs/format/CDataInterface.html
- The Arrow PyCapsule interface, https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

**Mojo dataframes and ecosystem**

- MojoFrame: Dataframe Library in Mojo Language, arXiv 2505.04080, https://arxiv.org/abs/2505.04080
- modular/modular discussion 1446, the Mojo way of replacing pandas, https://github.com/modular/modular/discussions/1446
- mojicians/awesome-mojo, https://github.com/mojicians/awesome-mojo
- ExtraMojo, https://extramojo.github.io/ExtraMojo/
- Lightbug-HQ/lightbug_http, archived, https://github.com/Lightbug-HQ/lightbug_http

**Benchmarks**

- duckdblabs/db-benchmark, https://github.com/duckdblabs/db-benchmark
- The Return of the H2O.ai Database-like Ops Benchmark, https://duckdb.org/2023/04/14/h2oai
- RAPIDS cuDF, https://github.com/rapidsai/cudf
