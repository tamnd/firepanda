# Architecture

## Layers

Five of them, with dependencies pointing strictly downward.

```
L4  Python        the extension module, PyCapsule, wheels        document 07
L3  Eager         DataFrame, Series, the pandas shaped surface   document 04
L2  Lazy          LazyFrame, Expr, plan, optimizer               the engine
L1  Storage       Buffer, Bitmap, Array, ChunkedArray, DType     Arrow layout
    Kernels       comptime parameterized over DType              document 03
L0  Runtime       morsel scheduler, hash table, memory manager
```

L3 is a facade over L2 and adds no capability the lazy layer does not have. `df.groupby("k").sum()` builds a plan, optimizes it and collects it, and the only difference from the lazy path is that collection is implicit. That is deliberate: it means the pandas shaped surface can be reshaped, or a second surface added, without the engine noticing.

L4 is a facade over L3 and adds nothing at all. If a method exists in Python and not in Mojo, that is a bug in the binding table, and document 07 has a test that asserts the two surfaces match.

The important property of this stack is that a user's own code can enter at any layer. A pandas user enters at L4. A Mojo user writing a fast path enters at L3 or L2. Somebody writing a custom kernel enters at L1 and gets the same monomorphization the built in kernels get. There is no layer at which the language changes, and that is the whole product.

## Memory

Arrow columnar layout, ours to build, because there is no Arrow implementation for Mojo.

```
Array[dt: DType]
  length, null_count       null_count == 0 is the hot path
  validity  Bitmap         one bit per value, 1 means valid, absent when null_count is 0
  data      Buffer         primitive values, or offsets, or 16 byte views
  children  List[AnyArray] for list, struct and map
```

`Array` is parameterized on its dtype. That is the central decision of this design and document 03 is about its consequences. A `Array[DType.float64]` and a `Array[DType.int32]` are different types with different generated code, and neither carries a runtime type tag inside its kernels.

`AnyArray` is the type erased form, holding a runtime `DType` tag plus the untyped buffers. It exists because a `DataFrame` is a list of columns of different dtypes and Mojo has no sum type. Converting from `AnyArray` back to `Array[dt]` is an integer compare and a `rebind`, and it happens once per kernel invocation, not once per element. Document 03 section 3.

A `Buffer` owns a 64 byte aligned allocation. Alignment is not free, since the allocator guarantees less than that, so an aligned buffer over allocates and offsets. It is worth paying for because aligned loads matter on AVX-512 and because it keeps every kernel from straddling a cache line at the boundary.

There is no reference counting. Buffers are owned values under Mojo's ownership model, and the `__deinit__` runs when the owner goes out of scope. Arrow's `Retain`/`Release` discipline is a C++ artifact and we are not importing it. Where a buffer genuinely needs shared ownership, which is the Arrow C Data Interface boundary and the scratch pool, that is explicit and local.

### Value semantics and the thing pandas took ten years to get

Frames are values. `df2 = df.filter(...)` produces a new frame and `df` is unchanged. Under the ownership model, the compiler knows whether `df` is still live and elides the copy when it is not.

This is Copy-on-Write, which pandas made the only mode in 3.0 after a decade of ambiguity, obtained as a compile time property rather than a runtime scheme. There is no reference count to consult, no defensive copy to skip, and no `ChainedAssignmentError` to raise, because the chained assignment does not compile in Mojo and raises at the Python boundary.

The cost is that Python users get one behaviour and Mojo users get a stricter version of it. Document 04 section 6 says exactly where the two diverge.

### Strings

The default representation for `String` and `Binary` columns is the Arrow variable size binary view layout. Each element is a 16 byte record: a 4 byte length, a 4 byte inline prefix, and then either the remaining 8 bytes inline for values of 12 bytes or fewer, or a buffer index and offset pair for longer ones.

This is the default rather than an option, for three reasons, and the third is specific to Mojo.

Equality and prefix comparison resolve from the inline prefix alone for most real data, which means no pointer chase and no cache miss on the common path.

Filtering and sorting become dense scans over fixed width 16 byte records, and fixed width scans vectorize.

And the alternative is catastrophic here in a way it is not in other languages. The MojoFrame authors stored non numeric columns as individual `String` objects and measured 20 bytes of overhead per element. Better than pandas' 49 bytes of Python string metadata and still a hundred million row string column paying two gigabytes for nothing. **`List[String]` never appears in a column representation.** The type is not permitted in `firepanda/array`.

String kernels operate on `Span[UInt8]` over the buffer. They never iterate a `String`, partly for speed and partly because Mojo 1.0 iterates strings by grapheme cluster, which is right for text and wrong for a scan.

The classic `Utf8` layout of offsets plus data is kept for IPC compatibility and converted at the boundary. Low cardinality columns additionally get dictionary encoding, which turns string group by keys into integer keys and skips hashing entirely. That is usually a larger win than any amount of vectorization applied to the string path, and it is the same trick MojoFrame's cardinality aware design uses for join keys.

### Chunking

A column is a `ChunkedArray`, an ordered list of `Array` chunks sharing a dtype.

The chunk is four things at once. It is the unit of vectorization, since a kernel operates on one chunk. It is the unit of parallelism, since a morsel is one chunk or a slice of one. It is the unit of the null fast path, since `null_count` is per chunk. And it is the unit of append, so growth never reallocates a column.

Target size is whatever keeps the working set in L2, somewhere between 1024 and 8192 rows depending on row width. The concrete default is determined by benchmark at M5 rather than guessed now.

## Types

```
Primitive   Bool Int8 Int16 Int32 Int64 Int128 UInt8 UInt16 UInt32 UInt64
            Float16 Float32 Float64
Decimal     Decimal128(precision, scale)
Temporal    Date32 Date64 Time32(u) Time64(u) Timestamp(u, tz) Duration(u) Interval
Binary      String Binary in view layout, LargeString LargeBinary, FixedSizeBinary(n)
Nested      List(T) LargeList(T) FixedSizeList(T, n) Struct(fields) Map(K, V)
Encoded     Dictionary(index, value)
Null
```

Mojo's `DType` covers the primitives directly, which is the reason the kernel design works. The types it does not cover, meaning decimals, temporals with a unit and a timezone, the binary layouts and everything nested, get a `LogicalType` descriptor that carries the parameters and names an underlying physical `DType`.

That split is worth being explicit about because it is easy to get wrong. A `Timestamp(us, "Europe/London")` column is physically `DType.int64` and every arithmetic kernel operates on the physical type. The timezone lives in the schema and is consulted by the operations that care, which is formatting, component extraction and DST aware arithmetic. There is exactly one place in the codebase where a timestamp is anything other than an int64, and that is `firepanda/temporal`.

`Field` is a name, a `LogicalType`, a nullable flag and metadata. `Schema` is an ordered list of fields plus a name to index map.

**Nulls are a validity bitmap and are distinct from NaN.** `NaN != NaN` and NaN is a valid float; null propagates. Aggregations skip nulls by default, matching pandas `skipna`, with an option to change it. Starting here avoids the entire family of bugs where an integer column silently becomes a float because something was missing, which is the single most common pandas surprise and the one pandas itself only escaped by adopting Arrow.

**There is no implicit upcasting.** Adding an int64 column to a float64 column is an error unless one side is a literal or the user writes a cast. pandas' silent upcasting is a correctness hazard and Polars' strictness is the right call. The error arrives at plan time, before any data is read.

This is a real divergence from pandas and it is the one users will complain about. Document 04 section 6 says what the error message has to contain to make it survivable, and document 06 marks every checklist row where it bites.

## Expressions

An expression wraps an immutable node.

```mojo
struct Expr(Copyable, Movable):
    var node: ArcPointer[Node]
```

Node kinds are `Column`, `Literal`, `Unary`, `Binary`, `Function`, `Cast`, `Ternary`, `Agg`, `Window`, `Sort`, `Alias`, `Wildcard`, `Exclude` and `DTypeSelector`.

Nodes are immutable and structurally shared. `a + b` allocates one node and reuses both operands, which makes common subexpression elimination a pointer identity check rather than a tree walk.

Errors are carried inside the node rather than raised at construction. A malformed expression poisons everything built on it and surfaces exactly once, at collection. This is what pays for a chaining API that does not force a `try` on every link, and it matters more in Mojo than in Go because Mojo's `raises` effect is viral and an expression builder that raises would infect every call site in the user's pipeline.

Expression building is the one genuinely allocation heavy path in the library.

**The operator set is where the DX is won.** Mojo supports `__add__`, `__lt__`, `__and__` and the rest on structs, so `col("price") > 100` is an expression and reads exactly like it does in Polars and in pandas 3.0's `pd.col()`. The one place this fails is `and` and `or`, which are keywords Mojo evaluates for truthiness and cannot be overloaded, so the API uses `&` and `|` with the same precedence trap Polars and pandas both have. We do not get to fix that, and we do get to produce a good error message when somebody writes `a > 1 and b < 2`, which document 04 requires.

## Plan and optimizer

```
LogicalPlan := Scan | Filter | Project | Aggregate | Join | Sort | Limit | Distinct
             | Union | Explode | Pivot | Unpivot | Window | Sink
```

The passes below run to fixpoint.

| Pass | Effect |
|---|---|
| Projection pushdown | The scan reads only the columns actually referenced. The single biggest Parquet win and usually the largest speedup in the whole optimizer. |
| Predicate pushdown | Filters sink to the scan, enabling Parquet row group and bloom filter skipping. |
| Slice pushdown | `head(n)` bounds how much the scan decodes. |
| Common subexpression elimination | A pointer identity check over shared nodes. |
| Expression simplification | Constant folding, `x & true`, De Morgan normalization so more predicates become pushable. |
| Type coercion | Explicit cast insertion, failing at plan time rather than partway through the data. |
| Join reordering | Cardinality driven, so the smaller side builds the hash table. |
| Partition pruning | On Hive partitioned scans, prune by path and pre partition group bys and joins on the partition keys. Metadata carried from M2, pass lands at M10. |
| Aggregate pushdown | A count over a Parquet scan is answered from file metadata without reading anything. |
| Kernel fusion | Adjacent elementwise expressions compile into one pass over the chunk. |
| Dictionary preservation | A group by on a dictionary encoded column groups on the codes and decodes once at the end. |

`explain()` prints the optimized plan and `profile()` adds per operator wall time, rows in and out, and bytes read.

These are shipping features rather than debugging tools and they exist from M4. Two reasons and both matter. Users need to see why a query is slow, and pandas gives them nothing at all here, so it is one of the larger DX advantages available. And internally, every optimization we claim becomes a test that asserts on the plan shape, rather than a claim in a commit message that nobody can check.

### Kernel fusion is different in this language

In an interpreted or dynamically dispatched engine, fusing `(a + b) * c` into one pass is an optimizer pass that builds a fused operator. Here, the expression tree can be lowered into a `comptime` recursive evaluator that the compiler inlines into a single loop body with no intermediate buffers, for the subset of expressions whose shape is known at compile time.

That subset is exactly the expressions a Mojo user writes literally in source, and it is not the expressions a Python user builds at runtime. So there are two paths: a compile time fused path for statically known expressions, and a runtime interpreted-over-vectorized-chunks path for everything else. The second is what a Python user gets and it is what Polars does; the first is a genuine advantage available only to the Mojo surface, and it is the reason a Mojo user's hot loop can be faster than the same query expressed through the Python front door.

This is designed at M4 and built at M8. It is called out here because the expression representation has to support it, meaning nodes have to be walkable at compile time when they were built at compile time, and retrofitting that is a rewrite.

## Execution

Vectorized and morsel driven. Operators process one chunk at a time. A scheduler hands morsels to a fixed worker pool sized to the physical core count, with work stealing.

```
Scan --> [morsel queue] --> worker 0 --+
                            worker 1 --+--> partitioned aggregate --> merge --> Sink
                            worker 2 --+
```

Six rules govern this layer.

**There is no GIL, so this is ordinary shared memory parallelism.** The whole reason the diagram above is uninteresting is the point. pandas cannot draw it.

**Late materialization.** A filter produces a selection vector and compaction happens only when a downstream operator needs dense data, which avoids copying columns that are about to be dropped.

**The null fast path branches once per chunk**, on `null_count == 0`, never per element. Most real columns are dense, so the unmasked kernel is the common case and the masked one is the exception. Getting this backwards is easy, because the masked version is the general one and therefore feels like the natural default, and it costs more than any individual kernel gains.

**Aggregation is partitioned.** Each worker builds a private hash table over a radix partition of the key space, so there is no contention on a shared table and the merge is a concatenation rather than a reduction. The table is ours, not `Dict`, for the reasons in document 01 section 7.

**Cancellation is a flag checked at morsel boundaries.** Mojo has no async model and no context type, so this is an atomic bool on the execution context, checked between morsels. From Python it is wired to `SIGINT`, so Ctrl-C interrupts a running query, which pandas cannot do reliably and which is a visible DX win.

**Deterministic output is the default**, with faster nondeterministic group ordering available as an option. Nondeterministic output makes tests unwritable, and a library whose tests are annoying to write ends up with bad tests.

### The thing to worry about here

Mojo has no race detector. Go's `-race` would have caught the entire class of bug this layer produces, and there is no equivalent.

The mitigation is structural rather than diagnostic. Workers own their partitions exclusively, shared state is limited to an explicitly enumerated list of atomics, and every piece of that list is documented in `firepanda/exec/shared.mojo` with the invariant it maintains. If it is not in that file it is not shared, and a code review that adds shared state without adding it to that file is rejected.

That is a weaker guarantee than a race detector and it should be stated as such rather than papered over. Document 09 says what testing has to compensate.

## Streaming and out of core

This is M10, but the operator interface has to accommodate it from M5, so it is designed now.

Polars' history is instructive. They built a correct in memory engine first, added streaming second, and made operators without a streaming implementation fall back transparently. That fallback is what let the streaming engine ship while incomplete, which is the only way a project of this size ever ships anything.

Three things to hold open in the operator interface from M5. Operators declare whether they can stream and the planner chooses per operator. Group by, join and sort need spillable sinks over a memory manager with a disk budget, along the lines of `POLARS_OOC_DISK_BUDGET_MB`. And there needs to be a sink API that never materializes the full result.

## The GPU backend

M9, and the operator interface holds it open from M5 in the same way it holds streaming open.

The design is that an operator declares a device affinity and the planner chooses per operator, with transfer nodes inserted at the boundaries and a cost model that refuses to move data for work that will not pay for the transfer. Same shape as the streaming fallback, same reason.

What makes this cheap enough to be worth doing is that the kernel source does not change. A kernel written as `fn sum[dt: DType](data: Span[Scalar[dt]]) -> Scalar[dt]` is the same source whether it runs under `vectorize` on a core or under `enqueue_function` on a device. What changes is the launch, the memory, and the fact that `Int` is not `DevicePassable` so signatures use fixed width types.

What makes it risky is that a GPU dataframe is not an empty category. cuDF is mature and this has to be honest about that, which is document 10's problem.

## What is deliberately not built

| Not built | Why |
|---|---|
| An implicit index with automatic alignment | pandas' single largest source of surprise. An explicit optional index is provided instead and joins always take explicit keys. |
| `inplace=True` mutation | Frames are values. pandas 3.0 went the same way with Copy-on-Write. |
| `dtype=object` | There is no column of boxed anything, ever. Use a struct column, or hand the frame to Python. |
| NaN as the missing value sentinel | A validity bitmap. NaN is a valid float. |
| Silent dtype upcasting | Explicit casts, with type errors at plan time. |
| `MultiIndex` as a default | Group keys are ordinary columns. A compound index is available when explicitly asked for. |
| Reference counted buffers | The ownership model exists. `Retain` and `Release` is a C++ artifact. |
| A `Dict` anywhere in a hot path | Measured to be unoptimized by the MojoFrame authors, and it is on the group by and join critical path. |
| `List[String]` as a column representation | 20 bytes per element of pure overhead. StringView or nothing. |
| A Python fallback for unimplemented operations | Tempting, and it would make every performance claim unfalsifiable. Missing is missing, and `to_pandas()` is right there. |
