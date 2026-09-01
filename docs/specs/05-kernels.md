# The kernel layer

## What is different here

In the Go sibling spec this document is an entire milestone, three implementations per kernel, a build tagged dispatch table, runtime CPU feature detection, and a fuzz suite whose main job is catching tail bugs.

Here it is one function per kernel, generic over `DType`, with `vectorize` writing the remainder. The compiler does the part that used to be dangerous.

So this document is shorter than its counterpart and it is about four things the compiler does not do for us: the hash table, the null fast path, the string kernels, and the GPU launch.

## 1. Shape of a kernel

Three variants per operation, and the third is the only one that is optional.

**Dense**, when `null_count == 0`. No mask, no branch, the whole chunk.

**Masked**, when there are nulls. The validity bitmap gates the accumulate.

**Selected**, when a selection vector from an upstream filter is present. Gather then accumulate, or accumulate under a mask, depending on selectivity.

The dense variant is the common case and it must be the one the code reads as the default. This is easy to get backwards, because the masked version is the more general one and therefore feels like the natural implementation with the dense one as an optimization. That instinct costs more than any individual kernel gains, since most real columns are dense and the branch is once per chunk.

```mojo
fn apply[dt: DType, op: BinaryOp](a: Array[dt], b: Array[dt]) -> Array[dt]:
    if a.null_count == 0 and b.null_count == 0:
        return _dense[dt, op](a, b)
    return _masked[dt, op](a, b)
```

One branch, per chunk, on a field.

## 2. Parallelism

`parallelize` over morsels, with the worker count from `sys.info.num_physical_cores()`.

The rule is that a kernel is never internally parallel. Parallelism lives in the executor, which hands whole chunks to workers; a kernel sees one chunk and runs on one thread. Nesting `parallelize` inside a kernel that is already being called from a parallel operator oversubscribes and it is the kind of thing that looks fine in a microbenchmark and falls over under load.

The single exception is a whole-column operation invoked directly from the eager surface with no executor above it, and that path calls the same kernel through a parallel wrapper rather than the kernel parallelizing itself.

There is no GIL, so this is ordinary shared memory parallelism with no ceremony. There is also no race detector, which is document 09's problem and the reason every piece of shared mutable state in the library lives in `firepanda/exec/morsel.mojo`.

## 3. The hash table, which we have to write

The MojoFrame authors identified Mojo's native dictionary as unoptimized, and specifically as the reason their high cardinality aggregation results fall behind. Group by and join are the two operations a dataframe is judged on, and both are a hash table with extra steps.

So `firepanda/hash` is ours, from M1, not later.

Open addressing with linear probing. Power of two capacity, so the modulo is a mask. Keys stored inline for fixed width dtypes, and as a 64 bit hash plus an offset into the key buffer for strings, with a full comparison only on hash collision. Load factor 0.5, growth by doubling with a full rehash. Group ordinals assigned on insertion so aggregation writes into a dense array indexed by ordinal rather than into the table itself.

Four things it does that a general purpose map cannot.

**Batch probe.** Probe N keys at once, computing hashes vectorized and issuing prefetches for the bucket loads before any of them are needed. The cache miss on the bucket load is the dominant cost in a hash aggregation, and hiding it behind a prefetch is worth more than anything done to the hash function.

**Radix partitioning.** Each worker owns a partition of the key space by the high bits of the hash, so worker tables are private, there is no contention, and the merge is a concatenation rather than a reduction.

**Dictionary bypass.** A group by on a dictionary encoded column groups on the integer codes and never hashes a string at all. This is the single largest win available on real string heavy data, and it is the same insight behind MojoFrame's cardinality aware factorization of join keys.

**Integer key bypass.** Low cardinality integer keys with a known range skip hashing entirely and index directly. A group by on a column of 12 distinct small integers should not touch a hash table.

Hash function is a vectorizable non cryptographic hash over the key bytes, seeded per query so that adversarial key distributions cannot be constructed against a published seed.

## 4. String kernels

Strings are the hottest real world dtype and the one where a partial implementation reads as a toy.

Everything operates on `Span[UInt8]` over the StringView data buffer. A `String` is never constructed in a kernel, both because of the per object overhead the MojoFrame paper measured and because Mojo 1.0 iterates strings by grapheme cluster, which is correct for text and wrong for a scan.

The StringView layout is what makes this fast. Equality checks the 4 byte length and the 4 byte prefix first, and for most real data that is the whole comparison. Values of 12 bytes or fewer are entirely inline. Sorting and filtering become dense scans over fixed width 16 byte records.

Regex is RE2 semantics: linear time, no backreferences, no lookaround. That is a user visible divergence from pandas' Python `re` and it needs documenting prominently rather than being discovered in a bug report. Implementation is ours, because there is no Mojo regex engine, and it is the largest single item in the string namespace at M6.

The complete `str` accessor is roughly fifty methods and it ships whole or not at all. Document 06 lists them.

## 5. Order of work

Deliberately not easiest first. This is the order in which kernels pay for themselves on a real workload.

| | Kernel group | Why here |
|---|---|---|
| 1 | CSV field scanning and number parsing | Ingestion dominates every first impression benchmark, and every user's first interaction is `read_csv`. |
| 2 | Comparison to bitmap | Every filter starts here. |
| 3 | Filter and compaction | Selection vector to dense, and the gather that follows it. |
| 4 | Hash and probe | Group by and join. The single most benchmarked operation. |
| 5 | Aggregations | sum, min, max, count, mean, var, std, and the two pass numerically stable variants. |
| 6 | Bitmap operations | and, or, not, popcount, and the append and slice paths. |
| 7 | String compare, search, and the prefix fast path | The hottest real dtype. |
| 8 | Sort | Radix for integers and, as it turned out, for text as well: the first eight bytes of an element pack into a `UInt64` whose integer order is the element's order, so the same passes run, and a stable comparison sort finishes only the runs whose first eight bytes were identical. Pattern defeating quicksort is not in the implementation and no dtype has needed it. |
| 9 | Elementwise arithmetic | Last, deliberately. It is the easiest to vectorize, the most satisfying to benchmark, and almost never the bottleneck in a real query. |

Number 9 being last is the point of the list. Arithmetic is where a naive optimization effort starts, and it is where the least time is spent in any query that reads data from disk.

## 6. Numerical correctness

Three rules, because these are the places a fast kernel is quietly wrong.

**Sum promotes.** Summing an int32 column accumulates into int64 and summing an int64 column checks for overflow. pandas does not overflow here and neither do we. This means the kernel signature carries an accumulator dtype separate from the data dtype, which is a constraint on how every reduction is written and is the reason document 03 section 5 mentions it.

**Variance and standard deviation are two pass or Welford.** The textbook single pass sum-of-squares formula loses catastrophic precision on data with a large mean and small variance, which is most real financial and sensor data. It is also the fastest and most obvious thing to write, which is why it needs to be a rule.

**Vectorized summation is not bit identical to scalar summation and is more accurate.** Tree reduction accumulates less error than a linear sum. Tests assert closeness with an explicit tolerance, never equality, and the tolerance is documented per kernel. This is true of Polars and DuckDB too and it is section 6 of document 04's divergence list for a reason.

## 7. The GPU path

M9. Designed now because the kernel signature has to accommodate it, built later because a GPU dataframe is not an empty category and the CPU engine has to be good first.

What makes it cheap: the kernel body is the same source. `fn sum[dt: DType](data: Span[Scalar[dt]])` runs under `vectorize` on a core or inside a kernel launched with `enqueue_function` on a device, and the difference is the launch, the memory and the indexing, not the arithmetic.

What has to be built around it:

`DeviceContext` acquisition guarded by `sys.has_accelerator()`, so the entire library builds, tests and works with no accelerator present and every GPU test skips.

Buffers that can live in device memory, meaning `Buffer` grows a location and the transfer is explicit and visible in `explain()`.

A cost model that refuses to move data for work that will not pay for the transfer. The default is CPU and the GPU is chosen per operator when the estimated row count clears a threshold, and the threshold is measured rather than guessed.

Fixed width types in kernel signatures, because `Int` and `UInt` stopped being `DevicePassable` in Mojo 1.0.

The honest framing, which document 10 enforces: cuDF exists, it is mature, and it is the comparison. What firepanda can offer that cuDF cannot is that a user defined function written once runs on both devices, because it is the same language on both sides. In cuDF a UDF is Numba compiled Python with its own restrictions or it is CUDA C++.

## 8. Testing a kernel

Four gates, all required, and a kernel does not merge without them.

**A scalar twin**, `@no_inline`, never called in production, which is the specification of what the kernel means.

**Differential fuzz against the twin.** All lengths from zero to twice the vector width, all null and no null shapes, and for floats the full menagerie of NaN, negative zero, infinity and denormals. The tail bug class is mostly gone thanks to `vectorize` but the float semantics class is not.

**A golden file** for the output on a fixed input, committed, so that a rewrite cannot silently change semantics.

**A benchmark** with the number recorded in CI from the first commit, so that a regression is attributable to a change while somebody still remembers writing it.
