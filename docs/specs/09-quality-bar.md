# Quality bar

The stated goal is that this reads like a standard library package. In a language that reached 1.0 two weeks ago there is no established standard library culture to imitate, so this document says what the bar is rather than pointing at one.

Two things about Mojo make this harder than the equivalent document for the Go sibling spec, and they are the reason this document exists in the shape it does.

**There is no race detector.** Go has `-race` and it would catch the entire class of bug the parallel executor produces. There is no equivalent here.

**There is no `vet` or `staticcheck`.** The compiler and the borrow checker are strong, and there is no linter above them looking for the mistakes that compile.

Everything below that looks like over-engineering is compensating for one of those two.

## API design

Every public name matches pandas 3.0 where a match exists. This is not a style preference, it is the product, and it overrides every other naming instinct in this document. Where pandas is inconsistent, we are inconsistent in the same places, because a user's muscle memory is the thing being preserved.

Every public function and struct gets a docstring with a runnable example. `mojo test` runs docstring tests, which makes them the only documentation that cannot rot.

Public signatures use stable standard library types. `String`, `List`, `Span`, `Optional`, `Bool` and our own types. No `Dict`, no `Variant`, nothing from the `algorithm` package, nothing from `max`. Those are unstable at Mojo 1.0 and pinning a public API to them means a compiler upgrade becomes a breaking release. Wrapping is cheap; unpinning later is not.

`raises` discipline: expression construction never raises, so a chain does not force a `try` on every link. Errors are carried inside the poisoned node and surface once, at collection. Anything that touches IO or executes a plan raises. This is the only reasonable arrangement given that `raises` is viral.

Nothing aborts across a public boundary. `abort` is for genuinely unrecoverable state and there are three permitted uses, all in `PyInit`.

Zero values are useful where they can be. A default constructed `Bitmap` is an empty bitmap, not a trap.

## Testing

| Layer | What is required |
|---|---|
| Unit | Table driven. 85 percent coverage overall, 95 percent on `bitmap`, `dtype`, `hash` and `kernel`. |
| Docstring | Every public symbol. Run by `mojo test`, so a broken example fails the build. |
| Fuzz | Every kernel against its scalar twin. Every parser against malformed input. Corpus committed. |
| Differential, internal | Eager against lazy after M4. Dense against masked. Fused against interpreted after M8. CPU against GPU after M9. |
| Differential, external | Against pandas 3.0.5 and DuckDB, in process, from M1. |
| Property | Round trip laws: sort of sort is sort, filter by true is identity, deserialize of serialize is identity. |
| Concurrency stress | Section 3. The substitute for a race detector. |
| Golden | Kernel outputs and `explain()` plan shapes committed as golden files. |
| Leak | A hundred thousand query cycles with flat memory, asserted. |
| Benchmark | Every kernel and operator, recorded per commit, with a regression gate. |
| Compile budget | Compile time and stripped binary size, graphed from M0, thresholded from M4. |
| Boundary | Every exported Python function forced to fail internally, asserting an exception comes back rather than an interpreter crash. |

Three of those matter more than the rest.

**The kernel differential fuzz.** `vectorize` writes the tail, which removes the largest bug class in hand written vector code, and it does not remove the float semantics class. NaN, negative zero, infinity, denormals, and the accumulation order difference between the tree reduction and the linear one. Write this before writing the kernel.

**The eager against lazy differential suite.** After M4 rewires the eager surface to build plans, a wrong optimizer pass returns a result that looks fine. The M1 test suite is the oracle, which is only true if it was written to be one.

**The concurrency stress suite.** Section 3.

## 3. Living without a race detector

The mitigation is structural, and stating it as structural rather than pretending a test suite substitutes for a sanitizer is the honest framing.

**Enumerate shared state.** `firepanda/exec/morsel.mojo` contains every atomic and every mutable global in the library, each with the invariant it maintains and the operators that touch it. A CI grep asserts that no `Atomic` and no mutable global appears anywhere else. A change that adds shared state without adding it to that file does not merge.

**Workers own their partitions exclusively.** The radix partitioned aggregation from document 05 is chosen partly because it needs no shared mutable state at all: private tables, concatenating merge. Where a design has a lock free and a shared variant with similar performance, take the one with no sharing.

**Stress rather than detect.** The concurrency suite runs every parallel operator ten thousand times under artificial load, with randomized worker counts, randomized chunk boundaries and randomized cancellation points, comparing against the single threaded result. It will not find every race and it will find the ones that reproduce, which after enough iterations is most of them.

**Cancellation is the sharp edge.** A morsel executor with a worker pool is exactly the shape that deadlocks or leaks on cancellation, and every cancellation path gets an explicit test asserting that every worker exits and no buffer is left owned.

**Use the borrow checker.** Mojo 1.0's interior origins mean a `Span` into a chunk that outlives a mutation to that chunk is a compile error. That catches a real class of bug the Go version could only test for. It is not thread safety, and it is more than nothing.

If a sanitizer becomes available in the toolchain, adopt it the week it lands and delete half of this section.

## 4. Testing against pandas, which is unusually easy here

Mojo can import pandas in the same process. That means the conformance suite from document 06 is:

```mojo
var pd = Python.import_module("pandas")
var expected = pd.read_csv(path).groupby("k").sum()
var actual   = firepanda.read_csv(path).groupby("k").sum()
assert_frame_equal(actual, expected)
```

No subprocess, no serialization, no bindings required. It works from M1, before the Python front door exists.

In the Go sibling spec this required an entire bindings milestone first, which is why differential testing against pandas was a late luxury there and is a default here. It is the single best thing about writing this library in a language with Python interop, and it is worth more to correctness than any amount of internal test discipline.

`assert_frame_equal` mirrors `pandas.testing.assert_frame_equal`: exact for integers and strings, tolerance based for floats with the tolerance documented per operation, dtype checked by default, and a diff on failure that shows the first differing row with context rather than dumping both frames.

The generated frame corpus covers every dtype, all-null and no-null columns, single row and empty frames, columns of one distinct value, columns of all distinct values, NaN and infinity and negative zero, empty strings, strings at exactly 12 bytes to hit the StringView discriminant, unicode including combining characters and emoji, and DST transition timestamps. Generated once, committed, extended whenever a bug escapes.

## 5. Floating point

Vectorized summation changes the association order, so `sum` is not bit identical to a naive scalar loop. That is expected and the vectorized version is more accurate, since a tree accumulates less error than a line.

Consequences, all of which need writing down before somebody files a bug.

Tests assert closeness with a documented tolerance per operation, never equality.

`sum` on a float column may differ in the last bits between one thread and sixteen, because the partition boundaries change the association order. Determinism of the *ordering* of output rows is guaranteed; bit identical float aggregation across worker counts is not, and the documentation says so. An option exists to force a fixed reduction tree for users who need reproducibility more than speed.

Variance uses Welford or two passes, never the single pass sum of squares, per document 05 section 6.

Integer sums promote and check for overflow. pandas does not silently overflow and neither do we.

## 6. Documentation

The docstring is the API reference and every one has a runnable example.

Beyond that, four documents, and the first is the important one.

**"pandas to firepanda", organized by pandas function name.** Somebody arriving at this library is holding a pandas program. The migration guide is a lookup table from what they have to what they write, with every divergence from document 04 section 6 appearing on the row where it bites rather than in a general caveats section nobody reads.

**A performance guide** that says where the cliffs are: Python callables in `apply`, materializing between steps, chunk sizes, when the GPU path pays.

**An architecture document** for contributors, which is a distillation of document 02.

**A stability policy** naming what is covered by semantic versioning and what is not, per document 11.

## 7. CI

Every commit: build, `mojo test`, docstring tests, unit, fuzz for a bounded time, the differential suites, the concurrency stress suite, benchmarks with a regression gate, and the compile time and binary size graph.

The matrix: Linux x86-64, Linux aarch64, macOS arm64, macOS x86-64, times the pinned Mojo version and Mojo nightly. Nightly is allowed to fail without blocking a merge and it is not allowed to fail silently; a broken nightly opens an issue automatically, because a stabilizing language will break us and finding out on release day is worse.

After M3, add: the four wheels, a free threaded 3.14 build, and an install test from a clean virtualenv on each platform.

After M9, add a runner with an accelerator. Until then the GPU tests skip, and skipping is reported in the summary rather than silently passing, because a test suite that reports green while skipping a third of itself is lying.

## 8. Release

Semantic versioning, from 0.1.

Pre 1.0, minor versions may break the API and every break is in the changelog with the migration.

The Mojo toolchain version is part of the release identity, both in the wheel metadata and in the conda recipe, because the ABI is not stable and a binary artifact built against one runtime is not guaranteed to load against another. This is the constraint from document 07 and it shows up in release engineering, not just in packaging.

Releases go to PyPI and to `modular-community`. The second is a pull request to somebody else's repository and it is slower, so the git install path stays working and documented.
