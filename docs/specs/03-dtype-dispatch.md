# Compile time dtypes and the generated dispatch table

## The problem every dataframe library has

A dataframe is a list of columns of different types. A kernel is a tight loop over one type. Between those two facts sits the oldest problem in the field: something has to turn a runtime dtype into a monomorphic loop, and every language solves it badly in its own way.

pandas boxes into NumPy arrays and dispatches in Python, so the loop is C but the decision about which loop costs a dictionary lookup and several object allocations. Polars generates code with Rust macros over a fixed dtype list and pays in compile time and in error messages nobody can read. Go has neither generics that monomorphize nor macros, so `arrow-go` writes the same loop eleven times by hand and `kuma` in the sibling spec resorts to a function pointer table keyed by dtype.

Mojo is the first language in this space where the answer is a language feature rather than a workaround. `DType` is a compile time value, `SIMD[dt, n]` is parameterized on it, and `Int` is literally `Scalar[DType.int]`. The vector type is not a library bolted on top of the scalar type; the scalar type is the vector type at width one.

This document is about exploiting that fully, and about the one place it does not reach.

## 1. Kernels are written once

A kernel is a function with a `DType` parameter.

```mojo
fn sum_dense[dt: DType](data: Span[Scalar[dt], _]) -> Scalar[dt]:
    comptime width = simdwidthof[dt]()
    var acc = SIMD[dt, width](0)

    @parameter
    fn step[w: Int](i: Int):
        acc += rebind[SIMD[dt, width]](data.unsafe_ptr().load[width=w](i))

    vectorize[step, width](len(data))
    return acc.reduce_add()
```

That is the whole thing. It compiles to a separate optimal implementation for every dtype it is instantiated with, on every architecture the compiler targets, with the remainder loop written by `vectorize` rather than by hand.

Compare what the Go sibling spec needs for the same kernel: a portable `simd` implementation, an `archsimd` amd64 implementation, an arm64 implementation, a scalar reference, build tags separating them, a runtime feature check selecting between them, and a differential fuzz suite over every tail length from zero to twice the vector width because the tail is where hand written vector code corrupts data silently.

**This is the reason there is no SIMD milestone in document 08.** It is the single largest structural difference between the two specs, and it is worth being precise about what it does and does not buy.

It buys correctness. The tail bug class is gone.

It buys portability for free. A new architecture is the compiler's problem.

It does not buy the last 20 percent. Where a hand tuned intrinsic sequence beats the generic lowering, we have no `archsimd` equivalent to drop into, and the answer is to specialize inside the same function with `comptime if` on the target rather than to fork the file. Section 5.

### The scalar twin is still required

Every kernel keeps a scalar reference implementation, marked `@no_inline` and never called in production, that the differential test compares against.

The reason is not tail bugs any more. It is that the vectorized version is a different program: `sum` in a tree accumulates floating point error differently from `sum` in a line, and comparisons on denormals, NaN and infinity can differ between the scalar and vector paths on some targets. The twin is the specification of what the kernel means, and without it a rewrite can silently change semantics with no test failing.

Document 09 makes this a hard rule. A kernel without a twin does not merge.

## 2. Columns are parameterized, frames are not

```mojo
struct Array[dt: DType](Copyable, Movable):
    var data: Buffer
    var validity: Optional[Bitmap]
    var length: Int
    var null_count: Int
```

`Array[DType.float64]` and `Array[DType.int32]` are different types. Inside any kernel operating on one of them there is no type tag, no branch on dtype, and no indirection.

A `DataFrame` cannot be built out of those directly, because it holds columns of different dtypes in one list and Mojo has no sum type. Pattern matching and unions are on the roadmap and have not landed.

So there is a type erased form:

```mojo
struct AnyArray(Copyable, Movable):
    var dtype: DType          # the runtime tag
    var logical: LogicalType  # decimal scale, tz, nested children
    var data: Buffer
    var validity: Optional[Bitmap]
    var length: Int
    var null_count: Int
```

Note what is not in there. There is no vtable, no boxed element, no `Variant` over fifteen types, and no `AnyType` existential. It is the same fields as `Array[dt]` with the dtype demoted from a parameter to a field. Erasure and re-typing are both free at runtime, because the layout is identical; only the compiler's knowledge changes.

`DataFrame` holds a `List[AnyArray]` and a `Schema`.

## 3. The bridge, which is the only interesting part

Getting from `AnyArray` back to `Array[dt]` is where a language without sum types usually starts hurting. Here it costs an integer compare.

```mojo
comptime NUMERIC = [
    DType.bool, DType.int8, DType.int16, DType.int32, DType.int64,
    DType.uint8, DType.uint16, DType.uint32, DType.uint64,
    DType.float16, DType.float32, DType.float64,
]

fn dispatch[
    op: fn[dt: DType] (Array[dt]) raises -> AnyArray
](col: AnyArray) raises -> AnyArray:
    comptime for candidate in NUMERIC:
        if col.dtype == candidate:
            return op[candidate](col.as_typed[candidate]())
    raise Error("unsupported dtype: ", col.dtype)
```

The `comptime for` unrolls at compile time into a chain of integer comparisons, exactly one of which is true. `op` is instantiated once per candidate dtype, so each branch calls a fully monomorphic kernel. `as_typed` is a `rebind` plus a debug assertion, and compiles to nothing.

Three things to notice, because each is a decision.

**Dispatch happens once per kernel invocation, not once per element.** The compare is amortized over a whole chunk, so at a chunk size of 4096 it is unmeasurable. If it ever shows up in a profile, the chunk size is wrong, not the dispatch.

**`rebind` is mandatory and it is the sharp edge.** Mojo does not instantiate functions during parsing, so inside the `if col.dtype == candidate` branch the compiler does not narrow the type on its own. `rebind` inserts a compile time assertion that the two types resolve to the same thing after instantiation. Get it wrong and the failure is a compile error, not a corruption, which is the right failure mode but produces error messages that are hard to read. Every `rebind` in the codebase is wrapped in `as_typed` and there are no bare ones outside `firepanda/array`.

**Which dtype list a kernel dispatches over is part of its signature.** `NUMERIC`, `INTEGER`, `FLOAT`, `ORDERED`, `HASHABLE` and `ALL` are separate `comptime` lists, and a kernel names the one it supports. `sum` over `NUMERIC` does not instantiate for `DType.bool` accidentally, and a dtype outside the list produces a raised error with the kernel name in it rather than a compile error in a template. This is the mechanism that keeps the code size in section 6 under control, and it is why the lists exist rather than everything dispatching over `ALL`.

## 4. Why not `Variant`

The obvious alternative is `Variant[Array[DType.int8], Array[DType.int16], ...]`, a tagged union over the fifteen column types, with the standard library's own discriminant. There is a well known community pattern that builds trait dispatch this way and it works.

We do not use it, for three reasons.

The layout is the size of the largest member plus a tag, and our members are all the same size anyway, so the union buys nothing structurally while adding a second discriminant next to the `DType` we already have.

The type list becomes part of every signature that touches a column, and with the parameterized logical types there are more than fifteen of them.

And the standard library's `Variant` is not in the stable API subset. `AnyArray` is fourteen lines of our own code with an explicit layout, and when something in the standard library moves, it does not move.

The community `Variant` pattern is the right answer for a heterogeneous container of user types. It is the wrong answer for fifteen types that all have identical memory layouts and differ only in how a loop should interpret the bytes.

## 5. Where the generic lowering is not good enough

`vectorize` over a generic body produces good code and it does not produce the best possible code everywhere. Three cases are known in advance.

**Byte level work with no scalar equivalent.** CSV delimiter scanning, UTF-8 validation and the StringView prefix compare all want a specific shuffle or a movemask, and expressing them as a generic reduction over `SIMD[dt, w]` produces something worse. The answer is `comptime if` on the target inside the same function, using `sys.info` to test for the feature and falling through to the generic body otherwise. One function, several bodies, no build tags, no separate files.

**Horizontal reductions with a known better sequence.** `reduce_add` on a wide accumulator lowers acceptably. Specific widths on specific targets have better sequences.

**Anything where the accumulator wants a different width from the data.** Summing int32 into int64 to avoid overflow means the vector widths differ between input and accumulator, and the generic body has to be written to allow that rather than assuming one width throughout. This is a design constraint on how kernels are written, not an optimization, and getting it wrong means `sum` on an int32 column overflows where pandas would not.

None of these are reasons to abandon the generic form. They are reasons the kernel signature takes the accumulator dtype as a second parameter and the body is allowed to branch on the target.

## 6. The cost, and when to stop paying it

`comptime for` unrolls at the beginning of compilation and the documentation is explicit that it can greatly expand both code size and compile time.

The arithmetic is not reassuring. Fifteen physical dtypes times roughly eighty kernels times a masked and an unmasked variant is 2400 monomorphic function bodies, before nested types, before the fused expression paths in document 02, and before the GPU instantiations at M9.

This is the assumption in this specification most likely to be wrong, and it is worth saying plainly rather than discovering at M6.

**What to measure, from M0 onward.** Compile time of the whole package and stripped binary size, recorded in CI on every commit, with a graph. Not a threshold at first, just a graph, because the shape of the curve tells you whether it is linear in kernels or worse.

**The thresholds that trigger a response.** A clean build over five minutes, or a shipped library over 50 MB, means the current approach has stopped being free.

**The responses, in order of preference.**

Narrow the dtype lists first. Most kernels do not need `int8` and `uint16` instantiations; they need `int32`, `int64`, `float32`, `float64` and a promotion at the boundary. Promoting narrow integer columns to `int32` before a kernel and demoting after costs one pass and removes six instantiations. This is the response that is almost certainly sufficient, and it is available at any time without an architectural change.

Split the package second, so that a user importing only the eager API does not pay for the lazy engine's instantiations. Document 11 keeps the tree shaped for this.

Fall back to a runtime function pointer table for cold kernels third, which is what the Go sibling spec does for everything, and is a real cost only on kernels nobody profiles.

**What not to do is defer the measurement.** A code size problem discovered at M6 is a rewrite of every kernel signature. Discovered at M0 it is a decision about which dtypes appear in a list.

## 7. What this gives a user

Everything above is internal, and one part of it reaches the surface. A user writing a custom kernel gets the same treatment the built in ones get:

```mojo
fn zscore[dt: DType](col: Array[dt]) -> Array[dt]:
    var m = mean(col)
    var s = stddev(col)
    return (col - m) / s

var out = df.apply["price", zscore]()
```

That compiles into the pipeline. It is not a callback, it is not an interpreted expression, and there is no boundary crossing. It is the same code the library's own `mean` goes through.

This is the answer to the question the project exists to answer. In pandas, `df.apply` is between 10x and 100x slower than a built in operation, and the gap is not about the operation, it is about which side of the C-to-Python boundary the code landed on. That is why the MojoFrame paper's largest measured wins are on UDF heavy TPC-H queries, and it is the one advantage in this specification that no amount of engineering effort inside pandas can erase.

From Python the same function is available and it is not free: a Python callable crossing the boundary per row is slow for exactly the reasons it is slow in pandas, and document 04 section 7 says how the API makes that cost visible instead of hiding it.
