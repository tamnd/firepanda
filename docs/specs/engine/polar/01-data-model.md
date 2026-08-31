# Polars data model

## Arrow, but their own Arrow

Polars uses the Arrow memory layout and not the Arrow Rust crate. They forked it into `polars-arrow` because the upstream crate carried abstractions they did not want and did not move fast enough. The layout is compatible, so the C Data Interface works and zero copy exchange with pyarrow works, but the code is theirs.

That is the same call firepanda made, for the same reason, and it is worth noting that the largest Arrow consumer in the ecosystem also concluded that using the reference implementation was not worth it.

## Series and ChunkedArray

A `Series` is a typed column. Underneath it is a `ChunkedArray<T>`, which is a list of Arrow arrays of the same type rather than one contiguous array.

Chunking is the load bearing decision. It means that appending two frames is a pointer operation, that a reader can produce chunks as it goes without knowing the total size, and that an operator can work on one chunk at a time. It also means every kernel has to either iterate chunks or call `rechunk()` first, and `rechunk()` is a full copy.

firepanda has `ChunkedArray` in `firepanda/array/` and almost nothing uses it. The kernels take `Array[dt]`, which is one contiguous buffer. That is the first thing `02-execution-model.md` proposes to change, because a chunked column is a precondition for everything else in this folder.

## Types

The physical types are the Arrow ones. Signed and unsigned integers from 8 to 64 bits, plus **Int128**, which was added recently. Float32 and Float64. Boolean as a bitmap. String and Binary. List, Array for fixed length, Struct. Date, Datetime with a time unit and an optional time zone, Duration, Time.

**Decimal** is stable as of the 1.3x line. 128 bit, up to 38 significant digits.

**Categorical and Enum were rewritten**, in pull request #23016, and the reason is instructive.

The old design had each Categorical Series carry its own string to integer mapping, with an optional global `StringCache` to make two Series comparable. That works in an engine where a column is one object. In a streaming engine where a column arrives as a sequence of morsels, each morsel would build its own mapping and every operator would have to merge mappings, and the global cache became a contention point.

The new design makes the category set an explicit object. A `Categories` has a name, a namespace and a physical type, which is UInt8, UInt16 or UInt32 depending on how many categories there are. Series that share a `Categories` are directly comparable with no merging. An `Enum` is the frozen version where the set is fixed at construction.

So: the mapping is a first class value that columns refer to, rather than a property each column carries. That is a lesson about what a streaming engine forces on the type system, and firepanda has a `dictionary` type in `firepanda/array/` that will hit exactly this wall.

## Validity

Arrow bitmaps. Same as everyone.

## Chunk size in practice

Readers produce chunks sized to what they read. The streaming engine's morsel is a separate concept from the chunk, and the typical morsel is 128 thousand rows, which is a stack of chunks rather than one. Document 03 covers the distinction.

## What we should take from this document

Make `ChunkedArray` the thing the kernels operate on rather than a wrapper nothing uses. This is the prerequisite change and it is large.

Learn from the categorical rewrite before writing the dictionary type properly, because a per column mapping does not survive contact with a chunked engine.

Int128 and Decimal128 as the target for decimal support, since matching Polars on the physical type makes zero copy exchange possible and picking a different width makes it impossible.
