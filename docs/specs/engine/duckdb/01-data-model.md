# DuckDB data model

## Logical types over physical types

DuckDB separates what a value means from how it is stored. `DECIMAL(18,2)`, `TIMESTAMP`, `DATE` and `BIGINT` are four logical types with one physical type between three of them, which is a 64 bit integer. `VARCHAR` is a logical type whose physical type is a 16 byte string handle. `ENUM` is a logical type whose physical type is an unsigned integer sized to the number of categories.

This is the same split firepanda already has, in `firepanda/dtype`, where `LogicalType` sits over `DType`. We got that right by copying it.

The nested types are `LIST`, `STRUCT`, `MAP`, `ARRAY` for fixed length lists, and `UNION`. Version 1.5 added `VARIANT`, which is a self describing binary encoding for JSON shaped data, and `GEOMETRY`, which puts spatial primitives in the core engine rather than in an extension.

`VARIANT` is worth a sentence because of what it replaces. Storing JSON as a string means paying the parse on every query that touches it. `VARIANT` deserializes once into a binary form the engine can index into, and DuckDB claims up to a hundredfold on JSON analysis with it together with shredding, which is the technique of pulling frequently accessed fields out into their own physical columns.

## Vectors

Execution moves data in vectors. A vector is one column's worth of up to `STANDARD_VECTOR_SIZE` values, which defaults to 2048. A `DataChunk` is a set of vectors, one per column, all the same length. That is the unit every operator consumes and produces.

2048 is chosen so that a vector of 64 bit values is sixteen kilobytes and a whole chunk of a handful of columns fits in L1 or L2. Small enough to stay in cache, large enough that the per call overhead of an operator is amortized over thousands of values.

## The four vector formats

The point of having formats is that compression should survive into execution rather than being undone at the scan boundary.

**Flat.** A contiguous array. The logical and physical layout are the same. This is what firepanda's `Array[dt]` is and it is the only format firepanda has.

**Constant.** One value, standing for the whole vector. A literal in an expression is a constant vector. So is a column segment that was stored with constant compression. An arithmetic kernel that sees a constant on one side can run a scalar loop rather than a vector one.

**Dictionary.** A child vector plus a selection vector of indices into it. This is what a dictionary compressed segment decompresses into, which is to say it does not decompress at all. It is also what `Slice` produces, so taking a subset of a vector is a selection vector and not a copy.

**Sequence.** A base value and an increment. A row number column is a sequence vector and costs sixteen bytes.

## The unified vector format

Four formats times four formats is sixteen cases for every binary kernel, and DuckDB does not write sixteen. `ToUnifiedFormat` converts any vector into a common view consisting of a data pointer, a validity mask and a selection vector. Generic code reads `data[sel[i]]` and checks `validity[sel[i]]`, and that expression is correct for all four formats. A flat vector gets the identity selection, a dictionary vector gets its own, a constant vector gets a selection that is all zeros.

DuckDB still writes specialized paths where a format makes an operation trivially cheap, for example constant on one side of an arithmetic operator. The unified format is the fallback that keeps the specialization list short rather than the only path.

This is the single most transferable idea in this document. firepanda has no equivalent, so every kernel we write assumes a flat contiguous column, and every operation that produces a subset materializes it. A `filter` copies. A `take` copies. A join output copies both sides. Adding a selection vector layer is a large change and `02-execution-model.md` argues for it.

## Validity

A bitmap, one bit per value, set means valid. Same as Arrow and same as firepanda. DuckDB has a fast path for the all valid case where the mask pointer is null, which is the same thing firepanda's `null_count() == 0` check does.

## Strings

`string_t` is sixteen bytes and is a union of two layouts.

```
inlined:  uint32 length,  char inlined[12]
pointer:  uint32 length,  char prefix[4],  char* ptr
```

Strings of twelve bytes or fewer live entirely inside the handle. Longer ones store a four byte prefix beside the pointer.

Two things come out of that. Equality and ordering compare the length and the prefix first, and most unequal strings differ inside the first four bytes, so most comparisons never chase a pointer. And short strings, which are most strings in practice, have no indirection at all.

firepanda already has this. `firepanda/array/strings.mojo` implements the Arrow StringView layout, which is sixteen bytes with a four byte length and a four byte prefix, inlining up to twelve. Arrow's version and DuckDB's are the same idea with the fields in a different order.

## What we should take from this document

The vector formats, and specifically the dictionary vector with its selection vector, because that is what makes filter and take free.

Nothing else. The type lattice we have, the validity representation we have, the string layout we have.
