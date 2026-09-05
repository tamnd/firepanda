# The Arrow capsule boundary

Document 07 section 4 makes the one rule that keeps the Python boundary from becoming the bottleneck: plans and scalars cross as objects, data crosses as Arrow, and implementing `__arrow_c_schema__` and `__arrow_c_array__` once buys zero copy interchange with pandas, Polars, DuckDB and pyarrow with no library specific code path for any of them.

M2 built half of that. `firepanda/io/arrow_export.mojo` hands a single column to a C consumer with its own buffers and no copy, and the Mojo suite asserts it by pointer address rather than by contents. What was missing was everything between that column and a Python caller: a frame is not a column, a capsule is not a struct, and the hardest part turned out to be neither of those but the question of who owns the memory once a consumer has it.

This document is the second half. It is worth writing down because the ownership answer changed a type in the binding layer, and a change like that is easy to read as an accident later.

## 1. A frame is a struct, not a list of arrays

Arrow's C Data Interface has no table. It has arrays, and a table is an array of struct type with one child per column, which is why `__arrow_c_array__` on a table like object hands back a single pair of capsules rather than one pair per column. The format string is `+s`, the field names live on the children rather than on the parent, and the parent carries one buffer slot for its validity, which a frame leaves null because a frame has no concept of a null row.

So `export_frame_schema` and `export_frame_array` build a parent around the children that `export_schema` and `export_array` already produced. The only new idea in them is that a parent owns its children: the box behind `private_data` holds the child structs by value, the `children` field points at an array of pointers into that box, and the parent's release callback releases every child before freeing itself. The pointers are filled in after the box has been moved into its allocation, because a pointer taken before the move names a list that is about to be moved out from under it.

Nothing extra is copied. The struct on top is one null buffer and a list of pointers.

## 2. The ownership problem, which is the whole of it

Zero copy across a language boundary is an ownership problem wearing a performance problem's clothes, and at the column level M2 could dodge it. `export_array` takes its column by value and consumes it, so the exported array is the only holder and the question of who else might be looking does not arise.

That answer does not survive contact with Python. A user writes this:

```python
df = firepanda.read_csv("trades.parquet")
table = pyarrow.record_batch(df)
print(df.head())
```

and expects both objects to work. The frame is still owned by the Python object, so the export cannot consume it, and firepanda columns are deep copied rather than refcounted, so an export that copied would be handing pyarrow a copy of every buffer in the frame. For a ten gigabyte frame that is ten gigabytes, which is exactly the cost this protocol exists to avoid.

Three ways out were available and two of them are wrong.

**Copy on export.** Correct, simple, and it gives up the only thing the protocol is for. It also fails the test document 07 section 4 asks for by name, since a copy's buffer address is not the frame's buffer address.

**Borrow and document the danger.** The export points at the frame's memory and the consumer is told not to outlive it. This is a use after free waiting for the first user who assigns a pyarrow table to a variable and lets the frame go out of scope, and the failure is silent bad data rather than an exception.

**Share the frame.** `PyDataFrame` holds an `ArcPointer[DataFrame]` rather than a `DataFrame`. An export takes a copy of the share, which is one atomic increment, and the box behind the exported array holds it. The consumer gets pointers into the real frame, the frame is destroyed when the last of its holders lets go, and neither side needs to know how many holders there are.

The third is what is built. The cost is one field's type in the binding layer and an atomic increment per exported column, and what it buys is that `del df` while pyarrow is still reading is a correct program.

## 3. Which is why there are two export paths

`export_array` still exists and still consumes its column, because a column nobody else wants should not pay for a share it does not need, and the reader and the engine both produce columns like that.

`export_array_borrowed` is the other one. It takes a pointer to a column and a keep alive of any type, and it is generic over the keep alive rather than knowing about `DataFrame`, so nothing in `firepanda/io/` has to import the frame. The two paths share `_fill_buffers`, which is where the actual zero copy lives, because the answer to which pointers go in the buffer array does not depend at all on who is holding the column afterwards.

Every child of a frame export holds its own share, rather than relying on the parent's. A consumer is allowed to move a child out of a struct array and release the parent, and a child whose lifetime depended on its parent's box would then be reading memory nobody was holding. `test_a_child_outliving_its_parent_is_still_readable` is that case written down.

## 4. The capsule is a third allocation and a second destructor

An `ArrowSchema` coming back from an export is a Mojo local. A `PyCapsule` holds a `void*` and outlives the function that made it, so `firepanda/py/convert.mojo` copies the struct into a `malloc` allocation and points the capsule at that. The copy is of the struct, which is a handful of pointers, and not of anything it points at.

The capsule names are `arrow_schema` and `arrow_array` and they are not negotiable. Every library that speaks this protocol checks the name before it touches the pointer, so a capsule called anything else is not an Arrow capsule to anybody, however correct the struct inside it is. They are written down in one place and a test reads them back through `ctypes.pythonapi.PyCapsule_IsValid` rather than through a consumer, so a rename fails on the name rather than on a confusing refusal from pyarrow.

The capsule destructor has to cope with either of two states. A consumer that takes the data moves the struct out and sets `release` to null, which is how the C Data Interface says this one is spent. A consumer that changes its mind, or a capsule that is simply garbage collected, leaves it untouched. So the destructor calls `release_schema` or `release_array`, which are already no-ops on a spent struct, and then frees the allocation either way.

Both allocations here are `malloc` and `free` rather than Mojo's allocator, for the reason document 07's export already had: they are freed from inside callbacks that a foreign runtime, or CPython's garbage collector, invokes on a thread firepanda knows nothing about.

The destructors swallow their errors. There is no way to report one from a capsule destructor, because CPython calls it while it is already tearing an object down, and the only thing an exception raised there could do is surface somewhere unrelated.

## 5. What Python sees, and what it cannot check

The two Mojo methods are `arrow_c_schema` and `arrow_c_array`, and the dunders are in the generated Python layer, for the reason document 13 gives: a bound Mojo type cannot carry a dunder. The Python half also turns the pair into a tuple, because the protocol says tuple and a bound method cannot return one.

The zero copy claim needs a word about how it is tested from Python, because the obvious test cannot be written. Nothing in Python can see a firepanda buffer's address, so there is no address on our side to compare pyarrow's against. What can be seen is that exporting the same frame twice hands out the same address, which a producer that copied could not do, because the first copy is still alive and holding its allocation while the second is made. The direct assertion lives in the Mojo suite, where both addresses are reachable, and the Python suite asserts the half that can be asserted from where the consumer stands.

## 6. What is refused, and why refusing is the right answer

**A requested schema.** The protocol lets a consumer ask for the data in a schema of its choosing and lets a producer refuse. Converting on the way out is not written, and the alternative to refusing is handing back a different schema from the one that was asked for and letting the consumer find out later, so anything other than `None` raises `NotImplementedError`.

**A column with more than one chunk.** A chunked column has no single Arrow array to be. The stream is where that is expressible in principle, and the stream export hands out one batch per frame, so it is refused there too. The refusal is explicit, names the column, and happens before anything is allocated rather than half way through an export.

**A column of the null type.** Inherited from M2. firepanda has the type and constructs no column that carries it, so there is nothing to export.

## 7. Where this leaves M3's exit criteria

Document 07 section 8 asks that `to_pandas`, `to_polars` and passing the frame to DuckDB all work with no copy. Two of the three are now true and tested: `pyarrow.record_batch(df)` and `polars.DataFrame(df)` both read a firepanda frame with no firepanda specific code on their side.

DuckDB did not, and the reason is worth recording because it was not a bug in any of this. DuckDB's replacement scan and its `from_arrow` both look for `__arrow_c_stream__` and neither accepts an object that offers only `__arrow_c_array__`. So the DuckDB row of the exit criteria was blocked on the stream protocol rather than on anything here, and it is now closed by the stream export in document 16 section 9.

The reverse direction, constructing a frame from anything that exposes the protocol, is document 16. It landed after this one and it changed the reading above: measuring what the libraries actually offer showed that `__arrow_c_array__` is almost never what a table has, so the import was built on the stream rather than on the array, and the stream export that DuckDB wants is the piece still missing.
