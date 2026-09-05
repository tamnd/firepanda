# Reading somebody else's Arrow

Document 15 is the export direction, which is how a firepanda frame reaches pyarrow, Polars and pandas. This is the other one, and it is the half that decides whether firepanda can be put in the middle of anything. Until it exists the only way to make a frame is to read a CSV, and a library that cannot be handed data is a library nobody can add to a pipeline that already runs.

It covers `firepanda.from_arrow`, the code behind it in `firepanda/io/arrow_stream.mojo`, and the two things measuring real producers changed about the plan.

## 1. The stream is the main road, which was not the plan

The plan in document 15 was to build the array direction first and the stream afterwards, on the reading that `__arrow_c_array__` is the basic case and `__arrow_c_stream__` is the one that adds chunking. That reading is wrong, and the way to find out was to ask the libraries rather than the specification.

Against pyarrow 25.0.1, Polars 1.44.1 and pandas 3.0.5, this is what is actually offered at the container level:

| Type | `__arrow_c_array__` | `__arrow_c_stream__` |
|---|---|---|
| `pyarrow.RecordBatch` | yes | yes |
| `pyarrow.Table` | no | yes |
| `pyarrow.Array` | yes | no |
| `polars.DataFrame` | no | yes |
| `polars.Series` | no | yes |
| `pandas.DataFrame` | no | yes |

One container type in six offers the array half. An importer that read only arrays would read a pyarrow record batch and nothing else, which is to say it would not read a pandas frame, which is the whole point. So the stream is not the follow up to the array direction, it is the road, and both were built together rather than one after the other.

The same measurement settles the DuckDB row of M3's exit criteria in document 08, which document 15 section 7 left open. DuckDB refuses an object that offers only `__arrow_c_array__` with `Python Object Type DataFrame is not an accepted Arrow Object`, and what it wants is `__arrow_c_stream__`. That is the export side of the stream, which is still to be written, and the import side landing first does not close it. It does mean the stream structure, its four callbacks and the release protocol are now declared, exercised and tested from the consumer side, which is most of what the export needs.

## 2. This direction copies, and that is not a compromise

The export borrows. A consumer gets pointers into the frame's own buffers and a share of the frame that keeps them alive, and document 15 section 2 is entirely about making that safe.

The import cannot do the reverse and should not try. A firepanda buffer is 64 byte aligned and over allocated at the end, because the kernels read past the last element rather than branching on the tail, and that over allocation is a promise no foreign buffer makes. Borrowing a pyarrow buffer would mean either giving up the tail read in every kernel, which is a cost paid on every operation for the life of the frame, or reading past the end of memory somebody else allocated. So the import allocates and copies once, on the way in, and everything downstream is a normal firepanda frame with no flag on it saying where it came from.

The copy is also what makes the ownership simple. The producer's memory is released before `from_arrow` returns, so there is no share held across the boundary in this direction and no question about who frees what later.

## 3. A stream is many batches and one allocation

A stream hands out a schema and then a batch at a time until it hands out a released array to say it is finished. firepanda has no chunking, so a table of three chunks has to become one column per field with all the rows in it.

The obvious way is a frame per batch and a concatenate at the end, and it allocates every column twice and copies every byte twice. The way this does it is the one `firepanda/io/assemble.mojo` already existed for: read every batch first, add up the row counts, allocate each column once at the final size, and then fill the batches into it in parallel. Each batch knows its own offset into the destination before any of them starts writing, so there is nothing to synchronise.

The batches have to outlive the assembly, and that is the one ordering rule in the file. What `struct_columns` hands back for each batch is a window, which is a copy of the child array's own fields with the parent's offset folded in and with the release callback deliberately blanked, because a window owns nothing. The real batch arrays are kept in a list and released together after `assemble` has copied the bytes, on the success path and on the failure path alike. Releasing a batch as soon as it was read would free memory the next batch's assembly is still going to read.

A stream that hands out no batches at all is still a stream that described itself, so it becomes a frame with the right columns and no rows. That case does not go through `assemble`, because there is no array to give it: a released `ArrowArray` has no buffer count and would be refused as malformed, which it is, rather than read as empty.

## 4. Where the offsets are, which is where the trap is

Arrow slices by recording an offset rather than by moving anything, and which structure the offset is recorded on is up to the producer. Both have to be honoured because different producers choose differently. pyarrow's `RecordBatch.slice(1)` records the offset on the children and leaves the parent at zero.

The trap underneath that is the meaning of a child's `length`. It counts from the child's own offset, not from the front of the child's buffers. So the check that a child is long enough for its parent is `parent.offset + parent.length <= child.length`, and the window to read is `offset = child.offset + parent.offset, length = parent.length`. Writing the check with the child's offset added to both sides looks symmetrical and refuses every sliced batch pyarrow produces, with a message saying a column of two rows is not enough for two rows, which is how it was found.

## 5. What the real producers taught the checks

Two of the refusals in the import path were written from the specification and were wrong about it, and both were found by pointing the code at a library rather than by rereading the document.

**A string view column has a minimum of three buffers, not four.** The layout is validity, the sixteen byte views, then zero or more variadic data buffers, then the buffer of their lengths. A producer is allowed to have no data buffers at all, and Polars has none for any column whose every string is short enough to sit inside its view, which is a very common column. The importer required four and refused every such column. Three is the floor, and a column that arrives with three is one that cannot legally contain a long view, so nothing downstream needs a second check.

**The buffer of lengths may be absent when there are no data buffers.** Same case, same producer. The lengths exist so that a view can be checked before it is followed, since a view is three numbers a stranger chose and following one unchecked is an out of bounds read waiting for a malformed file. With no data buffers there is nothing to check, so the list is not required.

Both of these were live bugs in the array import that landed with document 15's work, not new mistakes. Neither was reachable from the export side, which is the argument for testing an interchange format against the libraries that implement it rather than against a careful reading.

## 6. What is refused

**Anything that offers neither half of the protocol.** The message names the type that was passed and lists what to pass instead, because the caller who gets this has usually passed a list or a dict and wants to know what a frame is made from.

**A single column.** A `pyarrow.Array` offers `__arrow_c_array__` and is not a table. Arrow has no table type at this level, so a frame is an array of struct type, and a schema that is not `+s` is refused by name. Without the check the failure is quiet rather than loud: an array of int64 read as a struct is a schema with no children, which is a frame with no columns rather than an error.

**A struct array with null rows.** Arrow allows it. A frame has rows that are present and columns that are null, and never the other way round, so the shape is refused rather than flattened into something that loses information.

**A type firepanda does not have.** Dictionary encoded columns, nested columns, the null type, and every format string outside the fixed width set and the two view types. These are gaps rather than errors, and section 7 is about telling the two apart.

**A stream that never ends.** A producer that hands out more than sixteen million batches without saying it is finished is a bug in the producer, and the alternative to a limit is a loop that allocates until the machine stops.

## 7. Two ways of refusing, two exceptions

A caller does different work about the two kinds of failure. If firepanda has no column for what the producer sent, that is a gap in this library and no amount of passing better data will fix it, so it arrives as `NotImplementedError` through `firepanda.errors.UnsupportedError`. If the data does not hold together, that usually means a bug in whatever produced it, so it arrives as `ValueError` through `firepanda.errors.InvalidArgumentError`. Reporting a malformed buffer as a missing feature sends the reader looking in the wrong place, and document 14 is the rest of how one Mojo error becomes the right Python class.

The two are told apart by the message, in `_import_kind` in `firepanda/py/frame.mojo`, and that is the honest option rather than the lovely one. Tagging them where they are raised would put the binding's error vocabulary inside `firepanda/io/`, which is code that has to compile and work with no Python anywhere near it. So the rule is written down in one place instead: every refusal in the import path that means a gap says so with the word supported, and none of the ones about malformed data use it.

## 8. What is tested, and against whom

The Mojo suite in `tests/test_arrow_stream.mojo` builds a conforming `ArrowArrayStream` by hand around firepanda's own exporter. That is not a compromise, because it is the only way to get a stream into the Mojo suite without linking somebody else's library into it, and because every assertion is against the values that went in rather than against what the exporter said about them. What it buys that a round trip cannot is the two things a real producer does and ours cannot do by accident: hand out more than one batch, and fail part way through after batches have already been taken.

The Python suite in `python/tests/test_arrow_import.py` is the half that matters more, and it is deliberately written against pyarrow, Polars and pandas rather than against a fixture. A test that built its own conforming stream would be testing this code against my reading of the specification, which is the reading that produced the code. Those three implemented the protocol without knowing firepanda existed, so what they hand over is what the protocol actually is. It covers a record batch, a table, a table of several chunks, a slice, a table with no rows, a Polars frame, a Polars frame of only short strings, a pandas frame, a round trip out and back, a frame read after its producer has been collected, a thousand imports in a loop, and each of the refusals above.

The loop is there for the two failures a single import will not show. A leak of one frame is invisible and a double release is a crash that one import is unlikely to reach, and repetition turns both into something a test can see.

## 9. What is left

**The stream export.** `__arrow_c_stream__` on a frame, which is what DuckDB wants and what a chunked column needs in order to be exportable at all. The structure and the release protocol are now settled by this work.

**A `Series` on both sides.** There is no bound `Series` type yet, so neither direction has a column level entry point.

**`requested_schema`.** Still refused in both directions. Honouring it is a conversion, and a conversion belongs in the import path where it can allocate, which is now a place that exists.

**Dictionary and nested columns.** Both refused by name. The dictionary reader is what the categorical section of the conformance suite is waiting on.
