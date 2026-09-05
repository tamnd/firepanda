# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0, minor versions may break the API. Every break appears here with the migration, not just with a note that it happened.

The Mojo toolchain version is part of a release's identity and is recorded with each entry, because the Mojo ABI is not stable within 1.x and a binary artifact built against one runtime is not guaranteed to load against another.

## [Unreleased]

## [0.6.43] - 2026-09-05

Built against Mojo 1.0.0 (ed45d567). A frame gets an index, the streaming join becomes an operator the pipeline can actually run a query through, and three reductions stop reading the same column twice.

The largest change by surface area is the index. A pandas row is identified by a label and not by a position, and every operation that chooses rows carries the labels of the rows it chose. firepanda had none of that, so `df.tail(5)` came back labelled from zero and a sort threw the old labels away. `DataFrame` and `Series` now have one, it is either an arithmetic range costing no memory at all or an array of labels, and `take`, `filter` and `slice` carry it, which covers `head`, `tail`, `sort_values` and `drop_nulls` underneath. `group_by` grows an `as_index` flag on top of it. Measured against the pandas conformance suite, the groupby section goes from 54 passing to 111 and the failure count across the three sections goes from 88 to 31.

The engine work is the join. The built table, the code to row lists and the walk that pairs against them are all values now rather than phases inside one function, so a `Join` is a node a pipeline can hold, hashing its build side once in `bind` and emitting a chunk per chunk. A reduction sitting behind it folds each chunk on the core that produced it rather than on the driver's thread afterwards, which was worth a fifth off a join followed by a reduction at eight million rows. Run against the db-benchmark join queries on a 13900K, the three that join against a small table are now between 1.4 and 1.6 times DuckDB and hold two thirds of its memory. The two that join against a table their own size are not, and the reason is written down in issue #79 rather than glossed over.

The rest is arithmetic that was being done twice. A grouped mean asked for a sum and then asked for a count, and on a float column the count had to walk the whole column again first because a NaN is not a value. It reads the column once now, and the variance, the standard deviation and the skewness take their count from the same call, so a standard deviation is two passes over the column where it used to be four.

On the Python side, the distribution claim that had been measured by hand on one laptop is now a build script and four tests that run on every platform the project ships for. `pixi run build-extension` produces a self contained directory of 2.9 megabytes, and the tests import it from a child interpreter with no Mojo toolchain on the path, asserting that inside the child rather than trusting it. Binding a whole type was tried next and it does not work: the toolchain's type builder has no properties, no subscripting and no length, so `df["revenue"]` cannot be written in Mojo, and the object a user holds is a Python object holding a Mojo one by design rather than by convenience.

### The pandas surface cannot be written in Mojo, and now we know why

`docs/specs/13-the-bound-type-is-not-a-dataframe.md` is the measurement that document 12 did not make. Document 12 bound four functions, because that was all the distribution question needed. This one binds a type, and binding a type is a different exercise with a much larger consequence.

`PythonTypeBuilder` offers `def_py_init`, `def_method`, `def_staticmethod` and `finalize`, and nothing else. There is no `def_property`, no `def_getset`, no way at a type slot, and the type it produces is not an acceptable base class and its instances have no `__dict__`, so neither subclassing it nor decorating it after the fact is available either. Asked of a bound type directly, `f["a"]` is not subscriptable, `len(f)` has no len, `f.shape` does not exist and `iter(f)` is not iterable. `df["revenue"]` is the most common line of pandas ever written and it cannot be implemented in Mojo in this toolchain.

So the object a user holds is a pure Python object holding a Mojo one, and that is now the architecture rather than a convenience. It is where document 12 had already put the error mapping, and section 5 of the new document adds a reason that cannot be designed around: the binding layer raises its own untyped error for the commonest user mistake, an arity mismatch, as `Exception: TypeError: <mojo function>() takes 1 positional argument but 2 were given`, which `except TypeError` does not catch and which names no function.

The cost of that layer is measured rather than assumed, at 44 to 85 nanoseconds of delegation per call against 90 for the bound call itself, which is nothing next to any operation that touches a column and is the reason the boundary gets crossed once per column and never once per row.

Two smaller findings are recorded with it. A bound method takes at most seven arguments after `py_self`, and keyword arguments do not count against that and survive the crossing with their order intact, so the narrow convention goes underneath and the real pandas signature goes in the Python layer. And a type implementing `Writable` gets `__str__` and `__repr__` for free, but both of them come from `write_repr_to` and `write_to` is silently ignored.

M3 P2 grows a fourth output because of this. The binding table generates the Python wrapper as well as the registration, the stubs and the parity test, and the parity test gains a third surface to hold in agreement.

### A Python extension that builds, and a build that gets checked

`firepanda/py/` is the Python front door and it opens onto one function, which reports the version. That is the point rather than a limitation. The distribution claim in `docs/specs/12-the-python-front-door-measured.md` was measured by hand on one laptop on one afternoon, and every way it could stop being true is a quiet one, so this turns it into `pixi run build-extension` and four tests that run on every platform the project ships for.

The build script produces a directory holding the extension and the four Mojo runtime libraries, with nothing in it pointing anywhere outside itself, and it prints what that weighs. Today that is 2.9 megabytes. The tests stage that directory the way an installed wheel has it and import it from a child interpreter with the environment taken away, asserting inside the child that no Mojo toolchain is on `PATH` before it imports anything, because a test that quietly ran inside the build environment would pass and mean nothing.

Automating the hand copying found two traps worth writing down. Following dependencies by file name vendors the system C++ library, because the Mojo runtime names that one by absolute path while it names its own libraries through `@rpath`, and a second `libc++` in a host process is a worse problem than the 1.17 megabytes it costs. And on Apple silicon, editing a load command invalidates a binary's signature, after which the loader kills the process rather than reporting anything: the entire symptom is `import firepanda` exiting 137 with both streams empty. The script re-signs what it edits and edits as little as it can, so the vendored libraries come out byte identical to the toolchain's, and one of the tests verifies signatures purely so that failure has a name.

`python/firepanda/__init__.py` asks the extension what version it was built from rather than carrying a fourth copy of the string, and the test compares that against the wheel metadata, which is how a wheel assembled out of two different builds gets caught.

### A grouped mean reads its column once instead of twice

`_mean_core` asked for a grouped sum and then asked for a grouped count, and the two walked the same column and the same ordinals separately. On a float column it was worse than that, because a NaN is not a value, so the count had to build a presence bitmap over the whole column before it could count anything. A mean over ten million floats into a hundred groups read eighty megabytes of values twice and forty megabytes of ordinals three times, to produce two arrays of a hundred entries.

`_sum_and_count` does the two in one scatter. Each worker holds a table of sums and a table of counts side by side and writes both from the same read of the value, so the count costs an increment rather than a second pass. The rule about what counts is now decided once in the loop rather than reached independently by `_addend` and `_count_core`, which is the same answer arrived at more directly.

Only the private table route is fused. The serial route and the partitioned route keep the shapes they had and still ask for the sum and the count one at a time, because a partitioned scatter counts its rows into partitions before it reads a value and carrying a second accumulator through it is a different change.

The variance, the standard deviation and the skewness all take their counts from the same call now. Each of those used to call `_mean_core`, which computed a count internally, and then call `_count_core` again for the same numbers, so a standard deviation was four passes over the column where two will do.

Measured on an i9-13900K at 0.5GB of db-benchmark, three ABBA blocks of seven runs. q4, which is three means over ten million rows into a hundred groups, went from 0.015 s on every one of six runs to 0.012 or 0.013 on every one of six. q3, a sum and a mean into a hundred thousand groups, went from 0.047 to 0.049 down to 0.045 to 0.046. q6, a median and a standard deviation, went from 0.279 to 0.289 down to 0.273 to 0.286, which overlaps and is not claimed. q1, q5, q7 and q9 do not use a mean and did not move.

Against the rivals at the same size on the same machine, q4 goes from 0.015 s to 0.012 against polars at 0.012 and DuckDB at 0.011, so it is level rather than behind, and q3 is 0.045 against polars at 0.121 and DuckDB at 0.069.

### A join node running on a worker no longer asks for workers of its own

Three kernels inside a join hand themselves out in morsels once the row count is high enough: the probe at a hundred and thirty one thousand rows, the pairing at the same, and the gather at sixty five thousand. A pipeline chunk is a hundred and twenty eight thousand rows, so a `Join` node running inside a worker was starting a second layer of tasks inside the task it was already in, three times per chunk.

`take_any`, `probe_side` and `pair_probe` now take a `spread` flag, defaulting to true so every existing caller is unchanged, and `node_apply` passes false because it is the entry point the workers share. `Join.process` threads it down to all three.

This was measured on an i9-13900K at ten million rows joined against ten thousand and it moved nothing, in either direction, at chunk sizes of thirty two thousand, a hundred and thirty one thousand and five hundred and twelve thousand. The reason is that the morsel scheduler starts one task per worker rather than one per morsel, and a task that finds no free worker runs on the thread that made it, so the inner layer collapsed back onto the caller. It is recorded here as a change that costs nothing and is not visible today, kept because asking for thirty two times the tasks is only free while the queue is saturated, and the queue stops being saturated the moment a pipeline is narrower than the machine.

Two tests were added, one per kernel, that run the same probe and the same pairing both ways at a row count above the threshold and compare the ordinals element by element.

### The Python front door, measured rather than argued for

`docs/specs/07-python-bindings.md` was written before any of it existed and it asks the reader to check it against the installed toolchain. `docs/specs/12-the-python-front-door-measured.md` is that check, run against Mojo 1.0.0 on macOS arm64, and every number in it came out of a command.

The headline is that the problem document 07 calls the hardest one is largely solved and cost 2.9 megabytes. A Mojo Python extension links two Mojo runtime libraries which pull in two more, it does not link libpython at all, and the four of them vendored beside a 177 kilobyte extension import and run on a stock Homebrew CPython 3.14 launched with `env -i` and a `PATH` that has no Mojo binary on it. For scale, the pandas 3.0.3 and numpy 2.5.2 this project is measured against occupy 46.4 and 23.2 megabytes installed.

The Arrow crossing works today and the zero copy is proven by address rather than by eye. firepanda's Arrow C Data Interface landed at M2 and knew nothing about Python; wrapping an exported schema and array in PyCapsules is about forty lines, and `pyarrow.array` accepts the result with the right type, length, values and null count, with the values buffer address on the pyarrow side equal to the one firepanda handed out. What is still missing on that side is `ArrowArrayStream`, which is what `__arrow_c_stream__` needs.

Two things do not work and they are the real M3 risks, neither of which document 07 flags as one. A function registered with `def_function` can raise exactly one Python exception type and it is `Exception`, and setting a typed one by hand first does not survive, because the binding wrapper overwrites it, so the error mapping table has to live in the thin Python layer. And `PyErr_CheckSignals` is not exposed by `std.python._cpython` at all, so the Ctrl-C exit criterion needs either a `dlsym` of our own or a different threading model.

Also recorded: `GILReleased` works and was measured rather than assumed, at 673 ticks of another Python thread against 1 while the GIL is held, and the first attempt at that measurement was worthless because the compiler folded the workload away.

### A grouped result can be indexed by its key

pandas returns `df.groupby("k").sum()` indexed by the key, with the key gone from the columns and the level named after the column it came from. `group_by` grows an `as_index` flag that asks for exactly that, and `group_agg` and `group_count` pass it through, so the three spellings a caller reaches for all have it.

The key is read off the finished frame rather than off the gathered key column, because the dropna filter and the sort have both run by then and the labels have to be the keys of the rows that survived, in the order those rows came out in. Reading it earlier gives the right set of labels in the wrong order, and `sort=False` is the case that tells the two apart.

`as_index` defaults to false, which is the one default in this method that does not match pandas, and the reason is that pandas puts two keys into a MultiIndex and firepanda has none. Defaulting to true would mean a two key group by raising by default, which is a worse answer than a shape that differs. So it is off, asking for it with more than one key raises and says why, and the default flips when the MultiIndex lands. Everything else here already matches pandas: `dropna` and `sort` are both on. Nothing existing changes shape, so there is no migration.

Two gaps in the index shipped in the previous entry are fixed here and both are the same mistake. `select` builds its result from a schema and a column list rather than by copying the frame, so it was the one row preserving method that had to be told to carry the labels, and `drop` is `select` underneath, so it lost them too. `column` is the other: a column of a frame has the frame's rows, so it has the frame's labels, and without that `df.groupby("k")["v"].mean()` comes back with the labels dropped at the last step. The rule these three now follow is the previous entry's rule read the other way round: an operation that does not choose rows keeps the labels it was given.

Eight tests in `tests/test_index.mojo`, including one that asserts the flat and indexed forms hold the same numbers in the same order, since `as_index` moves a column and must not change an answer.

Measured on the conformance suite in [firepanda-compat](https://github.com/tamnd/firepanda-compat), the groupby section goes from 54 pass and 62 fail to 111 and 5. Of the 5 left, 3 are the Arrow reader refusing a dictionary encoded column and 2 are the two key grouping asking for a MultiIndex. basics is unchanged at 155 and 22 and stats at 52 and 4, so nothing regressed. Across the three sections the failure count goes from 88 to 31.

### Frames and series carry row labels

pandas identifies a row by a label and not by a position, and every operation that chooses rows carries the labels of the rows it chose. firepanda had none of that. A frame was a schema, some columns and a height, and the only answer it had to "which row is this" was the row's position, so `df.tail(5)` of a ten row frame came back labelled 0 through 4 where pandas labels it 5 through 9, `sort_values` came back in the new order with the old labels thrown away, and `dropna` came back renumbered instead of with holes in it. `DataFrame` and `Series` now have an `index`, and `take`, `filter` and `slice` carry it, which covers `head`, `tail`, `sort_values` and `drop_nulls` as well since each of those is one of the three underneath.

The design is in `firepanda/frame/index.mojo` and the whole of it is that an index has two forms. It is either an arithmetic range, which is a start and a length and no memory at all, or an array of labels. Every frame starts as a range and only pays for an array when something takes it apart, and a slice of a range is another range, so `head` and `tail` really do cost nothing. This is not an optimization added afterwards, it is the reason the field can exist at all: every frame in the library grows it, including the ones in the inner loop of a benchmark, so what it costs when nobody asked for it had to be the first question.

What it costs when somebody does ask for it was measured, and the guess was wrong in both directions. Against a `frame/slice_half` control that this change does not touch and that still moved 4.6 per cent between the two binaries, `frame/take` moved 2.8 per cent and `frame/filter` moved 16.8 per cent, so the gather is at the noise floor and the filter is not. Two new benchmark rows say why. `kernel/take_range` is 0.794 ms against `kernel/take_scattered` at 2.459, because a gather is a random read a row and a label needs no read at all, so labels are worth about a third of a real column there and they go on every core beside the other three. `kernel/filter_range` is 1.708 ms against `kernel/filter` at 1.768, level, because a compaction reads sequentially and the read it drops was being prefetched anyway, so on that side labels cost very nearly a whole extra column. Every number is a minimum rather than a mean, on a machine that was busy throughout, which is why the control is quoted next to them.

Two earlier versions of the loops were much worse and both are recorded in the module docstring, because both were the tidy thing to write. The first materialized the whole range into a column and gathered that, which costs a pass proportional to the height going in rather than the height coming out and took `frame/take` from 5.96 ms to 15.16. The second wrote `start + at` straight into the output and was still 46 per cent on `frame/take` and 129 per cent on `frame/filter`, because writing the loop at the frame layer in the obvious way meant a bitmap read-modify-write per null, an unhoisted read per row, and worst of the three a serial pass bolted onto a gather the kernel had gone to some trouble to spread across cores. The loops are `take_range` and `filter_range` in `kernel/select.mojo` now, beside the general forms they shadow. The lesson is not about ranges, it is that a frame level loop running next to a kernel inherits the kernel's standards.

`tests/test_index.mojo` adds twenty tests, and most of them check one representation against the other, since almost every bug this can have is the range and the array disagreeing. The kernel fuzzer gains two twins that put the range forms against the general ones over a materialized range, with the start drawn rather than left at zero because zero is the one value where using it wrongly cannot be seen, and both were confirmed to actually run by breaking `take_range` on purpose and watching it fail on step 0.

One operation had to be told explicitly not to carry labels, and it is worth writing down because the reason generalizes. A group by ends by dropping the groups whose key is null and then sorting the result by the key, and both of those now carry labels, so a grouped frame came back labelled with the permutation of its group ordinals. Rows labelled 9, 3 and 218 say only which group happened to land where in the hash table. A group is not a row of the input and does not inherit a row's label, so `group_by` resets to the default, which is also exactly what `as_index=False` asks for. This was not caught by the library's own tests, which had nothing to say about labels until this change added them. It was caught by the conformance suite, where it turned fifty passing cases red at once, and there is now a test for it in `tests/test_index.mojo`.

Nothing aligns on a label yet. `loc` does not exist, two frames added together do not match their indexes first, and `join` and `concat` still reset to a default range. Those are the `Index` API in https://github.com/tamnd/firepanda/issues/154, and this is the field that had to exist before any of it could be written.

### A reduction behind the parallel prefix folds on the core that made the chunk

`Reduce.process` was one function doing two things: reduce this chunk on its own, then merge that into the running row. The first is the expensive half and it does not need the node, so it is now `partial`, which reads the node and writes nothing to it. The second is `absorb`. `process` is the two of them called in order and behaves exactly as it did.

What that split buys is where the fold runs. The driver hands a batch of chunks out to every core, and if the operator immediately behind the prefix is a reduction, each worker now folds its own chunk into a one row partial before handing it back, while the chunk is still in that core's cache. The driver merges the partials afterwards, which is a pass over one row per chunk. Merging is left to the driver rather than done in the worker because the running row belongs to the node and a worker has no business writing it.

The reason this was worth doing is that there is no parallelism inside one fold to have. A chunk is a hundred and twenty eight thousand rows and a morsel is the same size, so a chunk reduced on its own takes the kernel's serial route. Before this the driver ran every one of those folds on one thread, one after another, after the parallel prefix had already finished. That was the whole of the regression the join node's benchmarks reported at eight million rows.

Measured on an i9-13900K, three blocks, at eight million rows. `exec/pipeline_join_reduce` went from 25.832 to 26.435 ms to 20.709 to 21.465, against a control of `exec/pipeline_join_two_steps` at 24.370 to 24.766 that this change does not touch, so a join followed by a reduction is now faster fused than split at both sizes rather than only at the smaller one. `exec/pipeline_reduce` went from about 24 ms to 15.845 to 16.062 against the same style of control at 24.4 to 24.8. At a million rows `exec/pipeline_join_reduce` is 2.257 to 2.391 ms against 3.427 to 3.540, and `exec/pipeline_reduce` is 1.651 to 1.715 against 2.767 to 2.770.

`exec/pipeline_reduce_only` is unchanged at 11.8 ms, and that is the control that says what the change is: it is a reduction with no prefix in front of it, so the driver has no parallel batch to fold inside and the node still folds on the calling thread.

Four tests in `tests/test_pipeline.mojo`, including a mean, which is the case that makes a two column partial row out of one aggregation.

### A join is an operator in the pipeline now

`Join` holds the right frame whole, builds its key table and buckets it once in `bind`, and after that reads them and nothing else. So one node can be handed to every core at once, the same way a filter is, and a join no longer has to be a break in the pipeline.

What that buys is not a faster probe. Pairing a million rows against a thousand is about half a millisecond and the whole join is about three, so five sixths of a join is the gathers that build its output, and those gathers cost the same here. What changes is what happens to their output. A join done as a whole frame operation writes every column of every paired row to memory and whatever runs next reads it all back, which on a five column join of a million rows is a hundred and sixty megabytes each way for an answer that might be three numbers. Done a chunk at a time in front of a `Reduce`, the chunk that came out of the probe is folded away while it is still in cache and the intermediate is never written at all.

The node does inner, left, semi and anti joins on a single fixed width key column. Right and outer are refused, because both have to emit right rows that nothing matched and that is not known until the last chunk has gone past, which makes them breakers wearing this node's clothes. A composite or text key is refused too, because the ordinal space those need is built by concatenating both sides and having both sides is what a stream does not have. Every refusal happens in `bind`, before a row moves, and a planner that meets one uses `Materialize` and the whole frame join, which is what it did before this node existed.

The projection is on the node rather than left to a `Project` afterwards, for the same reason `join_on` has one: a column that is going to be dropped is not a small waste at the end of a join, it is most of the work.

Nineteen tests in `tests/test_pipeline.mojo` and four benchmark rows in the pipeline section.

The numbers on an i9-13900K, three blocks each, say two things and the second one is a problem this change does not fix. The join itself is faster as an operator than as a whole frame call: at a million rows `exec/pipeline_join_only` is 2.34 to 2.39 ms against `exec/join_frame` at 2.87 to 2.94, and at eight million it is 20.2 to 20.8 against 22.6 to 22.7, with every streamed run below every whole frame run in both. But putting a `Reduce` behind it only wins at the smaller size: `exec/pipeline_join_reduce` is 2.92 to 3.00 ms against `exec/pipeline_join_two_steps` at 3.51 to 3.54 at a million rows, and 25.8 to 26.4 against 24.7 at eight million, where the fused form loses.

The reason is the reduction and not the join. A chunk is a hundred and twenty eight thousand rows and that is exactly one morsel, so the fold of each chunk takes the serial route, and the driver runs it on one thread after the parallel prefix has finished. The whole frame aggregation the two step form ends in has eight million rows in front of it and spreads over every core. So the fused form trades a round trip to memory for a serial fold, and at eight million rows the fold costs more than the round trip saves. Folding inside the parallel prefix, a running row per worker combined at the end, is the next change and is what makes the trade a win at every size.

### The join's code to row table is a value too, and the walk takes a stretch of rows

The other half of the same groundwork. `join_indices` did two things in one function: scan the right side into a table from ordinal to row number, then walk the left side against it emitting pairs. Those are now `bucket_side` and `pair_probe` with a `ProbeTable` between them, and `join_indices` calls them in order.

What the split buys is that the walk takes a stretch of rows rather than a frame. It numbers its output rows from the start of the stretch it was given, so a caller pairing a chunk gets rows numbered within that chunk and shifts them if it wants absolute ones, while the row numbers from the built side stay absolute because that side is one frame whichever chunk is going past. Together with `BuildSide` from the previous change, that is everything a join node needs to hold between chunks.

Three tests in `tests/test_join.mojo` pair a five row side in two pieces against one table and compare against pairing it in one call, over the inner, left, outer, semi and anti kinds, on the bucketed route and on the route taken when the built side's key is unique, and with an empty first piece. An outer join's unmatched rows on the built side are appended once every piece has gone past, which is what the sink would do at the end of a pipeline, and the shared bitmap that remembers them survives the pieces.

No behaviour change and no measurable one. 48 tests in `tests/test_join.mojo`, and the join fuzzer's two million cases still agree with the nested loop twin.

### The join's built table is a value a pipeline node can hold

`align_keys` builds a table over the smaller side's keys and asks it about every row of the larger one, and the two halves of that were already separate functions with the built table passed between them. What they were not is usable from outside the pass that made them, because the table was parameterized on the key dtype, and a pipeline node lives in a `Variant` that has to name every alternative it holds. One alternative per key dtype is not a list anyone can write.

So `BuildSide` carries the key dtype as a field instead of as a parameter, and the one value of that dtype it holds, the key that indexes slot zero of the direct table, is kept as the bits of a `UInt64` and cast back on the way out. The cast truncates to the low bits, so the round trip returns what was put in for every integer key, and a key that is not an integer never takes the direct route and never reaches the field. `build_side` and `probe_side` are the two halves under their own names, and `probe_side` refuses a probe column whose dtype is not the one the table was built from, which is the check the type parameter used to make for free.

Nothing about a join changes. The probe still only reads, which is what let it run on every core already and is also what lets it be called once per chunk with the same table, and the four new tests in `tests/test_join_keys.mojo` are that statement checked: build once, probe in two pieces, compare against probing in one, on the direct route and on the hashed route. The join microbenchmarks on the i9-13900K are where they were, `join/inner_1000` at 2.9 to 3.3 milliseconds over a million rows and `join/indices_1000` at half a millisecond, and the join fuzzer's two million cases still agree with the nested loop twin.

### A reduction over the whole input is an operator now

`Reduce` is a node that folds every chunk that goes past into one row: a sum, a mean, a minimum, a maximum or a count over the whole input, with no key to group by. It is a breaker, since the answer is not known until the last row has been seen, and it is the cheapest breaker there is, because what it holds between chunks is one row per state slot whatever the input was.

It is a separate node rather than a `Group` with an empty key list for the same reason `agg` is a separate method from `group_by`. A group by hashes every row to find out which group it belongs to and there is nothing here to find out, so the reductions read the column straight through and the hashing never happens. The docstring on `agg` puts that at eighty five milliseconds against five on ten million rows.

What it is for is the shape every one of the five db-benchmark join queries has: a join and then a sum. Run as two whole frame calls the join writes ten million rows to memory and the reduction reads them straight back, which on two float columns is a hundred and sixty megabytes out and a hundred and sixty in for an answer of three numbers. Run as a pipeline the reduction folds each chunk into a running row and the chunk is dropped while it is still in cache, so those bytes never leave the core. That is why this comes before the streaming join rather than after: without it the join would have nothing to hand its chunks to and the round trip would come straight back.

The merge is the same trick `Group` uses and it needs no kernel of its own. A chunk's answer and the running answer are both one row, so combining them is a reduction over a column of two rows run with the kind that combines partials, which is the kind itself for a sum, a minimum or a maximum and a sum for the two counts, since merging counts means adding them rather than counting them again. A mean is a sum and a count in two slots with the division held back to the end, because a mean of means is only the mean when every chunk is the same size, and there is a test on chunks of two, three and one rows that gives 3.5 where a mean of means would give 3.833.

Measured on the i9-13900K at a million rows, three blocks, medians. The query is a computed column, a filter that keeps half the rows, and then a sum, a minimum and a count. Fused it is 1.92 ms and split into a pipeline that hands back a frame followed by a reduction over that frame it is 2.77, so the fusing is worth 31 per cent, and every fused run was below every split run. The gap is the round trip: the split route pays 1.11 ms after the filter has finished, and the fused route pays 0.26 over the same pipeline ending in a projection.

Folding a chunk at a time turns out to cost nothing rather than a little, which is not what I expected. `exec/pipeline_reduce_only` is the node on its own over a frame in eight chunks at 0.83 ms and `exec/agg_frame` is `agg` over the same rows in one piece at 0.94, so eight whole column reductions and eight two row merges came out slightly ahead of one pass. Both are within a few per cent of each other and the honest reading is that the per chunk cost is below the noise, not that chunking is faster.

The reductions that do not fold are refused when the pipeline is built rather than when a row arrives. A median of medians is not a median and there is no state short of the values that would make it one, so the answer for those is still `Materialize`. An input that hands over no chunks at all produces no rows rather than a row of nothing, which is what `Group` does with the same input, and a pipeline that needs to match `agg`'s one row there can put a `Materialize` in instead.

Nine tests in `tests/test_pipeline.mojo` and four benchmark rows in the pipeline section.

### The fills step over a NaN, and answer in the dtype's own spelling for missing

`ffill` and `bfill` were the last place in the kernel where a NaN was still treated as an ordinary value, and a fill is the worst place for that to be true, because a fill does not only report what is missing, it copies a value over it. So the mistake is made twice. The row holding the NaN is left standing where pandas would have filled it, and then, worse, the NaN is picked up as the carry and runs forward over every null after it, which turns a gap that had a perfectly good value sitting behind it into a run of NaN. One NaN in the wrong row was enough to destroy an arbitrary stretch of a column. `fill_forward` and `fill_backward` in `kernel/nulls.mojo` now treat a NaN as missing on a float dtype, which fills over it and never carries it.

The other half of this is what a row that could not be filled comes back as, and that turns out to be dtype dependent for the same reason the whole column and the grouped reductions already are. On an int64 column it is a null, which is the only missing that dtype has. On a float column it is a NaN, because that is the only missing pandas has there, so a float column never comes out of a fill carrying a null at all. That is the rule `min` and `max` and `first` and `last` already took, and a fill has more claim on it than they do, since a fill's output is a statement about what is still missing afterwards and it should be spelled the way the dtype spells it.

Where the NaN test goes was measured rather than guessed, and as in the grouped reductions the tidy answer was the wrong one. Correcting a whole validity word in front of the loop keeps the loop unchanged and reads very well, and it was built that way first and thrown away, because it costs a whole extra read of the values: a clean float column of a million rows filled in 682 microseconds against 474 for the int64 column it ought to be level with, which is about half again for nothing. The kept arrangement folds a vector `isnan` into the block copy that already has the values in registers and reduces the lanes once at the end of the block rather than once per vector, and a block that turns out to have held a NaN just falls through to the row loop, which rewrites it. The row loop has the value in hand as well, so it tests there. Built that way the same two columns are 489 microseconds against 489, level to the microsecond, and a column that is mostly NaN got faster as well, 3.310 milliseconds against 3.612. On any dtype without a NaN the whole rule is behind a `comptime if` and compiles away.

Eight tests are added to `tests/test_nulls.mojo` and two benchmark rows, `nulls/ffill_float` and `nulls/ffill_nans`. The fills had no fuzz coverage at all before this, which is a fair part of why the bug lasted, so the kernel fuzzer now compares both of them against `fill_scalar` at four fill limits over the NaN shapes it already draws, checking whether a row is missing before it checks the value, and then asserting separately that neither result carries a null on a float column. That last check was confirmed to actually run by breaking `fill_scalar` on purpose and watching the fuzzer catch it ten steps in.

Four conformance cases close and they are two different bugs, which is worth separating. `basics/bfill` on `float64_no_nulls` was the carry rule: row 0 held a NaN and came back NaN where pandas fills it with the value behind it. `basics/bfill` on `float64_half_null` and `float64_all_null` and `basics/ffill` on `float64_all_null` were the spelling rule, 64 nulls in a float result where pandas has 64 NaNs. basics goes from 141 pass and 36 fail to 145 and 32, and stats and groupby are unchanged, which is what should happen since neither section fills.

What is left of https://github.com/tamnd/firepanda/issues/170 after this is the grouped nlargest and nsmallest, in `group_top_scalar` and `group_top_rows`, which is why `random_column` in the kernel fuzzer still does not draw NaNs globally, and `astype` to a string on a float column, where a NaN does not print the way pandas prints it.

### The parallel prefix hands chunks out by a counter instead of one task each

The driver that runs the front of a pipeline on every core started one task per chunk, and a task costs about ten microseconds to create on an i9-13900K with the creating serial on the calling thread. Sixty four chunks is six hundred microseconds of starting tasks before any of them runs, which on a query that takes two milliseconds is a third of it.

It goes through the morsel queue now, the same one every parallel kernel in the library already uses. That starts one task per worker whatever the batch holds, and each worker takes the next chunk off a shared counter when it has finished the one it had. The batch is four chunks per worker rather than one, so the tasks are paid for a quarter as often, and it is four rather than forty because a batch is held in memory all at once and at a hundred and twenty eight thousand rows a chunk, four per worker on a thirty two thread machine is already sixteen million rows in flight.

The other half of what a queue buys is that a chunk which turns out to be expensive costs the batch one chunk of tail rather than one worker's whole share. A batch handed out up front finishes when its slowest worker does.

`exec/pipeline_line_16k`, which is sixty four chunks through a computed column, a filter and a projection, goes from 1.95 ms to 1.60, with every new run below every old one. `exec/pipeline_line` is the same query in eight chunks and does not move, which is the control that says the gain is the task count and not the queue. The other six rows are all inside a point and a half.

The gate that keeps a projection or a cast off the parallel route stays, and the measurement that says it has to is worth recording. With the tasks down to one per worker a project over sixty four chunks was still forty per cent slower spread out than in a line. At that point the tasks are thirty two rather than sixty four and the remaining cost is close to what thirty two of them cost to start, which puts a floor of a few hundred microseconds under going parallel at all. A query shorter than that is better off on one core, and a projection at a million rows is shorter than that.

### The grouped reductions step over a NaN as well

The whole column reductions learned to treat a NaN as missing in the entry above this one, and the grouped ones did not, so `df.groupby(k).sum()` still took a NaN in and answered NaN for the whole group where pandas answers the total of the rows either side of it. That is the same disagreement as before, only worse, because a single NaN anywhere in a million row column now poisons exactly one group and leaves the rest looking right, which is the kind of wrong answer nobody notices. Every reduction in `kernel/group.mojo` now steps over a NaN the way it steps over a null: `group_sum`, `group_mean`, `group_min`, `group_max`, `group_count`, `group_first`, `group_last`, `group_median`, `group_quantile`, `group_nunique`, `group_var`, `group_std`, `group_sem`, `group_skew`, `group_cov` and `group_corr`.

There was a real bug hiding under this rather than just a difference of opinion with pandas. A group holding nothing but NaN reported `+inf` as its minimum, because `_extreme_core` carries its own seen flags in the output bitmap and a NaN looked to it like a value it had seen, so the group came back marked valid with the identity the reduction started from still sitting in it. It now comes back null, and there is a test named after that case.

Where the NaN test goes was decided by measuring rather than by taste, and the first answer was wrong. The tidy arrangement is to build a corrected bitmap in front of each reduction and leave the loops alone, and it was built that way first and then thrown away, because it costs a whole pass over the values to save nothing: a grouped minimum on a million float rows went from 321 to 491 microseconds, which is about half again, and the reason is that the extra pass is two thirds as much memory traffic as the reduction it is helping. So the test sits in the loop instead, in a small `_there` helper that every core already about to load the value calls, which makes it one compare and no extra traffic. Built that way the same grouped minimum costs about a tenth rather than about a half.

Two of them do not call `_there`. `_sum_core` reads no bitmap at all, and having it start reading one to find the NaNs would give up the thing that makes it fast, so `_addend` turns a NaN into a zero on the way into the accumulator instead, which is exactly the trick the null-is-zero invariant already uses and is free for the same reason. `_count_core` does not call it either, and is the one place in the file that does pay for a corrected bitmap, because a count reads ordinals and bits and never reads a value, so it has no load to reuse and there is no arrangement in which a float count does not grow a pass over the values. That cost is visible and is inherent: `group/count_float` is 432 us against `group/size` at 227 us on the same million rows.

The rest of the measurements, from the build that shipped, on a million rows: `group/sum` at 297 us against `group/sum_float` at 304 and `group/sum_float_nans` at 312, `group/min` at 278 against `group/min_float` at 303, and `group/mean_float` at 663. The int64 rows are the path that did not change and are there as the reference. On any dtype that is not floating point the whole rule is behind a `comptime if` and compiles away, so an int64 or a uint32 grouped reduction has no such instruction in it, and there is a test that reads an int column holding the bit pattern of a float NaN and confirms it is still counted and summed as the ordinary number it is.

The kernel fuzzer now runs its grouped comparison twice, once over the column it always drew and once over a column with NaNs in it, at the four shapes the whole column fuzzer already used: NaN only, null only, both, and nothing but NaN. That matters more here than it did there, because what the grouped kernels are checked against is `group_scalar`, the obvious version that walks one row at a time, and the two agreeing is only worth something if the fuzzer is drawing the values that would make them disagree. The full million case run passes. `random_column` still does not draw NaNs globally on purpose, because the grouped nlargest and nsmallest paths have not learned the rule yet and are a separate piece of https://github.com/tamnd/firepanda/issues/170.

Twelve tests are added to `tests/test_group.mojo` and five benchmark rows to the group section.

Six conformance failures close, and four of them were not the ones this set out to fix. `reduce_any` has a fast route for the reductions that can be done in one pass and sends everything else to `aggregate_group_any` over a single group, and the order statistics are all in the everything else: a whole column median, quantile or nunique is a grouped one with one group in it. So teaching the grouped cores the rule taught the whole column order statistics the same rule at the same time, and `basics/median` on `float64_no_nulls` and `float64_half_null`, `stats/median` on the same two, `stats/quantile-linear` on `float64_no_nulls` and `stats/quantile-nulls` on `float64_half_null` all pass now. basics goes from 139 pass and 38 fail to 141 and 36, and stats from 48 and 8 to 52 and 4.

No groupby case closes, and that is worth saying plainly rather than leaving it to be noticed. All 62 of them fail on something else: 59 because the result carries no index name where pandas carries the key's, which is https://github.com/tamnd/firepanda/issues/154, and 3 because the Arrow reader will not read a dictionary encoded column, which is https://github.com/tamnd/firepanda/issues/159. None of them was ever a NaN. The grouped NaN bug was real and was found by reading the code and by the fuzzer rather than by the conformance suite, which is a fair description of where the suite is useful and where it is not.

What is left of https://github.com/tamnd/firepanda/issues/170 after this is the fills, where `ffill` and `bfill` still carry a NaN forward as though it were a value instead of filling over it, and the grouped nlargest and nsmallest.

### The reductions step over a NaN too

Making a NaN missing when something asks whether a row is missing, which shipped in 0.6.42, did not make it missing to the code that reads the values. So a sum or a mean or a minimum still took the NaN in and came back NaN where pandas gives the answer sitting in the data. `isna` and `sum` disagreed with each other about the same row, which is worse than either of them being wrong on its own. `sum_of`, `mean_of`, `min_of` and `max_of` now step over a NaN on a float dtype, and so does everything `reduce.mojo` routes through them, which is `Series.sum`, `Series.mean`, `Series.min` and `Series.max` and the frame forms of all four.

The rule is applied in the loop rather than in front of it. A sum turns a NaN into a zero on the way into the accumulator, which is the same trick the null-is-zero invariant already uses and is free for the same reason: zero is the identity for addition. A minimum turns it into the identity the reduction started from, which loses every comparison it is in. Both are one compare and one select per vector and both are behind a `comptime if`, so an int64 or a uint32 or a string column has no such instruction anywhere in it and the loop is the loop it was.

The mean needed one more thing, because it wants the divisor as well as the total, and asking for the count separately would mean a second pass over the values. `_sum_range` counts the NaNs it stepped over and hands that back with the total, and `mean_over` subtracts it from the count of set validity bits it was given. Subtracting the one from the other cannot take the same row away twice, because a null holds a zero and a zero is not a NaN, so every NaN the sum stepped over was in a row whose bit was set.

The minimum has the one case that is genuinely awkward. The vectorized path used to take a full validity word as proof that it had seen a value, and that is no longer true, since a word can be full and every value under it can be a NaN. It cannot be recovered from what the lanes folded to either: a block of nothing but NaN folds to the identity, and so does a block holding one real infinity, and those two have to give different answers. So the lanes that held something real are accumulated on the side, one vertical OR per vector and one reduce per block, and there is a test for each half of that pair.

Five benchmark rows are added rather than argued about. The existing `kernel/sum_*` and `kernel/min_*` rows are int64 and measure the path that did not change; the new `kernel/sum_float`, `kernel/sum_nans`, `kernel/min_float`, `kernel/min_nans` and `kernel/mean_float` are the float shape of the same work, with `_nans` holding one NaN in eight. Across three runs on the machine used here every one of them lands inside the band the unchanged int64 rows occupy, between 179 and 567 us on a million rows, and the run to run spread on this machine is larger than anything the change could be costing.

The fuzz harness grew a column generator that draws NaNs, which it deliberately did not have before, and a check that runs the four reductions against their scalar twins over it at four shapes: NaN only, null only, both, and nothing but NaN. The twins learned the rule in one place, `_is_there`, so that the thing the fast kernels are checked against is still the obvious version written out one row at a time.

Two conformance failures close, `basics/sum` and `basics/mean` on `float64_half_null`, which holds one NaN and one infinity on top of its nulls and is the frame that made this visible. basics goes from 137 pass and 40 fail to 139 and 38. This is not all of https://github.com/tamnd/firepanda/issues/170: the grouped reductions still count and average a NaN, and the order statistics still sort one, and both are next.

## [0.6.42] - 2026-09-05

Built against Mojo 1.0.0 (ed45d567). The engine's driver stops being the serial part of a parallel library, several reductions get their arithmetic corrected, two more grouped reductions arrive, and one more allocation that was being zeroed before it was filled is not any more.

The change with the numbers behind it is the pipeline driver. The kernels underneath the engine have used every core for a long time, but the driver that pushes chunks along a line of operators ran on the thread that called it, so a filter run through the engine used one core where the same filter over a whole frame used sixteen. It now runs the elementwise operators at the front of the line on every core and keeps the rest of the line in chunk order, which is worth two thirds off a computed column and a filter and a projection over a million rows.

The rest is correctness. A variance or a standard deviation computed the two pass way was correcting against the wrong centre, an empty float reduction answered null where pandas and numpy answer NaN, and a mean over an integer column was computed from a sum that had already wrapped. None of those are edge cases nobody would hit, and all three came out of running the conformance suite rather than out of reading the code.

### The front of a pipeline runs on every core

Until now a pipeline ran one chunk at a time on the thread that called it, which meant the engine's own driver was the one part of the library that did not use the machine. The kernels underneath it were already parallel, so a filter over a whole frame used all sixteen cores and the same filter run as a pipeline used one.

The change is that the driver looks at the front of the line and asks how many operators in a row produce their output from their own input and write nothing to themselves. A filter, a projection, a computed column and a cast all qualify: give one of those a chunk and it hands one back, and it does not care what it was given before or what it will be given next. So one operator can be shared by every worker with no copy and no lock, which is what the four `process` methods becoming non mutating is for, and what `node_apply` is: a way to run a node on a chunk while only borrowing it. A group by or a limit does not qualify, because both hold something between chunks.

The driver runs that prefix a batch at a time, one chunk per worker and no more, then pushes the survivors through the rest of the line in chunk order on the calling thread. Taking a batch rather than the whole source is the point: running every chunk through the prefix first would hold the entire intermediate result in memory, which is the thing a chunked engine exists to avoid. Keeping the tail in chunk order is what makes this the same execution as before rather than an approximation of it, so the rows come out in the order the source had them and a group by downstream sees exactly the chunks it used to.

A pipeline with a limit anywhere in it stays on the old route. A limit is the only operator that can say it is finished before its input runs out, so reading a batch ahead on behalf of sixteen cores would read rows the limit was about to make unnecessary.

The second gate took the measurement to find. Spreading the work costs a task per chunk, and tasks are created on the calling thread one after another at roughly ten microseconds each, so the prefix has to have something for the other cores to do or the tasks are the only thing that got added. A filter and a computed column evaluate something per row and get faster on more cores. A projection only rebuilds a chunk out of columns it already has, and a cast walks a column through the allocator, so both are waiting on memory rather than on arithmetic and neither gains anything. Measured on the i9-13900K, a project on its own over sixty four chunks took twice as long spread out as it did in a line, and over eight chunks it was still slower. So the driver takes the parallel route only when the prefix contains a filter or a computed column.

On that machine, at a million rows, over a line of a computed column then a filter then a projection: `exec/pipeline_line_16k` 5.59 ms to 1.94 and `exec/pipeline_line` 5.26 to 1.66, both with every new run below every old one across three alternating blocks. `exec/pipeline_project`, `exec/pipeline_project_128k`, `exec/pipeline_cast` and `exec/pipeline_cast_128k` all take the sequential route now and all four land inside the noise, between minus 0.2 and plus 1.8 percent. `exec/pipeline_line_one_chunk` and `exec/pipeline_limited` are the controls and did not move.

Handing tasks out one per chunk is not the last word. A shared counter that gives each worker several chunks would pay for the tasks once per core rather than once per chunk, and would let the projection and the cast back onto the parallel route. That is a later change and it needs the batch to stay bounded.
### A NaN is missing when you ask whether a row is missing

The other side of the same question. pandas on the numpy backend has no separate presence bitmap for a float column, so NaN is the only missing it has there, and `Series([1.0, nan]).count()` is 1. firepanda followed Arrow, where a NaN is an ordinary float that happens to compare false against itself, so it answered 2. `isna` said False on a NaN, `notna` said True, `dropna` kept the row, and `hasnans` said a column with a NaN in it had none.

Those all agree with pandas now. `is_null`, `is_not_null` and `all_valid_mask` in `kernel/nulls.mojo` read the values on a float dtype and take a row as present when its validity bit is set and its value is not NaN. Every other dtype is untouched and reads nothing, because there is no NaN to find in an int64 column and the bit pattern that would be one is an ordinary number.

`Series.null_count` counts a NaN and `Array.null_count` does not, which is the line between the two halves of the library rather than an oversight: an `Array` is Arrow and says what is in the buffers, a `Series` is pandas and says what pandas would say. `Series.drop_nulls` and `DataFrame.drop_nulls` go through the pandas one, so they drop a NaN row. The values buffer still holds the NaN and still writes it back out, so an Arrow round trip is still a round trip and only the answer to "is this missing" changed.

The cost is a read of the values buffer where there used to be none, and it is arranged so the common shape pays as little as possible. A bitmap word covers sixty four rows, those rows are scanned with a vector `isnan` and a reduction, and only a word that turns out to hold a NaN is taken apart bit by bit. Four benchmark rows measure it, three of them new. On a million rows on the machine used here, whose interquartile ranges on this section run from ten to fifty percent so these are ranges rather than numbers: `nulls/is_null` on an int64 column with no nulls, which is the path a float column used to take, is 79 to 188 us across runs. `nulls/is_null_float` on a float column with no NaN is 237 to 560. `nulls/is_null_nans` at one NaN in eight, which is a worse column than anything real, is 691 to 917. `nulls/count_float` is 197 to 276 against a subtraction of two known numbers before, which is the one row where this is a regression rather than a cost inside something that was already walking memory.

This is the first half of https://github.com/tamnd/firepanda/issues/170 and it does not finish it. The reductions still add a NaN in rather than stepping over it, so a sum or a mean or a median over a column with one NaN in it is still NaN where pandas gives the answer sitting in the data, and a grouped `count` still counts a NaN as a value. Those are the inner loops of the operations this library exists to make fast and they are argued separately. Seven conformance failures close here, six in basics and one in statistics, and the pandas against pandas oracle is unchanged at 4000 for 4000.

### A float reduction with nothing to reduce answers NaN and not null

pandas has no separate presence bitmap for a float64 column. NaN is the only missing it has there, and `isna` on a float column is a NaN test. firepanda has a bitmap for every dtype, so it had a second way of saying missing, and until now the grouped and whole column reductions that answer in float64 used it. A mean of a group with no values came back null. So did a variance or a standard deviation or a standard error with fewer values than degrees of freedom, a skewness of fewer than three, a median or a quantile of nothing, and a correlation with no complete pairs or with a column that does not move.

Every one of those is NaN in pandas, in a row pandas considers present. The difference was visible at the API boundary and it is the kind that does not announce itself: a caller who asks `isna` gets True from both spellings, and a caller who does arithmetic on the result gets NaN out of the pandas one and a zero behind a cleared bit out of firepanda's. Nine reductions changed and they are `mean`, `var`, `std`, `sem`, `skew`, `median`, `quantile`, `corr` and `cov`.

`min`, `max`, `first` and `last` are a different case, because they keep the column's own dtype rather than widening to float64, so which spelling of missing they use is the column's decision and not the reduction's. Over a float64 column they now answer NaN in a valid row like everything above. Over an int64 column they still answer a null, because there is no NaN available to put in the slot and a null is the only spelling there is. That is one rule and a dtype that does not have both options, not two rules. `sum` and `count` still answer zero and `size` still counts rows including nulls. The full policy is written out at the top of `firepanda/kernel/group.mojo`.

One thing got simpler on the way. The parallel quantile cut the group range into pieces and needed the cut to land on a byte of the output validity bitmap, so that two workers marking empty groups did not clear each other's bits in a shared byte. Nothing marks anything now. Each worker writes a NaN into its own float64 slot and the constraint is gone, though the cut is left where it was because the test that would have caught a bad one is the only place the empty group path runs on more than one worker.

This closes nineteen conformance failures, eight of them in the statistics section and eleven in basics. The last two of those were what caught the `min` and `max` case: the suite reported `null, expected nan` on an all null float64 column long after the float64 valued reductions had been fixed, which is what a rule written as "answers in float64" rather than as "answers in a float dtype" looks like from the outside. It is half of https://github.com/tamnd/firepanda/issues/170, the half about what a reduction hands back. The other half is NaN as missing on the way in, where pandas steps over a NaN in a column exactly as it steps over a value that was never there and firepanda adds it in, and that half has a per row cost in the inner loop so it is argued separately.

### `Series.argsort` returns int64, and the kernel keeps its uint32

The permutation a sort produces is uint32 here, one value per row, rewritten in full on every pass, and it is the largest temporary in the sort path. pandas returns int64 from `Series.argsort`, because numpy's `argsort` returns the platform index type and pandas uses a negative position as its sentinel for a row that is not there. firepanda has no such sentinel, a null is placed by `nulls_first` rather than removed and marked, so the sign is dead weight and on a ten million row sort the wider form is eighty megabytes of ordinals rather than forty.

So the two do not have to agree everywhere, and now they do not. The kernel is unchanged. `Series.argsort`, which is the one place a user reads the permutation rather than a sort consuming it, widens on the way out, which is one pass over an array the sort has already been over many times. `Series.sort_values` calls `argsort_any` directly and pays nothing. `DataFrame.argsort` is unchanged and stays uint32 too, because pandas has no `DataFrame.argsort` and there is nobody to match.

Found by the conformance suite, which reported `dtype uint32, expected int64` on `stats/argsort` for two corpus frames with the values correct in both. One of the two passes completely now. The other turned out to be hiding something more interesting behind the dtype: `keys_10` has one duplicated value in ten thousand rows, firepanda's sort is stable and puts the earlier row first, and pandas' default `argsort` is a quicksort that puts the later one first. `Series.argsort(kind="stable")` in pandas gives firepanda's answer exactly. Nothing is changed here for that, because pandas documents its default as not stable and a stable answer is one of the answers an unstable sort is allowed to give.

### The join's ordinal list is not zeroed either

The same thing as the gather, one layer up. A join gives both sides one list of ordinals, one entry per row of the left side followed by one per row of the right, and it allocated that list zeroed. Nothing reads a zero out of it: the build side writes its own stretch and the probe side writes the rest, between them every slot. So the zeroing was a pass over both sides that only cost, and since the probe half is written by every core at once, it was also the pass that put most of the list on one core before the others wrote to it.

The one slot the build did not write is a build row whose key is null, on the grounds that nothing reads it. It is written now, because a slot nobody writes holds whatever the allocator last left there, and one store on a rare path is cheaper than that being true.

`join/indices_1000`, which pairs ten million rows against a thousand and gathers nothing, goes from 1.18 ms to 0.57. That row is almost entirely the ordinal list, which is why it halves. `join/inner_1000` 3.59 ms to 3.08 and `join/inner_100k` 3.76 to 3.56. On db-benchmark at 0.5GB, j1 20.6 ms to 19.4, j2 20.6 to 19.5, j4 56.6 to 54.5, with the checksums unchanged.

`join/inner_projected` went the other way, 1.77 ms to 1.93, consistently rather than on the median. It allocates the same list as the rows that gained and there is no reason it should differ, so it is recorded rather than explained, in the same box as the two string rows that moved in 0.6.41 without being touched.

### The standard error and the skewness are grouped reductions now

`AggKind.SEM` and `AggKind.SKEW`, with `group_sem` and `group_skew` beside `group_std` and `group_var`, reachable through `DataFrame.group_agg`, `DataFrame.group_by` and the erased `aggregate_group_any`, and through `reduce_any` for the whole column form because anything without a fast route there already falls through to the grouped one. They came out of the conformance suite: with them the flat groupby section runs 48 for 48 against pandas and without them it ran 44, and the four misses were not disagreements about an answer, they were this library not having the reduction at all.

Neither is built on the raw moments. The standard error is `_var_core` asking for the standard deviation and one division, because there is no numerical reason to walk the column twice for it, and the count under the root is the plain non-null count rather than the corrected divisor, which is what pandas does: the degrees of freedom belong to the variance and applying the correction twice is a different statistic. The skewness takes a second pass of its own for the reason the variance already takes one, and more so, since the third moment is the difference of larger quantities than the second and the one pass arrangement loses correspondingly more of it. The coefficient is the adjusted Fisher Pearson one pandas reports, which carries two edges that are pandas' answers rather than choices made here: a group of fewer than three values has no skewness, because the adjustment divides by `n - 2`, and a group whose values are all the same has a skewness of zero rather than a null, because the shape of a constant is symmetric rather than undefined.

The naive twin in `scalar.mojo` learned both, so the kernel fuzzer checks them against a readable loop rather than against nothing. Its rotation over the reductions is written out as a list now instead of counted from zero, because the single column kinds are no longer contiguous in the code space and a rotation over raw codes would have handed the single column path a `CORR` or a `COV`.

### The two pass moments correct for the centre they missed

`group_var`, `group_std`, `group_sem` and `group_skew` all work by taking each group's mean and then walking the column a second time to accumulate deviations from it. That second pass is what keeps a variance from being computed as the difference of two enormous numbers, and it has been there since the spread reductions were written. What it does not fix is that the centre it subtracts is a sum divided by a count, which is near the mean but is not the mean, so the deviations are measured from slightly the wrong place. That error does not cancel across the group the way rounding in a sum does. It biases every term in the same direction and it grows with the magnitude of the values.

The fix is one more accumulator. Alongside the squared deviations each worker now sums the plain deviations, which come to zero when the centre was exact and to the size of the miss when it was not, and the final loop subtracts the square of that residual from the second moment and the matching two terms from the third. It is the same algebraic identity the one pass form uses, but here it is applied to numbers that are already small, so there is nothing large left to cancel and the answer stops depending on how accurate the mean was.

The size of it, on five values spaced 1, 2, 4, 8 and 16 apart sitting at two to the fifty second, where every input is exactly representable and the centre is the only thing that can go wrong: the variance was 37.25 against a true 37.2, and the skewness was 1.4863469519931585 against a true 1.3253147098134046, which is twelve percent out. Both are exact now. On a more realistic column, four thousand int64 values near 4.6e18 with a spread of 4.2e9, the skewness was 0.15 percent from the true value and is now within 1e-13 of it. There are two new tests, one per reduction, that shift a column by a large constant and require the answer not to move, which is a property a variance should have and did not.

This is a place where being right and matching pandas are not the same thing, so it is worth saying plainly which one this is. pandas computes these moments the uncorrected way, and on that five value column at two to the fifty second pandas answers 37.25 and 1.4863469519931585, the values this library used to produce. So on small groups of very large values firepanda now disagrees with pandas by more than a rounding error, deliberately, because the alternative is to keep an answer that changes when you add a constant to the column. On groups large enough for the summation order of the mean to matter, which is the ordinary case, the correction moves firepanda towards pandas rather than away: on that four thousand value column pandas is 0.015 percent from the truth and firepanda used to be 0.15 percent, ten times worse, and is now closer to the truth than pandas is.

The cost is one addition per row and one float64 table of group by worker size, in a loop that already does a load, a subtract, a multiply and a read modify write. It does not show up on `group/std_sparse`, but that is a statement about the measurement rather than about the change: five interleaved runs of fifteen repetitions each gave 3.59 ms for the old kernel and 3.04 for the new one, with interquartile ranges between ten and fifty six percent, so the machine used here cannot resolve a difference this size on that row and the honest reading is that the second accumulator is somewhere under the noise floor. The row is memory bound and the added work is one register add, so that is the expected place for it to land, but it is recorded as unmeasured rather than as free.

The kernel fuzzer runs a million cases against the naive twin without a disagreement, which is the expected result rather than a surprising one: the fuzzer's columns are small enough that the correction is worth nothing on them, which is exactly the claim being made about well conditioned data. The twin in `scalar.mojo` is deliberately left uncorrected, because its job is to be the obvious loop that the clever one is checked against, and a twin that has learned the same trick checks nothing.

### Fixed

The first two bugs the pandas conformance suite found. There is now a Mojo driver in firepanda-compat that runs a case against this library and hands the answer back as Arrow, so the suite compares firepanda to pandas rather than pandas to itself, and the first thing it did was produce twenty four failures on the `basics` section alone. These are two of them, and they are the two that are unambiguously wrong rather than a difference of opinion about semantics.

#### A mean is no longer computed from a wrapped sum

`Series.mean`, `DataFrame.agg` and `groupby.mean` over an integer column added the values up in int64 and divided that. An int64 sum wraps, which is the right answer for a sum because numpy wraps there too, but pandas converts to float64 before dividing and so the mean of a column of large integers came out as whatever the wrap happened to leave behind. On the conformance corpus the mean of the int64 column was 2040395725.875 where pandas gives -4.611686016386992e+18, which is not a rounding difference, it is eighteen orders of magnitude and the wrong sign.

The accumulator is a parameter now, in both the whole column path and the grouped one, defaulting to the natural widening. `sum` keeps that default and keeps wrapping. `mean` asks for float64 and divides that. Nothing else changed and no other reduction moved.

#### A filter or a slice that keeps no rows produces a usable frame

A column of no rows was a column of no chunks, and `only()` raises on that shape, which is the borrow every kernel written against `AnyArray` reaches through. So `df.filter(mask)` where the mask happened to match nothing, or `df.head(0)`, or `df.slice(3, 3)`, produced a frame that could not be written to Arrow, aggregated or printed. It raised with a message about a column having zero chunks, which points at the writer rather than at the filter that made it.

The rest of the package already answered this the other way. `take` with no indices, the Arrow reader on a file with no record batches, and `ChunkedArray.combine` with nothing to combine all produce one empty chunk. `filter_chunked` and `slice_chunked` were the two that did not, and they do now. The empty chunk is made by slicing an existing chunk to nothing rather than by building an array, which is what keeps it correct for a type with children: an empty string column is not an empty buffer, it has an offsets buffer and a data buffer of its own, and the way to get an empty one of the right shape is to ask a full one for none of its rows.

A test asserted the old behaviour, as an observation rather than as a claim it was right. It asserts the new contract now, and it also asserts that the empty column can be borrowed, which is the property that was actually broken.

#### A negative head or tail counts from the other end

`head(-2)` in pandas means every row but the last two, and `tail(-2)` means every row but the first two. Both of them returned nothing here, on frames and on series alike, because the count was clamped into the valid range before being used and a negative count clamped to zero. On a ten thousand row frame the conformance suite asked for `head(-2)` and got zero rows where pandas gives 9998, which is not an edge case anybody would notice from the type signature and is exactly the kind of thing a conformance run is for.

The clamp is replaced by two functions that know which end they are working from. `_head_end` turns a count into the row to stop at, and `_tail_start` turns one into the row to start at, and both of them handle a negative count by counting from the other end and then clamping. A negative count larger than the frame still gives nothing, which is also what pandas does. The positive path is unchanged, including the clamp that keeps `head(10)` on a three row frame from raising.

## [0.6.41] - 2026-09-05

Built against Mojo 1.0.0 (ed45d567).

Three changes to the join, and between them they are the largest step this library has taken on a benchmark. On db-benchmark at 0.5GB on an i9-13900K, holding the driver fixed and alternating two builds of the library, j1 goes from 44.1 ms to 21.3 and j4 from 83.3 to 60.5. Against the current pandas 3.0.5, polars 1.44.1 and duckdb 1.5.5, all four engines run in one session and all four agreeing on the checksum for every query, firepanda is ahead of polars on fourteen of the fifteen queries and ahead of duckdb on nine, two of those nine by a margin small enough to call level. It was 2.1x to 3.3x behind duckdb on j1, j2 and j3 before this.

The one worth understanding is the smallest. `take` allocated its output with the zeroing constructor and then wrote every element of it anyway, so a memset ran in front of every gather. The memset itself is a pass over eighty megabytes, but the bigger cost was that it runs on one thread and is the pass that faults the output's pages in, so the gather that followed had thirty two cores writing into memory that belonged to one of them. Writing the zero inside the gather instead is worth 26 to 48 percent on every take and join row in the microbenchmark.

The other two are about doing less. A join takes the list of output columns it is wanted to build, so a column the query is going to drop is never gathered, and since three quarters of a join is building the output rather than deciding which rows go together, that is most of the query rather than a tidy-up at the end. And the build side of a join is a value now rather than a phase inside one function, which changes nothing today and is what a streaming join needs, because a streaming join builds once and probes with every chunk that arrives.

There is also a note on measurement in here that cost more than any of the code did. Three separate A/B runs said a change cost between three and eight percent when it cost nothing, because each ran the old variant then the new one in each pair while the machine drifted slower over the hour. Building both binaries up front and alternating them old, new, new, old is what makes a linear drift cancel, and it is how everything above was measured.

### A gather no longer zeroes the column it is about to fill

`take` allocated its output with the zeroing constructor and then wrote every element of it, including a zero where the index says null. So there was a memset of the whole output column in front of the gather. It cost what a memset of eighty megabytes costs, which is not nothing, but that was the smaller half of it. The memset runs on one thread, and it is the pass that faults the output's pages in, so every page of the output arrived on whichever core ran the allocation. The gather that followed then ran on thirty two cores writing into memory that all belonged to one of them. Allocating with `overwritten` and writing the zero inside the loop puts the first touch on the worker that is about to fill the page.

On an i9-13900K at a million rows, medians over twelve runs of two prebuilt binaries alternating old, new, new, old. `kernel/take_scattered` 686 us to 505, `frame/take` 2.46 ms to 1.67, `join/inner_1000` 5.74 ms to 3.00, `join/inner_100k` 7.04 to 3.72, `join/left_1000` 6.88 to 3.56, `join/inner_projected` 2.56 to 1.78, `join/outer` 15.69 to 12.42, `join/two_keys` 7.20 to 5.25, `join/semi` and `join/anti` both 2.53 to 2.12. Every old run of every one of those rows is slower than every new run of it.

The control rows are the ones that say this is the gather and not the weather. `join/indices_1000` does the pairing and none of the gathering and moves 0.3 percent. `strings/take` and `text/take_text` go through the string builder rather than through this loop and move under three percent.

End to end on db-benchmark at 0.5GB, memory mode, ten runs, the same alternation: j1 from 44.1 ms to 21.3 and j4 from 83.3 to 60.5, with the checksums unchanged. The parallelism the harness records for j1 goes from 9.9 to 21.6, which is the same statement from the other side.

It also explains something that had no explanation. A j1 that kept two output columns was slower than the same j1 keeping three, 42.2 ms against 25.9, which is not a thing a serial loop over the output columns can do. The two column output has two eighty megabyte float64 outputs and the three column one has those plus a forty megabyte int32, and where those memsets land relative to each other decides how much of the gather runs against pages owned by another core. With the memsets gone the two are 21.3 ms and 21.1, which is the ordering the work says they should have.

`filter` had the same shape and gets the same treatment, though less of it, because the compaction loop is serial and only the memset itself is saved. `kernel/filter` 1.54 ms to 1.34, `kernel/filter_sparse` 2.28 to 2.16, `frame/filter` 4.53 to 4.35. `kernel/filter_twin` is a different kernel and does not move.

Two rows moved that this cannot have touched. `strings/filter` and `text/filter_text` both go through the string compaction, which is untouched code, and both are about ten percent slower, consistently across every run. That is code layout, and it is recorded here rather than explained.

### A join builds the columns it was asked for and no others

`DataFrame.join` and `DataFrame.join_on` take an optional list of output column names, and a column not on the list is never gathered.

The reason this is worth an argument is what a join costs. Pairing ten million fact rows against a thousand dimension rows takes 7.9 ms on an i9-13900K, and the whole join takes 31.4, so three quarters of a join is not deciding which rows go together, it is building the output. A column the caller is going to drop is therefore not a small waste at the end of the query, it is most of the query, and there is no optimizer to notice that until M4.

The alternative available before this was to narrow the inputs first, which works and which the benchmark driver did, but `select` copies the columns it keeps, so avoiding three gathers of ten million rows cost a copy of two columns of ten million rows. Naming the wanted columns costs nothing at all.

Ten million fact rows against a thousand dimension rows on an i9-13900K, keeping two of the four output columns, is 18.44 ms against 30.17 for the same join keeping all four. Pairing is 7.75 ms of both, so the gather went from 22.4 ms to 10.7, which is half the columns costing half the gather. The other eight rows of the join benchmark are unchanged to within a percent, which is the check that the restructure this needed, planning the whole output before building any of it, costs nothing when there is no projection.

The names are the output's names and they are worked out exactly as they would be without a projection, including the suffix a colliding right column gets, so asking for a column does not change what it is called. They are also built in the order they were asked for. A name the result does not have, or one asked for twice, is an error rather than a silent difference in shape.

### The join's build side is a thing now, instead of a phase

Nothing about a join changes with this. It is the shape of the code that changes, and it is groundwork for the streaming join in M2b.

The key alignment used to build a table over the smaller side and probe it with the larger side in one function, which is right for a whole frame join, because such a join builds once, probes once and throws the table away. A streaming join cannot work that way. It builds once and then probes with every chunk that arrives, so the table has to outlive the pass that filled it. `_build` now returns the table and `_probe_into` reads it, and the whole frame join is those two called one after the other.

Both routes, the one indexed by the key value and the one indexed by its hash, get a probe function each. One loop behind a flag was tried first and it measured slower, but that measurement did not survive the protocol described below, so the two shapes have not actually been told apart. They are kept separate because that is the shape the hot loop wants, not because there is a number saying so.

Measured on an i9-13900K at ten million rows, twelve runs of two prebuilt binaries alternating in an old, new, new, old order so that a machine drifting over the session cancels out rather than landing on whichever variant ran second. Pairing is 8.35 ms before and 8.27 after, the full inner join 31.69 and 31.75, the projected one 19.43 and 19.59. That is a wash on every row, which is the whole claim.

The protocol is worth writing down because the first three attempts at this measurement said the change cost between three and eight percent, and it does not. Each of those ran old then new in each pair, rebuilding in between, and the machine was getting slower over the hour, so the later half of every pair was penalised. The alternation above is what makes the drift cancel, and building both binaries up front is what stops a rebuild sitting between the two things being compared.

## [0.6.40] - 2026-09-04

Built against Mojo 1.0.0 (ed45d567).

One feature, and it closes the last gap in db-benchmark. q8 asks for the two largest `v3` in each `id6`, and it was the one query in that suite firepanda skipped, because answering it needs a top n per group and there was no kernel for that. There is one now, `group_top_rows`, with `DataFrame.group_nlargest` and `DataFrame.group_nsmallest` on top of it, which is also the pandas spelling and has been on the parity list since M6.

It does not sort. The obvious implementation sorts the frame by the value column and walks each group taking the rows it sees first, which costs a full sort of every row, a permutation of the column and a pass to undo it, to answer a question about two rows in a hundred. This keeps a small table instead: `n` slots per group, with the worst value currently in those slots held in an array beside them, so the common path for a row is one load and one compare and a row that loses is never touched again. The slots themselves are only read when a row is going to be kept.

Ties break by row number rather than by whichever core saw the row first. That makes the comparison a total order over distinct rows, which is what lets the answer be the same whether the column was scanned by one core or by thirty two, and it is also the row pandas keeps. Nulls are not candidates and neither is NaN, because a NaN loses every comparison it is in and one left sitting in a slot would hold a real value out.

Each worker fills a private table over a contiguous stretch of rows and a parallel fold over blocks of groups merges the others into the first, reusing the same insert the scan uses, so the merge is correct for the same reason the scan is rather than for a second reason that would need its own argument. The worker count is capped against a 64 MB budget for those tables, and below 65536 rows the whole thing takes one worker and skips the fold.

Ten million rows and a hundred thousand groups on an i9-13900K is 12.82 ms. The version of this that zeroed its buffers up front, the way an ordinary `Buffer` does, was 14.83. None of the four buffers is read before it is written, so none of them needs the memset, and the per group counts and thresholds are prepared by the worker that owns them rather than by one thread in advance. At a thousand groups the buffers are small enough that this was never the cost and the two numbers are the same.

The end to end number is the one that matters. On db-benchmark q8 at 0.5GB, memory mode, ten runs on a quiet machine and repeated twice with the two runs agreeing, firepanda is 0.037 s against DuckDB's 0.071, Polars' 0.205 and pandas' 2.759. Peak resident set is 0.95 GB against 2.34, 1.34 and 1.65. All four engines return 200,000 rows and all four checksum identically, which is the bench harness's cross engine agreement check confirming that a new kernel is not quietly keeping the wrong row on a tie.

Half of that end to end number turned out not to be in the kernel at all. The driver first narrowed to the two columns the query reads and then took the rows, which is what pandas does, and `select` copies the columns it keeps, so narrowing first copied twenty million values to answer a question about two hundred thousand. Narrowing after the take was 37 ms against 75 for the same answer. That is a bench repository change rather than a library one, but it is the kind of thing worth knowing about a library whose `select` copies.

### Added

- A per group top-n kernel, `group_top_rows`, and the two frame spellings on top of it, `DataFrame.group_nlargest` and `DataFrame.group_nsmallest`. It keeps `n` slots per group and compares each row against the worst thing currently in its group's slots rather than sorting the frame, so a row that loses is never touched again. Ties are broken by row number, which makes the answer the same whichever way the rows were split across cores and the same answer pandas gives. Nulls and NaN are never candidates.

## [0.6.39] - 2026-09-04

Built against Mojo 1.0.0 (ed45d567).

Two releases ago every elementwise kernel started running on all thirty two cores. This one finishes the job on the kernels that were left, which were the ones that did not have the shape of an elementwise kernel and so could not be handed to the scheduler unchanged.

The whole column reductions were the larger of the two. A sum or a minimum produces one answer and not one answer per row, so it does not split the way an addition splits. Each morsel now reduces its own rows into a slot of its own and a serial loop combines the slots at the end, over the number of morsels rather than the number of rows, and a minimum carries a second slot per morsel saying whether that morsel saw a value at all so that a morsel of nothing but nulls can be skipped instead of contributing the identity. At ten million rows on an i9-13900K a sum went from 2.775 ms to 0.522 and a minimum over a column that is one in seven null from 5.153 to 0.672.

The null kernels were the smaller one. `is_null`, `is_not_null` and `coalesce` split cleanly and had simply not been done yet, and `coalesce` was also walking the column twice, once to copy and once to look for the gaps. It now does both over the rows one worker was handed. `is_null` went from 419 microseconds to 88 and `coalesce` over a sparse column from 19.947 ms to 4.463.

The reductions came with a correctness problem worth naming, because it was not found by a test failing. Every reduction test in the suite ran at three hundred and one rows, which is under the morsel size, so the parallel branch had never been executed by anything. The measurement is what gave it away: half a millisecond for eighty megabytes looked faster than the machine can read from memory, which turned out to be explainable by L3 residency but was worth an afternoon of doubt. Four tests across the two changes now run past the split and check every row against a single threaded reference.

The directional fills stay on one thread and are not an oversight. Row `i` takes the nearest present value before it, a dependency reaching back an unbounded distance, and splitting that needs a different algorithm rather than a different loop.

### Changed

- Whole column reductions run on every core. `sum_over` and `extreme_over` were the last kernels reading a whole column on one thread, and everything built on them was too: `sum_of`, `min_of`, `max_of`, `mean_of` and `mean_over`. A reduction does not split the way an elementwise kernel splits, because there is one answer rather than one answer per row, so each morsel now reduces its own rows into a slot of its own and a serial loop combines the slots afterwards. The combine runs over the number of morsels and not the number of rows, which is seventy six slots at ten million rows, so it costs nothing next to the pass it replaces.
- A minimum needs a second slot per morsel saying whether that morsel saw a value at all. A morsel that is entirely null has no value that could stand for it, and handing the identity to the combine would be a wrong answer if every morsel were like that, so the flags are what let the combine skip the empty ones and report an invalid result when there were no values anywhere. A sum needs no such flag, because a sum over nothing is a valid zero.
- Measured on an i9-13900K at ten million rows with the two builds alternated twice. `kernel/sum_dense` went from 2.775 and 2.924 ms to 0.522 and 0.546, `kernel/sum_sparse` from 2.710 and 2.758 to 0.538 and 0.546, `kernel/min_dense` from 3.216 and 3.666 to 0.593, `kernel/min_sparse` from 5.153 and 5.246 to 0.672, and `kernel/mean_sparse` from 2.771 and 2.840 to 0.670. The first of the two new halves picked up load from another process on the machine and only its sum rows are reported from it. Three rows stayed flat as controls and are meant to: `kernel/sum_twin` is the hand written serial reference the benchmark keeps for comparison and read 5.117, 5.124 and 5.096, `kernel/add_dense` read 3.982, 3.985 and 3.990, and `kernel/take_scattered` read 7.366, 7.609 and 7.650.
- Eighty megabytes of int64 in half a millisecond is faster than this machine can read from memory, and it is not a mistake. Thirty six megabytes of the column stay resident in L3 across the repeated timing runs, so a little under half of each pass never reaches DRAM and the part that does works out to about eighty four gigabytes a second, which is inside what dual channel DDR5 delivers here.
- The order the additions happen in has changed, which is only visible in floating point. A sum was already not adding left to right, because the vector unit keeps one running total per lane, and now the lane totals are per morsel as well. The combine runs in morsel order rather than in whichever order the workers finished, so the answer is the same on every run for the same input. It is not bit identical to what a single scalar loop would produce, and neither was the version before this one.
- `sum_of`, `min_of`, `max_of`, `mean_of`, `mean_over` and `_densify` in the grouping code now raise, for the same reason the elementwise kernels started to: the morsel runtime does. Nothing in them can fail on its own.
- `is_null`, `is_not_null` and `coalesce` run on every core, and neither of them zeroes the column it is about to fill. These were the last two kernels in `nulls.mojo` that could be split, and neither carries anything from one row to the next, so the split is the same one every elementwise kernel already uses.
- A worker derives its word loop from the rows it was handed rather than the other way round. A morsel boundary is a multiple of 131072 rows and so a multiple of sixty four, which means a validity word falls entirely on one side of it and no two workers ever look at the same one. That is what makes it safe for `coalesce` to clear a bit in the output's validity from inside a worker, which it has to do on the rows where both inputs were missing.
- `coalesce` now copies and repairs in the same worker, over the rows that worker was handed, so the rows it revisits are the ones it has just written and they are still in its cache. It used to copy the whole preferred column and then walk the whole thing again looking for gaps.
- Measured on an i9-13900K at ten million rows with the two builds alternated twice. `nulls/is_null` went from 419.016 and 418.092 microseconds to 88.465 and 88.821, `nulls/is_null_sparse` from 2.286 and 2.293 ms to 252.494 and 249.689 microseconds, `nulls/coalesce` from 6.952 and 6.987 ms to 2.623 and 2.596, and `nulls/coalesce_sparse` from 19.947 and 20.036 ms to 4.463 and 4.518. The sparse rows gain the most because the repair used to be a second walk over the whole column and is now folded into the first one.
- The directional fills stay on one thread and are the controls in that run. Row `i` takes the nearest present value before it, which is a dependency reaching back an unbounded distance, so splitting it needs a different algorithm and not a different loop. `nulls/ffill` read 6.938, 6.968, 7.449 and 7.034 ms across the four halves and `nulls/drop_nulls` read 49.728, 49.680, 47.863 and 49.215. `nulls/ffill_sparse` read 17.545, 17.420, 19.041 and 18.500, which is one to six percent higher in the two new halves. Nothing in that code changed, and the most likely reason is the clock after the parallel rows either side of it rather than anything in this release.
- `is_null`, `is_null_any`, `is_not_null`, `is_not_null_any`, `all_valid_mask`, `coalesce`, `Series.is_null` and `Series.is_not_null` now raise, for the same reason. `is_null_any` was the one kernel in the package that did not raise and it no longer is.

### Added

- Two tests that reach the parallel branch. Every reduction test until now ran at three hundred and one rows, which is well under the morsel size, so nothing had ever exercised the split. One runs at 393241 rows with a whole morsel in the middle blanked, so at least one worker comes back with nothing and the combine has to skip it, and checks the sum, the minimum and the maximum against a single threaded loop over the same column. The other runs at 262144 rows of nothing but nulls and checks that the minimum, the maximum and the mean come back invalid while the sum comes back as a valid zero.
- Two tests that reach the parallel branch of the null kernels. One expands a 393241 row bitmap whose nulls are laid out so that one run is exactly a word wide and sits on a morsel boundary while another straddles one, and checks every row of both masks against the bitmap. The other coalesces two columns of the same height with every third row of the first missing and every seventh of the second, which leaves rows missing from both inside every morsel, and checks the value and the validity of every row.

## [0.6.38] - 2026-09-04

Built against Mojo 1.0.0 (ed45d567).

The elementwise kernels now run on every core and no longer ask the allocator to zero memory they are about to fill. Three changes, and they are all the same two problems found in three places.

A kernel that writes one result per row has no carried state and no cross row dependency, so there was never a reason for it to run on one thread. Casts, the four column arithmetic operations, the constant forms of arithmetic, the six comparisons in both their column and constant forms, and both text comparisons all moved onto the morsel scheduler. Each of them also allocated its output with the constructor that memsets and then wrote every element of it, including the elements under the nulls, so the memset in front of the loop was a full pass thrown away every time. On an i9-13900K at ten million rows a cast to float64 went from 7.19 ms to 3.17, an addition of two columns from 8.76 to 4.00, a comparison against a constant from 4.20 to 1.11, and a text equality against a long constant from 10.9 to 1.90.

The null repair moved too, and it moved differently than expected. These kernels compute over the whole values buffer and then zero the values under the nulls afterwards, which used to be a walk over the finished column on one thread. Making that walk parallel in its own right measured slower on a column with no nulls, because there the walk is one comparison per sixty four rows and starting a parallel region over it costs more than the work inside it. Fusing it into the end of each morsel, in the worker that just wrote those rows, is what worked: a quarter null column went from 8.26 ms to 5.11 and the dense columns came down slightly as well rather than regressing.

### Changed

- The null repair happens in the worker that computed the rows, instead of in a pass of its own after all of them are done. An elementwise kernel computes over the whole values buffer and then has to zero the values under the nulls, and until now that was `apply_validity` walking the entire column on one thread once the workers had finished. It is now `repair_range`, called as the last statement inside each morsel, so the rows it touches are the ones that core has just written and are still in its cache. `kernel/add_sparse` went from 8.261 and 7.931 ms to 5.106 and 5.103 on an i9-13900K at ten million rows, about one and a half times, and the dense rows moved a little too because the repair no longer has to reread anything: `kernel/add_dense` 4.046 and 4.034 to 4.000 and 3.999, `kernel/multiply_dense` 4.000 and 4.000 to 3.948 and 3.953, `kernel/divide_dense` 4.021 and 4.022 to 3.975 and 3.966, `kernel/less_dense` 2.779 and 2.764 to 2.695 and 2.711, `kernel/add_constant` 2.686 and 2.677 to 2.636 and 2.633, `kernel/less_constant` 1.157 and 1.157 to 1.114 and 1.116. `kernel/take_scattered` read 7.913, 7.660, 7.636 and 7.651 across the four halves as a control.
- Fusing it was the second design and the first one was thrown away on the measurement. Making `apply_validity` a parallel pass of its own was tried first and it made a dense column slower, 4.371 and 4.344 ms on `kernel/add_dense` against 4.026 and 4.130 for the serial repair, because on a column with no nulls that pass is already down to one comparison per sixty four rows and starting a parallel region over it costs more than the work it splits. The fused form has no second region at all, which is why it helps the sparse column and does not hurt the dense one.
- Splitting the repair by row range is safe without any locking because a morsel boundary is a multiple of 131072 rows, so it is a multiple of sixty four, so no two workers ever look at the same validity word. `repair_range` documents that requirement and `apply_validity` is now the whole column spelling of it, for the callers that are not splitting anything.
- Text comparison runs on every core and no longer zeroes the column it fills. `compare_text` and `compare_text_const` were the last elementwise kernels still on one thread. A row's cost here depends on its own bytes and on nothing else, so there is no reason for the column not to split, and the one thing that had to move was the constant's view, which is built once above the split rather than once per worker. At ten million rows, `text/equal_long` went from 13.977 and 11.260 ms to 3.571 and 3.516, `text/equal_short` from 8.517 and 8.340 to 1.495 and 1.516, `text/equal_constant` from 10.563 and 12.163 to 1.896 and 1.904, and `text/equal_constant_short` from 1.824 and 2.013 to 0.443 and 0.458. `text/equal_number`, `text/take_text`, `text/concat_text`, `text/sort_repeated` and `text/group_distinct` are untouched by this and none of them moved.
- `compare_text` and `compare_text_const` now raise, for the same reason the numeric kernels started to: the morsel runtime does. Nothing in them can fail on its own.

- Elementwise arithmetic and comparison run on every core and no longer zero the column they are about to fill. This is the same pair of problems the cast had and the same fix. Nine loops move: the three column arithmetic operations, division, the constant forms of both, and the two comparison loops. Every one of them writes a result for every row, the null ones included, because these kernels compute over the whole values buffer and let `apply_validity` blank the nulls afterwards, so the memset the allocation did in front of the loop was a full pass thrown away in every case.
- Measured on an i9-13900K at ten million rows, the two builds alternated twice, load average near two for all four readings. `kernel/add_dense` went from 8.761 and 8.669 ms to 4.040 and 4.039, `kernel/multiply_dense` from 9.386 and 9.663 to 3.999 and 3.998, `kernel/divide_dense` from 9.616 and 9.256 to 3.997 and 4.011, `kernel/less_dense` from 6.516 and 5.780 to 2.779 and 2.762, `kernel/add_constant` from 7.220 and 7.122 to 2.686 and 2.681, and `kernel/less_constant` from 4.195 and 3.910 to 1.171 and 1.154. That is a bit over two times on the column forms and closer to three on the constant ones, which is the right shape: a constant form touches two columns rather than three, so it was the more bandwidth starved of the two on one core and has the most to gain from having thirty two.
- The same run carries `kernel/sum_dense`, `kernel/min_dense`, `kernel/take_scattered`, `kernel/filter` and both cast rows as controls, and none of them moved. That matters because this machine has been seen to sit in one of two modes on some rows, and a set of untouched rows reading the same on both halves is what says the halves are comparable.
- `kernel/add_sparse` gains less, 12.567 and 12.859 ms against 7.727 and 8.328, and the reason is worth writing down. That column is a quarter null, and `apply_validity` walks the validity a word at a time on one thread after the workers are finished, touching elements only in the words that are mixed. On a dense column it is one comparison per sixty four rows and invisible; on a sparse one it is most of what is left. Parallelizing the repair is the next step and is a separate change.
- A constant form keeps its branch on `flip` outside the inner loop, so it is one test per morsel rather than one per register, which is what it was when there was a single loop over the whole column.
- `add`, `subtract`, `multiply`, `divide`, the six comparisons, `arith_const`, `divide_const` and `compare_const` now raise, because the morsel runtime does. Nothing in them can fail on its own.

- A cast runs on every core and no longer zeroes the column it is about to fill. It was a serial loop, one SIMD register at a time, over an output allocated with the constructor that memsets, and neither of those is needed. The loop has no carried state and no cross row dependency at all, so it splits over morsels the same way the other kernels do, and it writes an element for every row including the null ones, so the memset in front of it was a full pass thrown away. Casting ten million int64 to float64 on an i9-13900K went from 7.194 and 7.226 ms to 3.171 and 3.178, and to int16 from 4.131 and 4.245 to 1.275 and 1.301.
- The two halves are worth separating because they are not the same size. A third build with only the allocation changed, still serial, measured 5.742 and 5.858 ms on the float64 row and 3.673 and 3.844 on the int16 one, so dropping the memset is a fifth of the float64 cast and a tenth of the int16 one, and the rest is the parallel loop. Each half was measured twice with the three builds alternated in order on a machine at a load average under 2.5, and no reading of one build overlapped any reading of another.
- The change is in `_cast_erased` as well as in `cast_to`, and that is the one that matters. `cast_to` is the typed entry point and only tests and benchmarks name it; a cast on a frame goes through `cast_any`, which dispatches on the runtime dtype into `_cast_erased`, and that function carried its own copy of the same serial zero filled loop. `frame/cast_one` went from 19.5 and 20.2 ms to 15.8 and 16.5, which is smaller in proportion because that row also builds the frame around the column.
- Splitting the loop is safe on the boundaries without a scalar tail, and there is now a test that says so rather than a comment. A morsel is 131072 rows and the widest register in play is 64 int8 lanes, so every interior boundary is a multiple of the step and no worker can write into the rows of the next one. Only the last morsel steps past the logical end, into the padding the buffer already guarantees, which is what the serial loop did too. The test casts 393241 rows, a prime past three morsels which leaves the last one short and off every register boundary at once, and checks every row against the scalar answer.

### Added

- `test_a_cast_past_the_split_converts_every_row` in the kernel tests, covering the morsel boundaries of a parallel cast at a length that is not a multiple of anything.
- `test_arithmetic_and_comparison_past_the_split_are_right_everywhere`, which asks the same boundary question of the elementwise kernels and also checks the rows under the nulls, since the repair that blanks those runs after the workers and would otherwise hide a bad row.

## [0.6.37] - 2026-09-04

Built against Mojo 1.0.0 (ed45d567).

Three changes to how a group by turns keys into ordinals, and all three are work that was being done and then thrown away.

Grouping on several integer keys no longer factorizes each one first. A group by reads only a key's group count and its ordinals as values, never their order and never their density, so a value minus its column's minimum will do as the ordinal, and the scan that finds that minimum is one the factorize was going to do anyway. What used to be a pass per key and a forty megabyte column per key is now a plan and a single walk over the raw columns. The packing that follows was also written as a fold, one pass per key over a column already in memory, and it is not a fold: the number it arrives at is positional notation and every multiplier in it is known before the first row is read, so it is one weighted sum per row. Six keys on ten million rows was five passes and about nine hundred and sixty megabytes of traffic and is now one pass and three hundred and twenty.

A factorize no longer zeroes the ordinals it is about to write. Four of its routes allocated the output with the constructor that memsets, and the build writes an ordinal for every row it is given, so the memset was a full pass thrown away. It is worse than a wasted pass, because it is one thread writing forty megabytes immediately in front of a section that runs on every core.

And a string factorize no longer reaches back into the column to settle a hash match. It was reading the representative key's view out of a buffer sixteen bytes a row wide, indexed by group ordinal, which is a cache miss per match into something far larger than any cache. Keeping the views as the ordinals are handed out makes that sixteen bytes a group instead. On the i9 at ten million rows the sliced route, which is the one every chunk of the streaming engine takes, is twenty percent faster on a hundred distinct keys and thirty percent on a hundred thousand.

### Changed

- Grouping on several keys no longer factorizes each key first when they are all integers of one dtype in ranges narrow enough to index. The composition gives every key its own dense ordinals and then packs those ordinals into one integer, and each of those factorizes is a scan for the range, a pass to assign, and a forty megabyte column of ordinals for the packing pass to read back. None of it is needed, because a group by on several keys reads only a key's group count and its ordinals as values, never their order and never their density, so a value minus its column's minimum will do as the ordinal. `direct_plan` already works that minimum out in the scan the factorize was going to do anyway, so what is left is a plan rather than a pass, and one walk over the raw key columns writes the packed value. On ten million rows, `group/ordinals_two_keys` went from 24.2 ms to 16.8 on an i9-13900K with the two builds alternated twice, and `group/ordinals_one_key` was unchanged as a control.
- The route is declined for a key that is not an integer, for a mix of dtypes, for a key with a null in it, and for a tuple whose combined range is too wide to lay a table over. Declining costs a scan, so each key is given only what the tuple has left to spend as its ceiling and `direct_plan` returns as soon as a key passes it, which means only the first key can ever cost a full scan for nothing whatever the key count. The new `group/ordinals_two_keys_declined` measures that: two keys of a hundred thousand values each, where either alone would pack and the pair cannot, went from 105 ms to 108, which is the one scan and 2.3 percent of a pass that shape costs anyway.
- No db-benchmark query moves on this. All of its multi-key group bys include one of `id1`, `id2` or `id3`, which are text columns there as they are in the published suite, so the route is correctly declined for every one of them. The gain is on integer key tuples, and closing the same gap for text keys is a separate piece of work.

- Grouping on several keys packs their ordinals into the combined key in one pass rather than one pass per key. The packing was written as a fold, taking a running value, multiplying it by the next key's group count and adding that key's ordinals, which reads and writes a whole column per key and has to be an int64 because a fold cannot know where its own product stops. It is not a fold. The number it arrives at is positional notation and every multiplier in it is known before the first row is read, so it is one weighted sum per row over ordinals that are all in memory already. Six keys on ten million rows was five passes and about nine hundred and sixty megabytes of traffic and is now one pass and three hundred and twenty. Where the product fits a uint32, which covers the common case of two or three low cardinality keys, the packed column is four bytes a row rather than eight and the vector is twice as wide. `group/ordinals_six_keys` went from 163.5 and 164.5 ms to between 157.7 and 161.6 across two separate alternating runs on an i9-13900K, so three to four percent, and db-benchmark q10 does not move outside the noise, which is what a change to an eighth of the grouping should look like. Its resident set is consistently about fifty megabytes lower.
- The fold is still there for a combined key space that leaves an int64, because that case renumbers partway through with `_condense` and a single pass has no partway. It needs six keys of a thousand values or twenty six of a hundred, so no real table reaches it, but the test is on the product rather than on the key count.

- A factorize no longer zeroes the ordinals it is about to write. Four of its routes allocated the output with the constructor that memsets, and the build loop writes an ordinal for every row it is given, a zero for the null ones, so the memset was a full pass over the column thrown away. It is worse than a wasted pass, because it is one thread writing forty megabytes on ten million rows immediately in front of a section that runs on every core, which is the worst place in the routine to put serial work. Measured directly on ten million uint32: the zeroing allocation is 0.74 ms on an i9-13900K and 4.40 ms on an eight core Xeon, against 0.014 and 0.023 ms for the one that leaves the memory alone. Filling the pages afterwards is also cheaper when they were not touched first, 0.13 ms against 0.44 on the i9 and 1.75 against 2.16 on the Xeon, so the saving is larger than the memset itself. On the i9 with the two builds alternated three times at one to three percent iqr, `group/ordinals_one_text_key` went from 11.17, 11.17 and 12.09 ms to 8.89, 9.83 and 9.70, and `group/ordinals_two_text_keys` from 26.13 and 27.56 to between 24.49 and 25.58. `group/ordinals_two_keys` and `group/ordinals_two_keys_declined` did not move, which is right: they take the direct route, and the routes the integer rows exercise had already been converted.

- A string factorize no longer reaches back into the column to settle a hash match. When a probe landed on a slot whose hash matched, it compared the row against the key that first produced that ordinal, and it got hold of that key by taking the row number the build had recorded and reading that row's view out of the column. The views buffer is sixteen bytes a row, a hundred and sixty megabytes on ten million, and reading it by group ordinal is one scattered cache line per row out of something far larger than any cache. The build now keeps the views themselves as it hands out ordinals, which is sixteen bytes a group rather than sixteen bytes a row, so a hundred thousand groups is 1.6 megabytes and stays resident. It stays exact for long strings, because this layout carries the length and the first four bytes inside the view and only reads the payload when those match, which is what the column comparison already did.
- The gain lands on the sliced route, where the scattered read was the only one. Measured on an i9-13900K at ten million rows on one thread, it went from 115.0 and 117.2 ms to 92.2 and 95.4 on a hundred distinct keys, and from 212.7 and 215.7 to 146.1 and 150.5 on a hundred thousand, so twenty and thirty percent. Nine further alternating pairs taken while the machine was loaded are not worth quoting as times, but the new build won all eighteen of those readings, which is the question a busy machine can still answer.
- That is the route the streaming engine takes for every chunk it groups. A chunk is 131072 rows and the gate onto the parallel string routes is 262144, so a per chunk string factorize is always the sliced one, and this is the path a group by over a text key in the chunked engine spends its time in.
- The partitioned route does not move, and the reason is that its comparison has two scattered reads rather than one. It sorts the rows into hash partitions first, so the probe arrives at them out of order, and reading the probe row's own view is a miss whatever the representative costs. Removing one of two misses that were already in flight together is worth much less than removing the only one, which is what the sliced route had, where the rows arrive in order and the hardware prefetches them. It measures a few percent at a hundred thousand keys and nothing at a hundred, and the honest reading is that it is unchanged.
- The `group/ordinals` text rows cannot adjudicate this on the machine it was measured on, and that is worth writing down rather than working around. Seven halves alternated at a load average under two with two to six percent iqr each put `group/ordinals_one_text_key` in one of two modes, near 7.7 ms or near 11.0, and both builds landed in both: three old and one new in the fast mode, two new and one old in the slow one. The spread between the modes is larger than the change being measured, so the row says nothing here in either direction, and the claim above rests on the route probe instead.

### Added

- `group/ordinals_one_wide_text_key`, one text key over `id3`, which holds a row per hundred rather than a hundred values in all. That is the shape db-benchmark q3 and q5 group on, and it is a different problem from the narrow key: the table stops fitting in cache and the routing sends it down the partitioned route instead of the sliced one. A change to what a probe costs shows up here and not on `group/ordinals_one_text_key`, where a hundred groups sit in L1 whatever the probe does.
- `group/ordinals_two_keys_declined`, the two key grouping pass over a pair the fused tuple pack cannot take. A route that is a win when it applies is not free when it does not, and this is the row that says by how much.
- `group/ordinals_one_text_key`, `group/ordinals_two_text_keys` and `group/ordinals_six_keys`, over a table shaped like db-benchmark's own: three text key columns and the same three values again as integers. Every multi key group by in that suite includes a text key, so the integer key rows were measuring a shape the suite never asks for. At ten million rows one text key is 10.5 ms, two are 27.1 and all six are 160, against 8.4 for one integer key.

## [0.6.36] - 2026-09-04

Built against Mojo 1.0.0 (ed45d567).

Four changes to the grouped reductions, all of them found by asking why one query cost more than another query that does the same amount of work. None of them changes an answer or an interface.

Two of them are memory that was moved for nothing. A grouped median, quantile and distinct count all begin by laying each group's values out next to each other, and that slab was zeroed before it was filled even though the fill covers every element of it, which is eighty megabytes written and thrown away on ten million rows. A grouped correlation cast both of its input columns to float64 before it started, which is two more full copies, and it did that because dispatching on two runtime dtypes is a hundred and forty four instantiations. Dispatching only on the case where the two dtypes agree is twelve, and it covers nearly every call, because the two columns of a correlation are two measurements out of one table.

The other two are loops that were not using the machine. The merge that folds the thread local tables at the end of a grouped min or max was a scalar loop with a branch and a bitmap read modify write per group per worker, three million of them at a hundred thousand groups over thirty two workers, where the sums have a vectorized walk of two arrays. The identity is neutral for a min the same way zero is for a sum, so a worker that never reached a group can be folded in unconditionally and the merge is now that same walk. And packing several keys into one integer no longer widens the first key's ordinals in a pass of their own, because the first packing step was already reading a column and writing the accumulator they were being widened into.

On the i9-13900K at 0.5GB, each measured with the two builds alternated at load average under one: db-benchmark q9 went from 0.050 s to 0.036, q7 from 0.056 to 0.048, q6 from 0.681 to 0.267, and q2 from 0.043 to 0.041. q6 and q7 are now level with or ahead of every other engine measured in the same invocation.

### Changed

- The merge at the end of a grouped min or max is vectorized. Each worker builds a private table and those tables are then folded into one, and that fold was a scalar loop carrying a branch and a bitmap read modify write per group per worker, which on a hundred thousand groups over thirty two workers is three million of them and was most of what a grouped extreme cost on a wide key. A worker that never reached a group left the reduction's identity in its slot and folding the identity in changes nothing, so the fold does not have to ask which groups a worker saw. It is now the same vectorized walk of two contiguous arrays that the sums have, with a comparison where they have an add, and the seen flags fold beside it a byte at a time. The two identity fills are vectorized with it and no longer zero the tables first. db-benchmark q7, a max minus a min over a hundred thousand groups, went from 0.056 s to 0.048 s at 0.5GB on an i9-13900K, with q3 and q5 over the same key unchanged as controls.
- Grouping on several keys no longer widens the first key's ordinals in a pass of their own. Packing a tuple into one integer walks the rows once per extra key, and the first of those walks was preceded by a pass that copied the first key's uint32 ordinals into the int64 the packing accumulates in. The first packing step is already reading a column and writing that accumulator, so it now widens as it goes and the separate pass is gone. On ten million rows that is a hundred and sixty megabytes less moved. Measured on an i9-13900K at 0.5GB with the two builds alternated, db-benchmark q2 went from 0.043 to 0.041 s, q9 from 0.038 to 0.037 s and q10 from 0.232 to 0.229 s, which is small and is the size a removed pass over that much memory should be at this machine's bandwidth. It was in the same direction in all six comparisons.

- The pass that lays a group's values out next to each other, which is what a grouped median, quantile and distinct count all need before they can start, is now partitioned and parallel above a few thousand groups. It was one core walking the rows and writing each into its group's run, which is a random write per row, and on a high cardinality key that was the entire cost of the reduction rather than a part of it. It now copies the rows into partition order first, where a partition is a run of group ordinals, and then fills one partition per worker into a slice of the slab small enough to stay in that core's cache. Three passes instead of one and none of them missing. On ten million int64 rows on an i9-13900K, nanoseconds a row: a median over a hundred thousand groups went from 12.739 to 1.748, and over six and a half million groups from 17.678 to 3.088. A median over a thousand groups is unchanged, because below the threshold the old loop had a handful of write streams and there was nothing there to fix.
- The slab that a grouped median, quantile and distinct count lay their values out in is no longer zeroed before it is filled. The fill writes one value per present row and the offsets it writes at cover the slab exactly, so every element is written before anything reads it and the zeroing pass was a write over the whole column for nothing. On ten million int64 rows that is eighty megabytes of memory traffic removed from each of those three kernels.
- db-benchmark q6, which is a median and a standard deviation over six and a third million groups, went from 0.681 s to 0.267 s at 0.5GB on the same machine. Measured in the same invocation as the other three engines, that is now ahead of all of them: Polars 0.325 s, DuckDB 0.478 s, pandas 1.214 s, and firepanda on the lowest peak memory of the four at 1.35 GB against DuckDB's 3.17 GB.

- A grouped correlation or covariance over two columns of the same dtype no longer converts either of them first. The erased entry point cast both inputs to float64 and called one instantiation, because dispatching on both dtypes is a hundred and forty four copies of the loop. Dispatching only on the case where the two agree is twelve copies, and it covers nearly every call, because the two columns are two measurements out of the same table. The kernel already reads a value and casts it to float64 in the same expression, so an instantiation on the column's own dtype does the conversion in a register as it goes instead of materialising two full copies of the input. db-benchmark q9, a squared correlation grouped on two keys, went from 0.050 s to 0.036 s at 0.5GB on an i9-13900K, with peak memory down from 1.15 GB to 1.05 GB, which is the two copies. A mixed dtype pair still casts.

### Added

- Four `group/median_cardinality_*` benchmarks over one column and one kernel at ten groups, a thousand, a hundred thousand and two thirds of the row count. The sorts a median pays for get shorter at every step, so a cost per row that rises across the four is a cost in the layout rather than in the reduction, which is how the pass above was found.

## [0.6.35] - 2026-09-03

Built against Mojo 1.0.0 (ed45d567).

This release is one fix, to the operator the last release shipped, and it is the fix that turns the streaming group by from a trade into a win.

The operator merged each chunk by stacking its running table with the chunk's own table and grouping the two together. That reads the whole running table, so it costs the number of groups seen so far on every chunk, and the number of chunks grows with the input. The work that is not proportional to the input therefore grew as chunks times groups. At a thousand groups that is invisible, which is why it shipped. At a hundred thousand it was the entire cost of the operator, and the shape of it was there to read in the numbers: a flat sixty five nanoseconds a row at one million, four million and sixteen million rows, while the materialised fallback the operator exists to replace fell from 14.3 to 4.8 over the same range. A cost that will not move while the comparison's falls is what a term of that shape looks like from the outside.

A group now keeps the ordinal it was handed the first time it was seen, for the whole query, in a map that outlives the chunk. Absorbing a chunk is then looking its rows up and folding them into the slots those ordinals name, which is a lookup and a fold per row and nothing at all proportional to the group count. The map is an array indexed by the key when the keys are packed close enough together and a hash table when they are not, which is the same choice `factorize` makes per column, and having both matters more here than there: at a thousand groups the array is around two and a half nanoseconds a row against the hash's six and a half, so a map that only knew how to hash would have made every query with few groups slower even while it made the wide ones ten times faster.

At sixteen million rows on an idle i9-13900K, one build against the other in a single run, in nanoseconds a row: a hundred thousand groups went from 64.054 to 3.199 against a fallback at 5.157, and a thousand groups over chunks the size the engine makes went from 6.223 to 2.593 against a fallback at 4.315. So the breaker is no longer a trade of speed for memory. It holds one row per group instead of every row and it is 1.6x faster than holding every row, at both ends of the group count.

One shape is slower and is meant to be. A single chunk of more than four million rows used to go through a factorize that splits across workers at that size and the lasting map is one thread, so that benchmark went from 2.409 to 2.925. The engine does not make chunks of four million rows, and making the map's lookup parallel belongs with the rest of the operator's parallelism rather than with this.

### Added

- `firepanda/hash/lasting.mojo`, a key to ordinal map that outlives the chunk it was given. A group keeps the ordinal it was handed the first time it was seen for the whole query, so a chunk is absorbed by looking its rows up and folding them into the slots those ordinals name rather than by working out how the chunk's groups line up with the running table's. It holds one of two maps, an array indexed by the key when the first chunk's keys are packed close enough together and a hash table when they are not, which is the same choice `factorize` makes per column and for the same reason: the array lookup is a subtraction and a load and the hash is a multiply, a mask, a load, a comparison and a cache miss.
- `firepanda/kernel/running.mojo`, the accumulator kernels. Every other aggregation kernel takes a column and gives back one row per group; these take a column and a table that already has a row per group, and add the one to the other. A slot no row has reached holds the reduction's identity so the inner loops have no branch on whether a group has been seen, and the extremes and the edges carry the reached flags in the validity bitmap they had to maintain anyway.
- `group/pipeline_stream_wide` and `group/pipeline_materialize_wide` in the benchmark suite, the same pair as the existing streaming and materialised group by benchmarks at a hundred thousand groups instead of a thousand. The pair at one group count says nothing about a cost that grows with the group count, which is what the change below is about.

### Changed

- The streaming group by no longer stacks its running table with each chunk's table and regroups the result. That merge costs the height of the running table on every chunk, so the work that is not proportional to the input grows as the number of chunks times the number of groups, and the number of chunks grows with the input. At a thousand groups the term is invisible. At a hundred thousand it was the whole cost of the operator.
- On an idle i9-13900K, in nanoseconds a row, at a hundred thousand groups: a million rows went from 61.392 to 6.008, four million from 66.455 to 3.847, and sixteen million from 64.054 to 3.199. The per row cost used to sit flat while the materialised fallback's fell, which is what a term of that shape looks like from the outside, and it now falls with the input the way the fallback's does.
- At a thousand groups over chunks the size the engine makes, a million rows went from 5.578 to 2.378, four million from 5.296 to 2.567, and sixteen million from 6.223 to 2.593.
- The breaker is no longer a trade of speed for memory. It holds one row per group instead of every row and it is now 1.6x faster than the materialised path that holds every row, at both ends of the group count: 2.593 against 4.315 at a thousand groups and 3.199 against 5.157 at a hundred thousand, both at sixteen million rows.
- One shape is slower and is meant to be. A single chunk of more than four million rows used to go through a factorize that splits across workers at that size and the lasting map is one thread, so `group/pipeline_stream_one_chunk` went from 2.409 to 2.925. The engine does not make chunks of four million rows, `MORSEL_ROWS` is a hundred and twenty eight thousand, and that is where the 2.4x above was measured.
- Text keys, key tuples of several columns, text value columns and a key column with a null in it all keep the older merge. The map is exact for a fixed width key because the hash is a bijection on the key bits and it is not exact for text, a running slot is a number in an array so a minimum over names has nowhere to live, and a null key would take an ordinal the map does not reserve, which would put the null group somewhere other than where its first null row was. The handover for the null case happens in the middle of a query, because whether a key column has a null is not known until the chunk holding it arrives.

## [0.6.34] - 2026-09-03

Built against Mojo 1.0.0 (ed45d567).

This release is the first pipeline breaker that is not the materialised fallback, and the discovery, while measuring it, that the engine had been leaving most of its group by speed on the floor for a reason that had nothing to do with the operator.

`Group` is the breaker. A materialised group by holds every row until the last one arrives, so the memory it needs is the size of its input. This one holds one row per group: it groups each chunk on its own, merges that chunk's answers into a running table, and throws the chunk away. A billion rows in a thousand groups is a table of a thousand rows the whole way through. The merge is the same operation as the aggregation, so there is no accumulator kernel here and no second implementation of anything, which is also why the eight reductions that fold are exactly the eight that have a partial state a group by can reduce. The ones that do not fold keep the fallback and say so at plan time rather than quietly answering a median of medians.

The second half is a constant. `PARALLEL_ROWS` decides whether a factorize runs on one thread or several, and it was `1 << 17`, which is exactly `MORSEL_ROWS`. The gate is `>=`, so a chunk sized factorize was always one row over the line, and the worker count is the height divided by the minimum slice, so a column that just clears the line gets the fewest workers the split ever runs on. Every factorize a pipeline did landed on the worst point of the curve, at 5.0x the cost of not splitting at all, and the eager path paid the same tax anywhere a column landed just past the line.

Measuring a build that never splits against one that always does then showed the threshold was four to five doublings too low regardless of the collision, and that text wanted the opposite, because a string key costs around thirty times what an int64 key costs per row and pays a split back that much sooner. One constant was serving two routes thirty times apart. It is now two, `1 << 22` for fixed width keys and `1 << 18` for text, both above the chunk size. One build against the other in a single run, that is 6.4x on the key factorize at a quarter million rows, 2.2x on a whole frame group by, and up to 1.46x on the streaming pipeline, with nothing measured slower.

Both thresholds are from one machine and what carries over is the shape rather than the numbers, since what decides a crossover is the ratio between what a row costs on one thread and what the merge costs. The honest version of either is a cost model over the row count and the group count, and that is written up on the constants.

### Added

- `Group`, the first pipeline breaker that is not the `Materialize` fallback. A materialised group by holds every row until the last one arrives, so the memory it needs is the size of its input. This one holds one row per group: it groups each chunk on its own, merges that chunk's answers into a running table, and throws the chunk away. A billion rows in a thousand groups is a table of a thousand rows the whole way through.
- The merge is the same operation as the aggregation, which is why there is no accumulator kernel in this change and no second implementation of anything. Two partial answers for a group are two rows, and reducing two rows to one is what a group by does, so merging the running table with a chunk's table is a group by over their concatenation, reduced by the kind that combines partials. That kind is the kind itself except for the two counts, where merging means adding rather than counting again.
- Eight reductions fold and the node runs those: sum, mean, minimum, maximum, count, size, first and last. A mean folds but not as a mean, so the running state is a sum and a count and the division happens once at the end, which is the only place a state column and an output column are not the same thing. The rest keep the fallback and that is the right answer rather than a gap: a median of medians is not a median and no state short of the values themselves makes it one. Asking for one on this node is an error at plan time.
- Everything that can be wrong with a group by without looking at the data is caught when the node is added to a pipeline: a key that is not a column, a key given twice, no keys at all, a reduction that does not fold, a sum of a column of names, and two output columns that would collide. The output schema is known at the same point, so a plan can be typed before a row moves.
- Text keys and text columns work, because the reductions that mean something over bytes are already there. A minimum over a column of names is the smallest name, and it folds like any other extreme.
- `group/pipeline_stream` and `group/pipeline_materialize` in the benchmark suite, which ask the same query of the same chunked rows through the same driver and differ only in which operator does the grouping. On a million rows in a thousand groups on an idle i9-13900K the streaming node takes 5.047 ms against the fallback's 4.251 ms, so it is 1.19x slower and holding one row per group instead of every row is what that buys.

### Changed

- The node deliberately does not sort and does not drop groups whose key is null, so the groups come out in first seen order with the null key among them. Both are decisions about the output rather than about the grouping, neither needs to see a row of input, and putting either inside the operator would make every query pay for it. `DataFrame.group_by` is unchanged and still defaults to both.
- Adding a column of floats in chunks and then adding the chunk sums is a different order of additions from adding it in one pass, so a sum or a mean of floats through the node can differ from the eager path in the last bits. Every other reduction here is exact.
- The factorize split threshold was one constant serving two routes with per row costs thirty times apart, and it is now two. `PARALLEL_ROWS` covers fixed width keys and moved from `1 << 17` to `1 << 22`, and `PARALLEL_STRING_ROWS` covers text and sits at `1 << 18`.
- The bug that started this was that the old value was exactly `MORSEL_ROWS` and the gate is `>=`, so a chunk sized factorize was always one row over the line and went parallel on four slices, which is the fewest workers the split ever runs on. Every factorize a pipeline did landed on the worst point of the curve. On an idle i9-13900K `group/ordinals_one_key` at 131071 rows is 0.588 ns a row and at 131072 rows it was 2.957, so one extra row cost 5.0x.
- Closing that cliff turned out not to be the same question as putting the line in the right place, and measuring a build that never splits against one that always does said so. On a thousand groups, serial against parallel in ns a row: a quarter million rows is 0.598 against 2.277, a million is 0.625 against 1.611, three million is 1.432 against 1.685, and four million is 0.947 against 0.816. Serial wins by 3.8x at the bottom of that range and the two do not cross until a little under four million rows, so everything below was being cut into slices that could not pay for the merge. The high cardinality route that goes through the hash table crosses in the same place.
- Text went the other way and that is why it now has its own constant. A string key is hashed over its bytes and settled with a comparison over its bytes, around 19 ns a row against about 0.6 for an int64, so the fixed cost of a split is paid back roughly thirty times sooner. `text/group_distinct` at 262144 rows is 18.893 ns a row split against 23.746 serial, and at a million rows it is 12.698 against 24.083. Holding text to the numeric threshold would have cost 1.26x to 1.9x on exactly the queries a string key is used for.
- Nothing about any of this was specific to the streaming engine and the eager path was paying it too. A group by of a frame at 131072 rows goes from 8.368 ns a row to 4.279, and the materialised fallback at the same height goes from 6.294 to 4.951.
- Measured as one build against the other in a single run, the change is 6.4x on `group/ordinals_one_key` at a quarter million rows, from 3.797 ns a row to 0.592, and 2.8x at a million rows. The high cardinality version of the same question is 3.1x at a quarter million, a group by of a whole frame is 2.2x, and the streaming group by is 1.30x. At four million rows, which is above the new threshold, the two builds take the same route and measure the same, and text measures the same at every height in that sweep. Nothing in the suite got slower.
- The streaming node is still behind the materialised fallback but by less, at 1.19x on a million rows where it was 1.36x. Closing the rest of that is the thread local partitioned tables, which is the next line on the milestone and is a change to the operator rather than to a constant.
- The same query over 16 thousand row chunks runs at 3.045 ms, which is 1.77x faster than the fallback and 1.81x faster than the same operator over 128 thousand row chunks. Both chunk sizes now take the same serial route and do the same work per row, so what is left is the running table and the chunk falling out of cache together at the larger size. That says the chunk size wants tuning for grouping specifically, and `group/pipeline_stream_16k` and `group/pipeline_stream_one_chunk` are in the suite to keep measuring it.
- Both values are from one machine and the shape of the answer is what carries over rather than the numbers. What decides a crossover is the ratio between what a row costs on one thread and what the merge costs, so a slower core or a shorter key moves it, and the honest version of either constant is a cost model over the row count and the group count rather than a row count alone. That is written up on the constants and is the next thing on this milestone.
- Three tests in `tests/test_hash.mojo` pin the thresholds against the chunk size, against the minimum slice and against each other, rather than against numbers. None of them can catch a wrong answer, which is why nothing else in that file would have caught any of this, and they fail if either threshold is put back on the chunk size, if the first column that splits would get fewer than eight workers, or if text is ever made to split later than numbers.
- `group/ordinals_one_key_wide` in the benchmark suite, which asks the one key question of a key with a hundred thousand values rather than a thousand. Which side of a threshold a factorize belongs on is not a question about the row count alone, because what the parallel route adds is a merge over every group every worker found, and that part grows with the cardinality rather than with the height.

## [0.6.33] - 2026-09-02

Built against Mojo 1.0.0 (ed45d567).

This release finishes the elementwise half of the streaming engine. Before it the engine could push chunks through a line of operators but the line could only be filter, select and limit, which is not enough to run a query. It can now run the shape almost every query actually has, which is a predicate over a column and a constant.

Three things landed to get there. The first is `Compute` and `Cast`, the two nodes that finish the elementwise family. `Compute` appends rather than replaces, which is what lets an expression tree be a line of nodes: `(a + b) < c` is one node that appends `a + b` and a second that compares it against `c`. `Cast` replaces in place, because a cast says a column is that type now and everything downstream that named it still means it. Both bind when they are added to a pipeline rather than when the first chunk arrives, so a column position outside the schema, or an operation with no answer on those two types, is an error before a row moves.

The second is `Value`, a scalar that carries its own type, and the constant kernels that go with it. A constant on either side of an expression is most of what a query contains, and until now `x > y` could be spelled and `x > 5` could not. The constant forms are their own loops rather than a column of a repeated value pushed through the pair forms, which measured 1.34x on addition at a million rows on an i9-13900K. Promotion looks at the constant's type rather than its value, so `x + 1` on an int32 column stays int32 and an expression's type stays something the plan can work out.

The third is comparison on text. A filter on a label is the other half of what people write and it could not be spelled either. It is a byte loop rather than an arm of the dtype dispatch, and there was a real trap in the old arrangement: a text column's physical dtype is uint8, so falling through to the uint8 arm would have compared the first byte of each sixteen byte view as a value and answered nonsense without an error. Against a short constant it runs at 0.401 ns a row where the same question on a column of int64 runs at 0.210, so filtering on a label is within 2x of filtering on a number.

The engine's elementwise operator line on the M2b checklist is now complete: filter, select, arithmetic, comparison, cast and with_column, on numbers and on text, none of them breaking a pipeline.

### Added

- `Compute` and `Cast`, the two engine nodes that finish the elementwise family. `Filter` and `Project` were already there, so the line on the M2b checklist that asks for filter, select, arithmetic, comparison, cast and with_column is now four operators rather than two, and none of them is a breaker.
- `Compute` is more general than it looks. An expression is a tree and a tree is a line of these, so `(a + b) < c` is one node that appends `a + b` and a second that compares the appended column against `c`, with a `Project` at the end to drop the intermediate. That is why it appends rather than replaces, and it is also why `Filter` can name its mask by position: the plan counted the appends. No node has to hold an expression tree and the plan can see every intermediate.
- `Cast` replaces a column in place, which is right for it and wrong for `Compute`, and the difference is what the operation means. A cast says a column is that type now, so the position keeps its meaning and everything downstream that referred to it still refers to the same thing. An expression makes a new column and giving it a position of its own is what lets the old one still be read.
- Both are bound when they are added to a pipeline rather than when the first chunk arrives, so a position outside the schema, or an operation with no answer on those two types, is an error at plan time. `Compute` reports the type of the column it will append, which is a function of the two operand types and the operation and nothing else, so an expression's type is known before any row moves even though the dtypes are values.
- `firepanda/kernel/binary.mojo`, which is the boundary crossing the nodes needed. The arithmetic and comparison loops take their dtype as a parameter and a frame does not have one, so this promotes the two operand types to a common type with the same `promote` a concat and a coalesce use, converts both sides to it, and only then resolves the dtype and calls the loop. The promotion is what makes the answer's type depend on the types rather than on the values: int32 with float32 is float64 here for the same reason it is float64 in NumPy, and uint64 with a signed type is float64 and lossy above 2^53 in the same way NumPy is lossy. Being wrong the way people expect is worth more than being right in a way that makes an expression's type depend on what is in the column.
- Division is not part of the arithmetic family. It answers float64 whatever went in, which is what `/` does in pandas, so it promotes to check that the operands are numbers and then hands the promoted type to a loop that reads its operands at their own width and answers float64 anyway.
- Two things are deliberately absent and both want the same missing piece. There is no comparison on text, because the loops underneath are over fixed width registers and a string comparison against a second column's bytes is a kernel rather than an arm of a dispatch, and there is no constant on either side, so `x > y` can be written and `x > 5` cannot. What both need is a value type that carries its own dtype, which is also what the series level reductions have been waiting on since #101. It should be one change rather than three partial ones.
- `Value`, the type-erased scalar, in `firepanda/array/value.mojo`. A column is a `Value` repeated, so this is to one element what `AnyArray` is to a column: at the frame boundary a dtype stops being a parameter and starts being a field. The payload is a union written out as fields, with the type acting as the tag. An integer or a bool is held in the bit pattern of its own dtype and the sign is put back by the reader rather than kept in the store, so a uint64 above the top of int64 round trips and so does a negative int64 read back as a float. A float is held at full width, so a float32 that went in exact comes out exact.
- A null is a value in that type rather than the absence of one, and it carries the type it would have had. That is not a nicety: pandas answers NaN for the mean of an empty column and 0.0 for the mean of a column holding one zero, and those two answers have to be distinguishable before anybody can look at them. A null reads as zero if a caller reads it anyway, which is the rule the column layout already follows.
- A constant on either side of an expression, which is what `Value` was written for. `x > 5` can be spelled now, and so can `5 - x`, and this is most of what a query actually contains. The constant forms are their own loops rather than a column of a repeated value pushed through the pair forms, because splatting the constant into a register once means each row costs one load instead of two, which on a column that does not fit in cache is the whole difference.
- The promotion looks at the constant's type rather than its value, so `x + 1` on an int32 column stays int32 and does not widen because a literal happened to arrive as an int64. An expression's type stays a function of things the plan can see.
- `5 < x` is answered by turning it into `x > 5` rather than by a second set of comparison loops, and the swap is exact for floats too, since both readings are false when either side is a NaN. Subtraction and division cannot be turned round that way, so they carry a flag whose branch sits outside the loop. A null constant is answered before any loop runs, because every row is null whatever the operation and whatever the column holds.
- `Compute` takes a constant operand through a second constructor rather than through a second node type, since a plan that had to choose between two node types every time it walked an expression would be choosing on something that makes no difference downstream. `x > 5` and `x > y` produce the same shape of output and break a pipeline in the same way, which is not at all. Both forms bind at plan time, so a constant with no answer against that column's type is still an error before a row moves.
- The new kernels have scalar twins in `kernel/scalar.mojo` and the kernel fuzzer runs all eight of them against those twins on every case, both directions of subtraction and division included.
- Comparison on text, in `firepanda/kernel/text.mojo`, which is the second of the two things listed above as deliberately absent. Six operations over two text columns and six more over a column and one string, all of them byte comparisons rather than register comparisons, which is why they are a file of their own and not another arm of the dtype dispatch. A text column's physical dtype is uint8, so the variable width case is answered before the dispatch is reached and never gets the chance to read a view byte as a value.
- Equality and ordering take different routes through the layout on purpose. An element of twelve bytes or fewer lives whole inside its own sixteen byte view, zero padded, so two short elements are equal exactly when their views are and the answer is four register compares with nothing loaded. Ordering cannot use that, because the bytes are packed into words in the order that makes equality one compare and makes ordering wrong, so it goes to the byte loop. The constant form turns a short constant into a view once, before any row is read, which is the shape a filter on a status column or a country code has.
- A comparison touching a null is null rather than false, the same three valued logic the numeric comparisons follow, and it is arrived at the same way: the loop writes whatever falls out and the repair pass clears the rows where either side was missing.
- Arithmetic on text is still an error and it is the same error it was. There is no common type between a string and a number, and adding two strings is a concatenation, which is a function here rather than an operator.
- Two scalar twins for the text kernels, written against `String` and the language's own comparison operators rather than against views and payloads, so an error in the way the kernel walks the layout cannot hide in a twin that walks it the same way. The string fuzzer runs both against them, comparing the column against a gather of itself so that pairs holding the same bytes turn up often, which two independently drawn columns would almost never do.

## [0.6.32] - 2026-09-02

Built against Mojo 1.0.0 (ed45d567).

This release is three changes to one kernel, the factorize, which is the thing that turns a key column into a group number and is the first step of every group by and every hash join in the tree. All three are about the same observation: the routes it had were written for columns with few distinct values in them, and the benchmark suite is full of columns with a hundred thousand or a million.

The first widens the direct table. A key whose values fall in a known narrow range does not need a hash table at all, it needs an array indexed by the value, and the old rule refused any such array wider than sixty five thousand slots on the grounds that it should fit in cache. The new rule is that a direct slot is four bytes where a hash slot is sixteen and the hash table needs a set of them per worker, so a direct table of a slot per four rows is smaller than the hash it replaces even when most of it is empty. On ten million rows that is a hundred thousand values in 12 ms against 36, a million in 22 against 99, and two and a half million in 36 against 129.

The second and third replace the shape of the parallel route when the groups are many. The old route cuts the rows into contiguous slices and gives each worker a table over its own slice, which is right when a slice holds a few of the groups and wrong when it holds nearly all of them, because then every worker builds nearly the whole answer and a merge folds the copies back together. The new route cuts by hash instead. Two partitions cannot share a key, so each table holds only its own share of the groups and there is no merge at the end at all. What that costs is first appearance order, which the slice route got for free from the shape of its own work, and `_rank_by_first_row` buys it back by numbering the groups by the row that introduced them, with no comparisons and a core per block. The gate between the two routes is one statement, `_crowded`, read by both so they cannot drift, and it flips where a table per worker leaves the shared cache. The numeric route landed first and the string route followed, and the argument is stronger for text, because a string group costs a byte comparison to establish rather than just a slot to hold.

Taken together on db-benchmark at 0.5GB in memory on an i9-13900K, q3 goes from 79 ms to 51, q5 from 52 to 29, q7 from 85 to 54, and q10 from 365 to 225. The probe that breaks q10 apart puts the whole six key ordinals pass at 161 ms where it was 301. Against pandas 3.0.5, Polars 1.44.1 and DuckDB 1.5.5 on the same box and the same run, firepanda is now the fastest of the four on q3, q5, j4 and j5, and holds the smallest peak resident memory of the four on every query in the suite.

### Changed

- `factorize` offers the direct table a slot for every four rows of a long column, where it used to refuse any table wider than sixty five thousand slots. The old bound was about a table fitting in cache and the new one is about a table being cheaper than the thing it replaces: a direct slot is four bytes against a hash table's sixteen, and the hash table needs its own set of them for every worker, so a direct table of this width is smaller than the hash even when most of it is empty. The join's build side has drawn its line in the same way all along, at a slot a row, and it takes a wider table because it has two passes to pay for it rather than one. A column of under a quarter of a million rows keeps the bound it had.
- Measured on ten million rows of int64 on the i9-13900K, direct against hashed on the same groups, span fully occupied: a hundred thousand slots is 12 ms against 36, a million is 22 against 99, and two and a half million, which is the new ceiling, is 36 against 129. The other shape, a wide span with hardly anything in it, is what the old limit was written to decline, and there the two routes are close: a million slots holding a thousand values is 11 ms direct against 7 hashed, five million holding a hundred is 12 against 13, and a million holding a hundred thousand is 17 against 33. What the sparse cases cost the direct route is the scan and the ordinals rather than the table, since a table with few values in it only ever touches a few slots, so the penalty is a fixed few milliseconds while the gain on the dense cases grows with the span. The ceiling is set below where the direct route stops winning, because a table of one byte a row cannot be a surprise in anybody's memory budget and everything above it is already handled.
- On db-benchmark at 0.5GB in memory, four engines in one run on the i9-13900K, this is q5 from 52 ms to 29 ms and q10 from 365 ms to 353 ms. Both group on an integer key of a hundred thousand values, which is a span the old bound declined by a factor of one and a half. q5 is now the fastest of the four by a wide margin, against 54 ms for Polars, 79 for DuckDB and 261 for pandas, and it uses less memory than any of them.
- `factorize` takes a third route on the hashed side, one that cuts the rows by hash rather than into contiguous slices, when a hash table per worker would not fit in the shared cache. The slice route has every worker build a table over its own slice of the rows, and a slice of a column with a million groups in it holds nearly all million of them, so eight workers hold eight million entries that the merge then folds back down to one. Cutting by hash instead makes the partitions disjoint, no two of them can hold the same key, each table holds only its own share of the groups, and the merge is not needed at all. The cost is three passes over the column rather than one, at twelve bytes a row, which is why the route is only taken where the merge it deletes is worth more than the passes it adds.
- The gate is the same statement that already sized the slice route, now named `_crowded` and read by both, so the two cannot drift. It flips where a table per worker leaves the shared cache, which is about thirty three thousand groups on a thirty two thread machine, and the measurements put the crossover in the same place: at ten thousand groups the slice route wins 8.9 ms to 16.5, at a hundred thousand the partitioned route wins 17 ms to 28. No new constant was needed.
- First appearance order is what made this harder than it looks, and `_rank_by_first_row` is the answer. The slice route gets that order out of the shape of its own work, because its workers hold contiguous slices in order, and a hash partitioned build has no such luck. So the groups are radix partitioned by which block of rows their first row falls in, and then each block orders its own by writing every group into a scratch table indexed by the row itself. That needs no comparisons, because two groups cannot have been introduced by the same row, and every block does it on a core of its own.
- Measured on ten million rows of int64 on the i9-13900K, medians of three, serial against the best worker count of each parallel route: every row its own group is 341 ms serial, 123 sliced and 40 partitioned, a million groups is 140, 78 and 23, and a hundred thousand is 77, 28 and 17. Below the gate the partitioned route bottoms out around 16 ms whatever the cardinality, which is its three passes and nothing else, and that is the floor the gate exists to keep away from.
- On db-benchmark at 0.5GB in memory on the i9-13900K, this is q10 from 332 ms to 253 ms, and `tools/probes/q10_ordinals.mojo` says where it went: the final factorize of the packed six key value is 83 ms where it was 178, and the whole ordinals pass is 200 ms where it was 301. The next largest piece of that pass is now the string key at 64 ms, which still takes the slice route, so the same treatment applied to `_factorize_strings_parallel` is the follow up.
- `factorize_strings` takes the partitioned route too, on the same gate and for the same reason, and the reason is stronger for text than it is for numbers. What the slice route wastes is a table entry per worker per group, and a string group costs a byte comparison to establish rather than just a slot to hold, so doing that work once instead of once per worker is worth more here. The merge this route does not do is `_merge_strings`, which probes on the hash and then compares the two rows behind every match, rather than `_merge_hashed`, which only probes.
- `HashTable.build_strings` gained the one thing that made it possible, which is a way to be told that its positions are not rows. A partition holds rows from all over the column, so the chunk's own positions still say where the hash is and where the ordinal goes, and a lookup says which row of the column each position stands for. Everything that asks the column a question, which is the validity, the comparison and the representative row, goes through the lookup, and everything about the chunk does not. It is a comptime parameter, so the ordinary build compiles to what it compiled to before. The numeric `build` needs none of this because it never looks at a row.
- Measured on gamingpc, db-benchmark at 0.5GB in memory: q3 is 51 ms where it was 79, q7 is 54 where it was 85, and q10 is 225 where it was 253. All three group on id3, a string column of a hundred thousand distinct values, and `tools/probes/q10_ordinals.mojo` puts that key at 35 ms where it was 64. The whole six key ordinals pass is now 161 ms, against 301 before the numeric partitioning and 200 after it.
- Against the other three engines on that run, q3 at 51 ms is now the fastest of the four, ahead of DuckDB at 57 and Polars at 113, and it holds the smallest peak RSS of the four on every query in the suite.

## [0.6.31] - 2026-09-02

Built against Mojo 1.0.0 (ed45d567).

This release is the engine itself, plus the last of the foundation it needed. Nothing here changes an API, an answer or a number, because nothing in the tree calls the engine yet. That is the point of shipping it this way: the machinery lands first and correct, the operators move onto it one at a time afterwards, and each of those moves is a change small enough to have a benchmark of its own.

The engine is a source, a line of nodes and a sink, with chunks pushed along the line. A node has three methods and the fallback node runs anything that has not been ported by collecting the chunks, calling today's whole frame function and handing the answer back in chunks. That fallback is what makes every later change a pull request rather than a rewrite.

Under it is the sortedness flag, which is the thing a sorted group by and a merge join will read. Neither of those operators exists yet, and the flag is cheap enough that having it in place before them costs nothing and having it missing would block both.

### Added

- The engine, in `firepanda/exec`: a `Chunk`, a `Node`, and a `Pipeline` that pushes chunks along a line of nodes from a source to a sink. A chunk is a horizontal slice of a frame, one array per column and a row count, and nothing else. It carries no schema, because names are a property of the plan and the plan is fixed before the first row moves. A node has three methods, `update_state`, `process` and `finish`, which is the shape Polars uses for `ComputeNode`. The driver takes a chunk from the source, hands it along the line, and puts whatever reaches the end into the sink.
- `Materialize`, the fallback, which is the thing that makes the rest of this shippable in pieces. It collects every chunk, calls a whole frame function, and hands the answer back in chunks. Any operator with no chunked implementation goes in one of these and runs exactly as it does today, in a pipeline beside operators that have been ported, so each later change removes one fallback and comes with a benchmark of its own. Polars shipped `InMemoryMap` and `InMemoryJoin` for the same reason and still has them eighteen months on. The function it wraps captures nothing, which is a deliberate limit: an operation that needs an argument has state, and the place for state is a node with fields rather than a closure that hides it.
- `Filter`, `Project` and `Limit`, the first three real nodes. `Filter` reads a boolean column of its own chunk by position, which is what makes it an operator rather than a call, since whatever computed the mask is another node earlier in the line. `Project` keeps columns in the order the plan chose and moves them rather than copying, so the ordinary projection copies nothing at all, and a column named twice is copied once and moved once. `Limit` passes rows through until it has the number it was asked for and then reports FINISHED, and because a pipeline is a line, one finished node makes everything upstream useless and the driver stops reading the source. That is limit pushdown falling out of the interface rather than out of an optimizer pass.
- Pipeline cutting at breakers. A breaker is a node that cannot emit until it has seen all its input, which today means `Materialize`. `cut_points` finds them in one traversal, for looking at a plan. The driver does not need it, because draining the operators in order after the source is empty is the same execution, with the breaker itself holding the material between the two stages.
- `ChunkedArray.into_chunks` and `DataFrame.into_columns`, which are the ways out of those types for a caller that wants the pieces rather than one contiguous array. The scan is the caller: it takes a frame apart into the chunks it was already made of, copying nothing, and the frame is gone by the time the first chunk moves.
- What is not here yet is parallelism. The driver takes chunks one at a time on the calling thread, and the morsel queue next door is what will hand different chunks to different workers. Every node is already written for that, since each is a function of its own chunk and its own fields. A pipeline whose answers are wrong is not improved by computing them on thirty two cores.
- A sortedness flag on a column, which is the last of the M2b foundation items. Two operators want it. A group by on a key whose equal values are adjacent needs no hash table at all, it walks the column and closes a group when the value changes, and a join of two sorted inputs is a merge and needs no hash table either. Neither of those exists yet; this is the thing they will read. `sort_values` sets it on the most significant key, which is the only key that comes out sorted over the whole frame, and it sets it without a check because the sort just did the work that establishes it. Any other column costs one pass the first time it is asked and remembers the answer, including a negative one, and the pass walks the chunks and the seams between them rather than flattening the column to ask a question about its order.
- `is_monotonic_increasing` and `is_monotonic_decreasing`, on both `Series` and `DataFrame`, which is what a pandas user calls the flag. A constant column answers true to both, as pandas does, so the scan records that as a state of its own rather than picking one of the two arbitrarily and having to be rerun to answer the other question.
- `is_sorted_any`, the runtime dtype dispatch over the typed `is_sorted` that has been in `firepanda/kernel/sort.mojo` since M1, with a text arm of its own because a string column's order is over its bytes rather than over a number.
- Null placement is deliberately not recorded. A column holding a null is never marked and never claims to be monotonic. The two ends a sort can put nulls at would need two more states, and neither operator that wants the flag can use them: a merge join has to know exactly, and a group by would rather read the null count, which is already a field, and take the ordinary path.
- Chunk walking entry points for the elementwise kernels, in `firepanda/kernel/chunked.mojo`, and `filter`, `slice`, `cast`, `astype`, `select` and `drop` now go through them. A row's output in that family depends on that row and nothing else, so the operation is cut the same way the column is cut, the existing contiguous kernel runs once per chunk, and the pieces are kept rather than stacked. A column read from a file with sixteen row groups stays sixteen chunks through a filter and a cast and never has to exist as one contiguous array. A filter that keeps nothing leaves a column with no chunks at all, because an empty chunk in a prefix sum would let one row position name two chunks. `slice` copies the chunks it takes whole rather than cutting and rejoining them, which is what `head` and `tail` want: a hundred rows out of ten million now copies at most one chunk's worth of bytes.
- `take` is the one in the family that does not fit, and it says so rather than pretending. Its output row `i` comes from input row `indices[i]`, which can be in any chunk, so there is no cut of it into independent per chunk calls that does not also need the pieces put back in order, and putting them back in order is another gather. It flattens a column of several chunks first, at the cost of one copy of the input, and a column of one chunk, which is every column in the tree today, skips that entirely. The selection vector work later in M2b is what fixes it properly.
- Nothing here runs chunks in parallel with each other. The kernels underneath are already parallel inside, and the last chunk of a column is usually smaller than the rest, so a task per chunk would leave cores idle at the end of every operation.

## [0.6.30] - 2026-09-02

Built against Mojo 1.0.0 (ed45d567).

This release is the foundation of the new engine plus the last of the group by parallelism that came before it. Nothing here changes an API and nothing here changes an answer.

A frame's columns are chunked now. That is the shape the things which produce columns already have, one array per Parquet row group and one per input to a concat, and turning those into contiguous memory costs a copy of the whole column that a scan, a filter or an aggregate does not need. Nothing yet builds a column of more than one chunk, so this release is the type change and the plumbing rather than the benefit, and the benefit arrives as the operators learn to walk chunks. Getting there took three steps: `ChunkedArray` gained a prefix sum, a running null count and a borrowing accessor for the single chunk case, then the seven helpers that read several of a frame's columns learned to borrow them instead of being handed the list, then the frame's field changed type.

The borrow is the piece worth knowing about if you are writing against the internals. It carries the origin of what it points into rather than erasing it, because an erased origin lets the compiler destroy a frame passed inline after the argument is evaluated and before the callee runs. That does not crash, it reads freed memory that still looks like a column, and a group by over eight rows with three distinct keys answered one group. With the origin in the type that program does not compile.

An operator that has not been taught about chunks raises on a column that has more than one rather than answering from the first, because a wrong answer over the first row group of a file looks exactly like a right one. Printing is the exception and says so where the table would have been.

The performance in this release is all from the group by side. Gathering rows from a text column runs on every core now, which was the last single threaded kernel in a group by and the largest single phase of db-benchmark q10. Each worker adds up the payload its own range will copy, the totals prefix sum into a base per worker, and each worker then writes its views and its payload independently. A column with an empty payload skips the counting pass, which is every column of labels. Separately, a group by asks each key column whether it has any nulls before walking the groups for `dropna`, which was sixty million random validity lookups on six keys of ten million tuples to discover that none of them had a null. It became a popcount, and once columns were chunked it became a field read.

On the i9-13900K at 0.5 GB, medians of five: the text gather went from 466.9 ms to 22.6 ms and q10 went from 909.3 ms to 352.0 ms. That is past Polars 1.44.1 at 749.8 ms, having been well behind it, and within about one and a half times DuckDB 1.5.5 at 216.5 ms. The other nine queries are unchanged.

### Changed

- A frame's columns are `ChunkedArray` rather than `AnyArray`. That is the shape the things which produce columns already have: a Parquet file gives one array per row group and a concat gives one per input, and flattening those into contiguous memory costs a copy of the whole column that a scan, a filter or an aggregate does not need. Nothing builds a column of more than one chunk yet, so every column still has exactly one, and every method reaches through `only()`, which borrows that chunk and copies nothing. The wrap on the way in moves rather than copying, which the frame tests now check by the address of the values buffer, and a column that does have more than one chunk raises at the borrow rather than being read as its first chunk, which the tests also check by building one by hand. A `dropna` null check got cheaper on the way past, because a chunked column keeps a running null count and the popcount it replaced is now not done at all. db-benchmark at 0.5 GB on an i9-13900K is unchanged within noise on all ten queries, which is the point of the PR rather than an aside: q10 353.0 ms before and 352.0 ms after, q6 379 and 354, j1 37 and 37.
- The seven helpers that read a set of a frame's columns now borrow them instead of taking the frame's own list. A group by, a join, the table renderer and the all-present mask a `dropna` builds all want to look at several columns and own none of them, and they were handed `frame.columns` directly, which works only for as long as the frame's columns are `AnyArray` and the caller holds the frame rather than a reference to it. Neither of those survives the frame holding chunked columns, and the alternative on the day it stops working is a copy of every column on every call. They now take a list of pointers, and `DataFrame.column_refs()` builds it. `dropna` with no subset stopped deep-copying every column on the way in as a side effect of the same change. The borrow carries the origin of what it points into rather than erasing it, which is not a formality: with the origin erased the compiler destroys a frame passed inline after the argument is evaluated and before the callee body runs, and a group by over eight rows with three distinct keys then reads freed memory and reports one group instead of crashing. With the origin in the type that program does not compile.
- `ChunkedArray` is ready to be the frame's column type, which is the first step of M2b. It has been in the tree since M0 with nobody using it, because a reader that produces one array per row group had nothing to hand it to. Three things were missing. Finding the chunk a row lives in walked the chunks adding lengths up again on every call, and is now a binary search over a prefix sum that is maintained as chunks arrive. The null count did the same walk and is now a running total. And there was no way to get at the single chunk of a column that has one without copying it, which matters more than the other two put together: almost every column in the system has exactly one chunk, every kernel in the tree is written against `AnyArray`, and if the bridge between those two were a copy then putting this type in front of the frame's columns would reintroduce the whole-column deep copy that was taken out of `take` and `filter`, on every operation. `only` is that bridge and it returns a reference. `combine` flattens a column when something genuinely needs contiguous memory, and at one chunk it is a move that copies nothing, which is what lets an operator that has not been taught about chunks stay correct while the port is in progress. An empty chunk is dropped on append rather than kept, because two equal entries in the prefix sum would let a row position name two chunks.
- Gathering rows from a text column runs on every core. It was the last kernel in a group by still doing ten million random reads on one thread, and on db-benchmark q10 it was the largest single phase of the query, larger than the ordinals and twenty eight times the cost of gathering the same rows from three integer columns. A gather does not change how an element is stored, so a short value stays short and a long one keeps its length, which means an output row's view is sixteen bytes wide wherever it lands and the only thing that depends on the rows before it is where its payload bytes go. Each worker adds up the payload its own range will copy, the totals prefix sum into a base per worker, and then each worker writes its views and its payload with a cursor of its own. A column with nothing in its payload skips the counting pass entirely, which is every column of labels and is the case a group by's key gather actually hits.
- A group by asks each key column whether it has any nulls before deciding whether `dropna` has anything to do. The check was a validity bit lookup per group per key with a random index behind it, so a group by on six keys with ten million tuples between them did sixty million of them on one thread to discover that none of the keys had a null in the first place. It is now a popcount over each key's validity, six passes over a megabyte and a quarter, and the groups are walked only for the keys that can actually answer yes.
- db-benchmark group by at 0.5 GB on an i9-13900K, medians of five: q10, a sum and a size over ten million groups on six keys, went from 909.3 ms to 353.0 ms. That is past Polars 1.44.1 at 749.8 ms, having been well behind it, and within about one and a half times DuckDB 1.5.5 at 216.5 ms. The two changes are worth roughly 170 ms and 350 ms of that in the order they are listed above.

### Added

- `tools/probes/q10_phases.mojo`, which splits db-benchmark q10 into the ordinals, the key gathers and the two reductions. It is checked in because of the split it found rather than the numbers it prints: on the query's own shape the ordinals were 303 ms, the three text gathers were 467 ms and everything else together was 42 ms, which is not where the work had been going.

## [0.6.29] - 2026-09-02

Built against Mojo 1.0.0 (ed45d567).

This release is one thing done seven times: a group by now runs on every core it is given, from the first pass over the key columns to the last write of the result. Nothing here changes an API and nothing here changes an answer. Every number a query produced before it produces after, in the same order, including the first-appearance ordering the factorize contracts promise.

The starting point was a measurement rather than a hunch. Running db-benchmark at 0.5 GB on a machine with thirty two logical cores and recording CPU seconds against wall seconds showed firepanda using between two and six cores on the queries where DuckDB used twenty two to twenty six and Polars used eight to twenty one. The wall clock gap was not an algorithmic one. It was that most of a group by was still a single loop on a single thread.

Both halves are now parallel. On the ordinals side, the merge at the end of a hashed factorize buckets the workers' groups on the top bits of their hash so the buckets have nothing to agree about, the direct route an integer key takes builds a private table per worker in two passes rather than three, the multi key packing is vectorized and cut into morsels, and reading a type-erased column as a typed one stopped deep-copying every byte. On the reduction side, the sums, counts, means and extremes build a private accumulator table per worker and merge them, the variance and correlation family does the same for both of its passes, the per group sorts behind the median and the distinct count are cut on an eight group boundary so no two workers share a byte of validity, and the case with too many groups to replicate a table at all now partitions the rows instead of giving up.

Three of the changes are about memory bandwidth rather than about threads, and they are worth naming because they were each larger than they look. `Buffer(copy=)` zeroed an allocation and then overwrote every byte of it, making a copy three passes over memory where two would do. Two columns that are fully written by the pass that follows were being zeroed first. And the typed view of an erased column was a copy, which a group by paid once per key column and a join once per side.

Measured on an i9-13900K, db-benchmark group by at 0.5 GB, memory io, medians of five runs, against 0.6.28. q3 went from 136 ms to 79 ms, q6 from 537 ms to 388 ms, q7 from 114 ms to 84 ms, q9 from 114 ms to 62 ms, q5 from 69 ms to 52 ms and q10 from 1.02 s to 870 ms. q6 is now ahead of DuckDB 1.5.5 on the same box and q9 is nearly four times faster than Polars 1.44.1. Peak resident memory is the lowest of the four engines on every query in the suite, which it already was and which none of this gave up.

What is not fixed: q10 at 870 ms against DuckDB's 217 ms is still the largest absolute gap, and its remaining cost is building ten million group ordinals and materialising ten million output rows across eight columns rather than the aggregation this release sped up. The joins are the other outstanding place, j1 and j3 both around 39 ms against DuckDB's 18 and 15.

Three probes are checked in under `tools/probes/` alongside the changes they justified, `group_phases.mojo`, `factorize_workers.mojo` and `direct_phases.mojo`. They are there so the next person to move one of these thresholds has the curve it was fitted against rather than a constant with a story attached.

### Changed

- A grouped sum, count, size and mean over more than about two million groups partitions the rows instead of giving up and running on one core. Past that point a private accumulator table per worker does not fit inside `PRIVATE_BYTES` and the reduction fell back to a single serial scatter, which is the worst case it has: every row is a random write into a table far larger than any cache, so each one moves a line in and a line out and the loop spends its whole time waiting on memory. The partitioned route cuts the group range instead of replicating the table. Each partition owns a run of ordinals no other partition holds, so there is one output table, no locks and no merge, and the run is a quarter of a megabyte so it sits in a core's private cache while it is being folded. It costs three passes over the column where the replicated route takes one, which is why it is taken only where the replicated route has given up entirely.
- db-benchmark group by at 0.5 GB on an i9-13900K, medians of five: q6, a median and a standard deviation over six million groups, went from 487.0 ms to 388.0 ms, and q10, a sum and a count over ten million, from 909.3 ms to 870.0 ms. q6 is now faster than DuckDB 1.5.5 on the same box, which it was not before.
- The variance, the standard deviation, the covariance and the correlation run on every core. These were the grouped reductions the parallel scatter never reached, and the reason is that each of them takes two passes over the column rather than one: the first works out every group's mean, the second accumulates deviations from it. The first pass was already parallel because it goes through the mean and the count, and the second one was still a serial loop, so a standard deviation cost more than the mean it is built on and used one core to do it. Both passes now build a private table per worker and merge them with the same vectorized adder the sums use. A correlation carries six tables rather than one, so the memory ceiling that decides the worker count is asked about six times the group count.
- The per group sort behind the median, the quantiles and the distinct count runs on every core. Those three reductions gather each group's values into its own run of a slab and then walk the groups one at a time, and one group's run is nobody else's, so this needs no private tables and no merge, only a cut of the group range. The cut falls on a multiple of eight because eight groups' presence bits are one byte of the output validity and clearing a bit is a read, a mask and a write, so two workers on the same byte would lose one of the two clears. Eight float64 results are also a cache line, so the same rounding keeps the writes off each other's lines.
- db-benchmark group by at 0.5 GB on an i9-13900K, medians of five: q9, which is a squared correlation by two keys, went from 113.8 ms to 58.0 ms, and q6, which is a median and a standard deviation by two keys, from 536.9 ms to 487.0 ms. q9 is now nearly four times faster than Polars 1.44.1 on the same box. q6 moved much less because its keys produce six million groups, which is past the point where a private table per worker fits in memory at all, so its mean, its count and its variance are all still serial scatters. Partitioning the rows by group ordinal is what that case wants and it is a separate change.
- A group by on more than one key packs the keys together on every core. Each key is factorized on its own and the ordinals are combined into one integer per row by widening the first key's and then multiplying and adding each later key's into it, and those passes were plain serial loops. On ten million rows each of them moves more memory than the factorize that produced its input, and a group by on six keys runs five of them. They are vectorized and cut into morsels now, like every other whole column loop in the engine. On ten million rows the ordinals for two keys went from 42.6 ms to 30.5 ms and for six from 454.0 ms to 341.8 ms.
- The packed key column is allocated unzeroed, because the pass that widens the first key's ordinals into it writes every row.
- Reading a type-erased column as a typed one no longer copies it. `AnyArray.as_typed` deep-copies every byte the column holds, and until now it was the only way to get an `Array[dt]` out of a borrowed `AnyArray`, so a group by paid it once per key column, a join paid it once per side, and `dispatch_typed` paid it for every kernel that wants a typed column rather than an erased one. `as_typed_view` is the borrowing version. `Array[dt]` holds one field and a struct of one field has that field's layout, so a reference to the storage is a reference to the array, which is asserted in `tests/test_array.mojo` rather than assumed. On ten million rows the ordinals of a group by on one integer key went from 7.2 ms to 4.8 ms at a hundred distinct values and from 12.3 ms to 7.4 ms at sixty five thousand, which is the copy and nothing else.
- Group by ordinals over ten million rows, medians of five, against where they stood two releases ago: a hundred distinct integer keys 20.0 ms to 5.9 ms, ten thousand 11.0 ms to 6.4 ms.
- The direct route of a factorize runs on every core when the value range is narrow. It is the route an integer key column takes when its range is small enough to index a table with the value itself, which on db-benchmark is `id4`, `id5` and every group by on more than one key, and it was the last single threaded loop left in the ordinals. Each worker builds a private table over the range, the merge reads the same slot out of all of them at each value and takes the first worker that has it, and because equal values land in the same slot the merge has no probing in it at all. First-appearance ordering comes out of the same `_rank_entries` pass the hashed merge uses, over the same entry order, so the ordinals are identical to the serial route's and not merely an equivalent grouping.
- The split is two passes over the column and not three, which is the difference between it being worth doing and not. The obvious arrangement has the first pass write a local ordinal per row and a third pass rewrite those into the merged ones, and measured that way two workers were about half again slower than one thread on every range tried. The direct loop is short of memory bandwidth rather than of arithmetic, so an extra read and write of a forty megabyte ordinal column costs more than a second core buys. So the first pass only discovers groups, the merge projects itself back down onto a single table indexed by value, and the second pass reads that table and writes each row once.
- The direct route is taken in parallel only while the merge's own footprint stays small, `DIRECT_MERGE_BYTES`, a quarter of a megabyte. The merge touches one cache line per worker per value, so its footprint is the range times the worker count, and past a core's private cache every value in it is a miss. Ten million rows on an i9-13900K, medians of seven: a range of a hundred is 4.05 ms serial and 2.63 ms on every core, a range of a thousand is 4.21 ms against 2.71 ms, a range of ten thousand wants six workers at 3.94 ms against 4.18 ms, and a range of sixty five thousand is better off on one thread and now stays there. There is a floor of four workers for the same reason two were slower than one.
- Copying a buffer no longer zeroes it before overwriting it. `Buffer(copy=)` allocated the zeroed way and then memcpyed every byte of the allocation, which made a copy three passes over memory where two would do. It matters because a group by makes a typed copy of every key column it is handed, so a forty megabyte key column paid for it on every query. The same ten million row group by went from 12.0 ms of ordinals to 9.9 ms on that change alone.
- A direct factorize allocates its ordinal column unzeroed. Every row is written by the route, including a zero where the value is null, so the pass that zeroes forty megabytes first was work with no reader.
- What that adds up to on a group by, ten million rows on an i9-13900K, medians of five. The ordinals over a hundred distinct integer keys go from 20.0 ms to 9.9 ms and over ten thousand from 11.0 ms to 9.2 ms.
- `tools/probes/direct_phases.mojo`, which splits the direct route into the range scan, the serial build, the build at every worker count, the whole factorize and the whole `group_ordinals`. It is checked in because of the gap it exposes rather than the numbers it prints: the factorize is 4.4 ms and `group_ordinals` around it is 7.2 ms, and the 2.8 ms in between is the typed copy `_factorize_any` makes of the key column. Removing that copy is the next thing and it is worth more than this change was.
- The merge at the end of a parallel factorize runs on every core. Each worker builds its own hash table over a slice of the column and the merge folds those into one numbering, and it was the one part of the route no thread could help with: it walked every group every worker had found and probed it into a single growing table. On ten million rows with a hundred thousand distinct string keys that was 27 ms of a 99 ms factorize. Equal keys hash equally, so bucketing the workers' groups on the top bits of their hash gives every bucket a key range no other bucket can hold, and the buckets have nothing to agree about. First-appearance ordering, which is a documented contract on `Factorized.codes` and `Factorized.firsts`, survives because the bucketing is stable and because the final numbers come from a prefix sum over the entries in worker order rather than from the order a bucket happened to run in.
- The worker count for a parallel factorize comes from the cache rather than from a cost model. The model existed to decide when the serial merge had eaten the split, and there is no serial merge any more. What is left is that every worker's table ends up holding roughly all of the column's groups, so the tables together are the group count times the worker count: while that fits in the shared cache every probe is a cache hit and the answer is every core, and once it does not every probe is a trip to memory and the answer is half of them. Measured against every worker count from two to thirty two on ten million rows, on integer and text columns of a hundred, a hundred thousand and a million groups, the rule is within about a tenth of the best time on all six. `REMAP_SHARE`, `MERGE_COST` and `SPLIT_MARGIN` are gone and `TABLE_BYTES_PER_GROUP`, `SHARED_CACHE_BYTES` and `CROWDED_SHARE` replace them.
- A column where every row is its own key now takes the parallel route. It was refused before, correctly, because the merge over one group per row was the whole column again on one thread. Ten million distinct integers factorize in 129 ms on sixteen workers against 174 ms serial.
- What that adds up to on the key shapes db-benchmark actually uses, ten million rows on an i9-13900K, medians of five. One string key of a hundred thousand distinct values goes from 100.7 ms to 64.4 ms, one integer key of the same from 40.6 ms to 38.1 ms, one string key of a hundred from 19.0 ms to 14.3 ms, and all six of the suite's key columns at once from 432.9 ms to 333.0 ms. Through the whole query, db-benchmark group by at 0.5 GB: q3 0.136 s to 0.078 s, q7 0.114 s to 0.084 s, q5 0.069 s to 0.055 s, q2 0.058 s to 0.051 s and q10 1.020 s to 0.895 s. That puts firepanda ahead of Polars 1.44.0 on q2, q3, q5, q7 and q9 and ahead of DuckDB on q5, with the lowest peak resident memory of the three on all nine.
- `tools/probes/factorize_workers.mojo`, which is the sweep the worker count rule was fitted against. It prints the whole curve, serial and every worker count from two to thirty two, for integer and text columns at four cardinalities, so the next person to move the rule has the same picture rather than a constant with a story attached.
- A grouped sum, count, size, mean, minimum and maximum runs on every core instead of one. A scatter cannot be split by handing each worker a slice of the rows and letting them all write to the same table, because two rows in two slices can belong to the same group and the increments would be lost, so each worker gets its own table and the tables are added together afterwards. That is the same shape `firepanda/hash/factorize.mojo` already uses for the same reason. Ten million float64 rows on an i9-13900K, medians of five: a sum over a hundred groups is 5.4 ms on one core and 1.9 ms on all of them, a mean is 8.5 ms against 2.8 ms, a maximum is 8.1 ms against 2.5 ms. At a hundred thousand groups a sum is 8.4 ms against 4.4 ms, and at a million, where the memory ceiling holds it to four workers, 16.4 ms against 10.0 ms.
- The route has a ceiling on it, `PRIVATE_BYTES`, because the merge costs `groups * workers` rather than `rows` and past a point it costs more than the scatter it is speeding up. Thirty two megabytes of tables, so a reduction over more than a million groups gets fewer workers than the machine has and one over many million stays serial. The answer for that case is partitioning the rows by code so that each worker owns a range of groups outright and there is no merge at all, which is a change to the executor rather than to the kernel.
- The grouped reductions now declare `raises`, because starting a worker is something that can fail and a reduction that stays on one core cannot. Nothing above them needed changing, because every caller was already in a raising context.
- `tools/probes/group_phases.mojo`, which times a group by phase by phase at four cardinalities. It is checked in because of what it found rather than what it does: on ten million rows the ordinals cost 20 ms at a hundred distinct keys, 50 ms at a hundred thousand and 144 ms at a million, against 2 to 10 ms for the reduction that follows. So a group by is mostly the factorize and this change speeds up the smaller half. The direct route in `firepanda/hash/factorize.mojo`, which is the one an integer key column takes, is still a single core loop and that is the next thing.

## [0.6.28] - 2026-09-01

Built against Mojo 1.0.0 (ed45d567).

This release is about the engine, and the thread that runs through it is that firepanda kept measuring itself doing work it did not need to do. A join factorized both sides when it could have built a table on the smaller one. A total was computed by grouping on a key that was the same for every row. A Parquet read converted DuckDB's columnar chunks into Arrow's columnar chunks. None of those were wrong, and all three were the cost of reusing a piece that already worked, which is the right first move and the wrong last one.

Joins got the largest share. Building the hash table on the smaller side and probing with the larger, instead of concatenating both key columns and factorizing the result, took db-benchmark j4 at 0.5 GB from 0.523 s to 0.180 s and j5 from 0.542 s to 0.171 s on an i9-13900K. An integer key whose build side has a narrow range skips hashing entirely and indexes a table by the value.

Reductions got the surprise. Timing db-benchmark's j1 phase by phase showed the join was 34 ms of a 131 ms query and the other 81 ms was the benchmark reducing five million joined rows to one by appending a column of zeros and grouping on it, because a frame had no way to reach the vectorized reductions that have been sitting in the kernel layer since M1. `DataFrame.agg` and `DataFrame.agg_all` are that way, and the same ten million rows now reduce in 6 ms rather than 81. j1 through j3 are around 0.042 s each, j4 and j5 around 0.096 s, which puts j4 at 2.1 times Polars 1.44.0 and j5 at 2.5 times, level with DuckDB, and with the lowest peak resident memory of the three on every one of the five.

Parquet stopped converting. Reading DuckDB's vectors directly is about eleven percent faster on an idle sixteen core machine and a good deal more on a loaded one, and the reason it is not larger is now written down in the phase timings rather than guessed at. `ParquetOptions` arrived alongside it, which turns the reader from something that reads a file into something that reads a dataset: Hive partitions, union by name, the source filename, and projection pushdown through all of them.

There is a JSON reader now, `read_ndjson`, one object per line, sixteen times faster than pandas 3.0.5 on a 166 MB file and not yet where it needs to be against Polars or pyarrow. Its scanner decodes nothing on the first pass and refuses everything malformed rather than guessing, which is half of what its tests are about.

Underneath all of it is `parallel_morsels`, a work stealing scheduler that hands out 128k row morsels from an atomic cursor instead of cutting a job into one piece per worker up front. On skewed work it is 3.9 times faster than the static split and on even work it costs nothing, and it is the shape every kernel in the new engine is being moved onto. `take` and the join emit are on it already, neither of them faster for it, both of them measured and written down here anyway, because the point of the move is the loop body's shape and not this quarter's number.


### Added

- `DataFrame.agg(specs)` and `DataFrame.agg_all(kind)`, which reduce a whole frame to one row. This is the last of `sum`, `mean`, `min`, `max` and `count` that was missing: the vectorized reductions have been in `firepanda/kernel/agg.mojo` since M1 and there was no way to reach them from a frame, so the only way to total a column was to group by a key that was the same for every row. That is not a small difference. Ten million float64 rows on an i9-13900K: 81 ms through the group by against 6 ms through `agg`, thirteen times, because the group by allocates forty megabytes of ordinals, walks them beside the values and scatters into a table with one entry in it, while the reduction is a vectorized add over the values buffer.
- `firepanda/kernel/reduce.mojo` and `reduce_any(col, kind)`, the dispatch behind it. Sum, mean, min, max, count and size take the whole column route. The variance, the order statistics and the distinct count go to `aggregate_group_any` with a single group, because they have no whole column spelling yet and a correct slow answer beats a missing one. Which reductions are on which side is one function, `_takes_fast_route`, so that adding a whole column variance later is a change in two places rather than a hunt. Every kind is tested against the grouped answer over the same column, including the slow ones, so a fast route that grew a branch and swallowed a kind would fail rather than quietly return something plausible.
- `sum_over`, `extreme_over` and `mean_over` in `firepanda/kernel/agg.mojo`, which take a values pointer, a validity bitmap and a row count rather than a typed column. The existing `_of` spellings are three lines each on top of them now. The reason for the second spelling is that turning an `AnyArray` into an `Array[dt]` copies the column, and a reduction that copies eighty megabytes before summing them is not a reduction.
- `firepanda/io/duckvector.mojo`, which reads a DuckDB data chunk's vectors directly rather than asking DuckDB to convert the chunk to Arrow first. A flat DuckDB vector of BIGINT is a contiguous array of int64, which is what an Arrow array of int64 is, so for every fixed width type the column is already in the shape we want and the work is describing it rather than copying it. Booleans are the one type DuckDB stores a byte at a time where Arrow wants a bit, and strings are the one type that needs the tail of each descriptor rewritten, which is sixteen bytes per row and not the string itself for anything under the twelve byte inline threshold that both layouts happen to share.
- `ParquetOptions` and a `read_parquet(path, options)` form, which is how a dataset is read rather than a file. `hive_partitioning` reads the `key=value` directories on the way to each file as columns, so a tree of `sales/year=2024/month=03/part-0.parquet` comes back with `year` and `month` columns that are in no file, and a query naming neither of them never opens the directories that cannot match. `union_by_name` lines files up by column name instead of by position, which is what a directory that grew a column halfway through its life needs. `filename` adds the path each row came from, which is how the one bad file in ten thousand gets found. `columns` is the same projection pushdown the two argument form already did, and it works on a partition column exactly as it does on a real one.
- `firepanda/io/jsonscan.mojo`, the JSON equivalent of `scan.mojo`. It takes a span of bytes and reports a span per key and a span per value, decoding nothing: a number stays the bytes it was written as until `parse.mojo` gets it on the column that turned out to be a number, and a string carries a flag saying whether a backslash is in it so the reader only unescapes the strings that need it. Nested objects and arrays are found and skipped rather than descended into, reported as one span covering the whole thing, because firepanda has no nested column type yet and a reader that silently flattened one would be inventing a schema nobody asked for. Escapes are handled on the way out, including surrogate pairs, so an emoji written the way JSON writes one comes back as itself rather than as two broken halves.
- Everything malformed is refused rather than guessed. A string that does not close, an escape that is not one of the eight JSON has, a `\u` that is not four hex digits, a number with a leading zero or with nothing after its decimal point or exponent, a missing colon, a missing comma, a key that is not a string: all of them are errors with the byte offset in them. Half of `tests/test_jsonscan.mojo` is those cases, because every one of them is a document some other reader accepts by inventing something, and an invented value in a data file is worse than a failed read.
- `read_ndjson(path)` and `read_ndjson_bytes(bytes)`, one JSON object per line, which is what `read_json(lines=True)` means in pandas. The columns are the union of every key in the file, so a line that leaves a key out gets a null there rather than being a ragged row, and the type ladder is the one `read.mojo` already uses for CSV so a column comes out the same type either way. A value that is not a string is kept as the bytes it was written as, which is what makes a column of mixed kinds and a column of nested objects both readable rather than both an error. Two million rows of four columns, 166 MB, everything inferred and no schema handed in, on an i9-13900K: firepanda 114 ms against pandas 3.0.5 at 1896 ms, Polars 1.44.0 at 47 ms and pyarrow 24.0.0 at 39 ms. Sixteen times pandas and not yet where it needs to be against the other two, for the reason recorded in issue 79.
- `parallel_morsels(body, total, rows)` in `firepanda/exec/morsel.mojo`, the scheduler the engine is being rebuilt on. `parallel_for` cuts a job into one piece per worker before any work is done, which is right when the pieces cost the same and leaves cores idle when they do not. A morsel queue decides nothing in advance: the work is a range of rows, the range is cut into 128k row morsels, and a worker that finishes one takes the next by bumping an atomic counter, so a worker that draws a cheap morsel comes back sooner and takes more of them. The tail is one morsel long rather than one worker's whole share. Eight million rows on an i9-13900K with 32 workers, where the first eighth of the rows costs fifteen times what the rest do: 89 ms for `parallel_for` against 23 ms for `parallel_morsels`, and 6 ms against 5 ms when the rows all cost the same, so the skewed case is 3.9 times faster and the even case pays nothing for it.
- `read_parquet`, `ParquetOptions` and `quote` are now exported from `firepanda.io`, which they should have been when the reader landed.
- The choice between the direct read and the Arrow conversion is made once for the whole result, by looking at the type of every column before any chunk is fetched. A result with a type `duckvector` does not read yet goes through `duckdb_data_chunk_to_arrow` exactly as it did before, all of it, because the assembler takes one layout and a result read half by one route and half by the other is two ways to be wrong instead of one.

### Changed

- A join builds a hash table on its smaller side and probes it with its larger, instead of concatenating both key columns and factorizing the result. The old route was chosen because it reuses every piece of key handling the group by already had, which is a real argument and cost a join against a small table a full factorize of the big one. On an i9-13900K over db-benchmark at 0.5 GB, medians of five runs: j4 0.523 s to 0.180 s, j5 0.542 s to 0.171 s, j3 0.193 s to 0.130 s, j2 0.183 s to 0.135 s, j1 0.139 s to 0.131 s. j4 and j5 now beat Polars 1.44.0, which takes 0.195 s and 0.231 s on the same data. j1 barely moves because its pairing was already a small part of it and what is left is materializing the joined columns, which is issue 79's next section rather than this one.

- The build side is picked by row count rather than by argument order. Either side works, because the ordinals only have to agree and not to mean anything in particular, so joining a ten row frame against a ten million row one costs a table of ten either way round.

- Single key joins on any fixed width dtype take the new route. Two or more key columns still go through the concat, because combining them into one ordinal is what `group_ordinals` does and doing it again here would be a second implementation to keep in agreement with the first. String keys go through it too, because the table compares hashes and a string needs its bytes compared on a hash match, against a column on the other side that the table has no way to reach.

- An integer key gets a table indexed by the value itself when the build side's range is narrow enough, which hashes nothing on either side and turns the probe into one load. The width worth accepting is the build side's own height rather than `DIRECT_LIMIT`, because the table is standing in for a hash table over the same keys at four bytes a slot against sixteen.

- Every atomic in the library now lives in `firepanda/exec/morsel.mojo`, and a grep in CI asserts it. The spec had planned a `shared.mojo` holding nothing but declarations, on the argument that one named file is the substitute for a race detector in a language without one. The argument still holds and the file does not: there is exactly one atomic, it is the morsel queue's cursor, and keeping a struct's own field in a different file from the struct hides the thing the rule exists to make obvious. The grep is new, so the rule is checked rather than merely written down.

- `take` and the join emit walk morsels instead of one slice per worker. Neither is faster today and both are measured, which is the point of writing it down here rather than in a commit message. A gather of eight million rows out of a twenty million row column is 12 ms either way, and a join emitting 39.5 million pairs is 272 ms either way, because both of them are moving more memory than they are doing arithmetic and a scheduler cannot make DRAM faster. What the change buys is that the loop body now takes a row range rather than a worker number, which is the shape a pipeline operator has, and the static split it replaces was machinery that would have had to come out anyway.

- `read_parquet` is about eleven percent faster on a fast machine and considerably more than that on a loaded one. Ten million rows of int64, float64 and an eleven character string, 176 MB of snappy Parquet in ten row groups, nine repetitions on an i9-13900K: medians 251 to 266 ms against the 274 to 302 ms the Arrow conversion path gets on the same file in the same run, and bests 232 to 239 against 262 to 268. Phase timing says why the win is not larger: of a read, pulling 4890 chunks out of DuckDB is 124 to 210 ms, describing every vector is 15 to 47 ms, and assembling is 33 to 109 ms. The conversion this change removes was 560 to 1100 ms on an eight core machine under load and much cheaper on sixteen idle ones, so the number here is the small end of the range rather than the large one.

Two things about Hive partitioning came out of testing it and are not what the DuckDB documentation led us to expect. The first is that `hive_partitioning` is detected rather than defaulted off, so a directory of `key=value` names is read as partitions whether or not anybody asked, and the setting that needs to be sayable is the one that turns it off. `ParquetOptions.hive_partitioning` therefore defaults to on and is written into every query either way, because leaving it out is not the same as writing false. The second is that partition columns come back after the file's own columns sorted by name, not in the order the path visits them, so `year=2024/month=03` produces `month` and then `year`. That is DuckDB's choice, it is stable, and the tests assert it so that a version that changes it is a test failure rather than a surprise.

A streaming result was tried and is not in this change, which is worth writing down because it looks like it should obviously win. `duckdb_query` builds the whole answer inside DuckDB and then hands it over a chunk at a time, so the rows are written once by the scan and read once by us, and `duckdb_execute_prepared_streaming` skips the first of those. Measured on the same file and machine, the streaming read is half again as slow, 400 to 445 ms against 251 to 266. A streaming result is produced by one thread pulling on the pipeline while a materialised one is produced by every thread DuckDB has, and the copy out is cheaper than the parallelism it costs.

## [0.6.27] - 2026-09-01

Built against Mojo 1.0.0 (ed45d567).

firepanda reads Parquet. `read_parquet` takes a path, or a path and a list of column names, and the path goes to DuckDB verbatim, so a glob, a directory of Hive partitions and an `s3://` URL are all just paths and none of them needed code here. Naming columns pushes the projection into the scan rather than reading everything and dropping, so a file of forty columns read for two costs two columns of decompression.

The library is DuckDB and not Arrow C++, which is what the spec said, and the spec has been changed to match. Arrow C++ exports no unmangled C symbols, so there is nothing for `dlsym` to find and nothing an `abi("C")` function type can describe. The dependency is soft: the library is dlopened on first use and a machine without it is a machine where `read_parquet` raises and nothing else in firepanda changes.

The honest number is that a read of ten million rows takes 292 ms against Polars' 32 ms, which is parity with the library we are calling and a long way from the 2x of Polars M2 asks for. The phase timing says the DuckDB scan is 83 ms of that and the per chunk Arrow conversion is most of the rest, and the conversion buys nothing because a DuckDB data chunk is already columnar. That is the next change.

Also in this release, the code that builds one frame out of many Arrow arrays moved out of the IPC reader, because a Parquet reader has row groups and a query result has result chunks and all three want the same thing done with them.

### Added

- `read_parquet` in `firepanda/io/parquet.mojo`, in two forms: a path, and a path with a list of column names. The path goes to DuckDB verbatim, so a glob, a directory of Hive partitions and an `s3://` URL are all just paths and none of them needed code here. Naming columns is a projection rather than a read followed by a drop, so a file of forty columns read for two costs two columns of decompression, and the order asked for is the order that comes back.
- `firepanda/io/duckdb.mojo`, a `dlopen` binding of the DuckDB C API. It is loaded on first use and it is soft: `FIREPANDA_DUCKDB` names a library if you have one somewhere unusual, otherwise the four ordinary names are tried in turn, and a machine with no libduckdb anywhere is a machine where `read_parquet` raises a message saying so and nothing else in firepanda changes.
- `benchmarks/read_parquet_file.mojo`, which times `read_parquet` on a real file, in the same shape as `read_arrow_file.mojo` beside it and for the same reason.
- `tests/test_parquet.mojo`, eleven tests against a Parquet file pyarrow wrote, checked in as the bytes pyarrow produced. Five columns and six rows, covering an int64, an int32, a double, a string and a bool, a null in the middle of four of them, an empty string next to one too long to inline, and one column with no nulls at all so the reader has to notice the difference.

### Changed

- The Parquet reader binds DuckDB, not Arrow C++ as `docs/specs/08-milestones.md` said. Arrow C++ exports no unmangled C symbols. `libparquet.so` is a C++ library with a C++ API, so there is nothing for `dlsym` to find and nothing an `abi("C")` function type can describe, and the pyarrow everybody actually uses is a Cython wrapper around that C++ rather than a C shim we could borrow. DuckDB ships a stable C header with an Arrow C Data Interface export built into it, which makes a Parquet read a `SELECT` whose result chunks hand out as Arrow arrays our importer already knows how to take. It also brings globbing, Hive partition discovery and projection pushdown for free. The spec now says so.

Two things about Mojo came out of writing the binding and are worth recording, because both of them look like bugs in DuckDB and neither one is.

A value dies at its last use, and that includes the expression it is being used in. `some_c_function(terminated(s).unsafe_ptr())` frees the buffer before the call runs, so C reads freed memory. This was found by calling `strlen` through the same machinery and getting zero back for an eleven character string. Every call site that hands C a pointer into a Mojo owned buffer now binds the buffer to a local and writes `_ = local^` after the call, and the same rule applies inside `__deinit__`, where a field's last use is still its last use and `self.lib.close(self.cells.at(0))` frees the cells before `close` reads them.

A stack local reached through `Pointer(to=x).unsafe_origin_cast[MutUntrackedOrigin]()` is not a usable C out parameter. Erasing the origin also erases the compiler's reason to believe anything ever wrote there, so C's write lands and the next read of the local still returns what it held before the call. This showed up as `duckdb_connect` returning success with a null connection, which reads exactly like a library bug, and the same program in C works five times in a row. Every out parameter now goes through a small `Cells` type that is a `List[UInt64]`, which is to say the heap, and a write through a pointer into the heap is a write.

Numbers, ten million rows of int64, float64 and an eleven character string, 176 MB snappy Parquet in ten row groups, seven repetitions, medians on an i9-13900K: firepanda 292 ms, DuckDB to an Arrow table 302 ms, pyarrow 69 ms, Polars 32 ms. So we are at parity with the library we are calling, which is what a thin binding should be, and we are nowhere near the 2x of Polars that M2 asks for.

The reason is in the phase timing. Of a read, the DuckDB scan is about 250 ms, pulling the chunks out is 145 ms, converting each 2048 row chunk to an Arrow array is 560 to 1100 ms, and assembling 4890 batches into a frame is 200 to 600 ms. The Arrow conversion is most of the cost and it buys nothing: DuckDB's data chunks are already columnar, a flat vector of int64 is a contiguous array of int64, and the next change is to copy out of the vectors directly and skip both the conversion and the assembly. That is a separate change and this one is the correct reader it will replace the middle of.

## [0.6.26] - 2026-09-01

Built against Mojo 1.0.0 (ed45d567).

This release finishes Arrow IPC. 0.6.25 could read a file, and now `write_arrow` and `write_ipc_stream` produce one, so a frame can go out to anything that speaks Arrow and come back. The read also got the change the last set of notes promised, which is that it no longer builds a column per record batch and stacks them at the end.

Two decisions in the writer are worth writing down. The first is that string columns go out as Utf8View and BinaryView rather than as the offset formats, which is the same coincidence the C Data Interface export leans on: a firepanda string column is Arrow's view layout byte for byte. Writing views is a copy of buffers we already have, and writing `u` would mean building an offset array and a compacted payload for every string column on the way out, for the benefit of a reader that does not know a layout that has been in the spec for years. Our own importer takes both, so nothing on this side depends on the choice.

The second is that the body streams straight out of the column's buffers. Writing a 320 MB frame allocates a few kilobytes of metadata and then hands the buffers to the file handle in order, so nothing between the frame and the disk holds a second copy of the data. The offsets each buffer will land at are computed once before anything is written and then checked again as the body goes out, because the entire format is offsets into a body whose padding rules are easy to get subtly wrong and hard to notice afterwards.

On ten million rows of int64, float64 and an eleven character string, both writers pointed at the same ext4 filesystem on an i9-13900K, firepanda writes one batch in a median of 213 ms and a best of 57 ms, against pyarrow's 298 and 276 for the same view layout and 405 and 355 for its default offset layout. The spread inside our own numbers is writeback rather than code: the runs that did not wait on the disk finished in a third of the time of the ones that did. Every file the writer produces was handed to pyarrow, which read back every value exactly, including the null, the empty string and the bools. That check lives outside the test suite because a Mojo test cannot call pyarrow, and a round trip through our own reader would happily agree with any misunderstanding the two of them share.

Then the read. The old path built a `DataFrame` per record batch and concatenated, which wrote every byte a second time and did it on one thread. The reader now decodes every batch's metadata before it copies anything, so the row count of the finished frame is known up front, and each column is allocated once and filled in place. The mistake worth recording is that the first version made a record batch the unit of work. That reads well, and it is what the file looks like, but how a file is chunked is a decision its writer made and says nothing about how many cores are sitting here. A file written as one batch had one task per column, so three columns used three of sixteen cores and came in at 71.6 ms while the 153 batch file beside it was already under 20. The unit is now a range of 65536 rows, which costs nothing to support because an Arrow array has carried an offset since the beginning, so a range of a batch is an ordinary array and the importer needed nothing new to read one.

Validity is the one part that still runs on one thread, because a range boundary lands in the middle of a byte and two threads writing that byte would each be writing bits the other one owns. Each range builds its own bitmap and they are pasted in afterwards, which is one bit per row rather than one value per row and does not show up in the numbers.

Fifteen repetitions on the same data, medians: 153 batches with offset strings went from 126 ms to 18.8 ms against pyarrow's 31.8, one batch from 89 ms to 19.3 against 28.2, and the view layout reads in 18.4 ms and 19.4 ms against 28.1 either way. So we went from 2.6x slower than pyarrow to about 1.5x faster, but the number to look at is that all four are now the same. Reading a file should cost what the bytes cost rather than what its writer picked for a chunk size, and until this release it did not.

### Added

- `write_arrow`, `write_ipc_stream`, `write_ipc_file_bytes` and `write_ipc_stream_bytes` in `firepanda/io/arrow_ipc_write.mojo`, which completes Arrow IPC in both directions. `write_arrow` takes a frame and a path and produces the random access file format, magic number at both ends and a footer listing every batch; `write_ipc_stream` produces the streaming format for anything that is going down a socket rather than onto a disk. Both take an `IpcWriteOptions` whose only field today is `rows_per_batch`, and the default of zero writes the whole frame as one record batch.
- String columns go out as Utf8View and BinaryView rather than as the offset formats. This is the same coincidence the C Data Interface export relies on: a firepanda string column is Arrow's view layout byte for byte, so writing views is a copy of the buffers we already have and writing `u` would mean building an offset array and a compacted payload for every string column on the way out. A reader that does not know views is a reader that does not know Arrow as of the current spec, and the importer on our own side accepts both.
- The body streams straight out of the column's buffers. Nothing between the frame and the file holds a second copy of it, so writing a 320 MB frame allocates a few kilobytes of metadata and then hands the buffers to the file handle in order. The offsets each buffer will land at are worked out once before anything is written and then checked again as the body goes out, because the whole layout is offsets into a body whose alignment rules are easy to get subtly wrong and hard to notice.
- `benchmarks/write_arrow_file.mojo`, which reads a real Arrow file once, untimed, and then times writing it back. Same reasoning as `read_arrow_file.mojo` beside it: what a write costs depends on what is in the frame, so the comparison has to be two writers given the same rows rather than two writers given two files of the same description.

Measured on an i9-13900K against pyarrow 25, ten million rows of int64, float64 and an eleven character string, both writers pointed at the same ext4 filesystem. Writing one batch, firepanda's median is 213 ms and its best 57 ms against pyarrow's 298 ms and 276 ms for the same view layout, and 405 ms and 355 ms for pyarrow's default offset layout. The spread inside firepanda's own numbers is writeback rather than code: the file is 320 MB and the runs that did not wait on the disk finished in a third of the time of the ones that did. Our file came out 320000778 bytes against pyarrow's 320000802 for the same data, and firepanda's reader reads the two at the same speed, 39.0 ms and 36.6 ms.

Every file this writer produces was also handed to pyarrow, which read back every value exactly, including the null, the empty string and the bools. That check is here rather than in a test because running pyarrow from a Mojo test is not something this repository can do, and a round trip through our own reader would agree with any misunderstanding the two happen to share.

### Changed

- `_pack_bools` in `firepanda/io/arrow_export.mojo` is now `pack_bools`. The IPC writer needs the same byte per value to bit per value pass the C export needs, and a second copy of it would be a second place to fix the same bug.
- The Arrow IPC reader fills each column in place instead of building a frame per record batch and concatenating them. It decodes every batch's metadata first, so it knows the row count of the finished frame before it copies anything, cuts the batches into ranges of 65536 rows, asks each range how much string payload it needs, prefix sums those, allocates each column exactly once, and then fills every range on its own core. Nothing is concatenated afterwards, which removes the second write of every byte that the concatenate was doing on one thread.
- The unit of work is a range of rows rather than a record batch. How a file is chunked is a decision its writer made and says nothing about how many cores are sitting here, and a file of one batch was the worst case for a per batch reader: it had one task per column and left thirteen of sixteen cores idle. An Arrow array has carried an offset since the beginning, so a range of a batch is an ordinary array and the importer needed nothing new to read one. The one part that does not fill in place is validity, because a range boundary lands in the middle of a byte and two threads would be writing bits neither of them owns, so each range builds its own bitmap and they are pasted in afterwards on one thread.
- `firepanda/io/arrow_import.mojo` gained `payload_for` and `fill_column` beside the existing `build_column`. The difference is which question is being answered. `build_column` answers what one array needs, which is what the C Data Interface asks, and the new pair answers where one array's rows go inside a column that something else has already sized, which is what a file asks. The scan `payload_for` does is a pass the copy was going to make anyway, so splitting it out costs nothing and makes the size known before the allocation happens. `build_column` is unchanged and still serves `import_array`.

Read numbers on the same machine and the same ten million rows, fifteen repetitions, medians. A file of 153 batches with offset strings went from 126 ms to 18.8 ms against pyarrow's 31.8 ms, and the same data in one batch went from 89 ms to 19.3 ms against pyarrow's 28.2 ms. The view layout reads in 19.4 ms as one batch and 18.4 ms as 153, against 28.1 ms both ways for pyarrow. The point of the change is the second number in each of those pairs rather than the first: how a file was chunked no longer changes what reading it costs, where before it was a difference of a third.

Every variant was read by a probe that sums the columns rather than only timing them, and all four agree with each other and with numpy on `id_sum 49999995000000`, `value_sum 4999791.013441796` and 110000000 bytes of string payload, with no nulls anywhere.

## [0.6.25] - 2026-09-01

Built against Mojo 1.0.0 (ed45d567).

This release reads Arrow IPC, which means `read_arrow("something.arrow")` now returns a frame, and it makes `concat` stop copying its parts twice on the way there.

The reader went in as two pieces because the first one is a wire format and the second one is a file format, and a mistake in a wire format surfaces a long way from where it was made. The first piece is FlatBuffers, which is how Arrow spells all of its metadata. Vendoring the C++ library or generating code with flatc were both available, and neither was worth it: the Arrow schemas use a small corner of the format, tables and scalars and strings and vectors, with no unions on the wire beyond a type byte and no shared strings or size prefixes. What we wanted instead was every read bounds checked against the length of the buffer, because an IPC file is data from somewhere else and a corrupt vtable offset is otherwise a straight read out of bounds. Scalars are read by memcpy rather than by a pointer cast, since a FlatBuffer embedded in an IPC message is only guaranteed 8 byte alignment.

The second piece is the reader, and the design decision in it is that it does not decode buffers at all. It points a synthesized `ArrowArray` at the record batch body and hands that to the importer that went in with the C Data Interface last release. That is why the C interface came first rather than IPC. Everything the importer knows about shifting a validity bitmap, turning offsets into views and checking a view against the buffer it names applies unchanged, and the alternative was a second implementation of all of it that would drift within a month. There is one genuine mismatch between the two: a view array over the C interface ends with a buffer of block lengths, and IPC keeps the same lengths in the record batch's own buffer entries, so the reader gathers them and appends the buffer the importer expects.

Every fixture byte in the tests came out of pyarrow 25 and is checked in as it came off the wire. A fixture this reader also wrote would agree with any misunderstanding the reader happens to have, and reading what somebody else wrote is the entire point of the format.

Then the real file. Ten million rows, three columns, 295 MB, written by pyarrow with its default chunking, which is 153 record batches. The read was 200 ms against pyarrow's 49, and the same data written as a single batch read in 102 ms. That gap is not the reader. It is that pyarrow keeps the batches as chunks and never joins them while we concatenate, and looking into that turned up something worse: the kernel's list spelling of `concat` took ownership of its parts, so the frame level deep copied every column into a list before handing it over, and that copy is exactly the size of the concat itself. So the multi batch read was copying every byte three times when one of them is genuinely necessary.

`concat_refs_any` takes references, which is what a caller whose columns live inside frames or inside a list of record batches actually has. While that was open, the fixed width path turned out to still be a single thread with a zeroed output buffer and a bit at a time validity loop that predates `Bitmap.paste`, where the string path beside it has run on every core since the CSV reader needed it. All three are fixed. At ten million rows on an i9-13900K, a concat of two parts went from 27.7 ms to 5.5 ms, of eight parts from 24.4 ms to 3.8 ms, and of a three column frame from 44.7 ms to 16.4 ms. The Arrow read went to 126 ms on the multi batch file and 89 ms on the single batch one.

That leaves us at 2.6x pyarrow rather than 4.1x, and the remaining gap is the copy that really is there: an Arrow buffer becomes a firepanda buffer, 64 byte aligned and padded, holding views rather than offsets. Closing it further means reading each batch straight into its place in the finished column instead of building a column per batch and stacking them, which is a change to the importer and is the next thing.

None of the concat problem was visible from the microbenchmarks. concat had benchmarks and they looked fine. It took pointing the reader at a real file that a real writer produced, with the chunking that writer chose rather than the chunking that suited us, to make it obvious.

### Added

- FlatBuffers reader and writer in `firepanda/io/flatbuf.mojo`. Arrow keeps its metadata in FlatBuffers, so this is a prerequisite for reading a single byte of an IPC file. Written rather than vendored or generated because the Arrow schemas use a small corner of the format and because every read wants to be bounds checked against the length of the buffer, which is not a taste question when the buffer came from somewhere else. Scalars are read by memcpy rather than by a pointer cast, since a FlatBuffer inside an IPC message is only guaranteed 8 byte alignment.
- `read_arrow`, `read_arrow_bytes`, `read_ipc_file` and `read_ipc_stream` in `firepanda/io/arrow_ipc.mojo`. `read_arrow` takes a path and works out from the magic number whether the file is the streaming format or the random access one, so a caller who was handed a `.arrow` file does not have to know which writer produced it. Multiple record batches concatenate into one frame. On a 295 MB three column file of ten million rows on an i9-13900K, written by pyarrow as 153 record batches, the read is 126 ms against pyarrow's own 49 ms, and the sums of both numeric columns and the total string bytes match numpy exactly.
- `benchmarks/read_arrow_file.mojo`, which times `read_arrow` on a file that already exists. Same reason `read_file.mojo` exists next to it: a fair comparison against another reader has to be two readers pointed at the same bytes, not two readers given two files of the same description.
- `build_column` is now public in `firepanda/io/arrow_import.mojo`. The IPC reader does not decode buffers itself. It points a synthesized `ArrowArray` at the message body and hands it to the importer that already exists and is already tested, which is the whole reason the C Data Interface went in first.

### Changed

- `concat` no longer copies its parts twice. The kernel's list spelling took ownership of the columns, so the frame level had to deep copy every part into a list before handing it over, and that copy is the same size as the concat itself. `concat_refs_any` takes references instead, which is what a caller whose columns live inside frames or inside a list of record batches actually has. The Arrow reader was paying this on every batch.
- Fixed width `concat` runs on every core past the same threshold the string path uses, allocates its output unzeroed since the parts write every row of it between them, and places validity with `Bitmap.paste` rather than a loop over bits. Measured at ten million rows on an i9-13900K: `concat/two_parts` 27.7 ms to 5.5 ms, `concat/eight_parts` 24.4 ms to 3.8 ms, and `concat/frame_three_columns` 44.7 ms to 16.4 ms. Reading the 153 batch Arrow file above went from 200 ms to 126 ms on the same change.
- `concat_arrays` now raises. Nothing in it can fail on its own; the signature says so because the copies run on every core and a worker's failure has to reach the caller.

### Fixed

- `import_array` no longer rejects an array whose release callback is null. The released check moved out of the structural validation and into `import_array`, which is the only place ownership actually means anything. An array synthesized over borrowed memory has no release callback and is not released in any sense that matters.

## [0.6.24] - 2026-09-01

Built against Mojo 1.0.0 (ed45d567).

This release is the Arrow C Data Interface, declared, exported and imported, plus the first two reductions that read a pair of columns.

The interface is what makes every later gap survivable, which is why it comes this early. Anything firepanda cannot read yet, something else can read and hand over, and anything firepanda cannot write yet can be handed to something that can. It landed in three pieces on purpose. The declarations went in on their own with the layout pinned by tests, because a mistake at that layer is invisible until it surfaces as a wrong pointer read inside somebody else's process. Then the export, then the import.

The export copies nothing for any type except bool. That was expected for fixed width columns and was not expected for strings. A firepanda string column turns out to be Arrow's view layout byte for byte, which is a coincidence rather than a plan, since the layout was chosen for short string inlining and prefix comparison. The tests read the actual bytes rather than trusting it, because the payload of a text column is usually the largest buffer in a frame and it is the one case where a copy would really cost. Bool is packed on the way out, a byte per value to a bit per value, and there is no way around that one.

The import copies, and that is structural rather than something left for later. Three separate reasons force it and any one would be enough. Our buffers are 64 byte aligned and padded to whole blocks so a kernel can read past the last value it cares about, and Arrow promises neither. An Arrow array carries a row offset, so what arrives is often a slice of somebody else's column. And a foreign view column may spread its long elements over any number of data buffers where ours has exactly one. The practical consequence is that the pointer identity property holds in the export direction only, which is worth knowing before someone goes looking for it in the other one.

The import also accepts the offset based string formats that the export side refuses, which is not an inconsistency. pyarrow emits `u` unless it is asked for views, so a reader that took only `vu` would be a reader that works with almost nothing.

Both string paths are two passes rather than a builder, because every length is known before the first byte is read and that is exactly what a builder cannot assume. A builder doubles its payload as it grows and then copies every view again on the way out. Measured over a million elements on an i9-13900K, the two pass reader spends 3.58 ns an element against the 14.36 ns per element that `strings/build_long` spends going through the builder. A producer with one data buffer and no row offset has handed over a column already shaped like ours, and that case reduces to two memcpys at 3.50 ns an element against 7.07 for the same column split over two data buffers. Fixed width imports at 0.23 ns a row, where two thirds of the time is the allocation rather than the copy.

Every view is checked against the length of the buffer it names before anything is read, including on the fast path. A view is three numbers a stranger chose, and following one unchecked is an out of bounds read waiting for a malformed file.

`CORR` and `COV` are the other half of the release. They are the first aggregations that read two columns instead of one, so `AggSpec` grew a second column name, and a correlation is now a spec like any other: asking for one alongside three sums over the same keys groups the rows once. Both centre their inputs on pairwise means, so a row where either value is null contributes to neither mean, because a covariance is a statement about rows in which both were observed.

### Added

- `firepanda/io/arrow_c.mojo`, the Arrow C Data Interface declared: `ArrowSchema`, `ArrowArray`, the flag constants, the format string in both directions, and the release protocol. No producer and no consumer yet, which is deliberate. This is the layer where a mistake is invisible until it is a wrong pointer read inside pyarrow, so it lands on its own with the layout pinned by tests before anything is built on it. The tests write a distinct value into every field and read the struct back as an array of eight byte words, which catches two fields swapping places, and they call an actual `abi("C")` function pointer through the release field, which is the mechanism the whole interface rests on and was the part in most doubt. `ArrowSchema` is seventy two bytes and `ArrowArray` is eighty, as the specification says.
- Three facts about Mojo 1.0 came out of building it and are written down in that file rather than rediscovered later. A C function pointer usable as a struct field is `def (args) thin abi("C") -> None`, where `thin` is what makes it a bare pointer instead of a trait a struct cannot hold. `Pointer` is not nullable, so every C field that may be null is `Optional[Pointer[...]]`, which is still eight bytes because a null pointer is the niche the discriminant packs into. That does not extend to function pointers, where `Optional` is sixteen bytes and would move every field after it, so the release fields are nullable void pointers with the reinterpretation done at the two ends.
- `firepanda/io/arrow_export.mojo`, the export half: `export_schema` and `export_array` hand a firepanda column to a C consumer without copying a value. `buffers[1]` is the column's own values pointer and `buffers[0]` is its own validity pointer, and the tests assert that by comparing addresses rather than contents, because contents compare equal for a copy too. Two coincidences make it possible and both are now asserted rather than assumed: firepanda's validity bitmap is one bit per row, least significant bit first, with one meaning present, which is exactly Arrow's, and a fixed width values buffer is the Arrow values buffer with no header on either side.
- Ownership is a heap box behind `private_data`, holding the column and the buffer pointer array, allocated with `malloc` and destroyed by the release callback. C's allocator rather than Mojo's, because the free happens inside a callback a foreign runtime invokes on a thread firepanda knows nothing about. `export_array` consumes its column, which is not a convenience choice: firepanda columns are deep copied rather than refcounted, so sharing one between a `DataFrame` and a consumer is not something this layer can honestly offer. Verified leak free under valgrind, zero bytes lost in any category and zero errors.
- Strings and binary export without copying either, which was not the expectation going in. A firepanda string column is already Arrow's view layout byte for byte: sixteen bytes per element, a little endian uint32 length, then the data inline when it fits in twelve bytes or a four byte prefix plus a uint32 buffer index and a uint32 offset when it does not. firepanda picked that layout for short string inlining and prefix comparison and landing on Arrow's was not one of the reasons, so a test reads the actual bytes of a short view and a long one rather than trusting the coincidence to keep holding. This is the case that would have hurt most to copy, because the payload of a text column is usually the largest buffer in a frame. Such a column exports as four buffers, validity, views, one payload block and the sizes, and the block is emitted even when every string inlines so that the count is a constant a consumer can rely on.
- Bool is the one type that is copied, and it is a packing pass rather than a copy in the usual sense. firepanda stores a bool as a byte because that is what a kernel wants to load and Arrow stores it as a bit, so the exported values buffer is a freshly packed bitmap that lives in the ownership box like everything else. It comes out eight times smaller than the column it was built from. A test asserts the exported pointer is not the column's byte buffer, which is the failure that would otherwise pass every test that only reads element zero.
- The null type is the only thing the exporter refuses, and it raises with the reason: firepanda has it as a `LogicalType` but has no column that carries it at run time, so there is nothing to hand over.
- `type_for_format` refuses `u`, `U`, `z` and `Z`, the offset based string formats, rather than quietly reading them as views. They are different memory from the view layout firepanda uses, and accepting one as the other is exactly the failure this file exists to prevent. Converting them is real work and belongs in the import path where it can allocate.
- `firepanda/io/arrow_import.mojo`, the other direction: `import_array` takes a schema and an array from a C producer and returns a firepanda column. It copies, and unlike the export that is not a temporary state of affairs. A firepanda buffer is 64-byte aligned and allocated in whole 64-byte blocks so that a kernel can read a full register past the last value it cares about, and Arrow requires eight bytes of alignment and says nothing about what follows the last byte. An Arrow array carries a row offset, so what arrives is often a slice of somebody else's column, and firepanda columns start at element zero. And a foreign view column may spread its long elements over any number of data buffers where firepanda has exactly one, so the views are rewritten regardless. Any one of the three would be enough on its own.
- The import accepts `u`, `U`, `z` and `Z` as well as the view formats, which is the promise the export side made when it refused them. That matters more than the symmetry does, because `u` is what pyarrow produces unless it is asked for views, so an import that took only `vu` would be an import that works with almost nothing. An element short enough to inline never reaches the payload at all, so a column of country codes arrives as a data buffer plus an offset per element and leaves as views alone.
- Both string paths are two passes rather than a `StringBuilder`. The lengths are all known before the first byte is read, which is exactly what a builder cannot assume, so the payload is sized once and each view is written straight into place. A builder doubles its payload as it goes and then copies every view a second time on the way out of `finish`, which on a ten million row column is a second pass over a hundred and sixty megabytes for nothing. Measured on an i9-13900K over a million elements of fourteen bytes: 3.58 ns an element for the offset layout against the 14.36 ns `strings/build_long` spends per element going through the builder.
- A producer with one data buffer and no row offset has handed over a column already shaped like ours, and that case is two memcpys: 3.50 ns an element against 7.07 for the same column split over two data buffers. The check pass still runs, because it is also where every view is tested against the length of the buffer it names. A view is three numbers a stranger chose and following one unchecked is an out of bounds read waiting for a malformed file. The offset layout gets the one check it allows, which is that the offsets go forwards, since a backwards pair is a negative element length and a negative length reaching a memcpy is an enormous one.
- Fixed width imports at 0.23 ns a row and bool at 0.36, the second being a bit at a time unpack into firepanda's byte per value. Two thirds of the fixed width number is the allocation rather than the copy: `buffer/alloc_fresh` alone is 156 microseconds for the eight megabytes that `arrow/import_int64` moves in 238.
- `import_array` releases both structures before it returns, including when it raises. A producer that has handed a structure over has no way to reclaim it, so a consumer that refuses the type still owes it the release call, and a refusal that leaked would make every unsupported column a slow leak in a loop over a directory of files. Since everything is copied the release happens immediately rather than being deferred, and nothing that comes out points into the producer's memory. Leak checked under valgrind alongside the export.
- Five benchmark rows under `arrow/`, covering the fixed width copy, the bool unpack, a view column in the shape our own exporter produces, the same column split over two data buffers, and the offset layout. The producers are built by hand rather than by the exporter, partly because the exporter consumes its column and a benchmark needs to hand over the same bytes every iteration, and partly because three of those shapes are ones firepanda can never produce.
- `AggKind.CORR` and `AggKind.COV`, the first two reductions that read a pair of columns rather than one. `AggSpec` gains a second column name and a constructor that takes it, so a correlation is a spec like any other and composes with the rest in one call: asking for a correlation and three sums over the same keys groups the rows once. The kernel centres both columns on their pairwise means before accumulating, the way `group_var` does and for the same reason, and pairwise means it: a row where either value is null contributes to neither mean, because a covariance is a statement about rows in which both were observed. A group with fewer than two such rows is null, and so is a correlation whose denominator is a zero. The erased entry point casts both columns to float64 and calls one instantiation rather than dispatching on both dtypes, which would be a hundred and forty four instantiations of a loop that reads its inputs as float64 in either case; the typed spellings stay generic and copy nothing.

## [0.6.23] - 2026-08-31

Built against Mojo 1.0.0 (ed45d567).

Two changes, one to the group by and one to the join, and both of them are the same observation from different ends: the library was choosing a general shape in places where it already knew the specific one.

The group by on several keys packs its keys into one integer per row and factorizes that at the end. Factorize decides between a direct table indexed by value and a hash table by scanning the column for its range, and it declines the table above sixty five thousand, because a scan is a measurement of data the library did not construct and a span of ten million says nothing about whether ten values or ten million occupy it. A packed key is not in that position. Its range is `g0 * g1 * g2 * ...`, computed on the way down out of ordinals that are dense by construction, and a range built that way out of dense parts is itself densely occupied. So there is now a `factorize_dense` for callers who can name the range, and its rule is the table against the column rather than the table against the cache: index it when the span fits in what the column already costs, which caps the direct route at four bytes a row. Two keys of a hundred by a hundred thousand is a span of exactly ten million on ten million rows, and it now indexes instead of hashing.

The join builds its right side by bucketing rows by code: count per code, prefix sum, scatter, undo the cursor. That is the general answer and it stays, because a right key can repeat and then a left row pairs with several. But a join onto a primary key has one right row per code, and there the counts, the prefix sum, the cursor walk and the bucket array all exist to say "one". The build now assumes the right key is unique and fills a single table from code to row in one pass, and the first code it finds already taken abandons that and runs the general build from the top. Being wrong costs part of one scan of the right side. Being right saves two walks of a table as long as the frame plus an array as long as it again, and the table is `int32` rather than `Int`, so the widest join in the suite carries forty megabytes where the general shape carried a hundred and sixty.

Ten million rows on an i9-13900K, v0.6.22 against this release, alternating builds, one process per measurement, median of the per round paired ratios.

| query | shape | 0.6.22 | 0.6.23 | ratio |
| --- | --- | --- | --- | --- |
| j5 | ten million to ten million, then aggregated | 777.5 ms | 625.8 ms | 1.28 |
| j4 | ten million to ten million | 753.3 ms | 618.5 ms | 1.21 |
| q6 | group by two keys, six million out | 627.8 ms | 522.0 ms | 1.19 |
| q2 | group by two keys, ten thousand out | 75.2 ms | 72.6 ms | 1.06 |
| j2 | ten million to a hundred thousand | 204.8 ms | 199.3 ms | 1.02 |
| q3 | group by one key, hundred thousand out | 138.3 ms | 135.6 ms | 1.02 |
| j1 | ten million to ten thousand | 155.8 ms | 152.7 ms | 1.01 |
| j3 | left join, ten million to a hundred thousand | 205.0 ms | 206.9 ms | 1.00 |
| q10 | group by six keys, ten million out | 1009.0 ms | 1015.7 ms | 0.99 |

The two halves land on disjoint queries, which is what their code predicts. q6 is the query the dense table was written for and j4 and j5 are the joins the unique table was written for, and neither change touches the other's shape. q10 is a control for the first and a control it stays: a space of ten to the eighteen declines the table on either route. j3 is a control for the second in the same sense, since its build side is a hundred thousand rows and there was never much there to save.

Two things learned in the measuring are worth carrying forward. The unique join route was first written as an `if` inside the two hot loops rather than as its own pair of loops, and in that form j4 and j5 gained about what they gain now while j3 read 0.898 and j1 read 0.994. The predicate is loop invariant and answers the same way on all ten million rows, and it is still a compare and a jump on each of them, so on the joins with a small build side it was pure cost. Splitting the loops put j3 back to 1.000 with nothing lost at the other end. And q10 first read 0.927 on eight rounds, which looked like a regression on the widest query and is not one: it is bimodal on this machine with modes near 980 and 1090 ms that both builds visit, and twenty two rounds put it at 0.974 with the two ranges overlapping across their whole width. A reading that contradicts the mechanism needs more samples before it gets a hypothesis.

### Added

- `factorize_dense`, for a caller that already knows the range its values are in. `factorize` learns the range by scanning, and what a scan finds is a bound on data the library did not construct, so it declines a direct table above `DIRECT_LIMIT` because a span of ten million says nothing about whether ten values or ten million occupy it. A group by on several keys is not in that position. Its packed key has a range of `g0 * g1 * g2 * ...`, computed on the way down out of ordinals that are dense by construction, and the shapes where that product is large are the shapes where most of it is occupied. So the bound for a caller who can name the range is the table against the column rather than the table against the cache, which caps the direct route at four bytes a row and never more. Two keys of a hundred by a hundred thousand is a span of exactly ten million on ten million rows, which now indexes a table instead of hashing.

### Changed

- The join build side takes one table from code to row when the right key is unique. Bucketing the right rows by code is the general answer, and it has to exist, because a key can repeat on the right and then a left row pairs with several. But a join onto a primary key has one right row per code, and there the counts, the prefix sum, the cursor walk and the bucket array are all machinery for saying "one". So the build assumes uniqueness and fills a single table in one pass, and the first code it finds already taken abandons that and runs the general build from the top. Being wrong costs part of one scan of the right side. Being right saves two walks of a table as long as the frame plus an array as long as it again, and the table is `int32` rather than `Int`, so the widest join in the suite carries forty megabytes where the general shape carried a hundred and sixty. The count and emit loops are written twice rather than branching on which build ran, because the branch answers the same way on all ten million rows and is still a compare and a jump on each of them: with the branch inside the loops j3 read 0.898, and with the loops split nothing goes backwards. Ten million rows on an i9-13900K, alternating builds, one process per measurement, eight rounds, median of the per round paired ratios: j5 777.5 ms to 625.8 ms for 1.278, j4 753.3 ms to 618.5 ms for 1.207, j2 1.018, j1 1.013, j3 1.000. j4 and j5 join ten million to ten million on a key that is one through n, which is the shape the table was written for. j1 through j3 join to a dimension of ten thousand or a hundred thousand, so their build side was small either way and the whole of the change there is the emit reading one `int32` instead of two `Int`s. The group by controls q6 and q10 sit at 1.003 and 0.989.
- The multi key group by asks for `factorize_dense` at the end and inside `_condense`, passing the space it already computed. Ten million rows on an i9-13900K, alternating builds, one process per measurement, median of the per round paired ratios: q6 627.8 ms to 522.0 ms for a ratio of 1.19 over eight rounds, and q2 75.2 ms to 72.6 ms for 1.06 over twenty two. q6 is the query the table route is for. q2 is a smaller and different win, since its space of ten thousand was under `DIRECT_LIMIT` and took the direct route already, and what it saves is the scan that used to walk ten million int64 values to learn a range the group by had computed. q10 is unchanged at 0.974 over twenty two rounds with the two distributions overlapping across their whole width, which is what its code predicts: a space of ten to the eighteen declines the table on either route, and the scan that no longer runs was stopping on its first few rows anyway. q3, q1 and j4 sit at 1.02, 1.01 and 0.99.

## [0.6.22] - 2026-08-31

Built against Mojo 1.0.0 (ed45d567).

The group by on several keys. Two changes, both of them work that was being done and then discarded, and on the widest shape in db-benchmark they take a third off the query.

Grouping on more than one key works by packing the keys into a single integer per row. Group on the first key and every row has an ordinal in `[0, g0)`; group on the second and it has another in `[0, g1)`; then `c0 * g1 + c1` names the pair exactly. Fold in a third key by multiplying by `g2` and adding, and so on down the list. Factorize the packed value at the end and the group by is done.

That packed value was being factorized after every single key it folded in, not just at the end. The reason was overflow: without the intermediate passes the running space is the product of all the group counts rather than the number of tuples actually present, and the concern was that the product would leave an int64. It does not, for anything anyone has. Nineteen digits is twenty six columns of a hundred distinct values, or six columns of a thousand. So the packing now happens in place and the factorize happens once, and `_condense` renumbers the running key into the tuples it actually holds if a column list ever does approach the bound. The bound is still checked before every multiply. A six key group by at ten million rows was eleven hashed passes over the full column, six for the keys and five to redensify after each fold, and it is now seven.

Separately, every hashed factorize was building a column of the distinct key values and returning it, and nothing in the library ever read it. A group by does not need it, because it gathers all of its key columns at once at the end by the rows that introduced each group. A join does not need it either, because it reads the ordinals and nothing else. So `Factorized` now carries those representative rows, which every route already had, and building the key values is something a caller asks for. On the six key query the old code produced five of those columns at eighty megabytes each and dropped all five.

Ten million rows on an i9-13900K, v0.6.21 against this release, alternating builds, six rounds of five runs each, one process per measurement, median of the per round paired ratios.

| query | shape | 0.6.21 | 0.6.22 | ratio |
| --- | --- | --- | --- | --- |
| q10 | group by six keys, ten million out | 1747.8 ms | 1132.0 ms | 1.56 |
| q6 | group by two keys, six million out | 811.3 ms | 663.4 ms | 1.21 |
| q2 | group by two keys, ten thousand out | 83.2 ms | 70.8 ms | 1.18 |
| j5 | ten million to ten million, then aggregated | 807.7 ms | 747.5 ms | 1.07 |
| j1 | ten million to ten thousand | 156.8 ms | 150.9 ms | 1.03 |
| j4 | ten million to ten million | 793.9 ms | 770.5 ms | 1.03 |
| q1 | group by one key, hundred out | 20.3 ms | 19.8 ms | 1.02 |
| q7 | group by one key, then a subtraction | 126.6 ms | 126.0 ms | 1.01 |
| j2 | ten million to hundred thousand | 206.5 ms | 204.8 ms | 1.01 |
| q5 | group by one key, hundred thousand out | 78.4 ms | 82.5 ms | 0.99 |
| q3 | group by one key, hundred thousand out | 135.2 ms | 138.3 ms | 0.98 |

The last two are inside the noise band this setup shows on changes known to do nothing, which is about three percent. q10 is not: its two distributions do not overlap, 1705 to 1784 ms before and 1022 to 1152 ms after.

Where the credit goes is worth being precise about, because the two changes help different shapes. The packing change is the whole of q10 and none of q6, since two keys never had a redundant fold to remove. The discarded key column is the whole of q6 and q2, since both produce enough groups for that column to be large. One key queries were never paying for either.

A note for anyone measuring this themselves: q6 on this machine is bimodal, with a fast mode near 600 ms and a slow one near 770 ms that every build visits, so its median moves by twenty percent depending on which mode the samples land in. Paired ratios are stable there and raw medians are not.

### Changed

- `Factorized` no longer carries a `keys` column. Read the key values with `Factorized.keys(col)`, passing the column that was factorized, which gathers them from the representative rows. Callers who were reading `.keys` as a field want `.keys(col)` as a call; callers who were ignoring it, which was everyone inside the library, want nothing.

## [0.6.21] - 2026-08-31

Built against Mojo 1.0.0 (ed45d567).

The join. Three changes, each taking a different pass off it, and together they roughly double it.

The pairing was doing all of its work on one thread. Deciding which left row goes with which right rows is a walk over the left side that carries no state between rows: what a row emits depends on that row and on the right side buckets, which are finished before the walk starts. It is now cut into slices, one per core, with a prefix sum of the per slice output counts in between telling each worker where its slice begins in the result. That prefix sum is why the counting walk, which already existed so the two index lists could be allocated once at the right size, had to be cut the same way. An outer join is left on one thread deliberately, because it has to record which right rows it paired so it can emit the rest afterwards, and that record is a bitmap whose set is a read modify write of a word eight rows share.

`take_rows` got the same treatment for the same reason. It is what actually builds the output columns of a join, once per column, and a gathered row depends on its own index and nothing else in the output. The only part of that loop which is not per row is the validity bitmap, which is built in a register and stored once every sixty four rows, so the slice boundaries round up to a multiple of sixty four and no two workers write the same word.

Then two passes that did not need to happen. The pairing built a per row flag over both frames saying whether that row's key tuple contained a null, one branch per key per row, and answered no every time on a frame that has no nulls; asking each key column for its null count first is a popcount per validity word. And the gather read the source's validity bit for every row, a second random read into a different array from the values, which a column with no nulls does not need either. Between them these are the largest single wins here relative to the code they replace, and neither is clever.

Last, the bucket build stopped allocating a second copy of the group table. The scatter needs a cursor per group, and on a join between two frames that are mostly one to one that table is as long as the frames. The offsets can be their own cursor: group `g` is written from `starts[g]` to `starts[g + 1]`, so afterwards each entry holds what its successor held, and one backwards pass puts them back.

Ten million rows on an i9-13900K, v0.6.20 against this release, alternating builds, six rounds of five runs each, one process per measurement, median of the per round paired ratios.

| query | shape | 0.6.20 | 0.6.21 | ratio |
| --- | --- | --- | --- | --- |
| j5 | 10M to 10M, then aggregated | 1745.6 ms | 758.5 ms | 2.30 |
| j4 | 10M to 10M, one match a row | 1818.4 ms | 746.8 ms | 2.18 |
| j1 | 10M to 10k, one match a row | 304.0 ms | 155.2 ms | 1.99 |
| j2 | 10M to 100k, inner join | 341.6 ms | 202.5 ms | 1.75 |
| j3 | 10M to 100k, left join | 343.6 ms | 209.2 ms | 1.56 |
| q10 | six key group by, no join | 1752.7 ms | 1728.1 ms | 1.01 |
| q3 | string key group by, no join | 136.8 ms | 139.7 ms | 1.00 |
| q1 | one key group by, no join | 19.8 ms | 20.7 ms | 0.95 |

The three group by queries are controls and none of them touches a join.

What is left in the j4 pairing is the factorize of the concatenated twenty million row key column, which is around 390 ms of the 747, and the scatter itself, which is ten million random writes into a table that does not fit in cache. The first of those is the more interesting one, because it is the same code path the group by queries spend their time in. DuckDB does these five queries in 15 to 95 ms, so this is progress rather than arrival.

### Changed

`take` gathers on every core and stops probing a validity bitmap the column has none of.

A gather's output row depends on its own index and on nothing else in the output, so it splits by output row. The one thing in there that is not per row is the validity bitmap, which is built a word at a time in a register and stored once every sixty four rows, so the slice boundaries are rounded up to a multiple of sixty four and no two workers touch the same word.

The other half is the probe. The loop read the source's validity bit for every row, which is a second random read into a different array from the values, and doubles the number of cache misses a gather takes. A column with no nulls does not need it. The negative index check has to stay, because that is how a left join reports a row the right side did not have, but the two halves of that condition are separate questions and only one of them was avoidable.

Ten million rows on an i9-13900K, eight paired rounds of five runs each. These are joins because that is where the big gathers are, one per output column.

| query | before | after | ratio |
| --- | --- | --- | --- |
| j1 | 235.7 ms | 181.4 ms | 1.34 |
| j2 | 282.2 ms | 223.7 ms | 1.28 |
| j3 | 266.6 ms | 226.8 ms | 1.21 |
| j5 | 961.4 ms | 806.1 ms | 1.18 |
| j4 | 958.7 ms | 822.2 ms | 1.18 |
| q1 | 22.6 ms | 21.7 ms | 1.00 |

`filter_rows` is deliberately not split the same way. Where a filtered row lands depends on how many rows before it survived, which is a prefix sum the gather does not need.

The join's bucket build stops allocating a second copy of the group table.

Bucketing the right side is a count, a prefix sum and a scatter, and the scatter needs a cursor per group saying where the next row of that group goes. It was a separate array, filled from the offsets. On a join between two frames that are mostly one to one the group table is as long as the frames, so that is an allocation and a copy the size of the input, plus the zero fill of the bucket array on top, which the scatter overwrites in full anyway.

The offsets can be their own cursor. Group `g` is written from `starts[g]` up to `starts[g + 1]`, so when the scatter finishes each entry holds what its successor held, and one backwards pass over the table puts them back. A group nothing was scattered into needs no special case, because its offset was already equal to its successor's.

Ten million rows on an i9-13900K, eight paired rounds of five runs each, on a quiet machine this time.

| query | groups | before | after | ratio |
| --- | --- | --- | --- | --- |
| j5 | 10M | 1094.7 ms | 909.9 ms | 1.17 |
| j4 | 10M | 1035.6 ms | 918.8 ms | 1.14 |
| j2 | 100k | 249.3 ms | 247.5 ms | 1.04 |
| j1 | 10k | 206.1 ms | 206.2 ms | 1.01 |
| j3 | 100k | 260.1 ms | 261.5 ms | 0.98 |
| q1 | control | 20.1 ms | 20.2 ms | 0.98 |

j1 through j3 join against a small right frame, so their group tables are ten and a hundred thousand entries and there was nothing there to save. The gain is the whole point of the change: it scales with the number of distinct keys, not with the number of rows.

The join emits its row pairs on every core, and stops looking for null keys in frames that have none.

`join_indices` was two serial walks over the left side, one counting the output rows and one writing them, and on db-benchmark's j queries the writing walk alone was half of the pairing. It is a walk with no carried state: what a left row emits depends on that row and on the right side buckets, which are finished before the walk starts. So both walks now split by left row across cores, with a prefix sum of the per slice counts in between telling each worker where in the output its slice begins. That is what replaces the append, and it is why the counting walk had to split the same way.

An outer join is left on one thread on purpose. It has to remember which right rows it paired so it can emit the leftovers afterwards, and that memory is a validity bitmap whose set is a read modify write of a word that eight rows share, so two workers marking at once would drop marks and invent unmatched rows.

The other half is a pass that did not need to run at all. Before touching any of the above, the pairing built a `List[Bool]` over both frames saying whether each row's key tuple contained a null, one branch per key per row, and on a frame with no nulls anywhere the answer was False every time. Asking each key column for its null count first is a popcount per validity word, a sixty fourth of the pass, and db-benchmark's keys have no nulls. That was 14 ms of the 97 ms pairing on j1 and 27 ms on j4.

Ten million rows on an i9-13900K, alternating builds, eight rounds of five runs each, one process per measurement, reported as the median of the per round paired ratios with the full range of those ratios in the last two columns. The machine was under an unrelated load again, which is what the pairing is for.

| query | shape | before | after | ratio | low | high |
| --- | --- | --- | --- | --- | --- | --- |
| j4 | 10M to 10M, one match a row | 2280.7 ms | 1335.8 ms | 1.72 | 1.26 | 1.90 |
| j5 | 10M to 10M, then aggregated | 2232.3 ms | 1342.6 ms | 1.67 | 1.37 | 2.07 |
| j1 | 10M to 10k, one match a row | 386.5 ms | 255.3 ms | 1.49 | 1.13 | 1.71 |
| j3 | 10M to 100k, left join | 417.2 ms | 321.8 ms | 1.32 | 0.97 | 1.66 |
| j2 | 10M to 100k, inner join | 413.3 ms | 318.0 ms | 1.30 | 1.01 | 1.46 |
| q1 | group by, no join | 26.7 ms | 25.8 ms | 1.03 | 0.79 | 1.37 |

q1 is the control and touches none of this. The before column is higher across the board than the numbers quoted in the 0.6.20 notes because the machine was busier during this run, which is exactly the reason the ratios are paired rather than divided at the end.

What is left in the j4 pairing is the scatter that fills the buckets, which is 354 ms of random writes into a 10M entry table, and the factorize of the 20M row concatenated key column, which is 390 ms. Neither is touched here.

## [0.6.20] - 2026-08-31

Built against Mojo 1.0.0 (ed45d567).

The numeric factorize routes hand back a representative row per group, which is the last thing that was keeping the renumbering pass on the critical path.

The change below took the pass off a string key with no nulls, because the string merge already produced that list on its way to comparing candidate keys. The numeric routes had the same row for the same reason and were throwing it away. All three of them recognize a new group by the row that introduced it: the direct one appends to `keys` on first sight, the hashed serial one already gets the list back from `HashTable.build` and reads the key values out of it, and the parallel merge picks its representative when a worker's local group turns out to be new. So `Factorized` grows a `firsts` and each route fills it where it already had the row in hand, which is one append per group rather than a pass.

That covers the first key. The other place the pass was running unconditionally is the combine step, once per key after the first. What it renumbers there is the packed column, which is written a row at a time from two code arrays and therefore has no nulls at all, so its factorize's ordinals are already in first-appearance order and its representative rows are already the ones the pass would have collected. That is five passes over every row on a six key group by.

What is left calling `_densify` is a first key with nulls, of any dtype. All four routes put the null group at ordinal zero wherever its first null actually is, and `sort=False` at the frame layer means first appearance, so something has to move it.

Ten million rows on an i9-13900K, alternating builds, twenty rounds of five runs each, one process per measurement. The machine was running an unrelated benchmark throughout, which is why this is reported as the median of the per round paired ratios rather than as a ratio of medians: the pairing is what makes the contention cancel. The full range of those per round ratios is in the last two columns.

| query | keys | before | after | ratio | low | high |
| --- | --- | --- | --- | --- | --- | --- |
| q4 | id4, integer | 51.6 ms | 41.9 ms | 1.22 | 1.14 | 1.38 |
| q5 | id6, integer | 94.7 ms | 80.1 ms | 1.17 | 0.97 | 1.31 |
| q6 | id4 and id5, integer | 898.3 ms | 813.4 ms | 1.11 | 1.08 | 1.15 |
| q10 | id1 through id6, mixed | 1904.0 ms | 1736.2 ms | 1.11 | 1.02 | 1.19 |
| q3 | id3, string | 130.1 ms | 133.5 ms | 0.98 | 0.91 | 1.07 |
| j1 | id1, join | 329.2 ms | 308.0 ms | 1.05 | 0.95 | 1.17 |

q4 and q5 are single integer keys and they get the first key pass removed. q6 gets that plus one combine, and q10 gets five combines on top of a string first key that was already free. q3 is the control: it went through the change below and there was nothing left here for it to gain, and it did not. j1 is the other control and its spread covers one, so the five percent is not a claim.

A group by no longer renumbers ordinals it has no reason to renumber.

`group_ordinals` ran `_densify` over every row of every key. That pass was written because `factorize` did not promise that every ordinal it can produce belongs to some row, and an ordinal nothing carries becomes an aggregation row nobody asked for. It does promise that now, on all three routes, and a fuzz over three thousand random int64 columns covering both numeric routes found no sparse ordinal on either, so the density is not what the pass is still buying. Two other things are. It puts the null group where its first null appears instead of at ordinal zero, which is what `sort=False` means at the frame layer, and it fills the representative row table the frame layer gathers key values with.

Neither is needed as often as the pass was being run. A string key with no nulls already has both: the merge hands out ordinals in first-appearance order, and the row list it built to compare candidate keys with is exactly the table `_densify` would have produced. A key that is not the first one needs neither either, whatever its dtype and whatever its nulls, because only its group count is read at that point, and the packed column is factorized again afterwards, which is what fixes the ordinals for the result.

So `_factorize_any` now reports what its route knows in a `KeyCodes`, and `group_ordinals` decides from that rather than paying unconditionally. What is left paying is a numeric first key and a string first key with nulls. The numeric routes could record representative rows during the build and skip the pass too, which is the obvious next step and is a separate change.

Ten million rows on an i9-13900K, alternating builds, five runs each, three rounds, medians of the per round medians, one process per measurement. The queries are db-benchmark's at the 0.5GB scale.

| query | key | before | after | ratio |
| --- | --- | --- | --- | --- |
| q1 | id1, string, 100 groups | 28.7 ms | 19.9 ms | 0.69 |
| q2 | id1 and id2, string | 115.2 ms | 90.0 ms | 0.78 |
| q3 | id3, string, 100k groups | 139.8 ms | 116.3 ms | 0.83 |
| q7 | id3, string, 100k groups | 130.7 ms | 110.4 ms | 0.84 |
| q10 | id1 through id6, mixed | 1968.1 ms | 1878.9 ms | 0.95 |
| q4 | id4, integer | 52.0 ms | 50.0 ms | 0.96 |
| q5 | id6, integer | 91.9 ms | 89.3 ms | 0.97 |
| q6 | id4 and id5, integer | 853.9 ms | 835.5 ms | 0.98 |
| j1 | id1, join | 326.3 ms | 319.7 ms | 0.98 |
| j4 | id1 through id3, join | 1775.0 ms | 1741.9 ms | 0.98 |

The first five are the string keyed queries, which is the set the change targets. The last five are controls and none of them moved beyond the run to run spread, which is what a change that only removes work should look like.

### Changed

- `_factorize_any` returns a `KeyCodes` carrying the ordinals, the group count and the representative rows, instead of the ordinals alone, and `group_ordinals` skips `_densify` for any key whose `KeyCodes` describes every group.
- `group_ordinals` skips it for the packed column too, which is one pass per key after the first.
- `Factorized` carries a `firsts`, so its constructor and `_finish` take a fourth argument. Both are internal to `firepanda/hash`, and `factorize` itself is unchanged at the call site.
- `_densify`'s docstring records that the sparse ordinal case it was written for no longer happens, and what it is still for.

### Added

- `FactorizedStrings.into_parts`, `Factorized.into_parts` and `KeyCodes.into_parts`, which give up the ordinals and the representative rows together, because a struct cannot have two of its fields moved out one at a time and unpacking a returned tuple copies.
- Tests that a numeric key without nulls keeps the factorize's ordinals on both the hashed and the direct route, that the packed column keeps them too, and that a null in the first of two keys still lands where its first null is.
- `test_no_shape_of_column_factorizes_to_a_sparse_ordinal`, a swept property test over a hundred and twenty random int64 columns with varying length, value span and null count, which holds the promise the skipping now depends on across both numeric routes.
- Tests that a numeric key is still renumbered, that a null key still lands where its first null is, that a text key is grouped without renumbering its ordinals, and that a text key with nulls is renumbered after all.

## [0.6.19] - 2026-08-31

Built against Mojo 1.0.0 (ed45d567).

A group by now chooses how many cores to use, rather than choosing between one and all of them.

Splitting a factorize across workers buys a shorter build and pays for it with a merge no thread can help with, and the merge grows with every worker added, because each one rediscovers whatever groups fall in its own slice. So the two curves cross, and the best worker count is at the crossing. Until now the code could only ask for all of them or none, and it decided with a guard that refused anything projecting more than half a slice of groups. That guard was reading the wrong number. `project_groups` exists to size a hash table, where guessing high costs memory and guessing low costs a rehash, so it extrapolates the discovery rate flat and overshoots on purpose. On ten million rows with a hundred thousand groups it answers six million, and a column that a split wins two and a third times on was going to one thread.

There are two pieces. `_estimate_groups` fits the coupon collector curve those two sample counts actually lie on instead of the tangent, which recovers ninety nine thousand for that column and nine hundred and thirty six thousand for a genuinely high cardinality one, so the refusals that should happen still happen. `_parallel_workers` then costs the route at every worker count the machine can offer, in rows touched, and takes the cheapest if it beats the serial cost by a quarter. The two weights in that cost, what a remap row costs against a build row and what a merged group costs against one, were measured rather than guessed: the route was timed at every worker count from two to thirty two across four cardinalities, and `MERGE_COST` is the weight that puts the model's answer on the measured minimum.

This supersedes the cardinality guard described below, which shipped in the same release cycle and never reached a version of its own.

Ten million rows on an i9-13900K, alternating builds, five reps each, four rounds, medians, one process per measurement. Fifteen untouched benchmarks ran as controls and fourteen of them sat within seven percent.

| benchmark | groups | before | after | ratio |
| --- | --- | --- | --- | --- |
| hash/factorize_100k | 100000 | 88.63 ms | 38.06 ms | 0.43 |
| text/group_medium | 100000 | 49.90 ms | 34.02 ms | 0.68 |
| hash/factorize_10k | 10000 | 11.56 ms | 9.80 ms | 0.85 |
| hash/factorize_nulls | 10000 | 10.64 ms | 9.40 ms | 0.88 |
| hash/factorize_100 | 100 | 6.10 ms | 6.23 ms | 1.02 |
| hash/factorize_all_distinct | 10000000 | 218.20 ms | 214.95 ms | 0.99 |

The first two are the columns this is for and neither was being split at all before. The next two were already parallel and gain from being given twenty five workers instead of thirty two, which is the model declining the last seven because their share of the merge costs more than their share of the build saves. The last two are the ends of the range, where the answer was already right and the point is that it did not change.

The string `factorize` runs on every core too.

Text was the one key type left on a single thread after 0.6.18, and it is the type that matters most for the group by queries anyone benchmarks. It now takes the same route the numeric one does: one contiguous slice per worker, a private table per slice, and a sequential merge that renumbers the local ordinals into global ones in workers-then-ordinals order, which is what preserves first-appearance ordering.

The merge is where the two differ. The numeric one probes on the hash alone, because for it the hash is the key. A string does not fit in a hash, so a match there is a candidate and the rows behind the two keys have to be compared, which is the new `HashTable.insert_string`. That comparison needs a representative row per group and produces one, so the list the merge builds is not scratch that gets discarded, it is the result `factorize_strings` returns. The workers feed it: each records the absolute row of every key that was new to its own table, and the merge visits those in an order that makes the surviving representative the earliest row in the column with that key, which is the row one thread would have picked.

The same cardinality guard applies. A column of distinct strings gets nothing from being split, because the merge would rebuild all of it after the workers already had, so a sample is built first and anything projecting more than half a slice of groups goes to the serial route.

Ten million rows on an i9-13900K, alternating builds in one session, five reps each, four rounds, medians. Fourteen untouched `text/*` and `hash/*` benchmarks ran alongside as controls and all but two sat within four percent.

| benchmark | before | after | ratio |
| --- | --- | --- | --- |
| text/group_repeated | 27.16 ms | 4.19 ms | 0.15 |
| text/group_distinct | 50.59 ms | 51.90 ms | 1.03 |
| text/group_prefixed | 55.74 ms | 55.55 ms | 1.00 |

Only the first of those three is meant to move. The other two are columns of distinct keys, which is exactly what the guard is there to keep on one thread, and their staying flat is the guard working rather than the change failing.

### Changed

- Both hashed routes ask `_parallel_workers` how many workers to use instead of testing a projected group count against half a slice, and `_projected_groups` and `_projected_groups_strings` answer with `_estimate_groups` instead of `project_groups`.
- `factorize_strings` is `raises`, and picks between `_factorize_strings_serial` and `_factorize_strings_parallel` the way the numeric one picks between its two.

### Added

- `_parallel_workers`, which costs the parallel route at every available worker count and returns the cheapest, or one to stay serial, with `REMAP_SHARE`, `MERGE_COST` and `SPLIT_MARGIN` as its weights.
- `_estimate_groups`, which reads a cardinality off two sample counts by fitting the curve they lie on rather than the tangent, replacing `MERGE_HEADROOM`.
- `_factorize_strings_parallel`, the multi threaded string route, `_factorize_strings_serial`, the single threaded one, and `_projected_groups_strings`, the sample build that chooses between them.
- `HashTable.insert_string`, which is `insert` with the key comparison `build_strings` does, for the merge at the end of a parallel string build.
- `hash/factorize_100k` and `text/group_medium` benchmarks, a numeric and a text column of a hundred thousand randomly drawn keys, which is the cardinality band the suite had nothing in.

## [0.6.18] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

`factorize` decides which route to take in one pass that can give up early, instead of two full passes that always finish.

The choice is between a direct table indexed by the value and a hash table, and it is made from the column's minimum and maximum. Those came from a `min_of` and a `max_of`, which is two complete reductions over the column before any grouping starts. Both are thrown away on every column that ends up hashed, which is every float column, every wide-ranging integer column, and every join key that is an identifier rather than a category. On ten million rows a standalone probe puts them at 8.5 ms of the 61 ms `factorize` takes at a hundred groups.

Nothing about the decision needs the exact bounds. The bounds only ever widen as the scan goes, so once the span has passed what a direct table is allowed to be, or once a value has landed outside the window the subtraction is safe in, no later row can bring it back. So the two reductions are fused into one and the test that was waiting for them is moved inside the loop, checked once per validity word. A column of scattered keys now answers in the first few thousand rows rather than in ten million.

The scan is the same shape the aggregation kernels use, which is what makes it cheap on the columns that do take the direct route. A validity word that is all ones and covers a whole block is read as SIMD lanes and reduced with `min` and `max`, and only a partial or gappy word falls back to a row at a time. The all-null column is the one case that has no bounds at all, and it gets a single slot it never reads rather than a hash table.

Measured against a build of the previous release, the two run alternately in one session, five reps each, six rounds, ten million rows on an i9-13900K. Medians. The machine was running other people's benchmark suites throughout, so the `hash/dict_*` rows are carried as a control: they call none of this code and should not move.

| benchmark | before | after | ratio |
| --- | --- | --- | --- |
| factorize_nulls | 65.1 ms | 44.3 ms | 0.68 |
| factorize_direct | 17.8 ms | 14.1 ms | 0.79 |
| factorize_100 | 53.8 ms | 43.1 ms | 0.80 |
| factorize_10k | 61.2 ms | 50.0 ms | 0.82 |
| factorize_all_distinct | 305.9 ms | 299.6 ms | 0.98 |
| dict_100 (control) | 36.2 ms | 35.9 ms | 0.99 |
| dict_10k (control) | 41.9 ms | 41.4 ms | 0.99 |
| dict_all_distinct (control) | 724.9 ms | 779.0 ms | 1.07 |

Every row that runs this code improves and the two quiet controls do not move. The third control is a heavy allocator benchmark whose variance is wide enough on a contended machine that its ratio says nothing either way, which is worth stating rather than dropping.

The size of each gain is the fraction of the work the bounds were. `all_distinct` barely moves because three hundred milliseconds of hashing swamps eight of scanning. `nulls` moves most because its bounds pass was the slowest of the five and the new one skips whole validity words of nulls without reading a value.

What this does not do is win the milestone's group by criterion. Against the language's own `Dict` at ten million rows on this machine, `factorize` is now 43.1 ms against 35.9 at a hundred groups and 50.0 against 41.4 at ten thousand, so `Dict` is still ahead at low and medium cardinality, and 299.6 against 779.0 at all distinct, where the table is ahead by two and a half times. Removing the wasted passes closed part of the gap and did not close it. What is left is in the probe and the build, not in the decision.

The hashed `factorize` runs on every core.

A group by spends most of its time turning a key column into ordinals, and until now that happened on one thread while every engine we are measured against used all of them. The column is now cut into one contiguous slice per worker, each worker builds a private table over its own slice, and a sequential merge afterwards renumbers what they all found into one set of ordinals.

Cutting a column into private tables is easy and getting first-appearance order out the other side is not, because that order is what pandas returns and what every test here compares against. It falls out of two facts. The slices are contiguous and taken in order, so a group's first row is in the earliest slice that contains the group at all. And within a slice the local ordinals are already in first-appearance order, because that is what a build produces. So the merge walking workers in order, and within a worker walking local ordinals in order, visits every group's first row before any later one, and assigning global ordinals in that visiting order gives exactly the sequence one thread would have produced.

The merge costs one probe per group per worker and it is the one part that does not parallelize, so the route is only worth taking when a column's groups are far fewer than a slice is long. A column where every row is its own key would spend the merge rebuilding the whole column serially, having already built it once in parallel, and that measured as a clear loss before this was guarded. So a sample of the column is built first, at most sixty five thousand rows, and the discovery rate over it is extrapolated by the same `project_groups` the table already sizes itself with. A projection of more than half a slice sends the column to the serial route.

Nothing is hashed twice. The workers' tables already hold the hashes, which is what this table calls a key, and `HashTable.keys_by_ordinal` scans the slots to write them out in ordinal order so the merge can probe with them directly. That scan is twice the group count and it replaces hashing the group count over again.

Measured against a build of the previous commit, the two run alternately in one session, five reps each, four rounds, ten million rows on an i9-13900K with thirty two logical cores. Medians. The machine was busy, so three untouched benchmarks are carried as controls.

| benchmark | before | after | ratio |
| --- | --- | --- | --- |
| factorize_100 | 45.5 ms | 8.2 ms | 0.18 |
| factorize_nulls | 46.1 ms | 14.1 ms | 0.31 |
| factorize_10k | 41.5 ms | 14.1 ms | 0.34 |
| factorize_all_distinct | 298.7 ms | 291.9 ms | 0.98 |
| dict_10k (control) | 45.4 ms | 42.0 ms | 0.93 |
| dict_100 (control) | 36.2 ms | 36.6 ms | 1.01 |
| table_probe (control) | 176.3 ms | 219.3 ms | 1.24 |

The three shapes a group by actually meets get between three and five and a half times faster, and `all_distinct` is left alone by the guard as intended. The controls are the honest part of this table: two of them sit within seven percent of where they started and the third moved twenty four percent against us on code neither build touched, which is the size of the noise on this machine on this day and is the reason nothing in the five to ten percent range is claimed here as a result either way.

This clears the milestone's group by criterion. Against the language's own `Dict` at ten million rows, `factorize` is now 8.2 ms against 36.2 at a hundred groups, 14.1 against 45.4 at ten thousand, and 291.9 against 739.5 at all distinct, so the table is ahead by four and a half, three, and two and a half times respectively. Before this it was behind at low and medium cardinality.

Strings are still on one thread. `build_strings` takes the new argument the parallel route needs and nothing calls it that way yet, which is the next piece of work, because four of the ten db-benchmark group by queries key on text.

### Changed

- `factorize` gets its route from `_direct_plan`, which fuses the minimum and the maximum into one scan and returns as soon as the answer is settled. `_direct_span` and the `min_of` and `max_of` pair it called are gone.
- `HashTable.build` and `HashTable.build_strings` take a `rank` alongside `base`. `base` stays the absolute row a chunk starts at, which is what the validity bitmap and the output are indexed by, and `rank` is how many rows this table has already seen, which is what the sizing schedule reads. They are the same number for a column built front to back on one thread, which is every existing caller.
- `factorize` and `_factorize_hashed` are `raises`, because the parallel route calls `parallel_for`.

### Added

- `DirectPlan`, the result of that decision, carrying the number of slots a direct table would need and the value that indexes slot zero.
- `_factorize_hashed_parallel`, the multi threaded hashed route, and `_factorize_hashed_serial`, the single threaded one it is chosen against.
- `_projected_groups`, which builds a sample of a column and extrapolates its group count, so the choice between those two can be made before either has run.
- `HashTable.keys_by_ordinal`, which writes every key a table holds into a buffer indexed by its ordinal.

## [0.6.17] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

Two changes to the CSV reader, one to each half of it. The fill stopped zeroing memory it was about to overwrite, and the scan stopped growing its index by doubling. Together they take a ten million row narrow file from 98.7 ms to 59.3 and a nine column numeric file from 197.9 to 75.6, on an i9-13900K reading off a warm mapping.

The fixed width columns are no longer zeroed before they are filled. The 0.6.16 entry ended with the fill being the larger half of a read for the first time and the numeric columns being where to look, so this is the measurement of that. A single integer column of ten million rows read in 27 ms, and a file of one digit integers read in the same time as a file of seven digit ones, which says the digits are not what is being paid for. What is being paid for is the allocation: `Array[int64](10_000_000)` takes 5.2 ms on its own, all of it a memset, and it runs on one thread before the parallel fill it is for can be handed out.

Nothing needs it. The sweep visits every row of every fixed width column, so the only slot it was leaving to the memset was one holding a missing or unparseable field, and writing a zero there is one store on a path that was already clearing a validity bit. So the columns are allocated with the `Buffer(overwritten=)` the string fill has used since 0.6.15, and the sweep writes the zero itself.

The rest of the entry is the sweep's inner loop, which the change made worth rewriting. It asked on every value which of three kinds of value it was about to parse, though the answer is settled per column before the rows are walked, and it reached the destination through a list index per value. Both are now hoisted into a `fill_tile` parameterized on the dtype, so a tile of rows is a straight loop with its destination in a register. A failed parse also stores now rather than branching around the store, since the parse leaves a zero to store either way.

Measured against a build of the previous release, the two run alternately in one session, five reps each, three rounds, on an i9-13900K, warm cache, reading off a mapping. Medians.

| file | fixed width columns | fill before | fill after | ratio | read before | read after | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nulls | 9 | 139.2 ms | 44.8 ms | 0.32 | 197.9 ms | 105.5 ms | 0.53 |
| narrow | 3 | 62.0 ms | 39.9 ms | 0.64 | 98.7 ms | 76.9 ms | 0.78 |
| wide | 40 | 102.3 ms | 75.4 ms | 0.74 | 145.6 ms | 116.5 ms | 0.80 |
| quoted | 1 | 77.9 ms | 73.5 ms | 0.94 | 111.4 ms | 106.9 ms | 0.96 |

The order of that table is the number of fixed width columns per row of the file, which is the amount of memset removed, and it is also the order of the gains. Nulls is nine columns of ten million values, so it was zeroing seven hundred and twenty megabytes before reading anything. Quoted is one, and it barely moves.

Peak resident memory is unchanged, within a couple of percent either way across repeated runs. It would be: the pages are touched by the fill whether or not they were touched by a memset first. What the change removes is a pass, not a page.

The four ingestion files give byte identical schemas and null counts before and after.

The field index is sized once instead of grown by doubling. With the fill halved the scan is the larger half again, and a scan does two things: it finds the boundaries and it records them. Finding them is a SIMD compare over the block, which is memory bound and near the limit. Recording them was an append to a list that started at nothing, so a ten million row file's index reached three hundred and twenty megabytes through twenty five reallocations that copied about as many bytes as the index ends up holding, on the memory system the compare is already saturating.

So the scan now guesses the size first. A quarter megabyte of the block is counted with the same SIMD compare, the count is extrapolated over the block, an eighth is added on top, and the result is clamped to the one field per two bytes that is the arithmetic ceiling. The guess does not have to be right, only close and on the high side: short by a little costs one reallocation of a nearly finished list, and long costs address space that is never written and so never becomes a page. A delimiter inside a quoted field is counted as a boundary it is not, which pushes the guess up, which is the safe direction.

Measured against a build of the previous release, the two run alternately in one session, five reps each, three rounds, on an i9-13900K, warm cache, reading off a mapping. Medians.

| file | fields | scan before | scan after | ratio | read before | read after | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nulls | 90.0M | 58.3 ms | 30.5 ms | 0.52 | 102.8 ms | 75.6 ms | 0.74 |
| narrow | 40.0M | 35.3 ms | 21.5 ms | 0.61 | 75.3 ms | 59.3 ms | 0.79 |
| wide | 50.0M | 39.0 ms | 25.4 ms | 0.65 | 115.0 ms | 100.1 ms | 0.87 |
| quoted | 30.0M | 32.1 ms | 25.9 ms | 0.81 | 102.1 ms | 97.4 ms | 0.95 |

The fill is flat on all four to within noise, which is the control: nothing outside the scan was touched.

How far off the guess is, whole file, estimated against actual: wide 56.3M against 50.0M, nulls 108.8M against 90.0M, narrow 53.7M against 40.0M, quoted 70.7M against 30.0M. Narrow is high because its early rows have the short ids and short labels and the rest of the file does not, and quoted is high because most of its commas are inside quotes. Both are the harmless direction, and the memory numbers say so.

Peak resident memory for one read in one process, the two builds run alternately, three repeats each, medians in megabytes: narrow 1167 to 1090, quoted 1611 to 1577, nulls 1788 to 1655, wide 1305 to 1277. Every file is lower, because the doubling had to hold the old buffer and the new one at once at every step and this does not. A reserved page that is never written costs nothing resident, which a standalone probe confirms: reserving eight hundred megabytes and touching none of it leaves the process at fourteen.

One caveat on that measurement, because an earlier version of this entry had it wrong. A process that reads the same file several times in a row has a peak resident set well above what one read costs, and it climbs with each read on both builds, because the allocator does not return every freed block to the system and the reads do not ask for the same sizes in the same order. Read once per process and the numbers above are what a read costs. Read four times and the high water mark is several hundred megabytes above that on both builds, which is a property of the arena rather than of the reader.

### Changed

- The sweep allocates its fixed width columns unzeroed and writes every slot itself, including a zero where the field is missing and where it did not parse. The declared type path in `fill_column` still allocates zeroed, because a block there stops at the first value that does not fit and so does not write every slot.
- The sweep's per value three way test on which group a column is in is hoisted out of the row loop into `fill_tile`, which takes the dtype as a parameter and the destination as a pointer.
- `scan_csv` reserves its field index from a sample of the block instead of growing it from nothing. The index still grows if the guess was short, so a file that defeats the sample is slower than it would have been and not wrong.

### Added

- `Array.__init__(overwritten=)` and `ColumnData.__init__(overwritten_bytes=)`, the column level form of the `Buffer` constructor of the same name, for a caller that will write every element.
- `Scan.__init__(capacity=)`, which sizes the field index up front, and `estimate_fields`, which says what to size it to.

## [0.6.16] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

The scan's row offsets are gone. The 0.6.15 notes said the next place to look was the fill, on the grounds that the scan is not search bound. That was right about the scan not being search bound and wrong about where its time went. Lined up against each other the four ingestion files say it plainly: narrow is ten million rows and forty million fields and scans in 60 ms, wide is one million rows and fifty million fields and scans in 42 ms. More fields, more bytes, less time. The scan's cost tracked the row count.

What costs a row is the offset recorded for it. The index is a flat list of packed fields plus a list saying where each row begins in it, which is the Arrow offsets shape and the obvious one. But almost every CSV file is rectangular, and for a rectangular file the offset of row r is r times the width and the list holding it is eight bytes a row of pure redundancy. On a ten million row file that is eighty megabytes written during the scan and read back during every fill.

So the width is kept as a number and the offsets are built only when a row turns up that disagrees, which is a file this reader refuses anyway. `at` multiplies instead of loading, `width` returns a field, and `is_ragged` went from a walk over every row to reading whether the offsets exist, which matters because a read called it once per block and once more for inference.

Measured against a build of the previous release, the two run alternately in one session, five reps each, three rounds, ten million rows on an i9-13900K, warm cache, reading off a mapping. Medians.

| file | rows | scan before | scan after | ratio | read before | read after | ratio | peak RSS after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| narrow | 10M | 59.9 ms | 35.3 ms | 0.59 | 125 ms | 96 ms | 0.77 | 0.93 |
| quoted | 10M | 55.5 ms | 33.6 ms | 0.61 | 141 ms | 117 ms | 0.83 | 0.95 |
| nulls | 10M | 82.2 ms | 58.0 ms | 0.71 | 230 ms | 208 ms | 0.91 | 0.93 |
| wide | 1M | 41.9 ms | 39.1 ms | 0.93 | 147 ms | 150 ms | 1.00 | 0.98 |

Wide is flat, which is the expected answer and the confirmation: one million rows had one million offsets to skip writing, and the file has fifty times as many fields as it has rows.

The four ingestion files give byte identical schemas and null counts before and after. A patch bump, since nothing changes shape.

### Changed

- `Scan` no longer keeps a row offset per row while the file is rectangular. It keeps the width and the row count, and `at(row, column)` indexes at `row * width + column`. A row that disagrees with the ones before it fills the offsets in for every row up to that point, since they all had the same width, and appends from then on, so a ragged file behaves exactly as it did.
- `Scan.is_ragged` is now a constant time test rather than a walk over every row.
- `Scan.end_row` replaces appending to the offsets directly, and both the vectorized scanner and the scalar reference one call it.

## [0.6.15] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

The string concat is gone. The 0.6.14 entry ended by saying that what remained of it was close to the cost of first touching the output pages, so the next change should remove it rather than speed it up, by sizing the string column up front from the field index and letting each block write into its own slice. That is this change.

Sizing it up front means knowing the payload before reading the file, and the field index already holds it. An element costs payload bytes only when it is longer than twelve, the index records every field's start and end, and the one length the index gets wrong is an escaped field's, whose doubled quotes collapse. So the block payload sizes are added up, prefix summed, and the column is allocated once at its full height; each block then writes its own slice of views and its own slice of payload at absolute offsets, so nothing is stacked and no offset is rebased afterwards.

Measuring is itself a second walk over the index, though, and a column whose elements all fit inside their views has no payload for that walk to find. That is the ordinary case, and it is what narrow and wide are: `"row9999999"` and `"s615"` both inline. So the fill is tried first on the guess that nothing reaches the payload, and a block that meets an element too long to inline stops where it stands and reports it; only then is the column measured and filled again. The guess is the whole read when it holds, and the block that disproves it usually does so within a few rows, because a column with long elements rarely hides them at the end. It is the bargain the type ladder already makes.

Measured against a build of the previous release, the two run alternately in one session, five reps each, three rounds, ten million rows on an i9-13900K, warm cache, reading off a mapping.

| file | columns | before | after | ratio | peak RSS before | peak RSS after | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| narrow | 4 | 127 ms | 128 ms | 1.01 | 1.65 GB | 1.50 GB | 0.90 |
| quoted | 3 | 215 ms | 151 ms | 0.70 | 2.70 GB | 1.75 GB | 0.65 |
| nulls | 9 | 240 ms | 207 ms | 0.86 | 2.17 GB | 2.19 GB | 1.01 |
| wide | 50 | 165 ms | 151 ms | 0.92 | 1.46 GB | 1.38 GB | 0.95 |

The nulls row should be read as flat, not as a gain. That file has no text column at all, so it does not reach any of this, and its 0.86 is the machine drifting between the two builds. The honest results are quoted, which is what the change was aimed at, and wide. Narrow is a wash on time and a tenth better on memory: its one text column is entirely inline, so it takes the guessed path, writes its views once instead of writing them into a per block piece and copying them into the column, and never allocates a payload.

The four ingestion files give byte identical schemas and null counts before and after. A patch bump, since nothing changes shape.

### Changed

- A string column is now filled in place rather than built per block and concatenated. The payload size is derived from the field index, the block sizes are prefix summed, and each block writes its views and its payload bytes into its own slice of one column. The concat of string columns is still there and still parallel, it is simply no longer on the path a read takes.
- The fill of a text column speculates that every element fits inside its view, which needs no payload and so no measuring pass. A block that meets a longer element stops and the column is measured and filled again. On a column that is entirely inline this removes the index walk that 0.6.14's design would have added.

### Added

- `collapse_into` in `firepanda.array.strings`, which copies a field's bytes to a destination and collapses doubled quotes as it goes, returning what it wrote. `StringBuilder.append_escaped` now delegates to it, and the reader calls it to write straight into a column's payload.
- `collapsed_length` in `firepanda.io.scan`, which reports what an escaped field measures once its doubled quotes collapse, for a reader sizing a column before it fills it.

## [0.6.14] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

The reader's text path. The 0.6.13 entry ended by saying the quoted file was flat because its time is in unescaping rather than in the index walk, and that this was where to look next. It was half right. Unescaping was costing something, but the larger cost was not the parse at all, it was the concat that stitches the per block columns back into one column at the end of a read. On the quoted file that concat was 175 ms against 68 ms for the parse of the same column.

So this release is three changes on the join rather than on the parse: unescape into the payload instead of into a temporary, do not zero a buffer that is about to be completely overwritten, and paste the parts in parallel now that every part's destination is a prefix sum known in advance. A patch bump, since nothing changes shape and the four ingestion files give byte identical schemas and null counts before and after.

Measured against a build of the previous release, the two run alternately in one session, ten million rows on an i9-13900K, warm cache, reading off a mapping.

| file | columns | before | after | ratio |
| --- | --- | --- | --- | --- |
| narrow | 4 | 157 ms | 130 ms | 0.83 |
| quoted | 3 | 292 ms | 197 ms | 0.68 |
| nulls | 9 | 225 ms | 221 ms | 0.98 |
| wide | 50 | 180 ms | 164 ms | 0.91 |

Per column, on the quoted file and in the same session, the `note` column's concat went from 159 and 152 ms to 29 and 35 ms, and `label`'s from 14.5 and 41 ms to 5.7 and 5.6 ms. Narrow and wide move as well because they have text columns of their own. The nulls file is flat, which is the expected answer: its columns are mostly fixed width and 0.6.13 already took that path.

What remains of the string concat is close to the cost of first touching the output pages, so the next change on this path is to remove the concat rather than speed it up, by sizing the string column up front from the field index and letting each block write into its own slice.

### Changed

- A quoted field is now unescaped straight into the string column's payload instead of into a temporary `String` that is then appended. Collapsing a doubled quote only ever shortens a field, so the builder reserves the raw length, copies the runs between the doubled pairs, and builds the view from what it actually wrote. A field that shortens past twelve bytes ends up inline and the payload offset does not move.
- Concatenating string columns now writes the parts in parallel when the output is at least 65536 rows and there is more than one part. Every part's destination is known before any of it is written, since the row and payload offsets are prefix sums over the parts, so each part memcpys its views and its payload into its own slice and then rebases the offsets in the views it just wrote. Validity is still merged serially, which costs nothing next to the payload copy. Below the threshold the old sequential paste runs unchanged.
- `concat_any` no longer has its own paste loop for the string case. It builds the same part descriptors the typed path builds and takes the same parallel route, which is what a read actually reaches.

### Added

- `Buffer(overwritten=n)`, for a caller that will write every one of the n bytes. It skips the zeroing pass an ordinary `Buffer` does and still zeroes the pad up to the 64-byte capacity, so a vectorized kernel reading one register past the end still sees zeroes. The string concat output is allocated this way.
- `make_inline_at` and `make_long_at` in `firepanda.array.strview`, which build a view from a pointer and a length rather than from a span, for a caller that has already written the bytes.

## [0.6.13] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

One change, to how a read walks the field index. A column at a time meant walking the whole index once per column, and a column's fields sit one row stride apart in it, so a wide file used one word of every cache line it fetched and then came back for the next column and fetched them all again. Filling every fixed width column of a block together, a tile of rows at a time, reads the index once and uses all of it. The wide file, fifty columns, halves. A patch bump: no API changes shape, and the four ingestion files give byte identical schemas and null counts before and after.

Measured against a build of the previous commit, the two run alternately in one session, ten million rows on an i9-13900K, warm cache, reading off a mapping.

| file | columns | before | after | ratio |
| --- | --- | --- | --- | --- |
| narrow | 4 | 167 ms | 151 ms | 0.90 |
| quoted | 3 | 291 ms | 297 ms | 1.02 |
| nulls | 9 | 317 ms | 219 ms | 0.69 |
| wide | 50 | 412 ms | 198 ms | 0.48 |

Peak RSS is unchanged, within a tenth on every file, which is the expected answer: the same buffers are allocated, in a different order.

The quoted file is flat because almost none of its time is in this loop. Two of its three columns are text and go through unescaping, which this change does not touch, and that is where the next reader change should go.

### Changed

- `read_csv` fills every fixed width column of a block in one pass over the block, a tile of rows at a time, instead of walking the whole file once per column. The tile is sized to keep its slice of the field index in the data cache, so every read after the first in a tile comes out of cache, and the columns are still done one at a time within a tile, so a column's running state stays in a register rather than in a list. The two have to be traded off against each other and the tile is where the trade is made.
- The fixed width columns of a frame are grouped by type before they are filled, so the branch that picks a parser is resolved once per column group instead of once per value.

### Added

- `TILE_BYTES`, how much of the field index one tile of a fixed width sweep works over, `sweep_fixed`, which does the sweeping, and `wanted_of`, which names a rung for an error message.

### Notes on the numbers

Three other shapes were built and measured the same way before this one was kept, because the first two were slower than what they replaced.

| shape | narrow | quoted | nulls | wide |
| --- | --- | --- | --- | --- |
| a row at a time, type chosen per value | 1.19 | 1.00 | 0.75 | 0.55 |
| a row at a time, columns grouped by type | 1.12 | 1.03 | 0.87 | 0.54 |
| one parallel region, still a column at a time | 1.07 | 1.10 | 1.01 | 0.97 |
| a tile of rows at a time, columns grouped | 0.90 | 1.02 | 0.69 | 0.48 |

The first two cost narrow more than they saved it. Going row major turns a column's accepted flag, its first bad row and its validity bitmap from things the compiler keeps in registers into list elements indexed by column, and on a four column file that is most of the loop. The third shape is the control: it keeps the column at a time walk and only removes the barrier between columns, and it moves nothing, which is what says the wide file's win is the index traversal and not the scheduling.

## [0.6.12] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

Two changes to the reader, both about not doing work twice. The scanned index is packed into one word per field instead of four, and the pass that decided every column's type before a single value was parsed is gone, replaced by a guess from a sample that corrects itself when it is wrong. A patch bump: nothing in the API changes shape and the reader gives the same answer on every file it gave before.

Each change was measured against a build of the commit before it, the two builds run alternately in one session, because this machine drifts by tens of percent over an afternoon and two numbers taken hours apart are not a comparison. That means the two are not on one scale and cannot be subtracted from each other, so what follows is each one's ratio against its own baseline, ten million rows on an i9-13900K, warm cache, reading off a mapping.

| file | packed index | speculative types | the two together |
| --- | --- | --- | --- |
| narrow | 0.81 | 0.83 | 0.67 |
| quoted | 0.87 | 0.95 | 0.83 |
| nulls | 0.70 | 0.76 | 0.54 |
| wide | 0.94 | 0.66 | 0.62 |

So a third off narrow, a sixth off quoted, not quite half off nulls and just under two fifths off wide. The last column is the product of the two before it, which assumes they do not interact, and they should not: one shrinks the index and the other removes a pass over it.

Peak RSS fell by about a third with the packed index and is unchanged by speculation, which is the expected answer, since speculation allocates the same buffers one pass earlier.

### Added

- `read_csv_as(path, schema)` and `read_csv_as(path, schema, options)`, which read a file with types the caller already knows. `read_csv_bytes_as` has always existed, so the only way to declare a schema was to open and copy the file yourself, which gave up the mapping and read the file twice as slowly for the trouble.
- `Scan.push` and `Scan.field`, the two functions that know how a scanned field is packed, and `Scan.long`, where the length of a field too long to pack is kept.
- `LongField`, the position and length of a field of four megabytes or more.
- `SPECULATE_ROWS`, how many rows of each block a read looks at before it picks a type, and `sample_columns`, which does the looking. `rung_of` maps a declared type back to its rung on the ladder.

### Changed

- A scanned field is one 64-bit word rather than a struct of two offsets and two flags. Forty bits address the buffer, twenty two hold the length and two are the flags. The index a scan produces is the largest thing a read allocates and it is written once and read once, so its size is very nearly all of its cost: for a four column file of ten million rows it was 960 MB over a 357 MB input and it is now 320 MB. `Scan.at` still returns a `FieldSpan`, so nothing above the scanner changed.
- A buffer larger than a terabyte is refused by the scanner with a message that says so, rather than recording offsets that do not fit. A field of four megabytes or more keeps its length in `Scan.long` and is not refused, because a file with one in it is a real file.
- Types are guessed from a sample and the guess is corrected if it turns out wrong, rather than decided by a pass over every value in the file. Each block looks at its first `SPECULATE_ROWS` rows, the column is filled at the rung that sample reached, and a value that does not fit moves the column one rung up and fills it again. A rung read off a sample is never higher than the rung the whole column needs and the ladder only climbs, so the type that comes out is the type the old full pass gave, on every file, and the schemas and null counts of the four ingestion files were compared build against build to check it. Refilling costs one more pass over one column, not over the file.
- `fill_block` and `fill_column` report a value that does not fit rather than raising on it, because whether it is an error at all is now the caller's question: a declared type says it is and a guessed one says the guess was too narrow. When it is an error, the row named is the lowest failing row in the file rather than whichever block reached one first, so the same malformed file always produces the same message.
- A bounded `infer_rows` still decides types first and fills second. A bound is a promise about what gets looked at, and speculating would quietly look past it.

### Notes on the numbers

Ten million rows on an i9-13900K, five repetitions per round, four rounds, warm cache, reading off a mapping. The two changes were measured separately and each against a build of the commit before it, run alternately in one session, because the machine drifts by tens of percent over an afternoon and numbers taken hours apart are not comparable.

| file | 0.6.11 | packed index | speculative types | peak RSS before | peak RSS after |
| --- | --- | --- | --- | --- | --- |
| narrow | 460 ms | 372 ms | 192 ms | 2.10 GB | 1.72 GB |
| quoted | 655 ms | 570 ms | 333 ms | 2.80 GB | 2.57 GB |
| nulls | 940 ms | 660 ms | 381 ms | 3.10 GB | 2.32 GB |
| wide | 830 ms | 780 ms | 473 ms | 2.03 GB | 1.43 GB |

The two middle columns are from different sessions and the machine was slower for the first, so read down a column rather than across the whole row. Against its own baseline the packed index took nulls from 940 to 660 and speculation took it from 499 to 381, and the wide file, which has fifty columns and the most inference to skip, went from 720 to 473.

Peak RSS is unchanged by speculation, within two percent either way, which is the expected answer: the same buffers are allocated, one pass earlier.

## [0.6.11] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

The reader stops copying. A file is mapped rather than read into a buffer, and a fixed width column is parsed straight into the column rather than into a per block piece that is joined afterwards. Two copies of the whole file's worth of data, both of them removed. A patch bump: nothing in the API changes shape, and the one new module is additive.

Reading the four ingestion files on an i9-13900K, ten million rows, warm cache, against 0.6.10: narrow 613 ms to 281 ms, quoted 948 ms to 402 ms, nulls 822 ms to 631 ms, wide 852 ms to 701 ms. Cold cache, which is what the ingestion suite measures, narrow 1045 ms to 625 ms and quoted 1977 ms to 822 ms.

### Added

- `firepanda/io/mapped.mojo`, which maps a file into memory instead of copying it. `MappedFile` owns the mapping and hands out a `Span` that borrows from it, so the span cannot outlive the mapping and the origin system is what enforces that rather than a comment. `map_file` is the same thing with the failure returned as a value, which is the shape a caller with a fallback wants.
- `tests/test_mapped.mojo`, the first tests in the repository that write a real file and read it back. Everything else in the reader is tested against a byte span built in memory, which is the right way to test a parser and no way at all to find out whether the file ever arrives.

### Changed

- A fixed width column is parsed straight into the finished column. The column is allocated at its full height before any block runs and each block writes its own contiguous range of it, so the per block pieces are gone and so is the copy that joined them. Validity is still per block, because two blocks meeting inside a byte would both have to read, modify and write the same word, and the bitmaps are pasted in afterwards, which is a pass over a bit per row rather than over a value per row. A block with no nulls in it is not pasted at all. String columns are unchanged and still build a piece per block, because a block's payload size is not known until the block has been read and there is nothing to write into.
- `read_csv` maps the file rather than copying it. The kernel already has the bytes and a mapping points at them where they are, so nothing is copied, and because the parser touches its blocks on every core at once the page faults are taken in parallel rather than serialised behind one `read` call. A file that cannot be mapped, and an empty one counts, is read the old way.

### Notes on the numbers

Ten million rows on an i9-13900K, five repetitions, warm page cache, reading off a mapping. The direct write is measured against the mapping alone, so the two columns below are the two changes in this release in the order they were made.

| file | mapped | mapped and written direct |
| --- | --- | --- |
| narrow | 348 ms | 281 ms |
| quoted | 431 ms | 402 ms |
| nulls | 791 ms | 631 ms |
| wide | 701 ms | 701 ms |

The narrow file is four columns of nothing but numbers over ten million rows, so all of the copy that went away was on its critical path, and it gains a fifth. The nulls file gains the most in absolute terms: nine values in ten are missing there, and the join was copying a validity bitmap along with the values on every one of eight columns. The wide file does not move, which is the expected answer rather than a disappointing one. It is fifty columns of a million rows, so a column is eight megabytes against the narrow file's eighty, and what a wide read spends its time on is fifty million fields rather than the bytes that come out of them.

Peak resident memory on the quoted file goes from 2.80 GB to 2.77 GB. That it barely moves is the honest result and it is worth saying why. The peak is not the columns, it is the scan index, which is a twenty four byte span per field plus eight bytes per row and comes to 960 MB on the narrow file for a 357 MB input. That is the next thing to go.

Ten million rows on an i9-13900K, five repetitions, warm page cache, both paths in the same process back to back so they share a cache and a heap.

| file | size | copy | map |
| --- | --- | --- | --- |
| narrow | 357 MB | 613 ms | 348 ms |
| quoted | 577 MB | 948 ms | 431 ms |
| wide | 383 MB | 852 ms | 701 ms |
| nulls | 207 MB | 822 ms | 791 ms |

The getting the bytes step itself goes from 199 ms to 0.026 ms on the narrow file and from 523 ms to 0.036 ms on the quoted one, because a map is three system calls and no work. The whole read saves more than that, which is the destination pages of the copy no longer being faulted in and zeroed.

Cold cache, with the page cache dropped before every single run, is the case the ingestion suite measures and it is better rather than worse: 1045 ms to 625 ms on the narrow file and 1977 ms to 822 ms on the quoted one. Thirty two workers faulting in parallel pull from the device harder than one sequential `read` does.

The same measurement on an eight core AMD EPYC, where the copy is a larger share of everything, warm: narrow 17311 ms to 8804 ms, quoted 12500 ms to 5687 ms, wide 21551 ms to 13096 ms, nulls 14639 ms to 12145 ms. Cold there, narrow 8753 ms to 4049 ms and quoted 14931 ms to 11230 ms.

Peak resident memory on the quoted file falls from 3.48 GB to 2.80 GB, which is the copy that no longer exists.

## [0.6.10] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

Stacking string columns stops touching elements. A patch bump: no API changes shape, one kernel gets between four and nine times faster, and two functions are added that exist because that kernel needed them.

### Added

- `concat_strings`, the typed spelling of a concat of string columns, beside the `concat_arrays` that has always been there for the fixed width ones.
- `Bitmap.paste`, which writes a run of bits into a bitmap at a bit offset. It is the mirror of `slice` and it is harder, because the destination is shared: the bits either side of the run belong to a different part and have to survive, so every word is a read, a mask and a write, and a run that does not start on a word boundary is written as two halves. Tested against a loop over bits at every offset from 0 to 70 for five run lengths.
- `StringView.shift_offset`, which moves a long element's payload offset along by a fixed amount. That is the whole of what stacking two payloads costs a view.
- Two benchmark rows, `strings/concat_short` and `strings/concat_long`, stacking eight parts of a quarter million elements each, which is the shape `read_csv` hands the kernel.

### Changed

- A concat of string columns copies blocks rather than elements. It walked every element into a `StringBuilder`, which copied the bytes into a growing payload, appended a view to one `List` and a flag to another, and then copied both `List`s again in `finish`. Four copies per element on a path `read_csv` runs once per column. The views of a part are now one memcpy and its payload is another, and the only per element work left is adding the part's payload base to the offset field of the views long enough to have one. Short elements are not touched at all.
- The validity of a part with nulls goes across through `Bitmap.paste` rather than one bit at a time.

### Notes on the numbers

Measured on an i9-13900K, eight parts of 262,144 elements, five repetitions: 2.339 ms to 0.265 ms on a column of eight byte elements, and 4.897 ms to 1.218 ms on a column of thirty two byte ones, which is 8.8 and 4.0 times. The gap between the two is the payload, which still has to be copied and is now the whole cost of the long case.

Inside `read_csv` at ten million rows, the phase that stacks the per block pieces: 833 ms to 83 ms on a file of two text columns, 268 ms to 43 ms on a four column one, 176 ms to 55 ms on a fifty column one. A file with no text column does not move, which is expected, since its stack was already a memcpy.

The end to end read does not move by as much as that, and it is worth saying why rather than quoting the phase number and stopping. Reading the file into memory is 400 ms of a narrow read and 760 ms of a quoted one, which is now the largest single part of it, and at these sizes the wall clock of a whole read on this machine varies by twenty to thirty per cent between runs. The file read is the next piece of work.

## [0.6.9] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

`read_csv` runs on every core. A patch bump: nothing changes shape, one thing gets faster, and the library gains its first piece of threading.

### Added

- `firepanda/exec`, which is the whole of the library's threading. One function, `parallel_for`, which runs a body once per index and returns when all of them have finished. It is built on `TaskGroup` from `std.runtime.asyncrt` rather than on `parallelize`, which is no longer in the standard library and now lives behind a GPU dependency in `max.algorithm`.
- `firepanda/io/split.mojo`, which cuts a CSV buffer into blocks that each begin on a row boundary. A newline inside a quoted field is data rather than a separator, so the quote state at an offset is what decides, and that state is the parity of the quote bytes before it. Counting bytes is parallel and a prefix sum over one number per block is free, so every block learns its starting state without reading what came before it.
- `scan_block`, which scans a byte range of a buffer while keeping the offsets absolute, and `scan_blocks`, which runs one per block and checks the result.
- A CSV round in `tests/stress/main.mojo`. It builds a file whose every value is a function of its row index, cuts it into a randomized number of blocks between one and thirty three, checks the blocks against a single pass field span by field span, then reads it the way a caller would and checks the values against the generator rather than against another run of the reader.
- `tests/test_split.mojo`, twenty two tests, and `tests/test_parallel.mojo`, eight.

### Changed

- `read_csv` runs on every core for a file large enough to be worth it, which is a quarter of a megabyte per core. The scan, the inference and the conversion all run one task per block. Nothing is shared between blocks: each scans into its own `Scan`, guesses its own types, and fills its own piece of each column.
- `Scan` records how many quote bytes it read as structure. That number is what makes the split checkable rather than merely plausible; see below.
- Columns are stacked one at a time rather than all at once. The parallelism is over blocks within a column, so every core is still busy, and the peak holds two copies of one column instead of two copies of the whole frame.

### Why a wrong split cannot pass silently

The parity argument holds under RFC 4180 and this reader is deliberately looser than RFC 4180: it accepts a bare quote in the middle of an unquoted field, because pandas does. Such a quote is data, it does not come in a pair, and it flips the parity of every offset after it.

So the split is checked. A block's structural quote count can never exceed the quote bytes in its own byte range, and the blocks partition the buffer, so if the totals agree over the whole file they agree over every block, which means no quote byte was read as data. Block zero starts at offset zero, which is a row boundary outside quotes by definition, so block zero's parse is correct, so the next block's start is a row boundary too, and the induction runs to the end.

The other half is that a boundary with the wrong parity cuts the block before it inside a quoted field, and a quoted field with no closing quote is an error the scanner already raises. When either fires the whole read is done again on one thread, which is also what produces the right file row number in the message for a file that really is malformed.

### Notes on the numbers

Measured on an i9-13900K, sixteen physical cores and thirty two logical, against one million rows of four columns, 28.2 MB, the same file for every reader and seven runs each.

firepanda reads it in 50.3 ms median, against 145.9 ms for the same code before this change. That is 2.9 times, on a machine with thirty two logical cores, and the reason it is not more is that only part of the work moved. The scan went from 45.2 ms to 7.5 ms, which is six times and close to what the hardware allows. The rest of the read, inference and conversion and stacking, is 32 ms of the remaining 39 ms and did not improve nearly as much. That is where the next piece of work is.

On the same file `pandas.read_csv` takes 237.5 ms with the C engine and 11.1 ms with the pyarrow engine, and `polars.read_csv` takes 6.2 ms. So firepanda is now four and a half times faster than the C engine and still four and a half times behind the pyarrow one. The M1 exit criterion asks for a reader that beats the pyarrow engine and this is not yet that reader.

## [0.6.8] - 2026-08-29

Built against Mojo 1.0.0 (ed45d567).

firepanda reads and writes CSV files, and a text column converts to a number and back. Those are the last two things on M1's scope list that were missing rather than merely slow.

A patch bump. Two fixes, and the rest is addition.

### Added

- A text column casts to a number and a number column casts to text. `cast_strings_to` reads bytes through the same parser the CSV reader uses, and `cast_to_strings` writes them the way the CSV writer writes them, so a value that survives a file round trip survives this one.
- `cast_any` and `Series.cast` and `DataFrame.cast` take a `LogicalType` as well as a `DType`. The dtype form cannot name text, text having no dtype of its own, so `series.cast(LogicalType.STRING)` is the way to ask for it.
- All three take a `strict` flag, defaulting to true. Strict raises on a text value that is not a number and names the row and the value. Not strict writes a null, which is what `to_numeric(errors="coerce")` does.
- `tests/test_text_cast.mojo`, twenty three tests.
- `read_csv`, which turns a file or a buffer of bytes into a `DataFrame`, inferring the schema from the first rows or taking one you hand it. Names come from the header when there is one and are `column_0` upward when there is not.
- `write_csv`, which turns a `DataFrame` back into a file or a buffer of bytes, quoting a field only when leaving it bare would move a boundary.
- `ReadOptions` and `WriteOptions`, carrying the dialect, whether there is a header, and how many rows the inference is allowed to look at. `INFER_ALL` reads the whole file before deciding.
- `benchmarks/read_file.mojo`, a standalone timed read of a file that already exists, so the comparison against `pandas.read_csv` can be run on byte identical input rather than on two files of the same description.
- Three benchmark rows, `csv/read_inferred`, `csv/read_declared` and `csv/write`.
- `tests/test_csv.mojo`, twenty eight tests.

### Changed

- `FieldSpan` records whether the field was written inside quotes. An empty field and a quoted empty field are the two ways a CSV file has of writing a missing value and the empty string, and without the flag the reader has to guess, and whichever way it guesses one of the two becomes unrepresentable.

### Fixed

- `parse_float` was an ulp or two off from a correctly rounded `strtod` outside the exponent range where a single multiply is exact, because it scaled in steps of 1e22 and rounded once per step. Five steps of that put `1.2345678901234567e100` one ulp away, so a float written at seventeen digits did not read back as itself. The fast path is unchanged and still one multiply. Everything outside it now goes to the platform's `strtod`, which is correctly rounded at every exponent, and the grammar is still checked here first, so the fallback answers the question and does not get to widen it.
- Casting a text column raised rather than converting, because the numeric path would have found its uint8 physical dtype and converted the first byte of every sixteen byte view. It converts now.

### Why the empty string is not a number

Coercing it to a null quietly would throw away the distinction the string column went to some trouble to keep. An empty field and a quoted empty field are different values in a CSV file, the reader keeps them apart, and a cast that collapses them undoes that two lines later.

### How the type of a column is decided

Inference climbs a ladder, bool to int64 to float64 to string, and never descends. A value that does not fit the current rung moves the column up one rung and the column keeps the widest rung any value forced. A missing value decides nothing, so a column of numbers with a gap in it is still a column of numbers, and a column with nothing in it at all is text, because text is the only rung that can hold whatever eventually turns up. A quoted field is not promoted to text for being quoted, since quoting says where a field ends and not what is in it, but a quoted empty string is the empty string and not a null, which is the one place quoting does change a value.

### Notes on the numbers

Measured on an i9-13900K against one million rows of four columns, 20.7 MB, the same file for every reader. firepanda reads it in 139.7 ms median. `pandas.read_csv` with the C engine takes 137.1 ms and with the pyarrow engine 28.8 ms from the file and 12.4 ms from memory. So this is at parity with the engine pandas has had for fifteen years and roughly eleven times behind the threaded Arrow reader. The gap is threads. The scan and the conversion both run on one core here, and Arrow runs both across all of them. The M1 exit criterion asks for a reader that beats the pyarrow engine and this is not that reader yet.

## [0.6.7] - 2026-08-28

Built against Mojo 1.0.0 (ed45d567).

A text column can be aggregated, not only grouped by. With 0.6.5 sorting one and 0.6.6 grouping by one, a string column is now a first class column everywhere a group by can reach it.

A patch bump. One fix, and the rest is addition.

### Added

- A text column can be aggregated. `count`, `size`, `nunique`, `first`, `last`, `min` and `max` all work on a string column, per group, through the same `group_by` call the number columns go through.
- `aggregate_group_strings`, the kernel behind it, and `group_text_scalar`, the obvious twin the tests compare it against.
- `tests/test_text_agg.mojo`, eighteen tests, and a round in the string fuzzer that checks the four value reductions against an inline reference.

### Fixed

- A string column reaching the aggregation dispatch matched `uint8` and was summed as bytes, so `sum` over a column of names returned a number rather than an error. The dispatch now asks whether the column is text first, the same way the sort, group by and join dispatches do. That is the last of the three places the hazard lived.

### Why nunique and min take different routes

`nunique` factorizes the column and then runs the number kernel over the ordinals, because two rows hold the same bytes exactly when they share an ordinal, and the number kernel already knows how to count distinct values per group. `first`, `last`, `min` and `max` never copy a value during the scan. They keep one row number per group and gather once at the end through a `StringBuilder`, so a column of long values costs the same to reduce as a column of short ones and only the surviving values are ever written.

### Notes on the numbers

Measured on an i9-13900K, two hundred thousand rows over a hundred groups, fifteen repetitions, interquartile range around 1 percent. `min` over a text column runs at 6.0 ns a row and `nunique` at 16.9, the gap being the factorize pass the second one pays for. On the AMD EPYC VPS the same rows are 31.3 and 64.5 ns.

## [0.6.6] - 2026-08-28

Built against Mojo 1.0.0 (ed45d567).

A text column can be a group by key and a join key. 0.6.5 made one sortable and this makes one groupable, which together are what db-benchmark's group by queries need from a string column.

A patch bump. One fix, and the rest is addition.

### Added

- A text column can be a group by key and a join key. `group_by` accepts one on its own or alongside number keys, with or without `dropna`, sorted or in first seen order, and the joins take one on either side.
- `factorize_strings`, which assigns a group ordinal to every element of a `StringArray`, and `FactorizedStrings`, which carries the codes, the row that first showed each group and the null group.
- `hash_bytes` and `hash_strings_chunk`, a length seeded hash over a run of bytes and the chunked form the table build consumes.
- `HashTable.build_strings`, the probe loop for keys that do not fit in the hash.
- `factorize_strings_linear`, the scalar twin, which compares bytes against every group it has seen and never hashes.
- `tests/test_text_group.mojo`, twenty one tests, and a factorize round in the string fuzzer against a reference that shares no code with the kernel.

### Fixed

- A string column reaching the group by and join dispatch matched `uint8` and was grouped on the first byte of each view, which put every value starting with the same letter in one group. Both dispatches now ask whether the column is text before the type walk. The same hazard was fixed in the sort in 0.6.5 and this is the rest of it.

### Why text needs its own table build

The existing table stores a 64 bit hash and treats hash equality as key equality. That is exact for a fixed width key, because the mix is a bijection on 64 bits, so two keys that hash alike are the same key. Text does not fit in 64 bits and the hash is a real hash, so `build_strings` compares the bytes against the row that first showed the group and keeps probing when they differ. The two builds are kept apart rather than merged behind a flag, so the fixed width one stays a loop with nothing extra in it.

### Notes on the numbers

Measured on an i9-13900K, one million rows, fifteen repetitions, interquartile range under 4 percent on every row. Distinct text keys group at 14.9 ns a row against 14.3 for an int64 column of the same height that also goes through the hash. A column where every element shares a nine byte prefix costs 16.8 ns a row, which is the case that costs the sort 307.6, because a group only has to find its bucket and does not have to order anything. A hundred repeated values cost 10.6 ns a row against 3.6 for the int64 column, which takes a direct route that text has no equivalent of.

## [0.6.5] - 2026-08-28

Built against Mojo 1.0.0 (ed45d567).

A text column can be sorted. 0.6.4 got one into the frame layer and left it unsortable, and this is the ordering that db-benchmark q1, q2, q3, q7 and q10 need.

A patch bump. Everything here is addition apart from one fix to a round trip that was returning mojibake.

### Added

- A text column can be ordered. `sort_values`, `argsort` and a frame sort all accept one, ascending or descending, nulls first or last, and the order is stable.
- `StringArray.sort_prefix`, `compare_elements` and `compare`, which are the three pieces the sort needs: a radix key, an ordering between two elements of a column, and an ordering between an element and a run of bytes for a future search.
- `tests/test_text_sort.mojo`, fourteen tests that separate the radix half from the tie break half so that neither can pass on the other's behalf.
- A differential sort round in the string fuzzer, against a reference insertion sort that shares no code with the kernel. It compares the permutation position by position rather than the sorted elements, because two permutations can produce the same elements and only one of them is stable.

### Fixed

- `StringArray.__getitem__` returned mojibake for any element with a byte above 127. It appended `chr(byte)` per byte, which treats every byte as a code point, so a column holding "ábove" read back as "Ã¡bove". It now hands the bytes over whole. The column stores bytes and does not interpret them, and this is the one place that has to say what they mean.

### How the sort works

The first eight bytes of an element pack into a `UInt64` most significant byte first, so comparing two of those integers gives the same answer as comparing the first eight bytes of the elements, and a shorter element sorts before one that extends it. That means the existing radix sort runs over text unchanged, at eight digits. Only the runs whose keys came out identical need a comparison, and those are finished by a stable insertion sort below sixteen rows and a stable bottom-up merge above it.

A run whose elements are all at most eight bytes and all the same length is already in its final order, because the key held the whole element, and it is skipped without a single comparison. That is not a corner case, it is the shape of the columns a dataframe most often sorts. Without it a column of a hundred distinct two byte labels cost 403 ns a row against 27 for an int64 column of the same height.

### Notes on the numbers

Measured on an i9-13900K, 262144 rows. Distinct keys 23.2 ns a row against 17.2 for an int64 column of the same height through the same entry point. A hundred repeated short values 14.8 ns a row, which is faster than the int64 column because the settled check skips every comparison and the low entropy keys let the radix passes skip digits. The pathological case is a column where every element shares a nine byte prefix, at 424 ns a row, where the radix does no work at all and the merge does all of it. A wider key would move that case rather than fix it.

## [0.6.4] - 2026-08-28

Built against Mojo 1.0.0 (ed45d567).

A `DataFrame` can hold a text column. 0.6.3 shipped `StringArray` as a standalone module with nothing above it able to hold one, and this connects it to `AnyArray`, which is the type-erased column a `Series` and a frame are made of.

A patch bump. Everything here is addition, and the four operations that now refuse a text column previously could not be handed one at all.

### Added

- `AnyArray` carries a string column. New `AnyArray(StringArray)` constructor, plus `is_string()`, `strings()` for a borrow and `into_strings()` for a move.
- `Series` takes a `StringArray`, and reports `is_string()`, `as_strings()` and `text(i)`.
- Take, filter, slice, concat, coalesce, fill forward and fill backward all carry text through the erased path, which means a `DataFrame` can select, reorder, cut and stack frames with text columns in them.
- The display layer prints a text value rather than its type, and `<NA>` for a null. Values are not quoted and not escaped, which is what pandas does too, because a table is read by a person and `to_csv` is where escaping belongs.
- `tests/test_text.mojo`, twenty one tests, one per operation that reads values.
- A `text/` benchmark group measuring the erased path, with every text row paired against an int64 column of the same height through the same entry point.

### Fixed

- Every kernel that dispatches by matching on `col.dtype()` now asks `is_string()` first. `LogicalType.STRING` has physical dtype uint8 and uint8 is a member of both `ALL` and `ORDERED`, so a string column reaching one of those dispatches would have matched the uint8 arm and read the first byte of a sixteen byte view as the value. That is not a crash and not an obviously wrong answer, it is a plausible number, and a sum over a column of country codes would have returned a total. This was unreachable before 0.6.3 because no frame could hold a text column, and it is closed in the same release that makes it reachable.
- `AnyArray.check_dtype` refuses a string column for every dtype, so `as_typed[DType.uint8]()` on a column of names raises instead of handing back a column over the views buffer.
- `concat` compares `is_string()` as well as `dtype()`. Both a text column and a column of bytes are uint8 physically, so the dtype check on its own let the pair through.

### Not yet

Cast, ordering comparison and group by aggregation refuse a text column, each with a message naming what is missing rather than a dtype the caller did not choose. Ordering is next and is what db-benchmark q1, q2, q3, q7 and q10 need.

### The validity duplication

`AnyArray` keeps the validity bitmap in `data` as well as inside the `StringArray`. `is_null`, `is_not_null` and the all-present mask that a join and a group by both build read `data.validity` and never look at a value, so keeping the bitmap where they already look means those three need no text case. A column is immutable once constructed, so both copies are written from the same source in the same call and cannot drift. The cost is one bit per row per text column and it goes away when columns are refcounted rather than deep copied.

### Notes on the numbers

Measured on an i9-13900K at 262144 rows, ten repetitions, median. Per row: take 78.5 ns for text against 2.0 ns for int64, filter 8.8 against 1.4, concat 25.1 against 0.4, slice 15.5 against 0.2. The `is_string` guard that is now on the front of every erased kernel costs 0.14 ns.

The 40x on the gather is mostly not inherent. A standalone probe separates it: at 32 byte elements a sequential take is 48 ns/row and a scattered one is 247, which is two dependent cache misses per row, the view and then the payload it points at. Every one of those addresses is derivable before the loop starts because the index list is the input, and none of them are prefetched today. That is worth fixing before string keys reach a join.

## [0.6.3] - 2026-08-28

Built against Mojo 1.0.0 (ed45d567).

The variable width string column, which is the last thing a CSV reader was missing and the reason 0.6.2 shipped a scanner with nowhere to put a text field. Plus the continuous integration work that took the pull request pipeline from about twelve minutes to about three.

A patch bump. The column is a new module and the pipeline changes are not API.

### Added

- `firepanda/array/strings.mojo`, holding `StringArray` and `StringBuilder`. A column is a views buffer of sixteen bytes per element, a payload buffer holding the bytes of the long elements, and a validity bitmap. An element of twelve bytes or fewer lives entirely inside its own view, so a column of country codes, status labels or short names is one flat array with no second buffer touched at all. A longer element keeps its length and its first four bytes in the view and its bytes in the payload.
- `StringBuilder`, which is the only way to make a column. Append a field, append a null, ask for the result. `finish` consumes the builder and hands its payload buffer to the column rather than copying it again.
- `StringArray.unsafe_bytes`, which returns a `Span` into whichever buffer the element lives in, so a kernel can read an element without allocating a `String` for it. The span is tied to the column's origin, so nothing it produces can outlive the column.
- `equals` for comparing an element against a run of bytes, and `element_equals` for comparing two elements of the same column. Both are the operations a join key or a group by needs.
- `slice`, `take`, `filter`, `to_list` and a deep copy, all of which produce an independent column.
- Twenty four tests in `tests/test_strings.mojo`, including every length from zero to thirty, the inline limit approached from both sides, a null after a long element, and two thousand random elements with nulls mixed in.
- `tests/fuzz/strings.mojo`, a fifth fuzzer, which checks the column against a `List[String]` reference through random rounds of slice, take, filter, copy and rebuild. A third of the lengths it draws land within two bytes of the inline limit.
- Ten benchmark rows under `strings/`, every one of them measured at eight bytes and at thirty two so the two storage paths are always side by side.

### How it works, and what follows from it

The classic Arrow string layout gives every element an offset into one data buffer, so reading any string is two dependent loads and knowing its length is two more. This layout puts the length and a four byte prefix in the element itself. A length costs one load whatever the element is, and two long elements that differ in their first four bytes are settled without either payload being read.

The cost is sixteen bytes per element against Arrow's four or eight. On a column of paragraphs that is a loss. On the short repeated text that dataframes are actually full of it is a large win, and short is the case worth optimizing for.

A finished column has exactly one payload block, so every long view carries block index zero. That field is not wasted. It is what will let a chunked string column share payload across chunks later without rewriting a single view.

Slicing, taking and filtering copy rather than pointing into the source column's payload. This is the same decision `Array` and `Bitmap.slice` already made, for the same reason: a view into another column's payload would make every column's lifetime depend on every column it was ever cut from.

### Notes on the numbers

Measured on gamingpc, sixteen physical cores, thirty two byte registers, at 262144 elements, ten repetitions, median reported.

```
strings/build_short          566.130 us     4.2%     8.638 ns   115.76 Mrows/s
strings/build_long           959.114 us     2.9%    14.635 ns    68.33 Mrows/s
strings/length_short          12.987 us     1.8%     0.198 ns     5.05 Grows/s
strings/length_long           12.385 us     3.0%     0.189 ns     5.29 Grows/s
strings/bytes_short           53.789 us     0.8%     0.821 ns     1.22 Grows/s
strings/bytes_long            63.092 us     1.7%     0.963 ns     1.04 Grows/s
strings/equals_prefix        148.520 us     0.9%     2.266 ns  441.25 Mpairs/s
strings/equals_payload       240.393 us     0.6%     3.668 ns  272.62 Mpairs/s
strings/take                   1.250 ms     6.3%    19.075 ns    52.42 Mrows/s
strings/filter               455.942 us     0.6%     6.957 ns  143.74 Mrows/s
```

`length_short` and `length_long` land on top of each other at a fifth of a nanosecond, which is the layout's first claim and the one the offsets layout cannot make. Asking a thirty two byte element how long it is costs exactly what asking an eight byte element costs.

`bytes_short` against `bytes_long` does not show the gap it looks like it should. Both walks are in order and a prefetcher handles two sequential streams as easily as one, so reading the payload is nearly free here. The inline layout is worth nothing to a scan. It is worth something to everything that jumps.

`equals_prefix` against `equals_payload` is where it pays. Both compare adjacent long elements that are not equal and both answer false, and the only difference is whether the difference is in the four bytes the view already holds. It is 1.6x on this machine and 1.9x on the eight core server, on elements of thirty two bytes with the payload in cache. Neither of those is the interesting case. The interesting case is a join key column that does not fit in cache, where settling a comparison in the view is a cache miss that does not happen.

Two things were slower before they were measured. The payload started as a `List[UInt8]` appended one byte at a time, which cost 143 ns per row on a long build, and `take` of a quarter of a million rows took 42 ms. Replacing the loop with one copy per field and a `resize` made it worse, not better, because `resize` allocates exactly what it was asked for and turns a growing payload into a quadratic copy: the same `take` went to 3.2 seconds. The payload is now a buffer that doubles, and the build is 14.6 ns per row.

### Changed

- The test runner runs several files at once rather than one after another. Each test file is a separate `mojo run` that compiles the library again, so the step was spending most of its wall clock in the compiler with one core busy. Output is still collected per file and printed in filename order, so the log reads the same. `FIREPANDA_TEST_JOBS=1` restores the serial run. On an eight core machine the suite went from 70.7 s to 26.9 s.
- The fuzzers run at once for the same reason, through `tools/run_fuzz.sh`. `--max-total-time=N` still means N seconds per fuzzer rather than N in total.
- Continuous integration no longer builds against the Mojo nightly toolchain on every pull request. `.github/workflows/nightly.yml` already does that every morning and opens a tracking issue when it breaks, and the duplicate was the slowest job in the pull request pipeline.
- The microbenchmark job measures the previous commit on pushes to main rather than the merge base on every pull request. Measuring both sides doubled the job, and what it bought was a comparison the job already prints as advisory. The performance gate with teeth is the reference machine run recorded in each pull request.
- The benchmark harness takes a `--max-time` flag, in milliseconds, for the wall clock ceiling on one repetition. It defaults to what the harness has always used, three times the minimum plus a quarter of a second, so nothing changes for a developer who does not pass it. It exists because that ceiling is what the suite's runtime is actually made of: `run` keeps sampling until the ceiling rather than stopping at the minimum, so the total is roughly the ceiling times the benchmark count times the repetitions, and the row count barely enters into it. Halving the rows moved a full run from 310 s to 276 s. Halving the ceiling halves it.
- The link check retries and accepts a rate limit response rather than failing the build on it. It failed a pull request because one host reset one connection, which is a flaky check rather than a broken link.
- Continuous integration runs the suite at five repetitions with a twenty millisecond minimum and a forty millisecond ceiling, which is 80 s against 316 s for the settings it used before. The measured spread hardly moves: median interquartile range 13.7% against 12.1%, ninetieth percentile identical at 50%. On a shared runner the noise floor is the machine rather than the sampling budget.

### Known limitations

- The column is not wired into `AnyArray`, `Series`, `DataFrame` or the display layer, so nothing that takes a frame can hold one yet. That is the next change and it is what actually unblocks `read_csv`.
- There is no comparison other than equality. Ordering strings needs a lexicographic compare that uses the prefix, and sorting or grouping by a string column needs that first.
- `element_equals` on two long elements with the same prefix walks the payload a word at a time. A column of URLs, where thousands of elements share their first four bytes, gets nothing from the prefix and pays the full walk.
- The bytes are not validated as UTF-8, deliberately. A CSV field is bytes and a column that refuses to hold what the file contains cannot read the file.

## [0.6.2] - 2026-08-28

Built against Mojo 1.0.0 (ed45d567).

The two halves of a CSV reader that do not need a frame: a scanner that finds where every field in a buffer is, and parsers that turn a field's bytes into an integer, a float or a boolean. There is no `read_csv` yet, because there is still no variable width string column for a text field to land in, and building that column and the reader and these kernels in one change would have made a pull request nobody could review.

A patch bump. Everything here is a new module and nothing that existed changed shape or behaviour.

### Added

- `firepanda/io/scan.mojo`, an RFC 4180 field scanner. `scan_csv` takes a span of bytes and a `Dialect` and returns a `Scan`, which is one flat list of field spans plus a list of row offsets into it, the same shape as an Arrow offsets buffer and for the same reason: one allocation for the file rather than one per row. A field span carries its start, its end and a flag saying whether it contains a doubled quote, so the reader only pays for unescaping on the fields that need it.
- `firepanda/io/parse.mojo`, the field parsers. `parse_int` handles an optional sign and digits with the range check done against the target dtype rather than against `Int64`, so `parse_int[DType.int8]` refuses 128 and `parse_int[DType.uint32]` refuses a negative. `parse_float` handles a sign, a decimal point, an exponent and the words `nan`, `inf` and `infinity` in any case. `parse_bool` takes `true` and `false` in the three usual capitalisations and nothing else. `is_missing` recognises the empty field along with `-`, `na`, `n/a`, `nan`, `nil`, `null` and `none`.
- `firepanda/io/scalar.mojo`, a byte at a time scanner that produces byte identical output to the vectorized one, so a block boundary landing in the middle of a field cannot pass a test unnoticed.
- `field_bytes` and `unescape` in `firepanda/io/scan.mojo`, for turning a span back into bytes with and without the doubled quotes collapsed.
- Forty six tests across `tests/test_parse.mojo` and `tests/test_scan.mojo`, including four hundred random buffers drawn from an alphabet loaded with commas, quotes, newlines and carriage returns, on which the two scanners have to agree on whether the buffer is legal and on every field boundary in it.
- Eight benchmark rows under `csv/`, covering a narrow file, a wide one, a quoted one, a long text one, and the two scanners run against each other on both the narrow and the long shapes.

### How it works, and what follows from it

A parse failure is a value and not an exception. `parse_int` and friends return a `Parsed[dt]` holding a value and an `ok` flag, because a reader with a bad field in row four million wants to record which row it was and carry on, and an exception per bad field would cost more than the parse.

Nothing is guessed. A field with trailing bytes after the number is a failure rather than a truncated value, so `12abc` does not become 12. A quoted field that never closes is an error naming the row and the byte, and so are bytes sitting between a closing quote and the next delimiter. Readers that accept those files do it by inventing a value, and an invented value in a data file is worse than a failed read.

The integer path is exact by construction and the float path is exact in the common case. The integer accumulator checks for overflow before the multiply rather than after it, so it never relies on wraparound. The float path builds the mantissa as a `UInt64` and applies the decimal exponent in one multiply or divide when the mantissa fits in 53 bits and the exponent is within 22, which is the range where a single rounding is provably correct, and falls back to scaling in exact steps of 1e22 outside it.

The scan is a separate pass over the buffer rather than a parse as it goes loop. That is one more pass over the text, and it buys three things: the second pass walks a compact offsets array instead of text, each column's parse knows its dtype and can be a tight typed loop instead of a switch, and the row count is known before a single column is allocated so nothing has to grow.

Blank lines are skipped, which is what pandas does and what stops a file ending in a newline from producing a phantom last row. Ragged rows are reported through `Scan.is_ragged()` rather than refused, because what to do about them is the reader's policy question and not the scanner's.

### Notes on the numbers

Measured on the reference machine, 16 physical cores, 32 byte registers, at 1,048,576 rows for the narrow shapes and 262,144 for the long one, 10 repetitions, median with the interquartile range next to it.

```
csv/scan_narrow               10.293 ms    34.8%    39.263 ns    25.47 Mrows/s
csv/scan_scalar_twin          10.419 ms     3.8%    39.745 ns    25.16 Mrows/s
csv/scan_long_text             3.049 ms     2.7%    46.523 ns    21.49 Mrows/s
csv/scan_long_twin             6.045 ms     5.9%    92.238 ns    10.84 Mrows/s
csv/parse_int                  3.165 ms     2.9%     6.037 ns 165.65 Mfields/s
csv/parse_float                3.061 ms     3.9%    11.676 ns  85.64 Mfields/s
```

The first version of the scanner was slower than its own byte at a time twin, and the benchmark is what caught it. A register can be tested against the delimiter, the newline and the carriage return in three instructions, but there is no packed movemask reachable from this stdlib, so a register that hits still has to be walked byte by byte to find the lane. In a file whose fields are five bytes long every register hits, which means the vector compare is added to the byte walk rather than replacing it.

The fix is that a search walks the first eight bytes one at a time and only starts testing registers once a field has proved it is longer than a word. Narrow files now scan at the same speed as the scalar scanner, 10.293 ms against 10.419 ms, and long text fields scan at twice its speed, 3.049 ms against 6.045 ms. Eight was measured rather than assumed; a whole register as the threshold gave up most of the win on the long fields and bought nothing on the short ones.

Both parsers are far cheaper than fetching the field they parse, 6.0 ns and 11.7 ns per field, which is the right shape for what comes next: once there is a reader, the cost will be in the scan and the allocation, not in turning digits into numbers.

### Known limitations

There is no `read_csv`, no schema inference and no writer. Those need a variable width string column, which is the next change.

The scanner assumes the buffer is one whole file or a block already split on a row boundary. Splitting a file into blocks for parallel scanning has to respect quoted newlines and is not done here.

Quoting is RFC 4180 only. There is no backslash escape mode and no comment character.

## [0.6.1] - 2026-08-28

Built against Mojo 1.0.0 (ed45d567).

Concat and the null handling functions. Stacking columns, series and frames, and the five things everybody does with a missing value: ask where they are, replace them from somewhere else, carry the last value forward, carry the next one back, and drop the rows.

A patch bump. Everything here is new surface and nothing that existed changed shape or behaviour.

### Added

- `firepanda.kernel.concat`: `concat_arrays` for a list of typed columns, `concat_any` for a list of erased ones, and `concat_two_any` for the two argument case that borrows rather than owns.
- `firepanda.kernel.nulls`: `is_null` and `is_not_null` in typed and erased spellings, `coalesce`, `fill_forward`, `fill_backward` with an optional run limit, and `all_valid_mask` for the intersection of several columns' validity.
- `firepanda.frame.concat`: the free functions `concat` for frames and `concat_series` for series, both re-exported from `firepanda`.
- `Series.is_null`, `Series.is_not_null`, `Series.drop_nulls`, `Series.fill_null`, `Series.fill_forward` and `Series.fill_backward`.
- `DataFrame.drop_nulls`, taking an optional subset of column names, and `DataFrame.fill_null`.
- Scalar twins in `firepanda.kernel.scalar`: `concat_scalar`, `coalesce_scalar`, `fill_scalar` and `is_null_scalar`, on the same terms as the twins already there.
- Tests: 21 in `tests/test_concat.mojo` and 32 in `tests/test_nulls.mojo`.
- Benchmarks: seven `nulls/*` rows and three `concat/*` rows.

### Changed

- `firepanda.join.pairs` no longer carries its own private concatenation. It keeps its own dtype error message, which is about key columns rather than about columns in general, and then calls `concat_two_any`. The kernel copies values a SIMD register at a time where the join's copy went element by element, so the join benchmarks came out level to slightly better: `join/indices_1000` 12.046 ms against 11.773 before, `join/inner_1000` 19.337 against 19.562, `join/many_to_many` 1.741 against 1.870.

### How it works, and what follows from it

- A fresh `Array` is zeroed and marked all present, so a part with no nulls costs concat nothing beyond the value copy. Only a part that actually has nulls pays a validity pass, and that pass runs a word at a time and skips any word that is all ones.
- The validity repair in concat runs a bit at a time rather than a word at a time, because the destination offset is a running total of the earlier parts' heights and is almost never a multiple of 64. Shifting and masking a source word into a destination that straddles two words is the faster spelling and is not written yet.
- Filling with a scalar and filling from a column are the same operation. A fallback of exactly one row broadcasts, so `fill_null` takes a `Series` in both cases rather than growing an erased scalar type that would have to carry its own dtype tag. That is also what SQL `COALESCE` already means.
- `fill_forward` and `fill_backward` are one core with a `comptime` direction rather than two loops, so the two directions cannot drift apart. A `limit` of zero means no limit, and the run counter resets at every present value rather than at the start of the column.
- Nothing here promotes. `coalesce` of an int32 column and a float64 fallback is refused, for the same reason concat and join refuse it.
- Frame concat matches columns by name, not by position, so a frame whose columns are in a different order still stacks. Stacking by position would put two different meanings in one column with nothing at the call site to catch it.
- `drop_nulls` implements only pandas' `how="any"` rule. `subset` is the control that matters and `how="all"` is a filter anybody can write in one line against `is_null`.
- There is no horizontal concat. Putting two frames side by side means deciding which row of one lines up with which row of the other, and without an index the only answer available is position. It waits until there is something to align on.

### Notes on the numbers

Measured on the reference machine, a 16 core x86_64 with 32 byte SIMD, at 1,048,576 rows over 10 repetitions. Sparse means one row in eight is null.

- `nulls/is_null` is 34.524 us on a column with no nulls and `nulls/is_null_sparse` is 380.886 us on one that has them. The 11x gap is the whole value of the word at a time scan: an all ones or all zeros validity word turns into a block store of the same answer, and only a mixed word is walked bit by bit.
- `nulls/coalesce` is 447.310 us and `nulls/coalesce_sparse` is 1.951 ms. `nulls/ffill` is 441.804 us and `nulls/ffill_sparse` is 1.955 ms. Both operations show the same 4.4x, which says the cost is the repair pass over the missing rows and not the copy that precedes it, and that the two operations are doing the same amount of work per missing row.
- `concat/eight_parts` is 1.723 ms against `concat/two_parts` at 1.694 ms for the same total height. Within 2%, so per part overhead is not measurable and the operation is memory bandwidth and nothing else.
- `concat/frame_three_columns` is 3.820 ms, a shade over twice the single column figure for three columns, so the schema walk and the name lookups cost nothing worth naming.
- `nulls/drop_nulls` is 6.885 ms, which is `is_null` plus a `filter` over three columns, and the filter is all of it.

### Known limitations

- Frame concat of three or more frames copies every column into a `List[AnyArray]` before concatenating, because the list spelling owns its parts. The two frame call, which is the common one, borrows and does not. Reference counted columns would remove the copy from both this and `DataFrame.drop_nulls`.
- `fill_forward` and `fill_backward` are serial by construction, since a filled value can depend on one arbitrarily far back. Splitting the column into blocks and resolving the carry between them is the parallel form and is not written.
- `coalesce` reads the fallback for every missing row rather than gathering the missing rows first, so a column that is mostly null pays a scattered read.

## [0.6.0] - 2026-08-26

Built against Mojo 1.0.0 (ed45d567).

Joins. All seven kinds, on one or more key columns, from `DataFrame.join` down to the row pairing underneath it.

The minor bump is for a new top level package, `firepanda.join`, and for three new methods on `DataFrame`. Nothing that existed changed shape.

### Added

- `firepanda.join`: `JoinKind` with `INNER`, `LEFT`, `RIGHT`, `OUTER`, `SEMI`, `ANTI` and `CROSS`, `JoinIndices` holding the paired row numbers, and `join_indices` producing them from two column lists and a set of keys.
- `DataFrame.join` for keys named the same on both sides, `DataFrame.join_on` for keys named differently, and `DataFrame.cross_join`.
- `firepanda.join.scalar.join_nested`, the nested loop twin, on the same terms as the twins in `firepanda/kernel` and `firepanda/hash`: it is never called in production and it is what the fast path is checked against.
- `tests/fuzz/join.mojo` and the `fuzz-join` pixi task, wired into `pixi run fuzz`. Two million cases pass in forty seconds.
- Tests: 34 in `tests/test_join.mojo`, covering each kind's row set, the null rule, the column naming rules and the row order.
- Benchmarks: nine `join/*` rows covering the pairing on its own, four of the kinds, dimension size, two keys, and the many to many case.

### How it works, and what follows from it

- The two frames are aligned by concatenating each key column with its opposite number and handing the result to `group_ordinals`. Two rows share a code exactly when they share a key tuple, whichever side they came from, and the multi-key packing and the small integer fast path come along unchanged. The cost is one extra pass over each key column plus the memory to hold the copy.
- A row whose key contains a null matches nothing, including another null. That is SQL's rule and Polars' default. pandas `merge` joins NaN keys together and this deliberately does not, because firepanda has a validity bitmap and does not need to overload a float value to mean missing. The rows are not dropped: an unmatched left row still survives a left join or an anti join.
- The result is in left row order, and within a left row, in right row order. A right join is the same operation with the sides exchanged, so it comes out in right row order. This is fixed rather than incidental, because a join whose row order moves between runs cannot be compared against another engine.
- A key column that both frames call by the same name appears once in the output and is filled from whichever side had the row. That only matters for a right or outer join, where an output row can have no left row at all, and taking the key from the left would put a null in the column the row was matched on.
- Key columns must have the same dtype on both sides. Promoting them here would mean a join silently finding fewer matches than either side expected, with nothing on screen to say why, so the cast is the caller's to write.

### Notes on the numbers

Measured on the reference machine, a 16 core x86_64 with 32 byte SIMD, at 1,048,576 fact rows against a 1,000 row dimension unless stated.

- `join/indices_1000` is 11.773 ms and `join/inner_1000` is 19.562 ms. The first is the pairing alone and the second is the whole operation, so building the output columns is the larger half at roughly 2 ms per gathered column.
- `join/inner_100k` is 30.528 ms against 19.562 ms for the same fact rows using the same thousand keys. The result is identical and only the dimension's unused rows differ. The extra 11 ms is the key alignment: a dimension spanning a hundred thousand values puts `factorize` on a 400 KB direct table where the thousand row dimension fits in 4 KB, and every one of the 1.1 million concatenated rows pays the difference.
- `join/semi` is 19.185 ms and `join/anti` is 18.723 ms, both against `join/inner_1000` at 19.562 ms. The filtering joins gather nothing from the right and stop at the first match, and they are barely cheaper, because the key alignment dominates and they pay all of it.
- `join/outer` is 30.241 ms. The extra over inner is the bitmap write per matched row and the coalesced key column, which is a two source gather rather than a straight one.
- `join/two_keys` is 28.652 ms. The second key adds a factorize, a pack and a refactorize, which is the same 9 ms it adds to a two key group by.
- `join/many_to_many` is 1.870 ms for 262,144 output rows out of two 4,096 row frames, or 7.13 ns per output row, the cheapest per row in the set because both sources are cache resident.

### Known limitations

- The right side is bucketed with a counting sort over the group ordinals, which is one array as wide as the number of distinct keys in both frames together. For a join between two large frames with high cardinality that array is the working set, and radix partitioning the probe is what `firepanda/hash/partition.mojo` exists for.
- The key alignment copies each key column. Rewriting `factorize` to take a pointer rather than an `Array` removes that copy and is the same open item `group_ordinals` already has.
- A cross join materializes the full product and nothing refuses a large one. Any threshold would be arbitrary and would be in the way of the case the operation exists for.
- Joining on a float column keys NaN to NaN and negative zero to zero, which is `key_bits` and is the same rule group by uses.

## [0.5.0] - 2026-08-26

Built against Mojo 1.0.0 (ed45d567).

The rest of group by. `std`, `var`, `median`, `quantile` and distinct count join the eight reductions from 0.4.0, which closes the group by line on the M1 issue.

The minor bump is for the five new reductions and for `AggKind` growing a `param` field to carry a delta degrees of freedom or a quantile. Anything that constructed an `AggKind` from a bare code still works and now gets the reduction's documented default rather than zero.

### Added

- `firepanda.kernel.group`: `group_var`, `group_std`, `group_median`, `group_quantile` and `group_nunique`, plus `AggKind.VAR`, `STD`, `MEDIAN`, `QUANTILE` and `NUNIQUE` so all five run through `aggregate_group` and `DataFrame.group_by` as well.
- `AggKind.var_with`, `AggKind.std_with` and `AggKind.quantile_at`, for the three reductions that take a number as well as a name.
- Tests: 24 more in `tests/test_group.mojo`, including the large value variance case that the one pass formula gets wrong.
- Benchmarks: seven `group/*` rows covering the two dispersions, the order statistics dense and sparse, the distinct count, and one erased quantile.

### Changed

- `AggKind` carries a `Float64` beside its code. `VAR` and `STD` read it as a delta degrees of freedom, `QUANTILE` and `MEDIAN` read it as the quantile, and everything else leaves it at zero. Two kinds compare equal on the code alone, so `kind == AggKind.QUANTILE` is true for the ninetieth percentile as well as for the median, which is what the dispatch chain needs.
- `tests/fuzz/kernel.mojo` rotates through all thirteen reductions rather than eight, with a fresh quantile position and a fresh degrees of freedom each time round. Its tolerance is relative above one, because a variance of a column near a million lands near 1e12 and an absolute tolerance there is asking floating point addition to be associative.

### Notes on the numbers

Measured on the reference machine, a 16 core x86_64 with 32 byte SIMD, at 1,048,576 rows and 1,000 groups.

- `group/var_sparse` is 4.413 ms against 1.609 ms for `group/mean_sparse`. Variance takes two passes over the column where a mean takes one, and the first of those two passes is the mean.
- `group/median` is 4.075 ms and `group/nunique` is 3.994 ms. Both build a slab of the non-null values grouped contiguously and sort each group's run, so they cost close to the same thing and the distinct count is the sort plus a scan for runs.
- `group/median_cardinality_10` is 18.601 ms against 4.075 ms for the same rows over a thousand groups. Ten groups means each slab run is a hundred times longer and the sort inside it is `n log n` on that length, which is the whole difference.
- `group/quantile_dispatched` is 3.792 ms, in line with the typed `group/median`, so the erased path costs nothing measurable once the reduction is this expensive.

### Known limitations

- The two dispersions read validity a bit at a time, like `mean`, `min` and the rest of the null-aware reductions. Reading a word at a time is still the open item in this file.
- `median`, `quantile` and `nunique` allocate a slab the size of the non-null values. A group by that computes several order statistics of the same column builds and sorts that slab once per reduction rather than sharing it.
- `nunique` counts by sorting rather than by hashing. That is the right call while the slab is being built anyway and it is `n log n` where a hash set would be linear.
- A float column containing NaN sorts it in an unspecified position, so a quantile over one is unspecified. Nulls are a separate thing and are excluded properly.

## [0.4.0] - 2026-08-26

Built against Mojo 1.0.0 (ed45d567).

Group by. `DataFrame.group_by` takes one or more key columns and a list of reductions and gives back one row per distinct key tuple, with pandas' answers for null keys, empty results and group ordering. This is the operation `firepanda/hash` was written for and it is the largest piece of M1.

The minor bump is for the new `firepanda.kernel.group` and `firepanda.hash.grouping` modules and for `AggSpec` in `firepanda.frame`. Nothing that existed in 0.3.0 changed shape.

### Added

- `firepanda.kernel.group`: `AggKind` and eight grouped reductions, `group_sum`, `group_mean`, `group_min`, `group_max`, `group_count`, `group_first`, `group_last` and `group_size`, plus the erased entry points `aggregate_group` and `aggregate_group_any` that a frame calls when the dtype is a runtime value. All eight share five pointer level cores, so the typed and erased spellings run the same code rather than two copies of it.
- `firepanda.hash.grouping`: `Grouping` and `group_ordinals`, which turn one or more key columns into a dense ordinal per row by composing `factorize` rather than hashing the tuple. Each additional key packs the running ordinal against the new one and refactorizes, which keeps the ordinal space bounded by the key tuples actually observed rather than by the cross product.
- `firepanda.frame`: `DataFrame.group_by`, `DataFrame.group_agg` and `DataFrame.group_count`, plus `AggSpec` to say which column, which reduction and what to call the result.
- `Factorized.into_codes`, which gives up the ordinals without copying them when the keys are not wanted.
- `firepanda.kernel.scalar.group_scalar`, the twin, which materializes each group's values into a list and reduces it with a plain loop. `tests/fuzz/kernel.mojo` now checks all eight reductions against it, one kind per case.
- Tests: 40 unit tests covering the null policy of each reduction one at a time, the multi-key ordinal combination, and the frame level behaviour of `dropna`, `sort` and output naming.
- Benchmarks: sixteen `group/*` rows, arranged so the interesting numbers are subtractions between neighbouring rows.

### Changed

- `tools/bench_compare.py` requires a fixed cost benchmark to move by an absolute margin as well as a percentage. `dispatch/call_1_row` measured 4.0 ns and 7.2 ns on two CI runs of unrelated changes, +82%, while the same two binaries measured 2.817 ns and 2.796 ns against each other on a dedicated machine. A percentage is the wrong instrument on a benchmark that reports a few nanoseconds, and the item count is what separates those from throughput rows, where an absolute margin would mask a genuine doubling.
- CI runs `tools/bench_compare.py` with a new `--advisory` flag, which prints the table and the verdicts and exits zero. `benchmarks/main.mojo` is one compilation unit and the benchmark bodies inline what they measure, so appending the sixteen `group/*` rows to the end of it changed which of the loops above them get vectorized: `array/sum_scalar` went from 386 us to 740 us and `bitmap/or_with` from 58 us to 92 us, in files this release does not touch, reproducibly to within half a percent across two runners. Every row measuring through a boundary the compiler will not inline across held still, and on the reference machine `kernel/sum_dense` and `kernel/sum_sparse` came out at 110.776 us and 110.855 us, which is the invariant those rows exist to check. The numbers are real, the attribution is not, and no gate can tell the two apart. The performance gate with teeth is the reference machine run recorded in each pull request, where the same tool runs without the flag.
- `_check_codes` in `firepanda.kernel.group` and the ordinal scan in `firepanda.hash.grouping` both use `max_of` rather than a scalar loop. A scalar scan of a million codes measured at roughly 350 us on the reference machine, which was more than the reduction it was guarding: `group/sum_dispatched` went from 960 us to 453 us and `group/frame_two_keys` from 11.1 ms to 8.8 ms.

### Notes on the numbers

Measured on the reference machine, a 16 core x86_64 with 32 byte SIMD, at 1,048,576 rows and 1,000 groups.

- `group/sum` is 348 us against 105 us for the ungrouped `kernel/sum_dense`. The 3.3x is what the scatter costs: a grouped sum reads a code, indexes an accumulator and writes it back, where an ungrouped one accumulates into a register.
- `group/sum_cardinality_10` is 374 us and `group/sum_cardinality_100k` is 805 us over the same rows and the same loop. The 2.2x is the accumulator array leaving cache, and it is the reason `firepanda/hash/partition.mojo` exists.
- `group/min` is 627 us dense and `group/min_sparse` is 1.654 ms. The difference is reading the validity bitmap a bit at a time, which `group/sum` never has to do because a null holds a zero. `mean`, `min`, `max`, `first` and `last` all pay it and all could be reading a validity word at a time instead. That is the next thing worth fixing in this file.
- `group/ordinals_one_key` is 2.534 ms of the 3.678 ms that `group/frame_one_key` takes, so on one integer key the grouping is 69% of the work and the reduction is the rest.
- `group/ordinals_two_keys` is 8.553 ms against 2.534 ms for one key, which is more than 2x because a second key costs two factorize passes rather than one plus a pass to pack them.

### Known limitations

- `std`, `var`, `median`, `quantile` and distinct count are not implemented. They need either a second pass or a buffer per group and they are the second half of the group by scope on the M1 issue.
- No joins, no IO, no strings as group keys.
- `group_ordinals` copies each key column once, because `factorize` reduces its input with `min_of` and `max_of` and those take an `Array` rather than a pointer. Rewriting that path to pointer form would remove a full column copy per key.
- The chained `df.groupby("k").sum()` spelling does not exist. `group_by` takes the keys and the reductions in one call, because the intermediate object needs either a borrow that outlives the expression or a copy of the frame, and neither is available until the plan layer at M4.

## [0.3.0] - 2026-08-26

Built against Mojo 1.0.0 (ed45d567).

The dataframe. `DataFrame` and `Series` exist, they hold real columns, they do the dozen operations that everything else is built out of, and they print themselves. This is the first release where the package does something a pandas user would recognise as the point of it.

The minor bump is for the new `firepanda.frame` package and for four new public functions in `firepanda.kernel`. Nothing that existed in 0.2.0 changed shape.

### Added

- `firepanda.frame`: `Series` and `DataFrame`, eager, positional and immutable. Column access, `select`, `drop`, `rename`, `with_column`, `cast`, `filter`, `take`, `slice`, `head`, `tail`, `argsort`, `sort_by` and `sort_values`.
- `firepanda.frame.display`: `render_table`, `render_column`, `render_value`, `format_float` and `DisplayOptions`. A frame prints as a table with a header, an integer index, right aligned cells and a shape line, and elides the middle of both axes when there is too much to print. `DataFrame.describe` reports the shape and the schema without rendering any values, which is what `write_to` used to do.
- `firepanda.kernel`: `take_any`, `filter_any`, `cast_any` and `argsort_any`, the type erased entry points a frame needs because its columns have different dtypes from each other. They share a body with the typed kernels rather than duplicating one, so the scalar twin still covers both.
- `AnyArray.slice`, which needs no dtype dispatch at all because a slice moves bytes without looking at them.
- `tools/probes/cast_matrix.mojo`, a compile budget probe for the first two sided dispatch in the package. `cast_any` instantiates 144 copies of its loop and they cost 72 KB and 1.3 seconds of compile time over a probe that dispatches over nothing.
- Tests: 61 unit tests covering the frame invariants, the error paths, the erased dispatch over all twelve dtypes, and the rendered output compared whole rather than probed at.
- Benchmarks: eleven `frame/*` rows, each paired with the kernel row underneath it so the frame layer's overhead is a number rather than an assertion.

### Changed

- Printing a `Series` or a `DataFrame` now writes the values. Both used to write a one line summary, which was a placeholder until this release.

### Notes on the numbers

- Rendering costs what it prints and not what it holds. `frame/render` measures 5.25 us on a frame of 1,024 rows and 5.43 us on the same frame at 1,048,576 rows, a thousandfold increase in height for three percent more work, because only the cells that appear are ever built.
- A null and a `NaN` print differently, as `<NA>` and `NaN`. pandas spells a null either way depending on the dtype backing the column. Every firepanda dtype is nullable through the validity bitmap and a float column can genuinely hold a `NaN`, so the two have to be distinguishable.

### Known limitations

- An operation that changes one column still copies the ones it did not touch, because a frame owns its columns outright. `frame/cast_one` measures 4.6 ms against 0.5 ms for the same cast at the kernel layer, and the difference is copying the other two columns. Sharing immutable columns by reference count is the fix.
- Fetching a column by name copies it, at 431 us on a million rows, where fetching by position borrows it at 0.9 ns. A borrowing accessor needs the column index in its return type, which a name lookup cannot supply until the plan layer resolves column references ahead of time.
- No group by, no joins, no IO.
- Strings still have a layout and no kernels, so a string column cannot be sorted or hashed, and the renderer prints `<string>` in place of a value rather than the value.

## [0.2.0] - 2026-08-26

Built against Mojo 1.0.0 (ed45d567).

Sorting. A dataframe still does not exist, but the operation that a dataframe spends the most time on outside of group by now does, and it is faster than the standard library's sort on the only comparison that can be made today.

The version is a minor bump rather than a patch because `firepanda.kernel` gained public functions. Nothing that existed in 0.1.0 changed shape.

### Added

- `firepanda.kernel.sort`: `argsort`, `argsort_into`, `argsort_multi`, `sort_rows` and `is_sorted`. A least significant digit radix sort on eight bit digits, stable, with null placement at either end and a direction per key.
- `sort_key` maps every numeric dtype and `bool` onto an unsigned integer whose ordering is the dtype's ordering, so a float is radix sorted exactly rather than approximately. Negative zero sorts below positive zero and NaN sorts above every finite value, both matching numpy.
- Every digit's histogram is counted in a single read of the keys, and a digit whose values are all identical is skipped, so an int64 column of small positive values costs two passes rather than eight.
- `firepanda/kernel/scalar.mojo`: `argsort_scalar`, an insertion sort comparing values with `<`. It does not go through `sort_key`, so the transform is checked rather than assumed.
- Tests: 35 unit tests, and the kernel fuzz harness now checks `argsort` against its twin on the permutation itself rather than on the sorted values, which makes it a stability check as well.
- Benchmarks: eight `sort/*` rows covering the pass count, null handling, multi-key sort and the standard library as a reference point.

### Known limitations

- A column that arrives already sorted is the slowest input the sort has, at 32.7 ms against 8.7 ms for a random column of the same value range and the same three radix passes. The cause is the scatter, not the pass count: sequential input visits the 256 write cursors in strict round robin and each one is evicted before it comes round again. Staging the writes through a per bucket buffer is the fix and it is not in this release.
- There is no comparison sort, so only the numeric dtypes and `bool` can be sorted. Strings arrive with the string kernels.
- Still no `DataFrame`, no `Series`, no IO.

## [0.1.0] - 2026-08-26

Built against Mojo 1.0.0 (ed45d567).

The first tagged release. There is no dataframe in it. What it contains is the layer a dataframe is built out of, plus the parts of the compute layer that sit directly on top: bitmaps, buffers, columns, the type lattice, the kernels, and the hash table that group by and join will use. It is tagged because the pieces underneath are now stable enough that changing them would be a break worth writing down, not because any of it is usable as a dataframe yet.

Install it and you get a library with no public API to speak of. The point of the tag is the version identity: the Mojo ABI is not stable within 1.x, so a build is identified by its own version and by the toolchain that produced it, and that pairing needs somewhere to start.

### Added

- The specification: twelve documents in `docs/specs/`, written against Mojo 1.0, pandas 3.0.5, Polars 1.43 and Arrow 25.0.0 as of August 2026.
- Milestone issues M0 through M11 covering the work to a defensible 1.0.
- CI: spec conformance checks, a guarded build and test matrix, wheel builds with clean-install verification, a nightly Mojo canary that files an issue on failure, a benchmark regression gate, and workflow, dependency and Scorecard auditing.
- Release: PyPI publishing through Trusted Publishing with build provenance attestations. No PyPI token exists in this repository.
- M0, the foundation layer, built against Mojo 1.0.0 (ed45d567):
  - `firepanda.bitmap`: an Arrow validity bitmap with word at a time popcount, boolean operators, ranged set, and both aligned and unaligned slicing.
  - `firepanda.buffer`: 64 byte aligned allocation and a size class pool.
  - `firepanda.array`: `Array[dt]`, the type erased `AnyArray`, `ChunkedArray`, and the StringView layout with its 16 byte inline prefix representation.
  - `firepanda.dtype`: the logical type lattice, `Schema` and `Field`, promotion that agrees with numpy on all 144 pairs, the `comptime` dtype lists, and the `dispatch` bridge from a runtime dtype tag to a compiled instantiation.
  - Tests: 90 unit tests, a ten million case bitmap fuzz against a `List[Bool]` reference, a concurrency stress harness, and a differential suite that runs against numpy and pyarrow in process.
  - Tools: a microbenchmark suite with a median and IQR report, a comparison tool that will not call anything a regression unless it clears the measured spread, and the compile time and binary size probes that make the monomorphization cost per dtype a number rather than a worry.

- The compute kernel layer, the first part of M1:
  - `firepanda.kernel`: sum, count, min, max and mean reductions; add, subtract, multiply and divide; the six comparisons; casts between any two numeric dtypes; validity masking; and take and filter.
  - A null holds a zero in the values buffer, which is what lets `sum` and `mean` run without reading the validity bitmap at all. The invariant, and the one way to break it, are written down in `firepanda/kernel/__init__.mojo`.
  - `firepanda/kernel/scalar.mojo`: a one element at a time twin of every kernel, never called in production, which is what the kernels are checked against.
  - Tests: 26 unit tests and a second fuzz harness that runs every kernel against its twin over six dtypes and four null shapes.
  - `Bitmap.slice` on an unaligned start now shifts a byte at a time instead of a bit at a time, and `Array.slice` copies its values with one memcpy. Unaligned bitmap slicing went from 1.326 to 0.051 ns per bit.

- The hash layer, the second part of M1:
  - `firepanda.hash`: `factorize`, which rewrites a column as dense group ordinals plus the keys those ordinals name. This is what group by, join, unique, value counts and the categorical dtype are all going to be built on.
  - An open addressing table with linear probing at a load factor of one half. It stores the hash rather than the key, which is exact rather than a shortcut, because the mixing function is a bijection on 64 bits.
  - Columns whose integer range is small enough skip the table entirely and index an array with the value. That route runs at 0.74 ns per row against 2.68 for a `Dict`, and it covers most of the categorical columns anyone actually has.
  - Sizing is measured, not asked for. The build watches its own group discovery rate at two checkpoints and extrapolates. Presizing to the row count instead was tried and was 1.7x slower on a ten thousand group column, because the table stops fitting in cache.
  - `firepanda/hash/scalar.mojo`: a quadratic twin to check against, and a `Dict` based factorize kept in the library so the comparison stays runnable.
  - Tests: 41 unit tests and a fuzz harness that checks the table, the partitioning and both factorize routes against the twin.

### Known limitations

- No `DataFrame`, no `Series`, no IO. See the status notice in the README.
- `factorize` loses to a `Dict` based implementation by about 1.3x on columns with a hundred or ten thousand groups, and beats it by 2.6x when every row is distinct and by 3.6x when the integer range is small enough to skip hashing. The tracking issue for M1 has the numbers and the reasoning.
- The string layout exists but no string kernels do, so a hash table keyed on strings is not possible yet.

[Unreleased]: https://github.com/tamnd/firepanda/compare/v0.6.33...HEAD
[0.6.33]: https://github.com/tamnd/firepanda/releases/tag/v0.6.33
[0.6.32]: https://github.com/tamnd/firepanda/releases/tag/v0.6.32
[0.6.31]: https://github.com/tamnd/firepanda/releases/tag/v0.6.31
[0.6.30]: https://github.com/tamnd/firepanda/releases/tag/v0.6.30
[0.6.29]: https://github.com/tamnd/firepanda/releases/tag/v0.6.29
[0.6.28]: https://github.com/tamnd/firepanda/releases/tag/v0.6.28
[0.6.27]: https://github.com/tamnd/firepanda/releases/tag/v0.6.27
[0.6.26]: https://github.com/tamnd/firepanda/releases/tag/v0.6.26
[0.6.25]: https://github.com/tamnd/firepanda/releases/tag/v0.6.25
[0.6.24]: https://github.com/tamnd/firepanda/releases/tag/v0.6.24
[0.6.23]: https://github.com/tamnd/firepanda/releases/tag/v0.6.23
[0.6.22]: https://github.com/tamnd/firepanda/releases/tag/v0.6.22
[0.6.21]: https://github.com/tamnd/firepanda/releases/tag/v0.6.21
[0.6.20]: https://github.com/tamnd/firepanda/releases/tag/v0.6.20
[0.6.19]: https://github.com/tamnd/firepanda/releases/tag/v0.6.19
[0.6.18]: https://github.com/tamnd/firepanda/releases/tag/v0.6.18
[0.6.17]: https://github.com/tamnd/firepanda/releases/tag/v0.6.17
[0.6.16]: https://github.com/tamnd/firepanda/releases/tag/v0.6.16
[0.6.15]: https://github.com/tamnd/firepanda/releases/tag/v0.6.15
[0.6.14]: https://github.com/tamnd/firepanda/releases/tag/v0.6.14
[0.6.13]: https://github.com/tamnd/firepanda/releases/tag/v0.6.13
[0.6.12]: https://github.com/tamnd/firepanda/releases/tag/v0.6.12
[0.6.11]: https://github.com/tamnd/firepanda/releases/tag/v0.6.11
[0.6.10]: https://github.com/tamnd/firepanda/releases/tag/v0.6.10
[0.6.9]: https://github.com/tamnd/firepanda/releases/tag/v0.6.9
[0.6.8]: https://github.com/tamnd/firepanda/releases/tag/v0.6.8
[0.6.7]: https://github.com/tamnd/firepanda/releases/tag/v0.6.7
[0.6.6]: https://github.com/tamnd/firepanda/releases/tag/v0.6.6
[0.6.5]: https://github.com/tamnd/firepanda/releases/tag/v0.6.5
[0.6.4]: https://github.com/tamnd/firepanda/releases/tag/v0.6.4
[0.6.3]: https://github.com/tamnd/firepanda/releases/tag/v0.6.3
[0.6.2]: https://github.com/tamnd/firepanda/releases/tag/v0.6.2
[0.6.1]: https://github.com/tamnd/firepanda/releases/tag/v0.6.1
[0.6.0]: https://github.com/tamnd/firepanda/releases/tag/v0.6.0
[0.5.0]: https://github.com/tamnd/firepanda/releases/tag/v0.5.0
[0.4.0]: https://github.com/tamnd/firepanda/releases/tag/v0.4.0
[0.3.0]: https://github.com/tamnd/firepanda/releases/tag/v0.3.0
[0.2.0]: https://github.com/tamnd/firepanda/releases/tag/v0.2.0
[0.1.0]: https://github.com/tamnd/firepanda/releases/tag/v0.1.0
