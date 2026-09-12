# 58. How much memory a frame is using

## 1. Two libraries measuring two different things

`Series.nbytes` and `memory_usage` on both classes answer a number of bytes, and the number firepanda answers is not the number pandas answers and cannot be made to be.

firepanda counts the Arrow buffers the data is actually stored in. A column of three int64 values is twenty four bytes of values and one byte of validity bitmap, and a column of three strings is three sixteen byte views plus whatever payload the strings that did not fit inside their views need. pandas counts the size of the numpy representation. A column of three strings is twenty four bytes of pointers there, plus what pandas 3 has started adding for the string data itself, and the labels of a frame that never declared any are one hundred and thirty two bytes, which is the size of the three Python integers a `RangeIndex` holds.

Neither number is wrong. They are answers to two different questions, and the reason this document exists rather than a one line changelog entry is that the two questions have the same name. A user who reads `df.memory_usage().sum()` in a pandas tutorial and runs it here gets a smaller number and will assume one of the two libraries is lying. Document 06 has it registered as a divergence and the compat registry carries the entry, and section 7 below says what a caller should do about it.

## 2. What `nbytes` counts

Everything the column has a buffer for and nothing else. The values, the validity bitmap at one byte per eight rows rather than at whatever the allocator rounded up to, and for text the views and the payload. For a dictionary column it is the codes and the categories both, which is the number that makes the type worth having: a million rows over four categories is a megabyte of codes and a few dozen bytes of text, and reporting either half alone would make the saving look imaginary or free.

The bitmap is counted even when nothing is missing. Arrow allows a column with no nulls to have no validity buffer at all and firepanda allocates one anyway, so a three row integer column is twenty five bytes rather than twenty four. That byte is really allocated and reporting it is honest. It is also why there is no frame anywhere in the corpus where the two libraries agree exactly, which is worth knowing before anybody goes looking for one.

An empty column is zero. There is no per object overhead in the number because there is no object, which is the other half of what makes this measurement different from pandas'.

## 3. `memory_usage` is `nbytes` per column plus the index

On a frame it answers a column of byte counts labelled by column name, with the index first under the label `Index`. That is pandas' shape down to the capital letter, including the part that reads like a bug: a frame with a column genuinely called `Index` produces two rows with the same label. pandas does that too and it is left alone, because a caller writing `usage["Index"]` on such a frame has asked a question with no right answer and hiding that would be worse than showing it.

On a column it answers one number rather than a column of one. The frame's version answers a column because a frame has several things to report and a column has one, and the same name on both classes is right for the same reason `axes` is: somebody adding up the memory of whatever they happen to be holding writes one line.

The numbers under it are not a second measurement. Each column's row is exactly what that column's `nbytes` answers and the index row is exactly what `Index.nbytes` answers, so there is one place where the counting is done and two places where it is reported.

## 4. The default includes the index and `nbytes` never does

`s.memory_usage()` with nothing passed counts the labels. `s.nbytes` never counts them. That is the difference between the two members and neither name says so.

It is pandas' rule and it is the opposite of what most readers would guess, since `memory_usage` sounds like the total and `nbytes` sounds like the raw one. It is also the reason both members exist rather than one of them being enough, and on a frame where the index was never declared the two agree exactly, which is the case most people will meet first and which will teach them the wrong thing. There is a test that says this out loud on a frame with a real index, because that is the only place the difference is visible.

## 5. `index=False` drops one row and changes no number

A column does not get larger or smaller depending on whether the labels beside it were counted, so the parameter removes the first row of the answer and leaves the rest alone. There is a test asserting exactly that, comparing the whole list against the same list with its head cut off, because an implementation that recomputed the columns under the flag would be free to get one of them subtly different.

## 6. `deep` is accepted and changes nothing

In pandas `deep=True` means to go and measure the Python objects an object column points at rather than the pointers pointing at them. There are no object columns here, and document 06 has the refusal of them registered as one of the genuine omissions, so every number this library answers is already the deep one.

Accepting the parameter and ignoring it is the right call rather than the lazy one. Refusing it would tell a caller that something is unavailable, when what is actually unavailable is the shallow answer, and a caller who wrote `deep=True` wanted the accurate number and is getting it. The docstring says this rather than leaving the parameter looking unimplemented.

Both flags go through the same boolean check every other flag in the library reads, so a `1` is refused the way pandas refuses it, a `None` is accepted and means False, and a numpy boolean is accepted.

## 7. What a caller should actually do with the difference

Compare like with like. `df.memory_usage().sum()` here is close to the size of the same data in a Parquet file with no compression and close to what the process actually grows by when the frame is loaded. In pandas it is close to neither of those for a frame with text in it.

A caller who needs pandas' number has `to_pandas().memory_usage()`, which costs a conversion and gives the number their tutorial predicted. A caller who wants to know whether their data fits in memory wants this one.

## 8. What is refused and what is left

Nothing here refuses anything except a flag that is not a flag.

`DataFrame.nbytes` is not here and is not a gap, because it does not exist in pandas 3 either. That was checked against a running pandas rather than assumed, and it is worth recording because the member reads like an obvious pair with `Series.nbytes` and somebody will eventually file it as missing.

`info` is left. It reports this kind of thing in a shape a person reads rather than a shape a program reads, it prints rather than answering, and it needs the column count and the null counts and the dtypes laid out in a fixed width table. Every number it needs now exists, so what is left of it is formatting.

`ChunkedArray.nbytes` adds its chunks up rather than reading one of them, which is right and is currently untestable from Python, because every column a user can build here has exactly one chunk. It will matter the first time a Parquet file with several row groups is read without being flattened, and it is written now so that the answer does not silently become one row group's worth on that day.
