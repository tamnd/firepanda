# An index of instants

`pandas.DatetimeIndex` is the fourth thing an ordinary pandas program holds, after a frame, a column and an index, and it is the first one that is a kind of another one rather than a new thing. Document 21 built the flat `Index` and document 22 built the `dt` accessor, and this is the work that notices they are the same work seen from two sides: an index of instants is an index whose labels happen to be a column of timestamps, and a calendar field of those labels is the accessor reading a column it did not know it had.

Almost nothing new was written for it. One kernel line changed, three bindings were added, and the rest is four hundred lines of Python that spell the pandas surface over parts that already existed. That is the point of the document as much as the type is.

## 1. Why it is a subclass and not a fourth bound type

The tempting shape is a second Mojo struct, `PyDatetimeIndex`, carrying a timestamp column and its own methods. It is the wrong shape, and the reason is worth writing down because the same question will come back for `TimedeltaIndex`, `PeriodIndex` and `CategoricalIndex`.

Every set operation, every lookup, every slice bound, every sort and every rename on a `DatetimeIndex` is the flat index's own behaviour and has nothing to do with the calendar. A timestamp column is int64 underneath, and an index over int64 already sorts, hashes, compares, deduplicates and searches by exactly the rules an index of instants wants. A second struct would be the first struct plus nineteen calendar fields, and the nineteen fields are the part that already exists somewhere else.

So the core does not know this type is here at all. `firepanda/frame/index.mojo` holds an `AnyArray` with whatever `LogicalType` it was handed, and it was already correct for timestamps before anybody asked it to be. `DatetimeIndex` is a Python subclass of `Index` with `__slots__ = ()`, holding the same `_inner`, and what makes it a datetime index is what its labels are rather than what its class is.

## 2. The three bindings, and why each one lives where it does

Three bindings were needed and the import graph decided which struct each went on.

`PySeries.to_index` makes the values of a column into the labels of an index, which is the direction `pandas.Index(series)` goes. It is on `PySeries` because `series.mojo` already imports `PyIndex`, so building an index inside a series method costs nothing. The other direction, an index handing back a series, would need `index.mojo` to import `PySeries`, which is a cycle, and that is why `isocalendar` is in section 6 rather than in the type.

`PyIndex.temporal_part` reads one calendar or clock part of the labels and answers another index. `PyIndex.temporal_word` reads the one part that is a word rather than a number, which is the unit and the zone. Both are on `PyIndex`, and both are only possible because `firepanda/py/temporal.mojo` imports core modules and nothing else, so an index can borrow the accessor's door without a cycle in either direction.

Each of the two materialises the labels into a `Series` and hands them to `temporal.mojo`'s existing `part` and `word`. A range index has no labels to materialise, so it makes them, which is the one place this costs anything, and a range index of instants is not a thing anybody builds.

## 3. One table of field names and one parser of frequencies

Nineteen calendar fields, four frequency spellings and a rounding rule are a lot of names, and the failure mode for a second copy of them is not that it is wrong on the day it is written. It is that `s.dt.is_quarter_start` and `index.is_quarter_start` drift apart three months later when one of them is fixed.

So there is one table, in `firepanda/py/temporal.mojo`, and `temporal_part` takes the field name as a string and looks it up there. `DatetimeIndex.is_quarter_start` and `Series.dt.is_quarter_start` reach the same kernel through two doors and cannot disagree. The same is true of `floor`, `ceil` and `round`, which means the answer to what `min` means as a frequency is written once.

The one thing the index side has to do for itself is resolve a spelling. pandas spells the day of the week three ways, `dayofweek`, `day_of_week` and `weekday`, and the day of the year two, and the core keeps one name for each because a kernel does not need three. The property resolves the pandas spelling to the kernel spelling before it calls, which is exactly what `tools/bindings.py` does when it generates the accessor, and both of them do it in the layer that is about the pandas surface rather than in the layer that is about the arithmetic.

## 4. What comes back from what

pandas is consistent about this and it is worth stating because it decides the return type of twenty six members. A member that answers something about an instant answers a plain `Index`, because the year of an instant is a number. A member that answers an instant answers a `DatetimeIndex`, because the answer is still a set of labels with a calendar on them.

So `year`, `quarter`, `dayofweek`, `date`, `day_name` and `strftime` hand back an `Index`, and `normalize`, `floor`, `ceil`, `round`, `as_unit`, `tz_localize` and `tz_convert` hand back a `DatetimeIndex`. That is two private helpers, `_part` and `_moved`, differing only in which class they wrap the result in.

There is a gap here that this work does not close. The generated members on `Index` hard code `Index._wrap`, so `sort_values` on a `DatetimeIndex` answers an `Index` where pandas answers a `DatetimeIndex`. Everything in it is right and the labels are still instants, so the calendar is one constructor call away, but the class is wrong and a caller who chains will notice. Making the generator wrap in `type(self)` is a small change to `tools/bindings.py` and is left for the slice that does it deliberately, since it changes the return type of thirty odd members at once.

## 5. Three answers that are not pandas' answer

The first is that an index of instants lists and prints as the whole numbers it stores. `DatetimeIndex(...).floor("D").tolist()` gives `1577836800000000` where pandas gives `Timestamp('2020-01-01 00:00:00')`. This is issue #348 and it is not new here, since a column of instants already does it, but it is a great deal more visible on an index because an index is a thing people look at. The tests compare through `strftime` rather than through the labels for that reason, and the comparison is better for it, since it does not depend on which resolution either side happens to hold.

The second is what a calendar field answers for a label that is not an instant. Seven of the fields answer yes or no, and pandas hands those back as a numpy boolean array, which has no missing value in it, so a `NaT` reads as `False` and cannot be told apart from a real instant that is not the first of its month. firepanda answers a missing value, as the accessor already does. That is the better answer to the question that was asked and it is a difference a test has to know about.

The third is small and is about spelling. A fixed offset zone reads back as `+09:00` here and as `UTC+09:00` in pandas. A named zone reads back as neither, because attaching one needs a transition table that firepanda does not have yet, which is issue #349, and that is why the tests use fixed offsets.

## 6. What is deliberately absent

Sixty eight of the hundred and forty four pandas members are not here. The rule from document 07 is that a name must not resolve and then refuse, so a member is absent until the thing underneath it exists, and these are the things underneath.

`freq`, `freqstr`, `inferred_freq` and `resolution` need frequency inference, which is looking at the gaps between labels and deciding whether they are all the same one, plus a vocabulary for saying so. `time` and `timetz` need a time of day column type, which Arrow has and firepanda has not bound. `to_period` needs a period type, which is a span rather than an instant and is its own piece of work. `to_pydatetime` and `to_julian_date` need a column of Python objects and a calendar conversion respectively. `isocalendar` needs an index to hand back a frame, and the kernel for it exists and is already reachable from the accessor, but the binding would make `index.mojo` import `PySeries` and that is the cycle section 2 describes.

`snap`, `indexer_at_time` and `indexer_between_time` are real work with no blocker, and they are next rather than absent on principle. `mean` and `std` over instants need the reductions to know that the average of two timestamps is a timestamp.

The remaining fifty three are not about the calendar at all. They are `Index` members this type would inherit the day the flat index has them, which is issue #154, and they are the reason the two workstreams were sequenced together: every name finished there arrives here for nothing.

## 7. What it is worth

The conformance board carries one resolution case for every pandas member of a name and one signature case for every callable one. Until this work `firepanda.DatetimeIndex` did not exist, so every one of those cases failed at L0, which is the level that asks only whether the name resolves, and none of them could be scored any further.

Seventy six of the hundred and forty four members now resolve, with the right kind for every one of them, and four signature mismatches, all four of which are inherited gaps in `Index` itself and already appear on the board under their own names. That is the largest single step available on the board at the time of writing, and it cost one kernel line and three bindings.

## 8. One kernel line

The line is in `numbers_to_timestamps`, and it is that a column with no rows in it reads as an empty column of instants whatever its type says. Without it `DatetimeIndex([])` is refused, because an empty list has nothing in it to look at and infers as a column of floats, and the parser then complains that floats are not counts. pandas builds an empty `DatetimeIndex` out of the same empty list. There is nothing there to relabel and an empty column carries no evidence of what it was going to hold, so reading it as what the caller asked for is both the friendlier answer and the correct one.
