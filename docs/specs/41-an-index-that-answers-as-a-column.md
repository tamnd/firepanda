# 41. An index that answers as a column

Status: implemented. One free function across the boundary, nine members on the index built on top of it, and an argument about where the rest of the index's surface is going to come from.

## 1. The largest block of missing names on the board, and the shape of it

`firepanda-compat` scores a name at a time, and bucketing the names it cannot resolve by section puts indexing second with a hundred and seventy three of them. Reading the list rather than the count is what makes it interesting. Forty nine of those names are missing from `Index` and from `DatetimeIndex` at the same time, and because `DatetimeIndex` subclasses `Index` in `python/firepanda/_datetime.py`, every one of them is two names on the board rather than one.

The forty nine are not a coherent feature. They are `astype`, `diff`, `dropna`, `duplicated`, `factorize`, `fillna`, `isna`, `map`, `max`, `min`, `nunique`, `repeat`, `shift`, `sort_values`, `str`, `to_frame`, `to_series`, `value_counts`, `where` and a couple of dozen more, and what they have in common is that thirteen of them already exist on the series and every one of the rest is a thing a column could answer if a column could be made out of the labels. That is the observation this document is about. The index is not missing forty nine features. It is missing one bridge.

## 2. The bridge, and why it is a free function

`Index.to_series` is that bridge, and on the Mojo side it is `index_to_series` in `firepanda/py/frame.mojo`, registered on the module as `_index_to_series` rather than as a method on either type.

The reason is the import graph and not taste, and it is the reason the function directly above it in the same file gives. `isocalendar` reads a series and answers a frame, `series.mojo` cannot import `frame.mojo` because `frame.mojo` already imports it, and a method on the frame that takes a column reads backwards. This one has the same shape one level down. It reads an index and answers a series, `index.mojo` cannot import `series.mojo` because `series.mojo` already imports it for `to_index`, and a method on the series that takes an index and hands back a series of the index reads backwards in exactly the same way.

So it belongs to neither bound type, and the module holds it. Both directions of the trip now sit on opposite sides of that line, which is a little untidy on paper and is the only arrangement the imports allow: `PySeries.to_index` is a method because a series may import the index, and `index_to_series` is a function because an index may not import the series.

## 3. The labels come back twice

`pandas.Index.to_series` answers a series whose values are the labels and whose own labels are the same labels again. That reads like an accident the first time and it is not. A caller who writes `idx.to_series().sort_values()` wants the labels sorted and wants to know where each of them came from, and the only place that information can live is the index of the answer.

It also happens to be exactly what the members below need. `dropna` takes the missing labels out of the values, and the surviving values are the surviving labels, so the answer is a matter of reading the values back out under the index's own name. If the values and the labels were not the same set there would be two answers to choose between and the choice would have to be made nine times.

The `index` argument replaces the labels the answer carries and `name` replaces the name it carries, which is pandas' signature and pandas' defaults. A set of labels that is not as long as the values is refused with a tagged `value` error rather than being stretched or truncated, because a series whose labels do not line up with its values is not a thing the rest of the library is prepared to be handed.

## 4. The name that means unnamed

pandas leaves the answer unnamed when the index is unnamed. A series here is named by a `String` rather than by an optional one, so the name that means unnamed over there is the empty string over here, and that is the one divergence this door has. It is the same divergence document 21 section 7 already records for the index name read back through the boundary, and it is the reason the squeeze refusal in document 36 exists, so it is not new and it is not going to be fixed by anything smaller than a typed label on the core series.

The values are the same either way, and so is everything built on top, so the divergence is confined to one attribute of one object and is recorded rather than worked around.

## 5. Nine members, of which seven are a line

`isna`, `isnull`, `notna` and `notnull` are the column's answer read out as a list. pandas hands back a numpy array of bools and this hands back a list of them, which is the divergence `values`, `__eq__` and `isin` already have and which document 21 records once for all of them rather than four times.

`nunique` is the column's `nunique` plus one line, and the line is the interesting part of this section. The column refuses `dropna=False` outright, with a message saying that counting a missing value as one more distinct value needs the count to know it saw one and that the kernel drops them before it counts. That is true of the kernel and it is not true of the index, which is asked how many of its labels are missing often enough to keep the answer standing by. So the index adds one to the column's count when the caller asked for it and there was a missing label, and answers what pandas answers where the column it is built on still refuses.

That is a member being more capable than the thing underneath it, which is usually a smell. It is left standing because the alternative is an index that refuses a parameter it can plainly answer, and because the same two lines would lift the refusal on the column and are the obvious next step there. Nothing here blocks that, and when it happens this line goes away.

`min` and `max` are the column's reductions with the numpy arguments held at rest, which section 7 is about.

`dropna` is the only one that has to come back as an index rather than as a value or a list, and section 6 is about the one decision in it. `to_series` itself is the door.

What is worth saying about that list is how little of it is new behaviour. Seven of the nine are a kernel that was already written, reached through a bridge that did not exist, and the reason they were missing was never that anybody had to decide what they meant.

## 6. `dropna` hands back the class it was given

Every other index method in `python/firepanda/_pandas.py` that answers an index wraps the answer in `Index`, which means `DatetimeIndex.insert` gives back a plain index and loses the calendar. That is a known follow up and it is not fixed here, because fixing it means going through every one of them at once and this document is not that change.

`dropna` is written the new way regardless, wrapping in `type(self)` rather than in `Index`. Taking a missing instant out of a set of instants leaves a set of instants, the class is a property of what the labels are rather than of which method was called, and writing the new member the old way to match would have meant knowingly adding a case to the list of things that follow up has to fix.

The mixin cannot see `_wrap`, which the generated half of the class writes, so the class goes through a local name the type checker leaves alone. That is a small ugliness and it is contained in one line with a comment on it.

## 7. numpy's arguments on `min` and `max`

pandas' `Index.min` takes `axis`, `skipna`, and then `*args` and `**kwargs`, and the last two are there because numpy calls `min` on whatever it is handed with keywords of its own. pandas accepts them and checks that they say nothing.

The axis is checked by pandas' rule rather than by this library's usual one. An index has one dimension, so `None`, `0` and `-1` all name the axis it has and anything else is pandas' sentence about the number of dimensions. The usual `_axis_number` helper would have said "No axis named 1 for object type Index", which is the frame's wording for a different check, and matching pandas here costs four lines.

The catch alls are refused outright when anything is actually passed in them, rather than being dropped. A caller who wrote `out=` meant something by it, and a reduction that quietly ignores where the caller asked for the answer to be written is worse than one that says it cannot do that. pandas accepts a handful of them at their numpy defaults, so a call that passes `out=None` is refused here and accepted there. That is the narrowest divergence in the change and it is the right side to be wrong on.

## 8. What this is worth and what comes next

Nine names on the index and nine more on the datetime index, because of the subclass. That is the arithmetic that made this the slice to do rather than the row read across the columns, which is the other thing indexing is waiting for and which is blocked on two core changes at once: a common type over a set of columns that follows pandas' rule rather than the promotion rule in `firepanda/dtype/logical.mojo`, and a series name that can hold a label rather than a string.

The rest of the forty nine come the same way and mostly a line at a time. `astype`, `diff`, `shift`, `sort_values`, `argsort`, `duplicated`, `drop_duplicates`, `factorize`, `value_counts`, `map`, `where`, `fillna` and `repeat` are all a column method underneath, and the only work in each is deciding what comes back and under what name. `to_frame` needs a frame made out of a series, which the library does not have a door for yet, and that is the next real gap rather than the next line.

## 9. What is deliberately not here

`to_frame`, for the reason just given. The lookup family, `asof`, `asof_locs`, `get_indexer_for`, `get_indexer_non_unique` and `join`, which are not column methods and need the index's own machinery. The level family, `droplevel`, `get_level_values`, `sortlevel` and `set_names`, which are the MultiIndex and are a milestone of their own. And `groupby` on an index, which answers a dict of labels to positions and is a different thing from the frame's groupby despite the name.
