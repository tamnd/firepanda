# 25. Writing a category back out

Document 24 taught the import to read a dictionary encoded column. This is the other direction, and it is the half that matters more than it sounds, because a frame that holds a column it cannot hand to anybody is worse than a frame that could not read one in the first place. A caller who reads a Parquet file and gets a category column they did not choose to have would then fail at the far end of whatever they were doing rather than at the near end.

## Three pieces, and the first is where the surprise is

The format string for a dictionary type is the format of its index. Not of its values. `format_for` on a dictionary over int8 codes answers `c`, which is int8, and says nothing about the categories being text. That is what the C Data Interface asks for: the field carries the layout of the buffer it actually has, which is a buffer of codes, and the fact that it is a dictionary at all is carried by a second schema hanging off the field's `dictionary` member. So `format_for` cannot describe a dictionary on its own and does not try. It writes the index format and leaves the rest to the caller, which is `export_schema`, because the value type is text by construction and the categories themselves are on the column rather than on the type.

The second piece is that both hung structures are allocations somebody has to free. The exported schema and the exported array each get their own `malloc` for the thing they hang, and the release callbacks find it through the member itself rather than through the box. That is deliberate. Every box in `arrow_export.mojo` keeps the shape it already had, and there are four of them, two for schemas and two for arrays, so a slot in each would have been four edits to reach one behaviour. Reading it off the member is one helper called from three release callbacks.

Getting that wrong is the kind of bug nothing fails over. There is no wrong answer, no traceback and no failing test, just one schema and one array left behind on every export of every category column, on a path that a program reading Parquet in a loop takes once per file.

The third piece is the ordered flag, which is one bit and is the difference between `a < b` answering and refusing. It goes on the field's flags next to the nullable bit.

## The one copy

The categories are copied on the way out. Nothing else in that file copies anything except a bool column's bit packing, so this is worth saying out loud rather than leaving in the diff.

The alternative is a borrow, and a borrow needs a keep alive naming whatever owns the column. The two array exporters disagree about what that is: one owns the column outright, having taken it by value, and the other holds a share of a frame that Python is still using. Threading a second keep alive through both, to borrow a buffer that is as long as the cardinality rather than as long as the column, buys a copy of the small side of the data by the whole point of the encoding. A column of ten million rows over four categories copies four strings.

So it is bought rather than conceded, and the reason is written next to the code that does it.

## A release that frees the dictionary caught a test being a bad producer

Worth recording, because it is the sort of thing that looks like a regression and is not.

The dictionary test producer in `tests/test_arrow_stream.mojo`, written for document 24, hung the categories by pointing the field at a child of a second exported frame. It borrowed. That was fine while nothing freed a dictionary, and the moment the release callbacks learned to free one, the whole file died in `malloc` on a double free.

The interface is explicit that a release callback releases the dictionary along with everything else, so the new behaviour is the correct one and the test producer was the non conforming half all along. It had been leaking nothing only because nobody was freeing anything. The fix is that the producer now allocates the two hung structures and hands them over, which is what a real producer does and is the same thing the exporter does three files away.

The general shape: a test that builds a structure by hand is a test of somebody's reading of the specification, and it can be wrong in ways the code under test cannot catch until the code under test grows a behaviour that depends on it.

## What a round trip has to keep

Three things, and the last one is the one that gets lost.

The values, obviously. The ordered flag, because a column that comes back unordered looks identical in every printed form and behaves differently. And the categories nobody used.

That last one is the test worth having. A column whose categories are `low`, `high` and `unused`, and whose rows never say `unused`, has to come back with three categories rather than two. An implementation that round tripped through the values instead of through the codes would pass every assertion about rows and quietly drop it, and the difference would show up later in a `value_counts` or a groupby over the column, a long way from anything that mentions Arrow.

## What this makes possible

The import tests could assert that a categorical had arrived and could not assert what was in it, because there was no way to read one back from Python. With the export in place there is, and the round trip is the assertion.

It is still not the same as a `cat` namespace. Working with a categorical from Python, `codes`, `categories`, `add_categories` and the rest, is a separate piece of work and is what the board is now asking for. This one is about not producing a frame that cannot leave the building.
