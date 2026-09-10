# 24. A column that arrives as codes

A dictionary encoded column is the commonest thing in an Arrow file that firepanda could not read. Every pandas `Categorical` becomes one on the way out, every Parquet file with a low cardinality string column has one in it, and a `pyarrow.Table` that has been concatenated or read back off disk carries several of them in a row. Until now all of it stopped at the door with a message about the format string, which is a strange thing to be told about a column type the library already has.

## What the interface says a dictionary is

The C Data Interface does not give a dictionary its own format string. The field keeps the format of its index, which is almost always `i` for int32, and hangs the value type off a separate `dictionary` member of the schema. The categories themselves are on the matching member of the array, and there is one set per array, which means one set per batch of a stream rather than one per column.

Two consequences fall out of that and they shape everything below.

The value type is only in the schema. A stream hands out the schema once and then releases it, and then hands out batches, so a consumer that wants to know what a dictionary holds has to read it before the schema goes away and remember it. That is the plan.

The categories are only on the array. So they arrive once per batch, and a stream of six batches hands over six sets, which may or may not be the same six.

## Reading it

`dictionary_plan` walks the struct schema once and records, per encoded field, which column it is, the format of its values, and whether the ordered flag is set. It runs before the schema is released. `frame_layout` continues to record `i` for that column, which is right: the codes are an int32 buffer and are copied by exactly the path any other int32 column takes, with no special case anywhere in the copying.

`batch_categories` then reads one batch's categories, and `assemble` already knew how to attach them, because `attach_dictionary` was written when the CSV reader learned about categories and is untouched by this.

The one thing that needed care is that `struct_columns` clears `dictionary` on every window it hands out. The window is a view of the producer's child array and the codes go down the ordinary path, so anything left in that member would reach the column importer, which refuses it and is right to: a set of codes with no categories beside it is not a column anybody can read. The categories go around by the other route and meet the codes again in `assemble`.

## When the batches disagree

Arrow permits a producer to replace a dictionary part way through a stream. It is legal and it is rare, and it means the code 0 in the fourth batch is a different word from the code 0 in the third.

Firepanda holds one set of categories per column, so a stream that does this cannot be read as it stands. There are two honest answers and only one of them is cheap. Unifying the dictionaries means rewriting every code in every batch that arrived before the one that changed, and a caller who did not ask for that would be paying for it silently on every read of every file, including the overwhelming majority where nothing changes. So the answer is a refusal, and the refusal names the column, says which shape it is complaining about, and names the fix, because there is one: pyarrow will do the unification on request with `table.unify_dictionaries()`.

That makes it a `NotImplementedError` rather than a `ValueError` under document 14's rule. The data is fine, the producer has no bug, Arrow allows the thing it did, and firepanda has not written the case. That is a gap and it says so.

Categories that are not text are the same shape of answer for a different reason. Firepanda holds categories in a `StringArray` and there is no dictionary of numbers to put anywhere. The message names the column and the dictionary's format rather than the field's, since the field's format is an index type and a reader who went and looked at it would find nothing wrong.

## What can be tested from where

The values are checked in Mojo, in `tests/test_arrow_stream.mojo`, because that is the only place the column can be opened up: an imported categorical cannot be exported yet, and the Python series has no `codes` or `categories` on it. Firepanda's own exporter cannot produce a dictionary encoded column either, so the producer in those tests is built by splicing two exported frames against each other, which is exactly the shape the interface describes.

The producers are checked in Python, in `python/tests/test_arrow_import.py`, with pyarrow and pandas, because a test built against my own reading of the specification is a test of the reading that produced the code. What those two libraries hand over is what the protocol actually is. From there the shape and the refusals can be asserted and the values cannot, which is why both halves are here and neither stands in for the other.

## What this does not do

Export. `__arrow_c_array__` on a frame with a category column in it still says there is no format string for one, so a categorical can be read in and not written back out. That asymmetry is worth naming rather than leaving to be discovered, and it is the next piece of this rather than a decision.

There is also still no `.cat` namespace and no `codes` or `categories` on the Python series, so what a caller can do with an imported categorical from Python is currently to see that it has arrived. The point of doing the import first is that everything else in that list was unreachable while the door was shut.
