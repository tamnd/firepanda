# A column with no name

## 1. Two states that had been one

pandas tells a column with no name apart from a column called nothing. `pd.Series([1]).name` is `None` and `pd.Series([1], name="").name` is `""`, and the difference is not a detail of the attribute. `to_frame` calls the first one's column `0` and the second one's column `""`. `reset_index` does the same. An operation between two columns that disagree on a name lands on `None` rather than on the empty string, and a frame reduction, which is about none of the columns it read, hands back a column with no name at all.

This library held a name as a Mojo `String`. A `String` has no absent value, so the empty string was doing both jobs, and every column that had never been given a name reported `""`. The two states were one state and pandas has told them apart all along.

## 2. How it was found, which was not by looking for it

The slice that added `prod`, `product`, `any` and `all` wrote four conformance cases for the per column forms on a frame, and all four failed the same way: `1 differences: name '', expected None`. Nothing about the reductions was wrong. The numbers agreed, the labels agreed, and what disagreed was how the answer carried the fact that it had no name.

That is worth recording as a shape rather than as an incident. A conformance board that compares whole objects finds differences in the parts nobody was working on, and the right response is to name the difference rather than to relax the comparison. Registering it as a divergence would have blessed a fidelity bug. Adding a `name` relaxation would have opened the closed set in `fpcompat/compare.py` for the same reason. The four cases were dropped with a comment saying what blocked them, and the block is this document.

## 3. The fix is the one the index already had

`Index.name` has been `Optional[String]` since it was written, because an index level with no name is ordinary and the type said so from the start. `Series.name` is `Optional[String]` now for the same reason, and `_shared_name`, which decides what a result of two columns is called, collapsed into the one `Index` already had: two absences agree on being absent, so the result has no name either, which is what pandas answers.

Most of the change is mechanical. A series' name is read in about sixty places in `firepanda/frame/series.mojo` and almost every one of them passes it straight into a result, so those became an explicit copy of an optional rather than of a string. The places that had to decide something are few and they are the boundaries.

## 4. A frame column is a schema field and a field has a name

This is the one real constraint. A `DataFrame` keeps its column names in a schema, a schema is a list of fields, and a field's name is a `String`. So a column with no name has to become something when it is put into a frame, and the something is the empty string.

`Series.column_name` says that in one place and returns the empty string for an absence. Nothing is lost on the frame's side, because the frame owns the name from then on and nobody asks a frame whether the column it calls `""` used to be called nothing instead.

It is lost on the way back, and that is where the care goes. Seven members on a column are implemented by putting the column into a frame of one, running the frame's method, and taking the column out again: `head`, `sort_values`, `nlargest`, `drop_duplicates` and the rest all go through `_through`. Every one of them would have come back named `""` if the round trip were left alone, so `_through` renames the answer to what the column was called, absence included. `python/tests/test_series_name.py` runs seven of them against pandas in one parametrized test, because the failure this catches is a member that forgot.

The tempting shortcut is to put the absence back by asking whether the name came out empty. That is the bug this whole document is about, written a second time in a smaller place, and it would be wrong for exactly the column pandas is careful about.

## 5. What the boundary carries

`PySeries.label` answers `None` rather than a string when the column has no name, which is what `PyIndex.label` has always done. `PySeries.relabel` takes `None` and means it, so `rename(None)` clears the name and `rename("")` sets an empty one, and those are now two different calls. The constructor takes `str | None` and `None` means no name, which also means that `firepanda.Series(another_series)` keeps the source's name while `firepanda.Series(another_series, name="")` overwrites it with the empty one.

The frame reduction path in `firepanda/py/frame.mojo` used to set the answer's name to the empty string on the way out and now leaves it absent, with a line saying why: one value per column is about none of the columns, so there is no name it could honestly carry.

## 6. What pandas does that this still does not

pandas names the column of an unnamed series with the integer `0`. This names it with the string `"0"`. That is not about the optional at all, it is that column labels are strings in this library and a label in pandas is any hashable, and it is a wider difference than one attribute. The test compares what the labels read as rather than what they are, and says so.

A pandas name can also be any hashable rather than only a string, so `pd.Series([1], name=42).name` is the integer. The boundary here takes a string or an absence. That is the same divergence `Index` already has and document 21 already records.

## 7. The tests that changed, which is the interesting part

Six tests failed when the name became optional and every one of them was asserting the old conflation on purpose. `test_a_result_keeps_a_name_only_when_both_sides_agree_on_it` had a docstring explaining that the nearest thing to `None` this library could say was `""`. The module docstring in `python/tests/test_arith.py` listed it as the third of three known divergences, with a note that the day one of them was fixed this file would be one of the places that had to be edited to say so. That is what a divergence written down honestly is for, and it worked: the file said where to look and the count went from three to two.

`test_a_series_named_with_the_empty_string_names_nothing` in `python/tests/test_index.py` is the one that reversed rather than moved. It asserted that a column called `""` gives an index with no level name, which was the correct reading of the old convention and is the wrong answer now, so it is named after the distinction instead.

## 8. What is not here

The grouped forms of `any`, `all` and `prod`, which the previous slice left and this one does not touch.

The integer column label for an unnamed series' frame, for the reason in section 6.

A name that is not a string.
