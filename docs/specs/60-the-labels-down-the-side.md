# 60. The labels down the side

## 1. A bug that only the display had

`fp.DataFrame({"k": ["p", "q"], "v": [1, 2]}).set_index("k")["v"]` printed `0` and `1` down the left where pandas prints `p` and `q`. The labels were in the column the whole time. `s.index.tolist()` gave them, `s["p"]` found the right row, `list(s.items())` paired them with the right values, and every one of the hundreds of conformance cases that reads a labelled column passed.

That combination is what makes this worth a document rather than a one line fix. A defect that only the renderer has is invisible to a test suite that reads values and visible to every person who ever looks at a frame, and printing is usually the thing somebody reached for because they already suspected something was wrong. Being told the wrong thing at that moment is worse than being told nothing.

It was filed as issue 719 and it was true of a frame as well as a column, which nobody had noticed because a frame does not print its rows from Python at all. Section 7 has that.

## 2. Why the renderer could not see the labels

`firepanda/frame/display.mojo` renders a `Schema` and a list of columns and deliberately imports neither `DataFrame` nor `Series`, which is what lets both of those call it without a cycle. `Index` is on the other side of the same line: `index.mojo` imports `render_value` and `visible` from the display module, so the display module cannot import `Index` back.

So the renderer had nothing to print except the row positions, and printing the positions was not a decision anybody made. It was what was reachable from where the code sat, and the header of the file said so in a sentence that read like a choice: there is no index name line, and the left column is always the row position.

The fix is not to break the cycle. It is to have the labels arrive already rendered, as an `IndexCells`, which is a name and one string per printed row. `Index.display_cells` builds one, and it lives in `index.mojo` where the labels are.

## 3. Only the labels on screen are ever rendered

`display_cells` takes the display options, asks `visible` which rows will be printed, and renders those. Eleven labels on a frame of any height, and a range is never materialized into an array to be read back out of.

That is the same rule the rest of the renderer already follows for values, and it is the rule that keeps printing a large frame cheap. It also decides where the ellipsis goes: the label column gets its gap in exactly the position the value column gets its gap, because both come from the same list of visible positions, so the two columns cannot drift apart.

A caller that supplies no labels at all still gets the positions. That is what every caller outside the frame layer does, including the tests and the benchmark, and it means the old behaviour is still reachable rather than deleted.

## 4. The labels are left aligned and the values are not

pandas puts the labels hard against the left edge whatever they are. A column labelled `10`, `200` and `3` prints its labels in that order down the left with no attempt to line the digits up, while the values beside them are right aligned in the usual way.

That looks like an oversight for a numeric index and it is what pandas does, so it is what this does, and `pad_right` exists beside `pad_left` for no other reason. The alternative is a rendering that is nearly pandas' and differs in one place a caller has to discover for themselves, which is worse than one that is obviously different.

## 5. The name goes in two different places

On a column the name of the level is printed on a line of its own above the listing, unpadded, so a level called `ix` puts `ix` on the first line and nothing else. On a frame the same name goes inside the table, on its own row directly under the header, in the label column with every other column left blank.

Two placements for one thing is pandas' layout rather than an idea anybody had here. The reason it makes sense is that a frame has a header row for the name to sit under and a column does not, so on a column there is nowhere to put it except above everything.

The frame's name row is padded out across the whole table rather than trimmed, so it ends in trailing spaces. pandas does that too, and matching it means a caller who diffs the two outputs sees nothing at all rather than a whitespace difference they have to think about.

An index that was never named prints no name line in either shape, and a named index prints one whether or not its labels are a range, because the name is not a property of having real labels.

## 6. A column with nothing in it still says nothing about its labels

`Series([], Name: v, dtype: int64)` is one line and has no room for a name line, and pandas leaves the name out here even when the index has one. This does the same, which falls out of the empty case returning before any of the label handling runs.

## 7. A missing label, and what a frame prints

A label that is missing prints `<NA>`, which is how every missing value prints here and is not how pandas prints it. That is the existing rule in the display header: firepanda has a validity bitmap on every dtype and a float `NaN` is a value a column can genuinely hold, so the two have to look different. Rendering pandas' spelling for a label would be a lie about what is in the frame.

Printing a frame from Python still gives the schema and the shape rather than the rows, which is document 13 section 2 and is a separate decision that this change does not touch. The table with the labels in it is what a Mojo caller sees and what `render_table` produces, and the day the Python side starts printing rows it will already be right.

## 8. What is left

`info` does not print a `MultiIndex` differently from a plain one and neither does this, for the same reason, which is that there is no MultiIndex here yet. When there is, a frame's label column becomes several columns and the name row becomes several names, and this section is where to start.

The elision itself is spelled differently from pandas' and was before this change. This prints `...` in both columns, pandas prints `..` and leaves the label column blank on a column and prints `..` in both on a frame. The gap is in the same row either way, which is the part that matters for reading the output, and the spelling is not worth changing on its own.

The footer of a long column also puts `Length:` before `Name:` where pandas puts it after. That is older than this change and is the same kind of difference, which is to say one that a caller comparing two outputs will see immediately and that costs them nothing.
