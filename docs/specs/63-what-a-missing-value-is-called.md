# What a missing value is called

## 1. The last thing left in the printing group

Documents 60, 61 and 62 took the renderer from printing the wrong labels to printing what pandas prints, cell for cell and space for space, and each of them ended by naming the same two differences as still outstanding. One is that a text column's type is spelled `string` here and `str` there, which is `engine/dtype-spelling` and was settled before any of this started. The other is that a cell with nothing in it prints `<NA>` here and, most of the time, `NaN` there.

This document settles the second one, and it settles it by keeping the difference rather than removing it. That needs an argument rather than an assertion, because everything else in the printing group was a bug that got fixed, and a reader who has followed the group this far is entitled to ask why this one is different.

## 2. pandas has four spellings and one of them is ours

The first thing to say is that pandas does not have a spelling for a missing value. It has four, and which one you get depends on the dtype.

```
float64             1    NaN
str                 1    NaN
object              1    None
datetime64[s]       1    NaT
timedelta64[s]      1    NaT
Int64               1    <NA>
boolean             1    <NA>
category            1    NaN
float64[pyarrow]    1    <NA>
```

Those are measured, one column each, a value in row zero and nothing in row one. Four spellings over nine types, and the split is not arbitrary. `NaN` is what you get when the column is a numpy array of floats and the missing value is literally the floating point NaN, because there is nowhere else in a float array to record absence. `NaT` is the same trick for a timestamp, where the sentinel is a particular integer. `None` is what you get when the column is an array of Python objects and the sentinel is the Python object. And `<NA>` is what you get when the column is one of pandas' nullable types, where absence is recorded in a mask beside the values rather than smuggled into them.

Which means `<NA>` is not this library inventing a spelling. It is pandas' own spelling for a column that keeps its missing values in a mask, and every firepanda column keeps its missing values in a mask, because every firepanda column is an Arrow array with a validity bitmap. The bottom line of that table is the closest thing pandas has to a firepanda column, and it prints `<NA>`.

## 3. The difference is not about printing

A caller who only ever looks at output could reasonably ask for the spelling to be changed and nothing else, since the values underneath are the same either way. They are not the same either way, and the printing is the end of the difference rather than the whole of it.

Take the corpus frame `float64_half_null`, whose float column holds a genuine NaN in row zero and a missing value in row one. Read the first three rows out as a list and pandas gives `[nan, nan, -inf]` while firepanda gives `[nan, None, -inf]`. pandas cannot tell you which of the first two rows was a NaN somebody computed and which was a hole in the data, because by the time the column exists they are the same float. firepanda can, because they are not the same thing in memory: one is a value and the other is the absence of one.

Printing them differently is that fact reaching the screen. Printing them the same would not make the library more like pandas, it would make the rendering say something about the column that is not true, which is the one thing a renderer must not do. Document 62 is eleven sections of reading a layout off a running pandas rather than deciding what it ought to be, so this is not a preference for our own taste over compatibility. It is the same rule pointed at the value rather than at the layout.

## 4. What it costs

The cost is width, and it lands on anybody comparing two renderings. `<NA>` is four characters and `NaN` is three, so a column with a missing value in it is one character wider here than there, and because the width of a column is the width of its widest cell, every value in that column moves one place. Two otherwise identical renderings diff on every line rather than on the line with the gap in it.

That is worse than it sounds and it is why this cannot just be waved through as a cosmetic difference. It is also why the conformance cases in the printing group were written the way they were: `basics/repr-text` reads only the one row corpus frame, because the two row frame has a missing value in its text column and comparing the rendering there would have been comparing this rather than the layout the case is about. With the difference registered, those cases can stop stepping around it, and a case that covers a null bearing frame is scored as divergent with a named reason rather than quietly left out.

## 5. What it does not change

Nothing about asking. `isna` gives the same mask on both sides, including on a genuine NaN, since pandas and firepanda both count a NaN as missing when asked directly. `count`, `dropna`, `fillna` and every reduction that skips missing values agree. A missing value compares as missing in firepanda and as False in pandas, but that is `engine/comparison-null` and predates all of this by a month.

So the registered difference is narrow: it is how the gap is written when it is written out, plus the one place where reading a column back as Python objects shows that a hole and a NaN were two different things all along.

## 6. One spelling rather than four

pandas varies the spelling by dtype and firepanda does not. A missing timestamp prints `NaT` there and `<NA>` here, and a missing entry in an object column prints `None` there and has no equivalent here at all, because there is no object column in this library and document 06 says why.

Having one spelling is the point rather than an oversight. pandas' four spellings are four different implementation strategies showing through: they tell you what the column is made of, not what happened to the row. A user who has to know that a hole in a float column is called one thing and a hole in a timestamp column is called another is being asked to learn the storage. In firepanda there is one way a value goes missing, so there is one word for it, and that word is already the one pandas uses for the columns that work the way ours do.

## 7. What is left

The registry entry is `engine/missing-spelling` and it is expected to differ rather than to be fixed. The two things it does not cover are worth naming. A missing value read out as a Python object is `None` here and `nan` or `NaT` there, which is the same fact one layer down and is registered with it. And `info` prints a missing label as `None` where pandas prints `nan`, which document 59 already named and which is this difference reaching a second renderer.

Nothing here is affected by the float formatting in document 62, because a missing cell is not a number for any of that: it does not block the stripping of trailing zeros, it is not stripped, and it does not count towards the length that decides whether a column goes to scientific notation.
