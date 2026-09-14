# The place in front of a value

## 1. A whitespace difference is still a difference

Document 60 fixed the labels a column prints down its side and left one thing behind, which is that the values beside those labels sat one place to the right of where pandas puts them whenever the column held a negative number. `pd.DataFrame({"k": [1, 2], "b": [1.5, -0.5]}).set_index("k")["b"]` prints `1    1.5` over `2   -0.5` in pandas and printed `1     1.5` over `2    -0.5` here. Nothing is wrong with either rendering read on its own. The values line up with each other in both, the labels are in the right place in both, and no user would ever look at one and say it is broken.

It is worth fixing anyway, and the reason is what a person does with two renderings. The only reason to care whether this library prints what pandas prints is so that somebody can put the two side by side, or diff them, or paste one into a bug report about the other, and see either nothing at all or something that means something. A difference on every numeric column that holds a negative number is neither of those. It is noise that has to be looked at once and then dismissed every single time, and once a reader has learned to dismiss whitespace differences they will dismiss the real one too.

The other reason is narrower and sharper. The five conformance cases added with document 60 all read one field out of a line rather than comparing the line, purely to route around this. That is a weaker assertion than it looks: a case that reads the first whitespace separated token of each line is not checking the layout at all, so it would have passed just as happily on a rendering with the columns in the wrong order. Once the padding matches, those cases can compare whole renderings, which is the assertion that would have caught the bug in document 60 on its own without anybody having to think of it.

## 2. Every value is written one place in

The rule underneath all of this is a single one, and once it is stated the rest follows. pandas writes every value one place in from the separator, and a negative number spends that place on its minus sign rather than taking a place of its own.

So a column holding `1.5` and `0.5` renders its cells as ` 1.5` and ` 0.5`, four wide, and a column holding `1.5` and `-0.5` renders its cells as ` 1.5` and `-0.5`, also four wide. Both columns are the same width and the second one starts one place to the left of where a naive reading would put it, because the minus is inside the column rather than in front of it. That is why `1    1.5` has four spaces in it and `2   -0.5` has three, on adjacent lines of the same output, which looks like a bug and is not one.

Stated that way it is easy to implement and easy to check. The cell is built with the place in it, the column is measured across the built cells, and the separator that is written before the cell is one space shorter than it looks, because the other space is already part of every cell. On a frame the separator is one space and the place makes it two. On a column it is three spaces and the place makes it four. Neither number was chosen here and both were measured.

## 3. The name is held in as well, but only on a numeric column

A column's name is written one place in too, and this is where the rule stops being uniform. It happens on the types pandas calls numeric and not on the others, so a frame with a text column named `aaaaaa` is one character narrower than the same frame with an integer column named `aaaaaa`, even though the values in both are one character wide and play no part in it.

```
  aaaaaa        aaaaaa
0      x     0       1
1      y     1       2
```

The left one is text and the right one is integers. Nothing about the data explains the difference and nothing but the name is wide enough to matter. What decides it is the dtype, because pandas indents the name by the same place it keeps for the sign, and it keeps that place on a numeric column whether or not anything in the column is negative.

The booleans are numeric for this purpose, which is the one place where that classification is visible to a reader. A boolean column named `available` is one wider than a text column named `available`, held in by a sign that no boolean will ever print. That is not defensible on its own terms and it is not ours to defend, it is what `is_numeric_dtype` answers for a boolean and it falls out of that.

## 4. A timestamp keeps no place at all

The temporal types are formatted by a path of their own in pandas that never keeps the place, so a timestamp column is written hard against its separator and its name is not held in either. Nothing a timestamp prints could begin with a minus, so there is nothing for the place to be for.

```
      aaaaaa
0 2020-01-01
1 2020-01-02
```

One space between the label and the value, where every other type would have two. This is the reason the predicate in the code is written as "not temporal" rather than as a list of the types that do keep the place: the list of types that do is long and open ended, and the list that does not is three.

## 5. Two dots in a narrow column and three in a wide one

The elision was three dots here in every position. In pandas it is three dots in a column wider than three and two dots in a column of three or fewer, decided per column, so a frame of narrow integer columns beside a wide float column elides its rows like this:

```
     a     b
0    0  -0.0
1    1  -1.0
..  ..   ...
```

Two dots under the labels, two under `a`, three under `b`. The width it asks about is the width the column was already padded to and not the width of the widest value in it, so a column made wide by a long name gets three dots even though nothing in it is long. That distinction is worth keeping straight because it is the difference between measuring the data and measuring the rendering, and only the second one is right.

## 6. Centred on a column, right aligned on a frame, left aligned under the labels

Where the dots sit inside their cell is a third rule, and there are three answers. Under the row labels they are left aligned, with the rest of the cell, which is the only one that follows from something else. In a value column on a frame they are right aligned. In the single value column of a printed series they are centred.

Centred means what Python's `str.center` means, which is not quite what centred means: the odd space goes to the right, except when both the padding and the width are odd, where it goes to the left. So two dots in a column three wide print as ` ..` rather than as `.. `, and three dots in a column four wide print as `... ` with a trailing space that survives to the end of the line. Neither of those is guessable, both of them are `str.center`, and pandas gets them by calling exactly that. The `pad_middle` helper in `firepanda/frame/display.mojo` is a transcription of it and says so.

The label cell on the elided row of a printed series is blank rather than dotted, which is the last piece of the same family. A frame puts dots under its labels and a series does not.

## 7. The elided column is four wide

When there are too many columns to print, the column of dots that stands in for the ones that were left out is four wide and not three, in every row including the header row and the row that carries the index name. pandas inserts the literal string `" ..."` down the whole column, which is where the fourth place comes from, and the effect is a gap of two spaces on the left of the dots and one on the right where every other column has one on the left.

```
     0  1  2  ...  7  8  9
idx           ...         
0    1  1  1  ...  1  1  1
```

The name row of the elided column carries dots too, which is worth seeing once. There is no reason for it beyond pandas filling the whole column when it inserts it, and a reader who did not know that would take it for a statement about the index.

## 8. The footer says the name before the length

A printed column that was truncated ends with `Name: v, Length: 30, dtype: int64`. It ended with `Length: 30, Name: v, dtype: int64` here, which reads about as well and is not what pandas writes. The order is the sentence pandas chose, what this is called and then how much of it there is, and it costs nothing to write the parts in that order.

## 9. What is left

The two renderers now agree with pandas on every layout question either of them has to answer, as far as either of them has been measured, and the measurements are in `python/tests/test_printing.py` where they compare whole renderings against a running pandas rather than against a literal somebody typed.

What is still different is the part document 60 named and this document does not change. A null prints as `<NA>` here and as `NaN` there, on purpose, and the reasoning is in the header of `firepanda/frame/display.mojo`. The shape line under a frame is always printed here and printed only after a truncation there. Printing a frame from Python reports its schema and its shape rather than its rows, which is document 13 section 2. And a `MultiIndex` prints like a plain index because there is no `MultiIndex` yet.

One thing in here is measured only as far as the types that exist. The place in front of a value is kept for everything but a temporal column, which is right for the types this library has today, and a decimal or an interval or a period would each have to be measured when it arrives rather than assumed into one bucket or the other.
