# 118. A group that names no flags

## 1. What this is

`(?)` is a flag group with nothing in it. RE2 takes it anywhere in a pattern and Python has no reading of it at all, and document 117 ended by pointing at it as the only construct still held out for the reason that neither reading here could read it. It was 67 patterns on `str.match` and 67 on `str.fullmatch` and none anywhere else, which is a shape worth explaining before the fix, because it is not a shape any caller made.

## 2. What RE2 does with it

Nothing, exactly and usefully. Measured against pyarrow 24.0.0:

```
    (?)        matches every row, including the empty one
    (?)a       is a
    a(?)b      is ab
    (?)(?)     is (?)
    ((?))      is a capturing group holding nothing
    (?-)       invalid perl operator: (?-)
```

So it is a flag setting that happens to set no flags. It is not an empty group, which is `(?:)` and is a different thing that RE2 also takes, and it is not the minus form, which RE2 refuses because a minus with no letter after it is not a flag group at all.

The position rule is the one worth saying out loud. Python has a rule since 3.11 that a global flag group may only be written at the very front of a pattern, and document 102 is about what RE2 does instead, which is read it as a flag that runs to the end of the enclosing group. A group naming no flags has no such question to answer: `a(?)b` is `ab` wherever the brackets are, because there is nothing to run to the end of anything.

## 3. Who writes one

Nobody.

The 67 patterns are 67 occurrences of two texts, `?` and `^?`. `str.contains("?")` hands Arrow the pattern as written and gets an `ArrowInvalid` saying there is no argument for the repetition operator, which is correct and is what this library already said. `str.match("?")` strips a caret the caller may have written and wraps what is left in a bracket, so what reaches Arrow is `^(?)`, and that is a pattern rather than an error. The same `?` is a raised exception one way round and a column of `True` the other.

That is upstream's shape and not this library's, and it is why the construct had to be read. A library that refuses `(?)` on the grounds that nobody writes one is a library that refuses `str.match("?")`, which pandas answers.

## 4. Where it goes in

One branch in `_flags` in `parse.mojo`, gated on the reading being RE2's.

The cursor arrives on the character after `(?`, and if that character is the closing bracket and the grammar being read is RE2's, the bracket is taken and `NOTHING` is returned. `NOTHING` is what the parser already returns for a global flag group, which is the node that is not a node, and it is the right answer here for a stronger reason than it is there: a global flag group at least sets a flag, and this one is a construct with no effect on anything the compiler will later look at.

The default reading is untouched and still refuses. Python's own sentence for `(?)` is `unknown extension ?)`, the parse gives up, and the branch document 117 added carries the pattern to the grammar reader, which says RE2 would take it, which is what asks for the second reading. So this slice is three lines of parser and nothing else, and it only works at all because the slice before it built the road.

## 5. The repeat that falls out

A flag group leaves no item behind, so a count written after one counts whatever was already there. That is the rule the parser has had since document 104 and it is RE2's rule, and putting the empty group on the same footing gets both of these right without either being written down anywhere:

```
    a(?)*      is a*
    a(?){2}    is a{2}
    (?)*       no argument for repetition operator
```

The last one is the interesting one. There is nothing in front of the group, so there is nothing for the star to count, and RE2 says so. The parser reaches the same conclusion and gives up, the grammar reader refuses the pattern on the way in, and the caller gets the reader's own sentence and a `ValueError` rather than a gap, which is the branch working the way document 117 described it.

## 6. What the reader needed

Nothing, and it had been right about this since it was written.

`re2.mojo` reads a flag group by taking letters until it runs out and then looking at what stopped it, so zero letters followed by a closing bracket has always been a flag group to it. That is why the 67 patterns were a gap rather than a refusal: the reader said RE2 reads this, and before document 117 nothing was listening, and after document 117 the thing listening asked the parser for a tree and the parser had no reading of it.

So this is the second time in fifteen slices that the parser was the one catching up with the reader, and both times were this month. The reader has been the more careful of the two statements of the grammar for longer than it looked.

## 7. The differential

The corpus gains twenty patterns, which is the group alone, beside a literal on each side, doubled, inside a capture, inside an alternation, inside a named group, against each of the four repeats, beside a written flag group, beside a word boundary, beside a class, inside a bracketed class where it is five ordinary characters, and the minus form RE2 refuses. It goes from 30453 to 30473.

```
                    before   after
    routing           831      831
    str.contains      230      230
    str.match         191      124
    str.fullmatch     191      124
    str.count         244      244
    str.replace       244      244
    str.findall       474      474
```

One hundred and thirty four held out patterns answered, and `Python's grammar cannot read this pattern` is now absent from all seven comparisons. The RE2 grammar reader reads 8710 and refuses 21763 and reads nothing RE2 refuses, with the three new refusals being `(?)*` and `(?)?` under no argument for the repetition operator and `(?-)` under invalid perl operator.

Every one of the seven is still at ten thousand agreements in ten thousand with no disagreements.

## 8. What is left

The four largest reasons are now the byte non boundary at 106 on three comparisons, the named character at 54 on four and 319 on `str.findall`, the three lookaround pairings at 60 between them, and the repeat that is too large at 10.

The front end is done. There is no pattern left in thirty thousand that RE2 reads and that this library turns away for want of a reading, on any of the seven comparisons, which is what documents 103 through 118 were for. Everything still held out wants an engine rather than a grammar: a machine that walks bytes, a table of thirty thousand names, a nested machine that can run two things at once, or a program representation that counts instead of copying.

The corpus is 30473 patterns against 96 texts. The texts are still the thing that has grown least, and a reading that is wrong only on a text nobody picked is still a reading all seven of these call right.
