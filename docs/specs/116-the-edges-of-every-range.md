# 116. The edges of every range

## 1. What this is

Document 115 found four names missing from the table that `\p{...}` is read against, and ended by saying that the ranges under the names it does have are checked by nothing at all. This is that check, and it is a seventh differential rather than a test because what it compares is this library against RE2 rather than this library against itself.

## 2. Why the table was unchecked

It was measured once. The generator asks the RE2 inside pyarrow about every code point there is, one name at a time, which is around two hundred calls over a column of 1112064 rows, and writes 200 names and 5820 ranges into a file that is then carried in the repository. That measurement was right when it was made and nothing has looked at it since.

Regenerating and finding the file unchanged is the honest check, and it is also minutes of compute and a file nobody wants to see in a diff, which is why it had never been part of anybody's loop.

The corpus cannot do it. A pattern that names a category proves that the name is found, the ranges are emitted, the class is built and the engine runs it. It proves nothing about whether those ranges are RE2's ranges, because a name is a set of up to a million code points and the corpus runs every pattern against a hundred texts. Document 115 added thirty eight patterns naming categories and scripts, and all thirty eight would pass over a table with a wrong number in it.

## 3. Why the edges are enough

A range table is wrong at its edges or it is not wrong at all.

The ways a generated table goes wrong are a low that is one too low, a high that is one too high, two ranges that should have been one, and one that should have been two. Every one of those shows up at a boundary. A range that is wrong in the middle is not a thing that happens, because nothing in the pipeline that produced the file can drop a code point out of the inside of a run and leave both ends where they were.

So the question asked is the four code points around each range, which are its two ends and the one outside each end. The ones outside are the half that matters: a range whose two ends both answer correctly can still be a range that swallowed the character next to it, and only the point beyond the end can say so.

That is 18834 questions rather than 1112064, fewer than four per range because the point after one range is often the point before the next, and the whole run takes half a second against a warm build.

## 4. What it asks and what it asks it of

pyarrow directly rather than pandas, which is the one thing here that differs from the other six differentials.

The other six ask pandas, because the question they are asking is what a caller gets, and what a caller gets is decided by pandas' router before either engine sees the pattern. This question is narrower. It is whether the table still says what the engine it was copied from says, and a router and an accessor in front of that would add nothing and would answer the same.

One call per name over a column holding one code point per row, which is the same shape the generator used, because it is still the only way to get a set out of an engine that will only say yes or no about a whole string.

A name RE2 no longer reads comes back as a refusal and is counted as a disagreement rather than an error. The table has the name, so RE2 having lost it is exactly the drift this is looking for. `Cs` is the one name with no ranges under it, since a surrogate has no UTF-8 encoding and no row can hold one, so it is asked about a letter instead: the answer has to be no, and a name that has been dropped answers with a refusal.

## 5. What it found

Nothing. 200 names, 18834 code points, zero disagreements, against the RE2 inside pyarrow 24.0.0.

That is the expected answer and it is worth having anyway, because until this ran it was an assumption. The table was generated against this release and is being checked against this release, so what this says today is that the file in the repository is the file the generator would write. What it is for is the day that stops being true, which is a pyarrow upgrade, and on that day this is half a second rather than a regeneration.

## 6. Checking the instrument

A comparison that has never failed is a comparison nobody has seen work. So one high was moved by one, from `\p{Zl}` covering U+2028 to covering U+2028 and U+2029, and the run was repeated.

```
    1 of 18835 code points answered differently
        \p{Zl} at 8233 is in here and out in RE2
    agreement 9999 in ten thousand, 1 disagreements
```

It names the character and says which side holds it, and the program exits with an error, which is what the other six do and is what makes a differential something a workflow could run. The count went up by one as well, since a range that reaches one further has one more edge.

## 7. What it cannot see

A range that is missing from a name altogether, because a range that is not in the table has no edge to ask about. If `\p{Thai}` had lost one of its blocks entirely this would not notice.

What catches that is the generator, which asks about every code point and writes the whole set. What catches a name missing altogether is the sweep document 115 added. Between the three there is full cover, and the ordering is by cost: this one is half a second and runs after a change, the sweep is a second and runs when the table is regenerated, and the full measurement is minutes and runs when somebody has a reason to doubt the file.

## 8. What is left

Nothing to run this from. It is the seventh regular expression differential and, like the other six, no workflow runs it, for the reason `tools/build_differential.sh` gives: the six were taking about half of a step that had become the longest pole in the pipeline, so they were moved out of the pull request build and are run by hand from the documents that introduce them. This one is the cheapest of the seven by a wide margin once it is built, and the cost of running it is entirely the cost of building it, which is the thing that was moved out. That is a tradeoff worth revisiting for this one alone on the day the table drifts.

The script names remain the part of the table that nothing exercises end to end, and that is now a smaller complaint than it was: 154 of the 163 have no character in the text corpus, but every one of them has had its ranges compared against RE2 by this, which is most of what a pairing would have shown.
