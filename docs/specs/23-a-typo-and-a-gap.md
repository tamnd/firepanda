# 23. A typo and a gap are different mistakes

Document 14 decided which class an error arrives as and document 22 decided what string it carries. This one is about a case both of them leave open: an argument was passed a value the call will not answer, and there are two entirely different reasons that can happen. Firepanda was giving both of them the same answer, and the answer was wrong for one of them.

## The two mistakes

`Series.quantile` takes an `interpolation` argument. Pandas accepts thirteen names for it. Firepanda has written one, `linear`, and the other twelve are on a list rather than in the code.

So there are two ways to get an answer you did not want:

```
s.quantile(0.5, interpolation="lower")   # a rule pandas has and firepanda has not written
s.quantile(0.5, interpolation="lowr")    # a rule neither library has
```

The first caller wants to know when it will work. The right answer for them is `NotImplementedError`, naming the issue, because the thing they asked for is real and is scheduled. Document 26 already argues at length that an argument accepted and ignored is worse than one refused, and this is the refusal it argues for.

The second caller has a typo four characters from a working line. The right answer for them is what pandas gives, which is a `ValueError` listing the thirteen names, because the list is the fix. Handing them a `NotImplementedError` about a feature firepanda has not got round to sends them to read a changelog for something that already exists under the spelling they meant, and the one thing they needed, the correct spelling, is nowhere in the message.

Both of these were `NotImplementedError` before. The gap answer was being given to the typo, because the code checked whether the value was the default and never checked whether it was a word at all.

## The rule

Ask whether pandas takes the value, not whether firepanda answers it.

A value pandas takes and firepanda has not written is a schedule. It is `NotImplementedError`, it names the reason and it names the issue.

A value pandas does not take either is a typo in the caller's own line. It is whatever pandas raises, with pandas' own sentence, for the reason document 22 gives about search boxes.

The check for the second has to come first, because a typo is also not the default and would otherwise be caught by the schedule branch on its way past.

## Not every fixed vocabulary gets one

The test is whether pandas checks, and the only way to find out is to run pandas. Reasoning about it gets the answer wrong, which is how this section came to exist.

`DatetimeProperties.tz_localize` takes an `ambiguous` argument with a small fixed vocabulary, and the obvious move is to give it the same treatment as `nonexistent` next door. Measured, pandas does not check the word on a column at all. `s.dt.tz_localize("UTC", ambiguous=3)` comes back with an answer in it. Adding a check would be firepanda refusing input pandas accepts, and that is the one direction of difference this library does not get to have: a wrong message is a bad afternoon and a wrong refusal is a program that stops.

The scalar is the other way round. `Timestamp.tz_localize` does check `ambiguous`, and refuses `infer`, which is a real word on a column, because a single moment has no neighbours to infer a direction from. So `_scalars.py` has a check that `_pandas.py` deliberately does not, and the two files differ there because pandas differs there.

## When an argument is read at all

The same measurement turned up a second thing, which is that an argument in a signature is not necessarily an argument the call reads.

`floor`, `ceil` and `round` all take `ambiguous` and `nonexistent`. Pandas reads them only when the value already carries a zone. Hand a naive timestamp a misspelled `nonexistent` and it comes back rounded, with the argument unread, because there is no daylight saving without a zone and so nothing for a policy to decide. `tz_localize` is not like this and always reads both, including when it is handed `None` and including when the moment is already zoned.

Firepanda was refusing both cases, which meant a naive column that rounds fine in pandas stopped. That is the wrong direction of difference again, and it was found by measuring rather than by reading, which is the point.

So the shape in the code is a `live` argument that says whether pandas looks at these on this call, with a comment recording what was measured. Where it costs a boundary crossing to answer, as it does for a column, the question is only asked when one of the two is off its default, since the default changes no answer either way.

## Where this applies

The pandas facing path, which is `python/firepanda/`, and both halves of it: the frame side in `_pandas.py` and the scalar side in `_scalars.py`. The two share the vocabularies by writing them out twice rather than importing across, because the scalar half has no extension under it and tying a pure Python `Timestamp` to a built `.so` for a four element tuple would be the more expensive mistake.

The differential tests are what hold this in place. `agree` in `python/tests/test_scalars.py` asks only which standard library class came out of each library, which is exactly the question this document is about, and it will fail the day a typo starts being answered as a gap.
