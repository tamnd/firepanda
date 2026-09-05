# Making a frame out of Python

Every pandas program ever written starts by making a frame, and until this work firepanda could only read one off disk or take one from another library. `pd.DataFrame({"a": [1, 2, 3]})` is the first line of every tutorial, every bug report and every test somebody writes before they trust a library, and answering it with a refusal means the compatibility work never gets looked at.

This is the conversion from Python values into Arrow columns, the type inference that decides what a list of Python objects is, and the place the pandas constructor signature lives.

## 1. What is accepted, and what is refused out loud

A frame is built from a mapping of column name to sequence, in the mapping's own order, which is what pandas does and is the reason dictionaries being ordered stopped being a detail. A series is built from one sequence. Everything else in the pandas constructor is refused rather than approximated, and the refusal names what was passed.

The sequences themselves go through `list()` first, so a tuple, a `range`, a generator and anything else iterable all arrive as the same thing. That is one line and it removes a whole category of "works with a list and not with a range" report.

Refused, each with a message saying so: an `index`, because there is no index type to put there yet, and a caller who passes one and is quietly ignored has been given a frame that is wrong rather than a frame that is incomplete. A `dtype`, because inference is the only path written and honouring an explicit request means the cast machinery. A `copy` flag, because there is exactly one behaviour and claiming to support both would be a lie about the one that is not implemented. A scalar where a sequence was expected, because pandas broadcasts it and firepanda does not, and broadcasting silently is the kind of difference that produces a plausible wrong answer. A list of dictionaries, which pandas reads as records, because the union of keys and the ordering rules are their own piece of work.

Columns of different lengths raise, with the same reading pandas gives it, because a frame is rectangular and there is no defensible thing to do with the short one.

## 2. Inference is a single pass and five outcomes

The values are walked once and classified. A `None` is missing and does not vote. Everything else votes for one of bool, integer, floating point or text, and the column type is decided from the set of votes.

Only `None`, or nothing at all, gives float64 with every row missing, and that is the least satisfying decision in this document. pandas answers with an object column, which is the representation firepanda exists to not have. pyarrow answers with the Arrow null type, which firepanda does not carry at run time, and `arrow_import.mojo` says so in as many words when it refuses to import one. So the choice is float64 or a refusal, and refusing to build a frame because one of its columns is empty would be absurd. float64 is the type that widens most freely if the column is later given numbers, and it is what pandas itself produces for a column of NaNs, which is the same list written a different way.

Bool alone gives bool. Integer alone gives int64. Floating point, alone or mixed with integer, gives float64, because that is the only type that holds both. Text alone gives string.

Anything else is refused, and the message names the row and the Python type that was found there. That includes bool mixed with integer, which pandas turns into an object column: `[True, 1]` is a list somebody made by accident far more often than it is one they meant, and there is no column type here that holds both without saying which of the two it thought they wanted.

An integer too large for int64 is refused rather than wrapped. pandas widens to an object column and keeps the value, which firepanda cannot do, and the two remaining options are to lose the value silently or to say so.

## 3. A NaN is a value, and a None is not

`float("nan")` in a list becomes a NaN in a float64 column, and the row is valid. `None` becomes a cleared validity bit, and the row is missing. They are different things here and the same thing in pandas, where a float column has no validity bitmap and a NaN is the only way it can record absence.

That is the storage. What counts them agrees with pandas: `null_count` on a float column counts the cleared bits and the NaNs, which is what the core is careful about and is why `count` and `hasnans` on a series say what a pandas user expects rather than what the bitmap says. So a NaN and a None are two things in memory and one thing in a reduction, which is issue #170's position arriving at the front door with nothing new added to it. The benefit of keeping them apart underneath is that a genuine NaN out of `0.0 / 0.0` has not been silently reclassified as data that was never there, and the day a caller wants to tell them apart the information is still there to tell them with.

## 4. The constructor is where the pandas signature lives

`DataFrame.__init__` takes `data`, `index`, `columns`, `dtype` and `copy`, in that order and with pandas' defaults, and `Series.__init__` takes `data`, `index`, `dtype`, `name` and `copy`. That is more than is implemented on purpose. The signature parity test in `python/tests/test_bindings.py` compares every parameter firepanda declares against the pandas parameter of the same name, so declaring the full signature and refusing four of them is a stronger statement than declaring one and matching nothing: a caller who passes `columns=` gets a message about `columns` rather than a `TypeError` about an unexpected keyword, and the day one of them is implemented no signature changes.

Both constructors are hand written, in `python/firepanda/_pandas.py`, and they are the second and third members to go there. They belong there by the rule document 17 section 2 sets out: what they do depends on their arguments. The generator writes everything else about both classes and knows that it is not writing these.

## 5. The wrapper had to stop being the constructor

The generated class used to hold the extension object and take it as the single argument of `__init__`, so `DataFrame(inner)` was how every method that returns a frame built its answer. That parameter is now in the way, because the public constructor has to be the pandas one and cannot also be an internal hand off with a private argument bolted to the end of it.

So the generator emits a `_wrap` classmethod instead, which allocates with `object.__new__` and sets the one slot without going through `__init__` at all, and every place the table says a member wraps its result now goes through it. That is a mechanical change across the generated files and it is the reason the diff is larger than the feature.

The slot moved with the constructor. `__slots__` is declared on the mixin now and the generated class declares an empty one, because a class cannot assign to a slot it does not own and the assignment is in the hand written `__init__`. Nothing about the property changed: an instance has one attribute and no dictionary to grow a second one in, and the test says that rather than asserting where the tuple is written.

It also removes a wart. `firepanda.DataFrame()` used to reach the Mojo `py_init` and refuse, which was the right answer to the wrong question, because the reason it refused was that the conversion did not exist. Now it makes an empty frame, which is what pandas does.

## 6. What the conversion costs

One pass to classify and one to fill, and a Python object read for every value in both. That is the slow way into a column, in exactly the way `tolist` in document 17 section 4 is the slow way out, and for the same reason: there is a Python object per element and nothing about that can be made fast.

It is worth having anyway and it is not the path that has to be fast. A caller with real data has it in a file, a database or a numpy array, and the fast paths for those are `read_csv`, `read_parquet` and `from_arrow`, all of which are already there and none of which touches this code. What this is for is the first ten lines of a program, a test fixture, and a small literal in a notebook, and in all three the list is short and the alternative is that the program does not run.

There is no dtype dispatch here, and it cost 60,256 bytes of binary. That is against 157 kilobytes for `tolist`, which does the same job in the other direction and does dispatch, and the difference between the two numbers is the whole of document 03's cost model in one comparison. The classification decides one of four concrete types and calls one concrete filler, rather than binding a runtime dtype to a comptime one and compiling twelve copies of the loop.

## 7. What is left

**An index.** `index=` is refused and stays refused until the `Index` type in #154 exists. Until then a frame's rows are numbered from zero, which is what pandas does when you do not pass one, so the common case is already right.

**An explicit dtype.** `dtype="int32"` is a cast on the way in and needs the cast machinery rather than the constructor.

**Records and the other shapes.** A list of dictionaries, a list of lists with `columns=`, a numpy array, a scalar to broadcast. Each is a separate reading of `data` and each is a small piece of work once the first one exists.

**The name of a text column.** pandas 3.0 calls it `str`, firepanda calls it `string`, and pyarrow and Polars both call it `string` too. The round trip test against pandas special cases exactly that one name and compares the rest directly, which is the honest way to record a divergence that is a spelling rather than a behaviour.

**numpy scalars.** A `numpy.float64` is a Python `float` and arrives correctly by accident. A `numpy.int64` is not a Python `int` and is refused. The right answer is not to special case them here, it is that a numpy array should arrive through the Arrow protocol, and that is the same work as reading a numpy array at all.
