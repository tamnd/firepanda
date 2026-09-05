# The bound type is not a DataFrame

Document 12 measured the Python front door and found that the hard problem was not the one document 07 expected. This document is the same exercise one level down. Document 12 proved that a Mojo extension can be built, vendored and imported on a machine with no toolchain, and it did all of that with a module holding four free functions on it. It never bound a type. M3 P2 is where types get bound, and the reference document 07 points at, `max/sys/_hal/mojo_module.mojo`, binds ten of them, so the reasonable assumption going in was that binding a type is the same exercise as binding a function with a little more ceremony.

That assumption is wrong, and the way it is wrong decides the architecture of the whole Python layer rather than the shape of one file. `PythonTypeBuilder` can attach methods to a type and nothing else. It cannot attach `__getitem__`, it cannot attach `__len__`, it cannot attach a property, and the type it produces cannot be subclassed from Python. Those four facts together mean that `df["revenue"]`, `len(df)`, `df.shape` and `for row in df` cannot be implemented in Mojo at all in this toolchain. Since `df["revenue"]` is the single most common line of pandas ever written, a firepanda whose user facing object is a bound Mojo type is not a pandas replacement, it is a library with an unfamiliar accessor syntax.

The conclusion is that the object a user holds is a pure Python object that holds a Mojo object, and this document is the measurement that argues for it, including what the extra layer costs in nanoseconds.

## What was actually run

Everything below was run against Mojo 1.0.0, toolchain `ed45d567`, on macOS arm64, the same setup document 12 used.

Two probes. The first is a module binding one type called `Frame`, with an initialiser, two methods and both of the `Writable` methods, built as a shared library and imported. Its job was to find out what a bound type actually offers a Python caller, and it answered by being asked for each of the things pandas needs and refusing most of them.

The second is a set of compile only probes in the manner of document 12, calling members of `PythonTypeBuilder` with no arguments and reading the declaration back out of the compiler error, and building a method at each arity from zero to nine to find the ceiling by bisection.

## 1. The builder has four methods, and that is the whole of it

`PythonTypeBuilder` exposes `def_py_init`, `def_method`, `def_staticmethod` and `finalize`. Every other name that a binding layer would reach for reports `'PythonTypeBuilder' value has no attribute`: `def_classmethod`, `def_getset`, `def_property`, `def_slot`, `def_repr`, `def_len` and `def_getitem` are all absent. There is no escape hatch on the builder for reaching a type slot that the builder does not already know about.

The builder itself is also not `ImplicitlyCopyable`, so it has to be bound with `ref` or transferred with `^`, which is a five minute problem rather than a design one but is recorded here because the error message does not suggest the fix.

## 2. What that costs, asked of the type rather than of the compiler

The bound `Frame` was handed to a Python interpreter and asked for the four things pandas users do constantly. All four failed, and they failed in the way an ordinary object fails rather than in some diagnosable binding specific way, which is worth noting because it means a partially bound type looks exactly like a badly designed one from the outside.

```
f["a"]        -> TypeError: 'Frame' object is not subscriptable
len(f)        -> TypeError: object of type 'Frame' has no len()
f.shape       -> AttributeError: 'Frame' object has no attribute 'shape'
list(iter(f)) -> TypeError: 'Frame' object is not iterable
```

The one thing that does work is printing. A bound type that implements `Writable` gets both `__str__` and `__repr__` for free, with no call to the builder, which is a genuinely nice piece of design. There is a wrinkle in it. Both of them come from `write_repr_to`, and `write_to` is silently ignored: the probe's `write_to` writes `Frame(7 rows)` and its `write_repr_to` writes `Frame(n=7)`, and both `str(f)` and `repr(f)` returned `Frame(n=7)`. For firepanda this is survivable, because pandas prints the same table for both, but a struct that carefully writes two different forms will find that only one of them is ever reachable and there is nothing in the API to say so.

## 3. Inheritance is not the way out either

The obvious workaround for a type that is missing dunders is to subclass it in Python and add them there, which costs one small class and keeps the object identity intact. That is not available. `class Sub(Frame)` fails with `TypeError: type 'Frame' is not an acceptable base type`, because the type is created without `Py_TPFLAGS_BASETYPE` and nothing in the builder offers to set it.

The instances have no `__dict__` either, so attributes cannot be attached from Python after the fact: `f.extra = 1` fails with `'Frame' object has no attribute 'extra' and no __dict__ for setting new attributes`. That closes the other workaround, which would have been to build the object in Mojo and decorate it in Python.

So the relationship between the Python object and the Mojo object has to be composition. The user holds a Python `DataFrame`, the Python `DataFrame` holds the bound Mojo value in a slot, and every method and dunder on it delegates.

## 4. The arity ceiling is eight, and it is lower than it sounds

`PyObjectFunction` carries about 138 `__init__` overloads, covering arities zero through eight in raises and non raises forms, returning `PythonObject` or nothing, with and without `var **kwargs: PythonObject`. Nine arguments is rejected. For `def_method` the `py_self` argument is one of the eight, so a bound method gets seven real ones.

Seven is not a comfortable number for this API. `DataFrame.merge` in pandas takes thirteen parameters, `read_csv` takes upwards of forty, and `groupby` takes nine. The saving grace is that keyword arguments do not count against the ceiling and they work properly: a method declared with `var **kwargs: PythonObject` compiles alongside up to seven positional arguments, and at runtime `probe.show('agg', volume='sum', p99='quantile')` returned `['agg', ['volume', 'sum'], ['p99', 'quantile']]`, so both the values and their order survive the crossing.

That means the ceiling is not a wall, but it does dictate a house style. Every bound method takes the arguments that are positional in pandas and pushes the rest through `**kwargs`, and since the Python layer from section 3 is already in front of every one of these, that layer is where the real pandas signature lives, with its defaults and its keyword only markers and its type annotations. The Mojo side gets a narrow calling convention and the Python side gets the pandas one. This is the same conclusion section 6 arrives at from a different direction.

## 5. The commonest mistake raises the wrong exception, anonymously

Document 12 section 5 found that a bound function can only raise `Exception`, and treated it as a problem about firepanda's own errors. It is worse than that, because the binding layer raises one itself, for the most common mistake a user can make.

Calling a bound method with the wrong number of arguments produces `Exception: TypeError: <mojo function>() takes 1 positional argument but 2 were given`. The message is exactly right and the type is wrong. It is an `Exception` with the words `TypeError` at the front of its message, so `except TypeError` does not catch it, and the function is named `<mojo function>` rather than by the name it was registered under, so a traceback from a call several frames deep does not say which call was wrong.

This is not firepanda's code and cannot be fixed in firepanda's code. It is one more thing the Python layer has to cover, and it raises the priority of that layer from a convenience to a requirement, because without it the first error a new user sees is untyped and unnamed.

## 6. So the surface is a Python object, and here is what that costs

Sections 1 through 5 all arrive at the same place from different directions. The dunders are not reachable, subclassing is not available, the arity ceiling wants a narrow convention underneath a wide signature, and both the library's errors and the binding layer's own errors need re-raising with the right type. Every one of those is solved by the same thing, a hand written or generated pure Python class holding the Mojo value, and none of them is solved by anything else.

That is a happier answer than it first looks, because it is where document 12 section 5 had already put the error mapping, and where document 07 section 7 had already put three other things. What changes is the status. It was a convenience layer that could have been dropped if it ever got in the way, and it is now the layer the pandas API lives in, with the Mojo bindings as a private calling convention underneath it that no user ever sees.

The cost is one Python call frame on every operation, and since the whole project is a performance claim, that number should be measured rather than waved at. Two hundred thousand iterations each, on the probe type.

| call | cost |
|---|---|
| an empty lambda, as the floor | 17.6 ns |
| a bound Mojo method returning an int | 90.4 ns |
| the same through a Python `__len__` | 134.2 ns |
| the same through a Python `__getitem__` | 175.7 ns |

So delegation costs between 44 and 85 nanoseconds per call. For anything that touches a column this is not a number worth discussing, because a groupby over ten million rows is milliseconds and this is nanoseconds. It matters in exactly one place, which is a Python loop doing scalar access, and pandas is slow at that too for the same reason, so the comparison firepanda is judged on is not threatened by it.

The number that deserves more attention is the 90.4 nanoseconds for the bound call itself, before any wrapper. That is expensive for a call that does nothing but read an integer out of a struct, and it is the real floor on how fine grained this API can afford to be. It says the boundary should be crossed once per operation on a column, never once per element, which is the same rule the Arrow crossing in document 12 section 3 is built on. Any part of the design that is tempted to cross per row should be read again.

## 7. The binding table cannot be a table

Document 07 section 3 asks for one declarative table that the registration is generated from, and both document 12 and this document have agreed with it. It is worth knowing what shape that table is allowed to be before writing one, because the obvious shape does not compile.

`def_method` is declared `def def_method[method_type: TrivialRegisterPassable, //, method: PyObjectFunction[method_type, method.self_type, method.has_kwargs]](mut self, method_name: StringSpan[ImmStaticOrigin], docstring: StringSpan[ImmStaticOrigin] = StringSpan(""))`. The `//` marks `method_type` as inferred only, so it cannot be passed and has to be recovered from the argument at the call site. That is the whole difficulty. A bare function converts implicitly to `PyObjectFunction`, and choosing which of its 138 constructors applies needs the concrete function type, which the compiler prints as `def height(py_self: PythonObject) raises thin -> PythonObject`. Pass that function through anything generic and the type is gone.

A plain comptime alias survives, so `comptime H = Frame.height` followed by `b.def_method[H]("height")` works and the method appears on the type. A struct holding the function as a parameter does not: `struct Entry[t: TrivialRegisterPassable, //, f: t]` accepts `Entry[Frame.width]("width")` without complaint, and then `b.def_method[e.f](e.name)` fails with `failed to infer parameter 'method_type'`. A helper parameterized the same way directly, with no struct in the middle, fails identically, which is what says the problem is the implicit conversion rather than the indirection.

The way through is to do the conversion where the function is named and forward the result, spelling out all three parameters at every layer.

```mojo
def bind[
    mt: TrivialRegisterPassable, st: Deinitable, hk: Bool, //,
    m: PyObjectFunction[mt, st, hk],
](mut b: PythonTypeBuilder, name: StaticString):
    _ = b.def_method[m](name)

comptime w = PyObjectFunction(Frame.width)
bind[w](t, "width")
```

That compiles, imports, and the method is callable, so a binding can be handed around after all. What cannot be done is collecting them. Every `PyObjectFunction` has a different type, so any list of them is heterogeneous, and a variadic parameter pack over them is rejected before the body is ever looked at, with `element type parameter must be an 'inferred' parameter` and then `inferred parameter cannot depend on non-inferred parameter` once per parameter.

So the registration is a sequence of calls, one per binding, and in this toolchain it cannot be anything else. The table cannot be a Mojo value that Mojo walks, which means the single source of truth has to sit outside Mojo and emit that sequence. The four outputs in section 8 are generated files, and CI regenerates them and diffs to prove they are current.

This is a smaller loss than it looks. The property document 07 wanted was that an upstream change to the binding API is one file rather than four hundred call sites, and that property survives intact. The one file is the generator rather than the table, and the four hundred call sites exist without anybody writing or reading them.

Two incidental notes from the same probes. `fn` is gone from type expressions as well as from declarations, so `fn (PythonObject) raises -> PythonObject` written as a type is a syntax error, and `thin` is part of the printed type and a hand written annotation without it will not match. And `alias` is now deprecated in favour of `comptime`, which has nothing to do with any of this but will appear as a warning in anything new.

## 8. What this changes about the plan

P2 as written produces three artifacts from the binding table: the Mojo registration, the `.pyi` stubs and a surface parity test. It now has to produce four. The Python wrapper class is generated from the same table, or at minimum its delegation is, because a table that generates the stubs but leaves the wrappers hand written puts the two most easily divergent files in the project on either side of a boundary with nothing checking them against each other.

The surface parity test changes shape with it. It was going to check the Mojo type against the table in both directions. It now has three surfaces to keep in agreement, the Mojo methods, the table, and the Python class, and the test that matters most is the one that was not planned at all: that the Python class exposes the pandas signature, with defaults and keyword only arguments in the right places, rather than merely exposing a method of the same name. That is checkable against pandas itself with `inspect.signature`, and it is the closest thing to a mechanical measurement of the compatibility goal that exists anywhere in this repository, so it should be built even though nothing in M3 originally asked for it.

The `.pyi` stub question gets simpler in one way and harder in another. Simpler because a pure Python class can carry its own annotations inline and needs no stub at all, so `python/firepanda/_firepanda.pyi` shrinks to the narrow private convention rather than growing to four hundred entries. Harder because the annotations now have to match pandas, and pandas has its own stubs with their own opinions.

Two things in document 12 section 8 should be restated in light of this. The surface parity test being "not started and depends on the BINDINGS table" is still true but understates it, because the table now has a fourth output and the test has a third surface. And the error mapping being "blocked, and the design that unblocks it is the Python layer" is no longer a recommendation among two, it is the only design left standing, since section 5 here shows the binding layer generating untyped errors of its own that no amount of care on the Mojo side can prevent.

Nothing here changes M3's ordering. P2 is still the next thing to build and it is still the thing everything after it depends on. It is simply larger than it was written to be, and the part that grew is the part that faces the user.

## How to run the probes again

The type probe. Build a module with `mojo build --emit shared-lib probe.mojo -o probe.so` from inside the pixi project, where `probe.mojo` declares a `@fieldwise_init struct Frame(Movable, Writable)` with a `py_init` static method, two static methods taking `py_self`, and both `write_to` and `write_repr_to` writing different text. Register it with `b.add_type[Frame]("Frame").def_py_init[Frame.py_init]().def_method[Frame.height]("height")`, remembering that the builder cannot be assigned to a `var`. Then import it and evaluate `f["a"]`, `len(f)`, `f.shape` and `list(iter(f))`, each in its own `try`, and compare `str(f)` against `repr(f)`.

The subclass probe is three lines in the same interpreter: `class Sub(Frame): pass`, then `f.extra = 1`, then `dir(Frame)`.

The arity probe generates a method with n `PythonObject` arguments for n in a range, builds each one, and counts `error:` lines in the output. Eight succeeds and nine fails for `def_function`, and for `def_method` the same bisection puts the last success at seven arguments after `py_self`.

The table probe from section 7 is four builds of the same module, each one adding a layer of indirection between the function and `def_method`: a bare `comptime` alias, a struct parameterized on the function, a helper parameterized the same way, and a `PyObjectFunction` built at the naming site and forwarded. The first and last build and the middle two do not. The pack question is one more build, adding `def register[*entries: Named]` over two entries of differing types, which fails in the parameter list rather than in the body.

The timing numbers come from `timeit` at two hundred thousand iterations against a wrapper class with `__slots__ = ('_inner',)`, comparing a direct bound method call against the same call reached through `__len__` and through `__getitem__`.
