# The Python front door, measured

Document 07 describes the Python front door and it was written before any of it existed. It says so in its own words, and it asks the reader to verify its code against the installed toolchain rather than against the page. This document is that verification. Everything below was run against Mojo 1.0.0, toolchain `ed45d567`, on macOS arm64, and every number in it came out of a command rather than out of an estimate. Where 07 guessed right this says so briefly. Where it guessed wrong, or where the thing it was worried about turned out not to be the thing worth worrying about, this says that at length, because those are the parts that change what M3 should do first.

The short version is that the problem 07 called the hardest one is largely solved and cost 2.9 megabytes, the Arrow crossing that 07 treated as the centrepiece works today with zero copy proven by pointer identity, and the two items 07 spent one paragraph each on, error mapping and Ctrl-C, are the two that do not work at all. M3 should be reordered around that.

## What was actually run

Three probes, all outside the firepanda tree so that none of this is in the library yet.

The first is a minimal Python extension module written in Mojo with four functions on it, built to a shared library, vendored beside the Mojo runtime libraries it needs, and imported from a Python interpreter that has no Mojo toolchain visible to it at all. Its job is the distribution question.

The second is the same thing linked against firepanda, exporting one int64 column through the Arrow C Data Interface wrapped in PyCapsules, and handed to pyarrow. Its job is the data crossing question, and it is the one whose result matters most.

The third is a set of small compile only probes against `std.python._cpython`, which is how the exact signature of every CPython function Mojo exposes was obtained. The Mojo compiler prints the declaration of a function when you call it with the wrong arguments, and prints a plain "has no attribute" when the function does not exist, so calling everything with no arguments at all is a reliable way to read an API that has no published reference.

## 1. The mechanism, corrected

The snippet in document 07 section 1 does not compile against Mojo 1.0.0, in three ways. `fn` was removed from the language and the declaration is `def`. The exported initialiser needs `abi("C")` in its effects position or the symbol is not what CPython looks for. And `abort` is called for its effect rather than returned, so the `return abort[PythonObject](...)` form is gone. The working shape, confirmed by building and importing it, is this.

```mojo
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_mojo_module() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("mojo_module")
        b.def_function[answer]("answer", docstring="Returns 42.")
        return b.finalize()
    except e:
        abort(t"failed to create Python module: {e}")


def answer() raises -> PythonObject:
    return PythonObject(42)
```

The module name in the `PyInit_` symbol, the string passed to `PythonModuleBuilder`, and the file name of the built library all have to agree, and nothing checks that for you. A mismatch produces an `ImportError` about a missing init function rather than anything that names the real cause.

Types are registered with `b.add_type[T]("Name").def_py_init[T.py_init]().def_method[T.method]("method")` and the calls chain. The definitive reference for that form is not documentation, it is `max/sys/_hal/mojo_module.mojo` inside the installed toolchain, which binds ten types and about sixty methods and is Modular's own shipping code. `max/_core_mojo/mojo_module.mojo` is the same thing for free functions and also shows raw `cpython` usage and numpy array interop. Both are worth reading in full before the `BINDINGS` table in section 3 of document 07 is designed, because they are the only worked examples of this API that exist.

The build command is `mojo build --emit shared-lib -o <name>.so <name>.mojo`, and it has to be run from a directory inside the pixi project or pixi cannot find its manifest, which is a footgun when the source lives outside the tree.

## 2. Distribution, which turned out not to be the wall

Document 07 calls this the real problem and puts a bold instruction in the middle of it: check the licensing question first, before writing any binding code, because a negative answer changes the entire distribution strategy. That instruction is still right about the licence. It is no longer right that everything has to wait for it, because the technical half of the question is now answered and the answer is small.

A built Mojo Python extension links three things. `@rpath/libKGENCompilerRTShared.dylib`, `@rpath/libAsyncRTMojoBindings.dylib`, and `/usr/lib/libSystem.B.dylib`. Those two Mojo libraries in turn pull in `libMSupportGlobals.dylib` and `libAsyncRTRuntimeGlobals.dylib`. That is the whole of it. Copied beside the extension, the four runtime libraries come to 2.72 megabytes and the extension itself is 177 kilobytes, so the complete self contained set is 2.9 megabytes. That is what `pixi run build-extension` prints today, and `python/tests/test_extension.py` holds it under a budget. Linux is smaller and needs one library fewer, at 2.02 megabytes on aarch64 and 2.12 on x86-64 with no `libAsyncRTMojoBindings` in the set at all, so the macOS figure is the ceiling of the three and section 6 has the rest of that.

The extension does not link libpython. Its only Python related undefined symbol is `_KGEN_CompilerRT_Python_SetPythonPath`, which is a Mojo runtime entry point and not a CPython one, and every CPython call goes through a function table the runtime resolves at load time. That is worth pausing on, because it is the fact that makes several other things in this document possible, and it is a better position than a C extension is normally in.

The built library carries exactly one `LC_RPATH`, and it is the absolute path of the pixi environment it was built in, which would be useless on any other machine. `install_name_tool -rpath <that absolute path> @loader_path <the library>` rewrites it and that is the only patching needed, because the four runtime libraries already list `@loader_path` first in their own rpaths and resolve each other correctly wherever they are put. The first attempt at this tried to patch all five and got an error on the four, and the error was right: there was nothing to patch.

Turning that by hand copying into `tools/build_extension.sh` found two things that hand copying had hidden, and both of them are the kind that would have been found much later and much more expensively.

The first is that a dependency has to be followed by how it is named and not by what it is called. The obvious script walks the dependency list, takes the file name, and copies that file out of the toolchain library directory if it is there. That script vendors `libc++.1.dylib`, because all four runtime libraries name the C++ standard library and the pixi environment happens to contain a copy of it. It is 1.17 megabytes, which took the set from 2.9 to 4.1 and was the visible symptom, but the size is the least of it: the runtime libraries name that library as `/usr/lib/libc++.1.dylib`, an absolute system path, so what the vendored copy would do in a host process that already has the system one loaded is at best nothing and at worst a duplicate symbol crash a long way from here. The rule that is actually right is that a Mojo library is named `@rpath/something` and a system library is named by absolute path, so following only the loader relative references is both simpler than a list of names to exclude and more likely to still be correct after a toolchain upgrade.

The second is that on Apple silicon a binary that has been edited has to be signed again. Every arm64 macOS binary carries a signature, ad hoc if nothing else, `install_name_tool` invalidates it, and the loader's response to an invalid signature is not an error, it is `SIGKILL`. What that looks like from the outside is a `python -c "import firepanda"` that exits 137 having printed nothing at all on either stream: no ImportError, no traceback, no dyld message, nothing to search for. `codesign --force --sign -` on each edited file fixes it. The related discipline is to edit as few files as possible, which is why the script now removes only absolute rpaths: the four runtime libraries carry several `@loader_path` relative entries each, mostly Bazel leftovers pointing at directories that do not exist here, and a search path that resolves to nowhere costs one failed stat while removing it costs a signature. The vendored libraries are now byte identical to the toolchain's, and only the extension is touched and re-signed.

`python/tests/test_extension.py` is where those stop being anecdotes. It stages the package the way a wheel has it, runs a child interpreter with the environment taken away from it, and asserts inside the child that no Mojo toolchain is on `PATH` before it imports anything, because a test that quietly ran inside the build environment would pass and mean nothing. Three more checks sit beside it: the total size against a budget, which is what catches a wrongly vendored library in general rather than `libc++` in particular; every search path being relative, which is the failure that running the thing cannot catch, since the machine that built it is the one machine where a path into the build environment still resolves; and every signature verifying, which catches the 137 and says which file and why.

The proof is the import. From a Homebrew CPython 3.14.7, launched with `env -i` so that it inherits nothing, and with `PATH` set to `/usr/bin:/bin` so that no Mojo binary and no pixi environment is reachable, the module imports and its function returns the right answer. The interpreter that loads it was not built by Modular, does not know Mojo exists, and has no toolchain to fall back on.

So `pip install firepanda` on a machine with no Mojo toolchain is technically possible, today, at a cost of 2.9 megabytes of vendored runtime. For scale, the pandas 3.0.3 and numpy 2.5.2 that this project is measured against occupy 46.4 and 23.2 megabytes installed, so the whole firepanda runtime is about four per cent of what pandas already asks a user to download. That is an install footprint number and nothing more, and it should not be quoted as though it were a memory or a CPU number, but it is the one resource axis where the project's stated goal of a tenth of the resources is already met by a wide margin and can be stated without a benchmark.

One more measurement bears on wheel size and it was a surprise. The second probe links the whole of firepanda's Arrow export path, which drags in the array layer, the bitmap, the buffer allocator, the dtype system and the logical type table, and it is 166 kilobytes. That is two kilobytes smaller than the trivial four function module, because the two were built at different times with different function counts, but the point stands: linking a real slice of the library added nothing measurable. Mojo dead strips hard, and a firepanda wheel is going to be the runtime plus a rounding error rather than the runtime plus a library.

There is one observation from the toolchain that belongs here as a maturity signal rather than as a problem. Modular's own Python packages do not use this path. `max/_core_mojo` and `max/sys/_hal` ship `.mojo` source files and rely on `import mojo.importer`, which installs a meta path finder that shells out to `mojo build --emit shared-lib` on first import and caches the result in a `__mojocache__` directory. That is a development convenience and it requires the toolchain to be present, which is exactly what firepanda cannot require. So the ahead of time path firepanda needs is a path Modular ships but does not itself exercise in its own products. The probe shows it works. It also means firepanda will be among the first to find out when it stops working, and the CI matrix in document 07 that builds against both the pinned toolchain and nightly is not paranoia, it is the only thing that will notice.

What remains open is the licence, and only the licence. Whether Modular's terms permit redistributing those four dylibs inside a wheel on PyPI is a question for a human to ask Modular and get an answer to in writing. Nothing in this document answers it and nothing in it can. What has changed is the cost of asking late: the binding work no longer has to wait on the answer, because the answer does not change any of the code, it changes only whether the artifact may be published. So the correct ordering is to ask the question now, in parallel, and to gate publication on it rather than gating the milestone on it.

## 3. Arrow crosses zero copy, and the address proves it

This is the result that most changes the shape of M3, and it is better news than 07 expects.

firepanda already has the Arrow C Data Interface. `firepanda/io/arrow_c.mojo` declares `ArrowSchema` and `ArrowArray` with the field order, field widths and format strings pinned by tests, `firepanda/io/arrow_export.mojo` fills them in without copying anything and installs release callbacks backed by heap boxes, and `firepanda/io/arrow_import.mojo` goes the other way. All of that landed at M2 and none of it knows about Python. What was missing between it and the PyCapsule protocol was a wrapper, and the wrapper is about forty lines.

Mojo's CPython binding exposes what that wrapper needs. The signatures, read out of the compiler, are these.

```
def PyCapsule_New(pointer: Pointer[NoneType, MutUntrackedOrigin], name: StringSpan[ImmStaticOrigin], destructor: def(PyObjectPtr) abi("C") thin -> None) -> PyObjectPtr
def PyCapsule_GetPointer(capsule: PyObjectPtr, var name: String) -> Pointer[NoneType, MutUntrackedOrigin]
def PyCapsule_IsValid(capsule: PyObjectPtr, var name: String) -> Bool
```

The destructor is not optional in that signature, which is fine, because the Arrow protocol requires one anyway: a capsule that is dropped without the consumer having taken it still has to release the struct it holds. `release_schema` and `release_array` in `arrow_c.mojo` are already defined to be no-ops on an already released struct, so the destructor is a `PyCapsule_GetPointer`, a release and a `free`, with no extra bookkeeping.

The probe puts an exported `ArrowSchema` and `ArrowArray` into malloc'd boxes, wraps each in a capsule named `arrow_schema` and `arrow_array` as the protocol requires, and returns the pair. A five line Python class with an `__arrow_c_array__` method that forwards to it is enough to make `pyarrow.array()` accept it. The result is an int64 array of length five with the right values and a null count of zero, and it goes into a `pyarrow.table` without complaint.

The zero copy claim is checked the way document 07 section 4 insists it must be, by address and not by eye. The probe returns the address that firepanda put in `buffers[1]` alongside the capsules, and Python compares it against `arr.buffers()[1].address`. They are the same address. Deleting the array and forcing a collection runs the capsule destructors and the release callbacks, and the process does not crash.

So the front half of document 07 section 4 is not a plan, it is a thing that works, and the `to_polars` and DuckDB halves of the M3 exit criteria come along with it because all three consume the same protocol.

The one Arrow item M3 does have to build is `ArrowArrayStream`. `arrow_c.mojo` says in its own header that the stream struct "arrives with whichever of them needs it first", and `__arrow_c_stream__` is the one that needs it. A stream is three function pointers and an error string rather than a data structure, so it is a smaller job than either of the two structs that already exist, but it is a job and it is not started.

## 4. The GIL releases, and it is measured rather than assumed

`std.python._cpython` exports `GILReleased` and it is used as a context manager.

```mojo
from std.python._cpython import GILReleased

with GILReleased(Python()):
    ...
```

Modular uses it in shipping code, in `max/_distributed_ops/distributed_ops.mojo` and `max/_distributed_ops/block_offload_ops.mojo`, with comments that say exactly what document 07 section 6 says: release it around the blocking work so other Python threads are not stalled.

The measurement is a Mojo function that sleeps for one second, called from Python while another Python thread increments a counter in a loop. Holding the GIL, the other thread ticked once. Inside `GILReleased`, over the same one second, it ticked 673 times. That is the whole claim and it holds.

The first attempt at that measurement was worthless and it is worth recording why, because the same trap is waiting in every benchmark this project will write. The workload was an arithmetic loop over a hundred million integers, and the compiler folded it to a constant, so the function returned the right sum in zero seconds and the other thread ticked once in both configurations. A workload the optimiser can see through measures nothing, and the fix was to use a real blocking sleep, which no optimiser is allowed to remove.

## 5. The two things that do not work

These are the M3 risks. Neither was flagged as a risk in document 07 and both are load bearing for its exit criteria.

**A bound function can raise exactly one Python exception type, and it is `Exception`.** A Mojo `raise Error("this is a mojo error")` inside a function registered with `def_function` arrives in Python as `Exception: this is a mojo error`. That much is expected. What is not expected is that setting the exception by hand first does not help. The probe calls `PyErr_SetString(cp.get_error_global("PyExc_KeyError"), message)` and then raises, and Python still sees `Exception`, carrying the Mojo message rather than the one that was set. The binding wrapper catches the Mojo error and sets its own, and it overwrites whatever was there.

The error globals themselves are fine and are reachable. `cp.get_error_global("PyExc_KeyError")` returns the real `<class 'KeyError'>`, and the same for `PyExc_NotImplementedError` and `PyExc_KeyboardInterrupt`, so the raw material for the table in document 07 section 5 exists. It is the `def_function` wrapper that stands between it and the caller.

That table is not decoration. pandas raises `KeyError` for a missing column and users write `except KeyError`, and a firepanda that raises bare `Exception` for everything fails pandas compatibility in a way no conformance case about values will catch. So this has to be solved, and there are two ways.

The first is to put the mapping in the thin `__init__.py`. Every public entry point is already going to be wrapped there for other reasons, document 07 section 7 lists three of them, so a decorator that catches `Exception`, reads a structured prefix off the message, and re-raises the right class costs one function and no Mojo. The cost is that the traceback gains a frame and that the prefix becomes part of the wire format between the two layers, which means it has to be tested from both sides.

The second is to stop using `def_function` for the entry points that need typed errors and register raw `PyCFunction`s directly, which puts firepanda in control of the return path and lets it leave a set exception in place. That is more code, it is code against an API Modular describes as early and expected to change, and it duplicates the argument conversion that `def_function` is doing for us.

The recommendation is the first one, and the reason is document 07's own framing of why the binding table exists: an upstream change should be one file rather than four hundred call sites. Doing the mapping in Python keeps it one file. If Modular later exposes a way to raise a typed error from a bound function, the Python layer is deleted and nothing else moves.

One incidental finding from this probe, recorded because it costs an afternoon to rediscover. `PyErr_SetString`'s message parameter is typed `Optional[Pointer[Int8, ImmutAnyOrigin]]` in Mojo, so passing `None` compiles. It then segfaults the interpreter, because CPython dereferences it. The message is not optional in practice. Building a null terminated message goes through a `List[Int8]` and `Pointer[Int8, ImmutAnyOrigin](unsafe_from_address=Int(bytes.unsafe_ptr()))`, since `String` has no `unsafe_cstr_ptr` in this toolchain and `unsafe_origin_cast` will not change mutability, so the bitcast has to come before the origin cast.

**`PyErr_CheckSignals` is not exposed.** The compiler's answer is flat: `'CPython' value has no attribute 'PyErr_CheckSignals'`. `PyErr_SetInterrupt` is not there either. So the third M3 exit criterion, that Ctrl-C interrupts a running query and raises `KeyboardInterrupt`, cannot be met through the stdlib binding as it stands.

There are two routes. One is to resolve the symbol ourselves, since libpython is already loaded in the process and `dlsym` will find it, and then call it at the morsel boundary exactly as document 07 describes. That is a small amount of unpleasant code in one place and it does not depend on Modular adding anything. The other is to keep the check on the Python side, running the query on a worker thread and letting the main thread take the signal, which is what several other extensions do and which needs no new symbols at all, but which changes the threading model of every call and interacts badly with the GIL release in section 4.

The recommendation is the first, with the second as the fallback if the symbol lookup proves fragile across platforms. Either way this is now a piece of engineering with a design question in it rather than a checkbox, and it should be its own issue.

## 6. What is still unknown

**Free threaded Python.** There is no free threaded interpreter on this machine so nothing here was tested against one. The reason for optimism is section 2: the extension does not link libpython, so the usual ABI mismatch between a standard and a free threaded build does not apply to it. The reason not to assume is `Py_mod_gil`. A single phase initialised extension that does not declare `Py_MOD_GIL_NOT_USED` causes a free threaded interpreter to re-enable the GIL at import time with a warning, which would technically pass a test suite while silently defeating the entire point. Nothing in `PythonModuleBuilder` that these probes found offers a way to declare it. So the free threaded wheel in the M3 matrix has to assert on the absence of that warning and on `sys._is_gil_enabled()` being false, not merely on the tests passing.

**manylinux.** Most of what this paragraph used to say is now measured rather than unknown, because `tools/build_extension.sh` runs on both Linux legs of CI and the numbers are in the log. The runtime libraries are `.so` rather than `.dylib`, the rpath is set with `patchelf` rather than `install_name_tool`, and the token is `$ORIGIN` rather than `@loader_path`, all as expected. What was not expected is that Linux needs one fewer library and the whole set is smaller: `libAsyncRTMojoBindings` does not appear at all, and the self contained set is 2.02 megabytes on aarch64 and 2.12 on x86-64 against 2.9 on macOS arm64. So the macOS figure is the ceiling of the three rather than a representative one, and any wheel size threshold should be set against it.

Two things about Linux remain unmeasured and both belong to P8. `auditwheel` will want to either bundle or be told to exclude the vendored Mojo libraries, and it has not been run. And the glibc version the Mojo runtime requires is still unknown, which is the one item here that could turn into a real constraint on which manylinux tag firepanda can claim, since nothing in a CI job that builds and imports on the same runner would reveal it.

One difference in how the two platforms decide what to vendor is worth recording, because it is not a detail of this script so much as a difference in what the two formats say. A Mach-O dependency carries the reference the linker recorded, so `@rpath/libKGEN...` and `/usr/lib/libc++.1.dylib` are distinguishable by looking at them. An ELF `DT_NEEDED` entry is a bare name either way and carries no such distinction, so on Linux the question has to be asked of the system instead: a library the loader can already find in `ldconfig -p` is the system's, and only what is left is ours.

**The licence.** As above. Unanswered, and it is the only item in this document that cannot be resolved by writing code.

## 7. What this changes about the plan

Document 07 section 2 says the licensing question is the item to check first at M3, before writing any binding code. That was the right instruction to write when the technical feasibility was unknown, because a negative answer would have meant a completely different distribution strategy and a lot of wasted work. It is now half wrong. The technical answer is known, the cost is 2.9 megabytes, and none of the binding code changes depending on how the licence question is answered. What changes is whether the artifact may be published to PyPI. So the question should be asked now and publication should be gated on it, and the binding work should start immediately rather than waiting.

The second reordering is bigger. Document 07 treats the Arrow crossing as the centrepiece of the milestone and the error mapping and signal handling as two paragraphs of detail. The measurements invert that. The Arrow crossing is nearly free because M2 already built the hard part, and what is left of it is one struct and a wrapper. The error mapping is blocked on a binding layer behaviour that has to be worked around, and the signal handling is blocked on a missing symbol. Those two should be scheduled early, because they are the ones with unknowns left in them, and the Arrow work should be scheduled where it belongs, which is close to the front but not because it is risky.

The third is about the `BINDINGS` table. Document 07 section 3 argues for it on the grounds that Modular describes this API as early and expected to change. Everything in this document supports that argument and one thing sharpens it: Modular's own products do not use the ahead of time build path, and the one exception mechanism a binding layer needs is not reachable through the public builder. Both of those are the sort of thing that gets fixed upstream in a minor release, and when they are, the table is what makes adopting the fix a small change. The table should be built before the four hundred bindings, not after the first fifty of them have been written by hand.

## 8. Exit criteria, with what is already met

The criteria from document 07 section 8 and from the M3 tracking issue, with the status each one is actually in.

The example from document 04 section 2 running from a clean virtualenv with no Mojo toolchain present is **half met**. The mechanism is proven end to end on macOS arm64 with a stock Homebrew interpreter and no toolchain reachable. What is not done is the example itself, the wheel around it, or Linux.

Zero copy to pyarrow verified by buffer pointer identity is **met for the array protocol on one dtype**. The addresses match. It is not met for `__arrow_c_stream__`, which needs `ArrowArrayStream`, and it has not been checked for strings, where the export path is more interesting because a firepanda string column and an Arrow one are shaped differently.

Ctrl-C raising `KeyboardInterrupt` is **blocked**, on section 5.

The surface parity test is **not started** and depends on the `BINDINGS` table.

The error mapping table exercised by a test per row is **blocked**, on section 5, and the design that unblocks it is the Python layer.

The free threaded wheel passing the same suite is **untested**, and its test needs the `sys._is_gil_enabled()` assertion described in section 6 or it will pass without meaning anything.

Wheel size recorded in CI with a threshold is **not started**, and section 2 suggests the threshold should be set low, somewhere near four megabytes, because the measurements say a firepanda wheel has no business being larger than the runtime it vendors.

## How to run the probes again

The extension probe. Build with `mojo build --emit shared-lib -o mojo_module.so mojo_module.mojo`, run from inside the pixi project. Copy `libKGENCompilerRTShared.dylib`, `libAsyncRTMojoBindings.dylib`, `libMSupportGlobals.dylib` and `libAsyncRTRuntimeGlobals.dylib` from the pixi environment's `lib` directory to sit beside it. Rewrite the one rpath with `install_name_tool -rpath <the absolute pixi lib path> @loader_path mojo_module.so`. Import it with `env -i PATH=/usr/bin:/bin python3.14`, having put the directory on `sys.path`.

The Arrow probe. The same build with `-I` pointing at the firepanda tree, then a Python file that defines a class with an `__arrow_c_array__` method returning the capsule pair and passes an instance of it to `pyarrow.array`. Compare the address the Mojo side reports against `arr.buffers()[1].address`.

The signature probes. Call any `Python().cpython()` member with no arguments and read the compiler error, which prints the full declaration. A member that does not exist says so instead.
