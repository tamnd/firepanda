# firepanda specification

Written August 2026, against Mojo 1.0 (Modular 26.5), pandas 3.0.5, Polars 1.43 and
Arrow 25.0.0. Start with [00-README.md](00-README.md), which is the real index and
says what was already decided and why.

| | |
|---|---|
| [00-README.md](00-README.md) | Index, settled decisions, prerequisites, honesty about scope |
| [01-research-2026.md](01-research-2026.md) | State of Mojo, pandas, Polars, Arrow and the Mojo ecosystem, with sources |
| [02-architecture.md](02-architecture.md) | Layers, memory, types, plan, optimizer, execution |
| [03-dtype-dispatch.md](03-dtype-dispatch.md) | Compile-time monomorphization and the generated dispatch table |
| [04-python-dx.md](04-python-dx.md) | The two front doors, and every deliberate divergence from pandas |
| [05-kernels.md](05-kernels.md) | Kernel shape, the hash table, strings, the GPU path |
| [06-pandas-parity.md](06-pandas-parity.md) | The full pandas 3.0 conformance checklist, milestone-tagged |
| [07-python-bindings.md](07-python-bindings.md) | `PythonModuleBuilder`, Arrow PyCapsule, wheels, and the ABI problem |
| [08-milestones.md](08-milestones.md) | M0 through M11, exit criteria, and four points to stop and reassess |
| [09-quality-bar.md](09-quality-bar.md) | Testing, and living without a race detector |
| [10-benchmarks.md](10-benchmarks.md) | What gets measured and against whom |
| [11-package-layout.md](11-package-layout.md) | The tree, and why Mojo 1.0's import rules decide it |
| [12-the-python-front-door-measured.md](12-the-python-front-door-measured.md) | Document 07 checked against a running toolchain, with what works, what does not, and what it cost |
| [13-the-bound-type-is-not-a-dataframe.md](13-the-bound-type-is-not-a-dataframe.md) | What `PythonTypeBuilder` cannot do, and why the pandas surface has to be a Python object |
| [14-errors-across-the-boundary.md](14-errors-across-the-boundary.md) | How one Mojo `Error` becomes the seven Python exception classes document 07 promised |
| [15-the-arrow-capsule-boundary.md](15-the-arrow-capsule-boundary.md) | How a frame reaches pyarrow and Polars with no copy, and who owns the memory afterwards |
| [16-reading-somebody-elses-arrow.md](16-reading-somebody-elses-arrow.md) | How a pyarrow, Polars or pandas frame becomes a firepanda frame, and what the real producers taught the checks |
| [17-the-column-you-can-hold.md](17-the-column-you-can-hold.md) | The bound `Series`, what `df[key]` does with each kind of key, how a column exports itself, and where the hand written members go |
| [18-making-a-frame-out-of-python.md](18-making-a-frame-out-of-python.md) | How `pd.DataFrame({...})` becomes an Arrow frame, what the inference decides, and where the pandas constructor signature lives |
| [engine/](engine/00-README.md) | How DuckDB and Polars execute a query, and what firepanda is going to do about it |

## How to read this

**Three claims carry the rest of the design.** That a kernel monomorphized over
`DType` is competitive with hand-written intrinsics without the tail bugs, that the
Arrow C Data Interface can be implemented in Mojo with genuinely zero copies, and
that a self-contained wheel can be built despite an unstable ABI. Documents 03, 02
and 07 respectively. If one of those is false, the milestone that depends on it is
where the plan changes.

Two of those three claims have since been checked against a running toolchain rather than argued for, and both held: the Arrow C Data Interface landed at M2 and its zero copy is asserted by pointer address in the test suite, and a self contained wheel was built and imported from an interpreter with no Mojo toolchain reachable. Document 12 is that check, and it also records the two things that turned out not to work.

**Claims marked [verify] are not confirmed.** Mojo reached 1.0 recently and the
standard library moved substantially on the way there, so any sample older than
August 2026 is likely to be wrong about names. Where a claim came from a secondary
source it says so. Check them against the toolchain you install.

**There are no time estimates.** Milestones are ordered by dependency and by risk.
Sizing them in weeks invites a reader to add the numbers up and treat the total as a
delivery date, and at this stage that number would be fiction.

**Conventions.** Documents refer to each other as "document NN section N", and CI
checks that those references resolve. Milestone tags `(M0)` through `(M11)` in the
parity checklist refer to headings in document 08, and CI checks that too.
