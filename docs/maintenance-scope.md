# Maintenance scope: Rust-first

mokume ships its computation in two codebases — a **Rust compute kernel**
(`rust/crates/`) and a **pure-Python package** (`python/mokume/`, `pip install
mokume-py`) — which expose the same four computation commands with different
support levels: `features2proteins`, `features2peptides`, `peptides2protein`, and
`correct-batches`. This page states which implementation leads, what each is
maintained for, and where new work goes, so overlapping behavior does not drift.

!!! abstract "The policy in one line"
    **Rust is the leading implementation. The pure-Python computation package is
    added value — a readable implementation and compatibility baseline where
    covered — not the place new computation lands first.**

## What this covers

This page is about the **computation implementations only**: the quantification,
normalization, imputation, batch-correction, and differential-expression logic
behind the four commands above.

It does **not** govern the Python periphery — the Python pipeline API and its
shared post-processing, plotting, reporting, and
[TissueMap](periphery/tissuemap.md). The installable Mokume Plugin is an
interface over the default Rust-backed wheel, not an additional computation
implementation. See
[Architecture](architecture.md) and [Rust Wheel](rust-wheel.md).

## The rule

1. **The Rust kernel leads the native computation commands.** New behavior,
   supported options, and validation are defined there first. Where Python
   implements the same capability, it follows the shared public contract. This
   does not transfer ownership of the Python pipeline's orchestration and
   post-processing, or of intentional Python-only fallbacks, to Rust. The CLI
   command and Python API installed by the `mokume` wheel are two front doors
   onto the same kernel (see [Architecture](architecture.md)).
2. **New computation is written in Rust first.** When a feature touches the
   computation commands — for example [native SDRF + MSstats input
   support](https://github.com/bigbio/mokume/pull/74) — it is implemented and
   tested in the Rust I/O, CLI, and pipeline crates. It does **not** need a
   matching pure-Python change to ship.
3. **The pure-Python computation package is added value.** It is kept public and
   usable so that people can plug individual functions into Python-based
   pipelines and inspect readable implementations. A Python equivalent of new
   computation is built **only** when someone needs it for their pipeline or
   maintainers choose to expand compatibility coverage — and that can be done
   later.
4. **Parity is checked, not assumed.** Where both implementations and
   compatibility coverage exist, tests compare results within the documented
   floating-point tolerance. `rust/tests/test_rust_python_equivalence.py` covers
   selected `features2proteins` paths against frozen Python-generated golden
   outputs; it does not assert that every option is synchronized.

## Support table

Both codebases implement the four computation commands today. The table records
which implementation leads and how each is maintained.

| Computation command | Rust kernel (`rust/`) | Pure-Python package (`python/mokume/`) |
| --- | --- | --- |
| `features2proteins` | ✅ Leading — authoritative | ✅ Added value · parity-checked where covered |
| `features2peptides`  | ✅ Leading — authoritative | ✅ Added value · best-effort |
| `peptides2protein`   | ✅ Leading — authoritative | ✅ Added value · best-effort |
| `correct-batches`    | ✅ Leading — authoritative (native ComBat) | ✅ Added value · best-effort |

Legend:

- **Leading — authoritative:** new computation lands here first; defines correct
  behavior, options, and validation.
- **Added value · parity-checked where covered:** maintained as a usable
  implementation; documented overlapping paths are covered by automated
  compatibility tests against the Rust kernel.
- **Added value · best-effort:** kept public and usable; updated when a user
  needs the function in a Python pipeline or maintainers choose to expand
  compatibility coverage, not as a release gate.

### Distribution selection

There is no runtime backend selector. Install `mokume` for the leading Rust
kernel or `mokume-py` for the separately maintained pure-Python implementation.
Both distributions provide the `mokume` import package and console command, so
use them in separate environments rather than installing both together.

## Practical guidance

- **Contributing computation changes?** Implement them in the Rust crates and
  their tests. A pure-Python counterpart is optional and can follow later.
- **Need a computation function in a Python pipeline?** Prefer the `mokume`
  wheel (`mokume.features2proteins(...)`, `mokume.peptides2protein(...)`, …),
  which runs the Rust kernel in-process. The pure-Python package remains
  available when you specifically want the readable Python implementation or to
  compose individual functions.
- **Filing an issue about behavior differences?** Treat the Rust kernel's result
  as the reference and the pure-Python result as the one to reconcile.
