# Lowering cut 2026-07-13 — vm_step oracle-mirror move + dual cache-key snapshot merge

Two land-ready, byte-identical LOC cuts toward the <8K model-build core goal,
from the "reducible NOW" list in `LOWERING_SIMPLIFICATION_2026_07_13.md`.

- Base: main `1820c97a` (`Flip C4_MUL_B1_DELIVERY default-ON`).
- Golden gate: `tools/_isa_golden_hash.py` DEFAULT state_dict_sha256
  = `9dda2af1f81e7c96db24a117141a609cc4177f02bc578c40d5e654aa6d6cc809`.
- Golden captured before ANY change, and re-verified UNCHANGED after each cut
  and on the final combined branch. Both cuts are pure byte-identity refactors:
  the moved mirrors and the cache-key snapshot are NOT part of the built weights.

Branch: `lowering-cut-2026-07-13` (commits `a856aac1`, `c425c182`). DO NOT MERGE
was honored — the branch is left for review.

---

## Cut #1 — move the 17 `vm_step._set_layer*` oracle mirrors to `tests/oracles/`

### Why they are safe to move
The 17 imperative `_set_layer*` bake helpers in `neural_vm/vm_step.py` are
**test-oracle-only** — dead on the model-build path:

- **settrace evidence:** a `sys.settrace` over a full
  `compile_full_vm_dynamic(disk_cache=False)` build fires **0 of 17**. The sole
  production bake authority is the declarative op factories collected in
  `compile_full_vm_dynamic`; these helpers are never called during a build.
- **grep evidence:** zero actual call sites in `neural_vm/` build code (only
  docstring/comment references). Every real caller lives under `tests/` or
  `tools/` and uses them as the *legacy "expected"* side of the
  declarative-bake parity tests (`test_declarative_ffn_bakes_l*`,
  `test_declarative_attention_specs`, per-op L8/L13/L15/L16 tests).
- **golden evidence:** `tools/_isa_golden_hash.py` is byte-identical with the
  helpers present or absent.

The 17 functions: `_set_layer3_ffn`, `_set_layer4_pc_relay`, `_set_layer4_ffn`,
`_set_layer6_attn`, `_set_layer6_routing_ffn`, `_set_layer6_relay_heads`,
`_set_layer7_memory_heads`, `_set_layer8_sp_gather`, `_set_layer8_multibyte_fetch`,
`_set_layer8_alu`, `_set_layer8_multibyte_routing`, `_set_layer9_alu`,
`_set_layer9_marker_suppress`, `_set_layer10_alu`,
`_set_layer15_memory_lookup_heads_0_3`, `_set_layer15_memory_lookup`,
`_set_layer16_lev_routing`.

Dependency graph (via bytecode LOAD_GLOBAL + in-function import scan): the only
module-level references are `INSTR_WIDTH`/`PC_OFFSET` (from `neural_vm.constants`),
the intra-set call `_set_layer8_multibyte_routing -> _set_layer8_alu`, and three
in-function relative imports (`.constants`, `.unified_compiler.primitives`,
`.unified_compiler.ops.l15_ops`). `_SetDim` is passed in as the `BD` argument
(never referenced as a module global), so it stays in `vm_step`.

### What moved
- New module `tests/oracles/vm_step_layer_bakes.py` (new package
  `tests/oracles/`, following the existing `tests/_l6_legacy_bake.py` precedent:
  "NOTHING here is imported by the compiler; it lives entirely under `tests/`").
  The 17 helpers were moved **verbatim** (source-identical, verified by
  `inspect.getsource` diff), except the three in-function `.`-relative imports
  were rewritten to absolute `neural_vm.*` so they resolve from the new location.
- Updated the ~18 test/tool importers to import the moved helpers from the oracle
  module, keeping `_SetDim` + any non-moved helpers (e.g. `_set_layer5_fetch`,
  `_set_opcode_decode_ffn`, `_set_layer9_lev_*`) on their `vm_step` import.

Files touched: `tests/oracles/{__init__,vm_step_layer_bakes}.py` (new) +
`vm_step.py` + 16 test files + `tools/verify_l8_alu_migration.py`.

### LOC
`neural_vm/vm_step.py`: **7914 -> 3956 lines (-3958)** of test-oracle-only
weight-writing code removed from the model-build core (bodies + orphaned section
banners + collapsed blank runs). The bodies now live under `tests/`.

### Verification
- Golden state_dict_sha256 **UNCHANGED** (`9dda2af1...`) before/after.
- All affected test files re-collect and import cleanly.
- Parity/per-op failure set is **IDENTICAL to pristine base `1820c97a`**: 0 new
  failures, 0 accidentally-fixed. (The declarative-bake suites carry ~52
  pre-existing bake-assert failures + 25 collect-ImportErrors that reproduce on
  base unchanged — the golden hash, not these asserts, is the authoritative
  byte-identity gate.) One transient regression during development —
  `_set_layer4_ffn` / `_set_layer15_memory_lookup` hit
  `ModuleNotFoundError: tests.oracles.unified_compiler` from the copied
  `.`-relative imports — was fixed by rewriting those three imports to absolute
  `neural_vm.*`; both tests then pass and the failure set matches base exactly.

---

## Cut #2 — merge the dual cache-key snapshots + fix a live drift bug

### The bug
`neural_vm/unified_compiler/full_vm_compiler_dynamic.py` carried **two ~600-line
kwargs snapshots** that key the compile cache:

- `_inproc_snapshot` — the in-process memo key, in `compile_full_vm_dynamic`.
- `kwargs_snapshot` — the on-disk cache key, in `_bake_from_scheduled_ops`
  (hashed by `_static._cache_key`).

They were 99% identical (88 shared keys, **0 value-diffs**, verified by AST diff)
but had **DRIFTED** — a cache-collision hazard where a flag registered on only one
snapshot lets that cache layer hand back a **stale model** when the flag is
toggled:

| flag                    | was in            | missing from        | stale-serve risk |
|-------------------------|-------------------|---------------------|------------------|
| `C4_L15_LOOKUP_CMP_VETO`| memo (`_inproc`)  | disk (`kwargs`)     | disk cache       |
| `C4_STACK0_NEXT_ARITH`  | memo (`_inproc`)  | disk (`kwargs`)     | disk cache       |
| `C4_ADDSUB_DECLARATIVE` | disk (`kwargs`)   | memo (`_inproc`)    | in-process memo  |
| `C4_L10_ENT_AXCARRY`    | disk (`kwargs`)   | memo (`_inproc`)    | in-process memo  |

All four are bake-affecting flags (their doc-comments say "ON/OFF builds must
NEVER share a memo/disk entry"), so the drift could silently serve wrong weights.

### The fix
Extracted a single module-level `_build_cache_key_snapshot(**kwargs)` as the sole
source of truth (placed next to `_inproc_cache_key`). Its body is the **union** of
the two snapshots = **92 keys** (the 88 shared keys with their byte-identical
values, plus all four formerly-drifted keys). Both call sites now call it. Since
both key hashers (`_inproc_cache_key` and `_static._cache_key`) use
`json.dumps(..., sort_keys=True)`, dict order in the builder is irrelevant.

### LOC
`full_vm_compiler_dynamic.py`: **4154 -> 3599 lines (-555 net)** (two ~600-line
inline dicts removed; one ~620-line builder + two ~22-line keyword call sites
added).

### Verification
- Golden state_dict_sha256 **UNCHANGED** (`9dda2af1...`). The snapshot is a cache
  KEY only, never part of the built weights (golden builds with
  `disk_cache=False`, where `_inproc_snapshot` is `None` and the disk key is
  skipped).
- **Drift fixed:** toggling each of the 4 formerly-drifted flags now flips BOTH
  the disk cache key AND the in-process memo key (proven directly).
- **Builder = 92 keys**, no duplicates, contains all 4 drift keys + `__dynamic`.
- **In-process memo still works** end-to-end: second `compile_full_vm_dynamic`
  call is 0.00s and returns the same object (memo hit through the new builder).
- Cache/staleness/preset test failures are **IDENTICAL to base `1820c97a`** — the
  `test_staleness_invariants` cross-file-pollution failure, the two qwen-preset
  crashes, and the legacy strict-mode test all reproduce on base unchanged.

---

## Combined result

- `vm_step.py` model-build core: **-3958 LOC** (bodies relocated to `tests/`).
- `full_vm_compiler_dynamic.py`: **-555 LOC** + a live cache-collision bug fixed.
- DEFAULT golden `9dda2af1...` UNCHANGED across both cuts and the combined branch.
- Both cuts are byte-identical and land-ready toward the <8K core goal.

### Note on the working-tree base
The shared worktree was found checked out at `3317df90` (an unmerged
`C4_DERIVE_CMP_CLEAN` feature commit on top of `1820c97a`, byte-golden-identical
because that flag is DEFAULT-OFF). This branch was reset to the stated base
`1820c97a` and carries ONLY the two cuts above — the unrelated
`C4_DERIVE_CMP_CLEAN` work (touching `l10_ops.py`, `shared.py`, and a new
`tools/derive_cmp_clean.py`) is NOT included here.
