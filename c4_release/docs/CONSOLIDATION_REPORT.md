# c4_min Consolidation Report — `c4min-consolidated-2026-07-20`

Consolidates this session's VERIFIED c4_min feature branches into one canonical
branch, byte-identity-gated. `main` is untouched (stays at `e45eb4ff`); nothing
is merged to main.

## Base

- Branch created from `fix-zfod-malloc-neural` (`8426cad5`), the tip of the
  verified trunk chain:
  `49907f07` (unify-full-vm-no-lean-splits) →
  `cc16ceed` (lib-into-unified-model, ported lib + addr32) →
  `8426cad5` (fix-zfod-malloc-neural, malloc-family byte-exact + eviction-default).
  Chain confirmed linear; all three descend from `main` at `e45eb4ff`.

## Byte-identity gate

Fingerprint = deterministic sha256 over the DENSE-reconstructed per-block weights
(fp64, sorted order) of the memory-safe streaming sparse build
(`build_compact_sparse_streaming`, peak ~10-11 GB RSS). Tool:
`c4_min/_fingerprint_build.py`. The full dense build is a 54-117 GB hazard; the
streaming build is byte-identical (L∞=0, `dense_kernel`) and stays under 20 GB.

| Stage | Fingerprint (sha256, first 16) | Note |
|-------|--------------------------------|------|
| base `fix-zfod-malloc-neural` (`8426cad5`) | `3566e0560d3b71f9` | reference |

## Per-branch log

### 1. `fix-frame-local-ptr-store` (#660) — MERGED CLEAN (fingerprint changed, EXPECTED)

- Merge: clean, no conflicts. `--no-ff` merge commit.
- NOTE: #660 is a strict descendant of #648 `libprog-neural-phase2` — merging it
  ALSO brings in the entire libprog corpus (item 10 below becomes a no-op).
- Build-path change: adds a NEW LEA-gated physical block `lea-addr-nib` (after
  `ax-nib-split`, before `ax-byte-nib`) that recomputes the frame-address nibbles
  from `BP_LOW+4*imm` with a round-to-nearest integer decode. Gated on
  `OP_IS[LEA]` → writes nothing on every non-LEA op.
- Fingerprint: `3566e0560d3b71f9` → `f61decf217c718ba` — **CHANGED, and this is
  EXPECTED**: a new physical block is added (n_blocks 305 → 306), so the
  state_dict necessarily differs. This is a real, VERIFIED functional fix
  (9/9 BP-low-byte LEA sweep, no regression per #660), listed explicitly in the
  consolidation brief as an additive coexisting block — NOT a byte-identity-neutral
  change. Functional non-regression is confirmed by the final test suite.
- New expected baseline from here: `f61decf217c718ba`.

### 2. `c4min-crossop-dedup` (#629) — MERGED CLEAN (fingerprint unchanged)

- Merge: clean, no conflicts. Adds `crossop_analyze.py`, `crossop_dedup.py`,
  `measure_crossop_dedup.py`, `test_crossop_dedup.py` — read-only analysis tools +
  a test. No weight-build file touched.
- Fingerprint: UNCHANGED (`f61decf217c718ba`) by construction — pure tool/test
  additions, none in the weight-build import graph.

### 3. `vanilla-exec-round-removal` (#657) — MERGED CLEAN (fingerprint unchanged)

- Merge: clean, no conflicts.
- Change: swaps the runtime `_requantize` (`nibble_bake`/`nibble_compiler`/
  `nibble_handoff`/`universal`) from `torch.round` to a new additive
  `compiler.vanilla_requantize` (`argmax_v (2vx − v²)` LM-head snap, argmax==round
  on the exact-integer VM state, residue-immune, no `torch.round`). This is the
  EXECUTION/decode path, NOT the weight build.
- Fingerprint: UNCHANGED (`f61decf217c718ba`) — verified none of the four changed
  runtime modules are in the weight-build import graph (`build_compact_sparse_streaming`
  → `nibble_pure_forward_complete` does not import them); `compiler.py` change is a
  pure function addition. This is the "#657 argmax==round" known-neutral case.

### 4. `fix-dense-build-memory-hazard` (#656) — MERGED CLEAN (fingerprint unchanged)

- Merge: clean; git auto-merged `test_exec_path_vanilla.py` (disjoint regions vs #657).
- Change: adds `conftest.py` (autouse RSS watchdog, machine-safety hard-abort at
  60 GB, 120 GB dense opt-in) + `_build_guard.py` (per-build 20 GB assertion) +
  memory-safe guards in `bundle_small.py`/`chat_eliza.py`/`_probe_chat_io.py`/
  `_top1_measure.py` + test routings to the streaming build. No weight-build code.
- Fingerprint: UNCHANGED (`f61decf217c718ba`). The conftest guard now protects all
  subsequent builds (landed early per brief).
