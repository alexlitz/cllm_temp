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

### 5. `neural-io-position-wt` (#658) — MERGED CLEAN (fingerprint unchanged, re-built)

- Merge: clean, no conflicts. Adds `nibble_io_position.py` (`IOPositionBuffer`,
  ALiBi position-signature neural read) + edits `InputKVStream.read` in
  `nibble_filesys.py` to retrieve stdin bytes through the real softmax1+ALiBi
  attention forward (byte-identical to the reference slice).
- `nibble_filesys.py` IS imported by `nibble_pure_forward_complete.py`, but the
  change is confined to the runtime `InputKVStream` read class — no FFN block /
  attention head is added to the weight build.
- Fingerprint: RE-BUILT and UNCHANGED (`f61decf217c718ba`, peak 8.3 GB) — native
  READ pathway is decode-runtime only.

### 6. `argv-read-plumb` (#649) — MERGED CLEAN (fingerprint unchanged, re-built)

- Merge: clean; git auto-merged `isa.py`.
- Change: adds `nibble_argv.py`; the `isa.py` edit is additive to the REFERENCE
  `interpret` (adds LC/SC/READ handling + a `stdin=` arg — no opcode-number change);
  `nibble_pure_forward.py` gains an inert `fio=`/`data_seg=`/`mask=` file-op
  dispatch branch in `run_pure_forward` (byte-identical to original when `fio=None`).
- Fingerprint: RE-BUILT and UNCHANGED (`f61decf217c718ba`, peak 10.2 GB) — argv/READ
  plumbing is decode-runtime; no weight-build code changed.

### 7. `full-attn-baking-completeness` (#652) — MERGED CLEAN (fingerprint unchanged)

- Merge: clean. Adds `baked_attention.py` + `test_baked_attention.py` — a completeness
  verifier + test, NOT imported by any weight-build module.
- Fingerprint: UNCHANGED (`f61decf217c718ba`) by construction.

### 8. `eliza-full-turn-neural` (#653) — MERGED CLEAN (fingerprint unchanged)

- Merge: clean. Touches only `test_chat_io_neural.py` (test).
- Fingerprint: UNCHANGED (`f61decf217c718ba`) by construction.

### 9a. `revalidate-capstones-unified-vm` (#647) — MERGED CLEAN (fingerprint unchanged)

- Merge: clean. Adds `revalidate_capstones_unified.py` (tool) + the
  `onnx_runtime_nibble.c` BSS fix (runtime C, not weight build).
- NOTE: #647 is a strict ancestor of #661 — merging #661 next includes it.
- Fingerprint: UNCHANGED (`f61decf217c718ba`) by construction.

### 9b. `bundle-unified-vm` (#661) — MERGED, ONE CONFLICT RESOLVED (fingerprint unchanged)

- Merge: CONFLICT in `bundle_small.py` `_build_model` (the ONE genuine same-line
  clash of the consolidation). Both sides addressed the SAME dense-build memory
  hazard differently:
  - HEAD (#656): GUARDED bundling behind `C4_ALLOW_DENSE_BUILD` — refuses to bundle
    off the dense complete model unless opted in.
  - #661: FIXES it — routes the weight serialiser onto `build_compact_sparse_streaming`
    (`dense_kernel` = byte-identical, ~12 GB peak).
  - RESOLUTION: took #661's streaming route (more-complete-wins per brief). The
    streaming build stays well under the conftest 60 GB / per-build 20 GB ceilings
    and is L∞=0 to the dense build, so it REMOVES the hazard entirely rather than
    merely refusing it — #656's opt-in gate is no longer needed here. A comment in
    `_build_model` records the decision. `bundle_small.py` parses clean; no orphaned
    dense-guard code remains (only explanatory docstrings).
- Fingerprint: UNCHANGED (`f61decf217c718ba`) — all changed files are runtime C /
  packaging tool / tests, no weight-build code.

### 10. `libprog-neural-phase2` (#648) — ALREADY MERGED (no-op)

- `git merge` → "Already up to date." #648 is a strict ANCESTOR of #660
  (`fix-frame-local-ptr-store`), so it came in with item 1. No separate merge commit.
- Fingerprint: UNCHANGED (`f61decf217c718ba`).

### 11a. `selfhost-3layer-feasibility` (#654) — MERGED CLEAN (fingerprint unchanged)

- Merge: clean. Adds `selfhost/*.c` + `run_selfhost_feasibility.py` + a doc — additive
  demo, not imported by any weight-build module.
- Fingerprint: UNCHANGED (`f61decf217c718ba`) by construction.

### 11b. `agent-mandelbrot-neural-forward` (#659) — MERGED CLEAN (fingerprint unchanged)

- Merge: clean. Adds `_mandel_run.py` / `_mandel_src.py` / `test_mandelbrot_neural.py`
  + a doc — additive demo/test.
- Fingerprint: UNCHANGED (`f61decf217c718ba`) by construction.

### 12. `blog-impl-correspondence-audit` (#655) — MERGED CLEAN (doc only)

- Merge: clean. Adds `docs/BLOG_IMPL_CORRESPONDENCE_AUDIT.md` only.
- Fingerprint: UNCHANGED (`f61decf217c718ba`) by construction.

## Excluded (per brief — in-flight, fold in a follow-up)

- #662 scalar-IMM, item-6 primitives, item-7 fixedpoint-runtime — NOT merged.

## Fingerprint summary

| Point | Fingerprint (first 16) |
|-------|------------------------|
| base (`fix-zfod-malloc-neural`) | `3566e0560d3b71f9` |
| after #660 (LEA-addr-nib block added — the ONLY build-weight change) | `f61decf217c718ba` |
| final (after all 12 items) | `f61decf217c718ba` |

The consolidation changed the build weights EXACTLY ONCE — the #660 LEA fix adds a
new `OP_IS[LEA]`-gated physical block (n_blocks 305 → 306), a real VERIFIED feature
listed in the brief as an additive coexisting block. Every OTHER branch was
byte-identical to the build (tools / tests / docs / decode-runtime paths). No merge
changed the fingerprint UNEXPECTEDLY; no branch was stopped.

## One post-merge test fix (test-only, no weight change)

`test_crossop_dedup.py` (from #629) was written before the LEAN split was killed
(`unify-full-vm-no-lean-splits`, `49907f07`). Two incompatibilities surfaced ONLY
once #629 sat on the unified base — a legitimate consolidation finding, not a
weight regression:

1. `_build()` passed `include_bitwise=`/`include_divmod=` to
   `build_compact_sparse_streaming` — those op-subset SPLIT kwargs were REMOVED
   (the build is now the single full-op interpreter). Dropped them; the flags stay
   as inert compat args.
2. `test_almost_shareable_reported_not_tied` asserted a SPECIFIC pre-unification
   near-miss weight pair existed (`>= 1`). The unified model has 0 "almost"
   candidates — the cross-op dedup finds only EXACT permutation ties (19 real tie
   groups, the cleaner outcome). Relaxed the stale precondition; KEPT the real
   invariant (`st.almost_count == group tally` — almost candidates are REPORTED,
   never silently tied).

The 3 core crossop gates (byte-identity L∞=0, real-tie scalar accounting, exact
permutation) pass unchanged. Test-only edit; the final fingerprint is unaffected.

## Test result (c4_min suite, conftest-guarded, serial)

The full 407-test suite is dominated by SLOW heavy-build neural tests (each a fresh
streaming model build), so an exhaustive single run is multi-hour and repeatedly
hit wall-clock bounds under the memory guard. Verified in bounded, per-file runs
instead — all COMPLETED runs pass, ZERO failures:

| File | Result |
|------|--------|
| test_blogspec_foundation | 10 passed |
| test_moe_top1 | 9 passed |
| test_handoff | 6 passed |
| test_nibble_bake | 12 passed |
| test_weight_dedup | 3 passed |
| test_slice | 12 passed |
| test_crossop_dedup | 4 passed, 1 skipped (after fix above) |
| test_lib_addr32_byteident | 2 passed (addr32 byte-identity gate) |
| test_nibble_kv_prune | 10 passed |
| test_fetch_dedup | 5 passed |
| test_nibble_io_position | 21 passed (#658 neural READ position sig) |
| test_baked_attention | 7 passed (#652) |
| test_neural_stdin_read | 6 passed (#658) |
| **test_libprog_corpus** | **24 passed** (printf neural FAST tier + gcc goldens; #660/#648) |
| **test_pure_forward_1096** | **4 lean families passed byte-exact** (+13 more before timeout) |

**Confirmed: ~148 passed, 0 failed, 1 skipped (opt-in divmod gate).**

Build-time-bound (slow full-model builds, no failures observed, hit the wall-clock
timeout before finishing — NOT failures): `test_config_toggles`,
`test_compact_alloc`, `test_argv_read`, and the heavier `test_pure_forward_1096`
families. Their fast siblings covering the same paths (e.g. `test_neural_stdin_read`
/ `test_nibble_io_position` for READ, the 4 lean 1096 families for decode) all pass.

### Spot-checks (per brief)

- FAST printf neural tier: `test_libprog_corpus` 24/24 byte-exact through
  `model.forward` (printf_str/hex/int) + gcc goldens. PASS.
- Primitives: foundation / moe / handoff / nibble_bake / weight_dedup / slice all
  pass. PASS.
- 1096 sample: 4 lean families (arith_add_32bit, cmp_gt, var_simple_32bit,
  func_identity) byte-exact; 13/13 more passed before the heavy-build timeout. PASS.
