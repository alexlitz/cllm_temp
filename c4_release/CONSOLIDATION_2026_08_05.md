# Consolidation #832 — 2026-08-05

One clean branch composing all of the session's VERIFIED gated levers on top of
`consolidate-0.5b` (tip `cddf79f2`). Every lever is gated **default-OFF**; the
flag-OFF golden `069cc32f` is re-verified byte-identical after the full set.

- **Base:** `consolidate-0.5b-2026-07-22` @ `cddf79f2` (already carries the
  build-kill `6b5e15d8` + CFM base `8e6946e7`).
- **Consolidated tip:** see the "Deliverable" line at the bottom of the git log —
  fast-forward `consolidate-0.5b` to it.

## Golden gate (flag-OFF, run on CPU via `python -m c4_min._fingerprint_build`)

| Gate | Expected | Result |
|------|----------|--------|
| default (all flags OFF) | `069cc32f…` | see verification section |
| CFM (`C4_PF_CFM=1`)     | `7d19cdc3…` | see verification section |

## Levers composed

| Lever | Flag (default OFF) | Source commit | Merge commit | Flag-ON fingerprint |
|-------|--------------------|---------------|--------------|---------------------|
| GPU-vectorized schedule BUILD | `C4_SCHED_GPU_BUILD` | `9ac62e4a` (⊇ `e472b882` ⊇ `6b5e15d8`) | `c822a4d8` | build-invariant (runtime schedule only) → `069cc32f` |
| Double-buffer pipeline overlap | `C4_SCHED_PIPELINE` | `9ac62e4a` | `c822a4d8` | build-invariant → `069cc32f` |
| On-chip fused `[Dff,K]` hidden | `C4_FFN_FUSED_HIDDEN` | `9ac62e4a` | `c822a4d8` | build-invariant → `069cc32f` |
| Megablock block-K tiling | `C4_MEGABLOCK_BLOCK_K` | `9ac62e4a` | `c822a4d8` | build-invariant → `069cc32f` |
| Cache resolved schedule | `C4_SCHED_CACHE_RESOLVED` | `9ac62e4a` | `c822a4d8` | build-invariant → `069cc32f` |
| DRAWSPAN render-macro op (47) | `C4_DOOM_DRAWSPAN` | `586ed407` | `12846f46` | `2f69350f` (widens `num_ops_effective` → `NUM_OPS_DRAWSPAN=48`; recorded by `586ed407`) |
| Direct-CAM EMIT high-address JIT | `C4_CFM_EMIT` | `71cd3815` (⊇ CFM base `8e6946e7`) | `9cbdd5e0` | build-invariant (driver-serviced runtime op) → `069cc32f` |

### Already-present levers (baked into `consolidate-0.5b` @ `cddf79f2`, kept intact)

| Lever | Flag (default OFF) | Flag-ON fingerprint |
|-------|--------------------|---------------------|
| Code-from-memory bake | `C4_PF_CFM` | `7d19cdc3` |
| Native float ops (F_ADD..F_DIV, 40-43) | `C4_FLOAT_OPS` | widens `num_ops_effective` → `NUM_OPS_FLOAT=44` (ON hash not recorded here) |
| CFM EMIT opcode 45 base | `C4_CFM_EMIT` (base) | build-invariant → `069cc32f` |

### Docs (doc/probe only, no build-path or compute change)

| Doc | Source commit | Merge commit |
|-----|---------------|--------------|
| `c4_min/DOOM_FASTPATH_FAITHFULNESS_AUDIT_2026_08_05.md` | `6b2f53d0` | `cb0e4a64` |
| `c4_min/SELFEMU_FASTPATH_TRANSFER_FINDINGS.md` + `_agent_selfemu_bigws.py` | `68d0317a` | `09c26873` |

## Ancestry found (what superseded what)

- `9ac62e4a` **⊇** `e472b882` **⊇** `6b5e15d8`: `9ac62e4a` is a descendant of
  `e472b882`, which is a descendant of the build-kill `6b5e15d8` (already on
  `consolidate-0.5b`). Merging `9ac62e4a` alone brings the **whole** composed
  perf stack — `e472b882` did NOT need a separate merge.
- `71cd3815` **⊇** `8e6946e7` (CFM base, already on `consolidate-0.5b`). Merging
  `71cd3815` brings the direct-CAM EMIT path on top of the already-present CFM.
- `586ed407` (DRAWSPAN) is on a sibling doom-render lineage that lacked the CFM
  EMIT=45 opcode; the `isa.py` opcode registrations were unioned (see conflicts).

## Conflicts resolved

- **`c4_release/c4_min/isa.py`** (DRAWSPAN merge, `586ed407`): the opcode-block
  region conflicted because HEAD defines `EMIT = 45` (CFM lineage) while the
  DRAWSPAN side defines `DRAWSPAN = 47` / `NUM_OPS_DRAWSPAN = 48`. These are
  **DISTINCT** opcodes → resolved by **keeping BOTH**. Kept the EMIT=45 block and
  the DRAWSPAN=47/NUM_OPS_DRAWSPAN=48 block. Corrected the DRAWSPAN comment's
  stale "44/45 = BLIT/MEMCPY" opcode-map (a reference to a different branch where
  45 was MEMCPY) to note that on THIS branch **45 is EMIT**. Git had already
  auto-merged `drawspan_enabled()`, the composing `num_ops_effective()`
  (`max` over `NUM_OPS` / `NUM_OPS_FLOAT` / `NUM_OPS_DRAWSPAN`), and the NAMES
  dict (DRAWSPAN is registered at runtime via `doom_drawspan.register_opcode()`'s
  `isa.NAMES.setdefault`, so it correctly stays OUT of the static NAMES dict).
  No compute-code conflict; both levers preserved default-OFF.
- **`c4_release/c4_min/_fingerprint_build.py`** (DRAWSPAN merge): merged CLEANLY
  (no conflict) — the DRAWSPAN `fingerprint()` docstring note appends after the
  SHR/LC re-baseline paragraph, which is present on HEAD.

## Skipped / notes

- `e472b882` NOT merged separately — subsumed by `9ac62e4a` (ancestry above).
- `8e6946e7` / `6b5e15d8` NOT merged — already ancestors of `cddf79f2`.
- Float-ops flag-ON fingerprint is not re-recorded here (only its band-widen to
  `NUM_OPS_FLOAT=44` is documented in `isa.py`); out of scope for the flag-OFF gate.
