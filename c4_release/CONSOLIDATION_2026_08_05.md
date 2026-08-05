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

---

# Second consolidation pass — 2026-08-05 (faithfulness levers)

Folds the two session-VERIFIED **faithfulness** levers on top of
`consolidate-0.5b` @ `10991626`. Both change the **verify PATH** (how the
speculative accept/reject compare runs) — **not** the weights — so every
flag-OFF golden is re-verified byte-identical, and both flag-ON builds are
also byte-identical (build-invariant, runtime-only).

- **Base:** `consolidate-0.5b-2026-07-22` @ `10991626` (tip; carries the first
  consolidation pass above + the composed perf stack + CFM base).
- **Both levers based on** `e472b882`, already an ancestor of `10991626`.

## Golden gate (re-verified on CPU via `python -m c4_min._fingerprint_build`)

| Gate | Expected | Result |
|------|----------|--------|
| default (all flags OFF) | `069cc32f…` | **`069cc32f…` PASS** |
| CFM (`C4_PF_CFM=1`)     | `7d19cdc3…` | **`7d19cdc3…` PASS** |
| `C4_FAITHFUL_ATTN_EVICT=1` (path, not weights) | `069cc32f…` | **`069cc32f…` PASS** |
| `C4_DIRECT_CAM_VERIFY_ADDR=1` (path, not weights) | `069cc32f…` | **`069cc32f…` PASS** |

## Levers composed (this pass)

| Lever | Flag (default OFF) | Source commit | Merge commit | Flag-ON fingerprint |
|-------|--------------------|---------------|--------------|---------------------|
| Faithful attention over the EVICTED cache (real softmax1+ALiBi verify — the model's OWN query independently resolves every memory read's address+value; tractable via the exact-evict liveness schedule) | `C4_FAITHFUL_ATTN_EVICT` | `9d9b8ec0` (branch `worktree-agent-a34bdb10635c848f9`, ⊇ `e472b882`) | `601af2ab` | path-invariant (verify path only, no weight change) → `069cc32f` |
| O(1) INDEPENDENT address verify on the direct-CAM fast path (decode the model's OWN queried CAM address from its computed query sign-pattern; compare to the draft's resolved addr; O(1) per read, mismatch → terminal FAIL) | `C4_DIRECT_CAM_VERIFY_ADDR` | `aa85a9d2` (branch `worktree-agent-a20f69212b510615b`, ⊇ `e472b882`) | `aba99b8f` | path-invariant (verify path only, no weight change) → `069cc32f` |

### Files touched (this pass)

- `C4_FAITHFUL_ATTN_EVICT` (`9d9b8ec0`): `pf_speculative.py` (adds
  `faithful_attn_evict_enabled()` + forces the per-op scored verify path /
  exact-evict schedule / frozen-row-skip OFF when ON), `direct_cam_batched.py`
  + `direct_local_cam.py` (faithful mode OVERRIDES the draft-trust CAM levers
  to their genuine-scoring forms), `FAITHFUL_ATTN_EVICT_2026_08_05.md` (doc),
  `_agent_faithful_evict_{byteexact,doom_cost,repro}.py` (3 verify/probe
  scripts).
- `C4_DIRECT_CAM_VERIFY_ADDR` (`aa85a9d2`): `direct_cam_batched.py` (adds
  `verify_addr_enabled()`, `ResolvedTable.addr`/`code_addr`/`addr_sink`,
  `DivergenceSink`, `_decode_model_query_addr`, and the O(1) sign-decode
  address check in `_install_direct_forward`), `pf_speculative.py` (adds the
  `addr_sink` poll after each verify span → terminal `kind='cam_addr'` FAIL),
  `DOOM_FASTPATH_ADDR_VERIFY_2026_08_05.md` (doc),
  `_agent_verify_addr_experiment.py` (experiment script).

## Conflicts resolved (this pass)

- **Both levers edit `direct_cam_batched.py` and `pf_speculative.py`'s
  `verify_blocks`** — the expected conflict. They add DIFFERENT, non-adjacent
  verify hooks:
  - `direct_cam_batched.py`: `C4_FAITHFUL_ATTN_EVICT` inserts its override
    INSIDE `direct_cam_batched_enabled()` (before its `return`); the addr
    lever inserts a NEW `verify_addr_enabled()` function AFTER that return, plus
    the `ResolvedTable`/`DivergenceSink`/decode/forward additions. Git's `ort`
    strategy auto-merged cleanly (distinct regions). **Verified by inspection:
    BOTH mechanisms present and preserved** — `direct_cam_batched_enabled()`
    keeps the faithful override at lines 89-90, `verify_addr_enabled()` is
    intact at lines 94-121.
  - `pf_speculative.py`: `C4_FAITHFUL_ATTN_EVICT` gates the EARLY part of
    `verify_blocks` (`faithful_attn_evict_enabled()` def ~L901; per-op-path /
    exact-evict / frozen-skip forcing ~L977/L1413/L1540); the addr lever adds
    the `_dcam_tbl.addr_sink` poll LATER in the same function (~L2017-2064).
    Non-overlapping → auto-merged cleanly. **Verified:** `_dcam_tbl` is
    installed at L1480 (`install_direct_cam_batched`, which populates
    `addr_sink` only when `verify_addr_enabled()`) and read by the addr poll at
    L2023 — the two hooks coexist correctly.
- **Non-interference confirmed:** each mechanism is behind its OWN default-OFF
  flag. `C4_FAITHFUL_ATTN_EVICT=1` forces the direct-CAM levers OFF (so there
  are no direct-CAM reads for the addr-verify to inspect — the two paths are
  mutually exclusive by construction, not conflicting). Spot-checked:
  all four `*_enabled()` predicates return `False` when unset; the faithful
  flag forces `direct_cam_batched_enabled()`/`direct_local_cam_enabled()` to
  `False`; the addr flag toggles `verify_addr_enabled()` independently.

## Notes (this pass)

- Shared files already present on `10991626` from the first pass
  (`DOOM_FASTPATH_FAITHFULNESS_AUDIT_2026_08_05.md`,
  `SELFEMU_FASTPATH_TRANSFER_FINDINGS.md`, `_agent_selfemu_bigws.py`) were NOT
  re-added — both feature branches carried identical copies, so the merges were
  no-ops for those paths.

---

# Third consolidation pass — 2026-08-05 (faithful single-dispatch)

Folds the session-VERIFIED **faithful single-dispatch** lever
(`C4_FAITHFUL_SINGLE_DISPATCH`) + its pipelined/vectorized value-verify onto
the consolidate base `ec86ef34` (which already carries the composed perf stack
+ `C4_FAITHFUL_ATTN_EVICT` + `C4_DIRECT_CAM_VERIFY_ADDR` from the first two
passes). This lever changes the **verify PATH** (a genuinely-computing
single-dispatch path on the fast substrate) — **not** the weights — so every
flag-OFF golden is re-verified byte-identical and the flag-ON build is also
byte-identical (build-invariant, runtime-only).

- **Base:** `consolidate-0.5b-2026-07-22` @ `ec86ef34` (already carries the
  composed perf + `C4_FAITHFUL_ATTN_EVICT` + `C4_DIRECT_CAM_VERIFY_ADDR`).
- **Merged:** `186b1865` (branch `worktree-agent-a5a3a3f13335fb707`) =
  `C4_FAITHFUL_SINGLE_DISPATCH` + pipelined/vectorized value-verify. It
  descends from `bdb1028b`, which merged `9d9b8ec0` + `aa85a9d2` (the SAME two
  faithfulness levers `ec86ef34` carries, via a different merge path). Because
  both sides share the `9d9b8ec0`/`aa85a9d2` ancestry, git's 3-way merge base
  was `aa85a9d2` and the expected conflicts in `pf_speculative.py` /
  `precomputed_schedule.py` / `direct_cam_batched.py` **auto-resolved cleanly**.

## Golden gate (re-verified on CPU via `python -m c4_min._fingerprint_build`)

| Gate | Expected | Result |
|------|----------|--------|
| default (all flags OFF) | `069cc32f…` | **`069cc32f…` PASS** |
| CFM (`C4_PF_CFM=1`)     | `7d19cdc3…` | **`7d19cdc3…` PASS** |
| `C4_FAITHFUL_SINGLE_DISPATCH=1` (path, not weights) | `069cc32f…` | **`069cc32f…` PASS** |
| `C4_FAITHFUL_ATTN_EVICT=1` (re-verify, kept from pass 2) | `069cc32f…` | **`069cc32f…` PASS** |
| `C4_DIRECT_CAM_VERIFY_ADDR=1` (re-verify, kept from pass 2) | `069cc32f…` | **`069cc32f…` PASS** |

## Levers composed (this pass)

| Lever | Flag (default OFF) | Source commit | Flag-ON fingerprint |
|-------|--------------------|---------------|---------------------|
| Faithful single-dispatch: the genuinely-computing path ON the fast substrate — one real dispatch per step whose accept/reject GENUINELY re-derives the step (rejects the self-consistent wrong draft at BOTH the address `cam_addr` and value `cam_value` layers), with the value-verify **pipelined + vectorized** to close the last fps gap | `C4_FAITHFUL_SINGLE_DISPATCH` | `bdb1028b` + `186b1865` (branch `worktree-agent-a5a3a3f13335fb707`) | path-invariant (verify path only, no weight change) → `069cc32f` |

### Files touched (this pass)

- `faithful_single_dispatch.py` (the single-dispatch verify engine), plus the
  `C4_FAITHFUL_SINGLE_DISPATCH` gating + pipelined/vectorized value-verify hooks
  in `pf_speculative.py` / `precomputed_schedule.py` / `direct_cam_batched.py`.

## Conflicts resolved (this pass)

- **NONE required manual resolution.** The task anticipated conflicts in
  `pf_speculative.py` / `precomputed_schedule.py` / `direct_cam_batched.py`
  because both merge parents independently carry `9d9b8ec0` + `aa85a9d2`. But
  since those two commits are a COMMON ancestor of both sides, git's `ort`
  3-way merge base was `aa85a9d2` and every hunk auto-merged. **Verified by
  inspection after merge:** `faithful_single_dispatch.py` (494 L),
  `pf_speculative.py` (2657 L), `precomputed_schedule.py` (1768 L),
  `direct_cam_batched.py` (730 L) all present; `FAITHFUL_SINGLE_DISPATCH` logic
  intact in `pf_speculative.py`; value-verify pipeline intact in
  `precomputed_schedule.py`; ZERO conflict markers anywhere.
- **All levers preserved default-OFF + non-interfering:** the merged tree
  carries `C4_FAITHFUL_SINGLE_DISPATCH`, `C4_FAITHFUL_ATTN_EVICT`,
  `C4_DIRECT_CAM_VERIFY_ADDR`, and `C4_PF_CFM`; each `*_enabled()` predicate
  returns `False` when unset (proven by the default `069cc32f` golden), and
  every individual flag-ON build re-verifies byte-identical (`069cc32f`, CFM
  `7d19cdc3`).
