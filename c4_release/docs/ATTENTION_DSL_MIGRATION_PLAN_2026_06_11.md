# Attention-DSL Migration Plan — 2026-06-11

Closes the attention-DSL coverage gap that produced the operand-relay
transmission bug: imperative `attn.W_q/W_k/W_v/W_o[...] = X` projection
writes and imperative `attn.alibi_slopes[...] = X` / `.fill_(...)` slope
writes that live OUTSIDE the declarative
`DeclarativeAttentionHeadSpec` + `Primitives.generate_attention_head`
path, where nothing detects a slope collision at compile time.

Companion artifacts:

- `tools/lint_raw_attn.py` — the ratchet (per-file baseline; counts only
  go down). Run clean at baseline today.
- `tools/alibi_slope_collision_map.py` +
  `docs/ALIBI_SLOPE_COLLISION_MAP_2026_06_11.md` — the per-head slope
  collision diagnostic that catches the relay-bug class.
- `neural_vm/unified_compiler/decl_verifier.py::verify_alibi_consistency`
  — the EXISTING declared-slope checker (see §4: it is blind to
  imperative writes, which is the root of the gap).

---

## 1. The gap, precisely

The attention DSL covers Q/K/V/O structure
(`DeclarativeAttentionHeadSpec`, `AP`/`AO`,
`Primitives.generate_attention_head`). The spec carries an
`alibi_slope` field. **But `alibi_slope=None` means "the op writes
`attn.alibi_slopes[...]` itself imperatively."** Two failure surfaces:

1. **Imperative projection writes.** Ops still write `attn.W_*[...]`
   directly (mostly inside `RuntimeAttentionFragment` bake-fns for
   shape-dependent heads, plus the legacy imperative core).
2. **Imperative slope writes with no ownership pass.** Multiple ops can
   write the same physical `(block, head)` slope. The later write
   silently clobbers the earlier one. **There is no compile-time check
   that catches this** for imperative writes (§4).

### The bug this caused (Wall 2)

`make_layer10_residual_alibi_slopes_op` (`ops/alu_ops.py:1855`, phase
~999.1) does `attn10.alibi_slopes[0..4] = ...` on the physical block
that logical L10 occupies. An EARLIER op
(`layer9_step_end_operand_relay`, `layer9_lev_*_relay` in `ops/l9_ops.py`)
already set those same physical heads to relay-appropriate slopes
(0.2/0.5). The L10 op overwrites them (5/1/0.5/1), muting the operand
relay. See
`project_operand_gather_hybrid_encoding_is_cmp_alu_root.md` Wall 2.

The collision map confirms this exactly — physical **block 11 heads
0/1/3/4** are `CROSS-OP CLOBBER`s, each with an `l9_ops` relay writer at
0.2/0.5 then `alu_ops:1873-1883` overwriting (logical L10 → physical
block 11 after `_expand_wrapper_blocks`).

---

## 2. Inventory (as ratcheted by `lint_raw_attn.py`, 2026-06-11)

Total imperative attention writes: **2161** across **26 files** —
**468 in `ops/*.py`** (the active DSL-migration front) and **1693 in the
legacy imperative core** (`vm_step.py`, `compiler.py`, `setup_helpers_*`,
`weight_modules/function_calls.py`).

### Imperative ALiBi-slope sites (the high-value, low-volume target)

**13 files** write slopes imperatively (the task's "11 ops files" +
`compiler.py` + the legacy core fill/index sites). In `ops/` only:

| File | alibi `[i]=` | `.fill_` | Notes |
|------|-------------:|---------:|-------|
| `ops/model_ops.py`    | 13 | 4 | `residual_alibi_slopes`, `opcode_relay_head` — the L6/L8/L24 fill+override clusters |
| `ops/l10_ops.py`      | 11 | 0 | L10 head slopes |
| `ops/l9_ops.py`       |  5 | 1 | relay heads (collision VICTIMS of L10) |
| `ops/alu_ops.py`      |  5 | 0 | **`make_layer10_residual_alibi_slopes_op` — the clobber AGGRESSOR** |
| `ops/flag_gated_ops.py` | 4 | 0 | |
| `ops/l7_ops.py`       |  4 | 1 | `layer7_memory_heads`/`operand_gather` (block-8 clobbers) |
| `ops/l1_ops.py`       |  3 | 1 | `layer1_threshold_attn` (same-op fill+override, benign) |
| `ops/l8_ops.py`       |  2 | 0 | |
| `ops/l14_ops.py`      |  2 | 0 | also declares `alibi_slopes={...}` (see §3) |
| `ops/l15_ops.py`      |  3 | 0 | also declares `alibi_slopes={...}` |
| `ops/l2_ops.py`       |  1 | 1 | |
| `ops/l0,l3,l5,l13`    |  0 | 1 each | bare `.fill_` defaults |

### Imperative projection writes (the high-volume target)

Concentrated in shape-dependent / legacy files:

| File | W_* writes | Migratability |
|------|-----------:|---------------|
| `compiler.py` | 735 | legacy imperative core — Phase 6/7 cut |
| `vm_step.py`  | 537 | legacy imperative core — Phase 6/7 cut |
| `ops/l15_ops.py` | 231 | mostly `RuntimeAttentionFragment` (shape-dependent on `num_heads`, see §5) |
| `setup_helpers_l10/l5/l6/...` | ~340 | legacy setup helpers — Phase 7.C cut |
| `ops/l8_ops.py` | 79 | partly shape-dependent fragments |
| `ops/l14_ops.py` | 78 | partly migratable |
| `ops/l9/l4` | 27 | small static heads — migratable |

---

## 3. The slope-allocation / ownership pass (the structural fix)

The relay-bug class can ONLY recur because slope writes are unowned. The
fix is a **compile-time slope-ownership pass** keyed on
**physical `(block, head)`**:

### 3a. Make `Operation.alibi_slopes` authoritative

`Operation.alibi_slopes: Dict[head_idx -> slope]` already exists, but
today it is **Tier-B annotation metadata only** — it is validated for
type (`layer_compiler.py:1307`) and cross-checked by
`verify_alibi_consistency`, but the ACTUAL write is still the imperative
`attn.alibi_slopes[...] = v` in the bake. The two can (and do) drift.

Proposal: lower `Operation.alibi_slopes` to the *actual* write, the way
`DeclarativeAttentionHeadSpec.alibi_slope` already lowers in
`generate_attention_head` (`primitives.py:748`). Then delete the
imperative write in the same commit. One source of truth per op.

### 3b. The ownership/collision pass (extend `verify_alibi_consistency`)

`verify_alibi_consistency` already builds a
`registry: (layer, head) -> [(op_name, slope), ...]` and flags
`len(entries) > 1` as a `double_write`. Three fixes turn it into the
guard that would have caught the operand-relay bug:

1. **Key on physical block, not logical layer.** Its
   `_resolve_alibi_layer` returns logical layer indices; it then indexes
   `model.blocks[layer_idx]` (a PHYSICAL block) — a mismatch that
   produces the bogus `layer 18 head 0 declared 5.0 actual 0.5` /
   `out of range` notes observed today. Resolve to the physical block
   (the mapping `tools/alibi_slope_collision_map.py` derives by buffer
   identity, and `tools/probe_groundtruth.py` documents).
2. **Cover imperative writes.** Until §3a lands, the registry only sees
   ops that DECLARE `alibi_slopes={...}` (5 of ~14 slope-writing ops).
   The relay collision is invisible to it. The migration in §6 brings
   every slope writer into the declaration, at which point the
   `double_write` check covers the whole surface.
3. **Make value-changing cross-op double-writes a hard error** (not a
   report entry) once the surface is covered, mirroring the
   `AttentionHeadAllocator` "two ops on one head_idx is a hard error"
   contract. A redundant double-write (same value) can stay a warning.

### 3c. Bridge today: run the collision map in CI

Until §3a/§3b land, `tools/alibi_slope_collision_map.py --json` is the
ground truth (it instruments the *actual* writes, so it sees imperative
and declared writes alike). Gate CI on
`n_cross_op_clobbers == <known baseline>` so no NEW clobber lands.
Current baseline: **10 cross-op clobbers** (see the collision map doc).

---

## 4. Why the existing verifier missed the bug

`verify_alibi_consistency` reported **0 collisions** for the
operand-relay class. Root cause: it only inspects
`Operation.alibi_slopes` *declarations*. The relay victims
(`layer9_*_relay`) and the aggressor
(`make_layer10_residual_alibi_slopes_op`) write **imperatively** and
declare nothing, so they never enter the registry. It is blind to the
exact write pattern that caused the bug. `tools/alibi_slope_collision_map.py`
closes that by instrumenting the buffer writes themselves. This is the
"diagnostic that would have caught the bug" the task asked for.

---

## 5. `RuntimeAttentionFragment` inventory (shape-dependent vs migratable)

`RuntimeAttentionFragment` (`ir.py:510`) carries an imperative attention
bake-fn whose head set depends on the live `attn.num_heads` (e.g. L15
`memory_lookup`: 9 heads in the default 16-layer build, 14 in the
17-layer LEV build). The per-head `DeclarativeAttentionHeadSpec` shape
cannot express `num_heads`-conditional head presence, so these stay
imperative **for the projection writes** — but the SLOPE can still move
into the spec/declaration.

| Site | Shape-dependent? | Verdict |
|------|------------------|---------|
| `ops/l15_ops.py` `memory_lookup` (LI/LC/LEV/pop heads, `num_heads` 9 vs 14) | **Yes** — head presence gated on `num_heads` | Keep projections imperative behind the fragment; **lift the 3 slope writes** (`[:4]=0.05`, `[12]=1.0`, `[13]=1.0`) into the spec/`Operation.alibi_slopes`. Note: heads 12/13 ALREADY declare `alibi_slopes={12:1.0}`/`{13:1.0}` redundantly — delete the imperative twin. |
| `ops/l14_ops.py` `mem_generation` (8-head recency band + head 8) | Partial | Already declares `alibi_slopes={0..7:5.0}` and `{8:1.0}` — delete imperative `fill_(5.0)`/`[8]=1.0` twins (and fix the §3b layer-resolution bug that makes them read as `slope_mismatch`). |
| `ops/l8_ops.py` per-head recency (`[head]=0.5` in a Python loop) | No (static loop, fixed 8 heads) | **Migratable** to `Operation.alibi_slopes={h:0.5 for h in range(8)}`. |
| `ops/l9_ops.py` relay heads | No | **Migratable** — and the collision VICTIMS; migrate alongside §6 Wave 3. |

Net: of the imperative-attention surface, the **slope writes are
universally migratable** (even for genuinely shape-dependent heads — the
slope is per-head scalar metadata, not shape-varying structure). Only the
**projection writes** inside `num_heads`-branching fragments (chiefly
L15, partly L8/L14) are genuinely shape-dependent and stay imperative.

---

## 6. Wave plan

Ordered to (a) deliver the structural fix first, (b) avoid the
currently-hot files (`l7/l8/l9/l10`) until the smoke agents land, and
(c) ratchet `lint_raw_attn.py` downward each wave.

### Wave 0 — guard (this PR, tools/docs/tests only)
- `tools/lint_raw_attn.py` ratchet at baseline (2161 writes / 26 files).
- `tools/alibi_slope_collision_map.py` + the collision-map doc.
- CI gate on `n_cross_op_clobbers <= 10` (§3c). **No ops edits.**

### Wave 1 — slope ownership infrastructure (compiler only, NOT ops)
- §3a: lower `Operation.alibi_slopes` to the actual write.
- §3b: fix `verify_alibi_consistency` physical-block resolution; add the
  value-changing cross-op hard error (kept as a warning until the
  surface is covered). Touches `decl_verifier.py` + `layer_compiler.py`
  only — no ops/weight files.

### Wave 2 — benign / cold slope migrations (low-risk ops)
Migrate the imperative slope writes that are SAME-OP fill+override or
already have a declarative twin (delete the imperative twin):
`ops/model_ops.py` (L6/L24 clusters), `ops/l1_ops.py`,
`ops/l2_ops.py`, `ops/l14_ops.py`, `ops/l15_ops.py` slopes,
`ops/l0/l3/l5/l13` bare `.fill_`. Decrement each file's
`lint_raw_attn` baseline. **Avoids l7/l8/l9/l10.**

### Wave 3 — the relay-bug fix (AFTER smoke agents land l7/l8/l9/l10)
The crux. Resolve the **10 cross-op clobbers**:
- `block 11 (logical L10) heads 0/1/3/4`: decide ONE owner per head.
  Either L9 relay or the L10 residual op writes it — not both. Move the
  winning slope into that op's `Operation.alibi_slopes` and delete the
  loser's write. Re-run the collision map to confirm 0 clobbers there.
- `block 8 heads 1/3/5/6` (`l7_memory_heads` 5.0 vs
  `residual_alibi_slopes` 0.5): same — pick the owner.
- `block 6 heads 6/7` (`residual_alibi_slopes` fill(0) vs
  `opcode_relay_head` 5.0): the fill-to-0 then set-to-5 is an ORDER
  dependency across two ops; make `opcode_relay_head` the declared
  owner and stop `residual_alibi_slopes` from touching those heads.
Migrate `ops/alu_ops.py`, `ops/l7_ops.py`, `ops/l8_ops.py`,
`ops/l9_ops.py`, `ops/l10_ops.py` slope writes into declarations and
flip §3b's hard error ON.

### Wave 4 — projection-write migration (largest, lowest collision risk)
Migrate static projection heads in `ops/l4_ops.py`, `ops/l9_ops.py`,
small `ops/l8_ops.py`/`ops/l14_ops.py` heads to
`DeclarativeAttentionHeadSpec`. Leave genuinely shape-dependent L15 (and
the L8/L14 `num_heads`-branching) fragments imperative — these stay in
`RuntimeAttentionFragment`, with only their slopes lifted (Wave 2/3).
Each migrated head decrements the file's `lint_raw_attn` baseline.

### Wave 5 — legacy imperative core
`vm_step.py` (537), `compiler.py` (749), `setup_helpers_*` (~340),
`weight_modules/function_calls.py` (51). These are the Phase 6/7 cut
targets; the ratchet just tracks them so they can't grow.

---

## 7. Acceptance per wave

Every wave: `lint_raw_attn.py` exits 0 (baselines decremented, never
grown); `tools/alibi_slope_collision_map.py` shows
`n_cross_op_clobbers` non-increasing (Wave 3 drives it to 0); the smoke
gate (spec_k=0) is byte-identical or improved; `tests/test_lint_raw_attn.py`
passes. Wave 1+ additionally: `verify_alibi_consistency` reports 0
physical-block `double_write` collisions for value-changing cross-op
writes.
