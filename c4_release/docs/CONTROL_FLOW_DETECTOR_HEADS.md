# Control-Flow Detector Attention Heads (design spike)

_Phase 8 design spike — NOT YET IMPLEMENTED. Read-only. Companion to
[`PHASE_8_PLAN.md`](PHASE_8_PLAN.md) Stream 1 (8.A) and the SCC=0 audit
at `c4_release/.agent-logs/scc_zero_audit.md`._

## 1. The pattern

A **control-flow detector head** is a declarative attention head (one
`DeclarativeAttentionHeadSpec` in the L8 attn block) whose Q matches
the current instruction-step row, whose K matches the **previous**
instruction-step's opcode-marker row in the KV cache, and whose V/O
projects the saved control-flow payload (return PC bytes, saved BP,
saved SP, branch target) into a fresh set of residual slots
(`PC_VIA_LEV_DETECTOR`, `BP_VIA_LEV_DETECTOR`, …) that the downstream
L8-L17 ops then read instead of `PC`/`BP`/`SP` directly. This
**replaces** (a) the `_PREV_STEP` dim-alias trick in
`ops/shared.py:743-883` and `dim_registry.py:610-882` (which is a
label trick — same numeric slot, different name, fooling Tarjan) and
(b) the proposed `requires["next_step_after"]` IR primitive from
`scc_zero_audit.md` §6 wave 3. It is V2-aligned because the cross-step
edge is now **expressed in the model weights** (an attention head that
literally looks back one instruction step) rather than in the
scheduler ("trust me, this back-edge is fine"). The dep graph sees no
back-edge because the detector's `reads={…, MARK_AX, HAS_SE, OP_LEV}`
are step-local while its writes are a fresh dim family — pure DAG.

## 2. LEV detector head — concrete spec

### 2.1 Semantic

C4's `LEV` opcode at step N (currently materialized by
`layer16_lev_routing` in `ops/l16_ops.py:1604-1702`) executes
`PC := saved_return_PC; BP := saved_BP; SP := BP + 1`. The current
implementation reads `MARK_PC`, `MARK_BP`, `MARK_SP`, `TEMP+0..31`
(staged return-PC bytes), `ADDR_B0_LO/HI+0..15` (saved BP nibbles
fetched at L15 memory_lookup) and writes them into `OUTPUT_LO` /
`OUTPUT_HI_THIS_STEP` at the SP/PC/BP marker rows _of step N_
(`l16_ops.py:101-200` for the `_layer16_lev_routing_rules` head; see
the SP=BP+16 family at lines 127-160 and the PC=TEMP family at
lines 162-205). At step N+1, the OUTPUT residual at those rows is read
back through the KV cache by L5 fetch (PC) and L8 sp_gather_bake (SP)
to drive the next instruction.

The dep graph today sees: `L9 alu` reads `ALU_LO/AX_CARRY_LO` →
`L16 lev_routing` writes `ALU_LO` → cycle. Plus the `requires={"after":
"layer16_lev_routing"}` declared at `l9_ops.py:1185` and
`l8_ops.py:1199` is a deliberate next-step edge that Tarjan can't
distinguish from a real one.

### 2.2 Detector head spec

Add **one head** at L8 attn (it has to fire *before* L9 alu so its
output can replace the `requires["after"]=layer16_lev_routing` dep at
`l9_ops.py:1185`). Use the existing `make_layer8_head6_ax_carry_refresh_op`
pattern (`l8_ops.py:1872-2001`) as the structural template — it is
already a "look back at prev-step OUTPUT via attention" head.

| Component | Detail | Mirrors |
|---|---|---|
| `head_idx` | allocated by `AttentionHeadAllocator` at L8 (pin=None) | `_L8_HEAD_LAYOUT_BY_NAME` |
| Q (current row) | `AP(0, MARK_AX, L)` + `AP(0, HAS_SE, L)` + `AP(0, CONST, -1.5L)` (fires at current AX marker on non-first steps) | `l8_ops.py:1915-1917` |
| Q gate slot | `AP(GATE, MARK_AX, L) + AP(GATE, CONST, -L/2)` (anti-leakage at non-AX rows) | `l8_ops.py:1929-1931` |
| K (target = prev-step opcode-marker row) | `AP(0, OP_LEV, L)` + `AP(0, MARK_PC, L)` + anti-current-opcode gates (`AP(j, OP_LEV, -L)` on current row's OP_LEV) | `l8_ops.py:1933-1949` (same anti-OP_* mechanism) |
| V slots 1..16 → output `PC_VIA_LEV_DETECTOR_LO` | `AP(base+1+k, TEMP+k, 1.0)` (saved return-PC LO from prev step's L16 staging) | `l8_ops.py:1921-1923` (V/W_v identical shape) |
| V slots 17..32 → output `PC_VIA_LEV_DETECTOR_HI` | `AP(base+17+k, TEMP+16+k, 1.0)` | same |
| V slots 33..48 → output `BP_VIA_LEV_DETECTOR_LO` | `AP(base+33+k, ADDR_B0_LO+k, 1.0)` | `l16_ops.py:144-160` (saved-BP gather) |
| V slots 49..64 → output `BP_VIA_LEV_DETECTOR_HI` | `AP(base+49+k, ADDR_B0_HI+k, 1.0)` | same |
| O writes | `AO(PC_VIA_LEV_DETECTOR_LO+k, base+1+k, 1.0)` etc. at the current AX marker query position | `l8_ops.py:1925-1927` |

### 2.3 New residual dims

Allocate four 16-wide bands in the residual stream, registered in
`dim_registry.py` alongside the existing PREV_STEP block at lines
743-883. Estimated 64 fresh slots total (4 × 16-wide). The d_model=512
budget has room (`dim_registry.py:1078-1100` shows the compact-layout
gap above 698). Names:

* `PC_VIA_LEV_DETECTOR_LO` (16)
* `PC_VIA_LEV_DETECTOR_HI` (16)
* `BP_VIA_LEV_DETECTOR_LO` (16)
* `BP_VIA_LEV_DETECTOR_HI` (16)

(`SP_VIA_LEV_DETECTOR` is unnecessary: L16's SP=BP+16 is a 4-bit add
on `BP_VIA_LEV_DETECTOR_LO` that L9 alu can do inline.)

### 2.4 Which existing reads it replaces

Per the audit's §3.6 "7-op sealed loop", three back-edges retire:

1. **`l9_ops.py:1158`**: `reads={..., "ALU_LO", "AX_CARRY_LO"}`. After
   the detector, L9 alu reads `PC_VIA_LEV_DETECTOR_LO` for the
   instruction-fetch path during LEV-following-step, falling back to
   the L7 operand_gather ALU_LO write for non-LEV-following steps.
   The `requires={"after": "layer16_lev_routing"}` at `l9_ops.py:1185`
   is **deleted** — the cross-step edge is now in the head, not the
   scheduler.
2. **`l8_ops.py:1707`**: `reads={..., "CMP_PREV_STEP"}` +
   `requires={"after": "layer9_alu"}` (line 1735) on
   `layer8_sp_gather_bake`. The SP=BP+16 path on LEV-following step
   reads `BP_VIA_LEV_DETECTOR_LO`; CMP_PREV_STEP is unchanged (still
   covers BZ branch resolution, not LEV).
3. **`l16_ops.py:1641-1664`** (`layer16_lev_routing.reads`): the
   `TEMP`, `ADDR_B0_LO/HI` reads stay (this op is the step-N
   _writer_), but the implicit step-N+1 dependence on its own output
   (declared as the `requires["after"]: layer15_memory_lookup` chain)
   is no longer load-bearing.

L14 `addr_key_neural_decode` and L15 `memory_lookup` / `store_stack0`
do **not** change — they participate in the loop only because LEV
reads ADDR_B0_LO that they write same-step. With the detector, those
reads stay same-step; only the L16→L9 back-edge breaks.

### 2.5 Byte-identity strategy

The clean option is **add the head and the dim-band, but don't yet
remove the existing OUTPUT_LO/HI_THIS_STEP writes at L16 LEV**. Both
paths fire; downstream readers (L9 alu, L8 sp_gather_bake) prefer the
detector dim when present (`PC_VIA_LEV_DETECTOR_LO` ≠ 0 implies
prev-step was LEV). Until we explicitly delete the L16 writes, every
bake produces the same OUTPUT bytes as today **plus** the new
detector dims — a strict superset. `compile_full_vm()` weight hash
changes (new attention head weights, new W_o rows into fresh dim
slots), so this is **not** byte-identical at the model level. Hash
gate via `compare_symbolic_to_lowered_attn` for the new head only;
update `tests/test_compile_determinism.py` baseline hash in the same
commit.

A truly byte-identical migration is possible if the detector's V/O
math reproduces the LEV path's existing OUTPUT writes _exactly_, then
the L16 `_layer16_lev_routing_rules` SP=BP+16 and PC=TEMP families
(`l16_ops.py:144-205`) can be deleted in a follow-up. Estimated value:
not worth the algebra — the controlled-diff path is cleaner and the
sentinel-mode gates in `PHASE_8_PLAN.md` §8 are the real acceptance
test, not weight-hash equality.

## 3. Generalization table

| Sub-cycle (from `scc_zero_audit.md`) | Apply detector? | Rationale | Notes |
|---|---|---|---|
| §3.6 7-op LEV→ALU loop | **YES — LEV detector** (this doc) | The whole loop is sealed by a true cross-step edge | Highest leverage: retires 7 ops + the `requires_after` IR primitive |
| §3.7 3-op L10 carry trio (`carry_relay` ↔ `carry_relay_bake` ↔ `l10_post_ops_combined`) | **NO — structural collapse** | The 3-cycle is a Phase 7 migration artifact (bake/non-bake doublet sharing one head, see `l10_ops.py:1882-1908`). Deleting `layer10_carry_relay` (the non-bake variant; its `bake_fn` is `return None`) breaks the cycle for free | Per audit §6 wave 3 option (a). 1-op removal. Detector head would be over-engineered |
| §3.8 2-op L5 fetch dep_anchor | **NO — drop writes** | `_layer5_fetch_dep_anchor.writes` is a label trick (`l5_ops.py:388-393`). Audit recommends emptying it (§6 wave 3 step 8). 1-line. | Detector inappropriate: there's no semantic "prev-step opcode = X" trigger; it's a topology placeholder |
| §3.1 L3 EMBED_LO cycle | **NO — PREV_STEP rename** | EMBED_LO is byte-data carried in residue, not a control-flow boundary. Audit wave 2 step 3 fits the existing `OUTPUT_LO_PREV_STEP` precedent | 1-line per reader site |
| §3.2 L6 routing OUTPUT_HI_THIS_STEP (24 back-edges) | **NO — PREV_STEP rename** | Same as §3.1: data path, not control flow. Audit wave 1 step 1 | The single largest leverage rename; not detector-shaped |
| §3.3 L7 OUTPUT_HI (8 back-edges) | **NO — PREV_STEP rename** | Same; audit wave 1 step 2 + `docs/B9_OUTPUT_HI_SPLIT_SPEC.md` | Previously reverted; re-attempt with admission gate |
| L14 TEMP cleanup (`temp_clear` → L12 `mul_combine`, 2 back-edges) | **MARGINAL — TEMP rotation detector** | TEMP is multi-purpose scratch (`dim_registry.py:755-763`); a "next-step's TEMP cycle starts fresh" detector head is overkill for 2 back-edges. Recommend wave 2 PREV_STEP rename | Cost/benefit doesn't favor detector here |
| L6 routing 4-op (AX_CARRY_LO + NEXT_SE + OUTPUT_BYTE_HI) | **NO — already in flight** | Cherry-pick `382a72b7` is the in-flight AX_CARRY_LO rename (audit wave 0). NEXT_SE / OUTPUT_BYTE_HI are same-step data carriers | Detector unnecessary |
| JSR (not currently in SCC) | **OPTIONAL — JSR detector head** | JSR pushes return PC and jumps; semantically symmetric to LEV. A `JSR_DETECTOR` head would let step-N+1's L8 sp_gather_bake read the just-pushed STACK0 byte0 via the head, eliminating today's same-step `requires["after"]=layer7_memory_heads` chain (`l8_ops.py:1735`) at the JSR-following step | Future work; the same V/O wiring pattern with K matching `OP_JSR` instead of `OP_LEV` |
| JMP / BZ-taken | **NO — fully same-step** | The L5 fetch in step N+1 reads the new PC from the KV cache row of the JMP/BZ marker, which is _written_ at step N's L9 alu and visible via the standard PC residual the next step. No same-step dep_graph cycle — no detector needed | Already fine |

## 4. Effort estimate vs the dim-alias + `requires["next_step_after"]` plan

| Approach | Cells added | Ops modified | New IR primitive? | Byte-identity risk | ETA (agent-days) |
|---|---:|---:|---|---|---:|
| **A. Dim-alias + `requires["next_step_after"]`** (audit §6 wave 3) | 0 (alias same slots) | 7 (L8 sp_gather, L9 alu, L14 addr_key, L14 clear, L15 memory_lookup, L15 store_stack0, L16 lev_routing — for the LEV loop) + 1 IR schema + 1 scheduler + 1 analyzer | YES (`requires["next_step_after"]` key) | Low — byte-identical at weight level | ~1.5 |
| **B. LEV detector head only** (this doc, scope = LEV loop) | ~64 W_q + ~256 W_v + ~64 W_o = **~384 weight cells** for 1 head; +64 residual dims | 3 (L8 sp_gather_bake reads, L9 alu reads, drop `requires["after"]=layer16_lev_routing` at l8/l9) + 1 new op factory (the detector itself) | NO (works in existing IR) | Medium — new attention head changes `state_dict()` SHA; sentinel-mode rerun required | ~2.5 |
| **C. Detector + delete L16 LEV OUTPUT writes** (Option B + algebra cleanup) | -~600 FFN cells (792 units × ~0.75 fully retired) + ~384 attn cells | 5-7 (B + L16 rule deletion + downstream STACK0 cleanup re-verify) | NO | High — bake-time algebra changes; full 1096 corpus rerun | ~4 |

**Recommendation: A first, then optionally B for the LEV loop only.**
A retires the LEV cycle for 1.5 days of work and unblocks SCC=0; B is
a follow-up V2-aligned cleanup that can land in Phase 9 once SCC=0 is
proven. C is post-Phase-8; the FFN-cell savings are small relative to
the 1096 corpus regression risk.

## 5. Risks & open questions

### R1 — Residual stream crowding

The 4 × 16-wide detector bands (64 slots) need to live somewhere that
isn't aliased onto a load-bearing existing slot. `dim_registry.py:1078-1100`
documents the compact-layout gap at 698+; **open question**: is the
gap actually 64 slots wide given the 7.A.3 `OUTPUT_LO_PREV_STEP`
aliasing convention? Mitigation: alias `*_VIA_LEV_DETECTOR_LO/HI`
**onto fresh slots above 730** rather than into the existing
PREV_STEP block, accepting a 64-slot d_model bump for the next bake
or compacting at compile time.

### R2 — Detector head budget

L8 already uses 6+ heads (`_L8_HEAD_LAYOUT_BY_NAME` from
`l8_ops.py:1748+`); `DEFAULT_LAYER_MAX_HEADS = 8` per
`attention_head_allocator.py`. **Open question**: does L8 have a free
head slot? If `head_6_ax_carry_refresh` (enable=False today,
`l8_ops.py:1882-1896`) is the only spare, the detector competes with
that op's eventual enablement. Mitigation: bump
`DEFAULT_LAYER_MAX_HEADS` to 9 for L8, or reuse the disabled head 6's
slot.

### R3 — False-positive misfire

The detector's Q fires on `MARK_AX + HAS_SE`; K matches any past
position with `OP_LEV + MARK_PC`. If the program has executed multiple
LEVs in its history, the softmax picks the **most recent** via the
ADDR_KEY top-nibble + ALiBi recency bias (same mechanism as L5 fetch
heads at `l5_ops.py:249-255` — the `dynamic_top_q` slot 35..50 with
ADDR_KEY+32+k pins the latest matching opcode row). **Open**: does
this hold at step boundaries when ADDR_KEY hasn't been written for
the just-finished LEV? Mitigation: condition the Q on `HAS_SE` (the
prev-step's STEP_END marker) at slot 34, mirroring
`l8_ops.py:1916-1917`. If misfire still occurs, the L9 alu downstream
read uses a `gate=OP_LEV_PREV_STEP` predicate to ignore the detector
output on non-LEV-following steps, falling back to the current
ALU_LO/AX_CARRY_LO path.

### R4 — The `requires["next_step_after"]` primitive is more general

Approach A's IR primitive solves the L10 carry trio, the L5 fetch
dep_anchor, and the LEV loop in one stroke. Approach B (detectors)
needs one per control-flow opcode (LEV, JSR, JMP-taken). For Phase
8's SCC=0 acceptance, **A is strictly cheaper**. B is V2-superior
(architecture expresses semantics; scheduler doesn't have to know
about step boundaries) and is the right Phase 9 cleanup.

### R5 — Interaction with KV eviction (Phase 8.E)

The detector head reads from prev-step's opcode-marker row via the
KV cache. If 8.E's overwrite-map evicts the OP_LEV marker row before
step N+1's L8 detector fires, the detector returns zeros and PC reads
fail. Mitigation: declare in `8.E.1`'s catalog that "OP_LEV residual
rows are live for 1 step after emission", same shape as the
existing transient-scratch / SE_marker live-window logic.

## 6. Recommendation

| Sub-cycle | Recommended fix | Wave |
|---|---|---|
| §3.6 7-op LEV loop | **Approach A** (`requires["next_step_after"]`) **first**; LEV detector head as Phase 9 V2-cleanup | 8.A wave 3 (A) + Phase 9 (B) |
| §3.7 L10 carry trio | Structural collapse: delete `layer10_carry_relay` (keep `_bake`) | 8.A wave 3 step 9 |
| §3.8 L5 fetch dep_anchor | Drop `writes` (1-line) | 8.A wave 3 step 8 |
| §3.1 L3 EMBED_LO | `EMBED_LO_PREV_STEP` rename | 8.A wave 2 step 3 |
| §3.2 L6 OUTPUT_HI_THIS_STEP | `OUTPUT_HI_THIS_STEP_PREV_STEP` rename | 8.A wave 1 step 1 |
| §3.3 L7 OUTPUT_HI | `OUTPUT_HI_PREV_STEP` rename (re-attempt) | 8.A wave 1 step 2 |
| L14 TEMP | `TEMP_PREV_STEP` rename | 8.A wave 2 step 6 |
| L6 4-op routing | In-flight cherry-pick `382a72b7` | 8.A wave 0 |
| JSR | JSR detector head | **Phase 9 only** — not in SCC today |
| JMP / BZ-taken | No action | already fine |

**Bottom line**: detector heads are the right architecture for cross-step
control flow (V2-aligned), but Phase 8's SCC=0 milestone is better
served by the cheaper Approach A. Land A this phase; demo B in Phase
9's `8.H` zero-pin op pattern (see `PHASE_8_PLAN.md` §8.H) where the
LEV detector head is a natural "fully-declarative new op" forcing
function. Approach C (delete L16 LEV OUTPUT rules) is deferred until
the 1096 corpus baseline is healthier — risk-adjusted, not now.

## 7. References

* `c4_release/.agent-logs/scc_zero_audit.md` — source of the 7-op
  structural cycle and the cross-step audit.
* `c4_release/neural_vm/unified_compiler/ops/l16_ops.py:101-205,
  1604-1702` — LEV routing rules + op declaration.
* `c4_release/neural_vm/unified_compiler/ops/l8_ops.py:1872-2001` —
  prior art (`layer8_head6_ax_carry_refresh`): an attention head that
  attends back to prev-step OUTPUT and writes a fresh dim.
* `c4_release/neural_vm/unified_compiler/ops/l9_ops.py:1140-1185` —
  the load-bearing `requires["after"]=layer16_lev_routing` next-step
  declaration.
* `c4_release/neural_vm/unified_compiler/ops/shared.py:743-883` —
  PREV_STEP alias registry (the workaround the detector replaces).
* `c4_release/neural_vm/dim_registry.py:606-628, 1078-1100` — residual
  stream slot map and compact-layout gap.
* `c4_release/neural_vm/unified_compiler/ir.py:276-469` — `AttentionOp`
  / `AttentionHeadIR` / `RuntimeAttentionFragment` (IR types for the
  detector head spec).
* `c4_release/neural_vm/unified_compiler/primitives.py:30-101` —
  `DeclarativeAttentionHeadSpec` + `AP` / `AO` constructors used in
  §2.2.
* `c4_release/neural_vm/unified_compiler/layer_compiler.py:260-313` —
  `Operation` fields (`reads`, `writes`, `requires`, `consumes_fresh`,
  `produces`) the detector op declaration uses.
* `c4_release/docs/PHASE_8_PLAN.md` §8.A (Stream 1 scheduler), §8.H
  (zero-pin demo), §7 (V1-V5 vision mapping).
