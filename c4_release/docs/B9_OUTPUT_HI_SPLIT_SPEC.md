# B9 — `OUTPUT_HI` split: design spec

_Status: DESIGN. Branch `speedup-cache-and-buckets` (HEAD ≈ `990a571`)._
_Author: B9 OUTPUT_HI sub-unit design agent, 2026-06-01._

Companion to `docs/DYNAMIC_SCHEDULER_MIGRATION_PLAN.md` §6.1 (the
parent migration plan)._ This document refines that plan's 1.5-day
`OUTPUT_HI` split sub-unit into a mechanical implementation brief.

This document is the load-bearing design for the largest single B9
sub-unit. Phase A diagnostic
(`.agent-logs/scheduler_phase_a_2026_06_01.md` line 47) attributes
**81 of the ~110 back-edges** inside the 57-op SCC to the `OUTPUT_HI`
dim alone — by far the largest single contributor.

Source-of-truth census: `c4_release/tools/census_output_hi.py`
(committed alongside this spec). The script is read-only; it imports
`all_core_ops()` + `all_alu_postop_attach_ops()`, walks every
declarative `Operation` field that names `OUTPUT_HI`, and tags each op
PRODUCER / SAME_STEP_READER / CROSS_STEP_READER / MIXED. Re-running it
on any future branch should reproduce the counts below.

The dim `OUTPUT_HI` is a 16-slot residual band carrying the high
nibble of the per-step output byte (BLOG_SPEC §registers). It is
populated incrementally over L3..L17 by ~32 PRODUCERs, consumed by 8
ops, and is the canonical example of the "same-step write vs
cross-step carry" naming collision the dynamic scheduler can't resolve
today.

---

## 1. Census (40 ops total touch `OUTPUT_HI`)

Categorisation rule (matches `tools/census_output_hi.py`):

- **PRODUCER** — declares `OUTPUT_HI` in `writes` and/or `produces`,
  does not declare it in `reads`/`consumes_fresh`. Writes the value
  for the current step's downstream consumers and/or the next step's
  carry-forward.
- **SAME_STEP_READER** — declares `OUTPUT_HI` in `reads` (or
  `consumes_fresh`) and consumes the value written by an earlier
  layer **within the same forward pass**.
- **CROSS_STEP_READER** — declares `OUTPUT_HI` in `reads` and the
  semantic value is the **previous step's** `OUTPUT_HI` (the
  carry-forward path). These edges are the load-bearing cycle
  creators: producer at L_u writes for step N; reader at
  L_v < L_u reads the value step N-1 already produced.
- **MIXED** — both writes AND reads `OUTPUT_HI`. The read may be
  same-step (correction / clear / gate-erase) or cross-step (carry
  forward) — see per-op manual review in §2.

### 1.1 PRODUCERs (32 ops)

| Op name | Layer | Kind | Role |
|---|---|---|---|
| `_layer3_ffn_dep_anchor` | 3 | ffn | topology anchor; bake is no-op (the real bake lives in `layer3_ffn`) |
| `layer3_ffn` | 3 | block | initial PC-cancellation + LEV-return PC HI nibble write |
| `layer6_ent_after_jsr_sp_byte0_fixup` | 6 | block | ENT-after-JSR SP byte 0 fixup |
| `layer8_multibyte_routing` | 8 | block | multibyte routing; also `produces` `OUTPUT_HI@AX_byte0` |
| `layer9_alibi_mem_attn` | 9 | block | ALiBi MEM attn writes OUTPUT_HI for MEM reads |
| `layer9_alu` | 9 | block | comparison ALU lookup |
| `layer9_marker_suppress` | 9 | ffn | suppression / cleanup |
| `layer10_alu` | 10 | block | boolean/shift ALU lookup |
| `layer10_bp_byte_passthrough_bake` | 10 | block | BP byte passthrough |
| `layer10_byte_passthrough` | 10 | attn | wide byte passthrough |
| `layer10_byte_passthrough_bake` | 10 | block | (paired bake) |
| `layer10_psh_stack0_passthrough` | 10 | attn | PSH STACK0 passthrough |
| `layer10_psh_stack0_passthrough_bake` | 10 | block | (paired bake) |
| `layer10_sp_byte_passthrough` | 10 | attn | SP byte passthrough |
| `layer10_sp_byte_passthrough_bake` | 10 | block | (paired bake) |
| `layer10_stack0_byte_relay` | 10 | attn | STACK0 byte relay |
| `layer10_stack0_byte_relay_bake` | 10 | block | (paired bake) |
| `layer12_mul_combine` | 12 | block | mul-combine; `produces` `OUTPUT_HI@AX_byte0` |
| `layer13_shifts` | 13 | block | shift lookup |
| `layer14_alu_nocarry_ax_bytes_zero` | 14 | block | ALU no-carry AX-byte zeroing |
| `layer14_clear_mem_marker_output` | 14 | block | clear MEM marker OUTPUT for OP_JSR/OP_ENT |
| `layer14_jsr_ax_bytes_zero` | 14 | block | JSR AX-byte zero |
| `layer14_lc_ax_bytes_zero` | 14 | block | LC AX-byte zero |
| `layer14_mem_generation` | 14 | attn | MEM-target generation |
| `layer14_temp_clear` | 14 | block | TEMP clear (writes OUTPUT_HI residue) |
| `conversational_io_output_routing` | 15 | block | conversational-IO output routing |
| `layer15_alu_high_byte_relay` | 15 | block | wide-ALU high-byte relay |
| `layer15_memory_lookup` | 15 | attn | memory lookup |
| `layer15_nibble_copy` | 15 | block | nibble copy block wrapper |
| `layer15_si_mem_addr0_from_stack0` | 15 | block | SI/SC mem addr 0 override from STACK0 |
| `nibble_copy_ffn` | 15 | ffn | nibble-copy FFN (the most ubiquitous OUTPUT_HI writer per the cycle reasons list) |
| `layer16_lev_routing` | 16 | ffn | LEV routing FFN |

### 1.2 SAME_STEP_READERs (2 ops, read-only)

| Op name | Layer | Kind | Reads OUTPUT_HI for … | Producer it depends on (same step) |
|---|---|---|---|---|
| `layer7_operand_gather` | 7 | block | operand B for LEA/ADJ/ENT — `OUTPUT_HI+k` at AX byte k is the latest in-step OUTPUT (post-L6 routing) | `layer3_ffn`, `layer6_routing_ffn`, `layer6_ent_after_jsr_sp_byte0_fixup` |
| `layer15_store_stack0_sp_byte0_addr` | 15 | block | the PSH store address: SP marker carries the freshly computed address in OUTPUT for PSH | L3 / L6 / L9 / L10 / L14 OUTPUT_HI writes |

### 1.3 CROSS_STEP_READERs (1 op, read-only)

| Op name | Layer | Kind | Reads `OUTPUT_HI` for … | Source (previous step) |
|---|---|---|---|---|
| `layer8_head6_ax_carry_refresh` | 8 | block | refresh `AX_CARRY_LO/HI` from **prev step's AX marker OUTPUT** (docstring: "refresh AX_CARRY from prev step's AX marker OUTPUT"). Without this op, AX_CARRY carries the prev-PREV step's value (the stale-AX_CARRY bug commit 3d1b700 fixed). | `layer3_ffn` … `layer16_lev_routing` of step N-1 |

Note: in production this op is `enable=False` by default
(`l8_ops.py:762`), but the `Operation` is registered so the staleness
analyzer sees its `produces` annotation. The dim edge is still
declared.

### 1.4 MIXED (5 ops — read + write)

| Op name | Layer | Kind | Read semantic | Write semantic |
|---|---|---|---|---|
| `layer3_carry_forward_attn` | 3 | attn | **CROSS_STEP**: head 5 `_ax_full_relay_head_spec` reads `OUTPUT_LO/HI+k` at the AX marker and routes to `AX_FULL_LO/HI+k`. L3 fires before any same-step OUTPUT_HI producer → the only available value is step N-1's residual. | Same-step: heads 0/1 write `EMBED_LO/HI` and `AX_CARRY_LO/HI`; the op's `writes={OUTPUT_LO, OUTPUT_HI, …}` reflects L3 FFN-style residual writes paired with this attn op (the dep_anchor + block bake at L3) |
| `layer6_routing_ffn` | 6 | block | **SAME_STEP**: reads `OUTPUT_HI+k` as a gate to negate / clear prior-layer (L3) writes when routing to FETCH/AX_CARRY paths (see `_layer6_jsr_sp_*_rules`: `writes=(("OUTPUT_HI+15", +), ("OUTPUT_HI+0", -))` is a same-step rewrite). | Same-step: AX/PC/SP/BP routing per opcode |
| `l10_post_ops_combined` | 10 | ffn | **SAME_STEP**: byte/marker cleanup over L3-L9 OUTPUT writes (negative blockers, step-boundary suppression — see `_strengthen_l10_carry_wrong_byte_blockers`) | Same-step nibble corrections |
| `layer14_clear_output_corruption` | 14 | block | **SAME_STEP**: 16 cleanup units zero `OUTPUT_HI+k` at MEM marker rows (PSH-on-OP_JSR cleanup). It is end-of-pipeline corruption clear, not a cross-step carry. | Same-step: `W_down[OUTPUT_HI+k, unit]` for nibble in 0..15 |
| `tail_bit32_result_correction` | 17 | block | **SAME_STEP**: post-pass wide-ALU tail correction over L10/L12/L13 ALU outputs. Reads in-step `OUTPUT_HI` to mask 32-bit overflows. | Same-step: bit32 result correction |

**Manual category breakdown after §2 review:**

| Category | Count |
|---|---:|
| PRODUCER | 32 |
| SAME_STEP_READER (pure) | 2 |
| CROSS_STEP_READER (pure) | 1 (`layer8_head6_ax_carry_refresh`) |
| MIXED with CROSS_STEP read | 1 (`layer3_carry_forward_attn`) |
| MIXED with SAME_STEP read only | 4 (`layer6_routing_ffn`, `l10_post_ops_combined`, `layer14_clear_output_corruption`, `tail_bit32_result_correction`) |
| **Total** | **40** |

Cross-step read ops: **2** (`layer3_carry_forward_attn` head 5 +
`layer8_head6_ax_carry_refresh`). Same-step read ops: **6**.

---

## 2. Per-op carry semantics (CROSS_STEP + MIXED)

### 2.1 `layer3_carry_forward_attn` head 5 — AX_FULL relay (CROSS_STEP)

**Carry semantic.** L3 head 5 ("`_ax_full_relay_head_spec`") attends
at the current AX marker, with V slots reading `OUTPUT_LO+k` (slots
1..16) and `OUTPUT_HI+k` (slots 17..32), routed via O to
`AX_FULL_LO/HI+k`. The attention key is `MARK_AX`, so the head
attends back along the autoregressive sequence to the **previous AX
marker token**, whose residual still holds **the previous step's
output byte at the AX position**. The OUTPUT_HI value being read is
step N-1's; L3 is the first layer in step N to touch OUTPUT_HI.

**Natural decomposition.** Head 5's V reads change from
`OUTPUT_LO/HI+k` to `OUTPUT_LO/HI_PREV_STEP+k`. Heads 0-4 (PC, AX,
SP, BP, STACK0 carry-forwards) of the same op write EMBED bands
(not OUTPUT), so they don't move. Heads 6-7 (LEV BP→PC, PC byte1
preserve) read CLEAN_EMBED, not OUTPUT — also unchanged.

The op's top-level `reads={…, "OUTPUT_LO", "OUTPUT_HI"}` becomes
`reads={…, "OUTPUT_LO_PREV_STEP", "OUTPUT_HI_PREV_STEP"}`. The
top-level `writes={…, "OUTPUT_LO", "OUTPUT_HI"}` is **renamed** to
`OUTPUT_LO_THIS_STEP`/`OUTPUT_HI_THIS_STEP` (uniform rename of every
OUTPUT producer; see §6).

### 2.2 `layer8_head6_ax_carry_refresh` — AX_CARRY refresh (CROSS_STEP)

**Carry semantic.** Per the docstring: "L8 attn head 6: refresh
AX_CARRY_LO/HI from prev step's AX marker OUTPUT." This op exists to
patch the stale-AX_CARRY bug: it reads `OUTPUT_LO/HI+k` at the AX
marker from the **previous step's residual** (still present in the
KV-cache row for the prev-step AX marker token) and routes the value
through V/O to refresh `AX_CARRY_LO/HI`. The huge `reads={…, "OP_*",
…}` set guards the refresh against a long list of opcodes.

**Natural decomposition.** Same as §2.1: change V reads from
`OUTPUT_LO/HI` to `OUTPUT_LO/HI_PREV_STEP`. The top-level
`reads={…, "OUTPUT_LO", "OUTPUT_HI"}` becomes
`reads={…, "OUTPUT_LO_PREV_STEP", "OUTPUT_HI_PREV_STEP"}`. This op
does NOT write OUTPUT_*, so no rename on its `writes` set.

### 2.3 MIXED — `layer3_carry_forward_attn` writes path

The op writes OUTPUT_LO/HI as part of L3's FFN-paired bake (head 5's
O matrix writes back into AX_FULL only — but the
`writes={"OUTPUT_LO", "OUTPUT_HI", …}` set is shared with the
sibling `layer3_ffn` block op). Under the rename convention these
becomes `OUTPUT_LO_THIS_STEP`/`OUTPUT_HI_THIS_STEP`.

The op now has both reads PREV_STEP and writes THIS_STEP — there is
no longer a cycle on OUTPUT_HI: the read (PREV_STEP) and the write
(THIS_STEP) are different dims.

### 2.4 SAME_STEP MIXED ops

For `layer6_routing_ffn`, `l10_post_ops_combined`,
`layer14_clear_output_corruption`, `tail_bit32_result_correction`:

- Read semantic is current-step (the op reads OUTPUT_HI written by an
  earlier same-step producer).
- Decomposition: both reads and writes rename
  `OUTPUT_HI` → `OUTPUT_HI_THIS_STEP`. No new dim.

---

## 3. Single carry-forward op design

A new declarative op carries `OUTPUT_HI_THIS_STEP` → `OUTPUT_HI_PREV_STEP`
across step boundaries.

### 3.1 Kind and layer placement

**Kind: attention head, NOT FFN.** Reason: an FFN can only read the
current token's residual, which by definition is the start of the
current step — empty for `OUTPUT_HI_THIS_STEP` until a producer fires.
An attention head can attend back to the **prior step's matching
marker token** whose residual still holds the prev-step write. This
is exactly the pattern already in use at L3 head 5
(`_ax_full_relay_head_spec`) and L8 head 6
(`layer8_head6_ax_carry_refresh`); B9 makes that pattern its own op.

**Layer placement: L0 (or pre-L3).** The carry-forward op must fire
before any L3+ consumer that reads `OUTPUT_HI_PREV_STEP`. Today the
only two such consumers are `layer3_carry_forward_attn` (L3, head 5)
and `layer8_head6_ax_carry_refresh` (L8). L0 is the earliest legal
slot (L1/L2 are the threshold-attn / lookback layers; placing the
carry-forward there competes for head slots used by structural attn).

**Concrete proposal: L0 attention head, 1 head slot.** Mirror the
shape of L3 head 5 / L8 head 6:

```
Q : MARK_AX (or per-marker variant; see §3.2) with HAS_SE gate
K : MARK_AX (attends to the matching marker in the prev-step rows)
V : OUTPUT_HI_THIS_STEP + k    for k=0..15
O : OUTPUT_HI_PREV_STEP + k    for k=0..15
```

Q/K together with the autoregressive mask ensure the head attends to
the **previous occurrence** of the matching marker token, which by
KV-cache invariants holds the prev-step OUTPUT residual. The
proof-of-concept ALiBi/positional bias is identical to L3 carry-forward
heads 0-3.

If the model has only one AX marker per step (the canonical case),
the carry-forward is exact. If a step has multiple markers (multi-byte
AX), the head picks the immediately-prior matching marker, which is
the **previous step's byte k** — correct.

### 3.2 Per-marker variants

OUTPUT is keyed per-marker (AX, SP, BP, STACK0, PC). The carry
semantic only matters for the AX marker today (both prev-step readers
are at AX). So **one head suffices**. If a future reader needs
prev-step OUTPUT at SP/BP/PC/STACK0, mirror the head shape with the
appropriate Q/K marker dim.

Recommendation: ship **one head, AX-marker keyed**, and add a clear
comment that SP/BP/PC variants are mechanical to add when a consumer
declares the need.

### 3.3 KV cache / autoregressive interaction

- Position IDs: the head attends backward; no change to position IDs.
- KV cache: works out of the box — the head's K/V projections are
  applied to every token's residual as it enters the cache. Token
  rows for the prior step's AX marker are already cached and
  available.
- Per-step reset semantics: `OUTPUT_HI_PREV_STEP` should be **read-
  only** for L1+. The L0 carry-forward op is its **sole producer**.
- The carry head writes `OUTPUT_HI_PREV_STEP` at the **current step's**
  AX marker row, where the prev-step OUTPUT_HI value (gathered via
  attention) now lives. L3 head 5 / L8 head 6 then read that value at
  the same AX marker row in the same step.

### 3.4 Slot cost

If we choose **rename + true split** (Q3 Option A; see §4):
`OUTPUT_HI_PREV_STEP` is a new 16-slot residual band. The L0 head
needs **0 additional head slots** (it occupies one of the dormant L0
attn slots; L0 already has unused heads per `_set_layer0_threshold_attn`).

If we choose **rename-only** (Q3 Option B): the carry-forward
remains implicit — `OUTPUT_HI_THIS_STEP` is the only declared OUTPUT_HI
dim, and the two cross-step reader ops (`layer3_carry_forward_attn`
head 5, `layer8_head6_ax_carry_refresh`) acquire a **`requires`
declaration** instead of a data dep. No new op, no new slots — but
the DAG-vs-cycle invariant is enforced by `requires` semantics from
B10, not by dim algebra.

**Recommendation (matches Q3 below): rename-only.** No carry-forward
op gets added in B9; B10's `requires["after"]` field is the
cycle-breaking mechanism for the 2 cross-step reads. The design above
(L0 attn head) is documented as the **fallback** if a future consumer
needs `OUTPUT_HI_PREV_STEP` as an explicit dim.

---

## 4. Slot impact — Q3 decision

The migration plan asks (Q3) whether `OUTPUT_HI_PREV_STEP` should be
a real 16-slot dim or whether we can rename in place.

| Option | New slots | Cycle break | Carry op needed | Reader work |
|---|---:|---|---|---|
| A — true split: `OUTPUT_HI_THIS_STEP` + `OUTPUT_HI_PREV_STEP` | 16 | by dim algebra alone | YES (L0 attn head) | each cross-step reader retargets `reads` to `OUTPUT_HI_PREV_STEP` |
| B — rename only: `OUTPUT_HI` → `OUTPUT_HI_THIS_STEP`, no prev_step dim | 0 | requires B10 `requires["after"]` for cross-step readers | NO | 2 ops get `requires["after"] = "<latest_producer>"` to acknowledge they read the residual cached by the prev step |

### 4.1 Decision: Option B (rename only) is decidable; recommend B.

**Rationale.**

1. **Only 2 ops read `OUTPUT_HI` cross-step.** Both
   (`layer3_carry_forward_attn` head 5, `layer8_head6_ax_carry_refresh`)
   already implement the cross-step semantic via attention V reads
   from the cached prev-step residual — they don't need a separate
   declared dim. They need the scheduler to know "this read is NOT a
   data dep on a same-step producer."
2. **B10's `requires["after"]` is the strictly-cheaper mechanism.**
   For each of the 2 cross-step ops, a single
   `requires["after"] = "layer16_lev_routing"` (or whatever the
   latest same-step producer is) tells the dynamic scheduler:
   "schedule me anywhere after this op; my OUTPUT_HI read is the
   prev step's residual, not a same-step data flow." This is exactly
   the `requires` field's purpose post-B10.
3. **16 slots saved.** §6.7 of the parent plan already shows B9 +
   B8-B/C/D competing for slots 101-115; saving 16 by choosing B
   means there is no slot contention with B8-B/C/D at all.
4. **Reversibility.** If a future consumer needs `OUTPUT_HI_PREV_STEP`
   as an explicit dim, the L0 carry head design in §3 is ready to
   land — we add the dim then. The B option does not preclude A.

### 4.2 Cost of Option B

- The 2 cross-step ops each gain a `requires["after"] = "<op_name>"`
  declaration. B10 lands this semantics; B9 consumes it.
- The compiler / verifier must NOT treat
  `reads(OUTPUT_HI_THIS_STEP)` on these 2 ops as a same-step
  data dep when an explicit `requires["after"]` is present. This is a
  small additional invariant on top of B10's schema (see §7.2).

### 4.3 Cost of Option A (if chosen anyway)

- 16 new slots for `OUTPUT_HI_PREV_STEP`. Per the parent plan §6.7 the
  free window is 101-115 (15 slots); Option A would NOT fit and would
  require either displacing all of B8-B/C/D or carving out a different
  window.
- 1 new op (L0 attn head, ~1 head slot).
- Per-op rewire for the 2 cross-step readers points to the new dim
  instead of `requires["after"]`.

Option A is **clean** but **overpays** for the 2 cross-step
declarations. Recommend B.

---

## 5. Validation strategy

The split must be byte-identical to the static path for the released
bake (parent plan N1). Validation has 3 stages.

### 5.1 Pre-split capture

Before any B9 dim rename lands:

1. Build the static bake (`compile_full_vm` at HEAD).
2. Run a `tools/verify_op.py`-style symbolic trace over the 1096
   corpus AND a small "OUTPUT_HI probe" program that exercises every
   PRODUCER (e.g. `IMM 10 / PSH / IMM 32 / ADD / LEA / ENT / LEV` —
   the AX_CARRY stale-bug regression program).
3. For each step in each trace, capture `residual[:, OUTPUT_HI+k]`
   after **every layer** for k = 0..15. Save to
   `.agent-logs/output_hi_pre_split_<date>.npz`.
4. Capture the cross-step values too: at L3 entry of step N, the value
   of `residual[:, OUTPUT_HI+k]` at the AX marker row (this is the
   value head 5 will read).

### 5.2 Post-split bake

1. Apply the B9 rename:
   - Rename the dim `OUTPUT_HI` → `OUTPUT_HI_THIS_STEP` in
     `declare_setdim_compat_dims` and the dim_registry.
   - Update `decl_verifier.py`, `band_contracts.py`,
     `band_guarantees.py`, `primitives.py` band-name defaults.
   - Per-op rewire (§6 below): change every `OUTPUT_HI` → either
     `OUTPUT_HI_THIS_STEP` (the default) or add
     `requires["after"]` (the 2 cross-step ops).
2. Re-build the bake.
3. Re-run the same trace.
4. Compare:
   - **`OUTPUT_HI_THIS_STEP[k] == OUTPUT_HI[k]`** (pre-split) at every
     layer of every step. Byte-identical.
   - For the cross-step reader ops, the post-split residual at L3/L8
     reader rows matches the **prev step's** pre-split
     `OUTPUT_HI` at the AX-marker token. Delay-by-1 step.

### 5.3 1096 sweep

Run `test_symbolic_declarative_runner_passes_full_1096_suite` (parent
plan AC4). The per-id pass list must match the static-path baseline
exactly. Q1 (delta tolerance) governs the acceptance threshold; per
the plan, **N = 0** during B14 acceptance is the recommended target
for B9.

### 5.4 Per-op claim verification

`tools/verify_op.py` runs the symbolic interpreter against each
PRODUCER's declared `claims`. Renaming the dim does not change which
cells are written, so claims with `column="OUTPUT_HI+k"` should be
mechanically updated to `column="OUTPUT_HI_THIS_STEP+k"`. The
verifier output is unchanged.

### 5.5 Cycle-break check

`tools/analyze_scheduler.py` reports `dep_graph_cycle_member` after
the B9 rename. The 81 back-edges should drop to 0 for `OUTPUT_HI*`.
Residual cycle members at this stage come from the other dim families
(`AX_CARRY_HI`, `ADDR_KEY`, etc. — separate B9 sub-units).

---

## 6. Per-op rewire plan (mechanical edits)

Each entry is a 1-2 line patch. PRODUCERs and SAME_STEP readers are
pure renames. The 2 CROSS_STEP readers each get an explicit
`requires["after"]` AND a rename.

### 6.1 Common rename (every PRODUCER)

For each of the 32 PRODUCERs above:

```python
# Replace
writes={..., "OUTPUT_HI", ...}
# With
writes={..., "OUTPUT_HI_THIS_STEP", ...}
```

Also rename in `claims` tuples: `"OUTPUT_HI+k"` → `"OUTPUT_HI_THIS_STEP+k"`.
Also rename in `produces`: `{"OUTPUT_HI": "AX_byte0"}` →
`{"OUTPUT_HI_THIS_STEP": "AX_byte0"}` (2 ops:
`layer8_multibyte_routing`, `layer12_mul_combine`,
`layer6_routing_ffn`).

The per-PRODUCER `_set_layerN_*` bake bodies that index
`BD.OUTPUT_HI` (and the verifier's `band_contracts.py` defaults) get
the same `OUTPUT_HI_THIS_STEP` rename. The numeric slot value is
unchanged.

### 6.2 SAME_STEP readers (rename only)

| Op | Patch |
|---|---|
| `layer7_operand_gather` | `reads={..., "OUTPUT_HI"}` → `reads={..., "OUTPUT_HI_THIS_STEP"}`; same for claims `"OUTPUT_HI+k"` → `"OUTPUT_HI_THIS_STEP+k"` |
| `layer15_store_stack0_sp_byte0_addr` | `reads={..., "OUTPUT_HI"}` → `reads={..., "OUTPUT_HI_THIS_STEP"}` |
| `layer6_routing_ffn` (MIXED, same-step) | rename both `reads` and `writes` |
| `l10_post_ops_combined` (MIXED, same-step) | rename both |
| `layer14_clear_output_corruption` (MIXED, same-step) | rename both |
| `tail_bit32_result_correction` (MIXED, same-step) | rename both |

### 6.3 CROSS_STEP readers (rename + `requires["after"]`)

`layer3_carry_forward_attn`:
```python
# Top-level Operation fields
reads = {..., "OUTPUT_LO_THIS_STEP", "OUTPUT_HI_THIS_STEP", ...}  # the rename
writes = {..., "OUTPUT_LO_THIS_STEP", "OUTPUT_HI_THIS_STEP", ...} # the rename
requires = {
    "after": "layer16_lev_routing",  # acknowledges the cross-step OUTPUT read
}
```

Head 5 `_ax_full_relay_head_spec` V reads via `BD.OUTPUT_HI+k` already
resolve through `BD` proxy — once `OUTPUT_HI_THIS_STEP` aliases the
same numeric slot, the body needs no change.

`layer8_head6_ax_carry_refresh`:
```python
reads = {..., "OUTPUT_LO_THIS_STEP", "OUTPUT_HI_THIS_STEP", ...}
requires = {
    "after": "layer16_lev_routing",
}
```

The `requires["after"]` value should be the latest same-step
PRODUCER. `layer16_lev_routing` (L16) is the structurally-latest
in-block OUTPUT_HI writer; `tail_bit32_result_correction` (L17) is a
post-block tail. Using `layer16_lev_routing` keeps both ops scheduled
strictly after the in-block OUTPUT_HI write chain — which is the
correct "OUTPUT for this step is finalised" anchor.

### 6.4 Dim_registry / declare_setdim_compat_dims

```python
# Replace OUTPUT_HI alloc
reg.alloc("OUTPUT_HI", 16, ...)
# With
reg.alloc("OUTPUT_HI_THIS_STEP", 16, ...)
# Optional alias for back-compat during the rollout:
reg.alias("OUTPUT_HI", "OUTPUT_HI_THIS_STEP")
```

The alias lets the bake bodies that still use `BD.OUTPUT_HI` continue
to compile during the per-op rewire wave, then the alias is removed
after the last rewrite. This is exactly the B7 lifecycle-bit pattern.

### 6.5 Verifier hooks

`decl_verifier.py:710-757` references the string `"OUTPUT_HI"`
directly to slice residuals. Replace the string with
`"OUTPUT_HI_THIS_STEP"` and re-run the verifier. Update
`band_contracts.py:706,741` and `band_guarantees.py:454` default arg
strings.

---

## 7. Risk surface

Top 3 risks specific to OUTPUT_HI's role.

### 7.1 R-OH-1: prev-step residual values are layout-sensitive (HIGH × HIGH)

L3 head 5 + L8 head 6 read prev-step `OUTPUT_HI` by attending back to
the prev-step **AX marker token row** in the cache. If the rename
shifts the numeric slot index of `OUTPUT_HI` (even by one), the cached
prev-step residuals at slot indices `OUTPUT_HI_THIS_STEP+k` will be
**zero on the very first step** (or whatever was in the cache at the
old slot index from the prior model). This is the
`layer_pin` / cell-identity invariant from parent plan §R1 specialised
to OUTPUT_HI.

**Mitigation.** The rename MUST land as a pure name change with the
same numeric base. `dim_registry` allocator's slot output is
deterministic given the input order; verify that
`reg.alloc("OUTPUT_HI_THIS_STEP", 16)` lands at the same base as
`reg.alloc("OUTPUT_HI", 16)` did. If the allocator's order is name-
hashed (it is not, per the current implementation), pin the slot
explicitly. The B7 lifecycle-bit alias pattern (`OUTPUT_HI` is an
alias of `OUTPUT_HI_THIS_STEP` for one rollout commit, then the alias
is dropped) is the safe execution path.

### 7.2 R-OH-2: the 28 cross-step back-edges are NOT all eliminated by `requires["after"]` (HIGH × MED)

The plan's R7 calls out "81 back-edges" but conflates two phenomena:

- ~32 forward-direction edges (producer at L_u → reader at L_v ≥ L_u)
  that are SAME_STEP and become acyclic immediately under the rename.
- ~49 back-direction edges (producer at L_u → reader at L_v < L_u)
  that are CROSS_STEP and become acyclic only if the cross-step reader
  has a valid `requires["after"]` AND the analyzer (`analyze_scheduler.py`)
  honours that edge as a dep AND the dep tooling does NOT also infer a
  back-edge from `writes ∩ reads` on the renamed dim.

The renamed dim does eliminate the back-edge by construction (`writes:
OUTPUT_HI_THIS_STEP` ∩ `reads: OUTPUT_HI_PREV_STEP` is empty under
Option A; Option B has `writes: OUTPUT_HI_THIS_STEP` and `reads:
OUTPUT_HI_THIS_STEP` so the back-edge SURVIVES textually but is
suppressed by `requires["after"]`).

**Mitigation.** B10's analyzer update must add: "an edge u→v from
`writes ∩ reads` on dim D is treated as a back-edge UNLESS v declares
`requires["after"] = <op X>` where X.writes ⊇ {D}." This is a 5-line
change to `analyze_scheduler.py:114-119` (mirror the
`requires` edge handling at line 137-143, but with priority over the
data-flow edge inferred from `writes ∩ reads`).

Without that suppression, Option B leaves the cycle textually present
even after the rename. Document this dependency on B10 explicitly.

The "28 +back-edges" specific category the parent plan calls out
(parent §R7 / Phase A line 47) maps to: the cross-step readers
(L3 head 5 + L8 head 6) each producing back-edges from every later
PRODUCER they textually depend on (~30 of the 32 PRODUCERs are
strictly later than L3 and L8 → ~28 back-edges per reader; only L3's
co-located PRODUCERs and L8's co-located PRODUCERs avoid being
"back" relative to those readers; one of the two readers is at L3
so its same-step + later producers are forward edges).

### 7.3 R-OH-3: `produces`/`consumes_fresh` covenants on AX_byte0 (MED × MED)

Three PRODUCERs (`layer8_multibyte_routing`, `layer12_mul_combine`,
`layer6_routing_ffn`) declare `produces={"OUTPUT_HI": "AX_byte0"}`.
This means "I supply the fresh `OUTPUT_HI` value at the AX_byte0
register." Renaming the dim must preserve the (dim, register) pairing
or the staleness analyzer will lose visibility into the AX_byte0
freshness chain.

**Mitigation.** Update the `produces` dict values consistently across
all 3 producers AND any `consumes_fresh` reader that depends on them
(today: none for `OUTPUT_HI` — the consumes_fresh chain for
AX_byte0 currently flows through `AX_CARRY_LO/HI` and `ALU_LO`,
NOT `OUTPUT_HI` directly). Re-run the staleness analyzer after the
rename to confirm no covenant breaks silently.

---

## 8. Effort estimate — refining the 1.5-day plan

The parent plan scopes "1.5d — OUTPUT_HI split (largest blast radius;
~30 producer/consumer edits across L3..L15)".

With this spec in hand, the implementation is mechanical. Refined
breakdown:

| Step | Effort | Output |
|---|---:|---|
| 1. Land `OUTPUT_HI_THIS_STEP` dim + alias in `declare_setdim_compat_dims` | 0.1d | new dim allocated; alias preserves bake compat |
| 2. Mechanical rename across 32 PRODUCERs + 4 same-step MIXED + 2 SAME_STEP readers (38 files / declarations) | 0.4d | search-and-replace; claims tuples and `produces`/`writes` updated |
| 3. Rewire 2 CROSS_STEP readers with `requires["after"]` | 0.1d | 2-line patch per op |
| 4. Update `decl_verifier.py`, `band_contracts.py`, `band_guarantees.py`, `primitives.py` band-name defaults | 0.1d | rename of one default string |
| 5. Per-op claim verification re-run (`tools/verify_op.py`) | 0.2d | all PRODUCERs / readers pass with renamed dim |
| 6. Drop the `OUTPUT_HI` alias; run analyzer to confirm cycle break | 0.1d | `dep_graph_cycle_member` count drops by ~57 → ~0 for OUTPUT_HI |
| 7. 1096 sweep on a shard runner (`.agent-logs/fast-shards-current`) — byte-identity required | 0.4d | per-id diff empty |
| 8. Document residual cycle members + go/no-go signal to next B9 sub-unit | 0.1d | hand-off note |
| **Total** | **1.5d** | matches parent estimate |

The plan estimate **holds**: 1.5 agent-days with this spec. The risk
is concentrated in Step 1 (slot-identity) and Step 7 (sweep
divergence); both have explicit mitigations in §7.

### 8.1 Does the OUTPUT_HI split unblock ALL 81 back-edges?

Yes — **all 81 OUTPUT_HI back-edges resolve**, but ONLY if:

- B10 lands `requires["after"]` semantics first (Option B), OR
- the analyzer is updated to suppress the data-flow edge inferred from
  `writes ∩ reads` on the renamed dim when the consumer declares
  `requires["after"]` (the R-OH-2 mitigation).

Without B10, the rename alone collapses **all SAME_STEP back-edges
(~53 of 81 by manual count)** but leaves the CROSS_STEP back-edges
(~28) textually intact. The 2 cross-step reader ops still appear to
depend on every later-layer PRODUCER under the rename-only scheme.

**Recommendation.** Sequence: B10 first (1d), then this B9 sub-unit
(1.5d). Total ≈ 2.5d to retire 100% of the OUTPUT_HI back-edges.
This matches the parent plan's sequencing — B9 and B10 are siblings
and B10 is independent.

---

## 9. Cross-references

- Parent migration plan: `c4_release/docs/DYNAMIC_SCHEDULER_MIGRATION_PLAN.md`
  §6.1 (OUTPUT_HI design sketch), §10 Q3 (slot-vs-rename decision).
- Phase A diagnostic: `.agent-logs/scheduler_phase_a_2026_06_01.md`
  line 47 (`OUTPUT_HI: 81 back-edges`).
- Census tool (commits alongside this spec):
  `c4_release/tools/census_output_hi.py`.
- ADDR_B0_VALID lifecycle-bit reference (B7-4 pattern this rename
  mirrors): `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:1-100`.
- Cross-step reader prototypes:
  - `c4_release/neural_vm/unified_compiler/ops/l3_ops.py:367-512`
    (`layer3_carry_forward_attn`, head 5
    `_ax_full_relay_head_spec`).
  - `c4_release/neural_vm/unified_compiler/ops/l8_ops.py:700-775`
    (`layer8_head6_ax_carry_refresh`).
- Slot 99-115 proposal:
  `c4_release/docs/SLOT_99_115_ALLOCATION_PROPOSAL.md` §1.2 (free
  window).
- Operation declaration schema:
  `c4_release/neural_vm/unified_compiler/layer_compiler.py:87-260`.

---

## 10. Open questions

**Q-OH-1.** §6.3 proposes `requires["after"] = "layer16_lev_routing"`
for the 2 cross-step readers. Is `layer16_lev_routing` the canonical
"OUTPUT_HI is finalised for this step" anchor, or should it be
`tail_bit32_result_correction` at L17? L17 is post-block; using it
would push the L3/L8 cross-step readers to wait for L17 every step,
which may not be desired (it puts them *after* the entire main
forward pass). Recommended: `layer16_lev_routing` — L17 is a tail
correction that does not change AX_byte0 OUTPUT.

**Q-OH-2.** The `enable=False` default on
`layer8_head6_ax_carry_refresh` (l8_ops.py:762) means the op's reads
do not contribute back-edges in the production bake — they only
contribute to the analyzer. If the analyzer is the source of the 81
back-edge count, does disabling the op change the count? Concretely:
which of the 81 back-edges live on this `enable=False` op? Suggest
re-running the analyzer with `enable=True` AND `enable=False` and
diffing.

**Q-OH-3.** Per §3.2, only AX marker needs cross-step OUTPUT
carry today. Should the spec pre-allocate the SP/BP/PC/STACK0 mirror
heads in L0 for future-proofing, or wait until a consumer declares
the need? Recommendation: wait. The L0 carry head is a small,
mechanical follow-up.
