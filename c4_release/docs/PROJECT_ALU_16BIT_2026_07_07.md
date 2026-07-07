# PROJECT: 16-bit ALU coverage — dimension-neutral scope + design (Phase 1)

**Date:** 2026-07-07 · **Phase:** 1 (SCOPE + DESIGN, NO weight changes) ·
**Golden gate:** `state_dict_sha256 =
f725c06e1ad4d27659817eb46acba9f7aefe5a2545dcb1357baa24a2406fde6c`
(`f725c06e`) — this doc only READS ops and WRITES documentation. Verified
identical pre-doc (`tools/_isa_golden_hash.py`, isolated cache). Campaign
config (`C4_CAMPAIGN=1`, STEP_TOKENS=30).

**TL;DR.** The 16-bit ALU is the largest uncovered blocked chunk, but the
Phase-1 investigation **overturns the brief's premise**:

1. The multipass workspace band's d_model/n_heads shift is **NOT** what
   regresses non-ALU programs. The head-dim-preserving + ALiBi-base-pinned
   auto-widen is **byte-behaviour-neutral**: MUL_MULTIPASS (n_heads 11→13) and
   even DIV_MULTIPASS (n_heads 11→**27**) both produce **logit L-inf = 0.0**
   vs golden on a probe context (measured, this session). The historical
   `C4_DIV_MULTIPASS=1 = 42/438` regression was **NaN-poisoning of the cascade
   block** on the real ~72-magnitude operand residual — **already fixed**
   (2026-07 operand-clamp in `MultiPassDivBlock.forward`, per
   `docs/FLAG_REGISTRY.md`).
2. **DIV/MOD byte-1+ is already delivered in the golden** by
   `C4_DIV_MULTIBYTE=1` (DEFAULT ON) — a dimension-neutral relay INSIDE the
   existing `FlattenedDivMod` long-division composite (+51: div 21→48/50,
   mod 24→48/50). No multipass, no widen. **DIV/MOD is largely solved.**
3. The real remaining gap is **MUL byte-1** in production `lookup` mode: the
   L11 mul_partial lookup writes only byte-0 partial products (`TEMP+partial`);
   nothing writes the (already-present, default-ON) `MUL_RESULT_HI_LO/HI`
   band, so 16-bit products truncate to byte 0.
4. **`expr_mul_div` / `expr_mod` are OUT OF SCOPE** for this project: their
   root is STACK0 cross-step frame persistence (task #221,
   `docs/EXPR_MULDIV_ROOT_IS_STACK0_PERSISTENCE_2026_06_15.md`), a **byte-0**
   failure where the 2nd op reads a stale frame — a MUL/DIV high-byte fix
   cannot help them.

**RECOMMENDATION: Option B (lookup-extension), MUL only.** Wire the existing
`wide_mul_rules(width_bytes=2)` 65,536-rule flat lookup — which already
computes the FULL 16-bit product and routes byte 1 to `MUL_RESULT_HI_LO/HI`
— into production `lookup` mode. It is **dimension-neutral by construction**
(the result band already exists in the golden; it adds only FFN hidden units
to the L11 MUL block, opcode-gated dark on every non-MUL row). No d_model
change, no n_heads change, no ALiBi shift, no multipass block. Estimated
program lever: **MUL 16-bit byte-1 cluster (~25–50 programs)**; DIV/MOD
already covered; expr clusters excluded.

---

## 1. The perturbation mechanism (brief question 1) — MEASURED

### 1.1 What the workspace band changes

Both multipass flags register `never_share=True`, **flag-gated** residual
bands via `register_residual_band` (`ops/alu_ops.py`):

| flag                | band(s)                                            | dims  |
|---------------------|----------------------------------------------------|-------|
| `C4_MUL_MULTIPASS`  | `MUL_MULTIPASS_WS`                                  | 240   |
| `C4_DIV_MULTIPASS`  | `DIV_MULTIPASS_WS` + `DIV_MP_{GATE,Q_LO,Q_HI,R_LO,R_HI}` | 1728 + 2 + 4×16 |

`collect_registered_residual_bands` (compile time) unions the active bands and
feeds them into the SINGLE head-dim-preserving auto-widen in
`_bake_from_scheduled_ops` (`full_vm_compiler_dynamic.py:3583+`). MEASURED
this session (`C4_CAMPAIGN=1`, `alu_mode='lookup'`):

| build                | d_model | n_heads | head_dim | alibi_base |
|----------------------|---------|---------|----------|------------|
| golden (default)     | 1221    | 11      | 111      | 11         |
| `C4_MUL_MULTIPASS=1` | 1443    | **13**  | 111      | 11         |
| `C4_DIV_MULTIPASS=1` | 2997    | **27**  | 111      | 11         |

So the band DOES grow d_model and ADD trailing heads. But crucially it keeps
**`head_dim` fixed at 111** (the widen rounds d_model up to a multiple of the
BASE head_dim and derives a larger n_heads — it ADDS heads rather than
repartitioning existing ones) and keeps **`alibi_base_heads` pinned at 11**
(via `collect_alibi_base_residual_bands`, which evaluates every flag at its
DEFAULT with `C4_*` cleared, so the flag-gated workspace band is EXCLUDED from
the slope base). This is the SI `num_heads` shift the brief worried about —
but it is *insulated*: existing heads keep their exact dim span, weights, and
ALiBi slope; the new trailing heads read the all-zero appended dims → produce
zero Q/K/V → contribute nothing.

### 1.2 The dimension shift does NOT perturb non-ALU programs — PROVEN

Direct logit comparison, golden vs flag-ON, on an identical probe context
(60-token random stream within vocab, `alu_mode='lookup'`, `C4_CAMPAIGN=1`,
CPU single-forward — `probe_pert3.py` / `probe_pert5.py` this session):

| build                | argmax rows differing | logit L-inf | NaN |
|----------------------|-----------------------|-------------|-----|
| `C4_MUL_MULTIPASS=1` | **0 / 60**            | **0.0**     | no  |
| `C4_DIV_MULTIPASS=1` | **0 / 60**            | **0.0**     | no  |

**The widen is byte-behaviour-identical.** The multipass FFN/post_op blocks
are opcode-gated (`OP_MUL` / `OP_DIV OR OP_MOD`) + marker-gated (`MARK_AX`),
so on every non-ALU row the SwiGLU is dark and the block is a pure residual
identity — confirmed by the 0.0 L-inf (a shared-block perturbation would show
non-zero here regardless of the widen).

### 1.3 So what WAS the historical `DIV_MULTIPASS=1 = 42/438`?

**NaN poisoning of the cascade block**, not the dimension shift
(`tools/_div_multipass_live_footprint.py` root-cause writeup, 2026-07-04):
the 43-pass amplitude-normalized long-division cascade is stable on clean
unit-magnitude one-hots but the REAL teacher-forced residual carries
`max_abs≈72` in the operand bands; the per-pass amplitude AMPLIFIES instead
of staying pinned → running residual explodes ~1.5e37 → inf → all-NaN block
output → every downstream block + the LM head decode token 0 → frame
collapse. **This is FIXED** (`docs/FLAG_REGISTRY.md`, 2026-07):
`MultiPassDivBlock.forward` now clean-onehot-clamps the operand bands to the
cascade's residual==1.0 seed invariant on the div/mod-AX rows before the
passes run (mirroring the composite's `GEToBDConverter._clean_onehot`). With
the clamp the DIV_MP block emits 0 NaN and divmod-correct output over a
3328×13×2 dirty sweep.

**Net:** the multipass path's ONLY dimension-shift consequence is inert
padding heads; its historical regression was a *numerical* bug that is now
closed. But the multipass path is nonetheless the WRONG lever for MUL — see §4.

---

## 2. OPTION A — dimension-neutral multipass (fit the workspace in existing dims)

**Question:** can the 240-dim MUL workspace (or 1728-dim DIV workspace) fit
into existing reserved/unused dims so no widen occurs?

### 2.1 Reserved/scratch dim inventory (`dim_registry_dynamic.py`)

The only explicitly *reserved* bands are tiny:
`RESERVED_315_327` (13 dims, "ENT/LEV staging"),
`RESERVED_400_415` (16 dims, "future PC binary encoding/IO"),
`TEMP` (32 dims, "general temporaries", but ALIASES `OUTPUT_BYTE_LO/HI` and is
live per-step scratch for many ops). Total genuinely-free ≈ 29 dims — an
order of magnitude short of the 240-dim MUL WS (and two orders short of the
1728-dim DIV WS).

### 2.2 The liveness allocator can't absorb it either

The workspace is `never_share=True` because the 7 (MUL) / 43 (DIV) passes
write+read the workspace one-hots WITHIN one block's staged `Sequential` — a
private dim-liveness slot no other op's merge may alias. Marking it
`never_share=False` to let the liveness allocator merge it onto dead donor
slots is unsafe: the cross-pass carry chain would be clobbered by any op whose
"lifetime ended" on the donor dim. And even a mergeable band needs a
same-width dead donor; there is no 240-wide (let alone 1728-wide) dead band.

### 2.3 Verdict on Option A

**Not viable and not necessary.** There is no pool of existing dims large
enough to host the workspace, and — per §1.2 — the widen the workspace
triggers is already byte-behaviour-neutral, so avoiding it buys nothing. The
multipass path's real problem is elsewhere (§4), not the dimensions.

---

## 3. OPTION B — lookup-extension (no multipass block) — RECOMMENDED

**Question:** can MUL/DIV byte-1+ be delivered by extending the existing
lookup tables within the current d_model?

### 3.1 The MUL byte-1 gap in the golden (lookup mode)

Production is `alu_mode='lookup'` (default). The live MUL chain there:

* L10 `_layer10_alu_mul_lo` → `OUTPUT_LO` byte-0 low nibble
* L11 `make_mul_partial_op` → `_mul_partial_rules` writes only
  `TEMP+partial` (byte-0 partials) — **VERIFIED**: `l11_ops.py:334-388`, the
  256×16 rules write `(f"TEMP+{partial}", 10.0/S)` and nothing else.
* L12 `mul_combine` → `OUTPUT_HI` byte-0 high nibble.

Nothing on the lookup path writes `MUL_RESULT_HI_LO/HI`, so 16-bit products
truncate to byte 0. The `MUL_RESULT_HI_LO/HI` band (16+16 dims) and the L13
`layer13_mul_result_hi_relay` (a same-row CAM that copies MUL_RESULT_HI →
AX_FULL for the byte-1 emit) **already exist and are DEFAULT-ON in the
golden** (registered via `flag=mul_width2_enabled`, which defaults ON —
confirmed present in `collect_registered_residual_bands()` this session).
There is simply no writer feeding them in lookup mode.

### 3.2 The writer already exists: `wide_mul_rules(width_bytes=2)`

`wide_alu_dsl.wide_mul_rules(width_bytes=2)` emits a **65,536-rule flat
lookup** (one 5-way-AND rule per `(a_lo, a_hi, b_lo, b_hi)` quad) that
computes the FULL 16-bit product and, with `result_byte1_lo_base` /
`result_byte1_hi_base` supplied, routes:

* byte 0 lo → `OUTPUT_LO+nib0`, byte 0 hi → `OUTPUT_HI+nib1`
* byte 1 lo → `MUL_RESULT_HI_LO+nib2`, byte 1 hi → `MUL_RESULT_HI_HI+nib3`

This is **exactly the byte-1 writer the golden is missing**. It is already
wired — but only in `make_efficient_l11_alumul_wrap_op`, which bakes ONLY
when `alu_mode=='efficient'` (`alu_ops.py:1449` `if alu_mode != 'efficient':
return`). In production `lookup` mode it never runs.

### 3.3 Dimension-neutrality of Option B

The 65,536-rule lookup writes into **already-present residual bands**
(`OUTPUT_LO/HI` + the default-ON `MUL_RESULT_HI_LO/HI`); it adds only **FFN
hidden units to the L11 MUL block**. FFN hidden width does NOT enter the
residual-stream d_model or the attention head split — so **no d_model change,
no n_heads change, no ALiBi shift, no new residual band, no widen.** This is
dimension-neutral by construction.

Cost check: 65,536 hidden units in one block. The L11 lookup block already
carries 4,096 mul_partial units; the flat-lookup slab is ~16× that in one
layer. This is the one real feasibility risk — SwiGLU `W_up` can express a
bounded number of distinct input-pattern hyperplanes per layer
(`docs/DSL_W5_MULDIV_LIMIT.md`), and 65,536 one-hot AND-planes in a single
`PureFFN` is a large but (per the doc) *tractable* matrix at width-2. The
5-way-AND + cell-0-artifact-blocker weights are already tuned for the real
(dirty, ~5.84-magnitude) operand-gather residual
(`alu_ops.py:1478+`, `tools/tune_mul_width2.py`), and the
`C4_MUL_W2_THRESH_FIX` follow-up (threshold 19.5→19.0) already fixes the
handful of clean-operand razor-thin misses.

### 3.4 DIV/MOD is already lookup-extended (no work needed)

`C4_DIV_MULTIBYTE=1` (DEFAULT ON) already delivers multi-byte DIV/MOD inside
the existing `FlattenedDivMod` long-division composite — a real MSB→LSB
pipeline that reads the dividend as a full 8-nibble GE vector and recovers
byte 1 from `STACK0_BYTE_VAL_1_LO/HI` via the `BDToGEConverter`, with the
compute relocated to L11 (after the PSH broadcast populates the high byte).
**Dimension-neutral** (a relay + a converter source swap, no widen). Verified
lever: +51 (div 21→48/50, mod 24→48/50), smoke 51/0. Per-nibble DIV is
mathematically impossible (`0xFF/0x0F` counterexample,
`docs/DSL_W5_MULDIV_LIMIT.md`) so a flat DIV lookup was never the path; the
long-division composite is the correct construct and it is already multi-byte.

---

## 4. RECOMMENDATION + Phase-2 build

### 4.1 Path

**Option B for MUL; DIV/MOD already done.** Do NOT pursue the MUL multipass
cascade (`C4_MUL_MULTIPASS`) even though its widen is now proven neutral:
it is a 7-pass `Sequential` carrying the same cross-pass-carry numerical
fragility that bit DIV (the operand-clamp precedent shows the class of bug),
and it duplicates a capability the flat lookup already delivers within
existing dims. The flat lookup is simpler, single-pass (no cross-pass carry to
destabilize), and dimension-neutral.

### 4.2 Exact Phase-2 build

1. **Wire the width=2 flat MUL lookup into `lookup` mode.** In
   `make_mul_partial_op.bake` (`ops/l11_ops.py:613`), when a new
   `mul_lookup_byte1_enabled()` flag is ON (DEFAULT OFF for byte-identity),
   replace the byte-0-only `_mul_partial_rules` bake with a
   `wide_mul_rules(width_bytes=2, ...)` bake:
   - `operand_a_base="ALU_LO"` (A lo+hi nibbles at +0/+16),
     `operand_b_base="AX_CARRY_LO"` (B lo+hi at +0/+16) — the SAME operand
     bands the efficient path and the multipass cascade read (verified
     byte-identical routing in `_install_multipass_mul`).
   - `result_base="OUTPUT_LO"` (byte-0 lo→OUTPUT_LO, hi→OUTPUT_HI),
     `result_byte1_lo_base="MUL_RESULT_HI_LO"`,
     `result_byte1_hi_base="MUL_RESULT_HI_HI"`.
   - `opcode_gate=dim_ref("opcode_flag","MUL")`, `marker_gate="MARK_AX"`
     (the L13 relay + operand gather live at MARK_AX in lookup mode; NOT
     MARK_SE_ONLY, which the byte-0 mul_partial migrated to).
   - `operand_a_artifact_blocker_weight=3.0` + the tuned 5-way-AND weights
     (ALU 0.6 / AX_CARRY 6.0 / marker 4.0 / thr 19.5, or 19.0 with the
     `C4_MUL_W2_THRESH_FIX` follow-up) from the efficient-path install.
2. **Gate off the now-redundant byte-0 writers** when the flag is ON (the
   flat lookup owns every result nibble): the L10 mul_lo + L12 mul_combine
   byte-0 writers — mirror the `mul_multipass_enabled()` gating already in
   `make_mul_partial_op` and the L10/L12 ops.
3. **No residual-band or widen work.** `MUL_RESULT_HI_*` + the L13 relay are
   already default-ON; do not register a new band; do not touch
   `full_vm_compiler_dynamic.py`.

### 4.3 Verification gates (Phase 2)

- `tools/_isa_golden_hash.py` — flag-OFF MUST stay `f725c06e` (byte-identity).
- `tools/lint_cross_op_ffn.py --flag C4_MUL_LOOKUP_BYTE1 --expect OP_MUL` —
  the L11 MUL block writes shared OUTPUT_LO/HI that add/sub/div read one block
  later; assert their read band is unperturbed (the documented mul-l14
  entanglement class).
- `tools/flag_regression_gate.py --flag C4_MUL_LOOKUP_BYTE1` — per-cluster
  campaign-config ON-vs-OFF via `cpu_full_trace`; watch add/sub/div/mod for
  `ok→fail`.
- `tools/cpu_full_trace.py --ids <mul 16-bit ids>` — confirm the byte-1
  emit lands (mul_overflow 100×5=500=0x01F4 both bytes).

### 4.4 Estimated program count

| cluster                    | in scope? | rationale |
|----------------------------|-----------|-----------|
| MUL 16-bit (byte-1 product)| **YES**   | the gap Option B closes; ~25–50 progs (mul + mul_overflow band) |
| DIV 16-bit                 | already ON| `C4_DIV_MULTIBYTE` (+51 landed): div 48/50 |
| MOD 16-bit                 | already ON| `C4_DIV_MULTIBYTE`: mod 48/50 |
| `expr_mul_div` (850-874)   | **NO**    | root = STACK0 cross-step frame persistence (#221), byte-0, 2nd op reads stale `a`; a high-byte fix cannot help (`docs/EXPR_MULDIV_ROOT_IS_STACK0_PERSISTENCE_2026_06_15.md`) |
| `expr_mod` (875-899)       | **NO**    | same STACK0-persistence root |

**Honest lever for THIS project ≈ the MUL 16-bit byte-1 cluster only** (order
tens of programs, not ~150). The ~150 estimate in the brief conflated (a)
DIV/MOD 16-bit — already covered by the default-ON `C4_DIV_MULTIBYTE`, and
(b) the expr_mul/div/mod clusters — blocked by an unrelated STACK0
cross-step-frame root that no ALU high-byte work can move. Scoping to the true
MUL byte-1 lever avoids a Phase-2 that ships a correct fix yet moves the
corpus count by ~0 because the counted programs fail upstream of the ALU.

---

## 5. Files touched (Phase 2) — none in Phase 1

Phase-1 (this doc): READ-only. Golden `f725c06e` unchanged.

Phase-2 build surface: `ops/l11_ops.py` (`make_mul_partial_op.bake`),
`ops/shared.py` (new `mul_lookup_byte1_enabled` predicate),
`ops/l10_ops.py` + `ops/l12_ops.py` (gate byte-0 writers off under the flag),
BOTH cache-key snapshots in `full_vm_compiler_dynamic.py` (register the flag,
NOT a band). Reuses `wide_alu_dsl.wide_mul_rules(width_bytes=2)` unchanged.
