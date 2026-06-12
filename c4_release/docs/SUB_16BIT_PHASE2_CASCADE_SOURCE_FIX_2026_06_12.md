# sub_16bit Phase 2: the cascade-minuend source fix (design + why it is multi-session)

**Date:** 2026-06-12 (spec_k=0, hook-free, GPU 0). **Status:** Phase 1
(declarative migration of the L14 carry/borrow cascade) LANDED and
byte-identical. Phase 2 (the minuend source fix) is specified here with fresh
probe evidence; it is a multi-component change that cannot be done as the
single FFN same-row source swap the brief assumed, for the empirical reason
below.

## Phase 1 (landed, commit 51a6634c)

`vm_step.CarryPropagationPostOp` (the L14 inter-byte ADD/SUB carry/borrow
cascade) is now authored declaratively. The live model's
`make_l10_post_op_attach_op` builds the three carry post-ops via
`_build_l10_carry_post_op` → `Primitives.lower_ffn_rules` over the
pre-existing `_l10_carry_propagation_rules`. The imperative `_bake_weights`
loop is replaced by a delegation to the same lowering (legacy `_SetDim` path
builds the dim map from `_SetDim`). vm_step.py shrinks 9158 → 9056 lines.

Byte-identity gate `tools/verify_carry_migration.py`: element-wise tensor
diff = 0 vs the legacy bake for all three `(byte_idx, cascade)` instances in
BOTH the compact (d_model=870) and legacy (`_SetDim`) layouts;
lowering-contract OK. Smoke unchanged: 35 pass / 6 fail / 8-of-8 guardrails.

## The Phase 2 premise is empirically FALSE as a same-row FFN swap

The brief said: change the cascade's per-byte minuend READ for bytes 1/2/3
from `OUTPUT_LO/HI` to `STACK0_BYTE_VAL_1/2/3` (dims 734+). An FFN rule only
sees its OWN sequence row's residual. **At the SUB byte-1/2/3 EMIT rows the
`STACK0_BYTE_VAL_h` band is 0x00 for every program** — the discriminator is
not there.

Probe `tools/probe_sub_minuend_source.py` (spec_k=0), sub_16bit `(0x100-1)`
vs sub_borrow `(0-1)`, AX result at `REG_AX@155`, byte-1/2/3 emit-predict
rows = 156/157/158:

| band @ row 156 (predicts byte 1) | sub_16bit | sub_borrow |
|----------------------------------|-----------|------------|
| `STACK0_BYTE_VAL_1_LO/HI`         | **0x00**  | **0x00**   |
| `OUTPUT_LO/HI`                    | 0x00      | 0x00       |
| `ALU_LO/HI`                       | 0x00      | 0x00       |
| `AX_FULL_LO/HI`                   | empty     | empty      |
| `CLEAN_EMBED_LO/HI`               | 0xFF      | 0xFF       |
| `MEM_VAL_B1`                      | 0x0a      | 0x0a       |

Every value band is byte-identical between the two programs at the cascade
input row. The 0x01-vs-0x00 discriminator exists ONLY at the PSH-frame STACK0
byte-1 rows (66/101) and is **never relayed** to row 156 at any of physical
blocks 12/14/26 (confirmed by an all-rows scan of `STACK0_BYTE_VAL_1`).

So the cascade cannot read the discriminator at its own row: it must be
GATHERED there by an attention relay first. That is exactly the L13-head
`layer13_sub_byte1_minuend_gather` that three prior agents built and reverted.

## Why the obvious relay band (OUTPUT / AX_FULL) re-introduces the collision

The cascade relays the byte-0 borrow-out via `CARRY+3` through the OUTPUT
band (probe: `CARRY+3 = +172.94` at row 156, block 18 — the cross-row borrow
that triggers the byte-1 stage). Any relay that writes the minuend into
`OUTPUT_LO/HI` at row 156/157 perturbs that relay → the byte-1 borrow stage
stops firing → sub_borrow regresses 0xFFFFFFFF→0xFFFF (the documented Wall-2
result; `SUB_16BIT_ROOT_IS_L14_BORROW_CASCADE_NOT_PSH_STORE_2026_06_12.md`).

`AX_FULL_LO/HI` (471/487) is NOT a safe alternative: it is read by
`layer15_alu_high_byte_relay` (L15 head 8) and relayed straight into OUTPUT
byte-1. Widening the L13 OR/XOR gather (head 3) to also fire on SUB would put
the value into AX_FULL — and L15 would then relay it to OUTPUT, reproducing
the same CARRY+3 collision. AX_FULL is the OUTPUT-byte-1 path, not a
decoupled scratch.

## The exact Phase 2 change (decoupled-scratch, multi-component)

Now that the cascade is declarative, the clean fix is a co-design that keeps
CARRY+3 on OUTPUT and routes only the MINUEND through a fresh scratch band:

1. **New residual dim** `SUB_MINUEND_h_LO/HI` (h ∈ {1,2,3}), a 16-wide
   one-hot nibble pair per byte — a scratch band read ONLY by the cascade
   SUB byte-h minuend rules, written ONLY by the relay head in (2). It must
   NOT alias OUTPUT/ALU/AX_FULL/CARRY. **Blocker:** the compact layout is
   full (`d_model=872`, max named dim = 869 `_pad`); there is no free slot,
   so this requires widening d_model (or reclaiming a provably-dead band).
   This is the single biggest reason Phase 2 is multi-session.

2. **New attention head** (model on `_layer13_bitwise_byte1_gather`, l13_ops
   head 3): Q fires at the SUB byte-h emit row (`TEMP+9` = relayed SUB AND
   `BYTE_INDEX_h`), K selects the STACK0 byte-h value row with a large
   magnitude, V copies `STACK0_BYTE_VAL_h_LO/HI`, O writes
   `SUB_MINUEND_h_LO/HI`. ALiBi recency tie-break to the current frame's PSH
   (later carry rows decay to 0x00). Gate hard on every non-SUB-byte-h row.

3. **Cascade rule change** (the now-declarative
   `_l10_carry_propagation_rules`, `sub_rule_for`, for `byte_idx ∈ {1,2}`
   i.e. `cascade=True`): change the two OUTPUT-match condition terms from
   `("OUTPUT_LO+lo", 20.0)` / `("OUTPUT_HI_THIS_STEP+hi", 20.0)` to
   `("SUB_MINUEND_h_LO+lo", 20.0)` / `("SUB_MINUEND_h_HI+hi", 20.0)`. KEEP
   the `CARRY+3` gate, the wrong-byte blockers, and the OUTPUT result
   `writes` unchanged — only the minuend SELECTOR moves off OUTPUT. The
   byte-0 stage (`cascade=False`) is untouched (OUTPUT byte 0 is correct).

### Byte-identity property (the trap the fix must thread)

For 8-bit SUB (sub_basic, 50-8) the true byte-1 minuend is 0x00, so the relay
writes `SUB_MINUEND_1 = 0x00` = the current OUTPUT byte-1 = 0x00 → identical
result. For sub_16bit the relay writes 0x01 (from STACK0_BYTE_VAL_1) so the
byte-1 SUB rule fires the `0x01 - borrow = 0x00` cell instead of
`0x00 - borrow = 0xFF`. For sub_borrow the relay writes 0x00 → byte-1 stays
0xFF and the borrow CONTINUES — sub_borrow stays 0xFFFFFFFF. The
discriminator that makes 0x100-1 ≠ 0-1 is `STACK0_BYTE_VAL_1 = 0x01 vs 0x00`,
carried cleanly through the scratch band, never touching CARRY+3.

## Why it was not landed this session

- The same-row FFN swap the brief specified is architecturally impossible
  (value absent at the cascade row — proven above).
- The decoupled relay needs a NEW residual dim, but the compact layout has no
  free slot (d_model widening required) — a model-shape change, not a rule
  edit.
- The relay + cascade co-design is the exact multi-session Wall-2 surface
  three prior agents hit; landing a partial version (OUTPUT/AX_FULL relay)
  regresses sub_borrow, violating the non-negotiable zero-regression gate
  (`feedback_single_rule_fixes_are_zero_sum`).

Phase 1 is the solid, byte-identical landing; Phase 2 is fully specified
above and is now a clean cascade-rule edit (3) once the scratch dim (1) +
relay head (2) exist.

## RESOLUTION (2026-06-12, LANDED — sub_16bit PASS, zero smoke regression)

Phase 2 landed WITHOUT a new dim or d_model widening. Two premises in
"Why it was not landed" above were wrong:

1. **No new scratch dim is needed.** The decoupled band the cascade reads
   is the *existing* `STACK0_BYTE_VAL_{h}` family itself (enabler #2 in the
   brief). A new L13 attention head (`layer13_sub_minuend_relay`, head 4)
   re-deposits `STACK0_BYTE_VAL_{k+1}` from the populated PSH frame back
   INTO `STACK0_BYTE_VAL_{k+1}` at the SUB byte-emit row. STACK0_BYTE_VAL
   is neither OUTPUT nor AX_FULL nor CARRY, so the CARRY+3 borrow relay is
   untouched and sub_borrow stays 0xFFFFFFFF. No d_model change.

2. **The byte_idx → output-byte mapping is +1, not identity.** The cascade
   is an INTER-byte stage: the rule scoped on `BYTE_INDEX_k` produces
   **output byte k+1** (the autoregressive predictor row for byte k+1,
   verified via the LM-head decode at OUTPUT_LO=69/HI=85). So output byte 1
   needs `byte_idx=0` reading `STACK0_BYTE_VAL_1` at the `BYTE_INDEX_0` row
   — NOT `byte_idx∈{1,2}` as this doc originally specified. The relay
   delivers byte (k+1) to the `BYTE_INDEX_k` row; the cascade `sub_rule_for`
   (all byte_idx) reads `STACK0_BYTE_VAL_{byte_idx+1}` and EMITS on OUTPUT
   (cancel cell 0 = the un-relayed 0x00, set the computed nibble).

ALiBi note: the runtime applies `-slope * |dist|` (vm_step), so a NEGATIVE
slope rewards distance → selects the deepest/oldest PSH frame. The K
source-flag match is boosted to `K_FLAG=200` so STACK0_BYTE rows dominate
the alibi distance penalty under the ~0.096 score scale; the per-byte Q
selector folds in TEMP+9 so the head is dark off-SUB (STACK0_BYTE_VAL
aliases FORMAT_PTR/LEV_DETECTOR in the compact layout — writing it off-SUB
would regress).

Verified spec_k=0: sub_16bit→0xFF PASS, sub_borrow→0xFFFFFFFF PASS,
sub_basic→42 PASS (byte-identical). `pytest tests/test_smoke.py`:
**46 passed / 3 failed / 2 xfailed** (was 45/4/2; sub_16bit flipped;
the 3 remaining — mul_basic/eq_true/eq_false — are the pre-existing
CMP/mul surface; ZERO regression). 1096 sub cluster: **3/50 → 12/50**
(+9; the multi-byte-minuend cases with a single-nibble high byte AND a
byte-0 borrow). Commits: `feat(l13): ...relay head (Part 1)`,
`feat(l10): SUB cascade reads relayed multi-byte minuend (Part 2)`.

### Remaining (follow-ups, NOT this surface)

- ~~**No-borrow multi-byte SUB**~~ **LANDED 2026-06-12** (commit
  `feat(l14): SUB no-borrow ... passthrough`). The borrow-gated cascade
  cannot reach the no-borrow path (its SUB cells are *multiplicatively*
  gated on the byte-0 borrow-out CARRY+2, so they emit nothing when there
  is no borrow). Adding a non-borrow rule to the cascade is impossible: the
  3 carry instances are pinned to exactly 512 units (256 ADD + 256 SUB),
  no spare cells. The landed fix is a NEW L14 cleanup-chain FFN op
  `layer14_sub_noborrow_high_byte_passthrough` (16 units, one per minuend
  byte-1 nibble): at the SUB byte-1 predictor row (TEMP+9 + H1[AX] +
  IS_BYTE + BYTE_INDEX_0), gated to fire ONLY when CARRY+2 is absent
  (no byte-0 borrow), it reads the relayed minuend byte 1
  (STACK0_BYTE_VAL_1, delivered by `layer13_sub_minuend_relay`) and writes
  OUTPUT byte 1 = that value. Discriminator verified spec_k=0 (block 15,
  SUB byte-1 predictor): no-borrow CARRY+2=0.0; borrow CARRY+2=2.0. 8-bit
  SUB relays 0x00 = the default → byte-identical (sub_basic 50-8 → 42).
  Borrow path untouched (cascade keeps owning it). **1096 sub: 12/50 →
  43/50 (+31)**; pytest smoke 46/3/2 unchanged.
- **2-nibble high bytes** (e.g. 0x1234-0x34, byte1=0x12): the UPSTREAM
  `layer10_psh_ax_broadcast` stores only the LOW nibble of STACK0_BYTE_VAL_h
  (probe: 0x1234 → STACK0_BYTE_VAL_1 = 0x02 not 0x12). The relay faithfully
  copies the truncated value; the fix is upstream in the PSH broadcast head.
  The 7 remaining sub fails are NOT this — they are pre-existing byte-0
  ALU precision / runner-error cases (ids 63-66, 70, 72, 87; minuend byte1
  all ≤ 0x06, single-nibble), not high-byte issues.
- **ADD multi-byte** (add 3/50, NOT landed): ATTEMPTED and reverted
  2026-06-12. Unlike SUB (subtrahend byte1 = 0x00 for every 1096 case,
  so result byte1 = minuend_byte1 - borrow, a single relayed value), ADD
  needs **a_byte1 + b_byte1 + carry** — a TWO-operand add. The signals
  exist at the ADD byte-1 predictor row (verified spec_k=0, teacher-forced
  block 15 r141): a_byte1 relayable into STACK0_BYTE_VAL_1 (a working ADD
  relay was built, head 5 gated on TEMP+8, delivered a1 correctly);
  b_byte1 in ADDR_B1_LO; carry in CARRY+1. A 128-unit FFN adder
  (per (a1,b1)×carry → a1+b1+carry) was written and gated on the carry
  state. **It regressed in the REAL autoregressive runner**: the carry
  discriminator (CARRY+1) that reads cleanly in the teacher-forced residual
  does NOT match in the autoregressive decode path (add_basic 5+3, no
  carry, fired the carry-present rule → byte1=0x01). The autoregressive
  byte-1 predictor row is the byte-0 *token*'s position in the growing
  context, NOT a fixed teacher-forced row, so teacher-forced CARRY+1 is the
  wrong reference. The existing ADD carry cascade also emits a 0x88 default
  byte1 that the adder must override. **Lesson for the next ADD agent:**
  do NOT trust the teacher-forced residual for the ADD carry signal — drive
  the real `BatchedPureNeuralRunner` (spec_k=0) and instrument the actual
  per-step byte predictor, OR find a single-value (copy, not add) ADD
  formulation. The SUB fix worked precisely because it was a single relayed
  copy gated on an absence (no-borrow), not a value-dependent arithmetic
  match.

## Artifacts

- `tools/verify_carry_migration.py` — Phase 1 byte-identity gate (committed).
- `tools/probe_sub_minuend_source.py` — the spec_k=0 probe behind the table
  above (OUTPUT vs STACK0_BYTE_VAL_h at the cascade rows).
- `tools/probe_sub_relay_design.py`, `tools/probe_sub_cascade_rows.py` —
  the spec_k=0 probes behind the RESOLUTION (relay delivery + cascade map).
