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

## Artifacts

- `tools/verify_carry_migration.py` — Phase 1 byte-identity gate (committed).
- `tools/probe_sub_minuend_source.py` — the spec_k=0 probe behind the table
  above (OUTPUT vs STACK0_BYTE_VAL_h at the cascade rows).
