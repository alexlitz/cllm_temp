# var_simple root cause investigation: L0-L2 search (2026-06-02)

## Diagnostic confirmation

`bash tests/runners/run_1096_diag_chunk.sh var_simple 250 24` confirms all 25
var_simple cases fail uniformly:

    step0:SP_byte2 abs=105 gen=13 expected=0x00 neural=0x01
    residual_diagnosis=first-loss-after-symbolic-support block=3 layer=3 width=134
    OUT_LO[0]=-0.00 arg=1/+0.94  OUT_HI[0]=+0.94 arg=0/+0.94
    violations=OUTPUT_LO[0]/active_margin_low + OUTPUT_LO[1]/inactive_too_high

Block=3/layer=3/width=134 = the L3 FFN block; the lesion materializes inside L3.

## Where +0.94 at OUTPUT_LO[1] originates

**Proximate writer: L3 FFN unit 12, rule `layer3_ffn.sp_byte_1_first_step_lo`**
(`ops/l3_ops.py:261-267`). Despite the name "sp_byte_1", the layout comment at
line 46 documents it as **"SP byte 2 first-step (LO=1, HI=0)"** — it writes
`OUTPUT_LO+1 = 2.0/S` when `H1+_SP_I AND BYTE_INDEX[1] AND NOT HAS_SE` fires
(BYTE_INDEX_1 ≡ byte-position 2). 2.0/S * saturation ≈ +0.94, matching the
observed magnitude exactly.

**Why the rule exists / why it's load-bearing**: STACK_INIT = 0x10000 (vm_step.py:3168,
3206). Little-endian byte 2 of the initial SP = 0x01, so writing
`OUTPUT_LO+1=1` at SP_byte_2 first-step is CORRECT for the pre-prologue
initial-SP shape. The downstream depends on this residue (memory note's
warning).

## L0-L2 search results (negative)

- **OUTPUT_LO/HI writes**: L0/L1/L2 ops do not touch `OUTPUT_LO+*` directly.
  Confirmed via grep across `l0_ops.py`, `l1_ops.py`, `l2_ops.py`.
- **Attention V/O slots**: L0 threshold-attn writes H0..H7 (per-marker
  threshold banks); L1 writes L1H0..L1H4 + HAS_SE + IN_STEP_FRESH; L2
  writes BYTE_INDEX_1/2/3 + STACK0_BYTE1/2/3 + MEM_VAL_B0..3. **No
  attention head in L0-L2 writes OUTPUT_LO/HI.**
- **Embedding-time bake** (`model_ops._embedding_bake_rules`): per-token
  writes are CONST + MARK_<X> + IS_MARK + EMBED_LO/HI/CLEAN_EMBED_*/IS_BYTE
  only. No OUTPUT_LO write at SP-related tokens.

## The real lesion is L6 ENT byte-2 override under-firing

`l6_ent_first_step_sp_byte2_lo` (`l6_ops.py:1213,1224-1230`) is the canonical
SP byte-2 ENT override: writes `OUTPUT_LO+0 = 5.0/S` when `OP_ENT AND H1+2
AND BYTE_INDEX_1 AND IS_BYTE AND NOT HAS_SE`. Strength 2.5× the L3 default,
so when it fires it dominates and corrects the byte to 0x00.

Observed `OUT_LO[0] = -0.00` indicates L6's override contributes 0. Either
OP_ENT, H1+2, IS_BYTE, or BYTE_INDEX_1 is absent at abs=105 at the L6
phase. L6 is below the L3-residual-snapshot block, so the lesion shows up at
L3 (where the default has already written +0.94 and nothing has cancelled
it). Subsequent layers don't repair it.

## Recommendation (do not apply)

The L0-L2 search returned no direct writer or embedding bake into
`OUTPUT_LO+1`. The +0.94 lives in L3's default. The true diagnostic
target is **why L6 `l6_ent_first_step_sp_byte2_lo` doesn't fire at
abs=105 gen=13** — most likely OP_ENT/CMP staleness at the SP_byte2
prediction row (cf. recent Phase 8.A `layer6_routing_ffn: CMP read ->
CMP_PREV_STEP`, commit ab323573 — metadata-only per the commit msg, but
worth verifying the runtime CMP residue path at gen=13). A targeted
probe should dump `OP_ENT`, `H1+2`, `IS_BYTE`, `BYTE_INDEX_1`, and
`HAS_SE` activations at abs=105 immediately before L6 to identify which
condition is missing. Per the zero-sum rule, no fix attempted.
