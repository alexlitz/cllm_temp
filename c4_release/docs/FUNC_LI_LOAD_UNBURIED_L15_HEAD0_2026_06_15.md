# func/var frame-local LI/LC load FIXED — was the L15 head-0 suppressor chain, NOT a value relay (2026-06-15)

Worktree base / HEAD before this work: `92a1afc7`. Two commits landed:
`9165de91` (the fix) + `e8170282` (lint baseline). Smoke **51/0** at the
chosen defaults; flag-off byte-identical to HEAD
(`PARAM_HASH=0791d32e...5965042`, `total_elems=458018907`).

## TL;DR — the brief's "byte-0 relay" model was WRONG; the LI consumer is L15

The brief framed the first root as "no `STACK0_BYTE_VAL_0` band / no byte-0
broadcast head — teach the L11 block-13 LI-load relay to read it." **There is no
L11 LI-load relay.** The frame-local `LI` is the **L15 content-addressable
`memory_lookup`** (heads 0-3, physical block 30). The PSH-arg store value (70) is
ALREADY correctly stored and tagged (pos 243: `MEM_STORE=1`, `ADDR_B0/B1=0xFFE8`,
`CLEAN_EMBED` decodes 70). The LI returned 0 because the **L15 head-0 lookup was
BURIED**, not because the value was missing.

## Root (spec_k=0, BUILT dims, func_identity_0 id550 step-7 LI)

The L15 head-0 load works cleanly in isolation (`si_li_roundtrip`: AX=42, head 0
attends the stored value, sink=0). In a FUNCTION frame the callee `ENT` step's
`OP_ENT` **BROADCASTS** a ~0.3-2 residue onto (a) the prior PSH-arg store tokens
and (b) the `LI` query (AX-marker) row. That residue mis-fires a CHAIN of head-0
PC/SP/STACK0-marker discriminator slots whose large NOT-blocker magnitudes (and
store-address-dependent K) bury / mis-order the lookup:

| slot | role | misfire on the func LI row |
|------|------|----------------------------|
| 58 | early-ENT STACK0 discriminator | Q≈-1.2e10; K[58]=1e4·OP_ENT is nonzero ONLY on the OP_ENT-residue store tokens → penalises EXACTLY the PSH-arg store the LI must read (-3.3e13) → softmax1 sink wins → AX=0 |
| 62 | sp_byte_blocker | `q[62,OP_ENT]=5e5` × K=-3e5·CONST → uniform -1.1e11 on every key |
| 35 | pc_byte_blocker | 0xFFE8 e8-signature boost only half-cancelled by its -10000 OP_LI_RELAY guard → -1e9 baseline |
| 34/59/60/61 | pop_low8 / preserve_e8 / top_store_e8 / ax_li_e8 | smaller store-address-dependent biases |

PROOF (live-patch): zeroing head-0 slots {34,35,58,59,60,61,62} makes the func LI
attend the value-70 store (pos 243, weight 1.0) → step-7 AX=70, and
`var_simple_0/12` + `rec_factorial_0` still decode. The address bits (L15 slots
4-27) ALONE pick the right store once the suppressors are out of the way — the
clean si_li ordering.

### Why naïve cancels FAILED (a float32-precision trap worth remembering)

Cancelling slot 58's -1e10 baseline with a +1e10 relay term loses ~1e3 to
float32; ×K=1e4·OP_ENT that residual re-orders stores BY OP_ENT (the genuine ENT
stores, OP_ENT~17, then win) and re-breaks the load. The fix RESCALES slot 58 by
1e4 (1e10→1e6) FIRST so the cancel of two ~1e6 numbers is precision-safe.

## The fix (`C4_L15_LI_SUPPR_INERT`, DEFAULT-ON; head 0 only)

On L15 `memory_lookup` head 0, for each mis-firing suppressor add a **per-
suppressor CANCEL slot** (a free over-width slot 64-70) whose **Q copies the
suppressor's Q** and whose **K = -(suppressor's K)**. The cancel slot's per-key
product is therefore exactly `-(suppressor's per-key product)` on EVERY row →
suppressor + cancel = 0 → the slot is inert (equivalent to zeroing it), but
expressed ADDITIVELY so flag-OFF (cancel slots omitted) is byte-identical with
HEAD. slot 58 is rescaled 1e4 first. Landed in BOTH
`_layer15_memory_lookup_heads_0_3_specs_with_overrides` (declarative) and
`_suppress_l15_lookup_heads_0_3` (imperative; the writer that actually lands via
`make_l15_attention_resize_op`), plus the two compile cache keys.

## Result — LI value root FIXED cluster-wide

| cluster | criterion | flag OFF | flag ON |
|---------|-----------|----------|---------|
| func_identity 550-561 | exit_code | 0/12 | **1/12** |
| var_simple 250-261 | exit_code | 0/12 | **9/12** |
| func_identity 550-574 | full_trace divergence | step **7** (LI, ax=0) | step **8** (LEV) |

`var_simple` exit-code jumps 0→9 because its frame-local LI (read-back of a local)
now delivers the value; the 3 var fails + the var full_trace step-4 PC mismatch
(AX is CORRECT) are the SEPARATE 37-token framing drift, not the LI value.
`func_*` advance from step-7 (LI) to **step-8 (LEV epilogue)** but the EXIT still
fails for most because of the NEXT root.

Smoke 51/0 (si_li/sc_lc/LI/LC green); add/sub/mul guards 7/10 unchanged (the 3
fails are pre-existing 16-bit carry/mul); flag-off byte-identical.

## THE NEXT ROOT — the LEV epilogue (func/nested/rec, NOT var)

`func_identity_0` step 8 (`LEV 0  50->90`): `expected(pc=90,ax=70)` got
`(pc=10, ax=1606)`. TWO problems at the LEV step:

1. **PC restore = 10 not 90.** The saved return address in the frame is
   corrupted: at the callee `ENT` step (step 5) the MEM section decodes
   `[240,90,10,10]` — the return-addr byte 90 + 0x0A garbage leak into the
   saved-BP store. The LEV PC-restore (`make_layer9_lev_bp_to_pc_relay_op` +
   `layer3_carry_forward_attn.head_6`) then reads a corrupt `mem[BP+8]`. This is
   the ENT-frame store-corruption surface (`C4_BP_SAVE_DUMP` territory).
2. **AX byte-1 leak = 0x06** (`0x46`→`0x646`). AX is byte-clean at every prior
   step (steps 2-5,7 all 0x46); the leak appears ONLY at the LEV step, so it is
   a LEV-specific byte-1 producer, NOT the general H1-one-hot dump wall (which
   would leak at every step). This is the EXIT-blocking issue for func_identity
   (the exit = final AX). Values whose result happens to have a clean byte-1
   (e.g. identity(96) id561) already PASS — that is the +1.

Either alone blocks `func_*` exit. `var_*` has no LEV-after-LI so it already
passes. Next agent: probe the LEV step's saved-frame read (return addr) and the
LEV-step AX byte-1 producer; both are downstream of the (now clean) LI value.

## Reproducers

```
# LI now delivers the value (was AX=0):
CUDA_VISIBLE_DEVICES=0 C4_BP_SAVE_DUMP=1 C4_ENT_SP_BYTE1_FF_H1_HARDEN=1 \
  C4_PSH_ARG_VAL_AX=1 C4_AX_BYTE23_DUMP=1 C4_POST_ENT_SE_SUPPRESS=1 \
  C4_L15_LI_SUPPR_INERT=1 python tools/_probe_func_li.py 550   # step 7 AX=0x46=70

# var_simple exit-code 0/12 -> 9/12:
... C4_L15_LI_SUPPR_INERT=1 python tools/run_1096_canonical.py --ids 250-261 --criterion exit_code

# byte-identity (all flags off == HEAD):
CUDA_VISIBLE_DEVICES=0 C4_BP_SAVE_DUMP=0 C4_ENT_SP_BYTE1_FF_H1_HARDEN=0 \
  C4_PSH_ARG_VAL_AX=0 C4_AX_BYTE23_DUMP=0 C4_POST_ENT_SE_SUPPRESS=0 \
  C4_L15_LI_SUPPR_INERT=0 python tools/probe_model_param_hash.py
```
