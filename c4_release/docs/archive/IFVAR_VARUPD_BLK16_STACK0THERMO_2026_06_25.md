# if_var GT-result + var_update step-12: corrected block-16 root + STACK0-thermometer discriminator

Date: 2026-06-25. Campaign config (DEFAULT: `C4_NO_STACK0_EMIT=1
C4_OPERAND_FROM_MEMSP=1`), spec_k=0, GPU1. Built from main `8c458f9c`; the
MEM-marker recovery is committed at `760e7d98`. NEW golden `79292fe5`
(flag-OFF byte-identical).

## What landed (committed)

`make_no_stack0_mem_marker_output_clear_op` recovered from the abandoned
`agent-a02e5d5f` worktree (was uncommitted, at risk of loss) and committed at
`760e7d98`, gated by `C4_NO_STACK0_EMIT` + `C4_MEM_MARKER_OUTPUT_CLEAR`
(default ON under the campaign flag). It clears OUTPUT at the MEM-marker-
predicting row in the 30-token frame so the SI-store / BZ-branch step emits
`Token.MEM` (restoring the NEXT_MEM->NEXT_SE->NEXT_PC chain) instead of a stray
leaked ALU-result byte — fixing the var_update step-13 / if_var step-11 SILENCE
collapse.

Gates (all green): flag-OFF `_isa_golden_hash` = `79292fe5` (byte-identical);
default build +32 units; `run_full_smoke` 41/41; GPU full_trace
`325-349,425-449,250-274,350-399` = **82/125** with **var_simple 25/25, if_gt
25/25, if_lt 25/25** all HOLD, **if_var 7/25** (recovered from silence
collapse), var_update 0/25.

## Residual #1 — if_var GT-result: CORRECTED root (block 16 is FFN, not attention)

The prior doc (`IFVAR_VARUPD_PCFRAME_2026_06_24.md`, on branch
`worktree-agent-a02e5d5f`) localized the flip to "physical block 16 (L11)
ATTENTION" and blueprinted an attention re-scope. **That is wrong.** Per the
validated 37-block map (`docs/PROBE_GROUNDTRUTH_2026_06_10.md`), **physical
block 16 is a logical-L14 post_op EXPANSION passthrough** — its attention is
zero-init (`x + attn(x) = x`, residual identity). So the block-16 OUTPUT_LO
cmp-result write CANNOT be an attention head; it is the **L14 post_op FFN**.
(`tools/probe_blk16_structure.py` confirms the zero-init attn + post_op ffn
structure on the built model.)

### The discriminator (hard evidence)

`tools/probe_ifvar_blk16_indiff.py` dumps the FULL post-block-15 residual (the
INPUT to block 16) at the GT-result SE row, for the FAILING `ifVAR 66>24` vs
the PASSING literal `ifGT 66>24`, and prints every dim that differs. Result —
the cmp gates (SE_OP_GT, SE_CMP_GROUP) and the CMP cascade are ~identical; the
**dominant differing band is the STACK0 byte-0 cross-step thermometer**:

```
  dim 1007-1010 [STACK0_B0_H3_PREV+0..3]  ifVAR=+567.28   ifGT=~0   d=+567.28
  dim 1002-1006 [STACK0_B0_H1_PREV+2..6]  ifVAR=+567.28   ifGT=~0   d=+567.28
  dim 1004      [STACK0_B0_H1_PREV+4    ]  ifVAR=+567.28   ifGT=+160 d=+407.28
  dim  671      [CMP.*.-1+1            ]  ifVAR=+14.55    ifGT=+7.73 d=+6.82
  dim  826/841  [SE_ALU_HI+0 / +15     ]  ~±4.7
```

`STACK0_B0_H1_PREV` / `STACK0_B0_H3_PREV` (band owner
`make_stack0_byte0_dump_carry_op`, l11_ops.py:122; the `C4_STACK0_B0_DUMP`
default-ON carry) is the documented **L10 store-value thermometer** (l10_ops.py:579,
"~+2876 for an SI store vs ~+40 for a func PSH"). In the ifVAR program the
`LI x` loads the variable onto STACK0, so this thermometer is populated (+567);
in the literal ifGT there is no load, so it is ~0.

### Mechanism

At the GT-result SE row, block 16's FFN writes OUTPUT_LO. The +567 thermometer
band perturbs that write:

```
block 16   ifGT (PASS): OUTPUT_LO cell0=+2.0, cell1=+3.0 -> argmax cell1 -> result 1 ✓
           ifVAR (FAIL): OUTPUT_LO cell0=+5.0, cell1=+0.0 -> argmax cell0 -> result 0 ✗
```

(matches the +240 SE-clear-relative dump: ifGT -237.997/-237.003, ifVAR
-235.001/-240.000.)

### Where the band actually couples in

No FFN op *conditions* on `STACK0_B0_*_PREV` to write OUTPUT_LO (grep is clean
across all `ops/*.py`). So the +567 band does not directly gate the block-16
unit. It couples in via an **earlier-block (0..15) attention head that has the
`STACK0_B0_*_PREV` dims in its K/V** — the +567 magnitude shifts that head's
softmax over the loaded-variable's stack/memory rows, which moves OUTPUT-band
mass that block-16's FFN then reads. This is the same softmax1-normalization
cross-op entanglement the cross-op-attention lint guards. Pinning the exact
head needs a per-block hook-split of blocks 0..15 with the thermometer band
ablated (a hook-based probe; the no-hook `stop_after_block` path cannot ablate
a band mid-stream).

### Fix blueprint (campaign-gated, DEFAULT OFF)

Two viable surfaces — prefer (A):

A. **Thermometer NOT-blocker at the cmp result row (FFN).** Add a campaign-gated
   FFN unit (or extend the existing L14 cmp-result OUTPUT_LO writer) that, on a
   comparison step at the SE result row (gate: MARK_SE_ONLY + SE_CMP_GROUP/
   SE_OP_GT), drives the OUTPUT_LO cmp-result cells from the CLEAN cmp result
   band ONLY, with a hard NOT-blocker on `STACK0_B0_H1_PREV+*` /
   `STACK0_B0_H3_PREV+*` so the loaded-var thermometer cannot bleed in — the
   FFN counterpart of `CmpOperandSeRecoverFFN`. This is the cleanest because the
   cause (the +567 band) and the symptom (block-16 OUTPUT_LO) are both nameable.

B. **Sink the thermometer at the comparison step.** Campaign-gated FFN that
   zeroes `STACK0_B0_*_PREV` on a comparison step's SE row (it is dead there —
   no store/load is happening on a cmp step). Risk: the band is also read by the
   func/var store-vs-PSH discriminator (l10_ops.py:579), so the sink MUST be
   scoped to MARK_SE_ONLY + cmp-opcode rows only.

Gates (MANDATORY): fresh flag DEFAULT OFF -> `_isa_golden_hash` = `79292fe5`;
`lint_cross_op_attention.py --flag <f>` (if_gt/lt/eq + add/sub softmax
unchanged — the band feeds a SHARED head); `lint_cross_op_ffn.py` (shared L14
OUTPUT band); `flag_regression_gate.py`; `run_full_smoke` 41/41; GPU full_trace
`325-349,425-449,250-274,350-399` with **var_simple + if_gt + if_lt MUST HOLD,
0 ok->fail**.

## Residual #2 — var_update step-12: ADD result +240 (0xF0) byte-1 leak (NOT the LEA root)

The prior doc said var_update's first divergence is the step-14 `return x` LEA
(`exp ax=0xFFE8 got 57`). With the MEM-marker fix ON the framing is intact and
the **first divergence is now step 12, the `x = x + 7` ADD result**, NOT the
LEA:

```
id=0325 step=12 exp(ax=57)  got(ax=297)   x=50,x=x+7   297-57 = +240
id=0326 step=12 exp(ax=78)  got(ax=318)   x=50,x=x+28  318-78 = +240
id=0327 step=12 exp(ax=27)  got(ax=267)               267-27 = +240
... ALL 25 cases: got = expected + 240 (0xF0) exactly.
```

Per-byte: `expected=0x39 -> got=0x129`: byte0 high-nibble drops 1 (0x39->0x29 =
-0x10) and byte1 gains 1 (0x00->0x01 = +0x100); net +0xF0=+240. So the ADD
result's byte-0 high-nibble is being relayed/duplicated into byte-1 in the var
frame (a byte-1 carry/relay leak specific to the var-frame ADD step, distinct
from both the LEA root and the if_var cmp root). Sequence this with the AX
byte-1 / OUTPUT_HI campaign-frame carry family (tasks #311/#316), NOT the cmp
work.

## Diagnostic tooling added (model byte-identical; tools only)

* `tools/probe_blk16_structure.py` — proves block 16 = zero-init attn + L14
  post_op ffn; measures the block15->16 OUTPUT_LO cmp-result delta.
* `tools/probe_ifvar_blk16_indiff.py` — dumps the post-block-15 residual diff
  ifVAR vs ifGT; isolates the STACK0_B0_*_PREV thermometer as the discriminator.
* (recovered from a02e5d5f) `probe_pcframe_tokdump`, `probe_ifvar_ar_residual`,
  `probe_ifvar_se_op_gt`, `probe_ifvar_gt_operands`.
