# Post-ENT desync is now a 34-token SHORT step (STACK0 byte-3 dropped), NOT a
# 37-token over-emit — and the byte-3 fix is a clean func/nested win but a
# net -1 var TRADE on exit_code (2026-06-16)

Worktree base / HEAD: `348d7ad9`. Smoke at HEAD and at the landed
(default-OFF) state: **51 passed / 0 failed**
(`CUDA_VISIBLE_DEVICES=1 python -m pytest tests/test_smoke.py`). The model
built by the smoke gate is the d_model=1090 / **52-physical-block** efficient-ALU
build (the 37-block / d_model=872 numbers in older docs are stale). All evidence
below is spec_k=0, hook-free, FREE-RUN (`GroundTruthProbe.probe` == `run_batch`
spec_k=0) at BUILT `model.embed._dim_positions`.

## TL;DR — the failure mode INVERTED since 2026-06-14

`POST_ENT_DESYNC_REAL_PRODUCER_2026_06_14.md` localized a **37-token
OVER-emit** (2 extra leading 0xFF) at the post-ENT STEP_END row. The building
blocks landed since then (`C4_PSH_ARG_VAL_AX`, `C4_AX_BYTE23_DUMP`,
`C4_BP_SAVE_DUMP`, the L15 LI fixes, `C4_ENT_SP_BYTE1_FF_H1_HARDEN`) changed the
frame so the CURRENT desync is the **OPPOSITE**: a **34-token SHORT step** at the
PSH-of-argument store (func_identity/func_add/nested **step 3**). The 4th STACK0
value byte is dropped and the MEM marker is emitted one position early, so the
fixed-35-token slicer mis-aligns and the verifier reads `got pc=None` at the
next checked step. Use `tools/_probe_freerun_tokens.py <id>` (NEW) to see the
per-step counts — the teacher-forced `_probe_func_tokens.py` cannot show this
(its steps are 35 by construction).

```
func_identity_0 (550) FREE-RUN per-step token counts at HEAD:
  step 0..2: 35   step 3: 34  <== SHORT (STACK0 byte-3 dropped)   step 4..8: 35
```

## Root (precisely localized, dim-by-dim)

1. The STACK0 byte-3 value token is predicted at the **byte-2 query row**
   (`BYTE_INDEX_2` ~0.97) of the PSH step. At that row the LM head wants token 0
   (value byte 0) but `logit[0] = -5325` so the MEM marker (261, logit -9.6)
   wins. The crush is driven by `OUTPUT_LO+0`/`OUTPUT_HI+0` (dims 69/85) = **-532**
   (`W_head[0, 69/85] = +5.0`).
2. The -532 enters at **physical block 16 = logical L11 ATTENTION**, head 3 =
   `layer10_psh_stack0_passthrough_bake.head_3` (l10_ops.py
   `_layer10_psh_stack0_passthrough_head_spec`). This head is the PSH STACK0
   store-value passthrough: gated on `PSH_AT_SP` + `IS_BYTE` (slot 33), it
   attends from each STACK0 byte-h query row to the BYTE_INDEX-matched AX byte
   source row and relays its OUTPUT band (net = **3× source OUTPUT**, the
   CLEAN_EMBED routing slots 0-31 + the OUTPUT-CLEAN_EMBED diff routing slots
   32-63 sum to 3×OUTPUT).
3. For the byte-3 relay the matched source is the **AX byte-3 row (off9,
   `BYTE_INDEX_3`)** — which is the row right before the SP register marker. The
   L10 `NEXT_SP` OUTPUT-darkening unit (**block 14 = logical L10, FFN unit
   3400**, `W_down[OUTPUT_*+k] = -1` gated on `NEXT_SP`, present on this row in
   EVERY step) has driven that row's whole OUTPUT band to **-218**. Head 3
   relays 3×(-218) = **-654** into the STACK0 byte-3 decode row, crushing the
   value-0 token. This NEXT_SP darkening is LEGITIMATE (it makes off10 emit the
   SP marker, not a value byte) and must not be removed — the bug is that head 3
   relays it.

Confirmation (`tools/_probe_off24_compare.py 550 2 3`): CLEAN step-2 byte-2 row
has `OUTPUT_LO[0]=+0.94`; BROKEN step-3 byte-2 row has `-651.99 @ block16`.
Ablation (zero head-3 O at that row): `logit[0]` flips from -5325 to +2.2e10 and
the byte-3 value token wins -> 35 tokens.

## The fix (flag `C4_PSH_STACK0_BYTE3_RELAY_DARKEN`, DEFAULT-OFF)

A hard subtractive NOT-blocker on a fully-free Q/K slot of head 3, keyed on
`BYTE_INDEX_2` (`Q[8] = -2e9·BYTE_INDEX_2`, `K[8] = CONST`), mirroring the
slot-7 register-marker darkening already in this head. On the byte-2 query row
(the only row predicting byte-3) the head softmax1-zeros, so the clean OUTPUT
default (+0.94) survives and the byte-3 value-0 token wins. byte-0/byte-1/byte-2
relays (written at the byte-0/byte-1 query rows) and all non-PSH rows are
untouched. **Flag-off is byte-identical to HEAD** (param-hash
`88ae0573…` == HEAD).

EFFECT (flag ON, criterion full_trace, GPU 1):
* `func_add` (575): divergence **step 4 (`got pc=None`) -> step 11**
  (pc=66 CORRECT; only the AX value is wrong now). 12 leading steps all 35
  tokens, all PCs correct.
* `nested_quad` (950-974): every step-3/8/13 34-token SHORT step -> 35; **PC
  correct (186) across the whole cluster**; the remaining `ax=0` is the
  downstream LI-from-frame VALUE wall.
* `func_identity` exit_code: 10/10 (unchanged); smoke 51/0; SI/SC/LI/LC +
  add/sub/mul/div all green.

## THE TRADE — why it is held DEFAULT-OFF (the documented wall)

On the strict exit_code gate this fix is **net -1**: `var_simple_11` (id 261,
`x = 961`) PASSES exit_code at HEAD purely by **FRAMING LUCK**. Its store ALSO
34-token-SHORTs at step 3, but a SEPARATE **multi-byte step-4 mega-over-emit**
(56 tokens — a fully DUPLICATED register block: two REG_AX / REG_SP / STACK0 /
MEM sequences, triggered by the multi-byte stored value 961 = 0x3C1) happens to
re-align the exit readback back to 961. Fixing the step-3 SHORT exposes that
step-4 wall and the exit mis-aligns (961 -> 65512).

**The byte-2 query row is BYTE-IDENTICAL between a func-arg PSH and a var SI/SC
store** in every local marker dim (`BYTE_INDEX_2`, `PSH_AT_SP`, `IS_BYTE`,
`MARK_STACK0`, `OP_*`, `MEM_STORE` — verified equal for 550/575/950 vs 261). The
ONLY separator is the L10 store-value thermometer `STACK0_B0_H1_PREV`
(dims 983-989): ~+2876 (summed) on an SI/SC store byte-2 row vs ~+40 on a func
PSH push. A thermometer-cancel Q term to restore the var path was attempted and
REJECTED: the +8.3e5/dim slot-8 cancel leaks onto other rows (the thermometer is
active off the byte-2 row too) and broke 261 worse (step-2 -> 245 tokens). So
**the byte-3 relay cannot be cleanly separated func-vs-var at this head.**

### The real wall for the NEXT agent

The var-store **multi-byte step-4 mega-over-emit** (the duplicated register
block for values > 255) is a separate, deeper desync that must be fixed FIRST.
Once it is gone, `var_simple` no longer depends on the step-3 framing luck and
`C4_PSH_STACK0_BYTE3_RELAY_DARKEN=1` becomes a clean net win (it can be flipped
default-ON). The framing fix is landed flag-gated-off (HEAD-identical) as the
verified FRAMING half, exactly like `C4_POST_ENT_SE_SUPPRESS` before it.

Note also: even with the framing fixed, func/nested produce **0 new full_trace
PASSES** — they advance the divergence point but still hit the downstream
LI-from-frame VALUE wall (`ax=0`). So the framing fix alone is net-neutral on
full_trace passes and -1 on exit_code; it is a building block, not a pass win.

## New diagnostic tools (all `tools/_probe_*`, spec_k=0, hook-free)

* `_probe_freerun_tokens.py <id> [ms]` — FREE-RUN per-step token counts
  (the SHORT/OVER desync is directly visible; the canonical repro tool).
* `_probe_freerun_step.py <id> <step> [ms]` — full per-token dump of ONE
  free-run step with per-token top-2 logits + the 35-token layout offsets.
* `_probe_stack0_short.py <id> [ms]` — locates the 34-token SHORT step's off24
  row, LM-head logit attribution of token 0 (the crushed value byte), and a
  block-by-block trace of the dominant suppressor dim.
* `_probe_off24_compare.py <id> <clean_step> <broken_step> [ms]` — OUTPUT byte-0
  (dims 69/85) at the byte-2 row, CLEAN step vs BROKEN step, at the final
  residual and at block 16 (the L11 head-3 write point).
