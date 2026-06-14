# var step-1 STACK0 leak is L16 ENT-frame OP_ENT-broadcast, NOT L15 (2026-06-14)

Status: **PARTIAL FIX LANDED** (step-1 STACK0 now byte-correct; smoke 51/0).
The remaining var divergence is precisely localized to the step-2 PC byte rows
(same OP_ENT-broadcast rule family). This doc CORRECTS the root attribution in
`VAR_STEP1_STACK0_LEA_DESYNC_2026_06_13.md`, which blamed the L15
`memory_lookup` pop-lookup K-aliasing — that is **refuted** below.

Path: spec_k=0, hook-free, `CUDA_VISIBLE_DEVICES=0`, HEAD `da09f54a`.
Program: `var_simple_0` (id 250, `int x; x=990; return x;`).

## TL;DR — the real root

The step-1 (LEA) STACK0 corruption (`[240,255,15,15]` vs oracle `[0,0,0,0]`) is
**NOT** an L15 attention K-alias. It is a family of **L16 ENT-frame FFN rules**
(in `_layer16_lev_routing_rules`, physical block 32 / logical L20) that gate on
`OP_ENT` and **misfire on the OP_ENT broadcast residue**.

`OP_ENT` is not one-hot in-step: one VM step after the ENT, the opcode marker
still BROADCASTS at ~9.8..17.5 onto every row of the next step
(audit `tools/probe_ent_f0_firing.py`). Three rules with weak `OP_ENT`-dominated
gates fire on rows they were never meant to:

1. `l16_ent_nested_stack0_saved_bp_byte0_f0` — gated on `OP_ENT*10 +
   MARK_STACK0*10` (threshold 80). It NEVER fires on a clean ENT-step STACK0 row
   (OP_ENT=0 there); it fires ONLY on the broadcast (the LEA-step STACK0 marker
   row), forcing `OUTPUT_HI` high nibble = 0xF → STACK0[0]=0xF0.
2. `l16_ent_frame_sp_byte1_ff` — the genuine SP-byte1=0xff override. Its `H1+2`
   (SP marker-distance) selector has weight 1.0, so the OP_ENT broadcast alone
   clears the threshold; it leaked onto the LEA-step STACK0 byte rows
   (STACK0[1]=0xFF at logit ~2.4e11) AND the STEP_END row AND the step-2 PC byte
   rows (the spurious 0xFF that desyncs step 2).

The wrong STACK0 is then AMPLIFIED ~31900x at logical L25 (the final
argmax-forcing OUTPUT bake) to ±1.1e9. The head reads `OUTPUT_HI+15`/`+0`
(byte-token `b` is scored `5*OUTPUT_LO[b&0xF] + 5*OUTPUT_HI[(b>>4)&0xF]`), so
the forced high nibble F wins token 0xF0/0xFF.

### Why L15 is refuted

`tools/probe_var_l15_attn.py` (spec_k=0): L15 `memory_lookup` head 0's attention
on the LEA-step STACK0[0] predictor row is **byte-identical** to its attention
on the WORKING SI/LI roundtrip STACK0 rows (max attn weight 0.0465, sum 0.95,
softmax1-diffuse in BOTH). Head 0 is NOT the differentiator. The leak band is
`OUTPUT_HI` (dim 85/100), written by the L20 (block 32) FFN, traced
block-by-block in `tools/probe_var_l20_ffn.py` / `probe_var_dimlabel.py`.

## The fix landed (byte-identity-safe, smoke 51/0)

`neural_vm/unified_compiler/ops/l16_ops.py`:

* `l16_ent_nested_stack0_saved_bp_byte0_f0`: threshold 80 → **2000** (above the
  OP_ENT-broadcast ceiling). The rule had NO legitimate firing (audit), so this
  is byte-identical everywhere; the empty-stack default (STACK0=0) survives.
* `l16_ent_frame_sp_byte1_ff`: `MARK_STACK0` blocker −10 → **−100** plus new
  **`STACK0_BYTE{0..3}` = −100** hard blockers. The genuine SP byte rows carry
  `MARK_STACK0 = STACK0_BYTE* = 0` (probe_stack0_byte_marker.py), so these are
  byte-identical on the genuine 0xff firing while vetoing the STACK0
  marker/byte misfires.

Result: var step-1 STACK0 now emits `[0,0,0,0]` (oracle-correct);
`full_trace` step 0 AND step 1 now PASS (`tools/probe_var_step2.py`).
Smoke = **51/0** (pytest, incl. all memory SI/LI/SC/LC + test_simple_function).
Per-op byte-identity drift checks (`tests/test_l16_per_op.py`) pass for every
rule family (the 2 unit-count assertions of 728 are PRE-EXISTING stale: the
layer actually has 827 units on baseline; this fix adds zero rules).

## The remaining root (next session)

`full_trace` now first diverges at **step 2, off0 PC marker** — the SAME
`l16_ent_frame_sp_byte1_ff` rule misfiring on the step-2 PC byte rows
(`IS_BYTE=1, BYTE_INDEX_0=1, H1+2=0, OP_ENT≈4.9` broadcast). The ONLY clean
separator is making `H1+2` (the SP selector) a HARD requirement, but the
genuine SP byte rows carry `H1+2 ≈ 0.99` (not exactly 1.0), so the net-zero
weight+threshold promotion that works for hard-binary dims leaves a ~1-point
margin loss that disturbs the genuine 0xff firing under OP_ENT-broadcast
variance (verified: H1+2→100/threshold→113.5 regressed step-1 SAFE 0→2). A
robust fix needs either (a) a sharper integer H1+2 selector at the L16 input,
or (b) suppressing the OP_ENT in-step broadcast at its source so these ENT-frame
rules see a clean one-hot. Both are multi-rule / upstream changes.

Net effect this session: var exit-code pass unchanged (0/25 → 0/25, NO
regression vs baseline) but the documented root is corrected and the step-1
STACK0 leak is permanently closed (byte-identity-safe), advancing the
full_trace divergence one step (step 1 → step 2).

## Tools (read-only, this session)
- `tools/probe_var_l15_attn.py` — refutes the L15 head-0 K-alias hypothesis.
- `tools/probe_var_dimlabel.py` / `probe_var_l20_ffn.py` — localize the leak to
  the L20 (block 32) FFN `OUTPUT_HI` writers, with real widened-layout dim
  labels.
- `tools/probe_ent_f0_firing.py` — proves the f0 rule has no legitimate firing.
- `tools/probe_stack0_byte_marker.py` — the STACK0_BYTE* vs H1+2 discriminator.
- `tools/probe_var_step2.py` / `probe_var_raw_tokens.py` — per-step full_trace
  alignment + raw token stream.
