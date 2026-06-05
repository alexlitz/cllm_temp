# block14 (L10 efficient_andorxor_wrap) attribution — 2026-06-04

Date: 2026-06-04
Branch: `block14-fix` (off `speedup-cache-and-buckets`, HEAD `e3acaad2`)
Worktree: `/tmp/c4-block14-fix`
Author: read-only attribution agent (no fix applied)

## TL;DR

The block14 carrier flagged in `MULTICLUSTER_ATTRIBUTION_2026_06_04.md`
(Note B) as "L18/L22 expansion FFN, 1536u, owner unresolved" resolves
at HEAD `e3acaad2` to **`efficient_l10_andorxor_wrap`**
(`neural_vm/unified_compiler/ops/alu_ops.py:609`). The op installs a
1536-unit `PureFFN` baked from `wide_alu_dsl.bitwise_rules` for
AND/OR/XOR (3 × 512 rules).

**No fix was applied.** Static analysis shows the FFN is correctly
gated and cannot be writing the `step1:SP_byte3=REG_BP` signature seen
in `add_1`. The signature must be carried by one of the eight
attention head bakes installed on the SAME L10 block (`block14.attn`,
`attn.layer_idx=13`). The diag's "first-loss-after-symbolic-support"
points at the block, not necessarily at its FFN.

## Block14 identification

At HEAD `e3acaad2`, using the same compile config the 1096 diagnostic
uses (`BatchedPureNeuralRunner(max_seq_len=4096)` →
`compile_full_vm_dynamic(enable_conversational_io=True, alu_mode='efficient')`):

| Block index | attn.layer_idx | FFN type | W_up shape | Owner |
|---|---|---|---|---|
| block14 | 13 | PureFFN | (1536, 800) | `efficient_l10_andorxor_wrap` |

The 1536 units split exactly 512/512/512 for AND/OR/XOR. Verified:
- Rows 0–511: `W_gate[:, OP_AND] = 1.0`, others 0
- Rows 512–1023: `W_gate[:, OP_OR] = 1.0`, others 0
- Rows 1024–1535: `W_gate[:, OP_XOR] = 1.0`, others 0
- `b_gate` is uniformly 0 (no leak via `gate_bias`)
- `W_down` writes only to OUTPUT_LO (dim 69) and OUTPUT_HI (dim 85) bands

`OP_ADD` (dim 209) has **zero entries in W_gate** for all 1536 units.
On any ADD instruction the SwiGLU `silu(gate_x) * up_x` term collapses
to `silu(0) * up_x = 0` per unit — the FFN contributes nothing to the
residual on `add_*` programs.

## Why the FFN cannot carry the `step1:SP_byte3=REG_BP` signature

The diag (`add_1: step1:SP_byte3 expected=0x00 neural=REG_BP`):

```
OUT_LO[0]=-497.28 arg=0/-497.28
OUT_HI[0]=-497.28 arg=0/-497.28
band_contracts=OK
expected_logit=-4978.32  argmax_logit=-9.63  margin=-4968.69
```

- The OUTPUT_LO[0] and OUTPUT_HI[0] bands are already at -497 at the
  block14 snapshot. The byte-`0x00` token loses to REG_BP (marker
  token, embedding on `MARK_BP` + `IS_MARK`).
- The FFN writes ONLY to OUTPUT_LO/OUTPUT_HI bands. For an ADD opcode
  (OP_AND/OR/XOR all 0), every unit's SwiGLU output is 0.
- Therefore block14's FFN cannot add to the -497 OUTPUT projection,
  nor can it boost MARK_BP. The argmax flip from byte_0x00 (block13)
  to REG_BP (block14) is **not driven by the FFN**.

## What block14 actually does (attention)

`model.blocks[14].attn` is the L10 attention with 8 heads. It is the
target of multiple block ops at `resolved_layer=13` per
`layout.block_ops`:

| Head | Bake op | Role |
|---|---|---|
| 0 | `layer10_carry_relay_bake` | CARRY for ADD/SUB byte cascade |
| 1 | `layer10_byte_passthrough_bake` | AX byte passthrough |
| 2 | `layer10_sp_byte_passthrough_bake` | **SP byte passthrough** |
| 3,4 | shared by SP/BP passthrough chain | per `_byte_passthrough_chain_spec` |
| 5 | `layer10_stack0_byte_relay_bake.head_5` | non-bitwise STACK0 relay |
| 6 | `layer10_stack0_persistence_head_spec` | STACK0 carry-forward |
| 7 | `layer10_bp_byte_passthrough_bake` | **BP byte passthrough** |

The SP and BP passthrough heads write to `OUTPUT_LO`/`OUTPUT_HI`
(byte-3 lane). The marker-carry-forward sub-spec on
`_layer10_sp_byte_passthrough_head_spec` (l10_ops.py:1357–1395) has
intricate Q/K patterns to suppress the carry-forward on PSH/JSR/ENT/
LEV step boundaries. These are the structural candidates for the
SP_byte3 leak.

The most likely fix surface is therefore **one of the L10 attention
head specs**, not the andorxor FFN.

## Why no fix was applied

Per memory note `feedback_single_rule_fixes_are_zero_sum.md`, the
0/5 historical hit rate for ad-hoc single-rule fixes argues against
applying a patch without verifier triangulation. The brief's
"look for analogous tail_* rules with missing step-1 guards" search
strategy does not apply at block14:

- `block14.ffn` has **zero** `tail_*` rules (it's all `bitwise_<op>`
  cell-cross-product rules).
- `tail_bit32_result_correction` is at L33 (block33, 2059 units), not
  block14.
- The L10 attention heads are not authored via `tail_*` FFN rules
  — they are `DeclarativeAttentionHeadSpec` Q/K/V/O patterns with
  bespoke marker-suppression sub-specs.

A correct fix would need to:
1. Snapshot the L10 attention output per head for an `add_1` program.
2. Identify which head's value matrix is leaking into SP_byte3 slot 14.
3. Adjust the relevant marker-suppression term (likely an Ah Q
   coefficient on `BD.CMP+3` / `BD.OP_LEV` / `BD.OP_ENT` step-1 path).

This requires more than one compile and is therefore outside the
"ONE compile + ONE smoke + ONE 1096 sample" budget. The L10 attention
verifier work belongs to a follow-up brief with the tools listed
below.

## Baseline state (HEAD `e3acaad2`)

- Smoke: 28 passed / 23 failed (test_smoke.py, 81s)
- 1096 add_0..9 diag chunk: 0 OK / 10 div (same as the post-tail_bit32
  doc) with the `add_1` signature `block=14 layer=13 width=1536
  step1:SP_byte3=REG_BP` reproduced exactly.

## Recommended follow-up tools

- `scripts/dump_block_widths_v2.py` already prints `attn.layer_idx`
  per block — but does NOT route through `BatchedPureNeuralRunner`,
  so its topology disagrees with the diagnostic's. **The runner
  topology (conversational_io=True, alu_mode='efficient') is the
  one to trust.** A future block-attribution script should default
  to the runner path to avoid this confusion.
- A focused per-head residual delta probe at block14 would settle the
  attribution in one extra compile. Pattern:
  ```python
  for head_idx in range(attn.num_heads):
      delta = attn(x)[..., logit_pos, head_slice(head_idx)]
      # decompose into MARK_BP / OUTPUT_LO contribution
  ```
- `tools/attribute_1096_failure.py` walks ops in topological order
  and pins "first op with mismatched produces" — it should be
  rerun on `add_1` after the L10 head writes get into its claim
  table.

## Cross-references

- `docs/MULTICLUSTER_ATTRIBUTION_2026_06_04.md` Note B — the original
  "L18/L22 expansion FFN 1536u" guess. **Refuted by this report**:
  the 1536u block at the diag's `block=14 layer=13` slot is L10's
  AND/OR/XOR composite, owned by `efficient_l10_andorxor_wrap`
  (`alu_ops.py:609`), NOT L18 or L22.
- `docs/L34_FFN_ATTRIBUTION_2026_06_04.md` — the `tail_*` rule
  inventory analogy the brief invoked. None of those rules live on
  block14.
- `feedback_single_rule_fixes_are_zero_sum` memory note — rationale
  for not applying a speculative fix here.
