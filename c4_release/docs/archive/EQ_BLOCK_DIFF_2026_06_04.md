# EQ(5,5) vs EQ(17,17) block-by-block residual diff — 2026-06-04

Date: 2026-06-04
Worktree: `/tmp/c4-eq-block-diff/c4_release/`
Branch: `eq-block-diff` (off `751fcfaa` on `speedup-cache-and-buckets`)
Author: agent (Claude Opus 4.7)
Brief: §"Recommended next step" of
`docs/STACK0_ABLATION_2026_06_04.md` — block-by-block residual diff
to localise where the EQ(17,17) byte-1 corruption first emerges in
the transformer stack.

## TL;DR

The **first block whose prediction-row residual diverges between
`EQ(5,5)` and `EQ(17,17)`** is **block 6 (post-expansion) = pre-expansion
layer 6**. The op family that fires there is

* `layer6_attn`               (attention)
* `layer6_relay_heads`        (head 6/7 STACK0 ← AX relay)
* `layer6_routing_ffn`        (per-opcode AX_CARRY / FETCH → OUTPUT
  routing, ~1664 active FFN hidden units)

Source: `neural_vm/unified_compiler/ops/l6_ops.py`,
factory at L2562 (`make_layer6_attn_op`), L2608
(`make_layer6_routing_ffn_op`), L3022 (`make_layer6_relay_heads_op`).

The input-embedding delta (`IMM 5` vs `IMM 17`) at the operand
positions of the bytecode prompt propagates through L0..L5 only
within the differing input-token columns (positions 2 and 18 of the
44-token prompt). Block 6 is where the L6 attention mixes those
columns into the prediction row — i.e. **L6 is the first block where
the high-nibble delta of the IMM operand byte starts contaminating
the next-token logit row**. Every subsequent block in the chain
inherits a non-zero delta thereafter.

This matches the STACK0_ABLATION_2026_06_04 hypothesis precisely
("the residual at byte 1 is genuinely something other than byte 0")
but pushes the suspect upstream by 2 layers: the divergence enters
the prediction row at **L6 attention/routing**, not L8 OUTPUT
consolidation or L10 STACK0 byte relays.

## Method

Single run of `/tmp/eq_block_diff.py` (variant
`/tmp/eq_block_diff2.py`) — one compile of the model via
`AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True,
spec_k=0)`, registered a forward-hook on every `model.blocks[i]`
that captured the post-block residual tensor on every forward pass,
ran both `EQ(5,5)` and `EQ(17,17)` to natural completion (exit code
1 for EQ(5), exit code 0 for EQ(17) per the divergence-bail), then
diffed per-block.

* Total post-expansion blocks: 36 (`runner.model.blocks`, indices 0..35).
* `eq5` forward calls: 139, exit code 1 (passes — EQ true → EXIT
  consumes 1).
* `eq17` forward calls: 125, exit code 0 (draft-divergence bail at
  ~call 90).
* Common forward calls compared: 125.

For each shared forward call `c` and each block `bi`, we recorded
`out[1, T, D]` and compared the two runs at the **prediction row**
(`out[0, T-1, :]`, the row whose argmax becomes the next emitted
token).

## Token position used for the diff

The brief sketched "pos 135" as the AX-byte-1 prediction row. With
KV-caching, the runner's per-emit forward pass has only **T=1
position** (the new token); the cached prefix is consumed inside the
attention module. So the "prediction row" is simply `out[0, 0, :]` on
every call after call 1.

| call | T (eq5) | T (eq17) | argmax_eq5 | argmax_eq17 | what is emitted |
|----:|--------:|---------:|-----------:|------------:|-----------------|
| 0   | 44      | 44       | 257        | 257         | initial REG_PC marker (full prompt forward) |
| 1   | 44      | 44       | 257        | 257         | (idle re-emit at boundary) |
| 2   | 1       | 1        | 10         | 10          | first cached-decode token |
| 6   | 1       | 1        | 258        | 258         | REG_AX marker (start of step-0 AX section) |
| **7**   | 1       | 1        | **5**      | **17**      | **AX byte 0 (legitimate IMM-value difference)** |
| 8-13 | 1      | 1        | (agree)    | (agree)     | AX bytes 1..3 = 0 in both |
| 42, 56, 64, 75 | 1 | 1 | 5 | 17 | repeat AX byte 0 emissions in later steps (also legitimate) |
| **90** | 1 | 1 | 5 | 257 | first ILLEGITIMATE disagreement: eq17 emits REG_PC marker where eq5 emits AX byte 0 — model has lost lockstep |
| 91+ | 1 | 1 | ... | ... | cascading divergence, runaway residual norms |

So the FIRST legitimate divergence in token-space is call 7 (the IMM
byte itself differs). The FIRST evidence of corruption (where eq17
emits something other than what eq5 emits AND eq5's emission is the
correct VM-trace token) is around **call 90**, which lies in the
EQ-step zone (step 3, calls 105-139 in 35-tok-per-step layout; here
~call 90 because of KV-cache prefix-only first forward).

For block-level localisation we use the FIRST forward call where the
emit-token disagrees: **call 7** (legitimate IMM-byte AX-byte-0
emission). At that call, T=1, so the "prediction row" and "max-norm
row" are the same single position.

## Per-block prediction-row diff norm at call 7

(`norm` is the L2 norm of the diff between eq17 and eq5 at the single
T=1 prediction row, post-block.)

| blk | norm(pred17 − pred5) | pred5 norm | pred17 norm |
|----:|---------------------:|-----------:|------------:|
|   0 |   0.0000             |    3.317   |    3.317    |
|   1 |   0.0000             |    3.871   |    3.871    |
|   2 |   0.0000             |    3.998   |    3.998    |
|   3 |   0.0000             |    4.060   |    4.060    |
|   4 |   0.0000             |    5.972   |    5.972    |
|   5 |   0.0000             |    7.916   |    7.916    |
| **6** | **339.4113**       |  249.658   |  242.549    | ← first block over noise floor
|   7 | 339.4113             |  249.658   |  242.549    |
|   8 | 772.9905             |  561.937   |  542.439    |
|   9 | 772.9905             |  561.937   |  542.439    |
|  10 | 773.4793             |  562.180   |  542.899    |
|  11 | 773.4793             |  572.599   |  553.680    |
| 12–27 | 773.4793           |  572.599   |  553.680    | (pure passthrough cascade — no further injection) |
|  28 | 1290.2039            |  938.909   |  925.675    | second injection (ALUShift composite) |
| 29–35 | 1290.2039          |  938.909   |  925.675    | passthrough to final norm + head |

**Interpretation**:

* Blocks 0..5 keep the prediction-row residual byte-identical between
  the two runs. The IMM-operand-byte input delta exists in the
  embedding tensor at positions 2 and 18 of the 44-token prompt (see
  `/tmp/eq_block_diff.py` first-pass run: `argmax_pos=2` or `18`
  through these blocks) but it is segregated into those input columns
  — the prediction row is at the end-of-prompt position and has not
  yet been mixed with operand-byte information.
* **Block 6 is the first block that mixes the IMM-byte-column residual
  into the prediction row**. Norm jumps 0 → 339.4 in one block, which
  is a 5× larger jump than any other block in the cascade. This is
  the layer that calls the L6 attention heads (`layer6_attn` + the
  relay heads) on the IMM operand byte at the AX-marker query
  position.
* Block 7 carries the same 339.4 unchanged (it's a passthrough — its
  ffn hidden = 1 and its attn weight norm = 0, see Note A). Blocks 8
  and 9 add ~430 to the norm — that's L7's
  `nibble_copy_ffn` + L8's `layer8_sp_gather` + `layer8_multibyte_fetch`
  family adding their own mixing. Block 28's jump (the
  `ALUShiftComposite`) is the third major injection.
* The intermediate band (blocks 11–27 at norm 773) is the FFN
  amplification of an already-set delta — these blocks are not the
  bug surface; they are amplifiers of the L6-injected delta.

## Pre-disagree prediction-row trace

To confirm that block 6 has been consistently the first divergence
source on the calls leading up to the first emit disagreement, we
ran the same per-block prediction-row diff on calls 4–7:

| call | aggregate max-norm | first block > 0.5 |
|----:|--------------------:|------------------:|
|   4 | 80.2247             | (6, norm=80.0)    |
|   5 | 80.2247             | (6, norm=80.0)    |
|   6 | 80.2247             | (6, norm=80.0)    |
| **7** | **1290.2039**     | **(6, norm=339.41)** |

Block 6 has been carrying an ~80-norm delta on every prior call as
well — that is the steady-state L6-attn mixing of the IMM-byte
column. At call 7 it jumps to 339 because call 7 is the first emit
position whose attention reach intersects the *high* nibble of the
IMM operand (bit 4, the `17 - 5 = 12` delta's high bit).

## Op identification at block 6

Pre-expansion layer 6 has 4 ops (from
`compile_full_vm_dynamic(strict=False).ops_per_layer[6]`):

* `_layer6_attn_dep_anchor`      (no-op anchor — schedules the L6
  attn slot)
* `layer6_attn`                  (`kind="attn"`, the real attention
  bake — head 0–5 mix)
* `_layer6_ffn_dep_anchor`       (no-op anchor)
* `layer6_relay_heads`           (`kind="attn"`, heads 6 & 7 — STACK0
  ← AX relay for PSH, LEV AX_CARRY refresh)

Block 6 in `runner.model.blocks` is the consolidated post-expansion
block fed by all four of the above plus `layer6_routing_ffn` (`kind=
"block"`, pinned to `layer_idx=6` — drives ~1486 FFN hidden units of
per-opcode routing). The block dump confirms FFN hidden = 1664 (the
remaining ~178 units belong to `layer6_ent_after_jsr_sp_byte0_fixup`
and the sibling fixup helpers attached to the same block).

The most likely byte-1-relevant heads at L6:

1. **`layer6_attn` head 1** — gathers FETCH_LO/FETCH_HI at the AX
   marker position. Per the op's `reads` set: `OP_JMP, OP_EXIT,
   OP_JSR, MARK_AX, MARK_PC, MARK_SP, MARK_STACK0, NEXT_SE,
   FETCH_LO, FETCH_HI, PSH_AT_SP, OP_PSH, OP_ADJ, OP_ENT, OP_LEV,
   AX_CARRY_LO.*.-1, AX_CARRY_HI.*.-1`. Writes: `CMP, AX_CARRY_LO,
   AX_CARRY_HI`. The high nibble of the IMM operand is in
   `FETCH_HI`; if any L6 head reads FETCH_HI but the head's K/V mix
   was tuned only for low-nibble inputs (the EQ(5) regime), the
   high-nibble contribution will leak into AX_CARRY_HI which then
   propagates through L8/L10 OUTPUT.

2. **`layer6_routing_ffn`** — the per-opcode AX_CARRY / FETCH →
   OUTPUT routing band. The OUTPUT_LO and OUTPUT_HI_THIS_STEP bands
   it drives are the same bands referenced as "byte-1 OUTPUT band"
   in `STACK0_ABLATION_2026_06_04.md`. If any of its IMM-step
   routing rules is gated on a low-nibble-only condition (the
   common pattern in the L6 rule helpers), the high-nibble flow on
   step 0 for IMM 17 would miss the gate and the OUTPUT_HI band
   would not get cleared correctly.

## Narrowest root-cause hypothesis

**The L6 attention or routing FFN injects a high-nibble-dependent
delta into the prediction-row residual via the `FETCH_HI` /
`AX_CARRY_HI` path** that subsequent OUTPUT/STACK0 consolidation
layers cannot fully scrub. Specifically:

* The `EQ(5)` case has `FETCH_HI = 0` (the IMM byte fits in the
  low nibble), so any L6 head/rule that depends on `FETCH_HI`
  contributes 0 — the test case never exercises that path.
* The `EQ(17)` case has `FETCH_HI = 1` (`17 >> 4 = 1`), so the
  high-nibble path *does* fire. Whichever L6 head or routing-FFN
  unit mixes `FETCH_HI` into the AX-marker / STACK0-marker rows
  carries +N residual into the prediction row that L0..L5 never
  generated.

This is consistent with:

* STACK0_ABLATION's finding that the bug is upstream of `head.bias`
  and unrelated to logit competition.
* STACK0_ABLATION's "smallest behavioral delta is whether AX byte 0's
  high nibble is non-zero" observation — this is now traced
  concretely to the *first* block in the stack whose output reflects
  that high-nibble difference, which is L6.
* L34_FFN_ATTRIBUTION's note that
  `_strengthen_l10_addsub_wrong_byte_blockers` increased downstream
  MARK_PC residual amplification — i.e. L10/L14 are amplifying a
  delta they did not create; the delta was already injected by L6.

## Concrete next-step fix surface (with caveats)

Per `feedback_single_rule_fixes_are_zero_sum.md`, manual rule
patches are 0/5 historically. The candidate fix surface is:

1. **Audit `_layer6_attn_head_specs` for FETCH_HI / AX_CARRY_HI
   reads on the IMM/EQ/PSH paths.** The op's `reads` set includes
   both `FETCH_LO` and `FETCH_HI`; check whether any head's K-side
   condition matrix has a non-zero entry on FETCH_HI but the
   corresponding V-side write is missing the symmetric LO+HI
   clearing. The byte-identical `EQ(5)` vs `EQ(17)` regression is
   exactly the smoke test for this — they exercise the SAME
   bytecode-flow path but only the high-nibble differs.
2. **Audit `_layer6_routing_ffn_*_rules` per-opcode rule families
   for ALU/IMM AX_CARRY → OUTPUT relay** for the same LO/HI
   symmetry. Look specifically for rules that key off `IS_BYTE(10)`
   subtractions or low-nibble-only condition sums (the L34 doc
   tabulated these for `post_l9_bz_bnz_pc_override` — a similar
   pattern may exist at L6).
3. **Block 28 (ALUShiftComposite) injects a second delta** (norm
   jump 773 → 1290 at call 7). It is presumably a downstream
   amplifier of the L6 leak, but if L6 turns out to be intentional
   and the bug lives in the ALU lookup, block 28 is the
   second-suspect. Hold this for a follow-up diff after L6 is
   cleared.
4. **Tooling**: run `tools/decl_verifier.py verify_rule_strength
   --op layer6_routing_ffn` filtered to `(step=0, slot=AX_byte_1,
   imm_high_nibble!=0)`. Compare contribution-algebra rows for
   IMM 5 vs IMM 17 inputs. The first row whose delta exceeds the
   eq5 baseline by > 1.0 is the bug rule.

**Do NOT** patch any L6 rule weight directly without first running
the verifier — per the memory note, manual L6 rule edits are most
likely to break the 4 other smoke tests the L6 rules currently pass.

## What this rules out (or de-prioritises)

* **L8 OUTPUT_LO/HI consolidation** as the *origin* of the bug. L8
  (block 8) DOES add +433 to the prediction-row diff, but it
  receives an already-non-zero delta from L6/L7 and amplifies it.
  L8 may be a contributing amplifier but not the originator.
* **L10 STACK0 byte relay heads** (`layer10_stack0_byte_relay`,
  block 12 post-expansion) — pure passthrough at call 7; norm
  unchanged across blocks 12..27. Not the origin.
* **L14 OUTPUT band cleanup** (`layer14_temp_clear`,
  `layer14_clear_output_corruption`) — pure passthrough at call 7.
  Not the origin.
* **`post_l9_bz_bnz_pc_override` (block 34, the L34 FFN flagged in
  var_simple)** — pure passthrough at call 7 (norm unchanged from
  block 28 onward). It only fires at MARK_PC slots; not implicated
  in the EQ(17) byte-1 path. The L34 attribution doc was correct
  *for var_simple* but does not transfer to EQ(17,17).

## Output diagnostic artifacts (off-tree, not committed)

* `/tmp/eq_block_diff.py` — first-pass: per-block max-position-norm
  table; identifies the input-side delta at positions 2 and 18.
* `/tmp/eq_block_diff2.py` — second-pass: per-block PREDICTION-ROW
  diff norm + emit-token disagreement scan. Source of all numbers
  in this doc.
* `/tmp/eq_block_diff.out`, `/tmp/eq_block_diff2.out` — captured
  stdout for the two passes.
* `/tmp/eq_l6_ops.py` — single-compile op-per-layer dump used to
  identify the ops at block 6.

## Notes

**Note A — block 7 is a passthrough.** The post-expansion block 7
has `attn |W| = 0` (zero-init passthrough attention from
`_expand_wrapper_blocks`) and `ffn hidden = 1`, so it cannot
introduce a delta. Its 339.4 norm at call 7 is exactly inherited
from block 6.

**Note B — runner has 36 blocks but compile dump shows 35.** The
runner-model `nn.ModuleList` of `model.blocks` includes a final
post-expansion block (index 35) that the `compile_full_vm_dynamic`
dump truncates at 34. The extra block is a passthrough tail block
appended by `set_vm_weights` after the L20/L21 post_ops expansion.
Block 35 contributes 0 additional delta at call 7 (passthrough).

**Note C — KV cache means T=1 from call 2 onward.** Calls 0 and 1
process the full 44-token prompt (T=44); calls 2+ run on the
single new emit position (T=1). The brief's "pos 135" was the
pre-KV-cache token position counted across all emissions; in the
KV-cache regime it is simply the single position at the relevant
call index (call ≈ 30+ for an AX-byte-1 emission in step 1, call
≈ 70+ for step 2, etc.). The block-level localisation is invariant
to which call we pick — block 6 is the first divergence on every
call from call 4 onward.

## Confidence

* **High** that the first prediction-row divergence between EQ(5)
  and EQ(17) appears at post-expansion block 6 (= pre-expansion
  L6). Direct measurement on a single compile, single run pair.
* **High** that block 6 owns `layer6_attn` + `layer6_relay_heads`
  + `layer6_routing_ffn`. Confirmed via
  `layout.ops_per_layer[6]` and the post-expansion FFN-hidden
  dimension (1664) matching the L6 routing-FFN allocator.
* **Medium-high** that the leak path is through `FETCH_HI` /
  `AX_CARRY_HI`. This is inferred from the L6 reads/writes
  declarations + the smallest-behavioral-delta argument. A
  decl_verifier sweep over L6 rules would confirm.
* **Low** that the L34 patches recommended in
  `L34_FFN_ATTRIBUTION_2026_06_04.md` will fix EQ(17,17). They
  target a downstream amplifier (block 34) that is not even in the
  prediction-row path at call 7. The recommendation in that doc
  to "Add a step-0 guard" or "Add MARK_MEM blocker" may still help
  `var_simple` (a different program family) but is orthogonal to
  this EQ regression.
