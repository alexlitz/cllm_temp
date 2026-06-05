# var_simple_0 / var_three_0 / if_var_0 block-by-block OUTPUT_LO[0] writer pin (2026-06-05)

Date: 2026-06-05
Worktree: `/tmp/c4-var-attribution-v3` (branch `var-attribution-v3`)
HEAD: `88c7b68f` (`docs(status): 1096 + smoke baseline 2026-06-05`)
Author: read-only attribution agent (forward-hook decomposition)

## TL;DR

The L34.ffn attribution (`post_l9_bz_bnz_pc_override` at block 35, width 192)
documented in
[`L34_FFN_ATTRIBUTION_2026_06_04.md`](L34_FFN_ATTRIBUTION_2026_06_04.md)
is **wrong**. A forward-hook split of every block's `attn` vs `ffn`
contribution at the failing prediction position shows that block 35
adds **+0.000** to OUTPUT_LO[0] for all three programs.

The real writers of the +2.55 OUTPUT_LO[0] mass at step-0:MEM_addr1
are:

| Block | Sublayer | Op | dLO0 (var_simple_0) | dLO0 (var_three_0) | dLO0 (if_var_0) |
|---:|---|---|---:|---:|---:|
|  3 | FFN | `layer3_ffn.mem_byte_0_default_{lo,hi}` (L3 MEM-byte-0 default) | **+0.940** | **+0.940** | **+0.940** |
| 14 | ATTN | `layer10_bp_byte_passthrough_bake` head 7 — `top_store_query` aux path | **+2.000** | +0.000 | **+2.000** |
| 26 | ATTN | (L17 attn / `layer14_mem_generation`) | -0.391 | -0.159 | -0.391 |
| 35 | FFN | `post_l9_bz_bnz_pc_override` (the prior L34 attribution) | +0.000 | +0.000 | +0.000 |

**The dominant +2.55 writer is block 14's attention head 7 (the
`layer10_bp_byte_passthrough_bake` op pinned at `target_op_name=
"layer10_carry_relay"` which resolves to runtime block 14 / attn
layer_idx=13).** The L3 FFN writes the smaller +0.94 baseline that is
shared by all three programs; var_three_0 lacks the +2.0 amplification
from block 14 (because its MEM_STORE/MEM_ADDR_SRC residual at the
failing slot is not large enough to fire head 7's `top_store_query`
augmentation), which is exactly why var_three_0 fails with a smaller
margin (-5.64) than var_simple_0 / if_var_0 (-26.49).

## Token positions used

| Program | Test idx | First divergence | logit_pos | Symbolic step:slot |
|---|---:|---|---:|---|
| var_simple_0 | 250 | abs=119, expected=0xff, neural=0x00 | 118 | step0:MEM_addr1 |
| var_three_0 | 300 | abs=247, expected=0xff, neural=0x00 | 246 | step0:MEM_addr1 |
| if_var_0 | 425 | abs=175, expected=0xff, neural=0x00 | 174 | step0:MEM_addr1 |

The prediction row at `logit_pos` is the residual at the MEM_addr0
input position (BYTE_INDEX_0 of the MEM region), used by the causal head
to predict the next token (MEM_addr1, expected 0xff).

## Block-by-block contribution table (var_simple_0)

(Identical structure for if_var_0; var_three_0 differs only at block 14.)

| blk | width | in_LO0 | attn_dLO0 | ffn_dLO0 | out_LO0 | label |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 7 | +0.000 | +0.000 | +0.000 | +0.000 | L0 |
| 1 | 5 | +0.000 | +0.000 | +0.000 | +0.000 | L1 |
| 2 | 10 | +0.000 | +0.000 | +0.000 | +0.000 | L2 |
| **3** | **134** | +0.000 | +0.000 | **+0.940** | +0.940 | **L3** (`layer3_ffn`) |
| 4..13 | various | +0.940 | +0.000 | +0.000 | +0.940 | L4..L12 passthrough |
| **14** | **1536** | +0.940 | **+2.000** | +0.000 | +2.940 | **L13** (`layer10_bp_byte_passthrough_bake`) |
| 15..25 | various | +2.940 | +0.000 | +0.000 | +2.940 | L14..L17 passthrough |
| **26** | **1874** | +2.940 | **-0.391** | +0.000 | +2.549 | **L17** attn (counter-write) |
| 27..34 | various | +2.549 | +0.000 | +0.000 | +2.549 | passthrough |
| 35 | 192 | +2.549 | +0.000 | +0.000 | +2.549 | L25 `post_l9_bz_bnz_pc_override` — **STATIC ZERO** |

Final OUTPUT_LO[0] = +2.549; OUTPUT_HI[0] also +2.549. Argmax then
prefers 0x00 over the expected 0xff.

(Block 35's +0.000 confirms that the prior L34_FFN_ATTRIBUTION_2026_06_04
report attributed to the wrong op. The HAS_SE step-0 guard at commit
`877335ae` was statically correct in disabling the op at step 0; the
divergence persists because the +2.55 mass was never coming from L34.ffn.)

## Block-by-block contribution table (var_three_0)

| blk | dLO0 attn | dLO0 ffn | running out | comment |
|---:|---:|---:|---:|---|
|  3 | 0 | **+0.940** | +0.940 | L3 MEM-byte-0 default fires identically |
| 14 | **+0.000** | 0 | +0.940 | **The +2.0 head-7 amplifier is dormant here** |
| 26 | -0.159 | 0 | +0.782 | L17 counter still fires, smaller magnitude |
| 35 | 0 | 0 | +0.782 | L34 still zero |

var_three_0 PROVES that block 14 head 7's amplification is the
distinguishing factor. The other failing programs both have a step-0
MEM-store residual that drives slot 40..47 of head 7's
`top_store_query` augmentation above its gate; var_three_0 does not.

## How block 14 head 7 produces +2.000 at OUTPUT_LO[0]

Block 14 is the **post-expansion runtime index of the L13 attention
block**. Its attention has 8 heads × 100 head-dim = 800. Of its 8 heads,
**head 7** is the sole writer to OUTPUT_LO[0] / OUTPUT_HI[0] at the
failing slot:

```
W_o[OUTPUT_LO+k, head7 + k]     = 2.0   for k in 0..15  (LO byte passthrough)
W_o[OUTPUT_HI+k, head7 + 16+k]  = 2.0   for k in 0..15  (HI byte passthrough)
W_v[head7 + k,    CLEAN_EMBED_LO + k]  = 1.0
W_v[head7 + 16+k, CLEAN_EMBED_HI + k]  = 1.0
```

The head was DESIGNED for **BP-byte passthrough** (head_dim 0..7
gating on `H1+BP_IDX=H1+3` to read the BP marker and copy its CLEAN_EMBED
forward as OUTPUT). That's the
`_layer10_bp_byte_passthrough_head_spec` helper in
`neural_vm/unified_compiler/ops/l10_ops.py:1398-1472`.

The same head also carries an **auxiliary `top_store_query` block**
(slots 40..47, l10_ops.py:1432-1471) added on top of the base BP-pass
spec. The aux block fires when:

```
W_q[head7 + 40, MEM_STORE]      = +5000   (M = 50 * S = 50 * 100)
W_q[head7 + 40, MEM_ADDR_SRC]   = +5000
W_q[head7 + 40, CMP+3]          = +500
W_q[head7 + 40, HAS_SE]         = +500
W_q[head7 + 40, MARK_STACK0]    = +5000   (TARGET_DIM=MARK_STACK0 for slot 40,41)
W_q[head7 + 40, ADDR_B0_LO+0]   = +1000   (TOP_STORE_ADDR)
W_q[head7 + 40, ADDR_B0_HI+14]  = +1000
W_q[head7 + 40, ADDR_B0_LO+8]   = -20000  (block lo+8 collision)
W_q[head7 + 40, ADDR_B0_HI+15]  = -20000  (block hi+15 collision)
W_q[head7 + 40, H1+0..3]        = -15000  (block marker positions)
W_q[head7 + 40, CONST]          = -9500
W_k[head7 + 40, H1+AX_IDX=H1+1] = +5000
```

At the failing prediction row (logit_pos=118 = MEM_addr0 of step 0 in
var_simple_0):

```
MEM_STORE        = +2.0   → +10000  (* W_q slot40 weight)
MEM_ADDR_SRC     = +1.0   → +5000
CMP+3            ≈ 0      → 0
HAS_SE           = 0      → 0       (HAS_SE is 0 on step 0)
MARK_STACK0      = 0      → 0       (this is MEM, not STACK0)
ADDR_B0_LO+0     ≈ 0      → 0
ADDR_B0_HI+14    ≈ 0      → 0
H1+0..3          = 0      → 0       (we're at H1+4 = MARK_MEM)
CONST            = +1.0   → -9500
SUM              ≈ +5500              <--- POSITIVE, slot 40 fires
```

The slot-40 K position must have `H1+1 = H1+AX_IDX = +5000`. Position 99
(AX_byte1) is exactly that — at AX_byte1 the residual has `H1+1 = 1.0`
(MARK_AX is at H1+1). Softmax of the head's per-slot attention scores
gives attn_w[key=99] = 1.000, so V at key=99 is copied unconditionally
into the head output. V[99] is the CLEAN_EMBED of AX_byte1 = 0x00:

```
CLEAN_EMBED_LO+0 = 1.0   (nibble 0 of byte 0x00)
CLEAN_EMBED_HI+0 = 1.0   (nibble 0 of byte 0x00)
```

W_o then maps head_dim 0 → OUTPUT_LO+0 with scale 2.0, and head_dim 16 →
OUTPUT_HI+0 with scale 2.0. Result: **+2.0 added to OUTPUT_LO[0] and
+2.0 added to OUTPUT_HI[0]** at logit_pos=118.

### Why var_three_0 escapes the +2.0

var_three_0 has the same MEM_STORE flag pattern at its failing
logit_pos=246, but the residual at that position differs in
**CMP / ADDR_B0 / HAS_SE** lanes such that the slot-40 gate sum stays
below threshold. Empirically the head 7 attn-delta at LO0 is 0.000 for
var_three_0 (vs +2.000 for var_simple_0 and if_var_0). The slot
attribution at this granularity would need further per-slot scoring
to localise which lane saves var_three_0; this report does not
enumerate it.

## How block 3 (L3.ffn) produces +0.940 baseline

Block 3 (L3.ffn, 134 units) fires unit 34 at the failing logit_pos:

```
W_up[unit34, BYTE_INDEX_0] = +100
W_up[unit34, H1+4 (MARK_MEM)] = +100
W_gate[unit34, ...] (matching)
W_down[OUTPUT_LO+0, unit34] = +0.02
hidden[unit34] = silu(200 + b) * (gate ≈ 47) ≈ 47.0
contrib_to_LO0 = +0.94
```

This matches the declared L3 FFN rule
`layer3_ffn.mem_byte_0_default_{lo,hi}` (l3_ops.py:372-387):

```python
("MARK_MEM", 1.0), ("BYTE_INDEX_0", 1.0), threshold=1.5,
writes=(("OUTPUT_LO+0", 2.0/S),)
```

Semantics: "at MEM-marker byte-0 positions, predict next MEM addr byte
as 0x00". The +0.94 is the SwiGLU realisation of the 2.0/S target write
(the same +0.94 magnitude appears at MEM_byte_1/2/3 default rules too,
all by design). The L3 default is the BASELINE that the var_simple
failure cluster lands on; var_simple amplifies it via block 14, while
var_three keeps it at +0.94 (then gets a -0.16 counter from L17 →
final +0.78 OUTPUT_LO[0], small margin -5.64).

The L3 rule predicts "0x00 next byte" because MEM addrs default to
high-byte=0x00. For stack-relative addresses with negative offsets
(SP - 8 → 0xff at byte 1), this default is **wrong** but expected to be
overridden by a downstream writer that consumes
`MEM_ADDR_SRC=STACK0` + `byte_index=1` context. The L17 -0.391
counter is precisely the partial-override path; its magnitude is
insufficient.

## How block 26 (L17 attn) produces -0.391 counter

Block 26 is L17.attn (1874-unit FFN); its attention writes **-0.391** to
both OUTPUT_LO[0] and OUTPUT_HI[0] at the failing slot. This is a
**partial address-sign-extension correction** — it knows that the
expected MEM addr byte should be 0xff (sign-extended SP), so it
SUBTRACTS from OUTPUT_LO[0] to reduce the wrong "predict 0x00" mass.

But -0.391 alone cannot beat +0.940 + +2.000 = +2.940 → final +2.549
mass at OUTPUT_LO[0]. The argmax for the prediction is still 0x00 and
the expected 0xff loses.

## Does the same op dominate var_three_0 + if_var_0?

| Program | Dominant attn writer | dLO0 attn | Dominant ffn writer | dLO0 ffn |
|---|---|---:|---|---:|
| var_simple_0 | block 14 head 7 (`layer10_bp_byte_passthrough_bake`) | **+2.000** | block 3 (`layer3_ffn.mem_byte_0_default`) | **+0.940** |
| var_three_0 | (none; block 14 head 7 dormant) | +0.000 | block 3 (same rule) | **+0.940** |
| if_var_0 | block 14 head 7 (same as var_simple_0) | **+2.000** | block 3 (same rule) | **+0.940** |

So:

* **L3 `mem_byte_0_default` baseline is universal** to all three programs
  (and to all `step0:MEM_addr1=0xff` failures with the +0.94 floor).
* **Block 14 head 7 `top_store_query` aux amplification** is the
  distinguishing high-margin contributor for var_simple_0 and if_var_0.
  var_three_0 fails with the smaller baseline alone.

The +0.78 final OUT_LO[0] for var_three_0 is consistent with
+0.940 (L3) - 0.159 (L17 counter) ≈ 0.781.

The +2.549 final OUT_LO[0] for var_simple_0 / if_var_0 is consistent with
+0.940 (L3) + 2.000 (block 14 head 7) - 0.391 (L17 counter) ≈ 2.549.

## Concrete fix hypothesis

### Primary surface — block 14 head 7 `top_store_query` aux block

The `top_store_query` slots (40..47) of
`_layer10_bp_byte_passthrough_head_spec` fire INAPPROPRIATELY at MEM
address-byte positions in step 0. The intended firing condition was a
PSH/SI/SC store at a STACK0 marker; the actual gate is permissive
enough that MEM_STORE+MEM_ADDR_SRC alone (without MARK_STACK0) reach
threshold.

Three patch options ranked by zero-sum risk:

1. **Add `MARK_MEM = -M` to every slot 40..47 query.** This kills the
   `top_store_query` path on MEM-address rows definitively. The aux
   block was meant for STACK0-marker positions, not MEM positions; a
   MARK_MEM blocker is semantically correct. Risk: **low** — none of
   the legitimate STACK0-store rows have MARK_MEM=1 (mutual exclusion
   via marker registry).

2. **Add `H1+4 = -M` to every slot 40..47 query.** Functionally
   identical to option 1 (MARK_MEM is the H1+4 dim alias). Same risk
   profile.

3. **Raise the slot 40..47 threshold via reducing TOP_STORE_BIAS more
   negative (e.g. -98*S instead of -95*S).** Suppresses the leak by
   margin instead of by predicate. Requires re-baking and may not
   generalise to other residual leakage paths. Risk: **medium**.

### Secondary surface — block 3 L3 `mem_byte_0_default` baseline

Even with block 14 head 7 silenced, the +0.94 from L3 is what var_three_0
fails on. The L3 rule predicts "next MEM addr byte = 0x00", which is
right for positive global addresses but wrong for stack-relative
addresses. Adding a step-0 sign-extension override at L3 (or upstream of
the L17 counter) would close var_three_0 in addition to var_simple/if_var.

Possible patch: gate the L3 `mem_byte_0_default` rule on
`MEM_ADDR_SRC = 0` (i.e. only fire for SP-source addresses, not
STACK0-source). var_simple_0 uses an `int x = 990` write which is a
**stack-store** (MEM_ADDR_SRC=1 = STACK0); the L3 default for that case
is the wrong prediction. Risk: **medium** — would need verifier
sweep to confirm no regression on global-address loads.

### Pure-fix-of-symptom — L17 counter

Strengthening block 26's L17-attn counter from -0.391/-0.159 to
e.g. -2.5 would mask the failure without touching the root cause.
Risk: **high** (per `feedback_single_rule_fixes_are_zero_sum`).
Not recommended.

## What this rules out

* **`post_l9_bz_bnz_pc_override` at block 35 / L34.ffn (the
  `L34_FFN_ATTRIBUTION_2026_06_04.md` claim).** Block 35 contributes
  **0.000** to OUTPUT_LO[0] at the failing prediction row of all three
  programs. The HAS_SE step-0 guard at commit `877365ae` worked as
  intended; the residual diagnosis "block=35 layer=25 width=192" was
  the dynamic verifier's **first-loss-after-symbolic-support** kind of
  artefact (the symbolic support is already lost upstream, and the
  verifier reports the last block in the cascade).

* **`tail_bit32_result_correction` (L33.ffn / block 34 / 2059 units).**
  Also contributes 0.000 at the failing row. Already consistent with
  the 2026-06-05 status doc finding that this op is no longer firing
  on var_simple.

* **Logit-competition / head-bias fixes.** The +2.94 OUT_LO[0] mass is
  pure residual contribution, accumulated by block 14. No head.bias
  tweak can defeat the residual without breaking the legitimate
  0x00-prediction paths that depend on L3 / L17.

## Methodology

Single forward-hook pass on the post-expansion runtime model:

* Pre-hook on each `block.attn` captures the input residual.
* Post-hook on each `block.attn` captures the output (which is
  `attn(x) = x + delta_attn` per `AutoregressiveAttention.forward`
  vm_step.py:597, 605).
* `attn_delta = attn_out - attn_in` at the prediction row.
* Same scheme for `block.ffn` (PureFFN.forward returns `x + delta_ffn`
  per `base_layers.py:118`).
* Block-level pre/post hooks for sanity-check on attn+ffn == block_out
  conservation.

Per-head attribution at block 14 used a manual replay of the attention
forward (Q/K/V projections, scaled dot-product, softmax, V-matmul, W_o)
on the captured `x_in` to isolate the +2.000 to head 7 alone (other 7
heads contribute 0.000 at LO0).

Per-unit attribution at block 3 used the same SwiGLU forward replay
to isolate the +0.940 to unit 34 (the L3 `mem_byte_0_default` rule).

W_o / W_v / W_q nonzero-pattern inspection of head 7 confirmed
operationally that the head is the `layer10_bp_byte_passthrough_bake`
spec from `l10_ops.py:1398-1472`.

`find_producers([all_ops], "OUTPUT_LO")` listed 39 ops that write
OUTPUT_LO. The forward-hook measurement narrowed the actual writers
at the failing slot to the four entries in the TL;DR table.

Single compile, three teacher-forced runs (var_simple_0, var_three_0,
if_var_0). One additional compile for the per-head/per-unit decomposition.

## Smoke regression (read-only this report)

No production code changed. Smoke baseline at HEAD `88c7b68f` is
**46/52 pass + 11 xpass** per
[`STATUS_1096_2026_06_05.md`](STATUS_1096_2026_06_05.md). Identical
baseline expected after this commit (docs-only).

## Artifacts (off-tree)

* `/tmp/var_real_attribution_split.py` — per-block attn/ffn split hook
* `/tmp/inspect_block14_attn.py` — per-head decomposition at block 14
* `/tmp/inspect_block14_head7.py` — W_q/W_k/W_v/W_o head-7 dump
* `/tmp/inspect_block3_ffn.py` — per-unit decomposition at block 3
* `/tmp/check_residual_118.py` — residual-dim dump at logit_pos / key=99
* `/tmp/var_real_attribution_split.log` — split table for all three programs

## Suggested next step

Add a `MARK_MEM = -50*S` (or equivalent magnitude) blocker to each of
the 8 `top_store_query` slot definitions (40..47) inside
`_layer10_bp_byte_passthrough_head_spec` at l10_ops.py:1432-1471. This
silences block 14 head 7's wrong fire path on MEM-address rows
without disturbing the legitimate STACK0-store path. Smoke + 1096
regression required before commit.

If the L3 baseline +0.94 then becomes the sole binding constraint
(var_three_0 still failing), a gated step-0 `MEM_ADDR_SRC=0` predicate
on the L3 `mem_byte_0_default` rule is the natural follow-up.
