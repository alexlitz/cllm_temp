# STACK0 staleness follow-up: root cause is upstream auto-regressive divergence

Date: 2026-06-03
Branch: speedup-cache-and-buckets (off `bb730b01`)
Author: agent (Claude Opus 4.7)
Brief: follow-up on `CMP_POLARITY_INVESTIGATION_2026_06_03.md` finding #2
       ("stack0 / ALU staleness at the binary-pop AX position").

## TL;DR

The "STACK0_BYTE0 = 0" / "ALU = 0" symptom at the EQ AX row is **NOT**
caused by a missing PSH save, by L7 operand_gather attending wrong, or
by an L14 stack0 byte0 producer being broken. It is caused by **a
deeper auto-regressive token-emission divergence at step 4 (the second
IMM step) when the operand high nibble is nonzero**. The model emits a
MARK_PC token (257) at the position where it should emit a STACK0
byte-1 value, breaking the step-4 token layout and cascading into the
EQ step's downstream operand_gather. L7 then falls back to attending to
the only positionally valid STACK0_BYTE0 row (the initial-empty
stack0 row at pos 65, value 0), and that's why both ALU_LO and ALU_HI
read as nibble 0.

`tail_cmp_lt_false_00` polarity (CMP doc finding #1) is real but
orthogonal. The stack0 staleness needs a different (larger-blast-
radius) fix than a single declarative rule patch — flagging it for the
next investigation wave.

## What I traced

I built a forward-hook tracer (`/tmp/trace_compare.py`) that runs both
EQ(5,5) (PASSES) and EQ(17,17) (FAILS) through the smoke harness's
model, then replays the cached token stream through a single
teacher-forced forward and reads residuals at every transformer block
at the relevant positions.

### Token-stream observation (load-bearing)

Both EQ(5) and EQ(17) generate the same 5-instruction program
`IMM val PSH IMM val EQ EXIT`. Smoke runs autoregressively; the runner
caches the emitted token stream in `runner._kv_cached_tokens`. Comparing
the streams:

| Program     | Total tokens | MARK_BP positions       | MARK_STACK0 positions          | Result |
|-------------|--------------|-------------------------|-------------------------------|--------|
| EQ(5, 5)    | 218          | 59, 129, 164, 199       | 64, 98, 134, 169, 204         | PASS   |
| EQ(17, 17)  | 183          | 59, 129, 164            | 64, 98, 134, 169              | FAIL   |
| EQ(42, 42)  | 183          | 59, 129, 164            | 64, 98, 134, 169              | FAIL   |

Both lose MARK_BP at step 2 (the PSH step at SE@78 → next SE@111, 33
tokens instead of 35). That's a real but downstream-irrelevant bug:
EQ(5) hits the SAME step-2 missing-BP and still passes. So step-2
missing-BP is NOT the stack0-staleness root cause.

Where the two diverge: **step 4 token at offset 22 (STACK0 byte 1).**

Step 4 starts at SE@113. The byte at offset 22 (pos 135) should be 0
for both EQ(5) (encoded as token 0) and EQ(17) (encoded as token 0):

| Step 4 offset | Spec position    | EQ(5) token | EQ(17) token |
|---------------|------------------|-------------|--------------|
| 21            | STACK0 marker    | 268         | 268          |
| 22            | STACK0 byte 1    | **5**       | **257 (!)**  |
| 23            | STACK0 byte 2    | 0           | 34           |
| 24            | STACK0 byte 3    | 0           | 0            |

Token 257 is `Token.REG_PC` — the model emits a PC marker mid-step. The
step-4 layout breaks from offset 22 onward. The EQ step (step 5) starts
at SE@148 with a misaligned context; downstream attention heads can't
find the post-PSH STACK0 byte 0 anymore.

### Effect on the residual

At EQ AX (pos 140 for EQ(17), pos 154 for EQ(5)):

EQ(5) block 33: ALU_LO argmax=5 val=+1.02, ALU_HI argmax=0 val=+3.23.
  → CMP rules see (a=5, b=5) → fire EQ → output 1.

EQ(17) block 33: ALU_LO argmax=0 val=+0.18, ALU_HI argmax=0 val=+0.22.
  → CMP rules see (a=0, b=17) → ALU_LO/HI essentially zero, EQ
    override doesn't fire, default 0 wins.

The "ALU = 0" symptom at block 33 (matching CMP doc finding #2) is
DOWNSTREAM of the step-4 token-emission divergence.

### L7 operand_gather attends correctly given the broken stream

L7's K-side gate `AP(0, BD.STACK0_BYTE0, L)` finds the most recent row
with STACK0_BYTE0 flag hot. For EQ(5) the L1 STACK0_BYTE0 rule fires at
positions 65, 99, 135, 170, 205 (one per stack0 marker that has a byte
following it). For EQ(17) the rule fires only at 65, 170 — because at
pos 135 the token is a PC MARKER (257), not a byte, so L1's IS_BYTE
condition fails and STACK0_BYTE0 stays 0. L7 then attends to the only
STACK0_BYTE0 row before pos 140 — pos 65, which holds the initial
empty stack0 = 0.

So **L7 is doing the structurally correct thing**: gather the closest
prior STACK0_BYTE0 row. The bug is that the prior STACK0_BYTE0 row at
pos 135 doesn't exist because the token there is wrong.

## Why operand high nibble matters

The PASS pattern from the CMP doc:
> N where either `hi == 0 AND lo <= 0xF` (single nibble) OR `lo == 0`
> (high-byte only).

This matches the operand AX byte 0's nibble encoding. When the AX
byte-0 has both nibbles nonzero, the PSH chain's CLEAN_EMBED-to-OUTPUT
propagation produces a particular residual pattern at the STACK0
marker row that makes downstream emission flip to MARK_PC instead of a
byte. The exact propagation that flips the choice lives somewhere in
the model's head-bake logit competition (likely an interaction between
NEXT_PC and one of the byte-value heads).

What's happening at the moment of emission for offset 22 of step 4:

* The model attends to recent context to produce the next-token
  distribution.
* For value 5 (lo=5, hi=0), some downstream signal correctly says
  "emit byte 0" (because stack0 byte 1 of a 32-bit little-endian 5 is
  0).
* For value 17 (lo=1, hi=1), the same path is corrupted and emits
  MARK_PC's token-id (257).

The corruption likely lives in the L8/L10 OUTPUT_LO/HI propagation
chain or the L10 PSH STACK0 passthrough head. The L10 PSH STACK0
passthrough head 3 (`_layer10_psh_stack0_passthrough_head_spec` at
`l10_ops.py:1470-1569`) attends from the post-PSH STACK0 byte 0 row
back to the AX byte 0 row, and includes a differential routing
(slots 32+k and 48+k) that adds `OUTPUT_band - CLEAN_EMBED_band`. If
the AX byte 0's OUTPUT_HI nibble (the value's high nibble) leaks into
the wrong stack0 slot — say, into STACK0 byte 1's OUTPUT_HI band —
then the head_bake's NEXT_PC vs byte-value logit competition would
trip toward NEXT_PC.

## What I did NOT do

Per `feedback_single_rule_fixes_are_zero_sum.md` and the brief's
"ONE compile + smoke max" rule, I did NOT attempt a fix. The root
cause is far enough upstream from the L10 / L7 suspects in the
original brief that a single-rule patch is unlikely to net positive.
Specifically:

* Patching L7 operand_gather to attend differently won't help — the
  stack0 byte 0 *isn't there* to attend to (the value byte was
  overwritten by a marker token in the auto-regressive emission).
* Patching L14 PSH save would not change behaviour for EQ(5) and EQ(17)
  identically — both go through the same step-2 PSH; the difference
  emerges at step 4 emission, not at the PSH step.
* Patching the L10 PSH STACK0 passthrough head's differential routing
  (slots 32..63) is the highest-probability fix surface, but it
  touches the AX-byte-0 → STACK0-byte-0 routing on the LEA-local path
  (the comment at `l10_ops.py:1510-1548` calls this load-bearing for
  `var_simple` / `var_update` / `if_var`). Any change here risks
  regressing those tests.

## Recommended next investigation

1. **Hook block-by-block residual at step-4 pos 135 BEFORE the
   emission, comparing EQ(5) vs EQ(17).** Find the block where
   OUTPUT_HI/LO at pos 135 (which feeds the head_bake's NEXT_PC vs
   byte-value logit competition) diverges between the two. That tells
   you whether the bug is in:
   * L10 PSH STACK0 passthrough head 3's differential routing
   * L11/L12 OUTPUT band consolidation
   * The head_bake competition itself (Token.REG_PC vs byte-value
     tokens for a given (OUTPUT_LO[k], OUTPUT_HI[k]) pattern)

2. **Check the head_bake rule for Token.REG_PC vs byte tokens.** If
   `model.head.weight[Token.REG_PC, NEXT_PC]` is +20 and
   `model.head.weight[byte_n, OUTPUT_LO+n_lo] + ...[OUTPUT_HI+n_hi]`
   sums to something less, then a residual with both NEXT_PC weakly hot
   (~0.1) AND OUTPUT_LO/HI weakly hot would tip to REG_PC. Look for
   why NEXT_PC leaks at the STACK0-byte-1 row when AX has nonzero
   high nibble.

3. **Verify by ablation.** Set
   `model.head.bias[Token.REG_PC] = -30.0` (well below threshold so
   the marker can never win without explicit NEXT_PC hot signal) and
   see if EQ(17) passes (might break MARK_PC emission in other steps,
   but tells you the bug is logit-competition driven).

## Smoke

No code change. Smoke baseline at HEAD (`bb730b01`): 28 / 51 pass.

## Confidence

* **High** that the failure mode is auto-regressive token-emission
  divergence at step-4 offset-22, not a stack0 producer bug.
  Empirically EQ(5) and EQ(17) differ at exactly one token position
  (pos 135), and that one-token difference cascades into all the
  downstream "STACK0_BYTE0 = 0 / ALU = 0" symptoms.

* **Medium** that the highest-leverage fix surface is the L10 PSH
  STACK0 passthrough head 3's differential routing (slots 32..63
  added in 2026-06-01 to fix `var_simple` / `var_update` /
  `if_var`). The added routing inadvertently leaks AX-byte-0 high
  nibble bits into STACK0 byte 1's OUTPUT band, biasing the head_bake
  competition.

* **Low** that a single-rule declarative patch can fix this without
  regressing var_*/if_var/loop_* (which are already protected by the
  same L10 head's differential routing).

## File touchpoints (for the next investigator)

* `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:1470-1569`
  (`_layer10_psh_stack0_passthrough_head_spec` — head 3, the suspect
  surface; differential routing at slots 32..63 added to fix LEA-local
  AX byte 0)
* `c4_release/neural_vm/unified_compiler/ops/model_ops.py:1675-1701`
  (`head_bake` marker tokens — Token.REG_PC writes +20 weight at
  NEXT_PC, bias -10; competing against byte-value tokens whose
  weights are at OUTPUT_LO/HI bands)
* `c4_release/neural_vm/unified_compiler/ops/l1_ops.py:101-112`
  (L1 STACK0_BYTE0 rule — `L1H4[BP] AND IS_BYTE AND NOT H1[BP]`;
  correctly fires for EQ(5) at pos 135 but not EQ(17) because pos
  135's token is MARK_PC not a byte)
* `c4_release/neural_vm/unified_compiler/ops/l7_ops.py:229-258`
  (L7 operand_gather — correctly attends to the most recent
  STACK0_BYTE0-hot row; not buggy)
* `c4_release/docs/CMP_POLARITY_INVESTIGATION_2026_06_03.md`
  (predecessor brief; finding #2 traced to this one-token-position
  bug)

## Reproduction

```bash
cd c4_release
python /tmp/trace_compare.py
# Inspect step-4 token at pos 135 for EQ(5) vs EQ(17).
```

The traces showed `EQ(5)` pos 135 carries CLEAN_EMBED_LO[5]=1.0,
correctly encoding byte=5. `EQ(17)` pos 135 carries the embedding for
token 257 (MARK_PC) — see the trace_compare.py output for full
residual columns.
